"""Discover and parse project instruction files."""

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

import yaml

from .. import constants
from ..utils import find_project_root, is_path_ignored
from .models import (
    AgentInstructionsSource,
    InstructionBudgetExceededError,
    LoadedAgentInstructions,
)


def get_skill_directories(
    working_directory: Path,
    relative_directory: Path = constants.DEFAULT_SKILLS_DIRECTORY,
) -> list[Path]:
    """Return skill directories in descending precedence order.

    Args:
        working_directory (Path): Directory whose repository-scoped instructions should apply.
        relative_directory (Path): Instruction directory relative to each project and user scope.

    Returns:
        list[Path]: Project directories from ``working_directory`` through the repository root,
            followed by the directory in the user's home. Outside a repository, only the
            working-directory and user scopes are returned.
    """
    project_root = find_project_root(working_directory)
    if project_root is None:
        directories = [working_directory / relative_directory]
    else:
        scoped = []
        directory = working_directory
        while True:
            scoped.append(directory / relative_directory)
            if directory == project_root:
                break
            directory = directory.parent
        directories = scoped
    directories.append(Path.home() / relative_directory)
    return directories


def read_instruction_frontmatter(
    location: Path,
    required_fields: Collection[str] = (),
) -> dict[str, Any]:
    """Read and validate the YAML frontmatter of an instruction file.

    Args:
        location (Path): Instruction file to read as UTF-8.
        required_fields (Collection[str]): Metadata fields that must contain non-empty strings.
            Valid values are stripped before being returned.

    Returns:
        dict[str, Any]: The decoded frontmatter mapping.

    Raises:
        OSError: The instruction file cannot be opened or read.
        UnicodeError: The instruction file is not valid UTF-8.
        ValueError: Frontmatter is missing, unterminated, or not a mapping.
        yaml.YAMLError: Frontmatter contains invalid YAML.
    """
    frontmatter = []
    with location.open(encoding="utf-8") as instruction_file:
        if instruction_file.readline().strip() != "---":
            raise ValueError(f"{location.name} must start with YAML frontmatter")
        for line in instruction_file:
            if line.strip() == "---":
                break
            frontmatter.append(line)
        else:
            raise ValueError(f"{location.name} frontmatter is not terminated")

    metadata = yaml.safe_load("".join(frontmatter))
    if not isinstance(metadata, dict):
        raise ValueError(  # noqa: TRY004 - malformed frontmatter has one validation contract.
            f"{location.name} frontmatter must be a mapping"
        )
    for field in required_fields:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location.name} requires a non-empty {field}")
        metadata[field] = value.strip()
    return metadata


def read_instruction_body(
    content: str,
    filename: str,
    *,
    require_frontmatter: bool = False,
) -> str:
    """Return an instruction body, excluding optional YAML frontmatter.

    Args:
        content (str): Complete instruction-file content.
        filename (str): Display name used in validation errors.
        require_frontmatter (bool): Whether a leading YAML frontmatter block is required.

    Returns:
        str: The trimmed Markdown body, without an optional frontmatter block.

    Raises:
        ValueError: Required frontmatter is missing, or present frontmatter is unterminated or
            is not a mapping.
        yaml.YAMLError: Present frontmatter contains invalid YAML.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        if require_frontmatter:
            raise ValueError(f"{filename} must start with YAML frontmatter")
        return content.strip()
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            metadata = yaml.safe_load("\n".join(lines[1:index]))
            if not isinstance(metadata, dict):
                raise ValueError(f"{filename} frontmatter must be a mapping")
            return "\n".join(lines[index + 1 :]).strip()
    raise ValueError(f"{filename} frontmatter is not terminated")


def get_agents_files(
    working_directory: Path | str,
    agents_filenames: Iterable[str] = (constants.DEFAULT_AGENTS_FILENAME,),
) -> list[Path]:
    """Return applicable recursively indexed agent instruction files in root-to-leaf order.

    Args:
        working_directory (Path | str): Directory whose instruction scope should be discovered.
        agents_filenames (Iterable[str]): Candidate names in precedence order.

    Returns:
        list[Path]: Canonical paths for existing instruction files in scope, choosing the first
            available candidate filename in each directory.
    """
    working_directory = Path(working_directory)
    project_root = find_project_root(working_directory)
    directories = [working_directory]
    if project_root is not None:
        directories = [
            directory
            for directory in reversed((working_directory, *working_directory.parents))
            if directory == project_root or project_root in directory.parents
        ]
    names = tuple(dict.fromkeys(agents_filenames))
    if project_root is None:
        available = {
            candidate.resolve()
            for filename in names
            if (candidate := working_directory / filename).is_file()
        }
    else:
        available = {
            candidate.resolve()
            for filename in names
            for candidate in project_root.rglob(filename)
            if candidate.is_file()
        }

    discovered = []
    for directory in directories:
        for filename in names:
            candidate = (directory / filename).resolve()
            if candidate in available and not is_path_ignored(
                candidate, root=project_root or directory
            ):
                discovered.append(candidate)
                break
    return discovered


def load_agents_instructions(
    working_directory: Path | str,
    agents_filenames: Iterable[str] = (constants.DEFAULT_AGENTS_FILENAME,),
    max_bytes: int = constants.MAX_AGENTS_BYTES,
) -> LoadedAgentInstructions:
    """Load complete applicable project instructions or reject instruction overflow.

    Args:
        working_directory (Path | str): Directory whose instruction scope should be loaded.
        agents_filenames (Iterable[str]): Candidate names in precedence order.
        max_bytes (int): Maximum UTF-8 bytes for the complete applicable instruction chain.

    Returns:
        LoadedAgentInstructions: Bounded content and per-source provenance.

    Raises:
        OSError: An applicable instruction file cannot be read.
        UnicodeError: An applicable instruction file is not valid UTF-8.
        InstructionBudgetExceededError: Applicable instructions exceed the byte limit.
    """
    discovered = []
    diagnostics = []
    for agents_file in get_agents_files(working_directory, agents_filenames):
        try:
            content = read_instruction_body(
                agents_file.read_text(encoding="utf-8"),
                agents_file.name,
            )
        except UnicodeError:
            raise
        except (ValueError, yaml.YAMLError) as exc:
            diagnostics.append(f"Skipped '{agents_file}': {exc}")
            continue
        if content:
            discovered.append((agents_file, content))

    if not discovered:
        return LoadedAgentInstructions(
            content=None,
            sources=(),
            max_bytes=max_bytes,
            diagnostics=tuple(diagnostics),
        )

    content = "\n\n".join(content for _, content in discovered)
    if len(content.encode("utf-8")) > max_bytes:
        sources = ", ".join(
            f"'{path}' ({len(body.encode('utf-8'))} bytes)" for path, body in discovered
        )
        raise InstructionBudgetExceededError(
            f"Applicable AGENTS instructions exceed the {max_bytes}-byte limit. "
            f"Sources: {sources}. Shorten the instruction sources or raise the configured limit "
            "for the selected model; no instructions were truncated."
        )
    return LoadedAgentInstructions(
        content=content,
        sources=tuple(
            AgentInstructionsSource(
                path=path,
                content=body,
                size_bytes=len(body.encode("utf-8")),
                included_bytes=len(body.encode("utf-8")),
                truncated=False,
            )
            for path, body in discovered
        ),
        max_bytes=max_bytes,
        diagnostics=tuple(diagnostics),
    )


def build_instructions(*sections: str | None) -> str | None:
    """Combine non-empty instruction sections without changing their content.

    Args:
        *sections (str | None): Optional instruction sections in desired output order.

    Returns:
        str | None: Sections separated by blank lines, or ``None`` when every section is empty.
    """
    sections = [str(section).strip() for section in sections if section]
    return "\n\n".join(section for section in sections if section) or None
