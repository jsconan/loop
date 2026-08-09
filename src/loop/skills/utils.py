"""Discover and parse project instruction files."""

from collections.abc import Collection
from pathlib import Path
from typing import Any

import yaml

from .models import AgentInstructionsSource, LoadedAgentInstructions
from ..utils import find_project_root

MAX_AGENTS_BYTES = 32 * 1024
DEFAULT_AGENTS_FILENAME = "AGENTS.md"
DEFAULT_SKILL_FILENAME = "SKILL.md"
DEFAULT_SKILLS_DIRECTORY = Path(".agents/skills")
TRUNCATION_MARKER = "\n\n[AGENTS.md truncated: instruction byte limit reached.]"


def get_skill_directories(
    working_directory: Path,
    relative_directory: Path = DEFAULT_SKILLS_DIRECTORY,
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
        raise ValueError(f"{location.name} frontmatter must be a mapping")
    for field in required_fields:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location.name} requires a non-empty {field}")
        metadata[field] = value.strip()
    return metadata


def read_instruction_body(content: str, filename: str) -> str:
    """Return the Markdown body following YAML frontmatter.

    Args:
        content (str): Complete instruction-file content.
        filename (str): Display name used in validation errors.

    Returns:
        str: The trimmed Markdown following the closing frontmatter delimiter.

    Raises:
        ValueError: Frontmatter is missing or unterminated.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{filename} must start with YAML frontmatter")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :]).strip()
    raise ValueError(f"{filename} frontmatter is not terminated")


def get_agents_files(
    working_directory: Path | str,
    agents_filename: str = DEFAULT_AGENTS_FILENAME,
) -> list[Path]:
    """Return discovered agent instruction files in root-to-leaf order.

    Args:
        working_directory (Path | str): Directory whose instruction scope should be discovered.
        agents_filename (str): Name of the instruction file to discover.

    Returns:
        list[Path]: Canonical paths for existing instruction files in scope.
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
    return [
        agents_file.resolve()
        for directory in directories
        if (agents_file := directory / agents_filename).is_file()
    ]


def load_agents_instructions(
    working_directory: Path | str,
    agents_filename: str = DEFAULT_AGENTS_FILENAME,
    max_bytes: int = MAX_AGENTS_BYTES,
) -> LoadedAgentInstructions:
    """Load project instructions with source and truncation diagnostics.

    Args:
        working_directory (Path | str): Directory whose instruction scope should be loaded.
        agents_filename (str): Name of the instruction file to discover.
        max_bytes (int): Maximum source-content bytes included before a truncation marker.

    Returns:
        LoadedAgentInstructions: Bounded content and per-source provenance.

    Raises:
        OSError: An applicable instruction file cannot be read.
        UnicodeError: An applicable instruction file is not valid UTF-8.
    """
    discovered = []
    for agents_file in get_agents_files(working_directory, agents_filename):
        content = agents_file.read_text(encoding="utf-8").strip()
        if content:
            discovered.append((agents_file, content))

    if not discovered:
        return LoadedAgentInstructions(
            content=None,
            sources=(),
            max_bytes=max_bytes,
            truncated=False,
        )

    complete_size = len("\n\n".join(content for _, content in discovered).encode("utf-8"))
    marker_size = len(TRUNCATION_MARKER.encode("utf-8"))
    marker = TRUNCATION_MARKER if marker_size <= max_bytes < complete_size else ""
    remaining = max_bytes - len(marker.encode("utf-8"))
    included = []
    sources = []
    truncated = False
    for index, (path, content) in enumerate(discovered):
        separator = "\n\n" if included else ""
        separator_size = len(separator.encode("utf-8"))
        encoded = content.encode("utf-8")
        available = max(remaining - separator_size, 0)
        included_content = encoded[:available].decode("utf-8", errors="ignore")
        included_bytes = len(included_content.encode("utf-8"))
        source_truncated = included_bytes < len(encoded)
        if included_content:
            included.append(f"{separator}{included_content}")
        remaining -= separator_size + included_bytes
        truncated = truncated or source_truncated
        sources.append(
            AgentInstructionsSource(
                path=path,
                size_bytes=len(encoded),
                included_bytes=included_bytes,
                truncated=source_truncated,
                content=included_content,
            )
        )
        if remaining <= 0:
            for omitted_path, omitted_content in discovered[index + 1 :]:
                omitted_size = len(omitted_content.encode("utf-8"))
                sources.append(
                    AgentInstructionsSource(
                        path=omitted_path,
                        size_bytes=omitted_size,
                        included_bytes=0,
                        truncated=True,
                        content="",
                    )
                )
            truncated = truncated or index + 1 < len(discovered)
            break

    content = "".join(included) or None
    if truncated and marker:
        content = f"{content or ''}{marker}"
    return LoadedAgentInstructions(
        content=content,
        sources=tuple(sources),
        max_bytes=max_bytes,
        truncated=truncated,
    )


def build_instructions(*sections: str | None) -> str | None:
    """Combine non-empty instruction sections without changing their content.

    Args:
        *sections (str | None): Optional instruction sections in desired output order.

    Returns:
        str | None: Sections separated by blank lines, or ``None`` when every section is empty.
    """
    included = [section for section in sections if section]
    return "\n\n".join(included) or None
