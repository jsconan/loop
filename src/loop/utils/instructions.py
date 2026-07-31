"""Discover and parse project instruction files."""

from collections.abc import Collection
from pathlib import Path
from typing import Any

import yaml

from .path import find_project_root

MAX_AGENTS_BYTES = 32 * 1024
DEFAULT_AGENTS_FILENAME = "AGENTS.md"
DEFAULT_SKILLS_DIRECTORY = Path(".agents/skills")


def instruction_directories(
    working_directory: Path,
    relative_directory: Path = DEFAULT_SKILLS_DIRECTORY,
) -> list[Path]:
    """Return project instruction directories in scope order, followed by the user directory.

    Args:
        working_directory: Directory whose repository-scoped instructions should apply.
        relative_directory: Instruction directory relative to each project and user scope.

    Returns:
        Project directories from the repository root through ``working_directory``, followed
        by the directory in the user's home. Outside a repository, only the working-directory
        and user scopes are returned.
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
        directories = list(reversed(scoped))
    directories.append(Path.home() / relative_directory)
    return directories


def read_instruction_frontmatter(
    location: Path,
    required_fields: Collection[str] = (),
) -> dict[str, Any]:
    """Read and validate the YAML frontmatter of an instruction file.

    Args:
        location: Instruction file to read as UTF-8.
        required_fields: Metadata fields that must contain non-empty strings. Valid values
            are stripped before being returned.

    Returns:
        The decoded frontmatter mapping.

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
        content: Complete instruction-file content.
        filename: Display name used in validation errors.

    Returns:
        The trimmed Markdown following the closing frontmatter delimiter.

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


def load_agents_instructions(
    working_directory: Path | str,
    agents_filename: str = DEFAULT_AGENTS_FILENAME,
    max_bytes: int = MAX_AGENTS_BYTES,
) -> str | None:
    """Load AGENTS.md files from the project root through the working directory.

    Args:
        working_directory: Directory whose instruction scope should be loaded.
        agents_filename: Name of the instruction file to discover.
        max_bytes: Maximum encoded size of the combined instructions.

    Returns:
        The combined instructions in scope, truncated to ``max_bytes``, or ``None``
        when no non-empty AGENTS.md file applies.

    Raises:
        OSError: An applicable instruction file cannot be read.
        UnicodeError: An applicable instruction file is not valid UTF-8.
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

    instructions = []
    for directory in directories:
        agents_file = directory / agents_filename
        if agents_file.is_file():
            content = agents_file.read_text(encoding="utf-8").strip()
            if content:
                instructions.append(content)

    if not instructions:
        return None

    combined = "\n\n".join(instructions)
    encoded = combined.encode("utf-8")
    if len(encoded) <= max_bytes:
        return combined
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def build_instructions(*sections: str | None) -> str | None:
    """Combine non-empty instruction sections without changing their content.

    Args:
        *sections: Optional instruction sections in desired output order.

    Returns:
        Sections separated by blank lines, or ``None`` when every section is empty.
    """
    included = [section for section in sections if section]
    return "\n\n".join(included) or None
