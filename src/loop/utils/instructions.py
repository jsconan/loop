"""Discover project instructions from AGENTS.md files."""

from pathlib import Path

from .path import find_project_root

MAX_AGENTS_BYTES = 32 * 1024
DEFAULT_AGENTS_FILENAME = "AGENTS.md"


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
    """Combine non-empty instruction sections without changing their content."""
    included = [section for section in sections if section]
    return "\n\n".join(included) or None
