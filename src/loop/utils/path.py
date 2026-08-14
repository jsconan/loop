"""Provide repository-aware path discovery and traversal utilities."""

from pathlib import Path
from typing import Iterator

from pathspec import GitIgnoreSpec

from .. import constants
from .models import IgnoreRule, IgnoreRules


def find_project_root(working_directory: Path | str) -> Path | None:
    """Return the closest Git project root containing the working directory.

    Args:
        working_directory (Path | str): Directory from which to search upward.

    Returns:
        Path | None: The closest directory containing a ``.git`` marker, or ``None`` when no
        project root is found.
    """
    working_directory = Path(working_directory)
    for directory in (working_directory, *working_directory.parents):
        if (directory / constants.GIT_DIRECTORY).exists():
            return directory
    return None


def _load_ignore_rules(directory: Path, rules: IgnoreRules) -> None:
    """Add ignore files in a directory to their active hierarchical rule sets."""
    for filename in constants.IGNORE_FILENAMES:
        ignore_file = directory / filename
        if ignore_file.is_file():
            rules[filename].append(
                (
                    directory,
                    GitIgnoreSpec.from_lines(ignore_file.read_text(encoding="utf-8").splitlines()),
                )
            )


def _initial_ignore_rules(folder: Path) -> IgnoreRules:
    """Load ignore rules in scope from the Git project root through a folder."""
    rules = {filename: [] for filename in constants.IGNORE_FILENAMES}
    root = find_project_root(folder) or folder
    relative_folder = folder.relative_to(root)
    directory = root
    _load_ignore_rules(directory, rules)
    for part in relative_folder.parts:
        directory /= part
        _load_ignore_rules(directory, rules)
    return rules


def _ignore_decision(path: Path, is_directory: bool, rules: list[IgnoreRule]) -> bool | None:
    """Return the last matching decision from one hierarchical ignore source."""
    decision = None
    for base, spec in rules:
        relative_path = path.relative_to(base).as_posix()
        if is_directory:
            relative_path += "/"
        result = spec.check_file(relative_path)
        if result.include is not None:
            decision = result.include
    return decision


def _is_ignored(path: Path, is_directory: bool, rules: IgnoreRules) -> bool:
    """Evaluate a path against already-loaded Git and agent ignore rules."""
    if is_directory and path.name == constants.GIT_DIRECTORY.name:
        return True

    agent_decision = _ignore_decision(path, is_directory, rules[constants.AGENT_IGNORE_FILENAME])
    if agent_decision is not None:
        return agent_decision
    return _ignore_decision(path, is_directory, rules[constants.GIT_IGNORE_FILENAME]) is True


def is_path_ignored(path: Path | str, root: Path | str | None = None) -> bool:
    """Return whether an explicit path is excluded by scoped ignore files.

    Rules are loaded from the supplied root through the path's parent. When no
    root is supplied, the closest Git project root is used, falling back to the
    path's parent directory outside a repository. Ignored parent directories
    stop rule discovery, matching recursive traversal behavior.

    Args:
        path (Path | str): File or directory to evaluate.
        root (Path | str | None): Optional boundary for hierarchical ignore-file discovery.

    Returns:
        bool: Whether ``.gitignore``, higher-priority ``.agentignore``, or Git metadata
        exclusion hides the path.

    Raises:
        ValueError: If the path is outside the supplied root.
    """
    path = Path(path).resolve()
    root = Path(root).resolve() if root is not None else find_project_root(path.parent)
    root = root or path.parent
    relative_path = path.relative_to(root)
    rules = {filename: [] for filename in constants.IGNORE_FILENAMES}
    directory = root
    _load_ignore_rules(directory, rules)

    for index, part in enumerate(relative_path.parts):
        candidate = directory / part
        is_directory = candidate.is_dir()
        if _is_ignored(candidate, is_directory, rules):
            return True
        directory = candidate
        if index < len(relative_path.parts) - 1 and is_directory:
            _load_ignore_rules(directory, rules)
    return False


def _iter_visible_paths(folder: Path, recursive: bool, rules: IgnoreRules) -> Iterator[Path]:
    """Yield visible child paths, pruning ignored directories during recursion."""
    for entry in folder.iterdir():
        is_directory = entry.is_dir()
        if _is_ignored(entry, is_directory, rules):
            continue
        yield entry
        if recursive and is_directory and not entry.is_symlink():
            nested_rules = {name: active.copy() for name, active in rules.items()}
            _load_ignore_rules(entry, nested_rules)
            yield from _iter_visible_paths(entry, recursive=True, rules=nested_rules)


def iter_visible_paths(folder: Path | str, recursive: bool = False) -> Iterator[Path]:
    """Yield non-ignored paths immediately below a folder or throughout its tree.

    Hierarchical ``.gitignore`` and ``.agentignore`` files use Git pattern syntax,
    with agent-specific decisions taking precedence. Git metadata directories are
    always omitted, ignored directories are not traversed, and symbolic-link
    directories are listed without being followed.

    Args:
        folder (Path | str): Directory whose visible children should be discovered.
        recursive (bool): Whether to traverse visible child directories recursively.

    Yields:
        Path: Visible files and directories as absolute paths.
    """
    folder = Path(folder).resolve()
    yield from _iter_visible_paths(folder, recursive, _initial_ignore_rules(folder))
