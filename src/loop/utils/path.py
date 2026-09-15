"""Provide repository-aware path discovery and traversal utilities."""

import os
import shlex
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec

from .. import constants
from .models import IgnoreRule, IgnoreRules
from .process import parse_command_line


class VirtualPath:
    """Translate a fixed model-visible filesystem into local execution paths.

    Args:
        workspace (Path | None): Local directory represented as ``/workspace``.
        temporary_directory (Path | None): Local directory represented as ``/tmp``.
        skill_roots (Mapping[str, Path]): Skill resource directories represented below
            ``/skills/<name>``.

    Virtual paths are a typed tool-interface protocol, not shell aliases. Callers resolve them
    before a filesystem operation or a process ``cwd`` is planned. Command arguments remain opaque
    and should use relative paths from the selected virtual working directory.
    """

    WORKSPACE = "/workspace"
    TEMPORARY = "/tmp"
    SKILLS = "/skills"
    EXTERNAL = "<external>"

    _roots: dict[str, Path]

    def __init__(
        self,
        workspace: Path | None = None,
        temporary_directory: Path | None = None,
        skill_roots: Mapping[str, Path] | None = None,
    ) -> None:
        roots = {}
        if workspace is not None:
            roots[self.WORKSPACE] = workspace.absolute()
        if temporary_directory is not None:
            roots[self.TEMPORARY] = temporary_directory.absolute()
        roots.update(
            (f"{self.SKILLS}/{name}", root.absolute()) for name, root in (skill_roots or {}).items()
        )
        self._roots = roots

    def resolve(self, value: str) -> str:
        """Resolve one virtual or workspace-relative path for local execution.

        Args:
            value (str): Model-supplied virtual or workspace-relative path.

        Returns:
            str: Local lexical path, preserving leaf symlinks for mutation planners.

        Raises:
            ValueError: The path is outside a configured virtual root or escapes its root.
        """
        if not value or "//" in value:
            raise ValueError("Path must be a virtual or workspace-relative path.")
        if value.startswith(("workspace:", "scratch:", "skill:")):  # Detect deprecated path aliases
            raise ValueError("Path aliases are not supported; use a VirtualPath.")
        if value.startswith("/"):
            for virtual_root, local_root in self._roots.items():
                if value == virtual_root or value.startswith(f"{virtual_root}/"):
                    return self._local_path(local_root, value[len(virtual_root) :].lstrip("/"))
            raise ValueError("Absolute paths must be inside a configured virtual root.")
        workspace = self._roots.get(self.WORKSPACE)
        if workspace is None:
            raise ValueError("Workspace-relative paths require a configured workspace.")
        return self._local_path(workspace, value)

    def display(self, value: str) -> str:
        """Render a local path without disclosing configured host roots.

        Args:
            value (str): Local path or relative metadata path.

        Returns:
            str: Virtual path, unchanged relative path, or ``<external>``.
        """
        path = Path(value)
        if not path.is_absolute():
            return value
        for virtual_root, local_root in sorted(
            self._roots.items(), key=lambda pair: len(str(pair[1])), reverse=True
        ):
            if path.is_relative_to(local_root):
                relative = path.relative_to(local_root).as_posix()
                return virtual_root if relative == "." else f"{virtual_root}/{relative}"
        return self.EXTERNAL

    def resolve_command(self, command: str) -> str:
        """Resolve VirtualPath arguments in one restricted command line.

        Args:
            command (str): Model-supplied shell-free command line.

        Returns:
            str: Equivalent command line with virtual path arguments replaced by local paths.

        Raises:
            ValueError: The command is malformed or a virtual argument escapes its root.
        """
        argv = parse_command_line(command)
        return shlex.join(tuple(self._resolve_command_argument(argument) for argument in argv))

    def metadata(self, value: object, fields: tuple[tuple[str, ...], ...]) -> object:
        """Render declared local metadata fields as virtual paths.

        Args:
            value (object): Parsed tool result or reference metadata.
            fields (tuple[tuple[str, ...], ...]): Declared key paths; ``"*"`` selects list items.

        Returns:
            object: Copy with declared metadata path values represented virtually.
        """

        def rewrite(item: object, path: tuple[str, ...]) -> object:
            if not path:
                return self.display(item) if isinstance(item, str) else item
            head, *tail = path
            if head == "*" and isinstance(item, list):
                return [rewrite(child, tuple(tail)) for child in item]
            if isinstance(item, dict) and head in item:
                return {**item, head: rewrite(item[head], tuple(tail))}
            return item

        prepared = value
        for field in fields:
            prepared = rewrite(prepared, field)
        return prepared

    def redact(self, value: str) -> str:
        """Replace configured local roots in implementation-generated text.

        Args:
            value (str): Diagnostic text that may include one declared root.

        Returns:
            str: Text with configured local roots represented virtually.
        """
        for prefix, root in sorted(
            self._roots.items(), key=lambda pair: len(str(pair[1])), reverse=True
        ):
            value = value.replace(str(root), prefix)
        return value

    def _resolve_command_argument(self, argument: str) -> str:
        """Resolve an exact virtual argument or the value portion of an option."""
        value = argument
        prefix = ""
        if argument.startswith("-") and "=" in argument:
            prefix, value = argument.split("=", maxsplit=1)
            prefix += "="
        if any(value == root or value.startswith(f"{root}/") for root in self._roots):
            return prefix + self.resolve(value)
        return argument

    @staticmethod
    def _local_path(root: Path, suffix: str) -> str:
        """Join one virtual suffix to its local root without allowing traversal."""
        if ".." in PurePosixPath(suffix).parts:
            raise ValueError("Virtual path escapes its root.")
        candidate = Path(os.path.abspath(root / suffix))
        return str(candidate)


def canonical_path(path: Path | str) -> str:
    """Return an absolute normalized path without requiring it to exist.

    Args:
        path (Path | str): Filesystem path to normalize.

    Returns:
        str: Absolute canonical representation.
    """
    candidate = Path(path)
    if candidate.exists() or candidate.is_symlink():
        return str(candidate.resolve())
    return str(candidate.parent.resolve() / candidate.name)


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
    if is_directory and path.name in {
        constants.GIT_DIRECTORY.name,
        constants.APP_DIRECTORY.name,
    }:
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


def filter_paths_by_globs(
    paths: Iterable[Path], root: Path | str, patterns: Sequence[str] | None
) -> Iterator[Path]:
    """Yield paths whose root-relative names match any supplied Git-style glob.

    Args:
        paths (Iterable[Path]): Candidate paths to filter.
        root (Path | str): Root used to derive portable relative names.
        patterns (Sequence[str] | None): Inclusive Git-style globs, or ``None`` to include all.

    Yields:
        Path: Candidate paths selected by at least one pattern, or every path without patterns.
    """
    if not patterns:
        yield from paths
        return
    spec = GitIgnoreSpec.from_lines(patterns)
    root = Path(root).resolve()
    for path in paths:
        if spec.match_file(path.resolve().relative_to(root).as_posix()):
            yield path
