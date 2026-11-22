"""Discover literal shell and build entry points for executable binding."""

from __future__ import annotations

import os
import re
import shlex
import stat
from collections.abc import Callable, Collection
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from .. import constants
from ..permissions import CommandInspection, literal_shell_word, shell_tokens
from ..permissions.protection import is_protected_external_read
from ..utils import open_selected_file, open_selected_parent
from .sandbox.contracts import resolve_selected_executable

if TYPE_CHECKING:
    from ..permissions import CommandAnalysis

_MAX_METADATA_BYTES = 64 * 1024
_MAX_INSTALLATION_ENTRIES = 1024
_MAKE_WORD = re.compile(r"[A-Za-z0-9_.-]+")
_MAKE_CONTROL_ENVIRONMENT = frozenset({"MAKEFLAGS", "GNUMAKEFLAGS", "MFLAGS", "MAKEFILES"})


class WorkspaceExecutableLink(NamedTuple):
    """Bind a workspace executable spelling to its selected file identity.

    Args:
        name (str): Command spelling selected through PATH or a literal path.
        spelling (Path): Workspace path used to select the executable.
        resolved (Path): Canonical selected file.
        device (int): Device number of the selected file.
        inode (int): Inode number of the selected file.
    """

    name: str
    spelling: Path
    resolved: Path
    device: int
    inode: int


def linked_project_installations(
    workspace: Path,
    system_roots: tuple[Path, ...],
    may_list: Callable[[Path], bool],
) -> dict[Path, list[WorkspaceExecutableLink]]:
    """Identify external installations referenced by the project’s executable links.

    Args:
        workspace (Path): Canonical active checkout.
        system_roots (tuple[Path, ...]): Already visible native system paths.
        may_list (Callable[[Path], bool]): Current permission to inspect a project directory.

    Returns:
        dict[Path, list[WorkspaceExecutableLink]]: Candidate bin/lib installations with
            identity-bound executable spellings. Each new read scope requires review.
    """
    installations: dict[Path, list[WorkspaceExecutableLink]] = {}
    for name in workspace_tool_links(workspace, may_list):
        try:
            selected = resolve_selected_executable(name, "", workspace)
            if selected is None:
                continue
            spelling, executable = selected
            root = executable.parent.parent
            if (
                executable.parent.name != "bin"
                or root.is_relative_to(workspace)
                or workspace.is_relative_to(root)
                or Path.home().is_relative_to(root)
                or any(executable.is_relative_to(system) for system in system_roots)
                or is_protected_external_read(root)
                or not (root / "lib").is_dir()
            ):
                continue
            info = executable.stat()
            if info.st_nlink == 1 and stat.S_ISREG(info.st_mode):
                link = WorkspaceExecutableLink(name, spelling, executable, info.st_dev, info.st_ino)
                installations.setdefault(root, []).append(link)
        except (OSError, RuntimeError, ValueError):
            continue
    return installations


def workspace_shell_programs(
    source: str,
    workspace: Path,
    cwd: Path,
    may_read: Callable[[Path], bool],
) -> tuple[str, ...] | None:
    """Suggest command names mentioned in a directly selected readable project shell script.

    Suggestions only label possible installed executables for human review. Shell text can
    mention programs it never runs and it can choose other programs dynamically.

    Args:
        source (str): Original shell source, used only for discovery.
        workspace (Path): Canonical checkout root.
        cwd (Path): Native command working directory.
        may_read (Callable[[Path], bool]): Current permission to read the proposed script.

    Returns:
        tuple[str, ...] | None: Suggested names for a selected project script, or None
            when no readable project script was identified. A changed search environment
            prevents suggestions based on the inherited PATH.
    """
    try:
        tokens = shell_tokens(workspace_command_source(source, workspace, cwd))
        inherited_lookup = True
        while tokens:
            exporting = tokens[0].raw == "export"
            start = 1 if exporting else 0
            end = start
            while end < len(tokens) and (
                assignment := re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)=", tokens[end].raw)
            ):
                if assignment[1] in {"PATH", "IFS", "ENV", "BASH_ENV"}:
                    inherited_lookup = False
                end += 1
            if exporting:
                if end == start or end == len(tokens) or tokens[end].raw not in {";", "&&", "\n"}:
                    return None
                tokens = tokens[end + 1 :]
            else:
                tokens = tokens[end:]
                break
        if not tokens or tokens[0].operator or not literal_shell_word(tokens[0].raw):
            return None
        selected = tokens[0].value
        if Path(selected).name in CommandInspection.SHELLS:
            if len(tokens) < 2 or tokens[1].operator or not literal_shell_word(tokens[1].raw):
                return None
            selected = tokens[1].value
        elif "/" not in selected:
            return None
        script = cwd / selected
        if (
            not script.is_relative_to(workspace)
            or ".." in script.parts
            or is_protected_external_read(script.relative_to(workspace))
        ):
            return None
        body = _read_workspace_file(workspace, script, may_read)
        if not body or not body.startswith(b"#!"):
            return None
        header, _, commands = body.partition(b"\n")
        words = header[2:].decode().split()
        if not words or Path(words[-1]).name not in CommandInspection.SHELLS:
            return None
        if not inherited_lookup:
            return ()
        names = CommandInspection().analyze(commands.decode()).executables
        return tuple(name for name in names if _MAKE_WORD.fullmatch(name))[
            : constants.MAX_COMMAND_PATH_COUNT
        ]
    except (OSError, RuntimeError, ValueError):
        return None


def workspace_tool_links(workspace: Path, may_list: Callable[[Path], bool]) -> tuple[str, ...]:
    """Find linked executables in immediate project tool installations without running them.

    Inspect only the conventional bin directory of each immediate project directory.
    An incomplete or oversized inventory supplies no candidates. A candidate only identifies
    a tool installation for review and does not authorize reading its contents.

    Args:
        workspace (Path): Canonical active checkout.
        may_list (Callable[[Path], bool]): Current permission to inspect a workspace directory.

    Returns:
        tuple[str, ...]: Absolute executable link spellings in deterministic order.
    """
    limit = constants.MAX_COMMAND_PATH_COUNT
    if not may_list(workspace):
        return ()
    links = set()
    try:
        with os.scandir(workspace) as children:
            entries = tuple(islice(children, limit + 1))
            if len(entries) > limit:
                return ()
            directories = [entry.name for entry in entries if entry.is_dir(follow_symlinks=False)]
        for child in sorted(directories):
            directory = workspace / child / "bin"
            if not may_list(directory):
                continue
            try:
                with open_selected_parent(directory) as parent:
                    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_CLOEXEC
                    descriptor = os.open(directory.name, flags, dir_fd=parent)
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                continue
            try:
                with os.scandir(descriptor) as entries:
                    for count, entry in enumerate(entries):
                        if count >= _MAX_INSTALLATION_ENTRIES:
                            return ()
                        if entry.is_symlink():
                            links.add(str(directory / entry.name))
                            if len(links) > limit:
                                return ()
            finally:
                os.close(descriptor)
    except (OSError, RuntimeError):
        return ()
    return tuple(sorted(links))


def workspace_command_source(source: str, workspace: Path, cwd: Path) -> str:
    """Return the command after a verified no-op workspace ``cd``, if present.

    Args:
        source (str): Original shell source, which remains unchanged for execution.
        workspace (Path): Canonical workspace root bound to ``WORKSPACE_ROOT``.
        cwd (Path): Canonical initial working directory.

    Returns:
        str: The remaining source when the initial directory change is provably a no-op,
            otherwise the original source.
    """
    try:
        tokens = shell_tokens(source)
        if len(tokens) < 4 or tokens[0].raw != "cd":
            return source
        index = 1
        if tokens[index].value == "--" and literal_shell_word(tokens[index].raw):
            index += 1
        if index + 2 >= len(tokens):
            return source
        destination, separator = tokens[index : index + 2]
        if separator.value not in {";", "&&", "\n"} or not separator.operator:
            return source
        if destination.raw in {'"$WORKSPACE_ROOT"', '"${WORKSPACE_ROOT}"'}:
            target = workspace
        elif (
            not destination.operator
            and literal_shell_word(destination.raw)
            and destination.value
            and all(part == "." for part in destination.value.split("/") if part)
            and not destination.value.startswith("/")
        ):
            target = cwd
        else:
            return source
        return source[tokens[index + 2].start :] if target == cwd else source
    except ValueError:
        return source


def selected_workspace_executable(
    source: str,
    analysis: CommandAnalysis,
    workspace: Path,
) -> str | None:
    """Resolve a direct executable using only the bound workspace variable spelling.

    Args:
        source (str): Original shell source, retained unchanged for execution.
        analysis (CommandAnalysis): Parsed first executable and literal-selection status.
        workspace (Path): Canonical directory bound to ``WORKSPACE_ROOT``.

    Returns:
        str | None: Exact workspace executable path, or None for ambiguous expansion.
    """
    if (
        not analysis.first_executable_initial
        or analysis.first_executable_literal
        or not analysis.executables
    ):
        return None
    name = analysis.executables[0]
    try:
        tokens = shell_tokens(source)
    except ValueError:
        return None
    for token in tokens:
        if not token.operator and token.value == name:
            raw = token.raw
            break
        variable, separator, _ = token.value.partition("=")
        if (
            token.operator
            or not literal_shell_word(token.raw)
            or not separator
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable) is None
            or variable in {"WORKSPACE_ROOT", "IFS"}
        ):
            return None
    else:
        return None
    quoted = ('"$WORKSPACE_ROOT"/', '"${WORKSPACE_ROOT}"/')
    fully_quoted = ('"$WORKSPACE_ROOT/', '"${WORKSPACE_ROOT}/')
    unquoted = ("$WORKSPACE_ROOT/", "${WORKSPACE_ROOT}/")
    prefix = next(
        (item for item in (*quoted, *fully_quoted, *unquoted) if raw.startswith(item)), None
    )
    if prefix is None:
        return None
    suffix = raw[len(prefix) :]
    if prefix in fully_quoted:
        if not suffix.endswith('"'):
            return None
        suffix = suffix[:-1]
    relative = Path(suffix)
    if (
        not suffix
        or re.fullmatch(r"[A-Za-z0-9_./@+-]+", suffix) is None
        or relative.is_absolute()
        or ".." in relative.parts
        or prefix in unquoted
        and any(char in str(workspace) for char in " \t\r\n*?[]")
    ):
        return None
    return str(workspace / relative)


def timeout_child_source(source: str) -> str | None:
    """Identify a literal child of a simple timeout invocation for approval discovery.

    Args:
        source (str): Unmodified shell source after any verified no-op workspace change.

    Returns:
        str | None: Original source beginning at the literal child, or None when ambiguous.
    """
    try:
        tokens = shell_tokens(source)
    except ValueError:
        return None
    if len(tokens) < 3:
        return None
    wrapper, duration, child = tokens[:3]
    if (
        wrapper.operator
        or duration.operator
        or child.operator
        or not all(literal_shell_word(token.raw) for token in (wrapper, duration, child))
        or wrapper.value not in {"timeout", "gtimeout"}
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[smhd]?", duration.value) is None
        or not child.value
        or "/" in child.value
        or child.value.startswith("-")
        or re.match(r"[A-Za-z_][A-Za-z0-9_]*=", child.value) is not None
    ):
        return None
    return source[child.start :]


def discover_scratch_shell_source(
    source: str,
    scratch: Path,
    cwd: Path,
    path_value: str,
    trusted_shell: Callable[[Path], bool],
    *,
    may_read: Callable[[Path], bool],
) -> str | None:
    """Read an unambiguously selected session shell script for advisory tool discovery.

    Args:
        source (str): Original command or its verified no-op workspace suffix.
        scratch (Path): Bound session temporary directory.
        cwd (Path): Original command working directory.
        path_value (str): Inherited PATH used to select the shell interpreter.
        trusted_shell (Callable[[Path], bool]): Validate a selected system shell path.
        may_read (Callable[[Path], bool]): Apply current policy before reading the script.

    Returns:
        str | None: Stable bounded UTF-8 script source, or None when selection is ambiguous.
    """
    try:
        tokens = shell_tokens(source)
        if len(tokens) < 2 or tokens[0].operator or tokens[1].operator:
            return None
        shell, argument = tokens[:2]
        if (
            not literal_shell_word(shell.raw)
            or Path(shell.value).name not in CommandInspection.SHELLS
        ):
            return None
        value = argument.value
        if argument.raw not in (value, f'"{value}"'):
            return None
        prefix = next((item for item in ("$TMPDIR/", "${TMPDIR}/") if value.startswith(item)), None)
        if prefix is None:
            return None
        suffix = value[len(prefix) :]
        if not suffix or any(char in suffix for char in "$`*?[]{}\\\n\r\x00"):
            return None
        relative = Path(suffix)
        if relative.is_absolute() or ".." in relative.parts or is_protected_external_read(relative):
            return None
        selected = resolve_selected_executable(shell.value, path_value, cwd)
        if selected is None or not all(trusted_shell(path) for path in selected):
            return None
    except PermissionError:
        raise
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        content = _read_workspace_file(scratch, scratch / relative, may_read)
    except PermissionError:
        raise
    except OSError:
        return None
    try:
        if content is None or not content.startswith(b"#!"):
            return None
        header, separator, body = content.partition(b"\n")
        if not separator:
            return None
        interpreter = os.fsdecode(header[2:].strip())
        if not interpreter.startswith("/") or any(char.isspace() for char in interpreter):
            return None
        if Path(interpreter).resolve(strict=True) != selected[1]:
            return None
        source = body.decode("utf-8")
        tokens = shell_tokens(source)
        if any(
            "`" in token.raw
            or token.raw in CommandInspection.SHELL_KEYWORDS | {"function", "(", ")", "<<", "<<-"}
            for token in tokens
        ):
            return None
        return source
    except (OSError, RuntimeError, ValueError):
        return None


def _system_make_selected(name: str, path_value: str, cwd: Path) -> bool:
    """Require PATH to select the platform's original system make executable."""
    try:
        selected = resolve_selected_executable(name, path_value, cwd)
        return selected is not None and selected[1] == Path("/usr/bin/make").resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return False


def _makefile_shadowed(cwd: Path) -> bool:
    """Reject a higher-priority makefile, allowing case-insensitive aliases of the same file."""
    alternate = cwd / "makefile"
    if _exists(cwd / "GNUmakefile"):
        return True
    if not _exists(alternate):
        return False
    try:
        return alternate.is_symlink() or not alternate.samefile(cwd / "Makefile")
    except OSError:
        return True


def discover_makefile_commands(
    source: str,
    workspace: Path,
    cwd: Path,
    path_value: str,
    *,
    may_read: Callable[[Path], bool],
    environment_names: Collection[str] = (),
) -> tuple[str, ...]:
    """Suggest literal recipes from a narrowly recognized system make invocation.

    Args:
        source (str): Original command source, never executed for discovery.
        workspace (Path): Canonical workspace containing the makefile.
        cwd (Path): Canonical directory in which make will run.
        path_value (str): Inherited PATH used to identify the selected make executable.
        may_read (Callable[[Path], bool]): Apply current policy before reading the recipe.
        environment_names (Collection[str]): Variables declared in the command environment.

    Returns:
        tuple[str, ...]: Literal candidate recipes in prerequisite order; an empty tuple means
            the invocation or makefile cannot be identified safely. Candidates are not authority.
    """
    if not _MAKE_CONTROL_ENVIRONMENT.isdisjoint(environment_names):
        return ()
    source = workspace_command_source(source, workspace, cwd)
    if len(source) > 2 * constants.MAX_COMMAND_PATH_LENGTH or "\r" in source:
        return ()
    try:
        inspection = CommandInspection()
        tokens = shell_tokens(source)
        pipe = next(
            (index for index, token in enumerate(tokens) if token.operator and token.value == "|"),
            None,
        )
        if pipe is not None:
            expecting_command = True
            for token in tokens[pipe + 1 :]:
                if token.operator and token.value == "|" and not expecting_command:
                    expecting_command = True
                elif (
                    token.operator
                    or not literal_shell_word(token.raw)
                    or not token.value
                    or expecting_command
                    and (
                        token.raw in CommandInspection.SHELL_KEYWORDS | {"!"}
                        or re.match(r"[A-Za-z_][A-Za-z0-9_]*=", token.value) is not None
                    )
                ):
                    return ()
                else:
                    expecting_command = False
            if expecting_command:
                return ()
            source = source[: tokens[pipe].start]
            tokens = tokens[:pipe]
        analysis = inspection.analyze(source)
        while tokens and tokens[-1].operator and tokens[-1].value == "\n":
            tokens.pop()
        if tokens and tokens[-1].operator and tokens[-1].value == ";":
            tokens.pop()
        for index, token in enumerate(tokens):
            if token.operator and (
                token.value not in {"<", ">", ">>", "<>", ">|", "<&", ">&"}
                or index + 1 >= len(tokens)
                or tokens[index + 1].operator
                or not literal_shell_word(tokens[index + 1].raw)
                or (
                    tokens[index + 1].value not in {"0", "1", "2", "-"}
                    if token.value in {"<&", ">&"}
                    else tokens[index + 1].value != "/dev/null"
                )
            ):
                return ()
        words, _ = inspection.strip_redirections(tokens, ())
    except ValueError:
        return ()
    arguments = analysis.first_executable_arguments
    if (
        not analysis.executables
        or not analysis.first_executable_initial
        or not analysis.first_executable_literal
        or not analysis.first_path_inherited
        or len(arguments) != 1
        or len(words) < 2
        or words[-2].value != analysis.executables[0]
        or words[-1].value != arguments[0]
        or any(not literal_shell_word(word.raw) for word in words[:-2])
        or any(word.value.partition("=")[0] in _MAKE_CONTROL_ENVIRONMENT for word in words[:-2])
        or arguments[0].startswith("-")
        or _MAKE_WORD.fullmatch(arguments[0]) is None
        or not cwd.is_relative_to(workspace)
        or is_protected_external_read(cwd.relative_to(workspace))
        or not _system_make_selected(analysis.executables[0], path_value, cwd)
        or _makefile_shadowed(cwd)
    ):
        return ()
    makefile = cwd / "Makefile"
    if not _exists(makefile):
        return ()
    content = _read_workspace_file(workspace, makefile, may_read)
    if content is None:
        return ()
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeError:
        return ()
    rules: dict[str, tuple[tuple[str, ...], list[str]]] = {}
    phony: set[str] = set()
    current = None
    for line in lines:
        if line.startswith("\t"):
            if current is None:
                return ()
            rules[current][1].append(line[1:].lstrip(" \t@+-"))
            continue
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\\" in line or "$" in line or "#" in line:
            return ()
        name, separator, prerequisites = line.partition(":")
        parts = prerequisites.split()
        if not separator or any(_MAKE_WORD.fullmatch(part) is None for part in parts):
            return ()
        if name == ".PHONY":
            phony.update(parts)
            current = None
            continue
        if (
            _MAKE_WORD.fullmatch(name) is None
            or name.startswith(".")
            or name in {"Makefile", "makefile", "GNUmakefile"}
            or name in rules
            or len(rules) >= constants.MAX_COMMAND_PATH_COUNT
        ):
            return ()
        current = name
        rules[name] = (tuple(parts), [])
    commands: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> bool:
        """Collect an explicit, acyclic phony rule and its supported literal recipes."""
        if name in visited:
            return True
        if name in visiting or name not in phony or name not in rules:
            return False
        visiting.add(name)
        prerequisites, recipes = rules[name]
        if not all(visit(item) for item in prerequisites):
            return False
        for recipe in recipes:
            if (
                len(commands) >= constants.MAX_COMMAND_PATH_COUNT
                or not recipe
                or any(char in recipe for char in "\\$`;&|<>\n")
            ):
                return False
            try:
                shlex.split(recipe)
                if any(token.operator for token in shell_tokens(recipe)):
                    return False
            except ValueError:
                return False
            commands.append(recipe)
        visiting.remove(name)
        visited.add(name)
        return True

    return tuple(commands) if visit(arguments[0]) else ()


def _read_workspace_file(
    workspace: Path,
    path: Path,
    may_read: Callable[[Path], bool],
) -> bytes | None:
    """Read a bounded stable regular workspace file without following path components."""
    if not may_read(path):
        return None
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        return None
    if before.st_size > _MAX_METADATA_BYTES:
        return None
    descriptor = open_selected_file(workspace, path, (before.st_dev, before.st_ino))
    try:
        bound = os.fstat(descriptor)
        if not may_read(path):
            return None
        content = os.read(descriptor, _MAX_METADATA_BYTES + 1)
        latest = os.fstat(descriptor)
        if len(content) > _MAX_METADATA_BYTES or (
            bound.st_size,
            bound.st_mtime_ns,
            bound.st_ctime_ns,
        ) != (latest.st_size, latest.st_mtime_ns, latest.st_ctime_ns):
            return None
        return content
    finally:
        os.close(descriptor)


def _exists(path: Path) -> bool:
    """Treat an inaccessible or dangling configuration path as present."""
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True
