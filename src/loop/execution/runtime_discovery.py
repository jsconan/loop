"""Discover identity-bound workspace runtime candidates for selected tools."""

from __future__ import annotations

import os
import re
import shlex
import stat
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .. import constants
from ..permissions import CommandInspection, literal_shell_word, shell_tokens
from ..permissions.protection import is_protected_external_read
from ..utils import open_selected_file
from .sandbox.contracts import resolve_selected_executable

if TYPE_CHECKING:
    from ..permissions import CommandAnalysis

type ExecutableIdentity = tuple[str, Path, Path, int, int]


@dataclass(frozen=True)
class RuntimeCandidate:
    """Bind a workspace executable and an alternative spelling of the same file.

    Args:
        identity (ExecutableIdentity): Workspace selection and target device/inode.
        alias_executable (Path): Metadata spelling that resolves to the same executable.
    """

    identity: ExecutableIdentity
    alias_executable: Path


_MAX_METADATA_BYTES = 64 * 1024
_VERSION = re.compile(r"\d+\.\d+(?:\.\d+)?")
_MINIMUM_VERSION = re.compile(r">=\s*(\d+\.\d+(?:\.\d+)?)")
_PROJECT_ENVIRONMENT = "UV_PROJECT_ENVIRONMENT="
_MAKE_WORD = re.compile(r"[A-Za-z0-9_.-]+")
_MAKE_CONTROL_ENVIRONMENT = frozenset({"MAKEFLAGS", "GNUMAKEFLAGS", "MFLAGS", "MAKEFILES"})


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
    source: str, analysis: CommandAnalysis, workspace: Path
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
    source: str, scratch: Path, cwd: Path, path_value: str, trusted_shell: Callable[[Path], bool]
) -> str | None:
    """Read an unambiguously selected session shell script for advisory tool discovery.

    Args:
        source (str): Original command or its verified no-op workspace suffix.
        scratch (Path): Bound session temporary directory.
        cwd (Path): Original command working directory.
        path_value (str): Inherited PATH used to select the shell interpreter.
        trusted_shell (Callable[[Path], bool]): Validate a selected system shell path.

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
        content = _read_workspace_file(scratch, scratch / relative, strict=True)
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
    source: str, workspace: Path, cwd: Path, path_value: str
) -> tuple[str, ...]:
    """Suggest literal recipes from a narrowly recognized system make invocation.

    Args:
        source (str): Original command source, never executed for discovery.
        workspace (Path): Canonical workspace containing the makefile.
        cwd (Path): Canonical directory in which make will run.
        path_value (str): Inherited PATH used to identify the selected make executable.

    Returns:
        tuple[str, ...]: Literal candidate recipes in prerequisite order; an empty tuple means
            the invocation or makefile cannot be identified safely. Candidates are not authority.
    """
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
    content = _read_workspace_file(workspace, makefile, strict=True)
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


def _read_workspace_file(workspace: Path, path: Path, *, strict: bool = False) -> bytes | None:
    """Read a bounded stable regular workspace file without following path components."""
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None
        if before.st_size > _MAX_METADATA_BYTES:
            return None
        descriptor = open_selected_file(workspace, path, (before.st_dev, before.st_ino))
        try:
            bound = os.fstat(descriptor)
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
    except (OSError, ValueError):
        if strict:
            raise
        return None


def _exists(path: Path) -> bool:
    """Treat an inaccessible or dangling configuration path as present."""
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _version(value: str) -> tuple[int, ...] | None:
    """Recognize a literal major/minor or major/minor/patch version."""
    if _VERSION.fullmatch(value) is None:
        return None
    return tuple(int(part) for part in value.split("."))


def _environment_values(content: bytes) -> dict[str, str] | None:
    """Read unambiguous environment metadata as advisory candidate evidence."""
    values = {}
    try:
        for line in content.decode("utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                key = key.strip()
                if key in values:
                    return None
                values[key] = value.strip()
    except UnicodeError:
        return None
    return values


def _environment_version(values: dict[str, str]) -> tuple[int, ...] | None:
    """Recognize supported managed or standard CPython environment metadata."""
    if values.get("implementation", "").casefold() == "cpython":
        return _version(values.get("version_info", ""))
    if "implementation" not in values and "version_info" not in values and values.get("executable"):
        return _version(values.get("version", ""))
    return None


def _workspace_interpreter(
    workspace: Path, spelling: Path, values: dict[str, str]
) -> RuntimeCandidate | None:
    """Bind a selected workspace interpreter to its matching metadata spelling."""
    try:
        resolved = spelling.resolve(strict=True)
        details = resolved.stat()
        home = Path(values.get("home", ""))
        alias_executable = home / resolved.name
        if (
            resolved.is_relative_to(workspace)
            or not home.is_absolute()
            or alias_executable.resolve(strict=True) != resolved
            or not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or not os.access(spelling, os.X_OK)
        ):
            return None
        if "implementation" not in values:
            executable = Path(values.get("executable", ""))
            if not executable.is_absolute() or executable.resolve(strict=True) != resolved:
                return None
    except (OSError, RuntimeError, ValueError):
        return None
    return RuntimeCandidate(
        (str(spelling), spelling, resolved, details.st_dev, details.st_ino), alias_executable
    )


def _uv_project_runtime(
    source: str,
    analysis: CommandAnalysis,
    workspace: Path,
    cwd: Path,
    _path_value: str,
    identities: tuple[ExecutableIdentity, ...],
) -> tuple[RuntimeCandidate, ...]:
    """Identify the default existing environment for an unambiguous project run."""
    if (
        cwd != workspace
        or not analysis.first_executable_initial
        or not analysis.first_executable_literal
        or not analysis.first_path_inherited
        or not any(identity[0] == analysis.executables[0] for identity in identities)
    ):
        return ()
    environment = workspace / ".venv"
    try:
        tokens = shell_tokens(source)
    except ValueError:
        return ()
    for token in tokens:
        if token.operator or not literal_shell_word(token.raw):
            return ()
        if token.value == analysis.executables[0]:
            break
        name, separator, value = token.value.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            return ()
        if name == _PROJECT_ENVIRONMENT[:-1]:
            if not value:
                return ()
            selected = Path(value)
            environment = selected if selected.is_absolute() else workspace / selected
            if ".." in selected.parts or not environment.is_relative_to(workspace):
                return ()
        elif name in {"PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME"} or (
            name.startswith("UV_")
            and name not in {"UV_NO_CACHE", "UV_NO_SYNC", "UV_OFFLINE", "UV_CACHE_DIR"}
        ):
            return ()
    else:
        return ()
    arguments = list(analysis.first_executable_arguments)
    while arguments and arguments[0] in {"--no-cache", "--quiet", "--offline"}:
        arguments.pop(0)
    if not arguments or arguments.pop(0) != "run":
        return ()
    while arguments and arguments[0] in {"--no-sync", "--quiet", "--offline"}:
        arguments.pop(0)
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    if not arguments or arguments[0].startswith("-") or "/" in arguments[0]:
        return ()
    if arguments[0].endswith(".py") or _exists(workspace / "uv.toml"):
        return ()

    project = _read_workspace_file(workspace, workspace / "pyproject.toml")
    config = _read_workspace_file(workspace, environment / "pyvenv.cfg")
    if project is None or config is None:
        return ()
    try:
        document = tomllib.loads(project.decode("utf-8"))
        project_table = document["project"]
        tools = document.get("tool", {})
        if not isinstance(tools, dict):
            return ()
        tool_table = tools.get("uv", {})
        if not isinstance(project_table, dict) or not isinstance(tool_table, dict):
            return ()
        if set(tool_table) - {
            "package",
            "exclude-newer",
            "dev-dependencies",
            "default-groups",
            "conflicts",
        }:
            return ()
        values = _environment_values(config)
        if values is None:
            return ()
        version = _environment_version(values)
        if version is None:
            return ()
        required = project_table.get("requires-python")
        if required is not None:
            match = _MINIMUM_VERSION.fullmatch(required) if isinstance(required, str) else None
            minimum = _version(match.group(1)) if match else None
            if minimum is None or (version + (0, 0, 0))[:3] < (minimum + (0, 0, 0))[:3]:
                return ()
        version_file = workspace / ".python-version"
        if _exists(version_file):
            contents = _read_workspace_file(workspace, version_file)
            selected = _version(contents.decode("utf-8").strip()) if contents is not None else None
            if selected is None or version[: len(selected)] != selected:
                return ()
        spelling = environment / "bin/python"
        candidate = _workspace_interpreter(workspace, spelling, values)
        if candidate is None:
            return ()
    except (KeyError, OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        return ()
    return (candidate,)


_ADAPTERS = {"uv": _uv_project_runtime}


def discover_workspace_interpreters(
    workspace: Path, identities: tuple[ExecutableIdentity, ...]
) -> tuple[RuntimeCandidate, ...]:
    """Suggest runtimes for identity-bound workspace interpreters.

    Args:
        workspace (Path): Canonical workspace containing selected interpreters.
        identities (tuple[ExecutableIdentity, ...]): Selected workspace executable identities.

    Returns:
        tuple[RuntimeCandidate, ...]: Candidates requiring validation and approval.
    """
    candidates = []
    for identity in identities[:128]:
        _, spelling, _, _, _ = identity
        if (
            not spelling.is_relative_to(workspace)
            or ".." in spelling.parts
            or spelling.parent.name != "bin"
        ):
            continue
        content = _read_workspace_file(workspace, spelling.parent.parent / "pyvenv.cfg")
        values = _environment_values(content) if content is not None else None
        if values is None or _environment_version(values) is None:
            continue
        candidate = _workspace_interpreter(workspace, spelling, values)
        if candidate is not None and candidate.identity[1:] == identity[1:]:
            candidates.append(RuntimeCandidate(identity, candidate.alias_executable))
    return tuple(dict.fromkeys(candidates))


def discover_workspace_runtimes(
    source: str,
    analysis: CommandAnalysis,
    workspace: Path,
    cwd: Path,
    path_value: str,
    identities: tuple[ExecutableIdentity, ...],
) -> tuple[RuntimeCandidate, ...]:
    """Suggest workspace runtime executables selected by a recognized invocation.

    Args:
        source (str): Original shell source being prepared, never executed for discovery.
        analysis (CommandAnalysis): Bounded literal command analysis.
        workspace (Path): Canonical workspace containing candidate environments.
        cwd (Path): Canonical current directory for the selected command.
        path_value (str): Inherited PATH, preserved unchanged by discovery.
        identities (tuple[ExecutableIdentity, ...]): Identity-bound selected executables.

    Returns:
        tuple[RuntimeCandidate, ...]: Candidates requiring runtime validation and approval.
    """
    normalized = workspace_command_source(source, workspace, cwd)
    if normalized != source:
        source = normalized
        analysis = CommandInspection().analyze(source)
    if not analysis.executables:
        return ()
    adapter = _ADAPTERS.get(Path(analysis.executables[0]).name)
    return adapter(source, analysis, workspace, cwd, path_value, identities) if adapter else ()
