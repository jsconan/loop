"""Bind and execute shell requests behind one authorization facade."""

from __future__ import annotations

import json
import logging
import math
import os
import pwd
import re
import shlex
import stat
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

from .. import constants
from ..command_authorities import command_authorities
from ..errors import Problem, log_problem
from ..permissions import Action, ProcessBoundary, ProcessTarget
from ..permissions.protection import is_protected_external_read
from ..session.models import SessionExecutionConflictError
from ..telemetry import telemetry_trace_event
from ..utils import (
    ProcessCapture,
    VirtualPath,
    encode_content_cursor,
    open_selected_file,
    ripgrep_path,
    store_content,
)
from ..utils.process import render_terminal_text
from .coordinator import (
    CommandExecutionCoordinator,
    HostCommandExecutor,
    LocalHostCommandExecutor,
    command_deadline,
)
from .jobs import (
    JOB_CLOSE_SECONDS,
    JOB_RETENTION_SECONDS,
    MAX_ACTIVE_JOBS,
    MAX_JOB_OUTPUT_BYTES,
    MAX_JOB_OUTPUT_EVENTS,
    MAX_RETAINED_JOBS,
    CommandJob,
    CommandJobs,
)
from .runtime_discovery import (
    RuntimeCandidate,
    discover_makefile_commands,
    discover_scratch_shell_source,
    discover_workspace_interpreters,
    discover_workspace_runtimes,
    selected_workspace_executable,
    timeout_child_source,
    workspace_command_source,
)
from .sandbox import (
    CommandProcessResult,
    DirectoryIdentity,
    SandboxBackend,
    SandboxCapability,
    SandboxOutcome,
    SandboxRequest,
    path_search_roots,
    resolve_host_executable,
    select_sandbox_backend,
)
from .sandbox.constants import (
    HOME_ENVIRONMENT_VARIABLE,
    LANG_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    SYSTEM_LOCALE,
    TEMP_ENVIRONMENT_VARIABLE,
    TMP_ENVIRONMENT_VARIABLE,
    TMPDIR_ENVIRONMENT_VARIABLE,
    WORKSPACE_ROOT_ENVIRONMENT_VARIABLE,
    XDG_CACHE_HOME_ENVIRONMENT_VARIABLE,
)
from .sandbox.contracts import resolve_selected_executable as _selected_executable

_LOGGER = logging.getLogger(__name__)


def _command_paths(context: ToolContext) -> VirtualPath:
    """Use captured command roots when available, otherwise read the current roots."""
    return context.command_paths or context.instructions_manager.virtual_paths


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


class WorkspaceScriptInterpreters(NamedTuple):
    """Carry discovered interpreters, forwarding and selected workspace links.

    Args:
        interpreters (tuple[str, ...]): Selected interpreter spellings.
        followed_argument (bool): Whether shell forwarding followed a literal first argument.
        workspace_links (tuple[WorkspaceExecutableLink, ...]): Bound selected links.
    """

    interpreters: tuple[str, ...]
    followed_argument: bool
    workspace_links: tuple[WorkspaceExecutableLink, ...]


if TYPE_CHECKING:
    from ..tooling import ToolContext

_MAX_EXECUTABLE_GRANTS = 128
_MAX_EXECUTABLE_SYMLINKS = 40
_MAX_INDIRECT_PATH_ENTRIES = 1024
_MAX_INTERPRETER_LINE_BYTES = 4096
_MAX_RETURNED_DENIAL_CANDIDATES = 4
_HOST_AUTHORITY_NOTICE = (
    "The sandbox restrictions did not apply. The command ran with the host user's "
    "authority, subject to independent OS and outer-environment restrictions. "
    "A further request may repeat effects."
)
# Both streams may contain characters that expand sixfold in serialized JSON.
_STREAM_TAIL_BYTES = 512
# Keep generated interpreter lines within the shortest supported native limit.
_MAX_PORTABLE_SHEBANG_PREFIX_BYTES = 127
_FORWARDING_SHELL_HEADER = os.fsencode(f"#!{constants.SHELL_COMMAND_PATH}\n")
_FORWARDING_SHELL_PATH = os.fsencode(constants.SHELL_COMMAND_PATH)
_SAFE_FORWARDING_SHELL_ARGUMENT = re.compile(rb"(?:[ \t]+(?:-[efu]+|--))?")
_SAFE_FORWARDING_SHELL_ARGUMENTS = re.compile(rb"(?:[ \t]+-[efu]+)*(?:[ \t]+--)?")
_LITERAL_SHELL_WORD = (
    rb"""(?:[^ \t\r\n\x00;&|()<>'"\\$`*?\[\]{}~#]|'[^'\n\x00]*'|"[^"$`\\\n\x00]*")+"""
)
_LITERAL_FORWARDING_EXEC = re.compile(
    rb"exec[ \t]+("
    + _LITERAL_SHELL_WORD
    + rb"(?:[ \t]+"
    + _LITERAL_SHELL_WORD
    + rb""")*)([ \t]+"\$(?:@|\{@\})")?"""
)
_ENV_SHEBANG = re.compile(
    rb"[ \t]*/usr/bin/env[ \t]+(?:([A-Za-z0-9_./][A-Za-z0-9_./+-]*)"
    rb"|-S[ \t]+([A-Za-z0-9_./][A-Za-z0-9_./+-]*)(?:[ \t]+[A-Za-z0-9_./+=:-]+)*)"
)


@dataclass(frozen=True)
class ExecutableGrant:
    """Bind a discovered tool to its workspace, PATH, and filesystem identities.

    Args:
        name (str): Executable name looked up on PATH.
        workspace (Path | None): Workspace identity when a caller supplied one.
        cwd (Path): Working directory used to resolve relative PATH entries.
        relative_path (bool): Whether the grant's PATH snapshot depends on cwd.
        path_value (str): Bound PATH value used for lookup.
        spelling (Path): Executable path as found on PATH.
        resolved (Path): Canonical executable target.
        roots (tuple[DirectoryIdentity, ...]): Bound installation read roots.
        identity (tuple[int, int]): Executable device and inode.
        opt_aliases (tuple[tuple[Path, int, int, Path], ...]): Bound package aliases.
        managed_package (bool): Whether all roots qualify for automatic package reads.
    """

    name: str
    workspace: Path | None
    cwd: Path
    relative_path: bool
    path_value: str
    spelling: Path
    resolved: Path
    roots: tuple[DirectoryIdentity, ...]
    identity: tuple[int, int]
    opt_aliases: tuple[tuple[Path, int, int, Path], ...]
    managed_package: bool


def _stream_output(
    chunks: list[str],
    discarded: int | None,
    source: str,
) -> dict:
    """Return bounded stream previews with capture and decoding-loss metadata."""
    content = "".join(chunks)
    encoded = content.encode("utf-8")
    preview = encoded[: constants.MAX_TOOL_CONTENT_BYTES // 2].decode("utf-8", errors="ignore")
    included = len(preview.encode("utf-8"))
    result = {
        "content": preview,
        "captured_bytes": len(encoded),
        "included_bytes": included,
        "truncated": included < len(encoded) or discarded != 0,
        "capture_complete": discarded == 0,
        "discarded_characters": discarded,
        "decoding_loss": "none" if discarded == 0 and "\ufffd" not in content else "unknown",
    }
    if included < len(encoded):
        handle = store_content(encoded, source)
        result.update(
            handle=handle,
            next_cursor=encode_content_cursor(handle, included),
            continuation="Use read_cached_content with this handle and cursor.",
        )
        if discarded == 0:
            tail = encoded[max(included, len(encoded) - _STREAM_TAIL_BYTES) :].decode(
                "utf-8", errors="ignore"
            )
            result["tail"] = {
                "content": tail,
                "start_byte": len(encoded) - len(tail.encode("utf-8")),
                "end_byte": len(encoded),
            }
    return result


def _redact_denial_observation(
    record: str,
    redact: Callable[[str], str],
    display: Callable[[str], str],
) -> str:
    """Hide a protected or undisclosed denial resource before path rendering."""
    if len(tuple(re.finditer(r"\bdeny\(\d+\) \S+", record))) > 1:
        return "<ambiguous sandbox denial>"
    match = re.search(r"\bdeny\(\d+\) \S+ (.+)$", record)
    if match is None:
        match = re.match(r"\S+ (.+)$", record)
    resource = match.group(1) if match else record
    if is_protected_external_read(Path(resource)):
        record = record[: match.start(1)] + "<protected>" if match else "<protected sandbox denial>"
    elif match and Path(resource).is_absolute():
        record = record[: match.start(1)] + display(resource)
    return redact(record)


def _denial_operation_target(observation: str) -> tuple[str | None, str | None]:
    """Extract only a bounded native denial operation and resource from rendered evidence."""
    match = re.fullmatch(r"Sandbox: [^\r\n]*? deny\(\d+\) ([A-Za-z0-9_-]{1,64}) (.+)", observation)
    if match is None:
        match = re.fullmatch(r"apparmor-([A-Za-z0-9_-]{1,64}) (.+)", observation)
    if match is None:
        match = re.fullmatch(r"((?:file|network|process)-[A-Za-z0-9_-]{1,56}) (.+)", observation)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def _interrupted_effect_notice(boundary: str) -> str:
    """Explain that another execution may repeat an interrupted attempt's effects."""
    return (
        f"The {boundary} attempt may already have had effects and was not retried. "
        "Inspect captured output and state before another attempt; another attempt may repeat "
        "effects."
    )


def _automatic_executable_identities(
    names: tuple[str, ...],
    path_value: str,
    cwd: Path,
    *,
    trusted_owners: tuple[int, ...] | None = None,
) -> tuple[tuple[str, Path, Path, int, int], ...]:
    """Bind safe exact executable files found in the request's live PATH roots."""
    roots = path_search_roots({PATH_ENVIRONMENT_VARIABLE: path_value}, cwd)
    if trusted_owners is None:
        trusted_owners = (os.getuid(),)
    identities = []
    for name in dict.fromkeys(names[: constants.MAX_COMMAND_PATH_COUNT]):
        if "/" in name:
            continue
        try:
            match = resolve_host_executable(name, path_value, cwd)
        except (OSError, RuntimeError, ValueError):
            continue
        if match is None:
            continue
        spelling, resolved = match
        root = next(
            (
                canonical
                for path, canonical in roots
                if spelling.is_relative_to(path) and resolved.is_relative_to(canonical)
            ),
            None,
        )
        if root is None:
            continue
        if is_protected_external_read(spelling) or is_protected_external_read(resolved):
            continue
        try:
            root_details = root.stat()
            details = resolved.stat()
        except OSError:
            continue
        if root_details.st_uid not in trusted_owners:
            continue
        if root_details.st_mode & constants.OTHER_WRITE_MODE_BIT:
            continue
        if (
            root_details.st_mode & constants.GROUP_WRITE_MODE_BIT
            and root_details.st_gid != os.getgid()
        ):
            continue
        if details.st_uid not in trusted_owners:
            continue
        if details.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK:
            continue
        if details.st_nlink != 1:
            continue
        identities.append((name, spelling, resolved, details.st_dev, details.st_ino))
    return tuple(identities)


def _indirect_approved_executable_names(
    path_value: str, cwd: Path, roots: tuple[Path, ...], system_roots: tuple[Path, ...]
) -> tuple[str, ...]:
    """Find bounded PATH symlinks whose executable targets are already requested."""
    if not roots:
        return ()
    path_roots = path_search_roots({PATH_ENVIRONMENT_VARIABLE: path_value}, cwd)
    ordered = sorted(
        path_roots,
        key=lambda pair: any(pair[1].is_relative_to(root) for root in system_roots),
    )
    names = []
    remaining = _MAX_INDIRECT_PATH_ENTRIES
    prefixes = tuple(str(root).rstrip(os.sep) + os.sep for root in roots)
    for spelling, canonical in ordered:
        if is_protected_external_read(spelling) or is_protected_external_read(canonical):
            continue
        try:
            with os.scandir(spelling) as entries:
                for entry in entries:
                    remaining -= 1
                    if entry.is_symlink():
                        try:
                            resolved = os.path.realpath(entry.path, strict=True)
                            if any(
                                resolved.startswith(prefix) for prefix in prefixes
                            ) and os.path.isfile(resolved):
                                for earlier, _ in path_roots:
                                    selected = earlier / entry.name
                                    if selected.is_file() and os.access(selected, os.X_OK):
                                        if selected == Path(entry.path):
                                            names.append(entry.path)
                                        break
                        except (OSError, RuntimeError):
                            pass
                    if remaining == 0 or len(names) == constants.MAX_COMMAND_PATH_COUNT:
                        return tuple(sorted(set(names)))
        except OSError:
            continue
    return tuple(sorted(set(names)))


def _workspace_script_interpreters(
    names: tuple[str, ...],
    path_value: str,
    cwd: Path,
    workspace: Path,
    *,
    first_arguments: tuple[str, ...] = (),
    allow_path_lookup: bool = True,
) -> WorkspaceScriptInterpreters:
    """Find bounded workspace selections, interpreter chains, and forwarded arguments."""
    interpreters = []
    workspace_links = []
    followed_argument = False
    pending = deque(
        (name, first_arguments if index == 0 else ())
        for index, name in enumerate(dict.fromkeys(names[: constants.MAX_COMMAND_PATH_COUNT]))
    )
    seen = set()
    while pending and len(seen) < constants.MAX_COMMAND_PATH_COUNT:
        name, arguments = pending.popleft()
        if (name, arguments) in seen:
            continue
        seen.add((name, arguments))
        try:
            match = _selected_executable(name, path_value, cwd)
            if match is None:
                continue
            if (
                ".." not in match[0].parts
                and match[0].is_relative_to(workspace)
                and not match[1].is_relative_to(workspace)
            ):
                details = match[1].stat()
                if stat.S_ISREG(details.st_mode) and details.st_nlink == 1:
                    workspace_links.append(
                        WorkspaceExecutableLink(name, *match, details.st_dev, details.st_ino)
                    )
            _, script = match
            if not script.is_relative_to(workspace) or is_protected_external_read(script):
                continue
            before = script.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                continue
            descriptor = open_selected_file(workspace, script, (before.st_dev, before.st_ino))
            try:
                bound = os.fstat(descriptor)
                if not stat.S_ISREG(bound.st_mode) or bound.st_nlink != 1:
                    continue
                content = os.read(descriptor, _MAX_INTERPRETER_LINE_BYTES)
                forwarded = None
                forwarded_arguments = ()
                argument = arguments[0] if arguments else None
                line, separator, _ = content[2:].partition(b"\n")
                env = _ENV_SHEBANG.fullmatch(line) if content.startswith(b"#!") else None
                selected_interpreter = None
                group = 1 if env is not None and env.group(1) is not None else 2
                if env is not None and env.end(group) < _MAX_PORTABLE_SHEBANG_PREFIX_BYTES:
                    selected = os.fsdecode(env.group(group))
                    if "/" in selected or allow_path_lookup:
                        selected_interpreter = selected
                forwarding_shell = content.startswith(_FORWARDING_SHELL_HEADER)
                if (
                    not forwarding_shell
                    and separator
                    and len(line) < _MAX_PORTABLE_SHEBANG_PREFIX_BYTES
                    and line.startswith(_FORWARDING_SHELL_PATH)
                ):
                    forwarding_shell = (
                        _SAFE_FORWARDING_SHELL_ARGUMENT.fullmatch(
                            line[len(_FORWARDING_SHELL_PATH) :]
                        )
                        is not None
                    )
                body = content.partition(b"\n")[2].strip()
                # Normalize only shell word separators for discovery; retain the original source.
                body = re.sub(rb"^exec[ \t]+", b"exec ", body, count=1)
                if (
                    not forwarding_shell
                    and env is not None
                    and selected_interpreter is not None
                    and separator
                    and len(line) < _MAX_PORTABLE_SHEBANG_PREFIX_BYTES
                    and body.startswith((b"exec ", b"exec\t"))
                    and _SAFE_FORWARDING_SHELL_ARGUMENTS.fullmatch(line[env.end(group) :])
                    is not None
                ):
                    try:
                        shell = _selected_executable(selected_interpreter, path_value, cwd)
                        forwarding_shell = bool(
                            shell and shell[1] == Path(constants.SHELL_COMMAND_PATH).resolve()
                        )
                    except (OSError, RuntimeError, ValueError):
                        pass
                if forwarding_shell:
                    latest = os.fstat(descriptor)
                    if (
                        len(content) == bound.st_size
                        and len(content) < _MAX_INTERPRETER_LINE_BYTES
                        and (bound.st_size, bound.st_mtime_ns, bound.st_ctime_ns)
                        == (latest.st_size, latest.st_mtime_ns, latest.st_ctime_ns)
                    ):
                        if (
                            argument
                            and not argument.startswith("-")
                            and body
                            in {
                                b'exec "$1"',
                                b'exec "${1}"',
                                b'exec "$@"',
                                b'exec "${@}"',
                            }
                        ):
                            forwarded = argument
                            if body in {b'exec "$@"', b'exec "${@}"'}:
                                forwarded_arguments = arguments[1:]
                        elif (
                            argument
                            and not argument.startswith("-")
                            and body
                            in {
                                b'exec "./$1"',
                                b'exec "./${1}"',
                            }
                        ):
                            forwarded = f"./{argument}"
                        else:
                            literal = _LITERAL_FORWARDING_EXEC.fullmatch(body)
                            if literal:
                                words = shlex.split(os.fsdecode(literal.group(1)))
                                if words[0] and not words[0].startswith("-"):
                                    forwarded = words[0]
                                    forwarded_arguments = (
                                        (*words[1:], *arguments)
                                        if literal.group(2)
                                        else tuple(words[1:])
                                    )
            finally:
                os.close(descriptor)
            if forwarded is not None and ("/" in forwarded or allow_path_lookup):
                pending.append((forwarded, forwarded_arguments))
                followed_argument = True
            if not content.startswith(b"#!"):
                continue
            if not separator and len(content) == _MAX_INTERPRETER_LINE_BYTES:
                continue
            interpreter = line.lstrip(b" \t").split(b" ", 1)[0].split(b"\t", 1)[0]
            if interpreter.startswith(b"/") and b"\0" not in interpreter:
                spelling = os.fsdecode(interpreter)
                interpreters.append(spelling)
                if Path(spelling).is_relative_to(workspace):
                    pending.append((spelling, ()))
                if selected_interpreter is not None:
                    interpreters.append(selected_interpreter)
                    pending.append((selected_interpreter, ()))
        except (OSError, RuntimeError, ValueError):
            continue
    return WorkspaceScriptInterpreters(
        tuple(dict.fromkeys(interpreters)), followed_argument, tuple(workspace_links)
    )


def _executable_directory_links(
    names: tuple[str, ...],
    path_value: str,
    cwd: Path,
    roots: tuple[Path, ...] = (),
    excluded_roots: tuple[Path, ...] = (),
    workspace: Path | None = None,
) -> Iterator[tuple[Path, Path, Path]]:
    """Trace executable directory links and selected workspace file links."""
    for name in dict.fromkeys(names[: constants.MAX_COMMAND_PATH_COUNT]):
        try:
            match = _selected_executable(name, path_value, cwd)
            if (
                match is None
                or match[0] == match[1]
                or roots
                and not any(match[1].is_relative_to(root) for root in roots)
                or any(match[1].is_relative_to(root) for root in excluded_roots)
            ):
                continue
            candidate = match[0]
            for _ in range(_MAX_EXECUTABLE_SYMLINKS):
                link = next(
                    (
                        part
                        for part in reversed((candidate, *candidate.parents))
                        if part.is_symlink()
                    ),
                    None,
                )
                if link is None:
                    break
                suffix = candidate.relative_to(link)
                link = link.parent.resolve(strict=True) / link.name
                target = link.resolve(strict=True)
                if target.is_dir():
                    yield link, target, match[1]
                elif (
                    workspace is not None
                    and ".." not in match[0].parts
                    and match[0].is_relative_to(workspace)
                    and link == match[0]
                    and target.is_file()
                ):
                    yield link, target.parent, match[1]
                value = link.readlink()
                destination = value if value.is_absolute() else link.parent / value
                candidate = destination / suffix
        except (OSError, RuntimeError, ValueError):
            continue


def _approved_executable_aliases(
    names: tuple[str, ...],
    path_value: str,
    cwd: Path,
    roots: tuple[Path, ...],
    workspace: Path,
    system_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Find selected executable directory links into already requested read roots."""
    if not roots:
        return ()
    aliases = []
    for link, target, _ in _executable_directory_links(names, path_value, cwd, roots):
        try:
            if (
                any(target.is_relative_to(root) for root in roots)
                and not link.is_relative_to(workspace)
                and not any(link.is_relative_to(root) for root in (*roots, *system_roots))
                and not Path.home().resolve().is_relative_to(link.parent)
                and not is_protected_external_read(link)
            ):
                aliases.append(link)
        except (OSError, RuntimeError, ValueError):
            continue
    return tuple(dict.fromkeys(aliases))


def _interpreter_runtime_roots(
    names: tuple[str, ...],
    path_value: str,
    cwd: Path,
    workspace: Path,
    covered_roots: tuple[Path, ...],
    runtime_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Propose directory-link targets that contain a selected interpreter."""
    candidates = {}
    home = None
    for link, target, executable in _executable_directory_links(
        names, path_value, cwd, excluded_roots=covered_roots, workspace=workspace
    ):
        if (
            not executable.is_relative_to(target)
            or target.is_relative_to(workspace)
            or workspace.is_relative_to(target)
            or any(root.is_relative_to(target) for root in runtime_roots)
            or is_protected_external_read(link)
            or is_protected_external_read(target)
            or is_protected_external_read(executable)
        ):
            continue
        try:
            details = executable.stat()
        except OSError:
            continue
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            continue
        if home is None:
            try:
                home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
                home_details = home.stat()
            except (KeyError, OSError, RuntimeError, ValueError):
                return ()
            if not stat.S_ISDIR(home_details.st_mode) or home_details.st_uid != os.getuid():
                return ()
        if home.is_relative_to(target):
            continue
        current = candidates.get(executable)
        if current is None or len(target.parts) < len(current[0].parts):
            candidates[executable] = (target, details.st_dev, details.st_ino)
    verified = []
    for executable, (target, device, inode) in candidates.items():
        try:
            details = executable.stat()
        except OSError:
            continue
        if (
            stat.S_ISREG(details.st_mode)
            and details.st_nlink == 1
            and (details.st_dev, details.st_ino) == (device, inode)
        ):
            verified.append(target)
    return tuple(dict.fromkeys(verified))


def _backend_method(backend: SandboxBackend, name: str) -> Callable | None:
    """Bind an optional backend method only when its class actually defines it."""
    return getattr(backend, name) if callable(getattr(type(backend), name, None)) else None


def _installed_runtime_packages(
    backend: SandboxBackend,
    candidates: tuple[RuntimeCandidate, ...],
    workspace: Path,
    covered_roots: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Suggest verified installed runtime dependencies for already selected interpreters."""
    runtime_resolver = _backend_method(backend, "installed_runtime_roots")
    resolver = runtime_resolver or _backend_method(backend, "installed_tool_roots")
    package_root = _backend_method(backend, "installed_package_root")
    if not callable(package_root):
        package_root = _backend_method(backend, "managed_tool_root")
    if not callable(resolver) or not callable(package_root):
        return (), ()
    if not candidates:
        return (), ()
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError, RuntimeError, ValueError):
        return (), ()
    roots = []
    aliases = []
    for candidate in candidates:
        if is_protected_external_read(candidate.identity[2]):
            continue
        installation = (
            resolver(
                candidate.identity[2], spelling=candidate.alias_executable, workspace=workspace
            )
            if callable(runtime_resolver)
            else resolver(candidate.identity[2])
        )
        if installation is None:
            continue
        package_roots, package_aliases = installation
        if (
            not package_roots
            or not candidate.identity[2].is_relative_to(package_roots[0])
            or len(roots) + len(package_roots) > constants.MAX_COMMAND_PATH_COUNT
            or any(
                is_protected_external_read(root)
                or workspace.is_relative_to(root)
                or home.is_relative_to(root)
                or package_root(root) is not True
                for root in package_roots
            )
        ):
            continue
        for alias, device, inode, target in package_aliases:
            details = alias.lstat()
            if (
                (details.st_dev, details.st_ino) != (device, inode)
                or alias.resolve(strict=True) != target
                or target not in package_roots
                or is_protected_external_read(alias)
            ):
                break
        else:
            roots.extend(
                root
                for root in package_roots
                if not any(root.is_relative_to(covered) for covered in covered_roots)
            )
            aliases.extend(alias for alias, _, _, _ in package_aliases)
    return tuple(dict.fromkeys(roots)), tuple(dict.fromkeys(aliases))


class CommandExecutionService:
    """Own request binding and sandbox/host execution for registered command tools.

    Args:
        backend (SandboxBackend): Platform sandbox selected by the composition root.
        host_executor (HostCommandExecutor): Supervisor for separately approved host retries.
    """

    _backend: SandboxBackend
    _host_executor: HostCommandExecutor
    _executable_grants: dict[str, ExecutableGrant]
    _jobs: CommandJobs
    _backend_lock: threading.Lock
    _backend_condition: threading.Condition
    _active_backend_uses: int
    _closing: bool
    _backend_closed: bool

    def __init__(self, backend: SandboxBackend, host_executor: HostCommandExecutor) -> None:
        self._backend = backend
        self._host_executor = host_executor
        self._executable_grants = {}
        self._jobs = CommandJobs()
        self._backend_lock = threading.Lock()
        self._backend_condition = threading.Condition(self._backend_lock)
        self._active_backend_uses = 0
        self._closing = False
        self._backend_closed = False

    @contextmanager
    def _backend_lease(self) -> Generator[bool, None, None]:
        """Keep the selected backend alive for one public operation."""
        with self._backend_lock:
            available = not self._closing
            if available:
                self._active_backend_uses += 1
        try:
            yield available
        finally:
            if available:
                with self._backend_condition:
                    self._active_backend_uses -= 1
                    if (
                        self._closing
                        and self._active_backend_uses == 0
                        and not self._backend_closed
                    ):
                        try:
                            self._backend.close()
                        except Exception as error:  # noqa: BLE001 - preserve the command result.
                            log_problem(
                                _LOGGER,
                                Problem(
                                    code="sandbox.backend_cleanup_failed",
                                    title="Sandbox cleanup failed",
                                    detail="The native backend could not close "
                                    "after command settlement.",
                                    operation="run_command",
                                ),
                                error,
                            )
                        else:
                            self._backend_closed = True
                    self._backend_condition.notify_all()

    def refresh_backend(self) -> None:
        """Reselect native authority after an approved sandbox setup change.

        Rebuild the backend against the current installed generation, discard grants bound to the
        previous backend, and close its resources after the replacement is ready. Wait for
        concurrent admission to finish selection and refuse replacement while any operation
        still uses the current backend.

        Raises:
            RuntimeError: The service is closing or an active operation still depends on the
                current backend.
        """
        with self._backend_lock:
            if self._closing or self._active_backend_uses or self._jobs.has_active():
                raise RuntimeError(
                    "Cannot refresh the native sandbox while command execution is active."
                )
            replacement = select_sandbox_backend()
            previous = self._backend
            self._backend = replacement
            self._executable_grants.clear()
            previous.close()

    @property
    def backend_capabilities(self) -> frozenset[SandboxCapability]:
        """Expose the selected native backend's qualified feature set.

        Returns:
            frozenset[SandboxCapability]: Qualified capabilities declared by the backend,
                or empty for an invalid declaration.
        """
        capabilities = self._backend.capabilities
        return capabilities if isinstance(capabilities, frozenset) else frozenset()

    @classmethod
    def for_host(cls) -> CommandExecutionService:
        """Compose trusted built-in execution boundaries for the current host.

        Returns:
            CommandExecutionService: Native-probed sandbox and separate host supervisor.
        """
        return cls(select_sandbox_backend(), LocalHostCommandExecutor())

    def resolve_executable(
        self, context: ToolContext, name: str | list[str], cwd: str = "."
    ) -> dict:
        """Look up host tools and offer bound sandbox read references.

        Args:
            context (ToolContext): Invocation context and model path boundary.
            name (str | list[str]): One name or a bounded list of names to inspect on host PATH.
            cwd (str): Workspace working directory used to resolve relative PATH entries.

        Returns:
            dict: One lookup result, or ordered results under ``executables`` for a list.

        Raises:
            ValueError: If the name or PATH cannot be searched safely.
            RuntimeError: Native execution service has closed.
        """
        with self._backend_lease() as available:
            if not available:
                raise RuntimeError("Native execution service is closed.")
            path_value = os.environ.get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH)
            workspace = (
                Path(_command_paths(context).resolve(VirtualPath.WORKSPACE)).resolve()
                if context.instructions_manager is not None
                else None
            )
            if context.instructions_manager is not None:
                candidate = Path(cwd)
                if (
                    not candidate.is_absolute()
                    or cwd == VirtualPath.WORKSPACE
                    or cwd.startswith(f"{VirtualPath.WORKSPACE}/")
                ):
                    candidate = Path(_command_paths(context).resolve(cwd))
                try:
                    lookup_cwd = candidate.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise ValueError("Executable lookup working directory is unavailable.") from exc
            else:
                lookup_cwd = Path(cwd)
                if not lookup_cwd.is_absolute():
                    lookup_cwd = Path.cwd() / lookup_cwd
                try:
                    lookup_cwd = lookup_cwd.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise ValueError("Executable lookup working directory is unavailable.") from exc
            if not lookup_cwd.is_dir() or (
                workspace is not None and not lookup_cwd.is_relative_to(workspace)
            ):
                raise ValueError(
                    "Executable lookup working directory must be inside the workspace."
                )
            if isinstance(name, list):
                if (
                    not name
                    or len(name) > constants.MAX_COMMAND_PATH_COUNT
                    or not all(isinstance(item, str) for item in name)
                ):
                    raise ValueError(
                        "Executable lookup requires between 1 and "
                        f"{constants.MAX_COMMAND_PATH_COUNT} names."
                    )
                for item in name:
                    resolve_host_executable(item, path_value, lookup_cwd)
                return {
                    "executables": [self.resolve_executable(context, item, cwd) for item in name]
                }
            result, grant = self._discover_executable(
                context,
                name,
                path_value,
                lookup_cwd,
                workspace,
            )
            if grant is None:
                return result
            reference = uuid4().hex
            if len(self._executable_grants) >= _MAX_EXECUTABLE_GRANTS:
                self._executable_grants.pop(next(iter(self._executable_grants)))
            self._executable_grants[reference] = grant
            result["grant_reference"] = reference
            if context.instructions_manager is None:
                result["candidate_read_roots"] = [str(item.path) for item in grant.roots]
            result["grant_note"] = (
                "Pass grant_reference as executable_grant to run_command. Verified installed "
                "package roots are available without a separate prompt; other tool folders "
                "require an additional-read approval. Manual read roots and other expanded "
                "capabilities also require approval. "
                "If this tool starts another external executable, look up that executable "
                "and pass both references as a list; keep the command inside the sandbox."
            )
            return result

    def _discover_executable(
        self,
        context: ToolContext,
        name: str,
        path_value: str,
        lookup_cwd: Path,
        workspace: Path | None,
        *,
        match: tuple[Path, Path] | None = None,
        optional_metadata: bool = False,
    ) -> tuple[dict, ExecutableGrant | None]:
        """Discover one executable without publishing or evicting a lookup reference."""
        if match is None:
            match = resolve_host_executable(name, path_value, lookup_cwd)
        if match is None:
            return {"name": name, "status": "absent_from_host_path"}, None
        spelling, resolved = match
        relative_path = any(not Path(item).is_absolute() for item in path_value.split(os.pathsep))
        system_roots = getattr(self._backend, "system_read_roots", ())
        if not isinstance(system_roots, tuple):
            system_roots = ()
        readable = any(
            resolved.is_relative_to(root) and spelling.is_relative_to(root) for root in system_roots
        ) or (workspace is not None and resolved.is_relative_to(workspace))
        result = {
            "name": name,
            "status": "installed_on_host",
            "path": context.display_path(spelling),
            "resolved_path": context.display_path(resolved),
            "sandbox_access": "default_read" if readable else "external_read_grant_required",
        }
        if readable:
            return result, None
        if is_protected_external_read(resolved):
            result["sandbox_access"] = "protected_external_path"
            return result, None
        resolver = getattr(self._backend, "installed_tool_roots", None)
        optional_resolver = _backend_method(self._backend, "installed_runtime_roots")
        if optional_metadata and callable(optional_resolver):
            installation = optional_resolver(resolved)
        else:
            installation = resolver(resolved) if callable(resolver) else ((resolved.parent,), ())
        if installation is None:
            result["sandbox_access"] = "external_grant_unavailable"
            return result, None
        roots, opt_aliases = (
            installation
            if isinstance(installation, tuple) and len(installation) == 2
            else ((resolved.parent,), ())
        )
        if any(is_protected_external_read(item) for item in roots):
            result["sandbox_access"] = "protected_external_path"
            return result, None
        try:
            identities = tuple(DirectoryIdentity.capture(item) for item in roots)
            details = resolved.stat()
        except (OSError, ValueError):
            result["sandbox_access"] = "external_grant_unavailable"
            return result, None
        grant = ExecutableGrant(
            name=name,
            workspace=workspace,
            cwd=lookup_cwd,
            relative_path=relative_path,
            path_value=path_value,
            spelling=spelling,
            resolved=resolved,
            roots=identities,
            identity=(details.st_dev, details.st_ino),
            opt_aliases=opt_aliases,
            managed_package=all(self._backend.managed_tool_root(item) is True for item in roots)
            if callable(getattr(self._backend, "managed_tool_root", None))
            else False,
        )
        return result, grant

    def _executable_read_roots(
        self,
        grant: ExecutableGrant,
        workspace: Path,
        path_value: str,
        cwd: Path,
    ) -> tuple[Path, ...]:
        """Resolve an executable grant only while its host identity remains current."""
        name = grant.name
        bound_workspace = grant.workspace
        bound_cwd = grant.cwd
        cwd_bound = grant.relative_path
        bound_path = grant.path_value
        spelling = grant.spelling
        resolved = grant.resolved
        identities = grant.roots
        identity = grant.identity
        opt_aliases = grant.opt_aliases
        if (
            (bound_workspace is not None and bound_workspace != workspace)
            or bound_path != path_value
            or (cwd_bound and bound_cwd != cwd)
        ):
            raise ValueError("Executable grant context changed; look up the tool again.")
        match = resolve_host_executable(name, path_value, cwd)
        try:
            details = resolved.stat()
        except OSError as exc:
            raise ValueError("Executable grant target changed; look up the tool again.") from exc
        if (
            match != (spelling, resolved)
            or (details.st_dev, details.st_ino) != identity
            or not all(item.is_current() for item in identities)
        ):
            raise ValueError("Executable grant identity changed; look up the tool again.")
        if grant.managed_package and not all(
            self._backend.managed_tool_root(item.path) is True for item in identities
        ):
            raise ValueError("Installed package verification changed; look up the tool again.")
        for alias, device, inode, target in opt_aliases:
            try:
                alias_details = alias.lstat()
                if (alias_details.st_dev, alias_details.st_ino) != (device, inode) or alias.resolve(
                    strict=True
                ) != target:
                    raise ValueError("Executable alias changed; look up the tool again.")
            except OSError as exc:
                raise ValueError("Executable alias changed; look up the tool again.") from exc
        return tuple(item.path for item in identities)

    def _automatic_tool_grants(
        self,
        context: ToolContext,
        names: tuple[str, ...],
        workspace: Path,
        cwd: Path | None = None,
    ) -> tuple[ExecutableGrant, ...]:
        """Bind verified managed tools named by a shell command."""
        working_directory = cwd or workspace
        grants: list[ExecutableGrant] = []
        path_value = os.environ.get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH)
        bound_workspace = workspace if context.instructions_manager is not None else None
        for name in list(dict.fromkeys(names))[: constants.MAX_COMMAND_PATH_COUNT]:
            if "/" in name or name in {".", ".."}:
                continue
            try:
                match = resolve_host_executable(name, path_value, working_directory)
            except ValueError:
                continue
            if match is None:
                continue
            _, grant = self._discover_executable(
                context,
                name,
                path_value,
                working_directory,
                bound_workspace,
                match=match,
                optional_metadata=True,
            )
            if grant is not None and grant.managed_package:
                grants.append(grant)
        return tuple(grants)

    def close(self) -> None:
        """Stop new admission and close the backend when its last native user settles."""
        with self._backend_lock:
            self._closing = True
        jobs_settled = self._jobs.close()
        deadline = time.monotonic() + JOB_CLOSE_SECONDS
        with self._backend_condition:
            while self._active_backend_uses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._backend_condition.wait(remaining)
            can_close = self._active_backend_uses == 0
            if can_close and not self._backend_closed:
                self._backend.close()
                self._backend_closed = True
        if not jobs_settled or not can_close:
            log_problem(
                _LOGGER,
                Problem(
                    code="command_job.cleanup_uncertain",
                    title="Command cleanup uncertain",
                    detail="An active command did not settle within the shutdown cleanup budget.",
                    operation="run_command",
                ),
            )
            return

    @staticmethod
    def _job_owner(context: ToolContext) -> str:
        """Bind a continuation to the selected session's live scratch directory."""
        root = (
            dict(_command_paths(context).command_roots()).get(VirtualPath.TEMPORARY)
            if context.instructions_manager is not None
            else None
        )
        if root is None and context.permission_manager is not None:
            root = context.permission_manager.temporary_directory
        if root is None:
            raise ValueError("Session scratch identity is unavailable.")
        path = Path(root).resolve(strict=True)
        if context.session_manager is not None and path != Path(
            context.session_manager.temporary_directory
        ).resolve(strict=True):
            raise ValueError("Session scratch identity differs from its owner.")
        details = path.stat()
        return f"{path}:{details.st_dev}:{details.st_ino}"

    def has_active_jobs(self) -> bool:
        """Check whether any command worker is still active.

        Returns:
            bool: Whether an active worker still owns a command.
        """
        return self._jobs.has_active()

    def inspect_command_jobs(self, context: ToolContext) -> dict | Problem:
        """List only the selected session's retained command handles.

        Args:
            context (ToolContext): Current registry-bound session identity.

        Returns:
            dict | Problem: Owned job states and process-wide resource ceilings.
        """
        try:
            owner = self._job_owner(context)
        except (OSError, ValueError):
            return Problem(
                code="command_job.unavailable",
                title="Command jobs unavailable",
                detail="Session scratch identity is unavailable.",
                operation=context.tool_name,
            )
        return {
            "jobs": self._jobs.inspect(owner),
            "limits": {
                "active_jobs": MAX_ACTIVE_JOBS,
                "retained_jobs": MAX_RETAINED_JOBS,
                "output_bytes_per_job": MAX_JOB_OUTPUT_BYTES,
                "output_events_per_job": MAX_JOB_OUTPUT_EVENTS,
                "retention_seconds": JOB_RETENTION_SECONDS,
            },
        }

    def read_command_job(
        self,
        context: ToolContext,
        handle: str,
        cursor: int = 0,
        wait_seconds: float | None = None,
        include_output: bool = False,
    ) -> dict | Problem:
        """Read one bounded page from a session-owned command without replaying it.

        Args:
            context (ToolContext): Current registry-bound session identity.
            handle (str): Opaque handle returned by the original command call.
            cursor (int): Next output event cursor.
            wait_seconds (float | None): Maximum wait for output or completion, or None for the
                configured default; capped at the supported job-read limit.
            include_output (bool): Include final captured stream previews explicitly.

        Returns:
            dict | Problem: Ordered page and final result after retained output, or a safe refusal.
        """
        try:
            owner = self._job_owner(context)
        except (OSError, ValueError):
            return Problem(
                code="command_job.unavailable",
                title="Command job unavailable",
                detail="Session scratch identity is unavailable.",
                operation=context.tool_name,
            )
        job = self._jobs.get(owner, handle)
        if isinstance(job, Problem):
            return job
        if wait_seconds is None:
            wait_seconds = context.settings.command_job_wait_seconds
        if (
            not isinstance(wait_seconds, bool)
            and isinstance(wait_seconds, (int, float))
            and (isinstance(wait_seconds, int) or math.isfinite(wait_seconds))
            and wait_seconds >= 0
        ):
            wait_seconds = min(
                wait_seconds,
                threading.TIMEOUT_MAX,
            )
        try:
            return job.read(
                cursor,
                wait_seconds,
                include_output=include_output,
                cancellation_event=context.cancellation_event,
            )
        except ValueError as error:
            return Problem(
                code="command_job.invalid_read",
                title="Invalid command job read",
                detail=str(error),
                operation=context.tool_name,
            )

    def cancel_command_job(self, context: ToolContext, handle: str) -> dict | Problem:
        """Request cancellation of one owned job without claiming cleanup is complete.

        Args:
            context (ToolContext): Current registry-bound session identity.
            handle (str): Opaque handle returned by the original command call.

        Returns:
            dict | Problem: Current state with cancellation requested, or a safe refusal.
        """
        try:
            owner = self._job_owner(context)
        except (OSError, ValueError):
            return Problem(
                code="command_job.unavailable",
                title="Command job unavailable",
                detail="Session scratch identity is unavailable.",
                operation=context.tool_name,
            )
        job = self._jobs.get(owner, handle)
        if isinstance(job, Problem):
            return job
        job.cancel()
        return job.inspect()

    def control_command_terminal(
        self,
        context: ToolContext,
        handle: str,
        action: str,
        content: str | None = None,
        rows: int | None = None,
        columns: int | None = None,
    ) -> dict | Problem:
        """Apply one bounded control to an owned terminal without recording input.

        Args:
            context (ToolContext): Current registry-bound session identity.
            handle (str): Original interactive job handle.
            action (str): Write, resize, EOF, interrupt, or close action.
            content (str | None): Text supplied only for write.
            rows (int | None): Height supplied only for resize.
            columns (int | None): Width supplied only for resize.

        Returns:
            dict | Problem: Current or closed job state, or a bounded refusal.
        """
        try:
            owner = self._job_owner(context)
        except (OSError, ValueError):
            return Problem(
                code="command_terminal.unavailable",
                title="Interactive terminal unavailable",
                detail="Session scratch identity is unavailable.",
                operation=context.tool_name,
            )
        job = self._jobs.get(owner, handle)
        if isinstance(job, Problem):
            return job
        if not job.terminal:
            return Problem(
                code="command_terminal.unavailable",
                title="Interactive terminal unavailable",
                detail="This handle does not own an interactive terminal.",
                operation=context.tool_name,
            )
        try:
            if action == "write" and content is not None and rows is None and columns is None:
                job.write_terminal(content)
            elif (
                action == "resize" and content is None and rows is not None and columns is not None
            ):
                job.resize_terminal(rows, columns)
            elif action == "eof" and content is None and rows is None and columns is None:
                job.terminal_eof()
            elif action == "interrupt" and content is None and rows is None and columns is None:
                job.terminal_interrupt()
            elif action == "close" and content is None and rows is None and columns is None:
                return self._jobs.close_terminal(owner, handle)
            else:
                raise ValueError("Interactive terminal action arguments are invalid.")
        except (TypeError, ValueError) as error:
            return Problem(
                code="command_terminal.invalid_control",
                title="Invalid terminal control",
                detail=str(error),
                operation=context.tool_name,
            )
        return job.inspect()

    def run_trusted_search(
        self,
        arguments: tuple[str, ...],
        descriptors: tuple[int, ...],
        deadline: float,
    ) -> ProcessCapture | None:
        """Run installed ripgrep with fixed argv and selected read descriptors only.

        Args:
            arguments (tuple[str, ...]): Fixed ripgrep options, query, and fd paths.
            descriptors (tuple[int, ...]): Preopened selected files that may be read.
            deadline (float): Absolute monotonic search deadline.

        Returns:
            ProcessCapture | None: Bounded native result, or None for safe in-process fallback.

        Raises:
            OSError: Installed-tool dependency metadata cannot be read.
            RuntimeError: Trusted runtime path authority is invalid.
            ValueError: Installed-tool metadata access is vetoed.
        """
        with self._backend_lease() as available:
            if not available:
                return None
            runner = getattr(self._backend, "run_read_only_argv", None)
            if not callable(runner):
                return None
            try:
                executable = Path(ripgrep_path())
                executable = executable.resolve(strict=True)
                if executable.stat().st_nlink != 1:
                    return None
            except (OSError, RuntimeError, ValueError):
                return None
            try:
                roots = self._backend.system_read_roots
            except (OSError, ValueError) as error:
                raise RuntimeError("Native search runtime authority is invalid.") from error
            trusted = any(executable.is_relative_to(root) for root in roots)
            if not trusted:
                optional_resolver = _backend_method(self._backend, "installed_runtime_roots")
                installation = (
                    optional_resolver(executable)
                    if callable(optional_resolver)
                    else self._backend.installed_tool_roots(executable)
                )
                if installation is None:
                    return None
                roots, _ = installation
                trusted = bool(roots) and all(
                    self._backend.managed_tool_root(root) is True for root in roots
                )
            if not trusted:
                return None
            return runner(executable, arguments, descriptors, deadline)

    def _run_command_attempt(
        self,
        context: ToolContext,
        request: SandboxRequest,
        redact: Callable[[str], str],
        display_path: Callable[[str], str],
        *,
        job: CommandJob,
        skill_snapshot: tuple[tuple[str, Path, str], ...] = (),
        prelaunch_failure: CommandProcessResult | None = None,
        reason: str | None = None,
        admission_cancellation_event: threading.Event | None = None,
    ) -> dict | Problem:
        """Execute one bound request and render its output and diagnostics."""
        coordinator = CommandExecutionCoordinator(
            self._backend,
            self._host_executor,
            context.permission_manager,
            context.interaction,
            **({"host_audit": context.audit_host_detached} if context.audit_host_detached else {}),
        )
        reason_argument = {"reason": reason} if reason is not None else {}
        invalidated = False
        invalidate = context.invalidate_detached or context.invalidate_instructions
        admit_skills = (
            context.admit_skills_detached or context.instructions_manager.admit_command_skills
            if skill_snapshot
            else None
        )

        def sandbox_admitted() -> bool | None:
            """Commit job admission only while its used skills remain active."""
            if admit_skills is not None:
                return admit_skills(skill_snapshot, job.admitted)
            job.admitted()
            return None

        try:
            attempt = coordinator.run_sandboxed(
                request,
                context.tool_name,
                prelaunch_failure=prelaunch_failure,
                execution_timeout=context.settings.command_hard_timeout,
                cancellation_event=context.cancellation_event,
                admission_cancellation_event=admission_cancellation_event,
                on_admitted=sandbox_admitted,
                output_observer=job.publish,
                **({"terminal_input": job.next_terminal_frame} if job.terminal else {}),
                **(
                    {"egress_authority": context.egress_authority}
                    if context.egress_authority is not None and request.egress_origins
                    else {}
                ),
                **reason_argument,
            )
            possible_effects = attempt.result is not None and (
                attempt.result.outcome is SandboxOutcome.COMPLETED
                or attempt.result.possible_effects
            )
            retry = None
            if attempt.result is not None and attempt.offer is not None:
                reserved_interaction = job.begin_interaction()
                if reserved_interaction:

                    def complete_host_admission() -> None:
                        """Release the approval interaction before host supervision."""
                        nonlocal reserved_interaction
                        if (
                            admission_cancellation_event is not None
                            and admission_cancellation_event.is_set()
                        ):
                            job.cancel()
                        job.admitted()
                        job.end_interaction()
                        reserved_interaction = False

                    def host_authorized() -> bool:
                        """Commit skill authority after approval without releasing the owner."""
                        return admit_skills(skill_snapshot, lambda: None)

                    try:
                        retry = coordinator.retry_host_command(
                            attempt.offer,
                            request,
                            execution_timeout=context.settings.command_hard_timeout,
                            cancellation_event=context.cancellation_event,
                            on_admitted=complete_host_admission,
                            **({"on_authorized": host_authorized} if admit_skills else {}),
                        )
                    finally:
                        if reserved_interaction:
                            job.end_interaction()
            if possible_effects or (
                retry is not None
                and (retry.outcome is SandboxOutcome.COMPLETED or retry.possible_effects)
            ):
                invalidated = True
                invalidate()
        except BaseException as exc:
            if not invalidated:
                try:
                    invalidate()
                except BaseException as invalidation_error:
                    raise exc from invalidation_error
            raise
        if attempt.result is None:
            return Problem(
                code="tool.denied",
                title="Tool call denied",
                detail="Native command was not approved.",
                operation=context.tool_name,
            )
        result = attempt.result
        boundary = "sandbox"
        if (
            retry is None
            and attempt.offer is not None
            and context.cancellation_event is not None
            and context.cancellation_event.is_set()
        ):
            result = replace(result, outcome=SandboxOutcome.CANCELLED)
        if retry is not None:
            result = retry
            boundary = "host"
        if request.terminal:
            result = replace(
                result,
                stdout=render_terminal_text(result.stdout),
                stderr=render_terminal_text(result.stderr),
            )
        observed_denial = attempt.result.observed_denial
        denial_candidates = []
        startup_baseline = "not_checked"
        if observed_denial and attempt.result.outcome is SandboxOutcome.COMPLETED:
            classify = (
                _backend_method(self._backend, "classify_denial_candidates")
                if attempt.result.diagnostic_source != "unknown"
                else None
            )
            if callable(classify):
                records, startup_baseline = classify(observed_denial)
            else:
                records = ((observed_denial, False),)
                startup_baseline = "unavailable"
            rendered_records = tuple(
                (
                    _redact_denial_observation(record, redact, display_path),
                    startup_match,
                    record.endswith("…"),
                )
                for record, startup_match in records
            )
            telemetry_trace_event(
                "sandbox.unattributed_denial",
                payload_factory=lambda selected=rendered_records: {
                    "observed_denials": [record for record, _, _ in selected],
                    "source": attempt.result.diagnostic_source,
                },
                attempt_id=request.attempt_id,
            )
            denial_candidates = []
            for record, startup_match, truncated in rendered_records[
                :_MAX_RETURNED_DENIAL_CANDIDATES
            ]:
                operation, target = (
                    _denial_operation_target(record)
                    if attempt.result.diagnostic_source != "unknown"
                    else (None, None)
                )
                candidate = {
                    "observation": record,
                    "classification": (
                        "also_observed_in_successful_startup_probe"
                        if startup_match
                        else "unclassified"
                    ),
                    "attribution": "unknown",
                    "truncated": truncated,
                }
                if operation is not None:
                    candidate["operation"] = operation
                if target is not None and not truncated:
                    candidate["target"] = target
                denial_candidates.append(candidate)
            observed_denial = ""
        collection = attempt.result.diagnostic_status
        if (
            collection == "unknown"
            and getattr(type(self._backend), "denial_observation_supported", None) is False
        ):
            collection = "unavailable"
        output = {
            "boundary": boundary,
            "exit_code": result.exit_code,
            "lifecycle": {
                "cancel_requested": bool(
                    context.cancellation_event and context.cancellation_event.is_set()
                ),
                "main_process": "exited"
                if result.outcome is SandboxOutcome.COMPLETED
                else "unknown",
                "descendant_cleanup": "verified_absent"
                if result.cleanup_verified is True
                else "uncertain"
                if result.outcome is SandboxOutcome.COMPLETED or result.possible_effects
                else "not_started",
            },
            "stdout": _stream_output([result.stdout], result.stdout_discarded, "command stdout"),
            "stderr": _stream_output([result.stderr], result.stderr_discarded, "command stderr"),
            "sandbox_diagnostics": {
                "outcome": attempt.result.outcome.value,
                "exit_code": attempt.result.exit_code,
                "possible_effects": possible_effects,
                "detail": redact(attempt.result.detail) if attempt.result.detail else None,
                "collection": collection,
                "source": (
                    attempt.result.diagnostic_source
                    if observed_denial or denial_candidates
                    else "unknown"
                ),
                "observed_denial": redact(observed_denial) if observed_denial else None,
                "cause": "unknown",
                "denial_candidates": denial_candidates,
                "startup_probe": startup_baseline,
                "observations_may_be_incomplete": True,
            },
        }
        if request.terminal:
            output["terminal_output"] = "escaped-controls"
        if boundary == "host":
            output["authority_notice"] = _HOST_AUTHORITY_NOTICE
        if attempt.result.outcome is not SandboxOutcome.COMPLETED or attempt.result.exit_code != 0:
            output["sandbox_diagnostics"].update(
                requested_operation="process_execute",
                requested_cwd=redact(str(request.cwd)),
                required_capability="native_sandbox" if prelaunch_failure else None,
                failure_stage=(
                    redact(attempt.result.failure_context)
                    if attempt.result.outcome is not SandboxOutcome.COMPLETED
                    and (
                        not possible_effects
                        or attempt.result.failure_context
                        != "Native sandbox execution could not start."
                    )
                    else None
                ),
                failure_source=(
                    "capability_check"
                    if prelaunch_failure is not None
                    else "native_backend"
                    if attempt.result.outcome is not SandboxOutcome.COMPLETED
                    else "command_exit"
                ),
                cause_confidence="unknown",
                sandbox_approval_supported=False if prelaunch_failure else None,
                next_request=(
                    "request_host_command" if attempt.offer and boundary == "sandbox" else None
                ),
            )
        if denial_candidates:
            output["sandbox_diagnostics"]["candidate_note"] = (
                (
                    "The command succeeded; denied operations may have been handled or optional "
                    "and do not establish failure. "
                    if attempt.result.exit_code == 0
                    else ""
                )
                + "These observations do not establish the cause of the exit. "
                "A startup-probe match may be incidental but can still matter to this command. "
                "Unclassified observations may also be incidental. Names and paths are untrusted."
            )
        if boundary == "host":
            output["possible_effects"] = (
                result.outcome is SandboxOutcome.COMPLETED or result.possible_effects
            )
        if result.outcome is not SandboxOutcome.COMPLETED and result.possible_effects:
            output["recovery_hint"] = _interrupted_effect_notice(
                "sandboxed" if boundary == "sandbox" else "host"
            )
        if boundary == "sandbox":
            output["scratch"] = {
                "path": redact(request.shell_environment()[TMPDIR_ENVIRONMENT_VARIABLE]),
                "lifetime": "this session",
                "retrieval": (
                    f"Use {VirtualPath.TEMPORARY}/... in file tools; quote it in shell commands."
                ),
            }
            if result.possible_effects and result.outcome in {
                SandboxOutcome.UNAVAILABLE,
                SandboxOutcome.DENIED,
            }:
                output["recovery_hint"] += (
                    " If host access is needed, request_host_command is a separate "
                    "request subject to authorization."
                )
            if result.outcome is SandboxOutcome.COMPLETED and result.exit_code in (126, 127):
                output["executable_hint"] = (
                    "This exit status does not prove a tool is absent from the host. "
                    "If the output indicates an unavailable executable, use resolve_executable "
                    "with its name to check host PATH; "
                    "pass its grant_reference as executable_grant for sandboxed access "
                    "when an external tool needs explicit access. If additional indirect "
                    "tools remain unidentified, look up only those needed. "
                    "The command may already have had effects; inspect state before another run."
                )
        if observed_denial:
            observed = redact(observed_denial)
            if boundary == "sandbox":
                output["sandbox_diagnostic"] = observed
            else:
                output["sandbox_attempt"] = {
                    "exit_code": attempt.result.exit_code,
                    "observed_denial": observed,
                }
        if attempt.offer is not None and boundary == "sandbox":
            failure = redact(result.detail)
            output["host_offer"] = (
                f"Sandbox failure: {failure}\nCommand: "
                f"{json.dumps(redact(request.source))}\n"
                "Any unrestricted host retry requires separate authorization."
            )
        if result.outcome is SandboxOutcome.COMPLETED:
            if result.exit_code != 0:
                return Problem(
                    code="process.nonzero_exit",
                    title="Command failed",
                    detail=f"Command exited with code {result.exit_code}.",
                    operation=context.tool_name,
                    metadata=output,
                )
            return output
        return Problem(
            code=f"{boundary if boundary == 'sandbox' else 'host_command'}.{result.outcome.value}",
            title="Sandbox command did not complete"
            if boundary == "sandbox"
            else "Host command did not complete",
            detail=redact(result.detail or result.outcome.value),
            operation=context.tool_name,
            metadata=output,
        )

    def request_host_command(
        self, context: ToolContext, command: str, cwd: str, reason: str
    ) -> dict | Problem:
        """Request one explicitly authorized unrestricted command without a sandbox attempt.

        Args:
            context (ToolContext): Registry-bound host operation and user interaction.
            command (str): Exact opaque shell source from the authorized operation.
            cwd (str): Canonical working directory from the authorized operation.
            reason (str): Explanation of why unrestricted execution is requested.

        Returns:
            dict | Problem: Bounded host output or a typed denial or failure.
        """
        operations = context.operations
        operation = operations[0] if len(operations) == 1 else None
        target = operation.target if operation is not None else None
        if (
            context.permission_manager is None
            or operation is None
            or operation.action is not Action.PROCESS_EXECUTE
            or operation.tool_id != context.tool_name
            or not isinstance(target, ProcessTarget)
            or target.boundary is not ProcessBoundary.HOST
            or target.argv
            != (constants.SHELL_COMMAND_PATH, constants.SHELL_COMMAND_OPTION, command)
            or target.cwd != cwd
            or not reason.strip()
        ):
            return Problem(
                code="host_command.invalid_request",
                title="Host command unavailable",
                detail="A bound host operation and a reason are required.",
                operation=context.tool_name,
            )
        try:
            command_cwd = Path(cwd).resolve(strict=True)
            workspace = (
                Path(_command_paths(context).resolve(VirtualPath.WORKSPACE)).resolve(strict=True)
                if context.instructions_manager is not None
                else command_cwd
            )
            paths = (
                _command_paths(context)
                if context.instructions_manager is not None
                else VirtualPath(
                    workspace=workspace,
                    temporary_directory=context.permission_manager.temporary_directory,
                )
            )
            scratch = dict(paths.command_roots()).get(
                VirtualPath.TEMPORARY, context.permission_manager.temporary_directory
            )
            environment = {
                PATH_ENVIRONMENT_VARIABLE: os.environ.get(
                    PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH
                ),
                LANG_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                WORKSPACE_ROOT_ENVIRONMENT_VARIABLE: str(workspace),
                TMPDIR_ENVIRONMENT_VARIABLE: str(scratch),
                TMP_ENVIRONMENT_VARIABLE: str(scratch),
                TEMP_ENVIRONMENT_VARIABLE: str(scratch),
                XDG_CACHE_HOME_ENVIRONMENT_VARIABLE: str(scratch / "cache"),
            }
            request = SandboxRequest.create(
                source=command,
                cwd=command_cwd,
                workspace=workspace,
                read_roots=(),
                write_roots=(workspace,),
                network=False,
                environment=environment,
                policy_version="host-request-v1",
                deadline=time.monotonic() + context.settings.command_hard_timeout,
                workspace_id=str(workspace),
                scratch_root=scratch,
            )
        except (OSError, ValueError) as exc:
            return Problem(
                code="host_command.invalid_request",
                title="Host command unavailable",
                detail=str(exc),
                operation=context.tool_name,
            )
        coordinator = CommandExecutionCoordinator(
            self._backend, self._host_executor, context.permission_manager, context.interaction
        )
        try:
            result = coordinator.request_host_command(
                request,
                context.tool_name,
                reason,
                execution_timeout=context.settings.command_hard_timeout,
                cancellation_event=context.cancellation_event,
            )
        except BaseException as exc:
            try:
                context.invalidate_instructions()
            except BaseException as invalidation_error:
                raise exc from invalidation_error
            raise
        if result is None:
            return Problem(
                code="host_command.denied",
                title="Host command denied",
                detail="The host request was denied or its reviewed context changed.",
                operation=context.tool_name,
            )
        if result.outcome is SandboxOutcome.COMPLETED or result.possible_effects:
            context.invalidate_instructions()
        output = {
            "boundary": "host",
            "authority_notice": _HOST_AUTHORITY_NOTICE,
            "exit_code": result.exit_code,
            "possible_effects": result.outcome is SandboxOutcome.COMPLETED
            or result.possible_effects,
            "stdout": _stream_output([result.stdout], result.stdout_discarded, "command stdout"),
            "stderr": _stream_output([result.stderr], result.stderr_discarded, "command stderr"),
        }
        if result.outcome is not SandboxOutcome.COMPLETED and result.possible_effects:
            output["recovery_hint"] = _interrupted_effect_notice("host")
        if result.outcome is SandboxOutcome.COMPLETED:
            if result.exit_code == 0:
                return output
            return Problem(
                code="process.nonzero_exit",
                title="Command failed",
                detail=f"Command exited with code {result.exit_code}.",
                operation=context.tool_name,
                metadata=output,
            )
        return Problem(
            code=f"host_command.{result.outcome.value}",
            title="Host command did not complete",
            detail=paths.redact(result.detail or result.outcome.value),
            operation=context.tool_name,
            metadata=output,
        )

    def run_command(
        self,
        context: ToolContext,
        command: str,
        cwd: str = ".",
        read_only: bool = False,
        read_roots: list[str] | None = None,
        write_roots: list[str] | None = None,
        executable_grant: str | list[str] | None = None,
        network: bool = False,
        git_write: bool = False,
        reason: str | None = None,
        egress_origins: list[str] | None = None,
        interactive: bool = False,
        wait_seconds: float | None = None,
        job: CommandJob | None = None,
        admission_cancellation_event: threading.Event | None = None,
    ) -> dict | Problem:
        """Bind and authorize a shell request, with separate host recovery when needed.

        Args:
            context (ToolContext): Registered identity and dispatch authority.
            command (str): Opaque source passed to the host shell.
            cwd (str): Resolved host working directory.
            read_only (bool): Disable workspace writes while permitting private scratch writes.
            read_roots (list[str] | None): Additional explicitly approved read roots.
            write_roots (list[str] | None): Additional identity-bound writable directories.
            executable_grant (str | list[str] | None): Bound references from executable lookups.
            network (bool): Request general network egress.
            egress_origins (list[str] | None): Canonical HTTP(S) origins for a qualified mediator.
            git_write (bool): Request Git metadata write authority.
            reason (str | None): Explanation shown to the user for additional sandbox authority.
            interactive (bool): Return an owned terminal handle after approval where qualified.
            wait_seconds (float | None): Initial wait after admission, or None for the configured
                default.
            job (CommandJob | None): Internal owner of a previously reserved command attempt.
            admission_cancellation_event (threading.Event | None): Original run stop signal
                retained only until the detached attempt enters native admission.

        Returns:
            dict | Problem: Final bounded output, an owned continuation, or a typed failure that
                may include a separate host offer.
        """
        with self._backend_lease() as available:
            if not available:
                return Problem(
                    code="sandbox.unavailable",
                    title="Sandbox unavailable",
                    detail="Command execution service is closed.",
                    operation=context.tool_name,
                )
            return self._run_command(
                context,
                command,
                cwd=cwd,
                read_only=read_only,
                read_roots=read_roots,
                write_roots=write_roots,
                executable_grant=executable_grant,
                network=network,
                git_write=git_write,
                reason=reason,
                egress_origins=egress_origins,
                interactive=interactive,
                wait_seconds=wait_seconds,
                job=job,
                admission_cancellation_event=admission_cancellation_event,
            )

    def _run_command(
        self,
        context: ToolContext,
        command: str,
        cwd: str = ".",
        read_only: bool = False,
        read_roots: list[str] | None = None,
        write_roots: list[str] | None = None,
        executable_grant: str | list[str] | None = None,
        network: bool = False,
        git_write: bool = False,
        reason: str | None = None,
        egress_origins: list[str] | None = None,
        interactive: bool = False,
        wait_seconds: float | None = None,
        job: CommandJob | None = None,
        admission_cancellation_event: threading.Event | None = None,
    ) -> dict | Problem:
        """Bind and execute a command while its selected backend lease remains active."""
        try:
            command_deadline(
                context.settings.command_hard_timeout, constants.COMMAND_PREPARATION_SECONDS
            )
        except ValueError:
            return Problem(
                code="command_job.invalid_runtime",
                title="Command runtime limit is not representable",
                detail="Reduce tools.command_hard_timeout to a value supported by this clock.",
                operation=context.tool_name,
            )
        if interactive and SandboxCapability.TERMINAL not in self.backend_capabilities:
            return Problem(
                code="command_terminal.unavailable",
                title="Interactive terminal unavailable",
                detail="This native sandbox has no qualified owned interactive terminal.",
                operation=context.tool_name,
            )
        if interactive and egress_origins:
            return Problem(
                code="command_terminal.unavailable",
                title="Interactive terminal unavailable",
                detail="Mediated egress cannot share the interactive terminal control path.",
                operation=context.tool_name,
            )
        operation = context.operations[0] if context.operations else None
        target = operation.target if operation is not None else None
        if (
            not isinstance(target, ProcessTarget)
            or target.boundary is not ProcessBoundary.SANDBOXED
        ):
            return Problem(
                code="sandbox.unavailable",
                title="Sandbox unavailable",
                detail="Bound sandbox request is missing.",
                operation=context.tool_name,
            )
        if context.permission_manager is None:
            return Problem(
                code="sandbox.unavailable",
                title="Sandbox unavailable",
                detail="Native command authorization is unavailable.",
                operation=context.tool_name,
            )
        if job is None:
            runtime_settings = replace(context.settings)
            if wait_seconds is None:
                wait_seconds = runtime_settings.command_timeout
            if (
                isinstance(wait_seconds, bool)
                or not isinstance(wait_seconds, (int, float))
                or isinstance(wait_seconds, float)
                and not math.isfinite(wait_seconds)
                or wait_seconds < 0
            ):
                return Problem(
                    code="command_job.invalid_wait",
                    title="Invalid command wait",
                    detail="The initial wait must be a finite nonnegative number of seconds.",
                    operation=context.tool_name,
                )
            if context.cancellation_event is not None and context.cancellation_event.is_set():
                return Problem(
                    code="sandbox.cancelled",
                    title="Command cancelled",
                    detail="The originating run was cancelled before command admission.",
                    operation=context.tool_name,
                )
            manager = context.session_manager
            permissions = context.permission_manager
            reserved: CommandJob | Problem | None = None
            authority_lease = None
            retrieving_result = False
            try:
                standalone = (
                    context.standalone_execution or permissions.standalone_execution
                    if manager is None
                    else None
                )
                with (
                    manager.interaction_execution()
                    if manager is not None
                    else standalone() as lease
                ):
                    session_id = manager.session.id if manager is not None else None
                    authorities = (
                        context.command_authorities
                        if context.command_authorities is not None
                        else (permissions, context.instructions_manager)
                    )
                    authority_lease = command_authorities.reserve((self, *authorities))
                    if context.instructions_manager is not None:
                        context = replace(context, command_paths=_command_paths(context))
                    owner = self._job_owner(context)
                    if manager is not None and Path(permissions.temporary_directory).resolve(
                        strict=True
                    ) != Path(manager.temporary_directory).resolve(strict=True):
                        raise ValueError("Permission scratch identity differs from its session.")
                    if manager is None and Path(lease).resolve(strict=True) != Path(
                        owner.rsplit(":", 2)[0]
                    ):
                        raise ValueError("Standalone scratch identity differs from its owner.")

                    def execute(owned: CommandJob) -> dict | Problem:
                        """Run the exact planned command once under a job-owned cancel signal."""
                        with (
                            manager.interaction_execution(session_id)
                            if manager is not None
                            else standalone()
                        ):
                            return self.run_command(
                                replace(
                                    context,
                                    settings=runtime_settings,
                                    cancellation_event=owned.cancellation_signal(
                                        context.cancellation_event
                                    ),
                                ),
                                command,
                                cwd,
                                read_only,
                                read_roots,
                                write_roots,
                                executable_grant,
                                network,
                                git_write,
                                reason,
                                egress_origins,
                                interactive=interactive,
                                job=owned,
                                admission_cancellation_event=context.cancellation_event,
                            )

                    with (
                        context.command_admission()
                        if context.command_admission is not None
                        else nullcontext()
                    ):
                        reserved = self._jobs.start(
                            owner,
                            execute,
                            runtime_limit_seconds=runtime_settings.command_hard_timeout,
                            on_settled=authority_lease.release,
                            **({"terminal": True} if interactive else {}),
                        )
                    if isinstance(reserved, Problem):
                        return reserved
                    reserved.ready.wait()
                    if not interactive:
                        reserved.wait(
                            min(wait_seconds, threading.TIMEOUT_MAX), context.cancellation_event
                        )
                    if (
                        context.cancellation_event is not None
                        and context.cancellation_event.is_set()
                    ):
                        reserved.cancel()
                    retrieving_result = True
                    result = reserved.result() if not interactive else None
                    retrieving_result = False
                    reserved.handoff(context.cancellation_event)
                    if result is not None:
                        return result
                    response = {
                        **reserved.inspect(),
                        "continuation": (
                            "Use read_command_job with this handle and its next cursor."
                        ),
                    }
                    page = reserved.read(response["oldest_cursor"])
                    if isinstance(page, Problem):
                        return page.model_copy(
                            update={"metadata": {**page.metadata, "handle": reserved.handle}}
                        )
                    response.update(page)
                    return response
            except (OSError, ValueError, SessionExecutionConflictError):
                if isinstance(reserved, CommandJob):
                    reserved.cancel()
                    reserved.handoff()
                if retrieving_result:
                    raise
                return Problem(
                    code="command_job.unavailable",
                    title="Command job unavailable",
                    detail="Session scratch ownership is unavailable.",
                    operation=context.tool_name,
                )
            except BaseException:
                if isinstance(reserved, CommandJob):
                    reserved.cancel()
                    reserved.handoff()
                raise
            finally:
                if authority_lease is not None and not isinstance(reserved, CommandJob):
                    authority_lease.release()

        if network and (
            network_limit := context.permission_manager.sandbox_network_limit(context.tool_name)
        ):
            return Problem(
                code="sandbox.unavailable",
                title="Sandbox network grant unavailable",
                detail=network_limit,
                operation=context.tool_name,
            )
        if egress_origins and SandboxCapability.MEDIATED_EGRESS not in self.backend_capabilities:
            return Problem(
                code="sandbox.unavailable",
                title="Mediated egress unavailable",
                detail=(
                    "This platform has no qualified sandboxed HTTP proxy route. Direct network "
                    "remains subject to hard limits. A trusted user may separately change the "
                    "applicable limit or request explicit host execution."
                ),
                operation=context.tool_name,
            )
        if egress_origins and (
            egress_limit := context.permission_manager.sandbox_egress_limit(
                context.tool_name, tuple(egress_origins)
            )
        ):
            return Problem(
                code="sandbox.unavailable",
                title="Mediated egress unavailable",
                detail=egress_limit,
                operation=context.tool_name,
            )
        failure = self._backend.capability_failure()
        prelaunch_failure = (
            failure
            if isinstance(failure, CommandProcessResult)
            else CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail=failure,
                failure_context="Native sandbox capability check failed.",
            )
            if isinstance(failure, str)
            else None
        )
        paths = None
        redact = None
        try:
            command_cwd = Path(target.cwd).resolve(strict=True)
            workspace = (
                Path(_command_paths(context).resolve(VirtualPath.WORKSPACE)).resolve(strict=True)
                if context.instructions_manager is not None
                else command_cwd
            )
            path_value = os.environ.get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH)
            analysis = context.permission_manager.analyze_command(command)
            discovery_source = workspace_command_source(command, workspace, command_cwd)
            discovery_analysis = (
                analysis
                if discovery_source == command
                else context.permission_manager.analyze_command(discovery_source)
            )
            absolute_path = all(Path(entry).is_absolute() for entry in path_value.split(os.pathsep))
            scratch = (
                dict(_command_paths(context).command_roots()).get(
                    VirtualPath.TEMPORARY, context.permission_manager.temporary_directory
                )
                if context.instructions_manager is not None
                else context.permission_manager.temporary_directory
            )
            trusted_system_path = getattr(type(self._backend), "trusted_system_path", None)
            scratch_source = (
                discover_scratch_shell_source(
                    discovery_source,
                    scratch,
                    command_cwd,
                    path_value,
                    lambda path: trusted_system_path(self._backend, path),
                )
                if prelaunch_failure is None and callable(trusted_system_path)
                else None
            )
            if scratch_source is not None:
                scratch_source = workspace_command_source(scratch_source, workspace, command_cwd)
            scratch_analysis = (
                context.permission_manager.analyze_command(scratch_source)
                if scratch_source is not None
                else None
            )
            child_source = (
                timeout_child_source(discovery_source)
                if prelaunch_failure is None
                and discovery_analysis.first_executable_initial
                and discovery_analysis.first_executable_literal
                and discovery_analysis.first_path_inherited
                else None
            )
            child_analysis = (
                context.permission_manager.analyze_command(child_source)
                if child_source is not None
                and _selected_executable(discovery_analysis.executables[0], path_value, command_cwd)
                is not None
                else None
            )
            if child_analysis is None:
                child_source = None
            eligible_executables = tuple(
                dict.fromkeys(
                    name
                    for name, cwd_unchanged in discovery_analysis.inherited_path_executables
                    if cwd_unchanged or absolute_path
                )
            )
            if scratch_analysis is not None:
                combined = tuple(
                    dict.fromkeys(
                        (
                            *eligible_executables,
                            *(
                                name
                                for name, unchanged in scratch_analysis.inherited_path_executables
                                if unchanged or absolute_path
                            ),
                        )
                    )
                )
                if len(combined) <= constants.MAX_COMMAND_PATH_COUNT:
                    eligible_executables = combined
                else:
                    scratch_source = scratch_analysis = None
            make_source = child_source or discovery_source
            make_analysis = child_analysis or discovery_analysis
            make_recipes = (
                discover_makefile_commands(make_source, workspace, command_cwd, path_value)
                if prelaunch_failure is None
                else ()
            )
            selected_make = (Path("/"), Path("/"))
            if make_recipes:
                make_name = make_analysis.executables[0]
                selection = _selected_executable(make_name, path_value, command_cwd)
                if (
                    selection is None
                    or selection[1] != Path("/usr/bin/make").resolve()
                    or not callable(trusted_system_path)
                    or not trusted_system_path(self._backend, selection[0])
                    or not trusted_system_path(self._backend, selection[1])
                ):
                    make_recipes = ()
                else:
                    selected_make = selection
            make_identity = ()
            if make_recipes and "/" not in make_name:
                try:
                    entries = tuple(Path(item) for item in path_value.split(os.pathsep) if item)
                    if (
                        len(entries) != len(path_value.split(os.pathsep))
                        or any(not path.is_absolute() or ".." in path.parts for path in entries)
                        or not trusted_system_path(
                            self._backend, selected_make[0].parent, directory=True
                        )
                    ):
                        make_recipes = ()
                    else:
                        for path in entries:
                            path.resolve()
                        details = selected_make[1].stat()
                        make_identity = (
                            (make_name, *selected_make, details.st_dev, details.st_ino),
                        )
                except (OSError, RuntimeError, ValueError):
                    make_recipes = ()
            make_analyses = tuple(
                context.permission_manager.analyze_command(recipe) for recipe in make_recipes
            )
            make_names = tuple(
                dict.fromkeys(
                    (
                        *(
                            name
                            for item in make_analyses
                            for name, unchanged in item.inherited_path_executables
                            if unchanged or absolute_path
                        ),
                        *(
                            tuple(
                                name
                                for name, unchanged in child_analysis.inherited_path_executables[:1]
                                if unchanged or absolute_path
                            )
                            if child_analysis is not None
                            else ()
                        ),
                    )
                )
            )
            if len(make_names) > constants.MAX_COMMAND_PATH_COUNT:
                make_names = ()
                make_recipes = ()
                prelaunch_failure = CommandProcessResult(
                    SandboxOutcome.UNSUPPORTED,
                    detail=(
                        "The selected build requires more executable identities "
                        "than the sandbox can bind."
                    ),
                )
            if context.instructions_manager is not None:
                roots = dict(_command_paths(context).command_roots())
                paths = VirtualPath(
                    workspace=roots.get(VirtualPath.WORKSPACE),
                    temporary_directory=roots.get(VirtualPath.TEMPORARY),
                    skill_roots={
                        name.removeprefix(f"{VirtualPath.SKILLS}/"): root
                        for name, root in roots.items()
                        if name.startswith(f"{VirtualPath.SKILLS}/")
                    },
                )
            else:
                paths = VirtualPath(
                    workspace=workspace,
                    temporary_directory=context.permission_manager.temporary_directory,
                )
            manual_reads = tuple(
                Path(
                    paths.resolve(root)
                    if any(
                        root == virtual or root.startswith(f"{virtual}/")
                        for virtual, _ in paths.command_roots()
                    )
                    else root
                ).resolve(strict=True)
                for root in read_roots or ()
            )
            manual_writes = tuple(
                dict.fromkeys(
                    Path(
                        paths.resolve(root)
                        if any(
                            root == virtual or root.startswith(f"{virtual}/")
                            for virtual, _ in paths.command_roots()
                        )
                        else root
                    ).resolve(strict=True)
                    for root in write_roots or ()
                )
            )
            if read_only and manual_writes:
                raise ValueError("Read-only workspace commands cannot request write roots.")
            if write_limit := context.permission_manager.sandbox_write_root_limit(
                workspace, manual_writes
            ):
                raise ValueError(write_limit)
            system_make_names = []
            selected_system_roots = getattr(self._backend, "system_read_roots", ())
            if not isinstance(selected_system_roots, tuple):
                selected_system_roots = ()
            if callable(trusted_system_path):
                for name in make_names:
                    selection = _selected_executable(name, path_value, command_cwd)
                    if selection is not None:
                        try:
                            trusted = all(
                                trusted_system_path(self._backend, path) for path in selection
                            )
                        except (OSError, ValueError):
                            trusted = False
                        if trusted:
                            candidate = Path(name).name if "/" in name else name
                            path_selection = _selected_executable(
                                candidate, path_value, command_cwd
                            )
                            same_spelling = "/" not in name or (
                                path_selection is not None
                                and path_selection[0].name == selection[0].name
                                and path_selection[0].parent.resolve()
                                == selection[0].parent.resolve()
                            )
                            if (
                                path_selection is not None
                                and path_selection[1] == selection[1]
                                and same_spelling
                                and all(
                                    any(path.is_relative_to(root) for root in selected_system_roots)
                                    for path in selection
                                )
                            ):
                                system_make_names.append(candidate)
                            elif "/" in name:
                                checker = getattr(
                                    type(self._backend), "system_executable_requires_alias", None
                                )
                                try:
                                    requires_alias = callable(checker) and checker(
                                        self._backend, *selection, workspace, manual_reads
                                    )
                                except (OSError, ValueError):
                                    requires_alias = True
                                if requires_alias:
                                    prelaunch_failure = CommandProcessResult(
                                        SandboxOutcome.UNSUPPORTED,
                                        detail=(
                                            "Selected system executable alias cannot be bound "
                                            "to this PATH."
                                        ),
                                    )
            if (
                len(dict.fromkeys((*eligible_executables, *system_make_names)))
                > constants.MAX_COMMAND_PATH_COUNT
            ):
                make_names = ()
                make_recipes = ()
                system_make_names = []
                prelaunch_failure = CommandProcessResult(
                    SandboxOutcome.UNSUPPORTED,
                    detail=(
                        "The selected build requires more executable identities "
                        "than the sandbox can bind."
                    ),
                )
            findings = analysis.findings
            git_write = git_write or any(finding.requests_git_write for finding in findings)
            if read_only and git_write:
                raise ValueError("A read-only command cannot request Git metadata writes.")
            git_reads, git_roots, git_create = (
                context.permission_manager.git_sandbox_roots(
                    workspace, read=analysis.git_read, write=git_write
                )
                if git_write or analysis.git_read
                else ((), (), False)
            )
            if isinstance(executable_grant, str):
                explicit_references = (executable_grant,) if executable_grant else ()
            else:
                explicit_references = tuple(executable_grant or ())
            explicit_grants = {}
            for reference in dict.fromkeys(explicit_references):
                grant = self._executable_grants.get(reference)
                if grant is None:
                    raise ValueError("Executable grant reference is unknown or expired.")
                explicit_grants[reference] = grant
            direct_grants = self._automatic_tool_grants(
                context, eligible_executables, workspace, command_cwd
            )
            make_grants = []
            for name in make_names:
                if "/" in name:
                    continue
                try:
                    match = resolve_host_executable(name, path_value, command_cwd)
                except ValueError:
                    continue
                if match is None:
                    continue
                _, grant = self._discover_executable(
                    context,
                    name,
                    path_value,
                    command_cwd,
                    workspace if context.instructions_manager is not None else None,
                    match=match,
                    optional_metadata=True,
                )
                if grant is not None:
                    make_grants.append(grant)
            grants = tuple(
                dict.fromkeys(
                    (
                        *explicit_grants.values(),
                        *direct_grants,
                        *make_grants,
                    )
                )
            )
            grant_reads = tuple(
                dict.fromkeys(
                    root
                    for grant in grants
                    for root in self._executable_read_roots(
                        grant, workspace, path_value, command_cwd
                    )
                )
            )
            automatic_reads = tuple(
                dict.fromkeys(
                    root
                    for grant in grants
                    if grant.managed_package
                    and (grant in direct_grants or grant in explicit_grants.values())
                    for root in self._executable_read_roots(
                        grant, workspace, path_value, command_cwd
                    )
                )
            )
            read_aliases = tuple(
                dict.fromkeys(alias[0] for grant in grants for alias in grant.opt_aliases)
            )
            trusted_owners = getattr(self._backend, "trusted_file_owner_ids", None)
            if not isinstance(trusted_owners, tuple):
                trusted_owners = (os.getuid(),)
            executable_identities = tuple(
                dict.fromkeys(
                    (
                        *(
                            (
                                grant.name,
                                grant.spelling,
                                grant.resolved,
                                *grant.identity,
                            )
                            for grant in grants
                        ),
                        *make_identity,
                        *_automatic_executable_identities(
                            (*eligible_executables, *system_make_names),
                            path_value,
                            command_cwd,
                            trusted_owners=trusted_owners,
                        ),
                    )
                )
            )
            runtime_candidates = discover_workspace_runtimes(
                discovery_source,
                discovery_analysis,
                workspace,
                command_cwd,
                path_value,
                executable_identities,
            )
            if child_source is not None and child_analysis is not None:
                runtime_candidates += discover_workspace_runtimes(
                    child_source,
                    child_analysis,
                    workspace,
                    command_cwd,
                    path_value,
                    executable_identities,
                )
            if scratch_source is not None and scratch_analysis is not None:
                runtime_candidates += discover_workspace_runtimes(
                    scratch_source,
                    scratch_analysis,
                    workspace,
                    command_cwd,
                    path_value,
                    executable_identities,
                )
            for recipe in make_recipes:
                candidate_analysis = context.permission_manager.analyze_command(recipe)
                runtime_candidates += discover_workspace_runtimes(
                    recipe,
                    candidate_analysis,
                    workspace,
                    command_cwd,
                    path_value,
                    executable_identities,
                )
            selected_workspace = selected_workspace_executable(
                discovery_source, discovery_analysis, workspace
            )
            direct_names = (
                (selected_workspace,)
                if selected_workspace is not None
                else discovery_analysis.executables[:1]
                if discovery_analysis.first_executable_literal
                and discovery_analysis.first_executable_initial
                and (
                    "/" in discovery_analysis.executables[0]
                    or discovery_analysis.first_path_inherited
                )
                and (
                    len(discovery_analysis.executables) == 1
                    or "/" in discovery_analysis.executables[0]
                )
                else ()
            )
            script_discovery = (
                _workspace_script_interpreters(
                    direct_names,
                    path_value,
                    command_cwd,
                    workspace,
                    allow_path_lookup=discovery_analysis.first_path_inherited,
                    first_arguments=discovery_analysis.first_executable_arguments
                    or (
                        (discovery_analysis.first_executable_argument,)
                        if discovery_analysis.first_executable_argument is not None
                        else ()
                    ),
                )
                if direct_names
                else WorkspaceScriptInterpreters((), False, ())
            )
            recipe_names = tuple(
                dict.fromkeys(
                    item.executables[0]
                    for item in make_analyses
                    if item.executables
                    and item.first_executable_initial
                    and item.first_executable_literal
                    and item.first_path_inherited
                )
            )
            if recipe_names:
                recipe_script = _workspace_script_interpreters(
                    recipe_names,
                    path_value,
                    command_cwd,
                    workspace,
                )
                script_discovery = WorkspaceScriptInterpreters(
                    tuple(
                        dict.fromkeys((*script_discovery.interpreters, *recipe_script.interpreters))
                    ),
                    script_discovery.followed_argument,
                    tuple(
                        dict.fromkeys(
                            (*script_discovery.workspace_links, *recipe_script.workspace_links)
                        )
                    ),
                )
            workspace_links = script_discovery.workspace_links
            runtime_candidates = tuple(
                dict.fromkeys(
                    (
                        *runtime_candidates,
                        *discover_workspace_interpreters(
                            workspace, script_discovery.workspace_links
                        ),
                    )
                )
            )
            runtime_reader = getattr(type(self._backend), "automatic_tool_read_roots", None)
            path_roots = path_search_roots({PATH_ENVIRONMENT_VARIABLE: path_value}, command_cwd)
            runtime_reads = (
                runtime_reader(self._backend, workspace, path_roots, executable_identities)
                if runtime_reader is not None
                else ()
            )
            runtime_candidates = tuple(
                candidate
                for candidate in runtime_candidates
                if not any(candidate.identity[2].is_relative_to(root) for root in runtime_reads)
            )
            candidate_reads = ()
            candidate_rejected = False
            if runtime_reader is not None and runtime_candidates:
                try:
                    candidate_reads = runtime_reader(
                        self._backend,
                        workspace,
                        tuple(
                            (item.identity[1].parent, item.identity[1].parent.resolve())
                            for item in runtime_candidates
                        ),
                        tuple(item.identity for item in runtime_candidates),
                    )
                except TimeoutError:
                    raise
                except (OSError, ValueError):
                    candidate_reads = ()
                    candidate_rejected = True
                original_candidate_reads = candidate_reads
                candidate_reads = tuple(
                    root
                    for root in candidate_reads
                    if not root.is_relative_to(workspace)
                    and any(
                        candidate.identity[2].is_relative_to(root)
                        for candidate in runtime_candidates
                    )
                )
                if original_candidate_reads and not candidate_reads:
                    candidate_rejected = True
                runtime_reads = tuple(dict.fromkeys((*runtime_reads, *candidate_reads)))
            runtime_reads = tuple(
                root for root in runtime_reads if not root.is_relative_to(workspace)
            )
            system_roots = getattr(self._backend, "system_read_roots", ())
            if not isinstance(system_roots, tuple):
                system_roots = ()
            workspace_link_names = tuple(item.name for item in workspace_links)
            selected_runtimes = (
                *(() if manual_reads else workspace_link_names),
                *script_discovery.interpreters,
                *(
                    tuple(str(candidate.alias_executable) for candidate in runtime_candidates)
                    if not candidate_reads and not candidate_rejected
                    else ()
                ),
            )
            identified_reads = (
                _interpreter_runtime_roots(
                    selected_runtimes,
                    path_value,
                    command_cwd,
                    workspace,
                    (*system_roots, *manual_reads, *grant_reads, *runtime_reads),
                    runtime_reads,
                )
                if selected_runtimes
                else ()
            )
            active_candidates = tuple(
                candidate
                for candidate in runtime_candidates
                if any(
                    candidate.identity[2].is_relative_to(root)
                    for root in (*candidate_reads, *identified_reads, *manual_reads)
                )
            )
            package_reads, package_aliases = _installed_runtime_packages(
                self._backend,
                active_candidates,
                workspace,
                (*manual_reads, *grant_reads, *runtime_reads, *identified_reads, *system_roots),
            )
            workspace_links = tuple(
                dict.fromkeys(
                    (
                        *workspace_links,
                        *(WorkspaceExecutableLink(*item.identity) for item in active_candidates),
                    )
                )
            )
            workspace_link_names = tuple(item.name for item in workspace_links)
            alias_roots = tuple(
                root
                for root in (*manual_reads, *grant_reads, *identified_reads)
                if not root.is_relative_to(workspace)
                and not any(
                    root.is_relative_to(runtime) or runtime.is_relative_to(root)
                    for runtime in runtime_reads
                )
            )
            if not alias_roots:
                script_interpreters = ()
            elif len(analysis.executables) == 1 and (
                not analysis.path_arguments
                or script_discovery.followed_argument
                and analysis.path_arguments == (analysis.first_executable_argument,)
            ):
                script_interpreters = script_discovery.interpreters
            else:
                script_interpreters = _workspace_script_interpreters(
                    (
                        *analysis.executables,
                        *(word if "/" in word else f"./{word}" for word in analysis.path_arguments),
                    ),
                    path_value,
                    command_cwd,
                    workspace,
                    allow_path_lookup=analysis.first_path_inherited,
                ).interpreters
            selected_alias_names = (
                *workspace_link_names,
                *eligible_executables,
                *(str(candidate.alias_executable) for candidate in active_candidates),
            )
            read_aliases = tuple(
                dict.fromkeys(
                    (
                        *read_aliases,
                        *package_aliases,
                        *_approved_executable_aliases(
                            tuple(
                                str(candidate.alias_executable) for candidate in runtime_candidates
                            ),
                            path_value,
                            command_cwd,
                            candidate_reads,
                            workspace,
                            system_roots,
                        ),
                        *_approved_executable_aliases(
                            selected_alias_names,
                            path_value,
                            command_cwd,
                            alias_roots,
                            workspace,
                            system_roots,
                        ),
                        *_approved_executable_aliases(
                            tuple(
                                name
                                for name in script_interpreters
                                if name not in selected_alias_names
                            ),
                            path_value,
                            command_cwd,
                            alias_roots,
                            workspace,
                            system_roots,
                        ),
                        *_approved_executable_aliases(
                            tuple(
                                grant.name
                                for grant in grants
                                if grant.name not in eligible_executables
                            ),
                            path_value,
                            command_cwd,
                            alias_roots,
                            workspace,
                            system_roots,
                        ),
                        *_approved_executable_aliases(
                            _indirect_approved_executable_names(
                                path_value, command_cwd, alias_roots, system_roots
                            ),
                            path_value,
                            command_cwd,
                            alias_roots,
                            workspace,
                            system_roots,
                        ),
                    )
                )
            )
            used_skills = tuple(
                (name, root)
                for name, root in paths.command_roots()
                if name.startswith(f"{VirtualPath.SKILLS}/")
                and (
                    str(root) in command
                    or any(
                        requested == name or requested.startswith(f"{name}/")
                        for requested in (*(read_roots or ()), *(write_roots or ()))
                    )
                )
            )
            skill_snapshot = ()
            if used_skills and context.instructions_manager is not None:
                skill_snapshot = context.instructions_manager.command_skill_snapshot(
                    tuple(
                        (name.removeprefix(f"{VirtualPath.SKILLS}/"), root)
                        for name, root in used_skills
                    )
                )
                if skill_snapshot is None:
                    raise ValueError("Command skill roots changed before execution.")
            skill_reads = tuple(root for _, root in used_skills if str(root) in command)
            extra_reads = tuple(
                root
                for root in dict.fromkeys(
                    (
                        *grant_reads,
                        *manual_reads,
                        *runtime_reads,
                        *identified_reads,
                        *package_reads,
                        *skill_reads,
                    )
                )
                if not root.is_relative_to(workspace)
            )
            if any(is_protected_external_read(root) for root in extra_reads):
                raise ValueError("Protected external directory cannot be a read root.")
            read_context = context.permission_manager.describe_sandbox_reads(
                workspace,
                extra_reads,
                tuple(root for root in manual_reads if not root.is_relative_to(workspace)),
                tuple(grant.name for grant in grants),
                bool(grant_reads) and set(grant_reads) <= set(automatic_reads),
            )
            if identified_reads:
                read_context = (
                    f"{read_context}; identified executable folder, including neighboring files"
                )
            if make_grants:
                detail = "identified indirect tool folder, including neighboring files"
                read_context = f"{read_context}; {detail}" if read_context else f"Read {detail}"
            if candidate_reads:
                detail = "selected project runtime folder, including neighboring files"
                read_context = f"{read_context}; {detail}" if read_context else f"Read {detail}"
            if package_reads:
                detail = "selected runtime packages and their declared dependencies"
                read_context = f"{read_context}; {detail}" if read_context else f"Read {detail}"
            selected_identities = tuple(
                tuple(identity)
                for identity in workspace_links
                if any(identity.resolved.is_relative_to(root) for root in extra_reads)
            )
            scratch = dict(paths.command_roots()).get(
                VirtualPath.TEMPORARY, context.permission_manager.temporary_directory
            )
            environment = {
                PATH_ENVIRONMENT_VARIABLE: path_value,
                LANG_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                WORKSPACE_ROOT_ENVIRONMENT_VARIABLE: str(workspace),
                TMPDIR_ENVIRONMENT_VARIABLE: str(scratch),
                TMP_ENVIRONMENT_VARIABLE: str(scratch),
                TEMP_ENVIRONMENT_VARIABLE: str(scratch),
                XDG_CACHE_HOME_ENVIRONMENT_VARIABLE: str(scratch / "cache"),
                HOME_ENVIRONMENT_VARIABLE: str(scratch),
            }
            generated_environment = tuple(
                name
                for name, value in environment.items()
                if name != WORKSPACE_ROOT_ENVIRONMENT_VARIABLE
                and (value == str(scratch) or value.startswith(f"{scratch}/"))
            )

            redaction_roots = {*extra_reads, *read_aliases, *git_reads}
            advertised_roots = {root for _, root in paths.command_roots()}
            native_output_roots = ()

            def render_command_output(value: str) -> str:
                """Render canonical path prefixes while hiding unrelated host paths."""
                value = paths.redact(value, roots=native_output_roots)
                for root in sorted(redaction_roots, key=lambda item: len(str(item)), reverse=True):
                    if (
                        root in advertised_roots
                        or root.is_relative_to(workspace)
                        or any(root.is_relative_to(system) for system in system_roots)
                    ):
                        continue
                    value = paths.redact(value, roots=((str(root), VirtualPath.EXTERNAL),))
                return paths.redact(value)

            redact = render_command_output

            file_creation_mask = constants.GROUP_AND_OTHER_WRITE_MODE_MASK
            current_mask = getattr(self._backend, "file_creation_mask", None)
            if isinstance(current_mask, int) and not isinstance(current_mask, bool):
                file_creation_mask = current_mask
            if any(
                self._executable_grants.get(reference) is not grant
                for reference, grant in explicit_grants.items()
            ):
                raise ValueError("Executable grant reference is unknown or expired.")

            request = SandboxRequest.create(
                source=command,
                terminal=interactive,
                cleanup_proof=True,
                scratch_root=scratch,
                scratch_data=scratch == context.permission_manager.temporary_directory
                and context.permission_manager.temporary_directory_allows_private_names,
                cwd=command_cwd,
                workspace=workspace,
                read_roots=(*extra_reads, *git_reads),
                write_roots=()
                if read_only
                else tuple(dict.fromkeys((workspace, *git_roots, *manual_writes))),
                network=network,
                egress_origins=tuple(egress_origins or ()),
                local_network=network
                or (
                    SandboxCapability.LOCAL_NETWORK in self.backend_capabilities
                    and context.permission_manager.sandbox_local_network_enabled()
                ),
                environment=environment,
                policy_version=self._backend.policy_version
                if isinstance(getattr(self._backend, "policy_version", None), str)
                else "native-v1",
                deadline=time.monotonic() + context.settings.command_hard_timeout,
                workspace_id=str(workspace),
                git_create=git_create,
                read_context=read_context,
                read_aliases=read_aliases,
                executable_identities=executable_identities,
                workspace_executable_identities=selected_identities,
                automatic_tool_reads=automatic_reads,
                protected_instruction_names=context.permission_manager.protected_sandbox_paths(
                    context.instructions_manager.agents_filenames
                    if context.instructions_manager is not None
                    else ()
                ).instruction_names,
                generated_environment=generated_environment,
                file_creation_mask=file_creation_mask,
            )
            output_paths = getattr(type(self._backend), "output_path_roots", None)
            if output_paths is not None:
                native_output_roots = output_paths(self._backend, request)
        except (OSError, ValueError) as exc:
            return Problem(
                code="sandbox.unavailable",
                title="Sandbox unavailable",
                detail=redact(str(exc))
                if redact
                else (paths.redact(str(exc)) if paths else "Command preparation failed."),
                operation=context.tool_name,
            )
        else:
            scope_limit = context.permission_manager.sandbox_scope_limit(request)
            if (
                scope_limit
                and context.interaction is not None
                and context.permission_manager.review_legacy_sandbox_read_limit(
                    request, context.interaction
                )
            ):
                scope_limit = context.permission_manager.sandbox_scope_limit(request)
            if scope_limit:
                return Problem(
                    code="sandbox.unavailable",
                    title="Sandbox scope denied by policy limit",
                    detail=redact(scope_limit),
                    operation=context.tool_name,
                )
            return self._run_command_attempt(
                context,
                request,
                redact,
                paths.display if context.instructions_manager is not None else str,
                prelaunch_failure=prelaunch_failure,
                reason=reason,
                job=job,
                skill_snapshot=skill_snapshot,
                admission_cancellation_event=admission_cancellation_event,
            )
