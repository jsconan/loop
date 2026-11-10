"""Define authority and outcomes shared by native command sandbox backends."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from ... import constants
from ...permissions.protection import ProtectedWorkspacePaths, protected_workspace_paths
from ...utils.hashing import sha256_digest
from ...utils.json import load_unique_json
from ...utils.process import is_finite_deadline
from .constants import MINIMAL_SYSTEM_PATH, PATH_ENVIRONMENT_VARIABLE, TMPDIR_ENVIRONMENT_VARIABLE

if TYPE_CHECKING:
    from ...permissions.manager import PermissionManager


class SandboxCapability(StrEnum):
    """Name a qualified client-facing native sandbox feature."""

    TERMINAL = "terminal"
    MEDIATED_EGRESS = "mediated_egress"
    INNER_SECCOMP = "inner_seccomp"
    LOCAL_NETWORK = "local_network"


class QualificationProbeLayout(NamedTuple):
    """Place a native enforcement probe and its independent outside canary.

    Args:
        probe_parent (Path | None): Parent for the temporary probe workspace, or the system
            temporary directory when None.
        outside_parent (Path | None): Independent parent for the outside canary, or None to
            place the canary beside the workspace within the probe root.
    """

    probe_parent: Path | None
    outside_parent: Path | None


# These literals describe the retained receiver's stable wire contract. Additive protocol
# changes must retain these meanings so newer clients can choose a supported request before launch.
NATIVE_REQUEST_SCHEMAS = (1, 2)
NATIVE_REQUEST_FEATURES = (
    "workspace-executable-identities-v1",
    "system-executable-alias-v1",
    "mediated-egress-v1",
    "owned-terminal-v1",
    "sandbox-local-network-v1",
    "live-output-beyond-capture-v1",
)
NATIVE_RESULT_SCHEMAS = (1,)
# The client may decode more result schemas than this retained receiver can emit.
NATIVE_ACCEPTED_RESULT_SCHEMAS = (1, 2)
NATIVE_POLICY_CAPABILITIES = ("linux-native-v1", "unix-host-isolation-v1")
NATIVE_REQUEST_OPTIONAL_FIELDS = (
    "workspace_executable_identities",
    "egress_origins",
    "stream_output",
    "runtime_timeout",
    "cleanup_proof",
    "terminal",
    "local_network",
)
NATIVE_REQUEST_SCHEMA2_ABSENT_FIELDS = ("execution_source",)
NATIVE_REQUEST_FIELDS = (
    "source",
    "cwd",
    "workspace",
    "read_roots",
    "write_roots",
    "network",
    "environment",
    "policy_version",
    "deadline",
    "workspace_id",
    "_identities",
    "path_roots",
    "execution_source",
    "git_create",
    "attempt_id",
    "read_context",
    "read_aliases",
    "_read_alias_identities",
    "executable_identities",
    "workspace_executable_identities",
    "automatic_tool_reads",
    "generated_environment",
    "protected_instruction_names",
    "file_creation_mask",
    "scratch_root",
    "scratch_data",
    "egress_origins",
    "stream_output",
    "runtime_timeout",
    "cleanup_proof",
    "terminal",
    "local_network",
)


def normalize_egress_origin(value: str) -> str:
    """Return one unambiguous HTTP or HTTPS origin for mediated egress.

    Args:
        value (str): Absolute origin with an optional explicit TCP port.

    Returns:
        str: Lowercase ASCII origin with a canonical IP address or DNS name.

    Raises:
        ValueError: The spelling is ambiguous, contains request data, or is unsupported.
    """
    if not isinstance(value, str) or not value.isascii() or len(value) > 255:
        raise ValueError("Mediated egress requires a bounded ASCII HTTP(S) origin.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        host = parsed.hostname
    except ValueError as error:
        raise ValueError("Invalid mediated egress origin.") from error
    if (
        parsed.scheme not in {"http", "https"}
        or host is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or not parsed.netloc
        or value != value.strip()
        or "%" in host
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("Mediated egress requires an HTTP(S) origin without credentials or path.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if (
            host.endswith(".")
            or len(host) > 253
            or any(
                not 1 <= len(label) <= 63
                or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label) is None
                for label in host.split(".")
            )
            or re.fullmatch(r"[0-9.]+", host) is not None
        ):
            raise ValueError("Mediated egress host is ambiguous.") from None
        normalized = host.lower()
    else:
        normalized = f"[{address.compressed}]" if address.version == 6 else str(address)
    expected = f"{parsed.scheme}://{normalized}"
    if port is not None and port != (80 if parsed.scheme == "http" else 443):
        expected += f":{port}"
    if value.lower() != expected:
        raise ValueError("Mediated egress origin must use its canonical spelling.")
    return expected


def _generated_environment_values(
    environment: dict[str, str], names: tuple[str, ...]
) -> tuple[tuple[str, str, str], ...] | None:
    """Validate generated scratch paths and retain their exact relative spellings."""
    if not names:
        return ()
    scratch = environment.get(TMPDIR_ENVIRONMENT_VARIABLE)
    if scratch is None:
        return None
    scratch_path = Path(scratch)
    result = []
    for name in names:
        value = environment.get(name)
        if value is None:
            return None
        path = Path(value)
        if ".." in path.parts or not path.is_relative_to(scratch_path):
            return None
        try:
            if not path.resolve(strict=False).is_relative_to(scratch_path):
                return None
        except (OSError, RuntimeError):
            return None
        canonical_scratch = str(scratch_path)
        if value == canonical_scratch or value.startswith(f"{canonical_scratch}/"):
            result.append((name, "relative", value[len(canonical_scratch) :]))
        else:
            result.append((name, "literal", value))
    return tuple(result)


def _wire_text(value: object) -> str:
    """Reject nontext and embedded NUL before restoring any request field."""
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("Invalid sandbox request text field.")
    return value


def _wire_items(value: list, shape: tuple[type, ...]) -> tuple[tuple, ...]:
    """Restore fixed-width text/path/integer rows from the request schema."""
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != len(shape):
            raise ValueError("Invalid sandbox request row.")
        restored = []
        for item, kind in zip(row, shape, strict=True):
            if kind in (str, Path):
                text = _wire_text(item)
                restored.append(Path(text) if kind is Path else text)
            else:
                if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                    raise ValueError("Invalid sandbox request inode identity.")
                restored.append(item)
        result.append(tuple(restored))
    return tuple(result)


class SandboxRequestExpired(TimeoutError):
    """Retain validated request identity for bounded expiry-result delivery.

    Args:
        request (SandboxRequest): Fully restored original request; grants no execution time.
    """

    request: SandboxRequest

    def __init__(self, request: SandboxRequest):
        super().__init__("Sandbox request restoration deadline expired.")
        self.request = request


class SandboxOutcome(StrEnum):
    """Classify a command result; unsupported capabilities are client-local, not wire outcomes."""

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    DENIED = "denied"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    STALE = "stale"
    INVALID = "invalid"


@dataclass(frozen=True)
class DirectoryIdentity:
    """Bind a directory pathname to its device and inode at authorization time."""

    path: Path
    device: int
    inode: int

    @classmethod
    def capture(cls, path: Path) -> DirectoryIdentity:
        """Capture an absolute, real directory without following its final component."""
        if not path.is_absolute():
            raise ValueError(f"Sandbox directory must be absolute: {path}")
        if path.resolve(strict=True) != path:
            raise ValueError(f"Sandbox directory must use its canonical path: {path}")
        details = path.lstat()
        if not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"Sandbox directory is not a real directory: {path}")
        return cls(path, details.st_dev, details.st_ino)

    def is_current(self) -> bool:
        """Check that the approved pathname still names the same real directory."""
        try:
            details = self.path.lstat()
            canonical = self.path.resolve(strict=True)
        except OSError:
            return False
        return (
            canonical == self.path
            and stat.S_ISDIR(details.st_mode)
            and (details.st_dev, details.st_ino) == (self.device, self.inode)
        )


def path_search_roots(environment: dict[str, str], cwd: Path) -> tuple[tuple[Path, Path], ...]:
    """Bind existing PATH entries to their search directories.

    Args:
        environment (dict[str, str]): Sanitized process environment containing PATH.
        cwd (Path): Absolute working directory used to resolve relative PATH entries.

    Returns:
        tuple[tuple[Path, Path], ...]: Ordered PATH spellings and canonical directories.

    Raises:
        ValueError: If cwd is not absolute.
    """
    if not cwd.is_absolute():
        raise ValueError("PATH lookup requires an absolute working directory.")
    path_value = environment.get(PATH_ENVIRONMENT_VARIABLE)
    if path_value is None:
        return ()
    roots = []
    for item in path_value.split(os.pathsep):
        path = Path(item) if item else cwd
        if not path.is_absolute():
            path = cwd / path
            try:
                path = path.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
        try:
            if path.is_dir():
                roots.append((path, path.resolve(strict=True)))
        except (OSError, RuntimeError):
            continue
    return tuple(dict.fromkeys(roots))


def resolve_host_executable(
    name: str,
    path_value: str,
    cwd: Path | None = None,
) -> tuple[Path, Path] | None:
    """Find one named executable using the same bound PATH directories as a command.

    Args:
        name (str): Simple executable name without a path separator or shell syntax.
        path_value (str): Host PATH snapshot that a command would inherit.
        cwd (Path | None): Absolute lookup working directory; defaults to the current directory.

    Returns:
        tuple[Path, Path] | None: First executable PATH spelling and canonical file, or None.

    Raises:
        ValueError: If the name, PATH, or working directory cannot be safely searched.
    """
    if (
        len(name) > constants.MAX_COMMAND_PATH_LENGTH
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", name) is None
    ):
        raise ValueError(
            "Executable lookup requires a simple name of at most "
            f"{constants.MAX_COMMAND_PATH_LENGTH} characters."
        )
    if (
        len(path_value) > constants.MAX_PATH_LENGTH
        or len(path_value.split(os.pathsep)) > constants.MAX_COMMAND_PATH_COUNT
    ):
        raise ValueError("PATH is too large for bounded executable lookup.")
    try:
        working_directory = (cwd or Path.cwd()).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Executable lookup working directory is unavailable.") from exc
    if not working_directory.is_dir():
        raise ValueError("Executable lookup working directory must be a directory.")
    for spelling, _ in path_search_roots(
        {PATH_ENVIRONMENT_VARIABLE: path_value}, working_directory
    ):
        candidate = spelling / name
        try:
            resolved = candidate.resolve(strict=True)
            details = resolved.stat()
        except (OSError, RuntimeError):
            continue
        if stat.S_ISREG(details.st_mode) and os.access(candidate, os.X_OK):
            return candidate, resolved
    return None


def resolve_selected_executable(name: str, path_value: str, cwd: Path) -> tuple[Path, Path] | None:
    """Resolve a literal executable spelling without running it.

    Args:
        name (str): Literal executable name or path from the selected command.
        path_value (str): Bound PATH used for simple names.
        cwd (Path): Absolute working directory for relative spellings.

    Returns:
        tuple[Path, Path] | None: Selected spelling and canonical file, or None when unavailable.

    Raises:
        ValueError: A simple name or PATH cannot be safely searched.
        OSError: The selected path cannot be resolved.
        RuntimeError: Path resolution encounters a symbolic-link loop.
    """
    if "/" not in name:
        return resolve_host_executable(name, path_value, cwd)
    if len(name) > constants.MAX_PATH_LENGTH:
        return None
    spelling = Path(name)
    spelling = spelling if spelling.is_absolute() else cwd / spelling
    if not os.access(spelling, os.X_OK):
        return None
    resolved = spelling.resolve(strict=True)
    return (spelling, resolved) if resolved.is_file() and os.access(spelling, os.X_OK) else None


@dataclass(frozen=True)
class SandboxRequest:
    """Bind an approved shell command to its OS authority and path identities.

    Args:
        source (str): Opaque POSIX shell source passed to the backend.
        cwd (Path): Real host working directory.
        workspace (Path): Real host workspace root.
        read_roots (tuple[Path, ...]): Additional approved readable directory roots.
        write_roots (tuple[Path, ...]): Approved writable directory roots.
        network (bool): Whether general network egress was explicitly granted.
        environment (tuple[tuple[str, str], ...]): Sanitized, ordered effective environment,
            including WORKSPACE_ROOT and the live private temporary environment when facade-owned.
        policy_version (str): Reviewed policy revision used for this approval.
        deadline (float): Absolute monotonic deadline for this attempt.
        workspace_id (str): Stable identity of the selected workspace.
        path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and canonical search roots.
        execution_source (str): Exact execution source, including internal search wrappers.
        git_create (bool): Whether this approval permits creating absent top-level Git metadata.
        read_context (str | None): Plain-language description of additional read authority.
        read_aliases (tuple[Path, ...]): Bound directory symlinks into approved read roots.
        executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Named host
            executables, their PATH spellings, canonical files, and device/inode identities.
        workspace_executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Selected
            workspace spellings and their external executable identities requiring revalidation.
        automatic_tool_reads (tuple[Path, ...]): Read roots derived from bound executable and
            runtime references that may run within the default sandbox without an extra prompt.
        generated_environment (tuple[str, ...]): Names of runtime-generated paths resolving
            within the bound private scratch directory.
        protected_instruction_names (tuple[str, ...]): Instruction filenames denied native
            writes at every depth of the workspace.
        file_creation_mask (int): Caller process umask applied before command source starts.
        attempt_id (str): Opaque identifier correlating this attempt's audit records.
        scratch_root (Path | None): Explicit session-owned physical root matching TMPDIR.
        scratch_data (bool): Permit ordinary names only in a fresh agent-data generation.
        egress_origins (tuple[str, ...]): Canonical HTTP(S) origins for one mediated attempt.
        stream_output (bool): Whether authenticated native transport may send bounded live output.
        runtime_timeout (float | None): Command runtime budget renewed after native launch; None
            retains the original single absolute deadline for direct backend callers.
        cleanup_proof (bool): Request an authenticated receiver verdict after its concrete
            descendant-membership check.
        terminal (bool): Request one receiver-owned pseudo-terminal for the command's standard
            streams; ordinary commands keep separate pipes.
        local_network (bool): Permit attempt-local loopback when the platform qualifies it.
    """

    source: str
    cwd: Path
    workspace: Path
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]
    network: bool
    environment: tuple[tuple[str, str], ...]
    policy_version: str
    deadline: float
    workspace_id: str
    _identities: tuple[DirectoryIdentity, ...]
    path_roots: tuple[tuple[Path, Path], ...]
    execution_source: str
    git_create: bool
    attempt_id: str
    read_context: str | None = None
    read_aliases: tuple[Path, ...] = ()
    _read_alias_identities: tuple[tuple[Path, int, int, Path], ...] = ()
    executable_identities: tuple[tuple[str, Path, Path, int, int], ...] = ()
    workspace_executable_identities: tuple[tuple[str, Path, Path, int, int], ...] = ()
    automatic_tool_reads: tuple[Path, ...] = ()
    generated_environment: tuple[str, ...] = ()
    protected_instruction_names: tuple[str, ...] = ()
    file_creation_mask: int = constants.GROUP_AND_OTHER_WRITE_MODE_MASK
    scratch_root: Path | None = None
    scratch_data: bool = False
    egress_origins: tuple[str, ...] = ()
    stream_output: bool = False
    runtime_timeout: float | None = None
    cleanup_proof: bool = False
    terminal: bool = False
    local_network: bool = True

    @property
    def read_alias_targets(self) -> tuple[Path, ...]:
        """Return the canonical directory targets bound when read aliases were approved.

        Returns:
            tuple[Path, ...]: Bound targets in the same order as the read aliases. Call
                ``paths_are_current`` to verify that their identities remain valid.
        """
        return tuple(target for _, _, _, target in self._read_alias_identities)

    @property
    def session_scratch(self) -> bool:
        """Return whether TMPDIR is bound to the file tools' session temporary root.

        Returns:
            bool: Whether an explicit session root binds scratch.
        """
        return self.scratch_root is not None

    @property
    def protected_paths(self) -> ProtectedWorkspacePaths:
        """Return the permission-owned native write-protection specification.

        Returns:
            ProtectedWorkspacePaths: Active instruction and control path names.
        """
        return protected_workspace_paths(self.protected_instruction_names)

    @classmethod
    def create(
        cls,
        *,
        source: str,
        cwd: Path,
        workspace: Path,
        read_roots: tuple[Path, ...],
        write_roots: tuple[Path, ...],
        network: bool,
        environment: dict[str, str],
        policy_version: str,
        deadline: float,
        workspace_id: str,
        execution_source: str | None = None,
        git_create: bool = False,
        read_context: str | None = None,
        read_aliases: tuple[Path, ...] = (),
        executable_identities: tuple[tuple[str, Path, Path, int, int], ...] = (),
        workspace_executable_identities: tuple[tuple[str, Path, Path, int, int], ...] = (),
        automatic_tool_reads: tuple[Path, ...] = (),
        generated_environment: tuple[str, ...] = (),
        protected_instruction_names: tuple[str, ...] = (),
        file_creation_mask: int = constants.GROUP_AND_OTHER_WRITE_MODE_MASK,
        scratch_root: Path | None = None,
        scratch_data: bool = False,
        egress_origins: tuple[str, ...] = (),
        stream_output: bool = False,
        runtime_timeout: float | None = None,
        cleanup_proof: bool = False,
        terminal: bool = False,
        local_network: bool = True,
    ) -> SandboxRequest:
        """Create a request and capture every authority-bearing directory identity.

        Args:
            source (str): Nonempty opaque shell source.
            cwd (Path): Absolute real working directory.
            workspace (Path): Absolute real workspace root.
            read_roots (tuple[Path, ...]): Additional readable directory roots.
            write_roots (tuple[Path, ...]): Writable directory roots.
            network (bool): Explicit general egress grant.
            egress_origins (tuple[str, ...]): Exact mediated HTTP(S) origins, or empty for none.
            stream_output (bool): Request bounded live output over native transport.
            runtime_timeout (float | None): Positive finite runtime seconds after native launch,
                or None for the original absolute deadline.
            cleanup_proof (bool): Request an authenticated descendant-cleanup verdict.
            terminal (bool): Request a receiver-owned pseudo-terminal for one interactive job.
            local_network (bool): Permit isolated attempt-local loopback when qualified.
            environment (dict[str, str]): Sanitized effective environment. An included TMPDIR
                must name a live private directory in the process temp root or the bound
                ``$TMPDIR`` root.
            policy_version (str): Reviewed policy revision.
            deadline (float): Positive finite absolute monotonic deadline.
            workspace_id (str): Stable workspace identity.
            execution_source (str | None): Exact internal execution source, or the original when
                omitted.
            git_create (bool): Permit creation of a currently absent top-level `.git` directory.
            read_context (str | None): Honest plain-language context for extra read authority.
            read_aliases (tuple[Path, ...]): Directory symlinks into approved read roots.
            executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Bound
                executable names, spellings, canonical files, and device/inode identities.
            workspace_executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Bound
                workspace-selected spellings, canonical files, and device/inode identities.
            automatic_tool_reads (tuple[Path, ...]): Narrow roots supplied by validated
                executable/runtime references, excluding manual reads and Git metadata.
            generated_environment (tuple[str, ...]): Runtime-generated environment names whose
                values resolve within private scratch and may vary between attempts.
            protected_instruction_names (tuple[str, ...]): Validated instruction basenames
                whose workspace writes the native policy must deny.
            file_creation_mask (int): Caller process umask applied before command source starts.
            scratch_root (Path | None): Session-owned physical root matching TMPDIR. Explicit
                roots need no global temporary directory. Defaults to system temporary-root
                validation.
            scratch_data (bool): Allow ordinary names in fresh agent-only scratch. Requires an
                explicit disjoint scratch_root; defaults to restricted scratch semantics.

        Returns:
            SandboxRequest: Immutable request with bound directory identities.

        Raises:
            ValueError: If source, policy, identity, environment, or a directory is invalid.
            TypeError: If the streaming flag is not a Boolean.
            OSError: If an approved directory cannot be inspected.
        """
        if (
            not source
            or not policy_version
            or not workspace_id
            or not is_finite_deadline(deadline)
            or deadline <= 0
        ):
            raise ValueError(
                "Sandbox source, policy, workspace identity, and deadline are required."
            )
        if (
            not isinstance(file_creation_mask, int)
            or isinstance(file_creation_mask, bool)
            or not 0 <= file_creation_mask <= constants.FILE_MODE_PERMISSION_MASK
        ):
            raise ValueError("Sandbox file creation mask is invalid.")
        protected_workspace_paths(protected_instruction_names)
        if execution_source is not None and not execution_source:
            raise ValueError("Translated sandbox source cannot be empty.")
        if git_create and (workspace / ".git").is_symlink():
            raise ValueError("Git creation target cannot be a symlink.")
        if git_create and (workspace / ".git").exists():
            raise ValueError("Git creation grant requires absent top-level metadata.")
        if git_create and workspace not in write_roots:
            raise ValueError("Git creation requires an approved workspace write root.")
        if any(
            not name or "=" in name or "\x00" in name or "\x00" in value
            for name, value in environment.items()
        ):
            raise ValueError("Sandbox environment contains an invalid entry.")
        scratch_value = environment.get(TMPDIR_ENVIRONMENT_VARIABLE)
        if (
            not isinstance(scratch_data, bool)
            or scratch_root is not None
            and str(scratch_root) != scratch_value
            or scratch_data
            and (
                scratch_root is None
                or any(
                    root.is_relative_to(scratch_root) or scratch_root.is_relative_to(root)
                    for root in (workspace, *read_roots, *write_roots)
                )
            )
        ):
            raise ValueError("Session scratch authority must be explicit and disjoint.")
        if scratch_value is not None:
            scratch = Path(scratch_value)
            details = scratch.lstat()
            if (
                not scratch.is_absolute()
                or scratch.resolve(strict=True) != scratch
                or not stat.S_ISDIR(details.st_mode)
                or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != constants.PRIVATE_DIRECTORY_MODE
                or (
                    scratch_root != scratch
                    and scratch.parent != Path(tempfile.gettempdir()).resolve(strict=True)
                )
            ):
                raise ValueError("Command scratch must be a private real directory.")
        if _generated_environment_values(environment, generated_environment) is None:
            raise ValueError("Generated command environment must remain inside private scratch.")
        path_roots = path_search_roots(environment, cwd)
        for name, spelling, resolved, device, inode in executable_identities:
            match = resolve_host_executable(
                name, environment.get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH), cwd
            )
            details = resolved.stat()
            if match != (spelling, resolved) or (details.st_dev, details.st_ino) != (device, inode):
                raise ValueError("Executable identity changed before request binding.")
        if len(workspace_executable_identities) > constants.MAX_COMMAND_PATH_COUNT:
            raise ValueError("Too many selected workspace executables.")
        for name, spelling, resolved, device, inode in workspace_executable_identities:
            if (
                not isinstance(name, str)
                or not name
                or "\x00" in name
                or not isinstance(spelling, Path)
                or not isinstance(resolved, Path)
                or not isinstance(device, int)
                or isinstance(device, bool)
                or device < 0
                or not isinstance(inode, int)
                or isinstance(inode, bool)
                or inode < 0
                or not spelling.is_absolute()
                or ".." in spelling.parts
                or not spelling.is_relative_to(workspace)
                or resolved.is_relative_to(workspace)
                or not any(resolved.is_relative_to(root) for root in read_roots)
            ):
                raise ValueError("Selected workspace executable is outside its approved scope.")
            try:
                match = resolve_selected_executable(
                    name, environment.get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH), cwd
                )
                details = resolved.stat()
            except RuntimeError as error:
                raise ValueError("Selected workspace executable is unavailable.") from error
            if (
                match != (spelling, resolved)
                or not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or (details.st_dev, details.st_ino) != (device, inode)
            ):
                raise ValueError("Selected workspace executable changed before request binding.")
        if automatic_tool_reads and (
            not executable_identities or not set(automatic_tool_reads) <= set(read_roots)
        ):
            raise ValueError("Automatic tool reads require bound executable roots.")
        read_alias_identities = []
        for alias in read_aliases:
            details = alias.lstat()
            target = alias.resolve(strict=True)
            if (
                not alias.is_absolute()
                or not stat.S_ISLNK(details.st_mode)
                or not target.is_dir()
                or not any(target.is_relative_to(root) for root in read_roots)
            ):
                raise ValueError("Invalid executable read alias.")
            read_alias_identities.append((alias, details.st_dev, details.st_ino, target))
        directories = tuple(
            dict.fromkeys(
                (
                    cwd,
                    workspace,
                    *read_roots,
                    *write_roots,
                    *(alias.parent for alias in read_aliases),
                    *(target for _, _, _, target in read_alias_identities),
                    *(real for _, real in path_roots),
                    *((scratch,) if scratch_value is not None else ()),
                )
            )
        )
        identities = tuple(DirectoryIdentity.capture(path) for path in directories)
        if not cwd.is_relative_to(workspace):
            raise ValueError("Sandbox cwd must be inside the approved workspace.")
        if network and egress_origins:
            raise ValueError("General and mediated network grants cannot be combined.")
        if not isinstance(local_network, bool):
            raise TypeError("Native local-network flag must be a boolean.")
        if network and not local_network:
            raise ValueError("A general network grant cannot enforce a local-network deny.")
        if (
            not isinstance(egress_origins, tuple)
            or len(egress_origins) > 16
            or any(normalize_egress_origin(origin) != origin for origin in egress_origins)
            or len(set(egress_origins)) != len(egress_origins)
        ):
            raise ValueError("Invalid mediated egress origin grant.")
        if not isinstance(stream_output, bool):
            raise TypeError("Native output streaming flag must be a boolean.")
        if not isinstance(cleanup_proof, bool):
            raise TypeError("Native cleanup proof flag must be a boolean.")
        if not isinstance(terminal, bool):
            raise TypeError("Native terminal flag must be a boolean.")
        if runtime_timeout is not None and (
            not is_finite_deadline(runtime_timeout) or runtime_timeout <= 0
        ):
            raise ValueError("Native command runtime budget must be positive and finite.")
        return cls(
            source,
            cwd,
            workspace,
            read_roots,
            write_roots,
            network,
            tuple(sorted(environment.items())),
            policy_version,
            deadline,
            workspace_id,
            identities,
            path_roots,
            execution_source or source,
            git_create,
            str(uuid4()),
            read_context,
            read_aliases,
            tuple(read_alias_identities),
            executable_identities,
            workspace_executable_identities,
            automatic_tool_reads,
            tuple(dict.fromkeys(generated_environment)),
            tuple(dict.fromkeys(protected_instruction_names)),
            file_creation_mask,
            scratch_root,
            scratch_data,
            egress_origins,
            stream_output,
            runtime_timeout,
            cleanup_proof,
            terminal,
            local_network,
        )

    def directory_matches_descriptor(self, path: Path, descriptor: int) -> bool:
        """Check an open directory against the request's originally approved identity.

        Args:
            path (Path): Canonical directory path captured in this request.
            descriptor (int): Open descriptor to the directory being checked.

        Returns:
            bool: True only when the descriptor and current path retain the bound identity.
        """
        identity = next((item for item in self._identities if item.path == path), None)
        if identity is None:
            return False
        try:
            details = os.fstat(descriptor)
            return (
                stat.S_ISDIR(details.st_mode)
                and (details.st_dev, details.st_ino) == (identity.device, identity.inode)
                and identity.is_current()
            )
        except OSError:
            return False

    def paths_are_current(self) -> bool:
        """Check directory identities and generated scratch containment before launch.

        Returns:
            bool: Whether directory identities and generated scratch containment remain valid.
        """
        if not all(identity.is_current() for identity in self._identities) or (
            _generated_environment_values(self.shell_environment(), self.generated_environment)
            is None
        ):
            return False
        for name, spelling, resolved, device, inode in self.executable_identities:
            try:
                match = resolve_host_executable(
                    name, self.shell_environment()[PATH_ENVIRONMENT_VARIABLE], self.cwd
                )
                details = resolved.stat()
            except (OSError, ValueError, KeyError):
                return False
            if match != (spelling, resolved) or (details.st_dev, details.st_ino) != (device, inode):
                return False
        for name, spelling, resolved, device, inode in self.workspace_executable_identities:
            try:
                match = resolve_selected_executable(
                    name,
                    self.shell_environment().get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH),
                    self.cwd,
                )
                details = resolved.stat()
            except (OSError, RuntimeError, ValueError):
                return False
            if (
                match != (spelling, resolved)
                or not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or (details.st_dev, details.st_ino) != (device, inode)
            ):
                return False
        for alias, device, inode, target in self._read_alias_identities:
            try:
                details = alias.lstat()
                if (
                    not stat.S_ISLNK(details.st_mode)
                    or (details.st_dev, details.st_ino) != (device, inode)
                    or alias.resolve(strict=True) != target
                ):
                    return False
            except OSError:
                return False
        try:
            current_path_roots = path_search_roots(self.shell_environment(), self.cwd)
        except (OSError, RuntimeError, ValueError):
            return False
        if current_path_roots != self.path_roots:
            return False
        git_path = self.workspace / ".git"
        return not (self.git_create and (git_path.exists() or git_path.is_symlink()))

    def shell_environment(self) -> dict[str, str]:
        """Return a fresh environment mapping for the command process.

        Returns:
            dict[str, str]: Exact sanitized environment approved in this request.
        """
        return dict(self.environment)

    def to_wire(self) -> bytes:
        """Serialize an exact current request snapshot for authenticated local admission.

        Preserve every constructor and captured identity field, including the original deadline
        and attempt ID. This bounded canonical JSON is private transport data, never model output
        or an authorization token. The local receiver must restore and revalidate it before setup.

        Returns:
            bytes: Canonical UTF-8 request snapshot of at most 64 KiB.

        Raises:
            ValueError: Authority is stale, the deadline is nonfinite or the snapshot is oversized.
        """
        if not is_finite_deadline(self.deadline) or not self.paths_are_current():
            raise ValueError("Sandbox request authority is stale or its deadline is invalid.")
        value = asdict(self)
        if not self.workspace_executable_identities:
            del value["workspace_executable_identities"]
        if not self.egress_origins:
            del value["egress_origins"]
        if not self.stream_output:
            del value["stream_output"]
        if self.runtime_timeout is None:
            del value["runtime_timeout"]
        if not self.cleanup_proof:
            del value["cleanup_proof"]
        if not self.terminal:
            del value["terminal"]
        if self.local_network:
            del value["local_network"]
        envelope = {"schema": 1, "request": value}
        data = json.dumps(
            envelope,
            default=str,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(data) > constants.MAX_NATIVE_REQUEST_BYTES and self.source == self.execution_source:
            del value["execution_source"]
            envelope["schema"] = 2
            data = json.dumps(
                envelope,
                default=str,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        if len(data) > constants.MAX_NATIVE_REQUEST_BYTES:
            raise ValueError("Sandbox request snapshot exceeds the admission bound.")
        return data

    @classmethod
    def from_wire(cls, data: bytes) -> SandboxRequest:
        """Restore the shared request without recapturing changed paths as new authority.

        Call only after original installed authority and ordinary peer/frame admission. Validate
        a duplicate-free fixed schema, restore constructor inputs through create,
        retain the original canonical attempt ID and require the complete revalidated snapshot
        to equal the transmitted snapshot. Changes to directories, aliases, PATH or executable
        identities refuse. No source runs, grant expands, deadline resets or approval is inferred.

        Args:
            data (bytes): Private UTF-8 request snapshot received through authenticated admission,
                nonempty and at most 64 KiB.

        Returns:
            SandboxRequest: Exact current request, with its original deadline and attempt ID.

        Raises:
            ValueError: JSON/schema/types, bounds, original snapshot or current authority differ.
            SandboxRequestExpired: The original deadline expires; the exception retains only
                fully validated request authority for bounded result delivery.
            OSError: A required approved path or executable cannot be inspected.
        """
        if not isinstance(data, bytes) or not 0 < len(data) <= constants.MAX_NATIVE_REQUEST_BYTES:
            raise ValueError("Invalid bounded sandbox request snapshot.")
        try:
            envelope = load_unique_json(
                data.decode("utf-8"), duplicate_message="Duplicate sandbox request field."
            )
        except (UnicodeError, RecursionError) as error:
            raise ValueError("Invalid sandbox request JSON.") from error
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"schema", "request"}
            or not isinstance(envelope["schema"], int)
            or isinstance(envelope["schema"], bool)
            or envelope["schema"] not in (1, 2)
            or not isinstance(envelope["request"], dict)
            or frozenset(envelope["request"])
            not in {
                frozenset(
                    {field.name for field in fields(cls)}
                    - omitted
                    - streaming
                    - runtime
                    - proof
                    - terminal
                    - local
                    - ({"execution_source"} if envelope["schema"] == 2 else set())
                )
                for omitted in (
                    set(),
                    {"workspace_executable_identities"},
                    {"egress_origins"},
                    {"workspace_executable_identities", "egress_origins"},
                )
                for streaming in (set(), {"stream_output"})
                for runtime in (set(), {"runtime_timeout"})
                for proof in (set(), {"cleanup_proof"})
                for terminal in (set(), {"terminal"})
                for local in (set(), {"local_network"})
            }
        ):
            raise ValueError("Invalid sandbox request schema.")
        value = dict(envelope["request"])
        value.setdefault("workspace_executable_identities", [])
        value.setdefault("egress_origins", [])
        value.setdefault("stream_output", False)
        value.setdefault("runtime_timeout", None)
        value.setdefault("cleanup_proof", False)
        value.setdefault("terminal", False)
        value.setdefault("local_network", True)
        if envelope["schema"] == 2:
            value["execution_source"] = value["source"]
        deadline = value["deadline"]
        if not is_finite_deadline(deadline):
            raise ValueError("Invalid sandbox request deadline.")
        expired = time.monotonic() >= deadline
        text_names = ("source", "execution_source", "policy_version", "workspace_id", "attempt_id")
        text = {name: _wire_text(value[name]) for name in text_names}
        if str(UUID(text["attempt_id"])) != text["attempt_id"]:
            raise ValueError("Invalid sandbox request attempt identity.")
        if any(
            not isinstance(value[name], bool)
            for name in (
                "network",
                "git_create",
                "scratch_data",
                "stream_output",
                "cleanup_proof",
                "terminal",
                "local_network",
            )
        ):
            raise ValueError("Invalid sandbox request boolean authority.")
        if value["runtime_timeout"] is not None and (
            not is_finite_deadline(value["runtime_timeout"]) or value["runtime_timeout"] <= 0
        ):
            raise ValueError("Invalid sandbox request runtime budget.")
        file_creation_mask = value["file_creation_mask"]
        if (
            not isinstance(file_creation_mask, int)
            or isinstance(file_creation_mask, bool)
            or not 0 <= file_creation_mask <= constants.FILE_MODE_PERMISSION_MASK
        ):
            raise ValueError("Invalid sandbox request file creation mask.")
        if value["read_context"] is not None:
            _wire_text(value["read_context"])
        collections = (
            "read_roots",
            "write_roots",
            "read_aliases",
            "automatic_tool_reads",
            "environment",
            "executable_identities",
            "workspace_executable_identities",
            "generated_environment",
            "protected_instruction_names",
            "egress_origins",
        )
        if any(not isinstance(value[name], list) for name in collections):
            raise ValueError("Invalid sandbox request collection.")
        paths = {
            name: tuple(Path(_wire_text(path)) for path in value[name])
            for name in ("read_roots", "write_roots", "read_aliases", "automatic_tool_reads")
        }
        restored = cls.create(
            source=text["source"],
            execution_source=text["execution_source"],
            policy_version=text["policy_version"],
            workspace_id=text["workspace_id"],
            cwd=Path(_wire_text(value["cwd"])),
            workspace=Path(_wire_text(value["workspace"])),
            **paths,
            network=value["network"],
            egress_origins=tuple(_wire_text(origin) for origin in value["egress_origins"]),
            git_create=value["git_create"],
            deadline=deadline,
            read_context=value["read_context"],
            environment=dict(_wire_items(value["environment"], (str, str))),
            executable_identities=_wire_items(
                value["executable_identities"], (str, Path, Path, int, int)
            ),
            workspace_executable_identities=_wire_items(
                value["workspace_executable_identities"], (str, Path, Path, int, int)
            ),
            generated_environment=tuple(
                _wire_text(name) for name in value["generated_environment"]
            ),
            protected_instruction_names=tuple(
                _wire_text(name) for name in value["protected_instruction_names"]
            ),
            file_creation_mask=file_creation_mask,
            scratch_root=Path(_wire_text(value["scratch_root"]))
            if value["scratch_root"] is not None
            else None,
            scratch_data=value["scratch_data"],
            stream_output=value["stream_output"],
            runtime_timeout=value["runtime_timeout"],
            cleanup_proof=value["cleanup_proof"],
            terminal=value["terminal"],
            local_network=value["local_network"],
        )
        restored = replace(restored, attempt_id=text["attempt_id"])
        canonical = json.dumps(
            envelope, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if restored.to_wire() != canonical:
            raise ValueError("Sandbox request snapshot differs from current authority.")
        if expired or time.monotonic() >= deadline:
            raise SandboxRequestExpired(restored)
        return restored

    def host_command_signature(self) -> str:
        """Describe a host rule using only virtual workspace coordinates.

        Returns:
            str: Path-independent digest of virtual cwd and safe command shape.
        """
        payload = (
            str(self.cwd.relative_to(self.workspace)),
            self.source if "/" not in self.source else "<path-bearing-command>",
            self.policy_version,
            self.network,
            self.git_create,
        )
        return sha256_digest(repr(payload))

    def host_command_runtime_signature(self) -> str:
        """Bind concrete host authority only in session memory.

        Returns:
            str: Digest of captured directory and alias identities, exact stable environment,
                scratch-relative generated values, and executable identities.
        """
        workspace = self.workspace.stat()
        source = self.execution_source
        scratch = self.shell_environment().get(TMPDIR_ENVIRONMENT_VARIABLE)
        generated_scratch = (
            Path(scratch)
            if scratch is not None
            and TMPDIR_ENVIRONMENT_VARIABLE in self.generated_environment
            and self.scratch_root is None
            else None
        )
        stable_directories = {
            self.cwd,
            self.workspace,
            *self.read_roots,
            *self.write_roots,
            *(alias.parent for alias in self.read_aliases),
            *(target for _, _, _, target in self._read_alias_identities),
            *(target for _, target in self.path_roots),
        }
        if scratch is not None and generated_scratch is None:
            stable_directories.add(Path(scratch))

        def independently_authorized(path: Path) -> bool:
            """Retain generated directories that overlap other bound authority."""
            return any(
                path.is_relative_to(root) or root.is_relative_to(path)
                for root in stable_directories
            )

        identities = tuple(
            identity for identity in self._identities if independently_authorized(identity.path)
        )
        payload = (
            self.source,
            source,
            self.cwd,
            self.workspace,
            self.workspace_id,
            workspace.st_dev,
            workspace.st_ino,
            identities,
            tuple(
                (key, value)
                for key, value in self.environment
                if key not in self.generated_environment
            ),
            self.generated_environment,
            _generated_environment_values(self.shell_environment(), self.generated_environment),
            self.read_roots,
            self.write_roots,
            self.network,
            self.policy_version,
            self.path_roots,
            self.read_aliases,
            self._read_alias_identities,
            self.executable_identities,
            self.automatic_tool_reads,
            self.git_create,
            self.protected_instruction_names,
            self.file_creation_mask,
            self.scratch_root,
            self.scratch_data,
        )
        if self.workspace_executable_identities:
            payload = (*payload, self.workspace_executable_identities)
        if self.egress_origins:
            payload = (*payload, self.egress_origins)
        if not self.local_network:
            payload = (*payload, "local-network-denied")
        return sha256_digest(repr(payload))

    def sandbox_command_signature(self, identity: str) -> str:
        """Return the authority identity used for a remembered sandbox-command approval.

        Args:
            identity (str): Scope-specific policy identity selected by the permission manager.

        Returns:
            str: SHA-256 digest binding the sandbox authority to the selected identity, paths,
                environment, policy version, and executable source.
        """
        relative_cwd = str(self.cwd.relative_to(self.workspace))
        roots = tuple(
            "workspace" if root == self.workspace else str(root) for root in self.write_roots
        )
        payload = (
            identity,
            relative_cwd,
            self.read_roots,
            self.automatic_tool_reads,
            self.read_aliases,
            self.executable_identities,
            roots,
            self.network,
            tuple(
                (key, value)
                for key, value in self.environment
                if key not in self.generated_environment
            ),
            self.generated_environment,
            _generated_environment_values(self.shell_environment(), self.generated_environment),
            self.path_roots,
            self.policy_version,
            self.git_create,
            self.file_creation_mask,
            self.scratch_root is not None,
            self.scratch_data,
            (),  # Preserve stored approval signatures for requests without virtual aliases.
        )
        if self.execution_source != self.source:
            payload = (*payload, self.execution_source)
        if self.workspace_executable_identities:
            payload = (*payload, self.workspace_executable_identities)
        if self.egress_origins:
            payload = (*payload, self.egress_origins)
        if not self.local_network:
            payload = (*payload, "local-network-denied")
        return sha256_digest(repr(payload))


@dataclass(frozen=True)
class CommandProcessResult:
    """Describe a sandboxed attempt and any attributable boundary failure.

    Args:
        outcome (SandboxOutcome): Command completion or classified sandbox failure.
        exit_code (int | None): Shell exit code only when the command completed.
        stdout (str): Bounded captured stdout.
        stderr (str): Bounded captured stderr.
        stdout_discarded (int | None): Characters discarded, or unknown if draining stopped.
        stderr_discarded (int | None): Characters discarded, or unknown if draining stopped.
        detail (str): Diagnostic for a sandbox failure.
        observed_denial (str): Bounded OS denial evidence observed during a shell attempt.
            Backends can encode multiple records; none establishes why the shell exited.
        possible_effects (bool): Whether the attempt may have changed state before failure.
        failure_context (str): Path-free explanation of the failed native execution stage,
            safe to display in a host approval prompt.
        diagnostic_status (Literal["unknown", "available", "unavailable"]): Availability of
            local denial collection, independently of whether a denial was observed.
        diagnostic_source (Literal["unknown", "macos-kernel-log", "linux-kernel-journal"]):
            Provenance of an observed denial, or unknown when no source is established.
        cleanup_verified (bool | None): Authenticated receiver evidence that its concrete
            cgroup membership check established descendant absence, or unknown when absent.

    Raises:
        ValueError: The outcome and exit code disagree, or diagnostic metadata is invalid.
    """

    outcome: SandboxOutcome
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_discarded: int | None = 0
    stderr_discarded: int | None = 0
    detail: str = ""
    observed_denial: str = ""
    possible_effects: bool = False
    failure_context: str = "Native sandbox execution could not start."
    diagnostic_status: Literal["unknown", "available", "unavailable"] = "unknown"
    diagnostic_source: Literal["unknown", "macos-kernel-log", "linux-kernel-journal"] = "unknown"
    cleanup_verified: bool | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if (self.outcome is SandboxOutcome.COMPLETED) != (self.exit_code is not None):
            raise ValueError("Only completed commands have an exit code.")
        if not isinstance(self.diagnostic_status, str) or self.diagnostic_status not in {
            "unknown",
            "available",
            "unavailable",
        }:
            raise ValueError("Invalid sandbox diagnostic status.")
        if not isinstance(self.diagnostic_source, str) or self.diagnostic_source not in {
            "unknown",
            "macos-kernel-log",
            "linux-kernel-journal",
        }:
            raise ValueError("Invalid sandbox diagnostic source.")
        if self.diagnostic_source != "unknown" and not self.observed_denial:
            raise ValueError("Diagnostic provenance requires an observed denial.")
        if self.observed_denial and self.diagnostic_status == "unavailable":
            raise ValueError("An observed denial requires available or unknown diagnostics.")
        if self.diagnostic_source != "unknown" and self.diagnostic_status != "available":
            raise ValueError("Known diagnostic provenance requires available diagnostics.")
        if self.cleanup_verified is not None and not isinstance(self.cleanup_verified, bool):
            raise ValueError("Invalid local descendant cleanup evidence.")

    def to_wire(self, attempt_id: str, *, include_cleanup_proof: bool = False) -> bytes:
        """Encode a bounded result bound to the original command attempt.

        Args:
            attempt_id (str): Original canonical request UUID, never a replacement attempt.
            include_cleanup_proof (bool): Include the optional receiver verdict only when the
                original request negotiated it; legacy clients retain their exact result shape.

        Returns:
            bytes: Version-one or version-two UTF-8 result envelope with bounded fields.

        Raises:
            ValueError: Result fields, attempt identity or encoded size are invalid.
        """
        value = asdict(self)
        if not include_cleanup_proof:
            del value["cleanup_verified"]
        schema = (
            1 if self.diagnostic_status == "unknown" and self.diagnostic_source == "unknown" else 2
        )
        if schema == 1:
            del value["diagnostic_status"]
            del value["diagnostic_source"]
        data = json.dumps(
            {"schema": schema, "attempt_id": attempt_id, "result": value},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        restored = self.from_wire(data, attempt_id)
        if restored != self or (
            include_cleanup_proof and restored.cleanup_verified != self.cleanup_verified
        ):
            raise ValueError("Sandbox result differs after encoding.")
        return data

    @classmethod
    def from_wire(cls, data: bytes, attempt_id: str) -> CommandProcessResult:
        """Restore an exact bounded result for the original attempt without inferring a denial.

        Accept only the supported schemas, JSON field types and completion/exit relationship.
        Version one has no diagnostic metadata and restores it as unknown.
        Output retains the common supervisor's character limits; diagnostics are each at most
        2,000 characters. Unknown discarded counts remain None. Parsing supplies no native
        enforcement evidence or retry authority.

        Args:
            data (bytes): Nonempty UTF-8 result envelope, at most the fixed wire ceiling.
            attempt_id (str): Original canonical request UUID expected by the ordinary client.

        Returns:
            CommandProcessResult: Exact typed result without executing or replaying source.

        Raises:
            ValueError: JSON, schema, types, bounds, attempt identity or outcome is invalid.
        """
        # JSON escapes each output character into at most twelve ASCII bytes (a surrogate pair).
        if not isinstance(data, bytes) or not 0 < len(data) <= constants.MAX_NATIVE_RESULT_BYTES:
            raise ValueError("Invalid bounded sandbox result envelope.")
        if not isinstance(attempt_id, str) or str(UUID(attempt_id)) != attempt_id:
            raise ValueError("Invalid sandbox result attempt identity.")
        try:
            envelope = load_unique_json(
                data.decode("utf-8"), duplicate_message="Duplicate sandbox request field."
            )
        except (UnicodeError, RecursionError) as error:
            raise ValueError("Invalid sandbox result JSON.") from error
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"schema", "attempt_id", "result"}
            or not isinstance(envelope["schema"], int)
            or isinstance(envelope["schema"], bool)
            or envelope["schema"] not in NATIVE_ACCEPTED_RESULT_SCHEMAS
            or envelope["attempt_id"] != attempt_id
            or not isinstance(envelope["result"], dict)
            or set(envelope["result"])
            not in (
                {
                    field.name
                    for field in fields(cls)
                    if field.name != "cleanup_verified"
                    and (
                        envelope["schema"] == 2
                        or field.name not in {"diagnostic_status", "diagnostic_source"}
                    )
                },
                {
                    field.name
                    for field in fields(cls)
                    if envelope["schema"] == 2
                    or field.name not in {"diagnostic_status", "diagnostic_source"}
                },
            )
        ):
            raise ValueError("Invalid sandbox result schema or attempt binding.")
        value = dict(envelope["result"])
        if not isinstance(value["outcome"], str):
            # Malformed wire values remain ValueError under the from_wire contract.
            raise ValueError("Invalid sandbox result outcome.")  # noqa: TRY004
        value["outcome"] = SandboxOutcome(value["outcome"])
        if value["outcome"] is SandboxOutcome.UNSUPPORTED:
            raise ValueError("Unsupported capability is a client-only outcome.")
        for name in ("stdout", "stderr", "detail", "observed_denial", "failure_context"):
            maximum = constants.MAX_OUTPUT_CHARS if name in ("stdout", "stderr") else 2000
            if not isinstance(value[name], str) or len(value[name]) > maximum:
                raise ValueError("Invalid bounded sandbox result text.")
        for name in ("stdout_discarded", "stderr_discarded"):
            if value[name] is not None and (
                not isinstance(value[name], int)
                or isinstance(value[name], bool)
                or not 0 <= value[name] <= (1 << 63) - 1
            ):
                raise ValueError("Invalid sandbox result discarded count.")
        if value["exit_code"] is not None and (
            not isinstance(value["exit_code"], int)
            or isinstance(value["exit_code"], bool)
            or not -(1 << 31) <= value["exit_code"] < 1 << 31
        ):
            raise ValueError("Invalid sandbox result exit code.")
        if not isinstance(value["possible_effects"], bool):
            # Malformed wire values remain ValueError under the from_wire contract.
            raise ValueError("Invalid sandbox result effects flag.")  # noqa: TRY004
        return cls(**value)


class SandboxBackend(Protocol):
    """Launch a request under one verified operating-system sandbox policy.

    Attributes:
        capabilities (frozenset[SandboxCapability]): Qualified features of this selected backend.
        qualification_capabilities (frozenset[SandboxCapability]): Features the independent
            startup probe may request before optional qualification completes.
    """

    capabilities: frozenset[SandboxCapability]
    qualification_capabilities: frozenset[SandboxCapability]

    def capability_failure(self) -> str | CommandProcessResult | None:
        """Report preflight native authority failure before an enforcement probe.

        Returns:
            str | CommandProcessResult | None: Failure or None when probing may proceed.
        """

    def qualification_layout(self) -> QualificationProbeLayout:
        """Select safe temporary parents for an independent enforcement probe.

        Returns:
            QualificationProbeLayout: Workspace and outside-canary parent placement.

        Raises:
            OSError: The host has no safe probe parent.
        """

    def qualification_failure(
        self, result: CommandProcessResult | None, reason: str
    ) -> str | CommandProcessResult:
        """Classify a failed native enforcement probe without permitting a host retry.

        Args:
            result (CommandProcessResult | None): Native result, or None if setup failed.
            reason (str): Sanitized probe failure explanation.

        Returns:
            str | CommandProcessResult: Backend-specific fail-closed selection result.
        """

    def close(self) -> None:
        """Release resources held by a rejected or retired backend."""

    def qualification_probe(self, request: SandboxRequest) -> CommandProcessResult:
        """Run a native request for independent enforcement verification.

        Args:
            request (SandboxRequest): Owned probe with observable denied effects.

        Returns:
            CommandProcessResult: Native probe result.
        """

    def complete_qualification(self, request: SandboxRequest) -> None:
        """Finish capability qualification after the probe's effects are verified.

        Args:
            request (SandboxRequest): Probe whose confinement was verified.
        """

    @property
    def policy_version(self) -> str:
        """Return the identity of the backend's reviewed native policy.

        Returns:
            str: Native policy revision bound into approvals and audit.
        """

    @property
    def scratch_prefix(self) -> str:
        """Return the backend's private scratch-directory naming requirement.

        Returns:
            str: Prefix accepted during native request validation.
        """

    @property
    def trusted_file_owner_ids(self) -> tuple[int, ...]:
        """Return file owners trusted for automatic executable binding.

        Returns:
            tuple[int, ...]: Platform-verified owner IDs in the caller's current namespace.
        """

    @property
    def system_read_roots(self) -> tuple[Path, ...]:
        """Return reviewed system roots that require no extra read approval.

        Returns:
            tuple[Path, ...]: Platform adapter's fixed default read roots.
        """

    def managed_tool_root(self, root: Path) -> bool:
        """Report whether one installed tool root has verified automatic-read provenance.

        Args:
            root (Path): Canonical installation directory.

        Returns:
            bool: Whether the backend permits this root without an extra prompt.
        """

    def installed_tool_roots(
        self,
        resolved: Path,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]:
        """Return backend-specific bound read roots for one host executable.

        Args:
            resolved (Path): Canonical executable chosen from PATH.

        Returns:
            tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]: Read roots and
                installation aliases whose identities must remain stable.

        Raises:
            OSError: If installation metadata cannot be read.
            ValueError: If installation metadata is invalid.
        """

    def automatic_tool_read_roots(
        self,
        workspace: Path,
        path_roots: tuple[tuple[Path, Path], ...],
        executable_identities: tuple[tuple[str, Path, Path, int, int], ...],
    ) -> tuple[Path, ...]:
        """Discover runtime roots needed by identity-bound tools for separate approval.

        Args:
            workspace (Path): Canonical selected workspace.
            path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and directories.
            executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Exact names,
                canonical files, and file identities admitted for this request.

        Returns:
            tuple[Path, ...]: Candidate runtime roots that require scoped read authorization.

        Raises:
            ValueError: A discovered runtime reference is unsafe or ambiguous.
        """

    def run(
        self,
        request: SandboxRequest,
        *,
        cancellation_event: Event | None = None,
        output_observer: Callable[[str, str], None] | None = None,
        terminal_input: Callable[[], bytes | None] | None = None,
        policy_authority: PermissionManager | None = None,
        approved_revision: int | None = None,
    ) -> CommandProcessResult:
        """Run one approved request or return a typed sandbox failure.

        Args:
            request (SandboxRequest): Exact authorized command and policy.
            cancellation_event (Event | None): Optional request to stop and clean up its process.
            output_observer (Callable[[str, str], None] | None): Optional bounded live output sink.
            terminal_input (Callable[[], bytes | None] | None): Optional transient interactive
                control frames after native payload entry.
            policy_authority (PermissionManager | None): Active mediated-egress authority.
            approved_revision (int | None): Revision captured with the exact approval.

        Returns:
            CommandProcessResult: Captured completion or classified boundary outcome.
        """
