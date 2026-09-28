"""Define authority and outcomes shared by native command sandbox backends."""

from __future__ import annotations

import os
import re
import stat
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ... import constants
from ...permissions import ProtectedWorkspacePaths, protected_workspace_paths
from ...utils import VirtualPath, sha256_digest


class SandboxOutcome(StrEnum):
    """Classify a command result without mistaking its exit code for a sandbox denial."""

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


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
    path_value = environment.get("PATH")
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
    name: str, path_value: str, cwd: Path | None = None
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
        raise ValueError("Executable lookup requires a simple name of at most 128 characters.")
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
    for spelling, _ in path_search_roots({"PATH": path_value}, working_directory):
        candidate = spelling / name
        try:
            resolved = candidate.resolve(strict=True)
            details = resolved.stat()
        except (OSError, RuntimeError):
            continue
        if stat.S_ISREG(details.st_mode) and os.access(candidate, os.X_OK):
            return candidate, resolved
    return None


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
            including the live private TMPDIR when the facade owns the attempt.
        policy_version (str): Reviewed policy revision used for this approval.
        deadline (float): Absolute monotonic deadline for this attempt.
        workspace_id (str): Stable identity of the selected workspace.
        path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and canonical search roots.
        execution_source (str): Shell source after literal virtual-root substitution.
        aliases (tuple[tuple[str, Path, Path], ...]): Virtual root, private symlink, and canonical
            target bindings retained through sandbox execution and any host retry.
        git_create (bool): Whether this approval permits creating absent top-level Git metadata.
        read_context (str | None): Plain-language description of additional read authority.
        read_aliases (tuple[Path, ...]): Bound directory symlinks into approved read roots.
        executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Named host
            executables, their PATH spellings, canonical files, and device/inode identities.
        automatic_tool_reads (tuple[Path, ...]): Read roots derived only from bound executable
            references that may run within the default sandbox without an extra prompt.
        generated_environment (tuple[str, ...]): Names of runtime-generated values strictly
            beneath the bound private scratch directory.
        protected_instruction_names (tuple[str, ...]): Instruction filenames denied native
            writes at every depth of the workspace.
        attempt_id (str): Opaque identifier correlating this attempt's audit records.
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
    aliases: tuple[tuple[str, Path, Path], ...]
    _alias_identities: tuple[tuple[Path, int, int, Path], ...]
    git_create: bool
    attempt_id: str
    read_context: str | None = None
    read_aliases: tuple[Path, ...] = ()
    _read_alias_identities: tuple[tuple[Path, int, int, Path], ...] = ()
    executable_identities: tuple[tuple[str, Path, Path, int, int], ...] = ()
    automatic_tool_reads: tuple[Path, ...] = ()
    generated_environment: tuple[str, ...] = ()
    protected_instruction_names: tuple[str, ...] = ()

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
        aliases: tuple[tuple[str, Path, Path], ...] = (),
        git_create: bool = False,
        read_context: str | None = None,
        read_aliases: tuple[Path, ...] = (),
        executable_identities: tuple[tuple[str, Path, Path, int, int], ...] = (),
        automatic_tool_reads: tuple[Path, ...] = (),
        generated_environment: tuple[str, ...] = (),
        protected_instruction_names: tuple[str, ...] = (),
    ) -> SandboxRequest:
        """Create a request and capture every authority-bearing directory identity.

        Args:
            source (str): Nonempty opaque shell source.
            cwd (Path): Absolute real working directory.
            workspace (Path): Absolute real workspace root.
            read_roots (tuple[Path, ...]): Additional readable directory roots.
            write_roots (tuple[Path, ...]): Writable directory roots.
            network (bool): Explicit general egress grant.
            environment (dict[str, str]): Sanitized effective environment. An included TMPDIR
                must name a live private Loop scratch directory in the process temp root.
            policy_version (str): Reviewed policy revision.
            deadline (float): Absolute monotonic deadline.
            workspace_id (str): Stable workspace identity.
            execution_source (str | None): Translated source, or the original when omitted.
            aliases (tuple[tuple[str, Path, Path], ...]): Approved virtual-root symlinks.
            git_create (bool): Permit creation of a currently absent top-level `.git` directory.
            read_context (str | None): Honest plain-language context for extra read authority.
            read_aliases (tuple[Path, ...]): Directory symlinks into approved read roots.
            executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Bound
                executable names, spellings, canonical files, and device/inode identities.
            automatic_tool_reads (tuple[Path, ...]): Narrow roots supplied by validated
                executable references, excluding manual reads and Git metadata.
            generated_environment (tuple[str, ...]): Runtime-generated environment names whose
                values are within private scratch and may vary between attempts.
            protected_instruction_names (tuple[str, ...]): Validated instruction basenames
                whose workspace writes the native policy must deny.

        Returns:
            SandboxRequest: Immutable request with bound directory identities.

        Raises:
            ValueError: If source, policy, identity, environment, or a directory is invalid.
            OSError: If an approved directory cannot be inspected.
        """
        if not source or not policy_version or not workspace_id or deadline <= 0:
            raise ValueError(
                "Sandbox source, policy, workspace identity, and deadline are required."
            )
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
        scratch_value = environment.get("TMPDIR")
        if scratch_value is not None:
            scratch = Path(scratch_value)
            details = scratch.lstat()
            if (
                not scratch.is_absolute()
                or scratch.resolve(strict=True) != scratch
                or not stat.S_ISDIR(details.st_mode)
                or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != 0o700
                or scratch.parent != Path(tempfile.gettempdir()).resolve(strict=True)
            ):
                raise ValueError("Command scratch must be a private real directory.")
        if generated_environment and (
            scratch_value is None
            or any(
                key not in environment
                or not Path(environment[key]).is_relative_to(Path(scratch_value))
                for key in generated_environment
            )
        ):
            raise ValueError("Generated command environment must remain inside private scratch.")
        path_roots = path_search_roots(environment, cwd)
        for name, spelling, resolved, device, inode in executable_identities:
            match = resolve_host_executable(name, environment.get("PATH", "/usr/bin:/bin"), cwd)
            details = resolved.stat()
            if match != (spelling, resolved) or (details.st_dev, details.st_ino) != (device, inode):
                raise ValueError("Executable identity changed before request binding.")
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
        alias_identities = []
        for virtual_root, alias, target in aliases:
            if (
                not virtual_root.startswith("/")
                or not alias.is_absolute()
                or re.fullmatch(r"/[A-Za-z0-9_/-]+", str(alias)) is None
                or target.resolve(strict=True) != target
            ):
                raise ValueError("Invalid command path alias.")
            parent = alias.parent
            DirectoryIdentity.capture(parent)
            parent_stat = parent.lstat()
            alias_stat = alias.lstat()
            if (
                parent_stat.st_uid != os.getuid()
                or stat.S_IMODE(parent_stat.st_mode) != 0o700
                or not stat.S_ISLNK(alias_stat.st_mode)
                or alias.resolve(strict=True) != target
            ):
                raise ValueError("Unsafe command path alias.")
            alias_identities.append((alias, alias_stat.st_dev, alias_stat.st_ino, target))
        directories = tuple(
            dict.fromkeys(
                (
                    cwd,
                    workspace,
                    *read_roots,
                    *write_roots,
                    *(item[1].parent for item in aliases),
                    *(alias.parent for alias in read_aliases),
                    *(item[2] for item in aliases),
                    *(real for _, real in path_roots),
                    *((scratch,) if scratch_value is not None else ()),
                )
            )
        )
        identities = tuple(DirectoryIdentity.capture(path) for path in directories)
        if not cwd.is_relative_to(workspace):
            raise ValueError("Sandbox cwd must be inside the approved workspace.")
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
            aliases,
            tuple(alias_identities),
            git_create,
            str(uuid4()),
            read_context,
            read_aliases,
            tuple(read_alias_identities),
            executable_identities,
            automatic_tool_reads,
            tuple(dict.fromkeys(generated_environment)),
            tuple(dict.fromkeys(protected_instruction_names)),
        )

    def paths_are_current(self) -> bool:
        """Check all approved directory names immediately before process launch.

        Returns:
            bool: Whether every directory still has its approved device and inode.
        """
        if not all(identity.is_current() for identity in self._identities):
            return False
        for name, spelling, resolved, device, inode in self.executable_identities:
            try:
                match = resolve_host_executable(name, self.shell_environment()["PATH"], self.cwd)
                details = resolved.stat()
            except (OSError, ValueError, KeyError):
                return False
            if match != (spelling, resolved) or (details.st_dev, details.st_ino) != (device, inode):
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
        if self.git_create and (git_path.exists() or git_path.is_symlink()):
            return False
        for alias, device, inode, target in self._alias_identities:
            try:
                parent = alias.parent.lstat()
                details = alias.lstat()
                resolved = alias.resolve(strict=True)
            except OSError:
                return False
            if (
                not stat.S_ISDIR(parent.st_mode)
                or parent.st_uid != os.getuid()
                or stat.S_IMODE(parent.st_mode) != 0o700
                or not stat.S_ISLNK(details.st_mode)
                or (details.st_dev, details.st_ino) != (device, inode)
                or resolved != target
            ):
                return False
        return True

    def shell_environment(self) -> dict[str, str]:
        """Return a fresh environment mapping for the command process.

        Returns:
            dict[str, str]: Exact sanitized environment approved in this request.
        """
        return dict(self.environment)

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
            tuple(
                virtual
                if virtual
                in {
                    VirtualPath.WORKSPACE,
                    VirtualPath.TEMPORARY,
                }
                or virtual.startswith(f"{VirtualPath.SKILLS}/")
                else "<other-virtual-root>"
                for virtual, _, _ in self.aliases
            ),
        )
        return sha256_digest(repr(payload))

    def host_command_runtime_signature(self) -> str:
        """Bind concrete host authority only in session memory.

        Returns:
            str: Digest of current real roots, stable environment, and executable identities.
        """
        workspace = self.workspace.stat()
        source = self.execution_source
        for virtual, alias, _ in self.aliases:
            source = source.replace(str(alias), virtual)
        payload = (
            self.source,
            source,
            self.cwd,
            self.workspace,
            self.workspace_id,
            workspace.st_dev,
            workspace.st_ino,
            tuple(
                (key, value)
                for key, value in self.environment
                if key not in self.generated_environment
            ),
            self.generated_environment,
            self.read_roots,
            self.write_roots,
            self.network,
            self.policy_version,
            self.path_roots,
            self.read_aliases,
            self.executable_identities,
            self.automatic_tool_reads,
            self.git_create,
            self.protected_instruction_names,
            tuple((virtual, target) for virtual, _, target in self.aliases),
        )
        return sha256_digest(repr(payload))

    def sandbox_command_signature(self, identity: str) -> str:
        """Return the authority identity used for a remembered sandbox-command approval.

        Args:
            identity (str): Scope-specific policy identity selected by the permission manager.

        Returns:
            str: SHA-256 digest binding the sandbox authority to the selected identity, paths,
                environment, policy version, and virtual-root targets.
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
            self.path_roots,
            self.policy_version,
            self.git_create,
            tuple(
                (virtual, "workspace" if target == self.workspace else str(target))
                for virtual, _, target in self.aliases
            ),
        )
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
        observed_denial (str): Verified OS denial observed during a completed shell attempt;
            it does not establish why the shell exited.
        possible_effects (bool): Whether the attempt may have changed state before failure.
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

    def __post_init__(self) -> None:
        if (self.outcome is SandboxOutcome.COMPLETED) != (self.exit_code is not None):
            raise ValueError("Only completed commands have an exit code.")


class SandboxBackend(Protocol):
    """Launch a request under one verified operating-system sandbox policy."""

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
        """

    def run(self, request: SandboxRequest) -> CommandProcessResult:
        """Run one approved request or return a typed sandbox failure.

        Args:
            request (SandboxRequest): Exact authorized command and policy.

        Returns:
            CommandProcessResult: Captured completion or classified boundary outcome.
        """
