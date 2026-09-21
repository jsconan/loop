"""Define immutable command sandbox plans."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

SANDBOX_POLICY_VERSION = "1"
_BACKEND_POLICY_VERSIONS = {
    "Linux": "bubblewrap-seccomp-v1",
    "Darwin": "seatbelt-v1",
    "Windows": "appcontainer-bfs-v1",
}
_RUNTIME_SEARCH_ROOTS = tuple(
    Path(root).resolve() for root in os.defpath.split(os.pathsep) if root and Path(root).is_dir()
)

type FileIdentity = tuple[int, int, int, int, int, int, str]


def _file_identity(path: Path) -> FileIdentity:
    """Return metadata and content identity used to detect authority replacement."""
    metadata = path.stat()
    digest = ""
    if stat.S_ISREG(metadata.st_mode):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_ctime_ns,
        metadata.st_mtime_ns,
        digest,
    )


@dataclass(frozen=True, slots=True)
class SandboxPlan:
    """Describe one deny-by-default process sandbox.

    Args:
        argv (tuple[str, ...]): Exact command argument vector.
        executable (Path): Canonical executable selected before launch.
        cwd (Path): Canonical working directory.
        workspace (Path): Writable workspace root.
        temporary_directory (Path | None): Writable private temporary root.
        read_only_roots (tuple[Path, ...]): Additional readable roots.
        readable_roots (tuple[Path, ...]): Effective filesystem read roots.
        writable_roots (tuple[Path, ...]): Effective filesystem write roots.
        authority_identities (Mapping[str, FileIdentity]): Immutable identities for the executable
            and every exposed root.
        environment (Mapping[str, str]): Immutable explicit child environment.
        policy_digest (str): Stable digest binding authorization to this policy.
        backend_policy (str): Native backend and policy compiler version.
    """

    argv: tuple[str, ...]
    executable: Path
    cwd: Path
    workspace: Path
    temporary_directory: Path | None
    read_only_roots: tuple[Path, ...]
    readable_roots: tuple[Path, ...]
    writable_roots: tuple[Path, ...]
    authority_identities: Mapping[str, FileIdentity]
    environment: Mapping[str, str]
    policy_digest: str
    backend_policy: str

    @classmethod
    def create(
        cls,
        argv: tuple[str, ...],
        cwd: Path,
        workspace: Path,
        temporary_directory: Path | None = None,
        read_only_roots: tuple[Path, ...] = (),
        readable_roots: tuple[Path, ...] | None = None,
        writable_roots: tuple[Path, ...] | None = None,
    ) -> SandboxPlan:
        """Create and validate a canonical sandbox plan.

        Args:
            argv (tuple[str, ...]): Non-empty exact command argument vector.
            cwd (Path): Requested working directory.
            workspace (Path): Writable workspace root.
            temporary_directory (Path | None): Optional writable private temporary root.
            read_only_roots (tuple[Path, ...]): Additional readable roots.
            readable_roots (tuple[Path, ...] | None): Effective readable roots. ``None`` uses the
                workspace and private temporary root.
            writable_roots (tuple[Path, ...] | None): Effective writable roots. ``None`` uses the
                workspace and private temporary root.

        Returns:
            SandboxPlan: Validated immutable execution plan.

        Raises:
            ValueError: If the command is empty, unavailable, or outside the exposed roots.
        """
        if not argv:
            raise ValueError("Sandbox command must include an executable.")
        canonical_cwd = cwd.resolve(strict=True)
        canonical_workspace = workspace.resolve(strict=True)
        canonical_temporary = (
            temporary_directory.resolve(strict=True) if temporary_directory is not None else None
        )
        default_roots = (canonical_workspace,) + (
            (canonical_temporary,) if canonical_temporary is not None else ()
        )
        canonical_readable = tuple(
            dict.fromkeys(
                root.resolve(strict=True)
                for root in (default_roots if readable_roots is None else readable_roots)
            )
        )
        canonical_writable = tuple(
            dict.fromkeys(
                root.resolve(strict=True)
                for root in (default_roots if writable_roots is None else writable_roots)
            )
        )
        visible_policy_roots = tuple(dict.fromkeys((*canonical_readable, *canonical_writable)))
        if not any(canonical_cwd.is_relative_to(root) for root in visible_policy_roots):
            raise ValueError("Sandbox working directory is outside readable roots.")
        command = argv[0]
        has_separator = os.sep in command or (os.altsep is not None and os.altsep in command)
        resolved: str | None = None
        if not has_separator and not Path(command).is_absolute():
            visible_roots = (*visible_policy_roots, *_RUNTIME_SEARCH_ROOTS)
            for raw_root in os.environ.get("PATH", os.defpath).split(os.pathsep):
                if not raw_root:
                    continue
                root = Path(raw_root).resolve()
                if not any(
                    root == visible or root.is_relative_to(visible) for visible in visible_roots
                ):
                    continue
                candidate = root / command
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    resolved = str(candidate)
                    break
        if resolved is None:
            candidate = Path(argv[0])
            if not candidate.is_absolute():
                candidate = canonical_cwd / candidate
            if not candidate.is_file():
                raise ValueError(f"Sandbox executable is unavailable: {argv[0]}")
            resolved = str(candidate)
        executable = Path(resolved).resolve(strict=True)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Sandbox executable is not executable: {argv[0]}")
        runtime_candidates = (
            executable.parent,
            Path(sys.executable).resolve().parent,
            Path(sys.base_prefix).resolve(strict=True),
            Path(sys.prefix).resolve(strict=True),
            Path(__file__).resolve().parents[2],
        )
        implicit_runtime_roots = [
            root
            for root in runtime_candidates
            if not any(root.is_relative_to(visible) for visible in visible_policy_roots)
        ]
        canonical_read_only = tuple(
            dict.fromkeys(
                root.resolve(strict=True) for root in (*read_only_roots, *implicit_runtime_roots)
            )
        )
        temporary_root = canonical_temporary or canonical_workspace
        path_entries = (str(executable.parent),)
        environment = {
            "HOME": str(canonical_workspace),
            "PATH": os.pathsep.join(path_entries),
            "TEMP": str(temporary_root),
            "TMP": str(temporary_root),
            "TMPDIR": str(temporary_root),
        }
        identity_paths = (
            executable,
            *visible_policy_roots,
            *canonical_read_only,
        )
        authority_identities = {
            str(path): _file_identity(path) for path in dict.fromkeys(identity_paths)
        }
        system = platform.system()
        backend_policy = _BACKEND_POLICY_VERSIONS.get(system, f"unsupported:{system}")
        payload = {
            "version": SANDBOX_POLICY_VERSION,
            "argv": [str(executable), *argv[1:]],
            "executable": str(executable),
            "cwd": str(canonical_cwd),
            "workspace": str(canonical_workspace),
            "temporary_directory": (
                str(canonical_temporary) if canonical_temporary is not None else None
            ),
            "read_only_roots": [str(root) for root in canonical_read_only],
            "readable_roots": [str(root) for root in canonical_readable],
            "writable_roots": [str(root) for root in canonical_writable],
            "authority_identities": authority_identities,
            "environment": environment,
            "network": "deny",
            "backend_policy": backend_policy,
            "resource_limits": {
                "cpu_seconds": 30,
                "memory_bytes": 1_073_741_824,
                "processes": 64,
                "open_files": 256,
                "file_size_bytes": 16_777_216,
                "core_size_bytes": 0,
            },
            "protected_workspace_paths": [".loop", ".git", ".gitignore", ".agentignore"],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            argv=(str(executable), *argv[1:]),
            executable=executable,
            cwd=canonical_cwd,
            workspace=canonical_workspace,
            temporary_directory=canonical_temporary,
            read_only_roots=canonical_read_only,
            readable_roots=canonical_readable,
            writable_roots=canonical_writable,
            authority_identities=MappingProxyType(authority_identities),
            environment=MappingProxyType(environment),
            policy_digest=digest,
            backend_policy=backend_policy,
        )

    def validate_authority(self) -> None:
        """Reject executable or root replacement between authorization and launch.

        Raises:
            ValueError: If an exposed filesystem object changed after planning.
        """
        system = platform.system()
        if self.backend_policy != _BACKEND_POLICY_VERSIONS.get(system, f"unsupported:{system}"):
            raise ValueError("Sandbox backend changed after authorization.")
        for raw_path, expected in self.authority_identities.items():
            path = Path(raw_path)
            try:
                current = _file_identity(path)
            except OSError as exc:
                raise ValueError(f"Sandbox authority disappeared before launch: {path}") from exc
            if current != expected:
                raise ValueError(f"Sandbox authority changed before launch: {path}")


@dataclass(frozen=True, slots=True)
class HostProcessPlan:
    """Describe one exact, separately confirmed host-process invocation.

    Args:
        argv (tuple[str, ...]): Exact resolved executable and argument boundaries.
        executable (Path): Canonical executable selected before confirmation.
        cwd (Path): Canonical host working directory.
        environment (Mapping[str, str]): Sanitized immutable child environment.
        incompatibility_reason (str): Specific reason sandbox execution is unavailable.
        authority_identities (Mapping[str, FileIdentity]): Executable identity to revalidate.
        confirmation_digest (str): Digest binding every displayed and executed field.
    """

    argv: tuple[str, ...]
    executable: Path
    cwd: Path
    environment: Mapping[str, str]
    incompatibility_reason: str
    authority_identities: Mapping[str, FileIdentity]
    confirmation_digest: str

    @classmethod
    def create(cls, sandbox_plan: SandboxPlan, reason: str) -> HostProcessPlan:
        """Create a structurally distinct host plan from freshly revalidated inputs.

        Args:
            sandbox_plan (SandboxPlan): Original sandbox intent used only as canonical input.
            reason (str): Specific sandbox incompatibility shown to the user.

        Returns:
            HostProcessPlan: Immutable host authority requiring fresh confirmation.

        Raises:
            ValueError: If authority changed or the reason is empty.
        """
        if not reason.strip():
            raise ValueError("Host execution requires a specific sandbox incompatibility reason.")
        sandbox_plan.validate_authority()
        identities = {
            str(path): _file_identity(path) for path in (sandbox_plan.executable, sandbox_plan.cwd)
        }
        payload = {
            "argv": sandbox_plan.argv,
            "cwd": str(sandbox_plan.cwd),
            "environment": dict(sandbox_plan.environment),
            "executable_identity": identities,
            "incompatibility_reason": reason,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            argv=sandbox_plan.argv,
            executable=sandbox_plan.executable,
            cwd=sandbox_plan.cwd,
            environment=sandbox_plan.environment,
            incompatibility_reason=reason,
            authority_identities=MappingProxyType(identities),
            confirmation_digest=digest,
        )

    def validate_authority(self) -> None:
        """Reject executable changes after host confirmation.

        Raises:
            ValueError: If executable identity changed after confirmation.
        """
        for raw_path, expected in self.authority_identities.items():
            if _file_identity(Path(raw_path)) != expected:
                raise ValueError("Confirmed host authority changed before launch.")
