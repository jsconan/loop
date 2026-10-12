"""Select a qualified built-in command sandbox for this host."""

from __future__ import annotations

import platform
import shlex
import tempfile
import time
from pathlib import Path

from .contracts import CommandProcessResult, SandboxBackend, SandboxOutcome, SandboxRequest


def _macos_backend() -> SandboxBackend:
    """Load the built-in macOS adapter only for a Darwin capability probe."""
    from .macos import MacOSSeatbeltBackend

    return MacOSSeatbeltBackend()


class UnavailableSandboxBackend:
    """Reject command execution when this host has no qualified native backend.

    Args:
        reason (str | CommandProcessResult): Path-free capability explanation or typed integrity
            failure; invalid/stale failures never permit a host retry.
    """

    _reason: str | CommandProcessResult

    def __init__(self, reason: str | CommandProcessResult) -> None:
        self._reason = reason

    @property
    def system_read_roots(self) -> tuple[Path, ...]:
        """Return no default reads while native execution is unavailable.

        Returns:
            tuple[Path, ...]: Empty authority.
        """
        return ()

    @property
    def policy_version(self) -> str:
        """Return an unavailable policy identity that grants no execution.

        Returns:
            str: Stable unavailable-backend marker.
        """
        return "unavailable-native-v1"

    @property
    def scratch_prefix(self) -> str:
        """Return a generic scratch prefix for typed unavailable requests.

        Returns:
            str: Private scratch-directory prefix.
        """
        return "loop-sandbox-"

    def automatic_tool_read_roots(
        self,
        workspace: Path,
        path_roots: tuple[tuple[Path, Path], ...],
        executable_identities: tuple[tuple[str, Path, Path, int, int], ...],
    ) -> tuple[Path, ...]:
        """Return no additional reads when no native policy can enforce them.

        Args:
            workspace (Path): Canonical selected workspace.
            path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and directories.
            executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Bound named
                executable files and their device/inode identities.

        Returns:
            tuple[Path, ...]: Empty tuple because execution is unavailable.
        """
        del self, workspace, path_roots, executable_identities
        return ()

    @staticmethod
    def managed_tool_root(root: Path) -> bool:
        """Deny automatic installed-tool reads without a native backend.

        Args:
            root (Path): Candidate installation directory.

        Returns:
            bool: Always false while no policy can enforce the read root.
        """
        return False

    @staticmethod
    def installed_tool_roots(
        resolved: Path,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]:
        """Return only a narrow manual-read candidate without native package trust.

        Args:
            resolved (Path): Canonical installed executable.

        Returns:
            tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]: Parent read root
                and no automatic alias bindings.
        """
        return (resolved.parent,), ()

    def capability_failure(self) -> str | CommandProcessResult:
        """Describe why ordinary sandboxed commands cannot start.

        Returns:
            str | CommandProcessResult: Sanitized availability or typed integrity failure.
        """
        return self._reason

    def run(self, request: SandboxRequest) -> CommandProcessResult:
        """Return a typed failure without launching a child.

        Args:
            request (SandboxRequest): Command that cannot be executed on this host.

        Returns:
            CommandProcessResult: Original typed capability or integrity failure.
        """
        if isinstance(self._reason, CommandProcessResult):
            return self._reason
        return CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            detail=self._reason,
            failure_context="Native sandbox capability check failed.",
        )


def select_sandbox_backend() -> SandboxBackend:
    """Select authenticated Linux installation or natively probed macOS confinement.

    Returns:
        SandboxBackend: Available native backend or a fail-closed unavailable backend.
    """
    host = platform.system()
    if host == "Linux":
        from .linux import LinuxNativeBackend

        backend = LinuxNativeBackend()
        failure = backend.capability_failure()
        if failure is not None:
            return UnavailableSandboxBackend(failure)
    elif host == "Darwin":
        backend = _macos_backend()
    else:
        return UnavailableSandboxBackend("No native command sandbox is qualified for this host.")
    try:
        with (
            tempfile.TemporaryDirectory(prefix="loop-sandbox-probe-") as directory,
            tempfile.TemporaryDirectory(prefix=backend.scratch_prefix) as scratch_directory,
        ):
            root = Path(directory).resolve(strict=True)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            scratch = Path(scratch_directory).resolve(strict=True)
            private = workspace / ".ssh"
            private.mkdir()
            (private / "id_ed25519").write_text("probe-secret", encoding="utf-8")
            canary = outside / "canary"
            canary.write_text("keep", encoding="utf-8")
            request = SandboxRequest.create(
                source=(
                    "printf allowed > allowed; "
                    "if /bin/cat .ssh/id_ed25519 >/dev/null 2>&1; "
                    "then printf leaked > leaked; fi; "
                    f"printf overwritten > {shlex.quote(str(canary))}"
                ),
                cwd=workspace,
                workspace=workspace,
                read_roots=(),
                write_roots=(workspace,),
                network=False,
                environment={"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(scratch)},
                policy_version=backend.policy_version,
                deadline=time.monotonic() + 5,
                workspace_id="native-probe",
            )
            result = backend.run(request)
            enforced = (
                result.outcome is SandboxOutcome.COMPLETED
                and result.exit_code != 0
                and (workspace / "allowed").read_text(encoding="utf-8") == "allowed"
                and not (workspace / "leaked").exists()
                and canary.read_text(encoding="utf-8") == "keep"
            )
    except (OSError, ValueError):
        reason = "Native sandbox capability probe could not be prepared."
    else:
        if enforced:
            return backend
        reason = (
            "Nested sandbox prevented native sandbox initialization."
            if "sandbox_apply" in result.detail
            else f"Native sandbox capability probe failed ({result.outcome.value})."
        )
    if host == "Linux":
        return UnavailableSandboxBackend(
            CommandProcessResult(
                SandboxOutcome.INVALID,
                detail=reason,
            )
        )
    backend.close()
    return UnavailableSandboxBackend(reason)
