"""Select a qualified built-in command sandbox for this host."""

from __future__ import annotations

import os
import platform
import shlex
import tempfile
import time
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from threading import Event
from types import ModuleType
from typing import TYPE_CHECKING

from .constants import (
    COMMAND_SANDBOX_SCRATCH_PREFIX,
    LANG_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    SYSTEM_LOCALE,
    TMPDIR_ENVIRONMENT_VARIABLE,
)
from .contracts import (
    CommandProcessResult,
    QualificationProbeLayout,
    SandboxBackend,
    SandboxCapability,
    SandboxOutcome,
    SandboxRequest,
)

if TYPE_CHECKING:
    from ...permissions.manager import PermissionManager


_BACKEND_MODULES = {"Linux": ".linux.backend", "Darwin": ".macos.backend"}


def selected_backend_module() -> ModuleType | None:
    """Load only the reviewed native backend module for this host.

    Returns:
        ModuleType | None: Matching backend module, or None for an unsupported host.
    """
    name = _BACKEND_MODULES.get(platform.system())
    return import_module(name, __package__) if name is not None else None


class UnavailableSandboxBackend:
    """Reject command execution when this host has no qualified native backend.

    Args:
        reason (str | CommandProcessResult): Path-free capability explanation or typed integrity
            failure; invalid/stale failures never permit a host retry.
    """

    _reason: str | CommandProcessResult
    capabilities = frozenset()
    qualification_capabilities = frozenset()

    def __init__(self, reason: str | CommandProcessResult) -> None:
        self._reason = reason

    def qualification_layout(self) -> QualificationProbeLayout:
        """Return no probe parents when native execution is unavailable.

        Returns:
            QualificationProbeLayout: Default placement unused by this backend.
        """
        return QualificationProbeLayout(None, None)

    def qualification_failure(
        self, result: CommandProcessResult | None, reason: str
    ) -> str | CommandProcessResult:
        """Retain the original unavailable reason instead of reclassifying it.

        Args:
            result (CommandProcessResult | None): Unused native probe result.
            reason (str): Unused failure explanation.

        Returns:
            str | CommandProcessResult: Original unavailable reason.
        """
        del result, reason
        return self._reason

    def close(self) -> None:
        """Release no resources from an unavailable backend."""

    @property
    def trusted_file_owner_ids(self) -> tuple[int, ...]:
        """Return file owners trusted for automatic executable binding.

        Returns:
            tuple[int, ...]: Current user and any verified host administrator identity.
        """
        return (os.getuid(),)

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
        return COMMAND_SANDBOX_SCRATCH_PREFIX

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

    def qualification_probe(self, request: SandboxRequest) -> CommandProcessResult:
        """Reject a probe when no native sandbox is available.

        Args:
            request (SandboxRequest): Probe request that cannot run.

        Returns:
            CommandProcessResult: The original typed unavailability result.
        """
        return self.run(request)

    def complete_qualification(self, request: SandboxRequest) -> None:
        """Keep an unavailable backend without advertised capabilities.

        Args:
            request (SandboxRequest): Unused probe request.
        """
        del request

    def run(
        self,
        request: SandboxRequest,
        *,
        cancellation_event: Event | None = None,
        policy_authority: PermissionManager | None = None,
        approved_revision: int | None = None,
    ) -> CommandProcessResult:
        """Return a typed failure without launching a child.

        Args:
            request (SandboxRequest): Command that cannot be executed on this host.
            cancellation_event (Event | None): Optional run cancellation signal.
            policy_authority (PermissionManager | None): Unused unavailable mediator authority.
            approved_revision (int | None): Unused unavailable revision.

        Returns:
            CommandProcessResult: Original typed capability or integrity failure.
        """
        del policy_authority, approved_revision
        if isinstance(self._reason, CommandProcessResult):
            return self._reason
        return CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            detail=self._reason,
            failure_context="Native sandbox capability check failed.",
        )


def select_sandbox_backend() -> SandboxBackend:
    """Select and independently probe the native backend for this host.

    Bind probe scratch explicitly and let each backend place its protected probe workspace.

    Returns:
        SandboxBackend: Available native backend or a fail-closed unavailable backend.
    """
    backend_module = selected_backend_module()
    if backend_module is None:
        return UnavailableSandboxBackend("No native command sandbox is qualified for this host.")
    backend = backend_module.create_backend()
    failure = backend.capability_failure()
    if failure is not None:
        return UnavailableSandboxBackend(failure)
    result = None
    try:
        with ExitStack() as directories:
            layout = backend.qualification_layout()
            directory = directories.enter_context(
                tempfile.TemporaryDirectory(
                    prefix=f"{COMMAND_SANDBOX_SCRATCH_PREFIX}probe-",
                    dir=layout.probe_parent,
                )
            )
            root = Path(directory).resolve(strict=True)
            workspace = root / "workspace"
            workspace.mkdir()
            if layout.outside_parent is None:
                outside = root / "outside"
                outside.mkdir()
            else:
                outside = Path(
                    directories.enter_context(
                        tempfile.TemporaryDirectory(
                            prefix=f"{COMMAND_SANDBOX_SCRATCH_PREFIX}outside-",
                            dir=layout.outside_parent,
                        )
                    )
                ).resolve(strict=True)
            scratch_directory = directories.enter_context(
                tempfile.TemporaryDirectory(prefix=backend.scratch_prefix)
            )
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
                local_network=(
                    SandboxCapability.LOCAL_NETWORK in backend.qualification_capabilities
                ),
                environment={
                    PATH_ENVIRONMENT_VARIABLE: MINIMAL_SYSTEM_PATH,
                    LANG_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                    TMPDIR_ENVIRONMENT_VARIABLE: str(scratch),
                },
                policy_version=backend.policy_version,
                deadline=time.monotonic() + 5,
                workspace_id="native-probe",
                scratch_root=scratch,
            )
            result = backend.qualification_probe(request)
            enforced = (
                result.outcome is SandboxOutcome.COMPLETED
                and result.exit_code != 0
                and (workspace / "allowed").read_text(encoding="utf-8") == "allowed"
                and not (workspace / "leaked").exists()
                and canary.read_text(encoding="utf-8") == "keep"
            )
            if enforced:
                backend.complete_qualification(request)
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
    backend.close()
    return UnavailableSandboxBackend(backend.qualification_failure(result, reason))
