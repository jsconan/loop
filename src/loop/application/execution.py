"""Select and compose the ordinary command boundary for the active platform."""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from pathlib import Path

from .. import constants
from ..execution.broker import SecretAuthority
from ..execution.command import SandboxCommandExecutor
from ..execution.infrastructure import InfrastructureProcessRunner
from ..execution.runtime.bootstrap import RuntimeBootstrapper
from ..execution.runtime.download import HttpxArtifactTransport
from ..execution.runtime.image import load_sandbox_image_definition
from ..execution.runtime.install import ArchiveInstaller, FileInstaller
from ..execution.runtime.models import AcquisitionKind
from ..execution.sandbox.macos import (
    MacosProductAdapter,
    MacosSandboxBackend,
    load_macos_runtime_release,
)
from ..execution.sandbox.oci.spec import OciResourceLimits
from ..execution.service import ExecutionService
from ..permissions import ExecutionPermissionAdapter, PermissionManager, SubjectIdentity
from ..utils import sha256_digest
from ..workspace import Workspace
from .paths import ApplicationPaths, WorkspacePaths


def create_command_executor(
    workspace: Workspace,
    paths: ApplicationPaths,
    workspace_paths: WorkspacePaths,
    permissions: PermissionManager,
    agent_run_id: Callable[[], str],
    *,
    progress: Callable[[str], None] = lambda _: None,
    system: str | None = None,
    machine: str | None = None,
    secret_authority: SecretAuthority | None = None,
) -> SandboxCommandExecutor | None:
    """Create the sandbox-only ordinary command executor for a supported host.

    Args:
        workspace (Workspace): Initialized active workspace and publication root.
        paths (ApplicationPaths): Global private application paths.
        workspace_paths (WorkspacePaths): Workspace-scoped durable state paths.
        permissions (PermissionManager): Sole application-facing permission facade.
        agent_run_id (Callable[[], str]): Resolver for the active supervised session identity.
        progress (Callable[[str], None]): User-visible sandbox preparation status callback.
        system (str | None): Platform-system override for isolated tests.
        machine (str | None): CPU-architecture override for isolated tests.
        secret_authority (SecretAuthority | None): Application-owned exact secret resolver retained
            for broker composition. Secret effects remain explicitly unsupported until output
            non-disclosure and cumulative network controls are both enforceable.

    Returns:
        SandboxCommandExecutor | None: macOS sandbox executor, or ``None`` until another platform
        backend is release-qualified.

    Raises:
        ValueError: If the workspace is not initialized.
    """
    if workspace.id is None:
        raise ValueError("Command execution requires an initialized workspace.")
    host_system = platform.system() if system is None else system
    host_machine = platform.machine() if machine is None else machine
    if host_system != "Darwin" or host_machine not in {"arm64", "aarch64"}:
        return None
    manifest = load_macos_runtime_release()
    image = load_sandbox_image_definition()
    runtime_root = paths.data_root / "execution" / "runtime"
    workspace_state = workspace_paths.data / "execution" / "macos"
    control_state = Path("/private/tmp") / (
        f"loop-{os.getuid()}-{sha256_digest(workspace.id.encode())[:12]}"
    )
    bootstrapper = RuntimeBootstrapper(
        manifest,
        runtime_root,
        HttpxArtifactTransport(),
        {
            AcquisitionKind.ARCHIVE: ArchiveInstaller(),
            AcquisitionKind.FILE: FileInstaller(),
        },
        constants.RUNTIME_DEFAULT_DOWNLOAD_LIMIT,
    )
    backend = MacosSandboxBackend(
        manifest,
        bootstrapper,
        InfrastructureProcessRunner(),
        workspace_state,
        workspace.id,
        state_root=control_state,
        progress=progress,
    )
    permission_adapter = ExecutionPermissionAdapter(
        permissions,
        SubjectIdentity(
            tool_id="run_command",
            publisher="loop",
            profile_id="core",
            profile_version=image.source_version,
        ),
        workspace.id,
        constants.EXECUTION_POLICY_VERSION,
        agent_run_id,
    )
    adapter = MacosProductAdapter(
        backend,
        manifest,
        workspace.root,
        workspace_state / "workspace",
        permission_adapter,
        OciResourceLimits(
            memory_bytes=constants.DEFAULT_OCI_MEMORY_BYTES,
            pids=constants.DEFAULT_OCI_PIDS,
            cpu_quota_us=constants.DEFAULT_OCI_CPU_QUOTA_US,
            open_files=constants.DEFAULT_OCI_OPEN_FILES,
        ),
        secret_authority=secret_authority,
    )
    return SandboxCommandExecutor(
        ExecutionService(adapter),
        permission_adapter,
        workspace.id,
        agent_run_id,
        image.source_version,
        runtime_resolver=adapter.ensure_sandbox,
        supports_secret_exposures=False,
    )
