"""Expose macOS-specific managed-runtime requirement checks."""

__all__ = [
    "DurableJobState",
    "DurableJobStatus",
    "DurableJobStore",
    "LimaClient",
    "LimaConfigurationError",
    "LimaInstanceConfiguration",
    "LimaInstanceContext",
    "LimaLifecycleState",
    "MacosAttemptLayer",
    "MacosBrokerError",
    "MacosBrokerLease",
    "MacosCandidateError",
    "MacosControlPlane",
    "MacosDeltaArchiveInspector",
    "MacosDurableJobManager",
    "MacosEffectBroker",
    "MacosExecutionAdapter",
    "MacosLeasedWorkspace",
    "MacosProductAdapter",
    "MacosRequirementError",
    "MacosRequirementEvidence",
    "MacosRequirements",
    "MacosSandboxBackend",
    "MacosWorkspaceCoordinator",
    "MacosWorkspaceMaterializer",
    "MacosWorkspaceProvider",
    "ManagedLimaInstance",
    "PreparedMacosRuntime",
    "bootstrap_lima_executable",
    "build_lima_instance_configuration",
    "load_macos_runtime_candidate",
    "load_macos_runtime_release",
    "macos_artifact",
    "macos_component_version",
    "macos_entitlement_keys",
]

from .backend import MacosSandboxBackend, PreparedMacosRuntime
from .broker import MacosBrokerError, MacosBrokerLease, MacosEffectBroker
from .candidate import (
    MacosCandidateError,
    load_macos_runtime_candidate,
    load_macos_runtime_release,
    macos_artifact,
    macos_component_version,
    macos_entitlement_keys,
)
from .composition import MacosProductAdapter
from .control_plane import MacosControlPlane
from .execution import MacosExecutionAdapter, MacosWorkspaceProvider
from .jobs import (
    DurableJobState,
    DurableJobStatus,
    DurableJobStore,
    MacosDurableJobManager,
)
from .lima import (
    LimaClient,
    LimaConfigurationError,
    LimaInstanceConfiguration,
    LimaInstanceContext,
    LimaLifecycleState,
    ManagedLimaInstance,
    bootstrap_lima_executable,
    build_lima_instance_configuration,
)
from .requirements import MacosRequirementError, MacosRequirementEvidence, MacosRequirements
from .workspace import (
    MacosAttemptLayer,
    MacosDeltaArchiveInspector,
    MacosLeasedWorkspace,
    MacosWorkspaceCoordinator,
    MacosWorkspaceMaterializer,
)
