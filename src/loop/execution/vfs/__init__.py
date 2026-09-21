"""Provide portable workspace VFS contracts without runtime dependencies."""

__all__ = [
    "AffectedPathIdentity",
    "AgentWorkspaceContext",
    "AgentWorkspaceCorruptionError",
    "AgentWorkspaceManager",
    "AuthenticatedWorkspaceRoot",
    "BaseSnapshotId",
    "BranchId",
    "CanonicalDelta",
    "CanonicalDeltaEntry",
    "CommitBroker",
    "CommitConflictError",
    "ContentReference",
    "DestinationFilesystemCapabilities",
    "FilesystemCapabilities",
    "FilesystemRepresentabilityValidator",
    "GenerationId",
    "GenerationView",
    "HostObjectIdentity",
    "HostPrecondition",
    "JournalCorruptionError",
    "JournalPhase",
    "JournalTransaction",
    "LineageEpoch",
    "MetadataPreservation",
    "ObjectKind",
    "ObservedDelta",
    "ObservedDeltaEntry",
    "ObservedObjectKind",
    "PublicationCoordinator",
    "SanitizedWorkspaceContext",
    "SnapshotBuilder",
    "SnapshotManifest",
    "SnapshotManifestEntry",
    "StagedContentStore",
    "TransactionJournal",
    "UnrepresentableDeltaError",
    "WorkspaceBusyError",
    "WorkspaceMaterializer",
    "WorkspaceSnapshot",
    "normalize_delta",
    "sanitize_workspace_context",
]
from .agent_workspace import (
    AgentWorkspaceContext,
    AgentWorkspaceCorruptionError,
    AgentWorkspaceManager,
)
from .commit_broker import CommitBroker, CommitConflictError
from .coordinator import PublicationCoordinator
from .delta import (
    ObservedDelta,
    ObservedDeltaEntry,
    ObservedObjectKind,
    UnrepresentableDeltaError,
    normalize_delta,
)
from .fs_capabilities import FilesystemCapabilities
from .journal import (
    HostPrecondition,
    JournalCorruptionError,
    JournalPhase,
    JournalTransaction,
    TransactionJournal,
)
from .manifest import SnapshotManifest, SnapshotManifestEntry
from .materializer import GenerationView, WorkspaceMaterializer
from .models import (
    AffectedPathIdentity,
    BaseSnapshotId,
    BranchId,
    CanonicalDelta,
    CanonicalDeltaEntry,
    ContentReference,
    DestinationFilesystemCapabilities,
    GenerationId,
    HostObjectIdentity,
    LineageEpoch,
    MetadataPreservation,
    ObjectKind,
)
from .path_broker import AuthenticatedWorkspaceRoot
from .representability import FilesystemRepresentabilityValidator
from .sanitizer import SanitizedWorkspaceContext, sanitize_workspace_context
from .snapshot import SnapshotBuilder, WorkspaceBusyError, WorkspaceSnapshot
from .staging import StagedContentStore
