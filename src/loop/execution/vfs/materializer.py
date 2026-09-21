"""Define the platform-neutral seam that owns opaque branch materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import BaseSnapshotId, BranchId, CanonicalDelta, GenerationId


@dataclass(frozen=True, slots=True)
class GenerationView:
    """Represent an opaque immutable read-only generation reference.

    Args:
        reference (str): Platform-owned opaque generation reference.
    """

    reference: str = ""


@runtime_checkable
class WorkspaceMaterializer(Protocol):
    """Materialize opaque branches without exposing host paths or runtime handles."""

    def create_branch(self, base: BaseSnapshotId) -> BranchId:
        """Create one private branch over an immutable base."""

    def recover_branch(
        self,
        base: BaseSnapshotId,
        branch: BranchId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> None:
        """Recover or reconstruct one persisted private branch after runtime restart."""

    def generation_view(self, branch: BranchId, generation: GenerationId) -> GenerationView:
        """Return one opaque read-only reference for a branch generation."""

    def apply_delta(self, branch: BranchId, transaction_id: str, delta: CanonicalDelta) -> None:
        """Idempotently apply one canonical delta for a durable transaction."""

    def fork_branch(
        self,
        base: BaseSnapshotId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> BranchId:
        """Create an independent branch by replaying identified committed deltas.

        Args:
            base (BaseSnapshotId): Immutable base shared with the parent lineage.
            committed_transactions (tuple[tuple[str, CanonicalDelta], ...]): Ordered transaction
                identities and canonical deltas to replay idempotently.

        Returns:
            BranchId: Opaque identity of the independent replayed branch.
        """

    def dispose_branch(self, branch: BranchId) -> None:
        """Dispose one opaque private branch after its leases are released."""
