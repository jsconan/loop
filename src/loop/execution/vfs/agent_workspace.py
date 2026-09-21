"""Own isolated logical workspace lineages for agent executions."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...utils import json_encode, matches_digest, sha256_digest
from .materializer import GenerationView, WorkspaceMaterializer
from .models import (
    AffectedPathIdentity,
    BaseSnapshotId,
    BranchId,
    CanonicalDelta,
    GenerationId,
    LineageEpoch,
)


class AgentWorkspaceContext(BaseModel):
    """Describe one agent's host-path-free logical workspace state.

    Args:
        workspace_id (str): Authenticated workspace identity.
        agent_run_id (str): Owning agent execution identity.
        base_snapshot_id (BaseSnapshotId): Immutable shared base snapshot.
        branch_id (BranchId): Opaque private branch identity.
        generation_id (GenerationId): Current private branch generation.
        lineage_epoch (LineageEpoch): Explicit synchronization lineage epoch.
        committed_deltas (tuple[CanonicalDelta, ...]): Successfully published agent effects.
        committed_transaction_ids (tuple[str, ...]): Idempotency keys for those effects.
        affected_path_identities (tuple[AffectedPathIdentity, ...]): Current host identities for
            paths changed within this private lineage.
    """

    model_config = ConfigDict(frozen=True)

    workspace_id: str = Field(min_length=1)
    agent_run_id: str = Field(min_length=1)
    base_snapshot_id: BaseSnapshotId
    branch_id: BranchId
    generation_id: GenerationId
    lineage_epoch: LineageEpoch
    committed_deltas: tuple[CanonicalDelta, ...] = ()
    committed_transaction_ids: tuple[str, ...] = ()
    affected_path_identities: tuple[AffectedPathIdentity, ...] = ()

    @model_validator(mode="after")
    def validate_committed_transactions(self) -> AgentWorkspaceContext:
        """Require a unique idempotency identity for every committed delta.

        Returns:
            AgentWorkspaceContext: Validated immutable lineage state.

        Raises:
            ValueError: Delta and transaction identities differ or identities repeat.
        """
        if len(self.committed_deltas) != len(self.committed_transaction_ids):
            raise ValueError("Committed deltas and transaction identities must correspond.")
        if len(set(self.committed_transaction_ids)) != len(self.committed_transaction_ids):
            raise ValueError("Committed transaction identities must be unique.")
        return self


class AgentWorkspaceCorruptionError(RuntimeError):
    """Report malformed or checksum-invalid durable agent lineage state."""


class AgentWorkspaceState(BaseModel):
    """Carry one versioned durable agent workspace context."""

    model_config = ConfigDict(frozen=True)

    format_version: int = Field(default=1, ge=1, le=1)
    context: AgentWorkspaceContext


class AgentWorkspaceManager:
    """Manage independent agent branches through an injected materializer.

    Production callers must provide the platform materializer. This manager deliberately
    owns logical identifiers and synchronization only; it never sees host paths or mount details.

    Args:
        materializer (WorkspaceMaterializer): Platform implementation for opaque branch actions.
        published_commit_verifier (Callable[[AgentWorkspaceContext], bool]): Trusted verifier used
            before a refresh drops an existing private branch.
        state_directory (Path): Loop-private directory for recoverable lineage state.
    """

    materializer: WorkspaceMaterializer
    published_commit_verifier: Callable[[AgentWorkspaceContext], bool]
    state_directory: Path
    _contexts: dict[tuple[str, str], AgentWorkspaceContext]
    _locks: dict[tuple[str, str], RLock]
    _leases: dict[tuple[str, str], int]
    _registry_lock: RLock

    def __init__(
        self,
        materializer: WorkspaceMaterializer,
        published_commit_verifier: Callable[[AgentWorkspaceContext], bool],
        state_directory: Path,
    ) -> None:
        """Restore durable agent-local lineage state."""
        self.materializer = materializer
        self.published_commit_verifier = published_commit_verifier
        self.state_directory = state_directory
        self.state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._registry_lock = RLock()
        contexts = self._load_contexts()
        self._contexts = {
            (context.workspace_id, context.agent_run_id): context for context in contexts
        }
        for context in contexts:
            self.materializer.recover_branch(
                context.base_snapshot_id,
                context.branch_id,
                tuple(
                    zip(
                        context.committed_transaction_ids,
                        context.committed_deltas,
                        strict=True,
                    )
                ),
            )
        self._locks = {key: RLock() for key in self._contexts}
        self._leases = dict.fromkeys(self._contexts, 0)

    def create(
        self,
        workspace_id: str,
        agent_run_id: str,
        base_snapshot_id: BaseSnapshotId,
    ) -> AgentWorkspaceContext:
        """Create an isolated initial branch for an agent.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.
            base_snapshot_id (BaseSnapshotId): Base snapshot used to create the branch.

        Returns:
            AgentWorkspaceContext: Newly created agent workspace context.

        Raises:
            ValueError: If this agent already has a workspace context.
        """
        with self._registry_lock:
            key = (workspace_id, agent_run_id)
            if key in self._contexts:
                raise ValueError("Agent workspace context already exists.")
            context = AgentWorkspaceContext(
                workspace_id=workspace_id,
                agent_run_id=agent_run_id,
                base_snapshot_id=base_snapshot_id,
                branch_id=self.materializer.create_branch(base_snapshot_id),
                generation_id=GenerationId(value="0"),
                lineage_epoch=LineageEpoch(value="0"),
            )
            self._persist_context(context)
            self._contexts[key] = context
            self._locks[key] = RLock()
            self._leases[key] = 0
            return context

    def get(self, workspace_id: str, agent_run_id: str) -> AgentWorkspaceContext:
        """Return an existing agent context.

        Raises:
            KeyError: If no matching agent context exists.
        """
        return self._contexts[(workspace_id, agent_run_id)]

    def require_publishable(self, context: AgentWorkspaceContext) -> None:
        """Require the exact current generation to be free of active attempt leases.

        Args:
            context (AgentWorkspaceContext): Generation proposed for host publication.

        Raises:
            RuntimeError: The generation changed or still has an active attempt lease.
        """
        key = (context.workspace_id, context.agent_run_id)
        with self._locks[key]:
            if self._contexts[key] != context:
                raise RuntimeError("Cannot publish from a stale workspace generation.")
            if self._leases[key]:
                raise RuntimeError("Cannot publish with active generation leases.")

    @contextmanager
    def lease_generation(
        self,
        workspace_id: str,
        agent_run_id: str,
    ) -> Generator[tuple[AgentWorkspaceContext, GenerationView], None, None]:
        """Lease the current immutable generation and return its opaque view.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.

        Returns:
            Generator[tuple[AgentWorkspaceContext, GenerationView], None, None]: Context and
                opaque generation view.
        """
        key = (workspace_id, agent_run_id)
        lock = self._locks[key]
        with lock:
            context = self._contexts[key]
            self._leases[key] += 1
            view = self.materializer.generation_view(context.branch_id, context.generation_id)
        try:
            yield context, view
        finally:
            with lock:
                self._leases[key] -= 1

    def fork(
        self,
        parent_workspace_id: str,
        parent_agent_run_id: str,
        child_agent_run_id: str,
    ) -> AgentWorkspaceContext:
        """Fork one parent's committed view into a distinct child branch.

        Args:
            parent_workspace_id (str): Authenticated workspace identity of the parent.
            parent_agent_run_id (str): Owning agent identity of the parent.
            child_agent_run_id (str): Owning agent identity of the child.

        Returns:
            AgentWorkspaceContext: Newly created child agent workspace context.
        """
        with self._registry_lock:
            parent_key = (parent_workspace_id, parent_agent_run_id)
            with self._locks[parent_key]:
                parent = self.get(parent_workspace_id, parent_agent_run_id)
                return self._create_fork(parent, child_agent_run_id)

    def _create_fork(
        self,
        parent: AgentWorkspaceContext,
        child_agent_run_id: str,
    ) -> AgentWorkspaceContext:
        """Create a child context after replaying only durable parent effects."""
        key = (parent.workspace_id, child_agent_run_id)
        if key in self._contexts:
            raise ValueError("Agent workspace context already exists.")
        context = AgentWorkspaceContext(
            workspace_id=parent.workspace_id,
            agent_run_id=child_agent_run_id,
            base_snapshot_id=parent.base_snapshot_id,
            branch_id=self.materializer.fork_branch(
                parent.base_snapshot_id,
                tuple(
                    zip(
                        parent.committed_transaction_ids,
                        parent.committed_deltas,
                        strict=True,
                    )
                ),
            ),
            generation_id=GenerationId(value=str(len(parent.committed_deltas))),
            lineage_epoch=parent.lineage_epoch,
            committed_deltas=parent.committed_deltas,
            committed_transaction_ids=parent.committed_transaction_ids,
            affected_path_identities=parent.affected_path_identities,
        )
        self._persist_context(context)
        self._contexts[key] = context
        self._locks[key] = RLock()
        self._leases[key] = 0
        return context

    def refresh(
        self,
        workspace_id: str,
        agent_run_id: str,
        base_snapshot_id: BaseSnapshotId,
    ) -> AgentWorkspaceContext:
        """Replace a quiescent context with an explicit new base and empty branch.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.
            base_snapshot_id (BaseSnapshotId): New base snapshot for the refreshed context.

        Returns:
            AgentWorkspaceContext: Newly refreshed agent workspace context.

        Raises:
            RuntimeError: If an attempt still leases the current generation.
        """
        key = (workspace_id, agent_run_id)
        with self._locks[key]:
            if self._leases[key]:
                raise RuntimeError("Cannot refresh a workspace with active generation leases.")
            previous = self._contexts[key]
            if not self.published_commit_verifier(previous):
                raise RuntimeError("Cannot refresh before published commits are verified.")
            branch = self.materializer.create_branch(base_snapshot_id)
            context = previous.model_copy(
                update={
                    "base_snapshot_id": base_snapshot_id,
                    "branch_id": branch,
                    "generation_id": GenerationId(value="0"),
                    "lineage_epoch": LineageEpoch(value=str(int(previous.lineage_epoch.value) + 1)),
                    "committed_deltas": (),
                    "committed_transaction_ids": (),
                    "affected_path_identities": (),
                }
            )
            self._persist_context(context)
            self._contexts[key] = context
            self.materializer.dispose_branch(previous.branch_id)
            return context

    def advance(
        self,
        workspace_id: str,
        agent_run_id: str,
        transaction_id: str,
        delta: CanonicalDelta,
        resulting_identities: tuple[AffectedPathIdentity, ...] = (),
    ) -> AgentWorkspaceContext:
        """Durably apply one published delta to only its owning agent branch.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.
            transaction_id (str): Transaction being applied.
            delta (CanonicalDelta): Delta to be applied.
            resulting_identities (tuple[AffectedPathIdentity, ...]): Published host identities for
                paths affected by the transaction.

        Returns:
            AgentWorkspaceContext: Updated agent workspace context.

        Raises:
            RuntimeError: If an attempt still leases the generation being advanced.
        """
        key = (workspace_id, agent_run_id)
        with self._locks[key]:
            if self._leases[key]:
                raise RuntimeError("Cannot advance a workspace with active generation leases.")
            context = self._contexts[key]
            if transaction_id in context.committed_transaction_ids:
                return context
            self.materializer.apply_delta(context.branch_id, transaction_id, delta)
            updated = context.model_copy(
                update={
                    "generation_id": GenerationId(value=str(int(context.generation_id.value) + 1)),
                    "committed_deltas": (*context.committed_deltas, delta),
                    "committed_transaction_ids": (
                        *context.committed_transaction_ids,
                        transaction_id,
                    ),
                    "affected_path_identities": _merge_identities(
                        context.affected_path_identities, resulting_identities
                    ),
                }
            )
            self._persist_context(updated)
            self._contexts[key] = updated
            return updated

    def dispose(self, workspace_id: str, agent_run_id: str) -> None:
        """Dispose a quiescent agent branch.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.

        Raises:
            RuntimeError: If an attempt still leases the branch.
        """
        with self._registry_lock:
            key = (workspace_id, agent_run_id)
            with self._locks[key]:
                if self._leases[key]:
                    raise RuntimeError("Cannot dispose a workspace with active generation leases.")
                context = self._contexts[key]
                self._delete_context(context)
                self.materializer.dispose_branch(context.branch_id)
                del self._contexts[key], self._locks[key], self._leases[key]

    def _load_contexts(self) -> tuple[AgentWorkspaceContext, ...]:
        """Load and authenticate all persisted workspace contexts."""
        contexts = []
        for path in self.state_directory.iterdir():
            if not path.is_file() or path.suffix == ".tmp":
                continue
            context = self._load_context(path)
            if path != self._context_path(context.workspace_id, context.agent_run_id):
                raise AgentWorkspaceCorruptionError("Agent workspace identity is invalid.")
            contexts.append(context)
        return tuple(contexts)

    def _load_context(self, path: Path) -> AgentWorkspaceContext:
        """Load and authenticate one durable workspace context record."""
        try:
            raw = path.read_bytes()
            envelope = json.loads(raw)
            payload = envelope["payload"]
            checksum = envelope["checksum"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise AgentWorkspaceCorruptionError("Agent workspace state is malformed.") from error
        encoded = json_encode(payload).encode()
        if not isinstance(checksum, str) or not matches_digest(encoded, checksum):
            raise AgentWorkspaceCorruptionError("Agent workspace state checksum is invalid.")
        try:
            return AgentWorkspaceState.model_validate(payload).context
        except ValueError as error:
            raise AgentWorkspaceCorruptionError("Agent workspace state is invalid.") from error

    def _persist_context(self, context: AgentWorkspaceContext) -> None:
        """Atomically persist and fsync one active logical lineage."""
        state = AgentWorkspaceState(context=context)
        payload = state.model_dump(mode="json")
        encoded = json_encode(payload).encode()
        envelope = json_encode({"checksum": sha256_digest(encoded), "payload": payload}).encode()
        target = self._context_path(context.workspace_id, context.agent_run_id)
        temporary = target.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(descriptor, envelope)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        directory_descriptor = os.open(self.state_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def _delete_context(self, context: AgentWorkspaceContext) -> None:
        """Durably remove one disposed logical lineage record."""
        self._context_path(context.workspace_id, context.agent_run_id).unlink()
        directory_descriptor = os.open(self.state_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def _context_path(self, workspace_id: str, agent_run_id: str) -> Path:
        """Return a confined path for one durable logical lineage identity."""
        identity = json_encode([workspace_id, agent_run_id]).encode()
        return self.state_directory / sha256_digest(identity)


def _merge_identities(
    current: tuple[AffectedPathIdentity, ...],
    changed: tuple[AffectedPathIdentity, ...],
) -> tuple[AffectedPathIdentity, ...]:
    """Merge affected-path identities into deterministic path order."""
    identities = {entry.path: entry for entry in current}
    identities.update({entry.path: entry for entry in changed})
    return tuple(identities[path] for path in sorted(identities))
