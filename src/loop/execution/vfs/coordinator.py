"""Coordinate host publication with agent-private branch advancement."""

from __future__ import annotations

from .agent_workspace import AgentWorkspaceContext, AgentWorkspaceManager
from .commit_broker import CommitBroker
from .journal import JournalPhase, JournalTransaction, TransactionJournal
from .models import CanonicalDelta


class PublicationCoordinator:
    """Finish an already-authorized publication without runtime authority.

    Args:
        manager (AgentWorkspaceManager): Owner of private agent lineage.
        broker (CommitBroker): Descriptor-confined host publication boundary.
        journal (TransactionJournal): Shared durable publication evidence.
    """

    manager: AgentWorkspaceManager
    broker: CommitBroker
    journal: TransactionJournal

    def __init__(
        self,
        manager: AgentWorkspaceManager,
        broker: CommitBroker,
        journal: TransactionJournal,
    ) -> None:
        """Bind the two ordered publication halves to one journal."""
        self.manager = manager
        self.broker = broker
        self.journal = journal

    def publish(
        self,
        transaction_id: str,
        context: AgentWorkspaceContext,
        delta: CanonicalDelta,
    ) -> AgentWorkspaceContext:
        """Commit a delta to the host then advance exactly the owning agent branch.

        Args:
            transaction_id (str): Transaction being published.
            context (AgentWorkspaceContext): Current agent workspace context.
            delta (CanonicalDelta): Delta to be published and applied.

        Returns:
            AgentWorkspaceContext: Updated agent workspace context.
        """
        transaction = self.broker.commit(
            transaction_id=transaction_id,
            workspace_id=context.workspace_id,
            agent_run_id=context.agent_run_id,
            base_snapshot_id=context.base_snapshot_id,
            generation_id=context.generation_id.value,
            delta=delta,
        )
        return self._advance(transaction)

    def recover(self, transaction_id: str) -> AgentWorkspaceContext | None:
        """Resume branch application for one transaction whose host phase completed.

        Args:
            transaction_id (str): Transaction being recovered.

        Returns:
            AgentWorkspaceContext | None: Updated agent workspace context if recovery is needed,
                otherwise None.
        """
        transaction = self.journal.load(transaction_id)
        if transaction is None:
            return None
        if transaction.phase is JournalPhase.COMMITTED:
            self.broker.finalize(transaction)
            return None
        if transaction.phase is JournalPhase.PREPARED:
            transaction = self.broker.recover(transaction_id)
        if transaction is None:
            return None
        return self._advance(transaction)

    def recover_incomplete(self) -> tuple[AgentWorkspaceContext, ...]:
        """Recover every incomplete durable transaction during application startup.

        Returns:
            tuple[AgentWorkspaceContext, ...]: Agent contexts advanced by deterministic recovery.
        """
        recovered = []
        for transaction in self.journal.transactions():
            context = self.recover(transaction.transaction_id)
            if context is not None:
                recovered.append(context)
        return tuple(recovered)

    def _advance(self, transaction: JournalTransaction) -> AgentWorkspaceContext:
        """Apply host-published data idempotently and durably mark completion."""
        current = self.manager.get(transaction.workspace_id, transaction.agent_run_id)
        already_applied = transaction.transaction_id in current.committed_transaction_ids
        if not already_applied and (
            current.base_snapshot_id != transaction.base_snapshot_id
            or current.generation_id.value != transaction.generation_id
        ):
            raise RuntimeError("Transaction does not match the durable agent generation.")
        context = self.manager.advance(
            transaction.workspace_id,
            transaction.agent_run_id,
            transaction.transaction_id,
            transaction.delta,
            transaction.resulting_identities,
        )
        if transaction.phase is JournalPhase.HOST_APPLIED:
            transaction = transaction.model_copy(update={"phase": JournalPhase.BRANCH_APPLIED})
            self.journal.persist(transaction)
        if transaction.phase is JournalPhase.BRANCH_APPLIED:
            transaction = transaction.model_copy(update={"phase": JournalPhase.COMMITTED})
            self.journal.persist(transaction)
        self.broker.finalize(transaction)
        return context
