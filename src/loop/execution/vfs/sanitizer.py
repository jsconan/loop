"""Produce model-safe workspace lineage summaries without branch internals."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .agent_workspace import AgentWorkspaceContext


class SanitizedWorkspaceContext(BaseModel):
    """Describe model-safe logical workspace lineage state.

    Args:
        workspace_id (str): Opaque authenticated workspace identity.
        agent_run_id (str): Owning agent execution identity.
        generation_id (str): Current logical generation identifier.
        lineage_epoch (str): Explicit synchronization lineage epoch.
    """

    model_config = ConfigDict(frozen=True)

    workspace_id: str = Field(min_length=1)
    agent_run_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    lineage_epoch: str = Field(min_length=1)


def sanitize_workspace_context(context: AgentWorkspaceContext) -> SanitizedWorkspaceContext:
    """Remove private branch, snapshot, delta, and transaction identifiers from a context.

    Args:
        context (AgentWorkspaceContext): Privileged logical lineage state.

    Returns:
        SanitizedWorkspaceContext: Stable model-safe lineage summary.
    """
    return SanitizedWorkspaceContext(
        workspace_id=context.workspace_id,
        agent_run_id=context.agent_run_id,
        generation_id=context.generation_id.value,
        lineage_epoch=context.lineage_epoch.value,
    )
