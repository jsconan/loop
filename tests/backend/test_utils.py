"""Verify explicit model context projection and executable local path aliases."""

import pytest

from loop.backend.utils import project_context, project_portable_context
from loop.models import (
    CompactionContextItem,
    ContentArtifact,
    ContextReference,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
)


def test_projection_excludes_local_snapshots_and_artifacts_without_losing_continuation():
    """Only preview bytes and continuation metadata cross the model projection boundary."""
    reference = ContextReference(
        kind="file",
        path="notes.sql",
        content="preview",
        size_bytes=50,
        included_bytes=7,
        truncated=True,
        handle="handle",
        next_cursor="cursor",
    )
    projected = project_context(Message(role="user", content="do the task", context=(reference,)))
    assert projected["content"] == "do the task"
    assert projected["context"][0]["content"] == "preview"
    assert projected["context"][0]["next_cursor"] == "cursor"
    assert "private-tail" not in str(projected)
    result = ToolResult(
        call_id="call",
        output="result",
        artifacts=(
            ContentArtifact(handle="handle", source="local-private-path", reloadable=False),
        ),
    )
    assert project_context(result) == {"call_id": "call", "output": "result"}
    call = ToolCall(call_id="call", name="demo", arguments="{}")
    assert project_context(call)["arguments"] == "{}"
    assert (
        project_context(Reasoning(content="reason", encrypted_content="opaque"))[
            "encrypted_content"
        ]
        == "opaque"
    )
    assert project_context(CompactionContextItem(provider="loop", data={"content": "checkpoint"}))[
        "data"
    ] == {"content": "checkpoint"}
    with pytest.raises(TypeError, match="Unsupported conversation item"):
        project_context({})


def test_portable_projection_omits_provider_replay_state_but_keeps_semantics():
    """Portable summaries keep semantic checkpoints without opaque protocol state."""
    reasoning = Reasoning(
        content="decision", summary="summary", encrypted_content="cipher", id="provider-id"
    )
    assert project_portable_context(reasoning) == {
        "content": "decision",
        "summary": "summary",
    }
    assert project_portable_context(ToolCall(call_id="call", name="tool", arguments="{}")) == {
        "call_id": "call",
        "name": "tool",
        "arguments": "{}",
    }
    assert (
        project_portable_context(
            CompactionContextItem(provider="openai", data={"type": "compaction", "id": "id"})
        )
        is None
    )
    assert project_portable_context(
        CompactionContextItem(provider="loop", data={"content": "checkpoint", "opaque": "private"})
    ) == {"provider": "loop", "data": {"content": "checkpoint"}}
