"""Verify explicit model context projection and executable local path aliases."""

import pytest

from loop.backend import estimate_context
from loop.backend.utils import project_context, project_portable_context
from loop.models import (
    CompactionContextItem,
    ContentArtifact,
    ContextReference,
    DeveloperInput,
    Message,
    Reasoning,
    ReasoningConfiguration,
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
    assert project_context(CompactionContextItem(data={"content": "checkpoint"}))["data"] == {
        "content": "checkpoint"
    }
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
        project_portable_context(CompactionContextItem(data={"type": "compaction", "id": "id"}))
        is None
    )
    portable = Message(role="user", content="checkpoint")
    assert project_portable_context(portable)["content"] == "checkpoint"


def test_portable_context_omits_model_only_updates_and_opaque_binary_attachment():
    """Portable summaries retain reference metadata without transmitting encoded binary."""
    from loop.backend import project_portable_context
    from loop.models import DeveloperInput, ReasoningConfiguration

    ref = ContextReference(
        kind="file",
        path="image.png",
        media_type="image/png",
        size_bytes=3,
        included_bytes=3,
        truncated=False,
        content=b"\xff\x00\x01",
    )
    message = Message(role="user", content="Attachment", context=(ref,))
    assert project_portable_context(DeveloperInput("State update")) is None
    assert project_portable_context(ReasoningConfiguration("low")) is None
    assert project_portable_context(ToolResult(call_id="pair", output="done"))["output"] == "done"
    assert project_portable_context(message)["context"][0]["content"] is None
    assert estimate_context([message]) >= 1024
    reasoning = Reasoning(content="Working", encrypted_content="opaque" * 100)
    assert estimate_context([reasoning]) == estimate_context([Reasoning(content="Working")])
    assert (
        estimate_context(
            [message.model_copy(update={"context": (ref.model_copy(update={"content": ""}),)})]
        )
        >= 1024
    )
    assert (
        estimate_context(
            [message.model_copy(update={"context": (ref.model_copy(update={"reused": True}),)})]
        )
        < 1024
    )


def test_estimate_budgets_hydrated_text_or_its_recorded_prefix():
    """Fallback sizing uses sanitized text when present and saved byte bounds otherwise."""
    reference = ContextReference(
        kind="file",
        path="notes.txt",
        content="<redacted:secret>",
        included_bytes=900,
        size_bytes=900,
        truncated=False,
        payload_redacted=True,
        media_type="text/plain",
    )
    hydrated = Message(role="user", content="Read this", context=(reference,))
    unhydrated = hydrated.model_copy(
        update={"context": (reference.model_copy(update={"content": None}),)}
    )

    assert estimate_context([hydrated]) < 300 <= estimate_context([unhydrated])


@pytest.mark.parametrize(
    ("item", "wire"),
    [
        (DeveloperInput("rules"), {"role": "developer", "content": "rules"}),
        (
            ReasoningConfiguration("high"),
            {"type": "configuration_update", "reasoning": {"effort": "high"}},
        ),
        (
            ReasoningConfiguration(None),
            {"type": "configuration_update", "reasoning": {"effort": None}},
        ),
    ],
)
def test_request_metadata_has_explicit_wire_shape_and_no_portable_projection(item, wire):
    """Developer instructions and reasoning changes stay out of portable summaries."""
    assert project_context(item) == wire
    assert project_portable_context(item) is None
