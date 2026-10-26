"""Verify session transcript and archive exports."""

import json
import stat
from datetime import UTC, datetime
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from loop import (
    Action,
    ApprovalChoice,
    CompactionEvent,
    ContextReference,
    ConversationItemEvent,
    Decision,
    FileTarget,
    MemorySessionStore,
    Message,
    Operation,
    PermissionEvent,
    PolicyDecision,
    Reasoning,
    RunCompletedEvent,
    RunMetrics,
    Session,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolResult,
    export_session,
)
from loop.permissions import AuthorizationResult
from loop.session import SessionStore, UnsupportedConversationItemError
from loop.session import exporter as exporter_module
from loop.session.models import SessionPersistenceSnapshot
from loop.utils import content_digest, content_identity, encode_content_cursor


def _context_reference(content: bytes) -> ContextReference:
    """Build a complete immutable file reference for one session message."""
    handle, digest = content_identity(content)
    text = content.decode("utf-8")
    return ContextReference(
        kind="file",
        path="inputs/source.txt",
        content=text,
        size_bytes=len(content),
        included_bytes=len(content),
        truncated=False,
        handle=handle,
        version=digest,
        media_type="text/plain",
    )


def _permission_event(approval: ApprovalChoice | None = None) -> PermissionEvent:
    """Build one prompted permission event for transcript output."""
    operation = Operation(
        tool_id="read",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path="/workspace/source.txt"),
    )
    return PermissionEvent(
        id=f"permission-event-{approval or 'none'}",
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
        result=AuthorizationResult(
            operations=(operation,),
            policy=PolicyDecision(
                decision=Decision.ASK,
                reason="approval required",
                sources=("default:filesystem.read",),
            ),
            decision=Decision.ALLOW,
            prompted=True,
            prompt="Private prompt details",
            reason="allowed",
            source="user",
            approval_choice=approval,
        ),
    )


def _external_reference_session(content: bytes) -> Session:
    """Return a session whose serialized reference content must come from its store."""
    session = Session(
        messages=[
            Message(role="user", content="Inspect this.", context=(_context_reference(content),))
        ]
    )
    return Session.deserialize(session.serialize())


def test_markdown_export_renders_timeline_and_omits_private_payloads(tmp_path):
    """Markdown prefers reasoning summaries and omits captured context and approval prompts."""
    session = Session(
        name="Transcript\nTitle",
        name_source="user",
        model="served-model",
        messages=[
            Message(
                role="user",
                content="Question",
                context=(_context_reference(b"captured file secret"),),
            ),
            Reasoning(content="private reasoning content", summary="Visible summary"),
            Message(role="assistant", content="Answer"),
            ToolCall(call_id="call-1", name="search", arguments='{"q":"```"}'),
            ToolResult(call_id="call-1", output="result ``` with details"),
        ],
    )
    item_events = {
        event.item_index: event
        for event in session.events
        if isinstance(event, ConversationItemEvent)
    }
    session.events = [
        item_events[0],
        CompactionEvent(
            id="compaction-event",
            created_at=datetime(2026, 8, 20, tzinfo=UTC),
            compaction_index=0,
        ),
        item_events[1],
        _permission_event(ApprovalChoice.SESSION),
        _permission_event(),
        item_events[2],
        item_events[3],
        ToolExecutionStartedEvent(
            id="tool-start",
            created_at=datetime(2026, 8, 20, tzinfo=UTC),
            call_id="call-1",
        ),
        ToolExecutionCompletedEvent(
            id="tool-end",
            created_at=datetime(2026, 8, 20, tzinfo=UTC),
            call_id="call-1",
            succeeded=True,
            duration_seconds=0.1,
        ),
        item_events[4],
        RunCompletedEvent(
            id="run-event",
            created_at=datetime(2026, 8, 20, tzinfo=UTC),
            stop_reason="completed",
            started_at=datetime(2026, 8, 20, tzinfo=UTC),
            metrics=RunMetrics(
                active_duration_seconds=1,
                model_duration_seconds=1,
                tool_duration_seconds=0,
                message_count=2,
                item_count=5,
            ),
        ),
    ]
    target = tmp_path / "transcript.md"

    export_session(session, MemorySessionStore(), target)

    transcript = target.read_text(encoding="utf-8")
    assert transcript.startswith("# Transcript Title\n")
    assert "## User\n\nQuestion" in transcript
    assert "## Assistant\n\nAnswer" in transcript
    assert "Visible summary" in transcript
    assert "### Reasoning summary\n\nVisible summary\n" in transcript
    assert "private reasoning content" not in transcript
    assert "captured file secret" not in transcript
    assert "Private prompt details" not in transcript
    assert "Outcome: allow (session)" in transcript
    assert "Outcome: allow\n" in transcript
    assert "Context compacted" in transcript
    assert "Stop reason: completed" in transcript
    assert '````json\n{"q":"```"}\n````' in transcript
    assert "````text\nresult ``` with details\n````" in transcript
    assert transcript.index("## User") < transcript.index("Context compacted")
    assert transcript.index("Context compacted") < transcript.index("Visible summary")
    assert transcript.index("Visible summary") < transcript.index("Outcome: allow (session)")
    assert transcript.index("Outcome: allow (session)") < transcript.index("## Assistant")
    assert transcript.index("### Tool result") < transcript.index("Run completed")


def test_markdown_export_omits_empty_reasoning_and_marks_unplaced_calls(tmp_path):
    """Markdown omits empty reasoning and makes unplaced tool calls visible."""
    session = Session(
        messages=[
            Reasoning(content="", encrypted_content="private"),
            ToolCall(call_id="pending", name="inspect", arguments="{}"),
        ],
        events=[
            ToolExecutionStartedEvent(
                id="tool-start",
                created_at=datetime(2026, 8, 20, tzinfo=UTC),
                call_id="pending",
            )
        ],
    )
    target = tmp_path / "pending.md"

    export_session(session, MemorySessionStore(), target)

    transcript = target.read_text(encoding="utf-8")
    assert "private" not in transcript
    assert "Pending tool call: inspect" in transcript


@pytest.mark.parametrize("recorded", [True, False])
def test_markdown_export_includes_reasoning_text_without_a_summary(tmp_path, recorded):
    """Markdown preserves displayed reasoning from recorded and unplaced session items."""
    session = Session(
        messages=[Reasoning(content="Visible reasoning", encrypted_content="opaque-state")]
    )
    session = Session.deserialize(session.serialize())
    if not recorded:
        session.events = []
    target = tmp_path / "reasoning.md"

    export_session(session, MemorySessionStore(), target)

    transcript = target.read_text(encoding="utf-8")
    assert "### Reasoning\n\nVisible reasoning\n" in transcript
    assert "opaque-state" not in transcript


def test_markdown_export_formats_json_tool_results_for_readability(tmp_path):
    """Markdown pretty-prints structured tool results as syntax-highlighted JSON."""
    session = Session(
        messages=[
            ToolResult(
                call_id="structured-result",
                output='{"ok":true,"result":{"answer":42,"items":[1,2]}}',
            )
        ]
    )
    target = tmp_path / "tool-result.md"

    export_session(session, MemorySessionStore(), target)

    transcript = target.read_text(encoding="utf-8")
    assert (
        '```json\n{\n  "ok": true,\n  "result": {\n    "answer": 42,\n'
        '    "items": [\n      1,\n      2\n    ]\n  }\n}\n```'
    ) in transcript


def test_markdown_export_uses_fallback_metadata_and_rejects_unknown_items(tmp_path):
    """Markdown supplies neutral metadata defaults and rejects unsupported item types."""
    target = tmp_path / "empty.md"
    export_session(Session(), MemorySessionStore(), target)
    assert "# Untitled session" in target.read_text(encoding="utf-8")
    assert "- Model: `unknown`" in target.read_text(encoding="utf-8")

    invalid = Session()
    invalid.messages.append(object())
    with pytest.raises(UnsupportedConversationItemError, match="Unsupported conversation item"):
        export_session(invalid, MemorySessionStore(), tmp_path / "invalid.md")


@pytest.mark.parametrize("format", ["markdown", "archive"])
def test_export_preserves_sessions_with_malformed_unicode_identifiers(tmp_path, format):
    """Export safely represents an identifier containing an unpaired surrogate."""
    session_id = chr(0xD800)
    session = Session(id=session_id)
    target = tmp_path / ("session.md" if format == "markdown" else "session.zip")

    export_session(session, MemorySessionStore(), target, format=format)

    if format == "markdown":
        assert "\\ud800" in target.read_text(encoding="utf-8")
    else:
        with ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            serialized = json.loads(archive.read("session.json"))
        assert manifest["session_id"] == session_id
        assert serialized["id"] == session_id


def test_archive_export_contains_versioned_session_and_verified_artifacts(tmp_path):
    """Archive export includes the session snapshot, manifest, and transient reference bytes."""
    content = b"captured reference bytes"
    reference = _context_reference(content)
    session = Session(
        messages=[
            Message(role="user", content="Question", context=(reference,)),
            Reasoning(content="archive continuation state", encrypted_content="encrypted-state"),
        ]
    )
    target = tmp_path / "nested" / "session.zip"

    export_session(session, MemorySessionStore(), target, format="archive")

    with ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        serialized = archive.read("session.json").decode("utf-8")
        assert manifest["format"] == "loop-session-archive"
        assert manifest["format_version"] == 1
        assert manifest["session_id"] == session.id
        assert json.loads(serialized)["id"] == session.id
        assert len(manifest["artifacts"]) == 1
        artifact = manifest["artifacts"][0]
        assert artifact["digest"] == reference.version
        assert artifact["size_bytes"] == len(content)
        assert archive.read(artifact["path"]) == content
        assert Session.deserialize(serialized).id == session.id
        assert "archive continuation state" in serialized
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_archive_export_loads_persisted_artifacts_by_digest(tmp_path):
    """Archive export resolves content omitted from the session payload through its store."""
    content = b"stored reference bytes"
    session = _external_reference_session(content)
    store = Mock(spec=SessionStore)
    store.load_reference.return_value = content
    target = tmp_path / "session.zip"

    export_session(session, store, target, format="archive")

    store.load_reference.assert_called_once_with(content_digest(content))
    with ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read(manifest["artifacts"][0]["path"]) == content


def test_archive_export_preserves_empty_artifacts(tmp_path):
    """Archive export retains valid zero-byte references without requiring a store blob."""
    handle, digest = content_identity(b"")
    reference = ContextReference(
        kind="file",
        path="empty.txt",
        content="",
        size_bytes=0,
        included_bytes=0,
        truncated=False,
        handle=handle,
        version=digest,
    )
    session = Session(messages=[Message(role="user", content="Empty", context=(reference,))])
    store = Mock(spec=SessionStore)
    target = tmp_path / "empty.zip"

    export_session(session, store, target, format="archive")

    store.load_reference.assert_not_called()
    with ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read(manifest["artifacts"][0]["path"]) == b""


def test_archive_export_includes_empty_references_without_a_store_blob(tmp_path, monkeypatch):
    """Archive export reconstructs a valid empty snapshot when no durable blob exists."""
    reference = _context_reference(b"")
    session = Session(messages=[Message(role="user", content="Empty", context=(reference,))])
    payload = session.serialize()
    monkeypatch.setattr(
        session,
        "persistence_snapshot",
        lambda: SessionPersistenceSnapshot(payload=payload, reference_artifacts=()),
    )
    store = Mock(spec=SessionStore)
    target = tmp_path / "empty-no-blob.zip"

    export_session(session, store, target, format="archive")

    store.load_reference.assert_not_called()
    with ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read(manifest["artifacts"][0]["path"]) == b""


def test_archive_export_writes_a_manifest_without_referenced_artifacts(tmp_path):
    """Archive export keeps a valid empty artifact catalog for ordinary sessions."""
    target = tmp_path / "plain.zip"

    export_session(Session(), MemorySessionStore(), target, format="archive")

    with ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["artifacts"] == []
        assert archive.namelist() == ["manifest.json", "session.json"]


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        (None, "is unavailable"),
        ("not bytes", "integrity check"),
        (b"different bytes", "integrity check"),
    ],
)
def test_archive_export_rejects_unavailable_or_corrupt_references(
    tmp_path,
    stored,
    message,
):
    """Archive export fails instead of silently omitting missing or corrupt reference bytes."""
    content = b"expected bytes"
    session = _external_reference_session(content)
    store = Mock(spec=SessionStore)
    store.load_reference.return_value = stored

    with pytest.raises(ValueError, match=message):
        export_session(session, store, tmp_path / "invalid.zip", format="archive")

    assert not (tmp_path / "invalid.zip").exists()


def test_archive_export_rejects_artifact_size_mismatch(tmp_path):
    """Archive export validates stored content size even when its digest is correct."""
    content = b"full bytes"
    handle, digest = content_identity(content)
    reference = ContextReference(
        kind="file",
        path="truncated.txt",
        content="",
        size_bytes=len(content) + 1,
        included_bytes=0,
        truncated=True,
        handle=handle,
        next_cursor=encode_content_cursor(handle, 0),
        version=digest,
    )
    session = Session.deserialize(
        Session(
            messages=[Message(role="user", content="Question", context=(reference,))]
        ).serialize()
    )
    store = Mock(spec=SessionStore)
    store.load_reference.return_value = content

    with pytest.raises(ValueError, match="integrity check"):
        export_session(session, store, tmp_path / "mismatch.zip", format="archive")


def test_archive_export_rejects_incomplete_artifact_identity(tmp_path, monkeypatch):
    """Archive export reports missing digest identity instead of inventing a blob name."""
    session = Session(
        messages=[
            Message(
                role="user",
                content="Question",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        content="",
                        size_bytes=0,
                        included_bytes=0,
                        truncated=False,
                        handle="0" * 32,
                        version=None,
                    ),
                ),
            )
        ]
    )
    monkeypatch.setattr(
        session,
        "persistence_snapshot",
        lambda: SessionPersistenceSnapshot(payload="{}", reference_artifacts=()),
    )

    with pytest.raises(ValueError, match="incomplete identity"):
        export_session(session, MemorySessionStore(), tmp_path / "incomplete.zip", format="archive")


def test_export_refuses_overwrite_by_default_and_replaces_when_authorized(tmp_path):
    """Export files are private, exclusive by default, and replaced only by explicit choice."""
    target = tmp_path / "existing.md"
    target.write_text("preserve", encoding="utf-8")
    session = Session(messages=[Message(role="user", content="Updated")])

    with pytest.raises(FileExistsError, match="already exists"):
        export_session(session, MemorySessionStore(), target)
    assert target.read_text(encoding="utf-8") == "preserve"

    export_session(session, MemorySessionStore(), target, overwrite=True)
    assert "Updated" in target.read_text(encoding="utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".existing.md.*.tmp")) == []


def test_export_rejects_invalid_format_and_session_database_destination(tmp_path):
    """Export rejects unknown formats and protects the backing session database."""
    store = Mock(spec=SessionStore)
    database = tmp_path / "sessions.sqlite"
    database.write_text("database", encoding="utf-8")
    store.path = database

    with pytest.raises(ValueError, match="Unsupported session export format"):
        export_session(Session(), store, tmp_path / "bad", format="pdf")
    with pytest.raises(ValueError, match="must not be the session database"):
        export_session(Session(), store, database)
    with pytest.raises(ValueError, match="must be a file"):
        export_session(Session(), MemorySessionStore(), tmp_path)
    assert database.read_text(encoding="utf-8") == "database"


def test_export_allows_a_destination_separate_from_the_session_database(tmp_path):
    """Export protects the backing database while allowing another path beside it."""
    store = Mock(spec=SessionStore)
    store.path = tmp_path / "sessions.sqlite"

    export_session(Session(), store, tmp_path / "session.md")

    assert (tmp_path / "session.md").is_file()


def test_export_cleans_up_when_private_file_permissions_cannot_be_set(tmp_path, monkeypatch):
    """Export removes its temporary file if private permissions cannot be applied."""

    def fail_to_set_permissions(*args, **kwargs):
        """Simulate a permissions failure after the temporary export has been created."""
        raise OSError("private file permissions are unavailable")

    monkeypatch.setattr(exporter_module.os, "chmod", fail_to_set_permissions)
    target = tmp_path / "unwritable.md"

    with pytest.raises(OSError, match="private file permissions are unavailable"):
        export_session(Session(), MemorySessionStore(), target)

    assert not target.exists()
    assert list(tmp_path.glob(".unwritable.md.*")) == []
