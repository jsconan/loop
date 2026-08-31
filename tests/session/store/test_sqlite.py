"""Tests for SQLite session persistence."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest

from loop import (
    Message,
    Reasoning,
    RunCompletedEvent,
    RunMetrics,
    Session,
    SessionNotFoundError,
    SessionWorkspaceMismatchError,
    SQLiteSessionStore,
    ToolCall,
    ToolResult,
)


def function_call() -> ToolCall:
    """Build a completed local function-tool call."""
    return ToolCall(
        call_id="call_123",
        name="get_current_datetime",
        arguments="{}",
        id="fc_123",
    )


def test_store_stays_absent_until_save_and_reports_missing_sessions(tmp_path):
    """Read operations do not create storage and missing identifiers fail clearly."""
    store = SQLiteSessionStore(tmp_path / ".loop" / "sessions.db", workspace_id="workspace")

    assert store.path == tmp_path / ".loop" / "sessions.db"
    assert store.list() == []
    with pytest.raises(SessionNotFoundError, match="missing"):
        store.load("missing")
    assert not (tmp_path / ".loop").exists()


def test_store_round_trips_complete_typed_contexts_and_updates_metadata(tmp_path):
    """SQLite snapshots preserve every item type, tokens, model, and stable identity."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    session = Session(
        messages=[
            Message(role="user", content="hello"),
            Reasoning(content="thinking", id="reasoning"),
            function_call(),
            ToolResult(call_id="call_123", output="done"),
        ],
        tokens=12,
        model="model-a",
    )

    session_id = store.save(session)
    session.add_message(Message(role="assistant", content="answer"))
    session.tokens = 18
    session.model = "model-b"
    assert store.save(session) == session_id

    loaded = store.load(session_id)
    listings = store.list()
    assert loaded == session
    assert listings[0].id == session_id
    assert listings[0].message_count == 5
    assert listings[0].updated_at.tzinfo == UTC


def test_store_preserves_session_ownership_after_the_workspace_moves(tmp_path):
    """A relocated workspace reopens sessions through durable identity rather than its path."""
    original = tmp_path / "original"
    store = SQLiteSessionStore(original / ".loop" / "sessions.db", workspace_id="workspace")
    session_id = store.save(Session())
    moved = tmp_path / "moved"
    original.rename(moved)

    restored = SQLiteSessionStore(moved / ".loop" / "sessions.db", workspace_id="workspace").load(
        session_id
    )

    assert restored.workspace_id == "workspace"


def test_store_lists_most_recent_sessions_first(tmp_path):
    """Listings order sessions by their latest persisted update."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    first_id = store.save(Session())
    second_id = store.save(Session())
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", first_id),
        )

    assert [item.id for item in store.list()] == [second_id, first_id]
    with pytest.raises(SessionNotFoundError, match="unknown"):
        store.load("unknown")


def test_store_migrates_and_names_existing_sessions(tmp_path):
    """Opening a legacy database backfills display names without losing snapshots."""
    path = tmp_path / "sessions.db"
    session = Session(messages=[Message(role="user", content="Recover legacy sessions")])
    payload = session.serialize()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """CREATE TABLE sessions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL, session TEXT NOT NULL
                )"""
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            ("legacy", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", 1, payload),
        )

    store = SQLiteSessionStore(path, workspace_id="workspace")

    assert store.list()[0].name == "Recover legacy sessions"
    assert store.load("legacy").name == "Recover legacy sessions"


def test_store_upgrades_path_owned_snapshots_once_on_load(tmp_path):
    """Loading replaces legacy path ownership with this database's durable workspace ID."""
    path = tmp_path / "sessions.db"
    session = Session(messages=[Message(role="user", content="question")])
    payload = json.loads(session.serialize())
    payload["version"] = 9
    payload["workspace_root"] = "/old/location"
    payload.pop("workspace_id")
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """CREATE TABLE sessions (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, name_source TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL, session TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "Legacy",
                "user",
                "2026-08-20T00:00:00+00:00",
                "2026-08-20T00:00:00+00:00",
                1,
                json.dumps(payload),
            ),
        )

    restored = SQLiteSessionStore(path, workspace_id="workspace").load("legacy")

    assert restored.workspace_id == "workspace"
    with closing(sqlite3.connect(path)) as connection:
        stored = json.loads(
            connection.execute("SELECT session FROM sessions WHERE id = 'legacy'").fetchone()[0]
        )
    assert stored["version"] == 10
    assert stored["workspace_id"] == "workspace"
    assert "workspace_root" not in stored


def test_store_upgrades_version_four_compactions_on_load(tmp_path):
    """Loading preserves legacy checkpoints while durably adopting workspace identity."""
    path = tmp_path / "sessions.db"
    session = Session(messages=[Message(role="user", content="question")])
    payload = json.loads(session.serialize())
    payload["version"] = 4
    payload.pop("events")
    payload["compactions"] = [
        {
            "id": "checkpoint",
            "boundary": 1,
            "created_at": "2026-08-20T00:00:00Z",
            "provider": "test",
            "model": "model",
            "context": [{"provider": "test", "data": {}}],
            "instructions": {
                "working_directory": "/project",
                "content": None,
                "digest": "digest",
                "active_skills": [],
            },
            "input_tokens_before": 10,
            "input_tokens_after": 5,
        }
    ]
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """CREATE TABLE sessions (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, name_source TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL, session TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "Legacy",
                "user",
                "2026-08-20T00:00:00+00:00",
                "2026-08-20T00:00:00+00:00",
                1,
                json.dumps(payload),
            ),
        )

    restored = SQLiteSessionStore(path, workspace_id="workspace").load("legacy")

    assert [event.type for event in restored.events] == ["conversation_item", "compaction"]
    assert restored.workspace_id == "workspace"


def test_store_upgrades_version_eight_sessions_on_load(tmp_path):
    """Loading a pre-workspace-identity session remains compatible."""
    path = tmp_path / "sessions.db"
    session = Session(messages=[Message(role="user", content="question")])
    payload = json.loads(session.serialize())
    payload["version"] = 8
    payload.pop("workspace_id")

    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """CREATE TABLE sessions (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, name_source TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL, session TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                "Legacy",
                "user",
                "2026-08-20T00:00:00+00:00",
                "2026-08-20T00:00:00+00:00",
                1,
                json.dumps(payload),
            ),
        )

    restored = SQLiteSessionStore(path, workspace_id="workspace").load(session.id)

    assert restored.workspace_id == "workspace"


def test_store_preserves_version_eight_completion_events_on_load(tmp_path):
    """Loading a completed pre-workspace session does not trigger recovery."""
    path = tmp_path / "sessions.db"
    session = Session(messages=[Message(role="user", content="question")])
    session.events.append(
        RunCompletedEvent(
            id="run",
            created_at=datetime(2026, 8, 20, tzinfo=UTC),
            started_at=datetime(2026, 8, 20, tzinfo=UTC),
            stop_reason="completed",
            metrics=RunMetrics(
                active_duration_seconds=0,
                model_duration_seconds=0,
                tool_duration_seconds=0,
                message_count=1,
                item_count=1,
            ),
        )
    )
    payload = json.loads(session.serialize())
    payload["version"] = 8
    payload.pop("workspace_id")

    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """CREATE TABLE sessions (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, name_source TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL, session TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                "Legacy",
                "user",
                "2026-08-20T00:00:00+00:00",
                "2026-08-20T00:00:00+00:00",
                1,
                json.dumps(payload),
            ),
        )

    restored = SQLiteSessionStore(path, workspace_id="workspace").load(session.id)

    assert restored.recovery_state() is None


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"version":9,"messages":[],"tokens":0,"model":null}',
        '{"version":1,"messages":[{"type":"unknown","data":{}}],"tokens":0,"model":null}',
        '{"version":1,"messages":[],"tokens":true,"model":null}',
        '{"version":1,"messages":[],"tokens":0,"model":42}',
        '{"version":1,"messages":null,"tokens":0,"model":null}',
        '{"version":4,"messages":[],"compactions":[1],"tokens":0,"model":null}',
    ],
)
def test_store_rejects_invalid_or_unsupported_persisted_data(tmp_path, payload):
    """Loading rejects corrupt, unknown, and incorrectly typed snapshot data."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    session_id = store.save(Session())
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("UPDATE sessions SET session = ?", (payload,))

    with pytest.raises(ValueError):
        store.load(session_id)


def test_store_rejects_an_empty_workspace_identifier(tmp_path):
    """Workspace-scoped storage requires a non-empty durable identity."""
    with pytest.raises(ValueError, match="must not be empty"):
        SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="")


def test_store_binds_unowned_sessions_and_rejects_other_workspaces(tmp_path):
    """Storage stamps new sessions and refuses snapshots owned by another workspace."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    session = Session()

    store.save(session)

    assert session.workspace_id == "workspace"
    with pytest.raises(SessionWorkspaceMismatchError, match="belongs to workspace"):
        store.save(Session(workspace_id="another"))


def test_store_rejects_loaded_sessions_from_another_workspace(tmp_path):
    """Loading detects a snapshot whose durable owner does not match its database."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    session_id = store.save(Session())
    payload = Session(workspace_id="another").serialize()
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("UPDATE sessions SET session = ? WHERE id = ?", (payload, session_id))

    with pytest.raises(SessionWorkspaceMismatchError, match="belongs to workspace"):
        store.load(session_id)
