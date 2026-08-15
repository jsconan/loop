"""Tests for SQLite session persistence."""

import sqlite3
from contextlib import closing
from datetime import UTC

import pytest

from loop import (
    Message,
    Reasoning,
    Session,
    SessionNotFoundError,
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
    store = SQLiteSessionStore(tmp_path / ".loop" / "sessions.db")

    assert store.path == tmp_path / ".loop" / "sessions.db"
    assert store.list() == []
    with pytest.raises(SessionNotFoundError, match="missing"):
        store.load("missing")
    assert not (tmp_path / ".loop").exists()


def test_store_round_trips_complete_typed_contexts_and_updates_metadata(tmp_path):
    """SQLite snapshots preserve every item type, tokens, model, and stable identity."""
    store = SQLiteSessionStore(tmp_path / "sessions.db")
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
    session.messages.append(Message(role="assistant", content="answer"))
    session.tokens = 18
    session.model = "model-b"
    assert store.save(session) == session_id

    loaded = store.load(session_id)
    listings = store.list()
    assert loaded == session
    assert listings[0].id == session_id
    assert listings[0].message_count == 5
    assert listings[0].updated_at.tzinfo == UTC


def test_store_lists_most_recent_sessions_first(tmp_path):
    """Listings order sessions by their latest persisted update."""
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    first_id = store.save(Session())
    second_id = store.save(Session())
    with closing(sqlite3.connect(store.path)) as connection:
        with connection:
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
    with closing(sqlite3.connect(path)) as connection:
        with connection:
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

    store = SQLiteSessionStore(path)

    assert store.list()[0].name == "Recover legacy sessions"
    assert store.load("legacy").name == "Recover legacy sessions"


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"version":2,"messages":[],"tokens":0,"model":null}',
        '{"version":1,"messages":[{"type":"unknown","data":{}}],"tokens":0,"model":null}',
        '{"version":1,"messages":[],"tokens":true,"model":null}',
        '{"version":1,"messages":[],"tokens":0,"model":42}',
    ],
)
def test_store_rejects_invalid_or_unsupported_persisted_data(tmp_path, payload):
    """Loading rejects corrupt, unknown, and incorrectly typed snapshot data."""
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session_id = store.save(Session())
    with closing(sqlite3.connect(store.path)) as connection:
        with connection:
            connection.execute("UPDATE sessions SET session = ?", (payload,))

    with pytest.raises(ValueError):
        store.load(session_id)
