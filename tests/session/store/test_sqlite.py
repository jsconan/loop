"""Tests for SQLite session persistence."""

import json
import sqlite3
import stat
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
    SessionRevisionConflictError,
    SessionWorkspaceMismatchError,
    SQLiteSessionStore,
    ToolCall,
    ToolResult,
)
from loop.utils import PathHolder


def test_sqlite_store_repairs_private_database_permissions(tmp_path):
    """Session storage and its containing directory remain accessible only to the owner."""
    parent = tmp_path / "sessions"
    parent.mkdir(mode=0o755)
    path = parent / "sessions.db"
    store = SQLiteSessionStore(path, workspace_id="workspace")
    store.save(Session())
    path.chmod(0o644)

    assert store.list()
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_store_snapshots_its_database_path(tmp_path):
    """A store keeps the immutable database destination selected at construction."""
    path = PathHolder(tmp_path / "first.db")
    store = SQLiteSessionStore(path, workspace_id="workspace")
    path.set(tmp_path / "second.db")
    session = Session(workspace_id="workspace")

    store.save(session)

    assert store.path == tmp_path / "first.db"
    assert store.load(session.id).id == session.id
    assert not (tmp_path / "second.db").exists()


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


def test_legacy_import_refuses_to_replace_an_existing_session_database(tmp_path):
    """A completed legacy import is idempotent at the session-store boundary."""
    source = tmp_path / "legacy.db"
    destination = tmp_path / "sessions.db"
    sqlite3.connect(source).close()
    destination.touch()

    assert not SQLiteSessionStore(destination, workspace_id="workspace").import_legacy(source)


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


def test_store_rejects_stale_snapshots_without_losing_newer_history(tmp_path):
    """Independent loads share a base revision and stale history cannot overwrite a winner."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    original = Session(messages=[Message(role="user", content="start")])
    store.save(original)
    writer_a = store.load(original.id)
    writer_b = store.load(original.id)

    assert writer_a.revision == writer_b.revision == 1
    writer_a.add_message(Message(role="assistant", content="winner"))
    store.save(writer_a)

    assert writer_a.revision == 2
    writer_b.add_message(Message(role="assistant", content="stale"))
    with pytest.raises(SessionRevisionConflictError) as conflict:
        store.save(writer_b)

    assert conflict.value.expected_revision == 1
    assert conflict.value.current_revision == 2
    preserved = store.load(original.id)
    assert [message.content for message in preserved.messages] == ["start", "winner"]

    rebased = store.load(original.id)
    rebased.add_message(Message(role="assistant", content="rebased"))
    store.save(rebased)
    assert rebased.revision == 3
    assert [message.content for message in store.load(original.id).messages] == [
        "start",
        "winner",
        "rebased",
    ]


def test_store_advances_snapshot_and_revision_in_one_transaction(tmp_path):
    """Each committed snapshot exposes exactly its matching advanced revision."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    session = Session(messages=[Message(role="user", content="one")])

    store.save(session)
    session.add_message(Message(role="assistant", content="two"))
    store.save(session)

    with closing(sqlite3.connect(store.path)) as connection:
        row = connection.execute(
            "SELECT revision, message_count, session FROM sessions WHERE id = ?", (session.id,)
        ).fetchone()
    assert row[:2] == (2, 2)
    assert len(Session.deserialize(row[2]).messages) == 2


def test_store_rejects_duplicate_new_and_deleted_loaded_snapshots(tmp_path):
    """Revision conflicts distinguish duplicate identities and vanished persisted bases."""
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    session = Session()
    store.save(session)

    with pytest.raises(SessionRevisionConflictError) as duplicate:
        store.save(Session(id=session.id))
    assert duplicate.value.current_revision == 1

    loaded = store.load(session.id)
    with (
        closing(sqlite3.connect(store.path)) as connection,
        connection,
        closing(connection.execute("DELETE FROM sessions WHERE id = ?", (session.id,))),
    ):
        pass
    with pytest.raises(SessionRevisionConflictError) as deleted:
        store.save(loaded)
    assert deleted.value.current_revision == 0


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
    restored = store.load("legacy")
    assert restored.name == "Recover legacy sessions"
    assert restored.revision == 2
    store.save(restored)
    with closing(sqlite3.connect(path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
        revision = connection.execute(
            "SELECT revision FROM sessions WHERE id = 'legacy'"
        ).fetchone()[0]
    assert "revision" in columns
    assert revision == 3


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
    assert stored["version"] == 11
    assert stored["workspace_id"] == "workspace"
    assert "workspace_root" not in stored
    assert restored.revision == 2


def test_store_rejects_a_concurrent_write_during_workspace_upgrade(tmp_path, monkeypatch):
    """Legacy adoption uses revision CAS and cannot overwrite a concurrent writer."""
    path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(path, workspace_id="workspace")
    session = Session()
    store.save(session)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE sessions SET session = ? WHERE id = ?",
            (Session(id=session.id).serialize(), session.id),
        )

    connect = sqlite3.connect

    class RacingConnection(sqlite3.Connection):
        """Advance the durable revision immediately before legacy adoption CAS."""

        def execute(self, sql, parameters=()):
            """Inject one competing revision update at the migration boundary."""
            if "UPDATE sessions SET session = ?, revision = revision + 1" in sql:
                with closing(connect(path)) as competing, competing:
                    competing.execute(
                        "UPDATE sessions SET revision = revision + 1 WHERE id = ?", (session.id,)
                    )
            return super().execute(sql, parameters)

    monkeypatch.setattr(
        "loop.session.store.sqlite.sqlite3.connect",
        lambda database: connect(database, factory=RacingConnection),
    )

    with pytest.raises(SessionRevisionConflictError) as conflict:
        store.load(session.id)

    assert conflict.value.expected_revision == 1
    assert conflict.value.current_revision == 2


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
