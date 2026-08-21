"""Tests for in-memory session persistence."""

from datetime import UTC

import pytest

from loop import MemorySessionStore, Message, Session, SessionNotFoundError


def test_store_starts_empty_and_reports_missing_sessions():
    """A fresh store has no sessions and missing identifiers fail clearly."""
    store = MemorySessionStore()

    assert store.list() == []
    with pytest.raises(SessionNotFoundError, match="missing"):
        store.load("missing")


def test_store_round_trips_snapshots_and_updates_metadata():
    """Saved snapshots stay isolated and updates preserve the session identity."""
    store = MemorySessionStore()
    session = Session(messages=[Message(role="user", content="hello")], tokens=12)

    session_id = store.save(session)
    session.add_message(Message(role="assistant", content="answer"))
    assert store.load(session_id) != session

    session.model = "model-a"
    assert store.save(session) == session_id

    loaded = store.load(session_id)
    listings = store.list()
    assert loaded == session
    assert loaded is not session
    assert listings[0].id == session_id
    assert listings[0].message_count == 2
    assert listings[0].updated_at.tzinfo == UTC


def test_store_lists_recent_sessions_first_and_keeps_instances_isolated():
    """Listings follow update recency and separate stores do not share sessions."""
    store = MemorySessionStore()
    first = Session()
    second = Session()
    first_id = store.save(first)
    second_id = store.save(second)
    first.messages.append(Message(role="user", content="updated"))
    store.save(first)

    assert [item.id for item in store.list()] == [first_id, second_id]
    assert store.load(second_id) == second
    assert store.load(second_id) is not second
    assert MemorySessionStore().list() == []
