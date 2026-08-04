"""Tests for in-memory session persistence."""

from datetime import UTC

import pytest

from loop import LoopContext, MemorySessionStore, Message, SessionNotFoundError


def test_store_starts_empty_and_reports_missing_sessions():
    """A fresh store has no sessions and missing identifiers fail clearly."""
    store = MemorySessionStore()

    assert store.list() == []
    with pytest.raises(SessionNotFoundError, match="missing"):
        store.load("missing")


def test_store_round_trips_snapshots_and_updates_metadata():
    """Saved snapshots stay isolated and updates preserve the session identity."""
    store = MemorySessionStore()
    context = LoopContext(messages=[Message(role="user", content="hello")], tokens=12)

    session_id = store.save(None, context)
    context.messages.append(Message(role="assistant", content="answer"))
    assert store.load(session_id) != context

    context.model = "model-a"
    assert store.save(session_id, context) == session_id

    loaded = store.load(session_id)
    listings = store.list()
    assert loaded == context
    assert loaded is not context
    assert listings[0].id == session_id
    assert listings[0].message_count == 2
    assert listings[0].updated_at.tzinfo == UTC


def test_store_lists_recent_sessions_first_and_keeps_instances_isolated():
    """Listings follow update recency and separate stores do not share sessions."""
    store = MemorySessionStore()
    first_id = store.save("first", LoopContext())
    second_id = store.save("second", LoopContext())
    store.save(first_id, LoopContext(messages=[Message(role="user", content="updated")]))

    assert [item.id for item in store.list()] == [first_id, second_id]
    assert store.load(second_id) == LoopContext()
    assert MemorySessionStore().list() == []
