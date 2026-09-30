"""Tests for in-memory session persistence."""

import json
from datetime import UTC, datetime

import pytest

from loop import (
    ContextReference,
    MemorySessionStore,
    Message,
    Session,
    SessionNotFoundError,
    SessionRevisionConflictError,
)
from loop.utils import content_identity


def test_store_starts_empty_and_reports_missing_sessions():
    """A fresh store has no sessions and missing identifiers fail clearly."""
    store = MemorySessionStore()

    assert store.list() == []
    with pytest.raises(SessionNotFoundError, match="missing"):
        store.load("missing")


def test_store_deduplicates_immutable_reference_content():
    """Equal reference bytes share one stable digest-backed artifact."""
    store = MemorySessionStore()

    handle, digest = content_identity(b"content")
    reference = ContextReference(
        kind="file",
        path="first.txt",
        content="content",
        size_bytes=7,
        included_bytes=7,
        truncated=False,
        handle=handle,
        version=digest,
    )
    session = Session(messages=[Message(role="user", content="review", context=(reference,))])

    store.save(session)

    assert store.load_reference(digest) == b"content"
    assert store.load_reference("missing") is None


def test_store_persists_a_verified_unowned_reference():
    """Reference seeding retains valid bytes without creating a session."""
    store = MemorySessionStore()
    content = b"content"
    handle, digest = content_identity(content)

    store.store_reference(content, handle=handle, digest=digest)

    assert store.load_reference(digest) == content
    assert store.list() == []
    with pytest.raises(ValueError, match="integrity"):
        store.store_reference(b"changed", handle=handle, digest=digest)


def test_store_rejects_a_handle_rebound_to_different_content():
    """One opaque content capability cannot be rebound to another artifact."""
    store = MemorySessionStore()
    handle, first_digest = content_identity(b"first")
    _, second_digest = content_identity(b"second")

    store.store_reference(b"first", handle=handle, digest=first_digest)

    with pytest.raises(ValueError, match="integrity"):
        store.store_reference(b"second", handle=handle, digest=second_digest)


def test_store_rejects_a_session_handle_rebound_to_different_content():
    """Atomic session saves preserve existing opaque capability bindings."""
    store = MemorySessionStore()
    handle, first_digest = content_identity(b"first")
    _, second_digest = content_identity(b"second")
    first = ContextReference(
        kind="file",
        path="first.txt",
        content="first",
        size_bytes=5,
        included_bytes=5,
        truncated=False,
        handle=handle,
        version=first_digest,
    )
    second = first.model_copy(
        update={
            "path": "second.txt",
            "content": "second",
            "size_bytes": 6,
            "included_bytes": 6,
            "version": second_digest,
        }
    )

    store.save(Session(messages=[Message(role="user", content="first", context=(first,))]))

    with pytest.raises(ValueError, match="integrity"):
        store.save(Session(messages=[Message(role="user", content="second", context=(second,))]))

    assert store.load_reference(first_digest) == b"first"
    assert store.load_reference(second_digest) is None


def test_store_collects_only_artifacts_released_by_session_updates():
    """Garbage collection removes artifacts after their last session owner releases them."""
    store = MemorySessionStore()
    handle, digest = content_identity(b"content")
    reference = ContextReference(
        kind="file",
        path="first.txt",
        content="content",
        size_bytes=7,
        included_bytes=7,
        truncated=False,
        handle=handle,
        version=digest,
    )
    session = Session(messages=[Message(role="user", content="review", context=(reference,))])
    store.save(session)

    assert store.collect_unreferenced() == 0
    session.messages.clear()
    store.save(session)
    assert store.collect_unreferenced() == 1
    assert store.load_reference(digest) is None


def test_store_rejects_invalid_or_missing_reference_artifacts():
    """Atomic saves validate staged bytes and every session-owned digest."""
    store = MemorySessionStore()
    handle, digest = content_identity(b"content")
    with pytest.raises(ValueError, match="integrity"):
        store.store_reference(b"content", handle="invalid", digest=digest)
    invalid = ContextReference(
        kind="file",
        path="invalid.txt",
        content="changed",
        size_bytes=7,
        included_bytes=7,
        truncated=False,
        handle=handle,
        version=digest,
    )
    with pytest.raises(ValueError, match="integrity"):
        store.save(Session(messages=[Message(role="user", content="review", context=(invalid,))]))

    reference = ContextReference(
        kind="file",
        path="missing.txt",
        size_bytes=7,
        included_bytes=7,
        truncated=False,
        handle=handle,
        version=digest,
    )
    session = Session(messages=[Message(role="user", content="review", context=(reference,))])
    with pytest.raises(ValueError, match="unavailable"):
        store.save(session)

    incomplete = Session(
        messages=[
            Message(
                role="user",
                content="review",
                context=(
                    ContextReference(
                        kind="file",
                        path="incomplete.txt",
                        size_bytes=1,
                        included_bytes=0,
                        truncated=True,
                    ),
                ),
            )
        ]
    )
    with pytest.raises(ValueError, match="artifact identity"):
        store.save(incomplete)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("handle", "invalid"),
        ("size_bytes", 8),
    ],
)
def test_store_rejects_contentless_reference_metadata_that_disagrees_with_an_artifact(field, value):
    """Contentless occurrences must match the durable artifact exactly before persistence."""
    store = MemorySessionStore()
    handle, digest = content_identity(b"content")
    source = ContextReference(
        kind="file",
        path="source.txt",
        content="content",
        size_bytes=7,
        included_bytes=7,
        truncated=False,
        handle=handle,
        version=digest,
    )
    corrupted = source.model_copy(update={"content": "", field: value})

    with pytest.raises(ValueError, match="integrity|Invalid (content|serialized reference)"):
        store.save(
            Session(
                messages=[
                    Message(role="user", content="capture", context=(source,)),
                    Message(role="user", content="reuse", context=(corrupted,)),
                ]
            )
        )


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


def test_store_load_externalizes_session_owned_legacy_artifacts():
    """Loading a legacy snapshot commits session-owned artifacts before returning it."""
    store = MemorySessionStore()
    payload = json.loads(Session(messages=[Message(role="user", content="review")]).serialize())
    payload["version"] = 11
    payload["messages"][0]["data"]["context"] = [
        {
            "kind": "file",
            "path": "source.txt",
            "content": "snapshot",
            "size_bytes": 8,
            "included_bytes": 8,
            "truncated": False,
            "handle": None,
            "next_cursor": None,
            "snapshot_content": None,
        }
    ]
    legacy = Session.deserialize(json.dumps(payload))
    store._sessions.append(
        {
            "id": legacy.id,
            "name": "Legacy",
            "name_source": "user",
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "message_count": 1,
            "session": json.dumps(payload),
            "revision": 1,
        }
    )

    restored = store.load(legacy.id)

    assert restored.messages[0].context[0].content == "snapshot"
    assert restored.revision == 2
    assert store.load_reference(content_identity(b"snapshot")[1]) == b"snapshot"


def test_store_rejects_stale_in_memory_snapshots():
    """The adapter contract rejects divergent snapshots consistently in memory."""
    store = MemorySessionStore()
    session = Session()
    store.save(session)
    first = store.load(session.id)
    stale = store.load(session.id)

    first.add_message(Message(role="user", content="winner"))
    store.save(first)

    with pytest.raises(SessionRevisionConflictError):
        store.save(stale)

    with pytest.raises(SessionRevisionConflictError):
        store.save(Session(id="missing", revision=1))


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
