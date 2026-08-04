"""Tests for shared session persistence models."""

from datetime import UTC, datetime

from loop import Session, SessionInfo


def test_session_extends_summary_with_serialized_context():
    """A complete session remains usable wherever session metadata is expected."""
    session = Session(
        id="session-id",
        updated_at=datetime(2026, 8, 4, tzinfo=UTC),
        message_count=2,
        context='{"version":1}',
    )

    assert isinstance(session, SessionInfo)
    assert session.context == '{"version":1}'
