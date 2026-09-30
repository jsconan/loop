"""Tests for consistent internal timestamp creation."""

from datetime import UTC, datetime, timedelta, timezone

from loop.utils import as_utc, local_now, utc_now


def test_as_utc_normalizes_aware_and_legacy_naive_datetimes():
    """Timestamp normalization converts offsets and treats legacy naive values as UTC."""
    offset = timezone(timedelta(hours=2))

    assert as_utc(datetime(2026, 8, 28, 12, tzinfo=offset)) == datetime(2026, 8, 28, 10, tzinfo=UTC)
    legacy_naive = datetime(2026, 8, 28, 10)  # noqa: DTZ001 - exercise legacy input.
    assert as_utc(legacy_naive) == datetime(2026, 8, 28, 10, tzinfo=UTC)


def test_local_now_returns_timezone_aware_local_datetime():
    """User-facing timestamps use the system's local timezone."""
    timestamp = local_now()

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() is not None


def test_utc_now_returns_timezone_aware_utc_datetime():
    """Internal timestamps are always represented in UTC."""
    timestamp = utc_now()

    assert timestamp.tzinfo is UTC
    assert timestamp.utcoffset().total_seconds() == 0
