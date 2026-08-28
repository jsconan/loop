"""Provide consistent internal timestamp creation."""

from datetime import UTC, datetime


def as_utc(timestamp: datetime) -> datetime:
    """Normalize a date and time to UTC.

    Naive timestamps are interpreted as UTC for compatibility with legacy persisted data.

    Args:
        timestamp (datetime): Date and time to normalize.

    Returns:
        datetime: Timezone-aware UTC date and time.
    """
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def local_now() -> datetime:
    """Return the current timezone-aware local date and time for user-facing output.

    Returns:
        datetime: Current date and time in the system's local timezone.
    """
    return datetime.now().astimezone()


def utc_now() -> datetime:
    """Return the current timezone-aware UTC date and time.

    Returns:
        datetime: Current date and time with the UTC timezone.
    """
    return datetime.now(UTC)
