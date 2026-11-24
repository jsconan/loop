"""Provide consistent internal timestamp creation."""

import os
import re
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_OFFSET = re.compile(r"([+-])([0-9]{2}):([0-9]{2})")
_ZONE_ERROR = "Use an IANA timezone such as Europe/Luxembourg or UTC."
_SELECTOR_ERROR = f"{_ZONE_ERROR} A signed UTC offset such as +05:45 is also accepted."


def _named_timezone(name: str) -> tzinfo:
    """Load an unambiguous zone, including UTC on hosts without timezone data."""
    if name == "UTC":
        return UTC
    try:
        return ZoneInfo(name)
    except (OSError, ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(_ZONE_ERROR) from error


def validate_timezone(name: str) -> str:
    """Validate an IANA timezone name.

    Args:
        name (str): IANA identifier, including UTC.

    Returns:
        str: Validated identifier.

    Raises:
        ValueError: If the input is ambiguous, invalid, or unavailable.
    """
    _named_timezone(name)
    return name


def _selected_timezone(selector: str) -> tuple[tzinfo, str | None]:
    """Resolve a tool selector into an offset or named zone."""
    match = _OFFSET.fullmatch(selector)
    if match is not None:
        sign, hours_text, minutes_text = match.groups()
        hours, minutes = int(hours_text), int(minutes_text)
        if hours >= 24 or minutes >= 60:
            raise ValueError(_SELECTOR_ERROR)
        delta = timedelta(hours=hours, minutes=minutes)
        return timezone(-delta if sign == "-" else delta), None
    try:
        return _named_timezone(selector), selector
    except ValueError as error:
        raise ValueError(_SELECTOR_ERROR) from error


def validate_timezone_selector(selector: str) -> str:
    """Validate a clock selector for the structured tool schema.

    Args:
        selector (str): IANA timezone or signed UTC offset with explicit minutes.

    Returns:
        str: Validated selector.

    Raises:
        ValueError: If the selector does not name a supported timezone or offset.
    """
    _selected_timezone(selector)
    return selector


def _host_timezone() -> str | None:
    """Find the named zone from the process override or the OS timezone files."""
    override = os.environ.get("TZ")
    if override is not None:
        return override.removeprefix(":").split("/zoneinfo/")[-1]
    try:
        target = str(Path("/etc/localtime").resolve())
        if "/zoneinfo/" in target:
            return target.split("/zoneinfo/", maxsplit=1)[1]
        return Path("/etc/timezone").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError, RuntimeError):
        return None


def resolve_local_datetime(
    instant: datetime,
    *,
    explicit: str | None = None,
    client: str | None = None,
    configured: str | None = None,
) -> tuple[datetime, str | None]:
    """Convert one instant using a call selector, configured override, or detected timezone.

    Args:
        instant (datetime): Sampled instant, with naive values interpreted as UTC.
        explicit (str | None): Timezone or fixed offset selected for this call.
        client (str | None): IANA timezone received from trusted client context.
        configured (str | None): Validated override that takes precedence over detection.

    Returns:
        tuple[datetime, str | None]: Offset-bearing local instant and its IANA name, if known.

    Raises:
        ValueError: If an explicit timezone or offset is invalid.
    """
    instant = as_utc(instant)
    if explicit is not None:
        zone, name = _selected_timezone(explicit)
        return instant.astimezone(zone), name
    for name in (configured, client):
        if name is not None:
            try:
                return instant.astimezone(_named_timezone(name)), name
            except ValueError:
                continue
    local = instant.astimezone()
    name = _host_timezone()
    if name is not None:
        try:
            if instant.astimezone(_named_timezone(name)).utcoffset() == local.utcoffset():
                return local, name
        except ValueError:
            pass
    return local, None


def utc_offset(timestamp: datetime) -> str:
    """Format the offset of an aware timestamp.

    Args:
        timestamp (datetime): Offset-bearing date and time.

    Returns:
        str: Signed ISO offset with explicit minutes and seconds when present.
    """
    return timestamp.strftime("%:z")


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
