"""Tests for consistent internal timestamp creation."""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock
from zoneinfo import ZoneInfoNotFoundError

from loop.utils import as_utc, local_now, resolve_local_datetime, utc_now, utc_offset


def test_as_utc_normalizes_aware_and_legacy_naive_datetimes():
    """Timestamp normalization converts offsets and treats legacy naive values as UTC."""
    offset = timezone(timedelta(hours=2))

    assert as_utc(datetime(2026, 8, 28, 12, tzinfo=offset)) == datetime(2026, 8, 28, 10, tzinfo=UTC)
    legacy_naive = datetime(2026, 8, 28, 10)
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


def test_time_resolver_prefers_a_configured_override_then_detected_time(monkeypatch):
    """Use a valid override ahead of client detection and fall back to the host offset."""
    instant = datetime(2026, 11, 23, tzinfo=UTC)
    monkeypatch.setenv("TZ", "not-a-zone")
    configured, name = resolve_local_datetime(instant, client="Asia/Tokyo", configured="UTC")
    assert configured == instant
    assert name == "UTC"
    assert resolve_local_datetime(instant, client="UTC", configured="unknown") == (instant, "UTC")
    local, name = resolve_local_datetime(instant, client="unknown", configured="unknown")
    assert local.utcoffset() == instant.astimezone().utcoffset()
    assert name is None
    assert utc_offset(local)[0] in "+-"


def test_explicit_utc_remains_available_without_zone_data(monkeypatch):
    """The fixed UTC selection does not need an installed IANA timezone database."""

    def unavailable(name):
        """Emulate missing system and packaged named-zone data."""
        raise ZoneInfoNotFoundError(name)

    monkeypatch.setattr("loop.utils.dates.ZoneInfo", unavailable)
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    assert resolve_local_datetime(instant, explicit="UTC") == (instant, "UTC")


def test_host_discovery_reads_os_timezone_files_when_no_override(monkeypatch):
    """Discover a named host zone from its symlink or timezone file when offsets agree."""
    clock = datetime(2026, 11, 23, tzinfo=UTC)
    monkeypatch.delenv("TZ", raising=False)
    local = clock.astimezone()
    localtime = Mock(resolve=Mock(return_value="/usr/share/zoneinfo/Etc/UTC"))
    timezone_file = Mock(read_text=Mock(return_value="Europe/Luxembourg\n"))
    paths = {"/etc/localtime": localtime, "/etc/timezone": timezone_file}
    monkeypatch.setattr("loop.utils.dates.Path", paths.__getitem__)
    zone_info = Mock(return_value=timezone(local.utcoffset()))
    monkeypatch.setattr("loop.utils.dates.ZoneInfo", zone_info)

    assert resolve_local_datetime(clock) == (local, "Etc/UTC")
    localtime.resolve.return_value = "/etc/localtime"
    assert resolve_local_datetime(clock) == (local, "Europe/Luxembourg")
    zone_info.return_value = UTC if local.utcoffset() else timezone(timedelta(hours=1))
    assert resolve_local_datetime(clock) == (local, None)
    timezone_file.read_text.side_effect = OSError("unavailable host zone")
    assert resolve_local_datetime(clock) == (local, None)


def test_utc_offset_retains_seconds_when_present():
    """An offset with seconds describes the same instant as the ISO timestamp."""
    timestamp = datetime(2026, 11, 23, tzinfo=timezone(timedelta(seconds=-15)))

    assert utc_offset(timestamp) == "-00:00:15"
