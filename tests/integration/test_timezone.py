"""Check the host timezone resolver against native libc and zoneinfo."""

import os
import sys
import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from loop.utils import resolve_local_datetime, utc_offset

pytestmark = pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"}, reason="POSIX timezone discovery requires libc"
)


def test_native_host_timezone_agrees_with_local_offset(monkeypatch):
    """Use the host setting only when it agrees with the native offset at that instant."""
    instant = datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
    with monkeypatch.context() as context:
        context.delenv("TZ", raising=False)
        time.tzset()
        try:
            local, name = resolve_local_datetime(instant)
            assert local.utcoffset() == instant.astimezone().utcoffset()
            assert local.astimezone(UTC) == instant
            if name is not None:
                assert instant.astimezone(ZoneInfo(name)).utcoffset() == local.utcoffset()
        finally:
            context.undo()
            time.tzset()


@pytest.mark.parametrize(
    "environment, expected_zone, expected_offset",
    [("Europe/Luxembourg", "Europe/Luxembourg", "+02:00"), ("CST6", None, "-06:00")],
)
def test_native_clock_uses_timezone_override_or_offset_fallback(
    monkeypatch, environment, expected_zone, expected_offset
):
    """Match DST rules for an IANA override and preserve offset-only libc operation."""
    instant = datetime(2026, 7, 1, tzinfo=UTC)
    with monkeypatch.context() as context:
        context.setenv("TZ", environment)
        time.tzset()
        try:
            assert os.environ["TZ"] == environment
            local, zone = resolve_local_datetime(instant)
            assert zone == expected_zone
            assert utc_offset(local) == expected_offset
        finally:
            context.undo()
            time.tzset()
