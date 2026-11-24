"""Exercise registered clock arguments and exact time results."""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from loop import ToolRegistry, get_current_datetime
from loop.tooling import ToolRuntimeSettings


def test_direct_clock_keeps_the_no_argument_call(monkeypatch):
    """The exported Python tool can read host time without a registry context."""
    instant = datetime(2026, 11, 22, 23, 30, tzinfo=UTC)
    monkeypatch.setattr("loop.tools.dates.utc_now", lambda: instant)

    result = get_current_datetime()

    assert result["datetime"] == instant.astimezone().isoformat(timespec="seconds")
    assert result["utc"] == "2026-11-22T23:30:00Z"


@pytest.mark.parametrize("selector", [{}, {"timezone": None}])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_clock_defaults_and_nullable_schema(monkeypatch, selector, asynchronous):
    """The clock uses trusted client time and advertises timezone selection in its schema."""
    monkeypatch.setattr(
        "loop.tools.dates.utc_now",
        lambda: datetime(2026, 11, 22, 23, 30, tzinfo=UTC),
    )
    registry = ToolRegistry(
        [get_current_datetime],
        settings=ToolRuntimeSettings(client_timezone="Europe/Luxembourg"),
    )
    parameters = registry.definitions()[0].parameters
    assert list(parameters["properties"]) == ["timezone"]
    timezone = parameters["properties"]["timezone"]
    assert timezone["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert "choose its IANA timezone" in timezone["description"]
    assert "if the place is ambiguous" in timezone["description"]
    assert "if the country spans multiple zones" in timezone["description"]
    arguments = json.dumps(selector)
    output = (
        asyncio.run(registry.call_async("get_current_datetime", arguments))
        if asynchronous
        else registry.call("get_current_datetime", arguments)
    )
    result = json.loads(output)
    assert result == {
        "ok": True,
        "result": {
            "datetime": "2026-11-23T00:30:00+01:00",
            "utc": "2026-11-22T23:30:00Z",
            "timezone": "Europe/Luxembourg",
            "utc_offset": "+01:00",
            "weekday": "Monday",
        },
    }


@pytest.mark.parametrize(
    "hour, selector, wall_time, offset, zone",
    [
        (0, "Europe/Luxembourg", "02:30:00", "+02:00", "Europe/Luxembourg"),
        (1, "Europe/Luxembourg", "02:30:00", "+01:00", "Europe/Luxembourg"),
        (0, "+05:45", "06:15:00", "+05:45", None),
        (0, "-00:00", "00:30:00", "+00:00", None),
        (0, "-03:30", "21:00:00", "-03:30", None),
        (0, "UTC", "00:30:00", "+00:00", "UTC"),
        (0, "GMT", "00:30:00", "+00:00", "GMT"),
    ],
)
def test_clock_converts_one_instant_and_leaves_its_default(
    monkeypatch, hour, selector, wall_time, offset, zone
):
    """Explicit zones preserve the instant and leave the following default invocation unchanged."""
    calls = []
    instant = datetime(2026, 10, 25, hour, 30, tzinfo=UTC)

    def clock():
        """Record the sampling boundary for one tool invocation."""
        calls.append(instant)
        return instant

    monkeypatch.setattr("loop.tools.dates.utc_now", clock)
    registry = ToolRegistry([get_current_datetime], settings=ToolRuntimeSettings(timezone="UTC"))
    result = json.loads(registry.call("get_current_datetime", json.dumps({"timezone": selector})))
    timestamp = result["result"]
    assert timestamp["datetime"].endswith(f"T{wall_time}{offset}")
    assert timestamp["utc"] == instant.isoformat().replace("+00:00", "Z")
    assert timestamp["utc_offset"] == offset
    assert timestamp["timezone"] == zone
    assert calls == [instant]
    assert json.loads(registry.call("get_current_datetime", "{}"))["result"]["timezone"] == "UTC"


@pytest.mark.parametrize("selector", ["CST", "No/Such_Zone", "UTC+3", "+5:45", "+01:60", "+24:00"])
def test_clock_reports_correctable_selector_errors(selector):
    """Invalid names, abbreviations and offsets produce model-readable argument errors."""
    registry = ToolRegistry([get_current_datetime])
    result = json.loads(registry.call("get_current_datetime", json.dumps({"timezone": selector})))
    assert result["ok"] is False
    assert result["problem"]["code"] == "tool.invalid_arguments"
    (issue,) = result["problem"]["metadata"]["fields"]
    assert issue["loc"] == ["timezone"]
    assert "Use an IANA timezone" in issue["msg"]
    assert "signed UTC offset" in issue["msg"]


def test_clock_command_reports_a_structured_selector_error():
    """Direct slash commands report invalid zones through the tool error envelope."""
    registry = ToolRegistry([get_current_datetime])

    result = json.loads(
        registry.command("get_current_datetime", ["timezone=Europe/Belvius"]).output
    )

    assert result["ok"] is False
    assert result["problem"]["code"] == "tool.invalid_arguments"
    (issue,) = result["problem"]["metadata"]["fields"]
    assert issue["loc"] == ["timezone"]
    assert "signed UTC offset" in issue["msg"]
    assert "Use an IANA timezone" in result["problem"]["detail"]
    correct = json.loads(registry.command("get_current_datetime", ["timezone=-03:30"]).output)
    assert correct["result"]["utc_offset"] == "-03:30"
    assert correct["result"]["timezone"] is None


def test_clock_reports_unreadable_timezone_data_as_a_selector_error(monkeypatch):
    """A timezone file that cannot be opened produces a correctable tool error."""

    def unreadable(_name):
        """Simulate an unavailable system timezone file."""
        raise PermissionError("private timezone path")

    monkeypatch.setattr("loop.utils.dates.ZoneInfo", unreadable)
    registry = ToolRegistry([get_current_datetime])
    result = json.loads(registry.command("get_current_datetime", ["timezone=Europe/London"]).output)

    assert result["problem"]["code"] == "tool.invalid_arguments"
    (issue,) = result["problem"]["metadata"]["fields"]
    assert "Use an IANA timezone" in issue["msg"]
    assert "private timezone path" not in json.dumps(result)
