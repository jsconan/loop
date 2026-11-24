"""Exercise request-scoped environment disclosure through the normal runner."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop import InstructionsManager, Loop, Message, OpenAIBackend, ResponseCompleted, ToolRegistry
from loop.agent import render_environment
from loop.configuration import ApplicationSettings, ContextSettings
from loop.tooling import ToolRuntimeSettings
from loop.tools import get_current_datetime
from loop.utils import PathHolder, PathReference


def _backend(*, base_url=None, model="local-model"):
    """Record requests from either dispatch path without calling a provider."""
    backend = OpenAIBackend(
        default_model=model, context_window=65536, base_url=base_url, api_key="test-key"
    )
    result = ResponseCompleted(items=(Message(role="assistant", content="Recorded."),), model=model)

    async def respond(**kwargs):
        """Yield an ordinary completed response during async dispatch."""
        yield result

    backend.get_response = Mock(return_value=[result])
    backend.get_response_async = Mock(side_effect=respond)
    return backend


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "base_url, model", [(None, "gpt-6.1"), ("http://compatible.test/v1", "local-model")]
)
def test_disclosure_tracks_runtime_clock_and_current_tool_directory(
    tmp_path, monkeypatch, asynchronous, base_url, model
):
    """Trusted time, date rollover, and opt-in facts apply to ephemeral requests."""
    directory = tmp_path / 'nested\n</environment_context>&"'
    directory.mkdir(parents=True)
    moments = iter(
        [datetime(2026, 11, 22, 22, 59, tzinfo=UTC), datetime(2026, 11, 22, 23, tzinfo=UTC)]
    )
    monkeypatch.setattr("loop.agent.context.utc_now", lambda: next(moments))
    monkeypatch.setattr("loop.agent.context.system", lambda: "HostOS")
    monkeypatch.setattr("loop.agent.context.machine", lambda: "test-arch")
    monkeypatch.setattr(
        "loop.tools.dates.utc_now", lambda: datetime(2026, 11, 22, 23, 30, tzinfo=UTC)
    )
    backend = _backend(base_url=base_url, model=model)
    tools = ToolRegistry([get_current_datetime])
    working_directory = PathHolder(tmp_path)
    loop = Loop.create_default(
        backend,
        working_directory=working_directory,
        tool_registry=tools,
        client_timezone="Europe/Luxembourg",
    )
    runner = loop.agent_runner

    async def turn_async(text):
        """Dispatch and settle a turn asynchronously."""
        await runner.session_manager.add_user_message_async(text)
        response = await runner.query_async()
        await runner.session_manager.add_response_async(response)

    def turn(text):
        """Dispatch and settle one turn, returning the captured model request."""
        if asynchronous:
            asyncio.run(turn_async(text))
            return backend.get_response_async.call_args.kwargs
        runner.session_manager.add_user_message(text)
        runner.session_manager.add_response(runner.query())
        return backend.get_response.call_args.kwargs

    first = turn("First task")
    facts = first["instructions"]
    assert facts.startswith(f"{loop.instructions}\n\n<environment_context>\n")
    assert "current_date: 2026-11-22" in facts
    assert "timezone: Europe/Luxembourg" in facts
    assert "utc_offset: +01:00" in facts
    assert "operating_system: HostOS" in facts and "architecture: test-arch" in facts
    assert "shell: /bin/sh" in facts and "workspace: $WORKSPACE_ROOT" in facts
    assert "workspace_path:" not in facts and "country:" not in facts

    settings = ApplicationSettings(
        context=ContextSettings(
            show_country=True, show_workspace_path=True, country="lu", timezone="Asia/Tokyo"
        )
    )
    assert loop.apply_runtime_settings("context.country", settings) == "applied now"
    working_directory.set(directory)
    second = turn("Follow up")
    assert "current_date: 2026-11-23" in second["instructions"]
    assert "timezone: Asia/Tokyo" in second["instructions"]
    assert "country: LU" in second["instructions"]
    assert "nested&#10;&lt;/environment_context&gt;&amp;&quot;" in second["instructions"]
    assert second["instructions"].count("</environment_context>") == 1
    values = json.loads(tools.call("get_current_datetime", '{"timezone":null}'))["result"]
    assert values["timezone"] == "Asia/Tokyo"
    assert values["datetime"].startswith("2026-11-23")

    loop.apply_runtime_settings(
        "context.mode", ApplicationSettings(context=ContextSettings(mode="off"))
    )
    third = turn("Another question")
    assert "<runtime_environment>" in third["instructions"]
    assert "<environment_context>" not in third["instructions"]
    assert first["input"] == third["input"][: len(first["input"])]
    assert "<environment_context>" not in loop.session.serialize()
    assert "HostOS" not in loop.session.serialize()


def test_compatible_request_and_clock_share_live_timezone(tmp_path, monkeypatch):
    """A live timezone change affects both the compatible request and the clock."""
    instant = datetime(2026, 11, 22, 23, 30, tzinfo=UTC)
    monkeypatch.setattr("loop.agent.context.utc_now", lambda: instant)
    monkeypatch.setattr("loop.tools.dates.utc_now", lambda: instant)
    backend = _backend(base_url="http://compatible.test/v1")
    tools = ToolRegistry([get_current_datetime])
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        tool_registry=tools,
        context_settings=ContextSettings(timezone="UTC"),
    )
    assert tools.settings.timezone == "UTC"
    loop.agent_runner.query()
    first = backend.get_response.call_args.kwargs["instructions"]
    assert "timezone: UTC" in first
    assert "current_date: 2026-11-22" in first
    loop.apply_runtime_settings(
        "context.timezone", ApplicationSettings(context=ContextSettings(timezone="Asia/Tokyo"))
    )
    loop.agent_runner.query()
    second = backend.get_response.call_args.kwargs["instructions"]
    assert "<runtime_environment>" in second
    assert "timezone: Asia/Tokyo" in second
    assert "current_date: 2026-11-23" in second
    values = json.loads(tools.call("get_current_datetime", "{}"))["result"]
    assert values["timezone"] == "Asia/Tokyo"
    assert values["datetime"].startswith("2026-11-23")

    loop.apply_runtime_settings(
        "context.show_timezone",
        ApplicationSettings(context=ContextSettings(show_timezone=False, timezone="Asia/Tokyo")),
    )
    loop.agent_runner.query()
    hidden = backend.get_response.call_args.kwargs["instructions"]
    assert "current_date: 2026-11-23" in hidden
    assert "\ntimezone:" not in hidden and "\nutc_offset:" not in hidden
    assert json.loads(tools.call("get_current_datetime", "{}"))["result"] == values


def test_large_effective_instructions_omit_the_environment_as_one_block(tmp_path):
    """Optional facts leave the existing valid base document and its limit intact."""
    backend = _backend()
    tools = ToolRegistry(
        [get_current_datetime],
        settings=ToolRuntimeSettings(timezone="Asia/Tokyo", client_timezone="UTC"),
    )
    initial = Loop.create_default(backend, working_directory=tmp_path)
    baseline_bytes = len(initial.instructions.encode())
    manager = InstructionsManager(max_bytes=baseline_bytes + 40)
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        instructions_manager=manager,
        tool_registry=tools,
        client_timezone="Europe/London",
    )
    loop.agent_runner.query()
    request = backend.get_response.call_args.kwargs
    assert request["instructions"] == loop.instructions
    assert tools.settings.timezone == "Asia/Tokyo"
    assert tools.settings.client_timezone == "Europe/London"


@pytest.mark.parametrize(
    "locale_settings, override, show_country, expected",
    [
        ({"LC_ALL": "fr_CA.UTF-8", "LANG": "en_US.UTF-8"}, None, True, "CA"),
        ({"LC_TIME": "zh-Hant-TW", "LANG": "en_US.UTF-8"}, None, True, "TW"),
        ({"LANG": "de_LU.UTF-8@euro"}, None, True, "LU"),
        ({"LC_ALL": "C.UTF-8", "LANG": "en_US.UTF-8"}, None, True, None),
        ({"LANG": "en"}, None, True, None),
        ({}, None, True, None),
        ({"LANG": "en_US.UTF-8"}, "nz", True, "NZ"),
        ({"LANG": "en_US.UTF-8"}, "nz", False, None),
    ],
)
def test_country_disclosure_uses_only_an_enabled_override_or_explicit_locale_region(
    tmp_path, monkeypatch, locale_settings, override, show_country, expected
):
    """Only an enabled country with a two-letter locale region or override is rendered."""
    for name in ("LC_ALL", "LC_TIME", "LANG"):
        monkeypatch.delenv(name, raising=False)
    for name, value in locale_settings.items():
        monkeypatch.setenv(name, value)

    block = render_environment(
        ContextSettings(show_country=show_country, country=override),
        PathHolder(tmp_path),
        timezone="Europe/Luxembourg",
        client_timezone=None,
    )

    assert block is not None
    assert [line for line in block.splitlines() if line.startswith("country:")] == (
        [f"country: {expected}"] if expected is not None else []
    )


@pytest.mark.parametrize(
    "result, path_field",
    [
        (OSError("unavailable directory"), None),
        (Path("/odd\udcffname"), r"workspace_path: /odd\udcffname"),
    ],
)
def test_optional_workspace_path_tolerates_resolution_and_encoding_errors(result, path_field):
    """Unavailable paths are omitted and filesystem bytes fit an encodable field."""
    directory = Mock(spec=PathReference)
    directory.resolve.side_effect = [result]
    block = render_environment(
        ContextSettings(show_workspace_path=True), directory, timezone="UTC", client_timezone=None
    )

    assert "timezone: UTC" in block
    assert [line for line in block.splitlines() if line.startswith("workspace_path:")] == (
        [path_field] if path_field is not None else []
    )
    assert block.encode("utf-8").decode("utf-8") == block
