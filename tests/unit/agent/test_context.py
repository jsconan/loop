"""Exercise request-scoped environment disclosure through the normal runner."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop import (
    InstructionsManager,
    Loop,
    Message,
    OpenAIBackend,
    ResponseCompleted,
    Session,
    SessionManager,
    SQLiteSessionStore,
    ToolRegistry,
)
from loop.agent import render_environment
from loop.configuration import ApplicationSettings, ContextSettings
from loop.models import DeveloperInput
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
    "base_url, model", [(None, "gpt-6.2"), ("http://compatible.test/v1", "local-model")]
)
def test_disclosure_tracks_runtime_clock_and_current_tool_directory(
    tmp_path, monkeypatch, asynchronous, base_url, model
):
    """Place automatic time after the prefix and keep opt-in environment facts scoped."""
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
    if base_url is None:
        assert isinstance(first["input"][0], DeveloperInput)
        assert facts.startswith("<environment_context>\n")
    else:
        assert facts.startswith(f"{loop.instructions}\n\n<environment_context>\n")
    if base_url is None:
        assert "current_date: 2026-11-22" in facts
        assert "timezone: Europe/Luxembourg" in facts
        assert "utc_offset: +01:00" in facts
    else:
        assert all(field not in facts for field in ("current_date:", "timezone:", "utc_offset:"))
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
    if base_url is None:
        assert "current_date: 2026-11-23" in second["instructions"]
        assert "timezone: Asia/Tokyo" in second["instructions"]
    else:
        assert "current_date:" not in second["instructions"]
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
    if base_url is None:
        assert third["instructions"] is None
        assert "<runtime_environment>" in third["input"][0].content
    else:
        assert "<runtime_environment>" in third["instructions"]
    assert "<environment_context>" not in (third["instructions"] or "")
    assert first["input"] == third["input"][: len(first["input"])]
    assert "<environment_context>" not in loop.session.serialize()
    assert "HostOS" not in loop.session.serialize()


@pytest.mark.parametrize(
    ("before", "after", "local_date", "offset"),
    [
        (
            datetime(2026, 11, 24, 22, 59, tzinfo=UTC),
            datetime(2026, 11, 24, 23, 1, tzinfo=UTC),
            "2026-11-25",
            "+01:00",
        ),
        (
            datetime(2026, 10, 25, 0, 59, tzinfo=UTC),
            datetime(2026, 10, 25, 1, 1, tzinfo=UTC),
            "2026-10-25",
            "+01:00",
        ),
    ],
)
def test_leading_instructions_survive_date_and_offset_rollovers(
    tmp_path, monkeypatch, before, after, local_date, offset
):
    """Midnight and daylight saving leave the system prompt and earlier input byte-stable."""
    now = before
    monkeypatch.setattr("loop.agent.context.utc_now", lambda: now)
    monkeypatch.setattr("loop.tools.dates.utc_now", lambda: now)
    backend = _backend(base_url="http://compatible.test/v1")
    tools = ToolRegistry([get_current_datetime])
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        tool_registry=tools,
        context_settings=ContextSettings(timezone="Europe/Luxembourg"),
    )
    runner = loop.agent_runner
    runner.session_manager.add_user_message("Start a task")
    runner.session_manager.add_response(runner.query())
    first = backend.get_response.call_args.kwargs
    first_clock = json.loads(tools.call("get_current_datetime", "{}"))["result"]

    now = after
    runner.session_manager.add_user_message("Continue the task")
    runner.session_manager.add_response(runner.query())
    second = backend.get_response.call_args.kwargs
    clock = json.loads(tools.call("get_current_datetime", "{}"))["result"]

    assert second["instructions"] == first["instructions"]
    assert second["input"][: len(first["input"])] == first["input"]
    assert second["tools"] == first["tools"]
    assert "operating_system:" in second["instructions"]
    for field in ("current_date:", "timezone:", "utc_offset:"):
        assert field not in second["instructions"]
    assert clock["datetime"].startswith(local_date)
    assert clock["utc_offset"] == offset
    assert clock["datetime"] != first_clock["datetime"]
    assert not loop.session.compactions


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("base_url", "model", "automatic_time"),
    [
        (None, "gpt-6.2", True),
        (None, "gpt-6.1", False),
        ("http://compatible.test/v1", "local-model", False),
    ],
)
def test_next_day_resume_preserves_the_previous_provider_input(
    tmp_path, monkeypatch, asynchronous, base_url, model, automatic_time
):
    """Reopen persisted history with fresh runtime state and a later calendar date."""
    now = datetime(2026, 11, 24, 23, 59, tzinfo=UTC)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr("loop.agent.context.utc_now", lambda: now)
    database = tmp_path / "sessions.db"

    def open_loop(session_id=None):
        """Compose a fresh app and backend against the same saved session store."""
        backend = _backend(base_url=base_url, model=model)
        loop = Loop.create_default(
            backend,
            working_directory=tmp_path,
            session=session_id,
            session_manager=SessionManager(
                session_store=SQLiteSessionStore(database, workspace_id="workspace")
            ),
            context_settings=ContextSettings(timezone="UTC"),
        )
        return loop, backend

    def turn(loop, backend, text):
        """Commit a normal user input and capture the request through the selected path."""
        runner = loop.agent_runner

        async def turn_async():
            """Dispatch and commit through the asynchronous session writer."""
            await runner.session_manager.add_user_message_async(text)
            await runner.session_manager.add_response_async(await runner.query_async())

        if asynchronous:
            asyncio.run(turn_async())
            return backend.get_response_async.call_args.kwargs
        runner.session_manager.add_user_message(text)
        runner.session_manager.add_response(runner.query())
        return backend.get_response.call_args.kwargs

    original, first_backend = open_loop()
    first = turn(original, first_backend, "Remember the previous request.")
    session_id = original.session.id
    now = datetime(2026, 11, 25, 0, 1, tzinfo=UTC)
    resumed, second_backend = open_loop(session_id)
    second = turn(resumed, second_backend, "Continue the saved request.")

    assert resumed.session.window.id == original.session.window.id
    assert second["input"][: len(first["input"])] == first["input"]
    assert second["tools"] == first["tools"]
    assert "<environment_context>" not in resumed.session.serialize()
    if automatic_time:
        assert "current_date: 2026-11-24" in first["instructions"]
        assert "current_date: 2026-11-25" in second["instructions"]
    else:
        assert second["instructions"] == first["instructions"]
        assert "current_date:" not in second["instructions"]


def test_leading_routes_apply_clock_timezone_edits_without_rebuilding(tmp_path, monkeypatch):
    """Timezone edits apply to the clock immediately and after session resume."""
    instant = datetime(2026, 11, 22, 23, 30, tzinfo=UTC)
    monkeypatch.setattr("loop.tools.dates.utc_now", lambda: instant)
    backend = _backend(base_url="http://compatible.test/v1")
    tools = ToolRegistry([get_current_datetime])
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        tool_registry=tools,
        context_settings=ContextSettings(timezone="UTC"),
    )
    runner = loop.agent_runner
    runner.session_manager.add_user_message("First")
    runner.session_manager.add_response(runner.query())
    first = backend.get_response.call_args.kwargs["instructions"]
    window = loop.session.window
    original_clock = json.loads(tools.call("get_current_datetime", "{}"))["result"]
    assert original_clock["timezone"] == "UTC"
    assert original_clock["datetime"].startswith("2026-11-22")

    settings = ApplicationSettings(context=ContextSettings(timezone="Asia/Tokyo"))
    assert loop.apply_runtime_settings("context.timezone", settings) == "applied now"
    assert loop.configuration_status("context.timezone") is None
    runner.session_manager.add_user_message("Keep the prefix")
    runner.session_manager.add_response(runner.query())
    assert backend.get_response.call_args.kwargs["instructions"] == first
    assert loop.session.window == window
    updated_clock = json.loads(tools.call("get_current_datetime", "{}"))["result"]
    assert updated_clock["timezone"] == "Asia/Tokyo"
    assert updated_clock["datetime"].startswith("2026-11-23")

    restored_tools = ToolRegistry([get_current_datetime])
    restored = Loop.create_default(
        backend,
        working_directory=tmp_path,
        tool_registry=restored_tools,
        context_settings=settings.context,
        session=Session.deserialize(loop.session.serialize()),
    )
    restored.agent_runner.query()
    assert backend.get_response.call_args.kwargs["instructions"] == first
    assert json.loads(restored_tools.call("get_current_datetime", "{}"))["result"] == updated_clock
    assert not restored.session.compactions


def test_suffix_disclosure_can_hide_timezone_without_hiding_local_date(tmp_path, monkeypatch):
    """A verified suffix includes the local date independently of timezone visibility."""
    instant = datetime(2026, 11, 22, 23, 30, tzinfo=UTC)
    monkeypatch.setattr("loop.agent.context.utc_now", lambda: instant)
    backend = _backend(model="gpt-6.2")
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        context_settings=ContextSettings(show_timezone=False, timezone="Europe/Luxembourg"),
    )

    loop.agent_runner.query()

    request = backend.get_response.call_args.kwargs
    assert "current_date: 2026-11-23" in request["instructions"]
    assert "timezone:" not in request["instructions"]
    assert "utc_offset:" not in request["instructions"]
    assert "current_date:" not in request["input"][0].content


def test_large_effective_instructions_omit_the_environment_as_one_block(tmp_path):
    """Optional facts leave the existing valid base document and its limit intact."""
    backend = _backend(base_url="http://compatible.test/v1")
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
        client_timezone=None,
        include_time=False,
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
        ContextSettings(show_workspace_path=True, timezone="UTC"),
        directory,
        client_timezone=None,
        include_time=True,
    )

    assert "timezone: UTC" in block
    assert [line for line in block.splitlines() if line.startswith("workspace_path:")] == (
        [path_field] if path_field is not None else []
    )
    assert block.encode("utf-8").decode("utf-8") == block
