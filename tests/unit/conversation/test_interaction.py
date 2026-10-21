"""Verify marshalled interaction and response-scope forwarding."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import UUID

import pytest

from loop import ConsoleInteraction, Interaction, Problem, RunMetrics, ToolResultPresentationSpec
from loop.conversation.interaction import MailboxInteraction
from loop.utils import interaction_operation, interaction_run


def test_mailbox_async_input_uses_the_source_when_unbound():
    """Setup-time async prompts preserve source defaults and explicit exit terms."""
    source = MagicMock(spec=ConsoleInteraction)
    source.prompt_async = AsyncMock(side_effect=["answer", False])
    source.confirm_async = AsyncMock(return_value=True)
    interaction = MailboxInteraction(source)

    async def scenario():
        """Exercise prompt defaults and an explicit confirmation without a controller."""
        assert await interaction.prompt_async("Question?") == "answer"
        assert await interaction.prompt_async("Optional?", exit_commands=None) is False
        assert await interaction.confirm_async("Approve?") is True

    asyncio.run(scenario())

    first_prompt = source.prompt_async.await_args_list[0]
    second_prompt = source.prompt_async.await_args_list[1]
    assert "exit_commands" not in first_prompt.kwargs
    assert second_prompt.kwargs["exit_commands"] is None
    source.confirm_async.assert_awaited_once_with("Approve?", default=False)


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    [
        ("user", ("user text",), {}),
        ("reasoning", ("thought",), {}),
        ("reasoning_delta", ("thought delta",), {}),
        ("answer", ("answer",), {}),
        ("answer_delta", ("answer delta",), {}),
        ("report", (Problem(code="test", title="Test", detail="detail"),), {}),
        ("error", ("error",), {}),
        ("warning", ("warning",), {}),
        ("debug", ({"event": True},), {}),
        ("info", ("information",), {}),
        ("table", ([{"name": "item"}],), {"title": "Table", "max_rows": 1}),
        ("columns", (["item"],), {"marker": "numbered"}),
        ("list", (["item"],), {"marker": "bullet"}),
        ("json", ({"ok": True},), {}),
        ("tree", ([{"path": "item", "folder": "file"}],), {}),
        ("content", ("text",), {"identifier": "item.py", "start_line": 2}),
        ("tool_call", ("inspect", "{}"), {}),
        ("tool_result", ('{"ok":true}', ToolResultPresentationSpec()), {}),
        (
            "run_metrics",
            (
                RunMetrics(
                    active_duration_seconds=1,
                    model_duration_seconds=1,
                    tool_duration_seconds=0,
                    message_count=1,
                    item_count=1,
                ),
            ),
            {},
        ),
        ("permission", ("Approve?", "deny"), {}),
        ("conversation_ended", (), {}),
    ],
)
def test_mailbox_output_forwards_semantics_to_its_source(name, args, kwargs):
    """Each semantic output operation preserves arguments through acknowledged dispatch."""
    source = MagicMock(spec=Interaction)
    interaction = MailboxInteraction(source)
    requests = []

    def dispatch(event):
        """Execute an acknowledged call at the injected UI boundary."""
        requests.append(event)
        return event.function(*event.args, **event.kwargs)

    interaction.bind(dispatch)
    getattr(interaction, name)(*args, **kwargs)

    assert len(requests) == 1
    assert requests[0].kind == "output"
    assert requests[0].function == getattr(source, name)
    assert requests[0].args == args
    assert all(requests[0].kwargs[name] == value for name, value in kwargs.items())
    getattr(source, name).assert_called_once_with(*requests[0].args, **requests[0].kwargs)


def test_mailbox_uses_direct_interaction_before_and_after_binding():
    """Setup and settled runtimes use the original UI without requiring a mailbox."""
    source = Mock(spec=Interaction)
    interaction = MailboxInteraction(source)
    interaction.info("before")
    interaction.bind(Mock())
    interaction.bind(None)
    interaction.info("after")

    assert [item.args for item in source.info.call_args_list] == [("before",), ("after",)]


def test_mailbox_prompts_preserve_masking_choices_and_confirmation_defaults():
    """Input calls retain their separate kind and all safety-relevant prompt metadata."""
    source = Mock(spec=Interaction)
    interaction = MailboxInteraction(source)
    dispatch = Mock(return_value="selected")
    interaction.bind(dispatch)

    assert (
        interaction.prompt(
            "Choose:", choices={"selected": "Label"}, index=["a"], default="selected", secret=True
        )
        == "selected"
    )
    request = dispatch.call_args.args[0]
    assert request.kind == "prompt"
    assert request.args == ("Choose:",)
    assert request.kwargs == {
        "completer": None,
        "choices": {"selected": "Label"},
        "index": ["a"],
        "default": "selected",
        "secret": True,
    }
    dispatch.return_value = False
    assert interaction.confirm("Approve?", default=True) is False
    request = dispatch.call_args.args[0]
    assert request.kind == "confirm"
    assert request.args == ("Approve?",)
    assert request.kwargs == {"default": True}
    source.prompt.assert_not_called()
    source.confirm.assert_not_called()


def test_mailbox_requests_carry_scoped_session_run_and_operation_identity():
    """Every marshalled request has a unique ID and inherits its active operation identity."""
    source = Mock(spec=Interaction)
    requests = []
    interaction = MailboxInteraction(source)
    interaction.bind(lambda request: requests.append(request))

    with interaction_run("session-a"), interaction_operation("tool-call-a"):
        interaction.info("working")
        interaction.confirm("Approve?")

    assert len(requests) == 2
    assert requests[0].request_id != requests[1].request_id
    assert UUID(requests[0].request_id).version in {4, 7}
    assert all(request.session_id == "session-a" for request in requests)
    assert all(request.run_id is not None for request in requests)
    assert all(request.run_id == requests[0].run_id for request in requests)
    assert all(request.operation_id == "tool-call-a" for request in requests)


def test_mailbox_async_request_carries_scoped_correlation_identity():
    """An asynchronous prompt carries the same immutable correlation as sync requests."""
    source = MagicMock(spec=ConsoleInteraction)
    source.prompt_async = AsyncMock(return_value="answer")
    requests = []

    async def dispatch(request):
        """Capture the correlated request and return the deterministic answer."""
        requests.append(request)
        return await request.function(*request.args, **request.kwargs)

    interaction = MailboxInteraction(source)
    interaction.bind(None, dispatch)

    async def scenario():
        """Issue an async prompt inside an operation identity scope."""
        with interaction_run("session-b"), interaction_operation("tool-call-b"):
            assert await interaction.prompt_async("Question?") == "answer"

    asyncio.run(scenario())

    request = requests[0]
    assert request.request_id
    assert UUID(request.request_id).version in {4, 7}
    assert request.session_id == "session-b"
    assert request.run_id is not None
    assert request.operation_id == "tool-call-b"


@pytest.mark.parametrize("exit_commands", [None, "stop", ["q", "quit"]])
def test_mailbox_preserves_explicit_prompt_exit_terms(exit_commands):
    """Explicit exit terms, including disabling them, survive mailbox forwarding."""
    source = Mock(spec=Interaction)
    interaction = MailboxInteraction(source)

    interaction.prompt("Choose:", exit_commands=exit_commands)

    assert source.prompt.call_args.kwargs["exit_commands"] == exit_commands


@pytest.mark.parametrize("failure", [None, RuntimeError("failed")])
@pytest.mark.parametrize("suppressed", [False, True])
def test_mailbox_response_scope_finalizes_on_success_or_failure(failure, suppressed):
    """Presentation contexts retain normal exception propagation and suppression semantics."""
    source = MagicMock(spec=Interaction)
    scope = MagicMock()
    scope.__exit__.return_value = suppressed
    source.response_context.return_value = scope
    interaction = MailboxInteraction(source)

    if failure is not None and not suppressed:
        with pytest.raises(RuntimeError, match="failed"), interaction.response_context():
            raise failure
    else:
        with interaction.response_context():
            if failure is not None:
                raise failure

    scope.__enter__.assert_called_once_with()
    scope.__exit__.assert_called_once()
    assert scope.__exit__.call_args.args[0] is (type(failure) if failure is not None else None)
