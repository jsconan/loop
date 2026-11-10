"""Verify marshalled interaction and response-scope forwarding."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import UUID

import pytest
from prompt_toolkit.completion import DummyCompleter

from loop import (
    ConsoleInteraction,
    Interaction,
    Problem,
    ProblemException,
    RunMetrics,
    ToolResultPresentationSpec,
)
from loop.conversation.interaction import ApprovalInteraction, MailboxInteraction
from loop.permissions import PermissionManager
from loop.utils import ChoiceItem, interaction_operation, interaction_run


@pytest.mark.parametrize("supported", [False, True])
def test_mailbox_inherits_fresh_host_approval_capability(supported, tmp_path):
    """Nested adapters preserve the source's fresh host-decision capability."""
    source = MagicMock(spec=ConsoleInteraction)
    source.supports_fresh_host_approval = supported
    mailbox = MailboxInteraction(source)
    assert mailbox.supports_fresh_host_approval is supported
    approval = ApprovalInteraction(mailbox, PermissionManager(tmp_path))
    assert approval.supports_fresh_host_approval is supported


def test_approval_adapter_uses_current_permission_manager(tmp_path):
    """Rebinding an adapter cannot retain automatic authority from an older manager."""
    source = MagicMock(spec=Interaction)
    source.confirm.return_value = False
    old = PermissionManager(tmp_path, yolo=True)
    current = PermissionManager(tmp_path, yolo=False)
    original = ApprovalInteraction.for_permissions(source, old)
    assert ApprovalInteraction.for_permissions(original, old) is original
    rebound = ApprovalInteraction.for_permissions(original, current)
    assert rebound.confirm("Proceed?", default=False) is False
    source.confirm.assert_called_once_with("Proceed?", default=False)
    nested = ApprovalInteraction(original, current)
    flattened = ApprovalInteraction.for_permissions(nested, current)
    assert flattened.confirm("Again?", default=False) is False
    source.confirm.assert_called_with("Again?", default=False)
    old.close()
    current.close()


def test_approval_adapter_forwards_information_prompt_options(tmp_path):
    """An information prompt retains its source options when YOLO is disabled."""
    source = MagicMock(spec=Interaction)
    permissions = PermissionManager(tmp_path)
    interaction = ApprovalInteraction(source, permissions)
    completer = DummyCompleter()
    interaction.prompt(
        "Input?",
        completer=completer,
        exit_commands=None,
        index={"value": "v"},
        default="value",
        secret=True,
    )
    source.prompt.assert_called_once_with(
        "Input?",
        completer=completer,
        exit_commands=None,
        index={"value": "v"},
        default="value",
        secret=True,
    )
    permissions.close()


def test_approval_interaction_preserves_owner_routed_worker_state(tmp_path):
    """A nested approval adapter reports only active worker relay from either layer."""
    source = MailboxInteraction(MagicMock(spec=Interaction))
    interaction = ApprovalInteraction(source, PermissionManager(tmp_path))

    assert interaction.worker_calls_routed is False
    source.bind(lambda _call: None)
    assert interaction.worker_calls_routed is True
    source.bind(None)
    interaction.bind(lambda _call: None)
    assert interaction.worker_calls_routed is True
    interaction.bind(None)
    assert interaction.worker_calls_routed is False


def test_approval_interaction_uses_current_yolo_state_for_sync_and_async_confirmation(tmp_path):
    """All ordinary adapter confirmations follow revocable trusted policy without UI input."""
    source = MagicMock(spec=ConsoleInteraction)
    source.confirm.return_value = False
    source.confirm_async = AsyncMock(return_value=False)
    source.prompt_async = AsyncMock(return_value="choice")
    permissions = PermissionManager(tmp_path, yolo=True)
    interaction = ApprovalInteraction(source, permissions)

    assert interaction.yolo is True
    assert interaction.confirm("Delete data?") is True
    assert asyncio.run(interaction.confirm_async("Delete data?")) is True
    assert asyncio.run(interaction.prompt_async("Which model?")) == "choice"
    source.confirm.assert_not_called()
    source.confirm_async.assert_not_awaited()

    permissions.set_yolo(False)
    assert interaction.yolo is False
    assert interaction.confirm("Delete data?") is False
    assert asyncio.run(interaction.confirm_async("Delete data?")) is False
    source.prompt.return_value = "manual"
    assert interaction.prompt("Choose:", choices={"only": "Only option"}) == "manual"
    source.confirm.assert_called_once_with("Delete data?", default=False)
    source.confirm_async.assert_awaited_once_with("Delete data?", default=False)


def test_approval_interaction_does_not_invent_native_async_input(tmp_path):
    """YOLO can decide confirmation headlessly while missing information still needs input."""
    source = MagicMock(spec=Interaction)
    permissions = PermissionManager(tmp_path, yolo=True)
    interaction = ApprovalInteraction(source, permissions)

    assert asyncio.run(interaction.confirm_async("Proceed?")) is True
    with pytest.raises(ProblemException, match="asynchronous input"):
        asyncio.run(interaction.prompt_async("Which model?"))
    nested = ApprovalInteraction(MailboxInteraction(source), permissions)
    with pytest.raises(ProblemException, match="asynchronous input"):
        asyncio.run(nested.prompt_async("Which model?"))
    permissions.set_yolo(False)
    with pytest.raises(ProblemException, match="asynchronous input"):
        asyncio.run(interaction.confirm_async("Proceed?"))
    with pytest.raises(ProblemException, match="asynchronous input"):
        asyncio.run(nested.confirm_async("Proceed?"))


@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        ({"only": "Only option"}, "only"),
        ([ChoiceItem("O", 7, "Only option")], 7),
        ((item for item in ("only",)), "only"),
    ],
)
def test_approval_interaction_accepts_console_single_choice_without_prompt(
    tmp_path, choices, expected
):
    """The console's one-choice confirmation is an approval, including in headless mode."""
    source = MagicMock(spec=Interaction)
    interaction = ApprovalInteraction(source, PermissionManager(tmp_path, yolo=True))

    assert interaction.prompt("Choose:", choices=choices) == expected
    assert (
        asyncio.run(
            interaction.prompt_async("Choose:", choices=[ChoiceItem("O", expected, "Only option")])
        )
        == expected
    )
    source.prompt.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_single_choice_requires_current_approval_at_the_decision_boundary(tmp_path, asynchronous):
    """A stale managed policy cannot turn the prior YOLO check into an approval."""
    source = MagicMock(spec=ConsoleInteraction)
    source.prompt.return_value = False
    source.prompt_async = AsyncMock(return_value=False)
    permissions = PermissionManager(tmp_path, yolo=True)
    permissions.confirmation_decision = Mock(return_value=None)
    interaction = ApprovalInteraction(source, permissions)
    if asynchronous:
        result = asyncio.run(interaction.prompt_async("Choose:", choices={"only": "Only"}))
        source.prompt_async.assert_awaited_once()
    else:
        result = interaction.prompt("Choose:", choices={"only": "Only"})
        source.prompt.assert_called_once()
    assert result is False


@pytest.mark.parametrize("choices", [{}, {"a": "A", "b": "B"}, [], ["a", "b"]])
def test_approval_interaction_keeps_real_information_choices_interactive(tmp_path, choices):
    """Empty and multi-choice questions still need information from the user."""
    source = MagicMock(spec=ConsoleInteraction)
    source.prompt.return_value = "selected"
    source.prompt_async = AsyncMock(return_value="selected")
    interaction = ApprovalInteraction(source, PermissionManager(tmp_path, yolo=True))

    assert interaction.prompt("Choose:", choices=choices) == "selected"
    assert asyncio.run(interaction.prompt_async("Choose:", choices=choices)) == "selected"
    source.prompt.assert_called_once()
    source.prompt_async.assert_awaited_once()


def test_approval_interaction_preserves_async_owner_dispatch(tmp_path):
    """Async input retains prompt metadata and the wrapper's bound UI owner."""
    source = MagicMock(spec=ConsoleInteraction)
    source.prompt_async = AsyncMock(return_value="selected")
    source.confirm_async = AsyncMock(return_value=True)
    interaction = ApprovalInteraction(source, PermissionManager(tmp_path))
    requests = []

    async def dispatch(request):
        """Record the interaction envelope before delivering native async input."""
        requests.append(request)
        return await request.function(*request.args, **request.kwargs)

    interaction.bind(None, dispatch)

    async def scenario():
        """Exercise both async input methods through the bound wrapper."""
        assert await interaction.prompt_async("Choose:", choices=["a", "b"], initial="draft") == (
            "selected"
        )
        assert await interaction.confirm_async("Proceed?", default=True) is True

    asyncio.run(scenario())

    assert [request.kind for request in requests] == ["prompt", "confirm"]
    assert requests[0].function == source.prompt_async
    assert requests[0].kwargs["initial"] == "draft"
    assert "exit_commands" not in requests[0].kwargs
    assert requests[1].function == source.confirm_async
    assert requests[1].kwargs == {"default": True}


def test_approval_interaction_preserves_editor_drafts(tmp_path):
    """Nested adapters retain the source's editable draft API when available."""
    source = MagicMock(spec=ConsoleInteraction)
    source.current_input.return_value = "unfinished"
    interaction = ApprovalInteraction(MailboxInteraction(source), PermissionManager(tmp_path))

    assert interaction.current_input() == "unfinished"
    interaction.set_current_input("restored")
    source.set_current_input.assert_called_once_with("restored")

    unavailable = ApprovalInteraction(
        MailboxInteraction(MagicMock(spec=Interaction)), PermissionManager(tmp_path)
    )
    assert unavailable.current_input() is None
    unavailable.set_current_input("ignored")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"choices": ["only"], "completer": DummyCompleter()}, "custom completer"),
        ({"choices": {"only": "Only"}, "index": {"wrong": "W"}}, "index must map"),
        (
            {"choices": ["only"], "index": ["q"], "exit_commands": "q"},
            "selection indexes cannot conflict",
        ),
    ],
)
def test_approval_interaction_rejects_invalid_single_choice_prompts(tmp_path, arguments, message):
    """YOLO cannot turn malformed console choices into an automatic decision."""
    source = MagicMock(spec=ConsoleInteraction)
    interaction = ApprovalInteraction(source, PermissionManager(tmp_path, yolo=True))

    with pytest.raises(ValueError, match=message):
        interaction.prompt("Choose:", **arguments)
    with pytest.raises(ValueError, match=message):
        asyncio.run(interaction.prompt_async("Choose:", **arguments))
    source.prompt.assert_not_called()
    source.prompt_async.assert_not_awaited()


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
@pytest.mark.parametrize("approval_wrapped", [False, True])
def test_mailbox_output_forwards_semantics_to_its_source(
    name, args, kwargs, approval_wrapped, tmp_path
):
    """Each semantic output operation preserves arguments through acknowledged dispatch."""
    source = MagicMock(spec=Interaction)
    mailbox = MailboxInteraction(source)
    interaction = (
        ApprovalInteraction(mailbox, PermissionManager(tmp_path)) if approval_wrapped else mailbox
    )
    requests = []

    def dispatch(event):
        """Execute an acknowledged call at the injected UI boundary."""
        requests.append(event)
        return event.function(*event.args, **event.kwargs)

    mailbox.bind(dispatch)
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
@pytest.mark.parametrize("approval_wrapped", [False, True])
def test_mailbox_response_scope_finalizes_on_success_or_failure(
    failure, suppressed, approval_wrapped, tmp_path
):
    """Presentation contexts retain normal exception propagation and suppression semantics."""
    source = MagicMock(spec=Interaction)
    scope = MagicMock()
    scope.__exit__.return_value = suppressed
    source.response_context.return_value = scope
    mailbox = MailboxInteraction(source)
    interaction = (
        ApprovalInteraction(mailbox, PermissionManager(tmp_path)) if approval_wrapped else mailbox
    )

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
