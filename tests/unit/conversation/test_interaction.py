"""Verify marshalled interaction and response-scope forwarding."""

from unittest.mock import MagicMock, Mock

import pytest

from loop import Interaction, Problem, RunMetrics, ToolResultPresentationSpec
from loop.conversation.interaction import MailboxInteraction


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
