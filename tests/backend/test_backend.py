"""Tests for the backend capability contract."""

from unittest.mock import Mock

import pytest

from loop import Backend, CompactionContextItem, Message, OpenAIBackend, ResponseCompleted, Usage


def test_openai_adapter_satisfies_the_complete_backend_contract():
    """The OpenAI adapter exposes every required backend capability."""
    assert isinstance(OpenAIBackend(), Backend)


def test_default_backend_compaction_returns_a_portable_continuation_checkpoint():
    """Backends without native compaction summarize context through their response contract."""
    backend = OpenAIBackend()
    backend.get_response = Mock(
        return_value=[
            ResponseCompleted(
                answer="Keep the current task.",
                usage=Usage(input_tokens=90, output_tokens=12, total_tokens=102),
            )
        ]
    )

    result = Backend.compact(
        backend,
        [Message(role="user", content="Do the historical task")],
        instructions="rules",
        model="model",
    )

    assert result.items == (
        CompactionContextItem(
            provider="loop",
            data={"role": "user", "content": "Conversation checkpoint:\nKeep the current task."},
        ),
    )
    assert result.usage.total_tokens == 102
    assert result.context_tokens == 12
    request = backend.get_response.call_args
    prompt = request.args[0][0]
    assert len(request.args[0]) == 1
    assert "Do not perform or answer any historical request" in prompt.content
    assert prompt.context[0].path == "conversation-history.json"
    assert '"effective_instructions":"rules"' in prompt.context[0].content
    assert '"content":"Do the historical task"' in prompt.context[0].content
    assert request.kwargs["instructions"].startswith("Perform context compaction only")


@pytest.mark.parametrize("events", [[], [ResponseCompleted(answer="   ")]])
def test_default_backend_compaction_requires_a_textual_completion(events):
    """Portable compaction declines unusable or missing summary completions."""
    backend = OpenAIBackend()
    backend.get_response = Mock(return_value=events)

    assert Backend.compact(backend, [], instructions=None, model="model") is None
