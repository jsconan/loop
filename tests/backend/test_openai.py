"""Tests for the OpenAI-compatible backend adapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from openai import APIConnectionError
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseTextDeltaEvent,
)

from loop import (
    AnswerCompleted,
    AnswerDelta,
    Message,
    ModelInfo,
    OpenAIBackend,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    ResponseCompleted,
    ToolCall,
    ToolCallCompleted,
    ToolRegistry,
    ToolResult,
    Usage,
)


@pytest.fixture(autouse=True)
def isolate_backend_environment(monkeypatch):
    """Prevent host configuration from influencing backend tests."""
    for variable in ("DEFAULT_MODEL", "BASE_URL", "OPENAI_API_KEY", "CONTEXT_WINDOW"):
        monkeypatch.delenv(variable, raising=False)


def sdk_completion_event(total_tokens=12, model="served-model", output=None):
    """Build an OpenAI terminal event with optional usage and model metadata."""
    usage = None
    if total_tokens is not None:
        usage = {
            "input_tokens": 10,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": total_tokens,
        }
    return ResponseCompletedEvent.model_validate(
        {
            "type": "response.completed",
            "sequence_number": 10,
            "response": {
                "id": "response_1",
                "created_at": 0,
                "model": model,
                "object": "response",
                "output": output or [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": usage,
            },
        }
    )


class AsyncEvents:
    """Adapt a finite event sequence to an asynchronous iterator."""

    def __init__(self, events):
        self._events = iter(events)
        self.yielded = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        self.yielded += 1
        return event


async def collect_events(events):
    """Collect an asynchronous response event iterator."""
    return [event async for event in events]


def test_configuration_and_lazy_clients_use_explicit_credentials(monkeypatch):
    """Configured values are exposed and used once for each lazy SDK client."""
    monkeypatch.setenv("DEFAULT_MODEL", "environment-model")
    monkeypatch.setenv("BASE_URL", "https://environment.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    registry = ToolRegistry()
    sync_sdk = Mock()
    async_sdk = Mock()
    sync_sdk.models.list.return_value = []
    async_sdk.models.list = AsyncMock(return_value=[])

    with (
        patch("loop.backend.openai.OpenAI", return_value=sync_sdk) as openai,
        patch("loop.backend.openai.AsyncOpenAI", return_value=async_sdk) as async_openai,
    ):
        client = OpenAIBackend("chosen-model", "https://example.test/v1", "secret", registry)

        assert client.default_model == "chosen-model"
        assert client.base_url == "https://example.test/v1"
        assert client.tool_registry is registry
        assert client.get_models() == []
        assert client.get_models() == []
        assert asyncio.run(client.get_models_async()) == []
        assert asyncio.run(client.get_models_async()) == []

    openai.assert_called_once_with(base_url="https://example.test/v1", api_key="secret")
    async_openai.assert_called_once_with(base_url="https://example.test/v1", api_key="secret")


def test_configuration_comes_from_environment(monkeypatch):
    """Omitted model and URL values come from the environment."""
    monkeypatch.setenv("DEFAULT_MODEL", "environment-model")
    monkeypatch.setenv("BASE_URL", "https://environment.test/v1")
    monkeypatch.setenv("CONTEXT_WINDOW", "32768")

    client = OpenAIBackend()

    assert client.default_model == "environment-model"
    assert client.base_url == "https://environment.test/v1"
    assert client.context_window == 32768


def test_configuration_falls_back_to_built_in_defaults():
    """Omitted model and URL values use the built-in local defaults."""
    client = OpenAIBackend()

    assert client.default_model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert client.base_url == "http://localhost:8000/v1"


def test_api_key_comes_from_environment_or_falls_back(monkeypatch):
    """Omitted credentials prefer the environment and otherwise use the local key."""
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    with patch("loop.backend.openai.OpenAI") as openai:
        openai.return_value.models.list.return_value = []
        OpenAIBackend().get_models()
    assert openai.call_args.kwargs["api_key"] == "environment-key"

    monkeypatch.delenv("OPENAI_API_KEY")
    with patch("loop.backend.openai.OpenAI") as openai:
        openai.return_value.models.list.return_value = []
        OpenAIBackend().get_models()
    assert openai.call_args.kwargs["api_key"] == "local-api-key"


@pytest.mark.parametrize("context_window", [0, -1])
def test_context_window_must_be_positive(context_window):
    """Explicit context limits reject zero and negative values."""
    with pytest.raises(ValueError, match="positive integer"):
        OpenAIBackend(context_window=context_window)


def test_models_are_listed_from_the_backend():
    """Model listing returns a concrete list from the synchronous SDK."""
    models = [
        SimpleNamespace(id="first", model_extra=None),
        SimpleNamespace(id="second", model_extra=None),
    ]
    sdk = Mock()
    sdk.models.list.return_value = iter(models)

    with patch("loop.backend.openai.OpenAI", return_value=sdk):
        assert OpenAIBackend().get_models() == [ModelInfo(id="first"), ModelInfo(id="second")]

    sdk.models.list.assert_called_once_with(timeout=2.0)


def test_models_are_listed_asynchronously_from_the_backend():
    """Async model listing returns a concrete list from the asynchronous SDK."""
    models = [
        SimpleNamespace(id="first", model_extra=None),
        SimpleNamespace(id="second", model_extra=None),
    ]
    sdk = Mock()
    sdk.models.list = AsyncMock(return_value=iter(models))

    with patch("loop.backend.openai.AsyncOpenAI", return_value=sdk):
        assert asyncio.run(OpenAIBackend().get_models_async()) == [
            ModelInfo(id="first"),
            ModelInfo(id="second"),
        ]

    sdk.models.list.assert_awaited_once_with(timeout=2.0)


def test_context_window_is_discovered_from_matching_model_metadata():
    """vLLM model metadata supplies and caches the deployed context limit."""
    sdk = Mock()
    sdk.models.list.return_value = [
        SimpleNamespace(id="other", model_extra={"max_model_len": 1000}),
        SimpleNamespace(id="default", model_extra={"max_model_len": 65536}),
    ]
    client = OpenAIBackend("default")

    with patch("loop.backend.openai.OpenAI", return_value=sdk):
        assert client.context_window == 65536
        assert client.context_window == 65536

    sdk.models.list.assert_called_once_with(timeout=2.0)

    direct = Mock()
    direct.models.list.return_value = [
        SimpleNamespace(id="default", model_extra={"max_model_len": 131072})
    ]
    with patch("loop.backend.openai.OpenAI", return_value=direct):
        assert OpenAIBackend("default").context_window == 131072

    selected = Mock()
    selected.models.list.return_value = [
        SimpleNamespace(id="served-model", model_extra={"max_model_len": 262144})
    ]
    with patch("loop.backend.openai.OpenAI", return_value=selected):
        client = OpenAIBackend("requested-model")
        assert client.get_context_window("served-model") == 262144
        assert client.get_context_window("served-model") == 262144

    selected.models.list.assert_called_once_with(timeout=2.0)


def test_context_window_discovery_gracefully_handles_unavailable_metadata():
    """Missing models and connection failures leave the context limit unknown."""
    unavailable = Mock()
    unavailable.models.list.side_effect = APIConnectionError(
        request=httpx.Request("GET", "https://example.test/v1/models")
    )
    with patch("loop.backend.openai.OpenAI", return_value=unavailable):
        assert OpenAIBackend("default").context_window is None

    missing = Mock()
    missing.models.list.return_value = [
        SimpleNamespace(id="other", model_extra={"max_model_len": 1000}),
        SimpleNamespace(id="default", model_extra={}),
    ]
    with patch("loop.backend.openai.OpenAI", return_value=missing):
        assert OpenAIBackend("default").context_window is None

    absent = Mock()
    absent.models.list.return_value = [
        SimpleNamespace(id="other", model_extra={"max_model_len": 1000})
    ]
    with patch("loop.backend.openai.OpenAI", return_value=absent):
        assert OpenAIBackend("default").context_window is None

    malformed = Mock()
    malformed.models.list.return_value = [
        SimpleNamespace(id="default", model_extra={"max_model_len": "invalid"})
    ]
    with patch("loop.backend.openai.OpenAI", return_value=malformed):
        assert OpenAIBackend("default").context_window is None


def test_prompt_tokens_use_the_server_tokenizer_when_available():
    """Prompt counting uses vLLM's tokenizer without adding a model special token."""
    response = Mock()
    response.json.return_value = {"count": 3}
    with patch("loop.backend.openai.httpx.post", return_value=response) as post:
        backend = OpenAIBackend("model", "http://localhost:8000/v1", "key")
        assert (
            backend.count_tokens("Hi", model="active") == 3
        )

    post.assert_called_once_with(
        "http://localhost:8000/tokenize",
        json={"model": "active", "prompt": "Hi", "add_special_tokens": False},
        headers={"Authorization": "Bearer key"},
        timeout=2.0,
    )
    response.raise_for_status.assert_called_once_with()


def test_prompt_tokenizer_preserves_a_nonstandard_api_base_path():
    """Tokenizer requests only remove the conventional trailing v1 path."""
    response = Mock()
    response.json.return_value = {"count": 1}
    with patch("loop.backend.openai.httpx.post", return_value=response) as post:
        assert OpenAIBackend(base_url="https://example.test/api").count_tokens("x") == 1

    assert post.call_args.args[0] == "https://example.test/api/tokenize"


def test_async_model_metadata_and_tokenizer_use_the_selected_model():
    """Async metadata and tokenization mirror the model-aware synchronous helpers."""
    sdk = Mock()
    sdk.models.list = AsyncMock(
        return_value=[SimpleNamespace(id="active", model_extra={"max_model_len": 65536})]
    )
    response = Mock()
    response.json.return_value = {"count": 4}
    http = AsyncMock()
    http.__aenter__.return_value.post = AsyncMock(return_value=response)
    client = OpenAIBackend("default", "http://localhost:8000/v1", "key")

    with (
        patch("loop.backend.openai.AsyncOpenAI", return_value=sdk),
        patch("loop.backend.openai.httpx.AsyncClient", return_value=http),
    ):
        assert asyncio.run(client.get_context_window_async("active")) == 65536
        assert asyncio.run(client.get_context_window_async("active")) == 65536
        assert asyncio.run(client.count_tokens_async("Hi", model="active")) == 4

    http.__aenter__.return_value.post.assert_awaited_once_with(
        "http://localhost:8000/tokenize",
        json={"model": "active", "prompt": "Hi", "add_special_tokens": False},
        headers={"Authorization": "Bearer key"},
        timeout=2.0,
    )
    sdk.models.list.assert_awaited_once_with(timeout=2.0)


def test_async_usage_helpers_gracefully_handle_configuration_and_failures():
    """Async helpers honor explicit limits and return unknown for unavailable data."""
    assert asyncio.run(OpenAIBackend(context_window=4096).get_context_window_async()) == 4096

    sdk = Mock()
    sdk.models.list = AsyncMock(
        side_effect=APIConnectionError(
            request=httpx.Request("GET", "https://example.test/v1/models")
        )
    )
    response = Mock()
    response.raise_for_status.side_effect = httpx.HTTPError("unavailable")
    http = AsyncMock()
    http.__aenter__.return_value.post = AsyncMock(return_value=response)
    client = OpenAIBackend("default", "https://example.test/api", "key")

    with (
        patch("loop.backend.openai.AsyncOpenAI", return_value=sdk),
        patch("loop.backend.openai.httpx.AsyncClient", return_value=http),
    ):
        assert asyncio.run(client.get_context_window_async()) is None
        assert asyncio.run(client.count_tokens_async("Hi")) is None

    assert http.__aenter__.return_value.post.await_args.args[0] == (
        "https://example.test/api/tokenize"
    )


@pytest.mark.parametrize("payload", [{}, {"count": "invalid"}])
def test_prompt_counting_gracefully_handles_unavailable_counts(payload):
    """Unsupported and malformed tokenizer responses leave prompt usage unknown."""
    response = Mock()
    response.json.return_value = payload
    with patch("loop.backend.openai.httpx.post", return_value=response):
        assert OpenAIBackend().count_tokens("Hi") is None


def test_sync_response_forwards_schema_streaming_and_model_selection():
    """Synchronous requests include tool schemas and honor a model override."""
    registry = Mock()
    definition = SimpleNamespace(
        name="demo", description="Demo.", parameters={"type": "object"}, strict=True
    )
    registry.definitions.return_value = [definition]
    sdk = Mock()
    sdk.responses.create.return_value = []
    client = OpenAIBackend("default", tool_registry=registry)

    with patch("loop.backend.openai.OpenAI", return_value=sdk):
        result = list(
            client.get_response(
                "hello",
                instructions="Follow the project rules.",
                stream=True,
                model="override",
            )
        )

    assert result == []
    sdk.responses.create.assert_called_once_with(
        model="override",
        input="hello",
        instructions="Follow the project rules.",
        stream=True,
        stream_options={"include_usage": True},
        tools=[
            {
                "type": "function",
                "name": "demo",
                "description": "Demo.",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ],
    )


def test_async_response_uses_default_model():
    """Asynchronous requests use the default model when none is supplied."""
    registry = Mock()
    registry.definitions.return_value = []
    sdk = Mock()
    sdk.responses.create = AsyncMock(
        return_value=SimpleNamespace(
            output=[], output_text="", usage=None, model="served-model"
        )
    )
    client = OpenAIBackend("default", tool_registry=registry)

    with patch("loop.backend.openai.AsyncOpenAI", return_value=sdk):
        result = asyncio.run(
            collect_events(
                client.get_response_async([Message(role="user", content="hi")])
            )
        )

    assert result == [ResponseCompleted(model="served-model")]
    sdk.responses.create.assert_awaited_once_with(
        model="default",
        input=[{"role": "user", "content": "hi"}],
        instructions=None,
        stream=False,
        stream_options={"include_usage": True},
        tools=[],
    )


def test_completed_response_normalizes_items_and_serializes_local_history():
    """Completed responses and request history cross the boundary as local models."""
    reasoning = ResponseReasoningItem.model_validate(
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "think"}],
        }
    )
    message = ResponseOutputMessage.model_validate(
        {
            "id": "message_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "answer", "annotations": []},
                {"type": "refusal", "refusal": "ignored"},
            ],
        }
    )
    sdk_call = ResponseFunctionToolCall(
        id="fc_1",
        call_id="call_1",
        name="demo",
        arguments="{}",
        type="function_call",
        status="completed",
    )
    sdk = Mock()
    sdk.responses.create.return_value = SimpleNamespace(
        output=[reasoning, message, sdk_call, SimpleNamespace(type="unknown")],
        output_text="authoritative answer",
        usage=SimpleNamespace(total_tokens=21),
        model="served-model",
    )
    registry = Mock()
    registry.definitions.return_value = []
    history = [
        Message(role="user", content="hello"),
        Reasoning(content="", id="reasoning_old"),
        Reasoning(content="prior"),
        ToolCall(call_id="call_old", name="demo", arguments="{}", id="fc_old"),
        ToolCall(call_id="call_no_id", name="demo", arguments="{}"),
        ToolResult(call_id="call_old", output="done"),
    ]

    with patch("loop.backend.openai.OpenAI", return_value=sdk):
        events = list(OpenAIBackend(tool_registry=registry).get_response(history))

    local_call = ToolCall(call_id="call_1", name="demo", arguments="{}", id="fc_1")
    items = (
        Reasoning(content="think", id="reasoning_1"),
        Message(role="assistant", content="answer"),
        local_call,
    )
    assert events == [
        ReasoningCompleted(text="think"),
        ToolCallCompleted(call=local_call),
        AnswerCompleted(text="authoritative answer"),
        ResponseCompleted(
            items=items,
            usage=Usage(total_tokens=21),
            model="served-model",
            answer="authoritative answer",
            reasoning="think",
        ),
    ]
    assert sdk.responses.create.call_args.kwargs["input"] == [
        {"role": "user", "content": "hello"},
        {"type": "reasoning", "summary": [], "content": [], "id": "reasoning_old"},
        {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "prior"}],
        },
        {
            "type": "function_call",
            "call_id": "call_old",
            "name": "demo",
            "arguments": "{}",
            "status": "completed",
            "id": "fc_old",
        },
        {
            "type": "function_call",
            "call_id": "call_no_id",
            "name": "demo",
            "arguments": "{}",
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call_old", "output": "done"},
    ]


def test_completed_response_emits_only_final_reasoning():
    """A non-streaming response emits only one final reasoning event."""
    first = ResponseReasoningItem.model_validate(
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "first"}],
        }
    )
    final = ResponseReasoningItem.model_validate(
        {
            "id": "reasoning_2",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "final"}],
        }
    )
    sdk = Mock()
    sdk.responses.create.return_value = SimpleNamespace(
        output=[first, final], output_text="", usage=None, model=None
    )

    with patch("loop.backend.openai.OpenAI", return_value=sdk):
        events = list(OpenAIBackend().get_response("hello"))

    assert events == [
        ReasoningCompleted(text="final"),
        ResponseCompleted(
            items=(
                Reasoning(content="first", id="reasoning_1"),
                Reasoning(content="final", id="reasoning_2"),
            ),
            reasoning="final",
        ),
    ]


def test_completed_response_ignores_empty_text_and_invalid_metadata():
    """Empty output items remain history while malformed metadata stays unknown."""
    reasoning = ResponseReasoningItem(id="r", type="reasoning", summary=[], content=[])
    message = ResponseOutputMessage(
        id="m", type="message", role="assistant", status="completed", content=[]
    )
    sdk = Mock()
    sdk.responses.create.return_value = SimpleNamespace(
        output=[reasoning, message],
        output_text="",
        usage=SimpleNamespace(total_tokens=-1),
        model=42,
    )

    with patch("loop.backend.openai.OpenAI", return_value=sdk):
        events = list(OpenAIBackend().get_response("hello"))

    assert events == [
        ResponseCompleted(
            items=(Reasoning(content="", id="r"), Message(role="assistant", content=""))
        )
    ]


def test_response_rejects_non_local_history_items():
    """Request serialization rejects values outside the local conversation model."""
    with pytest.raises(TypeError, match="Unsupported conversation item"):
        list(OpenAIBackend().get_response([{"role": "user", "content": "hello"}]))


def test_streaming_response_normalizes_provider_events():
    """Streaming provider events become the same event types as completed responses."""
    call = ResponseFunctionToolCall(
        id="fc_1",
        call_id="call_1",
        name="demo",
        arguments="{}",
        type="function_call",
        status="completed",
    )
    reasoning = ResponseReasoningItem.model_validate(
        {
            "id": "r",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "final thought"}],
        }
    )
    message = ResponseOutputMessage.model_validate(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "final answer", "annotations": []}
            ],
        }
    )
    provider_events = [
        ResponseReasoningTextDeltaEvent(
            delta="think",
            item_id="r",
            output_index=0,
            content_index=0,
            sequence_number=1,
            type="response.reasoning_text.delta",
        ),
        ResponseTextDeltaEvent(
            delta="answer",
            item_id="m",
            output_index=1,
            content_index=0,
            sequence_number=2,
            type="response.output_text.delta",
            logprobs=[],
        ),
        ResponseReasoningTextDoneEvent(
            text="final thought",
            item_id="r",
            output_index=0,
            content_index=0,
            sequence_number=3,
            type="response.reasoning_text.done",
        ),
        ResponseTextDoneEvent(
            text="final answer",
            item_id="m",
            output_index=1,
            content_index=0,
            sequence_number=4,
            type="response.output_text.done",
            logprobs=[],
        ),
        ResponseOutputItemDoneEvent(
            item=reasoning, output_index=0, sequence_number=5, type="response.output_item.done"
        ),
        ResponseOutputItemDoneEvent(
            item=message, output_index=1, sequence_number=6, type="response.output_item.done"
        ),
        ResponseOutputItemDoneEvent(
            item=call, output_index=2, sequence_number=7, type="response.output_item.done"
        ),
        ResponseOutputItemDoneEvent.model_construct(
            item=SimpleNamespace(type="unknown"),
            output_index=3,
            sequence_number=8,
            type="response.output_item.done",
        ),
        sdk_completion_event(output=[reasoning, message, call]),
        SimpleNamespace(type="unknown"),
    ]
    sdk = Mock()
    sdk.responses.create.return_value = provider_events

    with patch("loop.backend.openai.OpenAI", return_value=sdk):
        events = list(OpenAIBackend().get_response("hello", stream=True))

    local_call = ToolCall(call_id="call_1", name="demo", arguments="{}", id="fc_1")
    assert events == [
        ReasoningDelta(text="think"),
        AnswerDelta(text="answer"),
        ToolCallCompleted(call=local_call),
        ResponseCompleted(
            items=(
                Reasoning(content="final thought", id="r"),
                Message(role="assistant", content="final answer"),
                local_call,
            ),
            usage=Usage(total_tokens=12),
            model="served-model",
            answer="final answer",
            reasoning="final thought",
        ),
    ]


def test_async_streaming_response_uses_the_normalized_event_contract():
    """Asynchronous streaming returns the same local response events."""
    provider_events = [
        ResponseTextDeltaEvent(
            delta="answer",
            item_id="m",
            output_index=0,
            content_index=0,
            sequence_number=1,
            type="response.output_text.delta",
            logprobs=[],
        ),
        sdk_completion_event(total_tokens=None),
    ]
    sdk = Mock()
    sdk.responses.create = AsyncMock(return_value=AsyncEvents(provider_events))

    with patch("loop.backend.openai.AsyncOpenAI", return_value=sdk):
        events = asyncio.run(
            collect_events(OpenAIBackend().get_response_async("hello", stream=True))
        )

    assert events == [AnswerDelta(text="answer"), ResponseCompleted(model="served-model")]


def test_async_streaming_yields_before_the_provider_stream_completes():
    """The first translated event is observable without consuming later provider events."""
    provider_events = AsyncEvents(
        [
            ResponseTextDeltaEvent(
                delta="first",
                item_id="m",
                output_index=0,
                content_index=0,
                sequence_number=1,
                type="response.output_text.delta",
                logprobs=[],
            ),
            ResponseTextDeltaEvent(
                delta="second",
                item_id="m",
                output_index=0,
                content_index=0,
                sequence_number=2,
                type="response.output_text.delta",
                logprobs=[],
            ),
        ]
    )
    sdk = Mock()
    sdk.responses.create = AsyncMock(return_value=provider_events)

    async def read_first_event():
        """Read and close the first response event."""
        events = OpenAIBackend().get_response_async("hello", stream=True)
        first = await anext(events)
        await events.aclose()
        return first

    with patch("loop.backend.openai.AsyncOpenAI", return_value=sdk):
        first = asyncio.run(read_first_event())

    assert first == AnswerDelta(text="first")
    assert provider_events.yielded == 1
