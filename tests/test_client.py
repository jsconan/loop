"""Tests for client configuration and request forwarding."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from openai import APIConnectionError

from loop.client import Client
from loop.tooling import ToolRegistry


@pytest.fixture(autouse=True)
def isolate_client_environment(monkeypatch):
    """Prevent host configuration from influencing client tests."""
    for variable in ("DEFAULT_MODEL", "BASE_URL", "OPENAI_API_KEY", "CONTEXT_WINDOW"):
        monkeypatch.delenv(variable, raising=False)


def test_configuration_and_lazy_clients_use_explicit_credentials(monkeypatch):
    """Configured values are exposed and used once for each lazy SDK client."""
    monkeypatch.setenv("DEFAULT_MODEL", "environment-model")
    monkeypatch.setenv("BASE_URL", "https://environment.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    registry = ToolRegistry()
    sync_sdk = Mock()
    async_sdk = Mock()

    with (
        patch("loop.client.OpenAI", return_value=sync_sdk) as openai,
        patch("loop.client.AsyncOpenAI", return_value=async_sdk) as async_openai,
    ):
        client = Client("chosen-model", "https://example.test/v1", "secret", registry)

        assert client.default_model == "chosen-model"
        assert client.base_url == "https://example.test/v1"
        assert client.tool_registry is registry
        assert client.get_client() is sync_sdk
        assert client.get_client() is sync_sdk
        assert client.get_async_client() is async_sdk
        assert client.get_async_client() is async_sdk

    openai.assert_called_once_with(base_url="https://example.test/v1", api_key="secret")
    async_openai.assert_called_once_with(base_url="https://example.test/v1", api_key="secret")


def test_configuration_comes_from_environment(monkeypatch):
    """Omitted model and URL values come from the environment."""
    monkeypatch.setenv("DEFAULT_MODEL", "environment-model")
    monkeypatch.setenv("BASE_URL", "https://environment.test/v1")
    monkeypatch.setenv("CONTEXT_WINDOW", "32768")

    client = Client()

    assert client.default_model == "environment-model"
    assert client.base_url == "https://environment.test/v1"
    assert client.context_window == 32768


def test_configuration_falls_back_to_built_in_defaults():
    """Omitted model and URL values use the built-in local defaults."""
    client = Client()

    assert client.default_model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert client.base_url == "http://localhost:8000/v1"


def test_api_key_comes_from_environment_or_falls_back(monkeypatch):
    """Omitted credentials prefer the environment and otherwise use the local key."""
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    with patch("loop.client.OpenAI") as openai:
        Client().get_client()
    assert openai.call_args.kwargs["api_key"] == "environment-key"

    monkeypatch.delenv("OPENAI_API_KEY")
    with patch("loop.client.OpenAI") as openai:
        Client().get_client()
    assert openai.call_args.kwargs["api_key"] == "local-api-key"


@pytest.mark.parametrize("context_window", [0, -1])
def test_context_window_must_be_positive(context_window):
    """Explicit context limits reject zero and negative values."""
    with pytest.raises(ValueError, match="positive integer"):
        Client(context_window=context_window)


def test_models_are_listed_from_the_backend():
    """Model listing returns a concrete list from the synchronous SDK."""
    models = [SimpleNamespace(id="first"), SimpleNamespace(id="second")]
    sdk = Mock()
    sdk.models.list.return_value = iter(models)

    with patch("loop.client.OpenAI", return_value=sdk):
        assert Client().get_models() == models

    sdk.models.list.assert_called_once_with(timeout=2.0)


def test_models_are_listed_asynchronously_from_the_backend():
    """Async model listing returns a concrete list from the asynchronous SDK."""
    models = [SimpleNamespace(id="first"), SimpleNamespace(id="second")]
    sdk = Mock()
    sdk.models.list = AsyncMock(return_value=iter(models))

    with patch("loop.client.AsyncOpenAI", return_value=sdk):
        assert asyncio.run(Client().get_models_async()) == models

    sdk.models.list.assert_awaited_once_with(timeout=2.0)


def test_context_window_is_discovered_from_matching_model_metadata():
    """vLLM model metadata supplies and caches the deployed context limit."""
    sdk = Mock()
    sdk.models.list.return_value = [
        SimpleNamespace(id="other", max_model_len=1000, model_extra={}),
        SimpleNamespace(id="default", max_model_len=None, model_extra={"max_model_len": 65536}),
    ]
    client = Client("default")

    with patch("loop.client.OpenAI", return_value=sdk):
        assert client.context_window == 65536
        assert client.context_window == 65536

    sdk.models.list.assert_called_once_with(timeout=2.0)

    direct = Mock()
    direct.models.list.return_value = [
        SimpleNamespace(id="default", max_model_len=131072, model_extra={})
    ]
    with patch("loop.client.OpenAI", return_value=direct):
        assert Client("default").context_window == 131072

    selected = Mock()
    selected.models.list.return_value = [
        SimpleNamespace(id="served-model", max_model_len=262144, model_extra={})
    ]
    with patch("loop.client.OpenAI", return_value=selected):
        client = Client("requested-model")
        assert client.get_context_window("served-model") == 262144
        assert client.get_context_window("served-model") == 262144

    selected.models.list.assert_called_once_with(timeout=2.0)


def test_context_window_discovery_gracefully_handles_unavailable_metadata():
    """Missing models and connection failures leave the context limit unknown."""
    unavailable = Mock()
    unavailable.models.list.side_effect = APIConnectionError(
        request=httpx.Request("GET", "https://example.test/v1/models")
    )
    with patch("loop.client.OpenAI", return_value=unavailable):
        assert Client("default").context_window is None

    missing = Mock()
    missing.models.list.return_value = [
        SimpleNamespace(id="other", max_model_len=1000, model_extra={}),
        SimpleNamespace(id="default", max_model_len=None, model_extra={}),
    ]
    with patch("loop.client.OpenAI", return_value=missing):
        assert Client("default").context_window is None

    absent = Mock()
    absent.models.list.return_value = [
        SimpleNamespace(id="other", max_model_len=1000, model_extra={})
    ]
    with patch("loop.client.OpenAI", return_value=absent):
        assert Client("default").context_window is None

    malformed = Mock()
    malformed.models.list.return_value = [
        SimpleNamespace(id="default", max_model_len="invalid", model_extra={})
    ]
    with patch("loop.client.OpenAI", return_value=malformed):
        assert Client("default").context_window is None


def test_prompt_tokens_use_the_server_tokenizer_when_available():
    """Prompt counting uses vLLM's tokenizer without adding a model special token."""
    response = Mock()
    response.json.return_value = {"count": 3}
    with patch("loop.client.httpx.post", return_value=response) as post:
        assert Client("model", "http://localhost:8000/v1", "key").count_tokens(
            "Hi", model="active"
        ) == 3

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
    with patch("loop.client.httpx.post", return_value=response) as post:
        assert Client(base_url="https://example.test/api").count_tokens("x") == 1

    assert post.call_args.args[0] == "https://example.test/api/tokenize"


def test_async_model_metadata_and_tokenizer_use_the_selected_model():
    """Async metadata and tokenization mirror the model-aware synchronous helpers."""
    sdk = Mock()
    sdk.models.list = AsyncMock(
        return_value=[SimpleNamespace(id="active", max_model_len=65536, model_extra={})]
    )
    response = Mock()
    response.json.return_value = {"count": 4}
    http = AsyncMock()
    http.__aenter__.return_value.post = AsyncMock(return_value=response)
    client = Client("default", "http://localhost:8000/v1", "key")

    with (
        patch("loop.client.AsyncOpenAI", return_value=sdk),
        patch("loop.client.httpx.AsyncClient", return_value=http),
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
    assert asyncio.run(Client(context_window=4096).get_context_window_async()) == 4096

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
    client = Client("default", "https://example.test/api", "key")

    with (
        patch("loop.client.AsyncOpenAI", return_value=sdk),
        patch("loop.client.httpx.AsyncClient", return_value=http),
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
    with patch("loop.client.httpx.post", return_value=response):
        assert Client().count_tokens("Hi") is None


def test_sync_response_forwards_schema_streaming_and_model_selection():
    """Synchronous requests include tool schemas and honor a model override."""
    registry = Mock()
    registry.schemas.return_value = [{"type": "function", "name": "demo"}]
    sdk = Mock()
    sdk.responses.create.return_value = "response"
    client = Client("default", tool_registry=registry)

    with patch("loop.client.OpenAI", return_value=sdk):
        result = client.get_response(
            "hello",
            instructions="Follow the project rules.",
            stream=True,
            model="override",
        )

    assert result == "response"
    sdk.responses.create.assert_called_once_with(
        model="override",
        input="hello",
        instructions="Follow the project rules.",
        stream=True,
        stream_options={"include_usage": True},
        tools=[{"type": "function", "name": "demo"}],
    )


def test_async_response_uses_default_model():
    """Asynchronous requests use the default model when none is supplied."""
    registry = Mock()
    registry.schemas.return_value = []
    sdk = Mock()
    sdk.responses.create = AsyncMock(return_value="async response")
    client = Client("default", tool_registry=registry)

    with patch("loop.client.AsyncOpenAI", return_value=sdk):
        result = asyncio.run(client.get_response_async([{"role": "user", "content": "hi"}]))

    assert result == "async response"
    sdk.responses.create.assert_awaited_once_with(
        model="default",
        input=[{"role": "user", "content": "hi"}],
        instructions=None,
        stream=False,
        stream_options={"include_usage": True},
        tools=[],
    )
