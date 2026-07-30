"""Tests for client configuration and request forwarding."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from loop.client import Client
from loop.tooling import ToolRegistry


@pytest.fixture(autouse=True)
def isolate_client_environment(monkeypatch):
    """Prevent host configuration from influencing client tests."""
    for variable in ("DEFAULT_MODEL", "BASE_URL", "OPENAI_API_KEY"):
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

    client = Client()

    assert client.default_model == "environment-model"
    assert client.base_url == "https://environment.test/v1"


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
