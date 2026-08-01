"""Client implementation for connecting to the LLM backend."""

import os

import httpx
from openai import APIError, AsyncOpenAI, BaseModel, OpenAI
from openai.types import Model
from openai.types.responses import ResponseInputParam

from .tooling import ToolRegistry
from .tooling import tool_registry as default_tool_registry

_BASE_URL = "http://localhost:8000/v1"
_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
_DEFAULT_API_KEY = "local-api-key"


class Client:
    """Manage synchronous and asynchronous clients for the LLM backend.

    Args:
        default_model: Model identifier used when a request does not specify one.
        base_url: Base URL of the OpenAI-compatible backend.
        api_key: API key for the backend. Defaults to ``OPENAI_API_KEY`` or a local key.
        tool_registry: Registry supplying tool schemas for requests.
        context_window: Deployed model context limit. Defaults to ``CONTEXT_WINDOW`` or
            best-effort model metadata discovery.

    Raises:
        ValueError: If the configured context window is not a positive integer.
    """

    _client: OpenAI | None
    _async_client: AsyncOpenAI | None
    _base_url: str
    _api_key: str
    _default_model: str
    _tool_registry: ToolRegistry
    _configured_context_window: int | None
    _context_windows: dict[str, int | None]

    def __init__(
        self,
        default_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tool_registry: ToolRegistry | None = None,
        context_window: int | None = None,
    ) -> None:
        self._client = None
        self._async_client = None
        self._default_model = default_model or os.getenv("DEFAULT_MODEL", _MODEL)
        self._base_url = base_url or os.getenv("BASE_URL", _BASE_URL)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", _DEFAULT_API_KEY)
        self._tool_registry = tool_registry or default_tool_registry
        configured_window = (
            context_window if context_window is not None else os.getenv("CONTEXT_WINDOW")
        )
        self._configured_context_window = (
            int(configured_window) if configured_window is not None else None
        )
        if self._configured_context_window is not None and self._configured_context_window <= 0:
            raise ValueError("Context window must be a positive integer.")
        self._context_windows = {}

    @property
    def tool_registry(self) -> ToolRegistry:
        """Return the registry used for declarations and runtime dispatch.

        Returns:
            The configured tool registry.
        """
        return self._tool_registry

    @property
    def default_model(self) -> str:
        """Return the model used when a request does not specify one.

        Returns:
            The default model identifier.
        """
        return self._default_model

    @property
    def base_url(self) -> str:
        """Return the base URL of the configured LLM backend.

        Returns:
            The configured backend URL.
        """
        return self._base_url

    @property
    def context_window(self) -> int | None:
        """Return the default model context limit when available.

        Returns:
            The context limit, or ``None`` when it cannot be determined.
        """
        return self.get_context_window()

    def get_client(self) -> OpenAI:
        """Return the lazily initialized synchronous OpenAI client.

        Returns:
            The synchronous OpenAI client.
        """
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def get_async_client(self) -> AsyncOpenAI:
        """Return the lazily initialized asynchronous OpenAI client.

        Returns:
            The asynchronous OpenAI client.
        """
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._async_client

    def get_models(self) -> list[Model]:
        """Return the models available from the configured backend.

        Returns:
            The available models.
        """
        return list(self.get_client().models.list(timeout=2.0))

    async def get_models_async(self) -> list[Model]:
        """Asynchronously return the models available from the configured backend.

        Returns:
            The available models.
        """
        return list(await self.get_async_client().models.list(timeout=2.0))

    def get_response(
        self,
        input: str | ResponseInputParam,  # pylint: disable=redefined-builtin
        instructions: str | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> BaseModel:
        """Create a synchronous response from the configured backend.

        Args:
            input: Text or structured input to send to the model.
            instructions: System or developer instructions to apply to the request.
            stream: Whether to return a streaming response.
            model: Model identifier to use instead of the default model.

        Returns:
            The response produced by the backend.
        """
        response = self.get_client().responses.create(
            model=model or self._default_model,
            input=input,
            instructions=instructions,
            stream=stream,
            stream_options={"include_usage": True},
            tools=self._tool_registry.schemas(),
        )
        return response

    async def get_response_async(
        self,
        input: str | ResponseInputParam,  # pylint: disable=redefined-builtin
        instructions: str | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> BaseModel:
        """Create an asynchronous response from the configured backend.

        Args:
            input: Text or structured input to send to the model.
            instructions: System or developer instructions to apply to the request.
            stream: Whether to return a streaming response.
            model: Model identifier to use instead of the default model.

        Returns:
            The response produced by the backend.
        """
        response = await self.get_async_client().responses.create(
            model=model or self._default_model,
            input=input,
            instructions=instructions,
            stream=stream,
            stream_options={"include_usage": True},
            tools=self._tool_registry.schemas(),
        )
        return response

    def get_context_window(self, model: str | None = None) -> int | None:
        """Return the deployed context limit for a selected model when available.

        Args:
            model: Model identifier to inspect instead of the default model.

        Returns:
            The configured or discovered context limit, or ``None`` when unavailable.
        """
        if self._configured_context_window is not None:
            return self._configured_context_window
        selected_model = model or self._default_model
        if selected_model not in self._context_windows:
            try:
                models = self.get_models()
            except APIError:
                models = []
            self._context_windows[selected_model] = self._context_window_from_models(
                models, selected_model
            )
        return self._context_windows[selected_model]

    async def get_context_window_async(self, model: str | None = None) -> int | None:
        """Asynchronously return the selected model's deployed context limit.

        Args:
            model: Model identifier to inspect instead of the default model.

        Returns:
            The configured or discovered context limit, or ``None`` when unavailable.
        """
        if self._configured_context_window is not None:
            return self._configured_context_window
        selected_model = model or self._default_model
        if selected_model not in self._context_windows:
            try:
                models = await self.get_models_async()
            except APIError:
                models = []
            self._context_windows[selected_model] = self._context_window_from_models(
                models, selected_model
            )
        return self._context_windows[selected_model]

    def count_tokens(self, prompt: str, model: str | None = None) -> int | None:
        """Count text tokens for the selected model when available.

        Args:
            prompt: Text to tokenize.
            model: Model identifier to use instead of the default model.

        Returns:
            The token count, or ``None`` when tokenization fails or is unavailable.
        """
        base_url = self._base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        try:
            response = httpx.post(
                f"{base_url}/tokenize",
                json={
                    "model": model or self._default_model,
                    "prompt": prompt,
                    "add_special_tokens": False,
                },
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=2.0,
            )
            response.raise_for_status()
            return int(response.json()["count"])
        except httpx.HTTPError, KeyError, TypeError, ValueError:
            return None

    async def count_tokens_async(self, prompt: str, model: str | None = None) -> int | None:
        """Asynchronously count text tokens for the selected model when available.

        Args:
            prompt: Text to tokenize.
            model: Model identifier to use instead of the default model.

        Returns:
            The token count, or ``None`` when tokenization fails or is unavailable.
        """
        base_url = self._base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/tokenize",
                    json={
                        "model": model or self._default_model,
                        "prompt": prompt,
                        "add_special_tokens": False,
                    },
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=2.0,
                )
            response.raise_for_status()
            return int(response.json()["count"])
        except httpx.HTTPError, KeyError, TypeError, ValueError:
            return None

    @staticmethod
    def _context_window_from_models(models, model_name: str) -> int | None:
        """Extract a model's context limit from a model-list response."""
        for model in models:
            if model.id != model_name:
                continue
            max_model_len = getattr(model, "max_model_len", None)
            if max_model_len is None and model.model_extra is not None:
                max_model_len = model.model_extra.get("max_model_len")
            if max_model_len is None:
                return None
            try:
                return int(max_model_len)
            except TypeError, ValueError:
                return None
        return None
