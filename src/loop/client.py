"""Client implementation for connecting to the LLM backend."""

import os

from openai import AsyncOpenAI, BaseModel, OpenAI
from openai.types.responses import ResponseInputParam

from .config import BASE_URL, MODEL
from .tools import TOOLS

_DEFAULT_API_KEY = "local-api-key"


class Client:
    """Manage synchronous and asynchronous clients for the LLM backend."""

    _client: OpenAI | None
    _async_client: AsyncOpenAI | None
    _base_url: str
    _api_key: str
    _default_model: str

    def __init__(
        self,
        default_model: str = MODEL,
        base_url: str = BASE_URL,
        api_key: str | None = None,
    ) -> None:
        self._client = None
        self._async_client = None
        self._default_model = default_model
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", _DEFAULT_API_KEY)

    @property
    def default_model(self) -> str:
        """Return the model used when a request does not specify one."""
        return self._default_model

    @property
    def base_url(self) -> str:
        """Return the base URL of the configured LLM backend."""
        return self._base_url

    def get_client(self) -> OpenAI:
        """Return the lazily initialized synchronous OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def get_async_client(self) -> AsyncOpenAI:
        """Return the lazily initialized asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._async_client

    def get_response(
        self,
        input: str | ResponseInputParam,  # pylint: disable=redefined-builtin
        stream: bool = False,
        model: str | None = None,
    ) -> BaseModel:
        """Create a synchronous response from the configured backend.

        Args:
            input: Text or structured input to send to the model.
            stream: Whether to return a streaming response.
            model: Model identifier to use instead of the default model.

        Returns:
            The response produced by the backend.
        """
        response = self.get_client().responses.create(
            model=model or self._default_model,
            input=input,
            stream=stream,
            stream_options={"include_usage": True},
            tools=TOOLS,
        )
        return response

    async def get_response_async(
        self,
        input: str | ResponseInputParam,  # pylint: disable=redefined-builtin
        stream: bool = False,
        model: str | None = None,
    ) -> BaseModel:
        """Create an asynchronous response from the configured backend.

        Args:
            input: Text or structured input to send to the model.
            stream: Whether to return a streaming response.
            model: Model identifier to use instead of the default model.

        Returns:
            The response produced by the backend.
        """
        response = await self.get_async_client().responses.create(
            model=model or self._default_model,
            input=input,
            stream=stream,
            stream_options={"include_usage": True},
            tools=TOOLS,
        )
        return response
