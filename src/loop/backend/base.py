"""Define the backend contract consumed by conversation loops."""

from collections.abc import AsyncIterator, Iterable
from typing import Protocol, runtime_checkable

from ..models import ConversationItem, ModelInfo, ResponseEvent
from ..tooling import ToolRegistry


@runtime_checkable
class Backend(Protocol):
    """Provide the capabilities required by a conversation loop."""

    @property
    def tool_registry(self) -> ToolRegistry:
        """Return the registry used for declarations and dispatch.

        Returns:
            ToolRegistry: The configured tool registry.
        """

    @property
    def default_model(self) -> str:
        """Return the model used when a request does not specify one.

        Returns:
            str: The default model identifier.
        """

    @property
    def context_window(self) -> int | None:
        """Return the default model's deployed context limit when available.

        Returns:
            int | None: Context limit, or ``None`` when unavailable.
        """

    def get_models(self) -> list[ModelInfo]:
        """Return models available from the backend.

        Returns:
            list[ModelInfo]: Available model descriptions.
        """

    async def get_models_async(self) -> list[ModelInfo]:
        """Asynchronously return models available from the backend.

        Returns:
            list[ModelInfo]: Available model descriptions.
        """

    def get_response(
        self,
        input: str | Iterable[ConversationItem],  # pylint: disable=redefined-builtin
        instructions: str | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> Iterable[ResponseEvent]:
        """Return normalized response events.

        Args:
            input (str | Iterable[ConversationItem]): Text or conversation history to send.
            instructions (str | None): System or developer instructions for the request.
            stream (bool): Whether events should be produced incrementally.
            model (str | None): Model identifier to use instead of the default model.

        Returns:
            Iterable[ResponseEvent]: Response events in output order.
        """

    async def get_response_async(
        self,
        input: str | Iterable[ConversationItem],  # pylint: disable=redefined-builtin
        instructions: str | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> AsyncIterator[ResponseEvent]:
        """Asynchronously yield response events.

        Args:
            input (str | Iterable[ConversationItem]): Text or conversation history to send.
            instructions (str | None): System or developer instructions for the request.
            stream (bool): Whether events should be produced incrementally.
            model (str | None): Model identifier to use instead of the default model.

        Yields:
            ResponseEvent: Response events in output order.
        """

    def get_context_window(self, model: str | None = None) -> int | None:
        """Return a model's deployed context limit when available.

        Args:
            model (str | None): Model identifier to inspect instead of the default model.

        Returns:
            int | None: Context limit, or ``None`` when unavailable.
        """

    async def get_context_window_async(self, model: str | None = None) -> int | None:
        """Asynchronously return a model's deployed context limit when available.

        Args:
            model (str | None): Model identifier to inspect instead of the default model.

        Returns:
            int | None: Context limit, or ``None`` when unavailable.
        """

    def count_tokens(self, prompt: str, model: str | None = None) -> int | None:
        """Count text tokens for a selected model when available.

        Args:
            prompt (str): Text to tokenize.
            model (str | None): Model identifier to use instead of the default model.

        Returns:
            int | None: Token count, or ``None`` when unavailable.
        """

    async def count_tokens_async(self, prompt: str, model: str | None = None) -> int | None:
        """Asynchronously count text tokens for a selected model when available.

        Args:
            prompt (str): Text to tokenize.
            model (str | None): Model identifier to use instead of the default model.

        Returns:
            int | None: Token count, or ``None`` when unavailable.
        """
