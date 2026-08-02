"""Define the backend contract consumed by conversation loops."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable

from ..models import ConversationItem, ModelInfo, ResponseEvent
from ..tooling import ToolRegistry


class Backend(ABC):
    """Provide common configuration and capabilities for conversation backends.

    Args:
        base_url (str | None): Service URL used by the backend.
        default_model (str | None): Model used when a request does not specify one.
        api_key (str | None): Credential used privately by the backend implementation.
        tool_registry (ToolRegistry): Registry used for declarations and dispatch.
    """

    def __init__(
        self,
        *,
        base_url: str | None,
        default_model: str | None,
        api_key: str | None,
        tool_registry: ToolRegistry,
    ) -> None:
        self._base_url = base_url
        self._default_model = default_model
        self._api_key = api_key
        self._tool_registry = tool_registry

    @property
    def tool_registry(self) -> ToolRegistry:
        """Return the registry used for declarations and dispatch.

        Returns:
            ToolRegistry: The configured tool registry.
        """
        return self._tool_registry

    @property
    def base_url(self) -> str | None:
        """Return the configured service URL.

        Returns:
            str | None: The service URL, or ``None`` when the client supplies it.
        """
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """Return the model used when a request does not specify one.

        Returns:
            str | None: The default model identifier, or ``None`` when unconfigured.
        """
        return self._default_model

    def _select_model(self, model: str | None) -> str:
        """Return the requested or default model, rejecting missing configuration."""
        selected_model = model or self._default_model
        if not selected_model:
            raise ValueError("No model was selected and the backend has no default model.")
        return selected_model

    @property
    @abstractmethod
    def context_window(self) -> int | None:
        """Return the default model's deployed context limit when available.

        Returns:
            int | None: Context limit, or ``None`` when unavailable.
        """

    @abstractmethod
    def get_models(self) -> list[ModelInfo]:
        """Return models available from the backend.

        Returns:
            list[ModelInfo]: Available model descriptions.
        """

    @abstractmethod
    async def get_models_async(self) -> list[ModelInfo]:
        """Asynchronously return models available from the backend.

        Returns:
            list[ModelInfo]: Available model descriptions.
        """

    @abstractmethod
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

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """

    @abstractmethod
    async def get_response_async(
        self,
        input: str | Iterable[ConversationItem],  # pylint: disable=redefined-builtin
        instructions: str | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> AsyncIterator[ResponseEvent]:
        """Asynchronously yield normalized response events.

        Args:
            input (str | Iterable[ConversationItem]): Text or conversation history to send.
            instructions (str | None): System or developer instructions for the request.
            stream (bool): Whether events should be produced incrementally.
            model (str | None): Model identifier to use instead of the default model.

        Yields:
            ResponseEvent: Response events in output order.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """

    @abstractmethod
    def get_context_window(self, model: str | None = None) -> int | None:
        """Return a model's deployed context limit when available.

        Args:
            model (str | None): Model identifier to inspect instead of the default model.

        Returns:
            int | None: Context limit, or ``None`` when unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """

    @abstractmethod
    async def get_context_window_async(self, model: str | None = None) -> int | None:
        """Asynchronously return a model's deployed context limit when available.

        Args:
            model (str | None): Model identifier to inspect instead of the default model.

        Returns:
            int | None: Context limit, or ``None`` when unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """

    @abstractmethod
    def count_tokens(self, prompt: str, model: str | None = None) -> int | None:
        """Count text tokens for a selected model when available.

        Args:
            prompt (str): Text to tokenize.
            model (str | None): Model identifier to use instead of the default model.

        Returns:
            int | None: Token count, or ``None`` when unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """

    @abstractmethod
    async def count_tokens_async(self, prompt: str, model: str | None = None) -> int | None:
        """Asynchronously count text tokens for a selected model when available.

        Args:
            prompt (str): Text to tokenize.
            model (str | None): Model identifier to use instead of the default model.

        Returns:
            int | None: Token count, or ``None`` when unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """
