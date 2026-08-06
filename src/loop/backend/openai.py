"""Adapt OpenAI-compatible APIs to conversation response events."""

from collections.abc import AsyncIterator, Iterable, Iterator

import httpx
from openai import APIError, AsyncOpenAI, OpenAI
from openai.types.model import Model as OpenAIModel
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseCompletedEvent as OpenAIResponseCompletedEvent
from openai.types.responses import ResponseFunctionToolCall as OpenAIFunctionToolCall
from openai.types.responses import ResponseOutputItem as OpenAIResponseOutputItem
from openai.types.responses import ResponseOutputItemDoneEvent as OpenAIOutputItemDoneEvent
from openai.types.responses import ResponseOutputMessage as OpenAIOutputMessage
from openai.types.responses import ResponseReasoningItem as OpenAIReasoningItem
from openai.types.responses import ResponseReasoningTextDeltaEvent as OpenAIReasoningDeltaEvent
from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent
from openai.types.responses import ResponseTextDeltaEvent as OpenAITextDeltaEvent

from ..models import (
    AnswerCompleted,
    AnswerDelta,
    ConversationItem,
    Message,
    ModelInfo,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    ResponseCompleted,
    ResponseEvent,
    ResponseMetadata,
    ToolCall,
    ToolCallCompleted,
    ToolDefinition,
    ToolResult,
    Usage,
)
from ..tooling import ToolRegistry
from ..tooling import tool_registry as default_tool_registry
from .base import Backend


class OpenAIBackend(Backend):
    """Adapt an OpenAI-compatible API to conversation models and events.

    Args:
        default_model (str | None): Model identifier used when a request does not specify one.
        base_url (str | None): Base URL of the OpenAI-compatible backend.
        api_key (str | None): API key used privately by the backend client.
        tool_registry (ToolRegistry | None): Registry supplying tool schemas for requests.
            Defaults to the package
            registry.
        context_window (int | None): Deployed model context limit, or ``None`` to use best-effort
            model metadata discovery.

    Raises:
        ValueError: If the configured context window is not an integer or is not positive.
    """

    _client: OpenAI | None
    _async_client: AsyncOpenAI | None
    _configured_context_window: int | None
    _context_windows: dict[str, int | None]

    def __init__(
        self,
        *,
        default_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tool_registry: ToolRegistry | None = None,
        context_window: int | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            default_model=default_model,
            api_key=api_key,
            tool_registry=tool_registry or default_tool_registry,
        )
        self._client = None
        self._async_client = None
        self._configured_context_window = context_window
        if self._configured_context_window is not None and self._configured_context_window <= 0:
            raise ValueError("Context window must be a positive integer.")
        self._context_windows = {}

    @property
    def context_window(self) -> int | None:
        """Return the default model context limit when available.

        Returns:
            int | None: The context limit, or ``None`` when it cannot be determined.
        """
        return self.get_context_window()

    def _get_client(self) -> OpenAI:
        """Return the lazily initialized synchronous OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def _get_async_client(self) -> AsyncOpenAI:
        """Return the lazily initialized asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._async_client

    def get_models(self) -> list[ModelInfo]:
        """Return the models available from the configured backend.

        Returns:
            list[ModelInfo]: The available models.
        """
        models = self._get_client().models.list(timeout=2.0)
        return [self._model_info(model) for model in models]

    async def get_models_async(self) -> list[ModelInfo]:
        """Asynchronously return the models available from the configured backend.

        Returns:
            list[ModelInfo]: The available models.
        """
        models = await self._get_async_client().models.list(timeout=2.0)
        return [self._model_info(model) for model in models]

    def get_response(
        self,
        input: str | Iterable[ConversationItem],  # pylint: disable=redefined-builtin
        instructions: str | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> Iterator[ResponseEvent]:
        """Yield normalized events from a synchronous response.

        Args:
            input (str | Iterable[ConversationItem]): Text or conversation history to send.
            instructions (str | None): System or developer instructions to apply to the request.
            stream (bool): Whether to return a streaming response.
            model (str | None): Model identifier to use instead of the default model.

        Yields:
            ResponseEvent: Response events in output order.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """
        selected_model = self._select_model(model)
        response = self._get_client().responses.create(
            model=selected_model,
            input=self._serialize_input(input),
            instructions=instructions,
            stream=stream,
            stream_options={"include_usage": True},
            tools=self._serialize_tools(self._tool_registry.definitions()),
        )
        if stream:
            items = []
            for event in response:
                yield from self._translated_stream_event(event, items)
            return
        yield from self._response_events(response)

    async def get_response_async(
        self,
        input: str | Iterable[ConversationItem],  # pylint: disable=redefined-builtin
        instructions: str | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> AsyncIterator[ResponseEvent]:
        """Yield events from an asynchronous response.

        Args:
            input (str | Iterable[ConversationItem]): Text or conversation history to send.
            instructions (str | None): System or developer instructions to apply to the request.
            stream (bool): Whether to return a streaming response.
            model (str | None): Model identifier to use instead of the default model.

        Yields:
            ResponseEvent: Response events in output order.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """
        selected_model = self._select_model(model)
        response = await self._get_async_client().responses.create(
            model=selected_model,
            input=self._serialize_input(input),
            instructions=instructions,
            stream=stream,
            stream_options={"include_usage": True},
            tools=self._serialize_tools(self._tool_registry.definitions()),
        )
        if not stream:
            for event in self._response_events(response):
                yield event
            return
        items = []
        async for event in response:
            for translated in self._translated_stream_event(event, items):
                yield translated

    def get_context_window(self, model: str | None = None) -> int | None:
        """Return the deployed context limit for a selected model when available.

        Args:
            model (str | None): Model identifier to inspect instead of the default model.

        Returns:
            int | None: The configured or discovered context limit, or ``None`` when unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """
        if self._configured_context_window is not None:
            return self._configured_context_window
        selected_model = self._select_model(model)
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
            model (str | None): Model identifier to inspect instead of the default model.

        Returns:
            int | None: The configured or discovered context limit, or ``None`` when unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """
        if self._configured_context_window is not None:
            return self._configured_context_window
        selected_model = self._select_model(model)
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
            prompt (str): Text to tokenize.
            model (str | None): Model identifier to use instead of the default model.

        Returns:
            int | None: The token count, or ``None`` when tokenization fails or is unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """
        selected_model = self._select_model(model)
        if self._base_url is None:
            return None
        base_url = self._base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        try:
            response = httpx.post(
                f"{base_url}/tokenize",
                json={
                    "model": selected_model,
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
            prompt (str): Text to tokenize.
            model (str | None): Model identifier to use instead of the default model.

        Returns:
            int | None: The token count, or ``None`` when tokenization fails or is unavailable.

        Raises:
            ValueError: If neither the request nor backend selects a model.
        """
        selected_model = self._select_model(model)
        if self._base_url is None:
            return None
        base_url = self._base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/tokenize",
                    json={
                        "model": selected_model,
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
    def _context_window_from_models(models: Iterable[ModelInfo], model_name: str) -> int | None:
        """Extract a model's context limit from a model-list response."""
        for model in models:
            if model.id != model_name:
                continue
            return model.context_window
        return None

    @staticmethod
    def _model_info(model: OpenAIModel) -> ModelInfo:
        """Translate OpenAI model metadata into a model description."""
        context_window = (model.model_extra or {}).get("max_model_len")
        try:
            context_window = int(context_window) if context_window is not None else None
        except TypeError, ValueError:
            context_window = None
        return ModelInfo(id=model.id, context_window=context_window)

    @classmethod
    def _serialize_input(cls, input: str | Iterable[ConversationItem]):  # pylint: disable=redefined-builtin
        """Translate conversation items into OpenAI-compatible request items."""
        if isinstance(input, str):
            return input
        return [cls._serialize_item(item) for item in input]

    @staticmethod
    def _serialize_item(item: ConversationItem) -> dict:
        """Translate one conversation item into an OpenAI-compatible request item."""
        if isinstance(item, Message):
            return {"role": item.role, "content": item.content}
        if isinstance(item, Reasoning):
            content = [{"type": "reasoning_text", "text": item.content}] if item.content else []
            result = {"type": "reasoning", "summary": [], "content": content}
            if item.id is not None:
                result["id"] = item.id
            return result
        if isinstance(item, ToolCall):
            result = {
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
                "status": "completed",
            }
            if item.id is not None:
                result["id"] = item.id
            return result
        if isinstance(item, ToolResult):
            return {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": item.output,
            }
        raise TypeError(f"Unsupported conversation item: {type(item)}")

    @staticmethod
    def _serialize_tools(definitions: Iterable[ToolDefinition]) -> list[dict]:
        """Translate tool definitions into OpenAI-compatible declarations."""
        return [
            {
                "type": "function",
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters,
                "strict": definition.strict,
            }
            for definition in definitions
        ]

    @classmethod
    def _response_events(cls, response: OpenAIResponse) -> Iterator[ResponseEvent]:
        """Translate a completed OpenAI response into normalized events."""
        items = [
            translated
            for item in response.output
            if (translated := cls._translate_item(item)) is not None
        ]
        final_reasoning = next(
            (item for item in reversed(items) if isinstance(item, Reasoning) and item.content),
            None,
        )
        for translated in items:
            if isinstance(translated, Reasoning) and translated.content:
                if translated is final_reasoning:
                    yield ReasoningCompleted(text=translated.content)
            elif isinstance(translated, ToolCall):
                yield ToolCallCompleted(call=translated)
        answer = response.output_text
        if isinstance(answer, str) and answer:
            yield AnswerCompleted(text=answer)
        yield cls._completion(response, items)

    @classmethod
    def _translated_stream_event(
        cls,
        event: OpenAIResponseStreamEvent,
        items: list[ConversationItem],
    ) -> list[ResponseEvent]:
        """Translate one OpenAI stream event and update completed history items."""
        if isinstance(event, OpenAIReasoningDeltaEvent):
            return [ReasoningDelta(text=event.delta)]
        if isinstance(event, OpenAITextDeltaEvent):
            return [AnswerDelta(text=event.delta)]
        if isinstance(event, OpenAIOutputItemDoneEvent):
            item = cls._translate_item(event.item)
            if item is None:
                return []
            items.append(item)
            return [ToolCallCompleted(call=item)] if isinstance(item, ToolCall) else []
        if isinstance(event, OpenAIResponseCompletedEvent):
            return [cls._completion(event.response, items)]
        return []

    @staticmethod
    def _translate_item(item: OpenAIResponseOutputItem) -> ConversationItem | None:
        """Translate a supported OpenAI output item into a conversation item."""
        if isinstance(item, OpenAIReasoningItem):
            return Reasoning(
                content="".join(content.text for content in item.content or []),
                id=item.id,
            )
        if isinstance(item, OpenAIOutputMessage):
            text = "".join(content.text for content in item.content if hasattr(content, "text"))
            return Message(
                role="assistant",
                content=text,
            )
        if isinstance(item, OpenAIFunctionToolCall):
            return ToolCall(
                id=item.id,
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
            )
        return None

    @classmethod
    def _completion(
        cls,
        response: OpenAIResponse,
        items: Iterable[ConversationItem],
    ) -> ResponseCompleted:
        """Translate terminal OpenAI response content and metadata."""
        usage = cls._usage(response)
        model = response.model
        response_id = getattr(response, "id", None)
        metadata = ResponseMetadata(
            response_id=response_id if isinstance(response_id, str) else None,
            model=model if isinstance(model, str) else None,
            usage=usage if usage.model_dump() else None,
        )
        completed_items = tuple(items)
        if metadata.model_dump():
            for item in completed_items:
                item.metadata = metadata
        answer = response.output_text
        reasoning = next(
            (
                item.content
                for item in reversed(completed_items)
                if isinstance(item, Reasoning) and item.content
            ),
            "",
        )
        return ResponseCompleted(
            items=completed_items,
            usage=usage,
            model=model if isinstance(model, str) else None,
            answer=answer if isinstance(answer, str) else "",
            reasoning=reasoning,
        )

    @staticmethod
    def _usage(response: OpenAIResponse) -> Usage:
        """Translate non-negative token counts from an OpenAI response."""
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)

        def token_count(obj: object, field: str) -> int | None:
            value = getattr(obj, field, None)
            return value if isinstance(value, int) and value >= 0 else None

        return Usage(
            input_tokens=token_count(usage, "input_tokens"),
            output_tokens=token_count(usage, "output_tokens"),
            total_tokens=token_count(usage, "total_tokens"),
            cached_tokens=token_count(input_details, "cached_tokens"),
            reasoning_tokens=token_count(output_details, "reasoning_tokens"),
        )
