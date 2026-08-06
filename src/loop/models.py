"""Define conversation and response models."""

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Describe token usage reported for a response.

    Args:
        input_tokens (int | None): Input tokens consumed when reported.
        output_tokens (int | None): Output tokens generated when reported.
        total_tokens (int | None): Total tokens consumed when reported.
        cached_tokens (int | None): Cached input tokens consumed when reported.
        reasoning_tokens (int | None): Reasoning output tokens generated when reported.
    """

    input_tokens: int | None = Field(default=None, exclude_if=lambda value: value is None)
    output_tokens: int | None = Field(default=None, exclude_if=lambda value: value is None)
    total_tokens: int | None = Field(default=None, exclude_if=lambda value: value is None)
    cached_tokens: int | None = Field(default=None, exclude_if=lambda value: value is None)
    reasoning_tokens: int | None = Field(default=None, exclude_if=lambda value: value is None)


class ResponseMetadata(BaseModel):
    """Describe the provider response that produced a conversation item.

    Usage describes the complete response and is not an item-specific token count.

    Args:
        response_id (str | None): Provider response identifier when reported.
        model (str | None): Model identifier reported for the response.
        usage (Usage | None): Token usage for the complete response.
    """

    response_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    model: str | None = Field(default=None, exclude_if=lambda value: value is None)
    usage: Usage | None = Field(default=None, exclude_if=lambda value: value is None)


class ConversationItemModel(BaseModel):
    """Provide metadata shared by conversation item models.

    Args:
        metadata (ResponseMetadata | None): Metadata for the provider response that produced the
            item, or ``None`` for input and locally produced items.
    """

    metadata: ResponseMetadata | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class Message(ConversationItemModel):
    """Represent a user or assistant conversation message.

    Args:
        role (Literal["user", "assistant"]): Participant that produced the message.
        content (str): Message text.
        metadata (ResponseMetadata | None): Metadata for the provider response that produced the
            message.
    """

    role: Literal["user", "assistant"]
    content: str


class Reasoning(ConversationItemModel):
    """Represent reasoning retained for conversation continuity.

    Args:
        content (str): Completed reasoning text.
        id (str | None): Response item identifier when available.
        metadata (ResponseMetadata | None): Metadata for the provider response that produced the
            reasoning.
    """

    content: str
    id: str | None = None


class ToolCall(ConversationItemModel):
    """Represent a completed function-tool request.

    Args:
        call_id (str): Identifier used to associate the call with its result.
        name (str): Registered tool name.
        arguments (str): JSON-encoded tool arguments.
        id (str | None): Response item identifier when available.
        metadata (ResponseMetadata | None): Metadata for the provider response that produced the
            request.
    """

    call_id: str
    name: str
    arguments: str
    id: str | None = None


class ToolResult(ConversationItemModel):
    """Represent the serialized result of a function-tool request.

    Args:
        call_id (str): Identifier of the corresponding tool call.
        output (str): Serialized tool output.
        metadata (ResponseMetadata | None): Metadata for the provider response that produced the
            result, or ``None`` for a locally produced result.
    """

    call_id: str
    output: str


ConversationItem: TypeAlias = Message | Reasoning | ToolCall | ToolResult


class ToolDefinition(BaseModel):
    """Describe a function tool exposed to a model.

    Args:
        name (str): Public tool name.
        description (str): Description presented to the model.
        parameters (dict[str, Any]): JSON Schema describing accepted arguments.
        strict (bool): Whether the argument schema should be enforced.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True


class ModelInfo(BaseModel):
    """Describe an available language model.

    Args:
        id (str): Model identifier.
        context_window (int | None): Maximum context size when reported.
    """

    id: str
    context_window: int | None = None


class ReasoningDelta(BaseModel):
    """Carry an incremental reasoning fragment.

    Args:
        text (str): Reasoning fragment.
    """

    text: str


class AnswerDelta(BaseModel):
    """Carry an incremental answer fragment.

    Args:
        text (str): Answer fragment.
    """

    text: str


class ReasoningCompleted(BaseModel):
    """Carry the provider's completed reasoning text.

    Args:
        text (str): Completed reasoning text.
    """

    text: str


class AnswerCompleted(BaseModel):
    """Carry the provider's completed answer text.

    Args:
        text (str): Completed answer text.
    """

    text: str


class ToolCallCompleted(BaseModel):
    """Report a completed function-tool request.

    Args:
        call (ToolCall): Completed tool request.
    """

    call: ToolCall


class ResponseCompleted(BaseModel):
    """Report complete response content, history, and metadata.

    Args:
        items (tuple[ConversationItem, ...]): Items to retain in conversation history.
        usage (Usage): Token usage reported for the response.
        model (str | None): Model identifier reported for the response.
        answer (str): Completed answer text.
        reasoning (str): Completed reasoning text.
    """

    items: tuple[ConversationItem, ...] = Field(default_factory=tuple)
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    answer: str = ""
    reasoning: str = ""


ResponseEvent: TypeAlias = (
    ReasoningDelta
    | AnswerDelta
    | ReasoningCompleted
    | AnswerCompleted
    | ToolCallCompleted
    | ResponseCompleted
)


class Response(BaseModel):
    """Collect a completed response from response events.

    Args:
        answer (str): Final answer text.
        reasoning (str): Completed reasoning text.
        tool_calls (tuple[ToolCall, ...]): Function-tool requests made by the model.
        items (tuple[ConversationItem, ...]): Items to retain in conversation history.
        usage (Usage): Token usage reported for the response.
        model (str | None): Model identifier reported for the response.
    """

    answer: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...] = Field(default_factory=tuple)
    items: tuple[ConversationItem, ...] = Field(default_factory=tuple)
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
