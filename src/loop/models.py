"""Define conversation and response models."""

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represent a user or assistant conversation message.

    Args:
        role (Literal["user", "assistant"]): Participant that produced the message.
        content (str): Message text.
    """

    role: Literal["user", "assistant"]
    content: str


class Reasoning(BaseModel):
    """Represent reasoning retained for conversation continuity.

    Args:
        content (str): Completed reasoning text.
        id (str | None): Response item identifier when available.
    """

    content: str
    id: str | None = None


class ToolCall(BaseModel):
    """Represent a completed function-tool request.

    Args:
        call_id (str): Identifier used to associate the call with its result.
        name (str): Registered tool name.
        arguments (str): JSON-encoded tool arguments.
        id (str | None): Response item identifier when available.
    """

    call_id: str
    name: str
    arguments: str
    id: str | None = None


class ToolResult(BaseModel):
    """Represent the serialized result of a function-tool request.

    Args:
        call_id (str): Identifier of the corresponding tool call.
        output (str): Serialized tool output.
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


class Usage(BaseModel):
    """Describe token usage reported for a completed response.

    Args:
        total_tokens (int | None): Total tokens in the resulting context when reported.
    """

    total_tokens: int | None = None


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
