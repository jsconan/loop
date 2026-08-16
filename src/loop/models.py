"""Define conversation and response models."""

import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from jsonschema import FormatChecker, SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
StructuredOutputValidator: TypeAlias = Callable[[JsonValue], Any]

_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", re.IGNORECASE | re.DOTALL)


class StructuredOutputValidationError(ValueError):
    """Report output that does not satisfy its requested structure.

    Args:
        format_name (str): Name of the structured output contract.
        raw_output (str): Complete rejected provider output.
        errors (tuple[str, ...]): Machine-readable-path validation diagnostics.
        attempt (int | None): One-based generation attempt, when assigned by a backend.
        model (str | None): Provider model that generated the rejected output.
        mode (str | None): Structured-output transport used for the failed attempt.
        usage (Usage | None): Aggregate usage across failed attempts, when available.
        category (str): Failure category, such as validation, refusal, or incomplete.
    """

    def __init__(
        self,
        format_name: str,
        raw_output: str,
        errors: tuple[str, ...],
        *,
        attempt: int | None = None,
        model: str | None = None,
        mode: str | None = None,
        usage: "Usage | None" = None,
        category: str = "validation",
    ) -> None:
        self.format_name = format_name
        self.raw_output = raw_output
        self.errors = errors
        self.attempt = attempt
        self.model = model
        self.mode = mode
        self.usage = usage
        self.category = category
        details = "; ".join(errors) if errors else "unknown validation error"
        super().__init__(
            f"Response does not satisfy structured output format {format_name!r}: {details}"
        )


@dataclass(frozen=True, slots=True)
class StructuredOutputFormat:
    """Describe and validate a JSON Schema-constrained model response.

    Args:
        name (str): Portable identifier for the output schema.
        schema (Mapping[str, object]): Provider-facing JSON Schema, also used for local validation
            when no separate canonical schema is supplied.
        description (str | None): Optional description of the desired output.
        strict (bool): Whether the provider should enforce strict schema adherence.
        model (type[BaseModel] | None): Pydantic model retained for output validation.
        validator (StructuredOutputValidator | None): Callback that receives decoded JSON and
            returns its validated or transformed value when no Pydantic model is supplied.
        validation_schema (Mapping[str, object] | None): Canonical schema used for local
            validation when ``schema`` is a provider-specific strict representation.

    Raises:
        ValueError: If the name or schema is invalid, or both validation mechanisms are supplied.
    """

    name: str
    schema: Mapping[str, object]
    description: str | None = None
    strict: bool = True
    model: type[BaseModel] | None = None
    validator: StructuredOutputValidator | None = None
    validation_schema: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Structured output format name must not be empty.")
        if self.model is not None and self.validator is not None:
            raise ValueError("Structured output format cannot define both a model and validator.")
        object.__setattr__(self, "schema", deepcopy(dict(self.schema)))
        if self.validation_schema is not None:
            object.__setattr__(self, "validation_schema", deepcopy(dict(self.validation_schema)))
        canonical_schema = dict(self.validation_schema or self.schema)
        try:
            schema_validator = validator_for(canonical_schema)
            schema_validator.check_schema(canonical_schema)
        except SchemaError as error:
            raise ValueError(f"Invalid structured output JSON Schema: {error.message}") from error

    @staticmethod
    def _strict_json_schema(schema: dict[str, object]) -> dict[str, object]:
        """Return a copy normalized to the strict structured-output schema subset."""
        root = deepcopy(schema)

        def resolve(reference: str) -> dict[str, object]:
            value = root
            for part in reference[2:].split("/"):
                value = value[part.replace("~1", "/").replace("~0", "~")]
            return value

        def normalize(value: dict[str, object]) -> dict[str, object]:
            reference = value.get("$ref")
            if isinstance(reference, str) and len(value) > 1:
                value = {**deepcopy(resolve(reference)), **value}
                value.pop("$ref")
            definitions = value.get("$defs")
            if isinstance(definitions, dict):
                value["$defs"] = {
                    key: normalize(definition) for key, definition in definitions.items()
                }
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)
                value["properties"] = {
                    key: normalize(property_schema) for key, property_schema in properties.items()
                }
            if value.get("type") == "object" and "additionalProperties" not in value:
                value["additionalProperties"] = False
            items = value.get("items")
            if isinstance(items, dict):
                value["items"] = normalize(items)
            any_of = value.get("anyOf")
            if isinstance(any_of, list):
                value["anyOf"] = [normalize(variant) for variant in any_of]
            if value.get("default", ...) is None:
                value.pop("default")
            return value

        return normalize(root)

    @classmethod
    def from_model(
        cls,
        model: type[BaseModel],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = True,
    ) -> "StructuredOutputFormat":
        """Create a structured output format from a Pydantic model.

        Args:
            model (type[BaseModel]): Model used to create the schema and validate output.
            name (str | None): Schema identifier, defaulting to the model class name.
            description (str | None): Optional description of the desired output.
            strict (bool): Whether the provider should enforce strict schema adherence.

        Returns:
            StructuredOutputFormat: Format retaining the model type for validation.
        """
        schema = model.model_json_schema()
        return cls(
            name=name or model.__name__,
            schema=(cls._strict_json_schema(schema) if strict else schema),
            description=description,
            strict=strict,
            model=model,
            validation_schema=schema,
        )

    @staticmethod
    def _normalize_structured_text(text: str) -> str:
        """Return the text stripped of whitespace and fenced code blocks."""
        stripped = text.strip()
        fenced = _JSON_FENCE.fullmatch(stripped)
        return fenced.group(1) if fenced is not None else stripped

    def _single_string_field(self) -> str | None:
        """Return the single string field name when the schema is a single string property."""
        properties = self.schema.get("properties")
        if self.model is not None and isinstance(properties, Mapping) and len(properties) == 1:
            field_name, field_schema = next(iter(properties.items()))
            if isinstance(field_schema, Mapping) and field_schema.get("type") == "string":
                return field_name
        return None

    def _decode_structured_text(self, text: str) -> JsonValue:
        """Decode JSON from the text."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            if (string_field := self._single_string_field()) is not None:
                return {string_field: text}
            raise
        if (string_field := self._single_string_field()) is not None and isinstance(value, str):
            return {string_field: value}
        return value

    def validate(self, text: str) -> Any:
        """Decode and locally validate completed JSON response text.

        Args:
            text (str): Complete raw JSON or JSON-fenced response text.

        Returns:
            Any: Pydantic model, callback result, or decoded JSON value.

        Raises:
            StructuredOutputValidationError: If JSON decoding or configured validation fails.
        """
        try:
            payload = self._normalize_structured_text(text)
            value = self._decode_structured_text(payload)
            canonical_schema = dict(self.validation_schema or self.schema)
            schema_validator = validator_for(canonical_schema)(
                canonical_schema, format_checker=FormatChecker()
            )
            schema_errors = sorted(
                schema_validator.iter_errors(value), key=lambda error: list(error.absolute_path)
            )
            if schema_errors:
                errors = tuple(
                    "$"
                    + "".join(
                        f"[{part}]" if isinstance(part, int) else f".{part}"
                        for part in schema_error.absolute_path
                    )
                    + f": {schema_error.message}"
                    for schema_error in schema_errors
                )
                raise StructuredOutputValidationError(self.name, text, errors)
            if self.model is not None:
                return self.model.model_validate(value)
            if self.validator is not None:
                return self.validator(value)
            return value
        except StructuredOutputValidationError:
            raise
        except (
            json.JSONDecodeError,
            PydanticValidationError,
            ValueError,
            TypeError,
        ) as error:
            if isinstance(error, PydanticValidationError):
                errors = tuple(
                    "$"
                    + "".join(
                        f"[{part}]" if isinstance(part, int) else f".{part}"
                        for part in detail["loc"]
                    )
                    + f": {detail['msg']}"
                    for detail in error.errors(include_url=False)
                )
            else:
                errors = (str(error),)
            raise StructuredOutputValidationError(self.name, text, errors) from error


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


class ContextReference(BaseModel):
    """Describe one resolved, bounded user context snapshot.

    Args:
        kind (Literal["file", "directory"]): Referenced filesystem object kind.
        path (str): User-facing project-relative path.
        content (str): Bounded content captured when the turn was submitted.
        size_bytes (int): Complete source size in bytes.
        included_bytes (int): Number of content bytes included in the snapshot.
        truncated (bool): Whether content was omitted from the snapshot.
        handle (str | None): Opaque handle for reading an omitted immutable remainder.
        next_cursor (str | None): Opaque cursor at which the first continuation starts.
        snapshot_content (str | None): Complete immutable content retained for cache restoration;
            never sent to the model provider.
    """

    kind: Literal["file", "directory"]
    path: str
    content: str
    size_bytes: int
    included_bytes: int
    truncated: bool
    handle: str | None = None
    next_cursor: str | None = None
    snapshot_content: str | None = None


class Message(ConversationItemModel):
    """Represent a user or assistant conversation message.

    Args:
        role (Literal["user", "assistant"]): Participant that produced the message.
        content (str): Message text.
        context (tuple[ContextReference, ...]): Explicit resolved context snapshots.
        metadata (ResponseMetadata | None): Metadata for the provider response that produced the
            message.
    """

    role: Literal["user", "assistant"]
    content: str
    context: tuple[ContextReference, ...] = Field(
        default_factory=tuple,
        exclude_if=lambda value: not value,
    )


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


class ContentArtifact(BaseModel):
    """Describe session-persisted metadata for one out-of-context artifact.

    Args:
        handle (str): Opaque canonical artifact handle.
        source (str): Human-readable source used to reload reproducible content.
        reloadable (bool): Whether the source can recreate an expired artifact.
    """

    handle: str
    source: str
    reloadable: bool


class ToolResult(ConversationItemModel):
    """Represent the serialized result of a function-tool request.

    Args:
        call_id (str): Identifier of the corresponding tool call.
        output (str): Serialized tool output.
        artifacts (tuple[ContentArtifact, ...]): Local persisted artifact metadata. Defaults to an
            empty tuple and is not sent to the model provider.
        metadata (ResponseMetadata | None): Metadata for the provider response that produced the
            result, or ``None`` for a locally produced result.
    """

    call_id: str
    output: str
    artifacts: tuple[ContentArtifact, ...] = Field(default_factory=tuple)


ConversationItem: TypeAlias = Message | Reasoning | ToolCall | ToolResult


class CompactionContextItem(BaseModel):
    """Preserve one provider-specific item used to resume compacted context.

    Args:
        provider (str): Backend namespace that can interpret the item.
        data (dict[str, Any]): Serialized provider input item.
    """

    provider: str
    data: dict[str, Any]


ModelContextItem: TypeAlias = ConversationItem | CompactionContextItem


class CompactionResult(BaseModel):
    """Describe replacement context produced by a backend compactor.

    Args:
        items (tuple[CompactionContextItem, ...]): Provider items replacing earlier context.
        usage (Usage): Token usage reported for the compaction operation.
        context_tokens (int | None): Tokens in the replacement context, excluding the source
            context consumed by the compaction request.
    """

    items: tuple[CompactionContextItem, ...]
    usage: Usage = Field(default_factory=Usage)
    context_tokens: int | None = Field(default=None, ge=0)


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
        structured_output (Any | None): Validated structured answer, when requested.
    """

    items: tuple[ConversationItem, ...] = Field(default_factory=tuple)
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    answer: str = ""
    reasoning: str = ""
    structured_output: Any | None = Field(default=None, exclude_if=lambda value: value is None)


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
        structured_output (Any | None): Validated structured answer, when requested.
    """

    answer: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...] = Field(default_factory=tuple)
    items: tuple[ConversationItem, ...] = Field(default_factory=tuple)
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    structured_output: Any | None = Field(default=None, exclude_if=lambda value: value is None)
