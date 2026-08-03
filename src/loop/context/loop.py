"""Define conversation context."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..models import ConversationItem, Message, Reasoning, ToolCall, ToolResult

_SCHEMA_VERSION = 1
_ITEM_TYPES = {
    "message": Message,
    "reasoning": Reasoning,
    "tool_call": ToolCall,
    "tool_result": ToolResult,
}
_TYPE_NAMES = {model: name for name, model in _ITEM_TYPES.items()}


class UnsupportedConversationItemError(ValueError):
    """Report an unsupported conversation item type in a serialized context."""


@dataclass
class LoopContext:
    """Store conversation history and the latest model context usage.

    Args:
        messages (list[ConversationItem]): Initial conversation items. Defaults to an empty list.
        tokens (int): Initial total tokens in the context after the latest response. Defaults to
            zero.
        model (str | None): Initial model identifier reported by the latest response, or ``None``
            when unknown.
    """

    messages: list[ConversationItem] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None

    def add_message(self, message: ConversationItem) -> None:
        """Add one message to the conversation history.

        Args:
            message (ConversationItem): Conversation item to add.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        self.messages.append(self._get_message(message))

    def add_messages(self, messages: Iterable[ConversationItem]) -> None:
        """Add messages to the conversation history.

        Args:
            messages (Iterable[ConversationItem]): Conversation items to add.

        Raises:
            ValueError: If any value is not a supported conversation item.
        """
        self.messages.extend([self._get_message(message) for message in messages])

    @staticmethod
    def _get_message(message: ConversationItem) -> ConversationItem:
        """Return a validated conversation item for storage."""
        if not isinstance(message, ConversationItem):
            raise ValueError(f"Expected a conversation item, got {type(message)}")
        return message

    def serialize(self) -> str:
        """Serialize the complete context into its versioned JSON format.

        Returns:
            str: Compact JSON representation of the context.

        Raises:
            UnsupportedConversationItemError: If a conversation item type is not supported.
        """
        messages = []
        for message in self.messages:
            item_type = type(message)
            if item_type not in _TYPE_NAMES:
                raise UnsupportedConversationItemError(
                    f"Unsupported conversation item type: {item_type.__name__}."
                )
            messages.append(
                {
                    "type": _TYPE_NAMES[item_type],
                    "data": message.model_dump(mode="json"),
                }
            )

        return json.dumps(
            {
                "version": _SCHEMA_VERSION,
                "messages": messages,
                "tokens": self.tokens,
                "model": self.model,
            },
            separators=(",", ":"),
        )

    @classmethod
    def deserialize(cls, value: str) -> "LoopContext":
        """Deserialize and validate a context from its versioned JSON format.

        Args:
            value (str): JSON representation of a complete context.

        Returns:
            LoopContext: Reconstructed context state.

        Raises:
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If the serialized context is invalid or uses an unsupported version.
        """
        try:
            payload = json.loads(value)
            version = payload["version"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid serialized loop context.") from error

        if version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop context version {version}.")

        try:
            messages = []
            for item in payload["messages"]:
                item_type = item["type"]
                if item_type not in _ITEM_TYPES:
                    raise UnsupportedConversationItemError(
                        f"Unsupported conversation item type: {item_type!r}."
                    )
                model = _ITEM_TYPES[item_type]
                messages.append(model.model_validate(item["data"]))

            tokens = payload["tokens"]
            model = payload["model"]
        except (KeyError, TypeError, ValidationError) as error:
            raise ValueError("Invalid serialized loop context.") from error

        if (
            not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or (model is not None and not isinstance(model, str))
        ):
            raise ValueError("Invalid serialized loop context.")

        return cls(messages=messages, tokens=tokens, model=model)
