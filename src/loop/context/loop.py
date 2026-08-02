"""Define conversation context."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..models import ConversationItem


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
