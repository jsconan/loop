"""Define tool invocation context."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from openai import BaseModel


@dataclass
class LoopContext:
    """Store conversation history and the latest model context usage.

    Args:
        messages (list[dict]): Initial Responses API input items. Defaults to an empty list.
        tokens (int): Initial total tokens in the context after the latest response. Defaults to
            zero.
        model (str | None): Initial model identifier reported by the latest response, or ``None``
            when unknown.
    """

    messages: list[dict] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None

    def add_message(self, message: BaseModel | dict) -> None:
        """Add one message to the conversation history.

        Args:
            message (BaseModel | dict): Responses API input item to add. Models are dumped to
                dictionaries.

        Raises:
            ValueError: If the message is not a dictionary or a BaseModel.
        """
        self.messages.append(self._get_message(message))

    def add_messages(self, messages: Iterable[BaseModel | dict]) -> None:
        """Add messages to the conversation history.

        Args:
            messages (Iterable[BaseModel | dict]): Responses API input items to add. Models are
                dumped to dictionaries.

        Raises:
            ValueError: If any message is not a dictionary or a BaseModel.
        """
        self.messages.extend([self._get_message(message) for message in messages])

    def _get_message(self, message: dict | BaseModel) -> dict:
        """Convert a message to a dictionary for storage in the conversation history."""
        if isinstance(message, BaseModel):
            message = message.model_dump(exclude_none=True)
        if not isinstance(message, dict):
            raise ValueError(f"Expected message to be a dict or BaseModel, got {type(message)}")
        return message
