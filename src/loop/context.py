"""Define tool invocation context."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openai import BaseModel

from .types import Interaction

if TYPE_CHECKING:
    from .skills import SkillManager


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


@dataclass(frozen=True)
class ToolContext:
    """Provide runtime services and metadata to a context-aware tool.

    Args:
        interaction (Interaction): Service used to communicate with the user.
        tool_name (str): Public name of the tool being invoked.
        skill_manager (SkillManager | None): Skill manager active for the current conversation,
            or ``None`` when
            skills are unavailable.
    """

    interaction: Interaction
    tool_name: str
    skill_manager: SkillManager | None = None

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to confirm an action through the interaction service.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the action.
        """
        return self.interaction.confirm(message, default=default)
