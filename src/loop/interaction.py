"""Define user interaction abstractions and tool invocation context."""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pformat
from typing import TYPE_CHECKING, Any, Protocol

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.prompt import Confirm

if TYPE_CHECKING:
    from .skills import SkillManager


class Interaction(Protocol):
    """Provide semantically classified user interaction independently of a UI."""

    def input(self) -> str:
        """Read textual user input without surrounding whitespace.

        Returns:
            The stripped text entered by the user.
        """

    def reasoning(self, message: str) -> None:
        """Display model reasoning output."""

    def reasoning_delta(self, delta: str, *, start: bool = False) -> None:
        """Display a streamed model reasoning delta."""

    def answer(self, message: str) -> None:
        """Display model answer output."""

    def answer_delta(self, delta: str, *, start: bool = False) -> None:
        """Display a streamed model answer delta."""

    def error(self, message: str) -> None:
        """Display an error message."""

    def warning(self, message: str) -> None:
        """Display a warning message."""

    def debug(self, value: Any) -> None:
        """Display diagnostic output."""

    def info(self, message: str = "") -> None:
        """Display neutral status information."""

    def tool_call(self, name: str, arguments: str) -> None:
        """Display a model-requested tool call.

        Args:
            name: Name of the requested tool.
            arguments: JSON arguments supplied to the tool.
        """

    def invalid_input(self) -> None:
        """Display feedback for empty user input."""

    def token_usage(
        self,
        model: str | None,
        context_tokens: int | None,
        context_window: int | None,
    ) -> None:
        """Display the current model and context occupancy."""

    def response_finished(self) -> None:
        """Finish the presentation of a model response."""

    def conversation_ended(self) -> None:
        """Display that the conversation has ended."""

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to approve an operation.

        Args:
            message: Confirmation question to display.
            default: Answer to use when the user enters no response.

        Returns:
            Whether the user approved the operation.
        """


@dataclass(frozen=True)
class ToolContext:
    """Provide runtime services and metadata to a context-aware tool.

    Args:
        interaction: Service used to communicate with the user.
        tool_name: Public name of the tool being invoked.
        skill_manager: Skill manager active for the current conversation.
    """

    interaction: Interaction
    tool_name: str
    skill_manager: SkillManager | None = None

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to confirm an action through the interaction service.

        Args:
            message: Confirmation question to display.
            default: Answer to use when the user enters no response.

        Returns:
            Whether the user approved the action.
        """
        return self.interaction.confirm(message, default=default)


class ConsoleInteraction:
    """Interact with a user through a rich, editable process terminal.

    Args:
        console: Rich console used for terminal output.
        session: Prompt session used for editable user input.
    """

    def __init__(
        self,
        console: Console | None = None,
        session: PromptSession[str] | None = None,
    ) -> None:
        self._console = console or Console()
        self._session = session or PromptSession()

    def input(self) -> str:
        """Read stripped textual input from the terminal.

        Returns:
            The stripped text entered by the user.
        """
        return self._session.prompt("\nYou: ").strip()

    def _reasoning_heading(self) -> None:
        """Write a reasoning heading to the terminal."""
        self._console.print("\nThinking...\n", style="dim cyan", markup=False)

    def reasoning(self, message: str) -> None:
        """Write model reasoning to the terminal."""
        self._reasoning_heading()
        self._console.print(message, style="dim", markup=False, highlight=False)

    def reasoning_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model reasoning delta to the terminal."""
        if start:
            self._reasoning_heading()
        self._console.print(
            delta, end="", style="dim", markup=False, highlight=False, soft_wrap=True
        )

    def _answer_heading(self) -> None:
        """Write an answer heading to the terminal."""
        self._console.print("\nAnswer:", end="", style="bold bright_green", markup=False)

    def answer(self, message: str) -> None:
        """Write a model answer to the terminal."""
        self._answer_heading()
        self._console.print(message, style="bold", markup=False, highlight=False)

    def answer_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model answer delta to the terminal."""
        if start:
            self._answer_heading()
        self._console.print(
            delta, end="", style="bold", markup=False, highlight=False, soft_wrap=True
        )

    def error(self, message: str) -> None:
        """Write an error to the terminal."""
        self._console.print(f"Error: {message}", style="bold red", markup=False)

    def warning(self, message: str) -> None:
        """Write a warning to the terminal."""
        self._console.print(f"Warning: {message}", style="bold yellow", markup=False)

    def debug(self, value: Any) -> None:
        """Write diagnostic output to the terminal."""
        self._console.print(f"\n[DEBUG EVENT]: {type(value)}", style="dim blue", markup=False)
        self._console.print(pformat(value), style="dim", markup=False, highlight=False)

    def info(self, message: str = "") -> None:
        """Write neutral status information to the terminal."""
        self._console.print(message, markup=False, highlight=False)

    def tool_call(self, name: str, arguments: str) -> None:
        """Write a model-requested tool call to the terminal.

        Args:
            name: Name of the requested tool.
            arguments: JSON arguments supplied to the tool.
        """
        self._console.print(
            f"\n[TOOL CALL]: {name}({arguments})", style="dim magenta", markup=False
        )

    def invalid_input(self) -> None:
        """Ask the terminal user to enter a non-empty message."""
        self.warning("Please enter a message!")

    def token_usage(
        self,
        model: str | None,
        context_tokens: int | None,
        context_window: int | None,
    ) -> None:
        """Write the current model and context occupancy."""
        current_model = model or "?"
        used = f"{context_tokens:,}" if context_tokens is not None else "?"
        capacity = f"{context_window:,}" if context_window is not None else "?"
        self._console.print(
            f"Model: {current_model} · Context: {used} / {capacity} tokens",
            style="dim cyan",
            markup=False,
            soft_wrap=True,
        )

    def response_finished(self) -> None:
        """Terminate streamed terminal output with a newline."""
        self.info()

    def conversation_ended(self) -> None:
        """Report conversation termination in the terminal."""
        self.info("\nConversation ended.")

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask for a yes-or-no answer and apply an empty-answer default.

        Args:
            message: Confirmation question to display.
            default: Answer to use when the user enters no response.

        Returns:
            Whether the user approved the operation.
        """
        return Confirm.ask(message, default=default, console=self._console)
