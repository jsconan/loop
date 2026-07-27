"""Define user interaction abstractions and tool invocation context."""

from dataclasses import dataclass
from pprint import pformat
from typing import Any, Protocol


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

    def thinking(self) -> None:
        """Display that a model response is pending."""

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
    """

    interaction: Interaction
    tool_name: str

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
    """Interact with a user through the process terminal."""

    def input(self) -> str:
        """Read stripped textual input from the terminal.

        Returns:
            The stripped text entered by the user.
        """
        return input("\nYou: ").strip()

    def reasoning(self, message: str) -> None:
        """Write model reasoning to the terminal."""
        print("\n[THOUGHT PROCESS]:")
        print(message)

    def reasoning_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model reasoning delta to the terminal."""
        if start:
            print("\n[THOUGHT PROCESS]:")
        print(delta, end="", flush=True)

    def answer(self, message: str) -> None:
        """Write a model answer to the terminal."""
        print("\n[ANSWER]:")
        print(message)

    def answer_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model answer delta to the terminal."""
        if start:
            print("\n[ANSWER]:")
        print(delta, end="", flush=True)

    def error(self, message: str) -> None:
        """Write an error to the terminal."""
        print(f"Error: {message}")

    def warning(self, message: str) -> None:
        """Write a warning to the terminal."""
        print(f"Warning: {message}")

    def debug(self, value: Any) -> None:
        """Write diagnostic output to the terminal."""
        print(f"\n[DEBUG EVENT]: {type(value)}")
        print(pformat(value))

    def info(self, message: str = "") -> None:
        """Write neutral status information to the terminal."""
        print(message)

    def tool_call(self, name: str, arguments: str) -> None:
        """Write a model-requested tool call to the terminal.

        Args:
            name: Name of the requested tool.
            arguments: JSON arguments supplied to the tool.
        """
        print(f"\n[TOOL CALL]: {name}({arguments})")

    def invalid_input(self) -> None:
        """Ask the terminal user to enter a non-empty message."""
        self.warning("Please enter a message!")

    def thinking(self) -> None:
        """Indicate in the terminal that a model response is pending."""
        self.info("\nThinking...")

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
        suffix = "[Y/n]" if default else "[y/N]"
        answer = input(f"{message} {suffix}: ").strip().lower()
        if not answer:
            return default
        return answer in {"y", "yes"}
