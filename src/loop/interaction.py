"""Define user interaction abstractions and tool invocation context."""

from dataclasses import dataclass
from typing import Protocol


class Interaction(Protocol):
    """Provide user input, output, and confirmation independently of a UI."""

    def prompt(self, message: str) -> str:
        """Read textual input after displaying a prompt.

        Args:
            message: Prompt to display before reading input.

        Returns:
            The text entered by the user.
        """

    def write(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        """Write user-visible output.

        Args:
            message: Text to write.
            end: Text appended after the message.
            flush: Whether to flush the output destination immediately.
        """

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

    def prompt(self, message: str) -> str:
        """Read textual input from the terminal.

        Args:
            message: Prompt to display before reading input.

        Returns:
            The text entered by the user.
        """
        return input(message)

    def write(self, message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        """Write output to the terminal.

        Args:
            message: Text to write.
            end: Text appended after the message.
            flush: Whether to flush the terminal immediately.
        """
        print(message, end=end, flush=flush)

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask for a yes-or-no answer and apply an empty-answer default.

        Args:
            message: Confirmation question to display.
            default: Answer to use when the user enters no response.

        Returns:
            Whether the user approved the operation.
        """
        suffix = "[Y/n]" if default else "[y/N]"
        answer = self.prompt(f"{message} {suffix}: ").strip().lower()
        if not answer:
            return default
        return answer in {"y", "yes"}
