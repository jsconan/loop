"""Define user interaction abstractions."""

from typing import Any, Protocol


class Interaction(Protocol):
    """Provide semantically classified user interaction independently of a UI."""

    def input(self) -> str | False:
        """Read a non-empty user message or an exit command.

        Returns:
            str | False: The stripped text entered by the user, or ``False`` when the user
            requests to exit.
        """

    def reasoning(self, message: str) -> None:
        """Display model reasoning output.

        Args:
            message (str): Complete reasoning text to display.
        """

    def reasoning_delta(self, delta: str, *, start: bool = False) -> None:
        """Display a streamed model reasoning delta.

        Args:
            delta (str): Incremental reasoning text to display.
            start (bool): Whether this is the first reasoning delta in the response.
        """

    def answer(self, message: str) -> None:
        """Display model answer output.

        Args:
            message (str): Complete answer text to display.
        """

    def answer_delta(self, delta: str, *, start: bool = False) -> None:
        """Display a streamed model answer delta.

        Args:
            delta (str): Incremental answer text to display.
            start (bool): Whether this is the first answer delta in the response.
        """

    def error(self, message: str) -> None:
        """Display an error message.

        Args:
            message (str): Error text to display.
        """

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message (str): Warning text to display.
        """

    def debug(self, value: Any) -> None:
        """Display diagnostic output.

        Args:
            value (Any): Diagnostic value to display.
        """

    def info(self, message: str = "") -> None:
        """Display neutral status information.

        Args:
            message (str): Status text to display, or an empty string for a blank line.
        """

    def tool_call(self, name: str, arguments: str) -> None:
        """Display a model-requested tool call.

        Args:
            name (str): Name of the requested tool.
            arguments (str): JSON arguments supplied to the tool.
        """

    def token_usage(
        self,
        model: str | None,
        context_tokens: int | None,
        context_window: int | None,
    ) -> None:
        """Display the current model and context occupancy.

        Args:
            model (str | None): Current model identifier, when known.
            context_tokens (int | None): Number of tokens currently in the context, when known.
            context_window (int | None): Maximum context size in tokens, when known.
        """

    def response_finished(self) -> None:
        """Finish the presentation of a model response."""

    def conversation_ended(self) -> None:
        """Display that the conversation has ended."""

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to approve an operation.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the operation.
        """
