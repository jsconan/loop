"""Define user interaction abstractions and tool invocation context."""

from collections.abc import Generator
from contextlib import contextmanager
from pprint import pformat
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.prompt import Confirm

from ..commands import Command
from ..utils import format_tool_call_arguments
from .interaction import Interaction


class ConsoleInteraction(Interaction):
    """Interact with a user through a rich, editable process terminal.

    Args:
        console (Console | None): Rich console used for terminal output. Defaults to a new console.
        session (PromptSession[str] | None): Prompt session used for editable user input. Defaults
            to a new session.
    """

    def __init__(
        self,
        console: Console | None = None,
        session: PromptSession[str] | None = None,
    ) -> None:
        self._console = console or Console()
        self._session = session or PromptSession()
        self._streamed_output = False

    @contextmanager
    def response(self) -> Generator[None]:
        """Present one model response and finalize streamed output.

        Yields:
            None: Control while the response is being presented.
        """
        self._streamed_output = False
        try:
            yield
        finally:
            if self._streamed_output:
                self.info()
            self._streamed_output = False

    def input(
        self,
        commands: tuple[Command, ...] = (),
        message: str | None = None,
    ) -> str | False:
        """Prompt for a non-empty user message or an exit command.

        Args:
            commands (tuple[Command, ...]): Commands available for input completion.
            message (str | None): Prompt message displayed before reading input.
                Defaults to ``None`` for the default prompt.

        Returns:
            str | False: The entered message, or ``False`` when the user requests to exit.
        """
        if message is None:
            message = "\nYou: "
        completer = WordCompleter(
            [f"/{command.name}" for command in commands],
            meta_dict={f"/{command.name}": command.description for command in commands},
            sentence=True,
        )
        while True:
            try:
                user_input = self._session.prompt(message, completer=completer).strip()
            except KeyboardInterrupt, EOFError:
                return False
            if not user_input:
                self.warning("Please enter a message!")
                continue
            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                return False
            return user_input

    def _reasoning_heading(self) -> None:
        """Write a reasoning heading to the terminal."""
        self._console.print("\nThinking...\n", style="dim cyan", markup=False)

    def reasoning(self, message: str) -> None:
        """Write model reasoning to the terminal.

        Args:
            message (str): Complete reasoning text to write.
        """
        self._reasoning_heading()
        self._console.print(message, style="dim", markup=False, highlight=False)

    def reasoning_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model reasoning delta to the terminal.

        Args:
            delta (str): Incremental reasoning text to write.
            start (bool): Whether to write the reasoning heading before the delta.
        """
        if start:
            self._reasoning_heading()
        self._streamed_output = True
        self._console.print(
            delta, end="", style="dim", markup=False, highlight=False, soft_wrap=True
        )

    def _answer_heading(self) -> None:
        """Write an answer heading to the terminal."""
        self._console.print("\nAnswer:", end="", style="bold bright_green", markup=False)

    def answer(self, message: str) -> None:
        """Write a model answer to the terminal.

        Args:
            message (str): Complete answer text to write.
        """
        self._answer_heading()
        self._console.print(message, style="bold", markup=False, highlight=False)

    def answer_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model answer delta to the terminal.

        Args:
            delta (str): Incremental answer text to write.
            start (bool): Whether to write the answer heading before the delta.
        """
        if start:
            self._answer_heading()
        self._streamed_output = True
        self._console.print(
            delta, end="", style="bold", markup=False, highlight=False, soft_wrap=True
        )

    def error(self, message: str) -> None:
        """Write an error to the terminal.

        Args:
            message (str): Error text to write.
        """
        self._console.print(f"Error: {message}", style="bold red", markup=False)

    def warning(self, message: str) -> None:
        """Write a warning to the terminal.

        Args:
            message (str): Warning text to write.
        """
        self._console.print(f"Warning: {message}", style="bold yellow", markup=False)

    def debug(self, value: Any) -> None:
        """Write diagnostic output to the terminal.

        Args:
            value (Any): Diagnostic value to write.
        """
        self._console.print(f"\n[DEBUG EVENT]: {type(value)}", style="dim blue", markup=False)
        self._console.print(pformat(value), style="dim", markup=False, highlight=False)

    def info(self, message: str = "") -> None:
        """Write neutral status information to the terminal.

        Args:
            message (str): Status text to write, or an empty string for a blank line.
        """
        self._console.print(message, markup=False, highlight=False)

    def tool_call(self, name: str, arguments: str) -> None:
        """Write a model-requested tool call to the terminal.

        Args:
            name (str): Name of the requested tool.
            arguments (str): JSON arguments supplied to the tool.
        """
        self._console.print(
            f"\n[TOOL CALL]: {name}({format_tool_call_arguments(arguments)})",
            style="dim magenta",
            markup=False,
        )

    def token_usage(
        self,
        model: str | None,
        context_tokens: int | None,
        context_window: int | None,
    ) -> None:
        """Write the current model and context occupancy.

        Args:
            model (str | None): Current model identifier, when known.
            context_tokens (int | None): Number of tokens currently in the context, when known.
            context_window (int | None): Maximum context size in tokens, when known.
        """
        current_model = model or "?"
        used = f"{context_tokens:,}" if context_tokens is not None else "?"
        capacity = f"{context_window:,}" if context_window is not None else "?"
        self._console.print(
            f"Model: {current_model} · Context: {used} / {capacity} tokens",
            style="dim cyan",
            markup=False,
            soft_wrap=True,
        )

    def conversation_ended(self) -> None:
        """Report conversation termination in the terminal."""
        self.info("\nConversation ended.")

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask for a yes-or-no answer and apply an empty-answer default.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the operation.
        """
        return Confirm.ask(message, default=default, console=self._console)
