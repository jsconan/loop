"""Carry acknowledged interaction calls across the conversation boundary."""

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from prompt_toolkit.completion import Completer

EVENT_OUTPUT = "output"
EVENT_PROMPT = "prompt"
EVENT_CONFIRM = "confirm"
EVENT_TERMINAL = "terminal"
EVENT_TERMINAL_EXIT = "terminal_exit"


@dataclass(frozen=True)
class ConversationStatus:
    """Snapshot the user-visible activity and queued input count.

    Args:
        state (str): Current activity shown by the conversation status command.
        queued_inputs (int): Number of ordinary inputs waiting in the mailbox.
    """

    state: Literal["Interrupting", "Waiting for your response", "Idle", "Working"]
    queued_inputs: int


class SteeringOutcome(Enum):
    """Describe whether a steering message was admitted and interrupted the active run."""

    INTERRUPTED = "interrupted"
    RUN_INACTIVE = "run_inactive"
    INTERRUPT_REJECTED = "interrupt_rejected"
    QUEUE_FULL = "queue_full"


@dataclass
class InteractionCall:
    """Carry one UI call whose execution and result belong to the UI thread.

    Args:
        function (Callable[..., Any]): Actual UI operation to invoke.
        args (tuple): Positional arguments preserved from execution.
        kwargs (dict): Keyword arguments preserved from execution.
        kind (str): Output, separate input, or terminal handoff operation. Defaults to output.
        result (Future): Acknowledgement received by the execution thread. Defaults to a fresh
            future for this call.
    """

    function: Callable[..., Any]
    args: tuple
    kwargs: dict
    kind: Literal["output", "prompt", "confirm", "terminal", "terminal_exit"] = EVENT_OUTPUT
    result: Future = field(default_factory=Future)


@dataclass
class InputReady:
    """Announce a safe conversation input boundary and its completion source.

    Args:
        completer (Completer): Live completion source guarded by execution ownership.
        busy (bool): Whether execution has consumed input and is starting work. Defaults to False.
        result (Future): Acknowledgement received by the execution thread. Defaults to a fresh
            future for this transition.
    """

    completer: Completer
    busy: bool = False
    result: Future = field(default_factory=Future)
