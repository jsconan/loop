"""Carry acknowledged interaction calls across the conversation boundary."""

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from prompt_toolkit.completion import Completer

from ..utils import new_uuid

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
        queued_inputs (int): Number of steering and queued inputs waiting in the mailbox.
    """

    state: Literal[
        "Interrupting",
        "Cancelling",
        "Cancelled",
        "Steering",
        "Waiting for your response",
        "Idle",
        "Working",
    ]
    queued_inputs: int


class SteeringOutcome(Enum):
    """Describe whether a steering message was admitted and interrupted the active run."""

    INTERRUPTED = "interrupted"
    RUN_INACTIVE = "run_inactive"
    INTERRUPT_REJECTED = "interrupt_rejected"
    QUEUE_FULL = "queue_full"
    QUEUED = "queued"


class PendingInputPriorityOutcome(Enum):
    """Describe the result of changing the newest waiting input's priority."""

    NO_PENDING_INPUT = "no_pending_input"
    UPDATED = "updated"
    INTERRUPTED = "interrupted"


class PendingInputRecallOutcome(Enum):
    """Describe whether a waiting input was returned to the editor."""

    NO_PENDING_INPUT = "no_pending_input"
    RECALLED = "recalled"


class ResponseOutcome(Enum):
    """Describe how a pending interaction response was handled."""

    NO_REQUEST = "no_request"
    PROMPTED = "prompted"
    ANSWERED = "answered"
    INVALID = "invalid"
    PROMPT_REQUIRED = "prompt_required"
    MASKED_INPUT_REQUIRED = "masked_input_required"


class InputAdmissionKind(Enum):
    """Identify ordinary input and owner shutdown admissions."""

    MESSAGE = "message"
    STOP = "stop"


@dataclass(order=True, frozen=True)
class InputAdmission:
    """Carry one prioritized input or shutdown event into the session owner.

    Args:
        priority (int): Queue priority, with steering ahead of ordinary queued input.
        sequence (int): Monotonic arrival order used to break priority ties.
        kind (InputAdmissionKind): Whether this event carries a message or shutdown signal.
        message (object | None): Submitted message or durable input envelope, or ``None`` for
            shutdown.
        input_id (str | None): Durable identity, when the input was accepted by a session store.
        session_id (str | None): Owning session for a durable input.
        bypass_commands (bool): Whether message content was already classified as ordinary text.
    """

    priority: int
    sequence: int
    kind: InputAdmissionKind = field(compare=False)
    message: object | None = field(default=None, compare=False)
    input_id: str | None = field(default=None, compare=False)
    session_id: str | None = field(default=None, compare=False)
    bypass_commands: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class AdmittedConversationInput:
    """Carry durable input identity from mailbox delivery to canonical history.

    Args:
        input_id (str): Durable identity used to atomically consume the accepted input.
        session_id (str): Session that accepted the input.
        message (str): User-provided message text.
        bypass_commands (bool): Whether the message must bypass command dispatch.
    """

    input_id: str
    session_id: str
    message: str
    bypass_commands: bool


class ConversationRunState(Enum):
    """Describe the owner lifecycle around an input boundary."""

    IDLE = "idle"
    WORKING = "working"


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
        request_id (str): Unique identity for this interaction request.
        session_id (str | None): Session owning the request, when available.
        run_id (str | None): Agent run owning the request, when available.
        operation_id (str | None): Tool operation owning the request, when available.
    """

    function: Callable[..., Any]
    args: tuple
    kwargs: dict
    kind: Literal["output", "prompt", "confirm", "terminal", "terminal_exit"] = EVENT_OUTPUT
    result: Future = field(default_factory=Future)
    request_id: str = field(default_factory=lambda: str(new_uuid()))
    session_id: str | None = None
    run_id: str | None = None
    operation_id: str | None = None


@dataclass
class RunStateChanged:
    """Publish that the conversation owner reached an input or work boundary.

    Args:
        completer (Completer): Live completion source guarded by execution ownership.
        state (ConversationRunState): Whether the owner is waiting for input or working.
        result (Future): Acknowledgement received by the execution thread. Defaults to a fresh
            future for this transition.
    """

    completer: Completer
    state: ConversationRunState = ConversationRunState.IDLE
    result: Future = field(default_factory=Future)


@dataclass
class ConversationCompletion:
    """Carry the settled outcome of the conversation owner task.

    Args:
        error (BaseException | None): Failure raised by the owner, or ``None`` after success.
    """

    error: BaseException | None = None
