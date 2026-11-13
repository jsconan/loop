"""Retain bounded, session-owned command continuations without replaying shell source."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from itertools import islice
from typing import NamedTuple
from uuid import uuid4

from ..errors import Problem, log_problem
from ..utils.process import render_terminal_text

_LOGGER = logging.getLogger(__name__)
MAX_ACTIVE_JOBS = 4
MAX_RETAINED_JOBS = 8
MAX_REMEMBERED_CALL_IDS = 4096
MAX_JOB_OUTPUT_BYTES = 1_000_000
MAX_OUTPUT_EVENT_CHARS = 1_024
MAX_JOB_OUTPUT_EVENTS = 16_384
MAX_READ_EVENTS = 4
JOB_RETENTION_SECONDS = 600.0
JOB_CLOSE_SECONDS = 5.0
JOB_READ_CANCELLATION_CHECK_SECONDS = 0.1
MAX_TERMINAL_INPUT_BYTES = 4_096
MAX_TERMINAL_PENDING_BYTES = 65_536
MAX_TERMINAL_SIZE = 500


class CommandOutputEvent(NamedTuple):
    """Retain one ordered output chunk with its stream and cursor.

    Args:
        cursor (int): Monotonic cursor for readers.
        stream (str): Captured stdout or stderr stream.
        content (str): Decoded output chunk.
    """

    cursor: int
    stream: str
    content: str


class CommandJob:
    """Own one admitted attempt, its bounded output, and its cancellation signal.

    Args:
        owner (str): Stable session scratch identity allowed to inspect this job.
        call_id (str | None): Model call identity used to avoid duplicate in-process launch.
        terminal (bool): Whether this job owns a native interactive terminal.
    """

    owner: str
    call_id: str | None
    terminal: bool
    handle: str
    cancellation_event: threading.Event
    ready: threading.Event
    finished: threading.Event
    thread: threading.Thread | None
    completed_at: float | None
    _condition: threading.Condition
    _events: deque[CommandOutputEvent]
    _event_bytes: int
    _oldest_cursor: int
    _next_cursor: int
    _exposed_cursor: int
    _status: str
    _result: dict | Problem | None
    _dropped_characters: int
    _terminal_frames: deque[bytes]
    _terminal_pending_bytes: int

    def __init__(self, owner: str, call_id: str | None, terminal: bool = False) -> None:
        self.owner = owner
        self.call_id = call_id
        self.terminal = terminal
        self.handle = uuid4().hex
        self.cancellation_event = threading.Event()
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.thread = None
        self.completed_at = None
        self._condition = threading.Condition()
        self._events = deque()
        self._event_bytes = 0
        self._oldest_cursor = 0
        self._next_cursor = 0
        self._exposed_cursor = 0
        self._status = "preparing"
        self._result = None
        self._dropped_characters = 0
        self._terminal_frames = deque()
        self._terminal_pending_bytes = 0

    def admitted(self) -> None:
        """Mark authorization complete before native preparation starts."""
        with self._condition:
            if self.finished.is_set():
                return
            self._status = "running"
            self.ready.set()
            self._condition.notify_all()

    def publish(self, stream: str, content: str) -> None:
        """Retain ordered bounded output events until the job completes.

        Args:
            stream (str): Exact ``stdout`` or ``stderr`` stream name.
            content (str): Decoded text observed by the native supervisor.

        Raises:
            ValueError: The stream name is not one of the two captured process streams.
        """
        if stream not in {"stdout", "stderr"}:
            raise ValueError("Unknown command output stream.")
        if self.terminal:
            content = render_terminal_text(content)
        with self._condition:
            if self.finished.is_set():
                return
            start = 0
            while start < len(content):
                last = self._events[-1] if self._events else None
                extend = (
                    last is not None
                    and last.stream == stream
                    and last.cursor >= self._exposed_cursor
                    and len(last.content) < MAX_OUTPUT_EVENT_CHARS
                )
                capacity = MAX_OUTPUT_EVENT_CHARS - len(last.content) if extend else 0
                chunk = content[start : start + (capacity if extend else MAX_OUTPUT_EVENT_CHARS)]
                start += len(chunk)
                size = len(chunk.encode("utf-8"))
                if extend:
                    self._events[-1] = last._replace(content=last.content + chunk)
                else:
                    self._events.append(CommandOutputEvent(self._next_cursor, stream, chunk))
                    self._next_cursor += 1
                self._event_bytes += size
                while (
                    self._event_bytes > MAX_JOB_OUTPUT_BYTES
                    or len(self._events) > MAX_JOB_OUTPUT_EVENTS
                ):
                    dropped = self._events.popleft()
                    self._event_bytes -= len(dropped.content.encode("utf-8"))
                    self._dropped_characters += len(dropped.content)
                    self._oldest_cursor = dropped.cursor + 1
            self._condition.notify_all()

    def complete(self, result: dict | Problem) -> None:
        """Publish the final result once and wake any blocked readers.

        Args:
            result (dict | Problem): Rendered command outcome or structured failure.
        """
        with self._condition:
            if self.finished.is_set():
                return
            self._result = result
            self._terminal_frames.clear()
            self._terminal_pending_bytes = 0
            self._status = "complete"
            self.completed_at = time.monotonic()
            self.ready.set()
            self.finished.set()
            self._condition.notify_all()

    def cancel(self) -> None:
        """Request cancellation without claiming that descendants have exited."""
        with self._condition:
            if not self.finished.is_set():
                self.cancellation_event.set()
                self._terminal_frames.clear()
                self._terminal_pending_bytes = 0
                self._condition.notify_all()

    def _enqueue_terminal(self, frame: bytes) -> None:
        """Retain one bounded control frame only until the native worker sends it."""
        with self._condition:
            if not self.terminal or self.finished.is_set() or self.cancellation_event.is_set():
                raise ValueError("Interactive terminal is unavailable for this job.")
            if self._terminal_pending_bytes + len(frame) > MAX_TERMINAL_PENDING_BYTES:
                raise ValueError("Interactive terminal input is backed up; read output and retry.")
            self._terminal_frames.append(frame)
            self._terminal_pending_bytes += len(frame)
            self._condition.notify_all()

    def write_terminal(self, content: str) -> None:
        """Queue bounded UTF-8 terminal input without logging or durable storage.

        Args:
            content (str): Exact text to send to the owned terminal.

        Raises:
            TypeError: Input is not text.
            ValueError: Input is empty, oversized, backed up, or the job is unavailable.
        """
        if not isinstance(content, str):
            raise TypeError("Interactive terminal input must be text.")
        data = content.encode("utf-8")
        if not 0 < len(data) <= MAX_TERMINAL_INPUT_BYTES:
            raise ValueError("Interactive terminal input must contain 1 to 4096 UTF-8 bytes.")
        self._enqueue_terminal(b"I" + len(data).to_bytes(2, "big") + data)

    def resize_terminal(self, rows: int, columns: int) -> None:
        """Queue one bounded window-size change for the owned terminal.

        Args:
            rows (int): New terminal height in cells.
            columns (int): New terminal width in cells.

        Raises:
            ValueError: Dimensions lie outside the qualified range or job is unavailable.
        """
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= MAX_TERMINAL_SIZE
            for value in (rows, columns)
        ):
            raise ValueError("Terminal rows and columns must be between 1 and 500.")
        self._enqueue_terminal(b"Z" + rows.to_bytes(2, "big") + columns.to_bytes(2, "big"))

    def terminal_eof(self) -> None:
        """Queue the terminal's canonical end-of-input character.

        Raises:
            ValueError: The job has no live interactive terminal.
        """
        self._enqueue_terminal(b"F")

    def terminal_interrupt(self) -> None:
        """Queue the terminal's canonical interrupt character.

        Raises:
            ValueError: The job has no live interactive terminal.
        """
        self._enqueue_terminal(b"X")

    def next_terminal_frame(self) -> bytes | None:
        """Remove the next transient terminal control frame for the native transport.

        Returns:
            bytes | None: Next bounded frame, or None when the queue is empty.
        """
        with self._condition:
            if not self._terminal_frames:
                return None
            frame = self._terminal_frames.popleft()
            self._terminal_pending_bytes -= len(frame)
            return frame

    def inspect(self) -> dict:
        """Return the lifecycle state without command source or output content.

        Returns:
            dict: Handle, status, cancellation flag, cursors, and output loss count.
        """
        with self._condition:
            outcome = self._result.metadata if isinstance(self._result, Problem) else self._result
            lifecycle = outcome.get("lifecycle", {}) if isinstance(outcome, dict) else {}
            return {
                "handle": self.handle,
                "terminal": self.terminal,
                "status": self._status,
                "cancel_requested": self.cancellation_event.is_set(),
                "main_process": lifecycle.get("main_process", "unknown"),
                "descendant_cleanup": lifecycle.get("descendant_cleanup", "uncertain"),
                "oldest_cursor": self._oldest_cursor,
                "next_cursor": self._next_cursor,
                "dropped_characters": self._dropped_characters,
            }

    def read(
        self,
        cursor: int,
        wait_seconds: float = 0,
        *,
        include_output: bool = False,
        cancellation_event: threading.Event | None = None,
    ) -> dict | Problem:
        """Return a bounded output page and the result after retained output is delivered.

        Args:
            cursor (int): Next event cursor requested by this session.
            wait_seconds (float): Maximum time to wait for output or completion.
            include_output (bool): Include the final captured stream previews explicitly.
            cancellation_event (threading.Event | None): Stop waiting when this read is interrupted,
                without cancelling the command.

        Returns:
            dict | Problem: Ordered page and next cursor, or an expired-cursor error.

        Raises:
            ValueError: Cursor is invalid or beyond produced output, or the wait budget is invalid.
        """
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError("Command output cursor must be a nonnegative integer.")
        if (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, (int, float))
            or not 0 <= wait_seconds <= threading.TIMEOUT_MAX
            or not math.isfinite(wait_seconds)
        ):
            raise ValueError("Command job wait must be finite and within the supported range.")
        with self._condition:
            if cursor > self._next_cursor:
                raise ValueError("Command output cursor is beyond the produced output.")
            deadline = time.monotonic() + wait_seconds if cancellation_event is not None else 0
            while True:
                remaining = (
                    max(0, deadline - time.monotonic())
                    if cancellation_event is not None
                    else wait_seconds
                )
                interval = (
                    min(remaining, JOB_READ_CANCELLATION_CHECK_SECONDS)
                    if cancellation_event is not None
                    else remaining
                )
                ready = self._condition.wait_for(
                    lambda: (
                        cursor < self._oldest_cursor
                        or cursor < self._next_cursor
                        or self.finished.is_set()
                        or (cancellation_event is not None and cancellation_event.is_set())
                    ),
                    timeout=interval,
                )
                if ready or cancellation_event is None or remaining <= 0:
                    break
            if cursor < self._oldest_cursor:
                return Problem(
                    code="command_job.cursor_expired",
                    title="Command output cursor expired",
                    detail="Earlier command output exceeded the retained job limit.",
                    operation="read_command_job",
                    metadata={"oldest_cursor": self._oldest_cursor},
                )
            events = list(
                islice(
                    (
                        {"cursor": event.cursor, "stream": event.stream, "content": event.content}
                        for event in self._events
                        if event.cursor >= cursor
                    ),
                    MAX_READ_EVENTS,
                )
            )
            replayed = cursor < self._exposed_cursor and bool(events)
            next_cursor = events[-1]["cursor"] + 1 if events else cursor
            self._exposed_cursor = max(self._exposed_cursor, next_cursor)
            result = self._result if next_cursor >= self._next_cursor else None
            if isinstance(result, Problem):
                result = result.model_dump()
            if isinstance(result, dict) and not include_output:
                result = dict(result)
                container = result.get("metadata") if "metadata" in result else result
                if isinstance(container, dict):
                    container = dict(container)
                    for stream in ("stdout", "stderr"):
                        value = container.get(stream)
                        if isinstance(value, dict):
                            container[stream] = {
                                key: item
                                for key, item in value.items()
                                if key not in {"content", "tail"}
                            }
                    if "metadata" in result:
                        result["metadata"] = container
                    else:
                        result = container
            page = {
                **self.inspect(),
                "events": events,
                "next_cursor": next_cursor,
                "result": result,
            }
            if replayed and result is None:
                page["continuation"] = (
                    "This page includes previously read output. Pass this page's next_cursor "
                    "to read subsequent output and wait for changes."
                )
            return page


class CommandJobs:
    """Limit and own all process-local command jobs across session switches."""

    _lock: threading.RLock
    _jobs: dict[str, CommandJob]
    _calls: dict[tuple[str, str, int | None], str]
    _closed: bool

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs = {}
        self._calls = {}
        self._closed = False

    def start(
        self,
        owner: str,
        call_id: str | None,
        runner: Callable[[CommandJob], dict | Problem],
        *,
        terminal: bool = False,
        occurrence: int | None = None,
    ) -> CommandJob | Problem:
        """Start one bounded job or reuse the same in-process model call.

        Args:
            owner (str): Stable session scratch identity.
            call_id (str | None): Model call identity, if one exists.
            runner (Callable[[CommandJob], dict | Problem]): Authorized execution workflow.
            terminal (bool): Whether the worker owns one interactive terminal.
            occurrence (int | None): Stable session-history index of this call, if available.

        Returns:
            CommandJob | Problem: Owned job or a capacity or shutdown refusal.
        """

        def execute() -> None:
            """Settle one owned worker even when execution raises unexpectedly."""
            with self._lock:
                if self._jobs.get(job.handle) is not job:
                    return
            try:
                result = runner(job)
            except BaseException as error:  # noqa: BLE001 - settle every worker failure.
                problem = Problem(
                    code="command_job.failed",
                    title="Command job failed",
                    detail="Command execution failed; prior effects may have occurred.",
                    operation="run_command",
                )
                log_problem(_LOGGER, problem, error)
                result = problem
            job.complete(result)

        with self._lock:
            if self._closed:
                return self._problem("Command execution service is closing.")
            key = (owner, call_id, occurrence) if call_id is not None else None
            if key is not None and (handle := self._calls.get(key)) is not None:
                previous = self._jobs.get(handle)
                if previous is None:
                    return self._problem(
                        "Command job handle was evicted; source will not be replayed."
                    )
                if (
                    previous.completed_at is not None
                    and time.monotonic() - previous.completed_at >= JOB_RETENTION_SECONDS
                ):
                    return self._problem(
                        "Command job handle has expired; source will not be replayed."
                    )
                return previous
            if key is not None and len(self._calls) >= MAX_REMEMBERED_CALL_IDS:
                return self._problem("The remembered model call limit has been reached.")
            if sum(not job.finished.is_set() for job in self._jobs.values()) >= MAX_ACTIVE_JOBS:
                return self._problem("The active command job limit has been reached.")
            completed = sorted(
                (job for job in self._jobs.values() if job.finished.is_set()),
                key=lambda job: job.completed_at or 0,
            )
            while len(self._jobs) >= MAX_RETAINED_JOBS and completed:
                old = completed.pop(0)
                del self._jobs[old.handle]
            job = CommandJob(owner, call_id, terminal)
            self._jobs[job.handle] = job
            if key is not None:
                self._calls[key] = job.handle
            thread = threading.Thread(target=execute, daemon=True)
            job.thread = thread
            try:
                thread.start()
            except BaseException as error:  # noqa: BLE001 - release a failed reservation.
                del self._jobs[job.handle]
                if key is not None:
                    self._calls.pop(key, None)
                problem = self._problem("Command job worker could not start.")
                log_problem(_LOGGER, problem, error)
                return problem
            return job

    def get(self, owner: str, handle: str) -> CommandJob | Problem:
        """Resolve only a live process-local job owned by the selected session.

        Args:
            owner (str): Current session scratch identity.
            handle (str): Opaque handle returned when the job began.

        Returns:
            CommandJob | Problem: Owned job or a safe unavailable-handle error.
        """
        with self._lock:
            job = self._jobs.get(handle)
            if (
                job is None
                or job.owner != owner
                or job.completed_at is not None
                and time.monotonic() - job.completed_at >= JOB_RETENTION_SECONDS
            ):
                return self._problem(
                    "Command job handle is unknown, expired, or belongs to another session."
                )
            return job

    def close_terminal(self, owner: str, handle: str) -> dict | Problem:
        """Cancel and forget a settled owned terminal without losing an unsettled handle.

        Args:
            owner (str): Current session scratch identity.
            handle (str): Opaque terminal handle to close.

        Returns:
            dict | Problem: Closed state or an inspectable pending cleanup state.
        """
        job = self.get(owner, handle)
        if isinstance(job, Problem):
            return job
        if not job.terminal:
            return self._problem("The handle does not own an interactive terminal.")
        job.cancel()
        if not job.finished.wait(JOB_CLOSE_SECONDS):
            return {**job.inspect(), "closed": False}
        with self._lock:
            self._jobs.pop(handle, None)
        return {**job.inspect(), "closed": True}

    def inspect(self, owner: str) -> list[dict]:
        """List current-session jobs without revealing other sessions' handles.

        Args:
            owner (str): Current session scratch identity.

        Returns:
            list[dict]: Retained owned job states, including completed jobs.
        """
        with self._lock:
            return [
                job.inspect()
                for job in self._jobs.values()
                if job.owner == owner
                and (
                    job.completed_at is None
                    or time.monotonic() - job.completed_at < JOB_RETENTION_SECONDS
                )
            ]

    def has_active(self) -> bool:
        """Report whether a native backend still owns an unsettled job.

        Returns:
            bool: Whether any reserved job has not completed.
        """
        with self._lock:
            return any(not job.finished.is_set() for job in self._jobs.values())

    def close(self) -> bool:
        """Request every active job to stop and wait for bounded cleanup.

        Returns:
            bool: Whether every worker settled within the cleanup budget.
        """
        with self._lock:
            self._closed = True
            jobs = tuple(self._jobs.values())
        for job in jobs:
            if not job.finished.is_set():
                job.cancel()
        deadline = time.monotonic() + JOB_CLOSE_SECONDS
        for job in jobs:
            if job.thread is not None:
                job.thread.join(max(0, deadline - time.monotonic()))
        return all(job.finished.is_set() for job in jobs)

    @staticmethod
    def _problem(detail: str) -> Problem:
        """Return a safe refusal without exposing another job or its command source."""
        return Problem(
            code="command_job.unavailable",
            title="Command job unavailable",
            detail=detail,
            operation="run_command",
        )
