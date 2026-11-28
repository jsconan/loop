"""Retain session-owned command jobs and complete output without replaying shell source."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from contextvars import copy_context
from uuid import uuid4

from .. import constants
from ..errors import Problem, log_problem
from ..utils.content import _ContentSpool, decode_content_cursor, encode_content_cursor
from ..utils.process import render_terminal_text

_LOGGER = logging.getLogger(__name__)
_OUTPUT_RESULT_KEYS = frozenset({"stdout", "stderr", "terminal_output", "output_handles"})
MAX_ACTIVE_JOBS = 4
MAX_RETAINED_JOBS = 8
JOB_RETENTION_SECONDS = 600.0
JOB_CLOSE_SECONDS = 5.0
JOB_READ_CANCELLATION_CHECK_SECONDS = 0.1
MAX_TERMINAL_INPUT_BYTES = 4_096
MAX_TERMINAL_PENDING_BYTES = 65_536
MAX_TERMINAL_SIZE = 500


class CommandCancellation(threading.Event):
    """Combine job cancellation with the originating run until handoff.

    Args:
        job (threading.Event): Job-owned cancellation signal.
        origin (threading.Event | None): Original run cancellation signal.
        handed_off (threading.Event): Signal that ownership has transferred to the job.
        lock (threading.Lock): Synchronize signal reads with the ownership transfer.
    """

    _job: threading.Event
    _origin: threading.Event | None
    _handed_off: threading.Event
    _lock: threading.Lock

    def __init__(
        self,
        job: threading.Event,
        origin: threading.Event | None,
        handed_off: threading.Event,
        lock: threading.Lock,
    ) -> None:
        super().__init__()
        self._job = job
        self._origin = origin
        self._handed_off = handed_off
        self._lock = lock

    def is_set(self) -> bool:
        """Return whether either active cancellation source has been signalled.

        Returns:
            bool: Whether cancellation has been requested.
        """
        with self._lock:
            return (
                super().is_set()
                or self._job.is_set()
                or (
                    not self._handed_off.is_set()
                    and self._origin is not None
                    and self._origin.is_set()
                )
            )

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for either active cancellation source.

        Args:
            timeout (float | None): Maximum wait, or None to wait indefinitely.

        Returns:
            bool: Whether cancellation was requested before the wait ended.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            self._job.wait(
                JOB_READ_CANCELLATION_CHECK_SECONDS
                if remaining is None
                else min(remaining, JOB_READ_CANCELLATION_CHECK_SECONDS)
            )
        return True


class CommandJob:
    """Own one admitted attempt, its complete output spools, and its cancellation signal.

    Args:
        owner (str): Stable session scratch identity allowed to inspect this job.
        runtime_limit_seconds (float): Hard execution limit captured when the job starts.
        terminal (bool): Whether this job owns a native interactive terminal.
    """

    owner: str
    runtime_limit_seconds: float
    terminal: bool
    handle: str
    cancellation_event: threading.Event
    ready: threading.Event
    finished: threading.Event
    thread: threading.Thread | None
    completed_at: float | None
    admitted_at: float | None
    _condition: threading.Condition
    _outputs: dict[str, _ContentSpool]
    _observed: dict[str, bool]
    _capture_complete: dict[str, bool]
    _output_handles_exposed: bool
    _status: str
    _result: dict | Problem | None
    _error: BaseException | None
    _capture_error: OSError | None
    _terminal_frames: deque[bytes]
    _terminal_pending_bytes: int
    _handoff_lock: threading.Lock
    _handed_off: bool
    _handoff_complete: threading.Event
    _cancellation_lock: threading.Lock

    def __init__(self, owner: str, runtime_limit_seconds: float, terminal: bool = False) -> None:
        self.owner = owner
        self.runtime_limit_seconds = runtime_limit_seconds
        self.terminal = terminal
        self.handle = uuid4().hex
        self.cancellation_event = threading.Event()
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.thread = None
        self.completed_at = None
        self.admitted_at = None
        self._condition = threading.Condition()
        self._outputs = {
            stream: _ContentSpool(f"command {stream}") for stream in ("stdout", "stderr")
        }
        self._observed = {stream: False for stream in self._outputs}
        self._capture_complete = {stream: False for stream in self._outputs}
        self._output_handles_exposed = False
        self._status = "preparing"
        self._result = None
        self._error = None
        self._capture_error = None
        self._terminal_frames = deque()
        self._terminal_pending_bytes = 0
        self._handoff_lock = threading.Lock()
        self._handed_off = False
        self._handoff_complete = threading.Event()
        self._cancellation_lock = threading.Lock()

    def handoff(self, cancellation_event: threading.Event | None = None) -> None:
        """Prevent further prompts and carry cancellation through the handoff boundary.

        Args:
            cancellation_event (threading.Event | None): Originating run's stop signal.
        """
        error = None
        result = None
        with self._handoff_lock:
            with self._cancellation_lock:
                if cancellation_event is not None and cancellation_event.is_set():
                    self.cancel()
                self._handed_off = True
                self._handoff_complete.set()
                if cancellation_event is not None and cancellation_event.is_set():
                    self.cancel()
            with self._condition:
                error, self._error = self._error, None
                result = self._result
        if error is not None and isinstance(result, Problem):
            log_problem(_LOGGER, result, error)

    def begin_interaction(self) -> bool:
        """Reserve the original interaction while an approval is being evaluated.

        Returns:
            bool: Whether the caller must release the reservation.
        """
        self._handoff_lock.acquire()
        if self._handed_off:
            self._handoff_lock.release()
            return False
        return True

    def cancellation_signal(self, origin: threading.Event | None) -> CommandCancellation:
        """Combine the origin and job signals until ownership is handed off.

        Args:
            origin (threading.Event | None): Original run's cancellation signal.

        Returns:
            CommandCancellation: Signal for the native supervisor.
        """
        return CommandCancellation(
            self.cancellation_event, origin, self._handoff_complete, self._cancellation_lock
        )

    def end_interaction(self) -> None:
        """Release a previously reserved interaction."""
        self._handoff_lock.release()

    def admitted(self) -> None:
        """Mark authorization complete before native preparation starts."""
        with self._condition:
            if self.finished.is_set() or self.admitted_at is not None:
                return
            self.admitted_at = time.monotonic()
            self._status = "running"
            self.ready.set()
            self._condition.notify_all()

    def publish(self, stream: str, content: str) -> None:
        """Append every observed output character to its stream spool.

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
            if self.finished.is_set() or self._capture_error is not None:
                return
            self._observed[stream] |= bool(content)
            try:
                self._outputs[stream].append(content)
            except OSError as error:
                self._capture_error = error
                self.cancellation_event.set()

    def output_sizes(self) -> dict[str, int]:
        """Return the successfully retained byte count for each process stream.

        Returns:
            dict[str, int]: Retained UTF-8 byte counts keyed by ``stdout`` and ``stderr``.
        """
        with self._condition:
            return {stream: output.size_bytes for stream, output in self._outputs.items()}

    def capture_result_output(self, result: object, before: dict[str, int]) -> None:
        """Retain result text when a backend did not publish that stream incrementally.

        Args:
            result (object): Result containing optional ``stdout`` and ``stderr`` text.
            before (dict[str, int]): Per-stream byte counts taken before the attempt.
        """
        with self._condition:
            for stream in ("stdout", "stderr"):
                content = getattr(result, stream, "")
                if not isinstance(content, str) or not content:
                    continue
                if self._outputs[stream].size_bytes == before[stream]:
                    try:
                        self._outputs[stream].append(
                            render_terminal_text(content) if self.terminal else content
                        )
                    except OSError as error:
                        self._capture_error = error
                        self.cancellation_event.set()

    def read_output(self, stream: str, cursor: str | None, max_bytes: int) -> dict:
        """Read one immediately available page from a retained stream.

        Args:
            stream (str): Captured ``stdout`` or ``stderr`` stream name.
            cursor (str | None): Opaque content cursor, or None to start at byte zero.
            max_bytes (int): Maximum UTF-8 bytes returned in the page.

        Returns:
            dict: One page, continuation cursor when needed, and capture completion state.

        Raises:
            ValueError: The stream, cursor, or requested page size is invalid.
        """
        if stream not in self._outputs:
            raise ValueError("Unknown command output stream.")
        with self._condition:
            output = self._outputs[stream]
            if not output.retained:
                raise ValueError("Command output is no longer retained.")
            page = output.read_page(cursor, max_bytes)
            end = page["end_byte"]
            result = {
                **page,
                "job_complete": self.finished.is_set(),
                "capture_complete": self._capture_complete[stream],
                "capture_failed": self._capture_error is not None,
            }
            if end < output.size_bytes or not result["job_complete"]:
                result["next_cursor"] = encode_content_cursor(output.handle, end)
            return result

    def stream_result(self, stream: str, discarded: int | None) -> dict:
        """Return a bounded result preview backed by the retained stream.

        Args:
            stream (str): Captured ``stdout`` or ``stderr`` stream name.
            discarded (int | None): Native capture overflow count, or None if draining stopped.

        Returns:
            dict: Bounded preview, loss metadata and cached-content continuation when needed.

        Raises:
            ValueError: The stream name is invalid.
        """
        if stream not in self._outputs:
            raise ValueError("Unknown command output stream.")
        with self._condition:
            complete = discarded is not None and (discarded == 0 or self._observed[stream])
            self._capture_complete[stream] = complete
            return self._outputs[stream].stream_result(complete)

    def output_handles(self) -> dict[str, str]:
        """Return stable cached-content handles for both growing streams.

        Returns:
            dict[str, str]: Cached-content handles keyed by ``stdout`` and ``stderr``.
        """
        with self._condition:
            handles = {
                stream: output.handle for stream, output in self._outputs.items() if output.retained
            }
            self._output_handles_exposed |= bool(handles)
            return handles

    def _output_page(self, stream: str, cursor: str | None, max_bytes: int) -> dict:
        """Render the output available after one stream cursor."""
        output = self._outputs[stream]
        if max_bytes:
            page = self.read_output(stream, cursor, max_bytes)
        else:
            start = decode_content_cursor(cursor, output.handle) if cursor is not None else 0
            size = output.size_bytes
            page = {
                "content": "",
                "start_byte": start,
                "end_byte": start,
                "included_bytes": 0,
                "size_bytes": size,
            }
            if start < size or not self.finished.is_set():
                page["next_cursor"] = encode_content_cursor(output.handle, start)

        capture_complete = self._capture_complete[stream]
        size = output.size_bytes
        end = page["end_byte"]
        more_available = end < size or not self.finished.is_set()
        result = {
            "content": page["content"],
            "captured_bytes": size,
            "included_bytes": page["included_bytes"],
            "start_byte": page["start_byte"],
            "end_byte": end,
            "truncated": more_available or not capture_complete,
            "capture_complete": capture_complete,
            "discarded_characters": 0 if capture_complete else None,
            "decoding_loss": (
                "none" if capture_complete and not output.has_replacement else "unknown"
            ),
        }
        if "next_cursor" in page:
            result["next_cursor"] = page["next_cursor"]
        if end < size:
            result["next_tool_call"] = {
                "name": "read_command_output",
                "arguments": {
                    "handle": self.handle,
                    "stream": stream,
                    "cursor": page["next_cursor"],
                    "max_bytes": constants.MAX_TOOL_CONTENT_BYTES,
                },
            }
            if max_bytes:
                outcome = (
                    self._result.metadata if isinstance(self._result, Problem) else self._result
                )
                final_stream = (outcome or {}).get(stream)
                tail = final_stream.get("tail") if isinstance(final_stream, dict) else None
                if tail is not None and tail["end_byte"] - tail["start_byte"] <= max_bytes:
                    result["tail"] = tail
        return result

    def status_with_output(
        self,
        stdout_cursor: str | None = None,
        stderr_cursor: str | None = None,
    ) -> dict:
        """Return current job status with bounded pages from both streams.

        Returns:
            dict: Current lifecycle status and cursor pages sized to fit the tool-result limit.
        """
        with self._condition:
            max_bytes = constants.MAX_TOOL_CONTENT_BYTES
            while True:
                result = self.inspect()
                result["stdout"] = self._output_page("stdout", stdout_cursor, max_bytes)
                result["stderr"] = self._output_page("stderr", stderr_cursor, max_bytes)
                encoded = json.dumps({"ok": True, "result": result}).encode("utf-8")
                if len(encoded) <= constants.MAX_TOOL_RESULT_BYTES - 512 or max_bytes == 0:
                    return result
                max_bytes //= 2

    def release_unreferenced_output(self, result: dict | Problem) -> None:
        """Discard completed stream files that were returned inline and never exposed by handle.

        Args:
            result (dict | Problem): Final tool result returned by the initial command call.
        """
        with self._condition:
            if self._output_handles_exposed:
                return
            output = result.metadata if isinstance(result, Problem) else result
            handles = output.get("output_handles", {}) if isinstance(output, dict) else {}
            for stream, spool in self._outputs.items():
                item = output.get(stream) if isinstance(output, dict) else None
                if (not isinstance(item, dict) or item.get("handle") != spool.handle) and (
                    not isinstance(handles, dict) or handles.get(stream) != spool.handle
                ):
                    spool.discard()

    def complete(self, result: dict | Problem, error: BaseException | None = None) -> None:
        """Publish the final result once and wake any blocked readers.

        Args:
            result (dict | Problem): Rendered command outcome or structured failure.
            error (BaseException | None): Unexpected failure to propagate to the initial caller.
        """
        with self._handoff_lock:
            with self._condition:
                if self.finished.is_set():
                    return
                capture_error = self._capture_error
                if capture_error is not None:
                    result = Problem(
                        code="command_output.capture_failed",
                        title="Command output capture failed",
                        detail=(
                            "The command was stopped because its output could not be retained "
                            "completely. Previously captured output remains readable."
                        ),
                        operation="run_command",
                        metadata={
                            "output_handles": self.output_handles(),
                            "capture_failed": True,
                        },
                    )
                    error = None
                self._result = result
                self._error = error if not self._handed_off else None
                self._terminal_frames.clear()
                self._terminal_pending_bytes = 0
                self._status = "complete"
                self.completed_at = time.monotonic()
                self.ready.set()
                self.finished.set()
                self._condition.notify_all()
            report_capture = capture_error is not None
            report = self._handed_off and error is not None and isinstance(result, Problem)
        if report_capture:
            log_problem(_LOGGER, result, capture_error)
        if report:
            log_problem(_LOGGER, result, error)

    def cancel(self) -> None:
        """Request cancellation without claiming that descendants have exited."""
        with self._condition:
            if not self.finished.is_set():
                self.cancellation_event.set()
                self._terminal_frames.clear()
                self._terminal_pending_bytes = 0
                self._condition.notify_all()

    def result(self) -> dict | Problem | None:
        """Return the settled result, or None while the command is active.

        Returns:
            dict | Problem | None: Original result after settlement, otherwise None.

        Raises:
            BaseException: Unexpected worker failure that settled before handoff.
        """
        with self._condition:
            error, self._error = self._error, None
            result = self._result
        if error is not None:
            raise error
        return result

    def wait(self, seconds: float, cancellation_event: threading.Event | None) -> None:
        """Wait for settlement or an originating run cancellation.

        Args:
            seconds (float): Maximum time to wait.
            cancellation_event (threading.Event | None): Signal interrupting the initial wait.
        """
        if cancellation_event is None:
            self.finished.wait(seconds)
            return
        deadline = time.monotonic() + seconds
        while not self.finished.is_set() and not cancellation_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.finished.wait(min(remaining, JOB_READ_CANCELLATION_CHECK_SECONDS))

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
            dict: Handle, lifecycle status, elapsed time, runtime limit, output sizes and result
                summary.
        """
        with self._condition:
            result = self._result
            outcome = result.metadata if isinstance(result, Problem) else result
            lifecycle = outcome.get("lifecycle", {}) if isinstance(outcome, dict) else {}
            diagnostics = (
                outcome.get("sandbox_diagnostics", {}) if isinstance(outcome, dict) else {}
            )
            elapsed = (
                max(
                    0.0,
                    (self.completed_at if self.completed_at is not None else time.monotonic())
                    - self.admitted_at,
                )
                if self.admitted_at is not None
                else None
            )
            return {
                "handle": self.handle,
                "terminal": self.terminal,
                "status": self._status,
                "elapsed_seconds": elapsed,
                "runtime_limit_seconds": self.runtime_limit_seconds,
                "cancel_requested": self.cancellation_event.is_set(),
                "main_process": lifecycle.get("main_process", "unknown"),
                "descendant_cleanup": lifecycle.get("descendant_cleanup", "uncertain"),
                "stdout_bytes": self._outputs["stdout"].size_bytes,
                "stderr_bytes": self._outputs["stderr"].size_bytes,
                "capture_failed": self._capture_error is not None,
                "stdout_capture_complete": self._capture_complete["stdout"],
                "stderr_capture_complete": self._capture_complete["stderr"],
                "boundary": outcome.get("boundary") if isinstance(outcome, dict) else None,
                "exit_code": outcome.get("exit_code") if isinstance(outcome, dict) else None,
                "outcome": diagnostics.get("outcome")
                or (result.code if isinstance(result, Problem) else None),
                "result": self._result_summary(result),
                "output_handles": self.output_handles(),
            }

    @staticmethod
    def _result_summary(result: dict | Problem | None) -> dict | None:
        """Return final status details without embedding command output content.

        Args:
            result (dict | Problem | None): Settled command result, if available.

        Returns:
            dict | None: Safe non-output summary for a completed command.
        """
        if isinstance(result, Problem):
            summary = result.model_dump(exclude={"metadata"})
            metadata = {
                key: value
                for key, value in result.metadata.items()
                if key not in _OUTPUT_RESULT_KEYS
            }
        elif isinstance(result, dict):
            summary = {}
            metadata = {
                key: value for key, value in result.items() if key not in _OUTPUT_RESULT_KEYS
            }
        else:
            return None
        if metadata:
            summary["metadata"] = metadata
        return summary

    def poll(
        self,
        wait_seconds: float,
        *,
        max_wait_seconds: float = constants.MAX_COMMAND_WAIT_SECONDS,
        stdout_cursor: str | None = None,
        stderr_cursor: str | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> dict:
        """Wait for job completion or timeout, then return available output pages.

        Args:
            wait_seconds (float): Maximum time to wait for job completion.
            max_wait_seconds (float): Configured maximum accepted wait for this call.
            stdout_cursor (str | None): Cursor for the next stdout page.
            stderr_cursor (str | None): Cursor for the next stderr page.
            cancellation_event (threading.Event | None): Stop waiting when this read is interrupted,
                without cancelling the command.

        Returns:
            dict: Current lifecycle status and available pages for both process streams.
                Invalid cursors are reported in cursor_errors, with a fresh page for that stream.

        Raises:
            ValueError: The wait budget or configured maximum is invalid or unsupported.
        """
        if (
            isinstance(max_wait_seconds, bool)
            or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds)
            or not 0 <= max_wait_seconds <= constants.MAX_COMMAND_WAIT_SECONDS
        ):
            raise ValueError("Command job poll maximum is outside the supported range.")
        if (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, (int, float))
            or not 0 <= wait_seconds <= max_wait_seconds
            or not math.isfinite(wait_seconds)
        ):
            raise ValueError(
                "Command job poll wait must be finite and between 0 and "
                f"{max_wait_seconds:g} seconds."
            )
        with self._condition:
            cursors = {"stdout": stdout_cursor, "stderr": stderr_cursor}
            cursor_errors = {}
            for stream, cursor in cursors.items():
                if cursor is not None:
                    output = self._outputs[stream]
                    try:
                        offset = decode_content_cursor(cursor, output.handle)
                        if offset > output.size_bytes:
                            raise ValueError("Content cursor is beyond the end of the content.")
                    except ValueError as error:
                        cursors[stream] = None
                        cursor_errors[stream] = str(error)
            deadline = time.monotonic() + wait_seconds
            while not self.finished.is_set() and not (
                cancellation_event is not None and cancellation_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                interval = (
                    min(remaining, JOB_READ_CANCELLATION_CHECK_SECONDS)
                    if cancellation_event is not None
                    else remaining
                )
                self._condition.wait(interval)
            result = self.status_with_output(cursors["stdout"], cursors["stderr"])
            if cursor_errors:
                result["cursor_errors"] = cursor_errors
            return result


class CommandJobs:
    """Limit and own all process-local command jobs across session switches."""

    _lock: threading.RLock
    _jobs: dict[str, CommandJob]
    _closed: bool

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs = {}
        self._closed = False

    def start(
        self,
        owner: str,
        runner: Callable[[CommandJob], dict | Problem],
        *,
        runtime_limit_seconds: float,
        terminal: bool = False,
        on_settled: Callable[[], None] | None = None,
    ) -> CommandJob | Problem:
        """Start one bounded, independently owned job.

        Args:
            owner (str): Stable session scratch identity.
            runner (Callable[[CommandJob], dict | Problem]): Authorized execution workflow.
            runtime_limit_seconds (float): Hard execution limit captured for the job.
            terminal (bool): Whether the worker owns one interactive terminal.
            on_settled (Callable[[], None] | None): Release worker-owned resources once settled.

        Returns:
            CommandJob | Problem: Owned job or a capacity or shutdown refusal.
        """

        def execute() -> None:
            """Settle one owned worker even when execution raises unexpectedly."""
            try:
                with self._lock:
                    if self._jobs.get(job.handle) is not job:
                        return
                error = None
                try:
                    result = runner(job)
                except BaseException as caught:  # noqa: BLE001 - settle every worker failure.
                    error = caught
                    result = Problem(
                        code="command_job.failed",
                        title="Command job failed",
                        detail="Command execution failed; prior effects may have occurred.",
                        operation="run_command",
                    )
                job.complete(result, error)
            finally:
                if on_settled is not None:
                    on_settled()

        with self._lock:
            if self._closed:
                return self._problem("Command execution service is closing.")
            if sum(not job.finished.is_set() for job in self._jobs.values()) >= MAX_ACTIVE_JOBS:
                return self._problem("The active command job limit has been reached.")
            completed = sorted(
                (job for job in self._jobs.values() if job.finished.is_set()),
                key=lambda job: job.completed_at or 0,
            )
            while len(self._jobs) >= MAX_RETAINED_JOBS and completed:
                old = completed.pop(0)
                del self._jobs[old.handle]
            job = CommandJob(owner, runtime_limit_seconds, terminal)
            self._jobs[job.handle] = job
            execution_context = copy_context()
            thread = threading.Thread(target=lambda: execution_context.run(execute), daemon=True)
            job.thread = thread
            try:
                thread.start()
            except BaseException as error:  # noqa: BLE001 - release a failed reservation.
                del self._jobs[job.handle]
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
