"""Expose the stable telemetry facade and background persistence lifecycle."""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar, Token
from typing import cast
from uuid import uuid4

from .. import constants
from ..utils import payload_digest, safe_scalar
from .adapters import NoOpTelemetryAdapter, TelemetryAdapter
from .models import TelemetryContext, TelemetryRecord, TelemetrySeverity, TelemetrySignal
from .policy import OperationalDisclosurePolicy, freeze

_LOGGER = logging.getLogger(__name__)
_OPERATIONAL_LOGGER = logging.getLogger("loop.operational")
_CURRENT_CONTEXT: ContextVar[TelemetryContext | None] = ContextVar(
    "loop_telemetry_context", default=None
)
_ACTIVE_TELEMETRY: Telemetry | None = None
_OPERATIONAL_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
}


class Telemetry:
    """Record correlated activity, errors, audit evidence, and full traces safely.

    Args:
        adapter (TelemetryAdapter | None): Persistence adapter. Defaults to a no-op adapter.
        queue_capacity (int): Maximum buffered records before overload behavior applies.
        batch_size (int): Maximum records persisted in one adapter call.
        flush_seconds (float): Maximum delay before a partial batch is persisted.
    """

    def __init__(
        self,
        adapter: TelemetryAdapter | None = None,
        *,
        queue_capacity: int = constants.DEFAULT_TELEMETRY_QUEUE_CAPACITY,
        batch_size: int = constants.DEFAULT_TELEMETRY_BATCH_SIZE,
        flush_seconds: float = constants.DEFAULT_TELEMETRY_FLUSH_SECONDS,
    ) -> None:
        if queue_capacity <= 0 or batch_size <= 0 or flush_seconds <= 0:
            raise ValueError("Telemetry queue, batch, and flush values must be positive.")
        self._adapter = adapter or NoOpTelemetryAdapter()
        self._queue: queue.Queue[TelemetryRecord | object] = queue.Queue(queue_capacity)
        self._batch_size = batch_size
        self._flush_seconds = flush_seconds
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop = object()
        self._flush = object()
        self._closed = False
        self._dropped_activity = 0
        self._write_failed = threading.Event()
        self._policy = OperationalDisclosurePolicy()
        self._thread = threading.Thread(
            target=self._run_writer,
            name="loop-telemetry-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_activity(self) -> int:
        """Return the number of activity records discarded under overload.

        Returns:
            int: Dropped activity count.
        """
        return self._dropped_activity

    @staticmethod
    def trace_id() -> str:
        """Generate a W3C-compatible trace identifier.

        Returns:
            str: Random 32-character lowercase hexadecimal identifier.
        """
        return secrets.token_hex(16)

    @staticmethod
    def span_id() -> str:
        """Generate a W3C-compatible span identifier.

        Returns:
            str: Random 16-character lowercase hexadecimal identifier.
        """
        return secrets.token_hex(8)

    def activity(
        self,
        event_name: str,
        *,
        severity: TelemetrySeverity = "debug",
        **attributes: object,
    ) -> None:
        """Record minimized operational activity.

        Args:
            event_name (str): Stable event class name.
            severity (TelemetrySeverity): Operational importance. Defaults to ``"debug"``.
            **attributes (object): Structured operational metadata governed by minimization policy.
        """
        self._emit("activity", event_name, severity=severity, attributes=attributes)

    def error(
        self,
        event_name: str,
        *,
        error_type: str,
        severity: TelemetrySeverity = "error",
        exception: BaseException | None = None,
        **attributes: object,
    ) -> None:
        """Record a sanitized error with optional correlation context.

        Args:
            event_name (str): Stable event class name.
            error_type (str): Stable machine-readable error code.
            severity (TelemetrySeverity): Error severity label.
            exception (BaseException | None): Exception supplying only its qualified type.
            **attributes (object): Structured minimized diagnostic metadata.
        """
        safe = dict(attributes)
        safe["error.type"] = error_type
        if exception is not None:
            safe["exception.type"] = f"{type(exception).__module__}.{type(exception).__qualname__}"
        self._emit("error", event_name, severity=severity, attributes=safe)

    def audit(self, event_name: str, **attributes: object) -> None:
        """Record a minimized audit decision.

        Args:
            event_name (str): Stable audit event class name.
            **attributes (object): Structured decision metadata.
        """
        self._emit("audit", event_name, attributes=attributes)

    def trace_event(
        self,
        event_name: str,
        *,
        payload: object = None,
        payload_sha256: str | None = None,
        **attributes: object,
    ) -> None:
        """Record complete policy-prepared execution evidence without further scrubbing.

        Args:
            event_name (str): Stable trace event class name.
            payload (object): Complete content already safe for its execution boundary.
            payload_sha256 (str | None): Digest proving exact payload identity when applicable.
            **attributes (object): Structured trace metadata.
        """
        self._emit(
            "trace",
            event_name,
            attributes=attributes,
            payload=payload,
            payload_sha256=payload_sha256,
        )

    @contextmanager
    def context(
        self,
        *,
        session_id: str | None = None,
        message_sequence: int | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> Generator[TelemetryContext]:
        """Bind correlation metadata for nested component calls.

        Args:
            session_id (str | None): Persistent or provisional session identifier.
            message_sequence (int | None): Message sequence within the session.
            trace_id (str | None): W3C-compatible trace identifier, generated when omitted.
            span_id (str | None): Current span identifier, generated when omitted.

        Yields:
            TelemetryContext: Bound correlation context.
        """
        parent = _CURRENT_CONTEXT.get() or TelemetryContext()
        context = TelemetryContext(
            session_id=session_id if session_id is not None else parent.session_id,
            message_sequence=(
                message_sequence if message_sequence is not None else parent.message_sequence
            ),
            trace_id=trace_id or parent.trace_id or self.trace_id(),
            span_id=span_id or self.span_id(),
            parent_span_id=parent.span_id,
        )
        token = _CURRENT_CONTEXT.set(context)
        try:
            yield context
        finally:
            _CURRENT_CONTEXT.reset(token)

    def bind(self, context: TelemetryContext) -> Token[TelemetryContext | None]:
        """Bind an explicit context until its token is reset.

        Args:
            context (TelemetryContext): Correlation context to activate.

        Returns:
            Token[TelemetryContext | None]: Token accepted by ``reset``.
        """
        return _CURRENT_CONTEXT.set(context)

    def reset(self, token: Token[TelemetryContext | None]) -> None:
        """Restore the context that preceded an explicit bind.

        Args:
            token (Token[TelemetryContext | None]): Token returned by ``bind``.
        """
        _CURRENT_CONTEXT.reset(token)

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for accepted records and adapter buffers to flush.

        Args:
            timeout (float | None): Maximum seconds to wait, or indefinitely when omitted.

        Returns:
            bool: Whether the queue drained before the timeout.
        """
        if self._closed:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            self._queue.put(self._flush, timeout=timeout)
        except queue.Full:
            return False
        while self._queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(constants.DEFAULT_TELEMETRY_FLUSH_POLL_SECONDS)
        try:
            with self._write_lock:
                self._adapter.flush()
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            _diagnose("flush_failed", error)
            return False
        return not self._write_failed.is_set()

    def close(self, timeout: float | None = constants.DEFAULT_TELEMETRY_SHUTDOWN_TIMEOUT) -> bool:
        """Flush accepted records, stop the writer, and close the adapter.

        Args:
            timeout (float | None): Maximum seconds to wait for queued records before shutdown.

        Returns:
            bool: Whether queued records flushed and adapter shutdown succeeded.
        """
        if self._closed:
            return True
        flushed = self.flush(timeout)
        if not flushed and self._queue.unfinished_tasks:
            return False
        self._queue.put(self._stop)
        self._thread.join()
        closed = True
        try:
            with self._write_lock:
                self._adapter.close()
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            _diagnose("close_failed", error)
            closed = False
        self._closed = True
        return flushed and closed

    def _emit(
        self,
        signal: TelemetrySignal,
        event_name: str,
        *,
        attributes: Mapping[str, object],
        severity: TelemetrySeverity | None = None,
        payload: object = None,
        payload_sha256: str | None = None,
    ) -> None:
        if self._closed:
            return
        try:
            context = _CURRENT_CONTEXT.get() or TelemetryContext()
            now = time.time_ns()
            safe_attributes = (
                self._policy.normalize(attributes) if signal != "trace" else freeze(attributes)
            )
            record = TelemetryRecord(
                record_id=f"tel_{uuid4().hex}",
                timestamp_ns=now,
                observed_timestamp_ns=now,
                signal=signal,
                event_name=event_name,
                event_sequence=self._next_sequence(),
                session_id=context.session_id,
                message_sequence=context.message_sequence,
                trace_id=context.trace_id,
                span_id=context.span_id,
                parent_span_id=context.parent_span_id,
                severity=severity,
                attributes=safe_attributes,
                payload=freeze(payload),
                payload_sha256=payload_sha256,
            )
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            _diagnose("record_rejected", error)
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            if signal == "activity":
                self._dropped_activity += 1
                return
            try:
                self._queue.put(
                    record,
                    timeout=constants.DEFAULT_TELEMETRY_PRIORITY_ENQUEUE_TIMEOUT,
                )
            except queue.Full:
                self._write((record,))

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _run_writer(self) -> None:
        batch: list[TelemetryRecord] = []
        while True:
            try:
                item = self._queue.get(timeout=self._flush_seconds)
            except queue.Empty:
                if batch:
                    self._write(tuple(batch))
                    batch.clear()
                continue
            if item is self._stop:
                self._queue.task_done()
                return
            if item is self._flush:
                if batch:
                    self._write(tuple(batch))
                    batch.clear()
                self._queue.task_done()
                continue
            batch.append(cast(TelemetryRecord, item))
            self._queue.task_done()
            if len(batch) >= self._batch_size:
                self._write(tuple(batch))
                batch.clear()

    def _write(self, records: tuple[TelemetryRecord, ...]) -> None:
        try:
            with self._write_lock:
                self._adapter.write_batch(records)
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            self._write_failed.set()
            _diagnose("adapter_failed", error)


def set_telemetry(telemetry: Telemetry | None) -> None:
    """Set the process-wide facade used by package instrumentation.

    Args:
        telemetry (Telemetry | None): Active facade, or ``None`` to restore no-op behavior.
    """
    global _ACTIVE_TELEMETRY  # pylint: disable=global-statement
    _ACTIVE_TELEMETRY = telemetry


def get_telemetry() -> Telemetry | None:
    """Return the process-wide facade when telemetry is configured.

    Returns:
        Telemetry | None: Active facade, or ``None`` before initialization.
    """
    return _ACTIVE_TELEMETRY


def telemetry_activity(
    event_name: str,
    *,
    severity: TelemetrySeverity = "debug",
    **attributes: object,
) -> None:
    """Record optional process-wide operational activity.

    Args:
        event_name (str): Stable event class name.
        severity (TelemetrySeverity): Operational importance. Defaults to ``"debug"``.
        **attributes (object): Structured operational metadata governed by minimization policy.
    """
    _OPERATIONAL_LOGGER.log(
        _OPERATIONAL_LEVELS.get(severity, logging.DEBUG),
        "Operational activity",
        extra={"event.name": safe_scalar(event_name)},
    )
    if _ACTIVE_TELEMETRY is not None:
        _ACTIVE_TELEMETRY.activity(event_name, severity=severity, **attributes)


def telemetry_error(
    event_name: str,
    *,
    error_type: str,
    severity: TelemetrySeverity = "error",
    exception: BaseException | None = None,
    **attributes: object,
) -> None:
    """Record an optional process-wide sanitized error.

    Args:
        event_name (str): Stable event class name.
        error_type (str): Stable machine-readable error code.
        severity (TelemetrySeverity): Error severity label.
        exception (BaseException | None): Exception supplying only its qualified type.
        **attributes (object): Structured minimized diagnostic metadata.
    """
    diagnostic = {
        "event.name": safe_scalar(event_name),
        "error.type": safe_scalar(error_type),
    }
    if exception is not None:
        diagnostic["exception.type"] = (
            f"{type(exception).__module__}.{type(exception).__qualname__}"
        )
    _OPERATIONAL_LOGGER.log(
        _OPERATIONAL_LEVELS.get(severity, logging.ERROR),
        "Operational error",
        extra=diagnostic,
    )
    if _ACTIVE_TELEMETRY is not None:
        _ACTIVE_TELEMETRY.error(
            event_name,
            error_type=error_type,
            severity=severity,
            exception=exception,
            **attributes,
        )


def telemetry_audit(event_name: str, **attributes: object) -> None:
    """Record an optional process-wide audit decision.

    Args:
        event_name (str): Stable audit event class name.
        **attributes (object): Structured decision metadata.
    """
    if _ACTIVE_TELEMETRY is not None:
        _ACTIVE_TELEMETRY.audit(event_name, **attributes)


def telemetry_trace_event(
    event_name: str,
    *,
    payload: object = None,
    payload_factory: Callable[[], object] | None = None,
    payload_sha256: str | None = None,
    digest_payload: bool = False,
    **attributes: object,
) -> None:
    """Record an optional process-wide full trace event.

    Args:
        event_name (str): Stable trace event class name.
        payload (object): Complete content already safe for its execution boundary.
        payload_factory (Callable[[], object] | None): Deferred payload construction used only
            when telemetry is active.
        payload_sha256 (str | None): Digest proving exact payload identity when applicable.
        digest_payload (bool): Whether to compute the canonical payload digest after construction.
        **attributes (object): Structured trace metadata.
    """
    if _ACTIVE_TELEMETRY is None:
        return
    try:
        traced_payload = payload_factory() if payload_factory is not None else payload
        digest = payload_digest(traced_payload) if digest_payload else payload_sha256
    except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        _diagnose("payload_preparation_failed", error)
        return
    _ACTIVE_TELEMETRY.trace_event(
        event_name,
        payload=traced_payload,
        payload_sha256=digest,
        **attributes,
    )


def telemetry_span(
    *,
    session_id: str | None = None,
    message_sequence: int | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> AbstractContextManager[TelemetryContext | None]:
    """Return an optional process-wide correlation span.

    Args:
        session_id (str | None): Persistent or provisional session identifier.
        message_sequence (int | None): Message sequence within the session.
        trace_id (str | None): W3C-compatible trace identifier, generated when omitted.
        span_id (str | None): Current span identifier, generated when omitted.

    Returns:
        AbstractContextManager[TelemetryContext | None]: Active span or a no-op context manager.
    """
    if _ACTIVE_TELEMETRY is None:
        return nullcontext()
    return _ACTIVE_TELEMETRY.context(
        session_id=session_id,
        message_sequence=message_sequence,
        trace_id=trace_id,
        span_id=span_id,
    )


def _diagnose(failure: str, error: BaseException) -> None:
    _LOGGER.error(
        "Telemetry operation failed",
        extra={
            "error.type": "telemetry.internal_failure",
            "exception.type": f"{type(error).__module__}.{type(error).__qualname__}",
            "telemetry.failure": failure,
        },
    )
