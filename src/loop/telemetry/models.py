"""Define immutable normalized telemetry records and correlation context."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from ..utils.models import Scalar

type TelemetryValue = Scalar | tuple["TelemetryValue", ...] | MappingProxyType
TelemetrySignal = Literal["activity", "error", "audit", "trace"]
TelemetrySeverity = Literal["debug", "info", "warning", "error", "fatal"]


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    """Carry optional correlation metadata across component boundaries."""

    session_id: str | None = None
    message_sequence: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Represent one immutable, normalized record accepted by telemetry adapters."""

    record_id: str
    timestamp_ns: int
    observed_timestamp_ns: int
    signal: TelemetrySignal
    event_name: str
    event_sequence: int
    session_id: str | None = None
    message_sequence: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    severity: str | None = None
    attributes: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    payload: TelemetryValue = None
    payload_sha256: str | None = None
    schema_version: int = 1
