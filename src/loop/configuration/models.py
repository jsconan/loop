"""Define validated workspace configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings

from .. import constants
from ..models import (
    FileInputMode,
    HyperparameterPolicy,
    ReasoningEffort,
    RepetitionDetection,
    RetentionPolicy,
    StructuredOutputMode,
)

CONFIGURATION_VERSION = 1


class BackendSettings(BaseModel):
    """Configure the OpenAI-compatible backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = constants.DEFAULT_BASE_URL
    default_model: str = constants.DEFAULT_MODEL
    api_key: SecretStr = SecretStr(constants.DEFAULT_API_KEY)
    context_window: int | None = Field(default=None, gt=0)
    max_retries: int = Field(default=constants.DEFAULT_MAX_RETRIES, ge=0)
    request_timeout_seconds: float = Field(default=constants.DEFAULT_REQUEST_TIMEOUT_SECONDS, gt=0)
    max_generation_seconds: float | None = Field(
        default=constants.DEFAULT_MAX_GENERATION_SECONDS, gt=0
    )
    max_response_chars: int | None = Field(default=constants.DEFAULT_MAX_RESPONSE_CHARS, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    repetition_detection: RepetitionDetection = constants.DEFAULT_REPETITION_DETECTION
    repetition_penalty: float | None = Field(default=None, gt=0)
    repetition_min_pattern_size: int = Field(
        default=constants.DEFAULT_REPETITION_MIN_PATTERN_SIZE, ge=0
    )
    repetition_max_pattern_size: int = Field(
        default=constants.DEFAULT_REPETITION_MAX_PATTERN_SIZE, ge=0
    )
    repetition_min_count: int = Field(default=constants.DEFAULT_REPETITION_MIN_COUNT, ge=0)
    file_input_mode: FileInputMode | None = None
    structured_output_mode: StructuredOutputMode = constants.DEFAULT_STRUCTURED_OUTPUT_MODE
    structured_output_max_retries: int = Field(
        default=constants.DEFAULT_STRUCTURED_OUTPUT_MAX_RETRIES, ge=0
    )
    hyperparameter_policy: HyperparameterPolicy = constants.DEFAULT_HYPERPARAMETER_POLICY
    retention_policy: RetentionPolicy | None = constants.DEFAULT_RETENTION_POLICY

    def model_post_init(self, __context: object, /) -> None:
        """Validate finite durations and coherent repetition thresholds."""
        if not all(
            isfinite(value)
            for value in (self.request_timeout_seconds, self.max_generation_seconds)
            if value is not None
        ):
            raise ValueError("Backend duration settings must be finite.")
        if self.repetition_penalty is not None and not isfinite(self.repetition_penalty):
            raise ValueError("Repetition penalty must be finite.")
        if self.repetition_min_pattern_size > self.repetition_max_pattern_size:
            raise ValueError("Minimum repetition pattern size cannot exceed the maximum.")
        if self.repetition_max_pattern_size > 0 and self.repetition_min_count < 2:
            raise ValueError("Repetition minimum count must be at least two when enabled.")


class LoopSettings(BaseModel):
    """Configure interactive-loop behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_name: str = constants.DEFAULT_AGENT_NAME
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: ReasoningEffort | None = None
    stream: bool = True
    debug: bool = False
    compaction_threshold: float = Field(default=constants.DEFAULT_COMPACTION_THRESHOLD, gt=0, le=1)
    prompt_on_recoverable_error: bool = True
    max_agent_turns: int = Field(default=constants.DEFAULT_AGENT_MAX_TURNS, ge=0)


class WebSettings(BaseModel):
    """Configure web-content retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_agent: str = constants.DEFAULT_USER_AGENT


class ToolSettings(BaseModel):
    """Configure built-in tool execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command_timeout: float = Field(default=constants.DEFAULT_COMMAND_TIMEOUT, gt=0)


class LoggingSettings(BaseModel):
    """Configure operational logging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        constants.DEFAULT_OPERATIONAL_LOG_LEVEL
    )
    max_bytes: int = Field(default=constants.DEFAULT_OPERATIONAL_LOG_BYTES, gt=0)
    backup_count: int = Field(default=constants.DEFAULT_OPERATIONAL_LOG_BACKUPS, ge=0)


class TelemetrySettings(BaseModel):
    """Configure telemetry batching and shutdown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    queue_capacity: int = Field(default=constants.DEFAULT_TELEMETRY_QUEUE_CAPACITY, gt=0)
    batch_size: int = Field(default=constants.DEFAULT_TELEMETRY_BATCH_SIZE, gt=0)
    flush_seconds: float = Field(default=constants.DEFAULT_TELEMETRY_FLUSH_SECONDS, gt=0)
    shutdown_timeout: float = Field(default=constants.DEFAULT_TELEMETRY_SHUTDOWN_TIMEOUT, gt=0)
    sqlite_busy_timeout_ms: int = Field(
        default=constants.DEFAULT_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS, gt=0
    )


class ApplicationSettings(BaseSettings):
    """Represent the immutable effective application configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: int = CONFIGURATION_VERSION
    backend: BackendSettings = BackendSettings()
    loop: LoopSettings = LoopSettings()
    web: WebSettings = WebSettings()
    tools: ToolSettings = ToolSettings()
    logging: LoggingSettings = LoggingSettings()
    telemetry: TelemetrySettings = TelemetrySettings()


@dataclass(frozen=True)
class ConfigurationEntry:
    """Describe one user-configurable effective value.

    Args:
        path (str): Dot-separated setting path.
        value (object): Redacted effective value.
        source (str): Source currently supplying the effective value.
        default (object): Redacted built-in default value.
        description (str): Human-readable setting description.
        choices (tuple[object, ...]): Finite allowed values, when applicable.
        secret (bool): Whether input and output must be treated as sensitive.
    """

    path: str
    value: object
    source: str
    default: object
    description: str
    choices: tuple[object, ...]
    secret: bool
