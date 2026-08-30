"""Load, validate, and persist project configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings

from . import constants
from .models import FileInputMode, HyperparameterPolicy, ReasoningEffort, StructuredOutputMode

_CONFIGURATION_VERSION = 1


class BackendSettings(BaseModel):
    """Configure the OpenAI-compatible backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = constants.DEFAULT_BASE_URL
    default_model: str = constants.DEFAULT_MODEL
    api_key: SecretStr = SecretStr(constants.DEFAULT_API_KEY)
    context_window: int | None = Field(default=None, gt=0)
    max_retries: int = Field(default=constants.DEFAULT_MAX_RETRIES, ge=0)
    file_input_mode: FileInputMode | None = None
    structured_output_mode: StructuredOutputMode = constants.DEFAULT_STRUCTURED_OUTPUT_MODE
    structured_output_max_retries: int = Field(
        default=constants.DEFAULT_STRUCTURED_OUTPUT_MAX_RETRIES, ge=0
    )
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: ReasoningEffort | None = None
    hyperparameter_policy: HyperparameterPolicy = constants.DEFAULT_HYPERPARAMETER_POLICY


class LoopSettings(BaseModel):
    """Configure interactive-loop behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_name: str = constants.DEFAULT_AGENT_NAME
    model: str | None = None
    stream: bool = True
    debug: bool = False
    compaction_threshold: float = Field(default=constants.DEFAULT_COMPACTION_THRESHOLD, gt=0, le=1)
    prompt_on_recoverable_error: bool = True
    max_agent_turns: int = Field(default=constants.DEFAULT_AGENT_MAX_TURNS, ge=0)


class WebSettings(BaseModel):
    """Configure web-content retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_agent: str = constants.DEFAULT_USER_AGENT


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

    config_version: int = _CONFIGURATION_VERSION
    backend: BackendSettings = BackendSettings()
    loop: LoopSettings = LoopSettings()
    web: WebSettings = WebSettings()
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


_ENVIRONMENT_FIELDS = {
    "BASE_URL": ("backend", "base_url"),
    "DEFAULT_MODEL": ("backend", "default_model"),
    "OPENAI_API_KEY": ("backend", "api_key"),
    "CONTEXT_WINDOW": ("backend", "context_window"),
    "OPENAI_MAX_RETRIES": ("backend", "max_retries"),
    "OPENAI_TEMPERATURE": ("backend", "temperature"),
    "OPENAI_REASONING_EFFORT": ("backend", "reasoning_effort"),
    "OPENAI_HYPERPARAMETER_POLICY": ("backend", "hyperparameter_policy"),
    "USER_AGENT": ("web", "user_agent"),
    "LOOP_AGENT_NAME": ("loop", "agent_name"),
    "LOOP_MODEL": ("loop", "model"),
    "LOOP_STREAM": ("loop", "stream"),
    "LOOP_DEBUG": ("loop", "debug"),
    "LOOP_COMPACTION_THRESHOLD": ("loop", "compaction_threshold"),
    "LOOP_PROMPT_ON_RECOVERABLE_ERROR": ("loop", "prompt_on_recoverable_error"),
    "LOOP_MAX_AGENT_TURNS": ("loop", "max_agent_turns"),
    "LOOP_LOG_LEVEL": ("logging", "level"),
    "LOOP_LOG_MAX_BYTES": ("logging", "max_bytes"),
    "LOOP_LOG_BACKUP_COUNT": ("logging", "backup_count"),
    "LOOP_TELEMETRY_QUEUE_CAPACITY": ("telemetry", "queue_capacity"),
    "LOOP_TELEMETRY_BATCH_SIZE": ("telemetry", "batch_size"),
    "LOOP_TELEMETRY_FLUSH_SECONDS": ("telemetry", "flush_seconds"),
    "LOOP_TELEMETRY_SHUTDOWN_TIMEOUT": ("telemetry", "shutdown_timeout"),
    "LOOP_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS": ("telemetry", "sqlite_busy_timeout_ms"),
}


class ConfigurationManager:
    """Manage one project's comment-preserving TOML configuration document.

    Args:
        project_root (Path | str): Root directory that owns the ``.loop`` application directory.
    """

    _project_root: Path
    _document: tomlkit.TOMLDocument
    _effective: ApplicationSettings | None
    _sources: dict[str, str]
    _environment: dict[str, str]
    _session_values: dict[str, Any]

    def __init__(self, project_root: Path | str) -> None:
        self._project_root = Path(project_root).resolve()
        self._document = tomlkit.document()
        self._effective = None
        self._sources = {}
        self._environment = {}
        self._session_values = {}

    @property
    def path(self) -> Path:
        """Return the durable configuration path.

        Returns:
            Path: ``.loop/config.toml`` beneath the project root.
        """
        return self._project_root / constants.APP_DIRECTORY / constants.APP_CONFIGURATION_FILENAME

    def initialize(self) -> Path:
        """Create the complete default configuration document when it is absent.

        Returns:
            Path: The existing or newly created configuration path.
        """
        if self.path.exists():
            return self.path
        self._document = self._default_document()
        self.save()
        return self.path

    def load(self, environment: Mapping[str, str] | None = None) -> ApplicationSettings:
        """Load and validate effective settings.

        Args:
            environment (Mapping[str, str] | None): Environment override source. Defaults to the
                current process environment.

        Returns:
            ApplicationSettings: Immutable settings resolved from environment, TOML, and defaults.
        """
        self._document = self._read_document()
        if environment is None:
            environment = os.environ
        self._environment = dict(environment)
        return self._resolve()

    @property
    def effective(self) -> ApplicationSettings:
        """Return the most recently resolved immutable settings snapshot.

        Returns:
            ApplicationSettings: Latest configuration loaded by this manager.

        Raises:
            RuntimeError: If settings have not been loaded.
        """
        if self._effective is None:
            raise RuntimeError("Configuration has not been loaded.")
        return self._effective

    def reload(self, environment: Mapping[str, str] | None = None) -> ApplicationSettings:
        """Reload the document and return a new validated settings snapshot.

        Args:
            environment (Mapping[str, str] | None): Environment override source.

        Returns:
            ApplicationSettings: Fresh effective settings.
        """
        return self.load(environment)

    @property
    def entries(self) -> tuple[ConfigurationEntry, ...]:
        """Return every editable field with redacted effective values and metadata.

        Returns:
            tuple[ConfigurationEntry, ...]: Entries in application-settings declaration order.
        """
        defaults = ApplicationSettings()
        values = []
        for section_name, section_field in ApplicationSettings.model_fields.items():
            if section_name == "config_version":
                continue
            section_type = section_field.annotation
            schema = section_type.model_json_schema()
            for field_name, field in section_type.model_fields.items():
                path = f"{section_name}.{field_name}"
                current = self.get(path)
                values.append(
                    ConfigurationEntry(
                        path=path,
                        value=current,
                        source=self.source_for(path),
                        default=getattr(getattr(defaults, section_name), field_name),
                        description=field.description or field_name.replace("_", " "),
                        choices=self._choices(
                            schema["properties"][field_name], schema.get("$defs", {})
                        ),
                        secret=isinstance(current, SecretStr),
                    )
                )
        return tuple(values)

    def get(self, dotted_path: str) -> object:
        """Return one effective configuration value without exposing secret text.

        Args:
            dotted_path (str): Dot-separated configuration field path.

        Returns:
            object: Effective value, with secrets represented by ``SecretStr``.
        """
        section, field = self._split_path(dotted_path)
        return getattr(getattr(self.effective, section), field)

    def source_for(self, dotted_path: str) -> str:
        """Return the highest-priority source that supplied one setting.

        Args:
            dotted_path (str): Dot-separated configuration field path.

        Returns:
            str: ``"environment"``, ``"config"``, or ``"default"``.
        """
        self._split_path(dotted_path)
        return self._sources.get(dotted_path, "default")

    def set(self, dotted_path: str, value: Any) -> ApplicationSettings:
        """Set one TOML value and return the validated effective configuration.

        Args:
            dotted_path (str): Dot-separated configuration field path.
            value (Any): Replacement value stored in the TOML document.

        Returns:
            ApplicationSettings: Validated configuration after the edit.
        """
        sections = self._split_path(dotted_path)
        self._document = self._read_document()
        table = self._document.get(sections[0])
        defaults = ApplicationSettings().model_dump(mode="python")
        if (
            not isinstance(table, Mapping)
            or sections[0] not in defaults
            or sections[1] not in defaults[sections[0]]
        ):
            raise ValueError(f"Unknown configuration field '{dotted_path}'.")
        table[sections[1]] = value
        self._validate_candidate()
        self.save()
        return self._resolve()

    def set_session(self, dotted_path: str, value: Any) -> ApplicationSettings:
        """Set one in-memory override without changing the configuration file.

        Args:
            dotted_path (str): Dot-separated configuration field path.
            value (Any): Replacement value for this process only.

        Returns:
            ApplicationSettings: Validated effective configuration after the edit.
        """
        self._split_path(dotted_path)
        self._session_values[dotted_path] = value
        try:
            return self._resolve()
        except Exception:
            del self._session_values[dotted_path]
            raise

    def reset(self, dotted_path: str) -> ApplicationSettings:
        """Reset one file-level setting to its built-in default.

        Args:
            dotted_path (str): Dot-separated configuration field path.

        Returns:
            ApplicationSettings: Validated settings after storing the default.
        """
        return self.set(dotted_path, self._default_value(dotted_path))

    def reset_session(self, dotted_path: str) -> ApplicationSettings:
        """Reset one in-memory override to its built-in default.

        Args:
            dotted_path (str): Dot-separated configuration field path.

        Returns:
            ApplicationSettings: Effective configuration after storing the default.
        """
        return self.set_session(dotted_path, self._default_value(dotted_path))

    def reset_all(self, *, scope: Literal["session", "file"]) -> ApplicationSettings:
        """Reset every value in one configuration scope.

        Args:
            scope (Literal["session", "file"]): ``"session"`` stores defaults as process-only
                values; ``"file"`` rewrites the complete default TOML document.

        Returns:
            ApplicationSettings: Effective configuration after the reset.
        """
        if scope == "session":
            self._session_values = {
                entry.path: self._default_value(entry.path) for entry in self.entries
            }
            return self._resolve()
        self._document = self._default_document()
        self.save()
        return self._resolve()

    def unset(self, dotted_path: str) -> ApplicationSettings:
        """Remove one file-level setting and return the new effective configuration.

        Args:
            dotted_path (str): Dot-separated configuration field path.

        Returns:
            ApplicationSettings: Validated settings after the removal.

        Raises:
            ValueError: If the field is not present in the configuration document.
        """
        section, field = self._split_path(dotted_path)
        self._document = self._read_document()
        table = self._document.get(section)
        if not isinstance(table, Mapping) or field not in table:
            raise ValueError(f"Unknown configuration field '{dotted_path}'.")
        del table[field]
        self._validate_candidate()
        self.save()
        return self._resolve()

    def unset_session(self, dotted_path: str) -> ApplicationSettings:
        """Remove one in-memory override and resolve lower-precedence sources.

        Args:
            dotted_path (str): Dot-separated configuration field path.

        Returns:
            ApplicationSettings: Effective configuration after removing the override.

        Raises:
            ValueError: If the path has no in-memory override.
        """
        self._split_path(dotted_path)
        if dotted_path not in self._session_values:
            raise ValueError(f"No session override exists for '{dotted_path}'.")
        del self._session_values[dotted_path]
        return self._resolve()

    def unset_all(self, *, scope: Literal["session", "file"]) -> ApplicationSettings:
        """Remove every value from one configuration scope.

        Args:
            scope (Literal["session", "file"]): ``"session"`` clears process-only values;
                ``"file"`` clears the configuration document.

        Returns:
            ApplicationSettings: Effective configuration after the removal.
        """
        if scope == "session":
            self._session_values.clear()
            return self._resolve()
        self._document.clear()
        self.save()
        return self._resolve()

    def save(self) -> Path:
        """Atomically save the current comment-preserving TOML document.

        Returns:
            Path: Persisted configuration path.
        """
        self.path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        self.path.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(tomlkit.dumps(self._document), encoding="utf-8")
        temporary.chmod(constants.PRIVATE_FILE_MODE)
        temporary.replace(self.path)
        self.path.chmod(constants.PRIVATE_FILE_MODE)
        return self.path

    def _read_document(self) -> tomlkit.TOMLDocument:
        """Return the current comment-preserving TOML document, or a default if absent."""
        if not self.path.exists():
            return self._default_document()
        return tomlkit.parse(self.path.read_text(encoding="utf-8"))

    def _plain_document_values(self) -> dict[str, Any]:
        """Return ordinary Python values from a TOML document."""
        return self._document.unwrap()

    def _resolve(self) -> ApplicationSettings:
        """Resolve configured, environment, and session values into one snapshot."""
        values = self._plain_document_values()
        self._sources = self._configured_sources(values)
        for variable, (section, field) in _ENVIRONMENT_FIELDS.items():
            value = self._environment.get(variable)
            if value is not None:
                values.setdefault(section, {})[field] = value
                self._sources[f"{section}.{field}"] = "environment"
        for path, value in self._session_values.items():
            section, field = self._split_path(path)
            values.setdefault(section, {})[field] = value
            self._sources[path] = "session"
        self._effective = ApplicationSettings.model_validate(values)
        return self._effective

    def _validate_candidate(self) -> None:
        """Validate the current file and retained higher-precedence values."""
        self._resolve()

    @staticmethod
    def _default_value(dotted_path: str) -> object:
        """Return and validate the built-in default for one configuration path."""
        section, field = ConfigurationManager._split_path(dotted_path)
        defaults = ApplicationSettings()
        try:
            return getattr(getattr(defaults, section), field)
        except AttributeError as error:
            raise ValueError(f"Unknown configuration field '{dotted_path}'.") from error

    @staticmethod
    def _choices(schema: Mapping[str, Any], definitions: Mapping[str, Any]) -> tuple[object, ...]:
        """Return finite interactive choices described by one validation schema."""
        reference = schema.get("$ref")
        if reference:
            return ConfigurationManager._choices(
                definitions[str(reference).removeprefix("#/$defs/")], definitions
            )
        if "enum" in schema:
            return tuple(schema["enum"])
        if schema.get("type") == "boolean":
            return (True, False)
        return tuple(
            choice
            for nested in schema.get("anyOf", ())
            if isinstance(nested, Mapping)
            for choice in ConfigurationManager._choices(nested, definitions)
        )

    @staticmethod
    def _split_path(dotted_path: str) -> tuple[str, str]:
        """Validate and split one public configuration path."""
        sections = dotted_path.split(".")
        if len(sections) != 2:
            raise ValueError("Configuration fields must use 'section.field' paths.")
        return sections[0], sections[1]

    @staticmethod
    def _configured_sources(values: Mapping[str, Any]) -> dict[str, str]:
        """Return provenance for values explicitly present in a TOML document."""
        return {
            f"{section}.{field}": "config"
            for section, table in values.items()
            if isinstance(table, Mapping)
            for field in table
        }

    @staticmethod
    def _default_document() -> tomlkit.TOMLDocument:
        """Build the complete commented default TOML document."""
        document = tomlkit.document()
        document.add(
            tomlkit.comment(
                "Loop project configuration. Environment variables override values here."
            )
        )
        document.add("config_version", _CONFIGURATION_VERSION)
        defaults = ApplicationSettings()
        default_values = defaults.model_dump(mode="python")
        default_values["backend"]["api_key"] = defaults.backend.api_key.get_secret_value()
        for name, settings in default_values.items():
            if name == "config_version":
                continue
            table = tomlkit.table()
            for key, value in settings.items():
                if value is not None:
                    table.add(key, value)
            document.add(name, table)
        return document
