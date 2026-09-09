"""Load, validate, and persist workspace configuration."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import tomlkit
from filelock import FileLock
from pydantic import SecretStr

from .. import constants
from .models import (
    ApplicationSettings,
    ConfigurationEntry,
)

_ENVIRONMENT_FIELDS = {
    "BASE_URL": ("backend", "base_url"),
    "DEFAULT_MODEL": ("backend", "default_model"),
    "OPENAI_API_KEY": ("backend", "api_key"),
    "CONTEXT_WINDOW": ("backend", "context_window"),
    "OPENAI_MAX_RETRIES": ("backend", "max_retries"),
    "OPENAI_HYPERPARAMETER_POLICY": ("backend", "hyperparameter_policy"),
    "USER_AGENT": ("web", "user_agent"),
    "LOOP_COMMAND_TIMEOUT": ("tools", "command_timeout"),
    "LOOP_AGENT_NAME": ("loop", "agent_name"),
    "LOOP_MODEL": ("loop", "model"),
    "LOOP_TEMPERATURE": ("loop", "temperature"),
    "LOOP_REASONING_EFFORT": ("loop", "reasoning_effort"),
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
    """Manage complete user and sparse workspace TOML configuration documents.

    Args:
        path (Path | str): Complete user TOML configuration path.
        workspace_path (Path | str | None): Optional sparse workspace override path.
    """

    _path: Path
    _document: tomlkit.TOMLDocument
    _workspace_path: Path | None
    _workspace_document: tomlkit.TOMLDocument
    _effective: ApplicationSettings | None
    _sources: dict[str, str]
    _environment: dict[str, str]
    _session_values: dict[str, Any]

    def __init__(self, path: Path | str, workspace_path: Path | str | None = None) -> None:
        self._path = Path(path).expanduser().resolve()
        self._workspace_path = (
            Path(workspace_path).expanduser().resolve() if workspace_path is not None else None
        )
        self._document = tomlkit.document()
        self._workspace_document = tomlkit.document()
        self._effective = None
        self._sources = {}
        self._environment = {}
        self._session_values = {}

    @property
    def path(self) -> Path:
        """Return the durable configuration path.

        Returns:
            Path: Workspace TOML configuration path.
        """
        return self._path

    def initialize(self) -> Path:
        """Create the complete default configuration document when it is absent.

        Returns:
            Path: The existing or newly created configuration path.
        """
        with self._lock_for("user"):
            if self.path.exists():
                return self.path
            self._document = self._default_document()
            self._save_document(self.path, self._document)
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
        self._workspace_document = self._read_workspace_document()
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
            str: ``"session"``, ``"environment"``, ``"workspace"``, ``"user"``, or
                ``"default"``.
        """
        self._split_path(dotted_path)
        return self._sources.get(dotted_path, "default")

    def set(
        self,
        dotted_path: str,
        value: Any,
        *,
        scope: Literal["user", "workspace"] = "workspace",
    ) -> ApplicationSettings:
        """Set one TOML value and return the validated effective configuration.

        Args:
            dotted_path (str): Dot-separated configuration field path.
            value (Any): Replacement value stored in the TOML document.
            scope (Literal["user", "workspace"]): Durable destination. Defaults to workspace.

        Returns:
            ApplicationSettings: Validated configuration after the edit.
        """
        with self._lock_for(scope):
            sections = self._split_path(dotted_path)
            document = self._document_for_write(scope)
            original = tomlkit.dumps(document)
            table = document.get(sections[0])
            defaults = ApplicationSettings().model_dump(mode="python")
            if sections[0] not in defaults or sections[1] not in defaults[sections[0]]:
                raise ValueError(f"Unknown configuration field '{dotted_path}'.")
            if not isinstance(table, Mapping):
                table = tomlkit.table()
                document[sections[0]] = table
            table[sections[1]] = value
            try:
                self._validate_candidate()
            except Exception:
                if scope == "workspace" and self._workspace_path is not None:
                    self._workspace_document = tomlkit.parse(original)
                else:
                    self._document = tomlkit.parse(original)
                self._resolve()
                raise
            self._save_scope(scope)
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

    def reset(
        self,
        dotted_path: str,
        *,
        scope: Literal["user", "workspace"] = "workspace",
    ) -> ApplicationSettings:
        """Reset one setting to its built-in default in the specified scope.

        Args:
            dotted_path (str): Dot-separated configuration field path.
            scope (Literal["user", "workspace"]): Scope to reset. Defaults to workspace.

        Returns:
            ApplicationSettings: Validated settings after storing the default.
        """
        if scope == "workspace":
            return self.unset(dotted_path, scope="workspace")
        return self.set(dotted_path, self._default_value(dotted_path), scope=scope)

    def reset_session(self, dotted_path: str) -> ApplicationSettings:
        """Reset one in-memory override to its built-in default.

        Args:
            dotted_path (str): Dot-separated configuration field path.

        Returns:
            ApplicationSettings: Effective configuration after storing the default.
        """
        return self.set_session(dotted_path, self._default_value(dotted_path))

    def reset_all(
        self,
        *,
        scope: Literal["session", "user", "workspace"] = "workspace",
    ) -> ApplicationSettings:
        """Reset every value in one configuration scope.

        Args:
            scope (Literal["session", "user", "workspace"]): Scope whose values are restored.
                Defaults to workspace.

        Returns:
            ApplicationSettings: Effective configuration after the reset.
        """
        if scope == "session":
            self._session_values = {
                entry.path: self._default_value(entry.path) for entry in self.entries
            }
            return self._resolve()
        if scope == "workspace":
            with self._lock_for(scope):
                self._workspace_document = tomlkit.document()
                self._save_scope("workspace")
            return self._resolve()
        with self._lock_for(scope):
            self._document = self._read_document()
            default_values = self._stored_defaults()
            self._document["config_version"] = default_values["config_version"]
            for section, settings in default_values.items():
                if section == "config_version":
                    continue
                table = self._document.get(section)
                if not isinstance(table, Mapping):
                    table = tomlkit.table()
                    self._document[section] = table
                for field, value in settings.items():
                    if value is None:
                        if field in table:
                            del table[field]
                    else:
                        table[field] = value
            self._save_document(self.path, self._document)
        return self._resolve()

    def unset(
        self,
        dotted_path: str,
        *,
        scope: Literal["user", "workspace"] = "workspace",
    ) -> ApplicationSettings:
        """Reset one user value or remove one workspace override.

        Args:
            dotted_path (str): Dot-separated configuration field path.
            scope (Literal["user", "workspace"]): Durable scope. Defaults to workspace.

        Returns:
            ApplicationSettings: Validated settings after the removal.

        Raises:
            ValueError: If the field is not present in the configuration document.
        """
        if scope == "user":
            return self.set(dotted_path, self._default_value(dotted_path), scope="user")
        with self._lock_for(scope):
            section, field = self._split_path(dotted_path)
            document = self._document_for_write(scope)
            table = document.get(section)
            if not isinstance(table, Mapping) or field not in table:
                raise ValueError(f"Unknown configuration field '{dotted_path}'.")
            del table[field]
            if not table:
                del document[section]
            self._validate_candidate()
            self._save_scope(scope)
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

    def unset_all(
        self,
        *,
        scope: Literal["session", "user", "workspace"] = "workspace",
    ) -> ApplicationSettings:
        """Remove every value from one configuration scope.

        Args:
            scope (Literal["session", "user", "workspace"]): Scope to clear or restore.
                Defaults to workspace.

        Returns:
            ApplicationSettings: Effective configuration after the removal.
        """
        if scope == "session":
            self._session_values.clear()
            return self._resolve()
        return self.reset_all(scope=scope)

    def save(self) -> Path:
        """Atomically save the current comment-preserving TOML document.

        Returns:
            Path: Persisted configuration path.
        """
        with self._lock_for("user"):
            return self._save_document(self.path, self._document)

    @staticmethod
    def _save_document(path: Path, document: tomlkit.TOMLDocument) -> Path:
        """Atomically persist one private TOML document."""
        path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        path.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            constants.PRIVATE_FILE_MODE,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(tomlkit.dumps(document))
        temporary.replace(path)
        path.chmod(constants.PRIVATE_FILE_MODE)
        return path

    def _read_document(self) -> tomlkit.TOMLDocument:
        """Return the current comment-preserving TOML document, or a default if absent."""
        if not self.path.exists():
            return self._default_document()
        return tomlkit.parse(self.path.read_text(encoding="utf-8"))

    def _read_workspace_document(self) -> tomlkit.TOMLDocument:
        """Return the sparse workspace document or an empty document."""
        if self._workspace_path is None or not self._workspace_path.exists():
            return tomlkit.document()
        return tomlkit.parse(self._workspace_path.read_text(encoding="utf-8"))

    def _document_for_write(self, scope: Literal["user", "workspace"]) -> tomlkit.TOMLDocument:
        """Reload and return the requested durable configuration document."""
        if scope == "workspace":
            self._workspace_document = self._read_workspace_document()
            return self._workspace_document
        self._document = self._read_document()
        return self._document

    def _save_scope(self, scope: Literal["user", "workspace"]) -> Path:
        """Persist one durable configuration scope."""
        if scope == "user":
            return self._save_document(self.path, self._document)
        if self._workspace_path is None:
            raise RuntimeError("Workspace path is not configured.")
        if not self._workspace_document:
            path = self._workspace_path
            path.unlink(missing_ok=True)
            return path
        return self._save_document(self._workspace_path, self._workspace_document)

    def _lock_for(self, scope: Literal["user", "workspace"]) -> FileLock:
        """Return the interprocess lock protecting one complete TOML transaction."""
        path = (
            self._path if scope == "user" or self._workspace_path is None else self._workspace_path
        )
        path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        return FileLock(path.with_name(f".{path.name}.lock"), mode=constants.PRIVATE_FILE_MODE)

    def _plain_document_values(self) -> dict[str, Any]:
        """Return ordinary Python values from a TOML document."""
        return self._document.unwrap()

    def _resolve(self) -> ApplicationSettings:
        """Resolve configured, environment, and session values into one snapshot."""
        values = self._plain_document_values()
        self._sources = self._configured_sources(
            values, "user" if self._workspace_path is not None else "workspace"
        )
        workspace_values = self._workspace_document.unwrap()
        for section, table in workspace_values.items():
            if section == "config_version":
                continue
            if isinstance(table, Mapping):
                values.setdefault(section, {}).update(table)
            else:
                values[section] = table
        self._sources.update(self._configured_sources(workspace_values, "workspace"))
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
    def _configured_sources(values: Mapping[str, Any], source: str = "workspace") -> dict[str, str]:
        """Return provenance for values explicitly present in a TOML document."""
        return {
            f"{section}.{field}": source
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
                "Loop workspace configuration. Environment variables override values here."
            )
        )
        default_values = ConfigurationManager._stored_defaults()
        document.add("config_version", default_values["config_version"])
        for name, settings in default_values.items():
            if name == "config_version":
                continue
            table = tomlkit.table()
            for key, value in settings.items():
                if value is not None:
                    table.add(key, value)
                else:
                    table.add(tomlkit.comment(f"{key} = <unset>"))
            document.add(name, table)
        return document

    @staticmethod
    def _stored_defaults() -> dict[str, Any]:
        """Return built-in defaults in a TOML-storable representation."""
        values = ApplicationSettings().model_dump(mode="python")
        return ConfigurationManager._toml_values(values)

    @staticmethod
    def _toml_values(value: Any) -> Any:
        """Convert settings values to TOML-safe values without redacting secrets."""
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        if isinstance(value, Mapping):
            return {key: ConfigurationManager._toml_values(item) for key, item in value.items()}
        return value
