"""Load, validate, and persist workspace configuration."""

from __future__ import annotations

import errno
import logging
import os
import stat
import tempfile
import time
from _thread import RLock
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import tomlkit
from filelock import FileLock
from pydantic import SecretStr, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError
from tomlkit.exceptions import TOMLKitError

from .. import constants
from ..errors import Problem, log_problem, safe_error_detail
from ..interaction import Interaction, prompt_recovery_async
from ..models import CommandCapability
from ..utils import ensure_private_directory, open_selected_file
from ..utils.path import has_extended_access_acl
from .models import (
    ApplicationSettings,
    BackendSettings,
    ConfigurationEntry,
)

_LOGGER = logging.getLogger(__name__)

_ENVIRONMENT_FIELDS = {
    "BASE_URL": ("backend", "base_url"),
    "DEFAULT_MODEL": ("backend", "default_model"),
    "OPENAI_API_KEY": ("backend", "api_key"),
    "CONTEXT_WINDOW": ("backend", "context_window"),
    "OPENAI_MAX_RETRIES": ("backend", "max_retries"),
    "LOOP_REQUEST_TIMEOUT_SECONDS": ("backend", "request_timeout_seconds"),
    "LOOP_MAX_GENERATION_SECONDS": ("backend", "max_generation_seconds"),
    "LOOP_MAX_RESPONSE_CHARS": ("backend", "max_response_chars"),
    "LOOP_MAX_OUTPUT_TOKENS": ("backend", "max_output_tokens"),
    "LOOP_REPETITION_DETECTION": ("backend", "repetition_detection"),
    "LOOP_REPETITION_PENALTY": ("backend", "repetition_penalty"),
    "LOOP_REPETITION_MIN_PATTERN_SIZE": ("backend", "repetition_min_pattern_size"),
    "LOOP_REPETITION_MAX_PATTERN_SIZE": ("backend", "repetition_max_pattern_size"),
    "LOOP_REPETITION_MIN_COUNT": ("backend", "repetition_min_count"),
    "OPENAI_HYPERPARAMETER_POLICY": ("backend", "hyperparameter_policy"),
    "USER_AGENT": ("web", "user_agent"),
    "LOOP_COMMAND_TIMEOUT": ("tools", "command_timeout"),
    "LOOP_COMMAND_HARD_TIMEOUT": ("tools", "command_hard_timeout"),
    "LOOP_COMMAND_JOB_WAIT_SECONDS": ("tools", "command_job_wait_seconds"),
    "LOOP_AGENT_NAME": ("app", "agent_name"),
    "LOOP_CONVERSATION_INPUT_MODE": ("conversation", "input_mode"),
    "LOOP_MODEL": ("app", "model"),
    "LOOP_TEMPERATURE": ("app", "temperature"),
    "LOOP_REASONING_EFFORT": ("app", "reasoning_effort"),
    "LOOP_STREAM": ("app", "stream"),
    "LOOP_DEBUG": ("app", "debug"),
    "LOOP_COMPACTION_THRESHOLD": ("app", "compaction_threshold"),
    "LOOP_PROMPT_ON_RECOVERABLE_ERROR": ("app", "prompt_on_recoverable_error"),
    "LOOP_MAX_AGENT_TURNS": ("app", "max_agent_turns"),
    "LOOP_LOG_LEVEL": ("logging", "level"),
    "LOOP_LOG_MAX_BYTES": ("logging", "max_bytes"),
    "LOOP_LOG_BACKUP_COUNT": ("logging", "backup_count"),
    "LOOP_TELEMETRY_QUEUE_CAPACITY": ("telemetry", "queue_capacity"),
    "LOOP_TELEMETRY_BATCH_SIZE": ("telemetry", "batch_size"),
    "LOOP_TELEMETRY_FLUSH_SECONDS": ("telemetry", "flush_seconds"),
    "LOOP_TELEMETRY_SHUTDOWN_TIMEOUT": ("telemetry", "shutdown_timeout"),
    "LOOP_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS": ("telemetry", "sqlite_busy_timeout_ms"),
}


@cache
def _configuration_adapter(section: str, name: str | None) -> TypeAdapter:
    """Reuse Pydantic field adapters across repeated configuration resolutions."""
    field = ApplicationSettings.model_fields[section]
    if name is not None:
        field = field.annotation.model_fields[name]
    return TypeAdapter(field.rebuild_annotation())


class ConfigurationParseError(ValueError):
    """Report a TOML syntax error without exposing the input text.

    Args:
        path (Path): Configuration file containing the error.
        error (TOMLKitError | UnicodeDecodeError): Original parsing or decoding diagnostic.
    """

    def __init__(self, path: Path, error: TOMLKitError | UnicodeDecodeError) -> None:
        self.path = path
        if isinstance(error, UnicodeDecodeError):
            position = f" at byte {error.start}"
        elif getattr(error, "line", None) is not None:
            position = f" at line {error.line}, column {error.col}"
        else:
            position = ""
        super().__init__(f"Invalid TOML in {path}{position} ({type(error).__name__}).")


class ConfigurationSectionError(ValueError):
    """Report a non-table section without exposing an unknown setting name.

    Args:
        section (str): Section encountered during configuration merging.
        source (str | None): File or environment source containing the invalid section.
    """

    def __init__(self, section: str, source: str | None = None) -> None:
        self.source = source
        safe_section = section if section in ApplicationSettings.model_fields else "<unknown>"
        suffix = f" in {source}" if source is not None else ""
        super().__init__(f"Configuration section '{safe_section}' must be a table{suffix}.")


class ConfigurationResetError(OSError):
    """Report a failed reset with the archived file's recovery location."""


_CONFIGURATION_ERROR_POLICY = {
    "trusted_types": (ConfigurationParseError, ConfigurationSectionError),
    "trusted_messages": frozenset(
        {
            "Workspace configuration cannot control app.yolo.",
            "Persistent app.yolo must be a TOML boolean.",
            "Configuration cannot contain both [loop] and [app].",
        }
    ),
    "redacted_prefixes": {
        "Persistent app.yolo source is untrusted:": (
            "Persistent app.yolo source is untrusted; verify the user configuration file."
        ),
    },
}


@dataclass(frozen=True)
class ConfigurationRecovery:
    """Identify the exact invalid file authorized for an archival reset.

    Args:
        scope (Literal["user", "workspace"]): Configuration scope.
        path (Path): Failed configuration file.
        identity (tuple[int, int]): Device and inode captured from the file.
        content (bytes): Exact invalid content observed before prompting.
    """

    scope: Literal["user", "workspace"]
    path: Path
    identity: tuple[int, int]
    content: bytes


class ConfigurationManager:
    """Manage complete user and sparse workspace TOML configuration documents.

    Args:
        path (Path | str): Complete user TOML configuration path.
        workspace_path (Path | str | None): Optional sparse workspace override path.
    """

    _path: Path
    _document: tomlkit.TOMLDocument
    _workspace_path: Path | None
    _workspace_source_path: Path | None
    _workspace_document: tomlkit.TOMLDocument
    _effective: ApplicationSettings | None
    _sources: dict[str, str]
    _environment: dict[str, str]
    _retry_environment: dict[str, str]
    _session_values: dict[str, Any]
    _defaults_only: bool
    _state_lock: RLock

    def __init__(self, path: Path | str, workspace_path: Path | str | None = None) -> None:
        self._path = Path(path).expanduser().absolute()
        self._workspace_source_path = (
            Path(workspace_path).expanduser().absolute() if workspace_path is not None else None
        )
        self._workspace_path = (
            self._workspace_source_path.resolve()
            if self._workspace_source_path is not None
            else None
        )
        self._document = tomlkit.document()
        self._workspace_document = tomlkit.document()
        self._effective = None
        self._sources = {}
        self._environment = {}
        self._retry_environment = {}
        self._session_values = {}
        self._defaults_only = False
        self._state_lock = RLock()

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
        with self._state_lock:
            if self._defaults_only:
                try:
                    self.path.lstat()
                except FileNotFoundError:
                    self._require_durable_writes()
                return self.path
            with self._lock_for("user"):
                try:
                    self.path.lstat()
                except FileNotFoundError:
                    self._document = self._default_document()
                    self._save_document(self.path, self._document)
                else:
                    return self.path
        return self.path

    def load(self, environment: Mapping[str, str] | None = None) -> ApplicationSettings:
        """Load and validate effective settings.

        Args:
            environment (Mapping[str, str] | None): Environment override source. Defaults to the
                current process environment.

        Returns:
            ApplicationSettings: Immutable settings resolved from environment, TOML, and defaults.

        Raises:
            ValueError: If an untrusted source tries to control YOLO or activation cannot be
                verified.
        """
        with self._state_lock:
            if environment is None:
                environment = os.environ
            candidate_environment = dict(environment)
            self._retry_environment = candidate_environment
            candidate_document = self._read_document()
            candidate_workspace_document = self._read_workspace_document()
            effective, sources = self._resolved(
                candidate_document, candidate_workspace_document, candidate_environment
            )
            self._document = candidate_document
            self._workspace_document = candidate_workspace_document
            self._environment = candidate_environment
            self._effective = effective
            self._sources = sources
            self._defaults_only = False
            return effective

    def use_defaults(self) -> ApplicationSettings:
        """Use built-in settings for this process while protecting saved configuration.

        Returns:
            ApplicationSettings: Built-in settings with persistent writes disabled until reload.
        """
        with self._state_lock:
            self._document = self._default_document()
            self._workspace_document = tomlkit.document()
            self._environment = {}
            self._session_values.clear()
            self._defaults_only = True
            return self._resolve()

    def recovery_candidate(self, error: Exception) -> ConfigurationRecovery | None:
        """Identify a safely resettable file that independently reproduces the failure.

        Args:
            error (Exception): Failure from loading configuration.

        Returns:
            ConfigurationRecovery | None: Verified file snapshot, or None when reset is unsafe.
        """
        if isinstance(error, ConfigurationParseError):
            source = str(error.path)
        elif isinstance(error, ConfigurationSectionError):
            source = error.source
        elif isinstance(error, ValidationError) and error.title.startswith("Configuration in "):
            source = error.title.removeprefix("Configuration in ")
        else:
            return None
        for scope, path in (("user", self.path), ("workspace", self._workspace_path)):
            if path is None or str(path) != source:
                continue
            try:
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
                    return None
                identity = (before.st_dev, before.st_ino)
                descriptor = open_selected_file(path.parent, path, identity)
                with os.fdopen(descriptor, "rb") as stream:
                    content = stream.read()
                try:
                    document = self._parse_document(path, content)
                    self._validate_loaded_fields(document.unwrap(), str(path))
                except (
                    ConfigurationParseError,
                    ConfigurationSectionError,
                    ValidationError,
                ) as found:
                    same = type(found) is type(error) and str(found) == str(error)
                    if isinstance(found, ValidationError) and isinstance(error, ValidationError):
                        same = found.errors(include_url=False) == error.errors(include_url=False)
                    if same:
                        return ConfigurationRecovery(scope, path, identity, content)
                except ValueError:
                    return None
            except OSError:
                return None
        return None

    def reset_invalid_file(self, candidate: ConfigurationRecovery) -> Path:
        """Archive a verified invalid file and create defaults without overwriting newer data.

        Args:
            candidate (ConfigurationRecovery): File snapshot selected for recovery.

        Returns:
            Path: Archived configuration file.

        Raises:
            OSError: If the file changed or archival or replacement fails.
            ValueError: If the candidate does not belong to this manager.
        """
        expected = self.path if candidate.scope == "user" else self._workspace_path
        if candidate.scope not in {"user", "workspace"} or candidate.path != expected:
            raise ValueError("Configuration recovery candidate does not belong to this manager.")
        with self._state_lock, self._lock_for(candidate.scope):
            directory = Path(
                tempfile.mkdtemp(
                    prefix=f".{candidate.path.name}.backup-", dir=candidate.path.parent
                )
            )
            backup = directory / candidate.path.name
            candidate.path.rename(backup)
            try:
                info = backup.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or (info.st_dev, info.st_ino) != candidate.identity
                    or backup.read_bytes() != candidate.content
                ):
                    raise OSError("Configuration changed during recovery.")
            except OSError as error:
                try:
                    os.link(backup, candidate.path, follow_symlinks=False)
                except OSError:
                    pass
                raise ConfigurationResetError(
                    f"Configuration could not be verified; preserved at {backup}."
                ) from error
            if candidate.scope == "user":
                temporary = directory / "defaults.toml"
                try:
                    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        stream.write(tomlkit.dumps(self._default_document()))
                    try:
                        os.link(temporary, candidate.path)
                    except FileExistsError:
                        pass
                    finally:
                        temporary.unlink()
                except OSError as error:
                    raise ConfigurationResetError(
                        f"Configuration reset failed; preserved at {backup}."
                    ) from error
            return backup

    @staticmethod
    async def recover_reset_failure_async(
        interaction: Interaction,
        error: OSError,
        continue_label: str,
    ) -> str:
        """Report a failed reset and require another explicit recovery decision.

        Args:
            interaction (Interaction): Interface receiving the failure and prompt.
            error (OSError): Failure while archiving or replacing the file.
            continue_label (str): Description of the available session fallback.

        Returns:
            str: Retry or continue, as explicitly selected.

        Raises:
            ShutdownRequested: If the user halts or input is invalid.
            ProblemException: If asynchronous interaction is unavailable.
        """
        problem = Problem(
            code="configuration.reset_failed",
            title="Configuration reset failed",
            detail=safe_error_detail(
                error, ApplicationSettings, trusted_types=(ConfigurationResetError,)
            ),
            operation="configuration.reset",
        )
        log_problem(_LOGGER, problem, error)
        interaction.report(problem)
        return await prompt_recovery_async(
            interaction,
            "Configuration recovery:",
            operation="configuration recovery",
            continue_label=continue_label,
        )

    async def load_interactively_async(self, interaction: Interaction) -> ApplicationSettings:
        """Load settings and offer recovery when the saved configuration is invalid.

        Args:
            interaction (Interaction): User interface for reporting and recovery choices.

        Returns:
            ApplicationSettings: Loaded settings or explicitly selected session defaults.

        Raises:
            ShutdownRequested: If the user declines to continue.
        """
        while True:
            try:
                self.initialize()
                return self.load()
            except (OSError, ValueError) as error:
                problem = Problem(
                    code="configuration.startup_failed",
                    title="Configuration loading failed",
                    detail=safe_error_detail(
                        error,
                        ApplicationSettings,
                        **_CONFIGURATION_ERROR_POLICY,
                    ),
                    operation="startup",
                )
                log_problem(_LOGGER, problem, error)
                interaction.report(problem)
                defaults = ApplicationSettings()
                candidate = self.recovery_candidate(error)
                choice = await prompt_recovery_async(
                    interaction,
                    "Configuration recovery:",
                    operation="configuration recovery",
                    continue_label=(
                        f"Use defaults (backend {defaults.backend.base_url}, model "
                        f"{defaults.backend.default_model}); ignore files and environment"
                    ),
                    reset_label=(
                        f"Archive {candidate.path} and reset it to defaults" if candidate else None
                    ),
                )
                if choice == "retry":
                    continue
                if choice == "reset" and candidate is not None:
                    try:
                        backup = self.reset_invalid_file(candidate)
                    except OSError as failure:
                        choice = await self.recover_reset_failure_async(
                            interaction, failure, "Use built-in defaults for this session"
                        )
                        if choice == "retry":
                            continue
                    else:
                        interaction.info(f"Archived invalid configuration at {backup}.")
                        continue
                settings = self.use_defaults()
                interaction.warning(
                    "Using built-in defaults, including the default backend, with saved "
                    "configuration writes disabled until a successful reload."
                )
                return settings

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
            environment (Mapping[str, str] | None): Replacement environment override source. When
                omitted, retains the environment captured by the previous load.

        Returns:
            ApplicationSettings: Fresh effective settings.

        Raises:
            ValueError: If the reloaded configuration has ambiguous application sections or an
                untrusted YOLO activation.
        """
        source = (
            (self._retry_environment if self._defaults_only else self._environment)
            if environment is None
            else environment
        )
        return self.load(source)

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
                configured_capabilities = field.json_schema_extra or {}
                required_capabilities = frozenset(
                    CommandCapability(value)
                    for value in configured_capabilities.get("required_capabilities", ())
                )
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
                        required_capabilities=(
                            required_capabilities or frozenset({CommandCapability.SESSION_IDLE})
                        ),
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
                ``"default"``. YOLO reports user scope for its saved value.
        """
        self._split_path(dotted_path)
        return self._sources.get(dotted_path, "default")

    @property
    def saved_yolo(self) -> bool | None:
        """Return the trusted user setting independently of a session override.

        Returns:
            bool | None: Saved activation, or None when the file was bypassed during recovery.
        """
        if self._defaults_only:
            return None
        table = self._document.get("app")
        return table.unwrap().get("yolo", False) is True if isinstance(table, Mapping) else False

    def set(
        self,
        dotted_path: str,
        value: Any,
        *,
        scope: Literal["user", "workspace"] = "workspace",
        before_save: Callable[[], None] | None = None,
    ) -> ApplicationSettings:
        """Set one TOML value and return the validated effective configuration.

        Args:
            dotted_path (str): Dot-separated configuration field path.
            value (Any): Replacement value stored in the TOML document.
            scope (Literal["user", "workspace"]): Durable destination. Defaults to workspace.
            before_save (Callable[[], None] | None): Optional trusted validation while the
                destination lock is held, immediately before publication.

        Returns:
            ApplicationSettings: Validated configuration after the edit.

        Raises:
            ValueError: If durable writes are suspended during recovery, YOLO is assigned to
                workspace scope, given an ambiguous value, or enabled from an untrusted destination.
        """
        self._require_durable_writes()
        with self._state_lock, self._lock_for(scope):
            self._require_durable_writes()
            if dotted_path == "app.yolo" and scope != "user":
                raise ValueError("app.yolo can only be saved for the user or this session.")
            if dotted_path == "app.yolo":
                value = self._normalize_yolo_value(value)
                if value:
                    self._require_yolo_destination()
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
                if before_save is not None:
                    before_save()
                self._require_durable_writes()
                self._save_scope(scope)
            except BaseException:
                if not self._defaults_only:
                    if scope == "workspace" and self._workspace_path is not None:
                        self._workspace_document = tomlkit.parse(original)
                    else:
                        self._document = tomlkit.parse(original)
                    self._resolve()
                raise
            return self._resolve()

    def set_session(self, dotted_path: str, value: Any) -> ApplicationSettings:
        """Set one in-memory override without changing the configuration file.

        Args:
            dotted_path (str): Dot-separated configuration field path.
            value (Any): Replacement value for this process only.

        Returns:
            ApplicationSettings: Validated effective configuration after the edit.

        Raises:
            ValueError: If YOLO is given a value other than true or false.
        """
        self._split_path(dotted_path)
        if dotted_path == "app.yolo":
            value = self._normalize_yolo_value(value)
        previous = self._session_values.copy()
        self._session_values[dotted_path] = value
        try:
            return self._resolve()
        except Exception:
            self._session_values = previous
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

        Raises:
            ValueError: If a durable reset is attempted while recovery defaults are active.
        """
        if scope == "session":
            self._session_values = {
                entry.path: self._default_value(entry.path) for entry in self.entries
            }
            return self._resolve()
        self._require_durable_writes()
        if scope == "workspace":
            with self._state_lock, self._lock_for(scope):
                self._require_durable_writes()
                self._workspace_document = tomlkit.document()
                self._save_scope("workspace")
            return self._resolve()
        with self._state_lock, self._lock_for(scope):
            self._require_durable_writes()
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
            ValueError: If durable writes are suspended or the field is not present.
        """
        if scope == "user":
            return self.set(dotted_path, self._default_value(dotted_path), scope="user")
        self._require_durable_writes()
        if dotted_path == "app.yolo":
            raise ValueError("Workspace configuration cannot control app.yolo.")
        with self._state_lock, self._lock_for(scope):
            self._require_durable_writes()
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

        Raises:
            ValueError: If recovery defaults are active and durable writes are suspended.
        """
        self._require_durable_writes()
        with self._state_lock, self._lock_for("user"):
            self._require_durable_writes()
            return self._save_document(self.path, self._document)

    def _require_durable_writes(self) -> None:
        """Keep an explicitly bypassed file intact until a successful reload."""
        if self._defaults_only:
            raise ValueError("Repair and reload configuration before saving changes.")

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
            os.fchmod(output.fileno(), constants.PRIVATE_FILE_MODE)
        temporary.replace(path)
        return path

    def _read_document(self) -> tomlkit.TOMLDocument:
        """Return the current comment-preserving TOML document, or a default if absent."""
        raw = self._read_optional_file(self.path)
        if raw is None:
            return self._default_document()
        document = self._parse_document(self.path, raw)
        table = document.get("app")
        if isinstance(table, Mapping) and "yolo" in table:
            value = document.unwrap()["app"]["yolo"]
            if not isinstance(value, bool):
                raise ValueError("Persistent app.yolo must be a TOML boolean.")
            if value:
                self._verify_trusted_yolo_source(raw)
        return document

    @staticmethod
    def _normalize_legacy_app_section(
        document: tomlkit.TOMLDocument,
    ) -> tomlkit.TOMLDocument:
        """Read saved version-one ``[loop]`` settings under the current ``[app]`` name."""
        if isinstance(document.get("loop"), Mapping):
            if "app" in document:
                raise ValueError("Configuration cannot contain both [loop] and [app].")
            document["app"] = document.pop("loop")
        return document

    @staticmethod
    def _normalize_yolo_value(value: Any) -> bool:
        """Accept only an explicit boolean for trusted YOLO configuration."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError("app.yolo requires true or false.")

    def _verify_trusted_yolo_source(self, raw: bytes) -> None:
        """Reject a persistent activation whose owner or inode cannot be verified."""
        path = self.path
        try:
            self._require_yolo_destination()
            before = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != constants.PRIVATE_FILE_MODE
            ):
                raise ValueError("Persistent app.yolo requires an owner-controlled user config.")
            descriptor = open_selected_file(path.parent, path, (before.st_dev, before.st_ino))
            with os.fdopen(descriptor, "rb") as stream:
                if has_extended_access_acl(stream.fileno()):
                    raise ValueError("Persistent app.yolo config has an extended access ACL.")
                content = stream.read()
            after = path.lstat()
            if content != raw or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("Persistent app.yolo changed while its source was verified.")
        except (OSError, ValueError) as error:
            raise ValueError(f"Persistent app.yolo source is untrusted: {error}") from error

    def _require_yolo_destination(self) -> None:
        """Keep a persistent activation in real, private storage outside the workspace."""
        if self._workspace_path is not None and self.path.resolve().is_relative_to(
            self._workspace_path.parent.parent
        ):
            raise ValueError("Persistent app.yolo config cannot be inside the workspace.")
        ensure_private_directory(self.path.parent, create=False)
        if self.path.is_symlink():
            raise ValueError("Persistent app.yolo cannot use an aliased user config.")

    def _read_workspace_document(self) -> tomlkit.TOMLDocument:
        """Return the sparse workspace document or an empty document."""
        if self._workspace_path is None:
            return tomlkit.document()
        raw = self._read_optional_file(self._workspace_path)
        if raw is None:
            if (
                self._workspace_source_path != self._workspace_path
                and self._read_optional_file(self._workspace_source_path) is not None
            ):
                raise OSError(
                    errno.ESTALE,
                    "Workspace configuration changed during loading",
                    str(self._workspace_source_path),
                )
            return tomlkit.document()
        return self._parse_document(self._workspace_path, raw)

    @staticmethod
    def _read_optional_file(path: Path) -> bytes | None:
        """Read a file, distinguishing an absent path from a broken link or read failure."""
        try:
            return path.read_bytes()
        except FileNotFoundError:
            try:
                path.lstat()
            except FileNotFoundError:
                for parent in path.parents:
                    try:
                        details = parent.lstat()
                    except FileNotFoundError:
                        continue
                    if stat.S_ISLNK(details.st_mode):
                        parent.stat()
                return None
            raise

    @staticmethod
    def _parse_document(path: Path, content: bytes) -> tomlkit.TOMLDocument:
        """Parse one named file without exposing source snippets in syntax diagnostics."""
        try:
            return ConfigurationManager._normalize_legacy_app_section(
                tomlkit.parse(content.decode("utf-8"))
            )
        except (TOMLKitError, UnicodeDecodeError) as error:
            raise ConfigurationParseError(path, error) from None

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

    def _resolve(self) -> ApplicationSettings:
        """Resolve configured, environment, and session values into one snapshot."""
        effective, sources = self._resolved(
            self._document, self._workspace_document, self._environment
        )
        self._effective = effective
        self._sources = sources
        return effective

    def _resolved(
        self,
        document: tomlkit.TOMLDocument,
        workspace_document: tomlkit.TOMLDocument,
        environment: Mapping[str, str],
    ) -> tuple[ApplicationSettings, dict[str, str]]:
        """Return an effective snapshot and provenance without changing manager state."""
        values = document.unwrap()
        self._validate_loaded_fields(values, str(self._path))

        def section_values(section: str) -> dict:
            """Return a validated section table."""
            return values.setdefault(section, {})

        sources = self._configured_sources(
            values, "user" if self._workspace_path is not None else "workspace"
        )
        if "app.yolo" in sources:
            sources["app.yolo"] = "user"
        workspace_values = workspace_document.unwrap()
        self._validate_loaded_fields(workspace_values, str(self._workspace_path))
        if isinstance(workspace_values.get("app"), Mapping) and "yolo" in workspace_values["app"]:
            raise ValueError("Workspace configuration cannot control app.yolo.")
        for section, table in workspace_values.items():
            if section == "config_version":
                continue
            section_values(section).update(table)
        sources.update(self._configured_sources(workspace_values, "workspace"))
        backend_environment = {}
        for variable, (section, field) in _ENVIRONMENT_FIELDS.items():
            value = environment.get(variable)
            if value is not None:
                self._validate_loaded_fields(
                    {section: {field: value}}, f"environment variable {variable}"
                )
                section_values(section)[field] = value
                sources[f"{section}.{field}"] = "environment"
                if section == "backend":
                    backend_environment[field] = value
        if backend_environment:
            self._validate_loaded_fields({"backend": backend_environment}, "environment variables")
        for path, value in self._session_values.items():
            section, field = self._split_path(path)
            section_values(section)[field] = value
            sources[path] = "session"
        return ApplicationSettings.model_validate(values), sources

    @staticmethod
    def _validate_loaded_fields(values: dict, source: str) -> None:
        """Validate each supplied field before higher-priority layers can hide it."""
        for section, table in values.items():
            section_field = ApplicationSettings.model_fields.get(section)
            if section_field is None:
                issues = [{"type": "extra_forbidden", "loc": (section,), "input": table}]
                raise ValidationError.from_exception_data(f"Configuration in {source}", issues)
            if section != "config_version" and not isinstance(table, dict):
                raise ConfigurationSectionError(section, source)
            entries = ((None, table),) if section == "config_version" else table.items()
            for name, value in entries:
                if name is not None and name not in section_field.annotation.model_fields:
                    issues = [{"type": "extra_forbidden", "loc": (section, name), "input": value}]
                    raise ValidationError.from_exception_data(f"Configuration in {source}", issues)
                try:
                    _configuration_adapter(section, name).validate_python(value)
                except ValidationError as error:
                    issues = error.errors(include_url=False)
                    for issue in issues:
                        prefix = (section,) if name is None else (section, name)
                        issue["loc"] = (*prefix, *issue["loc"])
                    raise ValidationError.from_exception_data(
                        f"Configuration in {source}", issues
                    ) from None
            if section == "backend" and len(table) > 1:
                try:
                    BackendSettings.model_validate(table, context={"partial_source": True})
                except ValidationError as error:
                    issues = error.errors(include_url=False)
                    for issue in issues:
                        issue["type"] = PydanticCustomError(
                            issue["type"],
                            issue["msg"],
                            issue.get("ctx"),
                        )
                        issue["loc"] = (section, *issue["loc"])
                    raise ValidationError.from_exception_data(
                        f"Configuration in {source}", issues
                    ) from None

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
            tomlkit.comment("Workspace configuration. Environment variables override values here.")
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
