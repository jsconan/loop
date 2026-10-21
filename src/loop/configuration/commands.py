"""Expose application configuration through interactive commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import (
    CommandCompletion,
    CompletionProviderRegistration,
    CompletionValue,
    SchemaCompletionProviderRegistration,
)
from ..interaction import require_async_interaction
from ..models import CommandCapability
from .manager import ApplicationSettings, ConfigurationManager

Result = TypeVar("Result")
Effect = TypeVar("Effect")


class ConfigurationGetArguments(BaseModel):
    """Describe the path accepted by configuration get.

    Args:
        path (str): Public configuration path to display.
    """

    path: Annotated[
        str,
        Field(description="Public configuration path to display."),
        CommandCompletion(provider="configuration_read_entries"),
    ]


class ConfigurationSetArguments(BaseModel):
    """Describe arguments accepted by configuration set.

    Args:
        path (str): Public configuration path to update.
        value (str): New setting value.
        scope (Literal["session", "user", "workspace"] | None): Override scope, or ``None`` to
            choose interactively.
    """

    path: Annotated[
        str,
        Field(description="Public configuration path to update."),
        CommandCompletion(provider="configuration_public_entries"),
    ]
    value: Annotated[str, Field(description="New setting value.")]
    scope: Annotated[
        Literal["session", "user", "workspace"] | None,
        Field(description="Override scope, or omit it to choose interactively."),
    ] = None


class ConfigurationSecretArguments(BaseModel):
    """Describe arguments accepted by configuration secret.

    Args:
        path (str): Secret configuration path to update.
        scope (Literal["session", "user", "workspace"] | None): Override scope, or ``None`` to
            choose interactively.
    """

    path: Annotated[
        str,
        Field(description="Secret configuration path to update."),
        CommandCompletion(provider="configuration_secret_entries"),
    ]
    scope: Annotated[
        Literal["session", "user", "workspace"] | None,
        Field(description="Override scope, or omit it to choose interactively."),
    ] = None


class ConfigurationResetArguments(BaseModel):
    """Describe arguments accepted by configuration reset.

    Args:
        path (str | None): Configuration path, or ``None`` to reset every entry.
        scope (Literal["session", "user", "workspace"] | None): Override scope, or ``None`` to
            choose interactively.
    """

    path: Annotated[
        str | None,
        Field(description="Configuration path, or omit to reset every entry."),
        CommandCompletion(provider="configuration_all_entries"),
    ] = None
    scope: Annotated[
        Literal["session", "user", "workspace"] | None,
        Field(description="Override scope, or omit it to choose interactively."),
    ] = None


class ConfigurationCommands:
    """Expose one configuration manager through interactive commands.

    Args:
        configuration (ConfigurationManager): Configuration owner for the active application.
        apply (Callable[[str, ApplicationSettings], str]): Applies a validated effective setting
            snapshot and returns its user-facing effect status.
        apply_many (Callable[[tuple[str, ...], ApplicationSettings], Mapping[str, str]] | None):
            Applies multiple changed settings as one reload operation. Defaults to applying each
            setting separately.
    """

    _configuration: ConfigurationManager
    _apply: Callable[[str, ApplicationSettings], str]
    _apply_many: Callable[[tuple[str, ...], ApplicationSettings], Mapping[str, str]] | None

    def __init__(
        self,
        configuration: ConfigurationManager,
        apply: Callable[[str, ApplicationSettings], str],
        apply_many: Callable[[tuple[str, ...], ApplicationSettings], Mapping[str, str]]
        | None = None,
    ) -> None:
        self._configuration = configuration
        self._apply = apply
        self._apply_many = apply_many

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the configuration command registration.

        Returns:
            tuple[CommandRegistration, ...]: The ``/config`` command.
        """
        return (
            CommandRegistration(
                self.config,
                name="config",
                capabilities=frozenset(
                    {CommandCapability.CONTEXT_READ, CommandCapability.CONTEXT_UPDATE}
                ),
                completion=CommandCompletion(
                    values=(
                        CompletionValue(
                            "get",
                            "Display one configuration entry.",
                            required_capabilities=frozenset({CommandCapability.CONTEXT_READ}),
                        ),
                        CompletionValue(
                            "set",
                            "Update a setting for subsequent runtime operations.",
                            required_capabilities=frozenset({CommandCapability.CONTEXT_UPDATE}),
                        ),
                        CompletionValue(
                            "secret",
                            "Update one secret configuration entry.",
                            required_capabilities=frozenset({CommandCapability.SESSION_IDLE}),
                        ),
                        CompletionValue(
                            "reset",
                            "Reset one entry or every entry.",
                            required_capabilities=frozenset({CommandCapability.SESSION_IDLE}),
                        ),
                        CompletionValue(
                            "reload",
                            "Reload configuration files from disk.",
                            required_capabilities=frozenset({CommandCapability.SESSION_IDLE}),
                        ),
                    ),
                    children={
                        "get": CommandCompletion(schema_provider="configuration_arguments"),
                        "set": CommandCompletion(schema_provider="configuration_arguments"),
                        "secret": CommandCompletion(schema_provider="configuration_arguments"),
                        "reset": CommandCompletion(schema_provider="configuration_arguments"),
                        "reload": CommandCompletion(schema_provider="configuration_arguments"),
                    },
                ),
            ),
        )

    def get_completion_providers(
        self,
    ) -> tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
        """Return dynamic configuration values and action-specific argument schemas.

        Returns:
            tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
                Current setting paths and configuration argument schemas.
        """
        return (
            CompletionProviderRegistration(
                "configuration_public_entries", self._public_entry_values
            ),
            CompletionProviderRegistration("configuration_read_entries", self._read_entry_values),
            CompletionProviderRegistration("configuration_all_entries", self._all_entry_values),
            CompletionProviderRegistration(
                "configuration_secret_entries", self._secret_entry_values
            ),
            SchemaCompletionProviderRegistration(
                "configuration_arguments", self._configuration_arguments
            ),
        )

    @staticmethod
    def _configuration_arguments(tokens: tuple[str, ...]) -> type[BaseModel] | None:
        """Return the argument schema for the selected configuration operation."""
        return {
            ("get",): ConfigurationGetArguments,
            ("set",): ConfigurationSetArguments,
            ("secret",): ConfigurationSecretArguments,
            ("reset",): ConfigurationResetArguments,
        }.get(tokens)

    def _public_entry_values(self) -> tuple[CompletionValue, ...]:
        """Return paths that are safe to update in the current execution context."""
        return tuple(
            CompletionValue(
                entry.path,
                entry.description,
                required_capabilities=entry.required_capabilities,
            )
            for entry in self._configuration.entries
            if not entry.secret
        )

    def _read_entry_values(self) -> tuple[CompletionValue, ...]:
        """Return non-secret configuration paths available for inspection during a run."""
        return tuple(
            CompletionValue(
                entry.path,
                entry.description,
                required_capabilities=frozenset({CommandCapability.CONTEXT_READ}),
            )
            for entry in self._configuration.entries
            if not entry.secret
        )

    def _all_entry_values(self) -> tuple[CompletionValue, ...]:
        """Return dynamically current configuration paths for reset completion."""
        return tuple(
            CompletionValue(
                entry.path,
                entry.description,
                required_capabilities=frozenset({CommandCapability.SESSION_IDLE}),
            )
            for entry in self._configuration.entries
        )

    def _secret_entry_values(self) -> tuple[CompletionValue, ...]:
        """Return dynamically current secret configuration paths for completion."""
        return tuple(
            CompletionValue(
                entry.path,
                entry.description,
                required_capabilities=frozenset({CommandCapability.SESSION_IDLE}),
            )
            for entry in self._configuration.entries
            if entry.secret
        )

    async def config(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["get", "set", "secret", "reset", "reload"] | None,
            Field(description="Operation to perform, or omit to display all settings."),
        ] = None,
        path: Annotated[
            str | None,
            Field(description="Setting path such as loop.temperature."),
        ] = None,
        value: Annotated[
            str | None,
            Field(description="New setting value for the set operation."),
        ] = None,
        scope: Annotated[
            Literal["session", "user", "workspace"] | None,
            Field(description="Override scope, or omit it to choose interactively."),
        ] = None,
    ) -> None:
        """Show, update, reset, or reload the application configuration.

        Args:
            context (CommandContext): Interaction and capability snapshot for this invocation.
            action (Literal["get", "set", "secret", "reset", "reload"] | None): Configuration
                operation, or ``None`` to show every setting.
            path (str | None): Setting path such as ``loop.temperature``.
            value (str | None): New setting value for the set operation.
            scope (Literal["session", "user", "workspace"] | None): Override scope, or ``None``
                to choose it interactively.

        Raises:
            CommandArgumentError: If the operation is unavailable or its arguments are invalid.
            ProblemException: If an interactive prompt requires unavailable asynchronous input.
            asyncio.CancelledError: If asynchronous prompt input is cancelled.
        """
        if CommandCapability.RUN_ACTIVE in (context.capabilities or frozenset()):
            if action in {"secret", "reset", "reload"}:
                raise CommandArgumentError(
                    f"'/config {action}' cannot run while the agent is working because it may "
                    "prompt or replace active runtime settings. Wait for the run to pause."
                )
            if action == "set" and path is not None and context.capabilities is not None:
                entry = next(
                    (item for item in self._configuration.entries if item.path == path), None
                )
                if entry is not None and not entry.required_capabilities <= context.capabilities:
                    raise CommandArgumentError(
                        f"Configuration field '{path}' cannot be changed while the agent is "
                        "working; its owner requires an idle session."
                    )
        if action is None:
            self._show(context, None)
            return
        if action == "get":
            if path is None:
                raise CommandArgumentError("The get operation requires a setting path.")
            self._show(context, path)
            return
        if action == "reload":
            if path is not None or value is not None or scope is not None:
                raise CommandArgumentError("The reload operation does not accept arguments.")
            await self._reload(context)
            return
        if action == "secret":
            await self._set_secret(context, path, value, scope)
            return
        if action == "reset" and path is None:
            selected_scope = await self._select_scope(context, scope)
            if selected_scope is not None and await self._confirm_reset(
                context,
                path=None,
                scope=selected_scope,
                explicit_scope=scope is not None,
            ):
                await self._reset_all(context, selected_scope)
            return
        if action == "reset":
            selected_scope = await self._select_scope(context, scope)
            if selected_scope is not None and await self._confirm_reset(
                context,
                path=path,
                scope=selected_scope,
                explicit_scope=scope is not None,
            ):
                await self._reset(context, path, selected_scope)
            return
        if path is None:
            raise CommandArgumentError("The set operation requires a setting path.")
        if value is None:
            raise CommandArgumentError("The set operation requires a value.")
        selected_scope = await self._select_scope(context, scope)
        if selected_scope is not None:
            await self._set(context, path, value, selected_scope)

    @staticmethod
    async def _select_scope(
        context: CommandContext,
        scope: Literal["session", "user", "workspace"] | None,
    ) -> Literal["session", "user", "workspace"] | None:
        """Return an explicit or letter-selected configuration scope.

        Args:
            context (CommandContext): Interaction used to select a scope.
            scope (Literal["session", "user", "workspace"] | None): Explicit scope, when supplied.

        Returns:
            Literal["session", "user", "workspace"] | None: Selected scope, or ``None`` when
                cancelled.

        Raises:
            ProblemException: If the interaction does not provide asynchronous input.
            asyncio.CancelledError: If asynchronous prompt input is cancelled.
        """
        if scope is not None:
            return scope
        selected = await require_async_interaction(
            context.interaction, "configuration scope selection"
        ).prompt_async(
            "Choose configuration scope:",
            choices={
                "user": "Save for current user",
                "workspace": "Save for current workspace",
                "session": "This session only",
                "cancel": "Cancel",
            },
            index={"user": "u", "workspace": "w", "session": "s", "cancel": "c"},
        )
        return selected if selected in {"session", "user", "workspace"} else None

    @staticmethod
    async def _confirm_reset(
        context: CommandContext,
        path: str | None,
        scope: Literal["session", "user", "workspace"],
        *,
        explicit_scope: bool,
    ) -> bool:
        """Confirm a reset when its scope was provided directly.

        The letter-based scope picker already acts as confirmation when the scope is
        omitted. An explicit scope needs a separate confirmation because it would
        otherwise make a destructive reset execute immediately.
        """
        if not explicit_scope:
            return True
        target = path if path is not None else "all configuration entries"
        return await require_async_interaction(
            context.interaction, "configuration reset confirmation"
        ).confirm_async(f"Reset {target} for {scope}?", default=False)

    async def _reset_all(
        self,
        context: CommandContext,
        scope: Literal["session", "user", "workspace"],
    ) -> None:
        """Reset every setting in one selected scope to its built-in default."""
        paths = tuple(entry.path for entry in self._configuration.entries)
        if scope == "session":
            settings = self._configuration.reset_all(scope=scope)
            statuses = {self._apply(path, settings) for path in paths}
        else:
            _, applied = await self._run_blocking_with_effect(
                context,
                self._configuration.reset_all,
                lambda settings: {self._apply(path, settings) for path in paths},
                scope=scope,
            )
            statuses = applied
        detail = ", ".join(sorted(statuses)) if statuses else "no overrides existed"
        context.interaction.info(f"Reset all configuration entries for {scope}: {detail}.")

    async def _reload(self, context: CommandContext) -> None:
        """Reload disk configuration and apply only effective changes."""
        previous = self._configuration.effective
        try:
            _, statuses = await self._run_blocking_with_effect(
                context,
                self._configuration.reload,
                lambda settings: self._apply_changes(previous, settings),
            )
        except (OSError, ValueError, ValidationError) as error:
            raise CommandArgumentError(str(error)) from error
        paths = tuple(statuses)
        if not paths:
            context.interaction.info("Configuration reloaded: no effective changes.")
            return
        grouped = {}
        for path in paths:
            grouped.setdefault(statuses[path], []).append(path)
        detail = "; ".join(f"{status}: {', '.join(paths)}" for status, paths in grouped.items())
        context.interaction.info(f"Configuration reloaded: {detail}.")

    def _apply_changes(
        self,
        previous: ApplicationSettings,
        current: ApplicationSettings,
    ) -> Mapping[str, str]:
        """Apply only runtime settings whose effective values changed."""
        paths = self._changed_paths(previous, current)
        if not paths:
            return {}
        if self._apply_many is not None:
            return self._apply_many(paths, current)
        return {path: self._apply(path, current) for path in paths}

    async def _run_blocking_with_effect(
        self,
        context: CommandContext,
        function: Callable[..., Result],
        apply: Callable[[Result], Effect],
        *args: object,
        **kwargs: object,
    ) -> tuple[Result, Effect]:
        """Settle sync persistence before applying its result on the owner, even on cancellation."""
        pending = asyncio.create_task(context.run_blocking(function, *args, **kwargs))
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError as cancellation:
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except BaseException:  # noqa: BLE001 - inspect the settled worker outcome below
                    break
            try:
                result = pending.result()
            except BaseException as error:
                cancellation.add_note(
                    f"Configuration persistence also stopped with {type(error).__name__}."
                )
                raise cancellation from error
            else:
                apply(result)
            raise
        return result, apply(result)

    @staticmethod
    def _changed_paths(
        previous: ApplicationSettings,
        current: ApplicationSettings,
    ) -> tuple[str, ...]:
        """Return effective paths whose values differ between two snapshots."""
        paths = []
        for section_name, section in ApplicationSettings.model_fields.items():
            if section_name == "config_version":
                continue
            for field_name in section.annotation.model_fields:
                if getattr(getattr(previous, section_name), field_name) != getattr(
                    getattr(current, section_name), field_name
                ):
                    paths.append(f"{section_name}.{field_name}")
        return tuple(paths)

    def _show(self, context: CommandContext, path: str | None) -> None:
        """Render one or all effective settings without exposing secrets."""
        entries = tuple(entry for entry in self._configuration.entries if not entry.secret)
        if path is not None:
            entry = next(
                (entry for entry in self._configuration.entries if entry.path == path), None
            )
            if entry is None:
                raise CommandArgumentError(f"Unknown configuration field '{path}'.")
            if entry.secret:
                raise CommandArgumentError(
                    "Secret fields are not displayed with 'get'. Use 'secret' to update them."
                )
            entries = (entry,)
        context.interaction.table(
            entries,
            title="Application configuration:",
            columns=("path", "value", "source", "description"),
        )

    async def _set(
        self,
        context: CommandContext,
        path: str,
        value: str,
        scope: str,
        *,
        allow_secret: bool = False,
    ) -> None:
        """Validate, store, and apply one setting."""
        if self._is_secret(path) and not allow_secret:
            raise CommandArgumentError(
                "Use the secret operation to update secret configuration fields."
            )
        try:
            if scope == "session":
                settings = self._configuration.set_session(path, value)
                status = self._apply(path, settings)
            else:
                _, status = await self._run_blocking_with_effect(
                    context,
                    self._configuration.set,
                    lambda settings: self._apply(path, settings),
                    path,
                    value,
                    scope=scope,
                )
        except (ValueError, ValidationError) as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Updated {path} for {scope}: {status}")

    async def _set_secret(
        self,
        context: CommandContext,
        path: str | None,
        value: str | None,
        scope: Literal["session", "user", "workspace"] | None,
    ) -> None:
        """Prompt for and persist one secret configuration value."""
        if path is None:
            raise CommandArgumentError("The secret operation requires a setting path.")
        if value is not None:
            raise CommandArgumentError("The secret operation accepts its value only by prompt.")
        if not self._is_secret(path):
            raise CommandArgumentError(f"Configuration field '{path}' is not a secret.")
        secret_value = await require_async_interaction(
            context.interaction, "secret configuration input"
        ).prompt_async(f"Enter {path}:", secret=True)
        if secret_value is False:
            context.interaction.info(f"Secret {path} was not changed.")
            return
        selected_scope = await self._select_scope(context, scope)
        if selected_scope is not None:
            await self._set(context, path, str(secret_value), selected_scope, allow_secret=True)

    def _is_secret(self, path: str) -> bool:
        """Return whether a known configuration path is secret."""
        entry = next((entry for entry in self._configuration.entries if entry.path == path), None)
        if entry is None:
            raise CommandArgumentError(f"Unknown configuration field '{path}'.")
        return entry.secret

    async def _reset(self, context: CommandContext, path: str, scope: str) -> None:
        """Reset one selected-scope value and apply the resulting setting."""
        try:
            if scope == "session":
                settings = self._configuration.reset_session(path)
                status = self._apply(path, settings)
            else:
                _, status = await self._run_blocking_with_effect(
                    context,
                    self._configuration.reset,
                    lambda settings: self._apply(path, settings),
                    path,
                    scope=scope,
                )
        except (ValueError, ValidationError) as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Reset {path} for {scope}: {status}")
