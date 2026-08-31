"""Expose application configuration through interactive commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionValue
from .manager import ApplicationSettings, ConfigurationManager


class ConfigurationCommands:
    """Expose one configuration manager through interactive commands.

    Args:
        configuration (ConfigurationManager): Configuration owner for the active application.
        apply (Callable[[str, ApplicationSettings], str]): Applies a validated effective setting
            snapshot and returns its user-facing effect status.
    """

    _configuration: ConfigurationManager
    _apply: Callable[[str, ApplicationSettings], str]

    def __init__(
        self,
        configuration: ConfigurationManager,
        apply: Callable[[str, ApplicationSettings], str],
    ) -> None:
        self._configuration = configuration
        self._apply = apply

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the configuration command registration.

        Returns:
            tuple[CommandRegistration, ...]: The ``/config`` command.
        """
        entry_completion = CommandCompletion(provider=self._public_entry_values)
        scope_completion = CommandCompletion(
            values=(
                CompletionValue("scope=workspace", "Save for this workspace."),
                CompletionValue("scope=session", "Apply only for this session."),
            )
        )
        set_entry_completion = CommandCompletion(
            provider=self._public_entry_values,
            next=CommandCompletion(next=scope_completion),
        )
        secret_completion = CommandCompletion(
            provider=self._secret_entry_values,
            next=scope_completion,
        )
        reset_completion = CommandCompletion(
            values=scope_completion.values,
            provider=self._all_entry_values,
            next=scope_completion,
        )
        return (
            CommandRegistration(
                self.config,
                name="config",
                completion=CommandCompletion(
                    values=(
                        CompletionValue("get", "Display one configuration entry."),
                        CompletionValue("set", "Update one configuration entry."),
                        CompletionValue("secret", "Update one secret configuration entry."),
                        CompletionValue("reset", "Reset one entry or every entry."),
                    ),
                    children={
                        "get": entry_completion,
                        "set": set_entry_completion,
                        "secret": secret_completion,
                        "reset": reset_completion,
                    },
                ),
            ),
        )

    def _public_entry_values(self) -> tuple[CompletionValue, ...]:
        """Return dynamically current non-secret configuration paths for completion."""
        return tuple(
            CompletionValue(entry.path, entry.description)
            for entry in self._configuration.entries
            if not entry.secret
        )

    def _all_entry_values(self) -> tuple[CompletionValue, ...]:
        """Return dynamically current configuration paths for reset completion."""
        return tuple(
            CompletionValue(entry.path, entry.description) for entry in self._configuration.entries
        )

    def _secret_entry_values(self) -> tuple[CompletionValue, ...]:
        """Return dynamically current secret configuration paths for completion."""
        return tuple(
            CompletionValue(entry.path, entry.description)
            for entry in self._configuration.entries
            if entry.secret
        )

    def config(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["get", "set", "secret", "reset"] | None,
            Field(description="Operation to perform, or omit to display all settings."),
        ] = None,
        path: Annotated[
            str | None,
            Field(description="Setting path such as backend.temperature."),
        ] = None,
        value: Annotated[
            str | None,
            Field(description="New setting value for the set operation."),
        ] = None,
        scope: Annotated[
            Literal["session", "workspace"] | None,
            Field(description="Override scope, or omit it to choose interactively."),
        ] = None,
    ) -> None:
        """Show, update, or reset the application configuration."""
        if action is None:
            self._show(context, None)
            return
        if action == "get":
            if path is None:
                raise CommandArgumentError("The get operation requires a setting path.")
            self._show(context, path)
            return
        if action == "secret":
            if path is None:
                raise CommandArgumentError("The secret operation requires a setting path.")
            if value is not None:
                raise CommandArgumentError("The secret operation accepts its value only by prompt.")
            if not self._is_secret(path):
                raise CommandArgumentError(f"Configuration field '{path}' is not a secret.")
            secret_value = context.interaction.prompt(f"Enter {path}:", secret=True)
            if secret_value is False:
                context.interaction.info(f"Secret {path} was not changed.")
                return
            selected_scope = self._select_scope(context, scope)
            if selected_scope is not None:
                self._set(context, path, str(secret_value), selected_scope, allow_secret=True)
            return
        if action == "reset" and path is None:
            selected_scope = self._select_scope(context, scope)
            if selected_scope is not None and self._confirm_reset(
                context, path=None, scope=selected_scope, explicit_scope=scope is not None
            ):
                self._reset_all(context, selected_scope)
            return
        if action == "reset":
            selected_scope = self._select_scope(context, scope)
            if selected_scope is not None and self._confirm_reset(
                context, path=path, scope=selected_scope, explicit_scope=scope is not None
            ):
                self._reset(context, path, selected_scope)
            return
        if path is None:
            raise CommandArgumentError("The set operation requires a setting path.")
        if value is None:
            raise CommandArgumentError("The set operation requires a value.")
        selected_scope = self._select_scope(context, scope)
        if selected_scope is not None:
            self._set(context, path, value, selected_scope)

    @staticmethod
    def _select_scope(
        context: CommandContext,
        scope: Literal["session", "workspace"] | None,
    ) -> Literal["session", "workspace"] | None:
        """Return an explicit or letter-selected configuration scope.

        Args:
            context (CommandContext): Interaction used to select a scope.
            scope (Literal["session", "workspace"] | None): Explicit scope, when supplied.
        Returns:
            Literal["session", "workspace"] | None: Selected scope, or ``None`` when cancelled.
        """
        if scope is not None:
            return scope
        selected = context.interaction.prompt(
            "Choose configuration scope:",
            choices={
                "workspace": "Save for this workspace",
                "session": "This session only",
                "cancel": "Cancel",
            },
            index={"workspace": "w", "session": "s", "cancel": "c"},
        )
        return selected if selected in {"session", "workspace"} else None

    @staticmethod
    def _confirm_reset(
        context: CommandContext,
        path: str | None,
        scope: Literal["session", "workspace"],
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
        return context.interaction.confirm(f"Reset {target} for {scope}?", default=False)

    def _reset_all(self, context: CommandContext, scope: Literal["session", "workspace"]) -> None:
        """Reset every setting in one selected scope to its built-in default."""
        paths = tuple(entry.path for entry in self._configuration.entries)
        settings = self._configuration.reset_all(scope=scope)
        statuses = {self._apply(path, settings) for path in paths}
        detail = ", ".join(sorted(statuses)) if statuses else "no overrides existed"
        context.interaction.info(f"Reset all configuration entries for {scope}: {detail}.")

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

    def _set(
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
            settings = (
                self._configuration.set_session(path, value)
                if scope == "session"
                else self._configuration.set(path, value)
            )
            status = self._apply(path, settings)
        except (ValueError, ValidationError) as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Updated {path} for {scope}: {status}")

    def _is_secret(self, path: str) -> bool:
        """Return whether a known configuration path is secret."""
        entry = next((entry for entry in self._configuration.entries if entry.path == path), None)
        if entry is None:
            raise CommandArgumentError(f"Unknown configuration field '{path}'.")
        return entry.secret

    def _reset(self, context: CommandContext, path: str, scope: str) -> None:
        """Reset one selected-scope value and apply the resulting setting."""
        try:
            settings = (
                self._configuration.reset_session(path)
                if scope == "session"
                else self._configuration.reset(path)
            )
            status = self._apply(path, settings)
        except (ValueError, ValidationError) as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Reset {path} for {scope}: {status}")
