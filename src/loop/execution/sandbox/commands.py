"""Expose Linux installation management only on Linux hosts."""

import platform
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field

from ...commands import CommandArgumentError, CommandContext, CommandRegistration
from ...completion import CommandCompletion, CompletionValue
from ...errors import Problem, ProblemException
from ...telemetry import telemetry_audit, telemetry_span


def _failure_lines(error: Exception) -> tuple[str, ...]:
    """Return each distinct owner failure as one bounded line."""
    if isinstance(error, ExceptionGroup):
        return tuple(
            dict.fromkeys(line for cause in error.exceptions for line in _failure_lines(cause))
        )
    lines = str(error).splitlines()
    return ((lines[0].strip()[:160] if lines else "") or "Native management failed.",)


class SandboxCommands:
    """Own native management command schemas, completion and human review.

    Args:
        main_app (bool): Permit fresh terminal management decisions only in the direct main app;
            False by default keeps library command dispatch unable to elevate.
        refresh_backend (Callable[[], None] | None): Reselect the live execution backend after an
            approved setup operation. Defaults to ``None`` when no execution service is composed.
    """

    _main_app: bool
    _refresh_backend: Callable[[], None] | None

    def __init__(
        self,
        *,
        main_app: bool = False,
        refresh_backend: Callable[[], None] | None = None,
    ) -> None:
        self._main_app = main_app
        self._refresh_backend = refresh_backend

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return sandbox management command registrations.

        Returns:
            tuple[CommandRegistration, ...]: The Linux ``/sandbox`` command, or no registrations
                on other hosts.
        """
        if platform.system() != "Linux":
            return ()
        return (
            CommandRegistration(
                self.sandbox,
                name="sandbox",
                completion=CommandCompletion(
                    values=(
                        CompletionValue("status", "Check setup; use details=true for diagnostics."),
                        CompletionValue("install", "Install Linux sandbox setup."),
                        CompletionValue("reinstall", "Remove and reinstall Linux sandbox setup."),
                        CompletionValue("refresh", "Remove and reinstall Linux sandbox setup."),
                        CompletionValue("uninstall", "Remove Linux sandbox setup."),
                    ),
                    children={
                        "status": CommandCompletion(
                            values=(
                                CompletionValue(
                                    "details=true", "Show the full installation diagnostics."
                                ),
                            )
                        )
                    },
                ),
            ),
        )

    def sandbox(
        self,
        context: CommandContext,
        action: Annotated[
            Literal[
                "status",
                "install",
                "reinstall",
                "refresh",
                "uninstall",
            ],
            Field(description="Native management operation."),
        ] = "status",
        details: Annotated[
            bool,
            Field(description="Show detailed Linux installation diagnostics with status."),
        ] = False,
    ) -> None:
        """Check Linux setup or manage it with fresh confirmation.

        Args:
            context (CommandContext): Slash-command interaction used to show status and request
                consent.
            action (str): Status, install, reinstall, refresh, or uninstall operation.
            details (bool): Show the detailed installation inventory with status; status only.

        Returns:
            None: The command reports its result through the interaction.

        Raises:
            CommandArgumentError: Linux management is unavailable or the arguments are invalid.
            ProblemException: Native management is incomplete; the problem reports current state.
        """
        if platform.system() != "Linux":
            raise CommandArgumentError("Linux sandbox management is unavailable on this host.")

        from .linux.management import installation_status
        from .linux.setup import manage_setup

        if action == "status":
            context.interaction.info(installation_status(details=details))
            return
        if details:
            raise CommandArgumentError("The details option is available only with /sandbox status.")

        setup_approved = False

        def review(description: str, operation: str = action) -> bool:
            """Display the user-facing effect and audit fresh direct human consent."""
            nonlocal setup_approved
            context.interaction.info(description)
            approved = context.interaction.confirm(
                f"{'Install' if operation == 'install' else 'Uninstall'} the Linux sandbox?",
                default=False,
            )
            setup_approved = setup_approved or approved is True
            telemetry_audit(
                "sandbox.management.decided", action=operation, approved=approved is True
            )
            return approved is True

        def manage(operation: str) -> bool:
            """Run one reviewed setup operation and refresh authority after approved changes."""
            nonlocal setup_approved
            setup_approved = False
            try:
                return manage_setup(
                    operation,
                    lambda description: review(description, operation),
                    main_app=self._main_app,
                )
            finally:
                if setup_approved and self._refresh_backend is not None:
                    self._refresh_backend()

        try:
            with telemetry_span(metadata={"operation": "sandbox." + action}):
                if action in {"reinstall", "refresh"}:
                    outcomes = []
                    failure = None
                    for operation in ("uninstall", "install"):
                        try:
                            completed = manage(operation)
                        except (ValueError, OSError, ExceptionGroup) as error:
                            outcomes.extend(
                                f"{operation.capitalize()}: incomplete ({reason})"
                                for reason in _failure_lines(error)
                            )
                            failure = failure or error
                            completed = False
                            continue
                        outcomes.append(
                            f"{operation.capitalize()}: {'completed' if completed else 'declined'}"
                        )
                        if not completed:
                            break
                    if failure is not None and not completed:
                        telemetry_audit(
                            "sandbox.management.finished", action="reinstall", success=False
                        )
                        raise ProblemException(
                            Problem(
                                code="sandbox.management.incomplete",
                                title="Native reinstall incomplete",
                                detail="\n".join((*outcomes, installation_status())),
                                severity="error",
                                operation="sandbox.reinstall",
                            )
                        ) from failure
                    if not completed:
                        context.interaction.info("\n".join((*outcomes, installation_status())))
                        return
                    context.interaction.info("\n".join((*outcomes, installation_status())))
                    completed = True
                else:
                    completed = manage(action)
        except CommandArgumentError:
            raise
        except (ValueError, OSError, ExceptionGroup) as error:
            telemetry_audit("sandbox.management.finished", action=action, success=False)
            raise ProblemException(
                Problem(
                    code="sandbox.management.incomplete",
                    title=f"Native {action} incomplete",
                    detail="\n".join(
                        (
                            *(
                                f"{action.capitalize()}: {reason}"
                                for reason in _failure_lines(error)
                            ),
                            installation_status(),
                        )
                    ),
                    severity="error",
                    operation=f"sandbox.{action}",
                )
            ) from error
        if completed:
            telemetry_audit("sandbox.management.finished", action=action, success=True)
            if action in {"reinstall", "refresh"}:
                return
            context.interaction.info(installation_status())
        else:
            context.interaction.info(f"{action.capitalize()}: declined.\n" + installation_status())
