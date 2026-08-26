"""Define tool invocation context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import Problem, ProblemException
from ..interaction import Interaction

if TYPE_CHECKING:
    from ..permissions import Operation, OperationPlan
    from ..skills import InstructionsManager

AdditionalAuthorizer = Callable[[dict[str, object]], "OperationPlan"]


@dataclass(frozen=True)
class ToolContext:
    """Provide runtime services and metadata to a context-aware tool.

    Args:
        interaction (Interaction): Service used to communicate with the user.
        tool_name (str): Public name of the tool being invoked.
        instructions_manager (InstructionsManager | None): Manager for instructions active in the
            current conversation, or ``None`` when instruction management is unavailable.
        operations (tuple[Operation, ...]): Authorized operations for this invocation.
        call_id (str | None): Stable model call identifier suitable as an idempotency key, or
            ``None`` for direct user-command invocations.
        additional_authorizer (AdditionalAuthorizer | None): Registry-owned callback that plans
            and authorizes effects discovered during execution, or ``None`` when unavailable.
    """

    interaction: Interaction
    tool_name: str
    instructions_manager: InstructionsManager | None = None
    operations: tuple[Operation, ...] = ()
    call_id: str | None = None
    additional_authorizer: AdditionalAuthorizer | None = None

    def observe_file(self, path: Path | str) -> None:
        """Report a successfully loaded file to instruction management.

        Args:
            path (Path | str): File whose containing instruction scope is now relevant.
        """
        if self.instructions_manager is not None:
            self.instructions_manager.observe_path(path)

    def observe_directory(self, path: Path | str) -> None:
        """Report a successfully navigated directory to instruction management.

        Args:
            path (Path | str): Directory whose instruction scope is now relevant.
        """
        if self.instructions_manager is not None:
            self.instructions_manager.observe_path(path, directory=True)

    def invalidate_instructions(self, path: Path | str | None = None) -> None:
        """Report that a discovered instruction source may have changed.

        Args:
            path (Path | str | None): Changed source, or ``None`` for unconditional invalidation.
        """
        if self.instructions_manager is not None:
            self.instructions_manager.invalidate(path)

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to confirm an action through the interaction service.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the action.
        """
        return self.interaction.confirm(message, default=default)

    def authorize_additional(self, arguments: dict[str, object]) -> OperationPlan:
        """Plan and authorize effects discovered during this tool invocation.

        Args:
            arguments (dict[str, object]): Raw replacement arguments for the current registered
                tool's operation planner.

        Returns:
            OperationPlan: Canonical plan authorized for execution.

        Raises:
            ProblemException: If runtime authorization is unavailable or the plan is denied.
            ValueError: If the additional arguments cannot produce a valid operation plan.
        """
        if self.additional_authorizer is None:
            raise ProblemException(
                Problem(
                    code="tool.authorization_unavailable",
                    title="Additional authorization unavailable",
                    detail=(
                        f"Tool '{self.tool_name}' discovered an additional operation, but this "
                        "execution path cannot authorize it."
                    ),
                    severity="warning",
                    operation=self.tool_name,
                )
            )
        return self.additional_authorizer(arguments)
