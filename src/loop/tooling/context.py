"""Define tool invocation context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from ..errors import Problem, ProblemException
from ..interaction import Interaction
from ..utils import content_digest
from .models import ToolRuntimeSettings

if TYPE_CHECKING:
    from ..execution import CommandExecutionService
    from ..instructions import InstructionsManager
    from ..permissions import OperationPlan, Operations, PermissionManager

AdditionalAuthorizer = Callable[[dict[str, object]], "OperationPlan"]


@dataclass(frozen=True)
class ToolContext:
    """Provide runtime services and metadata to a context-aware tool.

    Args:
        interaction (Interaction | None): Optional service used to communicate with the user.
        tool_name (str): Public name of the tool being invoked.
        instructions_manager (InstructionsManager | None): Manager for instructions active in the
            current conversation, or ``None`` when instruction management is unavailable.
        operations (Operations): Authorized operations for this invocation.
        prerequisite_operations (Operations): Authorized inspection operations that
            produced the executable plan.
        call_id (str | None): Stable model call identifier suitable as an idempotency key, or
            ``None`` for direct user-command invocations.
        additional_authorizer (AdditionalAuthorizer | None): Registry-owned callback that plans
            and authorizes effects discovered during execution, or ``None`` when unavailable.
        settings (ToolRuntimeSettings): Scoped settings available to tool implementations.
        permission_manager (PermissionManager | None): Authority for a native command's one-off
            approval, or None outside registry dispatch.
        execution_service (CommandExecutionService | None): Registry-injected facade for tools
            with native execution authorization, or ``None`` otherwise.
        file_versions (dict[str, str | None]): Versions successfully written by this call, or
            ``None`` for paths successfully removed.
            Defaults to an independent empty dictionary.
        read_versions (dict[str, str | None]): Registry-owned read receipts. File paths map to
            complete inspected source versions, including the source of a committed edit, or
            ``None`` for observed absence; cache handles map to
            ``None`` and retain their existing immutable reference identity.
        cancellation_event (Event | None): Run-scoped signal set when shutdown or interruption
            requests cancellation, or ``None`` for standalone tool dispatch.
    """

    interaction: Interaction | None
    tool_name: str
    instructions_manager: InstructionsManager | None = None
    operations: Operations = ()
    prerequisite_operations: Operations = ()
    call_id: str | None = None
    additional_authorizer: AdditionalAuthorizer | None = None
    settings: ToolRuntimeSettings = field(default_factory=ToolRuntimeSettings)
    permission_manager: PermissionManager | None = None
    execution_service: CommandExecutionService | None = None
    file_versions: dict[str, str | None] = field(default_factory=dict)
    read_versions: dict[str, str | None] = field(default_factory=dict)
    cancellation_event: Event | None = None

    def record_file_read(self, path: Path, version: str | None) -> None:
        """Attest to the source version actually inspected by a file reader.

        Args:
            path (Path): Canonical inspected source.
            version (str | None): Complete source identity, or ``None`` for observed absence.
        """
        self.read_versions[self.display_path(path)] = version

    def record_cached_read(self, handle: str) -> None:
        """Attest that a cached source was read through its immutable capability.

        Args:
            handle (str): Successfully read content capability.
        """
        self.read_versions[handle] = None

    def record_file_version(self, path: Path, content: str) -> None:
        """Record the exact content committed by a successful file mutation.

        Args:
            path (Path): Canonical destination used by the executor.
            content (str): Complete UTF-8 content committed by this call.
        """
        self.file_versions[self.display_path(path)] = content_digest(content)

    def record_file_removal(self, path: Path) -> None:
        """Record a path successfully removed by a model mutation.

        Args:
            path (Path): Canonical destination removed by the executor.
        """
        self.file_versions[self.display_path(path)] = None

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

    def display_path(self, path: Path | str) -> str:
        """Return a model-safe display form for one resolved local path.

        Args:
            path (Path | str): Local path used by the tool implementation.

        Returns:
            str: Logical or virtual path when instruction management is available.
        """
        rendered = str(path)
        if self.instructions_manager is not None:
            return self.instructions_manager.virtual_paths.display(rendered)
        return rendered

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to confirm an action through the interaction service.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the action.
        """
        if self.interaction is None:
            raise ProblemException(
                Problem(
                    code="tool.interaction_unavailable",
                    title="Interaction unavailable",
                    detail=f"Tool '{self.tool_name}' requires user interaction.",
                    severity="warning",
                    operation=self.tool_name,
                )
            )
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
