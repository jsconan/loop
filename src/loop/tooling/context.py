"""Define tool invocation context."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..interaction import Interaction
from ..permissions import Capability, Decision, PermissionManager, PermissionRequest

if TYPE_CHECKING:
    from ..skills import InstructionsManager


@dataclass(frozen=True)
class ToolContext:
    """Provide runtime services and metadata to a context-aware tool.

    Args:
        interaction (Interaction): Service used to communicate with the user.
        tool_name (str): Public name of the tool being invoked.
        instructions_manager (InstructionsManager | None): Manager for instructions active in the
            current conversation, or ``None`` when instruction management is unavailable.
        permission_manager (PermissionManager | None): Central manager for additional authorization.
        grants (frozenset[PermissionRequest]): Requests authorized before tool execution.
    """

    interaction: Interaction
    tool_name: str
    instructions_manager: InstructionsManager | None = None
    permission_manager: PermissionManager | None = None
    grants: frozenset[PermissionRequest] = frozenset()

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

    def authorize(
        self,
        capability: Capability,
        *,
        resource: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """Request additional policy authorization discovered by a running tool.

        Args:
            capability (Capability): Additional authority required.
            resource (str | None): Normalized affected resource.
            reason (str | None): Explanation shown to the user.

        Returns:
            bool: Whether the permission policy authorized the operation.
        """
        request = PermissionRequest(
            tool_name=self.tool_name,
            capability=capability,
            resource=resource,
            reason=reason,
        )
        if request in self.grants:
            return True
        if self.permission_manager is None:
            return False
        result = self.permission_manager.authorize(request)
        return result.decision is Decision.ALLOW
