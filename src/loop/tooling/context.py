"""Define tool invocation context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Literal

from ..errors import Problem, ProblemException
from ..interaction import Interaction, require_async_interaction
from ..utils import VirtualPath, content_digest
from .models import ToolRuntimeSettings

if TYPE_CHECKING:
    from ..execution import CommandExecutionService
    from ..instructions import InstructionsManager
    from ..permissions import OperationPlan, Operations, PermissionManager
    from ..permissions.manager import EgressRevocationLease
    from ..session import SessionManager

AdditionalAuthorizer = Callable[[dict[str, object]], "OperationPlan"]
AdditionalAuthorizerAsync = Callable[[dict[str, object]], Awaitable["OperationPlan"]]


@dataclass(frozen=True)
class CommandEgressAuthority:
    """Retain only thread-safe egress policy operations for a continued command.

    Args:
        bind (Callable[[int, EgressRevocationLease], None]): Bind an approved lease.
        unbind (Callable[[EgressRevocationLease], None]): Remove a settled lease.
        revision (Callable[[], int | None]): Check the active policy revision.
    """

    bind: Callable[[int, EgressRevocationLease], None]
    unbind: Callable[[EgressRevocationLease], None]
    revision: Callable[[], int | None]

    def bind_egress(self, revision: int, lease: EgressRevocationLease) -> None:
        """Bind an egress lease to its approved policy revision.

        Args:
            revision (int): Approved policy revision.
            lease (EgressRevocationLease): Native attempt's revocation channel.

        Raises:
            OSError: The authority is closing or its approved policy is no longer current.
        """
        self.bind(revision, lease)

    def unbind_egress(self, lease: EgressRevocationLease) -> None:
        """Remove a settled egress lease.

        Args:
            lease (EgressRevocationLease): Previously bound revocation channel.
        """
        self.unbind(lease)

    def try_egress_policy_revision(self) -> int | None:
        """Check the policy revision without blocking an update.

        Returns:
            int | None: Current revision, or None while an update holds the authority.

        Raises:
            OSError: Required managed authority is unavailable or revocation is uncertain.
        """
        return self.revision()


def _confirmation_decision(value: object, tool_name: str) -> bool:
    """Reject malformed interaction results without treating them as user decisions."""
    if value is True or value is False:
        return value
    raise ProblemException(
        Problem(
            code="tool.invalid_confirmation",
            title="Invalid confirmation",
            detail="The interaction did not return a boolean confirmation decision.",
            severity="warning",
            operation=tool_name,
        )
    )


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
        call_id (str | None): Provider identifier for this model call, or ``None`` for direct
            user-command invocations. Identifiers can recur in later responses.
        additional_authorizer (AdditionalAuthorizer | None): Registry-owned callback that plans
            and authorizes effects discovered during execution, or ``None`` when unavailable.
        additional_authorizer_async (AdditionalAuthorizerAsync | None): Native asynchronous
            callback for runtime-discovered effects, or ``None`` when unavailable.
        settings (ToolRuntimeSettings): Scoped settings available to tool implementations.
        permission_manager (PermissionManager | None): Authority for a native command's one-off
            approval, or None outside registry dispatch.
        execution_service (CommandExecutionService | None): Registry-injected facade for tools
            with native execution authorization, or ``None`` otherwise.
        session_manager (SessionManager | None): Selected session whose native ownership lease
            must outlive a detached command, or ``None`` for standalone dispatch.
        file_versions (dict[str, str | None]): Versions successfully written by this call, or
            ``None`` for paths successfully removed.
            Defaults to an independent empty dictionary.
        read_versions (dict[str, str | None]): Registry-owned read receipts. File paths map to
            complete inspected source versions, including the source of a committed edit, or
            ``None`` for observed absence; cache handles map to
            ``None`` and retain their existing immutable reference identity.
        cancellation_event (Event | None): Run or async-dispatch signal set on interruption,
            or ``None`` when no signal is supplied to a synchronous dispatch.
        execution_mode (Literal["sync", "async"]): Callback mode selected by the owning registry.
            Defaults to ``"sync"`` for contexts assembled outside registry dispatch.
        standalone_execution (Callable[[], AbstractContextManager[Path]] | None): Thread-safe
            scratch lease for commands without a selected session.
        invalidate_detached (Callable[[], None] | None): Thread-safe instruction invalidation
            for work that outlives its originating async dispatch.
        egress_authority (CommandEgressAuthority | None): Thread-safe, limited policy authority
            for a mediated command that outlives its originating async dispatch.
        audit_host_detached (Callable[..., None] | None): Thread-safe host-boundary audit callback
            for work that outlives its originating async dispatch.
        command_admission (Callable[[], AbstractContextManager[None]] | None): Registry gate
            protecting the selected session through command reservation.
        command_authorities (tuple[object, ...] | None): Original shared authorities when
            owner-thread proxies are used, or None when invocation values are already original.
        command_paths (VirtualPath | None): Immutable command root snapshot, if captured.
        admit_skills_detached (Callable[..., bool] | None): Thread-safe skill admission on the
            original instructions manager for detached command workers.
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
    session_manager: SessionManager | None = None
    file_versions: dict[str, str | None] = field(default_factory=dict)
    read_versions: dict[str, str | None] = field(default_factory=dict)
    cancellation_event: Event | None = None
    additional_authorizer_async: AdditionalAuthorizerAsync | None = None
    execution_mode: Literal["sync", "async"] = "sync"
    standalone_execution: Callable[[], AbstractContextManager[Path]] | None = None
    invalidate_detached: Callable[[], None] | None = None
    egress_authority: CommandEgressAuthority | None = None
    audit_host_detached: Callable[..., None] | None = None
    command_admission: Callable[[], AbstractContextManager[None]] | None = None
    command_authorities: tuple[object, ...] | None = None
    command_paths: VirtualPath | None = None
    admit_skills_detached: Callable[..., bool] | None = None

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

        Raises:
            ProblemException: If interaction is unavailable, returns an invalid decision, or an
                async tool attempts blocking confirmation.
        """
        if self.execution_mode == "async":
            raise ProblemException(
                Problem(
                    code="tool.async_required",
                    title="Asynchronous interaction required",
                    detail=(
                        f"Tool '{self.tool_name}' runs asynchronously; await "
                        "context.confirm_async() instead."
                    ),
                    operation=self.tool_name,
                )
            )
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
        return _confirmation_decision(
            self.interaction.confirm(message, default=default), self.tool_name
        )

    async def confirm_async(self, message: str, *, default: bool = False) -> bool:
        """Ask the user for confirmation through native asynchronous input.

        Args:
            message (str): Confirmation question.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the action.

        Raises:
            ProblemException: If interaction is unavailable, lacks native async input, or returns
                an invalid decision.
            asyncio.CancelledError: If the interaction is cancelled.
        """
        interaction = require_async_interaction(self.interaction, "tool confirmation")
        return _confirmation_decision(
            await interaction.confirm_async(message, default=default), self.tool_name
        )

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
        if self.execution_mode == "async":
            raise ProblemException(
                Problem(
                    code="tool.async_required",
                    title="Asynchronous authorization required",
                    detail=(
                        f"Tool '{self.tool_name}' runs asynchronously; await "
                        "context.authorize_additional_async() instead."
                    ),
                    operation=self.tool_name,
                )
            )
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

    async def authorize_additional_async(self, arguments: dict[str, object]) -> OperationPlan:
        """Authorize runtime-discovered effects through the async registry path.

        Args:
            arguments (dict[str, object]): Raw replacement arguments for the current registered
                tool's operation planner.

        Returns:
            OperationPlan: Canonical plan authorized for execution.

        Raises:
            ProblemException: If runtime authorization is unavailable or the plan is denied.
            ValueError: If the additional arguments cannot produce a valid operation plan.
            asyncio.CancelledError: If asynchronous authorization is cancelled.
        """
        if self.additional_authorizer_async is None:
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
        return await self.additional_authorizer_async(arguments)
