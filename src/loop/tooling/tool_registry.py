"""Register and dispatch typed functions exposed to an LLM."""

import logging
from collections.abc import Callable, Iterable
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from ..commands.models import CommandArgumentError
from ..commands.utils import parse_model_arguments
from ..constants import OMIT, Omit
from ..errors import Problem, ProblemException, log_problem
from ..interaction import Interaction
from ..models import (
    ToolDefinition,
    ToolExecutionResult,
    ToolResultPresentationDeclaration,
)
from ..permissions import (
    Action,
    Decision,
    Operation,
    OperationPlan,
    OperationPlanner,
    PermissionManager,
)
from ..skills import InstructionsManager
from ..utils import callable_name
from .context import ToolContext
from .models import ToolPreflight, ToolRegistrationError, ToolStatus
from .tool import Tool, ToolRegistration
from .utils import serialize_tool_problem

_LOGGER = logging.getLogger(__name__)


class ToolRegistry:
    """Collect tool declarations and route model calls to their implementations.

    Args:
        tools (Iterable[Callable[..., Any] | ToolRegistration] | None): Functions or configured
            registrations to add in iteration order, or ``None`` to construct an empty registry.
            Functions may optionally carry metadata from the standalone ``@tool`` decorator.
        interaction (Interaction | None): Default interaction used by context-aware tools when
            dispatch does not provide one, or ``None`` to require an invocation-specific
            interaction.
        permission_manager (PermissionManager | None): Central policy manager guarding every call.
            Defaults to an in-memory supervised policy manager.
    """

    _tools: dict[str, Tool]
    _interaction: Interaction | None
    _permission_manager: PermissionManager
    _registration_problems: list[Problem]

    def __init__(
        self,
        tools: Iterable[Callable[..., Any] | ToolRegistration] | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> None:
        self._tools = {}
        self._registration_problems = []
        self._interaction = interaction
        self._permission_manager = permission_manager or PermissionManager(interaction=interaction)
        for tool in tools or ():
            self.register(tool)

    @property
    def interaction(self) -> Interaction | None:
        """Return the default interaction used during tool dispatch.

        Returns:
            Interaction | None: The default interaction, or ``None`` when none is configured.
        """
        return self._interaction

    @interaction.setter
    def interaction(self, interaction: Interaction | None) -> None:
        """Set or clear the default interaction used during tool dispatch.

        Args:
            interaction (Interaction | None): Default interaction to use, or ``None`` to clear it.
        """
        self._interaction = interaction
        self._permission_manager.interaction = interaction

    @property
    def permission_manager(self) -> PermissionManager:
        """Return the permission manager guarding dispatch.

        Returns:
            PermissionManager: Active centralized permission manager.
        """
        return self._permission_manager

    @permission_manager.setter
    def permission_manager(self, manager: PermissionManager) -> None:
        """Replace the permission manager guarding dispatch.

        Args:
            manager (PermissionManager): Manager to use for future calls.
        """
        self._permission_manager = manager

    @property
    def tools(self) -> list[Tool]:
        """Return registered tools sorted alphabetically by name.

        Returns:
            list[Tool]: Registered tools.
        """
        return sorted(self._tools.values(), key=lambda tool: tool.name.casefold())

    @property
    def names(self) -> list[str]:
        """Return registered tool names sorted alphabetically.

        Returns:
            list[str]: Registered tool names.
        """
        return sorted(self._tools.keys(), key=str.casefold)

    @property
    def registration_problems(self) -> tuple[Problem, ...]:
        """Return immutable diagnostics from skipped tool registrations.

        Returns:
            tuple[Problem, ...]: Problems encountered by preflight checks in registration order.
        """
        return tuple(self._registration_problems)

    def register(
        self,
        function: Callable[..., Any] | ToolRegistration,
        *,
        name: str | None = None,
        description: str | None = None,
        actions: Iterable[Action] | None = None,
        operation_planner: OperationPlanner | None | Omit = OMIT,
        result_presentation: ToolResultPresentationDeclaration | Omit = OMIT,
        preflight: ToolPreflight | None | Omit = OMIT,
        required: bool | Omit = OMIT,
    ) -> bool:
        """Create and register a tool from a callable or configured registration.

        Args:
            function (Callable[..., Any] | ToolRegistration): Function to expose as a tool, or a
                registration that supplies its local metadata.
            name (str | None): Container-specific public name. Defaults to the declared name or
                function name.
            description (str | None): Container-specific public description. Defaults to the
                declared description or docstring summary.
            actions (Iterable[Action] | None): Container-specific action upper bound. Defaults to
                declared actions or no effects.
            operation_planner (OperationPlanner | None | Omit): Container-specific planner.
                Omit it to inherit the declared planner; pass ``None`` to remove one.
            result_presentation (ToolResultPresentationDeclaration | Omit): Container-specific
                presentation declaration. Omit it to inherit.
            preflight (ToolPreflight | None | Omit): Container-specific readiness check. Omit it
                to inherit; pass ``None`` to remove one.
            required (bool | Omit): Whether a broken tool requires an explicit choice to continue.
                Omit it to inherit.

        Returns:
            bool: Whether the tool was registered.

        Raises:
            ToolRegistrationError: If the resolved name is already registered, the function has
                no description, its parameters cannot be represented by an arguments model, or a
                required broken tool halts registration.
        """
        if isinstance(function, ToolRegistration):
            registration = function
            function = registration.function
            name = name or registration.name
            description = description or registration.description
            actions = actions if actions is not None else registration.actions
            operation_planner = (
                registration.operation_planner
                if isinstance(operation_planner, Omit)
                else operation_planner
            )
            result_presentation = (
                registration.result_presentation
                if isinstance(result_presentation, Omit)
                else result_presentation
            )
            preflight = registration.preflight if isinstance(preflight, Omit) else preflight
            required = registration.required if isinstance(required, Omit) else required
        declared_tool = Tool.get_declaration(function)
        if declared_tool is None:
            declared_tool = Tool(function=function)
        tool_name = name or declared_tool.name or callable_name(function)
        if tool_name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool_name}' is already registered.")
        registered = declared_tool.registered(
            name=name,
            description=description,
            actions=actions,
            operation_planner=operation_planner,
            result_presentation=result_presentation,
            preflight=preflight,
            required=required,
        )
        if registered.preflight is not None:
            preflight_error = None
            try:
                readiness = registered.preflight()
            except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
                preflight_error = exc
                readiness_status = ToolStatus.BROKEN
                problem = Problem.from_exception(
                    exc,
                    code="tool.preflight_error",
                    title="Tool readiness check failed",
                    detail=f"Could not determine whether tool '{tool_name}' is available.",
                    operation=tool_name,
                )
            else:
                readiness_status = readiness.status
                problem = (
                    None
                    if readiness_status is ToolStatus.READY
                    else Problem(
                        code=f"tool.preflight_{readiness_status.value}",
                        title=f"Tool {readiness_status.value}",
                        detail=readiness.detail
                        or f"Tool '{tool_name}' is {readiness_status.value}.",
                        severity="warning" if readiness_status is ToolStatus.DEGRADED else "error",
                        operation=tool_name,
                    )
                )
            if problem is not None:
                log_problem(_LOGGER, problem, preflight_error)
                self._registration_problems.append(problem)
                if self._interaction is not None:
                    self._interaction.warning(problem.detail)
                if readiness_status is ToolStatus.DEGRADED:
                    self._tools[tool_name] = registered
                    return True
                if registered.required and (
                    self._interaction is None
                    or self._interaction.prompt(
                        f"Required tool '{tool_name}' is unavailable:",
                        exit_commands=(),
                        choices={
                            "halt": "Halt startup",
                            "continue": "Continue without this tool",
                        },
                    )
                    != "continue"
                ):
                    raise ToolRegistrationError(problem.detail)
                return False
        self._tools[tool_name] = registered
        return True

    def definitions(self) -> list[ToolDefinition]:
        """Return definitions for all registered tools.

        Returns:
            list[ToolDefinition]: Function-tool definitions in registration order.
        """
        return [tool.definition() for tool in self._tools.values()]

    def call(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        call_id: str | None = None,
        execution_started: Callable[[], None] | None = None,
    ) -> str:
        """Dispatch a synchronous tool call by registered name.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation. Overrides the
                registry default.
            instructions_manager (InstructionsManager | None): Instruction manager active for the
                current conversation.
            permission_manager (PermissionManager | None): Invocation policy overriding the
                registry default.
            call_id (str | None): Stable model request identifier exposed to context-aware tools.
            execution_started (Callable[[], None] | None): Callback invoked immediately before
                the validated and authorized tool function runs.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        output, _ = self.call_with_timing(
            name,
            arguments,
            interaction=interaction,
            instructions_manager=instructions_manager,
            permission_manager=permission_manager,
            call_id=call_id,
            execution_started=execution_started,
        )
        return output

    def call_with_timing(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        call_id: str | None = None,
        execution_started: Callable[[], None] | None = None,
    ) -> tuple[str, float]:
        """Dispatch a tool and measure only its function execution.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation.
            instructions_manager (InstructionsManager | None): Active instruction manager.
            permission_manager (PermissionManager | None): Invocation permission policy.
            call_id (str | None): Stable model request identifier exposed to context-aware tools.
            execution_started (Callable[[], None] | None): Callback invoked immediately before
                the validated and authorized tool function runs.

        Returns:
            tuple[str, float]: Serialized result and tool-function duration in seconds. Validation
                and authorization failures have a zero duration because no tool ran.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return self._problem(
                "tool.unknown", "Tool unavailable", f"Tool '{name}' is not available.", name
            ), 0
        validated, error = tool.validate_arguments(arguments)
        if error is not None:
            return error, 0
        active_permissions = permission_manager or self._permission_manager
        try:
            plan = tool.plan(validated)
        except ProblemException as exc:
            return serialize_tool_problem(exc.problem), 0
        except ValueError as exc:
            return self._problem("tool.planning_failed", "Tool planning failed", str(exc), name), 0
        denied = self._authorize(tool, plan.operations, interaction, active_permissions)
        if denied is not None:
            return denied, 0
        context = self._context_for(
            tool,
            interaction,
            instructions_manager,
            plan.operations,
            call_id,
            active_permissions,
        )
        if execution_started is not None:
            execution_started()
        started = perf_counter()
        output = tool.call(plan.arguments, context)
        return output, perf_counter() - started

    async def call_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        call_id: str | None = None,
        execution_started: Callable[[], None] | None = None,
    ) -> str:
        """Dispatch an asynchronous or synchronous tool call by registered name.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation. Overrides the
                registry default.
            instructions_manager (InstructionsManager | None): Instruction manager active for the
                current conversation.
            permission_manager (PermissionManager | None): Invocation policy overriding the
                registry default.
            call_id (str | None): Stable model request identifier exposed to context-aware tools.
            execution_started (Callable[[], None] | None): Callback invoked immediately before
                the validated and authorized tool function runs.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        output, _ = await self.call_with_timing_async(
            name,
            arguments,
            interaction=interaction,
            instructions_manager=instructions_manager,
            permission_manager=permission_manager,
            call_id=call_id,
            execution_started=execution_started,
        )
        return output

    async def call_with_timing_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        call_id: str | None = None,
        execution_started: Callable[[], None] | None = None,
    ) -> tuple[str, float]:
        """Dispatch a tool asynchronously and measure only its function execution.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation.
            instructions_manager (InstructionsManager | None): Active instruction manager.
            permission_manager (PermissionManager | None): Invocation permission policy.
            call_id (str | None): Stable model request identifier exposed to context-aware tools.
            execution_started (Callable[[], None] | None): Callback invoked immediately before
                the validated and authorized tool function runs.

        Returns:
            tuple[str, float]: Serialized result and tool-function duration in seconds. Validation
                and authorization failures have a zero duration because no tool ran.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return self._problem(
                "tool.unknown", "Tool unavailable", f"Tool '{name}' is not available.", name
            ), 0
        validated, error = tool.validate_arguments(arguments)
        if error is not None:
            return error, 0
        active_permissions = permission_manager or self._permission_manager
        try:
            plan = tool.plan(validated)
        except ProblemException as exc:
            return serialize_tool_problem(exc.problem), 0
        except ValueError as exc:
            return self._problem("tool.planning_failed", "Tool planning failed", str(exc), name), 0
        denied = self._authorize(tool, plan.operations, interaction, active_permissions)
        if denied is not None:
            return denied, 0
        context = self._context_for(
            tool,
            interaction,
            instructions_manager,
            plan.operations,
            call_id,
            active_permissions,
        )
        if execution_started is not None:
            execution_started()
        started = perf_counter()
        output = await tool.call_async(plan.arguments, context)
        return output, perf_counter() - started

    def command(
        self,
        name: str,
        arguments: list[str] | tuple[str, ...],
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
    ) -> ToolExecutionResult:
        """Dispatch a planned user-command tool call without permission evaluation.

        Args:
            name (str): Registered tool name.
            arguments (list[str] | tuple[str, ...]): Positional and ``name=value`` argument tokens.
            interaction (Interaction | None): Interaction for this invocation. Overrides the
                registry default.
            instructions_manager (InstructionsManager | None): Instruction manager active for the
                current conversation.
        Returns:
            ToolExecutionResult: Serialized result and its user-presentation metadata.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                self._problem(
                    "tool.unknown", "Tool unavailable", f"Tool '{name}' is not available.", name
                )
            )
        try:
            validated = parse_model_arguments(tool.arguments_model, arguments).model_dump()
        except (CommandArgumentError, ValidationError) as exc:
            details = (
                exc.errors(include_url=False)
                if isinstance(exc, ValidationError)
                else [{"type": "argument_binding", "msg": str(exc)}]
            )
            return ToolExecutionResult(
                serialize_tool_problem(
                    Problem(
                        code="tool.invalid_arguments",
                        title="Invalid tool arguments",
                        detail=f"Invalid arguments for tool '{name}'.",
                        severity="warning",
                        operation=name,
                        metadata={"fields": details},
                    )
                )
            )
        try:
            plan = tool.plan(validated)
        except ProblemException as exc:
            return ToolExecutionResult(serialize_tool_problem(exc.problem))
        except ValueError as exc:
            return ToolExecutionResult(
                self._problem("tool.planning_failed", "Tool planning failed", str(exc), name)
            )
        context = self._context_for(
            tool,
            interaction,
            instructions_manager,
            plan.operations,
        )
        return tool.execute(plan.arguments, context)

    def _authorize(
        self,
        tool: Tool,
        operations: tuple[Operation, ...],
        interaction: Interaction | None,
        permission_manager: PermissionManager,
    ) -> str | None:
        """Return a serialized denial or authorize the complete operation plan."""
        active_interaction = interaction if interaction is not None else self._interaction
        result = permission_manager.authorize(operations, interaction=active_interaction)
        if result.decision is Decision.DENY:
            return serialize_tool_problem(
                Problem(
                    code="tool.denied",
                    title="Tool call denied",
                    detail=f"Tool '{tool.name}' was not executed: {result.reason}",
                    severity="warning",
                    operation=tool.name,
                )
            )
        return None

    @staticmethod
    def _problem(code: str, title: str, detail: str, operation: str) -> str:
        """Serialize one tool problem with its operation context."""
        return serialize_tool_problem(
            Problem(code=code, title=title, detail=detail, operation=operation)
        )

    def _context_for(
        self,
        tool: Tool,
        interaction: Interaction | None,
        instructions_manager: InstructionsManager | None,
        operations: tuple[Operation, ...] = (),
        call_id: str | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> ToolContext | None:
        """Build a tool context from the invocation override or registry default."""
        if interaction is None:
            interaction = self._interaction
        if interaction is None:
            return None

        def authorize_additional(arguments: dict[str, object]) -> OperationPlan:
            """Plan and authorize one runtime-discovered operation set."""
            if permission_manager is None:  # pragma: no cover - callback is omitted below.
                raise RuntimeError("Additional authorization is unavailable.")
            plan = tool.plan(arguments)
            result = permission_manager.authorize(plan.operations, interaction=interaction)
            if result.decision is Decision.DENY:
                raise ProblemException(
                    Problem(
                        code="tool.denied",
                        title="Additional operation denied",
                        detail=f"Tool '{tool.name}' stopped: {result.reason}",
                        severity="warning",
                        operation=tool.name,
                    )
                )
            return plan

        return ToolContext(
            interaction=interaction,
            tool_name=tool.name,
            call_id=call_id,
            instructions_manager=instructions_manager,
            operations=operations,
            additional_authorizer=(
                authorize_additional if permission_manager is not None else None
            ),
        )
