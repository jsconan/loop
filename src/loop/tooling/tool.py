"""Declare, validate, and invoke typed functions exposed to an LLM."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, overload

from pydantic import BaseModel, ValidationError

from ..constants import OMIT, Omit
from ..errors import Problem, log_problem
from ..models import (
    RAW_TOOL_RESULT_PRESENTATION,
    ToolDefinition,
    ToolExecutionResult,
    ToolResultPresentationDeclaration,
    ToolResultPresentationSpec,
)
from ..permissions import Action, OperationPlan, OperationPlanner
from ..utils import callable_name
from .context import ToolContext
from .utils import (
    ToolRegistrationError,
    get_tool_arguments_model,
    get_tool_description,
    get_tool_schema,
    is_async_callable,
    serialize_tool_problem,
    serialize_tool_result,
    takes_tool_context,
)

_TOOL_ATTR = "__loop_tool__"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tool:
    """Describe and, once registered, invoke an LLM-callable function.

    Args:
        function (Callable[..., Any]): Python function invoked for the tool.
        name (str | None): Public name exposed to the model, or ``None`` to use the function name.
        description (str | None): Public description, or ``None`` to use the docstring summary.
        actions (frozenset[Action]): Upper bound of authority-bearing effects.
        operation_planner (OperationPlanner | None): Optional planner producing canonical
            arguments and typed operations.
        arguments_model (type[BaseModel] | None): Pydantic model used to validate arguments after
            registration, or ``None`` for a passive declaration.
        result_presentation (ToolResultPresentationDeclaration): Fixed presentation or selector
            applied to each successful raw result.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    actions: frozenset[Action] = frozenset()
    operation_planner: OperationPlanner | None = None
    arguments_model: type[BaseModel] | None = None
    result_presentation: ToolResultPresentationDeclaration = RAW_TOOL_RESULT_PRESENTATION

    def registered(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        actions: Iterable[Action] | None = None,
        operation_planner: OperationPlanner | None | Omit = OMIT,
        result_presentation: ToolResultPresentationDeclaration | Omit = OMIT,
    ) -> Tool:
        """Return a registry-ready copy with resolved metadata and argument validation.

        Args:
            name (str | None): Container-specific public name, or ``None`` to inherit or derive it.
            description (str | None): Container-specific description, or ``None`` to inherit or
                derive it.
            actions (Iterable[Action] | None): Container-specific action upper bound, or
                ``None`` to inherit.
            operation_planner (OperationPlanner | None | Omit): Container-specific planner.
                Omit it to inherit; pass ``None`` to remove one.
            result_presentation (ToolResultPresentationDeclaration | Omit): Container-specific
                presentation declaration. Omit it to inherit.

        Returns:
            Tool: Immutable, fully resolved tool for one registry.
        """
        resolved_name = name or self.name or callable_name(self.function)
        registered = replace(
            self,
            name=resolved_name,
            description=description or self.description or get_tool_description(self.function),
            actions=frozenset(actions) if actions is not None else self.actions,
            operation_planner=(
                self.operation_planner if isinstance(operation_planner, Omit) else operation_planner
            ),
            result_presentation=(
                self.result_presentation
                if isinstance(result_presentation, Omit)
                else result_presentation
            ),
            arguments_model=get_tool_arguments_model(self.function, resolved_name),
        )
        if registered.actions and registered.operation_planner is None:
            raise ToolRegistrationError(
                f"Tool '{resolved_name}' declares authority-bearing actions without an "
                "operation planner."
            )
        return registered

    def definition(self) -> ToolDefinition:
        """Return the function-tool definition.

        Returns:
            ToolDefinition: The strict function-tool definition.

        Raises:
            ValueError: If the tool has not been registered.
        """
        if self.arguments_model is None or self.name is None or self.description is None:
            raise ValueError("A tool must be registered before it can be defined.")
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=get_tool_schema(self.arguments_model.model_json_schema()),
        )

    def call(self, arguments: dict[str, Any], context: ToolContext | None = None) -> str:
        """Invoke a synchronous registered tool with validated arguments.

        Args:
            arguments (dict[str, Any]): Validated keyword arguments.
            context (ToolContext | None): Runtime tool context.

        Returns:
            str: Serialized tool result or model-readable error.
        """
        return self.execute(arguments, context).output

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        """Invoke a synchronous tool and retain its user-presentation metadata.

        Args:
            arguments (dict[str, Any]): Validated canonical keyword arguments.
            context (ToolContext | None): Runtime tool context.

        Returns:
            ToolExecutionResult: Serialized output and selected presentation.
        """
        if is_async_callable(self.function):
            return ToolExecutionResult(
                serialize_tool_problem(
                    Problem(
                        code="tool.async_required",
                        title="Asynchronous tool invocation required",
                        detail=f"Tool '{self._name_for_error()}' must be called asynchronously.",
                        operation=self._name_for_error(),
                    )
                )
            )
        try:
            result = self._call_function(arguments, context)
            return ToolExecutionResult(
                serialize_tool_result(result), self._presentation_for(arguments, result)
            )
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            problem = Problem.from_exception(
                exc,
                code="tool.execution_failed",
                title="Tool execution failed",
                detail=f"Tool '{self._name_for_error()}' failed.",
                operation=self._name_for_error(),
            )
            log_problem(_LOGGER, problem, exc)
            return ToolExecutionResult(serialize_tool_problem(problem))

    async def call_async(
        self, arguments: dict[str, Any], context: ToolContext | None = None
    ) -> str:
        """Invoke any registered tool asynchronously with validated arguments.

        Args:
            arguments (dict[str, Any]): Validated keyword arguments.
            context (ToolContext | None): Runtime tool context.

        Returns:
            str: Serialized tool result or model-readable error.
        """
        return (await self.execute_async(arguments, context)).output

    async def execute_async(
        self,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        """Invoke any tool asynchronously and retain its user-presentation metadata.

        Args:
            arguments (dict[str, Any]): Validated canonical keyword arguments.
            context (ToolContext | None): Runtime tool context.

        Returns:
            ToolExecutionResult: Serialized output and selected presentation.
        """
        try:
            result = self._call_function(arguments, context)
            if inspect.isawaitable(result):
                result = await result
            return ToolExecutionResult(
                serialize_tool_result(result), self._presentation_for(arguments, result)
            )
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            problem = Problem.from_exception(
                exc,
                code="tool.execution_failed",
                title="Tool execution failed",
                detail=f"Tool '{self._name_for_error()}' failed.",
                operation=self._name_for_error(),
            )
            log_problem(_LOGGER, problem, exc)
            return ToolExecutionResult(serialize_tool_problem(problem))

    def validate_arguments(self, arguments: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return validated keyword arguments or a model-readable error.

        Args:
            arguments (str): JSON-encoded arguments supplied by the model.

        Returns:
            tuple[dict[str, Any] | None, str | None]: Validated arguments and no error, or no
                arguments and an error.

        Raises:
            ValueError: If the tool has not been registered.
        """
        if self.arguments_model is None:
            raise ValueError("A tool must be registered before it can validate arguments.")
        try:
            validated = self.arguments_model.model_validate_json(arguments or "{}")
        except ValidationError as exc:
            return None, serialize_tool_problem(
                Problem(
                    code="tool.invalid_arguments",
                    title="Invalid tool arguments",
                    detail=f"Invalid arguments for tool '{self._name_for_error()}'.",
                    severity="warning",
                    operation=self._name_for_error(),
                    metadata={"fields": exc.errors(include_url=False)},
                )
            )
        return validated.model_dump(), None

    def plan(self, arguments: dict[str, Any]) -> OperationPlan:
        """Return canonical execution arguments and the complete operation set.

        Args:
            arguments (dict[str, Any]): Validated tool arguments.

        Returns:
            OperationPlan: Canonical arguments and complete typed operations.

        Raises:
            ValueError: If a planner returns an undeclared action.
        """
        plan = (
            self.operation_planner(arguments)
            if self.operation_planner is not None
            else OperationPlan(arguments=arguments)
        )
        undeclared = {operation.action for operation in plan.operations} - self.actions
        if undeclared:
            values = ", ".join(sorted(action.value for action in undeclared))
            raise ValueError(
                f"Tool '{self._name_for_error()}' planned undeclared actions: {values}."
            )
        return plan.model_copy(
            update={
                "operations": tuple(
                    operation.model_copy(update={"tool_id": self._name_for_error()})
                    for operation in plan.operations
                )
            }
        )

    def _name_for_error(self) -> str:
        """Return the resolved name or a stable fallback for diagnostics."""
        return self.name or callable_name(self.function)

    def _presentation_for(
        self,
        arguments: dict[str, Any],
        result: Any,
    ) -> ToolResultPresentationSpec:
        """Return a fixed or dynamically selected presentation, falling back safely."""
        declaration = self.result_presentation
        if isinstance(declaration, ToolResultPresentationSpec):
            return declaration
        try:
            presentation = declaration(arguments, result)
        except Exception:  # noqa: BLE001  # pylint: disable=broad-except
            return RAW_TOOL_RESULT_PRESENTATION
        return (
            presentation
            if isinstance(presentation, ToolResultPresentationSpec)
            else RAW_TOOL_RESULT_PRESENTATION
        )

    def _call_function(self, arguments: dict[str, Any], context: ToolContext | None) -> Any:
        """Call the function with an explicit context when it requests one."""
        if takes_tool_context(self.function):
            if context is None:
                raise ValueError(f"Tool '{self._name_for_error()}' requires a ToolContext.")
            return self.function(context, **arguments)
        return self.function(**arguments)

    @staticmethod
    def get_declaration(function: Callable[..., Any]) -> Tool | None:
        """Return the passive tool declaration attached to a callable."""
        declaration = getattr(function, _TOOL_ATTR, None)
        return declaration if isinstance(declaration, Tool) else None

    @staticmethod
    def set_declaration(function: Callable[..., Any], declaration: Tool) -> None:
        """Attach a passive tool declaration to a callable."""
        setattr(function, _TOOL_ATTR, declaration)


@dataclass(frozen=True)
class ToolRegistration:
    """Configure one callable for registration in a specific container.

    Args:
        function (Callable[..., Any]): Callable to expose as a tool.
        name (str | None): Container-specific name, or ``None`` to inherit or derive it.
        description (str | None): Container-specific description, or ``None`` to inherit or
            derive it.
        actions (frozenset[Action] | None): Container-specific action upper bound, or ``None``
            to inherit declared actions.
        operation_planner (OperationPlanner | None | Omit): Container-specific planner.
            Omit it to inherit a declared planner; pass ``None`` to remove one.
        result_presentation (ToolResultPresentationDeclaration | Omit): Container-specific
            presentation declaration, or omit it to inherit.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    actions: frozenset[Action] | None = None
    operation_planner: OperationPlanner | None | Omit = OMIT
    result_presentation: ToolResultPresentationDeclaration | Omit = OMIT


@overload
def tool[ToolFunction: Callable[..., Any]](function: ToolFunction, /) -> ToolFunction: ...


@overload
def tool[ToolFunction: Callable[..., Any]](
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    actions: Iterable[Action] | None = None,
    operation_planner: OperationPlanner | None = None,
    result_presentation: ToolResultPresentationDeclaration = RAW_TOOL_RESULT_PRESENTATION,
) -> Callable[[ToolFunction], ToolFunction]: ...


def tool[ToolFunction: Callable[..., Any]](
    function: ToolFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    actions: Iterable[Action] | None = None,
    operation_planner: OperationPlanner | None = None,
    result_presentation: ToolResultPresentationDeclaration = RAW_TOOL_RESULT_PRESENTATION,
) -> ToolFunction | Callable[[ToolFunction], ToolFunction]:
    """Declare a function as an LLM-callable tool without registering it.

    Args:
        function (ToolFunction | None): Function to declare when used without options.
        name (str | None): Public tool name. Defaults to the function name at registration.
        description (str | None): Public description. Defaults to the docstring summary at
            registration.
        actions (Iterable[Action] | None): Upper bound of authority-bearing effects.
            Defaults to no effects.
        operation_planner (OperationPlanner | None): Optional planner producing canonical
            arguments and typed operations.
        result_presentation (ToolResultPresentationDeclaration): Fixed presentation or selector
            applied to each successful raw result.

    Returns:
        ToolFunction | Callable[[ToolFunction], ToolFunction]: The unchanged declared function,
            or a decorator when ``function`` is omitted.
    """

    def _declare(target: ToolFunction) -> ToolFunction:
        if Tool.get_declaration(target) is not None:
            raise ToolRegistrationError(f"Tool '{callable_name(target)}' is already declared.")
        Tool.set_declaration(
            target,
            Tool(
                function=target,
                name=name,
                description=description,
                actions=frozenset(() if actions is None else actions),
                operation_planner=operation_planner,
                result_presentation=result_presentation,
            ),
        )
        return target

    return _declare(function) if function is not None else _declare
