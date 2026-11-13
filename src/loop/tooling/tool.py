"""Declare, validate, and invoke typed functions exposed to an LLM."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, NamedTuple, NoReturn, overload

from pydantic import BaseModel, ValidationError

from ..constants import OMIT, Omit
from ..errors import Problem, ProblemException, log_problem
from ..models import (
    RAW_TOOL_RESULT_PRESENTATION,
    ToolDefinition,
    ToolExecutionResult,
    ToolResultPresentationDeclaration,
    ToolResultPresentationSpec,
)
from ..permissions import Action, OperationPlan, OperationPlanner, Operations
from ..utils import (
    await_value_async,
    callable_name,
    has_running_event_loop,
    is_async_callable,
    run_in_worker_async,
    settle_future_async,
)
from .context import ToolContext
from .models import (
    OwnedToolTaskCancelled,
    ToolExecutionAdmissionRejected,
    ToolExecutionCancelled,
    ToolExecutionInterrupted,
    ToolExecutionOutcomeUnknown,
    ToolFinalInterruption,
    ToolPreflight,
)
from .utils import (
    ToolRegistrationError,
    get_tool_arguments_model,
    get_tool_description,
    get_tool_schema,
    serialize_tool_problem,
    serialize_tool_result,
    takes_tool_context,
)

_TOOL_ATTR = "__loop_tool__"
_LOGGER = logging.getLogger(__name__)
_TOOL_INVOCATION: ContextVar[object | None] = ContextVar("tool_invocation", default=None)


class ToolContinuation(Coroutine[Any, Any, Any]):
    """Track cancellation delivered to a tool without changing its coroutine semantics.

    Args:
        coroutine (Coroutine[Any, Any, Any]): Coroutine driven by the runner's task.
    """

    _coroutine: Coroutine[Any, Any, Any]
    cancellation_received: bool
    owner: asyncio.Task[Any] | None

    def __init__(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        self._coroutine = coroutine
        self.cancellation_received = False
        self.owner = None

    def __await__(self) -> Iterator[Any]:
        return iter(self)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        return self.send(None)

    def send(self, value: Any) -> Any:
        """Send a value into the wrapped coroutine.

        Args:
            value (Any): Value supplied by the task.

        Returns:
            Any: Value yielded by the wrapped coroutine.
        """
        if self.owner is None:
            self.owner = asyncio.current_task()
        return self._coroutine.send(value)

    def throw(self, *args: Any) -> Any:
        """Deliver an exception to the wrapped coroutine and remember cancellation.

        Args:
            args (tuple[Any, ...]): Exception arguments supplied by the task.

        Returns:
            Any: Value yielded after the exception is delivered.
        """
        exception = args[0]
        if isinstance(exception, asyncio.CancelledError) or (
            isinstance(exception, type) and issubclass(exception, asyncio.CancelledError)
        ):
            self.cancellation_received = True
        return self._coroutine.throw(*args)

    def close(self) -> None:
        """Close the wrapped coroutine."""
        self._coroutine.close()


def _require_owner_loop_thread(loop: asyncio.AbstractEventLoop) -> None:
    """Reject operations on a running loop from a different thread."""
    if loop.is_running():
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not loop:
            raise RuntimeError("Owned tool work requires its event-loop thread.")


class OwnedToolTask:
    """Own one tool coroutine and retain its cancellation history from its first execution.

    The coroutine remains unstarted until explicitly started on its owning event loop. Its creator
    must transfer it exclusively and must not schedule it or share it with another owner, including
    before its first execution. The owned task is created directly so an opaque task factory cannot
    detach its work from its owner. Operations on a running loop must use its thread. The creator
    must exclusively transfer the handle to one dispatcher and synchronize access while the loop is
    stopped.

    Args:
        coroutine (Coroutine[Any, Any, Any]): Unstarted native coroutine owned by this handle.
        final_result (bool): Declare that a settled result or failure is the final outcome of the
            enclosing tool, including when cancellation prevents its remaining code from running.

    Raises:
        TypeError: If the value is not a native coroutine.
        ValueError: If the coroutine has already started or finished.
    """

    _coroutine: Coroutine[Any, Any, Any] | None
    _tracked: ToolContinuation | None
    _task: asyncio.Task[Any] | None
    _claimed: bool
    _final_result: bool

    def __init__(self, coroutine: Coroutine[Any, Any, Any], *, final_result: bool = False) -> None:
        if not inspect.iscoroutine(coroutine):
            raise TypeError("Owned tool work requires a native coroutine.")
        if inspect.getcoroutinestate(coroutine) != inspect.CORO_CREATED:
            raise ValueError("Owned tool work requires an unstarted coroutine.")
        self._coroutine = coroutine
        self._tracked = None
        self._task = None
        self._claimed = False
        self._final_result = final_result

    def __await__(self) -> Iterator[Any]:
        async def await_owned_async() -> Any:
            """Claim the work and preserve settlement after cancellation."""
            task = self.claim(asyncio.get_running_loop())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError as cancellation:
                self.cancel()
                await settle_future_async(task)
                if not task.cancelled():
                    failure = task.exception()
                    if self._final_result and (failure is None or isinstance(failure, Exception)):
                        result = task.result() if failure is None else None
                        raise OwnedToolTaskCancelled(
                            result, failure, _TOOL_INVOCATION.get()
                        ) from cancellation
                    if failure is not None:
                        cancellation.add_note(
                            f"The owned operation also stopped with {type(failure).__name__}."
                        )
                raise

        return await_owned_async().__await__()

    @property
    def task(self) -> asyncio.Task[Any] | None:
        """Return the task after it has been started.

        Returns:
            asyncio.Task[Any] | None: The owned task, or ``None`` before startup.
        """
        return self._task

    def start(self, loop: asyncio.AbstractEventLoop) -> asyncio.Task[Any]:
        """Start the owned task once on its event-loop thread.

        Args:
            loop (asyncio.AbstractEventLoop): Loop that will run and own the task.

        Returns:
            asyncio.Task[Any]: The started or previously started task.

        Raises:
            RuntimeError: If the handle is closed or a running loop belongs to another thread.
            ValueError: If the task already belongs to a different loop.
        """
        if self._task is not None:
            if self._task.get_loop() is not loop:
                raise ValueError("Owned tool work belongs to a different event loop.")
            _require_owner_loop_thread(loop)
            return self._task
        if self._coroutine is None:
            raise RuntimeError("Owned tool work is closed.")
        _require_owner_loop_thread(loop)
        tracked = ToolContinuation(self._coroutine)
        continuation = await_value_async(tracked)
        try:
            task = asyncio.Task(continuation, loop=loop)
        except BaseException:
            continuation.close()
            tracked.close()
            self._coroutine = None
            raise
        self._coroutine = None
        self._tracked = tracked
        self._task = task
        task.add_done_callback(lambda _: tracked.close() if tracked.owner is None else None)
        return task

    def claim(self, loop: asyncio.AbstractEventLoop) -> asyncio.Task[Any]:
        """Transfer this handle to one dispatcher and start it if necessary.

        Args:
            loop (asyncio.AbstractEventLoop): Loop that will own the dispatch.

        Returns:
            asyncio.Task[Any]: The task owned by that dispatcher.

        Raises:
            RuntimeError: If another dispatcher already claimed the handle or startup fails.
            ValueError: If the task already belongs to a different loop.
        """
        if self._task is not None and self._task.get_loop() is not loop:
            raise ValueError("Owned tool work belongs to a different event loop.")
        _require_owner_loop_thread(loop)
        self.reserve()
        return self.start(loop)

    def reserve(self) -> None:
        """Transfer this handle to one dispatcher without starting the coroutine.

        Raises:
            RuntimeError: If another dispatcher already claimed the handle.
        """
        if self._claimed:
            raise RuntimeError("Owned tool work was already claimed.")
        self._claimed = True

    def cancel(self) -> None:
        """Request cancellation once without interrupting cleanup already in progress."""
        if self._task is None:
            self.close()
        elif (
            not self._task.done()
            and not self._task.cancelling()
            and self._tracked is not None
            and not self._tracked.cancellation_received
        ):
            _require_owner_loop_thread(self._task.get_loop())
            self._task.cancel()

    def close(self) -> None:
        """Close unstarted work without cancelling a started task."""
        if self._coroutine is not None:
            self._coroutine.close()
            self._coroutine = None


def _release_unhosted_work(result: Any, interruption: BaseException | None = None) -> None:
    """Close unstarted work or stop dispatch if started work cannot be settled."""
    owned = result if isinstance(result, OwnedToolTask) else None
    if owned is not None:
        try:
            owned.reserve()
        except RuntimeError as error:
            if interruption is not None and not isinstance(interruption, Exception):
                raise interruption from None
            raise ToolExecutionOutcomeUnknown(
                "Tool work returned a handle owned by another dispatcher."
            ) from error
    task = owned.task if owned is not None else result
    if isinstance(task, asyncio.Task) and not task.done():
        raise SystemExit("Started tool work cannot be safely settled.")
    if owned is not None:
        owned.close()
    elif inspect.iscoroutine(result):
        _close_unstarted_coroutine(result)


def _close_unstarted_coroutine(value: Any) -> None:
    """Close only a native coroutine that has not been entered by any owner."""
    if inspect.iscoroutine(value) and inspect.getcoroutinestate(value) == inspect.CORO_CREATED:
        value.close()


def _coroutine_is_borrowed(value: Any) -> bool:
    """Identify a native coroutine that has already entered or finished."""
    return inspect.iscoroutine(value) and inspect.getcoroutinestate(value) != inspect.CORO_CREATED


def _unhosted_result_is_borrowed(value: Any) -> bool:
    """Identify returned work whose state cannot be settled by this dispatcher."""
    if isinstance(value, OwnedToolTask):
        return value.task is not None
    if inspect.iscoroutine(value):
        return _coroutine_is_borrowed(value)
    return inspect.isawaitable(value)


def _start_borrowed_awaitable(value: Awaitable[Any]) -> asyncio.Task[Any]:
    """Drive a custom awaitable without transferring cancellation ownership."""
    coroutine = await_value_async(value)
    try:
        return asyncio.Task(coroutine, loop=asyncio.get_running_loop())
    except BaseException:
        coroutine.close()
        raise


def _cancel_unstarted_borrowed_task(task: asyncio.Task[Any]) -> None:
    """Prevent a new wrapper from admitting work after cancellation."""
    if not task.done() and inspect.getcoroutinestate(task.get_coro()) == inspect.CORO_CREATED:
        task.cancel()


def _settle_runner_future(
    loop: asyncio.AbstractEventLoop,
    runner: asyncio.Runner,
    future: asyncio.Future[Any],
    interruption: BaseException,
) -> None:
    """Settle one runner-owned or borrowed future without losing the primary interruption."""

    def consume_shield_error(completed: asyncio.Future[Any]) -> None:
        """Retrieve only the shield's error if its waiter was interrupted."""
        if not completed.cancelled():
            completed.exception()

    while not future.done():
        try:
            protected = asyncio.shield(future)
            protected.add_done_callback(consume_shield_error)
            loop.run_until_complete(protected)
        except asyncio.CancelledError:
            break

        except BaseException as cleanup_error:  # noqa: BLE001
            interruption.add_note(f"Tool cleanup also stopped with {type(cleanup_error).__name__}.")
            if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                continue
            if (
                isinstance(cleanup_error, RuntimeError)
                and str(cleanup_error) == "Event loop stopped before Future completed."
                and not loop.is_closed()
            ):
                continue
            if not future.done():
                try:
                    runner.close()
                except BaseException as close_error:  # noqa: BLE001
                    interruption.add_note(
                        f"Tool runner closure also stopped with {type(close_error).__name__}."
                    )
                if not loop.is_closed():
                    raise SystemExit("An unsettled tool runner could not be safely closed.") from (
                        interruption
                    )
            break


def _raise_unknown_interruption(interruption: BaseException) -> NoReturn:
    """Preserve an interrupted invocation whose returned work has no final result."""
    if isinstance(interruption, Exception):
        raise ToolExecutionOutcomeUnknown(
            "Interrupted tool work did not yield an authoritative result."
        ) from interruption
    raise interruption


def _settle_interrupted_value(
    value: Any,
    loop: asyncio.AbstractEventLoop,
    runner: asyncio.Runner,
    interruption: BaseException,
) -> Any:
    """Settle already-started returned work without admitting deferred work."""
    seen = []
    while isinstance(value, OwnedToolTask) or inspect.isawaitable(value):
        if any(value is previous for previous in seen):
            _raise_unknown_interruption(interruption)
        seen.append(value)
        owned = value if isinstance(value, OwnedToolTask) else None
        if owned is not None:
            if owned.task is not None and owned.task.get_loop() is not loop:
                _raise_unknown_interruption(interruption)
            try:
                owned.reserve()
            except RuntimeError:
                _raise_unknown_interruption(interruption)
            if owned.task is None:
                owned.close()
                _raise_unknown_interruption(interruption)
            value = owned.task
        if not isinstance(value, asyncio.Future):
            _close_unstarted_coroutine(value)
            _raise_unknown_interruption(interruption)
        if value.get_loop() is not loop:
            _raise_unknown_interruption(interruption)
        if owned is not None:
            owned.cancel()
        _settle_runner_future(loop, runner, value, interruption)
        if value.cancelled():
            _raise_unknown_interruption(interruption)
        try:
            value = value.result()
        except ToolExecutionAdmissionRejected as error:
            raise RuntimeError("Final tool work failed after it was admitted.") from error
    return value


def _require_execution_admission(admission: Callable[[], bool] | None) -> None:
    """Refuse dispatch or preserve an admission-check failure outside tool error handling."""
    if admission is None:
        return
    try:
        accepted = admission()
    except Exception as error:
        raise ToolExecutionAdmissionRejected from error
    if not accepted:
        raise ToolExecutionAdmissionRejected


class PlatformArgumentCapability(NamedTuple):
    """Bind one tool argument to a named native platform capability.

    Args:
        argument (str): Tool argument shown only when the capability is available.
        capability (str): Named capability of the live execution service.
    """

    argument: str
    capability: str


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
        result_path_fields (tuple[tuple[str, ...], ...]): Structured result fields containing local
            paths that may be displayed through logical aliases.
        preflight (ToolPreflight | None): Optional readiness check run before registration.
        required (bool): Whether a user must explicitly choose to continue when the tool is
            broken. Without an interaction, a broken required tool raises ``ToolRegistrationError``.
        execution_authorization (bool): Require an injected native execution service and a
            sandbox-bound process operation before invoking this tool.
        requires_execution_service (bool): Inject the native execution service without granting
            process execution authority.
        requires_fresh_host_approval (bool): Mark a separate explicit host shell whose plan needs
            fresh human approval or an applicable trusted no-prompt host grant.
        private_arguments (bool): Remove arguments from saved model calls and routine display.
        platform_capability (str | None): Native platform capability required to advertise this
            tool to the model; direct dispatch remains available for typed refusal.
        platform_argument_capabilities (tuple[PlatformArgumentCapability | tuple[str, str], ...]):
            Arguments and native platform capabilities omitted from unsupported model schemas.
            Bare pairs are normalized to named records during registration.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    actions: frozenset[Action] = frozenset()
    operation_planner: OperationPlanner | None = None
    arguments_model: type[BaseModel] | None = None
    result_presentation: ToolResultPresentationDeclaration = RAW_TOOL_RESULT_PRESENTATION
    result_path_fields: tuple[tuple[str, ...], ...] = ()
    preflight: ToolPreflight | None = None
    required: bool = False
    execution_authorization: bool = False
    requires_execution_service: bool = False
    requires_fresh_host_approval: bool = False
    private_arguments: bool = False
    platform_capability: str | None = None
    platform_argument_capabilities: tuple[PlatformArgumentCapability | tuple[str, str], ...] = ()

    def registered(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        actions: Iterable[Action] | None = None,
        operation_planner: OperationPlanner | None | Omit = OMIT,
        result_presentation: ToolResultPresentationDeclaration | Omit = OMIT,
        result_path_fields: tuple[tuple[str, ...], ...] | None = None,
        preflight: ToolPreflight | None | Omit = OMIT,
        required: bool | Omit = OMIT,
        execution_authorization: bool | Omit = OMIT,
        requires_execution_service: bool | Omit = OMIT,
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
            result_path_fields (tuple[tuple[str, ...], ...] | None): Declared result path fields,
                or ``None`` to inherit.
            preflight (ToolPreflight | None | Omit): Container-specific readiness check. Omit it
                to inherit; pass ``None`` to remove one.
            required (bool | Omit): Whether the tool must be available for the container to be
                considered ready. Omit it to inherit; pass ``None`` to remove one.
            execution_authorization (bool | Omit): Native execution capability for this
                registration, or omit to inherit its declaration.
            requires_execution_service (bool | Omit): Native service dependency, or omit to
                inherit its declaration.

        Returns:
            Tool: Immutable, fully resolved tool for one registry.

        Raises:
            ToolRegistrationError: If an authority-bearing declaration lacks a matching planner
                or execution capability.
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
            result_path_fields=(
                self.result_path_fields if result_path_fields is None else tuple(result_path_fields)
            ),
            preflight=self.preflight if isinstance(preflight, Omit) else preflight,
            required=self.required if isinstance(required, Omit) else required,
            execution_authorization=(
                self.execution_authorization
                if isinstance(execution_authorization, Omit)
                else execution_authorization
            ),
            requires_execution_service=(
                self.requires_execution_service
                if isinstance(requires_execution_service, Omit)
                else requires_execution_service
            ),
            arguments_model=get_tool_arguments_model(self.function, resolved_name),
        )
        if registered.actions and registered.operation_planner is None:
            raise ToolRegistrationError(
                f"Tool '{resolved_name}' declares authority-bearing actions without an "
                "operation planner."
            )
        if registered.execution_authorization and (
            Action.PROCESS_EXECUTE not in registered.actions
            or registered.operation_planner is None
            or not takes_tool_context(registered.function)
        ):
            raise ToolRegistrationError(
                f"Tool '{resolved_name}' requires a context-aware process planner "
                "for execution authorization."
            )
        if registered.requires_execution_service and not takes_tool_context(registered.function):
            raise ToolRegistrationError(
                f"Tool '{resolved_name}' requires a context-aware execution service."
            )
        if registered.requires_fresh_host_approval and (
            Action.PROCESS_EXECUTE not in registered.actions
            or not registered.requires_execution_service
            or registered.execution_authorization
        ):
            raise ToolRegistrationError(
                f"Tool '{resolved_name}' requires a separately authorized host process plan."
            )
        capabilities = []
        for item in registered.platform_argument_capabilities:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or any(not isinstance(value, str) or not value for value in item)
            ):
                raise ToolRegistrationError(
                    f"Tool '{resolved_name}' has an invalid platform argument capability."
                )
            capabilities.append(PlatformArgumentCapability(*item))
        registered = replace(registered, platform_argument_capabilities=tuple(capabilities))
        unknown_arguments = {item.argument for item in capabilities} - set(
            registered.arguments_model.model_fields
        )
        if unknown_arguments:
            raise ToolRegistrationError(
                f"Tool '{resolved_name}' has unknown capability-bound arguments: "
                f"{', '.join(sorted(unknown_arguments))}."
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

        Raises:
            ToolExecutionOutcomeUnknown: If an entered invocation has no authoritative result.
        """
        return self.execute(arguments, context).output

    def resolve_awaitable(
        self,
        result: Any,
        runner: asyncio.Runner,
        *,
        invocation: object | None = None,
    ) -> Any:
        """Resolve returned work through the runner while retaining ownership on interruption.

        Args:
            result (Any): Awaitable or explicitly owned work returned by a tool or callback.
            runner (asyncio.Runner): Runner owning synchronous coroutine dispatch.
            invocation (object | None): Identity of the tool invocation, or ``None`` for callbacks.

        Returns:
            Any: Resolved value produced by the returned work.

        Raises:
            ProblemException: If returned work belongs to a different event loop.
            RuntimeError: If owned work has already been claimed or cannot start.
            SystemExit: If unsettled work cannot be safely isolated from future dispatch.
            ToolExecutionAdmissionRejected: If final admission is refused.
            ToolExecutionOutcomeUnknown: If an interrupted or foreign-owned result cannot be
                attributed to this invocation.
            ToolFinalInterruption: If Ctrl+C settles the invocation with a known final outcome.
        """
        owned = result if isinstance(result, OwnedToolTask) else None
        try:
            loop = runner.get_loop()
        except BaseException as error:
            borrowed = _unhosted_result_is_borrowed(result)
            _release_unhosted_work(result, interruption=error)
            if invocation is not None and isinstance(error, Exception) and borrowed:
                raise ToolExecutionOutcomeUnknown(
                    "Returned tool work cannot be settled without its owner."
                ) from error
            raise
        future = owned.task if owned is not None else result
        if isinstance(future, asyncio.Future) and future.get_loop() is not loop:
            if invocation is not None:
                raise ToolExecutionOutcomeUnknown(
                    "Tool work belongs to another event loop and cannot be settled here."
                )
            raise ProblemException(
                Problem(
                    code="tool.loop_mismatch",
                    title="Tool event loop mismatch",
                    detail=(
                        f"Tool '{self._name_for_error()}' returned a result attached to a "
                        "different event loop."
                    ),
                    operation=self._name_for_error(),
                )
            )
        if owned is not None:
            try:
                owned.reserve()
            except RuntimeError as error:
                raise ToolExecutionOutcomeUnknown(
                    "Tool work returned a handle owned by another dispatcher."
                ) from error
        returned_future = result if isinstance(result, asyncio.Future) else None

        async def await_returned_async() -> Any:
            """Keep a returned future isolated from wrapper cancellation."""
            return await asyncio.shield(returned_future)

        async def await_owned_async() -> Any:
            """Start explicitly owned work on the runner's loop."""
            return await asyncio.shield(owned.start(asyncio.get_running_loop()))

        tracked = ToolContinuation(
            await_owned_async()
            if owned is not None
            else (
                await_returned_async() if returned_future is not None else await_value_async(result)
            )
        )
        runner_owner = None
        continuation = None
        continuation_allowed = True
        settled_outcome = None
        try:
            factory = loop.get_task_factory()

            async def invoke_tracked_async() -> Any:
                """Restore task creation before entering the tool coroutine."""
                nonlocal settled_outcome
                if loop.get_task_factory() is capture_owner:
                    loop.set_task_factory(factory)
                if not continuation_allowed:
                    return None
                token = _TOOL_INVOCATION.set(invocation)
                try:
                    return await tracked
                except OwnedToolTaskCancelled as outcome:
                    settled_outcome = outcome
                    raise
                finally:
                    _TOOL_INVOCATION.reset(token)

            continuation = invoke_tracked_async()

            def capture_owner(
                task_loop: asyncio.AbstractEventLoop,
                coroutine: Coroutine[Any, Any, Any],
                **kwargs: Any,
            ) -> asyncio.Future[Any]:
                """Capture the runner's task while preserving its configured factory."""
                nonlocal runner_owner
                if factory is None:
                    task = asyncio.Task(coroutine, loop=task_loop, **kwargs)
                else:
                    task = factory(task_loop, coroutine, **kwargs)
                if coroutine is continuation:
                    runner_owner = task
                return task

            loop.set_task_factory(capture_owner)
            try:
                result = runner.run(continuation)
            finally:
                if not loop.is_closed() and loop.get_task_factory() is capture_owner:
                    loop.set_task_factory(factory)
        except BaseException as interruption:
            continuation_allowed = False
            if tracked.owner is None:
                tracked.close()
                if owned is not None and owned.task is None:
                    owned.close()
                if inspect.iscoroutine(result):
                    _close_unstarted_coroutine(result)
            tasks = asyncio.all_tasks(loop)
            owner = tracked.owner or next(
                (task for task in tasks if task.get_coro() is continuation), runner_owner
            )
            if owner is not None:
                if owned is not None:
                    owned.cancel()
                if (
                    not owner.done()
                    and not (isinstance(owner, asyncio.Task) and owner.cancelling())
                    and not tracked.cancellation_received
                ):
                    owner.cancel()
                owners = []
                for candidate in (
                    owned.task if owned is not None else returned_future,
                    owner,
                    runner_owner,
                ):
                    if candidate is not None and candidate not in owners:
                        owners.append(candidate)
                for owner in owners:
                    _settle_runner_future(loop, runner, owner, interruption)
                if isinstance(interruption, ToolExecutionAdmissionRejected):
                    raise
                if (
                    isinstance(interruption, KeyboardInterrupt)
                    and invocation is not None
                    and returned_future is None
                    and tracked.owner is not None
                ):
                    known = False
                    value = None
                    failure = None
                    if (
                        runner_owner is not None
                        and runner_owner.done()
                        and not runner_owner.cancelled()
                    ):
                        try:
                            value = runner_owner.result()
                        except Exception as error:  # noqa: BLE001
                            failure = error
                        known = True
                    elif settled_outcome is not None and settled_outcome._invocation is invocation:
                        value, failure = settled_outcome.result, settled_outcome.error
                        known = True
                    if known:
                        if failure is None:
                            try:
                                value = _settle_interrupted_value(value, loop, runner, interruption)
                            except Exception as error:  # noqa: BLE001
                                failure = error
                        final = ToolFinalInterruption(value, failure)
                        for note in getattr(interruption, "__notes__", ()):
                            final.add_note(note)
                        raise final from interruption
                if isinstance(interruption, Exception):
                    if (
                        tracked.owner is not None
                        and runner_owner is not None
                        and runner_owner.done()
                        and not runner_owner.cancelled()
                    ):
                        try:
                            outer_result = runner_owner.result()
                        except ToolExecutionAdmissionRejected as error:
                            raise RuntimeError(
                                "Final tool work failed after it was admitted."
                            ) from error
                        return _settle_interrupted_value(outer_result, loop, runner, interruption)
                    if (
                        invocation is not None
                        and returned_future is None
                        and settled_outcome is not None
                        and settled_outcome._invocation is invocation
                    ):
                        if settled_outcome.error is not None:
                            if isinstance(settled_outcome.error, ToolExecutionAdmissionRejected):
                                raise RuntimeError(
                                    "Final tool work failed after it was admitted."
                                ) from settled_outcome.error
                            raise settled_outcome.error
                        return _settle_interrupted_value(
                            settled_outcome.result, loop, runner, interruption
                        )
                    receipt = owned.task if owned is not None else returned_future
                    if receipt is None:
                        receipt = tracked.owner
                    if receipt is not None and receipt.done() and not receipt.cancelled():
                        return _settle_interrupted_value(receipt, loop, runner, interruption)
                    if (
                        invocation is not None
                        and returned_future is None
                        and receipt is not None
                        and receipt.cancelled()
                    ):
                        try:
                            receipt.result()
                        except OwnedToolTaskCancelled as outcome:
                            if outcome._invocation is not invocation:
                                raise interruption
                            if outcome.error is not None:
                                if isinstance(outcome.error, ToolExecutionAdmissionRejected):
                                    raise RuntimeError(
                                        "Final tool work failed after it was admitted."
                                    ) from outcome.error
                                raise outcome.error
                            return _settle_interrupted_value(
                                outcome.result, loop, runner, interruption
                            )
                        except asyncio.CancelledError:
                            pass
            else:
                if continuation is not None:
                    continuation.close()
                tracked.close()
                if owned is not None and owned.task is None:
                    owned.close()
                if inspect.iscoroutine(result):
                    _close_unstarted_coroutine(result)
                pending = owned.task if owned is not None else returned_future
                if isinstance(pending, asyncio.Task):
                    if owned is not None and not isinstance(interruption, Exception):
                        owned.cancel()
                    _settle_runner_future(loop, runner, pending, interruption)
                    if isinstance(interruption, Exception):
                        return _settle_interrupted_value(pending, loop, runner, interruption)
            raise
        return result

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
        *,
        async_runner: asyncio.Runner | None = None,
        async_runner_factory: Callable[[], asyncio.Runner | None] | None = None,
        execution_admission: Callable[[], bool] | None = None,
    ) -> ToolExecutionResult:
        """Invoke a synchronous tool and retain its user-presentation metadata.

        Calls from an active event loop require the asynchronous API, including when the
        tool function itself is synchronous.

        Args:
            arguments (dict[str, Any]): Validated canonical keyword arguments.
            context (ToolContext | None): Runtime tool context.
            async_runner (asyncio.Runner | None): Host-owned runner for awaiting a tool result
                when synchronous dispatch is outside an active event loop.
            async_runner_factory (Callable[[], asyncio.Runner | None] | None): Lazy host-owned
                runner provider used only when a synchronous callable returns an awaitable.
            execution_admission (Callable[[], bool] | None): Optional final admission check,
                evaluated on the thread that starts the tool.

        Returns:
            ToolExecutionResult: Serialized output and selected presentation.

        Raises:
            SystemExit: If an unsettled tool's runner cannot be safely closed.
            ToolExecutionAdmissionRejected: If final admission is refused.
            ToolExecutionInterrupted: If Ctrl+C settles the tool with a known result.
            ToolExecutionOutcomeUnknown: If an entered invocation has no authoritative result.
        """
        runner = async_runner
        if has_running_event_loop() or (is_async_callable(self.function) and runner is None):
            return self._async_required_result()
        started = perf_counter()
        try:
            invocation = object()
            if is_async_callable(self.function) and execution_admission is not None:

                async def invoke_admitted_async() -> Any:
                    """Admit an async callable on the loop that starts its body."""
                    _require_execution_admission(execution_admission)
                    return await self._call_function(arguments, context)

                result = invoke_admitted_async()
            else:
                _require_execution_admission(execution_admission)
                result = self._call_function(arguments, context)
            seen_awaitables = []
            while isinstance(result, OwnedToolTask) or inspect.isawaitable(result):
                if any(result is seen for seen in seen_awaitables):
                    raise RuntimeError("Tool returned cyclic awaitables.")
                seen_awaitables.append(result)
                if _coroutine_is_borrowed(result):
                    raise ToolExecutionOutcomeUnknown(
                        "Tool coroutine has already been entered or completed by another owner."
                    )
                if runner is None and async_runner_factory is not None:
                    try:
                        runner = async_runner_factory()
                    except BaseException as error:
                        borrowed = _unhosted_result_is_borrowed(result)
                        _release_unhosted_work(result, interruption=error)
                        if isinstance(error, Exception) and borrowed:
                            raise ToolExecutionOutcomeUnknown(
                                "Returned tool work cannot be settled without its owner."
                            ) from error
                        raise
                if runner is None or has_running_event_loop():
                    borrowed = _unhosted_result_is_borrowed(result)
                    _release_unhosted_work(result)
                    if borrowed:
                        raise ToolExecutionOutcomeUnknown(
                            "Returned tool work cannot be settled without its owner."
                        )
                    return self._async_required_result()
                result = self.resolve_awaitable(result, runner, invocation=invocation)
            return ToolExecutionResult(
                serialize_tool_result(result), self._presentation_for(arguments, result)
            )
        except ToolFinalInterruption as interruption:
            try:
                if interruption.error is not None:
                    raise interruption.error
                output = serialize_tool_result(interruption.result)
            except ProblemException as error:
                output = serialize_tool_problem(error.problem)
            except Exception as error:  # noqa: BLE001  # pylint: disable=broad-except
                problem = Problem.from_exception(
                    error,
                    code="tool.execution_failed",
                    title="Tool execution failed",
                    detail=f"Tool '{self._name_for_error()}' failed.",
                    operation=self._name_for_error(),
                )
                log_problem(_LOGGER, problem, error)
                output = serialize_tool_problem(problem)
            normalized = ToolExecutionInterrupted(output, perf_counter() - started)
            for note in getattr(interruption, "__notes__", ()):
                normalized.add_note(note)
            raise normalized from interruption
        except (ToolExecutionAdmissionRejected, ToolExecutionOutcomeUnknown):
            raise
        except ProblemException as exc:
            return ToolExecutionResult(serialize_tool_problem(exc.problem))
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

    def _async_required_result(self) -> ToolExecutionResult:
        """Return the standard error for async callables without a safe blocking host."""
        return ToolExecutionResult(
            serialize_tool_problem(
                Problem(
                    code="tool.async_required",
                    title="Asynchronous tool invocation required",
                    detail=(
                        f"Tool '{self._name_for_error()}' must be called through the "
                        "asynchronous tool API."
                    ),
                    operation=self._name_for_error(),
                )
            )
        )

    async def call_async(
        self, arguments: dict[str, Any], context: ToolContext | None = None
    ) -> str:
        """Invoke a tool asynchronously with validated arguments.

        Synchronous context-free callbacks use the shared bounded worker. Context-aware
        synchronous callbacks require the registry's owner-thread relay.

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
        *,
        sync_dispatch: Callable[[Callable[[], Any]], Awaitable[Any]] | None = None,
        execution_admission: Callable[[], bool] | None = None,
        settled_cancellation: Callable[[asyncio.CancelledError], None] | None = None,
        pending_cancellation: Callable[[], asyncio.CancelledError | None] | None = None,
    ) -> ToolExecutionResult:
        """Invoke a tool asynchronously and retain its user-presentation metadata.

        The registry supplies ``sync_dispatch`` for synchronous callbacks with an owner-routed
        context. Direct context-free calls use the shared bounded worker; a direct synchronous
        callback with a live context is rejected because this method cannot relay its services.

        Args:
            arguments (dict[str, Any]): Validated canonical keyword arguments.
            context (ToolContext | None): Runtime tool context.
            sync_dispatch (Callable[[Callable[[], Any]], Any] | None): Optional registry-owned
                adapter that runs a synchronous invocation outside the event-loop thread.
            execution_admission (Callable[[], bool] | None): Optional final admission check,
                evaluated on the thread that starts the tool.
            settled_cancellation (Callable[[asyncio.CancelledError], None] | None): Receives
                cancellation when owned work settles with a known result.
            pending_cancellation (Callable[[], asyncio.CancelledError | None] | None): Returns
                an interruption retained by the registry during synchronous dispatch.

        Returns:
            ToolExecutionResult: Serialized output and selected presentation.

        Raises:
            asyncio.CancelledError: If cancellation is requested, after an admitted synchronous
                callback settles.
            ToolExecutionAdmissionRejected: If final admission is refused.
            ToolExecutionOutcomeUnknown: If an entered invocation has no authoritative result.
        """
        direct_cancellation = None
        settled_interruption = None
        started = perf_counter()
        caller = asyncio.current_task()
        prior_cancellations = caller.cancelling() if caller is not None else 0
        invocation = object()
        context_token = _TOOL_INVOCATION.set(invocation)

        def finish(value: ToolExecutionResult) -> ToolExecutionResult:
            """Propagate interruption with the receipt when no registry owns it."""
            if direct_cancellation is not None:
                raise ToolExecutionCancelled(
                    value.output, perf_counter() - started
                ) from direct_cancellation
            return value

        try:
            invoked = []

            def invoke() -> Any:
                """Admit the invocation immediately before starting the tool callable."""
                _require_execution_admission(execution_admission)
                value = self._call_function(arguments, context)
                invoked.append(value)
                return value

            if not is_async_callable(self.function):
                if sync_dispatch is not None:
                    result = await sync_dispatch(invoke)
                elif context is None:
                    try:
                        result = await run_in_worker_async(invoke, cancel_pending=True)
                    except asyncio.CancelledError as cancellation:
                        if not invoked:
                            raise
                        settled_interruption = cancellation
                        direct_cancellation = cancellation
                        result = invoked[0]
                else:
                    raise ProblemException(
                        Problem(
                            code="tool.context_relay_unavailable",
                            title="Tool context relay unavailable",
                            detail=(
                                f"Synchronous tool '{self._name_for_error()}' requires the "
                                "ToolRegistry owner-thread relay when called asynchronously."
                            ),
                            operation=self._name_for_error(),
                        )
                    )
            else:
                result = invoke()
            seen_awaitables = []
            while isinstance(result, OwnedToolTask) or inspect.isawaitable(result):
                if settled_interruption is None and pending_cancellation is not None:
                    settled_interruption = pending_cancellation()
                if any(result is seen for seen in seen_awaitables):
                    if settled_interruption is not None:
                        raise settled_interruption
                    raise RuntimeError("Tool returned cyclic awaitables.")
                seen_awaitables.append(result)
                borrowed_wrapper = None
                if inspect.iscoroutine(result):
                    if _coroutine_is_borrowed(result):
                        if settled_interruption is not None:
                            raise settled_interruption
                        raise ToolExecutionOutcomeUnknown(
                            "Tool coroutine has already been entered or completed by another owner."
                        )
                    result = OwnedToolTask(result)
                elif not isinstance(result, (OwnedToolTask, asyncio.Future)):
                    if settled_interruption is not None:
                        raise settled_interruption
                    result = _start_borrowed_awaitable(result)
                    borrowed_wrapper = result
                owned = result if isinstance(result, OwnedToolTask) else None
                if owned is not None and owned.task is None and settled_interruption is not None:
                    try:
                        owned.reserve()
                    except RuntimeError:
                        raise settled_interruption from None
                    owned.close()
                    raise settled_interruption
                if owned is not None:
                    loop = asyncio.get_running_loop()
                    if owned.task is None or owned.task.get_loop() is loop:
                        try:
                            result = owned.claim(loop)
                        except RuntimeError:
                            if settled_interruption is not None:
                                raise settled_interruption from None
                            raise ToolExecutionOutcomeUnknown(
                                "Tool work returned a handle owned by another dispatcher."
                            ) from None
                        if settled_interruption is not None:
                            owned.cancel()
                    else:
                        result = owned.task
                if result.get_loop() is not asyncio.get_running_loop():
                    if settled_interruption is not None:
                        raise settled_interruption
                    raise ToolExecutionOutcomeUnknown(
                        "Tool work belongs to another event loop and cannot be settled here."
                    )
                try:
                    result = await asyncio.shield(result)
                except ToolExecutionAdmissionRejected as error:
                    if settled_interruption is None:
                        raise
                    raise RuntimeError("Final tool work failed after it was admitted.") from error
                except asyncio.CancelledError as cancellation:
                    if (
                        caller is not None
                        and caller.cancelling() > prior_cancellations
                        and context is not None
                        and context.cancellation_event is not None
                    ):
                        context.cancellation_event.set()
                    if owned is not None:
                        owned.cancel()
                    if borrowed_wrapper is not None:
                        _cancel_unstarted_borrowed_task(borrowed_wrapper)
                    future = owned.task if owned is not None else result
                    await settle_future_async(future)
                    if future.cancelled():
                        try:
                            future.result()
                        except OwnedToolTaskCancelled as outcome:
                            caller = asyncio.current_task()
                            pending = pending_cancellation() if pending_cancellation else None
                            if (
                                owned is None
                                or outcome._invocation is not invocation
                                or (
                                    not (caller and caller.cancelling())
                                    and settled_interruption is None
                                    and pending is None
                                )
                            ):
                                raise cancellation
                            settled_interruption = cancellation
                            if settled_cancellation is not None:
                                settled_cancellation(cancellation)
                            else:
                                direct_cancellation = cancellation
                            if outcome.error is not None:
                                if isinstance(outcome.error, ToolExecutionAdmissionRejected):
                                    raise RuntimeError(
                                        "Final tool work failed after it was admitted."
                                    ) from outcome.error
                                raise outcome.error
                            result = outcome.result
                            continue
                        except asyncio.CancelledError:
                            raise cancellation
                    settled_interruption = cancellation
                    if settled_cancellation is not None:
                        settled_cancellation(cancellation)
                    else:
                        direct_cancellation = cancellation
                    try:
                        result = future.result()
                    except ToolExecutionAdmissionRejected as error:
                        raise RuntimeError(
                            "Final tool work failed after it was admitted."
                        ) from error
            return finish(
                ToolExecutionResult(
                    serialize_tool_result(result), self._presentation_for(arguments, result)
                )
            )
        except (ToolExecutionAdmissionRejected, ToolExecutionOutcomeUnknown):
            raise
        except ProblemException as exc:
            return finish(ToolExecutionResult(serialize_tool_problem(exc.problem)))
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            problem = Problem.from_exception(
                exc,
                code="tool.execution_failed",
                title="Tool execution failed",
                detail=f"Tool '{self._name_for_error()}' failed.",
                operation=self._name_for_error(),
            )
            log_problem(_LOGGER, problem, exc)
            return finish(ToolExecutionResult(serialize_tool_problem(problem)))
        finally:
            _TOOL_INVOCATION.reset(context_token)

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
        return self.normalize_plan(plan)

    def normalize_plan(self, plan: OperationPlan) -> OperationPlan:
        """Validate and bind one planner-produced phase to this registered tool.

        Args:
            plan (OperationPlan): Planner-produced phase to validate and bind.

        Returns:
            OperationPlan: Phase whose operation identities name this tool.

        Raises:
            ValueError: If the phase contains an undeclared action.
        """
        planned_operations = (
            plan.operations + plan.prerequisite_operations + plan.boundary_operations
        )
        undeclared = {operation.action for operation in planned_operations} - self.actions
        if undeclared:
            values = ", ".join(sorted(action.value for action in undeclared))
            raise ValueError(
                f"Tool '{self._name_for_error()}' planned undeclared actions: {values}."
            )
        return plan.model_copy(
            update={
                "operations": self._copy_operations(plan.operations),
                "prerequisite_operations": self._copy_operations(plan.prerequisite_operations),
                "boundary_operations": self._copy_operations(plan.boundary_operations),
            }
        )

    def _copy_operations(self, operations: Operations) -> Operations:
        """Return a copy of the operations with the tool_id updated to this tool's name."""
        return tuple(
            operation.model_copy(update={"tool_id": self._name_for_error()})
            for operation in operations
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
        preflight (ToolPreflight | None | Omit): Container-specific readiness check, or omit it
            to inherit.
        required (bool | Omit): Whether the tool must be available for the container to be
            considered ready. Omit it to inherit; pass ``None`` to remove one.
        execution_authorization (bool | Omit): Native execution capability for this
            registration, or omit to inherit its declaration.
        requires_execution_service (bool | Omit): Native service dependency, or omit to inherit.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    actions: frozenset[Action] | None = None
    operation_planner: OperationPlanner | None | Omit = OMIT
    result_presentation: ToolResultPresentationDeclaration | Omit = OMIT
    preflight: ToolPreflight | None | Omit = OMIT
    required: bool | Omit = OMIT
    execution_authorization: bool | Omit = OMIT
    requires_execution_service: bool | Omit = OMIT


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
    result_path_fields: tuple[tuple[str, ...], ...] = (),
    preflight: ToolPreflight | None = None,
    required: bool = False,
    execution_authorization: bool = False,
    requires_execution_service: bool = False,
    requires_fresh_host_approval: bool = False,
    private_arguments: bool = False,
    platform_capability: str | None = None,
    platform_argument_capabilities: tuple[PlatformArgumentCapability | tuple[str, str], ...] = (),
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
    result_path_fields: tuple[tuple[str, ...], ...] = (),
    preflight: ToolPreflight | None = None,
    required: bool = False,
    execution_authorization: bool = False,
    requires_execution_service: bool = False,
    requires_fresh_host_approval: bool = False,
    private_arguments: bool = False,
    platform_capability: str | None = None,
    platform_argument_capabilities: tuple[PlatformArgumentCapability | tuple[str, str], ...] = (),
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
        result_path_fields (tuple[tuple[str, ...], ...]): Structured result fields containing
            local paths eligible for logical display aliases.
        preflight (ToolPreflight | None): Optional readiness check run before registration.
        required (bool): Whether a user must explicitly choose to continue when the tool is
            broken. Without an interaction, a broken required tool raises ``ToolRegistrationError``.
        execution_authorization (bool): Require an injected sandbox execution service.
        requires_execution_service (bool): Inject the execution service without process authority.
        requires_fresh_host_approval (bool): Mark a separate explicit host shell requiring fresh
            human approval or an applicable trusted no-prompt host grant. Defaults to False.
        private_arguments (bool): Remove model arguments from saved calls and routine display.
        platform_capability (str | None): Native platform capability required for model exposure.
        platform_argument_capabilities (tuple[PlatformArgumentCapability | tuple[str, str], ...]):
            Named records or bare argument and native capability pairs for model exposure.

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
                result_path_fields=result_path_fields,
                preflight=preflight,
                required=required,
                execution_authorization=execution_authorization,
                requires_execution_service=requires_execution_service,
                requires_fresh_host_approval=requires_fresh_host_approval,
                private_arguments=private_arguments,
                platform_capability=platform_capability,
                platform_argument_capabilities=platform_argument_capabilities,
            ),
        )
        return target

    return _declare(function) if function is not None else _declare
