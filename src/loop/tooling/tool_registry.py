"""Register and dispatch typed functions exposed to an LLM."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from threading import Event, Lock, get_ident
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError

from ..commands.models import CommandArgumentError
from ..commands.utils import parse_model_arguments
from ..constants import (
    OMIT,
    SHELL_COMMAND_OPTION,
    SHELL_COMMAND_PATH,
    Omit,
)
from ..errors import Problem, ProblemException, log_problem
from ..execution import CommandExecutionService
from ..instructions import InstructionsManager
from ..interaction import Interaction, require_async_interaction
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
    Operations,
    PermissionManager,
    ProcessBoundary,
    ProcessTarget,
)
from ..telemetry import telemetry_audit
from ..utils import (
    VirtualPath,
    WorkerCapacity,
    WorkerEntry,
    callable_name,
    has_running_event_loop,
    is_async_callable,
    settle_future_async,
)
from .context import ToolContext
from .file_observations import FileObservations
from .models import (
    ToolExecutionAdmissionRejected,
    ToolExecutionCancelled,
    ToolExecutionInterrupted,
    ToolExecutionNotStarted,
    ToolExecutionOutcomeUnknown,
    ToolPreflight,
    ToolRegistrationError,
    ToolRuntimeSettings,
    ToolStatus,
)
from .tool import (
    OwnedToolTask,
    PlatformArgumentCapability,
    Tool,
    ToolRegistration,
    _cancel_unstarted_borrowed_task,
    _close_unstarted_coroutine,
    _release_unhosted_work,
    _start_borrowed_awaitable,
)
from .utils import serialize_tool_problem

if TYPE_CHECKING:
    from ..session import SessionManager

_LOGGER = logging.getLogger(__name__)
_TOOL_WORKER_COUNT = 4
_TOOL_WORKER_QUEUE_SIZE = 4


class ToolExecutionPortal:
    """Relay synchronous extension requests to the event-loop owner.

    Args:
        loop (asyncio.AbstractEventLoop): Running loop that owns session services.
    """

    _loop: asyncio.AbstractEventLoop
    _owner_thread: int

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._owner_thread = get_ident()

    def call(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one synchronous callback on the owner loop and return its detached result.

        Args:
            callback (Callable[..., Any]): Owner-thread operation to invoke.
            *args (Any): Positional callback arguments.
            **kwargs (Any): Keyword callback arguments.

        Returns:
            Any: Detached callback result.

        Raises:
            ProblemException: If an owner-thread callback returns an awaitable.
            RuntimeError: If the owner loop has stopped.
        """
        if self._loop.is_closed() or not self._loop.is_running():
            raise RuntimeError("Tool context relay is unavailable from this thread.")

        if get_ident() == self._owner_thread:
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                if isinstance(result, OwnedToolTask):
                    if result.task is None:
                        try:
                            result.reserve()
                        except RuntimeError:
                            pass
                        else:
                            result.close()
                elif inspect.iscoroutine(result):
                    _close_unstarted_coroutine(result)
                raise ProblemException(
                    Problem(
                        code="tool.async_required",
                        title="Asynchronous context operation required",
                        detail=(
                            "This context operation must be awaited through the asynchronous "
                            "tool API."
                        ),
                    )
                )
            return _detach_owner_value(result, self)

        async def invoke_async() -> Any:
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return _detach_owner_value(result, self)

        return asyncio.run_coroutine_threadsafe(invoke_async(), self._loop).result()

    def call_method(
        self,
        target: object,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke an owner service method, preferring its native async counterpart.

        Args:
            target (object): Service owned by the event-loop thread.
            name (str): Synchronous method requested by the extension.
            *args (Any): Positional method arguments.
            **kwargs (Any): Keyword method arguments.

        Returns:
            Any: Detached method result.

        Raises:
            RuntimeError: If the owner loop cannot service the request.
            ProblemException: If no async input method exists for a prompt request.
        """

        def invoke() -> Any:
            method = getattr(target, name)
            async_method = getattr(target, f"{name}_async", None)
            if name in {"confirm", "prompt"} and not callable(async_method):
                raise ProblemException(
                    Problem(
                        code="tool.async_required",
                        title="Asynchronous interaction required",
                        detail=(
                            "This interaction must provide asynchronous input when a tool "
                            "runs in the async application."
                        ),
                    )
                )

            if callable(async_method):
                method = async_method
            return method(*args, **kwargs)

        return self.call(invoke)

    async def call_method_async(
        self,
        target: object,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke an asynchronous owner method and return its detached result."""
        if self._loop.is_closed() or not self._loop.is_running():
            raise RuntimeError("Tool context relay is unavailable from this thread.")

        async def invoke_async() -> Any:
            """Run and await one service method on the loop owner."""
            result = getattr(target, name)(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return _detach_owner_value(result, self)

        if get_ident() == self._owner_thread:
            return await invoke_async()
        future = asyncio.run_coroutine_threadsafe(invoke_async(), self._loop)
        return await asyncio.wrap_future(future)

    def get_attribute(self, target: object, name: str) -> tuple[bool, Any]:
        """Read an owner service attribute without exposing its live value to a worker.

        Args:
            target (object): Service owned by the event-loop thread.
            name (str): Attribute to inspect.

        Returns:
            tuple[bool, Any]: Whether the attribute is callable and a safe value or proxy.
        """

        def read() -> tuple[bool, Any]:
            value = getattr(target, name)
            return callable(value), None if callable(value) else _detach_owner_value(value, self)

        return self.call(read)


class ToolOwnerServiceProxy:
    """Expose owner-thread services to a synchronous tool worker through a relay.

    Args:
        target (object): Service that must remain on the event-loop owner thread.
        portal (ToolExecutionPortal): Relay used for service access.
    """

    _target: object
    _portal: ToolExecutionPortal

    def __init__(self, target: object, portal: ToolExecutionPortal) -> None:
        self._target = target
        self._portal = portal

    def __getattr__(self, name: str) -> Any:
        callable_attribute, value = self._portal.get_attribute(self._target, name)
        if callable_attribute:
            if name.endswith("_async"):

                async def invoke_async(*args: Any, **kwargs: Any) -> Any:
                    """Call one native async service method through the owner loop."""
                    return await self._portal.call_method_async(self._target, name, *args, **kwargs)

                return invoke_async
            return lambda *args, **kwargs: self._portal.call_method(
                self._target, name, *args, **kwargs
            )
        return value


def _detach_owner_value(value: Any, portal: ToolExecutionPortal) -> Any:
    """Copy plain values and proxy service objects before returning them to a worker."""
    if isinstance(value, ToolOwnerServiceProxy):
        return value
    if value is None or isinstance(value, (str, int, float, bool, bytes, Enum, Path)):
        return value
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    if is_dataclass(value) and not isinstance(value, type):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return tuple(_detach_owner_value(item, portal) for item in value)
    if isinstance(value, list):
        return [_detach_owner_value(item, portal) for item in value]
    if isinstance(value, dict):
        return {
            _detach_owner_value(key, portal): _detach_owner_value(item, portal)
            for key, item in value.items()
        }
    if isinstance(value, (set, frozenset)):
        return type(value)(_detach_owner_value(item, portal) for item in value)
    return ToolOwnerServiceProxy(value, portal)


class ToolRegistry:
    """Register tools and dispatch each callable through the host's sync or async boundary.

    Synchronous handlers in async dispatch use bounded registry workers and receive relays for
    owner-thread services. Native async handlers run on the caller's loop. Synchronous calls may
    reuse a registry-owned runner for async handlers on its first caller thread when outside a
    running loop; ``close`` settles workers and releases the runner and execution service.

    Args:
        tools (Iterable[Callable[..., Any] | ToolRegistration] | None): Functions or configured
            registrations to add in iteration order, or ``None`` to construct an empty registry.
            Functions may optionally carry metadata from the standalone ``@tool`` decorator.
        interaction (Interaction | None): Default interaction used by context-aware tools when
            dispatch does not provide one, or ``None`` to require an invocation-specific
            interaction.
        permission_manager (PermissionManager | None): Central policy manager guarding every call.
            Defaults to an in-memory supervised policy manager.
        settings (ToolRuntimeSettings | None): Scoped settings supplied to context-aware tools, or
            ``None`` to use an independent default instance.
        execution_service (CommandExecutionService | None): Injected command execution service, or
            ``None`` to reject execution-capable tools before invocation.
        session_manager (SessionManager | None): Selected session owner for detached command
            leases, or ``None`` until an application instance binds it.
    """

    _tools: dict[str, Tool]
    _interaction: Interaction | None
    _permission_manager: PermissionManager
    _registration_problems: list[Problem]
    _settings: ToolRuntimeSettings
    _execution_service: CommandExecutionService | None
    _session_manager: SessionManager | None
    _async_runner: asyncio.Runner | None
    _runner_thread: int | None
    _runner_lock: Lock
    _executor: ThreadPoolExecutor
    _worker_slots: WorkerCapacity
    _closed: bool

    def __init__(
        self,
        tools: Iterable[Callable[..., Any] | ToolRegistration] | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
        settings: ToolRuntimeSettings | None = None,
        execution_service: CommandExecutionService | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self._tools = {}
        self._registration_problems = []
        self._interaction = interaction
        self._permission_manager = permission_manager or PermissionManager(interaction=interaction)
        self._settings = settings or ToolRuntimeSettings()
        self._execution_service = execution_service
        self._session_manager = session_manager
        self._async_runner = None
        self._runner_thread = None
        self._runner_lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=_TOOL_WORKER_COUNT,
            thread_name_prefix="loop-tool",
        )
        self._worker_slots = WorkerCapacity(_TOOL_WORKER_COUNT + _TOOL_WORKER_QUEUE_SIZE)
        self._closed = False
        for tool in tools or ():
            self.register(tool)

    def close(self) -> None:
        """Settle dispatched tools and release registry-owned execution resources.

        Raises:
            RuntimeError: If the running event loop still owns unsettled worker dispatch.
        """
        with self._runner_lock:
            if self._closed:
                return
            if has_running_event_loop() and self._worker_slots.busy:
                raise RuntimeError("Wait for tool dispatch to settle before closing on its loop.")
            self._closed = True
            async_runner = self._async_runner
            executor = self._executor
        executor.shutdown(wait=True, cancel_futures=False)
        if async_runner is not None:
            async_runner.close()
        if self._execution_service is not None:
            self._execution_service.close()

    def bind_session_manager(self, session_manager: SessionManager) -> None:
        """Bind a selected session before public command dispatch begins.

        Args:
            session_manager (SessionManager): Owner whose native lease protects detached jobs.
        """
        self._session_manager = session_manager

    def _sync_runner(self) -> asyncio.Runner | None:
        """Return the registry-owned runner only to its synchronous owner thread."""
        if has_running_event_loop():
            return None
        with self._runner_lock:
            if self._closed:
                return None
            if self._async_runner is None:
                self._async_runner = asyncio.Runner(loop_factory=asyncio.new_event_loop)
                self._runner_thread = get_ident()
            elif self._runner_thread != get_ident():
                return None
            return self._async_runner

    def _async_required(self, tool: Tool) -> ProblemException:
        """Build the structured error for callables that cannot use a sync host."""
        return ProblemException(
            Problem(
                code="tool.async_required",
                title="Asynchronous tool invocation required",
                detail=f"Tool '{tool.name}' must be called through the asynchronous tool API.",
                operation=tool.name,
            )
        )

    def _invoke(self, tool: Tool, function: Callable[..., Any], *args: Any) -> Any:
        """Invoke an extension callback directly or through the owned blocking runner."""
        runner = self._sync_runner() if is_async_callable(function) else None
        if is_async_callable(function) and runner is None:
            raise self._async_required(tool)
        result = function(*args)
        seen_awaitables = []
        while isinstance(result, OwnedToolTask) or inspect.isawaitable(result):
            if any(result is seen for seen in seen_awaitables):
                raise RuntimeError("Tool callback returned cyclic awaitables.")
            seen_awaitables.append(result)
            if runner is None:
                try:
                    runner = self._sync_runner()
                except BaseException as interruption:
                    _release_unhosted_work(result, interruption=interruption)
                    raise
            if runner is None:
                _release_unhosted_work(result)
                raise self._async_required(tool)
            result = tool.resolve_awaitable(result, runner)
        return result

    async def _invoke_async(
        self,
        tool: Tool,
        function: Callable[..., Any],
        *args: Any,
        operation: str = "tool callback",
        cancellation_event: Event | None = None,
    ) -> Any:
        """Invoke callbacks and retain cancellation while owned returned tasks settle."""
        if is_async_callable(function):
            callback = OwnedToolTask(function(*args))
            future = callback.claim(asyncio.get_running_loop())
            try:
                result = await self._await_owned_future_async(future)
            except asyncio.CancelledError:
                if cancellation_event is not None:
                    cancellation_event.set()
                await self._settle_returned_future_async(future, callback)
                if not future.cancelled():
                    failure = future.exception()
                    if failure is None:
                        await self._settle_cancelled_callback_result_async(
                            future.result(), operation
                        )
                    self._log_callback_failure(failure, operation)
                raise
        else:
            result = await self._run_worker_async(
                function,
                *args,
                cancellation_event=cancellation_event,
                operation=operation,
                cancel_pending=True,
            )
        seen_awaitables = []
        while isinstance(result, OwnedToolTask) or inspect.isawaitable(result):
            if any(result is previous for previous in seen_awaitables):
                raise RuntimeError("Tool callback returned cyclic awaitables.")
            seen_awaitables.append(result)
            borrowed_wrapper = None
            if inspect.iscoroutine(result):
                result = OwnedToolTask(result)
            elif not isinstance(result, (OwnedToolTask, asyncio.Future)):
                result = _start_borrowed_awaitable(result)
                borrowed_wrapper = result
            owned = result if isinstance(result, OwnedToolTask) else None
            if owned is not None:
                loop = asyncio.get_running_loop()
                if owned.task is None or owned.task.get_loop() is loop:
                    result = owned.claim(loop)
                else:
                    result = owned.task
            if (
                isinstance(result, asyncio.Future)
                and result.get_loop() is not asyncio.get_running_loop()
            ):
                raise ProblemException(
                    Problem(
                        code="tool.loop_mismatch",
                        title="Tool event loop mismatch",
                        detail=(
                            f"Tool '{tool.name}' returned a result attached to a different "
                            "event loop."
                        ),
                    )
                )
            try:
                result = await self._await_owned_future_async(result)
            except asyncio.CancelledError:
                if cancellation_event is not None:
                    cancellation_event.set()
                if borrowed_wrapper is not None:
                    _cancel_unstarted_borrowed_task(borrowed_wrapper)
                await self._settle_returned_future_async(result, owned)
                if not result.cancelled():
                    failure = result.exception()
                    if failure is None:
                        await self._settle_cancelled_callback_result_async(
                            result.result(), operation
                        )
                    self._log_callback_failure(failure, operation)
                raise
        return result

    async def _settle_cancelled_callback_result_async(self, result: Any, operation: str) -> None:
        """Release returned work without admitting a cancelled callback's continuation."""
        seen = []
        while isinstance(result, OwnedToolTask) or inspect.isawaitable(result):
            if any(result is previous for previous in seen):
                return
            seen.append(result)
            owned = result if isinstance(result, OwnedToolTask) else None
            if owned is not None:
                if (
                    owned.task is not None
                    and owned.task.get_loop() is not asyncio.get_running_loop()
                ):
                    return
                try:
                    owned.reserve()
                except RuntimeError:
                    return
                if owned.task is None:
                    owned.close()
                    return
                result = owned.task
            if not isinstance(result, asyncio.Future):
                if inspect.iscoroutine(result):
                    _close_unstarted_coroutine(result)
                return
            if result.get_loop() is not asyncio.get_running_loop():
                return
            await self._settle_returned_future_async(result, owned)
            if result.cancelled():
                return
            failure = result.exception()
            if failure is not None:
                self._log_callback_failure(failure, operation)
                return
            result = result.result()

    async def _run_worker_async(
        self,
        function: Callable[..., Any],
        *args: Any,
        cancellation_event: Event | None = None,
        operation: str = "tool callback",
        settled_cancellation: Callable[[asyncio.CancelledError], None] | None = None,
        cancel_pending: bool = False,
        not_started_token: object | None = None,
    ) -> Any:
        """Run a sync callback with bounded admission and settle it after cancellation."""
        token = object()
        gate = None
        future = None
        try:
            if self._closed or not self._worker_slots.acquire(token):
                unavailable = self._worker_slots.quarantined
                raise ProblemException(
                    Problem(
                        code="tool.dispatch_unavailable" if unavailable else "tool.dispatch_busy",
                        title="Tool dispatch unavailable" if unavailable else "Tool dispatch busy",
                        detail=(
                            "Worker pool unavailable after submission failure; restart Loop "
                            "or recreate the registry."
                            if unavailable
                            else (
                                "Synchronous tool capacity is full; "
                                "retry after active work settles."
                            )
                        ),
                    )
                )
            call_context = contextvars.copy_context()
            gate = WorkerEntry(self._worker_slots, token)
            operation_call = partial(call_context.run, function, *args)
            with self._runner_lock:
                if self._closed:
                    raise ProblemException(
                        Problem(
                            code="tool.dispatch_busy",
                            title="Tool dispatch busy",
                            detail="Synchronous tool capacity is full or unavailable.",
                        )
                    )
                future = gate.submit(self._executor, asyncio.get_running_loop(), operation_call)
            return await self._await_owned_future_async(future)
        except asyncio.CancelledError as cancellation:
            if future is None:
                raise
            if cancellation_event is not None:
                cancellation_event.set()
            if cancel_pending and gate is not None and gate.cancel_before_start():
                if future.cancelled():
                    gate.release()
                if settled_cancellation is not None:
                    raise ToolExecutionNotStarted(not_started_token) from cancellation
                raise
            settled = asyncio.create_task(asyncio.wait({future}))
            while not settled.done():
                try:
                    await asyncio.shield(settled)
                except asyncio.CancelledError:
                    continue
            settled.result()
            if not future.cancelled():
                worker_error = future.exception()
                if worker_error is None:
                    result = future.result()
                    owned = result if isinstance(result, OwnedToolTask) else None
                    if owned is not None:
                        if (
                            owned.task is not None
                            and owned.task.get_loop() is not asyncio.get_running_loop()
                        ):
                            result = owned.task
                        else:
                            try:
                                owned.reserve()
                            except RuntimeError:
                                raise cancellation from None
                            if owned.task is not None:
                                result = owned.task
                    if owned is not None and owned.task is None:
                        owned.close()
                    elif (
                        isinstance(result, asyncio.Future)
                        and result.get_loop() is asyncio.get_running_loop()
                    ):
                        await self._settle_returned_future_async(result, owned)
                        if not result.cancelled():
                            if settled_cancellation is not None:
                                settled_cancellation(cancellation)
                                return result.result()
                            worker_error = result.exception()
                            if worker_error is None:
                                await self._settle_cancelled_callback_result_async(
                                    result.result(), operation
                                )
                    elif inspect.isawaitable(result):
                        if inspect.iscoroutine(result):
                            _close_unstarted_coroutine(result)
                    elif settled_cancellation is not None:
                        settled_cancellation(cancellation)
                        return result
                self._log_callback_failure(worker_error, operation)
            raise
        except BaseException as interruption:
            if (
                isinstance(interruption, Exception)
                and gate is not None
                and future is None
                and gate.submission_attempted
                and not gate.submission_confirmed
            ):
                self._worker_slots.quarantine()
                problem = Problem.from_exception(
                    interruption,
                    code="tool.dispatch_unavailable",
                    title="Tool dispatch unavailable",
                    detail=(
                        "Worker pool unavailable after submission failure; restart Loop "
                        "or recreate the registry."
                    ),
                    operation=operation,
                )
                log_problem(_LOGGER, problem, interruption)
                raise ProblemException(problem) from interruption
            if (
                gate is not None
                and future is not None
                and (not cancel_pending or not gate.cancel_before_start())
            ):
                if not future.done():
                    await settle_future_async(future)
                if not future.cancelled():
                    failure = future.exception()
                    if failure is None:
                        await self._settle_cancelled_callback_result_async(
                            future.result(), operation
                        )
                    elif failure is not interruption:
                        interruption.add_note(
                            f"The blocking operation also stopped with {type(failure).__name__}."
                        )
            raise
        finally:
            if gate is None:
                self._worker_slots.release(token)
            else:
                if future is None:
                    try:
                        gate.cancel_before_start()
                    finally:
                        try:
                            if gate.submission_attempted and not gate.submission_confirmed:
                                self._worker_slots.quarantine()
                        finally:
                            if not gate.submission_confirmed:
                                self._worker_slots.release(token)
                elif cancel_pending:
                    gate.cancel_before_start()

    @staticmethod
    def _log_callback_failure(error: BaseException | None, operation: str) -> None:
        """Report an unexpected callback failure once while cancellation settles."""
        if isinstance(error, ToolExecutionAdmissionRejected):
            error = error.__cause__
        elif isinstance(error, ProblemException):
            return
        if error is None:
            return
        problem = Problem.from_exception(
            error,
            code="tool.callback_failed",
            title="Tool callback failed",
            detail=f"Tool callback '{operation}' failed while cancellation settled.",
            operation=operation,
        )
        log_problem(_LOGGER, problem, error)

    @staticmethod
    async def _await_owned_future_async(future: asyncio.Future[Any]) -> Any:
        """Wait for an owned future without cancelling it or losing a shielded error."""
        protected = asyncio.shield(future)

        def consume_error(completed: asyncio.Future[Any]) -> None:
            """Drain a result that may arrive after the waiting task is cancelled."""
            if not completed.cancelled():
                completed.exception()

        protected.add_done_callback(consume_error)
        await asyncio.wait({protected})
        return protected.result()

    @staticmethod
    async def _settle_returned_future_async(
        future: asyncio.Future[Any], owned: OwnedToolTask | None = None
    ) -> None:
        """Settle returned work, cancelling it only when ownership was explicitly transferred."""
        if owned is not None:
            owned.cancel()
        await settle_future_async(future)

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

    @property
    def settings(self) -> ToolRuntimeSettings:
        """Return settings supplied to context-aware tool invocations.

        Returns:
            ToolRuntimeSettings: Active scoped tool settings.
        """
        return self._settings

    @settings.setter
    def settings(self, settings: ToolRuntimeSettings) -> None:
        """Replace settings supplied to subsequent context-aware tool invocations.

        Args:
            settings (ToolRuntimeSettings): Replacement runtime settings.
        """
        self._settings = settings

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
    def private_argument_names(self) -> frozenset[str]:
        """Return registered names whose model arguments must never enter session history.

        Returns:
            frozenset[str]: Names carrying private argument metadata across aliases.
        """
        return frozenset(name for name, tool in self._tools.items() if tool.private_arguments)

    def view(self) -> ToolRegistryView:
        """Return a live agent-facing view of this registry.

        Returns:
            ToolRegistryView: Restricted facade backed by this registry's current catalog and
                execution path.
        """
        return ToolRegistryView(self)

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
        execution_authorization: bool | Omit = OMIT,
        requires_execution_service: bool | Omit = OMIT,
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
            execution_authorization (bool | Omit): Native execution capability for this
                registration, or omit to inherit its declaration.
            requires_execution_service (bool | Omit): Native service dependency, or omit to
                inherit its declaration.

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
            execution_authorization = (
                registration.execution_authorization
                if isinstance(execution_authorization, Omit)
                else execution_authorization
            )
            requires_execution_service = (
                registration.requires_execution_service
                if isinstance(requires_execution_service, Omit)
                else requires_execution_service
            )
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
            execution_authorization=execution_authorization,
            requires_execution_service=requires_execution_service,
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
                    or self._permission_manager.yolo is True
                    or self._interaction.prompt(
                        f"Required tool '{tool_name}' is unavailable:",
                        exit_commands=(),
                        choices={
                            "halt": "Exit application",
                            "continue": "Continue without this tool",
                        },
                        index={"halt": "Q", "continue": "C"},
                    )
                    != "continue"
                ):
                    raise ToolRegistrationError(problem.detail)
                return False
        self._tools[tool_name] = registered
        return True

    def definitions(self) -> list[ToolDefinition]:
        """Return definitions supported by the current execution service.

        Returns:
            list[ToolDefinition]: Supported function-tool definitions in registration order.
        """
        definitions = []
        for tool in self._tools.values():
            if tool.platform_capability and not self._platform_capability(tool.platform_capability):
                continue
            definition = tool.definition()
            for item in tool.platform_argument_capabilities:
                item = cast(PlatformArgumentCapability, item)
                if not self._platform_capability(item.capability):
                    definition.parameters["properties"].pop(item.argument)
                    definition.parameters["required"].remove(item.argument)
            definitions.append(definition)
        return definitions

    def _platform_capability(self, name: str) -> bool:
        """Check the live native backend's declared capability set."""
        capabilities = getattr(self._execution_service, "backend_capabilities", frozenset())
        return isinstance(capabilities, frozenset) and name in capabilities

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
        file_observations: FileObservations | None = None,
        cancellation_event: Event | None = None,
    ) -> str:
        """Dispatch a synchronous tool call from a synchronous caller by registered name.

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
            execution_started (Callable[[], None] | None): Callback invoked after authorization
                and before dispatch, for durable execution intent or interruption checks.
            file_observations (FileObservations | None): Frozen view supplied to this generation,
                or ``None`` for standalone dispatch without model context.
            cancellation_event (Event | None): Run cancellation signal passed to context-aware
                tools, or ``None`` for standalone dispatch.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ToolExecutionInterrupted: If Ctrl+C settles the tool with a known result.
            ToolExecutionOutcomeUnknown: If an entered invocation has no authoritative result.
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
            file_observations=file_observations,
            cancellation_event=cancellation_event,
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
        execution_admission: Callable[[], bool] | None = None,
        file_observations: FileObservations | None = None,
        cancellation_event: Event | None = None,
    ) -> tuple[str, float]:
        """Dispatch from a synchronous caller and measure only function execution.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation.
            instructions_manager (InstructionsManager | None): Active instruction manager.
            permission_manager (PermissionManager | None): Invocation permission policy.
            call_id (str | None): Stable model request identifier exposed to context-aware tools.
            execution_started (Callable[[], None] | None): Callback invoked after authorization
                and before dispatch, for durable execution intent or interruption checks.
            execution_admission (Callable[[], bool] | None): Optional final admission check on
                the thread that starts the tool.
            file_observations (FileObservations | None): Frozen view supplied to this generation,
                or ``None`` for standalone dispatch without model context.
            cancellation_event (Event | None): Run cancellation signal passed to context-aware
                tools, or ``None`` for standalone dispatch.

        Returns:
            tuple[str, float]: Serialized result and tool-function duration in seconds. Validation
                and authorization failures have a zero duration because no tool ran.

        Raises:
            ToolExecutionAdmissionRejected: If final admission is refused.
            ToolExecutionInterrupted: If Ctrl+C settles the tool with a known result.
            ToolExecutionOutcomeUnknown: If an entered invocation has no authoritative result.
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return self._problem(
                "tool.unknown", "Tool unavailable", f"Tool '{name}' is not available.", name
            ), 0
        if has_running_event_loop():
            return serialize_tool_problem(self._async_required(tool).problem), 0
        validated, error = tool.validate_arguments(arguments)
        if error is not None:
            return error, 0
        sync_runner = self._sync_runner() if is_async_callable(tool.function) else None
        if is_async_callable(tool.function) and sync_runner is None:
            return serialize_tool_problem(self._async_required(tool).problem), 0
        active_permissions = permission_manager or self._permission_manager
        plan, planning_error = self._authorized_plan(
            tool,
            validated,
            interaction,
            active_permissions,
            instructions_manager,
            file_observations,
        )
        if planning_error is not None:
            return self._model_result(planning_error, instructions_manager, tool), 0
        context = self._context_for(
            tool,
            interaction=interaction,
            instructions_manager=instructions_manager,
            operations=plan.operations,
            prerequisite_operations=plan.prerequisite_operations,
            call_id=call_id,
            permission_manager=active_permissions,
            cancellation_event=cancellation_event,
            file_observations=file_observations,
        )
        if sync_runner is not None:
            try:
                with self._runner_lock:
                    if self._closed:
                        return self._problem(
                            "tool.dispatch_busy",
                            "Tool dispatch busy",
                            "Synchronous tool capacity is full; retry after active work settles.",
                            tool.name or "tool",
                        ), 0
                    sync_runner.get_loop()
            except Exception as error:  # noqa: BLE001  # pylint: disable=broad-except
                # Runner setup has the same failure contract as tool execution.
                problem = Problem.from_exception(
                    error,
                    code="tool.execution_failed",
                    title="Tool execution failed",
                    detail=f"Tool '{tool.name}' failed.",
                    operation=tool.name,
                )
                log_problem(_LOGGER, problem, error)
                return serialize_tool_problem(problem), 0
        if execution_started is not None:
            execution_started()
        started = perf_counter()
        try:
            output = tool.execute(
                plan.arguments,
                context,
                async_runner=sync_runner,
                async_runner_factory=self._sync_runner,
                execution_admission=execution_admission,
            ).output
        except ToolExecutionInterrupted as interruption:
            output = self._model_result(interruption.output, instructions_manager, tool, context)
            normalized = ToolExecutionInterrupted(output, perf_counter() - started)
            for note in getattr(interruption, "__notes__", ()):
                normalized.add_note(note)
            raise normalized from interruption
        return self._model_result(
            output,
            instructions_manager,
            tool,
            context,
        ), perf_counter() - started

    async def call_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        call_id: str | None = None,
        execution_started: Callable[[], Any] | None = None,
        file_observations: FileObservations | None = None,
        cancellation_event: Event | None = None,
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
            execution_started (Callable[[], Any] | None): Callback invoked after authorization
                and before dispatch; async callbacks run on the owner loop and sync callbacks run
                in a bounded worker. Returned owner-loop tasks/futures settle on cancellation
                without starting the tool.
            file_observations (FileObservations | None): Frozen view supplied to this generation,
                or ``None`` for standalone dispatch without model context.
            cancellation_event (Event | None): Run cancellation signal passed to context-aware
                tools, or ``None`` to create a dispatch-local signal.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
            ToolExecutionCancelled: If cancellation settles a synchronous tool with a known
                normalized output and execution duration.
            ToolExecutionOutcomeUnknown: If an entered invocation has no authoritative result.
            asyncio.CancelledError: If dispatch is cancelled without an authoritative result.
        """
        output, _ = await self.call_with_timing_async(
            name,
            arguments,
            interaction=interaction,
            instructions_manager=instructions_manager,
            permission_manager=permission_manager,
            call_id=call_id,
            execution_started=execution_started,
            file_observations=file_observations,
            cancellation_event=cancellation_event,
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
        execution_started: Callable[[], Any] | None = None,
        execution_admission: Callable[[], bool] | None = None,
        file_observations: FileObservations | None = None,
        cancellation_event: Event | None = None,
    ) -> tuple[str, float]:
        """Dispatch a tool asynchronously and measure only its function execution.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation.
            instructions_manager (InstructionsManager | None): Active instruction manager.
            permission_manager (PermissionManager | None): Invocation permission policy.
            call_id (str | None): Stable model request identifier exposed to context-aware tools.
            execution_started (Callable[[], Any] | None): Callback invoked after authorization
                and before dispatch; async callbacks run on the owner loop and sync callbacks run
                in a bounded worker. Returned owner-loop tasks/futures settle on cancellation
                without starting the tool.
            execution_admission (Callable[[], bool] | None): Optional final admission check on
                the thread that starts the tool.
            file_observations (FileObservations | None): Frozen view supplied to this generation,
                or ``None`` for standalone dispatch without model context.
            cancellation_event (Event | None): Run cancellation signal passed to context-aware
                tools, or ``None`` to create a dispatch-local signal.

        Returns:
            tuple[str, float]: Serialized result and tool-function duration in seconds. Validation
                and authorization failures have a zero duration because no tool ran.

        Raises:
            ToolExecutionAdmissionRejected: If final admission is refused.
            ValueError: If the tool requires a context but none is provided.
            ToolExecutionCancelled: If cancellation settles a synchronous tool with a known
                normalized output and execution duration.
            ToolExecutionOutcomeUnknown: If an entered invocation has no authoritative result.
            asyncio.CancelledError: If dispatch is cancelled without an authoritative result.
        """
        tool = self._tools.get(name)
        if tool is None:
            return self._problem(
                "tool.unknown", "Tool unavailable", f"Tool '{name}' is not available.", name
            ), 0
        validated, error = tool.validate_arguments(arguments)
        if error is not None:
            return error, 0
        virtual_paths = getattr(instructions_manager, "virtual_paths", None)
        redaction_roots = None
        if virtual_paths is not None:
            try:
                redaction_roots = await self._run_worker_async(
                    virtual_paths.redaction_roots,
                    operation=f"{tool.name or 'tool'} path redaction preparation",
                )
            except ProblemException as exc:
                return serialize_tool_problem(exc.problem), 0
        active_permissions = permission_manager or self._permission_manager
        plan, planning_error = await self._authorized_plan_async(
            tool,
            validated,
            interaction,
            active_permissions,
            instructions_manager,
            file_observations,
        )
        if planning_error is not None:
            return self._model_result(
                planning_error,
                instructions_manager,
                tool,
                virtual_paths=virtual_paths,
                redaction_roots=redaction_roots,
            ), 0
        dispatch_cancellation = cancellation_event if cancellation_event is not None else Event()
        context = self._context_for(
            tool,
            interaction=interaction,
            instructions_manager=instructions_manager,
            operations=plan.operations,
            prerequisite_operations=plan.prerequisite_operations,
            call_id=call_id,
            permission_manager=active_permissions,
            cancellation_event=dispatch_cancellation,
            file_observations=file_observations,
        )
        if execution_started is not None:
            await self._invoke_async(
                tool,
                execution_started,
                operation=f"{tool.name} execution-start callback",
                cancellation_event=cancellation_event,
            )
        started = perf_counter()
        worker_context = None
        active_context = context
        cancellation = None
        not_started_token = object()

        def retain_cancellation(error: asyncio.CancelledError) -> None:
            """Retain interruption while a completed synchronous result is normalized."""
            nonlocal cancellation
            cancellation = error

        if not is_async_callable(tool.function):
            portal = ToolExecutionPortal(asyncio.get_running_loop())
            worker_context = self._worker_context(
                context,
                portal,
                lambda dynamic_arguments: self._authorized_plan_additional_async(
                    tool,
                    dynamic_arguments,
                    interaction,
                    active_permissions,
                    instructions_manager,
                    file_observations,
                ),
            )
            active_context = worker_context
        try:

            async def dispatch_sync_async(invoke: Callable[[], Any]) -> Any:
                caller = asyncio.current_task()
                prior_cancellations = caller.cancelling() if caller is not None else 0
                value = await self._run_worker_async(
                    invoke,
                    cancellation_event=active_context.cancellation_event,
                    operation=tool.name or "tool",
                    settled_cancellation=retain_cancellation,
                    cancel_pending=True,
                    not_started_token=not_started_token,
                )
                if cancellation is not None:
                    return value
                owned = value if isinstance(value, OwnedToolTask) else None
                if owned is not None:
                    loop = asyncio.get_running_loop()
                    if owned.task is None or owned.task.get_loop() is loop:
                        try:
                            value = owned.claim(loop)
                        except RuntimeError as error:
                            raise ToolExecutionOutcomeUnknown(
                                "Tool work returned a handle owned by another dispatcher."
                            ) from error
                    else:
                        value = owned.task
                if (
                    isinstance(value, asyncio.Future)
                    and value.get_loop() is asyncio.get_running_loop()
                ):
                    try:
                        return await self._await_owned_future_async(value)
                    except asyncio.CancelledError as error:
                        if caller is not None and caller.cancelling() > prior_cancellations:
                            dispatch_cancellation.set()
                        await self._settle_returned_future_async(value, owned)
                        if value.cancelled():
                            raise
                        retain_cancellation(error)
                        return value.result()
                return value

            try:
                result = await tool.execute_async(
                    plan.arguments,
                    active_context,
                    sync_dispatch=dispatch_sync_async,
                    execution_admission=execution_admission,
                    settled_cancellation=retain_cancellation,
                    pending_cancellation=lambda: cancellation,
                )
            except ToolExecutionNotStarted as error:
                if error._invocation is not not_started_token:
                    raise asyncio.CancelledError from error
                raise
        finally:
            if worker_context is not None:
                context.file_versions.update(worker_context.file_versions)
                context.read_versions.update(worker_context.read_versions)
        output = result.output
        output = self._model_result(
            output,
            instructions_manager,
            tool,
            context,
            virtual_paths=virtual_paths,
            redaction_roots=redaction_roots,
        )
        duration = perf_counter() - started
        if cancellation is not None:
            raise ToolExecutionCancelled(output, duration) from cancellation
        return output, duration

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
        plan, planning_error = self._command_plan(tool, validated, instructions_manager)
        if planning_error is not None:
            return ToolExecutionResult(planning_error)
        context = self._context_for(
            tool,
            interaction=interaction,
            instructions_manager=instructions_manager,
            operations=plan.operations,
            prerequisite_operations=plan.prerequisite_operations,
            permission_manager=(self._permission_manager if tool.execution_authorization else None),
        )
        result = tool.execute(
            plan.arguments,
            context,
            async_runner=(self._sync_runner() if is_async_callable(tool.function) else None),
            async_runner_factory=self._sync_runner,
        )
        return ToolExecutionResult(
            self._model_result(result.output, instructions_manager, tool, context),
            result.presentation,
        )

    @staticmethod
    def _model_result(
        output: str,
        instructions_manager: InstructionsManager | None,
        tool: Tool,
        context: ToolContext | None = None,
        *,
        virtual_paths: VirtualPath | None = None,
        redaction_roots: tuple[tuple[str, str], ...] | None = None,
    ) -> str:
        """Minimize result paths and attach trusted file read and mutation receipts."""
        if context is not None and (context.file_versions or context.read_versions):
            value = json.loads(output)
            if value.get("ok") is True and context.file_versions:
                value["file_versions"] = context.file_versions
            if context.read_versions and (
                value.get("ok") is True
                or value.get("problem", {}).get("code") == "filesystem.path_missing"
            ):
                value["read_versions"] = context.read_versions
            output = json.dumps(value, ensure_ascii=False)
        if instructions_manager is None:
            return output
        try:
            value = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return output
        problem = value.get("problem") if isinstance(value, dict) else None
        problem_detail = problem.get("detail") if isinstance(problem, dict) else None
        if not tool.result_path_fields and not isinstance(problem_detail, str):
            return output
        paths = virtual_paths or instructions_manager.virtual_paths
        prepared = (
            paths.metadata(value, tool.result_path_fields) if tool.result_path_fields else value
        )
        problem = prepared.get("problem") if isinstance(prepared, dict) else None
        if isinstance(problem, dict) and isinstance(problem.get("detail"), str):
            prepared = {
                **prepared,
                "problem": {
                    **problem,
                    "detail": paths.redact(problem["detail"], roots=redaction_roots),
                },
            }
        return output if value == prepared else json.dumps(prepared, ensure_ascii=False)

    def _plan(self, tool: Tool, arguments: dict[str, object]) -> OperationPlan:
        """Run and normalize one planner through the synchronous host policy."""
        if tool.operation_planner is None:
            return tool.plan(arguments)
        planned = self._invoke(tool, tool.operation_planner, copy.deepcopy(arguments))
        return tool.normalize_plan(planned)

    async def _plan_async(self, tool: Tool, arguments: dict[str, object]) -> OperationPlan:
        """Run and normalize one planner through async or bounded-worker dispatch."""
        if tool.operation_planner is None:
            return tool.plan(arguments)
        planned = await self._invoke_async(
            tool,
            tool.operation_planner,
            copy.deepcopy(arguments),
            operation=tool.name or "tool planner",
        )
        return tool.normalize_plan(planned)

    def _authorized_plan(
        self,
        tool: Tool,
        arguments: dict[str, object],
        interaction: Interaction | None,
        permission_manager: PermissionManager,
        instructions_manager: InstructionsManager | None = None,
        file_observations: FileObservations | None = None,
    ) -> tuple[OperationPlan | None, str | None]:
        """Resolve and authorize every phase of one operation plan."""
        try:
            if instructions_manager is not None:
                arguments = self._resolve_model_paths(tool, arguments, instructions_manager)
            plan = self._plan(tool, arguments)
            prerequisites = ()
            while True:
                boundary_denial = permission_manager.check_boundaries(plan.boundary_operations)
                if boundary_denial is not None:
                    return None, self._denied_problem(tool, boundary_denial.reason)
                if file_observations is not None:
                    absence_receipts = {}
                    conflict = file_observations.check(
                        plan.boundary_operations + plan.operations,
                        plan.arguments,
                        authorized_reads=prerequisites,
                        absence_receipts=absence_receipts,
                    )
                    if conflict is not None:
                        output = serialize_tool_problem(conflict)
                        if absence_receipts:
                            value = json.loads(output)
                            value["read_versions"] = absence_receipts
                            output = json.dumps(value, ensure_ascii=False)
                        return None, output
                denied = self._authorize(
                    tool,
                    plan.operations,
                    interaction,
                    permission_manager,
                    instructions_manager,
                )
                if denied is not None:
                    return None, denied
                if plan.continuation is None:
                    return plan.model_copy(
                        update={
                            "prerequisite_operations": (
                                prerequisites + plan.prerequisite_operations
                            )
                        }
                    ), None
                prerequisites += plan.operations
                plan = tool.normalize_plan(self._invoke(tool, plan.continuation))
        except ProblemException as exc:
            return None, serialize_tool_problem(exc.problem)
        except (OSError, ValueError) as exc:
            return None, self._problem(
                "tool.planning_failed", "Tool planning failed", str(exc), tool.name or "tool"
            )

    async def _authorized_plan_async(
        self,
        tool: Tool,
        arguments: dict[str, object],
        interaction: Interaction | None,
        permission_manager: PermissionManager,
        instructions_manager: InstructionsManager | None = None,
        file_observations: FileObservations | None = None,
    ) -> tuple[OperationPlan | None, str | None]:
        """Resolve and authorize every phase of one operation plan asynchronously."""
        try:
            if instructions_manager is not None:
                arguments = self._resolve_model_paths(tool, arguments, instructions_manager)
            plan = await self._plan_async(tool, arguments)
            prerequisites = ()
            while True:
                boundary_denial = await permission_manager.check_boundaries_async(
                    plan.boundary_operations
                )
                if boundary_denial is not None:
                    return None, self._denied_problem(tool, boundary_denial.reason)
                if file_observations is not None:
                    absence_receipts = {}
                    conflict = await self._run_worker_async(
                        partial(
                            file_observations.check,
                            plan.boundary_operations + plan.operations,
                            plan.arguments,
                            authorized_reads=prerequisites,
                            absence_receipts=absence_receipts,
                        ),
                        operation=f"{tool.name or 'tool'} file observation check",
                    )
                    if conflict is not None:
                        output = serialize_tool_problem(conflict)
                        if absence_receipts:
                            value = json.loads(output)
                            value["read_versions"] = absence_receipts
                            output = json.dumps(value, ensure_ascii=False)
                        return None, output
                denied = await self._authorize_async(
                    tool,
                    plan.operations,
                    interaction,
                    permission_manager,
                    instructions_manager,
                )
                if denied is not None:
                    return None, denied
                if plan.continuation is None:
                    return plan.model_copy(
                        update={
                            "prerequisite_operations": (
                                prerequisites + plan.prerequisite_operations
                            )
                        }
                    ), None
                prerequisites += plan.operations
                plan = tool.normalize_plan(
                    await self._invoke_async(
                        tool,
                        plan.continuation,
                        operation=tool.name or "tool continuation",
                    )
                )
        except ProblemException as exc:
            return None, serialize_tool_problem(exc.problem)
        except (OSError, ValueError) as exc:
            return None, self._problem(
                "tool.planning_failed", "Tool planning failed", str(exc), tool.name or "tool"
            )

    def _resolve_model_paths(
        self,
        tool: Tool,
        arguments: dict[str, object],
        instructions_manager: InstructionsManager,
    ) -> dict[str, object]:
        """Resolve typed model paths through the virtual-path boundary."""
        resolved = {}
        for name, value in arguments.items():
            metadata = tool.arguments_model.model_fields[name].metadata
            if VirtualPath.METADATA_KEY in metadata:
                resolved[name] = instructions_manager.virtual_paths.resolve(str(value))
            else:
                resolved[name] = value
        return resolved

    def _command_plan(
        self,
        tool: Tool,
        arguments: dict[str, object],
        instructions_manager: InstructionsManager | None = None,
    ) -> tuple[OperationPlan | None, str | None]:
        """Resolve every planning phase for a direct user command."""
        try:
            if instructions_manager is not None:
                arguments = self._resolve_model_paths(tool, arguments, instructions_manager)
            plan = self._plan(tool, arguments)
            prerequisites = ()
            while plan.continuation is not None:
                prerequisites += plan.operations
                plan = tool.normalize_plan(self._invoke(tool, plan.continuation))
            return plan.model_copy(
                update={"prerequisite_operations": prerequisites + plan.prerequisite_operations}
            ), None
        except ProblemException as exc:
            return None, serialize_tool_problem(exc.problem)
        except (OSError, ValueError) as exc:
            return None, self._problem(
                "tool.planning_failed", "Tool planning failed", str(exc), tool.name or "tool"
            )

    def _authorized_plan_additional(
        self,
        tool: Tool,
        arguments: dict[str, object],
        interaction: Interaction | None,
        permission_manager: PermissionManager,
        instructions_manager: InstructionsManager | None,
        file_observations: FileObservations | None,
    ) -> OperationPlan:
        """Authorize runtime-discovered effects through the regular sync policy path."""
        plan, error = self._authorized_plan(
            tool,
            arguments,
            interaction,
            permission_manager,
            instructions_manager,
            file_observations,
        )
        if error is not None:
            raise ProblemException(self._additional_problem_from_output(error, tool))
        return cast(OperationPlan, plan)

    async def _authorized_plan_additional_async(
        self,
        tool: Tool,
        arguments: dict[str, object],
        interaction: Interaction | None,
        permission_manager: PermissionManager,
        instructions_manager: InstructionsManager | None,
        file_observations: FileObservations | None,
    ) -> OperationPlan:
        """Authorize runtime-discovered effects through the regular async policy path."""
        plan, error = await self._authorized_plan_async(
            tool,
            arguments,
            interaction,
            permission_manager,
            instructions_manager,
            file_observations,
        )
        if error is not None:
            raise ProblemException(self._additional_problem_from_output(error, tool))
        return cast(OperationPlan, plan)

    @staticmethod
    def _problem_from_output(output: str) -> Problem:
        """Recover a structured problem from an already-serialized planning failure."""
        return Problem.model_validate(json.loads(output)["problem"])

    @classmethod
    def _additional_problem_from_output(cls, output: str, tool: Tool) -> Problem:
        """Retain the additional-operation error contract after full policy revalidation."""
        problem = cls._problem_from_output(output)
        if problem.code != "tool.denied":
            return problem
        reason = problem.detail.partition(": ")[2] or problem.detail
        return problem.model_copy(
            update={
                "title": "Additional operation denied",
                "detail": f"Tool '{tool.name}' stopped: {reason}",
            }
        )

    @staticmethod
    def _worker_context(
        context: ToolContext,
        portal: ToolExecutionPortal,
        additional_authorizer: (Callable[[dict[str, object]], Awaitable[OperationPlan]] | None),
    ) -> ToolContext:
        """Give a worker detached values and owner-thread proxies for session services."""
        return ToolContext(
            interaction=(
                ToolOwnerServiceProxy(context.interaction, portal)
                if context.interaction is not None
                else None
            ),
            tool_name=context.tool_name,
            instructions_manager=(
                ToolOwnerServiceProxy(context.instructions_manager, portal)
                if context.instructions_manager is not None
                else None
            ),
            operations=copy.deepcopy(context.operations),
            prerequisite_operations=copy.deepcopy(context.prerequisite_operations),
            call_id=context.call_id,
            additional_authorizer=(
                lambda arguments: (
                    portal.call(additional_authorizer, copy.deepcopy(arguments))
                    if additional_authorizer is not None
                    else None
                )
            ),
            settings=copy.deepcopy(context.settings),
            permission_manager=(
                ToolOwnerServiceProxy(context.permission_manager, portal)
                if context.permission_manager is not None
                else None
            ),
            execution_service=context.execution_service,
            session_manager=context.session_manager,
            file_versions={},
            read_versions={},
            cancellation_event=context.cancellation_event,
        )

    def _authorize(
        self,
        tool: Tool,
        operations: Operations,
        interaction: Interaction | None,
        permission_manager: PermissionManager,
        instructions_manager: InstructionsManager | None,
    ) -> str | None:
        """Return a serialized denial or authorize the complete operation plan."""
        if tool.requires_fresh_host_approval:
            return self._authorize_host(tool, operations, interaction, permission_manager)
        if tool.execution_authorization:
            return self._authorization_unavailable(tool, operations)

        active_interaction = interaction if interaction is not None else self._interaction
        result = permission_manager.authorize(
            operations,
            interaction=active_interaction,
            protected_instruction_names=(
                instructions_manager.agents_filenames if instructions_manager is not None else ()
            ),
        )

        if result.decision is Decision.DENY:
            return self._denied_problem(tool, result.reason)
        return None

    async def _authorize_async(
        self,
        tool: Tool,
        operations: Operations,
        interaction: Interaction | None,
        permission_manager: PermissionManager,
        instructions_manager: InstructionsManager | None,
    ) -> str | None:
        """Return a serialized denial or asynchronously authorize the complete operation plan."""
        if tool.requires_fresh_host_approval:
            return await self._authorize_host_async(
                tool, operations, interaction, permission_manager
            )
        if tool.execution_authorization:
            return self._authorization_unavailable(tool, operations)

        active_interaction = interaction if interaction is not None else self._interaction
        result = await permission_manager.authorize_async(
            operations,
            interaction=active_interaction,
            protected_instruction_names=(
                instructions_manager.agents_filenames if instructions_manager is not None else ()
            ),
        )

        if result.decision is Decision.DENY:
            return self._denied_problem(tool, result.reason)
        return None

    @staticmethod
    def _explicit_host_operation(operations: Operations) -> Operation | None:
        """Require the registered explicit tool's exact single host shell effect."""
        if len(operations) != 1:
            return None
        operation = operations[0]
        if operation.action is not Action.PROCESS_EXECUTE:
            return None
        target = operation.target
        if (
            not isinstance(target, ProcessTarget)
            or target.boundary is not ProcessBoundary.HOST
            or len(target.argv) != 3
            or target.argv[:2] != (SHELL_COMMAND_PATH, SHELL_COMMAND_OPTION)
        ):
            return None
        return operation

    @staticmethod
    def _trusted_host_rule(operation: Operation, permission_manager: PermissionManager) -> bool:
        """Match only one exact explicit host process against human session grants."""
        target = cast(ProcessTarget, operation.target)
        return bool(
            permission_manager.matching_trusted_host_command_rule(
                target.argv[2], Path(target.cwd), operation.tool_id
            )
        )

    @staticmethod
    def _record_host_denial(
        permission_manager: PermissionManager, operation: Operation | None, source: str
    ) -> None:
        """Persist a minimized denial before the host coordinator receives a request."""
        target = operation.target if operation is not None else None
        command = target.argv[2] if isinstance(target, ProcessTarget) else ""
        permission_manager.record_host_boundary("host_command.denied", command, source=source)
        telemetry_audit("host_command.denied", source=source)

    def _authorize_host(
        self,
        tool: Tool,
        operations: Operations,
        interaction: Interaction | None,
        permission_manager: PermissionManager,
    ) -> str | None:
        """Admit an explicit host request without a generic process approval prompt."""
        active_interaction = interaction if interaction is not None else self._interaction
        operation = self._explicit_host_operation(operations)
        if operation is None:
            self._record_host_denial(permission_manager, None, "invalid_plan")
            return self._problem(
                "host_command.invalid_request",
                "Host command unavailable",
                "An exact bound host shell operation is required.",
                tool.name or "tool",
            )
        policy = permission_manager.evaluate(operations)
        if policy.decision is Decision.DENY:
            revision = permission_manager.host_gate_enable_revision(operations)
            if (
                revision is None
                or active_interaction is None
                or (getattr(active_interaction, "supports_fresh_host_approval", False) is not True)
            ):
                self._record_host_denial(permission_manager, operation, "policy")
                return self._denied_problem(tool, policy.reason)
            active_interaction.info(
                "Enable host-process eligibility for this session? This allows future explicit "
                "host requests to ask for approval; it does not approve any command."
            )
            answer = active_interaction.prompt(
                "Enable session host-process eligibility?",
                exit_commands=None,
                choices={"deny": "Deny", "enable": "Enable for this session"},
                index={"deny": "N", "enable": "Y"},
            )
            if answer != "enable" or not permission_manager.enable_host_gate_if_current(
                operations, revision
            ):
                self._record_host_denial(
                    permission_manager, operation, "eligibility_declined_or_stale"
                )
                return self._denied_problem(tool, "Host eligibility was declined or changed.")
            policy = permission_manager.evaluate(operations)
        if policy.decision is Decision.DENY:
            self._record_host_denial(permission_manager, operation, "policy")
            return self._denied_problem(tool, policy.reason)
        if not self._trusted_host_rule(operation, permission_manager) and (
            permission_manager.yolo is not True
            and (
                active_interaction is None
                or getattr(active_interaction, "supports_fresh_host_approval", False) is not True
            )
        ):
            self._record_host_denial(permission_manager, operation, "fresh_input_unavailable")
            return self._problem(
                "host_command.denied",
                "Host command denied",
                "Fresh interactive host approval is unavailable.",
                tool.name or "tool",
            )
        return None

    async def _authorize_host_async(
        self,
        tool: Tool,
        operations: Operations,
        interaction: Interaction | None,
        permission_manager: PermissionManager,
    ) -> str | None:
        """Admit a host plan with native asynchronous gate input when needed."""
        active_interaction = interaction if interaction is not None else self._interaction
        operation = self._explicit_host_operation(operations)
        if operation is None:
            await self._run_worker_async(
                partial(self._record_host_denial, permission_manager, None, "invalid_plan"),
                operation="host denial audit",
            )
            return self._problem(
                "host_command.invalid_request",
                "Host command unavailable",
                "An exact bound host shell operation is required.",
                tool.name or "tool",
            )
        policy = await permission_manager.evaluate_async(operations)
        if policy.decision is Decision.DENY:
            revision = await self._run_worker_async(
                partial(permission_manager.host_gate_enable_revision, operations),
                operation="host gate eligibility",
            )
            if (
                revision is None
                or active_interaction is None
                or (getattr(active_interaction, "supports_fresh_host_approval", False) is not True)
            ):
                await self._run_worker_async(
                    partial(self._record_host_denial, permission_manager, operation, "policy"),
                    operation="host denial audit",
                )
                return self._denied_problem(tool, policy.reason)
            active_interaction.info(
                "Enable host-process eligibility for this session? This allows future explicit "
                "host requests to ask for approval; it does not approve any command."
            )
            answer = await require_async_interaction(
                active_interaction, "host eligibility"
            ).prompt_async(
                "Enable session host-process eligibility?",
                exit_commands=None,
                choices={"deny": "Deny", "enable": "Enable for this session"},
                index={"deny": "N", "enable": "Y"},
            )
            if answer != "enable":
                await self._run_worker_async(
                    partial(
                        self._record_host_denial,
                        permission_manager,
                        operation,
                        "eligibility_declined",
                    ),
                    operation="host denial audit",
                )
                return self._denied_problem(tool, "Host eligibility was declined.")
            enabled = await self._run_worker_async(
                partial(permission_manager.enable_host_gate_if_current, operations, revision),
                operation="host gate enablement",
            )
            if not enabled:
                await self._run_worker_async(
                    partial(
                        self._record_host_denial,
                        permission_manager,
                        operation,
                        "eligibility_stale",
                    ),
                    operation="host denial audit",
                )
                return self._denied_problem(tool, "Host eligibility was declined or changed.")
            policy = await permission_manager.evaluate_async(operations)
        if policy.decision is Decision.DENY:
            await self._run_worker_async(
                partial(self._record_host_denial, permission_manager, operation, "policy"),
                operation="host denial audit",
            )
            return self._denied_problem(tool, policy.reason)
        trusted = await self._run_worker_async(
            partial(self._trusted_host_rule, operation, permission_manager),
            operation="trusted host policy",
        )
        if not trusted and (
            permission_manager.yolo is not True
            and (
                active_interaction is None
                or getattr(active_interaction, "supports_fresh_host_approval", False) is not True
            )
        ):
            await self._run_worker_async(
                partial(
                    self._record_host_denial,
                    permission_manager,
                    operation,
                    "fresh_input_unavailable",
                ),
                operation="host denial audit",
            )
            return self._problem(
                "host_command.denied",
                "Host command denied",
                "Fresh interactive host approval is unavailable.",
                tool.name or "tool",
            )
        return None

    def _authorization_unavailable(
        self,
        tool: Tool,
        operations: Operations,
    ) -> str | None:
        """Return a serialized problem if authorization is unavailable."""
        if self._execution_service is None:
            return self._problem(
                "sandbox.unavailable",
                "Sandbox unavailable",
                "Native execution service is unavailable.",
                tool.name,
            )
        if not operations or any(
            operation.action is not Action.PROCESS_EXECUTE
            or not isinstance(operation.target, ProcessTarget)
            or operation.target.boundary is not ProcessBoundary.SANDBOXED
            for operation in operations
        ):
            return self._problem(
                "sandbox.unavailable",
                "Sandbox unavailable",
                "Native execution request is not bound to a sandbox.",
                tool.name,
            )
        return None

    @staticmethod
    def _denied_problem(tool: Tool, reason: str) -> str:
        """Serialize one centralized authorization denial."""
        return serialize_tool_problem(
            Problem(
                code="tool.denied",
                title="Tool call denied",
                detail=f"Tool '{tool.name}' was not executed: {reason}",
                severity="warning",
                operation=tool.name,
            )
        )

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
        operations: Operations = (),
        prerequisite_operations: Operations = (),
        call_id: str | None = None,
        permission_manager: PermissionManager | None = None,
        cancellation_event: Event | None = None,
        file_observations: FileObservations | None = None,
    ) -> ToolContext | None:
        """Build a tool context from the invocation override or registry default."""
        if interaction is None:
            interaction = self._interaction

        def authorize_additional(arguments: dict[str, object]) -> OperationPlan:
            """Plan and authorize one runtime-discovered operation set."""
            return self._authorized_plan_additional(
                tool,
                arguments,
                interaction,
                permission_manager,
                instructions_manager,
                file_observations,
            )

        async def authorize_additional_async(
            arguments: dict[str, object],
        ) -> OperationPlan:
            """Plan and authorize dynamic operations through native async policy APIs."""
            return await self._authorized_plan_additional_async(
                tool,
                arguments,
                interaction,
                permission_manager,
                instructions_manager,
                file_observations,
            )

        return ToolContext(
            interaction=interaction,
            tool_name=tool.name,
            call_id=call_id,
            instructions_manager=instructions_manager,
            operations=operations,
            prerequisite_operations=prerequisite_operations,
            additional_authorizer=(
                authorize_additional if permission_manager is not None else None
            ),
            settings=self._settings,
            permission_manager=permission_manager,
            execution_service=self._execution_service
            if tool.execution_authorization or tool.requires_execution_service
            else None,
            session_manager=self._session_manager,
            cancellation_event=cancellation_event,
            additional_authorizer_async=(
                authorize_additional_async if permission_manager is not None else None
            ),
            execution_mode="async" if is_async_callable(tool.function) else "sync",
        )


class ToolRegistryView:
    """Expose one live tool registry to an agent without duplicating its state.

    Args:
        registry (ToolRegistry): Registry owning declarations and runtime execution dependencies.
    """

    _registry: ToolRegistry

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        """Return the single registry backing this live view.

        Returns:
            ToolRegistry: Registry owning the tool catalog and execution dependencies.
        """
        return self._registry

    @property
    def tools(self) -> list[Tool]:
        """Return the registry's currently registered tools.

        Returns:
            list[Tool]: Registered tools sorted by name.
        """
        return self._registry.tools

    @property
    def private_argument_names(self) -> frozenset[str]:
        """Expose the registry's argument privacy policy through its live view.

        Returns:
            frozenset[str]: Registered private-argument tool names.
        """
        return self._registry.private_argument_names

    @property
    def names(self) -> list[str]:
        """Return the registry's current tool names.

        Returns:
            list[str]: Registered tool names sorted alphabetically.
        """
        return self._registry.names

    @property
    def registration_problems(self) -> tuple[Problem, ...]:
        """Return current diagnostics from skipped registrations.

        Returns:
            tuple[Problem, ...]: Registration problems in discovery order.
        """
        return self._registry.registration_problems

    def definitions(self) -> list[ToolDefinition]:
        """Return live model definitions supported by the current execution service.

        Returns:
            list[ToolDefinition]: Supported function-tool definitions in registration order.
        """
        return self._registry.definitions()

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
        execution_admission: Callable[[], bool] | None = None,
        file_observations: FileObservations | None = None,
        cancellation_event: Event | None = None,
    ) -> tuple[str, float]:
        """Dispatch through the owning registry and measure tool execution.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation.
            instructions_manager (InstructionsManager | None): Active instruction manager.
            permission_manager (PermissionManager | None): Invocation permission policy.
            call_id (str | None): Stable model request identifier.
            execution_started (Callable[[], Any] | None): Callback run before tool dispatch;
                async callbacks run on the owner loop and sync callbacks run in a bounded worker.
                Returned owner-loop tasks/futures settle on cancellation without starting the tool.
            execution_admission (Callable[[], bool] | None): Optional final admission check on
                the thread that starts the tool.
            file_observations (FileObservations | None): Frozen view supplied to this generation,
                or ``None`` for standalone dispatch without model context.
            cancellation_event (Event | None): Run cancellation signal passed to context-aware
                tools, or ``None`` for standalone dispatch.

        Returns:
            tuple[str, float]: Serialized result and tool-function duration in seconds.

        Raises:
            ToolExecutionAdmissionRejected: If final admission is refused.
        """
        return self._registry.call_with_timing(
            name,
            arguments,
            interaction=interaction,
            instructions_manager=instructions_manager,
            permission_manager=permission_manager,
            call_id=call_id,
            execution_started=execution_started,
            execution_admission=execution_admission,
            file_observations=file_observations,
            cancellation_event=cancellation_event,
        )

    async def call_with_timing_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        call_id: str | None = None,
        execution_started: Callable[[], Any] | None = None,
        execution_admission: Callable[[], bool] | None = None,
        file_observations: FileObservations | None = None,
        cancellation_event: Event | None = None,
    ) -> tuple[str, float]:
        """Dispatch asynchronously through the owning registry and measure tool execution.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation.
            instructions_manager (InstructionsManager | None): Active instruction manager.
            permission_manager (PermissionManager | None): Invocation permission policy.
            call_id (str | None): Stable model request identifier.
            execution_started (Callable[[], Any] | None): Callback run before tool dispatch;
                async callbacks run on the owner loop and sync callbacks run in a bounded worker.
                Returned owner-loop tasks/futures settle on cancellation without starting the tool.
            execution_admission (Callable[[], bool] | None): Optional final admission check on
                the thread that starts the tool.
            file_observations (FileObservations | None): Frozen view supplied to this generation,
                or ``None`` for standalone dispatch without model context.
            cancellation_event (Event | None): Run cancellation signal passed to context-aware
                tools, or ``None`` for standalone dispatch.

        Returns:
            tuple[str, float]: Serialized result and tool-function duration in seconds.
        Raises:
            ToolExecutionAdmissionRejected: If final admission is refused.
            ToolExecutionCancelled: If a cancelled synchronous invocation has a known result.
            asyncio.CancelledError: If dispatch is cancelled without an authoritative result.
        """
        return await self._registry.call_with_timing_async(
            name,
            arguments,
            interaction=interaction,
            instructions_manager=instructions_manager,
            permission_manager=permission_manager,
            call_id=call_id,
            execution_started=execution_started,
            execution_admission=execution_admission,
            file_observations=file_observations,
            cancellation_event=cancellation_event,
        )
