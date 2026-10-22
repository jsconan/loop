"""Run one agent through bounded model and tool turns."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import ExitStack, aclosing, nullcontext
from datetime import datetime
from time import perf_counter, sleep

from .. import constants
from ..backend import (
    Backend,
    BackendError,
    BackendNotFoundError,
    BackendRepetitionError,
)
from ..compaction import ContextCompaction
from ..control import RunControl, RunInterrupted
from ..errors import Problem, log_problem
from ..instructions import InstructionBudgetExceededError, InstructionsManager
from ..interaction import AsyncInteractionInput, Interaction
from ..model_selection import ModelSelection, ModelSelectionError
from ..models import (
    AgentRunStopReason,
    Message,
    ModelCallMetrics,
    Response,
    ResponseEvent,
    ResponseMetrics,
    RunMetrics,
    ToolCall,
    ToolExecutionMetrics,
    Usage,
)
from ..permissions import PermissionManager
from ..session import SessionManager, SessionRecoveryState
from ..telemetry import telemetry_activity, telemetry_span, telemetry_trace_event
from ..tooling.file_observations import FileObservations
from ..tooling.utils import serialize_tool_problem
from ..utils import PathReference, interaction_operation, interaction_run, utc_now
from .agent import Agent
from .models import AgentRecoveryStatus, AgentRunResult

_LOGGER = logging.getLogger(__name__)


class AgentRunner:
    """Run a configured agent against one conversation session.

    Args:
        agent (Agent): Reusable agent definition to execute.
        backend (Backend): Backend used to produce model responses.
        instructions_manager (InstructionsManager): Contextual instruction aggregation service.
        permission_manager (PermissionManager): Host policy authorizing tool effects.
        session_manager (SessionManager): Manager owning conversation history and persistence.
        model_selection (ModelSelection): Active model selection for the conversation.
        compaction (ContextCompaction): Context compactor run before model requests.
        interaction (Interaction): Service used to render responses and request recovery choices.
        working_directory (PathReference): Current instruction-directory reference.
        stream (bool): Whether backend response events should be streamed.
        debug (bool): Whether raw response events should be displayed.
        max_turns (int): Maximum completed model turns allowed in one run. Set to ``0`` to disable
            the turn limit (unlimited). Defaults to ``25``.
        prompt_on_recoverable_error (bool): Whether exhausted recoverable backend failures should
            be offered to the user for another attempt.
        run_control (RunControl | None): Optional conversation shutdown and scoped run interrupt
            signals. Active synchronous tools settle before cancellation is observed.

    Raises:
        ValueError: If ``max_turns`` is negative.
    """

    _agent: Agent
    _backend: Backend
    _instructions_manager: InstructionsManager
    _permission_manager: PermissionManager
    _session_manager: SessionManager
    _model_selection: ModelSelection
    _compaction: ContextCompaction
    _interaction: Interaction
    _working_directory: PathReference
    _stream: bool
    _debug: bool
    _max_turns: int
    _prompt_on_recoverable_error: bool
    _file_observations: FileObservations | None
    _run_control: RunControl | None
    _interrupted_tools: list[ToolExecutionMetrics] | None

    def __init__(
        self,
        agent: Agent,
        backend: Backend,
        instructions_manager: InstructionsManager,
        permission_manager: PermissionManager,
        session_manager: SessionManager,
        model_selection: ModelSelection,
        compaction: ContextCompaction,
        interaction: Interaction,
        working_directory: PathReference,
        *,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = constants.DEFAULT_AGENT_MAX_TURNS,
        prompt_on_recoverable_error: bool = True,
        run_control: RunControl | None = None,
    ) -> None:
        if max_turns < 0:
            raise ValueError("Agent max turns must be non-negative.")
        self._agent = agent
        self._backend = backend
        self._instructions_manager = instructions_manager
        self._permission_manager = permission_manager
        self._session_manager = session_manager
        self._model_selection = model_selection
        self._compaction = compaction
        self._interaction = interaction
        self._working_directory = working_directory
        self._stream = stream
        self._debug = debug
        self._max_turns = max_turns
        self._prompt_on_recoverable_error = prompt_on_recoverable_error
        self._run_control = run_control
        self._interrupted_tools = None
        self._file_observations = None

    @property
    def agent(self) -> Agent:
        """Return the agent executed by this runner.

        Returns:
            Agent: Configured agent.
        """
        return self._agent

    @agent.setter
    def agent(self, agent: Agent) -> None:
        """Replace the agent definition used for subsequent model requests.

        Args:
            agent (Agent): Fully configured replacement agent.
        """
        self._agent = agent

    @property
    def backend(self) -> Backend:
        """Return the backend used to produce model responses.

        Returns:
            Backend: Configured backend.
        """
        return self._backend

    @backend.setter
    def backend(self, backend: Backend) -> None:
        """Replace the backend used for subsequent model requests.

        Args:
            backend (Backend): Fully configured replacement backend.
        """
        self._backend = backend

    @property
    def instructions_manager(self) -> InstructionsManager:
        """Return the contextual instruction service used by this runtime.

        Returns:
            InstructionsManager: Active contextual instruction manager.
        """
        return self._instructions_manager

    @property
    def permission_manager(self) -> PermissionManager:
        """Return the host policy used to authorize tool effects.

        Returns:
            PermissionManager: Active permission manager.
        """
        return self._permission_manager

    @property
    def session_manager(self) -> SessionManager:
        """Return the conversation state owner used by this runtime.

        Returns:
            SessionManager: Active session manager.
        """
        return self._session_manager

    @property
    def model_selection(self) -> ModelSelection:
        """Return the model selection used by this runtime.

        Returns:
            ModelSelection: Active model selection.
        """
        return self._model_selection

    @property
    def compaction(self) -> ContextCompaction:
        """Return the context compaction service used by this runtime.

        Returns:
            ContextCompaction: Active context compaction service.
        """
        return self._compaction

    @property
    def interaction(self) -> Interaction:
        """Return the interaction service used by this runtime.

        Returns:
            Interaction: Active user interaction service.
        """
        return self._interaction

    @property
    def debug(self) -> bool:
        """Return whether raw response event output is enabled.

        Returns:
            bool: Whether debug output is enabled.
        """
        return self._debug

    @debug.setter
    def debug(self, debug: bool) -> None:
        """Enable or disable raw response event output.

        Args:
            debug (bool): Whether to enable debug output.
        """
        self._debug = debug

    @property
    def stream(self) -> bool:
        """Return whether responses are requested incrementally.

        Returns:
            bool: Whether response streaming is enabled.
        """
        return self._stream

    @stream.setter
    def stream(self, stream: bool) -> None:
        """Enable or disable streaming for subsequent backend requests.

        Args:
            stream (bool): Whether subsequent responses should stream.
        """
        self._stream = stream

    @property
    def max_turns(self) -> int:
        """Return the model-turn limit for one run.

        Returns:
            int: Maximum number of completed model turns. 0 means unlimited.
        """
        return self._max_turns

    @max_turns.setter
    def max_turns(self, max_turns: int) -> None:
        """Set the completed-turn limit for subsequent agent runs.

        Args:
            max_turns (int): Maximum completed turns, or zero for unlimited turns.

        Raises:
            ValueError: If the limit is negative.
        """
        if max_turns < 0:
            raise ValueError("Agent max turns must be non-negative.")
        self._max_turns = max_turns

    @property
    def prompt_on_recoverable_error(self) -> bool:
        """Return whether recoverable failures prompt for another attempt.

        Returns:
            bool: Whether exhausted recoverable failures prompt the user.
        """
        return self._prompt_on_recoverable_error

    @prompt_on_recoverable_error.setter
    def prompt_on_recoverable_error(self, enabled: bool) -> None:
        """Set recovery prompting for subsequent agent runs.

        Args:
            enabled (bool): Whether exhausted recoverable failures prompt the user.
        """
        self._prompt_on_recoverable_error = enabled

    def run(self) -> AgentRunResult:
        """Run model and tool turns until completion, cancellation, or the safety limit.

        Returns:
            AgentRunResult: Final response, completed turn count, and termination reason. Scoped
                interruption settles outstanding calls and returns a cancelled result.

        Raises:
            KeyboardInterrupt: If the configured control requests a stop.
            RuntimeError: If the configured control already owns another run.
            SessionExecutionConflictError: If another owner is executing this logical session.
        """
        session_id = self._session_manager.session.id
        with self._session_manager.execution(), telemetry_span():
            run = self._session_manager.begin_run()
            with interaction_run(session_id, run.run_id):
                telemetry_activity("agent.run.started", component="agent_runner")
                result = self._run_from_boundary(started_at=run.started_at)
                telemetry_activity(
                    "agent.run.completed",
                    component="agent_runner",
                    stop_reason=result.stop_reason,
                    turns=result.turns,
                )
                return result

    async def run_async(self) -> AgentRunResult:
        """Run model and tool turns through native asynchronous operations.

        Returns:
            AgentRunResult: Final response, completed turn count, and termination reason. Scoped
                interruption settles outstanding calls and returns a cancelled result.

        Raises:
            KeyboardInterrupt: If the configured control requests a stop.
            RuntimeError: If the configured control already owns another run.
            SessionExecutionConflictError: If another owner is executing this logical session.
            asyncio.CancelledError: If the owning async task is cancelled.
        """
        session_id = self._session_manager.session.id
        with self._session_manager.execution(), telemetry_span():
            run = await self._session_manager.begin_run_async()
            with interaction_run(session_id, run.run_id):
                telemetry_activity("agent.run.started", component="agent_runner")
                result = await self._run_from_boundary_async(started_at=run.started_at)
                telemetry_activity(
                    "agent.run.completed",
                    component="agent_runner",
                    stop_reason=result.stop_reason,
                    turns=result.turns,
                )
                return result

    def recover_session(self) -> AgentRecoveryStatus:
        """Execute an approved recovery plan for the active session.

        Returns:
            AgentRecoveryStatus: Recovered run result and whether unresolved recovery remains.

        Raises:
            KeyboardInterrupt: If the configured control requests a stop.
            RuntimeError: If the configured control already owns another run.
            SessionExecutionConflictError: If another owner is executing this logical session.
        """
        with interaction_run(
            self._session_manager.session.id,
            self._session_manager.session.active_run_id,
        ):
            self._checkpoint()
            state = self._session_manager.recovery_state
            if state is None:
                return AgentRecoveryStatus(pending=False)
            if not self._interaction.confirm(
                "This session stopped during an agent run. Recover and continue it?",
                default=False,
            ):
                return AgentRecoveryStatus(pending=True)
            with self._session_manager.execution():
                return AgentRecoveryStatus(pending=False, result=self._recover(state))

    async def recover_session_async(self) -> AgentRecoveryStatus:
        """Execute an approved recovery plan using asynchronous interaction and tools.

        Returns:
            AgentRecoveryStatus: Recovered run result and whether unresolved recovery remains.

        Raises:
            KeyboardInterrupt: If the configured control requests a stop.
            RunInterrupted: If the configured control interrupts recovery.
            SessionExecutionConflictError: If another owner is executing this logical session.
            asyncio.CancelledError: If the recovery task is cancelled.
        """
        with interaction_run(
            self._session_manager.session.id,
            self._session_manager.session.active_run_id,
        ):
            self._checkpoint()
            state = self._session_manager.recovery_state
            if state is None:
                return AgentRecoveryStatus(pending=False)
            if not await self._confirm_async(
                "This session stopped during an agent run. Recover and continue it?",
                default=False,
            ):
                return AgentRecoveryStatus(pending=True)
            with self._session_manager.execution():
                return AgentRecoveryStatus(pending=False, result=await self._recover_async(state))

    def _recover(self, state: SessionRecoveryState) -> AgentRunResult:
        """Continue one approved recovery plan from its durable execution boundary."""
        if state.action == "finalize_run":
            return self._record_run_result(
                final_response=None,
                turns=0,
                stop_reason="completed",
                started_at=utc_now(),
                calls=[],
                tools=[],
            )

        return self._run_from_boundary(recovery=state)

    async def _recover_async(self, state: SessionRecoveryState) -> AgentRunResult:
        """Continue one approved recovery plan through asynchronous operations."""
        if state.action == "finalize_run":
            return await self._record_run_result_async(
                final_response=None,
                turns=0,
                stop_reason="completed",
                started_at=utc_now(),
                calls=[],
                tools=[],
            )

        return await self._run_from_boundary_async(recovery=state)

    def _recover_tools(
        self,
        state: SessionRecoveryState,
        tools: list[ToolExecutionMetrics],
    ) -> None:
        """Resolve approved recovery calls while preserving each settled tool's statistics."""
        self._file_observations = FileObservations.for_recovery(
            self._session_manager.model_context,
            self._instructions_manager.virtual_paths,
            (
                pending.call
                for pending in state.pending_calls
                if pending.status != "result_available"
            ),
            prepare_context=self._backend.prepare_context,
        )
        for pending in state.pending_calls:
            self._checkpoint()
            if pending.status == "result_available":
                continue
            retry = True
            if pending.status == "outcome_unknown":
                with interaction_operation(pending.call.call_id):
                    retry = self._interaction.confirm(
                        f"Tool '{pending.call.name}' may already have run. Retry it?",
                        default=False,
                    )
            if not retry:
                self._persist_interrupted_tool_result(pending.call)
                continue
            tools.extend(
                self._execute_tool_calls((pending.call,), present=pending.status == "not_started")
            )

    async def _recover_tools_async(
        self,
        state: SessionRecoveryState,
        tools: list[ToolExecutionMetrics],
    ) -> None:
        """Resolve approved recovery calls without retrying unknown outcomes silently."""
        self._file_observations = FileObservations.for_recovery(
            self._session_manager.model_context,
            self._instructions_manager.virtual_paths,
            (
                pending.call
                for pending in state.pending_calls
                if pending.status != "result_available"
            ),
            prepare_context=self._backend.prepare_context,
        )
        for pending in state.pending_calls:
            self._checkpoint()
            if pending.status == "result_available":
                continue
            retry = True
            if pending.status == "outcome_unknown":
                with interaction_operation(pending.call.call_id):
                    retry = await self._confirm_async(
                        f"Tool '{pending.call.name}' may already have run. Retry it?",
                        default=False,
                    )
            if not retry:
                await self._persist_interrupted_tool_result_async(pending.call)
                continue
            tools.extend(
                await self._execute_tool_calls_async(
                    (pending.call,), present=pending.status == "not_started"
                )
            )

    def _run_from_boundary(
        self,
        *,
        recovery: SessionRecoveryState | None = None,
        started_at: datetime | None = None,
    ) -> AgentRunResult:
        """Run model turns from a fresh or recovered durable boundary."""
        total_turns = 0
        turns_since_confirmation = 0
        response = None
        calls = []
        tools = []
        started_at = started_at or utc_now()
        with self._run_control.execution() if self._run_control is not None else nullcontext():
            self._interrupted_tools = tools
            try:
                if recovery is not None:
                    self._recover_tools(recovery, tools)
                while True:
                    response = self._query_with_recovery()
                    if response is None:
                        return self._record_run_result(
                            final_response=None,
                            turns=total_turns,
                            stop_reason="cancelled",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    total_turns += 1
                    turns_since_confirmation += 1
                    calls.append(
                        ModelCallMetrics(
                            model=response.model,
                            duration_seconds=response.metrics.duration_seconds,
                            time_to_first_chunk_seconds=(
                                response.metrics.time_to_first_chunk_seconds
                            ),
                            usage=response.usage,
                        )
                    )
                    self._model_selection.record_response(response.model)
                    self._session_manager.add_response(response)
                    if self._steering_pending():
                        self._cancel_pending_tools()
                        return self._record_run_result(
                            final_response=response,
                            turns=total_turns,
                            stop_reason="steered",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    executions = self.handle_tool_calls(response)
                    tools.extend(executions)
                    self._checkpoint()
                    if self._steering_pending():
                        self._cancel_pending_tools()
                        return self._record_run_result(
                            final_response=response,
                            turns=total_turns,
                            stop_reason="steered",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    if not executions:
                        return self._record_run_result(
                            final_response=response,
                            turns=total_turns,
                            stop_reason="completed",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    if 0 < self._max_turns <= turns_since_confirmation:
                        prompt = (
                            f"Agent has reached the {self._max_turns}-turn safety limit. "
                            "Do you want to continue?"
                        )
                        if self._interaction.confirm(prompt, default=False) is not True:
                            break
                        turns_since_confirmation = 0

                self._interaction.warning(
                    f"Agent stopped after reaching the {self._max_turns}-turn safety limit."
                )
                return self._record_run_result(
                    final_response=response,
                    turns=total_turns,
                    stop_reason="max_turns",
                    started_at=started_at,
                    calls=calls,
                    tools=tools,
                )
            except RunInterrupted:
                self._cancel_pending_tools()
                self._interaction.info("Run interrupted; the conversation remains open.")
                return self._record_run_result(
                    final_response=None,
                    turns=total_turns,
                    stop_reason="cancelled",
                    started_at=started_at,
                    calls=calls,
                    tools=tools,
                )
            finally:
                self._interrupted_tools = None

    async def _run_from_boundary_async(
        self,
        *,
        recovery: SessionRecoveryState | None = None,
        started_at: datetime | None = None,
    ) -> AgentRunResult:
        """Run model turns from a fresh or recovered boundary using async operations."""
        total_turns = 0
        turns_since_confirmation = 0
        response = None
        calls = []
        tools = []
        started_at = started_at or utc_now()
        with (
            self._run_control.execution(async_owner=True)
            if self._run_control is not None
            else nullcontext()
        ):
            self._interrupted_tools = tools
            try:
                if recovery is not None:
                    await self._recover_tools_async(recovery, tools)
                while True:
                    response = await self._query_with_recovery_async()
                    if response is None:
                        return await self._record_run_result_async(
                            final_response=None,
                            turns=total_turns,
                            stop_reason="cancelled",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    total_turns += 1
                    turns_since_confirmation += 1
                    calls.append(
                        ModelCallMetrics(
                            model=response.model,
                            duration_seconds=response.metrics.duration_seconds,
                            time_to_first_chunk_seconds=(
                                response.metrics.time_to_first_chunk_seconds
                            ),
                            usage=response.usage,
                        )
                    )
                    await self._model_selection.record_response_async(response.model)
                    await self._session_manager.add_response_async(response)
                    if self._steering_pending():
                        await self._cancel_pending_tools_async()
                        return await self._record_run_result_async(
                            final_response=response,
                            turns=total_turns,
                            stop_reason="steered",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    executions = await self.handle_tool_calls_async(response)
                    tools.extend(executions)
                    self._checkpoint()
                    if self._steering_pending():
                        await self._cancel_pending_tools_async()
                        return await self._record_run_result_async(
                            final_response=response,
                            turns=total_turns,
                            stop_reason="steered",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    if not executions:
                        return await self._record_run_result_async(
                            final_response=response,
                            turns=total_turns,
                            stop_reason="completed",
                            started_at=started_at,
                            calls=calls,
                            tools=tools,
                        )
                    if 0 < self._max_turns <= turns_since_confirmation:
                        prompt = (
                            f"Agent has reached the {self._max_turns}-turn safety limit. "
                            "Do you want to continue?"
                        )
                        if not await self._confirm_async(prompt, default=False):
                            break
                        turns_since_confirmation = 0

                self._interaction.warning(
                    f"Agent stopped after reaching the {self._max_turns}-turn safety limit."
                )
                return await self._record_run_result_async(
                    final_response=response,
                    turns=total_turns,
                    stop_reason="max_turns",
                    started_at=started_at,
                    calls=calls,
                    tools=tools,
                )
            except RunInterrupted:
                await self._cancel_pending_tools_async()
                self._interaction.info("Run interrupted; the conversation remains open.")
                return await self._record_run_result_async(
                    final_response=None,
                    turns=total_turns,
                    stop_reason="cancelled",
                    started_at=started_at,
                    calls=calls,
                    tools=tools,
                )
            except asyncio.CancelledError:
                if self._run_control is None or not (
                    self._run_control.stopped or self._run_control.interrupted
                ):
                    raise
                await self._cancel_pending_tools_async()
                if self._run_control.interrupted and not self._run_control.stopped:
                    self._interaction.info("Run interrupted; the conversation remains open.")
                return await self._record_run_result_async(
                    final_response=None,
                    turns=total_turns,
                    stop_reason="cancelled",
                    started_at=started_at,
                    calls=calls,
                    tools=tools,
                )
            finally:
                self._interrupted_tools = None

    def handle_tool_calls(self, response: Response) -> tuple[ToolExecutionMetrics, ...]:
        """Execute and persist every tool call in a model response.

        Args:
            response (Response): Model response containing zero or more tool calls.

        Returns:
            tuple[ToolExecutionMetrics, ...]: Timing and outcome for every handled tool call.

        Raises:
            ValueError: If a requested tool call is not present in canonical session context.
            KeyboardInterrupt: If the configured control requests a stop.
            RunInterrupted: If the active execution scope has been interrupted.
            SessionExecutionConflictError: If another owner is executing this logical session.
        """
        if not response.tool_calls:
            return ()

        with self._session_manager.execution():
            return self._execute_tool_calls(response.tool_calls, present=True)

    async def handle_tool_calls_async(self, response: Response) -> tuple[ToolExecutionMetrics, ...]:
        """Execute and persist tool calls from a model response asynchronously.

        Args:
            response (Response): Model response containing zero or more tool calls.

        Returns:
            tuple[ToolExecutionMetrics, ...]: Timing and outcome for every handled tool call.

        Raises:
            ValueError: If a requested tool call is not present in canonical session context.
            KeyboardInterrupt: If the configured control requests a stop.
            RunInterrupted: If the active execution scope has been interrupted.
            SessionExecutionConflictError: If another owner is executing this logical session.
            asyncio.CancelledError: If an asynchronous tool call is cancelled.
        """
        if not response.tool_calls:
            return ()

        with self._session_manager.execution():
            return await self._execute_tool_calls_async(response.tool_calls, present=True)

    def _execute_tool_calls(
        self,
        tool_calls: tuple[ToolCall, ...],
        *,
        present: bool,
    ) -> tuple[ToolExecutionMetrics, ...]:
        """Execute and persist canonical tool requests from one durable boundary."""
        instructions = self._instructions_manager
        executions = []
        try:
            for tool_call in tool_calls:
                if self._steering_pending():
                    break
                with (
                    interaction_run(self._session_manager.session.id),
                    interaction_operation(tool_call.call_id),
                    telemetry_span(),
                ):
                    executions.append(
                        self._execute_tool_call(
                            tool_call,
                            instructions=instructions,
                            present=present,
                        )
                    )
        except RunInterrupted:
            if self._interrupted_tools is not None:
                self._interrupted_tools.extend(executions)
            raise
        return tuple(executions)

    async def _execute_tool_calls_async(
        self,
        tool_calls: tuple[ToolCall, ...],
        *,
        present: bool,
    ) -> tuple[ToolExecutionMetrics, ...]:
        """Execute tool requests sequentially through async registry dispatch."""
        instructions = self._instructions_manager
        executions = []
        try:
            for tool_call in tool_calls:
                if self._steering_pending():
                    break
                with (
                    interaction_run(self._session_manager.session.id),
                    interaction_operation(tool_call.call_id),
                    telemetry_span(),
                ):
                    executions.append(
                        await self._execute_tool_call_async(
                            tool_call,
                            instructions=instructions,
                            present=present,
                        )
                    )
        except RunInterrupted:
            if self._interrupted_tools is not None:
                self._interrupted_tools.extend(executions)
            raise
        except asyncio.CancelledError:
            if self._interrupted_tools is not None:
                self._interrupted_tools.extend(executions)
            raise
        return tuple(executions)

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        instructions: InstructionsManager,
        present: bool,
    ) -> ToolExecutionMetrics:
        """Execute, trace, and persist one tool request inside its correlation span."""
        self._checkpoint()
        telemetry_trace_event(
            "tool.request",
            payload={
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        )
        if present:
            self._session_manager.add_tool_call_event(tool_call.call_id)
        self._interaction.tool_call(tool_call.name, tool_call.arguments)

        def execution_started() -> None:
            """Honor a stop after approval and before recording the execution boundary."""
            self._checkpoint()
            self._session_manager.record_tool_execution_started(tool_call.call_id)

        tool_result, duration = self._agent.tools.call_with_timing(
            tool_call.name,
            tool_call.arguments,
            interaction=self._interaction,
            instructions_manager=instructions,
            permission_manager=self._permission_manager,
            call_id=tool_call.call_id,
            cancellation_event=(
                self._run_control.cancellation_event if self._run_control is not None else None
            ),
            **(
                {"file_observations": self._file_observations}
                if self._file_observations is not None
                else {}
            ),
            execution_started=execution_started,
        )
        if self._file_observations is not None:
            self._file_observations.observe_result(tool_call, tool_result, include_reads=False)
        telemetry_trace_event(
            "tool.response",
            payload={"call_id": tool_call.call_id, "output": tool_result},
            duration_seconds=duration,
        )
        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            payload = None
        succeeded = isinstance(payload, dict) and payload.get("ok") is True
        self._session_manager.add_tool_call(
            call_id=tool_call.call_id,
            output=tool_result,
            working_directory=str(instructions.working_directory or self._working_directory),
            active_skills=instructions.active_skill_identities,
            succeeded=succeeded,
            duration_seconds=duration,
        )
        return ToolExecutionMetrics(
            name=tool_call.name,
            duration_seconds=duration,
            succeeded=succeeded,
        )

    async def _execute_tool_call_async(
        self,
        tool_call: ToolCall,
        *,
        instructions: InstructionsManager,
        present: bool,
    ) -> ToolExecutionMetrics:
        """Execute, trace, and persist one tool request through async dispatch."""
        self._checkpoint()
        telemetry_trace_event(
            "tool.request",
            payload={
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        )
        if present:
            await self._session_manager.add_tool_call_event_async(tool_call.call_id)
        self._interaction.tool_call(tool_call.name, tool_call.arguments)

        async def execution_started() -> None:
            """Persist execution intent after approval and before invoking the tool."""
            self._checkpoint()
            await self._session_manager.record_tool_execution_started_async(tool_call.call_id)

        tool_result, duration = await self._agent.tools.call_with_timing_async(
            tool_call.name,
            tool_call.arguments,
            interaction=self._interaction,
            instructions_manager=instructions,
            permission_manager=self._permission_manager,
            call_id=tool_call.call_id,
            cancellation_event=(
                self._run_control.cancellation_event if self._run_control is not None else None
            ),
            **(
                {"file_observations": self._file_observations}
                if self._file_observations is not None
                else {}
            ),
            execution_started=execution_started,
        )
        if self._file_observations is not None:
            self._file_observations.observe_result(tool_call, tool_result, include_reads=False)
        telemetry_trace_event(
            "tool.response",
            payload={"call_id": tool_call.call_id, "output": tool_result},
            duration_seconds=duration,
        )
        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            payload = None
        succeeded = isinstance(payload, dict) and payload.get("ok") is True
        await self._session_manager.add_tool_call_async(
            call_id=tool_call.call_id,
            output=tool_result,
            working_directory=str(instructions.working_directory or self._working_directory),
            active_skills=instructions.active_skill_identities,
            succeeded=succeeded,
            duration_seconds=duration,
        )
        return ToolExecutionMetrics(
            name=tool_call.name,
            duration_seconds=duration,
            succeeded=succeeded,
        )

    def _cancel_pending_tools(self) -> None:
        """Resolve outstanding calls without executing or automatically retrying any effects."""
        state = self._session_manager.recovery_state
        if state is not None:
            for pending in state.pending_calls:
                if pending.status != "result_available":
                    self._persist_interrupted_tool_result(
                        pending.call,
                        not_started=pending.status == "not_started",
                    )

    def _persist_interrupted_tool_result(
        self,
        call: ToolCall,
        *,
        not_started: bool = False,
    ) -> None:
        """Resolve one cancelled or uncertain invocation without repeating side effects."""
        instructions = self._instructions_manager
        output = self._interrupted_tool_output(call, not_started=not_started)
        self._session_manager.add_tool_call(
            call_id=call.call_id,
            output=output,
            working_directory=str(instructions.working_directory or self._working_directory),
            active_skills=instructions.active_skill_identities,
            succeeded=False,
            duration_seconds=0,
        )

    async def _cancel_pending_tools_async(self) -> None:
        """Resolve pending calls asynchronously without repeating uncertain effects."""
        state = self._session_manager.recovery_state
        if state is not None:
            for pending in state.pending_calls:
                if pending.status != "result_available":
                    await self._persist_interrupted_tool_result_async(
                        pending.call,
                        not_started=pending.status == "not_started",
                    )

    async def _persist_interrupted_tool_result_async(
        self,
        call: ToolCall,
        *,
        not_started: bool = False,
    ) -> None:
        """Persist one cancelled or uncertain invocation through async session storage."""
        instructions = self._instructions_manager
        await self._session_manager.add_tool_call_async(
            call_id=call.call_id,
            output=self._interrupted_tool_output(call, not_started=not_started),
            working_directory=str(instructions.working_directory or self._working_directory),
            active_skills=instructions.active_skill_identities,
            succeeded=False,
            duration_seconds=0,
        )

    @staticmethod
    def _interrupted_tool_output(call: ToolCall, *, not_started: bool) -> str:
        """Serialize the explicit result for an interrupted or uncertain tool invocation."""
        return serialize_tool_problem(
            Problem(
                code="tool.execution_interrupted",
                title="Tool execution interrupted",
                detail=(
                    f"Tool '{call.name}' was cancelled before execution."
                    if not_started
                    else (
                        f"Tool '{call.name}' was interrupted and its external outcome is unknown. "
                        "Do not assume the operation failed."
                    )
                ),
                severity="warning",
                operation=call.name,
            )
        )

    def query(self) -> Response:
        """Request one response using current session context and agent capabilities.

        Returns:
            Response: Collected normalized model response.

        Raises:
            BackendError: If the backend cannot produce a complete response.
            ValueError: If neither the selection nor backend provides a model.
            KeyboardInterrupt: If the configured control requests a stop.
            RunInterrupted: If the active execution scope has been interrupted.
        """
        with telemetry_span():
            return self._query()

    async def query_async(self) -> Response:
        """Request and collect one response through the backend's asynchronous stream.

        Returns:
            Response: Collected normalized model response.

        Raises:
            BackendError: If the backend cannot produce a complete response.
            ValueError: If neither the selection nor backend provides a model.
            KeyboardInterrupt: If the configured control requests a stop.
            RunInterrupted: If the active execution scope has been interrupted.
            asyncio.CancelledError: If the query task is cancelled.
        """
        with telemetry_span():
            return await self._query_async()

    def _query(self, recovery: str | None = None) -> Response:
        """Request and collect one response inside the active model-operation span."""
        self._checkpoint()
        selected_model = self._model_selection.effective
        snapshot = self._instructions_manager.prepare(self._agent, recovery=recovery)
        self._session_manager.update_instruction_state(
            working_directory=str(snapshot.working_directory or self._working_directory),
            active_skills=list(snapshot.active_skills),
        )
        self._model_selection.synchronize_session()
        self._compaction.compact_if_needed(snapshot)
        self._checkpoint()
        started_at = perf_counter()
        first_chunk_at = None
        model_context = self._backend.prepare_context(self._session_manager.model_context)
        self._file_observations = FileObservations(
            model_context,
            self._instructions_manager.virtual_paths,
        )
        events = self._backend.get_response(
            input=model_context,
            instructions=snapshot.content,
            stream=self._stream,
            model=selected_model,
            hyperparameters=self._agent.hyperparameters,
            tools=self._agent.tools.definitions(),
        )

        def measured_events():
            nonlocal first_chunk_at
            for event in events:
                self._checkpoint()
                if first_chunk_at is None:
                    first_chunk_at = perf_counter()
                yield event

        with ExitStack() as resources:
            close_stream = getattr(events, "close", None)
            if callable(close_stream):
                resources.callback(close_stream)
            response = self._session_manager.response(
                measured_events(),
                debug=self._debug,
                interaction=self._interaction,
            )
        completed_at = perf_counter()
        response.metrics = ResponseMetrics(
            duration_seconds=completed_at - started_at,
            time_to_first_chunk_seconds=(
                first_chunk_at - started_at if self._stream and first_chunk_at is not None else None
            ),
        )
        return response

    async def _query_async(self, recovery: str | None = None) -> Response:
        """Collect one asynchronous response inside the active model-operation span."""
        self._checkpoint()
        selected_model = self._model_selection.effective
        snapshot = self._instructions_manager.prepare(self._agent, recovery=recovery)
        self._session_manager.update_instruction_state(
            working_directory=str(snapshot.working_directory or self._working_directory),
            active_skills=list(snapshot.active_skills),
        )
        await self._model_selection.synchronize_session_async()
        await self._compaction.compact_if_needed_async(snapshot)
        self._checkpoint()
        started_at = perf_counter()
        first_chunk_at = None
        model_context = self._backend.prepare_context(self._session_manager.model_context)
        self._file_observations = FileObservations(
            model_context,
            self._instructions_manager.virtual_paths,
        )
        events = self._backend.get_response_async(
            input=model_context,
            instructions=snapshot.content,
            stream=self._stream,
            model=selected_model,
            hyperparameters=self._agent.hyperparameters,
            tools=self._agent.tools.definitions(),
        )

        async def measured_events() -> AsyncIterator[ResponseEvent]:
            """Check cancellation and record first-event timing while yielding backend events."""
            nonlocal first_chunk_at
            async for event in events:
                self._checkpoint()
                if first_chunk_at is None:
                    first_chunk_at = perf_counter()
                yield event

        async with aclosing(measured_events()) as stream:
            try:
                response = await self._session_manager.response_async(
                    stream,
                    debug=self._debug,
                    interaction=self._interaction,
                )
            finally:
                closer = getattr(events, "aclose", None)
                if callable(closer):
                    await closer()
        completed_at = perf_counter()
        response.metrics = ResponseMetrics(
            duration_seconds=completed_at - started_at,
            time_to_first_chunk_seconds=(
                first_chunk_at - started_at if self._stream and first_chunk_at is not None else None
            ),
        )
        return response

    def _record_run_result(
        self,
        final_response: Response | None,
        turns: int,
        stop_reason: AgentRunStopReason,
        started_at: datetime,
        calls: list[ModelCallMetrics],
        tools: list[ToolExecutionMetrics],
    ) -> AgentRunResult:
        """Persist and return one completed run's aggregate operation statistics."""
        metrics = self._build_run_metrics(calls, tools)
        self._session_manager.record_run(stop_reason, started_at, metrics)
        return AgentRunResult(
            final_response=final_response,
            turns=turns,
            stop_reason=stop_reason,
            metrics=metrics,
        )

    async def _record_run_result_async(
        self,
        final_response: Response | None,
        turns: int,
        stop_reason: AgentRunStopReason,
        started_at: datetime,
        calls: list[ModelCallMetrics],
        tools: list[ToolExecutionMetrics],
    ) -> AgentRunResult:
        """Persist and return one completed run through async session storage."""
        metrics = self._build_run_metrics(calls, tools)
        await self._session_manager.record_run_async(stop_reason, started_at, metrics)
        return AgentRunResult(
            final_response=final_response,
            turns=turns,
            stop_reason=stop_reason,
            metrics=metrics,
        )

    def _build_run_metrics(
        self,
        calls: list[ModelCallMetrics],
        tools: list[ToolExecutionMetrics],
    ) -> RunMetrics:
        """Build aggregate run metrics from completed model and tool operations."""
        usage_values = {}
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ):
            values = [getattr(call.usage, field) for call in calls]
            usage_values[field] = (
                sum(value for value in values if value is not None)
                if values and all(value is not None for value in values)
                else None
            )
        messages = self._session_manager.messages
        calls_duration = sum(call.duration_seconds for call in calls)
        tools_duration = sum(tool.duration_seconds for tool in tools)
        metrics = RunMetrics(
            active_duration_seconds=calls_duration + tools_duration,
            model_duration_seconds=calls_duration,
            tool_duration_seconds=tools_duration,
            model_calls=tuple(calls),
            tools=tuple(tools),
            message_count=sum(isinstance(item, Message) for item in messages),
            item_count=len(messages),
            usage=Usage(**usage_values),
            model=self._session_manager.model,
            context_tokens=self._session_manager.tokens,
            context_window=self._session_manager.context_window,
        )
        return metrics

    def _query_with_recovery(self) -> Response | None:
        """Request one response while escalating exhausted failures to the user."""
        recovery = None
        while True:
            self._checkpoint()
            try:
                if recovery is None:
                    return self.query()
                with telemetry_span():
                    return self._query(recovery)
            except BackendNotFoundError:
                self._interaction.warning("The selected model is not available.")
                if not self._model_selection.select_fallback(self._interaction):
                    return None
            except (InstructionBudgetExceededError, ModelSelectionError) as error:
                problem = Problem.from_exception(
                    error,
                    code="agent.query_failed",
                    title="Could not prepare the request",
                    detail=(
                        "The request could not be prepared, for example because no model is "
                        "selected or applicable instructions exceed the configured budget. "
                        "Fix the underlying condition before sending another message."
                    ),
                    operation="query",
                )
                log_problem(_LOGGER, problem, error)
                self._interaction.report(problem)
                return None
            except BackendError as error:
                if isinstance(error, BackendRepetitionError) and recovery is None:
                    telemetry_activity("agent.repetition_recovery", component="agent_runner")
                    self._interaction.warning(
                        "The model response was stopped after repetitive output. Retrying once..."
                    )
                    recovery = "repetition"
                    continue
                self._report_backend_error(error)
                if not error.recoverable or not self._prompt_on_recoverable_error:
                    return None
                prompt = "Retry the complete response?"
                if error.response_started:
                    prompt = "Partial output was discarded. Retry the complete response?"
                if error.retry_after is not None and error.retry_after > 0:
                    prompt = f"{prompt[:-1]} after at least {error.retry_after:g} seconds?"
                if not self._interaction.confirm(prompt, default=False):
                    return None
                if error.retry_after is not None and error.retry_after > 0:
                    delay = min(error.retry_after, 60.0)
                    self._interaction.info(f"Retrying in {delay:g} seconds...")
                    if self._run_control is None:
                        sleep(delay)
                    else:
                        self._run_control.wait(delay)

    async def _query_with_recovery_async(self) -> Response | None:
        """Request an asynchronous response while preserving existing recovery decisions.

        Returns:
            Response | None: Completed response, or ``None`` when recovery is declined or
                request preparation cannot continue.

        Raises:
            KeyboardInterrupt: If the configured control requests shutdown.
            RunInterrupted: If the configured control interrupts the active run.
            BackendError: If model fallback cannot retrieve updated model metadata.
            OSError: If persisting a fallback model preference fails.
            asyncio.CancelledError: If an async query or user interaction is cancelled.
        """
        recovery = None
        while True:
            self._checkpoint()
            try:
                if recovery is None:
                    return await self.query_async()
                with telemetry_span():
                    return await self._query_async(recovery)
            except BackendNotFoundError:
                self._interaction.warning("The selected model is not available.")
                if not await self._model_selection.select_fallback_async(self._interaction):
                    return None
            except (InstructionBudgetExceededError, ModelSelectionError) as error:
                problem = Problem.from_exception(
                    error,
                    code="agent.query_failed",
                    title="Could not prepare the request",
                    detail=(
                        "The request could not be prepared, for example because no model is "
                        "selected or applicable instructions exceed the configured budget. "
                        "Fix the underlying condition before sending another message."
                    ),
                    operation="query",
                )
                log_problem(_LOGGER, problem, error)
                self._interaction.report(problem)
                return None
            except BackendError as error:
                if isinstance(error, BackendRepetitionError) and recovery is None:
                    telemetry_activity("agent.repetition_recovery", component="agent_runner")
                    self._interaction.warning(
                        "The model response was stopped after repetitive output. Retrying once..."
                    )
                    recovery = "repetition"
                    continue
                self._report_backend_error(error)
                if not error.recoverable or not self._prompt_on_recoverable_error:
                    return None
                prompt = "Retry the complete response?"
                if error.response_started:
                    prompt = "Partial output was discarded. Retry the complete response?"
                if error.retry_after is not None and error.retry_after > 0:
                    prompt = f"{prompt[:-1]} after at least {error.retry_after:g} seconds?"
                if not await self._confirm_async(prompt, default=False):
                    return None
                if error.retry_after is not None and error.retry_after > 0:
                    delay = min(error.retry_after, 60.0)
                    self._interaction.info(f"Retrying in {delay:g} seconds...")
                    if self._run_control is None:
                        await asyncio.sleep(delay)
                    else:
                        await self._run_control.wait_async(delay)

    async def _confirm_async(self, prompt: str, *, default: bool) -> bool:
        """Use native async confirmation when available, retaining sync adapter support."""
        if isinstance(self._interaction, AsyncInteractionInput):
            return await self._interaction.confirm_async(prompt, default=default)
        return self._interaction.confirm(prompt, default=default)

    def _checkpoint(self) -> None:
        """Honor optional cooperative stop requests before a new execution boundary."""
        if self._run_control is not None:
            self._run_control.checkpoint()

    def _steering_pending(self) -> bool:
        """Return whether an admitted steering message should stop new work in this run."""
        return self._run_control is not None and self._run_control.steering_pending is True

    def _report_backend_error(self, error: BackendError) -> None:
        """Display a normalized backend failure and useful diagnostic identifiers."""
        problem = error.to_problem()
        log_problem(_LOGGER, problem, error)
        self._interaction.report(problem)
