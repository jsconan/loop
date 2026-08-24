"""Run one agent through bounded model and tool turns."""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep

from .. import constants
from ..backend import BackendError, BackendNotFoundError
from ..compaction import ContextCompaction
from ..errors import log_problem
from ..interaction import Interaction
from ..model_selection import ModelSelection
from ..models import (
    AgentRunStopReason,
    Message,
    ModelCallMetrics,
    Response,
    ResponseMetrics,
    RunMetrics,
    ToolExecutionMetrics,
    Usage,
)
from ..session import SessionManager
from .agent import Agent
from .models import AgentRunResult

_LOGGER = logging.getLogger(__name__)


class AgentRunner:
    """Run a configured agent against one conversation session.

    Args:
        agent (Agent): Agent whose model, instructions, tools, and policy are executed.
        session_manager (SessionManager): Manager owning conversation history and persistence.
        model_selection (ModelSelection): Active model selection for the conversation.
        compaction (ContextCompaction): Context compactor run before model requests.
        interaction (Interaction): Service used to render responses and request recovery choices.
        working_directory (Callable[[], Path]): Provider of the current instruction directory.
        stream (bool): Whether backend response events should be streamed.
        debug (bool): Whether raw response events should be displayed.
        max_turns (int): Maximum completed model turns allowed in one run. Set to ``0`` to disable
            the turn limit (unlimited). Defaults to ``25``.
        prompt_on_recoverable_error (bool): Whether exhausted recoverable backend failures should
            be offered to the user for another attempt.

    Raises:
        ValueError: If ``max_turns`` is negative.
    """

    _agent: Agent
    _session_manager: SessionManager
    _model_selection: ModelSelection
    _compaction: ContextCompaction
    _interaction: Interaction
    _working_directory: Callable[[], Path]
    _stream: bool
    _debug: bool
    _max_turns: int
    _prompt_on_recoverable_error: bool

    def __init__(
        self,
        agent: Agent,
        session_manager: SessionManager,
        model_selection: ModelSelection,
        compaction: ContextCompaction,
        interaction: Interaction,
        working_directory: Callable[[], Path],
        *,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = constants.DEFAULT_AGENT_MAX_TURNS,
        prompt_on_recoverable_error: bool = True,
    ) -> None:
        if max_turns < 0:
            raise ValueError("Agent max turns must be non-negative.")
        self._agent = agent
        self._session_manager = session_manager
        self._model_selection = model_selection
        self._compaction = compaction
        self._interaction = interaction
        self._working_directory = working_directory
        self._stream = stream
        self._debug = debug
        self._max_turns = max_turns
        self._prompt_on_recoverable_error = prompt_on_recoverable_error

    @property
    def agent(self) -> Agent:
        """Return the agent executed by this runner.

        Returns:
            Agent: Configured agent.
        """
        return self._agent

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

    @property
    def max_turns(self) -> int:
        """Return the model-turn limit for one run.

        Returns:
            int: Maximum number of completed model turns. 0 means unlimited.
        """
        return self._max_turns

    def run(self) -> AgentRunResult:
        """Run model and tool turns until completion, cancellation, or the safety limit.

        Returns:
            AgentRunResult: Final response, completed turn count, and termination reason.
        """
        turn = 0
        response = None
        calls = []
        tools = []
        started_at = datetime.now(UTC)
        while True:
            turn += 1

            response = self._query_with_recovery()
            if response is None:
                return self._record_run_result(
                    final_response=None,
                    turns=turn - 1,
                    stop_reason="cancelled",
                    started_at=started_at,
                    calls=calls,
                    tools=tools,
                )
            calls.append(
                ModelCallMetrics(
                    model=response.model,
                    duration_seconds=response.metrics.duration_seconds,
                    time_to_first_chunk_seconds=response.metrics.time_to_first_chunk_seconds,
                    usage=response.usage,
                )
            )
            self._model_selection.record_response(response.model)
            self._session_manager.add_response(response)
            executions = self.handle_tool_calls(response)
            if not executions:
                return self._record_run_result(
                    final_response=response,
                    turns=turn,
                    stop_reason="completed",
                    started_at=started_at,
                    calls=calls,
                    tools=tools,
                )
            tools.extend(executions)

            if self._max_turns > 0 and turn >= self._max_turns:
                prompt = (
                    f"Agent has reached the {self._max_turns}-turn safety limit. "
                    "Do you want to continue?"
                )
                if self._interaction.confirm(prompt, default=False) is not True:
                    break
                turn = 0  # reset counter to allow another round

        self._interaction.warning(
            f"Agent stopped after reaching the {self._max_turns}-turn safety limit."
        )
        return self._record_run_result(
            final_response=response,
            turns=self._max_turns,
            stop_reason="max_turns",
            started_at=started_at,
            calls=calls,
            tools=tools,
        )

    def handle_tool_calls(self, response: Response) -> tuple[ToolExecutionMetrics, ...]:
        """Execute and persist every tool call in a model response.

        Args:
            response (Response): Model response containing zero or more tool calls.

        Returns:
            tuple[ToolExecutionMetrics, ...]: Timing and outcome for every handled tool call.

        Raises:
            ValueError: If a requested tool call is not present in canonical session context.
        """
        if not response.tool_calls:
            return ()

        instructions = self._agent.instructions_manager
        executions = []
        for tool_call in response.tool_calls:
            self._session_manager.add_tool_call_event(tool_call.call_id)
            self._interaction.tool_call(tool_call.name, tool_call.arguments)
            tool_result, duration = self._agent.tool_registry.call_with_timing(
                tool_call.name,
                tool_call.arguments,
                interaction=self._interaction,
                instructions_manager=instructions,
                permission_manager=self._agent.permission_manager,
            )
            try:
                payload = json.loads(tool_result)
            except json.JSONDecodeError:
                payload = None
            succeeded = isinstance(payload, dict) and payload.get("ok") is True
            executions.append(
                ToolExecutionMetrics(
                    name=tool_call.name,
                    duration_seconds=duration,
                    succeeded=succeeded,
                )
            )
            self._session_manager.add_tool_call(
                call_id=tool_call.call_id,
                output=tool_result,
                working_directory=str(instructions.working_directory or self._working_directory()),
                active_skills=instructions.active_skill_identities,
            )
        return tuple(executions)

    def query(self) -> Response:
        """Request one response using current session context and agent capabilities.

        Returns:
            Response: Collected normalized model response.

        Raises:
            BackendError: If the backend cannot produce a complete response.
            ValueError: If neither the selection nor backend provides a model.
        """
        selected_model = self._model_selection.effective
        instructions = self._agent.instructions_manager
        instructions.prepare()
        self._session_manager.update_instruction_state(
            working_directory=str(instructions.working_directory or self._working_directory()),
            active_skills=instructions.active_skill_identities,
        )
        self._model_selection.synchronize_session()
        self._compaction.compact_if_needed()
        started_at = perf_counter()
        first_chunk_at = None
        events = self._agent.backend.get_response(
            input=self._session_manager.model_context,
            instructions=instructions.instructions,
            stream=self._stream,
            model=selected_model,
            tools=self._agent.tool_registry.definitions(),
        )

        def measured_events():
            nonlocal first_chunk_at
            for event in events:
                if first_chunk_at is None:
                    first_chunk_at = perf_counter()
                yield event

        response = self._session_manager.response(
            measured_events(), debug=self._debug, interaction=self._interaction
        )
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
        final_response: Response,
        turns: int,
        stop_reason: AgentRunStopReason,
        started_at: datetime,
        calls: list[ModelCallMetrics],
        tools: list[ToolExecutionMetrics],
    ) -> AgentRunResult:
        """Persist and return one completed run's aggregate operation statistics."""
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
        self._session_manager.record_run(stop_reason, started_at, metrics)
        return AgentRunResult(
            final_response=final_response, turns=turns, stop_reason=stop_reason, metrics=metrics
        )

    def _query_with_recovery(self) -> Response | None:
        """Request one response while escalating exhausted failures to the user."""
        while True:
            try:
                return self.query()
            except BackendNotFoundError as error:
                self._report_backend_error(error)
                if not self._model_selection.select_fallback(self._interaction):
                    return None
            except BackendError as error:
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
                    sleep(delay)

    def _report_backend_error(self, error: BackendError) -> None:
        """Display a normalized backend failure and useful diagnostic identifiers."""
        problem = error.to_problem()
        log_problem(_LOGGER, problem, error)
        self._interaction.report(problem)
