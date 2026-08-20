"""Run one agent through bounded model and tool turns."""

from collections.abc import Callable
from pathlib import Path
from time import sleep

from .. import constants
from ..backend import BackendError, BackendNotFoundError
from ..compaction import ContextCompaction
from ..interaction import Interaction
from ..model_selection import ModelSelection
from ..models import Response
from ..session import SessionManager
from .agent import Agent
from .models import AgentRunResult


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
        max_turns (int): Maximum completed model turns allowed in one run. Defaults to ``25``.
        prompt_on_recoverable_error (bool): Whether exhausted recoverable backend failures should
            be offered to the user for another attempt.

    Raises:
        ValueError: If ``max_turns`` is not positive.
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
        if max_turns <= 0:
            raise ValueError("Agent max turns must be greater than zero.")
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
            int: Positive maximum number of completed model turns.
        """
        return self._max_turns

    def run(self) -> AgentRunResult:
        """Run model and tool turns until completion, cancellation, or the safety limit.

        Returns:
            AgentRunResult: Final response, completed turn count, and termination reason.
        """
        for turn in range(1, self._max_turns + 1):
            response = self._query_with_recovery()
            if response is None:
                return AgentRunResult(
                    final_response=None,
                    turns=turn - 1,
                    stop_reason="cancelled",
                )
            self._session_manager.add_response(response)
            if not self.handle_tool_calls(response):
                return AgentRunResult(
                    final_response=response,
                    turns=turn,
                    stop_reason="completed",
                )

        self._interaction.warning(
            f"Agent stopped after reaching the {self._max_turns}-turn safety limit."
        )
        return AgentRunResult(
            final_response=response,
            turns=self._max_turns,
            stop_reason="max_turns",
        )

    def handle_tool_calls(self, response: Response) -> bool:
        """Execute and persist every tool call in a model response.

        Args:
            response (Response): Model response containing zero or more tool calls.

        Returns:
            bool: ``True`` when at least one tool call was handled; otherwise ``False``.
        """
        if not response.tool_calls:
            return False

        instructions = self._agent.instructions_manager
        for tool_call in response.tool_calls:
            self._interaction.tool_call(tool_call.name, tool_call.arguments)
            tool_result = self._agent.tool_registry.call(
                tool_call.name,
                tool_call.arguments,
                interaction=self._interaction,
                instructions_manager=instructions,
                permission_manager=self._agent.permission_manager,
            )
            self._session_manager.add_tool_call(
                call_id=tool_call.call_id,
                output=tool_result,
                working_directory=str(instructions.working_directory or self._working_directory()),
                active_skills=instructions.active_skill_identities,
            )
        return True

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
        events = self._agent.backend.get_response(
            input=self._session_manager.model_context,
            instructions=instructions.instructions,
            stream=self._stream,
            model=selected_model,
            tools=self._agent.tool_registry.definitions(),
        )
        return self._interaction.response(events, debug=self._debug)

    def _query_with_recovery(self) -> Response | None:
        """Request one response while escalating exhausted failures to the user."""
        while True:
            try:
                return self.query()
            except BackendNotFoundError as error:
                self._report_backend_error(error)
                if not self._select_fallback_model():
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
        message = str(error)
        diagnostics = []
        if error.status_code is not None:
            diagnostics.append(f"HTTP {error.status_code}")
        if error.request_id:
            diagnostics.append(f"request {error.request_id}")
        if diagnostics:
            message = f"{message} ({', '.join(diagnostics)})"
        self._interaction.error(message)

    def _select_fallback_model(self) -> bool:
        """Let the user replace a missing model selection."""
        try:
            models = self._model_selection.available()
        except BackendError as error:
            self._interaction.error(f"Could not list available models: {error}")
            return False
        if not models:
            self._interaction.warning("The backend reported no available models.")
            return False
        failing_model = self._model_selection.selected
        while True:
            selection = self._interaction.prompt(
                "Select a replacement model, or enter 'q' to stop: ",
                choices={model.id: model.id for model in models},
            )
            if selection is False:
                return False
            if selection != failing_model:
                break
            self._interaction.warning(
                f"Model '{selection}' was already unavailable; the same "
                "failure is likely to re-occur unless the backend is updated."
            )
            if self._interaction.confirm(
                "Continue with this model, or select a different one?",
                default=True,
            ):
                break
        self._model_selection.select(selection)
        self._interaction.info(f"Using model: {selection}")
        return True
