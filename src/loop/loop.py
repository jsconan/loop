"""Run an interactive conversation with an LLM backend."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Self

from . import constants
from .agent import Agent, AgentRunner, AgentRunResult
from .backend import Backend
from .commands import CommandManager
from .compaction import CompactionCommands, ContextCompaction
from .completion import (
    CommandCompletionAdapter,
    CompletionManager,
)
from .errors import Problem, log_problem
from .instructions import InstructionsManager, RuntimeEnvironment, SkillCommands
from .interaction import Interaction
from .mentions import MentionManager, ProjectPathMentionHandler, SkillMentionHandler
from .model_selection import ModelCommands, ModelSelection
from .models import ConversationItem
from .permissions import PermissionCommands, PermissionManager
from .session import (
    BackendSessionNameGenerator,
    Session,
    SessionCommands,
    SessionManager,
    SessionNameGenerator,
)
from .telemetry import telemetry_activity
from .tooling import ToolCommands, ToolRegistry
from .utils import find_project_root

_LOGGER = logging.getLogger(__name__)


class Loop:
    """Run an interactive conversation using normalized response events.

    Args:
        agent_runner (AgentRunner): Runtime executing the agent.
        command_manager (CommandManager): Registered interactive commands.
        completion_manager (CompletionManager): Interactive completion service.
        session_name_generator (SessionNameGenerator): Session naming service.
        mention_manager (MentionManager): Mention resolution service.
        working_directory (Path): Initial workspace directory.

    """

    _session_manager: SessionManager
    _interaction: Interaction
    _working_directory: Path
    _debug: bool
    _stream: bool
    _model_selection: ModelSelection
    _compaction: ContextCompaction
    _command_manager: CommandManager
    _completion_manager: CompletionManager
    _session_name_generator: SessionNameGenerator
    _mention_manager: MentionManager
    _agent: Agent
    _agent_runner: AgentRunner

    def __init__(
        self,
        *,
        agent_runner: AgentRunner,
        command_manager: CommandManager,
        completion_manager: CompletionManager,
        session_name_generator: SessionNameGenerator,
        mention_manager: MentionManager,
        working_directory: Path,
    ) -> None:
        self._agent_runner = agent_runner
        self._agent = agent_runner.agent
        self._session_manager = agent_runner.session_manager
        self._model_selection = agent_runner.model_selection
        self._compaction = agent_runner.compaction
        self._interaction = agent_runner.interaction
        self._command_manager = command_manager
        self._completion_manager = completion_manager
        self._session_name_generator = session_name_generator
        self._mention_manager = mention_manager
        self._instructions_manager = agent_runner.instructions_manager
        self._permission_manager = agent_runner.permission_manager
        self._backend = agent_runner.backend
        self._working_directory = working_directory
        self._stream = agent_runner.stream
        self._debug = agent_runner.debug

    @classmethod
    def create_default(
        cls,
        backend: Backend,
        *,
        agent_name: str = "Loop",
        model: str | None = None,
        on_model_select: Callable[[str], None] | None = None,
        instructions_manager: InstructionsManager | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
        tool_registry: ToolRegistry | None = None,
        mention_manager: MentionManager | None = None,
        working_directory: Path | str | None = None,
        agents_filenames: tuple[str, ...] = (constants.DEFAULT_AGENTS_FILENAME,),
        session: Session | str | None = None,
        session_manager: SessionManager | None = None,
        session_name_generator: SessionNameGenerator | None = None,
        stream: bool = False,
        debug: bool = False,
        compaction_threshold: float = constants.DEFAULT_COMPACTION_THRESHOLD,
        prompt_on_recoverable_error: bool = True,
        max_agent_turns: int = constants.DEFAULT_AGENT_MAX_TURNS,
    ) -> Self:
        """Assemble an interactive loop with project defaults.

        Args:
            backend (Backend): Backend used to request model responses.
            agent_name (str): Human-readable identity. Defaults to ``"Loop"``.
            model (str | None): Explicit model, or ``None`` for the backend default.
            on_model_select (Callable[[str], None] | None): Durable preference writer invoked
                before an explicit model selection takes effect. Defaults to no durable writer.
            instructions_manager (InstructionsManager | None): Injected contextual instruction
                service, or ``None`` to discover one for the workspace.
            interaction (Interaction | None): User interaction service.
            permission_manager (PermissionManager | None): Tool authorization service.
            tool_registry (ToolRegistry | None): Tools exposed by the default agent.
            mention_manager (MentionManager | None): Mention resolution service.
            working_directory (Path | str | None): Initial workspace directory.
            agents_filenames (tuple[str, ...]): Project instruction filenames in precedence order.
            session (Session | str | None): Initial session or persisted identifier.
            session_manager (SessionManager | None): Injected session state owner.
            session_name_generator (SessionNameGenerator | None): Injected session naming service.
            stream (bool): Whether model response events are streamed.
            debug (bool): Whether raw model response events are displayed.
            compaction_threshold (float): Utilization threshold for automatic compaction.
            prompt_on_recoverable_error (bool): Whether recoverable errors prompt after retries.
            max_agent_turns (int): Maximum model turns per run, or zero for unlimited turns.

        Returns:
            Loop: Fully assembled interactive application.
        """
        if session_manager is not None:
            configured_sessions = session_manager
            if session:
                configured_sessions.load_session(session)
        else:
            configured_sessions = SessionManager(
                interaction=interaction,
                session=session,
            )
        configured_interaction = interaction or configured_sessions.interaction
        configured_name_generator = session_name_generator or BackendSessionNameGenerator(backend)

        restored_directory = configured_sessions.session.instruction_working_directory
        configured_directory = Path(working_directory or restored_directory or Path.cwd()).resolve()
        configured_permissions = permission_manager or PermissionManager(
            find_project_root(configured_directory) or configured_directory,
            interaction=configured_interaction,
        )
        configured_instructions = instructions_manager or InstructionsManager.discover(
            configured_directory,
            agents_filenames=agents_filenames,
        )
        if instructions_manager is None:
            configured_instructions.reactivate_skills(configured_sessions.session.active_skills)
        configured_instructions.set_runtime_environment(
            RuntimeEnvironment(
                working_directory=configured_directory,
                temporary_directory=configured_permissions.temporary_directory,
            )
        )
        configured_permissions.recorder = configured_sessions
        configured_tools = tool_registry or ToolRegistry(
            interaction=configured_interaction,
            permission_manager=configured_permissions,
        )
        configured_agent = Agent(agent_name, tools=configured_tools)
        configured_instructions.prepare(configured_agent)
        configured_selection = ModelSelection(
            backend,
            configured_sessions,
            selected=(
                configured_sessions.model if configured_sessions.model is not None else model
            ),
            on_select=on_model_select,
        )
        application: Loop
        configured_compaction = ContextCompaction(
            backend,
            configured_sessions,
            configured_selection,
            lambda: configured_instructions.prepare(configured_agent),
            configured_interaction,
            lambda: application.working_directory,
            threshold=compaction_threshold,
        )
        configured_runner = AgentRunner(
            configured_agent,
            backend,
            configured_instructions,
            configured_permissions,
            configured_sessions,
            configured_selection,
            configured_compaction,
            configured_interaction,
            lambda: application.working_directory,
            stream=stream,
            debug=debug,
            max_turns=max_agent_turns,
            prompt_on_recoverable_error=prompt_on_recoverable_error,
        )
        providers = (
            SessionCommands(configured_sessions, configured_name_generator),
            PermissionCommands(configured_permissions),
            SkillCommands(configured_instructions),
            ToolCommands(configured_agent.tools, configured_instructions),
            ModelCommands(configured_selection),
            CompactionCommands(configured_compaction),
        )
        configured_commands = CommandManager(
            providers=providers,
            interaction=configured_interaction,
        )
        configured_mentions = mention_manager or MentionManager(
            (
                ProjectPathMentionHandler(lambda: application.working_directory),
                SkillMentionHandler(configured_instructions),
            )
        )
        configured_completion = CompletionManager(
            (
                CommandCompletionAdapter(
                    lambda: configured_commands.commands,
                    providers=providers,
                ),
                *configured_mentions.completion_adapters,
            )
        )
        application = cls(
            agent_runner=configured_runner,
            command_manager=configured_commands,
            completion_manager=configured_completion,
            session_name_generator=configured_name_generator,
            mention_manager=configured_mentions,
            working_directory=configured_directory,
        )
        return application

    @property
    def command_manager(self) -> CommandManager:
        """Return the registered interactive command manager.

        Returns:
            CommandManager: Active command manager.
        """
        return self._command_manager

    @property
    def permission_manager(self) -> PermissionManager:
        """Return the permission manager guarding model tool calls.

        Returns:
            PermissionManager: Active local permission manager.
        """
        return self._permission_manager

    @property
    def agent(self) -> Agent:
        """Return the agent used for model and tool execution.

        Returns:
            Agent: Agent configured for this conversation.
        """
        return self._agent

    @property
    def agent_runner(self) -> AgentRunner:
        """Return the runner responsible for model and tool turns.

        Returns:
            AgentRunner: Runner configured for this conversation.
        """
        return self._agent_runner

    @property
    def tool_registry(self) -> ToolRegistry:
        """Return the tools active for this conversation.

        Returns:
            ToolRegistry: Agent-scoped tool declarations and implementations.
        """
        return self._agent.tools

    @property
    def backend(self) -> Backend:
        """Return the backend used to request model responses.

        Returns:
            Backend: The configured LLM backend.
        """
        return self._backend

    @property
    def instructions(self) -> str:
        """Return the complete instructions for the next backend request.

        Returns:
            str: Agent instructions (identity, intrinsic instructions), project, runtime, catalog,
                and activated-skill instructions. Accessing this property refreshes stale project
                sources before composing the document.
        """
        return self._instructions_manager.prepare(self._agent).content

    @property
    def instructions_manager(self) -> InstructionsManager:
        """Return the instruction manager active for this conversation.

        Returns:
            InstructionsManager: The active instruction manager.
        """
        return self._instructions_manager

    @property
    def messages(self) -> list[ConversationItem]:
        """Return the current conversation history.

        Returns:
            list[ConversationItem]: Items accumulated during the conversation.
        """
        return self._session_manager.messages

    @property
    def session(self) -> Session:
        """Return the active session.

        Returns:
            Session: The mutable session used by the loop.
        """
        return self._session_manager.session

    @property
    def interaction(self) -> Interaction:
        """Return the service used for user input and output.

        Returns:
            Interaction: The configured interaction service.
        """
        return self._interaction

    @property
    def working_directory(self) -> Path:
        """Return the directory used to discover project instructions.

        Returns:
            Path: The resolved working directory.
        """
        return self._working_directory

    def set_working_directory(self, working_directory: Path | str) -> None:
        """Change the working and instruction-discovery directory.

        Args:
            working_directory (Path | str): Existing directory to make active.

        Raises:
            NotADirectoryError: If the resolved path is not an existing directory.
        """
        directory = Path(working_directory).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Working directory '{directory}' does not exist.")
        self._working_directory = directory
        self._instructions_manager.observe_path(directory, directory=True)

    @property
    def debug(self) -> bool:
        """Return whether raw response event output is enabled.

        Returns:
            bool: Whether debug output is enabled.
        """
        return self._agent_runner.debug

    @debug.setter
    def debug(self, debug: bool) -> None:
        """Enable or disable raw response event output.

        Args:
            debug (bool): Whether to enable debug output.
        """
        self._debug = debug
        self._agent_runner.debug = debug

    @property
    def stream(self) -> bool:
        """Return whether responses are requested incrementally.

        Returns:
            bool: Whether response streaming is enabled.
        """
        return self._agent_runner.stream

    @property
    def model(self) -> str | None:
        """Return the model explicitly selected for requests.

        Returns:
            str | None: The selected model, or ``None`` when the backend default is used.
        """
        return self._model_selection.selected

    def select_model(self, model: str) -> None:
        """Select a model for subsequent requests and refresh its context limit.

        Args:
            model (str): Exact backend model identifier to select.
        """
        self._model_selection.select(model)

    def run(self):
        """Run the conversation until the user requests to exit."""
        recovery_pending = self._recover_session()
        while not self._command_manager.exit_requested:
            user_input = self._interaction.prompt(completer=self._completion_manager)
            if user_input is False:
                break
            session = self.session
            if self._command_manager.handle_user_command(user_input):
                if self.session is not session:
                    self._model_selection.restore(self.session.model)
                    recovery_pending = self._recover_session()
                continue
            if recovery_pending:
                self._interaction.warning(
                    "Recover the interrupted run or start a new session before sending a message."
                )
                continue
            try:
                context = self._mention_manager.resolve(user_input)
            except (OSError, UnicodeError, ValueError) as error:
                problem = Problem.from_exception(
                    error,
                    code="mention.resolution_failed",
                    title="Could not resolve mention",
                    operation="resolve_mention",
                )
                log_problem(_LOGGER, problem, error)
                self._interaction.report(problem)
                continue
            with self._session_manager.next_message_span():
                telemetry_activity("message.accepted", component="session_manager")
                self._session_manager.add_user_message(user_input, context=context)
                result = self._agent_runner.run()
                self._complete_agent_run(result)

        self._interaction.conversation_ended()

    def _complete_agent_run(self, result: AgentRunResult) -> None:
        """Present metrics and finalize naming for one completed or recovered run."""
        if result.metrics is None:
            raise TypeError("Agent run did not produce completion metrics.")
        self._interaction.run_metrics(result.metrics)
        if result.stop_reason != "completed":
            return

        if self._session_manager.session.has_initial_name():
            self._session_manager.generate_session_name(self._session_name_generator)

    def _recover_session(self) -> bool:
        """Recover a previously interrupted run, if any."""
        recovery = self._agent_runner.recover_session()
        recovery_pending = recovery.pending
        if recovery.result is not None:
            self._complete_agent_run(recovery.result)
        return recovery_pending
