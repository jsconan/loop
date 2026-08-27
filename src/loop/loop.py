"""Run an interactive conversation with an LLM backend."""

from pathlib import Path

from . import constants
from .agent import Agent, AgentRunner, AgentRunResult
from .backend import Backend
from .commands import CommandManager
from .compaction import CompactionCommands, ContextCompaction
from .completion import (
    CommandCompletionAdapter,
    CompletionManager,
)
from .errors import Problem
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
from .skills import InstructionsManager, RuntimeEnvironment, SkillCommands
from .tooling import ToolCommands, ToolRegistry
from .utils import find_project_root


class Loop:
    """Run an interactive conversation using normalized response events.

    Args:
        backend (Backend): Backend used to request model responses.
        agent_name (str): Human-readable identity for the configured agent. Defaults to
            ``"Assistant"``.
        model (str | None): Model selected for requests, or ``None`` to use the backend default.
        instructions_manager (InstructionsManager | None): Manager used to compose the complete
            backend instructions. Defaults to discovering project instructions and Agent Skills
            for ``working_directory``.
        interaction (Interaction | None): Service used for all user input and output.
        permission_manager (PermissionManager | None): Manager used to authorize model tool calls.
            Defaults to loading local policy from the project ``.loop`` folder.
        tool_registry (ToolRegistry | None): Agent-scoped tools exposed to the model and used to
            dispatch its calls. Defaults to an empty registry.
        mention_manager (MentionManager | None): Injected mention capability registry. Defaults to
            live project-path and skill handlers.
        working_directory (Path | str | None): Directory used to discover applicable AGENTS.md
            files.
        agents_filenames (tuple[str, ...]): Ordered instruction filenames, where a later name is
            used only when earlier names are absent in the same directory.
        session (Session | str | None): Session or persisted session identifier to load.
            Defaults to a fresh session; an injected store persists it after its first query.
        session_manager (SessionManager | None): Manager used to persist and retrieve sessions.
            Defaults to an instance-local memory store when ``None`` is provided.
        session_name_generator (SessionNameGenerator | None): Service used to improve the
            provisional name after the first response. Defaults to the conversation backend.
        stream (bool): Whether the backend should produce response events incrementally.
        debug (bool): Whether to print raw response events.
        compaction_threshold (float): Context-window utilization that triggers automatic
            compaction. Defaults to ``0.8``.
        prompt_on_recoverable_error (bool): Whether to offer an interactive retry after automatic
            retries exhaust a recoverable backend failure.
        max_agent_turns (int): Maximum model turns permitted for one user input. Defaults to
            ``25``.

    Raises:
        ValueError: If the session, agent name, compaction threshold, or model-turn limit is
            invalid.

    """

    _backend: Backend
    _instructions_manager: InstructionsManager
    _session_manager: SessionManager
    _interaction: Interaction
    _working_directory: Path
    _debug: bool
    _stream: bool
    _model_selection: ModelSelection
    _compaction: ContextCompaction
    _command_manager: CommandManager
    _permission_manager: PermissionManager
    _tool_registry: ToolRegistry
    _completion_manager: CompletionManager
    _session_name_generator: SessionNameGenerator
    _mention_manager: MentionManager
    _prompt_on_recoverable_error: bool
    _agent: Agent
    _agent_runner: AgentRunner

    def __init__(
        self,
        backend: Backend,
        *,
        agent_name: str = "Assistant",
        model: str | None = None,
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
    ) -> None:
        if session_manager is not None:
            self._session_manager = session_manager
            if session:
                self._session_manager.load_session(session)
        else:
            self._session_manager = SessionManager(
                interaction=interaction,
                session=session,
            )
        self._interaction = interaction or self._session_manager.interaction
        self._backend = backend
        self._session_name_generator = session_name_generator or BackendSessionNameGenerator(
            backend
        )

        restored_directory = self._session_manager.session.instruction_working_directory
        self._working_directory = Path(
            working_directory or restored_directory or Path.cwd()
        ).resolve()
        self._permission_manager = permission_manager or PermissionManager(
            find_project_root(self._working_directory) or self._working_directory,
            interaction=self._interaction,
        )
        self._instructions_manager = instructions_manager or InstructionsManager.discover(
            self._working_directory,
            agents_filenames=agents_filenames,
        )
        if instructions_manager is None:
            self._instructions_manager.reactivate_skills(
                self._session_manager.session.active_skills
            )
        self._instructions_manager.set_runtime_environment(
            RuntimeEnvironment(
                working_directory=self._working_directory,
                temporary_directory=self._permission_manager.temporary_directory,
            )
        )
        self._stream = stream
        self._debug = debug
        self._model_selection = ModelSelection(
            self._backend,
            self._session_manager,
            selected=model,
        )
        self._compaction = ContextCompaction(
            self._backend,
            self._session_manager,
            self._model_selection,
            self._instructions_manager,
            self._interaction,
            lambda: self._working_directory,
            threshold=compaction_threshold,
        )
        self._prompt_on_recoverable_error = prompt_on_recoverable_error
        self._permission_manager.recorder = self._session_manager
        self._tool_registry = tool_registry or ToolRegistry(
            interaction=self._interaction,
            permission_manager=self._permission_manager,
        )
        self._agent = Agent(
            agent_name,
            self._backend,
            self._instructions_manager,
            self._tool_registry,
            self._permission_manager,
        )
        self._agent_runner = AgentRunner(
            self._agent,
            self._session_manager,
            self._model_selection,
            self._compaction,
            self._interaction,
            lambda: self._working_directory,
            stream=stream,
            debug=debug,
            max_turns=max_agent_turns,
            prompt_on_recoverable_error=prompt_on_recoverable_error,
        )
        providers = (
            SessionCommands(self._session_manager),
            PermissionCommands(self._permission_manager),
            SkillCommands(self._instructions_manager),
            ToolCommands(self._tool_registry, self._instructions_manager),
            ModelCommands(self._model_selection),
            CompactionCommands(self._compaction),
        )
        self._command_manager = CommandManager(
            providers=providers,
            interaction=self._interaction,
        )
        self._mention_manager = mention_manager or MentionManager(
            (
                ProjectPathMentionHandler(lambda: self._working_directory),
                SkillMentionHandler(self._instructions_manager),
            )
        )
        self._completion_manager = CompletionManager(
            (
                CommandCompletionAdapter(
                    lambda: self._command_manager.commands,
                    providers=providers,
                ),
                *self._mention_manager.completion_adapters,
            )
        )

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
        return self._tool_registry

    @property
    def backend(self) -> Backend:
        """Return the backend used to request model responses.

        Returns:
            Backend: The configured LLM backend.
        """
        return self._backend

    @property
    def instructions(self) -> str | None:
        """Return the complete instructions for the next backend request.

        Returns:
            str | None: Project, catalog, and activated-skill instructions, or ``None`` when no
                instructions are available.
        """
        return self._instructions_manager.instructions

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
                self._interaction.report(
                    Problem.from_exception(
                        error,
                        code="mention.resolution_failed",
                        title="Could not resolve mention",
                        operation="resolve_mention",
                    )
                )
                continue
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
