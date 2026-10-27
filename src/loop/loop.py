"""Run an interactive conversation with an LLM backend."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import get_ident
from typing import Self

from . import constants
from .agent import Agent, AgentRunner, AgentRunResult
from .backend import Backend, GenerationHyperparameters
from .commands import CommandManager, CommandsProvider
from .compaction import CompactionCommands, ContextCompaction
from .completion import (
    CompletionManager,
    CompletionValue,
    MarkerCompletionAdapter,
)
from .configuration import ApplicationSettings
from .control import RunControl
from .conversation.models import AdmittedConversationInput
from .errors import Problem, log_problem
from .execution.sandbox import SandboxCommands
from .instructions import InstructionsManager, RuntimeEnvironment, SkillCommands
from .interaction import Interaction, require_async_interaction
from .mentions import MentionManager, ProjectPathMentionHandler
from .mentions.parser import parse_mentions
from .model_selection import ModelCommands, ModelSelection
from .models import (
    AgentRunStopReason,
    ConversationItem,
    ReasoningEffort,
    SteeringMessage,
    SyncRecoveryRequested,
)
from .permissions import PermissionCommands, PermissionManager
from .session import (
    BackendSessionNameGenerator,
    ResponsePersisted,
    Session,
    SessionCommands,
    SessionManager,
    SessionNameGenerator,
)
from .telemetry import telemetry_activity
from .tooling import ToolCommands, ToolRegistry
from .utils import PathHolder, PathReference, find_project_root, has_running_event_loop

_LOGGER = logging.getLogger(__name__)


class Loop:
    """Run an interactive conversation using normalized response events.

    Args:
        agent_runner (AgentRunner): Runtime executing the agent.
        command_manager (CommandManager): Registered interactive commands.
        completion_manager (CompletionManager): Interactive completion service.
        session_name_generator (SessionNameGenerator): Session naming service.
        mention_manager (MentionManager): Mention resolution service.
        working_directory (PathReference): Current workspace-directory reference.
        owns_session_name_generator (bool): Whether the naming service is a backend-dependent
            default owned by this loop. Defaults to ``False`` for explicitly assembled loops.
        skill_hints_enabled (bool): Whether ``$skill`` references prioritize matching metadata for
            one default-configured request. Defaults to ``False`` for explicitly assembled loops.
        run_control (RunControl | None): Optional stop signal for the active conversation run.
            Defaults to no external stop signal.

    """

    _session_manager: SessionManager
    _interaction: Interaction
    _working_directory: PathReference
    _model_selection: ModelSelection
    _compaction: ContextCompaction
    _command_manager: CommandManager
    _completion_manager: CompletionManager
    _session_name_generator: SessionNameGenerator
    _mention_manager: MentionManager
    _agent: Agent
    _agent_runner: AgentRunner
    _owns_session_name_generator: bool
    _skill_hints_enabled: bool
    _run_control: RunControl | None
    _session_name_task_group: asyncio.TaskGroup | None
    _session_name_tasks: dict[str, asyncio.Task[None]]
    _session_name_attempted_this_run: bool
    _sync_command_thread: int | None

    def __init__(
        self,
        *,
        agent_runner: AgentRunner,
        command_manager: CommandManager,
        completion_manager: CompletionManager,
        session_name_generator: SessionNameGenerator,
        mention_manager: MentionManager,
        working_directory: Path | PathReference,
        owns_session_name_generator: bool = False,
        skill_hints_enabled: bool = False,
        run_control: RunControl | None = None,
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
        self._owns_session_name_generator = owns_session_name_generator
        self._skill_hints_enabled = skill_hints_enabled
        self._run_control = run_control
        self._session_name_task_group = None
        self._session_name_tasks = {}
        self._session_name_attempted_this_run = False
        self._sync_command_thread = None
        self._mention_manager = mention_manager
        self._instructions_manager = agent_runner.instructions_manager
        self._permission_manager = agent_runner.permission_manager
        self._backend = agent_runner.backend
        self._working_directory = PathHolder.from_value(working_directory)

    @classmethod
    def create_default(
        cls,
        backend: Backend,
        *,
        agent_name: str = "Loop",
        model: str | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        on_model_select: Callable[[str], None] | None = None,
        instructions_manager: InstructionsManager | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
        tool_registry: ToolRegistry | None = None,
        mention_manager: MentionManager | None = None,
        working_directory: Path | str | PathReference | None = None,
        agents_filenames: tuple[str, ...] = (constants.DEFAULT_AGENTS_FILENAME,),
        session: Session | str | None = None,
        session_manager: SessionManager | None = None,
        session_name_generator: SessionNameGenerator | None = None,
        stream: bool = False,
        debug: bool = False,
        compaction_threshold: float = constants.DEFAULT_COMPACTION_THRESHOLD,
        prompt_on_recoverable_error: bool = True,
        max_agent_turns: int = constants.DEFAULT_AGENT_MAX_TURNS,
        run_control: RunControl | None = None,
        main_app: bool = False,
        refresh_sandbox_backend: Callable[[], None] | None = None,
    ) -> Self:
        """Assemble an interactive loop with project defaults.

        Args:
            backend (Backend): Backend used to request model responses.
            agent_name (str): Human-readable identity. Defaults to ``"Loop"``.
            model (str | None): Explicit model, or ``None`` for the backend default.
            temperature (float | None): Sampling temperature for this agent's model requests.
            reasoning_effort (ReasoningEffort | None): Reasoning effort for this agent's model
                requests.
            on_model_select (Callable[[str], None] | None): Durable preference writer invoked
                before an explicit model selection takes effect. Defaults to no durable writer.
            instructions_manager (InstructionsManager | None): Injected contextual instruction
                service, or ``None`` to discover one for the workspace.
            interaction (Interaction | None): User interaction service.
            permission_manager (PermissionManager | None): Tool authorization service.
            tool_registry (ToolRegistry | None): Tools exposed by the default agent.
            mention_manager (MentionManager | None): Mention resolution service.
            working_directory (Path | str | PathReference | None): Initial directory or stable
                workspace-owned reference.
            agents_filenames (tuple[str, ...]): Project instruction filenames in precedence order.
            session (Session | str | None): Initial session or persisted identifier.
            session_manager (SessionManager | None): Injected session state owner.
            session_name_generator (SessionNameGenerator | None): Injected session naming service.
            stream (bool): Whether model response events are streamed.
            debug (bool): Whether raw model events are displayed.
            compaction_threshold (float): Automatic compaction threshold.
            prompt_on_recoverable_error (bool): Whether recoverable errors prompt after retries.
            max_agent_turns (int): Maximum turns, or zero for unlimited.
            run_control (RunControl | None): Optional cooperative stop signal for model and tool
                boundaries. Defaults to no external cancellation signal.
            main_app (bool): Enable direct main-app terminal management review; False
                (default) prevents library command dispatch from elevating.
            refresh_sandbox_backend (Callable[[], None] | None): Reselect the command execution
                backend after approved Linux setup changes, or ``None`` when not composed.

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
        owns_name_generator = session_name_generator is None
        configured_name_generator = session_name_generator or BackendSessionNameGenerator(backend)

        restored_directory = configured_sessions.session.instruction_working_directory
        if isinstance(working_directory, PathReference):
            directory_reference = working_directory
            configured_directory = working_directory.resolve()
        else:
            configured_directory = Path(
                working_directory or restored_directory or Path.cwd()
            ).resolve()
            directory_reference = PathHolder(configured_directory)
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
        configured_permissions.bind_temporary_directory(
            configured_sessions.temporary_directory,
            ordinary_names=configured_sessions.working_directory_allows_private_names,
        )
        configured_instructions.set_runtime_environment(
            RuntimeEnvironment(
                working_directory=configured_directory,
                temporary_directory=configured_sessions.temporary_directory,
            )
        )
        configured_permissions.recorder = configured_sessions
        configured_tools = tool_registry or ToolRegistry(
            interaction=configured_interaction,
            permission_manager=configured_permissions,
        )
        configured_agent = Agent(
            agent_name,
            tools=configured_tools,
            hyperparameters=GenerationHyperparameters(
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            ),
        )
        configured_instructions.prepare(configured_agent)
        configured_commands = CommandManager(interaction=configured_interaction)
        configured_sessions.set_async_blocking_runner(configured_commands.run_blocking_async)
        configured_permissions.set_async_blocking_runner(configured_commands.run_blocking_async)
        configured_selection = ModelSelection(
            backend,
            configured_sessions,
            selected=(
                configured_sessions.model if configured_sessions.model is not None else model
            ),
            on_select=on_model_select,
            async_blocking_runner=configured_commands.run_blocking_async,
        )
        configured_compaction = ContextCompaction(
            backend,
            configured_sessions,
            configured_selection,
            lambda: configured_instructions.prepare(configured_agent),
            configured_interaction,
            directory_reference,
            threshold=compaction_threshold,
            instructions_async=lambda: configured_instructions.prepare_async(configured_agent),
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
            directory_reference,
            stream=stream,
            debug=debug,
            max_turns=max_agent_turns,
            run_control=run_control,
            prompt_on_recoverable_error=prompt_on_recoverable_error,
            async_blocking_runner=configured_commands.run_blocking_async,
        )
        providers = (
            SessionCommands(configured_sessions, configured_name_generator),
            PermissionCommands(configured_permissions),
            SkillCommands(configured_instructions),
            ToolCommands(configured_tools, configured_instructions),
            ModelCommands(configured_selection),
            CompactionCommands(configured_compaction),
            SandboxCommands(
                main_app=main_app,
                refresh_backend=refresh_sandbox_backend,
            ),
        )
        configured_commands.register_providers(providers)
        skill_hints_enabled = mention_manager is None
        configured_mentions = mention_manager or MentionManager(
            (
                ProjectPathMentionHandler(
                    directory_reference,
                    sync_blocking_runner=configured_commands.run_blocking,
                    async_blocking_runner=configured_commands.run_blocking_async,
                ),
            )
        )
        completion_adapters = list(configured_mentions.completion_adapters)
        if skill_hints_enabled:
            completion_adapters.append(
                MarkerCompletionAdapter(
                    "$",
                    lambda: tuple(
                        CompletionValue(skill.name, skill.description)
                        for skill in configured_instructions.skill_manager.skills
                    ),
                )
            )
        configured_completion = CompletionManager(
            (
                configured_commands.completion_adapter,
                *completion_adapters,
            )
        )
        return cls(
            agent_runner=configured_runner,
            command_manager=configured_commands,
            completion_manager=configured_completion,
            session_name_generator=configured_name_generator,
            mention_manager=configured_mentions,
            working_directory=directory_reference,
            owns_session_name_generator=owns_name_generator,
            skill_hints_enabled=skill_hints_enabled,
            run_control=run_control,
        )

    @property
    def command_manager(self) -> CommandManager:
        """Return the registered interactive command manager.

        Returns:
            CommandManager: Active command manager.
        """
        return self._command_manager

    def register_command_provider(self, provider: CommandsProvider) -> None:
        """Register a command provider for dispatch and contextual completion.

        Args:
            provider (CommandsProvider): Provider whose commands and completion sources should be
                added to this loop.
        """
        self._command_manager.register_provider(provider)

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
        """Return the registry backing the active agent's live tool view.

        Returns:
            ToolRegistry: Single owner of the agent's declarations and execution dependencies.
        """
        return self._agent.tools.registry

    def apply_runtime_settings(self, path: str, settings: ApplicationSettings) -> str:
        """Apply a supported live setting to the active loop.

        Args:
            path (str): Changed dot-separated configuration path.
            settings (ApplicationSettings): Newly validated effective application settings.

        Returns:
            str: Whether the setting was applied now or requires restart.
        """
        if path == "loop.debug":
            self._agent_runner.debug = settings.loop.debug
        elif path == "loop.stream":
            self._agent_runner.stream = settings.loop.stream
        elif path == "loop.max_agent_turns":
            self._agent_runner.max_turns = settings.loop.max_agent_turns
        elif path == "loop.prompt_on_recoverable_error":
            self._agent_runner.prompt_on_recoverable_error = (
                settings.loop.prompt_on_recoverable_error
            )
        elif path == "loop.compaction_threshold":
            self._compaction.threshold = settings.loop.compaction_threshold
        elif path == "web.user_agent":
            self.tool_registry.settings.user_agent = settings.web.user_agent
        elif path == "tools.command_timeout":
            self.tool_registry.settings.command_timeout = settings.tools.command_timeout
        elif path == "loop.model":
            self._model_selection.restore(settings.loop.model)
        elif path == "loop.temperature":
            self._agent_runner.agent.hyperparameters.temperature = settings.loop.temperature

        elif path == "loop.reasoning_effort":
            self._agent_runner.agent.hyperparameters.reasoning_effort = (
                settings.loop.reasoning_effort
            )
        else:
            return "saved; restart required"
        return "applied now"

    async def apply_runtime_settings_async(self, path: str, settings: ApplicationSettings) -> str:
        """Apply a supported live setting through native async services when needed.

        Args:
            path (str): Changed dot-separated configuration path.
            settings (ApplicationSettings): Newly validated effective application settings.

        Returns:
            str: Whether the setting was applied now or requires restart.

        Raises:
            asyncio.CancelledError: If asynchronous model metadata lookup is cancelled.
        """
        if path == "loop.model":
            await self._model_selection.restore_async(settings.loop.model)
            return "applied now"
        return self.apply_runtime_settings(path, settings)

    def replace_backend(self, backend: Backend) -> None:
        """Replace the shared backend used by all future loop operations.

        Args:
            backend (Backend): Fully configured replacement backend.
        """
        self._set_backend(backend)
        self._model_selection.restore(self._model_selection.selected)

    async def replace_backend_async(self, backend: Backend) -> None:
        """Replace the backend and refresh model metadata through its async API.

        Args:
            backend (Backend): Fully configured replacement backend.

        Raises:
            asyncio.CancelledError: If asynchronous model metadata lookup is cancelled.
        """
        self._set_backend(backend)
        await self._model_selection.restore_async(self._model_selection.selected)

    def _set_backend(self, backend: Backend) -> None:
        """Publish one backend to every runtime component that owns it."""
        self._backend = backend
        self._agent_runner.backend = backend
        self._model_selection.backend = backend
        self._compaction.backend = backend
        if self._owns_session_name_generator and isinstance(
            self._session_name_generator, BackendSessionNameGenerator
        ):
            self._session_name_generator.backend = backend

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
    def recovery_pending(self) -> bool:
        """Return whether the active session blocks ordinary input until recovery.

        Returns:
            bool: Whether an unresolved run requires recovery.
        """
        return self._session_manager.recovery_pending

    @property
    def continuable(self) -> bool:
        """Return whether a live pause or durable boundary can be continued.

        Returns:
            bool: Whether ``/continue`` can resume the current run state.
        """
        return (
            self._run_control is not None and self._run_control.paused
        ) or self._session_manager.continuable

    @property
    def retry_pending(self) -> bool:
        """Return whether a failed event authorizes an explicit retry.

        Returns:
            bool: Whether ``/retry`` is available for the active session.
        """
        return self._session_manager.retry_pending

    def recover_pending(self, *, retry: bool = False) -> bool:
        """Recover a durable boundary on the synchronous conversation owner.

        Args:
            retry (bool): Persist authorization for a recorded failure before recovery.
                Defaults to ``False`` for continuation.

        Returns:
            bool: Whether recovery remains pending after the attempt.

        Raises:
            RuntimeError: If invoked from an active event loop.
            SessionExecutionConflictError: If another execution owns the session.
        """
        if has_running_event_loop():
            raise RuntimeError("Await Loop.recover_pending_async() inside an event loop.")
        if not retry and self._run_control is not None and self._run_control.continue_run():
            return False
        with self._session_manager.execution():
            if retry:
                if not self._session_manager.retry_pending:
                    return False
                self._session_manager.begin_run_retry()
            elif not self._session_manager.continuable:
                return False
            return self._recover_session(explicit=True)

    async def recover_pending_async(self, *, retry: bool = False) -> bool:
        """Continue a live pause or recover a durable boundary after a command.

        Args:
            retry (bool): Persist authorization for a recorded failure before recovery.
                Defaults to ``False`` for continuation.

        Returns:
            bool: Whether recovery remains pending after the attempt.

        Raises:
            asyncio.CancelledError: If the recovery operation is cancelled.
            SyncRecoveryRequested: Return recovery to a synchronous command's existing owner.
        """
        if self._sync_command_thread == get_ident():
            raise SyncRecoveryRequested(retry)
        if not retry and self._run_control is not None and self._run_control.continue_run():
            return False
        with self._session_manager.execution():
            if retry:
                if not self._session_manager.retry_pending:
                    return False
                await self._session_manager.begin_run_retry_async()
            elif not self._session_manager.continuable:
                return False
            async with self._session_name_task_scope_async():
                return await self._recover_session_async(explicit=True)

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
        return self._working_directory.get()

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
        self._working_directory.set(directory)
        self._instructions_manager.set_working_directory(directory)

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
        self._agent_runner.debug = debug

    @property
    def stream(self) -> bool:
        """Return whether responses are requested incrementally.

        Returns:
            bool: Whether response streaming is enabled.
        """
        return self._agent_runner.stream

    @stream.setter
    def stream(self, stream: bool) -> None:
        """Enable or disable streaming for subsequent backend requests.

        Args:
            stream (bool): Whether subsequent responses should stream.
        """
        self._agent_runner.stream = stream

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

    def restore_model(self, model: str | None) -> None:
        """Restore a configured model selection and synchronize session metadata.

        Args:
            model (str | None): Explicit model or ``None`` to use the backend default.
        """
        self._model_selection.restore(model)

    def run(self, *, input_source: Callable[[CompletionManager], object] | None = None) -> None:
        """Run the conversation until the user requests to exit.

        Args:
            input_source (Callable[[CompletionManager], object] | None): Optional mailbox reader
                for ordinary conversation input. It receives the live completion manager and
                returns text or ``False`` to exit. Nested prompts still use the interaction.
                ``SteeringMessage`` values use normal message preparation without command
                dispatch. Defaults to synchronous interaction prompts.

        Raises:
            RuntimeError: If called from an active event loop.
            SessionExecutionConflictError: If another owner is executing this session.
            TypeError: If an agent run or recovered run has no completion metrics.
            KeyboardInterrupt: If the configured stop signal interrupts agent execution.
            WorkspaceSwitchRequested: If a command requests a workspace runtime rebuild.

        Returns:
            None: After the conversation has ended.
        """
        if has_running_event_loop():
            raise RuntimeError(
                "Synchronous Loop.run() cannot run inside an event loop; "
                "await Loop.run_async() instead."
            )

        with self._session_manager.execution():
            self._recover_session()
        while not self._command_manager.exit_requested:
            with self._session_manager.execution(allow_session_selection=True):
                user_input = (
                    self._interaction.prompt(completer=self._completion_manager)
                    if input_source is None
                    else input_source(self._completion_manager)
                )
                if user_input is False:
                    break
                session = self.session
                admitted_input = (
                    user_input if isinstance(user_input, AdmittedConversationInput) else None
                )
                if admitted_input is not None:
                    if admitted_input.session_id != self._session_manager.session.id:
                        self._interaction.warning(
                            "Queued input belongs to a different session and was skipped."
                        )
                        continue
                    user_input = admitted_input.message
                handled_command = False
                bypass_commands = isinstance(user_input, SteeringMessage) or (
                    admitted_input is not None and admitted_input.bypass_commands
                )
                if not bypass_commands:
                    self._sync_command_thread = get_ident()
                    try:
                        handled_command = self._command_manager.handle_user_command(user_input)
                    except SyncRecoveryRequested as request:
                        pending = self.recover_pending(retry=request.retry)
                        if pending:
                            self._interaction.warning(
                                "Recovery remains pending; use /retry to try again."
                            )
                        else:
                            self._interaction.info(
                                "Recovery completed." if request.retry else "Run continued."
                            )
                        handled_command = True
                    finally:
                        self._sync_command_thread = None
                if handled_command:
                    if self.session is not session:
                        with self._session_manager.execution():
                            self._model_selection.restore(self.session.model)
                            self._recover_session()
                    continue
                if self._skill_hints_enabled:
                    try:
                        skill_mentions = tuple(
                            dict.fromkeys(
                                mention.value
                                for mention in parse_mentions(
                                    user_input,
                                    {"$": self._instructions_manager.skill_manager.names},
                                )
                            )
                        )
                        if not self._instructions_manager.set_skill_catalog_priorities(
                            skill_mentions, self._agent
                        ):
                            self._interaction.warning(
                                "Referenced skill metadata could not fit in the instruction "
                                "budget; use activate_skill to load a skill."
                            )
                    except (OSError, UnicodeError, ValueError) as error:
                        problem = Problem.from_exception(
                            error,
                            code="skill_hint.preparation_failed",
                            title="Could not prepare skill hint",
                            operation="prepare_skill_hint",
                        )
                        log_problem(_LOGGER, problem, error)
                        self._interaction.report(problem)
                        if admitted_input is not None:
                            self._session_manager.reject_input(admitted_input.input_id)
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
                    if admitted_input is not None:
                        self._session_manager.reject_input(admitted_input.input_id)
                    if self._skill_hints_enabled:
                        self._clear_skill_catalog_priorities()
                    continue
                try:
                    with self._session_manager.next_message_span():
                        for reference in context:
                            included = reference.included_bytes
                            size = reference.size_bytes
                            if included < size:
                                self._interaction.info(
                                    f"Attached {reference.path}: {included} of {size} bytes"
                                    "; additional content stays local until read."
                                )
                            else:
                                self._interaction.info(
                                    f"Attached {reference.path}: {included} bytes."
                                )
                        recovering = self._session_manager.recovery_pending
                        self._session_manager.add_user_message(
                            user_input,
                            context=context,
                            input_id=admitted_input.input_id
                            if admitted_input is not None
                            else None,
                        )
                        telemetry_activity("message.accepted", component="session_manager")
                        if recovering:
                            self._recover_session(explicit=True)
                        else:
                            result = self._agent_runner.run()
                            self._complete_agent_run(result)
                finally:
                    if self._skill_hints_enabled:
                        self._clear_skill_catalog_priorities()

        self._interaction.conversation_ended()

    async def run_async(
        self,
        *,
        input_source: Callable[[CompletionManager], Awaitable[object]] | None = None,
    ) -> None:
        """Run conversation orchestration and own title tasks through async input and execution.

        Args:
            input_source (Callable[[CompletionManager], Awaitable[object]] | None): Optional
                awaitable mailbox reader for ordinary conversation input. It receives the live
                completion manager and returns text or ``False`` to exit. Without it, the
                interaction must provide native asynchronous input.

        Raises:
            KeyboardInterrupt: If the configured stop signal interrupts agent execution.
            ProblemException: If the interaction does not provide asynchronous input.
            SessionExecutionConflictError: If another owner is executing this session.
            WorkspaceSwitchRequested: If a command requests a workspace runtime rebuild.
            asyncio.CancelledError: If the conversation task is cancelled.
        """
        async with self._session_name_task_scope_async():
            await self._run_async(input_source=input_source)

    async def _run_async(
        self,
        *,
        input_source: Callable[[CompletionManager], Awaitable[object]] | None = None,
    ) -> None:
        """Run conversation orchestration through asynchronous input and agent operations."""
        with self._session_manager.execution():
            await self._recover_session_async()
        while not self._command_manager.exit_requested:
            with self._session_manager.execution(allow_session_selection=True):
                if input_source is not None:
                    user_input = await input_source(self._completion_manager)
                else:
                    interaction = require_async_interaction(self._interaction, "conversation input")
                    user_input = await interaction.prompt_async(completer=self._completion_manager)
                if user_input is False:
                    break
                session = self.session
                admitted_input = (
                    user_input if isinstance(user_input, AdmittedConversationInput) else None
                )
                if admitted_input is not None:
                    if admitted_input.session_id != self._session_manager.session.id:
                        self._interaction.warning(
                            "Queued input belongs to a different session and was skipped."
                        )
                        continue
                    user_input = admitted_input.message
                handled_command = False
                bypass_commands = isinstance(user_input, SteeringMessage) or (
                    admitted_input is not None and admitted_input.bypass_commands
                )
                if not bypass_commands:
                    handled_command = await self._command_manager.handle_user_command_async(
                        user_input
                    )
                if handled_command:
                    if self.session is not session:
                        with self._session_manager.execution():
                            await self._model_selection.restore_async(self.session.model)
                            await self._recover_session_async()
                    continue
                if self._skill_hints_enabled:
                    try:
                        skill_mentions = tuple(
                            dict.fromkeys(
                                mention.value
                                for mention in parse_mentions(
                                    user_input,
                                    {"$": self._instructions_manager.skill_manager.names},
                                )
                            )
                        )
                        if not await self._instructions_manager.set_skill_catalog_priorities_async(
                            skill_mentions, self._agent
                        ):
                            self._interaction.warning(
                                "Referenced skill metadata could not fit in the instruction "
                                "budget; use activate_skill to load a skill."
                            )
                    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
                        problem = Problem.from_exception(
                            error,
                            code="skill_hint.preparation_failed",
                            title="Could not prepare skill hint",
                            operation="prepare_skill_hint",
                        )
                        log_problem(_LOGGER, problem, error)
                        self._interaction.report(problem)
                        if admitted_input is not None:
                            await self._session_manager.reject_input_async(admitted_input.input_id)
                        continue
                try:
                    context = await self._mention_manager.resolve_async(user_input)
                except (OSError, UnicodeError, ValueError, RuntimeError) as error:
                    problem = Problem.from_exception(
                        error,
                        code="mention.resolution_failed",
                        title="Could not resolve mention",
                        operation="resolve_mention",
                    )
                    log_problem(_LOGGER, problem, error)
                    self._interaction.report(problem)
                    if admitted_input is not None:
                        await self._session_manager.reject_input_async(admitted_input.input_id)
                    if self._skill_hints_enabled:
                        await self._clear_skill_catalog_priorities_async()
                    continue
                try:
                    with self._session_manager.next_message_span():
                        for reference in context:
                            included = reference.included_bytes
                            size = reference.size_bytes
                            if included < size:
                                self._interaction.info(
                                    f"Attached {reference.path}: {included} of {size} bytes"
                                    "; additional content stays local until read."
                                )
                            else:
                                self._interaction.info(
                                    f"Attached {reference.path}: {included} bytes."
                                )
                        recovering = self._session_manager.recovery_pending
                        await self._session_manager.add_user_message_async(
                            user_input,
                            context=context,
                            input_id=admitted_input.input_id
                            if admitted_input is not None
                            else None,
                        )
                        telemetry_activity("message.accepted", component="session_manager")
                        self._session_name_attempted_this_run = False
                        if recovering:
                            await self._recover_session_async(explicit=True)
                        else:
                            result = await self._agent_runner.run_async()
                            await self._complete_agent_run_async(result)
                finally:
                    if self._skill_hints_enabled:
                        await self._clear_skill_catalog_priorities_async()

        self._interaction.conversation_ended()

    @asynccontextmanager
    async def _session_name_task_scope_async(self) -> AsyncGenerator[None, None]:
        """Own automatically generated title tasks through conversation cleanup."""
        if self._session_name_task_group is not None:
            yield
            return
        scope_error: BaseException | None = None
        with self._session_manager.observe_responses(self._on_response_persisted):
            try:
                async with asyncio.TaskGroup() as task_group:
                    self._session_name_task_group = task_group
                    try:
                        yield
                    # Let the group settle children without wrapping Loop's original exception.
                    # pylint: disable=broad-exception-caught
                    except BaseException as error:  # noqa: BLE001
                        scope_error = error
                        for task in self._session_name_tasks.values():
                            task.cancel()
            finally:
                self._session_name_task_group = None
        if scope_error is not None:
            raise scope_error

    def _on_response_persisted(self, event: ResponsePersisted) -> None:
        """Schedule title generation for a response persisted to the active session."""
        if event.session_id == self._session_manager.session.id:
            self._schedule_session_name_generation()

    def _schedule_session_name_generation(self) -> None:
        """Start one title request for the first usable exchange in the current run."""
        task_group = self._session_name_task_group
        if task_group is None or self._session_name_attempted_this_run:
            return
        context = self._session_manager.session_name_context()
        if context is None:
            return
        pending = self._session_name_tasks.get(context.session_id)
        if pending is not None and not pending.done():
            self._session_name_attempted_this_run = True
            return
        self._session_name_attempted_this_run = True
        task = task_group.create_task(
            self._session_manager.generate_session_name_async(
                self._session_name_generator,
                context=context,
            ),
            name="session-name-generation",
        )
        self._session_name_tasks[context.session_id] = task
        task.add_done_callback(
            lambda completed, session_id=context.session_id: self._discard_session_name_task(
                session_id,
                completed,
            )
        )

    def _discard_session_name_task(
        self,
        session_id: str,
        task: asyncio.Task[None],
    ) -> None:
        """Remove a completed title task without dropping a newer task for that session."""
        if self._session_name_tasks.get(session_id) is task:
            del self._session_name_tasks[session_id]

    def _clear_skill_catalog_priorities(self) -> None:
        """Clear request-scoped skill priorities without masking the preceding outcome."""
        try:
            self._instructions_manager.set_skill_catalog_priorities(())
        except (OSError, UnicodeError, ValueError) as error:
            problem = Problem.from_exception(
                error,
                code="skill_hint.cleanup_failed",
                title="Could not clear skill hint",
                operation="clear_skill_hint",
            )
            log_problem(_LOGGER, problem, error)
            self._interaction.report(problem)

    async def _clear_skill_catalog_priorities_async(self) -> None:
        """Clear request-scoped skill priorities without masking the preceding outcome."""
        try:
            await self._instructions_manager.set_skill_catalog_priorities_async(())
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            problem = Problem.from_exception(
                error,
                code="skill_hint.cleanup_failed",
                title="Could not clear skill hint",
                operation="clear_skill_hint",
            )
            log_problem(_LOGGER, problem, error)
            self._interaction.report(problem)

    def _complete_agent_run(self, result: AgentRunResult) -> None:
        """Present metrics and finalize naming for one completed or recovered run."""
        if result.metrics is None:
            raise TypeError("Agent run did not produce completion metrics.")
        self._interaction.run_metrics(result.metrics)
        if result.stop_reason not in {
            AgentRunStopReason.COMPLETED,
            AgentRunStopReason.STEERED,
        }:
            return

        if self._run_control is not None:
            self._run_control.checkpoint()
        if self._session_manager.session.has_initial_name():
            self._session_manager.generate_session_name(self._session_name_generator)

    async def _complete_agent_run_async(self, result: AgentRunResult) -> None:
        """Present run metrics and schedule a fallback title request when needed."""
        if result.metrics is None:
            raise TypeError("Agent run did not produce completion metrics.")
        self._interaction.run_metrics(result.metrics)
        if result.stop_reason not in {
            AgentRunStopReason.COMPLETED,
            AgentRunStopReason.STEERED,
        }:
            return

        if self._run_control is not None:
            self._run_control.checkpoint()
        self._schedule_session_name_generation()

    def _recover_session(self, *, explicit: bool = False) -> bool:
        """Recover a previously interrupted run, if any."""
        recovery = (
            self._agent_runner.recover_session(confirmed=True)
            if explicit
            else self._agent_runner.recover_session()
        )
        if recovery.result is not None:
            self._complete_agent_run(recovery.result)
        return recovery.pending or self._session_manager.recovery_pending

    async def _recover_session_async(self, *, explicit: bool = False) -> bool:
        """Recover an interrupted run through its existing async recovery procedure."""
        self._session_name_attempted_this_run = False
        recovery = (
            await self._agent_runner.recover_session_async(confirmed=True)
            if explicit
            else await self._agent_runner.recover_session_async()
        )
        if recovery.result is not None:
            await self._complete_agent_run_async(recovery.result)
        return recovery.pending or self._session_manager.recovery_pending
