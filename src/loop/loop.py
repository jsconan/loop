"""Run an interactive conversation with an LLM backend."""

from pathlib import Path
from time import sleep

from . import constants
from .backend import Backend, BackendError, BackendNotFoundError
from .commands import CommandManager, CommandRegistration
from .completion import (
    CommandCompletionAdapter,
    CompletionManager,
    CompletionValue,
)
from .interaction import Interaction
from .mentions import MentionManager, ProjectPathMentionHandler, SkillMentionHandler
from .models import ConversationItem, Response
from .permissions import PermissionCommands, PermissionManager
from .session import (
    BackendSessionNameGenerator,
    Session,
    SessionCommands,
    SessionManager,
    SessionNameGenerator,
)
from .skills import InstructionsManager, SkillCommands
from .tooling import ToolCommands
from .utils import find_project_root


class Loop:
    """Run an interactive conversation using normalized response events.

    Args:
        backend (Backend): Backend used to request model responses.
        model (str | None): Model selected for requests, or ``None`` to use the backend default.
        instructions_manager (InstructionsManager | None): Manager used to compose the complete
            backend instructions. Defaults to discovering project instructions and Agent Skills
            for ``working_directory``.
        interaction (Interaction | None): Service used for all user input and output.
        permission_manager (PermissionManager | None): Manager used to authorize model tool calls.
            Defaults to loading local policy from the project ``.loop`` folder.
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
        prompt_on_recoverable_error (bool): Whether to offer an interactive retry after automatic
            retries exhaust a recoverable backend failure.

    Raises:
        ValueError: If the session is invalid.

    """

    _backend: Backend
    _instructions_manager: InstructionsManager
    _session_manager: SessionManager
    _interaction: Interaction
    _working_directory: Path
    _debug: bool
    _stream: bool
    _model: str | None
    _command_manager: CommandManager
    _permission_manager: PermissionManager
    _completion_manager: CompletionManager
    _session_name_generator: SessionNameGenerator
    _mention_manager: MentionManager
    _prompt_on_recoverable_error: bool

    def __init__(
        self,
        backend: Backend,
        *,
        model: str | None = None,
        instructions_manager: InstructionsManager | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
        mention_manager: MentionManager | None = None,
        working_directory: Path | str | None = None,
        agents_filenames: tuple[str, ...] = (constants.DEFAULT_AGENTS_FILENAME,),
        session: Session | str | None = None,
        session_manager: SessionManager | None = None,
        session_name_generator: SessionNameGenerator | None = None,
        stream: bool = False,
        debug: bool = False,
        prompt_on_recoverable_error: bool = True,
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
        self._instructions_manager = instructions_manager or InstructionsManager.discover(
            self._working_directory,
            agents_filenames=agents_filenames,
        )
        if instructions_manager is None:
            self._instructions_manager.reactivate_skills(
                self._session_manager.session.active_skills
            )
        self._stream = stream
        self._debug = debug
        self._model = model
        self._prompt_on_recoverable_error = prompt_on_recoverable_error
        selected_model = self._model or self._backend.default_model
        if selected_model:
            self._report_context_window(selected_model)
        self._permission_manager = permission_manager or PermissionManager(
            find_project_root(self._working_directory) or self._working_directory,
            interaction=self._interaction,
        )
        self._command_manager = CommandManager(interaction=self._interaction)
        self._command_manager.register_providers(
            (
                SessionCommands(self._session_manager),
                PermissionCommands(self._permission_manager),
                SkillCommands(self._instructions_manager),
                ToolCommands(self._backend.tool_registry, self._instructions_manager),
            )
        )
        self._command_manager.register(CommandRegistration(self.compact, name="compact"))
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
                    providers={
                        "tools": lambda: (
                            CompletionValue(tool.name, tool.description)
                            for tool in self._backend.tool_registry.tools
                        ),
                        "skills": lambda: (
                            CompletionValue(skill.name, skill.description)
                            for skill in self._instructions_manager.skill_manager.skills
                        ),
                        "sessions": lambda: (
                            CompletionValue(
                                session.id,
                                str(session.updated_at),
                                display=session.name,
                                sort_order=index,
                            )
                            for index, session in enumerate(self._session_manager.store.list())
                        ),
                    },
                    schema_providers={
                        "tool_arguments": lambda tokens: next(
                            (
                                tool.arguments_model
                                for tool in self._backend.tool_registry.tools
                                if tokens and tool.name == tokens[0]
                            ),
                            None,
                        )
                    },
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
    def model(self) -> str | None:
        """Return the model explicitly selected for requests.

        Returns:
            str | None: The selected model, or ``None`` when the backend default is used.
        """
        return self._model

    def run(self):
        """Run the conversation until the user requests to exit."""
        while not self._command_manager.exit_requested:
            user_input = self._interaction.prompt(completer=self._completion_manager)
            if user_input is False:
                break
            session = self.session
            if self._command_manager.handle_user_command(user_input):
                if self.session is not session and self.session.model:
                    self._model = self.session.model
                continue
            try:
                context = self._mention_manager.resolve(user_input)
            except (OSError, UnicodeError, ValueError) as error:
                self._interaction.error(str(error))
                continue
            self._session_manager.add_user_message(user_input, context=context)

            while True:
                response = self._query_with_recovery()
                if response is None:
                    break
                self._session_manager.add_response(response)

                if not self.handle_tool_calls(response):
                    break

            if response is None:
                continue

            if self._session_manager.session.has_initial_name():
                try:
                    self._interaction.info("Generating a session name...")
                    self._session_manager.generate_session_name(self._session_name_generator)
                    self._interaction.info(f"Session name: {self._session_manager.session.name}")
                except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    self._interaction.warning(f"Could not generate the session name: {error}")

            self._interaction.token_usage(
                self._session_manager.model,
                self._session_manager.tokens,
                self._session_manager.context_window,
            )

        self.end()

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
        """List available models and let the user replace a missing selection.

        Re-prompts when the user selects the already-failed model; only returns
        when the user picks a different model or cancels.
        """
        try:
            models = self._backend.get_models()
        except BackendError as error:
            self._interaction.error(f"Could not list available models: {error}")
            return False
        if not models:
            self._interaction.warning("The backend reported no available models.")
            return False
        failing_model = self._model
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
        self._model = selection
        self._session_manager.model = selection
        self._interaction.info(f"Using model: {selection}")
        return True

    def handle_tool_calls(self, response: Response) -> bool:
        """Handle tool calls made by the LLM during reasoning.

        Args:
            response (Response): The LLM response containing tool call events.

        Returns:
            bool: ``True`` if at least one tool call was handled; otherwise ``False``.
        """
        if not response.tool_calls:
            return False

        for tool_call in response.tool_calls:
            self._interaction.tool_call(tool_call.name, tool_call.arguments)
            tool_result = self._backend.tool_registry.call(
                tool_call.name,
                tool_call.arguments,
                interaction=self._interaction,
                instructions_manager=self._instructions_manager,
                permission_manager=self._permission_manager,
            )
            self._session_manager.add_tool_call(
                call_id=tool_call.call_id,
                output=tool_result,
                working_directory=str(
                    self._instructions_manager.working_directory or self._working_directory
                ),
                active_skills=self._instructions_manager.active_skill_identities,
            )
        return True

    def query(self) -> Response:
        """Request a response from the backend using the current session context and instructions.

        Returns:
            Response: The response from the backend, wrapped in a ``Response`` object.

        Raises:
            ValueError: If neither the loop nor the backend selects a model.
        """
        selected_model = self._get_selected_model()
        self._instructions_manager.prepare()
        self._session_manager.update_instruction_state(
            working_directory=str(
                self._instructions_manager.working_directory or self._working_directory
            ),
            active_skills=self._instructions_manager.active_skill_identities,
        )
        self._report_context_window(selected_model)
        if self._session_manager.compaction_needed():
            self._compact(selected_model)
        events = self._backend.get_response(
            input=self._session_manager.model_context,
            instructions=self._instructions_manager.instructions,
            stream=self._stream,
            model=selected_model,
        )
        return self._interaction.response(events, debug=self._debug)

    def compact(self) -> bool:
        """Manually compact the active session using current model and instructions.

        Returns:
            bool: ``True`` when a new compaction checkpoint was persisted, otherwise ``False``.

        Raises:
            ValueError: If neither the loop nor backend selects a model.
        """
        if not self._session_manager.can_compact():
            self._interaction.warning("There is no new session context to compact.")
            return False
        self._instructions_manager.prepare()
        model = self._get_selected_model()
        self._report_context_window(model)
        return self._compact(model)

    def end(self) -> None:
        """Display the conversation termination message."""
        self._interaction.conversation_ended()

    def _get_selected_model(self) -> str:
        """Return the model selected for requests, or the backend default."""
        selected_model = self._model or self._backend.default_model
        if not selected_model:
            raise ValueError("No model was selected and the backend has no default model.")
        return selected_model

    def _report_context_window(self, model: str) -> None:
        """Report the context window for the selected model."""
        self._session_manager.model = model
        reported_context_window = self._backend.get_context_window(model)
        self._session_manager.context_window = (
            reported_context_window
            if isinstance(reported_context_window, int)
            and not isinstance(reported_context_window, bool)
            else None
        )

    def _compact(self, model: str) -> bool:
        """Request, report, and persist one replacement-context checkpoint."""
        self._interaction.info("Compacting session context...")
        try:
            result = self._backend.compact(
                self._session_manager.model_context,
                instructions=self._instructions_manager.instructions,
                model=model,
            )
        except NotImplementedError:
            self._interaction.warning("The selected backend does not support context compaction.")
            return False
        if result is None or not result.items:
            self._interaction.warning("The selected backend did not produce compacted context.")
            return False
        working_directory = str(
            self._instructions_manager.working_directory or self._working_directory
        )
        previous_tokens = self._session_manager.tokens
        self._session_manager.add_compaction(
            result,
            model=model,
            instructions=self._instructions_manager.instructions,
            working_directory=working_directory,
            active_skills=self._instructions_manager.active_skill_identities,
        )
        current_tokens = self._session_manager.tokens
        if current_tokens != previous_tokens:
            self._interaction.info(
                f"Compacted session context from {previous_tokens:,} to {current_tokens:,} tokens."
            )
        else:
            self._interaction.info("Compacted session context.")
        return True
