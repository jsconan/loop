"""Run an interactive conversation with an LLM backend."""

from collections.abc import Iterable
from pathlib import Path

from . import constants
from .backend import Backend
from .commands import CommandManager
from .completion import (
    CommandCompletionAdapter,
    CompletionManager,
    CompletionValue,
)
from .interaction import Interaction
from .mentions import MentionManager, ProjectPathMentionHandler, SkillMentionHandler
from .models import (
    ConversationItem,
    Response,
    ResponseEvent,
)
from .permissions import PermissionManager
from .session import Session, SessionManager
from .skills import InstructionsManager
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
        stream (bool): Whether the backend should produce response events incrementally.
        debug (bool): Whether to print raw response events.

    Raises:
        ValueError: The session is invalid.

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
    _mention_manager: MentionManager

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
        stream: bool = False,
        debug: bool = False,
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
        self._permission_manager = permission_manager or PermissionManager(
            find_project_root(self._working_directory) or self._working_directory,
            interaction=self._interaction,
        )
        self._command_manager = CommandManager(
            interaction=self._interaction,
            permission_manager=self._permission_manager,
            skill_manager=self._instructions_manager.skill_manager,
            tool_registry=self._backend.tool_registry,
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
                    providers={
                        "tools": lambda: (
                            CompletionValue(tool.name, tool.description)
                            for tool in self._backend.tool_registry.tools
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
            user_input = self._interaction.input(completer=self._completion_manager)
            if user_input is False:
                break
            if self._command_manager.handle_user_command(user_input):
                continue
            try:
                context = self._mention_manager.resolve(user_input)
            except (OSError, UnicodeError, ValueError) as error:
                self._interaction.error(str(error))
                continue
            self._session_manager.add_user_message(user_input, context=context)

            while True:
                events = self.query()
                response = self._interaction.output(events, debug=self._debug)
                self._session_manager.add_response(response)

                if not self.handle_tool_calls(response):
                    break

            self._interaction.token_usage(
                self._session_manager.model,
                self._session_manager.tokens,
                self._backend.get_context_window(self._model),
            )

        self.end()

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

    def query(self) -> Iterable[ResponseEvent]:
        """Request normalized events for the current conversation history.

        Returns:
            Iterable[ResponseEvent]: Events returned by the configured backend.

        Raises:
            ValueError: If neither the loop nor the backend selects a model.
        """
        selected_model = self._model or self._backend.default_model
        if not selected_model:
            raise ValueError("No model was selected and the backend has no default model.")
        self._instructions_manager.prepare()
        self._session_manager.update_instruction_state(
            working_directory=str(
                self._instructions_manager.working_directory or self._working_directory
            ),
            active_skills=self._instructions_manager.active_skill_identities,
        )
        self._session_manager.model = selected_model
        return self._backend.get_response(
            input=self._session_manager.messages,
            instructions=self._instructions_manager.instructions,
            stream=self._stream,
            model=selected_model,
        )

    def end(self) -> None:
        """Display the conversation termination message."""
        self._interaction.conversation_ended()
