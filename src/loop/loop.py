"""Run an interactive conversation with an LLM backend."""

from collections.abc import Iterable
from pathlib import Path

from .backend import Backend
from .commands import CommandManager
from .context import LoopContext
from .interaction import ConsoleInteraction, Interaction
from .models import (
    AnswerCompleted,
    AnswerDelta,
    ConversationItem,
    Message,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    ResponseEvent,
    ToolCallCompleted,
    ToolResult,
    Usage,
)
from .session import MemorySessionStore, SessionStore
from .skills import SkillManager, build_instructions, load_agents_instructions


class Loop:
    """Run an interactive conversation using normalized response events.

    Args:
        backend (Backend): Backend used to request model responses.
        model (str | None): Model selected for requests, or ``None`` to use the backend default.
        skill_manager (SkillManager | None): Manager used to discover and progressively activate
            Agent Skills.
        interaction (Interaction | None): Service used for all user input and output.
        working_directory (Path | str | None): Directory used to discover applicable AGENTS.md
            files.
        session (LoopContext | str | None): Conversation context or persisted session identifier
            to load. Defaults to a fresh context; an injected store persists it after its first
            query.
        session_store (SessionStore | None): Session store, or ``None`` to use an instance-local
            memory store.
        stream (bool): Whether the backend should produce response events incrementally.
        debug (bool): Whether to print raw response events.

    """

    _backend: Backend
    _instructions: str | None
    _session: LoopContext
    _session_id: str | None
    _session_store: SessionStore
    _skill_manager: SkillManager
    _interaction: Interaction
    _working_directory: Path
    _debug: bool
    _stream: bool
    _model: str | None
    _command_manager: CommandManager

    def __init__(
        self,
        backend: Backend,
        *,
        model: str | None = None,
        skill_manager: SkillManager | None = None,
        interaction: Interaction | None = None,
        working_directory: Path | str | None = None,
        session: LoopContext | str | None = None,
        session_store: SessionStore | None = None,
        stream: bool = False,
        debug: bool = False,
    ) -> None:
        self._interaction = interaction or backend.tool_registry.interaction or ConsoleInteraction()
        self._backend = backend
        self._session_id = None
        self._session_store = session_store or MemorySessionStore()
        if isinstance(session, str):
            self._session_id = session
            session = self._session_store.load(session)
        self._session = session or LoopContext()
        self._working_directory = Path(working_directory or Path.cwd()).resolve()
        self._skill_manager = skill_manager or SkillManager.discover(self._working_directory)
        self._instructions = build_instructions(
            load_agents_instructions(self._working_directory),
            self._skill_manager.catalog(),
        )
        self._stream = stream
        self._debug = debug
        self._model = model
        self._command_manager = CommandManager(interaction=self._interaction)

    @property
    def backend(self) -> Backend:
        """Return the backend used to request model responses.

        Returns:
            Backend: The configured LLM backend.
        """
        return self._backend

    @property
    def instructions(self) -> str | None:
        """Return the AGENTS.md instructions loaded for this session.

        Returns:
            str | None: The combined project instructions, or ``None`` when none were found.
        """
        return self._instructions

    @property
    def messages(self) -> list[ConversationItem]:
        """Return the current conversation history.

        Returns:
            list[ConversationItem]: Items accumulated during the conversation.
        """
        return self._session.messages

    @property
    def session(self) -> LoopContext:
        """Return the active conversation session.

        Returns:
            LoopContext: The mutable conversation context used by the loop.
        """
        return self._session

    @property
    def skill_manager(self) -> SkillManager:
        """Return the skill manager active for this conversation.

        Returns:
            SkillManager: The active skill manager.
        """
        return self._skill_manager

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

    @property
    def debug(self) -> bool:
        """Return whether raw response event output is enabled.

        Returns:
            bool: Whether debug output is enabled.
        """
        return self._debug

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

    @debug.setter
    def debug(self, debug: bool) -> None:
        """Enable or disable raw response event output.

        Args:
            debug (bool): Whether to enable debug output.
        """
        self._debug = debug

    def run(self):
        """Run the conversation until the user requests to exit."""
        while not self._command_manager.exit_requested:
            user_input = self._interaction.input(self._command_manager.commands)
            if user_input is False:
                break
            if self._command_manager.handle_user_command(user_input):
                continue
            self._add_message(Message(role="user", content=user_input))

            while True:
                response = self.output(self.query())
                self._add_message(response)

                if not self.handle_tool_calls(response):
                    break

            self._interaction.token_usage(
                self._session.model,
                self._session.tokens,
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
            self._add_message(
                ToolResult(
                    call_id=tool_call.call_id,
                    output=self._backend.tool_registry.call(
                        tool_call.name,
                        tool_call.arguments,
                        interaction=self._interaction,
                        skill_manager=self._skill_manager,
                    ),
                )
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
        self._session.model = selected_model
        return self._backend.get_response(
            input=self._session.messages,
            instructions=self._instructions,
            stream=self._stream,
            model=selected_model,
        )

    def _add_message(self, message: ConversationItem | Response) -> None:
        """Add conversation items and persist the resulting complete session."""
        if isinstance(message, Response):
            self._session.add_messages(message.items)
            if message.usage.total_tokens is not None:
                self._session.tokens = message.usage.total_tokens
            if isinstance(message.model, str):
                self._session.model = message.model
        else:
            self._session.add_message(message)
        self._session_id = self._session_store.save(self._session_id, self._session)

    def output(self, events: Iterable[ResponseEvent]) -> Response:
        """Display and collect normalized response events.

        Args:
            events (Iterable[ResponseEvent]): Response events to display and collect.

        Returns:
            Response: The collected answer and reasoning text.
        """
        reasoning = ""
        answer = ""
        tool_calls = []
        items = ()
        usage = Usage()
        model = None
        reasoning_started = False
        answer_started = False

        with self._interaction.response():
            for event in events:
                if self._debug:
                    self._interaction.debug(event)

                if isinstance(event, ReasoningDelta):
                    self._interaction.reasoning_delta(event.text, start=not reasoning_started)
                    reasoning_started = True
                    continue

                if isinstance(event, AnswerDelta):
                    self._interaction.answer_delta(event.text, start=not answer_started)
                    answer_started = True
                    continue

                if isinstance(event, ReasoningCompleted):
                    reasoning = event.text
                    self._interaction.reasoning(event.text)
                    continue

                if isinstance(event, AnswerCompleted):
                    answer = event.text
                    self._interaction.answer(event.text)
                    continue

                if isinstance(event, ToolCallCompleted):
                    tool_calls.append(event.call)
                    continue

                if isinstance(event, ResponseCompleted):
                    items = event.items
                    usage = event.usage
                    model = event.model
                    answer = event.answer
                    reasoning = event.reasoning

        return Response(
            answer=answer,
            reasoning=reasoning,
            tool_calls=tuple(tool_calls),
            items=items,
            usage=usage,
            model=model,
        )

    def end(self) -> None:
        """Display the conversation termination message."""
        self._interaction.conversation_ended()
