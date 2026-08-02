"""Run an interactive conversation with an LLM backend."""

from collections.abc import Iterable
from pathlib import Path

from .backend import Backend
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
        context (LoopContext | None): Conversation state to use. Supply the same context to share
            state between loops.
        stream (bool): Whether the backend should produce response events incrementally.
        debug (bool): Whether to print raw response events.

    """

    _backend: Backend
    _instructions: str | None
    _context: LoopContext
    _skill_manager: SkillManager
    _interaction: Interaction
    _working_directory: Path
    _debug: bool
    _stream: bool
    _model: str | None

    def __init__(
        self,
        backend: Backend,
        *,
        model: str | None = None,
        skill_manager: SkillManager | None = None,
        interaction: Interaction | None = None,
        working_directory: Path | str | None = None,
        context: LoopContext | None = None,
        stream: bool = False,
        debug: bool = False,
    ) -> None:
        self._interaction = interaction or backend.tool_registry.interaction or ConsoleInteraction()
        self._backend = backend
        self._context = context or LoopContext()
        self._working_directory = Path(working_directory or Path.cwd()).resolve()
        self._skill_manager = skill_manager or SkillManager.discover(self._working_directory)
        self._instructions = build_instructions(
            load_agents_instructions(self._working_directory),
            self._skill_manager.catalog(),
        )
        self._stream = stream
        self._debug = debug
        self._model = model

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
        return self._context.messages

    @property
    def context(self) -> LoopContext:
        """Return the shared conversation context.

        Returns:
            LoopContext: The mutable conversation state used by the loop.
        """
        return self._context

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
        while True:
            user_input = self._interaction.input()
            if user_input is False:
                break
            self._context.add_message(Message(role="user", content=user_input))

            while True:
                response = self.output(self.query())
                self._context.add_messages(response.items)

                if not self.handle_tool_calls(response):
                    break

            self._interaction.token_usage(
                self._context.model or self._model or self._backend.default_model,
                self._context.tokens,
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
            self._context.add_message(
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
        return self._backend.get_response(
            input=self._context.messages,
            instructions=self._instructions,
            stream=self._stream,
            model=selected_model,
        )

    def _update_context(self, usage: Usage, model: str | None) -> None:
        """Track the context produced by the latest model response."""
        if usage.total_tokens is not None:
            self._context.tokens = usage.total_tokens
        if isinstance(model, str):
            self._context.model = model

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

        self._update_context(usage, model)
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
