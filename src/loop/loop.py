"""Run an interactive conversation with an LLM backend."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from openai import BaseModel
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from .client import Client
from .interaction import ConsoleInteraction, Interaction
from .skills import SkillManager
from .tooling import ToolRegistry
from .tooling import tool_registry as default_tool_registry
from .utils.instructions import build_instructions, load_agents_instructions


@dataclass
class Response:
    """Collect answer and reasoning text from an LLM response.

    Attributes:
        answer: The final answer text.
        reasoning: The model's reasoning text.
        tool_calls: Function tool calls requested by the model.
        output_items: Raw output items returned by the model.
        usage: Total tokens in the context after the response, when reported by the backend.
        model: Model identifier reported by the backend.
    """

    answer: str
    reasoning: str
    tool_calls: list[ResponseFunctionToolCall] = field(default_factory=list)
    output_items: list[BaseModel] = field(default_factory=list)
    usage: int | None = None
    model: str | None = None


@dataclass
class LoopContext:
    """Store conversation history and the latest model context usage.

    Args:
        messages: Initial Responses API input items accumulated during the conversation.
        tokens: Initial total tokens in the context after the latest response.
        model: Initial model identifier reported by the latest response.
    """

    messages: list[dict] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None

    def add_message(self, message: BaseModel | dict) -> None:
        """Add one message to the conversation history.

        Args:
            message: Responses API input item to add. Models are dumped to dictionaries.

        Raises:
            ValueError: If the message is not a dictionary or a BaseModel.
        """
        self.messages.append(self._get_message(message))

    def add_messages(self, messages: Iterable[BaseModel | dict]) -> None:
        """Add messages to the conversation history.

        Args:
            messages: Responses API input items to add. Models are dumped to dictionaries.

        Raises:
            ValueError: If any message is not a dictionary or a BaseModel.
        """
        self.messages.extend([self._get_message(message) for message in messages])

    def _get_message(self, message: dict | BaseModel) -> dict:
        """Convert a message to a dictionary for storage in the conversation history."""
        if isinstance(message, BaseModel):
            message = message.model_dump(exclude_none=True)
        if not isinstance(message, dict):
            raise ValueError(f"Expected message to be a dict or BaseModel, got {type(message)}")
        return message


class BaseLoop:
    """Run an interactive conversation using non-streaming responses.

    Args:
        client: Client used to request model responses. When omitted, a client is created.
        tool_registry: Registry used by the automatically created client.
        skill_manager: Manager used to discover and progressively activate Agent Skills.
        interaction: Service used for all user input and output.
        working_directory: Directory used to discover applicable AGENTS.md files.
        context: Conversation state to use. Supply the same context to share state between loops.
        debug: Whether to print raw response events.

    Raises:
        ValueError: If both a client and tool registry are supplied.
    """

    _client: Client
    _instructions: str | None
    _context: LoopContext
    _skill_manager: SkillManager
    _interaction: Interaction
    _working_directory: Path
    _debug: bool

    def __init__(
        self,
        client: Client | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_manager: SkillManager | None = None,
        interaction: Interaction | None = None,
        working_directory: Path | str | None = None,
        context: LoopContext | None = None,
        debug: bool = False,
    ) -> None:
        if client is not None and tool_registry is not None:
            raise ValueError("Pass either client or tool_registry, not both.")

        if tool_registry is None:
            tool_registry = client.tool_registry if client is not None else default_tool_registry

        self._interaction = interaction or tool_registry.interaction or ConsoleInteraction()
        self._client = client or Client(tool_registry=tool_registry)
        self._context = context or LoopContext()
        self._working_directory = Path(working_directory or Path.cwd()).resolve()
        self._skill_manager = skill_manager or SkillManager.discover(self._working_directory)
        self._instructions = build_instructions(
            load_agents_instructions(self._working_directory),
            self._skill_manager.catalog(),
        )
        self._debug = debug
        default_model = self._client.default_model
        if self._context.model is None and isinstance(default_model, str):
            self._context.model = default_model

    @property
    def client(self) -> Client:
        """Return the client used to request model responses."""
        return self._client

    @property
    def instructions(self) -> str | None:
        """Return the AGENTS.md instructions loaded for this session."""
        return self._instructions

    @property
    def messages(self) -> list[dict]:
        """Return the current conversation history."""
        return self._context.messages

    @property
    def context(self) -> LoopContext:
        """Return the shared conversation context."""
        return self._context

    @property
    def skill_manager(self) -> SkillManager:
        """Return the skill manager active for this conversation."""
        return self._skill_manager

    @property
    def interaction(self) -> Interaction:
        """Return the service used for user input and output."""
        return self._interaction

    @property
    def working_directory(self) -> Path:
        """Return the directory used to discover project instructions."""
        return self._working_directory

    @property
    def debug(self) -> bool:
        """Return whether raw response event output is enabled."""
        return self._debug

    @debug.setter
    def debug(self, debug: bool) -> None:
        """Enable or disable raw response event output."""
        self._debug = debug

    def run(self):
        """Run the conversation until the user requests to exit."""
        while True:
            user_input = self._interaction.input()
            if user_input is False:
                break
            self._context.add_message({"role": "user", "content": user_input})

            while True:
                response = self.output(self.query())
                self.record_output(response)

                if not self.handle_tool_calls(response):
                    break

            self._interaction.token_usage(
                self._context.model,
                self._context.tokens,
                self._client.get_context_window(self._context.model),
            )

        self.end()

    def record_output(self, response: Response) -> None:
        """Append completed response output items to the conversation history.

        Args:
            response: The LLM response containing output items.
        """
        self._context.add_messages(response.output_items)

    def handle_tool_calls(self, response: Response) -> bool:
        """Handle tool calls made by the LLM during reasoning.

        Args:
            response: The LLM response containing tool call events.

        Returns:
            ``True`` if at least one tool call was handled; otherwise ``False``.
        """
        if not response.tool_calls:
            return False

        for tool_call in response.tool_calls:
            self._interaction.tool_call(tool_call.name, tool_call.arguments)
            self._context.add_message(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": self._client.tool_registry.call(
                        tool_call.name,
                        tool_call.arguments,
                        interaction=self._interaction,
                        skill_manager=self._skill_manager,
                    ),
                }
            )
        return True

    def query(self) -> None:
        """Request a response for the current conversation history.

        Returns:
            The response returned by the configured backend.
        """
        return self._client.get_response(
            input=self._context.messages,
            instructions=self._instructions,
        )

    def _update_context(self, usage, model: str | None) -> int | None:
        """Track the context produced by the latest model response."""
        total_tokens = getattr(usage, "total_tokens", None)
        if isinstance(total_tokens, int) and total_tokens >= 0:
            self._context.tokens = total_tokens
        if isinstance(model, str):
            self._context.model = model
        return total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else None

    def output(self, response) -> Response:
        """Display and collect reasoning and answer content from a response.

        Args:
            response: A completed response returned by the LLM backend.

        Returns:
            The collected answer and reasoning text.
        """
        thinking_text = ""
        answer_text = ""
        tool_calls = []
        raw_usage = getattr(response, "usage", None)
        model = getattr(response, "model", None)

        for message in response.output:
            if self._debug:
                self._interaction.debug(message)

            if isinstance(message, ResponseReasoningItem):
                content = message.content[0].text if message.content else ""
                thinking_text += content
                self._interaction.reasoning(content)
                continue

            if isinstance(message, ResponseOutputMessage):
                content = message.content[0].text if message.content else ""
                answer_text += content
                self._interaction.answer(content)
                continue

            if isinstance(message, ResponseFunctionToolCall):
                tool_calls.append(message)
                continue

        return Response(
            answer=answer_text.strip(),
            reasoning=thinking_text.strip(),
            tool_calls=tool_calls,
            output_items=list(response.output),
            usage=self._update_context(raw_usage, model),
            model=model if isinstance(model, str) else None,
        )

    def end(self) -> None:
        """Display the conversation termination message."""
        self._interaction.conversation_ended()


class StreamingLoop(BaseLoop):
    """Run an interactive conversation while streaming response events."""

    def query(self) -> None:
        """Request a streaming response for the current conversation history.

        Returns:
            An iterable streaming response returned by the configured backend.
        """
        return self._client.get_response(
            input=self._context.messages,
            instructions=self._instructions,
            stream=True,
        )

    def output(self, response) -> Response:
        """Display and collect events from a streaming response.

        Args:
            response: An iterable of streaming events returned by the LLM backend.

        Returns:
            The collected answer and reasoning text.
        """
        is_thinking = False
        answer_started = False
        thinking_text = ""
        answer_text = ""
        tool_calls = []
        output_items = []
        usage = None
        model = None

        for event in response:
            if self._debug:
                self._interaction.debug(event)

            if isinstance(event, ResponseReasoningTextDeltaEvent):
                self._interaction.reasoning_delta(event.delta, start=not is_thinking)
                is_thinking = True
                continue

            if isinstance(event, ResponseTextDeltaEvent):
                self._interaction.answer_delta(event.delta, start=not answer_started)
                answer_started = True
                continue

            if isinstance(event, ResponseOutputItemDoneEvent):
                output_items.append(event.item)
                if isinstance(event.item, ResponseFunctionToolCall):
                    tool_calls.append(event.item)
                    continue

            if isinstance(event, ResponseCompletedEvent):
                usage = event.response.usage
                model = event.response.model
                continue

            if isinstance(event, ResponseReasoningTextDoneEvent):
                thinking_text += event.text

            if isinstance(event, ResponseTextDoneEvent):
                answer_text += event.text

        if is_thinking or answer_started:
            self._interaction.response_finished()

        return Response(
            answer=answer_text.strip(),
            reasoning=thinking_text.strip(),
            tool_calls=tool_calls,
            output_items=output_items,
            usage=self._update_context(usage, model),
            model=model,
        )
