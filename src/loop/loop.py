"""Run an interactive conversation with an LLM backend."""

from dataclasses import dataclass, field
from pprint import pprint

from openai import BaseModel
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from .client import Client
from .tooling import ToolRegistry


@dataclass
class Response:
    """Collected answer and reasoning text from an LLM response.

    Attributes:
        answer: The final answer text.
        reasoning: The model's reasoning text.
        tool_calls: Function tool calls requested by the model.
        output_items: Raw output items returned by the model.
    """

    answer: str
    reasoning: str
    tool_calls: list[ResponseFunctionToolCall] = field(default_factory=list)
    output_items: list[BaseModel] = field(default_factory=list)


class BaseLoop:
    """Run an interactive conversation using non-streaming responses.

    Args:
        client: Client used to request model responses. When omitted, a client is created.
        debug: Whether to print raw response events.
        tool_registry: Registry used by the automatically created client.

    Raises:
        ValueError: If both ``client`` and ``tool_registry`` are provided.
    """

    _client: Client
    _messages: list[dict]
    _debug: bool

    def __init__(
        self,
        client: Client | None = None,
        debug: bool = False,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        if client is not None and tool_registry is not None:
            raise ValueError("Pass either client or tool_registry, not both.")
        self._client = client or Client(tool_registry=tool_registry)
        self._messages = []
        self._debug = debug

    def run(self):
        """Run the conversation until the user requests to exit."""
        while True:
            user_input = self.input()
            if user_input is False:
                break
            self._messages.append({"role": "user", "content": user_input})

            while True:
                response = self.output(self.query())
                self.record_output(response)

                if not self.handle_tool_calls(response):
                    break

        self.end()

    def record_output(self, response: Response) -> None:
        """Append completed response output items to the conversation history.

        Args:
            response: The LLM response containing output items.
        """
        self._messages.extend(item.model_dump(exclude_none=True) for item in response.output_items)

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
            print(f"\n[TOOL CALL]: {tool_call.name}({tool_call.arguments})")
            self._messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": self._client.tool_registry.call(
                        tool_call.name,
                        tool_call.arguments,
                    ),
                }
            )
        return True

    def query(self) -> None:
        """Request a response for the current conversation history.

        Returns:
            The response returned by the configured backend.
        """
        return self._client.get_response(input=self._messages)

    def input(self) -> str | False:
        """Prompt for a non-empty user message or an exit command.

        Returns:
            The entered message, or ``False`` when the user requests to exit.
        """
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                print("Please enter a message!")
                continue

            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                return False
            return user_input

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

        for message in response.output:
            if self._debug:
                print(f"\n[DEBUG EVENT]: {type(message)}")
                pprint(message)

            if isinstance(message, ResponseReasoningItem):
                thinking_text += message.content[0].text if message.content else ""
                print(f"Reasoning: {message.content[0].text}")
                continue

            if isinstance(message, ResponseOutputMessage):
                content = message.content[0].text if message.content else ""
                answer_text += content
                print(f"Message: {content}")
                continue

            if isinstance(message, ResponseFunctionToolCall):
                tool_calls.append(message)
                continue

        print("")

        return Response(
            answer=answer_text.strip(),
            reasoning=thinking_text.strip(),
            tool_calls=tool_calls,
            output_items=list(response.output),
        )

    def end(self) -> None:
        """Display the conversation termination message."""
        print("\nConversation ended.")


class StreamingLoop(BaseLoop):
    """Run an interactive conversation while streaming response events."""

    def query(self) -> None:
        """Request a streaming response for the current conversation history.

        Returns:
            An iterable streaming response returned by the configured backend.
        """
        return self._client.get_response(input=self._messages, stream=True)

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

        print("\nThinking...")

        for event in response:
            if self._debug:
                print(f"\n[DEBUG EVENT]: {type(event)}")
                pprint(event)

            if isinstance(event, ResponseReasoningTextDeltaEvent):
                if not is_thinking:
                    print("\n[THOUGHT PROCESS]:")
                    is_thinking = True
                thinking_text += event.delta

            if isinstance(event, ResponseTextDeltaEvent):
                if not answer_started:
                    print("\n[ANSWER]:")
                    answer_started = True
                answer_text += event.delta

            if isinstance(event, ResponseOutputItemDoneEvent):
                output_items.append(event.item)
                if isinstance(event.item, ResponseFunctionToolCall):
                    tool_calls.append(event.item)
                    continue

            if isinstance(event, (ResponseTextDeltaEvent, ResponseReasoningTextDeltaEvent)):
                print(event.delta, end="", flush=True)
                continue

        print("")

        return Response(
            answer=answer_text.strip(),
            reasoning=thinking_text.strip(),
            tool_calls=tool_calls,
            output_items=output_items,
        )
