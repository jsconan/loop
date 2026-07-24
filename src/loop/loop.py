"""Run an interactive conversation with an LLM backend."""

from dataclasses import dataclass
from pprint import pprint

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from .client import Client


@dataclass
class Response:
    """Collected answer and reasoning text from an LLM response."""

    answer: str
    reasoning: str


class BaseLoop:
    """Run an interactive conversation using non-streaming responses."""

    _client: Client
    _messages: list[dict]
    _debug: bool

    def __init__(self, client: Client = None, debug: bool = False) -> None:
        self._client = client or Client()
        self._messages = []
        self._debug = debug

    def run(self):
        """Run the conversation until the user requests to exit."""
        while True:
            user_input = self.input()
            if user_input is False:
                break
            self._messages.append({"role": "user", "content": user_input})

            response = self.output(self.query())
            self._messages.append(
                {
                    "role": "assistant",
                    "content": response.answer.strip(),
                    "reasoning": response.reasoning.strip(),
                }
            )

        self.end()

    def query(self) -> None:
        """Request a response for the current conversation history."""
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

        print("")

        return Response(answer=answer_text.strip(), reasoning=thinking_text.strip())

    def end(self) -> None:
        """Display the conversation termination message."""
        print("\nConversation ended.")


class StreamingLoop(BaseLoop):
    """Run an interactive conversation while streaming response events."""

    def query(self) -> None:
        """Request a streaming response for the current conversation history."""
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

            if isinstance(event, (ResponseTextDeltaEvent, ResponseReasoningTextDeltaEvent)):
                print(event.delta, end="", flush=True)
                continue

        print("")

        return Response(answer=answer_text.strip(), reasoning=thinking_text.strip())
