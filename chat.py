"""Minimal chat loop: prompt -> query -> output."""

import os

from dotenv import find_dotenv, load_dotenv

from loop import (
    CommandManager,
    ConsoleInteraction,
    Message,
    OpenAIBackend,
    Session,
    ShutdownRequested,
    ToolRegistry,
    register_shutdown_signals,
)

_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
_API_KEY = "local-api-key"


def main() -> None:
    """Run a simple prompt/query/output loop until the user exits."""
    load_dotenv(find_dotenv(usecwd=True))
    register_shutdown_signals()

    try:
        interaction = ConsoleInteraction()
        command_manager = CommandManager(interaction=interaction)

        context_window = os.getenv("CONTEXT_WINDOW")
        backend = OpenAIBackend(
            base_url=os.getenv("BASE_URL", _BASE_URL),
            default_model=os.getenv("DEFAULT_MODEL", _DEFAULT_MODEL),
            api_key=os.getenv("OPENAI_API_KEY", _API_KEY),
            context_window=int(context_window) if context_window is not None else None,
            tool_registry=ToolRegistry(),
        )

        interaction.info("Hello from Chat!")

        session = Session()
        while not command_manager.exit_requested:
            prompt = interaction.input(commands=command_manager.commands)
            if prompt is False:
                break
            if command_manager.handle_user_command(prompt):
                continue

            session.add_message(Message(role="user", content=prompt))

            events = backend.get_response(input=session.messages, stream=True)
            response = interaction.response(events)
            session.add_message(response)

        interaction.conversation_ended()

    except EOFError, KeyboardInterrupt, ShutdownRequested:
        interaction.conversation_ended()
    except Exception as e:  # pylint: disable=broad-except
        interaction.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
