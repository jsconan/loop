"""Minimal chat loop: prompt -> query -> output."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from loop import (
    CommandCompletionAdapter,
    CommandManager,
    CompletionManager,
    CompletionValue,
    ConsoleInteraction,
    MemorySessionStore,
    MentionManager,
    OpenAIBackend,
    ProjectPathMentionHandler,
    SessionManager,
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
        session_manager = SessionManager(
            interaction=interaction,
            session_store=MemorySessionStore(),
        )
        command_manager = CommandManager(interaction=interaction, session_manager=session_manager)

        mention_manager = MentionManager((ProjectPathMentionHandler(Path.cwd),))
        completion_manager = CompletionManager(
            (
                CommandCompletionAdapter(
                    lambda: command_manager.commands,
                    providers={
                        "sessions": lambda: (
                            CompletionValue(
                                session.id,
                                str(session.updated_at),
                                display=session.name,
                                sort_order=index,
                            )
                            for index, session in enumerate(session_manager.store.list())
                        ),
                    },
                ),
                *mention_manager.completion_adapters,
            )
        )

        context_window = os.getenv("CONTEXT_WINDOW")
        max_retries = os.getenv("OPENAI_MAX_RETRIES")
        backend = OpenAIBackend(
            base_url=os.getenv("BASE_URL", _BASE_URL),
            default_model=os.getenv("DEFAULT_MODEL", _DEFAULT_MODEL),
            api_key=os.getenv("OPENAI_API_KEY", _API_KEY),
            context_window=int(context_window) if context_window else None,
            max_retries=int(max_retries) if max_retries else 2,
            tool_registry=ToolRegistry(),
        )

        interaction.info("Hello from Chat!")

        while not command_manager.exit_requested:
            prompt = interaction.input(completer=completion_manager)
            if prompt is False:
                break
            if command_manager.handle_user_command(prompt):
                continue

            try:
                context = mention_manager.resolve(prompt)
            except (OSError, UnicodeError, ValueError) as error:
                interaction.error(str(error))
                continue
            session_manager.add_user_message(prompt, context=context)

            events = backend.get_response(input=session_manager.messages, stream=True)
            response = interaction.response(events)
            session_manager.add_response(response)

        interaction.conversation_ended()

    except EOFError, KeyboardInterrupt, ShutdownRequested:
        interaction.conversation_ended()
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-except
        interaction.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
