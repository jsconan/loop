"""Main entry point for the loop package."""

from dotenv import find_dotenv, load_dotenv

from loop import ConsoleInteraction, ShutdownRequested, StreamingLoop, register_shutdown_signals


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    load_dotenv(find_dotenv(usecwd=True))
    register_shutdown_signals()
    interaction = ConsoleInteraction()

    try:
        interaction.info("Hello from loop!")
        loop = StreamingLoop(interaction=interaction)
        loop.run()
    except EOFError, KeyboardInterrupt, ShutdownRequested:
        interaction.info("\nStopping loop. Goodbye!")
    except Exception as e:  # pylint: disable=broad-except
        interaction.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
