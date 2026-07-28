"""Main entry point for the loop package."""

from loop import ConsoleInteraction, StreamingLoop
from loop.types import ShutdownRequested
from loop.utils.signals import register_shutdown_signals


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    register_shutdown_signals()
    interaction = ConsoleInteraction()

    try:
        interaction.info("Hello from loop!")
        loop = StreamingLoop(interaction=interaction)
        loop.run()
    except EOFError, KeyboardInterrupt, ShutdownRequested:
        interaction.info("\nStopping loop. Goodbye!")


if __name__ == "__main__":
    main()
