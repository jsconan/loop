"""Main entry point for the loop package."""

from loop import StreamingLoop
from loop.types import ShutdownRequested
from loop.utils.signals import register_shutdown_signals


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    register_shutdown_signals()

    try:
        print("Hello from loop!")
        loop = StreamingLoop()
        loop.run()
    except (EOFError, KeyboardInterrupt, ShutdownRequested):
        print("\nStopping loop. Goodbye!")


if __name__ == "__main__":
    main()
