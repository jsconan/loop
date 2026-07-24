"""Main entry point for the loop package."""

from loop import StreamingLoop


def main():
    """Run an interactive conversation with an LLM backend."""
    print("Hello from loop!")
    loop = StreamingLoop()
    loop.run()


if __name__ == "__main__":
    main()
