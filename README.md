# loop

`loop` is a small, experimental Python conversation loop for learning about agentic loops with local or remote [OpenAI-compatible Responses API](https://platform.openai.com/docs/api-reference/responses) servers. It keeps conversation history, supports streaming and asynchronous requests, and lets models call a few local tools.

The project defaults to a local server at `http://localhost:8000/v1` running `nvidia/Qwen3.6-35B-A3B-NVFP4`.

## Features

- Interactive, multi-turn conversations
- Responsive terminal input with queued follow-ups, active-run steering and interruption, and separate approval replies
- Agent-scoped instructions, tools, permissions, and bounded model/tool execution
- Streaming response and reasoning output
- Synchronous and asynchronous backend access
- Responses API function-call handling
- A decorator-based registry for synchronous and asynchronous Python tools
- Built-in tools for filesystem access, shell commands, and the current date and time
- Confirmation for file writes, destructive workspace commands, Git changes, and commands that expand the default workspace sandbox
- File edits checked against the source snapshots supplied to the model, with reread warnings when files change during generation
- Hierarchical project instructions from `AGENTS.md` files
- Progressive discovery and activation of Agent Skills from `SKILL.md` files

## Requirements

- Python 3.12 or later
- An OpenAI-compatible server that implements the Responses API
- [`uv`](https://docs.astral.sh/uv/) (recommended), or another Python package manager

## Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/jsconan/loop.git
cd loop
uv sync
```

To install optional ripgrep for shell commands, include the `tools` extra:

```bash
uv sync --extra tools
```

To install directly as a `uv` tool, run:

```bash
uv tool install git+https://github.com/jsconan/loop.git
```

The installation provides the `loop` command.

Before running the examples, start your Responses API server. By default, `loop` expects it at:

```text
http://localhost:8000/v1
```

If the server requires authentication, set its API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

For local servers that do not require authentication, no environment variable is needed; the OpenAI backend supplies a placeholder key.

## Run Loop

Start your Responses API server, then run:

```bash
uv run loop
```

If installed as a `uv` tool, run `loop` from any directory. Use `/stop` to settle an active run while keeping the conversation available, and send a message to resume with new instructions or use `/continue` to resume it. Use `/exit` or `/quit` to leave the application.

## Guides

- [Use the interactive app and Python backend](docs/using-loop.md)
- [Configure Loop](docs/configuration.md)
- [Review tool permissions](docs/permissions.md)
- [Use built-in tools and the command sandbox](docs/tools.md)
- [Use value holders from Python](docs/library.md)
- [Develop and test Loop](docs/development.md)
