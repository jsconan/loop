# loop

`loop` is a small, experimental Python conversation loop for learning about
agentic loops with local or remote
[OpenAI-compatible Responses API](https://platform.openai.com/docs/api-reference/responses)
servers. It keeps conversation history, supports streaming and asynchronous
requests, and lets models call a few local tools.

The project defaults to a local server at `http://localhost:8000/v1` running
`nvidia/Qwen3.6-35B-A3B-NVFP4`.

## Features

- Interactive, multi-turn conversations
- Streaming response and reasoning output
- Synchronous and asynchronous clients
- Responses API function-call handling
- A decorator-based registry for synchronous and asynchronous Python tools
- Built-in tools for filesystem access, shell commands, and the current date and time
- Confirmation before a model writes a file or runs a shell command
- Hierarchical project instructions from `AGENTS.md` files
- Progressive discovery and activation of Agent Skills from `SKILL.md` files

## Requirements

- Python 3.14 or later
- An OpenAI-compatible server that implements the Responses API
- [`uv`](https://docs.astral.sh/uv/) (recommended), or another Python package manager

## Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/jsconan/things.git
cd loop
uv sync
```

Before running the examples, start your Responses API server. By default, `loop`
expects it at:

```text
http://localhost:8000/v1
```

If the server requires authentication, set its API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

For local servers that do not require authentication, no environment variable is
needed; the client supplies a placeholder key.

## Usage

### Interactive conversation

Run the streaming chat loop:

```bash
uv run python main.py
```

Enter a message at the `You:` prompt. To stop, enter `exit`, `quit`, `bye`, or
`q`.

At startup, the loop loads non-empty `AGENTS.md` files from the Git project root through the
current working directory. Instructions are combined from least to most specific and remain fixed
for the conversation. Other agent instruction filenames are ignored.

When the model requests a file write or shell command, the loop displays the
operation and asks for confirmation first. File and directory reads do not
require confirmation.

### Use the client directly

```python
from loop import Client

client = Client()
response = client.get_response(
    input="Explain why the sky is blue in two sentences.",
)

print(response.output_text)
```

### Asynchronous requests

```python
import asyncio

from loop import Client


async def main() -> None:
    client = Client()
    response = await client.get_response_async(
        input="Give me three names for a lunar rover.",
    )
    print(response.output_text)


asyncio.run(main())
```

### Context-aware tools

A decorated function can request an explicit `ToolContext`. The registry injects it during
dispatch, while the context parameter is omitted from the schema exposed to the model:

```python
from loop import ToolContext, ToolRegistry

registry = ToolRegistry()


@registry.tool
def describe_tool(context: ToolContext, value: str) -> str:
    """Return a value labeled with its registered tool name."""
    return f"{context.tool_name}: {value}"
```

The context provides invocation metadata and access to the injected user interaction service.
For loop-managed calls it also exposes the active `SkillManager`.

### Agent Skills

At startup, the loop discovers `SKILL.md` files under `.agents/skills` from the repository root
through the working directory, followed by `~/.agents/skills`. Only each skill's YAML `name` and
`description` are initially loaded and disclosed to the model. The complete Markdown instructions
are read and cached only when the model calls `manage_skills` with the `activate` action.

A minimal skill looks like this:

```markdown
---
name: review-changes
description: Review code changes for correctness, regressions, and missing tests.
---

Follow the repository's review workflow and report findings by severity.
```

## Configuration

Pass configuration directly to `Client` when using a different server or model:

```python
from loop import Client

client = Client(
    base_url="https://example.com/v1",
    api_key="your-api-key",
    default_model="your-model-id",
)
```

Constructor arguments take precedence over the corresponding environment
variables and built-in defaults:

| Setting  | Environment variable | Built-in default               |
| -------- | -------------------- | ------------------------------ |
| Base URL | `BASE_URL`           | `http://localhost:8000/v1`     |
| Model    | `DEFAULT_MODEL`      | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| API key  | `OPENAI_API_KEY`     | `local-api-key`                |

## Built-in tools

The default registry exposes these functions to the model:

| Tool                   | Behavior                                                                      |
| ---------------------- | ----------------------------------------------------------------------------- |
| `list_folder`          | Lists typed file/folder entries, optionally including nested entries           |
| `read_text_file`       | Reads a UTF-8 text file                                                       |
| `write_text_file`      | Writes a UTF-8 text file after interactive confirmation                       |
| `get_current_datetime` | Returns the current local date and time                                       |
| `run_command`          | Runs a shell command after interactive confirmation, with a 30-second timeout |
| `manage_skills`        | Lists skill metadata or activates one skill's instructions on demand          |

These tools operate with the permissions of the process running `loop`. Reads
are not sandboxed, and approved commands are passed to the system shell. Run the
project only in an environment where you are comfortable granting the model that
access.

## Development

Install the development dependencies:

```bash
uv sync
```

Run the test suite and linters:

```bash
uv run pytest
uv run ruff check .
uv run pylint src
```

The package uses a `src` layout. Its main components are:

```text
src/loop/client.py  OpenAI-compatible sync and async client
src/loop/loop.py    Interactive and streaming conversation loops
src/loop/tooling.py Tool registration, schemas, and dispatch
src/loop/tools/     Built-in tool implementations
```
