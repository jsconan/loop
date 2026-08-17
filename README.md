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
- Synchronous and asynchronous backend access
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
git clone https://github.com/jsconan/loop.git
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
needed; the OpenAI backend supplies a placeholder key.

## Usage

### Interactive conversation

Run the chat loop, which enables streaming in `main.py`:

```bash
uv run python main.py
```

Enter a message at the `You:` prompt. To stop, enter `exit`, `quit`, `bye`, or
`q`.

Terminal completion is available while typing. Use `/` for commands and their known values, `@`
for visible files and directories relative to the working directory, and `$` for discovered skill
names. Matching is case-insensitive and accepts fragments at the beginning, middle, or end. File
completion respects `.gitignore` and `.agentignore`; `/permissions` completes operations, modes,
decisions, registered tools, and capabilities.

Command arguments use shell-like quoting and may be supplied by position, by name, or both, for
example `/command first-value count=3 enabled=true`. Each value is decoded independently and then
validated against the command function's annotated parameter model. Command completion offers
remaining `name=` parameters and infers values for `Enum`, `Literal`, optional finite, and boolean
annotations. A command can declare a dynamic domain with `Annotated`, for example `Annotated[str,
CommandCompletion(provider=session_values)]` for a zero-argument session-value provider.

Completion capabilities are registered independently with `CompletionManager`. Use
`MarkerCompletionAdapter` for another front marker, or implement `CompletionAdapter` for a custom
keyword grammar. Each adapter owns its value source and activation rules; adding one does not
require changing the manager or a shared context model.

Submitted mentions are resolved by an independently injected `MentionManager`. The default
registry attaches bounded snapshots for `@` project paths and activates explicitly selected `$`
skills before the first model request. Each mention handler owns one marker namespace, its live
completion source, validation, and resolution behavior; library callers can inject a replacement
manager or compose different handlers without modifying the manager. OpenAI requests represent
referenced files as native multipart `input_file` content and precede the payloads with a compact
manifest containing each reference's kind, path, original byte size, included byte size, and
truncation state. Inline files use MIME-qualified base64 data URLs. File contents are not
duplicated in the text manifest.

The CLI persists conversations as sessions in `.loop/sessions.db` at the Git project root. The
first user message assigns a provisional name from its first 48 characters and creates the stored
session. After the first completed answer, a separate structured-output request generates a more
descriptive name without adding the title request to conversation history. A failed title request
leaves the provisional name intact.

Use `/sessions` to show names, stable IDs, update times, and message counts in a table. Use
`/resume` with name-based completion to select a session; a captured name-to-ID resolver keeps its
stable ID internal. `/new` starts a fresh unpersisted session, and `/rename` assigns a name that
automatic generation will not overwrite. Use `/use` to load a skill for subsequent model requests,
`/tools` to list available tools, `/skills` to inspect active and discoverable skills, and
`/call` to invoke a tool directly by name. The usual `/help`, `/exit`, and `/quit` commands remain
available.

Library callers can supply a `Session` or a persisted session identifier directly:

```python
from loop import Loop, Session, SessionManager, SQLiteSessionStore

fresh = Loop(backend, session=Session())
store = SQLiteSessionStore(".loop/sessions.db")
sessions = SessionManager(session_store=store)
resumed = Loop(backend, session="019c...", session_manager=sessions)
```

`Loop` does not choose or create a session file. Library callers opt into persistence by supplying
a session manager configured with a durable store; without one, the session remains in memory.

At startup, the loop recursively indexes instruction files in the Git repository and loads the
files applicable to the current working directory, from the project root through that directory.
`AGENTS.md` is the default filename. Library callers can set `agents_filenames=("AGENTS.md",
"CUSTOM.md")`; a later filename is used only when earlier names are absent in the same directory.
Sources are combined from least to most specific. Files may optionally begin with YAML frontmatter
delimited by `---`; its metadata is validated and excluded from the instruction body, with malformed
frontmatter reported in `manage_skills` diagnostics. The 32 KiB project-instruction limit emits a
visible truncation marker when space permits, while diagnostics report source paths and exact
included and omitted sizes. Successful local file reads, directory listings, and writes update the
active instruction scope before the next model request.

When the model requests a file write or shell command, the loop displays the
operation and asks for confirmation first. File and directory reads do not
require confirmation.

### Use the backend directly

```python
from loop import AnswerDelta, OpenAIBackend

backend = OpenAIBackend(
    base_url="http://localhost:8000/v1",
    api_key="local-api-key",
    default_model="nvidia/Qwen3.6-35B-A3B-NVFP4",
)
events = backend.get_response(
    input="Explain why the sky is blue in two sentences.",
)

answer = "".join(event.text for event in events if isinstance(event, AnswerDelta))
print(answer)
```

### Asynchronous requests

```python
import asyncio

from loop import AnswerDelta, OpenAIBackend


async def main() -> None:
    backend = OpenAIBackend(
        base_url="http://localhost:8000/v1",
        api_key="local-api-key",
        default_model="nvidia/Qwen3.6-35B-A3B-NVFP4",
    )
    answer_parts = []
    async for event in backend.get_response_async(
        input="Give me three names for a lunar rover.", stream=True
    ):
        if isinstance(event, AnswerDelta):
            answer_parts.append(event.text)
    answer = "".join(answer_parts)
    print(answer)


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
are read and cached only when the model calls `manage_skills` with the `activate` action. Activated
skill identities persist with the session and are restored only while the same canonical
definition remains available. The model can deactivate one or all skills when their instructions
are no longer needed.

Skill files under `references/`, `scripts/`, and `assets/` remain unloaded until requested through
`manage_skills`. Resource paths are confined to the activated skill root. Text resources support
line-oriented or byte-oriented continuation and return at most 16 KiB per call. Binary assets are
returned as smaller base64 pages so their encoded result remains bounded; resource contents do not
become permanent instructions.

The complete instruction document retains a deterministic 64 KiB UTF-8 safety ceiling.
Section-level sizes and stable content digests make prompt growth and cache-prefix churn observable
without adding model-metadata or tokenizer requests to the backend request path. When explicitly
configured, the backend context window is reported alongside context usage after each completed
response.

A minimal skill looks like this:

```markdown
---
name: review-changes
description: Review code changes for correctness, regressions, and missing tests.
---

Follow the repository's review workflow and report findings by severity.
```

## Configuration

Pass configuration directly to `OpenAIBackend` when using a different server or model:

```python
from loop import OpenAIBackend

backend = OpenAIBackend(
    base_url="https://example.com/v1",
    api_key="your-api-key",
    default_model="your-model-id",
    max_retries=2,
)
```

Referenced text files use native `input_file` parts when `base_url` is omitted for the official
OpenAI endpoint. Setting a custom `base_url` defaults to portable `input_text` parts because many
OpenAI-compatible servers do not implement `input_file`. Override either default explicitly with
`file_input_mode="text"` or `file_input_mode="native"`. Both modes retain the reference metadata
manifest.

The executable in `main.py` resolves environment variables and applies these application defaults:

| Setting           | Environment variable  | Built-in default               |
| ----------------- | --------------------- | ------------------------------ |
| Base URL          | `BASE_URL`            | `http://localhost:8000/v1`     |
| Model             | `DEFAULT_MODEL`       | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| API key           | `OPENAI_API_KEY`      | `local-api-key`                |
| Automatic retries | `OPENAI_MAX_RETRIES`  | `2`                            |

`OpenAIBackend` itself does not read environment variables or provide deployment defaults. Library
callers configure it explicitly, and credentials remain private backend state.

Transient connection, timeout, conflict, rate-limit, and server failures use the OpenAI SDK's
bounded exponential-backoff retries. When those attempts are exhausted, the interactive loop
offers to retry the complete response. A missing selected model triggers model discovery and lets
the user choose an available replacement. Library callers still receive normalized backend
exceptions and can apply their own recovery policy.

The `fetch_content` tool sends a browser-like user agent by default. Set `USER_AGENT` to override
it for web requests.

### Tool permissions

Every model-originated tool call is authorized centrally after its arguments are validated and
before its function is invoked. The default `confirm_all` mode asks about every call, including
read-only and pure tools. A required approval is denied when no interactive user is available.

The local policy is stored at `.loop/permissions.yaml` under the Git project root. It is created
when the policy is first changed. Decisions are appended to
`.loop/permissions-audit.jsonl`. Use `/permissions` to display the active policy:

```text
/permissions
/permissions mode read_only
/permissions add allow read_text_file filesystem.read "/project/docs/*"
/permissions session deny run_command process.exec
```

Modes are `confirm_all`, `read_only`, `workspace_write`, `locked_down`, and `unrestricted`.
Rules select `allow`, `ask`, or `deny`, followed by a tool glob and optional capability and
resource globs. Deny rules take precedence over ask rules, which take precedence over allow rules.
Session rules disappear when the process exits; rules added with `add` are persisted locally.

Capabilities are `pure`, `filesystem.read`, `filesystem.write`, `filesystem.delete`,
`process.exec`, `network.read`, `network.write`, and `session.write`. The `workspace_write` mode
automatically permits in-workspace writes but asks before deletions. Shell calls are always
classified as `process.exec`; loop does not infer safety from a command prefix. Permission is an
intent check rather than a sandbox, so process-level filesystem and network isolation should still
be used for untrusted workloads.

## Built-in tools

The default registry exposes these functions to the model:

| Tool                   | Behavior                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| `list_folder`          | Lists typed file/folder entries, optionally including nested entries     |
| `read_text_file`       | Reads bounded UTF-8 text by line range                                   |
| `write_text_file`      | Writes a UTF-8 text file after centralized authorization                 |
| `delete_path`          | Permanently deletes an authorized file, symbolic link, or folder tree    |
| `get_current_datetime` | Returns the current local date and time                                  |
| `fetch_content`        | Streams authorized HTTP(S) text into a bounded resumable cache           |
| `read_cached_content`  | Reads cached text by line or opaque cursor, optionally re-fetching a URL |
| `run_command`          | Runs an authorized shell command with a 30-second timeout                |
| `manage_skills`        | Manages skill activation and progressively loads bounded skill resources |

Text reads report exact source and included byte sizes, returned ranges, truncation reasons, and
continuation positions. File reads also report line ranges while retaining the byte ceiling. As a
final safeguard, any serialized tool result above 20 KiB is cached outside conversation history;
the session receives only a bounded preview plus an opaque `read_cached_content` handle and
continuation cursor.
Web artifact handles retain source metadata on their persisted session tool result and recover
automatically after cache expiry or session resume; re-fetching is independently authorized as a
network read. Other artifact types remain process-local because they do not have a reproducible
external source.

`list_folder` follows `.gitignore` and `.agentignore` files from the Git project root through
nested folders, using Git's pattern syntax. Agent-specific rules take precedence over Git rules,
and ignored directories are not traversed. This filtering controls file discovery only; it does
not prevent an explicitly requested file from being read or changed.

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
src/loop/backend/     Backend contract and OpenAI-compatible adapter
src/loop/commands/    User-command definitions and dispatch
src/loop/completion/  Declarative completion models, adapters, and aggregation
src/loop/constants.py Shared application constants
src/loop/context/     Tool invocation context definitions
src/loop/interaction/ User interaction interfaces
src/loop/loop.py      Interactive conversation loop and streaming configuration
src/loop/mentions/    User-mention resolution
src/loop/models.py    Conversation and response events
src/loop/permissions/ Permission capabilities, requests, and policy management
src/loop/session/     Session persistence contracts and implementations
src/loop/skills/      Agent Skills, catalog, and instruction management
src/loop/tooling/     Tool registration, definitions, and dispatch
src/loop/tools/       Built-in tool implementations
src/loop/utils/       Common utilities
```
