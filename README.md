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
- Agent-scoped instructions, tools, permissions, and bounded model/tool execution
- Streaming response and reasoning output
- Synchronous and asynchronous backend access
- Responses API function-call handling
- A decorator-based registry for synchronous and asynchronous Python tools
- Built-in tools for filesystem access, shell commands, and the current date and time
- Confirmation before a model writes a file or runs a shell command
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

To install the optional tools dependency, which provides ripgrep for the
`search_text` tool, include the `tools` extra:

```bash
uv sync --extra tools
```

To install the latest version directly as a `uv` tool instead, run:

```bash
uv tool install git+https://github.com/jsconan/loop.git
```

The installation provides the `loop` command.

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

Run the chat loop, which enables streaming:

```bash
uv run loop
```

When installed as a `uv` tool, run it from any directory with:

```bash
loop
```

Enter a message at the `You:` prompt. To stop, enter `/exit`, `/quit`, or `q`.

Terminal completion is available while typing. Use `/` for commands and their known values, `@`
for visible files and directories relative to the working directory, and `$` for discovered skill
names. Matching is case-insensitive and accepts fragments at the beginning, middle, or end. File
completion respects `.gitignore` and `.agentignore`; `/permissions` completes operations, scopes,
decisions, limits, registered tools, and current rule identifiers.

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

Each session also owns an ordered event timeline for faithful replay. It records conversation-item
placement, tool-execution boundaries, compactions, permission decisions, and completed-run
statistics. Run statistics include
per-model token usage and latency, tool-function durations and outcomes, active checkpoint time,
message and item counts, model identity, and context occupancy. Human permission and recovery
waits are excluded from active time. `/resume` renders this timeline, including the
metrics originally shown after each run and any permission prompts originally presented to the
user. Historical permission decisions are informational and are never reapplied as grants.
Legacy session schemas are upcast in memory when read; loading or listing sessions never rewrites
their stored payloads. A modified session is written using the current schema on its next save.

When startup or `/resume` finds history written after the last completed run, the loop offers to
recover it before accepting another user message. Recovery resumes at the durable boundary: it
requeries after stored tool results, executes calls that definitely never started, and never
silently retries a call whose external outcome is unknown. For an uncertain call, declining the
explicit retry records a model-visible interrupted result so the model can reconcile safely.

Use `/sessions` to show names, stable IDs, update times, and message counts in a table. Use
`/resume` with name-based completion to select a session; a captured name-to-ID resolver keeps its
stable ID internal. `/new` starts a fresh unpersisted session, and `/rename` assigns a name that
automatic generation will not overwrite. Use `/use` to load a skill for subsequent model requests,
`/tools` to list available tools, `/skills` to inspect active and discoverable skills, and
`/call` to invoke a tool directly by name, and `/check` to verify backend and model availability.
The usual `/help`, `/exit`, and `/quit` commands remain
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

Each submitted user message starts one bounded agent run. The configured `Agent` owns the backend,
dynamic instructions, tool registry, and permission policy; `AgentRunner` performs repeated model
and tool turns until the model returns a final response. `Loop` remains responsible for user input,
commands, mentions, session naming, and metrics display. Runs default to at most 25 model turns so a
model that repeatedly requests tools cannot continue indefinitely; set `max_agent_turns=0` to
disable the limit (unlimited turns). Library callers can change the limit with
`Loop(..., max_agent_turns=10)` and inspect the configured identity through `loop.agent`. When the
turn limit is reached, the user is prompted to confirm whether to continue.

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
from loop import AnswerDelta, OpenAIBackend, create_default_tool_registry

backend = OpenAIBackend(
    base_url="http://localhost:8000/v1",
    api_key="local-api-key",
    default_model="nvidia/Qwen3.6-35B-A3B-NVFP4",
)
tools = create_default_tool_registry()
events = backend.get_response(
    input="Explain why the sky is blue in two sentences.",
    tools=tools.definitions(),
)

answer = "".join(event.text for event in events if isinstance(event, AnswerDelta))
print(answer)
```

### Asynchronous requests

```python
import asyncio

from loop import AnswerDelta, OpenAIBackend, create_default_tool_registry


async def main() -> None:
    backend = OpenAIBackend(
        base_url="http://localhost:8000/v1",
        api_key="local-api-key",
        default_model="nvidia/Qwen3.6-35B-A3B-NVFP4",
    )
    tools = create_default_tool_registry()
    answer_parts = []
    async for event in backend.get_response_async(
        input="Give me three names for a lunar rover.",
        stream=True,
        tools=tools.definitions(),
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
from loop import ToolContext, ToolRegistry, tool


@tool
def describe_tool(context: ToolContext, value: str) -> str:
    """Return a value labeled with its registered tool name."""
    return f"{context.tool_name} ({context.call_id}): {value}"


registry = ToolRegistry([describe_tool])
```

For model-originated calls, `context.call_id` is the stable provider call identifier and can be
used as an idempotency key by tools that perform external mutations. It is `None` for direct
user-command invocations.

Tools may declare a `preflight` check that returns a `ToolPreflightResult`. Ready tools are
registered normally. Degraded tools are registered after a warning, while broken tools are logged,
shown as a warning when an interaction is available, and omitted from the registry. Marking a tool
`required=True` asks an interactive user whether to halt or continue without a broken tool;
headless registration halts because continuing cannot be authorized.

Registries can also create tools from undecorated callables and override metadata locally without
changing the callable or another registry:

```python
registry = ToolRegistry()
registry.register(
    describe_tool,
    name="describe_value",
    description="Describe one value.",
)
```

For declarative bulk composition, wrap locally configured entries in `ToolRegistration`:

```python
from loop import ToolRegistration

registry = ToolRegistry(
    [
        ToolRegistration(
            describe_tool,
            name="describe_value",
            description="Describe one value.",
            actions=frozenset(),
            operation_planner=None,
        )
    ]
)
```

Omitting `operation_planner` inherits the passive declaration; explicitly passing `None` removes
the declared planner for that registry only. Pure tools declare no actions. Authority-bearing tools
declare an action upper bound and an operation planner that canonicalizes arguments and returns the
complete typed effect set before any implementation code runs.

The context provides invocation metadata and access to the injected user interaction service.
For loop-managed calls it also exposes the active `SkillManager`.

Tool declaration is passive: importing a tool does not register it anywhere. Compose the desired
capabilities explicitly with `ToolRegistry(BUILTIN_TOOLS)` for all bundled tools, a selected tuple
for a restricted backend, or no registry argument for a tool-less backend. Every backend without
an injected registry receives its own empty registry; no registry state is shared globally.

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

The `loop` command resolves environment variables and applies these application defaults:

| Setting           | Environment variable | Built-in default               |
| ----------------- | -------------------- | ------------------------------ |
| Base URL          | `BASE_URL`           | `http://localhost:8000/v1`     |
| Model             | `DEFAULT_MODEL`      | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| API key           | `OPENAI_API_KEY`     | `local-api-key`                |
| Automatic retries | `OPENAI_MAX_RETRIES` | `2`                            |

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
before its function is invoked. A tool first produces one complete, canonical operation plan; the
whole plan is then allowed, denied, or approved atomically. Pure tools require no authority. The
default supervised policy permits reads in the workspace and Loop-owned temporary directory, asks
for mutations and network access, and denies host-process execution at a user-configurable
boundary. Ordinary rules cannot override boundaries. A required approval is denied when no
interactive user is available.

The local policy is stored at `.loop/permissions.yaml` under the Git project root. It is created
when the policy is first changed. Decisions are appended to `.loop/permissions-audit.jsonl` and
recorded as structured events in the active session. The session event includes the normalized
request, effective result and source, whether the user was prompted, and the exact displayed
prompt. Use `/permissions` to display the active policy:

```text
/permissions
/permissions show effective
/permissions default set workspace filesystem.delete deny
/permissions default set session process.execute ask
/permissions rule add workspace allow read_text_file filesystem.read "/project/docs/*"
/permissions rule add session deny run_command process.execute "*"
/permissions limit set session host-process allow
/permissions limit add workspace network-origin https://example.com
/permissions limit add workspace read-root system-temp
/permissions limit reset session host-process
/permissions session reset
/permissions explain run_command process.execute "git status"
/permissions preset list
/permissions preset diff workspace observe
/permissions preset replace session workspace
```

The versioned YAML document is the complete policy; there is no closed set of compiled permission
postures. Each action has an independent fallback default. Rules select `allow`, `ask`, or `deny`
and match a tool glob, action, and canonical-resource glob. Deny rules take precedence over ask
rules, which take precedence over allow rules. Session rules disappear when the process exits.
`/permissions show` displays the workspace policy, session overrides, effective defaults and
limits, all active rules, and precedence. Every default, rule, and limit change takes an explicit
`workspace` or `session` scope. `/permissions reload` validates external YAML edits before replacing
the workspace layer while preserving session overrides; `/permissions explain` reports the
determining source for one concrete operation. Creating a rule with a wildcard tool, action, or
resource requires explicit confirmation because it may affect more operations than intended.

Permission presets are versioned YAML artifacts rather than compiled Python modes. The built-in
`observe`, `supervised`, `workspace`, and `locked` presets define a complete fallback decision for
every action and may include rules. Use `/permissions preset list` and `show` to inspect them,
`diff` to preview one replacement, and `replace` to replace only the selected scope's defaults and
rules after confirmation. Presets never change filesystem, network, or process enforcement limits,
nor the other policy layer. The replacement prompt lists the replaced and installed defaults and
rules, and warns that the change may relax or tighten policy. Installed rules retain the preset
identifier, revision, and content hash for policy diagnostics. The shorter `/permissions preset`
form also lists presets, while `/permissions rule` lists both rule scopes; use `rule list workspace`
or `rule list session` to filter the output. Likewise, `/permissions limit` lists workspace values
and session overrides; `limit list workspace` and `limit list session` select one layer.

Actions are `filesystem.list`, `filesystem.read`, `filesystem.create`, `filesystem.replace`,
`filesystem.delete`, `network.request`, `process.execute`, and `session.mutate`. Filesystem roots,
control paths, network origins/private addresses, and host processes are configurable enforcement
boundaries that rules cannot override. `workspace` and `loop-temp` roots are available by default;
add `system-temp` only when cross-application temporary-file access is necessary. Enabling host
processes or private-network access is explicit. Network origins use glob patterns: `*` permits
all origins, while an empty origin list denies all network requests. Adding the first specific
origin replaces the default `*`, making the boundary restrictive. Relative filesystem roots in
the YAML policy are resolved from the workspace, not from the shell's launch directory. Each Loop
instance owns a private `loop-temp` directory and announces its exact path to the model. Host
process execution requires both opening the `host-process` boundary and choosing an appropriate
`process.execute` default or rule. The built-in executor does not supply an operating-system
sandbox, so this boundary must remain closed for untrusted process execution. Ignore files limit
discovery only; they are not authorization policy.
The command tool accepts an exact argument vector and never invokes a shell, while web requests do
not follow redirects implicitly. Policy is still distinct from operating-system containment, so an
enforcement sandbox remains required before enabling untrusted process execution.

## Built-in tools

The default registry exposes these functions to the model:

| Tool                   | Behavior                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| `list_folder`          | Lists typed file/folder entries, optionally including nested entries     |
| `read_text_file`       | Reads bounded UTF-8 text by line range                                   |
| `search_text`          | Searches files or folders with bounded, structured ripgrep matches       |
| `write_text_file`      | Writes a UTF-8 text file after centralized authorization                 |
| `edit_text_file`       | Replaces exact UTF-8 text with ambiguity and change safeguards           |
| `delete_path`          | Permanently deletes an authorized file, symbolic link, or folder tree    |
| `get_current_datetime` | Returns the current local date and time                                  |
| `fetch_content`        | Streams authorized HTTP(S) text into a bounded resumable cache           |
| `read_cached_content`  | Reads cached text by line or opaque cursor, optionally re-fetching a URL |
| `run_command`          | Runs an authorized argument vector with a 30-second timeout              |
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

`search_text` performs literal smart-case matching by default, with optional regular expressions,
case control, inclusive Git-style globs, neighboring lines, and a global result limit. Folder
searches reuse Loop's ignore traversal before passing explicit visible files to ripgrep, skip
binary files and symbolic links, and return paths relative to the requested folder.
Loop does not bundle ripgrep: `rg` must be installed separately and available on `PATH`. When it
is unavailable, searches return a structured `filesystem.search_unavailable` problem.

These tools operate with the permissions of the process running `loop`. Filesystem access is
authorized but not OS-sandboxed; approved commands use an exact argument vector and never invoke
a shell. Run the project only in an environment where you are comfortable granting the model that
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
src/loop/interaction/ User interaction interfaces
src/loop/loop.py      Interactive conversation loop and streaming configuration
src/loop/mentions/    User-mention resolution
src/loop/model_selection/ Active model selection and model commands
src/loop/models.py    Conversation and response models
src/loop/permissions/ Permission capabilities, requests, and policy management
src/loop/session/     Session persistence contracts and implementations
src/loop/skills/      Agent Skills, catalog, and instruction management
src/loop/tooling/     Tool context, registration, definitions, and dispatch
src/loop/tools/       Built-in tool implementations
src/loop/utils/       Common utilities
```
