# Using Loop

## Interactive conversation

### Start a conversation

Run the chat loop, which enables streaming:

```bash
uv run loop
```

When installed as a `uv` tool, run it from any directory with:

```bash
loop
```

Enter a message at the `You:` prompt. To stop, enter `/exit`, `/quit`, or `q`.

### Commands and completion

Terminal completion is available while typing. Use `/` for commands and their known values, `@` for visible files and directories relative to the working directory, and `$` for discovered skill names. Matching is case-insensitive and accepts fragments at the beginning, middle, or end. File completion respects `.gitignore` and `.agentignore`; `/permissions` completes operations, scopes, decisions, limits, registered tools, and current rule identifiers.

Command arguments use shell-like quoting and may be supplied by position, by name, or both, for example `/command first-value count=3 enabled=true`. Each value is decoded independently and then validated against the command function's annotated parameter model. Command completion offers remaining `name=` parameters and infers values for `Enum`, `Literal`, optional finite, and boolean annotations. A command can declare a dynamic domain with `Annotated`, for example `Annotated[str, CommandCompletion(provider=session_values)]` for a zero-argument session-value provider.

Library authors can extend completion with `CompletionManager` and `CompletionAdapter`. Use `MarkerCompletionAdapter` for another marker, or implement `CompletionAdapter` for a custom keyword grammar. Each adapter owns its value source and activation rules.

### Mention files and skills

Use `@` to include project files or directories and `$` to refer to discovered skills. Loop snapshots mentioned content so the model sees bounded, consistent file contents. Explicit skill mentions help select matching skill metadata; Loop loads the skill instructions only when the model activates that skill.

Library authors can inject a `MentionManager` or compose their own handlers. The default manager resolves `@` paths into bounded snapshots and uses `$` references to select from the current skill catalog. OpenAI requests send file contents as multipart `input_file` parts, with a compact manifest for each reference. Inline files use MIME-qualified base64 data URLs.

### Sessions

The CLI stores conversations in the active workspace. A Git worktree is one workspace; outside Git, the launch directory is the workspace. Sessions are tied to their workspace and cannot be resumed from another one. The first user message creates the session and gives it a provisional name. After the first completed answer, Loop can ask the model for a more descriptive name. This title request does not appear in the conversation, and if it fails, the provisional name remains.

Model-assisted naming is the default. Library callers can use `LocalSessionNameGenerator` for deterministic local titles.

### Provider privacy and retention

Before sending requests, the OpenAI backend replaces credentials it recognizes in conversation text, attachments, and remote token-counting requests. Replacements cannot be expanded back to plaintext by model output or tool arguments. Detection is heuristic and may miss confidential text, personal information, or encoded secrets. Official OpenAI requests use `store=False`. Compatible servers may manage retention themselves; their transport setting does not guarantee provider-wide retention.

### Command paths and results

Instructions expose `/workspace`, `/tmp`, and `/skills/<name>` as virtual paths. A command's `cwd` can be `workspace`, a workspace-relative path, or a virtual workspace path. Loop translates these paths in submitted shell source and maps command output back to virtual paths; it does not rewrite scripts or binaries already on disk. Command results include the exit status and bounded stdout and stderr previews, including output captured before a timeout. Use continuation handles to read more captured output; anything beyond the capture limit is marked as lost.

For sandbox behavior and approval requirements, see [Built-in tools and command sandbox](tools.md).

### Session history and recovery

Each session records an ordered history of messages, tool calls, compactions, permission decisions, and completed-run statistics. Run statistics include token use and latency by model, tool durations and results, active time, message counts, and context use. `/resume` displays this history and the permission prompts shown at the time. Past permission decisions are shown for reference and never become current grants.

If Loop finds history from an incomplete run at startup or during `/resume`, it offers to recover before accepting another message. It continues from the last saved boundary, and does not silently repeat a tool call whose external outcome is unknown. If you decline an explicit retry, Loop records an interrupted result so the model can account for it.

Common session commands:

- `/sessions` lists session names, IDs, update times, and message counts.
- `/session` or `/sessions show` shows details for the current session.
- `/resume` selects a saved session. Name completion keeps the session ID hidden.
- `/new` starts a new, unpersisted session.
- `/rename [name]` gives the current session a name that automatic naming will not replace. `/session rename [name]` is an alias.

Other useful commands: `/use` loads a skill for later requests; `/tools` lists tools or `/tools call <tool> [arguments]` invokes one; `/skills` shows active and discoverable skills; `/call <tool> [arguments]` invokes a tool directly; `/check` checks backend and model availability. Use `/help` for command help, and `/exit` or `/quit` to stop Loop.

### Use sessions from Python

Library callers can supply a `Session` or a persisted session identifier directly:

```python
from loop import Loop, Session, SessionManager, SQLiteSessionStore

fresh = Loop.create_default(backend, session=Session())
store = SQLiteSessionStore(".loop/sessions.db", workspace_id="workspace-id")
sessions = SessionManager(session_store=store, workspace_id="workspace-id")
resumed = Loop.create_default(backend, session="019c...", session_manager=sessions)
```

`Loop` does not choose or create a session file. To persist conversations, pass a session manager configured with a durable store. Without one, the session stays in memory.

## Agents and instructions

### Agent runs

Each user message starts a bounded agent run. The default limit is 25 model turns; set `max_agent_turns=0` to remove the limit. When the limit is reached, Loop asks whether to continue. Library callers can set a different limit with `Loop.create_default(..., max_agent_turns=10)` and inspect the agent through `loop.agent`.

The default `Agent` contains its identity, built-in instructions, and model-visible tools. Its identity and built-in instructions are always included and are not truncated. The same agent definition can be reused across sessions. `AgentRunner` manages each run; `Loop` handles input, commands, mentions, session names, and displayed metrics. Library callers can create an `Agent` with explicit `AgentInstructions` when they need different instructions.

### Project instructions

At startup, Loop indexes instruction files in the Git repository and loads the files that apply from the project root through the current working directory. `AGENTS.md` is the default filename. Library callers can set `agents_filenames=("AGENTS.md", "CUSTOM.md")`; Loop uses a later filename only when earlier names are absent in the same directory. More specific project instructions follow less specific ones, then runtime context, the skill catalog, and active skills. Optional YAML frontmatter is validated and removed before instructions are sent. The complete instruction document has a 64 KiB UTF-8 safety limit. If the combined instructions exceed the configured limit, Loop reports the problem before generation instead of silently dropping rules.

Successful local file reads, directory listings, and writes update the active instruction scope before the next model request. Context diagnostics show source paths, sizes, and digests.

### File and command permissions

When the model requests a file write, Loop shows the operation and asks for confirmation. File and directory reads do not require confirmation. Routine shell commands run in the workspace sandbox without a separate prompt; extra read roots, Git metadata writes, network access, and host retries need separate approval.

The shell sandbox writes to the live checkout. Concurrent commands can change the same files. Loop does not detect these conflicts or roll back shell changes.

## Use the backend directly

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

## Asynchronous requests

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

## Context-aware tools

A decorated function can request an explicit `ToolContext`. The registry injects it during dispatch, while the context parameter is omitted from the schema exposed to the model:

```python
from loop import ToolContext, ToolRegistry, tool


@tool
def describe_tool(context: ToolContext, value: str) -> str:
    """Return a value labeled with its registered tool name."""
    return f"{context.tool_name} ({context.call_id}): {value}"


registry = ToolRegistry([describe_tool])
```

For model-originated calls, `context.call_id` is the stable provider call identifier and can be used as an idempotency key by tools that perform external mutations. It is `None` for direct user-command invocations.

Tools may declare a `preflight` check that returns a `ToolPreflightResult`. Ready tools are registered normally. Degraded tools are registered after a warning, while broken tools are logged, shown as a warning when an interaction is available, and omitted from the registry. Marking a tool `required=True` asks an interactive user whether to halt or continue without a broken tool; headless registration halts because continuing cannot be authorized.

Registries can also create tools from undecorated callables and override metadata locally without changing the callable or another registry:

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

Omitting `operation_planner` inherits the passive declaration; explicitly passing `None` removes the declared planner for that registry only. Pure tools declare no actions. Authority-bearing tools declare an action upper bound and an operation planner that canonicalizes arguments and returns the complete typed effect set before any implementation code runs.

The context provides invocation metadata and access to the injected user interaction service. For loop-managed calls it also exposes instruction-scope observation through the active `InstructionsManager`.

Tool declaration is passive: importing a tool does not register it anywhere. Compose the desired capabilities explicitly with `ToolRegistry(BUILTIN_TOOLS)` for all bundled tools, a selected tuple for a restricted backend, or no registry argument for a tool-less backend. Every backend without an injected registry receives its own empty registry; no registry state is shared globally.

## Agent Skills

At startup, the loop discovers `SKILL.md` files under `.agents/skills` from the repository root through the working directory, followed by `~/.agents/skills`. Only each skill's YAML `name` and `description` are initially loaded and disclosed to the model. Before task work, the agent compares the request with those descriptions and calls `activate_skill` for every match. The complete Markdown instructions are then read and cached for the following model turn. Activated skill identities persist with the session and are restored only while the same canonical definition remains available. The model can deactivate one or all skills when their instructions are no longer needed.

Skill files under `references/`, `scripts/`, and `assets/` remain unloaded until requested through `manage_skills`. Resource paths are confined to the activated skill root. Text resources support line-oriented or byte-oriented continuation and return at most 16 KiB per call. Binary assets are returned as smaller base64 pages so their encoded result remains bounded; resource contents do not become permanent instructions.

The complete instruction document retains a deterministic 64 KiB UTF-8 safety ceiling. Section-level sizes and stable content digests make prompt growth and cache-prefix churn observable without adding model-metadata or tokenizer requests to the backend request path. When explicitly configured, the backend context window is reported alongside context usage after each completed response.

A minimal skill looks like this:

```markdown
---
name: review-changes
description: Review code changes for correctness, regressions, and missing tests.
---

Follow the repository's review workflow and report findings by severity.
```
