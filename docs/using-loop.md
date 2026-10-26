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

Enter a message at the `You:` prompt. Use `/stop` to settle the active run and keep the conversation available. Send another message to resume with new instructions, or use `/continue` to resume without adding a message. Use `/exit`, `/quit`, or `q` to leave the application.

The editor stays available while the agent works. Plain text follows `conversation.input_mode`, which defaults to queueing a follow-up; set it to `steer` to interrupt the active run with plain text. `/queue <message>` and `/steer <message>` select a priority explicitly. `/queue` and `/steer` without text change the newest message still waiting in the mailbox. `/status` reports whether Loop is idle, working, waiting for a response, or interrupting, along with the number of queued inputs.

Use `/interrupt` to end the active agent run while keeping the conversation open. Existing queued follow-ups remain accepted and run afterward. Use `/steer <message>` to interrupt the active run and process a correction before those follow-ups. Multiple accepted corrections retain their submission order. `/edit` or `/recall` removes the newest waiting message and places its text in the editor; submitting it creates a new queued input. Cancelling the editor leaves the recalled input removed. Queue and steering arguments remain message text even when they begin with `/`.

Use `/pause` to park the active run at a safe boundary while its owner stays available; `/continue` releases that same run. Use `/stop` to settle the run at a safe boundary and leave a cold durable recovery point in the same conversation. `/continue` resumes paused or stopped runs, including runs stopped at the turn limit. Use `/retry` only when the durable run history records a failure, such as when you decline an automatic retry. Sending a message also resumes a paused or stopped run and authorizes one recovery attempt after a recorded failure. The new message is saved before recovery, and unstarted calls from the old response are cancelled so the next model request uses the new instructions. A live pause treats the message as steering even in queue mode. Messages queued before a stop remain parked until continuation or a newly submitted message. Continue, retry, and automatic recovery preserve completed results; a tool call with an unknown external outcome still asks before it is repeated.

Interruption acts at the next safe boundary. Partial model output is discarded, completed tool results are preserved, and remaining tool calls receive explicit interruption results instead of being executed. An uncertain external outcome stays marked unknown. Pending approval requests are revoked; corrections never grant tool permission. New work retains normal authorization, recovery, and turn-limit checks. `/interrupt` records a cancelled run; `/pause` leaves it incomplete for durable recovery.

When a tool, command, or recovery operation asks a question, press Enter on an empty editor or use `/respond` to open that question's input prompt. Text already queued for the conversation never becomes an approval. An unsent editor draft stays intact when a question arrives. Startup questions appear before ordinary conversation input begins.

Up to 16 inputs, including corrections, may wait behind active work. A full queue explicitly rejects further inputs; rejected steering does not interrupt the current run. With a durable session store, Loop saves accepted text before confirming queue admission and restores unconsumed inputs when that session is reopened. Recovery of an interrupted run comes first; queued follow-ups stay pending until recovery is resolved. Inputs are consumed atomically with their canonical user messages, and attached paths are resolved and snapshotted at that point. Memory-only sessions cannot restore inputs after process exit.

`/stop` changes run state and leaves the conversation owner, controller, and application runtime in place. `/exit` and `/quit` close the application. Pause, stop, continue, and retry use run state and durable events; they do not cancel or rebuild the owner. Loop checks the state at safe execution boundaries, and a stop waits for the current operation to reach one. Blocking model reads and synchronous tools finish their current operation under existing deadlines before observing interruption or shutdown; unbounded custom tools can delay either indefinitely. Such tools must impose their own deadlines. Immediate cancellation of arbitrary tools requires a cancellable execution contract rather than terminating their worker thread.

### Commands and completion

Terminal completion is available while typing. Use `/` for commands and their known values, `@` for visible files and directories relative to the working directory, and `$` for discovered skill names. Matching is case-insensitive and accepts fragments at the beginning, middle, or end. File completion respects `.gitignore` and `.agentignore`; `/permissions` completes operations, scopes, decisions, limits, registered tools, and current rule identifiers.

Command arguments use shell-like quoting and may be supplied by position, by name, or both, for example `/command first-value count=3 enabled=true`. Each value is decoded independently and then validated against the command function's annotated parameter model. Command completion offers remaining `name=` parameters and infers values for `Enum`, `Literal`, optional finite, and boolean annotations. A command can declare a dynamic domain with `Annotated`, for example `Annotated[str, CommandCompletion(provider=session_values)]` for a zero-argument session-value provider.

Library authors can extend completion with `CompletionManager` and `CompletionAdapter`. Use `MarkerCompletionAdapter` for another marker, or implement `CompletionAdapter` for a custom keyword grammar. Each adapter owns its value source and activation rules.

Live completion sources are available while the conversation is idle. Completion is suspended during active execution so it cannot read changing session, policy, or instruction state.

### Mention files and skills

Use `@` to include project files or directories and `$` to refer to discovered skills. Loop snapshots mentioned content so the model sees bounded, consistent file contents. Explicit skill mentions help select matching skill metadata; Loop loads the skill instructions only when the model activates that skill.

Library authors can inject a `MentionManager` or compose their own handlers. The default manager resolves `@` paths into bounded snapshots and uses `$` references to select from the current skill catalog. OpenAI requests send file contents as multipart `input_file` parts, with a compact manifest for each reference. Inline files use MIME-qualified base64 data URLs.

### Sessions

The CLI stores conversations in the active workspace. A Git worktree is one workspace; outside Git, the launch directory is the workspace. Sessions are tied to their workspace and cannot be resumed from another one. The first user message creates the session and gives it a provisional name. In asynchronous runs, Loop starts a separate title request as soon as it has saved the first non-empty assistant response, while the agent continues handling that response. The title request does not appear in the conversation, and Loop keeps the provisional name if generation fails or the session is renamed before the generated title is saved. Synchronous `Loop.run()` calls start naming after each agent run completes.

Model-assisted naming is the default. Library callers can use `LocalSessionNameGenerator` for deterministic local titles.

### Provider privacy and retention

Before sending requests, the OpenAI backend replaces credentials it recognizes in conversation text, attachments, and remote token-counting requests. Replacements cannot be expanded back to plaintext by model output or tool arguments. Detection is heuristic and may miss confidential text, personal information, or encoded secrets. Official OpenAI requests use `store=False`. Compatible servers may manage retention themselves; their transport setting does not guarantee provider-wide retention.

### Command paths and results

Instructions expose `/workspace` and `/skills/<name>` as virtual paths and `$TMPDIR` as the session working root for commands and file tools. A command's `cwd` can be `workspace`, a workspace-relative path, or a virtual workspace path. Loop translates workspace and skill paths in submitted shell source and maps command output to logical paths; `$TMPDIR` uses ordinary shell expansion through the subprocess environment. `/tmp` is an OS path outside the session retention guarantee. Loop does not rewrite scripts or binaries already on disk. Command results include the exit status and bounded stdout and stderr previews, including output captured before a timeout. Use continuation handles to read more captured output; anything beyond the capture limit is marked as lost.

For sandbox behavior and approval requirements, see [Built-in tools and command sandbox](tools.md).

### Session history and recovery

Each session records an ordered history of messages, tool calls, compactions, permission decisions, and completed-run statistics. Run statistics include token use and latency by model, tool durations and results, active time, message counts, and context use. `/resume` displays this history and the permission prompts shown at the time. Past permission decisions are shown for reference and never become current grants.

If Loop finds history from an incomplete run at startup or during `/resume`, it offers to recover. If you decline, sending a message resumes with those new instructions; `/continue` starts recovery without adding a message. `/continue` also resumes a run paused with `/pause`, settled with `/stop`, or waiting after the turn-limit prompt was declined. Use `/retry` only after a failure event, such as when you decline an automatic backend retry. Both commands continue from the last saved boundary and do not silently repeat a tool call whose external outcome is unknown; those calls require per-tool confirmation.

Common session commands:

- `/sessions` lists persisted session names, IDs, update times, and message counts; `/sessions export <id> [path]` exports that session without resuming it.
- `/session` shows details for the current session; `/session export [path]` exports only that session. `/export` exports the current session; `/export <id> [path] [format]` selects a persisted session without resuming it. For an explicit destination on the active session, use `/export destination=<path>`.
- `/resume` selects a saved session. Name completion keeps the session ID hidden.
- `/new` starts a new, unpersisted session.
- `/rename [name]` gives the current session a name that automatic naming will not replace. `/session rename [name]` is an alias.
- If `path` is omitted, Loop creates `loop-session-<session-id>.md` in the current directory; long or filesystem-unsafe IDs use a stable hash in the filename. Use `format=archive` for a self-contained ZIP; the generated filename ends in `.zip`. If the destination already exists, Loop asks a yes-or-no question and defaults to no.

Markdown exports include conversation messages, tool calls and results, displayed reasoning, and timeline markers. Reasoning uses the summary when available, otherwise the provider-returned reasoning text, matching the session display. JSON tool results are pretty-printed in JSON code blocks; other results are preserved as text. Markdown exports omit encrypted provider continuation state and captured file contents. Archive exports preserve the serialized session and include verified referenced file snapshots. Treat both formats as sensitive because they may contain conversation text and tool data; archives may also contain captured context and provider state.

Each session has private working storage shared by the agent's command `$TMPDIR` and file-tool paths such as `$TMPDIR/report.txt`. Saved sessions retain these files across application restarts on the same machine; storage is isolated by database, workspace, and session identity. Selecting a different conversation is allowed while idle and rejected during model or tool execution. Working storage is kept separately from conversation history and is not included in Markdown or archive exports. It has no automatic expiry; copy important deliverables into the workspace. Validated existing storage in the same namespace is adopted in place. Bound storage that is missing, replaced or unsafe causes an explicit recovery failure instead of an empty replacement. Native ownership locks exclude cooperating harness operations on one session while allowing independent sessions to run concurrently. They do not terminate detached macOS writers or revoke old file descriptors.

Archive exports are records for inspection or external processing. Loop does not currently import an archive back into its session store.

Examples:

```text
/session export
/session export format=archive
/session export ./session.md
/session export ./session.zip format=archive
/export
/export format=archive
/export destination=./session.md
/export <session-id>
/export <session-id> ./saved-session.md archive
/export <session-id> destination=./saved-session.zip format=archive
/sessions export <session-id>
/sessions export <session-id> ./saved-session.md format=archive
```

Other useful commands: `/use` loads a skill for later requests; `/tools` lists tools or `/tools call <tool> [arguments]` invokes one; `/skills` shows active and discoverable skills; `/call <tool> [arguments]` invokes a tool directly; `/check` checks backend and model availability. Use `/help` for command help, `/stop` to leave a recoverable run in the current conversation, and `/exit` or `/quit` to leave Loop.

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

### Input, presentation, and response collection

`InteractionInput` defines synchronous prompts and confirmation. `InteractionOutput` defines presentation without reading input. `Interaction` combines both interfaces for existing adapters; the separate `AsyncInteractionInput` capability defines asynchronous prompts and confirmation. `ConsoleInteraction` implements all of them. Its `prompt_async()` and `confirm_async()` methods use native asynchronous terminal input with the same choices, completion, masking, defaults, and exit handling as synchronous prompts. Task cancellation propagates; terminal interruption or end-of-file returns `False`. Interrupted asynchronous confirmation declines the operation. Only one task should read a terminal's input at a time.

`ResponseCollector` validates normalized backend events without a UI or session store. Feed it events with `add()`, exhaust the source, then call `finish()`:

```python
from loop import ResponseCollector

collector = ResponseCollector()
for event in backend.get_response(input="Explain the result."):
    collector.add(event)
response = collector.finish()
```

The explicit `ResponseCompleted` event supplies authoritative final text and metadata. Missing completion, duplicate completion, and any event after completion raise `BackendResponseError`. Unknown extension events before completion are ignored. Partial text is never returned as a completed response.

`ResponsePresenter` renders accepted events through an `InteractionOutput`, suppressing repeated completed text for channels that already streamed. Create one presenter per response, use the output's `response_context()` to finalize rendering on success or failure, and validate each event with the collector before presenting it.

`SessionManager.response()` coordinates collection and presentation for a synchronous iterable; `response_async()` does the same for an asynchronous iterable. Both accept an output-only `interaction` override, finalize presentation on failure or cancellation, and leave session history and persistence unchanged. The event producer owns stream cleanup; use `aclosing()` when consuming an asynchronous generator. Inside an async function:

```python
from contextlib import aclosing
from loop import ConsoleInteraction, SessionManager

console = ConsoleInteraction()
sessions = SessionManager(interaction=console)
async with aclosing(backend.get_response_async(input="Explain the result.", stream=True)) as events:
    response = await sessions.response_async(events)
```

### Responsive conversation execution

`SessionController` runs a synchronous conversation on one execution thread while its asyncio loop owns UI input and presentation. Build the conversation with the controller's `interaction` and `control`, and use the supplied mailbox reader for ordinary conversation input:

```python
from loop import ConsoleInteraction, Loop, SessionController, SessionManager
from loop.conversation import ConversationCommands


async def chat(backend):
    controller = SessionController(ConsoleInteraction())
    sessions = SessionManager(interaction=controller.interaction)
    conversation = Loop.create_default(
        backend,
        interaction=controller.interaction,
        run_control=controller.control,
        session_manager=sessions,
    )
    controller.bind_session_manager(sessions)
    conversation.register_command_provider(ConversationCommands(controller))
    controller.bind_command_manager(conversation.command_manager)
    async def own_conversation(receive):
        await conversation.run_async(input_source=receive)

    await controller.run_async(own_conversation)
```

Synchronous callers can pass a synchronous conversation callback to `controller.run()`. That entry point owns its asyncio loop and cannot be called from a thread that already has a running event loop; async callers should use `run_async()`.

Each controller runs once. The injected `RunControl` owns a scoped interrupt, pause, or cold stop for each agent run: `interrupt()`, `pause()`, and `request_stop_run()` report whether the active run accepted the signal. `active`, `interrupted`, `paused`, and `stop_requested` report its state. `execution()` brackets controlled work; `checkpoint()` and `wait()` observe interruption and cold stop, and park through pause until continue. An interrupt records a cancelled run; pause and cold stop leave the durable boundary available for recovery. Pause keeps pending questions intact; interruption can revoke them. Scoped signals clear on settlement; application shutdown remains sticky.

`stop()` can request shutdown from the UI or another thread; `running` stays true until owned execution has settled. Do not directly mutate conversation services while it runs. Synchronous interaction adapters continue to use the existing direct `Loop.run()` path. `ApplicationRuntime` selects mailbox execution when its UI implements `AsyncInteractionInput` and refuses to close services while execution is active. Use `close()` for a synchronous runtime and await `close_async()` for a runtime that initialized asynchronous resources, on their owning event loop before it closes.

`Interaction.terminal_context()` reserves stdin for operations that read the terminal directly, such as administrator authentication. Synchronous adapters need no handoff. The mailbox UI asks for an explicit handoff, suspends its editor, and restores it after the scope exits. Questions inside that scope still use their normal explicit confirmation; handing over the terminal does not grant permission for an operation.

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

Synchronous tools can use `context.confirm()` and `context.authorize_additional()`. Native async tools should await `context.confirm_async()` and `context.authorize_additional_async()` so input and runtime-discovered permissions stay on the owner loop. Direct async calls run context-free synchronous tools through bounded worker admission; context-aware synchronous tools must be dispatched by `ToolRegistry` so their owner services can be relayed safely.

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
