# Configuration

## Config files and scope

The CLI stores a complete, commented user configuration in the platform-native application configuration directory. An optional `.loop/config.toml` contains workspace-specific overrides. Both files are private to your account; do not commit secrets from the local override.

Most settings resolve in this order, from lowest to highest priority: built-in defaults, user configuration, workspace overrides, environment variables, and session overrides. Configuration commands ask for a scope when one is omitted. `app.yolo` accepts a trusted user setting or a session setting; `/config set app.yolo` defaults to session scope. Workspace files cannot control YOLO, and unrelated environment variables such as `APP` and `APP__YOLO` are ignored.

The application uses one active workspace per process. Sessions and permission policies belong to that workspace. Telemetry, the audit database, and the `app.log` file are shared across workspaces.

Set `LOOP_CONFIG_HOME`, `LOOP_DATA_HOME`, or `LOOP_STATE_HOME` to override the corresponding platform-native directory. Run `/app dirs` to see the resolved locations.

New CLI sessions keep working files in a private directory under `~/.loop-scratch` or under `scratch` below `LOOP_DATA_HOME`. Loop prefers a path without whitespace and requires a location outside the checkout. Quote `$TMPDIR` in shell commands so they work with either path. Existing sessions reopen their recorded working directories while those locations remain safe and outside the active checkout.

Use these commands to manage workspace identities and locations:

- `/workspace attach <path> [workspace-id]` registers a location or associates it with an existing workspace identity.
- `/workspace forget <id-or-path>` deactivates a location mapping without deleting the workspace's stored data.
- `/workspace rekey <path>` gives a copied workspace a fresh identity while retaining the original identity and data.
- `/workspace switch <id-or-path>` switches the active workspace and rebuilds workspace-scoped services. If rebuilding fails, Loop restores the previous workspace.

Forget and rekey require confirmation. Loop imports legacy sessions, policies, logs, and audit records into the new stores without deleting the original files. Interrupted imports can be retried safely.

## See all settings

The generated file contains every setting and its built-in value. The example below shows commonly used settings. Settings that can be unset, such as `context_window`, `file_input_mode`, `model`, `temperature`, and `reasoning_effort`, appear as commented `<unset>` entries because TOML has no null value.

Existing version 1 files with a `[loop]` settings table remain readable as `[app]`; Loop writes the current table name when that file is next saved. A file containing both tables is rejected because their precedence is ambiguous.

```toml
# user config.toml
config_version = 1

[backend]
base_url = "http://localhost:8000/v1"
default_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
api_key = "local-api-key"
max_retries = 2
request_timeout_seconds = 600.0
# max_generation_seconds = <unset>
# max_response_chars = <unset>
# max_output_tokens = <unset>
repetition_detection = "auto" # "off" | "client" | "server" | "both" | "auto"
# repetition_penalty = <unset>
repetition_min_pattern_size = 1024
repetition_max_pattern_size = 16384
repetition_min_count = 3
structured_output_mode = "auto"
structured_output_max_retries = 1
hyperparameter_policy = "fallback"

[conversation]
input_mode = "queue" # "queue" | "steer"

[app]
agent_name = "Loop"
# temperature = <unset>
# reasoning_effort = <unset>
stream = true
debug = false
yolo = false
compaction_threshold = 0.8
prompt_on_recoverable_error = true
max_agent_turns = 25

# Web requests use a browser-like user agent by default.
# See "Web requests" below to override it.

[context]
mode = "basic" # "basic" | "off"
show_timezone = true
show_country = false
show_workspace_path = false
# timezone = "Europe/Luxembourg"
# country = "LU"

[tools]
sandbox_environment = {}
sandbox_path_aliases = []
command_job_runtime_limit_seconds = 3600.0
command_default_initial_wait_seconds = 10.0
command_default_poll_wait_seconds = 30.0
command_max_wait_seconds = 60.0

[logging]
level = "INFO"
max_bytes = 5242880
backup_count = 3

[telemetry]
queue_capacity = 4096
batch_size = 128
flush_seconds = 0.1
shutdown_timeout = 2.0
sqlite_busy_timeout_ms = 2500
```

## Edit settings while Loop is running

Loop validates settings at startup. If loading fails, it reports a redacted diagnostic and asks whether to retry after repair, use built-in defaults for this session, or halt. The defaults bypass ignores saved and environment settings, turns YOLO and direct-host execution off, and leaves saved files intact until you explicitly save replacement settings. Saving from defaults archives the original file first; a concurrent edit requires reloading before saving. Missing configuration is created from defaults. It uses the built-in backend settings, so check the destination before sending requests. On reload failure, Loop offers retry, continuing with the last valid settings, or halting. When a specific invalid file can be verified, recovery also offers to archive it in a private directory and reset that scope; environment and ambiguous failures require manual repair. Changes made through `/config` preserve comments and formatting in the configuration file.

Use `/config` to browse, inspect, change, or reset effective settings. Completion suggests setting names and values:

```text
/config
/config get app.temperature
/config set app.temperature 0.7 scope=session
/config set app.stream false scope=workspace
/config set app.yolo true
/config get app.yolo
/config set app.yolo false scope=session
/config secret backend.api_key
/config reset app.temperature scope=workspace
/config reset  # prompts for scope and confirmation
```

Reset behavior depends on the scope. At workspace scope, reset removes the override so the user setting applies. At user or session scope, reset restores the built-in default.

Session settings apply only to the current Loop process and take precedence over environment and file values. A file value remains saved when an environment variable overrides it, so the saved value applies again when the variable is absent. Live settings apply on the next relevant operation. Logging and telemetry changes require a restart. API-key input and configuration output are masked.

The command settings name four separate durations. `tools.command_job_runtime_limit_seconds` sets the finite, positive deadline for the whole command, starting when the native process launches. `tools.command_default_initial_wait_seconds` is the default wait in the first `run_command` call; `tools.command_default_poll_wait_seconds` is the default wait in `poll_command_job`. Both defaults accept finite nonnegative values. `tools.command_max_wait_seconds` is the shared cap for an explicit or default `wait_seconds` value in either call; it defaults to 60 seconds and may be configured from 0 through 3,600 seconds. The initial and poll defaults must each be no greater than this cap. A wait of `0` returns immediately. A poll ends only when its wait expires or the command completes, regardless of output arriving during the wait. The runtime limit is captured when a job starts; wait settings apply to subsequent calls. Runtime limits that exceed the clock's ability to preserve the native preparation deadline are rejected.

Ordinary commands inherit the selected search path. When a project launcher or a linked workspace executable identifies an additional installed runtime, the command approval includes its readable paths. The settings below provide explicit overrides for other installations and project-specific environment variables.

Set `tools.sandbox_environment` to a table of non-secret variables for sandboxed commands. String values are literal. A literal `PATH` replaces the inherited search path for both executable lookup and execution. Other entries are assigned after the shell enters confinement. Use `{ output = "relative/path" }` to set a variable to a path beneath that command's private `$LOOP_OUTPUT_DIR`. Use `.` to select the output directory itself. Loop creates the attempt directory. The command creates any additional directories or files. Loop reserves `IFS` and its generated workspace, home, temporary, cache and transport variables.

```toml
[tools.sandbox_environment]
REPORT_FILE = { output = "report.json" }
```

Set `tools.sandbox_path_aliases` to exact absolute directory symlinks when an already approved runtime needs an alternative spelling. Loop activates an alias when a saved read rule or the command's requested read roots cover its target. Unavailable aliases stay dormant. A command that selects an executable through an unapproved alias needs explicit read access to the canonical runtime directory. A declaration of a variable, search path or alias grants no file or network access. [Sandbox command and runtime access](tools.md#run-shell-commands).

To select an installed Bash by name, put its `bin` directory first in the configured search path. Include the system directories your commands use and grant read access to the actual runtime directory and any support directories through `/permissions path set workspace read` if they are outside the platform preset. For commands that automatically traverse the entire checkout, put this setting in user configuration or apply it for the session. A workspace-local `.loop` directory is protected, including its metadata on macOS. Traversal tools using that configuration layout must exclude the protected directory through their own discovery options. On macOS, Loop still starts command text in Dash and an absolute `#!/bin/bash` shebang still selects Apple's Bash. [Sandbox paths](permissions.md#sandbox-paths).

```toml
[tools.sandbox_environment]
PATH = "/path/to/installed/bash/bin:/usr/bin:/bin"
```

Host account lookup and stock process pools start off on macOS. The agent can request an available service when the command needs it, and the owner reviews its effects in the prepared command approval. A previously enabled workspace service is included when an ordinary command invokes a recognized project launcher or build recipe. A saved verification task uses only the services in its owner-controlled definition. [Optional sandbox capabilities](permissions.md#optional-sandbox-capabilities).

Edit the TOML file to supply a table for `tools.sandbox_environment` or an array for `tools.sandbox_path_aliases`. A saved verification task can supply its own `environment` object through `/verify save`. Use `{"output":"report.json"}` for a task-specific output path. Task entries override workspace entries of the same name.

For installations with versioned directories, you can put a stable directory symlink on `PATH` and declare the exact symlink in `tools.sandbox_path_aliases`. The separate read scope must cover the current target. Loop revalidates bound search directories and executable identities at admission. If the install changes during approval, submit a fresh request after the upgrade.

`app.allow_direct_host_execution` is off by default. A trusted user may enable it for the session with `/config set app.allow_direct_host_execution true`, or persist it with `scope=user`. Workspace configuration, environment variables, permission presets, and model requests cannot enable it. Persistent activation uses the same owner-private storage checks as YOLO. `/config get app.allow_direct_host_execution` shows its effective state and source. Enabling it makes the tool eligible for the next context window. Disabling it withdraws dispatch authority immediately, invalidates pending approvals, and leaves remembered grants dormant. It does not terminate an already running host process.

When enabled, `request_host_command` requests a separate unrestricted run with the exact command, working directory and an agent-supplied reason. Managed host prohibitions and explicit host denies still apply. An explicit host `ask` requires one-time human approval. With YOLO also enabled, eligible host requests can execute without a prompt and do not save a grant. Sandbox resource denials do not constrain an authorized host run. A sandbox failure never enables this setting or executes a host retry.

`app.yolo` is off by default. Enabling it skips Loop confirmation prompts for eligible operations: the agent can request private data, change or delete files, and use networks. Ordinary commands remain sandboxed. If the separate trusted `app.allow_direct_host_execution` setting is enabled, eligible explicit host requests run unrestricted without a prompt. Hard policy denies, invalid identities, and unavailable enforcement still stop requests; the native sandbox remains active for ordinary sandboxed commands. Loop displays this warning before a human `/config set app.yolo true` takes effect. To keep YOLO active after restart, explicitly run `/config set app.yolo true scope=user`. A session setting takes precedence over the saved value. A child process launched inside a native sandbox inherits that sandbox even when its own YOLO setting is enabled. `/config get app.yolo` shows the effective state, source, and saved user value. A session `false` masks a saved user `true` only until restart; set `scope=user` to `false` to keep it off. A user `false` does not erase an active session `true`; disable that session too. Startup announces whether a trusted saved activation is active or suppressed by managed policy. Persistent activation requires an owner-private, real user configuration directory and file. An untrusted or ambiguous saved activation fails closed.

YOLO grants each eligible request once and does not save an approval rule. A failed sandbox command never automatically launches a host command. Recovery that could repeat a previous effect or needs a missing choice stays unresolved for an explicit action. Disabling YOLO affects new decisions; a process already running may retain its admitted authority, so stop the active run when immediate withdrawal is needed. An installed [protected administrator policy](permissions.md#optional-administrator-policy) can prohibit YOLO activation, deny operations, require sandbox execution, prohibit host execution, or require a fresh human decision; YOLO cannot override those restrictions. Managed enforcement is unavailable on an unmanaged installation.

While an agent run is active, `conversation.input_mode = "queue"` admits plain text as a follow-up that runs after the current operation settles. Set it to `"steer"` to admit plain text at steering priority and stop the run at its next safe boundary. Explicit `/queue` and `/steer` commands override this setting.

## Environment and time

`[context]` uses the normal user, workspace, and session scopes. Workspace settings override user settings, and session edits apply for the current Loop process. For example, `/config set context.show_country true scope=session` opts into country disclosure for the current process. Basic disclosure supplies the execution OS and architecture, command shell, and the `$WORKSPACE_ROOT` alias. Set `show_workspace_path` to disclose the resolved tool directory. Set `mode = "off"` to disable automatic disclosure. The clock and runtime path guidance remain available.

Loop supplies the local date only in a verified request suffix, outside replayed history. `show_timezone` controls the timezone name and offset in that suffix. Routes without a verified suffix use `get_current_datetime` for the date and time. This keeps a date or offset change from rewriting the existing prefix when a session is reopened. Both the automatic date and the clock use a configured IANA `timezone` override, then the trusted client timezone, then host local time. Timezone changes apply immediately.

With `show_country` enabled, Loop uses a configured two-letter `country` override or the explicit region from the process locale. Locale discovery checks `LC_ALL`, `LC_TIME`, and `LANG` in that order. The region reflects locale settings. Loop omits the field when the effective locale supplies no two-letter region. Country disclosure defaults to off. Changes to disclosed non-time facts can change the leading prompt. Disclosure controls apply as soon as they are loaded or edited. Saved request history, compaction, and title requests omit the automatic facts.

## Context window settings

The first model request pins the agent name, instruction template, file input mode, ordered tool definitions, and four command timing values. Changes to these settings show as pending in `/config` with their current effective values. The next window uses the saved values after a successful compaction, an actual model or provider switch, or when a new session starts. Reopening a session restores the pinned values. A pending edit leaves the automatic compaction threshold unchanged.

Temperature and reasoning effort apply on the next model request. On routes that support append-only reasoning changes, Loop keeps the initial request-level effort and replays changes at their original positions. Other endpoints use the current effort on each request. Replacing a backend client on the same route for a timeout or credential change preserves the window.

## Configure the backend from Python

Pass settings directly to `OpenAIBackend` when using a different server or model:

```python
from loop import OpenAIBackend

backend = OpenAIBackend(
    base_url="https://my-host.local/v1",
    api_key="your-api-key",
    default_model="your-model-id",
    max_retries=2,
    request_timeout_seconds=600.0,
    repetition_detection="auto",
    repetition_min_pattern_size=1024,
    repetition_max_pattern_size=16384,
    repetition_min_count=3,
)
```

## Backend behavior

### Repetition and response limits

For compatible local models, `repetition_detection = "auto"` sends optional repetition settings to the server and records a sanitized advisory when Loop detects a substantial exact repeated block. `client` and `both` opt in to client-side cancellation. `server` uses the client detector only if the server rejects its extension. A server `repetition_detected` result is retried once. These server settings are optional and fall back gracefully.

`request_timeout_seconds` limits idle time between transport activity; it is not a total generation deadline. Loop has no generation-duration or response-length limit by default. For unattended deployments, set `max_generation_seconds`, `max_response_chars`, or `max_output_tokens` when a budget is needed. Write large artifacts with file tools instead of returning them as chat text.

### File mentions

Compatible endpoints use portable `input_text` parts for referenced text files by default. The SDK's default OpenAI endpoint uses native `input_file` parts. Set `file_input_mode="text"` or `file_input_mode="native"` to choose explicitly.

Project-path mentions use immutable content snapshots so resumed conversations see the same file contents. Loop includes at most two small files directly. Larger files, additional files, and directory contents use bounded previews and `read_cached_content` handles for the remainder. Loop stores snapshot bytes once and loads them into model context when needed.

## Environment variables

The `loop` command accepts environment overrides for automation:

| Setting                     | Environment variable                        | Built-in default               |
| --------------------------- | ------------------------------------------- | ------------------------------ |
| Base URL                    | `BASE_URL`                                  | `http://localhost:8000/v1`     |
| Backend model               | `DEFAULT_MODEL`                             | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| API key                     | `OPENAI_API_KEY`                            | `local-api-key`                |
| Context window              | `CONTEXT_WINDOW`                            | unset                          |
| Automatic retries           | `OPENAI_MAX_RETRIES`                        | `2`                            |
| Temperature                 | `LOOP_TEMPERATURE`                          | unset                          |
| Reasoning effort            | `LOOP_REASONING_EFFORT`                     | unset                          |
| Hyperparameter policy       | `OPENAI_HYPERPARAMETER_POLICY`              | `fallback`                     |
| Web user agent              | `USER_AGENT`                                | browser-like user agent        |
| Command runtime limit       | `LOOP_COMMAND_JOB_RUNTIME_LIMIT_SECONDS`    | `3600.0`                       |
| Command initial wait        | `LOOP_COMMAND_DEFAULT_INITIAL_WAIT_SECONDS` | `10.0`                         |
| Command completion poll     | `LOOP_COMMAND_DEFAULT_POLL_WAIT_SECONDS`    | `30.0`                         |
| Maximum command wait        | `LOOP_COMMAND_MAX_WAIT_SECONDS`             | `60.0`                         |
| Agent name                  | `LOOP_AGENT_NAME`                           | `Loop`                         |
| Active input mode           | `LOOP_CONVERSATION_INPUT_MODE`              | `queue`                        |
| Model                       | `LOOP_MODEL`                                | unset                          |
| Stream responses            | `LOOP_STREAM`                               | `true`                         |
| Debug responses             | `LOOP_DEBUG`                                | `false`                        |
| Compaction threshold        | `LOOP_COMPACTION_THRESHOLD`                 | `0.8`                          |
| Prompt on recoverable error | `LOOP_PROMPT_ON_RECOVERABLE_ERROR`          | `true`                         |
| Maximum agent turns         | `LOOP_MAX_AGENT_TURNS`                      | `25`                           |
| Log level                   | `LOOP_LOG_LEVEL`                            | `INFO`                         |
| Log size                    | `LOOP_LOG_MAX_BYTES`                        | `5242880`                      |
| Log backups                 | `LOOP_LOG_BACKUP_COUNT`                     | `3`                            |
| Telemetry queue capacity    | `LOOP_TELEMETRY_QUEUE_CAPACITY`             | `4096`                         |
| Telemetry batch size        | `LOOP_TELEMETRY_BATCH_SIZE`                 | `128`                          |
| Telemetry flush interval    | `LOOP_TELEMETRY_FLUSH_SECONDS`              | `0.1`                          |
| Telemetry shutdown timeout  | `LOOP_TELEMETRY_SHUTDOWN_TIMEOUT`           | `2.0`                          |
| SQLite busy timeout         | `LOOP_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS`     | `2500`                         |

The CLI supplies its configured `backend.base_url` to the adapter and defaults to the local compatible endpoint above. A non-empty `base_url` passed to `OpenAIBackend` takes precedence over `OPENAI_BASE_URL`. To use the SDK's default OpenAI endpoint, leave `OPENAI_BASE_URL` unset and omit `base_url` in Python or set `backend.base_url = ""` in the CLI.

## Hyperparameters and retries

`temperature` and `reasoning_effort` are optional per-agent settings, so agents can share a backend with different controls. With `hyperparameter_policy = "fallback"`, Loop retries a request without a parameter if the selected model rejects it. Use `"strict"` to keep the provider error instead. Loop also checks compatible-provider model metadata when available and omits controls declared unsupported. Set either `temperature` or `top_p`; the OpenAI API recommends changing one sampling control at a time.

Connection, timeout, conflict, rate-limit, and server failures use bounded exponential-backoff retries from the OpenAI SDK. If retries run out, the interactive loop offers to retry the response. If the selected model is missing, Loop discovers available models and lets the user choose a replacement. Python callers receive normalized backend exceptions and can handle them directly.

## Web requests

The `fetch_content` tool sends a browser-like user agent by default. Set `[web].user_agent` in user or workspace configuration, or set `USER_AGENT`, to change it.
