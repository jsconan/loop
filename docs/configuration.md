# Configuration

## Config files and scope

The CLI stores a complete, commented user configuration in the platform-native Loop configuration directory. An optional `.loop/config.toml` contains workspace-specific overrides. Both files are private to your account; do not commit secrets from the local override.

Settings resolve in this order, from lowest to highest priority: built-in defaults, user configuration, workspace overrides, environment variables, and session overrides. Configuration commands use user scope by default; pass `scope=workspace` or `scope=session` to save a narrower override.

Loop uses one active workspace per process. Sessions and permission policies belong to that workspace. Telemetry, the audit database, and the application log are shared across workspaces.

Set `LOOP_CONFIG_HOME`, `LOOP_DATA_HOME`, or `LOOP_STATE_HOME` to override the corresponding platform-native directory. Run `/app dirs` to see the resolved locations.

Use these commands to manage workspace identities and locations:

- `/workspace attach <path> [workspace-id]` registers a location or associates it with an existing workspace identity.
- `/workspace forget <id-or-path>` deactivates a location mapping without deleting the workspace's stored data.
- `/workspace rekey <path>` gives a copied workspace a fresh identity while retaining the original identity and data.
- `/workspace switch <id-or-path>` switches the active workspace and rebuilds workspace-scoped services. If rebuilding fails, Loop restores the previous workspace.

Forget and rekey require confirmation. Loop imports legacy sessions, policies, logs, and audit records into the new stores without deleting the original files. Interrupted imports can be retried safely.

## See all settings

The generated file contains every setting and its built-in value. The example below shows commonly used settings. Settings that can be unset, such as `context_window`, `file_input_mode`, `model`, `temperature`, and `reasoning_effort`, appear as commented `<unset>` entries because TOML has no null value.

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

[loop]
agent_name = "Loop"
# temperature = <unset>
# reasoning_effort = <unset>
stream = true
debug = false
compaction_threshold = 0.8
prompt_on_recoverable_error = true
max_agent_turns = 25

# Web requests use a browser-like user agent by default.
# See "Web requests" below to override it.

[tools]
command_timeout = 60.0

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

Loop validates settings at startup. An invalid value or unknown key stops startup without exposing secret values in logs. Changes made through `/config` preserve comments and formatting in the configuration file.

Use `/config` to browse, inspect, change, or reset effective settings. Completion suggests setting names and values:

```text
/config
/config get loop.temperature
/config set loop.temperature 0.7 scope=session
/config set loop.stream false scope=workspace
/config secret backend.api_key
/config reset loop.temperature scope=workspace
/config reset  # prompts for scope and confirmation
```

Reset behavior depends on the scope. At workspace scope, reset removes the override so the user setting applies. At user or session scope, reset restores the built-in default.

Session settings apply only to the current Loop process and take precedence over environment and file values. A file value remains saved when an environment variable overrides it, so the saved value applies again when the variable is absent. Backend and supported interactive-loop settings apply immediately. Logging, telemetry, and agent identity changes require a restart. API-key input and configuration output are masked.

While an agent run is active, `conversation.input_mode = "queue"` admits plain text as a follow-up that runs after the current operation settles. Set it to `"steer"` to admit plain text at steering priority and stop the run at its next safe boundary. Explicit `/queue` and `/steer` commands override this setting.

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

When using the official OpenAI endpoint, referenced text files use native `input_file` parts by default. A custom `base_url` defaults to portable `input_text` parts because many compatible servers do not support `input_file`. Set `file_input_mode="text"` or `file_input_mode="native"` to choose explicitly.

Project-path mentions use immutable content snapshots so resumed conversations see the same file contents. Loop includes at most two small files directly. Larger files, additional files, and directory contents use bounded previews and `read_cached_content` handles for the remainder. Loop stores snapshot bytes once and loads them into model context when needed.

## Environment variables

The `loop` command accepts environment overrides for automation:

| Setting                     | Environment variable                    | Built-in default               |
| --------------------------- | --------------------------------------- | ------------------------------ |
| Base URL                    | `BASE_URL`                              | `http://localhost:8000/v1`     |
| Backend model               | `DEFAULT_MODEL`                         | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| API key                     | `OPENAI_API_KEY`                        | `local-api-key`                |
| Context window              | `CONTEXT_WINDOW`                        | unset                          |
| Automatic retries           | `OPENAI_MAX_RETRIES`                    | `2`                            |
| Temperature                 | `LOOP_TEMPERATURE`                      | unset                          |
| Reasoning effort            | `LOOP_REASONING_EFFORT`                 | unset                          |
| Hyperparameter policy       | `OPENAI_HYPERPARAMETER_POLICY`          | `fallback`                     |
| Web user agent              | `USER_AGENT`                            | browser-like user agent        |
| Command timeout             | `LOOP_COMMAND_TIMEOUT`                  | `60.0`                         |
| Loop agent name             | `LOOP_AGENT_NAME`                       | `Loop`                         |
| Active input mode           | `LOOP_CONVERSATION_INPUT_MODE`          | `queue`                        |
| Loop model                  | `LOOP_MODEL`                            | unset                          |
| Stream responses            | `LOOP_STREAM`                           | `true`                         |
| Debug responses             | `LOOP_DEBUG`                            | `false`                        |
| Compaction threshold        | `LOOP_COMPACTION_THRESHOLD`             | `0.8`                          |
| Prompt on recoverable error | `LOOP_PROMPT_ON_RECOVERABLE_ERROR`      | `true`                         |
| Maximum agent turns         | `LOOP_MAX_AGENT_TURNS`                  | `25`                           |
| Log level                   | `LOOP_LOG_LEVEL`                        | `INFO`                         |
| Log size                    | `LOOP_LOG_MAX_BYTES`                    | `5242880`                      |
| Log backups                 | `LOOP_LOG_BACKUP_COUNT`                 | `3`                            |
| Telemetry queue capacity    | `LOOP_TELEMETRY_QUEUE_CAPACITY`         | `4096`                         |
| Telemetry batch size        | `LOOP_TELEMETRY_BATCH_SIZE`             | `128`                          |
| Telemetry flush interval    | `LOOP_TELEMETRY_FLUSH_SECONDS`          | `0.1`                          |
| Telemetry shutdown timeout  | `LOOP_TELEMETRY_SHUTDOWN_TIMEOUT`       | `2.0`                          |
| SQLite busy timeout         | `LOOP_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS` | `2500`                         |

`OpenAIBackend` does not read environment variables or apply deployment defaults. Python callers configure it directly; credentials remain private to the backend.

## Hyperparameters and retries

`temperature` and `reasoning_effort` are optional per-agent settings, so agents can share a backend with different controls. With `hyperparameter_policy = "fallback"`, Loop retries a request without a parameter if the selected model rejects it. Use `"strict"` to keep the provider error instead. Loop also checks compatible-provider model metadata when available and omits controls declared unsupported. Set either `temperature` or `top_p`; the OpenAI API recommends changing one sampling control at a time.

Connection, timeout, conflict, rate-limit, and server failures use bounded exponential-backoff retries from the OpenAI SDK. If retries run out, the interactive loop offers to retry the response. If the selected model is missing, Loop discovers available models and lets the user choose a replacement. Python callers receive normalized backend exceptions and can handle them directly.

## Web requests

The `fetch_content` tool sends a browser-like user agent by default. Set `[web].user_agent` in user or workspace configuration, or set `USER_AGENT`, to change it.
