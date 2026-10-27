# Development

Install the development dependencies:

```bash
uv sync
```

Run the test suite and linters:

```bash
make test
make test-integration
make check
```

`make test` runs the portable unit suite with 100% statement and branch coverage. Integration and end-to-end checks live separately under `tests/integration/` and skip unsupported platforms. `make test-all` runs both sets. See [test organization, execution and isolation](../tests/README.md).

`make check` runs Ruff lint and formatting checks, checks diff whitespace, and runs Pylint on the source. Run `make format` to apply Ruff formatting.

## macOS native integration

Run `tests/integration/macos/` from a terminal whose parent process permits Seatbelt initialization. Public native-search checks also need a trusted ripgrep installation on `PATH` before pytest starts; verified Homebrew installations are supported. Missing prerequisites produce skips, which do not verify native enforcement. Separate cases exercise the Python search fallback.

## Linux native integration

Provision the Linux sandbox explicitly through the main app before running installed execution checks. Ordinary integration tests do not install, refresh, or remove it. Use an installation matching the package being tested; incompatible installations refuse execution until refreshed. Without a working installation, the suite checks unavailable behavior and skips enforcement cases.

Put the test virtual environment and `uv` on `PATH`, with pytest, pytest-cov and Ruff installed, to exercise the development-tool workflows. Some checks also require explicitly provisioned fake runtime data or a directory on the sandbox listener’s filesystem. See [native Linux test setup](../tests/README.md#native-linux-test-setup) for the environment variables and fixture requirements.

Capacity and management lifecycle checks are opt-in and require a disposable Linux host. The capacity check fills the policy cache and requires a refresh afterward. The lifecycle check installs and removes the sandbox, requires a real terminal with administrator authentication, and must run separately. Do not raise limits, broaden permissions, or omit assertions to make native checks pass.

## Source organization

Backends expose `prepare_context` so generation and file observations share the same prepared source text. The default preserves context items. Adapters that transform source text must preserve immutable source metadata and mark transformed reference or read-result payloads with `payload_redacted` and transformed tool arguments with `arguments_redacted`. These flags are transient; canonical session history remains unchanged. Interrupted-batch recovery prepares only the preceding generation context and replays completed own mutations using their original arguments.

The package uses a `src` layout. Its main components are:

```text
src/loop/agent/         Agent identity, instructions, and execution
src/loop/application/   Application composition, immutable storage paths, and commands
src/loop/backend/       Backend contract and OpenAI-compatible adapter
src/loop/commands/      User-command definitions and dispatch
src/loop/compaction/    Context compaction
src/loop/completion/    Declarative completion models, adapters, and aggregation
src/loop/configuration/ Configuration management, models, and commands
src/loop/control.py     Cooperative stop checkpoints and retry waits
src/loop/conversation/  Serialized execution owner, input mailbox, and UI dispatch
src/loop/constants.py   Shared application constants
src/loop/errors.py      Application-wide problem reporting
src/loop/interaction/   Input/output interfaces, console input, and response presentation
src/loop/loop.py        Interactive conversation loop and streaming configuration
src/loop/main.py        Application entry point and composition root
src/loop/mentions/      User-mention resolution
src/loop/model_selection/ Active model selection and model commands
src/loop/models.py      Conversation and response models
src/loop/permissions/   Permission capabilities, requests, and policy management
src/loop/response.py    UI-independent normalized response collection and validation
src/loop/session/       Session persistence contracts and implementations
src/loop/telemetry/     Telemetry and operational logging
src/loop/instructions/  Base agent policy, Agent Skills, catalog, and instruction management
src/loop/tooling/       Tool context, registration, definitions, and dispatch
src/loop/tools/         Built-in tool implementations
src/loop/utils/         Common utilities
src/loop/workspace/     Workspace discovery, identity, catalog, and commands
```

## Conversation ownership

The responsive runtime has one execution thread for conversation services and one asyncio UI owner. Acknowledged calls keep presentation on the UI thread, and a bounded mailbox applies backpressure to output producers. Conversation inputs share a bounded priority queue; accepted steering retains FIFO ahead of ordinary follow-ups; questions and approvals use their own result futures and never read that queue. A terminal scope suspends the ordinary editor before direct administrator authentication.

Completion providers share an ownership gate with execution and only inspect live state while the execution owner is waiting for input. Application shutdown closes the current mailbox, prevents new model and tool boundaries, revokes pending interactions, closes response streams, and drains owned work before resources are closed. Inputs already accepted by a durable session remain pending until consumed or rejected and are restored only after the session's recovery gate clears. Tool results that have already completed are persisted before the next safe run checkpoint. Native command supervisors retain their existing deadlines and descendant cleanup. Arbitrary synchronous tools must provide their own bounded execution; Python threads are not forcibly terminated.

Scoped interruption, pause, cold stop, and application shutdown have separate state transitions. An interrupt cancels the active run and records a cancelled result; pause parks the same owner at a safe checkpoint; cold stop settles the run and leaves an incomplete durable boundary; shutdown remains sticky and closes owned resources. Continue and retry are derived from session events, with retry enabled only after a recorded failure. The runner retains settled tool results and preserves unknown outcomes. Corrections use ordinary message preparation and telemetry correlation; the UI never mutates session history or interprets them as approvals.
