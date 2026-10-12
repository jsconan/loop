# Development

Install the development dependencies:

```bash
uv sync
```

Run the test suite and linters:

```bash
make test
make test-integration
uv run ruff check .
uv run ruff format --check .
git diff --check
uv run pylint src
```

`make test` runs the portable unit suite with 100% statement and branch coverage. Integration and end-to-end checks live separately under `tests/integration/` and skip unsupported platforms. `make test-all` runs both sets. See [test organization, execution and isolation](../tests/README.md).

## Linux native integration

Linux native checks require explicit host provisioning and do not install, refresh, or remove administrator authority. Without a qualified native installation, the integration suite exercises the unavailable path; skipped checks are not enforcement evidence.

The native test fixtures require:

- `LOOP_NATIVE_BWRAP` to name an explicitly selected, root-protected Bubblewrap qualification helper for `linux/test_namespace.py`.
- A current administrator-provisioned production installation for `linux/test_installed_execution.py`.
- `LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY` to name a private directory owned by the ordinary user, mode `0700`, on the listener's filesystem when running socket-alias cases.

The installed execution corpus uses immutable policy profiles. A generation permits at most 29 payload profiles, including operational selection probes. Run the corpus in bounded subsets, using a fresh reviewed installation and checked uninstall between subsets; do not increase the limit or drop cases to make the corpus pass. Native qualification procedures and evidence are maintained under `plans/native-command-sandbox/`.

The package uses a `src` layout. Its main components are:

```text
src/loop/agent/         Agent identity, instructions, and execution
src/loop/application/   Application composition, immutable storage paths, and commands
src/loop/backend/       Backend contract and OpenAI-compatible adapter
src/loop/commands/      User-command definitions and dispatch
src/loop/compaction/    Context compaction
src/loop/completion/    Declarative completion models, adapters, and aggregation
src/loop/configuration/ Configuration management, models, and commands
src/loop/constants.py   Shared application constants
src/loop/errors.py      Application-wide problem reporting
src/loop/interaction/   User interaction interfaces
src/loop/loop.py        Interactive conversation loop and streaming configuration
src/loop/main.py        Application entry point and composition root
src/loop/mentions/      User-mention resolution
src/loop/model_selection/ Active model selection and model commands
src/loop/models.py      Conversation and response models
src/loop/permissions/   Permission capabilities, requests, and policy management
src/loop/session/       Session persistence contracts and implementations
src/loop/telemetry/     Telemetry and operational logging
src/loop/instructions/  Base agent policy, Agent Skills, catalog, and instruction management
src/loop/tooling/       Tool context, registration, definitions, and dispatch
src/loop/tools/         Built-in tool implementations
src/loop/utils/         Common utilities
src/loop/workspace/     Workspace discovery, identity, catalog, and commands
```
