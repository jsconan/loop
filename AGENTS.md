## Project-wide coding guidelines

### Design and implementation

- Understand the request and trace the affected code path and callers before choosing a solution. State material assumptions and tradeoffs; clarify ambiguity when it could change the implementation.
- Make the smallest complete change and preserve the existing architecture, supported behavior, and compatibility unless changing them is necessary.
- Follow KISS, DRY, and YAGNI: prefer no change when none is needed, then reuse an existing pattern, the standard library, or an installed dependency; add only what remains necessary. Consolidate genuine duplication; avoid boilerplate, single-use abstractions, speculative extensibility, and unrequested configuration. Prefer direct code and fewer files, and choose the edge-case-safe option when equally small alternatives exist.
- Before adding a helper, feature, or integration, search relevant source, tests, and documentation for an existing implementation or extension point. Reuse it when its semantics fit; explain a mismatch before creating a parallel path.
- For bug fixes, identify the root cause and inspect callers. Fix the shared cause once when appropriate rather than patching only the reported symptom.
- Keep changes surgical: match surrounding style, avoid unrelated refactors or formatting, remove only code made unused by the current change, and mention unrelated dead code rather than deleting it.
- Apply least privilege across components and scripts: limit data, dependencies, credentials, permissions, and authority to what each task needs; enforce authorization at the boundary that owns the protected operation.
- Do not introduce regressions or weaken trust-boundary validation, authorization, security controls, accessibility, or data-loss protections.
- Loop is a final application and its APIs currently have no external consumers. For APIs used only within Loop, update the full internal call chain together; do not retain old signatures, aliases, adapters, or branches for hypothetical compatibility. Preserve compatibility for persisted data and configuration, provider or native interfaces, and explicitly supported APIs.
- When deliberately accepting a known limitation, document its current product behavior, practical ceiling, and the condition for replacing it.

### Verification

- Fix unexpected warnings and errors at their cause. Do not suppress them, weaken assertions, or change warning/error filters to make checks pass. Capture and assert expected diagnostics specifically.
- Define concrete success criteria before implementation and verify them afterward. For multi-step work, use a brief plan with an observable check for each step.
- Architecture tests cover repository-wide static contracts, unit tests cover portable module behavior, and integration tests cover native or dependency boundaries. Run affected architecture and unit checks for code changes, plus applicable integration checks; use the testing skill for suite commands and prerequisites.
- A bug fix needs a check that reproduces the failure. A refactor should preserve passing checks before and after. Non-trivial new logic should have the smallest check that would fail if it broke; trivial one-line changes do not require a new test.

### Documentation

- Keep maintained documentation, including the main and test READMEs, focused on supported behavior, usage, configuration, architecture, test organization, and durable constraints. Update it when those contracts change, not after an individual development or verification run.
- Do not put review findings, progress notes, migration history, dated verification results, test counts, host-specific benchmark results, or rollout plans in maintained documentation. Report transient evidence in the task response or an explicitly requested temporary artifact.
- Describe limitations as product behavior. Do not replace concrete safety limits with roadmap promises, historical qualification claims, or assertions about future support.
- Markdown has no line-length limit. Keep each prose paragraph on one source line without wrapping; preserve structural newlines in headings, lists, tables, and code fences.
- Treat documentation as part of implementation. Update affected documentation in the same change when a signature or behavior changes, and use the coding and testing skills for detailed docstring conventions.

## Skills

- Use the coding skill when adding, modifying, refactoring, fixing, reviewing, or diagnosing code under src/. For Python entry points, scripts, and benchmarks outside src/, apply the skill's general design, style, and documentation rules when relevant; src-specific guidance remains scoped to src/. It owns detailed Python design, observability, and validation conventions, including the Law of Demeter, separation of concerns, and composition or inheritance guidance.
- Use the testing skill when adding, changing, reorganizing, or diagnosing tests, or when verifying behavior, fixtures, mocks, and error paths. It owns test-specific organization, isolation, docstrings, and commands.
- Keep detailed implementation and test conventions in their owning skills; keep this file focused on cross-project rules and constraints.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:

- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
