## Project-wide coding guidelines

### Design and implementation

- Understand the request and trace the affected code path before choosing a solution. State
  material assumptions and tradeoffs; if ambiguity would change the implementation, clarify it
  instead of silently guessing.
- Make the smallest change that fully satisfies the request. Preserve the existing architecture
  and behavior unless changing them is necessary.
- Follow KISS: prefer the simplest implementation that is clear, correct, and maintainable.
- Follow DRY: reuse existing abstractions and consolidate genuinely duplicated logic, without
  introducing abstractions for hypothetical future needs.
- Apply YAGNI. Before writing new code, prefer, in order: doing nothing when no change is needed,
  reusing an existing project pattern or helper, using the standard library or a native platform
  feature, and using an already-installed dependency. Only then add the minimum new code required.
- Avoid new dependencies when the existing codebase or standard library is sufficient. Avoid
  boilerplate, single-use abstractions, speculative extensibility, and unrequested configuration.
- Prefer boring, direct code and fewer files over cleverness. When equally small alternatives
  exist, choose the one that handles edge cases correctly.
- For bug fixes, identify the root cause and inspect callers of the code being changed. Fix the
  shared cause once when appropriate rather than patching only the reported symptom.
- Keep changes surgical: match the surrounding style, avoid unrelated refactors or formatting,
  and remove only imports or code made unused by the current change. Mention unrelated dead code
  rather than deleting it.
- Do not trade away trust-boundary validation, data-loss prevention, security, accessibility, or
  behavior explicitly required by the request in pursuit of a smaller diff.
- When deliberately accepting a known limitation, document its ceiling and the condition or path
  for replacing it.

### Verification

- Define concrete success criteria before implementation and verify them after the change. For
  multi-step work, use a brief plan whose steps each have an observable check.
- A bug fix should have a check that reproduces the failure; a refactor should preserve passing
  checks before and after. Non-trivial new logic should leave behind the smallest runnable test or
  assertion that would fail if it broke. Trivial one-line changes do not require a new test.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
