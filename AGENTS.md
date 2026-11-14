## Project-wide coding guidelines

### Design and implementation

- Understand the request and trace the affected code path before choosing a solution. State material assumptions and tradeoffs; if ambiguity would change the implementation, clarify it instead of silently guessing.
- Make the smallest change that fully satisfies the request. Preserve the existing architecture and behavior unless changing them is necessary.
- Follow KISS: prefer the simplest implementation that is clear, correct, and maintainable.
- Follow DRY: reuse existing abstractions and consolidate genuinely duplicated logic, without introducing abstractions for hypothetical future needs.
- Apply YAGNI. Before writing new code, prefer, in order: doing nothing when no change is needed, reusing an existing project pattern or helper, using the standard library or a native platform feature, and using an already-installed dependency. Only then add the minimum new code required.
- Before implementing a helper, feature, or integration, search the relevant source, tests, and documentation for an existing implementation or extension point. Reuse it when it matches the required semantics; if it does not, explain the mismatch before adding a parallel path.
- Avoid new dependencies when the existing codebase or standard library is sufficient. Avoid boilerplate, single-use abstractions, speculative extensibility, and unrequested configuration.
- Prefer boring, direct code and fewer files over cleverness. When equally small alternatives exist, choose the one that handles edge cases correctly.
- Declare instance attributes, including their type hints, at class level; initialize their values in `__init__`. Do not combine declaration and initialization in the constructor body.
- Follow least privilege: give each component and operation only the data, dependencies, and permissions needed for its task and lifetime. Prefer narrow interfaces and explicit inputs over ambient state or broad handles, and do not pass credentials or sensitive data to components that do not need them. Check authorization at the owning boundary before protected work.
- Respect the Law of Demeter: have callers express intent through the owning component's cohesive API. Avoid reaching through an object's collaborators, exposing their internals, or making callers construct internal models or coordinate another component's lifecycle and invariants.
- Choose composition or inheritance based on the relationship and ownership of the shared behavior. A base class or mixin is appropriate when a stable, cohesive contract or capability owns behavior shared by subclasses, including concrete methods and per-instance state; subclasses must be able to honor its documented contract. Prefer composition when dependencies or capabilities need independent configuration, replacement, or lifetime, or when no coherent base contract exists. Avoid mutable class-level state unless it is intentionally shared.
- Separate concerns around cohesive ownership: keep each component responsible for a coherent set of policies and invariants, and separate responsibilities that change independently. Keep closely related behavior and invariants together rather than splitting them across layers for form alone.
- For bug fixes, identify the root cause and inspect callers of the code being changed. Fix the shared cause once when appropriate rather than patching only the reported symptom.
- Keep changes surgical: match the surrounding style, avoid unrelated refactors or formatting, and remove only imports or code made unused by the current change. Mention unrelated dead code rather than deleting it.
- Do not introduce regressions in existing supported features or public behavior. Do not weaken or remove trust-boundary validation, authorization, security controls, accessibility, or data-loss prevention to reduce a diff or simplify an implementation.
- Loop is a final application and its APIs currently have no external consumers. For APIs used only within Loop, update the full internal call chain together; do not retain old signatures, aliases, adapters, or branches solely for hypothetical backward compatibility. Preserve compatibility when required by existing user data or configuration, persisted formats, provider or native interfaces, or an explicitly supported external consumer.
- When deliberately accepting a known limitation, document its ceiling and the condition or path for replacing it.

### Verification

- Fix the root cause of unexpected warnings and errors. Do not suppress them, weaken assertions, or change warning/error filters to make checks pass. When a diagnostic is an intended test outcome, capture and assert its specific category, message, or structured result in that test.
- Define concrete success criteria before implementation and verify them after the change. For multi-step work, use a brief plan whose steps each have an observable check.
- Select checks by boundary: architecture tests cover repository-wide static contracts, unit tests cover portable module behavior, and integration tests cover native or dependency boundaries. Run affected architecture and unit checks for code changes, plus applicable integration checks when those boundaries are involved; use `tests/README.md` for suite commands and prerequisites.
- A bug fix should have a check that reproduces the failure; a refactor should preserve passing checks before and after. Non-trivial new logic should leave behind the smallest runnable test or assertion that would fail if it broke. Trivial one-line changes do not require a new test.

### Documentation

- Markdown has no line-length limit. Keep each prose paragraph on one source line; do not insert line breaks to wrap paragraphs. Preserve structural newlines in headings, lists, tables, and fenced code blocks.
- Keep maintained documentation, including the main and test READMEs, focused on supported behavior, usage, configuration, architecture, test organization, and durable constraints. Update it when those contracts change, not after an individual development or verification run.
- Do not put review findings, progress notes, migration history, dated verification results, test counts, host-specific benchmark results, or rollout plans in maintained documentation. Report transient evidence in the task response or an explicitly requested temporary artifact.
- Describe limitations as product behavior. Do not replace concrete safety limits with roadmap promises, historical qualification claims, or assertions about future support.
- Treat documentation as part of the implementation. When a signature or behavior changes, update the affected docstring in the same change and compare it against the final implementation.
- Add a complete docstring to every public function and method. Treat `__init__` as documented by its class docstring, which includes initialization behavior and arguments; do not duplicate that documentation on `__init__`. Document other public special-method behavior when it is non-obvious or part of the supported contract. Keep private functions and methods documented with a concise summary, adding detail only when their behavior is non-obvious.
- Complete every applicable docstring section: document every argument with its type and purpose, every returned or yielded value with its type and purpose, and every intentionally raised exception. Do not leave sections partial merely because annotations or the signature provide some of the same information.
- Exempt registered tool docstrings from argument, return, yield, and type documentation; keep them concise for use as tool descriptions and document their arguments through Pydantic fields instead.
- Follow the detailed source and test docstring conventions in the required `coding` and `testing` skills.

## Skills

- Use the `coding` skill when adding, modifying, refactoring, fixing, or reviewing code under `src/`.
- Use the `testing` skill when adding, changing, reorganizing, or diagnosing tests under `tests/`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:

- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
