---
name: coding
description: Implement, modify, refactor, fix, and review Python code under src/ while preserving the repository's architecture, public behavior, and documentation conventions. Use for any source-code change in src/, including modules, classes, functions, methods, properties, private helpers, and registered tools.
---

# Coding

Trace the affected source path and its callers before choosing a solution. Implement the smallest
change that fully satisfies the request, preserve surrounding architecture and style, and avoid
unrelated refactoring or new dependencies.

## Implement the behavior

- Reuse existing project patterns, standard-library features, and installed dependencies before
  adding new abstractions or packages.
- Before adding a private helper, consider whether its semantics are generic and reusable across
  modules. When they are, prefer a focused common helper in `src/loop/utils/`; keep behavior that
  is specific to one implementation private to that module or class.
- Keep domain knowledge behind one cohesive boundary. Let callers express intent without
  constructing another component's internal models, coordinating its lifecycle or rollback, or
  duplicating its parsing, validation, and completion rules.
- Do not add thin convenience methods that merely accept and forward an already-constructed
  internal object. Give the component that owns construction and invariant enforcement the raw
  inputs it needs, so callers do not need to know its internal representation or workflow.
- When a new domain is complex enough to require multiple responsibilities or implementation
  modules, group it in a package with one deliberate public facade. Keep parsing, models, and
  orchestration private unless callers independently need those APIs.
- Fix shared root causes when appropriate instead of patching only a reported symptom.
- Preserve validation, security, accessibility, data-loss prevention, and explicitly required
  behavior.
- Declare class attributes at class level, including attributes whose values are initialized in
  `__init__`; keep their value initialization in `__init__`.
- Do not use `assert` outside unit tests. For runtime validation and error conditions, raise the
  most accurate exception type with a clear, specific message.
- Do not use local or otherwise lazy imports to work around cyclic dependencies. A local import
  is still a cyclic dependency and is forbidden for that purpose; resolve the underlying design
  or module-boundary problem instead. Local or lazy imports are allowed only when they prevent
  systematically loading a genuinely heavy resource or loading an optional resource. Imports
  used only for annotations may be placed under `if TYPE_CHECKING:`.
- Remove only imports or code made unused by the current change.
- State material assumptions, tradeoffs, and deliberately accepted limitations.

## Document the public surface

- Add docstrings to every public module, class, function, method, and property.
- Start each docstring with a concise, imperative summary.
- Use complete Google-style sections for the final signature and behavior:
  - `Args:` documents every parameter except `self` and `cls` as `name (type): description`, even
    when the signature has a type annotation. Include the meaning of defaults when it is not
    obvious. In a class docstring, document every `__init__` parameter there.
  - `Returns:` documents every non-`None` return value as `type: description`, even when the
    signature has a return annotation. For a tuple, document the composite tuple type and the
    meaning of each member. Omit it only when every normal path returns `None`.
  - `Yields:` replaces `Returns:` for an iterator and documents each yielded value as
    `type: description`.
  - `Raises:` documents each exception that the callable deliberately raises or exposes as part of
    its contract. Do not list incidental implementation exceptions.
- Do not omit an applicable entry or its type merely because the signature or annotation provides
  the same information.
- Do not add docstrings to dunder methods.
- Document initialization behavior and arguments in the class docstring, not `__init__`.
- Keep private function and method docstrings concise. Add argument, return, or error sections only
  when the behavior is complex enough to require them; when adding a section, complete it using the
  same rules as public docstrings.
- Treat registered tool functions as the deliberate exception: keep their docstrings to the
  imperative summary because the full docstring becomes the tool description. Argument, return,
  yield, and type documentation requirements do not apply to tools; put argument details in
  Pydantic field descriptions instead of an `Args:` section.

After implementing, compare each added or changed callable's docstring with its final signature,
return and yield paths, and explicit exceptions. Update stale documentation on the affected public
surface without rewriting unrelated docstrings.

## Preserve observability

Preserve observability when changing operational activity, failures, auditable decisions or state
changes, and execution boundaries.

- Instrument through the process-wide `loop.telemetry` facade and use `telemetry_span` for operation
  boundaries. Preserve existing correlation context; do not instantiate adapters or write storage
  from call sites.
- Report operational failures once, at the boundary that owns or handles them, through
  `loop.errors.log_problem`. Keep this path independent of telemetry, isolate telemetry failures
  from application behavior, and do not report expected control flow as errors.
- Keep activity, error, and audit records sanitized and free of execution content or credentials.
  Record full execution data only in traces at the owning boundary. At model boundaries, apply
  `ModelInputPolicy` before transmission and trace the exact prepared input, excluding transport
  credentials.
- Audit security-relevant decisions and durable state changes. When observability behavior changes,
  test the affected disclosure boundary, correlation, failure isolation, and outcomes.

## Verify the change

Define an observable success criterion. Include docstring completeness in the check whenever a
public signature or behavior changes. Run the narrowest relevant checks first, then the broader
project checks warranted by the risk. Use the `testing` skill when test code must be added or
changed.

After changing code, run:

```shell
.venv/bin/ruff format src
.venv/bin/ruff check src
git diff --check
```

Address any linter or formatting issues before committing.
