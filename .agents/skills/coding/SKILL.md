---
name: coding
description: Implement, modify, refactor, fix, review, diagnose, or analyze Python source under src/, preserving the repository's architecture, public behavior, and documentation conventions. Activate for any source-code change in src/ (edits, refactors, additions, removals) and also for reviews or diagnosis of existing src/ code when the task only inspects or analyzes modules, classes, functions, methods, properties, private helpers, or registered tools, rather than changing them.
---

# Coding

Trace the affected source path and its callers before choosing a solution. Implement the smallest change that fully satisfies the request, preserve surrounding architecture and style, and avoid unrelated refactoring or new dependencies.

## Implement the behavior

- Keep Python source lines within the configured 100-character limit, including comments and docstrings. Formatter success alone does not guarantee compliance; verify line lengths explicitly.
- Put function and method signatures and calls on one line when the complete line fits. Otherwise use a multiline layout with one parameter or argument per line, a trailing comma, and closing delimiters on their own line. Preserve valid syntax for generator expressions and multiline arguments.
- Prefer descriptive named constants for policy values, limits, timeouts, and repeated meaningful literals. Reuse existing constants with matching semantics; leave structural values inline when naming them adds no meaning, and do not introduce unnecessary configuration or abstractions. Keep constants in the closest appropriate shared scope: module-local values stay with their callers, and values shared within a feature stay in that feature. Place constants in the main application constants module only when they need central configuration; do not centralize endemic values merely because they are constants.
- Use CapWords for classes and short lowercase names for modules, adding underscores when they improve readability. Prefix intentionally internal classes, modules, functions, methods, and attributes with one leading underscore so their private status is clear; expose supported APIs through the deliberate package facade. Reserve double-leading underscores for avoiding name clashes with subclasses, except for required special names such as `__init__.py`.
- Follow PEP 8 for all Python source: preserve its naming, indentation, whitespace, blank-line, import-grouping, line-length, and module-ordering conventions. Treat Ruff's configured PEP 8 checks as required, and resolve violations rather than suppressing them without a documented, code-specific reason.
- Reuse existing project patterns, standard-library features, and installed dependencies before adding new abstractions or packages.
- Before implementing a helper, feature, or integration, search the relevant source, tests, and documentation for an existing implementation or extension point. Reuse it when it matches the required semantics; if it does not, record the mismatch and why a parallel implementation is necessary.
- Before adding a private helper, check whether its behavior is genuinely generic and already has multiple callers. When it does, prefer a focused common helper in `src/loop/utils/`; keep single-use or domain-specific behavior with its owner, and do not centralize it based on hypothetical reuse.
- Respect the Law of Demeter and separate concerns around cohesive boundaries. Keep domain knowledge behind the component that owns it; let callers express intent through that component's API instead of reaching through collaborators, constructing internal models, coordinating lifecycle or rollback, or duplicating parsing, validation, and completion rules. Separate responsibilities that change independently while keeping related policies and invariants together.
- Apply least privilege: give each component only the data, dependencies, and permissions required for its task and lifetime. Prefer explicit inputs and narrow interfaces to ambient state or broad service handles, scope credentials to operations that need them, and enforce authorization at the boundary that owns the protected operation.
- Choose composition or inheritance based on the relationship and ownership of the shared behavior. A base class or mixin is appropriate when a stable, cohesive contract or capability owns behavior shared by subclasses, including concrete methods and per-instance state; subclasses must be able to honor its documented contract. Prefer composition when dependencies or capabilities need independent configuration, replacement, or lifetime, or when no coherent base contract exists. Avoid mutable class-level state unless it is intentionally shared.
- Avoid wrapper methods that merely forward values without adding behavior, policy, or a useful API boundary. When callers would otherwise construct another component's private model or duplicate its invariants, let the owning component accept the raw intent or data and construct and validate its representation. Accept public domain or value objects when they are intentionally part of the component's API.
- When a new domain is complex enough to require multiple responsibilities or implementation modules, group it in a package with one deliberate public facade. Keep parsing, models, and orchestration private unless callers independently need those APIs.
- Fix shared root causes when appropriate instead of patching only a reported symptom.
- Preserve existing supported features and public behavior; do not introduce regressions. Preserve trust-boundary validation, authorization, security controls, accessibility, data-loss prevention, and behavior explicitly required by the request.
- Declare class attributes at class level, including attributes whose values are initialized in `__init__`; keep their value initialization in `__init__`.
- Do not use `assert` for runtime validation, authorization, or behavior required for production correctness because Python may remove assertions under optimization. Assertions may express internal developer invariants and may be used in all test suites. For runtime validation and error conditions, raise the most accurate exception type with a clear, specific message.
- Prefer top-level imports and resolve cycles by improving module boundaries instead of routinely hiding them with function-local imports; a local import defers the dependency but does not remove it. Use lazy imports when delayed loading is part of a clear optional or heavy-resource boundary, or document a justified case where restructuring would make ownership less clear. Imports used only for annotations may be placed under `if TYPE_CHECKING:`.
- In a new `__init__.py`, put required `__future__` imports after the module docstring, then declare module-level `__all__` before ordinary imports, following PEP 8's module-level dunder ordering.
- Remove only imports or code made unused by the current change.
- State material assumptions, tradeoffs, and deliberately accepted limitations.

## Document the public surface

Apply the project-wide documentation rules to maintained READMEs and reference documents: explain the final supported behavior and durable limitations, without review notes, development history, dated test results, benchmark observations, or transient plan references.

- Add docstrings to every public module, class, function, method, and property. Treat `__init__` as documented by its class docstring, which includes initialization behavior and arguments; do not duplicate that documentation on `__init__`.
- Start each docstring with a concise, imperative summary.
- Use complete Google-style sections for the final signature and behavior:
    - `Args:` documents every parameter except `self` and `cls` as `name (type): description`, even when the signature has a type annotation. Include the meaning of defaults when it is not obvious. In a class docstring, document every `__init__` parameter there; do not duplicate an `Args:` section on `__init__`.
    - `Returns:` documents every non-`None` return value as `type: description`, even when the signature has a return annotation. For a tuple, document the composite tuple type and the meaning of each member. Omit it only when every normal path returns `None`.
    - `Yields:` replaces `Returns:` for an iterator and documents each yielded value as `type: description`.
    - `Raises:` documents each exception that the callable deliberately raises or exposes as part of its contract. Do not list incidental implementation exceptions.
- Do not omit an applicable entry or its type merely because the signature or annotation provides the same information.
- Document public special-method behavior when it is non-obvious or part of the supported contract; do not add boilerplate docstrings when the standard protocol and class API fully explain the behavior.
- Document initialization behavior and arguments in the class docstring, not `__init__`.
- Keep private function and method docstrings concise. Add argument, return, or error sections only when the behavior is complex enough to require them; when adding a section, complete it using the same rules as public docstrings.
- Treat registered tool functions as the deliberate exception: keep their docstrings to the imperative summary because the full docstring becomes the tool description. Argument, return, yield, and type documentation requirements do not apply to tools; put argument details in Pydantic field descriptions instead of an `Args:` section.

After implementing, compare each added or changed callable's docstring with its final signature, return and yield paths, and explicit exceptions. Update stale documentation on the affected public surface without rewriting unrelated docstrings.

## Preserve observability

Preserve observability when changing operational activity, failures, auditable decisions or state changes, and execution boundaries.

- Instrument through the process-wide `loop.telemetry` facade and use `telemetry_span` for operation boundaries. Preserve existing correlation context; do not instantiate adapters or write storage from call sites.
- Report operational failures once, at the boundary that owns or handles them, through `loop.errors.log_problem`. Keep this path independent of telemetry, isolate telemetry failures from application behavior, and do not report expected control flow as errors.
- Keep activity, error, and audit records sanitized and free of execution content or credentials. Record full execution data only in traces at the owning boundary. At model boundaries, apply `ModelInputPolicy` before transmission and trace the exact prepared input, excluding transport credentials.
- Audit security-relevant decisions and durable state changes. When observability behavior changes, test the affected disclosure boundary, correlation, failure isolation, and outcomes.

## Verify the change

Define an observable success criterion. Include docstring completeness in the check whenever a public signature or behavior changes. Select architecture, unit and integration checks according to the affected boundaries described in `AGENTS.md` and `tests/README.md`; run the narrowest relevant checks first, then broader project checks warranted by the risk. Use the `testing` skill when test code must be added or changed.

After changing code, run:

```shell
uv --no-cache run ruff format src
uv --no-cache run ruff check src
git diff --check
```

Address any linter or formatting issues before committing.
