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

## Verify the change

Define an observable success criterion. Include docstring completeness in the check whenever a
public signature or behavior changes. Run the narrowest relevant checks first, then the broader
project checks warranted by the risk. Use the `testing` skill when test code must be added or
changed.
