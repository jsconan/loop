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
- Fix shared root causes when appropriate instead of patching only a reported symptom.
- Preserve validation, security, accessibility, data-loss prevention, and explicitly required
  behavior.
- Remove only imports or code made unused by the current change.
- State material assumptions, tradeoffs, and deliberately accepted limitations.

## Document the public surface

- Add docstrings to every public module, class, function, method, and property.
- Start each docstring with a concise, imperative summary.
- Use Google-style `Args:`, `Returns:`, `Yields:`, and `Raises:` sections when applicable.
- Omit sections that only repeat the signature, annotations, or summary. Omit `Returns:` when a
  callable does not return a value.
- Do not add docstrings to dunder methods.
- Document initialization behavior and arguments in the class docstring, not `__init__`.
- Keep private function and method docstrings concise. Add argument, return, or error sections only
  when the behavior is complex enough to require them.
- Keep registered tool function docstrings concise because the full docstring becomes the tool
  description. Put argument details in Pydantic field descriptions.

Review and update stale documentation on the affected public surface without rewriting unrelated
docstrings.

## Verify the change

Define an observable success criterion. Run the narrowest relevant checks first, then the broader
project checks warranted by the risk. Use the `unit-testing` skill when test code must be added or
changed.
