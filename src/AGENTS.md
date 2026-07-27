# Source Documentation Guidelines

These instructions apply to all source files under `src/`.

## Documentation

- Add docstrings to all public modules, classes, functions, methods, and properties.
- Start each docstring with a concise, imperative summary of the documented behavior.
- Use Google-style `Args:`, `Returns:`, `Yields:`, and `Raises:` sections when applicable. Omit
  sections that would only repeat information already clear from the signature and summary.
- Do not add a `Returns:` section when a callable does not return a value.
- Do not add docstrings to dunder methods. Their documentation does not surface usefully.
- For constructor documentation, document initialization behavior and arguments in the owning
  class docstring instead of adding a docstring to `__init__`.
- Keep private function and method docstrings to a concise description. Add argument, return, or
  error sections only when the behavior is complex enough to need the additional explanation.
- Keep registered tool function docstrings to a concise description. The complete docstring is
  exposed as the tool description, while Pydantic field descriptions document its arguments.
