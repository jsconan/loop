# Source Documentation Guidelines

These instructions apply to all source files under `src/`.

## Documentation

- Do not add docstrings to dunder methods. Their documentation does not surface usefully.
- For constructor documentation, document initialization behavior and arguments in the owning
  class docstring instead of adding a docstring to `__init__`.
