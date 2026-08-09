"""Define passive path utility models."""

from pathlib import Path

from pathspec import GitIgnoreSpec

type IgnoreRule = tuple[Path, GitIgnoreSpec]
type IgnoreRules = dict[str, list[IgnoreRule]]
