"""Verify the permission-owned protected workspace path specification."""

from pathlib import Path

import pytest

from loop.permissions.protection import protected_workspace_paths


def test_protected_paths_bind_default_and_configured_names():
    """Native policy and structured permissions share active control path names."""
    paths = protected_workspace_paths(("POLICY+.md", "AGENTS.md"))

    assert paths.instruction_names == ("AGENTS.md", "SKILL.md", "POLICY+.md")
    assert paths.directories == (".git", ".loop", ".agents/skills")
    assert paths.files == (".gitignore", ".agentignore")


@pytest.mark.parametrize(
    ("relative", "protected"),
    [
        (".agents/skills/demo/asset.txt", True),
        ("nested/.agents/skills/demo/asset.txt", True),
        ("nested/.git/config", True),
        ("nested/.loop/policy", True),
        ("nested/.agents/skills-other/asset.txt", False),
        ("nested/project.git/config", False),
    ],
)
def test_protected_directories_apply_at_nested_instruction_scopes(relative, protected):
    """Nested skill and control roots receive the same classification as root scopes."""
    paths = protected_workspace_paths()

    assert paths.protects_directory(Path(relative)) is protected


@pytest.mark.parametrize("name", ("", ".", "..", "a/b", "a\\b", "a\x00b"))
def test_protected_paths_reject_non_basename_instructions(name):
    """An instruction configuration cannot inject a path or native policy syntax."""
    with pytest.raises(ValueError, match="plain filenames"):
        protected_workspace_paths((name,))
