"""Tests for typed operation and policy model behavior."""

import pytest
from pydantic import ValidationError

from loop import (
    Action,
    Decision,
    FileTarget,
    NetworkTarget,
    Operation,
    PermissionConfiguration,
    ProcessTarget,
    SessionTarget,
)
from loop.permissions import PermissionPreset


def test_actions_map_to_distinct_prompt_icons():
    """Every authority-bearing action exposes its assigned prompt icon."""
    assert {action: action.icon for action in Action} == {
        Action.FILESYSTEM_LIST: "📂",
        Action.FILESYSTEM_READ: "📖",
        Action.FILESYSTEM_CREATE: "📝",
        Action.FILESYSTEM_REPLACE: "✏️",
        Action.FILESYSTEM_DELETE: "🗑️",
        Action.NETWORK_REQUEST: "🌐",
        Action.PROCESS_EXECUTE: "⚙️",
        Action.SESSION_MUTATE: "💾",
    }


def test_operations_expose_stable_resource_representations():
    """Typed targets derive canonical strings only for rule matching and display."""
    operations = (
        Operation(tool_id="file", action=Action.FILESYSTEM_READ, target=FileTarget(path="/x")),
        Operation(
            tool_id="web",
            action=Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com/a", origin="https://example.com"),
        ),
        Operation(
            tool_id="exec",
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(argv=("git", "status"), cwd="/x"),
        ),
        Operation(
            tool_id="skills",
            action=Action.SESSION_MUTATE,
            target=SessionTarget(identifier="activate:demo"),
        ),
    )

    assert tuple(operation.resource for operation in operations) == (
        "/x",
        "https://example.com/a",
        "git status",
        "activate:demo",
    )


def test_operations_require_the_target_type_associated_with_their_action():
    """An action cannot omit or disguise the concrete resource policy must evaluate."""
    with pytest.raises(ValidationError, match="requires a FileTarget"):
        Operation(
            tool_id="invalid",
            action=Action.FILESYSTEM_READ,
            target=SessionTarget(identifier="not-a-file"),
        )


def test_policy_configuration_has_a_versioned_complete_default():
    """The editable YAML schema exposes a version and every action fallback."""
    configuration = PermissionConfiguration()

    assert configuration.version == 1
    assert set(configuration.defaults) == set(Action)
    assert configuration.defaults[Action.FILESYSTEM_READ] is Decision.ALLOW


def test_permission_presets_reject_duplicate_rule_identifiers():
    """A preset rule set cannot contain ambiguous rule identifiers."""
    with pytest.raises(ValueError, match="rule identifiers must be unique"):
        PermissionPreset.model_validate(
            {
                "metadata": {
                    "id": "duplicate-rules",
                    "revision": "1",
                    "title": "Duplicate rules",
                    "description": "Invalid preset.",
                },
                "defaults": {action.value: Decision.DENY.value for action in Action},
                "rules": [
                    {"id": "same", "decision": "allow"},
                    {"id": "same", "decision": "deny"},
                ],
            }
        )


def test_builtin_permission_presets_are_sorted_and_have_stable_hashes():
    """Packaged presets load in catalog order and expose deterministic content hashes."""
    presets = PermissionPreset.builtin_presets()

    assert [preset.metadata.id for preset in presets] == [
        "locked",
        "observe",
        "supervised",
        "workspace",
    ]
    assert all(preset.content_hash.startswith("sha256:") for preset in presets)
    assert presets[0].content_hash == presets[0].model_copy(deep=True).content_hash


def test_builtin_permission_presets_reject_duplicate_packaged_ids(monkeypatch, tmp_path):
    """The packaged catalog rejects two artifacts claiming the same identifier."""
    defaults = "\n".join(f"  {action.value}: deny" for action in Action)
    payload = (
        "version: 1\nkind: loop.permission-preset\nmetadata:\n"
        "  id: duplicate\n  revision: '1'\n  title: Duplicate\n"
        "  description: Invalid\ndefaults:\n"
        f"{defaults}\nrules: []\n"
    )
    (tmp_path / "one.yaml").write_text(payload)
    (tmp_path / "two.yaml").write_text(payload)

    class ResourcePackage:
        """Expose the temporary preset directory as an importlib resource."""

        def joinpath(self, name):
            """Return the requested temporary resource directory."""
            assert name == "presets"
            return tmp_path

    monkeypatch.setattr("loop.permissions.models.files", lambda package: ResourcePackage())
    with pytest.raises(ValueError, match="Built-in permission preset identifiers"):
        PermissionPreset.builtin_presets()
