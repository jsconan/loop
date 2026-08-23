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
