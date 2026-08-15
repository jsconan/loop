"""Tests for permission model behavior."""

from loop import Capability


def test_capabilities_map_to_distinct_prompt_icons():
    """Every capability exposes its assigned permission-prompt icon."""
    assert {capability: capability.icon for capability in Capability} == {
        Capability.PURE: "🧠",
        Capability.FILESYSTEM_READ: "📖",
        Capability.FILESYSTEM_WRITE: "✏️",
        Capability.FILESYSTEM_DELETE: "🗑️",
        Capability.PROCESS_EXEC: "⚙️",
        Capability.NETWORK_READ: "🌐",
        Capability.NETWORK_WRITE: "📡",
        Capability.SESSION_WRITE: "💾",
    }
