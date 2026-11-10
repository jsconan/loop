"""Verify generic sandbox management registration follows the selected backend."""

import platform

from loop.execution.sandbox import SandboxCommands


def test_unsupported_host_registers_no_native_management(monkeypatch):
    """An unsupported host exposes no native management commands."""
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    assert SandboxCommands().get_commands() == ()


def test_macos_backend_registers_no_installation_management(monkeypatch):
    """The unprivileged macOS backend exposes no host installation command."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    assert SandboxCommands().get_commands() == ()
