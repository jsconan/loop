"""Tests for the in-sandbox resource-limit launcher."""

from unittest.mock import MagicMock

import pytest

from loop.sandbox import launcher


def test_launcher_lowers_soft_limits_before_hard_limits(monkeypatch):
    """Limits become irrevocable only after lowering a currently higher soft limit."""
    monkeypatch.setattr(launcher, "_LIMITS", ((1, 10), (2, 5), (3, 5)))
    monkeypatch.setattr(launcher.resource, "RLIM_INFINITY", -1)
    monkeypatch.setattr(
        launcher.resource,
        "getrlimit",
        MagicMock(side_effect=[(100, 100), (0, 10), (0, 5)]),
    )
    setrlimit = MagicMock()
    monkeypatch.setattr(launcher.resource, "setrlimit", setrlimit)

    launcher._apply_resource_limits()  # pylint: disable=protected-access

    assert setrlimit.call_args_list == [
        ((1, (10, 100)),),
        ((1, (10, 10)),),
        ((2, (5, 5)),),
    ]


def test_launcher_continues_when_macos_rejects_a_resource_limit(monkeypatch):
    """Unsupported Darwin resource ceilings cannot prevent filesystem sandbox execution."""
    monkeypatch.setattr(launcher, "_LIMITS", ((1, 10),))
    monkeypatch.setattr(launcher.resource, "RLIM_INFINITY", -1)
    monkeypatch.setattr(launcher.resource, "getrlimit", lambda _kind: (100, 100))
    monkeypatch.setattr(launcher.resource, "setrlimit", MagicMock(side_effect=ValueError("no")))
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")

    launcher._apply_resource_limits()  # pylint: disable=protected-access


def test_launcher_fails_closed_when_a_non_macos_limit_cannot_be_applied(monkeypatch):
    """A Linux resource-limit failure stops launch before the approved command executes."""
    monkeypatch.setattr(launcher, "_LIMITS", ((1, 10),))
    monkeypatch.setattr(launcher.resource, "RLIM_INFINITY", -1)
    monkeypatch.setattr(launcher.resource, "getrlimit", lambda _kind: (100, 100))
    monkeypatch.setattr(launcher.resource, "setrlimit", MagicMock(side_effect=ValueError("no")))
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")

    with pytest.raises(ValueError, match="no"):
        launcher._apply_resource_limits()  # pylint: disable=protected-access


@pytest.mark.parametrize(
    ("argv", "message"),
    [(["launcher"], "separator"), (["launcher", "--"], "executable")],
)
def test_launcher_rejects_missing_command_arguments(monkeypatch, argv, message):
    """The launcher never starts without a complete approved command vector."""
    monkeypatch.setattr(launcher.sys, "argv", argv)

    with pytest.raises(ValueError, match=message):
        launcher.main()


def test_launcher_applies_limits_then_executes_the_exact_command(monkeypatch):
    """The launcher replaces itself only after establishing limits for the approved argv."""
    applied = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(launcher.sys, "argv", ["launcher", "--", "/tool", "argument"])
    monkeypatch.setattr(launcher, "_apply_resource_limits", applied)
    monkeypatch.setattr(launcher.os, "execvpe", execute)

    launcher.main()

    applied.assert_called_once_with()
    execute.assert_called_once_with("/tool", ["/tool", "argument"], launcher.os.environ)
