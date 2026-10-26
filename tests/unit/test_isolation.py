"""Verify the suite rejects external resources and production filesystem mutations."""

import ctypes
import os
import pwd
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.mark.parametrize("boundary", ["Popen", "system", "getaddrinfo"])
def test_external_boundary_is_blocked(boundary):
    """Unmocked process and DNS boundaries fail before consuming real resources."""
    # Resolve at call time so collection does not retain the original resource boundary.
    guarded = {
        "Popen": subprocess.Popen,
        "system": os.system,
        "getaddrinfo": socket.getaddrinfo,
    }[boundary]
    with pytest.raises(AssertionError, match="external resource"):
        guarded("unused")


def test_network_connection_is_blocked():
    """An unmocked socket connection fails before contacting another process."""
    with socket.socket() as connection, pytest.raises(AssertionError, match="external resource"):
        connection.connect(("127.0.0.1", 1))


def test_environment_and_current_directory_are_disposable(tmp_path):
    """Application roots and working-directory discovery are confined to fresh scratch."""
    root = tmp_path.parent
    assert Path.cwd().is_relative_to(root)
    for name in ("HOME", "LOOP_CONFIG_HOME", "LOOP_DATA_HOME", "LOOP_STATE_HOME", "TMPDIR"):
        assert Path(os.environ[name]).is_relative_to(root)
    assert "OPENAI_API_KEY" not in os.environ
    assert "SSH_AUTH_SOCK" not in os.environ


def test_unowned_output_is_blocked_before_creation(tmp_path):
    """Writes and directory creation outside test ownership leave no output behind."""
    target = tmp_path.parent.parent / f"loop-forbidden-{uuid4().hex}"
    with pytest.raises(AssertionError, match="escapes disposable scratch"):
        target.write_text("must not be written", encoding="utf-8")
    with pytest.raises(AssertionError, match="escapes disposable scratch"):
        target.mkdir()
    assert not target.exists()


def test_descriptor_relative_output_cannot_escape_scratch(tmp_path):
    """An outside read descriptor cannot authorize a relative filesystem mutation."""
    target = f"loop-forbidden-{uuid4().hex}"
    descriptor = os.open(tmp_path.parent.parent, os.O_RDONLY)
    try:
        with pytest.raises(AssertionError, match="escapes disposable scratch"):
            os.open(target, os.O_WRONLY | os.O_CREAT, dir_fd=descriptor)
        with pytest.raises(AssertionError, match="escapes disposable scratch"):
            os.mkdir(target, dir_fd=descriptor)
        assert not (tmp_path.parent.parent / target).exists()
    finally:
        os.close(descriptor)


def test_native_sqlite_output_is_blocked_before_creation(tmp_path):
    """The SQLite C boundary cannot create a database outside fixture ownership."""
    target = tmp_path.parent.parent / f"loop-forbidden-{uuid4().hex}.db"
    with pytest.raises(AssertionError, match="escapes disposable scratch"):
        sqlite3.connect(target)
    assert not target.exists()


@pytest.mark.parametrize(
    "event",
    [
        "subprocess.Popen",
        "os.system",
        "os.fork",
        "os.exec",
        "os.kill",
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "ctypes.dlopen",
    ],
)
def test_native_audit_boundary_cannot_bypass_scoped_doubles(event):
    """Retained native function aliases still encounter the unit resource guard."""
    with pytest.raises(AssertionError, match="external resource"):
        sys.audit(event)


def test_default_console_input_is_isolated_and_requires_an_explicit_response():
    """Default editors are independent and unconfigured input fails without a terminal."""
    from loop.interaction import ConsoleInteraction

    first = ConsoleInteraction()
    second = ConsoleInteraction()
    first.set_current_input("owned draft")
    assert first.current_input() == "owned draft"
    assert second.current_input() == ""
    with pytest.raises(AssertionError, match="external resource"):
        second.prompt()


def test_native_library_loading_is_blocked_before_dlopen():
    """Unconfigured ctypes loading cannot invoke host libraries in a portable unit test."""
    with pytest.raises(AssertionError, match="external resource"):
        ctypes.CDLL(None)


def test_account_lookup_uses_the_disposable_home():
    """Trusted-home discovery uses synthetic account data instead of host directory services."""
    account = pwd.getpwuid(os.getuid())
    assert account.pw_dir == os.environ["HOME"]
    assert account.pw_name == "test-user"
    with pytest.raises(AssertionError, match="external resource"):
        pwd.getpwnam("unused")


@pytest.mark.parametrize("repeat", [0, 1])
def test_permission_catalog_templates_never_share_mutable_defaults(repeat):
    """Consumers and repeated cases cannot mutate the validated permission-catalog template."""
    from loop.permissions import PermissionPreset

    presets = PermissionPreset.builtin_presets()
    expected = dict(presets[0].defaults)
    presets[0].defaults.clear()
    fresh = PermissionPreset.builtin_presets()
    assert fresh[0] is not presets[0]
    assert fresh[0].defaults == expected
