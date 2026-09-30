"""Verify the suite rejects external resources and production filesystem mutations."""

import os
import socket
import sqlite3
import subprocess
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
