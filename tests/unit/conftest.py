"""Fail immediately when a unit test tries to use a real external boundary."""

import os
import socket
import subprocess

import pytest


def forbidden_resource(*args, **kwargs):
    """Reject real launches, signals, network connections and DNS in unit tests."""
    raise AssertionError("Unit tests must replace external resource boundaries with scoped doubles")


@pytest.fixture(autouse=True)
def deny_external_resources(monkeypatch, isolated_environment):
    """Require explicit doubles instead of consuming host processes or network services."""
    monkeypatch.setattr(subprocess, "Popen", forbidden_resource)
    for name in (
        "system",
        "fork",
        "forkpty",
        "posix_spawn",
        "posix_spawnp",
        "execv",
        "execve",
        "execl",
        "execlp",
        "execlpe",
        "execvp",
        "execvpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
    ):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden_resource)
    monkeypatch.setattr(os, "kill", forbidden_resource)
    if hasattr(os, "killpg"):
        monkeypatch.setattr(os, "killpg", forbidden_resource)
    monkeypatch.setattr(socket.socket, "connect", forbidden_resource)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_resource)
    monkeypatch.setattr(socket.socket, "bind", forbidden_resource)
    monkeypatch.setattr(socket.socket, "listen", forbidden_resource)
    for name in (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        monkeypatch.setattr(socket, name, forbidden_resource)
