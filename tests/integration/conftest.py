"""Select integration checks only on supported platforms and isolate native helpers."""

import importlib
import platform
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def supported_platform(request):
    """Skip native checks before setup when their operating system is unsupported."""
    system = platform.system()
    if request.node.get_closest_marker("macos") is not None and system != "Darwin":
        pytest.skip("requires macOS")
    if request.node.get_closest_marker("linux") is not None and system != "Linux":
        pytest.skip("requires Linux")
    if system not in {"Darwin", "Linux"}:
        pytest.skip("native integration checks support macOS and Linux")


@pytest.fixture
def process_boundary(monkeypatch):
    """Compose real supervisors with deterministic process, clock and reader boundaries."""
    process = Mock(pid=123, returncode=-9)
    process.stdout = Mock()
    process.stderr = Mock()
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    process.stdout.fileno.return_value = 101
    process.stderr.fileno.return_value = 102
    process.wait.return_value = -9
    threads = []

    def thread_factory(*, target, daemon):
        """Run pipe draining synchronously without scheduling or elapsed-time waits."""
        thread = Mock()
        thread.start.side_effect = target
        thread.is_alive.return_value = False
        threads.append(thread)
        return thread

    supervisor = importlib.import_module("loop.utils.process")
    coordinator = importlib.import_module("loop.execution.coordinator")
    kill = Mock()
    close = Mock()
    monkeypatch.setattr(supervisor, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(coordinator, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(coordinator.threading, "Thread", thread_factory)
    monkeypatch.setattr(coordinator.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(supervisor.os, "killpg", kill)
    monkeypatch.setattr(supervisor.os, "close", close)
    return SimpleNamespace(
        process=process, threads=threads, thread_factory=thread_factory, kill=kill, close=close
    )
