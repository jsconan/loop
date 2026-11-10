"""Exercise a live mediated tunnel with an external VM or authority controller.

Run this script only with the owned routed egress fixture on a disposable Linux VM.
The controller is a separate process; it does not rely on Loop's own cleanup.
"""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loop import PermissionManager, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
from loop.tools.system import run_command

_ORIGIN = "https://11.0.0.2:18443"


class _RefusingHost:
    """Fail the probe if native mediation offers a host execution path."""

    def run_host_command(self, request):
        """Reject an unrequested host command.

        Args:
            request (object): Unexpected host execution request.

        Raises:
            AssertionError: Always, because mediation must stay sandboxed.
        """
        raise AssertionError("Mediated egress offered host execution.")


def _probe(mode: str) -> None:
    """Start one public streaming attempt and expose its owned ready marker."""
    root = Path(f"/tmp/loop-egress-{mode}-probe")
    root.mkdir(mode=0o700)
    workspace = root / "workspace"
    workspace.mkdir()
    manager = PermissionManager(workspace, yolo=True)
    registry = ToolRegistry(
        [run_command],
        permission_manager=manager,
        execution_service=CommandExecutionService(LinuxNativeBackend(), _RefusingHost()),
    )
    request = {
        "command": (
            f'curl -kfsS --no-buffer --max-time 20 {_ORIGIN}/slow > "$WORKSPACE_ROOT/stream"'
        ),
        "cwd": str(workspace),
        "read_only": False,
        "egress_origins": [_ORIGIN],
    }
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: json.loads(registry.call("run_command", json.dumps(request)))
            )
            stream = workspace / "stream"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if stream.exists() and "started" in stream.read_text():
                    break
                time.sleep(0.02)
            assert stream.exists() and "started" in stream.read_text()
            (root / "ready").write_text(str(os.getpid()))
            print(f"READY pid={os.getpid()} mode={mode}", flush=True)
            if mode == "suspend":
                result = future.result(timeout=25)
                assert result["ok"], result
                assert "finished" in stream.read_text()
                print("PASS resumed stream completed", flush=True)
            else:
                time.sleep(60)
    finally:
        registry.close()


def _active_socket() -> bool:
    """Report whether the owned HTTPS fixture has an established host transport."""
    result = subprocess.run(
        ["ss", "-Htn", "state", "established"], capture_output=True, text=True, check=True
    )
    return any("11.0.0.2:18443" in line for line in result.stdout.splitlines())


def _control_death() -> None:
    """Kill the ordinary authority and independently time receiver and route closure."""
    root = Path("/tmp/loop-egress-death-probe")
    deadline = time.monotonic() + 20
    while not (root / "ready").is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (root / "ready").is_file(), "Authority probe never became ready."
    pid = int((root / "ready").read_text())
    units = subprocess.run(
        [
            "systemctl",
            "list-units",
            "--type=service",
            "--state=running",
            "--plain",
            "--no-legend",
            "loopgen_native_*@*.service",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    names = [line.split()[0] for line in units.stdout.splitlines() if line.strip()]
    assert len(names) == 1, names
    assert _active_socket(), "Owned streaming tunnel was absent before authority death."
    unit = names[0]
    started = time.monotonic()
    os.kill(pid, signal.SIGKILL)
    while time.monotonic() - started < 2.0:
        socket_closed = not _active_socket()
        status = subprocess.run(
            ["systemctl", "is-active", unit], capture_output=True, text=True, check=False
        ).stdout.strip()
        if socket_closed and status != "active":
            print(
                f"PASS authority death closed tunnel and receiver in "
                f"{time.monotonic() - started:.3f}s; status={status}"
            )
            return
        time.sleep(0.05)
    raise AssertionError(
        f"Authority death left egress or receiver live after 2s: "
        f"socket_closed={socket_closed}, status={status}"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["control-death"]:
        _control_death()
    elif sys.argv[1:] in (["suspend"], ["death"]):
        _probe(sys.argv[1])
    else:
        raise SystemExit("Usage: egress_lifecycle_probe.py suspend|death|control-death")
