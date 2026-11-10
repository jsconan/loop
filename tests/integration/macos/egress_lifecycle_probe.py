"""Run an owned macOS mediated tunnel under an external native test controller."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from loop import PermissionManager, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.sandbox import SandboxCapability, select_sandbox_backend
from loop.execution.sandbox.macos import backend as macos_backend
from loop.tools.system import run_command

_ORIGIN = "https://11.0.0.2:18443"


class _RefusingHost:
    """Reject any unrequested host execution from the controlled probe."""

    def run_host_command(self, request):
        """Fail if the mediated attempt offers a host process.

        Args:
            request (object): Unexpected host execution request.

        Raises:
            AssertionError: Always; the test never grants a host process.
        """
        raise AssertionError("Mediated egress offered host execution.")


def main() -> None:
    """Launch one streaming sandbox command and leave cleanup to the controller."""
    workspace = Path(sys.argv[1])
    parent = Path(sys.argv[2])
    device, device_id, name, volume_uuid = sys.argv[3:]
    volume = macos_backend.LocalSocketVolume(device, parent, int(device_id), name, volume_uuid)
    macos_backend._PRIVATE_TEMP_ROOT = parent
    macos_backend._mount_local_socket_volume = lambda _deadline: volume
    macos_backend._eject_local_socket_volume = lambda _volume: None
    monitor = Mock()
    monitor.register.return_value = None
    monitor.denial.return_value = None
    macos_backend.SeatbeltDiagnosticService.monitor = lambda *_args: monitor
    macos_backend._denial_from_log = lambda *_args: None
    backend = select_sandbox_backend()
    assert SandboxCapability.MEDIATED_EGRESS in backend.capabilities
    manager = PermissionManager(workspace, yolo=True)
    registry = ToolRegistry(
        [run_command],
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, _RefusingHost()),
    )
    source = (
        'printf %s "$http_proxy" > "$WORKSPACE_ROOT/proxy-url"; '
        'printf %s "$$" > "$WORKSPACE_ROOT/payload-pid"; '
        f"exec /usr/bin/curl -kfsS --no-buffer --max-time 20 {_ORIGIN}/slow "
        '> "$WORKSPACE_ROOT/stream"'
    )
    request = {
        "command": source,
        "cwd": str(workspace),
        "egress_origins": [_ORIGIN],
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            lambda: json.loads(registry.call("run_command", json.dumps(request)))
        )
        stream = workspace / "stream"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if stream.exists() and "started" in stream.read_text():
                break
            if running.done():
                raise AssertionError(f"Native stream ended before readiness: {running.result()}")
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        (workspace / "ready").write_text(str(os.getpid()), encoding="utf-8")
        result = running.result(timeout=25)
        (workspace / "result").write_text(json.dumps(result), encoding="utf-8")
    registry.close()


if __name__ == "__main__":
    main()
