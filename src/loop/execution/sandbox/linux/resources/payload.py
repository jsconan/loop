"""Enter the original enforcing payload from the installed receiver's private setup input."""

import base64
import os
import sys
from pathlib import Path
from typing import Any


def _prepare_import_path() -> None:
    """Validate the retained entry before exposing its package directory to imports."""
    entry = Path(__file__).absolute()
    if entry.name != "loop_payload.py" or entry.resolve(strict=True) != entry:
        raise ValueError("Payload requires its fixed canonical installed path.")
    sys.path.insert(0, str(entry.parent))


def _restore_trusted_setup() -> tuple[Any, dict[str, Any]]:
    """Restore the bounded receiver-prepared request and setup envelope."""
    from loop.constants import MAX_NATIVE_SETUP_BYTES
    from loop.execution.sandbox.contracts import SandboxRequest
    from loop.utils.json import load_unique_json

    data = sys.stdin.buffer.read(MAX_NATIVE_SETUP_BYTES + 1)
    if len(data) > MAX_NATIVE_SETUP_BYTES:
        raise ValueError("Trusted setup input exceeds its bound.")
    value = load_unique_json(data, duplicate_message="Duplicate trusted setup field.")
    request = SandboxRequest.from_wire(base64.b64decode(value["request"], validate=True))
    return request, value


def _bound_host_file(value: dict[str, Any], role: str) -> Any:
    """Restore one host-authenticated file binding without rebinding its identity."""
    from loop.execution.sandbox.linux.host import HostFile

    record = value[role]
    return HostFile(Path(record["path"]), tuple(record["identity"]), record["sha256"])


def _enter_and_execute(request: Any, value: dict[str, Any]) -> None:
    """Enter trusted bootstrap, discard private stdin and irreversibly start the payload."""
    from loop.execution.sandbox.linux.bootstrap import enter_bootstrap, execute_payload

    library = _bound_host_file(value, "apparmor")
    enter_bootstrap(library, value["namespace"], value["uid"])
    descriptor = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.dup2(descriptor, 0)
    finally:
        os.close(descriptor)
    execute_payload(
        request,
        library,
        _bound_host_file(value, "bootstrap"),
        _bound_host_file(value, "cache_lock"),
        base64.b64decode(value["compiled"], validate=True),
        value["namespace"],
        value["uid"],
        reservation=_bound_host_file(value, "reservation"),
    )


def main() -> None:
    """Consume trusted receiver-prepared inputs and irreversibly replace mapped setup.

    Only the authenticated receiver constructs this anonymous input. It is never read from a
    client socket or command argument. Its protected code and control views must be mounted
    read-only by that receiver; host-root startup is refused by the mapped bootstrap validator.
    Any failed restoration or native transition ends setup without executing command source.

    Raises:
        ValueError: The fixed entry, request or original native authority is invalid.
        TimeoutError: The original request deadline expires.
        OSError: Native preparation, transition or shell replacement fails.
    """
    _prepare_import_path()
    request, value = _restore_trusted_setup()
    _enter_and_execute(request, value)


if __name__ == "__main__":
    main()
