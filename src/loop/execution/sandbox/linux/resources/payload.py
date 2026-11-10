"""Enter the original enforcing payload from the installed receiver's private setup input."""

import base64
import fcntl
import os
import stat
import sys
import termios
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
    from loop.execution.sandbox.linux.constants import NATIVE_TRACKING_PATH

    library = _bound_host_file(value, "apparmor")
    tracking = value["tracking"]
    if (
        not isinstance(tracking, list)
        or len(tracking) != 3
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in tracking[:2]
        )
        or not isinstance(tracking[2], str)
    ):
        raise ValueError("Invalid generation tracking input.")
    enter_bootstrap(library, value["namespace"], value["uid"])
    if request.terminal:
        if not os.isatty(1) or not os.isatty(2):
            raise ValueError("Interactive payload lacks its receiver-owned terminal slave.")
        os.dup2(1, 0)
        if os.getsid(0) != os.getpid():
            os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    else:
        descriptor = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.dup2(descriptor, 0)
        finally:
            os.close(descriptor)
    egress_fd = value.get("egress_fd")
    origins = getattr(request, "egress_origins", ())
    if isinstance(origins, tuple) and origins:
        if not isinstance(egress_fd, int) or isinstance(egress_fd, bool) or egress_fd <= 2:
            raise ValueError("Mediated egress descriptor is unavailable.")
        if not stat.S_ISSOCK(os.fstat(egress_fd).st_mode):
            raise ValueError("Mediated egress descriptor is not a socket.")
    elif egress_fd is not None:
        raise ValueError("Unexpected mediated egress descriptor.")
    network_filter = value.get("local_network_filter")
    if (network_filter is None) != request.local_network:
        raise ValueError("Local-network filter and request authority differ.")
    execute_payload(
        request,
        library,
        _bound_host_file(value, "bootstrap"),
        _bound_host_file(value, "cache_lock"),
        base64.b64decode(value["compiled"], validate=True),
        value["profile"],
        value["namespace"],
        value["uid"],
        reservation=_bound_host_file(value, "reservation"),
        tracking=(NATIVE_TRACKING_PATH, (tracking[0], tracking[1]), tracking[2]),
        **({"egress_descriptor": egress_fd} if egress_fd is not None else {}),
        **(
            {"local_network_filter": base64.b64decode(network_filter, validate=True)}
            if network_filter is not None
            else {}
        ),
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
