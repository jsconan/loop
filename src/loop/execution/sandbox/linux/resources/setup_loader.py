"""Authenticate exact reviewed setup input before importing privileged package code.

Execute this standard-library-only loader through isolated Python -c with inherited terminal
streams. The human's fresh review binds this loader and the entire input digest. No credential
handling or source download occurs. The root temporary import snapshot is always removed.
"""

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

_MAX_SETUP_INPUT_BYTES = 34 * 1024 * 1024
_MAX_SETUP_CONTENT_BYTES = 17 * 1024 * 1024
_MAX_SETUP_CONTENT_FILES = 256
_MAX_SETUP_ERROR_COUNT = 16
_MAX_SETUP_ERROR_CHARS = 160
_INITIAL_USER_NAMESPACE_UID_MAP = ("0", "0", "4294967295")


def _validate_invocation() -> tuple[str, str, str, str, str, str, str]:
    """Require initial-host root authority and the fixed reviewed setup arguments."""
    if os.getuid() != 0 or (
        tuple(Path("/proc/self/uid_map").read_text(encoding="ascii").split())
        != _INITIAL_USER_NAMESPACE_UID_MAP
    ):
        raise PermissionError("Setup requires initial-host administrator authentication.")

    if len(sys.argv) != 8 or re.fullmatch("[a-f0-9]{64}", sys.argv[2]) is None:
        raise ValueError("Supply exact reviewed setup input and authority.")
    return tuple(sys.argv[1:])


def _read_reviewed_input(path: str, expected_digest: str) -> bytes:
    """Read one bounded regular setup input and authenticate its exact bytes."""
    # Read once, bound allocation, then authenticate that same immutable in-memory snapshot.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_size > _MAX_SETUP_INPUT_BYTES
        ):
            raise ValueError("Reviewed setup input is not a bounded regular file.")
        data = stream.read(_MAX_SETUP_INPUT_BYTES + 1)

    if len(data) > _MAX_SETUP_INPUT_BYTES or hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError("Reviewed setup input changed or exceeds its bound.")
    return data


def _restore_import_inventory(data: bytes) -> tuple[dict[str, bytes], list[str]]:
    """Validate the reviewed package schema and return its bounded import inventory."""
    payload = json.loads(data)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"contents", "runtime"}
        or not isinstance(payload["contents"], dict)
        or not isinstance(payload["runtime"], list)
    ):
        raise ValueError("Invalid reviewed setup input schema.")

    contents = {name: bytes.fromhex(value) for name, value in payload["contents"].items()}
    if (
        not 1 <= len(contents) <= _MAX_SETUP_CONTENT_FILES
        or sum(map(len, contents.values())) > _MAX_SETUP_CONTENT_BYTES
    ):
        raise ValueError("Reviewed setup import inventory exceeds its bound.")

    for name in contents:
        path = Path(name)
        if (
            re.fullmatch("[A-Za-z0-9_./-]{1,1024}", name) is None
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
        ):
            raise ValueError("Invalid reviewed import member.")

    if (
        len(set(payload["runtime"])) != len(payload["runtime"])
        or not set(payload["runtime"]) <= contents.keys()
    ):
        raise ValueError("Invalid reviewed runtime inventory.")
    return contents, payload["runtime"]


def _invoke_reviewed_setup(
    contents: dict[str, bytes], runtime_names: list[str], arguments: tuple[str, ...]
) -> None:
    """Materialize the authenticated imports privately and invoke the fixed setup owner."""
    with tempfile.TemporaryDirectory(prefix="loop-native-authenticated-", dir="/run") as temporary:
        root = Path(temporary)
        for name, value in contents.items():
            target = root / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(value)
                os.fchmod(stream.fileno(), 0o400)
        sys.path.insert(0, str(root))

        from loop.execution.sandbox.linux.setup import install_reviewed

        install_reviewed(
            arguments[2],
            arguments[3],
            int(arguments[4]),
            int(arguments[5]),
            arguments[6],
            {name: contents[name] for name in runtime_names},
        )


def _execute_reviewed_setup() -> None:
    """Authenticate the setup input and invoke the reviewed fixed sandbox setup owner."""
    arguments = _validate_invocation()
    data = _read_reviewed_input(arguments[0], arguments[1])
    contents, runtime_names = _restore_import_inventory(data)
    _invoke_reviewed_setup(contents, runtime_names, arguments)


def main() -> int:
    """Run reviewed setup and return a bounded process status without exposing tracebacks.

    Returns:
        int: Zero after completion, one after an expected management refusal.
    """
    try:
        _execute_reviewed_setup()
    except (ValueError, OSError, ExceptionGroup) as error:
        pending = [error]
        reasons = []
        while pending and len(reasons) < _MAX_SETUP_ERROR_COUNT:
            cause = pending.pop(0)
            if isinstance(cause, ExceptionGroup):
                pending.extend(cause.exceptions[:_MAX_SETUP_ERROR_COUNT])
            else:
                lines = str(cause).splitlines()
                reason = re.sub(r"[^ -~]", "?", lines[0] if lines else "Management refused.")[
                    :_MAX_SETUP_ERROR_CHARS
                ]
                if reason not in reasons:
                    reasons.append(reason)
        print(json.dumps({"errors": reasons}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
