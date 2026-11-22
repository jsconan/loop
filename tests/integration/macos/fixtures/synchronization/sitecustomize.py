"""Select explicit compatibility bindings for native workload qualification."""

import os
import pwd

from synchronization_client import enable as enable_synchronization


def _bind_private_home() -> None:
    """Retain the assigned private home when a workload clears its environment."""
    home = os.environ["HOME"]
    if not os.path.isabs(home) or not os.path.isdir(home):
        raise ValueError("The test account requires an existing absolute HOME.")
    uid = os.getuid()
    account = pwd.struct_passwd((f"sandbox-{uid}", "*", uid, os.getgid(), "", home, "/bin/sh"))
    native_lookup = pwd.getpwuid

    def getpwuid(requested_uid):
        """Supply the private record for the current UID and delegate other lookups."""
        return account if requested_uid == uid else native_lookup(requested_uid)

    pwd.getpwuid = getpwuid


if os.environ.get("LOOP_TEST_PRIVATE_ACCOUNT") == "1":
    _bind_private_home()
if "LOOP_SYNC_SOCKET" in os.environ:
    enable_synchronization()
