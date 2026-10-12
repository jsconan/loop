"""Define constants shared by the Linux native command sandbox."""

import struct
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
INITIAL_USER_NAMESPACE_UID_MAP = ("0", "0", "4294967295")
MINIMAL_NATIVE_PATH = "/usr/bin:/bin"
NATIVE_REQUEST_MAGIC = b"LOOP\x01"
PEER_CREDENTIALS_FORMAT = "3i"
PEER_CREDENTIALS_SIZE = struct.calcsize(PEER_CREDENTIALS_FORMAT)
PROC_UID_MAP_PATH = Path("/proc/self/uid_map")
REQUEST_LENGTH_FORMAT = "!I"
REQUEST_LENGTH_SIZE = struct.calcsize(REQUEST_LENGTH_FORMAT)
SO_PEERCRED_OPTION_FALLBACK = 17
UNSAFE_HOST_PATH_MODE_BITS = 0o6022
SYSTEM_COMMAND_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
SYSTEM_LOCALE = "C"
SYSTEM_PYTHON_PATH = Path("/usr/bin/python3")
SETPRIV_PATH = Path("/usr/bin/setpriv")
SYSTEMCTL_PATH = Path("/usr/bin/systemctl")
