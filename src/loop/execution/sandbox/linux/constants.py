"""Define constants shared by the Linux native command sandbox."""

import struct
from pathlib import Path

from .... import constants as app_constants

# Reviewed import members follow the Python package, not the installed application name.
PACKAGE_NAMESPACE = __package__.partition(".")[0]
PACKAGE_ARTIFACT_PREFIX = f"{PACKAGE_NAMESPACE}/"
LINUX_ARTIFACT_DIRECTORY = f"{PACKAGE_ARTIFACT_PREFIX}execution/sandbox/linux/"
BOOTSTRAP_ARTIFACT = f"{LINUX_ARTIFACT_DIRECTORY}bootstrap.py"

INITIAL_USER_NAMESPACE_UID_MAP = ("0", "0", "4294967295")
CGROUP_ROOT = Path("/sys/fs/cgroup")
CGROUP_PROCS_FILENAME = "cgroup.procs"
CACHE_LOCK_FILENAME = "cache.lock"
BOOT_FILENAME = f"{PACKAGE_NAMESPACE}_boot.py"
BOOT_SERVICE_NAME = f"{app_constants.APP_NAME}-native-boot.service"
COMPILER_LOCK_FILENAME = "compiler.lock"
MANAGEMENT_LOCK_FILENAME = "management.lock"
MANAGEMENT_RECORD_NAME = "management.json"
NATIVE_RESERVATIONS_DIRECTORY_NAME = "reservations"
# The namespace path must also be valid as the parent of a bound command alias.
NATIVE_SCRATCH_PATH = Path("/") / f"{app_constants.APP_NAME}-private-scratch"
NATIVE_ALIAS_PATH = Path("/") / f".{app_constants.APP_NAME}-private-paths"
NATIVE_TRACKING_PATH = NATIVE_ALIAS_PATH / ".tracking"
NATIVE_PROFILE_GROUP_HASH_CHARS = 48
NATIVE_CONVENTIONAL_TMP_PATH = Path("/tmp")
SESSION_CONVENTIONAL_TMP_NAME = f"{app_constants.APP_NAME}-conventional-tmp"
NATIVE_POLICY_VERSION = "linux-native-v1"
# Change when retained receiver semantics require a checked installation refresh.
NATIVE_RECEIVER_REVISION = "sandbox-local-network-v1"
# This runtime revision uses a shell-safe private scratch namespace path.
NATIVE_PAYLOAD_MARKER_PREFIX = "LOOP_PAYLOAD_"
NATIVE_REQUEST_MAGIC = b"LOOP\x01"
# The receiver retains its admitted socket for non-consuming stalled-spawn control observation.
# The receiver checks this control before spawn and while supervising the child.
NATIVE_CANCEL_FRAME = b"\x00"
PAYLOAD_FILENAME = f"{PACKAGE_NAMESPACE}_payload.py"
PEER_CREDENTIALS_FORMAT = "3i"
PEER_CREDENTIALS_SIZE = struct.calcsize(PEER_CREDENTIALS_FORMAT)
PROC_STATUS_PATH = Path("/proc/self/status")
PROC_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
PROC_UID_MAP_PATH = Path("/proc/self/uid_map")
RECEIPT_FILENAME = "receipt.json"
RECEIVER_FILENAME = f"{PACKAGE_NAMESPACE}_receiver.py"
REQUEST_LENGTH_FORMAT = "!I"
REQUEST_LENGTH_SIZE = struct.calcsize(REQUEST_LENGTH_FORMAT)
SELECTED_DIRECTORY_NAME = "selected"
SO_PEERCRED_OPTION_FALLBACK = 17
SYSTEMD_ACTIVE_STATE_PROPERTY = "ActiveState"
SYSTEMD_CONTROL_GROUP_PROPERTY = "ControlGroup"
SYSTEMD_DROP_IN_PATHS_PROPERTY = "DropInPaths"
SYSTEMD_FRAGMENT_PATH_PROPERTY = "FragmentPath"
SYSTEMD_LOAD_STATE_PROPERTY = "LoadState"
SYSTEMD_NEED_DAEMON_RELOAD_PROPERTY = "NeedDaemonReload"
SYSTEMD_SERVICE_TEMPLATE_SUFFIX = "@.service"
SYSTEMD_SLICE_UNIT_SUFFIX = ".slice"
SYSTEMD_SOCKET_UNIT_SUFFIX = ".socket"
SYSTEMD_SUB_STATE_PROPERTY = "SubState"
SYSTEMD_TRANSIENT_PROPERTY = "Transient"
SYSTEMD_COLORS_ENVIRONMENT_VARIABLE = "SYSTEMD_COLORS"
SYSTEMD_PAGER_ENVIRONMENT_VARIABLE = "SYSTEMD_PAGER"
SYSTEM_RUNTIME_DIRECTORY = Path("/run")
UNSAFE_HOST_PATH_MODE_BITS = 0o6022
UNINSTALLER_FILENAME = f"{PACKAGE_NAMESPACE}_uninstall.py"
SYSTEM_COMMAND_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
SYSTEM_PYTHON_PATH = Path("/usr/bin/python3")
SETPRIV_PATH = Path("/usr/bin/setpriv")
SYSTEMCTL_PATH = Path("/usr/bin/systemctl")
