"""Authenticate ordinary socket admission inside a protected systemd generation.

The trusted installer supplies fixed authority; client bytes never select users, units or
tracking paths. This owner does not elevate, launch source, remove policy or enable Linux.
"""

import os
import re
import socket
import stat
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from .... import constants
from ....constants import (
    MAX_NATIVE_REQUEST_BYTES,
    MAX_NATIVE_UNIT_BYTES,
    NATIVE_ADMISSION_SOCKET_MODE,
    NATIVE_UNIT_PREFIX,
)
from ..constants import INITIAL_USER_NAMESPACE_UID_MAP
from .constants import (
    CGROUP_PROCS_FILENAME,
    CGROUP_ROOT,
    NATIVE_REQUEST_MAGIC,
    PEER_CREDENTIALS_FORMAT,
    PEER_CREDENTIALS_SIZE,
    PROC_STATUS_PATH,
    PROC_UID_MAP_PATH,
    REQUEST_LENGTH_FORMAT,
    REQUEST_LENGTH_SIZE,
    SETPRIV_PATH,
    SO_PEERCRED_OPTION_FALLBACK,
    SYSTEM_PYTHON_PATH,
    SYSTEM_RUNTIME_DIRECTORY,
    SYSTEMD_CONTROL_GROUP_PROPERTY,
    SYSTEMD_SERVICE_TEMPLATE_SUFFIX,
    SYSTEMD_SLICE_UNIT_SUFFIX,
    SYSTEMD_SOCKET_UNIT_SUFFIX,
)
from .host import HostFile, bind_host_file, protected_path
from .installation import generation_limits
from .mountinfo import MINIMUM_MOUNTINFO_FIELDS, read_descriptor_mount_ids, read_mountinfo
from .preflight import attest_units

MAX_FRAME_BYTES = MAX_NATIVE_REQUEST_BYTES
ADMISSION_SECONDS = 5
_SO_DOMAIN_OPTION = getattr(socket, "SO_DOMAIN", 39)
_SO_PEERCRED_OPTION = getattr(socket, "SO_PEERCRED", SO_PEERCRED_OPTION_FALLBACK)
_SOCKET_BACKLOG = 16
_SOCKET_MAX_CONNECTIONS = 32
_RECEIVER_UMASK = constants.GROUP_AND_OTHER_MODE_MASK


@dataclass(frozen=True)
class GenerationAdmission:
    """Bind fixed administrator-selected ordinary identity and generation entry.

    Args:
        generation (str): Simple installation generation, at most 128 ASCII characters.
        uid (int): Positive ordinary host UID selected by the installer.
        gid (int): Positive ordinary host GID selected by the installer.
        receiver (Path): Protected fixed receiver entry, using an ASCII installation path.
        slice_unit (HostFile): Authenticated immutable generation slice fragment.
        reservation (HostFile): Bound installation-wide retained-generation capacity record.
    """

    generation: str
    uid: int
    gid: int
    receiver: Path
    slice_unit: HostFile
    reservation: HostFile

    def unit_sources(self) -> dict[str, str]:
        """Render fixed socket, ordinary service template and protected generation slice.

        No client data reaches ExecStart or root setup. The receiver runs with host privileges
        removed before reading any frame. ExitType=cgroup keeps detached descendants tracked
        after their parent exits. No dependency on listener stop kills accepted services.
        The launcher uses NoNewPrivileges without a systemd seccomp filter: systemd's
        privilege-drop setup for such filters can remove SETUID before setpriv executes.
        Memory pressure watching is disabled to prevent a writable pressure-file submount.
        This is installation input, not permission to publish, activate or enable units.

        Returns:
            dict[str, str]: Three exact fragment names and complete UTF-8 unit sources.

        Raises:
            ValueError: A fixed identity, path or slice-fragment spelling is invalid.
        """
        if (
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", self.generation) is None
            or not isinstance(self.uid, int)
            or isinstance(self.uid, bool)
            or self.uid <= 0
            or not isinstance(self.gid, int)
            or isinstance(self.gid, bool)
            or self.gid <= 0
            or re.fullmatch(r"/[A-Za-z0-9_./-]+", str(self.receiver)) is None
            or ".." in self.receiver.parts
        ):
            raise ValueError("Invalid fixed generation admission authority.")

        prefix = NATIVE_UNIT_PREFIX + self.generation
        if self.slice_unit.path.name != prefix + SYSTEMD_SLICE_UNIT_SUFFIX:
            raise ValueError("Generation slice fragment differs from fixed authority.")

        return {
            prefix
            + SYSTEMD_SLICE_UNIT_SUFFIX: "[Unit]\nDescription=Loop protected generation\n[Slice]\n",
            prefix + SYSTEMD_SOCKET_UNIT_SUFFIX: (
                "[Unit]\nDescription=Loop ordinary command admission\n[Socket]\n"
                f"ListenStream={SYSTEM_RUNTIME_DIRECTORY}/{prefix}{SYSTEMD_SOCKET_UNIT_SUFFIX}\nAccept=yes\n"
                f"SocketUser={self.uid}\nSocketGroup={self.gid}\n"
                f"SocketMode={NATIVE_ADMISSION_SOCKET_MODE:04o}\n"
                f"Backlog={_SOCKET_BACKLOG}\n"
                f"MaxConnections={_SOCKET_MAX_CONNECTIONS}\nRemoveOnStop=yes\n"
            ),
            prefix + SYSTEMD_SERVICE_TEMPLATE_SUFFIX: (
                "[Unit]\nDescription=Loop protected command receiver\n[Service]\n"
                "Type=exec\nExitType=cgroup\n"
                f"ExecStart={SETPRIV_PATH} --reuid={self.uid} --regid={self.gid} "
                "--clear-groups --bounding-set=-all --inh-caps=-all --ambient-caps=-all "
                f"--no-new-privs {SYSTEM_PYTHON_PATH} -I -B {self.receiver}\n"
                "User=0\nGroup=0\n"
                f"Slice={prefix}{SYSTEMD_SLICE_UNIT_SUFFIX}\n"
                "StandardInput=socket\nStandardOutput=socket\nStandardError=journal\n"
                "ProtectControlGroups=yes\nDelegate=no\nKillMode=control-group\n"
                "MemoryPressureWatch=off\n"
                "NoNewPrivileges=yes\n"
                "CapabilityBoundingSet=CAP_SETUID CAP_SETGID CAP_SETPCAP\n"
                "AmbientCapabilities=\n"
                f"UMask={_RECEIVER_UMASK:04o}\nWorkingDirectory=/\n"
            ),
        }


def _remaining(deadline: float) -> float:
    """Refuse continuation when the single admission budget expires."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Generation admission deadline expired.")
    return remaining


def _tracking(config: GenerationAdmission) -> tuple[str, Path]:
    """Verify host membership, the kernel-selected readonly mount and protected ancestry."""
    if (
        os.getresuid() != (config.uid,) * 3
        or os.getresgid() != (config.gid,) * 3
        or os.getgroups()
        or tuple(PROC_UID_MAP_PATH.read_text(encoding="ascii").split())
        != INITIAL_USER_NAMESPACE_UID_MAP
    ):
        raise ValueError("Receiver must have only its fixed ordinary host identity.")

    status = dict(
        line.split(":", 1) for line in PROC_STATUS_PATH.read_text(encoding="ascii").splitlines()
    )
    if status.get("NoNewPrivs", "").strip() != "1" or any(
        int(status.get(name, "-1"), 16) != 0
        for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    ):
        raise ValueError("Receiver retains host privilege authority.")

    prefix = NATIVE_UNIT_PREFIX + config.generation
    membership = Path("/proc/self/cgroup").read_text(encoding="ascii").strip()
    if (
        re.fullmatch(
            r"0::/"
            + re.escape(prefix)
            + r"\.slice/"
            + re.escape(prefix)
            + r"@[A-Za-z0-9_.:\\-]{1,128}\.service",
            membership,
        )
        is None
    ):
        raise ValueError("Receiver has no protected generation membership.")

    relative = Path(membership[3:])
    mounts = []
    for fields in read_mountinfo():
        if len(fields) < MINIMUM_MOUNTINFO_FIELDS or "-" not in fields:
            raise ValueError("Ambiguous generation mount authority.")
        mountpoint = Path(fields[4])
        if mountpoint == CGROUP_ROOT or mountpoint.is_relative_to(CGROUP_ROOT):
            mounts.append(fields)

    # Bind to the accessible mount, not a hidden underlying mount or mountinfo row order.
    descriptor = os.open(CGROUP_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        mount_ids = read_descriptor_mount_ids(descriptor)
    finally:
        os.close(descriptor)

    visible = [fields for fields in mounts if [fields[0]] in mount_ids]
    if len(mount_ids) != 1 or len(visible) != 1:
        raise ValueError("Generation cgroup mount ID is missing or ambiguous.")

    if "ro" not in visible[0][5].split(","):
        raise ValueError("Generation cgroup mount is writable.")

    if any(
        fields[4] != str(CGROUP_ROOT)
        or fields[3] != "/"
        or fields[2] != visible[0][2]
        or fields[fields.index("-") + 1] != "cgroup2"
        for fields in mounts
    ):
        raise ValueError("Generation cgroup mount has a submount or different hierarchy.")

    group = CGROUP_ROOT / relative.relative_to("/")
    for path in (CGROUP_ROOT, group.parent, group):
        if protected_path(path, directory=True) != path:
            raise ValueError("Generation cgroup ancestry is aliased.")

        for name in (CGROUP_PROCS_FILENAME, "cgroup.threads", "cgroup.subtree_control"):
            details = (path / name).lstat()

            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_uid != 0
                or details.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK
            ):
                raise ValueError("Generation membership authority is delegated.")

    if (group / "cgroup.type").read_text(encoding="ascii").strip() != "domain":
        raise ValueError("Generation membership is not a process domain.")

    return group.name, group


def admission_group(config: GenerationAdmission) -> Path:
    """Authenticate the receiver's immutable kernel generation membership.

    Use only in the trusted installed receiver after authenticated admission. This check
    supplies a kernel cleanup boundary, not request authorization or permission to launch.

    Args:
        config (GenerationAdmission): Fixed installed ordinary identity and generation.

    Returns:
        Path: Canonical protected concrete service cgroup containing this receiver.

    Raises:
        ValueError: Identity, capabilities, membership or mount authority is unsafe.
        OSError: Required kernel membership or mount metadata cannot be inspected.
    """
    config.unit_sources()
    return _tracking(config)[1]


def receive_request(
    config: GenerationAdmission,
    files: dict[str, HostFile],
    deadline: float,
    *,
    descriptor: int = 0,
) -> bytes:
    """Authenticate fixed activated admission and receive exactly one bounded opaque frame.

    Call only from the installed trusted ordinary receiver, before parsing request fields or
    launching command code. Authenticate kernel membership, installed unit bytes and actual
    loaded socket/concrete-service properties before touching client data. Peers must match the
    installer-selected ordinary UID/GID; PIDs are diagnostic identity, never migration targets.
    The frame is LOOP/version-one, a four-byte big-endian length, nonempty opaque bytes and EOF.
    No descriptor-receiving API is used. Revalidate membership and installed authority after EOF.
    This transport grant supplies no filesystem, network, policy or command authorization.

    Args:
        config (GenerationAdmission): Fixed trusted installation admission authority.
        files (dict[str, HostFile]): Current trusted installed authority, including receiver,
            socket_unit and service_unit. Full preparation supplies installation_files results.
        deadline (float): Absolute monotonic budget, capped to five seconds for all admission.
        descriptor (int): Activated connected socket descriptor; standard input by default.

    Returns:
        bytes: One opaque request of at most 64 KiB, received as the fixed ordinary peer.

    Raises:
        ValueError: Identity, tracking, installed/loaded authority, peer or frame is invalid.
        TimeoutError: Admission exceeds its single bounded deadline.
        OSError: Native authority, activated socket or bounded frame reception fails.
    """
    sources = config.unit_sources()
    deadline = min(deadline, time.monotonic() + ADMISSION_SECONDS)
    _remaining(deadline)
    generation_limits(config.reservation, config.generation, deadline=deadline)
    service, group = _tracking(config)
    prefix = NATIVE_UNIT_PREFIX + config.generation

    for role, name in (
        ("socket_unit", prefix + SYSTEMD_SOCKET_UNIT_SUFFIX),
        ("service_unit", prefix + SYSTEMD_SERVICE_TEMPLATE_SUFFIX),
    ):
        with files[role].open_verified(deadline=deadline) as fd:
            if os.pread(fd, MAX_NATIVE_UNIT_BYTES + 1, 0) != sources[name].encode():
                raise ValueError("Installed admission unit differs from fixed contract.")

    with config.slice_unit.open_verified(deadline=deadline) as fd:
        if (
            os.pread(fd, MAX_NATIVE_UNIT_BYTES + 1, 0)
            != sources[prefix + SYSTEMD_SLICE_UNIT_SUFFIX].encode()
        ):
            raise ValueError("Installed generation slice differs from fixed contract.")

    if files["receiver"].path != config.receiver:
        raise ValueError("Installed receiver differs from fixed contract.")

    native_inputs = [
        bind_host_file(Path(path), deadline=deadline) for path in (SETPRIV_PATH, SYSTEM_PYTHON_PATH)
    ]

    command = (
        sources[prefix + SYSTEMD_SERVICE_TEMPLATE_SUFFIX]
        .split("ExecStart=", 1)[1]
        .split("\n", 1)[0]
    )
    attest_units(
        files,
        {
            prefix + SYSTEMD_SOCKET_UNIT_SUFFIX: {
                "Listen": f"{SYSTEM_RUNTIME_DIRECTORY}/{prefix}{SYSTEMD_SOCKET_UNIT_SUFFIX} (Stream)",
                "Accept": "yes",
            },
            service: {
                "ExecStart": (
                    f"{{ path={SETPRIV_PATH} ; argv[]="
                    + command
                    + " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; "
                    + "pid=0 ; code=(null) ; status=0/0 }"
                ),
                "User": "0",
                "Group": "0",
                "MainPID": str(os.getpid()),
                "Slice": prefix + SYSTEMD_SLICE_UNIT_SUFFIX,
                SYSTEMD_CONTROL_GROUP_PROPERTY: "/" + str(group.relative_to(CGROUP_ROOT)),
                "NoNewPrivileges": "yes",
                "ExitType": "cgroup",
                "MemoryPressureWatch": "off",
                "CapabilityBoundingSet": "cap_setgid cap_setuid cap_setpcap",
            },
        },
        deadline,
    )

    with socket.fromfd(descriptor, socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        if (
            connection.getsockopt(socket.SOL_SOCKET, _SO_DOMAIN_OPTION) != socket.AF_UNIX
            or connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
            or connection.getsockname()
            != f"{SYSTEM_RUNTIME_DIRECTORY}/{prefix}{SYSTEMD_SOCKET_UNIT_SUFFIX}"
        ):
            raise ValueError("Receiver descriptor is not its activated local stream.")

        pid, uid, gid = struct.unpack(
            PEER_CREDENTIALS_FORMAT,
            connection.getsockopt(socket.SOL_SOCKET, _SO_PEERCRED_OPTION, PEER_CREDENTIALS_SIZE),
        )

        if pid <= 0 or (uid, gid) != (config.uid, config.gid):
            raise ValueError("Admission peer differs from fixed ordinary identity.")

        def read_exact(size):
            """Bound each stream read by the original deadline and reject early EOF."""
            data = bytearray()
            while len(data) < size:
                connection.settimeout(_remaining(deadline))
                block = connection.recv(size - len(data))
                if not block:
                    raise ValueError("Truncated admission frame.")
                data.extend(block)
            return bytes(data)

        header = read_exact(len(NATIVE_REQUEST_MAGIC) + REQUEST_LENGTH_SIZE)
        length = struct.unpack(REQUEST_LENGTH_FORMAT, header[len(NATIVE_REQUEST_MAGIC) :])[0]
        if (
            header[: len(NATIVE_REQUEST_MAGIC)] != NATIVE_REQUEST_MAGIC
            or not 0 < length <= MAX_FRAME_BYTES
        ):
            raise ValueError("Invalid admission frame header or length.")

        payload = read_exact(length)
        connection.settimeout(_remaining(deadline))
        if connection.recv(1):
            raise ValueError("Trailing admission frame bytes.")

    if _tracking(config) != (service, group):
        raise ValueError("Generation membership changed during admission.")

    for bound in (*files.values(), config.slice_unit, config.reservation, *native_inputs):
        with bound.open_verified(deadline=deadline):
            pass

    _remaining(deadline)
    return payload
