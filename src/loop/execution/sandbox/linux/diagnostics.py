"""Observe authenticated Linux kernel denials without modifying installed enforcement."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

from ....constants import NATIVE_UNIT_PREFIX
from ....utils.json import load_unique_json
from .constants import CGROUP_ROOT
from .host import HostFile, bind_host_file

_NETLINK_SOCK_DIAG = 4
_SOCK_DIAG_BY_FAMILY = 20
_UNIX_DIAG_PEER = 2
_UDIAG_SHOW_PEER = 4
_NETLINK_HEADER = struct.Struct("=IHHII")
_UNIX_REQUEST = struct.Struct("=BBHIIIII")
_UNIX_MESSAGE = struct.Struct("=BBBBIII")
_ATTRIBUTE_HEADER = struct.Struct("=HH")
_MAX_JOURNAL_LINE = 65536
_MAX_EVENTS = 32
_MAX_SERVICES = 256
_MAX_PROCESSES = 128
_MAX_CGROUP_PROCESS_BYTES = 4096
_IDENTITY_ATTEMPTS = 20
_IDENTITY_RETRY_SECONDS = 0.025
_MAX_SEARCH_DIRECTORIES = 128
_MAX_DIAGNOSTIC = 500
_MAX_CANDIDATES = 4
DENIAL_RECORDS_PREFIX = "\x1eLOOP_LINUX_DENIALS:"
_PROFILE = re.compile(r"payload_[a-f0-9]{64}_([a-f0-9]{48})\Z")
_OPERATION = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _journal_program() -> HostFile | None:
    """Find an administrator-protected journal reader on the host search path."""
    search = os.defpath + os.pathsep + os.environ.get("PATH", "")
    seen = set()
    for index, directory in enumerate(search.split(os.pathsep)):
        if index >= _MAX_SEARCH_DIRECTORIES:
            break
        if directory in seen or not Path(directory).is_absolute():
            continue
        seen.add(directory)
        executable = shutil.which("journalctl", path=directory)
        if executable is not None:
            try:
                bound = bind_host_file(Path(executable))
                with bound.open_verified() as descriptor:
                    if os.pread(descriptor, 4, 0) == b"\x7fELF":
                        return bound
            except (OSError, ValueError):
                continue
    return None


def _peer_inode(connection: socket.socket) -> int | None:
    """Read this live connection's peer inode from a bounded kernel socket query."""
    inode = os.fstat(connection.fileno()).st_ino
    request = _UNIX_REQUEST.pack(
        socket.AF_UNIX, 0, 0, 0xFFFFFFFF, inode, _UDIAG_SHOW_PEER, 0xFFFFFFFF, 0xFFFFFFFF
    )
    header = _NETLINK_HEADER.pack(
        _NETLINK_HEADER.size + len(request), _SOCK_DIAG_BY_FAMILY, 1, 1, 0
    )
    with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, _NETLINK_SOCK_DIAG) as query:
        query.settimeout(0.02)
        query.bind((0, 0))
        query.connect((0, 0))
        query.sendall(header + request)
        data, sender = query.recvfrom(4096)
    if sender != (0, 0) or len(data) < _NETLINK_HEADER.size + _UNIX_MESSAGE.size:
        return None
    length, kind, _, sequence, _ = _NETLINK_HEADER.unpack_from(data)
    if length != len(data) or kind != _SOCK_DIAG_BY_FAMILY or sequence != 1:
        return None
    family, _, _, _, returned_inode, _, _ = _UNIX_MESSAGE.unpack_from(data, _NETLINK_HEADER.size)
    if family != socket.AF_UNIX or returned_inode != inode:
        return None
    offset = _NETLINK_HEADER.size + _UNIX_MESSAGE.size
    peer = None
    while offset + _ATTRIBUTE_HEADER.size <= length:
        size, attribute = _ATTRIBUTE_HEADER.unpack_from(data, offset)
        if size < _ATTRIBUTE_HEADER.size or offset + size > length:
            return None
        if attribute == _UNIX_DIAG_PEER:
            if peer is not None or size != 8:
                return None
            peer = struct.unpack_from("=I", data, offset + _ATTRIBUTE_HEADER.size)[0]
        offset += (size + 3) & ~3
    return peer if offset == length and peer else None


def _receiver_group(connection: socket.socket, generation: str, peer: int) -> str | None:
    """Bind the socket peer to one live protected generation service."""
    prefix = NATIVE_UNIT_PREFIX + generation
    service_pattern = re.compile(re.escape(prefix) + r"@[A-Za-z0-9_.:\\-]{1,128}\.service\Z")
    parent = CGROUP_ROOT / (prefix + ".slice")
    matches = []
    process_count = 0
    with os.scandir(parent) as services:
        for index, entry in enumerate(services):
            if index >= _MAX_SERVICES:
                return None
            if not service_pattern.fullmatch(entry.name) or not entry.is_dir(follow_symlinks=False):
                continue
            group = parent / entry.name
            details = group.stat(follow_symlinks=False)
            if details.st_uid != 0 or details.st_mode & 0o022:
                return None
            with (group / "cgroup.procs").open(encoding="ascii") as stream:
                process_list = stream.read(_MAX_CGROUP_PROCESS_BYTES + 1)
            if len(process_list) > _MAX_CGROUP_PROCESS_BYTES:
                return None
            pids = process_list.splitlines()
            process_count += len(pids)
            if process_count > _MAX_PROCESSES:
                return None
            for pid in pids:
                if not pid.isdecimal():
                    return None
                proc = Path("/proc") / pid
                try:
                    before = (
                        (proc / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()[19]
                    )
                    descriptor = os.readlink(proc / "fd/0")
                    membership = (proc / "cgroup").read_text(encoding="ascii").strip()
                    after = (
                        (proc / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()[19]
                    )
                except (OSError, IndexError):
                    continue
                if (
                    descriptor == f"socket:[{peer}]"
                    and before == after
                    and membership == f"0::/{prefix}.slice/{entry.name}"
                ):
                    matches.append(entry.name)
    if len(matches) != 1 or _peer_inode(connection) != peer:
        return None
    return matches[0]


def _journal_record(
    line: bytes,
    generation: str,
    boot: str,
    started: int,
) -> tuple[str, str] | None:
    """Accept only current-boot kernel AppArmor records with unique structured fields."""
    try:
        event = load_unique_json(line, duplicate_message="Ambiguous journal event.")
        if not isinstance(event, dict) or event.get("_TRANSPORT") != "kernel":
            return None
        if event.get("_BOOT_ID") != boot or int(event["__MONOTONIC_TIMESTAMP"]) < started:
            return None
        if int(event["_SOURCE_MONOTONIC_TIMESTAMP"]) < started:
            return None
        message = event["MESSAGE"]
        if not isinstance(message, str) or not message.startswith("audit: type=1400 audit("):
            return None
        words = shlex.split(message)
        if len(words) < 4 or re.fullmatch(r"audit\(\d+\.\d+:\d+\):", words[2]) is None:
            return None
        fields = {}
        for word in words[3:]:
            key, separator, value = word.partition("=")
            if not separator or key in fields:
                return None
            fields[key] = value
        profile = _PROFILE.fullmatch(fields.get("profile", ""))
        operation = fields.get("operation", "")
        resource = fields.get("name", "")
        if (
            fields.get("apparmor") != "DENIED"
            or fields.get("namespace") != "root//" + generation
            or profile is None
            or _OPERATION.fullmatch(operation) is None
        ):
            return None
        if not resource.startswith("/") or any(
            ord(char) < 32 or ord(char) == 127 for char in resource
        ):
            resource = "<resource unavailable>"
        record = f"apparmor-{operation} {resource}"
        if len(record) > _MAX_DIAGNOSTIC:
            record = record[: _MAX_DIAGNOSTIC - 1] + "…"
        return profile.group(1), record
    except (ValueError, KeyError, TypeError):
        return None


class LinuxDenialObserver:
    """Collect bounded current-attempt kernel journal observations without changing execution."""

    _process: subprocess.Popen[bytes] | None
    _stop: threading.Event
    _lock: threading.Lock
    _events: list[tuple[str, str]]
    _group: str | None
    _generation: str
    _boot: str
    _started: int

    def __init__(self) -> None:
        self._process = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._events = []
        self._group = None
        self._generation = ""
        self._boot = ""
        self._started = 0

    def start(self, connection: socket.socket, generation: str) -> None:
        """Start optional observation for one authenticated live admission connection.

        Args:
            connection (socket.socket): Original connected client socket.
            generation (str): Authenticated installed generation name.
        """
        self._generation = generation
        self._started = int(time.monotonic() * 1_000_000)
        try:
            threading.Thread(target=self._read, daemon=True).start()
            threading.Thread(target=self._identify, args=(connection,), daemon=True).start()
        except RuntimeError:
            self.finish()

    def _identify(self, connection: socket.socket) -> None:
        """Identify a unique live receiver while its original connection remains open."""
        try:
            if self._stop.is_set():
                return
            peer = _peer_inode(connection)
            for _ in range(_IDENTITY_ATTEMPTS):
                if peer is None or self._stop.is_set():
                    return
                group = _receiver_group(connection, self._generation, peer)
                if group is not None:
                    with self._lock:
                        self._group = hashlib.sha256(group.encode()).hexdigest()[:48]
                    return
                self._stop.wait(_IDENTITY_RETRY_SECONDS)
        except (OSError, ValueError, OverflowError, struct.error):
            return

    def _read(self) -> None:
        """Retain only bounded, validated kernel events for this installed generation."""
        process = None
        oversized = False
        try:
            if self._stop.is_set():
                return
            bound = _journal_program()
            if bound is None:
                return
            boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
            self._boot = boot.replace("-", "")
            if re.fullmatch(r"[a-f0-9]{32}", self._boot) is None:
                return
            if self._stop.is_set():
                return
            with bound.open_verified() as descriptor:
                if os.pread(descriptor, 4, 0) != b"\x7fELF":
                    return
                process = subprocess.Popen(
                    [
                        f"/proc/self/fd/{descriptor}",
                        "--no-pager",
                        "--quiet",
                        "--boot",
                        "--dmesg",
                        "--follow",
                        "--lines=0",
                        "--output=json",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    env={"PATH": os.defpath, "LC_ALL": "C"},
                    cwd="/",
                    pass_fds=(descriptor,),
                    close_fds=True,
                    start_new_session=True,
                )
            with self._lock:
                self._process = process
            if self._stop.is_set():
                return
            if process.stdout is None:
                return
            while raw := process.stdout.readline(_MAX_JOURNAL_LINE + 1):
                if oversized:
                    oversized = not raw.endswith(b"\n")
                    continue
                if len(raw) > _MAX_JOURNAL_LINE or not raw.endswith(b"\n"):
                    oversized = not raw.endswith(b"\n")
                    continue
                record = _journal_record(raw, self._generation, self._boot, self._started)
                if record is not None:
                    with self._lock:
                        self._events.append(record)
                        del self._events[:-_MAX_EVENTS]
        except (OSError, ValueError, RuntimeError):
            pass
        finally:
            if process is not None:
                if self._stop.is_set() and process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
                if process.stdout is not None:
                    process.stdout.close()
                process.wait()

    def finish(self) -> tuple[str, str]:
        """Stop observation and return bounded unverified candidates and collection status.

        Returns:
            tuple[str, str]: Candidate and available or unknown status.
        """
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        with self._lock:
            candidates = tuple(
                dict.fromkeys(record for suffix, record in self._events if suffix == self._group)
            )[-_MAX_CANDIDATES:]
        if len(candidates) > 1:
            candidate = DENIAL_RECORDS_PREFIX + json.dumps(
                candidates, ensure_ascii=False, separators=(",", ":")
            )
            while len(candidate) > 2000:
                candidates = candidates[1:]
                candidate = DENIAL_RECORDS_PREFIX + json.dumps(
                    candidates, ensure_ascii=False, separators=(",", ":")
                )
        else:
            candidate = candidates[0] if candidates else ""
        return candidate, "available" if candidate else "unknown"
