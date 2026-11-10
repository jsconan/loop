"""Exercise disconnect ownership around a deliberately stalled native spawn."""

import errno
import os
import select
import socket
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

from loop.execution.sandbox.linux import supervision
from loop.execution.sandbox.linux.mountinfo import read_descriptor_mount_ids, read_mountinfo

pytestmark = [
    pytest.mark.integration,
    pytest.mark.linux,
    pytest.mark.skipif(
        os.environ.get("LOOP_NATIVE_WATCHDOG_TEST") != "disposable-linux-host",
        reason="Set LOOP_NATIVE_WATCHDOG_TEST=disposable-linux-host for owned user services",
    ),
]

_SERVICE = r"""
import os, socket, subprocess, sys
sys.path.insert(0, sys.argv[1])
from loop.execution.sandbox.linux import supervision
if sys.argv[6] == "failure":
    original_poll = supervision.select.poll
    class FailingPoll:
        def __init__(self):
            self.poller = original_poll()
        def register(self, *args):
            self.poller.register(*args)
        def poll(self, timeout):
            events = self.poller.poll(timeout)
            if os.path.exists(sys.argv[7]):
                raise OSError("synthetic watcher failure")
            return events
    supervision.select.poll = FailingPoll
connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
connection.connect(sys.argv[2])
if sys.argv[6] == "wrong-sender":
    sender = os.fork()
    if sender == 0:
        connection.sendall(b"R")
        os._exit(0)
    os.waitpid(sender, 0)
else:
    connection.sendall(b"R")
    if sys.argv[6] == "mixed-sender":
        sender = os.fork()
        if sender == 0:
            connection.sendall(b"X")
            os._exit(0)
        os.waitpid(sender, 0)
if connection.recv(1) != b"G":
    connection.close()
    sys.exit(0)
def stall():
    fd = os.open(sys.argv[4], os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    os.close(fd)
    fd = os.open(sys.argv[3], os.O_RDONLY)
    os.read(fd, 1)
    os.close(fd)
with supervision.guard_client_disconnect(connection) as spawn_returned:
    code = "from pathlib import Path;Path(%r).write_text('executed')" % sys.argv[5]
    process = subprocess.Popen([sys.executable, "-I", "-c", code], preexec_fn=stall,
                               start_new_session=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    spawn_returned.set()
    if process.wait(timeout=5) != 0:
        raise RuntimeError("Synthetic child failed")
    connection.sendall(b"D")
    if sys.argv[6] in ("cancel", "cancel-halfclose"):
        connection.sendall(connection.recv(1))
connection.close()
"""


def user_manager_environment():
    """Bind a private, correctly owned user-manager runtime and bus."""
    runtime = Path("/run/user") / str(os.getuid())
    details = runtime.lstat()
    bus = runtime / "bus"
    assert runtime.resolve(strict=True) == runtime
    assert stat.S_ISDIR(details.st_mode) and details.st_uid == os.getuid()
    assert stat.S_IMODE(details.st_mode) == 0o700
    assert stat.S_ISSOCK(bus.lstat().st_mode) and bus.lstat().st_uid == os.getuid()
    return os.environ | {
        "XDG_RUNTIME_DIR": str(runtime),
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={bus}",
        "LC_ALL": "C",
    }


def test_kernel_refuses_forged_sender_pid():
    """An unprivileged sender cannot claim another process's socket credentials."""
    code = """
import errno, os, socket, struct, sys
connection = socket.socket(fileno=int(sys.argv[1]))
claimed = struct.pack('3i', int(sys.argv[2]), os.getuid(), os.getgid())
try:
    connection.sendmsg([b'forged'], [(socket.SOL_SOCKET, socket.SCM_CREDENTIALS, claimed)])
except OSError as error:
    if error.errno != errno.EPERM:
        raise
else:
    raise RuntimeError('kernel accepted a foreign sender PID')
connection.sendall(b'X')
"""
    receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    receiver.settimeout(6)
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", code, str(sender.fileno()), str(os.getpid())],
            pass_fds=(sender.fileno(),),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        data, ancillary, flags, _ = receiver.recvmsg(64, socket.CMSG_SPACE(struct.calcsize("3i")))
        assert data == b"X" and not flags & socket.MSG_CTRUNC
        assert len(ancillary) == 1
        level, kind, raw = ancillary[0]
        assert (level, kind) == (socket.SOL_SOCKET, socket.SCM_CREDENTIALS)
        assert struct.unpack("3i", raw) == (process.pid, os.getuid(), os.getgid())
        stdout, stderr = process.communicate(timeout=6)
        assert process.returncode == 0 and stdout == b"" and stderr == b""
    finally:
        sender.close()
        receiver.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=6)


@pytest.mark.parametrize(
    "control",
    [
        "halfclose",
        "cancel",
        "cancel-halfclose",
        "cancel-stalled",
        "cancel-halfclose-stalled",
        "disconnect",
        "failure",
        "wrong-sender",
        "mixed-sender",
    ],
)
def test_disconnect_guard_covers_stalled_spawn_in_owned_systemd_scope(tmp_path, control):
    """Bind pre-launch identity and drain stalled work on disconnect or observer failure."""
    source = tmp_path / "service.py"
    source.write_text(_SERVICE, encoding="utf-8")
    fifo = tmp_path / "stall"
    os.mkfifo(fifo, 0o600)
    started = tmp_path / "started"
    effect = tmp_path / "effect"
    failure = tmp_path / "watcher-failure"
    socket_path = tmp_path / "socket"
    unit = "loop-spawn-guard-test-" + uuid4().hex + ".service"
    manager = ["/usr/bin/systemctl", "--user"]
    environment = user_manager_environment()
    subprocess.run(
        manager + ["show-environment"], check=True, capture_output=True, timeout=6, env=environment
    )
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.settimeout(6)
    server.bind(str(socket_path))
    server.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    server.listen(1)
    process_directory = None
    group_directory = None
    group_events = None
    try:
        subprocess.run(
            [
                "/usr/bin/systemd-run",
                "--user",
                "--quiet",
                "--unit=" + unit,
                "--property=ExitType=main",
                "--property=KillMode=control-group",
                "--property=KillSignal=SIGKILL",
                "--property=SendSIGKILL=yes",
                sys.executable,
                "-I",
                str(source),
                str(Path(supervision.__file__).resolve().parents[4]),
                str(socket_path),
                str(fifo),
                str(started),
                str(effect),
                control,
                str(failure),
            ],
            check=True,
            timeout=6,
            env=environment,
        )
        connection, _ = server.accept()
        connection.settimeout(6)
        with connection:
            ready, ancillary, flags, _ = connection.recvmsg(
                1, socket.CMSG_SPACE(struct.calcsize("3i"))
            )
            assert ready == b"R" and not flags & socket.MSG_CTRUNC
            assert len(ancillary) == 1
            level, kind, raw_credentials = ancillary[0]
            assert (level, kind) == (socket.SOL_SOCKET, socket.SCM_CREDENTIALS)
            sender, uid, gid = struct.unpack("3i", raw_credentials)
            assert (uid, gid) == (os.getuid(), os.getgid())
            output = subprocess.run(
                manager
                + ["show", unit, "--property=ControlGroup,MainPID,ExitType,KillMode,KillSignal"],
                check=True,
                text=True,
                capture_output=True,
                timeout=6,
                env=environment,
            ).stdout
            properties = dict(line.split("=", 1) for line in output.splitlines())
            assert properties["ExitType"] == "main"
            assert properties["KillMode"] == "control-group"
            assert properties["KillSignal"] == "9"
            main = int(properties["MainPID"])
            assert main > 0
            group = Path("/sys/fs/cgroup") / properties["ControlGroup"].lstrip("/")
            members = {int(value) for value in (group / "cgroup.procs").read_text().split()}
            assert main in members
            if control in {"wrong-sender", "mixed-sender"}:
                if control == "wrong-sender":
                    assert sender != main
                else:
                    assert sender == main
                    extra, credentials, extra_flags, _ = connection.recvmsg(
                        1, socket.CMSG_SPACE(struct.calcsize("3i"))
                    )
                    assert extra == b"X" and not extra_flags & socket.MSG_CTRUNC
                    assert len(credentials) == 1
                    level, kind, raw = credentials[0]
                    assert (level, kind) == (socket.SOL_SOCKET, socket.SCM_CREDENTIALS)
                    other, other_uid, other_gid = struct.unpack("3i", raw)
                    assert other != sender and (other_uid, other_gid) == (uid, gid)
                connection.close()
                assert not started.exists() and not effect.exists()
            else:
                assert sender == main
                group_directory = os.open(
                    group, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
                )
                original_group = os.fstat(group_directory)
                mount_ids = read_descriptor_mount_ids(group_directory)
                mounts = read_mountinfo()
                matching = [row for row in mounts if [row[0]] in mount_ids]
                assert len(mount_ids) == len(matching) == 1
                assert matching[0][3:5] == ["/", "/sys/fs/cgroup"]
                assert matching[0][matching[0].index("-") + 1] == "cgroup2"
                assert not any(Path(row[4]).is_relative_to(group) for row in mounts)
                group_events = os.open(
                    "cgroup.events",
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=group_directory,
                )
                assert b"populated 1\n" in os.pread(group_events, 4096, 0)
                process_directory = os.open(
                    Path("/proc") / str(sender),
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                )
                original_process = os.fstat(process_directory)
                assert original_process.st_uid == os.getuid()
                membership = os.open(
                    "cgroup", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=process_directory
                )
                try:
                    assert (
                        os.read(membership, 4096).strip()
                        == ("0::" + properties["ControlGroup"]).encode()
                    )
                finally:
                    os.close(membership)
                connection.sendall(b"G")
                deadline = time.monotonic() + 5
                while not started.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert started.exists(), "Synthetic child did not enter the stalled spawn"
                members = {int(value) for value in (group / "cgroup.procs").read_text().split()}
                assert main in members and len(members) >= 2
            if control in {
                "disconnect",
                "failure",
                "cancel-stalled",
                "cancel-halfclose-stalled",
                "wrong-sender",
                "mixed-sender",
            }:
                if control == "disconnect":
                    connection.close()
                elif control == "failure":
                    failure.touch()
                elif control in {"cancel-stalled", "cancel-halfclose-stalled"}:
                    connection.sendall(b"\x00")
                    if control == "cancel-halfclose-stalled":
                        connection.shutdown(socket.SHUT_WR)
                deadline = time.monotonic() + 5
                remaining = "unknown"
                while time.monotonic() < deadline:
                    try:
                        remaining = (group / "cgroup.procs").read_text().strip()
                    except FileNotFoundError:
                        remaining = ""
                    if not remaining:
                        break
                    time.sleep(0.02)
                assert not remaining, "Synthetic service scope did not drain"
                assert not effect.exists()
                if group_directory is not None:
                    current_group = os.fstat(group_directory)
                    assert (current_group.st_dev, current_group.st_ino) == (
                        original_group.st_dev,
                        original_group.st_ino,
                    )
                    event_deadline = time.monotonic() + 5
                    while True:
                        try:
                            events = os.pread(group_events, 4096, 0)
                        except OSError as error:
                            assert error.errno == errno.ENODEV and not group.exists()
                            break
                        if b"populated 0\n" in events:
                            break
                        if time.monotonic() >= event_deadline:
                            pytest.fail("Synthetic service cgroup did not report idle")
                        time.sleep(0.02)
                    try:
                        latest_group = group.stat()
                    except FileNotFoundError:
                        latest_group = None
                    if latest_group is not None:
                        assert (latest_group.st_dev, latest_group.st_ino) == (
                            original_group.st_dev,
                            original_group.st_ino,
                        )
                if control == "failure":
                    assert connection.recv(1) == b""
                elif control in {"cancel-stalled", "cancel-halfclose-stalled"}:
                    try:
                        response = connection.recv(1)
                    except ConnectionResetError as error:
                        assert error.errno == errno.ECONNRESET
                    else:
                        assert response == b""
                if process_directory is not None:
                    deadline = time.monotonic() + 5
                    while True:
                        try:
                            membership = os.open(
                                "cgroup",
                                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=process_directory,
                            )
                        except ProcessLookupError as error:
                            assert error.errno == errno.ESRCH
                            break
                        else:
                            os.close(membership)
                            assert time.monotonic() < deadline, "Receiver identity remained live"
                            time.sleep(0.02)
                    current_process = os.fstat(process_directory)
                    assert (current_process.st_dev, current_process.st_ino) == (
                        original_process.st_dev,
                        original_process.st_ino,
                    )
            else:
                if control == "halfclose":
                    connection.shutdown(socket.SHUT_WR)
                else:
                    connection.sendall(b"\x00")
                    if control == "cancel-halfclose":
                        connection.shutdown(socket.SHUT_WR)
                readable, _, _ = select.select([connection], [], [], 0.2)
                assert not readable, (
                    "The guard stopped while the stalled client was still connected"
                )
                descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                try:
                    os.write(descriptor, b"x")
                finally:
                    os.close(descriptor)
                assert connection.recv(1) == b"D"
                if control in {"cancel", "cancel-halfclose"}:
                    assert connection.recv(1) == b"\x00"
                assert effect.read_text(encoding="utf-8") == "executed"
    finally:
        server.close()
        if process_directory is not None:
            os.close(process_directory)
        if group_events is not None:
            os.close(group_events)
        if group_directory is not None:
            os.close(group_directory)
        stop = subprocess.run(
            manager + ["stop", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
            env=environment,
        )
        if stop.returncode and not (
            stop.returncode == 5 and f"Unit {unit} not loaded." in stop.stderr
        ):
            raise RuntimeError(f"Could not stop synthetic service: {stop.stderr}")
        reset = subprocess.run(
            manager + ["reset-failed", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
            env=environment,
        )
        if reset.returncode and "not loaded" not in reset.stderr:
            raise RuntimeError(f"Could not reset synthetic service: {reset.stderr}")


def test_service_cgroup_path_can_be_reused_after_original_removal():
    """A reused path can name a live replacement while a pinned old cgroup is gone."""
    unit = "loop-cgroup-reuse-test-" + uuid4().hex + ".service"
    manager = ["/usr/bin/systemctl", "--user"]
    environment = user_manager_environment()
    starter = [
        "/usr/bin/systemd-run",
        "--user",
        "--quiet",
        "--unit=" + unit,
        "/usr/bin/sleep",
        "20",
    ]
    directory = None
    events = None
    try:
        subprocess.run(starter, check=True, capture_output=True, timeout=6, env=environment)
        output = subprocess.run(
            manager + ["show", unit, "--property=ControlGroup", "--value"],
            check=True,
            text=True,
            capture_output=True,
            timeout=6,
            env=environment,
        ).stdout.strip()
        assert output.startswith("/") and output.endswith("/" + unit)
        group = Path("/sys/fs/cgroup") / output.lstrip("/")
        directory = os.open(group, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        original = os.fstat(directory)
        events = os.open(
            "cgroup.events", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory
        )
        assert b"populated 1\n" in os.pread(events, 4096, 0)
        subprocess.run(
            manager + ["stop", unit], check=True, capture_output=True, timeout=6, env=environment
        )
        deadline = time.monotonic() + 5
        while group.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not group.exists()
        try:
            old_events = os.pread(events, 4096, 0)
        except OSError as error:
            assert error.errno == errno.ENODEV
        else:
            assert b"populated 0\n" in old_events

        deadline = time.monotonic() + 5
        while True:
            restarted = subprocess.run(
                starter, check=False, capture_output=True, text=True, timeout=6, env=environment
            )
            if restarted.returncode == 0:
                break
            if "already exists" not in restarted.stderr or time.monotonic() >= deadline:
                restarted.check_returncode()
            time.sleep(0.02)
        replacement = group.stat()
        assert (replacement.st_dev, replacement.st_ino) != (original.st_dev, original.st_ino)
        assert b"populated 1\n" in (group / "cgroup.events").read_bytes()
    finally:
        if events is not None:
            os.close(events)
        if directory is not None:
            os.close(directory)
        stop = subprocess.run(
            manager + ["stop", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
            env=environment,
        )
        if stop.returncode and not (
            stop.returncode == 5 and f"Unit {unit} not loaded." in stop.stderr
        ):
            raise RuntimeError(f"Could not stop synthetic service: {stop.stderr}")
        reset = subprocess.run(
            manager + ["reset-failed", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
            env=environment,
        )
        if reset.returncode and "not loaded" not in reset.stderr:
            raise RuntimeError(f"Could not reset synthetic service: {reset.stderr}")
