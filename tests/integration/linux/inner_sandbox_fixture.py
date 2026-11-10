"""Exercise a nested Linux seccomp restriction inside Loop's installed outer sandbox."""

import errno
import json
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path


def _filter_count():
    """Read the kernel's current count of inherited seccomp filters."""
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    return int(
        next(line.split()[1] for line in status.splitlines() if line.startswith("Seccomp_filters:"))
    )


def _errno(operation):
    """Return the errno from a denied operation, or zero if it unexpectedly succeeds."""
    try:
        operation()
    except OSError as error:
        return error.errno
    return 0


def _load_inner_filter():
    """Layer an inherited socket-creation denial over Loop's existing restrictions."""
    import ctypes

    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = (ctypes.c_uint32,)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    )
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = (ctypes.c_void_p,)
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = (ctypes.c_void_p,)

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "Cannot set no_new_privs for inner filter")
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("Cannot allocate inner seccomp filter")
    try:
        number = library.seccomp_syscall_resolve_name(b"socket")
        if number < 0 or library.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0):
            raise RuntimeError("Cannot add inner socket restriction")
        result = library.seccomp_load(context)
        if result:
            raise OSError(-result, "Cannot install inner seccomp filter")
    finally:
        library.seccomp_release(context)


def _open_unix_socket():
    """Try a Unix endpoint without retaining one if the outer filter permits it."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
        pass


def _connect_host_loopback(port):
    """Try only the owned test listener in the host's loopback namespace."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        connection.connect(("127.0.0.1", int(port)))


def _connect_host_unix(path):
    """Try only the test-owned Unix listener outside the sandbox mount view."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(path)


def _receive_descriptor(path):
    """Attempt ancillary descriptor receipt through an owned connected socketpair."""
    sender, receiver = socket.socketpair()
    with sender, receiver, path.open("rb") as source:
        sender.sendmsg(
            [b"x"],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", source.fileno()))],
        )
        receiver.recvmsg(1, socket.CMSG_SPACE(struct.calcsize("i")))


def _unexpected_descriptors():
    """List every inherited descriptor beyond the process's intended standard streams."""
    targets = []
    for name in os.listdir("/proc/self/fd"):
        if int(name) <= 2:
            continue
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        targets.append(target)
    return targets


def _open_cgroup_control():
    """Try opening the cgroup membership control without writing a process ID."""
    with Path("/sys/fs/cgroup/cgroup.procs").open("w"):
        pass


def _adversarial_checks(workspace, bwrap):
    """Probe blocked namespace and mount controls only on disposable Linux hosts."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.unshare.argtypes = (ctypes.c_int,)
    libc.unshare.restype = ctypes.c_int
    libc.setns.argtypes = (ctypes.c_int, ctypes.c_int)
    libc.setns.restype = ctypes.c_int
    libc.mount.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    )
    libc.mount.restype = ctypes.c_int

    def attempt(call):
        """Return the native errno, or zero when an outer control unexpectedly allows a change."""
        return 0 if call() == 0 else ctypes.get_errno()

    target = workspace / "inner-mount"
    target.mkdir()
    userns_errno = attempt(lambda: libc.unshare(0x10000000))  # CLONE_NEWUSER
    if userns_errno == 0:
        raise RuntimeError("Outer sandbox unexpectedly permitted a user namespace")
    try:
        with Path("/proc/self/ns/mnt").open("rb") as namespace:
            namespace_handle_errno = 0
            setns_errno = attempt(lambda: libc.setns(namespace.fileno(), 0))
            if setns_errno == 0:
                raise RuntimeError("Outer sandbox unexpectedly permitted setns")
    except OSError as error:
        namespace_handle_errno = error.errno
        setns_errno = None
    mount_errno = attempt(lambda: libc.mount(b"tmpfs", os.fsencode(target), b"tmpfs", 0, None))
    if mount_errno == 0:
        raise RuntimeError("Outer sandbox unexpectedly permitted an inner mount")
    remount_errno = attempt(
        lambda: libc.mount(None, os.fsencode(workspace), None, 32 | 4096 | 1, None)
    )  # MS_REMOUNT | MS_BIND | MS_RDONLY
    if remount_errno == 0:
        raise RuntimeError("Outer sandbox unexpectedly permitted remounting the workspace")
    version = subprocess.run([bwrap, "--version"], capture_output=True, text=True, check=False)
    if version.returncode != 0 or not version.stdout.startswith("bubblewrap "):
        raise RuntimeError("Copied Bubblewrap executable failed its version control")
    nested = subprocess.run(
        [
            bwrap,
            "--unshare-user",
            "--uid",
            "0",
            "--gid",
            "0",
            "--ro-bind",
            "/",
            "/",
            "/bin/sh",
            "-c",
            "printf unexpected-inner-bwrap",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if nested.returncode == 0 or "unexpected-inner-bwrap" in nested.stdout:
        raise RuntimeError("Nested Bubblewrap unexpectedly gained namespace authority")
    cgroup_write_errno = _errno(_open_cgroup_control)
    if cgroup_write_errno == 0:
        raise RuntimeError("Outer sandbox unexpectedly exposed cgroup control")
    apparmor = ctypes.CDLL("libapparmor.so.1", use_errno=True)
    apparmor.aa_change_profile.argtypes = (ctypes.c_char_p,)
    apparmor.aa_change_profile.restype = ctypes.c_int
    profile_change_errno = attempt(lambda: apparmor.aa_change_profile(b"unconfined"))
    if profile_change_errno == 0:
        raise RuntimeError("Outer sandbox unexpectedly permitted profile replacement")
    detached = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 30"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "userns_errno": userns_errno,
        "namespace_handle_errno": namespace_handle_errno,
        "setns_errno": setns_errno,
        "mount_errno": mount_errno,
        "remount_errno": remount_errno,
        "bwrap_refused": nested.returncode != 0 and "unexpected-inner-bwrap" not in nested.stdout,
        "bwrap_version": version.stdout.strip(),
        "cgroup_write_errno": cgroup_write_errno,
        "profile_change_errno": profile_change_errno,
        "detached_started": detached.pid > 0,
    }


def main():
    """Run useful child work, then report inner and outer boundary checks."""
    workspace, outside, control, host_pid, host_port, mode, bwrap = sys.argv[1:]
    workspace = Path(workspace)
    outside = Path(outside)
    control = Path(control)
    inherited_descriptors = _unexpected_descriptors()
    before = _filter_count()
    profile = Path("/proc/self/attr/current").read_text(encoding="utf-8")
    outer_unix_errno = _errno(_open_unix_socket)
    cross_session_unix_errno = _errno(lambda: _connect_host_unix(str(outside / "peer.sock")))
    outer_loopback_errno = _errno(lambda: _connect_host_loopback(host_port))
    _load_inner_filter()
    after = _filter_count()
    child = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-B",
            "-c",
            (
                "import os,pathlib,socket,sys\n"
                "for name in os.listdir('/proc/self/fd'):\n"
                " if int(name)>2:\n"
                "  try: os.readlink('/proc/self/fd/'+name)\n"
                "  except OSError: pass\n"
                "  else: raise RuntimeError('inherited descriptor escaped')\n"
                "assert pathlib.Path('/proc/self/attr/current').read_text()==sys.argv[3]\n"
                "for value in sys.argv[1:3]:\n"
                " try: pathlib.Path(value).read_bytes()\n"
                " except OSError: pass\n"
                " else: raise RuntimeError('outer read escaped')\n"
                "p=pathlib.Path('allowed'); p.write_text('inner-work')\n"
                "assert p.read_text()=='inner-work'\n"
                "print('child-ok')\n"
                "socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
            ),
            str(workspace / ".ssh" / "key"),
            str(outside / "secret"),
            profile,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    checks = {
        "filter_before": before,
        "filter_after": after,
        "outer_unix_errno": outer_unix_errno,
        "cross_session_unix_errno": cross_session_unix_errno,
        "outer_loopback_errno": outer_loopback_errno,
        "profile_unchanged": Path("/proc/self/attr/current").read_text(encoding="utf-8") == profile,
        "child_output": child.stdout,
        "child_socket_denied": child.returncode != 0 and "Operation not permitted" in child.stderr,
        "allowed_content": (workspace / "allowed").read_text(encoding="utf-8"),
        "inner_socket_errno": _errno(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
        "private_read_errno": _errno(lambda: (workspace / ".ssh" / "key").read_bytes()),
        "outside_read_errno": _errno(lambda: (outside / "secret").read_bytes()),
        "outside_write_errno": _errno(lambda: (outside / "write").write_text("changed")),
        "control_write_errno": _errno(lambda: control.write_text("changed")),
        "host_signal_errno": _errno(lambda: os.kill(int(host_pid), 0)),
        "descriptor_receive_errno": _errno(lambda: _receive_descriptor(workspace / "allowed")),
        "unexpected_descriptors": inherited_descriptors,
    }
    if mode == "adversarial":
        checks.update(_adversarial_checks(workspace, bwrap))
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
