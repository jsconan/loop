"""Compile sandbox plans into platform-native process launches."""

from __future__ import annotations

import os
import platform
import struct
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import HostProcessPlan, SandboxPlan


class SandboxUnavailableError(RuntimeError):
    """Indicate that the required native sandbox cannot be enforced."""


_MACOS_BASE_POLICY = """(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow file-write-data (literal \"/dev/null\"))
(allow sysctl-read)
(allow mach-lookup (global-name \"com.apple.system.opendirectoryd.libinfo\"))
"""

_ARM64_DENIED_SYSCALLS = (
    *range(198, 213),  # socket through recvmsg
    39,  # umount2
    40,  # mount
    41,  # pivot_root
    97,  # unshare
    105,  # init_module
    106,  # delete_module
    117,  # ptrace
    142,  # reboot
    217,  # add_key
    218,  # request_key
    219,  # keyctl
    241,  # perf_event_open
    264,  # name_to_handle_at
    265,  # open_by_handle_at
    268,  # setns
    270,  # process_vm_readv
    271,  # process_vm_writev
    273,  # finit_module
    280,  # bpf
    282,  # userfaultfd
    425,  # io_uring_setup
    426,  # io_uring_enter
    427,  # io_uring_register
)
_X86_64_DENIED_SYSCALLS = (
    *range(41, 56),  # socket through getsockopt
    101,  # ptrace
    155,  # pivot_root
    165,  # mount
    166,  # umount2
    169,  # reboot
    175,  # init_module
    176,  # delete_module
    246,  # kexec_load
    248,  # add_key
    249,  # request_key
    250,  # keyctl
    272,  # unshare
    298,  # perf_event_open
    303,  # name_to_handle_at
    304,  # open_by_handle_at
    308,  # setns
    310,  # process_vm_readv
    311,  # process_vm_writev
    313,  # finit_module
    321,  # bpf
    323,  # userfaultfd
    425,  # io_uring_setup
    426,  # io_uring_enter
    427,  # io_uring_register
)
_AUDIT_ARCHITECTURES = {
    "aarch64": (0xC00000B7, _ARM64_DENIED_SYSCALLS),
    "arm64": (0xC00000B7, _ARM64_DENIED_SYSCALLS),
    "x86_64": (0xC000003E, _X86_64_DENIED_SYSCALLS),
    "amd64": (0xC000003E, _X86_64_DENIED_SYSCALLS),
}
_BPF_LOAD_WORD_ABSOLUTE = 0x20
_BPF_JUMP_EQUAL = 0x15
_BPF_RETURN = 0x06
_SECCOMP_ALLOW = 0x7FFF0000
_SECCOMP_ERRNO = 0x00050000
_SECCOMP_KILL_PROCESS = 0x80000000


_LIMIT_LAUNCHER_MODULE = "loop.sandbox.launcher"


def _limited_command(plan: SandboxPlan) -> list[str]:
    """Return the trusted in-sandbox launcher followed by the approved argv."""
    return [sys.executable, "-m", _LIMIT_LAUNCHER_MODULE, "--", *plan.argv]


def _seatbelt_literal(path: Path) -> str:
    """Return a safely quoted Seatbelt string literal."""
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _macos_profile(plan: SandboxPlan) -> str:
    """Compile a deny-by-default Seatbelt profile."""
    readable = {
        Path("/System"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/dev"),
        plan.executable.parent,
        Path(sys.executable).resolve().parent,
        Path(sys.base_prefix).resolve(),
        *plan.readable_roots,
        *plan.read_only_roots,
    }
    if plan.executable == Path(sys.executable).resolve():
        readable.add(Path(sys.base_prefix).resolve())
    writable = set(plan.writable_roots)
    lines = [_MACOS_BASE_POLICY]
    for root in sorted(readable | writable, key=str):
        lines.append(f'(allow file-read* (subpath "{_seatbelt_literal(root)}"))')
    for root in sorted(writable, key=str):
        lines.append(f'(allow file-write* (subpath "{_seatbelt_literal(root)}"))')
    lines.append(f'(deny file-read* (subpath "{_seatbelt_literal(plan.workspace / ".loop")}"))')
    for protected in (
        plan.workspace / ".loop",
        plan.workspace / ".git",
        plan.workspace / ".gitignore",
        plan.workspace / ".agentignore",
    ):
        lines.append(f'(deny file-write* (subpath "{_seatbelt_literal(protected)}"))')
    return "\n".join(lines)


def _linux_parent_directories(path: Path) -> list[str]:
    """Return Bubblewrap directory creation arguments for a bind target."""
    arguments: list[str] = []
    current = Path("/")
    for part in path.parent.parts[1:]:
        current /= part
        arguments.extend(("--dir", str(current)))
    return arguments


_BUBBLEWRAP_LOCATIONS = (
    Path("/usr/bin/bwrap"),
    Path("/bin/bwrap"),
    Path("/usr/local/bin/bwrap"),
)


def _agent_writable(path: Path, writable_roots: tuple[Path, ...]) -> bool:
    """Return whether policy or host permissions let the agent replace a path."""
    if any(path == root or path.is_relative_to(root) for root in writable_roots):
        return True
    metadata = path.stat()
    if metadata.st_mode & 0o002:
        return True
    if metadata.st_uid == os.geteuid() and metadata.st_mode & 0o200:
        return True
    groups = set(os.getgroups()) | {os.getegid()}
    return metadata.st_gid in groups and bool(metadata.st_mode & 0o020)


def _trusted_bubblewrap(writable_roots: tuple[Path, ...] = ()) -> Path:
    """Resolve Bubblewrap only from fixed, non-agent-writable system locations."""
    candidate = next((path for path in _BUBBLEWRAP_LOCATIONS if path.is_file()), None)
    if candidate is None:
        raise SandboxUnavailableError("Bubblewrap is required for sandboxed commands on Linux.")
    candidate = candidate.resolve(strict=True)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise SandboxUnavailableError("The trusted Bubblewrap launcher is not executable.")
    for checked in (candidate, *candidate.parents):
        if _agent_writable(checked, writable_roots):
            raise SandboxUnavailableError(
                "Bubblewrap or one of its parent directories is agent-writable."
            )
    return candidate


def _linux_seccomp_filter() -> bytes:
    """Compile a deny filter for networking and high-risk kernel interfaces."""
    architecture = platform.machine().lower()
    if architecture not in _AUDIT_ARCHITECTURES:
        raise SandboxUnavailableError(
            f"Seccomp filtering is unsupported on Linux architecture {architecture}."
        )
    audit_architecture, denied_syscalls = _AUDIT_ARCHITECTURES[architecture]
    instructions = [
        (_BPF_LOAD_WORD_ABSOLUTE, 0, 0, 4),
        (_BPF_JUMP_EQUAL, 1, 0, audit_architecture),
        (_BPF_RETURN, 0, 0, _SECCOMP_KILL_PROCESS),
        (_BPF_LOAD_WORD_ABSOLUTE, 0, 0, 0),
    ]
    for syscall in denied_syscalls:
        instructions.extend(
            (
                (_BPF_JUMP_EQUAL, 0, 1, syscall),
                (_BPF_RETURN, 0, 0, _SECCOMP_ERRNO | 1),
            )
        )
    instructions.append((_BPF_RETURN, 0, 0, _SECCOMP_ALLOW))
    return b"".join(struct.pack("=HBBI", *instruction) for instruction in instructions)


def _linux_seccomp_descriptor() -> int:
    """Return a sealed-in-practice anonymous descriptor containing the seccomp program."""
    descriptor, path = tempfile.mkstemp(prefix="loop-seccomp-")
    os.unlink(path)
    try:
        os.write(descriptor, _linux_seccomp_filter())
        os.lseek(descriptor, 0, os.SEEK_SET)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _linux_command(plan: SandboxPlan, seccomp_descriptor: int) -> list[str]:
    """Compile a minimal Bubblewrap namespace command."""
    writable_roots = plan.writable_roots
    bwrap = _trusted_bubblewrap(writable_roots)
    arguments = [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--cap-drop",
        "ALL",
        "--seccomp",
        str(seccomp_descriptor),
        "--tmpfs",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
    ]
    for root in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/nix/store"):
        if Path(root).exists():
            arguments.extend(("--ro-bind", root, root))
    arguments.extend(("--dir", "/etc"))
    for path in ("/etc/passwd", "/etc/group", "/etc/nsswitch.conf", "/etc/localtime"):
        if Path(path).exists():
            arguments.extend(("--ro-bind", path, path))
    binding_modes = {
        **{root: True for root in (*plan.readable_roots, *plan.read_only_roots)},
        **{root: False for root in plan.writable_roots},
    }
    bindings = sorted(binding_modes.items(), key=lambda item: len(item[0].parts))
    seen_directories: set[str] = set()
    for root, read_only in bindings:
        for index in range(0, len(_linux_parent_directories(root)), 2):
            directory = _linux_parent_directories(root)[index + 1]
            if directory not in seen_directories:
                arguments.extend(("--dir", directory))
                seen_directories.add(directory)
        arguments.extend(("--ro-bind" if read_only else "--bind", str(root), str(root)))
    for protected in (
        plan.workspace / ".git",
        plan.workspace / ".gitignore",
        plan.workspace / ".agentignore",
    ):
        source = (
            protected if protected.exists() and not protected.is_symlink() else Path("/dev/null")
        )
        arguments.extend(("--ro-bind", str(source), str(protected)))
    control_path = plan.workspace / ".loop"
    if control_path.is_dir() and not control_path.is_symlink():
        arguments.extend(("--tmpfs", str(control_path)))
    else:
        arguments.extend(("--ro-bind", "/dev/null", str(control_path)))
    arguments.extend(("--chdir", str(plan.cwd), "--", *_limited_command(plan)))
    return arguments


def _platform_command(plan: SandboxPlan) -> tuple[list[str], int, tuple[int, ...]]:
    """Return the native wrapper command and process creation flags."""
    system = platform.system()
    if system == "Darwin":
        executable = "/usr/bin/sandbox-exec"
        if not Path(executable).is_file():
            raise SandboxUnavailableError("Seatbelt sandbox-exec is unavailable on macOS.")
        return [executable, "-p", _macos_profile(plan), "--", *_limited_command(plan)], 0, ()
    if system == "Linux":
        writable_roots = plan.writable_roots
        _trusted_bubblewrap(writable_roots)
        descriptor = _linux_seccomp_descriptor()
        try:
            return _linux_command(plan, descriptor), 0, (descriptor,)
        except BaseException:
            os.close(descriptor)
            raise
    if system == "Windows":
        raise SandboxUnavailableError(
            "Windows requires the AppContainer sandbox launcher, which is unavailable."
        )
    raise SandboxUnavailableError(f"Sandboxed commands are unsupported on {system}.")


def sandbox_status() -> tuple[bool, str | None]:
    """Report whether this host exposes the mandatory native sandbox backend.

    Returns:
        tuple[bool, str | None]: Availability and an actionable unavailable reason.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            executable = Path("/usr/bin/sandbox-exec")
            if not executable.is_file():
                return False, "Seatbelt sandbox-exec is unavailable."
            result = subprocess.run(
                [
                    str(executable),
                    "-p",
                    (
                        "(version 1)"
                        "(deny default)"
                        "(allow process-exec)"
                        '(allow file-read* (literal "/usr/bin/true"))'
                    ),
                    "--",
                    "/usr/bin/true",
                ],
                check=False,
                capture_output=True,
                timeout=2,
            )
            if result.returncode != 0:
                return False, "Seatbelt is present but cannot enforce a child sandbox."
            return True, None
        if system == "Linux":
            shell = Path("/bin/bash")
            if not shell.is_file():
                return False, "A trusted Bash executable is required for the Linux sandbox probe."
            with tempfile.TemporaryDirectory(prefix="loop-sandbox-probe-") as raw_directory:
                probe_root = Path(raw_directory).resolve()
                plan = SandboxPlan.create(
                    (
                        str(shell),
                        "-c",
                        "test ! -e /etc/shadow && ! (exec 3<>/dev/tcp/127.0.0.1/1)",
                    ),
                    probe_root,
                    probe_root,
                )
                descriptor = _linux_seccomp_descriptor()
                try:
                    command = _linux_command(plan, descriptor)
                    result = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        timeout=2,
                        pass_fds=(descriptor,),
                    )
                finally:
                    os.close(descriptor)
            if result.returncode != 0:
                return False, "Bubblewrap is installed but cannot create the required namespaces."
            return True, None
    except (OSError, SandboxUnavailableError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if system == "Windows":
        return False, "The Windows AppContainer launcher is not installed."
    return False, f"Sandboxed commands are unsupported on {system}."


def spawn_sandboxed(
    plan: SandboxPlan,
    *,
    popen_options: Mapping[str, Any],
) -> subprocess.Popen[str]:
    """Spawn a process only through a supported native sandbox.

    Args:
        plan (SandboxPlan): Immutable command and containment policy.
        popen_options (Mapping[str, Any]): Pipe and text options owned by the caller.

    Returns:
        subprocess.Popen[str]: Running native sandbox wrapper process.

    Raises:
        SandboxUnavailableError: If the platform cannot enforce the complete policy.
        OSError: If native sandbox process creation fails.
    """
    plan.validate_authority()
    available, reason = sandbox_status()
    if not available:
        raise SandboxUnavailableError(reason or "The native sandbox backend is unavailable.")
    command, creationflags, inherited_descriptors = _platform_command(plan)
    try:
        return subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            shell=False,
            cwd=plan.cwd,
            env=plan.environment,
            start_new_session=os.name == "posix",
            creationflags=creationflags,
            pass_fds=inherited_descriptors,
            **dict(popen_options),
        )
    finally:
        for descriptor in inherited_descriptors:
            os.close(descriptor)


def spawn_host(
    plan: HostProcessPlan,
    *,
    popen_options: Mapping[str, Any],
) -> subprocess.Popen[str]:
    """Spawn one freshly confirmed host plan without shell interpretation.

    Args:
        plan (HostProcessPlan): Exact separately confirmed host authority.
        popen_options (Mapping[str, Any]): Pipe and text options owned by the caller.

    Returns:
        subprocess.Popen[str]: Running host process.

    Raises:
        OSError: If authority validation or process creation fails.
        ValueError: If executable identity changed after confirmation.
    """
    plan.validate_authority()
    return subprocess.Popen(  # pylint: disable=consider-using-with
        _limited_command(plan),
        shell=False,
        cwd=plan.cwd,
        env=plan.environment,
        start_new_session=os.name == "posix",
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        close_fds=True,
        **dict(popen_options),
    )
