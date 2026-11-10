"""Export native IPC restrictions through protected installed libseccomp."""

import ctypes
import platform
import tempfile
import time

from .host import library_candidates

NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True
NATIVE_SCOPED_SOCKET_FAMILIES_V1 = True

_ALLOW = 0x7FFF0000
_KILL_PROCESS = 0x80000000
_ERRNO = 0x00050000
_BAD_ARCH = 2  # SCMP_FLTATR_ACT_BADARCH, part of libseccomp's public API.
_COMPARE_MASKED_EQUAL = 7  # SCMP_CMP_MASKED_EQ.
_LINUX_INT_MASK = 0xFFFFFFFF
_LINUX_AF_UNIX = 1
_LINUX_AF_INET = 2
_LINUX_AF_INET6 = 10
_LINUX_AF_NETLINK = 16
_LINUX_SOCK_STREAM = 1
_LINUX_SOCKET_TYPE_BITS = (1, 2, 8)
_LINUX_INTEGER_BITS = 32
# Linux errno ABI values; host Python errno constants can differ on macOS.
_LINUX_EPERM = 1
_LINUX_ENOSYS = 38
_MAX_FILTER_INSTRUCTIONS = 4096
_BPF_INSTRUCTION_BYTES = 8
_MAX_FILTER_BYTES = _MAX_FILTER_INSTRUCTIONS * _BPF_INSTRUCTION_BYTES
_NATIVE_NAMES = {
    "aarch64": b"aarch64",
    "arm64": b"aarch64",
    "x86_64": b"x86_64",
    "amd64": b"x86_64",
}
_DENIED_SYSCALLS = (
    (b"recvmsg", _LINUX_EPERM),
    (b"recvmmsg", _LINUX_EPERM),
    (b"io_uring_setup", _LINUX_ENOSYS),
    (b"io_uring_enter", _LINUX_ENOSYS),
    (b"io_uring_register", _LINUX_ENOSYS),
)


class ArgumentComparison(ctypes.Structure):
    """Represent libseccomp's public scmp_arg_cmp ABI."""

    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def _load_library(deadline) -> ctypes.CDLL:
    """Load a protected native library and bind the required public API signatures."""
    for path in library_candidates("libseccomp", deadline=deadline):
        try:
            library = ctypes.CDLL(str(path))
        except OSError:
            # A protected multilib entry may belong to another ELF ABI.
            continue
        signatures = {
            "seccomp_arch_native": ([], ctypes.c_uint32),
            "seccomp_arch_resolve_name": ([ctypes.c_char_p], ctypes.c_uint32),
            "seccomp_init": ([ctypes.c_uint32], ctypes.c_void_p),
            "seccomp_release": ([ctypes.c_void_p], None),
            "seccomp_attr_set": (
                [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint32],
                ctypes.c_int,
            ),
            "seccomp_syscall_resolve_name_arch": (
                [ctypes.c_uint32, ctypes.c_char_p],
                ctypes.c_int,
            ),
            "seccomp_rule_add_exact_array": (
                [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_int,
                    ctypes.c_uint,
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
            "seccomp_export_bpf": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        }
        try:
            for name, (arguments, returns) in signatures.items():
                function = getattr(library, name)
                function.argtypes = arguments
                function.restype = returns
        except AttributeError as error:
            raise OSError("Required libseccomp API is unavailable.") from error
        return library

    raise OSError("Installed libseccomp is unavailable.")


def _require_result(result: int) -> None:
    """Refuse failed native resolver or export operations before returning any filter."""
    if result != 0:
        raise OSError(-result, "Native seccomp operation failed.")


def descriptor_filter(*, deadline: float | None = None) -> bytes:
    """Build a native 64-bit filter using installed libseccomp ABI resolution and export.

    Support native ARM64 and x86-64, including their platform-name aliases. Only the native
    architecture is included: foreign/32-bit ABIs and x32-number aliases are killed.
    recvmsg/recvmmsg return EPERM; io_uring setup/enter/register return ENOSYS. Socket creation
    permits IPv4, IPv6 and NETLINK_ROUTE only. Socket pairs permit only already-connected Unix
    stream and sequenced-packet pairs. Other socket families and netlink protocols return EPERM.
    Other native calls retain the surrounding namespace, AppArmor and capability restrictions.
    The caller must authenticate and install these bytes through Bubblewrap before any
    command code, then qualify enforcement. This builder neither installs a filter nor
    establishes backend readiness. Administrator compromise and an already compromised
    dynamic loader are outside its trust boundary.

    Args:
        deadline (float | None): Optional absolute monotonic preparation budget.

    Returns:
        bytes: Native classic BPF instructions accepted by Bubblewrap's --seccomp.

    Raises:
        ValueError: Host ABI, discovered authority, syscall mapping or exported bytes are unsafe.
        OSError: Protected host discovery, a required libseccomp API, allocation or export fails.
        TimeoutError: The preparation budget expires.
    """
    architecture_name = _NATIVE_NAMES.get(platform.machine().lower())
    if platform.system() != "Linux" or architecture_name is None:
        raise ValueError("Unsupported native descriptor-filter ABI.")
    library = _load_library(deadline)
    architecture = library.seccomp_arch_resolve_name(architecture_name)
    if not architecture or architecture != library.seccomp_arch_native():
        raise ValueError("Native libseccomp ABI does not match the host.")
    context = library.seccomp_init(_ALLOW)
    if not context:
        raise OSError("Native seccomp allocation failed.")
    try:
        _require_result(library.seccomp_attr_set(context, _BAD_ARCH, _KILL_PROCESS))
        for name, error in _DENIED_SYSCALLS:
            number = library.seccomp_syscall_resolve_name_arch(architecture, name)
            if number < 0:
                raise ValueError("Required native seccomp syscall is unavailable.")
            _require_result(
                library.seccomp_rule_add_exact_array(context, _ERRNO | error, number, 0, None)
            )
        # The kernel truncates domain and protocol to int, regardless of high register bits.
        unix = ArgumentComparison(0, _COMPARE_MASKED_EQUAL, _LINUX_INT_MASK, _LINUX_AF_UNIX)
        netlink = ArgumentComparison(0, _COMPARE_MASKED_EQUAL, _LINUX_INT_MASK, _LINUX_AF_NETLINK)
        for name in (b"socket", b"socketpair"):
            number = library.seccomp_syscall_resolve_name_arch(architecture, name)
            if number < 0:
                raise ValueError("Required native seccomp syscall is unavailable.")
            if name == b"socket":
                # The only admitted domains are 2 (IPv4), 10 (IPv6), and 16 (netlink).
                # Their low-bit patterns are distinguished below; every other bit must be zero.
                masks = [
                    (1 << bit, 1 << bit)
                    for bit in range(_LINUX_INTEGER_BITS)
                    if bit not in (1, 3, 4)
                ]
                masks.extend(((18, 0), (18, 18), (24, 24)))
                for mask, value in masks:
                    comparison = ArgumentComparison(0, _COMPARE_MASKED_EQUAL, mask, value)
                    _require_result(
                        library.seccomp_rule_add_exact_array(
                            context, _ERRNO | _LINUX_EPERM, number, 1, ctypes.byref(comparison)
                        )
                    )
                # Only NETLINK_ROUTE (protocol 0) is needed for interface discovery. Other
                # protocols can communicate with host processes in the host network namespace.
                for bit in range(_LINUX_INTEGER_BITS):
                    mask = 1 << bit
                    comparisons = (ArgumentComparison * 2)(
                        netlink, ArgumentComparison(2, _COMPARE_MASKED_EQUAL, mask, mask)
                    )
                    _require_result(
                        library.seccomp_rule_add_exact_array(
                            context, _ERRNO | _LINUX_EPERM, number, 2, comparisons
                        )
                    )
            else:
                # Only AF_UNIX socket pairs are admitted, independent of kernel support for
                # other families. Reject every bit pattern except the low 32-bit value 1.
                for bit in range(_LINUX_INTEGER_BITS):
                    mask = 1 << bit
                    unexpected = 0 if bit == 0 else mask
                    comparison = ArgumentComparison(0, _COMPARE_MASKED_EQUAL, mask, unexpected)
                    _require_result(
                        library.seccomp_rule_add_exact_array(
                            context, _ERRNO | _LINUX_EPERM, number, 1, ctypes.byref(comparison)
                        )
                    )
                # Allow only STREAM (1) or SEQPACKET (5) in the low type bits; the kernel
                # separately validates flags and ignores high register bits.
                for bit in _LINUX_SOCKET_TYPE_BITS:
                    unexpected = 0 if bit == _LINUX_SOCK_STREAM else bit
                    comparisons = (ArgumentComparison * 2)(
                        unix, ArgumentComparison(1, _COMPARE_MASKED_EQUAL, bit, unexpected)
                    )
                    _require_result(
                        library.seccomp_rule_add_exact_array(
                            context, _ERRNO | _LINUX_EPERM, number, 2, comparisons
                        )
                    )
        with tempfile.TemporaryFile() as output:
            _require_result(library.seccomp_export_bpf(context, output.fileno()))
            output.seek(0)
            program = output.read(_MAX_FILTER_BYTES + 1)
        if not program or len(program) > _MAX_FILTER_BYTES or len(program) % _BPF_INSTRUCTION_BYTES:
            raise ValueError("Invalid native seccomp filter export.")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Native descriptor-filter deadline expired.")
        return program
    finally:
        library.seccomp_release(context)


def local_network_filter(*, mediated_egress: bool, deadline: float | None = None) -> bytes:
    """Compile a payload filter before host authority is removed in the private namespace.

    The outer namespace filter already denies Unix socket creation and descriptor receipt.
    This additional inherited filter denies all addressable listeners and, without a mediator,
    all IP sockets. With a mediator it admits only IPv4 stream clients, whose sole listener in
    the private network namespace is the trusted attempt proxy. It must be loaded after the
    trusted proxy forks, before command source runs, with NoNewPrivs and no capabilities.

    Args:
        mediated_egress (bool): Retain IPv4 stream clients for the attempt-bound proxy.
        deadline (float | None): Optional absolute host preparation deadline.

    Returns:
        bytes: Native classic BPF to install after trusted proxy setup.

    Raises:
        TypeError: A non-boolean mode was supplied.
        ValueError: The native architecture is unsupported.
        OSError: Protected libseccomp, rule creation, or filter export failed.
        TimeoutError: The preparation budget expired.
    """
    if not isinstance(mediated_egress, bool):
        raise TypeError("Invalid local-network mediator mode.")
    architecture_name = _NATIVE_NAMES.get(platform.machine().lower())
    if platform.system() != "Linux" or architecture_name is None:
        raise ValueError("Unsupported native local-network filter ABI.")
    library = _load_library(deadline)
    architecture = library.seccomp_arch_resolve_name(architecture_name)
    if not architecture or architecture != library.seccomp_arch_native():
        raise ValueError("Native libseccomp ABI does not match the host.")
    context = library.seccomp_init(_ALLOW)
    if not context:
        raise OSError("Native local-network filter allocation failed.")
    try:
        _require_result(library.seccomp_attr_set(context, _BAD_ARCH, _KILL_PROCESS))
        for name in (b"bind", b"listen"):
            number = library.seccomp_syscall_resolve_name_arch(architecture, name)
            if number < 0:
                raise ValueError("Required native local-network syscall is unavailable.")
            _require_result(
                library.seccomp_rule_add_exact_array(
                    context, _ERRNO | _LINUX_EPERM, number, 0, None
                )
            )
        number = library.seccomp_syscall_resolve_name_arch(architecture, b"socket")
        if number < 0:
            raise ValueError("Required native socket syscall is unavailable.")

        def deny(*comparisons: ArgumentComparison) -> None:
            """Add one exact denied socket argument combination."""
            arguments = (ArgumentComparison * len(comparisons))(*comparisons)
            _require_result(
                library.seccomp_rule_add_exact_array(
                    context, _ERRNO | _LINUX_EPERM, number, len(arguments), arguments
                )
            )

        if not mediated_egress:
            # IPv4 (2) and IPv6 (10) differ only in bit 3.
            deny(ArgumentComparison(0, _COMPARE_MASKED_EQUAL, _LINUX_INT_MASK & ~8, 2))
        else:
            inet = ArgumentComparison(0, _COMPARE_MASKED_EQUAL, _LINUX_INT_MASK, _LINUX_AF_INET)
            deny(ArgumentComparison(0, _COMPARE_MASKED_EQUAL, _LINUX_INT_MASK, _LINUX_AF_INET6))
            for bit in range(4):
                deny(
                    inet,
                    ArgumentComparison(
                        1,
                        _COMPARE_MASKED_EQUAL,
                        1 << bit,
                        0 if bit == 0 else 1 << bit,
                    ),
                )
        with tempfile.TemporaryFile() as output:
            _require_result(library.seccomp_export_bpf(context, output.fileno()))
            output.seek(0)
            program = output.read(_MAX_FILTER_BYTES + 1)
        if not program or len(program) > _MAX_FILTER_BYTES or len(program) % _BPF_INSTRUCTION_BYTES:
            raise ValueError("Invalid native local-network filter export.")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Native local-network filter deadline expired.")
        return program
    finally:
        library.seccomp_release(context)


def install_local_network_filter(program: bytes) -> None:
    """Install a trusted precompiled filter after mediator startup and before shell entry.

    Args:
        program (bytes): Bounded native classic BPF from protected host preparation.

    Raises:
        ValueError: The program shape is invalid.
        OSError: The kernel refuses to install the inherited seccomp filter.
    """
    if (
        not isinstance(program, bytes)
        or not program
        or len(program) > _MAX_FILTER_BYTES
        or len(program) % _BPF_INSTRUCTION_BYTES
    ):
        raise ValueError("Invalid inherited local-network filter.")

    class FilterProgram(ctypes.Structure):
        """Represent the kernel's sock_fprog ABI for an immutable BPF buffer."""

        _fields_ = [("length", ctypes.c_ushort), ("instructions", ctypes.c_void_p)]

    instructions = ctypes.create_string_buffer(program)
    specification = FilterProgram(
        len(program) // _BPF_INSTRUCTION_BYTES, ctypes.addressof(instructions)
    )
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    _PR_SET_SECCOMP = 22
    _SECCOMP_MODE_FILTER = 2
    result = libc.prctl(
        _PR_SET_SECCOMP,
        _SECCOMP_MODE_FILTER,
        ctypes.addressof(specification),
        0,
        0,
    )
    if result != 0:
        raise OSError(ctypes.get_errno(), "Native local-network filter installation failed.")
