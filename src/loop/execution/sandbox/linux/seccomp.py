"""Export native descriptor-reception restrictions through protected installed libseccomp."""

import ctypes
import platform
import tempfile
import time

from .host import library_candidates

_ALLOW = 0x7FFF0000
_KILL_PROCESS = 0x80000000
_ERRNO = 0x00050000
_BAD_ARCH = 2  # SCMP_FLTATR_ACT_BADARCH, part of libseccomp's public API.
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
    recvmsg/recvmmsg return EPERM; io_uring setup/enter/register return ENOSYS. Other native
    calls retain the surrounding namespace, AppArmor and capability restrictions.
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
