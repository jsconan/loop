"""Verify portable libseccomp bindings, exact syscall restrictions and fail-closed discovery."""

import errno
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import seccomp as linux_seccomp


@pytest.fixture
def host(monkeypatch):
    """Provide protected host discovery and native APIs without launching a process."""
    monkeypatch.setattr(linux_seccomp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(linux_seccomp.platform, "machine", lambda: "aarch64")
    candidates = Mock(return_value=(Path("/lib/libseccomp.so.2"),))
    monkeypatch.setattr(linux_seccomp, "library_candidates", candidates)
    library = SimpleNamespace(
        seccomp_arch_native=Mock(return_value=0xC00000B7),
        seccomp_arch_resolve_name=Mock(return_value=0xC00000B7),
        seccomp_init=Mock(return_value=123),
        seccomp_release=Mock(),
        seccomp_attr_set=Mock(return_value=0),
        seccomp_syscall_resolve_name_arch=Mock(side_effect=[212, 243, 425, 426, 427, 198, 199]),
        seccomp_rule_add_exact_array=Mock(return_value=0),
        seccomp_export_bpf=Mock(),
    )

    def export(context, descriptor):
        """Supply opaque exported bytes through the real owned temporary descriptor."""
        os.write(descriptor, b"\x06\0\0\0\0\0\xff\x7f")
        return 0

    library.seccomp_export_bpf.side_effect = export
    loader = Mock(return_value=library)
    monkeypatch.setattr(linux_seccomp.ctypes, "CDLL", loader)
    return SimpleNamespace(library=library, loader=loader, candidates=candidates)


@pytest.mark.parametrize(
    ("machine", "name", "architecture", "numbers"),
    [
        ("aarch64", b"aarch64", 0xC00000B7, [212, 243, 425, 426, 427, 198, 199]),
        ("arm64", b"aarch64", 0xC00000B7, [212, 243, 425, 426, 427, 198, 199]),
        ("x86_64", b"x86_64", 0xC000003E, [47, 299, 425, 426, 427, 41, 53]),
        ("AMD64", b"x86_64", 0xC000003E, [47, 299, 425, 426, 427, 41, 53]),
    ],
)
def test_descriptor_filter_resolves_native_rules_and_exports(
    host, monkeypatch, machine, name, architecture, numbers
):
    """Each native ABI uses maintained resolution and exact restrictions before BPF export."""
    monkeypatch.setattr(linux_seccomp.platform, "machine", lambda: machine)
    library = host.library
    library.seccomp_arch_native.return_value = architecture
    library.seccomp_arch_resolve_name.return_value = architecture
    library.seccomp_syscall_resolve_name_arch.side_effect = numbers
    assert linux_seccomp.descriptor_filter() == b"\x06\0\0\0\0\0\xff\x7f"
    library.seccomp_arch_resolve_name.assert_called_once_with(name)
    library.seccomp_init.assert_called_once_with(0x7FFF0000)
    library.seccomp_attr_set.assert_called_once_with(123, 2, 0x80000000)
    assert library.seccomp_syscall_resolve_name_arch.call_args_list == [
        ((architecture, syscall),)
        for syscall in (
            b"recvmsg",
            b"recvmmsg",
            b"io_uring_setup",
            b"io_uring_enter",
            b"io_uring_register",
            b"socket",
            b"socketpair",
        )
    ]
    calls = library.seccomp_rule_add_exact_array.call_args_list
    assert calls[:5] == [
        ((123, 0x50000 | error, number, 0, None),)
        for number, error in zip(numbers[:5], [1, 1, 38, 38, 38], strict=True)
    ]
    assert len(calls[5:]) == 99
    assert all(call.args[:2] == (123, 0x50001) for call in calls[5:])
    assert [(call.args[2], call.args[3]) for call in calls[5:]] == (
        [(numbers[5], 1)] * 32
        + [(numbers[5], 2)] * 32
        + [(numbers[6], 1)] * 32
        + [(numbers[6], 2)] * 3
    )
    library.seccomp_release.assert_called_once_with(123)
    host.loader.assert_called_once_with("/lib/libseccomp.so.2")
    assert library.seccomp_init.restype is linux_seccomp.ctypes.c_void_p
    assert library.seccomp_rule_add_exact_array.argtypes[-1] is linux_seccomp.ctypes.c_void_p


@pytest.mark.parametrize("mediated", [False, True])
def test_local_network_filter_compiles_for_attempt_mode(host, mediated):
    """Host preparation exports a filter blocking listeners and excess socket families."""
    library = host.library
    library.seccomp_syscall_resolve_name_arch.side_effect = [49, 50, 41]
    assert linux_seccomp.local_network_filter(mediated_egress=mediated) == (
        b"\x06\0\0\0\0\0\xff\x7f"
    )
    assert [call.args[1] for call in library.seccomp_syscall_resolve_name_arch.call_args_list] == [
        b"bind",
        b"listen",
        b"socket",
    ]
    library.seccomp_export_bpf.assert_called_once()
    assert library.seccomp_rule_add_exact_array.call_count == (7 if mediated else 3)
    library.seccomp_release.assert_called_once_with(123)


def test_local_network_filter_rejects_wrong_mode_and_unsupported_host(host, monkeypatch):
    """An invalid setup input or architecture cannot install a partial filter."""
    with pytest.raises(TypeError, match="mediator mode"):
        linux_seccomp.local_network_filter(mediated_egress=1)
    monkeypatch.setattr(linux_seccomp.platform, "machine", lambda: "i386")
    with pytest.raises(ValueError, match="filter ABI"):
        linux_seccomp.local_network_filter(mediated_egress=False)
    host.loader.assert_not_called()


@pytest.mark.parametrize("failure", ["arch", "allocation", "syscall", "socket", "rule", "export"])
def test_local_network_filter_fails_closed_on_native_errors(host, failure):
    """Native preparation and export errors stop before command source runs."""
    library = host.library
    library.seccomp_syscall_resolve_name_arch.side_effect = [49, 50, 41]
    if failure == "arch":
        library.seccomp_arch_native.return_value = 0
    elif failure == "allocation":
        library.seccomp_init.return_value = 0
    elif failure == "syscall":
        library.seccomp_syscall_resolve_name_arch.side_effect = [-1]
    elif failure == "socket":
        library.seccomp_syscall_resolve_name_arch.side_effect = [49, 50, -1]
    elif failure == "rule":
        library.seccomp_rule_add_exact_array.return_value = -errno.EPERM
    else:
        library.seccomp_export_bpf.side_effect = lambda _context, _fd: -errno.EPERM
    with pytest.raises((ValueError, OSError)):
        linux_seccomp.local_network_filter(mediated_egress=False)
    if failure in {"arch", "allocation"}:
        library.seccomp_release.assert_not_called()
    else:
        library.seccomp_release.assert_called_once_with(123)


def test_local_network_filter_rejects_invalid_export_and_expired_budget(host):
    """An empty BPF export or elapsed preparation deadline cannot authorize a payload."""
    host.library.seccomp_syscall_resolve_name_arch.side_effect = [49, 50, 41]
    host.library.seccomp_export_bpf.side_effect = lambda _context, _fd: 0
    with pytest.raises(ValueError, match="Invalid native local-network filter export"):
        linux_seccomp.local_network_filter(mediated_egress=False)

    def export(_context, descriptor):
        """Supply a minimal valid native program after the prior invalid export."""
        os.write(descriptor, b"\x06\0\0\0\0\0\xff\x7f")
        return 0

    host.library.seccomp_export_bpf.side_effect = export
    host.library.seccomp_syscall_resolve_name_arch.side_effect = [49, 50, 41]
    with pytest.raises(TimeoutError, match="deadline expired"):
        linux_seccomp.local_network_filter(mediated_egress=False, deadline=0)


def test_local_network_filter_install_validates_program_and_kernel_result(monkeypatch):
    """The isolated payload installs only a bounded BPF program and fails closed on refusal."""
    with pytest.raises(ValueError, match="inherited local-network filter"):
        linux_seccomp.install_local_network_filter(b"bad")
    library = SimpleNamespace(prctl=Mock(return_value=0))
    monkeypatch.setattr(linux_seccomp.ctypes, "CDLL", Mock(return_value=library))
    linux_seccomp.install_local_network_filter(b"\x06\0\0\0\0\0\xff\x7f")
    assert library.prctl.call_args.args[0:2] == (22, 2)
    library.prctl.return_value = -1
    with pytest.raises(OSError, match="filter installation failed"):
        linux_seccomp.install_local_network_filter(b"\x06\0\0\0\0\0\xff\x7f")


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Darwin", "arm64"),
        ("Linux", "i386"),
        ("Linux", "x32"),
        ("Linux", "armv7l"),
        ("Linux", "riscv64"),
    ],
)
def test_descriptor_filter_rejects_unsupported_hosts(host, monkeypatch, system, machine):
    """Foreign and compatibility host ABIs fail before discovery or allocation."""
    monkeypatch.setattr(linux_seccomp.platform, "system", lambda: system)
    monkeypatch.setattr(linux_seccomp.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Unsupported native"):
        linux_seccomp.descriptor_filter()
    host.candidates.assert_not_called()


@pytest.mark.parametrize("architecture", [0, 0xC000003E, 0x40000028, 0x40000003])
def test_descriptor_filter_rejects_missing_or_mismatched_abi(host, architecture):
    """Host spelling cannot override the installed library's native ABI identity."""
    host.library.seccomp_arch_resolve_name.return_value = architecture
    with pytest.raises(ValueError, match="ABI does not match"):
        linux_seccomp.descriptor_filter()
    host.library.seccomp_init.assert_not_called()


def test_descriptor_filter_rejects_allocation_failure(host):
    """A missing native context produces no export or release of an invalid pointer."""
    host.library.seccomp_init.return_value = None
    with pytest.raises(OSError, match="allocation failed"):
        linux_seccomp.descriptor_filter()
    host.library.seccomp_release.assert_not_called()


@pytest.mark.parametrize(
    "function", ["seccomp_attr_set", "seccomp_rule_add_exact_array", "seccomp_export_bpf"]
)
def test_descriptor_filter_releases_context_after_api_failure(host, function):
    """Native operation errors stop further preparation and release allocated context."""
    operation = getattr(host.library, function)
    operation.side_effect = None
    operation.return_value = -errno.EINVAL
    with pytest.raises(OSError, match="Native seccomp operation") as failure:
        linux_seccomp.descriptor_filter()
    assert failure.value.errno == errno.EINVAL
    host.library.seccomp_release.assert_called_once_with(123)


@pytest.mark.parametrize("number", [-1, -1024])
def test_descriptor_filter_refuses_missing_and_pseudo_syscalls(host, number):
    """An unresolved or pseudo syscall cannot silently omit a mandatory denial."""
    host.library.seccomp_syscall_resolve_name_arch.side_effect = [number]
    with pytest.raises(ValueError, match="syscall is unavailable"):
        linux_seccomp.descriptor_filter()
    host.library.seccomp_rule_add_exact_array.assert_not_called()
    host.library.seccomp_export_bpf.assert_not_called()
    host.library.seccomp_release.assert_called_once_with(123)


def test_descriptor_filter_refuses_missing_unix_creation_syscall(host):
    """An unavailable Unix socket syscall cannot silently remove its denial."""
    host.library.seccomp_syscall_resolve_name_arch.side_effect = [212, 243, 425, 426, 427, -1]
    with pytest.raises(ValueError, match="syscall is unavailable"):
        linux_seccomp.descriptor_filter()
    host.library.seccomp_export_bpf.assert_not_called()
    host.library.seccomp_release.assert_called_once_with(123)


@pytest.mark.parametrize(
    "program", [b"", b"short", b"x" * 32769], ids=["empty", "unaligned", "oversized"]
)
def test_descriptor_filter_refuses_invalid_exports(host, program):
    """Empty, misaligned and oversized exported filters cannot reach a launcher."""

    def export(context, descriptor):
        """Write the controlled malformed export to its owned descriptor."""
        os.write(descriptor, program)
        return 0

    host.library.seccomp_export_bpf.side_effect = export
    with pytest.raises(ValueError, match="filter export"):
        linux_seccomp.descriptor_filter()
    host.library.seccomp_release.assert_called_once_with(123)


@pytest.mark.parametrize("missing", ["seccomp_init", "seccomp_export_bpf"])
def test_descriptor_filter_refuses_missing_library_api(host, missing):
    """API availability is checked independently of library versions."""
    delattr(host.library, missing)
    with pytest.raises(OSError, match="Required libseccomp API"):
        linux_seccomp.descriptor_filter()


def test_descriptor_filter_discovers_loadable_native_multilib_entry(host):
    """An incompatible protected cache entry cannot hide the usable native library."""
    host.candidates.return_value = (
        Path("/lib/foreign/libseccomp.so.2"),
        Path("/lib/native/libseccomp.so.2"),
    )
    host.loader.side_effect = [OSError("wrong ELF class"), host.library]
    linux_seccomp.descriptor_filter()
    assert host.loader.call_args_list == [
        (("/lib/foreign/libseccomp.so.2",),),
        (("/lib/native/libseccomp.so.2",),),
    ]


def test_descriptor_filter_refuses_unloadable_library(host):
    """An unusable installed library never produces an unfiltered substitute."""
    host.loader.side_effect = OSError("library cannot load")
    with pytest.raises(OSError, match="Installed libseccomp is unavailable"):
        linux_seccomp.descriptor_filter()


@pytest.mark.parametrize("expired", [False, True])
def test_descriptor_filter_shares_preparation_deadline(host, monkeypatch, expired):
    """Library discovery receives the budget and expired exports release native context."""
    monkeypatch.setattr(linux_seccomp.time, "monotonic", lambda: 20 if expired else 10)
    if expired:
        with pytest.raises(TimeoutError, match="deadline"):
            linux_seccomp.descriptor_filter(deadline=15)
    else:
        assert linux_seccomp.descriptor_filter(deadline=15)
    host.candidates.assert_called_once_with("libseccomp", deadline=15)
    host.library.seccomp_release.assert_called_once_with(123)
