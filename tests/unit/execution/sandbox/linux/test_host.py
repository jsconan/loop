"""Verify protected native host discovery and inode-bound preparation authority."""

import ctypes
import errno
import hashlib
import io
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import host as linux_host
from loop.utils.process import ProcessCapture, ProcessCaptureStatus


@pytest.fixture
def host(monkeypatch):
    """Provide protected host discovery and native APIs without launching a process."""
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "resolve", lambda self, **options: self)
    details = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o755, st_nlink=1)
    monkeypatch.setattr(Path, "lstat", lambda self: details)
    monkeypatch.setattr(Path, "stat", lambda self: details)
    monkeypatch.setattr(linux_host.os, "listxattr", lambda path: [], raising=False)
    process = Mock()
    monkeypatch.setattr(linux_host.subprocess, "Popen", process)
    capture = Mock(
        return_value=ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            0,
            "1 libs found\n\tlibseccomp.so.2 (libc6) => /lib/libseccomp.so.2\n",
        )
    )
    monkeypatch.setattr(linux_host, "supervise_process", capture)
    return SimpleNamespace(process=process, capture=capture, details=details)


def test_library_candidates_refuses_absent_linker(host, monkeypatch):
    """No workspace PATH lookup substitutes for protected management tooling."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(OSError, match="discovery is unavailable"):
        linux_host.library_candidates("libseccomp")
    host.process.assert_not_called()


def test_library_candidates_discovers_alternate_linker_location(host, monkeypatch):
    """Hosts exposing only the other protected management location remain supported."""
    monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/usr/sbin/ldconfig")
    linux_host.library_candidates("libseccomp")
    assert host.process.call_args.args == (["/usr/sbin/ldconfig", "-p"],)


@pytest.mark.parametrize(
    "result",
    [
        ProcessCapture(ProcessCaptureStatus.TIMED_OUT),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 1),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, stderr="unexpected"),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, stdout_discarded=1),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, stderr_discarded=None),
    ],
)
def test_library_candidates_refuses_incomplete_discovery(host, result):
    """Timeout, nonzero, diagnostics and truncated inventories fail before loading."""
    host.capture.return_value = result
    with pytest.raises(OSError, match="discovery failed"):
        linux_host.library_candidates("libseccomp")


@pytest.mark.parametrize(
    ("inventory", "exception", "message"),
    [
        ("0 libs found", OSError, "Installed libseccomp"),
        ("libother.so (libc6) => /lib/libother.so", OSError, "Installed libseccomp"),
        ("libseccomp.so.2 (libc6) => relative/library", ValueError, "library location"),
    ],
)
def test_library_candidates_refuses_missing_or_relative_library(
    host, inventory, exception, message
):
    """Only a discovered absolute libseccomp path may be loaded."""
    host.capture.return_value = ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, inventory)
    with pytest.raises(exception, match=message):
        linux_host.library_candidates("libseccomp")


@pytest.mark.parametrize(
    "scenario", ["owner", "write", "setuid", "directory", "hardlink", "capability"]
)
@pytest.mark.parametrize("target", ["linker", "library"])
def test_library_candidates_refuses_unsafe_host_authority(host, monkeypatch, scenario, target):
    """Unsafe tools, libraries and their ancestors cannot execute or produce filter bytes."""

    def details(path):
        """Return a controlled unsafe authority at the selected trust boundary."""
        affected = str(path) == ("/sbin" if target == "linker" else "/lib/libseccomp.so.2")
        return SimpleNamespace(
            st_uid=1000 if affected and scenario == "owner" else 0,
            st_mode=(stat.S_IFDIR if affected and scenario == "directory" else stat.S_IFREG)
            | (
                0o777
                if affected and scenario == "write"
                else 0o4755
                if affected and scenario == "setuid"
                else 0o755
            ),
            st_nlink=2 if affected and scenario == "hardlink" else 1,
        )

    # File-type/link checks apply to the resolved file, not ancestor directories.
    if target == "linker" and scenario in ("directory", "hardlink", "capability"):
        selected = "/sbin/ldconfig"
    else:
        selected = "/sbin" if target == "linker" else "/lib/libseccomp.so.2"

    def selected_details(path):
        """Apply file-only negatives to the file and ancestor negatives to ancestors."""
        if str(path) == selected:
            proxy = Path("/sbin" if target == "linker" else "/lib/libseccomp.so.2")
            return details(proxy)
        return host.details

    monkeypatch.setattr(Path, "lstat", selected_details)
    monkeypatch.setattr(Path, "stat", selected_details)
    monkeypatch.setattr(
        linux_host.os,
        "listxattr",
        lambda path: (
            ["security.capability"] if str(path) == selected and scenario == "capability" else []
        ),
    )
    with pytest.raises(ValueError, match="native host"):
        linux_host.library_candidates("libseccomp")


def test_library_candidates_validates_resolved_library_target(host, monkeypatch):
    """A protected symlink cannot redirect discovery into a caller-owned library."""
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, **options: (
            Path("/untrusted/libseccomp") if str(self) == "/lib/libseccomp.so.2" else self
        ),
    )
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: (
            SimpleNamespace(st_uid=1000, st_mode=stat.S_IFREG | 0o755)
            if str(self) == "/untrusted"
            else host.details
        ),
    )
    with pytest.raises(ValueError, match="host authority"):
        linux_host.library_candidates("libseccomp")


def test_library_candidates_accepts_protected_distribution_symlinks(host, monkeypatch):
    """Root-owned library aliases remain usable when the resolved file is protected."""
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, **options: (
            Path("/usr/lib/libseccomp.so.2.6") if str(self) == "/lib/libseccomp.so.2" else self
        ),
    )
    monkeypatch.setattr(Path, "readlink", lambda self: Path("/usr/lib/libseccomp.so.2.6"))
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: (
            SimpleNamespace(st_uid=0, st_mode=stat.S_IFLNK | 0o777, st_nlink=1)
            if str(self) == "/lib/libseccomp.so.2"
            else host.details
        ),
    )
    linux_host.library_candidates("libseccomp")
    assert linux_host.library_candidates("libseccomp") == (Path("/usr/lib/libseccomp.so.2.6"),)


@pytest.mark.parametrize("path", [Path("relative"), Path("/usr/../tmp/tool")])
def test_protected_path_rejects_unbound_spellings(path):
    """Relative paths and traversal cannot supply host authority."""
    with pytest.raises(ValueError, match="Invalid native host path"):
        linux_host.protected_path(path)


@pytest.mark.parametrize("unsafe", ["owner", "write", "link_owner"])
def test_protected_path_rejects_unprotected_intermediate_symlink(host, monkeypatch, unsafe):
    """A protected final inode cannot launder an intermediate caller-controlled alias."""
    spelling = Path("/usr/lib/alias")
    target = Path("/usr/lib/native")
    links = {spelling: Path("/redirect/alias"), Path("/redirect/alias"): target}
    monkeypatch.setattr(Path, "resolve", lambda path, **options: target)
    monkeypatch.setattr(Path, "readlink", lambda path: links[path])

    def metadata(path):
        """Model an unsafe redirect while keeping both endpoint ancestries protected."""
        affected = path == Path("/redirect/alias" if unsafe == "link_owner" else "/redirect")
        return SimpleNamespace(
            st_uid=1000 if affected and unsafe != "write" else 0,
            st_mode=(stat.S_IFLNK | 0o777 if path in links else stat.S_IFREG | 0o755)
            | (0o022 if affected and unsafe == "write" else 0),
            st_nlink=1,
        )

    monkeypatch.setattr(Path, "lstat", metadata)
    with pytest.raises(ValueError, match="Unsafe native host authority"):
        linux_host.protected_path(spelling)


@pytest.mark.parametrize("relative", [False, True])
def test_protected_path_accepts_protected_intermediate_symlinks(host, monkeypatch, relative):
    """Protected absolute and parent-relative distribution chains retain canonical authority."""
    spelling = Path("/usr/lib/alias")
    target = Path("/usr/lib/native")
    redirect = Path("/redirect/alias")
    links = {spelling: redirect, redirect: Path("../usr/lib/native") if relative else target}
    monkeypatch.setattr(Path, "resolve", lambda path, **options: target)
    monkeypatch.setattr(Path, "readlink", lambda path: links[path])
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: SimpleNamespace(
            st_uid=0,
            st_mode=stat.S_IFLNK | 0o777 if path in links else stat.S_IFREG | 0o755,
            st_nlink=1,
        ),
    )
    assert linux_host.protected_path(spelling) == target


def test_protected_path_bounds_distinct_symlink_inspection(host, monkeypatch):
    """An excessive alias graph refuses preparation rather than consuming unbounded work."""
    target = Path("/usr/lib/native")
    monkeypatch.setattr(Path, "resolve", lambda path, **options: target)
    monkeypatch.setattr(Path, "readlink", lambda path: Path("/alias" + str(int(path.name[5:]) + 1)))
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: SimpleNamespace(
            st_uid=0,
            st_mode=stat.S_IFLNK | 0o777 if path.name.startswith("alias") else stat.S_IFREG | 0o755,
            st_nlink=1,
        ),
    )
    with pytest.raises(ValueError, match="symlink inspection exceeds its bound"):
        linux_host.protected_path(Path("/alias0"))


@pytest.mark.parametrize("stem", ["libapparmor.so.1", "../libevil", "", "libBad"])
def test_library_candidates_rejects_caller_selected_names(host, stem):
    """Library stems cannot encode search paths or fixed SONAME selection."""
    with pytest.raises(ValueError, match="Invalid native library stem"):
        linux_host.library_candidates(stem)
    host.process.assert_not_called()


@pytest.fixture
def binding(tmp_path, monkeypatch):
    """Model protected ownership around a real disposable inode and descriptor."""
    path = tmp_path / "library"
    path.write_bytes(b"native library bytes")
    path.chmod(0o600)
    real_lstat = Path.lstat

    def protected_metadata(candidate):
        """Keep real inode metadata while presenting administrator-only ownership."""
        details = real_lstat(candidate)
        return SimpleNamespace(st_uid=0, st_mode=details.st_mode & ~0o6022)

    monkeypatch.setattr(Path, "lstat", protected_metadata)
    monkeypatch.setattr(linux_host.os, "listxattr", lambda path: [], raising=False)
    return path


def test_host_file_decodes_original_evidence_without_rebinding_absent_file(binding):
    """Decoding retains saved authority after deletion and cannot make the original pin succeed."""
    bound = linux_host.bind_host_file(binding)
    record = bound.to_record()
    binding.unlink()
    restored = linux_host.HostFile.from_record(record)
    assert restored == bound
    with pytest.raises(FileNotFoundError), restored.open_verified():
        pytest.fail("Absent original evidence must not become current file authority.")


@pytest.mark.parametrize(
    "defect",
    [
        "object",
        "keys",
        "path",
        "identity",
        "length",
        "bool",
        "negative",
        "digest_type",
        "digest_shape",
    ],
)
def test_host_file_refuses_malformed_original_records(defect):
    """Saved identity decoding preserves exact integer, schema and digest boundary checks."""
    record = {"path": "/original", "identity": [0] * 7, "sha256": "a" * 64}
    if defect == "object":
        record = []
    elif defect == "keys":
        record["extra"] = True
    elif defect == "path":
        record["path"] = 1
    elif defect == "identity":
        record["identity"] = tuple(record["identity"])
    elif defect == "length":
        record["identity"].pop()
    elif defect in {"bool", "negative"}:
        record["identity"][0] = True if defect == "bool" else -1
    else:
        record["sha256"] = 1 if defect == "digest_type" else "A" * 64
    with pytest.raises(ValueError, match="Invalid saved native file authority"):
        linux_host.HostFile.from_record(record)


@pytest.mark.parametrize("changed", [False, True])
def test_host_file_exports_only_revalidated_original_authority(binding, changed):
    """Serialization preserves original inode/digest fields and refuses subsequent byte drift."""
    bound = linux_host.bind_host_file(binding)
    if changed:
        binding.write_bytes(b"replaced authority")
        with pytest.raises(ValueError, match="Prepared native host authority changed"):
            bound.to_record()
    else:
        assert bound.to_record() == {
            "path": str(bound.path),
            "identity": list(bound.identity),
            "sha256": bound.digest,
        }


def test_bind_host_file_revalidates_pinned_bytes_after_uid_mapping(binding, monkeypatch):
    """Mapped ownership never replaces the prepared inode and byte identity."""
    bound = linux_host.bind_host_file(binding)
    monkeypatch.setattr(Path, "lstat", lambda path: SimpleNamespace(st_uid=65534))
    with bound.open_verified() as descriptor:
        assert os.read(descriptor, 20) == b"native library bytes"
        assert os.fstat(descriptor).st_ino == binding.stat().st_ino
    with pytest.raises(OSError) as caught:
        os.fstat(descriptor)
    assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("change", ["bytes", "inode", "mode", "digest"])
def test_host_file_refuses_changed_prepared_authority(binding, change):
    """Changed bytes, replacement inodes, permissions and digests cannot reach loading."""
    bound = linux_host.bind_host_file(binding)
    if change == "bytes":
        binding.write_bytes(b"changed library bytes")
    elif change == "inode":
        replacement = binding.with_name("replacement")
        replacement.write_bytes(binding.read_bytes())
        replacement.replace(binding)
    elif change == "mode":
        binding.chmod(0o400)
    else:
        bound = replace(bound, digest="0" * 64)
    with (
        pytest.raises(ValueError, match="Prepared native host authority changed"),
        bound.open_verified(),
    ):
        pytest.fail("Changed authority was admitted")


def test_host_file_stops_chunked_verification_at_deadline(binding, monkeypatch):
    """A deadline expiry between digest chunks stops further file reads."""
    binding.write_bytes(b"x" * 150000)
    bound = linux_host.bind_host_file(binding)
    pread = Mock(wraps=os.pread)
    monkeypatch.setattr(linux_host.os, "pread", pread)
    times = iter((10, 10, 50))
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: next(times))
    with (
        pytest.raises(TimeoutError, match="host authority deadline"),
        bound.open_verified(deadline=50),
    ):
        pytest.fail("Expired host-file verification was admitted")
    assert pread.call_count == 1


def test_host_digest_checks_deadline_after_final_read(monkeypatch):
    """The final deadline check catches expiry after a short file reaches EOF."""
    monkeypatch.setattr(linux_host.os, "pread", lambda *_args: b"")
    times = iter((10, 50))
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="host authority deadline"):
        linux_host._digest(0, deadline=50)


def test_host_digest_accepts_exact_size_limit(binding, monkeypatch):
    """A file exactly at the configured host-file limit remains valid."""
    monkeypatch.setattr(linux_host, "_MAX_HOST_FILE_BYTES", 4)
    binding.write_bytes(b"four")
    bound = linux_host.bind_host_file(binding)
    assert bound.digest == hashlib.sha256(b"four").hexdigest()


def test_host_digest_rejects_read_larger_than_requested_chunk(monkeypatch):
    """Unexpected oversized descriptor reads cannot bypass the host-file bound."""
    monkeypatch.setattr(linux_host, "_MAX_HOST_FILE_BYTES", 4)
    monkeypatch.setattr(linux_host.os, "pread", lambda _fd, size, _offset: b"x" * (size + 1))
    with pytest.raises(ValueError, match="exceeds the preparation bound"):
        linux_host._digest(0)


def test_host_file_checks_deadline_before_open(binding, monkeypatch):
    """An already expired verification does not open the saved pathname."""
    bound = linux_host.bind_host_file(binding)
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: 50)
    with (
        pytest.raises(TimeoutError, match="host authority deadline"),
        bound.open_verified(deadline=50),
    ):
        pytest.fail("Expired host-file verification was admitted")


def test_host_file_checks_deadline_after_hash(binding, monkeypatch):
    """Verification refuses expiry after hashing and before yielding its descriptor."""
    bound = linux_host.bind_host_file(binding)
    times = iter((10, 10, 10, 10, 50))
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: next(times))
    with (
        pytest.raises(TimeoutError, match="host authority deadline"),
        bound.open_verified(deadline=50),
    ):
        pytest.fail("Expired host-file verification was admitted")


def test_host_file_closes_descriptor_on_caller_failure(binding):
    """A failed setup callback retains no host library descriptor."""
    bound = linux_host.bind_host_file(binding)
    with pytest.raises(RuntimeError, match="caller failed"), bound.open_verified() as descriptor:
        raise RuntimeError("caller failed")
    with pytest.raises(OSError) as caught:
        os.fstat(descriptor)
    assert caught.value.errno == errno.EBADF


def test_bind_host_file_refuses_oversized_inputs(binding, monkeypatch):
    """Finite host-input hashing bounds reject an oversized regular file."""
    monkeypatch.setattr(linux_host, "_MAX_HOST_FILE_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds the preparation bound"):
        linux_host.bind_host_file(binding)


def test_bind_host_file_stops_before_authentication_when_expired(binding, monkeypatch):
    """An expired binding deadline refuses before inspecting current host authority."""
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: 50)
    with pytest.raises(TimeoutError, match="host authority deadline"):
        linux_host.bind_host_file(binding, deadline=50)


def test_bind_host_file_stops_after_revalidation_when_expired(binding, monkeypatch):
    """A binding that crosses its deadline cannot publish a completed authority record."""
    times = iter((10, 10, 10, 10, 50))
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="host authority deadline"):
        linux_host.bind_host_file(binding, deadline=50)


@pytest.mark.parametrize("when", ["descriptor", "path"])
def test_bind_host_file_refuses_drift_during_preparation(binding, monkeypatch, when):
    """Binding refuses a changed inode before publishing a trusted authority record."""
    if when == "descriptor":
        real_fstat = os.fstat

        def changed(descriptor):
            """Present a different inode only at descriptor revalidation."""
            details = real_fstat(descriptor)
            fields = [
                getattr(details, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            ]
            return SimpleNamespace(
                **dict(
                    zip(
                        (
                            "st_dev",
                            "st_ino",
                            "st_mode",
                            "st_nlink",
                            "st_size",
                            "st_mtime_ns",
                            "st_ctime_ns",
                        ),
                        (fields[0], fields[1] + 1, *fields[2:]),
                        strict=True,
                    )
                )
            )

        monkeypatch.setattr(os, "fstat", changed)
    else:
        real_stat = Path.stat
        calls = []

        def changed(path, **options):
            """Replace bytes after the initial pathname binding observation."""
            calls.append(path)
            if len(calls) == 3:
                path.write_bytes(b"different")
            return real_stat(path, **options)

        monkeypatch.setattr(Path, "stat", changed)
    with pytest.raises(ValueError, match="changed during preparation"):
        linux_host.bind_host_file(binding)


@pytest.mark.parametrize("name", ["bwrap", "apparmor_parser"])
def test_discover_tool_binds_protected_locations(binding, monkeypatch, name):
    """Only fixed management locations are searched, independently of caller PATH."""
    binding.chmod(0o700)
    real_resolve = Path.resolve
    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(Path, "resolve", lambda path, **options: real_resolve(binding))
    # Original management spelling is modeled as root-owned and non-writable too.
    monkeypatch.setattr(Path, "lstat", lambda path: SimpleNamespace(st_uid=0, st_mode=0o100700))
    bound = linux_host.discover_tool(name)
    assert bound.path == binding


@pytest.mark.parametrize("name", ["bwrap", "apparmor_parser", "caller-tool"])
def test_discover_tool_refuses_missing_or_caller_selected_tools(monkeypatch, name):
    """Missing prerequisites and request-selected executables never fall back to PATH."""
    monkeypatch.setattr(Path, "exists", lambda path: False)
    exception = ValueError if name == "caller-tool" else OSError
    with pytest.raises(exception, match="Invalid native|Installed"):
        linux_host.discover_tool(name)


def test_discover_tool_refuses_nonexecutable_authority(binding, monkeypatch):
    """Root ownership alone cannot turn a data file into a management executable."""
    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(Path, "resolve", lambda path, **options: binding)
    monkeypatch.setattr(Path, "lstat", lambda path: SimpleNamespace(st_uid=0, st_mode=0o100600))
    with pytest.raises(ValueError, match="not executable"):
        linux_host.discover_tool("bwrap")


@pytest.mark.parametrize("outcome", ["native", "foreign_first", "unloadable", "missing_api"])
def test_apparmor_library_binds_required_native_api(binding, monkeypatch, outcome):
    """Protected multilib selection requires a loadable transition API without a SONAME pin."""
    # Discovery is the separately tested public collaborator of library selection.
    monkeypatch.setattr(linux_host, "library_candidates", Mock(return_value=(binding, binding)))
    library = SimpleNamespace(aa_change_profile=Mock())
    loader = Mock(return_value=library)
    if outcome == "foreign_first":
        loader.side_effect = [OSError("foreign ABI"), library]
    elif outcome == "unloadable":
        loader.side_effect = OSError("unavailable")
    elif outcome == "missing_api":
        del library.aa_change_profile
    monkeypatch.setattr(ctypes, "CDLL", loader)
    if outcome in ("unloadable", "missing_api"):
        with pytest.raises(OSError, match="Installed libapparmor|Required libapparmor API"):
            linux_host.apparmor_library()
    else:
        bound = linux_host.apparmor_library()
        assert bound.path == binding
        assert library.aa_change_profile.argtypes == [ctypes.c_char_p]
        assert library.aa_change_profile.restype is ctypes.c_int
    assert all(
        call.args == (str(binding),) and call.kwargs == {"use_errno": True}
        for call in loader.call_args_list
    )


@pytest.mark.parametrize("when", ["after_hash", "pathname"])
def test_host_file_refuses_drift_while_verifying(binding, monkeypatch, when):
    """Hashing and pathname revalidation cannot admit a replacement during setup."""
    bound = linux_host.bind_host_file(binding)
    if when == "after_hash":
        real_pread = os.pread
        changed = False

        def read_and_change(descriptor, size, offset):
            """Change the pinned file after the digest reads its original bytes."""
            nonlocal changed
            content = real_pread(descriptor, size, offset)
            if not changed:
                binding.write_bytes(b"changed after read")
                changed = True
            return content

        monkeypatch.setattr(linux_host.os, "pread", read_and_change)
    else:
        real_fstat = os.fstat
        calls = []

        def validate_then_replace(descriptor):
            """Replace the name after the pinned descriptor's final identity check."""
            calls.append(descriptor)
            details = real_fstat(descriptor)
            if len(calls) == 2:
                replacement = binding.with_name("replacement")
                replacement.write_bytes(b"new inode")
                replacement.replace(binding)
            return details

        monkeypatch.setattr(os, "fstat", validate_then_replace)
    with (
        pytest.raises(ValueError, match="Prepared native host authority changed"),
        bound.open_verified(),
    ):
        pytest.fail("Changing native authority was admitted")


@pytest.mark.parametrize("mode", [stat.S_IFREG | 0o755, stat.S_IFIFO | 0o600])
def test_protected_path_requires_directory_type_when_requested(host, mode):
    """Runtime directory preparation cannot accept a regular file or special node."""
    host.details.st_mode = mode
    with pytest.raises(ValueError, match="Unsafe native host file|symlink"):
        linux_host.protected_path(Path("/usr"), directory=True)


def test_runtime_read_roots_covers_lib64_and_protected_aliases(host, monkeypatch):
    """Separate lib64 layouts and merged distribution aliases retain literal mount spellings."""
    host.details.st_mode = stat.S_IFDIR | 0o755
    monkeypatch.setattr(
        Path, "resolve", lambda self, **options: Path("/usr/bin") if self == Path("/bin") else self
    )
    roots = linux_host.runtime_read_roots()
    assert roots == tuple(map(Path, ["/usr", "/bin", "/usr/bin", "/sbin", "/lib", "/lib64"]))


def test_runtime_read_roots_omits_absent_optional_directories(host, monkeypatch):
    """The mandatory runtime root remains while missing aliases add no authority."""
    host.details.st_mode = stat.S_IFDIR | 0o755
    monkeypatch.setattr(Path, "exists", lambda self: False)
    assert linux_host.runtime_read_roots() == (Path("/usr"),)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Construct disposable installed artifacts while isolating the host trust boundary."""
    monkeypatch.setattr(
        linux_host, "protected_path", lambda path, **options: path.resolve(strict=True)
    )
    monkeypatch.setattr(linux_host.os, "listxattr", lambda path: [], raising=False)
    roles = (
        "helper",
        "parser",
        "limiter",
        "apparmor",
        "bootstrap",
        "receiver",
        "compiler_lock",
        "cache_lock",
        "socket_unit",
        "service_unit",
    )
    record = {"schema": 1, "files": {}}
    for role in roles:
        path = tmp_path / role
        path.write_bytes(role.encode())
        path.chmod(0o444 if role.endswith("lock") else 0o755)
        record["files"][role] = {
            "path": str(path),
            "sha256": linux_host.bind_host_file(path).digest,
        }
    receipt = tmp_path / "receipt"

    def save():
        """Write the exact candidate receipt for this test."""
        receipt.write_text(json.dumps(record))

    save()
    return SimpleNamespace(record=record, receipt=receipt, save=save)


def test_installation_files_binds_every_required_role_and_receipt(installed):
    """Receipt authentication returns exact byte/inode authority without readiness claims."""
    files = linux_host.installation_files(installed.receipt)
    assert set(files) == set(installed.record["files"]) | {"receipt"}
    for role, entry in installed.record["files"].items():
        assert files[role].path == Path(entry["path"])
        assert files[role].digest == entry["sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "list",
        "extra",
        "schema_type",
        "schema_version",
        "files_type",
        "missing",
        "role_extra",
        "entry_type",
        "entry_keys",
        "path_type",
        "hash_type",
        "hash_format",
        "hash_drift",
        "alias",
        "executable",
        "lock",
    ],
)
def test_installation_files_rejects_ambiguous_or_changed_authority(installed, change):
    """Malformed schemas, missing roles, stale bytes and unsafe artifact roles fail closed."""
    entry = installed.record["files"]["helper"]
    if change == "list":
        installed.record = []
        installed.receipt.write_text("[]")
    elif change == "extra":
        installed.record["extra"] = 1
    elif change == "schema_type":
        installed.record["schema"] = True
    elif change == "schema_version":
        installed.record["schema"] = 2
    elif change == "files_type":
        installed.record["files"] = []
    elif change == "missing":
        del installed.record["files"]["receiver"]
    elif change == "role_extra":
        installed.record["files"]["extra"] = entry
    elif change == "entry_type":
        installed.record["files"]["helper"] = []
    elif change == "entry_keys":
        entry["extra"] = "bad"
    elif change == "path_type":
        entry["path"] = 1
    elif change == "hash_type":
        entry["sha256"] = 1
    elif change == "hash_format":
        entry["sha256"] = "x" * 64
    elif change == "hash_drift":
        Path(entry["path"]).write_bytes(b"changed")
    elif change == "alias":
        installed.record["files"]["parser"] = entry.copy()
    elif change == "executable":
        Path(entry["path"]).chmod(0o644)
    else:
        Path(installed.record["files"]["cache_lock"]["path"]).chmod(0o644)
    if change != "list":
        installed.save()
    with pytest.raises(ValueError, match="receipt|artifact|entry point|lock"):
        linux_host.installation_files(installed.receipt)


def test_installation_files_rejects_duplicate_json_keys(installed):
    """Duplicate object keys cannot silently choose a different administrator authority."""
    installed.receipt.write_text('{"schema":1,"schema":1,"files":{}}')
    with pytest.raises(ValueError, match="Duplicate installation receipt"):
        linux_host.installation_files(installed.receipt)


def test_installation_files_rejects_oversized_receipt_before_read(installed):
    """The receipt metadata bound prevents unbounded parsing."""
    installed.receipt.write_bytes(b" " * 65537)
    with pytest.raises(ValueError, match="exceeds"):
        linux_host.installation_files(installed.receipt)


def test_installation_files_bounds_actual_read_even_if_metadata_understates_size(
    installed, monkeypatch
):
    """A size race cannot widen receipt decoding beyond the fixed read ceiling."""
    pread = os.pread

    def overread(descriptor, size, offset):
        """Oversize only the explicit receipt read, leaving inode hashing genuine."""
        if size == 65537 and offset == 0:
            return b" " * 65537
        return pread(descriptor, size, offset)

    monkeypatch.setattr(linux_host.os, "pread", overread)
    with pytest.raises(ValueError, match="exceeds"):
        linux_host.installation_files(installed.receipt)


@pytest.mark.parametrize("size", [0, 8, 9])
def test_policy_export_reads_are_bounded_without_reported_size(tmp_path, size):
    """Kernel export consumers accept exact bounds and refuse one extra byte."""
    path = tmp_path / "export"
    path.write_bytes(b"x" * size)
    if size > 8:
        with pytest.raises(ValueError, match="Oversized kernel policy export"):
            linux_host.read_policy_bytes(path, 8)
    else:
        assert linux_host.read_policy_bytes(path, 8) == b"x" * size


@pytest.mark.parametrize("limit", [-1, True, 1.0, 1048577])
def test_policy_export_refuses_invalid_bounds_before_reading(tmp_path, limit):
    """Invalid limits cannot turn a bounded kernel export read into an unbounded operation."""
    with pytest.raises(ValueError, match="Invalid kernel policy export bound"):
        linux_host.read_policy_bytes(tmp_path / "missing", limit)


@pytest.fixture
def policy_compiler(monkeypatch):
    """Provide host-bound native inputs and a supervised byte-producing parser double."""

    @contextmanager
    def verified(**_kwargs):
        """Model a verified close-on-exec input descriptor."""
        yield 99

    def authority(name):
        """Create one distinct trusted input identity."""
        return SimpleNamespace(
            path=Path("/installed") / name,
            identity=(1, 2, 0o100444, 1, 1, 1, 1),
            open_verified=Mock(side_effect=verified),
        )

    state = SimpleNamespace(
        parser=authority("parser"),
        limiter=authority("limiter"),
        abi=authority("abi"),
        deadline=100,
        result=None,
        source=b"profile payload {}\n",
        times=iter([10, 10, 10]),
        output=b"\x00\xff\r\n\r\n\n",
        errors=b"",
        input=None,
        process=None,
    )

    def launch(argv, **options):
        """Capture source and fixed launch controls without starting a process."""
        state.input = options["stdin"].read()
        state.process = SimpleNamespace(
            stdout=io.BytesIO(state.output), stderr=io.BytesIO(state.errors)
        )
        return state.process

    def supervise(process, deadline, **limits):
        """Consume and close the byte-preserving wrappers like the common supervisor."""
        stdout, stderr = process.stdout.read(), process.stderr.read()
        process.stdout.close()
        process.stderr.close()
        return state.result or ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, stdout, stderr)

    state.launch = Mock(side_effect=launch)
    state.supervise = Mock(side_effect=supervise)
    monkeypatch.setattr(
        linux_host,
        "discover_tool",
        lambda name, **_kwargs: {
            "apparmor_parser": state.parser,
            "prlimit": state.limiter,
        }[name],
    )
    monkeypatch.setattr(linux_host, "bind_host_file", lambda path, **_kwargs: state.abi)
    monkeypatch.setattr(linux_host.subprocess, "Popen", state.launch)
    monkeypatch.setattr(linux_host, "supervise_process", state.supervise)
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: next(state.times))
    return state


def compile_source(state):
    """Compile trusted source through the shared public preparation owner."""
    return linux_host.compile_policy_source(state.source, state.deadline)


def test_compile_policy_source_preserves_binary_and_bounds_every_launch(policy_compiler):
    """Binary CR/LF bytes survive transport and fixed controls prevent implicit policy loading."""
    assert compile_source(policy_compiler) == policy_compiler.output
    assert policy_compiler.input == policy_compiler.source
    argv = policy_compiler.launch.call_args.args[0]
    assert argv == [
        "/installed/limiter",
        "--as=268435456:268435456",
        "--cpu=5:5",
        "--core=0:0",
        "--",
        "/installed/parser",
        "--config-file=/dev/null",
        "--base=/etc/apparmor.d",
        "--jobs=0",
        "--stdout",
        "--skip-kernel-load",
        "--skip-cache",
        "--warn=all",
        "--Werror",
    ]
    options = policy_compiler.launch.call_args.kwargs
    assert options["cwd"] == "/" and options["close_fds"] and options["start_new_session"]
    assert options["env"] == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}
    assert policy_compiler.supervise.call_args.args[1] == 15
    assert policy_compiler.supervise.call_args.kwargs == {
        "stdout_limit": 1048576,
        "stderr_limit": 65536,
    }
    for bound in (policy_compiler.parser, policy_compiler.limiter, policy_compiler.abi):
        assert bound.open_verified.call_count == 2


@pytest.mark.parametrize(
    "cause", ["source", "type", "empty", "expired", "discovery_deadline", "drift"]
)
def test_compile_policy_source_refuses_unsafe_or_expired_preparation(policy_compiler, cause):
    """Unsafe inputs and stale preparation budgets never start the parser."""
    exception, message = ValueError, "bound|read-only|changed"
    if cause == "source":
        policy_compiler.source = b"x" * 65537
    elif cause == "type":
        policy_compiler.source = "text"
    elif cause == "empty":
        policy_compiler.source = b""
    elif cause in ("expired", "discovery_deadline"):
        policy_compiler.deadline = 10
        policy_compiler.times = iter([9, 10, 10]) if cause == "expired" else iter([9, 9, 10])
        exception, message = TimeoutError, "deadline expired"
    else:
        policy_compiler.parser.open_verified.side_effect = ValueError("authority changed")
    with pytest.raises(exception, match=message):
        compile_source(policy_compiler)
    policy_compiler.launch.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        ProcessCapture(ProcessCaptureStatus.TIMED_OUT),
        ProcessCapture(ProcessCaptureStatus.CANCELLED),
        ProcessCapture(ProcessCaptureStatus.UNAVAILABLE),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 1, "bytes"),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, ""),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "bytes", "diagnostic"),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "bytes", stdout_discarded=1),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "bytes", stderr_discarded=1),
        ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "bytes", stdout_discarded=None),
    ],
)
def test_compile_policy_source_refuses_failed_or_truncated_output(policy_compiler, result):
    """Timeout, cancellation, diagnostics, errors and discarded bytes cannot become policy."""
    policy_compiler.result = result
    exception = TimeoutError if result.status is ProcessCaptureStatus.TIMED_OUT else OSError
    with pytest.raises(exception, match="deadline expired|compilation failed"):
        compile_source(policy_compiler)
    assert policy_compiler.process.stdout.closed and policy_compiler.process.stderr.closed


def test_compile_policy_source_preserves_shorter_caller_deadline(policy_compiler):
    """An existing execution budget never expands to the compiler's full limit."""
    policy_compiler.deadline = 12
    compile_source(policy_compiler)
    assert policy_compiler.supervise.call_args.args[1] == 12


def test_compile_policy_source_enumerates_with_identical_strict_controls(policy_compiler):
    """Use the protected bounded parser without loading or relaxing diagnostics."""
    policy_compiler.output = b"payload\n"
    assert linux_host.compile_policy_source(policy_compiler.source, 100, names=True) == b"payload\n"
    argv = policy_compiler.launch.call_args.args[0]
    assert "--names" in argv and "--stdout" not in argv
    assert "--Werror" in argv and "--skip-kernel-load" in argv


@pytest.mark.parametrize("path,directory", [("/usr/bin/python3", False), ("/usr/bin/sudo", True)])
def test_elevation_binding_requires_fixed_file(path, directory):
    """The setuid allowance cannot authorize arbitrary tools or directory ancestry."""
    with pytest.raises(ValueError, match="fixed sudo executable"):
        linux_host.protected_path(Path(path), directory=directory, system_elevation=True)


@pytest.mark.parametrize("case", ["safe", "plain", "ancestor", "group", "write", "not_executable"])
def test_elevation_binding_only_allows_fixed_sudo_setuid_leaf(host, monkeypatch, case):
    """Expected sudo privilege leaves ownership, ancestry, write and setgid checks intact."""

    def metadata(path):
        """Present root-owned sudo with one selected privilege boundary defect."""
        leaf = path == Path("/usr/bin/sudo")
        mode = 0o4755 if leaf else 0o755
        if case == "plain" and leaf:
            mode = 0o755
        if case == "ancestor" and path == Path("/usr"):
            mode |= 0o4000
        if case == "group" and leaf:
            mode |= 0o2000
        if case == "write" and leaf:
            mode |= 0o022
        if case == "not_executable" and leaf:
            mode = 0o444
        return SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | mode, st_nlink=1)

    monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(Path, "stat", metadata)
    if case in {"safe", "plain"}:
        assert linux_host.protected_path(Path("/usr/bin/sudo"), system_elevation=True) == Path(
            "/usr/bin/sudo"
        )
        if case == "safe":
            with pytest.raises(ValueError, match="Unsafe native host authority"):
                linux_host.protected_path(Path("/usr/bin/sudo"))
    else:
        with pytest.raises(ValueError, match="Unsafe native host authority|not executable"):
            linux_host.protected_path(Path("/usr/bin/sudo"), system_elevation=True)


def test_elevation_binding_pins_fixed_sudo_bytes(binding, monkeypatch):
    """The expected privilege allowance still binds and revalidates the real inode and bytes."""
    binding.chmod(0o755)
    monkeypatch.setattr(Path, "resolve", lambda self, **options: binding)
    bound = linux_host.bind_host_file(Path("/usr/bin/sudo"), system_elevation=True)
    with bound.open_verified() as descriptor:
        assert os.read(descriptor, 20) == b"native library bytes"


@pytest.mark.parametrize("state", ["present", "missing", "dangling", "unsafe"])
def test_runtime_read_files_authenticates_only_fixed_system_inputs(tmp_path, monkeypatch, state):
    """Optional absence is omitted; present aliases and unsafe targets require authentication."""
    path = tmp_path / "resolv.conf"
    if state == "present":
        path.write_text("nameserver 127.0.0.1\n")
    elif state in {"dangling", "unsafe"}:
        path.symlink_to(tmp_path / "target")
    monkeypatch.setattr(linux_host, "RUNTIME_READ_FILES", (path,))
    bound = Mock()
    binder = Mock(return_value=bound)
    if state in {"dangling", "unsafe"}:
        binder.side_effect = (
            OSError("missing target") if state == "dangling" else ValueError("unsafe target")
        )
    monkeypatch.setattr(linux_host, "bind_host_file", binder)
    if state in {"dangling", "unsafe"}:
        with pytest.raises(OSError if state == "dangling" else ValueError, match="target"):
            linux_host.runtime_read_files()
    else:
        assert linux_host.runtime_read_files() == (() if state == "missing" else ((path, bound),))
    assert binder.call_count == (state != "missing")


@pytest.fixture
def resolver(tmp_path, monkeypatch):
    """Model root-authorized systemd resolver data around real pinned disposable bytes."""
    spelling = Path("/etc/resolv.conf")
    target = Path("/run/systemd/resolve/stub-resolv.conf")
    data = tmp_path / "resolver"
    data.write_text("nameserver 127.0.0.53\n")
    data.chmod(0o644)
    passwd = tmp_path / "passwd"
    passwd.write_text(
        "ordinary:x:1000:1000:René:/home/ordinary:/bin/sh\n"
        "systemd-resolve:x:992:992:resolver:/:/usr/sbin/nologin\n"
    )
    authority = linux_host.HostFile(passwd, (), "unused")
    state = SimpleNamespace(
        target=target,
        data=data,
        passwd=passwd,
        metadata={},
        protected=False,
        bind_deadlines=[],
        open_deadlines=[],
    )
    original_open = os.open
    original_stat = Path.stat
    original_lstat = Path.lstat

    def authenticate(path, **kwargs):
        """Keep root ancestry checks observable and select the service-data exception."""
        if path == spelling and not state.protected:
            raise ValueError("Unsafe native host authority.")
        return data if path == spelling else path

    def metadata(path):
        """Expose only the controlled alias and two service-owned entries."""
        if path in state.metadata:
            return state.metadata[path]
        if path == spelling:
            return SimpleNamespace(st_uid=0, st_mode=stat.S_IFLNK | 0o777)
        if path == state.target.parent:
            return SimpleNamespace(st_uid=992, st_mode=stat.S_IFDIR | 0o755)
        if path == state.target:
            return SimpleNamespace(st_uid=992, st_mode=stat.S_IFREG | 0o644, st_nlink=1)
        return original_lstat(path)

    @contextmanager
    def account_bytes(bound, *, deadline=None):
        """Deliver authenticated account bytes through an owned descriptor."""
        state.open_deadlines.append(deadline)
        descriptor = original_open(passwd, os.O_RDONLY)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    def bind_account(path, *, deadline=None):
        """Supply the protected account record without masking resolver binding."""
        assert path == Path("/etc/passwd")
        state.bind_deadlines.append(deadline)
        return replace(authority, identity=(0, 0, 0, 0, passwd.stat().st_size))

    monkeypatch.setattr(linux_host, "RUNTIME_READ_FILES", (spelling,))
    monkeypatch.setattr(linux_host, "protected_path", authenticate)
    monkeypatch.setattr(linux_host, "bind_host_file", bind_account)
    monkeypatch.setattr(linux_host.HostFile, "open_verified", account_bytes)
    monkeypatch.setattr(Path, "readlink", lambda path: state.target)
    monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(
        Path, "stat", lambda path: original_stat(data if path == state.target else path)
    )
    monkeypatch.setattr(
        os, "open", lambda path, flags: original_open(data if path == state.target else path, flags)
    )
    monkeypatch.setattr(os, "getuid", lambda: 1000)
    monkeypatch.setattr(os, "listxattr", lambda path: [], raising=False)
    return state


@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_runtime_resolver_binds_only_root_or_authenticated_service_data(
    resolver, monkeypatch, protected, relative
):
    """Fixed service files retain exact byte and inode authority without widening tool trust."""
    resolver.protected = protected
    if relative:
        monkeypatch.setattr(
            Path, "readlink", lambda path: Path("../run/systemd/resolve/stub-resolv.conf")
        )
    bound = linux_host.runtime_read_files()[0][1]
    assert bound.path == (resolver.data if protected else resolver.target)
    assert bound.identity == (
        resolver.data.stat().st_dev,
        resolver.data.stat().st_ino,
        resolver.data.stat().st_mode,
        1,
        resolver.data.stat().st_size,
        resolver.data.stat().st_mtime_ns,
        resolver.data.stat().st_ctime_ns,
    )
    assert bound.digest == hashlib.sha256(resolver.data.read_bytes()).hexdigest()


def test_runtime_resolver_threads_deadline_through_account_binding(resolver):
    """Resolver service identity binding shares the command's hash-verification deadline."""
    deadline = linux_host.time.monotonic() + 50
    linux_host.runtime_read_files(deadline=deadline)
    assert resolver.bind_deadlines == [deadline, deadline]
    assert resolver.open_deadlines == [deadline, deadline]


@pytest.mark.parametrize("alias", ["owner", "target"])
def test_runtime_resolver_refuses_unprotected_or_unknown_alias(resolver, alias):
    """Only the root-owned fixed systemd link can delegate resolver-data ownership."""
    if alias == "owner":
        resolver.metadata[Path("/etc/resolv.conf")] = SimpleNamespace(st_uid=1000)
    else:
        resolver.target = Path("/run/other/resolv.conf")
    with pytest.raises(ValueError, match="Unsafe system resolver alias"):
        linux_host.runtime_read_files()


@pytest.mark.parametrize(
    "account",
    [
        "root:x:0:0::/:/bin/sh\n",
        "systemd-resolve:x:992:992::/:/bin/false\n" * 2,
        "systemd-resolve:x:992\n",
        "systemd-resolve:x:invalid:992::/:/bin/false\n",
        "systemd-resolve:x:0:0::/:/bin/false\n",
        "systemd-resolve:x:1001:1001::/:/bin/false\n",
        "systemd-resolve:x:999:999::/:/bin/false\n",
    ],
)
def test_runtime_resolver_refuses_unsafe_service_identity(resolver, monkeypatch, account):
    """Missing, ambiguous, malformed, privileged and caller identities cannot authorize data."""
    resolver.passwd.write_text(account)
    monkeypatch.setattr(os, "getuid", lambda: 999)
    with pytest.raises(ValueError, match="Unsafe system resolver service identity"):
        linux_host.runtime_read_files()


@pytest.mark.parametrize(
    ("entry", "defect"),
    [
        (entry, defect)
        for entry in ("directory", "file")
        for defect in ("owner", "type", "write", "privilege", "link", "execute", "capability")
        if entry == "file" or defect not in {"link", "execute"}
    ],
)
def test_runtime_resolver_refuses_unsafe_service_data(resolver, monkeypatch, entry, defect):
    """Service authority never admits replaceable, executable or privileged resolver inputs."""
    path = resolver.target if entry == "file" else resolver.target.parent
    kind = stat.S_IFREG if path == resolver.target else stat.S_IFDIR
    resolver.metadata[path] = SimpleNamespace(
        st_uid=1000 if defect == "owner" else 992,
        st_mode=(stat.S_IFLNK if defect == "type" else kind)
        | (
            0o666
            if defect == "write"
            else 0o4644
            if defect == "privilege"
            else 0o755
            if defect == "execute"
            else 0o644
        ),
        st_nlink=2 if defect == "link" else 1,
    )
    monkeypatch.setattr(
        os,
        "listxattr",
        lambda selected: (
            ["security.capability"] if selected == path and defect == "capability" else []
        ),
        raising=False,
    )
    with pytest.raises(ValueError, match="Unsafe system resolver service data"):
        linux_host.runtime_read_files()


@pytest.mark.parametrize("deadline", [9, 12, 100])
def test_library_candidates_honors_request_budget(host, monkeypatch, deadline):
    """Discovery cannot start after expiry or replace a shorter request deadline."""
    monkeypatch.setattr(linux_host.time, "monotonic", lambda: 10)
    if deadline <= 10:
        with pytest.raises(TimeoutError, match="deadline"):
            linux_host.library_candidates("libseccomp", deadline=deadline)
        host.process.assert_not_called()
    else:
        linux_host.library_candidates("libseccomp", deadline=deadline)
        assert host.capture.call_args.args[1] == min(deadline, 15)
