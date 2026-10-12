"""Verify packaged isolated runtime assembly, closed policy and exact helper byte authority."""

import hashlib
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import catalog as linux_catalog
from loop.execution.sandbox.linux.host import HostFile


@pytest.fixture
def helper(tmp_path, monkeypatch):
    """Provide exact real disposable helper bytes without discovering native host tools."""
    path = tmp_path / "bwrap"
    path.write_bytes(b"authenticated helper")
    path.chmod(0o555)
    info = path.stat()
    bound = HostFile(
        path,
        (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(linux_catalog, "discover_tool", Mock(return_value=bound))
    return bound


def test_catalog_uses_distribution_resources_and_authenticated_helper(helper):
    """Retained code uses passive barrels, shipped resources and a private exact helper copy."""
    catalog, executable, sources, attachment = linux_catalog.runtime_catalog(
        Path("/usr/local/libexec/loop/native"), "native_test"
    )
    assert set(linux_catalog.runtime_member_names()) == set(catalog)
    linux_catalog.discover_tool.assert_called_once_with("bwrap")
    assert catalog["helper"] == helper.path.read_bytes()
    assert executable == {"loop_receiver.py", "loop_uninstall.py", "helper"}
    for name in (
        "loop/__init__.py",
        "loop/utils/__init__.py",
        "loop/permissions/__init__.py",
        "loop/execution/__init__.py",
        "loop/execution/sandbox/__init__.py",
        "loop/execution/sandbox/linux/__init__.py",
    ):
        assert catalog[name] == b""
    assert b"receive_installed_request" in catalog["loop_receiver.py"]
    assert b"uninstall_generation" in catalog["loop_uninstall.py"]
    assert b"def supervise_process" in catalog["loop/utils/process.py"]
    assert b"def receive_request" in catalog["loop/execution/sandbox/linux/admission.py"]
    assert b"NATIVE_REQUEST_MAGIC" in catalog["loop/execution/sandbox/linux/constants.py"]
    assert b"def read_mountinfo" in catalog["loop/execution/sandbox/linux/mountinfo.py"]
    assert b"def exchange_request" in catalog["loop/execution/sandbox/linux/transport.py"]
    assert b"def from_wire" in catalog["loop/execution/sandbox/contracts.py"]
    assert b"protected_workspace_paths" in catalog["loop/permissions/protection.py"]
    assert b"execute_installed_request(entry.parent" in catalog["loop_receiver.py"]
    assert b"def compile_payload" in catalog["loop/execution/sandbox/linux/compiler.py"]
    assert b"def payload_policy" in catalog["loop/execution/sandbox/linux/policy.py"]
    assert b"def mount_arguments" in catalog["loop/execution/sandbox/linux/paths.py"]
    assert b"def read_generation" in catalog["loop/execution/sandbox/linux/records.py"]
    assert b"def execute_installed_request" in catalog["loop/execution/sandbox/linux/runtime.py"]
    assert catalog["mask.empty"] == b""
    assert b"def namespace_arguments" in catalog["loop/execution/sandbox/linux/namespace.py"]
    assert set(sources) == {""}
    assert set(sources[""]) == {"bootstrap"}
    assert b"allow capability mac_admin" in sources[""]["bootstrap"]
    assert b"change_profile -> &payload_*" in sources[""]["bootstrap"]
    assert b'profile loopgen_native_test "/usr/local/libexec/loop/native/helper"' in attachment
    assert b"change_profile -> &:native_test:bootstrap" in attachment
    assert catalog["cache.lock"] == b"native_test\n"
    assert b"execute_payload" in catalog["loop_payload.py"]
    assert b"def supervise_payload" in catalog["loop/execution/sandbox/linux/supervision.py"]


@pytest.mark.parametrize(
    "directory,generation",
    [
        ("relative", "test"),
        ("/root/../other", "test"),
        ("/root/sp ace", "test"),
        ("/", "test"),
        ("/native", ""),
        ("/native", "../escape"),
        ("/native", "a" * 121),
    ],
)
def test_catalog_invalid_literal_identity_refuses_before_reads(helper, directory, generation):
    """Invalid source identifiers and roots cannot select package or helper authority."""
    with pytest.raises(ValueError, match="Invalid fixed runtime catalog"):
        linux_catalog.runtime_catalog(Path(directory), generation)
    linux_catalog.discover_tool.assert_not_called()


@pytest.mark.parametrize("read", [b"short", b"x" * len(b"authenticated helper")])
def test_catalog_short_or_changed_helper_read_refuses(helper, monkeypatch, read):
    """A short read or changed same-sized bytes cannot become trusted installation helper input."""
    real_pread = os.pread

    def selected_read(descriptor, size, offset):
        """Change only the catalog's bounded helper read, not HostFile digest verification."""
        if size == helper.identity[4] + 1 and offset == 0:
            return read
        return real_pread(descriptor, size, offset)

    monkeypatch.setattr(linux_catalog.os, "pread", selected_read)
    with pytest.raises(ValueError, match="helper changed"):
        linux_catalog.runtime_catalog(Path("/native"), "test")
