"""Provide shared protected-record fixtures for Linux sandbox unit suites."""

import hashlib

import pytest

from loop.execution.sandbox.linux import host as linux_host
from loop.execution.sandbox.linux import management as linux_management
from loop.execution.sandbox.linux import records as linux_records
from loop.execution.sandbox.linux.admission import GenerationAdmission


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """Prepare immutable generation artifacts with isolated host ownership checks."""

    def bind(path, **_kwargs):
        """Bind real inode and bytes while isolating trusted host ownership discovery."""
        details = path.stat()
        return linux_host.HostFile(
            path.resolve(),
            (
                details.st_dev,
                details.st_ino,
                details.st_mode,
                details.st_nlink,
                details.st_size,
                details.st_mtime_ns,
                details.st_ctime_ns,
            ),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def stage(directory, contents, executable, lock):
        """Simulate the new protected management-record publication."""
        record = directory / "management.json"
        record.write_bytes(contents["management.json"])
        record.chmod(0o444)
        return {"management.json": bind(record)}

    monkeypatch.setattr(linux_host, "bind_host_file", bind)
    monkeypatch.setattr(linux_management, "bind_host_file", bind)
    monkeypatch.setattr(linux_records, "bind_host_file", bind)
    monkeypatch.setattr(linux_management, "stage_generation_files", stage)
    directory = tmp_path / "generation"
    directory.mkdir()

    def file(name, data=b"content", mode=0o444, parent=directory):
        """Create one immutable original artifact."""
        path = parent / name
        path.write_bytes(data)
        path.chmod(mode)
        return linux_host.bind_host_file(path)

    helper = file("helper", b"helper", 0o555)
    receiver = file("receiver", b"receiver", 0o555)
    entry = file("loop_uninstall.py", b"uninstall", 0o555)
    bootstrap = file("bootstrap.bin", b"bootstrap")
    compiled = file("attachment.bin", b"compiled")
    source = file("attachment.apparmor", b"source")
    lock = file("management.lock", b"lock")
    compiler = file("compiler.lock", b"compiler")
    reservation = file("reservation", b"reservation")
    socket_unit = file("loopgen_test.socket", b"socket", parent=tmp_path)
    service = file("loopgen_test@.service", b"service", parent=tmp_path)
    slice_unit = file("loopgen_test.slice", b"slice", parent=tmp_path)
    config = GenerationAdmission("test", 1000, 1000, receiver.path, slice_unit, reservation)
    files = {"receiver": receiver, "socket_unit": socket_unit, "service_unit": service}
    artifacts = {
        bound.path.name: bound for bound in (helper, receiver, entry, bootstrap, compiled, source)
    }
    arguments = (
        config,
        files,
        lock,
        compiler,
        (4, 5),
        {"": {"bootstrap": b"bootstrap"}},
        helper,
        b"compiled",
        b"source",
        artifacts,
    )
    return directory / "management.json", arguments
