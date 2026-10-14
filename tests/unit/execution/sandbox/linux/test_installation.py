"""Verify protected receipt publication and installation-wide immutable reservations."""

import errno
import io
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import host as linux_host
from loop.execution.sandbox.linux import installation as linux_installation


@pytest.mark.parametrize("wrong_record", [False, True])
def test_bound_lifecycle_retires_only_its_original_record(retirement, monkeypatch, wrong_record):
    """Retire the authenticated inventory and reject records outside its root."""
    fixture = retirement
    record = fixture.root / ("other.json" if wrong_record else "management.json")
    record.write_bytes(b"original management record")
    record.chmod(0o444)
    authority = linux_host.bind_host_file(record)
    owner = linux_installation.GenerationLifecycle(
        fixture.config,
        fixture.files,
        fixture.lock,
        fixture.compiler_lock,
        fixture.artifacts,
        fixture.namespace_identity,
        fixture.policies,
        fixture.helper,
        fixture.attachment_bytes,
        fixture.attachment_source,
    )
    group = Path("/sys/fs/cgroup") / fixture.config.slice_unit.path.name
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: not fixture.stopped if path == group else original_exists(path),
    )
    closed_before = fixture.closed
    calls_before = list(fixture.calls)
    if wrong_record:
        with pytest.raises(ValueError, match="Invalid original management record location"):
            owner.retire(authority, 20)
        assert record.exists() and fixture.config.reservation.path.exists()
        assert fixture.closed == closed_before and fixture.calls == calls_before
    else:
        owner.retire(authority, 20)
        assert not record.exists() and not fixture.config.reservation.path.exists()
        assert not list(fixture.root.iterdir())


@pytest.fixture
def bound_owner(retirement):
    """Bind an exact lifecycle and original record for checked removal."""
    state = retirement
    record = state.root / "management.json"
    record.write_bytes(b"original management record")
    record.chmod(0o444)
    authority = linux_host.bind_host_file(record)
    owner = linux_installation.GenerationLifecycle(
        state.config,
        state.files,
        state.lock,
        state.compiler_lock,
        state.artifacts,
        state.namespace_identity,
        state.policies,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
    )
    return SimpleNamespace(
        state=state,
        owner=owner,
        record=authority,
    )


@pytest.fixture
def installation_owner(bound_owner):
    """Place persistent locks and full reservation inside the production generation root layout."""
    fixture = bound_owner
    state = fixture.state
    moved = {}
    for name, bound in (("management.lock", state.lock), ("compiler.lock", state.compiler_lock)):
        path = state.root / name
        bound.path.rename(path)
        moved[name] = linux_host.bind_host_file(path)
    pool = state.root / "reservations"
    pool.mkdir(mode=0o755)
    state.config.reservation.path.rename(pool / state.config.generation)
    state.config = replace(
        state.config, reservation=linux_host.bind_host_file(pool / state.config.generation)
    )
    state.lock = moved["management.lock"]
    state.compiler_lock = moved["compiler.lock"]
    fixture.owner = replace(
        fixture.owner, config=state.config, lock=state.lock, compiler_lock=state.compiler_lock
    )
    return fixture


def recovery_manager(fixture, monkeypatch):
    """Model native removed-fragment authority and exact fresh reload/reset effects."""
    previous = linux_installation.supervise_process
    reloaded = []
    reset = []

    def capture(process, *args, **kwargs):
        """Model real manager responses after exact fragment removal and approved reload/reset."""
        argv = process.argv
        if "daemon-reload" in argv:
            reloaded.append(True)
        if "reset-failed" in argv:
            reset.append(True)
        result = previous(process, *args, **kwargs)
        if "show" in argv and argv[-1] in {
            fixture.state.config.slice_unit.path.name,
            fixture.state.files["socket_unit"].path.name,
        }:
            bound = (
                fixture.state.config.slice_unit
                if argv[-1].endswith(".slice")
                else fixture.state.files["socket_unit"]
            )
            if not bound.path.exists() and reloaded:
                fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
                fields.update(
                    FragmentPath="", ActiveState="inactive", SubState="dead", ControlGroup=""
                )
                if argv[-1].endswith(".socket"):
                    fields.update(LoadState="not-found", Accept="no", RemoveOnStop="no")
                    fields.pop("Listen", None)
                result.stdout = "".join(f"{key}={value}\n" for key, value in fields.items())
        if (
            "list-units" in argv
            and argv[-1].endswith("@*.service")
            and not fixture.state.files["service_unit"].path.exists()
            and reset
        ):
            result.stdout = ""
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)


def test_attachment_removal_rechecks_selected_profile_before_write(attachment_removal, monkeypatch):
    """A disappearing final attachment selection refuses mutation after helper withdrawal."""
    state = attachment_removal
    previous = Path.iterdir
    calls = []
    profiles = Path("/sys/kernel/security/apparmor/policy/profiles")

    def entries(path):
        """Hide the fixed attachment only at its final selection after both attestations."""
        if path == profiles:
            calls.append(path)
            if len(calls) == 3:
                return iter(())
        return previous(path)

    monkeypatch.setattr(Path, "iterdir", entries)
    with pytest.raises(ValueError, match="Generation attachment changed before removal"):
        remove_attachment(state)
    assert len(calls) == 3
    assert state.config.reservation.path.exists()
    assert stat.S_IMODE(state.helper.path.lstat().st_mode) == 0o444
    assert state.attachment_entry.exists()


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """Isolate administrator ownership while retaining real disposable file operations."""

    def protected(path, *, directory=False, system_elevation=False):
        """Isolate ownership while preserving required inode types."""
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError("Unsafe native host file.")
        return path.resolve(strict=True)

    monkeypatch.setattr(linux_host, "protected_path", protected)
    monkeypatch.setattr(linux_installation, "protected_path", linux_host.protected_path)
    monkeypatch.setattr(linux_host.os, "listxattr", lambda path: [], raising=False)
    monkeypatch.setattr(linux_installation.os, "getuid", lambda: 0)
    real_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, **kw: (
            "0 0 4294967295" if str(path) == "/proc/self/uid_map" else real_read(path, **kw)
        ),
    )
    lock = tmp_path / "management.lock"
    lock.write_bytes(b"lock")
    lock.chmod(0o444)
    directory = tmp_path / "reservations"
    directory.mkdir(mode=0o755)
    return directory, linux_host.bind_host_file(lock)


@pytest.fixture
def runtime_staging(installation, monkeypatch):
    """Supply isolated private deployment paths and an authoritative empty mount inventory."""
    directory, lock = installation
    previous = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, **kwargs: (
            "1 0 0:1 / / rw - tmpfs tmpfs rw\n"
            if str(path) == "/proc/self/mountinfo"
            else previous(path, **kwargs)
        ),
    )
    return directory, lock, {"pkg/module.py": b"module", "entry.py": b"entry"}


def stage_runtime(state):
    """Publish through the public trusted runtime staging owner."""
    directory, lock, contents = state
    return linux_installation.stage_generation_files(directory, contents, {"entry.py"}, lock)


def test_runtime_staging_publishes_nested_immutable_files_and_exact_reuse(runtime_staging):
    """Reviewed files deploy without replacing existing inodes or changing unrelated locks."""
    directory, lock, contents = runtime_staging
    first = stage_runtime(runtime_staging)
    assert stage_runtime(runtime_staging) == first
    assert set(first) == set(contents)
    assert {name: bound.path.read_bytes() for name, bound in first.items()} == contents
    assert stat.S_IMODE(first["entry.py"].identity[2]) == 0o555
    assert stat.S_IMODE(first["pkg/module.py"].identity[2]) == 0o444
    assert lock.path.read_bytes() == b"lock"
    (directory / "inside.lock").write_bytes(b"lock")
    (directory / "inside.lock").chmod(0o444)
    inside = linux_host.bind_host_file(directory / "inside.lock")
    assert (
        linux_installation.stage_generation_files(directory, contents, {"entry.py"}, inside)
        == first
    )


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "absolute",
        "traversal",
        "alias",
        "root",
        "mode",
        "type",
        "oversized",
        "count",
        "conflict",
        "entries",
    ],
)
def test_runtime_staging_rejects_invalid_catalog_before_mutation(runtime_staging, change):
    """Invalid or excessive deployment catalogs never create a file or directory."""
    directory, lock, contents = runtime_staging
    executable = {"entry.py"}
    if change == "empty":
        contents = {}
    elif change in {"absolute", "traversal", "alias", "root"}:
        contents[
            {"absolute": "/other", "traversal": "../other", "alias": "./other", "root": "."}[change]
        ] = b"x"
    elif change == "mode":
        executable = {"unreviewed"}
    elif change == "type":
        contents["entry.py"] = "untrusted type"
    elif change == "oversized":
        contents["entry.py"] = b"x" * (linux_installation.MAX_RETAINED_BYTES + 1)
    elif change == "count":
        contents.update({str(i): b"x" for i in range(257)})
    elif change == "conflict":
        contents["pkg"] = b"file conflicts with directory"
    else:
        contents.update({f"dir{i}/file": b"x" for i in range(128)})
    with pytest.raises(ValueError, match="catalog|inventory"):
        linux_installation.stage_generation_files(directory, contents, executable, lock)
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        "drift",
        "mode",
        "unknown",
        "alias",
        "ancestry",
        "mount",
        "empty_mount",
        "short_mount",
        "device",
    ],
)
def test_runtime_staging_refuses_existing_drift_before_deploying_missing_files(
    runtime_staging, monkeypatch, change
):
    """Unknown, changed, aliased or mounted existing state prevents additional publication."""
    directory, lock, contents = runtime_staging
    if change in {"drift", "mode"}:
        (directory / "entry.py").write_bytes(b"wrong" if change == "drift" else b"entry")
        (directory / "entry.py").chmod(0o444 if change == "mode" else 0o555)
    elif change == "unknown":
        (directory / "unexpected").write_bytes(b"unknown")
    elif change == "alias":
        alias = directory.parent / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        directory = alias
    elif change == "ancestry":
        other = directory.parent / "outside"
        other.mkdir()
        (directory / "pkg").symlink_to(other, target_is_directory=True)
    elif change == "device":
        previous = Path.lstat

        def metadata(path):
            """Expose a changed existing parent device without creating host mounts."""
            info = previous(path)
            if path == directory / "pkg":
                values = list(info)
                values[2] += 1
                return os.stat_result(values)
            return info

        (directory / "pkg").mkdir()
        monkeypatch.setattr(Path, "lstat", metadata)
    else:
        previous = Path.read_text
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda path, **kw: (
                (
                    ""
                    if change == "empty_mount"
                    else "invalid\n"
                    if change == "short_mount"
                    else f"2 1 0:1 / {directory}/pkg rw - tmpfs tmpfs rw\n"
                )
                if str(path) == "/proc/self/mountinfo"
                else previous(path, **kw)
            ),
        )
    with pytest.raises(ValueError, match="authority|inventory|ancestry|canonical"):
        linux_installation.stage_generation_files(directory, contents, {"entry.py"}, lock)
    assert not (directory / "pkg/module.py").exists()


def test_runtime_staging_retains_completed_partial_files_for_exact_retry(
    runtime_staging, monkeypatch
):
    """A failed no-replace publication keeps complete immutable files and never overwrites them."""
    directory, _, _ = runtime_staging
    original = os.link
    calls = []

    def link(source, target, **kwargs):
        """Fail only the second publication after the first complete file was linked."""
        calls.append(target)
        if len(calls) == 2:
            raise OSError("publication interrupted")
        return original(source, target, **kwargs)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(OSError, match="publication interrupted"):
        stage_runtime(runtime_staging)
    first = linux_host.bind_host_file(directory / "pkg/module.py")
    assert not list(directory.rglob(".loop-file-*"))
    assert stage_runtime(runtime_staging)["pkg/module.py"] == first


def test_runtime_staging_refuses_new_parent_drift(runtime_staging, monkeypatch):
    """A newly created parent must remain canonical before any file is published."""
    directory, _, _ = runtime_staging
    original = linux_installation.protected_path
    monkeypatch.setattr(
        linux_installation,
        "protected_path",
        lambda path, **kwargs: (
            directory / "alias" if path == directory / "pkg" else original(path, **kwargs)
        ),
    )
    with pytest.raises(ValueError, match="ancestry"):
        stage_runtime(runtime_staging)
    assert not (directory / "pkg/module.py").exists()


def test_runtime_staging_refuses_parent_identity_change(runtime_staging, monkeypatch):
    """A changed pinned directory prevents reporting successful deployment."""
    directory, _, _ = runtime_staging
    original = os.link

    def link(source, target, **kwargs):
        """Expose a replacement directory inode only after successful file publication."""
        original(source, target, **kwargs)
        previous = Path.lstat

        def metadata(path):
            """Model a changed pinned directory identity at final synchronization."""
            info = previous(path)
            if path == directory:
                values = list(info)
                values[1] += 1
                return os.stat_result(values)
            return info

        monkeypatch.setattr(Path, "lstat", metadata)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(ValueError, match="identity changed"):
        stage_runtime(runtime_staging)


def reserve(installation, generation="generation"):
    """Reserve through the public installation owner."""
    directory, lock = installation
    return linux_installation.reserve_generation(directory, lock, generation)


@pytest.mark.parametrize("identity", ["uid", "mapping", "writable_lock"])
def test_reservation_requires_initial_administrator_and_readonly_lock(
    installation, monkeypatch, identity
):
    """Mapped root, ordinary callers and writable locks cannot change installation capacity."""
    directory, lock = installation
    if identity == "uid":
        monkeypatch.setattr(linux_installation.os, "getuid", lambda: 1000)
    elif identity == "mapping":
        monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs: "0 1000 1")
    else:
        lock.path.chmod(0o644)
        installation = directory, linux_host.bind_host_file(lock.path)
    with pytest.raises((PermissionError, ValueError), match="administrator|read-only"):
        reserve(installation)
    assert list(directory.iterdir()) == []


def test_reservation_reuse_and_retained_generation_capacity(installation):
    """A retained generation reserves the entire fixed ceiling and refresh cannot reset it."""
    bound = reserve(installation)
    assert reserve(installation) == bound
    assert linux_installation.generation_limits(bound, "generation") == (29, 14680064)
    with pytest.raises(OSError) as caught:
        reserve(installation, generation="next")
    assert caught.value.errno == errno.ENOSPC
    assert list(installation[0].iterdir()) == [bound.path]


@pytest.mark.parametrize("generation", ["", "bad/name", "x" * 129])
def test_reservation_rejects_invalid_inputs(installation, generation):
    """Malformed generation identities cannot reserve capacity."""
    with pytest.raises(ValueError, match="Invalid installation generation"):
        reserve(installation, generation)


def test_reservation_does_not_wait_on_contention(installation, monkeypatch):
    """Busy management fails immediately without filesystem mutation."""
    monkeypatch.setattr(
        linux_installation.fcntl, "flock", Mock(side_effect=BlockingIOError(errno.EAGAIN, "busy"))
    )
    with pytest.raises(BlockingIOError):
        reserve(installation)
    assert list(installation[0].iterdir()) == []


@pytest.mark.parametrize(
    "change", ["name", "directory", "writable", "empty", "changed", "extra", "alias"]
)
def test_reservation_rejects_corrupt_inventory(installation, change):
    """Corrupt retained records refuse admission without eviction or repair-by-deletion."""
    bound = reserve(installation)
    entry = bound.path
    if change == "name":
        entry.rename(entry.parent / "invalid!")
    elif change == "directory":
        entry.unlink()
        entry.mkdir()
    elif change == "alias":
        other = entry.parent.parent / "external"
        entry.rename(other)
        entry.symlink_to(other)
    elif change == "writable":
        entry.chmod(0o644)
    elif change in ("empty", "changed"):
        entry.chmod(0o644)
        entry.write_bytes(b"" if change == "empty" else b"generation\n29\n14680064\n")
        entry.chmod(0o444)
    else:
        (entry.parent / "extra").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory|authority|file"):
        reserve(installation)
    assert len(list(installation[0].iterdir())) >= 1


@pytest.mark.parametrize("change", ["name", "read"])
def test_generation_limits_refuses_changed_authority(installation, monkeypatch, change):
    """Mapped consumers require exact namespace binding and unchanged protected bytes."""
    bound = reserve(installation)
    if change == "read":
        real_pread = os.pread

        def changed_read(descriptor, size, offset):
            """Change the first digest chunk without fabricating further bytes."""
            return b"changed" if offset == 0 else real_pread(descriptor, size, offset)

        monkeypatch.setattr(linux_installation.os, "pread", changed_read)
    with pytest.raises(ValueError, match="authority"):
        linux_installation.generation_limits(
            bound, "bad/name" if change == "name" else "generation"
        )


def test_reservation_requires_canonical_directory(installation, monkeypatch):
    """An installation alias cannot select a different ledger after validation."""
    monkeypatch.setattr(linux_installation, "protected_path", lambda *args, **kw: Path("/other"))
    with pytest.raises(ValueError, match="canonical"):
        reserve(installation)


def test_failed_reservation_is_retained_for_explicit_recovery(installation, monkeypatch):
    """An I/O failure after creation retains the slot instead of freeing uncertain authority."""
    monkeypatch.setattr(linux_installation.os, "fsync", Mock(side_effect=OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        reserve(installation)
    assert len(list(installation[0].iterdir())) == 1


@pytest.fixture
def artifacts(installation):
    """Prepare the exact disposable installation artifact roles."""
    directory, lock = installation
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
    files = {}
    for role in roles:
        path = directory.parent / role
        path.write_bytes(role.encode())
        path.chmod(0o444 if role.endswith("lock") else 0o755)
        files[role] = path
    return directory.parent / "receipt.json", files, lock


def test_provision_receipt_authenticates_roles_and_never_replaces(artifacts):
    """Publication binds complete artifacts while an existing receipt refuses replacement."""
    receipt, files, lock = artifacts
    result = linux_installation.provision_receipt(receipt, files, lock)
    assert result == linux_host.installation_files(receipt)
    assert receipt.stat().st_mode & 0o777 == 0o444
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        linux_installation.provision_receipt(receipt, files, lock)
    assert receipt.read_bytes() == before
    assert not list(receipt.parent.glob(".receipt-*"))


@pytest.mark.parametrize("failure", [False, True])
def test_provision_receipt_syncs_publication_and_retains_authority_on_failure(
    artifacts, monkeypatch, failure
):
    """Sync the final directory name or fail with immutable authority intact."""
    receipt, files, lock = artifacts
    real_sync = os.fsync
    directories = []

    def sync(descriptor):
        """Require complete no-replace publication before syncing its directory."""
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            assert receipt.exists()
            assert not list(receipt.parent.glob(".receipt-*"))
            directories.append(descriptor)
            if failure:
                raise OSError("receipt directory sync failed")
        real_sync(descriptor)

    monkeypatch.setattr(linux_installation.os, "fsync", sync)
    if failure:
        with pytest.raises(OSError, match="receipt directory sync failed"):
            linux_installation.provision_receipt(receipt, files, lock)
    else:
        assert linux_installation.provision_receipt(receipt, files, lock) == (
            linux_host.installation_files(receipt)
        )
    assert len(directories) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(directories[0])
    assert receipt.stat().st_mode & 0o777 == 0o444
    assert set(linux_host.installation_files(receipt)) == set(files) | {"receipt"}
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        linux_installation.provision_receipt(receipt, files, lock)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize("change", ["relative", "traversal", "alias", "missing", "nonexecutable"])
def test_provision_receipt_validates_before_publication(artifacts, monkeypatch, change):
    """Incomplete artifact authority cannot publish a partial installation receipt."""
    receipt, files, lock = artifacts
    if change == "relative":
        receipt = Path("relative")
    elif change == "traversal":
        receipt = receipt.parent / ".." / "receipt.json"
    elif change == "alias":
        monkeypatch.setattr(
            linux_installation, "protected_path", lambda *args, **kw: Path("/other")
        )
    elif change == "missing":
        del files["helper"]
    else:
        files["helper"].chmod(0o444)
    with pytest.raises(ValueError, match="location|canonical|schema|executable"):
        linux_installation.provision_receipt(receipt, files, lock)
    assert not list(lock.path.parent.glob(".receipt-*"))


@pytest.fixture
def publication(installation, monkeypatch):
    """Prepare protected staged inputs and a disposable runtime unit directory."""
    from loop.execution.sandbox.linux.admission import GenerationAdmission

    directory, lock = installation
    runtime = directory.parent / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(linux_installation, "_UNIT_DIRECTORY", runtime)
    receiver = directory.parent / "receiver"
    receiver.write_bytes(b"receiver")
    receiver.chmod(0o555)
    slice_path = directory.parent / "loopgen_generation.slice"
    config = GenerationAdmission(
        "generation",
        1000,
        1000,
        receiver,
        linux_host.HostFile(slice_path, (), ""),
        reserve(installation),
    )
    slice_path.write_text(config.unit_sources()[slice_path.name])
    slice_path.chmod(0o444)
    config = GenerationAdmission(
        "generation",
        1000,
        1000,
        receiver,
        linux_host.bind_host_file(slice_path),
        config.reservation,
    )
    return config, lock, runtime


def test_generation_publication_exact_reuse_and_partial_completion(publication):
    """Complete publication and missing-fragment continuation preserve exact existing inodes."""
    config, lock, runtime = publication
    result = linux_installation.provision_generation_units(config, lock)
    assert set(result) == {"slice_unit", "socket_unit", "service_unit", "receiver"}
    for name, source in config.unit_sources().items():
        assert (runtime / name).read_text() == source
        assert (runtime / name).stat().st_mode & 0o777 == 0o444
    assert linux_installation.provision_generation_units(config, lock) == result
    result["service_unit"].path.unlink()
    resumed = linux_installation.provision_generation_units(config, lock)
    assert resumed["slice_unit"] == result["slice_unit"]
    assert resumed["socket_unit"] == result["socket_unit"]
    assert not list(runtime.glob(".loop-unit-*"))
    assert config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [
        "receiver_mode",
        "receiver_alias",
        "slice_mode",
        "slice_source",
        "directory_alias",
        "unknown",
        "existing_source",
        "existing_mode",
        "existing_alias",
        "dangling",
    ],
)
def test_generation_publication_refuses_unsafe_inputs_before_writes(
    publication, monkeypatch, change
):
    """Unsafe authority and partial drift retain capacity without creating other fragments."""
    from dataclasses import replace

    config, lock, runtime = publication
    source = config.slice_unit.path
    if change == "receiver_mode":
        config.receiver.chmod(0o755)
    elif change == "receiver_alias":
        alias = config.receiver.parent / "alias"
        alias.symlink_to(config.receiver)
        config = replace(config, receiver=alias)
    elif change in {"slice_mode", "slice_source"}:
        source.chmod(0o644)
        if change == "slice_source":
            source.write_bytes(b"wrong")
            source.chmod(0o444)
        config = replace(config, slice_unit=linux_host.bind_host_file(source))
    elif change == "directory_alias":
        real = linux_installation.protected_path
        monkeypatch.setattr(
            linux_installation,
            "protected_path",
            lambda p, **kw: runtime.parent if p == runtime else real(p, **kw),
        )
    else:
        path = runtime / (
            "loopgen_generation.unknown" if change == "unknown" else "loopgen_generation@.service"
        )
        if change in {"existing_alias", "dangling"}:
            target = runtime.parent / "target"
            if change == "existing_alias":
                target.write_text(config.unit_sources()[path.name])
                target.chmod(0o444)
            path.symlink_to(target)
        else:
            path.write_text(
                "wrong"
                if change in {"unknown", "existing_source"}
                else config.unit_sources()[path.name]
            )
            path.chmod(0o644 if change == "existing_mode" else 0o444)
    before = set(runtime.iterdir())
    with pytest.raises(
        (ValueError, FileNotFoundError),
        match="immutable|canonical|source|Unknown|No such|Unsafe native host file",
    ):
        linux_installation.provision_generation_units(config, lock)
    assert set(runtime.iterdir()) == before
    assert config.reservation.path.exists()


@pytest.mark.parametrize("failure", ["sync", "link", "late_input", "contention"])
def test_generation_publication_failure_retains_complete_partial_authority(
    publication, monkeypatch, failure
):
    """Publication never frees capacity or overwrites a racing file after interrupted setup."""
    config, lock, runtime = publication
    if failure == "contention":
        monkeypatch.setattr(
            linux_installation.fcntl, "flock", Mock(side_effect=BlockingIOError("busy"))
        )
    elif failure == "sync":
        monkeypatch.setattr(
            linux_installation.os, "fsync", Mock(side_effect=[None, OSError("sync failed")])
        )
    else:
        real_link = os.link

        def link(source, target, **kw):
            """Inject a no-replace race or stale preparation at the publication boundary."""
            if failure == "link":
                Path(target).write_bytes(b"racer")
            else:
                config.receiver.chmod(0o755)
            return real_link(source, target, **kw)

        monkeypatch.setattr(linux_installation.os, "link", link)
    with pytest.raises((OSError, ValueError), match="busy|sync failed|exists|changed"):
        linux_installation.provision_generation_units(config, lock)
    assert config.reservation.path.exists()
    assert not list(runtime.glob(".loop-unit-*"))
    if failure == "sync":
        assert (runtime / "loopgen_generation.slice").read_text() == config.unit_sources()[
            "loopgen_generation.slice"
        ]
    elif failure == "link":
        assert (runtime / "loopgen_generation.slice").read_bytes() == b"racer"


@pytest.fixture
def drain(installation, monkeypatch):
    """Model fixed manager and kernel boundaries without native enforcement claims."""
    from types import SimpleNamespace

    from loop.execution.sandbox.linux.admission import GenerationAdmission

    directory, lock = installation
    root = directory.parent
    receiver = root / "receiver"
    receiver.write_bytes(b"receiver")
    receiver.chmod(0o555)
    config = GenerationAdmission(
        "generation",
        1000,
        1000,
        receiver,
        linux_host.HostFile(root / "loopgen_generation.slice", (), ""),
        reserve(installation),
    )
    sources = config.unit_sources()
    for name, text in sources.items():
        path = root / name
        path.write_text(text)
        path.chmod(0o444)
    config = GenerationAdmission(
        config.generation,
        config.uid,
        config.gid,
        receiver,
        linux_host.bind_host_file(config.slice_unit.path),
        config.reservation,
    )
    files = {
        "receiver": linux_host.bind_host_file(receiver),
        "socket_unit": linux_host.bind_host_file(root / "loopgen_generation.socket"),
        "service_unit": linux_host.bind_host_file(root / "loopgen_generation@.service"),
    }
    tool = root / "systemctl"
    tool.write_bytes(b"tool")
    tool.chmod(0o555)
    real_bind = linux_installation.bind_host_file
    monkeypatch.setattr(linux_installation, "bind_host_file", lambda path: real_bind(tool))
    state = SimpleNamespace(
        config=config,
        files=files,
        lock=lock,
        calls=[],
        change="",
        closed=False,
        events=b"populated 0\nfrozen 0\n",
        mount="12 1 0:20 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ids="mnt_id:\t12\n",
        entered=False,
    )
    monkeypatch.setattr(linux_installation.time, "monotonic", lambda: 10)

    def popen(argv, **kwargs):
        """Record public subprocess arguments; no actual process is created."""
        state.calls.append(argv)
        return SimpleNamespace(argv=argv)

    def capture(process, *args, **kwargs):
        """Supply bounded system-manager results by the actual requested properties."""
        argv = process.argv
        output = ""
        if "stop" in argv:
            state.closed = True
        elif "list-jobs" in argv:
            output = "1 pending start waiting\n" if state.change == "pending" else ""
        else:
            name = argv[-1]
            slice_unit = name.endswith(".slice")
            fields = {
                "Id": name,
                "LoadState": "loaded",
                "FragmentPath": str(root / name),
                "DropInPaths": "",
                "NeedDaemonReload": "no",
                "Transient": "no",
                "ActiveState": "active" if slice_unit or not state.closed else "inactive",
                "SubState": "active" if slice_unit else "dead" if state.closed else "listening",
                "ControlGroup": "/loopgen_generation.slice" if slice_unit else "",
            }
            if slice_unit:
                fields["Delegate"] = "no"
            else:
                fields.update(Accept="yes", RemoveOnStop="yes", Listen="/run/" + name + " (Stream)")
            if state.change in fields:
                fields[state.change] = "changed"
            if state.closed and state.change == "closure":
                fields["ActiveState"] = "active"
            if state.closed and state.change == "substate" and not slice_unit:
                fields["SubState"] = "listening"
            if state.closed and state.change == "socketgroup" and not slice_unit:
                fields["ControlGroup"] = "/unexpected"
            if state.change == "missinggroup" and slice_unit:
                fields["ControlGroup"] = ""
            if state.change == "inactiveslice" and slice_unit:
                fields["ActiveState"] = "inactive"
            output = "".join(f"{k}={v}\n" for k, v in fields.items())
            if state.change == "duplicate":
                output += "Id=" + name + "\n"
            if state.change == "malformed":
                output += "bad\n"
            if state.change == "extra":
                output += "extra=bad\n"
        result = SimpleNamespace(
            status=linux_installation.ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout=output,
            stderr="",
            stdout_discarded=0,
            stderr_discarded=0,
        )
        if state.change in (
            "status",
            "exit_code",
            "stderr",
            "stdout_discarded",
            "stderr_discarded",
        ):
            setattr(result, state.change, "failed" if state.change in ("status", "stderr") else 1)
        return result

    monkeypatch.setattr(linux_installation.subprocess, "Popen", popen)
    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    real_open = linux_installation.os.open
    real_close = linux_installation.os.close
    monkeypatch.setattr(
        linux_installation.os,
        "open",
        lambda path, *args, **kw: (
            9999 if str(path).startswith("/sys/fs/cgroup") else real_open(path, *args, **kw)
        ),
    )
    monkeypatch.setattr(
        linux_installation.os, "close", lambda fd: None if fd == 9999 else real_close(fd)
    )
    monkeypatch.setattr(linux_installation.os, "read", lambda fd, size: state.events)
    real_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, **kw: (
            state.ids
            if str(path).startswith("/proc/self/fdinfo/")
            else state.mount
            if str(path) == "/proc/self/mountinfo"
            else "threaded\n"
            if str(path).endswith("/cgroup.type") and state.change == "threaded"
            else "domain\n"
            if str(path).endswith("/cgroup.type")
            else real_text(path, **kw)
        ),
    )
    real_stat = Path.lstat
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: (
            SimpleNamespace(
                st_uid=1000 if state.change == "owner" else 0,
                st_dev=1,
                st_ino=2,
                st_nlink=1,
                st_size=0,
                st_mtime_ns=0,
                st_ctime_ns=0,
                st_mode=0o040755
                if path.name == "loopgen_generation.slice"
                else 0o040755
                if state.change == "type"
                else 0o100666
                if state.change == "mode"
                else 0o100644,
            )
            if str(path).startswith("/sys/fs/cgroup/")
            else real_stat(path)
        ),
    )
    real_protected = linux_installation.protected_path
    monkeypatch.setattr(
        linux_installation,
        "protected_path",
        lambda path, **kw: (
            Path("/other")
            if str(path).startswith("/sys/fs/cgroup") and state.change == "alias"
            else path
            if str(path).startswith("/sys/fs/cgroup")
            else real_protected(path, **kw)
        ),
    )
    real_exists = Path.exists
    real_link = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: (
            state.change == "listener"
            if str(path).startswith("/run/loopgen_")
            else real_exists(path)
        ),
    )
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: (
            state.change == "listenerlink"
            if str(path).startswith("/run/loopgen_")
            else real_link(path)
        ),
    )
    return state


def enter_drain(state, deadline=20):
    """Exercise the public scoped drain interface and observe whether it yields."""
    with linux_installation.drained_generation(state.config, state.files, state.lock, deadline):
        state.entered = True


def test_drain_closes_only_listener_and_holds_lock(drain, monkeypatch):
    """An idle recursive hierarchy permits only a lock-held scope with no removal or restart."""
    flock = Mock()
    monkeypatch.setattr(linux_installation.fcntl, "flock", flock)
    with linux_installation.drained_generation(drain.config, drain.files, drain.lock, 20):
        flock.assert_called_once()
        assert drain.closed
    assert [argv[-1] for argv in drain.calls if "stop" in argv] == ["loopgen_generation.socket"]
    assert not any("start" in argv for argv in drain.calls)
    assert drain.config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [
        "Id",
        "DropInPaths",
        "Delegate",
        "duplicate",
        "malformed",
        "extra",
        "closure",
        "substate",
        "socketgroup",
        "missinggroup",
        "inactiveslice",
        "listener",
        "listenerlink",
        "pending",
        "alias",
        "owner",
        "mode",
        "type",
        "threaded",
        "status",
        "exit_code",
        "stderr",
        "stdout_discarded",
        "stderr_discarded",
    ],
)
def test_drain_refuses_unsafe_or_incomplete_authority(drain, change):
    """Loaded drift, failed closure, pending activation and unsafe tracking never yield drain."""
    drain.change = change
    with pytest.raises((ValueError, OSError)):
        enter_drain(drain)
    assert not drain.entered
    assert drain.config.reservation.path.exists()


@pytest.mark.parametrize(
    "data",
    [
        b"populated 1\n",
        b"",
        b"populated 0\npopulated 0\n",
        b"populated 0\npopulated 1\n",
        b"x" * 4097,
    ],
)
def test_drain_refuses_live_or_ambiguous_recursive_population(drain, data):
    """Recursive population is mandatory and retains confinement on ambiguity or live children."""
    drain.events = data
    with pytest.raises(OSError, match="live or ambiguous"):
        enter_drain(drain)
    assert drain.closed and not drain.entered


@pytest.mark.parametrize(
    "change", ["ids", "duplicate", "short", "separator", "root", "filesystem", "submount"]
)
def test_drain_requires_authoritative_kernel_hierarchy(drain, change):
    """Missing or ambiguous mounts and submounts cannot masquerade as recursive kernel evidence."""
    if change == "ids":
        drain.ids = ""
    elif change == "duplicate":
        drain.mount *= 2
    elif change == "short":
        drain.mount = "12 1 0:20 / /sys/fs/cgroup rw\n"
    elif change == "separator":
        drain.mount = drain.mount.replace(" - ", " x ")
    elif change == "root":
        drain.mount = drain.mount.replace("0:20 / ", "0:20 /hidden ")
    elif change == "filesystem":
        drain.mount = drain.mount.replace("cgroup2", "tmpfs")
    else:
        drain.mount += (
            "13 12 0:20 / /sys/fs/cgroup/loopgen_generation.slice/x rw - cgroup2 cgroup rw\n"
        )
    with pytest.raises(ValueError, match="kernel hierarchy"):
        enter_drain(drain)
    assert drain.closed and not drain.entered


@pytest.mark.parametrize("change", ["fragment", "role", "missing", "receiver", "expired"])
def test_drain_validates_fixed_inputs_before_scope(drain, change):
    """Changed fixed fragments, roles, receiver and deadline never grant an idle scope."""
    if change == "fragment":
        path = drain.files["socket_unit"].path
        path.chmod(0o644)
        path.write_text("changed")
        path.chmod(0o444)
        drain.files["socket_unit"] = linux_host.bind_host_file(path)
    elif change == "role":
        drain.files["socket_unit"] = drain.config.slice_unit
    elif change == "missing":
        del drain.files["receiver"]
    elif change == "receiver":
        drain.files["receiver"] = drain.files["service_unit"]
    with pytest.raises((ValueError, TimeoutError)):
        enter_drain(drain, 9 if change == "expired" else 20)
    assert not drain.entered and not drain.closed


def test_drain_deadline_includes_final_authority_revalidation(drain, monkeypatch):
    """Expiry after closure and kernel reads cannot yield a late drain scope."""
    ticks = iter([10] * 7 + [16])
    monkeypatch.setattr(linux_installation.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="drain deadline"):
        enter_drain(drain)
    assert drain.closed and not drain.entered


@pytest.fixture
def policy_removal(drain, tmp_path, monkeypatch):
    """Model kernel-owned policy exports atop the public drain-boundary fixture."""

    root = Path("/sys/kernel/security")
    backing = tmp_path / "security"
    namespace = root / "apparmor/policy/namespaces/generation"
    policies = {
        "": {"bootstrap": b"outer"},
        "inner": {"bootstrap": b"inner", "payload": b"payload"},
    }
    drain.mount += "13 1 0:21 / /sys/kernel/security rw - securityfs securityfs rw\n"
    drain.policies = policies
    drain.namespace = namespace
    drain.removed = []
    drain.policy_change = ""

    real_mkdir = Path.mkdir

    def mkdir(path, *args, **kwargs):
        """Give simulated kernel directories fixed default modes; preserve explicit unsafe modes."""
        if path.is_relative_to(backing) and not args and "mode" not in kwargs:
            kwargs["mode"] = 0o755
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)

    def mapped(path):
        """Translate only the disposable kernel-policy view."""
        return backing / path.relative_to(root) if path.is_relative_to(root) else path

    for name, profiles in policies.items():
        directory = namespace / "namespaces" / name if name else namespace
        mapped(directory / "namespaces").mkdir(parents=True)
        mapped(directory / "profiles").mkdir()
        for index, (profile, data) in enumerate(profiles.items()):
            entry = mapped(directory / "profiles" / ("entry" + str(index)))
            entry.mkdir()
            (entry / "name").write_text(profile + "\n")
            (entry / "mode").write_text("enforce\n")
            (entry / "raw_data").write_bytes(data)
    (backing / "apparmor/.ns_level").write_bytes(b"0\n")
    drain.backing = backing
    real_open, real_iter, real_exists, real_link, real_stat = (
        Path.open,
        Path.iterdir,
        Path.exists,
        Path.is_symlink,
        Path.stat,
    )
    real_lstat, real_readlink = Path.lstat, Path.readlink
    policy = root / "apparmor/policy"

    def metadata(path, *, follow_symlinks=True):
        """Model kernel-owned ancestry and its sole securityfs magic link."""
        details = real_stat(mapped(path), follow_symlinks=follow_symlinks)
        if not path.is_relative_to(root):
            return details
        values = list(details)
        values[4] = 0
        if path == policy and not follow_symlinks:
            values[0] = stat.S_IFLNK | 0o444
        changes = {
            "link_type": (policy, 0, stat.S_IFREG | 0o444),
            "link_owner": (policy, 4, 1000),
            "target_type": (policy, 0, stat.S_IFREG | 0o444),
            "target_owner": (policy, 4, 1000),
            "target_write": (policy, 0, stat.S_IFDIR | 0o777),
            "directory_type": (namespace, 0, stat.S_IFLNK | 0o755),
            "directory_owner": (namespace, 4, 1000),
            "directory_write": (namespace, 0, stat.S_IFDIR | 0o777),
            "directory_device": (namespace, 2, 99999),
        }
        if drain.policy_change in changes:
            target, field, value = changes[drain.policy_change]
            target_change = drain.policy_change.startswith("target")
            if path == target and (path != policy or follow_symlinks == target_change):
                values[field] = value
        return os.stat_result(
            values,
            {"st_mtime_ns": details.st_mtime_ns, "st_ctime_ns": details.st_ctime_ns},
        )

    real_protected = linux_installation.protected_path
    monkeypatch.setattr(
        Path, "open", lambda path, *args, **kw: real_open(mapped(path), *args, **kw)
    )
    monkeypatch.setattr(
        Path, "iterdir", lambda path: (path / p.name for p in real_iter(mapped(path)))
    )
    monkeypatch.setattr(Path, "exists", lambda path: real_exists(mapped(path)))
    monkeypatch.setattr(Path, "is_symlink", lambda path: real_link(mapped(path)))
    monkeypatch.setattr(Path, "stat", lambda path, **kw: metadata(path, **kw))
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: (
            metadata(path, follow_symlinks=False) if path.is_relative_to(root) else real_lstat(path)
        ),
    )
    monkeypatch.setattr(
        Path,
        "readlink",
        lambda path: (
            Path(
                "other:[1]"
                if drain.policy_change == "link_target"
                else f"apparmorfs:[{real_stat(mapped(policy)).st_ino}]"
            )
            if path == policy
            else real_readlink(path)
        ),
    )
    monkeypatch.setattr(
        linux_installation,
        "protected_path",
        lambda path, **kw: (
            Path("/alias")
            if path.is_relative_to(root) and drain.policy_change == "alias"
            else path
            if path.is_relative_to(root)
            else real_protected(path, **kw)
        ),
    )
    identity = mapped(namespace).stat()
    drain.namespace_identity = (identity.st_dev, identity.st_ino)
    real_rmdir = Path.rmdir

    def remove(path):
        """Simulate securityfs namespace removal without treating exports as disk directories."""
        if not path.is_relative_to(root):
            return real_rmdir(path)
        if drain.policy_change == "remove_failure":
            raise OSError("kernel removal failure")
        drain.removed.append(path)
        if drain.policy_change != "remains":
            shutil.rmtree(mapped(path))

    monkeypatch.setattr(Path, "rmdir", remove)
    return drain


def remove_policy(state):
    """Exercise checked policy removal through its public owner."""
    linux_installation.remove_generation_policy(
        state.config, state.files, state.lock, state.namespace_identity, state.policies, 20
    )


def test_policy_removal_validates_exact_tree_and_keeps_capacity(policy_removal):
    """Idle exact namespaces are removed child-first while the full reservation remains bound."""
    state = policy_removal
    remove_policy(state)
    assert state.removed == [state.namespace / "namespaces/inner", state.namespace]
    assert linux_host.bind_host_file(state.config.reservation.path) == state.config.reservation
    assert state.closed and not state.namespace.exists()


@pytest.mark.parametrize("change", ["identity", "bytes", "child"])
def test_policy_removal_continues_only_exact_remaining_outer_tree(
    policy_removal, monkeypatch, change
):
    """Require exact remaining inventory and outer authority during partial removal."""
    state = policy_removal
    original = Path.rmdir

    def remove(path):
        """Fail only the outer kernel removal after the child has actually disappeared."""
        if path == state.namespace:
            raise OSError("outer namespace removal failed")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rmdir", remove)
        with pytest.raises(OSError, match="outer namespace removal failed"):
            remove_policy(state)
    assert state.removed == [state.namespace / "namespaces/inner"]
    assert state.namespace.exists() and state.closed
    assert linux_host.bind_host_file(state.config.reservation.path) == state.config.reservation
    with pytest.raises(ValueError, match="Unexpected generation namespace inventory"):
        remove_policy(state)
    original_policies = state.policies
    state.policies = {"": dict(original_policies[""])}
    if change == "identity":
        identity = state.namespace_identity
        state.namespace_identity = (identity[0], identity[1] + 1)
        with pytest.raises(ValueError, match="namespace identity changed"):
            remove_policy(state)
        state.namespace_identity = identity
    elif change == "bytes":
        state.policies[""]["bootstrap"] = b"wrong outer"
        with pytest.raises(ValueError, match="policy differs"):
            remove_policy(state)
        state.policies[""] = dict(original_policies[""])
    else:
        directory = state.backing / "apparmor/policy/namespaces/generation/namespaces/unknown"
        directory.mkdir()
        with pytest.raises(ValueError, match="Unexpected generation namespace inventory"):
            remove_policy(state)
        directory.rmdir()
    assert state.namespace.exists() and state.config.reservation.path.exists()
    remove_policy(state)
    assert state.removed == [state.namespace / "namespaces/inner", state.namespace]
    assert original_policies.keys() == {"", "inner"}
    assert linux_host.bind_host_file(state.config.reservation.path) == state.config.reservation


@pytest.mark.parametrize(
    "change",
    [
        "notdict",
        "root",
        "namespaces",
        "name",
        "namespace_type",
        "profiles_type",
        "bootstrap",
        "profile_type",
        "profile_name",
        "data_type",
        "empty",
        "source_size",
        "count",
        "aggregate",
    ],
)
def test_policy_removal_refuses_invalid_fixed_authority(policy_removal, change):
    """Malformed or excessive fixed policy manifests never authorize kernel namespace removal."""
    state = policy_removal
    if change == "notdict":
        state.policies = []
    elif change == "root":
        del state.policies[""]
    elif change == "namespaces":
        state.policies["extra"] = {"bootstrap": b"x"}
    elif change == "name":
        state.policies["../bad"] = state.policies.pop("inner")
    elif change == "namespace_type":
        state.policies[1] = state.policies.pop("inner")
    elif change == "profiles_type":
        state.policies[""] = []
    elif change == "bootstrap":
        state.policies[""] = {"payload": b"x"}
    elif change == "profile_type":
        state.policies[""][1] = b"x"
    elif change == "profile_name":
        state.policies[""]["bad/name"] = b"x"
    elif change == "data_type":
        state.policies[""]["bootstrap"] = "data"
    elif change == "empty":
        state.policies[""]["bootstrap"] = b""
    elif change == "source_size":
        state.policies[""]["bootstrap"] = b"x" * 1048577
    elif change == "count":
        state.policies[""].update({"p" + str(i): b"x" for i in range(32)})
    else:
        state.policies[""].update({"p" + str(i): b"x" * 1048576 for i in range(16)})
    with pytest.raises(ValueError, match="fixed generation policy"):
        remove_policy(state)
    assert state.closed and not state.removed and state.config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [
        "mount",
        "separator",
        "filesystem",
        "level",
        "alias",
        "identity",
        "link_type",
        "link_owner",
        "link_target",
        "target_type",
        "target_owner",
        "target_write",
        "directory_type",
        "directory_owner",
        "directory_write",
        "directory_device",
        "child",
        "nested",
        "extra",
        "duplicate",
        "unknown",
        "mode",
        "bytes",
        "missing",
        "hats",
        "oversize",
        "reservation_name",
        "reservation_parent",
        "remove_failure",
        "remains",
        "late",
        "final_late",
    ],
)
def test_policy_removal_preserves_capacity_on_drift_or_failure(policy_removal, monkeypatch, change):
    """Kernel drift, nesting, missing authority, expiry and partial removal never free capacity."""
    state = policy_removal
    directory = state.backing / "apparmor/policy/namespaces/generation"
    entry = directory / "profiles/entry0"
    if change == "mount":
        state.mount = state.mount.split("13 1")[0]
    elif change == "separator":
        state.mount = state.mount.replace("- securityfs", "x securityfs")
    elif change == "filesystem":
        state.mount = state.mount.replace("securityfs securityfs", "tmpfs tmpfs")
    elif change == "level":
        (state.backing / "apparmor/.ns_level").write_bytes(b"1\n")
    elif change in ("alias", "remove_failure", "remains") or change.startswith(
        ("link_", "target_", "directory_")
    ):
        state.policy_change = change
    elif change == "identity":
        state.namespace_identity = (0, 0)
    elif change == "child":
        (directory / "namespaces/other").mkdir()
    elif change == "nested":
        (directory / "namespaces/inner/namespaces/deeper").mkdir()
    elif change in ("extra", "duplicate"):
        other = directory / "profiles/extra"
        other.mkdir()
        if change == "duplicate":
            # Keep cardinality below expected until the duplicate semantic name is read.
            state.policies[""]["second"] = b"x"
        for name, data in (
            ("name", b"bootstrap\n"),
            ("mode", b"enforce\n"),
            ("raw_data", b"outer"),
        ):
            (other / name).write_bytes(data)
    elif change == "unknown":
        (entry / "name").write_bytes(b"other\n")
    elif change == "mode":
        (entry / "mode").write_bytes(b"complain\n")
    elif change == "bytes":
        (entry / "raw_data").write_bytes(b"changed")
    elif change == "oversize":
        (entry / "raw_data").write_bytes(b"x" * 1048577)
    elif change == "missing":
        state.policies[""]["missing"] = b"missing"
    elif change == "hats":
        (entry / "profiles/hat").mkdir(parents=True)
    elif change == "reservation_name":
        from dataclasses import replace

        state.config = replace(state.config, generation="other")
    elif change == "reservation_parent":
        real = linux_installation.protected_path
        monkeypatch.setattr(
            linux_installation,
            "protected_path",
            lambda path, **kw: (
                Path("/alias") if path == state.config.reservation.path.parent else real(path, **kw)
            ),
        )
    elif change in ("late", "final_late"):
        # Two cap reads and six manager starts, followed by drain's final check.
        ticks = iter([10] * (9 if change == "late" else 11) + [16])
        monkeypatch.setattr(linux_installation.time, "monotonic", lambda: next(ticks))
    with pytest.raises((ValueError, OSError, TimeoutError)):
        remove_policy(state)
    assert state.config.reservation.path.exists()


def test_policy_removal_rejects_live_members_before_policy_reads(policy_removal):
    """A detached populated generation cannot lose any policy or reservation."""
    state = policy_removal
    state.events = b"populated 1\n"
    with pytest.raises(OSError, match="live or ambiguous"):
        remove_policy(state)
    assert state.closed and not state.removed and state.namespace.exists()


def test_policy_removal_never_resolves_kernel_magic_link(policy_removal, monkeypatch):
    """Kernel jump-link traversal works when userspace canonicalization cannot resolve it."""
    real_resolve = Path.resolve
    real_protected = linux_installation.protected_path

    def resolve(path, *args, **kwargs):
        """Reproduce the kernel display spelling that is not a filesystem pathname."""
        if path.is_relative_to(Path("/sys/kernel/security/apparmor/policy")):
            raise FileNotFoundError("apparmorfs:[3229]")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(
        linux_installation,
        "protected_path",
        lambda path, **kwargs: (
            path.resolve(strict=True)
            if path.is_relative_to(Path("/sys/kernel/security/apparmor/policy"))
            else real_protected(path, **kwargs)
        ),
    )
    remove_policy(policy_removal)
    assert len(policy_removal.removed) == 2
    assert policy_removal.config.reservation.path.exists()


@pytest.fixture
def attachment_removal(policy_removal, monkeypatch):
    """Model a private helper and kernel-owned global attachment after namespace removal."""
    import shutil

    state = policy_removal
    shutil.rmtree(state.backing / "apparmor/policy/namespaces/generation")
    state.helper_path = state.config.receiver.parent / "helper"
    state.helper_path.write_bytes(b"private helper")
    state.helper_path.chmod(0o555)
    state.helper = linux_host.bind_host_file(state.helper_path)
    state.compiler_lock_path = state.config.receiver.parent / "compiler.lock"
    state.compiler_lock_path.write_bytes(b"compiler")
    state.compiler_lock_path.chmod(0o444)
    state.compiler_lock = linux_host.bind_host_file(state.compiler_lock_path)
    state.attachment_bytes = b"compiled attachment"
    state.attachment_source = (
        f'abi <abi/4.0>,\nprofile loopgen_generation "{state.helper_path}" {{ }}\n'.encode()
    )
    state.compile_attachment = Mock(
        side_effect=lambda source, deadline, **kwargs: (
            b"loopgen_generation\n" if kwargs.get("names") else b"compiled attachment"
        )
    )
    monkeypatch.setattr(linux_installation, "compile_policy_source", state.compile_attachment)
    state.attachment_entry = state.backing / "apparmor/policy/profiles/owned"
    state.attachment_entry.mkdir(parents=True)
    for name, value in {
        "name": b"loopgen_generation\n",
        "mode": b"enforce\n",
        "attach": str(state.helper_path).encode() + b"\n",
        "raw_data": state.attachment_bytes,
    }.items():
        (state.attachment_entry / name).write_bytes(value)
    state.attachment_change = ""
    state.commands = []
    previous_bind = linux_installation.bind_host_file
    monkeypatch.setattr(
        linux_installation,
        "bind_host_file",
        lambda path: (
            linux_host.bind_host_file(path) if path == state.helper_path else previous_bind(path)
        ),
    )
    previous_open = Path.open

    class Removal(io.BytesIO):
        """Model the exact kernel write without invoking a parser or touching system state."""

        def write(self, command):
            """Record targeted removal and retain the profile on an intended kernel failure."""
            state.commands.append(command)
            if state.attachment_change == "write_error":
                raise OSError("kernel write failed")
            if state.attachment_change == "short":
                return len(command) - 1
            if state.attachment_change != "remains":
                shutil.rmtree(state.attachment_entry)
            return len(command)

    monkeypatch.setattr(
        Path,
        "open",
        lambda path, *args, **kwargs: (
            Removal()
            if path == Path("/sys/kernel/security/apparmor/.remove")
            else previous_open(path, *args, **kwargs)
        ),
    )
    return state


def remove_attachment(state):
    """Exercise attachment withdrawal through the public checked management owner."""
    linux_installation.remove_generation_attachment(
        state.config,
        state.files,
        state.lock,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
        state.compiler_lock,
        20,
    )


def test_attachment_removal_withdraws_helper_before_exact_unload_and_retains_capacity(
    attachment_removal, monkeypatch
):
    """Only the authenticated attachment is removed, with future helper execution denied first."""
    state = attachment_removal
    original_chmod = os.fchmod
    calls = []

    def withdraw(descriptor, mode):
        """Check the confinement is still loaded when execute permission is revoked."""
        assert state.attachment_entry.exists() and not state.commands
        calls.append(mode)
        original_chmod(descriptor, mode)

    monkeypatch.setattr(os, "fchmod", withdraw)
    unrelated = state.attachment_entry.parent / "unrelated"
    unrelated.mkdir()
    (unrelated / "name").write_bytes(b"unrelated\n")
    remove_attachment(state)
    assert calls == [0o444] and state.commands == [b"loopgen_generation\0"]
    assert state.helper_path.stat().st_mode & 0o777 == 0o444
    assert unrelated.exists() and not state.attachment_entry.exists()
    assert state.config.reservation.path.exists() and state.closed


def test_attachment_removal_accepts_verified_absent_pre_activation_tracking(
    attachment_removal, monkeypatch
):
    """Pre-record removal rechecks an absent cgroup after withdrawing its helper."""
    from contextlib import nullcontext

    state = attachment_removal
    checked = []
    monkeypatch.setattr(
        linux_installation,
        "_recursive_idle",
        lambda group, *, absent=False: checked.append((group, absent)),
    )
    linux_installation._remove_generation_attachment(
        state.config,
        {},
        state.lock,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
        state.compiler_lock,
        20,
        nullcontext(),
        tracking_absent=True,
    )
    assert checked == [(Path("/sys/fs/cgroup") / state.config.slice_unit.path.name, True)]
    assert not state.attachment_entry.exists()


@pytest.mark.parametrize("change", ["helper", "type", "empty", "oversize"])
def test_attachment_removal_rejects_invalid_fixed_authority(attachment_removal, change):
    """Wrong private helper locations and malformed compiled authority never trigger withdrawal."""
    from dataclasses import replace

    state = attachment_removal
    if change == "helper":
        state.helper = replace(state.helper, path=state.helper.path.with_name("other"))
    elif change == "type":
        state.attachment_bytes = "bytes"
    elif change == "empty":
        state.attachment_bytes = b""
    else:
        state.attachment_bytes = b"x" * 1048577
    with pytest.raises(ValueError, match="fixed generation attachment"):
        remove_attachment(state)
    assert not state.commands and state.helper_path.stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize(
    "change", ["namespace", "missing", "mode", "attach", "bytes", "hats", "duplicate", "inventory"]
)
def test_attachment_removal_retains_confinement_on_unsafe_policy_authority(
    attachment_removal, monkeypatch, change
):
    """Remaining namespaces, drift and ambiguous or excessive global inventories refuse mutation."""

    state = attachment_removal
    entry = state.attachment_entry
    if change == "namespace":
        (state.backing / "apparmor/policy/namespaces/generation").mkdir()
    elif change == "missing":
        shutil.rmtree(entry)
    elif change == "hats":
        (entry / "profiles/hat").mkdir(parents=True)
    elif change in ("duplicate", "inventory"):
        other = entry.parent / "other"
        other.mkdir()
        (other / "name").write_bytes(
            b"loopgen_generation\n" if change == "duplicate" else b"other\n"
        )
        if change == "inventory":
            previous_iter = Path.iterdir
            monkeypatch.setattr(
                Path,
                "iterdir",
                lambda path: (
                    iter([path / "other"] * 4097)
                    if path == Path("/sys/kernel/security/apparmor/policy/profiles")
                    else previous_iter(path)
                ),
            )
    else:
        (entry / {"bytes": "raw_data"}.get(change, change)).write_bytes(b"drift\n")
    with pytest.raises(ValueError):
        remove_attachment(state)
    assert state.helper_path.stat().st_mode & 0o777 == 0o555 and not state.commands
    assert state.closed and state.config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "digest",
        "permission",
        "sync",
        "write_error",
        "short",
        "remains",
        "late",
        "final_late",
    ],
)
def test_attachment_removal_preserves_withdrawal_and_capacity_after_partial_failure(
    attachment_removal, monkeypatch, change
):
    """Withdrawal or kernel-removal failure cannot restore helper execution or free reservations."""
    from dataclasses import replace

    state = attachment_removal
    state.attachment_change = change
    if change in ("identity", "digest", "permission"):
        previous_bind = linux_installation.bind_host_file

        def rebind(path):
            """Model drift after the helper permission transition."""
            bound = previous_bind(path)
            if path == state.helper_path and path.stat().st_mode & 0o777 == 0o444:
                if change == "identity":
                    return replace(bound, identity=(0, 0, *bound.identity[2:]))
                if change == "permission":
                    return replace(
                        bound,
                        identity=(*bound.identity[:2], stat.S_IFREG | 0o555, *bound.identity[3:]),
                    )
                return replace(bound, digest="changed")
            return bound

        monkeypatch.setattr(linux_installation, "bind_host_file", rebind)
    elif change == "sync":
        monkeypatch.setattr(os, "fsync", Mock(side_effect=OSError("withdrawal sync failed")))
    elif change in ("late", "final_late"):
        ticks = iter([10] * (9 if change == "late" else 10) + [16])
        monkeypatch.setattr(linux_installation.time, "monotonic", lambda: next(ticks))
    with pytest.raises((ValueError, OSError, TimeoutError)):
        remove_attachment(state)
    assert state.helper_path.stat().st_mode & 0o777 == 0o444
    assert state.closed and state.config.reservation.path.exists()
    if change == "sync":
        assert not state.commands and state.attachment_entry.exists()


def test_attachment_removal_refuses_live_members_before_helper_withdrawal(attachment_removal):
    """Live protected descendants retain the helper and global confinement unchanged."""
    state = attachment_removal
    state.events = b"populated 1\n"
    with pytest.raises(OSError, match="live or ambiguous"):
        remove_attachment(state)
    assert state.helper_path.stat().st_mode & 0o777 == 0o555 and state.attachment_entry.exists()
    assert not state.commands and state.config.reservation.path.exists()


def test_attachment_removal_reauthenticates_protected_helper_before_policy_access(
    attachment_removal, monkeypatch
):
    """Changed host trust authority refuses removal before execute permissions are revoked."""
    from dataclasses import replace

    state = attachment_removal
    previous_bind = linux_installation.bind_host_file
    monkeypatch.setattr(
        linux_installation,
        "bind_host_file",
        lambda path: (
            replace(state.helper, digest="changed")
            if path == state.helper_path
            else previous_bind(path)
        ),
    )
    with pytest.raises(ValueError, match="helper differs from fixed authority"):
        remove_attachment(state)
    assert state.helper_path.stat().st_mode & 0o777 == 0o555
    assert not state.commands and state.config.reservation.path.exists()


def test_attachment_removal_accepts_opaque_display_only_with_compilation_proof(attachment_removal):
    """An opaque DFA display requires the exact fixed single-profile source and loaded bytes."""
    state = attachment_removal
    (state.attachment_entry / "attach").write_bytes(b"<unknown>\n")
    remove_attachment(state)
    assert state.compile_attachment.call_args_list[0].kwargs == {"names": True}
    assert state.compile_attachment.call_args_list[1].args == (state.attachment_source, 15)
    assert state.commands == [b"loopgen_generation\0"]


@pytest.mark.parametrize(
    "change",
    ["source_type", "source_empty", "source_large", "header", "metachar", "names", "compiled"],
)
def test_attachment_removal_refuses_unproved_literal_compilation(attachment_removal, change):
    """Missing literal provenance, extra profiles and compilation drift retain all authority."""
    from dataclasses import replace

    state = attachment_removal
    (state.attachment_entry / "attach").write_bytes(b"<unknown>\n")
    if change == "source_type":
        state.attachment_source = "text"
    elif change == "source_empty":
        state.attachment_source = b""
    elif change == "source_large":
        state.attachment_source += b" " * 65536
    elif change == "header":
        state.attachment_source = state.attachment_source.replace(b"loopgen_generation", b"other")
    elif change == "metachar":
        state.config = replace(
            state.config,
            receiver=state.config.receiver.parent.with_name("wild*") / state.config.receiver.name,
        )
        state.helper = replace(state.helper, path=state.config.receiver.parent / "helper")
    elif change == "names":
        state.compile_attachment.side_effect = [b"loopgen_generation\nextra\n"]
    else:
        state.compile_attachment.side_effect = [b"loopgen_generation\n", b"drift"]
    with pytest.raises(ValueError, match="fixed authority|fixed generation attachment"):
        remove_attachment(state)
    assert not state.commands and state.helper_path.stat().st_mode & 0o777 == 0o555
    assert state.config.reservation.path.exists() and state.attachment_entry.exists()


@pytest.mark.parametrize("change", ["writable", "same", "busy", "drift"])
def test_attachment_removal_requires_distinct_serialized_compiler_authority(
    attachment_removal, monkeypatch, change
):
    """Unsafe, reused, contended or changed compiler locks retain helper execution and policy."""
    from dataclasses import replace

    state = attachment_removal
    exception, message = ValueError, "fixed generation attachment|compiler changed"
    if change == "writable":
        state.compiler_lock = replace(
            state.compiler_lock,
            identity=(
                *state.compiler_lock.identity[:2],
                0o100644,
                *state.compiler_lock.identity[3:],
            ),
        )
    elif change == "same":
        state.compiler_lock = state.lock
    elif change == "busy":
        monkeypatch.setattr(
            linux_installation.fcntl,
            "flock",
            Mock(side_effect=[None, BlockingIOError("compiler busy")]),
        )
        exception, message = BlockingIOError, "compiler busy"
    else:
        original_verified = linux_host.HostFile.open_verified
        calls = []

        def verified(bound, **kwargs):
            """Refuse compiler authority that changes after successful compilation."""
            if bound == state.compiler_lock:
                calls.append(bound)
                if len(calls) == 2:
                    raise ValueError("compiler changed")
            return original_verified(bound, **kwargs)

        monkeypatch.setattr(linux_host.HostFile, "open_verified", verified)
    with pytest.raises(exception, match=message):
        remove_attachment(state)
    assert not state.commands and state.helper_path.stat().st_mode & 0o777 == 0o555
    assert state.config.reservation.path.exists() and state.attachment_entry.exists()


@pytest.fixture
def policy_loading(attachment_removal, monkeypatch):
    """Model add-only protected kernel loading with disposable exported policy files."""
    from types import SimpleNamespace

    state = attachment_removal
    shutil.rmtree(state.attachment_entry)
    state.sources = {"": {"bootstrap": b"outer"}, "inner": {"bootstrap": b"inner"}}
    state.load_change = ""
    state.loaded = []
    state.compile_attachment.side_effect = lambda source, deadline, **kwargs: (
        (b"loopgen_generation\n" if source == state.attachment_source else b"bootstrap\n")
        if kwargs.get("names")
        else b"compiled " + source
    )
    root = Path("/sys/kernel/security")
    real_mkdir, real_open, real_fstat, real_write, real_close = (
        Path.mkdir,
        os.open,
        os.fstat,
        os.write,
        os.close,
    )
    descriptors = {}

    def mapped(path):
        """Translate authenticated kernel endpoints to the isolated native model."""
        path = Path(path)
        return state.backing / path.relative_to(root) if path.is_relative_to(root) else path

    def mkdir(path, *args, **kwargs):
        """Create the kernel-generated namespace controls after exclusive creation."""
        real_mkdir(mapped(path), *args, **kwargs)
        if path.is_relative_to(root):
            real_mkdir(mapped(path) / "namespaces")
            real_mkdir(mapped(path) / "profiles")
            (mapped(path) / ".load").write_bytes(b"")
            (mapped(path) / ".load").chmod(0o640)

    def opened(path, flags, *args, **kwargs):
        """Track only the fixed kernel load descriptors."""
        descriptor = real_open(mapped(path), flags, *args, **kwargs)
        if Path(path).is_relative_to(root):
            descriptors[descriptor] = Path(path)
        return descriptor

    def metadata(descriptor):
        """Require kernel endpoint ownership and reject simulated unsafe metadata."""
        info = real_fstat(descriptor)
        if descriptor not in descriptors:
            return info
        return SimpleNamespace(
            st_mode=info.st_mode,
            st_dev=info.st_dev,
            st_uid=1000 if state.load_change == "endpoint" else 0,
        )

    def write(descriptor, data):
        """Publish exact kernel exports only after an accepted bounded add-only write."""
        if descriptor not in descriptors:
            return real_write(descriptor, data)
        if state.load_change == "short":
            return len(data) - 1
        path = mapped(descriptors[descriptor]).parent
        entry = path / "profiles" / ("entry" + str(len(state.loaded)))
        entry.mkdir()
        (entry / "name").write_bytes(
            b"loopgen_generation\n" if path == state.backing / "apparmor/policy" else b"bootstrap\n"
        )
        (entry / "mode").write_bytes(b"enforce\n")
        (entry / "raw_data").write_bytes(data)
        (entry / "attach").write_bytes(str(state.helper.path).encode())
        state.loaded.append(data)
        if state.load_change == "late":
            monkeypatch.setattr(linux_installation.time, "monotonic", lambda: 21)
        return len(data)

    def close(descriptor):
        """Discard endpoint tracking before the operating system can reuse its descriptor."""
        descriptors.pop(descriptor, None)
        real_close(descriptor)

    (state.backing / "apparmor/policy/.load").write_bytes(b"")
    (state.backing / "apparmor/policy/.load").chmod(0o640)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "fstat", metadata)
    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(os, "close", close)
    return state


def load_policy(state):
    """Exercise fresh policy publication through the installation owner."""
    return linux_installation.load_generation_policy(
        state.config,
        state.lock,
        state.helper,
        state.sources,
        state.attachment_source,
        state.compiler_lock,
        20,
    )


def test_policy_loading_compiles_all_inputs_and_attests_exact_new_exports(policy_loading):
    """Fresh add-only publication preserves capacity and supplies bound activation authority."""
    state = policy_loading
    identity, compiled, attachment = load_policy(state)
    info = state.namespace.lstat()
    assert identity == (info.st_dev, info.st_ino)
    assert compiled == {
        "": {"bootstrap": b"compiled outer"},
        "inner": {"bootstrap": b"compiled inner"},
    }
    assert attachment == b"compiled " + state.attachment_source
    assert len(state.loaded) == 3 and state.config.reservation.path.exists()
    with pytest.raises(ValueError, match="already exists"):
        load_policy(state)
    assert len(state.loaded) == 3


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "identity",
        "oversized",
        "helper",
        "profile",
        "attachment",
        "compiled",
        "deadline",
        "endpoint",
        "short",
        "late",
        "completion",
    ],
)
def test_policy_loading_refuses_unsafe_inputs_and_retains_partial_authority(
    policy_loading, monkeypatch, change
):
    """No unsafe or incomplete loading opens admission or frees reserved capacity."""
    state = policy_loading
    if change == "source":
        state.sources = {}
    elif change == "identity":
        from dataclasses import replace

        state.config = replace(state.config, uid=0)
    elif change == "oversized":
        state.sources[""]["bootstrap"] = b"x" * 65537
    elif change == "helper":
        state.helper.path.chmod(0o444)
        state.helper = linux_host.bind_host_file(state.helper.path)
    elif change in {"profile", "attachment", "compiled"}:
        previous = state.compile_attachment.side_effect
        state.compile_attachment.side_effect = lambda source, deadline, **kw: (
            b"wrong\n"
            if kw.get("names")
            and (
                change == "profile" or source == state.attachment_source and change == "attachment"
            )
            else b""
            if not kw.get("names") and change == "compiled"
            else previous(source, deadline, **kw)
        )
    elif change == "deadline":
        monkeypatch.setattr(linux_installation.time, "monotonic", lambda: 21)
    elif change == "completion":
        previous = linux_installation.read_policy_bytes

        def read(path, limit):
            """Expire the budget only after complete post-load export verification."""
            result = previous(path, limit)
            if path.name == "attach":
                monkeypatch.setattr(linux_installation.time, "monotonic", lambda: 21)
            return result

        monkeypatch.setattr(linux_installation, "read_policy_bytes", read)
    else:
        state.load_change = change
    with pytest.raises(
        (ValueError, OSError),
        match="policy|source|immutable|profile|deadline|endpoint|kernel load|attachment|admission",
    ):
        load_policy(state)
    assert state.config.reservation.path.exists() and not state.commands


@pytest.fixture
def activation(attachment_removal, monkeypatch):
    """Model actual policy and manager boundaries for fixed generation activation."""
    from types import SimpleNamespace

    state = attachment_removal
    for relative, profiles in state.policies.items():
        root = state.backing / "apparmor/policy/namespaces/generation"
        directory = root / "namespaces" / relative if relative else root
        (directory / "namespaces").mkdir(parents=True)
        (directory / "profiles").mkdir()
        for index, (name, data) in enumerate(profiles.items()):
            entry = directory / "profiles" / ("entry" + str(index))
            entry.mkdir()
            (entry / "name").write_text(name + "\n")
            (entry / "mode").write_text("enforce\n")
            (entry / "raw_data").write_bytes(data)
    metadata = state.namespace.stat()
    state.namespace_identity = metadata.st_dev, metadata.st_ino
    state.activation_change = ""
    state.slice_active = True
    state.listener_open = False
    state.socket_group = ""
    state.attempted = False
    state.stop_count = 0
    tool_path = state.backing / "systemctl"
    tool_path.write_bytes(b"fixture systemctl")
    tool_path.chmod(0o555)
    tool = linux_host.bind_host_file(tool_path)
    monkeypatch.setattr(
        linux_installation,
        "bind_host_file",
        lambda path: (
            tool if path == Path("/usr/bin/systemctl") else linux_host.bind_host_file(path)
        ),
    )
    monkeypatch.setattr(linux_installation, "_UNIT_DIRECTORY", state.config.receiver.parent)
    previous_capture = linux_installation.supervise_process

    def capture(process, *args, **kw):
        """Supply native manager effects and structured listener states without processes."""
        argv = process.argv
        if "start" in argv:
            if argv[-1].endswith(".slice"):
                state.slice_active = True
            else:
                state.attempted = True
                state.closed = False
                state.listener_open = True
                if state.activation_change in {"start_error", "cleanup_error"}:
                    raise OSError("activation start failed")
        if "stop" in argv:
            state.stop_count += 1
            if state.attempted and state.activation_change == "cleanup_error":
                raise OSError("cleanup stop failed")
            state.listener_open = bool(
                state.attempted and state.activation_change == "cleanup_remains"
            )
        if "cat" in argv:
            output = (
                "# "
                + str(state.files["service_unit"].path)
                + "\n"
                + state.config.unit_sources()["loopgen_generation@.service"]
            )
            if state.activation_change == "template":
                output += "# unexpected drop-in\n"
        elif "list-units" in argv or "daemon-reload" in argv or "start" in argv:
            output = ""
        else:
            result = previous_capture(process, *args, **kw)
            output = result.stdout
            if "show" in argv and argv[-1].endswith(".slice") and not state.slice_active:
                output = output.replace("ActiveState=active", "ActiveState=inactive").replace(
                    "ControlGroup=/loopgen_generation.slice", "ControlGroup="
                )
            if (
                state.attempted
                and "show" in argv
                and argv[-1].endswith(".socket")
                and not state.closed
            ):
                for key, value in {
                    "socket_active": "ActiveState=inactive",
                    "socket_sub": "SubState=dead",
                    "socket_group": "ControlGroup=/other",
                }.items():
                    if state.activation_change == key:
                        field = value.split("=")[0]
                        output = (
                            "\n".join(
                                value if line.startswith(field + "=") else line
                                for line in output.splitlines()
                            )
                            + "\n"
                        )
        if (
            state.attempted
            and "show" in argv
            and argv[-1].endswith(".socket")
            and not state.closed
            and state.socket_group
        ):
            output = output.replace("ControlGroup=\n", "ControlGroup=" + state.socket_group + "\n")
        return SimpleNamespace(
            status=linux_installation.ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout=output,
            stderr="",
            stdout_discarded=0,
            stderr_discarded=0,
        )

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    previous_exists, previous_lstat = Path.exists, Path.lstat
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: (
            state.listener_open
            if str(path) == "/run/loopgen_generation.socket"
            else state.slice_active
            if str(path) == "/sys/fs/cgroup/loopgen_generation.slice"
            else previous_exists(path)
        ),
    )

    def metadata(path):
        """Model only the manager-created listener inode's fixed ordinary authority."""
        if str(path) != "/run/loopgen_generation.socket":
            return previous_lstat(path)
        if state.activation_change == "missing_listener":
            raise FileNotFoundError("listener missing")
        if state.activation_change == "late":
            monkeypatch.setattr(linux_installation.time, "monotonic", lambda: 16)
        return SimpleNamespace(
            st_mode=(stat.S_IFREG if state.activation_change == "listener_type" else stat.S_IFSOCK)
            | (0o644 if state.activation_change == "listener_mode" else 0o600),
            st_uid=1001 if state.activation_change == "listener_owner" else 1000,
            st_gid=1000,
        )

    monkeypatch.setattr(Path, "lstat", metadata)
    return state


def activate(state):
    """Exercise the public installer activation transaction with complete fixed authority."""
    return linux_installation.activate_generation(
        state.config,
        state.lock,
        state.helper,
        state.namespace_identity,
        state.policies,
        state.attachment_bytes,
        state.attachment_source,
        state.compiler_lock,
        20,
    )


@pytest.mark.parametrize("inactive", [False, True])
def test_activation_publishes_attests_and_opens_only_idle_tracking(activation, inactive):
    """Activation preserves capacity and starts only its authenticated slice and listener."""
    state = activation
    state.slice_active = not inactive
    config, files = activate(state)
    assert config == state.config and files == state.files
    assert state.listener_open and state.slice_active
    assert state.helper.path.stat().st_mode & 0o777 == 0o555
    assert state.config.reservation.path.exists() and not state.commands and not state.removed
    starts = [call[-1] for call in state.calls if "start" in call]
    assert starts == (["loopgen_generation.slice"] if inactive else []) + [
        "loopgen_generation.socket"
    ]
    assert activate(state) == (config, files)


def test_activation_refuses_active_instances_before_kernel_idle_check(activation, monkeypatch):
    """A live concrete service closes admission and retains all generation authority."""
    state = activation
    state.listener_open = True
    previous = linux_installation.supervise_process
    name = "loopgen_generation@live.service"
    fields = {
        "Id": name,
        "LoadState": "loaded",
        "FragmentPath": str(state.files["service_unit"].path),
        "DropInPaths": "",
        "NeedDaemonReload": "no",
        "Transient": "no",
        "MainPID": "0",
        "ControlPID": "0",
        "ControlGroup": "/loopgen_generation.slice/" + name,
        "Slice": "loopgen_generation.slice",
        "Delegate": "no",
        "ExitType": "cgroup",
        "ActiveState": "active",
    }

    def capture(process, *args, **kwargs):
        """Expose a native-shaped active instance through the manager boundary."""
        result = previous(process, *args, **kwargs)
        if "list-units" in process.argv:
            result.stdout = name + " loaded active running receiver\n"
        elif "show" in process.argv and process.argv[-1] == name:
            result.stdout = "".join(key + "=" + value + "\n" for key, value in fields.items())
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(
        ValueError, match=r"^Retired service authority differs; capacity retained\.$"
    ):
        activate(state)
    assert not state.listener_open and not state.attempted and state.stop_count == 1
    assert state.slice_active and state.config.reservation.path.exists()
    assert not state.commands and not state.removed


@pytest.mark.parametrize(
    "change", ["helper", "compile", "template", "pending", "tracking", "live", "closure"]
)
def test_activation_refuses_before_listener_start_and_preserves_policy(activation, change):
    """No invalid, live or pending generation opens admission or releases capacity."""
    state = activation
    if change == "helper":
        state.helper.path.chmod(0o444)
        state.helper = linux_host.bind_host_file(state.helper.path)
    elif change == "compile":
        state.compile_attachment.side_effect = lambda *a, **kw: b"wrong"
    elif change == "template":
        state.activation_change = "template"
    elif change == "live":
        state.events = b"populated 1\n"
    else:
        state.change = {"tracking": "missinggroup"}.get(change, change)
    with pytest.raises(
        (ValueError, OSError), match="immutable|compilation|template|Pending|tracking|live|closure"
    ):
        activate(state)
    assert not state.attempted and not state.removed and not state.commands
    assert state.config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [
        "start_error",
        "socket_active",
        "socket_sub",
        "socket_group",
        "listener_type",
        "listener_mode",
        "listener_owner",
        "missing_listener",
        "late",
    ],
)
def test_activation_failure_closes_only_listener_and_retains_confinement(activation, change):
    """Failed or expired activation closes admission while preserving tracking and all policy."""
    state = activation
    state.activation_change = change
    with pytest.raises((OSError, ValueError), match="start failed|listener|deadline"):
        activate(state)
    assert not state.listener_open and state.stop_count == 2
    assert state.slice_active and state.config.reservation.path.exists()
    assert not state.commands and not state.removed


@pytest.mark.parametrize("change", ["cleanup_error", "cleanup_remains"])
def test_activation_reports_failed_listener_cleanup_together_with_original_error(
    activation, change
):
    """A failed close is explicit and never reported as a successful activation."""
    state = activation
    state.activation_change = change
    if change == "cleanup_remains":
        state.activation_change = "listener_owner"
        previous = linux_installation.supervise_process

        def capture(process, *args, **kw):
            """Make the post-start validation fail before modelling a remaining listener."""
            result = previous(process, *args, **kw)
            if state.attempted and "stop" in process.argv:
                state.listener_open = True
            return result

        # Set only the public process collaborator; manager validation remains real.
        from unittest.mock import patch

        with (
            patch.object(linux_installation, "supervise_process", capture),
            pytest.raises(BaseExceptionGroup, match="listener closure") as caught,
        ):
            activate(state)
    else:
        with pytest.raises(BaseExceptionGroup, match="listener closure") as caught:
            activate(state)
    assert len(caught.value.exceptions) == 2
    assert state.config.reservation.path.exists() and not state.removed


@pytest.fixture
def reclamation(attachment_removal, monkeypatch):
    """Provide retired policy, exact private artifacts and inactive manager records."""
    from dataclasses import replace
    from types import SimpleNamespace

    state = attachment_removal
    remove_attachment(state)
    root = state.config.receiver.parent / "private"
    root.mkdir(mode=0o755)
    state.config.receiver.rename(root / "receiver")
    state.helper_path.rename(root / "helper")
    state.config = replace(state.config, receiver=root / "receiver")
    state.files["receiver"] = linux_host.bind_host_file(state.config.receiver)
    for name, data in state.config.unit_sources().items():
        path = state.config.slice_unit.path.parent / name
        path.chmod(0o644)
        path.write_text(data)
        path.chmod(0o444)
    state.config = replace(
        state.config, slice_unit=linux_host.bind_host_file(state.config.slice_unit.path)
    )
    state.files["socket_unit"] = linux_host.bind_host_file(state.files["socket_unit"].path)
    state.files["service_unit"] = linux_host.bind_host_file(state.files["service_unit"].path)
    (root / "cache").mkdir(mode=0o755)
    (root / "cache/payload").write_bytes(b"compiled cached policy")
    (root / "cache/payload").chmod(0o444)
    state.artifacts = {
        str(p.relative_to(root)): linux_host.bind_host_file(p)
        for p in root.rglob("*")
        if p.is_file()
    }
    state.root = root
    state.cleanup_change = ""
    state.stopped = False
    state.instance_count = 1
    tool_path = state.backing / "systemctl"
    tool_path.write_bytes(b"fixture systemctl")
    tool_path.chmod(0o555)
    tool = linux_host.bind_host_file(tool_path)
    state.tool = tool
    monkeypatch.setattr(
        linux_installation,
        "bind_host_file",
        lambda path: (
            tool if path == Path("/usr/bin/systemctl") else linux_host.bind_host_file(path)
        ),
    )
    previous_capture = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Model only new manager interactions through the existing public subprocess boundary."""
        argv = process.argv
        change = state.cleanup_change
        output = ""
        if "list-units" in argv and argv[-1].endswith("@*.service"):
            names = [f"loopgen_generation@{i}.service" for i in range(state.instance_count)]
            if change == "duplicate_instance":
                names = names * 2
            if change == "foreign_instance":
                names = ["unrelated@0.service"]
            if change == "blank_instance":
                names = [" "]
            output = "\n".join(names)
        elif "show" in argv and "@" in argv[-1]:
            fields = {
                "Id": argv[-1],
                "LoadState": "loaded",
                "FragmentPath": str(state.files["service_unit"].path),
                "DropInPaths": "",
                "NeedDaemonReload": "no",
                "Transient": "no",
                "MainPID": "0",
                "ControlPID": "0",
                "ControlGroup": "",
                "Slice": "loopgen_generation.slice",
                "Delegate": "no",
                "ExitType": "cgroup",
                "ActiveState": "failed",
            }
            if change == "instance_drift":
                fields["MainPID"] = "7"
            if change == "instance_live":
                fields["ActiveState"] = "active"
            if change == "instance_missing":
                del fields["Id"]
            output = "".join(f"{key}={value}\n" for key, value in fields.items())
            if change == "instance_duplicate":
                output += "Id=duplicate\n"
            if change == "instance_malformed":
                output += "malformed\n"
        elif "stop" in argv and argv[-1].endswith(".slice"):
            state.stopped = True
        elif "show" in argv and argv[-1].endswith(".slice") and state.stopped:
            result = previous_capture(process, *args, **kwargs)
            result.stdout = (
                result.stdout.replace("ActiveState=active", "ActiveState=inactive")
                .replace("SubState=active", "SubState=dead")
                .replace("ControlGroup=/loopgen_generation.slice", "ControlGroup=")
            )
            if change == "slice_active":
                result.stdout = result.stdout.replace("ActiveState=inactive", "ActiveState=active")
            if change == "slice_substate":
                result.stdout = result.stdout.replace("SubState=dead", "SubState=running")
            if change == "slice_group":
                result.stdout = result.stdout.replace("ControlGroup=\n", "ControlGroup=/other\n")
            return result
        elif "list-units" in argv:
            output = "remaining" if change == "remaining_units" else ""
        elif "list-jobs" in argv and state.stopped:
            output = "remaining" if change == "remaining_jobs" else ""
        elif "daemon-reload" not in argv and "reset-failed" not in argv:
            return previous_capture(process, *args, **kwargs)
        return SimpleNamespace(
            status=linux_installation.ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout=output,
            stderr="",
            stdout_discarded=0,
            stderr_discarded=0,
        )

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    return state


def reclaim(state, **kwargs):
    """Exercise exact artifact reclamation through its public trusted-management owner."""
    linux_installation.reclaim_generation(
        state.config, state.files, state.lock, state.compiler_lock, state.artifacts, 20, **kwargs
    )


def test_reclamation_releases_capacity_only_after_exact_owned_cleanup(reclamation):
    """Artifacts, cache, fragments and retired records disappear while shared authority survives."""
    state = reclamation
    reclaim(state)
    assert not list(state.root.iterdir())
    assert not state.config.reservation.path.exists()
    assert (
        state.lock.path.exists() and state.compiler_lock.path.exists() and state.tool.path.exists()
    )
    assert state.stopped and state.closed
    assert all(
        not bound.path.exists()
        for bound in (
            state.config.slice_unit,
            state.files["socket_unit"],
            state.files["service_unit"],
        )
    )
    replacement = linux_installation.reserve_generation(
        state.config.reservation.path.parent, state.lock, "next"
    )
    assert linux_installation.generation_limits(replacement, "next") == (29, 14680064)


def test_reclamation_deletes_support_files_before_helper_and_receiver(reclamation, monkeypatch):
    """Caller inventory ordering cannot delete the recovery anchors ahead of cached artifacts."""
    state = reclamation
    state.artifacts = {
        name: state.artifacts[name] for name in ("helper", "receiver", "cache/payload")
    }
    deleted = []
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        """Record actual owned file deletion at the filesystem boundary."""
        if path.is_relative_to(state.root):
            deleted.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    reclaim(state)
    assert deleted[-2:] == [state.root / "helper", state.config.receiver]
    assert set(deleted[:-2]) == {state.root / "cache/payload"}


def test_retirement_continues_authenticated_remaining_cache_inventory(reclamation, monkeypatch):
    """Keep anchors and capacity when partial cache deletion fails."""
    state = reclamation
    first = state.root / "00-first"
    blocked = state.root / "01-blocked"
    for path in (first, blocked):
        path.write_bytes(b"fixed-cache")
        path.chmod(0o444)
        state.artifacts[path.name] = linux_host.bind_host_file(path)
    state.helper = state.artifacts["helper"]
    state.artifacts = dict(reversed(list(state.artifacts.items())))
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        """Model an exact operating-system deletion refusal without modifying the owner."""
        if path == blocked:
            raise PermissionError(13, "cache unlink denied", str(path))
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unlink)
        with pytest.raises(PermissionError, match="cache unlink denied"):
            reclaim(state)
    assert not first.exists()
    assert blocked.exists() and state.config.receiver.exists() and state.helper.path.exists()
    assert state.closed and not state.stopped
    assert linux_host.bind_host_file(state.config.reservation.path) == state.config.reservation
    assert all(state.files[name].path.exists() for name in ("socket_unit", "service_unit"))
    with pytest.raises(FileNotFoundError):
        reclaim(state)
    state.artifacts.pop(first.name)
    linux_installation.retire_generation(
        state.config,
        state.files,
        state.lock,
        (0, 0),
        {},
        state.helper,
        b"",
        b"",
        state.compiler_lock,
        state.artifacts,
        20,
        recovery=linux_installation.RetirementState.ATTACHMENT_REMOVED,
    )
    assert not list(state.root.iterdir())
    assert not state.config.reservation.path.exists()


@pytest.mark.parametrize("composed", [False, True])
def test_reclamation_continues_explicit_known_removed_helper(reclamation, composed):
    """Finish helper deletion only with its pre-deletion binding intact."""
    state = reclamation
    helper = state.artifacts.pop("helper")
    helper.path.unlink()
    with pytest.raises(ValueError, match="helper is not withdrawn"):
        reclaim(state)
    if composed:
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            helper,
            b"",
            b"",
            state.compiler_lock,
            state.artifacts,
            20,
            recovery=linux_installation.RetirementState.HELPER_REMOVED,
        )
    else:
        reclaim(
            state, recovery=linux_installation.RetirementState.HELPER_REMOVED, removed_helper=helper
        )
    assert not list(state.root.iterdir()) and not state.config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [
        "path",
        "identity_length",
        "device",
        "type",
        "executable",
        "links",
        "digest",
        "present",
        "symlink",
        "stale_inventory",
    ],
)
def test_reclamation_refuses_invalid_known_removed_helper(reclamation, change):
    """Require the complete withdrawn record before accepting an absent helper."""
    from dataclasses import replace

    state = reclamation
    helper = state.artifacts.pop("helper")
    if change != "present":
        helper.path.unlink()
    identity = list(helper.identity)
    if change == "path":
        helper = replace(helper, path=state.root / "other")
    elif change == "identity_length":
        helper = replace(helper, identity=helper.identity[:2])
    elif change in ("device", "type", "executable", "links"):
        if change == "device":
            identity[0] += 1
        elif change == "type":
            identity[2] = stat.S_IFDIR | 0o444
        elif change == "executable":
            identity[2] = stat.S_IFREG | 0o555
        else:
            identity[3] = 2
        helper = replace(helper, identity=tuple(identity))
    elif change == "digest":
        helper = replace(helper, digest="lost")
    elif change == "symlink":
        helper.path.symlink_to(state.root / "absent")
    elif change == "stale_inventory":
        state.artifacts["helper"] = helper
    with pytest.raises(ValueError, match="Invalid known removed helper authority"):
        reclaim(
            state, recovery=linux_installation.RetirementState.HELPER_REMOVED, removed_helper=helper
        )
    assert state.config.receiver.exists() and state.config.reservation.path.exists()
    assert not state.stopped and state.closed


@pytest.mark.parametrize("remaining", ["namespace", "global"])
def test_removed_helper_continuation_refuses_remaining_policy(retirement, remaining):
    """Explicit helper loss cannot bypass any remaining enforcing namespace or global attachment."""
    state = retirement
    if remaining == "global":
        remove_policy(state)
    state.helper.path.chmod(0o444)
    helper = linux_host.bind_host_file(state.helper.path)
    helper.path.unlink()
    state.artifacts.pop("helper")
    state.helper = helper
    with pytest.raises(ValueError, match="Generation policy remains"):
        retire(state, recovery=linux_installation.RetirementState.HELPER_REMOVED)
    assert state.attachment_entry.exists() and state.config.receiver.exists()
    assert state.config.reservation.path.exists() and not state.stopped


@pytest.mark.parametrize("change", ["receiver", "unit", "tracking", "live"])
def test_removed_helper_continuation_requires_intact_drain_authority(reclamation, change):
    """Known helper loss never permits missing admission/tracking anchors or live membership."""
    state = reclamation
    helper = state.artifacts.pop("helper")
    helper.path.unlink()
    if change == "receiver":
        state.config.receiver.unlink()
    elif change == "unit":
        state.files["service_unit"].path.unlink()
    elif change == "tracking":
        state.change = "inactiveslice"
    else:
        state.events = b"populated 1\n"
    exception = (
        FileNotFoundError
        if change in ("receiver", "unit")
        else ValueError
        if change == "tracking"
        else OSError
    )
    with pytest.raises(exception, match="No such file|tracking is missing|live or ambiguous"):
        reclaim(
            state, recovery=linux_installation.RetirementState.HELPER_REMOVED, removed_helper=helper
        )
    assert state.config.reservation.path.exists() and not state.stopped


@pytest.mark.parametrize("symlink", [False, True])
def test_removed_helper_continuation_refuses_late_helper_reappearance(
    reclamation, monkeypatch, symlink
):
    """A helper recreated during final manager verification cannot release capacity."""
    state = reclamation
    helper = state.artifacts.pop("helper")
    helper.path.unlink()
    original = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Recreate authority only at the final public manager boundary."""
        result = original(process, *args, **kwargs)
        if state.stopped and "list-jobs" in process.argv:
            if symlink:
                helper.path.symlink_to(state.root / "missing")
            else:
                helper.path.write_bytes(b"unexpected helper")
                helper.path.chmod(0o444)
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(ValueError, match="Retired generation authority remains"):
        reclaim(
            state, recovery=linux_installation.RetirementState.HELPER_REMOVED, removed_helper=helper
        )
    assert linux_host.bind_host_file(state.config.reservation.path) == state.config.reservation


@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_reclamation_continues_known_removed_receiver(reclamation, composed, empty):
    """Require original bindings and fragments when the receiver is lost."""
    state = reclamation
    helper = state.artifacts.pop("helper")
    receiver = state.artifacts.pop("receiver")
    helper.path.unlink()
    receiver.path.unlink()
    if empty:
        state.artifacts.pop("cache/payload").path.unlink()
        (state.root / "cache").rmdir()
    with pytest.raises(ValueError if empty else FileNotFoundError):
        reclaim(
            state, recovery=linux_installation.RetirementState.HELPER_REMOVED, removed_helper=helper
        )
    if composed:
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            helper,
            b"",
            b"",
            state.compiler_lock,
            state.artifacts,
            20,
            recovery=linux_installation.RetirementState.RECEIVER_REMOVED,
        )
    else:
        reclaim(
            state,
            removed_helper=helper,
            recovery=linux_installation.RetirementState.RECEIVER_REMOVED,
            removed_receiver=receiver,
        )
    assert not list(state.root.iterdir()) and not state.config.reservation.path.exists()
    assert state.stopped and state.closed


@pytest.mark.parametrize(
    "change",
    [
        "path",
        "identity_length",
        "device",
        "type",
        "mode",
        "links",
        "digest",
        "present",
        "symlink",
        "drain_binding",
        "missing_helper",
        "stale_inventory",
    ],
)
def test_removed_receiver_requires_original_binding_and_exact_remaining_state(reclamation, change):
    """Retain capacity for wrong authority, a present receiver or stale inventory."""
    from dataclasses import replace

    state = reclamation
    helper = state.artifacts.pop("helper")
    receiver = state.artifacts.pop("receiver")
    helper.path.unlink()
    if change != "present":
        receiver.path.unlink()
    identity = list(receiver.identity)
    if change == "path":
        receiver = replace(receiver, path=state.root / "other")
    elif change == "identity_length":
        receiver = replace(receiver, identity=receiver.identity[:2])
    elif change in ("device", "type", "mode", "links"):
        if change == "device":
            identity[0] += 1
        elif change == "type":
            identity[2] = stat.S_IFDIR | 0o555
        elif change == "mode":
            identity[2] = stat.S_IFREG | 0o777
        else:
            identity[3] = 2
        receiver = replace(receiver, identity=tuple(identity))
    elif change == "digest":
        receiver = replace(receiver, digest="lost")
    elif change == "symlink":
        receiver.path.symlink_to(state.root / "absent")
    elif change == "stale_inventory":
        state.artifacts["receiver"] = receiver
    elif change == "drain_binding":
        receiver = replace(receiver, digest="0" * 64)
    if change != "drain_binding":
        state.files["receiver"] = receiver
    exception = FileNotFoundError if change == "stale_inventory" else ValueError
    with pytest.raises(exception, match="authority|fixed binding|No such file"):
        reclaim(
            state,
            removed_helper=None if change == "missing_helper" else helper,
            recovery=linux_installation.RetirementState.RECEIVER_REMOVED,
            removed_receiver=receiver,
        )
    assert state.config.reservation.path.exists() and not state.stopped
    assert state.artifacts.get("cache/payload").path.exists()


@pytest.mark.parametrize("change", ["namespace", "global", "unit", "tracking", "live", "compiler"])
def test_removed_receiver_preserves_policy_unit_tracking_and_serialization_gates(
    retirement, monkeypatch, change
):
    """Require enforcing policy, authenticated units, idle proof and locks after receiver loss."""
    state = retirement
    if change not in ("namespace", "global"):
        remove_policy(state)
        remove_attachment(state)
    elif change == "global":
        remove_policy(state)
    state.helper.path.chmod(0o444)
    helper = linux_host.bind_host_file(state.helper.path)
    state.artifacts.pop("helper")
    receiver = state.artifacts.pop("receiver")
    helper.path.unlink()
    receiver.path.unlink()
    if change == "unit":
        state.files["service_unit"].path.unlink()
    elif change == "tracking":
        state.change = "inactiveslice"
    elif change == "live":
        state.events = b"populated 1\n"
    elif change == "compiler":
        monkeypatch.setattr(
            linux_installation.fcntl,
            "flock",
            Mock(side_effect=[None, BlockingIOError("compiler busy")]),
        )
    exception = (
        FileNotFoundError
        if change == "unit"
        else OSError
        if change in ("live", "compiler")
        else ValueError
    )
    with pytest.raises(
        exception,
        match="policy remains|No such file|tracking is missing|live or ambiguous|compiler busy",
    ):
        reclaim(
            state,
            removed_helper=helper,
            recovery=linux_installation.RetirementState.RECEIVER_REMOVED,
            removed_receiver=receiver,
        )
    assert state.config.reservation.path.exists() and not state.stopped


@pytest.mark.parametrize("late", ["drain", "final_file", "final_symlink"])
def test_removed_receiver_reappearance_never_releases_capacity(reclamation, monkeypatch, late):
    """Refuse cleanup if the receiver returns during drain or manager checks."""
    state = reclamation
    helper = state.artifacts.pop("helper")
    receiver = state.artifacts.pop("receiver")
    helper.path.unlink()
    receiver.path.unlink()
    original = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Recreate the absent receiver at the selected public manager verification boundary."""
        result = original(process, *args, **kwargs)
        if "list-jobs" in process.argv and (late == "drain" or state.stopped):
            if late == "final_symlink":
                receiver.path.symlink_to(state.root / "missing")
            else:
                receiver.path.write_bytes(b"unexpected receiver")
                receiver.path.chmod(0o555)
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(
        ValueError, match="removed receiver authority|Retired generation authority remains"
    ):
        reclaim(
            state,
            removed_helper=helper,
            recovery=linux_installation.RetirementState.RECEIVER_REMOVED,
            removed_receiver=receiver,
        )
    assert linux_host.bind_host_file(state.config.reservation.path) == state.config.reservation


def test_receiver_removed_retirement_requires_retained_receiver_record(reclamation):
    """An explicit phase cannot reconstruct a missing original receiver binding."""
    state = reclamation
    del state.files["receiver"]
    closed = state.closed
    with pytest.raises(ValueError, match="lacks pre-deletion receiver binding"):
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            state.helper,
            b"",
            b"",
            state.compiler_lock,
            state.artifacts,
            20,
            recovery=linux_installation.RetirementState.RECEIVER_REMOVED,
        )
    assert state.config.reservation.path.exists() and state.closed == closed


@pytest.fixture
def stopped_tracking(reclamation):
    """Retain bindings and capacity after artifact deletion and tracking stop."""
    state = reclamation
    state.helper = state.artifacts.pop("helper")
    state.receiver = state.artifacts.pop("receiver")
    state.helper.path.unlink()
    state.receiver.path.unlink()
    state.artifacts.pop("cache/payload").path.unlink()
    (state.root / "cache").rmdir()
    state.stopped = True
    return state


@pytest.mark.parametrize("composed", [False, True])
def test_retirement_finishes_explicit_stopped_tracking(stopped_tracking, composed):
    """Release capacity last after cgroup absence and inactive bound records."""
    state = stopped_tracking
    with pytest.raises(ValueError, match="tracking is missing"):
        reclaim(
            state,
            removed_helper=state.helper,
            recovery=linux_installation.RetirementState.RECEIVER_REMOVED,
            removed_receiver=state.receiver,
        )
    calls = len(state.calls)
    if composed:
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            state.helper,
            b"",
            b"",
            state.compiler_lock,
            {},
            20,
            recovery=linux_installation.RetirementState.TRACKING_REMOVED,
        )
    else:
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.TRACKING_REMOVED,
        )
    assert not state.config.reservation.path.exists()
    assert not list(state.root.iterdir())
    assert all(
        not bound.path.exists()
        for bound in (
            state.config.slice_unit,
            state.files["socket_unit"],
            state.files["service_unit"],
        )
    )
    assert not any("stop" in call for call in state.calls[calls:])


@pytest.mark.parametrize(
    "change",
    [
        "active",
        "substate",
        "group",
        "present",
        "symlink",
        "alias",
        "mount",
        "jobs",
        "listener",
        "fragment",
        "instance",
        "unknown_file",
        "unknown_directory",
        "compiler",
        "artifacts",
        "directories",
        "receiver_binding",
    ],
)
def test_stopped_tracking_refuses_ambiguous_or_remaining_authority(
    stopped_tracking, monkeypatch, change
):
    """Keep kernel and original authority checks mandatory after tracking stops."""
    state = stopped_tracking
    if change in ("active", "substate", "group", "instance"):
        state.cleanup_change = {
            "active": "slice_active",
            "substate": "slice_substate",
            "group": "slice_group",
            "instance": "instance_live",
        }[change]
    elif change == "jobs":
        state.cleanup_change = "remaining_jobs"
    elif change in ("alias", "listener"):
        state.change = change
    elif change == "mount":
        state.mount = state.mount.replace(" / /sys/fs/cgroup", " /hidden /sys/fs/cgroup")
    elif change in ("present", "symlink"):
        original = Path.exists if change == "present" else Path.is_symlink
        monkeypatch.setattr(
            Path,
            "exists" if change == "present" else "is_symlink",
            lambda path: (
                True if str(path) == "/sys/fs/cgroup/loopgen_generation.slice" else original(path)
            ),
        )
    elif change == "fragment":
        state.files["service_unit"].path.unlink()
    elif change == "unknown_file":
        (state.root / "unknown").write_bytes(b"unknown")
    elif change == "unknown_directory":
        (state.root / "unknown").mkdir()
    elif change == "compiler":
        monkeypatch.setattr(
            linux_installation.fcntl,
            "flock",
            Mock(side_effect=[None, BlockingIOError("compiler busy")]),
        )
    elif change == "artifacts":
        path = state.root / "remaining"
        path.write_bytes(b"remaining")
        path.chmod(0o444)
        state.artifacts["remaining"] = linux_host.bind_host_file(path)
    exception = (
        FileNotFoundError
        if change == "fragment"
        else OSError
        if change == "compiler"
        else ValueError
    )
    with pytest.raises(
        exception,
        match=(
            "unsafe|remains|hierarchy|activation|No such file|differs|compiler busy|"
            "cleanup authority|Invalid explicit retirement state"
        ),
    ):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=None if change == "receiver_binding" else state.receiver,
            recovery=linux_installation.RetirementState.TRACKING_REMOVED,
            empty_directories={"unknown": (0, 0)} if change == "directories" else None,
        )
    assert linux_host.bind_host_file(state.config.reservation.path) == state.config.reservation
    assert not any("stop" in call for call in state.calls[-1:])


def test_stopped_drain_requires_original_removed_receiver(drain):
    """An explicit absent-tracking drain cannot use an intact receiver or inferred binding."""
    state = drain
    with (
        pytest.raises(ValueError, match="requires known removed receiver"),
        linux_installation.drained_generation(
            state.config,
            state.files,
            state.lock,
            20,
            recovery=linux_installation.RetirementState.TRACKING_REMOVED,
        ),
    ):
        pytest.fail("invalid stopped tracking yielded")
    assert not state.closed and state.config.reservation.path.exists()


@pytest.mark.parametrize("remaining", ["namespace", "global"])
def test_stopped_tracking_requires_actual_policy_absence(retirement, remaining):
    """Keep capacity while a namespace or global policy remains loaded."""
    state = retirement
    if remaining == "global":
        remove_policy(state)
    state.helper.path.chmod(0o444)
    helper = linux_host.bind_host_file(state.helper.path)
    helper.path.unlink()
    state.artifacts.pop("helper")
    receiver = state.artifacts.pop("receiver")
    receiver.path.unlink()
    state.artifacts.pop("cache/payload").path.unlink()
    (state.root / "cache").rmdir()
    state.stopped = True
    with pytest.raises(ValueError, match="Generation policy remains"):
        reclaim(
            state,
            removed_helper=helper,
            removed_receiver=receiver,
            recovery=linux_installation.RetirementState.TRACKING_REMOVED,
        )
    assert state.config.reservation.path.exists() and state.config.slice_unit.path.exists()


@pytest.mark.parametrize("late", ["before_fragments", "final"])
def test_stopped_tracking_rechecks_kernel_absence_before_fragment_deletion(
    stopped_tracking, monkeypatch, late
):
    """Tracking recreation before fragment deletion or final release retains capacity."""
    state = stopped_tracking
    appeared = []
    original_capture = linux_installation.supervise_process
    original_exists = Path.exists

    def capture(process, *args, **kwargs):
        """Recreate tracking at the public instance enumeration boundary."""
        result = original_capture(process, *args, **kwargs)
        if (
            late == "before_fragments"
            and "list-units" in process.argv
            and process.argv[-1].endswith("@*.service")
            or late == "final"
            and "list-jobs" in process.argv
            and not state.config.slice_unit.path.exists()
        ):
            appeared.append(True)
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: (
            bool(appeared)
            if str(path) == "/sys/fs/cgroup/loopgen_generation.slice"
            else original_exists(path)
        ),
    )
    with pytest.raises(ValueError, match="tracking remains"):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.TRACKING_REMOVED,
        )
    assert state.config.reservation.path.exists()
    assert state.config.slice_unit.path.exists() == (late == "before_fragments")


@pytest.fixture
def removed_slice(stopped_tracking, monkeypatch):
    """Preserve original bindings after checked tracking stop and known first fragment deletion."""
    state = stopped_tracking
    state.config.slice_unit.path.unlink()
    state.calls.clear()
    previous = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Model an implicit loaded slice with an empty manager fragment."""
        result = previous(process, *args, **kwargs)
        if "show" in process.argv and process.argv[-1].endswith(".slice"):
            result.stdout = result.stdout.replace(
                "FragmentPath=" + str(state.config.slice_unit.path), "FragmentPath="
            )
            if state.cleanup_change == "slice_fragment":
                result.stdout = result.stdout.replace(
                    "FragmentPath=\n", "FragmentPath=" + str(state.config.slice_unit.path) + "\n"
                )
            if state.cleanup_change == "slice_not_found":
                result.stdout = result.stdout.replace("LoadState=loaded", "LoadState=not-found")
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    return state


@pytest.mark.parametrize("composed", [False, True])
def test_retirement_continues_known_first_slice_fragment_deletion(removed_slice, composed):
    """Require original bindings and inert absence before final fragment cleanup."""
    state = removed_slice
    with pytest.raises(FileNotFoundError):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.TRACKING_REMOVED,
        )
    if composed:
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            state.helper,
            b"",
            b"",
            state.compiler_lock,
            {},
            20,
            recovery=linux_installation.RetirementState.SLICE_REMOVED,
        )
    else:
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.SLICE_REMOVED,
        )
    assert not state.config.reservation.path.exists()
    assert (
        not state.files["socket_unit"].path.exists()
        and not state.files["service_unit"].path.exists()
    )
    assert not any("stop" in call for call in state.calls)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "identity",
        "present",
        "symlink",
        "fragment",
        "not_found",
        "active",
        "group",
        "socket",
        "service",
        "unknown",
        "compiler",
        "tracking",
        "receiver",
    ],
)
def test_known_removed_slice_refuses_drift_and_incomplete_authority(
    removed_slice, monkeypatch, change
):
    """Require original authority, inventory and locks despite missing-state claims."""
    from dataclasses import replace

    state = removed_slice
    if change == "source":
        state.config = replace(
            state.config, slice_unit=replace(state.config.slice_unit, digest="a" * 64)
        )
    elif change == "identity":
        unit = state.config.slice_unit
        state.config = replace(
            state.config,
            slice_unit=replace(unit, identity=(unit.identity[0] + 1, *unit.identity[1:])),
        )
    elif change == "present":
        state.config.slice_unit.path.write_text("reappeared")
    elif change == "symlink":
        state.config.slice_unit.path.symlink_to(state.root / "absent")
    elif change in ("fragment", "not_found", "active", "group"):
        state.cleanup_change = {
            "fragment": "slice_fragment",
            "not_found": "slice_not_found",
            "active": "slice_active",
            "group": "slice_group",
        }[change]
    elif change in ("socket", "service"):
        state.files[change + "_unit"].path.unlink()
    elif change == "unknown":
        (state.root / "unknown").touch()
    elif change == "compiler":
        monkeypatch.setattr(
            linux_installation.fcntl,
            "flock",
            Mock(side_effect=[None, BlockingIOError("compiler busy")]),
        )
    exception = (
        FileNotFoundError
        if change in ("socket", "service")
        else BlockingIOError
        if change == "compiler"
        else ValueError
    )
    with pytest.raises(
        exception,
        match=(
            "fixed source|removed.*authority|Loaded generation authority|No such file|"
            "inventory differs|compiler busy|cleanup authority|Invalid explicit retirement state"
        ),
    ):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=None if change == "receiver" else state.receiver,
            recovery=linux_installation.RetirementState.SLICE_REMOVED
            if change != "tracking"
            else object(),
        )
    assert state.config.reservation.path.exists()
    assert not any("stop" in call for call in state.calls)


def test_known_removed_slice_drain_requires_explicit_removed_receiver(removed_slice):
    """A slice-absence declaration cannot yield without its original known removed receiver."""
    state = removed_slice
    with (
        pytest.raises(ValueError, match="requires explicit stopped tracking and receiver binding"),
        linux_installation.drained_generation(
            state.config,
            state.files,
            state.lock,
            20,
            recovery=linux_installation.RetirementState.SLICE_REMOVED,
        ),
    ):
        pytest.fail("incomplete known state yielded")
    assert state.config.reservation.path.exists()


@pytest.mark.parametrize("late", ["before_removal", "final"])
def test_known_removed_slice_reappearance_retains_capacity(removed_slice, monkeypatch, late):
    """A recreated slice file before remaining removal or final release blocks capacity reuse."""
    state = removed_slice
    previous = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Recreate the fixed slice path at a selected public manager verification boundary."""
        result = previous(process, *args, **kwargs)
        if (
            late == "before_removal"
            and "list-units" in process.argv
            and process.argv[-1].endswith("@*.service")
            or late == "final"
            and "daemon-reload" in process.argv
        ):
            state.config.slice_unit.path.write_text("recreated")
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(ValueError, match="removed|remains"):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.SLICE_REMOVED,
        )
    assert state.config.reservation.path.exists()


@pytest.fixture
def removed_socket(removed_slice, monkeypatch):
    """Retain original bindings after known slice and second socket-fragment deletion."""
    state = removed_slice
    state.files["socket_unit"].path.unlink()
    previous = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Model only the native inert not-found socket at the public manager boundary."""
        result = previous(process, *args, **kwargs)
        if "show" in process.argv and process.argv[-1].endswith(".socket"):
            lines = (
                result.stdout.replace("LoadState=loaded", "LoadState=not-found")
                .replace("Accept=yes", "Accept=no")
                .replace("RemoveOnStop=yes", "RemoveOnStop=no")
                .replace("FragmentPath=" + str(state.files["socket_unit"].path), "FragmentPath=")
                .splitlines()
            )
            result.stdout = (
                "\n".join("Listen=" if line.startswith("Listen=") else line for line in lines)
                + "\n"
            )
            replacements = {
                "socket_fragment": ("FragmentPath=\n", "FragmentPath=/other\n"),
                "socket_loaded": ("LoadState=not-found", "LoadState=loaded"),
                "socket_accept": ("Accept=no", "Accept=yes"),
                "socket_removal": ("RemoveOnStop=no", "RemoveOnStop=yes"),
            }
            if state.cleanup_change in replacements:
                before, after = replacements[state.cleanup_change]
                result.stdout = result.stdout.replace(before, after)
            if state.cleanup_change == "socket_listen":
                result.stdout = result.stdout.replace("Listen=\n", "Listen=/other (Stream)\n")
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    return state


@pytest.mark.parametrize("composed", [False, True])
def test_retirement_continues_known_second_socket_fragment_deletion(removed_socket, composed):
    """Require original bindings and absent manager/kernel state to remove the template."""
    state = removed_socket
    with pytest.raises(FileNotFoundError):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.SLICE_REMOVED,
        )
    if composed:
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            state.helper,
            b"",
            b"",
            state.compiler_lock,
            {},
            20,
            recovery=linux_installation.RetirementState.SOCKET_REMOVED,
        )
    else:
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.SOCKET_REMOVED,
        )
    assert (
        not state.config.reservation.path.exists() and not state.files["service_unit"].path.exists()
    )
    assert not any("stop" in call for call in state.calls)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "identity",
        "path",
        "binding",
        "present",
        "symlink",
        "fragment",
        "loaded",
        "accept",
        "removal",
        "listen",
        "template",
        "unknown",
        "compiler",
        "slice",
        "jobs",
        "listener",
    ],
)
def test_known_removed_socket_refuses_drift_and_incomplete_authority(
    removed_socket, monkeypatch, change
):
    """Require original source and manager inventory when the socket is lost."""
    from dataclasses import replace

    state = removed_socket
    socket = state.files["socket_unit"]
    if change == "source":
        state.files["socket_unit"] = replace(socket, digest="0" * 64)
    elif change == "identity":
        state.files["socket_unit"] = replace(
            socket, identity=(socket.identity[0] + 1, *socket.identity[1:])
        )
    elif change == "path":
        state.files["socket_unit"] = replace(socket, path=socket.path.with_name("other.socket"))
    elif change == "binding":
        state.files.pop("socket_unit")
    elif change == "present":
        socket.path.write_text("reappeared")
    elif change == "symlink":
        socket.path.symlink_to(state.root / "absent")
    elif change in ("fragment", "loaded", "accept", "removal", "listen"):
        state.cleanup_change = "socket_" + change
    elif change == "template":
        state.files["service_unit"].path.unlink()
    elif change == "unknown":
        (state.root / "unknown").touch()
    elif change == "compiler":
        monkeypatch.setattr(
            linux_installation.fcntl,
            "flock",
            Mock(side_effect=[None, BlockingIOError("compiler busy")]),
        )
    elif change == "jobs":
        state.cleanup_change = "remaining_jobs"
    elif change == "listener":
        state.change = "listener"
    exception = (
        FileNotFoundError
        if change == "template"
        else BlockingIOError
        if change == "compiler"
        else ValueError
    )
    with pytest.raises(
        exception,
        match=(
            "fixed source|removed.*authority|Loaded generation authority|Ambiguous generation unit|"
            "No such file|inventory differs|compiler busy|requires known slice|activation|"
            "listener remains|cleanup authority|Invalid explicit retirement state"
        ),
    ):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.SOCKET_REMOVED
            if change != "slice"
            else object(),
        )
    assert state.config.reservation.path.exists()
    assert not any("stop" in call for call in state.calls)


def test_known_removed_socket_drain_requires_known_slice(removed_socket):
    """Require the known slice state before declaring the next fragment removed."""
    state = removed_socket
    with (
        pytest.raises(ValueError, match="Invalid explicit retirement state"),
        linux_installation.drained_generation(
            state.config,
            state.files,
            state.lock,
            20,
            removed_receiver=state.receiver,
            recovery=object(),
        ),
    ):
        pytest.fail("incomplete explicit state yielded")
    assert state.config.reservation.path.exists()


@pytest.mark.parametrize("late", ["before_removal", "final"])
def test_known_removed_socket_reappearance_retains_capacity(removed_socket, monkeypatch, late):
    """Socket path reappearance before template removal or final release cannot free capacity."""
    state = removed_socket
    previous = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Recreate only the fixed socket file at the selected public manager boundary."""
        result = previous(process, *args, **kwargs)
        if (
            late == "before_removal"
            and "list-units" in process.argv
            and process.argv[-1].endswith("@*.service")
            or late == "final"
            and "daemon-reload" in process.argv
        ):
            state.files["socket_unit"].path.write_text("recreated")
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(ValueError, match="removed|remains"):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.SOCKET_REMOVED,
        )
    assert state.config.reservation.path.exists()


@pytest.fixture
def removed_template(removed_socket):
    """Preserve all original bindings with final template absent and no concrete service records."""
    state = removed_socket
    state.files["service_unit"].path.unlink()
    state.instance_count = 0
    return state


@pytest.mark.parametrize("composed", [False, True])
def test_retirement_finishes_known_final_template_deletion(removed_template, composed):
    """Release capacity only after original bindings and records are absent."""
    state = removed_template
    with pytest.raises(FileNotFoundError):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.SOCKET_REMOVED,
        )
    if composed:
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            state.helper,
            b"",
            b"",
            state.compiler_lock,
            {},
            20,
            recovery=linux_installation.RetirementState.TEMPLATE_REMOVED,
        )
    else:
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.TEMPLATE_REMOVED,
        )
    assert not state.config.reservation.path.exists() and not list(state.root.iterdir())
    assert not any("stop" in call or "reset-failed" in call for call in state.calls)
    with pytest.raises(FileNotFoundError):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.TEMPLATE_REMOVED,
        )


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "identity",
        "path",
        "binding",
        "present",
        "symlink",
        "instance",
        "unknown",
        "compiler",
        "socket",
        "jobs",
        "listener",
    ],
)
def test_known_removed_template_refuses_drift_and_remaining_authority(
    removed_template, monkeypatch, change
):
    """Do not reconstruct authority or release capacity from a final-state claim."""
    from dataclasses import replace

    state = removed_template
    template = state.files["service_unit"]
    if change == "source":
        state.files["service_unit"] = replace(template, digest="0" * 64)
    elif change == "identity":
        state.files["service_unit"] = replace(
            template, identity=(template.identity[0] + 1, *template.identity[1:])
        )
    elif change == "path":
        state.files["service_unit"] = replace(
            template, path=template.path.with_name("other@.service")
        )
    elif change == "binding":
        state.files.pop("service_unit")
    elif change == "present":
        template.path.write_text("reappeared")
    elif change == "symlink":
        template.path.symlink_to(state.root / "absent")
    elif change == "instance":
        state.instance_count = 1
    elif change == "unknown":
        (state.root / "unknown").touch()
    elif change == "compiler":
        monkeypatch.setattr(
            linux_installation.fcntl,
            "flock",
            Mock(side_effect=[None, BlockingIOError("compiler busy")]),
        )
    elif change == "jobs":
        state.cleanup_change = "remaining_jobs"
    elif change == "listener":
        state.change = "listener"
    exception = BlockingIOError if change == "compiler" else ValueError
    with pytest.raises(
        exception,
        match=(
            "fixed source|removed.*authority|instance records|inventory differs|compiler busy|"
            "cleanup authority|activation|listener remains|Invalid explicit retirement state"
        ),
    ):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.TEMPLATE_REMOVED
            if change != "socket"
            else object(),
        )
    assert state.config.reservation.path.exists()
    assert not any("stop" in call or "reset-failed" in call for call in state.calls)


def test_known_removed_template_drain_requires_known_socket(removed_template):
    """Final fragment declaration requires explicit known preceding socket deletion."""
    state = removed_template
    with (
        pytest.raises(ValueError, match="Invalid explicit retirement state"),
        linux_installation.drained_generation(
            state.config,
            state.files,
            state.lock,
            20,
            removed_receiver=state.receiver,
            recovery=object(),
        ),
    ):
        pytest.fail("incomplete explicit state yielded")
    assert state.config.reservation.path.exists()


@pytest.mark.parametrize("late", ["before_removal", "final", "instance"])
def test_known_removed_template_reappearance_retains_capacity(removed_template, monkeypatch, late):
    """A recreated template or service history before removal or final release retains capacity."""
    state = removed_template
    previous = linux_installation.supervise_process
    enumerations = []

    def capture(process, *args, **kwargs):
        """Recreate only fixed template authority at selected public manager boundaries."""
        if "list-units" in process.argv and process.argv[-1].endswith("@*.service"):
            enumerations.append(True)
            if len(enumerations) == 3:
                if late == "before_removal":
                    state.files["service_unit"].path.write_text("recreated")
                elif late == "instance":
                    state.instance_count = 1
        result = previous(process, *args, **kwargs)
        if late == "final" and "daemon-reload" in process.argv:
            state.files["service_unit"].path.write_text("recreated")
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(ValueError, match="removed|remains|instance records"):
        reclaim(
            state,
            removed_helper=state.helper,
            removed_receiver=state.receiver,
            recovery=linux_installation.RetirementState.TEMPLATE_REMOVED,
        )
    assert state.config.reservation.path.exists()


@pytest.mark.parametrize("composed", [False, True])
def test_reclamation_continues_known_empty_cache_directory(reclamation, monkeypatch, composed):
    """Preserve anchors when directory deletion fails or authority changes."""
    state = reclamation
    directory = state.root / "cache"
    payload = directory / "payload"
    original = Path.rmdir

    def rmdir(path):
        """Refuse only the owned cache removal at the filesystem boundary."""
        if path == directory:
            raise PermissionError(13, "cache directory denied", str(path))
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rmdir", rmdir)
        with pytest.raises(PermissionError, match="cache directory denied"):
            reclaim(state)
    assert not payload.exists() and directory.exists()
    assert state.config.receiver.exists() and state.artifacts["helper"].path.exists()
    assert state.config.reservation.path.exists() and state.closed and not state.stopped
    state.artifacts.pop("cache/payload")
    with pytest.raises(ValueError, match="inventory differs"):
        reclaim(state)
    details = directory.stat()
    directories = {"cache": (details.st_dev, details.st_ino)}
    if composed:
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            (0, 0),
            {},
            state.artifacts["helper"],
            b"",
            b"",
            state.compiler_lock,
            state.artifacts,
            20,
            recovery=linux_installation.RetirementState.ATTACHMENT_REMOVED,
            empty_directories=directories,
        )
    else:
        reclaim(state, empty_directories=directories)
    assert directories == {"cache": (details.st_dev, details.st_ino)}
    assert not list(state.root.iterdir()) and not state.config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [
        "oversize",
        "empty",
        "absolute",
        "alias",
        "traversal",
        "root",
        "resolved_alias",
        "identity",
        "nonempty",
    ],
)
def test_reclamation_refuses_invalid_empty_directory_authority(reclamation, monkeypatch, change):
    """Invalid, changed or nonempty prescribed directories retain all recovery authority."""
    state = reclamation
    directory = state.root / "known"
    directory.mkdir()
    details = directory.stat()
    directories = {"known": (details.st_dev, details.st_ino)}
    if change == "oversize":
        directories = {str(i): (0, 0) for i in range(257)}
    elif change in ("empty", "absolute", "alias", "traversal", "root"):
        name = {
            "empty": "",
            "absolute": str(directory),
            "alias": "known/.",
            "traversal": "../known",
            "root": ".",
        }[change]
        directories = {name: (details.st_dev, details.st_ino)}
    elif change == "resolved_alias":
        original = linux_installation.protected_path
        monkeypatch.setattr(
            linux_installation,
            "protected_path",
            lambda path, **kwargs: state.root if path == directory else original(path, **kwargs),
        )
    elif change == "identity":
        directories["known"] = (details.st_dev, details.st_ino + 1)
    else:
        (directory / "unexpected").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="authority|changed|not empty"):
        reclaim(state, empty_directories=directories)
    assert state.config.receiver.exists() and state.config.reservation.path.exists()
    assert state.artifacts["cache/payload"].path.exists()


@pytest.mark.parametrize("change", ["pin", "alias", "replacement", "deadline"])
def test_reclamation_refuses_directory_drift_and_deadline_before_anchor_removal(
    reclamation, monkeypatch, change
):
    """Directory pinning, identity and budget checks precede helper and receiver removal."""
    from types import SimpleNamespace

    state = reclamation
    directory = state.root / "cache"
    payload = directory / "payload"
    if change == "pin":
        original = os.fstat

        def fstat(descriptor):
            """Report a changed directory inode at its opened descriptor boundary."""
            details = original(descriptor)
            if stat.S_ISDIR(details.st_mode):
                return SimpleNamespace(st_dev=details.st_dev, st_ino=details.st_ino + 1)
            return details

        monkeypatch.setattr(os, "fstat", fstat)
    elif change == "alias":
        original = linux_installation.protected_path
        monkeypatch.setattr(
            linux_installation,
            "protected_path",
            lambda path, **kwargs: (
                state.root
                if path == directory and not payload.exists()
                else original(path, **kwargs)
            ),
        )
    elif change == "replacement":
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            """Replace only the emptied cache directory after actual file deletion."""
            result = original(path, *args, **kwargs)
            if path == payload:
                directory.rmdir()
                directory.mkdir()
            return result

        monkeypatch.setattr(Path, "unlink", unlink)
    else:
        calls = []

        def clock():
            """Expire only after the support file unlink and next helper deadline check."""
            if not payload.exists():
                calls.append(True)
            return 20 if len(calls) >= 2 else 10

        monkeypatch.setattr(linux_installation.time, "monotonic", clock)
    with pytest.raises(
        TimeoutError if change == "deadline" else ValueError, match="directory.*(changed|deadline)"
    ):
        reclaim(state)
    assert state.config.receiver.exists() and state.artifacts["helper"].path.exists()
    assert state.config.reservation.path.exists() and not state.stopped


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "oversize",
        "compiler_writable",
        "compiler_same",
        "reservation_name",
        "namespace",
        "attachment",
        "alias",
        "absolute",
        "canonical_alias",
        "traversal",
        "empty_name",
        "root_name",
        "bound_path",
        "writable",
        "missing_helper",
        "executable_helper",
        "unknown_file",
        "unknown_directory",
        "symlink",
        "mount",
        "record_inventory",
    ],
)
def test_reclamation_refuses_unsafe_fixed_inventory_before_deletion(
    reclamation, monkeypatch, change
):
    """Incomplete, unsafe or unexpected authority leaves every artifact and full budget retained."""
    from dataclasses import replace

    state = reclamation
    before = set(state.root.rglob("*"))
    budget_path = state.config.reservation.path
    if change == "empty":
        state.artifacts = {}
    elif change == "oversize":
        state.artifacts = {str(i): state.artifacts["helper"] for i in range(257)}
    elif change == "compiler_writable":
        state.compiler_lock = replace(
            state.compiler_lock,
            identity=(
                *state.compiler_lock.identity[:2],
                0o100644,
                *state.compiler_lock.identity[3:],
            ),
        )
    elif change == "compiler_same":
        state.compiler_lock = state.lock
    elif change == "reservation_name":
        state.config = replace(
            state.config,
            reservation=replace(
                state.config.reservation, path=state.config.reservation.path.with_name("other")
            ),
        )
    elif change == "namespace":
        (state.backing / "apparmor/policy/namespaces/generation").mkdir()
    elif change == "attachment":
        state.attachment_entry.mkdir()
        (state.attachment_entry / "name").write_bytes(b"loopgen_generation\n")
    elif change == "alias":
        previous = linux_installation.protected_path
        monkeypatch.setattr(
            linux_installation,
            "protected_path",
            lambda path, **kwargs: (
                Path("/other") if path == state.root else previous(path, **kwargs)
            ),
        )
    elif change in {
        "absolute",
        "canonical_alias",
        "traversal",
        "empty_name",
        "root_name",
        "bound_path",
    }:
        name = {
            "absolute": "/helper",
            "canonical_alias": "./helper",
            "traversal": "../helper",
            "empty_name": "",
            "root_name": ".",
            "bound_path": "wrong",
        }[change]
        state.artifacts[name] = state.artifacts.pop("helper")
    elif change == "writable":
        bound = state.artifacts["helper"]
        state.artifacts["helper"] = replace(
            bound, identity=(*bound.identity[:2], 0o100644, *bound.identity[3:])
        )
    elif change == "missing_helper":
        del state.artifacts["helper"]
    elif change == "executable_helper":
        state.artifacts["helper"].path.chmod(0o555)
        state.artifacts["helper"] = linux_host.bind_host_file(state.artifacts["helper"].path)
    elif change == "unknown_file":
        (state.root / "unknown").write_bytes(b"unknown")
    elif change == "unknown_directory":
        (state.root / "unknown").mkdir()
    elif change == "symlink":
        (state.root / "unknown").symlink_to("helper")
    elif change == "mount":
        state.mount += f"14 1 0:20 / {state.root}/cache rw - tmpfs tmpfs rw\n"
    else:
        (state.config.reservation.path.parent / "other").write_bytes(b"other")
    with pytest.raises(ValueError, match="authority|remains|withdrawn|inventory|Aliased|type"):
        reclaim(state)
    assert before <= set(state.root.rglob("*"))
    assert budget_path.exists() and not state.stopped


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_instance",
        "foreign_instance",
        "blank_instance",
        "instance_drift",
        "instance_live",
        "instance_missing",
        "instance_duplicate",
        "instance_malformed",
        "too_many",
    ],
)
def test_reclamation_authenticates_inactive_instances_before_deletion(reclamation, change):
    """Foreign, duplicate, live, malformed or excessive instance records prevent any unlink."""
    state = reclamation
    state.cleanup_change = change
    if change == "too_many":
        state.instance_count = 33
    with pytest.raises(ValueError, match="instances|service authority|unit authority"):
        reclaim(state)
    assert (
        state.config.receiver.exists()
        and state.config.reservation.path.exists()
        and not state.stopped
    )


@pytest.mark.parametrize(
    "change", ["slice_active", "slice_substate", "slice_group", "remaining_units", "remaining_jobs"]
)
def test_reclamation_retains_budget_after_partial_tracking_cleanup(reclamation, change):
    """Partial owned cleanup never frees capacity when final tracking or manager proof fails."""
    state = reclamation
    state.cleanup_change = change
    with pytest.raises(ValueError, match="tracking remains|authority remains"):
        reclaim(state)
    assert state.config.reservation.path.exists() and state.closed
    assert not state.config.receiver.exists()


def test_reclamation_refuses_live_members_and_compiler_contention(reclamation, monkeypatch):
    """Live membership and a concurrent compiler preserve the artifact set and full reservation."""
    state = reclamation
    state.events = b"populated 1\n"
    with pytest.raises(OSError, match="live"):
        reclaim(state)
    state.events = b"populated 0\n"
    monkeypatch.setattr(
        linux_installation.fcntl,
        "flock",
        Mock(side_effect=[None, BlockingIOError("compiler busy")]),
    )
    with pytest.raises(BlockingIOError, match="compiler busy"):
        reclaim(state)
    assert state.config.receiver.exists() and state.config.reservation.path.exists()


def test_reclamation_restores_reservation_when_release_sync_fails(reclamation, monkeypatch):
    """Restore the readonly budget when final directory sync fails."""
    state = reclamation
    real_sync = os.fsync
    failed = []

    def sync(descriptor):
        """Fail only the first directory sync after the reservation unlink."""
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not failed:
            failed.append(True)
            raise OSError("release sync failed")
        real_sync(descriptor)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(OSError, match="release sync failed"):
        reclaim(state)
    restored = linux_host.bind_host_file(state.config.reservation.path)
    assert linux_installation.generation_limits(restored, "generation") == (29, 14680064)
    assert stat.S_IMODE(restored.identity[2]) == 0o444 and not state.config.receiver.exists()


@pytest.mark.parametrize(
    "change",
    [
        "first_artifact",
        "late_artifact",
        "fragment",
        "walk_bound",
        "directory_alias",
        "directory_device",
        "artifact_deadline",
        "release_deadline",
    ],
)
def test_reclamation_retains_capacity_on_late_integrity_and_budget_failures(
    reclamation, monkeypatch, change
):
    """Changed artifacts, unsafe directories and expired stages cannot reclaim a reservation."""
    from dataclasses import replace

    state = reclamation
    exception, message = ValueError, "artifact|fragment|directory|inventory"
    if change in {"first_artifact", "late_artifact", "fragment"}:
        original = linux_installation.bind_host_file
        calls = []
        target = (
            state.config.slice_unit.path if change == "fragment" else state.artifacts["helper"].path
        )

        def bind(path):
            """Model a host authority replacement at the indicated public rebind boundary."""
            result = original(path)
            if path == target:
                calls.append(path)
                if (
                    change == "first_artifact"
                    or change == "late_artifact"
                    and len(calls) == 2
                    or change == "fragment"
                    and state.stopped
                ):
                    return replace(result, digest="changed")
            return result

        monkeypatch.setattr(linux_installation, "bind_host_file", bind)
    elif change == "walk_bound":
        for index in range(257):
            (state.root / str(index)).mkdir()
    elif change == "directory_alias":
        previous = linux_installation.protected_path
        monkeypatch.setattr(
            linux_installation,
            "protected_path",
            lambda path, **kwargs: (
                Path("/other") if path == state.root / "cache" else previous(path, **kwargs)
            ),
        )
    elif change == "directory_device":
        original_stat = Path.lstat

        def metadata(path):
            """Model an unexpected mounted filesystem beneath the artifact root."""
            result = original_stat(path)
            if path == state.root / "cache":
                values = list(result)
                values[2] += 1
                return os.stat_result(values)
            return result

        monkeypatch.setattr(Path, "lstat", metadata)
    else:
        original_capture = linux_installation.supervise_process
        expired = []

        def capture(process, *args, **kwargs):
            """Expire only after instance proof or the final post-cleanup job proof."""
            result = original_capture(process, *args, **kwargs)
            argv = process.argv
            if (
                change == "artifact_deadline"
                and "show" in argv
                and "@" in argv[-1]
                or change == "release_deadline"
                and "list-jobs" in argv
                and state.stopped
            ):
                expired.append(True)
            return result

        monkeypatch.setattr(linux_installation, "supervise_process", capture)
        monkeypatch.setattr(linux_installation.time, "monotonic", lambda: 20 if expired else 10)
        exception, message = TimeoutError, "deadline expired"
    with pytest.raises(exception, match=message):
        reclaim(state)
    assert state.config.reservation.path.exists() and state.closed


@pytest.mark.parametrize(
    "change",
    [
        "group",
        "group_symlink",
        "listener",
        "artifact",
        "artifact_symlink",
        "namespace",
        "attachment",
    ],
)
def test_reclamation_requires_all_retired_authority_absent_before_release(
    reclamation, monkeypatch, change
):
    """Residual kernel, listener or filesystem authority keeps the complete reservation."""
    state = reclamation
    original_exists = Path.exists
    original_link = Path.is_symlink
    group = Path("/sys/fs/cgroup/loopgen_generation.slice")
    listener = Path("/run/loopgen_generation.socket")
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: (
            True
            if state.stopped
            and (change == "group" and path == group or change == "listener" and path == listener)
            else original_exists(path)
        ),
    )
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: (
            True
            if state.stopped and change == "group_symlink" and path == group
            else original_link(path)
        ),
    )
    original_capture = linux_installation.supervise_process

    def capture(process, *args, **kwargs):
        """Expose a residual object after the actual fragment deletion/reload boundary."""
        result = original_capture(process, *args, **kwargs)
        if "daemon-reload" in process.argv:
            if change == "artifact":
                (state.root / "helper").write_bytes(b"unexpected")
            elif change == "artifact_symlink":
                (state.root / "helper").symlink_to("absent")
            elif change == "namespace":
                (state.backing / "apparmor/policy/namespaces/generation").mkdir()
            elif change == "attachment":
                state.attachment_entry.mkdir()
                (state.attachment_entry / "name").write_bytes(b"loopgen_generation\n")
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(ValueError, match="tracking remains|authority remains"):
        reclaim(state)
    assert state.config.reservation.path.exists() and state.closed


def test_reclamation_preserves_shared_artifact_ancestry_without_instance_records(reclamation):
    """Retain shared compiler locks when removing nearby cache entries."""
    state = reclamation
    state.instance_count = 0
    compiler = state.root / "cache/shared.lock"
    compiler.write_bytes(b"shared compiler")
    compiler.chmod(0o444)
    state.compiler_lock = linux_host.bind_host_file(compiler)
    reclaim(state)
    assert list((state.root / "cache").iterdir()) == [compiler]
    assert not state.config.reservation.path.exists()


def test_reclamation_never_deletes_a_manifest_entry_aliasing_shared_authority(reclamation):
    """A shared compiler lock cannot be declared as a removable generation artifact."""
    state = reclamation
    compiler = state.root / "shared.lock"
    compiler.write_bytes(b"compiler")
    compiler.chmod(0o444)
    state.compiler_lock = linux_host.bind_host_file(compiler)
    state.artifacts["shared.lock"] = state.compiler_lock
    with pytest.raises(ValueError, match="Invalid generation artifact authority"):
        reclaim(state)
    assert compiler.exists() and state.config.reservation.path.exists()


def test_reclamation_requires_final_record_at_its_original_path(reclamation):
    """A displaced management binding cannot become cleanup authority."""
    state = reclamation
    state.artifacts["management.json"] = state.artifacts["helper"]
    with pytest.raises(ValueError, match="Invalid retained management record authority"):
        reclaim(state)
    assert state.config.reservation.path.exists()


@pytest.mark.parametrize("change", ["unchanged", "symlink"])
def test_reclamation_reports_incomplete_reservation_unlink_without_overwriting_it(
    reclamation, monkeypatch, change
):
    """A surviving or aliased capacity record is never overwritten or reported released."""
    state = reclamation
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        """Model a reported unlink that fails its explicit absence postcondition."""
        if path == state.config.reservation.path:
            if change == "symlink":
                original_unlink(path)
                path.symlink_to("absent")
        else:
            original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(ValueError, match="reservation remains"):
        reclaim(state)
    assert state.config.reservation.path.exists() or state.config.reservation.path.is_symlink()


@pytest.mark.parametrize(
    "record", ["slice", "socket", "service", "pid", "loaded", "duplicate", "blank"]
)
def test_reclamation_accepts_only_known_inert_manager_records(reclamation, monkeypatch, record):
    """Cached not-found unit identities are harmless only when every authority field is empty."""
    from types import SimpleNamespace

    state = reclamation
    original_capture = linux_installation.supervise_process
    final = []
    name = {"slice": "loopgen_generation.slice", "socket": "loopgen_generation.socket"}.get(
        record, "loopgen_generation@0.service"
    )

    def capture(process, *args, **kwargs):
        """Model cached identities independently of actual fragment and artifact removal."""
        argv = process.argv
        if "list-units" in argv and argv[-1] == "loopgen_generation*":
            final.append(True)
            output = (
                " " if record == "blank" else name + ("\n" + name if record == "duplicate" else "")
            )
        elif "show" in argv and final:
            fields = {
                "Id": name,
                "LoadState": "not-found",
                "FragmentPath": "",
                "DropInPaths": "",
                "NeedDaemonReload": "no",
                "Transient": "no",
                "ActiveState": "inactive",
                "ControlGroup": "",
            }
            if name.endswith(".service"):
                fields.update(MainPID="0", ControlPID="0")
            if record == "pid":
                fields["MainPID"] = "1"
            if record == "loaded":
                fields["LoadState"] = "loaded"
            output = "".join(f"{key}={value}\n" for key, value in fields.items())
        else:
            return original_capture(process, *args, **kwargs)
        return SimpleNamespace(
            status=linux_installation.ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout=output,
            stderr="",
            stdout_discarded=0,
            stderr_discarded=0,
        )

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    if record in {"pid", "loaded", "duplicate", "blank"}:
        with pytest.raises(ValueError, match="authority remains"):
            reclaim(state)
        assert state.config.reservation.path.exists()
    else:
        reclaim(state)
        assert not state.config.reservation.path.exists()
    assert state.stopped and not state.config.receiver.exists()


@pytest.fixture
def retirement(reclamation):
    """Supply a complete bound generation with real files and controlled native collaborators."""
    state = reclamation
    state.removed.clear()
    directory = state.backing / "apparmor/policy/namespaces/generation"
    for name, profiles in state.policies.items():
        target = directory / "namespaces" / name if name else directory
        (target / "namespaces").mkdir(parents=True)
        (target / "profiles").mkdir()
        for index, (name, data) in enumerate(profiles.items()):
            entry = target / "profiles" / str(index)
            entry.mkdir()
            (entry / "name").write_text(name + "\n")
            (entry / "mode").write_text("enforce\n")
            (entry / "raw_data").write_bytes(data)
    metadata = directory.stat()
    state.namespace_identity = metadata.st_dev, metadata.st_ino
    state.helper_path = state.root / "helper"
    state.helper_path.chmod(0o555)
    state.helper = linux_host.bind_host_file(state.helper_path)
    state.artifacts["helper"] = state.helper
    state.attachment_source = (
        f'abi <abi/4.0>,\nprofile loopgen_generation "{state.helper_path}" {{ }}\n'.encode()
    )
    state.attachment_entry.mkdir()
    for name, value in {
        "name": b"loopgen_generation\n",
        "mode": b"enforce\n",
        "attach": str(state.helper_path).encode() + b"\n",
        "raw_data": state.attachment_bytes,
    }.items():
        (state.attachment_entry / name).write_bytes(value)
    state.commands.clear()
    return state


@pytest.fixture
def untracked_retirement(bound_owner, monkeypatch):
    """Model exact absent policy and tracking with systemd's inert implicit slice."""
    fixture = bound_owner
    state = fixture.state
    shutil.rmtree(state.backing / "apparmor/policy/namespaces/generation")
    shutil.rmtree(state.attachment_entry)
    for bound in (
        state.config.slice_unit,
        state.files["socket_unit"],
        state.files["service_unit"],
    ):
        bound.path.unlink()
    state.recovery_failure = ""
    state.native_calls = []
    state.recovery_events = []

    def capture(process, *args, **kwargs):
        """Return only the fixed absent unit state and bounded recovery inventory."""
        argv = process.argv
        state.native_calls.append(argv)
        if "show" in argv:
            name = argv[-1]
            if name == state.config.slice_unit.path.name:
                fields = {
                    "Id": name,
                    "LoadState": "loaded",
                    "FragmentPath": "",
                    "DropInPaths": "",
                    "NeedDaemonReload": "no",
                    "Transient": "no",
                    "ActiveState": "inactive",
                    "SubState": "dead",
                    "ControlGroup": "",
                    "Delegate": "no",
                }
            elif name == state.files["socket_unit"].path.name:
                fields = {
                    "Id": name,
                    "LoadState": "not-found",
                    "FragmentPath": "",
                    "DropInPaths": "",
                    "NeedDaemonReload": "no",
                    "Transient": "no",
                    "ActiveState": "inactive",
                    "SubState": "dead",
                    "ControlGroup": "",
                    "Accept": "no",
                    "RemoveOnStop": "no",
                }
                if state.recovery_failure == "active_socket":
                    fields.update(
                        LoadState="loaded",
                        FragmentPath=str(state.files["socket_unit"].path),
                        ActiveState="active",
                        SubState="listening",
                        Listen="/run/loopgen_generation.socket (Stream)",
                    )
            else:
                raise AssertionError(f"Unexpected manager query: {argv!r}")
            output = "".join(f"{key}={value}\n" for key, value in fields.items())
        elif "list-units" in argv and argv[-1].endswith("@*.service"):
            output = ""
        elif "list-units" in argv and argv[-1] == "loopgen_generation*":
            output = (
                "loopgen_generation.unexpected\n"
                if state.recovery_failure == "unknown_record"
                else "loopgen_generation.slice\n"
            )
        elif "list-jobs" in argv:
            output = ""
        else:
            raise AssertionError(f"Unexpected manager operation: {argv!r}")
        return SimpleNamespace(
            status=linux_installation.ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout=output,
            stderr="",
            stdout_discarded=0,
            stderr_discarded=0,
        )

    def recursive_idle(group, *, absent=False):
        """Record every absent-membership proof and optionally model a live cgroup."""
        state.recovery_events.append(("cgroup", absent))
        assert group.name == state.config.slice_unit.path.name
        assert absent
        if state.recovery_failure == "populated":
            raise OSError("Generation is live or ambiguous; confinement retained.")

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    monkeypatch.setattr(linux_installation, "_recursive_idle", recursive_idle)
    return fixture


@pytest.fixture
def absent_tracking_cleanup(bound_owner, monkeypatch):
    """Prepare direct reclamation checks after all policy and tracking authority is absent."""
    fixture = bound_owner
    state = fixture.state
    shutil.rmtree(state.backing / "apparmor/policy/namespaces/generation")
    shutil.rmtree(state.attachment_entry)
    for bound in (
        state.config.slice_unit,
        state.files["socket_unit"],
        state.files["service_unit"],
    ):
        bound.path.unlink()
    monkeypatch.setattr(
        linux_installation,
        "_unit_state",
        lambda *args, **kwargs: {
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "",
        },
    )
    monkeypatch.setattr(linux_installation, "_retired_instances", lambda *args, **kwargs: [])
    monkeypatch.setattr(linux_installation, "_recursive_idle", lambda *args, **kwargs: None)
    return fixture


def reclaim_untracked(state, *, recovery=linux_installation.RetirementState.ATTACHMENT_REMOVED):
    """Exercise the private finalizer with exact previously authenticated test authority."""
    from contextlib import nullcontext

    linux_installation._reclaim_generation(
        state.config,
        state.files,
        state.lock,
        state.compiler_lock,
        state.artifacts,
        20,
        nullcontext(),
        None,
        None,
        None,
        recovery,
        tracking_absent=True,
    )


def retire(state, *, recovery=linux_installation.RetirementState.COMPLETE):
    """Run the complete retirement through the public exclusive transaction owner."""
    linux_installation.retire_generation(
        state.config,
        state.files,
        state.lock,
        state.namespace_identity,
        state.policies,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
        state.compiler_lock,
        state.artifacts,
        20,
        recovery=recovery,
    )


@pytest.mark.parametrize("initial_group", [False, True])
def test_observed_retirement_retries_after_artifact_removal(retirement, monkeypatch, initial_group):
    """An idle generation retires whether its slice cgroup exists before drain."""
    state = retirement
    record = state.root / "management.json"
    record.write_bytes(b"original management record")
    record.chmod(0o444)
    authority = linux_host.bind_host_file(record)
    owner = linux_installation.GenerationLifecycle(
        state.config,
        state.files,
        state.lock,
        state.compiler_lock,
        dict(state.artifacts),
        state.namespace_identity,
        state.policies,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
    )
    group = Path("/sys/fs/cgroup") / state.config.slice_unit.path.name
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: (
            initial_group and not state.stopped if path == group else original_exists(path)
        ),
    )
    owner._retire_observed(authority, 20)
    assert not state.config.reservation.path.exists()
    assert not record.exists()


def test_observed_retirement_retries_an_interrupted_artifact(retirement, monkeypatch):
    """A deletion fault leaves enough original authority for direct checked retry."""
    state = retirement
    record = state.root / "management.json"
    record.write_bytes(b"original management record")
    record.chmod(0o444)
    authority = linux_host.bind_host_file(record)
    owner = linux_installation.GenerationLifecycle(
        state.config,
        state.files,
        state.lock,
        state.compiler_lock,
        dict(state.artifacts),
        state.namespace_identity,
        state.policies,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
    )
    group = Path("/sys/fs/cgroup") / state.config.slice_unit.path.name
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: not state.stopped if path == group else original_exists(path),
    )
    original_unlink = Path.unlink
    target = min(
        bound.path for bound in state.artifacts.values() if bound.path != state.helper.path
    )
    interrupted = False

    def unlink(path, *args, **kwargs):
        """Fail after one exact artifact is gone, then allow a fresh retry."""
        nonlocal interrupted
        result = original_unlink(path, *args, **kwargs)
        if path == target and not interrupted:
            interrupted = True
            raise OSError("artifact unlink interrupted")
        return result

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="artifact unlink interrupted"):
        owner._retire_observed(authority, 20)
    assert interrupted and record.exists() and state.config.reservation.path.exists()
    owner._retire_observed(authority, 20)
    assert not record.exists() and not state.config.reservation.path.exists()


def test_retirement_refuses_record_rebind_after_capacity_release(bound_owner, monkeypatch):
    """The original record remains when its final post-release rebind changes."""
    fixture = bound_owner
    state = fixture.state
    group = Path("/sys/fs/cgroup") / state.config.slice_unit.path.name
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: not state.stopped if path == group else original_exists(path),
    )
    original_bind = linux_installation.bind_host_file

    def bind(path):
        """Change only the record rebind after the owned reservation is gone."""
        current = original_bind(path)
        if path == fixture.record.path and not state.config.reservation.path.exists():
            return replace(current, digest="changed")
        return current

    monkeypatch.setattr(linux_installation, "bind_host_file", bind)
    with pytest.raises(ValueError, match="Management record changed before removal"):
        fixture.owner.retire(fixture.record, 20)
    assert fixture.record.path.exists() and not state.config.reservation.path.exists()


def test_observed_retirement_refuses_out_of_order_receiver_loss(retirement):
    """Missing receiver cannot be treated as a checked prefix while helper remains."""
    state = retirement
    record = state.root / "management.json"
    record.write_bytes(b"original management record")
    record.chmod(0o444)
    authority = linux_host.bind_host_file(record)
    owner = linux_installation.GenerationLifecycle(
        state.config,
        state.files,
        state.lock,
        state.compiler_lock,
        dict(state.artifacts),
        state.namespace_identity,
        state.policies,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
    )
    state.config.receiver.unlink()
    with pytest.raises(ValueError, match="fixed order"):
        owner.retire(authority, 20)
    assert record.exists() and state.config.reservation.path.exists()


def test_observed_retirement_does_not_recover_with_partial_tracking(bound_owner):
    """Missing only some runtime fragments cannot authorize unconfined-helper withdrawal."""
    fixture = bound_owner
    state = fixture.state
    shutil.rmtree(state.backing / "apparmor/policy/namespaces/generation")
    shutil.rmtree(state.attachment_entry)
    state.files["socket_unit"].path.unlink()

    with pytest.raises(ValueError, match="Attachment absent while helper remains executable"):
        fixture.owner.retire(fixture.record, 20)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o555
    assert fixture.record.path.exists() and state.config.reservation.path.exists()


def test_observed_retirement_recovers_absent_policy_and_tracking(untracked_retirement, monkeypatch):
    """Withdraw an unconfined helper only after exact native authority and cgroup absence."""
    fixture = untracked_retirement
    state = fixture.state
    original_fchmod = os.fchmod

    def fchmod(descriptor, mode):
        """Record helper withdrawal ordering while preserving the real mode change."""
        state.recovery_events.append(("chmod", mode))
        return original_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "fchmod", fchmod)
    fixture.owner.retire(fixture.record, 20)

    assert not fixture.record.path.exists() and not state.config.reservation.path.exists()
    assert not state.config.receiver.exists() and not state.helper_path.exists()
    assert all(not bound.path.exists() for bound in state.files.values())
    assert state.recovery_events.index(("cgroup", True)) < state.recovery_events.index(
        ("chmod", 0o444)
    )
    assert not any("stop" in call or "daemon-reload" in call for call in state.native_calls)


@pytest.mark.parametrize(
    ("failure", "error", "message"),
    [
        ("active_socket", ValueError, "authority changed"),
        ("populated", OSError, "live or ambiguous"),
        ("unknown_record", ValueError, "authority remains"),
    ],
)
def test_untracked_retirement_refuses_live_or_ambiguous_authority(
    untracked_retirement, failure, error, message
):
    """Do not withdraw an executable helper until every tracking absence is authenticated."""
    fixture = untracked_retirement
    state = fixture.state
    state.recovery_failure = failure

    with pytest.raises(error, match=message):
        fixture.owner.retire(fixture.record, 20)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o555
    assert fixture.record.path.exists() and state.config.reservation.path.exists()
    assert state.config.receiver.exists()


def test_missing_tracking_recovery_requires_attachment_removed_phase(absent_tracking_cleanup):
    """Direct drain and cleanup entry points reject unrelated recovery phases."""
    fixture = absent_tracking_cleanup
    state = fixture.state

    with (
        pytest.raises(ValueError, match="Missing tracking recovery"),
        linux_installation.drained_generation(
            state.config,
            state.files,
            state.lock,
            20,
            tracking_absent=True,
        ),
    ):
        pass
    with pytest.raises(ValueError, match="Missing tracking recovery"):
        reclaim_untracked(state, recovery=linux_installation.RetirementState.COMPLETE)
    with pytest.raises(ValueError, match="Missing tracking recovery"):
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            state.namespace_identity,
            state.policies,
            state.helper,
            state.attachment_bytes,
            state.attachment_source,
            state.compiler_lock,
            state.artifacts,
            20,
            tracking_absent=True,
        )
    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o555


def test_missing_tracking_recovery_requires_original_helper(absent_tracking_cleanup):
    """Do not withdraw or reclaim when the authenticated artifact inventory lost the helper."""
    fixture = absent_tracking_cleanup
    state = fixture.state
    artifacts = dict(state.artifacts)
    artifacts.pop("helper")
    state.artifacts = artifacts

    with pytest.raises(ValueError, match="lacks original helper authority"):
        reclaim_untracked(state)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o555
    assert state.config.reservation.path.exists()


def test_missing_tracking_recovery_obeys_deadline_before_helper_withdrawal(
    absent_tracking_cleanup, monkeypatch
):
    """An expired transaction leaves the still-executable helper and capacity untouched."""
    fixture = absent_tracking_cleanup
    state = fixture.state
    monkeypatch.setattr(linux_installation.time, "monotonic", lambda: 21)

    with pytest.raises(TimeoutError, match="artifact removal deadline"):
        reclaim_untracked(state)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o555
    assert state.config.reservation.path.exists()


def test_missing_tracking_recovery_revalidates_helper_before_withdrawal(
    absent_tracking_cleanup, monkeypatch
):
    """A helper rebind mismatch refuses before any mode change or artifact deletion."""
    fixture = absent_tracking_cleanup
    state = fixture.state
    previous = linux_installation.bind_host_file

    def bind(path):
        """Change the current helper binding presented at the withdrawal boundary."""
        current = previous(path)
        return replace(current, digest="0" * 64) if path == state.helper_path else current

    monkeypatch.setattr(linux_installation, "bind_host_file", bind)
    with pytest.raises(ValueError, match="helper differs from fixed authority"):
        reclaim_untracked(state)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o555
    assert state.config.reservation.path.exists()


def test_missing_tracking_recovery_rejects_unsafe_helper_mode(absent_tracking_cleanup):
    """Only the original executable mode or an already withdrawn helper may continue."""
    fixture = absent_tracking_cleanup
    state = fixture.state
    state.helper_path.chmod(0o400)
    state.artifacts["helper"] = linux_host.bind_host_file(state.helper_path)

    with pytest.raises(ValueError, match="unsafe mode"):
        reclaim_untracked(state)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o400
    assert state.config.reservation.path.exists()


def test_missing_tracking_recovery_accepts_already_withdrawn_helper(
    absent_tracking_cleanup, monkeypatch
):
    """An interrupted safe withdrawal can retry from the exact already-inert helper inode."""
    fixture = absent_tracking_cleanup
    state = fixture.state
    state.helper_path.chmod(0o444)
    state.artifacts["helper"] = linux_host.bind_host_file(state.helper_path)
    previous = linux_installation.bind_host_file
    calls = 0

    def bind(path):
        """Fail the post-withdrawal rebind after accepting the already-inert source."""
        nonlocal calls
        current = previous(path)
        if path == state.helper_path:
            calls += 1
            if calls == 2:
                return replace(current, digest="0" * 64)
        return current

    monkeypatch.setattr(linux_installation, "bind_host_file", bind)
    with pytest.raises(ValueError, match="changed during withdrawal"):
        reclaim_untracked(state)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o444
    assert state.config.reservation.path.exists()


def test_missing_tracking_recovery_revalidates_withdrawn_helper(
    absent_tracking_cleanup, monkeypatch
):
    """A changed helper digest after mode withdrawal retains capacity for authenticated retry."""
    fixture = absent_tracking_cleanup
    state = fixture.state
    previous = linux_installation.bind_host_file
    calls = 0

    def bind(path):
        """Change only the binding returned after the atomic helper withdrawal."""
        nonlocal calls
        current = previous(path)
        if path == state.helper_path:
            calls += 1
            if calls == 2:
                return replace(current, digest="0" * 64)
        return current

    monkeypatch.setattr(linux_installation, "bind_host_file", bind)
    with pytest.raises(ValueError, match="changed during withdrawal"):
        reclaim_untracked(state)

    assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o444
    assert state.config.reservation.path.exists()


def test_retired_records_reject_loaded_non_slice_as_implicit(reclamation, monkeypatch):
    """Only the already-verified implicit slice may remain loaded during missing-tracking cleanup."""
    state = reclamation
    result = SimpleNamespace(
        status=linux_installation.ProcessCaptureStatus.COMPLETED,
        exit_code=0,
        stdout="loopgen_generation.socket\n",
        stderr="",
        stdout_discarded=0,
        stderr_discarded=0,
    )
    monkeypatch.setattr(linux_installation, "supervise_process", lambda *args, **kwargs: result)

    with pytest.raises(ValueError, match="Unexpected implicit retired manager record"):
        linux_installation._retired_records_absent(
            state.tool,
            "loopgen_generation",
            {"loopgen_generation.socket"},
            20,
            loaded_implicit={"loopgen_generation.socket"},
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing_child",
        "missing_attachment",
        "helper_drift",
        "namespace_alias",
        "attachment_without_withdrawal",
        "artifact_with_attachment",
        "slice_missing",
        "capacity_missing",
    ],
)
def test_observed_retirement_refuses_changed_native_prefix(bound_owner, change):
    """Direct retry refuses native and file states outside the original removal sequence."""
    fixture = bound_owner
    state = fixture.state
    owner = fixture.owner
    namespace = state.backing / "apparmor/policy/namespaces/generation"
    if change == "missing_child":
        owner = replace(owner, policies={**owner.policies, "other": {"bootstrap": b"compiled"}})
        message = "namespace inventory differs"
    elif change == "missing_attachment":
        shutil.rmtree(state.attachment_entry)
        message = "attachment removed while namespace"
    elif change == "helper_drift":
        state.helper_path.chmod(0o644)
        state.helper_path.write_bytes(b"changed")
        state.helper_path.chmod(0o555)
        message = "Withdrawn helper differs"
    elif change == "namespace_alias":
        shutil.rmtree(namespace)
        namespace.symlink_to(state.backing / "missing")
        message = "namespace was replaced with an alias"
    elif change == "attachment_without_withdrawal":
        shutil.rmtree(namespace)
        shutil.rmtree(state.attachment_entry)
        message = "Attachment absent while helper remains executable"
    elif change == "artifact_with_attachment":
        shutil.rmtree(namespace)
        target = min(
            bound.path
            for bound in state.artifacts.values()
            if bound.path not in {state.helper.path, state.config.receiver}
        )
        target.unlink()
        message = "Artifacts removed while attachment remains"
    elif change == "slice_missing":
        state.config.slice_unit.path.unlink()
        message = "Unit removal differs from the fixed order"
    else:
        state.config.reservation.path.unlink()
        message = "Capacity absent before native retirement"
    with pytest.raises(ValueError, match=message):
        owner.retire(fixture.record, 20)
    assert fixture.record.path.exists()


@pytest.mark.parametrize(
    "change", ["child_removed", "helper_removed", "receiver_unbound", "artifacts_unbound"]
)
def test_observed_retirement_derives_only_ordered_remaining_authority(
    bound_owner, monkeypatch, change
):
    """Retry derives its phase from original files and refuses malformed surviving ownership."""
    fixture = bound_owner
    state = fixture.state
    owner = fixture.owner
    namespace = state.backing / "apparmor/policy/namespaces/generation"
    if change == "child_removed":
        shutil.rmtree(namespace / "namespaces" / "inner")
    elif change == "helper_removed":
        shutil.rmtree(namespace)
        shutil.rmtree(state.attachment_entry)
        for bound in state.artifacts.values():
            if bound.path != state.config.receiver:
                bound.path.unlink()
    elif change == "receiver_unbound":
        owner = replace(
            owner,
            artifacts={
                name: bound
                for name, bound in owner.artifacts.items()
                if bound.path != state.config.receiver
            },
        )
        state.config.receiver.unlink()
    else:
        shutil.rmtree(namespace)
        shutil.rmtree(state.attachment_entry)
        state.config.receiver.unlink()
        state.helper.path.unlink()
        owner = replace(
            owner,
            artifacts={
                name: bound
                for name, bound in owner.artifacts.items()
                if bound.path not in {state.config.receiver, state.helper.path}
            },
        )
    retire = Mock()
    monkeypatch.setattr(linux_installation, "retire_generation", retire)
    if change in {"receiver_unbound", "artifacts_unbound"}:
        with pytest.raises(
            ValueError,
            match="Receiver absent while helper remains|Artifacts remain after tracking removal",
        ):
            owner.retire(fixture.record, 20)
        retire.assert_not_called()
    else:
        owner.retire(fixture.record, 20)
        assert retire.call_args.kwargs["recovery"] == (
            linux_installation.RetirementState.COMPLETE
            if change == "child_removed"
            else linux_installation.RetirementState.HELPER_REMOVED
        )
        if change == "child_removed":
            assert set(retire.call_args.args[4]) == {""}


def test_observed_retirement_reloads_after_interrupted_unit_removal(retirement, monkeypatch):
    """Retry reconciles a removed unit before checking the manager's cached view."""
    state = retirement
    record = state.root / "management.json"
    record.write_bytes(b"original management record")
    record.chmod(0o444)
    authority = linux_host.bind_host_file(record)
    owner = linux_installation.GenerationLifecycle(
        state.config,
        state.files,
        state.lock,
        state.compiler_lock,
        dict(state.artifacts),
        state.namespace_identity,
        state.policies,
        state.helper,
        state.attachment_bytes,
        state.attachment_source,
    )
    group = Path("/sys/fs/cgroup") / state.config.slice_unit.path.name
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: not state.stopped if path == group else original_exists(path),
    )
    recovery_manager(SimpleNamespace(state=state), monkeypatch)
    original_unlink = Path.unlink
    interrupted = False

    def unlink(path, *args, **kwargs):
        """Model an error immediately after the fixed slice fragment is deleted."""
        nonlocal interrupted
        result = original_unlink(path, *args, **kwargs)
        if path == state.config.slice_unit.path and not interrupted:
            interrupted = True
            raise OSError("unit unlink interrupted")
        return result

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="unit unlink interrupted"):
        owner._retire_observed(authority, 20)
    assert interrupted and record.exists() and state.config.reservation.path.exists()
    owner._retire_observed(authority, 20)
    assert not record.exists() and not state.config.reservation.path.exists()


@pytest.mark.parametrize("inert_instance", [False, True])
def test_observed_retirement_finishes_after_capacity_release(
    installation_owner, monkeypatch, inert_instance
):
    """The original record can be removed after verified native and capacity absence."""
    fixture = installation_owner
    state = fixture.state
    group = Path("/sys/fs/cgroup") / state.config.slice_unit.path.name
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: not state.stopped if path == group else original_exists(path),
    )
    recovery_manager(fixture, monkeypatch)
    original_unlink = Path.unlink
    interrupted = False

    def unlink(path, *args, **kwargs):
        """Leave the original record after successful capacity release."""
        nonlocal interrupted
        if path == fixture.record.path and not interrupted:
            interrupted = True
            raise OSError("record unlink interrupted")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="record unlink interrupted"):
        fixture.owner._retire_observed(fixture.record, 20)
    assert interrupted and fixture.record.path.exists()
    assert not state.config.reservation.path.exists()
    if inert_instance:
        previous = linux_installation._manager
        inert = "loopgen_generation@retired.service"
        monkeypatch.setattr(
            linux_installation,
            "_manager",
            lambda tool, arguments, deadline: (
                inert + " loaded inactive dead\n"
                if arguments[0] == "list-units" and arguments[-1].endswith("*")
                else previous(tool, arguments, deadline)
            ),
        )
        checked = Mock()
        monkeypatch.setattr(linux_installation, "_retired_records_absent", checked)
    fixture.owner._retire_observed(fixture.record, 20)
    assert not fixture.record.path.exists()
    if inert_instance:
        assert inert in checked.call_args.args[2]


@pytest.mark.parametrize("change", ["manager", "jobs", "inventory", "record"])
def test_observed_retirement_checks_final_record_removal(installation_owner, monkeypatch, change):
    """A retry after capacity release still checks manager, private files, and original record."""
    fixture = installation_owner
    state = fixture.state
    group = Path("/sys/fs/cgroup") / state.config.slice_unit.path.name
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: not state.stopped if path == group else original_exists(path),
    )
    recovery_manager(fixture, monkeypatch)
    original_unlink = Path.unlink

    def interrupt(path, *args, **kwargs):
        """Stop only after the original capacity has been released."""
        if path == fixture.record.path:
            raise OSError("record removal interrupted")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt)
        with pytest.raises(OSError, match="record removal interrupted"):
            fixture.owner.retire(fixture.record, 20)
    assert not state.config.reservation.path.exists()
    if change in {"manager", "jobs"}:
        original_manager = linux_installation._manager
        monkeypatch.setattr(
            linux_installation,
            "_manager",
            lambda tool, arguments, deadline: (
                ("unknown.slice loaded inactive dead" if change == "manager" else "1 pending")
                if arguments[0] == ("list-units" if change == "manager" else "list-jobs")
                and (change != "manager" or arguments[-1].endswith("*"))
                else original_manager(tool, arguments, deadline)
            ),
        )
        message = "Unknown retired manager record|Retired native authority remains"
    elif change == "inventory":
        (state.root / "unknown").write_bytes(b"changed")
        message = "Unexpected private files"
    else:
        original_manager = linux_installation._manager

        def change_record(tool, arguments, deadline):
            """Change the record after its verified descriptor has entered the final scope."""
            result = original_manager(tool, arguments, deadline)
            if arguments[0] == "list-jobs":
                fixture.record.path.chmod(0o644)
                fixture.record.path.write_bytes(b"changed")
                fixture.record.path.chmod(0o444)
            return result

        monkeypatch.setattr(linux_installation, "_manager", change_record)
        message = "Original management record changed"
    with pytest.raises(ValueError, match=message):
        fixture.owner.retire(fixture.record, 20)
    assert fixture.record.path.exists()


def test_retirement_holds_management_authority_across_all_removal_stages(retirement, monkeypatch):
    """One drain serializes every native stage and releases capacity only after complete cleanup."""
    import fcntl

    state = retirement
    previous = linux_installation.supervise_process
    checked = []

    def capture(process, *args, **kwargs):
        """Attempt a competing management operation at each manager boundary."""
        with state.lock.open_verified() as descriptor, pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(process.argv)
        return previous(process, *args, **kwargs)

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    original = dict(state.artifacts)
    retire(state)
    assert checked and state.closed and not state.config.reservation.path.exists()
    assert state.artifacts == original
    assert state.removed == [state.namespace / "namespaces/inner", state.namespace]
    assert state.commands == [b"loopgen_generation\0"]
    assert not state.config.receiver.exists() and not state.helper_path.exists()
    with pytest.raises(FileNotFoundError):
        retire(state)
    assert linux_installation.reserve_generation(
        state.config.reservation.path.parent, state.lock, "next"
    )


def test_activation_accepts_only_exact_empty_native_socket_manager_group(activation):
    """Allow the listener cgroup only after recursive kernel idle proof."""
    state = activation
    state.socket_group = "/system.slice/loopgen_generation.socket"
    assert activate(state) == (state.config, state.files)
    assert state.listener_open


def test_activation_native_socket_group_population_refuses_and_closes_listener(
    activation, monkeypatch
):
    """A populated socket-manager group fails the repaired check and retains all policy/capacity."""
    state = activation
    state.socket_group = "/system.slice/loopgen_generation.socket"
    previous = linux_installation.supervise_process

    def capture(process, *args, **kw):
        """Introduce population only after the fixed socket's actual start boundary."""
        result = previous(process, *args, **kw)
        if state.attempted:
            state.events = b"populated 1\n"
        return result

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    with pytest.raises(OSError, match="live or ambiguous"):
        activate(state)
    assert not state.listener_open and state.config.reservation.path.exists()


@pytest.mark.parametrize("change", ["helper", "live", "namespace", "attachment", "artifact"])
def test_retirement_stops_on_partial_failure_without_releasing_capacity(retirement, change):
    """Keep admission closed and capacity reserved for unsafe or partial stages."""
    state = retirement
    expected, message = ValueError, "helper|policy|inventory"
    if change == "helper":
        state.artifacts.pop("helper")
    elif change == "live":
        state.events = b"populated 1\n"
        expected, message = OSError, "live"
    elif change == "namespace":
        state.policies["inner"]["payload"] = b"changed"
    elif change == "attachment":
        state.attachment_bytes = b"changed"
        message = "compilation"
    else:
        (state.root / "unknown").write_bytes(b"unknown")
    with pytest.raises(expected, match=message):
        retire(state)
    assert state.config.reservation.path.exists() and state.config.receiver.exists()
    if change in {"attachment", "artifact"}:
        assert not state.namespace.exists()
    if change == "artifact":
        assert stat.S_IMODE(state.helper_path.stat().st_mode) == 0o444
        assert not state.attachment_entry.exists()
    else:
        assert state.attachment_entry.exists()


@pytest.mark.parametrize("change", ["inode", "bytes", "mode"])
def test_retirement_revalidates_withdrawn_helper_before_artifact_cleanup(
    retirement, monkeypatch, change
):
    """Late helper drift cannot reach artifact deletion or capacity release after policy removal."""
    from dataclasses import replace

    state = retirement
    previous = linux_installation.bind_host_file

    def bind(path):
        """Change only the final withdrawn helper proof after exact global removal."""
        result = previous(path)
        if path == state.helper_path and not state.attachment_entry.exists():
            if change == "bytes":
                return replace(result, digest="changed")
            identity = list(result.identity)
            identity[0 if change == "inode" else 2] += 1
            return replace(result, identity=tuple(identity))
        return result

    monkeypatch.setattr(linux_installation, "bind_host_file", bind)
    with pytest.raises(ValueError, match="helper changed after withdrawal"):
        retire(state)
    assert state.config.reservation.path.exists() and state.config.receiver.exists()
    assert not state.namespace.exists() and not state.attachment_entry.exists()


def test_retirement_finishes_authenticated_remaining_outer_namespace(retirement, monkeypatch):
    """Revalidate outer authority before completing retirement."""
    state = retirement
    original = Path.rmdir
    original_policies = state.policies

    def remove(path):
        """Leave only the fixed outer namespace after a child-first failure."""
        if path == state.namespace:
            raise OSError("outer namespace removal failed")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rmdir", remove)
        with pytest.raises(OSError, match="outer namespace removal failed"):
            retire(state)
    assert state.namespace.exists() and state.attachment_entry.exists()
    assert stat.S_IMODE(state.helper.path.stat().st_mode) == 0o555
    assert state.config.receiver.exists() and state.config.reservation.path.exists()
    with pytest.raises(ValueError, match="Unexpected generation namespace inventory"):
        retire(state)
    state.policies = {"": dict(original_policies[""])}
    retire(state)
    assert not state.namespace.exists() and not state.attachment_entry.exists()
    assert not state.config.reservation.path.exists() and not list(state.root.iterdir())
    assert original_policies.keys() == {"", "inner"}


@pytest.mark.parametrize(
    "phase",
    [
        linux_installation.RetirementState.NAMESPACE_REMOVED,
        linux_installation.RetirementState.ATTACHMENT_REMOVED,
    ],
)
def test_retirement_finishes_only_explicit_intact_partial_authority(retirement, monkeypatch, phase):
    """Authenticated continuation completes remaining stages under an exclusive drain."""
    import fcntl

    state = retirement
    remove_policy(state)
    if phase == linux_installation.RetirementState.ATTACHMENT_REMOVED:
        remove_attachment(state)
    previous = linux_installation.supervise_process
    checked = []

    def capture(process, *args, **kwargs):
        """Require the continuation lock at every manager boundary."""
        with state.lock.open_verified() as descriptor, pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(process.argv)
        return previous(process, *args, **kwargs)

    monkeypatch.setattr(linux_installation, "supervise_process", capture)
    original = dict(state.artifacts)
    retire(state, recovery=phase)
    assert checked and not state.config.reservation.path.exists()
    assert not state.config.receiver.exists() and state.artifacts == original
    assert state.commands == [b"loopgen_generation\0"]
    with pytest.raises(FileNotFoundError):
        retire(state, recovery=phase)


@pytest.mark.parametrize("phase", ["", "missing", None, []])
def test_retirement_rejects_unknown_explicit_continuation_before_mutation(retirement, phase):
    """Unrecognized phase never authorizes removing or bypassing existing authority."""
    state = retirement
    with pytest.raises(ValueError, match="Invalid explicit retirement state"):
        retire(state, recovery=phase)
    assert not state.removed and not state.commands and state.config.reservation.path.exists()


@pytest.mark.parametrize(
    "phase",
    [
        linux_installation.RetirementState.NAMESPACE_REMOVED,
        linux_installation.RetirementState.ATTACHMENT_REMOVED,
    ],
)
def test_retirement_continuation_requires_real_policy_absence(retirement, phase):
    """An explicit phase cannot bypass remaining namespace/global authority or executable helper."""
    state = retirement
    message = (
        "namespace remains"
        if phase == linux_installation.RetirementState.NAMESPACE_REMOVED
        else "helper changed"
    )
    with pytest.raises(ValueError, match=message):
        retire(state, recovery=phase)
    assert not state.removed and not state.commands and state.config.reservation.path.exists()


@pytest.mark.parametrize("remaining", ["namespace", "global"])
def test_retirement_continuation_preserves_capacity_for_remaining_policy(retirement, remaining):
    """Withdrawn helper alone cannot authorize cleanup while generation policy remains."""
    state = retirement
    if remaining == "global":
        remove_policy(state)
    state.helper_path.chmod(0o444)
    with pytest.raises(ValueError, match="Generation policy remains"):
        retire(state, recovery=linux_installation.RetirementState.ATTACHMENT_REMOVED)
    assert state.config.reservation.path.exists() and state.config.receiver.exists()
    assert state.attachment_entry.exists()


@pytest.mark.parametrize("change", ["artifact", "tracking"])
def test_retirement_continuation_cannot_resume_missing_artifacts_or_tracking(retirement, change):
    """Missing artifact or inactive tracking refuses without treating partial state as success."""
    state = retirement
    remove_policy(state)
    remove_attachment(state)
    if change == "artifact":
        state.config.receiver.unlink()
        with pytest.raises(FileNotFoundError):
            retire(state, recovery=linux_installation.RetirementState.ATTACHMENT_REMOVED)
    else:
        state.change = "inactiveslice"
        with pytest.raises(ValueError, match="tracking is missing or inactive"):
            retire(state, recovery=linux_installation.RetirementState.ATTACHMENT_REMOVED)
    assert state.config.reservation.path.exists()


@pytest.mark.parametrize(
    "change",
    [None, "name", "empty", "mode", "bootstrap", "capacity", "bytes", "duplicate", "flags"],
)
def test_generation_policy_cache_authenticates_add_only_payloads(retirement, monkeypatch, change):
    """Bound cache inspection refuses unknown, ambiguous or over-budget enforcing exports."""
    state = retirement
    profiles = state.backing / "apparmor/policy/namespaces/generation/profiles"
    entry = profiles / "payload"
    entry.mkdir()
    name = "payload_" + "a" * 64
    (entry / "name").write_text(("foreign" if change == "name" else name) + "\n")
    (entry / "raw_data").write_bytes(b"" if change == "empty" else b"payload")
    (entry / "mode").write_bytes(b"complain" if change == "mode" else b"enforce")
    monkeypatch.setattr(
        linux_installation,
        "generation_limits",
        lambda *args: (
            0 if change == "capacity" else 29,
            1 if change == "bytes" else 14 * 1024 * 1024,
        ),
    )
    if change == "bootstrap":
        for item in profiles.iterdir():
            if (item / "name").read_text().strip() in state.policies[""]:
                (item / "raw_data").write_bytes(b"changed")
                break
    if change == "duplicate":
        duplicate = profiles / "duplicate"
        duplicate.mkdir()
        for child in entry.iterdir():
            (duplicate / child.name).write_bytes(child.read_bytes())
    if change in {None, "flags"}:
        result = linux_installation.generation_policy_cache(
            state.config,
            state.namespace_identity,
            state.policies,
            state.helper,
            state.attachment_bytes,
        )
        assert result[""][name] == b"payload"
        assert result[""]["bootstrap"] == state.policies[""]["bootstrap"]
        if change == "flags":
            state.attachment_source = state.attachment_source.replace(
                b'" {', b'" flags=(attach_disconnected) {'
            )
        linux_installation.retire_generation(
            state.config,
            state.files,
            state.lock,
            state.namespace_identity,
            state.policies,
            state.helper,
            state.attachment_bytes,
            state.attachment_source,
            state.compiler_lock,
            state.artifacts,
            20,
            cached_payloads=True,
        )
        assert not state.namespace.exists()
    else:
        with pytest.raises(ValueError, match="cache"):
            linux_installation.generation_policy_cache(
                state.config,
                state.namespace_identity,
                state.policies,
                state.helper,
                state.attachment_bytes,
            )
