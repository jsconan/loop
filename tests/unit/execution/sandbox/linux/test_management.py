"""Verify original management authority persistence, safe status and exact checked uninstall."""

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import host as linux_host
from loop.execution.sandbox.linux import installation as linux_installation
from loop.execution.sandbox.linux import management as linux_management
from loop.execution.sandbox.linux.admission import GenerationAdmission

from .helpers import publish, rewrite


@pytest.fixture(autouse=True)
def native_inventory(monkeypatch):
    """Keep fixed host tracking and listener inventory independent of real installations."""
    original = Path.glob

    def glob(path, pattern, **kwargs):
        """Expose only disposable inventory unless a case injects its own foreign host state."""
        if path in {Path("/sys/fs/cgroup"), Path("/run")}:
            return iter(())
        return original(path, pattern, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)


def test_original_record_status_export_and_uninstall(installation, monkeypatch):
    """Persistent authority supplies the exact original transaction after reconstruction."""
    record, arguments = installation
    authority = publish(installation)
    assert authority.path == record
    assert record.stat().st_mode & 0o777 == 0o444
    assert "authenticated" in linux_management.generation_status(record)
    assert linux_management.uninstall_authority(record) == (
        authority,
        arguments[-1]["loop_uninstall.py"],
    )
    retire = Mock()
    monkeypatch.setattr(linux_installation.GenerationLifecycle, "retire", retire)
    monkeypatch.setattr(linux_management.time, "monotonic", lambda: 10)
    linux_management.uninstall_generation(record, 123)
    assert retire.call_args.args[0] == authority
    assert retire.call_args.args[1] == 123
    with pytest.raises(FileExistsError, match="already exists"):
        publish(installation)
    assert linux_host.bind_host_file(record) == authority


def test_saved_record_cannot_rebind_a_different_existing_file(installation):
    """A changed saved digest does not adopt the surviving artifact's current bytes."""
    record, _ = installation
    publish(installation)
    rewrite(record, lambda value: value["files"]["receiver"].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="Saved installation authority changed"):
        linux_management.generation_status(record)


def test_absent_record_requires_empty_private_and_native_inventory(tmp_path, monkeypatch):
    """Idempotent uninstall proves the private root and native settings absent."""
    root = tmp_path / "native"
    unit_directory = tmp_path / "units"
    unit_directory.mkdir()
    policy = tmp_path / "policy"
    (policy / "namespaces").mkdir(parents=True)
    (policy / "profiles").mkdir()
    seed = root / "management.lock"

    def bind(path):
        """Bind the disposable shared seed and isolate unrelated native tooling."""
        if path != seed:
            return Mock()
        details = seed.stat()
        return linux_host.HostFile(
            seed,
            (
                details.st_dev,
                details.st_ino,
                details.st_mode,
                details.st_nlink,
                details.st_size,
                details.st_mtime_ns,
                details.st_ctime_ns,
            ),
            hashlib.sha256(seed.read_bytes()).hexdigest(),
        )

    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", unit_directory)
    monkeypatch.setattr(linux_management, "protected_path", lambda path, **kwargs: path)
    monkeypatch.setattr(linux_management, "_policy_root", lambda: (policy, 1))
    monkeypatch.setattr(linux_management, "_policy_directory", Mock())
    monkeypatch.setattr(linux_management, "_manager", Mock(return_value=""))
    monkeypatch.setattr(linux_management, "_retired_records_absent", Mock())
    monkeypatch.setattr(linux_management, "bind_host_file", bind)
    linux_management.verify_uninstalled(root, 20)
    root.mkdir()
    seed.write_bytes(linux_management.NATIVE_LOCK_SEEDS["management.lock"])
    seed.chmod(0o444)
    linux_management.verify_uninstalled(root, 20)
    staging = unit_directory / (".loop-unit-native_" + "a" * 32 + "-interrupted")
    staging.write_bytes(b"partial")
    with pytest.raises(ValueError, match="unit fragments remain"):
        linux_management.verify_uninstalled(root, 20)
    staging.unlink()
    (root / "unknown").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="Partial private installation inventory"):
        linux_management.verify_uninstalled(root, 20)


@pytest.mark.parametrize(
    "case",
    [
        "root_alias",
        "seed",
        "capacity",
        "capacity_alias",
        "unit_alias",
        "unit",
        "unknown_record",
        "duplicate_record",
        "jobs",
        "tracking",
        "listener",
        "namespace",
        "attachment",
        "namespace_limit",
        "profile_limit",
    ],
)
def test_absent_record_refuses_remaining_authority(tmp_path, monkeypatch, case):
    """No-record uninstall rejects each private, systemd, cgroup, socket, and policy residue."""
    root = tmp_path / "native"
    root.mkdir()
    units = tmp_path / "units"
    units.mkdir()
    policy = tmp_path / "policy"
    (policy / "namespaces").mkdir(parents=True)
    (policy / "profiles").mkdir()
    prefix = linux_management.NATIVE_UNIT_PREFIX + "native_" + "a" * 32
    valid = prefix + ".slice"
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", units)
    monkeypatch.setattr(linux_management, "_policy_root", lambda: (policy, 1))
    monkeypatch.setattr(linux_management, "_policy_directory", Mock())
    tool = Mock()
    tool.open_verified.return_value = nullcontext(0)
    monkeypatch.setattr(linux_management, "bind_host_file", Mock(return_value=tool))
    monkeypatch.setattr(linux_management, "_retired_records_absent", Mock())
    monkeypatch.setattr(
        linux_management,
        "protected_path",
        lambda path, **kwargs: (
            path.parent
            if case == "root_alias" and path == root or case == "unit_alias" and path == units
            else path
        ),
    )
    rows = ""
    jobs = ""
    if case == "seed":
        (root / "management.lock").write_bytes(b"wrong")
    elif case == "capacity":
        (root / "reservations").mkdir()
        (root / "reservations" / "occupied").write_bytes(b"held")
    elif case == "capacity_alias":
        (root / "reservations").mkdir()
        monkeypatch.setattr(
            linux_management,
            "protected_path",
            lambda path, **kwargs: path.parent if path == root / "reservations" else path,
        )
    elif case == "unit":
        (units / valid).write_bytes(b"fragment")
    elif case == "unknown_record":
        rows = "unknown.slice loaded inactive dead\n"
    elif case == "duplicate_record":
        rows = f"{valid} loaded inactive dead\n{valid} loaded inactive dead\n"
    elif case == "jobs":
        jobs = "1 pending"
    elif case == "namespace":
        (policy / "namespaces" / ("native_" + "a" * 32)).mkdir()
    elif case == "attachment":
        entry = policy / "profiles" / "0"
        entry.mkdir()
        (entry / "name").write_text(prefix + "\n")
    elif case in {"namespace_limit", "profile_limit"}:
        selected = policy / ("namespaces" if case == "namespace_limit" else "profiles")
        original_iterdir = Path.iterdir
        monkeypatch.setattr(
            Path,
            "iterdir",
            lambda path: (
                iter([path / "other"] * 4097) if path == selected else original_iterdir(path)
            ),
        )
        monkeypatch.setattr(linux_management, "read_policy_bytes", lambda *args: b"other")
    monkeypatch.setattr(
        linux_management,
        "_manager",
        lambda _tool, arguments, _deadline: jobs if arguments[0] == "list-jobs" else rows,
    )
    if case in {"tracking", "listener"}:
        original_glob = Path.glob
        selected = Path("/sys/fs/cgroup") if case == "tracking" else Path("/run")
        monkeypatch.setattr(
            Path,
            "glob",
            lambda path, pattern, **kwargs: (
                iter([path / "remaining"])
                if path == selected
                else original_glob(path, pattern, **kwargs)
            ),
        )
    with pytest.raises(
        ValueError,
        match=(
            "alias|seed|private|capacity|unit|manager|jobs|tracking|"
            "listener|namespace|profile|attachment"
        ),
    ):
        linux_management.verify_uninstalled(root, 20)


@pytest.mark.parametrize(
    "case",
    [
        "clean",
        "changed",
        "temporary",
        "linked",
        "linked_receipt",
        "native",
        "bad_temporary",
        "bad_receipt",
        "bad_unit",
        "active_unit",
        "unknown_child",
        "bad_profile",
        "unsafe_temporary",
        "unknown_link",
        "bad_unit_temporary",
        "jobs",
        "instances",
        "listener",
        "missing_helper_attachment",
        "bad_compiler_source",
        "bad_attachment_source",
        "unknown_file",
        "root_alias",
        "mount",
        "native_profile",
        "native_child",
        "nested_child",
        "record_present",
        "helper_withdrawn",
        "directory_alias",
        "temporary_alias",
        "temporary_changed_before_open",
        "temporary_changed_after_read",
        "unit_directory_alias",
        "unknown_native_unit",
        "foreign_unit_temporary",
        "foreign_manager",
        "foreign_tracking",
        "foreign_listener",
        "foreign_namespace",
        "foreign_attachment",
        "current_manager_record",
        "unrelated_attachment",
        "attachment_limit",
    ],
)
def test_unrecorded_stage_is_cleaned_from_reviewed_bytes(tmp_path, monkeypatch, case):
    """A pre-record failure removes only exact reviewed stage files and its reservation."""
    from contextlib import nullcontext

    root = tmp_path / "native"
    root.mkdir()
    reservations = root / "reservations"
    reservations.mkdir()
    generation = "native_" + "a" * 32
    reservation = reservations / generation
    reservation.write_text(
        f"{generation}\n{linux_installation._GENERATION_PROFILES}\n"
        f"{linux_installation._GENERATION_BYTES}\n"
    )
    reservation.chmod(0o444)
    for name, data in linux_management.NATIVE_LOCK_SEEDS.items():
        path = root / name
        path.write_bytes(data)
        path.chmod(0o444)
    runtime = {
        "loop_receiver.py": b"receiver",
        "helper": b"helper",
        "loop_uninstall.py": b"uninstall",
        "cache.lock": b"cache",
        "loop/execution/sandbox/linux/bootstrap.py": b"bootstrap",
    }
    for name, data in runtime.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o555 if name in {"loop_receiver.py", "helper", "loop_uninstall.py"} else 0o444)
    tool = tmp_path / "systemctl"
    tool.write_bytes(b"tool")
    tool.chmod(0o555)

    def bind(path):
        """Bind only disposable files through the real descriptor verifier."""
        path = tool if path == Path("/usr/bin/systemctl") else path
        details = path.stat()
        return linux_host.HostFile(
            path,
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

    config = GenerationAdmission(
        generation,
        1000,
        1000,
        root / "loop_receiver.py",
        linux_host.HostFile(root / f"loopgen_{generation}.slice", (), ""),
        bind(reservation),
    )
    policy = tmp_path / "policy"
    (policy / "namespaces").mkdir(parents=True)
    (policy / "profiles").mkdir()
    units = tmp_path / "units"
    units.mkdir()
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", units)
    monkeypatch.setattr(linux_management, "_administration", lambda lock: nullcontext())
    monkeypatch.setattr(linux_management, "bind_host_file", bind)
    monkeypatch.setattr(linux_management, "protected_path", lambda path, **kwargs: path)
    monkeypatch.setattr(linux_management.os, "listxattr", lambda path: [], raising=False)
    monkeypatch.setattr(linux_management, "_policy_root", lambda: (policy, 1))
    monkeypatch.setattr(linux_management, "_policy_directory", Mock())
    monkeypatch.setattr(linux_management, "_attachment_entry", lambda *args: None)
    monkeypatch.setattr(linux_management, "_manager", Mock(return_value=""))
    monkeypatch.setattr(linux_management, "_recursive_idle", Mock())
    monkeypatch.setattr(linux_management, "_retired_records_absent", Mock())
    monkeypatch.setattr(
        linux_management,
        "_retired_properties",
        lambda tool, name, fields, deadline: {
            "LoadState": "loaded" if (units / name).exists() else "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "",
            "FragmentPath": str(units / name) if (units / name).exists() else "",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
            "Transient": "no",
        },
    )
    monkeypatch.setattr(linux_management, "verify_uninstalled", Mock())
    monkeypatch.setattr(
        linux_management,
        "compile_policy_source",
        lambda data, deadline, names=False: (
            (b"bootstrap\n" if data == b"profile" else f"loopgen_{generation}\n".encode())
            if names
            else b"compiled"
        ),
    )
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "1 0 0:1 / / rw - tmpfs tmpfs rw\n"
            if path == Path("/proc/self/mountinfo")
            else read_text(path, *args, **kwargs)
        ),
    )
    if case == "changed":
        helper = root / "helper"
        helper.chmod(0o644)
        helper.write_bytes(b"changed")
        helper.chmod(0o555)
    elif case == "temporary":
        (root / ".loop-file-pending").write_bytes(b"hel")
        (root / ".receipt-pending").write_bytes(b"partial")
        (units / f".loop-unit-{generation}-pending").write_bytes(
            next(iter(config.unit_sources().values())).encode()[:12]
        )
        for pending in (*root.glob(".*-pending"), *units.glob(".*-pending")):
            pending.chmod(0o600)
    elif case == "linked":
        os.link(root / "helper", root / ".loop-file-linked")
    elif case == "linked_receipt":
        receipt = root / "receipt.json"
        receipt.write_bytes(b"partial")
        receipt.chmod(0o444)
        os.link(receipt, root / ".receipt-linked")
    elif case == "native":
        import shutil

        for name, source in config.unit_sources().items():
            fragment = units / name
            fragment.write_bytes(source.encode())
            fragment.chmod(0o444)
        namespace = policy / "namespaces" / generation
        (namespace / "namespaces").mkdir(parents=True)
        (namespace / "profiles").mkdir()
        attachment = policy / "attachment"
        attachment.mkdir()
        monkeypatch.setattr(
            linux_management,
            "_attachment_entry",
            lambda *args: attachment if attachment.exists() else None,
        )
        monkeypatch.setattr(linux_management, "_attest_attachment", Mock())
        monkeypatch.setattr(linux_management, "_policy_directory", lambda *args: None)
        original_rmdir = Path.rmdir
        monkeypatch.setattr(
            Path,
            "rmdir",
            lambda path: shutil.rmtree(path) if path == namespace else original_rmdir(path),
        )
        remove_attachment = Mock(side_effect=lambda *args, **kwargs: shutil.rmtree(attachment))
        monkeypatch.setattr(linux_management, "_remove_generation_attachment", remove_attachment)
    elif case in {"native_profile", "native_child", "nested_child"}:
        import shutil

        namespace = policy / "namespaces" / generation
        (namespace / "namespaces").mkdir(parents=True)
        (namespace / "profiles").mkdir()
        if case == "native_profile":
            entry = namespace / "profiles" / "0"
            profile_name = "bootstrap"
        else:
            child = namespace / "namespaces" / "inner"
            (child / "namespaces").mkdir(parents=True)
            (child / "profiles").mkdir()
            if case == "nested_child":
                (child / "namespaces" / "grandchild").mkdir()
            entry = child / "profiles" / "0"
            profile_name = "bootstrap"
        entry.mkdir()
        (entry / "name").write_text(profile_name + "\n")
        (entry / "mode").write_text("enforce\n")
        (entry / "raw_data").write_bytes(b"compiled")
        monkeypatch.setattr(linux_management, "_policy_directory", lambda *args: None)
        original_rmdir = Path.rmdir
        monkeypatch.setattr(
            Path,
            "rmdir",
            lambda path: (
                shutil.rmtree(path)
                if path == namespace or path == namespace / "namespaces" / "inner"
                else original_rmdir(path)
            ),
        )
    elif case == "bad_temporary":
        pending = root / ".loop-file-changed"
        pending.write_bytes(b"other")
        pending.chmod(0o600)
    elif case == "bad_receipt":
        receipt = root / "receipt.json"
        receipt.write_bytes(b"partial")
        receipt.chmod(0o555)
    elif case == "bad_unit":
        fragment = units / next(iter(config.unit_sources()))
        fragment.write_bytes(b"changed")
        fragment.chmod(0o444)
    elif case == "active_unit":
        original_properties = linux_management._retired_properties
        monkeypatch.setattr(
            linux_management,
            "_retired_properties",
            lambda *args: {**original_properties(*args), "ActiveState": "active"},
        )
    elif case in {"unknown_child", "bad_profile"}:
        namespace = policy / "namespaces" / generation
        (namespace / "namespaces").mkdir(parents=True)
        (namespace / "profiles").mkdir()
        monkeypatch.setattr(linux_management, "_policy_directory", lambda *args: None)
        if case == "unknown_child":
            (namespace / "namespaces" / "unknown").mkdir()
        else:
            entry = namespace / "profiles" / "0"
            entry.mkdir()
            (entry / "name").write_text("unknown\n")
            (entry / "mode").write_text("enforce\n")
            (entry / "raw_data").write_bytes(b"compiled")
    elif case in {"unsafe_temporary", "unknown_link"}:
        pending = root / ".loop-file-pending"
        if case == "unknown_link":
            outside = tmp_path / "outside"
            outside.write_bytes(b"hel")
            outside.chmod(0o444)
            os.link(outside, pending)
        else:
            pending.write_bytes(b"hel")
            pending.chmod(0o666)
    elif case == "bad_unit_temporary":
        pending = units / f".loop-unit-{generation}-pending"
        pending.write_bytes(b"wrong")
        pending.chmod(0o600)
    elif case in {"jobs", "instances"}:
        monkeypatch.setattr(
            linux_management,
            "_manager",
            lambda tool, arguments, deadline: (
                "pending"
                if arguments[0] == ("list-jobs" if case == "jobs" else "list-units")
                and (case != "instances" or arguments[-1].endswith("@*.service"))
                else ""
            ),
        )
    elif case == "listener":
        original_exists = Path.exists
        monkeypatch.setattr(
            Path,
            "exists",
            lambda path: (
                True
                if path
                == Path("/run") / (linux_management.NATIVE_UNIT_PREFIX + generation + ".socket")
                else original_exists(path)
            ),
        )
    elif case == "missing_helper_attachment":
        (root / "helper").unlink()
        monkeypatch.setattr(linux_management, "_attachment_entry", lambda *args: object())
    elif case == "bad_compiler_source":
        monkeypatch.setattr(
            linux_management,
            "compile_policy_source",
            lambda data, deadline, names=False: b"wrong\n" if names else b"compiled",
        )
    elif case == "bad_attachment_source":
        monkeypatch.setattr(
            linux_management,
            "compile_policy_source",
            lambda data, deadline, names=False: (
                (b"bootstrap\n" if data == b"profile" else b"wrong\n") if names else b"compiled"
            ),
        )
    elif case == "unknown_file":
        (root / "unrelated").write_bytes(b"unknown")
    elif case == "root_alias":
        monkeypatch.setattr(
            linux_management,
            "protected_path",
            lambda path, **kwargs: path.parent if path == root else path,
        )
    elif case == "mount":
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda path, *args, **kwargs: (
                f"1 0 0:1 / {root} rw - tmpfs tmpfs rw\n"
                if path == Path("/proc/self/mountinfo")
                else read_text(path, *args, **kwargs)
            ),
        )
    elif case == "record_present":
        (root / "management.json").write_bytes(b"record")
    elif case == "helper_withdrawn":
        (root / "helper").chmod(0o444)
    elif case == "directory_alias":
        monkeypatch.setattr(
            linux_management,
            "protected_path",
            lambda path, **kwargs: path.parent if path == root / "loop" else path,
        )
    elif case == "temporary_alias":
        pending = root / ".loop-file-pending"
        pending.write_bytes(b"hel")
        pending.chmod(0o600)
        calls = 0

        def alias_after_root(path, **kwargs):
            """Model a staging parent alias discovered after root validation."""
            nonlocal calls
            if path == root:
                calls += 1
                return root.parent if calls > 1 else root
            return path

        monkeypatch.setattr(linux_management, "protected_path", alias_after_root)
    elif case in {"temporary_changed_before_open", "temporary_changed_after_read"}:
        pending = root / ".loop-file-pending"
        pending.write_bytes(b"hel")
        pending.chmod(0o600)
        inode = pending.stat().st_ino
        original_fstat = os.fstat
        inspected = 0

        def drifting_fstat(descriptor):
            """Expose a changed temporary inode at one verified descriptor check."""
            nonlocal inspected
            details = original_fstat(descriptor)
            if details.st_ino == inode:
                inspected += 1
                target = 1 if case == "temporary_changed_before_open" else 2
                if inspected == target:
                    values = list(details)
                    values[6] += 1
                    return os.stat_result(values)
            return details

        monkeypatch.setattr(linux_management.os, "fstat", drifting_fstat)
    elif case == "unit_directory_alias":
        monkeypatch.setattr(
            linux_management,
            "protected_path",
            lambda path, **kwargs: path.parent if path == units else path,
        )
    elif case == "unknown_native_unit":
        (
            units / (linux_management.NATIVE_UNIT_PREFIX + "native_" + "b" * 32 + ".slice")
        ).write_bytes(b"foreign")
    elif case == "foreign_unit_temporary":
        (units / (".loop-unit-native_" + "b" * 32 + "-pending")).write_bytes(b"foreign")
    elif case == "foreign_manager":
        foreign = linux_management.NATIVE_UNIT_PREFIX + "native_" + "b" * 32 + ".slice"
        monkeypatch.setattr(
            linux_management,
            "_manager",
            lambda tool, arguments, deadline: (
                foreign + " loaded inactive dead\n"
                if arguments[0] == "list-units" and arguments[-1].endswith("native_*")
                else ""
            ),
        )
    elif case in {"foreign_tracking", "foreign_listener"}:
        original_glob = Path.glob
        selected = Path("/sys/fs/cgroup") if case == "foreign_tracking" else Path("/run")
        monkeypatch.setattr(
            Path,
            "glob",
            lambda path, pattern, **kwargs: (
                iter([path / "foreign"])
                if path == selected
                and pattern.startswith(linux_management.NATIVE_UNIT_PREFIX + "native_")
                else original_glob(path, pattern, **kwargs)
            ),
        )
    elif case == "foreign_namespace":
        (policy / "namespaces" / ("native_" + "b" * 32)).mkdir()
        monkeypatch.setattr(linux_management, "_policy_directory", Mock())
    elif case == "foreign_attachment":
        entry = policy / "profiles" / "0"
        entry.mkdir()
        (entry / "name").write_text(
            linux_management.NATIVE_UNIT_PREFIX + "native_" + "b" * 32 + "\n"
        )
        monkeypatch.setattr(linux_management, "_policy_directory", Mock())
    elif case == "current_manager_record":
        current = linux_management.NATIVE_UNIT_PREFIX + generation + ".slice"
        monkeypatch.setattr(
            linux_management,
            "_manager",
            lambda tool, arguments, deadline: (
                current + " loaded inactive dead\n"
                if arguments[0] == "list-units" and arguments[-1].endswith("native_*")
                else ""
            ),
        )
    elif case in {"unrelated_attachment", "attachment_limit"}:
        entry = policy / "profiles" / "0"
        entry.mkdir()
        (entry / "name").write_text("café\n" if case == "unrelated_attachment" else "unrelated\n")
        if case == "attachment_limit":
            original_iterdir = Path.iterdir
            selected = policy / "profiles"
            monkeypatch.setattr(
                Path,
                "iterdir",
                lambda path: iter([entry] * 4097) if path == selected else original_iterdir(path),
            )
    policy_sources = {"": {"bootstrap": b"profile"}}
    if case in {"native_child", "nested_child"}:
        policy_sources["inner"] = {"bootstrap": b"profile"}
    request = (
        config,
        bind(root / "management.lock"),
        bind(root / "compiler.lock"),
        runtime,
        {"loop_receiver.py", "helper", "loop_uninstall.py"},
        policy_sources,
        b"attachment",
        20,
    )
    refusals = {
        "changed": "differs from reviewed package",
        "bad_temporary": "differs from reviewed staging",
        "bad_receipt": "receipt differs",
        "bad_unit": "unit differs",
        "active_unit": "unit remains active",
        "unknown_child": "Unknown partial namespace child",
        "bad_profile": "profile differs",
        "unsafe_temporary": "Unsafe partial installation staging",
        "unknown_link": "unknown hard link",
        "bad_unit_temporary": "unit staging differs",
        "jobs": "pending manager jobs",
        "instances": "receiver instances",
        "listener": "listener is active",
        "missing_helper_attachment": "lacks its reviewed helper",
        "bad_compiler_source": "different AppArmor profile",
        "bad_attachment_source": "attachment source differs",
        "unknown_file": "Unknown partial installation file inventory",
        "root_alias": "directory aliases",
        "mount": "crosses a mount boundary",
        "nested_child": "Nested partial AppArmor namespace remains",
        "record_present": "Recorded installation requires",
        "directory_alias": "Aliased partial installation directory",
        "temporary_alias": "Partial staging directory aliases",
        "temporary_changed_before_open": "staging changed during inspection",
        "temporary_changed_after_read": "staging changed during inspection",
        "unit_directory_alias": "unit directory aliases",
        "unknown_native_unit": "Unknown partial installation unit inventory",
        "foreign_unit_temporary": "Unknown partial installation unit inventory",
        "foreign_manager": "Foreign native manager record",
        "foreign_tracking": "Foreign native tracking",
        "foreign_listener": "Foreign native listener",
        "foreign_namespace": "Foreign native AppArmor namespace",
        "foreign_attachment": "Foreign native AppArmor attachment",
        "attachment_limit": "AppArmor attachment inventory exceeds",
    }
    if case in refusals:
        with pytest.raises(ValueError, match=refusals[case]):
            linux_management.clean_unrecorded_installation(*request)
        assert reservation.exists()
        if case in {
            "bad_unit",
            "bad_unit_temporary",
            "unit_directory_alias",
            "unknown_native_unit",
            "foreign_unit_temporary",
        }:
            linux_management._manager.assert_not_called()
        return
    linux_management.clean_unrecorded_installation(*request)
    assert set(root.rglob("*")) == {
        root / "management.lock",
        root / "compiler.lock",
        reservations,
    }
    assert not list(units.iterdir())
    if case == "native":
        assert not namespace.exists() and not attachment.exists()
        remove_attachment.assert_called_once()
        assert remove_attachment.call_args.kwargs == {"tracking_absent": True}
    if case in {"native_profile", "native_child"}:
        assert not namespace.exists()


@pytest.mark.parametrize("change", ["inode", "bytes", "missing", "unknown"])
def test_changed_or_incomplete_authority_never_reconstructed(installation, monkeypatch, change):
    """Refuse original identity loss and unknown inventory without adopting current state."""
    record, arguments = installation
    publish(installation)
    helper = arguments[6].path
    if change == "inode":
        before = helper.read_bytes()
        helper.rename(helper.with_name("old-helper"))
        helper.write_bytes(before)
        helper.chmod(0o555)
    elif change == "bytes":
        helper.chmod(0o644)
        helper.write_bytes(b"changed")
        helper.chmod(0o555)
    elif change == "missing":
        helper.unlink()
    else:
        (record.parent / "unknown").write_bytes(b"unknown")
    retire = Mock(side_effect=ValueError("Unknown generation cleanup inventory."))
    monkeypatch.setattr(linux_installation, "retire_generation", retire)
    with pytest.raises((ValueError, OSError)):
        linux_management.uninstall_generation(record, 10)
    retire.assert_not_called()


@pytest.mark.parametrize(
    "change",
    ["existing-symlink", "missing-policy", "oversize", "changed-publication", "external-locks"],
)
def test_record_publication_refuses_unbound_state_and_supports_external_shared_locks(
    installation, monkeypatch, change
):
    """Use only original policy/catalog authority and refuse ambiguous publication results."""
    record, original = installation
    arguments = list(original)
    if change == "existing-symlink":
        record.symlink_to(record.parent / "absent")
        with pytest.raises(FileExistsError):
            linux_management.publish_generation_record(*arguments)
        return
    if change == "missing-policy":
        arguments[7] = b"not in catalog"
    elif change == "oversize":
        arguments[1] = {"x" * 65536: arguments[1]["receiver"], **arguments[1]}
    elif change == "changed-publication":
        actual = linux_management.stage_generation_files

        def stage(*args):
            """Return a deliberately stale record binding after real complete publication."""
            result = actual(*args)
            result["management.json"] = original[-1]["helper"]
            return result

        monkeypatch.setattr(linux_management, "stage_generation_files", stage)
    else:
        for index in (2, 3):
            old = arguments[index]
            new = record.parent.parent / old.path.name
            old.path.rename(new)
            arguments[index] = linux_host.bind_host_file(new)
        assert linux_management.publish_generation_record(*arguments).path == record
        return
    with pytest.raises(ValueError, match="policy|bound|changed"):
        linux_management.publish_generation_record(*arguments)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema=True),
        lambda value: value.update(schema=2),
        lambda value: value.update(extra=1),
        lambda value: value.update(config=[]),
        lambda value: value["config"].pop("uid"),
        lambda value: value["config"].update(generation=[]),
        lambda value: value["config"].update(uid=True),
        lambda value: value["config"].update(gid="invalid"),
        lambda value: value.update(files=[]),
        lambda value: value["files"].pop("receiver"),
        lambda value: value.update(artifacts=[]),
        lambda value: value.update(artifacts={}),
        lambda value: value.update(namespace=[]),
        lambda value: value.update(namespace=[0, 1]),
        lambda value: value.update(namespace="invalid"),
        lambda value: value.update(policies=[]),
        lambda value: value.update(policies={"": []}),
        lambda value: value.update(compiled="missing"),
        lambda value: value.update(source=[]),
        lambda value: value["lock"].update(identity=[1]),
        lambda value: value["lock"].update(identity=[-1] * 7),
        lambda value: value["lock"].update(identity="invalid"),
        lambda value: value["lock"].update(path=1),
        lambda value: value["lock"].update(sha256=1),
        lambda value: value["lock"].update(sha256="invalid"),
        lambda value: value["lock"].update(extra=True),
        lambda value: value.update(lock=[]),
        lambda value: value["artifacts"].update({"/outside": value["artifacts"].pop("helper")}),
        lambda value: value["artifacts"].update({"../helper": value["artifacts"].pop("helper")}),
        lambda value: value["artifacts"].update(
            {"management.json": value["artifacts"].pop("helper")}
        ),
        lambda value: value["artifacts"].update({"other": value["artifacts"].pop("helper")}),
    ],
)
def test_malformed_saved_authority_refuses(installation, mutation):
    """Malformed schemas, original identities and policy references never authorize management."""
    record, _ = installation
    publish(installation)
    rewrite(record, mutation)
    with pytest.raises(ValueError):
        linux_management.generation_status(record)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "list",
        "oversize",
        "writable",
        "wrong-name",
        "wrong-directory",
        "entry-missing",
        "entry-inert",
    ],
)
def test_record_location_size_ambiguity_and_export_authority(installation, change):
    """Reject ambiguous records and incomplete executable export while retaining authority."""
    record, arguments = installation
    publish(installation)
    if change == "duplicate":
        record.chmod(0o644)
        record.write_text('{"schema":1,"schema":1}')
        record.chmod(0o444)
    elif change == "list":
        record.chmod(0o644)
        record.write_text("[]")
        record.chmod(0o444)
    elif change == "oversize":
        record.chmod(0o644)
        record.write_bytes(b"x" * 65537)
        record.chmod(0o444)
    elif change == "writable":
        record.chmod(0o644)
    elif change == "wrong-name":
        target = record.with_name("other.json")
        record.rename(target)
        record = target
    elif change == "wrong-directory":
        target = record.parent.parent / record.name
        record.rename(target)
        record = target
    elif change == "entry-missing":
        rewrite(record, lambda value: value["artifacts"].pop("loop_uninstall.py"))
    else:
        entry = arguments[-1]["loop_uninstall.py"].path
        entry.chmod(0o444)
        bound = linux_host.bind_host_file(entry)
        rewrite(
            record,
            lambda value: value["artifacts"].update(
                {
                    "loop_uninstall.py": {
                        "path": str(bound.path),
                        "identity": list(bound.identity),
                        "sha256": bound.digest,
                    }
                }
            ),
        )
    with pytest.raises(ValueError):
        linux_management.uninstall_authority(record)


def test_fixed_status_safe_unavailable_without_elevation(monkeypatch, tmp_path):
    """Status marks settings unknown when native absence inspection is unavailable."""
    monkeypatch.setattr(linux_management.sys, "platform", "darwin")
    assert "this platform" in linux_management.installation_status(details=True)
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    root = tmp_path / "native"
    units = tmp_path / "units"
    units.mkdir()
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", root / "management.json")
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", units)

    def protected(path, **_kwargs):
        """Report only the owned unit directory as present."""
        if path == units:
            return path
        raise FileNotFoundError

    monkeypatch.setattr(linux_management, "protected_path", protected)
    verify = Mock(side_effect=OSError("Native inspection unavailable."))
    monkeypatch.setattr(linux_management, "verify_uninstalled", verify)
    status = Mock(return_value="authenticated")
    monkeypatch.setattr(linux_management, "generation_status", status)
    text = linux_management.installation_status(details=True)
    assert text.startswith("authenticated\nFiles:\n")
    assert f"{root / 'management.json'}: absent" in text
    assert "systemd socket: unknown" in text
    verify.assert_called_once()
    assert status.call_args.args[0] == root / "management.json"
    status.side_effect = ValueError("changed")
    assert "partial or changed" in linux_management.installation_status(details=True)


def test_installation_status_defaults_to_authenticated_summary(monkeypatch, tmp_path):
    """Default status stays concise and does not enumerate files or native settings."""
    record = tmp_path / "management.json"
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", record)
    authenticated = Mock(return_value="verbose internal authority report")
    monkeypatch.setattr(linux_management, "generation_status", authenticated)
    inspect_absence = Mock()
    monkeypatch.setattr(linux_management, "verify_uninstalled", inspect_absence)
    inventory = Mock(side_effect=AssertionError("default status must not build inventory"))
    monkeypatch.setattr(linux_management, "_managed_settings", inventory)

    assert linux_management.installation_status() == (
        "Installed. Runtime enforcement is checked before each command."
    )
    authenticated.assert_called_once_with(record)
    inspect_absence.assert_not_called()
    inventory.assert_not_called()


def test_installation_status_reports_verified_absence_concisely(monkeypatch, tmp_path):
    """A missing record is called not installed only after native absence is verified."""
    record = tmp_path / "management.json"
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", record)
    monkeypatch.setattr(
        linux_management,
        "generation_status",
        Mock(side_effect=FileNotFoundError("record missing")),
    )
    verify = Mock()
    monkeypatch.setattr(linux_management, "verify_uninstalled", verify)

    assert linux_management.installation_status() == (
        "Not installed. Sandboxed commands are unavailable. Run /sandbox install to set it up."
    )
    verify.assert_called_once()


@pytest.mark.parametrize(
    "generation_error, absence_error",
    [
        (ValueError("changed"), None),
        (FileNotFoundError("record missing"), OSError("residual state")),
    ],
)
def test_installation_status_routes_uncertain_state_to_diagnostics(
    monkeypatch, tmp_path, generation_error, absence_error
):
    """Changed authority or unverifiable absence requests details without dumping them."""
    record = tmp_path / "management.json"
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", record)
    monkeypatch.setattr(
        linux_management,
        "generation_status",
        Mock(side_effect=generation_error),
    )
    verify = Mock(side_effect=absence_error) if absence_error else Mock()
    monkeypatch.setattr(linux_management, "verify_uninstalled", verify)

    assert linux_management.installation_status() == (
        "Linux sandbox setup needs attention. Run /sandbox status details=true to inspect."
    )
    if absence_error:
        verify.assert_called_once()
    else:
        verify.assert_not_called()


@pytest.mark.parametrize("mode", ["enforce", "complain", "unknown-mode"])
def test_installation_status_lists_bound_files_and_native_settings(
    installation, monkeypatch, tmp_path, mode
):
    """Status reports the installed generation's files and observed systemd and AppArmor state."""
    record, _ = installation
    publish(installation)
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", record)
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", tmp_path)
    monkeypatch.setattr(linux_management, "protected_path", lambda path, **kwargs: path)
    native_bind = linux_management.bind_host_file
    tool = Mock()
    tool.open_verified.return_value = nullcontext()

    def bind(path):
        """Use owned record files and a fake read-only systemctl binding."""
        return tool if path == Path("/usr/bin/systemctl") else native_bind(path)

    monkeypatch.setattr(linux_management, "bind_host_file", bind)
    monkeypatch.setattr(
        linux_management,
        "_retired_properties",
        lambda _tool, name, _fields, _deadline: {
            "LoadState": "not-found" if name.endswith(".socket") else "loaded",
            "ActiveState": "inactive",
            "UnitFileState": "bad\x1bvalue",
        },
    )
    policy = tmp_path / "policy"
    namespace = policy / "namespaces" / "test"
    (namespace / "namespaces").mkdir(parents=True)
    profile = namespace / "profiles" / "0"
    profile.mkdir(parents=True)
    (profile / "name").write_text("bootstrap\n")
    (profile / "mode").write_text(mode + "\n")
    monkeypatch.setattr(linux_management, "_policy_root", lambda: (policy, 1))
    monkeypatch.setattr(linux_management, "_attachment_entry", lambda *_args: profile)
    monkeypatch.setattr(linux_management, "_policy_directory", Mock())

    text = linux_management.installation_status(details=True)

    assert text.startswith("Installed authority authenticated")
    assert f"{record}: present" in text
    assert f"{record.parent / 'helper'}: present" in text
    assert f"{tmp_path / 'loopgen_test.socket'}: present" in text
    assert "systemd slice: current" in text
    assert "systemd socket: absent" in text
    assert "systemd service: current" in text
    assert "AppArmor namespace: current" in text
    assert "AppArmor attachment: current" in text
    assert "AppArmor profiles: current" in text
    assert "AppArmor profile test/bootstrap: current" in text
    assert "observed mode=" + ("unknown" if mode == "unknown-mode" else mode) in text
    assert "active=inactive enablement=unknown" in text
    monkeypatch.setattr(linux_management, "_attachment_entry", lambda *_args: None)
    assert "AppArmor attachment: absent" in linux_management.installation_status(details=True)
    assert "Inventory inspection: unknown" not in text
    child = namespace / "namespaces" / "inner"
    child_profile = child / "profiles" / "0"
    child_profile.mkdir(parents=True)
    (child_profile / "name").write_text("payload_child\n")
    (child_profile / "mode").write_text("enforce\n")
    assert "AppArmor profile inner/payload_child: current" in linux_management.installation_status(
        details=True
    )
    (child_profile / "name").write_text("changed/name\n")
    changed = linux_management.installation_status(details=True)
    assert "AppArmor profiles: unknown" in changed
    assert changed.count("AppArmor namespace: current") == 1
    (child_profile / "name").write_text("payload_child\n")
    (child_profile / "mode").write_text("enforce\n")
    for index in range(33):
        (namespace / "namespaces" / f"extra_{index}").mkdir()
    assert "AppArmor profiles: unknown" in linux_management.installation_status(details=True)
    for index in range(33):
        (namespace / "namespaces" / f"extra_{index}").rmdir()
    for index in range(1, 65):
        extra = namespace / "profiles" / str(index)
        extra.mkdir()
        (extra / "name").write_text(f"profile_{index}\n")
        (extra / "mode").write_text("enforce\n")
    assert "AppArmor profiles: unknown" in linux_management.installation_status(details=True)


def test_installation_status_reports_verified_native_absence(tmp_path, monkeypatch):
    """Status reports absent settings after a no-record native inspection succeeds."""
    root = tmp_path / "native"
    units = tmp_path / "units"
    units.mkdir()
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", root / "management.json")
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", units)

    def protected(path, **kwargs):
        """Expose only the disposable unit directory."""
        if path != units:
            raise FileNotFoundError(path)
        return path

    monkeypatch.setattr(linux_management, "protected_path", protected)
    monkeypatch.setattr(
        linux_management,
        "generation_status",
        Mock(side_effect=FileNotFoundError("no record")),
    )
    verified = Mock()
    monkeypatch.setattr(linux_management, "verify_uninstalled", verified)
    text = linux_management.installation_status(details=True)
    assert text.startswith("Linux execution unavailable; installation absent.")
    assert "systemd slice: absent" in text
    assert "AppArmor profiles: absent" in text
    verified.assert_called_once()


@pytest.mark.parametrize("case", ["valid", "alias", "multiple", "invalid", "empty"])
def test_installation_status_uses_retained_partial_generation(tmp_path, monkeypatch, case):
    """Status names a sole reviewed generation and flags ambiguous partial capacity."""
    root = tmp_path / "native"
    root.mkdir()
    units = tmp_path / "units"
    units.mkdir()
    reservations = root / "reservations"
    reservations.mkdir()
    generation = "native_" + "a" * 32
    if case != "empty":
        (reservations / ("other" if case == "invalid" else generation)).write_bytes(b"partial")
    if case == "multiple":
        (reservations / ("native_" + "b" * 32)).write_bytes(b"partial")
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", root / "management.json")
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", units)
    monkeypatch.setattr(
        linux_management,
        "protected_path",
        lambda path, **kwargs: path.parent if case == "alias" and path == reservations else path,
    )
    monkeypatch.setattr(
        linux_management, "generation_status", Mock(side_effect=ValueError("partial"))
    )
    monkeypatch.setattr(
        linux_management,
        "verify_uninstalled",
        Mock(side_effect=ValueError("retained capacity is not absent")),
    )
    settings = Mock(return_value=["  AppArmor profiles: unknown"])
    monkeypatch.setattr(linux_management, "_managed_settings", settings)
    text = linux_management.installation_status(details=True)
    if case == "valid":
        settings.assert_called_once_with(generation)
        assert f"{units / ('loopgen_' + generation + '.slice')}: absent" in text
        assert f"{reservations / generation}: present" in text
    elif case == "empty":
        settings.assert_called_once_with(None)
        assert "AppArmor profiles: unknown" in text
    else:
        settings.assert_called_once_with(None)
        assert "Inventory inspection: unknown" in text


@pytest.mark.parametrize(
    "failure",
    ["root_alias", "unit_alias", "private_limit", "unit_limit", "path_io", "scan_io"],
)
def test_installation_status_reports_unverifiable_paths(monkeypatch, tmp_path, failure):
    """Read-only status marks unsafe, inaccessible, or oversized inventories as unknown."""
    root = tmp_path / "native"
    root.mkdir()
    units = tmp_path / "units"
    units.mkdir()
    if failure == "private_limit":
        for index in range(513):
            (root / f"entry-{index}").touch()
    if failure == "unit_limit":
        for index in range(65):
            (units / f"loopgen_test{index}.socket").touch()
    if failure == "unit_alias":
        (units / "loopgen_test.socket").touch()
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", root / "management.json")
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", units)
    monkeypatch.setattr(
        linux_management,
        "generation_status",
        Mock(side_effect=ValueError("unavailable")),
    )
    monkeypatch.setattr(
        linux_management,
        "verify_uninstalled",
        Mock(side_effect=ValueError("native settings unavailable")),
    )

    def protected(path, **_kwargs):
        """Return a changed root or unit authority for the selected case."""
        return (
            path.with_name("alias")
            if (
                failure == "root_alias"
                and path == root
                or failure == "unit_alias"
                and path == units
            )
            else path
        )

    monkeypatch.setattr(linux_management, "protected_path", protected)
    if failure == "scan_io":
        original_rglob = Path.rglob

        def rglob(path, pattern):
            """Surface a private directory scan failure after opening the root."""
            if path == root:
                raise PermissionError("directory changed")
            return original_rglob(path, pattern)

        monkeypatch.setattr(Path, "rglob", rglob)
    if failure == "path_io":
        original_lstat = Path.lstat

        def lstat(path, *args, **kwargs):
            """Make one expected file inaccessible without changing its peers."""
            if path == root / "cache.lock":
                raise PermissionError("inaccessible")
            return original_lstat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", lstat)

    text = linux_management.installation_status(details=True)

    if failure == "path_io":
        assert f"{root / 'cache.lock'}: unknown" in text
        assert "Inventory inspection: unknown" not in text
    else:
        assert "Inventory inspection: unknown" in text
    if failure == "root_alias":
        assert f"{root / 'management.json'}: unknown" in text
    if failure == "unit_alias":
        assert f"{units}: unknown" in text


@pytest.mark.parametrize(
    "change",
    [
        "no_config",
        "invalid_generation",
        "numeric_generation",
        "empty_artifacts",
        "absolute_artifact",
        "parent_artifact",
        "invalid_artifacts",
    ],
)
def test_installation_status_handles_partial_record_inventory(
    installation, monkeypatch, tmp_path, change
):
    """Status reports current paths without trusting malformed record metadata as authority."""
    record, _ = installation
    publish(installation)
    mutations = {
        "no_config": lambda value: value.pop("config"),
        "invalid_generation": lambda value: value["config"].update(generation="bad!"),
        "numeric_generation": lambda value: value["config"].update(generation=12),
        "empty_artifacts": lambda value: value.update(artifacts={}),
        "absolute_artifact": lambda value: value.update(artifacts={"/outside": {}}),
        "parent_artifact": lambda value: value.update(artifacts={"../outside": {}}),
        "invalid_artifacts": lambda value: value.update(artifacts=12),
    }
    rewrite(record, mutations[change])
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", record)
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", tmp_path)
    monkeypatch.setattr(linux_management, "protected_path", lambda path, **kwargs: path)
    native_bind = linux_management.bind_host_file

    def bind(path):
        """Inspect the owned record while keeping host systemctl out of unit tests."""
        if path == Path("/usr/bin/systemctl"):
            raise OSError("system manager unavailable")
        return native_bind(path)

    monkeypatch.setattr(linux_management, "bind_host_file", bind)
    monkeypatch.setattr(linux_management, "_policy_root", Mock(side_effect=OSError("unavailable")))

    text = linux_management.installation_status(details=True)

    assert "installation absent, partial or changed" in text
    assert f"{record}: present" in text
    if change in {"absolute_artifact", "parent_artifact", "invalid_artifacts"}:
        assert "Inventory inspection: unknown" in text
    else:
        assert "Inventory inspection: unknown" not in text
    if change in {"no_config", "invalid_generation", "numeric_generation"}:
        assert "systemd socket: unknown" in text


def test_installation_status_handles_unavailable_native_settings(
    installation, monkeypatch, tmp_path
):
    """Settings remain unknown when systemd or AppArmor inspection cannot verify them."""
    record, _ = installation
    publish(installation)
    monkeypatch.setattr(linux_management.sys, "platform", "linux")
    monkeypatch.setattr(linux_management, "_NATIVE_RECORD", record)
    monkeypatch.setattr(linux_management, "_UNIT_DIRECTORY", tmp_path)
    monkeypatch.setattr(linux_management, "protected_path", lambda path, **kwargs: path)
    native_bind = linux_management.bind_host_file

    def bind(path):
        """Refuse a host tool without touching the real installation record fixture."""
        if path == Path("/usr/bin/systemctl"):
            raise OSError("system manager unavailable")
        return native_bind(path)

    monkeypatch.setattr(linux_management, "bind_host_file", bind)
    monkeypatch.setattr(linux_management, "_policy_root", Mock(side_effect=ValueError("unsafe")))
    text = linux_management.installation_status(details=True)
    assert "systemd slice: unknown" in text
    assert "AppArmor namespace: unknown" in text
    assert "AppArmor attachment: unknown" in text


def test_management_private_inventory_bound_refuses_before_retirement(installation, monkeypatch):
    """Enforce the private management inventory ceiling on known entries."""
    record, _ = installation
    publish(installation)
    monkeypatch.setattr(Path, "rglob", lambda *args: iter([record] * 257))
    with pytest.raises(ValueError, match="inventory"):
        linux_management.generation_status(record)


@pytest.fixture
def composition(installation, monkeypatch):
    """Bind realistic installer inputs and isolate each native lifecycle owner at its public API."""
    record, arguments = installation
    config, files, lock, compiler, namespace, policies, helper, compiled, source, original = (
        arguments
    )
    contents = {
        name: bound.path.read_bytes()
        for name, bound in original.items()
        if name not in {"attachment.bin", "attachment.apparmor", "bootstrap.bin"}
    }
    contents.update({"cache.lock": b"cache", "loop/execution/sandbox/linux/bootstrap.py": b"code"})
    executables = {"helper", "receiver", "loop_uninstall.py"}
    calls = []

    def collaborator(name, result):
        """Record observable owner ordering without executing native management."""

        def invoke(*args):
            """Return a controlled public collaborator result."""
            calls.append((name, args))
            return result(*args) if callable(result) else result

        stub = Mock(side_effect=invoke)
        monkeypatch.setattr(
            linux_installation if name == "retire_generation" else linux_management, name, stub
        )
        return stub

    def stage(directory, catalog, executable, _lock):
        """Publish real disposable immutable files as the staging boundary result."""
        result = {}
        for name, data in catalog.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
                path.chmod(0o555 if name in executable else 0o444)
            result[name] = linux_host.bind_host_file(path)
        return result

    monkeypatch.setattr(linux_management, "drained_generation", Mock(return_value=nullcontext()))
    stage_mock = collaborator("stage_generation_files", stage)
    collaborator("load_generation_policy", (namespace, policies, compiled))
    collaborator("provision_generation_units", {**files, "slice_unit": config.slice_unit})
    record.write_bytes(b"record")
    record.chmod(0o444)
    authority = linux_host.bind_host_file(record)
    collaborator("provision_receipt", {"receipt": authority})
    publish_mock = collaborator("publish_generation_record", authority)
    activate_mock = collaborator("activate_generation", (config, files))
    monkeypatch.setattr(linux_management, "discover_tool", Mock(return_value=helper))
    monkeypatch.setattr(linux_management, "apparmor_library", Mock(return_value=helper))
    inputs = (
        config,
        lock,
        compiler,
        contents,
        executables,
        {"": {"bootstrap": b"source"}, "inner": {"bootstrap": b"inner"}},
        source,
        50.0,
    )
    # Loaded exports contain both root and child policies to check distinct cache paths.
    monkeypatch.setattr(
        linux_management,
        "load_generation_policy",
        Mock(
            side_effect=lambda *args: (
                calls.append(("load_generation_policy", args))
                or (
                    namespace,
                    {"": {"bootstrap": b"bootstrap"}, "inner": {"bootstrap": b"inner"}},
                    compiled,
                )
            )
        ),
    )
    return inputs, calls, authority, stage_mock, publish_mock, activate_mock


def test_composed_installation_persists_authority_before_activation(composition):
    """Fresh installation stages exact exports and both durable records before opening admission."""
    inputs, calls, authority, stage, publication, activation = composition
    contents = dict(inputs[3])
    executables = set(inputs[4])
    assert linux_management.install_generation(*inputs) == authority
    assert [name for name, _args in calls] == [
        "stage_generation_files",
        "load_generation_policy",
        "stage_generation_files",
        "provision_generation_units",
        "provision_receipt",
        "publish_generation_record",
        "activate_generation",
    ]
    assert stage.call_args_list[1].args[1]["policies/root/bootstrap.bin"] == b"bootstrap"
    assert stage.call_args_list[1].args[1]["policies/child/inner/bootstrap.bin"] == b"inner"
    assert publication.call_args.args[-1]["receipt.json"] == authority
    assert activation.call_args.args[-1] == inputs[-1]
    assert inputs[3] == contents
    assert inputs[4] == executables


@pytest.mark.parametrize("change", ["missing", "executable", "reserved"])
def test_composed_installation_incomplete_catalog_refuses_before_native_work(composition, change):
    """Missing roles, executable entries and generated-artifact collisions refuse before staging."""
    inputs, calls, *_ = composition
    if change == "missing":
        del inputs[3]["helper"]
    elif change == "executable":
        inputs[4].remove("loop_uninstall.py")
    else:
        inputs[3]["receipt.json"] = b"conflict"
    with pytest.raises(ValueError, match="Incomplete or conflicting installation catalog"):
        linux_management.install_generation(*inputs)
    assert calls == (["retire"] if change == "stale" else [])


@pytest.mark.parametrize(
    "boundary",
    [
        "load_generation_policy",
        "stage_generation_files",
        "provision_generation_units",
        "provision_receipt",
        "publish_generation_record",
        "activate_generation",
    ],
)
def test_composed_installation_failure_never_runs_dependent_owner(
    composition, monkeypatch, boundary
):
    """Stop on lifecycle failure without cleanup or early admission."""
    inputs, calls, *_ = composition
    failure = Mock(side_effect=OSError("retained authority"))
    monkeypatch.setattr(
        linux_installation if boundary == "retire_generation" else linux_management,
        boundary,
        failure,
    )
    with pytest.raises(OSError, match="retained authority"):
        linux_management.install_generation(*inputs)
    assert failure.call_count == 1
    assert boundary not in [name for name, _args in calls]
    assert "activate_generation" not in [name for name, _args in calls]


@pytest.mark.parametrize("drift", ["config", "files"])
def test_composed_installation_activation_identity_must_match_record(composition, drift):
    """Activation cannot report changed slice or file bindings as installation success."""
    inputs, _calls, _authority, _stage, _publication, activation = composition
    activation.side_effect = None
    activation.return_value = (None, {}) if drift == "config" else (inputs[0], {})
    with pytest.raises(ValueError, match="Activated installation differs"):
        linux_management.install_generation(*inputs)


def test_composed_installation_shared_authority_is_not_cleanup_artifact(composition):
    """Shared locks and reserved capacity remain outside private teardown artifact authority."""
    inputs, _calls, _authority, _stage, publication, _activation = composition
    for bound in (inputs[0].reservation, inputs[1], inputs[2]):
        inputs[3][bound.path.name] = bound.path.read_bytes()
    linux_management.install_generation(*inputs)
    assert (
        not {inputs[0].reservation.path.name, inputs[1].path.name, inputs[2].path.name}
        & publication.call_args.args[-1].keys()
    )


@pytest.mark.parametrize("closure_fails", [False, True])
def test_composed_installation_late_record_loss_closes_admission(
    composition, monkeypatch, closure_fails
):
    """Late record integrity failure closes admission and preserves both errors if closure fails."""
    inputs, _calls, authority, _stage, _publication, activation = composition
    original = activation.side_effect

    def lose_record(*args):
        """Simulate a reachable file loss after the activation boundary returns."""
        result = original(*args)
        authority.path.unlink()
        return result

    activation.side_effect = lose_record
    drain = (
        Mock(side_effect=OSError("closure failed"))
        if closure_fails
        else Mock(return_value=nullcontext())
    )
    monkeypatch.setattr(linux_management, "drained_generation", drain)
    if closure_fails:
        with pytest.raises(
            ExceptionGroup, match="validation and admission closure failed"
        ) as caught:
            linux_management.install_generation(*inputs)
        assert isinstance(caught.value.exceptions[0], FileNotFoundError)
        assert str(caught.value.exceptions[1]) == "closure failed"
    else:
        with pytest.raises(FileNotFoundError):
            linux_management.install_generation(*inputs)
    assert drain.call_count == 1
    assert drain.call_args.args[0] == inputs[0]
    assert drain.call_args.args[2] == inputs[1]
