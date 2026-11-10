"""Verify original management authority persistence, safe status and exact checked uninstall."""

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from loop.execution.sandbox.linux import host as linux_host
from loop.execution.sandbox.linux import installation as linux_installation
from loop.execution.sandbox.linux import management as linux_management
from loop.execution.sandbox.linux import records as linux_records
from loop.execution.sandbox.linux.admission import GenerationAdmission

from .helpers import mountinfo_path, publish, rewrite


@pytest.fixture(autouse=True)
def native_inventory(monkeypatch, tmp_path):
    """Keep fixed host tracking and listener inventory independent of real installations."""
    monkeypatch.setattr(linux_management, "_BOOT_UNIT_DIRECTORY", tmp_path / "absent-boot-units")
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
    monkeypatch.setattr(linux_management, "_remove_boot_unit", Mock())
    monkeypatch.setattr(linux_management.time, "monotonic", lambda: 10)
    linux_management.uninstall_generation(record, 123)
    assert retire.call_args.args[0] == authority
    assert retire.call_args.args[1] == 123
    with pytest.raises(FileExistsError, match="already exists"):
        publish(installation)
    assert linux_host.bind_host_file(record) == authority


@pytest.fixture
def boot_recovery(installation, monkeypatch, tmp_path):
    """Isolate reboot orchestration while retaining real record and marker verification."""
    record, arguments = installation
    authority = publish(installation)
    config, files = arguments[:2]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    boot_id = tmp_path / "boot-id"
    boot_id.write_text("current-boot\n")
    uid_map = tmp_path / "uid-map"
    uid_map.write_text("0 0 4294967295\n")
    tool = tmp_path / "systemctl"
    tool.write_bytes(b"tool")
    tool.chmod(0o555)
    monkeypatch.setattr(linux_records.os, "getuid", lambda: 0)
    monkeypatch.setattr(linux_records, "PROC_UID_MAP_PATH", uid_map)
    monkeypatch.setattr(linux_records, "PROC_BOOT_ID_PATH", boot_id)
    monkeypatch.setattr(linux_records, "SYSTEM_RUNTIME_DIRECTORY", runtime)
    monkeypatch.setattr(linux_management, "PROC_BOOT_ID_PATH", boot_id)
    monkeypatch.setattr(linux_management, "SYSTEM_RUNTIME_DIRECTORY", runtime)
    monkeypatch.setattr(linux_management, "SYSTEMCTL_PATH", tool)
    monkeypatch.setattr(linux_management, "_administration", lambda _: nullcontext())
    monkeypatch.setattr(linux_management, "protected_path", lambda path, **kwargs: path)
    monkeypatch.setattr(linux_management.time, "monotonic", lambda: 10)
    units = {**files, "slice_unit": config.slice_unit}
    monkeypatch.setattr(linux_management, "_provision_generation_units", Mock(return_value=units))
    monkeypatch.setattr(linux_management, "installation_files", Mock(return_value=files))
    policy = Mock(return_value=(101, 202))
    monkeypatch.setattr(linux_management, "_recover_generation_policy", policy)
    manager = Mock(return_value="")
    monkeypatch.setattr(linux_management, "_manager", manager)
    monkeypatch.setattr(
        linux_management,
        "attest_idle_generation_instances",
        Mock(return_value=["retained.service"]),
    )
    monkeypatch.setattr(
        linux_management,
        "_unit_state",
        Mock(
            return_value={
                "ActiveState": "active",
                "SubState": "active",
                "ControlGroup": "/" + config.slice_unit.path.name,
            }
        ),
    )

    def publish_marker(directory, name, data):
        """Model one atomic publication in the isolated runtime directory."""
        path = directory / name
        path.write_bytes(data)
        path.chmod(0o444)

    monkeypatch.setattr(linux_management, "publish_volatile_file", publish_marker)
    return record, arguments, authority, runtime, units, policy, manager


def test_boot_recovery_recreates_bindings_and_is_idempotent(boot_recovery):
    """Recreated authority is published once and repeated recovery preserves the same marker."""
    record, _, authority, runtime, units, policy, manager = boot_recovery
    linux_management.recover_boot_generation(record, 50)
    marker = runtime / "loopgen_test.boot"
    original = linux_host.bind_host_file(marker)
    recovered, rebound = linux_records.read_generation(record)
    assert rebound == authority and recovered.namespace == (101, 202)
    assert recovered.config.slice_unit == units["slice_unit"]
    linux_management.recover_boot_generation(record, 50)
    assert linux_host.bind_host_file(marker) == original
    assert policy.call_count == 2
    assert sum(call.args[1][0] == "start" for call in manager.call_args_list) == 2


def test_boot_recovery_starts_idle_tracking_before_admission(boot_recovery, monkeypatch):
    """An empty absent cgroup is restored before the receiver socket opens."""
    record, _, _, _, units, _, manager = boot_recovery
    slice_name = units["slice_unit"].path.name
    inactive = {"ActiveState": "inactive", "SubState": "dead", "ControlGroup": ""}
    active = {"ActiveState": "active", "SubState": "active", "ControlGroup": "/" + slice_name}
    linux_management._unit_state.side_effect = [inactive, inactive, inactive, active]
    idle = Mock()
    monkeypatch.setattr(linux_management, "_recursive_idle", idle)
    linux_management.recover_boot_generation(record, 50)
    assert idle.call_args_list == [
        call(linux_management.CGROUP_ROOT / slice_name, absent=True),
        call(linux_management.CGROUP_ROOT / slice_name),
    ]
    starts = [call.args[1][-1] for call in manager.call_args_list if call.args[1][0] == "start"]
    assert starts == [slice_name, units["socket_unit"].path.name]
    assert linux_management.attest_idle_generation_instances.call_count == 2


@pytest.mark.parametrize(
    "change",
    ["jobs", "instances", "cgroup", "changed", "listener", "jobs_after", "instances_after"],
)
def test_boot_recovery_rejects_ambiguous_idle_tracking(boot_recovery, monkeypatch, change):
    """Neither an ambiguous manager nor a changed cgroup can reopen admission."""
    record, _, _, _, _, _, manager = boot_recovery
    inactive = {"ActiveState": "inactive", "SubState": "dead", "ControlGroup": ""}
    linux_management._unit_state.return_value = inactive
    if change in {"jobs", "instances", "jobs_after", "instances_after"}:
        seen = []

        def response(tool, args, deadline):
            """Report an unexpected job or receiver before or after tracking starts."""
            if args[0] != "list-jobs":
                return ""
            seen.append(args[0])
            occurrence = 2 if change.endswith("_after") else 1
            return "unexpected\n" if change.startswith("jobs") and len(seen) == occurrence else ""

        manager.side_effect = response
        if change.startswith("instances"):
            failure = ValueError("Generation receiver instances changed")
            linux_management.attest_idle_generation_instances.side_effect = (
                [[], failure] if change.endswith("_after") else failure
            )
        monkeypatch.setattr(linux_management, "_recursive_idle", Mock())
        if change.endswith("_after"):
            active = {"ActiveState": "active", "SubState": "active", "ControlGroup": "/test"}
            linux_management._unit_state.side_effect = [inactive, inactive, inactive, active]
    elif change == "listener":
        active = {"ActiveState": "active", "SubState": "listening", "ControlGroup": ""}
        linux_management._unit_state.side_effect = [inactive, inactive, active]
    else:
        idle = Mock(side_effect=ValueError("tracking changed")) if change == "cgroup" else Mock()
        monkeypatch.setattr(linux_management, "_recursive_idle", idle)
    with pytest.raises(ValueError, match="Pending|instances|tracking|listener"):
        linux_management.recover_boot_generation(record, 50)
    assert not any(
        call.args[1][-1].endswith(".socket")
        for call in manager.call_args_list
        if call.args[1][0] == "start"
    )


@pytest.mark.parametrize("change", ["missing", "receipt", "policy", "binding", "runtime"])
def test_boot_recovery_refuses_drift_before_starting_admission(boot_recovery, monkeypatch, change):
    """Missing or altered durable and boot authority cannot start the sandbox socket."""
    record, arguments, _, _, units, policy, manager = boot_recovery
    if change == "missing":
        arguments[-1]["loop_uninstall.py"].path.unlink()
    elif change == "receipt":
        linux_management.installation_files.return_value = {**units, "receiver": arguments[6]}
    elif change == "policy":
        policy.side_effect = ValueError("changed policy")
    elif change == "binding":
        linux_management.recover_boot_generation(record, 50)
        manager.reset_mock()
        policy.return_value = (303, 404)
    else:
        monkeypatch.setattr(
            linux_management, "protected_path", lambda path, **kwargs: path / "alias"
        )
    with pytest.raises(ValueError, match="incomplete|receipt|policy|bindings|aliases"):
        linux_management.recover_boot_generation(record, 50)
    assert not any(call.args[1][0] == "start" for call in manager.call_args_list)


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
        "implicit_slice",
        "implicit_other_unit",
        "wrong_unit_id",
        "implicit_slice_active",
        "implicit_slice_substate",
        "implicit_slice_cgroup",
        "implicit_slice_fragment",
        "implicit_slice_dropins",
        "implicit_slice_reload",
        "implicit_slice_transient",
        "implicit_slice_extra",
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
def test_unrecorded_stage_is_cleaned_from_reviewed_bytes(
    tmp_path, monkeypatch, logical_admission_sources, case
):
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
        "loop_boot.py": b"boot",
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
            "Id": "foreign.slice" if case == "wrong_unit_id" else name,
            "LoadState": (
                "loaded"
                if (units / name.replace("@status.service", "@.service")).exists()
                or case.startswith("implicit_slice")
                and name.endswith(".slice")
                or case == "implicit_other_unit"
                and name.endswith(".socket")
                else "not-found"
            ),
            "ActiveState": "inactive",
            "SubState": "dead",
            "ControlGroup": "",
            "FragmentPath": (
                str(units / name.replace("@status.service", "@.service"))
                if (units / name.replace("@status.service", "@.service")).exists()
                else ""
            ),
            "DropInPaths": "",
            "NeedDaemonReload": "no",
            "Transient": "no",
        },
    )
    if case.startswith("implicit_slice_"):
        original_properties = linux_management._retired_properties
        changes = {
            "implicit_slice_active": {"ActiveState": "active"},
            "implicit_slice_substate": {"SubState": "running"},
            "implicit_slice_cgroup": {"ControlGroup": "/unexpected"},
            "implicit_slice_fragment": {"FragmentPath": "/unexpected"},
            "implicit_slice_dropins": {"DropInPaths": "/unexpected"},
            "implicit_slice_reload": {"NeedDaemonReload": "yes"},
            "implicit_slice_transient": {"Transient": "yes"},
            "implicit_slice_extra": {"Unexpected": "value"},
        }
        monkeypatch.setattr(
            linux_management,
            "_retired_properties",
            lambda tool, name, fields, deadline: {
                **original_properties(tool, name, fields, deadline),
                **(changes[case] if name.endswith(".slice") else {}),
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
    read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            b"1 0 0:1 / / rw - tmpfs tmpfs rw\n"
            if path == Path("/proc/self/mountinfo")
            else read_bytes(path)
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
            "read_bytes",
            lambda path: (
                f"1 0 0:1 / {mountinfo_path(root)} rw - tmpfs tmpfs rw\n".encode()
                if path == Path("/proc/self/mountinfo")
                else read_bytes(path)
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
        "implicit_other_unit": "unit remains active",
        "wrong_unit_id": "unit remains active",
        **{
            name: "unit remains active"
            for name in (
                "implicit_slice_active",
                "implicit_slice_substate",
                "implicit_slice_cgroup",
                "implicit_slice_fragment",
                "implicit_slice_dropins",
                "implicit_slice_reload",
                "implicit_slice_transient",
                "implicit_slice_extra",
            )
        },
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
def composition(installation, monkeypatch, tmp_path):
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
    contents.update(
        {
            "cache.lock": b"cache",
            "loop_boot.py": b"boot",
            "loop/execution/sandbox/linux/bootstrap.py": b"code",
        }
    )
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
    boot_units = tmp_path / "boot-units"
    boot_units.mkdir()
    (boot_units / "multi-user.target.wants").mkdir()
    tool = tmp_path / "systemctl"
    tool.write_bytes(b"tool")
    tool.chmod(0o555)
    monkeypatch.setattr(linux_management, "_BOOT_UNIT_DIRECTORY", boot_units)
    monkeypatch.setattr(linux_management, "SYSTEMCTL_PATH", tool)
    monkeypatch.setattr(linux_management, "_administration", lambda _: nullcontext())
    monkeypatch.setattr(linux_management, "protected_path", lambda path, **_: path)
    monkeypatch.setattr(linux_management, "_manager", Mock())
    monkeypatch.setattr(linux_management.time, "monotonic", lambda: 10)
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


@pytest.mark.parametrize("existing", [False, True])
def test_installation_enables_exact_boot_service(composition, existing):
    """Install creates or exactly reuses the fixed persistent boot activation."""
    inputs, _, authority, *_ = composition
    if existing:
        assert linux_management.install_generation(*inputs) == authority
    else:
        (linux_management._BOOT_UNIT_DIRECTORY / "multi-user.target.wants").rmdir()
    assert linux_management.install_generation(*inputs) == authority
    unit = linux_management._BOOT_UNIT_DIRECTORY / "loop-native-boot.service"
    link = unit.parent / "multi-user.target.wants" / unit.name
    assert unit.is_file() and unit.stat().st_mode & 0o777 == 0o444
    assert link.is_symlink() and os.readlink(link) == "../" + unit.name
    assert str(inputs[0].receiver.parent / "loop_boot.py") in unit.read_text()


@pytest.mark.parametrize(
    "output,expected",
    [
        ("ActiveState=activating\nJob=\n", "activating"),
        ("ActiveState=inactive\nJob=17\n", "activating"),
        ("ActiveState=active\nJob=\n", "active"),
        ("ActiveState=failed\nJob=\n", "failed"),
    ],
)
def test_boot_state_authenticates_unit_and_manager(composition, output, expected):
    """Only the fixed protected unit and an unambiguous systemd state drive boot waiting."""
    inputs, *_ = composition
    linux_management.install_generation(*inputs)
    linux_management._manager.return_value = output
    assert linux_management.boot_activation_state(inputs[0].receiver.parent, 50) == expected


@pytest.mark.parametrize("change", ["unit", "duplicate", "malformed", "missing", "unknown"])
def test_boot_state_refuses_untrusted_or_ambiguous_input(composition, change):
    """A changed boot unit or ambiguous manager response cannot authorize a wait."""
    inputs, *_ = composition
    linux_management.install_generation(*inputs)
    unit = linux_management._BOOT_UNIT_DIRECTORY / "loop-native-boot.service"
    if change == "unit":
        unit.chmod(0o644)
    outputs = {
        "duplicate": "ActiveState=active\nActiveState=active\nJob=\n",
        "malformed": "bad-line\n",
        "missing": "ActiveState=active\n",
        "unknown": "ActiveState=other\nJob=\n",
    }
    linux_management._manager.return_value = outputs.get(change, "ActiveState=active\nJob=\n")
    with pytest.raises(ValueError, match="Boot|boot"):
        linux_management.boot_activation_state(inputs[0].receiver.parent, 50)


@pytest.mark.parametrize("change", ["unit", "link", "entry", "directory", "wants"])
def test_installation_refuses_unsafe_boot_activation(composition, monkeypatch, change):
    """Conflicting boot files or ancestry stop installation without opening new authority."""
    inputs, _, _, *_ = composition
    root = linux_management._BOOT_UNIT_DIRECTORY
    unit = root / "loop-native-boot.service"
    link = root / "multi-user.target.wants" / unit.name
    if change == "unit":
        unit.write_text("foreign")
        unit.chmod(0o444)
    elif change == "link":
        link.symlink_to("../foreign.service")
    elif change == "entry":
        real_bind = linux_management.bind_host_file
        monkeypatch.setattr(
            linux_management,
            "bind_host_file",
            lambda path, **kwargs: (
                real_bind(inputs[0].receiver, **kwargs)
                if path.name == "loop_boot.py"
                else real_bind(path, **kwargs)
            ),
        )
    else:
        selected = root if change == "directory" else root / "multi-user.target.wants"
        monkeypatch.setattr(
            linux_management,
            "protected_path",
            lambda path, **kwargs: path / "other" if path == selected else path,
        )
    with pytest.raises(ValueError, match="boot|Boot"):
        linux_management.install_generation(*inputs)


def test_boot_state_rejects_unsafe_boot_entry_path(composition):
    """A boot entry path that cannot be represented safely in systemd is rejected."""
    inputs, *_ = composition
    linux_management.install_generation(*inputs)
    with pytest.raises(ValueError, match="Invalid installed boot entry directory"):
        linux_management.boot_activation_state(Path("/unsafe;path"), 50)


@pytest.mark.parametrize("directory", [Path("/safe path"), Path("/safe-path")])
def test_boot_state_accepts_safe_entry_path(composition, directory):
    """A space in a boot entry is represented as one quoted systemd argument."""
    inputs, *_ = composition
    linux_management.install_generation(*inputs)
    unit = linux_management._BOOT_UNIT_DIRECTORY / "loop-native-boot.service"
    original = str(inputs[0].receiver.parent / "loop_boot.py")
    original_argument = f'"{original}"' if " " in original else original
    alternate = str(directory / "loop_boot.py")
    alternate_argument = f'"{alternate}"' if " " in alternate else alternate
    source = unit.read_text().replace(original_argument, alternate_argument)
    unit.chmod(0o644)
    unit.write_text(source)
    unit.chmod(0o444)
    linux_management._manager.return_value = "ActiveState=active\nJob=\n"
    assert linux_management.boot_activation_state(directory, 50) == "active"


@pytest.mark.parametrize(
    "change",
    [
        None,
        "marker",
        "missing_unit",
        "missing_link",
        "missing_both",
        "bad_unit",
        "bad_link",
        "bad_wants",
        "bad_directory",
    ],
)
def test_uninstall_retires_boot_activation(composition, monkeypatch, tmp_path, change):
    """Uninstall removes only authenticated boot state and preserves conflicting host files."""
    from types import SimpleNamespace

    inputs, _, authority, *_ = composition
    linux_management.install_generation(*inputs)
    config, lock = inputs[0], inputs[1]
    root = linux_management._BOOT_UNIT_DIRECTORY
    unit = root / "loop-native-boot.service"
    link = root / "multi-user.target.wants" / unit.name
    runtime = tmp_path / "run"
    runtime.mkdir()
    monkeypatch.setattr(linux_management, "SYSTEM_RUNTIME_DIRECTORY", runtime)
    marker = runtime / f"loopgen_{config.generation}.boot"
    if change == "marker":
        marker.write_text("recorded boot")
        marker.chmod(0o444)
    elif change == "missing_unit":
        unit.unlink()
    elif change == "missing_link":
        link.unlink()
    elif change == "missing_both":
        unit.unlink()
        link.unlink()
    elif change == "bad_unit":
        unit.chmod(0o644)
        unit.write_text("foreign")
        unit.chmod(0o444)
    elif change == "bad_link":
        link.unlink()
        link.symlink_to("../foreign.service")
    elif change in {"bad_wants", "bad_directory"}:
        selected = link.parent if change == "bad_wants" else root
        monkeypatch.setattr(
            linux_management,
            "protected_path",
            lambda path, **kwargs: path / "other" if path == selected else path,
        )
    installed = SimpleNamespace(config=config, lock=lock, retire=Mock())
    monkeypatch.setattr(
        linux_management, "read_generation", Mock(return_value=(installed, authority))
    )
    if change in {"bad_unit", "bad_link", "bad_wants", "bad_directory"}:
        with pytest.raises(ValueError, match="Boot"):
            linux_management.uninstall_generation(authority.path, 50)
    else:
        linux_management.uninstall_generation(authority.path, 50)
        assert not unit.exists() and not link.is_symlink() and not marker.exists()
    installed.retire.assert_called_once_with(authority, 50)


def test_uninstalled_check_rejects_remaining_boot_activation(composition, tmp_path):
    """An enabled persistent boot service prevents reporting a completed uninstall."""
    inputs, *_ = composition
    linux_management.install_generation(*inputs)
    with pytest.raises(ValueError, match="boot activation remains installed"):
        linux_management.verify_uninstalled(tmp_path / "absent", 50)


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
