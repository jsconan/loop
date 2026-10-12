"""Verify fixed generation admission, protected membership and bounded peer frames."""

import socket
import stat
import struct
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import admission as linux_admission


@pytest.fixture
def admission(monkeypatch):
    """Model fixed installed authority and kernel/socket operations without native execution."""
    prefix = "loopgen_owned"
    group = Path("/sys/fs/cgroup") / (prefix + ".slice") / (prefix + "@1.service")
    records = {}

    def bound(path, **kwargs):
        """Provide a scoped installed inode double."""

        @contextmanager
        def verified(**kwargs):
            """Expose a fixed input descriptor for the authority's lifetime."""
            yield str(path)

        return SimpleNamespace(path=Path(path), open_verified=Mock(side_effect=verified))

    config = linux_admission.GenerationAdmission(
        "owned",
        1000,
        1000,
        Path("/installed/receiver"),
        bound("/installed/" + prefix + ".slice"),
        bound("/installed/reservations/owned"),
    )
    sources = config.unit_sources()
    files = {
        "socket_unit": bound("/installed/" + prefix + ".socket"),
        "service_unit": bound("/installed/" + prefix + "@.service"),
        "receiver": bound(config.receiver),
    }
    for name, data in sources.items():
        records["/installed/" + name] = data.encode()
    text = {
        "/proc/self/uid_map": "0 0 4294967295\n",
        "/proc/self/status": "NoNewPrivs: 1\n"
        + "\n".join(
            name + ": 00000000" for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
        ),
        "/proc/self/cgroup": "0::/" + str(group.relative_to("/sys/fs/cgroup")) + "\n",
        "/proc/self/mountinfo": "1 0 0:1 / /sys/fs/cgroup ro - cgroup2 cgroup rw\n",
        "/proc/self/fdinfo/9876": "pos: 0\nmnt_id: 1\n",
        str(group / "cgroup.type"): "domain\n",
    }
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    connection.getsockname.return_value = "/run/" + prefix + ".socket"
    options = {
        getattr(socket, "SO_DOMAIN", 39): socket.AF_UNIX,
        socket.SO_TYPE: socket.SOCK_STREAM,
        getattr(socket, "SO_PEERCRED", 17): struct.pack("3i", 123, 1000, 1000),
    }
    connection.getsockopt.side_effect = lambda level, option, *args: options[option]
    connection.recv.side_effect = [b"LOOP\x01" + struct.pack("!I", 3), b"abc", b""]
    state = SimpleNamespace(
        config=config,
        files=files,
        text=text,
        records=records,
        connection=connection,
        options=options,
        group=group,
        details=SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=0),
    )
    state.uid = Mock(return_value=(1000,) * 3)
    state.gid = Mock(return_value=(1000,) * 3)
    state.groups = Mock(return_value=[])
    state.attest = Mock()
    state.capacity = Mock(return_value=(29, 14680064))
    state.protected = Mock(side_effect=lambda path, **kwargs: path)
    monkeypatch.setattr(linux_admission.os, "getresuid", state.uid, raising=False)
    monkeypatch.setattr(linux_admission.os, "getresgid", state.gid, raising=False)
    monkeypatch.setattr(linux_admission.os, "getgroups", state.groups)
    monkeypatch.setattr(linux_admission.os, "getpid", lambda: 42)
    monkeypatch.setattr(linux_admission.os, "pread", lambda fd, size, offset: records[fd])
    monkeypatch.setattr(linux_admission.time, "monotonic", lambda: 10)
    monkeypatch.setattr(linux_admission.socket, "fromfd", Mock(return_value=connection))
    monkeypatch.setattr(linux_admission, "bind_host_file", bound)
    monkeypatch.setattr(linux_admission, "protected_path", state.protected)
    monkeypatch.setattr(linux_admission, "attest_units", state.attest)
    monkeypatch.setattr(linux_admission, "generation_limits", state.capacity)
    original_read = Path.read_text
    original_stat = Path.lstat
    original_open = linux_admission.os.open
    original_close = linux_admission.os.close
    state.close = Mock()
    monkeypatch.setattr(
        linux_admission.os,
        "open",
        lambda path, flags, *args, **kwargs: (
            9876 if path == Path("/sys/fs/cgroup") else original_open(path, flags, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        linux_admission.os,
        "close",
        lambda descriptor: (
            state.close(descriptor) if descriptor == 9876 else original_close(descriptor)
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            text[str(path)] if str(path) in text else original_read(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: (
            state.details if path.is_relative_to("/sys/fs/cgroup") else original_stat(path)
        ),
    )
    return state


def receive(state, deadline=100):
    """Call the public receiver with fixed preparation authority."""
    return linux_admission.receive_request(state.config, state.files, deadline)


def test_unit_sources_have_fixed_root_drop_and_no_listener_kill_dependency(admission):
    """Root executes only fixed privilege-drop arguments before the ordinary frame receiver."""
    sources = admission.config.unit_sources()
    service = sources["loopgen_owned@.service"]
    assert "--reuid=1000 --regid=1000 --clear-groups --bounding-set=-all" in service
    assert "--no-new-privs /usr/bin/python3 -I -B /installed/receiver" in service
    assert "ExitType=cgroup" in service and "Slice=loopgen_owned.slice" in service
    assert "ProtectControlGroups=yes" in service and "Delegate=no" in service
    assert "MemoryPressureWatch=off\n" in service
    assert "NoNewPrivileges=yes\n" in service
    assert "CapabilityBoundingSet=CAP_SETUID CAP_SETGID CAP_SETPCAP\n" in service
    assert "AmbientCapabilities=\n" in service
    # Keep SETUID available to the fixed launcher, then require zero caps in the receiver.
    assert "RestrictSUIDSGID=" not in service
    assert "BindsTo=" not in service and "PartOf=" not in service
    assert "SocketMode=0600" in sources["loopgen_owned.socket"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", "bad/name"),
        ("uid", 0),
        ("uid", True),
        ("gid", 0),
        ("gid", True),
        ("receiver", Path("/bad path")),
        ("receiver", Path("/../receiver")),
        ("slice_unit", SimpleNamespace(path=Path("/wrong.slice"))),
    ],
)
def test_unit_sources_refuse_invalid_fixed_authority(admission, field, value):
    """Malformed administrator inputs cannot become systemd syntax or root arguments."""
    values = vars(admission.config) | {field: value}
    with pytest.raises(ValueError, match="fixed|fragment"):
        linux_admission.GenerationAdmission(**values).unit_sources()


def test_admission_group_authenticates_fixed_kernel_scope(admission):
    """The cleanup boundary exposes only the receiver's validated protected kernel domain."""
    assert linux_admission.admission_group(admission.config) == admission.group
    admission.connection.recv.assert_not_called()


def test_admission_group_refuses_invalid_fixed_identity(admission):
    """Malformed configuration cannot supply a cleanup boundary even with a trusted kernel view."""
    config = linux_admission.GenerationAdmission(**(vars(admission.config) | {"uid": 0}))
    with pytest.raises(ValueError, match="fixed generation admission authority"):
        linux_admission.admission_group(config)


def test_receive_request_attests_before_reading_and_revalidates(admission):
    """A complete bounded frame returns only after peer, kernel and installed authority checks."""
    assert receive(admission) == b"abc"
    expected = admission.attest.call_args.args[1]["loopgen_owned@1.service"]
    assert expected["MainPID"] == "42"
    assert expected["ControlGroup"] == "/loopgen_owned.slice/loopgen_owned@1.service"
    assert expected["User"] == "0" and "--clear-groups" in expected["ExecStart"]
    assert expected["MemoryPressureWatch"] == "off"
    assert admission.attest.call_args.args[2] == 15
    assert admission.config.slice_unit.open_verified.call_count == 2
    assert admission.connection.settimeout.call_args.args == (5,)
    admission.connection.__exit__.assert_called_once()


@pytest.mark.parametrize("field", ["uid", "gid", "groups", "mapping", "nnpriv", "capability"])
def test_receive_request_refuses_retained_privilege_before_client_reads(admission, field):
    """Host root, supplementary identity, UID mapping and retained capabilities fail closed."""
    if field in ("uid", "gid"):
        getattr(admission, field).return_value = (0, 1000, 1000)
    elif field == "groups":
        admission.groups.return_value = [1000]
    elif field == "mapping":
        admission.text["/proc/self/uid_map"] = "0 1000 1"
    elif field == "nnpriv":
        admission.text["/proc/self/status"] = "NoNewPrivs: 0"
    else:
        admission.text["/proc/self/status"] += "\nCapEff: 1"
    with pytest.raises(ValueError, match="identity|privilege"):
        receive(admission)
    admission.connection.recv.assert_not_called()
    admission.attest.assert_not_called()


@pytest.mark.parametrize("text", ["0::/user.slice", "0::/loopgen_owned.slice/fake.service", ""])
def test_receive_request_refuses_untracked_direct_entry(admission, text):
    """Launching the receiver directly cannot bypass generation service membership."""
    admission.text["/proc/self/cgroup"] = text
    with pytest.raises(ValueError, match="membership"):
        receive(admission)
    admission.connection.recv.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        "invalid",
        "1 0 0:1 / /sys/fs/cgroup ro cgroup2 cgroup rw",
        "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw",
        "1 0 0:1 / /sys/fs/cgroup ro - tmpfs cgroup rw",
        "1 0 0:1 /other /sys/fs/cgroup ro - cgroup2 cgroup rw",
        "",
        "1 0 0:1 / /sys/fs/cgroup/child ro - cgroup2 cgroup rw",
        (
            "1 0 0:1 / /sys/fs/cgroup ro - cgroup2 cgroup rw\n"
            "2 1 0:2 / /sys/fs/cgroup/child ro - cgroup2 cgroup rw"
        ),
    ],
)
def test_receive_request_refuses_ambiguous_or_writable_cgroup_mount(admission, text):
    """Only one readonly unified whole hierarchy can establish protected membership."""
    admission.text["/proc/self/mountinfo"] = text
    with pytest.raises(ValueError, match="mount|unified"):
        receive(admission)
    admission.connection.recv.assert_not_called()


def test_receive_request_refuses_writable_pressure_file_submount(admission):
    """A pressure-file write exception cannot bypass the protected whole-hierarchy contract."""
    pressure = "/loopgen_owned.slice/loopgen_owned@1.service/memory.pressure"
    admission.text["/proc/self/mountinfo"] = (
        "6915 6912 0:34 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime "
        "shared:926 master:10 - cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n"
        f"6950 6915 0:34 {pressure} /sys/fs/cgroup{pressure} "
        "rw,nosuid,nodev,noexec,relatime shared:927 master:10 "
        "- cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot"
    )
    admission.text["/proc/self/fdinfo/9876"] = "mnt_id:\t6915\n"
    with pytest.raises(ValueError, match="submount or different hierarchy"):
        receive(admission)
    admission.connection.recv.assert_not_called()
    admission.attest.assert_not_called()


@pytest.mark.parametrize("reverse", [False, True])
def test_receive_request_accepts_kernel_bound_readonly_mount_stack(admission, reverse):
    """A hidden writable mount cannot override the kernel-selected readonly bind view."""
    rows = [
        "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw",
        "2 1 0:1 / /sys/fs/cgroup ro - cgroup2 cgroup rw",
    ]
    admission.text["/proc/self/mountinfo"] = "\n".join(rows[::-1] if reverse else rows)
    admission.text["/proc/self/fdinfo/9876"] = "mnt_id: 2\n"
    assert receive(admission) == b"abc"
    assert admission.close.call_count == 2


@pytest.mark.parametrize(
    "fdinfo,extra,reason",
    [
        ("mnt_id: 1", "", "writable"),
        ("mnt_id: 3", "", "ID is missing or ambiguous"),
        ("mnt_id: 2\nmnt_id: 2", "", "ID is missing or ambiguous"),
        ("mnt_id:", "", "ID is missing or ambiguous"),
        ("", "", "ID is missing or ambiguous"),
        (
            "mnt_id: 2",
            "\n3 2 0:2 / /sys/fs/cgroup ro - cgroup2 other rw",
            "submount or different hierarchy",
        ),
        (
            "mnt_id: 2",
            "\n2 1 0:1 / /sys/fs/cgroup ro - cgroup2 cgroup rw",
            "ID is missing or ambiguous",
        ),
    ],
)
def test_receive_request_refuses_unbound_or_writable_visible_mount(
    admission, fdinfo, extra, reason
):
    """Hidden readonly rows, ambiguous mount IDs and another hierarchy cannot grant admission."""
    admission.text["/proc/self/mountinfo"] = (
        "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        "2 1 0:1 / /sys/fs/cgroup ro - cgroup2 cgroup rw" + extra
    )
    admission.text["/proc/self/fdinfo/9876"] = fdinfo
    with pytest.raises(ValueError, match=reason):
        receive(admission)
    admission.close.assert_called_once_with(9876)
    admission.connection.recv.assert_not_called()


@pytest.mark.parametrize("change", ["alias", "owner", "writable", "type", "threaded"])
def test_receive_request_refuses_delegated_or_aliased_tracking(admission, change):
    """Every ancestor's migration interfaces remain root-protected process domains."""
    if change == "alias":
        admission.protected.side_effect = lambda path, **kwargs: Path("/alias")
    elif change == "threaded":
        admission.text[str(admission.group / "cgroup.type")] = "threaded"
    else:
        admission.details = SimpleNamespace(
            st_mode=(stat.S_IFDIR if change == "type" else stat.S_IFREG)
            | (0o666 if change == "writable" else 0o644),
            st_uid=1000 if change == "owner" else 0,
        )
    with pytest.raises(ValueError, match="ancestry|delegated|domain"):
        receive(admission)
    admission.connection.recv.assert_not_called()


@pytest.mark.parametrize("role", ["socket_unit", "service_unit", "slice", "receiver", "loaded"])
def test_receive_request_refuses_changed_installed_or_loaded_authority(admission, role):
    """Unit, receiver and manager drift stop before client data is read."""
    if role == "receiver":
        admission.files[role].path = Path("/another/receiver")
    elif role == "loaded":
        admission.attest.side_effect = ValueError("Loaded authority changed")
    else:
        path = admission.config.slice_unit.path if role == "slice" else admission.files[role].path
        admission.records[str(path)] += b"changed"
    with pytest.raises(ValueError, match="unit|slice|receiver|Loaded"):
        receive(admission)
    admission.connection.recv.assert_not_called()


@pytest.mark.parametrize("change", ["domain", "type", "path", "pid", "uid", "gid"])
def test_receive_request_refuses_wrong_socket_and_peers(admission, change):
    """Unactivated, foreign or wrong-identity connections never supply command frames."""
    if change in ("domain", "type"):
        option = getattr(socket, "SO_DOMAIN", 39) if change == "domain" else socket.SO_TYPE
        admission.options[option] = -1
    elif change == "path":
        admission.connection.getsockname.return_value = "/run/other.socket"
    else:
        values = [123, 1000, 1000]
        values[{"pid": 0, "uid": 1, "gid": 2}[change]] = 0
        admission.options[getattr(socket, "SO_PEERCRED", 17)] = struct.pack("3i", *values)
    with pytest.raises(ValueError, match="descriptor|peer"):
        receive(admission)
    admission.connection.recv.assert_not_called()


@pytest.mark.parametrize(
    "parts",
    [
        [b""],
        [b"LOOP\x02" + struct.pack("!I", 3)],
        [b"LOOP\x01" + struct.pack("!I", 0)],
        [b"LOOP\x01" + struct.pack("!I", 65537)],
        [b"LOOP\x01" + struct.pack("!I", 3), b"a", b""],
        [b"LOOP\x01" + struct.pack("!I", 3), b"abc", b"x"],
    ],
)
def test_receive_request_refuses_malformed_truncated_or_trailing_frames(admission, parts):
    """Bounded protocol parsing accepts neither partial frames nor multiple requests."""
    admission.connection.recv.side_effect = parts
    with pytest.raises(ValueError, match="frame"):
        receive(admission)


def test_receive_request_handles_fragmentation_and_exact_capacity(admission):
    """Fragmented header and maximum-size payload remain inside a single shared budget."""
    admission.connection.recv.side_effect = [
        b"L",
        b"OOP\x01" + struct.pack("!I", 65536),
        b"a" * 65536,
        b"",
    ]
    assert receive(admission) == b"a" * 65536


@pytest.mark.parametrize("boundary", ["start", "read", "end"])
def test_receive_request_refuses_expired_budget(admission, monkeypatch, boundary):
    """The admission deadline covers attestation, all stream reads and final revalidation."""
    if boundary == "read":
        admission.attest.side_effect = lambda *args: monkeypatch.setattr(
            linux_admission.time, "monotonic", lambda: 100
        )
    elif boundary == "end":
        admission.connection.__exit__.side_effect = lambda *args: monkeypatch.setattr(
            linux_admission.time, "monotonic", lambda: 100
        )
    with pytest.raises(TimeoutError, match="deadline"):
        receive(admission, deadline=0 if boundary == "start" else 100)


def test_receive_request_propagates_socket_timeout(admission):
    """A peer that stalls cannot yield partial admitted data."""
    admission.connection.recv.side_effect = TimeoutError("peer stalled")
    with pytest.raises(TimeoutError, match="peer stalled"):
        receive(admission)


def test_receive_request_refuses_membership_change_after_frame(admission):
    """Late migration cannot return request bytes as valid generation admission."""
    admission.connection.__exit__.side_effect = lambda *args: admission.text.update(
        {
            "/proc/self/cgroup": "0::/loopgen_owned.slice/loopgen_owned@2.service",
            "/sys/fs/cgroup/loopgen_owned.slice/loopgen_owned@2.service/cgroup.type": "domain",
        }
    )
    with pytest.raises(ValueError, match="changed"):
        receive(admission)


def test_receive_request_refuses_late_installed_drift(admission):
    """Authority changes after EOF prevent the receiver from returning client bytes."""
    admission.connection.__exit__.side_effect = lambda *args: setattr(
        admission.files["receiver"].open_verified, "side_effect", ValueError("receiver drift")
    )
    with pytest.raises(ValueError, match="receiver drift"):
        receive(admission)


def test_receive_request_accepts_other_mounts_without_widening_tracking(admission):
    """Unrelated host mounts do not substitute for or obscure the cgroup authority."""
    admission.text["/proc/self/mountinfo"] += "2 0 0:2 / /other rw - tmpfs other rw\n"
    assert receive(admission) == b"abc"


def test_receive_request_retains_original_readonly_input_descriptors(admission):
    """Close duplicated receiver transport and retain scoped installation inputs."""
    assert receive(admission) == b"abc"
    for bound in admission.files.values():
        assert bound.open_verified.called
    linux_admission.socket.fromfd.assert_called_once_with(0, socket.AF_UNIX, socket.SOCK_STREAM)


def test_receive_request_requires_existing_generation_reservation(admission):
    """Tracking cannot admit an unreserved generation or reset installation capacity."""
    admission.capacity.side_effect = ValueError("changed generation capacity")
    with pytest.raises(ValueError, match="capacity"):
        receive(admission)
    admission.connection.recv.assert_not_called()
    admission.capacity.assert_called_once_with(admission.config.reservation, "owned", deadline=15)
