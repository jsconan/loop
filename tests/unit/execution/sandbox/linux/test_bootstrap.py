"""Verify setup authority is irreversibly removed before a payload transition."""

import ctypes
import errno
import io
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import bootstrap as linux_bootstrap
from loop.execution.sandbox.linux.constants import NATIVE_SCRATCH_PATH


@pytest.fixture
def native(monkeypatch):
    """Supply isolated native interfaces and exact mapped bootstrap state."""
    setup = (1 << 33) | (1 << 8)
    keys = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")

    def status(value):
        """Render every capability set and NoNewPrivs as a proc status record."""
        return (
            "Name:\tpython\n"
            + "".join(f"{key}:\t{value:016x}\n" for key in keys)
            + ("NoNewPrivs:\t1\n")
        )

    state = {
        "uid": 0,
        "reads": {
            "/proc/self/uid_map": "         0       1000          1\n",
            "/proc/self/attr/current": "bootstrap (enforce)\n",
            "/sys/kernel/security/apparmor/.ns_name": "loop_sandbox\n",
            "/sys/kernel/security/apparmor/.ns_level": "1\n",
            "/sys/kernel/security/apparmor/.ns_stacked": "yes\n",
            "/proc/sys/kernel/cap_last_cap": "40\n",
        },
        "statuses": [status(setup), status(0), status(0)],
        "events": [],
        "libc": Mock(),
        "apparmor": Mock(),
        "payload_label": "bootstrap//&payload (enforce)\n",
    }

    def read(path, **kwargs):
        """Return bound metadata without accessing any native filesystem."""
        assert kwargs == {"encoding": "ascii"}
        if str(path) == "/proc/self/status":
            return state["statuses"].pop(0)
        return state["reads"][str(path)]

    def prctl(*args):
        """Record irreversible operations in their actual order."""
        state["events"].append(("prctl", args))
        return 0

    def capset(header, words):
        """Check version, self PID and both zero-filled capability words."""
        state["events"].append(("capset",))
        raw = ctypes.string_at(header, 8)
        assert int.from_bytes(raw[:4], "little") == 0x20080522
        assert raw[4:] == bytes(4)
        assert ctypes.string_at(words, 24) == bytes(24)
        return 0

    def change(profile):
        """Require stacked entry only after capability and NoNewPrivs removal."""
        state["events"].append(("stack", profile))
        assert profile == b"&payload"
        assert state["events"][-2] == ("prctl", (38, 1, 0, 0, 0))
        state["reads"]["/proc/self/attr/current"] = state["payload_label"]
        return 0

    def library(name, **kwargs):
        """Expose only the two fixed native libraries."""
        assert kwargs == {"use_errno": True}
        assert name in (None, "/proc/self/fd/99")
        return state["libc"] if name is None else state["apparmor"]

    @contextmanager
    def verified():
        """Expose only the pinned host-prepared library descriptor."""
        state["events"].append(("library_open",))
        try:
            yield 99
        finally:
            state["events"].append(("library_close",))

    state["authority"] = Mock(open_verified=verified)
    state["libc"].prctl.side_effect = prctl
    state["libc"].capset.side_effect = capset
    state["apparmor"].aa_change_profile.side_effect = change
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(linux_bootstrap.os, "getuid", lambda: state["uid"])
    monkeypatch.setattr(ctypes, "CDLL", Mock(side_effect=library))
    monkeypatch.setattr(ctypes, "get_errno", lambda: 13)
    return state


@pytest.mark.parametrize(
    "failure", [None, "name", "uid", "level", "mapping", "api", "native", "enforcement"]
)
def test_enter_bootstrap_requires_exact_mapping_native_authority_and_enforcement(native, failure):
    """Only an authenticated native stack in the exact enforcing mapped setup can return."""
    namespace, uid, level = "loop_sandbox", 1000, 1
    transition = native["apparmor"].aa_change_profile
    transition.side_effect = None
    transition.return_value = 0
    if failure == "name":
        namespace = ":other:"
    elif failure == "uid":
        uid = True
    elif failure == "level":
        level = 3
    elif failure == "mapping":
        native["reads"]["/proc/self/uid_map"] = "0 0 4294967295"
    elif failure == "api":
        del native["apparmor"].aa_change_profile
    elif failure == "native":
        transition.return_value = -1
    elif failure == "enforcement":
        native["reads"]["/proc/self/attr/current"] = "bootstrap (complain)"
    if failure:
        error, message = {
            "name": (ValueError, "Invalid payload transition identity"),
            "uid": (ValueError, "Invalid payload transition identity"),
            "level": (ValueError, "Invalid payload transition identity"),
            "mapping": (ValueError, "Unverified bootstrap user mapping"),
            "api": (OSError, "Required libapparmor API is unavailable"),
            "native": (OSError, "Bootstrap profile stack failed"),
            "enforcement": (ValueError, "Unverified bootstrap confinement"),
        }[failure]
        with pytest.raises(error, match=message):
            linux_bootstrap.enter_bootstrap(
                native["authority"], namespace, uid, namespace_level=level
            )
        if failure in ("name", "uid", "level", "mapping"):
            assert native["events"] == []
            transition.assert_not_called()
        else:
            assert native["events"] == [("library_open",), ("library_close",)]
    else:
        linux_bootstrap.enter_bootstrap(native["authority"], namespace, uid)
        transition.assert_called_once_with(b"&:loop_sandbox:bootstrap")
        assert transition.argtypes == [ctypes.c_char_p]
        assert transition.restype == ctypes.c_int
        assert native["events"] == [("library_open",), ("library_close",)]


@pytest.fixture
def cache(tmp_path, native, monkeypatch):
    """Model kernel-exported profiles and add-only writes behind a shared read-only lock."""
    security = Path("/sys/kernel/security/apparmor")
    directory = tmp_path / "profiles"
    directory.mkdir()
    lock = tmp_path / "lock"
    lock.write_bytes(b"loop_sandbox\n")
    lock.chmod(0o444)
    state = {
        "directory": directory,
        "lock": lock,
        "loads": [],
        "after_load": None,
        "load_error": None,
        "written": None,
        "flock": Mock(),
        "closed": [],
    }
    native["statuses"] = [native["statuses"][0]] * 32
    real_open = Path.open
    real_iterdir = Path.iterdir
    real_close = os.close

    def add(name, data=b"compiled", mode=b"enforce\n"):
        """Export one enforcing policy and its exact serialized bytes."""
        entry = directory / name
        entry.mkdir()
        (entry / "name").write_bytes(name.encode("ascii") + b"\n")
        (entry / "mode").write_bytes(mode)
        (entry / "raw_data").write_bytes(data)
        return entry

    class LoadStream(io.BytesIO):
        """Expose one kernel load write with controllable outcomes and exported data."""

        def write(self, data):
            """Record admission without running any native process or changing host policy."""
            state["loads"].append(data)
            if state["load_error"] is not None:
                raise state["load_error"]
            if state["after_load"] is None:
                add("payload", data)
            else:
                state["after_load"](data)
            return len(data) if state["written"] is None else state["written"]

    def open_path(path, *args, **kwargs):
        """Redirect only the kernel write interface to the deterministic model."""
        if path == security / ".load":
            assert args == ("wb",) and kwargs == {"buffering": 0}
            return LoadStream()
        return real_open(path, *args, **kwargs)

    def iterdir(path):
        """Provide a real bounded inventory from disposable exported profiles."""
        return real_iterdir(directory if path == security / "policy/profiles" else path)

    def close(descriptor):
        """Record lock release on every successful and failed outcome."""
        state["closed"].append(descriptor)
        real_close(descriptor)

    add("bootstrap")
    state["add"] = add
    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(linux_bootstrap.fcntl, "flock", state["flock"])
    monkeypatch.setattr(os, "close", close)
    return state


def test_load_cached_payload_adds_once_and_reuses_only_exact_bytes(cache):
    """An enforcing byte-identical entry is reused without a second kernel write."""
    args = ("payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1)
    assert linux_bootstrap.load_cached_payload(*args) is True
    assert linux_bootstrap.load_cached_payload(*args) is False
    assert cache["loads"] == [b"compiled"]
    assert len(cache["closed"]) == 2
    assert all(call.args[1] == 6 for call in cache["flock"].call_args_list)


@pytest.mark.parametrize("actual, expected", [(1, 2), (2, 1)])
def test_load_cached_payload_refuses_a_different_namespace_depth(cache, native, actual, expected):
    """A requested namespace depth cannot admit cache access from another native depth."""
    native["reads"]["/sys/kernel/security/apparmor/.ns_level"] = f"{actual}\n"
    with pytest.raises(ValueError, match="Unverified bootstrap confinement"):
        linux_bootstrap.load_cached_payload(
            "payload",
            b"compiled",
            "loop_sandbox",
            1000,
            cache["lock"],
            1,
            namespace_level=expected,
        )
    assert cache["loads"] == []
    cache["flock"].assert_not_called()


def test_load_cached_payload_adds_and_reuses_in_explicit_nested_context(cache, native):
    """An explicitly bound nested bootstrap retains add-only exact reuse semantics."""
    native["reads"]["/sys/kernel/security/apparmor/.ns_level"] = "2\n"
    args = ("payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1)
    assert linux_bootstrap.load_cached_payload(*args, namespace_level=2) is True
    assert linux_bootstrap.load_cached_payload(*args, namespace_level=2) is False
    assert cache["loads"] == [b"compiled"]


@pytest.mark.parametrize(
    ("data", "lock", "capacity"),
    [
        (b"", None, 1),
        (bytearray(b"x"), None, 1),
        (b"x" * 1048577, None, 1),
        (b"x", Path("relative"), 1),
        (b"x", None, 0),
        (b"x", None, True),
    ],
    ids=["empty", "mutable", "oversized", "relative_lock", "zero_capacity", "boolean_capacity"],
)
def test_load_cached_payload_rejects_unbounded_or_invalid_inputs(cache, data, lock, capacity):
    """Invalid bounds and lock paths fail before admission or lock acquisition."""
    with pytest.raises(ValueError, match="Invalid policy cache input"):
        linux_bootstrap.load_cached_payload(
            "payload", data, "loop_sandbox", 1000, lock or cache["lock"], capacity
        )
    assert not cache["loads"] and not cache["closed"]


@pytest.mark.parametrize("change", ["writable", "hardlink", "contents", "suffix", "symlink"])
def test_load_cached_payload_rejects_unsafe_lock(cache, tmp_path, change):
    """Writable, aliased or misbound locks cannot serialize policy admission."""
    lock = cache["lock"]
    if change == "writable":
        lock.chmod(0o644)
    elif change == "hardlink":
        os.link(lock, tmp_path / "alias")
    elif change in ("contents", "suffix"):
        lock.chmod(0o644)
        lock.write_bytes(b"other_sandbox\n" if change == "contents" else b"loop_sandbox\nx")
        lock.chmod(0o444)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(lock)
        lock = alias
    with pytest.raises((ValueError, OSError)) as caught:
        linux_bootstrap.load_cached_payload("payload", b"compiled", "loop_sandbox", 1000, lock, 1)
    if change == "symlink":
        assert caught.value.errno == errno.ELOOP
        assert not cache["closed"]
    else:
        assert str(caught.value) == "Invalid policy cache lock."
        assert len(cache["closed"]) == 1
    assert not cache["loads"]


def test_load_cached_payload_rejects_nonregular_lock(cache, monkeypatch):
    """A descriptor to a nonregular object cannot establish admission serialization."""
    monkeypatch.setattr(os, "fstat", lambda fd: SimpleNamespace(st_mode=0o040444))
    with pytest.raises(ValueError, match="Invalid policy cache lock"):
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
    assert not cache["loads"] and len(cache["closed"]) == 1


def test_load_cached_payload_refuses_busy_lock(cache):
    """Concurrent admission fails immediately instead of loading outside the lock."""
    cache["flock"].side_effect = BlockingIOError(errno.EWOULDBLOCK, "busy")
    with pytest.raises(BlockingIOError, match="busy"):
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
    assert not cache["loads"] and len(cache["closed"]) == 1


@pytest.mark.parametrize("when", ["existing", "new"])
def test_load_cached_payload_rejects_different_loaded_bytes(cache, when):
    """A matching name never substitutes for exact compiled-policy identity."""
    if when == "existing":
        cache["add"]("payload", b"weaker")
    else:
        cache["after_load"] = lambda data: cache["add"]("payload", b"weaker")
    with pytest.raises(ValueError, match="Loaded policy differs"):
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
    assert len(cache["loads"]) == (when == "new")


def test_load_cached_payload_refuses_new_variant_at_capacity_but_reuses_existing(cache):
    """Capacity does not revoke active policies or prevent exact reuse."""
    cache["add"]("other")
    with pytest.raises(OSError, match="administrator refresh required") as caught:
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
    assert caught.value.errno == errno.ENOSPC
    assert (
        linux_bootstrap.load_cached_payload(
            "other", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
        is False
    )
    assert not cache["loads"]


def test_load_cached_payload_accepts_maximum_length_profile_name(cache):
    """A valid 128-character kernel profile name includes its newline in the read bound."""
    name = "p" * 128
    cache["add"](name)
    assert (
        linux_bootstrap.load_cached_payload(
            name, b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
        is False
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("overfull", "exceeds its capacity"),
        ("duplicate", "Ambiguous"),
        ("invalid", "Ambiguous"),
        ("mode", "not enforcing"),
        ("nested", "Nested profiles"),
        ("bootstrap", "no bootstrap"),
        ("oversized", "Oversized"),
    ],
)
def test_load_cached_payload_rejects_invalid_kernel_inventory(cache, change, message):
    """Malformed, nested, overfull or nonenforcing inventories fail closed."""
    entry = cache["directory"] / "bootstrap"
    if change == "overfull":
        cache["add"]("one")
        cache["add"]("two")
    elif change == "duplicate":
        (cache["add"]("other") / "name").write_bytes(b"bootstrap\n")
    elif change == "invalid":
        (entry / "name").write_bytes(b"nested//name\n")
    elif change == "mode":
        (entry / "mode").write_bytes(b"complain\n")
    elif change == "nested":
        (entry / "profiles").mkdir()
        (entry / "profiles/child").mkdir()
    elif change == "bootstrap":
        (entry / "name").write_bytes(b"other\n")
    else:
        (entry / "name").write_bytes(b"x" * 130)
    with pytest.raises(ValueError, match=message):
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
    assert not cache["loads"]


def test_load_cached_payload_accepts_empty_child_inventory(cache):
    """An empty optional subprofile directory adds no hidden authority."""
    (cache["directory"] / "bootstrap/profiles").mkdir()
    assert (
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
        is True
    )


@pytest.mark.parametrize("failure", ["write", "short", "inventory", "missing_export"])
def test_load_cached_payload_stops_on_failed_or_unverifiable_admission(cache, failure):
    """A failed write or mismatched post-load export never authorizes a payload."""
    if failure == "write":
        cache["load_error"] = PermissionError(errno.EACCES, "denied")
    elif failure == "short":
        cache["written"] = 1
    elif failure == "inventory":
        cache["after_load"] = lambda data: cache["add"]("unexpected", data)
    else:
        cache["after_load"] = lambda data: (cache["add"]("payload", data) / "raw_data").unlink()
    expected = {
        "write": (PermissionError, "denied"),
        "short": (OSError, "Incomplete"),
        "inventory": (ValueError, "Unexpected policy cache change"),
        "missing_export": (FileNotFoundError, "raw_data"),
    }
    exception, message = expected[failure]
    with pytest.raises(exception, match=message):
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 1
        )
    assert len(cache["closed"]) == 1


def test_enter_payload_revokes_every_capability_before_stacking(native):
    """A successful return retains confinement and zeroes every authority set."""
    assert (
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000) is None
    )
    assert native["events"] == [
        ("library_open",),
        ("library_close",),
        *(("prctl", (24, cap, 0, 0, 0)) for cap in range(41)),
        ("prctl", (47, 4, 0, 0, 0)),
        ("capset",),
        ("prctl", (38, 1, 0, 0, 0)),
        ("stack", b"&payload"),
    ]


def test_enter_payload_revokes_authority_in_explicit_nested_context(native):
    """Nested payload entry irreversibly removes authority before the same stack operation."""
    native["reads"]["/sys/kernel/security/apparmor/.ns_level"] = "2\n"
    assert (
        linux_bootstrap.enter_payload(
            native["authority"], "payload", "loop_sandbox", 1000, namespace_level=2
        )
        is None
    )
    assert native["events"][-4:] == [
        ("prctl", (47, 4, 0, 0, 0)),
        ("capset",),
        ("prctl", (38, 1, 0, 0, 0)),
        ("stack", b"&payload"),
    ]


@pytest.mark.parametrize("level", [0, 3, -1, True, "2", None])
def test_enter_payload_rejects_unsupported_namespace_depth_without_revocation(native, level):
    """Unsupported depths cannot reach native authority changes or profile stacking."""
    with pytest.raises(ValueError, match="Invalid payload transition identity"):
        linux_bootstrap.enter_payload(
            native["authority"], "payload", "loop_sandbox", 1000, namespace_level=level
        )
    assert native["events"] == []


@pytest.mark.parametrize(
    ("profile", "namespace", "uid"),
    [
        ("&payload", "loop_sandbox", 1000),
        ("bootstrap", "loop_sandbox", 1000),
        ("payload", ":other:", 1000),
        ("payload", "loop_sandbox", True),
        ("payload", "loop_sandbox", 0),
    ],
)
def test_enter_payload_rejects_invalid_transition_identity(native, profile, namespace, uid):
    """Malformed or setup-directed identities cannot reach any native operation."""
    with pytest.raises(ValueError, match="Invalid payload transition identity"):
        linux_bootstrap.enter_payload(native["authority"], profile, namespace, uid)
    assert not native["events"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("/proc/self/uid_map", "0 0 4294967295"),
        ("/proc/self/attr/current", "unconfined"),
        ("/sys/kernel/security/apparmor/.ns_name", "other"),
        ("/sys/kernel/security/apparmor/.ns_level", "0"),
        ("/sys/kernel/security/apparmor/.ns_stacked", "no"),
    ],
)
def test_enter_payload_rejects_unbound_native_confinement(native, path, value):
    """Wrong mappings, labels or namespace identity fail before capabilities change."""
    native["reads"][path] = value
    with pytest.raises(ValueError, match="Unverified bootstrap confinement"):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert not native["events"]


def test_enter_payload_requires_mapped_root_not_the_host_loop_process(native):
    """An ordinary Loop process cannot accidentally enter this child-only primitive."""
    native["uid"] = 1000
    with pytest.raises(ValueError, match="Unverified bootstrap confinement"):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert not native["events"]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("CapEff: 0\n", "Incomplete bootstrap status"),
        ("CapEff: 0\nCapEff: 0\n", "Duplicate bootstrap status field"),
        ("CapEff: invalid\n", "invalid literal"),
    ],
)
def test_enter_payload_requires_complete_unambiguous_status(native, status, message):
    """Missing, duplicated and malformed native status cannot authorize a transition."""
    native["statuses"][0] = status
    with pytest.raises(ValueError, match=message):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert not native["events"]


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ((0, "NoNewPrivs:\t1", "NoNewPrivs:\t0"), None, "Unexpected bootstrap capabilities"),
        ((0, "0000000200000100", "0000000000000000"), None, "Unexpected bootstrap capabilities"),
        (
            (0, "CapAmb:\t0000000200000100", "CapAmb:\t0000000200000101"),
            None,
            "Unexpected bootstrap capabilities",
        ),
        (None, (1, "NoNewPrivs:\t1", "NoNewPrivs:\t0"), "Bootstrap authority was not revoked"),
        (
            None,
            (1, "CapBnd:\t0000000000000000", "CapBnd:\t0000000000000001"),
            "Bootstrap authority was not revoked",
        ),
        (
            None,
            (2, "CapAmb:\t0000000000000000", "CapAmb:\t0000000000000001"),
            "Unexpected payload confinement",
        ),
    ],
)
def test_enter_payload_checks_initial_revoked_and_final_authority(native, before, after, message):
    """Authority mismatch never returns successful payload readiness."""
    for replacement in (before, after):
        if replacement is not None:
            index, original, changed = replacement
            native["statuses"][index] = native["statuses"][index].replace(original, changed)
    with pytest.raises(ValueError, match=message):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    if message != "Unexpected payload confinement":
        native["apparmor"].aa_change_profile.assert_not_called()


@pytest.mark.parametrize("last", ["32", "64", "invalid"])
def test_enter_payload_rejects_unsupported_capability_range(native, last):
    """Unknown capability ranges cannot leave unhandled bounding bits."""
    native["reads"]["/proc/sys/kernel/cap_last_cap"] = last
    with pytest.raises(ValueError):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert not native["events"]


@pytest.mark.parametrize("operation", ["bounding", "ambient", "capset", "nnp", "stack"])
def test_enter_payload_stops_on_native_operation_failure(native, operation):
    """Each irreversible native failure stops rather than entering a fallback context."""
    if operation in ("bounding", "ambient", "nnp"):
        command = {"bounding": 24, "ambient": 47, "nnp": 38}[operation]
        native["libc"].prctl.side_effect = lambda *args: -1 if args[0] == command else 0
    elif operation == "capset":
        native["libc"].capset.side_effect = None
        native["libc"].capset.return_value = -1
    else:
        native["apparmor"].aa_change_profile.side_effect = None
        native["apparmor"].aa_change_profile.return_value = -1
    with pytest.raises(OSError) as error:
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert error.value.errno == 13
    if operation != "stack":
        native["apparmor"].aa_change_profile.assert_not_called()


def test_enter_payload_rejects_unexpected_post_transition_label(native):
    """A successful system call with the wrong enforcing label is not readiness."""
    native["payload_label"] = "payload (complain)"
    with pytest.raises(ValueError, match="Unexpected payload confinement"):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)


def test_enter_payload_propagates_missing_native_interface_without_transition(native):
    """Missing proc/security metadata cannot turn into a successful launch."""
    native["reads"]["/proc/self/uid_map"] = None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "read_text", Mock(side_effect=FileNotFoundError("missing proc")))
        with pytest.raises(FileNotFoundError, match="missing proc"):
            linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert not native["events"]


def test_enter_payload_refuses_changed_library_before_revocation(native):
    """Library drift cannot reach any irreversible operation or profile transition."""
    native["authority"].open_verified = Mock(side_effect=ValueError("Prepared authority changed"))
    with pytest.raises(ValueError, match="Prepared authority changed"):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert native["events"] == []
    native["apparmor"].aa_change_profile.assert_not_called()


@pytest.mark.parametrize("failure", ["load", "api"])
def test_enter_payload_closes_library_on_preparation_failure(native, monkeypatch, failure):
    """Library load/API failures close the prepared descriptor before any capability changes."""
    if failure == "load":
        real_loader = ctypes.CDLL
        monkeypatch.setattr(
            ctypes,
            "CDLL",
            lambda name, **options: (
                real_loader(name, **options)
                if name is None
                else (_ for _ in ()).throw(OSError("load failed"))
            ),
        )
    else:
        del native["apparmor"].aa_change_profile
    with pytest.raises(OSError, match="load failed|Required libapparmor API"):
        linux_bootstrap.enter_payload(native["authority"], "payload", "loop_sandbox", 1000)
    assert native["events"] == [("library_open",), ("library_close",)]


@pytest.fixture
def activation(cache, native, monkeypatch, tmp_path):
    """Bind installed bootstrap bytes and isolate descriptor closure around real composition."""
    path = tmp_path / "bootstrap-policy"
    path.write_bytes(b"compiled")
    limits = Mock(return_value=(31, 16777216))
    monkeypatch.setattr(linux_bootstrap, "generation_limits", limits)

    @contextmanager
    def verified():
        """Pin the disposable installed-policy descriptor for authentication."""
        descriptor = os.open(path, os.O_RDONLY)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    @contextmanager
    def lock_verified():
        """Model the host-bound lock surrounding its independently opened admission lock."""
        yield 99

    state = SimpleNamespace(
        bootstrap=SimpleNamespace(
            identity=(1, 2, 0o100444, 1, 8, 1, 1), open_verified=Mock(side_effect=verified)
        ),
        lock=SimpleNamespace(
            path=cache["lock"],
            identity=(1, 3, 0o100444, 1, 13, 1, 1),
            open_verified=Mock(side_effect=lock_verified),
        ),
        reservation=object(),
        limits=limits,
        descriptors=["0", "1", "2"],
        cache=cache,
        native=native,
    )
    monkeypatch.setattr(os, "listdir", lambda path: state.descriptors)
    return state


def activate(state, **changes):
    """Invoke the complete public setup composition on a bound disposable generation."""
    linux_bootstrap.activate_payload(
        state.native["authority"],
        state.bootstrap,
        state.lock,
        "payload",
        b"compiled",
        "loop_sandbox",
        1000,
        reservation=state.reservation,
        **changes,
    )


def test_activate_payload_authenticates_loads_and_enters_irreversibly(activation):
    """Installed bootstrap authentication and add-only admission precede actual revocation."""
    native = activation.native
    native["statuses"] = [native["statuses"][0]] * 3 + [
        "".join(f"{key}: 0\n" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"))
        + "NoNewPrivs: 1\n"
    ] * 2
    activate(activation)
    assert activation.cache["loads"] == [b"compiled"]
    assert native["events"][-1] == ("stack", b"&payload")
    assert activation.lock.open_verified.call_count == 2


@pytest.mark.parametrize(
    "change", ["metadata", "empty", "oversized", "loaded", "lock", "drift", "postload"]
)
def test_activate_payload_refuses_unauthenticated_bootstrap(activation, monkeypatch, change):
    """Absent, changed and oversized bootstrap authority never reaches payload entry."""
    if change == "metadata":
        activation.bootstrap.identity = (1, 2, 0o100444, 1, 1048577, 1, 1)
    elif change in ("empty", "oversized"):
        monkeypatch.setattr(os, "pread", lambda *args: b"" if change == "empty" else b"x" * 1048577)
    elif change == "loaded":
        (activation.cache["directory"] / "bootstrap/raw_data").write_bytes(b"wrong")
    elif change == "lock":
        activation.lock.identity = (1, 2, 0o100644, 1, 1, 1, 1)
    elif change == "drift":
        activation.bootstrap.open_verified.side_effect = ValueError("authority changed")
    else:

        def load(data):
            """Expose a bootstrap mutation during a nominally successful add-only load."""
            activation.cache["add"]("payload", data)
            (activation.cache["directory"] / "bootstrap/raw_data").write_bytes(b"wrong")

        activation.cache["after_load"] = load
    with pytest.raises(ValueError, match="bootstrap|authority"):
        activate(activation)
    assert not activation.native["events"]


def test_activate_payload_refuses_generation_byte_capacity(activation):
    """Aggregate exported policy bytes cannot exceed the finite generation ceiling."""
    for index in range(16):
        activation.cache["add"]("other" + str(index), b"x" * 1048576)
    with pytest.raises(OSError, match="policy bytes exceed capacity") as caught:
        activate(activation)
    assert caught.value.errno == errno.ENOSPC
    assert not activation.cache["loads"] and not activation.native["events"]


@pytest.mark.parametrize("descriptors", [["0"] * 257, ["0", "1", "2", "invalid"]])
def test_activate_payload_refuses_ambiguous_descriptor_inventory(activation, descriptors):
    """Unbounded or malformed setup descriptor state cannot enter the payload."""
    activation.descriptors = descriptors
    with pytest.raises(ValueError, match="descriptor inventory exceeds"):
        activate(activation)
    assert not activation.native["events"]


@pytest.mark.parametrize("failure", [None, errno.EBADF, errno.EIO])
def test_activate_payload_closes_setup_descriptors_before_entry(activation, monkeypatch, failure):
    """Every setup descriptor closes and only the already-closed proc iterator is tolerated."""
    activation.descriptors = ["0", "1", "2", "1000"]
    real_close = os.close
    events = []

    def close(descriptor):
        """Record explicit closure and model the proc iterator's already-closed descriptor."""
        if descriptor != 1000:
            return real_close(descriptor)
        events.append("close")
        if failure is not None:
            raise OSError(failure, "closure failed")

    def enter(*args, **kwargs):
        """Record public transition invocation after descriptors close."""
        events.append("enter")

    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(linux_bootstrap, "enter_payload", enter)
    if failure == errno.EIO:
        with pytest.raises(OSError, match="closure failed"):
            activate(activation)
        assert events == ["close"]
    else:
        activate(activation, namespace_level=1)
        assert events == ["close", "enter"]


def test_activate_payload_refuses_missing_installation_reservation(activation):
    """Invalid generation reservation blocks admission and irreversible source entry."""
    activation.limits.side_effect = ValueError("invalid generation reservation")
    with pytest.raises(ValueError, match="generation reservation"):
        activate(activation)
    assert activation.limits.call_args.args == (activation.reservation, "loop_sandbox")
    assert not activation.cache["loads"] and not activation.native["events"]


@pytest.mark.parametrize("limit", [0, -1, 16777217, True])
def test_load_cached_payload_refuses_invalid_reserved_byte_limit(cache, limit):
    """A trusted reservation cannot widen the namespace beyond finite installation bounds."""
    with pytest.raises(ValueError, match="Invalid policy cache input"):
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 31, aggregate_limit=limit
        )


def test_load_cached_payload_enforces_smaller_reserved_byte_limit(cache):
    """Namespace admission uses the actual reserved budget rather than the default ceiling."""
    with pytest.raises(OSError) as caught:
        linux_bootstrap.load_cached_payload(
            "payload", b"compiled", "loop_sandbox", 1000, cache["lock"], 31, aggregate_limit=8
        )
    assert caught.value.errno == errno.ENOSPC
    assert not cache["loads"]


@pytest.mark.parametrize(
    "failure", ["expired", "stale", "cwd", "activation", "late", "exec", "returned"]
)
def test_execute_payload_requires_activation_and_original_deadline(monkeypatch, failure):
    """Run source only after exact authority validation and successful activation."""
    request = Mock(
        deadline=20,
        cwd=Path("/work"),
        execution_source="printf safe",
        environment=(),
        aliases=(),
        file_creation_mask=0o027,
        attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7",
        generated_environment=(),
    )
    request.paths_are_current.return_value = failure != "stale"
    request.host_command_runtime_signature.return_value = "signature"
    request.shell_environment.return_value = {"PATH": "/usr/bin:/bin"}
    monkeypatch.setattr(Path, "cwd", lambda: Path("/other" if failure == "cwd" else "/work"))
    clock = Mock(side_effect=[30 if failure == "expired" else 10, 30 if failure == "late" else 10])
    monkeypatch.setattr(linux_bootstrap.time, "monotonic", clock)
    activation = Mock(
        side_effect=ValueError("activation refused") if failure == "activation" else None
    )
    monkeypatch.setattr(linux_bootstrap, "activate_payload", activation)
    replacement = Mock(side_effect=OSError("exec refused") if failure == "exec" else None)
    monkeypatch.setattr(linux_bootstrap.os, "execve", replacement)
    file_mask = Mock()
    monkeypatch.setattr(linux_bootstrap.os, "umask", file_mask)
    error, message = {
        "expired": (TimeoutError, "Payload execution deadline expired"),
        "stale": (ValueError, "Payload execution paths or cwd changed"),
        "cwd": (ValueError, "Payload execution paths or cwd changed"),
        "activation": (ValueError, "activation refused"),
        "late": (TimeoutError, "Payload execution deadline expired"),
        "exec": (OSError, "exec refused"),
        "returned": (RuntimeError, "Payload shell replacement unexpectedly returned"),
    }[failure]
    with pytest.raises(error, match=message):
        linux_bootstrap.execute_payload(
            request,
            "library",
            "bootstrap",
            "lock",
            b"compiled",
            "payload_profile",
            "generation",
            1000,
            reservation="budget",
        )
    if failure in ("expired", "stale", "cwd"):
        activation.assert_not_called()
    else:
        activation.assert_called_once_with(
            "library",
            "bootstrap",
            "lock",
            "payload_profile",
            b"compiled",
            "generation",
            1000,
            reservation="budget",
            namespace_level=1,
        )
    if failure in ("exec", "returned"):
        file_mask.assert_called_once_with(request.file_creation_mask)
        replacement.assert_called_once_with(
            "/bin/sh",
            [
                "/bin/sh",
                "-c",
                'printf "%s\\n" "$1" && exec /bin/sh -c "$2"',
                "loop-payload",
                "LOOP_PAYLOAD_" + request.attempt_id,
                "printf safe",
            ],
            {"PATH": "/usr/bin:/bin"},
        )
    else:
        file_mask.assert_not_called()
        replacement.assert_not_called()


def test_execute_payload_translates_only_bound_virtual_aliases(monkeypatch):
    """Preserve opaque quoting while selecting the request-local alias after host validation."""
    request = Mock(
        deadline=20,
        cwd=Path("/work"),
        execution_source="cat '/tmp/bound/r0/file'; printf '%s' /tmp/bound/r0-other",
        aliases=(("/workspace", Path("/tmp/bound/r0"), Path("/work")),),
        generated_environment=(),
        file_creation_mask=0o022,
        attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7",
    )
    request.shell_environment.return_value = {"PATH": "/usr/bin:/bin"}
    monkeypatch.setattr(Path, "cwd", lambda: request.cwd)
    monkeypatch.setattr(linux_bootstrap.time, "monotonic", lambda: 10)
    monkeypatch.setattr(linux_bootstrap, "activate_payload", Mock())
    monkeypatch.setattr(linux_bootstrap.os, "umask", Mock())
    replacement = Mock()
    monkeypatch.setattr(linux_bootstrap.os, "execve", replacement)
    with pytest.raises(RuntimeError, match="replacement unexpectedly returned"):
        linux_bootstrap.execute_payload(
            request,
            "library",
            "bootstrap",
            "lock",
            b"compiled",
            "payload",
            "generation",
            1000,
            reservation="budget",
        )
    assert (
        replacement.call_args.args[1][-1]
        == "cat '/.loop-private-paths/r0/file'; printf '%s' /tmp/bound/r0-other"
    )


def test_payload_environment_maps_generated_paths_to_namespace_scratch():
    """All generated paths stay inside the request's unique namespace scratch mount."""
    request = Mock(
        generated_environment=("TMPDIR", "XDG_CACHE_HOME", "COVERAGE_FILE"),
        shell_environment=Mock(
            return_value={
                "PATH": "/usr/bin:/bin",
                "TMPDIR": "/tmp/loop-scratch-one",
                "XDG_CACHE_HOME": "/tmp/loop-scratch-one/cache",
                "COVERAGE_FILE": "/tmp/loop-scratch-one/.coverage",
            }
        ),
    )

    result = linux_bootstrap._payload_environment(request)

    assert result == {
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(NATIVE_SCRATCH_PATH),
        "XDG_CACHE_HOME": str(NATIVE_SCRATCH_PATH / "cache"),
        "COVERAGE_FILE": str(NATIVE_SCRATCH_PATH / ".coverage"),
    }


def test_payload_environment_maps_direct_request_tmpdir():
    """Direct Linux requests also receive only their private namespace scratch spelling."""
    request = Mock(
        generated_environment=(),
        shell_environment=Mock(return_value={"PATH": "/usr/bin:/bin", "TMPDIR": "/tmp/direct"}),
    )

    assert linux_bootstrap._payload_environment(request) == {
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(NATIVE_SCRATCH_PATH),
    }


@pytest.mark.parametrize(
    ("environment", "generated", "message"),
    [
        ({"COVERAGE_FILE": "/tmp/scratch/.coverage"}, ("COVERAGE_FILE",), "no TMPDIR"),
        ({"TMPDIR": "/tmp/scratch"}, ("TMPDIR", "XDG_CACHE_HOME"), "incomplete"),
        ({"TMPDIR": "/tmp/scratch", "COVERAGE_FILE": "relative"}, ("COVERAGE_FILE",), "unsafe"),
        (
            {"TMPDIR": "/tmp/scratch", "COVERAGE_FILE": "/tmp/scratch/../outside"},
            ("COVERAGE_FILE",),
            "unsafe",
        ),
        (
            {"TMPDIR": "/tmp/scratch", "COVERAGE_FILE": "/tmp/other/.coverage"},
            ("COVERAGE_FILE",),
            "escaped",
        ),
    ],
)
def test_payload_environment_refuses_invalid_generated_paths(environment, generated, message):
    """Malformed generated scratch values cannot point outside the private bind."""
    request = Mock(
        generated_environment=generated, shell_environment=Mock(return_value=environment)
    )

    with pytest.raises(ValueError, match=message):
        linux_bootstrap._payload_environment(request)
