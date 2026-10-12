"""Verify setup intent, exact input review and reuse of fixed native lifecycle owners."""

import hashlib
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import setup as module
from loop.execution.sandbox.linux.host import HostFile


def bound(path, data=b"seed", mode=0o444):
    """Bind isolated collaborator authority without host discovery."""
    return SimpleNamespace(
        path=path,
        digest=hashlib.sha256(data).hexdigest(),
        identity=(1, 2, mode, 0, len(data), 1, 1),
        open_verified=lambda: nullcontext(10),
    )


@pytest.fixture
def terminal(monkeypatch, tmp_path):
    """Provide fixed ordinary terminal collaborators without launching system processes."""
    monkeypatch.setattr(
        module,
        "sys",
        SimpleNamespace(
            platform="linux",
            stdin=Mock(isatty=Mock(return_value=True)),
            stdout=Mock(isatty=Mock(return_value=True)),
        ),
    )
    monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    monkeypatch.setattr(module.os, "getgid", lambda: 1000)
    root = tmp_path / "native"
    root.mkdir()
    (root / "management.json").write_bytes(b"record")
    monkeypatch.setattr(module, "_DIRECTORY", root)
    monkeypatch.setattr(
        module, "runtime_catalog", Mock(return_value=({"helper": b"helper"}, {"helper"}, {}, b""))
    )
    monkeypatch.setattr(
        module, "bind_host_file", Mock(side_effect=lambda path, **options: bound(path))
    )
    original = (
        bound(Path("/original/management.json")),
        bound(Path("/original/loop_uninstall.py")),
    )
    monkeypatch.setattr(module, "uninstall_authority", Mock(return_value=original))
    process = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(module.subprocess, "run", process)
    return process


@pytest.mark.parametrize("action", ["install", "uninstall"])
@pytest.mark.parametrize("decision", [True, False, None])
def test_manage_setup_exact_review_terminal_and_cleanup(terminal, action, decision):
    """Consent binds loader/input/original identity and cleans only its ordinary input afterward."""
    reviewed = []

    def review(description):
        """Inspect the concise user summary without exposing bound implementation details."""
        reviewed.append(description)
        expected = "Linux sandbox installation" if action == "install" else "Linux sandbox removal"
        assert description.startswith(expected)
        assert "Administrator authentication may be requested next." in description
        assert "SHA256" not in description
        assert "Command:" not in description
        assert "/usr/bin/" not in description
        if action == "install":
            assert "No system packages are installed." in description
        else:
            assert "Sandboxed commands will stay unavailable" in description
        return decision

    assert module.manage_setup(action, review, main_app=True) is (decision is True)
    if decision is True:
        argv = terminal.call_args.args[0]
        assert argv[:7] == ["/usr/bin/sudo", "--", "/usr/bin/python3", "-I", "-S", "-B", "-c"]
        assert not Path(argv[8]).exists()
        assert terminal.call_args.kwargs["check"] is False
        assert terminal.call_args.kwargs["timeout"] == 120
        assert "stderr" not in terminal.call_args.kwargs
        assert "stdin" not in terminal.call_args.kwargs
    else:
        terminal.assert_not_called()
    assert module.bind_host_file.call_args_list[0].kwargs == {"system_elevation": True}
    assert module.bind_host_file.call_args_list[1].kwargs == {}
    assert len(reviewed) == 1


@pytest.mark.parametrize(
    "case", ["library", "stdin", "stdout", "platform", "action", "root", "group"]
)
def test_manage_setup_refuses_before_review(terminal, monkeypatch, case):
    """Library, headless, unsupported and privileged identities cannot request elevation."""
    main_app = case != "library"
    if case in {"stdin", "stdout"}:
        monkeypatch.setattr(getattr(module.sys, case), "isatty", lambda: False)
    if case == "platform":
        monkeypatch.setattr(module.sys, "platform", "darwin")
    if case in {"root", "group"}:
        monkeypatch.setattr(module.os, "getuid" if case == "root" else "getgid", lambda: 0)
    review = Mock()
    with pytest.raises((ValueError, OSError), match="terminal|unavailable|ordinary"):
        module.manage_setup("invalid" if case == "action" else "install", review, main_app=main_app)
    review.assert_not_called()
    terminal.assert_not_called()


@pytest.mark.parametrize("decision", [True, False, None])
def test_manage_uninstall_reviews_current_package_instead_of_retained_entry(terminal, decision):
    """Start authenticated uninstall code with the original record decision bound."""
    review = Mock(return_value=decision)
    assert module.manage_setup("uninstall", review, main_app=True) is (decision is True)
    description = review.call_args.args[0]
    assert description.startswith("Linux sandbox removal")
    assert "SHA256" not in description
    assert "Command:" not in description
    if decision is True:
        argv = terminal.call_args.args[0]
        assert argv[-5] == "uninstall"
        assert argv[-1] == module.uninstall_authority.return_value[0].digest
        assert argv[6] == "-c"
    else:
        terminal.assert_not_called()


@pytest.mark.parametrize("action", ["install", "uninstall"])
def test_manage_setup_reviews_retained_partial_generation(terminal, monkeypatch, action):
    """Review binds the original reserved generation for partial setup or removal."""
    record = module._DIRECTORY / "management.json"
    record.unlink()
    reservations = module._DIRECTORY / "reservations"
    reservations.mkdir()
    (reservations / ("native_" + "a" * 32)).write_bytes(b"capacity")
    monkeypatch.setattr(module, "protected_path", lambda path, **kwargs: path)
    monkeypatch.setattr(module, "generation_limits", Mock(return_value=(1, 1)))
    assert module.manage_setup(action, lambda description: True, main_app=True)
    assert terminal.call_args.args[0][-4] == "native_" + "a" * 32


def test_manage_setup_uses_fresh_generation_with_empty_capacity(terminal, monkeypatch):
    """An empty retained capacity directory does not supply stale generation identity."""
    (module._DIRECTORY / "management.json").unlink()
    (module._DIRECTORY / "reservations").mkdir()
    monkeypatch.setattr(module, "protected_path", lambda path, **kwargs: path)
    assert module.manage_setup("install", Mock(return_value=True), main_app=True)
    assert terminal.call_args.args[0][-4].startswith("native_")


@pytest.mark.parametrize("case", ["alias", "multiple"])
def test_manage_setup_refuses_ambiguous_partial_capacity(terminal, monkeypatch, case):
    """Review cannot proceed when pre-record capacity is aliased or ambiguous."""
    (module._DIRECTORY / "management.json").unlink()
    reservations = module._DIRECTORY / "reservations"
    reservations.mkdir()
    (reservations / ("native_" + "a" * 32)).write_bytes(b"capacity")
    if case == "alias":
        monkeypatch.setattr(module, "protected_path", lambda path, **kwargs: path.parent)
    else:
        monkeypatch.setattr(module, "protected_path", lambda path, **kwargs: path)
        (reservations / ("native_" + "b" * 32)).write_bytes(b"capacity")
    with pytest.raises(ValueError, match="aliases|Ambiguous"):
        module.manage_setup("uninstall", Mock(return_value=True), main_app=True)
    terminal.assert_not_called()


@pytest.mark.parametrize("case", ["changed", "timeout", "nonzero", "size"])
def test_manage_setup_changed_or_failed_handoff(terminal, monkeypatch, case):
    """Decision drift and bounded system failures never retry or claim setup success."""
    if case == "changed":
        original = module.uninstall_authority.return_value
        module.uninstall_authority.side_effect = [original, (bound(Path("/changed")), original[1])]
    elif case == "timeout":
        terminal.side_effect = subprocess.TimeoutExpired("sudo", 120)
    elif case == "nonzero":
        terminal.return_value.returncode = 1
    else:
        monkeypatch.setattr(module.json, "dumps", lambda *a, **k: "x" * (34 * 1024 * 1024 + 1))
    with pytest.raises((ValueError, OSError), match="changed|expired|failed|exceeds"):
        module.manage_setup("install", lambda text: True, main_app=True)
    assert terminal.call_count <= 1


@pytest.fixture
def administrator(tmp_path, monkeypatch):
    """Provide an isolated fixed-root lifecycle with public collaborators and real file effects."""
    root = tmp_path / "libexec/loop/native"
    (tmp_path / "libexec").mkdir()
    (tmp_path / "libexec").chmod(0o755)
    monkeypatch.setattr(module, "_DIRECTORY", root)
    monkeypatch.setattr(module.os, "getuid", lambda: 0)
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **k: (
            "0 0 4294967295" if str(path) == "/proc/self/uid_map" else read_text(path, *a, **k)
        ),
    )
    monkeypatch.setattr(module, "protected_path", lambda path, **k: path)
    monkeypatch.setattr(module, "validate_generation_catalog", Mock())
    monkeypatch.setattr(module, "discover_tool", lambda name: bound(Path("/bwrap"), b"helper"))

    def bind(path):
        """Read real isolated bytes through a pinned descriptor for scaffold validation."""
        data = path.read_bytes()
        return SimpleNamespace(
            path=path,
            digest=hashlib.sha256(data).hexdigest(),
            identity=(1, 2, path.stat().st_mode, 0, len(data), 1, 1),
            open_verified=lambda: path.open("rb"),
        )

    # HostFile descriptors are integers; keep only their public contract in this collaborator.
    from contextlib import contextmanager

    def bind_integer(path):
        """Expose the normal integer descriptor contract for real isolated scaffold files."""
        value = bind(path)

        @contextmanager
        def opened():
            """Close each isolated file descriptor after use."""
            with path.open("rb") as stream:
                yield stream.fileno()

        value.open_verified = opened
        return value

    monkeypatch.setattr(module, "bind_host_file", bind_integer)

    def reserve(directory, lock, generation):
        """Publish a disposable reservation through the mocked public reservation boundary."""
        path = directory / generation
        path.write_bytes(b"reservation")
        return bind_integer(path)

    monkeypatch.setattr(module, "reserve_generation", Mock(side_effect=reserve))

    def stage(directory, contents, executable, lock):
        """Return bound requested slice through the public immutable staging boundary."""
        return {name: HostFile(directory / name, (), "") for name in contents}

    monkeypatch.setattr(module, "stage_generation_files", Mock(side_effect=stage))
    monkeypatch.setattr(module, "install_generation", Mock())
    monkeypatch.setattr(module, "verify_uninstalled", Mock())
    monkeypatch.setattr(
        module,
        "uninstall_authority",
        Mock(return_value=(bound(root / "management.json"), bound(root / "loop_uninstall.py"))),
    )
    return root


def test_install_reviewed_fixed_scaffolding_and_repeat(administrator):
    """Fresh setup composes the existing owners and only exact shared scaffolding can be reused."""
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    assert (administrator / "management.lock").read_bytes() == b"Loop native management\n"
    assert module.install_generation.call_count == 1
    assert module.install_generation.call_args.args[0].uid == 1000
    # Model completed removal while retaining the protected shared scaffolding.
    (administrator / "reservations/generation").unlink()
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    assert module.install_generation.call_count == 2


def test_install_reviewed_cleans_unrecorded_partial_generation(administrator, monkeypatch):
    """An existing reviewed reservation triggers cleanup before a fresh reservation."""
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    partial = administrator / "reservations/generation"
    monkeypatch.setattr(module, "generation_limits", Mock(return_value=(1, 1)))

    def cleanup(*args):
        """Model completed checked cleanup of the exact retained reservation."""
        assert args[0].generation == "generation"
        assert args[0].reservation.path == partial
        partial.unlink()

    owner = Mock(side_effect=cleanup)
    monkeypatch.setattr(module, "clean_unrecorded_installation", owner)
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    owner.assert_called_once()
    assert module.install_generation.call_count == 2


def test_install_reviewed_refuses_a_different_retained_reservation(administrator):
    """Fresh setup cannot silently replace another generation's held capacity."""
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    with pytest.raises(ValueError, match="reservation differs from reviewed generation"):
        module.install_reviewed("install", "other", 1000, 1000, "", {"helper": b"helper"})
    assert module.install_generation.call_count == 1


def test_install_reviewed_cleans_recorded_partial_state(administrator, monkeypatch):
    """Install checks the reviewed original record and removes it before staging."""
    administrator.mkdir(parents=True)
    record = administrator / "management.json"
    entry = administrator / "loop_uninstall.py"
    record.write_bytes(b"record")
    entry.write_bytes(b"entry")
    record.chmod(0o444)
    entry.chmod(0o555)
    authority = module.bind_host_file(record)
    exported = module.bind_host_file(entry)
    monkeypatch.setattr(module, "uninstall_authority", Mock(return_value=(authority, exported)))

    def uninstall(path, deadline, **kwargs):
        """Model completion of the checked native teardown collaborator."""
        assert path == record
        record.unlink()
        entry.unlink()

    monkeypatch.setattr(module, "uninstall_generation", Mock(side_effect=uninstall))
    module.install_reviewed(
        "install", "generation", 1000, 1000, authority.digest, {"helper": b"helper"}
    )
    module.uninstall_generation.assert_called_once()
    module.install_generation.assert_called_once()


def test_install_reviewed_refuses_changed_recorded_partial_authority(administrator, monkeypatch):
    """Install refuses a saved record that differs from the reviewed removal decision."""
    administrator.mkdir(parents=True)
    record = administrator / "management.json"
    record.write_bytes(b"record")
    record.chmod(0o444)
    monkeypatch.setattr(
        module,
        "uninstall_authority",
        Mock(return_value=(module.bind_host_file(record), module.bind_host_file(record))),
    )
    owner = Mock()
    monkeypatch.setattr(module, "uninstall_generation", owner)
    with pytest.raises(ValueError, match="Reviewed partial installation authority changed"):
        module.install_reviewed("install", "generation", 1000, 1000, "wrong", {"helper": b"helper"})
    owner.assert_not_called()


def test_uninstall_retry_verifies_absence_after_record_removal(administrator, monkeypatch):
    """A repeated uninstall proves final absence when its record is already gone."""
    verified = Mock()
    monkeypatch.setattr(module, "verify_uninstalled", verified)
    monkeypatch.setattr(module.time, "monotonic", lambda: 10)
    module.install_reviewed("uninstall", "ignored", 1000, 1000, "", {})
    verified.assert_called_once_with(administrator, 15)
    module.uninstall_authority.assert_not_called()
    with pytest.raises(ValueError, match="authority disappeared"):
        module.install_reviewed("uninstall", "ignored", 1000, 1000, "digest", {})
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    (administrator / "reservations/generation").unlink()
    module.install_reviewed("uninstall", "ignored", 1000, 1000, "", {})
    assert verified.call_count == 3


def test_uninstall_cleans_reviewed_unrecorded_installation(administrator, monkeypatch):
    """Uninstall removes an authenticated reserved stage even before record publication."""
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    partial = administrator / "reservations/generation"
    monkeypatch.setattr(module, "generation_limits", Mock(return_value=(1, 1)))

    def cleanup(config, *args):
        """Model checked removal of the exact reviewed generation."""
        assert config.generation == "generation"
        assert config.reservation.path == partial
        partial.unlink()

    owner = Mock(side_effect=cleanup)
    monkeypatch.setattr(module, "clean_unrecorded_installation", owner)
    module.install_reviewed("uninstall", "generation", 1000, 1000, "", {"helper": b"helper"})
    owner.assert_called_once()
    module.verify_uninstalled.assert_called()
    assert not partial.exists()


@pytest.mark.parametrize("case", ["alias", "multiple", "generation", "helper"])
def test_uninstall_refuses_changed_unrecorded_reservation(administrator, monkeypatch, case):
    """A pre-record uninstall refuses changed capacity, identity, or reviewed helper bytes."""
    module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    reservations = administrator / "reservations"
    monkeypatch.setattr(module, "generation_limits", Mock(return_value=(1, 1)))
    if case == "alias":
        monkeypatch.setattr(
            module,
            "protected_path",
            lambda path, **kwargs: path.parent if path == reservations else path,
        )
    elif case == "multiple":
        (reservations / "another").write_bytes(b"capacity")
    elif case == "helper":
        monkeypatch.setattr(module, "discover_tool", Mock(return_value=bound(Path("/changed"))))
    with pytest.raises(ValueError, match="aliases|Ambiguous|generation changed|helper changed"):
        module.install_reviewed(
            "uninstall",
            "other" if case == "generation" else "generation",
            1000,
            1000,
            "",
            {"helper": b"helper"},
        )


@pytest.mark.parametrize(
    "case",
    [
        "uid",
        "mapping",
        "action",
        "user",
        "group",
        "helper",
        "record",
        "directory",
        "inventory",
        "lock",
        "writable",
        "reservation",
    ],
)
def test_install_reviewed_refuses_changed_fixed_authority(administrator, monkeypatch, case):
    """Root setup refuses privilege, input and scaffold drift without weakening lifecycle checks."""
    if case == "uid":
        monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    elif case == "mapping":
        monkeypatch.setattr(Path, "read_text", lambda *a, **k: "0 1000 1")
    elif case in {"directory", "reservation"}:
        target = administrator if case == "directory" else administrator / "reservations"
        monkeypatch.setattr(
            module, "protected_path", lambda path, **k: Path("/alias") if path == target else path
        )
    elif case in {"inventory", "lock", "writable"}:
        administrator.mkdir(parents=True)
        if case == "inventory":
            (administrator / "unknown").write_bytes(b"x")
        else:
            path = administrator / "management.lock"
            path.write_bytes(b"changed" if case == "lock" else b"Loop native management\n")
            path.chmod(0o444 if case == "lock" else 0o644)
    with pytest.raises(
        (PermissionError, ValueError),
        match="administrator|identity|helper|replacement|alias|partial|private files|lock",
    ):
        module.install_reviewed(
            "bad" if case == "action" else "install",
            "generation",
            0 if case == "user" else 1000,
            0 if case == "group" else 1000,
            "digest" if case == "record" else "",
            {"helper": b"drift" if case == "helper" else b"helper"},
        )
    module.install_generation.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
def test_install_reviewed_uninstall_uses_exact_original_record(administrator, monkeypatch, changed):
    """Authenticated current owner removes only the freshly reviewed original generation."""
    administrator.mkdir(parents=True)
    (administrator / "management.json").write_bytes(b"original")
    retire = Mock()
    monkeypatch.setattr(module, "uninstall_generation", retire)
    monkeypatch.setattr(module.time, "monotonic", lambda: 10)
    digest = module.uninstall_authority.return_value[0].digest
    if changed:
        with pytest.raises(ValueError, match="uninstall authority changed"):
            module.install_reviewed("uninstall", "ignored", 1000, 1000, "changed", {})
        retire.assert_not_called()
    else:
        module.install_reviewed("uninstall", "ignored", 1000, 1000, digest, {})
        retire.assert_called_once_with(administrator / "management.json", 15)
    module.reserve_generation.assert_not_called()


def test_install_reviewed_restrictive_umask_and_parent_alias(administrator, monkeypatch):
    """Grant fresh files fixed reads and reject aliased parents before creation."""
    previous = os.umask(0o077)
    try:
        module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    finally:
        os.umask(previous)
    assert administrator.stat().st_mode & 0o777 == 0o755
    assert (administrator / "management.lock").stat().st_mode & 0o777 == 0o444
    monkeypatch.setattr(module, "protected_path", lambda path, **k: Path("/alias"))
    with pytest.raises(ValueError, match="parent aliases"):
        module.install_reviewed("install", "other", 1000, 1000, "", {"helper": b"helper"})


def test_install_reviewed_refuses_inaccessible_existing_runtime(administrator):
    """Preexisting private directory permissions cannot produce inaccessible receiver authority."""
    administrator.mkdir(parents=True, mode=0o700)
    with pytest.raises(ValueError, match="read/traversal"):
        module.install_reviewed("install", "generation", 1000, 1000, "", {"helper": b"helper"})
    assert administrator.stat().st_mode & 0o777 == 0o700
    module.reserve_generation.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"errors": ["live state", "live state", "closure refused"]},
        {"errors": []},
        {"errors": ["control\x1b"]},
        {"errors": [1]},
        {"errors": ["x" * 161]},
        {"errors": ["x"] * 17},
        {"errors": "wrong"},
        {"extra": []},
        [],
        None,
        b"not json",
        b"x" * 4097,
        b"\xff",
    ],
)
def test_manage_setup_transports_only_bounded_owner_diagnostics(terminal, payload):
    """Preserve failure owners and generalize malformed or excessive output."""

    def fail(argv, **options):
        """Write the privileged result to its inherited stdout descriptor."""
        options["stdout"].write(
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )
        return SimpleNamespace(returncode=1)

    terminal.side_effect = fail
    if isinstance(payload, dict) and payload.get("errors") == [
        "live state",
        "live state",
        "closure refused",
    ]:
        with pytest.raises(ExceptionGroup) as raised:
            module.manage_setup("install", lambda text: True, main_app=True)
        assert [str(error) for error in raised.value.exceptions] == [
            "live state",
            "closure refused",
            "Inspect /sandbox status; resolve the refusal, then retry the action.",
        ]
    else:
        with pytest.raises(OSError, match="Administrator setup failed"):
            module.manage_setup("install", lambda text: True, main_app=True)
