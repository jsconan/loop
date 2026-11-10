"""Tests for protected administrator policy loading and source identity."""

import errno
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from loop.permissions import managed
from loop.permissions.managed import ManagedPolicyError, ManagedPolicyStore

_IDENTITY = "a" * 32


def installed_policy(tmp_path: Path) -> tuple[ManagedPolicyStore, Path]:
    """Create a protected installation under an owned unit-test trust anchor."""
    directory = tmp_path / "etc" / "loop"
    directory.mkdir(parents=True)
    directory.parent.chmod(0o755)
    directory.chmod(0o755)
    store = ManagedPolicyStore(directory, trust_anchor=tmp_path, owner_uid=os.getuid())
    (directory.parent / "loop.managed.expect.json").write_text(
        json.dumps({"version": 1, "identity": _IDENTITY, "minimum_revision": 1})
    )
    (directory / "managed-policy.json").write_text(
        json.dumps({"version": 1, "identity": _IDENTITY, "revision": 1})
    )
    (directory.parent / "loop.managed.expect.json").chmod(0o644)
    (directory / "managed-policy.json").chmod(0o644)
    return store, directory


def test_absent_installation_is_unmanaged_but_missing_expected_file_is_invalid(tmp_path):
    """A missing installation is optional; an existing installation needs both files."""
    directory = tmp_path / "etc" / "loop"
    store = ManagedPolicyStore(directory, trust_anchor=tmp_path, owner_uid=os.getuid())
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc").chmod(0o755)
    assert store.directory == directory
    assert store.may_be_installed() is False
    assert store.load() is None
    directory.mkdir()
    directory.chmod(0o755)
    assert store.may_be_installed() is True
    with pytest.raises(ManagedPolicyError, match="expectation is missing"):
        store.load()


def test_expectation_survives_policy_directory_disappearance(tmp_path):
    """A fresh reader denies a missing policy directory while its expectation remains."""
    store, directory = installed_policy(tmp_path)
    (directory / "managed-policy.json").unlink()
    directory.rmdir()
    assert store.may_be_installed() is True
    with pytest.raises(ManagedPolicyError, match="Expected administrator policy directory"):
        store.load()


def test_unsupported_platform_has_no_managed_source(monkeypatch):
    """An unsupported platform reports managed enforcement unavailable."""
    monkeypatch.setattr(managed.sys, "platform", "unsupported")
    store = ManagedPolicyStore()
    assert store.directory is None
    assert store.may_be_installed() is False
    assert store.load() is None


def test_default_source_absence_and_presence_are_distinct(tmp_path, monkeypatch):
    """The fixed platform source avoids inaccessible ancestry when absent."""
    directory = tmp_path / "etc" / "loop"
    directory.parent.mkdir()
    directory.parent.chmod(0o755)
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    assert store.may_be_installed() is False
    assert store.load() is None
    _, installed = installed_policy(tmp_path)
    assert installed == store.directory
    assert store.may_be_installed() is True
    assert store.load().policy.identity == _IDENTITY


def test_hidden_ancestor_metadata_never_proves_optional_absence(tmp_path, monkeypatch):
    """Denied ancestry cannot make an existing installation appear unmanaged."""
    directory = tmp_path / "etc" / "loop"
    directory.parent.mkdir()
    directory.parent.chmod(0o755)
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    original = Path.lstat

    def hidden_ancestor(path, *args, **kwargs):
        """Simulate an outer sandbox that hides ancestry and installed entries."""
        if path == directory.parent:
            raise PermissionError("ancestor metadata hidden")
        if path in (directory, store.expectation_path):
            raise FileNotFoundError("entry hidden")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", hidden_ancestor)
    installed_policy(tmp_path)
    assert store.may_be_installed() is True
    assert store.load().policy.identity == _IDENTITY


def test_hidden_ancestor_metadata_allows_verified_absence(tmp_path, monkeypatch):
    """A searchable protected parent can prove both fixed entries absent."""
    directory = tmp_path / "etc" / "loop"
    directory.parent.mkdir()
    directory.parent.chmod(0o755)
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    original = Path.lstat

    def hidden_ancestor(path, *args, **kwargs):
        """Hide pathname metadata without hiding protected descriptor traversal."""
        if path == directory.parent:
            raise PermissionError("ancestor metadata hidden")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", hidden_ancestor)
    assert store.may_be_installed() is False
    assert store.load() is None


def test_uninspectable_default_source_is_never_treated_as_absent(tmp_path, monkeypatch):
    """A source presence check that fails still requires protected validation."""
    directory = tmp_path / "etc" / "loop"
    directory.parent.mkdir()
    directory.parent.chmod(0o755)
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    original_lstat = Path.lstat
    original_stat = managed.os.stat

    def denied_lstat(path, *args, **kwargs):
        """Simulate native denial of the fixed source metadata probe."""
        if path == directory:
            raise PermissionError("denied")
        return original_lstat(path, *args, **kwargs)

    def denied_stat(path, *args, **kwargs):
        """Deny the descriptor proof of the same fixed source."""
        if path == "loop.managed.expect.json" and "dir_fd" in kwargs:
            raise PermissionError("denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    monkeypatch.setattr(managed.os, "stat", denied_stat)
    assert store.may_be_installed() is True
    with pytest.raises(ManagedPolicyError):
        store.load()


def test_failed_source_metadata_probe_cannot_prove_absence(tmp_path, monkeypatch):
    """An unexpected source metadata error requires protected validation."""
    directory = tmp_path / "etc" / "loop"
    directory.parent.mkdir()
    directory.parent.chmod(0o755)
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    original = Path.lstat

    def failed_lstat(path, *args, **kwargs):
        """Report an I/O failure for one fixed protected source entry."""
        if path == directory:
            raise OSError(errno.EIO, "source metadata unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", failed_lstat)
    assert store.may_be_installed() is True


def test_parent_disappearance_before_absence_proof_is_not_optional(tmp_path, monkeypatch):
    """A vanished protected parent cannot establish an unmanaged installation."""
    parent = tmp_path / "etc"
    parent.mkdir()
    parent.chmod(0o755)
    directory = parent / "loop"
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    original = Path.lstat

    def disappearing_parent(path, *args, **kwargs):
        """Remove the parent after both source names initially appear absent."""
        if path == directory:
            with pytest.raises(FileNotFoundError):
                original(path, *args, **kwargs)
            parent.rename(tmp_path / "retired")
            raise FileNotFoundError("source absent before parent removal")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", disappearing_parent)
    assert store.may_be_installed() is True


@pytest.mark.parametrize("replacement", [False, True])
def test_parent_change_during_absence_proof_is_not_optional(tmp_path, monkeypatch, replacement):
    """A removed or replaced protected parent cannot prove absence at the fixed path."""
    parent = tmp_path / "etc"
    parent.mkdir()
    parent.chmod(0o755)
    directory = parent / "loop"
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    original = managed.os.stat

    def switching_stat(path, *args, **kwargs):
        """Switch the parent after the second descriptor-relative leaf lookup."""
        if path == "loop" and "dir_fd" in kwargs:
            with pytest.raises(FileNotFoundError):
                original(path, *args, **kwargs)
            parent.rename(tmp_path / "retired")
            if replacement:
                parent.mkdir()
                parent.chmod(0o755)
            raise FileNotFoundError("source absent before parent switch")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(managed.os, "stat", switching_stat)
    assert store.may_be_installed() is True


def test_default_source_rejects_unsafe_ancestor_even_when_both_entries_are_absent(
    tmp_path, monkeypatch
):
    """A symlinked installation parent cannot hide an expected source as unmanaged."""
    real = tmp_path / "real"
    real.mkdir()
    real.chmod(0o755)
    (tmp_path / "etc").symlink_to(real)
    directory = tmp_path / "etc" / "loop"
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    store = ManagedPolicyStore(trust_anchor=tmp_path, owner_uid=os.getuid())
    assert store.may_be_installed() is True
    with pytest.raises(ManagedPolicyError):
        store.load()


@pytest.mark.parametrize("unsafe", ["owner", "mode"])
def test_default_source_rejects_untrusted_absent_parent(tmp_path, monkeypatch, unsafe):
    """Untrusted ownership or write access cannot masquerade as optional absence."""
    parent = tmp_path / "etc"
    parent.mkdir()
    parent.chmod(0o755)
    directory = parent / "loop"
    monkeypatch.setitem(managed._SYSTEM_DIRECTORIES, sys.platform, directory)
    if unsafe == "mode":
        parent.chmod(0o777)
    store = ManagedPolicyStore(
        trust_anchor=tmp_path, owner_uid=-1 if unsafe == "owner" else os.getuid()
    )
    assert store.may_be_installed() is True
    with pytest.raises(ManagedPolicyError):
        store.load()


def test_valid_policy_binds_revision_rules_and_exact_source_bytes(tmp_path):
    """A valid protected installation yields a stable snapshot that changes on replacement."""
    store, directory = installed_policy(tmp_path)
    policy = directory / "managed-policy.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "identity": _IDENTITY,
                "revision": 2,
                "require_sandbox": True,
                "forbid_host": True,
                "rules": [
                    {"id": "deny-network", "decision": "deny", "action": "network.request"},
                    {"id": "human-write", "decision": "ask", "action": "filesystem.create"},
                ],
            }
        )
    )
    first = store.load()
    assert first.policy.revision == 2
    assert first.policy.require_sandbox is True
    assert first.policy.forbid_host is True
    assert len(first.policy.rules) == 2
    assert store.load().identity == first.identity
    replacement = directory / "replacement"
    replacement.write_bytes(policy.read_bytes())
    replacement.chmod(0o644)
    replacement.replace(policy)
    assert store.load().identity != first.identity


@pytest.mark.parametrize(
    ("expectation", "policy", "message"),
    [
        ('{"version": 1, "identity": "a", "minimum_revision": 1}', None, "cannot be verified"),
        ('{"version": 1, "identity": "a", "identity": "b"}', None, "Duplicate"),
        (None, '{"version": 1, "identity": "' + "b" * 32 + '", "revision": 1}', "stale"),
        (None, '{"version": 1, "identity": "a"}', "cannot be verified"),
        (
            None,
            '{"version": 1, "identity": "' + _IDENTITY + '", "revision": 0}',
            "cannot be verified",
        ),
        (
            '{"version": 1, "identity": "' + _IDENTITY + '", "minimum_revision": 2}',
            None,
            "stale",
        ),
        (
            None,
            '{"version": 1, "identity": "'
            + _IDENTITY
            + '", "revision": 1, "rules": [{"decision": "deny"}]}',
            "explicit stable identifiers",
        ),
        (
            None,
            '{"version": 1, "identity": "'
            + _IDENTITY
            + '", "revision": 1, "rules": [{"id": "x", "decision": "allow"}]}',
            "cannot be verified",
        ),
        (
            None,
            '{"version": 1, "identity": "'
            + _IDENTITY
            + '", "revision": 1, "rules": [{"id": "x", "decision": "deny"}, '
            '{"id": "x", "decision": "ask"}]}',
            "cannot be verified",
        ),
        (
            None,
            '{"version": 1, "identity": "'
            + _IDENTITY
            + '", "revision": 1, "rules": [{"id": "unsafe\\nsource", '
            '"decision": "deny"}]}',
            "cannot be verified",
        ),
        (
            None,
            '{"version": 1, "identity": "'
            + _IDENTITY
            + '", "revision": 1, "rules": [{"id": "x", "decision": "deny", '
            '"source": {"preset_id": "x", "revision": "1", '
            '"content_hash": "hash", "rule_id": "x"}}]}',
            "cannot be verified",
        ),
    ],
)
def test_invalid_installation_is_rejected(tmp_path, expectation, policy, message):
    """Malformed, ambiguous, stale, or permissive administrator data fails closed."""
    store, directory = installed_policy(tmp_path)
    if expectation is not None:
        (directory.parent / "loop.managed.expect.json").write_text(expectation)
    if policy is not None:
        (directory / "managed-policy.json").write_text(policy)
    with pytest.raises(ManagedPolicyError, match=message):
        store.load()


@pytest.mark.parametrize(
    "unsafe",
    ["ancestor_mode", "file_mode", "file_link", "hardlink", "owner", "acl", "size", "type"],
)
def test_unsafe_installation_authority_is_rejected(tmp_path, monkeypatch, unsafe):
    """Writable ancestry, links, ACLs, ownership, and non-files cannot grant authority."""
    store, directory = installed_policy(tmp_path)
    policy = directory / "managed-policy.json"
    if unsafe == "ancestor_mode":
        (tmp_path / "etc").chmod(0o777)
    elif unsafe == "file_mode":
        policy.chmod(0o666)
    elif unsafe == "file_link":
        policy.unlink()
        policy.symlink_to(directory.parent / "loop.managed.expect.json")
    elif unsafe == "hardlink":
        os.link(policy, directory / "duplicate")
    elif unsafe == "owner":
        store = ManagedPolicyStore(directory, trust_anchor=tmp_path, owner_uid=-1)
    elif unsafe == "acl":
        monkeypatch.setattr(managed, "has_extended_access_acl", lambda _fd: True)
    elif unsafe == "size":
        policy.write_bytes(b" " * 65_537)
    elif unsafe == "type":
        policy.unlink()
        policy.mkdir()
    with pytest.raises(ManagedPolicyError):
        store.load()


@pytest.mark.parametrize("path", [Path("relative"), Path("/unrelated"), Path("/root/../escape")])
def test_invalid_location_is_rejected(tmp_path, path):
    """A trusted source cannot be selected by a relative or escaping path."""
    store = ManagedPolicyStore(path, trust_anchor=tmp_path, owner_uid=os.getuid())
    assert store.may_be_installed() is True
    with pytest.raises(ManagedPolicyError):
        store.load()


def test_missing_intermediate_directory_is_invalid(tmp_path):
    """A partial installation path cannot be mistaken for an unmanaged source."""
    store = ManagedPolicyStore(
        tmp_path / "etc" / "loop", trust_anchor=tmp_path, owner_uid=os.getuid()
    )
    assert store.may_be_installed() is True
    with pytest.raises(ManagedPolicyError, match="expectation parent is unavailable"):
        store.load()


def test_missing_nonfinal_ancestor_is_invalid(tmp_path):
    """A missing intermediate protected ancestor fails before optional absence."""
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc").chmod(0o755)
    store = ManagedPolicyStore(
        tmp_path / "etc" / "nested" / "deeper" / "loop",
        trust_anchor=tmp_path,
        owner_uid=os.getuid(),
    )
    with pytest.raises(ManagedPolicyError, match="path is incomplete"):
        store.load()


def test_file_replacement_during_read_is_rejected(tmp_path, monkeypatch):
    """Descriptor binding detects replacement before policy bytes are admitted."""
    store, directory = installed_policy(tmp_path)
    original = managed.os.stat

    def replacing_stat(path, *args, **kwargs):
        """Replace the policy just before the final name identity check."""
        if path == "managed-policy.json" and "dir_fd" in kwargs:
            replacement = directory / "replacement"
            replacement.write_bytes((directory / "managed-policy.json").read_bytes())
            replacement.replace(directory / "managed-policy.json")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(managed.os, "stat", replacing_stat)
    with pytest.raises(ManagedPolicyError, match="changed during loading"):
        store.load()


def test_in_place_policy_write_during_final_walk_is_rejected(tmp_path, monkeypatch):
    """An in-place edit after initial file validation cannot enter a loaded snapshot."""
    store, directory = installed_policy(tmp_path)
    original = managed.os.open
    visits = 0

    def editing_open(path, *args, **kwargs):
        """Edit protected policy bytes before the final directory walk."""
        nonlocal visits
        if path == "loop" and "dir_fd" in kwargs:
            visits += 1
            if visits == 2:
                policy = directory / "managed-policy.json"
                content = json.loads(policy.read_text())
                content["revision"] = 2
                policy.write_text(json.dumps(content))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(managed.os, "open", editing_open)
    with pytest.raises(ManagedPolicyError, match="policy file changed during loading"):
        store.load()


def test_expectation_replacement_during_final_walk_is_rejected(tmp_path, monkeypatch):
    """A changed protected expectation cannot authorize the earlier policy bytes."""
    store, directory = installed_policy(tmp_path)
    original = managed.os.open
    visits = 0

    def editing_open(path, *args, **kwargs):
        """Advance the minimum revision before the final expectation walk."""
        nonlocal visits
        if path == "etc" and "dir_fd" in kwargs:
            visits += 1
            if visits == 3:
                expectation = directory.parent / "loop.managed.expect.json"
                content = json.loads(expectation.read_text())
                content["minimum_revision"] = 2
                expectation.write_text(json.dumps(content))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(managed.os, "open", editing_open)
    with pytest.raises(ManagedPolicyError, match="expectation changed during loading"):
        store.load()


@pytest.mark.parametrize("replacement", [False, True])
def test_directory_disappearance_or_replacement_during_read_is_rejected(
    tmp_path, monkeypatch, replacement
):
    """A name switch after opening files cannot authenticate another directory."""
    store, directory = installed_policy(tmp_path)
    original = managed.os.open
    visits = 0

    def switching_open(path, *args, **kwargs):
        """Switch the installation name before the second protected walk."""
        nonlocal visits
        if path == "loop" and "dir_fd" in kwargs:
            visits += 1
            if visits == 2:
                directory.rename(directory.with_name("retired"))
                if replacement:
                    directory.mkdir()
                    directory.chmod(0o755)
                    shutil.copyfile(
                        directory.with_name("retired") / "managed-policy.json",
                        directory / "managed-policy.json",
                    )
                    (directory / "managed-policy.json").chmod(0o644)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(managed.os, "open", switching_open)
    expected = "directory changed" if replacement else "directory disappeared"
    with pytest.raises(ManagedPolicyError, match=expected):
        store.load()


def test_expectation_parent_disappearance_during_load_is_rejected(tmp_path, monkeypatch):
    """A removed expectation parent cannot authenticate a loaded policy."""
    store, directory = installed_policy(tmp_path)
    original = managed.os.open
    visits = 0

    def removing_open(path, *args, **kwargs):
        """Remove the expectation parent before the final protected walk."""
        nonlocal visits
        if path == "etc" and "dir_fd" in kwargs:
            visits += 1
            if visits == 3:
                directory.parent.rename(directory.parent.with_name("retired"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(managed.os, "open", removing_open)
    with pytest.raises(ManagedPolicyError, match="expectation parent disappeared"):
        store.load()
