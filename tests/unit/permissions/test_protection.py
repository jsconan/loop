"""Verify the permission-owned protected workspace path specification."""

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.permissions.protection import (
    inspect_hardlink_aliases,
    is_environment_value,
    is_hard_resource,
    is_protected_external_read,
    protected_workspace_paths,
    validate_hard_denied_paths,
    write_file_state,
)


def test_write_file_state_rejects_an_aliased_parent(tmp_path):
    """An exact scope cannot follow a directory alias even if the leaf is absent."""
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual)
    with pytest.raises(ValueError, match="existing real parent"):
        write_file_state(alias / ".coverage")


def test_protected_paths_bind_default_and_configured_names():
    """Native policy and structured permissions share active control path names."""
    paths = protected_workspace_paths(("POLICY+.md", "AGENTS.md"))

    assert paths.instruction_names == ("AGENTS.md", "SKILL.md", "POLICY+.md")
    assert paths.directories == (".git", ".loop", ".agents/skills")
    assert paths.files == (".gitignore", ".agentignore")


@pytest.mark.parametrize(
    ("relative", "protected"),
    [
        (".agents/skills/demo/asset.txt", True),
        ("nested/.agents/skills/demo/asset.txt", True),
        ("nested/.git/config", True),
        ("nested/.loop/policy", True),
        ("nested/.agents/skills-other/asset.txt", False),
        ("nested/project.git/config", False),
    ],
)
def test_protected_directories_apply_at_nested_instruction_scopes(relative, protected):
    """Nested skill and control roots receive the same classification as root scopes."""
    paths = protected_workspace_paths()

    assert paths.protects_directory(Path(relative)) is protected


@pytest.mark.parametrize("name", ("", ".", "..", "a/b", "a\\b", "a\x00b"))
def test_protected_paths_reject_non_basename_instructions(name):
    """An instruction configuration cannot inject a path or native policy syntax."""
    with pytest.raises(ValueError, match="plain filenames"):
        protected_workspace_paths((name,))


@pytest.mark.parametrize(
    ("name", "denied"),
    [
        (".env", True),
        (".env.local", True),
        (".ENV.PRODUCTION", True),
        (".env.example", False),
        (".env.production.example", False),
        (".env.sample", False),
        (".env.template", False),
        (".env.example.local", True),
        ("environment", False),
    ],
)
def test_environment_values_and_public_templates_have_one_classification(name, denied):
    """Final template suffixes remain public unless trusted policy designates them separately."""
    assert is_environment_value(name) is denied


@pytest.mark.parametrize(
    "path",
    [
        "relative",
        "/a/../b",
        "/a//b",
        "/a/",
        "/a/*",
        "/a/?",
        "/a/[x]",
        "/a/{x}",
        "/a\x00b",
        "/a\nb",
        "/a\\b",
    ],
)
def test_hard_designations_reject_unrepresentable_paths(path):
    """Hard rules never accept glob syntax or ambiguous path normalization."""
    with pytest.raises(ValueError, match="normalized absolute exact paths"):
        validate_hard_denied_paths((path,))


def test_hard_designations_are_bounded_and_do_not_require_existing_files():
    """Future exact files/directories are valid but an oversized policy is refused."""
    assert validate_hard_denied_paths(("/future/secret",)) == ("/future/secret",)
    with pytest.raises(ValueError, match="at most 256"):
        validate_hard_denied_paths(tuple(f"/secret/{number}" for number in range(257)))


def test_hard_names_are_relative_to_workspace_and_scratch(tmp_path):
    """Protected ancestors leave root contents usable and only agent data exempts private names."""
    workspace = tmp_path / "Library/workspace"
    scratch = tmp_path / "Library/scratch"
    assert not is_hard_resource(workspace / "ordinary", workspace=workspace, scratch=scratch)
    assert is_hard_resource(workspace / ".ssh/key", workspace=workspace, scratch=scratch)
    assert is_hard_resource(tmp_path / ".config/key", workspace=None, scratch=scratch)
    assert not is_hard_resource(scratch / "ordinary", workspace=workspace, scratch=scratch)
    assert is_hard_resource(scratch / ".env.local", workspace=workspace, scratch=scratch)
    assert not is_hard_resource(
        scratch / ".env.local", workspace=workspace, scratch=scratch, scratch_data=True
    )


def test_external_private_components_retain_their_read_boundary():
    """External private ancestry remains denied without treating every environment as private."""
    assert is_protected_external_read(Path("/owned/.SSH/key"))
    assert not is_protected_external_read(Path("/owned/project/public"))


def test_alias_inventory_accounts_for_overlapping_roots_and_skips_symlinks(tmp_path):
    """Every name of a benign inode is counted once across overlapping no-follow roots."""
    nested = tmp_path / "nested"
    nested.mkdir()
    first, second = tmp_path / "first", nested / "second"
    first.write_text("ordinary")
    os.link(first, second)
    (nested / "single").write_text("unlinked")
    (nested / "symbolic").symlink_to(tmp_path.parent, target_is_directory=True)
    assert inspect_hardlink_aliases((nested, tmp_path, nested), float("inf")) == {
        first: (first, second),
        second: (first, second),
    }


def test_metadata_sources_complete_only_visible_groups_without_opening_files(tmp_path, monkeypatch):
    """Find needed names across sources without reading payloads or chasing unrelated links."""
    view, cache = tmp_path / "workspace", tmp_path / "cache"
    view.mkdir()
    cache.mkdir()
    nested = cache / "nested"
    nested.mkdir()
    first, other = view / "first", view / "other"
    first.touch()
    other.touch()
    (cache / "one").hardlink_to(first)
    (nested / "two").hardlink_to(other)
    unrelated = cache / "unrelated"
    unrelated.touch()
    (tmp_path / "outside").hardlink_to(unrelated)
    (cache / "symbolic").symlink_to(view, target_is_directory=True)
    (cache / "ordinary").touch()
    original_open = os.open

    def only_open_directories(path, flags, **kwargs):
        """Only the admitted view or a pinned source directory can be opened for listing."""
        assert flags & os.O_DIRECTORY and flags & os.O_NOFOLLOW
        if path not in (view, "."):
            search = getattr(os, "O_SEARCH", getattr(os, "O_PATH", os.O_RDONLY))
            assert flags & search == search
        return original_open(path, flags, **kwargs)

    monkeypatch.setattr(os, "open", only_open_directories)
    aliases = inspect_hardlink_aliases((view,), float("inf"), hardlink_sources=(view, cache, cache))
    groups = (tuple(sorted((first, cache / "one"))), tuple(sorted((other, nested / "two"))))
    assert aliases == {path: group for group in groups for path in group}


@pytest.mark.parametrize("case", ["missing", "private-file", "private-source", "policy", "symlink"])
def test_metadata_sources_return_only_visible_unaccounted_names(tmp_path, case):
    """A denied, missing or symbolic peer cannot complete a partially witnessed inode."""
    view, source = tmp_path / "workspace", tmp_path / "source"
    view.mkdir()
    source.mkdir()
    visible = view / "installed"
    visible.touch()
    (source / "public").hardlink_to(visible)
    hidden = {
        "private-file": source / ".env.production",
        "private-source": source / ".SSH/key",
        "policy": source / "designated",
    }.get(case, tmp_path / "outside/peer")
    hidden.parent.mkdir(exist_ok=True)
    hidden.hardlink_to(visible)
    if case == "symlink":
        (source / "other").symlink_to(hidden.parent, target_is_directory=True)
    unresolved = set()
    aliases = inspect_hardlink_aliases(
        (view,),
        float("inf"),
        hardlink_sources=(source, hidden.parent / ".config"),
        denied_peer=lambda path: case == "policy" and path == hidden,
        unaccounted=unresolved,
    )
    assert not aliases and unresolved == {visible}


def test_metadata_sources_require_canonical_stable_directories(tmp_path, monkeypatch):
    """Refuse ambiguous roots, symlink ancestors and changed directories queued for traversal."""
    view, source = tmp_path / "workspace", tmp_path / "cache"
    view.mkdir()
    source.mkdir()
    installed = view / "installed"
    installed.touch()
    child = source / "child"
    child.mkdir()
    (child / "peer").hardlink_to(installed)
    alias = tmp_path / "symbolic"
    alias.symlink_to(source, target_is_directory=True)
    for root in (Path("relative"), source / ".." / source.name):
        with pytest.raises(ValueError, match="canonical absolute"):
            inspect_hardlink_aliases((view,), float("inf"), hardlink_sources=(root,))
    with pytest.raises(OSError):
        inspect_hardlink_aliases((view,), float("inf"), hardlink_sources=(alias / "child",))
    scandir = os.scandir
    root_identity = source.stat()

    @contextmanager
    def replace_queued_directory(descriptor):
        """Change the source child after it was queued but before it can provide evidence."""
        with scandir(descriptor) as entries:
            yield entries
        if os.path.samestat(root_identity, os.fstat(descriptor)):
            child.rename(tmp_path / "retired")
            child.mkdir()

    monkeypatch.setattr(os, "scandir", replace_queued_directory)
    with pytest.raises(ValueError, match="directory changed"):
        inspect_hardlink_aliases((view,), float("inf"), hardlink_sources=(source,))


@pytest.mark.parametrize("location", ["visible", "source", "duplicate"])
def test_metadata_inventory_counts_physical_directory_entries(tmp_path, monkeypatch, location):
    """A mounted inode or a repeated directory entry cannot substitute for an external name."""
    view, source = tmp_path / "workspace", tmp_path / "cache"
    view.mkdir()
    source.mkdir()
    visible = view / "file"
    visible.touch()
    (source / "peer").hardlink_to(visible)
    replaced = source if location == "source" else view
    identity = replaced.stat()
    scandir = os.scandir

    @contextmanager
    def mounted_entry(descriptor):
        """Represent a mounted node whose underlying parent entry names another inode."""
        with scandir(descriptor) as entries:
            if not os.path.samestat(identity, os.fstat(descriptor)):
                yield entries
            else:
                item = next(entries)
                if location == "duplicate":
                    yield [item, item]
                else:
                    yield [SimpleNamespace(name=item.name, stat=item.stat, inode=lambda: -1)]

    monkeypatch.setattr(os, "scandir", mounted_entry)
    diagnostic = {
        "visible": "physical directory entries",
        "source": "Incomplete hardlink",
        "duplicate": "duplicate directory entries",
    }[location]
    with pytest.raises(ValueError, match=diagnostic):
        inspect_hardlink_aliases((view,), float("inf"), hardlink_sources=(source,))


def test_metadata_sources_share_the_deadline_and_entry_limit(tmp_path, monkeypatch):
    """Source traversal spends the original inventory budget and stops at its deadline."""
    view, source = tmp_path / "workspace", tmp_path / "cache"
    view.mkdir()
    source.mkdir()
    visible = view / "file"
    visible.touch()
    (source / "peer").hardlink_to(visible)
    with pytest.raises(ValueError, match="entry bound"):
        inspect_hardlink_aliases((view,), float("inf"), hardlink_sources=(source,), max_entries=1)
    times = iter((0, 0, 2))
    monkeypatch.setattr("loop.permissions.protection.time.monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="deadline"):
        inspect_hardlink_aliases((view,), 1, hardlink_sources=(source,))


@pytest.mark.parametrize("case", ["outside", "relative", "symlink", "bound", "deadline"])
def test_alias_inventory_refuses_uncertain_or_unbounded_views(tmp_path, case):
    """Outside aliases, unsafe roots and exhausted budgets never produce an admitted view."""
    root = tmp_path / "root"
    root.mkdir()
    source = root / "ordinary"
    source.write_text("canary")
    options = {"deadline": float("inf")}
    error, diagnostic = ValueError, "Incomplete hardlink"
    if case == "outside":
        os.link(source, tmp_path / "outside")
    elif case == "relative":
        root = Path("relative")
        diagnostic = "canonical absolute"
    elif case == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
        diagnostic = "canonical absolute"
    elif case == "bound":
        options["max_entries"] = 0
        diagnostic = "entry bound"
    else:
        options["deadline"] = 0
        error, diagnostic = TimeoutError, "deadline"
    with pytest.raises(error, match=diagnostic):
        inspect_hardlink_aliases((root,), **options)


@pytest.mark.parametrize("stage", ["root", "child", "entry", "scan-deadline", "final-deadline"])
def test_alias_inventory_rechecks_pinned_identities_and_complete_snapshot(
    tmp_path, monkeypatch, stage
):
    """Directory replacement, post-scan link changes and late deadline expiry refuse admission."""
    child = tmp_path / "child"
    child.mkdir()
    source = child / "ordinary"
    source.write_text("canary")
    original_fstat = os.fstat
    original_lstat = Path.lstat
    calls = {"fstat": 0, "lstat": 0, "clock": 0}

    def fstat(descriptor):
        """Replace only the identity observed at the selected pinned boundary."""
        details = original_fstat(descriptor)
        calls["fstat"] += 1
        if (stage == "root" and calls["fstat"] == 1) or (stage == "child" and calls["fstat"] == 2):
            return SimpleNamespace(st_dev=details.st_dev, st_ino=-1)
        return details

    def lstat(path, **kwargs):
        """Change the regular-file topology before its final metadata check."""
        if stage == "entry" and path == source:
            os.link(source, tmp_path / "injected")
        return original_lstat(path, **kwargs)

    def clock():
        """Expire after scanning both directory entries, before validating the snapshot."""
        calls["clock"] += 1
        return 2 if calls["clock"] >= (2 if stage == "scan-deadline" else 4) else 0

    monkeypatch.setattr("loop.permissions.protection.os.fstat", fstat)
    monkeypatch.setattr(Path, "lstat", lstat)
    if stage.endswith("deadline"):
        monkeypatch.setattr("loop.permissions.protection.time.monotonic", clock)
    with pytest.raises(
        TimeoutError if stage.endswith("deadline") else ValueError,
        match="deadline" if stage.endswith("deadline") else "changed",
    ):
        inspect_hardlink_aliases((tmp_path,), 1 if stage.endswith("deadline") else float("inf"))


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ("completed-output", None),
        ("external-link", "Incomplete hardlink aliases"),
        ("entry-budget", "entry bound exhausted"),
        ("persistent-writer", "entry changed"),
        ("new-root", "scratch identity changed"),
    ],
)
def test_alias_inventory_rechecks_private_scratch_membership(
    tmp_path, monkeypatch, change, failure
):
    """Recount new scratch outputs within the original budget and refuse unsafe or unstable views."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "existing").touch()
    external = tmp_path / "external"
    external.touch()
    root_identity = scratch.stat()
    scandir = os.scandir
    visits = []

    @contextmanager
    def publish_after_scan(descriptor):
        """Publish an output after enumeration and before its directory's final identity check."""
        with scandir(descriptor) as entries:
            yield entries
        if not os.path.samestat(os.fstat(descriptor), root_identity):
            return
        visits.append(descriptor)
        if change == "persistent-writer":
            (scratch / f"output-{len(visits)}").mkdir()
        elif len(visits) == 1:
            if change == "new-root":
                scratch.rename(tmp_path / "retired")
                scratch.mkdir()
            else:
                output.mkdir()
                if change == "external-link":
                    alias.hardlink_to(external)
                else:
                    source.touch()
                    alias.hardlink_to(source)

    output = scratch / "output"
    source = output / "source"
    alias = output / "alias"
    monkeypatch.setattr(os, "scandir", publish_after_scan)
    options = {"scratch": scratch, "max_entries": 4 if change == "entry-budget" else 30}
    if failure:
        with pytest.raises(ValueError, match=failure):
            inspect_hardlink_aliases((scratch,), float("inf"), **options)
    else:
        names = tuple(sorted((source, alias)))
        assert inspect_hardlink_aliases((scratch,), float("inf"), **options) == dict.fromkeys(
            names, names
        )
        assert len(visits) == 2


def test_alias_inventory_refuses_replaced_or_untraversed_scratch(tmp_path, monkeypatch):
    """The bound private directory must keep its identity and take part in every complete walk."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="did not traverse"):
        inspect_hardlink_aliases((), float("inf"), scratch=scratch)
    original_open = os.open

    def replace_before_open(path, flags):
        """Replace the scratch path between initial binding and opening the root."""
        scratch.rename(tmp_path / "retired")
        scratch.mkdir()
        return original_open(path, flags)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(ValueError, match="scratch identity changed"):
        inspect_hardlink_aliases((scratch,), float("inf"), scratch=scratch)
