"""Verify the permission-owned protected workspace path specification."""

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.permissions.protection import (
    HardlinkInventory,
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


def _search_for_peers(view, sources, *, deadline=float("inf"), denied_peer=None, max_records=100):
    """Gather source metadata and return all accounted and incomplete inode groups."""
    inventory = HardlinkInventory(deadline, max_records=max_records)
    inventory.scan((view,))
    inventory.search(sources, denied_peer)
    assert inventory.verify()
    return [(set(names), visible, complete) for names, visible, complete in inventory.groups()]


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
        if path not in (view, ".") and not (kwargs.get("dir_fd") and path == nested.name):
            search = getattr(os, "O_SEARCH", getattr(os, "O_PATH", os.O_RDONLY))
            assert flags & search == search
        return original_open(path, flags, **kwargs)

    monkeypatch.setattr(os, "open", only_open_directories)
    groups = _search_for_peers(view, (view, cache, cache))
    assert all(visible == 1 and complete for _, visible, complete in groups)
    assert {frozenset(names) for names, _, _ in groups} == {
        frozenset((first, cache / "one")),
        frozenset((other, nested / "two")),
    }


@pytest.mark.parametrize(
    "case", ["missing", "private-file", "private-source", "policy", "denied-root", "symlink"]
)
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
    groups = _search_for_peers(
        view,
        (source, hidden.parent / ".config"),
        denied_peer=lambda path, _details: (
            case == "policy" and path == hidden or case == "denied-root" and path == source
        ),
    )
    expected_names = {visible} if case == "denied-root" else {visible, source / "public"}
    assert groups == [(expected_names, 1, False)]


@pytest.mark.parametrize("phase", ["before-open", "after-scan"])
def test_metadata_sources_require_canonical_stable_directories(tmp_path, monkeypatch, phase):
    """Refuse ambiguous roots, symlink ancestors and unstable source directories."""
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
            _search_for_peers(view, (root,))
    with pytest.raises(OSError):
        _search_for_peers(view, (alias / "child",))
    scandir = os.scandir
    open_directory = os.open
    root_identity = source.stat()

    def replace_child():
        """Replace the observed source directory with a different inode."""
        child.rename(tmp_path / "retired")
        child.mkdir()

    @contextmanager
    def replace_scanned_directory(descriptor):
        """Invalidate source evidence after its parent has been scanned."""
        with scandir(descriptor) as entries:
            yield entries
        if phase == "after-scan" and os.path.samestat(root_identity, os.fstat(descriptor)):
            replace_child()

    def replace_opened_directory(path, flags, **kwargs):
        """Replace a child between observing and opening its directory entry."""
        if phase == "before-open" and path == child.name and kwargs.get("dir_fd") is not None:
            replace_child()
        return open_directory(path, flags, **kwargs)

    monkeypatch.setattr(os, "scandir", replace_scanned_directory)
    monkeypatch.setattr(os, "open", replace_opened_directory)
    with pytest.raises(ValueError, match="inventory .* changed"):
        _search_for_peers(view, (source,))


@pytest.mark.parametrize("location", ["visible", "source", "duplicate", "changed-type"])
def test_metadata_inventory_counts_physical_directory_entries(tmp_path, monkeypatch, location):
    """Only matching physical file entries can complete a group."""
    view, source = tmp_path / "workspace", tmp_path / "cache"
    view.mkdir()
    source.mkdir()
    visible = view / "file"
    visible.touch()
    (source / "peer").hardlink_to(visible)
    if location == "changed-type":
        (source / "ordinary").touch()
    replaced = source if location in {"source", "changed-type"} else view
    identity = replaced.stat()
    scandir = os.scandir

    @contextmanager
    def mounted_entry(descriptor):
        """Simulate stale directory entries or a duplicate spelling in a pinned directory."""
        with scandir(descriptor) as entries:
            if not os.path.samestat(identity, os.fstat(descriptor)):
                yield entries
            else:
                item = next(
                    entry
                    for entry in entries
                    if location != "changed-type" or entry.name == "ordinary"
                )
                inode = item.inode()
                if location == "duplicate":
                    yield [item, item]
                else:
                    entry = SimpleNamespace(
                        name=item.name, is_dir=item.is_dir, stat=item.stat, inode=item.inode
                    )
                    if location == "visible":
                        entry.inode = lambda: -1
                    elif location == "source":
                        entry.stat = lambda **_: SimpleNamespace(st_dev=-1, st_ino=inode)
                    else:
                        entry.is_dir = lambda **_: True
                    yield [entry]

    monkeypatch.setattr(os, "scandir", mounted_entry)
    diagnostic = {
        "visible": "physical directory entries",
        "source": "Incomplete hardlink",
        "duplicate": "duplicate directory entries",
        "changed-type": "Incomplete hardlink",
    }[location]
    if location in ("source", "changed-type"):
        assert _search_for_peers(view, (source,)) == [({visible}, 1, False)]
    else:
        with pytest.raises(ValueError, match=diagnostic):
            _search_for_peers(view, (source,))


def test_inventory_refuses_different_observations_of_one_inode(tmp_path, monkeypatch):
    """A hardlink group is rejected when membership changes between physical name observations."""
    root = tmp_path / "scoped"
    root.mkdir()
    first, second = root / "first", root / "second"
    first.write_text("content")
    second.hardlink_to(first)
    actual_scan = os.scandir
    outside = tmp_path / "late"

    @contextmanager
    def changed_scan(descriptor):
        """Supply two observations with different link counts for the same inode."""
        with actual_scan(descriptor) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)

            def entry_observations():
                """Add a new name only after the first in-scope observation."""
                yield ordered[0]
                outside.hardlink_to(first)
                yield ordered[1]

            yield entry_observations()

    monkeypatch.setattr(os, "scandir", changed_scan)
    with pytest.raises(ValueError, match="Alias inventory entry changed"):
        inspect_hardlink_aliases((root,), float("inf"))


def test_metadata_sources_share_the_deadline_and_record_limit(tmp_path, monkeypatch):
    """Source traversal spends the original inventory budget and stops at its deadline."""
    view, source = tmp_path / "workspace", tmp_path / "cache"
    view.mkdir()
    source.mkdir()
    visible = view / "file"
    visible.touch()
    (source / "peer").hardlink_to(visible)
    with pytest.raises(ValueError, match="metadata limit"):
        _search_for_peers(view, (source,), max_records=3)
    times = iter((0, 0, 2))
    monkeypatch.setattr("loop.permissions.protection.time.monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="deadline"):
        _search_for_peers(view, (source,), deadline=1)


@pytest.mark.parametrize("outside_view", [False, True])
def test_alias_inventory_budget_tracks_link_evidence(tmp_path, outside_view):
    """Unrelated files and metadata-only branches leave the evidence budget available."""
    view, cache = tmp_path / "workspace", tmp_path / "cache"
    view.mkdir()
    cache.mkdir()
    visible = view / "installed"
    visible.touch()
    directory = cache if outside_view else view
    for index in range(12):
        (directory / f"ordinary-{index}").touch()
        if outside_view:
            unrelated = directory / f"unrelated-{index}"
            unrelated.mkdir()
            (unrelated / "file").touch()
    peer = (cache / "package" if outside_view else view) / "peer"
    peer.parent.mkdir(exist_ok=True)
    peer.hardlink_to(visible)

    def denied_peer(path, _details):
        """Only possible matching names and their directories require policy inspection."""
        assert path == peer or path.is_dir()
        return False

    assert _search_for_peers(
        view,
        (cache,) if outside_view else (),
        max_records=5,
        denied_peer=denied_peer,
    ) == [({visible, peer}, 1 if outside_view else 2, True)]


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
        options["max_records"] = 0
        diagnostic = "metadata limit"
    else:
        options["deadline"] = 0
        error, diagnostic = TimeoutError, "deadline"
    with pytest.raises(error, match=diagnostic):
        inspect_hardlink_aliases((root,), **options)


@pytest.mark.parametrize(
    "stage",
    ["root", "child", "recheck", "new-link", "known-link", "scan-deadline", "final-deadline"],
)
def test_alias_inventory_rechecks_pinned_identities_and_complete_snapshot(
    tmp_path, monkeypatch, stage
):
    """Directory replacement, post-scan link changes and late deadline expiry refuse admission."""
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    source = child / "ordinary"
    source.write_text("canary")
    if stage == "known-link":
        (child / "known").hardlink_to(source)
    child_identity = child.stat()
    original_fstat = os.fstat
    original_scandir = os.scandir
    calls = {"fstat": 0, "child_scans": 0, "clock": 0}

    def fstat(descriptor):
        """Replace only the identity observed at the selected pinned boundary."""
        details = original_fstat(descriptor)
        calls["fstat"] += 1
        if calls["fstat"] == {"root": 1, "child": 2, "recheck": 4}.get(stage):
            return SimpleNamespace(st_dev=details.st_dev, st_ino=-1)
        return details

    @contextmanager
    def scandir(descriptor):
        """Introduce an external peer after inventorying the admitted directory."""
        is_child = os.path.samestat(original_fstat(descriptor), child_identity)
        if is_child:
            calls["child_scans"] += 1
            if stage == "new-link" and calls["child_scans"] == 2:
                (tmp_path / "injected").hardlink_to(source)
        with original_scandir(descriptor) as entries:
            yield entries
        if is_child and stage == "known-link" and calls["child_scans"] == 1:
            (tmp_path / "injected").hardlink_to(source)

    def clock():
        """Expire after scanning both directory entries, before validating the snapshot."""
        calls["clock"] += 1
        return 2 if calls["clock"] >= (2 if stage == "scan-deadline" else 4) else 0

    monkeypatch.setattr("loop.permissions.protection.os.fstat", fstat)
    monkeypatch.setattr(os, "scandir", scandir)
    if stage.endswith("deadline"):
        monkeypatch.setattr("loop.permissions.protection.time.monotonic", clock)
    with pytest.raises(
        TimeoutError if stage.endswith("deadline") else ValueError,
        match="deadline" if stage.endswith("deadline") else "changed",
    ):
        inspect_hardlink_aliases((root,), 1 if stage.endswith("deadline") else float("inf"))


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ("completed-output", None),
        ("external-link", "Incomplete hardlink aliases"),
        ("record-budget", "metadata limit exceeded"),
        ("persistent-writer", "entry changed"),
        ("new-root", "scratch identity changed"),
    ],
)
def test_alias_inventory_rechecks_private_scratch_membership(
    tmp_path, monkeypatch, change, failure
):
    """Recount new scratch outputs within the original budget and refuse unsafe views."""
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
    options = {"scratch": scratch, "max_records": 4 if change == "record-budget" else 30}
    if failure:
        with pytest.raises(ValueError, match=failure):
            inspect_hardlink_aliases((scratch,), float("inf"), **options)
    else:
        names = tuple(sorted((source, alias)))
        assert inspect_hardlink_aliases((scratch,), float("inf"), **options) == dict.fromkeys(
            names, names
        )
        assert len(visits) == 3


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
