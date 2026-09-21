"""Test trusted observed-delta normalization into portable VFS effects."""

from __future__ import annotations

from itertools import permutations

import pytest

from loop.execution.contracts import DeltaEffect
from loop.execution.vfs import (
    ContentReference,
    ObjectKind,
    ObservedDelta,
    ObservedDeltaEntry,
    ObservedObjectKind,
    UnrepresentableDeltaError,
    normalize_delta,
)


def _content(size: int = 2) -> ContentReference:
    """Return an opaque regular-file content reference of the requested size."""
    return ContentReference(digest="sha256:ab", size=size, reference="content:one")


def _observed(
    path: str,
    effect: DeltaEffect = DeltaEffect.CREATE,
    kind: ObservedObjectKind | None = ObservedObjectKind.FILE,
    **changes: object,
) -> ObservedDeltaEntry:
    """Return one valid portable observed effect with optional focused changes."""
    values: dict[str, object] = {
        "effect": effect,
        "destination_path": path,
        "object_kind": kind,
        "content": _content() if kind is ObservedObjectKind.FILE else None,
    }
    values.update(changes)
    return ObservedDeltaEntry(**values)


def test_normalize_delta_orders_all_portable_effects_deterministically():
    """Create, replace, delete, rename, symlink, and metadata effects normalize stably."""
    delta = normalize_delta(
        ObservedDelta(
            entries=(
                _observed("z", DeltaEffect.METADATA),
                _observed("old", DeltaEffect.RENAME, source_path="source"),
                _observed("removed", DeltaEffect.DELETE, None),
                _observed(
                    "link", kind=ObservedObjectKind.SYMLINK, content=None, symlink_target="target"
                ),
                _observed("a", DeltaEffect.REPLACE),
            )
        )
    )

    assert [entry.destination_path for entry in delta.entries] == [
        "a",
        "link",
        "old",
        "removed",
        "z",
    ]
    assert delta.entries[1].object_kind is ObjectKind.SYMLINK
    assert delta.entries[2].source_path == "source"
    assert delta.entries[3].object_kind is None


def test_normalize_delta_has_permutation_invariant_canonical_serialization():
    """Every ordering of independent inspector effects produces the same persisted delta."""
    entries = (
        _observed("a", DeltaEffect.REPLACE),
        _observed("directory", kind=ObservedObjectKind.DIRECTORY),
        _observed("directory/file"),
    )
    normalized = {
        normalize_delta(ObservedDelta(entries=ordering)).model_dump_json()
        for ordering in permutations(entries)
    }

    assert len(normalized) == 1


def test_normalize_delta_allows_a_created_directory_with_child_effects():
    """A directory result can consistently contain separately observed child effects."""
    delta = normalize_delta(
        ObservedDelta(
            entries=(
                _observed("directory", kind=ObservedObjectKind.DIRECTORY),
                _observed("directory/file"),
            )
        )
    )

    assert [entry.destination_path for entry in delta.entries] == ["directory", "directory/file"]


def test_normalize_delta_orders_recursive_deletion_deepest_first():
    """An observed removed tree is explicit and ordered so directories are empty at removal."""
    delta = normalize_delta(
        ObservedDelta(
            entries=(
                _observed("directory", DeltaEffect.DELETE, None),
                _observed("directory/child", DeltaEffect.DELETE, None),
                _observed("directory/child/file", DeltaEffect.DELETE, None),
            )
        )
    )

    assert [entry.destination_path for entry in delta.entries] == [
        "directory/child/file",
        "directory/child",
        "directory",
    ]


def test_normalize_delta_allows_an_acyclic_rename_chain():
    """A topologically orderable rename chain remains a committable canonical graph."""
    delta = normalize_delta(
        ObservedDelta(
            entries=(
                _observed("b", DeltaEffect.RENAME, source_path="a"),
                _observed("c", DeltaEffect.RENAME, source_path="b"),
            )
        )
    )

    assert [(entry.source_path, entry.destination_path) for entry in delta.entries] == [
        ("a", "b"),
        ("b", "c"),
    ]


@pytest.mark.parametrize(
    ("entries", "message"),
    (
        ((_observed("same"), _observed("same", DeltaEffect.REPLACE)), "duplicate destinations"),
        (
            (_observed("parent", DeltaEffect.DELETE, None), _observed("parent/child")),
            "parent and child",
        ),
        ((_observed("file"), _observed("file/child")), "parent and child"),
        (
            (
                _observed("a", DeltaEffect.RENAME, source_path="b"),
                _observed("b", DeltaEffect.RENAME, source_path="a"),
            ),
            "rename cycle",
        ),
        (
            (
                _observed("one", DeltaEffect.RENAME, source_path="source"),
                _observed("two", DeltaEffect.RENAME, source_path="source"),
            ),
            "duplicate rename sources",
        ),
        (
            (
                _observed("destination", DeltaEffect.RENAME, source_path="source"),
                _observed("source", DeltaEffect.REPLACE),
            ),
            "overwrites a rename source",
        ),
    ),
)
def test_normalize_delta_rejects_conflicting_effect_graphs(
    entries: tuple[ObservedDeltaEntry, ...], message: str
):
    """Ambiguous duplicate, tree, and rename graphs fail before canonicalization."""
    with pytest.raises(UnrepresentableDeltaError, match=message):
        normalize_delta(ObservedDelta(entries=entries))


@pytest.mark.parametrize(
    ("kind", "flag", "message"),
    (
        (ObservedObjectKind.FIFO, None, "node type"),
        (ObservedObjectKind.SOCKET, None, "node type"),
        (ObservedObjectKind.BLOCK_DEVICE, None, "node type"),
        (ObservedObjectKind.CHARACTER_DEVICE, None, "node type"),
        (ObservedObjectKind.HARDLINK, None, "node type"),
        (ObservedObjectKind.FILE, "ownership_changed", "ownership"),
        (ObservedObjectKind.FILE, "acl_changed", "ACL"),
        (ObservedObjectKind.FILE, "security_xattrs_changed", "security xattrs"),
        (ObservedObjectKind.FILE, "hardlink_topology_changed", "hardlink topology"),
    ),
)
def test_normalize_delta_rejects_unsupported_nodes_and_metadata_changes(
    kind: ObservedObjectKind, flag: str | None, message: str
):
    """Nonportable nodes and metadata mutations are rejected rather than emulated."""
    changes = {flag: True} if flag is not None else {}
    with pytest.raises(UnrepresentableDeltaError, match=message):
        normalize_delta(ObservedDelta(entries=(_observed("entry", kind=kind, **changes),)))


def test_normalize_delta_enforces_entry_and_content_quotas():
    """Trusted observations cannot exceed bounded effect or staged-content accounting."""
    with pytest.raises(UnrepresentableDeltaError, match="entry quota"):
        normalize_delta(ObservedDelta(entries=(_observed("a"),), maximum_entries=0))
    with pytest.raises(UnrepresentableDeltaError, match="content quota"):
        normalize_delta(
            ObservedDelta(
                entries=(_observed("a", content=_content(size=3)),), maximum_content_bytes=2
            )
        )


def test_observed_delta_rejects_untrusted_host_and_effect_shapes():
    """The inspector feed retains virtual paths and unambiguous effect-specific fields."""
    with pytest.raises(ValueError, match="relative virtual"):
        _observed("/host/path")
    with pytest.raises(ValueError, match="Rename"):
        _observed("destination", DeltaEffect.RENAME)
    with pytest.raises(ValueError, match="Only rename"):
        _observed("destination", source_path="source")
    with pytest.raises(ValueError, match="Delete"):
        _observed("deleted", DeltaEffect.DELETE, ObservedObjectKind.FILE)
    with pytest.raises(ValueError, match="Non-delete"):
        _observed("missing", kind=None)
