"""Tests for mutable path references and path discovery helpers."""

import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from loop.utils import PathHolder, ValueHolder


class ExamplePathLike(os.PathLike[str]):
    """Provide a minimal string-backed path-like value for compatibility tests."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __fspath__(self) -> str:
        return self.value


@pytest.mark.parametrize("value", ["folder/file.txt", Path("folder/file.txt")])
def test_path_holder_constructs_from_string_and_path(value):
    """Construction normalizes string and Path inputs into a Path value."""
    holder = PathHolder(value)

    assert holder.value == Path("folder/file.txt")
    assert isinstance(holder.value, Path)


def test_path_holder_shares_the_common_value_holder_api_and_storage():
    """Value, get, and set expose one Path through a stable holder identity."""
    holder = PathHolder("first")
    consumer = holder

    assert isinstance(holder, ValueHolder)
    assert holder.value == holder.get() == Path("first")

    holder.value = ExamplePathLike("second")
    assert holder.value == holder.get() == Path("second")

    assert holder.set("third") is None
    assert consumer is holder
    assert consumer.value == consumer.get() == Path("third")

    holder.value = "fourth"
    assert holder.value == holder.get() == Path("fourth")


def test_path_holder_constructs_from_and_retargets_to_pathlike_values():
    """Construction and mutation accept arbitrary compatible string path-like values."""
    holder = PathHolder(ExamplePathLike("first"))

    holder.set(ExamplePathLike("second"))

    assert holder.value == Path("second")


def test_path_holder_retargets_in_place():
    """Replacing path changes future operations without replacing the holder object."""
    holder = PathHolder("first")
    identity = id(holder)

    holder.set(Path("second"))

    assert id(holder) == identity
    assert str(holder) == "second"


def test_path_holder_supports_pathlike_and_stdlib_filesystem_apis(tmp_path):
    """The holder works directly with fspath, open, stat, and os.path APIs."""
    target = tmp_path / "content.txt"
    holder = PathHolder(target)

    with open(holder, "w", encoding="utf-8") as stream:
        stream.write("content")

    assert os.fspath(holder) == os.fspath(target)
    assert os.stat(holder).st_size == 7
    assert os.path.exists(holder)


def test_path_holder_string_and_representation_show_the_current_path():
    """String conversion mirrors Path while repr identifies the mutable wrapper."""
    holder = PathHolder("current/path")

    assert str(holder) == str(Path("current/path"))
    assert repr(holder) == f"PathHolder({Path('current/path')!r})"


def test_path_holder_exposes_common_path_properties():
    """Lexical Path properties reflect the holder's current value."""
    holder = PathHolder("/archive/report.tar.gz")

    assert holder.anchor == "/"
    assert holder.drive == ""
    assert holder.name == "report.tar.gz"
    assert holder.parent == Path("/archive")
    assert list(holder.parents)[:2] == [Path("/archive"), Path("/")]
    assert holder.parts == ("/", "archive", "report.tar.gz")
    assert holder.root == "/"
    assert holder.stem == "report.tar"
    assert holder.suffix == ".gz"
    assert holder.suffixes == [".tar", ".gz"]


def test_path_holder_delegates_common_file_and_directory_methods(tmp_path):
    """Common filesystem mutations and queries operate on the current target."""
    holder = PathHolder(tmp_path / "nested")
    holder.mkdir()
    assert holder.exists() and holder.is_dir()

    holder.set(holder / "content.txt")
    assert holder.write_text("hello", encoding="utf-8") == 5
    assert holder.exists() and holder.is_file()
    assert holder.read_text(encoding="utf-8") == "hello"
    assert holder.read_bytes() == b"hello"
    assert holder.stat().st_size == 5
    assert holder.match("*.txt")
    holder.unlink()
    assert not holder.exists()


def test_path_holder_delegates_lexical_and_platform_queries(tmp_path):
    """Lexical, ownership, and special-file queries mirror the current Path."""
    holder = PathHolder(tmp_path)

    assert holder.as_posix() == tmp_path.as_posix()
    assert holder.as_uri() == tmp_path.as_uri()
    assert holder.expanduser() == tmp_path.expanduser()
    assert holder.group() == tmp_path.group()
    assert holder.owner() == tmp_path.owner()
    assert holder.is_absolute()
    assert holder.is_mount() == tmp_path.is_mount()
    assert holder.is_relative_to(tmp_path.parent)
    with pytest.deprecated_call():
        assert holder.is_reserved() == tmp_path.is_reserved()
    assert not holder.is_block_device()
    assert not holder.is_char_device()
    assert not holder.is_fifo()
    assert not holder.is_junction()
    assert not holder.is_socket()
    assert not holder.is_symlink()


def test_path_holder_delegates_extended_filesystem_operations(tmp_path):
    """Links, metadata, moves, binary I/O, and directory removal target the current path."""
    source = tmp_path / "source.bin"
    holder = PathHolder(source)
    holder.touch(mode=0o600)
    holder.chmod(0o640)
    holder.lchmod(0o640)
    assert holder.lstat().st_size == 0
    assert holder.write_bytes(b"data") == 4
    with holder.open("rb") as stream:
        assert stream.read() == b"data"
    assert holder.samefile(source)

    renamed = tmp_path / "renamed.bin"
    assert holder.rename(renamed) == renamed
    holder.set(renamed)
    replaced = tmp_path / "replaced.bin"
    replaced.write_bytes(b"old")
    assert holder.replace(replaced) == replaced

    hardlink = PathHolder(tmp_path / "hardlink.bin")
    hardlink.hardlink_to(replaced)
    assert hardlink.samefile(replaced)
    symlink = PathHolder(tmp_path / "symlink.bin")
    symlink.symlink_to(replaced)
    assert symlink.readlink() == replaced

    directory = PathHolder(tmp_path / "empty")
    directory.mkdir()
    directory.rmdir()
    assert not directory.exists()


def test_path_holder_relative_to_returns_a_path_snapshot():
    """Relative conversion delegates its path and walk-up options to pathlib."""
    holder = PathHolder("/one/two/file.txt")

    assert holder.relative_to("/one/two") == Path("file.txt")
    assert holder.relative_to("/elsewhere", walk_up=True) == Path("../one/two/file.txt")


def test_path_holder_derived_paths_are_path_snapshots():
    """Path-producing lexical and normalization operations return immutable snapshots."""
    holder = PathHolder("/old/report.txt")
    snapshots = [
        holder / "child",
        "parent" / holder,
        holder.joinpath("child"),
        holder.parent,
        holder.with_name("other.txt"),
        holder.with_stem("summary"),
        holder.with_suffix(".md"),
        holder.absolute(),
        holder.resolve(),
    ]

    holder.set("/new/report.txt")

    assert all(isinstance(path, Path) for path in snapshots)
    assert snapshots[0] == Path("/old/report.txt/child")
    assert snapshots[2] == Path("/old/report.txt/child")
    assert snapshots[3] == Path("/old")
    assert snapshots[4] == Path("/old/other.txt")
    assert snapshots[5] == Path("/old/summary.txt")
    assert snapshots[6] == Path("/old/report.md")


def test_path_holder_path_iterators_yield_path_snapshots(tmp_path):
    """Directory, glob, recursive glob, and walk results contain ordinary Paths."""
    nested = tmp_path / "nested"
    nested.mkdir()
    text = nested / "note.txt"
    text.write_text("note", encoding="utf-8")
    holder = PathHolder(tmp_path)

    children = list(holder.iterdir())
    globbed = list(holder.glob("**/*.txt"))
    recursively_globbed = list(holder.rglob("*.txt"))
    walked = list(holder.walk())
    holder.set(tmp_path / "elsewhere")

    assert children == [nested]
    assert globbed == [text]
    assert recursively_globbed == [text]
    assert walked[0][0] == tmp_path
    assert all(isinstance(path, Path) for path in [*children, *globbed, *recursively_globbed])
    assert all(isinstance(root, Path) for root, _, _ in walked)


def test_path_holder_equality_tracks_current_path_and_supports_ordering():
    """Equality and ordering use current Path values for holders and ordinary Paths."""
    first = PathHolder("a")
    same = PathHolder(Path("a"))
    later = PathHolder("b")

    assert first == same
    assert first == Path("a")
    assert first != Path("b")
    assert first < later
    assert first <= same
    assert later > first
    assert later >= Path("b")

    same.set("c")

    assert first != same


def test_path_holder_comparisons_unwrap_other_value_holder_types():
    """Cross-holder comparisons follow the underlying values' normal Python semantics."""
    holder = PathHolder("a")
    generic = ValueHolder(Path("a"))

    assert holder == generic
    assert generic == holder
    assert holder <= generic


def test_path_holder_is_unhashable_because_its_equality_value_is_mutable():
    """A holder cannot be hashed because replacing its path changes equality."""
    with pytest.raises(TypeError, match="unhashable type"):
        hash(PathHolder("value"))


def test_path_iterators_snapshot_before_the_holder_is_retargeted(tmp_path):
    """Lazy path iterators remain bound to the path current when each call began."""
    original = tmp_path / "original"
    original.mkdir()
    child = original / "child.txt"
    child.write_text("content", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    holder = PathHolder(original)

    children = holder.iterdir()
    globbed = holder.glob("*.txt")
    recursively_globbed = holder.rglob("*.txt")
    walked = holder.walk()
    holder.set(replacement)

    assert list(children) == [child]
    assert list(globbed) == [child]
    assert list(recursively_globbed) == [child]
    assert next(walked)[0] == original


def test_path_iteration_does_not_block_retargeting(tmp_path):
    """An unconsumed filesystem iterator never retains the holder's synchronization lock."""
    original = tmp_path / "original"
    original.mkdir()
    (original / "child").touch()
    holder = PathHolder(original)
    iterator_created = threading.Event()
    allow_iteration = threading.Event()

    def iterate() -> list[Path]:
        """Create an iterator, pause, and consume its snapshotted directory later."""
        iterator = holder.iterdir()
        iterator_created.set()
        assert allow_iteration.wait(timeout=2)
        return list(iterator)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(iterate)
        assert iterator_created.wait(timeout=2)
        holder.set(tmp_path / "replacement")
        allow_iteration.set()
        assert future.result(timeout=2) == [original / "child"]


def test_path_holder_pickle_preserves_the_path_with_a_fresh_lock(tmp_path):
    """Pickle round trips retain PathHolder behavior without serializing its lock."""
    holder = PathHolder(tmp_path)

    restored = pickle.loads(pickle.dumps(holder))
    restored.set(tmp_path / "other")

    assert holder.get() == tmp_path
    assert restored.get() == tmp_path / "other"
