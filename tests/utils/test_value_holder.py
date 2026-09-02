"""Tests for mutable scalar value holders."""

import copy
import operator
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from loop.utils import (
    BoolValueHolder,
    FloatValueHolder,
    IntValueHolder,
    PathHolder,
    StrValueHolder,
    ValueHolder,
)


def test_value_holder_exposes_one_value_through_properties_and_methods():
    """The generic holder preserves values and shares storage across its public API."""
    initial = object()
    replacement = object()
    holder = ValueHolder(initial)

    assert holder.value is initial
    assert holder.get() is initial

    holder.value = replacement
    assert holder.get() is replacement

    assert holder.set(initial) is None
    assert holder.value is initial


@pytest.mark.parametrize(
    ("holder", "expected", "value_type"),
    [
        (StrValueHolder(123), "123", str),
        (IntValueHolder("42"), 42, int),
        (FloatValueHolder("3.14"), 3.14, float),
        (BoolValueHolder(1), True, bool),
    ],
)
def test_typed_holders_coerce_values_during_construction(holder, expected, value_type):
    """Every typed holder stores its constructor input as the declared scalar type."""
    assert holder.value == expected
    assert isinstance(holder.value, value_type)


@pytest.mark.parametrize(
    ("holder", "assigned", "set_value", "expected_assigned", "expected_set"),
    [
        (StrValueHolder(""), 123, False, "123", "False"),
        (IntValueHolder(0), "42", 3.9, 42, 3),
        (FloatValueHolder(0), "3.14", 2, 3.14, 2.0),
        (BoolValueHolder(False), "false", "", True, False),
    ],
)
def test_typed_holders_coerce_property_and_method_mutations(
    holder, assigned, set_value, expected_assigned, expected_set
):
    """Property assignment and set apply the same subclass coercion hook."""
    holder.value = assigned
    assert holder.value == expected_assigned

    holder.set(set_value)
    assert holder.get() == expected_set


def test_value_holder_retains_identity_and_shares_mutations():
    """Mutation is visible through retained references without replacing the holder."""
    holder = IntValueHolder(1)
    consumer_reference = holder
    identity = id(holder)

    holder.value = 2
    assert id(holder) == identity
    assert int(holder) == 2

    holder.set(3)
    assert consumer_reference.value == 3


def test_value_holder_string_boolean_and_representation_use_current_value():
    """Generic conversions expose the value while repr identifies the runtime holder class."""
    holder = ValueHolder("x")

    assert str(holder) == "x"
    assert repr(holder) == "ValueHolder('x')"
    assert bool(holder)

    holder.value = ""
    assert not holder
    assert repr(holder) == "ValueHolder('')"


def test_typed_holder_scalar_conversions_use_current_values():
    """Integer, index, float, string, and truth conversions follow their scalar values."""
    integer = IntValueHolder("42")
    floating = FloatValueHolder("3.14")
    string = StrValueHolder(123)
    boolean = BoolValueHolder(0)

    assert int(integer) == 42
    assert operator.index(integer) == 42
    assert float(floating) == 3.14
    assert str(string) == "123"
    assert not boolean


def test_value_holder_equality_tracks_values_and_python_cross_type_semantics():
    """Equality compares current values with holders and raw values using Python semantics."""
    integer = IntValueHolder(1)
    same = IntValueHolder("1")

    assert integer == 1
    assert integer == same
    assert integer == True
    assert integer == 1.0
    assert integer != 2

    same.value = 2
    assert integer != same
    assert same == 2


def test_value_holder_ordering_delegates_to_current_underlying_values():
    """Ordering unwraps holders and preserves the wrapped values' native behavior."""
    integer = IntValueHolder(1)
    floating = FloatValueHolder(2.0)

    assert integer < floating
    assert integer <= 1
    assert floating > integer
    assert floating >= 2.0

    integer.set(3)
    assert integer > floating

    with pytest.raises(TypeError):
        _ = integer < "3"


@pytest.mark.parametrize(
    "holder",
    [
        ValueHolder("value"),
        StrValueHolder("value"),
        IntValueHolder(1),
        FloatValueHolder(1),
        BoolValueHolder(1),
    ],
)
def test_value_holders_are_unhashable(holder):
    """Mutable holders cannot violate hash-container invariants after value changes."""
    with pytest.raises(TypeError, match="unhashable type"):
        hash(holder)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", False), ("false", True), (0, False), (1, True)],
)
def test_bool_value_holder_uses_standard_truth_value_coercion(value, expected):
    """Boolean coercion treats strings and numbers exactly as the bool builtin does."""
    assert BoolValueHolder(value).value is expected


@pytest.mark.parametrize(
    ("holder_type", "value", "exception_type"),
    [(IntValueHolder, "not-an-int", ValueError), (FloatValueHolder, object(), TypeError)],
)
def test_typed_holders_propagate_builtin_conversion_errors(holder_type, value, exception_type):
    """Invalid typed values expose the normal exception from the conversion builtin."""
    with pytest.raises(exception_type):
        holder_type(value)


def test_failed_set_and_update_preserve_the_previous_value():
    """Failed coercion and callbacks leave a typed holder unchanged."""
    holder = IntValueHolder(10)

    with pytest.raises(ValueError):
        holder.set("invalid")
    assert holder.get() == 10

    with pytest.raises(ValueError):
        holder.update(lambda _current: "invalid")
    assert holder.get() == 10

    with pytest.raises(RuntimeError, match="failed"):
        holder.update(lambda _current: (_ for _ in ()).throw(RuntimeError("failed")))
    assert holder.get() == 10


def test_update_is_atomic_across_concurrent_workers():
    """Concurrent atomic increments cannot lose read-modify-write updates."""
    holder = IntValueHolder(0)
    workers = 8
    increments = 500

    def increment() -> None:
        """Increment the shared holder repeatedly through its atomic API."""
        for _ in range(increments):
            holder.update(lambda current: current + 1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _worker: increment(), range(workers)))

    assert holder.get() == workers * increments


def test_shared_reads_writes_and_conversions_remain_consistent():
    """Readers observe complete coerced values while another thread replaces them."""
    holder = IntValueHolder(0)
    barrier = threading.Barrier(5)

    def write(seed: int) -> None:
        """Replace the shared value after all workers are ready."""
        barrier.wait()
        for offset in range(500):
            holder.value = seed + offset

    def read() -> None:
        """Read and convert complete integer snapshots during replacement."""
        barrier.wait()
        for _ in range(500):
            assert isinstance(holder.get(), int)
            assert isinstance(holder.value, int)
            assert isinstance(int(holder), int)
            assert isinstance(str(holder), str)
            assert holder == holder.get() or holder != holder.get()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write, seed * 1_000) for seed in range(2)]
        futures.extend(executor.submit(read) for _ in range(3))
        for future in futures:
            future.result()


def test_opposite_order_holder_comparisons_do_not_deadlock():
    """Two threads can compare the same holders in opposite orders without lock nesting."""
    first = IntValueHolder(1)
    second = IntValueHolder(1)
    barrier = threading.Barrier(2)

    def compare(left: IntValueHolder, right: IntValueHolder) -> bool:
        """Start with the peer and compare repeatedly in one operand order."""
        barrier.wait()
        return all(left == right for _ in range(1_000))

    with ThreadPoolExecutor(max_workers=2) as executor:
        forward = executor.submit(compare, first, second)
        reverse = executor.submit(compare, second, first)
        assert forward.result(timeout=2)
        assert reverse.result(timeout=2)


def test_compare_and_set_atomically_matches_and_coerces_replacements():
    """Compare-and-set changes only a matching value and preserves it on errors."""
    holder = IntValueHolder(10)

    assert holder.compare_and_set(10, "20") is True
    assert holder.get() == 20
    assert holder.compare_and_set(10, 30) is False
    assert holder.get() == 20
    with pytest.raises(ValueError):
        holder.compare_and_set(20, "invalid")
    assert holder.get() == 20


@pytest.mark.parametrize(
    "holder",
    [
        ValueHolder("value"),
        StrValueHolder("value"),
        IntValueHolder(1),
        FloatValueHolder(1),
        BoolValueHolder(1),
    ],
)
def test_value_holders_copy_and_pickle_with_independent_synchronization(holder):
    """Copying and pickling preserve values without attempting to serialize locks."""
    replicas = [copy.copy(holder), copy.deepcopy(holder), pickle.loads(pickle.dumps(holder))]

    holder.set(holder.get())

    assert all(replica is not holder and replica == holder for replica in replicas)


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


def test_path_holder_derived_paths_are_path_snapshots():
    """Path-producing lexical and normalization operations return immutable snapshots."""
    holder = PathHolder("/old/report.txt")
    snapshots = [
        holder / "child",
        "parent" / holder,
    ]

    holder.set("/new/report.txt")

    assert all(isinstance(path, Path) for path in snapshots)
    assert snapshots[0] == Path("/old/report.txt/child")
    assert snapshots[1] == Path("parent/old/report.txt")
