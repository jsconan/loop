"""Tests for mutable scalar value holders."""

import operator

import pytest

from loop.utils import (
    BoolValueHolder,
    FloatValueHolder,
    IntValueHolder,
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
