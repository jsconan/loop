"""Verify duplicate-free JSON decoding and boundary-specific diagnostics."""

from io import StringIO

import pytest

from loop.utils.json import json_encode, load_unique_json


def test_json_encode_returns_sorted_compact_json():
    """JSON encoding sorts keys recursively and omits optional whitespace."""
    value = {"z": [2, {"b": True, "a": "text"}], "a": 1}

    assert json_encode(value) == '{"a":1,"z":[2,{"a":"text","b":true}]}'


def test_json_encode_writes_encoded_value_to_stream():
    """JSON encoding returns the value and writes the same text to a stream."""
    stream = StringIO()

    result = json_encode({"b": 2, "a": 1}, stream)

    assert result == '{"a":1,"b":2}'
    assert stream.getvalue() == result


def test_unique_json_preserves_values_and_nested_objects():
    """Ordinary JSON retains exact bool/int/null and nested object values."""
    assert load_unique_json('{"a": [true, 1, null, {"b": 2}]}', duplicate_message="duplicate") == {
        "a": [True, 1, None, {"b": 2}]
    }


@pytest.mark.parametrize("data", ['{"a": 1, "a": 2}', '{"nested": {"a": 1, "a": 2}}'])
def test_unique_json_rejects_duplicates_at_every_depth(data):
    """Ambiguous authority fails with the owning boundary's diagnostic."""
    with pytest.raises(ValueError, match="exact boundary diagnostic"):
        load_unique_json(data, duplicate_message="exact boundary diagnostic")


def test_unique_json_rejects_malformed_input():
    """Malformed syntax keeps the standard decoder's error contract."""
    with pytest.raises(ValueError, match="Expecting"):
        load_unique_json("{", duplicate_message="duplicate")
