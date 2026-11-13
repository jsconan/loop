"""Verify safe permission YAML parsing and duplicate-key detection."""

import pytest
from ruamel.yaml import YAMLError

from loop.utils.yaml import dump_yaml, load_yaml


def test_dump_yaml_round_trips_ambiguous_scalars():
    """Saved strings do not become booleans, numbers, or keys of another type."""
    value = {"y": "yes", "n": "no", "on": "off", "octal": "012", "time": "1:20"}
    assert load_yaml(dump_yaml(value)) == value


def test_load_permission_yaml_accepts_safe_mappings_and_merge_overrides():
    """Explicit overrides of YAML merges remain supported."""
    assert load_yaml("base: &b {allow: false}\npolicy: {<<: *b, allow: true}\n") == {
        "base": {"allow": False},
        "policy": {"allow": True},
    }
    assert load_yaml("") is None


@pytest.mark.parametrize(
    "content",
    [
        "key: 1\nkey: 2\n",
        "nested: {key: 1, key: 2}\n",
        "limits: {<<: {deny_local_networks: true, deny_local_networks: false}}\n",
        "base: &b {a: 1}\nlimits: {<<: *b, <<: *b}\n",
    ],
)
def test_load_permission_yaml_rejects_duplicate_keys(content):
    """Ambiguous explicit keys are rejected at every mapping depth."""
    with pytest.raises(YAMLError, match="duplicate key"):
        load_yaml(content)


def test_load_permission_yaml_accepts_sequence_keys():
    """YAML sequence keys are represented as immutable tuples."""
    assert load_yaml("? [a, b]\n: value\n") == {("a", "b"): "value"}


def test_load_permission_yaml_handles_recursive_aliases():
    """Recursive aliases do not make duplicate-key inspection recurse indefinitely."""
    result = load_yaml("root: &root {child: *root}\n")
    assert result["root"]["child"] is result["root"]


def test_load_yaml_uses_the_document_version_for_scalar_values():
    """Both YAML constructors use the same effective document version."""
    assert load_yaml("octal: 012\ntime: 1:20\nyes: no\n") == {
        "octal": 12,
        "time": "1:20",
        "yes": "no",
    }
    assert load_yaml("%YAML 1.1\n---\noctal: 012\ntime: 1:20\n") == {
        "octal": 10,
        "time": 80,
    }
    assert load_yaml("%YAML 1.2\n---\noctal: 012\ntime: 1:20\n") == {
        "octal": 12,
        "time": "1:20",
    }


def test_load_yaml_redacts_invalid_tagged_values():
    """Invalid scalar tags do not expose the scalar value in diagnostics."""
    with pytest.raises(YAMLError) as raised:
        load_yaml("limits: {deny_local_networks: !!int synthetic-secret}")
    assert "synthetic-secret" not in str(raised.value)


def test_load_permission_yaml_reports_excessive_nesting():
    """Excessively nested input is a validation failure rather than a runtime crash."""
    with pytest.raises(YAMLError, match="nesting exceeds"):
        load_yaml("[" * 2000 + "0" + "]" * 2000)
