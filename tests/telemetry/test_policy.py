"""Tests for centralized observability disclosure policies."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from loop.telemetry.policy import ModelInputPolicy, OperationalDisclosurePolicy, freeze, thaw


class Value(BaseModel):
    """Provide one structured value for normalization tests."""

    name: str


@dataclass
class DataclassValue:
    """Provide one dataclass value for normalization tests."""

    enabled: bool


def test_model_input_policy_redacts_registered_structured_and_formatted_secrets():
    """Model preparation removes exact, structured, bearer, token, and private-key secrets."""
    policy = ModelInputPolicy(("registered-secret",))
    source = {
        "authorization": "anything",
        "nested": [
            "registered-secret Bearer abc.def.ghi sk_abcdefghijklmnopqrstuvwxyz",
            "-----BEGIN PRIVATE KEY-----\nvalue\n-----END PRIVATE KEY-----",
        ],
        "tuple": ("safe",),
    }

    prepared = policy.apply(source)

    assert prepared["authorization"] == "<redacted:secret>"
    assert "registered-secret" not in prepared["nested"][0]
    assert "Bearer" not in prepared["nested"][0]
    assert "sk_" not in prepared["nested"][0]
    assert "PRIVATE KEY" not in prepared["nested"][1]
    assert prepared["tuple"] == ("safe",)


def test_operational_policy_minimizes_fields_and_sanitizes_lines():
    """Operational metadata excludes content and secrets while retaining safe scalar facts."""
    attributes = OperationalDisclosurePolicy().normalize(
        {
            "prompt": "private",
            "api-key": "credential",
            "component": "backend\nforged",
            "count": 2,
            "exception": RuntimeError("private"),
        }
    )

    assert dict(attributes) == {
        "component": "backend\\nforged",
        "count": 2,
        "exception": "RuntimeError",
    }


def test_freeze_and_thaw_support_models_dataclasses_and_nested_values():
    """Telemetry normalization creates immutable structures and reversible JSON values."""
    frozen = freeze(
        {
            "model": Value(name="value"),
            "dataclass": DataclassValue(enabled=True),
            "items": [1, None],
        }
    )

    assert thaw(frozen) == {
        "model": {"name": "value"},
        "dataclass": {"enabled": True},
        "items": [1, None],
    }


def test_freeze_rejects_arbitrary_objects():
    """Normalization never invokes arbitrary object string conversion."""
    with pytest.raises(TypeError, match="Unsupported telemetry value type"):
        freeze(object())
