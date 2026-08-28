"""Tests for general scalar and key normalization utilities."""

from loop.utils import normalized_key, safe_scalar


def test_normalized_key_collapses_punctuation_and_case():
    """Key normalization produces stable lowercase underscore-separated names."""
    assert normalized_key("  API--Key.Value  ") == "api_key_value"


def test_safe_scalar_preserves_scalars_and_sanitizes_other_values():
    """Scalar normalization bounds strings and avoids arbitrary object conversion."""
    assert safe_scalar(None) is None
    assert safe_scalar(3) == 3
    assert safe_scalar("a\r\nb", max_string_length=5) == "a\\r\\n"
    assert safe_scalar(object()) == "object"
