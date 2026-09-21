"""Test the application-owned bounded command-secret authority."""

import pytest

from loop.application.secrets import ApplicationSecretAuthority


def test_secret_authority_requires_exact_audience_and_erases_on_close() -> None:
    """Resolve only exact bindings without disclosure and disable them after cleanup."""
    authority = ApplicationSecretAuthority({("token", "api.example"): b"sensitive"})

    assert authority.resolve("token", "api.example") == b"sensitive"
    assert "sensitive" not in repr(authority)
    assert "token" not in repr(authority)
    with pytest.raises(KeyError):
        authority.resolve("token", "other.example")

    authority.close()
    authority.close()
    with pytest.raises(KeyError):
        authority.resolve("token", "api.example")


@pytest.mark.parametrize(
    "material,limit",
    [
        ({("", "audience"): b"value"}, 32),
        ({("secret", "audience"): b""}, 32),
        ({("secret", "audience"): b"too large"}, 2),
        ({("secret", "audience"): b"value"}, 0),
    ],
)
def test_secret_authority_rejects_unbounded_or_ambiguous_material(material, limit) -> None:
    """Reject empty identities, empty values, oversized values, and invalid bounds."""
    with pytest.raises(ValueError, match="bound|exact"):
        ApplicationSecretAuthority(material, limit)
