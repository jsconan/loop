"""Apply centralized disclosure and model-input secret policies."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from ..utils import normalized_key, safe_scalar
from .models import TelemetryValue

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credentials",
        "password",
        "private_key",
        "secret",
        "set_cookie",
        "token",
    }
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?"
        r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----"
    ),
    re.compile(r"\b(?:sk|gh[opusr])_[A-Za-z0-9_-]{16,}\b"),
)
_OPERATIONAL_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "arguments",
        "context",
        "exception_message",
        "instructions",
        "output",
        "payload",
        "prompt",
        "reasoning",
        "response",
        "result",
        "stacktrace",
        "tool_definitions",
    }
)


class ModelInputPolicy:
    """Remove registered and recognizable credentials before model submission.

    Args:
        secrets (tuple[str, ...]): Exact non-empty secret values to replace in model-visible data.
        replacement (str): Stable marker substituted for every detected secret.
    """

    def __init__(
        self,
        secrets: tuple[str, ...] = (),
        replacement: str = "<redacted:secret>",
    ) -> None:
        self._secrets = tuple(secret for secret in secrets if secret)
        self._replacement = replacement

    def apply(self, value: object) -> object:
        """Return model-visible data with known credentials replaced.

        Args:
            value (object): Serialized model request value to inspect recursively.

        Returns:
            object: Structurally equivalent value safe to submit and trace.
        """
        return self._apply(value, key=None)

    def _apply(self, value: object, *, key: str | None) -> object:
        """Return a structurally equivalent value with known credentials replaced."""
        if key is not None and normalized_key(key) in _SENSITIVE_KEYS:
            return self._replacement
        if isinstance(value, str):
            result = value
            for secret in self._secrets:
                result = result.replace(secret, self._replacement)
            for pattern in _CREDENTIAL_PATTERNS:
                result = pattern.sub(self._replacement, result)
            return result
        if isinstance(value, Mapping):
            return {
                str(item_key): self._apply(item, key=str(item_key))
                for item_key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            converted = [self._apply(item, key=None) for item in value]
            return tuple(converted) if isinstance(value, tuple) else converted
        return value


class OperationalDisclosurePolicy:
    """Normalize minimized operational attributes without arbitrary content."""

    def normalize(self, attributes: Mapping[str, object]) -> MappingProxyType:
        """Return immutable allowlisted operational attributes.

        Args:
            attributes (Mapping[str, object]): Structured diagnostic metadata.

        Returns:
            MappingProxyType: Immutable safe attribute collection.
        """
        normalized = {}
        for key, value in attributes.items():
            safe_key = normalized_key(key)
            if safe_key in _SENSITIVE_KEYS or safe_key in _OPERATIONAL_FORBIDDEN_KEYS:
                continue
            normalized[str(key)] = safe_scalar(value)
        return MappingProxyType(normalized)


def freeze(value: object) -> TelemetryValue:
    """Convert supported structured data into recursively immutable telemetry values.

    Args:
        value (object): Value already governed by its disclosure policy.

    Returns:
        TelemetryValue: Immutable normalized representation.

    Raises:
        TypeError: If the value cannot be represented without arbitrary string conversion.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return freeze(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return freeze(asdict(value))
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    raise TypeError(f"Unsupported telemetry value type: {type(value).__qualname__}")


def thaw(value: TelemetryValue) -> Any:
    """Return a JSON-serializable copy of an immutable telemetry value.

    Args:
        value (TelemetryValue): Immutable normalized value.

    Returns:
        Any: JSON-compatible scalar, list, or dictionary.
    """
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value
