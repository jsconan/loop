"""Apply centralized disclosure and model-input secret policies."""

from __future__ import annotations

import re
from base64 import b64decode, urlsafe_b64decode
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from json import JSONDecodeError, loads
from math import log2
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
_MIN_REGISTERED_SECRET_LENGTH = 16
_MIN_OPAQUE_CREDENTIAL_LENGTH = 20
_MAX_OPAQUE_CREDENTIAL_LENGTH = 2_048
_MIN_CREDENTIAL_ENTROPY = 3.0
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?"
    r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----"
)
_GENERIC_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:[A-Za-z0-9]{1,64}[_.-]){0,4}"
    r"(?:api[_-]?key|apikey|token|secret|password|credential|authorization)"
    r"(?:[_.-][A-Za-z0-9]{1,64}){0,4}[\"']?\s*(?::=|=>|<-|=|:)\s*[\"']?"
    rf"(?P<credential>(?:[A-Za-z0-9_~+/-][A-Za-z0-9._~+/-]{{14,2046}}"
    r"[A-Za-z0-9_~+/-]|"
    rf"[A-Za-z0-9_~+/-][A-Za-z0-9._~+/-]{{13,{_MAX_OPAQUE_CREDENTIAL_LENGTH - 34}}}"
    r"[A-Za-z0-9_~+/-]={1,32}))"
    r"(?=$|[\s,;&/#?\"'`)\]}]|\.(?=$|\s))"
)
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)\b(?:proxy-)?authorization[\"']?\s*(?::=|=>|<-|=|:)\s*"
    r"(?:[\"']\s*)?"
    r"(?P<scheme>bearer|basic)[ \t]+"
    rf"(?P<credential>(?:[A-Za-z0-9._~+/-]{{0,{_MAX_OPAQUE_CREDENTIAL_LENGTH - 1}}}"
    r"[A-Za-z0-9_~+/-]|"
    rf"[A-Za-z0-9._~+/-]{{0,{_MAX_OPAQUE_CREDENTIAL_LENGTH - 33}}}"
    r"[A-Za-z0-9_~+/-]={1,32}))"
    r"(?=$|[\s,;\"'`)\]}]|\.(?=$|\s))"
)
_PROVIDER_CREDENTIAL_PATTERNS = (
    (
        "openai_api_key",
        re.compile(
            r"(?<![A-Za-z0-9_-])(?P<credential>sk-"
            r"(?:(?:proj|svcacct|admin|None)-)?[A-Za-z0-9_-]{40,})"
            r"(?![A-Za-z0-9_-]|\.(?=\S))"
        ),
    ),
    (
        "github_token",
        re.compile(
            r"(?<![A-Za-z0-9_-])(?P<credential>gh[opusr]_[A-Za-z0-9]{36})"
            r"(?![A-Za-z0-9_-]|\.(?=\S))"
        ),
    ),
    (
        "github_fine_grained_token",
        re.compile(
            r"(?<![A-Za-z0-9_-])(?P<credential>github_pat_[A-Za-z0-9_]{82})"
            r"(?![A-Za-z0-9_-]|\.(?=\S))"
        ),
    ),
)
_MAX_WEAK_SECRET_CONTEXT_LENGTH = 256
_WEAK_SECRET_PREFIX_PATTERN = re.compile(
    r"(?i)(?:\b(?:[A-Za-z0-9]+[_.-])*"
    r"(?:api[_-]?key|apikey|token|secret|password|credential|authorization)"
    r"(?:[_.-][A-Za-z0-9]+)*[\"']?\s*(?::=|=>|<-|=|:)\s*[\"']?"
    r"(?:(?:bearer|basic)[ \t]+)?|"
    r"(?:\b(?:proxy-)?authorization[\"']?\s*(?::=|=>|<-|=|:)\s*[\"']?"
    r"(?:bearer|basic)[ \t]+))$"
)
_WEAK_SECRET_END_PATTERN = re.compile(r"(?:$|[\s,;&/#?\"'`)\]}]|\.(?=$|\s))")
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
    """Remove high-confidence registered and recognizable credentials before submission.

    Registered values of at least 16 characters are replaced exactly. Shorter values are replaced
    only in explicit credential assignments, including serialized JSON and authorization headers,
    because ordinary-language matches are ambiguous. Generic Basic and Bearer detection likewise
    requires an explicit authorization header and a 20-character opaque-token minimum;
    provider-specific formats and registered values do not. Bare authentication prose does not
    establish a credential assignment for a short registered value.
    Authorization candidates are capped at 2,048 characters to keep matching time bounded; longer
    opaque credentials require registration for exact replacement. Weak-secret assignment context
    is limited to the preceding 256 characters for the same reason.
    The backend omits its documented built-in placeholder before constructing this policy.

    Args:
        secrets (tuple[str, ...]): Non-empty credential values to protect according to confidence.
        replacement (str): Stable marker substituted for every detected secret.
        reporter (Callable[[Mapping[str, int]], None] | None): Optional content-free callback that
            receives redaction counts grouped by detector. Defaults to ``None``.
    """

    def __init__(
        self,
        secrets: tuple[str, ...] = (),
        replacement: str = "<redacted:secret>",
        reporter: Callable[[Mapping[str, int]], None] | None = None,
    ) -> None:
        candidates = tuple(
            sorted(dict.fromkeys(secret for secret in secrets if secret), key=len, reverse=True)
        )
        self._strong_secrets = tuple(
            secret for secret in candidates if len(secret) >= _MIN_REGISTERED_SECRET_LENGTH
        )
        self._weak_secrets = tuple(
            secret for secret in candidates if len(secret) < _MIN_REGISTERED_SECRET_LENGTH
        )
        self._replacement = replacement
        self._reporter = reporter

    def apply(self, value: object) -> object:
        """Return model-visible data with known credentials replaced.

        Args:
            value (object): Serialized model request value to inspect recursively.

        Returns:
            object: Structurally equivalent value safe to submit and trace.
        """
        counts = {}
        prepared = self._apply(value, counts=counts)
        if counts and self._reporter is not None:
            try:
                self._reporter(MappingProxyType(counts))
            except Exception:  # noqa: BLE001  # Observability must not block model submission.
                return prepared
        return prepared

    def _apply(self, value: object, *, counts: dict[str, int]) -> object:
        """Return a structurally equivalent value with known credentials replaced."""
        if isinstance(value, str):
            return self._redact_text(value, counts)
        if isinstance(value, Mapping):
            return {
                str(item_key): self._apply(item, counts=counts) for item_key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            converted = [self._apply(item, counts=counts) for item in value]
            return tuple(converted) if isinstance(value, tuple) else converted
        return value

    def _redact_text(self, value: str, counts: dict[str, int]) -> str:
        """Return text with only high-confidence credential matches replaced."""
        result = value
        for secret in self._strong_secrets:
            occurrences = result.count(secret)
            if occurrences:
                result = result.replace(secret, self._replacement)
                counts["registered_exact"] = counts.get("registered_exact", 0) + occurrences
        for secret in self._weak_secrets:
            result = self._redact_weak_secret(result, secret, counts)
        result = self._replace_matches(
            result, _PRIVATE_KEY_PATTERN, "private_key", counts, validator=None
        )
        result = self._replace_matches(
            result,
            _GENERIC_SECRET_ASSIGNMENT_PATTERN,
            "generic_assignment",
            counts,
            validator=lambda match: self._valid_opaque_credential(match.group("credential")),
        )
        result = self._replace_matches(
            result,
            _AUTHORIZATION_PATTERN,
            "authorization",
            counts,
            validator=self._valid_authorization,
        )
        for detector, pattern in _PROVIDER_CREDENTIAL_PATTERNS:
            result = self._replace_matches(
                result,
                pattern,
                detector,
                counts,
                validator=self._valid_provider_credential,
            )
        return result

    def _redact_weak_secret(self, value: str, secret: str, counts: dict[str, int]) -> str:
        """Replace one weak registered value only after a bounded credential assignment."""
        parts = []
        position = 0
        while (start := value.find(secret, position)) >= 0:
            end = start + len(secret)
            context = value[max(0, start - _MAX_WEAK_SECRET_CONTEXT_LENGTH) : start]
            if _WEAK_SECRET_END_PATTERN.match(value, end) and _WEAK_SECRET_PREFIX_PATTERN.search(
                context
            ):
                parts.extend((value[position:start], self._replacement))
                counts["registered_context"] = counts.get("registered_context", 0) + 1
            else:
                parts.append(value[position:end])
            position = end
        if not parts:
            return value
        parts.append(value[position:])
        return "".join(parts)

    def _replace_matches(
        self,
        value: str,
        pattern: re.Pattern[str],
        detector: str,
        counts: dict[str, int],
        *,
        validator: Callable[[re.Match[str]], bool] | None,
    ) -> str:
        """Replace validated pattern matches and update one detector count."""

        def replacement(match: re.Match[str]) -> str:
            if validator is not None and not validator(match):
                return match.group(0)
            counts[detector] = counts.get(detector, 0) + 1
            credential = match.groupdict().get("credential")
            if credential is None:
                return self._replacement
            return match.group(0).replace(credential, self._replacement)

        return pattern.sub(replacement, value)

    @staticmethod
    def _valid_authorization(match: re.Match[str]) -> bool:
        """Return whether an authorization candidate has credential structure and strength."""
        scheme = match.group("scheme").casefold()
        credential = match.group("credential")
        if scheme == "basic":
            if "=" in credential and len(credential) % 4:
                return False
            try:
                decoded = b64decode(
                    credential
                    if "=" in credential
                    else f"{credential}{'=' * (-len(credential) % 4)}",
                    validate=True,
                ).decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                return False
            username, separator, password = decoded.partition(":")
            return bool(separator and username and password)
        if credential.count(".") == 2:
            return ModelInputPolicy._valid_jwt(credential)
        return ModelInputPolicy._valid_opaque_credential(credential)

    @staticmethod
    def _valid_opaque_credential(value: str) -> bool:
        """Return whether an opaque credential has conservative generic-secret properties."""
        return (
            len(value) >= _MIN_OPAQUE_CREDENTIAL_LENGTH
            and ModelInputPolicy._entropy(value) >= _MIN_CREDENTIAL_ENTROPY
            and any(character.isalpha() for character in value)
            and any(character.isdigit() for character in value)
        )

    @staticmethod
    def _valid_provider_credential(match: re.Match[str]) -> bool:
        """Return whether a provider-shaped credential has sufficient entropy."""
        return ModelInputPolicy._entropy(match.group("credential")) >= _MIN_CREDENTIAL_ENTROPY

    @staticmethod
    def _valid_jwt(value: str) -> bool:
        """Return whether text has decodable JWT header, payload, and signature segments."""
        segments = value.split(".")
        if len(segments) != 3 or not all(
            segment and re.fullmatch(r"[A-Za-z0-9_-]+", segment) for segment in segments
        ):
            return False
        try:
            header, payload = (
                loads(urlsafe_b64decode(f"{segment}{'=' * (-len(segment) % 4)}"))
                for segment in segments[:2]
            )
        except (JSONDecodeError, UnicodeDecodeError, ValueError):
            return False
        algorithm = header.get("alg") if isinstance(header, dict) else None
        try:
            signature = urlsafe_b64decode(f"{segments[2]}{'=' * (-len(segments[2]) % 4)}")
        except ValueError:
            return False
        return (
            isinstance(algorithm, str)
            and bool(algorithm)
            and algorithm.casefold() != "none"
            and isinstance(payload, dict)
            and len(signature) >= 32
            and ModelInputPolicy._entropy(segments[2]) >= _MIN_CREDENTIAL_ENTROPY
        )

    @staticmethod
    def _entropy(value: str) -> float:
        """Return Shannon entropy in bits per character for one candidate value."""
        return -sum(
            (count / len(value)) * log2(count / len(value))
            for count in (value.count(character) for character in set(value))
        )


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
