"""Own bounded audience-bound command secrets in application memory."""

from __future__ import annotations

from collections.abc import Mapping

from .. import constants


class ApplicationSecretAuthority:
    """Resolve exact command-secret audiences and erase owned material on cleanup.

    The authority intentionally has no ambient configuration or environment fallback. Callers
    must register each opaque identifier and exact audience through trusted application code.

    Args:
        material (Mapping[tuple[str, str], bytes]): Exact ``(secret_id, audience)`` bindings.
        maximum_secret_bytes (int): Positive maximum material size for any one binding.
    """

    maximum_secret_bytes: int
    _material: dict[tuple[str, str], bytearray]
    _closed: bool

    def __init__(
        self,
        material: Mapping[tuple[str, str], bytes],
        maximum_secret_bytes: int = constants.MAX_SECRET_BYTES,
    ) -> None:
        if maximum_secret_bytes <= 0:
            raise ValueError("Secret material bound must be positive.")
        values: dict[tuple[str, str], bytearray] = {}
        for key, value in material.items():
            if (
                len(key) != 2
                or not all(isinstance(part, str) and part for part in key)
                or not isinstance(value, bytes)
                or not value
                or len(value) > maximum_secret_bytes
            ):
                raise ValueError("Secret authority bindings must be exact and bounded.")
            values[key] = bytearray(value)
        self.maximum_secret_bytes = maximum_secret_bytes
        self._material = values
        self._closed = False

    def resolve(self, secret_id: str, audience: str) -> bytes:
        """Return a bounded copy for one exact identifier and audience.

        Args:
            secret_id (str): Opaque registered secret identifier.
            audience (str): Exact registered audience.

        Returns:
            bytes: Short-lived material copy owned by the broker planner.

        Raises:
            KeyError: If the authority is closed or the exact binding is absent.
        """
        if self._closed:
            raise KeyError((secret_id, audience))
        try:
            value = self._material[(secret_id, audience)]
        except KeyError:
            raise KeyError((secret_id, audience)) from None
        return bytes(value)

    def close(self) -> None:
        """Overwrite all owned mutable buffers and disable future resolution."""
        if self._closed:
            return
        for value in self._material.values():
            value[:] = b"\0" * len(value)
        self._material.clear()
        self._closed = True

    def __repr__(self) -> str:
        """Return a representation that never discloses identifiers or material."""
        return f"{type(self).__name__}(bindings={len(self._material)}, closed={self._closed})"
