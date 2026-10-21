"""Verify UUID generation across Python versions."""

import uuid
from unittest.mock import Mock
from uuid import uuid4

from loop.utils import new_uuid


def test_new_uuid_uses_runtime_uuid7_when_available(monkeypatch):
    """Runtimes with UUID7 retain the time-ordered identifier behavior."""
    generated = uuid4()
    uuid7 = Mock(return_value=generated)
    monkeypatch.setattr(uuid, "uuid7", uuid7, raising=False)

    assert new_uuid() is generated
    uuid7.assert_called_once_with()


def test_new_uuid_falls_back_to_uuid4_without_runtime_uuid7(monkeypatch):
    """Older runtimes generate a standard UUID4 without importing an unavailable API."""
    monkeypatch.delattr(uuid, "uuid7", raising=False)

    assert new_uuid().version == 4
