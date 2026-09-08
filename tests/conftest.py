"""Shared lifecycle fixtures for the ordinary test suite."""

from collections.abc import Callable

import pytest

from loop.permissions import PermissionManager


@pytest.fixture(autouse=True)
def close_permission_managers(monkeypatch):
    """Close temporary directories owned by permission managers created in each test."""
    managers: list[PermissionManager] = []
    initialize: Callable[..., None] = PermissionManager.__init__

    def tracked_initialize(manager: PermissionManager, *args: object, **kwargs: object) -> None:
        initialize(manager, *args, **kwargs)
        managers.append(manager)

    monkeypatch.setattr(PermissionManager, "__init__", tracked_initialize)
    yield
    for manager in reversed(managers):
        manager.close()
