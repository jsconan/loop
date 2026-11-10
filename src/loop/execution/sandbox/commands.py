"""Register management commands offered by the selected native backend."""

from collections.abc import Callable

from ...commands import CommandRegistration
from .selection import selected_backend_module


class SandboxCommands:
    """Expose the host backend's management commands without owning their workflow.

    Args:
        main_app (bool): Whether this is the direct interactive application.
        refresh_backend (Callable[[], None] | None): Reselect native execution authority after
            an approved management change, or None when no execution service is composed.
    """

    _main_app: bool
    _refresh_backend: Callable[[], None] | None

    def __init__(
        self,
        *,
        main_app: bool = False,
        refresh_backend: Callable[[], None] | None = None,
    ) -> None:
        self._main_app = main_app
        self._refresh_backend = refresh_backend

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the selected host backend's management command registrations.

        Returns:
            tuple[CommandRegistration, ...]: Commands supported by this host backend.
        """
        backend_module = selected_backend_module()
        return (
            backend_module.management_commands(self._main_app, self._refresh_backend)
            if backend_module is not None
            else ()
        )
