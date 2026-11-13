"""Coordinate shared command authorities while jobs and session setup overlap."""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from threading import Lock, get_ident


class CommandAuthorityConflict(RuntimeError):
    """Report an attempt to change authority still owned by a command."""


class CommandAuthorityLease:
    """Release a shared-authority reservation exactly once after its owner settles.

    Args:
        owner (CommandAuthorities): Coordinator that owns the reservation.
        resources (tuple[object, ...]): Strong references to reserved authorities.

    Attributes:
        resources (tuple[object, ...]): Authorities retained until the lease settles.
        released (bool): Release state, guarded by the coordinator lock.
    """

    _owner: "CommandAuthorities"
    resources: tuple[object, ...]
    released: bool

    def __init__(self, owner: "CommandAuthorities", resources: tuple[object, ...]) -> None:
        self._owner = owner
        self.resources = resources
        self.released = False

    def release(self) -> None:
        """Release the reservation if another cleanup path has not already done so."""
        self._owner.release(self)


class CommandAuthorities:
    """Track active command and composition leases by shared object identity."""

    _lock: Lock
    _active: dict[int, tuple[object, int]]
    _composing: dict[int, tuple[object, int, int]]

    def __init__(self) -> None:
        self._lock = Lock()
        self._active = {}
        self._composing = {}

    @staticmethod
    def _resources(resources: Iterable[object | None]) -> tuple[object, ...]:
        """Retain each non-null object once, regardless of equality semantics."""
        return tuple(
            {id(resource): resource for resource in resources if resource is not None}.values()
        )

    def reserve(self, resources: Iterable[object | None]) -> CommandAuthorityLease:
        """Reserve shared authorities for a job until its worker has settled.

        Args:
            resources (Iterable[object | None]): Actual objects used by the command.

        Returns:
            CommandAuthorityLease: Idempotent reservation holding the selected objects.

        Raises:
            ValueError: Session composition currently owns one of the authorities.
        """
        selected = self._resources(resources)
        with self._lock:
            if any(id(resource) in self._composing for resource in selected):
                raise ValueError("Command authority is being composed.")
            for resource in selected:
                identity = id(resource)
                previous = self._active.get(identity)
                self._active[identity] = (resource, previous[1] + 1 if previous else 1)
        return CommandAuthorityLease(self, selected)

    def release(self, lease: CommandAuthorityLease) -> None:
        """Release one acquired job lease after its worker has stopped using authorities.

        Args:
            lease (CommandAuthorityLease): Reservation returned by ``reserve``.
        """
        with self._lock:
            if lease.released:
                return
            lease.released = True
            for resource in lease.resources:
                identity = id(resource)
                current = self._active[identity]
                if current[1] == 1:
                    del self._active[identity]
                else:
                    self._active[identity] = (resource, current[1] - 1)

    @contextmanager
    def composition(self, resources: Iterable[object | None]) -> Generator[None, None, None]:
        """Exclude command reservations and other composition while bindings change.

        Args:
            resources (Iterable[object | None]): Actual objects to bind to the session.

        Yields:
            None: Control while the selected authorities are reserved for composition.

        Raises:
            RuntimeError: A command or another composition owns a selected authority.
        """
        selected = self._resources(resources)
        owner = get_ident()
        with self._lock:
            if any(
                id(resource) in self._active
                or (id(resource) in self._composing and self._composing[id(resource)][1] != owner)
                for resource in selected
            ):
                raise CommandAuthorityConflict(
                    "Cannot compose the session while commands are active."
                )
            for resource in selected:
                identity = id(resource)
                previous = self._composing.get(identity)
                self._composing[identity] = (resource, owner, previous[2] + 1 if previous else 1)
        try:
            yield
        finally:
            with self._lock:
                for resource in selected:
                    identity = id(resource)
                    current = self._composing[identity]
                    if current[2] == 1:
                        del self._composing[identity]
                    else:
                        self._composing[identity] = (resource, owner, current[2] - 1)


command_authorities = CommandAuthorities()
