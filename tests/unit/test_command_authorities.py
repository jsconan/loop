"""Tests for shared command-authority reservations and session composition."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from loop.command_authorities import CommandAuthorities, CommandAuthorityConflict


def test_reservations_retain_each_identity_and_allow_disjoint_composition():
    """A job protects each actual authority once, including unhashable equal objects."""

    class EqualAuthority:
        """Compare equal without making separate instances the same authority."""

        __hash__ = None

        def __eq__(self, other):
            return isinstance(other, EqualAuthority)

    authority = CommandAuthorities()
    shared, distinct, unrelated = EqualAuthority(), EqualAuthority(), object()
    lease = authority.reserve((None, shared, shared, distinct))
    try:
        for resource in (shared, distinct):
            with (
                pytest.raises(CommandAuthorityConflict, match="commands are active"),
                authority.composition((resource,)),
            ):
                pytest.fail("Active authority cannot be recomposed")
        with authority.composition((unrelated,)):
            pass
    finally:
        lease.release()
    with authority.composition((shared, distinct)):
        pass


def test_independent_reservations_and_repeated_release():
    """Releasing a lease twice cannot release another job's authority reservation."""
    authority = CommandAuthorities()
    shared = object()
    first = authority.reserve((shared,))
    second = authority.reserve((shared,))
    first.release()
    first.release()
    with (
        pytest.raises(CommandAuthorityConflict, match="commands are active"),
        authority.composition((shared,)),
    ):
        pytest.fail("The second job still owns this authority")
    second.release()
    with authority.composition((shared,)):
        pass


def test_concurrent_release_retains_other_job_reservation():
    """Concurrent cleanup of one lease cannot release another job's authority."""
    authority = CommandAuthorities()
    shared = object()
    first = authority.reserve((shared,))
    second = authority.reserve((shared,))
    barrier = Barrier(3)

    def release_once():
        """Race the two permitted cleanup paths for the first reservation."""
        barrier.wait(timeout=5)
        first.release()

    with ThreadPoolExecutor(max_workers=2) as executor:
        workers = [executor.submit(release_once) for _ in range(2)]
        barrier.wait(timeout=5)
        for worker in workers:
            worker.result(timeout=5)
    with (
        pytest.raises(CommandAuthorityConflict, match="commands are active"),
        authority.composition((shared,)),
    ):
        pytest.fail("The other command still owns this authority")
    second.release()
    with authority.composition((shared,)):
        pass


def test_nested_composition_keeps_authority_reserved_until_outer_exit():
    """Same-thread nested composition remains exclusive through the outermost exit."""
    authority = CommandAuthorities()
    shared, other = object(), object()
    with authority.composition((shared, None, shared)):
        with (
            authority.composition((shared, other)),
            pytest.raises(ValueError, match="Command authority is being composed"),
        ):
            authority.reserve((shared,))
        with pytest.raises(ValueError, match="Command authority is being composed"):
            authority.reserve((shared,))
        lease = authority.reserve((other,))
        lease.release()
    lease = authority.reserve((shared,))
    lease.release()


def test_failed_composition_releases_every_authority():
    """An exception in session setup releases all its composition markers."""
    authority = CommandAuthorities()
    first, second = object(), object()
    with pytest.raises(ValueError, match="fixture failure"), authority.composition((first, second)):
        raise ValueError("fixture failure")
    lease = authority.reserve((first, second))
    lease.release()


def test_conflicting_composition_is_atomic():
    """A rejected multi-authority composition leaves unrelated resources available."""
    authority = CommandAuthorities()
    active, unused = object(), object()
    lease = authority.reserve((active,))
    try:
        with (
            pytest.raises(CommandAuthorityConflict, match="commands are active"),
            authority.composition((unused, active)),
        ):
            pytest.fail("A partial composition must never start")
        other = authority.reserve((unused,))
        other.release()
    finally:
        lease.release()


def test_other_thread_cannot_reserve_or_compose_a_held_authority():
    """An in-progress composition excludes other threads without blocking disjoint work."""
    authority = CommandAuthorities()
    shared, unrelated = object(), object()
    entered, release = Event(), Event()

    def hold_composition():
        """Hold the authority until the calling thread has checked competing work."""
        with authority.composition((shared,)):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Fixture composition was not released")

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(hold_composition)
        try:
            assert entered.wait(5)
            with (
                pytest.raises(CommandAuthorityConflict, match="commands are active"),
                authority.composition((shared,)),
            ):
                pytest.fail("A different thread owns this authority")
            with pytest.raises(ValueError, match="Command authority is being composed"):
                authority.reserve((shared,))
            with authority.composition((unrelated,)):
                pass
        finally:
            release.set()
        worker.result(timeout=5)
    lease = authority.reserve((shared,))
    lease.release()
