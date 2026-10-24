"""Provide a bounded, thread-safe mailbox for conversation inputs."""

from dataclasses import replace
from heapq import heapify, heappop, heappush
from queue import Empty, Full
from threading import Condition

from .models import InputAdmission


class InputMailbox:
    """Manage prioritized conversation inputs and atomic shutdown.

    Args:
        capacity (int): Maximum number of inputs accepted before shutdown.

    Raises:
        ValueError: If the mailbox capacity is less than one.
    """

    _capacity: int
    _condition: Condition
    _closed: bool
    _items: list[InputAdmission]

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Input mailbox capacity must be positive.")
        self._capacity = capacity
        self._condition = Condition()
        self._closed = False
        self._items = []

    @property
    def capacity(self) -> int:
        """Return the maximum number of inputs accepted before shutdown.

        Returns:
            int: Maximum pending input count.
        """
        return self._capacity

    def size(self) -> int:
        """Return the number of pending admissions.

        Returns:
            int: Current pending input count.
        """
        with self._condition:
            return len(self._items)

    def is_empty(self) -> bool:
        """Check whether the mailbox has no pending admissions.

        Returns:
            bool: Whether the mailbox is empty.
        """
        with self._condition:
            return not self._items

    def snapshot(self) -> tuple[InputAdmission, ...]:
        """Return pending admissions in dequeue order.

        Returns:
            tuple[InputAdmission, ...]: Immutable priority-ordered snapshot.
        """
        with self._condition:
            return tuple(sorted(self._items))

    def try_put(self, admission: InputAdmission) -> bool:
        """Add an admission unless shutdown has closed the mailbox.

        Args:
            admission (InputAdmission): Admission to add.

        Returns:
            bool: Whether the admission was accepted.

        Raises:
            Full: If the mailbox reached its capacity.
        """
        with self._condition:
            if self._closed:
                return False
            if len(self._items) >= self._capacity:
                raise Full
            heappush(self._items, admission)
            self._condition.notify_all()
        return True

    def get_nowait(self) -> InputAdmission:
        """Remove the highest-priority pending admission.

        Returns:
            InputAdmission: Admission with the lowest numeric priority, then sequence value.

        Raises:
            Empty: If no admission is pending.
        """
        with self._condition:
            if not self._items:
                raise Empty
            return heappop(self._items)

    def wait_until_not_empty(self) -> bool:
        """Wait until an admission arrives or shutdown closes an empty mailbox.

        Returns:
            bool: Whether an admission is available after waking.
        """
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            return bool(self._items)

    def reprioritize(self, target: InputAdmission, priority: int) -> bool:
        """Change a queued admission's priority by object identity.

        Args:
            target (InputAdmission): Exact queued admission to replace.
            priority (int): New priority value.

        Returns:
            bool: Whether the target was found and updated.
        """
        with self._condition:
            index = next(
                (position for position, item in enumerate(self._items) if item is target),
                None,
            )
            if index is None:
                return False
            self._items[index] = replace(target, priority=priority)
            heapify(self._items)
        return True

    def stop_and_replace(self, stop_admission: InputAdmission) -> int:
        """Close admission, discard pending inputs, and enqueue the stop signal.

        Args:
            stop_admission (InputAdmission): Shutdown signal to enqueue.

        Returns:
            int: Number of pending admissions discarded.
        """
        with self._condition:
            if self._closed:
                return 0
            self._closed = True
            discarded = len(self._items)
            self._items.clear()
            heappush(self._items, stop_admission)
            self._condition.notify_all()
        return discarded
