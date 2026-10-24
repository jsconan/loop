"""Provide a bounded, thread-safe mailbox for conversation inputs."""

from dataclasses import replace
from heapq import heapify, heappop, heappush
from queue import Empty, Full
from threading import Condition

from .models import InputAdmission, InputAdmissionKind


class InputMailbox:
    """Manage prioritized inputs with one reserved command slot and atomic shutdown.

    Args:
        capacity (int): Maximum pending ordinary inputs, plus one reserved command slot.

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
        """Return the ordinary input capacity, excluding the reserved command slot.

        Returns:
            int: Maximum pending ordinary input count.
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
            Full: If ordinary input capacity or the single command slot is full.
        """
        with self._condition:
            if self._closed:
                return False
            commands = sum(item.kind is InputAdmissionKind.COMMAND for item in self._items)
            if (
                commands >= 1
                if admission.kind is InputAdmissionKind.COMMAND
                else len(self._items) - commands >= self._capacity
            ):
                raise Full
            heappush(self._items, admission)
            self._condition.notify_all()
        return True

    def get_nowait(self, *, commands_only: bool = False) -> InputAdmission:
        """Remove the highest-priority pending admission.

        Args:
            commands_only (bool): Park ordinary messages while allowing commands and shutdown.
                Defaults to ``False``.

        Returns:
            InputAdmission: Admission with the lowest numeric priority, then sequence value.

        Raises:
            Empty: If no admission is pending.
        """
        with self._condition:
            if commands_only:
                eligible = [
                    index
                    for index, item in enumerate(self._items)
                    if item.kind is not InputAdmissionKind.MESSAGE
                ]
                if not eligible:
                    raise Empty
                target = self._items.pop(min(eligible, key=lambda index: self._items[index]))
                heapify(self._items)
                return target
            if not self._items:
                raise Empty
            return heappop(self._items)

    def wait_until_not_empty(self, *, commands_only: bool = False) -> bool:
        """Wait until an admission arrives or shutdown closes an empty mailbox.

        Args:
            commands_only (bool): Wait for a command or shutdown while ordinary input is parked.
                Defaults to ``False``.

        Returns:
            bool: Whether an admission is available after waking.
        """
        with self._condition:
            while (
                not any(
                    not commands_only or item.kind is not InputAdmissionKind.MESSAGE
                    for item in self._items
                )
                and not self._closed
            ):
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

    def remove(self, target: InputAdmission) -> bool:
        """Remove a queued admission by object identity.

        Args:
            target (InputAdmission): Exact queued admission to remove.

        Returns:
            bool: Whether the target was present and removed.
        """
        with self._condition:
            index = next(
                (position for position, item in enumerate(self._items) if item is target),
                None,
            )
            if index is None:
                return False
            self._items.pop(index)
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
