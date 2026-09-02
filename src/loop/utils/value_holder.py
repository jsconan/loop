"""Provide thread-safe mutable indirection around values."""

import os
import threading
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

type PathInput = str | PathLike[str]


class ValueReference[T](Protocol):
    """Provide the current value without exposing mutation."""

    def get(self) -> T:
        """Return the current value.

        Returns:
            T: Current referenced value.
        """


class PathReference(ValueReference[Path], PathLike[str], Protocol):
    """Provide the current filesystem path without exposing mutation."""


class ValueHolder[T]:
    """Hold a mutable value behind a stable object identity.

    Individual reads and replacements are synchronized. Operations use a snapshot of the value
    observed when they begin. Separate reads and writes, including ``holder.value += 1`` and
    ``holder.set(holder.get() + 1)``, are not atomic; use :meth:`update` for an atomic
    read-modify-write operation. The update callback executes while this holder's reentrant lock
    is held and should remain short and non-blocking.

    Comparisons snapshot each holder independently and never hold two holder locks at once. A
    concurrent comparison therefore reflects whichever value was observed from each operand.
    Holders retain object identity across mutation and are deliberately unhashable.

    Typed subclasses add conversion protocols appropriate to their contained type. They remain
    holders rather than subclasses or complete proxies of the wrapped built-in values.

    Args:
        value (T): Initial contained value.
    """

    _lock: threading.RLock
    _value: T
    __hash__ = None

    def __init__(self, value: T) -> None:
        self._lock = threading.RLock()
        self._value = self._coerce(value)

    def _coerce(self, value: T) -> T:
        """Return the value to store."""
        return value

    @property
    def value(self) -> T:
        """Return the current contained value.

        Returns:
            T: Current contained value.
        """
        with self._lock:
            return self._value

    @value.setter
    def value(self, value: T) -> None:
        """Replace the contained value while retaining this holder's identity.

        Args:
            value (T): New value to contain.
        """
        self.set(value)

    def get(self) -> T:
        """Return the current contained value.

        Returns:
            T: Current contained value.
        """
        with self._lock:
            return self._value

    def set(self, value: T) -> None:
        """Replace the contained value while retaining this holder's identity.

        Args:
            value (T): New value to contain.
        """
        coerced = self._coerce(value)
        with self._lock:
            self._value = coerced

    def update(self, function: Callable[[T], T]) -> T:
        """Atomically transform and replace the contained value.

        The callback and coercion execute while this holder's reentrant lock is held. If either
        raises, the previous value remains stored.

        Args:
            function (Callable[[T], T]): Short, non-blocking transformation of the current value.

        Returns:
            T: New stored value after coercion.
        """
        with self._lock:
            candidate = function(self._value)
            coerced = self._coerce(candidate)
            self._value = coerced
            return coerced

    def compare_and_set(self, expected: object, value: T) -> bool:
        """Replace the value atomically when it equals an expected value.

        Another holder supplied as ``expected`` is snapshotted before this holder is locked, so
        locks from two holders are never nested. Coercion occurs only after a match and failure
        leaves the current value unchanged.

        Args:
            expected (object): Value required for replacement to proceed.
            value (T): Replacement value to coerce and store.

        Returns:
            bool: Whether the expected value matched and replacement succeeded.
        """
        comparison = self._comparison_value(expected)
        with self._lock:
            if self._value != comparison:
                return False
            coerced = self._coerce(value)
            self._value = coerced
            return True

    def __str__(self) -> str:
        return str(self.get())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.get()!r})"

    def __bool__(self) -> bool:
        return bool(self.get())

    def __eq__(self, other: object) -> bool:
        return self.get() == self._comparison_value(other)

    def __ne__(self, other: object) -> bool:
        return self.get() != self._comparison_value(other)

    def __lt__(self, other: Any) -> bool:
        return self.get() < self._comparison_value(other)

    def __le__(self, other: Any) -> bool:
        return self.get() <= self._comparison_value(other)

    def __gt__(self, other: Any) -> bool:
        return self.get() > self._comparison_value(other)

    def __ge__(self, other: Any) -> bool:
        return self.get() >= self._comparison_value(other)

    @staticmethod
    def _comparison_value(other: Any) -> Any:
        """Return another holder's value or an ordinary comparison operand."""
        return other.get() if isinstance(other, ValueHolder) else other

    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            return {name: value for name, value in self.__dict__.items() if name != "_lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._lock = threading.RLock()
        self.__dict__.update(state)


class StrValueHolder(ValueHolder[str]):
    """Hold a mutable value coerced with :class:`str`.

    Args:
        value (str): Initial value to convert to a string.
    """

    def _coerce(self, value: str) -> str:
        return str(value)


class IntValueHolder(ValueHolder[int]):
    """Hold a mutable value coerced with :class:`int`.

    Args:
        value (int): Initial value accepted by ``int(value)``.
    """

    def _coerce(self, value: int) -> int:
        return int(value)

    def __int__(self) -> int:
        return self.get()

    def __index__(self) -> int:
        return self.get()


class FloatValueHolder(ValueHolder[float]):
    """Hold a mutable value coerced with :class:`float`.

    Args:
        value (float): Initial value accepted by ``float(value)``.
    """

    def _coerce(self, value: float) -> float:
        return float(value)

    def __float__(self) -> float:
        return self.get()


class BoolValueHolder(ValueHolder[bool]):
    """Hold a mutable value coerced with normal Python truth-value semantics.

    Values are converted with ``bool(value)``; strings are not parsed. Consequently, an empty
    string is false while every non-empty string, including ``"false"``, is true.

    Args:
        value (bool): Initial value to convert with :class:`bool`.
    """

    def _coerce(self, value: bool) -> bool:
        return bool(value)


class PathHolder(ValueHolder[Path], os.PathLike[str]):
    """Hold a mutable reference to a filesystem path.

    Args:
        path (PathInput): Initial filesystem path.
    """

    def _coerce(self, value: PathInput) -> Path:
        """Normalize a path-like input into the sole contained Path value."""
        return Path(value)

    def __fspath__(self) -> str:
        return os.fspath(self.get())

    def __truediv__(self, key: PathInput) -> Path:
        return self.get() / key

    def __rtruediv__(self, key: PathInput) -> Path:
        return Path(key) / self.get().relative_to("/")
