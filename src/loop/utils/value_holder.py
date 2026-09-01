"""Provide stable mutable indirection around values."""

from typing import Any


class ValueHolder[T]:
    """Hold a mutable value behind a stable object identity.

    All holders expose the current value through :attr:`value` and :meth:`get`, and replace it
    through either :attr:`value` assignment or :meth:`set`. They stringify and compare by their
    current contained values while retaining object identity across mutation. Mutation makes
    holders deliberately unhashable.

    Typed subclasses add conversion protocols appropriate to their contained type. They remain
    holders rather than subclasses or complete proxies of the wrapped built-in values.

    Args:
        value (T): Initial contained value.
    """

    _value: T
    __hash__ = None

    def __init__(self, value: T) -> None:
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
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        """Replace the contained value while retaining this holder's identity.

        Args:
            value (T): New value to contain.
        """
        self._value = self._coerce(value)

    def get(self) -> T:
        """Return the current contained value.

        Returns:
            T: Current contained value.
        """
        return self.value

    def set(self, value: T) -> None:
        """Replace the contained value while retaining this holder's identity.

        Args:
            value (T): New value to contain.
        """
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == self._comparison_value(other)

    def __ne__(self, other: object) -> bool:
        return self.value != self._comparison_value(other)

    def __lt__(self, other: Any) -> bool:
        return self.value < self._comparison_value(other)

    def __le__(self, other: Any) -> bool:
        return self.value <= self._comparison_value(other)

    def __gt__(self, other: Any) -> bool:
        return self.value > self._comparison_value(other)

    def __ge__(self, other: Any) -> bool:
        return self.value >= self._comparison_value(other)

    @staticmethod
    def _comparison_value(other: Any) -> Any:
        """Return another holder's value or an ordinary comparison operand."""
        return other.value if isinstance(other, ValueHolder) else other


class StrValueHolder(ValueHolder[str]):
    """Hold a mutable value coerced with :class:`str`.

    Args:
        value (object): Initial value to convert to a string.
    """

    def _coerce(self, value: str) -> str:
        return str(value)


class IntValueHolder(ValueHolder[int]):
    """Hold a mutable value coerced with :class:`int`.

    Args:
        value (object): Initial value accepted by ``int(value)``.
    """

    def _coerce(self, value: int) -> int:
        return int(value)

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value


class FloatValueHolder(ValueHolder[float]):
    """Hold a mutable value coerced with :class:`float`.

    Args:
        value (object): Initial value accepted by ``float(value)``.
    """

    def _coerce(self, value: float) -> float:
        return float(value)

    def __float__(self) -> float:
        return self.value


class BoolValueHolder(ValueHolder[bool]):
    """Hold a mutable value coerced with normal Python truth-value semantics.

    Values are converted with ``bool(value)``; strings are not parsed. Consequently, an empty
    string is false while every non-empty string, including ``"false"``, is true.

    Args:
        value (object): Initial value to convert with :class:`bool`.
    """

    def _coerce(self, value: bool) -> bool:
        return bool(value)
