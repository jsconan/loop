"""Test shared callable inspection utilities."""

from functools import partial

from loop.utils import callable_hints, callable_name


def test_callable_name_uses_declared_names_and_callable_class_names():
    """Callable names prefer the declared function name and fall back to the class name."""

    def named_function() -> None:
        pass

    class _CallableObject:
        def __call__(self) -> None:
            pass

    assert callable_name(named_function) == "named_function"
    assert callable_name(_CallableObject()) == "_CallableObject"


def test_callable_hints_resolves_functions_methods_partials_and_callable_objects():
    """Annotation resolution supports every callable form used by registration code."""

    def annotated(value: int) -> str:
        return str(value)

    class _CallableObject:
        def __call__(self, value: int) -> str:
            return str(value)

    class _Container:
        def method(self, value: int) -> str:
            return str(value)

    expected = {"value": int, "return": str}
    assert callable_hints(annotated) == expected
    assert callable_hints(_Container().method) == expected
    assert callable_hints(partial(annotated, 1)) == expected
    assert callable_hints(_CallableObject()) == expected
