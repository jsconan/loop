"""Provide utilities for inspecting Python callables."""

import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, get_type_hints


def callable_name(function: Callable[..., Any]) -> str:
    """Return a stable default name for a function or callable object.

    Args:
        function (Callable[..., Any]): Callable whose name should be resolved.

    Returns:
        str: The callable's declared name, or its class name for callable objects.
    """
    return getattr(function, "__name__", type(function).__name__)


def callable_hints(function: Callable[..., Any]) -> dict[str, Any]:
    """Resolve annotations from a function, partial, or callable object's implementation.

    Args:
        function (Callable[..., Any]): Callable whose annotations should be resolved.

    Returns:
        dict[str, Any]: Resolved annotations, including annotations from ``typing`` extras.
    """
    if isinstance(function, partial):
        target = function.func
    elif inspect.isfunction(function) or inspect.ismethod(function):
        target = function
    else:
        target = function.__call__
    return get_type_hints(target, include_extras=True)
