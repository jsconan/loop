"""Provide recording test doubles for the model transport contract."""

from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock

from loop.backend import Backend


def model_backend(backend=None, **properties):
    """Supply stable route capabilities and file projection to an isolated transport mock."""
    defaults = {
        "default_model": "default-model",
        "route": "recording:model",
        "endpoint": "",
        "file_input_mode": None,
        "resolve_file_input_mode": lambda mode: mode,
        "file_projection": lambda mode: nullcontext(),
        "instruction_layout": lambda model: "leading",
        "supports_reasoning_updates": lambda model: False,
        "prepare_context": list,
        "get_context_window": Mock(return_value=None),
        "get_context_window_async": AsyncMock(return_value=None),
    }
    if backend is None:
        return Mock(spec=Backend, **(defaults | properties))
    for name, default in defaults.items():
        current = getattr(backend, name, None)
        if isinstance(current, Mock) and (
            name in ("prepare_context", "get_context_window", "default_model")
            or name == "get_context_window_async"
            and isinstance(current, AsyncMock)
        ):
            continue
        if current is None or isinstance(current, Mock):
            setattr(backend, name, default)
    return backend
