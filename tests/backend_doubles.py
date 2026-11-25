"""Provide recording test doubles for the model transport contract."""

from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock

from loop.backend import Backend, estimate_context


async def _estimate_async(items, *, instructions, tools, model):
    """Estimate a complete request without contacting a server."""
    return estimate_context(items, instructions=instructions, tools=tools)


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
        "can_replay_checkpoint": lambda items, model, **kwargs: True,
        "preserves_compacted_reasoning": False,
        "generation_reserve": lambda capacity: max(1, min(2048, capacity // 8)),
        "prepare_context": list,
        "estimate_generation": lambda items, *, instructions, tools, model: estimate_context(
            items, instructions=instructions, tools=tools
        ),
        "estimate_generation_async": _estimate_async,
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
