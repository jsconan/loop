"""Tests for active conversation model selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import ModelInfo, ModelSelection, Session, SessionManager


def backend(**attributes):
    """Build a backend double with configurable model metadata."""
    defaults = {
        "default_model": "default",
        "get_context_window": Mock(return_value=8192),
        "get_models": Mock(return_value=[ModelInfo(id="default")]),
    }
    return SimpleNamespace(**(defaults | attributes))


def test_selection_prefers_explicit_then_restored_then_backend_default_models():
    """Initialization applies explicit, restored, and backend-default precedence consistently."""
    explicit_session = SessionManager(session=Session(model="restored"))
    explicit = ModelSelection(backend(), explicit_session, selected="explicit")
    assert explicit.selected == "explicit"
    assert explicit.effective == "explicit"
    assert explicit_session.model == "explicit"
    assert explicit_session.context_window == 8192

    default_session = SessionManager()
    default = ModelSelection(backend(), default_session)
    assert default.selected is None
    assert default.effective == "default"
    assert default_session.model == "default"


def test_selection_restores_last_used_model_without_a_backend_fallback():
    """A durable last-used assignment remains usable when no backend default is configured."""
    session_manager = SessionManager(session=Session(model="stale", context_window=1024))
    selection = ModelSelection(backend(default_model=None), session_manager)

    assert selection.effective == "stale"
    assert selection.assignment.context_window == 8192
    assert session_manager.model == "stale"
    assert session_manager.context_window == 8192

    empty = ModelSelection(backend(default_model=None), SessionManager())
    with pytest.raises(ValueError, match="No model was selected"):
        _ = empty.effective
    with pytest.raises(ValueError, match="No model was selected"):
        empty.record_assignment()


@pytest.mark.parametrize("context_window", [None, True, False, 0, -1, "8192"])
def test_selection_normalizes_invalid_context_windows(context_window):
    """Only positive integer context-window metadata is persisted in the active session."""
    session_manager = SessionManager()
    ModelSelection(
        backend(get_context_window=Mock(return_value=context_window)),
        session_manager,
    )

    assert session_manager.context_window is None


def test_selection_lists_selects_restores_and_records_models():
    """Selection controls discovery, future intent, restoration, and durable use records."""
    models = [ModelInfo(id="first"), ModelInfo(id="second")]
    active_backend = backend(get_models=Mock(return_value=models))
    session_manager = SessionManager()
    selection = ModelSelection(active_backend, session_manager)

    assert selection.available() == models
    selection.select("second")
    assert selection.selected == "second"
    assert session_manager.model == "second"
    assert selection.record_response(None).model == "second"
    assert session_manager.model == "second"

    selection.restore(None)
    assert selection.selected is None
    assert selection.effective == "default"
    assert session_manager.model == "default"

    no_default = backend(default_model=None)
    unconfigured_session = SessionManager()
    unconfigured = ModelSelection(no_default, unconfigured_session, selected="temporary")
    unconfigured.restore(None)
    assert unconfigured.selected is None
    assert unconfigured_session.model is None
    assert unconfigured_session.context_window is None
