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


def test_selection_prefers_explicit_then_backend_default_models():
    """Initialization applies explicit and backend-default precedence consistently."""
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


def test_selection_requires_an_explicit_or_default_model():
    """An unconfigured selection remains constructible but rejects effective-model access."""
    session_manager = SessionManager(session=Session(model="stale", context_window=1024))
    selection = ModelSelection(backend(default_model=None), session_manager)

    assert session_manager.model is None
    assert session_manager.context_window is None

    with pytest.raises(ValueError, match="No model was selected"):
        _ = selection.effective
    with pytest.raises(ValueError, match="No model was selected"):
        selection.synchronize_session()


@pytest.mark.parametrize("context_window", [None, True, False, "8192"])
def test_selection_normalizes_invalid_context_windows(context_window):
    """Only integer context-window metadata is persisted in the active session."""
    session_manager = SessionManager()
    ModelSelection(
        backend(get_context_window=Mock(return_value=context_window)),
        session_manager,
    )

    assert session_manager.context_window is None


def test_selection_lists_selects_and_restores_models():
    """Discovery and selection operations delegate through one synchronized state owner."""
    models = [ModelInfo(id="first"), ModelInfo(id="second")]
    active_backend = backend(get_models=Mock(return_value=models))
    session_manager = SessionManager()
    selection = ModelSelection(active_backend, session_manager)

    assert selection.available() == models
    selection.select("second")
    assert selection.selected == "second"
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
