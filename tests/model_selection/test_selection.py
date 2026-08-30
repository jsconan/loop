"""Tests for active conversation model selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import BackendConnectionError, ModelInfo, ModelSelection, Session, SessionManager


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


def test_selection_persists_explicit_intent_while_reconciling_session_assignments():
    """Explicit choices persist, while restored and reported assignments update the session."""
    models = [ModelInfo(id="first"), ModelInfo(id="second")]
    active_backend = backend(get_models=Mock(return_value=models))
    session_manager = SessionManager()
    persist = Mock()
    selection = ModelSelection(active_backend, session_manager, on_select=persist)

    assert selection.available() == models
    selection.select("second")
    assert selection.selected == "second"
    persist.assert_called_once_with("second")
    assert session_manager.model == "second"
    assert selection.record_response(None).model == "second"
    assert session_manager.model == "second"

    selection.restore(None)
    assert selection.selected is None
    assert selection.effective == "default"
    assert session_manager.model == "default"

    assert selection.record_response("provider-alias").model == "provider-alias"
    assert selection.selected == "provider-alias"
    assert session_manager.model == "provider-alias"

    no_default = backend(default_model=None)
    unconfigured_session = SessionManager()
    unconfigured = ModelSelection(no_default, unconfigured_session, selected="temporary")
    unconfigured.restore(None)
    assert unconfigured.selected is None
    assert unconfigured_session.model is None
    assert unconfigured_session.context_window is None

    restored_session = SessionManager()
    restored = ModelSelection(no_default, restored_session)
    restored.restore(None)
    assert restored.selected is None
    assert restored_session.model is None


def test_selection_replaces_backend_and_invalidates_its_catalog():
    """A replacement backend becomes the catalog source without retaining stale values."""
    selection = ModelSelection(backend(), SessionManager())
    selection.available()
    replacement = backend(default_model="replacement")

    selection.backend = replacement

    assert selection.backend is replacement
    assert selection.effective == "replacement"
    assert selection.available() == replacement.get_models.return_value


def test_selection_does_not_change_when_its_durable_writer_fails():
    """A failed preference write leaves the active model and session assignment unchanged."""
    session_manager = SessionManager()
    selection = ModelSelection(
        backend(),
        session_manager,
        on_select=Mock(side_effect=OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        selection.select("second")

    assert selection.selected is None
    assert session_manager.model == "default"


def test_selection_reuses_a_recent_non_empty_model_catalog(monkeypatch):
    """Available models are reused until the short catalog cache expires."""
    models = [ModelInfo(id="first")]
    active_backend = backend(get_models=Mock(return_value=models))
    now = [10.0]
    monkeypatch.setattr("loop.model_selection.selection.monotonic", lambda: now[0])
    selection = ModelSelection(active_backend, SessionManager())

    assert selection.available() == models
    now[0] = 14.9
    assert selection.available() == models
    now[0] = 15.0
    assert selection.available() == models
    assert active_backend.get_models.call_count == 2


def test_selection_does_not_cache_empty_or_failed_model_catalogs():
    """Empty and failed discovery requests are retried rather than retained."""
    active_backend = backend(
        get_models=Mock(
            side_effect=[
                [],
                BackendConnectionError("offline", provider="test", operation="list"),
                [],
            ]
        )
    )
    selection = ModelSelection(active_backend, SessionManager())

    assert selection.available() == []
    with pytest.raises(BackendConnectionError, match="offline"):
        selection.available()
    assert selection.available() == []
    assert active_backend.get_models.call_count == 3
