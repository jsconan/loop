"""Tests for deterministic and backend-assisted session naming."""

import json
from types import SimpleNamespace

import pytest

from loop import (
    AnswerCompleted,
    BackendSessionNameGenerator,
    ResponseCompleted,
    initial_session_name,
    normalize_session_name,
    validate_session_source,
)


def test_initial_name_normalizes_whitespace_and_bounds_complete_words():
    """Provisional names are readable, single-line, and bounded."""
    name = initial_session_name(
        "  Diagnose\nwhy session retrieval returns the wrong conversation today  "
    )

    assert name == "Diagnose why session retrieval returns the…"
    assert initial_session_name(" \n ") == "Untitled session"


def test_backend_generator_requests_and_reads_structured_output():
    """Generated names isolate transcript data behind the structured-output contract."""
    backend = SimpleNamespace()
    request = {}

    def get_response(**kwargs):
        nonlocal request
        request = kwargs
        output_format = kwargs["output_format"]
        generated = output_format.model(title="SQLite Session Retrieval")
        yield ResponseCompleted(structured_output=generated)

    backend.get_response = get_response
    generator = BackendSessionNameGenerator(backend)

    assert generator.generate("Find my session", "Use the sessions table", "model-a") == (
        "SQLite Session Retrieval"
    )
    assert json.loads(request["input"]) == {
        "user_message": "Find my session",
        "assistant_message": "Use the sessions table",
    }
    assert "untrusted conversation content" in request["instructions"]
    assert "Do not answer" in request["instructions"]
    assert request["model"] == "model-a"
    assert request["output_format"].model is not None
    assert request["output_format"].schema["required"] == ["title"]


def test_backend_generator_returns_none_without_structured_output():
    """A missing structured title leaves the provisional name unchanged."""
    backend = SimpleNamespace(get_response=lambda **_kwargs: [ResponseCompleted()])

    assert BackendSessionNameGenerator(backend).generate("question", "answer", None) is None


def test_initial_name_length_must_be_positive():
    """Name normalization rejects unusable bounds."""
    with pytest.raises(ValueError, match="positive"):
        normalize_session_name("name", 0)

    assert normalize_session_name("x" * 90) == f"{'x' * 80}…"


def test_backend_generator_ignores_nonterminal_events_and_empty_names():
    """Only completed structured output can replace a provisional name."""
    backend = SimpleNamespace()

    def get_response(**kwargs):
        yield AnswerCompleted(text="ignored")
        yield ResponseCompleted(structured_output=kwargs["output_format"].model(title="  "))

    backend.get_response = get_response

    assert BackendSessionNameGenerator(backend).generate("question", "answer", None) is None


def test_validate_session_source_accepts_valid_sources():
    """Each known source is accepted without raising."""
    for source in ("initial", "generated", "user"):
        validate_session_source(source)


def test_validate_session_source_rejects_invalid_source():
    """An unknown source string raises ValueError with a descriptive message."""
    with pytest.raises(ValueError, match="Invalid session name source"):
        validate_session_source("bogus")


def test_validate_session_source_allows_none_when_flagged():
    """Passing allow_none=True treats None as a valid source."""
    validate_session_source(None, allow_none=True)


def test_validate_session_source_rejects_none_without_flag():
    """Without allow_none, None is rejected just like any other invalid source."""
    with pytest.raises(ValueError, match="Invalid session name source"):
        validate_session_source(None)


def test_validate_session_source_allows_valid_source_even_with_flag():
    """A valid source remains accepted when allow_none=True is set."""
    for source in ("initial", "generated", "user"):
        validate_session_source(source, allow_none=True)
