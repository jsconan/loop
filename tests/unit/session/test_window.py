"""Test immutable model-window data."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from loop.instructions import InstructionSection, PreparedInstructions
from loop.session import InstructionRevision, ModelWindow, WindowSettings
from loop.utils import sha256_digest


def _snapshot(text, sections=()):
    """Build a small canonical instruction snapshot with valid provenance."""
    return PreparedInstructions(
        content=text,
        generation=0,
        working_directory=None,
        active_skills=(),
        sections=sections,
        digest=sha256_digest(text),
    )


def test_window_rejects_transient_or_inconsistent_baselines():
    """Only durable canonical instruction snapshots become model baselines."""
    original = _snapshot("rules")
    fields = {
        "id": "window",
        "route": "route",
        "model": "model",
        "layout": "leading",
        "settings": WindowSettings(agent_name="Loop"),
        "tools": (),
    }
    for baseline in (replace(original, suffix="temporary"), replace(original, digest="wrong")):
        with pytest.raises(ValidationError, match="Invalid model-window instruction baseline"):
            ModelWindow(**fields, baseline=baseline)
    window = ModelWindow(**fields, baseline=original)
    assert window.leading_instructions() == "rules"


def test_window_wait_defaults_must_fit_its_advertised_limit():
    """The stored invocation timing and advertised bounds agree."""
    with pytest.raises(ValidationError, match="Pinned command waits exceed"):
        WindowSettings(agent_name="Loop", command_max_wait_seconds=1)


def test_saved_window_settings_load_without_the_obsolete_timezone_pin():
    """Load earlier window settings without restoring their former clock preference."""
    saved_settings = {"agent_name": "Loop", "timezone": "UTC"}

    settings = WindowSettings.model_validate(saved_settings)

    assert settings == WindowSettings(agent_name="Loop")
    assert "timezone" not in settings.model_dump()
    assert saved_settings["timezone"] == "UTC"


def test_revision_detects_reordering_without_replacing_identical_sections():
    """Precedence changes identify the new order without resending unchanged bodies."""
    first = InstructionSection("agents", "first", "/project/AGENTS.md")
    second = InstructionSection("agents", "second", "/project/nested/AGENTS.md")
    original = _snapshot("first\n\nsecond", (first, second))
    reordered = _snapshot("second\n\nfirst", (second, first))
    revision = InstructionRevision.compare(original, reordered)
    assert revision is not None
    assert '"replace": []' in revision.content
    assert f'"order": ["{second.identity}", "{first.identity}"]' in revision.content
    assert "/project" not in revision.content
