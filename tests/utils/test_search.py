"""Tests for bounded ripgrep-backed text search utilities."""

import base64
import json
from unittest.mock import MagicMock

import pytest

from loop.utils.search import ripgrep_path, search_text_paths


def _message(kind, path, line, text, submatches=None, *, encoded=False):
    """Return one ripgrep JSON protocol line for a test process."""
    value = {"bytes": base64.b64encode(text.encode()).decode()} if encoded else {"text": text}
    return json.dumps(
        {
            "type": kind,
            "data": {
                "path": {"text": path},
                "line_number": line,
                "lines": value,
                "submatches": submatches or [],
            },
        }
    )


def _process(lines, *, return_code=0, error=""):
    """Return a process double exposing iterable output and readable errors."""
    process = MagicMock()
    process.stdout = lines
    process.stderr.read.return_value = error
    process.wait.return_value = return_code
    return process


def test_ripgrep_path_reports_discovery_and_missing_installations(monkeypatch):
    """Executable discovery returns PATH matches and explains unavailable ripgrep."""
    monkeypatch.setattr("loop.utils.search.shutil.which", MagicMock(return_value="/bin/rg"))
    assert ripgrep_path() == "/bin/rg"

    monkeypatch.setattr("loop.utils.search.shutil.which", MagicMock(return_value=None))
    with pytest.raises(FileNotFoundError, match="not installed"):
        ripgrep_path()


def test_search_text_paths_handles_empty_inputs_and_structured_protocol(tmp_path, monkeypatch):
    """Search parses text and base64 records, skips metadata, and handles absent submatches."""
    source = tmp_path / "source.txt"
    source.touch()
    lines = [
        json.dumps({"type": "begin", "data": {}}),
        _message("context", "source.txt", 1, "before\n", encoded=True),
        _message("match", "source.txt", 2, "€ match\n", [{"start": 4, "end": 9}]),
        _message("match", "source.txt", 4, "another\n"),
    ]
    popen = MagicMock(side_effect=lambda *_args, **_kwargs: _process(lines))
    monkeypatch.setattr("loop.utils.search.subprocess.Popen", popen)

    assert search_text_paths([], "match", root=tmp_path) == ([], False)
    matches, truncated = search_text_paths(
        [source], "match", root=tmp_path, context_lines=1, executable="rg"
    )

    assert truncated is False
    assert matches == [
        {
            "path": "source.txt",
            "line": 2,
            "column": 3,
            "text": "€ match",
            "context": [{"line": 1, "text": "before"}],
        },
        {"path": "source.txt", "line": 4, "column": 1, "text": "another"},
    ]

    limited, truncated = search_text_paths(
        [source], "match", root=tmp_path, max_bytes=1, executable="rg"
    )
    assert not limited
    assert truncated is True

    limited, truncated = search_text_paths(
        [source],
        "match",
        root=tmp_path,
        regex=True,
        case="sensitive",
        max_results=1,
        executable="rg",
    )
    assert len(limited) == 1
    assert truncated is True
    assert "--case-sensitive" in popen.call_args.args[0]

    search_text_paths([source], "match", root=tmp_path, case="insensitive", executable="rg")
    assert "--ignore-case" in popen.call_args.args[0]


def test_search_text_paths_rejects_missing_streams_and_process_failures(tmp_path, monkeypatch):
    """Malformed process setup and non-search exit statuses become clear runtime failures."""
    source = tmp_path / "source.txt"
    source.touch()
    missing_stream = _process([])
    missing_stream.stdout = None
    kill = MagicMock()
    monkeypatch.setattr("loop.utils.search.kill_process_group", kill)
    popen = MagicMock(return_value=missing_stream)
    monkeypatch.setattr("loop.utils.search.subprocess.Popen", popen)

    with pytest.raises(RuntimeError, match="output streams"):
        search_text_paths([source], "text", root=tmp_path, executable="rg")
    kill.assert_called_once_with(missing_stream)

    popen.return_value = _process([], return_code=2, error="bad regex")
    with pytest.raises(RuntimeError, match="bad regex"):
        search_text_paths([source], "(", root=tmp_path, regex=True, executable="rg")

    popen.return_value = _process([], return_code=3)
    with pytest.raises(RuntimeError, match="status 3"):
        search_text_paths([source], "text", root=tmp_path, executable="rg")
