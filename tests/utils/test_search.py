"""Tests for bounded selected-file text search utilities."""

import json
import logging
import os
import subprocess

import pytest

from loop.utils.process import ProcessCapture, ProcessCaptureStatus, supervise_process
from loop.utils.search import ripgrep_path, search_text_paths


def test_ripgrep_path_uses_host_lookup_order(tmp_path, monkeypatch):
    """The locator preserves shell lookup order without launching the result."""
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "rg").write_bytes(b"untrusted executable")
    (hostile / "rg").chmod(0o755)
    monkeypatch.setenv("PATH", str(hostile))

    assert ripgrep_path() == str(hostile / "rg")


def test_ripgrep_path_reports_missing_binary(monkeypatch):
    """Search falls back when fixed installation locations contain no ripgrep."""
    monkeypatch.setattr("loop.utils.search.shutil.which", lambda *_args, **_kwargs: None)
    with pytest.raises(FileNotFoundError, match="not installed"):
        ripgrep_path()


def test_search_text_paths_matches_unicode_context_and_case(tmp_path):
    """Search returns deterministic Unicode columns and adjacent lines with smart case."""
    source = tmp_path / "source.txt"
    source.write_text("before\n€ match\nafter\nMATCH\nmatch\n", encoding="utf-8")
    assert search_text_paths([], "match", root=tmp_path) == ([], False)
    matches, truncated = search_text_paths([source], "match", root=tmp_path, context_lines=1)
    assert not truncated
    assert [(item["line"], item["column"]) for item in matches] == [(2, 3), (4, 1), (5, 1)]
    assert matches[0]["context"] == [{"line": 1, "text": "before"}, {"line": 3, "text": "after"}]
    sensitive, _ = search_text_paths([source], "MATCH", root=tmp_path)
    assert [item["line"] for item in sensitive] == [4]
    insensitive, _ = search_text_paths([source], "MATCH", root=tmp_path, case="insensitive")
    assert len(insensitive) == 3
    isolated = tmp_path / "isolated.txt"
    isolated.write_text("match\n", encoding="utf-8")
    no_context, _ = search_text_paths([isolated], "match", root=tmp_path, context_lines=1)
    assert "context" not in no_context[0]


def test_search_text_paths_regex_and_limits(tmp_path):
    """Regex matching and both global output bounds stop retained results."""
    source = tmp_path / "source.txt"
    source.write_text("token 1\ntoken 2\ntoken 3\n", encoding="utf-8")
    matches, truncated = search_text_paths(
        [source], r"token \d", root=tmp_path, regex=True, max_results=2
    )
    assert len(matches) == 2 and truncated
    matches, truncated = search_text_paths([source], "token", root=tmp_path, max_bytes=1)
    assert not matches and truncated
    with pytest.raises(RuntimeError, match="regex parse error"):
        search_text_paths([source], "(", root=tmp_path, regex=True)
    with pytest.raises(RuntimeError, match="length limit"):
        search_text_paths([source], "x" * 1001, root=tmp_path)
    source.write_text("x" * 2000, encoding="utf-8")
    matches, truncated = search_text_paths([source], "x", root=tmp_path)
    assert not matches and truncated


def test_search_text_paths_bounds_pathological_regex(tmp_path):
    """A pattern with catastrophic backtracking cannot hold the application indefinitely."""
    source = tmp_path / "source.txt"
    source.write_text("a" * 900 + "!\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="time limit"):
        search_text_paths([source], r"(a+)+$", root=tmp_path, regex=True)


def test_search_text_paths_continues_after_overlong_line_and_file(tmp_path):
    """A skipped oversized line does not hide later matches in either selected file."""
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("x" * 2000 + "\nneedle\n", encoding="utf-8")
    second.write_text("needle\n", encoding="utf-8")

    matches, truncated = search_text_paths([first, second], "needle", root=tmp_path)

    assert [(match["path"], match["line"]) for match in matches] == [
        ("first.txt", 2),
        ("second.txt", 1),
    ]
    assert truncated


def test_search_text_paths_observes_total_deadline(tmp_path, monkeypatch):
    """A search that outlives its total budget stops before further matching."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(
        "loop.utils.search.time", SimpleNamespace(monotonic=Mock(side_effect=[0, 31]))
    )
    with pytest.raises(RuntimeError, match="Search exceeded its time limit"):
        search_text_paths([source], "needle", root=tmp_path)


def test_search_text_paths_bounds_drain_of_oversized_line(tmp_path, monkeypatch):
    """An oversized line cannot hold the search past its total deadline."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    source = tmp_path / "source.txt"
    source.write_text("x" * 2000, encoding="utf-8")
    monkeypatch.setattr(
        "loop.utils.search.time", SimpleNamespace(monotonic=Mock(side_effect=[0, 31]))
    )
    with pytest.raises(RuntimeError, match="Search exceeded its time limit"):
        search_text_paths([source], "needle", root=tmp_path)


def test_search_text_paths_rejects_outside_and_unreadable_files(tmp_path, monkeypatch):
    """Explicit files remain inside the selected root and read errors are surfaced."""
    source = tmp_path / "source.txt"
    source.write_text("text", encoding="utf-8")
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("text", encoding="utf-8")
    with pytest.raises(RuntimeError, match="escaped"):
        search_text_paths([outside], "text", root=tmp_path)
    monkeypatch.setattr(
        "loop.utils.search.os.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(RuntimeError, match="Could not search"):
        search_text_paths([source], "text", root=tmp_path)


@pytest.mark.parametrize("replacement", ["symlink", "regular"])
@pytest.mark.parametrize("native", [False, True])
def test_search_text_paths_rejects_replaced_nested_file(tmp_path, monkeypatch, replacement, native):
    """A file changed after selection cannot redirect in-process reads outside the root."""
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "source.txt"
    source.write_text("needle", encoding="utf-8")
    outside = tmp_path.parent / (tmp_path.name + "-private.txt")
    outside.write_text("secret", encoding="utf-8")
    original_open = os.open

    def replace_on_open(path, flags, *args, **kwargs):
        """Replace the selected file just before its final descriptor opens."""
        if path == source.name:
            source.unlink()
            if replacement == "symlink":
                source.symlink_to(outside)
            else:
                source.write_text("secret", encoding="utf-8")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("loop.utils.search.os.open", replace_on_open)
    with pytest.raises(RuntimeError, match="Could not search"):
        search_text_paths(
            [source],
            "needle",
            root=tmp_path,
            native_runner=(lambda *_: None) if native else None,
        )
    assert outside.read_text(encoding="utf-8") == "secret"


def test_search_text_paths_never_executes_workspace_path_program(tmp_path, monkeypatch):
    """A workspace-selected rg cannot run during text search."""
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "rg"
    fake.write_text("#!/bin/sh\nprintf bypass > ../outside\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    matches, truncated = search_text_paths([source], "needle", root=tmp_path)
    assert not truncated and matches[0]["text"] == "needle"
    assert not (tmp_path / "outside").exists()


def test_installed_ripgrep_searches_literal_and_regex_across_selected_files(tmp_path, caplog):
    """A real ripgrep handles regex when selected and fallback retains Python syntax."""
    caplog.set_level(logging.WARNING, logger="loop.utils.process")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("x" * 2000 + "\n€ needle\n", encoding="utf-8")
    second.write_text("needle\n", encoding="utf-8")
    executable = ripgrep_path()

    def run(arguments, descriptors, deadline):
        """Supervise real ripgrep with only the selected inherited descriptors."""
        process = subprocess.Popen(
            [str(executable), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=descriptors,
            start_new_session=True,
        )
        return supervise_process(process, deadline)

    literal = search_text_paths([first, second], "needle", root=tmp_path, native_runner=run)
    assert [(item["path"], item["line"]) for item in literal[0]] == [
        ("first.txt", 2),
        ("second.txt", 1),
    ]
    expression = search_text_paths(
        [first, second], r"needle$", root=tmp_path, regex=True, native_runner=run
    )
    assert expression == literal
    with pytest.raises(RuntimeError, match="regex parse error"):
        search_text_paths([first], r"(?<=€ )needle", root=tmp_path, regex=True, native_runner=run)
    with pytest.raises(RuntimeError, match="regex parse error"):
        search_text_paths([first], r"(needle)\1", root=tmp_path, regex=True, native_runner=run)
    fallback, _ = search_text_paths([first], r"(?<=€ )needle", root=tmp_path, regex=True)
    assert fallback[0]["column"] == 3
    bounded = search_text_paths(
        [first, second], "needle", root=tmp_path, max_results=1, native_runner=run
    )
    assert len(bounded[0]) == 1 and bounded[1]


def test_native_literal_search_binds_selected_fds_unicode_context_and_case(tmp_path):
    """Native JSON events retain selected paths, Unicode columns, context, and literal flags."""
    source = tmp_path / "source.txt"
    source.write_text("before\n€ needle\nafter\n", encoding="utf-8")
    invocations = []

    def run(arguments, descriptors, deadline):
        """Return ripgrep-shaped events for the selected live descriptor."""
        invocations.append((arguments, descriptors, deadline))
        selected = f"/dev/fd/{descriptors[0]:05d}"
        events = [
            {"type": "begin", "data": {"path": {"text": selected}}},
            {
                "type": "context",
                "data": {
                    "path": {"text": selected},
                    "lines": {"text": "before\n"},
                    "line_number": 1,
                },
            },
            {
                "type": "match",
                "data": {
                    "path": {"text": selected},
                    "lines": {"text": "€ needle\n"},
                    "line_number": 2,
                    "submatches": [{"start": 4, "end": 10}],
                },
            },
            {
                "type": "context",
                "data": {
                    "path": {"text": selected},
                    "lines": {"text": "after\n"},
                    "line_number": 3,
                },
            },
        ]
        return ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout="\n".join(json.dumps(event) for event in events),
        )

    matches, truncated = search_text_paths(
        [source], "needle", root=tmp_path, context_lines=1, native_runner=run
    )

    assert not truncated
    assert matches == [
        {
            "path": "source.txt",
            "line": 2,
            "column": 3,
            "text": "€ needle",
            "context": [{"line": 1, "text": "before"}, {"line": 3, "text": "after"}],
        }
    ]
    assert "--fixed-strings" in invocations[0][0]
    assert "--smart-case" in invocations[0][0]
    assert "--context" in invocations[0][0]
    assert invocations[0][0][-3] == "needle"
    assert search_text_paths(
        [source], "needle", root=tmp_path, context_lines=1, max_bytes=1, native_runner=run
    ) == ([], True)
    sensitive, _ = search_text_paths(
        [source],
        "needle",
        root=tmp_path,
        case="sensitive",
        context_lines=1,
        native_runner=run,
    )
    assert sensitive[0]["column"] == 3
    assert "--case-sensitive" in invocations[-1][0]


def test_native_literal_search_falls_back_and_reports_output_limits(tmp_path):
    """Unavailable native execution falls back, while bounded native output reports omissions."""
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    assert (
        search_text_paths([source], "needle", root=tmp_path, native_runner=lambda *_: None)[0][0][
            "path"
        ]
        == "source.txt"
    )

    def run(_arguments, descriptors, _deadline):
        """Return one valid match followed by a bounded partial event."""
        selected = f"/dev/fd/{descriptors[0]:05d}"
        match = {
            "type": "match",
            "data": {
                "path": {"text": selected},
                "lines": {"text": "needle\n"},
                "line_number": 1,
                "submatches": [{"start": 0}],
            },
        }
        return ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout=json.dumps(match) + "\n{partial",
            stdout_discarded=1,
        )

    matches, truncated = search_text_paths([source], "needle", root=tmp_path, native_runner=run)
    assert len(matches) == 1 and truncated
    matches, truncated = search_text_paths(
        [source], "needle", root=tmp_path, max_bytes=1, native_runner=run
    )
    assert matches == [] and truncated


def test_native_literal_search_omits_empty_context_and_bounds_result_count(tmp_path):
    """Context is included only when emitted, and a result cap stops native parsing."""
    source = tmp_path / "source.txt"
    source.write_text("needle\nneedle\n", encoding="utf-8")

    def run(_arguments, descriptors, _deadline):
        """Return two selected-file matches without neighboring context events."""
        selected = f"/dev/fd/{descriptors[0]:05d}"
        events = [
            {
                "type": "match",
                "data": {
                    "path": {"text": selected},
                    "lines": {"text": "needle\n"},
                    "line_number": number,
                    "submatches": [{"start": 0}],
                },
            }
            for number in (1, 2)
        ]
        return ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout="\n".join(json.dumps(event) for event in events),
        )

    matches, truncated = search_text_paths(
        [source], "needle", root=tmp_path, context_lines=1, max_results=1, native_runner=run
    )
    assert len(matches) == 1 and truncated
    assert "context" not in matches[0]


@pytest.mark.parametrize(
    "outcome", ("exit", "empty_exit_one", "timeout", "invalid", "wrong_path", "non_text")
)
def test_native_literal_search_fails_safe_on_unusable_results(tmp_path, outcome):
    """Launcher errors, timeouts, and malformed events never become trusted matches."""
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")

    def run(_arguments, descriptors, _deadline):
        """Supply one native failure mode through the public search boundary."""
        if outcome == "exit":
            return ProcessCapture(ProcessCaptureStatus.COMPLETED, exit_code=2)
        if outcome == "empty_exit_one":
            return ProcessCapture(ProcessCaptureStatus.COMPLETED, exit_code=1)
        if outcome == "timeout":
            return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)
        selected = f"/dev/fd/{descriptors[0]:05d}"
        event = {
            "type": "match",
            "data": {
                "path": {"text": selected},
                "lines": {"text": "needle\n"},
                "line_number": 1,
                "submatches": [{"start": 0}],
            },
        }
        if outcome == "wrong_path":
            event["data"]["path"]["text"] = "/dev/fd/99999"
        if outcome == "non_text":
            event["data"]["lines"] = {"bytes": "abc"}
        return ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            exit_code=0,
            stdout="{invalid" if outcome == "invalid" else json.dumps(event),
        )

    if outcome == "empty_exit_one":
        assert search_text_paths([source], "needle", root=tmp_path, native_runner=run) == (
            [],
            False,
        )
    else:
        with pytest.raises(RuntimeError):
            search_text_paths([source], "needle", root=tmp_path, native_runner=run)
