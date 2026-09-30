"""Exercise owned native processes in disposable workspaces."""

import logging
import shutil
import subprocess

import pytest

from loop.utils.process import supervise_process
from loop.utils.search import search_text_paths

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def installed_ripgrep():
    """Locate the optional real helper before per-test environment isolation."""
    executable = shutil.which("rg")
    if executable is None:
        pytest.skip("requires installed ripgrep")
    return executable


def test_installed_ripgrep_searches_literal_and_regex_across_selected_files(
    tmp_path, caplog, installed_ripgrep
):
    """A real ripgrep handles regex when selected and fallback retains Python syntax."""
    caplog.set_level(logging.WARNING, logger="loop.utils.process")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("x" * 2000 + "\n€ needle\n", encoding="utf-8")
    second.write_text("needle\n", encoding="utf-8")
    executable = installed_ripgrep

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
