"""Provide fast, bounded text search across explicit filesystem paths."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from .. import constants
from .models import TextSearchCase, TextSearchContext, TextSearchMatch
from .process import kill_process_group

_PATH_BATCH_SIZE = 200


def ripgrep_path() -> str:
    """Return the installed ripgrep executable.

    Returns:
        str: Absolute or directly executable path to ripgrep.

    Raises:
        FileNotFoundError: If ripgrep is not installed or available on ``PATH``.
    """
    executable = shutil.which("rg")
    if executable is None:
        raise FileNotFoundError("ripgrep executable 'rg' is not installed or available on PATH.")
    return executable


def _rg_text(value: dict[str, str]) -> str:
    """Decode one text-or-base64 value from ripgrep's JSON protocol."""
    if "text" in value:
        return value["text"]
    return base64.b64decode(value["bytes"]).decode("utf-8", errors="replace")


def _chunks(paths: list[Path]) -> Iterable[list[Path]]:
    """Yield argument-vector-safe batches of paths."""
    for start in range(0, len(paths), _PATH_BATCH_SIZE):
        yield paths[start : start + _PATH_BATCH_SIZE]


def search_text_paths(  # pylint: disable=too-many-branches,too-many-statements
    paths: Iterable[Path],
    query: str,
    *,
    root: Path,
    regex: bool = False,
    case: TextSearchCase = "smart",
    context_lines: int = 0,
    max_results: int = 100,
    max_bytes: int = constants.MAX_TOOL_CONTENT_BYTES,
    executable: str | None = None,
) -> tuple[list[TextSearchMatch], bool]:
    """Search explicit files with ripgrep and return deterministic structured matches.

    Args:
        paths (Iterable[Path]): Explicit visible files to search without further discovery.
        query (str): Literal text or regular expression to locate.
        root (Path): Directory used for execution and relative result paths.
        regex (bool): Whether ``query`` is a ripgrep regular expression. Defaults to literal text.
        case (TextSearchCase): Case matching strategy.
        context_lines (int): Number of neighboring lines retained around every match.
        max_results (int): Maximum matching lines returned across every file.
        max_bytes (int): Maximum approximate result bytes retained across matches and context.
        executable (str | None): Ripgrep executable override, or ``None`` to discover ``rg``.

    Returns:
        tuple[list[TextSearchMatch], bool]: Sorted matches and whether further matches were omitted.

    Raises:
        FileNotFoundError: If ripgrep is unavailable.
        RuntimeError: If ripgrep rejects the query or fails to search the selected files.
    """
    root = root.resolve()
    candidates = sorted({path.resolve() for path in paths})
    if not candidates:
        return [], False
    command = executable or ripgrep_path()
    matches: list[TextSearchMatch] = []
    all_lines: dict[tuple[str, int], str] = {}
    truncated = False
    retained_bytes = 0

    for batch in _chunks(candidates):
        arguments = [
            command,
            "--json",
            "--no-config",
            "--no-ignore",
            "--hidden",
            "--color=never",
            "--max-columns=1000",
            "--max-columns-preview",
        ]
        if not regex:
            arguments.append("--fixed-strings")
        if case == "smart":
            arguments.append("--smart-case")
        elif case == "sensitive":
            arguments.append("--case-sensitive")
        else:
            arguments.append("--ignore-case")
        if context_lines:
            arguments.extend(("--context", str(context_lines)))
        arguments.extend(("--regexp", query, "--"))
        arguments.extend(str(path.relative_to(root)) for path in batch)
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            arguments,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name == "posix",
        )
        if process.stdout is None or process.stderr is None:
            kill_process_group(process)
            raise RuntimeError("ripgrep did not expose its output streams.")
        for raw_line in process.stdout:
            message = json.loads(raw_line)
            if message["type"] not in {"match", "context"}:
                continue
            data = message["data"]
            path = _rg_text(data["path"])
            line = data["line_number"]
            text = _rg_text(data["lines"]).rstrip("\r\n")
            retained_bytes += len(path.encode("utf-8")) + len(text.encode("utf-8")) + 64
            if retained_bytes > max_bytes:
                truncated = True
                process.kill()
                break
            all_lines[(path, line)] = text
            if message["type"] == "context":
                continue
            submatches = data["submatches"]
            column = (
                len(text.encode("utf-8")[: submatches[0]["start"]].decode("utf-8")) + 1
                if submatches
                else 1
            )
            matches.append({"path": path, "line": line, "column": column, "text": text})
            if len(matches) > max_results:
                matches.pop()
                truncated = True
                process.kill()
                break
        return_code = process.wait()
        error = process.stderr.read().strip()
        if truncated:
            break
        if return_code not in {0, 1}:
            raise RuntimeError(error or f"ripgrep exited with status {return_code}.")

    matches.sort(key=lambda item: (item["path"], item["line"], item["column"]))
    if context_lines:
        for match in matches:
            context: list[TextSearchContext] = []
            for line in range(match["line"] - context_lines, match["line"] + context_lines + 1):
                if line == match["line"]:
                    continue
                text = all_lines.get((match["path"], line))
                if text is not None:
                    context.append({"line": line, "text": text})
            if context:
                match["context"] = context
    return matches, truncated
