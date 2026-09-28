"""Provide bounded search across explicit files with a native or in-process matcher."""

from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import time
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from pathlib import Path

import regex as re

from .. import constants
from .models import TextSearchCase, TextSearchContext, TextSearchMatch
from .process import ProcessCapture, ProcessCaptureStatus

_MAX_QUERY_CHARS = 1000
_MAX_LINE_CHARS = 1000
_SEARCH_SECONDS = 30


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
    """Decode ripgrep JSON text or a base64 encoded byte value."""
    if "text" in value:
        return value["text"]
    try:
        return base64.b64decode(value["bytes"], validate=True).decode("utf-8", errors="replace")
    except (KeyError, TypeError, binascii.Error) as exc:
        raise RuntimeError("Native search returned invalid structured output.") from exc


def _open_selected_file(root: Path, path: Path, identity: tuple[int, int]) -> int:
    """Open a selected file through real directory descriptors and verify its inode."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    with ExitStack() as stack:
        directory = os.open(root, directory_flags)
        stack.callback(os.close, directory)
        for part in path.relative_to(root).parts[:-1]:
            directory = os.open(part, directory_flags, dir_fd=directory)
            stack.callback(os.close, directory)
        descriptor = os.open(
            path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory
        )
        details = os.fstat(descriptor)
        if (details.st_dev, details.st_ino) != identity:
            os.close(descriptor)
            raise OSError("Selected search file changed before opening.")
        return descriptor


def search_text_paths(
    paths: Iterable[Path],
    query: str,
    *,
    root: Path,
    regex: bool = False,
    case: TextSearchCase = "smart",
    context_lines: int = 0,
    max_results: int = 100,
    max_bytes: int = constants.MAX_TOOL_CONTENT_BYTES,
    native_runner: Callable[[tuple[str, ...], tuple[int, ...], float], ProcessCapture | None]
    | None = None,
) -> tuple[list[TextSearchMatch], bool]:
    """Search explicit files without granting an external executable host authority.

    Args:
        paths (Iterable[Path]): Explicit visible files to search without further discovery.
        query (str): Literal text or regular expression to locate.
        root (Path): Canonical directory used for relative result paths.
        regex (bool): Whether ``query`` is a regular expression. Defaults to literal text.
        case (TextSearchCase): Case matching strategy.
        context_lines (int): Number of neighboring lines retained around every match.
        max_results (int): Maximum matching lines returned across every file.
        max_bytes (int): Maximum approximate result bytes retained across matches and context.
        native_runner (Callable | None): Trusted read-only argv executor for ripgrep searches,
            or None to use the bounded in-process matcher.

    Returns:
        tuple[list[TextSearchMatch], bool]: Sorted matches and whether further matches were omitted.

    Raises:
        RuntimeError: If a pattern, selected file, or bounded search cannot be handled safely.
    """
    if len(query) > _MAX_QUERY_CHARS:
        raise RuntimeError("Search pattern exceeds the length limit.")
    root = root.resolve(strict=True)
    candidates = sorted({path.resolve(strict=True) for path in paths})
    if any(not path.is_relative_to(root) for path in candidates):
        raise RuntimeError("Selected search file escaped its authorized root.")
    if native_runner is not None and candidates:
        native = _search_native(
            candidates,
            query,
            root=root,
            regex=regex,
            case=case,
            context_lines=context_lines,
            max_results=max_results,
            max_bytes=max_bytes,
            deadline=time.monotonic() + _SEARCH_SECONDS,
            runner=native_runner,
        )
        if native is not None:
            return native
    # The bounded fallback uses Python regex syntax only when ripgrep cannot launch.
    flags = re.IGNORECASE if case == "insensitive" or (case == "smart" and query.islower()) else 0
    try:
        pattern = re.compile(query if regex else re.escape(query), flags)
    except re.error as exc:
        raise RuntimeError(f"regex parse error: {exc}") from exc
    matches: list[TextSearchMatch] = []
    contexts: dict[tuple[str, int], str] = {}
    retained_bytes = 0
    truncated = False
    result_limit_reached = False
    deadline = time.monotonic() + _SEARCH_SECONDS
    for path in candidates:
        relative = str(path.relative_to(root))
        previous: deque[tuple[int, str]] = deque(maxlen=context_lines)
        trailing = 0
        try:
            selected = path.stat()
            descriptor = _open_selected_file(root, path, (selected.st_dev, selected.st_ino))
            with os.fdopen(descriptor, "r", encoding="utf-8", errors="replace") as stream:
                number = 0
                while raw := stream.readline(_MAX_LINE_CHARS + 2):
                    number += 1
                    if len(raw) > _MAX_LINE_CHARS and not raw.endswith("\n"):
                        truncated = True
                        while raw and not raw.endswith("\n"):
                            if time.monotonic() >= deadline:
                                raise RuntimeError("Search exceeded its time limit.")
                            raw = stream.readline(_MAX_LINE_CHARS + 2)
                        previous.clear()
                        trailing = 0
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("Search exceeded its time limit.")
                    line = raw.rstrip("\r\n")
                    try:
                        found = pattern.search(line, timeout=min(0.1, remaining))
                    except TimeoutError as exc:
                        raise RuntimeError("Search pattern exceeded its time limit.") from exc
                    if found is not None:
                        retained_bytes += len(relative.encode()) + len(line.encode()) + 64
                        if retained_bytes > max_bytes or len(matches) >= max_results:
                            truncated = True
                            result_limit_reached = True
                            break
                        matches.append(
                            {
                                "path": relative,
                                "line": number,
                                "column": found.start() + 1,
                                "text": line,
                            }
                        )
                        for prior_number, prior_line in previous:
                            contexts[(relative, prior_number)] = prior_line
                        trailing = context_lines
                    elif trailing:
                        contexts[(relative, number)] = line
                        trailing -= 1
                    previous.append((number, line))
        except OSError as exc:
            raise RuntimeError(f"Could not search selected file: {path}") from exc
        if result_limit_reached:
            break
    if context_lines:
        for match in matches:
            context: list[TextSearchContext] = []
            for number in range(match["line"] - context_lines, match["line"] + context_lines + 1):
                if number == match["line"]:
                    continue
                line = contexts.get((match["path"], number))
                if line is not None:
                    context.append({"line": number, "text": line})
            if context:
                match["context"] = context
    return matches, truncated


def _search_native(
    candidates: list[Path],
    query: str,
    *,
    root: Path,
    regex: bool,
    case: TextSearchCase,
    context_lines: int,
    max_results: int,
    max_bytes: int,
    deadline: float,
    runner: Callable[[tuple[str, ...], tuple[int, ...], float], ProcessCapture | None],
) -> tuple[list[TextSearchMatch], bool] | None:
    """Search selected descriptors through a read-only, bounded ripgrep launch."""
    matches: list[TextSearchMatch] = []
    contexts: dict[tuple[str, int], str] = {}
    retained_bytes = 0
    truncated = False
    for offset in range(0, len(candidates), 200):
        with ExitStack() as stack:
            selected: dict[int, str] = {}
            for path in candidates[offset : offset + 200]:
                try:
                    details = path.stat()
                    descriptor = _open_selected_file(root, path, (details.st_dev, details.st_ino))
                except OSError as exc:
                    raise RuntimeError(f"Could not search selected file: {path}") from exc
                stack.callback(os.close, descriptor)
                selected[descriptor] = str(path.relative_to(root))
            arguments = [
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
            arguments.append(
                {
                    "smart": "--smart-case",
                    "sensitive": "--case-sensitive",
                    "insensitive": "--ignore-case",
                }[case]
            )
            if context_lines:
                arguments.extend(("--context", str(context_lines)))
            arguments.extend(("--regexp", query, "--", *(f"/dev/fd/{fd:05d}" for fd in selected)))
            capture = runner(tuple(arguments), tuple(selected), deadline)
            if capture is None:
                return None
            if capture.status is not ProcessCaptureStatus.COMPLETED:
                raise RuntimeError("Native search exceeded its time limit or was cancelled.")
            if capture.exit_code not in (0, 1):
                detail = (
                    capture.stderr.strip() or f"ripgrep exited with status {capture.exit_code}."
                )
                raise RuntimeError(f"regex parse error: {detail}" if regex else detail)
            truncated = truncated or capture.stdout_discarded != 0
            for raw in capture.stdout.splitlines():
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    if truncated:
                        break
                    raise RuntimeError(
                        "Native search returned invalid structured output."
                    ) from None
                kind = event.get("type")
                if kind not in {"match", "context"}:
                    continue
                data = event["data"]
                path_value = _rg_text(data["path"])
                relative = selected.get(int(path_value.rsplit("/", 1)[-1]))
                if relative is None:
                    raise RuntimeError("Native search returned an unselected path.")
                line = _rg_text(data["lines"])
                line = line.rstrip("\r\n")
                number = data["line_number"]
                if kind == "context":
                    retained_bytes += len(relative.encode()) + len(line.encode()) + 64
                    if retained_bytes > max_bytes:
                        truncated = True
                        break
                    contexts[(relative, number)] = line
                    continue
                start = data["submatches"][0]["start"] if data["submatches"] else 0
                column = len(line.encode("utf-8")[:start].decode("utf-8", errors="replace")) + 1
                retained_bytes += len(relative.encode()) + len(line.encode()) + 64
                if retained_bytes > max_bytes or len(matches) >= max_results:
                    truncated = True
                    break
                matches.append({"path": relative, "line": number, "column": column, "text": line})
            if truncated:
                break
    if context_lines:
        for match in matches:
            context = [
                {"line": number, "text": contexts[(match["path"], number)]}
                for number in range(
                    match["line"] - context_lines, match["line"] + context_lines + 1
                )
                if number != match["line"] and (match["path"], number) in contexts
            ]
            if context:
                match["context"] = context
    matches.sort(key=lambda item: (item["path"], item["line"], item["column"]))
    return matches, truncated
