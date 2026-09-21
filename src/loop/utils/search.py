"""Provide bounded in-process text search across explicit filesystem paths."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from .. import constants
from .models import TextSearchCase, TextSearchContext, TextSearchMatch


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
) -> tuple[list[TextSearchMatch], bool]:
    """Search explicit files in-process and return deterministic structured matches.

    Args:
        paths (Iterable[Path]): Explicit visible files to search without further discovery.
        query (str): Literal text or Python regular expression to locate.
        root (Path): Directory used for relative result paths.
        regex (bool): Whether ``query`` is a regular expression. Defaults to literal text.
        case (TextSearchCase): Case matching strategy.
        context_lines (int): Number of neighboring lines retained around every match.
        max_results (int): Maximum matching lines returned across every file.
        max_bytes (int): Maximum approximate result bytes retained across matches and context.

    Returns:
        tuple[list[TextSearchMatch], bool]: Sorted matches and whether further matches were omitted.

    Raises:
        ValueError: If the regular expression is invalid.
        OSError: If a selected file cannot be read.
    """
    root = root.resolve()
    candidates = sorted({path.resolve() for path in paths})
    if not candidates:
        return [], False
    flags = 0
    if case == "insensitive" or case == "smart" and query.lower() == query:
        flags = re.IGNORECASE
    try:
        pattern = re.compile(query if regex else re.escape(query), flags)
    except re.error as error:
        raise ValueError(f"regex parse error: {error}") from error
    matches: list[TextSearchMatch] = []
    retained_bytes = 0
    for candidate in candidates:
        path = candidate.relative_to(root).as_posix()
        previous: deque[tuple[int, str]] = deque(maxlen=context_lines)
        pending: list[tuple[TextSearchMatch, int]] = []
        with candidate.open("r", encoding="utf-8", errors="replace", newline=None) as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                text = raw_line.rstrip("\r\n")
                for pending_match, remaining in tuple(pending):
                    context_bytes = len(text.encode("utf-8")) + 16
                    if retained_bytes + context_bytes > max_bytes:
                        return matches, True
                    retained_bytes += context_bytes
                    context = pending_match.setdefault("context", [])
                    context.append({"line": line_number, "text": text})
                    pending.remove((pending_match, remaining))
                    if remaining > 1:
                        pending.append((pending_match, remaining - 1))
                found = pattern.search(text)
                if found is not None:
                    if len(matches) >= max_results:
                        return matches, True
                    context: list[TextSearchContext] = [
                        {"line": number, "text": value} for number, value in previous
                    ]
                    retained_bytes += (
                        len(path.encode("utf-8"))
                        + len(text.encode("utf-8"))
                        + sum(len(item["text"].encode("utf-8")) + 16 for item in context)
                        + 64
                    )
                    if retained_bytes > max_bytes:
                        return matches, True
                    match: TextSearchMatch = {
                        "path": path,
                        "line": line_number,
                        "column": found.start() + 1,
                        "text": text,
                    }
                    if context:
                        match["context"] = context
                    matches.append(match)
                    if context_lines:
                        pending.append((match, context_lines))
                previous.append((line_number, text))
    return matches, False
