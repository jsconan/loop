"""Provide general text formatting utilities."""

from difflib import unified_diff

from .. import constants


def format_content_preview(
    content: str,
    *,
    max_chars: int = constants.CONTENT_PREVIEW_MAX_CHARS,
    max_lines: int = constants.CONTENT_PREVIEW_MAX_LINES,
) -> str:
    """Format file content for a human-readable preview.

    Truncates oversized content by line count and character count,
    preserving line breaks for readability.

    Args:
        content (str): The raw file content to display.
        max_chars (int): Maximum allowed character count before truncation.
        max_lines (int): Maximum allowed line count before truncation.

    Returns:
        str: A formatted preview string with line numbers and optional truncation notices.
    """
    truncated = False
    truncated_message: str | None = None

    total_chars = len(content)
    if total_chars > max_chars:
        content = content[:max_chars]
        truncated = True
        truncated_message = f"... (truncated, total {total_chars} chars)"

    lines = content.split("\n")

    if len(lines) > max_lines:
        remaining = len(lines) - max_lines
        lines = lines[:max_lines]
        truncated = True
        truncated_message = f"... ({remaining} more lines omitted)"

    preview = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))

    if truncated:
        preview += f"\n     {truncated_message}"

    return preview


def format_content_diff(
    before: str,
    after: str,
    path: str,
    *,
    max_chars: int = constants.CONTENT_PREVIEW_MAX_CHARS,
    max_lines: int = constants.CONTENT_PREVIEW_MAX_LINES,
) -> str:
    """Format a bounded unified diff for a file replacement preview.

    Keeps complete changed hunks so the displayed preview is always valid unified-diff output.

    Args:
        before (str): Existing UTF-8 text in the destination file.
        after (str): Replacement UTF-8 text proposed for the destination file.
        path (str): Destination path used in the diff headers.
        max_chars (int): Maximum characters displayed, excluding the omission notice.
        max_lines (int): Maximum lines displayed, excluding the omission notice.

    Returns:
        str: A summary and bounded unified diff, or a no-change summary.
    """
    diff_lines = list(
        unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
            lineterm="",
        )
    )
    if not diff_lines:
        return "No content changes."

    additions = sum(line.startswith("+") and not line.startswith("+++") for line in diff_lines)
    deletions = sum(line.startswith("-") and not line.startswith("---") for line in diff_lines)
    headers = diff_lines[:2]
    hunks = []
    for line in diff_lines[2:]:
        if line.startswith("@@"):
            hunks.append([line])
        else:
            hunks[-1].append(line)

    included = headers.copy()
    included_hunks = 0
    for hunk in hunks:
        candidate = [*included, *hunk]
        if included_hunks and (len(candidate) > max_lines or len("\n".join(candidate)) > max_chars):
            break
        if not included_hunks and (
            len(candidate) > max_lines or len("\n".join(candidate)) > max_chars
        ):
            break
        included = candidate
        included_hunks += 1

    summary = f"{additions} addition(s), {deletions} deletion(s), {len(hunks)} changed hunk(s)"
    preview = "\n".join((summary, *included))
    omitted_hunks = len(hunks) - included_hunks
    if omitted_hunks:
        preview += f"\n... ({omitted_hunks} changed hunk(s) omitted; preview limit reached)"
    return preview
