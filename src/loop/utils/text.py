"""Provide general text formatting utilities."""

CONTENT_PREVIEW_MAX_LINES = 20
CONTENT_PREVIEW_MAX_CHARS = 2_000


def format_content_preview(
    content: str,
    *,
    max_chars: int = CONTENT_PREVIEW_MAX_CHARS,
    max_lines: int = CONTENT_PREVIEW_MAX_LINES,
) -> str:
    """Format file content for a human-readable preview.

    Truncates oversized content by line count and character count,
    preserving line breaks for readability.

    Args:
        content: The raw file content to display.
        max_chars: Maximum allowed character count before truncation.
        max_lines: Maximum allowed line count before truncation.

    Returns:
        A formatted preview string with line numbers and optional truncation notices.
    """
    truncated = False
    truncated_message: str | None = None

    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True
        truncated_message = f"... (truncated, total {len(content)} chars)"

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
