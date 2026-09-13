"""Provide general text formatting utilities."""

import json
from collections.abc import Iterable, Mapping
from difflib import unified_diff
from typing import Any

from .. import constants
from .models import ChoiceItem


def snippet(text: str, language: str | None = None, preserve_fence: bool = True) -> str:
    """Wrap text in a fenced code block, optionally specifying a language.

    Args:
        text (str): The text to wrap in a fenced code block.
        language (str | None, optional): The language identifier for syntax highlighting.
            Defaults to None.
        preserve_fence (bool, optional): Whether to preserve the supplied text byte-for-byte
            inside a safe outer fence. When false, an existing outer fence and surrounding line
            breaks are normalized. Defaults to True.

    Returns:
        str: The text wrapped in a fenced code block.
    """
    if preserve_fence:
        fence = "```"
        while fence in text:
            fence += "`"
        return f"{fence}{language or ''}\n{text}\n{fence}"

    text = text.strip("\n\r")
    opening, separator, body = text.partition("\n")
    opening = opening.strip()
    opening_fence = opening[: len(opening) - len(opening.lstrip("`"))]
    has_wrapping_fence = (
        len(opening_fence) >= 3
        and separator
        and body.rstrip().split("\n")[-1].rstrip("\r") == opening_fence
    )
    if has_wrapping_fence:
        text = body.rstrip()[: -len(opening_fence)]
    text = text.rstrip()
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language or ''}\n{text}\n{fence}"


def list_terms(terms: Iterable[str | None], exclusive: bool = True, quote: str = "") -> str:
    """Format a list of terms as a human-readable string.

    Args:
        terms (Iterable[str | None]): The terms to format. ``None`` is rendered without quotes.
        exclusive (bool, optional): Whether to use "or" (True) or "and" (False) before
            the last term. Defaults to True.
        quote (str, optional): The string to place around each term. An empty string disables
            quoting. Defaults to "".

    Returns:
        str: The formatted list of terms.
    """
    terms = [
        "None" if term is None else f"{quote}{term}{quote}" if quote else term for term in terms
    ]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    conjunction = "or" if exclusive else "and"
    separator = f", {conjunction} " if len(terms) > 2 else f" {conjunction} "
    return ", ".join(terms[:-1]) + separator + terms[-1]


def validate_term(value: Any, values: list[Any], message: str) -> None:
    """Validate that a value is in a list of allowed values.

    Args:
        value (Any): The value to validate.
        values (list[Any]): The list of allowed values.
        message (str): The error message to display if validation fails.

    Raises:
        ValueError: If the value is not in the list of allowed values.
    """
    if value not in values:
        terms = list_terms(values, quote="'")
        raise ValueError(f"{message} {terms}.")


def choice_items(
    values: Iterable[str | ChoiceItem] | Mapping[object, str],
    *,
    index: Iterable[str] | Mapping[object, str] | None = None,
) -> tuple[ChoiceItem, ...]:
    """Normalize selectable values and enforce unambiguous input.

    Args:
        values (Iterable[str | ChoiceItem] | Mapping[object, str]): Values to normalize. Mapping
            keys become returned values while mapping values become displayed labels.
        index (Iterable[str] | Mapping[object, str] | None): Optional selection indexes to display
            alongside each choice. If omitted, indexes are automatically generated as numeric
            indexes.
    Returns:
        tuple[ChoiceItem, ...]: Ordered choices with normalized values, labels, and selection
            indexes.

    Raises:
        ValueError: If no values are supplied, a name is empty, names are duplicated ignoring
            case, values are duplicated or unhashable, or indexes are invalid or conflict with
            names.
    """
    items = (
        tuple(values.items())
        if isinstance(values, Mapping)
        else tuple(value if isinstance(value, ChoiceItem) else str(value) for value in values)
    )
    if not items:
        raise ValueError("choices cannot be empty.")
    if index is None:
        index = tuple(str(number) for number in range(1, len(items) + 1))
    elif isinstance(index, Mapping):
        keys = {_get_item_key(item) for item in items}
        if set(index) != keys:
            raise ValueError("index must map every choice value to a selection index.")
        index = tuple(index[_get_item_key(item)] for item in items)
    else:
        index = tuple(index)
    if len(index) != len(items):
        raise ValueError("index must map every choice value to a selection index.")
    if any(
        not isinstance(selection, str) or not selection or selection != selection.strip()
        for selection in index
    ):
        raise ValueError("choice indexes must be non-empty strings without surrounding whitespace.")
    choices = tuple(_item_to_choice_item(item, i) for item, i in zip(items, index))
    if any(not isinstance(choice.name, str) or not choice.name for choice in choices):
        raise ValueError("choice names cannot be empty.")
    try:
        values = {choice.value for choice in choices}
    except TypeError as error:
        raise ValueError("choice values must be hashable.") from error
    if len(values) != len(choices):
        raise ValueError("choice values must be unique.")
    if len({choice.index.casefold() for choice in choices}) != len(choices):
        raise ValueError("choice indexes must be unique ignoring case.")
    if len({choice.name.casefold() for choice in choices}) != len(choices):
        raise ValueError("choice names must be unique ignoring case.")
    if {choice.name.casefold() for choice in choices} & {
        choice.index.casefold() for choice in choices
    }:
        raise ValueError("choice names cannot conflict with selection indexes.")
    return choices


def _get_item_key(item: str | tuple[object, str] | ChoiceItem) -> object:
    """Return a unique key for a choice item."""
    if isinstance(item, ChoiceItem):
        return item.value
    if isinstance(item, str):
        return item
    return item[0]


def _item_to_choice_item(item: str | tuple[object, str] | ChoiceItem, index: str) -> ChoiceItem:
    """Convert a single choice item to a normalized ChoiceItem."""
    if isinstance(item, ChoiceItem):
        return ChoiceItem(
            index=index,
            value=item.value,
            name=item.name,
            description=item.description,
        )
    if isinstance(item, str):
        return ChoiceItem(
            index=index,
            value=item,
            name=item,
        )
    value, name = item
    return ChoiceItem(
        index=index,
        value=value,
        name=name,
    )


def format_tool_call_arguments(
    arguments: str,
    *,
    max_chars: int = constants.TOOL_CALL_VALUE_MAX_CHARS,
) -> str:
    """Format tool-call arguments with bounded string values.

    Recursively truncates displayed string values and renders top-level object fields as a
    comma-separated parameter list. Invalid JSON and non-object JSON are treated as opaque strings
    and truncated as a whole.

    Args:
        arguments (str): JSON arguments supplied to a tool.
        max_chars (int): Maximum display length for each string value. Defaults to
            ``TOOL_CALL_VALUE_MAX_CHARS``.

    Returns:
        str: A parameter list with bounded string values, or a bounded raw argument string.

    Raises:
        ValueError: If ``max_chars`` is less than three characters.
    """
    if max_chars < 3:
        raise ValueError("max_chars must be at least 3 to retain a prefix, suffix, and ellipsis.")
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return _truncate_middle(arguments, max_chars)
    if not isinstance(value, dict):
        return _truncate_middle(arguments, max_chars)

    return ", ".join(f"{key}={_truncate_json(item, max_chars)}" for key, item in value.items())


def _truncate_json(value: Any, max_chars: int) -> str:
    """Return a JSON string with bounded string leaves."""
    return json.dumps(
        _truncate_json_strings(value, max_chars),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _truncate_json_strings(value: Any, max_chars: int) -> Any:
    """Return JSON-compatible values with bounded string leaves."""
    if isinstance(value, str):
        return _truncate_middle(value, max_chars)
    if isinstance(value, list):
        return [_truncate_json_strings(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: _truncate_json_strings(item, max_chars) for key, item in value.items()}
    return value


def _truncate_middle(value: str, max_chars: int) -> str:
    """Truncate text at its middle while retaining its start and end."""
    if len(value) <= max_chars:
        return value
    prefix_length = (max_chars - 1 + 1) // 2
    suffix_length = max_chars - 1 - prefix_length
    return f"{value[:prefix_length]}…{value[-suffix_length:]}"


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

    Prefers complete unified-diff hunks with three lines of context. When the first hunk does not
    fit, it reduces context down to zero; when even that cannot fit, it shows a clearly labelled
    bounded change excerpt instead of an empty diff.

    Args:
        before (str): Existing UTF-8 text in the destination file.
        after (str): Replacement UTF-8 text proposed for the destination file.
        path (str): Destination path used in the diff headers.
        max_chars (int): Maximum characters in the rendered diff, excluding its summary.
        max_lines (int): Maximum lines in the rendered diff, excluding its summary.

    Returns:
        str: A summary and bounded unified diff, or a no-change summary.

        The rendered diff never exceeds ``max_chars``/``max_lines`` unless the fixed metadata
        (the file headers, plus the excerpt descriptor and omission notice) is itself larger
        than the budget, in which case only the headers can remain.
    """
    diff_lines = _content_diff_lines(before, after, path, context_lines=3)
    if not diff_lines:
        return "No content changes."

    headers = diff_lines[:2]
    body_lines = diff_lines[2:]
    additions = sum(line.startswith("+") for line in body_lines)
    deletions = sum(line.startswith("-") for line in body_lines)
    first_hunks = _content_diff_hunks(body_lines)

    # Prefer complete unified-diff hunks with three lines of context. Reducing ``n`` changes
    # each hunk's body (fewer unchanged context lines), which lets a hunk fit once the first
    # cannot. Fall back to a bounded, labelled excerpt when no context level fits a hunk.
    for context_lines in range(3, -1, -1):
        hunks = (
            first_hunks
            if context_lines == 3
            else _content_diff_hunks(
                _content_diff_lines(before, after, path, context_lines=context_lines)[2:]
            )
        )
        included, included_hunks = _bounded_diff_hunks(headers, hunks, max_chars, max_lines)
        if included_hunks:
            summary = (
                f"{additions} addition(s), {deletions} deletion(s), {len(hunks)} changed hunk(s)"
            )
            omitted = len(hunks) - included_hunks
            preview = "\n".join((summary, *included))
            if omitted:
                preview += f"\n... ({omitted} changed hunk(s) omitted; preview limit reached)"
            return preview

    summary = (
        f"{additions} addition(s), {deletions} deletion(s), {len(first_hunks)} changed hunk(s)"
    )
    return "\n".join((summary, *_content_diff_excerpt(headers, first_hunks, max_chars, max_lines)))


def _content_diff_lines(before: str, after: str, path: str, *, context_lines: int) -> list[str]:
    """Return unified-diff lines with the requested amount of unchanged context."""
    return list(
        unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=context_lines,
            lineterm="",
        )
    )


def _content_diff_hunks(lines: list[str]) -> list[list[str]]:
    """Group unified-diff body lines into complete hunks."""
    hunks = []
    for line in lines:
        if line.startswith("@@"):
            hunks.append([line])
        else:
            hunks[-1].append(line)
    return hunks


def _bounded_diff_hunks(
    headers: list[str],
    hunks: list[list[str]],
    max_chars: int,
    max_lines: int,
) -> tuple[list[str], int]:
    """Return complete hunks that fit after the unified-diff headers.

    Returns the rendered lines — the headers followed by the hunks that fit — plus the number
    of hunks included. The rendered lines omit the trailing omission notice, which the caller
    appends when some hunks did not fit.
    """
    included = headers.copy()
    included_hunks = 0
    for index, hunk in enumerate(hunks):
        candidate = [*included, *hunk]
        omitted_hunks = len(hunks) - index - 1
        omission = (
            f"... ({omitted_hunks} changed hunk(s) omitted; preview limit reached)"
            if omitted_hunks
            else ""
        )
        rendered = [*candidate, omission] if omission else candidate
        if len(rendered) > max_lines or len("\n".join(rendered)) > max_chars:
            break
        included = candidate
        included_hunks += 1
    return included, included_hunks


def _content_diff_excerpt(
    headers: list[str],
    hunks: list[list[str]],
    max_chars: int,
    max_lines: int,
) -> list[str]:
    """Return a bounded, explicitly non-patch excerpt for an oversized first hunk.

    The result is the preview rendered after the summary line. Changed lines are packed
    inside the budget; when the metadata block (the file headers, descriptor, and omission
    notice) is itself larger than ``max_chars``/``max_lines``, the least-critical metadata
    is trimmed so the excerpt still respects the budget down to the header-only floor.
    """
    changed_lines = [line for hunk in hunks for line in hunk if line.startswith(("+", "-"))]
    required = [
        next((line for line in changed_lines if line.startswith(prefix)), None)
        for prefix in ("-", "+")
    ]
    selected = list(dict.fromkeys(line for line in required if line is not None))
    selected.extend(line for line in changed_lines if line not in selected)
    descriptor = "... (change excerpt; not a complete unified diff)"
    available_lines = max(0, max_lines - len(headers) - 2)
    selected = selected[:available_lines]

    def notice(omitted_lines: int) -> str:
        return (
            f"... ({omitted_lines} changed line(s) and {max(0, len(hunks) - 1)} later hunk(s) "
            "omitted; preview limit reached)"
        )

    final_notice = notice(len(changed_lines) - len(selected))
    available_chars = max_chars - len("\n".join((*headers, descriptor, final_notice)))
    shortened = []
    for index, line in enumerate(selected):
        available_chars -= 1
        remaining_lines = len(selected) - index
        if available_chars < 2 * remaining_lines:
            break
        shortened_line = _truncate_diff_line(line, available_chars // remaining_lines)
        shortened.append(shortened_line)
        available_chars -= len(shortened_line)
    final_notice = notice(len(changed_lines) - len(shortened))
    block = [*headers, descriptor, *shortened, final_notice]

    # Honour the requested budget even when it is smaller than the metadata floor. The
    # omission notice is always final, so truncating from the end drops it first and then
    # any remaining over-budget lines, degrading the preview down to the file headers.
    while len(block) > len(headers) and (
        len(block) > max_lines or len("\n".join(block)) > max_chars
    ):
        block.pop()
    return block


def _truncate_diff_line(line: str, max_chars: int) -> str:
    """Return one changed line bounded by the remaining excerpt character budget."""
    if len(line) <= max_chars:
        return line
    for visible_chars in range(min(len(line), max_chars), -1, -1):
        omitted_chars = len(line) - visible_chars
        candidate = f"{line[:visible_chars]}… ({omitted_chars} chars omitted)"
        if len(candidate) <= max_chars:
            return candidate
    return f"{line[:1]}…"
