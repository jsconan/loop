"""Provide general text formatting utilities."""

import json
from collections.abc import Iterable, Mapping
from difflib import unified_diff
from typing import Any

from .. import constants
from .models import ChoiceItem


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
