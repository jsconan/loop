"""Project durable conversation data onto the explicit model-visible contract."""

import json
from collections.abc import Iterable

from ..models import (
    CompactionContextItem,
    DeveloperInput,
    Message,
    ModelContextItem,
    Reasoning,
    ReasoningConfiguration,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def project_context(item: ModelContextItem) -> dict:
    """Select only fields required to continue a conversation.

    Local snapshots, response metadata and artifact storage records are deliberately excluded.
    Provider checkpoint data remains provider-owned; adapters must validate its supported shape.

    Args:
        item (ModelContextItem): Durable conversation item to project.

    Returns:
        dict: Explicit semantic and protocol fields for generation or compaction.

    Raises:
        TypeError: If the item has no declared model-visible representation.
    """
    if isinstance(item, DeveloperInput):
        return {"role": "developer", "content": item.content}
    if isinstance(item, ReasoningConfiguration):
        return {"type": "configuration_update", "reasoning": {"effort": item.effort}}
    if isinstance(item, Message):
        return {
            "role": item.role,
            "content": item.content,
            "context": [
                {
                    "kind": ref.kind,
                    "path": ref.path,
                    "content": ref.content,
                    "size_bytes": ref.size_bytes,
                    "included_bytes": ref.included_bytes,
                    "truncated": ref.truncated,
                    "handle": ref.handle,
                    "next_cursor": ref.next_cursor,
                    "version": ref.version,
                    "media_type": ref.media_type,
                    "payload_start_bytes": ref.payload_start_bytes,
                    "reused": ref.reused,
                    "payload_redacted": ref.payload_redacted,
                }
                for ref in item.context
            ],
        }
    if isinstance(item, Reasoning):
        return {
            "content": item.content,
            "summary": item.summary,
            "encrypted_content": item.encrypted_content,
            "status": item.status,
            "id": item.id,
        }
    if isinstance(item, ToolCall):
        return {
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
            "id": item.id,
        }
    if isinstance(item, ToolResult):
        return {"call_id": item.call_id, "output": item.output}
    if isinstance(item, CompactionContextItem):
        return item.model_dump()
    raise TypeError(f"Unsupported conversation item: {type(item).__name__}")


def project_portable_context(item: ModelContextItem) -> dict | None:
    """Select semantic history safe for provider-independent summarization.

    Args:
        item (ModelContextItem): Durable conversation item to summarize.

    Returns:
        dict | None: Semantic continuation data, or ``None`` for an opaque-only checkpoint.

    Raises:
        TypeError: If the item has no declared model-visible representation.
    """
    if isinstance(item, (DeveloperInput, ReasoningConfiguration)):
        return None
    if isinstance(item, Reasoning):
        return {"content": item.content, "summary": item.summary}
    if isinstance(item, ToolCall):
        return {"call_id": item.call_id, "name": item.name, "arguments": item.arguments}
    if isinstance(item, CompactionContextItem):
        return None
    projected = project_context(item)
    if isinstance(item, Message):
        for reference in projected["context"]:
            if isinstance(reference["content"], bytes):
                reference["content"] = None
    return projected


def estimate_context(
    items: Iterable[ModelContextItem],
    *,
    instructions: str | None = None,
    tools: Iterable[ToolDefinition] = (),
) -> int:
    """Estimate tokens from projected text, media, tool schemas, and item framing.

    Args:
        items (Iterable[ModelContextItem]): Model-visible context items.
        instructions (str | None): Separate leading or trailing developer instructions.
        tools (Iterable[ToolDefinition]): Ordered function definitions.

    Returns:
        int: Conservative request estimate for budgeting, independent of billing.
    """
    tokens = 0
    for item in items:
        projected = project_context(item)
        if isinstance(item, Reasoning):
            projected.pop("encrypted_content")
        if isinstance(item, Message):
            references = projected.pop("context")
            for reference in references:
                content = reference.pop("content")
                media_type = reference["media_type"] or ""
                if not reference["reused"] and reference["included_bytes"]:
                    if isinstance(content, bytes) or (
                        not content
                        and media_type.startswith(("application/pdf", "image/", "audio/", "video/"))
                    ):
                        tokens += max(1024, reference["included_bytes"] // 32)
                    else:
                        tokens += _estimate_text(content, reference["included_bytes"])
                projected.setdefault("references", []).append(reference)
        tokens += _estimate_json(projected) + 8
    return (
        tokens
        + _estimate_json(instructions or "")
        + sum(_estimate_json(tool.model_dump()) + 8 for tool in tools)
    )


def _estimate_text(content: str, include_bytes: int) -> int:
    """Estimate tokens for text content, falling back to included bytes if empty."""
    size = len(content.encode()) if content else include_bytes
    return (size + 2) // 3


def _estimate_json(value: object) -> int:
    """Estimate text tokens with extra room for punctuation, markup, and Unicode."""
    return _estimate_text(json.dumps(value, ensure_ascii=False), include_bytes=0)
