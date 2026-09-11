"""Project durable conversation data onto the explicit model-visible contract."""

from ..models import (
    CompactionContextItem,
    Message,
    ModelContextItem,
    Reasoning,
    ToolCall,
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
        return {"provider": item.provider, "data": dict(item.data)}
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
    if isinstance(item, Reasoning):
        return {"content": item.content, "summary": item.summary}
    if isinstance(item, ToolCall):
        return {"call_id": item.call_id, "name": item.name, "arguments": item.arguments}
    if isinstance(item, CompactionContextItem):
        semantic = {
            key: value
            for key, value in item.data.items()
            if key in {"content", "summary", "checkpoint", "text"} and isinstance(value, str)
        }
        return {"provider": item.provider, "data": semantic} if semantic else None
    return project_context(item)
