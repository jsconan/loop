"""Export session transcripts and portable archives."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from ..models import Message, Reasoning, ToolCall, ToolResult
from ..utils import content_digest, utc_now
from .models import (
    CompactionEvent,
    ConversationItemEvent,
    PermissionEvent,
    ReferenceArtifact,
    RunCompletedEvent,
    UnsupportedConversationItemError,
)
from .session import Session
from .store import SessionStore

SessionExportFormat = Literal["markdown", "archive"]


def export_session(
    session: Session,
    store: SessionStore,
    destination: Path | str,
    *,
    format: SessionExportFormat = "markdown",
    overwrite: bool = False,
) -> None:
    """Write a Markdown transcript or a self-contained session archive.

    Args:
        session (Session): Session to export.
        store (SessionStore): Store used to read persisted reference artifacts.
        destination (Path | str): File to create or replace.
        format (SessionExportFormat): Export representation. Defaults to ``"markdown"``.
        overwrite (bool): Whether to replace an existing destination. Defaults to ``False``.

    Raises:
        FileExistsError: If the destination exists and replacement was not authorized.
        OSError: If the destination cannot be written.
        UnsupportedConversationItemError: If the session contains an unsupported item type.
        ValueError: If the format, session artifacts, or destination is invalid.
    """
    if format not in {"markdown", "archive"}:
        raise ValueError(f"Unsupported session export format: {format!r}.")

    target = Path(destination).expanduser().resolve()
    store_path = getattr(store, "path", None)
    if store_path is not None and target == Path(store_path).expanduser().resolve():
        raise ValueError("Export destination must not be the session database.")
    if target.is_dir():
        raise ValueError("Export destination must be a file.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Export destination already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.",
        dir=target.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / "export.tmp"
        temporary_path.touch(mode=0o600)
        os.chmod(temporary_path, 0o600)
        if format == "markdown":
            temporary_path.write_text(
                _render_markdown(session),
                encoding="utf-8",
                errors="backslashreplace",
            )
        else:
            _write_archive(temporary_path, session, store)
        with temporary_path.open("rb+") as completed_file:
            completed_file.flush()
            os.fsync(completed_file.fileno())
        if overwrite:
            os.replace(temporary_path, target)
        else:
            os.link(temporary_path, target)


def _render_markdown(session: Session) -> str:
    """Render user-facing session history without provider-private reasoning state."""
    title = (session.name or "Untitled session").replace("\r", " ").replace("\n", " ")
    lines = [
        f"# {title}",
        "",
        f"- Session ID: `{session.id}`",
        f"- Model: `{session.model or 'unknown'}`",
        f"- Exported at: {utc_now().isoformat()}",
        "",
    ]
    recorded_items = set()
    for event in session.events:
        if isinstance(event, ConversationItemEvent):
            recorded_items.add(event.item_index)
            _append_item(lines, session.messages[event.item_index])
        elif isinstance(event, CompactionEvent):
            lines.extend(("## Context compacted", "", "The active context was compacted.", ""))
        elif isinstance(event, PermissionEvent):
            decision = event.result.decision.value
            if event.result.approval_choice is not None:
                decision += f" ({event.result.approval_choice.value})"
            lines.extend(("### Permission decision", "", f"Outcome: {decision}", ""))
        elif isinstance(event, RunCompletedEvent):
            lines.extend(("### Run completed", "", f"Stop reason: {event.stop_reason}", ""))

    for index, item in enumerate(session.messages):
        if index not in recorded_items:
            _append_item(lines, item, pending=True)

    return "\n".join(lines).rstrip() + "\n"


def _append_item(
    lines: list[str],
    item: object,
    *,
    pending: bool = False,
) -> None:
    """Append one conversation item to a Markdown transcript."""
    if isinstance(item, Message):
        role = "User" if item.role == "user" else "Assistant"
        lines.extend((f"## {role}", "", item.content, ""))
    elif isinstance(item, Reasoning):
        if item.summary:
            lines.extend(("### Reasoning summary", "", item.summary, ""))
    elif isinstance(item, ToolCall):
        heading = "Pending tool call" if pending else "Tool call"
        lines.extend(
            (
                f"### {heading}: {item.name}",
                "",
                f"Call ID: `{item.call_id}`",
                "",
                "Arguments:",
                "",
                _code_block(item.arguments, "json"),
                "",
            )
        )
    elif isinstance(item, ToolResult):
        output = item.output
        language = "text"
        try:
            output = json.dumps(json.loads(output), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
        else:
            language = "json"
        lines.extend(
            (
                f"### Tool result: `{item.call_id}`",
                "",
                _code_block(output, language),
                "",
            )
        )
    else:
        raise UnsupportedConversationItemError(
            f"Unsupported conversation item type: {type(item).__name__}."
        )


def _code_block(value: str, language: str = "text") -> str:
    """Fence arbitrary text without allowing embedded backticks to close the block."""
    fence = "```"
    while fence in value:
        fence += "`"
    return f"{fence}{language}\n{value}\n{fence}"


def _write_archive(path: Path, session: Session, store: SessionStore) -> None:
    """Write versioned session data and all referenced immutable artifacts to a ZIP file."""
    snapshot = session.persistence_snapshot()
    artifacts = _collect_artifacts(session, snapshot.reference_artifacts, store)
    artifact_entries = [
        {
            "digest": digest,
            "path": f"artifacts/sha256/{digest.removeprefix('sha256:')}",
            "size_bytes": len(content),
        }
        for digest, content in sorted(artifacts.items())
    ]
    manifest = {
        "format": "loop-session-archive",
        "format_version": 1,
        "exported_at": utc_now().isoformat(),
        "session_id": session.id,
        "session_file": "session.json",
        "artifacts": artifact_entries,
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8",
                errors="backslashreplace",
            ),
        )
        archive.writestr("session.json", snapshot.payload)
        for entry in artifact_entries:
            archive.writestr(
                entry["path"],
                artifacts[entry["digest"]],
                compress_type=ZIP_STORED,
            )


def _collect_artifacts(
    session: Session,
    transient_artifacts: tuple[ReferenceArtifact, ...],
    store: SessionStore,
) -> dict[str, bytes]:
    """Load and verify every immutable content reference used by the session."""
    transient = {artifact.digest: artifact.content for artifact in transient_artifacts}
    artifacts = {}
    for item in session.messages:
        if not isinstance(item, Message):
            continue
        for reference in item.context:
            if reference.version is None:
                raise ValueError(f"Referenced snapshot '{reference.path}' has incomplete identity.")
            content = transient.get(reference.version)
            if content is None and reference.size_bytes == 0:
                content = b""
            elif content is None:
                content = store.load_reference(reference.version)
            if content is None:
                raise ValueError(f"Referenced snapshot '{reference.path}' is unavailable.")
            if (
                not isinstance(content, bytes)
                or content_digest(content) != reference.version
                or len(content) != reference.size_bytes
            ):
                raise ValueError(f"Referenced snapshot '{reference.path}' failed integrity check.")
            artifacts[reference.version] = content
    return artifacts
