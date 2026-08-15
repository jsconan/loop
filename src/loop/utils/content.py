"""Bound textual tool content and retain oversized artifacts outside model context."""

import base64
import binascii
import json
from codecs import getincrementaldecoder
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import UUID, uuid4

from .. import constants
from .models import BoundedTextContent, CachedContentMetadata

_CACHE = TemporaryDirectory(prefix="loop-content-")
_SOURCES: dict[str, str] = {}
_METADATA: dict[str, CachedContentMetadata] = {}
_LOCK = RLock()
_SCAN_CHUNK_BYTES = 8 * 1024


def encode_content_cursor(handle: str, start_byte: int) -> str:
    """Encode an opaque continuation cursor for cached content.

    Args:
        handle (str): Canonical handle identifying the cached content.
        start_byte (int): Non-negative byte offset at which reading should resume.

    Returns:
        str: URL-safe opaque continuation cursor bound to ``handle``.

    Raises:
        ValueError: If the handle or byte offset is invalid.
    """
    _validate_handle(handle)
    if start_byte < 0:
        raise ValueError("Content cursor byte offset must be non-negative.")
    payload = json.dumps([1, handle, start_byte], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_content_cursor(cursor: str, handle: str) -> int:
    """Decode a cached-content cursor and return its byte offset.

    Args:
        cursor (str): Opaque cursor previously returned for cached content.
        handle (str): Canonical handle of the content being read.

    Returns:
        int: Non-negative byte offset at which reading should resume.

    Raises:
        ValueError: If the cursor is malformed, unsupported, or belongs to another handle.
    """
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        value = json.loads(payload)
        version, cursor_handle, start_byte = value
        if (
            version != 1
            or cursor_handle != handle
            or not isinstance(start_byte, int)
            or isinstance(start_byte, bool)
            or start_byte < 0
        ):
            raise ValueError
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid cached content cursor.") from exc
    return start_byte


def _validate_handle(handle: str) -> None:
    """Validate one canonical UUID content handle."""
    try:
        normalized = UUID(handle).hex
    except ValueError as exc:
        raise ValueError("Invalid content handle.") from exc
    if normalized != handle:
        raise ValueError("Invalid content handle.")


def register_cached_metadata(handle: str, source: str, reloadable: bool) -> None:
    """Register artifact metadata loaded from a tool result or newly cached source.

    Args:
        handle (str): Opaque canonical artifact handle.
        source (str): Human-readable source for the content.
        reloadable (bool): Whether the source can recreate an expired artifact.

    Raises:
        ValueError: If the handle or metadata values are invalid.
    """
    _validate_handle(handle)
    if not isinstance(source, str) or not isinstance(reloadable, bool):
        raise ValueError("Invalid cached content metadata.")
    with _LOCK:
        _METADATA[handle] = CachedContentMetadata(source=source, reloadable=reloadable)


def store_content(content: bytes | str, source: str) -> str:
    """Store content outside conversation history and return an opaque handle.

    Args:
        content (bytes | str): Raw content to retain.
        source (str): Human-readable origin reported with later reads.

    Returns:
        str: Opaque process-local content handle.
    """
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    handle = uuid4().hex
    with _LOCK:
        Path(_CACHE.name, handle).write_bytes(encoded)
        _SOURCES[handle] = source
        register_cached_metadata(handle, source, False)
    return handle


def store_text_stream(
    chunks: Iterable[bytes],
    source: str,
    max_bytes: int,
    *,
    handle: str | None = None,
    reloadable: bool = False,
) -> tuple[str, int]:
    """Stream validated UTF-8 text into the artifact cache under a hard byte ceiling.

    Args:
        chunks (Iterable[bytes]): Raw response chunks to validate and retain.
        source (str): Human-readable origin reported with later reads.
        max_bytes (int): Maximum complete content size accepted.
        handle (str | None): Existing canonical handle to repopulate, or ``None`` to create one.
        reloadable (bool): Whether session metadata identifies a reproducible source.

    Returns:
        tuple[str, int]: Opaque handle and complete stored byte size.

    Raises:
        ValueError: If content is binary, invalid UTF-8, or exceeds ``max_bytes``.
    """
    handle = handle or uuid4().hex
    path = Path(_CACHE.name, handle)
    _validate_handle(handle)
    size = 0
    decoder = getincrementaldecoder("utf-8")()
    try:
        with path.open("wb") as artifact:
            for chunk in chunks:
                if b"\0" in chunk:
                    raise ValueError("Content appears to be binary.")
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"Content exceeds the {max_bytes}-byte download limit.")
                decoder.decode(chunk)
                artifact.write(chunk)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError, ValueError:
        path.unlink(missing_ok=True)
        raise
    with _LOCK:
        _SOURCES[handle] = source
        register_cached_metadata(handle, source, reloadable)
    return handle, size


def cached_path(handle: str) -> tuple[Path, str] | None:
    """Resolve an opaque content handle to its cache path and source.

    Args:
        handle (str): Handle previously returned by :func:`store_content`.

    Returns:
        tuple[Path, str] | None: Cache path and source, or ``None`` for an unknown handle.
    """
    with _LOCK:
        source = _SOURCES.get(handle)
        if source is None:
            return None
        return Path(_CACHE.name, handle), source


def cached_metadata(handle: str) -> CachedContentMetadata | None:
    """Return session-restored source metadata for a current or expired handle.

    Args:
        handle (str): Opaque canonical content handle.

    Returns:
        CachedContentMetadata | None: Registered metadata, or ``None`` when unavailable.
    """
    try:
        _validate_handle(handle)
    except ValueError:
        return None
    with _LOCK:
        return _METADATA.get(handle)


def bound_tool_result(output: str, source: str) -> tuple[str, str | None]:
    """Replace an oversized serialized tool result with a bounded preview and cache handle.

    Args:
        output (str): Complete serialized tool result.
        source (str): Human-readable origin for the cached content.

    Returns:
        tuple[str, str | None]: Bounded result and its handle, or the unchanged result and
            ``None`` when it already fits.
    """
    encoded = output.encode("utf-8")
    if len(encoded) <= constants.MAX_TOOL_RESULT_BYTES:
        return output, None
    handle = store_content(encoded, source)
    preview_bytes = min(constants.MAX_TOOL_CONTENT_BYTES // 2, len(encoded))
    while True:
        preview = encoded[:preview_bytes].decode("utf-8", errors="ignore")
        result = json.dumps(
            {
                "content": preview,
                "size_bytes": len(encoded),
                "included_bytes": len(preview.encode("utf-8")),
                "truncated": True,
                "truncation_reason": "tool_result_limit",
                "handle": handle,
                "next_cursor": encode_content_cursor(handle, len(preview.encode("utf-8"))),
                "message": "Use read_cached_content with this handle and cursor to continue.",
            }
        )
        if len(result.encode("utf-8")) <= constants.MAX_TOOL_RESULT_BYTES:
            return result, handle
        preview_bytes //= 2


def _seek_to_line(stream, start_line: int) -> int:
    """Seek to a one-based line using fixed-size reads and return the reached line."""
    current_line = 1
    while current_line < start_line:
        chunk_start = stream.tell()
        chunk = stream.read(_SCAN_CHUNK_BYTES)
        if not chunk:
            break
        for index, value in enumerate(chunk):
            if value == ord("\n"):
                current_line += 1
                if current_line == start_line:
                    stream.seek(chunk_start + index + 1)
                    return current_line
    return current_line


def _validate_boundaries(
    start_byte: int | None,
    start_line: int | None,
    max_lines: int | None,
    max_bytes: int,
) -> tuple[int | None, int | None]:
    """Validate and normalize byte and line boundaries."""
    if start_byte == 0 and start_line == 1:
        if max_lines is None:
            start_line = None
        else:
            start_byte = None
    elif start_byte is not None and start_line is not None:
        raise ValueError("Specify either start_byte or start_line, not both.")
    if start_byte is None and start_line is None:
        start_line = 1
    if start_byte is not None and start_byte < 0:
        raise ValueError("start_byte must be non-negative.")
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be at least 1.")
    if max_lines is not None and max_lines < 1:
        raise ValueError("max_lines must be at least 1.")
    if max_bytes < 1 or max_bytes > constants.MAX_TOOL_CONTENT_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {constants.MAX_TOOL_CONTENT_BYTES}.")
    return start_byte, start_line


def read_bounded_text(
    path: Path,
    *,
    start_byte: int | None = None,
    start_line: int | None = 1,
    max_lines: int | None = None,
    max_bytes: int = constants.MAX_TOOL_CONTENT_BYTES,
    preserve_line_boundaries: bool = False,
) -> BoundedTextContent:
    """Read a UTF-8 text range subject to independent byte and line ceilings.

    Args:
        path (Path): File containing the text to read.
        start_byte (int | None): Zero-based byte offset for direct access. Mutually exclusive with
            a non-default ``start_line``. At the shared byte-zero/line-one origin, an explicit
            ``max_lines`` selects line mode and otherwise ``start_byte`` selects byte mode.
        start_line (int | None): One-based starting line, or ``None`` for byte-oriented access.
        max_lines (int | None): Optional maximum lines returned from either starting mode.
            Defaults to no line limit. When both ceilings are set, the first reached stops reading.
        max_bytes (int): Requested encoded byte ceiling, capped by the application hard limit.
        preserve_line_boundaries (bool): Whether line-oriented reads stop before consuming a
            partial line. Defaults to ``False`` for byte-resumable consumers.

    Returns:
        BoundedTextContent: Text, exact byte range, and continuation metadata.

    Raises:
        ValueError: If ranges are invalid, both starting modes are selected, the offset splits a
            UTF-8 character, or selected content is not UTF-8 text.
    """
    start_byte, start_line = _validate_boundaries(start_byte, start_line, max_lines, max_bytes)

    size = path.stat().st_size
    with path.open("rb") as stream:
        probe = stream.read(min(size, _SCAN_CHUNK_BYTES))
        if b"\0" in probe:
            raise ValueError("Content appears to be binary.")
        stream.seek(0)
        current_line = 1
        if start_byte is not None:
            stream.seek(start_byte)
            start = start_byte
        else:
            current_line = _seek_to_line(stream, start_line)
            start = stream.tell()

        chunks = []
        remaining = max_bytes
        lines_read = 0
        stopped_by_lines = False
        line_too_long = False
        while remaining:
            if max_lines is not None and lines_read >= max_lines:
                stopped_by_lines = True
                break
            chunk = stream.readline(remaining + 1)
            if not chunk:
                break
            if len(chunk) > remaining:
                if start_line is not None and preserve_line_boundaries:
                    stream.seek(-len(chunk), 1)
                    line_too_long = not chunks
                    break
                stream.seek(-(len(chunk) - remaining), 1)
                chunk = chunk[:remaining]
            chunks.append(chunk)
            remaining -= len(chunk)
            if chunk.endswith((b"\n", b"\r")):
                lines_read += 1
            elif remaining:
                lines_read += 1

        end = stream.tell()
        encoded = b"".join(chunks)
        if b"\0" in encoded:
            raise ValueError("Content appears to be binary.")
        try:
            content = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.end != len(encoded) or exc.reason != "unexpected end of data":
                raise ValueError("Selected range is not valid UTF-8 text.") from exc
            omitted = len(encoded) - exc.start
            encoded = encoded[: exc.start]
            end -= omitted
            content = encoded.decode("utf-8")
        more = end < size

    result = BoundedTextContent(
        content=content,
        size_bytes=size,
        start_byte=start,
        end_byte=end,
        included_bytes=len(encoded),
        truncated=more,
    )
    if start_line is not None:
        result["start_line"] = current_line
        if content:
            logical_lines = content.count("\n") + (not content.endswith("\n"))
            result["end_line"] = current_line + logical_lines - 1
        else:
            result["end_line"] = current_line - 1
    if more:
        if line_too_long:
            result["truncation_reason"] = "line_too_long"
        else:
            result["truncation_reason"] = "lines" if stopped_by_lines else "bytes"
        result["next_start_byte"] = end
        if start_line is not None and not line_too_long:
            result["next_start_line"] = current_line + lines_read
    return result
