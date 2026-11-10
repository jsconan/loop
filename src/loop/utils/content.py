"""Bound textual tool content and retain oversized artifacts outside model context."""

import base64
import binascii
import json
import os
from atexit import register
from codecs import getincrementaldecoder
from collections.abc import Iterable
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import UUID, uuid4

from .. import constants
from .hashing import sha256_digest
from .json import load_unique_json
from .models import BoundedTextContent, CachedContentMetadata

_CACHE = TemporaryDirectory(prefix=constants.TEMPORARY_CONTENT_DIRECTORY_PREFIX)
register(_CACHE.cleanup)
_SOURCES: dict[str, str] = {}
_METADATA: dict[str, CachedContentMetadata] = {}
_LOCK = RLock()
_SCAN_CHUNK_BYTES = 8 * 1024


def get_binary(content: str | bytes) -> bytes:
    """Get the binary representation of the content.

    Args:
        content (str | bytes): The content to convert.

    Returns:
        bytes: The binary representation of the content.
    """
    if isinstance(content, str):
        return content.encode("utf-8")
    return content


def base64_encode(content: str | bytes) -> str:
    """Encode content as a base64 string.

    Args:
        content (str | bytes): The content to encode.

    Returns:
        str: Base64-encoded representation of the content.
    """
    content = get_binary(content)
    return base64.b64encode(content).decode("ascii")


def base64_decode(encoded: str, binary: bool = False) -> str | bytes:
    """Decode a base64 string back into bytes.

    Args:
        encoded (str): Base64-encoded content.
        binary (bool): If True, return raw bytes; otherwise, decode as UTF-8.

    Returns:
        str | bytes: The original content, either as UTF-8 decoded string or raw bytes.
    """
    decoded = base64.b64decode(encoded, validate=True)
    return decoded if binary else decoded.decode("utf-8")


def data_url(content: str | bytes, media_type: str) -> str:
    """Encode content as a MIME-qualified data URL.

    Args:
        content (str | bytes): The content to encode.
        media_type (str): The MIME type of the content.

    Returns:
        str: A data URL representing the encoded content.
    """
    encoded = base64_encode(content)
    return f"data:{media_type};base64,{encoded}"


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
    validate_content_handle(handle)
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


def validate_content_handle(handle: str) -> None:
    """Validate an opaque content-read capability.

    Args:
        handle (str): Capability supplied by a persisted reference or tool request.

    Raises:
        ValueError: If the capability is not a canonical random UUID.
    """
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
    validate_content_handle(handle)
    if not isinstance(source, str) or not isinstance(reloadable, bool):
        raise ValueError(  # noqa: TRY004 - cached metadata has one validation contract.
            "Invalid cached content metadata."
        )
    with _LOCK:
        _METADATA[handle] = CachedContentMetadata(source=source, reloadable=reloadable)


def store_content(content: bytes | str, source: str, *, handle: str | None = None) -> str:
    """Store content outside conversation history and return an opaque handle.

    Args:
        content (bytes | str): Raw content to retain.
        source (str): Human-readable origin reported with later reads.
        handle (str | None): Existing canonical handle to populate, or ``None`` to create one.

    Returns:
        str: Opaque process-local content handle.
    """
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    if handle:
        validate_content_handle(handle)
    else:
        handle = uuid4().hex
    with _LOCK:
        Path(_CACHE.name, handle).write_bytes(encoded)
        _SOURCES[handle] = source
        register_cached_metadata(handle, source, False)
    return handle


def content_identity(content: bytes | str) -> tuple[str, str]:
    """Return an opaque capability and complete digest for immutable content.

    Args:
        content (bytes | str): Immutable content to identify.

    Returns:
        tuple[str, str]: Unpredictable cache handle and algorithm-qualified SHA-256 identity.
    """
    return uuid4().hex, content_digest(content)


def content_digest(content: bytes | str) -> str:
    """Return the algorithm-qualified SHA-256 identity for immutable content.

    Args:
        content (bytes | str): Immutable content to identify.

    Returns:
        str: Algorithm-qualified SHA-256 artifact identity.
    """
    return f"sha256:{sha256_digest(content)}"


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
    validate_content_handle(handle)
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
    except (UnicodeDecodeError, ValueError):
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
        validate_content_handle(handle)
    except ValueError:
        return None
    with _LOCK:
        return _METADATA.get(handle)


def _bound_file_result(output: str, source: str) -> tuple[str, str] | None:
    """Retain a verified file page and cache its original serialized result when oversized."""
    try:
        payload = load_unique_json(output, duplicate_message="Duplicate tool-result field.")
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True or "handle" in payload:
        return None
    result, receipts = payload.get("result"), payload.get("read_versions")
    if not isinstance(result, dict) or not isinstance(receipts, dict):
        return None
    path, version, content = result.get("path"), result.get("version"), result.get("content")
    start, end, size = result.get("start_byte"), result.get("end_byte"), result.get("size_bytes")
    if (
        not isinstance(path, str)
        or not isinstance(version, str)
        or receipts.get(path) != version
        or not isinstance(content, str)
        or not content
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in (start, end, size)
        )
        or start < 0
        or end > size
    ):
        return None
    try:
        content_bytes = content.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if end != start + len(content_bytes):
        return None
    start_line = result.get("start_line")
    if start_line is not None and (
        not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1
    ):
        return None
    handle = uuid4().hex
    bounded = {
        **payload,
        "handle": handle,
        "cached_snapshot_format": "serialized_tool_result",
        "result": dict(result),
    }
    page = bounded["result"]
    page.pop("next_start_line", None)
    page["truncated"] = True
    page["truncation_reason"] = "result_limit"
    low, high, best = 1, len(content), None
    while low <= high:
        middle = (low + high) // 2
        text = content[:middle]
        included = len(text.encode("utf-8"))
        page.update(
            content=text,
            included_bytes=included,
            end_byte=start + included,
            next_start_byte=start + included,
        )
        if "requested_max_bytes" in page:
            page["effective_max_bytes"] = included
        if isinstance(page.get("start_line"), int):
            page["end_line"] = page["start_line"] + text.count("\n") - int(text.endswith("\n"))
        try:
            candidate = json.dumps(bounded, ensure_ascii=False)
            candidate_size = len(candidate.encode("utf-8"))
        except (UnicodeEncodeError, RecursionError):
            return None
        if candidate_size <= constants.MAX_TOOL_RESULT_BYTES:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        return None
    store_content(output, source, handle=handle)
    return best, handle


def _bound_command_result(output: str, source: str) -> tuple[str, str] | None:
    """Keep command status and stream continuations when serialized previews exceed the limit."""
    try:
        payload = load_unique_json(output, duplicate_message="Duplicate tool-result field.")
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or "handle" in payload:
        return None
    if payload.get("ok") is True and isinstance(payload.get("result"), dict):
        body = payload["result"]
        bounded = {**payload, "result": dict(body)}
        target = bounded["result"]
    elif payload.get("ok") is False and isinstance(payload.get("problem"), dict):
        problem = payload["problem"]
        if not isinstance(problem.get("metadata"), dict):
            return None
        body = problem["metadata"]
        bounded = {**payload, "problem": {**problem, "metadata": dict(body)}}
        target = bounded["problem"]["metadata"]
    else:
        return None
    exit_code = body.get("exit_code")
    if (
        body.get("boundary") not in ("sandbox", "host")
        or "exit_code" not in body
        or (
            exit_code is not None
            and (not isinstance(exit_code, int) or isinstance(exit_code, bool))
        )
    ):
        return None
    streams = {}
    for name in ("stdout", "stderr"):
        item = body.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            return None
        try:
            data = item["content"].encode("utf-8")
        except UnicodeEncodeError:
            return None
        captured, included = item.get("captured_bytes"), item.get("included_bytes")
        discarded = item.get("discarded_characters")
        if (
            not isinstance(captured, int)
            or isinstance(captured, bool)
            or not isinstance(included, int)
            or isinstance(included, bool)
            or included != len(data)
            or captured < included
            or not isinstance(item.get("truncated"), bool)
            or not isinstance(item.get("capture_complete"), bool)
            or "discarded_characters" not in item
            or (
                discarded is not None
                and (not isinstance(discarded, int) or isinstance(discarded, bool) or discarded < 0)
            )
            or item["capture_complete"] != (discarded == 0)
            or item["truncated"] != (included < captured or discarded != 0)
            or item.get("decoding_loss") not in ("none", "unknown")
        ):
            return None
        previous = item.get("handle")
        if previous is None:
            if captured != included:
                return None
        else:
            if not isinstance(previous, str):
                return None
            try:
                validate_content_handle(previous)
                cached = cached_path(previous)
                if cached is None or cached[0].stat().st_size != captured:
                    return None
                with cached[0].open("rb") as stream:
                    if stream.read(included) != data:
                        return None
                if decode_content_cursor(item.get("next_cursor"), previous) != included:
                    return None
            except (OSError, TypeError, ValueError):
                return None
        streams[name] = (item["content"], data, previous)
        target[name] = dict(item)

    snapshot = uuid4().hex
    bounded["handle"] = snapshot
    bounded["cached_snapshot_format"] = "serialized_tool_result"
    pending = []
    for name in sorted(
        streams,
        key=lambda key: len(json.dumps(streams[key][0], ensure_ascii=False).encode("utf-8")),
        reverse=True,
    ):
        text, data, previous = streams[name]
        if not text:
            continue
        handle = previous or uuid4().hex
        page = target[name]
        page.update(
            handle=handle,
            truncated=True,
            continuation=page.get(
                "continuation", "Continue this stream using its handle and cursor."
            ),
        )
        if previous is None:
            pending.append((handle, data, name))
        low, high, best = 0, len(text) - 1, None
        while low <= high:
            middle = (low + high) // 2
            prefix = text[:middle]
            included = len(prefix.encode("utf-8"))
            page.update(
                content=prefix,
                included_bytes=included,
                next_cursor=encode_content_cursor(handle, included),
            )
            try:
                candidate = json.dumps(bounded, ensure_ascii=False)
                candidate_size = len(candidate.encode("utf-8"))
            except (UnicodeEncodeError, RecursionError):
                return None
            if candidate_size <= constants.MAX_TOOL_RESULT_BYTES:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            for content_handle, content, stream_name in pending:
                store_content(content, f"{source} {stream_name}", handle=content_handle)
            store_content(output, source, handle=snapshot)
            return best, snapshot
        page.update(content="", included_bytes=0, next_cursor=encode_content_cursor(handle, 0))
    return None


def bound_tool_result(output: str, source: str) -> tuple[str, str | None]:
    """Bound a serialized result while preserving file and command continuations.

    Args:
        output (str): Complete serialized tool result.
        source (str): Human-readable origin for the cached content.

    Returns:
        tuple[str, str | None]: Bounded result and its original-output cache handle, or the
            unchanged result and ``None`` when it already fits.
    """
    encoded = output.encode("utf-8")
    if len(encoded) <= constants.MAX_TOOL_RESULT_BYTES:
        return output, None
    if file_result := _bound_file_result(output, source):
        return file_result
    if command_result := _bound_command_result(output, source):
        return command_result
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
    include_version: bool = False,
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
            If the first character cannot fit, the reader includes that one character instead.
        preserve_line_boundaries (bool): Whether line-oriented reads prefer whole lines. The
            requested byte ceiling expands as needed for the first line, up to the hard limit;
            larger lines return a byte-resumable fragment. Defaults to ``False``.
        include_version (bool): Whether to hash the complete source through the read descriptor
            and reject changes during capture. Defaults to ``False``.

    Returns:
        BoundedTextContent: Text, exact byte range, continuation metadata, and the complete source
            digest when ``include_version`` is enabled. When the first character or line
            exceeds the requested ceiling, requested and effective limits identify the adjustment.

    Raises:
        ValueError: If ranges are invalid, both starting modes are selected, the offset is beyond
            the source or splits a UTF-8 character, selected content is not UTF-8 text, or a
            versioned source changes.
    """
    start_byte, start_line = _validate_boundaries(start_byte, start_line, max_lines, max_bytes)

    version = None
    size = path.stat().st_size
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno()) if include_version else None
        if before is not None:
            size = before.st_size
        if start_byte is not None and start_byte > size:
            raise ValueError("start_byte is beyond the end of the content.")
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
        split_line = False
        adjusted_limit = None
        while remaining:
            if max_lines is not None and lines_read >= max_lines:
                stopped_by_lines = True
                break
            chunk = stream.readline(remaining + 1)
            if not chunk:
                break
            if len(chunk) > remaining:
                if start_line is not None and preserve_line_boundaries:
                    if chunks:
                        stream.seek(-len(chunk), 1)
                        break
                    stream.seek(start)
                    chunk = stream.readline(constants.MAX_TOOL_CONTENT_BYTES + 1)
                    split_line = len(chunk) > constants.MAX_TOOL_CONTENT_BYTES
                    if split_line:
                        stream.seek(-1, 1)
                        chunk = chunk[:-1]
                    chunks.append(chunk)
                    adjusted_limit = len(chunk) if len(chunk) > max_bytes else None
                    lines_read = 0 if split_line else 1
                    stopped_by_lines = max_lines is not None and lines_read >= max_lines
                    break
                stream.seek(-(len(chunk) - remaining), 1)
                chunk = chunk[:remaining]
            chunks.append(chunk)
            remaining -= len(chunk)
            if chunk.endswith((b"\n", b"\r")) or remaining:
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
            if exc.start:
                if end == size:
                    raise ValueError("Selected range is not valid UTF-8 text.") from exc
                omitted = len(encoded) - exc.start
                encoded = encoded[: exc.start]  # pylint: disable=invalid-slice-index
                end -= omitted
                content = encoded.decode("utf-8")
            else:
                stream.seek(start)
                decoder = getincrementaldecoder("utf-8")()
                encoded, content = b"", ""
                while not content:
                    byte = stream.read(1)
                    if not byte:
                        raise ValueError("Selected range is not valid UTF-8 text.") from exc
                    encoded += byte
                    try:
                        content = decoder.decode(byte)
                    except UnicodeDecodeError as error:
                        raise ValueError("Selected range is not valid UTF-8 text.") from error
                end = start + len(encoded)
                adjusted_limit = len(encoded)
        more = end < size
        if before is not None:
            stream.seek(0)
            version = f"sha256:{file_digest(stream, 'sha256').hexdigest()}"
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(
                getattr(before, field) != getattr(current, field)
                for current in (os.fstat(stream.fileno()), path.stat())
                for field in fields
            ):
                raise ValueError("File changed while reading; read it again before editing.")

    result = BoundedTextContent(
        content=content,
        size_bytes=size,
        start_byte=start,
        end_byte=end,
        included_bytes=len(encoded),
        truncated=more,
    )
    if version is not None:
        result["version"] = version
    if adjusted_limit is not None:
        result["requested_max_bytes"] = max_bytes
        result["effective_max_bytes"] = adjusted_limit
    if start_line is not None:
        result["start_line"] = current_line
        if content:
            logical_lines = content.count("\n") + (not content.endswith("\n"))
            result["end_line"] = current_line + logical_lines - 1
        else:
            result["end_line"] = current_line - 1
    if more:
        result["truncation_reason"] = "lines" if stopped_by_lines else "bytes"
        result["next_start_byte"] = end
        if start_line is not None and not split_line:
            result["next_start_line"] = current_line + lines_read
    return result
