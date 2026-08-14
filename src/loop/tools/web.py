"""Provide tools for accessing content on the web."""

import os
from typing import Annotated

import httpx
from pydantic import Field, HttpUrl

from .. import constants
from ..permissions import Capability, PermissionRequest
from ..tooling import tool_registry
from ..utils import cached_metadata, cached_path, read_bounded_text, store_text_stream
from .models import CachedContentResult

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:153.0) Gecko/20100101 Firefox/153.0"
)


def _network_permission(arguments: dict[str, object]) -> tuple[PermissionRequest, ...]:
    """Describe network-read authority for one validated URL."""
    return (
        PermissionRequest(
            tool_name="fetch_content",
            capability=Capability.NETWORK_READ,
            resource=str(arguments["url"]),
        ),
    )


def _cached_content_permission(arguments: dict[str, object]) -> tuple[PermissionRequest, ...]:
    """Recover persisted authority requirements for one cached handle."""
    metadata = (
        None if cached_path(str(arguments["handle"])) else cached_metadata(str(arguments["handle"]))
    )
    source = metadata["source"] if metadata and metadata["reloadable"] else None
    return (
        PermissionRequest(
            tool_name="read_cached_content",
            capability=Capability.NETWORK_READ if source is not None else Capability.PURE,
            resource=str(source or arguments["handle"]),
        ),
    )


def _cache_url(url: str, handle: str | None = None) -> str:
    """Stream one validated web source into the temporary content cache."""
    with httpx.stream(
        "GET",
        url,
        headers={"User-Agent": os.getenv("USER_AGENT", _DEFAULT_USER_AGENT)},
        follow_redirects=True,
        timeout=30.0,
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not (
            content_type.startswith("text/")
            or "json" in content_type
            or "xml" in content_type
            or "javascript" in content_type
        ):
            raise ValueError(f"Content at '{url}' is not a supported text response.")
        try:
            handle, _ = store_text_stream(
                response.iter_bytes(),
                url,
                constants.MAX_FETCH_BYTES,
                handle=handle,
                reloadable=True,
            )
        except ValueError as exc:
            if str(exc) == "Content appears to be binary.":
                raise ValueError(f"Content at '{url}' appears to be binary.") from exc
            if "download limit" in str(exc):
                raise ValueError(str(exc).replace("Content", f"Content at '{url}'", 1)) from exc
            raise
    return handle


@tool_registry.tool(
    capabilities={Capability.NETWORK_READ},
    permission_resolver=_network_permission,
)
def fetch_content(
    url: Annotated[
        HttpUrl,
        Field(description="HTTP(S) URL of the content to fetch."),
    ],
) -> CachedContentResult | str:
    """Fetch text into a bounded cache and return its first resumable portion."""
    url = str(url)
    try:
        handle = _cache_url(url)
        resolved = cached_path(handle)
        if resolved is None:  # pragma: no cover - store and resolve are atomic
            raise RuntimeError("Fetched content could not be cached.")
        path, source = resolved
        return CachedContentResult(
            handle=handle,
            source=source,
            **read_bounded_text(path),
        )
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error fetching content: {exc}"


@tool_registry.tool(permission_resolver=_cached_content_permission)
def read_cached_content(
    handle: Annotated[str, Field(description="Opaque handle returned by a bounded tool result.")],
    start_byte: Annotated[
        int | None,
        Field(
            description="Zero-based byte offset; start_line may remain 1 only at byte zero.", ge=0
        ),
    ] = None,
    start_line: Annotated[
        int | None,
        Field(description="One-based starting line; set to null for byte access.", ge=1),
    ] = 1,
    max_lines: Annotated[
        int | None,
        Field(
            description="Optional line ceiling; the first reached line or byte limit wins.", ge=1
        ),
    ] = None,
    max_bytes: Annotated[
        int,
        Field(
            description="Maximum UTF-8 bytes returned.",
            ge=1,
            le=constants.MAX_TOOL_CONTENT_BYTES,
        ),
    ] = constants.MAX_TOOL_CONTENT_BYTES,
) -> CachedContentResult | str:
    """Read a bounded, resumable portion of cached textual content."""
    try:
        resolved = cached_path(handle)
        if resolved is None:
            metadata = cached_metadata(handle)
            if metadata is None or not metadata["reloadable"]:
                return "Error reading cached content: Unknown or expired content handle."
            _cache_url(metadata["source"], handle)
            resolved = cached_path(handle)
            if resolved is None:  # pragma: no cover - cache writes and lookup are atomic
                raise RuntimeError("Reloaded content could not be cached.")
        path, source = resolved
        return CachedContentResult(
            handle=handle,
            source=source,
            **read_bounded_text(
                path,
                start_byte=start_byte,
                start_line=start_line,
                max_lines=max_lines,
                max_bytes=max_bytes,
            ),
        )
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error reading cached content: {exc}"
