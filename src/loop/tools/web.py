"""Provide tools for accessing content on the web."""

import ipaddress
import os
import socket
from typing import Annotated
from urllib.parse import urlsplit

import httpcore
import httpx
from pydantic import Field, HttpUrl

from .. import constants
from ..models import ToolResultPresentation, ToolResultPresentationSpec
from ..permissions import Action, NetworkTarget, Operation, OperationPlan
from ..tooling import ToolContext, tool
from ..utils import (
    BoundedTextContent,
    cached_metadata,
    cached_path,
    decode_content_cursor,
    encode_content_cursor,
    read_bounded_text,
    store_text_stream,
)
from .models import CachedContentResult

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:153.0) Gecko/20100101 Firefox/153.0"
)


class PinnedAddressBackend:
    """Connect HTTP clients only to the addresses resolved during authorization planning."""

    _addresses: tuple[str, ...]
    _backend: httpcore.SyncBackend

    def __init__(self, addresses: tuple[str, ...]) -> None:
        self._addresses = addresses
        self._backend = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ):
        """Open a TCP stream to one pinned address while preserving HTTP host identity."""
        del host
        if not self._addresses:
            raise httpcore.ConnectError("No authorized network addresses are available.")
        return self._backend.connect_tcp(
            host=self._addresses[0],
            port=port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options=None):
        """Reject Unix-socket connections because web tools authorize only TCP targets."""
        del path, timeout, socket_options
        raise httpcore.ConnectError("Unix-socket connections are not authorized.")

    def sleep(self, seconds: float) -> None:
        """Delegate retry backoff to HTTP Core's synchronous backend."""
        self._backend.sleep(seconds)


class PinnedAddressTransport(httpx.HTTPTransport):
    """Create a direct HTTP transport bound to an authorized address set."""

    def __init__(self, addresses: tuple[str, ...]) -> None:
        super().__init__(trust_env=False)
        self._pool = httpcore.ConnectionPool(network_backend=PinnedAddressBackend(addresses))


def _resolve_addresses(hostname: str) -> tuple[str, ...]:
    """Resolve a hostname once and return the unique numeric addresses for an approved request."""
    try:
        resolved = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve network host '{hostname}'.") from exc
    addresses = tuple(dict.fromkeys(item[4][0] for item in resolved))
    if not addresses:
        raise ValueError(f"Could not resolve network host '{hostname}'.")
    try:
        return tuple(str(ipaddress.ip_address(address)) for address in addresses)
    except ValueError as exc:
        raise ValueError(f"Could not resolve network host '{hostname}'.") from exc


def _cached_result(
    handle: str,
    source: str,
    content: BoundedTextContent,
) -> CachedContentResult:
    """Add cached-content identity and an opaque continuation cursor."""
    next_start_byte = content.pop("next_start_byte", None)
    content.pop("next_start_line", None)
    result = CachedContentResult(handle=handle, source=source, **content)
    if next_start_byte is not None:
        result["next_cursor"] = encode_content_cursor(handle, next_start_byte)
    return result


def _network_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan one normalized outbound HTTP request."""
    url = str(arguments["url"])
    parsed = urlsplit(url)
    if parsed.hostname is None:  # pragma: no cover - HttpUrl validates host presence.
        raise ValueError("Network request requires a hostname.")
    origin = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port is not None:
        origin += f":{parsed.port}"
    normalized = dict(arguments)
    normalized["url"] = url
    return OperationPlan(
        arguments=normalized,
        operations=(
            Operation(
                tool_id="",
                action=Action.NETWORK_REQUEST,
                target=NetworkTarget(
                    url=url,
                    origin=origin,
                    addresses=_resolve_addresses(parsed.hostname),
                ),
            ),
        ),
    )


def _cached_content_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan local cached access or a required source reload."""
    metadata = (
        None if cached_path(str(arguments["handle"])) else cached_metadata(str(arguments["handle"]))
    )
    source = metadata["source"] if metadata and metadata["reloadable"] else None
    if source is None:
        return OperationPlan(arguments=arguments)
    network = _network_plan({"url": source})
    return OperationPlan(arguments=arguments, operations=network.operations)


def _cache_url(url: str, addresses: tuple[str, ...], handle: str | None = None) -> str:
    """Stream one validated web source into the temporary content cache."""
    with httpx.stream(
        "GET",
        url,
        headers={"User-Agent": os.getenv("USER_AGENT", _DEFAULT_USER_AGENT)},
        follow_redirects=False,
        timeout=30.0,
        transport=PinnedAddressTransport(addresses),
    ) as response:
        if response.is_redirect:
            raise ValueError(
                f"Cross-request redirects are not followed; authorize the redirected URL "
                f"explicitly: {response.headers.get('location', 'unknown destination')}"
            )
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


@tool(
    actions={Action.NETWORK_REQUEST},
    operation_planner=_network_plan,
    result_presentation=ToolResultPresentationSpec(kind=ToolResultPresentation.TEXT),
)
def fetch_content(
    context: ToolContext,
    url: Annotated[
        HttpUrl,
        Field(description="HTTP(S) URL of the content to fetch."),
    ],
) -> CachedContentResult | str:
    """Fetch text into a bounded cache and return its first resumable portion."""
    url = str(url)
    try:
        operation = context.operations[0] if context.operations else None
        target = operation.target if operation is not None else None
        if not isinstance(target, NetworkTarget):
            raise TypeError("Authorized network target is missing.")
        handle = _cache_url(url, target.addresses)
        resolved = cached_path(handle)
        if resolved is None:  # pragma: no cover - store and resolve are atomic
            raise RuntimeError("Fetched content could not be cached.")
        path, source = resolved
        return _cached_result(handle, source, read_bounded_text(path))
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        return f"Error fetching content: {exc}"


@tool(
    actions={Action.NETWORK_REQUEST},
    operation_planner=_cached_content_plan,
    result_presentation=ToolResultPresentationSpec(kind=ToolResultPresentation.TEXT),
)
def read_cached_content(
    context: ToolContext,
    handle: Annotated[str, Field(description="Opaque handle returned by a bounded tool result.")],
    cursor: Annotated[
        str | None,
        Field(description="Opaque continuation cursor returned by a previous content result."),
    ] = None,
    start_line: Annotated[
        int | None,
        Field(description="Optional one-based starting line for deliberate random access.", ge=1),
    ] = None,
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
            operation = context.operations[0] if context.operations else None
            target = operation.target if operation is not None else None
            if not isinstance(target, NetworkTarget):
                raise TypeError("Authorized network target is missing.")
            _cache_url(metadata["source"], target.addresses, handle)
            resolved = cached_path(handle)
            if resolved is None:  # pragma: no cover - cache writes and lookup are atomic
                raise RuntimeError("Reloaded content could not be cached.")
        path, source = resolved
        if cursor is not None and start_line is not None:
            raise ValueError("Specify either cursor or start_line, not both.")
        start_byte = decode_content_cursor(cursor, handle) if cursor is not None else None
        if start_byte is not None and start_byte > path.stat().st_size:
            raise ValueError("Cached content cursor is beyond the end of the content.")
        return _cached_result(
            handle,
            source,
            read_bounded_text(
                path,
                start_byte=start_byte,
                start_line=None if start_byte is not None else start_line,
                max_lines=max_lines,
                max_bytes=max_bytes,
            ),
        )
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        return f"Error reading cached content: {exc}"
