"""Tests for the built-in web access tools."""

import json
from unittest.mock import MagicMock

import httpcore
import pytest

from loop import BUILTIN_TOOLS, ConsoleInteraction, ToolContext, ToolRegistry
from loop.tools import web as web_module
from loop.utils import cached_path as resolve_cached_path
from loop.utils import encode_content_cursor

tool_registry = ToolRegistry(BUILTIN_TOOLS)


@pytest.fixture(autouse=True)
def fresh_tool_registry(monkeypatch):
    """Provide isolated tools and deterministic public DNS for each web-tool case."""
    global tool_registry  # pylint: disable=global-statement
    tool_registry = ToolRegistry(BUILTIN_TOOLS)
    monkeypatch.setattr(
        "loop.tools.web.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )


def stream_response(content, *, content_type="text/plain"):
    """Return a context-managed streaming HTTP response double."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.is_redirect = False
    response.headers = {"content-type": content_type}
    response.iter_bytes.return_value = iter([content])
    return response


def fetch_content(url):
    """Dispatch the context-aware content-fetching tool."""
    output = tool_registry.call(
        "fetch_content",
        json.dumps({"url": url}),
        interaction=ConsoleInteraction(),
    )
    payload = json.loads(output)
    return json.dumps(payload["result"]) if payload["ok"] else output


def read_cached_content(handle, **ranges):
    """Dispatch a cached-content continuation read."""
    output = tool_registry.call(
        "read_cached_content",
        json.dumps({"handle": handle, **ranges}),
        interaction=ConsoleInteraction(),
    )
    payload = json.loads(output)
    return json.dumps(payload["result"]) if payload["ok"] else output


def problem(output: str):
    """Return the problem from a failed tool result envelope."""
    return json.loads(output)["problem"]


def test_fetch_content_requires_confirmation_before_fetching(monkeypatch):
    """Fetching only starts after an affirmative confirmation."""
    monkeypatch.delenv("USER_AGENT", raising=False)
    confirm = MagicMock(side_effect=[False, True])
    response = stream_response(b"<html>fetched content</html>", content_type="text/html")
    stream = MagicMock(return_value=response)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    assert problem(fetch_content("https://example.com/file.txt"))["code"] == "tool.denied"
    stream.assert_not_called()

    result = json.loads(fetch_content("https://example.com/file.txt"))
    assert result["content"] == "<html>fetched content</html>"
    assert result["truncated"] is False
    assert stream.call_args.args == ("GET", "https://example.com/file.txt")
    assert stream.call_args.kwargs["headers"] == {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:153.0) Gecko/20100101 Firefox/153.0"
        ),
    }
    assert stream.call_args.kwargs["follow_redirects"] is False
    assert stream.call_args.kwargs["timeout"] == 30.0
    assert type(stream.call_args.kwargs["transport"]).__name__ == "PinnedAddressTransport"
    response.raise_for_status.assert_called_once_with()
    assert confirm.call_count == 2
    assert all("network.request" in item.args[0] for item in confirm.call_args_list)
    assert all("https://example.com/file.txt" in item.args[0] for item in confirm.call_args_list)


def test_fetch_content_uses_configured_user_agent(monkeypatch):
    """The user agent can be configured without changing application code."""
    response = stream_response(b"fetched content")
    stream = MagicMock(return_value=response)
    monkeypatch.setenv("USER_AGENT", "LoopBot/1.0")
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    assert json.loads(fetch_content("https://example.com"))["content"] == "fetched content"
    assert stream.call_args.kwargs["headers"] == {"User-Agent": "LoopBot/1.0"}


def test_fetch_content_reports_failures(monkeypatch):
    """Fetch failures become readable tool results."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr(
        "loop.tools.web.httpx.stream",
        MagicMock(side_effect=OSError("network unavailable")),
    )

    assert problem(fetch_content("https://example.com/file.txt"))["detail"] == (
        "network unavailable"
    )


def test_fetch_content_rejects_redirects_for_separate_authorization(monkeypatch):
    """A redirect cannot silently expand approval to a different network target."""
    response = stream_response(b"")
    response.is_redirect = True
    response.headers["location"] = "https://other.example/target"
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))

    result = fetch_content("https://example.com/redirect")

    assert "redirected URL explicitly" in result
    assert "https://other.example/target" in result


def test_fetch_content_plans_origins_with_explicit_ports(monkeypatch):
    """Network plans retain a non-default port in the approval target."""
    response = stream_response(b"content")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))

    assert json.loads(fetch_content("https://example.com:8443/file"))["content"] == "content"
    assert "https://example.com:8443/file" in confirm.call_args.args[0]


def test_fetch_content_denies_private_addresses_resolved_during_planning(monkeypatch):
    """A hostname resolving to a private address is denied before the request starts."""
    stream = MagicMock()
    monkeypatch.setattr(
        "loop.tools.web.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("127.0.0.1", 0))],
    )
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    result = fetch_content("https://service.test/private")

    assert problem(result)["code"] == "tool.denied"
    stream.assert_not_called()


@pytest.mark.parametrize(
    "resolver",
    [
        lambda *_args, **_kwargs: [],
        lambda *_args, **_kwargs: [(None, None, None, None, ("not-an-address", 0))],
    ],
)
def test_fetch_content_fails_closed_when_resolution_returns_no_usable_addresses(
    monkeypatch, resolver
):
    """Resolution errors stop planning before approval or a request can begin."""
    confirm = MagicMock(return_value=True)
    stream = MagicMock()
    monkeypatch.setattr("loop.tools.web.socket.getaddrinfo", resolver)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    result = fetch_content("https://service.test/content")

    assert problem(result)["code"] == "tool.planning_failed"
    confirm.assert_not_called()
    stream.assert_not_called()


def test_fetch_content_fails_closed_when_resolution_raises(monkeypatch):
    """A DNS failure stops planning before approval or a request can begin."""

    def unresolved(*_args, **_kwargs):
        """Simulate a resolver unable to locate the requested hostname."""
        raise __import__("socket").gaierror

    monkeypatch.setattr("loop.tools.web.socket.getaddrinfo", unresolved)

    assert problem(fetch_content("https://service.test/content"))["code"] == (
        "tool.planning_failed"
    )


def test_pinned_address_backend_connects_only_to_its_authorised_address():
    """The connection backend substitutes the planned address while retaining the request port."""
    backend = web_module.PinnedAddressBackend(("93.184.216.34",))
    backend._backend = MagicMock()  # pylint: disable=protected-access
    stream = object()
    backend._backend.connect_tcp.return_value = stream  # pylint: disable=protected-access

    assert backend.connect_tcp("example.com", 443, timeout=2) is stream
    backend._backend.connect_tcp.assert_called_once_with(  # pylint: disable=protected-access
        host="93.184.216.34",
        port=443,
        timeout=2,
        local_address=None,
        socket_options=None,
    )


def test_pinned_address_backend_rejects_missing_or_non_tcp_connections():
    """The connection backend refuses unplanned addresses and Unix sockets."""
    backend = web_module.PinnedAddressBackend(())

    with pytest.raises(httpcore.ConnectError, match="No authorized"):
        backend.connect_tcp("example.com", 443)
    with pytest.raises(httpcore.ConnectError, match="Unix-socket"):
        backend.connect_unix_socket("/tmp/socket")


def test_pinned_address_backend_delegates_retry_sleep():
    """The connection backend preserves HTTP Core retry timing behavior."""
    backend = web_module.PinnedAddressBackend(("93.184.216.34",))
    backend._backend = MagicMock()  # pylint: disable=protected-access

    backend.sleep(0.1)

    backend._backend.sleep.assert_called_once_with(0.1)  # pylint: disable=protected-access


def test_web_tools_fail_closed_without_an_authorized_network_operation(monkeypatch):
    """Direct execution cannot fetch or reload content without a planned network target."""
    context = ToolContext(ConsoleInteraction(), "fetch_content")
    monkeypatch.setattr("loop.tools.web.cached_path", lambda _handle: None)
    monkeypatch.setattr(
        "loop.tools.web.cached_metadata",
        lambda _handle: {"source": "https://example.com", "reloadable": True},
    )

    assert (
        "Authorized network target is missing"
        in web_module.fetch_content(context, "https://example.com").detail
    )
    assert (
        "Authorized network target is missing"
        in web_module.read_cached_content(context, "handle").detail
    )


def test_fetch_content_rejects_binary_content(monkeypatch):
    """Binary response content is not returned to the agent."""
    response = stream_response(b"binary\0content")
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))

    assert problem(fetch_content("https://example.com/file.bin"))["detail"] == (
        "Content at 'https://example.com/file.bin' appears to be binary."
    )
    response.raise_for_status.assert_called_once_with()


def test_fetch_content_is_bounded_and_cached_for_continuation(monkeypatch):
    """Large web text returns one bounded part and an opaque resumable handle."""
    content = ("line\n" * 5_000).encode()
    response = stream_response(content)
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))

    first = json.loads(fetch_content("https://example.com/large.txt"))
    second = json.loads(
        read_cached_content(
            first["handle"],
            cursor=first["next_cursor"],
        )
    )

    assert first["included_bytes"] <= 16 * 1024
    assert first["truncated"] is True
    assert second["start_byte"] == first["end_byte"]
    assert second["source"] == "https://example.com/large.txt"
    assert "next_start_byte" not in first
    assert (
        "start_byte"
        not in next(
            definition
            for definition in tool_registry.definitions()
            if definition.name == "read_cached_content"
        ).parameters["properties"]
    )


def test_fetch_content_rejects_unsupported_or_excessive_responses(monkeypatch):
    """Streaming rejects non-text media and bodies above the download cache ceiling."""
    binary = stream_response(b"image", content_type="image/png")
    excessive = stream_response(b"x" * (10 * 1024 * 1024 + 1))
    stream = MagicMock(side_effect=[binary, excessive])
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    assert "not a supported text response" in fetch_content("https://example.com/image.png")
    assert "download limit" in fetch_content("https://example.com/large.txt")


def test_fetch_content_rejects_invalid_utf8(monkeypatch):
    """Streaming rejects text responses that are not valid UTF-8."""
    response = stream_response(b"\xff")
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))

    assert "utf-8" in fetch_content("https://example.com/invalid.txt")


def test_read_cached_content_reports_unknown_handles(monkeypatch):
    """Cached reads report expired or unknown artifact handles."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    assert "Unknown or expired" in read_cached_content("missing")


def test_read_cached_content_reloads_an_expired_web_artifact_with_authorization(monkeypatch):
    """An expired web handle re-fetches its persisted source without another model argument."""
    initial = stream_response(b"initial")
    reloaded = stream_response(b"reloaded")
    stream = MagicMock(side_effect=[initial, reloaded])
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)
    fetched = json.loads(fetch_content("https://example.com/source.txt"))
    lookups = 0

    def expired_then_repopulated(handle):
        """Hide the old artifact until the reload writes the same handle."""
        nonlocal lookups
        lookups += 1
        return None if lookups <= 2 else resolve_cached_path(handle)

    monkeypatch.setattr("loop.tools.web.cached_path", expired_then_repopulated)

    result = json.loads(read_cached_content(fetched["handle"]))

    assert result["content"] == "reloaded"
    assert result["source"] == "https://example.com/source.txt"
    assert result["handle"] == fetched["handle"]
    assert "network.request" in confirm.call_args.args[0]
    assert "https://example.com/source.txt" in confirm.call_args.args[0]
    assert stream.call_count == 2


def test_read_cached_content_reports_invalid_cursors_and_selectors(monkeypatch):
    """Cached reads reject malformed cursors and conflicting selectors."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    response = stream_response(b"cached")
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))
    fetched = json.loads(fetch_content("https://example.com/content.txt"))

    malformed = read_cached_content(fetched["handle"], cursor="invalid")
    conflicting = read_cached_content(
        fetched["handle"], cursor=fetched.get("next_cursor", "invalid"), start_line=1
    )
    beyond_end = read_cached_content(
        fetched["handle"], cursor=encode_content_cursor(fetched["handle"], 7)
    )

    assert "Invalid cached content cursor" in malformed
    assert "either cursor or start_line" in conflicting
    assert "beyond the end" in beyond_end


def test_fetch_content_rejects_non_http_urls(monkeypatch):
    """Non-HTTP URL schemes are rejected before confirmation or network access."""
    confirm = MagicMock(return_value=True)
    stream = MagicMock()
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    result = fetch_content("file:///etc/passwd")

    assert "Invalid arguments for tool 'fetch_content'" in result
    confirm.assert_not_called()
    stream.assert_not_called()


def test_fetch_content_rejects_malformed_urls(monkeypatch):
    """Malformed HTTP URLs are rejected before confirmation or network access."""
    confirm = MagicMock(return_value=True)
    stream = MagicMock()
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    result = fetch_content("https://")

    assert "Invalid arguments for tool 'fetch_content'" in result
    confirm.assert_not_called()
    stream.assert_not_called()
