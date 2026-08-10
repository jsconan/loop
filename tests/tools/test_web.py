"""Tests for the built-in web access tools."""

import json
from unittest.mock import MagicMock, call

from loop import ConsoleInteraction, tool_registry
from loop.utils import cached_path as resolve_cached_path


def stream_response(content, *, content_type="text/plain"):
    """Return a context-managed streaming HTTP response double."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.headers = {"content-type": content_type}
    response.iter_bytes.return_value = iter([content])
    return response


def fetch_content(url):
    """Dispatch the context-aware content-fetching tool."""
    return tool_registry.call(
        "fetch_content",
        json.dumps({"url": url}),
        interaction=ConsoleInteraction(),
    )


def read_cached_content(handle, **ranges):
    """Dispatch a cached-content continuation read."""
    return tool_registry.call(
        "read_cached_content",
        json.dumps({"handle": handle, **ranges}),
        interaction=ConsoleInteraction(),
    )


def test_fetch_content_requires_confirmation_before_fetching(monkeypatch):
    """Fetching only starts after an affirmative confirmation."""
    monkeypatch.delenv("USER_AGENT", raising=False)
    confirm = MagicMock(side_effect=[False, True])
    response = stream_response(b"<html>fetched content</html>", content_type="text/html")
    stream = MagicMock(return_value=response)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.stream", stream)

    assert '"error": "tool_call_denied"' in fetch_content("https://example.com/file.txt")
    stream.assert_not_called()

    result = json.loads(fetch_content("https://example.com/file.txt"))
    assert result["content"] == "<html>fetched content</html>"
    assert result["truncated"] is False
    stream.assert_called_once_with(
        "GET",
        "https://example.com/file.txt",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; "
                "rv:153.0) Gecko/20100101 Firefox/153.0"
            ),
        },
        follow_redirects=True,
        timeout=30.0,
    )
    response.raise_for_status.assert_called_once_with()
    assert confirm.call_args_list == [
        call(
            "Agent wants to use 'fetch_content' for network.read on "
            "'https://example.com/file.txt'. Proceed?",
            default=False,
        ),
        call(
            "Agent wants to use 'fetch_content' for network.read on "
            "'https://example.com/file.txt'. Proceed?",
            default=False,
        ),
    ]


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

    assert fetch_content("https://example.com/file.txt") == (
        "Error fetching content: network unavailable"
    )


def test_fetch_content_rejects_binary_content(monkeypatch):
    """Binary response content is not returned to the agent."""
    response = stream_response(b"binary\0content")
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))

    assert fetch_content("https://example.com/file.bin") == (
        "Error fetching content: Content at 'https://example.com/file.bin' appears to be binary."
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
            start_byte=first["next_start_byte"],
            start_line=None,
        )
    )

    assert first["included_bytes"] <= 16 * 1024
    assert first["truncated"] is True
    assert second["start_byte"] == first["next_start_byte"]
    assert second["source"] == "https://example.com/large.txt"


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
    assert "network.read" in confirm.call_args.args[0]
    assert "https://example.com/source.txt" in confirm.call_args.args[0]
    assert stream.call_count == 2


def test_read_cached_content_reports_invalid_ranges(monkeypatch):
    """Cached reads convert invalid range combinations into readable tool errors."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    response = stream_response(b"cached")
    monkeypatch.setattr("loop.tools.web.httpx.stream", MagicMock(return_value=response))
    fetched = json.loads(fetch_content("https://example.com/content.txt"))

    result = read_cached_content(fetched["handle"], start_byte=1)

    assert "either start_byte or start_line" in result


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
