"""Tests for the built-in web access tools."""

import json
from unittest.mock import MagicMock, call

from loop import ConsoleInteraction, tool_registry


def fetch_content(url):
    """Dispatch the context-aware content-fetching tool."""
    return tool_registry.call(
        "fetch_content",
        json.dumps({"url": url}),
        interaction=ConsoleInteraction(),
    )


def test_fetch_content_requires_confirmation_before_fetching(monkeypatch):
    """Fetching only starts after an affirmative confirmation."""
    monkeypatch.delenv("USER_AGENT", raising=False)
    confirm = MagicMock(side_effect=[False, True])
    response = MagicMock()
    response.text = "<html>fetched content</html>"
    get = MagicMock(return_value=response)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.get", get)

    assert fetch_content("https://example.com/file.txt") == "Fetch operation cancelled by user."
    get.assert_not_called()

    assert fetch_content("https://example.com/file.txt") == "<html>fetched content</html>"
    get.assert_called_once_with(
        "https://example.com/file.txt",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        },
        follow_redirects=True,
        timeout=30.0,
    )
    response.raise_for_status.assert_called_once_with()
    assert confirm.call_args_list == [
        call(
            "Agent wants to fetch content from 'https://example.com/file.txt'. Proceed?",
            default=False,
        ),
        call(
            "Agent wants to fetch content from 'https://example.com/file.txt'. Proceed?",
            default=False,
        ),
    ]


def test_fetch_content_uses_configured_user_agent(monkeypatch):
    """The user agent can be configured without changing application code."""
    response = MagicMock(text="fetched content")
    get = MagicMock(return_value=response)
    monkeypatch.setenv("USER_AGENT", "LoopBot/1.0")
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.web.httpx.get", get)

    assert fetch_content("https://example.com") == "fetched content"
    assert get.call_args.kwargs["headers"] == {"User-Agent": "LoopBot/1.0"}


def test_fetch_content_reports_failures(monkeypatch):
    """Fetch failures become readable tool results."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr(
        "loop.tools.web.httpx.get",
        MagicMock(side_effect=OSError("network unavailable")),
    )

    assert fetch_content("https://example.com/file.txt") == (
        "Error fetching content: network unavailable"
    )


def test_fetch_content_rejects_non_http_urls(monkeypatch):
    """Non-HTTP URL schemes are rejected before confirmation or network access."""
    confirm = MagicMock(return_value=True)
    get = MagicMock()
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.get", get)

    result = fetch_content("file:///etc/passwd")

    assert "Invalid arguments for tool 'fetch_content'" in result
    confirm.assert_not_called()
    get.assert_not_called()


def test_fetch_content_rejects_malformed_urls(monkeypatch):
    """Malformed HTTP URLs are rejected before confirmation or network access."""
    confirm = MagicMock(return_value=True)
    get = MagicMock()
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.web.httpx.get", get)

    result = fetch_content("https://")

    assert "Invalid arguments for tool 'fetch_content'" in result
    confirm.assert_not_called()
    get.assert_not_called()
