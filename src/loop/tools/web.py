"""Provide tools for accessing content on the web."""

import os
from typing import Annotated

import httpx
from pydantic import Field, HttpUrl

from ..interaction import ToolContext
from ..tooling import tool_registry

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:153.0) Gecko/20100101 Firefox/153.0"
)


@tool_registry.tool
def fetch_content(
    context: ToolContext,
    url: Annotated[
        HttpUrl,
        Field(description="HTTP(S) URL of the content to fetch."),
    ],
) -> str:
    """Fetch and return text content from a URL."""
    url = str(url)
    if not context.confirm(f"Agent wants to fetch content from '{url}'. Proceed?"):
        return "Fetch operation cancelled by user."

    try:
        response = httpx.get(
            url,
            headers={"User-Agent": os.getenv("USER_AGENT", _DEFAULT_USER_AGENT)},
            follow_redirects=True,
            timeout=30.0,
        )
        response.raise_for_status()
        if b"\0" in response.content:
            raise ValueError(f"Content at '{url}' appears to be binary.")
        return response.text
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error fetching content: {exc}"
