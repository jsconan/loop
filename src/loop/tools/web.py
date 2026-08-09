"""Provide tools for accessing content on the web."""

import os
from typing import Annotated

import httpx
from pydantic import Field, HttpUrl

from ..permissions import Capability, PermissionRequest
from ..tooling import tool_registry

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


@tool_registry.tool(
    capabilities={Capability.NETWORK_READ},
    permission_resolver=_network_permission,
)
def fetch_content(
    url: Annotated[
        HttpUrl,
        Field(description="HTTP(S) URL of the content to fetch."),
    ],
) -> str:
    """Fetch and return text content from a URL."""
    url = str(url)
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
