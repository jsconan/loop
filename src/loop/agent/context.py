"""Render request-scoped environment facts for agent generation."""

import re
from contextlib import suppress
from html import escape
from os import environ
from platform import machine, system

from .. import constants
from ..configuration import ContextSettings
from ..utils import PathReference, VirtualPath, resolve_local_datetime, utc_now, utc_offset

_REGIONAL_LOCALE = re.compile(r"[A-Za-z]{2,8}(?:[-_][A-Za-z]{4})?[-_]([A-Za-z]{2})(?:[.@].*)?")


def render_environment(
    settings: ContextSettings,
    working_directory: PathReference,
    *,
    client_timezone: str | None,
    include_time: bool,
) -> str | None:
    """Render the allowlisted environment for one generation request.

    Args:
        settings (ContextSettings): Validated policy for the optional fields.
        working_directory (PathReference): Directory used by the active tool runtime.
        client_timezone (str | None): Trusted client timezone shared with the clock.
        include_time (bool): Include calendar and timezone facts in a verified request suffix.

    Returns:
        str | None: Escaped environment facts, including one clock sample when supported.
    """
    if settings.mode == "off":
        return None
    fields: dict[str, str | None] = {}
    if include_time:
        local, zone = resolve_local_datetime(
            utc_now(),
            client=client_timezone,
            configured=settings.timezone,
        )
        fields["current_date"] = local.date().isoformat()
        if settings.show_timezone:
            fields["timezone"] = zone
            fields["utc_offset"] = utc_offset(local)
    workspace_path = None
    if settings.show_workspace_path:
        with suppress(OSError, RuntimeError):
            workspace_path = str(working_directory.resolve())
    fields.update(
        operating_system=system(),
        architecture=machine(),
        shell=constants.SHELL_COMMAND_PATH,
        workspace=VirtualPath.WORKSPACE,
        workspace_path=workspace_path,
        country=(settings.country or _locale_country()) if settings.show_country else None,
    )
    lines = ["<environment_context>"]
    lines.extend(f"{name}: {_escape_fact(value)}" for name, value in fields.items() if value)
    lines.append("</environment_context>")
    return "\n".join(lines)


def _locale_country() -> str | None:
    """Read a two-letter territory from the process's regional locale."""
    locale = environ.get("LC_ALL") or environ.get("LC_TIME") or environ.get("LANG") or ""
    match = _REGIONAL_LOCALE.fullmatch(locale)
    return match[1].upper() if match is not None else None


def _escape_fact(value: str) -> str:
    """Keep each fact on one encodable line inside the environment block."""
    escaped = "".join(
        f"&#{ord(char)};" if char in "\r\n\t\v\f\x1c\x1d\x1e\x85\u2028\u2029" else escape(char)
        for char in value
    )
    return escaped.encode("utf-8", errors="backslashreplace").decode("utf-8")
