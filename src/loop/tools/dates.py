"""Provide tools for working with dates and times."""

from typing import Annotated, TypedDict

from pydantic import AfterValidator, Field
from pydantic_core import PydanticCustomError

from ..tooling import ToolContext, tool
from ..utils import resolve_local_datetime, utc_now, utc_offset, validate_timezone_selector

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class CurrentDatetime(TypedDict):
    """Describe one captured instant in local time and UTC."""

    datetime: str
    utc: str
    timezone: str | None
    utc_offset: str
    weekday: str


def _validate_timezone_selector(selector: str) -> str:
    """Report selector guidance through the tool's trusted validation error path."""
    try:
        return validate_timezone_selector(selector)
    except ValueError as error:
        raise PydanticCustomError("loop_timezone_selector", str(error)) from error


@tool
def get_current_datetime(
    context: ToolContext | None = None,
    timezone: Annotated[
        Annotated[str, AfterValidator(_validate_timezone_selector)] | None,
        Field(
            description=(
                "IANA timezone or signed ±HH:MM offset. Use JSON null (unquoted) for the default. "
                "For a place, choose its IANA timezone. Ask for the country or region if the "
                "place is ambiguous, or for a city or region if the country spans multiple zones."
            ),
        ),
    ] = None,
) -> CurrentDatetime:
    """Look up the current date and exact time in the requested timezone and in UTC."""
    instant = utc_now()
    settings = context.settings if context is not None else None
    local, name = resolve_local_datetime(
        instant,
        explicit=timezone,
        client=settings.client_timezone if settings is not None else None,
        configured=settings.timezone if settings is not None else None,
    )
    return CurrentDatetime(
        datetime=local.isoformat(timespec="seconds"),
        utc=instant.isoformat(timespec="seconds").replace("+00:00", "Z"),
        timezone=name,
        utc_offset=utc_offset(local),
        weekday=_WEEKDAYS[local.weekday()],
    )
