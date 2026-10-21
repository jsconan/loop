"""Generate identifiers with compatibility across supported Python releases."""

import uuid
from uuid import UUID, uuid4


def new_uuid() -> UUID:
    """Return a time-ordered UUID when available, or a random UUID on older Python.

    Returns:
        UUID: A UUID7 from Python versions that provide it, otherwise a UUID4.
    """
    return getattr(uuid, "uuid7", uuid4)()
