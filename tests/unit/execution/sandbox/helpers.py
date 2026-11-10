"""Share logical policy identities for owned sandbox fixture paths."""

from pathlib import Path

from loop.permissions.protection import is_protected_external_read


def logical_private_name(path, case_root):
    """Apply real private-name rules below an owned synthetic case root."""
    if path.is_relative_to(case_root):
        path = Path("/test-fixture") / path.relative_to(case_root)
    return is_protected_external_read(path)
