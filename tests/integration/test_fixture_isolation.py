"""Verify explicitly owned native fixture paths preserve the filesystem output guard."""

from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


def test_native_directory_permits_owned_output_and_refuses_unowned_sibling(native_directory):
    """Native fixture registration permits only its fresh child and keeps siblings protected."""
    directory = native_directory(Path("/tmp"))
    owned = directory / "owned"
    owned.write_text("owned-output")
    assert owned.read_text() == "owned-output"
    unowned = directory.parent / ("loop-unowned-test-" + uuid4().hex)
    with pytest.raises(AssertionError, match="Test output escapes disposable scratch"):
        unowned.write_text("must-not-be-created")
    assert not unowned.exists()
