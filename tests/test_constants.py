"""Tests for shared application constants."""

from loop import OMIT, Omit
from loop.constants import OMIT as module_omit


def test_omit_is_a_shared_omission_marker():
    """The public marker has one shared identity and omission type."""
    assert module_omit is OMIT
    assert isinstance(OMIT, Omit)
