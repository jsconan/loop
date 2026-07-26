"""Tests for package configuration values."""

from loop.config import BASE_URL, MODEL


def test_backend_defaults_are_complete():
    """The configured backend URL and model identifier are ready for client use."""
    assert BASE_URL == "http://localhost:8000/v1"
    assert MODEL == "nvidia/Qwen3.6-35B-A3B-NVFP4"
