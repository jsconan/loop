"""Expose backend contracts and implementations."""

__all__ = ["Backend", "OpenAIBackend"]

from .base import Backend
from .openai import OpenAIBackend
