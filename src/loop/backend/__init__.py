"""Expose backend contracts and implementations."""

__all__ = ["Backend", "OpenAIBackend"]

from .backend import Backend
from .openai import OpenAIBackend
