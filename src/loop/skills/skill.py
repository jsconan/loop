"""Define an Agent Skill."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    """Describe an Agent Skill without eagerly loading its instructions.

    Args:
        name (str): Public name declared by the skill.
        description (str): Summary used by the model to decide when to activate the skill.
        location (Path): Absolute path to the skill's ``SKILL.md`` file.
    """

    name: str
    description: str
    location: Path
