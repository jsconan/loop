"""Discover and progressively load Agent Skills."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import yaml

from .utils.instructions import (
    instruction_directories,
    read_instruction_body,
    read_instruction_frontmatter,
)

MAX_CATALOG_CHARS = 8_000


@dataclass(frozen=True)
class Skill:
    """Describe an Agent Skill without eagerly loading its instructions.

    Attributes:
        name: Public name declared by the skill.
        description: Summary used by the model to decide when to activate the skill.
        location: Absolute path to the skill's ``SKILL.md`` file.
    """

    name: str
    description: str
    location: Path


class SkillManager:
    """Discover skill metadata and load complete instructions on activation.

    Args:
        skills: Discovered skill metadata.
        diagnostics: Non-fatal discovery errors suitable for inspection.
    """

    def __init__(
        self,
        skills: list[Skill] | None = None,
        diagnostics: list[str] | None = None,
    ) -> None:
        self._skills = skills or []
        self._diagnostics = diagnostics or []
        self._activated: dict[Path, str] = {}

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Return discovered skill metadata in discovery order."""
        return tuple(self._skills)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return non-fatal errors encountered during discovery."""
        return tuple(self._diagnostics)

    @classmethod
    def discover(
        cls,
        working_directory: Path | str,
        skill_directories: list[Path] | None = None,
    ) -> "SkillManager":
        """Discover available skills while reading only their YAML metadata.

        Args:
            working_directory: Directory whose repository-scoped skills should apply.
            skill_directories: Explicit skill roots. When omitted, repository and user roots
                from the Agent Skills convention are searched.

        Returns:
            A manager containing valid skill metadata and discovery diagnostics.
        """
        working_directory = Path(working_directory).resolve()
        directories = (
            skill_directories
            if skill_directories is not None
            else instruction_directories(working_directory)
        )
        skills = []
        diagnostics = []

        for directory in directories:
            if not directory.is_dir():
                continue
            for location in sorted(directory.glob("*/SKILL.md")):
                try:
                    location = location.resolve()
                    metadata = read_instruction_frontmatter(
                        location, required_fields=("name", "description")
                    )
                    skills.append(Skill(metadata["name"], metadata["description"], location))
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                    diagnostics.append(f"Skipped '{location}': {exc}")

        return cls(skills, diagnostics)

    def list(self) -> dict[str, Any]:
        """Return available skills, activation state, and discovery diagnostics."""
        return {
            "skills": [self._summary(skill) for skill in self._skills],
            "diagnostics": list(self._diagnostics),
        }

    def activate(self, name: str) -> dict[str, Any]:
        """Load and return one skill's complete instructions.

        Args:
            name: Exact skill name to activate.

        Returns:
            Activated instructions, or a structured missing or ambiguous result.
        """
        matches = [skill for skill in self._skills if skill.name == name]
        if not matches:
            return {"error": "unknown_skill", "message": f"Skill '{name}' is not available."}
        if len(matches) > 1:
            return {
                "error": "ambiguous_skill",
                "message": f"Skill '{name}' is declared in more than one location.",
                "locations": [str(skill.location) for skill in matches],
            }

        skill = matches[0]
        if skill.location not in self._activated:
            content = skill.location.read_text(encoding="utf-8")
            self._activated[skill.location] = read_instruction_body(
                content, skill.location.name
            )
        return {
            **self._summary(skill),
            "skill_root": str(skill.location.parent),
            "instructions": self._activated[skill.location],
            "status": "activated",
        }

    def catalog(self, max_chars: int = MAX_CATALOG_CHARS) -> str | None:
        """Format a bounded metadata-only catalog for the model's initial instructions.

        Args:
            max_chars: Maximum number of characters in the returned catalog.

        Returns:
            The catalog, or ``None`` when no skills were discovered.
        """
        if not self._skills:
            return None
        header = (
            "<available_skills>\n"
            'Use the manage_skills tool with action="activate" before following a skill.\n'
        )
        footer = "</available_skills>"
        entries = []
        omitted = 0
        used = len(header) + len(footer)
        for skill in self._skills:
            entry = (
                "<skill>\n"
                f"<name>{escape(skill.name)}</name>\n"
                f"<description>{escape(skill.description)}</description>\n"
                f"<location>{escape(str(skill.location))}</location>\n"
                "</skill>\n"
            )
            if used + len(entry) > max_chars:
                omitted += 1
                continue
            entries.append(entry)
            used += len(entry)
        warning = (
            f"<warning>{omitted} skill(s) omitted by catalog limit.</warning>\n" if omitted else ""
        )
        return f"{header}{''.join(entries)}{warning}{footer}"[:max_chars]

    def _summary(self, skill: Skill) -> dict[str, Any]:
        """Return model-readable metadata for one skill."""
        return {
            "name": skill.name,
            "description": skill.description,
            "location": str(skill.location),
            "activated": skill.location in self._activated,
        }
