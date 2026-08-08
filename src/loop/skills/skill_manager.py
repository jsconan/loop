"""Discover and progressively load Agent Skills."""

from html import escape
from pathlib import Path
from typing import Any, Self

import yaml

from .skill import Skill
from .utils import default_skill_directories, read_instruction_body, read_instruction_frontmatter

MAX_CATALOG_CHARS = 8_000


class SkillManager:
    """Discover unique skill metadata and load complete instructions on activation.

    Args:
        skills (list[Skill] | None): Skill metadata in descending precedence order. The first
            definition of each name wins. Defaults to an empty list.
        diagnostics (list[str] | None): Non-fatal discovery errors suitable for inspection.
            Defaults to an empty list.
    """

    _skills: list[Skill]
    _skills_by_name: dict[str, Skill]
    _diagnostics: list[str]
    _activated: dict[Path, str]

    def __init__(
        self,
        skills: list[Skill] | None = None,
        diagnostics: list[str] | None = None,
    ) -> None:
        self._skills = []
        self._skills_by_name = {}
        self._diagnostics = list(diagnostics or [])
        for skill in skills or []:
            winner = self._skills_by_name.get(skill.name)
            if winner is not None:
                self._diagnostics.append(
                    f"Ignored '{skill.location}': skill '{skill.name}' "
                    f"is overridden by '{winner.location}'."
                )
                continue
            self._skills.append(skill)
            self._skills_by_name[skill.name] = skill
        self._activated = {}

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Return discovered skill metadata in discovery order.

        Returns:
            tuple[Skill, ...]: An immutable snapshot of the discovered skills.
        """
        return tuple(self._skills)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return non-fatal errors encountered during discovery.

        Returns:
            tuple[str, ...]: An immutable snapshot of discovery diagnostics.
        """
        return tuple(self._diagnostics)

    @property
    def activated_skills(self) -> tuple[Skill, ...]:
        """Return activated skills in discovery order.

        Returns:
            tuple[Skill, ...]: An immutable snapshot of the activated skills.
        """
        return tuple(skill for skill in self._skills if skill.location in self._activated)

    @property
    def count(self) -> int:
        """Return the number of available skills.

        Returns:
            int: The number of discovered skills.
        """
        return len(self._skills)

    @property
    def activated(self) -> int:
        """Return the number of activated skills.

        Returns:
            int: The number of skills whose instructions are loaded.
        """
        return len(self._activated)

    @classmethod
    def discover(
        cls,
        working_directory: Path | str,
        skill_directories: list[Path] | None = None,
    ) -> Self:
        """Discover available skills while reading only their YAML metadata.

        Args:
            working_directory (Path | str): Directory whose repository-scoped skills should apply.
            skill_directories (list[Path] | None): Explicit skill roots in descending precedence
                order. When omitted, repository scopes are searched from the working directory
                outward, followed by the user root from the Agent Skills convention.

        Returns:
            SkillManager: A manager containing valid skill metadata and discovery diagnostics.
        """
        working_directory = Path(working_directory).resolve()
        if skill_directories is not None:
            directories = skill_directories
        else:
            directories = default_skill_directories(working_directory)
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
        """Return available skills, activation state, and discovery diagnostics.

        Returns:
            dict[str, Any]: Model-readable skill summaries and discovery diagnostics.
        """
        return {
            "skills": [self._summary(skill) for skill in self._skills],
            "diagnostics": list(self._diagnostics),
        }

    def activate(self, name: str) -> dict[str, Any]:
        """Load and return one skill's complete instructions.

        Args:
            name (str): Exact skill name to activate.

        Returns:
            dict[str, Any]: Activated instructions or a structured missing result.
        """
        skill = self._skills_by_name.get(name)
        if skill is None:
            return {"error": "unknown_skill", "message": f"Skill '{name}' is not available."}
        if skill.location not in self._activated:
            content = skill.location.read_text(encoding="utf-8")
            self._activated[skill.location] = read_instruction_body(content, skill.location.name)
        return {
            **self._summary(skill),
            "skill_root": str(skill.location.parent),
            "instructions": self._activated[skill.location],
            "status": "activated",
        }

    def deactivate(self, name: str) -> dict[str, Any]:
        """Deactivate one skill and release its instructions.

        Args:
            name (str): Exact skill name to deactivate.

        Returns:
            dict[str, Any]: Deactivated skill metadata or a structured missing result.
        """
        skill = self._skills_by_name.get(name)
        if skill is None:
            return {"error": "unknown_skill", "message": f"Skill '{name}' is not available."}
        self._activated.pop(skill.location, None)
        return {**self._summary(skill), "status": "deactivated"}

    def deactivate_all(self) -> None:
        """Deactivate all skills and release their instructions."""
        self._activated.clear()

    def catalog(self, max_chars: int = MAX_CATALOG_CHARS) -> str | None:
        """Format a bounded metadata-only catalog for the model's initial instructions.

        Args:
            max_chars (int): Maximum number of characters in the returned catalog.

        Returns:
            str | None: The catalog, or ``None`` when no skills were discovered.
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
