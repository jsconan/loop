"""Discover and progressively load Agent Skills."""

from base64 import b64encode
from collections.abc import Iterable
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any, Self

import yaml

from .skill import Skill
from .utils import (
    DEFAULT_SKILL_FILENAME,
    get_skill_directories,
    read_instruction_body,
    read_instruction_frontmatter,
)

MAX_CATALOG_CHARS = 8_000
MAX_RESOURCE_BYTES = 64 * 1024
RESOURCE_DIRECTORIES = ("references", "scripts", "assets")


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
    _lifecycle_diagnostics: list[str]
    _activated: dict[Path, str]

    def __init__(
        self,
        skills: list[Skill] | None = None,
        diagnostics: list[str] | None = None,
    ) -> None:
        self._skills = []
        self._skills_by_name = {}
        self._diagnostics = list(diagnostics or [])
        self._lifecycle_diagnostics = []
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
    def lifecycle_diagnostics(self) -> tuple[str, ...]:
        """Return non-fatal activation restoration diagnostics.

        Returns:
            tuple[str, ...]: Immutable lifecycle diagnostics in occurrence order.
        """
        return tuple(self._lifecycle_diagnostics)

    @property
    def activated_skills(self) -> tuple[Skill, ...]:
        """Return activated skills in discovery order.

        Returns:
            tuple[Skill, ...]: An immutable snapshot of the activated skills.
        """
        return tuple(skill for skill in self._skills if skill.location in self._activated)

    @property
    def activated_instructions(self) -> tuple[tuple[Skill, str], ...]:
        """Return activated skill bodies in discovery order.

        Returns:
            tuple[tuple[Skill, str], ...]: Immutable skill and instruction pairs for every
                activated skill.
        """
        return tuple(
            (skill, self._activated[skill.location])
            for skill in self._skills
            if skill.location in self._activated
        )

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

    @property
    def active_identities(self) -> tuple[tuple[str, str], ...]:
        """Return persistent identities for all activated skills.

        Returns:
            tuple[tuple[str, str], ...]: Skill names paired with canonical instruction locations
                in discovery order.
        """
        return tuple((skill.name, str(skill.location)) for skill in self.activated_skills)

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
            directories = get_skill_directories(working_directory)
        skills = []
        diagnostics = []

        for directory in directories:
            if not directory.is_dir():
                continue
            for location in sorted(directory.glob(f"*/{DEFAULT_SKILL_FILENAME}")):
                try:
                    location = location.resolve()
                    metadata = read_instruction_frontmatter(
                        location, required_fields=("name", "description")
                    )
                    skills.append(Skill(metadata["name"], metadata["description"], location))
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                    diagnostics.append(f"Skipped '{location}': {exc}")

        return cls(skills, diagnostics)

    @staticmethod
    def discovery_signature(working_directory: Path | str) -> tuple:
        """Return a content-aware fingerprint of discoverable skill definitions.

        Args:
            working_directory (Path | str): Directory whose skill scopes should be fingerprinted.

        Returns:
            tuple: Stable fingerprint of skill roots and instruction files.
        """
        paths = []
        for directory in get_skill_directories(Path(working_directory).resolve()):
            paths.append(directory)
            if directory.is_dir():
                paths.extend(sorted(directory.glob(f"*/{DEFAULT_SKILL_FILENAME}")))
        fingerprints = []
        for path in paths:
            try:
                stat = path.stat()
                digest = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                fingerprints.append(
                    (str(path.resolve()), stat.st_ino, stat.st_mtime_ns, stat.st_size, digest)
                )
            except FileNotFoundError:
                fingerprints.append((str(path.resolve()), None))
        return tuple(fingerprints)

    @classmethod
    def rediscover(
        cls,
        working_directory: Path | str,
        active_identities: Iterable[tuple[str, str]],
    ) -> Self:
        """Rediscover skills and restore only unchanged active definitions.

        Args:
            working_directory (Path | str): Directory whose skill scopes should apply.
            active_identities (Iterable[tuple[str, str]]): Previously active names paired with
                canonical instruction locations.

        Returns:
            SkillManager: Refreshed manager with matching definitions reactivated and lifecycle
                diagnostics retained.
        """
        manager = cls.discover(working_directory)
        manager.restore(active_identities, refresh=True)
        return manager

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
        """Load one skill's complete instructions into managed state.

        Args:
            name (str): Exact skill name to activate.

        Returns:
            dict[str, Any]: Activation metadata or a structured missing result.
        """
        skill = self._skills_by_name.get(name)
        if skill is None:
            return {"error": "unknown_skill", "message": f"Skill '{name}' is not available."}
        instructions_updated = skill.location not in self._activated
        if instructions_updated:
            content = skill.location.read_text(encoding="utf-8")
            self._activated[skill.location] = read_instruction_body(content, skill.location.name)
        return {
            **self._summary(skill),
            "skill_root": str(skill.location.parent),
            "status": "activated",
            "instructions_updated": instructions_updated,
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
        instructions_updated = skill.location in self._activated
        self._activated.pop(skill.location, None)
        return {
            **self._summary(skill),
            "status": "deactivated",
            "instructions_updated": instructions_updated,
        }

    def deactivate_all(self) -> int:
        """Deactivate all skills and release their instructions.

        Returns:
            int: Number of skills that were active.
        """
        count = len(self._activated)
        self._activated.clear()
        return count

    def is_active(self, name: str) -> bool:
        """Return whether an available skill is active.

        Args:
            name (str): Exact available skill name.

        Returns:
            bool: Whether the named skill has loaded instructions.
        """
        skill = self._skills_by_name.get(name)
        return skill is not None and skill.location in self._activated

    def restore(
        self,
        identities: Iterable[tuple[str, str]],
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Activate definitions that exactly match persisted identities.

        Args:
            identities (Iterable[tuple[str, str]]): Skill names paired with canonical persisted
                instruction locations.
            refresh (bool): Whether diagnostics should describe an in-process refresh rather than
                session restoration.

        Returns:
            list[dict[str, Any]]: Activation results for definitions that still match.
        """
        results = []
        for name, location in identities:
            skill = self._skills_by_name.get(name)
            if skill is None or str(skill.location) != location:
                if refresh:
                    self._lifecycle_diagnostics.append(
                        f"Deactivated '{name}' during refresh: its definition was removed or "
                        "shadowed."
                    )
                else:
                    self._lifecycle_diagnostics.append(
                        f"Could not restore '{name}': its definition was removed or shadowed."
                    )
                continue
            try:
                results.append(self.activate(name))
            except (OSError, UnicodeError, ValueError) as exc:
                if refresh:
                    self._lifecycle_diagnostics.append(
                        f"Deactivated '{name}' during refresh: {exc}"
                    )
                else:
                    raise
        return results

    def list_resources(self, name: str) -> dict[str, Any]:
        """List bounded on-demand resources belonging to an active skill.

        Args:
            name (str): Exact active skill name.

        Returns:
            dict[str, Any]: Resource paths and sizes, or a structured error.
        """
        skill = self._skills_by_name.get(name)
        if skill is None:
            return {"error": "unknown_skill", "message": f"Skill '{name}' is not available."}
        if skill.location not in self._activated:
            return {
                "error": "skill_not_active",
                "message": f"Skill '{name}' must be activated before loading its resources.",
            }
        root = skill.location.parent.resolve()
        resources = []
        for directory_name in RESOURCE_DIRECTORIES:
            directory = root / directory_name
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                resolved = path.resolve()
                if path.is_file() and resolved.is_relative_to(root):
                    resources.append(
                        {"path": str(resolved.relative_to(root)), "size_bytes": path.stat().st_size}
                    )
        return {"name": name, "skill_root": str(root), "resources": resources}

    def read_resource(self, name: str, resource_path: str) -> dict[str, Any]:
        """Read one active skill resource without adding it to persistent instructions.

        Args:
            name (str): Exact active skill name.
            resource_path (str): Relative path beneath references, scripts, or assets.

        Returns:
            dict[str, Any]: Text or base64 resource content, or a structured error.
        """
        listed = self.list_resources(name)
        if "error" in listed:
            return listed
        root = Path(listed["skill_root"])
        candidate = (root / resource_path).resolve()
        allowed = any(
            candidate.is_relative_to(root / directory) for directory in RESOURCE_DIRECTORIES
        )
        if not allowed or not candidate.is_file():
            return {
                "error": "invalid_skill_resource",
                "message": "Resource must be a file beneath references, scripts, or assets.",
            }
        content = candidate.read_bytes()
        if len(content) > MAX_RESOURCE_BYTES:
            return {
                "error": "skill_resource_too_large",
                "message": f"Resource exceeds the {MAX_RESOURCE_BYTES}-byte loading limit.",
                "size_bytes": len(content),
            }
        result = {
            "name": name,
            "path": str(candidate.relative_to(root)),
            "size_bytes": len(content),
        }
        try:
            result.update({"encoding": "utf-8", "content": content.decode("utf-8")})
        except UnicodeDecodeError:
            result.update({"encoding": "base64", "content": b64encode(content).decode("ascii")})
        return result

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
            "Use manage_skills to activate before use and deactivate when no longer needed.\n"
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
