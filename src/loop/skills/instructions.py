"""Compose the developer instructions used for backend requests."""

from html import escape
from pathlib import Path
from typing import Any, Self

from .skill_manager import SkillManager
from .utils import build_instructions, load_agents_instructions

MAX_INSTRUCTIONS_BYTES = 64 * 1024


class InstructionsManager:
    """Maintain project, catalog, and activated-skill instructions.

    Args:
        project_instructions (str | None): Project instructions loaded from applicable AGENTS.md
            files, or ``None`` when none apply.
        skill_manager (SkillManager | None): Skill manager supplying the catalog and lazily loaded
            skill bodies. Defaults to an empty manager.
        max_bytes (int): Maximum encoded size of the complete instruction document.

    Raises:
        ValueError: If ``max_bytes`` is not a positive integer or the initial instructions exceed
            the configured limit.
    """

    _project_instructions: str | None
    _skill_manager: SkillManager
    _max_bytes: int

    def __init__(
        self,
        project_instructions: str | None = None,
        skill_manager: SkillManager | None = None,
        *,
        max_bytes: int = MAX_INSTRUCTIONS_BYTES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("Instruction limit must be a positive integer.")
        self._project_instructions = project_instructions
        self._skill_manager = skill_manager or SkillManager()
        self._max_bytes = max_bytes
        if self._encoded_size(self.instructions) > self._max_bytes:
            raise ValueError("Initial instructions exceed the configured instruction limit.")

    @classmethod
    def discover(
        cls,
        working_directory: Path | str,
        *,
        skill_manager: SkillManager | None = None,
        max_bytes: int = MAX_INSTRUCTIONS_BYTES,
    ) -> Self:
        """Discover project instructions and skills for a working directory.

        Args:
            working_directory (Path | str): Directory whose instruction scopes should apply.
            skill_manager (SkillManager | None): Explicit skill manager to use instead of
                discovering one.
            max_bytes (int): Maximum encoded size of the complete instruction document.

        Returns:
            InstructionsManager: Manager containing the discovered instruction sources.

        Raises:
            OSError: An applicable instruction file cannot be read.
            UnicodeError: An applicable instruction file is not valid UTF-8.
            ValueError: The configured limit is invalid or initial instructions exceed it.
        """
        directory = Path(working_directory).resolve()
        return cls(
            load_agents_instructions(directory),
            skill_manager or SkillManager.discover(directory),
            max_bytes=max_bytes,
        )

    @property
    def instructions(self) -> str | None:
        """Return the complete instructions for the next backend request.

        Returns:
            str | None: Project instructions, skill catalog, and active skill bodies in stable
                order, or ``None`` when no instructions are available.
        """
        return build_instructions(
            self._project_instructions,
            self._skill_manager.catalog(),
            self._active_skill_instructions(),
        )

    @property
    def skill_manager(self) -> SkillManager:
        """Return the skill manager providing catalog and activation state.

        Returns:
            SkillManager: The associated skill manager.
        """
        return self._skill_manager

    @property
    def max_bytes(self) -> int:
        """Return the complete instruction document size limit.

        Returns:
            int: Maximum encoded instruction size in bytes.
        """
        return self._max_bytes

    def list_skills(self) -> dict[str, Any]:
        """Return available skills and activation diagnostics.

        Returns:
            dict[str, Any]: Model-readable skill summaries and diagnostics.
        """
        return self._skill_manager.list()

    def activate_skill(self, name: str) -> dict[str, Any]:
        """Activate a skill for subsequent backend requests.

        Args:
            name (str): Exact available skill name.

        Returns:
            dict[str, Any]: Compact activation acknowledgement or structured error. The skill body
                is retained in the instruction document rather than returned in this result.

        Raises:
            OSError: The skill instruction file cannot be read.
            UnicodeError: The skill instruction file is not valid UTF-8.
            ValueError: The skill instruction file is malformed.
        """
        previously_active = any(
            skill.name == name for skill in self._skill_manager.activated_skills
        )
        result = self._skill_manager.activate(name)
        if "error" in result:
            return result
        size = self._encoded_size(self.instructions)
        if size > self._max_bytes:
            self._skill_manager.deactivate(name)
            return {
                "error": "instruction_budget_exceeded",
                "message": f"Activating skill '{name}' exceeds the instruction limit.",
                "max_bytes": self._max_bytes,
                "required_bytes": size,
            }
        result["instructions_updated"] = not previously_active
        return result

    def deactivate_skill(self, name: str) -> dict[str, Any]:
        """Deactivate a skill and remove it from subsequent backend requests.

        Args:
            name (str): Exact available skill name.

        Returns:
            dict[str, Any]: Compact deactivation acknowledgement or structured error.
        """
        return self._skill_manager.deactivate(name)

    def _active_skill_instructions(self) -> str | None:
        """Render active skill bodies in deterministic discovery order."""
        entries = []
        for skill, instructions in self._skill_manager.activated_instructions:
            entries.append(
                f'<skill name="{escape(skill.name)}" root="{escape(str(skill.location.parent))}">\n'
                f"{instructions}\n"
                "</skill>"
            )
        if not entries:
            return None
        return "<active_skills>\n" + "\n".join(entries) + "\n</active_skills>"

    @staticmethod
    def _encoded_size(instructions: str | None) -> int:
        """Return the UTF-8 size of an optional instruction document."""
        return len((instructions or "").encode("utf-8"))
