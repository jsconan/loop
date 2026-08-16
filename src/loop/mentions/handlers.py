"""Define independently injectable mention capabilities."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

from .. import constants
from ..completion import (
    CompletionAdapter,
    CompletionValue,
    MarkerCompletionAdapter,
    ProjectPathCompletionAdapter,
)
from ..models import ContextReference
from ..skills import InstructionsManager
from ..utils import is_path_ignored, iter_visible_paths, read_bounded_text


class MentionHandler(ABC):
    """Resolve one marker namespace and expose its matching completion capability."""

    @property
    @abstractmethod
    def marker(self) -> str:
        """Return the unique marker owned by this handler.

        Returns:
            str: Single marker character.
        """

    @property
    @abstractmethod
    def completion_adapter(self) -> CompletionAdapter:
        """Return the completion adapter for this mention namespace.

        Returns:
            CompletionAdapter: Handler-owned completion capability.
        """

    @property
    def accepts_markdown_links(self) -> bool:
        """Return whether ordinary Markdown links belong to this namespace.

        Returns:
            bool: Whether link destinations should be resolved gracefully by this handler.
        """
        return False

    @abstractmethod
    def candidates(self) -> Sequence[str]:
        """Return exact values accepted in submitted text.

        Returns:
            Sequence[str]: Current exact mention candidates.
        """

    @abstractmethod
    def resolve(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Resolve mentioned values and perform handler-owned effects atomically.

        Args:
            values (Sequence[str]): Mention values in source order.

        Returns:
            tuple[ContextReference, ...]: Context contributed by this handler.
        """

    @abstractmethod
    def resolve_optional(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Gracefully resolve values whose failures should leave ordinary prompt text.

        Args:
            values (Sequence[str]): Optional mention values in source order.

        Returns:
            tuple[ContextReference, ...]: Context contributed by successfully resolved values.
        """


class ProjectPathMentionHandler(MentionHandler):
    """Attach bounded snapshots for project path mentions.

    Args:
        working_directory (Callable[[], Path]): Lazy source of the current project directory.
        marker (str): Marker introducing project paths. Defaults to ``@``.
    """

    _working_directory: Callable[[], Path]
    _marker: str
    _completion_adapter: ProjectPathCompletionAdapter

    def __init__(self, working_directory: Callable[[], Path], marker: str = "@") -> None:
        self._working_directory = working_directory
        self._marker = marker
        self._completion_adapter = ProjectPathCompletionAdapter(marker, working_directory)

    @property
    def marker(self) -> str:
        """Return the configured path marker.

        Returns:
            str: Configured marker.
        """
        return self._marker

    @property
    def completion_adapter(self) -> CompletionAdapter:
        """Return short-lived project path completion.

        Returns:
            CompletionAdapter: Project path completion adapter.
        """
        return self._completion_adapter

    @property
    def accepts_markdown_links(self) -> bool:
        """Return whether project paths accept ordinary Markdown links.

        Returns:
            bool: Always true because relative link destinations may identify project paths.
        """
        return True

    def candidates(self) -> Sequence[str]:
        """Return current visible project-relative paths.

        Returns:
            Sequence[str]: Exact file and directory candidates.
        """
        return tuple(value.value for value in self._completion_adapter.values())

    def resolve(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Resolve unique paths into one bounded attachment set.

        Args:
            values (Sequence[str]): Project-relative paths in source order.

        Returns:
            tuple[ContextReference, ...]: Persistable file and directory snapshots.

        Raises:
            OSError: Referenced content cannot be read.
            UnicodeError: Referenced content is not valid UTF-8.
            ValueError: A path is unavailable, unsafe, unsupported, or exceeds the budget.
        """
        return self._resolve_paths(values, ignore_invalid=False)

    def resolve_optional(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Attach valid project paths while ignoring non-project Markdown destinations.

        Args:
            values (Sequence[str]): Markdown link destinations in source order.

        Returns:
            tuple[ContextReference, ...]: Snapshots for destinations that resolve safely.
        """
        return self._resolve_paths(values, ignore_invalid=True)

    def _resolve_paths(
        self,
        values: Sequence[str],
        *,
        ignore_invalid: bool,
    ) -> tuple[ContextReference, ...]:
        """Resolve unique paths under one shared attachment budget."""
        root = self._working_directory().resolve()
        references = []
        resolved_paths = set()
        remaining = constants.MAX_TOOL_CONTENT_BYTES
        for value in dict.fromkeys(values):
            try:
                path = (root / value.rstrip("/")).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Mentioned path '{value}' escapes the project.")
                if not path.exists() or is_path_ignored(path, root):
                    raise ValueError(f"Mentioned path '{value}' is unavailable.")
                if path in resolved_paths:
                    continue
                if remaining < 1:
                    raise ValueError("Mentioned paths exceed the context attachment limit.")
                if path.is_dir():
                    reference = self._directory_reference(path, value, remaining)
                elif path.is_file():
                    result = read_bounded_text(path, max_bytes=remaining)
                    reference = ContextReference(
                        kind="file",
                        path=value,
                        content=result["content"],
                        size_bytes=result["size_bytes"],
                        included_bytes=result["included_bytes"],
                        truncated=result["truncated"],
                    )
                else:
                    raise ValueError(f"Mentioned path '{value}' is not a file or directory.")
            except (OSError, UnicodeError, ValueError):
                if ignore_invalid:
                    continue
                raise
            resolved_paths.add(path)
            references.append(reference)
            remaining -= reference.included_bytes
        return tuple(references)

    @staticmethod
    def _directory_reference(path: Path, display_path: str, max_bytes: int) -> ContextReference:
        """Build one bounded immediate directory-listing snapshot."""
        content = "\n".join(
            child.relative_to(path).as_posix() + ("/" if child.is_dir() else "")
            for child in iter_visible_paths(path)
        )
        encoded = content.encode("utf-8")
        included = encoded[:max_bytes].decode("utf-8", errors="ignore")
        included_bytes = len(included.encode("utf-8"))
        return ContextReference(
            kind="directory",
            path=display_path,
            content=included,
            size_bytes=len(encoded),
            included_bytes=included_bytes,
            truncated=included_bytes < len(encoded),
        )


class SkillMentionHandler(MentionHandler):
    """Activate explicitly mentioned skills.

    Args:
        instructions_manager (InstructionsManager): Skill catalog and lifecycle owner.
        marker (str): Marker introducing skill names. Defaults to ``$``.
    """

    _instructions_manager: InstructionsManager
    _marker: str
    _completion_adapter: MarkerCompletionAdapter

    def __init__(self, instructions_manager: InstructionsManager, marker: str = "$") -> None:
        self._instructions_manager = instructions_manager
        self._marker = marker
        self._completion_adapter = MarkerCompletionAdapter(marker, self._completion_values)

    @property
    def marker(self) -> str:
        """Return the configured skill marker.

        Returns:
            str: Configured marker.
        """
        return self._marker

    @property
    def completion_adapter(self) -> CompletionAdapter:
        """Return skill-name completion.

        Returns:
            CompletionAdapter: Skill completion adapter.
        """
        return self._completion_adapter

    def candidates(self) -> Sequence[str]:
        """Return currently available skill names.

        Returns:
            Sequence[str]: Exact skill candidates.
        """
        return tuple(skill.name for skill in self._instructions_manager.skill_manager.skills)

    def resolve(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Atomically activate every uniquely mentioned skill.

        Args:
            values (Sequence[str]): Skill names in source order.

        Returns:
            tuple[ContextReference, ...]: Empty because skills contribute instructions.

        Raises:
            OSError: Skill instructions cannot be read.
            UnicodeError: Skill instructions are not valid UTF-8.
            ValueError: Skill activation fails.
        """
        previously_active = {name for name, _ in self._instructions_manager.active_skill_identities}
        activated = []
        try:
            for name in dict.fromkeys(values):
                result = self._instructions_manager.activate_skill(name)
                if "error" in result:
                    raise ValueError(result["message"])
                if name not in previously_active:
                    activated.append(name)
        except OSError, UnicodeError, ValueError:
            for name in reversed(activated):
                self._instructions_manager.deactivate_skill(name)
            raise
        return ()

    def resolve_optional(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Ignore optional values because skills require explicit mentions.

        Args:
            values (Sequence[str]): Optional values in source order.

        Returns:
            tuple[ContextReference, ...]: Always empty because skills are never inferred.
        """
        del values
        return ()

    def _completion_values(self) -> tuple[CompletionValue, ...]:
        """Return current skill completion values."""
        return tuple(
            CompletionValue(skill.name, skill.description)
            for skill in self._instructions_manager.skill_manager.skills
        )
