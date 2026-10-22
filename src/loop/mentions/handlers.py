"""Define independently injectable mention capabilities."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from .. import constants
from ..completion import (
    CompletionAdapter,
    CompletionValue,
    MarkerCompletionAdapter,
    ProjectPathCompletionAdapter,
)
from ..errors import Problem
from ..instructions import InstructionsManager
from ..instructions.models import SkillOperationError
from ..models import ContextReference, ContextReferenceKind
from ..utils import (
    PathReference,
    content_identity,
    encode_content_cursor,
    get_binary,
    is_path_ignored,
    iter_visible_paths,
    run_in_worker_async,
)


class MentionHandler(ABC):
    """Resolve one marker namespace and expose its matching completion capability.

    The default async adapters run synchronous callbacks in a bounded worker. Handlers that mutate
    owner-affine state must override the async callback and apply those changes on the owner task.
    """

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

    async def candidates_async(self) -> Sequence[str]:
        """Run synchronous candidate discovery through bounded worker admission by default.

        Returns:
            Sequence[str]: Current exact mention candidates.

        Raises:
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: The task is cancelled after candidate discovery settles.
        """
        return await run_in_worker_async(self.candidates)

    @abstractmethod
    def resolve(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Resolve mentioned values and perform handler-owned effects atomically.

        Args:
            values (Sequence[str]): Mention values in source order.

        Returns:
            tuple[ContextReference, ...]: Context contributed by this handler.
        """

    async def resolve_async(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Run synchronous resolution through bounded worker admission by default.

        Args:
            values (Sequence[str]): Mention values in source order.

        Returns:
            tuple[ContextReference, ...]: Context contributed by this handler.

        Raises:
            OSError: A handler cannot read its referenced resource.
            UnicodeError: Referenced content is not valid UTF-8.
            ValueError: A handler rejects a mention or cannot complete its operation.
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: The task is cancelled after resolution settles.
        """
        return await run_in_worker_async(self.resolve, tuple(values))

    @abstractmethod
    def resolve_optional(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Gracefully resolve values whose failures should leave ordinary prompt text.

        Args:
            values (Sequence[str]): Optional mention values in source order.

        Returns:
            tuple[ContextReference, ...]: Context contributed by successfully resolved values.
        """

    async def resolve_optional_async(
        self,
        values: Sequence[str],
    ) -> tuple[ContextReference, ...]:
        """Run synchronous optional resolution through bounded worker admission by default.

        Args:
            values (Sequence[str]): Optional mention values in source order.

        Returns:
            tuple[ContextReference, ...]: Context contributed by successfully resolved values.

        Raises:
            OSError: A handler cannot read its referenced resource.
            UnicodeError: Referenced content is not valid UTF-8.
            ValueError: A handler rejects a mention or cannot complete its operation.
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: The task is cancelled after resolution settles.
        """
        return await run_in_worker_async(self.resolve_optional, tuple(values))


class ProjectPathMentionHandler(MentionHandler):
    """Attach small previews and lazy immutable snapshots for project path mentions.

    Args:
        working_directory (PathReference): Current project-directory reference.
        marker (str): Marker introducing project paths. Defaults to ``@``.
        sync_blocking_runner (Callable[..., Any] | None): Bounded worker for synchronous UI
            completion. Defaults to direct traversal for standalone handlers.
        async_blocking_runner (Callable[..., Awaitable[Any]] | None): Bounded worker for async
            filesystem reads. Defaults to a shared bounded process worker.
    """

    _working_directory: PathReference
    _marker: str
    _completion_adapter: ProjectPathCompletionAdapter
    _async_blocking_runner: Callable[..., Awaitable[Any]] | None

    def __init__(
        self,
        working_directory: PathReference,
        marker: str = "@",
        *,
        sync_blocking_runner: Callable[..., Any] | None = None,
        async_blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._working_directory = working_directory
        self._marker = marker
        self._async_blocking_runner = async_blocking_runner
        self._completion_adapter = ProjectPathCompletionAdapter(
            marker,
            working_directory,
            sync_blocking_runner=sync_blocking_runner,
            async_blocking_runner=async_blocking_runner,
        )

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

    async def candidates_async(self) -> Sequence[str]:
        """Return current visible paths without traversing the project on the event loop.

        Returns:
            Sequence[str]: Exact file and directory candidates.

        Raises:
            OSError: The project path listing cannot be read.
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: The awaiting task is cancelled.
        """
        values = await self._completion_adapter.values_async()
        return tuple(value.value for value in values)

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

    async def resolve_async(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Resolve paths using detached filesystem inputs and build references on the owner.

        Args:
            values (Sequence[str]): Project-relative paths in source order.

        Returns:
            tuple[ContextReference, ...]: Persistable file and directory snapshots.

        Raises:
            OSError: Referenced content cannot be read.
            UnicodeError: Referenced content is not valid UTF-8.
            ValueError: A path is unavailable, unsafe, unsupported, or exceeds the budget.
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: The awaiting task is cancelled.
        """
        return await self._resolve_paths_async(values, ignore_invalid=False)

    async def resolve_optional_async(
        self,
        values: Sequence[str],
    ) -> tuple[ContextReference, ...]:
        """Resolve optional Markdown paths without blocking and skip invalid destinations.

        Args:
            values (Sequence[str]): Markdown link destinations in source order.

        Returns:
            tuple[ContextReference, ...]: Snapshots for destinations that resolve safely.

        Raises:
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: The awaiting task is cancelled.
        """
        return await self._resolve_paths_async(values, ignore_invalid=True)

    def _resolve_paths(
        self,
        values: Sequence[str],
        *,
        ignore_invalid: bool,
    ) -> tuple[ContextReference, ...]:
        """Resolve unique paths under one shared attachment budget."""
        root = self._working_directory.get()
        sources = self._read_mentioned_paths(root, tuple(values), ignore_invalid)
        return self._build_references(sources, ignore_invalid=ignore_invalid)

    async def _resolve_paths_async(
        self,
        values: Sequence[str],
        *,
        ignore_invalid: bool,
    ) -> tuple[ContextReference, ...]:
        """Read with a detached root and build returned references on the owner."""
        root = self._working_directory.get()
        sources = await (
            self._async_blocking_runner(
                self._read_mentioned_paths,
                root,
                tuple(values),
                ignore_invalid,
            )
            if self._async_blocking_runner is not None
            else run_in_worker_async(
                self._read_mentioned_paths,
                root,
                tuple(values),
                ignore_invalid,
            )
        )
        if self._working_directory.get() != root:
            raise ValueError("Working directory changed while resolving mentioned paths.")
        return self._build_references(sources, ignore_invalid=ignore_invalid)

    @staticmethod
    def _read_mentioned_paths(
        root: Path,
        values: tuple[str, ...],
        ignore_invalid: bool,
    ) -> tuple[tuple[ContextReferenceKind, str, str | bytes], ...]:
        """Read bounded path snapshots using only detached inputs."""
        root = root.resolve()
        sources = []
        resolved_paths = set()
        for value in dict.fromkeys(values):
            try:
                path = (root / value.rstrip("/")).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Mentioned path '{value}' escapes the project.")
                if not path.exists() or is_path_ignored(path, root):
                    raise ValueError(f"Mentioned path '{value}' is unavailable.")
                if path in resolved_paths:
                    continue
                if path.is_dir():
                    content = "\n".join(
                        child.relative_to(path).as_posix() + ("/" if child.is_dir() else "")
                        for child in iter_visible_paths(path)
                    )
                    kind = "directory"
                elif path.is_file():
                    if path.stat().st_size > constants.MAX_FETCH_BYTES:
                        raise ValueError(
                            f"Mentioned path '{value}' exceeds the "
                            f"{constants.MAX_FETCH_BYTES}-byte snapshot limit."
                        )
                    encoded = path.read_bytes()
                    if b"\0" in encoded:
                        content = encoded
                    else:
                        try:
                            content = encoded.decode("utf-8")
                        except UnicodeDecodeError:
                            content = encoded
                    kind = "file"
                else:
                    raise ValueError(f"Mentioned path '{value}' is not a file or directory.")
                content_bytes = get_binary(content)
                if len(content_bytes) > constants.MAX_FETCH_BYTES:
                    raise ValueError(
                        f"Mentioned path '{value}' exceeds the "
                        f"{constants.MAX_FETCH_BYTES}-byte snapshot limit."
                    )
            except (OSError, UnicodeError, ValueError):
                if ignore_invalid:
                    continue
                raise
            resolved_paths.add(path)
            sources.append((kind, value, content))
        return tuple(sources)

    def _build_references(
        self,
        sources: tuple[tuple[ContextReferenceKind, str, str | bytes], ...],
        *,
        ignore_invalid: bool,
    ) -> tuple[ContextReference, ...]:
        """Build attachment references from detached path snapshots."""
        include_file_previews = len(sources) <= 2
        lazy_binary_sources = [
            value
            for kind, value, content in sources
            if kind == "file"
            and isinstance(content, bytes)
            and (not include_file_previews or len(content) > constants.MAX_REFERENCE_PREVIEW_BYTES)
        ]
        if lazy_binary_sources:
            if ignore_invalid:
                lazy_binary_source_values = set(lazy_binary_sources)
                sources = [
                    source for source in sources if source[1] not in lazy_binary_source_values
                ]
                include_file_previews = len(sources) <= 2
            else:
                raise ValueError(
                    "Mentioned binary paths must fit the eager attachment budget: "
                    + ", ".join(repr(value) for value in lazy_binary_sources)
                    + "."
                )
        remaining_preview_bytes = constants.MAX_REFERENCE_TOTAL_PREVIEW_BYTES
        references = []
        for kind, value, content in sources:
            encoded_size = len(get_binary(content))
            desired_preview_bytes = (
                constants.MAX_REFERENCE_PREVIEW_BYTES
                if kind == "directory"
                else (
                    encoded_size
                    if include_file_previews
                    and encoded_size <= constants.MAX_REFERENCE_PREVIEW_BYTES
                    else 0
                )
            )
            preview_bytes = min(desired_preview_bytes, remaining_preview_bytes)
            reference = self._reference(
                kind,
                value,
                content,
                max_preview_bytes=preview_bytes,
            )
            references.append(reference)
            remaining_preview_bytes -= reference.included_bytes
        return tuple(references)

    def _reference(
        self,
        kind: ContextReferenceKind,
        display_path: str,
        content: str | bytes,
        *,
        max_preview_bytes: int,
    ) -> ContextReference:
        """Build an eager binary attachment or a text preview with a lazy snapshot.

        Binary resources must be fully included because the available continuation tool reads
        UTF-8 text only. Callers enforce that policy before constructing this reference.
        """
        encoded = get_binary(content)
        if isinstance(content, str):
            included = encoded[:max_preview_bytes].decode("utf-8", errors="ignore")
            included_bytes = len(included.encode("utf-8"))
        else:
            included_bytes = min(max_preview_bytes, len(encoded))
        truncated = included_bytes < len(encoded)
        handle, version = content_identity(encoded)
        media_type = guess_type(display_path)[0]
        if isinstance(content, bytes) and (media_type is None or media_type.startswith("text/")):
            media_type = "application/octet-stream"
        return ContextReference(
            kind=kind,
            path=display_path,
            content=content,
            size_bytes=len(encoded),
            included_bytes=included_bytes,
            truncated=truncated,
            handle=handle,
            next_cursor=(encode_content_cursor(handle, included_bytes) if truncated else None),
            version=version,
            media_type=media_type
            or ("text/plain" if isinstance(content, str) else "application/octet-stream"),
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

    async def candidates_async(self) -> Sequence[str]:
        """Return the in-memory skill catalog on the current owner task.

        Returns:
            Sequence[str]: Exact skill candidates.
        """
        return self.candidates()

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
                if isinstance(result, (Problem, SkillOperationError)):
                    raise ValueError(  # noqa: TRY004 - mention resolution exposes ValueError.
                        result.detail
                    )
                if name not in previously_active:
                    activated.append(name)
        except (OSError, UnicodeError, ValueError):
            for name in reversed(activated):
                self._instructions_manager.deactivate_skill(name)
            raise
        return ()

    async def resolve_async(self, values: Sequence[str]) -> tuple[ContextReference, ...]:
        """Activate mentioned skills while keeping manager state on the owner task.

        Args:
            values (Sequence[str]): Skill names in source order.

        Returns:
            tuple[ContextReference, ...]: Empty because skills contribute instructions.

        Raises:
            OSError: Skill instructions cannot be read.
            UnicodeError: Skill instructions are not valid UTF-8.
            ValueError: Skill activation fails.
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: Cancellation occurs after any admitted read settles.
        """
        previously_active = {name for name, _ in self._instructions_manager.active_skill_identities}
        activated = []
        try:
            for name in dict.fromkeys(values):
                result = await self._instructions_manager.activate_skill_async(name)
                if isinstance(result, (Problem, SkillOperationError)):
                    raise ValueError(  # noqa: TRY004 - mention resolution exposes ValueError.
                        result.detail
                    )
                if name not in previously_active:
                    activated.append(name)
        except (Exception, asyncio.CancelledError):
            for name in reversed(activated):
                await self._instructions_manager.deactivate_skill_async(name)
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
