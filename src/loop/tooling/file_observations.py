"""Bind file mutations to immutable versions present in one model request."""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import Problem
from ..models import ContextReference, Message, ModelContextItem, ToolCall, ToolResult
from ..permissions import Action, FileTarget, Operation, Operations
from ..utils import VirtualPath, content_digest, format_content_diff, read_bounded_text


@dataclass
class FileView:
    """Retain visible byte ranges belonging to one immutable source version.

    Args:
        version (str): Immutable version identifier of the source file.
        size (int): Total size of the source file in bytes.
        ranges (list[tuple[int, bytes]]): List of byte ranges and their content that are visible
            to the model. Each tuple contains the start offset and the corresponding byte content.
            Defaults to an independent empty list.
    """

    version: str
    size: int
    ranges: list[tuple[int, bytes]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Return whether all source bytes were supplied to the model.

        Returns:
            bool: Whether the ranges cover the complete source without gaps.
        """
        return (
            len(self.ranges) == 1 and self.ranges[0][0] == 0 and len(self.ranges[0][1]) == self.size
        )


class FileObservations:
    """Reconstruct file observations from the context supplied for one generation.

    Args:
        context (Iterable[ModelContextItem]): Active context after compaction, result bounding,
            and backend preparation. Opaque summaries do not establish file observations.
        paths (VirtualPath): Translation shared with model tool dispatch.

    File reads performed by the resulting tool batch are deliberately excluded until their
    results appear in another model request. Successful writes can advance the view because
    their content was supplied by the model itself. Instances must not be shared across requests.
    Each recorded call contributes at most one result; later calls may reuse an identifier.
    """

    _paths: VirtualPath
    _files: dict[str, FileView]
    _missing: set[str]
    _recovery_incomplete: bool
    _references: dict[str, tuple[str, str, int]]

    def __init__(self, context: Iterable[ModelContextItem], paths: VirtualPath) -> None:
        self._paths = paths
        self._files = {}
        self._missing = set()
        self._recovery_incomplete = False
        self._references = {}
        calls = {}
        for item in context:
            if isinstance(item, Message):
                for reference in item.context:
                    self._observe_reference(reference)
            elif isinstance(item, ToolCall):
                calls[item.call_id] = item
            elif isinstance(item, ToolResult) and item.call_id in calls:
                self.observe_result(calls.pop(item.call_id), item.output)

    def _observe_reference(self, reference: ContextReference) -> None:
        """Retain the reference identity and only bytes delivered as source text."""
        if reference.kind != "file" or reference.version is None:
            return
        if reference.handle is not None:
            self._references[reference.handle] = (
                reference.path,
                reference.version,
                reference.size_bytes,
            )
        if reference.reused:
            for view in tuple(self._files.values()):
                if view.version == reference.version:
                    for start, content in view.ranges:
                        self._record(
                            reference.path,
                            view.version,
                            view.size,
                            start,
                            content.decode("utf-8"),
                        )
        content = (
            reference.content
            if isinstance(reference.content, str) and not reference.payload_redacted
            else ""
        )
        self._record(
            reference.path,
            reference.version,
            reference.size_bytes,
            reference.payload_start_bytes,
            content,
        )

    @classmethod
    def for_recovery(
        cls,
        context: Iterable[ModelContextItem],
        paths: VirtualPath,
        pending_calls: Iterable[ToolCall],
        *,
        prepare_context: Callable[[Iterable[ModelContextItem]], Iterable[ModelContextItem]]
        | None = None,
    ) -> "FileObservations":
        """Recover the view used to generate an interrupted batch of tool calls.

        Args:
            context (Iterable[ModelContextItem]): Active persisted context, including completed
                results from the interrupted response.
            paths (VirtualPath): Translation shared with model tool dispatch.
            pending_calls (Iterable[ToolCall]): Unresolved requests from the interrupted batch.
            prepare_context (Callable[[Iterable[ModelContextItem]], Iterable[ModelContextItem]]
                | None): Backend preparation applied only to context delivered before the pending
                generation. Completed own mutations retain their original arguments. Defaults to
                preserving the supplied context.

        Returns:
            FileObservations: The preceding model view advanced only by completed own writes.
                Incomplete or mismatched history refuses file mutations until a new generation.

        Raises:
            ValueError: If the supplied backend preparation rejects a reference payload.
        """
        items = list(context)
        pending_items = tuple(pending_calls)
        pending = {call.call_id: call for call in pending_items}
        if not pending:
            return cls(prepare_context(items) if prepare_context is not None else items, paths)
        if len(pending) != len(pending_items):
            return cls._incomplete_recovery(paths)
        positions = {
            item.call_id: index
            for index, item in enumerate(items)
            if isinstance(item, ToolCall) and item.call_id in pending
        }
        if positions.keys() != pending.keys():
            return cls._incomplete_recovery(paths)
        for call_id, index in positions.items():
            item = items[index]
            call = pending[call_id]
            if item.name != call.name or item.arguments != call.arguments:
                return cls._incomplete_recovery(paths)
        first = min(positions.values())
        boundary = first
        while boundary:
            previous = items[boundary - 1]
            if isinstance(previous, ToolResult) or (
                isinstance(previous, Message) and previous.role == "user"
            ):
                break
            boundary -= 1
        if not cls._is_response_suffix(items[boundary:]):
            return cls._incomplete_recovery(paths)
        prefix = items[:boundary]
        observations = cls(
            prepare_context(prefix) if prepare_context is not None else prefix, paths
        )
        calls = {}
        for item in items[boundary:]:
            if isinstance(item, ToolCall):
                calls[item.call_id] = item
            elif isinstance(item, ToolResult):
                observations.observe_result(calls[item.call_id], item.output, include_reads=False)
        return observations

    @staticmethod
    def _is_response_suffix(items: Iterable[ModelContextItem]) -> bool:
        """Require unique calls followed by at most one matching result per call."""
        results_started = False
        calls = set()
        for item in items:
            if isinstance(item, ToolResult):
                if item.call_id not in calls:
                    return False
                calls.remove(item.call_id)
                results_started = True
                continue
            if results_started or (isinstance(item, Message) and item.role == "user"):
                return False
            if isinstance(item, ToolCall):
                if item.call_id in calls:
                    return False
                calls.add(item.call_id)
        return True

    @classmethod
    def _incomplete_recovery(cls, paths: VirtualPath) -> "FileObservations":
        """Require regeneration when the pending batch cannot be matched to its history."""
        observations = cls((), paths)
        observations._recovery_incomplete = True
        return observations

    def _path(self, value: str) -> str:
        """Resolve an observed path using the same mutation spelling as the planner."""
        path = Path(self._paths.resolve(value))
        return str(path.parent.resolve() / path.name)

    def _record(self, path: str, version: str, size: int, start: int, content: str) -> None:
        """Merge delivered ranges without treating omitted bytes as visible."""
        try:
            path = self._path(path)
        except (OSError, ValueError):
            return
        self._missing.discard(path)
        view = self._files.get(path)
        if view is None or view.version != version or view.size != size:
            view = FileView(version, size)
            self._files[path] = view
        if not content and size:
            return
        chunks = sorted([*view.ranges, (start, content.encode("utf-8"))])
        merged = []
        for offset, chunk in chunks:
            if merged and offset <= merged[-1][0] + len(merged[-1][1]):
                previous, text = merged[-1]
                overlap = min(previous + len(text) - offset, len(chunk))
                if text[offset - previous : offset - previous + overlap] != chunk[:overlap]:
                    view.ranges = []
                    return
                merged[-1] = (previous, text + chunk[overlap:])
            else:
                merged.append((offset, chunk))
        view.ranges = merged
        if view.complete and content_digest(merged[0][1]) != view.version:
            view.ranges = []

    def observe_result(self, call: ToolCall, output: str, *, include_reads: bool = True) -> None:
        """Consume delivered file observations or a successful mutation's committed version.

        Absence observations require an explicitly failed envelope with trusted read receipts.

        Args:
            call (ToolCall): Request corresponding to this tool result.
            output (str): Serialized result envelope retained in model context.
            include_reads (bool): Whether reads have been delivered to the model. Pass ``False``
                while executing the current response's batch. Defaults to ``True``.
        """
        try:
            payload = json.loads(output)
            if not isinstance(payload, dict):
                return
            receipts = payload.get("read_versions", {})
            if not isinstance(receipts, dict):
                return
            if payload.get("ok") is not True:
                if (
                    payload.get("ok") is False
                    and include_reads
                    and payload.get("problem", {}).get("code")
                    in {"filesystem.path_missing", "filesystem.stale_context"}
                ):
                    for source, version in receipts.items():
                        if version is None:
                            path = self._path(source)
                            self._files.pop(path, None)
                            self._missing.add(path)
                return
            result = payload.get("result")
            if include_reads and isinstance(result, dict) and "content" in result:
                self._observe_read(result, receipts)
            elif "file_versions" in payload:
                arguments = None if call.arguments_redacted else json.loads(call.arguments)
                for path, version in payload.get("file_versions", {}).items():
                    self._advance(self._path(path), version, arguments, receipts.get(path))
        except (AttributeError, KeyError, TypeError, ValueError):
            # Legacy, bounded, or no-longer-resolvable results require another explicit read.
            return

    def _observe_read(self, result: dict, receipts: dict) -> None:
        """Merge a validated range only when its source identity has a trusted receipt."""
        if "version" not in result:
            handle = result.get("handle")
            if handle not in receipts or receipts[handle] is not None:
                return
            reference = self._references.get(handle)
            if reference is None:
                return
            path, version, size = reference
        else:
            path, version, size = result["path"], result["version"], result["size_bytes"]
            if receipts.get(path) != version:
                return
        content, start = result["content"], result["start_byte"]
        if (
            not isinstance(content, str)
            or not isinstance(version, str)
            or not isinstance(size, int)
            or not isinstance(start, int)
        ):
            return
        if result.get("payload_redacted") is True:
            content = ""
        if start < 0 or start + len(content.encode("utf-8")) > size:
            return
        self._record(path, version, size, start, content)

    def _advance(
        self,
        path: str,
        version: str | None,
        arguments: dict | None,
        source_version: str | None,
    ) -> None:
        """Advance visible bytes only from the committed mutation's verified source version."""
        if version is None:
            self._files.pop(path, None)
            self._missing.add(path)
            return
        view = self._files.get(path)
        # Retain existence, but require a read unless the result can be reconstructed exactly.
        self._files[path] = FileView(version, 0)
        self._missing.discard(path)
        if arguments is None:
            return
        if "content" in arguments:
            content = arguments["content"].encode("utf-8")
            if content_digest(content) == version:
                self._files[path] = FileView(version, len(content), [(0, content)])
            return
        if view is None or view.version != source_version:
            return
        old = arguments["old_content"].encode("utf-8")
        new = arguments["new_content"].encode("utf-8")
        if not old or not any(old in content for _, content in view.ranges):
            return
        if arguments.get("replace_all") and not view.complete:
            return
        ranges = []
        shift = 0
        for offset, content in view.ranges:
            updated = content.replace(old, new)
            ranges.append((offset + shift, updated))
            shift += len(updated) - len(content)
        updated_view = FileView(version, view.size + shift, ranges)
        if updated_view.complete and content_digest(ranges[0][1]) != version:
            return
        self._files[path] = updated_view

    def _changed_diff(self, path: str, view: FileView, version: str) -> str | None:
        """Describe bounded divergence only when both complete source versions are available."""
        if not view.complete:
            return None
        try:
            current = read_bounded_text(Path(path))
        except (OSError, ValueError):
            return None
        if current["truncated"] or content_digest(current["content"]) != version:
            return None
        return format_content_diff(
            view.ranges[0][1].decode("utf-8"), current["content"], self._paths.display(path)
        )

    def check(
        self,
        operations: Operations,
        arguments: dict[str, object],
        *,
        authorized_reads: Operations = (),
        absence_receipts: dict[str, None] | None = None,
    ) -> Problem | None:
        """Reject a mutation based on missing, stale, or insufficient visible file content.

        Args:
            operations (Operations): Inspected mutation operations and future effects from
                authorized planning.
            arguments (dict[str, object]): Canonical arguments for the proposed mutation.
            authorized_reads (Operations): Reads authorized by earlier phases of this invocation.
                Defaults to no authority for additional reads while preparing conflict diffs.
            absence_receipts (dict[str, None] | None): Optional registry-owned output collector
                for virtual paths found absent by inspected creation targets. Receipts accompany
                rejected mutations without changing this generation's observations. Defaults to
                ``None``.

        Returns:
            Problem | None: Actionable warning requiring reconsideration, or ``None`` if
                applicable file preconditions agree with the model's view. Deletion warnings
                permit reconsidered creation in a subsequent generation without another read.
        """
        absence_receipts = {} if absence_receipts is None else absence_receipts
        for operation in operations:
            target = operation.target
            if operation.action not in {
                Action.FILESYSTEM_CREATE,
                Action.FILESYSTEM_REPLACE,
                Action.FILESYSTEM_DELETE,
            } or not isinstance(target, FileTarget):
                continue
            problem = self._check_mutation(operation, arguments, authorized_reads, absence_receipts)
            if problem is not None:
                return problem
        return None

    def _check_mutation(
        self,
        operation: Operation,
        arguments: dict[str, object],
        authorized_reads: Operations,
        absence_receipts: dict[str, None],
    ) -> Problem | None:
        """Check one file mutation against its known existence, revision, and visible ranges."""
        target = operation.target
        if self._recovery_incomplete:
            return Problem(
                code="filesystem.stale_context",
                title="File context needs refreshing",
                severity="warning",
                detail=(
                    "The interrupted generation's file context is unavailable. Read the "
                    "current paths and generate a new change. Nothing was written."
                ),
                operation=operation.tool_id,
                metadata={"reread_required": True, "regeneration_required": True},
            )
        if target.recursive:
            return None
        view = self._files.get(target.path)
        creating = operation.action is Action.FILESYSTEM_CREATE and target.expected_exists is False
        if (creating and view is None) or (not creating and target.expected_digest is None):
            return None
        metadata: dict[str, object] = {"reread_required": True}
        if creating or target.path in self._missing:
            code = "filesystem.stale_context"
            detail = (
                "The file's existence changed after your observation. Read the current path "
                "and reconsider the proposed change before proceeding."
            )
            if creating:
                detail = (
                    "The previously observed file was deleted. Reconsider whether to recreate "
                    "it in your next response; no additional missing-file read is required."
                )
                metadata = {"reread_required": False, "regeneration_required": True}
                absence_receipts[self._paths.display(target.path)] = None
        elif view is None or not view.ranges:
            code, detail = "filesystem.unread_file", "Read this file before modifying it."
        elif view.version != f"sha256:{target.expected_digest}":
            code = "filesystem.stale_context"
            detail = (
                "This file changed after the version supplied to you. Read the current file, "
                "reconsider the proposed change, and preserve the intervening changes."
            )
            if any(
                read.action is Action.FILESYSTEM_READ
                and isinstance(read.target, FileTarget)
                and read.target.path == target.path
                for read in authorized_reads
            ):
                difference = self._changed_diff(
                    target.path, view, f"sha256:{target.expected_digest}"
                )
                if difference is not None:
                    metadata["diff"] = difference
        else:
            if ("content" in arguments or arguments.get("replace_all")) and not view.complete:
                code = "filesystem.incomplete_read"
                detail = "Read the complete file before replacing its content or all matches."
            elif "old_content" in arguments and not any(
                str(arguments["old_content"]).encode("utf-8") in content
                for _, content in view.ranges
            ):
                code = "filesystem.unread_content"
                detail = "Read the file range containing the exact content to edit."
            else:
                return None
        return Problem(
            code=code,
            title="File context needs refreshing",
            severity="warning",
            detail=f"{self._paths.display(target.path)}: {detail} Nothing was written.",
            operation=operation.tool_id,
            metadata=metadata,
        )
