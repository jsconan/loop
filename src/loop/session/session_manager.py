"""Session manager for handling user sessions."""

import asyncio
import json
import logging
import os
import shutil
from asyncio import current_task
from collections.abc import AsyncIterable, Awaitable, Callable, Generator, Iterable
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock, get_ident
from typing import Any

from ..errors import Problem, log_problem
from ..instructions.models import CapturedInstruction
from ..interaction import ConsoleInteraction, Interaction, InteractionOutput, ResponsePresenter
from ..models import (
    AgentRunStopReason,
    CompactionResult,
    ContentArtifact,
    ContextReference,
    ConversationItem,
    Message,
    ModelAssignment,
    ModelContextItem,
    Reasoning,
    Response,
    ResponseEvent,
    RunMetrics,
    ToolCall,
    ToolResult,
)
from ..permissions import AuthorizationResult
from ..response import ResponseCollector
from ..telemetry import (
    TelemetryContext,
    telemetry_activity,
    telemetry_audit,
    telemetry_error,
    telemetry_span,
)
from ..utils import (
    PathHolder,
    PathReference,
    bound_tool_result,
    cached_metadata,
    cached_path,
    content_digest,
    content_identity,
    encode_content_cursor,
    ensure_private_directory,
    get_binary,
    new_uuid,
    register_cached_metadata,
    run_in_worker_async,
    sha256_digest,
    store_content,
    utc_now,
    validate_content_handle,
)
from ..utils.path import private_directory_lock
from .models import (
    SESSION_NAME_SOURCE_GENERATED,
    SESSION_NAME_SOURCE_INITIAL,
    Compaction,
    CompactionEvent,
    InputConsumedEvent,
    InputDeliveredEvent,
    InputPriorityChangedEvent,
    InputQueuedEvent,
    InputRecalledEvent,
    InputRejectedEvent,
    InstructionSnapshot,
    InteractionRequestedEvent,
    InteractionSettledEvent,
    PermissionEvent,
    ResponsePersisted,
    RunCompletedEvent,
    RunFailedEvent,
    RunRetryStartedEvent,
    RunStartedEvent,
    SessionEventModel,
    SessionExecutionConflictError,
    SessionNameContext,
    SessionNameGenerator,
    SessionRecoveryState,
    SessionWorkspaceMismatchError,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)
from .naming import initial_session_name
from .session import Session
from .store import MemorySessionStore, SessionStore, WorkingDirectoryBinding

type ExecutionKey = tuple[tuple[str, str | int], str]
type ExecutionOwner = tuple[int, int, int | None]

_LOGGER = logging.getLogger(__name__)
_EXECUTION_OWNERS: dict[ExecutionKey, tuple[ExecutionOwner, int]] = {}
_EXECUTION_OWNERS_LOCK = Lock()
_SESSION_SELECTION: ContextVar[object | None] = ContextVar("loop_session_selection", default=None)


def _execution_owner(manager: "SessionManager") -> ExecutionOwner:
    """Identify one synchronous thread or asynchronous task execution owner."""
    try:
        task = current_task()
    except RuntimeError:
        task = None
    return (id(manager), get_ident(), id(task) if task is not None else None)


class SessionManager:
    """Manage user sessions, including persistence and interaction.

    Args:
        interaction (Interaction | None): Service used for user input and output. Defaults to a
            console interaction.
        session (Session | str | None): Active session or persisted session identifier to load.
            Defaults to a fresh session.
        session_store (SessionStore | None): Store used to persist and retrieve sessions. Defaults
            to an instance-local memory store.
        workspace_id (str | None): Durable workspace identity owning every managed session.
        async_blocking_runner (Callable[..., Awaitable[Any]] | None): Application-owned bounded
            worker used for synchronous persistence from async workflows. Defaults to a shared
            bounded process worker.

    Raises:
        SessionNotFoundError: If the requested persisted session does not exist.
        UnsupportedConversationItemError: If a serialized conversation item type is unsupported.
        ValueError: If the session or persisted format is invalid.
        OSError: Working storage cannot be prepared for interaction recovery.
        SessionExecutionConflictError: Another harness owns a session requiring recovery.
    """

    _state_lock: RLock
    _execution_depth: int
    _execution_resources: ExitStack
    _selection_block_depth: int
    _interaction: Interaction
    _temporary_directory: PathHolder | None
    _session: Session
    _session_store: SessionStore
    _workspace_id: str | None
    _async_blocking_runner: Callable[..., Awaitable[Any]] | None
    _response_persisted_observers: list[tuple[object, Callable[[ResponsePersisted], None]]]
    _response_persisted_observers_lock: Lock
    _persistence_lock: Lock
    _commit_lock: Lock
    _async_commit_lock: asyncio.Lock

    def __init__(
        self,
        interaction: Interaction | None = None,
        session: Session | str | None = None,
        session_store: SessionStore | None = None,
        workspace_id: str | None = None,
        async_blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._state_lock = RLock()
        self._execution_depth = 0
        self._execution_resources = ExitStack()
        self._selection_block_depth = 0
        self._interaction = interaction or ConsoleInteraction()
        self._session_store = session_store or MemorySessionStore()
        self._async_blocking_runner = async_blocking_runner
        self._response_persisted_observers = []
        self._response_persisted_observers_lock = Lock()
        self._persistence_lock = Lock()
        self._commit_lock = Lock()
        self._async_commit_lock = asyncio.Lock()
        if workspace_id == "":
            raise ValueError("Workspace identifier must not be empty.")
        self._workspace_id = workspace_id
        self._temporary_directory = None

        if session and isinstance(session, (str, Session)):
            self.load_session(session)
        else:
            self._select_session(Session(workspace_id=self._workspace_id))

    @property
    def session(self) -> Session:
        """Return the active session.

        Returns:
            Session: The mutable session used by the loop.
        """
        with self._state_lock:
            return self._session

    @property
    def temporary_directory(self) -> PathReference:
        """Return the stable path reference following the active session.

        Returns:
            PathReference: Private working directory shared by file tools and commands. Durable
                stores retain it across restarts; in-memory stores retain it for their lifetime.

        Raises:
            ValueError: The session identity is empty or storage is aliased, not private, or
                owned by another user.
            OSError: The working directory cannot be created or inspected.
            SessionExecutionConflictError: Another harness owns this session's working storage.
        """
        with self._state_lock:
            if self._temporary_directory is None:
                with self._session_directory(self._session.id) as directory:
                    self._temporary_directory = PathHolder(directory)
            return self._temporary_directory

    @contextmanager
    def _session_directory(self, session_id: str) -> Generator[Path, None, None]:
        """Hold ownership of a validated generation without replacing lost data."""
        if not session_id:
            raise ValueError("Session identifier must not be empty.")
        root = self._session_store.temporary_root
        existing = self._session_store.working_binding(session_id)
        try:
            ensure_private_directory(root, create=existing is None or existing.retired)
        except OSError as error:
            raise ValueError(
                "Session working storage recovery failed: namespace unavailable."
            ) from error
        locks = root / ".locks"
        ensure_private_directory(locks)
        identity = sha256_digest(session_id.encode("utf-8", errors="surrogatepass"))
        with ExitStack() as resources:
            try:
                resources.enter_context(private_directory_lock(locks, identity))
            except BlockingIOError as error:
                raise SessionExecutionConflictError(session_id) from error
            yield self._bound_directory(session_id, root, identity)

    def _bound_directory(self, session_id: str, root: Path, identity: str) -> Path:
        """Validate durable identity or commit an independently created/adopted generation."""
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            return self._working_directory(session_id, root, identity, descriptor)
        finally:
            os.close(descriptor)

    def _working_directory(
        self,
        session_id: str,
        root: Path,
        identity: str,
        descriptor: int,
    ) -> Path:
        """Validate or create storage through an approved namespace descriptor."""
        binding = self._session_store.working_binding(session_id)
        anchor = (root, descriptor)
        if binding is not None and not binding.retired:
            relative = Path(binding.generation)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts != (identity,)
                and not (
                    len(relative.parts) == 2
                    and relative.parts[0].startswith(identity + ".")
                    and relative.parts[1] == "files"
                )
            ):
                raise ValueError("Session working storage recovery failed: invalid binding.")
            directory = root / relative
            try:
                if directory.parent != root:
                    ensure_private_directory(directory.parent, create=False, anchor=anchor)
                ensure_private_directory(directory, create=False, anchor=anchor)
                details = directory.stat()
                if (details.st_dev, details.st_ino) != (binding.device, binding.inode):
                    raise ValueError("Session working storage recovery failed: identity changed.")
            except OSError as error:
                raise ValueError(
                    "Session working storage recovery failed: root unavailable."
                ) from error
            return directory
        legacy = root / identity
        if binding is None and (legacy.exists() or legacy.is_symlink()):
            ensure_private_directory(legacy, create=False, anchor=anchor)
            directory = legacy
        else:
            parent = root / (identity + "." + str(new_uuid()))
            ensure_private_directory(parent, anchor=anchor)
            directory = parent / "files"
            ensure_private_directory(directory, anchor=anchor)
        details = directory.stat()
        self._session_store.bind_working_directory(
            session_id,
            WorkingDirectoryBinding(
                str(directory.relative_to(root)), details.st_dev, details.st_ino
            ),
        )
        telemetry_audit("session.working_directory.bound", session_id=session_id)
        return directory

    def discard_working_directory(self) -> None:
        """Retire current working storage before cleaning it, preserving conversation history.

        Commit a fresh generation before cleanup. Old descriptors and macOS descendants
        are not revoked; retirement never reuses their former location. Remove the generation
        parent only when empty, preserving unrelated contents and adopted legacy namespaces.

        Raises:
            SessionExecutionConflictError: An operation or another harness owns this session.
            ValueError: The current storage identity is unsafe or missing.
            OSError: Retirement or cleanup fails; a committed retirement remains in effect.
        """
        with self._state_lock:
            self._check_session_selection()
            root = self._session_store.temporary_root
            identity = sha256_digest(self._session.id.encode("utf-8", errors="surrogatepass"))
            with self._session_directory(self._session.id) as directory:
                binding = self._session_store.working_binding(self._session.id)
                self._session_store.bind_working_directory(
                    self._session.id, replace(binding, retired=True)
                )
                telemetry_audit("session.working_directory.retired", session_id=self._session.id)
                replacement = self._bound_directory(self._session.id, root, identity)
                if self._temporary_directory is not None:
                    self._temporary_directory.set(replacement)
                shutil.rmtree(directory)
                if directory.parent != root:
                    directory.parent.rmdir()

    def _select_session(self, session: Session, directory: Path | None = None) -> None:
        """Publish a session using its held generation or a newly acquired storage lease."""
        with self._state_lock, ExitStack() as resources:
            self._check_session_selection()
            if self._temporary_directory is not None:
                if directory is None:
                    directory = resources.enter_context(self._session_directory(session.id))
                self._temporary_directory.set(directory)
            self._session = session

    def _check_session_selection(self) -> None:
        """Refuse a selection that would retarget a running conversation's services."""
        if self._selection_block_depth or (
            self._execution_depth and _SESSION_SELECTION.get() is not self
        ):
            raise SessionExecutionConflictError(self._session.id)

    @property
    def interaction(self) -> Interaction:
        """Return the service used for user input and output.

        Returns:
            Interaction: The configured interaction service.
        """
        return self._interaction

    @property
    def store(self) -> SessionStore:
        """Return the session store used for persistence.

        Returns:
            SessionStore: The configured session store.
        """
        return self._session_store

    def set_async_blocking_runner(
        self,
        runner: Callable[..., Awaitable[Any]],
    ) -> None:
        """Use an application-owned worker for synchronous async-path persistence.

        Args:
            runner (Callable[..., Awaitable[Any]]): Bounded worker that invokes a synchronous
                function and returns its result.

        Returns:
            None: The async persistence worker is configured.
        """
        self._async_blocking_runner = runner

    @property
    def messages(self) -> list[ConversationItem]:
        """Return the current conversation history.

        Returns:
            list[ConversationItem]: Items accumulated during the conversation.
        """
        return self._session.messages

    @contextmanager
    def observe_responses(
        self,
        observer: Callable[[ResponsePersisted], None],
    ) -> Generator[None, None, None]:
        """Observe responses while this context is active.

        Every observer receives a typed notification after a response commit succeeds. Multiple
        observers can be active at once, and each registration is removed when its context exits.

        Args:
            observer (Callable[[ResponsePersisted], None]): Non-blocking handler invoked after a
                response is persisted.

        Yields:
            None: The response observation remains active for the duration of the context.
        """
        subscription = object()
        with self._response_persisted_observers_lock:
            self._response_persisted_observers.append((subscription, observer))
        try:
            yield
        finally:
            with self._response_persisted_observers_lock:
                self._response_persisted_observers[:] = [
                    entry
                    for entry in self._response_persisted_observers
                    if entry[0] is not subscription
                ]

    def next_message_span(self) -> AbstractContextManager[TelemetryContext | None]:
        """Return the correlation span for the next submitted user message.

        Returns:
            AbstractContextManager[TelemetryContext | None]: Active span or a no-op context
                manager.
        """
        message_sequence = sum(
            isinstance(item, Message) and item.role == "user" for item in self._session.messages
        )
        return telemetry_span(
            session_id=self._session.id,
            message_sequence=message_sequence,
        )

    @contextmanager
    def execution(
        self,
        *,
        allow_session_selection: bool = False,
    ) -> Generator[Path | None, None, None]:
        """Capture session storage and hold native ownership until all nested operations settle.

        Native locks serialize cooperating harnesses; they do not revoke detached descendants'
        descriptors. Model and tool execution prevents session selection. Conversation input may
        permit selection while idle; nested operation leases capture the newly selected resource.

        Args:
            allow_session_selection (bool): Permit idle conversation commands to select a session.
                Defaults to ``False``. Nested model and tool leases still prohibit selection.

        Yields:
            Path | None: Captured working root, or ``None`` for idle selection admission.

        Raises:
            SessionExecutionConflictError: If another thread, task, or process owns the session.
            ValueError: Working storage is missing or unsafe.
            OSError: Working storage cannot be inspected or locked.
        """
        with self._execution(allow_session_selection=allow_session_selection) as directory:
            yield directory

    @contextmanager
    def interaction_execution(self, session_id: str | None = None) -> Generator[None, None, None]:
        """Retain the selected session's ownership through a blocking interaction's settlement.

        Share this manager's existing execution lease across its owner and UI tasks, or acquire
        ownership when an idle command has none. Prevent selection until all shared leases settle.

        Args:
            session_id (str | None): Request's captured session identity. Defaults to the current
                session when absent; a different identity is rejected.

        Yields:
            None: Control while the interaction retains native session ownership.

        Raises:
            ValueError: The request belongs to another session or working storage is unsafe.
            OSError: Working storage cannot be inspected or locked.
            SessionExecutionConflictError: Another manager or process owns the session.
        """
        with self._execution(session_id=session_id, share_owner=True):
            yield

    @contextmanager
    def _execution(
        self,
        *,
        allow_session_selection: bool = False,
        session_id: str | None = None,
        share_owner: bool = False,
    ) -> Generator[Path | None, None, None]:
        """Admit an execution or retain its resource for the same manager's interaction task."""
        resources = ExitStack()
        with self._state_lock:
            if session_id is not None and session_id != self._session.id:
                raise ValueError("Interaction belongs to another session.")
            key = (
                (("manager", id(self)), "selection")
                if allow_session_selection
                else (("store", str(self._session_store.temporary_root)), self._session.id)
            )
            owner = _execution_owner(self)
            with _EXECUTION_OWNERS_LOCK:
                existing = _EXECUTION_OWNERS.get(key)
                if share_owner and existing is not None and existing[0][0] == id(self):
                    owner = existing[0]
                if existing is not None and existing[0] != owner:
                    raise SessionExecutionConflictError(self._session.id)
                depth = existing[1] + 1 if existing is not None else 1
                # Reserve this session before storage work without blocking other owners.
                _EXECUTION_OWNERS[key] = (owner, depth)
            if not allow_session_selection and self._selection_block_depth == 0:
                try:
                    _ = self.temporary_directory
                    root = self._session_store.temporary_root
                    identity = sha256_digest(
                        self._session.id.encode("utf-8", errors="surrogatepass")
                    )
                    resources.enter_context(private_directory_lock(root / ".locks", identity))
                    selected = self._bound_directory(self._session.id, root, identity)
                    self._temporary_directory.set(selected)
                    self._execution_resources = resources
                except BaseException as error:
                    try:
                        resources.close()
                    finally:
                        with _EXECUTION_OWNERS_LOCK:
                            del _EXECUTION_OWNERS[key]
                    if isinstance(error, BlockingIOError):
                        raise SessionExecutionConflictError(self._session.id) from error
                    raise
            self._execution_depth += 1
            if not allow_session_selection:
                self._selection_block_depth += 1
            captured = None if allow_session_selection else Path(self.temporary_directory)
        token = _SESSION_SELECTION.set(self if allow_session_selection else None)
        try:
            yield captured
        finally:
            _SESSION_SELECTION.reset(token)
            with self._state_lock:
                self._execution_depth -= 1
                if not allow_session_selection:
                    self._selection_block_depth -= 1
                with _EXECUTION_OWNERS_LOCK:
                    current_owner, depth = _EXECUTION_OWNERS[key]
                    if depth == 1:
                        del _EXECUTION_OWNERS[key]
                    else:
                        _EXECUTION_OWNERS[key] = (current_owner, depth - 1)
                if depth == 1 and not allow_session_selection:
                    self._execution_resources.close()

    def response(
        self,
        events: Iterable[ResponseEvent],
        *,
        debug: bool = False,
        interaction: InteractionOutput | None = None,
    ) -> Response:
        """Display and collect normalized response events.

        Collection leaves session history and persistence unchanged. The event producer owns
        stream cleanup on failure.

        Args:
            events (Iterable[ResponseEvent]): Response events to display and collect.
            debug (bool): Whether to display every raw response event. Defaults to ``False``.
            interaction (InteractionOutput | None): Presentation override. Defaults to the
                configured interaction.

        Returns:
            Response: The collected answer, reasoning, tool calls, items, usage, and model.

        Raises:
            BackendResponseError: If completion is missing or any event follows completion.
        """
        output = interaction or self._interaction
        collector = ResponseCollector()
        presenter = ResponsePresenter(output)
        with output.response_context():
            for event in events:
                if debug:
                    output.debug(event)
                collector.add(event)
                presenter.present(event)
        return collector.finish()

    async def response_async(
        self,
        events: AsyncIterable[ResponseEvent],
        *,
        debug: bool = False,
        interaction: InteractionOutput | None = None,
    ) -> Response:
        """Present and collect an asynchronous normalized response stream.

        Collection does not mutate or persist the session. The event producer retains ownership
        of its stream and must close it on failure or cancellation.

        Args:
            events (AsyncIterable[ResponseEvent]): Response events in backend order.
            debug (bool): Whether to display every raw response event. Defaults to ``False``.
            interaction (InteractionOutput | None): Output override. Defaults to the configured
                interaction.

        Returns:
            Response: Authoritative final content, tool calls, items, usage, and model.

        Raises:
            BackendResponseError: If completion is missing or any event follows completion.
            asyncio.CancelledError: If the collecting task is cancelled.
        """
        output = interaction or self._interaction
        collector = ResponseCollector()
        presenter = ResponsePresenter(output)
        with output.response_context():
            async for event in events:
                if debug:
                    output.debug(event)
                collector.add(event)
                presenter.present(event)
        return collector.finish()

    def replay(self, *, interaction: Interaction | None = None) -> None:
        """Display the active session timeline in its original durable order.

        Args:
            interaction (Interaction | None): Presentation override. Defaults to the configured
                interaction.
        """
        output = interaction or self._interaction
        for event in self._session.events:
            if isinstance(event, CompactionEvent):
                compaction = self._session.compactions[event.compaction_index]
                before = compaction.input_tokens_before
                after = compaction.input_tokens_after
                if before is not None and after is not None and before != after:
                    output.info(f"Compacted session context from {before:,} to {after:,} tokens.")
                else:
                    output.info("Compacted session context.")
                continue
            if isinstance(event, PermissionEvent):
                if event.result.prompted:
                    decision = event.result.decision.value
                    if event.result.approval_choice is not None:
                        decision += f" ({event.result.approval_choice.value})"
                    output.permission(
                        event.result.prompt or "Permission requested.",
                        decision,
                    )
                continue
            if isinstance(event, RunCompletedEvent):
                output.run_metrics(event.metrics)
                continue
            if isinstance(
                event,
                (
                    ToolExecutionStartedEvent,
                    ToolExecutionCompletedEvent,
                    InputQueuedEvent,
                    InputPriorityChangedEvent,
                    InputDeliveredEvent,
                    InputConsumedEvent,
                    InputRejectedEvent,
                    InputRecalledEvent,
                    InteractionRequestedEvent,
                    InteractionSettledEvent,
                    RunFailedEvent,
                    RunRetryStartedEvent,
                    RunStartedEvent,
                ),
            ):
                continue
            item = self._session.messages[event.item_index]
            if isinstance(item, Message):
                display = output.user if item.role == "user" else output.answer
                display(item.content)
            elif isinstance(item, Reasoning):
                output.reasoning(item.summary or item.content)
            elif isinstance(item, ToolCall):
                output.tool_call(item.name, item.arguments)

    @property
    def assignment(self) -> ModelAssignment | None:
        """Return the durable model assignment for the active session.

        Returns:
            ModelAssignment | None: Session assignment, or ``None`` when unknown.
        """
        return self._session.assignment

    @assignment.setter
    def assignment(self, value: ModelAssignment | None) -> None:
        """Set the durable model assignment for the active session.

        Args:
            value (ModelAssignment | None): Assignment to persist, or ``None`` to clear it.
        """
        self._session.assignment = value

    @property
    def model_context(self) -> list[ModelContextItem]:
        """Return hydrated context bounded by the latest compaction checkpoint.

        Returns:
            list[ModelContextItem]: Replacement context and subsequent history with verified
                reference payloads materialized once per immutable version.

        Raises:
            ValueError: If referenced content is missing, corrupt, or inconsistent with its
                occurrence metadata.
        """
        return self._hydrate_model_context(self._session.model_context(), self._session_store)

    async def model_context_async(self) -> list[ModelContextItem]:
        """Return hydrated context after loading immutable references in a bounded worker.

        Returns:
            list[ModelContextItem]: Detached, hydrated context bounded by the latest compaction
                checkpoint.

        Raises:
            ValueError: If referenced content is missing, corrupt, or inconsistent with its
                occurrence metadata.
            asyncio.CancelledError: If reference loading is cancelled after it settles.
        """
        context = deepcopy(self._session.model_context())
        return await self._run_blocking_async(
            self._hydrate_model_context, context, self._session_store
        )

    @staticmethod
    def _hydrate_model_context(
        context: list[ModelContextItem],
        session_store: SessionStore,
    ) -> list[ModelContextItem]:
        """Materialize and validate immutable reference payloads for a detached context."""
        if not any(isinstance(item, Message) and item.context for item in context):
            return context
        supplied_bytes = {}
        hydrated = []
        for item in context:
            if not isinstance(item, Message) or not item.context:
                hydrated.append(item)
                continue
            references = []
            for reference in item.context:
                if reference.handle is None:
                    raise ValueError(
                        f"Referenced snapshot '{reference.path}' has no artifact handle."
                    )
                if reference.version is None:
                    raise ValueError(
                        f"Referenced snapshot '{reference.path}' has no artifact identity."
                    )
                try:
                    validate_content_handle(reference.handle)
                except ValueError as error:
                    raise ValueError(
                        f"Referenced snapshot '{reference.path}' has an invalid artifact handle."
                    ) from error
                stored = session_store.load_reference(reference.version)
                if stored is None:
                    raise ValueError(f"Referenced snapshot '{reference.path}' is unavailable.")
                digest = content_digest(stored)
                content = stored
                if digest != reference.version or reference.size_bytes != len(content):
                    raise ValueError(
                        f"Referenced snapshot '{reference.path}' failed integrity validation."
                    )
                previously_supplied = supplied_bytes.get(digest, 0)
                payload_start = min(previously_supplied, reference.included_bytes)
                reused = reference.included_bytes > 0 and (
                    previously_supplied >= reference.included_bytes
                )
                supplied_bytes[digest] = max(previously_supplied, reference.included_bytes)
                included = content[payload_start : reference.included_bytes]
                try:
                    hydrated_content = included.decode("utf-8")
                except UnicodeDecodeError as error:
                    if reference.media_type and not reference.media_type.startswith("text/"):
                        hydrated_content = included
                    else:
                        raise ValueError(
                            f"Referenced snapshot '{reference.path}' has an invalid UTF-8 boundary."
                        ) from error
                references.append(
                    reference.model_copy(
                        update={
                            "content": hydrated_content,
                            "payload_start_bytes": payload_start,
                            "reused": reused,
                        }
                    )
                )
            hydrated.append(item.model_copy(update={"context": tuple(references)}))
        return hydrated

    @property
    def recovery_state(self) -> SessionRecoveryState | None:
        """Return the active session's required recovery boundary.

        Returns:
            SessionRecoveryState | None: Pending recovery state, or ``None`` when the latest run
                ended cleanly.
        """
        return self._session.recovery_state()

    @property
    def recovery_pending(self) -> bool:
        """Return whether the active session has an unresolved durable run boundary.

        Returns:
            bool: Whether recovery must settle before starting a new agent run.
        """
        return self.recovery_state is not None

    @property
    def retry_pending(self) -> bool:
        """Return whether the active durable run has a failed event awaiting retry.

        Returns:
            bool: Whether the latest run-state event permits ``/retry``.
        """
        return self._session.retry_pending

    @property
    def continuable(self) -> bool:
        """Return whether a durable run boundary can be resumed with ``/continue``.

        Returns:
            bool: Whether recovery is pending without a failed event requiring retry.
        """
        return self.recovery_pending and not self.retry_pending

    def record_run_failure(self) -> None:
        """Persist a recoverable failure marker for the current run.

        Raises:
            RuntimeError: If there is no active run or it already has a pending failure.
        """
        self._commit(lambda session: session.record_run_failure())

    async def record_run_failure_async(self) -> None:
        """Persist a recoverable run failure through the async session writer.

        Raises:
            RuntimeError: If there is no active run or it already has a pending failure.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        await self._commit_async(lambda session: session.record_run_failure())

    def begin_run_retry(self) -> None:
        """Persist explicit authorization to retry the current failed run.

        Raises:
            RuntimeError: If the active run has no recorded failure to retry.
        """
        self._commit(lambda session: session.begin_run_retry())

    async def begin_run_retry_async(self) -> None:
        """Persist explicit retry authorization through the async session writer.

        Raises:
            RuntimeError: If the active run has no recorded failure to retry.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        await self._commit_async(lambda session: session.begin_run_retry())

    def begin_run(self) -> RunStartedEvent:
        """Persist a unique run identity before invoking model or tool work.

        Returns:
            RunStartedEvent: Durable identity and start time for the active run.

        Raises:
            RuntimeError: If another durable run is still active.
        """
        if self._session.active_run_id is not None:
            raise RuntimeError("A durable session run is already active.")
        event = RunStartedEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            run_id=str(new_uuid()),
            started_at=utc_now(),
        )
        self._persist_event(event)
        return event

    async def begin_run_async(self) -> RunStartedEvent:
        """Persist a unique run identity before async model or tool work.

        Returns:
            RunStartedEvent: Durable identity and start time for the active run.

        Raises:
            RuntimeError: If another durable run is still active.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        if self._session.active_run_id is not None:
            raise RuntimeError("A durable session run is already active.")
        event = RunStartedEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            run_id=str(new_uuid()),
            started_at=utc_now(),
        )
        await self._persist_event_async(event)
        return event

    def queue_input(
        self,
        content: str,
        *,
        priority: int,
        bypass_commands: bool = False,
    ) -> InputQueuedEvent:
        """Durably accept one input before the controller acknowledges it.

        Args:
            content (str): User-provided message text.
            priority (int): Mailbox priority used to restore accepted inputs.
            bypass_commands (bool): Whether to route the text directly as message content.
                Defaults to ``False``.

        Returns:
            InputQueuedEvent: Persisted identity and ordering metadata for the input.

        Raises:
            TypeError: If content is not text.
        """
        if not isinstance(content, str):
            raise TypeError("Input content must be text.")
        event = self._new_input_event(content, priority, bypass_commands)
        self._persist_event(event)
        return event

    async def queue_input_async(
        self,
        content: str,
        *,
        priority: int,
        bypass_commands: bool = False,
    ) -> InputQueuedEvent:
        """Durably accept one input through the async session writer.

        Args:
            content (str): User-provided message text.
            priority (int): Mailbox priority used to restore accepted inputs.
            bypass_commands (bool): Whether to route the text directly as message content.
                Defaults to ``False``.

        Returns:
            InputQueuedEvent: Persisted identity and ordering metadata for the input.

        Raises:
            TypeError: If content is not text.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        if not isinstance(content, str):
            raise TypeError("Input content must be text.")
        event = self._new_input_event(content, priority, bypass_commands)
        await self._persist_event_async(event)
        return event

    def _new_input_event(
        self,
        content: str,
        priority: int,
        bypass_commands: bool,
    ) -> InputQueuedEvent:
        """Create the next input event using session-owned sequence and run state."""
        sequence = (
            max(
                (
                    event.sequence
                    for event in self._session.events
                    if isinstance(event, InputQueuedEvent)
                ),
                default=-1,
            )
            + 1
        )
        return InputQueuedEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            input_id=str(new_uuid()),
            session_id=self._session.id,
            content=content,
            priority=priority,
            sequence=sequence,
            bypass_commands=bypass_commands,
            run_id=self._session.active_run_id,
        )

    def mark_input_delivered(self, input_id: str) -> None:
        """Persist that an accepted input reached its conversation owner.

        Args:
            input_id (str): Identity of a pending accepted input.

        Raises:
            ValueError: If the input is unknown or already settled.
        """
        self._commit(lambda session: session.mark_input_delivered(input_id))

    async def mark_input_delivered_async(self, input_id: str) -> None:
        """Persist input delivery through the async session writer.

        Args:
            input_id (str): Identity of a pending accepted input.

        Raises:
            ValueError: If the input is unknown or already settled.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        await self._commit_async(lambda session: session.mark_input_delivered(input_id))

    def change_input_priority(self, input_id: str, priority: int) -> None:
        """Persist a priority change for an input that has not reached its owner.

        Args:
            input_id (str): Identity of the accepted input.
            priority (int): Replacement mailbox priority.

        Returns:
            None: After the priority change is durably committed.

        Raises:
            ValueError: If the input is no longer waiting in the mailbox.
        """
        self._commit(lambda session: session.change_input_priority(input_id, priority))

    async def change_input_priority_async(self, input_id: str, priority: int) -> None:
        """Persist a priority change through the asynchronous session writer.

        Args:
            input_id (str): Identity of the accepted input.
            priority (int): Replacement mailbox priority.

        Returns:
            None: After the priority change is durably committed.

        Raises:
            ValueError: If the input is no longer waiting in the mailbox.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        await self._commit_async(lambda session: session.change_input_priority(input_id, priority))

    def recall_input(self, input_id: str) -> None:
        """Persist the recall of an input that is still waiting in the mailbox.

        Args:
            input_id (str): Identity of the accepted input to return to the editor.

        Raises:
            ValueError: If the input is no longer waiting in the mailbox.
        """
        self._commit(lambda session: session.recall_input(input_id))

    async def recall_input_async(self, input_id: str) -> None:
        """Persist a mailbox input recall through the asynchronous session writer.

        Args:
            input_id (str): Identity of the accepted input to return to the editor.

        Raises:
            ValueError: If the input is no longer waiting in the mailbox.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        await self._commit_async(lambda session: session.recall_input(input_id))

    def reject_input(self, input_id: str) -> None:
        """Persist that a previously accepted input could not be admitted or prepared.

        Args:
            input_id (str): Identity of a pending accepted input.

        Raises:
            ValueError: If the input is unknown or already settled.
        """
        self._commit(lambda session: session.reject_input(input_id))

    async def reject_input_async(self, input_id: str) -> None:
        """Persist input rejection through the async session writer.

        Args:
            input_id (str): Identity of a pending accepted input.

        Raises:
            ValueError: If the input is unknown or already settled.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        await self._commit_async(lambda session: session.reject_input(input_id))

    async def record_interaction_requested_async(
        self,
        request_id: str,
        *,
        kind: str,
        session_id: str | None,
        run_id: str | None,
        operation_id: str | None,
    ) -> None:
        """Record safe identity metadata before presenting a blocking interaction.

        Args:
            request_id (str): Unique UI request identity.
            kind (str): Prompt, confirmation, or terminal request type.
            session_id (str | None): Request owner, or the active session when absent.
            run_id (str | None): Correlated run identity.
            operation_id (str | None): Correlated tool operation identity.

        Raises:
            ValueError: If the request belongs to another session or has an invalid kind.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        event = InteractionRequestedEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            request_id=request_id,
            session_id=session_id or self._session.id,
            run_id=run_id,
            operation_id=operation_id,
            kind=kind,
        )
        await self._commit_async(lambda session: session.request_interaction(event))

    async def record_interaction_settled_async(self, request_id: str, outcome: str) -> None:
        """Close a durable UI request without retaining any answer or reusable approval.

        Args:
            request_id (str): Unique UI request identity.
            outcome (str): Answered, revoked, or failed lifecycle outcome.

        Raises:
            ValueError: If the request is not pending or the outcome is invalid.
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        await self._commit_async(lambda session: session.settle_interaction(request_id, outcome))

    @property
    def model(self) -> str | None:
        """Return the model explicitly selected for requests.

        Returns:
            str | None: The selected model, or ``None`` when the backend default is used.
        """
        return self._session.model

    @model.setter
    def model(self, value: str | None) -> None:
        """Set the model explicitly selected for requests.

        Args:
            value (str | None): The model to use, or ``None`` to use the backend default.
        """
        self._session.model = value

    @property
    def tokens(self) -> int:
        """Return the number of tokens used in the current session.

        Returns:
            int: The number of tokens used in the current session.
        """
        return self._session.tokens

    @property
    def context_window(self) -> int | None:
        """Return the persisted context-window size for the selected model.

        Returns:
            int | None: Context-window size, or ``None`` when unknown.
        """
        return self._session.context_window

    @context_window.setter
    def context_window(self, value: int | None) -> None:
        """Set the context-window size associated with the selected model.

        Args:
            value (int | None): Positive context-window size, or ``None`` when unknown.

        Raises:
            ValueError: If a non-positive size is provided.
        """
        if value is not None and value <= 0:
            raise ValueError("Context window must be positive.")
        self._session.context_window = value

    def add_compaction(
        self,
        result: CompactionResult,
        *,
        model: str,
        instructions: str | None,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
        references: Iterable[CapturedInstruction] = (),
    ) -> None:
        """Persist one compaction checkpoint and its exact instruction state.

        Args:
            result (CompactionResult): Replacement context and reported token usage.
            model (str): Model used to compact the active context.
            instructions (str | None): Complete instructions supplied to the compactor.
            working_directory (str): Effective instruction-discovery directory.
            active_skills (Iterable[tuple[str, str]]): Active skill identities.
            references (Iterable[CapturedInstruction]): Immutable instruction-file provenance.

        Returns:
            None: The replacement context and instruction state are durably committed.

        Raises:
            ValueError: If the checkpoint does not advance or has invalid context.
        """
        identities = tuple(active_skills)
        instruction_references = tuple(references)
        provider = result.items[0].provider if result.items else ""
        compaction = Compaction(
            id=str(new_uuid()),
            boundary=len(self._session.messages),
            created_at=utc_now(),
            provider=provider,
            model=model,
            context=result.items,
            instructions=InstructionSnapshot(
                working_directory=working_directory,
                content=instructions,
                digest=sha256_digest(instructions or ""),
                active_skills=identities,
                references=instruction_references,
            ),
            input_tokens_before=self._session.tokens,
            input_tokens_after=result.context_tokens,
        )

        def apply(session: Session) -> None:
            session.add_compaction(compaction)
            if result.context_tokens is not None:
                session.tokens = result.context_tokens
            session.update_instruction_state(working_directory, identities)

        self._commit(apply)

    async def add_compaction_async(
        self,
        result: CompactionResult,
        *,
        model: str,
        instructions: str | None,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
        references: Iterable[CapturedInstruction] = (),
    ) -> None:
        """Persist one compaction checkpoint through the async session writer.

        Args:
            result (CompactionResult): Replacement context and reported token usage.
            model (str): Model used to compact the active context.
            instructions (str | None): Complete instructions supplied to the compactor.
            working_directory (str): Effective instruction-discovery directory.
            active_skills (Iterable[tuple[str, str]]): Active skill identities.
            references (Iterable[CapturedInstruction]): Immutable instruction-file provenance.

        Raises:
            ValueError: If the checkpoint does not advance or has invalid context.
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        identities = tuple(active_skills)
        instruction_references = tuple(references)
        provider = result.items[0].provider if result.items else ""
        compaction = Compaction(
            id=str(new_uuid()),
            boundary=len(self._session.messages),
            created_at=utc_now(),
            provider=provider,
            model=model,
            context=result.items,
            instructions=InstructionSnapshot(
                working_directory=working_directory,
                content=instructions,
                digest=sha256_digest(instructions or ""),
                active_skills=identities,
                references=instruction_references,
            ),
            input_tokens_before=self._session.tokens,
            input_tokens_after=result.context_tokens,
        )

        def apply(session: Session) -> None:
            session.add_compaction(compaction)
            if result.context_tokens is not None:
                session.tokens = result.context_tokens
            session.update_instruction_state(working_directory, identities)

        await self._commit_async(apply)

    def load_session(self, session: Session | str) -> None:
        """Load a session from the store.

        Args:
            session (Session | str): The session to load, or its identifier.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If its persisted format or bound working storage is invalid or unsafe.
            OSError: Bound working storage cannot be created or inspected.
            SessionWorkspaceMismatchError: If the session belongs to another workspace.
            SessionExecutionConflictError: This manager is executing or another harness owns
                the target session's working storage during selection or interaction recovery.
        """
        with self._state_lock, ExitStack() as resources:
            self._check_session_selection()
            if load_from_store := isinstance(session, str):
                session = self._session_store.load(session)
            if not isinstance(session, Session):
                raise ValueError("Invalid session type.")  # noqa: TRY004 - public API contract.
            self._validate_session_workspace(session)
            directory = None
            if session.pending_interactions:
                directory = resources.enter_context(self._session_directory(session.id))
                if load_from_store:
                    session = self._session_store.load(session.id)
                    self._validate_session_workspace(session)
            if session.workspace_id is None:
                session.workspace_id = self._workspace_id
            cache_restorations = []
            for message in session.messages:
                if isinstance(message, Message):
                    for reference in message.context:
                        if reference.handle is None or reference.version is None:
                            raise ValueError(
                                f"Referenced snapshot '{reference.path}' has an incomplete "
                                "artifact identity."
                            )
                        stored = self._session_store.load_reference(reference.version)
                        if (
                            stored is None
                            and not load_from_store
                            and (reference.content or reference.size_bytes == 0)
                        ):
                            content = get_binary(reference.content)
                            if (
                                content_digest(content) != reference.version
                                or len(content) != reference.size_bytes
                            ):
                                raise ValueError(
                                    f"Referenced snapshot '{reference.path}' failed integrity "
                                    "validation."
                                )
                            self._session_store.store_reference(
                                content,
                                handle=reference.handle,
                                digest=reference.version,
                            )
                            stored = content
                        if stored is None:
                            raise ValueError(
                                f"Referenced snapshot '{reference.path}' is unavailable."
                            )
                        digest = content_digest(stored)
                        if digest != reference.version or len(stored) != reference.size_bytes:
                            raise ValueError(
                                f"Referenced snapshot '{reference.path}' "
                                "failed integrity validation."
                            )
                        cached = cached_path(reference.handle)
                        cache_valid = cached is not None and cached[0].is_file()
                        if cache_valid:
                            try:
                                cache_valid = cached[0].read_bytes() == stored
                            except OSError:
                                cache_valid = False
                        if not cache_valid:
                            cache_restorations.append((reference, stored))
            for reference, stored in cache_restorations:
                store_content(
                    stored,
                    f"mentioned {reference.kind} {reference.path}",
                    handle=reference.handle,
                )
            for message in session.messages:
                if isinstance(message, ToolResult):
                    for artifact in message.artifacts:
                        register_cached_metadata(
                            artifact.handle,
                            artifact.source,
                            artifact.reloadable,
                        )
            if session.pending_interactions:
                candidate = deepcopy(session)
                for request in candidate.pending_interactions:
                    candidate.settle_interaction(request.request_id, "revoked")
                if load_from_store or session.revision > 0:
                    self._save_snapshot(self._session_store, self._persistence_lock, candidate)
                for session_field in fields(Session):
                    setattr(
                        session,
                        session_field.name,
                        deepcopy(getattr(candidate, session_field.name)),
                    )
            self._select_session(session, directory)

    def load_persisted_session(self, session_id: str) -> Session:
        """Read a persisted session without changing the active session.

        Args:
            session_id (str): Identifier of the persisted session to load.

        Returns:
            Session: Reconstructed persisted session after workspace validation.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            SessionWorkspaceMismatchError: If the session belongs to another workspace.
            ValueError: If the persisted format is invalid or unsupported.
        """
        session = self._session_store.load(session_id)
        self._validate_session_workspace(session)
        return session

    def _validate_session_workspace(self, session: Session) -> None:
        """Reject a session owned by a different workspace."""
        if self._workspace_id is not None and session.workspace_id not in {
            None,
            self._workspace_id,
        }:
            raise SessionWorkspaceMismatchError(
                f"Session '{session.id}' belongs to workspace '{session.workspace_id}', "
                f"not '{self._workspace_id}'."
            )

    def new_session(self) -> None:
        """Replace the idle session and select its isolated working directory when bound.

        Raises:
            SessionExecutionConflictError: If this manager is currently executing a session.
            ValueError: Bound working storage is aliased, not private, or owned by another user.
            OSError: The working directory cannot be created or inspected.
        """
        with self._state_lock:
            self._select_session(
                Session(model=self._session.model, workspace_id=self._workspace_id)
            )

    def rename_session(self, name: str) -> None:
        """Assign and persist a user-controlled name to the active session.

        Args:
            name (str): New non-empty session name.

        Raises:
            ValueError: If the name is empty after normalization.
        """
        self._commit(lambda session: session.rename(name))

    def rename_persisted_session(self, session_id: str, name: str) -> None:
        """Assign and persist a user-controlled name to one stored session.

        Args:
            session_id (str): Identifier of the persisted session to rename.
            name (str): New non-empty session name.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            SessionRevisionConflictError: If the stored session changes before it is saved.
            SessionWorkspaceMismatchError: If the session belongs to another workspace.
            ValueError: If the name is empty after normalization.
        """
        if session_id == self._session.id:
            self.rename_session(name)
            return
        session = self._session_store.load(session_id)
        session.rename(name)
        self._session_store.save(session)

    def generate_session_name(
        self,
        generator: SessionNameGenerator,
        *,
        context: SessionNameContext | None = None,
    ) -> None:
        """Generate, persist, and report a session name from a captured exchange.

        Args:
            generator (SessionNameGenerator): Auxiliary title generation service.
            context (SessionNameContext | None): Captured automatic-name exchange to name, or
                ``None`` to capture the active session and preserve explicit naming behavior.
        """
        self._interaction.info("Generating a session name...")
        try:
            require_initial_name = context is not None
            context = context or self._capture_session_name_context(require_initial_name=False)
            if context is None:
                self._interaction.info("No completed exchange found to generate a session name.")
                return
            if not self._is_current_name_context(
                context,
                require_initial_name=require_initial_name,
            ):
                return
            name = generator.generate(
                context.user_message,
                context.assistant_message,
                context.model,
            )
            if name:
                self._commit(
                    lambda session: self._rename_generated_session(
                        session,
                        context,
                        name,
                        require_initial_name=require_initial_name,
                    )
                )
            if self._session.id == context.session_id:
                self._interaction.info(f"Session name: {self._session.name}")
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            problem = Problem.from_exception(
                error,
                code="session.name_generation_failed",
                title="Could not generate session name",
                detail="Could not generate the session name.",
                severity="warning",
                retryable=True,
                operation="generate_session_name",
            )
            log_problem(_LOGGER, problem, error)
            self._interaction.report(problem)

    async def generate_session_name_async(
        self,
        generator: SessionNameGenerator,
        *,
        context: SessionNameContext | None = None,
    ) -> None:
        """Generate and persist a session name without blocking asynchronous model I/O.

        Generators with ``generate_async`` use their native asynchronous implementation. Existing
        synchronous generators remain supported for compatibility. A supplied context preserves
        the session identity and messages captured before this task was scheduled.

        Args:
            generator (SessionNameGenerator): Auxiliary title generation service.
            context (SessionNameContext | None): Captured automatic-name exchange to name, or
                ``None`` to capture the active session and preserve explicit naming behavior.

        Returns:
            None: The generated name is persisted when a usable exchange still matches its
                captured session and name state.

        Raises:
            asyncio.CancelledError: Naming is cancelled after any owned worker settles.
        """
        self._interaction.info("Generating a session name...")
        try:
            require_initial_name = context is not None
            context = context or self._capture_session_name_context(require_initial_name=False)
            if context is None:
                self._interaction.info("No completed exchange found to generate a session name.")
                return
            if not self._is_current_name_context(
                context,
                require_initial_name=require_initial_name,
            ):
                return
            async_generator = getattr(generator, "generate_async", None)
            if callable(async_generator):
                name = await async_generator(
                    context.user_message,
                    context.assistant_message,
                    context.model,
                )
            else:
                name = await self._run_blocking_async(
                    generator.generate,
                    context.user_message,
                    context.assistant_message,
                    context.model,
                )
            if name:
                await self._commit_async(
                    lambda session: self._rename_generated_session(
                        session,
                        context,
                        name,
                        require_initial_name=require_initial_name,
                    )
                )
            if self._session.id == context.session_id:
                self._interaction.info(f"Session name: {self._session.name}")
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            problem = Problem.from_exception(
                error,
                code="session.name_generation_failed",
                title="Could not generate session name",
                detail="Could not generate the session name.",
                severity="warning",
                retryable=True,
                operation="generate_session_name",
            )
            log_problem(_LOGGER, problem, error)
            self._interaction.report(problem)

    def session_name_context(self) -> SessionNameContext | None:
        """Capture the first usable exchange for an automatically named session.

        Returns:
            SessionNameContext | None: Immutable title input when the active session still has a
                provisional name and contains a user message plus non-empty assistant text.
        """
        return self._capture_session_name_context(require_initial_name=True)

    def _capture_session_name_context(
        self,
        *,
        require_initial_name: bool,
    ) -> SessionNameContext | None:
        """Capture the first user message and non-empty assistant response."""
        session = self._session
        if require_initial_name and not session.has_initial_name():
            return None
        user_message = next(
            (
                message.content
                for message in session.messages
                if isinstance(message, Message) and message.role == "user"
            ),
            "",
        )
        assistant_message = next(
            (
                message.content
                for message in session.messages
                if isinstance(message, Message)
                and message.role == "assistant"
                and message.content.strip()
            ),
            "",
        )
        if not user_message or not assistant_message:
            return None
        return SessionNameContext(
            session_id=session.id,
            user_message=user_message,
            assistant_message=assistant_message,
            model=session.model,
        )

    def _is_current_name_context(
        self,
        context: SessionNameContext,
        *,
        require_initial_name: bool,
    ) -> bool:
        """Return whether a captured request still matches the active session and name state."""
        return self._session.id == context.session_id and (
            not require_initial_name or self._session.has_initial_name()
        )

    @staticmethod
    def _rename_generated_session(
        session: Session,
        context: SessionNameContext,
        name: str,
        *,
        require_initial_name: bool,
    ) -> None:
        """Apply a generated name while its captured session and name state still match."""
        if session.id == context.session_id and (
            not require_initial_name or session.has_initial_name()
        ):
            session.rename(name, source=SESSION_NAME_SOURCE_GENERATED)

    def add_message(self, message: ConversationItem | Response) -> None:
        """Add conversation items and persist the resulting complete session.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Returns:
            None: The conversation item is durably committed.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        session_id = self._session.id
        self._commit(lambda session: session.add_message(message))
        if isinstance(message, Response):
            self._publish_response_persisted(message, session_id=session_id)

    async def add_message_async(self, message: ConversationItem | Response) -> None:
        """Add conversation items through the asynchronous persistence boundary.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Returns:
            None: The conversation item is durably committed before returning.

        Raises:
            ValueError: If the value is not a supported conversation item.
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        await self._commit_async(
            lambda session: session.add_message(message),
            persisted_response=message if isinstance(message, Response) else None,
        )

    def _publish_response_persisted(self, response: Response, *, session_id: str) -> None:
        """Notify active observers after the response is durably committed."""
        with self._response_persisted_observers_lock:
            observers = tuple(
                observer for _subscription, observer in self._response_persisted_observers
            )
        if not observers:
            return
        event = ResponsePersisted(session_id=session_id, response=response)
        for observer in observers:
            try:
                observer(event)
            except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                problem = Problem.from_exception(
                    error,
                    code="session.response_observer_failed",
                    title="Response observer failed",
                    detail="A response observer failed after the response was persisted.",
                    severity="warning",
                    operation="notify_response_observer",
                    metadata={"session_id": event.session_id},
                )
                log_problem(_LOGGER, problem, error)
                try:
                    self._interaction.report(problem)
                except Exception as report_error:  # noqa: BLE001
                    reporting_problem = Problem.from_exception(
                        report_error,
                        code="session.response_observer_report_failed",
                        title="Could not report response observer failure",
                        detail="A response observer failed and its warning could not be reported.",
                        severity="warning",
                        operation="report_response_observer_failure",
                        metadata={"session_id": event.session_id},
                    )
                    log_problem(_LOGGER, reporting_problem, report_error)

    def add_messages(self, messages: Iterable[ConversationItem]) -> None:
        """Add messages to the conversation history.

        Args:
            messages (Iterable[ConversationItem]): Conversation items to add.

        Raises:
            ValueError: If any value is not a supported conversation item.
        """
        values = tuple(messages)
        self._commit(lambda session: session.add_messages(values))

    def add_user_message(
        self,
        content: str,
        context: Iterable[ContextReference] = (),
        *,
        input_id: str | None = None,
    ) -> None:
        """Construct and persist one complete user message.

        Args:
            content (str): Submitted user-message text.
            context (Iterable[ContextReference]): Resolved context snapshots attached to the
                message. Defaults to no explicit context.
            input_id (str | None): Durable accepted-input identity to consume with this message.
                Defaults to ``None`` for direct or legacy input.

        Returns:
            None: The user message is durably committed before model work begins.

        Raises:
            ValueError: If occurrence metadata is incomplete or inconsistent with its content.
        """
        message = self._prepare_user_message(content, context)

        def apply(session: Session) -> None:
            if session.name is None:
                session.rename(
                    initial_session_name(content),
                    source=SESSION_NAME_SOURCE_INITIAL,
                )
            item_index = len(session.messages)
            session.add_message(message)
            if input_id is not None:
                session.consume_input(input_id, item_index)

        self._commit(apply)

    async def add_user_message_async(
        self,
        content: str,
        context: Iterable[ContextReference] = (),
        *,
        input_id: str | None = None,
    ) -> None:
        """Persist a user message before async model work begins.

        Args:
            content (str): Submitted user-message text.
            context (Iterable[ContextReference]): Resolved context snapshots attached to the
                message. Defaults to no explicit context.
            input_id (str | None): Durable accepted-input identity to consume with this message.
                Defaults to ``None`` for direct or legacy input.

        Raises:
            ValueError: If occurrence metadata is incomplete or inconsistent with its content.
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        message = self._prepare_user_message(content, context)

        def apply(session: Session) -> None:
            if session.name is None:
                session.rename(
                    initial_session_name(content),
                    source=SESSION_NAME_SOURCE_INITIAL,
                )
            item_index = len(session.messages)
            session.add_message(message)
            if input_id is not None:
                session.consume_input(input_id, item_index)

        await self._commit_async(apply)

    def _prepare_user_message(
        self,
        content: str,
        context: Iterable[ContextReference],
    ) -> Message:
        """Validate context references and construct one detached user message."""
        captured = []
        for reference in context:
            if (reference.handle is None) != (reference.version is None):
                raise ValueError(
                    "Reference handle and artifact identity must be provided together."
                )
            if reference.handle is None:
                complete = get_binary(reference.content)
                handle, version = content_identity(complete)
                reference = reference.model_copy(update={"handle": handle, "version": version})
            captured.append(reference)
        captured = tuple(
            reference.model_copy(
                update={
                    "next_cursor": (
                        encode_content_cursor(reference.handle, reference.included_bytes)
                        if reference.truncated
                        else None
                    )
                }
            )
            for reference in captured
        )
        handle_versions = {
            reference.handle: reference.version
            for message in self._session.messages
            if isinstance(message, Message)
            for reference in message.context
        }
        for reference in captured:
            try:
                validate_content_handle(reference.handle)
            except ValueError as error:
                raise ValueError(
                    f"Referenced snapshot '{reference.path}' has an invalid artifact handle."
                ) from error
            existing = handle_versions.setdefault(reference.handle, reference.version)
            if existing != reference.version:
                raise ValueError(
                    f"Referenced snapshot '{reference.path}' reuses an artifact handle for "
                    "different content."
                )
            complete = get_binary(reference.content)
            if len(complete) != reference.size_bytes:
                raise ValueError(
                    f"Referenced snapshot '{reference.path}' size does not match its content."
                )
            if not 0 <= reference.included_bytes <= reference.size_bytes:
                raise ValueError(
                    f"Referenced snapshot '{reference.path}' has invalid included bytes."
                )
            if reference.truncated != (reference.included_bytes < reference.size_bytes):
                raise ValueError(
                    f"Referenced snapshot '{reference.path}' has inconsistent truncation metadata."
                )
            if content_digest(complete) != reference.version:
                raise ValueError(
                    f"Referenced snapshot '{reference.path}' failed integrity validation."
                )
        for reference in captured:
            store_content(
                reference.content,
                f"mentioned {reference.kind} {reference.path}",
                handle=reference.handle,
            )
        message = Message(role="user", content=content, context=captured)

        return message

    def add_tool_call(
        self,
        call_id: str,
        output: str,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
        *,
        succeeded: bool = False,
        duration_seconds: float = 0,
    ) -> None:
        """Add a tool result and its instruction state to the session.

        Args:
            call_id (str): Identifier used to associate the call with its result.
            output (str): The content of the tool result.
            working_directory (str): Effective instruction directory after the tool call.
            active_skills (Iterable[tuple[str, str]]): Active skill names and canonical locations
                after the tool call.
            succeeded (bool): Whether the serialized tool result reports success.
            duration_seconds (float): Tool-function execution duration in seconds.
        """
        identities = tuple(active_skills)
        output, handle = bound_tool_result(output, f"tool result {call_id}")
        if handle is not None:
            self._interaction.info(
                f"Tool result '{call_id}' exceeded the context limit and was cached as '{handle}'."
            )
        result = ToolResult(
            call_id=call_id,
            output=output,
            artifacts=self._content_artifacts(output),
        )

        def apply(session: Session) -> None:
            session.update_instruction_state(working_directory, identities)
            session.add_message(result)
            if any(
                isinstance(item, ToolCall) and item.call_id == call_id for item in session.messages
            ):
                session.events.append(
                    ToolExecutionCompletedEvent(
                        id=str(new_uuid()),
                        created_at=utc_now(),
                        call_id=call_id,
                        succeeded=succeeded,
                        duration_seconds=duration_seconds,
                    )
                )

        self._commit(apply)

    async def add_tool_call_async(
        self,
        call_id: str,
        output: str,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
        *,
        succeeded: bool = False,
        duration_seconds: float = 0,
    ) -> None:
        """Persist a tool result and its instruction state asynchronously.

        Args:
            call_id (str): Identifier used to associate the call with its result.
            output (str): The content of the tool result.
            working_directory (str): Effective instruction directory after the tool call.
            active_skills (Iterable[tuple[str, str]]): Active skill names and canonical locations
                after the tool call.
            succeeded (bool): Whether the serialized tool result reports success.
            duration_seconds (float): Tool-function execution duration in seconds.

        Returns:
            None: The tool result and instruction state are durably committed.

        Raises:
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        identities = tuple(active_skills)
        output, handle = bound_tool_result(output, f"tool result {call_id}")
        if handle is not None:
            self._interaction.info(
                f"Tool result '{call_id}' exceeded the context limit and was cached as '{handle}'."
            )
        result = ToolResult(
            call_id=call_id,
            output=output,
            artifacts=self._content_artifacts(output),
        )

        def apply(session: Session) -> None:
            session.update_instruction_state(working_directory, identities)
            session.add_message(result)
            if any(
                isinstance(item, ToolCall) and item.call_id == call_id for item in session.messages
            ):
                session.events.append(
                    ToolExecutionCompletedEvent(
                        id=str(new_uuid()),
                        created_at=utc_now(),
                        call_id=call_id,
                        succeeded=succeeded,
                        duration_seconds=duration_seconds,
                    )
                )

        await self._commit_async(apply)

    def add_tool_call_event(self, call_id: str) -> None:
        """Persist one model tool call at its live presentation position.

        Args:
            call_id (str): Identifier of the canonical tool call to place.

        Returns:
            None: The presentation boundary is durably committed.

        Raises:
            ValueError: If the call is unknown or already recorded.
        """
        self._commit(lambda session: session.add_tool_call_event(call_id))

    async def add_tool_call_event_async(self, call_id: str) -> None:
        """Persist one model tool call at its live presentation position asynchronously.

        Args:
            call_id (str): Identifier of the canonical tool call to place.

        Raises:
            ValueError: If the call is unknown or already recorded.
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        await self._commit_async(lambda session: session.add_tool_call_event(call_id))

    def record_tool_execution_started(self, call_id: str) -> None:
        """Persist that one tool invocation is about to cross the execution boundary.

        Args:
            call_id (str): Identifier of the model tool request being executed.

        Returns:
            None: The pre-execution intent is durably committed.

        Raises:
            ValueError: If the call is not present in canonical session history.
        """
        if not any(
            isinstance(item, ToolCall) and item.call_id == call_id
            for item in self._session.messages
        ):
            raise ValueError(f"Unknown tool call '{call_id}'.")
        self._persist_event(
            ToolExecutionStartedEvent(id=str(new_uuid()), created_at=utc_now(), call_id=call_id)
        )

    async def record_tool_execution_started_async(self, call_id: str) -> None:
        """Persist tool execution intent before an async tool crosses its effect boundary.

        Args:
            call_id (str): Identifier of the model tool request being executed.

        Raises:
            ValueError: If the call is not present in canonical session history.
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        if not any(
            isinstance(item, ToolCall) and item.call_id == call_id
            for item in self._session.messages
        ):
            raise ValueError(f"Unknown tool call '{call_id}'.")
        await self._persist_event_async(
            ToolExecutionStartedEvent(id=str(new_uuid()), created_at=utc_now(), call_id=call_id)
        )

    @staticmethod
    def _content_artifacts(output: str) -> tuple[ContentArtifact, ...]:
        """Return registered artifact metadata referenced by one serialized result."""
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return ()
        if not isinstance(payload, dict) or not isinstance(payload.get("handle"), str):
            return ()
        handle = payload["handle"]
        metadata = cached_metadata(handle)
        if metadata is None:
            return ()
        return (
            ContentArtifact(
                handle=handle,
                source=metadata["source"],
                reloadable=metadata["reloadable"],
            ),
        )

    def update_instruction_state(
        self,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
    ) -> None:
        """Update the instruction state associated with the session.

        Args:
            working_directory (str): Effective instruction directory.
            active_skills (Iterable[tuple[str, str]]): Active skill names and canonical locations.
        """
        self._session.update_instruction_state(working_directory, active_skills)

    def add_response(self, response: Response) -> None:
        """Add a response to the session.

        Args:
            response (Response): The response to add.

        Returns:
            None: The response is durably committed before observers are notified.
        """
        self.add_message(response)

    async def add_response_async(self, response: Response) -> None:
        """Persist one complete backend response through the async session writer.

        Args:
            response (Response): The response to add.

        Returns:
            None: The response is durably committed before observers are notified.

        Raises:
            ValueError: If the response contains an unsupported conversation item.
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        await self.add_message_async(response)

    def record_authorization(self, result: AuthorizationResult) -> None:
        """Persist one atomic authorization decision in the session timeline.

        Args:
            result (AuthorizationResult): Policy, approval, and effective outcome.
        """
        event = PermissionEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            result=result,
        )
        self._persist_event(event)

    async def record_authorization_async(self, result: AuthorizationResult) -> None:
        """Persist one authorization decision without blocking the async owner.

        Args:
            result (AuthorizationResult): Policy, approval, and effective outcome.

        Returns:
            None: The authorization decision is durably committed to the timeline.

        Raises:
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        event = PermissionEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            result=result,
        )
        await self._persist_event_async(event)

    def record_run(
        self,
        stop_reason: AgentRunStopReason,
        started_at: datetime,
        metrics: RunMetrics,
    ) -> RunCompletedEvent:
        """Construct and persist one completed-run event.

        Args:
            stop_reason (AgentRunStopReason): Reason the agent run returned control.
            started_at (datetime): UTC time at which the run began.
            metrics (RunMetrics): Completed run statistics.

        Returns:
            RunCompletedEvent: Persisted event suitable for immediate presentation.
        """
        event = RunCompletedEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            stop_reason=stop_reason,
            started_at=started_at,
            metrics=metrics,
            run_id=self._session.active_run_id,
        )
        self._persist_event(event)
        return event

    async def record_run_async(
        self,
        stop_reason: AgentRunStopReason,
        started_at: datetime,
        metrics: RunMetrics,
    ) -> RunCompletedEvent:
        """Persist one completed-run event through the async session writer.

        Args:
            stop_reason (AgentRunStopReason): Reason the agent run returned control.
            started_at (datetime): UTC time at which the run began.
            metrics (RunMetrics): Completed run statistics.

        Returns:
            RunCompletedEvent: Persisted event suitable for immediate presentation.

        Raises:
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        event = RunCompletedEvent(
            id=str(new_uuid()),
            created_at=utc_now(),
            stop_reason=stop_reason,
            started_at=started_at,
            metrics=metrics,
            run_id=self._session.active_run_id,
        )
        await self._persist_event_async(event)
        return event

    def _persist_event(self, event: SessionEventModel) -> None:
        """Append and persist one standalone timeline event."""
        self._commit(lambda session: session.events.append(event))

    async def _persist_event_async(self, event: SessionEventModel) -> None:
        """Append and persist one standalone timeline event asynchronously."""
        await self._commit_async(lambda session: session.events.append(event))

    def _commit(self, mutation: Callable[[Session], None]) -> None:
        """Apply and persist one session mutation atomically in memory."""
        with self._commit_lock:
            current = self._session
            previous = deepcopy(current)
            try:
                mutation(current)
                telemetry_activity("session.persistence.started", component="session_manager")
                self._save_snapshot(self._session_store, self._persistence_lock, current)
                telemetry_activity(
                    "session.persistence.completed",
                    component="session_manager",
                )
            except Exception:
                telemetry_error(
                    "session.persistence.failed",
                    error_type="session.persistence_failed",
                    component="session_manager",
                )
                for session_field in fields(Session):
                    setattr(
                        current,
                        session_field.name,
                        deepcopy(getattr(previous, session_field.name)),
                    )
                raise
            if self._session is not current:
                raise RuntimeError("The active session changed while its snapshot was being saved.")

    async def _commit_async(
        self,
        mutation: Callable[[Session], None],
        *,
        persisted_response: Response | None = None,
    ) -> None:
        """Persist a detached mutation and notify response observers after a successful commit."""
        cancellation = None
        committed_session_id = None
        async with self._async_commit_lock:
            await self._acquire_commit_lock_async()
            try:
                current = self._session
                candidate = deepcopy(current)
                mutation(candidate)
                telemetry_activity("session.persistence.started", component="session_manager")
                pending = asyncio.ensure_future(
                    self._async_blocking_runner(
                        self._save_snapshot,
                        self._session_store,
                        self._persistence_lock,
                        candidate,
                    )
                    if self._async_blocking_runner is not None
                    else run_in_worker_async(
                        self._save_snapshot,
                        self._session_store,
                        self._persistence_lock,
                        candidate,
                    )
                )
                try:
                    await asyncio.wait({pending})
                except asyncio.CancelledError as error:
                    cancellation = error
                while not pending.done():
                    try:
                        await asyncio.wait({pending})
                    except asyncio.CancelledError as error:
                        cancellation = cancellation or error

                try:
                    pending.result()
                except BaseException as error:
                    telemetry_error(
                        "session.persistence.failed",
                        error_type="session.persistence_failed",
                        component="session_manager",
                    )
                    if cancellation is not None:
                        cancellation.add_note(
                            f"Session persistence also stopped with {type(error).__name__}."
                        )
                        raise cancellation  # pylint: disable=raise-missing-from
                    raise

                if self._session is not current:
                    if cancellation is not None:
                        raise cancellation
                    raise RuntimeError(
                        "The active session changed while its snapshot was being saved."
                    )
                for session_field in fields(Session):
                    setattr(
                        current,
                        session_field.name,
                        deepcopy(getattr(candidate, session_field.name)),
                    )
                telemetry_activity("session.persistence.completed", component="session_manager")
                committed_session_id = current.id
            finally:
                self._commit_lock.release()
        if persisted_response is not None and committed_session_id is not None:
            self._publish_response_persisted(persisted_response, session_id=committed_session_id)
        if cancellation is not None:
            raise cancellation  # pylint: disable=raise-missing-from

    async def _acquire_commit_lock_async(self) -> None:
        """Acquire the cross-thread commit lock without blocking the event loop."""
        pending = asyncio.ensure_future(
            self._async_blocking_runner(self._commit_lock.acquire)
            if self._async_blocking_runner is not None
            else run_in_worker_async(self._commit_lock.acquire)
        )
        cancellation = None
        try:
            await asyncio.wait({pending})
        except asyncio.CancelledError as error:
            cancellation = error
        while not pending.done():
            try:
                await asyncio.wait({pending})
            except asyncio.CancelledError as error:
                cancellation = cancellation or error

        try:
            pending.result()
        except BaseException as error:
            if cancellation is not None:
                cancellation.add_note(
                    f"Session commit lock acquisition also failed with {type(error).__name__}."
                )
                raise cancellation  # pylint: disable=raise-missing-from
            raise
        if cancellation is not None:
            self._commit_lock.release()
            raise cancellation

    @staticmethod
    def _save_snapshot(
        session_store: SessionStore,
        persistence_lock: Lock,
        session: Session,
    ) -> str:
        """Serialize store writes and persist one detached or active snapshot."""
        with persistence_lock:
            return session_store.save(session)

    async def _run_blocking_async(self, function: Callable[..., Any], *args: Any) -> Any:
        """Settle one synchronous operation before returning or propagating cancellation."""
        pending = asyncio.ensure_future(
            self._async_blocking_runner(function, *args)
            if self._async_blocking_runner is not None
            else run_in_worker_async(function, *args)
        )
        cancellation = None
        try:
            await asyncio.wait({pending})
        except asyncio.CancelledError as error:
            cancellation = error
        while not pending.done():
            try:
                await asyncio.wait({pending})
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        try:
            result = pending.result()
        except BaseException as error:
            if cancellation is not None:
                cancellation.add_note(
                    f"The blocking operation also stopped with {type(error).__name__}."
                )
                raise cancellation  # pylint: disable=raise-missing-from
            raise
        if cancellation is not None:
            raise cancellation
        return result
