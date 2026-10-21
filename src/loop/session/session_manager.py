"""Session manager for handling user sessions."""

import asyncio
import json
import logging
from asyncio import current_task
from collections.abc import AsyncIterable, Awaitable, Callable, Generator, Iterable
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
from threading import Lock, get_ident
from typing import Any
from uuid import uuid7

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
    telemetry_error,
    telemetry_span,
)
from ..utils import (
    bound_tool_result,
    cached_metadata,
    cached_path,
    content_digest,
    content_identity,
    encode_content_cursor,
    get_binary,
    register_cached_metadata,
    run_in_worker,
    sha256_digest,
    store_content,
    utc_now,
    validate_content_handle,
)
from .models import (
    SESSION_NAME_SOURCE_GENERATED,
    SESSION_NAME_SOURCE_INITIAL,
    Compaction,
    CompactionEvent,
    InstructionSnapshot,
    PermissionEvent,
    RunCompletedEvent,
    SessionEventModel,
    SessionExecutionConflictError,
    SessionNameGenerator,
    SessionRecoveryState,
    SessionWorkspaceMismatchError,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)
from .naming import initial_session_name
from .session import Session
from .store import MemorySessionStore, SessionStore

type ExecutionKey = tuple[tuple[str, str | int], str]
type ExecutionOwner = tuple[int, int, int | None]

_LOGGER = logging.getLogger(__name__)
_EXECUTION_OWNERS: dict[ExecutionKey, tuple[ExecutionOwner, int]] = {}
_EXECUTION_OWNERS_LOCK = Lock()


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
    """

    _interaction: Interaction
    _session: Session
    _session_store: SessionStore
    _workspace_id: str | None
    _async_blocking_runner: Callable[..., Awaitable[Any]] | None
    _persistence_lock: Lock

    def __init__(
        self,
        interaction: Interaction | None = None,
        session: Session | str | None = None,
        session_store: SessionStore | None = None,
        workspace_id: str | None = None,
        async_blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._interaction = interaction or ConsoleInteraction()
        self._session_store = session_store or MemorySessionStore()
        self._async_blocking_runner = async_blocking_runner
        self._persistence_lock = Lock()
        if workspace_id == "":
            raise ValueError("Workspace identifier must not be empty.")
        self._workspace_id = workspace_id

        if session and isinstance(session, (str, Session)):
            self.load_session(session)
        else:
            self._session = Session(workspace_id=self._workspace_id)

    @property
    def session(self) -> Session:
        """Return the active session.

        Returns:
            Session: The mutable session used by the loop.
        """
        return self._session

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
    def execution(self) -> Generator[None, None, None]:
        """Own this logical session's execution within the current process.

        This lease intentionally provides process-local exclusion only. Persisted revision checks
        independently prevent stale writers in other processes from overwriting newer snapshots.

        Yields:
            None: Control while this manager exclusively owns session execution.

        Raises:
            SessionExecutionConflictError: If another execution in this process owns the session.
        """
        namespace = (
            ("workspace", self._session.workspace_id)
            if self._session.workspace_id is not None
            else ("store", id(self._session_store))
        )
        key = (namespace, self._session.id)
        owner = _execution_owner(self)
        with _EXECUTION_OWNERS_LOCK:
            existing = _EXECUTION_OWNERS.get(key)
            if existing is not None and existing[0] != owner:
                raise SessionExecutionConflictError(self._session.id)
            depth = existing[1] + 1 if existing is not None else 1
            _EXECUTION_OWNERS[key] = (owner, depth)
        try:
            yield
        finally:
            with _EXECUTION_OWNERS_LOCK:
                current_owner, depth = _EXECUTION_OWNERS[key]
                if depth == 1:
                    del _EXECUTION_OWNERS[key]
                else:
                    _EXECUTION_OWNERS[key] = (current_owner, depth - 1)

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
            if isinstance(event, (ToolExecutionStartedEvent, ToolExecutionCompletedEvent)):
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
        context = self._session.model_context()
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
                stored = self._session_store.load_reference(reference.version)
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
            id=str(uuid7()),
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
            id=str(uuid7()),
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
            ValueError: If its persisted format is invalid or unsupported.
            SessionWorkspaceMismatchError: If the session belongs to another workspace.
        """
        if load_from_store := isinstance(session, str):
            session = self._session_store.load(session)
        if not isinstance(session, Session):
            raise ValueError("Invalid session type.")  # noqa: TRY004 - public API contract.
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
                        raise ValueError(f"Referenced snapshot '{reference.path}' is unavailable.")
                    digest = content_digest(stored)
                    if digest != reference.version or len(stored) != reference.size_bytes:
                        raise ValueError(
                            f"Referenced snapshot '{reference.path}' failed integrity validation."
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
        self._session = session

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
        """Replace the active session with a fresh unpersisted session."""
        self._session = Session(
            model=self._session.model,
            workspace_id=self._workspace_id,
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
    ) -> None:
        """Generate, persist, and report a session name from the first completed exchange.

        Args:
            generator (SessionNameGenerator): Auxiliary title generation service.
        """
        self._interaction.info("Generating a session name...")
        try:
            user_message = next(
                (
                    message.content
                    for message in self._session.messages
                    if isinstance(message, Message) and message.role == "user"
                ),
                "",
            )
            assistant_message = next(
                (
                    message.content
                    for message in self._session.messages
                    if isinstance(message, Message) and message.role == "assistant"
                ),
                "",
            )
            if not user_message or not assistant_message.strip():
                self._interaction.info("No completed exchange found to generate a session name.")
                return
            name = generator.generate(user_message, assistant_message, self._session.model)
            if name:
                self._commit(
                    lambda session: session.rename(name, source=SESSION_NAME_SOURCE_GENERATED)
                )
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

    async def generate_session_name_async(self, generator: SessionNameGenerator) -> None:
        """Generate and persist a session name without blocking asynchronous model I/O.

        Generators with ``generate_async`` use their native asynchronous implementation. Existing
        synchronous generators remain supported for compatibility.

        Args:
            generator (SessionNameGenerator): Auxiliary title generation service.

        Returns:
            None: The generated name is persisted when a usable exchange exists.

        Raises:
            asyncio.CancelledError: Naming is cancelled after any owned worker settles.
        """
        self._interaction.info("Generating a session name...")
        try:
            user_message = next(
                (
                    message.content
                    for message in self._session.messages
                    if isinstance(message, Message) and message.role == "user"
                ),
                "",
            )
            assistant_message = next(
                (
                    message.content
                    for message in self._session.messages
                    if isinstance(message, Message) and message.role == "assistant"
                ),
                "",
            )
            if not user_message or not assistant_message.strip():
                self._interaction.info("No completed exchange found to generate a session name.")
                return
            async_generator = getattr(generator, "generate_async", None)
            if callable(async_generator):
                name = await async_generator(user_message, assistant_message, self._session.model)
            else:
                name = await self._run_blocking(
                    generator.generate,
                    user_message,
                    assistant_message,
                    self._session.model,
                )
            if name:
                await self._commit_async(
                    lambda session: session.rename(name, source=SESSION_NAME_SOURCE_GENERATED)
                )
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

    def add_message(self, message: ConversationItem | Response) -> None:
        """Add conversation items and persist the resulting complete session.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Returns:
            None: The conversation item is durably committed.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        self._commit(lambda session: session.add_message(message))

    async def add_message_async(self, message: ConversationItem | Response) -> None:
        """Add conversation items through the asynchronous persistence boundary.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Raises:
            ValueError: If the value is not a supported conversation item.
            asyncio.CancelledError: If the owning task is cancelled after persistence settles.
        """
        await self._commit_async(lambda session: session.add_message(message))

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
    ) -> None:
        """Construct and persist one complete user message.

        Args:
            content (str): Submitted user-message text.
            context (Iterable[ContextReference]): Resolved context snapshots attached to the
                message. Defaults to no explicit context.

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
            session.add_message(message)

        self._commit(apply)

    async def add_user_message_async(
        self,
        content: str,
        context: Iterable[ContextReference] = (),
    ) -> None:
        """Persist a user message before async model work begins.

        Args:
            content (str): Submitted user-message text.
            context (Iterable[ContextReference]): Resolved context snapshots attached to the
                message. Defaults to no explicit context.

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
            session.add_message(message)

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
                        id=str(uuid7()),
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
                        id=str(uuid7()),
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
            ToolExecutionStartedEvent(id=str(uuid7()), created_at=utc_now(), call_id=call_id)
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
            ToolExecutionStartedEvent(id=str(uuid7()), created_at=utc_now(), call_id=call_id)
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
        """
        self.add_message(response)

    async def add_response_async(self, response: Response) -> None:
        """Persist one complete backend response through the async session writer.

        Args:
            response (Response): The response to add.

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
            id=str(uuid7()),
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
            id=str(uuid7()),
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
            id=str(uuid7()),
            created_at=utc_now(),
            stop_reason=stop_reason,
            started_at=started_at,
            metrics=metrics,
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
            id=str(uuid7()),
            created_at=utc_now(),
            stop_reason=stop_reason,
            started_at=started_at,
            metrics=metrics,
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
        previous = deepcopy(self._session)
        try:
            mutation(self._session)
            telemetry_activity("session.persistence.started", component="session_manager")
            self._save_snapshot(self._session_store, self._persistence_lock, self._session)
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
                    self._session,
                    session_field.name,
                    deepcopy(getattr(previous, session_field.name)),
                )
            raise

    async def _commit_async(self, mutation: Callable[[Session], None]) -> None:
        """Persist a detached session mutation through bounded blocking admission."""
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
            else run_in_worker(
                self._save_snapshot,
                self._session_store,
                self._persistence_lock,
                candidate,
            )
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
            telemetry_error(
                "session.persistence.failed",
                error_type="session.persistence_failed",
                component="session_manager",
            )
            if cancellation is not None:
                cancellation.add_note(
                    f"Session persistence also stopped with {type(error).__name__}."
                )
                raise cancellation
            raise

        if self._session is not current:
            if cancellation is not None:
                raise cancellation
            raise RuntimeError("The active session changed while its snapshot was being saved.")
        for session_field in fields(Session):
            setattr(
                current,
                session_field.name,
                deepcopy(getattr(candidate, session_field.name)),
            )
        telemetry_activity("session.persistence.completed", component="session_manager")
        if cancellation is not None:
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

    async def _run_blocking(self, function: Callable[..., Any], *args: Any) -> Any:
        """Settle one synchronous operation before returning or propagating cancellation."""
        pending = asyncio.ensure_future(
            self._async_blocking_runner(function, *args)
            if self._async_blocking_runner is not None
            else run_in_worker(function, *args)
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
                raise cancellation
            raise
        if cancellation is not None:
            raise cancellation
        return result
