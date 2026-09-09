"""Session manager for handling user sessions."""

import json
import logging
from asyncio import current_task
from collections.abc import Callable, Generator, Iterable
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
from threading import Lock, get_ident
from uuid import uuid7

from .. import constants
from ..backend.errors import BackendResponseError
from ..errors import Problem, log_problem
from ..instructions.models import CapturedInstruction
from ..interaction import ConsoleInteraction, Interaction
from ..models import (
    AgentRunStopReason,
    AnswerCompleted,
    AnswerDelta,
    CompactionResult,
    ContentArtifact,
    ContextReference,
    ConversationItem,
    Message,
    ModelAssignment,
    ModelContextItem,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    ResponseEvent,
    RunMetrics,
    ToolCall,
    ToolCallCompleted,
    ToolResult,
    Usage,
)
from ..permissions import AuthorizationResult
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
    register_cached_metadata,
    sha256_digest,
    store_text_stream,
    utc_now,
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

    Raises:
        SessionNotFoundError: If the requested persisted session does not exist.
        UnsupportedConversationItemError: If a serialized conversation item type is unsupported.
        ValueError: If the session or persisted format is invalid.
    """

    _interaction: Interaction
    _session: Session
    _session_store: SessionStore
    _workspace_id: str | None

    def __init__(
        self,
        interaction: Interaction | None = None,
        session: Session | str | None = None,
        session_store: SessionStore | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self._interaction = interaction or ConsoleInteraction()
        self._session_store = session_store or MemorySessionStore()
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
        interaction: Interaction | None = None,
    ) -> Response:
        """Display and collect normalized response events.

        Args:
            events (Iterable[ResponseEvent]): Response events to display and collect.
            debug (bool): Whether to display every raw response event.
            interaction (Interaction | None): Presentation override. Defaults to the configured
                interaction.

        Returns:
            Response: The collected answer, reasoning, tool calls, items, usage, and model.

        Raises:
            BackendResponseError: If the event sequence has no completion or more than one.
        """
        output = interaction or self._interaction
        reasoning = ""
        answer = ""
        tool_calls = []
        items = ()
        usage = None
        model = None
        structured_output = None
        completed = False
        response_observed = False
        reasoning_started = False
        answer_started = False

        with output.response_context():
            for event in events:
                if debug:
                    output.debug(event)

                if completed:
                    reason = (
                        "more than one response completion"
                        if isinstance(event, ResponseCompleted)
                        else "an event after response completion"
                    )
                    raise BackendResponseError(
                        f"Backend emitted {reason}.",
                        provider="backend",
                        operation="collect_response",
                        response_started=True,
                    )

                if isinstance(event, ReasoningDelta):
                    response_observed = True
                    output.reasoning_delta(event.text)
                    reasoning_started = True
                    continue
                if isinstance(event, AnswerDelta):
                    response_observed = True
                    output.answer_delta(event.text)
                    answer_started = True
                    continue
                if isinstance(event, ReasoningCompleted):
                    response_observed = True
                    reasoning = event.text
                    if not reasoning_started:
                        output.reasoning(event.text)
                    continue
                if isinstance(event, AnswerCompleted):
                    response_observed = True
                    answer = event.text
                    if not answer_started:
                        output.answer(event.text)
                    continue
                if isinstance(event, ToolCallCompleted):
                    response_observed = True
                    tool_calls.append(event.call)
                    continue
                if isinstance(event, ResponseCompleted):
                    completed = True
                    items = event.items
                    usage = event.usage
                    model = event.model
                    answer = event.answer
                    reasoning = event.reasoning
                    structured_output = event.structured_output

        if not completed:
            raise BackendResponseError(
                "Backend response ended without an explicit completion.",
                provider="backend",
                operation="collect_response",
                response_started=response_observed,
            )

        return Response(
            answer=answer,
            reasoning=reasoning,
            tool_calls=tuple(tool_calls),
            items=items,
            usage=usage or Usage(),
            model=model,
            structured_output=structured_output,
        )

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
        """Return context bounded by the latest compaction checkpoint.

        Returns:
            list[ModelContextItem]: Replacement context and subsequent history, or complete
                history when the session has not been compacted.
        """
        return self._session.model_context()

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
        if isinstance(session, str):
            session = self._session_store.load(session)
        if not isinstance(session, Session):
            raise ValueError("Invalid session type.")  # noqa: TRY004 - public API contract.
        if self._workspace_id is not None and session.workspace_id not in {
            None,
            self._workspace_id,
        }:
            raise SessionWorkspaceMismatchError(
                f"Session '{session.id}' belongs to workspace '{session.workspace_id}', "
                f"not '{self._workspace_id}'."
            )
        if session.workspace_id is None:
            session.workspace_id = self._workspace_id
        self._session = session
        for message in session.messages:
            if isinstance(message, Message):
                for reference in message.context:
                    cached = cached_path(reference.handle) if reference.handle is not None else None
                    if (
                        reference.handle is not None
                        and reference.snapshot_content is not None
                        and (cached is None or not cached[0].exists())
                    ):
                        store_text_stream(
                            [reference.snapshot_content.encode("utf-8")],
                            f"mentioned {reference.kind} {reference.path}",
                            constants.MAX_FETCH_BYTES,
                            handle=reference.handle,
                        )
            if isinstance(message, ToolResult):
                for artifact in message.artifacts:
                    register_cached_metadata(
                        artifact.handle,
                        artifact.source,
                        artifact.reloadable,
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

    def add_message(self, message: ConversationItem | Response) -> None:
        """Add conversation items and persist the resulting complete session.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        self._commit(lambda session: session.add_message(message))

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
        """
        message = Message(role="user", content=content, context=tuple(context))

        def apply(session: Session) -> None:
            if session.name is None:
                session.rename(initial_session_name(content), source=SESSION_NAME_SOURCE_INITIAL)
            session.add_message(message)

        self._commit(apply)

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

    def add_tool_call_event(self, call_id: str) -> None:
        """Persist one model tool call at its live presentation position.

        Args:
            call_id (str): Identifier of the canonical tool call to place.

        Raises:
            ValueError: If the call is unknown or already recorded.
        """
        self._commit(lambda session: session.add_tool_call_event(call_id))

    def record_tool_execution_started(self, call_id: str) -> None:
        """Persist that one tool invocation is about to cross the execution boundary.

        Args:
            call_id (str): Identifier of the model tool request being executed.

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

    def _persist_event(self, event: SessionEventModel) -> None:
        """Append and persist one standalone timeline event."""
        self._commit(lambda session: session.events.append(event))

    def _commit(self, mutation: Callable[[Session], None]) -> None:
        """Apply and persist one session mutation atomically in memory."""
        previous = deepcopy(self._session)
        try:
            mutation(self._session)
            telemetry_activity("session.persistence.started", component="session_manager")
            self._session_store.save(self._session)
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
