"""Session manager for handling user sessions."""

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import uuid7

from .. import constants
from ..interaction import ConsoleInteraction, Interaction
from ..models import (
    CompactionResult,
    ContentArtifact,
    ContextReference,
    ConversationItem,
    Message,
    ModelContextItem,
    Response,
    ToolResult,
)
from ..utils import (
    bound_tool_result,
    cached_metadata,
    cached_path,
    register_cached_metadata,
    sha256_digest,
    store_text_stream,
)
from .models import (
    SESSION_NAME_SOURCE_GENERATED,
    SESSION_NAME_SOURCE_INITIAL,
    Compaction,
    InstructionSnapshot,
    SessionNameGenerator,
)
from .naming import initial_session_name
from .session import Session, SessionStore
from .store import MemorySessionStore


class SessionManager:
    """Manage user sessions, including persistence and interaction.

    Args:
        interaction (Interaction | None): Service used for user input and output. Defaults to a
            console interaction.
        session (Session | str | None): Active session or persisted session identifier to load.
            Defaults to a fresh session.
        session_store (SessionStore | None): Store used to persist and retrieve sessions. Defaults
            to an instance-local memory store.
        compaction_threshold (float): Context-window utilization that requires compaction.
            Defaults to ``0.8``.

    Raises:
        SessionNotFoundError: If the requested persisted session does not exist.
        UnsupportedConversationItemError: If a serialized conversation item type is unsupported.
        ValueError: If the session, persisted format, or compaction threshold is invalid.
    """

    _interaction: Interaction
    _session: Session
    _session_store: SessionStore
    _compaction_threshold: float

    def __init__(
        self,
        interaction: Interaction | None = None,
        session: Session | str | None = None,
        session_store: SessionStore | None = None,
        compaction_threshold: float = constants.DEFAULT_COMPACTION_THRESHOLD,
    ) -> None:
        if not 0 < compaction_threshold < 1:
            raise ValueError("Compaction threshold must be between zero and one.")
        self._interaction = interaction or ConsoleInteraction()
        self._session_store = session_store or MemorySessionStore()
        self._compaction_threshold = compaction_threshold

        if session and isinstance(session, (str, Session)):
            self.load_session(session)
        else:
            self._session = Session()

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

    @property
    def model_context(self) -> list[ModelContextItem]:
        """Return context bounded by the latest compaction checkpoint.

        Returns:
            list[ModelContextItem]: Replacement context and subsequent history, or complete
                history when the session has not been compacted.
        """
        return self._session.model_context()

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

    @property
    def compaction_threshold(self) -> float:
        """Return the configured context-window utilization threshold.

        Returns:
            float: Utilization ratio that requires compaction.
        """
        return self._compaction_threshold

    def can_compact(self) -> bool:
        """Return whether complete history advanced beyond the latest checkpoint.

        Returns:
            bool: ``True`` when at least one uncompacted history item exists.
        """
        boundary = self._session.compactions[-1].boundary if self._session.compactions else 0
        return boundary < len(self._session.messages)

    def compaction_needed(self) -> bool:
        """Return whether current usage reached the automatic compaction threshold.

        Returns:
            bool: ``True`` when capacity is known, usage reached the threshold, and new history
                can be compacted.
        """
        context_window = self._session.context_window
        return (
            context_window is not None
            and self._session.tokens >= context_window * self._compaction_threshold
            and self.can_compact()
        )

    def add_compaction(
        self,
        result: CompactionResult,
        *,
        model: str,
        instructions: str | None,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
    ) -> None:
        """Persist one compaction checkpoint and its exact instruction state.

        Args:
            result (CompactionResult): Replacement context and reported token usage.
            model (str): Model used to compact the active context.
            instructions (str | None): Complete instructions supplied to the compactor.
            working_directory (str): Effective instruction-discovery directory.
            active_skills (Iterable[tuple[str, str]]): Active skill identities.

        Raises:
            ValueError: If the checkpoint does not advance or has invalid context.
        """
        identities = tuple(active_skills)
        provider = result.items[0].provider if result.items else ""
        compaction = Compaction(
            id=str(uuid7()),
            boundary=len(self._session.messages),
            created_at=datetime.now(UTC),
            provider=provider,
            model=model,
            context=result.items,
            instructions=InstructionSnapshot(
                working_directory=working_directory,
                content=instructions,
                digest=sha256_digest(instructions or ""),
                active_skills=identities,
            ),
            input_tokens_before=self._session.tokens,
            input_tokens_after=result.context_tokens,
        )
        previous_tokens = self._session.tokens
        previous_directory = self._session.instruction_working_directory
        previous_skills = list(self._session.active_skills)
        self._session.add_compaction(compaction)
        try:
            if result.context_tokens is not None:
                self._session.tokens = result.context_tokens
            self._session.update_instruction_state(working_directory, identities)
            self._session_store.save(self._session)
        except Exception:
            self._session.compactions.pop()
            self._session.tokens = previous_tokens
            self._session.instruction_working_directory = previous_directory
            self._session.active_skills = previous_skills
            raise

    def load_session(self, session: Session | str) -> None:
        """Load a session from the store.

        Args:
            session (Session | str): The session to load, or its identifier.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If its persisted format is invalid or unsupported.
        """
        if isinstance(session, str):
            session = self._session_store.load(session)
        if not isinstance(session, Session):
            raise ValueError("Invalid session type.")  # noqa: TRY004 - public API contract.
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
        self._session = Session()

    def rename_session(self, name: str) -> None:
        """Assign and persist a user-controlled name to the active session.

        Args:
            name (str): New non-empty session name.

        Raises:
            ValueError: If the name is empty after normalization.
        """
        self._session.rename(name)
        self._session_store.save(self._session)

    def generate_session_name(
        self,
        generator: SessionNameGenerator,
    ) -> None:
        """Generate a session name from the first completed exchange.

        Args:
            generator (SessionNameGenerator): Auxiliary title generation service.
        """
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
            return
        name = generator.generate(user_message, assistant_message, self._session.model)
        if name:
            self._session.rename(name, source=SESSION_NAME_SOURCE_GENERATED)
            self._session_store.save(self._session)

    def add_message(self, message: ConversationItem | Response) -> None:
        """Add conversation items and persist the resulting complete session.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        self._session.add_message(message)
        self._session_store.save(self._session)

    def add_messages(self, messages: Iterable[ConversationItem]) -> None:
        """Add messages to the conversation history.

        Args:
            messages (Iterable[ConversationItem]): Conversation items to add.

        Raises:
            ValueError: If any value is not a supported conversation item.
        """
        self._session.add_messages(messages)
        self._session_store.save(self._session)

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
        if self._session.name is None:
            self._session.rename(initial_session_name(content), source=SESSION_NAME_SOURCE_INITIAL)
        self.add_message(Message(role="user", content=content, context=tuple(context)))

    def add_tool_call(
        self,
        call_id: str,
        output: str,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
    ) -> None:
        """Add a tool result and its instruction state to the session.

        Args:
            call_id (str): Identifier used to associate the call with its result.
            output (str): The content of the tool result.
            working_directory (str): Effective instruction directory after the tool call.
            active_skills (Iterable[tuple[str, str]]): Active skill names and canonical locations
                after the tool call.
        """
        self.update_instruction_state(working_directory, active_skills)
        output, handle = bound_tool_result(output, f"tool result {call_id}")
        if handle is not None:
            self._interaction.info(
                f"Tool result '{call_id}' exceeded the context limit and was cached as '{handle}'."
            )
        self.add_message(
            ToolResult(
                call_id=call_id,
                output=output,
                artifacts=self._content_artifacts(output),
            )
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
