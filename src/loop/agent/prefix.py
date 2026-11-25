"""Prepare the durable agent input window and transient request context."""

from dataclasses import dataclass, replace

from ..backend import Backend, GenerationHyperparameters
from ..configuration import ApplicationSettings, ContextSettings
from ..errors import Problem, ProblemException
from ..instructions import InstructionsManager, PreparedInstructions
from ..model_selection import ModelSelection
from ..models import (
    DeveloperInput,
    InstructionLayout,
    ModelContextItem,
    ReasoningConfiguration,
    ReasoningEffort,
    ToolDefinition,
)
from ..session import (
    COMMAND_TIMING_FIELDS,
    Compaction,
    InstructionRevision,
    ModelDispatch,
    ModelWindow,
    Session,
    SessionManager,
    WindowSettings,
)
from ..utils import PathReference, new_uuid
from .agent import Agent
from .context import render_environment

WINDOW_SETTING_PATHS = {
    "app.agent_name": "agent_name",
    "backend.file_input_mode": "file_input_mode",
    **{f"tools.{name}": name for name in COMMAND_TIMING_FIELDS},
}


@dataclass(frozen=True)
class AgentRequest:
    """Capture one staged logical request and its immutable window boundary.

    Args:
        session_id (str): Session captured for this request.
        previous_window (str | None): Window that must still be active at commit.
        previous_checkpoint (str | None): Checkpoint that must still be active at commit.
        window (ModelWindow): Effective prefix used by the request.
        dispatch (ModelDispatch): Exact record of model-only changes at this boundary.
        recorded (bool): Whether an uncertain dispatch already committed this record.
        may_compact (bool): Whether this is a fresh request that can compact the active context.
        input (list[ModelContextItem]): Model-visible items before artifact hydration.
        instructions (str | None): Separate developer instructions when the layout uses them.
        hyperparameters (GenerationHyperparameters): Live generation controls.
        effort (ReasoningEffort | None): Live effort before projection onto a pinned prefix.
        environment_sample (str | None): Environment captured for this logical request.
        backend (Backend): Candidate provider client selected for this request.
        disclosure (ContextSettings): Disclosure policy used to capture this request.
        snapshot (PreparedInstructions): Source snapshot for the request and existing compactor.
    """

    session_id: str
    previous_window: str | None
    previous_checkpoint: str | None
    window: ModelWindow
    dispatch: ModelDispatch
    recorded: bool
    may_compact: bool
    input: list[ModelContextItem]
    instructions: str | None
    hyperparameters: GenerationHyperparameters
    effort: ReasoningEffort | None
    environment_sample: str | None
    backend: Backend
    disclosure: ContextSettings
    snapshot: PreparedInstructions


@dataclass(frozen=True)
class _WindowTarget:
    """Capture the effective render settings before reading instruction sources."""

    backend: Backend
    model: str
    layout: InstructionLayout
    current: ModelWindow | None
    checkpoint: Compaction | None
    provider_state_start: int
    settings: WindowSettings


class AgentContext:
    """Own the pinned prefix, live disclosure, and rebuild policy for an agent.

    Args:
        agent (Agent): Agent that supplies the intrinsic template and tool registry.
        backend (Backend): Active backend candidate.
        instructions (InstructionsManager): Instruction source and skill state owner.
        sessions (SessionManager): Durable context writer.
        selection (ModelSelection): Active model and explicit selection state.
        working_directory (PathReference): Directory used by the active tool runtime.
        settings (ContextSettings): User-owned disclosure and clock preferences.
    """

    agent: Agent
    backend: Backend
    _instructions: InstructionsManager
    _sessions: SessionManager
    _selection: ModelSelection
    _working_directory: PathReference
    _saved: WindowSettings
    _disclosure: ContextSettings
    _environment_sample: tuple[str, str | None] | None

    def __init__(
        self,
        agent: Agent,
        backend: Backend,
        instructions: InstructionsManager,
        sessions: SessionManager,
        selection: ModelSelection,
        working_directory: PathReference,
        *,
        settings: ContextSettings,
    ) -> None:
        self.agent = agent
        self.backend = backend
        self._instructions = instructions
        self._sessions = sessions
        self._selection = selection
        self._working_directory = working_directory
        self._saved = WindowSettings(
            agent_name=agent.name,
            file_input_mode=backend.file_input_mode,
            **{
                name: getattr(agent.tools.registry.settings, name) for name in COMMAND_TIMING_FIELDS
            },
        )
        self._disclosure = settings
        self._environment_sample = None

    def replace_agent(self, agent: Agent) -> None:
        """Capture the agent definition for the next eligible window.

        Args:
            agent (Agent): New agent definition and tool owner.
        """
        self.agent = agent
        self._saved = self._saved.model_copy(update={"agent_name": agent.name})
        self.apply_window(self._sessions.session.window)

    def replace_backend(self, backend: Backend) -> None:
        """Publish the next backend candidate and authorize an explicitly changed route.

        Args:
            backend (Backend): Validated backend candidate.
        """
        switched = self.backend.route != backend.route
        self.backend = backend
        if switched:
            self.allow_route_switch()
        self._saved = self._saved.model_copy(update={"file_input_mode": backend.file_input_mode})

    def allow_route_switch(self) -> None:
        """Accept the configured route for the active session after an explicit user action."""
        self._sessions.allow_model_route(self.backend.route)

    def configure(self, settings: ApplicationSettings) -> None:
        """Save validated prefix preferences and immediately update disclosure and the clock.

        Args:
            settings (ApplicationSettings): Latest validated settings.
        """
        window = self._sessions.session.window
        self._saved = WindowSettings(
            agent_name=settings.app.agent_name,
            file_input_mode=settings.backend.file_input_mode,
            **{name: getattr(settings.tools, name) for name in COMMAND_TIMING_FIELDS},
        )
        if self._disclosure != settings.context:
            self._disclosure = settings.context
            self._environment_sample = None
        self.apply_window(window)

    def pending(self, path: str) -> str | None:
        """Report an effective prefix value when its saved preference is deferred.

        Args:
            path (str): Dot-separated configuration key.

        Returns:
            str | None: Pending status including the effective value.
        """
        name = WINDOW_SETTING_PATHS.get(path)
        current = self._sessions.session.window
        if name is not None and current is not None:
            effective = getattr(current.settings, name)
            if getattr(self._candidate_settings(), name) != effective:
                return f"pending until the next context rebuild; effective value is {effective!r}"
        return None

    def apply_window(self, window: ModelWindow | None) -> None:
        """Apply the established timing values to live tool behavior.

        Args:
            window (ModelWindow | None): Active window, or None for current saved defaults.
        """
        effective = window.settings if window is not None else self._candidate_settings()
        self._instructions.bind_agent_budget(
            self._agent_for(effective), pinned=window.baseline if window is not None else None
        )
        registry = self.agent.tools.registry
        registry.settings = replace(
            registry.settings,
            timezone=self._disclosure.timezone,
            **{name: getattr(effective, name) for name in COMMAND_TIMING_FIELDS},
        )

    def prepare(self, recovery: str | None) -> AgentRequest:
        """Prepare or replay one logical model request.

        Args:
            recovery (str | None): Request-local recovery strategy.

        Returns:
            AgentRequest: Captured model input and its durable dispatch record.

        Raises:
            ProblemException: The saved route is unavailable or the target changed.
        """
        target = self._window_target()
        snapshot = self._pending_snapshot(target.current, recovery)
        if snapshot is None:
            snapshot = self._instructions.prepare(
                self._agent_for(target.settings),
                recovery=recovery,
                pinned=target.current.baseline if target.current is not None else None,
            )
        return self._assemble(target, snapshot)

    async def prepare_async(self, recovery: str | None) -> AgentRequest:
        """Prepare a logical request using asynchronous instruction discovery.

        Args:
            recovery (str | None): Request-local recovery strategy.

        Returns:
            AgentRequest: Captured model input and its durable dispatch record.

        Raises:
            ProblemException: The saved route is unavailable or the target changed.
        """
        target = self._window_target()
        snapshot = self._pending_snapshot(target.current, recovery)
        if snapshot is None:
            snapshot = await self._instructions.prepare_async(
                self._agent_for(target.settings),
                recovery=recovery,
                pinned=target.current.baseline if target.current is not None else None,
            )
        return self._assemble(target, snapshot)

    def _assemble(self, target: _WindowTarget, snapshot: PreparedInstructions) -> AgentRequest:
        """Project the prepared snapshot onto its captured window and replay order."""
        if self.backend is not target.backend or self._target_model() != target.model:
            self._raise_stale_request()
        session = self._sessions.session
        previous_window = session.window
        active_checkpoint = session.checkpoint()
        window = target.current or self._new_window(target, snapshot)
        records = session.model_dispatches(window)
        stored = self._unsettled(session, records)
        effort = self.agent.hyperparameters.reasoning_effort
        hyperparameters = replace(self.agent.hyperparameters)
        reasoning = None
        if target.backend.supports_reasoning_updates(target.model):
            hyperparameters.reasoning_effort = window.initial_reasoning_effort
            active_effort = next(
                (record.reasoning.effort for record in reversed(records) if record.reasoning),
                window.initial_reasoning_effort,
            )
            if effort != active_effort:
                reasoning = ReasoningConfiguration(effort)
        revision = None
        if target.current is not None and stored is None:
            revision = InstructionRevision.compare(
                self._effective_snapshot(window, records), replace(snapshot, suffix=None)
            )
        dispatch = (
            stored
            if stored is not None and reasoning is None
            else ModelDispatch(
                id=str(new_uuid()),
                window_id=window.id,
                position=len(session.messages),
                revision=revision,
                reasoning=reasoning,
            )
        )
        recorded = dispatch is stored
        follows_default = not self._selection.selected
        if not recorded and window.follows_default != follows_default:
            window = window.model_copy(update={"follows_default": follows_default})
        disclosure = self._disclosure
        sample = self._request_environment(
            window.layout, (stored or dispatch).id, snapshot, disclosure
        )
        instructions = _join_guidance(
            window.leading_instructions(
                revision.snapshot
                if revision is not None
                else self._effective_snapshot(window, records)
            ),
            snapshot.suffix if window.layout != "inline" else None,
            sample,
        )
        context = session.generation_context(window=window, dispatch=None if recorded else dispatch)
        if window.layout == "inline" and snapshot.suffix:
            context.append(DeveloperInput(snapshot.suffix))
        return AgentRequest(
            session_id=session.id,
            previous_window=previous_window.id if previous_window is not None else None,
            previous_checkpoint=active_checkpoint.id if active_checkpoint is not None else None,
            window=window,
            dispatch=dispatch,
            recorded=recorded,
            may_compact=stored is None and (previous_window is None or target.current is not None),
            input=context,
            instructions=instructions,
            hyperparameters=hyperparameters,
            effort=effort,
            environment_sample=sample,
            backend=target.backend,
            disclosure=disclosure,
            snapshot=snapshot,
        )

    def verify(self, request: AgentRequest, *, persisted: bool = False) -> None:
        """Check the replay boundary and the policy that authorizes transmission.

        Args:
            request (AgentRequest): The prepared model request.
            persisted (bool): The session write already checked the replay boundary.

        Raises:
            ProblemException: The session or request policy changed before transmission.
        """
        if not persisted:
            session = self._sessions.session
            checkpoint = session.checkpoint()
            if (
                session.id != request.session_id
                or len(session.messages) != request.dispatch.position
                or (session.window.id if session.window else None) != request.previous_window
                or (checkpoint.id if checkpoint else None) != request.previous_checkpoint
                or (
                    request.previous_window != request.window.id
                    and self._candidate_settings() != request.window.settings
                )
            ):
                self._raise_stale_request()
        live = self.agent.hyperparameters
        if (
            self.backend is not request.backend
            or self._disclosure != request.disclosure
            or (
                not request.recorded
                and self._instructions.generation != request.snapshot.generation
            )
            or self._target_model() != request.window.model
            or live.temperature != request.hyperparameters.temperature
            or live.reasoning_effort != request.effort
            or not self._tools_still_eligible(request.window)
        ):
            self._raise_stale_request()

    @staticmethod
    def _raise_stale_request() -> None:
        """Report a local request that must be prepared again."""
        raise ProblemException(
            Problem(
                code="agent.stale_request",
                title="Could not prepare the request",
                detail="The model request changed before dispatch. Retry the turn.",
                operation="prepare_generation",
            )
        )

    def remember_environment(self, request: AgentRequest) -> None:
        """Keep the environment sample for retries of this logical request.

        Args:
            request (AgentRequest): Durably admitted request.
        """
        self._environment_sample = (request.dispatch.id, request.environment_sample)

    def _target_model(self) -> str:
        """Resolve the authorized target for this session."""
        self._sessions.require_model_route(self.backend.route)
        return self._selection.effective

    def _window_target(self) -> _WindowTarget:
        """Select a reusable window and checkpoint for the authorized target."""
        model = self._target_model()
        session = self._sessions.session
        route = self.backend.route
        layout = self.backend.instruction_layout(model)
        checkpoint, provider_state_start = session.replay_boundary(route, model, layout)
        current = session.window
        if current is not None and (
            (current.route, current.model, current.layout) != (route, model, layout)
            or current.checkpoint_id != (checkpoint.id if checkpoint else None)
            or not self._tools_still_eligible(current)
        ):
            current = None
        return _WindowTarget(
            self.backend,
            model,
            layout,
            current,
            checkpoint,
            provider_state_start,
            current.settings if current is not None else self._candidate_settings(),
        )

    def _candidate_settings(self) -> WindowSettings:
        """Resolve the pending file projection for this deployment."""
        return self._saved.model_copy(
            update={
                "file_input_mode": self.backend.resolve_file_input_mode(self._saved.file_input_mode)
            }
        )

    def _agent_for(self, settings: WindowSettings) -> Agent:
        """Render the agent identity pinned to the existing or candidate window."""
        return replace(self.agent, identity=replace(self.agent.identity, name=settings.agent_name))

    def _tools(self, settings: WindowSettings) -> tuple[ToolDefinition, ...]:
        """Render schemas using candidate timings without mutating execution behavior."""
        registry = self.agent.tools.registry
        effective = replace(
            registry.settings, **{name: getattr(settings, name) for name in COMMAND_TIMING_FIELDS}
        )
        return tuple(registry.definitions(settings=effective))

    def _new_window(self, target: _WindowTarget, snapshot: PreparedInstructions) -> ModelWindow:
        """Capture baseline, tools, and the initial effort at a rebuild."""
        previous = self._sessions.session.window
        initial_effort = (
            previous.initial_reasoning_effort
            if previous is not None
            and (previous.route, previous.model) == (target.backend.route, target.model)
            else self.agent.hyperparameters.reasoning_effort
        )
        return ModelWindow(
            id=str(new_uuid()),
            route=target.backend.route,
            model=target.model,
            layout=target.layout,
            settings=target.settings,
            baseline=replace(snapshot, suffix=None),
            tools=self._tools(target.settings),
            initial_reasoning_effort=initial_effort,
            checkpoint_id=target.checkpoint.id if target.checkpoint is not None else None,
            provider_state_start=target.provider_state_start,
            follows_default=not self._selection.selected,
        )

    def _tools_still_eligible(self, window: ModelWindow) -> bool:
        """Detect a withdrawn capability while allowing optional schema changes to wait."""
        live = {tool.name: tool for tool in self._tools(window.settings)}
        return all(
            tool.name in live
            and set(tool.parameters.get("properties", {}))
            <= set(live[tool.name].parameters.get("properties", {}))
            for tool in window.tools
        )

    def _pending_snapshot(
        self,
        window: ModelWindow | None,
        recovery: str | None,
    ) -> PreparedInstructions | None:
        """Reuse the committed instructions with the current admission generation."""
        if window is None:
            return None
        session = self._sessions.session
        records = session.model_dispatches(window)
        if self._unsettled(session, records) is None:
            return None
        return replace(
            self._effective_snapshot(window, records),
            generation=self._instructions.generation,
            suffix=self._instructions.request_suffix(recovery),
        )

    @staticmethod
    def _unsettled(session: Session, records: tuple[ModelDispatch, ...]) -> ModelDispatch | None:
        """Find the last transport boundary when no canonical response has settled it."""
        if records:
            record = records[-1]
            if record.id == session.pending_dispatch and record.position == len(session.messages):
                return record
        return None

    @staticmethod
    def _effective_snapshot(
        window: ModelWindow,
        records: tuple[ModelDispatch, ...],
    ) -> PreparedInstructions:
        """Find the latest committed canonical instruction state in a window."""
        return next(
            (record.revision.snapshot for record in reversed(records) if record.revision),
            window.baseline,
        )

    def _request_environment(
        self,
        layout: InstructionLayout,
        dispatch_id: str,
        snapshot: PreparedInstructions,
        disclosure: ContextSettings,
    ) -> str | None:
        """Render one request sample and reserve time facts for a true suffix."""
        cached = self._environment_sample
        if cached is not None and cached[0] == dispatch_id:
            sample = cached[1]
        else:
            sample = render_environment(
                disclosure,
                self._working_directory,
                client_timezone=self.agent.tools.registry.settings.client_timezone,
                include_time=layout == "suffix",
            )
        if sample is not None and (
            len(snapshot.content_with_suffix.encode()) + len(sample.encode()) + 2
            > self._instructions.max_bytes
        ):
            return None
        return sample


def _join_guidance(*sections: str | None) -> str | None:
    """Combine applicable request-scoped developer sections."""
    return "\n\n".join(section for section in sections if section) or None
