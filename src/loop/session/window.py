"""Describe durable prefix settings and logical model dispatches."""

import json
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .. import constants
from ..instructions import PreparedInstructions
from ..models import (
    DeveloperInput,
    FileInputMode,
    InstructionLayout,
    ReasoningConfiguration,
    ReasoningEffort,
    ToolDefinition,
)
from ..utils import sha256_digest

COMMAND_TIMING_FIELDS = (
    "command_job_runtime_limit_seconds",
    "command_default_initial_wait_seconds",
    "command_default_poll_wait_seconds",
    "command_max_wait_seconds",
)


class WindowSettings(BaseModel):
    """Keep non-secret settings that affect a context window's prefix.

    Args:
        agent_name (str): Effective agent identity for this window.
        file_input_mode (FileInputMode | None): Pinned attachment representation.
        command_job_runtime_limit_seconds (float): Deadline for newly admitted jobs.
        command_default_initial_wait_seconds (float): Advertised initial wait.
        command_default_poll_wait_seconds (float): Advertised polling wait.
        command_max_wait_seconds (float): Maximum initial or polling wait.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_name: str
    file_input_mode: FileInputMode | None = None
    command_job_runtime_limit_seconds: float = Field(
        default=constants.DEFAULT_COMMAND_JOB_RUNTIME_LIMIT_SECONDS, gt=0, allow_inf_nan=False
    )
    command_default_initial_wait_seconds: float = Field(
        default=constants.DEFAULT_COMMAND_INITIAL_WAIT_SECONDS, ge=0, allow_inf_nan=False
    )
    command_default_poll_wait_seconds: float = Field(
        default=constants.DEFAULT_COMMAND_POLL_WAIT_SECONDS, ge=0, allow_inf_nan=False
    )
    command_max_wait_seconds: float = Field(
        default=constants.DEFAULT_COMMAND_MAX_WAIT_SECONDS,
        ge=0,
        le=constants.MAX_COMMAND_WAIT_SECONDS,
        allow_inf_nan=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _discard_legacy_timezone_pin(cls, data: object) -> object:
        """Read saved windows that pinned a timezone before time moved out of the prefix."""
        if isinstance(data, dict) and "timezone" in data:
            return {key: value for key, value in data.items() if key != "timezone"}
        return data

    @model_validator(mode="after")
    def validate_waits(self) -> "WindowSettings":
        """Require each pinned command wait to fit within the supported cap.

        Returns:
            WindowSettings: Validated settings.

        Raises:
            ValueError: A default wait exceeds the pinned maximum.
        """
        if max(
            self.command_default_initial_wait_seconds, self.command_default_poll_wait_seconds
        ) > (self.command_max_wait_seconds):
            raise ValueError("Pinned command waits exceed the configured maximum.")
        return self


class ModelWindow(BaseModel):
    """Pin the instructions, settings, and ordered tools for one request prefix.

    Args:
        id (str): Stable identity for the window.
        route (str): Credential-free route that owns its provider-specific context.
        model (str): Selected provider model for this window.
        layout (InstructionLayout): Developer instruction placement.
        settings (WindowSettings): Effective prefix-affecting settings.
        baseline (PreparedInstructions): Immutable initial instructions.
        tools (tuple[ToolDefinition, ...]): Tool definitions in provider order.
        initial_reasoning_effort (ReasoningEffort | None): Pinned effort on append-capable routes.
        checkpoint_id (str | None): Checkpoint that supplies the rebuilt canonical context.
        provider_state_start (int): History before this index is replayed without provider state.
        follows_default (bool): Whether the most recent dispatch followed the backend default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    route: str
    model: str = Field(min_length=1)
    layout: InstructionLayout
    settings: WindowSettings
    baseline: PreparedInstructions
    tools: tuple[ToolDefinition, ...]
    initial_reasoning_effort: ReasoningEffort | None = None
    checkpoint_id: str | None = None
    provider_state_start: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    follows_default: bool = False

    @model_validator(mode="after")
    def validate_baseline(self) -> "ModelWindow":
        """Validate persisted instructions and keep transient guidance out of the window.

        Returns:
            ModelWindow: Validated context window.

        Raises:
            ValueError: A baseline contains transient guidance or has an invalid digest.
        """
        if self.baseline.suffix or sha256_digest(self.baseline.content) != self.baseline.digest:
            raise ValueError("Invalid model-window instruction baseline.")
        return self

    def baseline_item(self) -> DeveloperInput:
        """Render the initial developer item and the rules for applying later revisions.

        Returns:
            DeveloperInput: Initial instruction sections with stable labels.
        """
        sections = [(part.identity, part.render()) for part in self.baseline.sections]
        content = json.dumps(sections, ensure_ascii=False)
        return DeveloperInput(
            "The following ordered instruction sections apply in order of precedence. "
            "An instruction revision replaces the listed sections, removes deactivated "
            "sections, and supplies their new order. Obsolete sections no longer apply. "
            f"\n<instruction_baseline>\n{content}\n</instruction_baseline>"
        )

    def leading_instructions(self, snapshot: PreparedInstructions | None = None) -> str | None:
        """Return the legacy leading document when that route requires it.

        Args:
            snapshot (PreparedInstructions | None): Effective revision, or the initial baseline.

        Returns:
            str | None: Effective instructions for a leading route, otherwise None.
        """
        if self.layout != "leading":
            return None
        return (snapshot or self.baseline).content


class InstructionRevision(BaseModel):
    """Record the exact replacements and canonical snapshot they establish.

    Args:
        snapshot (PreparedInstructions): Effective instruction state after the revision.
        content (str): Immutable developer item to replay at the dispatch boundary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot: PreparedInstructions
    content: str

    @classmethod
    def compare(
        cls, previous: PreparedInstructions, current: PreparedInstructions
    ) -> "InstructionRevision | None":
        """Build a replacement for changed sections and explicitly deactivate removed sources.

        Args:
            previous (PreparedInstructions): Last committed effective instructions.
            current (PreparedInstructions): Current canonical instruction snapshot.

        Returns:
            InstructionRevision | None: Effective change, if any.
        """
        old = {part.identity: part.digest for part in previous.sections}
        new = {part.identity: part for part in current.sections}
        replacements = [
            (key, part.render()) for key, part in new.items() if old.get(key) != part.digest
        ]
        removed = [key for key in old if key not in new]
        order = list(new)
        if not replacements and not removed and list(old) == order:
            return None
        update = json.dumps(
            {"replace": replacements, "deactivate": removed, "order": order},
            ensure_ascii=False,
        )
        return cls(
            snapshot=current,
            content=f"<instruction_revision>\n{update}\n</instruction_revision>",
        )


class ModelDispatch(BaseModel):
    """Bind model-only updates to one logical dispatch.

    Args:
        id (str): Stable identity of the admitted request.
        window_id (str): Window owning its input prefix.
        position (int): Exclusive canonical index after which the updates first apply.
        revision (InstructionRevision | None): Committed instruction change.
        reasoning (ReasoningConfiguration | None): Committed provider configuration update.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    revision: InstructionRevision | None = None
    reasoning: ReasoningConfiguration | None = None

    def model_items(self, *, inline: bool) -> Iterable[DeveloperInput | ReasoningConfiguration]:
        """Yield updates to replay at the original dispatch boundary.

        Args:
            inline (bool): Whether instruction revisions are valid inline input for this window.

        Yields:
            DeveloperInput | ReasoningConfiguration: One model-only input item.
        """
        if inline and self.revision is not None:
            yield DeveloperInput(self.revision.content)
        if self.reasoning is not None:
            yield self.reasoning
