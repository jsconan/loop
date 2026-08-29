"""Define immutable model-facing agent configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from html import escape
from importlib.resources import files

from ..constants import DEFAULT_AGENT_INSTRUCTIONS_SOURCE, DEFAULT_AGENT_INSTRUCTIONS_VERSION
from ..tooling import ToolRegistry
from ..utils import sha256_digest


@dataclass(frozen=True)
class AgentInstructions:
    """Define the intrinsic, versioned instructions of an agent.

    Args:
        content (str): Model-facing instruction body without its composition wrapper.
        version (str): Stable version used for provenance and diagnostics.
        source (str): Logical producer or canonical location of the instructions.

    Raises:
        ValueError: If content, version, or source is empty or whitespace-only.
    """

    content: str
    version: str
    source: str

    def __post_init__(self) -> None:
        for name, value in (
            ("content", self.content),
            ("version", self.version),
            ("source", self.source),
        ):
            if not value.strip():
                raise ValueError(f"Agent instructions {name} must not be empty.")

    @classmethod
    def default(cls) -> AgentInstructions:
        """Load Loop's bundled agent instructions.

        Returns:
            AgentInstructions: Versioned instructions packaged with Loop.
        """
        package, resource_path = DEFAULT_AGENT_INSTRUCTIONS_SOURCE.split("/", maxsplit=1)
        resource = files(package).joinpath(resource_path)
        return cls(
            content=resource.read_text(encoding="utf-8").strip(),
            version=DEFAULT_AGENT_INSTRUCTIONS_VERSION,
            source=DEFAULT_AGENT_INSTRUCTIONS_SOURCE,
        )

    @property
    def digest(self) -> str:
        """Return the instruction body's stable content digest.

        Returns:
            str: SHA-256 hexadecimal digest.
        """
        return sha256_digest(self.content)

    def render(self) -> str:
        """Render the instructions with provenance metadata.

        Returns:
            str: XML-like instruction section including version and source.
        """
        return (
            f'<agent_instructions version="{escape(self.version)}" '
            f'source="{escape(self.source)}">\n'
            f"{self.content}\n"
            "</agent_instructions>"
        )


@dataclass(frozen=True)
class AgentIdentity:
    """Define the human-readable identity of an agent.

    Args:
        name (str): Non-empty name presented to the model and users.
        description (str | None): Optional concise description of the agent's role.

    Raises:
        ValueError: If the name is empty or contains only whitespace.
    """

    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty.")

    def render(self) -> str:
        """Render the identity as a distinct instruction section.

        Returns:
            str: XML-like identity section safe for instruction composition.
        """
        name = escape(self.name.strip())
        description = (
            f"\n{escape(self.description.strip())}"
            if self.description and self.description.strip()
            else ""
        )
        return f'<agent_identity name="{name}">{description}\n</agent_identity>'


@dataclass(frozen=True)
class Agent:
    """Define a reusable agent independently from execution infrastructure.

    Args:
        identity (AgentIdentity | str): Agent identity or a non-empty shorthand name.
        instructions (AgentInstructions): Intrinsic model-facing behavior. Defaults to Loop's
            bundled instructions.
        tools (ToolRegistry): Tools exposed to the model. Defaults to an empty registry.
    """

    identity: AgentIdentity | str
    instructions: AgentInstructions = field(default_factory=AgentInstructions.default)
    tools: ToolRegistry = field(default_factory=ToolRegistry)

    def __post_init__(self) -> None:
        if isinstance(self.identity, str):
            object.__setattr__(self, "identity", AgentIdentity(self.identity))

    @property
    def name(self) -> str:
        """Return the human-readable agent name.

        Returns:
            str: Agent name.
        """
        return self.identity.name

    def render(self) -> str:
        """Render intrinsic instructions with the agent identity.

        Returns:
            str: Rendered intrinsic instructions containing escaped identity fields.
        """
        description = escape(self.identity.description.strip()) if self.identity.description else ""
        instructions = replace(
            self.instructions,
            content=(
                self.instructions.content.replace("{{name}}", escape(self.name.strip())).replace(
                    "{{description}}", description
                )
            ),
        )
        return instructions.render()
