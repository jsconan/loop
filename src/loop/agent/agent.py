"""Define an LLM agent independently from conversation orchestration."""

from ..backend import Backend
from ..permissions import PermissionManager
from ..skills import InstructionsManager
from ..tooling import ToolRegistry


class Agent:
    """Define the model-facing capabilities of one agent.

    Args:
        name (str): Human-readable identity for the agent.
        backend (Backend): Backend used to produce model responses.
        instructions_manager (InstructionsManager): Manager providing dynamic instructions.
        tool_registry (ToolRegistry): Agent-scoped tools exposed to the model.
        permission_manager (PermissionManager): Manager authorizing agent tool calls.

    Raises:
        ValueError: If ``name`` is empty or contains only whitespace.
    """

    _name: str
    _backend: Backend
    _instructions_manager: InstructionsManager
    _tool_registry: ToolRegistry
    _permission_manager: PermissionManager

    def __init__(
        self,
        name: str,
        *,
        backend: Backend,
        instructions_manager: InstructionsManager,
        tool_registry: ToolRegistry,
        permission_manager: PermissionManager,
    ) -> None:
        if not name.strip():
            raise ValueError("Agent name must not be empty.")
        self._name = name
        self._backend = backend
        self._instructions_manager = instructions_manager
        self._tool_registry = tool_registry
        self._permission_manager = permission_manager

    @property
    def name(self) -> str:
        """Return the human-readable agent identity.

        Returns:
            str: Agent name.
        """
        return self._name

    @property
    def backend(self) -> Backend:
        """Return the backend used for model requests.

        Returns:
            Backend: Configured response backend.
        """
        return self._backend

    @property
    def instructions_manager(self) -> InstructionsManager:
        """Return the agent's dynamic instruction manager.

        Returns:
            InstructionsManager: Configured instruction manager.
        """
        return self._instructions_manager

    @property
    def tool_registry(self) -> ToolRegistry:
        """Return the tools available to the agent.

        Returns:
            ToolRegistry: Agent-scoped tool registry.
        """
        return self._tool_registry

    @property
    def permission_manager(self) -> PermissionManager:
        """Return the policy manager guarding agent tool calls.

        Returns:
            PermissionManager: Configured permission manager.
        """
        return self._permission_manager
