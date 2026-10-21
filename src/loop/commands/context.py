"""Define command invocation context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..errors import Problem, ProblemException
from ..models import CommandCapability

if TYPE_CHECKING:
    from ..interaction import Interaction
    from .command import Command


@dataclass(frozen=True)
class CommandContext:
    """Provide runtime metadata and interaction services to a command.

    Args:
        name (str): Public name of the command being invoked.
        interaction (Interaction): Service used to communicate with the user.
        available_commands (tuple[Command, ...] | None): Context-filtered catalog available to the
            command, used by help. Defaults to ``None`` for the full registry.
        capabilities (frozenset[CommandCapability] | None): Capabilities of the current runtime
            context. ``None`` means that command availability is unrestricted.
        execution_mode (Literal["sync", "async"]): Dispatch API owning the command invocation.
            Defaults to ``"sync"`` for contexts assembled outside a manager.
        _blocking_runner (Callable[..., Awaitable[Any]] | None): Manager-owned bounded worker
            used by async commands for synchronous leaf operations. Set by ``CommandManager``.
    """

    name: str
    interaction: Interaction
    available_commands: tuple[Command, ...] | None = None
    capabilities: frozenset[CommandCapability] | None = None
    execution_mode: Literal["sync", "async"] = "sync"
    _blocking_runner: Callable[..., Awaitable[Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    async def run_blocking(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous leaf operation through the command manager's bounded worker.

        In a synchronous command invocation, call the operation directly. Async command
        invocations use the manager-owned worker so local I/O does not block the owner loop.

        Args:
            function (Callable[..., Any]): Synchronous operation to invoke.
            *args (Any): Positional arguments passed to the operation.
            **kwargs (Any): Keyword arguments passed to the operation.

        Returns:
            Any: Value returned by the synchronous operation.

        Raises:
            ProblemException: If async dispatch has no manager-owned worker available.
            asyncio.CancelledError: If the caller is cancelled after the worker settles.
        """
        if self.execution_mode == "sync":
            return function(*args, **kwargs)
        if self._blocking_runner is None:
            raise ProblemException(
                Problem(
                    code="command.blocking_dispatch_unavailable",
                    title="Command worker unavailable",
                    detail=(
                        "This asynchronous command was not dispatched by a CommandManager and "
                        "cannot safely run synchronous work on the event loop."
                    ),
                    operation=self.name,
                )
            )
        return await self._blocking_runner(function, *args, **kwargs)
