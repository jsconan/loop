"""Expose active context compaction through user commands."""

from ..backend import BackendError
from ..commands import CommandContext, CommandRegistration
from ..errors import Problem, ProblemException
from ..instructions import InstructionBudgetExceededError
from ..model_selection import ModelSelectionError
from .compaction import ContextCompaction


class CompactionCommands:
    """Expose context compaction through an interactive command.

    Args:
        compaction (ContextCompaction): Active conversation compaction feature.
    """

    _compaction: ContextCompaction

    def __init__(self, compaction: ContextCompaction) -> None:
        self._compaction = compaction

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the compaction command registration.

        Returns:
            tuple[CommandRegistration, ...]: Context compaction command.
        """
        return (CommandRegistration(self.compact_async, name="compact"),)

    async def compact_async(self, context: CommandContext) -> None:
        """Compact the active session context.

        Args:
            context (CommandContext): Dispatch mode for this invocation.

        Raises:
            ProblemException: If the backend or request preparation fails.
            asyncio.CancelledError: If asynchronous compaction is cancelled.
        """
        try:
            if context.execution_mode == "async":
                await self._compaction.compact_async()
            else:
                self._compaction.compact()
        except BackendError as error:
            raise ProblemException(error.to_problem()) from error
        except (InstructionBudgetExceededError, ModelSelectionError) as error:
            raise ProblemException(
                Problem.from_exception(
                    error,
                    code="compaction.failed",
                    title="Could not compact session",
                    operation="compact",
                )
            ) from error
