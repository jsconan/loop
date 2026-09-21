"""Define the sandbox-only ordinary execution service boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from pydantic import TypeAdapter

from .contracts import SandboxExecutionRequest
from .results import ExecutionResult, InfrastructureFailure
from .state_machine import AttemptState, AttemptStateMachine


class SandboxAdapter(Protocol):
    """Execute an authorized sandbox request without host-execution authority."""

    def execute(
        self,
        request: SandboxExecutionRequest,
        observer: AttemptObserver,
        cancellation: Callable[[], bool],
    ) -> ExecutionResult:
        """Run one sandbox request and return its closed result.

        Args:
            request (SandboxExecutionRequest): Validated authorized sandbox request.
            observer (AttemptObserver): Service-owned lifecycle observer.
            cancellation (Callable[[], bool]): Predicate requesting cancellation.

        Returns:
            ExecutionResult: Closed sandbox result.
        """


class AttemptObserver:
    """Expose only valid service-owned attempt lifecycle notifications.

    Args:
        attempt (AttemptStateMachine): Service-owned state machine for one request.
    """

    _attempt: AttemptStateMachine

    def __init__(self, attempt: AttemptStateMachine) -> None:
        self._attempt = attempt

    def backend_attested(self) -> None:
        """Record successful backend attestation before process creation."""
        self._attempt.transition(AttemptState.ATTESTED)

    def process_started(self) -> None:
        """Record successful sandbox process creation."""
        self._attempt.transition(AttemptState.RUNNING)


_SANDBOX_REQUEST_ADAPTER = TypeAdapter(SandboxExecutionRequest)


class ExecutionService:
    """Coordinate ordinary sandbox requests through one injected sandbox adapter.

    Args:
        adapter (SandboxAdapter): Attested sandbox implementation selected for the request.
    """

    adapter: SandboxAdapter

    def __init__(self, adapter: SandboxAdapter) -> None:
        """Initialize the service with the sole ordinary execution boundary."""
        self.adapter = adapter

    def execute(
        self,
        request: SandboxExecutionRequest,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> ExecutionResult:
        """Execute one sandbox request and terminate its state on every adapter outcome.

        Args:
            request (SandboxExecutionRequest): Already-authorized sandbox request.
            cancellation (Callable[[], bool]): Predicate requesting sandbox cancellation.

        Returns:
            ExecutionResult: The adapter's closed sandbox result.
        """
        sandbox_request = _SANDBOX_REQUEST_ADAPTER.validate_python(request)
        attempt = AttemptStateMachine()
        attempt.transition(AttemptState.AUTHORIZED)
        observer = AttemptObserver(attempt)
        try:
            result = self.adapter.execute(sandbox_request, observer, cancellation)
        except Exception:  # noqa: BLE001 - this trust boundary must fail closed.
            result = InfrastructureFailure(
                request_id=sandbox_request.request_id,
                diagnostic_id=f"diag_{uuid4().hex}",
            )
        attempt.finish(result)
        return result
