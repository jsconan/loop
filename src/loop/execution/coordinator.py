"""Coordinate sandbox execution and separately approved host retries."""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import get_clock_info
from typing import Protocol
from uuid import uuid4

from .. import constants
from ..interaction import Interaction
from ..permissions import (
    Action,
    ApprovalChoice,
    Decision,
    Operation,
    PermissionManager,
    ProcessBoundary,
    ProcessTarget,
)
from ..telemetry import telemetry_audit
from ..utils import (
    ProcessCaptureStatus,
    kill_process_group,
    read_bounded_stream,
    sha256_digest,
    supervise_process,
)
from ..utils.process import is_finite_deadline
from .sandbox import CommandProcessResult, SandboxBackend, SandboxOutcome, SandboxRequest


def _check_command_deadlines(request: SandboxRequest, timeout: float | None) -> None:
    """Reject invalid request deadlines or optional execution budgets before authorization."""
    if not is_finite_deadline(request.deadline) or (
        timeout is not None and (not is_finite_deadline(timeout) or timeout <= 0)
    ):
        raise ValueError("Command deadline must be finite and execution timeout positive.")


def command_deadline(timeout: float, preparation_seconds: float = 0) -> float:
    """Calculate a runtime deadline while retaining the native preparation budget.

    Args:
        timeout (float): Positive finite runtime budget in seconds.
        preparation_seconds (float): Separate preparation budget, or zero for none.

    Returns:
        float: Absolute monotonic deadline for the combined budgets.

    Raises:
        ValueError: The deadline is not finite or loses the preparation budget to rounding.
    """
    preparation_deadline = time.monotonic() + preparation_seconds
    deadline = preparation_deadline + timeout
    if not is_finite_deadline(deadline):
        raise ValueError("Renewed command deadline must be finite.")
    if preparation_seconds and abs((deadline - timeout) - preparation_deadline) > max(
        math.ulp(preparation_deadline), get_clock_info("monotonic").resolution
    ):
        raise ValueError("Renewed command deadline cannot represent the preparation budget.")
    return deadline


def _renew_command_deadline(
    request: SandboxRequest, timeout: float | None, *, preparation_seconds: float = 0
) -> SandboxRequest:
    """Renew an approved request's budget without admitting an overflowed deadline."""
    if timeout is None:
        return request
    deadline = command_deadline(timeout, preparation_seconds)
    return replace(
        request,
        deadline=deadline,
        runtime_timeout=timeout if preparation_seconds else request.runtime_timeout,
    )


class HostCommandExecutor(Protocol):
    """Run an explicitly approved host command through the host process supervisor."""

    def run_host_command(
        self,
        request: HostCommandRequest,
        *,
        cancellation_event: threading.Event | None = None,
    ) -> CommandProcessResult:
        """Run one exact, separately approved host request.

        Args:
            request (HostCommandRequest): Exact approved command and effective environment.
            cancellation_event (threading.Event | None): Optional signal requesting child cleanup.

        Returns:
            CommandProcessResult: Bounded host process result.
        """


class LocalHostCommandExecutor:
    """Supervise only a separately approved, unrestricted host shell request."""

    def run_host_command(
        self,
        request: HostCommandRequest,
        *,
        cancellation_event: threading.Event | None = None,
    ) -> CommandProcessResult:
        """Capture one approved host process within its bound deadline.

        Args:
            request (HostCommandRequest): Exact host command approved by the user.
            cancellation_event (threading.Event | None): Optional signal requesting child cleanup.

        Returns:
            CommandProcessResult: Bounded process output and exit or timeout outcome.

        Raises:
            ValueError: The command deadline is not finite.
        """
        if not is_finite_deadline(request.deadline):
            raise ValueError("Command deadline must be finite.")
        if time.monotonic() >= request.deadline:
            return CommandProcessResult(SandboxOutcome.TIMED_OUT)
        if cancellation_event is not None and cancellation_event.is_set():
            return CommandProcessResult(SandboxOutcome.CANCELLED)
        process = subprocess.Popen(
            [constants.SHELL_COMMAND_PATH, constants.SHELL_COMMAND_OPTION, request.source],
            cwd=request.cwd,
            env=dict(request.environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            close_fds=True,
            start_new_session=True,
        )
        capture = supervise_process(
            process,
            request.deadline,
            terminate=kill_process_group,
            reader=read_bounded_stream,
            thread_factory=threading.Thread,
            cancel_event=cancellation_event,
            preserve_newlines=True,
        )
        if capture.status is ProcessCaptureStatus.UNAVAILABLE:
            incomplete = capture.stdout_discarded is None or capture.stderr_discarded is None
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                stdout=capture.stdout,
                stderr=capture.stderr,
                stdout_discarded=capture.stdout_discarded,
                stderr_discarded=capture.stderr_discarded,
                detail="Host output capture failed."
                if incomplete
                else "Host child pipes unavailable.",
                possible_effects=True,
            )
        return CommandProcessResult(
            SandboxOutcome(capture.status.value),
            exit_code=capture.exit_code,
            stdout=capture.stdout,
            stderr=capture.stderr,
            stdout_discarded=capture.stdout_discarded,
            stderr_discarded=capture.stderr_discarded,
            possible_effects=capture.status is not ProcessCaptureStatus.COMPLETED,
        )


@dataclass(frozen=True)
class HostCommandRequest:
    """Carry only the separately authorized unrestricted process context.

    Args:
        source (str): Exact translated POSIX shell source bound to the host warning.
        cwd (Path): Exact bound working directory.
        workspace (Path): Exact bound workspace root.
        workspace_id (str): Exact selected workspace identity.
        environment (tuple[tuple[str, str], ...]): Exact effective environment bound to approval.
        deadline (float): Fresh monotonic host execution deadline.
    """

    source: str
    cwd: Path
    workspace: Path
    workspace_id: str
    environment: tuple[tuple[str, str], ...]
    deadline: float

    @classmethod
    def from_sandbox(cls, request: SandboxRequest) -> HostCommandRequest:
        """Create a separate host request from freshly checked retry context.

        Args:
            request (SandboxRequest): Fresh retry context matching the issued offer.

        Returns:
            HostCommandRequest: Minimal exact host process request.
        """
        return cls(
            request.execution_source,
            request.cwd,
            request.workspace,
            request.workspace_id,
            request.environment,
            request.deadline,
        )


@dataclass(frozen=True)
class HostCommandOffer:
    """Present a single-use opportunity to request unrestricted execution.

    Args:
        token (str): Unpredictable identifier for this offer.
        request (SandboxRequest): Exact command context offered for retry.
        failure (CommandProcessResult | None): Sandbox failure behind the offer, or None for an
            explicitly requested host command.
        warning (str): Full user-facing host execution warning.
        tool_id (str): Original execution tool identity used for policy checks. Defaults to
            ``run_command`` for callers constructing an offer directly.
    """

    token: str
    request: SandboxRequest
    failure: CommandProcessResult | None
    warning: str
    tool_id: str = "run_command"


@dataclass(frozen=True)
class CommandAttempt:
    """Return an authorized sandbox result and any reviewable host offer.

    Args:
        result (CommandProcessResult | None): Sandboxed attempt, or None when permission was denied.
        offer (HostCommandOffer | None): Host retry offer after a no-effects boundary failure.
    """

    result: CommandProcessResult | None
    offer: HostCommandOffer | None


class CommandExecutionCoordinator:
    """Enforce distinct sandbox and host approval paths for one application session.

    Args:
        backend (SandboxBackend): Native sandbox implementation selected by the application.
        host_executor (HostCommandExecutor): Host supervisor used after explicit approval.
        permissions (PermissionManager): Dedicated sandbox command permission authority.
        interaction (Interaction | None): Actual user prompt surface, or None in headless mode.
        auto_approval (bool): Whether ordinary prompts use automatic input. This does not
            authorize host execution; trusted YOLO is a distinct permission-manager decision.
        host_audit (Callable[..., None] | None): Thread-safe host audit callback, or None to use
            the permission manager.
    """

    _backend: SandboxBackend
    _host_executor: HostCommandExecutor
    _permissions: PermissionManager
    _interaction: Interaction | None
    _auto_approval: bool
    _offers: dict[str, HostCommandOffer]
    _host_audit: Callable[..., None]

    def __init__(
        self,
        backend: SandboxBackend,
        host_executor: HostCommandExecutor,
        permissions: PermissionManager,
        interaction: Interaction | None,
        *,
        auto_approval: bool = False,
        host_audit: Callable[..., None] | None = None,
    ) -> None:
        self._backend = backend
        self._host_executor = host_executor
        self._permissions = permissions
        self._interaction = interaction
        self._auto_approval = auto_approval
        self._offers = {}
        self._host_audit = host_audit or permissions.record_host_boundary

    def run_sandboxed(
        self,
        request: SandboxRequest,
        tool_id: str,
        *,
        prelaunch_failure: CommandProcessResult | None = None,
        reason: str | None = None,
        execution_timeout: float | None = None,
        cancellation_event: threading.Event | None = None,
        admission_cancellation_event: threading.Event | None = None,
        on_admitted: Callable[[], bool | None] | None = None,
        output_observer: Callable[[str, str], None] | None = None,
        terminal_input: Callable[[], bytes | None] | None = None,
        egress_authority: object | None = None,
    ) -> CommandAttempt:
        """Run an authorized request or offer retry for a no-effects boundary failure.

        Args:
            request (SandboxRequest): Exact approved command, policy, and path identities.
            tool_id (str): Registered execution tool identity used for permission evaluation.
            prelaunch_failure (CommandProcessResult | None): Typed startup capability failure to
                return without invoking the backend, after authorization and path validation.
            reason (str | None): Explanation for additional sandbox authority.
            execution_timeout (float | None): Runtime seconds starting at native launch, with a
                separate 60-second preparation ceiling after approval. None preserves the
                caller's absolute deadline.
            cancellation_event (threading.Event | None): Optional run cancellation signal.
            admission_cancellation_event (threading.Event | None): Original run stop signal
                checked until a detached job is admitted; later signals do not cancel that job.
            on_admitted (Callable[[], bool | None] | None): Callback after approval and final
                prelaunch validation. Returning False refuses native admission.
            output_observer (Callable[[str, str], None] | None): Bounded native output sink.
            terminal_input (Callable[[], bytes | None] | None): Transient control frames for an
                approved receiver-owned terminal.
            egress_authority (object | None): Thread-safe restricted egress authority, or None
                to use the permission manager.

        Returns:
            CommandAttempt: Sandboxed result and optional unexecuted host offer.

        Raises:
            ValueError: A deadline is not finite or cannot preserve the preparation budget.
        """
        _check_command_deadlines(request, execution_timeout)
        if admission_cancellation_event is not None and admission_cancellation_event.is_set():
            return CommandAttempt(CommandProcessResult(SandboxOutcome.CANCELLED), None)
        reason_argument = {"reason": reason} if reason is not None else {}
        approved_revision: list[int] = []
        if not self._permissions.authorize_sandboxed_command(
            request,
            tool_id=tool_id,
            interaction=self._interaction,
            approved_revision=approved_revision,
            **reason_argument,
        ):
            return CommandAttempt(None, None)
        request = _renew_command_deadline(
            request,
            execution_timeout,
            preparation_seconds=constants.COMMAND_PREPARATION_SECONDS,
        )
        if not request.paths_are_current():
            result = CommandProcessResult(
                SandboxOutcome.STALE,
                detail="Approved path identity changed before launch.",
            )
        elif time.monotonic() >= request.deadline:
            result = CommandProcessResult(
                SandboxOutcome.TIMED_OUT, detail="Command deadline expired before launch."
            )
        elif (
            cancellation_event is not None
            and cancellation_event.is_set()
            or admission_cancellation_event is not None
            and admission_cancellation_event.is_set()
        ):
            result = CommandProcessResult(SandboxOutcome.CANCELLED)
        elif prelaunch_failure is not None:
            result = prelaunch_failure
        elif len(approved_revision) != 1:
            result = CommandProcessResult(
                SandboxOutcome.INVALID, detail="Approved policy revision is unavailable."
            )
        elif not self._permissions.sandbox_policy_revision_current(approved_revision[0]):
            result = CommandProcessResult(
                SandboxOutcome.STALE, detail="Permission policy changed before launch."
            )
        else:
            if on_admitted is not None and on_admitted() is False:
                result = CommandProcessResult(
                    SandboxOutcome.STALE, detail="Command authority changed before launch."
                )
            else:
                output_argument = (
                    {"output_observer": output_observer} if output_observer is not None else {}
                )
                if terminal_input is not None:
                    output_argument["terminal_input"] = terminal_input
                if request.egress_origins:
                    result = self._backend.run(
                        request,
                        policy_authority=egress_authority or self._permissions,
                        approved_revision=approved_revision[0],
                        **(
                            {"cancellation_event": cancellation_event} if cancellation_event else {}
                        ),
                        **output_argument,
                    )
                else:
                    result = self._backend.run(
                        request,
                        **(
                            {"cancellation_event": cancellation_event} if cancellation_event else {}
                        ),
                        **output_argument,
                    )
        telemetry_audit(
            "sandbox.completed",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            outcome=result.outcome.value,
        )
        if (
            result.outcome not in {SandboxOutcome.UNAVAILABLE, SandboxOutcome.DENIED}
            or result.possible_effects
            or request.egress_origins
        ):
            return CommandAttempt(result, None)
        warning = PermissionManager.host_retry_prompt(
            request.source,
            result.outcome.value,
            result.exit_code,
            result.possible_effects,
            result.failure_context,
        )
        if request.execution_source != request.source:
            warning += f"\nExecuted command: {json.dumps(request.execution_source)}"
        offer = HostCommandOffer(str(uuid4()), request, result, warning, tool_id)
        self._offers[offer.token] = offer
        telemetry_audit(
            "host_command.offered",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            command_hash=sha256_digest(request.source),
            offer_id=offer.token,
            failure=result.outcome.value,
        )
        self._record_host_boundary("host_command.offered", offer, outcome=result.outcome.value)
        return CommandAttempt(result, offer)

    def request_host_command(
        self,
        request: SandboxRequest,
        tool_id: str,
        reason: str,
        *,
        execution_timeout: float | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> CommandProcessResult | None:
        """Request one explicit host run with fresh consent or an exact trusted grant.

        Args:
            request (SandboxRequest): Exact command and bound process context.
            tool_id (str): Registered tool identity used for host policy evaluation.
            reason (str): Model-supplied explanation for needing unrestricted execution.
            execution_timeout (float | None): Optional execution budget after approval.
            cancellation_event (threading.Event | None): Optional cancellation signal.

        Returns:
            CommandProcessResult | None: Host result, or None if authorization fails.

        Raises:
            ValueError: If the explanation or deadline is invalid.
        """
        _check_command_deadlines(request, execution_timeout)
        if not reason.strip():
            raise ValueError("An unrestricted host request requires a reason.")
        warning = (
            f"Command: {json.dumps(request.source)}\n"
            f"Executed command: {json.dumps(request.execution_source)}\n"
            f"Working directory: {json.dumps(str(request.cwd))}\n"
            f"Reason: {json.dumps(reason)}\n"
            "The sandbox restrictions do not apply. The command runs with the host user's "
            "authority, subject to independent OS and outer-environment restrictions. It can "
            "read private files, change files outside the workspace, and use the network. "
            "Previous attempts may already have had effects; running again may repeat them."
        )
        offer = HostCommandOffer(str(uuid4()), request, None, warning, tool_id)
        self._offers[offer.token] = offer
        self._record_host_boundary("host_command.offered", offer)
        return self.retry_host_command(
            offer,
            request,
            execution_timeout=execution_timeout,
            cancellation_event=cancellation_event,
        )

    def retry_host_command(
        self,
        offer: HostCommandOffer,
        current_request: SandboxRequest,
        *,
        execution_timeout: float | None = None,
        cancellation_event: threading.Event | None = None,
        on_authorized: Callable[[], bool] | None = None,
        on_admitted: Callable[[], None] | None = None,
    ) -> CommandProcessResult | None:
        """Approve an unchanged offer and launch at most one host process.

        Args:
            offer (HostCommandOffer): Previously issued, unused host retry offer.
            current_request (SandboxRequest): Freshly resolved command context at approval time.
            execution_timeout (float | None): Execution budget in seconds to start after host
                approval. Defaults to preserving the caller's absolute deadline.
            cancellation_event (threading.Event | None): Optional run cancellation signal.
            on_authorized (Callable[[], bool] | None): Final authority gate before recording
                approval; False refuses the host attempt.
            on_admitted (Callable[[], None] | None): Notify the owner before host execution.

        Returns:
            CommandProcessResult | None: Host result after approval, or None when no launch occurs.

        Raises:
            ValueError: The request deadline or renewed execution deadline is not finite.
        """
        _check_command_deadlines(current_request, execution_timeout)
        issued = self._offers.pop(offer.token, None)
        if (
            issued != offer
            or self._host_identity(current_request) != self._host_identity(offer.request)
            or not offer.request.paths_are_current()
            or not current_request.paths_are_current()
        ):
            telemetry_audit(
                "host_command.stale",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            self._record_host_boundary("host_command.stale", offer)
            return None
        if cancellation_event is not None and cancellation_event.is_set():
            self._record_host_boundary("host_command.cancelled", offer)
            return None
        host_operations = tuple(
            Operation(
                tool_id=offer.tool_id,
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=(constants.SHELL_COMMAND_PATH, constants.SHELL_COMMAND_OPTION, source),
                    cwd=str(current_request.cwd),
                    boundary=ProcessBoundary.HOST,
                ),
            )
            for source in dict.fromkeys((current_request.source, current_request.execution_source))
        )
        host_policy, policy_revision = self._permissions.evaluate_with_revision(host_operations)
        if offer.failure is not None and self._permissions.yolo:
            telemetry_audit(
                "host_command.denied",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
                source="yolo_requires_separate_explicit_request",
            )
            self._record_host_boundary(
                "host_command.denied", offer, source="yolo_requires_separate_explicit_request"
            )
            return None
        if host_policy.decision is Decision.DENY:
            telemetry_audit(
                "host_command.denied",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
                source="policy",
            )
            self._record_host_boundary("host_command.denied", offer, source="policy")
            return None
        explicit_asks = {
            source
            for source in host_policy.sources
            if host_policy.decision is Decision.ASK
            and source.startswith(("rule:", "managed:human:"))
        }
        managed_human = any(source.startswith("managed:human:") for source in explicit_asks)
        reusable = (
            offer.failure is not None
            and offer.failure.outcome in {SandboxOutcome.UNAVAILABLE, SandboxOutcome.DENIED}
            and not offer.failure.possible_effects
            and not explicit_asks
        )
        rule = self._permissions.matching_host_command_rule(current_request) if reusable else None
        trusted = (
            self._permissions.matching_trusted_host_command_rule(
                current_request.source,
                current_request.cwd,
                offer.tool_id,
                workspace=current_request.workspace,
                environment=current_request.shell_environment(),
            )
            if offer.failure is None
            and current_request.source == current_request.execution_source
            and not managed_human
            else None
        )
        grantable = (
            offer.failure is None
            and offer.tool_id == "request_host_command"
            and current_request.source == current_request.execution_source
            and not managed_human
        )
        if trusted is not None:
            answer = "trusted"
            if self._interaction is not None:
                self._interaction.info(offer.warning)
        elif rule is not None:
            answer = "rule"
        elif self._permissions.yolo:
            answer = "yolo"
            if self._interaction is not None:
                self._interaction.warning(offer.warning)
        elif (
            self._auto_approval
            or self._interaction is None
            or getattr(self._interaction, "supports_fresh_host_approval", False) is not True
        ):
            telemetry_audit(
                "host_command.denied",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
                source="fresh_input_unavailable",
            )
            self._record_host_boundary(
                "host_command.denied", offer, source="fresh_input_unavailable"
            )
            return None
        else:
            self._interaction.info(offer.warning)
            choices, index = self._permissions.host_retry_choices(reusable, grantable)
            answer = self._interaction.prompt(
                "Run this exact command without the OS sandbox?",
                exit_commands=None,
                choices=choices,
                index=index,
            )
        if cancellation_event is not None and cancellation_event.is_set():
            self._record_host_boundary("host_command.cancelled", offer)
            return None
        allowed_answers = {"approve", "session", "rule"} if reusable else {"approve", "trusted"}
        if self._permissions.yolo and offer.failure is None:
            allowed_answers.add("yolo")
        if grantable:
            allowed_answers.update({"trust_session", "trust_workspace", "trust_user"})
        if answer not in allowed_answers:
            telemetry_audit(
                "host_command.denied",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            self._record_host_boundary("host_command.denied", offer, source="interactive")
            return None
        current_request = _renew_command_deadline(current_request, execution_timeout)
        authority_current = (
            current_request.paths_are_current()
            and time.monotonic() < current_request.deadline
            and (
                rule is None
                or self._permissions.matching_host_command_rule(current_request) == rule
            )
            and (
                trusted is None
                or self._permissions.matching_trusted_host_command_rule(
                    current_request.source,
                    current_request.cwd,
                    offer.tool_id,
                    workspace=current_request.workspace,
                    environment=current_request.shell_environment(),
                )
                == trusted
            )
        )
        current_policy, current_revision = self._permissions.evaluate_with_revision(host_operations)
        if (
            not authority_current
            or current_policy.decision is Decision.DENY
            or current_revision != policy_revision
        ):
            telemetry_audit(
                "host_command.stale",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            self._record_host_boundary("host_command.stale", offer)
            return None
        if on_authorized is not None and not on_authorized():
            telemetry_audit(
                "host_command.stale",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            self._record_host_boundary("host_command.stale", offer)
            return None
        if cancellation_event is not None and cancellation_event.is_set():
            self._record_host_boundary("host_command.cancelled", offer)
            return None
        if answer == "session":
            rule = self._permissions.remember_host_command_rule_if_current(
                current_request, policy_revision
            )
            if rule is None:
                telemetry_audit(
                    "host_command.stale",
                    workspace_id=offer.request.workspace_id,
                    attempt_id=offer.request.attempt_id,
                    offer_id=offer.token,
                )
                self._record_host_boundary("host_command.stale", offer)
                return None
        elif answer.startswith("trust_"):
            scope = ApprovalChoice(answer.removeprefix("trust_"))
            installation = self._permissions.remember_trusted_host_command_rule_if_current(
                current_request, scope, policy_revision
            )
            if installation is None:
                self._record_host_boundary("host_command.stale", offer)
                return None
            trusted, policy_revision = installation
        final_policy, final_revision = self._permissions.evaluate_with_revision(host_operations)
        if (
            final_policy.decision is Decision.DENY
            or final_revision != policy_revision
            or not current_request.paths_are_current()
            or (
                trusted is not None
                and self._permissions.matching_trusted_host_command_rule(
                    current_request.source,
                    current_request.cwd,
                    offer.tool_id,
                    workspace=current_request.workspace,
                    environment=current_request.shell_environment(),
                )
                != trusted
            )
        ):
            telemetry_audit(
                "host_command.stale",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            self._record_host_boundary("host_command.stale", offer)
            return None
        telemetry_audit(
            "host_command.approved",
            workspace_id=offer.request.workspace_id,
            attempt_id=offer.request.attempt_id,
            offer_id=offer.token,
            source=answer if answer in {"rule", "trusted", "yolo"} else "interactive",
            rule_id=(trusted or rule).id if trusted is not None or rule is not None else None,
        )
        self._record_host_boundary(
            "host_command.approved",
            offer,
            source=answer if answer in {"rule", "trusted", "yolo"} else "interactive",
            rule_id=(trusted or rule).id if trusted is not None or rule is not None else None,
        )
        try:
            host_request = HostCommandRequest.from_sandbox(current_request)
            if on_admitted is not None:
                on_admitted()
            if cancellation_event is not None and cancellation_event.is_set():
                self._record_host_boundary("host_command.cancelled", offer)
                return None
            result = (
                self._host_executor.run_host_command(
                    host_request,
                    cancellation_event=cancellation_event,
                )
                if cancellation_event is not None
                else self._host_executor.run_host_command(host_request)
            )
        except Exception:
            telemetry_audit(
                "host_command.execution_failed",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            self._record_host_boundary("host_command.execution_failed", offer)
            raise
        telemetry_audit(
            "host_command.executed",
            workspace_id=offer.request.workspace_id,
            attempt_id=offer.request.attempt_id,
            offer_id=offer.token,
        )
        self._record_host_boundary(
            "host_command.executed",
            offer,
            outcome=result.outcome.value if isinstance(result, CommandProcessResult) else None,
        )
        return result

    def _record_host_boundary(
        self,
        event_name: str,
        offer: HostCommandOffer,
        *,
        source: str | None = None,
        rule_id: str | None = None,
        outcome: str | None = None,
    ) -> None:
        """Write a minimized durable record for one host offer or execution boundary."""
        self._host_audit(
            event_name,
            offer.request.execution_source,
            attempt_id=offer.request.attempt_id,
            offer_id=offer.token,
            source=source,
            rule_id=rule_id,
            outcome=outcome,
        )

    @staticmethod
    def _host_identity(request: SandboxRequest) -> SandboxRequest:
        """Bind the complete reviewed context except its renewable deadline and attempt ID."""
        return replace(request, deadline=0.0, attempt_id="", runtime_timeout=None)
