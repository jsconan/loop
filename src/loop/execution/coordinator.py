"""Coordinate sandbox execution and separately approved host retries."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .. import constants
from ..interaction import Interaction
from ..permissions import HostCommandRule, PermissionManager
from ..telemetry import telemetry_audit
from ..utils import (
    ProcessCaptureStatus,
    kill_process_group,
    read_bounded_stream,
    sha256_digest,
    supervise_process,
)
from .sandbox import CommandProcessResult, SandboxBackend, SandboxOutcome, SandboxRequest

_DENIED_OPERATION = re.compile(r"\b(file-(?:read|write)[\w-]*|network-[\w-]+)\s+(\S+)")
_NETWORK_ACTION = re.compile(
    r"\b(?:connect|bind|listen|send(?:to)?|recv(?:from)?|socket)\b", re.IGNORECASE
)
_PERMISSION_FAILURE_MARKERS = ("permission denied", "operation not permitted", "access denied")


def _denied_path_in_error_output(request: SandboxRequest, target: str, output: str) -> bool:
    """Match a denied path against absolute, relative, or symlinked error spellings."""
    resolved_target = Path(target).resolve(strict=False)
    for line in output.splitlines():
        if not any(marker in line.casefold() for marker in _PERMISSION_FAILURE_MARKERS):
            continue
        if target in line:
            return True
        for spelling in re.findall(r"[A-Za-z0-9_./@+-]+", line)[
            : constants.MAX_COMMAND_PATH_LENGTH + 1
        ]:
            candidate = Path(spelling)
            if not candidate.is_absolute():
                candidate = request.cwd / candidate
            try:
                if (
                    str(candidate.resolve(strict=False)).casefold()
                    == str(resolved_target).casefold()
                ):
                    return True
            except (OSError, RuntimeError):
                continue
    return False


def _denial_supports_host_offer(request: SandboxRequest, result: CommandProcessResult) -> bool:
    """Require a child failure related to the observed denied operation."""
    match = _DENIED_OPERATION.search(result.observed_denial)
    if match is None:
        return False
    operation, target = match.groups()
    output = f"{result.stdout}\n{result.stderr}"
    permission_failure = any(marker in output.casefold() for marker in _PERMISSION_FAILURE_MARKERS)
    if operation.startswith("network-"):
        if target.startswith("/"):
            lines = output.splitlines()
            return any(
                _NETWORK_ACTION.search(line)
                and target in line
                and any(
                    any(marker in candidate.casefold() for marker in _PERMISSION_FAILURE_MARKERS)
                    for candidate in lines[index : index + 4]
                )
                for index, line in enumerate(lines)
            )
        port = target.rpartition(":")[2]
        if not port.isdecimal():
            return False
        port_pattern = re.compile(rf"(?::|\bport\s+){re.escape(port)}\b", re.IGNORECASE)
        return any(
            any(marker in line.casefold() for marker in _PERMISSION_FAILURE_MARKERS)
            and port_pattern.search(line)
            and _NETWORK_ACTION.search(line)
            for line in output.splitlines()
        )
    if (
        not permission_failure
        or not target.startswith("/")
        or not _denied_path_in_error_output(request, target, output)
    ):
        return False
    if operation.startswith("file-write") and request.workspace not in request.write_roots:
        return not Path(target).is_relative_to(request.workspace)
    return True


class HostCommandExecutor(Protocol):
    """Run an explicitly approved host command through the host process supervisor."""

    def run_host_command(self, request: HostCommandRequest) -> CommandProcessResult:
        """Run one exact, separately approved host request.

        Args:
            request (HostCommandRequest): Exact approved command and effective environment.

        Returns:
            CommandProcessResult: Bounded host process result.
        """


class LocalHostCommandExecutor:
    """Supervise only a separately approved, unrestricted host shell request."""

    def run_host_command(self, request: HostCommandRequest) -> CommandProcessResult:
        """Capture one approved host process within its bound deadline.

        Args:
            request (HostCommandRequest): Exact host command approved by the user.

        Returns:
            CommandProcessResult: Bounded process output and exit or timeout outcome.
        """
        if time.monotonic() >= request.deadline:
            return CommandProcessResult(SandboxOutcome.TIMED_OUT)
        process = subprocess.Popen(
            ["/bin/sh", "-c", request.source],
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
        )
        if capture.status is ProcessCaptureStatus.UNAVAILABLE:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE, detail="Host child pipes unavailable."
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
        failure (CommandProcessResult): Sandbox failure or completed attempt with an observed
            denial behind the offer.
        warning (str): Full user-facing host execution warning.
    """

    token: str
    request: SandboxRequest
    failure: CommandProcessResult
    warning: str


@dataclass(frozen=True)
class CommandAttempt:
    """Return an authorized sandbox result and any reviewable host offer.

    Args:
        result (CommandProcessResult | None): Sandboxed attempt, or None when permission was denied.
        offer (HostCommandOffer | None): Host retry offer after a boundary failure or an
            observed denial during a nonzero shell attempt.
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
        auto_approval (bool): Whether prompts are being bypassed by an automatic mode.
    """

    _backend: SandboxBackend
    _host_executor: HostCommandExecutor
    _permissions: PermissionManager
    _interaction: Interaction | None
    _auto_approval: bool
    _offers: dict[str, HostCommandOffer]

    def __init__(
        self,
        backend: SandboxBackend,
        host_executor: HostCommandExecutor,
        permissions: PermissionManager,
        interaction: Interaction | None,
        *,
        auto_approval: bool = False,
    ) -> None:
        self._backend = backend
        self._host_executor = host_executor
        self._permissions = permissions
        self._interaction = interaction
        self._auto_approval = auto_approval
        self._offers = {}

    def run_sandboxed(
        self,
        request: SandboxRequest,
        tool_id: str,
        *,
        prelaunch_failure: str | None = None,
        execution_timeout: float | None = None,
    ) -> CommandAttempt:
        """Run an authorized request or offer retry for a typed prelaunch failure.

        Args:
            request (SandboxRequest): Exact approved command, policy, and path identities.
            tool_id (str): Registered execution tool identity used for permission evaluation.
            prelaunch_failure (str | None): Startup capability failure to return without invoking
                the backend, after authorization and path validation.
            execution_timeout (float | None): Execution budget in seconds to start after approval.
                Defaults to preserving the caller's absolute deadline.

        Returns:
            CommandAttempt: Sandboxed result and optional unexecuted host offer.
        """
        if not self._permissions.authorize_sandboxed_command(
            request, tool_id=tool_id, interaction=self._interaction
        ):
            return CommandAttempt(None, None)
        if execution_timeout is not None:
            request = replace(request, deadline=time.monotonic() + execution_timeout)
        if not request.paths_are_current():
            result = CommandProcessResult(
                SandboxOutcome.STALE,
                detail="Approved path identity changed before launch.",
            )
        elif time.monotonic() >= request.deadline:
            result = CommandProcessResult(
                SandboxOutcome.TIMED_OUT, detail="Command deadline expired before launch."
            )
        elif prelaunch_failure is not None:
            result = CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail=prelaunch_failure,
                failure_context="Native sandbox capability check failed.",
            )
        else:
            result = self._backend.run(request)
        telemetry_audit(
            "sandbox.completed",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            outcome=result.outcome.value,
        )
        uncertain = (
            result.outcome is SandboxOutcome.COMPLETED
            and result.exit_code not in (0, 72, 126, 127)
            and _denial_supports_host_offer(request, result)
        )
        if (
            result.outcome not in {SandboxOutcome.UNAVAILABLE, SandboxOutcome.DENIED}
            and not uncertain
        ):
            return CommandAttempt(result, None)
        offer = HostCommandOffer(
            str(uuid4()),
            request,
            result,
            self._permissions.host_retry_prompt(
                request.source,
                result.outcome.value,
                result.exit_code,
                result.possible_effects,
                result.failure_context,
            ),
        )
        self._offers[offer.token] = offer
        telemetry_audit(
            "host_command.offered",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            command_hash=sha256_digest(request.source),
            offer_id=offer.token,
            failure="observed_denial" if uncertain else result.outcome.value,
        )
        return CommandAttempt(result, offer)

    def retry_host_command(
        self,
        offer: HostCommandOffer,
        current_request: SandboxRequest,
        *,
        execution_timeout: float | None = None,
    ) -> CommandProcessResult | None:
        """Approve an unchanged offer and launch at most one host process.

        Args:
            offer (HostCommandOffer): Previously issued, unused host retry offer.
            current_request (SandboxRequest): Freshly resolved command context at approval time.
            execution_timeout (float | None): Execution budget in seconds to start after host
                approval. Defaults to preserving the caller's absolute deadline.

        Returns:
            CommandProcessResult | None: Host result after approval, or None when no launch occurs.
        """
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
            return None
        if self._auto_approval or self._interaction is None:
            telemetry_audit(
                "host_command.denied",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            return None
        reusable = (
            offer.failure.outcome in {SandboxOutcome.UNAVAILABLE, SandboxOutcome.DENIED}
            and not offer.failure.possible_effects
        )
        rule = self._permissions.matching_host_command_rule(current_request) if reusable else None
        if rule is None:
            self._interaction.info(offer.warning)
            choices, index = self._permissions.host_retry_choices(reusable)
            answer = self._interaction.prompt(
                "Run this exact command without the OS sandbox?",
                exit_commands=None,
                choices=choices,
                index=index,
            )
        else:
            answer = "rule"
        if answer not in ({"approve", "session", "rule"} if reusable else {"approve"}):
            telemetry_audit(
                "host_command.denied",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            return None
        if execution_timeout is not None:
            current_request = replace(
                current_request, deadline=time.monotonic() + execution_timeout
            )
        if (
            not current_request.paths_are_current()
            or time.monotonic() >= current_request.deadline
            or (
                rule is not None
                and self._permissions.matching_host_command_rule(current_request) != rule
            )
        ):
            telemetry_audit(
                "host_command.stale",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            return None
        if answer == "session":
            rule = self._permissions.remember_host_command_rule(current_request)
        telemetry_audit(
            "host_command.approved",
            workspace_id=offer.request.workspace_id,
            attempt_id=offer.request.attempt_id,
            offer_id=offer.token,
            source="rule" if answer == "rule" else "interactive",
            rule_id=rule.id if isinstance(rule, HostCommandRule) else None,
        )
        try:
            result = self._host_executor.run_host_command(
                HostCommandRequest.from_sandbox(current_request)
            )
        except Exception:
            telemetry_audit(
                "host_command.execution_failed",
                workspace_id=offer.request.workspace_id,
                attempt_id=offer.request.attempt_id,
                offer_id=offer.token,
            )
            raise
        telemetry_audit(
            "host_command.executed",
            workspace_id=offer.request.workspace_id,
            attempt_id=offer.request.attempt_id,
            offer_id=offer.token,
        )
        return result

    @staticmethod
    def _host_identity(request: SandboxRequest) -> tuple[object, ...]:
        """Bind the authority-bearing details of the offered host execution."""
        return (
            request.source,
            request.execution_source,
            request.aliases,
            request.read_aliases,
            request.executable_identities,
            request.cwd,
            request.workspace,
            request.workspace_id,
            request.environment,
        )
