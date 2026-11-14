"""Load, evaluate, persist, approve, and record local operation policies."""

# Pylint cannot infer mutable attributes declared by Pydantic models.
# pylint: disable=no-member

from __future__ import annotations

import asyncio
import errno
import inspect
import ipaddress
import json
import logging
import os
import re
import shlex
import sqlite3
import stat
import tempfile
from _thread import LockType
from atexit import register
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from fnmatch import fnmatchcase
from functools import partial
from importlib import import_module
from pathlib import Path
from threading import Lock, RLock
from time import monotonic
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol
from urllib.parse import quote, urlsplit
from uuid import uuid4
from weakref import WeakSet

from .. import constants
from ..command_authorities import CommandAuthorityConflict, command_authorities
from ..errors import Problem, log_problem, safe_error_detail
from ..interaction import (
    Interaction,
    prompt_recovery,
    prompt_recovery_async,
    require_async_interaction,
)
from ..telemetry import telemetry_audit, telemetry_error, telemetry_trace_event
from ..utils import (
    PathReference,
    VirtualPath,
    YAMLError,
    canonical_path,
    dump_yaml,
    ensure_private_directory,
    load_yaml,
    local_now,
    run_in_worker_async,
    sha256_digest,
)
from .audit import SQLitePermissionAudit
from .inspection import CommandAnalysis, CommandInspection
from .managed import ManagedPolicyError, ManagedPolicySnapshot, ManagedPolicyStore
from .models import (
    Action,
    ApprovalChoice,
    AuthorizationResult,
    CommandFinding,
    CommandReviewStatus,
    Decision,
    FileTarget,
    HostCommandRule,
    LimitClassification,
    NetworkTarget,
    Operation,
    Operations,
    OperationTarget,
    PermissionConfiguration,
    PermissionConfigurationError,
    PermissionLoadFailure,
    PermissionLoadPolicy,
    PermissionLoadResult,
    PermissionNamespaceError,
    PermissionPreset,
    PermissionPresetError,
    PermissionRecorder,
    PermissionRule,
    PolicyDecision,
    PolicyLimitClassifications,
    PolicyLimitOverrides,
    PolicyLimits,
    PolicyScope,
    PresetReplacementPreview,
    PresetSource,
    ProcessBoundary,
    ProcessTarget,
    SandboxedCommandRule,
    SessionPolicyOverrides,
    SessionTarget,
    UserPermissionConfiguration,
)
from .protection import (
    ProtectedWorkspacePaths,
    is_protected_external_read,
    protected_workspace_paths,
)


def _settle_egress_leases(
    leases: tuple[EgressRevocationLease, ...], action: Callable[[EgressRevocationLease], bool]
) -> list[bool]:
    """Settle independent attempt channels together within their control deadlines."""
    if len(leases) < 2:
        return [action(lease) for lease in leases]
    with ThreadPoolExecutor(max_workers=min(32, len(leases))) as executor:
        return list(executor.map(action, leases))


if TYPE_CHECKING:
    from ..execution.sandbox import SandboxRequest


class EgressRevocationLease(Protocol):
    """Settle mediated admission and closure when active policy changes."""

    def pause(self) -> bool:
        """Block new connections and confirm the admission barrier.

        Returns:
            bool: Whether new admission is blocked.
        """

    def resume(self) -> bool:
        """Resume only the unchanged policy revision.

        Returns:
            bool: Whether the live unchanged attempt resumed admission.
        """

    def revoke_revision(self, revision: int) -> bool:
        """Close covered transports and confirm the exact new revision.

        Args:
            revision (int): Published monotonic policy revision.

        Returns:
            bool: Whether covered transport closure was confirmed on time.
        """


_EGRESS_ACK_SECONDS = 2.0


_READ_ACTIONS = {Action.FILESYSTEM_LIST, Action.FILESYSTEM_READ}
_WRITE_ACTIONS = {
    Action.FILESYSTEM_CREATE,
    Action.FILESYSTEM_REPLACE,
    Action.FILESYSTEM_DELETE,
}
_LIMIT_NAMES = (
    "readable_roots",
    "writable_roots",
    "network_origins",
    "deny_private_networks",
    "deny_local_networks",
    "allow_host_processes",
)
_LOGGER = logging.getLogger(__name__)
_TRUSTED_POLICY_ERRORS = frozenset(
    {
        "Policy document must be a mapping.",
        "Policy document must not be empty.",
        "Marked policy fields are incomplete.",
        "Marked policy limits or provenance are incomplete.",
    }
)
_ASYNC_POLICY_ATTEMPTS = 3
_POLICY_LOCK_POLL_SECONDS = 0.01
_MAX_CANCELLATION_NOTES = 8
_RESET_PENDING_SUFFIX = ".reset-pending"
_BROAD_NETWORK_POLICY_ORIGIN = "https://loop.invalid"
_SIMPLE_COMMAND = re.compile(r"[A-Za-z0-9_./:@+=,-]+(?: [A-Za-z0-9_./:@+=,-]+)+")
_LIVE_MANAGERS: WeakSet[PermissionManager] = WeakSet()


def _retain_cancellation(
    current: asyncio.CancelledError | None,
    later: asyncio.CancelledError,
    *,
    owned_failure: bool,
    interruption: str,
) -> asyncio.CancelledError:
    """Keep the first signal and report bounded diagnostics from its later interruption."""
    if current is None:
        return later
    if not owned_failure:
        current.add_note(
            f"An additional cancellation was observed during permission {interruption}."
        )
        return current
    notes = getattr(later, "__notes__", ())
    if current is not later and isinstance(notes, (list, tuple)):
        for note in notes[:_MAX_CANCELLATION_NOTES]:
            if isinstance(note, str) and re.fullmatch(
                r"(?:Permission audit|The blocking permission operation) also stopped with "
                r"[A-Za-z_][A-Za-z0-9_]{0,127}\.",
                note,
            ):
                current.add_note(note)
    return current


def _close_live_managers() -> None:
    """Release temporary directories held by managers alive at process shutdown."""
    for manager in _LIVE_MANAGERS:
        manager.close()


register(_close_live_managers)


class RememberedApproval(NamedTuple):
    """Retain installed rule IDs, cancellation and policy application state.

    Args:
        rule_ids (tuple[str, ...]): IDs added by the approved policy update.
        cancellation (asyncio.CancelledError | None): Preserved cancellation after settlement.
        applied (bool): Whether the policy update was applied.
    """

    rule_ids: tuple[str, ...]
    cancellation: asyncio.CancelledError | None
    applied: bool


class PermissionManager:
    """Authorize complete operation plans using limits, policy rules, and approval.

    Args:
        workspace_root (Path | str | None): Workspace used to resolve policy root tokens.
        configuration_path (Path | str | None): YAML policy path. Defaults to the workspace's
            application policy file when a workspace is supplied.
        user_configuration_path (Path | str | None): User-wide remembered-approval YAML path.
            User scope is unavailable when omitted.
        audit_path (Path | str | None): Central SQLite audit database path.
        protected_host_roots (tuple[Path | str, ...]): Application storage roots that no
            manually granted sandbox read or write scope may intersect.
        workspace_id (str | None): Stable identity attached to centralized audit records.
        namespace_catalog_path (Path | str | None): Independent workspace registry used to
            validate durable permission-policy namespace identity and migration.
        bootstrap_pristine_policy (bool): Establish the built-in defaults automatically when a
            validated namespace has no policy, reservation, backup, or reset evidence.
        user_namespace_identity (str | None): Canonical user-data identity for the user-wide
            namespace when a catalog is supplied.
        interaction (Interaction | None): User interaction used for approval prompts.
        yolo (bool): Trusted no-prompt mode active at construction, including policy loading.
        recorder (PermissionRecorder | None): Default sink for authorization observations.
        configuration (PermissionConfiguration | None): Explicit policy instead of the local file;
            unavailable when durable policy namespaces are configured.
        presets (Iterable[PermissionPreset] | None): Additional selectable presets alongside
            the built-in catalog. Duplicate identifiers are rejected.
        accepted_preset_failures (tuple[PermissionLoadFailure, ...]): Exact invalid preset
            diagnostics already acknowledged for this startup attempt.
        load_policy (PermissionLoadPolicy | None): Artifact failure behavior. Defaults to
            interactive recovery when an interaction is available and strict errors otherwise.
        command_inspection (CommandInspection | None): Trusted command inspection. Defaults to
            the built-in matchers.
        async_blocking_runner (Callable[..., Awaitable[Any]] | None): Application-owned bounded
            worker for synchronous policy persistence, audit writes, and compatibility recorders.
            Defaults to a shared bounded process worker.
        managed_policy_store (ManagedPolicyStore | None): Trusted protected installation source.
            Defaults to the platform's fixed system location.

    Raises:
        PermissionConfigurationError: If a policy is invalid in strict mode or a pending reset
            requires recovery.
        PermissionPresetError: If built-in presets are invalid in strict mode.
        ShutdownRequested: If the user exits interactive configuration recovery.
        ValueError: If interactive recovery is selected without an interaction or preset
            identifiers are duplicated.
    """

    _workspace_root: Path | None
    _configuration_path: Path | None
    _user_configuration_path: Path | None
    _user_configuration_source_path: Path | None
    _interaction: Interaction | None
    _recorder: PermissionRecorder | None
    _configuration: PermissionConfiguration
    _user_configuration: UserPermissionConfiguration
    _load_policy: PermissionLoadPolicy
    _session_overrides: SessionPolicyOverrides
    _temporary_directory: tempfile.TemporaryDirectory[str]
    _temporary_path: Path | PathReference
    _temporary_ordinary_names: Callable[[], bool] | None
    _temporary_bound: bool
    _standalone_leases: int
    _closing: bool
    _closed: bool
    _presets: dict[str, PermissionPreset]
    _audit_store: SQLitePermissionAudit | None
    _protected_read_roots: tuple[Path, ...]
    _protected_host_roots: tuple[Path, ...]
    _workspace_id: str | None
    _namespace_catalog_path: Path | None
    _bootstrap_pristine_policy: bool
    _user_namespace_identity: str | None
    _namespace_errors: dict[str, str]
    _command_inspection: CommandInspection
    _host_runtime_bindings: dict[str, str]
    _host_rules_lock: LockType
    _host_rules_revision: int
    _policy_lock: RLock
    _async_policy_write_lock: LockType
    _policy_revision: int
    _egress_revokers: set[EgressRevocationLease]
    _egress_updating: bool
    _policy_update_depth: int
    _yolo: bool
    _yolo_generation: int
    _yolo_suspensions: list[int | None]
    _managed_policy_store: ManagedPolicyStore
    _managed_policy: ManagedPolicySnapshot | None
    _managed_error: str | None
    _workspace_storage_revision: bytes | None
    _user_storage_revision: bytes | None
    _failed_workspace_revision: bytes | None
    _failed_user_revision: bytes | None
    _async_blocking_runner: Callable[..., Awaitable[Any]] | None

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        *,
        configuration_path: Path | str | None = None,
        user_configuration_path: Path | str | None = None,
        audit_path: Path | str | None = None,
        protected_host_roots: tuple[Path | str, ...] = (),
        workspace_id: str | None = None,
        namespace_catalog_path: Path | str | None = None,
        bootstrap_pristine_policy: bool = False,
        user_namespace_identity: str | None = None,
        interaction: Interaction | None = None,
        yolo: bool = False,
        recorder: PermissionRecorder | None = None,
        configuration: PermissionConfiguration | None = None,
        presets: Iterable[PermissionPreset] | None = None,
        accepted_preset_failures: tuple[PermissionLoadFailure, ...] = (),
        load_policy: PermissionLoadPolicy | None = None,
        command_inspection: CommandInspection | None = None,
        async_blocking_runner: Callable[..., Awaitable[Any]] | None = None,
        managed_policy_store: ManagedPolicyStore | None = None,
    ) -> None:
        self._policy_lock = RLock()
        self._temporary_bound = False
        self._standalone_leases = 0
        self._closing = False
        self._closed = False
        self._async_policy_write_lock = Lock()
        self._host_rules_lock = Lock()
        self._host_rules_revision = 0
        self._egress_revokers = set()
        self._egress_updating = False
        self._policy_update_depth = 0
        self._policy_revision = 0
        self._yolo = yolo
        self._yolo_generation = 0
        self._yolo_suspensions = []
        self._managed_policy_store = managed_policy_store or ManagedPolicyStore()
        self._managed_policy = None
        self._managed_error = None
        self._load_managed_policy()
        self._workspace_storage_revision = None
        self._user_storage_revision = None
        self._failed_workspace_revision = None
        self._failed_user_revision = None
        self._namespace_catalog_path = (
            Path(namespace_catalog_path).expanduser().absolute()
            if namespace_catalog_path is not None
            else None
        )
        self._bootstrap_pristine_policy = bootstrap_pristine_policy
        self._user_namespace_identity = user_namespace_identity
        self._namespace_errors = {}
        self._async_blocking_runner = async_blocking_runner
        self._command_inspection = (
            command_inspection if command_inspection is not None else CommandInspection()
        )
        self._host_runtime_bindings = {}
        self._workspace_root = (
            Path(workspace_root).resolve() if workspace_root is not None else None
        )
        # pylint: disable-next=consider-using-with
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=constants.TEMPORARY_DIRECTORY_PREFIX
        )
        self._temporary_path = Path(self._temporary_directory.name).resolve()
        self._temporary_ordinary_names = None
        _LIVE_MANAGERS.add(self)
        try:
            self._configuration_path = (
                Path(configuration_path)
                if configuration_path is not None
                else self._workspace_root / constants.APP_DIRECTORY / constants.PERMISSIONS_FILENAME
                if self._workspace_root is not None
                else None
            )
            self._user_configuration_source_path = (
                Path(user_configuration_path).expanduser().absolute()
                if user_configuration_path is not None
                else None
            )
            self._user_configuration_path = (
                (
                    self._user_configuration_source_path
                    if self._namespace_catalog_path is not None
                    else self._user_configuration_source_path.resolve()
                )
                if self._user_configuration_source_path is not None
                else None
            )
            self._workspace_id = workspace_id
            if self._namespace_catalog_path is not None and (
                workspace_id is None
                or user_namespace_identity is None
                or self._workspace_root is None
                or self._user_configuration_path is None
                or configuration is not None
            ):
                raise ValueError(
                    "Durable policy namespaces require workspace and user paths and identities, "
                    "without an injected configuration."
                )
            self._audit_store = (
                SQLitePermissionAudit(audit_path) if audit_path is not None else None
            )
            managed_directory = self._managed_policy_store.directory
            managed_expectation = self._managed_policy_store.expectation_path
            protected_read = [Path(root).resolve() for root in protected_host_roots]
            protected = [Path(root) for root in protected_host_roots]
            if managed_directory is not None:
                protected_read.append(managed_directory)
                protected.append(managed_directory)
            if managed_expectation is not None:
                protected_read.append(managed_expectation)
                protected.append(managed_expectation)
            self._protected_read_roots = tuple(dict.fromkeys(protected_read))
            if self._configuration_path is not None and (
                self._workspace_root is None
                or not self._configuration_path.parent.is_relative_to(self._workspace_root)
            ):
                protected.append(self._configuration_path.parent)
            if self._user_configuration_path is not None:
                protected.append(self._user_configuration_path.parent)
            if audit_path is not None:
                protected.append(Path(audit_path).parent)
            self._protected_host_roots = tuple(dict.fromkeys(path.resolve() for path in protected))
            self._interaction = interaction
            self._recorder = recorder
            self._load_policy = load_policy or (
                PermissionLoadPolicy.INTERACTIVE
                if interaction is not None and not self._yolo
                else PermissionLoadPolicy.ERROR
            )
            if self._load_policy is PermissionLoadPolicy.INTERACTIVE and interaction is None:
                raise ValueError("Interactive permission loading requires an Interaction.")
            self._configuration = (
                configuration.model_copy(deep=True)
                if configuration is not None
                else PermissionConfiguration()
            )
            if configuration is None:
                self._load_configuration()
            elif self._configuration_path is not None:
                self._workspace_storage_revision = self._read_storage_revision(
                    self._configuration_path
                )
            self._user_configuration = UserPermissionConfiguration()
            self._load_user_configuration()
            if (
                self._namespace_catalog_path is not None
                and self._workspace_root is not None
                and any(
                    root.is_relative_to(self._workspace_root)
                    or self._workspace_root.is_relative_to(root)
                    for root in self._protected_read_roots
                )
            ):
                self._namespace_errors["workspace"] = (
                    "The workspace intersects protected application storage; select a disjoint "
                    "workspace before policy admission."
                )
            self._session_overrides = SessionPolicyOverrides()
            builtin_presets = self._load_presets(accepted_preset_failures)
            catalog = (*builtin_presets, *(presets or ()))
            identifiers = [preset.metadata.id for preset in catalog]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Permission preset identifiers must be unique.")
            self._presets = {preset.metadata.id: preset for preset in catalog}
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release the manager-owned temporary directory.

        Raises:
            OSError: A standalone command retains the directory, cleanup fails, or mediated
                egress closure could not be confirmed. Failed cleanup can be retried.
        """
        with self._policy_lock:
            if self._closed:
                return
            if self._standalone_leases:
                raise OSError("Standalone command execution still owns temporary storage.")
            self._closing = True
            confirmations = self._revoke_egress_leases(self._policy_revision + 1)
            self._temporary_directory.cleanup()
            if not all(confirmations):
                raise OSError("Mediated egress closure could not be confirmed.")
            self._closed = True
            self._closing = False

    def bind_egress(self, revision: int, lease: EgressRevocationLease) -> None:
        """Bind one live mediated attempt to the exact approving active revision.

        Args:
            revision (int): Revision captured atomically with the command approval.
            lease (EgressRevocationLease): Attempt-owned pause, resume, and revocation channel.

        Raises:
            OSError: The manager is closing or closed, policy changed, or managed authority
                became unavailable before activation.
        """
        with self._policy_lock:
            if (
                self._closing
                or self._closed
                or self._egress_updating
                or revision != self._policy_revision
                or self._managed_current_error() is not None
            ):
                raise OSError("Mediated egress policy changed before activation.")
            self._egress_revokers.add(lease)

    def unbind_egress(self, lease: EgressRevocationLease) -> None:
        """Remove a settled mediated attempt from policy-change notifications.

        Args:
            lease (EgressRevocationLease): The exact attempt previously bound to this manager.
        """
        with self._policy_lock:
            self._egress_revokers.discard(lease)

    def _revoke_egress_leases(self, revision: int) -> list[bool]:
        """Revoke channels concurrently and retire confirmed leases under the policy lock."""
        leases = tuple(self._egress_revokers)
        deadline = monotonic() + _EGRESS_ACK_SECONDS
        confirmations = _settle_egress_leases(leases, lambda lease: lease.revoke_revision(revision))
        if monotonic() > deadline:
            confirmations = [False] * len(leases)
        for lease, confirmed in zip(leases, confirmations):
            if confirmed:
                self._egress_revokers.discard(lease)
        return confirmations

    def egress_policy_revision(self) -> int:
        """Check managed-source freshness and report this manager's active revision.

        Returns:
            int: In-process revision; another manager's persisted write awaits trusted reload.

        Raises:
            OSError: Required managed authority is invalid, changed, or stale.
        """
        with self._policy_lock:
            previous = self._policy_revision
            error = self._managed_current_error()
            confirmations = (
                self._revoke_egress_leases(self._policy_revision)
                if self._policy_revision != previous
                else []
            )
            if not all(confirmations):
                raise OSError("Mediated egress revocation closure is uncertain.")
            if error is not None:
                raise OSError("Mediated egress managed policy is unavailable.")
            return self._policy_revision

    def try_egress_policy_revision(self) -> int | None:
        """Check live authority without blocking the independent heartbeat sender.

        Returns:
            int | None: Active revision, or None while an exclusive policy update runs.

        Raises:
            OSError: Required managed authority is invalid, changed, or stale.
        """
        if not self._policy_lock.acquire(blocking=False):
            return None
        try:
            return self.egress_policy_revision()
        finally:
            self._policy_lock.release()

    @property
    def yolo(self) -> bool:
        """Return no-prompt state after current managed-policy restrictions.

        Returns:
            bool: Active YOLO state; invalid or prohibiting managed policy disables it.
        """
        with self._policy_lock:
            if self._yolo:
                self._managed_current_error()
            return self._yolo and all(
                generation == self._yolo_generation for generation in self._yolo_suspensions
            )

    def recovery_retry_generation(self) -> int | None:
        """Capture the no-prompt generation for an explicit uncertain-outcome retry.

        Returns:
            int | None: The current generation when a human decision is possible, otherwise None.
        """
        with self._policy_lock:
            return None if self.yolo else self._yolo_generation

    def recovery_retry_current(self, generation: int) -> bool:
        """Check whether a retry retains its automatic-decision generation.

        A successful check at the tool callable's dispatch boundary admits that invocation;
        later policy changes do not revoke it.

        Args:
            generation (int): Generation captured before the human retry decision.

        Returns:
            bool: Whether the captured generation still permits retry admission.
        """
        with self._policy_lock:
            return not self.yolo and self._yolo_generation == generation

    @contextmanager
    def suspend_yolo_decisions(self) -> Generator[Callable[[], None], None, None]:
        """Block newly granted automatic decisions while an activation is being published.

        Yields:
            Callable[[], None]: Roll back automatic authority without restoring a revoked
                generation. If the transition cannot settle, disable automatic decisions without
                claiming that existing egress leases have closed.
        """

        def fail_closed() -> None:
            """Remove automatic authority even when normal policy transition cannot settle."""
            with self._policy_lock:
                if self._yolo:
                    self._yolo = False
                    self._yolo_generation += 1
                    self._policy_revision += 1
                    telemetry_audit(
                        "permission.yolo_changed", enabled=False, source="rollback_uncertain"
                    )

        with self._policy_lock:
            generation = self._yolo_generation if self.yolo else None
            self._yolo_suspensions.append(generation)

        def rollback() -> None:
            """Preserve only authority that was active and has not since been revoked."""
            try:
                with self.policy_update():
                    if self._yolo:
                        self._managed_current_error()
                    if generation is None or generation != self._yolo_generation:
                        self.set_yolo(False)
            except BaseException:
                fail_closed()
                raise

        try:
            yield rollback
        finally:
            with self._policy_lock:
                self._yolo_suspensions.remove(generation)

    def validate_yolo_activation(self) -> None:
        """Require current administrator authority to permit no-prompt activation.

        Raises:
            ValueError: Managed policy is invalid, changed, or prohibits YOLO.
        """
        with self._policy_lock:
            if error := self._managed_current_error():
                raise ValueError(f"Administrator policy recovery required: {error}")
            if self._managed_policy is not None and self._managed_policy.policy.forbid_yolo:
                raise ValueError("Administrator policy prohibits YOLO activation.")

    def set_yolo(self, enabled: bool) -> None:
        """Apply a trusted configuration change to subsequent authorization decisions.

        Args:
            enabled (bool): Whether the trusted human configuration enables YOLO.

        Raises:
            ValueError: Managed policy is invalid or prohibits activation.
        """
        with self.policy_update():
            if enabled:
                self.validate_yolo_activation()
            if self._yolo == enabled:
                return
            self._yolo = enabled
            self._yolo_generation += 1
            self._policy_revision += 1
        telemetry_audit("permission.yolo_changed", enabled=enabled)

    def _load_managed_policy(self) -> None:
        """Bind a protected administrator policy, retaining an invalid-state denial."""
        try:
            current = self._managed_policy_store.load()
            previous = self._managed_policy
            if previous is not None and current is None:
                raise ManagedPolicyError(
                    "Administrator policy installation disappeared during this process."
                )
            if (
                previous is not None
                and current is not None
                and (
                    current.policy.identity == previous.policy.identity
                    and current.identity != previous.identity
                    and current.policy.revision <= previous.policy.revision
                )
            ):
                raise ManagedPolicyError(
                    "Administrator policy replacement needs a higher revision."
                )
            self._managed_policy = current
            self._managed_error = None
            if current is not None and current.policy.forbid_yolo:
                self._disable_yolo_for_managed_policy()
        except ManagedPolicyError as error:
            self._managed_error = str(error)
            self._disable_yolo_for_managed_policy()

    def _disable_yolo_for_managed_policy(self) -> None:
        """Revoke automatic decisions without reactivating after a later policy reload."""
        if self._yolo:
            self._yolo = False
            self._yolo_generation += 1
            self._policy_revision += 1
            telemetry_audit("permission.yolo_changed", enabled=False, source="managed")

    def _managed_current_error(self) -> str | None:
        """Keep a changed or invalid installation denied until trusted reload."""
        if self._managed_error is not None:
            return self._managed_error
        if self._managed_policy is None and not self._managed_policy_store.may_be_installed():
            return None
        try:
            current = self._managed_policy_store.load()
        except ManagedPolicyError as error:
            self._managed_error = str(error)
        else:
            previous = self._managed_policy
            if (current.identity if current is not None else None) != (
                previous.identity if previous is not None else None
            ):
                self._managed_error = "Administrator policy changed; reload its trusted revision."
        if self._managed_error is not None:
            self._policy_revision += 1
            self._disable_yolo_for_managed_policy()
            telemetry_audit("permission.managed_unavailable", reason="changed_or_invalid")
        return self._managed_error

    def confirmation_decision(self, *, certain: bool = True) -> bool | None:
        """Resolve an otherwise confirmable action without a prompt in YOLO mode.

        Args:
            certain (bool): Whether inputs and the prior outcome permit an automatic decision.

        Returns:
            bool | None: Approval or recovery refusal in YOLO, or None for normal interaction.
        """
        if not self.yolo:
            return None
        telemetry_audit(
            "permission.confirmation_decided",
            decision="allow" if certain else "recovery_required",
            source="yolo",
        )
        return certain

    @contextmanager
    def policy_update(self) -> Iterator[None]:
        """Serialize one policy command with operation authorization checks.

        Yields:
            None: Exclusive access to policy state until the command completes.
        """
        with self._policy_lock:
            outer = self._policy_update_depth == 0
            if outer:
                self._egress_updating = True
                pauses = _settle_egress_leases(
                    tuple(self._egress_revokers), lambda lease: lease.pause()
                )
                if not all(pauses):
                    self._revoke_egress_leases(self._policy_revision + 1)
                    self._egress_updating = False
                    raise OSError("Mediated egress admission pause is uncertain.")
            self._policy_update_depth += 1
            revision = self._policy_revision
            try:
                yield
            except BaseException:
                if outer:
                    confirmations = self._revoke_egress_leases(self._policy_revision + 1)
                    if not all(confirmations):
                        raise OSError("Mediated egress revocation closure is uncertain.")
                raise
            else:
                if outer:
                    if self._policy_revision != revision:
                        confirmations = self._revoke_egress_leases(self._policy_revision)
                    else:
                        confirmations = _settle_egress_leases(
                            tuple(self._egress_revokers), lambda lease: lease.resume()
                        )
                    if not all(confirmations):
                        raise OSError("Mediated egress policy transition is uncertain.")
            finally:
                self._policy_update_depth -= 1
                if outer:
                    self._egress_updating = False

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    @property
    def configuration(self) -> PermissionConfiguration:
        """Return the persisted workspace policy.

        Returns:
            PermissionConfiguration: Workspace defaults, limits, and rules.
        """
        return self._configuration.model_copy(deep=True)

    @property
    def effective_configuration(self) -> PermissionConfiguration:
        """Return the workspace policy with session overrides applied.

        Returns:
            PermissionConfiguration: Complete effective defaults, limits, and active rules.
        """
        defaults = dict(self._configuration.defaults)
        defaults.update(self._session_overrides.defaults)
        limit_updates = {
            name: value
            for name in _LIMIT_NAMES
            if (value := getattr(self._session_overrides.limits, name)) is not None
        }
        limits = self._configuration.limits.model_copy(update=limit_updates)
        classifications = self._configuration.limit_classifications.model_copy(
            update={
                name: self._session_overrides.limit_classifications.get(
                    name,
                    LimitClassification.HARD
                    if name in limit_updates
                    else getattr(self._configuration.limit_classifications, name),
                )
                for name in ("readable_roots", "writable_roots", "network_origins")
            }
            | {
                "sources": {
                    **self._configuration.limit_classifications.sources,
                    **{
                        name: "session"
                        for name in ("readable_roots", "writable_roots", "network_origins")
                        if name in limit_updates
                        or name in self._session_overrides.limit_classifications
                    },
                }
            }
        )
        return self._configuration.model_copy(
            update={
                "defaults": defaults,
                "limits": limits,
                "limit_classifications": classifications,
                "rules": [*self._configuration.rules, *self._session_overrides.rules],
            },
            deep=True,
        )

    @property
    def session_overrides(self) -> SessionPolicyOverrides:
        """Return an immutable snapshot of active session policy changes.

        Returns:
            SessionPolicyOverrides: Deep copy of the process-local policy overlay.
        """
        return self._session_overrides.model_copy(deep=True)

    @property
    def configuration_path(self) -> Path | None:
        """Return the local policy path.

        Returns:
            Path | None: YAML path, or None for an in-memory manager.
        """
        return self._configuration_path

    def policy_namespace_identity(self, scope: str) -> str:
        """Return the expected identity for a trusted policy initialization command.

        Args:
            scope (str): `workspace` or `user` namespace.

        Returns:
            str: Durable expected identity.

        Raises:
            ValueError: If the namespace is unavailable or unknown.
        """
        identity = (
            self._workspace_id
            if scope == "workspace"
            else self._user_namespace_identity
            if scope == "user"
            else None
        )
        if identity is None or self._namespace_catalog_path is None:
            raise ValueError(f"Durable policy namespace '{scope}' is unavailable.")
        return identity

    @property
    def user_configuration_path(self) -> Path | None:
        """Return the user-wide remembered-approval policy path.

        Returns:
            Path | None: YAML path, or None when user-wide approvals are unavailable.
        """
        return self._user_configuration_path

    @property
    def temporary_directory(self) -> Path:
        """Return the active scratch directory allowed by the application temporary-root token.

        Returns:
            Path: Bound session directory, or manager-owned storage before session binding.
        """
        return Path(self._temporary_path)

    def bind_temporary_directory(
        self,
        directory: PathReference,
        *,
        ordinary_names: Callable[[], bool] | None = None,
    ) -> None:
        """Follow the session owner's working-directory reference without owning its cleanup.

        Args:
            directory (PathReference): Private storage reference following session selection.
            ordinary_names (Callable[[], bool] | None): Trusted session-owner query for fresh
                agent-data authority. None retains private-name restrictions.

        Raises:
            OSError: The manager is closing or closed, or a standalone command owns temporary
                storage.
        """
        try:
            with command_authorities.composition((self,)), self._policy_lock:
                if self._closed or self._closing:
                    raise OSError("Permission manager is closed or closing.")
                if self._standalone_leases:
                    raise OSError("Standalone command execution still owns temporary storage.")
                with self.policy_update():
                    self._temporary_path = directory
                    self._temporary_ordinary_names = ordinary_names
                    self._temporary_bound = True
                    self._policy_revision += 1
        except CommandAuthorityConflict as error:
            raise OSError("Command execution still owns temporary storage.") from error

    @contextmanager
    def standalone_execution(self) -> Iterator[Path]:
        """Retain manager-owned scratch while one standalone command runs.

        Yields:
            Path: The manager-owned temporary directory held until this lease exits.

        Raises:
            OSError: The manager is closing or closed, or its temporary directory belongs to a
                session.
        """
        with self._policy_lock:
            if self._closed or self._closing:
                raise OSError("Permission manager is closed or closing.")
            if self._temporary_bound:
                raise OSError("Session-bound temporary storage requires its session owner.")
            self._standalone_leases += 1
            directory = Path(self._temporary_path)
        try:
            yield directory
        finally:
            with self._policy_lock:
                self._standalone_leases -= 1

    @property
    def temporary_directory_allows_private_names(self) -> bool:
        """Return the session owner's current scratch-data classification.

        Returns:
            bool: Whether ordinary data basenames are allowed in this generation.

        Raises:
            ValueError: The session owner finds unsafe working storage.
            OSError: The session owner cannot inspect working storage.
        """
        return self._temporary_ordinary_names is not None and self._temporary_ordinary_names()

    @property
    def interaction(self) -> Interaction | None:
        """Return the interaction used for approvals.

        Returns:
            Interaction | None: Configured interaction, when available.
        """
        return self._interaction

    @interaction.setter
    def interaction(self, interaction: Interaction | None) -> None:
        """Set the interaction used for approvals.

        Args:
            interaction (Interaction | None): New interaction or None for headless use.
        """
        self._interaction = interaction

    @property
    def recorder(self) -> PermissionRecorder | None:
        """Return the authorization observation sink.

        Returns:
            PermissionRecorder | None: Configured recorder, when available.
        """
        return self._recorder

    @recorder.setter
    def recorder(self, recorder: PermissionRecorder | None) -> None:
        """Set the authorization observation sink.

        Args:
            recorder (PermissionRecorder | None): New recorder or None to disable recording.
        """
        self._recorder = recorder

    def set_async_blocking_runner(
        self,
        runner: Callable[..., Awaitable[Any]],
    ) -> None:
        """Use an application-owned bounded worker for synchronous async-path effects.

        Args:
            runner (Callable[..., Awaitable[Any]]): Worker that invokes a synchronous operation
                and returns its result. Used for policy and audit persistence and sync-only
                authorization recorders.

        Returns:
            None: The asynchronous audit worker is configured.
        """
        self._async_blocking_runner = runner

    @property
    def persistent_rules(self) -> tuple[PermissionRule, ...]:
        """Return persisted policy rules in display order.

        Returns:
            tuple[PermissionRule, ...]: Immutable snapshot of persisted rules.
        """
        return tuple(rule.model_copy(deep=True) for rule in self._configuration.rules)

    @property
    def user_rules(self) -> tuple[PermissionRule, ...]:
        """Return user-wide remembered approval rules in display order.

        Returns:
            tuple[PermissionRule, ...]: Immutable snapshot of user-wide rules.
        """
        return tuple(rule.model_copy(deep=True) for rule in self._user_configuration.rules)

    @property
    def session_rules(self) -> tuple[PermissionRule, ...]:
        """Return process-local policy rules in display order.

        Returns:
            tuple[PermissionRule, ...]: Immutable snapshot of session rules.
        """
        return tuple(rule.model_copy(deep=True) for rule in self._session_overrides.rules)

    @property
    def presets(self) -> tuple[PermissionPreset, ...]:
        """Return selectable permission presets ordered by stable identifier.

        Returns:
            tuple[PermissionPreset, ...]: Deep-copied preset catalog entries.
        """
        return tuple(
            self._presets[identifier].model_copy(deep=True) for identifier in sorted(self._presets)
        )

    def authorize(
        self,
        operations: Operations,
        *,
        interaction: Interaction | None = None,
        recorder: PermissionRecorder | None = None,
        protected_instruction_names: tuple[str, ...] = (),
    ) -> AuthorizationResult:
        """Evaluate and approve one complete operation set atomically.

        Args:
            operations (Operations): Complete normalized effects of one tool call.
            interaction (Interaction | None): Invocation interaction overriding the default.
            recorder (PermissionRecorder | None): Invocation recorder overriding the default.
            protected_instruction_names (tuple[str, ...]): Active instruction filenames that
                require a fresh decision when a structured tool changes them.

        Returns:
            AuthorizationResult: Policy, prompt, and effective result for the complete set.
        """
        policy, fresh_file_review = self._authorization_policy(
            operations, protected_instruction_names
        )
        active_interaction = interaction if interaction is not None else self._interaction
        prompt = None
        selected = None
        headless = policy.decision is Decision.ASK and active_interaction is None and not self.yolo
        if policy.decision is Decision.ASK and active_interaction is not None and not self.yolo:
            prompt_body = self._prompt(operations)
            prompt = self._authorization_prompt(prompt_body, fresh_file_review)
            selected = self.request_permission(
                prompt_body,
                interaction=active_interaction,
                fresh_only=fresh_file_review,
            )
        return self._complete_authorization(
            operations,
            policy,
            fresh_file_review,
            prompt,
            selected,
            headless=headless,
            recorder=recorder,
        )

    async def authorize_async(
        self,
        operations: Operations,
        *,
        interaction: Interaction | None = None,
        recorder: PermissionRecorder | None = None,
        protected_instruction_names: tuple[str, ...] = (),
    ) -> AuthorizationResult:
        """Evaluate and approve one complete operation set without blocking native async input.

        Args:
            operations (Operations): Complete normalized effects of one tool call.
            interaction (Interaction | None): Invocation interaction overriding the default.
            recorder (PermissionRecorder | None): Invocation recorder overriding the default.
            protected_instruction_names (tuple[str, ...]): Active instruction filenames that
                require a fresh decision when a structured tool changes them.

        Returns:
            AuthorizationResult: Policy, prompt, and effective result for the complete set.

        Raises:
            asyncio.CancelledError: If approval or owned audit and recording work is cancelled.
        """
        policy, fresh_file_review = await self._authorization_policy_async(
            operations, protected_instruction_names
        )
        active_interaction = interaction if interaction is not None else self._interaction
        prompt = None
        selected = None
        headless = policy.decision is Decision.ASK and active_interaction is None and not self.yolo
        if policy.decision is Decision.ASK and active_interaction is not None and not self.yolo:
            prompt_body = self._prompt(operations)
            prompt = self._authorization_prompt(prompt_body, fresh_file_review)
            selected = await self.request_permission_async(
                prompt_body,
                interaction=active_interaction,
                fresh_only=fresh_file_review,
            )
        result, cancellation = await self._complete_authorization_async(
            operations,
            policy,
            fresh_file_review,
            prompt,
            selected,
            headless=headless,
        )
        audit_failures: list[tuple[asyncio.CancelledError, BaseException]] = []
        try:
            await self._audit_async(result, _owned_failures=audit_failures)
        except asyncio.CancelledError as error:
            cancellation = _retain_cancellation(
                cancellation,
                error,
                owned_failure=bool(audit_failures and audit_failures[0][0] is error),
                interruption="audit",
            )
        active_recorder = recorder if recorder is not None else self._recorder
        recorder_error = None
        if active_recorder is not None:
            recorded = self._recordable_result(result)
            async_recorder = getattr(active_recorder, "record_authorization_async", None)
            native_async = callable(async_recorder) and inspect.iscoroutinefunction(async_recorder)
            recorder_failures: list[tuple[asyncio.CancelledError, BaseException]] = []
            try:
                if native_async:
                    await async_recorder(recorded)
                else:
                    _, recorder_cancellation = await self._run_blocking_settled_async(
                        active_recorder.record_authorization,
                        recorded,
                        _owned_failures=recorder_failures,
                    )
                    if recorder_cancellation is not None:
                        cancellation = _retain_cancellation(
                            cancellation,
                            recorder_cancellation,
                            owned_failure=False,
                            interruption="recorder",
                        )
            except asyncio.CancelledError as error:
                owned_failure = (
                    recorder_failures[0][1]
                    if recorder_failures
                    and recorder_failures[0][0] is error
                    and not isinstance(recorder_failures[0][1], asyncio.CancelledError)
                    else None
                )
                cancellation = _retain_cancellation(
                    cancellation,
                    error,
                    owned_failure=owned_failure is not None,
                    interruption="recorder",
                )
                recorder_error = owned_failure
            except Exception as error:
                if cancellation is None:
                    raise
                cancellation.add_note(
                    f"Permission recorder also stopped with {type(error).__name__}."
                )
                recorder_error = error
        if cancellation is not None:
            if recorder_error is not None:
                raise cancellation from recorder_error
            raise cancellation
        return result

    def _authorization_policy(
        self,
        operations: Operations,
        protected_instruction_names: tuple[str, ...],
    ) -> tuple[PolicyDecision, bool]:
        """Evaluate policy and impose a fresh prompt for protected file changes."""
        policy = self.evaluate(operations)
        file_review = self._requires_fresh_file_review(
            operations, self.protected_instruction_names(protected_instruction_names)
        )
        managed_fresh = any(source.startswith("managed:human:") for source in policy.sources)
        if file_review and policy.decision is not Decision.DENY:
            policy = PolicyDecision(
                decision=Decision.ASK,
                reason="A user-managed replacement, deletion, "
                "or instruction change needs fresh approval.",
                sources=("boundary:fresh_file_review",),
            )
        return policy, file_review or managed_fresh

    async def _authorization_policy_async(
        self,
        operations: Operations,
        protected_instruction_names: tuple[str, ...],
    ) -> tuple[PolicyDecision, bool]:
        """Evaluate initial async policy without blocking on concurrent policy updates."""
        policy = await self.evaluate_async(operations)
        fresh_file_review = False
        if any(
            isinstance(operation.target, FileTarget) and operation.action in _WRITE_ACTIONS
            for operation in operations
        ):
            fresh_file_review, cancellation = await self._run_blocking_settled_async(
                self._requires_fresh_file_review,
                operations,
                self.protected_instruction_names(protected_instruction_names),
            )
            if cancellation is not None:
                raise cancellation
        managed_fresh = any(source.startswith("managed:human:") for source in policy.sources)
        if fresh_file_review and policy.decision is not Decision.DENY:
            policy = PolicyDecision(
                decision=Decision.ASK,
                reason=(
                    "A user-managed replacement, deletion, "
                    "or instruction change needs fresh approval."
                ),
                sources=("boundary:fresh_file_review",),
            )
        return policy, fresh_file_review or managed_fresh

    @staticmethod
    def _authorization_prompt(prompt_body: str, fresh_file_review: bool) -> str:
        """Add the appropriate confirmation wording to an operation description."""
        return (
            f"{prompt_body}\nProceed with this one request?"
            if fresh_file_review
            else f"{prompt_body}\nProceed?"
        )

    def _complete_authorization(
        self,
        operations: Operations,
        policy: PolicyDecision,
        fresh_file_review: bool,
        prompt: str | None,
        selected: object | None,
        *,
        headless: bool,
        recorder: PermissionRecorder | None,
    ) -> AuthorizationResult:
        """Apply an approval response, record its decision, and return the final result."""
        (
            policy,
            decision,
            reason,
            source,
            approval_choice,
            approval_scope,
        ) = self._authorization_outcome(
            operations,
            policy,
            fresh_file_review,
            selected,
            headless=headless,
        )
        installed_rule_ids = ()
        if approval_scope is not None:
            try:
                with self.policy_update():
                    installed_rule_ids = self._remember_approval(
                        operations,
                        scope=approval_scope,
                    )
            except OSError as exc:
                decision = Decision.DENY
                reason = f"Could not persist the approved permission policy: {exc}"
                source = "persistence"

        result = self._authorization_result(
            operations,
            policy,
            decision,
            reason,
            source,
            prompt,
            approval_choice,
            installed_rule_ids,
        )
        self._audit(result)
        active_recorder = recorder if recorder is not None else self._recorder
        if active_recorder is not None:
            active_recorder.record_authorization(self._recordable_result(result))
        return result

    async def _complete_authorization_async(
        self,
        operations: Operations,
        policy: PolicyDecision,
        fresh_file_review: bool,
        prompt: str | None,
        selected: object | None,
        *,
        headless: bool,
    ) -> tuple[AuthorizationResult, asyncio.CancelledError | None]:
        """Apply an async approval without doing policy-file I/O on the event loop."""
        current_policy = None
        if policy.decision is Decision.ASK and (
            self.yolo
            or (
                not headless and self._selected_approval_choice(selected) is not ApprovalChoice.DENY
            )
        ):
            current_policy = await self.evaluate_async(operations)
        (
            policy,
            decision,
            reason,
            source,
            approval_choice,
            approval_scope,
        ) = self._authorization_outcome(
            operations,
            policy,
            fresh_file_review,
            selected,
            headless=headless,
            current_policy=current_policy,
        )
        installed_rule_ids = ()
        cancellation = None
        if approval_scope is not None:
            try:
                remembered = await self._remember_approval_async(operations, scope=approval_scope)
            except OSError as exc:
                decision = Decision.DENY
                reason = f"Could not persist the approved permission policy: {exc}"
                source = "persistence"
            else:
                installed_rule_ids = remembered.rule_ids
                cancellation = remembered.cancellation
                if not remembered.applied:
                    decision = Decision.DENY
                    reason = "Permission approval was cancelled before its rule was saved."
                    source = "cancelled"
                else:
                    try:
                        current_policy = await self.evaluate_async(operations)
                    except asyncio.CancelledError as error:
                        cancellation = cancellation or error
                        decision = Decision.DENY
                        reason = "Permission authorization was cancelled after its rule was saved."
                        source = "cancelled"
                    else:
                        if current_policy.decision is Decision.DENY:
                            policy = current_policy
                            decision = Decision.DENY
                            reason = (
                                "The active permission policy changed to deny this operation while "
                                "approval was pending."
                            )
                            source = "policy"

        result = self._authorization_result(
            operations,
            policy,
            decision,
            reason,
            source,
            prompt,
            approval_choice,
            installed_rule_ids,
        )
        return result, cancellation

    def _authorization_outcome(
        self,
        operations: Operations,
        policy: PolicyDecision,
        fresh_file_review: bool,
        selected: object | None,
        *,
        headless: bool,
        current_policy: PolicyDecision | None = None,
    ) -> tuple[
        PolicyDecision,
        Decision,
        str,
        str,
        ApprovalChoice | None,
        ApprovalChoice | None,
    ]:
        """Resolve prompt choice and current policy before applying remembered approval."""
        decision = policy.decision
        reason = policy.reason
        source = "policy"
        approval_choice = None
        approval_scope = None
        approval_policy_denied = False
        if policy.decision is Decision.ASK:
            if self.yolo:
                current_policy = current_policy or self.evaluate(operations)
                if current_policy.decision is Decision.DENY:
                    policy = current_policy
                    decision = Decision.DENY
                    reason = "The active policy denied this operation before YOLO admission."
                    source = "policy"
                else:
                    decision = Decision.ALLOW
                    reason = "Approved for this request by YOLO mode."
                    source = "yolo"
                    approval_choice = ApprovalChoice.ONCE
            elif headless:
                decision = Decision.DENY
                reason = "Approval is required but no interactive user is available."
                source = "headless"
            else:
                source = "user"
                approval_choice = self._selected_approval_choice(selected)
                if approval_choice is not ApprovalChoice.DENY:
                    current_policy = current_policy or self.evaluate(operations)
                    if current_policy.decision is Decision.DENY:
                        policy = current_policy
                        approval_choice = ApprovalChoice.DENY
                        approval_policy_denied = True
                        decision = Decision.DENY
                        reason = (
                            "The active permission policy changed to deny this operation while "
                            "approval was pending."
                        )
                        source = "policy"
                if fresh_file_review and approval_choice is not ApprovalChoice.ONCE:
                    approval_choice = ApprovalChoice.DENY
                if approval_choice is ApprovalChoice.WORKSPACE and self._configuration_path is None:
                    approval_choice = ApprovalChoice.DENY
                    decision = Decision.DENY
                    reason = "Workspace approval is unavailable without a workspace policy path."
                if approval_choice is ApprovalChoice.USER and self._user_configuration_path is None:
                    approval_choice = ApprovalChoice.DENY
                    decision = Decision.DENY
                    reason = "User approval is unavailable without a user policy path."
                if approval_choice is not ApprovalChoice.DENY:
                    decision = Decision.ALLOW
                    reason = f"Approved by the user with {approval_choice.value} scope."
                    if approval_choice in {
                        ApprovalChoice.SESSION,
                        ApprovalChoice.WORKSPACE,
                        ApprovalChoice.USER,
                    }:
                        approval_scope = approval_choice
                elif reason == policy.reason and not approval_policy_denied:
                    decision = Decision.DENY
                    reason = "Rejected by the user."
                if source not in {"persistence", "policy"}:
                    source = "user"
        return policy, decision, reason, source, approval_choice, approval_scope

    @staticmethod
    def _selected_approval_choice(selected: object | None) -> ApprovalChoice:
        """Normalize interaction values to one explicit approval choice."""
        if isinstance(selected, ApprovalChoice):
            return selected
        return ApprovalChoice.ONCE if selected is True else ApprovalChoice.DENY

    def _authorization_result(
        self,
        operations: Operations,
        policy: PolicyDecision,
        decision: Decision,
        reason: str,
        source: str,
        prompt: str | None,
        approval_choice: ApprovalChoice | None,
        installed_rule_ids: tuple[str, ...],
    ) -> AuthorizationResult:
        """Build and trace the immutable result shared by sync and async authorization."""
        result = AuthorizationResult(
            operations=operations,
            policy=policy,
            decision=decision,
            prompted=prompt is not None,
            prompt=prompt,
            reason=reason,
            source=source,
            approval_choice=approval_choice,
            installed_rule_ids=installed_rule_ids,
        )
        telemetry_audit(
            "permission.decided",
            decision=result.decision.value,
            source=result.source,
            prompted=result.prompted,
            approval_scope=(
                result.approval_choice.value if result.approval_choice is not None else None
            ),
            operation_count=len(result.operations),
        )
        telemetry_trace_event("permission.decision", payload=self._recordable_result(result))
        return result

    @staticmethod
    def _recordable_result(result: AuthorizationResult) -> AuthorizationResult:
        """Remove host command material from routine decision and session records."""
        if not any(
            isinstance(operation.target, ProcessTarget)
            and operation.target.boundary is ProcessBoundary.HOST
            for operation in result.operations
        ):
            return result
        return result.model_copy(
            update={
                "operations": (),
                "prompt": None,
                "reason": "Host process authorization decision.",
                "policy": result.policy.model_copy(
                    update={"reason": "Host process policy decision."}
                ),
            }
        )

    def _requires_fresh_file_review(
        self,
        operations: Operations,
        protected_instruction_names: tuple[str, ...],
    ) -> bool:
        """Identify user-owned replacement/deletion and instruction mutations."""
        protection = protected_workspace_paths(protected_instruction_names)
        for operation in operations:
            target = operation.target
            if not isinstance(target, FileTarget) or operation.action not in _WRITE_ACTIONS:
                continue
            if operation.action in {Action.FILESYSTEM_REPLACE, Action.FILESYSTEM_DELETE}:
                return True
            if self._workspace_root is None:
                continue
            path = Path(target.path).resolve(strict=False)
            if not path.is_relative_to(self._workspace_root):
                continue
            relative = path.relative_to(self._workspace_root)
            if relative.name in (
                *protection.instruction_names,
                *protection.files,
            ) or protection.protects_directory(relative):
                return True
        return False

    def inspect_command(self, source: str) -> tuple[CommandFinding, ...]:
        """Collect command findings from this manager's registry.

        Args:
            source (str): Opaque shell source to inspect heuristically.

        Returns:
            tuple[CommandFinding, ...]: Applicable findings; absence does not prove safety.
        """
        return self._command_inspection.inspect(source)

    def analyze_command(self, source: str) -> CommandAnalysis:
        """Collect advisory findings and executable names in one inspection.

        Args:
            source (str): Opaque shell source to inspect heuristically.

        Returns:
            CommandAnalysis: Bounded findings and candidate names; absence proves no safety.
        """
        return self._command_inspection.analyze(source)

    def protected_instruction_names(self, configured: Iterable[str] = ()) -> tuple[str, ...]:
        """Bind configured instruction filenames into native command protection.

        Args:
            configured (Iterable[str]): Active project instruction basenames.

        Returns:
            tuple[str, ...]: Default and active filenames requiring native write denial.

        Raises:
            ValueError: If a configured name is not a plain filename.
        """
        return protected_workspace_paths(configured).instruction_names

    def protected_sandbox_paths(self, configured: Iterable[str] = ()) -> ProtectedWorkspacePaths:
        """Describe workspace paths that ordinary sandbox commands cannot mutate.

        Args:
            configured (Iterable[str]): Active project instruction basenames.

        Returns:
            ProtectedWorkspacePaths: Instruction basenames and protected relative paths.

        Raises:
            ValueError: If a configured instruction name is not a plain filename.
        """
        return protected_workspace_paths(configured)

    @staticmethod
    def git_sandbox_roots(
        workspace: Path,
        *,
        read: bool,
        write: bool,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...], bool]:
        """Bind Git metadata reads, writes, and creation for one workspace.

        Args:
            workspace (Path): Canonical workspace root.
            read (bool): Whether the command inspects Git metadata.
            write (bool): Whether Git metadata mutation was requested.

        Returns:
            tuple[tuple[Path, ...], tuple[Path, ...], bool]: Linked metadata read roots,
                approved metadata write roots, and new repository creation intent.

        Raises:
            ValueError: If a linked worktree pointer or creation target is unsafe.
        """
        if not (read or write):
            return (), (), False
        git_directory = workspace / constants.GIT_DIRECTORY
        if git_directory.is_file():
            details = git_directory.lstat()
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise ValueError("Unsafe linked worktree Git pointer.")
            pointer = git_directory.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: "):
                raise ValueError("Invalid linked worktree Git directory pointer.")
            git_root = (git_directory.parent / pointer[8:]).resolve(strict=True)
            if not git_root.is_dir() or not (git_root / "HEAD").is_file():
                raise ValueError("Invalid linked worktree Git metadata directory.")
            reads = (git_root,)
            common_pointer = git_root / "commondir"
            if common_pointer.exists():
                common_details = common_pointer.lstat()
                if not stat.S_ISREG(common_details.st_mode) or common_details.st_nlink != 1:
                    raise ValueError("Unsafe linked worktree common Git pointer.")
                common_directory = (
                    git_root / common_pointer.read_text(encoding="utf-8").strip()
                ).resolve(strict=True)
                if not common_directory.is_dir() or not (common_directory / "HEAD").is_file():
                    raise ValueError("Invalid linked worktree common Git directory.")
                reads = (*reads, common_directory)
            return reads, reads if write else (), False
        if write:
            if git_directory.is_symlink():
                raise ValueError("Unsafe Git metadata creation target.")
            return (
                (),
                (git_directory,) if git_directory.exists() else (),
                not git_directory.exists(),
            )
        return (), (), False

    @staticmethod
    def describe_sandbox_reads(
        workspace: Path,
        roots: tuple[Path, ...],
        manual_roots: tuple[Path, ...],
        tool_names: tuple[str, ...],
        package_installation: bool,
    ) -> str | None:
        """Describe expanded read authority without exposing host paths in a prompt.

        Args:
            workspace (Path): Selected user workspace root.
            roots (tuple[Path, ...]): All effective extra read roots.
            manual_roots (tuple[Path, ...]): Roots supplied directly by the caller.
            tool_names (tuple[str, ...]): Names of identity-bound installed tools.
            package_installation (bool): Whether all tool roots are a recognized package install.

        Returns:
            str | None: Honest plain-language context for expanded reads, if any.
        """
        if not roots:
            return None
        if any(
            root == Path("/")
            or (workspace.is_relative_to(root) and root != workspace)
            or (root in manual_roots and len(root.parts) <= 3)
            for root in roots
        ):
            return "Read a broad area outside the workspace"
        if tool_names and manual_roots:
            return "Use installed tools and read an additional folder outside the workspace"
        if len(tool_names) > 1:
            return "Use installed developer tools"
        if tool_names:
            return (
                f"Use installed {tool_names[0]}"
                if package_installation
                else f"Read the installed {tool_names[0]} tool folder"
            )
        return "Read an additional folder outside the workspace"

    @staticmethod
    def host_retry_prompt(
        source: str,
        outcome: str,
        exit_code: int | None,
        possible_effects: bool,
        failure_context: str = "Native sandbox execution could not start.",
    ) -> str:
        """Describe an offered unrestricted host run and possible repeated effects.

        Args:
            source (str): Exact shell source proposed for the host retry.
            outcome (str): Completed, denied, or unavailable sandbox outcome.
            exit_code (int | None): Completed sandbox command's exit code, if any.
            possible_effects (bool): Whether the sandbox attempt may have changed state.
            failure_context (str): Path-free failure explanation supplied by the execution
                boundary. Defaults to an unspecified native start failure.

        Returns:
            str: Human warning for the separate host authorization boundary.
        """
        if outcome == "completed":
            reason = (
                "The command exited with code "
                f"{exit_code}, and the OS denied an operation during this attempt. "
                "The denial may be unrelated to the exit."
            )
        else:
            reason = (
                "The sandbox denied this command."
                if outcome == "denied"
                else "The sandbox was unavailable."
            )
            reason += f" {failure_context}"
        repeat = (
            "\nThe sandboxed attempt may already have had effects. "
            "Retrying may repeat those effects."
            if possible_effects or outcome == "completed"
            else ""
        )
        return (
            f"Command: {json.dumps(source)}\n"
            f"Reason: {reason} Running without the OS sandbox can read private files, "
            "change files outside the workspace, and use the network."
            f"{repeat}"
        )

    def host_retry_choices(
        self,
        reusable: bool,
        trusted: bool = False,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return permission-owned choices for a fresh or reusable host offer.

        Args:
            reusable (bool): Whether a prelaunch failure permits a stable exact session rule.
            trusted (bool): Whether this is an explicit exact host command eligible for grants.

        Returns:
            tuple[dict[str, str], dict[str, str]]: Choice labels and keyboard index.
        """
        choices = {"deny": "Deny", "approve": "Approve this one host run"}
        index = {"deny": "N", "approve": "Y"}
        if reusable:
            choices["session"] = "Allow this exact no-effects retry for this session"
            index["session"] = "S"
        if trusted:
            choices["trust_session"] = "Trust this exact host command for this session"
            index["trust_session"] = "T"
            if (
                self._namespace_catalog_path is not None
                and self._user_configuration_path is not None
            ):
                choices["trust_workspace"] = "Trust this exact host command in this workspace"
                choices["trust_user"] = "Trust this exact host command for this user"
                index["trust_workspace"] = "W"
                index["trust_user"] = "U"
        return choices, index

    def command_needs_git_write(self, source: str) -> bool:
        """Report whether this registry requests Git metadata write authority.

        Args:
            source (str): Opaque shell source to inspect conservatively.

        Returns:
            bool: Whether any inspector requests Git write authority. Unrecognized shell
                indirection remains subject to the OS protection on Git metadata.
        """
        return any(finding.requests_git_write for finding in self.inspect_command(source))

    def authorize_sandboxed_command(
        self,
        request: SandboxRequest,
        *,
        tool_id: str,
        interaction: Interaction | None = None,
        reason: str | None = None,
        approved_revision: list[int] | None = None,
    ) -> bool:
        """Authorize a bound sandbox request using sandbox-only scoped approval rules.

        Legacy process allow rules never approve this request. Explicit deny policy still takes
        precedence, and an explicit ask always requires a fresh decision. Commands within the
        default workspace sandbox run without a prompt; additional authority requires a
        sandbox-only approval or a one-request YOLO decision. A policy change observed before
        the final policy check invalidates the decision.

        Args:
            request (SandboxRequest): Exact command, authority, and path identities to approve.
            tool_id (str): Registered execution tool identity for policy and audit.
            interaction (Interaction | None): Interactive approval surface.
            reason (str | None): Model-provided explanation shown only in the user prompt.
            approved_revision (list[int] | None): Optional output for the exact policy revision
                that approved the sandbox attempt, populated only on approval.

        Returns:
            bool: Whether policy or the user approved this sandbox attempt.
        """
        with self._policy_lock:
            egress_limited = bool(
                request.egress_origins
                and self.sandbox_egress_limit(tool_id, request.egress_origins)
            )
            network_limited = bool(request.network and self.sandbox_network_limit(tool_id))
            operations = self._sandbox_operations(request, tool_id)
            policy, revision = self._sandbox_policy_snapshot(operations)
        if egress_limited:
            self._audit_sandbox_denial(request, source="egress_rule_limit")
            return False
        if self.sandbox_scope_limit(request) is not None:
            self._audit_sandbox_denial(request, source="scope_limit")
            return False
        if network_limited:
            self._audit_sandbox_denial(request, source="network_limit")
            return False
        if policy.decision is Decision.DENY:
            self._audit_sandbox_denial(request, source="policy")
            return False
        scratch_roots = self._sandbox_scratch_roots(request)
        explicit_ask = policy.decision is Decision.ASK and any(
            source.startswith(("rule:", "managed:human:")) for source in policy.sources
        )
        temporary_roots = tuple(scratch_roots)
        scratch = request.shell_environment().get("TMPDIR")
        if scratch is not None:
            temporary_roots += (Path(scratch),)
        findings = self._command_inspection.inspect(
            request.execution_source,
            temporary_roots=temporary_roots,
        )
        fresh_review = any(finding.status is CommandReviewStatus.FRESH for finding in findings)
        self._audit_sandbox_scope(request, findings)
        if self._has_default_sandbox_access(
            request,
            scratch_roots,
            explicit_ask=explicit_ask,
            findings=findings,
        ):
            if not self.sandbox_policy_revision_current(revision, approved_revision):
                self._audit_sandbox_denial(request, source="policy")
                return False
            self._audit_sandbox_decision(request, decision="allow", source="workspace_sandbox")
            return True
        reusable = not fresh_review and not explicit_ask
        matched_scope = self._matching_sandbox_rule_scope(request) if reusable else None
        if matched_scope is not None:
            if not self.sandbox_policy_revision_current(revision, approved_revision):
                self._audit_sandbox_denial(request, source="policy")
                return False
            self._audit_sandbox_decision(
                request,
                decision="allow",
                source=f"rule:{matched_scope.value}",
            )
            return True
        if not self.sandbox_policy_revision_current(revision):
            self._audit_sandbox_denial(request, source="policy")
            return False
        if self.yolo:
            if not self.sandbox_policy_revision_current(revision, approved_revision):
                self._audit_sandbox_denial(request, source="policy")
                return False
            self._audit_sandbox_decision(request, decision="allow", source="yolo")
            return True
        active_interaction = interaction if interaction is not None else self._interaction
        if active_interaction is None:
            self._audit_sandbox_denial(request, source="headless")
            return False

        approved = self._request_sandbox_approval(
            request,
            revision=revision,
            interaction=active_interaction,
            scratch_roots=scratch_roots,
            findings=findings,
            explicit_ask=explicit_ask,
            fresh_review=fresh_review,
            reason=reason,
            approved_revision=approved_revision,
        )
        self._audit_sandbox_decision(
            request,
            decision="allow" if approved else "deny",
            source="interactive",
        )
        return approved

    async def authorize_sandboxed_command_async(
        self,
        request: SandboxRequest,
        *,
        tool_id: str,
        interaction: Interaction | None = None,
        reason: str | None = None,
        approved_revision: list[int] | None = None,
    ) -> bool:
        """Authorize a bound sandbox request without blocking asynchronous approval input.

        Explicit asks require a fresh decision outside YOLO. A policy change observed before the
        final policy check invalidates the decision.

        Args:
            request (SandboxRequest): Exact command, authority, and path identities to approve.
            tool_id (str): Registered execution tool identity for policy and audit.
            interaction (Interaction | None): Interactive approval surface.
            reason (str | None): Model-provided explanation shown only in the user prompt.
            approved_revision (list[int] | None): Optional output for the exact policy revision
                that approved the sandbox attempt, populated only on approval.

        Returns:
            bool: Whether policy or the user approved this sandbox attempt.

        Raises:
            asyncio.CancelledError: If approval or owned audit and persistence work is cancelled.
            ProblemException: If interactive approval lacks native asynchronous input.
        """
        operations = self._sandbox_operations(request, tool_id)
        policy, revision = await self._sandbox_policy_snapshot_async(operations)
        await self._acquire_policy_lock_async()
        try:
            current_revision = revision == self._policy_revision
            egress_limited = bool(
                request.egress_origins
                and self.sandbox_egress_limit(tool_id, request.egress_origins)
            )
            network_limited = bool(request.network and self.sandbox_network_limit(tool_id))
        finally:
            self._policy_lock.release()
        if not current_revision:
            self._audit_sandbox_denial(request, source="policy")
            return False
        if egress_limited:
            self._audit_sandbox_denial(request, source="egress_rule_limit")
            return False
        if self.sandbox_scope_limit(request) is not None:
            self._audit_sandbox_denial(request, source="scope_limit")
            return False
        if network_limited:
            self._audit_sandbox_denial(request, source="network_limit")
            return False
        if policy.decision is Decision.DENY:
            self._audit_sandbox_denial(request, source="policy")
            return False
        scratch_roots = self._sandbox_scratch_roots(request)
        explicit_ask = policy.decision is Decision.ASK and any(
            source.startswith(("rule:", "managed:human:")) for source in policy.sources
        )
        temporary_roots = tuple(scratch_roots)
        scratch = request.shell_environment().get("TMPDIR")
        if scratch is not None:
            temporary_roots += (Path(scratch),)
        findings = self._command_inspection.inspect(
            request.execution_source,
            temporary_roots=temporary_roots,
        )
        fresh_review = any(finding.status is CommandReviewStatus.FRESH for finding in findings)
        _, cancellation = await self._run_blocking_settled_async(
            self._audit_sandbox_scope, request, findings
        )
        if cancellation is not None:
            raise cancellation
        if self._has_default_sandbox_access(
            request,
            scratch_roots,
            explicit_ask=explicit_ask,
            findings=findings,
        ):
            if not await self._sandbox_policy_revision_current_async(revision, approved_revision):
                self._audit_sandbox_denial(request, source="policy")
                return False
            self._audit_sandbox_decision(request, decision="allow", source="workspace_sandbox")
            return True
        reusable = not fresh_review and not explicit_ask
        matched_scope = self._matching_sandbox_rule_scope(request) if reusable else None
        if matched_scope is not None:
            if not await self._sandbox_policy_revision_current_async(revision, approved_revision):
                self._audit_sandbox_denial(request, source="policy")
                return False
            self._audit_sandbox_decision(
                request,
                decision="allow",
                source=f"rule:{matched_scope.value}",
            )
            return True
        if not await self._sandbox_policy_revision_current_async(revision):
            self._audit_sandbox_denial(request, source="policy")
            return False
        if self.yolo:
            if not await self._sandbox_policy_revision_current_async(revision, approved_revision):
                self._audit_sandbox_denial(request, source="policy")
                return False
            self._audit_sandbox_decision(request, decision="allow", source="yolo")
            return True
        active_interaction = interaction if interaction is not None else self._interaction
        if active_interaction is None:
            self._audit_sandbox_denial(request, source="headless")
            return False

        approved = await self._request_sandbox_approval_async(
            request,
            revision=revision,
            interaction=active_interaction,
            scratch_roots=scratch_roots,
            findings=findings,
            explicit_ask=explicit_ask,
            fresh_review=fresh_review,
            reason=reason,
            approved_revision=approved_revision,
        )
        self._audit_sandbox_decision(
            request,
            decision="allow" if approved else "deny",
            source="interactive",
        )
        return approved

    def sandbox_scope_limit(self, request: SandboxRequest) -> str | None:
        """Explain hard file and local-network ceilings for a sandbox request.

        Args:
            request (SandboxRequest): Bound command request and native file scopes.

        Returns:
            str | None: Required trusted action, or None when file scopes are eligible.
        """
        with self._policy_lock:
            if request.local_network and not self.sandbox_local_network_enabled():
                return "The effective user or managed policy denies sandbox-local networking."
            for root in request.read_roots:
                if root.is_relative_to(request.workspace):
                    continue
                if any(
                    root.is_relative_to(protected) or protected.is_relative_to(root)
                    for protected in self._protected_read_roots
                ):
                    return "A requested read root intersects protected application policy or state."
            effective = self.effective_configuration
            for name, roots in (
                (
                    "readable_roots",
                    tuple(
                        root
                        for root in request.read_roots
                        if root not in request.automatic_tool_reads
                    ),
                ),
                ("writable_roots", request.write_roots),
            ):
                if getattr(effective.limit_classifications, name) is not LimitClassification.HARD:
                    continue
                if any(
                    not self._in_roots(str(root), getattr(effective.limits, name)) for root in roots
                ):
                    label = "read-root" if name == "readable_roots" else "write-root"
                    return (
                        f"The effective {name} limit is hard and excludes a requested sandbox "
                        f"scope. A trusted user may inspect '/permissions limit list' and "
                        f"deliberately change that limit or classify the {label} limit as "
                        "approval-controlled before requesting this scope again."
                    )
        return None

    def review_legacy_sandbox_read_limit(
        self, request: SandboxRequest, interaction: Interaction
    ) -> bool:
        """Offer a trusted workspace-policy migration before reviewing an external read.

        Args:
            request (SandboxRequest): Bound command and external read scope.
            interaction (Interaction): Trusted human prompt for the policy change.

        Returns:
            bool: Whether the legacy hard read limit became approval-controlled.
        """
        revision = self._legacy_sandbox_read_review_revision(request)
        if self.yolo:
            return False
        if revision is None:
            return False
        roots = tuple(
            root for root in request.read_roots if root not in request.automatic_tool_reads
        )
        selected = interaction.prompt(
            "The saved legacy policy blocks this sandboxed command's external read scope. "
            "Allow scoped external-read approvals for this workspace? This changes the "
            "workspace limit classification, but does not approve this command or add any roots. "
            f"Requested roots: {json.dumps([str(root) for root in roots])}",
            exit_commands=None,
            choices={"deny": "Keep the hard limit", "review": "Allow scoped read approvals"},
            index={"deny": "N", "review": "Y"},
        )
        return selected == "review" and self._complete_legacy_sandbox_read_review(revision)

    async def review_legacy_sandbox_read_limit_async(
        self, request: SandboxRequest, interaction: Interaction
    ) -> bool:
        """Offer the trusted legacy read migration through asynchronous human input.

        Args:
            request (SandboxRequest): Bound command and external read scope.
            interaction (Interaction): Trusted human prompt for the policy change.

        Returns:
            bool: Whether the legacy hard read limit became approval-controlled.

        Raises:
            asyncio.CancelledError: If review or policy persistence is cancelled.
            ProblemException: If native asynchronous input is unavailable.
        """
        revision, cancellation = await self._run_blocking_settled_async(
            self._legacy_sandbox_read_review_revision, request
        )
        if cancellation is not None:
            raise cancellation
        if self.yolo:
            return False
        if revision is None:
            return False
        roots = tuple(
            root for root in request.read_roots if root not in request.automatic_tool_reads
        )
        async_interaction = require_async_interaction(interaction, "legacy sandbox read review")
        selected = await async_interaction.prompt_async(
            "The saved legacy policy blocks this sandboxed command's external read scope. "
            "Allow scoped external-read approvals for this workspace? This changes the "
            "workspace limit classification, but does not approve this command or add any roots. "
            f"Requested roots: {json.dumps([str(root) for root in roots])}",
            exit_commands=None,
            choices={"deny": "Keep the hard limit", "review": "Allow scoped read approvals"},
            index={"deny": "N", "review": "Y"},
        )
        if selected != "review":
            return False
        approved, cancellation = await self._run_blocking_settled_async(
            self._complete_legacy_sandbox_read_review, revision
        )
        if cancellation is not None:
            raise cancellation
        return approved

    def _legacy_sandbox_read_review_revision(self, request: SandboxRequest) -> int | None:
        """Return the current revision only for an eligible legacy read ceiling."""
        with self.policy_update():
            classifications = self._configuration.limit_classifications
            if (
                classifications.sources["readable_roots"] != "legacy"
                or self._session_overrides.limits.readable_roots is not None
                or "readable_roots" in self._session_overrides.limit_classifications
                or self._namespace_recovery_decision() is not None
                or not (self.sandbox_scope_limit(request) or "").startswith(
                    "The effective readable_roots limit is hard"
                )
            ):
                return None
            return self._policy_revision

    def _complete_legacy_sandbox_read_review(self, revision: int) -> bool:
        """Apply the reviewed classification only when its policy snapshot still matches."""
        with self.policy_update():
            if revision != self._policy_revision:
                return False
            self.classify_limit("readable_roots", LimitClassification.APPROVAL)
        return True

    def sandbox_write_root_limit(self, workspace: Path, roots: tuple[Path, ...]) -> str | None:
        """Reject external write scopes intersecting control or private data.

        Args:
            workspace (Path): Bound workspace root.
            roots (tuple[Path, ...]): Manually requested write roots.

        Returns:
            str | None: Fixed boundary denial, or None when scopes are eligible.
        """
        for root in roots:
            if root.is_relative_to(workspace):
                if self._is_protected_path(str(root), mutation=True, recursive=True):
                    return "A requested write root intersects protected workspace control data."
                continue
            common = Path(os.path.commonpath((workspace, root)))
            if workspace.is_relative_to(root) or is_protected_external_read(
                root.relative_to(common)
            ):
                return "An external write root intersects the workspace or private host data."
            if any(
                root.is_relative_to(protected) or protected.is_relative_to(root)
                for protected in self._protected_host_roots
            ):
                return "An external write root intersects protected application policy or state."
        return None

    def sandbox_network_limit(self, tool_id: str) -> str | None:
        """Explain why a broad command network grant cannot satisfy effective limits.

        Args:
            tool_id (str): Exact command tool identity used for policy matching.

        Returns:
            str | None: Required trusted policy change, or None when broad egress is eligible.
        """
        with self._policy_lock:
            limits = self.effective_configuration.limits
            if limits.deny_private_networks:
                return (
                    "The effective private-network limit is hard, while this sandbox backend "
                    "offers only a general network grant. A trusted user may deliberately run "
                    "'/permissions limit set workspace private-network allow' (or select session "
                    "scope), then request the general grant again. That grant can reach private "
                    "host and network services."
                )
            if limits.deny_local_networks or (
                self._managed_policy is not None and self._managed_policy.policy.deny_local_networks
            ):
                return (
                    "The sandbox-local network limit denies a general network grant, which "
                    "would also expose host loopback. Use mediated egress on a qualified backend "
                    "or remove the local deny through trusted policy."
                )
            if "*" not in limits.network_origins:
                return (
                    "The effective network-origin limit restricts destinations, while this "
                    "sandbox backend offers only a general network grant. A trusted user may "
                    "deliberately allow the '*' network origin in the applicable policy scope, "
                    "then request the general grant again. That grant is not origin-scoped."
                )
            if self._has_scoped_network_rule(tool_id):
                return (
                    "A network URL or exact-request rule cannot be enforced by a general "
                    "network grant. Use a qualified mediator for exact destinations or change "
                    "the rule through trusted policy."
                )
        return None

    def sandbox_local_network_enabled(self) -> bool:
        """Report whether current user and managed limits permit attempt-local networking.

        Returns:
            bool: Whether the command may use its isolated loopback interface.
        """
        with self._policy_lock:
            return not (
                self.effective_configuration.limits.deny_local_networks
                or self._managed_policy is not None
                and self._managed_policy.policy.deny_local_networks
            )

    def sandbox_egress_limit(self, tool_id: str, origins: tuple[str, ...]) -> str | None:
        """Explain rules that a destination-only mediator cannot enforce.

        Args:
            tool_id (str): Exact command tool identity used for policy matching.
            origins (tuple[str, ...]): Canonical HTTP(S) destinations requested by the agent.

        Returns:
            str | None: Unavailable reason for stronger rules, or None for destination scope.
        """
        with self._policy_lock:
            limits = self.effective_configuration.limits
            if any(origin.startswith("https://") for origin in origins) and (
                "*" not in limits.network_origins
            ):
                return (
                    "An HTTPS tunnel cannot enforce the configured network-origin limit "
                    "inside encrypted traffic."
                )
            if self._has_scoped_network_rule(tool_id):
                return (
                    "A network URL or exact-request rule cannot be enforced for every request "
                    "by this destination-only mediator."
                )
        return None

    @staticmethod
    def _resource_glob_can_match_network_url(resource: str) -> bool:
        """Check whether a glob can match an HTTP(S) URL scheme prefix.

        A star before the full scheme retains the rule; excluding impossible suffixes would
        require a complete glob-and-URL intersection check.
        """
        for scheme in ("http://", "https://"):
            index = 0
            for character in scheme:
                if index == len(resource):
                    break
                token = resource[index]
                if token == "*":
                    return True
                if token == "[":
                    end = index + 1
                    if end < len(resource) and resource[end] == "!":
                        end += 1
                    if end < len(resource) and resource[end] == "]":
                        end += 1
                    end = resource.find("]", end)
                    if end < 0 or not fnmatchcase(character, resource[index : end + 1]):
                        break
                    index = end + 1
                elif token == "?" or token == character:
                    index += 1
                else:
                    break
            else:
                return True
        return False

    def _has_scoped_network_rule(self, tool_id: str) -> bool:
        """Identify scoped network rules and generic rules that may match HTTP URLs."""
        rules = (
            *self._configuration.rules,
            *self._user_configuration.rules,
            *self._session_overrides.rules,
            *(self._managed_policy.policy.rules if self._managed_policy is not None else ()),
        )
        return any(
            (rule.tool == tool_id if rule.tool_exact else fnmatchcase(tool_id, rule.tool))
            and rule.action in (None, Action.NETWORK_REQUEST)
            and (
                isinstance(rule.target, NetworkTarget)
                or rule.resource is not None
                and rule.resource != "*"
                and (
                    rule.action is Action.NETWORK_REQUEST
                    or self._resource_glob_can_match_network_url(rule.resource)
                )
            )
            for rule in rules
        )

    @staticmethod
    def _sandbox_operations(request: SandboxRequest, tool_id: str) -> tuple[Operation, ...]:
        """Check both shell sources and the targetless policy of a broad network grant."""
        commands = tuple(
            Operation(
                tool_id=tool_id,
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=(constants.SHELL_COMMAND_PATH, constants.SHELL_COMMAND_OPTION, source),
                    cwd=str(request.cwd),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            )
            for source in dict.fromkeys((request.source, request.execution_source))
        )
        return (
            *commands,
            *(
                (
                    Operation(
                        tool_id=tool_id,
                        action=Action.NETWORK_REQUEST,
                        target=NetworkTarget(
                            url=_BROAD_NETWORK_POLICY_ORIGIN + "/",
                            origin=_BROAD_NETWORK_POLICY_ORIGIN,
                        ),
                    ),
                )
                if request.network
                else ()
            ),
            *(
                Operation(
                    tool_id=tool_id,
                    action=Action.NETWORK_REQUEST,
                    target=NetworkTarget(url=origin + "/", origin=origin),
                )
                for origin in request.egress_origins
            ),
        )

    def _sandbox_policy_snapshot(
        self,
        operations: tuple[Operation, ...],
    ) -> tuple[PolicyDecision, int]:
        """Capture the command policy and its revision under one lock."""
        return self.evaluate_with_revision(operations)

    async def _sandbox_policy_snapshot_async(
        self,
        operations: tuple[Operation, ...],
    ) -> tuple[PolicyDecision, int]:
        """Capture command policy and revision without blocking the owner event loop."""
        decision, revision = await self.evaluate_with_revision_async(operations)
        return decision, revision if revision is not None else -1

    def sandbox_policy_revision_current(
        self, revision: int, approved_revision: list[int] | None = None
    ) -> bool:
        """Check a sandbox approval without pausing unchanged mediated grants.

        Args:
            revision (int): Policy revision captured with the exact sandbox approval.
            approved_revision (list[int] | None): Optional output for the current revision,
                populated only when the approval remains valid.

        Returns:
            bool: Whether policy and namespace authority are still current.

        Raises:
            OSError: Active egress closure could not be confirmed after authority changed.
        """
        with self._policy_lock:
            previous = self._policy_revision
            try:
                current = revision == previous and self._namespace_recovery_decision() is None
            except BaseException:
                if not all(self._revoke_egress_leases(self._policy_revision + 1)):
                    raise OSError("Mediated egress revocation closure is uncertain.") from None
                raise
            if self._policy_revision != previous and not all(
                self._revoke_egress_leases(self._policy_revision)
            ):
                raise OSError("Mediated egress policy transition is uncertain.")
            if current and approved_revision is not None:
                approved_revision.append(revision)
            return current

    async def _sandbox_policy_revision_current_async(
        self,
        revision: int,
        approved_revision: list[int] | None = None,
    ) -> bool:
        """Recheck and settle any changed egress authority outside the owner event loop."""
        while True:
            current, cancellation = await self._run_blocking_settled_async(
                self._sandbox_policy_revision_unless_busy, revision
            )
            if cancellation is not None:
                raise cancellation
            if current is not None:
                break
            await asyncio.sleep(_POLICY_LOCK_POLL_SECONDS)
        if current and approved_revision is not None:
            approved_revision.append(revision)
        return current

    def _sandbox_policy_revision_unless_busy(self, revision: int) -> bool | None:
        """Check authority only when the worker can own the policy lock immediately."""
        if not self._policy_lock.acquire(blocking=False):
            return None
        try:
            return self.sandbox_policy_revision_current(revision)
        finally:
            self._policy_lock.release()

    @staticmethod
    def _sandbox_scratch_roots(request: SandboxRequest) -> set[Path]:
        """Return the explicitly bound session root."""
        return {request.scratch_root} if request.scratch_root is not None else set()

    def _audit_sandbox_scope(
        self,
        request: SandboxRequest,
        findings: tuple[CommandFinding, ...],
    ) -> None:
        """Record the exact sandbox authority considered for approval."""
        self._append_audit(
            "sandbox.permission_scope",
            {
                "attempt_id": request.attempt_id,
                "scratch_data": request.scratch_data,
                "read_roots": [str(root) for root in request.read_roots],
                "automatic_tool_reads": [str(root) for root in request.automatic_tool_reads],
                "read_aliases": [str(alias) for alias in request.read_aliases],
                "executables": [
                    {"name": name, "spelling": str(spelling), "target": str(resolved)}
                    for name, spelling, resolved, _, _ in request.executable_identities
                ],
                "write_roots": [str(root) for root in request.write_roots],
                "egress_origins": list(request.egress_origins),
                "policy_version": request.policy_version,
                "review_policies": [finding.policy_id for finding in findings],
            },
        )

    @staticmethod
    def _has_default_sandbox_access(
        request: SandboxRequest,
        scratch_roots: set[Path],
        *,
        explicit_ask: bool,
        findings: tuple[CommandFinding, ...],
    ) -> bool:
        """Report whether a request stays inside the unprompted workspace sandbox."""
        return (
            not explicit_ask
            and not findings
            and set(request.read_roots) <= set(request.automatic_tool_reads)
            and not request.network
            and not request.egress_origins
            and not request.git_create
            and set(request.write_roots) <= {request.workspace, *scratch_roots}
        )

    def _matching_sandbox_rule_scope(self, request: SandboxRequest) -> ApprovalChoice | None:
        """Return the first remembered scope that approves this exact request."""
        for scope in (
            ApprovalChoice.SESSION,
            ApprovalChoice.WORKSPACE,
            ApprovalChoice.USER,
        ):
            signature = request.sandbox_command_signature(
                self._sandbox_scope_identity(request, scope)
            )
            if any(
                self._sandbox_matches(rule, request.source, signature)
                for rule in self.sandboxed_command_rules(scope)
            ):
                return scope
        return None

    def _request_sandbox_approval(
        self,
        request: SandboxRequest,
        *,
        revision: int,
        interaction: Interaction,
        scratch_roots: set[Path],
        findings: tuple[CommandFinding, ...],
        explicit_ask: bool,
        fresh_review: bool,
        reason: str | None,
        approved_revision: list[int] | None = None,
    ) -> bool:
        """Prompt for a sandbox command and remember a selected approval when allowed."""
        message = self._sandbox_permission_message(
            request,
            scratch_roots=scratch_roots,
            findings=findings,
            explicit_ask=explicit_ask,
            reason=reason,
        )
        interaction.info(message)
        choices, index, prefix = self._sandbox_approval_choices(
            request, fresh_only=fresh_review or explicit_ask
        )
        full_scope = False
        while True:
            selected = interaction.prompt(
                "Allow this command?", exit_commands=None, choices=choices, index=index
            )
            if selected != "details":
                break
            full_scope = not full_scope
            index["details"] = "R" if full_scope else "F"
            choices = {
                **choices,
                "details": "Show reduced scope" if full_scope else "Show full scope",
            }
            interaction.info(
                self._sandbox_permission_message(
                    request,
                    scratch_roots=scratch_roots,
                    findings=findings,
                    explicit_ask=explicit_ask,
                    reason=reason,
                    full_scope=True,
                )
                if full_scope
                else message
            )
        if selected not in choices or selected == "deny":
            return False
        if selected == "approve":
            return self.sandbox_policy_revision_current(revision, approved_revision)
        return self._remember_sandbox_approval(
            request,
            revision=revision,
            selected=selected,
            prefix=prefix,
            approved_revision=approved_revision,
        )

    async def _request_sandbox_approval_async(
        self,
        request: SandboxRequest,
        *,
        revision: int,
        interaction: Interaction,
        scratch_roots: set[Path],
        findings: tuple[CommandFinding, ...],
        explicit_ask: bool,
        fresh_review: bool,
        reason: str | None,
        approved_revision: list[int] | None = None,
    ) -> bool:
        """Prompt asynchronously and settle any remembered approval before returning."""
        async_interaction = require_async_interaction(interaction, "sandbox command approval")
        message = self._sandbox_permission_message(
            request,
            scratch_roots=scratch_roots,
            findings=findings,
            explicit_ask=explicit_ask,
            reason=reason,
        )
        interaction.info(message)
        choices, index, prefix = self._sandbox_approval_choices(
            request, fresh_only=fresh_review or explicit_ask
        )
        full_scope = False
        while True:
            selected = await async_interaction.prompt_async(
                "Allow this command?", exit_commands=None, choices=choices, index=index
            )
            if selected != "details":
                break
            full_scope = not full_scope
            index["details"] = "R" if full_scope else "F"
            choices = {
                **choices,
                "details": "Show reduced scope" if full_scope else "Show full scope",
            }
            interaction.info(
                self._sandbox_permission_message(
                    request,
                    scratch_roots=scratch_roots,
                    findings=findings,
                    explicit_ask=explicit_ask,
                    reason=reason,
                    full_scope=True,
                )
                if full_scope
                else message
            )
        if selected not in choices or selected == "deny":
            return False
        if selected == "approve":
            return await self._sandbox_policy_revision_current_async(revision, approved_revision)
        approved, cancellation = await self._run_blocking_settled_async(
            partial(
                self._remember_sandbox_approval,
                request,
                revision=revision,
                selected=selected,
                prefix=prefix,
                approved_revision=approved_revision,
            )
        )
        if cancellation is not None:
            raise cancellation
        return approved

    def _sandbox_permission_message(
        self,
        request: SandboxRequest,
        *,
        scratch_roots: set[Path],
        findings: tuple[CommandFinding, ...],
        explicit_ask: bool,
        reason: str | None,
        full_scope: bool = False,
    ) -> str:
        """Summarize the command and its additional authority for user review."""
        reads_outside_workspace = bool(set(request.read_roots) - set(request.automatic_tool_reads))
        read_context = self._sandbox_read_context(request) if reads_outside_workspace else None
        reasons = self._sandbox_permission_reasons(
            request,
            scratch_roots=scratch_roots,
            findings=findings,
            reads_outside_workspace=reads_outside_workspace,
            explicit_ask=explicit_ask,
        )
        contexts = [read_context] if read_context is not None else []
        contexts.extend(finding.context for finding in findings)
        if not contexts:
            contexts.append("Run this command with additional sandbox access")
        scope: dict[str, Any] = {
            "workspace": str(request.workspace),
            "workspace_write": any(
                request.workspace.is_relative_to(root) for root in request.write_roots
            ),
        }
        if request.cwd != request.workspace:
            scope["working_directory"] = str(request.cwd)
        extra_reads = [
            root for root in request.read_roots if root not in request.automatic_tool_reads
        ]
        if extra_reads:
            scope["read_roots"] = [str(root) for root in extra_reads]
            if request.read_aliases:
                scope["read_aliases"] = len(request.read_aliases)
        extra_writes = [
            root
            for root in request.write_roots
            if root != request.workspace and root not in scratch_roots
        ]
        if extra_writes:
            scope["write_roots"] = [str(root) for root in extra_writes]
        if request.automatic_tool_reads:
            scope["automatic_tool_read_roots"] = len(request.automatic_tool_reads)
        if request.network:
            scope["network"] = True
        if request.egress_origins:
            scope["egress_origins"] = list(request.egress_origins)
        if request.git_create:
            scope["git_create"] = True
        model_reason = f"\nAgent explanation: {json.dumps(reason)}" if reason is not None else ""
        executed = (
            f"Executed command: {json.dumps(request.execution_source)}\n"
            if request.execution_source != request.source
            else ""
        )
        scope_label = "Full scope" if full_scope else "Scope"
        visible_scope = self._sandbox_scope(request) if full_scope else scope
        return (
            f"Command: {json.dumps(request.source)}\n"
            f"{executed}"
            f"{scope_label}: {json.dumps(visible_scope)}\n"
            f"Context: {'; '.join(contexts)}.\n"
            f"Reason: Needs permission to {', '.join(reasons)}.{model_reason}"
        )

    @staticmethod
    def _sandbox_scope(request: SandboxRequest) -> dict[str, Any]:
        """Describe the complete bound sandbox scope for optional review."""
        return {
            "workspace": str(request.workspace),
            "working_directory": str(request.cwd),
            "read_roots": [str(root) for root in request.read_roots],
            "automatic_tool_reads": [str(root) for root in request.automatic_tool_reads],
            "read_aliases": [str(alias) for alias in request.read_aliases],
            "write_roots": [str(root) for root in request.write_roots],
            "executables": [
                {"name": name, "spelling": str(spelling), "target": str(target)}
                for name, spelling, target, _, _ in request.executable_identities
            ],
            "network": request.network,
            "egress_origins": list(request.egress_origins),
            "git_create": request.git_create,
            "scratch_root": str(request.scratch_root) if request.scratch_root is not None else None,
            "temporary_directory": request.shell_environment().get(
                VirtualPath.TEMPORARY.removeprefix("$")
            ),
            "scratch_data": request.scratch_data,
        }

    @staticmethod
    def _sandbox_read_context(request: SandboxRequest) -> str:
        """Describe extra read authority at a human-safe level."""
        if request.read_context is not None:
            return request.read_context
        broad_roots = {
            Path("/"),
            Path("/Users"),
            Path("/opt"),
            Path("/usr"),
            Path("/private/tmp"),
            Path("/tmp"),
        }
        if any(
            root in broad_roots
            or (request.workspace.is_relative_to(root) and root != request.workspace)
            for root in request.read_roots
        ):
            return "Read a broad area outside the workspace"
        return "Read an additional folder outside the workspace"

    @staticmethod
    def _sandbox_permission_reasons(
        request: SandboxRequest,
        *,
        scratch_roots: set[Path],
        findings: tuple[CommandFinding, ...],
        reads_outside_workspace: bool,
        explicit_ask: bool,
    ) -> list[str]:
        """List the policy reasons that require an interactive decision."""
        reasons = ["read files outside this workspace"] if reads_outside_workspace else []
        extra_writes = set(request.write_roots) - {request.workspace, *scratch_roots}
        if extra_writes:
            reasons.append(
                "change Git metadata"
                if all(".git" in root.parts for root in extra_writes)
                else "write outside this workspace"
            )
        if request.git_create:
            reasons.append("create Git metadata")
        if request.network:
            reasons.append("use the network")
        if request.egress_origins:
            reasons.append("use the listed HTTP(S) destinations")
        reasons.extend(finding.reason for finding in findings)
        if explicit_ask and not reasons:
            reasons.append("run under your command approval rule")
        return reasons

    def _sandbox_approval_choices(
        self,
        request: SandboxRequest,
        *,
        fresh_only: bool,
    ) -> tuple[dict[str, str], dict[str, str], str | None]:
        """Build permitted interactive choices and their optional similar-command prefix."""
        choices = {"deny": "Deny", "approve": "Allow once", "session": "Allow for this session"}
        index = {"deny": "N", "approve": "Y", "session": "S"}
        if fresh_only:
            choices.pop("session")
            index.pop("session")
            prefix = None
        else:
            if self._user_configuration_path is not None:
                choices["workspace"] = "Allow in this workspace"
                index["workspace"] = "W"
                choices["user"] = "Allow for this user"
                index["user"] = "U"
            prefix = self._similar_prefix(request.source)
            if prefix is not None:
                self._add_similar_sandbox_choices(choices, index, prefix)
        choices["details"] = "Show full scope"
        index["details"] = "F"
        return choices, index, prefix

    @staticmethod
    def _add_similar_sandbox_choices(
        choices: dict[str, str],
        index: dict[str, str],
        prefix: str,
    ) -> None:
        """Add similar-command choices for each available remembered-rule scope."""
        for scope, shortcut in (("session", "A"), ("workspace", "B"), ("user", "C")):
            if scope in choices:
                choices[f"similar_{scope}"] = f"Allow similar '{prefix} …' commands ({scope})"
                index[f"similar_{scope}"] = shortcut

    def _remember_sandbox_approval(
        self,
        request: SandboxRequest,
        *,
        revision: int,
        selected: str,
        prefix: str | None,
        approved_revision: list[int] | None = None,
    ) -> bool:
        """Persist a remembered approval selected through the sandbox command prompt."""
        similar = selected.startswith("similar_")
        scope = ApprovalChoice(selected.removeprefix("similar_"))
        rule = SandboxedCommandRule(
            signature=request.sandbox_command_signature(
                self._sandbox_scope_identity(request, scope)
            ),
            source=prefix if similar else request.source,
            similar=similar,
            scope=scope,
        )
        try:
            with self.policy_update():
                if not self.sandbox_policy_revision_current(revision):
                    return False
                if scope is ApprovalChoice.SESSION:
                    self._session_overrides.sandboxed_command_rules.append(rule)
                    self._policy_revision += 1
                else:
                    updated_user = self._user_configuration.model_copy(deep=True)
                    updated_user.sandboxed_command_rules.append(rule)
                    self._replace_user_configuration(updated_user)
                if approved_revision is not None:
                    approved_revision.append(self._policy_revision)
        except OSError:
            return False
        telemetry_audit(
            "sandbox.permission_rule_added",
            workspace_id=request.workspace_id,
            scope=scope.value,
            rule_id=rule.id,
            similar=similar,
        )
        return True

    @staticmethod
    def _audit_sandbox_decision(request: SandboxRequest, *, decision: str, source: str) -> None:
        """Record a completed sandbox authorization decision."""
        telemetry_audit(
            "sandbox.permission_decided",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            decision=decision,
            source=source,
        )

    @staticmethod
    def _audit_sandbox_denial(request: SandboxRequest, *, source: str) -> None:
        """Record a sandbox denial before interactive approval is available."""
        telemetry_audit(
            "sandbox.permission_denied",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            source=source,
        )

    @staticmethod
    def _sandbox_scope_identity(request: SandboxRequest, scope: ApprovalChoice) -> str:
        """Return the policy identity applicable to one remembered-rule scope."""
        return request.workspace_id if scope is not ApprovalChoice.USER else "user"

    @staticmethod
    def _similar_prefix(source: str) -> str | None:
        """Offer similarity only for simple literal command words."""
        if _SIMPLE_COMMAND.fullmatch(source) is None:
            return None
        words = source.split()
        if words[0] in {
            "python",
            "python3",
            "sh",
            "bash",
            "zsh",
            "node",
            "ruby",
            "perl",
            "env",
        }:
            return None
        if words[1].startswith("-"):
            return None
        return " ".join(words[:2])

    @staticmethod
    def _sandbox_matches(rule: SandboxedCommandRule, source: str, signature: str) -> bool:
        """Match an exact source or reviewed simple prefix at one authority."""
        if rule.signature != signature:
            return False
        if not rule.similar:
            return rule.source == source
        return PermissionManager._similar_prefix(source) == rule.source and (
            source == rule.source or source.startswith(rule.source + " ")
        )

    def sandboxed_command_rules(self, scope: ApprovalChoice) -> tuple[SandboxedCommandRule, ...]:
        """Return sandbox-only approvals stored at one lifetime scope.

        Args:
            scope (ApprovalChoice): Session, workspace, or user policy layer.

        Returns:
            tuple[SandboxedCommandRule, ...]: Copies of remembered sandbox approvals.

        Raises:
            ValueError: If the scope cannot store approvals.
        """
        if scope is ApprovalChoice.SESSION:
            rules = self._session_overrides.sandboxed_command_rules
        elif scope in {ApprovalChoice.WORKSPACE, ApprovalChoice.USER}:
            rules = [
                rule
                for rule in self._user_configuration.sandboxed_command_rules
                if rule.scope is scope
            ]
        else:
            raise ValueError("Native command rules require session, workspace, or user scope.")
        return tuple(rule.model_copy(deep=True) for rule in rules)

    def remove_sandboxed_command_rule(self, scope: ApprovalChoice, rule_id: str) -> bool:
        """Remove one remembered sandbox-only approval by identifier.

        Args:
            scope (ApprovalChoice): Session, workspace, or user policy layer.
            rule_id (str): Identifier of the approval to remove.

        Returns:
            bool: Whether a rule was removed.

        Raises:
            ValueError: If the scope cannot store approvals.
            OSError: If the changed persistent policy cannot be written.
        """
        if scope is ApprovalChoice.SESSION:
            with self.policy_update():
                rules = list(self.sandboxed_command_rules(scope))
                remaining = [rule for rule in rules if rule.id != rule_id]
                if len(remaining) == len(rules):
                    return False
                self._session_overrides.sandboxed_command_rules = remaining
                self._policy_revision += 1
        else:
            with self.policy_update():
                rules = list(self.sandboxed_command_rules(scope))
                remaining = [rule for rule in rules if rule.id != rule_id]
                if len(remaining) == len(rules):
                    return False
                updated_user = self._user_configuration.model_copy(deep=True)
                updated_user.sandboxed_command_rules = [
                    rule
                    for rule in updated_user.sandboxed_command_rules
                    if rule.id != rule_id or rule.scope is not scope
                ]
                self._replace_user_configuration(updated_user)
        telemetry_audit("sandbox.permission_rule_removed", scope=scope.value, rule_id=rule_id)
        return True

    def host_command_rules(self) -> tuple[HostCommandRule, ...]:
        """List revocable exact host approvals for this session.

        Returns:
            tuple[HostCommandRule, ...]: Independent copies of active host rules.
        """
        with self._host_rules_lock:
            return tuple(
                rule.model_copy(deep=True) for rule in self._session_overrides.host_command_rules
            )

    def trusted_host_command_rules(self) -> tuple[HostCommandRule, ...]:
        """List human-installed no-prompt grants in every managed scope.

        Returns:
            tuple[HostCommandRule, ...]: Detached grants with no command contents.
        """
        with self.policy_update():
            return tuple(
                rule.model_copy(deep=True)
                for rule in (
                    *self._session_overrides.trusted_host_command_rules,
                    *(
                        self._user_configuration.trusted_host_command_rules
                        if self._namespace_catalog_path is not None
                        else ()
                    ),
                )
            )

    def _trusted_host_signature(
        self, source: str, cwd: Path, tool_id: str, scope: ApprovalChoice
    ) -> str | None:
        """Bind exact shell context to the selected grant scope and current authority."""
        root = self._workspace_root
        if root is None or tool_id != "request_host_command" or not cwd.is_relative_to(root):
            return None
        try:
            identity = root.stat()
            cwd.stat()
            relative = cwd.relative_to(root)
            if not cwd.is_dir() or cwd.resolve() != cwd:
                return None
            context: tuple[object, ...] = (source, str(relative), os.environ.get("PATH"), tool_id)
            if scope is not ApprovalChoice.USER:
                context += (str(root), identity.st_dev, identity.st_ino, self._workspace_id)
            if scope is ApprovalChoice.SESSION:
                scratch = self.temporary_directory
                scratch_identity = scratch.stat()
                context += (str(scratch), scratch_identity.st_dev, scratch_identity.st_ino)
        except OSError:
            return None
        return sha256_digest(repr(context))

    def matching_trusted_host_command_rule(
        self,
        source: str,
        cwd: Path,
        tool_id: str,
        *,
        workspace: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> HostCommandRule | None:
        """Find a current human-installed grant for one exact explicit host command.

        Args:
            source (str): Opaque shell source to execute.
            cwd (Path): Canonical process working directory.
            tool_id (str): Registered explicit host tool identity.
            workspace (Path | None): Optional request workspace to bind at execution admission.
            environment (dict[str, str] | None): Exact host environment for final admission.

        Returns:
            HostCommandRule | None: Matching no-prompt grant, if one is active.
        """
        with self.policy_update():
            if self._namespace_recovery_decision() is not None or (
                workspace is not None and workspace != self._workspace_root
            ):
                return None
            if environment is not None:
                scratch = self.temporary_directory
                expected = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "LANG": "C",
                    "WORKSPACE_ROOT": str(self._workspace_root),
                    "TMPDIR": str(scratch),
                    "TMP": str(scratch),
                    "TEMP": str(scratch),
                    "XDG_CACHE_HOME": str(scratch / "cache"),
                }
                if environment != expected:
                    return None
            signatures: dict[ApprovalChoice, str | None] = {}
            for rule in (
                *self._session_overrides.trusted_host_command_rules,
                *(
                    self._user_configuration.trusted_host_command_rules
                    if self._namespace_catalog_path is not None
                    else ()
                ),
            ):
                if rule.scope not in signatures:
                    signatures[rule.scope] = self._trusted_host_signature(
                        source, cwd, tool_id, rule.scope
                    )
                if rule.signature == signatures[rule.scope]:
                    return rule.model_copy(deep=True)
            return None

    def trust_host_command(
        self, source: str, scope: ApprovalChoice = ApprovalChoice.SESSION
    ) -> HostCommandRule:
        """Install one human-approved no-prompt grant for the workspace-root host shell.

        Args:
            source (str): Exact shell command provided through trusted user interaction.
            scope (ApprovalChoice): Session, workspace, or user lifetime.

        Returns:
            HostCommandRule: Revocable grant without command contents.

        Raises:
            ValueError: If the command or trusted workspace identity is unavailable.
        """
        if not source.strip() or scope not in {
            ApprovalChoice.SESSION,
            ApprovalChoice.WORKSPACE,
            ApprovalChoice.USER,
        }:
            raise ValueError("An exact host command is required.")
        with self.policy_update():
            if self._namespace_recovery_decision() is not None:
                raise ValueError("Permission policy recovery is required before trusting host use.")
            if self._workspace_root is None:
                raise ValueError("A current workspace is required for a trusted host grant.")
            signature = self._trusted_host_signature(
                source, self._workspace_root, "request_host_command", scope
            )
            if signature is None:
                raise ValueError("The workspace identity is unavailable.")
            rule = HostCommandRule(
                signature=signature,
                cwd=VirtualPath.WORKSPACE,
                source="<exact command>",
                scope=scope,
            )
            self._store_trusted_host_rule(rule)
        telemetry_audit("host_command.trusted_rule_added", rule_id=rule.id, scope=scope.value)
        self._append_audit(
            "host_command.trusted_rule_added",
            {"rule_id": rule.id, "command_hash": sha256_digest(source), "scope": scope.value},
        )
        return rule.model_copy(deep=True)

    def _store_trusted_host_rule(self, rule: HostCommandRule) -> None:
        """Store a reviewed grant in the trusted session or user-policy layer."""
        if rule.scope is ApprovalChoice.SESSION:
            self._session_overrides.trusted_host_command_rules.append(rule)
            self._policy_revision += 1
        else:
            if self._namespace_catalog_path is None or self._user_configuration_path is None:
                raise ValueError("A validated durable user permission policy is required.")
            updated = self._user_configuration.model_copy(deep=True)
            updated.trusted_host_command_rules.append(rule)
            self._replace_user_configuration(updated)

    def remember_trusted_host_command_rule_if_current(
        self, request: SandboxRequest, scope: ApprovalChoice, revision: int
    ) -> tuple[HostCommandRule, int] | None:
        """Persist the selected grant only while the reviewed host context remains current.

        Args:
            request (SandboxRequest): Exact host request approved by fresh human input.
            scope (ApprovalChoice): Selected session, workspace, or user lifetime.
            revision (int): Policy revision at the reviewed host prompt.

        Returns:
            tuple[HostCommandRule, int] | None: Installed grant and its policy revision,
                or None if authority became stale.
        """
        with self.policy_update():
            if (
                revision != self._policy_revision
                or self._namespace_recovery_decision() is not None
                or not request.paths_are_current()
                or request.workspace != self._workspace_root
                or request.source != request.execution_source
            ):
                return None
            signature = self._trusted_host_signature(
                request.source, request.cwd, "request_host_command", scope
            )
            if signature is None:
                return None
            relative = request.cwd.relative_to(request.workspace)
            rule = HostCommandRule(
                signature=signature,
                cwd=(
                    VirtualPath.WORKSPACE
                    if relative == Path(".")
                    else f"{VirtualPath.WORKSPACE}/{relative.as_posix()}"
                ),
                source="<exact command>",
                scope=scope,
            )
            self._store_trusted_host_rule(rule)
            installed_revision = self._policy_revision
        telemetry_audit("host_command.trusted_rule_added", rule_id=rule.id, scope=scope.value)
        self._append_audit(
            "host_command.trusted_rule_added",
            {
                "rule_id": rule.id,
                "command_hash": sha256_digest(request.source),
                "scope": scope.value,
            },
        )
        return rule.model_copy(deep=True), installed_revision

    def remove_trusted_host_command_rule(
        self, rule_id: str, scope: ApprovalChoice = ApprovalChoice.SESSION
    ) -> bool:
        """Revoke one human-installed no-prompt host grant.

        Args:
            rule_id (str): Identifier of the grant to revoke.
            scope (ApprovalChoice): Scope containing the grant.

        Returns:
            bool: Whether the grant was removed.
        """
        with self.policy_update():
            rules = (
                self._session_overrides.trusted_host_command_rules
                if scope is ApprovalChoice.SESSION
                else self._user_configuration.trusted_host_command_rules
            )
            remaining = [rule for rule in rules if rule.id != rule_id]
            if len(remaining) == len(rules) or any(
                rule.scope is not scope for rule in rules if rule.id == rule_id
            ):
                return False
            removed = next(rule for rule in rules if rule.id == rule_id)
            if scope is ApprovalChoice.SESSION:
                self._session_overrides.trusted_host_command_rules = remaining
                self._policy_revision += 1
            else:
                updated = self._user_configuration.model_copy(deep=True)
                updated.trusted_host_command_rules = remaining
                self._replace_user_configuration(updated)
        telemetry_audit("host_command.trusted_rule_removed", rule_id=rule_id, scope=scope.value)
        self._append_audit(
            "host_command.trusted_rule_removed",
            {"rule_id": rule_id, "signature": removed.signature, "scope": scope.value},
        )
        return True

    def record_host_boundary(
        self,
        event_name: str,
        command: str,
        *,
        attempt_id: str | None = None,
        offer_id: str | None = None,
        source: str | None = None,
        rule_id: str | None = None,
        outcome: str | None = None,
    ) -> None:
        """Durably audit a host admission or execution event without command material.

        Args:
            event_name (str): Fixed host boundary event name.
            command (str): Executed shell source, retained only as a digest.
            attempt_id (str | None): Bound command attempt identifier.
            offer_id (str | None): One-time host offer identifier.
            source (str | None): Authorization or denial category.
            rule_id (str | None): Exact session rule or trusted grant identifier.
            outcome (str | None): Process outcome, if execution settled.
        """
        payload: dict[str, object] = {"command_hash": sha256_digest(command)}
        for name, value in (
            ("attempt_id", attempt_id),
            ("offer_id", offer_id),
            ("source", source),
            ("rule_id", rule_id),
            ("outcome", outcome),
        ):
            if value is not None:
                payload[name] = value
        self._append_audit(event_name, payload)

    def matching_host_command_rule(self, request: SandboxRequest) -> HostCommandRule | None:
        """Find an exact host rule for the current bound command context.

        Args:
            request (SandboxRequest): Freshly validated command request.

        Returns:
            HostCommandRule | None: Matching session approval, if present.
        """
        signature = request.host_command_signature()
        with self._host_rules_lock:
            if not any(
                rule.signature == signature for rule in self._session_overrides.host_command_rules
            ):
                return None
        runtime_signature = request.host_command_runtime_signature()
        with self._host_rules_lock:
            return next(
                (
                    rule.model_copy(deep=True)
                    for rule in self._session_overrides.host_command_rules
                    if rule.signature == signature
                    and self._host_runtime_bindings.get(rule.id) == runtime_signature
                ),
                None,
            )

    def remember_host_command_rule(self, request: SandboxRequest) -> HostCommandRule:
        """Record one explicitly approved exact host retry for this session.

        Args:
            request (SandboxRequest): Unchanged context approved at the host boundary.

        Returns:
            HostCommandRule: New revocable host-only decision.

        Raises:
            ValueError: If the session host rules changed while preparing the approval.
        """
        with self._host_rules_lock:
            host_rules_revision = self._host_rules_revision
        rule, runtime_signature = self._prepare_host_command_rule(request)
        with self._host_rules_lock:
            if host_rules_revision != self._host_rules_revision:
                raise ValueError("Session host rules changed while preparing the approval.")
            self._publish_host_command_rule(rule, runtime_signature)
        return self._record_host_command_rule(request, rule)

    @staticmethod
    def _prepare_host_command_rule(request: SandboxRequest) -> tuple[HostCommandRule, str]:
        """Prepare the signature and identity without holding policy or host-rule locks."""
        relative_cwd = request.cwd.relative_to(request.workspace)
        rule = HostCommandRule(
            signature=request.host_command_signature(),
            cwd=(
                VirtualPath.WORKSPACE
                if relative_cwd == Path(".")
                else f"{VirtualPath.WORKSPACE}/{relative_cwd.as_posix()}"
            ),
            source=request.source if "/" not in request.source else "<path-bearing-command>",
        )

        return rule, request.host_command_runtime_signature()

    def _publish_host_command_rule(self, rule: HostCommandRule, runtime_signature: str) -> None:
        """Publish a prepared host rule while holding the host-rule lock."""
        self._session_overrides.host_command_rules.append(rule)
        self._host_runtime_bindings[rule.id] = runtime_signature
        self._host_rules_revision += 1

    def _record_host_command_rule(
        self, request: SandboxRequest, rule: HostCommandRule
    ) -> HostCommandRule:
        """Audit and detach a published host rule."""
        telemetry_audit(
            "host_command.rule_added", workspace_id=request.workspace_id, rule_id=rule.id
        )
        self._append_audit(
            "host_command.rule_added",
            {"rule_id": rule.id, "command_hash": sha256_digest(request.execution_source)},
        )
        return rule.model_copy(deep=True)

    def remember_host_command_rule_if_current(
        self, request: SandboxRequest, revision: int
    ) -> HostCommandRule | None:
        """Record an exact host approval only while its policy revision remains current.

        Args:
            request (SandboxRequest): Unchanged context approved at the host boundary.
            revision (int): Policy revision captured with the reviewed decision.

        Returns:
            HostCommandRule | None: The recorded rule, or None if policy or host rules changed.
        """
        with self._host_rules_lock:
            host_revision = self._host_rules_revision
        rule, runtime_signature = self._prepare_host_command_rule(request)
        with self.policy_update(), self._host_rules_lock:
            if (
                revision != self._policy_revision
                or host_revision != self._host_rules_revision
                or self._namespace_recovery_decision() is not None
            ):
                return None
            self._publish_host_command_rule(rule, runtime_signature)
        return self._record_host_command_rule(request, rule)

    async def remember_host_command_rule_if_current_async(
        self, request: SandboxRequest, revision: int
    ) -> HostCommandRule | None:
        """Record an exact host approval without blocking on a policy transaction.

        Args:
            request (SandboxRequest): Unchanged context approved at the host boundary.
            revision (int): Policy revision captured with the reviewed decision.

        Returns:
            HostCommandRule | None: The recorded rule, or None if policy or host rules changed.

        Raises:
            asyncio.CancelledError: If cancelled while waiting for the policy transaction.
        """
        with self._host_rules_lock:
            host_revision = self._host_rules_revision
        rule, runtime_signature = self._prepare_host_command_rule(request)
        await self._acquire_policy_lock_async()
        try:
            with self._host_rules_lock:
                if (
                    revision != self._policy_revision
                    or host_revision != self._host_rules_revision
                    or self._namespace_recovery_decision() is not None
                ):
                    return None
                self._publish_host_command_rule(rule, runtime_signature)
        finally:
            self._policy_lock.release()
        return self._record_host_command_rule(request, rule)

    def remove_host_command_rule(self, rule_id: str) -> bool:
        """Revoke one session host approval by identifier.

        Args:
            rule_id (str): Identifier of the recorded approval.

        Returns:
            bool: Whether a matching rule was removed.
        """
        with self._host_rules_lock:
            before = len(self._session_overrides.host_command_rules)
            self._session_overrides.host_command_rules = [
                rule for rule in self._session_overrides.host_command_rules if rule.id != rule_id
            ]
            removed = len(self._session_overrides.host_command_rules) != before
            if removed:
                self._host_runtime_bindings.pop(rule_id, None)
                self._host_rules_revision += 1
        if removed:
            telemetry_audit("host_command.rule_removed", rule_id=rule_id)
            self._append_audit("host_command.rule_removed", {"rule_id": rule_id})
        return removed

    def request_permission(
        self,
        prompt: str,
        *,
        interaction: Interaction | None = None,
        fresh_only: bool = False,
    ) -> ApprovalChoice:
        """Request one scoped approval through a generic user interaction.

        Args:
            prompt (str): Complete operation description to display.
            interaction (Interaction | None): Invocation interaction overriding the default.
            fresh_only (bool): Offer only a one-time choice for user-managed data changes.

        Returns:
            ApprovalChoice: Selected denial or approval lifetime. Cancellation denies.
        """
        if self.confirmation_decision() is True:
            return ApprovalChoice.ONCE
        choices, index = self._permission_options(fresh_only)
        active_interaction = interaction if interaction is not None else self._interaction
        active_interaction.info(prompt)
        selected = active_interaction.prompt(
            "Proceed?",
            exit_commands=None,
            choices=choices,
            index=index,
        )
        return selected if isinstance(selected, ApprovalChoice) else ApprovalChoice.DENY

    async def request_permission_async(
        self,
        prompt: str,
        *,
        interaction: Interaction | None = None,
        fresh_only: bool = False,
    ) -> ApprovalChoice:
        """Request scoped approval using asynchronous input when the adapter provides it.

        Args:
            prompt (str): Complete operation description to display.
            interaction (Interaction | None): Invocation interaction overriding the default.
            fresh_only (bool): Offer only a one-time choice for user-managed data changes.

        Returns:
            ApprovalChoice: Selected denial or approval lifetime. Cancellation denies.

        Raises:
            ProblemException: If the interaction does not provide asynchronous input.
            asyncio.CancelledError: If native asynchronous input is cancelled.
        """
        if self.confirmation_decision() is True:
            return ApprovalChoice.ONCE
        choices, index = self._permission_options(fresh_only)
        active_interaction = interaction if interaction is not None else self._interaction
        async_interaction = require_async_interaction(active_interaction, "permission approval")
        active_interaction.info(prompt)
        selected = await async_interaction.prompt_async(
            "Proceed?",
            exit_commands=None,
            choices=choices,
            index=index,
        )
        return selected if isinstance(selected, ApprovalChoice) else ApprovalChoice.DENY

    def _permission_options(
        self, fresh_only: bool
    ) -> tuple[dict[ApprovalChoice, str], dict[ApprovalChoice, str]]:
        """Build the supported approval choices and their terminal indexes."""
        choices = {
            ApprovalChoice.DENY: "Deny",
            ApprovalChoice.ONCE: "Allow once",
            ApprovalChoice.SESSION: "Allow for this session",
        }
        index = {
            ApprovalChoice.DENY: "N",
            ApprovalChoice.ONCE: "Y",
            ApprovalChoice.SESSION: "S",
        }
        if fresh_only:
            choices.pop(ApprovalChoice.SESSION)
            index.pop(ApprovalChoice.SESSION)
        if not fresh_only and self._configuration_path is not None:
            choices[ApprovalChoice.WORKSPACE] = "Allow in this workspace"
            index[ApprovalChoice.WORKSPACE] = "W"
        if not fresh_only and self._user_configuration_path is not None:
            choices[ApprovalChoice.USER] = "Always allow for this user"
            index[ApprovalChoice.USER] = "U"
        return choices, index

    def _remember_approval(
        self,
        operations: Operations,
        *,
        scope: ApprovalChoice,
    ) -> tuple[str, ...]:
        """Install exact, deduplicated allow rules for one approved operation set."""
        if scope is ApprovalChoice.SESSION:
            with self.policy_update():
                additions = self._approval_additions(
                    operations, self._session_overrides.rules, scope
                )
                self._session_overrides.rules.extend(additions)
                if additions:
                    self._policy_revision += 1
        else:
            with self.policy_update():
                if scope is ApprovalChoice.USER:
                    current = self._user_configuration
                else:
                    current = self._configuration
                additions = self._approval_additions(operations, current.rules, scope)
                updated = current.model_copy(deep=True)
                updated.rules.extend(additions)
                if scope is ApprovalChoice.USER:
                    self._replace_user_configuration(updated)
                else:
                    self._replace_configuration(updated)
        return tuple(rule.id for rule in additions)

    async def _remember_approval_async(
        self,
        operations: Operations,
        *,
        scope: ApprovalChoice,
    ) -> RememberedApproval:
        """Return rule IDs, retained cancellation, and whether the policy was applied."""
        if scope is ApprovalChoice.SESSION:
            await self._acquire_policy_lock_async()
            try:
                return RememberedApproval(
                    self._remember_approval(operations, scope=scope), None, True
                )
            finally:
                self._policy_lock.release()

        if scope is ApprovalChoice.USER:
            path = self._user_configuration_path
        elif scope is ApprovalChoice.WORKSPACE:
            path = self._configuration_path
        else:
            raise ValueError(f"Unsupported remembered approval scope: {scope.value}.")
        if path is None:
            raise ValueError(f"Permission policy path is unavailable for {scope.value} approval.")

        owned = False
        try:
            while not owned:
                owned = self._async_policy_write_lock.acquire(blocking=False)
                if not owned:
                    await asyncio.sleep(0.01)
            return await self._remember_persistent_approval_async(
                operations, scope=scope, path=path
            )
        finally:
            if owned:
                self._async_policy_write_lock.release()

    async def _remember_persistent_approval_async(
        self,
        operations: Operations,
        *,
        scope: ApprovalChoice,
        path: Path,
    ) -> RememberedApproval:
        """Write and publish one serialized approval, retaining cancellation and applied state."""
        cancellation = None
        while True:
            await self._acquire_policy_lock_async()
            try:
                if scope is ApprovalChoice.USER:
                    current = self._user_configuration
                else:
                    current = self._configuration
                revision = self._policy_revision
                additions = self._approval_additions(operations, current.rules, scope)
                updated = current.model_copy(deep=True)
                updated.rules.extend(additions)
            finally:
                self._policy_lock.release()

            written, stopped = await self._run_blocking_settled_async(
                self._persist_configuration_candidate,
                path,
                updated,
                revision,
            )
            cancellation = cancellation or stopped
            if not written:
                if cancellation is not None:
                    return RememberedApproval((), cancellation, False)
                continue
            return RememberedApproval(tuple(rule.id for rule in additions), cancellation, True)

    @staticmethod
    def _approval_additions(
        operations: Operations,
        existing: Iterable[PermissionRule],
        scope: ApprovalChoice,
    ) -> list[PermissionRule]:
        """Build the exact allow rules not already present at the requested scope."""
        additions = []
        for operation in operations:
            target = PermissionManager._approval_target(operation.target)
            if any(
                rule.decision is Decision.ALLOW
                and rule.tool == operation.tool_id
                and (rule.tool_exact or not any(character in rule.tool for character in "*?["))
                and rule.action is operation.action
                and rule.target == target
                for rule in (*existing, *additions)
            ):
                continue
            additions.append(
                PermissionRule(
                    decision=Decision.ALLOW,
                    tool=operation.tool_id,
                    tool_exact=True,
                    action=operation.action,
                    target=target,
                    description=f"Approved interactively for this {scope.value}.",
                )
            )
        return additions

    async def _acquire_policy_lock_async(self) -> None:
        """Wait for policy ownership without blocking the current event loop."""
        while not self._policy_lock.acquire(blocking=False):
            await asyncio.sleep(_POLICY_LOCK_POLL_SECONDS)

    def _persist_configuration_candidate(
        self,
        path: Path,
        configuration: PermissionConfiguration | UserPermissionConfiguration,
        revision: int,
    ) -> bool:
        """Persist and publish one current policy candidate in the same transaction."""
        with self.policy_update():
            if revision != self._policy_revision:
                return False
            self._persist_checked(path, configuration)
            if isinstance(configuration, UserPermissionConfiguration):
                self._user_configuration = configuration
            else:
                self._configuration = configuration
            self._policy_revision += 1
            return True

    async def _run_blocking_settled_async(
        self,
        function: Callable[..., Any],
        *args: Any,
        _owned_failures: list[tuple[asyncio.CancelledError, BaseException]] | None = None,
    ) -> tuple[Any, asyncio.CancelledError | None]:
        """Wait for bounded synchronous work and retain cancellation until its outcome is known."""
        pending = asyncio.ensure_future(
            self._async_blocking_runner(function, *args)
            if self._async_blocking_runner is not None
            else run_in_worker_async(function, *args)
        )
        cancellation = None
        try:
            await asyncio.wait({pending})
        except asyncio.CancelledError as error:
            cancellation = error
        while not pending.done():
            try:
                await asyncio.wait({pending})
            except asyncio.CancelledError as error:
                cancellation = cancellation or error

        try:
            result = pending.result()
        except BaseException as error:
            if cancellation is not None:
                if _owned_failures is not None:
                    _owned_failures.append((cancellation, error))
                cancellation.add_note(
                    f"The blocking permission operation also stopped with {type(error).__name__}."
                )
                raise cancellation from error
            raise
        return result, cancellation

    @staticmethod
    def _approval_target(target: OperationTarget) -> OperationTarget:
        """Return the stable security-relevant identity of an operation target."""
        if isinstance(target, FileTarget):
            return FileTarget(path=target.path)
        if isinstance(target, NetworkTarget):
            return NetworkTarget(
                url=target.url,
                origin=target.origin,
                method=target.method,
                sends_body=target.sends_body,
            )
        return target.model_copy(deep=True)

    def evaluate(self, operations: Operations) -> PolicyDecision:
        """Evaluate a complete operation set without prompting or recording.

        Args:
            operations (Operations): Complete normalized effects to evaluate.

        Returns:
            PolicyDecision: Composed allow, ask, or deny outcome and determining sources.
        """
        with self._policy_lock:
            return self._evaluate_policy(operations)

    def evaluate_with_revision(self, operations: Operations) -> tuple[PolicyDecision, int]:
        """Evaluate operations and capture the policy revision in one transaction.

        Args:
            operations (Operations): Complete normalized effects to evaluate.

        Returns:
            tuple[PolicyDecision, int]: The policy decision and its revision.
        """
        with self._policy_lock:
            return self.evaluate(operations), self._policy_revision

    def host_gate_enable_revision(self, operations: Operations) -> int | None:
        """Identify a host request blocked only by the user-changeable eligibility gate.

        Args:
            operations (Operations): Complete exact explicit host process plan.

        Returns:
            int | None: Current policy revision if a session gate can be offered.
        """
        with self.policy_update():
            if self._namespace_recovery_decision() is not None or not operations:
                return None
            if any(
                operation.action is not Action.PROCESS_EXECUTE
                or not isinstance(operation.target, ProcessTarget)
                or operation.target.boundary is not ProcessBoundary.HOST
                for operation in operations
            ):
                return None
            if self.effective_configuration.limits.allow_host_processes:
                return None
            if self._managed_host_prohibited():
                return None
            if any(
                self._evaluate_operation_policy(operation).decision is Decision.DENY
                for operation in operations
            ):
                return None
            return self._policy_revision

    def enable_host_gate_if_current(self, operations: Operations, revision: int) -> bool:
        """Enable session host eligibility only while the reviewed policy stays current.

        Args:
            operations (Operations): Exact plan offered to the human.
            revision (int): Policy revision shown before the gate prompt.

        Returns:
            bool: Whether a session gate was enabled.
        """
        with self.policy_update():
            if self.host_gate_enable_revision(operations) != revision:
                return False
            self._session_overrides.limits = self._session_overrides.limits.model_copy(
                update={"allow_host_processes": True}
            )
            self._policy_revision += 1
        self._audit_policy_change(
            "permission.limit_set", PolicyScope.SESSION, limit="allow_host_processes", enabled=True
        )
        return True

    async def evaluate_async(self, operations: Operations) -> PolicyDecision:
        """Evaluate one operation plan without blocking on a concurrent policy update.

        Retry a bounded number of times if policy changes across an awaited boundary; persistent
        churn returns a fail-closed stale decision.

        Args:
            operations (Operations): Complete normalized effects to evaluate.

        Returns:
            PolicyDecision: Composed allow, ask, or deny outcome and determining sources.

        Raises:
            asyncio.CancelledError: If boundary checks are cancelled after filesystem work settles.
        """
        result, _ = await self.evaluate_with_revision_async(operations)
        return result

    async def evaluate_with_revision_async(
        self, operations: Operations
    ) -> tuple[PolicyDecision, int | None]:
        """Evaluate operations and capture a revision without blocking policy updates.

        Args:
            operations (Operations): Complete normalized effects to evaluate.

        Returns:
            tuple[PolicyDecision, int | None]: Decision and revision, or a stale denial and None.

        Raises:
            asyncio.CancelledError: If boundary checks are cancelled after filesystem work settles.
        """
        for _ in range(_ASYNC_POLICY_ATTEMPTS):
            await self._acquire_policy_lock_async()
            try:
                revision = self._policy_revision
                result = await self._evaluate_policy_async(operations)
                if revision == self._policy_revision:
                    return result, revision
            finally:
                self._policy_lock.release()
        return self._stale_policy_decision(), None

    @staticmethod
    def _stale_policy_decision() -> PolicyDecision:
        """Fail closed when policy keeps changing during asynchronous evaluation."""
        return PolicyDecision(
            decision=Decision.DENY,
            reason="Permission policy changed repeatedly during evaluation.",
            sources=("policy:stale",),
        )

    async def _evaluate_policy_async(self, operations: Operations) -> PolicyDecision:
        """Evaluate one policy pass with filesystem boundaries off the owner loop."""
        managed_present = (
            self._managed_policy is not None
            or self._managed_error is not None
            or self._managed_policy_store.may_be_installed()
        )
        if (
            self._namespace_catalog_path is not None
            or self._configuration_path is not None
            or self._user_configuration_path is not None
            or self._namespace_errors
            or managed_present
        ):
            namespace, cancellation = await self._run_blocking_settled_async(
                self._namespace_recovery_decision
            )
            if cancellation is not None:
                raise cancellation
            if namespace is not None:
                return namespace
        if not operations:
            return self._compose_policy_decisions(operations, ())
        decisions = []
        for operation in operations:
            boundary = await self._boundary_decision_async(operation)
            decisions.append(
                boundary if boundary is not None else self._evaluate_operation_policy(operation)
            )
        return self._compose_policy_decisions(operations, tuple(decisions))

    async def _boundary_decision_async(self, operation: Operation) -> PolicyDecision | None:
        """Resolve filesystem boundaries off-loop and evaluate pure boundaries in place."""
        if not isinstance(operation.target, FileTarget):
            return self._boundary_decision(operation)
        denial, cancellation = await self._run_blocking_settled_async(
            self._boundary_decision, operation
        )
        if cancellation is not None:
            raise cancellation
        return denial

    def _evaluate_policy(self, operations: Operations) -> PolicyDecision:
        """Evaluate a policy snapshot while the caller owns the policy lock."""
        if namespace := self._namespace_recovery_decision():
            return namespace
        decisions = tuple(self._evaluate_operation(operation) for operation in operations)
        return self._compose_policy_decisions(operations, decisions)

    def _namespace_recovery_decision(self) -> PolicyDecision | None:
        """Refuse pending resets and stale policy namespaces before admission."""
        if error := self._managed_current_error():
            return PolicyDecision(
                decision=Decision.DENY,
                reason=f"Administrator policy recovery required: {error}",
                sources=("managed:recovery_required",),
            )
        for scope in ("workspace", "user"):
            if error := self._namespace_errors.get(scope):
                return PolicyDecision(
                    decision=Decision.DENY,
                    reason=f"Permission policy recovery required: {error}",
                    sources=(f"namespace:{scope}:recovery_required",),
                )
        if changed := self._live_namespace_error():
            scope, error = changed
            self._namespace_errors[scope] = error
            self._policy_revision += 1
            return PolicyDecision(
                decision=Decision.DENY,
                reason=f"Permission policy recovery required: {error}",
                sources=(f"namespace:{scope}:recovery_required",),
            )
        return None

    def require_policy_ready(self) -> None:
        """Stop interactive startup when a trusted policy requires recovery.

        Raises:
            PermissionNamespaceError: If a managed or durable policy cannot admit tools.
        """
        with self.policy_update():
            denial = self._namespace_recovery_decision()
            if denial is None:
                return
            source = denial.sources[0]
            if source == "managed:recovery_required":
                path = self._managed_policy_store.expectation_path or "administrator policy"
                mitigation = (
                    "Have an administrator repair the protected policy, then reload it or "
                    "restart the application."
                )
            elif source == "namespace:user:recovery_required":
                path = self._user_configuration_path or "user policy"
                mitigation = (
                    "Restore or repair the user policy and registry, then reload or restart."
                )
            else:
                path = self._configuration_path or "workspace policy"
                mitigation = (
                    "Restore or repair the workspace policy and registry, then reload or restart."
                )
            raise PermissionNamespaceError(path, f"{denial.reason} {mitigation}")

    def suspend_policy(self, scope: str, reason: str) -> None:
        """Block permission admission until a failed policy scope is successfully reloaded.

        Args:
            scope (str): Exact ``workspace`` or ``user`` policy scope.
            reason (str): Safe explanation of the loading failure.

        Raises:
            ValueError: If the scope or diagnostic is invalid.
        """
        if scope not in {"workspace", "user"} or not reason:
            raise ValueError("A valid policy scope and recovery reason are required.")
        with self.policy_update():
            self._namespace_errors[scope] = reason
            self._policy_revision += 1

    @staticmethod
    def _compose_policy_decisions(
        operations: Operations,
        decisions: tuple[PolicyDecision, ...],
    ) -> PolicyDecision:
        """Compose operation outcomes using the established batch precedence."""
        if not operations:
            return PolicyDecision(
                decision=Decision.ALLOW,
                reason="The tool plan requires no authority.",
                sources=("no_authority",),
            )
        if len(decisions) == 1:
            return decisions[0]
        for selected in (Decision.DENY, Decision.ASK, Decision.ALLOW):
            determining = tuple(item for item in decisions if item.decision is selected)
            if determining:
                return PolicyDecision(
                    decision=selected,
                    reason=(
                        f"{len(determining)} of {len(operations)} planned operation(s) "
                        f"require {selected.value}."
                    ),
                    sources=tuple(source for item in determining for source in item.sources),
                )
        raise RuntimeError("Every operation must produce a policy decision.")  # pragma: no cover

    def set_default(
        self,
        action: Action,
        decision: Decision,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> None:
        """Set the fallback decision for one action.

        Args:
            action (Action): Action whose fallback changes.
            decision (Decision): New fallback outcome.
            scope (PolicyScope): Workspace persistence or process-local lifetime.
        """
        if scope is PolicyScope.WORKSPACE:
            with self.policy_update():
                updated = self._configuration.model_copy(deep=True)
                updated.defaults[action] = decision
                self._replace_configuration(updated)
        else:
            with self.policy_update():
                self._session_overrides.defaults[action] = decision
                self._policy_revision += 1
        self._audit_policy_change(
            "permission.default_set",
            scope,
            action=action.value,
            decision=decision.value,
        )

    def reset_default(self, action: Action, *, scope: PolicyScope = PolicyScope.SESSION) -> bool:
        """Reset one default to its inherited or application value.

        Args:
            action (Action): Action whose default resets.
            scope (PolicyScope): Workspace bootstrap or session inheritance target.

        Returns:
            bool: Whether the selected layer changed.
        """
        if scope is PolicyScope.SESSION:
            with self.policy_update():
                changed = self._session_overrides.defaults.pop(action, None) is not None
                if changed:
                    self._policy_revision += 1
            if changed:
                self._audit_policy_change("permission.default_reset", scope, action=action.value)
            return changed
        default = PermissionConfiguration().defaults[action]
        with self.policy_update():
            if self._configuration.defaults.get(action, Decision.DENY) is default:
                return False
            updated = self._configuration.model_copy(deep=True)
            updated.defaults[action] = default
            self._replace_configuration(updated)
        self._audit_policy_change("permission.default_reset", scope, action=action.value)
        return True

    def add_rule(
        self,
        rule: PermissionRule,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> None:
        """Add one workspace or session-scoped policy rule.

        Args:
            rule (PermissionRule): Rule to add.
            scope (PolicyScope): Workspace persistence or process-local lifetime.

        Raises:
            ValueError: If another active rule already uses the same identifier.
        """
        rule = rule.model_copy(deep=True)
        with self.policy_update():
            if any(
                existing.id == rule.id
                for existing in (*self._configuration.rules, *self._session_overrides.rules)
            ):
                raise ValueError(f"Permission rule '{rule.id}' already exists.")
            if scope is PolicyScope.WORKSPACE:
                updated = self._configuration.model_copy(deep=True)
                updated.rules.append(rule)
                self._replace_configuration(updated)
            else:
                self._session_overrides.rules.append(rule)
                self._policy_revision += 1
        self._audit_policy_change(
            "permission.rule_added", scope, rule_id=rule.id, decision=rule.decision.value
        )

    def remove_rule(
        self,
        rule_id: str,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> bool:
        """Remove one rule by stable identifier.

        Args:
            rule_id (str): Identifier of the rule to remove.
            scope (PolicyScope): Workspace or session layer to search.

        Returns:
            bool: Whether a rule was removed.
        """
        if scope is PolicyScope.SESSION:
            with self.policy_update():
                rules = self._session_overrides.rules
                removed = next(
                    (index for index, rule in enumerate(rules) if rule.id == rule_id), None
                )
                if removed is not None:
                    rules.pop(removed)
                    self._policy_revision += 1
        else:
            with self.policy_update():
                rules = list(self._configuration.rules)
                removed = next(
                    (index for index, rule in enumerate(rules) if rule.id == rule_id), None
                )
                if removed is not None:
                    rules.pop(removed)
                    updated = self._configuration.model_copy(update={"rules": rules}, deep=True)
                    self._replace_configuration(updated)
        if removed is not None:
            self._audit_policy_change("permission.rule_removed", scope, rule_id=rule_id)
            return True
        return False

    def preset(self, preset_id: str) -> PermissionPreset:
        """Return one named permission preset artifact.

        Args:
            preset_id (str): Stable preset catalog identifier.

        Returns:
            PermissionPreset: Deep copy of the selected artifact.

        Raises:
            ValueError: If no configured preset uses this identifier.
        """
        try:
            return self._presets[preset_id].model_copy(deep=True)
        except KeyError as exc:
            raise ValueError(f"Unknown permission preset '{preset_id}'.") from exc

    def preview_preset_replacement(
        self,
        preset_id: str,
        *,
        scope: PolicyScope,
    ) -> PresetReplacementPreview:
        """Describe replacement of one scoped defaults-and-rules layer without mutating it.

        Limits and the non-selected policy layer are deliberately absent from this operation. The
        returned preview carries a revision that rejects stale replacements.

        Args:
            preset_id (str): Stable identifier of the selected preset.
            scope (PolicyScope): Workspace persistence or process-local replacement target.

        Returns:
            PresetReplacementPreview: Exact default and rule replacement plus a scope revision.
        """
        preset = self.preset(preset_id)
        with self.policy_update():
            return PresetReplacementPreview(
                preset=preset,
                scope=scope,
                removed_defaults=self._defaults_for_scope(scope),
                installed_defaults=dict(preset.defaults),
                removed_rules=self._rules_for_scope(scope),
                installed_rules=self._rules_from_preset(preset, scope),
                scope_revision=self._scope_revision(scope),
            )

    def replace_preset(self, preview: PresetReplacementPreview) -> None:
        """Atomically replace the defaults and rules described by a current preview.

        Args:
            preview (PresetReplacementPreview): Previously reviewed scoped replacement.

        Raises:
            ValueError: If the preview is stale or installs an identifier active in another layer.
        """
        with self.policy_update():
            self._replace_preset_locked(preview)
        self._audit_policy_change(
            "permission.preset_replaced",
            preview.scope,
            preset_id=preview.preset.metadata.id,
            revision=preview.preset.metadata.revision,
        )

    def _replace_preset_locked(self, preview: PresetReplacementPreview) -> None:
        """Validate and publish a scoped preset while policy writers are serialized."""
        session_revision = None
        host_rules_revision = None
        if preview.scope is PolicyScope.SESSION:
            session_revision = self._policy_revision
            with self._host_rules_lock:
                host_rules_revision = self._host_rules_revision
        if preview.scope_revision != self._scope_revision(preview.scope):
            raise ValueError("Permission preset preview is stale; generate a new preview.")
        replacement = [rule.model_copy(deep=True) for rule in preview.installed_rules]
        other_scope = (
            PolicyScope.SESSION if preview.scope is PolicyScope.WORKSPACE else PolicyScope.WORKSPACE
        )
        other_identifiers = {rule.id for rule in self._rules_for_scope(other_scope)}
        collision = next((rule.id for rule in replacement if rule.id in other_identifiers), None)
        if collision is not None:
            raise ValueError(
                f"Permission rule '{collision}' already exists in {other_scope.value}."
            )
        if preview.scope is PolicyScope.WORKSPACE:
            with self.policy_update():
                if preview.scope_revision != self._scope_revision(preview.scope):
                    raise ValueError("Permission preset preview is stale; generate a new preview.")
                updated = self._configuration.model_copy(
                    update={"defaults": preview.installed_defaults, "rules": replacement}, deep=True
                )
                self._replace_configuration(updated)
        else:
            with self._host_rules_lock:
                if (
                    session_revision != self._policy_revision
                    or host_rules_revision != self._host_rules_revision
                ):
                    raise ValueError("Permission preset preview is stale; generate a new preview.")
                self._session_overrides.defaults = dict(preview.installed_defaults)
                self._session_overrides.rules = replacement
                self._policy_revision += 1

    def set_limit(
        self,
        name: str,
        value: bool,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> None:
        """Set one boolean enforcement limit.

        Args:
            name (str): ``deny_private_networks``, ``deny_local_networks``, or
                ``allow_host_processes``.
            value (bool): New boolean value.
            scope (PolicyScope): Workspace persistence or process-local lifetime.

        Raises:
            ValueError: If ``name`` is not a supported boolean limit.
        """
        if name not in {"deny_private_networks", "deny_local_networks", "allow_host_processes"}:
            raise ValueError(f"Unknown boolean permission limit '{name}'.")
        if scope is PolicyScope.WORKSPACE:
            with self.policy_update():
                updated = self._configuration.model_copy(deep=True)
                updated.limits = updated.limits.model_copy(update={name: value})
                self._replace_configuration(updated)
        else:
            with self.policy_update():
                self._session_overrides.limits = self._session_overrides.limits.model_copy(
                    update={name: value}
                )
                self._policy_revision += 1
        self._audit_policy_change("permission.limit_set", scope, limit=name, enabled=value)

    def update_limit_values(
        self,
        name: str,
        value: str,
        *,
        add: bool,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> bool:
        """Add or remove one filesystem root or network origin limit.

        Args:
            name (str): ``readable_roots``, ``writable_roots``, or ``network_origins``.
            value (str): Root token/path or origin glob to update.
            add (bool): Add when true; remove when false.
            scope (PolicyScope): Workspace persistence or process-local lifetime.

        Returns:
            bool: Whether the configured collection changed.

        Raises:
            ValueError: If ``name`` is unsupported.
        """
        if name not in {"readable_roots", "writable_roots", "network_origins"}:
            raise ValueError(f"Unknown collection permission limit '{name}'.")
        if name == "network_origins" or value in {
            "workspace",
            "app-temp",
            "system-temp",
        }:
            normalized = value
        else:
            path = Path(value)
            if not path.is_absolute() and self._workspace_root is not None:
                path = self._workspace_root / path
            normalized = canonical_path(path)

        def revised_values(current: tuple[str, ...]) -> tuple[str, ...] | None:
            """Build the updated collection or report that no change is needed."""
            values = list(current)
            if add:
                if normalized in values:
                    return None
                if name == "network_origins" and values == ["*"] and normalized != "*":
                    values.clear()
                values.append(normalized)
            else:
                if normalized not in values:
                    return None
                values.remove(normalized)
            return tuple(values)

        if scope is PolicyScope.WORKSPACE:
            with self.policy_update():
                values = revised_values(getattr(self._configuration.limits, name))
                if values is None:
                    return False
                updated = self._configuration.model_copy(deep=True)
                updated.limits = updated.limits.model_copy(update={name: values})
                updated.limit_classifications = updated.limit_classifications.model_copy(
                    update={
                        name: LimitClassification.HARD,
                        "sources": {**updated.limit_classifications.sources, name: "user"},
                    }
                )
                self._replace_configuration(updated)
        else:
            with self.policy_update():
                current = getattr(self._session_overrides.limits, name)
                if current is None:
                    current = getattr(self._configuration.limits, name)
                values = revised_values(current)
                if values is None:
                    return False
                self._session_overrides.limits = self._session_overrides.limits.model_copy(
                    update={name: values}
                )
                self._session_overrides.limit_classifications[name] = LimitClassification.HARD
                self._policy_revision += 1
        self._audit_policy_change(
            "permission.limit_value_updated",
            scope,
            limit=name,
            operation="add" if add else "remove",
        )
        return True

    def classify_limit(
        self,
        name: str,
        classification: LimitClassification,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> None:
        """Record a trusted choice for one scoped collection limit.

        Args:
            name (str): Read roots, write roots, or network origins.
            classification (LimitClassification): Hard or approval-controlled behavior.
            scope (PolicyScope): Durable workspace or current session.

        Raises:
            ValueError: If the field or classification is invalid.
        """
        if name not in {"readable_roots", "writable_roots", "network_origins"}:
            raise ValueError(f"Unknown collection permission limit '{name}'.")
        selected = LimitClassification(classification)
        with self.policy_update():
            if scope is PolicyScope.WORKSPACE:
                updated = self._configuration.model_copy(deep=True)
                updated.limit_classifications = updated.limit_classifications.model_copy(
                    update={
                        name: selected,
                        "sources": {**updated.limit_classifications.sources, name: "user"},
                    }
                )
                self._replace_configuration(updated)
            else:
                self._session_overrides.limit_classifications[name] = selected
                self._policy_revision += 1
        self._audit_policy_change(
            "permission.limit_classified", scope, limit=name, classification=selected.value
        )

    def reset_limit(self, name: str, *, scope: PolicyScope = PolicyScope.SESSION) -> bool:
        """Reset one limit to its inherited or application value.

        Args:
            name (str): Field in ``PolicyLimits`` to reset.
            scope (PolicyScope): Workspace bootstrap or session inheritance target.

        Returns:
            bool: Whether the selected layer changed.

        Raises:
            ValueError: If ``name`` is not a policy limit.
        """
        if name not in _LIMIT_NAMES:
            raise ValueError(f"Unknown permission limit '{name}'.")
        if scope is PolicyScope.SESSION:
            with self.policy_update():
                if getattr(self._session_overrides.limits, name) is None:
                    return False
                self._session_overrides.limits = self._session_overrides.limits.model_copy(
                    update={name: None}
                )
                self._policy_revision += 1
            self._audit_policy_change("permission.limit_reset", scope, limit=name)
            return True
        default = getattr(PolicyLimits(), name)
        with self.policy_update():
            if getattr(self._configuration.limits, name) == default:
                return False
            updated = self._configuration.model_copy(deep=True)
            updated.limits = updated.limits.model_copy(update={name: default})
            self._replace_configuration(updated)
        self._audit_policy_change("permission.limit_reset", scope, limit=name)
        return True

    def reset_session(self) -> bool:
        """Clear every process-local default, limit, and rule override.

        Returns:
            bool: Whether any session policy state was cleared.
        """
        with self.policy_update(), self._host_rules_lock:
            changed = self._session_overrides != SessionPolicyOverrides()
            self._session_overrides = SessionPolicyOverrides()
            self._host_runtime_bindings.clear()
            self._host_rules_revision += 1
            if changed:
                self._policy_revision += 1
        if changed:
            self._audit_policy_change("permission.session_reset", PolicyScope.SESSION)
        return changed

    def reload(
        self,
        *,
        interaction: Interaction | None = None,
    ) -> PermissionLoadResult:
        """Reload the workspace policy according to the configured failure policy.

        Retain the active workspace policy when its configured file is missing.

        Args:
            interaction (Interaction | None): Invocation interaction used for recovery prompts, or
                ``None`` to use the manager's configured interaction.

        Returns:
            PermissionLoadResult: Whether new policy loaded or the active policy was retained.

        Raises:
            PermissionConfigurationError: If loading fails in strict mode or a pending reset
                requires recovery.
            ShutdownRequested: If the user exits interactive recovery.
        """
        active_interaction = interaction if interaction is not None else self._interaction
        with self.policy_update():
            self._load_managed_policy()
            self._policy_revision += 1
            telemetry_audit(
                "permission.managed_reloaded",
                status=(
                    "invalid"
                    if self._managed_error
                    else "active"
                    if self._managed_policy is not None
                    else "unavailable"
                ),
            )
        result = self._load_configuration(
            retain_on_failure=True,
            interaction=active_interaction,
        )
        if self._load_user_configuration(interaction=active_interaction):
            return PermissionLoadResult.BLOCKED
        return result

    def reset_configuration(self, scope: str) -> Path | None:
        """Archive one policy and replace it with its built-in defaults.

        Leave a pending marker if the reset does not complete so later file-based loads refuse.

        Args:
            scope (str): Exact ``workspace`` or ``user`` policy namespace to reset.

        Returns:
            Path | None: Backup path for the prior policy, or None when no file existed.

        Raises:
            ValueError: If the scope is invalid or its configuration path is unavailable.
            PermissionNamespaceError: If a durable policy namespace requires recovery.
            OSError: If the policy is stale or busy, or reset state cannot be written.
        """
        with self.policy_update():
            failed_revision = (
                self._failed_workspace_revision
                if scope == "workspace"
                else self._failed_user_revision
            )
            failure = None
            if self._namespace_catalog_path is None and failed_revision is not None:
                failure = self._namespace_errors.pop(scope, None)
            try:
                return self._reset_configuration(scope)
            except BaseException:
                if failure is not None:
                    self._namespace_errors[scope] = failure
                raise

    def _reset_configuration(self, scope: str) -> Path | None:
        """Reset a policy while its owner lock is held."""
        if scope not in {"workspace", "user"}:
            raise ValueError(f"Unknown permission policy scope '{scope}'.")
        path = self._configuration_path if scope == "workspace" else self._user_configuration_path
        if path is None:
            if scope == "workspace":
                raise ValueError("An in-memory PermissionManager cannot reset configuration.")
            raise ValueError("A user permission policy path is not configured.")
        if error := self._namespace_errors.get(scope):
            raise PermissionNamespaceError(path, error)
        with self.policy_update():
            revision = self._policy_revision
            expected = (
                (
                    self._failed_workspace_revision
                    if self._failed_workspace_revision is not None
                    else self._workspace_storage_revision
                )
                if scope == "workspace"
                else (
                    self._failed_user_revision
                    if self._failed_user_revision is not None
                    else self._user_storage_revision
                )
            )
            with self._storage_lock(path):
                if revision != self._policy_revision:
                    raise OSError("Permission policy changed; reload before retrying.")
                if namespace := self._namespace_recovery_decision():
                    raise PermissionNamespaceError(path, namespace.reason)
                self._check_storage_revision(path, expected)
                marker = path.with_name(path.name + _RESET_PENDING_SUFFIX)
                descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
                backup_path = None
                if path.exists():
                    timestamp = local_now().strftime("%Y%m%dT%H%M%S%f%z")
                    backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
                    descriptor = os.open(backup_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    try:
                        placeholder = os.fstat(descriptor)
                    finally:
                        os.close(descriptor)
                    try:
                        path.replace(backup_path)
                    except BaseException:
                        try:
                            current = backup_path.lstat()
                        except OSError:
                            pass
                        else:
                            if (current.st_dev, current.st_ino) == (
                                placeholder.st_dev,
                                placeholder.st_ino,
                            ):
                                backup_path.unlink(missing_ok=True)
                        raise
                configuration = (
                    PermissionConfiguration(
                        namespace_generation=self._configuration.namespace_generation
                    )
                    if scope == "workspace"
                    else UserPermissionConfiguration(
                        namespace_generation=self._user_configuration.namespace_generation
                    )
                )
                saved_revision = self._persist_configuration(path, configuration)
                marker.unlink()
                if scope == "workspace":
                    self._workspace_storage_revision = saved_revision
                    self._configuration = configuration
                    self._failed_workspace_revision = None
                else:
                    self._user_storage_revision = saved_revision
                    self._failed_user_revision = None
                    self._user_configuration = configuration
                self._policy_revision += 1
        self._audit_policy_change(
            "permission.configuration_reset",
            PolicyScope.WORKSPACE if scope == "workspace" else PolicyScope.SESSION,
            namespace=scope,
        )
        return backup_path

    def initialize_policy_namespace(self, scope: str, identity: str, *, authorized: bool) -> None:
        """Initialize a missing policy namespace after a trusted explicit action.

        Args:
            scope (str): Exact `workspace` or `user` namespace.
            identity (str): Expected workspace UUID or user-data root.
            authorized (bool): Whether a trusted human or authenticated deployment client
                supplied this explicit operation.

        Raises:
            ValueError: If durable namespaces are unavailable or scope is unknown.
            PermissionNamespaceError: If authorization, identity, or existing state conflicts.
        """
        catalog = self._namespace_catalog_path
        if catalog is None:
            raise ValueError("Durable permission namespaces are not configured.")
        if scope == "workspace":
            path = self._configuration_path
            expected = self._workspace_id
        elif scope == "user":
            path = self._user_configuration_path
            expected = self._user_namespace_identity
        else:
            raise ValueError(f"Unknown permission namespace scope '{scope}'.")
        if path is None or not authorized or identity != expected:
            raise PermissionNamespaceError(
                path or catalog,
                "Trusted authorization and the exact namespace identity are required.",
            )
        self._validate_namespace_catalog(path, catalog)
        with self._storage_lock(path):
            if os.path.lexists(path) or self._policy_backup_exists(path):
                raise PermissionNamespaceError(
                    path,
                    "Existing policy or backup data requires restore or recovery, "
                    "not initialization.",
                )
            if os.path.lexists(path.with_name(path.name + _RESET_PENDING_SUFFIX)):
                raise PermissionNamespaceError(path, "A pending policy reset requires recovery.")
            with closing(sqlite3.connect(catalog, timeout=5)) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS policy_namespaces ("
                    "scope TEXT NOT NULL, identity TEXT NOT NULL, generation TEXT NOT NULL, "
                    "status TEXT NOT NULL, legacy_hash TEXT, PRIMARY KEY(scope, identity))"
                )
                existing = connection.execute(
                    "SELECT status FROM policy_namespaces WHERE scope=? AND identity=?",
                    (scope, identity),
                ).fetchone()
                if existing is not None:
                    raise PermissionNamespaceError(
                        path,
                        "Existing namespace reservation requires recovery, not initialization.",
                    )
                if scope == "user" and identity != str(catalog.parent):
                    raise PermissionNamespaceError(
                        path, "User identity does not match the independent registry."
                    )
                active = connection.execute(
                    "SELECT canonical_path FROM workspace_locations WHERE workspace_id=? "
                    "AND status='active'",
                    (self._workspace_id,),
                ).fetchall()
                if (
                    len(active) != 1
                    or self._workspace_root is None
                    or active[0][0] != str(self._workspace_root)
                ):
                    raise PermissionNamespaceError(
                        path, "Workspace identity does not match the independent registry."
                    )
                connection.execute(
                    "INSERT INTO policy_namespaces(scope,identity,generation,status,legacy_hash) "
                    "VALUES (?,?,?,?,NULL)",
                    (scope, identity, uuid4().hex, "reserved_new"),
                )
        configuration = self._read_namespaced_configuration(path, scope, identity)
        with self.policy_update():
            if scope == "workspace":
                self._configuration = configuration
            else:
                self._user_configuration = configuration
            self._namespace_errors.pop(scope, None)
            self._policy_revision += 1
        self._audit_policy_change(
            "permission.namespace_initialized",
            PolicyScope.WORKSPACE if scope == "workspace" else PolicyScope.SESSION,
            namespace=scope,
        )

    def explain(self, tool: str, action: Action, resource: str) -> PolicyDecision:
        """Evaluate one concrete operation without prompting or recording.

        Args:
            tool (str): Registered tool identity to evaluate.
            action (Action): Typed action to evaluate.
            resource (str): Path, URL, command, or session identifier.

        Returns:
            PolicyDecision: Effective policy decision and determining sources.
        """
        if action in _READ_ACTIONS | _WRITE_ACTIONS:
            path = Path(resource)
            if not path.is_absolute() and self._workspace_root is not None:
                path = self._workspace_root / path
            target = FileTarget(path=canonical_path(path))
        elif action is Action.NETWORK_REQUEST:
            parsed = urlsplit(resource)
            if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
                raise ValueError("Network explanation requires an absolute HTTP(S) URL.")
            origin = f"{parsed.scheme}://{parsed.hostname}"
            if parsed.port is not None:
                origin += f":{parsed.port}"
            target = NetworkTarget(url=resource, origin=origin)
        elif action is Action.PROCESS_EXECUTE:
            argv = tuple(shlex.split(resource))
            if not argv:
                raise ValueError("Process explanation requires a non-empty command line.")
            target = ProcessTarget(
                argv=argv,
                cwd=str(self._workspace_root or Path.cwd()),
                boundary=ProcessBoundary.HOST,
            )
        else:
            target = SessionTarget(identifier=resource)
        return self.evaluate((Operation(tool_id=tool, action=action, target=target),))

    def save(self) -> None:
        """Persist the complete policy atomically.

        Raises:
            ValueError: If this manager has no configuration path.
            OSError: If the policy is stale or busy, or the configuration cannot be written.
        """
        self._persist(self._configuration)

    def _persist(self, configuration: PermissionConfiguration) -> None:
        if self._configuration_path is None:
            raise ValueError("An in-memory PermissionManager cannot persist configuration.")
        with self.policy_update():
            self._persist_checked(self._configuration_path, configuration)

    @staticmethod
    def _read_storage_revision(path: Path) -> bytes | None:
        """Read the exact policy bytes, distinguishing a missing policy from an empty file."""
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        return path.read_bytes()

    @staticmethod
    @contextmanager
    def _storage_lock(path: Path) -> Iterator[None]:
        """Exclusively own the stable advisory lock for one cooperative policy transaction."""
        try:
            locking = import_module("fcntl")
            windows_lock = False
        except ModuleNotFoundError:
            try:
                locking = import_module("msvcrt")
                windows_lock = True
            except ModuleNotFoundError as error:
                raise OSError(errno.ENOTSUP, "Native policy locking is unavailable.") from error
        path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        lock = path.with_name(path.name + ".lock")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NONBLOCK", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        else:
            try:
                existing = lock.lstat()
            except FileNotFoundError:
                flags |= os.O_EXCL
            else:
                if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                    raise OSError("Unsafe policy lock file.")
        descriptor = os.open(lock, flags, 0o600)
        try:
            details = os.fstat(descriptor)
            current = lock.lstat()
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or (details.st_dev, details.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise OSError("Unsafe policy lock file.")
            if windows_lock:
                locking.locking(descriptor, locking.LK_NBLCK, 1)
            else:
                locking.flock(descriptor, locking.LOCK_EX | locking.LOCK_NB)
            yield
        finally:
            os.close(descriptor)

    @staticmethod
    def _policy_backup_exists(path: Path) -> bool:
        """Report backup evidence, refusing an unverifiable policy directory listing."""
        try:
            with os.scandir(path.parent) as entries:
                return any(
                    entry.name.startswith(f"{path.name}.") and entry.name.endswith(".bak")
                    for entry in entries
                )
        except OSError as error:
            raise PermissionNamespaceError(
                path, "Policy backup evidence cannot be inspected."
            ) from error

    @staticmethod
    def _archive_policy_revision(path: Path, revision: bytes, name: str, label: str) -> None:
        """Preserve exact policy bytes before a recoverable namespace migration."""
        backup = path.with_name(f"{path.name}.{name}.bak")
        try:
            descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                descriptor = os.open(backup, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as stream:
                    details = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(details.st_mode)
                        or details.st_nlink != 1
                        or details.st_uid != os.getuid()
                        or details.st_mode & 0o022
                        or stream.read() != revision
                    ):
                        raise PermissionNamespaceError(
                            path, f"{label} policy backup generation does not match."
                        ) from None
            except OSError as error:
                raise PermissionNamespaceError(
                    path, f"{label} policy backup provenance is invalid."
                ) from error
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(revision)
                stream.flush()
                os.fsync(stream.fileno())

    def _check_storage_revision(self, path: Path, expected: bytes | None) -> None:
        """Reject a policy changed since its active revision was loaded."""
        if self._namespace_catalog_path is not None:
            current, mode = self._read_namespaced_file(path)
            if mode is None or mode & 0o022:
                raise PermissionNamespaceError(path, "Policy file provenance is invalid.")
        else:
            current = self._read_storage_revision(path)
        if current != expected:
            raise OSError("Permission policy changed; reload before retrying.")

    def _persist_checked(
        self, path: Path, configuration: PermissionConfiguration | UserPermissionConfiguration
    ) -> None:
        """Publish only when the manager still owns the loaded storage revision."""
        user_scope = isinstance(configuration, UserPermissionConfiguration)
        if user_scope and (error := self._namespace_errors.get("user")):
            raise PermissionNamespaceError(path, error)
        if not user_scope and (error := self._namespace_errors.get("workspace")):
            raise PermissionNamespaceError(path, error)
        expected = self._user_storage_revision if user_scope else self._workspace_storage_revision
        revision = self._policy_revision
        with self._storage_lock(path):
            if revision != self._policy_revision:
                raise OSError("Permission policy changed; reload before retrying.")
            if namespace := self._namespace_recovery_decision():
                raise PermissionNamespaceError(path, namespace.reason)
            self._require_completed_reset(path)
            self._check_storage_revision(path, expected)
            revision = self._persist_configuration(path, configuration)
            if user_scope:
                self._user_storage_revision = revision
                self._failed_user_revision = None
            else:
                self._workspace_storage_revision = revision
                self._failed_workspace_revision = None

    @staticmethod
    def _persist_configuration(
        path: Path,
        configuration: PermissionConfiguration | UserPermissionConfiguration,
    ) -> bytes:
        """Stage a policy without following an existing final-component link, then replace it.

        Without no-follow support, an existing staging entry prevents the write.
        """
        path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        payload = configuration.model_dump(mode="json")
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        content = f"# app-revision: {uuid4().hex}\n" + dump_yaml(payload)
        flags = os.O_WRONLY | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW | os.O_NONBLOCK
        else:
            flags |= os.O_EXCL
        descriptor = os.open(temporary_path, flags, 0o600)
        try:
            details = os.fstat(descriptor)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or details.st_uid != os.getuid()
                or details.st_mode & 0o022
            ):
                raise OSError("Unsafe policy staging file.")
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                descriptor = -1
                os.ftruncate(stream.fileno(), 0)
                stream.write(content)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        temporary_path.replace(path)
        return content.encode("utf-8")

    def _replace_configuration(self, configuration: PermissionConfiguration) -> None:
        """Persist and activate one workspace policy transactionally."""
        with self.policy_update():
            self._persist(configuration)
            with self.policy_update():
                self._configuration = configuration
                self._policy_revision += 1

    def _replace_user_configuration(self, configuration: UserPermissionConfiguration) -> None:
        """Persist and activate one user-wide remembered-approval policy transactionally."""
        configuration = UserPermissionConfiguration.model_validate(configuration.model_dump())
        if self._user_configuration_path is None:
            raise ValueError("A user permission policy path is not configured.")
        with self.policy_update():
            self._persist_checked(self._user_configuration_path, configuration)
            with self.policy_update():
                self._user_configuration = configuration
                self._policy_revision += 1

    def _rules_for_scope(self, scope: PolicyScope) -> tuple[PermissionRule, ...]:
        """Return deep-copied rules from one exact policy layer."""
        rules = (
            self._configuration.rules
            if scope is PolicyScope.WORKSPACE
            else self._session_overrides.rules
        )
        return tuple(rule.model_copy(deep=True) for rule in rules)

    def _defaults_for_scope(self, scope: PolicyScope) -> dict[Action, Decision]:
        """Return a deep-copied default map from one exact policy layer."""
        defaults = (
            self._configuration.defaults
            if scope is PolicyScope.WORKSPACE
            else self._session_overrides.defaults
        )
        return dict(defaults)

    @staticmethod
    def _rules_from_preset(
        preset: PermissionPreset,
        scope: PolicyScope,
    ) -> tuple[PermissionRule, ...]:
        """Materialize one artifact into collision-resistant, provenance-bearing policy rules."""
        metadata = preset.metadata
        return tuple(
            PermissionRule(
                id=(f"preset:{scope.value}:{metadata.id}:{metadata.revision}:{rule.id}"),
                decision=rule.decision,
                description=rule.description,
                tool=rule.tool,
                action=rule.action,
                resource=rule.resource,
                source=PresetSource(
                    preset_id=metadata.id,
                    revision=metadata.revision,
                    content_hash=preset.content_hash,
                    rule_id=rule.id,
                ),
            )
            for rule in preset.rules
        )

    def _scope_revision(self, scope: PolicyScope) -> str:
        """Return a semantic revision of the exact layer a replacement would overwrite."""
        if scope is PolicyScope.WORKSPACE:
            payload = self._configuration.model_dump(mode="json")
        else:
            with self._host_rules_lock:
                host_rules = tuple(self._session_overrides.host_command_rules)
            snapshot = self._session_overrides.model_copy(
                update={"host_command_rules": list(host_rules)}
            )
            payload = snapshot.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{sha256_digest(encoded)}"

    def describe(self, view: str = "all") -> str:
        """Return a user-facing summary of selected policy layers.

        Args:
            view (str): ``all``, ``workspace``, ``session``, ``effective``, or ``managed``.

        Returns:
            str: Requested defaults, limits, rules, and their policy sources.

        Raises:
            ValueError: If ``view`` is not supported.
        """
        if view not in {"all", "workspace", "session", "effective", "managed"}:
            raise ValueError(f"Unknown permission policy view '{view}'.")
        with self.policy_update():
            self._managed_current_error()
        path = str(self._configuration_path) if self._configuration_path else "in memory"
        lines = [f"Permission policy: {path}"]
        if self._managed_error is not None:
            lines.append(f"  Managed enforcement: recovery required ({self._managed_error})")
        elif self._managed_policy is None:
            lines.append("  Managed enforcement: unavailable (no protected installation)")
        else:
            managed = self._managed_policy.policy
            lines.append(
                f"  Managed enforcement: active identity={managed.identity} "
                f"revision={managed.revision} source={self._managed_policy_store.directory}"
            )
            if view in {"all", "managed"}:
                lines.append(
                    f"  Managed restrictions: require_sandbox={managed.require_sandbox}, "
                    f"forbid_host={managed.forbid_host}, "
                    f"deny_local_networks={managed.deny_local_networks}, "
                    f"forbid_yolo={managed.forbid_yolo}, rules={len(managed.rules)}"
                )
                lines.extend(f"    {self._describe_rule(rule)}" for rule in managed.rules)
        if self._namespace_catalog_path is not None:
            for scope in ("workspace", "user"):
                lines.append(
                    f"  {scope} namespace identity: {self.policy_namespace_identity(scope)}"
                )
                if error := self._namespace_errors.get(scope):
                    lines.append(f"  {scope} namespace recovery required: {error}")
        if view in {"all", "workspace"}:
            lines.extend(self._describe_configuration("Workspace policy", self._configuration))
        if view in {"all", "session"}:
            lines.extend(self._describe_session_overrides())
        if view in {"all", "effective"}:
            lines.extend(
                self._describe_configuration(
                    "Effective policy", self.effective_configuration, annotate_sources=True
                )
            )
        lines.append(
            "Precedence: protected and managed boundaries > deny rule > managed fresh human > "
            "exact allow rule > ask rule > allow rule > default"
        )
        return "\n".join(lines)

    def _describe_configuration(
        self,
        heading: str,
        configuration: PermissionConfiguration,
        *,
        annotate_sources: bool = False,
    ) -> list[str]:
        """Return display lines for one complete configuration."""
        lines = [f"{heading}:", "  Defaults:"]
        for action in Action:
            source = (
                (" [session]" if action in self._session_overrides.defaults else " [workspace]")
                if annotate_sources
                else ""
            )
            decision = configuration.defaults.get(action, Decision.DENY)
            lines.append(f"    {action.value}: {decision.value}{source}")
        lines.append("  Limits:")
        for name in _LIMIT_NAMES:
            value = getattr(configuration.limits, name)
            rendered = (", ".join(value) or "none") if isinstance(value, tuple) else str(value)
            source = (
                (
                    " [session]"
                    if getattr(self._session_overrides.limits, name) is not None
                    else " [workspace]"
                )
                if annotate_sources
                else ""
            )
            if name in {"readable_roots", "writable_roots", "network_origins"}:
                classification = getattr(configuration.limit_classifications, name).value
                provenance = configuration.limit_classifications.sources[name]
                lines.append(
                    f"    {name}: {rendered}{source} "
                    f"(classification={classification}, source={provenance})"
                )
            else:
                lines.append(f"    {name}: {rendered}{source}")
        rules = configuration.rules
        lines.append("  Rules:" if rules else "  Rules: none")
        if rules:
            session_ids = {rule.id for rule in self._session_overrides.rules}
            lines.extend(
                f"  {self._describe_rule(rule)}"
                f" [{'session' if rule.id in session_ids else 'workspace'}]"
                if annotate_sources
                else f"  {self._describe_rule(rule)}"
                for rule in rules
            )
        return lines

    def _describe_session_overrides(self) -> list[str]:
        """Return display lines for process-local policy differences."""
        overrides = self._session_overrides
        lines = ["Session overrides:"]
        if (
            not overrides.defaults
            and overrides.limits == PolicyLimitOverrides()
            and not overrides.rules
        ):
            return [*lines, "  none"]
        lines.append("  Defaults:")
        lines.extend(
            f"    {action.value}: {decision.value}"
            for action, decision in sorted(
                overrides.defaults.items(), key=lambda item: item[0].value
            )
        )
        if not overrides.defaults:
            lines.append("    none")
        lines.append("  Limits:")
        limit_count = 0
        for name in _LIMIT_NAMES:
            value = getattr(overrides.limits, name)
            if value is None:
                continue
            rendered = (", ".join(value) or "none") if isinstance(value, tuple) else str(value)
            lines.append(f"    {name}: {rendered}")
            limit_count += 1
        if not limit_count:
            lines.append("    none")
        lines.append("  Rules:" if overrides.rules else "  Rules: none")
        lines.extend(f"  {self._describe_rule(rule)}" for rule in overrides.rules)
        return lines

    @staticmethod
    def _describe_rule(rule: PermissionRule) -> str:
        description = f" — {rule.description}" if rule.description else ""
        source = (
            f" [preset={rule.source.preset_id}@{rule.source.revision} "
            f"hash={rule.source.content_hash}]"
            if rule.source is not None
            else ""
        )
        target = rule.target.model_dump_json() if rule.target is not None else None
        tool_match = " tool_match=exact" if rule.tool_exact else ""
        return (
            f"{rule.id} {rule.decision.value} tool={rule.tool}{tool_match} "
            f"action={rule.action.value if rule.action else '*'} "
            f"resource={target or rule.resource or '*'}{description}{source}"
        )

    def _read_configuration(
        self, *, retain_on_missing: bool = False
    ) -> PermissionConfiguration | None:
        """Read and validate the configured policy without changing active state."""
        if self._configuration_path is None:
            return None if retain_on_missing else PermissionConfiguration()
        if self._namespace_catalog_path is not None:
            return self._read_namespaced_configuration(
                self._configuration_path, "workspace", self._workspace_id
            )
        try:
            revision = self._read_optional_policy(self._configuration_path)
            if revision is None:
                return None if retain_on_missing else PermissionConfiguration()
            self._failed_workspace_revision = revision
            payload = load_yaml(revision.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(  # noqa: TRY004 - invalid document content, not a caller type.
                    "Policy document must be a mapping."
                )
            if not payload:
                raise ValueError("Policy document must not be empty.")
            values = payload
            if "limit_classifications" not in values:
                values = {
                    **values,
                    "limit_classifications": PolicyLimitClassifications(
                        readable_roots=LimitClassification.HARD,
                        writable_roots=LimitClassification.HARD,
                        network_origins=LimitClassification.HARD,
                        sources={
                            "readable_roots": "legacy",
                            "writable_roots": "legacy",
                            "network_origins": "legacy",
                        },
                    ).model_dump(mode="json"),
                }
            configuration = PermissionConfiguration.model_validate(values, extra="forbid")
            self._workspace_storage_revision = revision
            self._failed_workspace_revision = None
            return configuration
        except (OSError, UnicodeError, ValueError, YAMLError) as exc:
            raise PermissionConfigurationError(
                self._configuration_path,
                safe_error_detail(
                    exc,
                    PermissionConfiguration,
                    trusted_types=(YAMLError,),
                    trusted_messages=_TRUSTED_POLICY_ERRORS,
                ),
            ) from exc

    def _read_optional_policy(self, path: Path, source: Path | None = None) -> bytes | None:
        """Read a policy twice around reset checks before accepting a genuinely missing path."""
        try:
            return path.read_bytes()
        except FileNotFoundError:
            self._require_completed_reset(path)
        try:
            return path.read_bytes()
        except FileNotFoundError as missing:
            for candidate in (path, source):
                if candidate is None:
                    continue
                try:
                    candidate.lstat()
                except FileNotFoundError:
                    for parent in candidate.parents:
                        try:
                            details = parent.lstat()
                        except FileNotFoundError:
                            continue
                        if stat.S_ISLNK(details.st_mode):
                            parent.stat()
                else:
                    raise missing
            return None

    @staticmethod
    def _read_namespaced_file(path: Path) -> tuple[bytes | None, int | None]:
        """Read one policy inode without following its final link or accepting a swapped file."""
        try:
            if path.parent.resolve() != path.parent:
                raise PermissionNamespaceError(path, "Policy directory has an aliased ancestor.")
            try:
                details = path.lstat()
            except FileNotFoundError:
                return None, None
            if not hasattr(os, "O_NOFOLLOW"):
                raise PermissionNamespaceError(path, "No-follow policy loading is unavailable.")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_uid != os.getuid()
                    or (opened.st_dev, opened.st_ino) != (details.st_dev, details.st_ino)
                ):
                    raise PermissionNamespaceError(path, "Policy file provenance is invalid.")
                with os.fdopen(descriptor, "rb") as stream:
                    descriptor = -1
                    revision = stream.read()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            current = path.lstat()
        except OSError as error:
            if error.errno == errno.ELOOP:
                detail = "Policy file provenance is invalid; symbolic links are not allowed."
            elif error.errno in {errno.ENOENT, errno.ESTALE}:
                detail = "Policy file changed during loading."
            elif isinstance(error, PermissionError):
                detail = "Permission denied while reading the policy."
            elif error.errno is not None:
                detail = f"Policy read failed: {os.strerror(error.errno)} (errno {error.errno})."
            else:
                detail = "Policy read failed (I/O error; errno unavailable)."
            raise PermissionNamespaceError(path, detail) from error
        if (current.st_dev, current.st_ino) != (details.st_dev, details.st_ino):
            raise PermissionNamespaceError(path, "Policy file changed during loading.")
        return revision, opened.st_mode

    def _live_namespace_error(self) -> tuple[str, str] | None:
        """Check live reset markers and durable namespace identity before admission."""
        catalog = self._namespace_catalog_path
        if catalog is None:
            for scope, path in (
                ("workspace", self._configuration_path),
                ("user", self._user_configuration_path),
            ):
                if path is None:
                    continue
                try:
                    self._require_completed_reset(path)
                except PermissionConfigurationError as error:
                    return scope, error.message
            return None
        workspace_path = self._configuration_path
        user_path = self._user_configuration_path
        if workspace_path is None or user_path is None:
            return "workspace", "Durable policy paths are unavailable."
        try:
            catalog_before = self._validate_namespace_catalog(workspace_path, catalog)
            uri = f"file:{quote(str(catalog), safe='/')}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
                active = connection.execute(
                    "SELECT canonical_path FROM workspace_locations WHERE workspace_id=? "
                    "AND status='active'",
                    (self._workspace_id,),
                ).fetchall()
                if (
                    len(active) != 1
                    or self._workspace_root is None
                    or active[0][0] != str(self._workspace_root)
                ):
                    return "workspace", "Workspace identity changed; reload or restore policy."
                scopes = (
                    (
                        "workspace",
                        self._workspace_id,
                        self._configuration.namespace_generation,
                        workspace_path,
                        self._workspace_storage_revision,
                    ),
                    (
                        "user",
                        self._user_namespace_identity,
                        self._user_configuration.namespace_generation,
                        user_path,
                        self._user_storage_revision,
                    ),
                )
                for scope, identity, generation, path, expected in scopes:
                    try:
                        self._require_completed_reset(path)
                    except PermissionConfigurationError as error:
                        return scope, error.message
                    row = connection.execute(
                        "SELECT generation,status FROM policy_namespaces "
                        "WHERE scope=? AND identity=?",
                        (scope, identity),
                    ).fetchone()
                    if row != (generation, "initialized") or expected is None:
                        return scope, "Policy namespace generation or reservation changed."
                    try:
                        revision, mode = self._read_namespaced_file(path)
                    except PermissionNamespaceError as error:
                        return scope, error.message
                    if revision != expected or mode is None or mode & 0o022:
                        return scope, "Policy changed since loading; reload before admission."
            catalog_after = self._validate_namespace_catalog(workspace_path, catalog)
            if catalog_before != catalog_after:
                return "workspace", "Independent policy registry changed during admission."
        except (OSError, sqlite3.Error, PermissionNamespaceError) as error:
            return "workspace", f"Independent policy registry is unavailable: {error}"
        return None

    def _read_namespaced_configuration(
        self,
        path: Path,
        scope: str,
        identity: str | None,
    ) -> PermissionConfiguration | UserPermissionConfiguration:
        """Reconcile a durable policy with its independent namespace reservation."""
        if identity is None or self._namespace_catalog_path is None:
            raise PermissionNamespaceError(path, "Policy namespace identity is unavailable.")
        catalog = self._namespace_catalog_path
        self._validate_namespace_catalog(path, catalog)
        with self._storage_lock(path):
            self._require_completed_reset(path)
            revision, file_mode = self._read_namespaced_file(path)
            model = PermissionConfiguration if scope == "workspace" else UserPermissionConfiguration
            try:
                payload = load_yaml(revision.decode("utf-8")) if revision is not None else None
                if revision is not None and not isinstance(payload, dict):
                    raise ValueError("Policy document must be a mapping.")
                if revision is not None and not payload:
                    raise ValueError("Policy document must not be empty.")
                if (
                    revision is not None
                    and scope == "workspace"
                    and payload.get("namespace_generation") is None
                ):
                    limits = payload.setdefault("limits", {})
                    if isinstance(limits, dict):
                        limits.setdefault("deny_local_networks", True)
                if revision is not None and payload.get("namespace_generation") is not None:
                    fields = set(model.model_fields)
                    if payload.get("version") == 1:
                        if scope == "user" and set(payload) == fields - {
                            "trusted_host_command_rules"
                        }:
                            payload["trusted_host_command_rules"] = []
                        elif scope == "workspace" and set(payload) == fields:
                            limits = payload.get("limits")
                            if isinstance(limits, dict) and set(limits) == set(
                                PolicyLimits.model_fields
                            ) - {"deny_local_networks"}:
                                limits["deny_local_networks"] = True
                    if set(payload) != fields:
                        raise ValueError("Marked policy fields are incomplete.")
                    if scope == "workspace":
                        required = {
                            "limits": set(PolicyLimits.model_fields),
                            "limit_classifications": set(PolicyLimitClassifications.model_fields),
                            "defaults": {action.value for action in Action},
                        }
                        if any(
                            not isinstance(payload[name], dict) or payload[name].keys() != fields
                            for name, fields in required.items()
                        ):
                            raise ValueError("Marked policy limits or provenance are incomplete.")
                configuration = (
                    model.model_validate(payload, extra="forbid") if revision is not None else None
                )
            except (UnicodeError, ValueError, YAMLError) as error:
                raise PermissionNamespaceError(
                    path,
                    f"Policy content is invalid: {
                        safe_error_detail(
                            error,
                            model,
                            trusted_types=(YAMLError,),
                            trusted_messages=_TRUSTED_POLICY_ERRORS,
                        )
                    }",
                ) from error
            generation = configuration.namespace_generation if configuration is not None else None
            if revision is not None and file_mode & 0o022:
                private_ancestor = False
                if generation is None:
                    for ancestor in (path.parent, *path.parent.parents):
                        try:
                            ensure_private_directory(ancestor, create=False)
                        except (OSError, ValueError):
                            continue
                        private_ancestor = True
                        break
                if not private_ancestor:
                    raise PermissionNamespaceError(path, "Policy file provenance is invalid.")
            with closing(sqlite3.connect(catalog, timeout=5)) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS policy_namespaces ("
                    "scope TEXT NOT NULL, identity TEXT NOT NULL, generation TEXT NOT NULL, "
                    "status TEXT NOT NULL, legacy_hash TEXT, PRIMARY KEY(scope, identity))"
                )
                if scope == "user" and identity != str(catalog.parent):
                    raise PermissionNamespaceError(
                        path, "User policy identity does not match the data registry."
                    )
                try:
                    active = connection.execute(
                        "SELECT canonical_path FROM workspace_locations WHERE workspace_id=? "
                        "AND status='active'",
                        (self._workspace_id,),
                    ).fetchall()
                except sqlite3.Error as error:
                    raise PermissionNamespaceError(
                        path, "Independent workspace registry is invalid."
                    ) from error
                if (
                    len(active) != 1
                    or self._workspace_root is None
                    or active[0][0] != str(self._workspace_root)
                ):
                    raise PermissionNamespaceError(
                        path, "Workspace policy identity does not match the registry."
                    )
                row = connection.execute(
                    "SELECT generation,status,legacy_hash FROM policy_namespaces "
                    "WHERE scope=? AND identity=?",
                    (scope, identity),
                ).fetchone()
                if row is None:
                    if revision is None:
                        if (
                            not self._bootstrap_pristine_policy
                            or self._policy_backup_exists(path)
                            or os.path.lexists(path.with_name(path.name + _RESET_PENDING_SUFFIX))
                        ):
                            raise PermissionNamespaceError(
                                path,
                                "Policy namespace is missing or ambiguous. Restore existing "
                                "policy data or use trusted /permissions initialize with the "
                                "expected scope and identity.",
                            )
                        reserved_generation = uuid4().hex
                        connection.execute(
                            "INSERT INTO policy_namespaces(scope,identity,generation,status,"
                            "legacy_hash) VALUES (?,?,?,'reserved_new',NULL)",
                            (scope, identity, reserved_generation),
                        )
                        row = (reserved_generation, "reserved_new", None)
                    elif generation is not None:
                        raise PermissionNamespaceError(
                            path, "Policy marker has no independent registry anchor."
                        )
                    else:
                        reserved_generation = uuid4().hex
                        connection.execute(
                            "INSERT INTO policy_namespaces(scope,identity,generation,status,"
                            "legacy_hash) VALUES (?,?,?,?,?)",
                            (
                                scope,
                                identity,
                                reserved_generation,
                                "reserved_legacy",
                                sha256_digest(revision),
                            ),
                        )
                        row = (reserved_generation, "reserved_legacy", sha256_digest(revision))
            recorded_generation, status, legacy_hash = row
            if status == "initialized":
                if configuration is None or generation != recorded_generation:
                    raise PermissionNamespaceError(
                        path, "Initialized policy is missing or has a mismatched generation."
                    )
                if configuration.version == 1:
                    if self._read_namespaced_file(path)[0] != revision:
                        raise PermissionNamespaceError(
                            path, "Policy changed during schema migration."
                        )
                    self._archive_policy_revision(
                        path, revision, f"{recorded_generation}.schema-v1", "Schema"
                    )
                    if self._read_namespaced_file(path)[0] != revision:
                        raise PermissionNamespaceError(
                            path, "Policy changed during schema migration."
                        )
                    configuration.version = 2
                    revision = self._persist_configuration(path, configuration)
                    telemetry_audit("permission.policy_schema_migrated", scope=scope, version=2)
                    self._append_audit(
                        "permission.policy_schema_migrated", {"scope": scope, "version": 2}
                    )
            elif status == "reserved_new":
                if configuration is None:
                    if self._policy_backup_exists(path):
                        raise PermissionNamespaceError(
                            path, "Reserved new policy has backup evidence requiring recovery."
                        )
                    configuration = model(namespace_generation=recorded_generation)
                    revision = self._persist_configuration(path, configuration)
                elif generation != recorded_generation:
                    raise PermissionNamespaceError(
                        path, "Reserved new policy generation does not match."
                    )
                with closing(sqlite3.connect(catalog, timeout=5)) as connection, connection:
                    connection.execute(
                        "UPDATE policy_namespaces SET status='initialized' "
                        "WHERE scope=? AND identity=? AND generation=? AND status='reserved_new'",
                        (scope, identity, recorded_generation),
                    )
            elif status == "reserved_legacy":
                if configuration is None:
                    raise PermissionNamespaceError(path, "Reserved legacy policy is missing.")
                if generation is None and sha256_digest(revision) == legacy_hash:
                    self._archive_policy_revision(
                        path, revision, f"{recorded_generation}.legacy", "Legacy"
                    )
                    if scope == "workspace":
                        configuration.limit_classifications = PolicyLimitClassifications(
                            readable_roots=LimitClassification.HARD,
                            writable_roots=LimitClassification.HARD,
                            network_origins=LimitClassification.HARD,
                            sources={
                                "readable_roots": "legacy",
                                "writable_roots": "legacy",
                                "network_origins": "legacy",
                            },
                        )
                    configuration.namespace_generation = recorded_generation
                    configuration.version = 2
                    if self._read_storage_revision(path) != revision:
                        raise PermissionNamespaceError(
                            path, "Legacy policy changed during migration."
                        )
                    revision = self._persist_configuration(path, configuration)
                    telemetry_audit("permission.policy_legacy_migrated", scope=scope, version=2)
                    self._append_audit(
                        "permission.policy_legacy_migrated", {"scope": scope, "version": 2}
                    )
                elif generation != recorded_generation:
                    raise PermissionNamespaceError(
                        path, "Reserved policy generation or legacy content does not match."
                    )
                with closing(sqlite3.connect(catalog, timeout=5)) as connection, connection:
                    connection.execute(
                        "UPDATE policy_namespaces SET status='initialized' "
                        "WHERE scope=? AND identity=? AND generation=? "
                        "AND status='reserved_legacy'",
                        (scope, identity, recorded_generation),
                    )
            else:
                raise PermissionNamespaceError(path, "Policy namespace reservation is invalid.")
            if scope == "workspace":
                self._workspace_storage_revision = revision
                self._failed_workspace_revision = None
            else:
                self._user_storage_revision = revision
                self._failed_user_revision = None
            return configuration

    def _validate_namespace_catalog(self, path: Path, catalog: Path) -> tuple[int, int]:
        """Validate the independent policy registry and return its device and inode."""
        try:
            catalog_details = catalog.lstat()
            parent_details = catalog.parent.lstat()
        except OSError as error:
            raise PermissionNamespaceError(
                path, "Independent policy registry is missing."
            ) from error
        if (
            not stat.S_ISREG(catalog_details.st_mode)
            or catalog_details.st_nlink != 1
            or not stat.S_ISDIR(parent_details.st_mode)
            or any(
                details.st_uid != os.getuid() or details.st_mode & 0o022
                for details in (catalog_details, parent_details)
            )
        ):
            raise PermissionNamespaceError(path, "Independent policy registry is untrusted.")
        return catalog_details.st_dev, catalog_details.st_ino

    def _require_completed_reset(self, path: Path | None = None) -> None:
        """Refuse policy loading while a reset marker remains."""
        path = self._configuration_path if path is None else path
        if path is None:
            return
        marker = path.with_name(path.name + _RESET_PENDING_SUFFIX)
        try:
            marker.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise PermissionConfigurationError(path, str(exc)) from exc
        raise PermissionNamespaceError(
            path, "A policy reset is pending; inspect and recover the policy before retrying."
        )

    def _read_user_configuration(self) -> UserPermissionConfiguration | None:
        """Read and validate the user-wide approval policy without changing active state."""
        if self._namespace_catalog_path is not None and self._user_configuration_path is not None:
            return self._read_namespaced_configuration(
                self._user_configuration_path, "user", self._user_namespace_identity
            )
        if self._user_configuration_path is None:
            self._user_storage_revision = None
            self._failed_user_revision = None
            return None
        try:
            revision = self._read_optional_policy(
                self._user_configuration_path,
                self._user_configuration_source_path,
            )
            if revision is None:
                self._user_storage_revision = None
                self._failed_user_revision = None
                return None
            self._failed_user_revision = revision
            payload = load_yaml(revision.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(  # noqa: TRY004 - invalid document content, not a caller type.
                    "Policy document must be a mapping."
                )
            if not payload:
                raise ValueError("Policy document must not be empty.")
            configuration = UserPermissionConfiguration.model_validate(payload, extra="forbid")
            self._user_storage_revision = revision
            self._failed_user_revision = None
            return configuration
        except (OSError, UnicodeError, ValueError, YAMLError) as exc:
            raise PermissionConfigurationError(
                self._user_configuration_path,
                safe_error_detail(
                    exc,
                    UserPermissionConfiguration,
                    trusted_types=(YAMLError,),
                    trusted_messages=_TRUSTED_POLICY_ERRORS,
                ),
            ) from exc

    def _load_user_configuration(self, *, interaction: Interaction | None = None) -> bool:
        """Load user rules or request an explicit fail-closed recovery decision."""
        active_interaction = interaction if interaction is not None else self._interaction
        while True:
            failure, revision = self._try_load_user_configuration()
            if failure is None:
                return False
            if active_interaction is None:
                raise failure
            problem = Problem.from_exception(
                failure,
                code="permission.user_configuration_invalid",
                title="Invalid user permission policy",
                operation="load_user_permission_policy",
                metadata={"path": failure.path},
            )
            log_problem(_LOGGER, problem, failure)
            active_interaction.report(problem)
            choice = prompt_recovery(
                active_interaction,
                "User permission policy recovery:",
                continue_label="Continue with permission operations blocked",
                reset_label=(
                    None
                    if isinstance(failure.__cause__, OSError)
                    else "Archive the file and remove all saved user rules, including denials"
                ),
            )
            if choice == "retry":
                continue
            reset_failure = None
            with self.policy_update():
                if revision != self._policy_revision:
                    continue
                if choice == "reset":
                    try:
                        backup = self.reset_configuration("user")
                    except OSError as error:
                        reset_failure = error
                    else:
                        active_interaction.info(
                            f"Reset user approvals; archived the policy at {backup}."
                        )
                        return False
            if reset_failure is not None:
                choice = self._recover_reset_failure(
                    active_interaction,
                    reset_failure,
                    self._user_configuration_path,
                )
                if choice == "retry":
                    continue
            with self.policy_update():
                if revision != self._policy_revision:
                    continue
                self._namespace_errors["user"] = failure.message
                self._user_configuration = UserPermissionConfiguration(
                    rules=[
                        rule
                        for rule in self._user_configuration.rules
                        if rule.decision is not Decision.ALLOW
                    ]
                )
                self._policy_revision += 1
            active_interaction.warning(
                "Permission operations remain blocked until the policy is repaired."
            )
            return True

    def _try_load_user_configuration(self) -> tuple[PermissionConfigurationError | None, int]:
        """Load user rules under the policy lock or return an interactive failure."""
        with self.policy_update():
            try:
                if self._user_configuration_path is not None:
                    self._require_completed_reset(self._user_configuration_path)
                configuration = self._read_user_configuration()
            except PermissionNamespaceError as exc:
                self._namespace_errors["user"] = exc.message
                self._policy_revision += 1
                if self._namespace_catalog_path is None:
                    raise
                configuration = None
            except PermissionConfigurationError as exc:
                self._namespace_errors["user"] = exc.message
                self._policy_revision += 1
                self._require_completed_reset(self._user_configuration_path)
                if isinstance(exc.__cause__, OSError) and (
                    self.yolo or self._load_policy is not PermissionLoadPolicy.INTERACTIVE
                ):
                    raise
                if self.yolo or self._load_policy is PermissionLoadPolicy.ERROR:
                    raise
                if self._load_policy is PermissionLoadPolicy.INTERACTIVE:
                    return exc, self._policy_revision
                configuration = None
                self._report_user_configuration_error(exc)
            if configuration is None:
                configuration = UserPermissionConfiguration(
                    rules=[
                        rule
                        for rule in self._user_configuration.rules
                        if rule.decision is not Decision.ALLOW
                    ]
                )
            else:
                self._namespace_errors.pop("user", None)
            with self.policy_update():
                self._user_configuration = configuration
                self._policy_revision += 1
            return None, self._policy_revision

    def _load_configuration(
        self,
        *,
        retain_on_failure: bool = False,
        interaction: Interaction | None = None,
    ) -> PermissionLoadResult:
        """Load configuration and completely handle the selected failure policy."""
        while True:
            with self.policy_update():
                try:
                    self._require_completed_reset()
                    configuration = self._read_configuration(retain_on_missing=retain_on_failure)
                except PermissionNamespaceError as exc:
                    self._namespace_errors["workspace"] = exc.message
                    self._policy_revision += 1
                    if self._namespace_catalog_path is None:
                        raise
                    self._workspace_storage_revision = None
                    return PermissionLoadResult.RETAINED
                except PermissionConfigurationError as exc:
                    self._namespace_errors["workspace"] = exc.message
                    self._policy_revision += 1
                    self._require_completed_reset()
                    if isinstance(exc.__cause__, OSError) and (
                        self.yolo or self._load_policy is not PermissionLoadPolicy.INTERACTIVE
                    ):
                        raise
                    if self.yolo or self._load_policy is PermissionLoadPolicy.ERROR:
                        raise
                    if self._load_policy is PermissionLoadPolicy.AUTO:
                        self._report_configuration_error(exc)
                        if retain_on_failure:
                            return PermissionLoadResult.BLOCKED
                        self._workspace_storage_revision = self._failed_workspace_revision
                        with self.policy_update():
                            self._configuration = PermissionConfiguration()
                            self._policy_revision += 1
                        return PermissionLoadResult.BLOCKED
                    active_interaction = (
                        interaction if interaction is not None else self._interaction
                    )
                    if active_interaction is None:
                        raise
                    failure = exc
                    revision = self._policy_revision
                else:
                    if configuration is None:
                        return PermissionLoadResult.RETAINED
                    classifications = configuration.limit_classifications
                    if (
                        classifications.sources["readable_roots"] == "legacy"
                        and classifications.readable_roots is LimitClassification.HARD
                        and set(configuration.limits.readable_roots)
                        == set(PolicyLimits().readable_roots)
                    ):
                        configuration.limit_classifications = classifications.model_copy(
                            update={"readable_roots": LimitClassification.APPROVAL}
                        )
                    self._namespace_errors.pop("workspace", None)
                    with self.policy_update():
                        self._configuration = configuration
                        self._policy_revision += 1
                    return PermissionLoadResult.LOADED
            choice = self._recover_configuration_interactively(
                failure,
                interaction=active_interaction,
                retain_on_failure=retain_on_failure,
                revision=revision,
            )
            if choice is not PermissionLoadResult.LOADED:
                return choice

    def _recover_configuration_interactively(
        self,
        error: PermissionConfigurationError,
        *,
        interaction: Interaction,
        retain_on_failure: bool,
        revision: int,
    ) -> PermissionLoadResult:
        """Report one failure and return the user's selected recovery outcome."""
        problem = Problem.from_exception(
            error,
            code="permission.configuration_invalid",
            title="Invalid permission policy",
            operation="load_permission_policy",
            metadata={"path": error.path},
        )
        log_problem(_LOGGER, problem, error)
        interaction.report(problem)
        # Construction and reload are synchronous; async callers must use a native prompt path.
        choice = prompt_recovery(
            interaction,
            "Permission policy recovery:",
            continue_label="Continue with permission operations blocked",
            reset_label=(
                None
                if isinstance(error.__cause__, OSError)
                else "Archive the file and replace saved rules and limits with defaults"
            ),
        )
        if choice == "retry":
            return PermissionLoadResult.LOADED
        with self.policy_update():
            if revision != self._policy_revision:
                return PermissionLoadResult.LOADED
            if choice != "reset":
                return self._finish_configuration_recovery(
                    interaction=interaction,
                    retain_on_failure=retain_on_failure,
                )
            try:
                backup_path = self.reset_configuration("workspace")
            except OSError as failure:
                reset_error = failure
            else:
                interaction.info(
                    "Reset permission policy to supervised defaults; "
                    f"archived invalid file at {backup_path}."
                )
                return PermissionLoadResult.DEFAULTED
        choice = self._recover_reset_failure(interaction, reset_error, self._configuration_path)
        if choice == "retry":
            return PermissionLoadResult.LOADED
        interaction.warning("Permission operations remain blocked until the policy is repaired.")
        return PermissionLoadResult.BLOCKED

    @staticmethod
    def _recover_reset_failure(interaction: Interaction, error: OSError, path: Path | None) -> str:
        """Report a reset failure and request a decision after releasing policy locks."""
        trusted_messages = {
            "Permission policy changed; reload before retrying.",
            "Unsafe policy lock file.",
            "Unsafe policy staging file.",
        }
        if str(error) in trusted_messages:
            detail = str(error)
        elif isinstance(error, PermissionError):
            detail = "Permission denied while resetting the policy."
        elif error.errno is not None:
            detail = f"Policy reset failed: {os.strerror(error.errno)} (errno {error.errno})."
        else:
            detail = "Policy reset failed (I/O error; errno unavailable)."
        problem = Problem(
            code="permission.reset_failed",
            title="Permission policy reset failed",
            detail=(f"{detail} Check the policy, pending reset marker, and backups near {path}."),
            operation="permission.reset",
        )
        log_problem(_LOGGER, problem, error)
        interaction.report(problem)
        return prompt_recovery(
            interaction,
            "Permission policy recovery:",
            continue_label="Continue with permission operations blocked",
        )

    def _finish_configuration_recovery(
        self,
        interaction: Interaction,
        retain_on_failure: bool,
    ) -> PermissionLoadResult:
        """Apply a still-current policy recovery decision."""
        if retain_on_failure:
            interaction.warning(
                "Permission operations remain blocked until the policy is repaired."
            )
            return PermissionLoadResult.BLOCKED
        with self.policy_update():
            self._configuration = PermissionConfiguration()
            self._policy_revision += 1
        interaction.warning("Permission operations remain blocked until the policy is repaired.")
        return PermissionLoadResult.BLOCKED

    def _report_configuration_error(self, error: PermissionConfigurationError) -> None:
        """Report an automatically recovered configuration failure."""
        problem = Problem.from_exception(
            error,
            code="permission.configuration_invalid",
            title="Invalid permission policy",
            operation="load_permission_policy",
            metadata={"path": error.path},
        )
        log_problem(_LOGGER, problem, error)
        if self._interaction is not None:
            self._interaction.report(problem)
            self._interaction.warning(
                "Permission operations are blocked until the policy is repaired."
            )
        else:
            _LOGGER.warning("Invalid permission policy; permission operations are blocked")

    def _report_user_configuration_error(self, error: PermissionConfigurationError) -> None:
        """Report an invalid user approval policy while retaining fail-closed behavior."""
        problem = Problem.from_exception(
            error,
            code="permission.user_configuration_invalid",
            title="Invalid user permission policy",
            operation="load_user_permission_policy",
            metadata={"path": error.path},
        )
        log_problem(_LOGGER, problem, error)
        if self._interaction is not None:
            self._interaction.report(problem)
            self._interaction.warning("Ignoring user-wide approvals until the policy is fixed.")
        else:
            _LOGGER.warning("Invalid user-wide approval policy; permission operations are blocked")

    @staticmethod
    async def recover_startup_async(
        error: PermissionNamespaceError | PermissionPresetError,
        interaction: Interaction,
    ) -> bool:
        """Ask how to recover from a permission artifact failure during startup.

        Args:
            error (PermissionNamespaceError | PermissionPresetError): Failed permission artifact.
            interaction (Interaction): User interface for reporting and recovery choices.

        Returns:
            bool: Whether to exclude the reported invalid presets; False requests a retry.

        Raises:
            ShutdownRequested: If the user declines to continue.
        """
        if isinstance(error, PermissionPresetError):
            for failure in error.failures:
                interaction.report(
                    Problem(
                        code="permission.preset_invalid",
                        title="Invalid permission preset",
                        detail=f"{failure.path}: {failure.message}",
                        operation="startup",
                    )
                )
            choice = await prompt_recovery_async(
                interaction,
                "Permission preset recovery:",
                operation="permission recovery",
                continue_label="Continue without the listed invalid presets",
            )
            if choice == "continue":
                return True
        else:
            problem = Problem.from_exception(
                error,
                code="permission.policy_recovery_required",
                title="Permission policy recovery required",
                operation="startup",
            )
            log_problem(_LOGGER, problem, error)
            interaction.report(problem)
            interaction.warning(
                "Defaults and automatic repair are unavailable because the policy "
                "may contain restrictions. Repair or restore it before reloading."
            )
            choice = await prompt_recovery_async(
                interaction,
                "Permission policy recovery:",
                operation="permission recovery",
            )
        return False

    def _load_presets(
        self,
        accepted_failures: tuple[PermissionLoadFailure, ...],
    ) -> tuple[PermissionPreset, ...]:
        """Load presets and completely handle invalid catalog artifacts."""
        presets, failures = PermissionPreset.load_builtin_presets()
        if not failures:
            return presets
        error = PermissionPresetError(failures)
        if failures != accepted_failures:
            raise error
        for failure in failures:
            message = f"Excluded invalid permission preset at {failure.path}: {failure.message}"
            if self._interaction is not None:
                self._interaction.warning(message)
            else:
                _LOGGER.warning(
                    "Excluded invalid permission preset",
                    extra={"error.type": "permission.preset_invalid"},
                )
        return presets

    def _evaluate_operation(self, operation: Operation) -> PolicyDecision:
        boundary = self._boundary_decision(operation)
        if boundary is not None:
            return boundary
        return self._evaluate_operation_policy(operation)

    def _evaluate_operation_policy(self, operation: Operation) -> PolicyDecision:
        """Merge managed restrictions with user rules after fixed boundaries allow an operation."""
        managed = self._managed_rule_decision(operation)
        if managed is not None and managed.decision is Decision.DENY:
            return managed
        user = self._evaluate_user_operation_policy(operation)
        if user.decision is Decision.DENY:
            return user
        return managed if managed is not None else user

    def _managed_host_prohibited(self) -> bool:
        """Report a loaded administrator requirement that forbids host execution."""
        policy = self._managed_policy
        return policy is not None and (policy.policy.require_sandbox or policy.policy.forbid_host)

    def _managed_rule_decision(self, operation: Operation) -> PolicyDecision | None:
        """Return the strongest matching administrator rule, if any."""
        snapshot = self._managed_policy
        if snapshot is None:
            return None
        matching = tuple(rule for rule in snapshot.policy.rules if self._matches(rule, operation))
        for effect in (Decision.DENY, Decision.ASK):
            selected = tuple(rule for rule in matching if rule.decision is effect)
            if selected:
                denied = effect is Decision.DENY or self.yolo
                return PolicyDecision(
                    decision=Decision.DENY if denied else Decision.ASK,
                    reason=(
                        "Administrator policy denies this operation."
                        if effect is Decision.DENY
                        else "Administrator policy requires a fresh human decision; YOLO cannot "
                        "satisfy it."
                        if self.yolo
                        else "Administrator policy requires a fresh human decision."
                    ),
                    sources=tuple(
                        f"managed:{'deny' if effect is Decision.DENY else 'human'}:{rule.id}"
                        for rule in selected
                    ),
                )
        return None

    def _evaluate_user_operation_policy(self, operation: Operation) -> PolicyDecision:
        """Evaluate user rules and defaults after protected constraints."""
        matching = [
            (ApprovalChoice.USER.value, rule)
            for rule in self._user_configuration.rules
            if self._matches(rule, operation)
        ]
        matching.extend(
            (ApprovalChoice.WORKSPACE.value, rule)
            for rule in self._configuration.rules
            if self._matches(rule, operation)
        )
        matching.extend(
            (ApprovalChoice.SESSION.value, rule)
            for rule in self._session_overrides.rules
            if self._matches(rule, operation)
        )
        exact_allow = tuple(
            (scope, rule)
            for scope, rule in matching
            if rule.decision is Decision.ALLOW and rule.target is not None
        )
        ordered = (
            (
                Decision.DENY,
                tuple((scope, rule) for scope, rule in matching if rule.decision is Decision.DENY),
            ),
            (Decision.ALLOW, exact_allow),
            (
                Decision.ASK,
                tuple((scope, rule) for scope, rule in matching if rule.decision is Decision.ASK),
            ),
            (
                Decision.ALLOW,
                tuple(
                    (scope, rule)
                    for scope, rule in matching
                    if rule.decision is Decision.ALLOW and rule.target is None
                ),
            ),
        )
        for decision, determining in ordered:
            if determining:
                return PolicyDecision(
                    decision=decision,
                    reason=f"Matched explicit {decision.value} policy rule(s).",
                    sources=tuple(f"rule:{scope}:{rule.id}" for scope, rule in determining),
                )
        session_decision = self._session_overrides.defaults.get(operation.action)
        decision = session_decision or self._configuration.defaults.get(
            operation.action, Decision.DENY
        )
        scope = PolicyScope.SESSION if session_decision is not None else PolicyScope.WORKSPACE
        return PolicyDecision(
            decision=decision,
            reason=f"The {operation.action.value} default is {decision.value}.",
            sources=(f"default:{scope.value}:{operation.action.value}",),
        )

    def _boundary_decision(self, operation: Operation) -> PolicyDecision | None:
        target = operation.target
        if (
            isinstance(target, ProcessTarget)
            and target.boundary is ProcessBoundary.HOST
            and self._managed_host_prohibited()
        ):
            name = (
                "sandbox_required"
                if self._managed_policy.policy.require_sandbox
                else "host_forbidden"
            )
            return PolicyDecision(
                decision=Decision.DENY,
                reason="Administrator policy prohibits host process execution.",
                sources=(f"managed:{name}",),
            )
        limits = self.effective_configuration.limits
        if isinstance(target, FileTarget):
            if self._is_protected_path(
                target.path,
                mutation=operation.action in _WRITE_ACTIONS,
                recursive=target.recursive,
            ):
                return self._boundary_denial("protected_path")
            roots = (
                limits.readable_roots
                if operation.action in _READ_ACTIONS
                else limits.writable_roots
            )
            if not self._in_roots(target.path, roots):
                limit_name = (
                    "readable_roots" if operation.action in _READ_ACTIONS else "writable_roots"
                )
                return self._limit_denial("filesystem_root", limit_name)
        elif isinstance(target, NetworkTarget):
            if not any(fnmatchcase(target.origin, pattern) for pattern in limits.network_origins):
                return self._limit_denial("network_origin", "network_origins")
            if limits.deny_private_networks and self._is_private_network(target):
                return self._limit_denial("private_network", "deny_private_networks")
        elif (
            isinstance(target, ProcessTarget)
            and target.boundary is ProcessBoundary.HOST
            and not limits.allow_host_processes
            and not self.yolo
        ):
            return self._limit_denial("host_process", "allow_host_processes")
        return None

    def _limit_denial(self, name: str, field: str) -> PolicyDecision:
        """Return a denial attributed to the effective limit layer."""
        scope = (
            PolicyScope.SESSION
            if getattr(self._session_overrides.limits, field) is not None
            else PolicyScope.WORKSPACE
        )
        reason = f"The {scope.value} {name} boundary denied this operation."
        if field == "allow_host_processes":
            reason += (
                " A user may enable session host-process eligibility with "
                "/permissions limit set session host-process allow. This does not approve "
                "a command or override other policy denies. A host command has the host "
                "user's authority and needs a separate fresh decision."
            )
        return PolicyDecision(
            decision=Decision.DENY,
            reason=reason,
            sources=(f"limit:{scope.value}:{field}",),
        )

    @staticmethod
    def _boundary_denial(name: str) -> PolicyDecision:
        return PolicyDecision(
            decision=Decision.DENY,
            reason=f"The non-overridable {name} boundary denied this operation.",
            sources=(f"boundary:{name}",),
        )

    @staticmethod
    def _matches(rule: PermissionRule, operation: Operation) -> bool:
        exact_target = (
            PermissionManager._approval_target(operation.target) == rule.target
            if rule.target is not None
            else True
        )
        return (
            (
                operation.tool_id == rule.tool
                if rule.tool_exact
                else fnmatchcase(operation.tool_id, rule.tool)
            )
            and (rule.action is None or rule.action is operation.action)
            and exact_target
            and (
                rule.resource is None
                or operation.resource is not None
                and fnmatchcase(operation.resource, rule.resource)
            )
        )

    def _in_roots(self, resource: str, configured_roots: tuple[str, ...]) -> bool:
        path = Path(resource).resolve()
        for configured in configured_roots:
            root = self._root_path(configured)
            if root is None:
                continue
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _root_path(self, configured: str) -> Path | None:
        """Resolve one configured filesystem root token or absolute path."""
        if configured == "workspace":
            return self._workspace_root.resolve() if self._workspace_root else None
        if configured == "app-temp":
            return self.temporary_directory
        if configured == "system-temp":
            return Path(tempfile.gettempdir()).resolve()
        path = Path(configured)
        if path.is_absolute():
            return path.resolve()
        return (self._workspace_root / path).resolve() if self._workspace_root is not None else None

    def check_boundaries(self, operations: Operations) -> PolicyDecision | None:
        """Return the first non-overridable boundary denial for planned effects.

        This check performs no policy matching or interactive approval. It lets phased planners
        reject an unsafe eventual effect before prerequisite filesystem inspection.

        Args:
            operations (Operations): Future typed effects to check.

        Returns:
            PolicyDecision | None: First boundary denial, or ``None`` when all effects are inside
                the fixed safety boundaries.
        """
        for operation in operations:
            denial = self._boundary_decision(operation)
            if denial is not None:
                return denial
        return None

    async def check_boundaries_async(self, operations: Operations) -> PolicyDecision | None:
        """Check planned effects against filesystem boundaries without blocking the owner loop.

        Retry a bounded number of times if policy changes during a boundary check; persistent
        churn returns a fail-closed stale decision.

        Args:
            operations (Operations): Future typed effects to check.

        Returns:
            PolicyDecision | None: First boundary denial, or ``None`` when every effect is safe.

        Raises:
            asyncio.CancelledError: If a boundary check is cancelled after its filesystem work
                settles.
        """
        if not operations:
            return None
        for _ in range(_ASYNC_POLICY_ATTEMPTS):
            await self._acquire_policy_lock_async()
            try:
                revision = self._policy_revision
                denial = None
                for operation in operations:
                    denial = await self._boundary_decision_async(operation)
                    if denial is not None:
                        break
                if revision == self._policy_revision:
                    return denial
            finally:
                self._policy_lock.release()
        return self._stale_policy_decision()

    def _is_protected_path(self, resource: str, *, mutation: bool, recursive: bool = False) -> bool:
        """Protect control data and the working root without following a mutation's leaf link."""
        requested = Path(resource)
        path = requested.parent.resolve() / requested.name if mutation else requested.resolve()
        managed = self._managed_policy_store.directory
        expectation = self._managed_policy_store.expectation_path
        if managed is not None and (
            path.is_relative_to(managed) or mutation and recursive and managed.is_relative_to(path)
        ):
            return True
        if expectation is not None and (
            path == expectation or mutation and recursive and expectation.is_relative_to(path)
        ):
            return True
        temporary = self.temporary_directory
        if mutation and (path == temporary or recursive and temporary.is_relative_to(path)):
            return True
        if self._workspace_root is None:
            return False
        protected = (
            self._workspace_root / constants.APP_DIRECTORY,
            self._workspace_root / constants.GIT_DIRECTORY,
        )
        if mutation:
            protected += (
                self._workspace_root / constants.GIT_IGNORE_FILENAME,
                self._workspace_root / constants.AGENT_IGNORE_FILENAME,
            )
        for root in protected:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                if not (mutation and recursive):
                    continue
            try:
                root.relative_to(path)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _is_private_network(target: NetworkTarget) -> bool:
        hostname = urlsplit(target.url).hostname
        if hostname is None or hostname.casefold() == "localhost":
            return True
        addresses = target.addresses
        try:
            addresses = addresses or (str(ipaddress.ip_address(hostname)),)
        except ValueError:
            return False
        return not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        )

    def _prompt(self, operations: Operations) -> str:
        lines = ["Agent requests approval for the following operations:"]
        for operation in operations:
            resource = self._display_resource(operation)
            target = f" on '{resource}'" if resource else ""
            reason = f" — {operation.reason}" if operation.reason else ""
            lines.append(
                f"{operation.action.icon} {operation.tool_id}: "
                f"{operation.action.value}{target}{reason}"
            )
        return "\n".join(lines)

    def _display_resource(self, operation: Operation) -> str | None:
        """Return a workspace-friendly display name for the operation target.

        File paths are shown relative to the workspace root. Process targets include
        their working directory and quote argument boundaries after normalizing
        recognized absolute paths through ``VirtualPath``. Paths outside the
        configured virtual roots remain visible rather than being redacted.

        Args:
            operation (Operation): The operation whose target should be displayed.

        Returns:
            str | None: Cleaned display string, or the raw resource when no
                workspace is configured or the target is not a file or process.
        """
        if operation.resource is None or self._workspace_root is None:
            return operation.resource
        if isinstance(operation.target, FileTarget):
            try:
                relative = Path(operation.target.path).relative_to(self._workspace_root)
                return (
                    f"workspace root: {self._workspace_root}"
                    if relative == Path(".")
                    else str(relative)
                )
            except ValueError:
                return operation.resource
        if isinstance(operation.target, ProcessTarget):
            virtual_paths = VirtualPath(
                workspace=self._workspace_root,
                temporary_directory=self.temporary_directory,
            )

            def display_path(path: str) -> str:
                """Render a recognized virtual path or retain an external path verbatim."""
                rendered = virtual_paths.display(path)
                return path if rendered == VirtualPath.EXTERNAL else rendered

            cleaned_argv = tuple(
                display_path(arg) if Path(arg).is_absolute() else arg
                for arg in operation.target.argv
            )
            cwd = display_path(operation.target.cwd)
            return f"{shlex.join(cleaned_argv)} (cwd: {cwd})"
        return operation.resource

    def _audit(self, result: AuthorizationResult) -> None:
        """Append one independently durable, timestamped permission decision."""
        payload = self._recordable_result(result).model_dump(mode="json")
        if not payload["operations"] and result.operations:
            payload["operation_count"] = len(result.operations)
        self._append_audit("permission.decided", payload)

    async def _audit_async(
        self,
        result: AuthorizationResult,
        *,
        _owned_failures: list[tuple[asyncio.CancelledError, BaseException]],
    ) -> None:
        """Append one permission decision through bounded asynchronous worker admission."""
        if self._audit_store is None or self._workspace_id is None:
            return
        pending = asyncio.ensure_future(
            self._async_blocking_runner(self._audit, result)
            if self._async_blocking_runner is not None
            else run_in_worker_async(self._audit, result)
        )
        cancellation = None
        try:
            await asyncio.wait({pending})
        except asyncio.CancelledError as error:
            cancellation = error
        while not pending.done():
            try:
                await asyncio.wait({pending})
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        try:
            pending.result()
        except Exception as error:  # noqa: BLE001  # audits must not change policy outcomes
            if cancellation is not None:
                _owned_failures.append((cancellation, error))
                cancellation.add_note(f"Permission audit also stopped with {type(error).__name__}.")
                raise cancellation
            telemetry_error(
                "permission.audit_write_failed",
                error_type="permission.audit_write_failed",
                component="permission_manager",
            )
            return
        if cancellation is not None:
            raise cancellation

    def _append_audit(self, event_name: str, payload: dict[str, object]) -> None:
        """Append one bounded audit payload without affecting policy behavior."""
        if self._audit_store is None or self._workspace_id is None:
            return
        try:
            self._audit_store.append(self._workspace_id, event_name, payload)
        except (OSError, sqlite3.Error) as error:
            # The durable session recorder remains authoritative for application behavior;
            # diagnostic JSONL availability must not change an authorization outcome.
            telemetry_error(
                "permission.audit_write_failed",
                error_type="permission.audit_write_failed",
                exception=error,
                component="permission_manager",
            )

    def _audit_policy_change(
        self,
        event_name: str,
        scope: PolicyScope,
        **attributes: object,
    ) -> None:
        """Record one minimized permission-policy mutation."""
        telemetry_audit(event_name, scope=scope.value, **attributes)
        self._append_audit(event_name, {"scope": scope.value, **attributes})
