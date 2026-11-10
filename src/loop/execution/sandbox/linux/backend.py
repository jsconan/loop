"""Compose the installed ordinary Linux runtime with shared command and selected-search callers.

Selection requires original installed authority. Every command authenticates that authority and
enters native enforcement before source; alternate records support owned qualification fixtures.
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .... import constants
from ....constants import NATIVE_FILE_READ_CHUNK_BYTES, NATIVE_INSTALLATION_DIRECTORY
from ....utils.process import ProcessCapture, ProcessCaptureStatus, is_finite_deadline
from ....utils.virtual_path import VirtualPath
from ..constants import (
    COMMAND_SANDBOX_SCRATCH_PREFIX,
    LC_ALL_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    SYSTEM_LOCALE,
    TMPDIR_ENVIRONMENT_VARIABLE,
)
from ..contracts import (
    CommandProcessResult,
    QualificationProbeLayout,
    SandboxCapability,
    SandboxOutcome,
    SandboxRequest,
)
from .commands import LinuxSandboxCommands
from .constants import (
    MANAGEMENT_RECORD_NAME,
    NATIVE_POLICY_VERSION,
    NATIVE_RESERVATIONS_DIRECTORY_NAME,
    NATIVE_SCRATCH_PATH,
    PROC_STATUS_PATH,
    SELECTED_DIRECTORY_NAME,
    SYSTEM_PYTHON_PATH,
)
from .diagnostics import DENIAL_RECORDS_PREFIX, LinuxDenialObserver
from .host import (
    NATIVE_LOCK_SEEDS,
    bind_host_file,
    in_initial_host_user_namespace,
    protected_path,
    runtime_read_roots,
)
from .management import boot_activation_state
from .namespace import _runtime_executable_aliases
from .paths import python_runtime_mounts
from .runtime import (
    NativeFeatureUnavailable,
    NativeRepairRequired,
    exchange_installed_request,
    execution_authority,
)
from .transport import EgressLease

if TYPE_CHECKING:
    from ....commands import CommandRegistration
    from ....permissions.manager import PermissionManager

_RECORD = NATIVE_INSTALLATION_DIRECTORY / MANAGEMENT_RECORD_NAME
_MAX_SELECTED_FILE_DESCRIPTOR = 4096
_BOOT_WAIT_SECONDS = 130.0
_BOOT_POLL_SECONDS = 0.1
_SEARCH_EXEC = """
import fcntl,json,os,sys
try:
 value=json.loads(sys.argv[1])
 opened=[]
 try:
  for number,path in value['files']:
   fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
   opened.append((number,fcntl.fcntl(fd,fcntl.F_DUPFD_CLOEXEC,max(n for n,p in value['files'])+1)))
   os.close(fd)
  for number,fd in opened: os.dup2(fd,number,inheritable=True)
 finally:
  for number,fd in opened: os.close(fd)
 os.execve(value['executable'],[value['executable'],*value['arguments']],dict(os.environ))
except (OSError,ValueError):
 sys.exit(127)
"""


class LinuxNativeBackend:
    """Bind an installed runtime and collect optional unprivileged denial observations.

    Args:
        record (Path): Original protected installation record; defaults to the fixed production
            location. Alternate records are for owned native qualification fixtures.
    """

    _record: Path
    _authority_seen: bool
    capabilities = frozenset(
        {
            SandboxCapability.TERMINAL,
            SandboxCapability.MEDIATED_EGRESS,
            SandboxCapability.INNER_SECCOMP,
            SandboxCapability.LOCAL_NETWORK,
        }
    )
    qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})

    def __init__(self, record: Path = _RECORD) -> None:
        self._record = record
        self._authority_seen = False

    def qualification_layout(self) -> QualificationProbeLayout:
        """Create an owned private parent for independently scoped Linux probes.

        Returns:
            QualificationProbeLayout: Private workspace parent and separate outside parent.

        Raises:
            OSError: The temporary parent or existing probe root is unsafe.
        """
        parent = Path(tempfile.gettempdir()).resolve(strict=True)
        parent_mode = parent.stat()
        private_parent = (
            parent_mode.st_uid == os.getuid()
            and not parent_mode.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK
        )
        sticky_shared_parent = (
            parent_mode.st_uid == 0
            and bool(parent_mode.st_mode & stat.S_ISVTX)
            and bool(parent_mode.st_mode & stat.S_IWOTH)
        )
        if not stat.S_ISDIR(parent_mode.st_mode) or not (private_parent or sticky_shared_parent):
            raise OSError("Temporary parent is not private or root-owned and sticky.")

        root = parent / f"{COMMAND_SANDBOX_SCRATCH_PREFIX}probe-{os.getuid()}"
        try:
            root.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE)
        except FileExistsError:
            pass
        root_mode = root.lstat()
        if (
            not stat.S_ISDIR(root_mode.st_mode)
            or root_mode.st_uid != os.getuid()
            or stat.S_IMODE(root_mode.st_mode) != constants.PRIVATE_DIRECTORY_MODE
        ):
            raise OSError("Linux probe root is not an owned private directory.")
        return QualificationProbeLayout(root, parent)

    def qualification_failure(
        self, result: CommandProcessResult | None, reason: str
    ) -> CommandProcessResult:
        """Preserve unsupported native features and classify other probe failures as invalid.

        Args:
            result (CommandProcessResult | None): Native probe result, if execution began.
            reason (str): Sanitized probe failure explanation.

        Returns:
            CommandProcessResult: Typed fail-closed selection failure.
        """
        if result is not None and result.outcome is SandboxOutcome.UNSUPPORTED:
            return result
        return CommandProcessResult(SandboxOutcome.INVALID, detail=reason)

    def close(self) -> None:
        """Release no persistent resources from this installed backend."""

    def classify_denial_candidates(
        self,
        observed: str,
    ) -> tuple[tuple[tuple[str, bool], ...], str]:
        """Return bounded unclassified journal records without claiming a startup baseline.

        Args:
            observed (str): Bounded observations produced by the local journal observer.

        Returns:
            tuple[tuple[tuple[str, bool], ...], str]: Records and false startup-match flags,
                followed by unavailable startup-baseline status.
        """
        if not observed.startswith(DENIAL_RECORDS_PREFIX):
            return ((observed, False),), "unavailable"
        try:
            values = json.loads(observed.removeprefix(DENIAL_RECORDS_PREFIX))
        except ValueError:
            return (), "unavailable"
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 4
            or any(not isinstance(value, str) or not value or len(value) > 500 for value in values)
        ):
            return (), "unavailable"
        return tuple((value, False) for value in values), "unavailable"

    def _require_socket(self) -> None:
        """Require original installed authority and its currently present socket endpoint."""
        if not execution_authority(self._record).is_socket():
            raise FileNotFoundError("Native sandbox admission socket is not ready.")

    def _wait_for_boot(self) -> bool:
        """Join an authenticated in-progress boot activation without retrying a command."""
        if self._record != _RECORD:
            return False
        deadline = time.monotonic() + _BOOT_WAIT_SECONDS
        try:
            while True:
                state = boot_activation_state(self._record.parent, deadline)
                if state not in {"active", "activating"}:
                    return False
                if state == "active":
                    try:
                        self._require_socket()
                    except (OSError, ValueError):
                        return False
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(_BOOT_POLL_SECONDS, remaining))
        except (OSError, ValueError):
            return False

    def output_path_roots(self, request: SandboxRequest) -> tuple[tuple[str, str], ...]:
        """Describe native namespace paths for command-result disclosure.

        Args:
            request (SandboxRequest): Bound session request.

        Returns:
            tuple[tuple[str, str], ...]: Namespace scratch path and its public root.
                Raw native results and selected-file search bytes remain unchanged.
        """
        return ((str(NATIVE_SCRATCH_PATH), VirtualPath.TEMPORARY),)

    @property
    def trusted_file_owner_ids(self) -> tuple[int, ...]:
        """Return file owners trusted for automatic executable binding.

        Returns:
            tuple[int, ...]: Current user and any verified host administrator identity.
        """
        uid = os.getuid()
        if not in_initial_host_user_namespace():
            return (uid,)
        return tuple(dict.fromkeys((0, uid)))

    def trusted_system_path(self, path: Path, *, directory: bool = False) -> bool:
        """Authenticate a root-owned system path and its ancestry for executable selection.

        Args:
            path (Path): Absolute system path to validate.
            directory (bool): Require a directory instead of a regular executable file.

        Returns:
            bool: Whether the original host namespace supplies protected system authority.

        Raises:
            OSError: Protected host metadata cannot be inspected.
            ValueError: The path or its ancestry is not protected.
        """
        del self
        if not in_initial_host_user_namespace():
            return False
        return protected_path(path, directory=directory) == path.resolve(strict=True)

    def system_executable_requires_alias(
        self, spelling: Path, resolved: Path, workspace: Path, requested_roots: tuple[Path, ...]
    ) -> bool:
        """Check whether a protected executable needs an unbound namespace path.

        Args:
            spelling (Path): Selected absolute executable spelling.
            resolved (Path): Canonical protected executable target.
            workspace (Path): Current workspace, which is already mounted.
            requested_roots (tuple[Path, ...]): Explicit roots that still require normal approval.

        Returns:
            bool: Whether the executable needs an unbound scope or missing alternatives link.

        Raises:
            ValueError: The protected chain or its required alias is unsafe or unsupported.
            OSError: Required host identity or filesystem metadata is unavailable.
        """
        del self
        details = resolved.stat()
        request = SimpleNamespace(
            executable_identities=(
                (spelling.name, spelling, resolved, details.st_dev, details.st_ino),
            ),
            workspace=workspace,
            read_roots=requested_roots,
            write_roots=(),
        )
        bindings = tuple(
            (path, protected_path(path, directory=True)) for path in runtime_read_roots()
        )
        covered = (workspace, *requested_roots, *(path for pair in bindings for path in pair))
        if not all(
            any(path.is_relative_to(root) for root in covered) for path in (spelling, resolved)
        ):
            return True
        mounted = (*bindings, *((root, root) for root in (workspace, *requested_roots)))
        return bool(_runtime_executable_aliases(request, mounted))

    @property
    def system_read_roots(self) -> tuple[Path, ...]:
        """Return protected native runtime directories without an architecture allowlist.

        Returns:
            tuple[Path, ...]: Current administrator-protected distribution runtime directories.

        Raises:
            ValueError: Native runtime path authority is unsafe.
            OSError: Native runtime metadata is unavailable.
        """
        return runtime_read_roots()

    def automatic_tool_read_roots(
        self,
        workspace: Path,
        path_roots: tuple[tuple[Path, Path], ...],
        executable_identities: tuple[tuple[str, Path, Path, int, int], ...],
    ) -> tuple[Path, ...]:
        """Discover candidate runtime roots for selected workspace interpreters.

        Args:
            workspace (Path): Canonical selected workspace.
            path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and directories.
            executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Bound named
                executable files and their device/inode identities.

        Returns:
            tuple[Path, ...]: Candidate Python runtime roots requiring separate read approval.

        Raises:
            ValueError: A referenced user-local runtime is unsafe or ambiguous.
            OSError: Runtime path metadata cannot be inspected.
        """
        return tuple(
            dict.fromkeys(
                root
                for root, _ in python_runtime_mounts(
                    workspace,
                    path_roots,
                    executable_identities,
                    system_roots=self.system_read_roots,
                )
            )
        )

    @property
    def policy_version(self) -> str:
        """Return the installed filename/namespace/descriptor policy revision.

        Returns:
            str: Stable Linux policy version included in shared authorization bindings.
        """
        return NATIVE_POLICY_VERSION

    @property
    def scratch_prefix(self) -> str:
        """Return the common private Linux scratch prefix.

        Returns:
            str: Prefix for ordinary private per-command scratch.
        """
        return COMMAND_SANDBOX_SCRATCH_PREFIX

    @property
    def file_creation_mask(self) -> int:
        """Return the current process umask without changing it.

        Returns:
            int: POSIX permission bits removed from newly created files and directories.

        Raises:
            OSError: The live Linux process umask cannot be read unambiguously.
        """
        masks = [
            line.partition(":")[2].strip()
            for line in PROC_STATUS_PATH.read_text(encoding="ascii").splitlines()
            if line.startswith("Umask:")
        ]
        if len(masks) != 1:
            raise OSError("Linux process umask is unavailable.")
        try:
            value = int(masks[0], 8)
        except ValueError as error:
            raise OSError("Linux process umask is invalid.") from error
        if not 0 <= value <= constants.FILE_MODE_PERMISSION_MASK:
            raise OSError("Linux process umask is invalid.")
        return value

    def managed_tool_root(self, root: Path) -> bool:
        """Accept automatic narrow tool authority only for protected canonical directories.

        Args:
            root (Path): Canonical candidate installed-tool directory.

        Returns:
            bool: Whether administrator-protected ancestry authenticates this directory.
        """
        if sys.platform == "linux" and not in_initial_host_user_namespace():
            return False
        try:
            return protected_path(root, directory=True) == root
        except (OSError, ValueError):
            return False

    @staticmethod
    def installed_tool_roots(
        resolved: Path,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]:
        """Recommend a narrow explicit root for a tool outside the default runtime.

        Args:
            resolved (Path): Canonical executable selected by the shared resolver.

        Returns:
            tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]: Parent read root and
                no inferred package aliases or dependency directories.
        """
        return (resolved.parent,), ()

    def capability_failure(self) -> CommandProcessResult | None:
        """Authenticate installation authority while preserving fail-closed failure categories.

        Missing fresh setup permits separately approved host recovery. Partial, unsafe or
        previously observed installation authority never permits that recovery. Metadata checks
        alone do not prove enforcement; trusted payload entry checks it before source starts.

        Returns:
            CommandProcessResult | None: Path-free unavailable, stale or invalid failure, or None
                for intact installed authority.
        """
        try:
            try:
                self._record.lstat()
            except FileNotFoundError:
                if self._authority_seen:
                    return CommandProcessResult(
                        SandboxOutcome.STALE, detail="Native Linux installation authority changed."
                    )

                # Completed uninstall retains only shared scaffolding. Any other inventory
                # is partial authority, even when the management record itself is missing.
                try:
                    directory = protected_path(self._record.parent, directory=True)
                except FileNotFoundError:
                    # Authenticate the nearest existing ancestor before declaring fresh absence.
                    # A dangling alias or unsafe ancestor is invalid authority, not missing setup.
                    ancestor = self._record.parent
                    while not (ancestor.exists() or ancestor.is_symlink()):
                        ancestor = ancestor.parent

                    if protected_path(ancestor, directory=True) != ancestor:
                        raise ValueError("Aliased native installation ancestry.")

                    directory = None

                if directory is not None:
                    if directory != self._record.parent:
                        raise ValueError("Aliased native installation directory.")

                    inventory = {path.name for path in directory.iterdir()}
                    if inventory - {*NATIVE_LOCK_SEEDS, NATIVE_RESERVATIONS_DIRECTORY_NAME}:
                        raise ValueError("Partial native installation authority.")

                    for name in inventory & NATIVE_LOCK_SEEDS.keys():
                        seed = NATIVE_LOCK_SEEDS[name]
                        bound = bind_host_file(directory / name)
                        with bound.open_verified() as descriptor:
                            if (
                                bound.identity[2] & constants.WRITE_MODE_MASK
                                or os.pread(descriptor, len(seed) + 1, 0) != seed
                            ):
                                raise ValueError("Changed native shared lock authority.")

                    if NATIVE_RESERVATIONS_DIRECTORY_NAME in inventory:
                        reservations = directory / NATIVE_RESERVATIONS_DIRECTORY_NAME
                        if protected_path(reservations, directory=True) != reservations or any(
                            reservations.iterdir()
                        ):
                            raise ValueError("Retained native reservation authority.")

                return CommandProcessResult(
                    SandboxOutcome.UNAVAILABLE,
                    detail="Native Linux setup is absent; use checked sandbox setup.",
                    failure_context="Native Linux installation is absent.",
                )
            self._require_socket()
        except NativeFeatureUnavailable as error:
            return CommandProcessResult(SandboxOutcome.UNSUPPORTED, detail=str(error))
        except NativeRepairRequired as error:
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail=str(error),
            )
        except (OSError, ValueError):
            if self._wait_for_boot():
                self._authority_seen = True
                return None
            return CommandProcessResult(
                SandboxOutcome.STALE if self._authority_seen else SandboxOutcome.INVALID,
                detail=(
                    "Native Linux installation authority is unsafe or changed; use checked refresh."
                ),
            )

        self._authority_seen = True
        return None

    def qualification_probe(self, request: SandboxRequest) -> CommandProcessResult:
        """Run the ordinary installed native request as an enforcement probe.

        Args:
            request (SandboxRequest): Owned probe request with observable denied effects.

        Returns:
            CommandProcessResult: Native result for independent enforcement verification.
        """
        return self.run(request)

    def complete_qualification(self, request: SandboxRequest) -> None:
        """Retain installed capabilities after the selector verifies enforcement.

        Args:
            request (SandboxRequest): Owned request whose denied effects were verified.
        """
        del request

    def run(
        self,
        request: SandboxRequest,
        *,
        cancellation_event: Event | None = None,
        policy_authority: PermissionManager | None = None,
        approved_revision: int | None = None,
        output_observer: Callable[[str, str], None] | None = None,
        terminal_input: Callable[[], bytes | None] | None = None,
    ) -> CommandProcessResult:
        """Exchange one authorized request and restore only its original typed result.

        Args:
            request (SandboxRequest): Current shared request after public authorization.
            cancellation_event (Event | None): Optional run cancellation signal. Cancellation
                observed before transport prevents receiver launch; after transport begins it
                requests cleanup and effects may be possible.
            policy_authority (PermissionManager | None): Active authority for mediated egress.
            approved_revision (int | None): Revision captured with exact command approval.
            output_observer (Callable[[str, str], None] | None): Optional bounded live output sink.
            terminal_input (Callable[[], bytes | None] | None): Next bounded terminal control
                frame for the original interactive request.

        Returns:
            CommandProcessResult: Bounded exact outcome; failures never retry source.
        """
        if not is_finite_deadline(request.deadline):
            return CommandProcessResult(
                SandboxOutcome.INVALID, detail="Command deadline is invalid."
            )
        if time.monotonic() >= request.deadline:
            return CommandProcessResult(SandboxOutcome.TIMED_OUT)
        if cancellation_event is not None and cancellation_event.is_set():
            return CommandProcessResult(SandboxOutcome.CANCELLED)
        if request.terminal != (terminal_input is not None):
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Interactive terminal requires its original owned control channel.",
            )
        if request.terminal and output_observer is None:
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Interactive terminal requires its original bounded output channel.",
            )
        egress_requested = isinstance(request.egress_origins, tuple) and bool(
            request.egress_origins
        )
        if request.terminal and egress_requested:
            return CommandProcessResult(
                SandboxOutcome.UNSUPPORTED,
                detail="Interactive terminal and mediated egress cannot share this control path.",
            )
        if egress_requested and (policy_authority is None or approved_revision is None):
            return CommandProcessResult(
                SandboxOutcome.UNSUPPORTED,
                detail="Mediated egress requires active policy authority.",
            )

        if not request.paths_are_current():
            return CommandProcessResult(
                SandboxOutcome.STALE, detail="Approved command paths changed."
            )

        try:
            payload = (
                replace(request, stream_output=True).to_wire()
                if output_observer is not None
                else request.to_wire()
            )
        except ValueError:
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Approved command request is stale or exceeds the native transport limit.",
            )
        except TimeoutError:
            return CommandProcessResult(SandboxOutcome.TIMED_OUT)
        except OSError:
            return CommandProcessResult(
                SandboxOutcome.INVALID, detail="Approved command request could not be inspected."
            )
        except KeyboardInterrupt:
            return CommandProcessResult(SandboxOutcome.CANCELLED)

        observer = LinuxDenialObserver()
        try:
            exchange_arguments = (
                {"cancellation_event": cancellation_event} if cancellation_event is not None else {}
            )
            exchange_arguments["observer"] = observer
            if output_observer is not None:
                exchange_arguments["output_observer"] = output_observer
            if terminal_input is not None:
                exchange_arguments["terminal_input"] = terminal_input
            if egress_requested:
                exchange_arguments["egress_lease"] = EgressLease(
                    policy_authority, approved_revision
                )
            if cancellation_event is not None and cancellation_event.is_set():
                return CommandProcessResult(SandboxOutcome.CANCELLED)
            response = exchange_installed_request(
                self._record,
                payload,
                request.deadline,
                **exchange_arguments,
            )
            result = CommandProcessResult.from_wire(response, request.attempt_id)
            if (
                result.diagnostic_status != "unknown"
                or result.diagnostic_source != "unknown"
                or result.observed_denial
            ):
                raise ValueError("Linux receiver returned unsupported diagnostic provenance.")
            observed, status = observer.finish()
            if result.outcome is not SandboxOutcome.COMPLETED:
                observed = ""
                status = "unknown" if status == "available" else status
            return replace(
                result,
                observed_denial=observed,
                diagnostic_status=status,
                diagnostic_source="linux-kernel-journal" if observed else "unknown",
            )
        except TimeoutError:
            return CommandProcessResult(
                SandboxOutcome.TIMED_OUT,
                possible_effects=True,
                stdout_discarded=None,
                stderr_discarded=None,
            )
        except KeyboardInterrupt:
            return CommandProcessResult(
                SandboxOutcome.CANCELLED,
                possible_effects=True,
                stdout_discarded=None,
                stderr_discarded=None,
            )
        except NativeFeatureUnavailable as error:
            return CommandProcessResult(SandboxOutcome.UNSUPPORTED, detail=str(error))
        except NativeRepairRequired as error:
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail=str(error),
            )
        except ValueError:
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Installed request or result authority differs.",
                possible_effects=True,
                stdout_discarded=None,
                stderr_discarded=None,
            )
        except OSError:
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Installed native transport failed.",
                possible_effects=True,
                stdout_discarded=None,
                stderr_discarded=None,
                failure_context="Native receiver connection or exchange failed.",
            )
        finally:
            observer.finish()

    def run_read_only_argv(
        self,
        executable: Path,
        arguments: tuple[str, ...],
        descriptors: tuple[int, ...],
        deadline: float,
    ) -> ProcessCapture | None:
        """Run qualified ripgrep on bounded private copies of selected file descriptors.

        Args:
            executable (Path): Canonical protected ripgrep qualified by the shared facade.
            arguments (tuple[str, ...]): Fixed native search argv ending in selected fd paths.
            descriptors (tuple[int, ...]): Preopened authorized regular selected-file descriptors.
            deadline (float): Original monotonic search deadline.

        Returns:
            ProcessCapture | None: Native typed result, or None for safe prelaunch unavailability.

        Raises:
            RuntimeError: Selected content or installed native authority changes.
        """
        if time.monotonic() >= deadline:
            return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)

        try:
            if (
                not self.managed_tool_root(executable.parent)
                or not descriptors
                or (
                    len(set(descriptors)) != len(descriptors)
                    or any(
                        not isinstance(fd, int)
                        or isinstance(fd, bool)
                        or not 3 <= fd <= _MAX_SELECTED_FILE_DESCRIPTOR
                        for fd in descriptors
                    )
                    or arguments[-len(descriptors) :]
                    != tuple(f"/dev/fd/{fd}" for fd in descriptors)
                )
            ):
                return None

            with tempfile.TemporaryDirectory(prefix=self.scratch_prefix) as private:
                scratch = Path(private).resolve()
                workspace = scratch / SELECTED_DIRECTORY_NAME
                workspace.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE)
                selected = []
                for number in descriptors:
                    before = os.fstat(number)
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                        return None

                    path = workspace / str(number)
                    with path.open("xb") as stream:
                        offset = 0
                        while offset < before.st_size:
                            if time.monotonic() >= deadline:
                                return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)

                            chunk = os.pread(
                                number,
                                min(NATIVE_FILE_READ_CHUNK_BYTES, before.st_size - offset),
                                offset,
                            )
                            if not chunk:
                                raise ValueError("Selected search input changed during snapshot.")

                            stream.write(chunk)
                            offset += len(chunk)

                    after = os.fstat(number)
                    if (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        raise ValueError("Selected search input changed during snapshot.")

                    selected.append(
                        (number, str(NATIVE_SCRATCH_PATH / SELECTED_DIRECTORY_NAME / str(number)))
                    )

                helper = bind_host_file(executable)
                source = (
                    "cd "
                    + str(NATIVE_SCRATCH_PATH / SELECTED_DIRECTORY_NAME)
                    + " && exec "
                    + shlex.join(
                        [
                            str(SYSTEM_PYTHON_PATH),
                            "-I",
                            "-c",
                            _SEARCH_EXEC,
                            json.dumps(
                                {
                                    "files": selected,
                                    "executable": str(executable),
                                    "arguments": (
                                        *arguments[: -len(descriptors)],
                                        *(f"/dev/fd/{number}" for number in descriptors),
                                    ),
                                }
                            ),
                        ]
                    )
                )
                request = SandboxRequest.create(
                    source="selected read-only search",
                    execution_source=source,
                    cwd=workspace,
                    workspace=workspace,
                    read_roots=(executable.parent,),
                    write_roots=(),
                    network=False,
                    environment={
                        PATH_ENVIRONMENT_VARIABLE: str(executable.parent)
                        + ":"
                        + MINIMAL_SYSTEM_PATH,
                        LC_ALL_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                        TMPDIR_ENVIRONMENT_VARIABLE: str(scratch),
                    },
                    policy_version=self.policy_version,
                    deadline=deadline,
                    workspace_id="native-search",
                    executable_identities=(
                        (
                            executable.name,
                            executable,
                            executable,
                            helper.identity[0],
                            helper.identity[1],
                        ),
                    ),
                )
                result = self.run(request)

                if result.outcome in (SandboxOutcome.INVALID, SandboxOutcome.STALE):
                    raise RuntimeError("Native search authority changed or is invalid.")

                if result.outcome is SandboxOutcome.UNSUPPORTED:
                    if result.possible_effects:
                        raise RuntimeError("Native search may already have had effects.")
                    return None

                if result.outcome is SandboxOutcome.UNAVAILABLE:
                    return None

                if result.outcome is SandboxOutcome.COMPLETED and result.exit_code in (126, 127):
                    return None

                return ProcessCapture(
                    ProcessCaptureStatus(result.outcome.value),
                    result.exit_code,
                    result.stdout,
                    result.stderr,
                    result.stdout_discarded,
                    result.stderr_discarded,
                )

        except TimeoutError:
            return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)
        except ValueError as error:
            raise RuntimeError("Native search preparation authority is invalid.") from error
        except OSError:
            return None


def create_backend() -> LinuxNativeBackend:
    """Construct the selected Linux native backend.

    Returns:
        LinuxNativeBackend: Backend bound to the installed native generation.
    """
    return LinuxNativeBackend()


def management_commands(
    main_app: bool,
    refresh_backend: Callable[[], None] | None,
) -> tuple[CommandRegistration, ...]:
    """Register Linux native management commands.

    Args:
        main_app (bool): Whether direct interactive consent is permitted.
        refresh_backend (Callable[[], None] | None): Reselect authority after approved changes.

    Returns:
        tuple[CommandRegistration, ...]: Linux installation commands.
    """
    return LinuxSandboxCommands(main_app=main_app, refresh_backend=refresh_backend).get_commands()
