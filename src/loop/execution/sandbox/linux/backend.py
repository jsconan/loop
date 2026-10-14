"""Compose the installed ordinary Linux runtime with shared command and selected-search callers.

Selection requires original installed authority. Every command authenticates that authority and
enters native enforcement before source; alternate records support owned qualification fixtures.
"""

import json
import os
import shlex
import stat
import sys
import tempfile
import time
from pathlib import Path

from .... import constants
from ....constants import NATIVE_FILE_READ_CHUNK_BYTES, NATIVE_INSTALLATION_DIRECTORY
from ....utils.process import ProcessCapture, ProcessCaptureStatus
from ..constants import (
    COMMAND_SANDBOX_SCRATCH_PREFIX,
    LC_ALL_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    SYSTEM_LOCALE,
    TMPDIR_ENVIRONMENT_VARIABLE,
)
from ..contracts import CommandProcessResult, SandboxOutcome, SandboxRequest
from .constants import (
    MANAGEMENT_RECORD_NAME,
    NATIVE_POLICY_VERSION,
    NATIVE_RESERVATIONS_DIRECTORY_NAME,
    NATIVE_SCRATCH_PATH,
    PROC_STATUS_PATH,
    SELECTED_DIRECTORY_NAME,
    SYSTEM_PYTHON_PATH,
)
from .host import (
    NATIVE_LOCK_SEEDS,
    bind_host_file,
    in_initial_host_user_namespace,
    protected_path,
    runtime_read_roots,
)
from .paths import python_runtime_mounts
from .runtime import exchange_installed_request, execution_authority

_RECORD = NATIVE_INSTALLATION_DIRECTORY / MANAGEMENT_RECORD_NAME
_MAX_SELECTED_FILE_DESCRIPTOR = 4096
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
    """Bind an administrator-installed runtime for ordinary public command qualification.

    Args:
        record (Path): Original protected installation record; defaults to the fixed production
            location. Alternate records are for owned native qualification fixtures.
    """

    _record: Path
    _authority_seen: bool

    def __init__(self, record: Path = _RECORD) -> None:
        self._record = record
        self._authority_seen = False

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
        """Find the user-local Python runtime needed by an identity-bound uv command.

        Args:
            workspace (Path): Canonical selected workspace.
            path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and directories.
            executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Bound named
                executable files and their device/inode identities.

        Returns:
            tuple[Path, ...]: Safe read-only Python runtime roots referenced by a workspace venv.

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
            execution_authority(self._record)
        except (OSError, ValueError):
            return CommandProcessResult(
                SandboxOutcome.STALE if self._authority_seen else SandboxOutcome.INVALID,
                detail=(
                    "Native Linux installation authority is unsafe or changed; use checked refresh."
                ),
            )

        self._authority_seen = True
        return None

    def run(self, request: SandboxRequest) -> CommandProcessResult:
        """Exchange one authorized request and restore only its original typed result.

        Args:
            request (SandboxRequest): Current shared request after public authorization.

        Returns:
            CommandProcessResult: Bounded exact outcome; failures never retry source.
        """
        if time.monotonic() >= request.deadline:
            return CommandProcessResult(SandboxOutcome.TIMED_OUT)

        if not request.paths_are_current():
            return CommandProcessResult(
                SandboxOutcome.STALE, detail="Approved command paths changed."
            )

        try:
            response = exchange_installed_request(self._record, request.to_wire(), request.deadline)
            return CommandProcessResult.from_wire(response, request.attempt_id)
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
