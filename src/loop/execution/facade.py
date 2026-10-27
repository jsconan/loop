"""Bind and execute shell requests behind one authorization facade."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .. import constants
from ..errors import Problem
from ..permissions import ProcessBoundary, ProcessTarget
from ..permissions.protection import is_protected_external_read
from ..utils import ProcessCapture, VirtualPath, encode_content_cursor, ripgrep_path, store_content
from .coordinator import CommandExecutionCoordinator, HostCommandExecutor, LocalHostCommandExecutor
from .sandbox import (
    CommandProcessResult,
    DirectoryIdentity,
    SandboxBackend,
    SandboxOutcome,
    SandboxRequest,
    path_search_roots,
    resolve_host_executable,
    select_sandbox_backend,
)
from .sandbox.constants import (
    LANG_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    SYSTEM_LOCALE,
    TEMP_ENVIRONMENT_VARIABLE,
    TMP_ENVIRONMENT_VARIABLE,
    TMPDIR_ENVIRONMENT_VARIABLE,
    WORKSPACE_ROOT_ENVIRONMENT_VARIABLE,
    XDG_CACHE_HOME_ENVIRONMENT_VARIABLE,
)

if TYPE_CHECKING:
    from ..tooling import ToolContext


@dataclass(frozen=True)
class ExecutableGrant:
    """Bind a discovered tool to its workspace, PATH, and filesystem identities.

    Args:
        name (str): Executable name looked up on PATH.
        workspace (Path | None): Workspace identity when a caller supplied one.
        cwd (Path): Working directory used to resolve relative PATH entries.
        relative_path (bool): Whether the grant's PATH snapshot depends on cwd.
        path_value (str): Bound PATH value used for lookup.
        spelling (Path): Executable path as found on PATH.
        resolved (Path): Canonical executable target.
        roots (tuple[DirectoryIdentity, ...]): Bound installation read roots.
        identity (tuple[int, int]): Executable device and inode.
        opt_aliases (tuple[tuple[Path, int, int, Path], ...]): Bound package aliases.
        managed_package (bool): Whether all roots qualify for automatic package reads.
    """

    name: str
    workspace: Path | None
    cwd: Path
    relative_path: bool
    path_value: str
    spelling: Path
    resolved: Path
    roots: tuple[DirectoryIdentity, ...]
    identity: tuple[int, int]
    opt_aliases: tuple[tuple[Path, int, int, Path], ...]
    managed_package: bool


def _stream_output(
    chunks: list[str],
    discarded: int | None,
    source: str,
    redact: Callable[[str], str],
) -> dict:
    """Return a recoverable preview and explicit capture-loss status for one stream."""
    content = redact("".join(chunks))
    encoded = content.encode("utf-8")
    preview = encoded[: constants.MAX_TOOL_CONTENT_BYTES // 2].decode("utf-8", errors="ignore")
    included = len(preview.encode("utf-8"))
    result = {
        "content": preview,
        "captured_bytes": len(encoded),
        "included_bytes": included,
        "truncated": included < len(encoded) or discarded != 0,
        "capture_complete": discarded == 0,
        "discarded_characters": discarded,
    }
    if included < len(encoded):
        handle = store_content(encoded, source)
        result.update(
            handle=handle,
            next_cursor=encode_content_cursor(handle, included),
            continuation="Use read_cached_content with this handle and cursor.",
        )
    return result


def _automatic_executable_identities(
    names: tuple[str, ...],
    path_value: str,
    cwd: Path,
    *,
    trusted_owners: tuple[int, ...] | None = None,
) -> tuple[tuple[str, Path, Path, int, int], ...]:
    """Bind safe exact executable files found in the request's live PATH roots."""
    roots = path_search_roots({PATH_ENVIRONMENT_VARIABLE: path_value}, cwd)
    if trusted_owners is None:
        trusted_owners = (os.getuid(),)
    identities = []
    for name in dict.fromkeys(names[:128]):
        if "/" in name:
            continue
        try:
            match = resolve_host_executable(name, path_value, cwd)
        except (OSError, RuntimeError, ValueError):
            continue
        if match is None:
            continue
        spelling, resolved = match
        root = next(
            (
                canonical
                for path, canonical in roots
                if spelling.is_relative_to(path) and resolved.is_relative_to(canonical)
            ),
            None,
        )
        if root is None:
            continue
        if is_protected_external_read(spelling) or is_protected_external_read(resolved):
            continue
        try:
            root_details = root.stat()
            details = resolved.stat()
        except OSError:
            continue
        if root_details.st_uid not in trusted_owners:
            continue
        if root_details.st_mode & constants.OTHER_WRITE_MODE_BIT:
            continue
        if (
            root_details.st_mode & constants.GROUP_WRITE_MODE_BIT
            and root_details.st_gid != os.getgid()
        ):
            continue
        if details.st_uid not in trusted_owners:
            continue
        if details.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK:
            continue
        if details.st_nlink != 1:
            continue
        identities.append((name, spelling, resolved, details.st_dev, details.st_ino))
    return tuple(identities)


class CommandExecutionService:
    """Own request binding and sandbox/host execution for registered command tools.

    Args:
        backend (SandboxBackend): Platform sandbox selected by the composition root.
        host_executor (HostCommandExecutor): Supervisor for separately approved host retries.
    """

    _backend: SandboxBackend
    _host_executor: HostCommandExecutor
    _executable_grants: dict[str, ExecutableGrant]

    def __init__(self, backend: SandboxBackend, host_executor: HostCommandExecutor) -> None:
        self._backend = backend
        self._host_executor = host_executor
        self._executable_grants = {}

    def refresh_backend(self) -> None:
        """Reselect native authority after an approved sandbox setup change.

        Rebuild the backend against the current installed generation, discard grants bound to the
        previous backend, and close its resources after the replacement is ready.
        """
        replacement = select_sandbox_backend()
        previous = self._backend
        self._backend = replacement
        self._executable_grants.clear()
        close = getattr(previous, "close", None)
        if callable(close):
            close()

    @classmethod
    def for_host(cls) -> CommandExecutionService:
        """Compose trusted built-in execution boundaries for the current host.

        Returns:
            CommandExecutionService: Native-probed sandbox and separate host supervisor.
        """
        return cls(select_sandbox_backend(), LocalHostCommandExecutor())

    def resolve_executable(self, context: ToolContext, name: str, cwd: str = ".") -> dict:
        """Look up one host tool and offer a bound sandbox read reference.

        Args:
            context (ToolContext): Invocation context and model path boundary.
            name (str): Simple executable name to inspect on host PATH.
            cwd (str): Workspace working directory used to resolve relative PATH entries.

        Returns:
            dict: Host availability and, when safe, an opaque read grant reference.

        Raises:
            ValueError: If the name or PATH cannot be searched safely.
        """
        path_value = os.environ.get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH)
        workspace = (
            Path(
                context.instructions_manager.virtual_paths.resolve(VirtualPath.WORKSPACE)
            ).resolve()
            if context.instructions_manager is not None
            else None
        )
        if context.instructions_manager is not None:
            candidate = Path(cwd)
            if (
                not candidate.is_absolute()
                or cwd == VirtualPath.WORKSPACE
                or cwd.startswith(f"{VirtualPath.WORKSPACE}/")
            ):
                candidate = Path(context.instructions_manager.virtual_paths.resolve(cwd))
            try:
                lookup_cwd = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError("Executable lookup working directory is unavailable.") from exc
        else:
            lookup_cwd = Path(cwd)
            if not lookup_cwd.is_absolute():
                lookup_cwd = Path.cwd() / lookup_cwd
            try:
                lookup_cwd = lookup_cwd.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError("Executable lookup working directory is unavailable.") from exc
        if not lookup_cwd.is_dir() or (
            workspace is not None and not lookup_cwd.is_relative_to(workspace)
        ):
            raise ValueError("Executable lookup working directory must be inside the workspace.")
        match = resolve_host_executable(name, path_value, lookup_cwd)
        if match is None:
            return {"name": name, "status": "absent_from_host_path"}
        spelling, resolved = match
        relative_path = any(not Path(item).is_absolute() for item in path_value.split(os.pathsep))
        system_roots = getattr(self._backend, "system_read_roots", ())
        if not isinstance(system_roots, tuple):
            system_roots = ()
        readable = any(
            resolved.is_relative_to(root) and spelling.is_relative_to(root) for root in system_roots
        ) or (workspace is not None and resolved.is_relative_to(workspace))
        result = {
            "name": name,
            "status": "installed_on_host",
            "path": context.display_path(spelling),
            "resolved_path": context.display_path(resolved),
            "sandbox_access": "default_read" if readable else "external_read_grant_required",
        }
        if readable:
            return result
        resolver = getattr(self._backend, "installed_tool_roots", None)
        installation = resolver(resolved) if callable(resolver) else None
        roots, opt_aliases = (
            installation
            if isinstance(installation, tuple) and len(installation) == 2
            else ((resolved.parent,), ())
        )
        if any(is_protected_external_read(item) for item in roots):
            result["sandbox_access"] = "protected_external_path"
            return result
        try:
            identities = tuple(DirectoryIdentity.capture(item) for item in roots)
            details = resolved.stat()
        except (OSError, ValueError):
            result["sandbox_access"] = "external_grant_unavailable"
            return result
        reference = uuid4().hex
        if len(self._executable_grants) >= 128:
            self._executable_grants.pop(next(iter(self._executable_grants)))
        self._executable_grants[reference] = ExecutableGrant(
            name=name,
            workspace=workspace,
            cwd=lookup_cwd,
            relative_path=relative_path,
            path_value=path_value,
            spelling=spelling,
            resolved=resolved,
            roots=identities,
            identity=(details.st_dev, details.st_ino),
            opt_aliases=opt_aliases,
            managed_package=all(self._backend.managed_tool_root(item) is True for item in roots)
            if callable(getattr(self._backend, "managed_tool_root", None))
            else False,
        )
        result["grant_reference"] = reference
        if context.instructions_manager is None:
            result["candidate_read_roots"] = [str(item) for item in roots]
        result["grant_note"] = (
            "Pass grant_reference as executable_grant to run_command. Verified installed "
            "package roots are available without a separate prompt; other tool folders "
            "require an additional-read approval. Manual read roots and other expanded "
            "capabilities also require approval. "
            "If this tool starts another external executable, look up that executable "
            "and pass both references as a list; keep the command inside the sandbox."
        )
        return result

    def _executable_read_roots(
        self,
        reference: str,
        workspace: Path,
        path_value: str,
        cwd: Path,
    ) -> tuple[Path, ...]:
        """Resolve a lookup reference only while its host identity remains current."""
        grant = self._executable_grants.get(reference)
        if grant is None:
            raise ValueError("Executable grant reference is unknown or expired.")
        name = grant.name
        bound_workspace = grant.workspace
        bound_cwd = grant.cwd
        cwd_bound = grant.relative_path
        bound_path = grant.path_value
        spelling = grant.spelling
        resolved = grant.resolved
        identities = grant.roots
        identity = grant.identity
        opt_aliases = grant.opt_aliases
        if (
            (bound_workspace is not None and bound_workspace != workspace)
            or bound_path != path_value
            or (cwd_bound and bound_cwd != cwd)
        ):
            raise ValueError("Executable grant context changed; look up the tool again.")
        match = resolve_host_executable(name, path_value, cwd)
        try:
            details = resolved.stat()
        except OSError as exc:
            raise ValueError("Executable grant target changed; look up the tool again.") from exc
        if (
            match != (spelling, resolved)
            or (details.st_dev, details.st_ino) != identity
            or not all(item.is_current() for item in identities)
        ):
            raise ValueError("Executable grant identity changed; look up the tool again.")
        if grant.managed_package and not all(
            self._backend.managed_tool_root(item.path) is True for item in identities
        ):
            raise ValueError("Installed package verification changed; look up the tool again.")
        for alias, device, inode, target in opt_aliases:
            try:
                alias_details = alias.lstat()
                if (alias_details.st_dev, alias_details.st_ino) != (device, inode) or alias.resolve(
                    strict=True
                ) != target:
                    raise ValueError("Executable alias changed; look up the tool again.")
            except OSError as exc:
                raise ValueError("Executable alias changed; look up the tool again.") from exc
        return tuple(item.path for item in identities)

    def _automatic_tool_references(
        self,
        context: ToolContext,
        names: tuple[str, ...],
        workspace: Path,
        path_value: str,
        cwd: Path | None = None,
    ) -> tuple[str, ...]:
        """Bind verified managed tools named by a shell command or workspace PATH."""
        candidates = list(names)
        working_directory = cwd or workspace
        for path, _ in path_search_roots(
            {PATH_ENVIRONMENT_VARIABLE: path_value}, working_directory
        ):
            if not path.is_relative_to(workspace) or not path.is_dir():
                continue
            try:
                entries = tuple(path.iterdir())
            except OSError:
                continue
            if len(entries) > 128:
                continue
            candidates.extend(item.name for item in entries if item.is_symlink())
        references: list[str] = []
        for name in list(dict.fromkeys(candidates))[:128]:
            if "/" in name or name in {".", ".."}:
                continue
            try:
                result = self.resolve_executable(context, name, str(working_directory))
            except ValueError:
                continue
            reference = result.get("grant_reference")
            if reference is None:
                continue
            if self._executable_grants[reference].managed_package:
                references.append(reference)
            else:
                self._executable_grants.pop(reference, None)
        return tuple(references)

    def close(self) -> None:
        """Release an owned sandbox backend after command execution stops."""
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def run_trusted_search(
        self,
        arguments: tuple[str, ...],
        descriptors: tuple[int, ...],
        deadline: float,
    ) -> ProcessCapture | None:
        """Run installed ripgrep with fixed argv and selected read descriptors only.

        Args:
            arguments (tuple[str, ...]): Fixed ripgrep options, query, and fd paths.
            descriptors (tuple[int, ...]): Preopened selected files that may be read.
            deadline (float): Absolute monotonic search deadline.

        Returns:
            ProcessCapture | None: Bounded native result, or None for safe in-process fallback.

        Raises:
            RuntimeError: Trusted runtime path authority is invalid.
        """
        runner = getattr(self._backend, "run_read_only_argv", None)
        if not callable(runner):
            return None
        try:
            executable = Path(ripgrep_path())
            executable = executable.resolve(strict=True)
            if executable.stat().st_nlink != 1:
                return None
        except (OSError, RuntimeError, ValueError):
            return None
        try:
            roots = self._backend.system_read_roots
        except (OSError, ValueError) as error:
            raise RuntimeError("Native search runtime authority is invalid.") from error
        try:
            trusted = any(executable.is_relative_to(root) for root in roots)
            if not trusted:
                roots, _ = self._backend.installed_tool_roots(executable)
                trusted = bool(roots) and all(
                    self._backend.managed_tool_root(root) is True for root in roots
                )
            if not trusted:
                return None
        except (OSError, RuntimeError, ValueError):
            return None
        return runner(executable, arguments, descriptors, deadline)

    def _run_command_attempt(
        self,
        context: ToolContext,
        request: SandboxRequest,
        redact: Callable[[str], str],
        *,
        prelaunch_failure: CommandProcessResult | None = None,
    ) -> dict | Problem:
        """Execute one bound request and render canonical prefixes in model-visible output."""
        coordinator = CommandExecutionCoordinator(
            self._backend, self._host_executor, context.permission_manager, context.interaction
        )
        attempt = coordinator.run_sandboxed(
            request,
            context.tool_name,
            prelaunch_failure=prelaunch_failure,
            execution_timeout=context.settings.command_timeout,
            cancellation_event=context.cancellation_event,
        )
        if attempt.result is None:
            return Problem(
                code="tool.denied",
                title="Tool call denied",
                detail="Native command was not approved.",
                operation=context.tool_name,
            )
        result = attempt.result
        boundary = "sandbox"
        if attempt.offer is not None and context.interaction is not None:
            retry = coordinator.retry_host_command(
                attempt.offer,
                request,
                execution_timeout=context.settings.command_timeout,
                cancellation_event=context.cancellation_event,
            )
            if retry is not None:
                result = retry
                boundary = "host"
        output = {
            "boundary": boundary,
            "exit_code": result.exit_code,
            "stdout": _stream_output(
                [result.stdout], result.stdout_discarded, "command stdout", redact
            ),
            "stderr": _stream_output(
                [result.stderr], result.stderr_discarded, "command stderr", redact
            ),
        }
        if boundary == "sandbox":
            output["scratch"] = {
                "path": redact(request.shell_environment()[TMPDIR_ENVIRONMENT_VARIABLE]),
                "lifetime": "this session",
                "retrieval": (
                    f"Use {VirtualPath.TEMPORARY}/... in file tools; quote it in shell commands."
                ),
            }
            if result.outcome is SandboxOutcome.COMPLETED and result.exit_code != 0:
                output["executable_hint"] = (
                    "A nonzero exit does not prove a tool is absent from the host. "
                    "Use resolve_executable with a named tool to check host PATH; "
                    "pass its grant_reference as executable_grant for sandboxed access "
                    "when an external tool is installed. If that tool launches other external "
                    "tools, look up each one and pass their references as a list."
                )
        if attempt.result.observed_denial:
            observed = redact(attempt.result.observed_denial)
            if boundary == "sandbox":
                output["sandbox_diagnostic"] = observed
            else:
                output["sandbox_attempt"] = {
                    "exit_code": attempt.result.exit_code,
                    "observed_denial": observed,
                }
        if attempt.offer is not None and boundary == "sandbox":
            if result.outcome is SandboxOutcome.COMPLETED:
                output["host_offer"] = (
                    "The OS denied an operation during this attempt; it may be unrelated to "
                    f"exit code {result.exit_code}. Command: "
                    f"{json.dumps(redact(request.source))}\n"
                    "A separate, fresh host retry was offered to the user. "
                    "The sandboxed attempt may already have had effects; a retry may repeat them."
                )
            else:
                failure = redact(result.detail)
                output["host_offer"] = (
                    f"Sandbox failure: {failure}\nCommand: "
                    f"{json.dumps(redact(request.source))}\n"
                    "A separate host-specific retry was offered to the user."
                    + (
                        " The sandboxed attempt may already have had effects;"
                        " a host retry may repeat them."
                        if result.possible_effects
                        else ""
                    )
                )
        if result.outcome is SandboxOutcome.COMPLETED or result.possible_effects:
            context.invalidate_instructions()
        if result.outcome is SandboxOutcome.COMPLETED:
            if result.exit_code != 0:
                return Problem(
                    code="process.nonzero_exit",
                    title="Command failed",
                    detail=f"Command exited with code {result.exit_code}.",
                    operation=context.tool_name,
                    metadata=output,
                )
            return output
        return Problem(
            code=f"{boundary if boundary == 'sandbox' else 'host_command'}.{result.outcome.value}",
            title="Sandbox command did not complete"
            if boundary == "sandbox"
            else "Host command did not complete",
            detail=redact(result.detail or result.outcome.value),
            operation=context.tool_name,
            metadata=output,
        )

    def run_command(
        self,
        context: ToolContext,
        command: str,
        cwd: str = ".",
        read_only: bool = False,
        read_roots: list[str] | None = None,
        executable_grant: str | list[str] | None = None,
        network: bool = False,
        git_write: bool = False,
    ) -> dict | Problem:
        """Bind and authorize a shell request, with separate host recovery when needed.

        Args:
            context (ToolContext): Registered identity and dispatch authority.
            command (str): Opaque source passed to the host shell.
            cwd (str): Resolved host working directory.
            read_only (bool): Disable workspace writes while permitting private scratch writes.
            read_roots (list[str] | None): Additional explicitly approved read roots.
            executable_grant (str | list[str] | None): Bound references from executable lookups.
            network (bool): Request general network egress.
            git_write (bool): Request Git metadata write authority.

        Returns:
            dict | Problem: Bounded output or typed failure,
                which may include a separate host offer.
        """
        operation = context.operations[0] if context.operations else None
        target = operation.target if operation is not None else None
        if (
            not isinstance(target, ProcessTarget)
            or target.boundary is not ProcessBoundary.SANDBOXED
        ):
            return Problem(
                code="sandbox.unavailable",
                title="Sandbox unavailable",
                detail="Bound sandbox request is missing.",
                operation=context.tool_name,
            )
        if context.permission_manager is None:
            return Problem(
                code="sandbox.unavailable",
                title="Sandbox unavailable",
                detail="Native command authorization is unavailable.",
                operation=context.tool_name,
            )
        capability_failure = getattr(self._backend, "capability_failure", None)
        failure = capability_failure() if callable(capability_failure) else None
        prelaunch_failure = (
            failure
            if isinstance(failure, CommandProcessResult)
            else CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail=failure,
                failure_context="Native sandbox capability check failed.",
            )
            if isinstance(failure, str)
            else None
        )
        paths = None
        redact = None
        try:
            command_cwd = Path(target.cwd).resolve(strict=True)
            workspace = (
                Path(
                    context.instructions_manager.virtual_paths.resolve(VirtualPath.WORKSPACE)
                ).resolve(strict=True)
                if context.instructions_manager is not None
                else command_cwd
            )
            analysis = context.permission_manager.analyze_command(command)
            findings = analysis.findings
            git_write = git_write or any(finding.requests_git_write for finding in findings)
            if read_only and git_write:
                raise ValueError("A read-only command cannot request Git metadata writes.")
            git_reads, git_roots, git_create = (
                context.permission_manager.git_sandbox_roots(
                    workspace, read=analysis.git_read, write=git_write
                )
                if git_write or analysis.git_read
                else ((), (), False)
            )
            paths = (
                context.instructions_manager.virtual_paths
                if context.instructions_manager is not None
                else VirtualPath(
                    workspace=workspace,
                    temporary_directory=context.permission_manager.temporary_directory,
                )
            )
            path_value = os.environ.get(PATH_ENVIRONMENT_VARIABLE, MINIMAL_SYSTEM_PATH)
            if isinstance(executable_grant, str):
                explicit_references = (executable_grant,) if executable_grant else ()
            else:
                explicit_references = tuple(executable_grant or ())
            references = tuple(
                dict.fromkeys(
                    (
                        *explicit_references,
                        *self._automatic_tool_references(
                            context, analysis.executables, workspace, path_value, command_cwd
                        ),
                    )
                )
            )
            grant_reads = tuple(
                dict.fromkeys(
                    root
                    for reference in references
                    for root in self._executable_read_roots(
                        reference, workspace, path_value, command_cwd
                    )
                )
            )
            automatic_reads = tuple(
                dict.fromkeys(
                    root
                    for reference in references
                    if self._executable_grants[reference].managed_package
                    for root in self._executable_read_roots(
                        reference, workspace, path_value, command_cwd
                    )
                )
            )
            read_aliases = tuple(
                dict.fromkeys(
                    alias[0]
                    for reference in references
                    for alias in self._executable_grants[reference].opt_aliases
                )
            )
            trusted_owners = getattr(self._backend, "trusted_file_owner_ids", None)
            if not isinstance(trusted_owners, tuple):
                trusted_owners = (os.getuid(),)
            executable_identities = tuple(
                dict.fromkeys(
                    (
                        *(
                            (
                                self._executable_grants[reference].name,
                                self._executable_grants[reference].spelling,
                                self._executable_grants[reference].resolved,
                                *self._executable_grants[reference].identity,
                            )
                            for reference in dict.fromkeys(references)
                        ),
                        *_automatic_executable_identities(
                            analysis.executables,
                            path_value,
                            command_cwd,
                            trusted_owners=trusted_owners,
                        ),
                    )
                )
            )
            runtime_reader = getattr(type(self._backend), "automatic_tool_read_roots", None)
            runtime_reads = (
                runtime_reader(
                    self._backend,
                    workspace,
                    path_search_roots({PATH_ENVIRONMENT_VARIABLE: path_value}, command_cwd),
                    executable_identities,
                )
                if runtime_reader is not None
                else ()
            )
            manual_reads = tuple(
                Path(
                    paths.resolve(root)
                    if any(
                        root == virtual or root.startswith(f"{virtual}/")
                        for virtual, _ in paths.command_roots()
                    )
                    else root
                ).resolve(strict=True)
                for root in read_roots or ()
            )
            runtime_reads = tuple(
                root for root in runtime_reads if not root.is_relative_to(workspace)
            )
            skill_reads = tuple(
                root
                for name, root in paths.command_roots()
                if name.startswith(f"{VirtualPath.SKILLS}/") and str(root) in command
            )
            extra_reads = tuple(
                root
                for root in dict.fromkeys(
                    (*grant_reads, *manual_reads, *runtime_reads, *skill_reads)
                )
                if not root.is_relative_to(workspace)
            )
            automatic_reads = tuple(
                dict.fromkeys(
                    (
                        *automatic_reads,
                        *(root for root in runtime_reads if root not in manual_reads),
                    )
                )
            )
            if any(is_protected_external_read(root) for root in extra_reads):
                raise ValueError("Protected external directory cannot be a read root.")
            read_context = context.permission_manager.describe_sandbox_reads(
                workspace,
                extra_reads,
                tuple(root for root in manual_reads if not root.is_relative_to(workspace)),
                tuple(self._executable_grants[reference].name for reference in references),
                bool(grant_reads) and set(grant_reads) <= set(automatic_reads),
            )
            scratch = dict(paths.command_roots()).get(
                VirtualPath.TEMPORARY, context.permission_manager.temporary_directory
            )
            environment = {
                PATH_ENVIRONMENT_VARIABLE: path_value,
                LANG_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                WORKSPACE_ROOT_ENVIRONMENT_VARIABLE: str(workspace),
                TMPDIR_ENVIRONMENT_VARIABLE: str(scratch),
                TMP_ENVIRONMENT_VARIABLE: str(scratch),
                TEMP_ENVIRONMENT_VARIABLE: str(scratch),
                XDG_CACHE_HOME_ENVIRONMENT_VARIABLE: str(scratch / "cache"),
            }
            if read_only:
                environment.update(
                    COVERAGE_FILE=str(scratch / ".coverage"),
                    RUFF_NO_CACHE="true",
                    UV_NO_SYNC="1",
                )
            generated_environment = tuple(
                name
                for name, value in environment.items()
                if name != WORKSPACE_ROOT_ENVIRONMENT_VARIABLE
                and (value == str(scratch) or value.startswith(f"{scratch}/"))
            )

            redaction_roots = {*extra_reads, *read_aliases, *git_reads}
            advertised_roots = {root for _, root in paths.command_roots()}
            system_roots = getattr(self._backend, "system_read_roots", ())
            if not isinstance(system_roots, tuple):
                system_roots = ()
            native_output_roots = ()

            def redact(value: str) -> str:
                """Render canonical path prefixes while hiding unrelated host paths."""
                value = paths.redact(value, roots=native_output_roots)
                for root in sorted(redaction_roots, key=lambda item: len(str(item)), reverse=True):
                    if (
                        root in advertised_roots
                        or root.is_relative_to(workspace)
                        or any(root.is_relative_to(system) for system in system_roots)
                    ):
                        continue
                    value = paths.redact(value, roots=((str(root), VirtualPath.EXTERNAL),))
                return paths.redact(value)

            file_creation_mask = constants.GROUP_AND_OTHER_WRITE_MODE_MASK
            current_mask = getattr(self._backend, "file_creation_mask", None)
            if isinstance(current_mask, int) and not isinstance(current_mask, bool):
                file_creation_mask = current_mask

            request = SandboxRequest.create(
                source=command,
                scratch_root=scratch,
                scratch_data=scratch == context.permission_manager.temporary_directory
                and context.permission_manager.temporary_directory_allows_private_names,
                cwd=command_cwd,
                workspace=workspace,
                read_roots=(*extra_reads, *git_reads),
                write_roots=() if read_only else (workspace, *git_roots),
                network=network,
                environment=environment,
                policy_version=self._backend.policy_version
                if isinstance(getattr(self._backend, "policy_version", None), str)
                else "native-v1",
                deadline=time.monotonic() + context.settings.command_timeout,
                workspace_id=str(workspace),
                git_create=git_create,
                read_context=read_context,
                read_aliases=read_aliases,
                executable_identities=executable_identities,
                automatic_tool_reads=automatic_reads,
                protected_instruction_names=context.permission_manager.protected_sandbox_paths(
                    context.instructions_manager.agents_filenames
                    if context.instructions_manager is not None
                    else ()
                ).instruction_names,
                generated_environment=generated_environment,
                file_creation_mask=file_creation_mask,
            )
            output_paths = getattr(type(self._backend), "output_path_roots", None)
            if output_paths is not None:
                native_output_roots = output_paths(self._backend, request)
        except (OSError, ValueError) as exc:
            return Problem(
                code="sandbox.unavailable",
                title="Sandbox unavailable",
                detail=redact(str(exc))
                if redact
                else (paths.redact(str(exc)) if paths else "Command preparation failed."),
                operation=context.tool_name,
            )
        else:
            return self._run_command_attempt(
                context,
                request,
                redact,
                prelaunch_failure=prelaunch_failure,
            )
