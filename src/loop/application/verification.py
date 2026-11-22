"""Save and dispatch owner-selected workspace verification commands."""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionValue
from ..instructions import InstructionsManager
from ..models import CommandCapability, SandboxEnvironment
from ..sandbox_catalog import validate_sandbox_options
from ..telemetry import telemetry_audit
from ..tooling import ToolRegistry, VerificationScope
from ..utils import (
    VirtualPath,
    open_selected_parent,
    private_directory_lock,
    sha256_digest,
    write_text_atomically,
)

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_PREFIX = "verification-"
_MAX_TASK_BYTES = 64 * 1024


class _VerificationTask(BaseModel):
    """Store the exact shell source and its requested sandbox scopes."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    cwd: str = VirtualPath.WORKSPACE
    read_roots: list[str] = Field(default_factory=list)
    write_roots: list[str] = Field(default_factory=list)
    write_files: list[str] = Field(default_factory=list)
    backend_options: tuple[str, ...] = ()
    environment: SandboxEnvironment = Field(default_factory=dict)

    _validate_options = field_validator("backend_options")(validate_sandbox_options)


def _read_task(path: Path) -> tuple[bytes, os.stat_result] | None:
    """Read a bounded private record without following file or parent aliases."""
    with open_selected_parent(path) as parent:
        try:
            descriptor = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent
            )
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as stream:
            status = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o600
            ):
                raise ValueError("The saved verification task must be a private regular file.")
            body = stream.read(_MAX_TASK_BYTES + 1)
        if len(body) > _MAX_TASK_BYTES:
            raise ValueError("The saved verification task exceeds the storage limit.")
    return body, status


class VerificationCommands:
    """Manage verification tasks in one protected workspace state directory.

    Args:
        data (Path): Application-owned workspace data directory.
        tools (ToolRegistry): Active tool registry, including the normal native command tool.
        instructions (InstructionsManager): Active workspace paths used by the command tool.
    """

    _data: Path
    _tools: ToolRegistry
    _instructions: InstructionsManager

    def __init__(self, data: Path, tools: ToolRegistry, instructions: InstructionsManager) -> None:
        self._data = data
        self._tools = tools
        self._instructions = instructions

    def _names(self) -> list[str]:
        """List syntactically valid owner task names in this workspace."""
        return sorted(
            name
            for entry in self._data.glob(f"{_PREFIX}*.json")
            if _NAME.fullmatch(name := entry.stem.removeprefix(_PREFIX))
        )

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the verification command registration.

        Returns:
            tuple[CommandRegistration, ...]: The ``/verify`` command.
        """
        return (
            CommandRegistration(
                self.verify,
                name="verify",
                description="Manage and run saved workspace verification tasks.",
                capabilities=frozenset({CommandCapability.SESSION_IDLE}),
                completion=CommandCompletion(
                    values=(
                        CompletionValue("list", "List saved workspace tasks."),
                        CompletionValue("show", "Show a saved task and its requested scopes."),
                        CompletionValue(
                            "save", "Save a quoted shell command or a scoped task definition."
                        ),
                        CompletionValue("remove", "Remove a task from workspace state."),
                        CompletionValue("run", "Run a saved task under normal sandbox approval."),
                    ),
                ),
            ),
        )

    def verify(
        self,
        context: CommandContext,
        action: Literal["list", "show", "save", "remove", "run"] = "list",
        name: str | None = None,
        document: str | None = None,
        wait_seconds: float | None = None,
    ) -> None:
        """Manage a workspace task or run its current definition in the native sandbox.

        Args:
            context (CommandContext): Active user interaction.
            action (Literal["list", "show", "save", "remove", "run"]): Requested task operation.
            name (str | None): Exact saved task name, required except for listing.
            document (str | None): Shell command or JSON task object, required only when saving.
            wait_seconds (float | None): Initial command wait in seconds, for a run only.

        Raises:
            CommandArgumentError: The action arguments or saved task are invalid or unavailable.
        """
        interaction = context.interaction
        if (
            (document is not None) != (action == "save")
            or (wait_seconds is not None and action != "run")
            or (name is None) != (action == "list")
        ):
            raise CommandArgumentError(
                "Use /verify list, /verify save NAME 'command', /verify show NAME, "
                "/verify remove NAME, or /verify run NAME [wait_seconds=SECONDS]."
            )
        if name is None:
            names = self._names()
            interaction.info("\n".join(names) if names else "No saved verification tasks.")
            return
        if action == "run":
            arguments, scope = self._select(name)
            arguments["wait_seconds"] = wait_seconds
            result = self._tools.command(
                "run_command",
                tuple(f"{key}={json.dumps(value)}" for key, value in arguments.items()),
                interaction=interaction,
                instructions_manager=self._instructions,
                verification_scope=scope,
            )
            interaction.tool_result(result.output, result.presentation)
            return
        path = self._path(name)
        try:
            with (
                private_directory_lock(self._data, ".verification.lock")
                if action in {"save", "remove"}
                else nullcontext()
            ):
                record = _read_task(path)
                if document is not None:
                    self._save(path, document, record)
                    telemetry_audit("verification.task_saved", task=name)
                    interaction.info(f"Saved verification task {name!r}.")
                    return
                if record is None:
                    raise CommandArgumentError(f"No verification task named {name!r}.")
                if action == "remove":
                    with open_selected_parent(path) as parent:
                        os.unlink(path.name, dir_fd=parent)
                    telemetry_audit("verification.task_removed", task=name)
                    interaction.info(f"Removed verification task {name!r}.")
                    return
                task = _VerificationTask.model_validate_json(record[0])
        except CommandArgumentError:
            raise
        except BlockingIOError as error:
            raise CommandArgumentError(
                "Another client is updating saved verification tasks. Try again."
            ) from error
        except ValidationError as error:
            raise CommandArgumentError(
                "The verification task must contain a command and valid optional scopes."
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise CommandArgumentError(
                "The verification task storage is unavailable or unsafe."
            ) from error
        interaction.info(task.model_dump_json(indent=2))

    def _path(self, name: str) -> Path:
        """Resolve a bounded workspace-local task name."""
        if _NAME.fullmatch(name) is None:
            raise CommandArgumentError("Use 1 to 64 lowercase letters, digits, _ or -.")
        return self._data / f"{_PREFIX}{name}.json"

    def _select(self, name: str) -> tuple[dict[str, object], VerificationScope]:
        """Bind command arguments and after-approval revalidation to one private record."""
        path = self._path(name)
        try:
            record = _read_task(path)
            if record is None:
                raise CommandArgumentError(f"No verification task named {name!r}.")
            task = _VerificationTask.model_validate_json(record[0])
        except CommandArgumentError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise CommandArgumentError(
                "The verification task storage is unavailable or unsafe."
            ) from error

        def admission_refusal() -> str | None:
            """Refuse a changed, removed or unreachable definition after approval."""
            refusal = (
                f"The saved task changed before launch. Review it with '/verify show {name}' "
                "before requesting another run."
            )
            try:
                selected = _read_task(path)
            except (OSError, RuntimeError, ValueError):
                return refusal
            if selected is None or selected[0] != record[0]:
                return refusal
            before, after = record[1], selected[1]
            if (before.st_dev, before.st_ino, before.st_ctime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_ctime_ns,
            ):
                return refusal
            return None

        return task.model_dump(exclude={"environment", "backend_options"}), VerificationScope(
            task.environment, admission_refusal, task.backend_options
        )

    @staticmethod
    def _save(path: Path, document: str, record: tuple[bytes, os.stat_result] | None) -> None:
        """Validate and atomically publish a private task definition."""
        task = (
            _VerificationTask.model_validate_json(document)
            if re.match(r'\s*\{\s*["}]', document)
            else _VerificationTask(command=document)
        )
        body = task.model_dump_json(indent=2) + "\n"
        if len(body.encode()) > _MAX_TASK_BYTES:
            raise CommandArgumentError("The verification task exceeds the storage limit.")
        status = record[1] if record is not None else None
        write_text_atomically(
            path,
            body,
            expected_digest=sha256_digest(record[0]) if record is not None else None,
            expected_device=status.st_dev if status is not None else None,
            expected_inode=status.st_ino if status is not None else None,
            expected_ctime_ns=status.st_ctime_ns if status is not None else None,
        )
