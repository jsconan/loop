"""Tests for tool registration and routing."""

import asyncio
import contextvars
import importlib
import inspect
import io
import json
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Event, Lock, Thread, get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from loop import (
    Action,
    ApprovalChoice,
    ContextReference,
    Decision,
    FileTarget,
    Message,
    NetworkTarget,
    Operation,
    OperationPlan,
    PermissionConfiguration,
    PermissionManager,
    PolicyLimits,
    Problem,
    ProblemException,
    ProcessBoundary,
    ProcessTarget,
    RuntimeEnvironment,
    SessionTarget,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultPresentation,
    ToolResultPresentationSpec,
    delete_path,
    read_text_file,
    resolve_executable,
    run_command,
    write_text_file,
)
from loop.constants import MAX_TOOL_RESULT_BYTES
from loop.execution import CommandExecutionService
from loop.execution.coordinator import LocalHostCommandExecutor
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, SandboxRequest
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.instructions import InstructionsManager
from loop.interaction import ConsoleInteraction, Interaction
from loop.permissions import CommandAnalysis, PermissionRule, PolicyScope
from loop.permissions.protection import protected_workspace_paths
from loop.tooling import (
    FileObservations,
    OwnedToolTask,
    ToolContext,
    ToolPreflightResult,
    ToolRegistration,
    ToolRegistrationError,
    ToolRegistry,
    ToolRegistryView,
    ToolRuntimeSettings,
    ToolStatus,
)
from loop.tooling import tool as declare_tool
from loop.tooling.models import ToolExecutionNotStarted, ToolExecutionOutcomeUnknown
from loop.tooling.tool_registry import ToolExecutionPortal
from loop.utils import VirtualPath, cached_path, content_digest
from tests.scheduling import deliver_callbacks

tool_registry_module = importlib.import_module("loop.tooling.tool_registry")


def _approving_interaction():
    """Provide explicit native approval for path-boundary test requests."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.side_effect = lambda message, **kwargs: (
        "deny" if "without the OS sandbox" in message else "approve"
    )
    return interaction


@pytest.mark.parametrize("asynchronous", [False, True])
def test_yolo_admits_exact_registered_host_tool_without_fresh_input(tmp_path, asynchronous):
    """A registered explicit host tool passes its fresh-input gate only in YOLO mode."""
    target = ProcessTarget(
        argv=("/bin/sh", "-c", "printf host"),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )

    def plan(arguments):
        """Bind the declared host tool to its exact shell effect."""
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(tool_id="host_extension", action=Action.PROCESS_EXECUTE, target=target),
            ),
        )

    @declare_tool(
        actions={Action.PROCESS_EXECUTE},
        operation_planner=plan,
        requires_execution_service=True,
        requires_fresh_host_approval=True,
    )
    def host_extension(context: ToolContext) -> str:
        """Report admission without starting a real host process."""
        return "admitted"

    interaction = Mock(spec=Interaction, supports_fresh_host_approval=False)
    manager = PermissionManager(tmp_path, yolo=True)
    registry = ToolRegistry(
        [host_extension],
        interaction=interaction,
        permission_manager=manager,
        execution_service=Mock(),
    )
    output = (
        asyncio.run(registry.call_async("host_extension", "{}"))
        if asynchronous
        else registry.call("host_extension", "{}")
    )
    assert result_value(output) == "admitted"
    interaction.prompt.assert_not_called()
    manager.set_yolo(False)
    denied = (
        asyncio.run(registry.call_async("host_extension", "{}"))
        if asynchronous
        else registry.call("host_extension", "{}")
    )
    assert json.loads(denied)["problem"]["code"] == "tool.denied"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "shape",
    ["empty", "wrong_action", "sandbox", "short", "shell"],
)
def test_explicit_host_tool_rejects_unbound_operation_plans(tmp_path, asynchronous, shape):
    """A host-marked extension cannot bypass exact registered shell-plan validation."""
    target = ProcessTarget(
        argv=("/bin/sh", "-c", "printf host"),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )
    if shape == "wrong_action":
        target = FileTarget(path=str(tmp_path / "file"))
    elif shape == "sandbox":
        target = target.model_copy(update={"boundary": ProcessBoundary.SANDBOXED})
    elif shape == "short":
        target = target.model_copy(update={"argv": ("/bin/sh", "-c")})
    elif shape == "shell":
        target = target.model_copy(update={"argv": ("/bin/echo", "-c", "printf host")})
    planned = (
        ()
        if shape == "empty"
        else (
            Operation(
                tool_id="host_extension",
                action=Action.FILESYSTEM_READ
                if shape == "wrong_action"
                else Action.PROCESS_EXECUTE,
                target=target,
            ),
        )
    )

    def plan(arguments):
        """Return the selected malformed host effect."""
        return OperationPlan(arguments=arguments, operations=planned)

    @declare_tool(
        actions={Action.PROCESS_EXECUTE, Action.FILESYSTEM_READ},
        operation_planner=plan,
        requires_execution_service=True,
        requires_fresh_host_approval=True,
    )
    def host_extension(context: ToolContext) -> str:
        """Expose an inert extension behind host-plan authorization."""
        return "unreachable"

    registry = ToolRegistry(
        [host_extension],
        permission_manager=PermissionManager(tmp_path),
        execution_service=Mock(),
    )
    output = (
        asyncio.run(registry.call_async("host_extension", "{}"))
        if asynchronous
        else registry.call("host_extension", "{}")
    )

    assert json.loads(output)["problem"]["code"] == "host_command.invalid_request"


def result_value(output: str):
    """Return the successful value from a tool result envelope."""
    payload = json.loads(output)
    assert payload["ok"] is True
    return payload["result"]


def _permission_manager_mock():
    """Provide a permission mock that allows asynchronous boundary preflight by default."""
    permissions = Mock(spec=PermissionManager)
    permissions.temporary_directory = Path(tempfile.gettempdir())
    permissions.check_boundaries.return_value = None
    permissions.check_boundaries_async.return_value = None
    permissions.sandbox_scope_limit.return_value = None
    permissions.sandbox_write_root_limit.return_value = None
    permissions.sandbox_network_limit.return_value = None
    permissions.temporary_directory_allows_private_names = False
    permissions.standalone_execution.side_effect = lambda: nullcontext(
        permissions.temporary_directory
    )
    return permissions


def _use_managed_command_scratch(registry, instructions):
    """Give standalone command tests matching leased and model-visible scratch roots."""
    roots = dict(instructions.virtual_paths.command_roots())
    permissions = registry.permission_manager
    if isinstance(permissions, Mock):
        permissions.temporary_directory = roots[VirtualPath.TEMPORARY]
    scratch = permissions.temporary_directory
    instructions.set_runtime_environment(RuntimeEnvironment(roots[VirtualPath.WORKSPACE], scratch))
    return scratch


def _approve_sandbox_request(_request, **kwargs):
    """Give a mocked sandbox approval the revision required at admission."""
    kwargs["approved_revision"].append(1)
    return True


@declare_tool()
def passive_tool() -> str:
    """Return a passive result for sandbox-routing coverage."""
    return "ok"


def planner_for(action: Action):
    """Return a concrete operation planner for an authority-bearing test tool."""

    def plan(arguments):
        target = (
            NetworkTarget(url="https://my-host.local", origin="https://my-host.local")
            if action is Action.NETWORK_REQUEST
            else SessionTarget(identifier="test-state")
        )
        return OperationPlan(
            arguments=arguments,
            operations=(Operation(tool_id="", action=action, target=target),),
        )

    return plan


def register(registry: ToolRegistry):
    """Register and return a simple documented function."""

    @declare_tool
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)
    return calculate


def test_registry_closes_its_execution_service():
    """Registry teardown releases the bound command execution service once."""
    service = Mock()
    registry = ToolRegistry(execution_service=service)
    registry.close()
    registry.close()
    service.close.assert_called_once()
    ToolRegistry().close()


@pytest.mark.parametrize("live_state", ["active", "retained"])
def test_registry_rebind_waits_for_active_commands_but_allows_retained_handles(live_state):
    """Active jobs keep their session binding; completed retained handles do not block it."""
    first, second = Mock(), Mock()
    service = Mock()
    service.has_active_jobs.return_value = live_state == "active"

    @declare_tool
    def selected(context: ToolContext) -> str:
        """Report which synthetic session was captured by this tool invocation."""
        return "first" if context.session_manager is first else "second"

    registry = ToolRegistry([selected], execution_service=service, session_manager=first)
    try:
        registry.bind_session_manager(first)
        service.has_active_jobs.assert_not_called()
        if live_state == "active":
            with pytest.raises(RuntimeError, match="commands are active"):
                registry.bind_session_manager(second)
            assert result_value(registry.call("selected", "{}")) == "first"
        service.has_active_jobs.return_value = False
        registry.bind_session_manager(second)
        assert result_value(registry.call("selected", "{}")) == "second"
    finally:
        registry.close()


@pytest.mark.parametrize("same_session", [False, True])
def test_session_composition_refuses_active_jobs_before_mutations(same_session):
    """Composition rejects active jobs even when the requested session is unchanged."""
    first, second = Mock(), Mock()
    service = Mock()
    service.has_active_jobs.return_value = True
    registry = ToolRegistry(execution_service=service, session_manager=first)
    effects = []
    try:
        with (
            pytest.raises(RuntimeError, match="commands are active"),
            registry.compose_session(first if same_session else second),
        ):
            effects.append("mutated")
        assert effects == []
    finally:
        registry.close()


@pytest.mark.parametrize("fails", [False, True])
def test_session_composition_binds_only_after_success(fails):
    """A failed composition keeps the prior session; success selects the new one."""
    first, second = Mock(), Mock()

    @declare_tool
    def selected(context: ToolContext) -> str:
        """Report the bound synthetic session."""
        return "first" if context.session_manager is first else "second"

    registry = ToolRegistry([selected], session_manager=first)
    try:
        if fails:
            with (
                pytest.raises(ValueError, match="composition failed"),
                registry.compose_session(second),
            ):
                raise ValueError("composition failed")
        else:
            with registry.compose_session(second):
                pass
        assert result_value(registry.call("selected", "{}")) == ("first" if fails else "second")
    finally:
        registry.close()


def test_session_composition_excludes_stale_command_admission():
    """A command waiting on composition cannot start under the previously selected session."""
    first, second = Mock(), Mock()
    prepared, attempt, release = Event(), Event(), Event()
    effects = []

    @declare_tool(requires_execution_service=True)
    def command_probe(context: ToolContext) -> str:
        """Try to reserve a captured session after composition begins."""
        prepared.set()
        if not release.wait(5):
            raise TimeoutError("Composition did not release the test command.")
        attempt.set()
        try:
            with context.command_admission():
                effects.append("started")
        except ValueError:
            return "stale"
        return "admitted"

    service = Mock()
    service.has_active_jobs.return_value = False
    registry = ToolRegistry([command_probe], execution_service=service, session_manager=first)
    results = []
    thread = Thread(target=lambda: results.append(registry.call("command_probe", "{}")))
    try:
        thread.start()
        assert prepared.wait(5)
        with registry.compose_session(second):
            release.set()
            assert attempt.wait(5)
            assert effects == []
        thread.join(5)
        assert not thread.is_alive()
        assert result_value(results[0]) == "stale"
        assert effects == []
    finally:
        release.set()
        thread.join(5)
        registry.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_command_admission_rejects_a_session_rebound_after_context_creation(asynchronous):
    """A stale command context cannot reserve work after the registry changes sessions."""
    first, second = Mock(), Mock()
    service = Mock()
    service.has_active_jobs.return_value = False
    effects = []

    @declare_tool(requires_execution_service=True)
    def command_probe(context: ToolContext) -> str:
        """Attempt reservation after deliberately changing the selected session."""
        registry.bind_session_manager(second)
        try:
            with context.command_admission():
                effects.append("reserved")
        except ValueError as error:
            return str(error)
        return "admitted"

    registry = ToolRegistry([command_probe], execution_service=service, session_manager=first)
    try:
        output = (
            asyncio.run(registry.call_async("command_probe", "{}"))
            if asynchronous
            else registry.call("command_probe", "{}")
        )
        assert result_value(output) == "Command session changed before admission."
        assert effects == []
    finally:
        registry.close()


def test_closed_registry_rejects_new_sync_worker_calls():
    """A closed registry reports worker capacity as unavailable without invoking tools."""
    invoked = []

    @declare_tool
    def calculate() -> int:
        """Record whether a closed registry invokes the handler."""
        invoked.append(True)
        return 1

    registry = ToolRegistry([calculate])
    registry.close()

    output = asyncio.run(registry.call_async("calculate", "{}"))

    assert json.loads(output)["problem"]["code"] == "tool.dispatch_busy"
    assert invoked == []


def test_closed_registry_rejects_async_path_redaction_before_tool_execution(tmp_path):
    """Unavailable redaction capacity blocks async dispatch before invoking the tool."""
    invoked = []

    @declare_tool
    def calculate() -> int:
        """Record whether a closed registry invokes the handler."""
        invoked.append(True)
        return 1

    registry = ToolRegistry([calculate])
    registry.close()
    instructions = SimpleNamespace(virtual_paths=VirtualPath(tmp_path))

    output = asyncio.run(registry.call_async("calculate", "{}", instructions_manager=instructions))

    assert json.loads(output)["problem"]["code"] == "tool.dispatch_busy"
    assert invoked == []


def test_failed_mutations_do_not_publish_committed_file_versions(tmp_path):
    """An unsuccessful tool result cannot refresh the model's file observations."""

    @declare_tool()
    def failed_write(context: ToolContext) -> Problem:
        """Report a failed mutation after collecting a provisional content version."""
        context.record_file_version(tmp_path / "file.txt", "provisional")
        return Problem(code="filesystem.write_failed", title="Write failed", detail="not committed")

    output = json.loads(ToolRegistry([failed_write]).call("failed_write", "{}"))
    assert output["ok"] is False
    assert "file_versions" not in output


@pytest.mark.parametrize(
    "code,published", [("filesystem.path_missing", True), ("filesystem.read_failed", False)]
)
def test_read_receipts_attest_only_to_success_or_explicit_absence(tmp_path, code, published):
    """Failed reads cannot refresh file state except for a verified missing source."""

    @declare_tool()
    def failed_read(context: ToolContext) -> Problem:
        """Report a missing source or a failure after attempting a read."""
        context.record_file_read(tmp_path / "file.txt", None)
        return Problem(code=code, title="Could not read file", detail="unavailable")

    output = json.loads(ToolRegistry([failed_read]).call("failed_read", "{}"))
    assert output["ok"] is False
    assert ("read_versions" in output) is published
    if published:
        assert output["read_versions"] == {str(tmp_path / "file.txt"): None}


@pytest.mark.parametrize("deleted", [False, True])
def test_async_dispatch_rejects_stale_file_context_before_approval(tmp_path, deleted):
    """Asynchronous dispatch uses the same source observation gate as synchronous agent tools."""
    target = tmp_path / "file.txt"
    if not deleted:
        target.write_text("user edit", encoding="utf-8")
    observations = FileObservations(
        [
            Message(
                role="user",
                content="edit",
                context=(
                    ContextReference(
                        kind="file",
                        path="file.txt",
                        content="old",
                        size_bytes=3,
                        included_bytes=3,
                        truncated=False,
                        version=content_digest("old"),
                    ),
                ),
            )
        ],
        VirtualPath(tmp_path),
    )
    permissions = PermissionManager(tmp_path)
    permissions.request_permission = Mock()
    registry = ToolRegistry([write_text_file], permission_manager=permissions)
    output = json.loads(
        asyncio.run(
            registry.call_async(
                "write_text_file",
                json.dumps({"path": str(target), "content": "model edit"}),
                file_observations=observations,
            )
        )
    )

    assert output["problem"]["code"] == "filesystem.stale_context"
    if deleted:
        assert output["read_versions"] == {"$WORKSPACE_ROOT/file.txt": None}
        assert not target.exists()
    else:
        assert "read_versions" not in output
        assert target.read_text(encoding="utf-8") == "user edit"
    permissions.request_permission.assert_not_called()


def test_async_dispatch_checks_file_observations_in_a_worker(tmp_path, monkeypatch):
    """Async planning runs file observation checks outside the event-loop thread."""
    owner_thread = get_ident()
    check_threads = []
    original_check = FileObservations.check

    def track_check(observations, *args, **kwargs):
        check_threads.append(get_ident())
        return original_check(observations, *args, **kwargs)

    monkeypatch.setattr(FileObservations, "check", track_check)

    @declare_tool
    def calculate() -> str:
        """Return the calculation result."""
        return "done"

    registry = ToolRegistry([calculate])
    try:
        output = asyncio.run(
            registry.call_async(
                "calculate",
                "{}",
                file_observations=FileObservations((), VirtualPath(tmp_path)),
            )
        )
    finally:
        registry.close()

    assert result_value(output) == "done"
    assert len(check_threads) == 1
    assert check_threads[0] != owner_thread


def test_async_dispatch_redacts_canonical_paths_in_a_worker(tmp_path, monkeypatch):
    """Async result redaction resolves virtual roots before tool execution on a worker."""
    paths = VirtualPath(tmp_path)
    instructions = SimpleNamespace(virtual_paths=paths, agents_filenames=())
    owner_thread = get_ident()
    resolution_threads = []
    original_resolve = Path.resolve

    def track_resolve(path, strict=False):
        resolution_threads.append(get_ident())
        return original_resolve(path, strict=strict)

    @declare_tool
    def failed() -> Problem:
        """Return a local path in a diagnostic that requires virtual rendering."""
        return Problem(
            code="tool.failed",
            title="Tool failed",
            detail=f"Failed at {tmp_path}/source.txt",
        )

    registry = ToolRegistry([failed])
    monkeypatch.setattr(Path, "resolve", track_resolve)
    try:
        output = asyncio.run(registry.call_async("failed", "{}", instructions_manager=instructions))
    finally:
        registry.close()

    payload = json.loads(output)
    assert payload["problem"]["detail"] == "Failed at $WORKSPACE_ROOT/source.txt"
    assert resolution_threads
    assert owner_thread not in resolution_threads


@pytest.mark.parametrize("denial", ["read", "boundary", "creation"])
def test_deletion_receipts_preserve_inspection_and_creation_authorization(tmp_path, denial):
    """Absence receipts require authorized inspection and never grant creation or boundary access."""
    relative = ".git/file.txt" if denial == "boundary" else "file.txt"
    target = tmp_path / relative
    original = Message(
        role="user",
        content="edit",
        context=(
            ContextReference(
                kind="file",
                path=relative,
                content="old",
                size_bytes=3,
                included_bytes=3,
                truncated=False,
                version=content_digest("old"),
            ),
        ),
    )
    paths = VirtualPath(tmp_path)
    observations = FileObservations([original], paths)
    permissions = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            rules=[
                PermissionRule(
                    decision=Decision.DENY,
                    action=(
                        Action.FILESYSTEM_READ if denial == "read" else Action.FILESYSTEM_CREATE
                    ),
                )
            ]
        ),
    )
    permissions.request_permission = Mock()
    registry = ToolRegistry([write_text_file], permission_manager=permissions)
    call = ToolCall(
        call_id="write",
        name="write_text_file",
        arguments=json.dumps({"path": str(target), "content": "model"}),
    )
    output = registry.call(call.name, call.arguments, file_observations=observations)
    if denial == "creation":
        assert json.loads(output)["read_versions"] == {"$WORKSPACE_ROOT/file.txt": None}
        assert json.loads(output)["problem"]["code"] == "filesystem.stale_context"
        refreshed = FileObservations(
            [original, call, ToolResult(call_id=call.call_id, output=output)], paths
        )
        assert (
            json.loads(registry.call(call.name, call.arguments, file_observations=observations))[
                "problem"
            ]["code"]
            == "filesystem.stale_context"
        )
        output = registry.call(call.name, call.arguments, file_observations=refreshed)
    payload = json.loads(output)
    assert payload["problem"]["code"] == "tool.denied"
    assert "read_versions" not in payload
    assert "file_versions" not in payload
    assert not target.exists()
    permissions.request_permission.assert_not_called()


def test_constructor_registers_in_order_and_exposes_sorted_snapshots():
    """Construction preserves definitions while public tool snapshots sort and isolate state."""

    @declare_tool
    def zebra() -> str:
        """Return the zebra result."""
        return "zebra"

    @declare_tool
    def alpha() -> str:
        """Return the alpha result."""
        return "alpha"

    registry = ToolRegistry([zebra, alpha])
    tools = registry.tools
    names = registry.names

    assert [definition.name for definition in registry.definitions()] == [
        "zebra",
        "alpha",
    ]
    assert [tool.name for tool in tools] == ["alpha", "zebra"]
    assert names == ["alpha", "zebra"]
    tools.clear()
    names.clear()
    assert registry.names == ["alpha", "zebra"]
    assert result_value(registry.call("zebra", "{}")) == "zebra"


def test_registration_problems_returns_an_immutable_snapshot():
    """Registration diagnostics are exposed as a tuple independent of internal storage."""

    @declare_tool(preflight=lambda: ToolPreflightResult(ToolStatus.BROKEN, "Unavailable"))
    def unavailable() -> None:
        """Attempt unavailable work."""

    registry = ToolRegistry([unavailable])

    problems = registry.registration_problems

    assert isinstance(problems, tuple)
    assert len(problems) == 1
    assert problems[0].detail == "Unavailable"


def test_register_resolves_a_tool_from_declared_metadata():
    """Registration resolves an immutable runtime tool from a passive declaration."""
    registry = ToolRegistry()

    function = register(registry)
    registered = registry.tools[0]

    assert registered.function is function
    assert registered.name == "calculate"
    assert registered.description == "Calculate a number."
    assert registered.arguments_model is not None
    assert registered.actions == frozenset()


def test_register_accepts_explicit_metadata_and_rejects_duplicate_names():
    """Declaration options supply metadata and duplicate public names remain invalid."""
    registry = ToolRegistry()

    @declare_tool(name="selected", description="Explicit")
    def first(number: int) -> int:
        return number

    registry.register(first)
    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(first)


def test_register_accepts_undeclared_functions_with_derived_defaults():
    """Any documented callable can become a pure tool without prior declaration."""

    def ordinary(value: int) -> int:
        """Return an ordinary value."""
        return value

    registry = ToolRegistry([ordinary])

    assert registry.names == ["ordinary"]
    assert result_value(registry.call("ordinary", '{"value": 3}')) == 3


def test_register_overrides_metadata_for_only_one_container():
    """Registration options customize one registry without changing a reusable declaration."""

    @declare_tool(name="declared", description="Declared")
    def calculate(number: int) -> int:
        return number

    planner = Mock(side_effect=lambda arguments: OperationPlan(arguments=arguments))
    customized = ToolRegistry()
    customized.register(
        calculate,
        name="customized",
        description="Customized",
        actions={Action.NETWORK_REQUEST},
        operation_planner=planner,
    )
    standard = ToolRegistry([calculate])

    custom_tool = customized.tools[0]
    standard_tool = standard.tools[0]
    assert (
        custom_tool.name,
        custom_tool.description,
        custom_tool.actions,
        custom_tool.operation_planner,
    ) == (
        "customized",
        "Customized",
        frozenset({Action.NETWORK_REQUEST}),
        planner,
    )
    assert (
        standard_tool.name,
        standard_tool.description,
        standard_tool.actions,
        standard_tool.operation_planner,
    ) == ("declared", "Declared", frozenset(), None)


def test_register_rejects_authority_bearing_tools_without_a_planner():
    """Every authority-bearing registration must identify concrete targets before execution."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    with pytest.raises(ToolRegistrationError, match="without an operation planner"):
        ToolRegistry().register(calculate, actions={Action.SESSION_MUTATE})


def test_register_can_remove_a_declared_operation_planner():
    """An explicit null planner removes inherited operation planning in one registry."""
    planner = Mock(side_effect=lambda arguments: OperationPlan(arguments=arguments))

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    inherited = ToolRegistry([calculate])
    overridden = ToolRegistry()
    overridden.register(calculate, operation_planner=None)

    assert inherited.tools[0].operation_planner is planner
    assert overridden.tools[0].operation_planner is None


def test_register_accepts_container_specific_registration_records():
    """Registration records apply their local metadata through the public register method."""

    def calculate(number: int) -> int:
        return number

    registry = ToolRegistry()
    registry.register(
        ToolRegistration(
            calculate,
            name="local_calculate",
            description="Calculate locally.",
            actions=frozenset(),
        )
    )

    registered = registry.tools[0]
    assert registered.name == "local_calculate"
    assert registered.description == "Calculate locally."
    assert registered.actions == frozenset()


def test_register_admits_ready_tools_and_returns_registration_status():
    """A successful preflight runs once and admits the tool to every public catalog view."""
    preflight = Mock(return_value=ToolPreflightResult(ToolStatus.READY))

    @declare_tool(preflight=preflight)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry = ToolRegistry()

    assert registry.register(calculate) is True
    assert registry.names == ["calculate"]
    assert [tool.name for tool in registry.tools] == ["calculate"]
    assert [definition.name for definition in registry.definitions()] == ["calculate"]
    preflight.assert_called_once_with()


def test_register_skips_unavailable_optional_tools_and_reports_the_problem():
    """Optional readiness failures stay out of the catalog and notify the active user."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    registry = ToolRegistry(interaction=interaction)

    @declare_tool(
        preflight=lambda: ToolPreflightResult(ToolStatus.BROKEN, "Install the dependency.")
    )
    def unavailable() -> None:
        """Attempt unavailable work."""

    def available() -> None:
        """Perform available work."""

    assert registry.register(unavailable) is False
    assert registry.register(available) is True

    assert registry.names == ["available"]
    interaction.warning.assert_called_once_with("Install the dependency.")


def test_register_logs_headless_broken_tools_with_a_default_detail(caplog):
    """Headless registries log broken optional tools with an actionable default detail."""

    @declare_tool(preflight=lambda: ToolPreflightResult(ToolStatus.BROKEN))
    def unavailable() -> None:
        """Attempt unavailable work."""

    registry = ToolRegistry([unavailable])

    assert registry.names == []
    assert "Tool broken" in caplog.text
    assert "unavailable" not in caplog.text


def test_register_converts_unexpected_preflight_errors_and_logs_them(monkeypatch):
    """Unexpected readiness errors become safe optional diagnostics with private logging."""
    error = RuntimeError("private failure")
    logger = Mock()
    monkeypatch.setattr(tool_registry_module, "_LOGGER", logger)

    @declare_tool(preflight=Mock(side_effect=error))
    def unavailable() -> None:
        """Attempt unavailable work."""

    ToolRegistry([unavailable])

    assert logger.log.call_args.kwargs["extra"]["exception.type"] == "builtins.RuntimeError"
    assert "exc_info" not in logger.log.call_args.kwargs


@pytest.mark.parametrize("raises", [False, True])
def test_register_halts_when_the_user_rejects_a_broken_required_tool(raises):
    """Broken required tools warn the user and halt when the user chooses to halt."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "halt"
    preflight = (
        Mock(side_effect=RuntimeError("broken check"))
        if raises
        else Mock(
            return_value=ToolPreflightResult(ToolStatus.BROKEN, "Required capability is missing.")
        )
    )

    @declare_tool(preflight=preflight, required=True)
    def required() -> None:
        """Perform required work."""

    registry = ToolRegistry(interaction=interaction)

    with pytest.raises(ToolRegistrationError):
        registry.register(required)

    assert registry.names == []
    interaction.warning.assert_called_once()
    interaction.prompt.assert_called_once()
    assert interaction.prompt.call_args.kwargs["index"] == {"halt": "Q", "continue": "C"}


def test_yolo_reports_broken_required_tool_without_choosing_recovery(tmp_path):
    """No-prompt mode treats a missing required tool as unavailable instead of guessing."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, yolo=True)

    @declare_tool(
        preflight=lambda: ToolPreflightResult(ToolStatus.BROKEN, "Required dependency missing."),
        required=True,
    )
    def required() -> None:
        """Require the missing dependency."""

    registry = ToolRegistry(interaction=interaction, permission_manager=manager)
    with pytest.raises(ToolRegistrationError, match="Required dependency missing"):
        registry.register(required)
    interaction.prompt.assert_not_called()


def test_register_continues_without_a_broken_required_tool_when_selected():
    """The explicit continue choice skips a broken required tool without halting."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "continue"

    @declare_tool(
        preflight=lambda: ToolPreflightResult(ToolStatus.BROKEN, "Dependency missing."),
        required=True,
    )
    def required() -> None:
        """Perform required work."""

    registry = ToolRegistry(interaction=interaction)

    assert registry.register(required) is False
    assert registry.names == []
    assert interaction.prompt.call_args.kwargs["index"] == {"halt": "Q", "continue": "C"}


def test_register_admits_degraded_tools_after_warning():
    """Degraded tools remain available after their warning is logged and displayed."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)

    @declare_tool(preflight=lambda: ToolPreflightResult(ToolStatus.DEGRADED, "Limited capability."))
    def degraded() -> None:
        """Perform limited work."""

    registry = ToolRegistry(interaction=interaction)

    assert registry.register(degraded) is True
    assert registry.names == ["degraded"]
    interaction.warning.assert_called_once_with("Limited capability.")


def test_registration_overrides_can_replace_and_remove_declared_readiness():
    """Container registrations can replace checks and requiredness or remove a declared check."""
    declared_check = Mock(return_value=ToolPreflightResult(ToolStatus.BROKEN, "not used"))
    replacement = Mock(return_value=ToolPreflightResult(ToolStatus.READY))

    @declare_tool(preflight=declared_check, required=True)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    replaced = ToolRegistry()
    assert (
        replaced.register(ToolRegistration(calculate, preflight=replacement, required=False))
        is True
    )
    removed = ToolRegistry()
    assert removed.register(calculate, preflight=None, required=False) is True

    declared_check.assert_not_called()
    replacement.assert_called_once_with()


def test_independent_registries_allow_the_same_name_and_isolate_runtime_state():
    """Separate containers can reuse names while retaining their own interaction and policy."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    first_interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    second_interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    first_permissions = _permission_manager_mock()
    second_permissions = _permission_manager_mock()
    first = ToolRegistry(
        [calculate], interaction=first_interaction, permission_manager=first_permissions
    )
    second = ToolRegistry(
        [calculate], interaction=second_interaction, permission_manager=second_permissions
    )

    assert first.names == second.names == ["calculate"]
    assert first.interaction is first_interaction
    assert second.interaction is second_interaction
    assert first.permission_manager is first_permissions
    assert second.permission_manager is second_permissions


def test_registries_compose_different_subsets_of_reusable_functions():
    """Each container selects its own subset from the same available callables."""

    def first() -> str:
        """Return the first result."""
        return "first"

    def second() -> str:
        """Return the second result."""
        return "second"

    assert ToolRegistry([first]).names == ["first"]
    assert ToolRegistry([second]).names == ["second"]


def test_register_accepts_callable_objects():
    """Callable instances derive schemas, descriptions, names, and invocation behavior."""

    class Multiplier:
        """Multiply a number."""

        def __call__(self, number: int) -> int:
            return number * 2

    registry = ToolRegistry([Multiplier()])

    assert registry.names == ["Multiplier"]
    assert result_value(registry.call("Multiplier", '{"number": 3}')) == 6


def test_register_accepts_partials_with_explicit_local_metadata():
    """Partially bound functions expose only their remaining annotated parameters."""

    def add(first: int, second: int) -> int:
        return first + second

    registry = ToolRegistry()
    registry.register(partial(add, 2), name="add_two", description="Add two to a number.")

    assert result_value(registry.call("add_two", '{"second": 3}')) == 5


def test_sync_and_async_registry_calls_reuse_their_native_async_loops():
    """Async callables run through one owned sync loop or the caller's async loop."""
    loop_ids = []

    class AsyncMultiplier:
        """Multiply a number asynchronously."""

        async def __call__(self, number: int) -> int:
            loop_ids.append(id(asyncio.get_running_loop()))
            return number * 2

    registry = ToolRegistry([AsyncMultiplier()])

    try:
        assert result_value(registry.call("AsyncMultiplier", '{"number": 3}')) == 6
        assert result_value(registry.call("AsyncMultiplier", '{"number": 4}')) == 8
        assert loop_ids[0] == loop_ids[1]
        assert (
            result_value(asyncio.run(registry.call_async("AsyncMultiplier", '{"number": 5}'))) == 10
        )
        assert loop_ids[2] != loop_ids[0]
    finally:
        registry.close()


def test_sync_runner_is_created_once_for_concurrent_callers(monkeypatch, contended_lock):
    """Concurrent sync calls use one runner and reject the non-owner safely."""

    @declare_tool
    async def calculate() -> str:
        """Return a result from the registry-owned async runner."""
        return "done"

    registry = ToolRegistry([calculate])
    runner_creation_started = Event()
    second_call_started = Event()
    release_runner_creation = Event()
    monkeypatch.setattr(registry, "_runner_lock", contended_lock.lock)
    created_runners = []
    created_runners_lock = Lock()
    results = {}
    real_runner = asyncio.Runner
    second_thread: Thread | None = None

    def create_runner(**kwargs):
        """Hold the first runner creation to expose an unsynchronized check-then-act race."""
        with created_runners_lock:
            runner = real_runner(**kwargs)
            created_runners.append(runner)
            first_runner = len(created_runners) == 1
        if first_runner:
            runner_creation_started.set()
            assert release_runner_creation.wait(timeout=3)
        return runner

    monkeypatch.setattr(tool_registry_module.asyncio, "Runner", create_runner)

    def call_tool(key: str):
        """Record the public synchronous tool result from one caller thread."""
        if key == "second":
            second_call_started.set()
        results[key] = registry.call("calculate", "{}")

    first_thread = Thread(target=call_tool, args=("first",))
    try:
        first_thread.start()
        assert runner_creation_started.wait(timeout=3)
        second_thread = Thread(target=call_tool, args=("second",))
        second_thread.start()
        assert second_call_started.wait(timeout=3)
        assert contended_lock.contended.wait(timeout=3)
        assert len(created_runners) == 1
    finally:
        release_runner_creation.set()
        first_thread.join(timeout=3)
        if second_thread is not None:
            second_thread.join(timeout=3)
        registry.close()

    assert not first_thread.is_alive()
    assert second_thread is not None and not second_thread.is_alive()
    assert len(created_runners) == 1
    assert result_value(results["first"]) == "done"
    assert json.loads(results["second"])["problem"]["code"] == "tool.async_required"
    assert json.loads(registry.call("calculate", "{}"))["problem"]["code"] == (
        "tool.async_required"
    )
    assert len(created_runners) == 1


def test_sync_dispatch_rejects_async_tools_inside_a_running_loop():
    """A sync registry call in an active loop rejects before invoking an async tool."""
    invoked = []

    @declare_tool
    async def calculate(number: int) -> int:
        """Record whether synchronous dispatch reached the async callable."""
        invoked.append(number)
        return number * 2

    registry = ToolRegistry([calculate])

    async def dispatch():
        """Verify the owned runner and sync hub both reject the active loop."""
        assert registry._sync_runner() is None
        with pytest.raises(ProblemException) as captured:
            registry._invoke(registry._tools["calculate"], calculate, {"number": 3})
        assert captured.value.problem.code == "tool.async_required"
        return registry.call("calculate", '{"number": 3}')

    output = asyncio.run(dispatch())

    assert json.loads(output)["problem"]["code"] == "tool.async_required"
    assert not invoked
    registry.close()


def test_sync_dispatch_rejects_sync_tools_inside_a_running_loop():
    """Sync tools are never invoked by the synchronous hub on an event-loop thread."""
    invoked = []

    @declare_tool
    def calculate(number: int) -> int:
        """Record whether synchronous dispatch reached the tool."""
        invoked.append(number)
        return number * 2

    registry = ToolRegistry([calculate])

    async def dispatch():
        """Require the async registry API while an event loop is running."""
        return registry.call("calculate", '{"number": 3}')

    output = asyncio.run(dispatch())

    assert json.loads(output)["problem"]["code"] == "tool.async_required"
    assert not invoked
    registry.close()


def test_async_dispatch_runs_sync_handlers_off_loop_and_keeps_async_handlers_on_loop():
    """Sync handlers use workers while native async handlers remain on the owner loop."""
    loop_thread = get_ident()
    async_threads = []

    @declare_tool
    def calculate(number: int) -> int:
        """Return a value from a synchronous handler."""
        return number * 2

    @declare_tool
    async def identify() -> int:
        """Record the thread running a native async handler."""
        async_threads.append(get_ident())
        return 1

    registry = ToolRegistry([calculate, identify])

    async def dispatch():
        sync_result = await registry.call_async("calculate", '{"number": 3}')
        async_result = await registry.call_async("identify", "{}")
        return sync_result, async_result

    try:
        sync_output, async_output = asyncio.run(dispatch())
    finally:
        registry.close()

    assert result_value(sync_output) == 6
    assert result_value(async_output) == 1
    assert async_threads == [loop_thread]


def test_sync_callable_returning_awaitable_is_awaited_by_each_registry_path():
    """A sync callable may return an awaitable for either registry execution style."""

    @declare_tool
    def calculate(number: int) -> int:
        """Return an awaitable value from a synchronous callable."""

        async def result() -> int:
            return number * 2

        return result()

    registry = ToolRegistry([calculate])

    try:
        asynchronous = asyncio.run(registry.call_async("calculate", '{"number": 3}'))
        synchronous = registry.call("calculate", '{"number": 4}')
    finally:
        registry.close()

    assert result_value(asynchronous) == 6
    assert result_value(synchronous) == 8


def test_async_planners_and_continuations_use_registry_callable_dispatch():
    """Sync planning callbacks run in workers before async handlers run on the loop."""
    loop_thread = get_ident()
    planner_threads = []

    def continuation():
        planner_threads.append(get_ident())
        return OperationPlan(arguments={"number": 9})

    def planner(arguments):
        planner_threads.append(get_ident())
        return OperationPlan(arguments=arguments, continuation=continuation)

    @declare_tool(operation_planner=planner)
    async def calculate(number: int) -> int:
        """Return the value produced by the planning continuation."""
        assert get_ident() == loop_thread
        return number

    registry = ToolRegistry([calculate])

    try:
        output = asyncio.run(registry.call_async("calculate", '{"number": 1}'))
    finally:
        registry.close()

    assert result_value(output) == 9
    assert len(planner_threads) == 2
    assert all(thread_id != loop_thread for thread_id in planner_threads)


def test_async_planners_and_continuations_are_awaited_on_the_owner_loop():
    """Native async planning callbacks stay on and yield through the owner loop."""
    owner_thread = get_ident()
    planner_threads = []

    async def continuation():
        planner_threads.append(get_ident())
        return OperationPlan(arguments={"number": 7})

    async def planner(arguments):
        planner_threads.append(get_ident())
        return OperationPlan(arguments=arguments, continuation=continuation)

    @declare_tool(operation_planner=planner)
    async def calculate(number: int) -> int:
        """Return the value produced by the asynchronous planning continuation."""
        assert get_ident() == owner_thread
        return number

    registry = ToolRegistry([calculate])

    try:
        output = asyncio.run(registry.call_async("calculate", '{"number": 1}'))
    finally:
        registry.close()

    assert result_value(output) == 7
    assert planner_threads == [owner_thread, owner_thread]


def test_blocking_sync_tool_does_not_stall_the_async_event_loop():
    """The async loop processes owner callbacks while a synchronous tool is blocked."""
    started = Event()
    release = Event()

    @declare_tool
    def wait_for_release() -> str:
        """Wait for the test owner to release the worker."""
        started.set()
        assert release.wait(timeout=5), "test synchronization was not released"
        return "finished"

    registry = ToolRegistry([wait_for_release])

    async def dispatch():
        task = asyncio.create_task(registry.call_async("wait_for_release", "{}"))
        await asyncio.to_thread(started.wait)
        loop_callback = asyncio.Event()
        asyncio.get_running_loop().call_soon(loop_callback.set)
        await loop_callback.wait()
        release.set()
        return await task

    try:
        output = asyncio.run(dispatch())
    finally:
        release.set()
        registry.close()

    assert result_value(output) == "finished"


def test_async_tool_worker_admission_is_bounded():
    """Synchronous tool submissions beyond worker and queue capacity return a typed error."""
    release = Event()
    invocation_count = []

    @declare_tool
    def wait_for_release() -> str:
        """Hold a worker until the registry reports its capacity boundary."""
        invocation_count.append(True)
        assert release.wait(timeout=5), "test synchronization was not released"
        return "finished"

    registry = ToolRegistry([wait_for_release])

    async def dispatch():
        tasks = [
            asyncio.create_task(registry.call_async("wait_for_release", "{}")) for _ in range(9)
        ]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        first = json.loads(next(iter(done)).result())
        release.set()
        outputs = await asyncio.gather(*tasks)
        return first, [json.loads(output) for output in outputs]

    try:
        first, outputs = asyncio.run(dispatch())
    finally:
        release.set()
        registry.close()

    assert first["problem"]["code"] == "tool.dispatch_busy"
    assert sum(result.get("ok") is True for result in outputs) == 8
    assert (
        sum(result.get("problem", {}).get("code") == "tool.dispatch_busy" for result in outputs)
        == 1
    )
    assert len(invocation_count) == 8


@pytest.mark.parametrize("callback", [False, True])
def test_cancelled_queued_worker_never_enters_callable(callback, monkeypatch):
    """Canceling a queued callback or tool never starts it after capacity frees."""

    async def scenario():
        """Occupy all workers, queue one invocation, then cancel it."""
        loop = asyncio.get_running_loop()
        all_started, submitted = asyncio.Event(), asyncio.Event()
        release = Event()
        lock = Lock()
        active = []
        effects = []
        original = ThreadPoolExecutor.submit
        submissions = 0

        def observe_submission(executor, function, *args):
            """Signal after the target reaches the occupied worker queue."""
            nonlocal submissions
            future = original(executor, function, *args)
            submissions += 1
            if submissions == 5:
                submitted.set()
            return future

        def hold(number: int) -> str:
            """Occupy a worker until the test releases the group."""
            with lock:
                active.append(number)
                if len(active) == 4:
                    loop.call_soon_threadsafe(all_started.set)
            assert release.wait(timeout=5)
            return "done"

        def effect() -> str:
            """Record an incorrectly entered callable."""
            effects.append("entered")
            return "unexpected"

        registry = ToolRegistry([hold, effect])
        monkeypatch.setattr(ThreadPoolExecutor, "submit", observe_submission)
        holders = [
            asyncio.create_task(registry.call_async("hold", json.dumps({"number": number})))
            for number in range(4)
        ]
        target = None
        queued = []
        try:
            await all_started.wait()
            target = asyncio.create_task(
                registry.call_async("effect", "{}", execution_started=effect if callback else None)
            )
            await submitted.wait()
            target.cancel()
            await deliver_callbacks()
            await deliver_callbacks()
            assert target.done()
            with pytest.raises(asyncio.CancelledError):
                await target
            assert effects == []
            queued = [
                asyncio.create_task(registry.call_async("hold", json.dumps({"number": number})))
                for number in range(4, 8)
            ]
            finished, _ = await asyncio.wait(queued, return_when=asyncio.FIRST_COMPLETED)
            assert len(finished) == 1
            assert (
                json.loads(next(iter(finished)).result())["problem"]["code"] == "tool.dispatch_busy"
            )
        finally:
            release.set()
            await asyncio.gather(
                *holders, *queued, *([target] if target is not None else []), return_exceptions=True
            )
            await asyncio.to_thread(registry.close)
        assert effects == []

    asyncio.run(scenario())


def test_interruption_between_worker_submission_and_wait_cannot_start_tool():
    """An interruption at the wait boundary prevents a submitted tool from entering."""
    effects = []

    def effect() -> str:
        """Record an incorrectly admitted tool."""
        effects.append("entered")
        return "unexpected"

    registry = ToolRegistry([effect])
    source, first_line = inspect.getsourcelines(ToolRegistry._run_worker_async)
    await_index = next(
        i for i, line in enumerate(source) if "return await self._await_owned" in line
    )
    wait_line = first_line + await_index

    def interrupt(frame, event, argument):
        """Interrupt the worker path immediately before its wait begins."""
        if (
            event == "line"
            and frame.f_code is ToolRegistry._run_worker_async.__code__
            and frame.f_lineno == wait_line
        ):
            sys.settrace(None)
            raise KeyboardInterrupt
        return interrupt

    async def scenario():
        """Interrupt the submitted operation before the event loop can release its gate."""
        previous = sys.gettrace()
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt):
                await registry.call_async("effect", "{}")
        finally:
            sys.settrace(previous)
        await deliver_callbacks()

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    assert effects == []


def test_owner_loop_close_refuses_unsettled_worker_without_deadlock():
    """A synchronous close cannot block the loop before a worker starts."""
    effects = []
    close_errors = []

    def effect() -> str:
        """Record the original operation after the owner loop admits it."""
        effects.append("entered")
        return "done"

    registry = ToolRegistry([effect])

    async def scenario():
        """Attempt close before the queued worker's owner-loop signal."""
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(registry.call_async("effect", "{}"))

        def close_while_active():
            """Capture the synchronous refusal without stopping the owner loop."""
            try:
                registry.close()
            except RuntimeError as error:
                close_errors.append(str(error))

        loop.call_soon(close_while_active)
        assert result_value(await task) == "done"

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    assert close_errors == ["Wait for tool dispatch to settle before closing on its loop."]
    assert effects == ["entered"]


def test_close_after_admission_prevents_new_worker_submission():
    """Registry close wins before a previously admitted caller reaches submission."""
    effects = []
    closed = Event()

    def effect() -> str:
        """Record an incorrectly submitted operation."""
        effects.append("entered")
        return "unexpected"

    registry = ToolRegistry([effect])
    method = ToolRegistry._run_worker_async
    source, first_line = inspect.getsourcelines(method)
    target = first_line + next(
        index for index, line in enumerate(source) if line.strip() == "with self._runner_lock:"
    )

    def close_at_submission(frame, event, argument):
        """Close from another thread after admission and before lifecycle revalidation."""
        if event == "line" and frame.f_code is method.__code__ and frame.f_lineno == target:
            sys.settrace(None)

            def close_registry():
                """Close without blocking the event loop."""
                try:
                    registry.close()
                finally:
                    closed.set()

            closer = Thread(target=close_registry)
            closer.start()
            assert closed.wait(timeout=5)
            closer.join(timeout=5)
        return close_at_submission

    async def scenario():
        """Observe the public dispatch refusal after the close wins."""
        previous = sys.gettrace()
        sys.settrace(close_at_submission)
        try:
            output = await registry.call_async("effect", "{}")
        finally:
            sys.settrace(previous)
        assert json.loads(output)["problem"]["code"] == "tool.dispatch_busy"

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    assert effects == []


def test_interruption_after_worker_admission_does_not_leak_capacity():
    """A registry remains usable after interruption at the admission boundary."""

    def effect() -> str:
        """Return a visible result after successful admission."""
        return "completed"

    registry = ToolRegistry([effect])
    source, first_line = inspect.getsourcelines(tool_registry_module.WorkerCapacity.acquire)
    target = first_line + next(
        index for index, line in enumerate(source) if line.strip() == "return True"
    )

    def interrupt(frame, event, argument):
        """Interrupt immediately after adding the admission token."""
        if (
            event == "line"
            and frame.f_code is tool_registry_module.WorkerCapacity.acquire.__code__
            and frame.f_lineno == target
        ):
            sys.settrace(None)
            raise KeyboardInterrupt
        return interrupt

    async def scenario():
        """Interrupt the first call and complete a second on the same registry."""
        previous = sys.gettrace()
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt):
                await registry.call_async("effect", "{}")
        finally:
            sys.settrace(previous)
        assert result_value(await registry.call_async("effect", "{}")) == "completed"

    try:
        asyncio.run(scenario())
    finally:
        registry.close()


def test_worker_submission_cancellation_releases_registry_capacity(monkeypatch):
    """Cancellation while submitting a tool returns its bounded admission slot."""

    def effect() -> str:
        """Return a result when the registry accepts a later call."""
        return "recovered"

    registry = ToolRegistry([effect])

    async def scenario():
        """Interrupt before submission and then dispatch another tool normally."""
        with monkeypatch.context() as scoped:
            scoped.setattr(
                tool_registry_module.WorkerEntry, "submit", Mock(side_effect=asyncio.CancelledError)
            )
            with pytest.raises(asyncio.CancelledError):
                await registry.call_async("effect", "{}")
        assert result_value(await registry.call_async("effect", "{}")) == "recovered"

    try:
        asyncio.run(scenario())
    finally:
        registry.close()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("cancel_pending", [False, True])
def test_exceptional_worker_wait_settles_entered_tool(monkeypatch, outcome, cancel_pending):
    """An unexpected interruption waits for the entered worker on the owner loop."""
    registry = ToolRegistry()
    release = Event()
    effects = []

    async def scenario():
        """Interrupt a wait while an entered callback still needs the owner loop."""
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def effect():
            """Settle only when the owner loop releases this worker."""
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
            effects.append("settled")
            if outcome == "failure":
                raise ValueError("worker failed")
            if outcome == "cancelled":
                raise asyncio.CancelledError

        async def interrupted_wait(future):
            """Interrupt only after worker entry."""
            await started.wait()
            loop.call_soon(release.set)
            raise KeyboardInterrupt

        monkeypatch.setattr(registry, "_await_owned_future_async", interrupted_wait)
        with pytest.raises(KeyboardInterrupt) as captured:
            await registry._run_worker_async(effect, cancel_pending=cancel_pending)
        assert effects == ["settled"]
        if outcome == "failure":
            assert captured.value.__notes__ == [
                "The blocking operation also stopped with ValueError."
            ]

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        registry.close()


@pytest.mark.parametrize("preclaimed", [False, True])
def test_exceptional_wait_releases_completed_owned_result(monkeypatch, preclaimed):
    """A completed but unconsumed owned result is handled without taking another owner."""
    registry = ToolRegistry()

    async def scenario():
        """Interrupt after the worker has returned its owned coroutine."""

        async def child():
            """Return a result if its original owner starts it."""
            return "owned"

        coroutine = child()
        handle = OwnedToolTask(coroutine)
        if preclaimed:
            handle.reserve()

        async def interrupted_wait(future):
            """Wait until the worker has completed, then interrupt before consuming it."""
            await asyncio.shield(future)
            raise KeyboardInterrupt

        monkeypatch.setattr(registry, "_await_owned_future_async", interrupted_wait)
        with pytest.raises(KeyboardInterrupt):
            await registry._run_worker_async(lambda: handle, cancel_pending=True)
        if preclaimed:
            assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED
            assert await handle.start(asyncio.get_running_loop()) == "owned"
        else:
            assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED

    try:
        asyncio.run(scenario())
    finally:
        registry.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_foreign_not_started_marker_does_not_claim_this_tool_was_unstarted(asynchronous):
    """A nested operation cannot certify that its already-entered caller never started."""
    effects = []

    def sync_tool() -> str:
        """Commit an effect and report another invocation's cancellation."""
        effects.append("entered")
        raise ToolExecutionNotStarted(object())

    async def async_tool() -> str:
        """Commit an effect and await another invocation's cancelled future."""
        effects.append("entered")
        future = asyncio.get_running_loop().create_future()
        future.set_exception(ToolExecutionNotStarted(object()))
        return await future

    function = async_tool if asynchronous else sync_tool
    registry = ToolRegistry([function])
    try:
        with pytest.raises(asyncio.CancelledError) as captured:
            asyncio.run(registry.call_async(function.__name__, "{}"))
    finally:
        registry.close()

    assert not isinstance(captured.value, ToolExecutionNotStarted)
    assert effects == ["entered"]


def test_registry_rejects_foreign_dispatch_not_started_marker(monkeypatch):
    """Only the current dispatch can certify that its callable was never entered."""

    async def effect() -> str:
        """Represent an effect whose nested invocation returned a foreign marker."""
        return "unused"

    async def foreign_marker(*args, **kwargs):
        """Propagate another dispatch's not-started marker at the tool boundary."""
        raise ToolExecutionNotStarted(object())

    monkeypatch.setattr(Tool, "execute_async", foreign_marker)
    registry = ToolRegistry([effect])
    try:
        with pytest.raises(asyncio.CancelledError) as captured:
            asyncio.run(registry.call_async("effect", "{}"))
        assert not isinstance(captured.value, ToolExecutionNotStarted)
    finally:
        registry.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_effectful_tool_returning_preclaimed_work_has_unknown_outcome(asynchronous):
    """A different owner's work cannot turn an entered tool into an ordinary failure."""
    effects = []

    async def foreign():
        """Produce a result only when the original owner starts this work."""
        return "foreign"

    handle = OwnedToolTask(foreign())
    handle.reserve()

    def effect() -> str:
        """Record the tool effect before returning another owner's work."""
        effects.append("entered")
        return handle

    registry = ToolRegistry([effect])
    try:
        with pytest.raises(ToolExecutionOutcomeUnknown):
            if asynchronous:
                asyncio.run(registry.call_async("effect", "{}"))
            else:
                registry.call("effect", "{}")
        with asyncio.Runner() as owner:
            task = handle.start(owner.get_loop())

            async def receive():
                """Await the original owner's independent task."""
                return await task

            assert owner.run(receive()) == "foreign"
    finally:
        registry.close()
    assert effects == ["entered"]


def test_failed_async_tool_planning_preserves_existing_event_loop():
    """A refused plan does not initialize a runner or replace the caller's loop."""
    effects = []

    async def effect() -> None:
        """Record an admitted effect."""
        effects.append(True)

    def refuse(_arguments):
        """Reject planning before the async tool needs a runner."""
        raise ValueError("synthetic refusal")

    def exercise():
        """Own and restore the event-loop state within an isolated thread."""
        prior = asyncio.new_event_loop()
        asyncio.set_event_loop(prior)
        registry = ToolRegistry([ToolRegistration(effect, operation_planner=refuse)])
        try:
            output = json.loads(registry.call("effect", "{}"))
            assert output["problem"]["code"] == "tool.planning_failed"
            assert asyncio.get_event_loop() is prior
            assert effects == []
        finally:
            registry.close()
            asyncio.set_event_loop(None)
            prior.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(exercise).result(timeout=5)


def test_interrupted_async_tool_dispatch_preserves_existing_event_loop():
    """Interrupted admission leaves the caller's loop intact through registry closure."""
    effects = []

    async def effect() -> None:
        """Record an admitted effect."""
        effects.append(True)

    def interrupt():
        """Interrupt before the tool is dispatched."""
        raise KeyboardInterrupt

    def exercise():
        """Isolate the thread's caller-owned event-loop binding."""
        prior = asyncio.new_event_loop()
        asyncio.set_event_loop(prior)
        registry = ToolRegistry([effect])
        try:
            with pytest.raises(KeyboardInterrupt):
                registry.call("effect", "{}", execution_started=interrupt)
            assert asyncio.get_event_loop() is prior
            registry.close()
            assert asyncio.get_event_loop() is prior
            assert effects == []
        finally:
            registry.close()
            asyncio.set_event_loop(None)
            prior.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(exercise).result(timeout=5)


def test_async_tool_closed_during_planning_does_not_start():
    """Closure during planning cannot resurrect an uninitialized coroutine runner."""
    effects = []

    async def effect() -> None:
        """Record an admitted effect."""
        effects.append(True)

    def close_during_plan(arguments):
        """Close the owner after it creates its runner but before dispatch."""
        registry.close()
        return OperationPlan(arguments=arguments)

    registry = ToolRegistry([ToolRegistration(effect, operation_planner=close_during_plan)])
    started = Mock()
    try:
        output = json.loads(registry.call("effect", "{}", execution_started=started))
        assert output["problem"]["code"] == "tool.dispatch_busy"
        started.assert_not_called()
        assert effects == []
    finally:
        registry.close()


def test_async_tool_runner_initialization_failure_is_reported(monkeypatch, caplog):
    """A failed runner setup is logged without starting the tool or leaking its error."""
    effects = []

    async def effect() -> None:
        """Record an admitted effect."""
        effects.append(True)

    failed = Mock(spec=asyncio.Runner)
    failed.get_loop.side_effect = OSError("private loop setup detail")
    monkeypatch.setattr(tool_registry_module.asyncio, "Runner", lambda **_kwargs: failed)
    registry = ToolRegistry([effect])
    started = Mock()
    try:
        output = json.loads(registry.call("effect", "{}", execution_started=started))
        assert output["problem"]["code"] == "tool.execution_failed"
        assert "private loop setup detail" not in output["problem"]["detail"]
        started.assert_not_called()
        assert effects == []
        assert len(caplog.records) == 1
        assert caplog.records[0].__dict__["error.type"] == "tool.execution_failed"
        assert caplog.records[0].__dict__["exception.type"] == "builtins.OSError"
    finally:
        registry.close()


def test_cancellation_waits_for_sync_tool_settlement():
    """Cancelling async dispatch signals its worker and waits for the actual outcome."""
    started = Event()
    release = Event()
    settled = Event()
    cancellation = Event()

    @declare_tool
    def wait_for_release(context: ToolContext) -> str:
        """Finish only after observing the worker's cancellation signal."""
        started.set()
        assert release.wait(timeout=5), "test synchronization was not released"
        settled.set()
        return "finished"

    registry = ToolRegistry([wait_for_release])

    async def dispatch():
        task = asyncio.create_task(
            registry.call_async("wait_for_release", "{}", cancellation_event=cancellation)
        )
        await asyncio.to_thread(started.wait)
        task.cancel()
        await asyncio.to_thread(cancellation.wait)
        assert not task.done()
        task.cancel()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(dispatch())
    finally:
        release.set()
        registry.close()

    assert settled.is_set()


def test_cancellation_settles_sync_work_without_a_run_signal():
    """Registry cancellation still waits for workers when no run event was supplied."""
    started = Event()
    release = Event()
    settled = Event()

    @declare_tool
    def wait_for_release() -> str:
        """Hold the worker until the event loop advances cancellation."""
        started.set()
        assert release.wait(timeout=5), "test synchronization was not released"
        settled.set()
        return "finished"

    registry = ToolRegistry([wait_for_release])

    async def dispatch():
        task = asyncio.create_task(registry.call_async("wait_for_release", "{}"))
        await asyncio.to_thread(started.wait)
        task.cancel()
        asyncio.get_running_loop().call_soon(release.set)
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(dispatch())
    finally:
        release.set()
        registry.close()

    assert settled.is_set()


def test_async_dispatch_supplies_local_cancellation_signal():
    """Standalone async dispatch signals a context-aware worker when cancelled."""
    started = Event()
    observed = Event()

    @declare_tool
    def wait_for_cancellation(context: ToolContext) -> str:
        """Wait until the dispatch-local cancellation signal is set."""
        started.set()
        if context.cancellation_event.wait(timeout=5):
            observed.set()
        return "finished"

    registry = ToolRegistry([wait_for_cancellation])

    async def dispatch():
        """Cancel after the worker starts and await its settled result."""
        task = asyncio.create_task(registry.call_async("wait_for_cancellation", "{}"))
        await asyncio.to_thread(started.wait)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(dispatch())
    finally:
        registry.close()
    assert observed.is_set()


def test_async_tool_observes_dispatch_cancellation_during_cleanup():
    """A cancelled async tool sees its signal before settling its cleanup."""
    observed = []

    async def scenario():
        """Cancel a running tool and inspect the signal in its cancellation handler."""
        started = asyncio.Event()

        @declare_tool
        async def wait_for_cancellation(context: ToolContext) -> str:
            """Record whether the dispatch signal was set before cancellation cleanup."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                observed.append(context.cancellation_event.is_set())
                raise

        registry = ToolRegistry([wait_for_cancellation])
        try:
            task = asyncio.create_task(registry.call_async("wait_for_cancellation", "{}"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())
    assert observed == [True]


def test_async_tool_self_cancellation_does_not_cancel_run():
    """A tool that cancels itself does not set the caller's shared run signal."""
    cancellation = Event()

    @declare_tool
    async def cancel_itself() -> str:
        """Stop the tool without requesting run cancellation."""
        raise asyncio.CancelledError

    async def scenario():
        """Observe the tool's cancellation and the independent run signal."""
        registry = ToolRegistry([cancel_itself])
        try:
            with pytest.raises(asyncio.CancelledError):
                await registry.call_async("cancel_itself", "{}", cancellation_event=cancellation)
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())
    assert not cancellation.is_set()


def test_prior_handled_cancellation_does_not_cancel_next_tool_run():
    """A previously handled task cancellation is not a new run interruption."""
    cancellation = Event()

    @declare_tool
    async def cancel_itself() -> str:
        """Cancel only this tool's operation."""
        raise asyncio.CancelledError

    async def scenario():
        """Handle an earlier cancellation before dispatching a self-cancelling tool."""
        current = asyncio.current_task()
        current.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        registry = ToolRegistry([cancel_itself])
        try:
            with pytest.raises(asyncio.CancelledError):
                await registry.call_async("cancel_itself", "{}", cancellation_event=cancellation)
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())
    assert not cancellation.is_set()


@pytest.mark.parametrize("prior_cancellation", [False, True])
def test_cancelled_sync_tool_future_does_not_cancel_run(prior_cancellation):
    """A cancelled tool-returned future does not set the caller's run signal."""
    cancellation = Event()

    async def scenario():
        """Return an independently cancelled future from a synchronous tool."""
        if prior_cancellation:
            asyncio.current_task().cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.sleep(0)
        future = asyncio.get_running_loop().create_future()
        future.cancel()

        @declare_tool
        def cancelled_future():
            """Return a cancelled result without requesting run cancellation."""
            return future

        registry = ToolRegistry([cancelled_future])
        try:
            with pytest.raises(asyncio.CancelledError):
                await registry.call_async("cancelled_future", "{}", cancellation_event=cancellation)
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())
    assert not cancellation.is_set()


@pytest.mark.parametrize("during_start", [False, True])
def test_cancelled_worker_does_not_start_or_await_additional_execution(during_start):
    """Cancellation before execution or before an awaitable result never launches more work."""
    started = asyncio.Event()
    release = Event()
    ran = []
    returned = []
    owner_loop = None

    async def finish():
        """Record work that must not be started after cancellation."""
        ran.append("awaited")
        return "done"

    def hold():
        """Hold a synchronous boundary until cancellation has been requested."""
        owner_loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test synchronization was not released"

    def invoke():
        """Return an unstarted coroutine only when execution was already admitted."""
        if not during_start:
            hold()
        ran.append("invoked")
        result = finish()
        returned.append(result)
        return result

    registry = ToolRegistry([invoke])

    async def scenario():
        """Cancel the public dispatch at its controlled worker boundary."""
        nonlocal owner_loop
        owner_loop = asyncio.get_running_loop()
        task = asyncio.create_task(
            registry.call_with_timing_async(
                "invoke",
                "{}",
                execution_started=hold if during_start else None,
            )
        )
        try:
            await started.wait()
            task.cancel()
            await deliver_callbacks()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    assert ran == ([] if during_start else ["invoked"])
    assert all(result.cr_frame is None for result in returned)


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("boundary", ["tool", "sync_callback", "async_callback"])
def test_cancelled_dispatch_preserves_preclaimed_owned_work(started, boundary):
    """Cancellation neither takes another owner's work nor loses its own signal."""

    async def scenario():
        """Interrupt a tool or callback before it returns a preclaimed handle."""
        loop = asyncio.get_running_loop()
        foreign_started = asyncio.Event()
        foreign_release = asyncio.Event()
        boundary_started = asyncio.Event()
        worker_release = Event()
        effects = []

        async def foreign():
            """Complete only when its original owner releases it."""
            foreign_started.set()
            await foreign_release.wait()
            effects.append("foreign completed")
            return "foreign result"

        handle = OwnedToolTask(foreign())
        if started:
            foreign_task = handle.claim(loop)
            await foreign_started.wait()
        else:
            handle.reserve()
            foreign_task = None

        def worker():
            """Return the other owner's handle after interruption."""
            loop.call_soon_threadsafe(boundary_started.set)
            assert worker_release.wait(timeout=5)
            return handle

        async def async_callback():
            """Return the other owner's handle while settling cancellation."""
            try:
                boundary_started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return handle

        def calculate():
            """Run the worker only when this is the tool boundary."""
            if boundary == "tool":
                return worker()
            effects.append("tool")
            return "done"

        before = worker if boundary == "sync_callback" else async_callback
        registry = ToolRegistry([calculate])
        task = asyncio.create_task(
            registry.call_async(
                "calculate", "{}", execution_started=None if boundary == "tool" else before
            )
        )
        try:
            await boundary_started.wait()
            task.cancel()
            await deliver_callbacks()
            worker_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            if foreign_task is None:
                foreign_task = handle.start(loop)
            assert not foreign_task.cancelling()
            foreign_release.set()
            assert await foreign_task == "foreign result"
            assert effects == ["foreign completed"]
        finally:
            worker_release.set()
            foreign_release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize("started", [False, True])
def test_cancelled_final_outcome_preserves_preclaimed_owned_work(started):
    """A final nested handle cannot transfer another owner's work after interruption."""

    async def scenario():
        """Return a preclaimed handle while settling an interrupted tool."""
        ready, foreign_ready, release = (asyncio.Event() for _ in range(3))

        async def foreign():
            """Wait for the original owner to finish the operation."""
            foreign_ready.set()
            await release.wait()
            return "foreign result"

        handle = OwnedToolTask(foreign())
        if started:
            foreign_task = handle.claim(asyncio.get_running_loop())
            await foreign_ready.wait()
        else:
            handle.reserve()
            foreign_task = None

        async def child():
            """Return the other owner's handle on cancellation."""
            try:
                ready.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return handle

        async def calculate() -> str:
            """Return a final child outcome."""
            return await OwnedToolTask(child(), final_result=True)

        registry = ToolRegistry([calculate])
        task = asyncio.create_task(registry.call_async("calculate", "{}"))
        try:
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert not hasattr(caught.value, "output")
            if foreign_task is None:
                foreign_task = handle.start(asyncio.get_running_loop())
            assert not foreign_task.cancelling()
            release.set()
            assert await foreign_task == "foreign result"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize("during_start", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_cancelled_callback_retrieves_its_completed_future_without_starting_more_work(
    during_start, outcome, caplog
):
    """Completed futures preserve invocation results while cancelled start callbacks stay closed."""
    started = asyncio.Event()
    release = Event()
    owner_loop = None
    returned = None
    invoked = []

    def future_callback():
        """Return the completed owner-loop future after cancellation reaches the worker."""
        owner_loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test synchronization was not released"
        return returned

    def invoke():
        """Run only when cancellation did not revoke the execution-start boundary."""
        invoked.append(True)
        return "unexpected" if during_start else future_callback()

    registry = ToolRegistry([invoke])

    async def scenario():
        """Cancel dispatch before the synchronous callback releases its result."""
        nonlocal owner_loop, returned
        owner_loop = asyncio.get_running_loop()
        returned = owner_loop.create_future()
        if outcome == "success":
            returned.set_result("confirmed")
        elif outcome == "failure":
            returned.set_exception(RuntimeError("private callback failure"))
        else:
            returned.cancel()
        task = asyncio.create_task(
            registry.call_with_timing_async(
                "invoke",
                "{}",
                execution_started=future_callback if during_start else None,
            )
        )
        try:
            await started.wait()
            task.cancel()
            await deliver_callbacks()
            release.set()
            with pytest.raises(asyncio.CancelledError) as captured:
                await task
            if not during_start and outcome != "cancelled":
                output = json.loads(captured.value.output)
                if outcome == "success":
                    assert output == {"ok": True, "result": "confirmed"}
                else:
                    assert output["problem"]["code"] == "tool.execution_failed"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    assert invoked == ([] if during_start else [True])
    records = [record for record in caplog.records if record.levelname == "ERROR"]
    if outcome == "failure":
        assert len(records) == 1
        expected_code = "tool.callback_failed" if during_start else "tool.execution_failed"
        assert records[0].__dict__["error.type"] == expected_code
        assert "private callback failure" not in records[0].getMessage()
    else:
        assert records == []


@pytest.mark.parametrize("signal", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "problem", "cancelled"])
def test_cancelled_start_callback_settles_returned_task_without_invoking_tool(
    caplog, monkeypatch, signal, repeated, outcome
):
    """Cancellation after callback worker return owns cleanup and never admits the tool body."""
    started, awaiting, cancelling, finish = (asyncio.Event() for _ in range(4))
    cancellation = Event() if signal else None
    invoked, completed = [], []
    owned = None

    shield = asyncio.shield

    def observe_shield(value):
        """Expose when dispatch starts waiting for the owned task."""
        if owned is not None and value is owned.task:
            awaiting.set()
        return shield(value)

    monkeypatch.setattr(asyncio, "shield", observe_shield)

    async def before_execution():
        """Finish owned callback cleanup even when cancellation has revoked tool execution."""
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await finish.wait()
            completed.append(True)
            if outcome == "failure":
                raise RuntimeError("private callback cleanup failure")
            if outcome == "problem":
                raise ProblemException(
                    Problem(
                        code="test.callback_failed",
                        title="Callback rejected",
                        detail="Callback rejected before execution.",
                    )
                )
            if outcome == "cancelled":
                raise
            return "callback settled"

    def execution_started():
        """Return owner-loop work through the synchronous callback adapter."""
        return owned

    def invoke():
        """Record an effect that cancellation must prevent from starting."""
        invoked.append(True)
        return "unexpected receipt"

    registry = ToolRegistry([invoke])

    async def scenario():
        """Cancel at the returned task's public await subscription without timing delays."""
        nonlocal owned
        owned = OwnedToolTask(before_execution())
        owned_task = owned.start(asyncio.get_running_loop())
        await started.wait()
        task = asyncio.create_task(
            registry.call_with_timing_async(
                "invoke",
                "{}",
                execution_started=execution_started,
                cancellation_event=cancellation,
            )
        )
        try:
            await awaiting.wait()
            task.cancel()
            await cancelling.wait()
            if repeated:
                task.cancel()
                advanced = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_soon(advanced.set_result, None)
                await advanced
            assert not task.done()
            assert not owned_task.done()
            assert completed == []
            assert cancellation is None or cancellation.is_set()
            finish.set()
            with pytest.raises(asyncio.CancelledError) as captured:
                await task
            assert not hasattr(captured.value, "output")
            assert owned_task.done()
            assert owned_task.cancelled() is (outcome == "cancelled")
            assert completed == [True]
            assert invoked == []
        finally:
            finish.set()
            task.cancel()
            await asyncio.gather(task, owned_task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    if outcome == "failure":
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.name == "loop.tooling.tool_registry"
        assert record.__dict__["error.type"] == "tool.callback_failed"
        assert record.__dict__["exception.type"] == "builtins.RuntimeError"
        assert record.__dict__["error.operation"] == "invoke execution-start callback"
        assert "private callback cleanup failure" not in record.getMessage()
    else:
        assert caplog.records == []


@pytest.mark.parametrize("fails", [False, True])
def test_start_callback_returned_future_preserves_normal_completion(fails):
    """Successful callback completion admits execution while callback failure prevents it."""
    owned = None
    invoked = []
    failure = RuntimeError("callback rejected")

    def execution_started():
        """Return a future on the invocation's owner loop."""
        return owned

    def invoke():
        """Record execution admitted by a successful callback."""
        invoked.append(True)
        return "confirmed"

    registry = ToolRegistry([invoke])

    async def scenario():
        """Resolve callback work normally without cancelling its dispatch owner."""
        nonlocal owned
        owned = asyncio.get_running_loop().create_future()
        if fails:
            owned.set_exception(failure)
            with pytest.raises(RuntimeError) as captured:
                await registry.call_async("invoke", "{}", execution_started=execution_started)
            assert captured.value is failure
        else:
            owned.set_result(None)
            output = await registry.call_async("invoke", "{}", execution_started=execution_started)
            assert result_value(output) == "confirmed"

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    assert invoked == ([] if fails else [True])


def test_sync_callable_awaits_an_owner_loop_future_normally():
    """Normal dispatch still awaits owner-loop futures and serializes their actual result."""
    returned = None

    def invoke():
        """Return a future belonging to the dispatch loop."""
        return returned

    registry = ToolRegistry([invoke])

    async def scenario():
        """Complete the future independently of the synchronous invocation worker."""
        nonlocal returned
        returned = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(returned.set_result, "confirmed")
        return await registry.call_async("invoke", "{}")

    try:
        assert json.loads(asyncio.run(scenario())) == {"ok": True, "result": "confirmed"}
    finally:
        registry.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_foreign_future_ownership_is_preserved_during_dispatch_cancellation(cancelled):
    """A future from another loop is rejected without being driven or cancelled here."""
    foreign_loop = asyncio.new_event_loop()
    future = foreign_loop.create_future()
    started = asyncio.Event()
    release = Event()
    owner_loop = None

    def invoke():
        """Return a pending foreign future through a controlled worker boundary."""
        owner_loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test synchronization was not released"
        return future

    registry = ToolRegistry([invoke])

    async def scenario():
        """Reject foreign-loop work both normally and when the dispatch owner is cancelled."""
        nonlocal owner_loop
        owner_loop = asyncio.get_running_loop()
        task = asyncio.create_task(registry.call_async("invoke", "{}"))
        try:
            await started.wait()
            if cancelled:
                task.cancel()
                await deliver_callbacks()
            release.set()
            if cancelled:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(ToolExecutionOutcomeUnknown):
                    await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
        assert not future.done()
    finally:
        registry.close()
        future.cancel()
        foreign_loop.close()


@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize("clear_cancellation", [False, True])
def test_cancellation_settles_returned_task_already_cancelling(
    monkeypatch, callback, clear_cancellation
):
    """Borrowed cleanup finishes even when its cancellation state has been cleared."""
    started, cleaning, attached, release = (asyncio.Event() for _ in range(4))
    completed = []
    owned = None
    shield = asyncio.shield

    async def effect():
        """Finish cleanup after its first cancellation request."""
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            if clear_cancellation:
                asyncio.current_task().uncancel()
            cleaning.set()
            await release.wait()
            completed.append(True)

    def invoke():
        """Return owned work or an ordinary value after the callback."""
        return "done" if callback else owned

    def execution_started():
        """Return owned work from the execution-start callback."""
        return owned

    def observe_shield(value):
        """Expose when dispatch is waiting for the owned task."""
        if value is owned:
            attached.set()
        return shield(value)

    monkeypatch.setattr(asyncio, "shield", observe_shield)
    registry = ToolRegistry([invoke])

    async def scenario():
        """Interrupt dispatch while its returned task is already settling."""
        nonlocal owned
        owned = asyncio.create_task(effect())
        await started.wait()
        owned.cancel()
        await cleaning.wait()
        dispatch = asyncio.create_task(
            registry.call_async(
                "invoke", "{}", execution_started=execution_started if callback else None
            )
        )
        try:
            await attached.wait()
            dispatch.cancel()
            await deliver_callbacks()
            assert not owned.done()
            assert not dispatch.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await dispatch
            assert completed == [True]
        finally:
            release.set()
            await asyncio.gather(dispatch, owned, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        registry.close()


@pytest.mark.parametrize("signal", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_returned_task_cancellation_signals_context_and_retains_its_final_outcome(
    monkeypatch, caplog, signal, fails
):
    """Drains returned-task cleanup on cancellation and publishes its result exactly once."""
    cancelling, finish, awaiting = (asyncio.Event() for _ in range(3))
    cancellation = Event() if signal else None
    owned = None
    shield = asyncio.shield

    def observe_shield(value):
        """Expose when dispatch owns the returned task's result."""
        if owned is not None and value is owned.task:
            awaiting.set()
        return shield(value)

    monkeypatch.setattr(asyncio, "shield", observe_shield)

    async def effect():
        """Settle cancellation with a confirmed result or a structured reported exception."""
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert cancellation is None or cancellation.is_set()
            cancelling.set()
            await finish.wait()
            if fails:
                raise RuntimeError("private returned-task failure")
            return "confirmed"

    def invoke():
        """Transfer the owned task from a synchronous callback."""
        return owned

    registry = ToolRegistry([invoke])

    async def scenario():
        """Cancel only after the returned task is being awaited and settle repeated cancellation."""
        nonlocal owned
        owned = OwnedToolTask(effect())
        owned_task = owned.start(asyncio.get_running_loop())
        task = asyncio.create_task(
            registry.call_async("invoke", "{}", cancellation_event=cancellation)
        )
        try:
            await awaiting.wait()
            task.cancel()
            await cancelling.wait()
            task.cancel()
            finish.set()
            with pytest.raises(asyncio.CancelledError) as captured:
                await task
            output = json.loads(captured.value.output)
            if fails:
                assert output["problem"]["code"] == "tool.execution_failed"
            else:
                assert output == {"ok": True, "result": "confirmed"}
        finally:
            finish.set()
            if not owned_task.done():
                owned.cancel()
            await asyncio.gather(task, owned_task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
    assert owned.task.done()
    if fails:
        assert len(caplog.records) == 1
        assert caplog.records[0].__dict__["error.type"] == "tool.execution_failed"
        assert "private returned-task failure" not in caplog.records[0].getMessage()
    else:
        assert caplog.records == []


def test_worker_failure_is_logged_when_cancellation_waits_for_settlement(caplog):
    """A worker exception after cancellation is recorded once at the registry boundary."""
    started = Event()
    release = Event()
    cancellation = Event()

    @declare_tool
    def fail_after_release() -> str:
        """Raise after the cancellation owner releases the worker."""
        started.set()
        assert release.wait(timeout=5), "test synchronization was not released"
        raise RuntimeError("private failure detail")

    registry = ToolRegistry([fail_after_release])

    async def dispatch():
        task = asyncio.create_task(
            registry.call_async("fail_after_release", "{}", cancellation_event=cancellation)
        )
        await asyncio.to_thread(started.wait)
        task.cancel()
        await asyncio.to_thread(cancellation.wait)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(dispatch())
    finally:
        release.set()
        registry.close()

    records = [record for record in caplog.records if record.name == "loop.tooling.tool_registry"]
    assert len(records) == 1
    assert records[0].__dict__["error.type"] == "tool.callback_failed"
    assert records[0].__dict__["exception.type"] == "builtins.RuntimeError"
    assert "private failure detail" not in records[0].getMessage()


@pytest.mark.parametrize("failure", ["refusal", "oserror", "problem"])
def test_cancelled_worker_reports_only_unexpected_admission_failures(caplog, failure):
    """Cancellation retains unexpected policy-check diagnostics without logging refusal."""
    started = Event()
    release = Event()
    cancellation = Event()
    effects = []

    @declare_tool
    def effect() -> None:
        """Record an admitted synthetic effect."""
        effects.append(True)

    def admission() -> bool:
        """Settle after cancellation with a refusal or a failed policy check."""
        started.set()
        assert release.wait(timeout=5), "test synchronization was not released"
        if failure == "oserror":
            raise OSError("private policy detail")
        if failure == "problem":
            raise ProblemException(
                Problem(
                    code="policy.unavailable", title="Unavailable", detail="private policy detail"
                )
            )
        return False

    registry = ToolRegistry([effect])

    async def dispatch():
        """Cancel while the worker is still evaluating admission."""
        task = asyncio.create_task(
            registry.call_with_timing_async(
                "effect", "{}", execution_admission=admission, cancellation_event=cancellation
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        assert await asyncio.to_thread(cancellation.wait, 5)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(dispatch())
    finally:
        release.set()
        registry.close()
    assert effects == []
    records = [record for record in caplog.records if record.name == "loop.tooling.tool_registry"]
    if failure != "refusal":
        assert len(records) == 1
        assert records[0].__dict__["error.type"] == "tool.callback_failed"
        expected = "builtins.OSError" if failure == "oserror" else "loop.errors.ProblemException"
        assert records[0].__dict__["exception.type"] == expected
        assert "private policy detail" not in records[0].getMessage()
    else:
        assert records == []


def test_cancelled_executor_future_settles_without_reading_its_exception(monkeypatch):
    """A cancelled executor future still releases worker admission during settlement."""

    @declare_tool
    def calculate() -> int:
        """Return a value if the worker callback is submitted."""
        return 1

    registry = ToolRegistry([calculate])

    async def dispatch():
        future = Future()
        future.cancel()
        monkeypatch.setattr(ThreadPoolExecutor, "submit", lambda *_args, **_kwargs: future)
        with pytest.raises(asyncio.CancelledError):
            await registry.call_async("calculate", "{}")

    try:
        asyncio.run(dispatch())
    finally:
        registry.close()


def test_worker_context_relays_interaction_and_instruction_effects_and_returns_receipts(tmp_path):
    """Worker context effects stay on the owner and file receipts return with the result."""
    owner_thread = get_ident()
    service_threads = []
    path = tmp_path / "observed.txt"
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    observe_path = instructions.observe_path

    def observe_on_owner(path_value, *, directory=False):
        service_threads.append(get_ident())
        return observe_path(path_value, directory=directory)

    instructions.observe_path = observe_on_owner
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)

    async def confirm_on_owner(message, *, default=False):
        service_threads.append(get_ident())
        return True

    interaction.confirm_async = confirm_on_owner

    @declare_tool
    def inspect_context(context: ToolContext) -> dict:
        """Record a read, update instruction scope, and request owner-thread confirmation."""
        context.record_cached_read("cache://reference")
        context.observe_file(path)
        return {"allowed": context.confirm("Continue?")}

    registry = ToolRegistry([inspect_context])

    try:
        output = asyncio.run(
            registry.call_async(
                "inspect_context",
                "{}",
                interaction=interaction,
                instructions_manager=instructions,
            )
        )
    finally:
        registry.close()

    result = json.loads(output)
    assert result["ok"] is True
    assert result["result"] == {"allowed": True}
    assert result["read_versions"] == {"cache://reference": None}
    assert service_threads == [owner_thread, owner_thread]


@pytest.mark.parametrize("native_async", [False, True])
@pytest.mark.parametrize("answer", [True, False, "pending"])
def test_tool_confirmation_requires_explicit_boolean_approval(native_async, answer):
    """A nested foreign future cannot approve either a sync or native async tool."""
    effects = []
    foreign = asyncio.new_event_loop()
    future = foreign.create_future()
    interaction = Mock(spec=Interaction)
    interaction.confirm_async = AsyncMock(return_value=future if answer == "pending" else answer)
    interaction.prompt_async = AsyncMock()

    def sync_effect(context: ToolContext) -> bool:
        """Record an effect only if the user explicitly approved it."""
        if context.confirm("Proceed?"):
            effects.append("committed")
            return True
        return False

    async def async_effect(context: ToolContext) -> bool:
        """Record an effect only if the user explicitly approved it."""
        if await context.confirm_async("Proceed?"):
            effects.append("committed")
            return True
        return False

    function = async_effect if native_async else sync_effect
    registry = ToolRegistry([function], interaction=interaction)
    try:
        output = asyncio.run(registry.call_async(function.__name__, "{}"))
        if answer == "pending":
            assert json.loads(output)["problem"]["code"] == "tool.invalid_confirmation"
        else:
            assert result_value(output) is (answer is True)
        assert effects == (["committed"] if answer is True else [])
        assert not future.done()
    finally:
        registry.close()
        future.cancel()
        foreign.close()


def test_additional_authorization_rechecks_exact_dynamic_operations_on_owner():
    """Runtime-discovered operations receive their own exact async authorization check."""
    owner_thread = get_ident()
    authorization_calls = []
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None

    async def authorize_async(operations, **_kwargs):
        authorization_calls.append((get_ident(), operations))
        return SimpleNamespace(decision=Decision.ALLOW)

    permissions.authorize_async = AsyncMock(side_effect=authorize_async)

    def plan(arguments):
        url = arguments["url"]
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.NETWORK_REQUEST,
                    target=NetworkTarget(url=url, origin=url),
                ),
            ),
        )

    @declare_tool(actions={Action.NETWORK_REQUEST}, operation_planner=plan)
    def fetch_redirect(context: ToolContext, url: str) -> str:
        """Authorize a redirect before returning its canonical target."""
        redirected = context.authorize_additional({"url": "https://second.local"})
        return redirected.arguments["url"]

    registry = ToolRegistry([fetch_redirect], permission_manager=permissions)

    try:
        output = asyncio.run(registry.call_async("fetch_redirect", '{"url":"https://first.local"}'))
    finally:
        registry.close()

    assert result_value(output) == "https://second.local"
    assert len(authorization_calls) == 2
    assert all(thread_id == owner_thread for thread_id, _ in authorization_calls)
    assert [operations[0].target.url for _, operations in authorization_calls] == [
        "https://first.local",
        "https://second.local",
    ]


@pytest.mark.parametrize(
    ("additional_decision", "expected"),
    [(Decision.ALLOW, "https://second.local"), (Decision.DENY, "tool.denied")],
)
def test_sync_additional_authorization_preserves_its_denial_contract(additional_decision, expected):
    """Sync runtime authorization checks dynamic targets and keeps denial wording."""
    authorization_calls = []
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None

    def authorize(operations, **_kwargs):
        authorization_calls.append(operations)
        decision = Decision.ALLOW if len(authorization_calls) == 1 else additional_decision
        return SimpleNamespace(decision=decision, reason="redirect denied")

    permissions.authorize = Mock(side_effect=authorize)

    def plan(arguments):
        url = arguments["url"]
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.NETWORK_REQUEST,
                    target=NetworkTarget(url=url, origin=url),
                ),
            ),
        )

    @declare_tool(actions={Action.NETWORK_REQUEST}, operation_planner=plan)
    def fetch_redirect(context: ToolContext, url: str) -> str:
        """Authorize a redirect before returning its canonical target."""
        redirected = context.authorize_additional({"url": "https://second.local"})
        return redirected.arguments["url"]

    registry = ToolRegistry([fetch_redirect], permission_manager=permissions)

    try:
        output = json.loads(registry.call("fetch_redirect", '{"url":"https://first.local"}'))
    finally:
        registry.close()

    assert len(authorization_calls) == 2
    assert [operations[0].target.url for operations in authorization_calls] == [
        "https://first.local",
        "https://second.local",
    ]
    if additional_decision is Decision.ALLOW:
        assert output["result"] == expected
    else:
        assert output["problem"]["code"] == expected
        assert output["problem"]["title"] == "Additional operation denied"


def test_sync_additional_authorization_preserves_non_denial_planning_errors():
    """Dynamic planning errors keep their original structured problem through authorization."""
    authorization_calls = []
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None

    def authorize(operations, **_kwargs):
        authorization_calls.append(operations)
        return SimpleNamespace(decision=Decision.ALLOW)

    permissions.authorize = Mock(side_effect=authorize)

    def plan(arguments):
        if arguments["url"] == "https://second.local":
            raise ValueError("redirect could not be planned")
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.NETWORK_REQUEST,
                    target=NetworkTarget(url=arguments["url"], origin=arguments["url"]),
                ),
            ),
        )

    @declare_tool(actions={Action.NETWORK_REQUEST}, operation_planner=plan)
    def fetch_redirect(context: ToolContext, url: str) -> str:
        """Authorize a redirect before returning its canonical target."""
        return context.authorize_additional({"url": "https://second.local"}).arguments["url"]

    registry = ToolRegistry([fetch_redirect], permission_manager=permissions)

    try:
        output = json.loads(registry.call("fetch_redirect", '{"url":"https://first.local"}'))
    finally:
        registry.close()

    assert output["problem"]["code"] == "tool.planning_failed"
    assert output["problem"]["detail"] == "redirect could not be planned"
    assert len(authorization_calls) == 1


def test_async_additional_authorization_preserves_its_denial_contract():
    """Async runtime authorization returns a typed denial for a redirected target."""
    decisions = iter((Decision.ALLOW, Decision.DENY))
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None

    async def authorize_async(*_args, **_kwargs):
        return SimpleNamespace(decision=next(decisions), reason="redirect denied")

    permissions.authorize_async = AsyncMock(side_effect=authorize_async)

    def plan(arguments):
        url = arguments["url"]
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.NETWORK_REQUEST,
                    target=NetworkTarget(url=url, origin=url),
                ),
            ),
        )

    @declare_tool(actions={Action.NETWORK_REQUEST}, operation_planner=plan)
    def fetch_redirect(context: ToolContext, url: str) -> str:
        """Authorize a redirect before returning its canonical target."""
        redirected = context.authorize_additional({"url": "https://second.local"})
        return redirected.arguments["url"]

    registry = ToolRegistry([fetch_redirect], permission_manager=permissions)

    try:
        output = json.loads(
            asyncio.run(registry.call_async("fetch_redirect", '{"url":"https://first.local"}'))
        )
    finally:
        registry.close()

    assert output["problem"]["code"] == "tool.denied"
    assert output["problem"]["title"] == "Additional operation denied"


def test_native_async_tool_uses_async_context_authorization_on_the_owner():
    """Native async tools can authorize dynamic operations without the sync policy path."""
    owner_thread = get_ident()
    authorization_calls = []
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None

    async def authorize_async(operations, **_kwargs):
        authorization_calls.append((get_ident(), operations))
        return SimpleNamespace(decision=Decision.ALLOW)

    permissions.authorize_async = AsyncMock(side_effect=authorize_async)

    def plan(arguments):
        url = arguments["url"]
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.NETWORK_REQUEST,
                    target=NetworkTarget(url=url, origin=url),
                ),
            ),
        )

    @declare_tool(actions={Action.NETWORK_REQUEST}, operation_planner=plan)
    async def fetch_redirect(context: ToolContext, url: str) -> str:
        """Authorize a redirect before returning its canonical target."""
        redirected = await context.authorize_additional_async({"url": "https://second.local"})
        return redirected.arguments["url"]

    registry = ToolRegistry([fetch_redirect], permission_manager=permissions)

    try:
        output = asyncio.run(registry.call_async("fetch_redirect", '{"url":"https://first.local"}'))
    finally:
        registry.close()

    assert result_value(output) == "https://second.local"
    assert len(authorization_calls) == 2
    assert all(thread_id == owner_thread for thread_id, _ in authorization_calls)
    assert [operations[0].target.url for _, operations in authorization_calls] == [
        "https://first.local",
        "https://second.local",
    ]
    permissions.authorize.assert_not_called()


def test_sync_callable_rejects_future_bound_to_another_loop():
    """Both registry modes retain unknown outcomes for foreign-loop tool results."""
    foreign_loop = asyncio.new_event_loop()
    foreign_future = foreign_loop.create_future()
    foreign_future.set_result("unowned")

    @declare_tool
    def return_foreign_future() -> str:
        """Return a completed result attached to another event loop."""
        return foreign_future

    registry = ToolRegistry([return_foreign_future])

    try:
        with pytest.raises(ToolExecutionOutcomeUnknown):
            registry.call("return_foreign_future", "{}")
        with pytest.raises(ToolExecutionOutcomeUnknown):
            asyncio.run(registry.call_async("return_foreign_future", "{}"))
    finally:
        registry.close()
        foreign_loop.close()


def test_registry_reuses_a_runner_for_async_planners_in_sync_dispatch():
    """Sync dispatch uses the registry runner for async planners and continuations."""
    loop_ids = []

    async def continuation():
        loop_ids.append(id(asyncio.get_running_loop()))
        return OperationPlan(arguments={"number": 7})

    async def planner(arguments):
        loop_ids.append(id(asyncio.get_running_loop()))
        return OperationPlan(arguments=arguments, continuation=continuation)

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Return the value produced by the async plan."""
        return number

    registry = ToolRegistry([calculate])

    try:
        first = result_value(registry.call("calculate", '{"number": 1}'))
        second = result_value(registry.call("calculate", '{"number": 2}'))
    finally:
        registry.close()

    assert first == second == 7
    assert len(loop_ids) == 4
    assert len(set(loop_ids)) == 1


@pytest.mark.parametrize("async_dispatch", [False, True])
def test_registry_accepts_explicitly_owned_planner_work(async_dispatch):
    """Both dispatch modes resolve a planner's explicitly owned coroutine."""

    async def prepare(arguments):
        """Produce one normalized plan on the dispatch loop."""
        return OperationPlan(arguments=arguments)

    def planner(arguments):
        """Transfer one unstarted planner coroutine."""
        return OwnedToolTask(prepare(arguments))

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Return the planned number."""
        return number

    registry = ToolRegistry([calculate])
    try:
        output = (
            asyncio.run(registry.call_async("calculate", '{"number": 3}'))
            if async_dispatch
            else registry.call("calculate", '{"number": 3}')
        )
        assert result_value(output) == 3
    finally:
        registry.close()


def test_sync_dispatch_rejects_awaitable_planners_inside_a_running_loop():
    """Sync planning cannot create an unowned event loop from an active loop thread."""

    def planner(arguments):
        async def plan():
            return OperationPlan(arguments=arguments)

        return plan()

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Return the value only if sync planning was allowed."""
        return number

    registry = ToolRegistry([calculate])

    async def dispatch():
        """Reject a returned coroutine at the internal synchronous invocation boundary."""
        with pytest.raises(ProblemException) as captured:
            registry._invoke(registry._tools["calculate"], planner, {"number": 3})
        assert captured.value.problem.code == "tool.async_required"
        return registry.call("calculate", '{"number": 3}')

    output = asyncio.run(dispatch())

    assert json.loads(output)["problem"]["code"] == "tool.async_required"
    registry.close()


def test_sync_dispatch_rejects_async_planners_inside_a_running_loop():
    """Sync planning refuses coroutine planners when their caller already owns a loop."""

    async def planner(arguments):
        return OperationPlan(arguments=arguments)

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Return the value only if sync planning was allowed."""
        return number

    registry = ToolRegistry([calculate])

    async def dispatch():
        return registry.call("calculate", '{"number": 3}')

    try:
        output = asyncio.run(dispatch())
    finally:
        registry.close()

    assert json.loads(output)["problem"]["code"] == "tool.async_required"


def test_sync_dispatch_rejects_non_closeable_planner_results_inside_a_loop():
    """Sync planning tolerates awaitables without a close method when it cannot await them."""

    @declare_tool(operation_planner=lambda arguments: asyncio.get_running_loop().create_future())
    def calculate(number: int) -> int:
        """Return the value only if sync planning was allowed."""
        return number

    registry = ToolRegistry([calculate])

    async def dispatch():
        """Reject a future returned by sync planning without invoking the tool."""
        with pytest.raises(ProblemException) as captured:
            registry._invoke(
                registry._tools["calculate"],
                lambda arguments: asyncio.get_running_loop().create_future(),
                {"number": 3},
            )
        assert captured.value.problem.code == "tool.async_required"

    try:
        asyncio.run(dispatch())
    finally:
        registry.close()


def test_sync_dispatch_rejects_a_sync_callable_returning_awaitable_inside_a_loop():
    """A returned coroutine is closed when a sync callable cannot await it in an active loop."""

    @declare_tool
    def calculate() -> str:
        """Return an awaitable whose body must not start in a sync call."""

        async def result() -> str:
            pytest.fail("the rejected coroutine must not execute")

        return result()

    registry = ToolRegistry([calculate])

    async def dispatch():
        return registry.call("calculate", "{}")

    output = asyncio.run(dispatch())

    assert json.loads(output)["problem"]["code"] == "tool.async_required"


def test_async_sync_planner_rejects_a_foreign_loop_future():
    """An async planner result attached to another loop is refused before tool execution."""
    foreign_loop = asyncio.new_event_loop()
    foreign_future = foreign_loop.create_future()
    foreign_future.set_result(OperationPlan(arguments={"number": 4}))

    @declare_tool(operation_planner=lambda arguments: foreign_future)
    async def calculate(number: int) -> int:
        """Return the planned value if its planner result is owned."""
        return number

    registry = ToolRegistry([calculate])

    try:
        output = asyncio.run(registry.call_async("calculate", '{"number": 1}'))
    finally:
        registry.close()
        foreign_loop.close()

    assert json.loads(output)["problem"]["code"] == "tool.loop_mismatch"


def test_sync_async_planner_rejects_a_foreign_loop_future():
    """Sync dispatch refuses an async planner result tied to another event loop."""
    foreign_loop = asyncio.new_event_loop()
    foreign_future = foreign_loop.create_future()
    foreign_future.set_result(OperationPlan(arguments={"number": 4}))

    @declare_tool(operation_planner=lambda arguments: foreign_future)
    def calculate(number: int) -> int:
        """Return the planned value if its planner result is owned."""
        return number

    registry = ToolRegistry([calculate])

    try:
        output = registry.call("calculate", '{"number": 1}')
    finally:
        registry.close()
        foreign_loop.close()

    assert json.loads(output)["problem"]["code"] == "tool.loop_mismatch"


def test_worker_submission_failure_quarantines_ambiguous_executor(monkeypatch):
    """An ambiguous failure refuses reuse while a fresh registry can dispatch normally."""

    @declare_tool
    def calculate() -> int:
        """Return a value after executor admission recovers."""
        return 1

    registry = ToolRegistry([calculate])
    replacement = ToolRegistry([calculate])

    async def dispatch():
        with monkeypatch.context() as scoped:
            scoped.setattr(
                ThreadPoolExecutor,
                "submit",
                Mock(side_effect=RuntimeError("executor unavailable")),
            )
            failed = await registry.call_async("calculate", "{}")
        unavailable = await registry.call_async("calculate", "{}")
        succeeded = await replacement.call_async("calculate", "{}")
        return failed, unavailable, succeeded

    try:
        failed, unavailable, succeeded = asyncio.run(dispatch())
    finally:
        registry.close()
        replacement.close()

    assert json.loads(failed)["problem"]["code"] == "tool.dispatch_unavailable"
    assert json.loads(unavailable)["problem"]["code"] == "tool.dispatch_unavailable"
    assert result_value(succeeded) == 1


def test_interrupted_failed_submission_still_quarantines_registry(monkeypatch):
    """Interrupted cancellation cannot reuse an ambiguously submitted registry worker."""

    def calculate() -> int:
        """Return a value only if the worker was safely admitted."""
        return 1

    registry = ToolRegistry([calculate])
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(registry._executor, "submit", Mock(side_effect=RuntimeError))
            scoped.setattr(
                tool_registry_module.WorkerEntry,
                "cancel_before_start",
                Mock(side_effect=KeyboardInterrupt),
            )
            with pytest.raises(KeyboardInterrupt):
                asyncio.run(registry.call_async("calculate", "{}"))
        output = asyncio.run(registry.call_async("calculate", "{}"))
        assert json.loads(output)["problem"]["code"] == "tool.dispatch_unavailable"
    finally:
        registry.close()


def test_registry_quarantines_after_worker_thread_start_failure(monkeypatch, caplog):
    """A thread-start failure cannot run an abandoned call or reuse an uncertain pool."""
    effects = []
    original_start = Thread.start

    def effect() -> str:
        """Record only the operation accepted by a fresh registry."""
        effects.append("entered")
        return "recovered"

    def fail_worker_start(thread):
        """Reject tool worker creation at the real thread-start boundary."""
        if thread.name.startswith("loop-tool"):
            raise RuntimeError("synthetic worker start failure")
        return original_start(thread)

    registry = ToolRegistry([effect])
    replacement = ToolRegistry([effect])

    async def scenario():
        """Fail after queue insertion, then use an independently created registry."""
        with monkeypatch.context() as scoped:
            scoped.setattr(Thread, "start", fail_worker_start)
            failed = await registry.call_async("effect", "{}")
        assert json.loads(failed)["problem"]["code"] == "tool.dispatch_unavailable"
        unavailable = await registry.call_async("effect", "{}")
        assert json.loads(unavailable)["problem"]["code"] == "tool.dispatch_unavailable"
        assert result_value(await replacement.call_async("effect", "{}")) == "recovered"

    try:
        asyncio.run(scenario())
    finally:
        registry.close()
        replacement.close()
    assert effects == ["entered"]
    assert any(
        record.__dict__.get("error.type") == "tool.dispatch_unavailable"
        for record in caplog.records
    )


def test_worker_dispatch_preserves_context_variables():
    """Registry workers receive the caller's correlation context variables."""

    trace_id = contextvars.ContextVar("trace_id", default="missing")

    @declare_tool
    def inspect_context() -> str:
        """Return the active correlation identifier."""
        return trace_id.get()

    registry = ToolRegistry([inspect_context])

    async def dispatch():
        token = trace_id.set("trace-123")
        try:
            return await registry.call_async("inspect_context", "{}")
        finally:
            trace_id.reset(token)

    try:
        output = asyncio.run(dispatch())
    finally:
        registry.close()

    assert result_value(output) == "trace-123"


def test_worker_context_service_values_are_detached():
    """Owner service values return as copies or proxies instead of live services."""

    @dataclass
    class Record:
        """Provide a mutable owner value for relay isolation coverage."""

        value: str

    class OwnerService:
        """Provide typed owner data to a worker context.

        Initialize representative values for direct copies and nested service proxies.
        """

        agents_filenames: tuple[str, ...]
        empty: None
        text: str
        location: Path
        model: Problem
        record: Record
        sequence: tuple[int, int]
        items: list[str]
        mapping: dict[str, str]
        selected: set[str]
        nested: SimpleNamespace

        def __init__(self) -> None:
            self.agents_filenames = ()
            self.empty = None
            self.text = "owner-text"
            self.location = Path("$WORKSPACE_ROOT/item")
            self.model = Problem(code="test.owner", title="Owner", detail="safe")
            self.record = Record("record")
            self.sequence = (1, 2)
            self.items = ["first", "second"]
            self.mapping = {"key": "value"}
            self.selected = {"first", "second"}
            self.nested = SimpleNamespace(value="proxied")

    @declare_tool
    def inspect_values(context: ToolContext) -> dict:
        """Read all detached owner values through the invocation context."""
        service = context.instructions_manager
        return {
            "empty": service.empty,
            "text": service.text,
            "location": str(service.location),
            "model": service.model.code,
            "record": service.record.value,
            "sequence": service.sequence,
            "items": service.items,
            "mapping": service.mapping,
            "selected": sorted(service.selected),
            "nested": service.nested.value,
        }

    registry = ToolRegistry([inspect_values])

    try:
        output = asyncio.run(
            registry.call_async("inspect_values", "{}", instructions_manager=OwnerService())
        )
    finally:
        registry.close()

    assert result_value(output) == {
        "empty": None,
        "text": "owner-text",
        "location": "$WORKSPACE_ROOT/item",
        "model": "test.owner",
        "record": "record",
        "sequence": [1, 2],
        "items": ["first", "second"],
        "mapping": {"key": "value"},
        "selected": ["first", "second"],
        "nested": "proxied",
    }


def test_owner_portal_runs_sync_calls_and_rejects_unawaited_owner_results():
    """Owner-thread relay calls run directly and refuse to block on returned awaitables."""

    async def exercise():
        portal = ToolExecutionPortal(asyncio.get_running_loop())
        assert portal.call(lambda: "owner") == "owner"
        assert (
            await portal.call_method_async(
                SimpleNamespace(result_async=Mock(return_value="owner")), "result_async"
            )
            == "owner"
        )

        async def result():
            return "not awaited"

        with pytest.raises(ProblemException, match="must be awaited"):
            portal.call(result)

        coroutine = result()
        with pytest.raises(ProblemException, match="must be awaited"):
            portal.call(lambda: OwnedToolTask(coroutine))
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED

        claimed_coroutine = result()
        claimed = OwnedToolTask(claimed_coroutine)
        claimed.reserve()
        with pytest.raises(ProblemException, match="must be awaited"):
            portal.call(lambda: claimed)
        assert inspect.getcoroutinestate(claimed_coroutine) == inspect.CORO_CREATED
        assert await claimed.start(asyncio.get_running_loop()) == "not awaited"

        started = OwnedToolTask(result())
        owner_task = started.claim(asyncio.get_running_loop())
        with pytest.raises(ProblemException, match="must be awaited"):
            portal.call(lambda: started)
        assert await owner_task == "not awaited"

        future = asyncio.get_running_loop().create_future()
        with pytest.raises(ProblemException, match="must be awaited"):
            portal.call(lambda: future)
        future.cancel()

    asyncio.run(exercise())


def test_owner_portal_rejects_stopped_and_closed_loops():
    """A worker cannot enqueue owner calls after its event loop stops."""
    stopped_loop = asyncio.new_event_loop()
    stopped_portal = ToolExecutionPortal(stopped_loop)
    with pytest.raises(RuntimeError, match="relay is unavailable"):
        stopped_portal.call(lambda: None)
    with pytest.raises(RuntimeError, match="relay is unavailable"):
        asyncio.run(stopped_portal.call_method_async(SimpleNamespace(), "method_async"))
    stopped_loop.close()

    closed_loop = asyncio.new_event_loop()
    closed_portal = ToolExecutionPortal(closed_loop)
    closed_loop.close()
    with pytest.raises(RuntimeError, match="relay is unavailable"):
        closed_portal.call(lambda: None)
    with pytest.raises(RuntimeError, match="relay is unavailable"):
        asyncio.run(closed_portal.call_method_async(SimpleNamespace(), "method_async"))


def test_worker_context_rejects_sync_prompt_without_native_async_input():
    """A sync interaction cannot be called inline when a tool runs asynchronously."""

    @declare_tool
    def confirm(context: ToolContext) -> bool:
        """Request confirmation through the injected interaction."""
        return context.confirm("Continue?")

    registry = ToolRegistry([confirm])

    try:
        output = asyncio.run(
            registry.call_async(
                "confirm",
                "{}",
                interaction=Mock(spec=Interaction, supports_fresh_host_approval=True),
            )
        )
    finally:
        registry.close()

    assert json.loads(output)["problem"]["code"] == "tool.async_required"


def test_worker_permission_proxy_prefers_native_async_methods():
    """A sync context call uses the permission service's async owner-thread method."""
    owner_thread = get_ident()
    method_threads = []
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None

    async def authorize_async(*_args, **_kwargs):
        method_threads.append(get_ident())
        return SimpleNamespace(decision=Decision.ALLOW)

    permissions.authorize_async = AsyncMock(side_effect=authorize_async)

    @declare_tool
    def inspect_permissions(context: ToolContext) -> bool:
        """Use the sync permission method through its async owner relay."""
        result = context.permission_manager.authorize((), interaction=context.interaction)
        return result.decision is Decision.ALLOW

    registry = ToolRegistry([inspect_permissions], permission_manager=permissions)

    try:
        output = asyncio.run(registry.call_async("inspect_permissions", "{}"))
    finally:
        registry.close()

    assert result_value(output) is True
    assert method_threads == [owner_thread, owner_thread]


def test_worker_command_approval_uses_native_async_permission_and_interaction_methods(tmp_path):
    """A sync command tool can request sandbox approval without blocking the owner loop."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="approve")
    permissions = PermissionManager(tmp_path, interaction=interaction)
    request = SandboxRequest.create(
        source="printf text > output",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version="test",
        deadline=1_000_000_000,
        workspace_id="workspace-1",
    )

    @declare_tool
    def authorize(context: ToolContext) -> bool:
        """Authorize one bound command using its invocation services."""
        return context.permission_manager.authorize_sandboxed_command(
            request,
            tool_id="run_command",
            interaction=context.interaction,
        )

    registry = ToolRegistry([authorize], interaction=interaction, permission_manager=permissions)
    try:
        output = asyncio.run(registry.call_async("authorize", "{}", interaction=interaction))
    finally:
        registry.close()

    assert result_value(output) is True
    interaction.prompt_async.assert_awaited_once()


def test_worker_legacy_read_review_uses_native_async_prompt(tmp_path):
    """A synchronous command tool relays customized legacy scope review to native input."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 1\nlimits:\n  readable_roots: []\n", encoding="utf-8")
    external = tmp_path.parent / "legacy-tool-root"
    external.mkdir(exist_ok=True)
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="review")
    permissions = PermissionManager(tmp_path, interaction=interaction)
    request = SandboxRequest.create(
        source="printf text",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(external,),
        write_roots=(),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version="test",
        deadline=1_000_000_000,
        workspace_id="workspace-1",
    )

    @declare_tool
    def review(context: ToolContext) -> bool:
        """Review an external read scope before the sandbox command is authorized."""
        return context.permission_manager.review_legacy_sandbox_read_limit(
            request, context.interaction
        )

    registry = ToolRegistry([review], interaction=interaction, permission_manager=permissions)
    try:
        output = asyncio.run(registry.call_async("review", "{}", interaction=interaction))
    finally:
        registry.close()

    assert result_value(output) is True
    interaction.prompt_async.assert_awaited_once()
    assert permissions.sandbox_scope_limit(request) is None


def test_worker_proxy_can_await_native_async_method_from_its_worker_thread():
    """An async owner method called from a worker is scheduled on the owner loop."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.confirm_async = AsyncMock(return_value=True)

    @declare_tool
    def confirm(context: ToolContext) -> bool:
        """Request async confirmation through a worker-owned event loop."""
        return asyncio.run(context.interaction.confirm_async("Continue?"))

    registry = ToolRegistry([confirm], interaction=interaction)
    try:
        output = asyncio.run(registry.call_async("confirm", "{}", interaction=interaction))
    finally:
        registry.close()

    assert result_value(output) is True
    interaction.confirm_async.assert_awaited_once_with("Continue?")


def test_definitions_preserve_registration_order():
    """Definitions retain the order in which tools join a registry."""
    registry = ToolRegistry()

    register(registry)

    @declare_tool(name="second", description="Second")
    def _second() -> None:
        pass

    registry.register(_second)

    assert [definition.name for definition in registry.definitions()] == ["calculate", "second"]
    assert registry.names == ["calculate", "second"]


def test_registry_view_exposes_the_live_registry_catalog():
    """A registry view reflects later registrations and preserves the owning registry identity."""
    registry = ToolRegistry()
    view = registry.view()

    assert isinstance(view, ToolRegistryView)
    assert view.registry is registry
    assert view.tools == []
    assert view.names == []
    assert view.registration_problems == ()

    register(registry)

    assert [tool.name for tool in view.tools] == ["calculate"]
    assert view.names == ["calculate"]
    assert view.registration_problems == ()


def test_registry_view_forwards_definitions_and_timed_calls():
    """A registry view forwards definitions and timed calls through both dispatch paths."""
    registry = Mock(spec=ToolRegistry)
    definitions = [Mock()]
    timed_result = ("result", 1.5)
    registry.definitions.return_value = definitions
    registry.call_with_timing.return_value = timed_result
    registry.call_with_timing_async = AsyncMock(return_value=timed_result)
    view = ToolRegistryView(registry)
    execution_started = Mock()
    execution_admission = Mock(return_value=True)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    instructions_manager = Mock(spec=InstructionsManager)
    instructions_manager.virtual_paths.metadata.side_effect = lambda value: value
    instructions_manager.virtual_paths.redact.side_effect = lambda value: value
    permission_manager = _permission_manager_mock()
    cancellation_event = Event()

    assert view.definitions() is definitions
    assert (
        view.call_with_timing(
            "calculate",
            '{"number": 3}',
            interaction=interaction,
            instructions_manager=instructions_manager,
            permission_manager=permission_manager,
            call_id="call-123",
            execution_started=execution_started,
            execution_admission=execution_admission,
            cancellation_event=cancellation_event,
        )
        == timed_result
    )
    registry.definitions.assert_called_once_with()
    registry.call_with_timing.assert_called_once_with(
        "calculate",
        '{"number": 3}',
        interaction=interaction,
        instructions_manager=instructions_manager,
        permission_manager=permission_manager,
        call_id="call-123",
        execution_started=execution_started,
        execution_admission=execution_admission,
        file_observations=None,
        cancellation_event=cancellation_event,
    )
    assert (
        asyncio.run(
            view.call_with_timing_async(
                "calculate",
                '{"number": 3}',
                interaction=interaction,
                instructions_manager=instructions_manager,
                permission_manager=permission_manager,
                call_id="async-call-456",
                execution_started=execution_started,
                execution_admission=execution_admission,
                cancellation_event=cancellation_event,
            )
        )
        == timed_result
    )
    registry.call_with_timing_async.assert_awaited_once_with(
        "calculate",
        '{"number": 3}',
        interaction=interaction,
        instructions_manager=instructions_manager,
        permission_manager=permission_manager,
        call_id="async-call-456",
        execution_started=execution_started,
        execution_admission=execution_admission,
        file_observations=None,
        cancellation_event=cancellation_event,
    )


def test_interaction_property_can_be_replaced_and_cleared():
    """The registry exposes mutable default interaction configuration."""
    first = Mock(spec=Interaction, supports_fresh_host_approval=True)
    second = Mock(spec=Interaction, supports_fresh_host_approval=True)
    registry = ToolRegistry(interaction=first)

    assert registry.interaction is first
    registry.interaction = second
    assert registry.interaction is second
    registry.interaction = None
    assert registry.interaction is None

    replacement = PermissionManager()
    registry.permission_manager = replacement
    assert registry.permission_manager is replacement


def test_registry_exposes_and_replaces_runtime_settings():
    """Runtime settings are readable and replaceable for subsequent tool calls."""
    initial = ToolRuntimeSettings(user_agent="initial-agent")
    replacement = ToolRuntimeSettings(user_agent="replacement-agent")
    registry = ToolRegistry(settings=initial)

    assert registry.settings is initial
    registry.settings = replacement
    assert registry.settings is replacement


def test_registry_passes_replaced_runtime_settings_to_tool_context():
    """Context-aware tools receive the registry's current runtime settings."""
    registry = ToolRegistry(
        interaction=Mock(spec=Interaction, supports_fresh_host_approval=True),
        settings=ToolRuntimeSettings(user_agent="custom-agent"),
    )
    seen = []

    @declare_tool
    def inspect_settings(context: ToolContext) -> str:
        """Return the configured user agent."""
        seen.append(context.settings)
        return context.settings.user_agent

    registry.register(inspect_settings)

    assert result_value(registry.call("inspect_settings", "{}")) == "custom-agent"
    assert seen == [ToolRuntimeSettings(user_agent="custom-agent")]


def test_registry_passes_run_cancellation_to_tool_context():
    """Context-aware tools receive the run-owned cancellation event by identity."""
    cancellation_event = Event()
    registry = ToolRegistry()
    seen = []

    @declare_tool
    def inspect_cancellation(context: ToolContext) -> str:
        """Record whether the active tool can observe run cancellation."""
        seen.append(context.cancellation_event)
        return "ready"

    registry.register(inspect_cancellation)
    registry.call_with_timing("inspect_cancellation", "{}", cancellation_event=cancellation_event)

    assert seen == [cancellation_event]


def test_async_registry_passes_run_cancellation_to_tool_context():
    """Async context-aware tools receive the same run-owned cancellation event."""
    cancellation_event = Event()
    registry = ToolRegistry()
    seen = []

    @declare_tool
    async def inspect_cancellation(context: ToolContext) -> str:
        """Record whether the active async tool can observe run cancellation."""
        seen.append(context.cancellation_event)
        return "ready"

    registry.register(inspect_cancellation)
    asyncio.run(
        registry.call_with_timing_async(
            "inspect_cancellation", "{}", cancellation_event=cancellation_event
        )
    )

    assert seen == [cancellation_event]


def test_registry_shares_mutated_runtime_settings_with_later_calls():
    """In-place configuration changes remain visible to subsequent tool invocations."""
    settings = ToolRuntimeSettings(user_agent="initial")
    registry = ToolRegistry(settings=settings)

    @declare_tool
    def identify(context: ToolContext) -> str:
        """Return the configured user agent."""
        return context.settings.user_agent

    registry.register(identify)
    settings.user_agent = "updated"

    assert result_value(registry.call("identify", "{}")) == "updated"


def test_call_reports_unknown_tools():
    """Synchronous routing serializes an unknown-tool error at the registry boundary."""
    problem = json.loads(ToolRegistry().call("missing", "{}"))["problem"]
    assert problem["code"] == "tool.unknown"
    assert problem["detail"] == "Tool 'missing' is not available."


def test_call_routes_arguments_and_runtime_context():
    """Synchronous routing forwards arguments and builds context from the runtime interaction."""
    registry = ToolRegistry(interaction=Mock(spec=Interaction, supports_fresh_host_approval=True))
    contexts = []

    @declare_tool
    def calculate(context: ToolContext) -> str:
        """Return the current tool name."""
        contexts.append(context)
        return context.tool_name

    registry.register(calculate)
    runtime = Mock(spec=Interaction, supports_fresh_host_approval=True)
    manager = Mock(spec=InstructionsManager)
    manager.agents_filenames = ("AGENTS.md",)
    manager.virtual_paths.metadata.side_effect = lambda value: value
    manager.virtual_paths.redact.side_effect = lambda value: value
    execution_started = Mock()

    assert (
        result_value(
            registry.call(
                "calculate",
                "{}",
                interaction=runtime,
                instructions_manager=manager,
                call_id="call-123",
                execution_started=execution_started,
            )
        )
        == "calculate"
    )
    context = contexts[0]
    assert context.interaction is runtime
    assert context.tool_name == "calculate"
    assert context.instructions_manager is manager
    assert context.call_id == "call-123"
    execution_started.assert_called_once_with()


def test_call_uses_default_or_no_context():
    """Synchronous routing retains context when no interaction capability exists."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    registry = ToolRegistry(interaction=interaction)
    seen = []

    @declare_tool
    def calculate(context: ToolContext) -> str:
        """Record the supplied context."""
        seen.append(context)
        return "result"

    registry.register(calculate)

    registry.call("calculate", "{}")
    assert seen[0].interaction is interaction
    registry.interaction = None
    assert result_value(registry.call("calculate", "{}")) == "result"
    assert seen[1].interaction is None


def test_headless_authorized_tool_receives_execution_metadata():
    """Authorized headless calls receive plans and invocation identity without a UI."""
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None
    permissions.authorize.return_value = SimpleNamespace(decision=Decision.ALLOW)
    registry = ToolRegistry(permission_manager=permissions)
    seen = []

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def inspect(context: ToolContext) -> str:
        """Inspect authorized execution metadata."""
        seen.append(context)
        return context.call_id or "missing"

    registry.register(inspect)

    assert result_value(registry.call("inspect", "{}", call_id="headless-call")) == (
        "headless-call"
    )
    assert seen[0].interaction is None
    assert seen[0].operations[0].action is Action.SESSION_MUTATE
    permissions.authorize.assert_called_once()


def test_headless_authorization_denial_prevents_execution():
    """Removing the UI never bypasses centralized authorization."""
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None
    permissions.authorize.return_value = SimpleNamespace(
        decision=Decision.DENY,
        reason="headless policy denied the operation",
    )
    registry = ToolRegistry(permission_manager=permissions)
    called = []

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def mutate(context: ToolContext) -> None:
        """Record an authorized mutation."""
        called.append(context)

    registry.register(mutate)

    problem = json.loads(registry.call("mutate", "{}"))["problem"]

    assert problem["code"] == "tool.denied"
    assert called == []
    permissions.authorize.assert_called_once()


def test_headless_interaction_requirement_is_structured():
    """A headless tool that requests UI returns the standard interaction problem."""
    registry = ToolRegistry()

    @declare_tool
    def confirm(context: ToolContext) -> bool:
        """Require an explicit user confirmation."""
        return context.confirm("Continue?")

    registry.register(confirm)

    problem = json.loads(registry.call("confirm", "{}"))["problem"]

    assert problem["code"] == "tool.interaction_unavailable"
    assert problem["title"] == "Interaction unavailable"


def test_call_with_timing_excludes_permission_confirmation(monkeypatch):
    """Tool timing begins after authorization and uses the policy manager's recorder."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    recorder = Mock()
    permissions = PermissionManager(interaction=interaction, recorder=recorder)
    registry = ToolRegistry(interaction=interaction, permission_manager=permissions)

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)
    clock = Mock(side_effect=[10.0, 12.0])
    monkeypatch.setattr(tool_registry_module, "perf_counter", clock)

    output, duration = registry.call_with_timing("calculate", '{"number": 3}')

    assert result_value(output) == 3
    assert duration == 2
    interaction.prompt.assert_called_once()
    recorder.record_authorization.assert_called_once()


def test_call_command_parses_model_parameters_before_shared_dispatch():
    """Command routing validates parameters while bypassing the permission policy."""
    permissions = _permission_manager_mock()
    registry = ToolRegistry(
        interaction=Mock(spec=Interaction, supports_fresh_host_approval=True),
        permission_manager=permissions,
    )

    @declare_tool
    def describe(count: int, label: str = "default") -> dict:
        """Describe parsed values."""
        return {"count": count, "label": label}

    @declare_tool
    def has_no_policy(context: ToolContext) -> bool:
        """Report successful user-command context construction."""
        return context.tool_name == "has_no_policy"

    registry.register(describe)
    registry.register(has_no_policy)

    assert result_value(registry.command("describe", ("3", "label=two words")).output) == {
        "count": 3,
        "label": "two words",
    }
    assert result_value(registry.command("has_no_policy", ()).output) is True
    permissions.authorize.assert_not_called()


def test_call_command_retains_dynamic_result_presentation():
    """Command dispatch returns the presentation selected for canonical arguments and output."""
    presentation = ToolResultPresentationSpec(kind=ToolResultPresentation.TABLE)

    @declare_tool(result_presentation=lambda arguments, result: presentation)
    def describe(count: int) -> dict:
        """Describe parsed values."""
        return {"count": count}

    execution = ToolRegistry([describe]).command("describe", ("3",))

    assert result_value(execution.output) == {"count": 3}
    assert execution.presentation is presentation


def test_call_command_resolves_virtual_paths_before_execution(tmp_path):
    """Direct tool commands use the same virtual-path boundary as model tool calls."""
    target = tmp_path / "notes.txt"
    target.write_text("virtual content", encoding="utf-8")
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "temporary")
    )

    execution = ToolRegistry([read_text_file]).command(
        "read_text_file", ("$WORKSPACE_ROOT/notes.txt",), instructions_manager=instructions
    )

    assert result_value(execution.output)["content"] == "virtual content"
    assert result_value(execution.output)["path"] == "$WORKSPACE_ROOT/notes.txt"


def test_call_command_reports_unknown_tools_and_invalid_parameters():
    """Command routing serializes lookup, binding, validation, and planning failures."""
    registry = ToolRegistry()
    register(registry)

    def invalid_planner(_arguments):
        """Reject planning to verify the user-command error boundary."""
        raise ValueError("invalid operation plan")

    @declare_tool(actions={Action.SESSION_MUTATE}, operation_planner=invalid_planner)
    def invalid_plan() -> None:
        """Expose an intentionally invalid operation planner."""

    registry.register(invalid_plan)

    assert "tool.unknown" in registry.command("missing", ()).output
    assert "tool.invalid_arguments" in registry.command("calculate", ("unknown=1",)).output
    assert "argument_binding" in registry.command("calculate", ("unknown=1",)).output
    assert "tool.invalid_arguments" in registry.command("calculate", ("not-an-integer",)).output
    assert "tool.planning_failed" in registry.command("invalid_plan", ()).output


def test_call_async_reports_unknown_tools():
    """Asynchronous routing serializes an unknown-tool error at the registry boundary."""
    problem = json.loads(asyncio.run(ToolRegistry().call_async("missing", "{}")))["problem"]
    assert problem["code"] == "tool.unknown"


def test_call_async_routes_arguments_and_context():
    """Run synchronous execution intent off-loop before invoking the context-aware tool."""
    registry = ToolRegistry()
    contexts = []
    callback_threads = []
    events = []
    owner_thread = get_ident()

    @declare_tool
    async def calculate(context: ToolContext) -> str:
        """Return the supplied context tool name."""
        contexts.append(context)
        events.append("tool")
        return context.tool_name

    registry.register(calculate)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)

    def execution_started():
        """Record the thread that persists tool execution intent."""
        callback_threads.append(get_ident())
        events.append("started")

    try:
        result = asyncio.run(
            registry.call_async(
                "calculate",
                "{}",
                interaction=interaction,
                call_id="async-call",
                execution_started=execution_started,
            )
        )
    finally:
        registry.close()

    assert result_value(result) == "calculate"
    assert contexts[0].interaction is interaction
    assert contexts[0].call_id == "async-call"
    assert callback_threads and callback_threads[0] != owner_thread
    assert events == ["started", "tool"]


def test_call_async_returns_validation_and_permission_denials_before_invocation():
    """Async dispatch stops before invocation for invalid and rejected calls."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async.return_value = ApprovalChoice.DENY
    registry = ToolRegistry(interaction=interaction)

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)

    invalid = asyncio.run(registry.call_async("calculate", "bad"))
    denied = asyncio.run(registry.call_async("calculate", '{"number": 1}'))

    assert "tool.invalid_arguments" in invalid
    assert "tool.denied" in denied


def test_sync_and_async_dispatch_report_operation_planning_failures():
    """A failed canonical operation plan prevents both dispatch paths from invoking a tool."""
    calls = []

    def calculate() -> str:
        """Return an unreachable result."""
        calls.append(True)
        return "unreachable"

    def fail_plan(_arguments):
        raise ValueError("cannot canonicalize target")

    registry = ToolRegistry()
    registry.register(calculate, operation_planner=fail_plan)

    sync_result, sync_duration = registry.call_with_timing("calculate", "{}")
    async_result, async_duration = asyncio.run(registry.call_with_timing_async("calculate", "{}"))

    assert "tool.planning_failed" in sync_result
    assert "tool.planning_failed" in async_result
    assert sync_duration == async_duration == 0
    assert calls == []


def test_unavailable_command_paths_do_not_disclose_host_paths():
    """All dispatch paths return a safe problem when command roots cannot be captured."""

    class UnavailableInstructions:
        """Represent an unavailable instruction path snapshot."""

        @property
        def virtual_paths(self):
            """Raise a path-bearing filesystem failure."""
            raise OSError("/private/sensitive/path")

    instructions = UnavailableInstructions()
    registry = ToolRegistry([run_command])
    sync = registry.call("run_command", '{"command": ":"}', instructions_manager=instructions)
    asynchronous = asyncio.run(
        registry.call_async("run_command", '{"command": ":"}', instructions_manager=instructions)
    )
    direct = registry.command("run_command", (":",), instructions_manager=instructions).output
    for output in (sync, asynchronous, direct):
        assert json.loads(output)["problem"]["code"] == "tool.planning_failed"
        assert "/private/sensitive/path" not in output

    @declare_tool
    def ordinary() -> None:
        """Represent a tool without command execution authority."""

    registry.register(ordinary)
    with pytest.raises(OSError, match="/private/sensitive/path"):
        asyncio.run(registry.call_async("ordinary", "{}", instructions_manager=instructions))


@pytest.mark.parametrize(
    "failure",
    [
        IsADirectoryError("is a directory"),
        FileNotFoundError("missing"),
        NotADirectoryError("not a directory"),
        PermissionError("permission denied"),
    ],
)
def test_all_dispatch_paths_normalize_expected_filesystem_planning_failures(failure):
    """Expected filesystem planning errors remain structured across every dispatcher."""

    def fail_plan(_arguments):
        raise failure

    @declare_tool(operation_planner=fail_plan)
    def calculate() -> None:
        """Expose one expected filesystem planning failure."""

    registry = ToolRegistry([calculate])

    sync = json.loads(registry.call("calculate", "{}"))["problem"]
    asynchronous = json.loads(asyncio.run(registry.call_async("calculate", "{}")))["problem"]
    command = json.loads(registry.command("calculate", ()).output)["problem"]

    assert sync["code"] == asynchronous["code"] == command["code"] == "tool.planning_failed"
    assert sync["detail"] == asynchronous["detail"] == command["detail"] == str(failure)


def test_command_dispatch_resolves_planning_continuations():
    """Direct command dispatch executes a phased planner before invoking its tool."""

    def initial_plan(arguments):
        return OperationPlan(
            arguments=arguments,
            continuation=lambda: OperationPlan(arguments={"number": 7}),
        )

    @declare_tool(operation_planner=initial_plan)
    def calculate(number: int = 1) -> int:
        """Return the fully planned value."""
        return number

    registry = ToolRegistry([calculate])

    assert result_value(registry.command("calculate", ()).output) == 7


def test_all_dispatch_paths_preserve_structured_operation_planning_problems():
    """Structured planning failures retain their problem contract across every dispatcher."""
    planning_problem = Problem(
        code="test.specific",
        title="Specific planning failure",
        detail="Use different input.",
        operation="calculate",
    )

    def fail_plan(_arguments):
        raise ProblemException(planning_problem)

    @declare_tool(actions={Action.SESSION_MUTATE}, operation_planner=fail_plan)
    def calculate() -> None:
        """Expose a structured planning failure."""

    registry = ToolRegistry([calculate])

    sync = json.loads(registry.call("calculate", "{}"))["problem"]
    asynchronous = json.loads(asyncio.run(registry.call_async("calculate", "{}")))["problem"]
    command = json.loads(registry.command("calculate", ()).output)["problem"]

    assert sync["code"] == asynchronous["code"] == command["code"] == "test.specific"


def test_call_with_timing_async_excludes_permission_confirmation(monkeypatch):
    """Async tool timing spans awaited execution but excludes permission confirmation."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async.return_value = ApprovalChoice.ONCE
    recorder = Mock()
    permissions = PermissionManager(interaction=interaction, recorder=recorder)
    registry = ToolRegistry(interaction=interaction, permission_manager=permissions)

    @declare_tool(
        actions={Action.NETWORK_REQUEST},
        operation_planner=planner_for(Action.NETWORK_REQUEST),
    )
    async def calculate(number: int) -> int:
        """Calculate a number asynchronously."""
        return number

    registry.register(calculate)
    clock = Mock(side_effect=[10.0, 12.0])
    monkeypatch.setattr(tool_registry_module, "perf_counter", clock)

    output, duration = asyncio.run(registry.call_with_timing_async("calculate", '{"number": 3}'))

    assert result_value(output) == 3
    assert duration == 2
    interaction.prompt_async.assert_awaited_once()
    recorder.record_authorization.assert_called_once()


def test_async_execution_started_callback_is_awaited_before_tool_invocation():
    """Async execution intent settles before an effectful tool callback begins."""
    registry = ToolRegistry()
    events = []

    @declare_tool()
    async def calculate() -> str:
        """Record the tool invocation after execution intent."""
        events.append("tool")
        return "done"

    registry.register(calculate)

    async def execution_started():
        """Record execution intent at the async owner boundary."""
        events.append("started")

    result = asyncio.run(
        registry.call_async(
            "calculate",
            "{}",
            execution_started=execution_started,
        )
    )

    assert result_value(result) == "done"
    assert events == ["started", "tool"]


def test_cancelled_async_execution_started_callback_cannot_admit_tool():
    """An intent callback cannot admit tool effects after swallowing cancellation."""
    registry = ToolRegistry()
    effects = []

    @declare_tool()
    async def calculate() -> str:
        """Record any incorrectly admitted execution."""
        effects.append("tool")
        return "done"

    registry.register(calculate)

    async def scenario():
        started = asyncio.Event()

        async def execution_started():
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return

        task = asyncio.create_task(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert effects == []

    asyncio.run(scenario())


def test_cancelled_native_async_tool_retains_settled_receipt():
    """A tool that completes after cancellation reports its known effect receipt."""
    registry = ToolRegistry()
    effects = []

    async def scenario():
        started = asyncio.Event()

        @declare_tool()
        async def calculate() -> str:
            """Complete an effect despite interruption."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                effects.append("effect")
                return "confirmed"

        registry.register(calculate)
        task = asyncio.create_task(registry.call_async("calculate", "{}"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert result_value(caught.value.output) == "confirmed"
        assert effects == ["effect"]

    asyncio.run(scenario())


def test_async_dispatch_keeps_event_loop_responsive_during_approval():
    """Async tool dispatch yields while native approval is pending and runs only after it."""
    approval_started = asyncio.Event()
    continue_approval = asyncio.Event()
    executed = []

    async def approve(_message, **_kwargs):
        approval_started.set()
        await continue_approval.wait()
        return ApprovalChoice.ONCE

    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(side_effect=approve)
    permissions = PermissionManager(interaction=interaction)

    @declare_tool(
        actions={Action.NETWORK_REQUEST},
        operation_planner=planner_for(Action.NETWORK_REQUEST),
    )
    async def calculate(number: int) -> int:
        """Record execution after the approval boundary."""
        executed.append(number)
        return number

    registry = ToolRegistry([calculate], interaction=interaction, permission_manager=permissions)

    async def dispatch():
        task = asyncio.create_task(registry.call_with_timing_async("calculate", '{"number": 3}'))
        await approval_started.wait()
        await deliver_callbacks()
        assert not task.done()
        assert executed == []
        continue_approval.set()
        return await task

    output, duration = asyncio.run(dispatch())

    assert result_value(output) == 3
    assert duration >= 0
    assert executed == [3]
    interaction.prompt_async.assert_awaited_once()


def test_async_planning_stops_at_operation_boundaries():
    """Async planning rejects a boundary violation before requesting permission."""
    permissions = _permission_manager_mock()
    permissions.check_boundaries_async.return_value = SimpleNamespace(
        reason="Outside allowed roots."
    )

    @declare_tool(
        actions={Action.NETWORK_REQUEST},
        operation_planner=planner_for(Action.NETWORK_REQUEST),
    )
    async def calculate(number: int) -> int:
        """Return a value after the boundary check."""
        return number

    registry = ToolRegistry([calculate], permission_manager=permissions)
    output = asyncio.run(registry.call_async("calculate", '{"number": 3}'))

    assert json.loads(output)["problem"]["code"] == "tool.denied"
    permissions.authorize_async.assert_not_awaited()


@pytest.mark.parametrize(
    ("has_execution_service", "operations", "expected"),
    [
        (False, (), "sandbox.unavailable"),
        (
            True,
            (
                Operation(
                    tool_id="execute",
                    action=Action.SESSION_MUTATE,
                    target=SessionTarget(identifier="state"),
                ),
            ),
            "sandbox.unavailable",
        ),
        (
            True,
            (
                Operation(
                    tool_id="execute",
                    action=Action.PROCESS_EXECUTE,
                    target=ProcessTarget(
                        argv=("safe-command",),
                        cwd="$WORKSPACE_ROOT",
                        boundary=ProcessBoundary.SANDBOXED,
                    ),
                ),
            ),
            None,
        ),
    ],
)
def test_async_execution_authorization_keeps_sandbox_boundary(
    has_execution_service, operations, expected
):
    """Async dispatch only accepts operations authorized for the sandbox boundary."""
    execution_service = Mock() if has_execution_service else None

    def process_plan(arguments):
        """Return the arguments unchanged for direct authorization checks."""
        return OperationPlan(arguments=arguments)

    @declare_tool(
        actions={Action.PROCESS_EXECUTE},
        operation_planner=process_plan,
        execution_authorization=True,
    )
    async def execute(context: ToolContext) -> str:
        """Stand in for a sandbox-authorized tool."""
        return "ok"

    registry = ToolRegistry([execute])
    registry._execution_service = execution_service
    result = asyncio.run(
        registry._authorize_async(
            registry._tools["execute"],
            operations,
            None,
            PermissionManager(),
            None,
        )
    )

    assert (json.loads(result)["problem"]["code"] if result is not None else None) == expected


@pytest.mark.parametrize(
    ("name", "arguments", "error"),
    [
        ("missing", "{}", "tool.unknown"),
        ("calculate", "bad", "tool.invalid_arguments"),
        ("calculate", '{"number": 1}', "tool.denied"),
    ],
)
def test_call_with_timing_async_returns_zero_before_invocation(name, arguments, error, monkeypatch):
    """Async timing remains zero when lookup, validation, or authorization stops dispatch."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async.return_value = ApprovalChoice.DENY
    registry = ToolRegistry(interaction=interaction)

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)
    clock = Mock()
    monkeypatch.setattr(tool_registry_module, "perf_counter", clock)

    output, duration = asyncio.run(registry.call_with_timing_async(name, arguments))

    assert error in output
    assert duration == 0
    clock.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_virtual_file_paths_resolve_before_authorization_and_return_virtual_metadata(
    tmp_path, asynchronous
):
    """Virtual paths preserve file access while approvals receive the actual execution target."""
    path = tmp_path / "é notes.txt"
    path.write_text("useful content", encoding="utf-8")
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None
    permissions.authorize.return_value = SimpleNamespace(decision=Decision.ALLOW)
    permissions.authorize_async.return_value = SimpleNamespace(decision=Decision.ALLOW)
    registry = ToolRegistry([read_text_file], permission_manager=permissions)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    kwargs = {"instructions_manager": instructions}
    args = json.dumps({"path": "$WORKSPACE_ROOT/é notes.txt"})
    output, _ = (
        asyncio.run(registry.call_with_timing_async("read_text_file", args, **kwargs))
        if asynchronous
        else registry.call_with_timing("read_text_file", args, **kwargs)
    )
    result = result_value(output)
    assert result["content"] == "useful content"
    assert result["path"] == "$WORKSPACE_ROOT/é notes.txt"
    authorization = permissions.authorize_async if asynchronous else permissions.authorize
    assert str(path) in str(authorization.call_args.args[0])


def test_observed_file_scope_does_not_change_command_working_directory(tmp_path, monkeypatch):
    """A file observation cannot change the cwd used by a later process."""
    nested = tmp_path / "src" / "loop"
    nested.mkdir(parents=True)
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    instructions.observe_path(nested, directory=True)
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None
    permissions.authorize_sandboxed_command.side_effect = _approve_sandbox_request
    permissions.sandbox_policy_revision_current.return_value = True
    permissions.authorize.return_value = SimpleNamespace(decision=Decision.ALLOW)
    permissions.analyze_command.return_value = CommandAnalysis((), False, ())
    permissions.protected_sandbox_paths.return_value = protected_workspace_paths()
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=permissions,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout=f"{tmp_path}\n"
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": "pwd", "cwd": "."}),
        instructions_manager=instructions,
    )

    assert result_value(output)["stdout"]["content"].strip() == str(tmp_path)
    request = permissions.authorize_sandboxed_command.call_args.args[0]
    assert request.cwd == tmp_path


def test_standalone_command_refuses_unmanaged_scratch(tmp_path):
    """A standalone continuation cannot rely on caller-owned scratch without a lease."""
    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir(mode=0o700)
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, unmanaged))
    permissions = PermissionManager(tmp_path)
    backend = Mock()
    service = CommandExecutionService(backend, Mock())
    registry = ToolRegistry(
        [run_command], permission_manager=permissions, execution_service=service
    )
    try:
        output = registry.call(
            "run_command", json.dumps({"command": ":"}), instructions_manager=instructions
        )
        assert json.loads(output)["problem"]["code"] == "command_job.unavailable"
        backend.run.assert_not_called()
    finally:
        registry.close()
        permissions.close()


def test_command_workspace_prefix_selects_the_workspace_root(tmp_path, monkeypatch):
    """The canonical workspace prefix executes in the workspace root."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    permissions = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=permissions,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout=f"{tmp_path}\n"
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": "printf cwd", "cwd": "$WORKSPACE_ROOT"}),
        interaction=interaction,
        instructions_manager=instructions,
    )

    assert result_value(output)["stdout"]["content"].strip() == str(tmp_path)
    interaction.prompt.assert_not_called()
    assert backend.run.call_args.args[0].cwd == tmp_path


def test_command_virtual_workspace_read_roots_do_not_expand_authority(tmp_path, monkeypatch):
    """A workspace read root remains within default authority and needs no prompt."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    nested = tmp_path / "src"
    nested.mkdir()
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    registry = ToolRegistry(
        [run_command],
        interaction=interaction,
        permission_manager=PermissionManager(tmp_path, interaction=interaction),
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps(
            {
                "command": "cd $WORKSPACE_ROOT && git diff --cached --stat",
                "cwd": "$WORKSPACE_ROOT",
                "read_only": True,
                "read_roots": ["$WORKSPACE_ROOT", "$WORKSPACE_ROOT/src"],
            }
        ),
        instructions_manager=instructions,
    )

    assert result_value(output)["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.cwd == tmp_path
    assert request.read_roots == ()
    assert request.execution_source == request.source
    interaction.prompt.assert_not_called()


def test_denied_command_virtualizes_offer_and_preserves_cached_output(tmp_path, monkeypatch):
    """A public denial virtualizes diagnostics while preserving command output."""
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, None))
    interaction = _approving_interaction()
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    registry = ToolRegistry(
        [run_command],
        interaction=interaction,
        permission_manager=permissions,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.DENIED,
        stdout=(str(tmp_path) + "/output\n")
        * (MAX_TOOL_RESULT_BYTES // len("$WORKSPACE_ROOT/output\n") + 1),
        detail=f"Denied {tmp_path}/.ssh/fake",
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    source = f"cat {tmp_path}/.ssh/fake $WORKSPACE_ROOT/file"

    output = registry.call(
        "run_command",
        json.dumps({"command": source, "cwd": "$WORKSPACE_ROOT", "read_only": True}),
        interaction=interaction,
        instructions_manager=instructions,
    )

    problem = json.loads(output)["problem"]
    assert problem["code"] == "sandbox.denied", problem
    assert "$WORKSPACE_ROOT/.ssh/fake" in problem["metadata"]["host_offer"]
    assert any("$WORKSPACE_ROOT" in call.args[0] for call in interaction.info.call_args_list)
    handle = problem["metadata"]["stdout"]["handle"]
    path, _ = cached_path(handle)
    assert path.read_text() == backend.run.return_value.stdout


def test_command_accepts_virtual_workspace_cwd(tmp_path, monkeypatch):
    """The command tool resolves a virtual cwd before native authorization."""
    nested = tmp_path / "src"
    nested.mkdir()
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    permissions = PermissionManager(tmp_path)
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=permissions,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": "pwd", "cwd": "$WORKSPACE_ROOT/src"}),
        instructions_manager=instructions,
    )
    assert result_value(output)["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].cwd == nested


def test_command_output_and_failure_details_preserve_workspace_paths(tmp_path, monkeypatch):
    """Failed command streams preserve physical paths in captured text."""
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stdout=f"{tmp_path}/one\n",
        stderr=f"failed at {tmp_path}/two\n",
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": "pwd", "read_only": True}),
        instructions_manager=instructions,
    )

    problem = json.loads(output)["problem"]
    assert problem["metadata"]["stdout"]["content"] == f"{tmp_path}/one\n"
    assert problem["metadata"]["stderr"]["content"] == f"failed at {tmp_path}/two\n"


def test_command_sandbox_failure_preserves_workspace_paths(tmp_path, monkeypatch):
    """Sandbox failures retain usable workspace paths without exposing environment secrets."""
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.UNAVAILABLE, detail=f"failed to inspect {tmp_path}/source.py"
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": "pwd", "read_only": True}),
        instructions_manager=instructions,
    )

    assert str(tmp_path) not in output
    assert json.loads(output)["problem"]["detail"] == (
        "failed to inspect $WORKSPACE_ROOT/source.py"
    )
    assert "PATH=" not in json.loads(output)["problem"]["metadata"]["host_offer"]


def test_command_preserves_source_and_binds_concrete_roots(tmp_path, monkeypatch):
    """Default sandbox access binds concrete roots and leaves source bytes unchanged."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    scratch = tmp_path.with_name(tmp_path.name + "-scratch")
    scratch.mkdir(mode=0o700)
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, scratch))
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    observed = []

    def execute(request, **_kwargs):
        """Inspect bound roots and unchanged source at the execution boundary."""
        observed.append(request)
        assert request.paths_are_current()
        assert request.source == 'cat $WORKSPACE_ROOT/a "$TMPDIR/b"'
        assert request.execution_source == request.source
        assert request.scratch_root == scratch
        assert request.shell_environment()["TMPDIR"] == str(scratch)
        return CommandProcessResult(
            SandboxOutcome.COMPLETED,
            exit_code=0,
            stdout=f"{tmp_path}/a {scratch}/b",
        )

    backend = Mock()
    backend.run.side_effect = execute
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    scratch = _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": 'cat $WORKSPACE_ROOT/a "$TMPDIR/b"'}),
        interaction=interaction,
        instructions_manager=instructions,
    )

    assert result_value(output)["stdout"]["content"] == f"{tmp_path}/a {scratch}/b"
    interaction.prompt.assert_not_called()
    assert observed[0].paths_are_current()


def test_command_ignores_virtual_root_substrings(tmp_path, monkeypatch):
    """A path-like substring cannot require an alias for an unrelated root."""
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": "echo /tmp-old", "read_only": True}),
        instructions_manager=instructions,
    )

    assert result_value(output)["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.execution_source == "echo /tmp-old"


def test_command_host_retry_preserves_source(tmp_path, monkeypatch):
    """One-off host retry retains the exact approved source and process paths."""
    (tmp_path / "scratch").mkdir(mode=0o700)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    registry = ToolRegistry(
        [run_command],
        interaction=_approving_interaction(),
        permission_manager=permissions,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.UNAVAILABLE, detail="profile unavailable"
    )
    host = Mock()

    def execute(request, **_kwargs):
        """Observe the exact source while the separately approved host run is active."""
        assert request.source == "cat $WORKSPACE_ROOT"
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, stdout=request.source)

    host.run_host_command.side_effect = execute
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    _use_managed_command_scratch(registry, instructions)
    output = registry.call(
        "run_command",
        json.dumps({"command": "cat $WORKSPACE_ROOT"}),
        interaction=interaction,
        instructions_manager=instructions,
    )

    assert result_value(output)["boundary"] == "host"
    assert result_value(output)["stdout"]["content"] == "cat $WORKSPACE_ROOT"
    warning = interaction.info.call_args_list[-1].args[0]
    assert 'Command: "cat $WORKSPACE_ROOT"' in warning
    assert "Reason: The sandbox was unavailable." in warning
    assert str(tmp_path) not in warning
    assert "Execution source:" not in warning
    assert "Virtual path mappings:" not in warning
    assert "Environment:" not in warning
    assert str(tmp_path) not in output


def test_delete_path_accepts_a_virtual_workspace_path(tmp_path):
    """Deletion resolves its declared virtual path before planning and execution."""
    target = tmp_path / "obsolete.txt"
    target.write_text("obsolete", encoding="utf-8")
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    registry = ToolRegistry([delete_path], permission_manager=PermissionManager(tmp_path))

    output = registry.call(
        "delete_path",
        json.dumps({"path": "$WORKSPACE_ROOT/obsolete.txt"}),
        interaction=Mock(spec=Interaction, prompt=Mock(return_value=ApprovalChoice.ONCE)),
        instructions_manager=instructions,
    )

    assert result_value(output) == "Successfully deleted path '$WORKSPACE_ROOT/obsolete.txt'."
    assert not target.exists()


def test_delete_path_returns_the_canonical_workspace_path(tmp_path):
    """Canonical file paths resolve and return the same workspace prefix."""
    target = tmp_path / "obsolete.txt"
    target.write_text("obsolete", encoding="utf-8")
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "temporary")
    )
    registry = ToolRegistry([delete_path], permission_manager=PermissionManager(tmp_path))

    output = registry.call(
        "delete_path",
        json.dumps({"path": "$WORKSPACE_ROOT/obsolete.txt"}),
        interaction=Mock(spec=Interaction, prompt=Mock(return_value=ApprovalChoice.ONCE)),
        instructions_manager=instructions,
    )

    assert result_value(output) == "Successfully deleted path '$WORKSPACE_ROOT/obsolete.txt'."
    assert str(tmp_path) not in output
    assert not target.exists()


def test_model_result_returns_non_json_output_unchanged(tmp_path):
    """Non-JSON plain text tool results return unchanged without raising decode errors."""
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    raw_output = "Plain text output with path /tmp/dir {"
    output = ToolRegistry._model_result(
        raw_output, instructions, Tool(lambda: None, result_path_fields=(("result", "path"),))
    )
    assert output == raw_output


def test_execution_capability_requires_a_service(tmp_path):
    """A renamed native tool cannot run when no executor was injected."""
    registry = ToolRegistry([ToolRegistration(run_command, name="other_command")])

    output = registry.call("other_command", json.dumps({"command": "pwd", "cwd": str(tmp_path)}))

    assert json.loads(output)["problem"]["code"] == "sandbox.unavailable"


def test_execution_capability_rejects_an_unbound_plan(tmp_path):
    """An execution declaration cannot route a host-bound process through native approval."""

    def host_plan(arguments):
        """Return an inadmissible host process for this declared capability."""
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.PROCESS_EXECUTE,
                    target=ProcessTarget(
                        argv=("/bin/sh", "-c", str(arguments["command"])),
                        cwd=str(tmp_path),
                        boundary=ProcessBoundary.HOST,
                    ),
                ),
            ),
        )

    def second_command(context: ToolContext, command: str) -> object:
        """Route through the declared native service if authorized."""
        return context.execution_service.run_command(context, command)

    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None
    registry = ToolRegistry(
        [
            ToolRegistration(
                second_command,
                name="second_command",
                actions=frozenset({Action.PROCESS_EXECUTE}),
                operation_planner=host_plan,
                execution_authorization=True,
            )
        ],
        permission_manager=permissions,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )

    output = registry.call("second_command", json.dumps({"command": "pwd"}))

    assert json.loads(output)["problem"]["code"] == "sandbox.unavailable"


def test_execution_capability_validates_registration():
    """A declaration without a context-aware process planner is refused."""

    def invalid() -> str:
        """Return a placeholder value."""
        return "invalid"

    with pytest.raises(ToolRegistrationError, match="context-aware process planner"):
        ToolRegistry(
            [
                ToolRegistration(
                    invalid,
                    actions=frozenset({Action.PROCESS_EXECUTE}),
                    operation_planner=lambda args: OperationPlan(arguments=args),
                    execution_authorization=True,
                )
            ]
        )


@pytest.mark.parametrize("second_tool", [False, True])
def test_execution_capability_guards_renamed_and_second_tools(tmp_path, second_tool):
    """Every declared execution tool passes its registered identity into native approval."""

    def native_plan(arguments):
        """Bind one sandbox process for the second test tool."""
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.PROCESS_EXECUTE,
                    target=ProcessTarget(
                        argv=("/bin/sh", "-c", str(arguments["command"])),
                        cwd=str(tmp_path),
                        boundary=ProcessBoundary.SANDBOXED,
                    ),
                ),
            ),
        )

    def other_native(context: ToolContext, command: str) -> object:
        """Ask the registered facade to run the test shell source."""
        return context.execution_service.run_command(context, command)

    declaration = (
        ToolRegistration(
            other_native,
            name="second_command",
            actions=frozenset({Action.PROCESS_EXECUTE}),
            operation_planner=native_plan,
            execution_authorization=True,
        )
        if second_tool
        else ToolRegistration(run_command, name="renamed_command")
    )
    name = "second_command" if second_tool else "renamed_command"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    permissions = _permission_manager_mock()
    permissions.check_boundaries.return_value = None
    permissions.authorize_sandboxed_command.side_effect = _approve_sandbox_request
    permissions.sandbox_policy_revision_current.return_value = True
    permissions.analyze_command.return_value = CommandAnalysis((), False, ())
    permissions.protected_sandbox_paths.return_value = protected_workspace_paths()
    registry = ToolRegistry(
        [declaration],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, Mock()),
    )

    output = registry.call(
        name,
        json.dumps(
            {"command": "pwd", "cwd": str(tmp_path)} if not second_tool else {"command": "pwd"}
        ),
    )

    assert result_value(output)["boundary"] == "sandbox"
    assert permissions.authorize_sandboxed_command.call_args.kwargs["tool_id"] == name
    backend.run.assert_called_once()


def test_renamed_executable_lookup_receives_declared_service():
    """Execution service injection follows registration metadata after a lookup is renamed."""
    service = Mock()
    service.resolve_executable.return_value = {"name": "git", "status": "installed_on_host"}
    registry = ToolRegistry(
        [ToolRegistration(resolve_executable, name="find_tool")],
        execution_service=service,
    )

    output = registry.call("find_tool", json.dumps({"name": "git"}))

    assert result_value(output)["status"] == "installed_on_host"
    assert service.resolve_executable.call_args.args[0].tool_name == "find_tool"


@pytest.mark.parametrize("kind", ["started", "unstarted", "coroutine", "borrowed"])
def test_cancelled_async_callback_settles_nested_work(kind):
    """Nested work returned while cancellation settles cannot outlive the callback."""
    registry = ToolRegistry()
    events = []

    @declare_tool()
    async def calculate() -> str:
        """Detect accidental admission of the tool."""
        events.append("tool")
        return "done"

    registry.register(calculate)

    async def scenario():
        """Cancel a callback which returns nested work during cleanup."""
        started = asyncio.Event()
        release = asyncio.Event()
        coroutine = None

        async def child():
            """Remain active until cancellation or explicit release."""
            try:
                await release.wait()
                events.append("child")
            except asyncio.CancelledError:
                events.append("cancelled")

        async def execution_started():
            """Return different nested work after receiving cancellation."""
            nonlocal coroutine
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                coroutine = child()
                if kind == "coroutine":
                    return coroutine
                if kind == "borrowed":
                    return asyncio.create_task(coroutine)
                work = OwnedToolTask(coroutine)
                if kind == "started":
                    work.start(asyncio.get_running_loop())
                return work

        task = asyncio.create_task(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        await started.wait()
        task.cancel()
        await deliver_callbacks()
        if kind == "borrowed":
            assert not task.done()
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "tool" not in events
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED

    import inspect

    asyncio.run(scenario())


def test_cancelled_returned_callback_task_settles_nested_owned_work():
    """A callback's settled task cannot leave a nested owned task running."""
    registry = ToolRegistry()
    effects = []

    @declare_tool()
    async def calculate() -> str:
        """Detect incorrectly admitted effects."""
        effects.append("tool")
        return "done"

    registry.register(calculate)

    async def scenario():
        """Interrupt a returned task which returns another started owned handle."""
        started = asyncio.Event()
        nested = []

        async def child():
            """Wait until canceled by the invocation owner."""
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                effects.append("child cancelled")

        async def outer():
            """Return a started nested task after interruption."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                work = OwnedToolTask(child())
                nested.append(work.start(asyncio.get_running_loop()))
                return work

        def execution_started():
            """Transfer ownership of the callback operation."""
            return OwnedToolTask(outer())

        task = asyncio.create_task(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert nested and nested[0].done()
        assert "tool" not in effects

    asyncio.run(scenario())


def test_async_callback_awaits_all_nested_tasks_before_admission():
    """Nested callback tasks settle before execution starts."""
    registry = ToolRegistry()
    events = []

    @declare_tool()
    async def calculate() -> str:
        """Record the admitted tool invocation."""
        events.append("tool")
        return "done"

    registry.register(calculate)

    async def scenario():
        """Keep the nested callback task pending until explicitly released."""
        release = asyncio.Event()

        async def child():
            """Mark callback completion."""
            await release.wait()
            events.append("child")

        async def outer():
            """Return a second task."""
            return asyncio.create_task(child())

        async def execution_started():
            """Return the outer callback task."""
            return asyncio.create_task(outer())

        task = asyncio.create_task(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        await deliver_callbacks()
        assert events == []
        release.set()
        assert result_value(await task) == "done"
        assert events == ["child", "tool"]

    asyncio.run(scenario())


def test_cancelled_sync_dispatch_does_not_admit_returned_owned_child():
    """Retained registry interruption prevents starting nested owned work."""
    registry = ToolRegistry()
    effects = []

    async def scenario():
        """Cancel a returned owned task that produces an unstarted owned child."""
        started = asyncio.Event()
        invoked = asyncio.Event()
        loop = asyncio.get_running_loop()
        children = []

        async def child():
            """Record an incorrectly admitted operation."""
            effects.append("child")

        async def outer():
            """Return unstarted work while settling cancellation."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                coroutine = child()
                children.append(coroutine)
                return OwnedToolTask(coroutine)

        work = OwnedToolTask(outer())
        work.start(asyncio.get_running_loop())

        @declare_tool()
        def calculate() -> str:
            """Return already-started owned work."""
            loop.call_soon_threadsafe(invoked.set)
            return work

        registry.register(calculate)
        task = asyncio.create_task(registry.call_async("calculate", "{}"))
        await started.wait()
        await invoked.wait()
        await deliver_callbacks()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert effects == []
        assert children and inspect.getcoroutinestate(children[0]) == inspect.CORO_CLOSED

    import inspect

    asyncio.run(scenario())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_planner_rejects_cyclic_awaitables_without_executing_tool(asynchronous):
    """Cyclic planner results are rejected without admitting an effect."""
    invoked = []

    async def planner(arguments):
        """Return a completed future whose result is itself."""
        future = asyncio.get_running_loop().create_future()
        future.set_result(future)
        return future

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Record execution after planning."""
        invoked.append(number)
        return number

    registry = ToolRegistry([calculate])
    try:
        with pytest.raises(RuntimeError, match="Tool callback returned cyclic awaitables"):
            if asynchronous:
                asyncio.run(registry.call_async("calculate", '{"number": 3}'))
            else:
                registry.call("calculate", '{"number": 3}')
        assert invoked == []
    finally:
        registry.close()


def test_planner_runner_setup_failure_closes_returned_coroutine(monkeypatch):
    """A failed runner setup closes planner work before any tool can execute."""
    created, invoked = [], []

    async def prepare(arguments):
        """Produce a plan only if runner setup succeeds."""
        return OperationPlan(arguments=arguments)

    def planner(arguments):
        """Retain the coroutine so the test can verify its closure."""
        coroutine = prepare(arguments)
        created.append(coroutine)
        return coroutine

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Record execution after planning."""
        invoked.append(number)
        return number

    registry = ToolRegistry([calculate])
    monkeypatch.setattr(
        tool_registry_module.asyncio,
        "Runner",
        Mock(side_effect=OSError("synthetic runner setup failure")),
    )
    try:
        result = json.loads(registry.call("calculate", '{"number": 3}'))
        assert result["problem"]["code"] == "tool.planning_failed"
        assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED
        assert invoked == []
    finally:
        registry.close()


@pytest.mark.parametrize("callback", [False, True])
def test_foreign_owned_task_is_neither_driven_nor_cancelled(callback):
    """A foreign owned task remains with its loop when tool dispatch rejects it."""
    foreign_loop = asyncio.new_event_loop()
    effects = []

    async def work():
        """Detect any execution of the foreign task."""
        effects.append("foreign work")

    owned = OwnedToolTask(work())
    foreign_task = owned.start(foreign_loop)

    def calculate():
        """Return foreign work when it is the tool result."""
        return "done" if callback else owned

    def execution_started():
        """Return foreign work when it is the callback result."""
        return owned

    registry = ToolRegistry([calculate])
    try:
        if callback:
            with pytest.raises(ProblemException) as caught:
                asyncio.run(
                    registry.call_async("calculate", "{}", execution_started=execution_started)
                )
            assert caught.value.problem.code == "tool.loop_mismatch"
        else:
            with pytest.raises(ToolExecutionOutcomeUnknown):
                asyncio.run(registry.call_async("calculate", "{}"))
        assert not foreign_task.done()
        assert effects == []
    finally:
        registry.close()
        foreign_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            foreign_loop.run_until_complete(foreign_task)
        foreign_loop.run_until_complete(asyncio.sleep(0))
        foreign_loop.close()


def test_async_callback_resolves_custom_awaitable_before_tool_execution():
    """A callback's custom awaitable settles before the tool is admitted."""
    events = []

    class CallbackResult:
        """Represent an awaitable that is neither a coroutine nor a future."""

        def __await__(self):
            async def resolve():
                """Complete the callback before the tool body begins."""
                events.append("callback")

            return resolve().__await__()

    def execution_started():
        """Return a custom callback awaitable."""
        return CallbackResult()

    def calculate():
        """Record the admitted invocation."""
        events.append("tool")
        return "done"

    registry = ToolRegistry([calculate])
    try:
        output = asyncio.run(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        assert result_value(output) == "done"
        assert events == ["callback", "tool"]
    finally:
        registry.close()


def test_cancelled_custom_callback_cannot_admit_tool():
    """A custom awaitable cannot admit execution by suppressing cancellation."""
    events = []

    async def scenario():
        """Interrupt the callback while its custom awaitable is suspended."""
        started = asyncio.Event()
        release = asyncio.Event()

        class CallbackResult:
            """Represent borrowed custom callback work."""

            def __await__(self):
                async def resolve():
                    """Finish only after the work's owner releases it."""
                    started.set()
                    await release.wait()
                    events.append("callback settled")

                return resolve().__await__()

        def execution_started():
            """Return custom callback work."""
            return CallbackResult()

        def calculate():
            """Record any execution after cancellation."""
            events.append("tool")
            return "done"

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(
                registry.call_async("calculate", "{}", execution_started=execution_started)
            )
            await started.wait()
            task.cancel()
            await deliver_callbacks()
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert events == ["callback settled"]
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


def test_cancelled_custom_callback_preserves_borrowed_future():
    """A custom wrapper cannot transfer ownership of its caller's future."""

    async def scenario():
        """Cancel dispatch while the wrapped borrowed future is pending."""
        started = asyncio.Event()
        future = asyncio.get_running_loop().create_future()
        effects = []

        class CallbackResult:
            """Delegate to a caller-owned future."""

            def __await__(self):
                started.set()
                return future.__await__()

        def calculate():
            """Record an incorrectly admitted tool."""
            effects.append("tool")
            return "done"

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(
                registry.call_async("calculate", "{}", execution_started=CallbackResult)
            )
            await started.wait()
            task.cancel()
            await deliver_callbacks()
            assert not task.done()
            assert not future.cancelled()
            future.set_result(None)
            with pytest.raises(asyncio.CancelledError):
                await task
            assert effects == []
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize("callback", [False, True])
def test_cancelled_worker_does_not_close_returned_borrowed_awaitable(callback):
    """A worker returning after cancellation cannot close borrowed work."""

    async def scenario():
        """Interrupt a worker before it returns a borrowed custom awaitable."""
        started, release = Event(), Event()
        future = asyncio.get_running_loop().create_future()
        effects = []

        class BorrowedResult:
            """Make an incorrect close observable on the owner's future."""

            def __await__(self):
                return future.__await__()

            def close(self):
                future.cancel()

        def worker():
            """Return borrowed work once the owner permits completion."""
            started.set()
            release.wait(timeout=5)
            return BorrowedResult()

        def calculate():
            """Record an incorrectly admitted callback, or return tool work."""
            if callback:
                effects.append("tool")
                return "done"
            return worker()

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(
                registry.call_async(
                    "calculate", "{}", execution_started=worker if callback else None
                )
            )
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not future.cancelled()
            assert effects == []
        finally:
            release.set()
            future.cancel()
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


def test_cancelled_callback_does_not_close_nested_borrowed_awaitable():
    """A callback settling cancellation does not close a returned borrowed awaitable."""

    async def scenario():
        """Return a borrowed wrapper from a callback's cancellation handler."""
        started = asyncio.Event()
        future = asyncio.get_running_loop().create_future()

        class BorrowedResult:
            """Make an incorrect close observable on the owner's future."""

            def __await__(self):
                return future.__await__()

            def close(self):
                future.cancel()

        async def before():
            """Return borrowed work without starting it."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return BorrowedResult()

        def calculate():
            """Return a value only if incorrectly admitted."""
            return "unexpected"

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(
                registry.call_async("calculate", "{}", execution_started=before)
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not future.cancelled()
        finally:
            future.cancel()
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize("callback", [False, True])
def test_cancelled_custom_wrapper_does_not_start_work(callback, monkeypatch):
    """Cancellation before a borrowed wrapper's first turn does not admit work."""
    effects = []
    original_shield = asyncio.shield

    def interrupt_before_start(future):
        """Interrupt just before waiting on the new custom-awaitable wrapper."""
        if isinstance(future, asyncio.Task) and future.get_coro().__name__ == "await_value_async":
            raise asyncio.CancelledError
        return original_shield(future)

    class CustomResult:
        """Represent work that must not start after interruption."""

        def __await__(self):
            async def resolve():
                """Record forbidden entry into the borrowed work."""
                effects.append("started")
                return "done"

            return resolve().__await__()

    def calculate():
        """Return custom tool work or record an incorrectly admitted callback."""
        if callback:
            effects.append("tool")
            return "done"
        return CustomResult()

    monkeypatch.setattr(asyncio, "shield", interrupt_before_start)
    registry = ToolRegistry([calculate])
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                registry.call_async(
                    "calculate", "{}", execution_started=CustomResult if callback else None
                )
            )
        assert effects == []
    finally:
        registry.close()


def test_cancelled_custom_tool_awaitable_retains_receipt():
    """A custom tool awaitable retains its result while propagating cancellation."""
    effects = []

    async def scenario():
        """Interrupt the caller while its borrowed tool work settles."""
        started = asyncio.Event()
        release = asyncio.Event()

        class ToolResult:
            """Complete a side effect while its caller is interrupted."""

            def __await__(self):
                async def resolve():
                    """Return the known outcome after cancellation."""
                    started.set()
                    await release.wait()
                    effects.append("committed")
                    return "confirmed"

                return resolve().__await__()

        def calculate():
            """Return custom tool work."""
            return ToolResult()

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(registry.call_async("calculate", "{}"))
            await started.wait()
            task.cancel()
            await deliver_callbacks()
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert result_value(caught.value.output) == "confirmed"
            assert effects == ["committed"]
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize("declared_final", [False, True])
def test_awaited_owned_result_requires_final_declaration(declared_final):
    """Only an explicitly final child outcome becomes the interrupted tool receipt."""
    effects = []

    async def scenario():
        """Interrupt an async tool while its owned child settles."""
        started = asyncio.Event()

        async def effect():
            """Return a confirmed result after a synthetic side effect."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                effects.append("committed")
                return "confirmed"

        @declare_tool()
        async def calculate() -> str:
            """Await the owned operation as the tool's final step."""
            return await OwnedToolTask(effect(), final_result=declared_final)

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(registry.call_async("calculate", "{}"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            if declared_final:
                assert result_value(caught.value.output) == "confirmed"
            else:
                assert not hasattr(caught.value, "output")
            assert effects == ["committed"]
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


def test_intermediate_owned_result_never_admits_following_work_after_cancel():
    """An intermediate result cannot become a final receipt or resume tool effects."""
    effects = []

    async def scenario():
        """Interrupt an intermediate owned operation in an async tool."""
        started = asyncio.Event()

        async def child():
            """Complete one intermediate step during cancellation."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                effects.append("child")
                return "partial"

        @declare_tool()
        async def calculate() -> str:
            """Run two steps, the second of which must not start."""
            await OwnedToolTask(child())
            effects.append("later")
            return "finished"

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(registry.call_async("calculate", "{}"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert not hasattr(caught.value, "output")
            assert effects == ["child"]
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


def test_final_result_does_not_start_nested_custom_work_after_cancellation():
    """A declared final outcome cannot start returned custom work after interruption."""
    effects = []

    async def scenario():
        """Return a borrowed custom awaitable from a settling final operation."""
        started = asyncio.Event()

        class Deferred:
            """Represent work which has not started."""

            def __await__(self):
                async def resolve():
                    """Record any incorrectly admitted execution."""
                    effects.append("deferred")
                    return "unexpected"

                return resolve().__await__()

        async def child():
            """Return deferred work while settling interruption."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return Deferred()

        @declare_tool()
        async def calculate() -> str:
            """Declare the owned child as the final outcome."""
            return await OwnedToolTask(child(), final_result=True)

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(registry.call_async("calculate", "{}"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert not hasattr(caught.value, "output")
            assert effects == []
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


def test_final_result_nested_admission_failure_is_execution_failure(caplog):
    """A nested failure after an effect cannot report the invocation as unstarted."""
    effects = []

    async def scenario():
        """Return an already-failed borrowed future after cancellation."""
        started = asyncio.Event()

        async def child():
            """Commit an effect and return the nested failure."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                effects.append("committed")
                future = asyncio.get_running_loop().create_future()
                future.set_exception(tool_registry_module.ToolExecutionAdmissionRejected())
                return future

        @declare_tool()
        async def calculate() -> str:
            """Declare the child as the final outcome."""
            return await OwnedToolTask(child(), final_result=True)

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(registry.call_async("calculate", "{}"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert json.loads(caught.value.output)["problem"]["code"] == "tool.execution_failed"
            assert effects == ["committed"]
            assert any(
                record.__dict__.get("error.type") == "tool.execution_failed"
                for record in caplog.records
            )
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize("native_tool", [False, True])
@pytest.mark.parametrize("cancel_dispatch", [False, True])
def test_foreign_final_marker_is_not_tool_receipt(native_tool, cancel_dispatch):
    """An unrelated task's final marker cannot become this invocation's receipt."""

    async def scenario():
        """Cancel an independently owned operation while a tool is waiting for it."""
        started = asyncio.Event()

        async def foreign_child():
            """Return an unrelated receipt when its own owner cancels it."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "unrelated receipt"

        async def foreign_operation():
            """Expose the foreign operation's explicitly final outcome."""
            return await OwnedToolTask(foreign_child(), final_result=True)

        foreign = asyncio.create_task(foreign_operation())

        async def async_calculate():
            """Await a task owned by another operation."""
            return await foreign

        def sync_calculate():
            """Return a task owned by another operation."""
            return foreign

        calculate = async_calculate if native_tool else sync_calculate
        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(registry.call_async(calculate.__name__, "{}"))
            await started.wait()
            await deliver_callbacks()
            if cancel_dispatch:
                task.cancel()
            foreign.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert not hasattr(caught.value, "output")
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "outcome",
    ["child_failure", "admission_failure", "finally_failure", "finally_admission", "caught"],
)
def test_final_owned_outcome_respects_enclosing_tool_cleanup(outcome, caplog):
    """Actual parent outcomes supersede a declared child outcome during cleanup."""

    async def scenario():
        """Interrupt the declared final child and settle its parent."""
        started = asyncio.Event()

        async def child():
            """Settle the child during cancellation."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if outcome == "child_failure":
                    raise ValueError("child failure")
                if outcome == "admission_failure":
                    raise tool_registry_module.ToolExecutionAdmissionRejected
                return "child result"

        @declare_tool()
        async def calculate() -> str:
            """Return the declared final outcome unless cleanup changes it."""
            try:
                return await OwnedToolTask(child(), final_result=True)
            except asyncio.CancelledError:
                if outcome == "caught":
                    return "parent result"
                raise
            finally:
                if outcome == "finally_failure":
                    raise ValueError("parent cleanup failure")
                if outcome == "finally_admission":
                    raise tool_registry_module.ToolExecutionAdmissionRejected

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(registry.call_async("calculate", "{}"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            result = json.loads(caught.value.output)
            if outcome == "caught":
                assert result["result"] == "parent result"
            else:
                assert result["problem"]["code"] == "tool.execution_failed"
                assert any(
                    record.__dict__.get("error.type") == "tool.execution_failed"
                    for record in caplog.records
                )
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())


@pytest.mark.parametrize("second_interrupt", [False, True])
def test_sync_interrupted_owned_receipt_is_normalized_and_propagated(second_interrupt):
    """Synchronous Ctrl+C carries the final receipt through registry normalization."""
    effects = []

    def interrupt():
        """Interrupt the runner after the tool has started."""
        raise KeyboardInterrupt

    async def child():
        """Commit an effect and return its outcome during cancellation."""
        try:
            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if second_interrupt:
                asyncio.get_running_loop().call_soon(interrupt)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            effects.append("committed")
            return "confirmed"

    @declare_tool()
    async def calculate() -> str:
        """Declare the child outcome as final."""
        return await OwnedToolTask(child(), final_result=True)

    registry = ToolRegistry([calculate])
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            registry.call_with_timing("calculate", "{}")
        assert result_value(caught.value.output) == "confirmed"
        assert caught.value.duration >= 0
        assert effects == ["committed"]
        if second_interrupt:
            assert caught.value.__notes__ == ["Tool cleanup also stopped with KeyboardInterrupt."]
    finally:
        registry.close()


@pytest.mark.parametrize("outcome", ["cancelled", "failed", "returned"])
def test_cancelled_async_start_callback_preserves_signal_and_reports_failure(outcome, caplog):
    """Cancelled start callbacks settle, signal cancellation, and never admit a tool."""
    signal, effects = Event(), []

    def calculate():
        """Record forbidden execution after cancellation."""
        effects.append("tool")
        return "done"

    async def scenario():
        """Cancel while the asynchronous start callback is waiting."""
        started = asyncio.Event()

        async def execution_started():
            """Choose the callback's terminal outcome after cancellation."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if outcome == "failed":
                    raise RuntimeError("private callback cleanup failure")
                if outcome == "cancelled":
                    raise
                return

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(
                registry.call_async(
                    "calculate",
                    "{}",
                    execution_started=execution_started,
                    cancellation_event=signal,
                )
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert signal.is_set()
            assert effects == []
        finally:
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())
    if outcome == "failed":
        assert len(caplog.records) == 1
        assert caplog.records[0].__dict__["error.type"] == "tool.callback_failed"
        assert "private callback cleanup failure" not in caplog.records[0].getMessage()
    else:
        assert not caplog.records


@pytest.mark.parametrize("outcome", ["cycle", "cancelled", "failure", "foreign", "foreign_owned"])
def test_cancelled_callback_settles_nested_future_outcomes(outcome, caplog):
    """Nested callback cleanup preserves foreign ownership and reports failures once."""
    foreign_loop = asyncio.new_event_loop()
    foreign_future = foreign_loop.create_future()
    foreign_owned = None
    foreign_task = None
    effects = []

    async def foreign_work():
        """Detect accidental execution on a foreign owner loop."""
        effects.append("foreign")

    if outcome == "foreign_owned":
        foreign_owned = OwnedToolTask(foreign_work())
        foreign_task = foreign_owned.start(foreign_loop)

    def calculate():
        """Detect tool execution after cancellation."""
        effects.append("tool")
        return "done"

    async def scenario():
        """Return a nested future from the callback's cancellation handler."""
        started = asyncio.Event()

        async def execution_started():
            """Return a controlled cleanup result after cancellation."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if outcome == "foreign":
                    return foreign_future
                if outcome == "foreign_owned":
                    return foreign_owned
                future = asyncio.get_running_loop().create_future()
                if outcome == "cycle":
                    future.set_result(future)
                elif outcome == "cancelled":
                    future.cancel()
                else:
                    future.set_exception(RuntimeError("private nested failure"))
                return future

        registry = ToolRegistry([calculate])
        try:
            task = asyncio.create_task(
                registry.call_async("calculate", "{}", execution_started=execution_started)
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not foreign_future.done()
            assert foreign_task is None or not foreign_task.done()
            assert effects == []
        finally:
            await asyncio.to_thread(registry.close)

    try:
        asyncio.run(scenario())
        if outcome == "failure":
            assert len(caplog.records) == 1
            assert caplog.records[0].__dict__["error.type"] == "tool.callback_failed"
            assert "private nested failure" not in caplog.records[0].getMessage()
        else:
            assert not caplog.records
    finally:
        foreign_future.cancel()
        if foreign_task is not None:
            foreign_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                foreign_loop.run_until_complete(foreign_task)
            foreign_loop.run_until_complete(asyncio.sleep(0))
        foreign_loop.close()


def test_cancelled_callback_worker_preserves_foreign_owned_task():
    """A cancelled worker cannot claim or cancel a task on another loop."""
    foreign_loop = asyncio.new_event_loop()
    entered, release = Event(), Event()
    effects = []

    async def foreign_work():
        """Detect accidental execution of foreign owned work."""
        effects.append("foreign")

    owned = OwnedToolTask(foreign_work())
    foreign_task = owned.start(foreign_loop)

    def execution_started():
        """Return foreign work after cancellation reaches the worker."""
        entered.set()
        assert release.wait(timeout=5), "test synchronization was not released"
        return owned

    def calculate():
        """Detect tool execution after cancellation."""
        effects.append("tool")
        return "done"

    async def scenario():
        """Cancel dispatch while the callback worker is still active."""
        registry = ToolRegistry([calculate])
        task = asyncio.create_task(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        try:
            await asyncio.to_thread(entered.wait)
            task.cancel()
            await deliver_callbacks()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not foreign_task.done()
            assert effects == []
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(registry.close)

    try:
        asyncio.run(scenario())
    finally:
        foreign_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            foreign_loop.run_until_complete(foreign_task)
        foreign_loop.run_until_complete(asyncio.sleep(0))
        foreign_loop.close()


def test_sync_start_callback_returning_coroutine_is_awaited():
    """A synchronous start callback's coroutine settles before tool execution."""
    events = []

    async def prepare():
        """Record completion of the asynchronous callback work."""
        events.append("callback")

    def execution_started():
        """Return unstarted coroutine work from the worker."""
        return prepare()

    def calculate():
        """Record the admitted tool execution."""
        events.append("tool")
        return "done"

    registry = ToolRegistry([calculate])
    try:
        output = asyncio.run(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        assert result_value(output) == "done"
        assert events == ["callback", "tool"]
    finally:
        registry.close()


@pytest.mark.parametrize("started", [False, True])
def test_cancelled_callback_worker_settles_owned_work(started):
    """Cancelled callback workers close unstarted work or cancel started owned work."""
    entered, release = Event(), Event()
    effects = []

    def calculate():
        """Detect forbidden tool execution after cancellation."""
        effects.append("tool")
        return "done"

    async def scenario():
        """Release the callback only after dispatch has been cancelled."""
        running = asyncio.Event()

        async def work():
            """Settle a cancellation if this owned work has started."""
            try:
                running.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                effects.append("cancelled")

        coroutine = work()
        owned = OwnedToolTask(coroutine)
        if started:
            owned.start(asyncio.get_running_loop())
            await running.wait()

        def execution_started():
            """Transfer owned work once the worker is released."""
            entered.set()
            assert release.wait(timeout=5), "test synchronization was not released"
            return owned

        registry = ToolRegistry([calculate])
        task = asyncio.create_task(
            registry.call_async("calculate", "{}", execution_started=execution_started)
        )
        try:
            await asyncio.to_thread(entered.wait)
            task.cancel()
            await deliver_callbacks()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
            assert effects == (["cancelled"] if started else [])
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(registry.close)

    asyncio.run(scenario())
