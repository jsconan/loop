"""Test saved verification tasks through the trusted command and approval surfaces."""

import json
import shlex
from unittest.mock import Mock

import pytest

from loop import (
    Action,
    CommandManager,
    Decision,
    InstructionsManager,
    PermissionManager,
    ToolRegistry,
)
from loop.application.paths import ApplicationPaths
from loop.application.verification import VerificationCommands
from loop.execution import CommandExecutionService
from loop.execution.sandbox import CommandProcessResult, SandboxCapability, SandboxOutcome
from loop.instructions import RuntimeEnvironment
from loop.interaction import Interaction
from loop.tooling import ToolRuntimeSettings
from loop.tools.system import run_command
from loop.utils import private_directory_lock
from loop.workspace import Workspace, WorkspaceRepository


@pytest.fixture
def verification(tmp_path):
    """Own a durable workspace task directory and a controlled native dispatch boundary."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    data = tmp_path / "workspace data"
    data.mkdir(mode=0o700)
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    paths = ApplicationPaths(tmp_path / "config", data, tmp_path / "state")
    workspace = WorkspaceRepository(paths.workspace_catalog).initialize(
        Workspace.discover(checkout)
    )
    permissions = PermissionManager(
        checkout,
        configuration_path=paths.for_workspace(workspace.id, checkout).permissions,
        user_configuration_path=paths.user_permissions,
        user_namespace_identity=str(paths.data_root),
        protected_host_roots=(data,),
        workspace_id=workspace.id,
        namespace_catalog_path=paths.workspace_catalog,
        bootstrap_pristine_policy=True,
    )
    permissions.set_workspace_command_writes(True)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(checkout, permissions.temporary_directory)
    )
    backend = Mock(capabilities=frozenset({SandboxCapability.WRITE_FILES}))
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    registry = ToolRegistry(
        [run_command],
        settings=ToolRuntimeSettings(sandbox_environment={"MODE": "base", "STABLE": "value"}),
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, Mock()),
    )
    provider = VerificationCommands(data, registry, instructions)
    commands = CommandManager(interaction=interaction)
    commands.register_provider(provider)
    try:
        yield checkout, data, interaction, backend, registry, provider, commands
    finally:
        commands.close()
        registry.close()
        permissions.close()


def save(commands, task, name="check"):
    """Save a task through the real slash-command argument grammar."""
    commands.call("verify", f"save {name} {shlex.quote(json.dumps(task))}")


def test_verification_task_lifecycle_is_private_and_explicit(verification):
    """Saving never runs a task and a new command manager can show and remove it."""
    checkout, data, interaction, backend, _registry, provider, commands = verification
    commands.call("verify", "list")
    assert interaction.info.call_args.args[0] == "No saved verification tasks."
    save(commands, {"command": "printf '%s' \"$MODE\"", "environment": {"MODE": "task"}})
    assert "Saved verification task" in interaction.info.call_args.args[0]
    commands.call("verify", "list")
    assert interaction.info.call_args.args[0] == "check"
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()
    assert (data / "verification-check.json").stat().st_mode & 0o777 == 0o600
    assert not (checkout / "verification-check.json").exists()

    reopened = CommandManager(interaction=interaction)
    reopened.register_provider(provider)
    try:
        reopened.call("verify", "show check")
        shown = json.loads(interaction.info.call_args.args[0])
        assert shown["command"] == "printf '%s' \"$MODE\""
        assert shown["cwd"] == "$WORKSPACE_ROOT"
        assert shown["write_roots"] == shown["write_files"] == []
        reopened.call("verify", "remove check")
        assert "Removed verification task" in interaction.info.call_args.args[0]
        reopened.call("verify", "show check")
        assert "No verification task" in interaction.report.call_args.args[0].detail
        reopened.call("verify", "run check")
        assert "No verification task" in interaction.report.call_args.args[0].detail
        backend.run.assert_not_called()
    finally:
        reopened.close()


@pytest.mark.parametrize("arguments", ['save check \'{"command":"printf new"}\'', "remove check"])
def test_verification_mutation_refuses_a_competing_owner(verification, arguments):
    """An independent lock owner blocks mutations while the current task remains readable."""
    _checkout, data, interaction, _backend, _registry, _provider, commands = verification
    save(commands, {"command": "printf old"})
    with private_directory_lock(data, ".verification.lock"):
        commands.call("verify", arguments)
        interaction.report.assert_called_once()
        assert "Another client is updating saved verification tasks" in (
            interaction.report.call_args.args[0].detail
        )
        commands.call("verify", "show check")
        assert json.loads(interaction.info.call_args.args[0])["command"] == "printf old"


@pytest.mark.parametrize("explicit_scope", [False, True])
def test_verification_dispatch_uses_native_process_approval_and_its_own_scope(
    verification, explicit_scope
):
    """A selected task gets the requested cwd, file scope and per-command environment only."""
    checkout, _data, interaction, backend, registry, _provider, commands = verification
    output = checkout / "reports"
    output.mkdir()
    task = {"command": "printf '%s' \"$MODE\"", "environment": {"MODE": "task"}}
    if explicit_scope:
        task |= {
            "cwd": "$WORKSPACE_ROOT/reports",
            "write_roots": ["$WORKSPACE_ROOT/reports"],
            "write_files": ["summary"],
        }
    save(commands, task)

    commands.call("verify", "run check")

    interaction.prompt.assert_called_once()
    assert json.loads(interaction.tool_result.call_args.args[0])["ok"]
    request = backend.run.call_args.args[0]
    assert request.source == task["command"]
    assert "export MODE=task STABLE=value\n" in request.execution_source
    assert "MODE=task" in interaction.info.call_args.args[0]
    assert registry.settings.sandbox_environment == {"MODE": "base", "STABLE": "value"}
    assert request.cwd == (output if explicit_scope else checkout)
    assert request.write_roots == ((output,) if explicit_scope else ())
    assert request.write_files == ((checkout / "summary",) if explicit_scope else ())


def test_verification_task_overrides_workspace_environment_with_an_output_path(verification):
    """A saved binding retains its native output reference when reopened and dispatched."""
    _, _, interaction, backend, registry, _, commands = verification
    task = {"command": "printf '%s' \"$MODE\"", "environment": {"MODE": {"output": "my report"}}}
    save(commands, task)

    commands.call("verify", "show check")
    assert json.loads(interaction.info.call_args.args[0])["environment"] == task["environment"]
    commands.call("verify", "run check")

    assert json.loads(interaction.tool_result.call_args.args[0])["ok"]
    request = backend.run.call_args.args[0]
    assert request.execution_source == (
        "export MODE=\"${LOOP_OUTPUT_DIR:?}\"/'my report' STABLE=value\n" + task["command"]
    )
    assert registry.settings.sandbox_environment == {"MODE": "base", "STABLE": "value"}
    assert not request.write_roots and not request.write_files


@pytest.mark.parametrize("option", ["account_lookup", "process_pools"])
def test_verification_requires_owner_opt_in_for_each_saved_scope(verification, option):
    """Saved options require independent owner authorization before the command can start."""
    _checkout, _data, interaction, backend, registry, _provider, commands = verification
    backend.backend_options = (option,)
    save(commands, {"command": "printf approved", "backend_options": [option]})
    commands.call("verify", "show check")
    assert json.loads(interaction.info.call_args.args[0])["backend_options"] == [option]

    commands.call("verify", "run check")
    assert (
        "opt-in is off"
        in json.loads(interaction.tool_result.call_args.args[0])["problem"]["detail"]
    )
    backend.run.assert_not_called()
    interaction.prompt.assert_not_called()

    registry.permission_manager.set_sandbox_option(option, True)
    commands.call("verify", "run check")
    interaction.prompt.assert_called_once()
    request = backend.run.call_args.args[0]
    assert request.backend_options == (option,)
    assert not request.network and not request.write_roots


def test_verification_refuses_saved_macos_resource_on_an_unsupported_backend(verification):
    """A stored platform-specific task cannot silently launch with a different scope."""
    _checkout, _data, interaction, backend, _registry, _provider, commands = verification
    save(commands, {"command": "printf check", "backend_options": ["process_pools"]})
    commands.call("verify", "run check")
    assert json.loads(interaction.tool_result.call_args.args[0])["problem"]["code"] == (
        "sandbox.unsupported"
    )
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()


@pytest.mark.parametrize("directory", ["checkout", "parent", "alias"])
def test_verification_refuses_encompassing_directory_before_approval(verification, directory):
    """A saved task may request output directories without granting the whole checkout."""
    checkout, _data, interaction, backend, _registry, _provider, commands = verification
    alias = checkout / "alias"
    alias.symlink_to(checkout, target_is_directory=True)
    spelling = {"checkout": "$WORKSPACE_ROOT", "parent": str(checkout.parent), "alias": str(alias)}
    save(commands, {"command": "printf data", "write_roots": [spelling[directory]]})

    commands.call("verify", "run check")

    problem = json.loads(interaction.tool_result.call_args.args[0])["problem"]
    assert "bounded output directories" in problem["detail"]
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()


def test_verification_task_does_not_supply_process_consent(verification):
    """A saved scratch-only task still requires the default process approval."""
    _checkout, _data, interaction, backend, registry, _provider, commands = verification
    assert (
        registry.permission_manager.configuration.defaults[Action.PROCESS_EXECUTE] is Decision.ASK
    )
    interaction.prompt.return_value = "deny"
    save(commands, {"command": "printf denied"})

    commands.call("verify", "run check")

    interaction.prompt.assert_called_once()
    backend.run.assert_not_called()
    assert json.loads(interaction.tool_result.call_args.args[0])["problem"]["code"] == "tool.denied"


@pytest.mark.parametrize("mutation", ["change", "remove", "identical", "unsafe", "policy"])
def test_verification_cannot_launch_after_task_changes_during_approval(verification, mutation):
    """The admitted task and the approving policy must remain current through approval."""
    _checkout, data, interaction, backend, registry, provider, commands = verification
    save(commands, {"command": "printf old"})
    other = CommandManager(interaction=interaction)
    other.register_provider(provider)

    def change_task(*_args, **_kwargs):
        """Change the saved record before answering its pending approval."""
        if mutation == "remove":
            other.call("verify", "remove check")
        elif mutation == "unsafe":
            (data / "verification-check.json").chmod(0o644)
        elif mutation == "policy":
            registry.permission_manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY)
        else:
            save(other, {"command": "printf new" if mutation == "change" else "printf old"})
        return "approve"

    try:
        interaction.prompt.side_effect = change_task
        commands.call("verify", "run check")
        backend.run.assert_not_called()
        problem = json.loads(interaction.tool_result.call_args.args[0])["problem"]
        if mutation == "policy":
            assert problem["code"] == "tool.denied"
        else:
            assert problem["code"] == "sandbox.stale"
            assert "/verify show check" in problem["detail"]
    finally:
        other.close()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ("run", "/verify run NAME"),
        ("show check wait_seconds=0", "/verify run NAME"),
        ("list check", "/verify run NAME"),
        ("show ../check", "lowercase letters"),
        ("save check '{}'", "must contain a command"),
        ('save check \'{"command":"ok","other":false}\'', "valid optional scopes"),
        ('save check \'{"command":"ok","backend_options":false}\'', "valid optional scopes"),
    ],
)
def test_verification_rejects_invalid_owner_input(verification, arguments, message):
    """Invalid task names, arguments and documents are refused before command dispatch."""
    _checkout, _data, interaction, backend, _registry, _provider, commands = verification

    commands.call("verify", arguments)

    assert message in interaction.report.call_args.args[0].detail
    backend.run.assert_not_called()


def test_verification_rejects_oversized_task_saves_and_reads(verification):
    """Task storage bounds apply to owner saves and pre-existing task records."""
    _checkout, data, interaction, backend, _registry, _provider, commands = verification
    save(commands, {"command": "x" * (65 * 1024)})
    assert "exceeds the storage limit" in interaction.report.call_args.args[0].detail
    assert not list(data.glob("verification-*.json"))
    path = data / "verification-check.json"
    path.touch(mode=0o600)
    path.write_bytes(b"x" * (65 * 1024))
    for action in ("show", "run"):
        commands.call("verify", f"{action} check")
        assert "storage is unavailable or unsafe" in interaction.report.call_args.args[0].detail
    backend.run.assert_not_called()


@pytest.mark.parametrize("command", ["printf '%s' 'two words'", "{ printf ok; }"])
def test_plain_shell_shortcut_is_private_and_uses_normal_approval(verification, command):
    """Owners can record a command without constructing JSON or changing model tools."""
    _, _, interaction, backend, registry, _, commands = verification
    definitions = registry.definitions()

    commands.call("verify", f"save check {shlex.quote(command)}")
    commands.call("verify", "show check")

    assert json.loads(interaction.info.call_args.args[0])["command"] == command
    assert registry.definitions() == definitions
    assert [tool.name for tool in definitions] == ["run_command"]
    backend.run.assert_not_called()
    commands.call("verify", "run check")
    interaction.prompt.assert_called_once()
    assert backend.run.call_args.args[0].source == command
