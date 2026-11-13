"""Exercise skill authority changes while command approval is pending."""

import asyncio
import json
import shlex
from threading import Event
from unittest.mock import AsyncMock, Mock

import pytest

from loop import Interaction
from loop.execution import CommandExecutionService
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome
from loop.instructions import InstructionsManager, RuntimeEnvironment, Skill, SkillManager
from loop.permissions import PermissionManager, PolicyScope
from loop.session import SessionManager
from loop.tooling import ToolRegistry
from loop.tools.system import read_command_job, run_command

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("change", ["deactivate", "reactivate", "direct_reactivate", "unrelated"])
def test_skill_changes_during_command_approval(tmp_path, asynchronous, change):
    """Approval cannot launch a command using a skill revoked during its prompt."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills = []
    for name in ("guide", "other"):
        directory = tmp_path / name
        directory.mkdir()
        location = directory / "SKILL.md"
        location.write_text(f"---\nname: {name}\ndescription: Fixture\n---\nFixture\n")
        (directory / "data").write_text("fixture")
        skills.append(Skill(name, "Fixture", location))
    sessions = SessionManager()
    permissions = PermissionManager(workspace)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(workspace, sessions.temporary_directory),
        skill_manager=SkillManager(skills),
    )
    instructions.activate_skill("guide")
    instructions.activate_skill("other")
    native = Mock()
    native.capabilities = frozenset()
    native.system_read_roots = ()
    native.capability_failure.return_value = None
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(native, Mock())
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=service,
        session_manager=sessions,
    )
    interaction = Mock(spec=Interaction)
    prompt_messages = []

    def approve(message, **_kwargs):
        """Apply the requested change while the sandbox choice is displayed."""
        prompt_messages.append(message)
        if message == "Allow this command?":
            if change == "direct_reactivate":
                instructions.skill_manager.deactivate("guide")
                instructions.skill_manager.activate("guide")
            else:
                instructions.deactivate_skill("other" if change == "unrelated" else "guide")
            if change == "reactivate":
                instructions.activate_skill("guide")
        return "approve"

    async def approve_async(message, **kwargs):
        """Apply the same change at an asynchronous approval boundary."""
        return approve(message, **kwargs)

    interaction.prompt.side_effect = approve
    interaction.prompt_async = AsyncMock(side_effect=approve_async)
    interaction.confirm_async = AsyncMock(return_value=True)
    source = "cat " + shlex.quote(str(skills[0].location.parent / "data"))
    arguments = json.dumps({"command": source, "cwd": "$WORKSPACE_ROOT", "wait_seconds": 5})
    try:
        if asynchronous:
            output = asyncio.run(
                registry.call_async(
                    "run_command",
                    arguments,
                    interaction=interaction,
                    instructions_manager=instructions,
                )
            )
        else:
            output = registry.call(
                "run_command", arguments, interaction=interaction, instructions_manager=instructions
            )
        result = json.loads(output)
        assert "Allow this command?" in prompt_messages, result
        if change == "unrelated":
            assert result["ok"], result
            assert native.run.call_count == 1
            assert native.run.call_args.args[0].read_roots == (skills[0].location.parent,)
        else:
            assert not result["ok"], result
            assert result["problem"]["code"] == "sandbox.stale"
            native.run.assert_not_called()
    finally:
        registry.close()
        permissions.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("host_choice", ["approve", "session"])
@pytest.mark.parametrize("revoked", [False, True])
def test_skill_revocation_during_host_fallback_approval_blocks_host_launch(
    tmp_path, asynchronous, host_choice, revoked
):
    """Host retry needs a current skill and remembers rules only after successful admission."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_root = tmp_path / "guide"
    skill_root.mkdir()
    location = skill_root / "SKILL.md"
    location.write_text("---\nname: guide\ndescription: Fixture\n---\nFixture\n")
    (skill_root / "data").write_text("fixture")
    sessions = SessionManager()
    permissions = PermissionManager(workspace)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(workspace, sessions.temporary_directory),
        skill_manager=SkillManager([Skill("guide", "Fixture", location)]),
    )
    instructions.activate_skill("guide")
    native, host = Mock(), Mock()
    native.capabilities = frozenset()
    native.system_read_roots = ()
    native.capability_failure.return_value = None
    native.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="unavailable")
    host.run_host_command.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=CommandExecutionService(native, host),
        session_manager=sessions,
    )
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    messages = []

    def approve(message, **_kwargs):
        """Revoke the skill while reviewing the host retry, after sandbox execution failed."""
        messages.append(message)
        if "without the OS sandbox" in message:
            if revoked:
                instructions.deactivate_skill("guide")
            return host_choice
        return "approve"

    async def approve_async(message, **kwargs):
        """Apply the same revocation during asynchronous approval."""
        return approve(message, **kwargs)

    interaction.prompt.side_effect = approve
    interaction.prompt_async = AsyncMock(side_effect=approve_async)
    interaction.confirm_async = AsyncMock(return_value=True)
    source = "cat " + shlex.quote(str(skill_root / "data"))
    arguments = json.dumps({"command": source, "cwd": "$WORKSPACE_ROOT", "wait_seconds": 5})
    try:
        if asynchronous:
            output = asyncio.run(
                registry.call_async(
                    "run_command",
                    arguments,
                    interaction=interaction,
                    instructions_manager=instructions,
                )
            )
        else:
            output = registry.call(
                "run_command", arguments, interaction=interaction, instructions_manager=instructions
            )
        result = json.loads(output)
        assert any("without the OS sandbox" in message for message in messages), result
        assert native.run.call_count == 1
        if revoked:
            assert not result["ok"], result
            host.run_host_command.assert_not_called()
            assert permissions.host_command_rules() == ()
        else:
            assert result["ok"], result
            host.run_host_command.assert_called_once()
            assert len(permissions.host_command_rules()) == (host_choice == "session")
    finally:
        registry.close()
        permissions.close()


def test_async_host_fallback_with_skill_settles_after_origin_loop_closes(tmp_path):
    """An approved host retry keeps its skill and session authority after async handoff."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_root = tmp_path / "guide"
    skill_root.mkdir()
    location = skill_root / "SKILL.md"
    location.write_text("---\nname: guide\ndescription: Fixture\n---\nFixture\n")
    (skill_root / "data").write_text("fixture")
    sessions = SessionManager()
    permissions = PermissionManager(workspace)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(workspace, sessions.temporary_directory),
        skill_manager=SkillManager([Skill("guide", "Fixture", location)]),
    )
    instructions.activate_skill("guide")
    entered = Event()
    release = Event()
    native, host = Mock(), Mock()
    native.capabilities = frozenset()
    native.system_read_roots = ()
    native.capability_failure.return_value = None
    native.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="unavailable")

    def finish_host(_request, **_kwargs):
        """Keep the host job running until the originating event loop has closed."""
        entered.set()
        if not release.wait(15):
            raise TimeoutError("Fixture host command was not released")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, stdout="host-complete")

    host.run_host_command.side_effect = finish_host
    registry = ToolRegistry(
        [run_command, read_command_job],
        permission_manager=permissions,
        execution_service=CommandExecutionService(native, host),
        session_manager=sessions,
    )
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True

    async def approve_async(message, **_kwargs):
        """Approve the sandbox and remember the independent host decision for this session."""
        return "session" if "without the OS sandbox" in message else "approve"

    interaction.prompt_async = AsyncMock(side_effect=approve_async)
    interaction.confirm_async = AsyncMock(return_value=True)
    source = "cat " + shlex.quote(str(skill_root / "data"))
    arguments = json.dumps({"command": source, "cwd": "$WORKSPACE_ROOT", "wait_seconds": 1})
    try:
        first = json.loads(
            asyncio.run(
                registry.call_async(
                    "run_command",
                    arguments,
                    interaction=interaction,
                    instructions_manager=instructions,
                )
            )
        )
        assert first["ok"], first
        assert entered.wait(5)
        assert not release.is_set()
        assert first["result"]["status"] == "running"
        assert len(permissions.host_command_rules()) == 1
        release.set()
        page = json.loads(
            registry.call(
                "read_command_job",
                json.dumps(
                    {
                        "handle": first["result"]["handle"],
                        "cursor": first["result"]["next_cursor"],
                        "wait_seconds": 5,
                        "include_output": True,
                    }
                ),
                instructions_manager=instructions,
            )
        )
        assert page["ok"], page
        assert page["result"]["result"]["exit_code"] == 0
        assert page["result"]["result"]["stdout"]["content"] == "host-complete"
        assert native.run.call_count == host.run_host_command.call_count == 1
    finally:
        release.set()
        registry.close()
        permissions.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("changed_body", [False, True])
def test_discovered_skill_refresh_during_command_approval(
    tmp_path, monkeypatch, asynchronous, changed_body
):
    """Approval survives an unchanged refresh but rejects changed skill instructions."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    catalog = workspace / ".agents" / "skills"
    catalog.mkdir(parents=True)
    skill_root = tmp_path / "guide-source"
    skill_root.mkdir()
    (catalog / "guide").symlink_to(skill_root, target_is_directory=True)
    location = skill_root / "SKILL.md"
    original = "---\nname: guide\ndescription: Fixture\n---\nOriginal instructions\n"
    location.write_text(original)
    (skill_root / "data").write_text("fixture")
    monkeypatch.setattr(
        "loop.instructions.skill_manager.get_skill_directories", lambda *_args: [catalog]
    )
    sessions = SessionManager()
    permissions = PermissionManager(workspace)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    instructions = InstructionsManager.discover(workspace)
    instructions.set_runtime_environment(
        RuntimeEnvironment(workspace, sessions.temporary_directory)
    )
    instructions.activate_skill("guide")
    initial_manager = instructions.skill_manager
    native = Mock()
    native.capabilities = frozenset()
    native.system_read_roots = ()
    native.capability_failure.return_value = None
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=CommandExecutionService(native, Mock()),
        session_manager=sessions,
    )
    interaction = Mock(spec=Interaction)
    prompts = []

    def approve(message, **_kwargs):
        """Refresh the active discovered skill while sandbox approval is displayed."""
        prompts.append(message)
        if message == "Allow this command?":
            replacement = original.replace("Original", "Changed") if changed_body else original
            location.write_text(replacement)
            instructions.invalidate(location)
            instructions.refresh()
        return "approve"

    async def approve_async(message, **kwargs):
        """Refresh the same skill at the asynchronous approval boundary."""
        return approve(message, **kwargs)

    interaction.prompt.side_effect = approve
    interaction.prompt_async = AsyncMock(side_effect=approve_async)
    interaction.confirm_async = AsyncMock(return_value=True)
    source = "cat " + shlex.quote(str(skill_root / "data"))
    arguments = json.dumps({"command": source, "cwd": "$WORKSPACE_ROOT", "wait_seconds": 5})
    try:
        if asynchronous:
            output = asyncio.run(
                registry.call_async(
                    "run_command",
                    arguments,
                    interaction=interaction,
                    instructions_manager=instructions,
                )
            )
        else:
            output = registry.call(
                "run_command", arguments, interaction=interaction, instructions_manager=instructions
            )
        result = json.loads(output)
        assert "Allow this command?" in prompts, result
        assert instructions.skill_manager is not initial_manager
        if changed_body:
            assert not result["ok"], result
            assert result["problem"]["code"] == "sandbox.stale"
            native.run.assert_not_called()
        else:
            assert result["ok"], result
            assert native.run.call_count == 1
    finally:
        registry.close()
        permissions.close()
