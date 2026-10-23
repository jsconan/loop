"""Tests for tool invocation context."""

import asyncio
from unittest.mock import Mock

import pytest

from loop import InstructionsManager, OperationPlan, ProblemException, ToolContext
from loop.interaction import Interaction


def test_confirm_delegates_to_the_interaction():
    """Confirmation forwards the question and default through the interaction boundary."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    context = ToolContext(interaction=interaction, tool_name="guarded")

    assert context.confirm("Continue?", default=True) is True
    interaction.confirm.assert_called_once_with("Continue?", default=True)


def test_file_receipts_record_exact_versions_absence_and_cached_capabilities(tmp_path):
    """Invocation receipts stay independent and preserve source identity for registry wrapping."""
    context = ToolContext(None, "files")
    source = tmp_path / "file.txt"
    context.record_file_read(source, "sha256:read")
    assert context.read_versions == {str(source): "sha256:read"}
    context.record_file_read(source, None)
    context.record_cached_read("snapshot-handle")
    assert context.read_versions == {str(source): None, "snapshot-handle": None}
    context.record_file_version(source, "written")
    assert context.file_versions[str(source)].startswith("sha256:")
    context.record_file_removal(source)
    assert context.file_versions == {str(source): None}
    other = ToolContext(None, "files")
    assert other.read_versions == {}
    assert other.file_versions == {}


def test_instruction_observations_delegate_only_when_a_manager_is_available(tmp_path):
    """Context observations remain optional and forward each supported event."""
    interaction = Mock(spec=Interaction)
    manager = Mock(spec=InstructionsManager)
    context = ToolContext(interaction, "files", manager)

    context.observe_file(tmp_path / "file.txt")
    context.observe_directory(tmp_path)
    context.invalidate_instructions(tmp_path / "AGENTS.md")

    assert manager.observe_path.call_args_list[0].args == (tmp_path / "file.txt",)
    assert manager.observe_path.call_args_list[1].kwargs == {"directory": True}
    manager.invalidate.assert_called_once_with(tmp_path / "AGENTS.md")

    unavailable = ToolContext(interaction, "files")
    unavailable.observe_file(tmp_path / "file.txt")
    unavailable.observe_directory(tmp_path)
    unavailable.invalidate_instructions()


def test_additional_authorization_delegates_or_fails_closed():
    """Runtime-discovered arguments use the registry callback and fail closed without one."""
    interaction = Mock(spec=Interaction)
    plan = OperationPlan(arguments={"url": "https://my-host.local"})
    authorizer = Mock(return_value=plan)
    context = ToolContext(interaction, "fetch", additional_authorizer=authorizer)

    assert context.authorize_additional({"url": "https://my-host.local"}) is plan
    authorizer.assert_called_once_with({"url": "https://my-host.local"})

    with pytest.raises(ProblemException) as denied:
        ToolContext(interaction, "fetch").authorize_additional({"url": "https://my-host.local"})
    assert denied.value.problem.code == "tool.authorization_unavailable"


def test_async_context_uses_native_confirmation_and_authorization():
    """Async tools have owner-loop interaction and authorization helpers."""
    calls = []
    plan = OperationPlan(arguments={"url": "https://second.local"})

    class AsyncInteraction:
        """Provide native async confirmation while rejecting accidental sync input."""

        async def prompt_async(self, *_args, **_kwargs):
            return "answer"

        async def confirm_async(self, message, *, default=False):
            calls.append((message, default))
            return True

        def confirm(self, *_args, **_kwargs):
            raise AssertionError("Async tool used synchronous confirmation.")

    async def authorize(arguments):
        calls.append(arguments)
        return plan

    sync_authorizer = Mock()
    context = ToolContext(
        AsyncInteraction(),
        "fetch",
        additional_authorizer=sync_authorizer,
        additional_authorizer_async=authorize,
        execution_mode="async",
    )

    with pytest.raises(ProblemException) as confirmation_error:
        context.confirm("Continue?")
    with pytest.raises(ProblemException) as authorization_error:
        context.authorize_additional({"url": "https://second.local"})

    async def invoke():
        assert await context.confirm_async("Continue?", default=True)
        assert (await context.authorize_additional_async({"url": "https://second.local"})) is plan

    asyncio.run(invoke())

    assert confirmation_error.value.problem.code == "tool.async_required"
    assert authorization_error.value.problem.code == "tool.async_required"
    assert calls == [
        ("Continue?", True),
        {"url": "https://second.local"},
    ]
    sync_authorizer.assert_not_called()


def test_async_additional_authorization_fails_closed_without_registry_support():
    """An async tool cannot request dynamic effects without a registry authorizer."""
    context = ToolContext(None, "fetch", execution_mode="async")

    async def invoke():
        await context.authorize_additional_async({"url": "https://second.local"})

    with pytest.raises(ProblemException) as error:
        asyncio.run(invoke())

    assert error.value.problem.code == "tool.authorization_unavailable"
