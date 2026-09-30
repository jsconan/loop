"""Tests for layered operation-policy evaluation, approval, and persistence."""

import json
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import yaml

from loop import (
    Action,
    ApprovalChoice,
    AuthorizationResult,
    Decision,
    FileTarget,
    Interaction,
    NetworkTarget,
    Operation,
    PermissionConfiguration,
    PermissionConfigurationError,
    PermissionLoadPolicy,
    PermissionLoadResult,
    PermissionManager,
    PermissionPreset,
    PermissionPresetError,
    PermissionRule,
    PolicyLimits,
    PolicyScope,
    ProcessBoundary,
    ProcessTarget,
    SessionTarget,
)
from loop.execution.sandbox import SandboxRequest
from loop.permissions import (
    CommandFinding,
    CommandInspection,
    CommandReviewStatus,
    PermissionLoadFailure,
)
from loop.permissions import manager as manager_module
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.telemetry.policy import thaw


def operation(action: Action, *, tool: str = "demo", target=None, reason=None) -> Operation:
    """Build one representative typed operation."""
    return Operation(tool_id=tool, action=action, target=target, reason=reason)


def file_operation(action: Action, path) -> Operation:
    """Build one filesystem operation for a canonical path."""
    return operation(action, target=FileTarget(path=str(path)))


def native_request(root: Path, source: str = "git status", **changes) -> SandboxRequest:
    """Bind one native command to disposable policy roots."""
    values = {
        "source": source,
        "cwd": root,
        "workspace": root,
        "read_roots": (root.parent,),
        "write_roots": (root,),
        "network": False,
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C"},
        "policy_version": "macos-seatbelt-v1",
        "deadline": time.monotonic() + 60,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def test_git_sandbox_roots_ignore_commands_without_git_intent(tmp_path):
    """An ordinary command receives no Git metadata authority or creation intent."""
    assert PermissionManager.git_sandbox_roots(tmp_path, read=False, write=False) == (
        (),
        (),
        False,
    )


def test_host_rule_stores_only_virtual_context_and_checks_runtime_binding(tmp_path):
    """A host rule contains no real root and cannot cross to another concrete workspace."""
    other = tmp_path.parent / "other-host-workspace"
    other.mkdir()
    manager = PermissionManager(tmp_path)
    first = native_request(tmp_path, "cat /workspace/file", read_roots=())
    moved = native_request(other, "cat /workspace/file", read_roots=())

    rule = manager.remember_host_command_rule(first)

    assert first.host_command_signature() == moved.host_command_signature()
    assert tmp_path.as_posix() not in rule.model_dump_json()
    assert other.as_posix() not in rule.model_dump_json()
    assert rule.cwd == "/workspace"
    assert rule.source == "<path-bearing-command>"
    assert manager.matching_host_command_rule(first) == rule
    assert manager.matching_host_command_rule(moved) is None
    literal_host = native_request(tmp_path, f"cat {tmp_path / 'private'}", read_roots=())
    literal_rule = manager.remember_host_command_rule(literal_host)
    assert tmp_path.as_posix() not in literal_rule.model_dump_json()
    assert literal_rule.source == "<path-bearing-command>"
    assert manager.matching_host_command_rule(literal_host) == literal_rule
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_rule = manager.remember_host_command_rule(
        native_request(tmp_path, "printf nested", cwd=nested, read_roots=())
    )
    assert nested_rule.cwd == "/workspace/nested"
    assert manager.reset_session()
    assert manager.matching_host_command_rule(first) is None


@pytest.mark.parametrize("write_roots", [(), "workspace"])
def test_native_default_workspace_access_runs_without_prompt(tmp_path, write_roots):
    """Routine workspace commands run in either write mode without a prompt."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    roots = (tmp_path,) if write_roots == "workspace" else ()
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, "cat file", write_roots=roots, read_roots=()),
        tool_id="run_command",
    )
    interaction.prompt.assert_not_called()


def test_native_network_still_requires_permission(tmp_path):
    """General network access stays outside the automatic workspace boundary."""
    manager = PermissionManager(tmp_path)
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "cat file", write_roots=(), read_roots=(), network=True),
            tool_id="run_command",
        )
        is False
    )


def test_explicit_process_ask_rule_still_prompts_for_default_sandbox(tmp_path):
    """An explicit user ask rule overrides prompt-free default sandbox access."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            rules=[PermissionRule(decision=Decision.ASK, action=Action.PROCESS_EXECUTE)]
        ),
    )
    request = native_request(tmp_path, "printf safe", read_roots=())
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert "run under your command approval rule" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()


@pytest.mark.parametrize(
    "source",
    [
        "rm -rf build",
        "git add file",
        "env -i /bin/rm victim",
        "printf text > victim",
        "sh -c 'git add file'",
        "FOO=1 command mv source existing",
    ],
)
def test_native_destructive_workspace_commands_require_approval(tmp_path, source):
    """Recognizable destructive shell actions ask even within default workspace access."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)

    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, source, read_roots=()), tool_id="run_command"
    )
    if "git" in source:
        assert "Review a Git state-changing command" in interaction.info.call_args.args[0]
        assert "change repository state" in interaction.info.call_args.args[0]
    elif ">" in source:
        assert "Review an output destination" in interaction.info.call_args.args[0]
    else:
        assert "Review a destructive workspace command" in interaction.info.call_args.args[0]
        assert "delete or overwrite files" in interaction.info.call_args.args[0]
    assert "similar_session" not in interaction.prompt.call_args.kwargs["choices"]
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {"deny", "approve"}


def test_git_write_requires_fresh_approval_after_prior_approval(tmp_path):
    """An earlier one-off Git approval cannot authorize a later Git mutation."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["approve", "deny"]
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "git add file", read_roots=())
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert interaction.prompt.call_count == 2


def test_combined_command_findings_show_both_reasons_and_allow_once_only(tmp_path):
    """All matching command policies contribute to one fresh approval prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    audit_path = tmp_path / "audit.db"
    manager = PermissionManager(
        tmp_path, interaction=interaction, audit_path=audit_path, workspace_id="workspace-1"
    )

    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, "git add file && rm file", read_roots=()),
        tool_id="run_command",
    )
    message = interaction.info.call_args.args[0]
    assert "Review a Git state-changing command" in message
    assert "Review a destructive workspace command" in message
    assert "change repository state" in message
    assert "delete or overwrite files in this workspace" in message
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {"deny", "approve"}
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records "
            "WHERE event_name = 'sandbox.permission_scope'"
        ).fetchone()
    assert json.loads(row[0])["review_policies"] == ["git_change", "destructive_command"]


def test_fresh_command_finding_overrides_matching_remembered_rule(tmp_path):
    """A stored command approval cannot bypass a newly applicable fresh review policy."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "workspace"
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, "printf benign"), tool_id="run_command"
    )
    rule = manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)[0]
    source = "git add file && rm file"
    user_path.write_text(
        yaml.safe_dump(
            {
                "sandboxed_command_rules": [
                    rule.model_copy(update={"source": source}).model_dump(mode="json")
                ]
            }
        ),
        encoding="utf-8",
    )

    interaction.prompt.return_value = "deny"
    reloaded = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    assert not reloaded.authorize_sandboxed_command(
        native_request(tmp_path, source), tool_id="run_command"
    )
    assert interaction.prompt.call_count == 2


def test_injected_inspection_controls_command_review_and_git_grant_intent(tmp_path):
    """A composed matcher changes this manager's review and Git authority classification."""

    def match_special(command: list[str]) -> CommandFinding | None:
        """Request reusable review and Git authority for a special command."""
        if command != ["special", "command"]:
            return None
        return CommandFinding(
            policy_id="special",
            status=CommandReviewStatus.REVIEW,
            context="Review a special command",
            reason="run a special command",
            requests_git_write=True,
        )

    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    inspection = CommandInspection(()).with_matcher(match_special)
    manager = PermissionManager(tmp_path, interaction=interaction, command_inspection=inspection)
    assert manager.command_needs_git_write("special command")
    assert not manager.command_needs_git_write("git add file")
    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, "special command", read_roots=()),
        tool_id="run_command",
    )
    assert "Review a special command" in interaction.info.call_args.args[0]
    assert "run a special command" in interaction.info.call_args.args[0]
    assert "session" in interaction.prompt.call_args.kwargs["choices"]


@pytest.mark.parametrize("source", ["printf '%s' 'rm -rf build'", "echo '"])
def test_native_non_destructive_or_invalid_source_does_not_trigger_review(tmp_path, source):
    """Literal output, empty segments, and invalid shell text do not claim destructive intent."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, source, read_roots=()),
        tool_id="run_command",
    )
    interaction.prompt.assert_not_called()


def test_unused_external_path_does_not_expand_read_authority(tmp_path):
    """Search metadata alone does not require permission for routine commands."""
    tools = tmp_path.with_name(tmp_path.name + "-external-tools")
    tools.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    environment = {"PATH": str(tools), "LANG": "C"}
    first = native_request(tmp_path, "printf first", read_roots=(), environment=environment)
    assert manager.authorize_sandboxed_command(first, tool_id="run_command")
    second = native_request(tmp_path, "printf second", read_roots=(), environment=environment)
    assert manager.authorize_sandboxed_command(second, tool_id="run_command")
    interaction.prompt.assert_not_called()
    expanded = native_request(
        tmp_path, "printf first", read_roots=(tools,), environment=environment
    )
    assert not manager.authorize_sandboxed_command(expanded, tool_id="run_command")
    assert interaction.prompt.call_count == 1
    network = native_request(
        tmp_path, "printf second", read_roots=(), environment=environment, network=True
    )
    assert not manager.authorize_sandboxed_command(network, tool_id="run_command")
    assert interaction.prompt.call_count == 2


def test_native_git_creation_approval_does_not_reuse_existing_git_grant(tmp_path):
    """A remembered fresh-repository grant cannot silently approve later metadata writes."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(tmp_path, interaction=interaction)
    initial = native_request(tmp_path, source="git init", git_create=True)
    assert manager.authorize_sandboxed_command(initial, tool_id="run_command")
    (tmp_path / ".git").mkdir()
    interaction.prompt.return_value = "deny"
    assert not manager.authorize_sandboxed_command(
        native_request(
            tmp_path,
            source="git init",
            write_roots=(tmp_path, tmp_path / ".git"),
        ),
        tool_id="run_command",
    )
    assert interaction.prompt.call_count == 2


@pytest.mark.parametrize("scope", ["session", "workspace", "user"])
def test_native_scoped_approval_reuses_only_matching_authority(tmp_path, scope):
    """Native grants survive at their chosen scope but cannot approve broader effects."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = scope
    manager = PermissionManager(
        tmp_path,
        user_configuration_path=tmp_path / "user.yaml",
        interaction=interaction,
    )
    approved = native_request(tmp_path)
    assert manager.authorize_sandboxed_command(approved, tool_id="run_command") is True
    interaction.prompt.reset_mock()
    assert manager.authorize_sandboxed_command(approved, tool_id="run_command") is True
    interaction.prompt.assert_not_called()
    interaction.prompt.return_value = "deny"
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, network=True), interaction=None, tool_id="run_command"
        )
        is False
    )
    interaction.prompt.assert_called_once()
    assert len(manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)) == (
        1 if scope == "workspace" else 0
    )
    assert len(manager.user_rules) == 0


def test_native_similar_rule_accepts_simple_prefix_only(tmp_path):
    """A reviewed similar rule never matches a compound shell script."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "similar_session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "git status"), tool_id="run_command"
        )
        is True
    )
    interaction.prompt.reset_mock()
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "git status --short"), tool_id="run_command"
        )
        is True
    )
    interaction.prompt.assert_not_called()
    interaction.prompt.return_value = "deny"
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "git status; printf unsafe"), tool_id="run_command"
        )
        is False
    )
    assert interaction.prompt.call_count == 1
    assert "similar_session" not in interaction.prompt.call_args.kwargs["choices"]


def test_native_prompt_shows_command_and_reason_without_internal_paths(tmp_path):
    """Approval text names the command and reason without filesystem details."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "printf change"), tool_id="run_command"
        )
        is False
    )
    message = interaction.info.call_args.args[0]
    assert 'Command: "printf change"' in message
    assert "Reason:" in message
    assert "read files outside this workspace" in message
    assert "PATH:" not in message
    assert "macos-seatbelt" not in message
    assert str(tmp_path) not in message


def test_native_path_read_authority_is_disclosed_and_bound_to_remembered_rule(tmp_path):
    """A changed external read grant requires approval without showing local paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["session", "deny"]
    manager = PermissionManager(workspace, interaction=interaction)
    initial = native_request(
        workspace, "tool", environment={"PATH": str(first)}, read_roots=(first,)
    )
    assert manager.authorize_sandboxed_command(initial, tool_id="run_command")
    assert str(first) not in interaction.info.call_args.args[0]
    changed = native_request(
        workspace, "tool", environment={"PATH": str(second)}, read_roots=(second,)
    )
    assert not manager.authorize_sandboxed_command(changed, tool_id="run_command")
    assert str(second) not in interaction.info.call_args.args[0]
    assert interaction.prompt.call_count == 2


def test_native_remembered_read_rule_binds_executable_identity(tmp_path):
    """Approval for one installed binary cannot authorize another in the same read root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["session", "deny"]
    manager = PermissionManager(workspace, interaction=interaction)
    for name, expected in (("first", True), ("second", False)):
        executable = installation / name
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        details = executable.stat()
        bound = native_request(
            workspace,
            "opaque-tool",
            read_roots=(installation,),
            environment={"PATH": str(installation)},
            executable_identities=((name, executable, executable, details.st_dev, details.st_ino),),
        )
        assert manager.authorize_sandboxed_command(bound, tool_id="run_command") is expected
    assert interaction.prompt.call_count == 2


def test_native_prompt_without_path_does_not_claim_search_reads(tmp_path):
    """Approval text omits executable search details when PATH is absent."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, environment={"LANG": "C"})
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert "search directories" not in interaction.info.call_args.args[0]


def test_native_persistent_rules_reload_and_can_be_removed(tmp_path):
    """Workspace and user native approvals survive restart and remain revocable."""
    interaction = Mock(spec=Interaction)
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    request = native_request(tmp_path, "printf workspace")
    interaction.prompt.return_value = "workspace"
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    interaction.prompt.return_value = "user"
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, "printf user"), tool_id="run_command"
    )
    loaded = PermissionManager(tmp_path, user_configuration_path=user_path)
    workspace_rules = loaded.sandboxed_command_rules(ApprovalChoice.WORKSPACE)
    user_rules = loaded.sandboxed_command_rules(ApprovalChoice.USER)
    assert len(workspace_rules) == len(user_rules) == 1
    assert loaded.authorize_sandboxed_command(request, tool_id="run_command")
    assert loaded.remove_sandboxed_command_rule(ApprovalChoice.WORKSPACE, workspace_rules[0].id)
    assert len(loaded.sandboxed_command_rules(ApprovalChoice.USER)) == 1
    assert loaded.remove_sandboxed_command_rule(ApprovalChoice.USER, user_rules[0].id)
    assert not loaded.remove_sandboxed_command_rule(ApprovalChoice.USER, user_rules[0].id)
    assert not PermissionManager(
        tmp_path, user_configuration_path=user_path
    ).sandboxed_command_rules(ApprovalChoice.USER)


def test_native_persistent_rules_reject_duplicate_ids_and_unsupported_authority(tmp_path):
    """Malformed persisted sandbox approvals fail visibly before matching or revocation."""
    from loop.permissions import SandboxedCommandRule, UserPermissionConfiguration

    first = SandboxedCommandRule(signature="a", source="one", scope=ApprovalChoice.WORKSPACE)
    second = SandboxedCommandRule(
        id=first.id, signature="b", source="two", scope=ApprovalChoice.USER
    )
    with pytest.raises(ValueError, match="Sandbox command rule identifiers"):
        UserPermissionConfiguration(sandboxed_command_rules=[first, second])
    with pytest.raises(ValueError, match="all_sources"):
        SandboxedCommandRule.model_validate(
            {"signature": "a", "source": "one", "all_sources": True}
        )
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        yaml.safe_dump(
            {
                "sandboxed_command_rules": [
                    first.model_dump(mode="json"),
                    second.model_dump(mode="json"),
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionConfigurationError, match="identifiers must be unique"):
        PermissionManager(tmp_path, user_configuration_path=user_path)
    user_path.write_text(
        yaml.safe_dump(
            {"sandboxed_command_rules": [{**first.model_dump(mode="json"), "all_sources": True}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionConfigurationError, match="all_sources"):
        PermissionManager(tmp_path, user_configuration_path=user_path)


def test_native_workspace_approval_is_not_saved_in_repository_policy(tmp_path):
    """A repository policy cannot become the source of a native allow decision."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "workspace"
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    assert manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)
    workspace_policy = tmp_path / ".loop" / "permissions.yaml"
    assert (
        not workspace_policy.exists()
        or "sandboxed_command_rules" not in workspace_policy.read_text()
    )
    assert "sandboxed_command_rules" in user_path.read_text()


def test_native_session_rule_resets_and_rejects_invalid_scope(tmp_path):
    """Session approval is cleared by reset and cannot use a one-off scope."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    assert len(manager.sandboxed_command_rules(ApprovalChoice.SESSION)) == 1
    assert manager.reset_session()
    assert not manager.sandboxed_command_rules(ApprovalChoice.SESSION)
    with pytest.raises(ValueError, match="require session"):
        manager.sandboxed_command_rules(ApprovalChoice.ONCE)


def test_native_similar_rule_does_not_match_changed_environment(tmp_path):
    """A similar source cannot reuse authority after environment changes."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "similar_session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    interaction.prompt.return_value = "deny"
    changed = native_request(tmp_path, environment={"PATH": "/bin", "LANG": "C"})
    assert not manager.authorize_sandboxed_command(changed, tool_id="run_command")


def test_native_derived_cache_path_keeps_approval_but_external_cache_does_not(tmp_path):
    """Per-command scratch changes reuse approval while an external cache path cannot."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path, interaction=interaction)

    def request(scratch: Path, cache: Path, ruff_cache: Path | None = None) -> SandboxRequest:
        """Bind one command with an explicit process cache location."""
        return native_request(
            tmp_path,
            generated_environment=tuple(
                name
                for name, generated in (
                    ("TMPDIR", True),
                    ("XDG_CACHE_HOME", cache == scratch / "cache"),
                    ("RUFF_CACHE_DIR", ruff_cache is None),
                    ("COVERAGE_FILE", True),
                )
                if generated
            ),
            environment={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "TMPDIR": str(scratch),
                "XDG_CACHE_HOME": str(cache),
                "RUFF_CACHE_DIR": str(ruff_cache or scratch / "ruff-cache"),
                "COVERAGE_FILE": str(scratch / ".coverage"),
            },
        )

    with (
        tempfile.TemporaryDirectory(
            prefix="loop-seatbelt-", dir=Path(tempfile.gettempdir()).resolve()
        ) as first_value,
        tempfile.TemporaryDirectory(
            prefix="loop-seatbelt-", dir=Path(tempfile.gettempdir()).resolve()
        ) as second_value,
    ):
        first = Path(first_value)
        second = Path(second_value)
        assert manager.authorize_sandboxed_command(
            request(first, first / "cache"), tool_id="run_command"
        )
        assert manager.authorize_sandboxed_command(
            request(second, second / "cache"), tool_id="run_command"
        )
        interaction.prompt.assert_called_once()
        interaction.prompt.return_value = "deny"
        assert not manager.authorize_sandboxed_command(
            request(second, tmp_path / "external-cache"), tool_id="run_command"
        )
        assert interaction.prompt.call_count == 2
        assert not manager.authorize_sandboxed_command(
            request(second, second / "cache", tmp_path / "external-ruff-cache"),
            tool_id="run_command",
        )
        assert interaction.prompt.call_count == 3


def test_native_exception_grant_prompts_with_exact_path(tmp_path):
    """An extra read root prompts without exposing its local path."""
    external = tmp_path / "external"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "cat extra", read_roots=(external,), write_roots=())
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert "read files outside this workspace" in interaction.info.call_args.args[0]
    assert str(external) not in interaction.info.call_args.args[0]


def test_native_read_prompt_distinguishes_tool_folder_data_and_broad_scope(tmp_path):
    """Extra-read requests disclose their practical scope without exposing path details."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(workspace, interaction=interaction)
    cases = (
        ((external,), "Use installed python3"),
        ((external,), "Read an additional folder outside the workspace"),
        ((tmp_path,), "Read a broad area outside the workspace"),
    )
    for roots, expected in cases:
        context = expected if expected.startswith("Use installed") else None
        request = native_request(
            workspace, "opaque-command", read_roots=roots, read_context=context
        )
        assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
        message = interaction.info.call_args.args[0]
        assert f"Context: {expected}." in message
        assert 'Command: "opaque-command"' in message
        assert str(external) not in message
        assert str(tmp_path) not in message


def test_native_automatic_tool_reads_are_not_presented_as_permission_reasons(tmp_path):
    """A destructive review does not claim that already authorized tool reads need approval."""
    tool_root = tmp_path.parent / "installed-tool"
    tool_root.mkdir()
    executable = tool_root / "tool"
    executable.write_text("installed", encoding="utf-8")
    executable.chmod(0o700)
    details = executable.stat()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    bound = native_request(
        tmp_path,
        "rm victim",
        read_roots=(tool_root,),
        automatic_tool_reads=(tool_root,),
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
        environment={"PATH": str(tool_root)},
        read_context="Use installed developer tools",
    )
    assert not manager.authorize_sandboxed_command(bound, tool_id="run_command")
    message = interaction.info.call_args.args[0]
    assert "Review a destructive workspace command" in message
    assert "read files outside" not in message
    assert "Use installed developer tools" not in message


@pytest.mark.parametrize("escape", [False, True])
def test_native_redirection_exemption_uses_bound_temporary_alias(tmp_path, escape):
    """Only a translated, contained temporary destination can bypass output review."""
    scratch = tmp_path.parent / "managed-scratch"
    scratch.mkdir()
    alias = tmp_path.parent / "temporary-alias"
    alias.symlink_to(scratch, target_is_directory=True)
    (scratch / "link").symlink_to(tmp_path, target_is_directory=True)
    suffix = "link/victim" if escape else "report"
    bound = native_request(
        tmp_path,
        f"printf text >/tmp/{suffix}",
        execution_source=f"printf text >{alias}/{suffix}",
        aliases=(("/tmp", alias, scratch),),
        read_roots=(),
        write_roots=(tmp_path, scratch),
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(bound, tool_id="run_command") is not escape
    assert interaction.prompt.call_count == int(escape)


def test_native_literal_command_scratch_redirection_needs_no_review(tmp_path):
    """The bound private command TMPDIR is an output exemption without trusting variables."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        interaction = Mock(spec=Interaction)
        interaction.prompt.return_value = "deny"
        manager = PermissionManager(tmp_path, interaction=interaction)
        bound = native_request(
            tmp_path,
            f"printf text >{scratch}/report",
            read_roots=(),
            environment={"TMPDIR": str(scratch)},
        )
        assert manager.authorize_sandboxed_command(bound, tool_id="run_command")
        interaction.prompt.assert_not_called()


def test_native_read_scope_audit_keeps_exact_roots_out_of_prompt(tmp_path):
    """Local audit retains exact approved scope while the human message stays plain language."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "tool"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    audit_path = tmp_path / "audit.db"
    manager = PermissionManager(
        workspace, interaction=interaction, audit_path=audit_path, workspace_id="workspace"
    )
    approved = native_request(
        workspace, "opaque-tool", read_roots=(external,), read_context="Use installed tool"
    )
    assert manager.authorize_sandboxed_command(approved, tool_id="run_command")
    assert str(external) not in interaction.info.call_args.args[0]
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records "
            "WHERE event_name = 'sandbox.permission_scope'"
        ).fetchone()
    assert json.loads(row[0])["read_roots"] == [str(external)]


@pytest.mark.parametrize("source", ["python -c print", "git --version"])
def test_native_similar_choice_excludes_script_engines_and_flag_prefixes(tmp_path, source):
    """Broad script engines and option prefixes cannot create similar rules."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, source), tool_id="run_command"
    )
    assert "similar_session" not in interaction.prompt.call_args.kwargs["choices"]


def test_native_persistence_failure_denies_approval(tmp_path, monkeypatch):
    """A failed remembered-rule write cannot be treated as permission."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "workspace"
    manager = PermissionManager(
        tmp_path, user_configuration_path=tmp_path / "user.yaml", interaction=interaction
    )
    monkeypatch.setattr(
        manager, "_replace_user_configuration", Mock(side_effect=OSError("disk full"))
    )
    assert not manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")


def test_native_rule_removal_rejects_once_scope_and_removes_session_rule(tmp_path):
    """One-off approvals are never managed as persistent rules."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    rule_id = manager.sandboxed_command_rules(ApprovalChoice.SESSION)[0].id
    assert manager.remove_sandboxed_command_rule(ApprovalChoice.SESSION, rule_id)
    with pytest.raises(ValueError, match="require session"):
        manager.remove_sandboxed_command_rule(ApprovalChoice.ONCE, rule_id)


def test_legacy_process_approval_never_approves_native_sandbox_request(tmp_path):
    """Persisted v1 host rules and process defaults cannot skip fresh native approval."""
    configuration = PermissionConfiguration(
        defaults={**PermissionConfiguration().defaults, Action.PROCESS_EXECUTE: Decision.ALLOW},
        limits=PolicyLimits(allow_host_processes=True),
        rules=[
            PermissionRule(
                decision=Decision.ALLOW,
                tool="run_command",
                action=Action.PROCESS_EXECUTE,
                resource="*",
            )
        ],
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    policy_path = tmp_path / "permissions.yaml"
    policy_path.write_text(yaml.safe_dump(configuration.model_dump(mode="json")))
    user_rule = PermissionRule(
        decision=Decision.ALLOW,
        tool="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf ok"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.HOST,
        ),
    )
    user_path = tmp_path / "user-permissions.yaml"
    user_path.write_text(
        yaml.safe_dump({"version": 1, "rules": [user_rule.model_dump(mode="json")]})
    )
    manager = PermissionManager(
        tmp_path, configuration_path=policy_path, user_configuration_path=user_path
    )
    request = SandboxRequest.create(
        source="printf ok",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(tmp_path.parent,),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="macos-v1",
        deadline=time.monotonic() + 60,
        workspace_id="workspace-1",
    )

    assert (
        manager.authorize_sandboxed_command(request, interaction=interaction, tool_id="run_command")
        is False
    )
    interaction.prompt.assert_called_once()
    assert not manager.configuration.rules[0].target
    interaction.prompt.return_value = "approve"
    assert (
        manager.authorize_sandboxed_command(request, interaction=interaction, tool_id="run_command")
        is True
    )
    assert len(manager.configuration.rules) == 1


def test_shutdown_cleanup_releases_live_manager_temporary_directories():
    """Process shutdown cleanup releases temporary directories retained by live managers."""
    manager = PermissionManager()

    manager_module._close_live_managers()  # pylint: disable=protected-access

    assert not manager.temporary_directory.exists()


@pytest.mark.parametrize(
    ("workspace_available", "user_available", "selection", "expected", "workspace_label"),
    [
        (True, True, ApprovalChoice.WORKSPACE, ApprovalChoice.WORKSPACE, True),
        (False, False, ApprovalChoice.SESSION, ApprovalChoice.SESSION, False),
        (True, True, False, ApprovalChoice.DENY, True),
    ],
)
def test_request_permission_offers_valid_scopes_and_fails_closed(
    workspace_available,
    user_available,
    selection,
    expected,
    workspace_label,
    tmp_path,
):
    """Permission prompting owns scoped labels and converts cancellation to denial."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = selection
    recorder = Mock()
    manager = PermissionManager(
        tmp_path if workspace_available else None,
        user_configuration_path=tmp_path / "user.yaml" if user_available else None,
        interaction=interaction,
        recorder=recorder,
    )

    result = manager.request_permission(
        "Approve operations?",
        interaction=interaction,
    )

    assert result is expected
    choices = interaction.prompt.call_args.kwargs["choices"]
    assert (ApprovalChoice.WORKSPACE in choices) is workspace_label
    assert (ApprovalChoice.USER in choices) is user_available


def test_request_permission_forwards_index_map_to_prompt(tmp_path):
    """Permission prompts forward short letter indexes to the generic prompt layer."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path, user_configuration_path=tmp_path / "user.yaml", interaction=interaction
    )

    manager.request_permission("Approve?")

    index = interaction.prompt.call_args.kwargs["index"]
    assert index is not None
    assert index[ApprovalChoice.DENY] == "N"
    assert index[ApprovalChoice.ONCE] == "Y"
    assert index[ApprovalChoice.SESSION] == "S"
    assert index[ApprovalChoice.USER] == "U"
    interaction.info.assert_called_once_with("Approve?")
    assert interaction.prompt.call_args.args[0] == "Proceed?"


def test_request_permission_includes_workspace_index_when_configured(tmp_path):
    """Workspace scope adds a ``W`` index letter when a configuration path exists."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction)

    manager.request_permission("Approve?")

    index = interaction.prompt.call_args.kwargs["index"]
    assert index[ApprovalChoice.WORKSPACE] == "W"


def test_request_permission_displays_details_before_choices_and_proceed_prompt(tmp_path):
    """Permission details precede the approval catalog and its final input prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    calls = Mock()
    calls.attach_mock(interaction.info, "info")
    calls.attach_mock(interaction.prompt, "prompt")
    manager = PermissionManager(tmp_path, interaction=interaction)

    manager.request_permission("Approve these operations?")

    assert calls.mock_calls[0] == call.info("Approve these operations?")
    assert calls.mock_calls[1].args == ("Proceed?",)


def test_user_approval_persists_across_workspace_managers(tmp_path):
    """A remembered user approval authorizes its exact operation in another workspace."""
    user_path = tmp_path / "user" / "permissions.yaml"
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.USER
    requested = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com/api", origin="https://example.com"),
    )
    first = PermissionManager(
        first_workspace, user_configuration_path=user_path, interaction=interaction
    )

    approved = first.authorize((requested,))
    second = PermissionManager(second_workspace, user_configuration_path=user_path)

    assert approved.decision is Decision.ALLOW
    assert approved.approval_choice is ApprovalChoice.USER
    assert approved.installed_rule_ids
    assert second.evaluate((requested,)).decision is Decision.ALLOW
    assert len(second.user_rules) == 1


def test_workspace_deny_overrides_a_matching_user_approval(tmp_path):
    """A workspace-local denial remains stronger than a user-wide exact allow rule."""
    user_path = tmp_path / "user" / "permissions.yaml"
    requested = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com/api", origin="https://example.com"),
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.USER
    approving = PermissionManager(
        tmp_path / "first", user_configuration_path=user_path, interaction=interaction
    )
    approving.authorize((requested,))
    denying = PermissionManager(
        tmp_path / "second",
        user_configuration_path=user_path,
        configuration=PermissionConfiguration(
            rules=[
                PermissionRule(
                    decision=Decision.DENY,
                    tool=requested.tool_id,
                    tool_exact=True,
                    action=requested.action,
                    target=requested.target,
                )
            ]
        ),
    )

    result = denying.evaluate((requested,))

    assert result.decision is Decision.DENY
    assert result.sources[0].startswith("rule:workspace:")


def test_invalid_user_policy_is_ignored_without_being_replaced(tmp_path):
    """Invalid user approvals fail closed and remain available for manual repair."""
    user_path = tmp_path / "user" / "permissions.yaml"
    user_path.parent.mkdir()
    user_path.write_text("version: invalid\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)

    manager = PermissionManager(
        tmp_path / "workspace", user_configuration_path=user_path, interaction=interaction
    )

    assert not manager.user_rules
    assert user_path.read_text(encoding="utf-8") == "version: invalid\n"
    interaction.report.assert_called_once()
    interaction.warning.assert_called_once_with(
        "Ignoring user-wide approvals until the policy is fixed."
    )


def test_invalid_user_policy_raises_in_strict_mode(tmp_path):
    """Strict loading exposes an invalid user policy instead of applying any approval."""
    user_path = tmp_path / "user" / "permissions.yaml"
    user_path.parent.mkdir()
    user_path.write_text("version: invalid\n", encoding="utf-8")

    with pytest.raises(PermissionConfigurationError, match="permissions.yaml"):
        PermissionManager(tmp_path / "workspace", user_configuration_path=user_path)


def test_invalid_user_policy_reports_without_interaction_in_automatic_mode(tmp_path):
    """Automatic loading logs and ignores invalid user approvals in a headless process."""
    user_path = tmp_path / "user" / "permissions.yaml"
    user_path.parent.mkdir()
    user_path.write_text("version: invalid\n", encoding="utf-8")

    manager = PermissionManager(
        tmp_path / "workspace",
        user_configuration_path=user_path,
        load_policy=PermissionLoadPolicy.AUTO,
    )

    assert not manager.user_rules


def test_unavailable_user_approval_fails_closed(tmp_path, monkeypatch):
    """An unavailable user policy path cannot authorize a spoofed user selection."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    monkeypatch.setattr(manager, "request_permission", Mock(return_value=ApprovalChoice.USER))

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert result.decision is Decision.DENY
    assert result.approval_choice is ApprovalChoice.DENY
    assert manager.user_configuration_path is None


def test_default_policy_allows_scoped_reads_and_fails_closed_for_approval(tmp_path):
    """The supervised default permits workspace inspection and denies headless mutations."""
    manager = PermissionManager(tmp_path)

    read = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))
    write = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert read.decision is Decision.ALLOW
    assert read.policy.decision is Decision.ALLOW
    assert write.policy.decision is Decision.ASK
    assert write.decision is Decision.DENY
    assert write.source == "headless"
    assert not (tmp_path / ".loop" / "permissions-audit.jsonl").exists()


def test_permission_audit_uses_central_sqlite_storage(tmp_path):
    """Permission audit storage records workspace-correlated SQLite rows."""
    audit_path = tmp_path / "state" / "audit.db"
    manager = PermissionManager(tmp_path, audit_path=audit_path, workspace_id="workspace")

    for tool_id in ("first", "second", "third", "fourth"):
        manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / f"{tool_id}.txt"),))

    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT workspace_id, event_name FROM permission_audit_records"
        ).fetchall()
    assert rows == [("workspace", "permission.decided")] * 4
    assert audit_path.stat().st_mode & 0o777 == 0o600


def test_permission_audit_rotation_failure_does_not_change_authorization(tmp_path, monkeypatch):
    """A failed audit archive rotation cannot turn an allowed operation into a denial."""
    monkeypatch.setattr("loop.constants.DEFAULT_PERMISSIONS_AUDIT_BYTES", 1)
    manager = PermissionManager(tmp_path)
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),)
    manager.authorize(operation_set)
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("unavailable")))

    result = manager.authorize(operation_set)

    assert result.decision is Decision.ALLOW


def test_sqlite_permission_audit_failure_does_not_change_authorization(tmp_path, monkeypatch):
    """A failed centralized audit insert cannot change an authorization decision."""
    manager = PermissionManager(
        tmp_path,
        audit_path=tmp_path / "audit.db",
        workspace_id="workspace",
    )
    monkeypatch.setattr(
        "loop.permissions.audit.SQLitePermissionAudit.append",
        Mock(side_effect=sqlite3.OperationalError("busy")),
    )

    result = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))

    assert result.decision is Decision.ALLOW


def test_authorization_approves_one_complete_operation_set_and_records_policy(tmp_path):
    """One prompt and recorder event preserve both policy and effective outcomes."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    recorder = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction, recorder=recorder)
    operations = (
        file_operation(Action.FILESYSTEM_CREATE, tmp_path / "a.txt"),
        file_operation(Action.FILESYSTEM_DELETE, tmp_path / "b.txt"),
    )
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)

    try:
        result = manager.authorize(operations)
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert result.policy.decision is Decision.ASK
    assert result.decision is Decision.ALLOW
    assert result.prompted is True
    assert result.prompt is not None
    assert "filesystem.create" in result.prompt
    assert "filesystem.delete" in result.prompt
    interaction.prompt.assert_called_once()
    recorder.record_authorization.assert_called_once_with(result)
    assert adapter.records[0].event_name == "permission.decided"
    assert adapter.records[0].attributes["decision"] == "allow"
    assert adapter.records[1].event_name == "permission.decision"
    assert thaw(adapter.records[1].payload) == result.model_dump(mode="json")


def test_authorization_rejection_and_recorder_override_are_atomic(tmp_path):
    """A rejected batch is recorded once by an invocation-scoped recorder."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.DENY
    configured = Mock()
    override = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction, recorder=configured)

    result = manager.authorize(
        (file_operation(Action.FILESYSTEM_REPLACE, tmp_path / "a.txt"),),
        recorder=override,
    )

    assert result.decision is Decision.DENY
    assert result.source == "user"
    override.record_authorization.assert_called_once_with(result)
    configured.record_authorization.assert_not_called()
    assert manager.recorder is configured


def test_session_approval_remembers_exact_batch_targets_and_deduplicates(tmp_path):
    """Session approval installs exact grants once and reuses them without another prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    first = file_operation(Action.FILESYSTEM_CREATE, tmp_path / "literal[*].txt")
    second = file_operation(Action.FILESYSTEM_CREATE, tmp_path / "other.txt")

    approved = manager.authorize((first, second))
    repeated = manager.authorize((first, second))
    different = manager.evaluate(
        (file_operation(Action.FILESYSTEM_CREATE, tmp_path / "literal-x.txt"),)
    )

    assert approved.approval_choice is ApprovalChoice.SESSION
    assert len(approved.installed_rule_ids) == 2
    assert repeated.prompted is False
    assert repeated.decision is Decision.ALLOW
    assert different.decision is Decision.ASK
    assert len(manager.session_rules) == 2
    interaction.prompt.assert_called_once()


def test_session_approval_matches_tool_identifiers_literally(tmp_path):
    """Generated grants do not interpret glob characters in tool identifiers."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    target = FileTarget(path=str(tmp_path / "file.txt"))
    approved = operation(Action.FILESYSTEM_CREATE, tool="writer[*]", target=target)

    manager.authorize((approved,))

    assert manager.evaluate((approved,)).decision is Decision.ALLOW
    assert (
        manager.evaluate(
            (operation(Action.FILESYSTEM_CREATE, tool="writerx", target=target),)
        ).decision
        is Decision.ASK
    )


def test_workspace_approval_persists_exact_rules_and_audit_metadata(tmp_path):
    """Workspace approval atomically persists reusable typed grants and their audit identity."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    manager = PermissionManager(tmp_path, interaction=interaction)
    target = file_operation(Action.FILESYSTEM_CREATE, tmp_path / "module.py")

    approved = manager.authorize((target, target))
    reloaded = PermissionManager(tmp_path)

    assert approved.decision is Decision.ALLOW
    assert approved.approval_choice is ApprovalChoice.WORKSPACE
    assert len(approved.installed_rule_ids) == 1
    assert reloaded.persistent_rules[0].tool_exact is True
    assert "tool_match=exact" in reloaded.describe("workspace")
    assert reloaded.authorize((target,)).decision is Decision.ALLOW


@pytest.mark.parametrize(
    ("action", "relative"),
    [
        (Action.FILESYSTEM_REPLACE, "ordinary.txt"),
        (Action.FILESYSTEM_DELETE, "ordinary.txt"),
        (Action.FILESYSTEM_CREATE, "AGENTS.md"),
        (Action.FILESYSTEM_CREATE, ".agents/skills/demo/SKILL.md"),
        (Action.FILESYSTEM_CREATE, "nested/.agents/skills/demo/asset.txt"),
        (Action.FILESYSTEM_CREATE, "nested/.gitignore"),
        (Action.FILESYSTEM_CREATE, "nested/.agentignore"),
    ],
)
def test_user_data_and_instruction_changes_require_fresh_one_time_approval(
    tmp_path, action, relative
):
    """Broad allow rules and earlier consent never authorize another protected mutation."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = [ApprovalChoice.ONCE, ApprovalChoice.SESSION]
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(action, Decision.ALLOW, scope=PolicyScope.SESSION)
    change = file_operation(action, tmp_path / relative)

    first = manager.authorize((change,))
    second = manager.authorize((change,))

    assert first.decision is Decision.ALLOW
    assert first.approval_choice is ApprovalChoice.ONCE
    assert second.decision is Decision.DENY
    assert not first.installed_rule_ids
    assert not manager.session_rules
    assert interaction.prompt.call_count == 2
    assert all(
        set(call.kwargs["choices"]) == {ApprovalChoice.DENY, ApprovalChoice.ONCE}
        for call in interaction.prompt.call_args_list
    )


def test_new_ordinary_file_keeps_explicit_allow_policy(tmp_path):
    """A normal creation under an allow rule needs no fresh data-loss decision."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_CREATE, Decision.ALLOW, scope=PolicyScope.SESSION)

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert result.decision is Decision.ALLOW
    interaction.prompt.assert_not_called()


def test_configured_instruction_change_requires_fresh_decision(tmp_path):
    """An active custom instruction name cannot inherit a workspace-wide create grant."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.DENY
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_CREATE, Decision.ALLOW, scope=PolicyScope.SESSION)

    result = manager.authorize(
        (file_operation(Action.FILESYSTEM_CREATE, tmp_path / "nested" / "POLICY.md"),),
        protected_instruction_names=("POLICY.md",),
    )

    assert result.decision is Decision.DENY
    assert "boundary:fresh_file_review" in result.policy.sources
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {
        ApprovalChoice.DENY,
        ApprovalChoice.ONCE,
    }


def test_manager_rejects_unsafe_instruction_names(tmp_path):
    """Permission authority binds control paths and rejects unsafe instruction names."""
    manager = PermissionManager(tmp_path)

    paths = manager.protected_sandbox_paths(("POLICY.md",))
    assert paths.instruction_names == ("AGENTS.md", "SKILL.md", "POLICY.md")
    assert paths.files == (".gitignore", ".agentignore")

    for invalid in ("../POLICY.md", "a/b.md", "a\\b.md", "", ".", "a\x00b", 7):
        with pytest.raises(ValueError, match="plain filenames"):
            manager.protected_instruction_names((invalid,))


def test_manager_without_workspace_does_not_classify_a_new_file_as_instructions(tmp_path):
    """Only a bound workspace can identify protected instruction creation paths."""
    manager = PermissionManager()

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "AGENTS.md"),))

    assert "boundary:fresh_file_review" not in result.policy.sources


def test_policy_mutations_write_timestamped_local_and_structured_audit_records(tmp_path):
    """Permission configuration changes remain auditable with and without telemetry storage."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    manager = PermissionManager(tmp_path)

    try:
        manager.set_default(
            Action.FILESYSTEM_DELETE,
            Decision.DENY,
            scope=PolicyScope.SESSION,
        )
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert adapter.records[0].event_name == "permission.default_set"
    assert adapter.records[0].attributes["scope"] == "session"


def test_process_grants_distinguish_argument_boundaries_working_directory_and_sandbox(tmp_path):
    """Remembered process approval compares argv, cwd, and execution boundary structurally."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    approved = operation(
        Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("tool", "a b"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    ambiguous = operation(
        Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("tool", "a", "b"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )

    manager.authorize((approved,))

    assert manager.evaluate((approved,)).decision is Decision.ALLOW
    assert manager.evaluate((ambiguous,)).decision is Decision.ASK
    assert (
        manager.evaluate(
            (
                approved.model_copy(
                    update={
                        "target": approved.target.model_copy(update={"cwd": str(tmp_path / "sub")})
                    }
                ),
            )
        ).decision
        is Decision.ASK
    )


def test_network_grants_ignore_dns_addresses_but_retain_request_semantics(tmp_path):
    """Network grants survive DNS changes without widening method or body semantics."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    get = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(
            url="https://my-host.local/data",
            origin="https://my-host.local",
            addresses=("93.184.216.34",),
        ),
    )

    manager.authorize((get,))

    changed_dns = get.model_copy(
        update={"target": get.target.model_copy(update={"addresses": ("93.184.216.35",)})}
    )
    post = get.model_copy(
        update={"target": get.target.model_copy(update={"method": "POST", "sends_body": True})}
    )
    assert manager.evaluate((changed_dns,)).decision is Decision.ALLOW
    assert manager.evaluate((post,)).decision is Decision.ASK


def test_failed_workspace_persistence_denies_without_activating_grants(tmp_path, monkeypatch):
    """A failed durable write fails closed and leaves the active policy unchanged."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    manager = PermissionManager(tmp_path, interaction=interaction)
    monkeypatch.setattr("pathlib.Path.write_text", Mock(side_effect=OSError("disk full")))

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert result.decision is Decision.DENY
    assert result.source == "persistence"
    assert not result.installed_rule_ids
    assert not manager.persistent_rules


def test_workspace_choice_is_rejected_when_no_workspace_is_available():
    """A malformed interaction cannot persist workspace trust without a policy path."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    manager = PermissionManager(interaction=interaction)

    result = manager.authorize(
        (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="setting")),)
    )

    assert result.decision is Decision.DENY
    assert "unavailable" in result.reason
    interaction.prompt.assert_called_once()


def test_no_authority_plan_is_allowed_without_prompting():
    """Pure tool plans require neither policy rules nor interactive approval."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(interaction=interaction)

    result = manager.authorize(())

    assert result.decision is Decision.ALLOW
    assert result.policy.sources == ("no_authority",)
    interaction.confirm.assert_not_called()


def test_interaction_property_and_non_persisted_default_changes():
    """In-memory policy controls can be replaced without requiring a policy path."""
    manager = PermissionManager()
    interaction = Mock(spec=Interaction)

    manager.interaction = interaction
    manager.set_default(Action.FILESYSTEM_READ, Decision.ALLOW, scope=PolicyScope.SESSION)

    assert manager.interaction is interaction
    assert manager.effective_configuration.defaults[Action.FILESYSTEM_READ] is Decision.ALLOW


def test_permission_rules_reject_ambiguous_exact_and_glob_matchers(tmp_path):
    """One rule cannot combine a broad resource glob with an exact typed target."""
    with pytest.raises(ValueError, match="cannot combine"):
        PermissionRule(
            decision=Decision.ALLOW,
            resource="*.txt",
            target=FileTarget(path=str(tmp_path / "file.txt")),
        )


def test_rule_composition_uses_forbid_then_approval_then_permit(tmp_path):
    """Forbid and approval duties monotonically constrain matching permits."""
    rules = [
        PermissionRule(id="permit", decision=Decision.ALLOW, tool="read_*"),
        PermissionRule(
            id="approval",
            decision=Decision.ASK,
            action=Action.FILESYSTEM_READ,
        ),
        PermissionRule(
            id="forbid",
            decision=Decision.DENY,
            resource="*/secret.*",
        ),
    ]
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(rules=rules),
    )

    secret = manager.evaluate(
        (
            operation(
                Action.FILESYSTEM_READ,
                tool="read_file",
                target=FileTarget(path=str(tmp_path / "secret.env")),
            ),
        )
    )
    ordinary = manager.evaluate(
        (
            operation(
                Action.FILESYSTEM_READ,
                tool="read_file",
                target=FileTarget(path=str(tmp_path / "ordinary.txt")),
            ),
        )
    )

    assert secret.decision is Decision.DENY
    assert secret.sources == ("rule:workspace:forbid",)
    assert ordinary.decision is Decision.ASK
    assert ordinary.sources == ("rule:workspace:approval",)


def test_any_denied_operation_denies_a_batch_without_prompting(tmp_path):
    """A hard denial prevents prompts for other approval-requiring operations."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)

    result = manager.authorize(
        (
            file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),
            file_operation(Action.FILESYSTEM_READ, tmp_path.parent / "outside.txt"),
        )
    )

    assert result.policy.decision is Decision.DENY
    assert result.decision is Decision.DENY
    interaction.confirm.assert_not_called()


@pytest.mark.parametrize(
    ("path", "action", "source"),
    [
        ("outside", Action.FILESYSTEM_READ, "limit:workspace:readable_roots"),
        (".loop/policy", Action.FILESYSTEM_READ, "boundary:protected_path"),
        (".git/config", Action.FILESYSTEM_READ, "boundary:protected_path"),
        (".gitignore", Action.FILESYSTEM_REPLACE, "boundary:protected_path"),
        (".agentignore", Action.FILESYSTEM_DELETE, "boundary:protected_path"),
    ],
)
def test_filesystem_boundaries_cannot_be_overridden(tmp_path, path, action, source):
    """Filesystem roots and control paths remain forbidden despite permit rules."""
    target = tmp_path.parent / "outside" if path == "outside" else tmp_path / path
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(rules=[PermissionRule(decision=Decision.ALLOW)]),
    )

    result = manager.evaluate((file_operation(action, target),))

    assert result.decision is Decision.DENY
    assert result.sources == (source,)


@pytest.mark.parametrize(
    ("action", "target", "limits", "source"),
    [
        (
            Action.FILESYSTEM_READ,
            lambda root: FileTarget(path=str(root.parent / "outside")),
            PolicyLimits(),
            "limit:workspace:readable_roots",
        ),
        (
            Action.NETWORK_REQUEST,
            lambda root: NetworkTarget(
                url="https://outside.test/data", origin="https://outside.test"
            ),
            PolicyLimits(network_origins=("https://allowed.test",)),
            "limit:workspace:network_origins",
        ),
        (
            Action.PROCESS_EXECUTE,
            lambda root: ProcessTarget(
                argv=("git", "status"), cwd=str(root), boundary=ProcessBoundary.HOST
            ),
            PolicyLimits(),
            "limit:workspace:allow_host_processes",
        ),
    ],
)
def test_check_boundaries_returns_first_hard_denial_without_policy_matching(
    tmp_path, action, target, limits, source
):
    """Boundary preflight rejects unsafe effects before rules or defaults are evaluated."""
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            defaults={action: Decision.ALLOW},
            limits=limits,
            rules=[PermissionRule(decision=Decision.ALLOW)],
        ),
    )
    operation_target = target(tmp_path)
    operation_value = operation(action, target=operation_target)

    result = manager.check_boundaries((operation_value,))

    assert result is not None
    assert result.decision is Decision.DENY
    assert result.sources == (source,)


def test_check_boundaries_allows_safe_plans_and_stops_at_first_denial(tmp_path):
    """Boundary preflight ignores policy-only effects and returns the first unsafe effect."""
    manager = PermissionManager(tmp_path)
    safe = file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt")
    unsafe = file_operation(Action.FILESYSTEM_READ, tmp_path.parent / "outside.txt")

    assert manager.check_boundaries((safe,)) is None
    result = manager.check_boundaries((safe, unsafe, safe))

    assert result is not None
    assert result.sources == ("limit:workspace:readable_roots",)


def test_recursive_mutations_cannot_remove_protected_directories(tmp_path):
    """Recursive mutations are denied when they would contain protected control data."""
    target = FileTarget(path=str(tmp_path), recursive=True)
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            rules=[PermissionRule(decision=Decision.ALLOW)],
        ),
    )

    result = manager.evaluate((operation(Action.FILESYSTEM_DELETE, target=target),))

    assert result.decision is Decision.DENY
    assert result.sources == ("boundary:protected_path",)


def test_explicit_filesystem_roots_expand_the_hard_boundary(tmp_path):
    """Configured absolute roots deliberately extend readable and writable scope."""
    outside = tmp_path.parent / "shared"
    configuration = PermissionConfiguration(
        limits=PolicyLimits(
            readable_roots=("workspace", str(outside)),
            writable_roots=(str(outside),),
        )
    )
    manager = PermissionManager(tmp_path, configuration=configuration)

    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_READ, outside / "a.txt"),)).decision
        is Decision.ALLOW
    )
    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_CREATE, outside / "a.txt"),)).decision
        is Decision.ASK
    )


def test_loop_temp_is_allowed_by_default_and_system_temp_requires_an_explicit_root(tmp_path):
    """Loop temporary storage is safe by default; the full OS temp directory is opt-in."""

    manager = PermissionManager(tmp_path)
    external_temp = Path(tempfile.gettempdir()) / "outside-loop-temporary-file"

    assert (
        manager.evaluate(
            (file_operation(Action.FILESYSTEM_READ, manager.temporary_directory / "file.txt"),)
        ).decision
        is Decision.ALLOW
    )
    assert manager.evaluate((file_operation(Action.FILESYSTEM_READ, external_temp),)).sources == (
        "limit:workspace:readable_roots",
    )

    manager.update_limit_values("readable_roots", "system-temp", add=True)
    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_READ, external_temp),)).decision
        is Decision.ALLOW
    )


def test_workspace_root_token_requires_a_configured_workspace(tmp_path):
    """An in-memory manager cannot resolve the special workspace root token."""
    manager = PermissionManager()

    result = manager.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))

    assert result.sources == ("limit:workspace:readable_roots",)


def test_approval_prompt_displays_targets_outside_the_workspace(tmp_path):
    """Approved expanded roots retain their absolute target in the prompt."""
    outside = tmp_path.parent / "shared" / "file.txt"
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(writable_roots=(str(outside.parent),))
        ),
    )

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, outside),))

    assert str(outside) in result.prompt


@pytest.mark.parametrize(
    "url",
    (
        "http://localhost/a",
        "http://127.0.0.1/a",
        "http://169.254.169.254/a",
        "http://[::1]/a",
    ),
)
def test_private_network_targets_are_hard_denied(url):
    """Local and non-global literal network targets cannot be approved."""
    parsed_origin = url.rsplit("/", 1)[0]
    target = NetworkTarget(url=url, origin=parsed_origin)

    result = PermissionManager().evaluate((operation(Action.NETWORK_REQUEST, target=target),))

    assert result.decision is Decision.DENY
    assert result.sources == ("limit:workspace:deny_private_networks",)


def test_network_origin_allowlist_is_a_non_overridable_boundary():
    """Configured origin globs reject otherwise permitted destinations."""
    manager = PermissionManager(
        configuration=PermissionConfiguration(
            limits=PolicyLimits(network_origins=("https://*.my-host.local",))
        )
    )
    allowed = NetworkTarget(url="https://api.my-host.local/a", origin="https://api.my-host.local")
    denied = NetworkTarget(url="https://other.test/a", origin="https://other.test")

    assert (
        manager.evaluate((operation(Action.NETWORK_REQUEST, target=allowed),)).decision
        is Decision.ASK
    )
    assert manager.evaluate((operation(Action.NETWORK_REQUEST, target=denied),)).sources == (
        "limit:workspace:network_origins",
    )


def test_hostname_resolution_fails_closed_for_private_and_unresolved_addresses():
    """Network policy leaves hostname resolution to the pinned request transport."""
    target = NetworkTarget(url="https://service.test/a", origin="https://service.test")
    assert (
        PermissionManager().evaluate((operation(Action.NETWORK_REQUEST, target=target),)).decision
        is Decision.ASK
    )


def test_relative_roots_resolve_against_workspace_and_temp_is_manager_owned(tmp_path):
    """Portable YAML roots and scratch directories are scoped to their manager."""
    first = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(limits=PolicyLimits(readable_roots=("shared",))),
    )
    second = PermissionManager(tmp_path)

    assert (
        first.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path / "shared/a"),)).decision
        is Decision.ALLOW
    )
    assert first.temporary_directory != second.temporary_directory
    assert first.temporary_directory.is_dir()


def test_host_processes_require_an_explicit_boundary_opt_in(tmp_path):
    """Policy rules cannot authorize host-process execution by default."""
    target = ProcessTarget(argv=("git", "status"), cwd=str(tmp_path), boundary=ProcessBoundary.HOST)
    denied = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(rules=[PermissionRule(decision=Decision.ALLOW)]),
    )
    allowed = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
            defaults={Action.PROCESS_EXECUTE: Decision.ALLOW},
        ),
    )

    assert denied.evaluate((operation(Action.PROCESS_EXECUTE, target=target),)).sources == (
        "limit:workspace:allow_host_processes",
    )
    assert (
        allowed.evaluate((operation(Action.PROCESS_EXECUTE, target=target),)).decision
        is Decision.ALLOW
    )


def test_policy_mutations_persist_defaults_and_rule_lifetimes(tmp_path):
    """Policy changes round-trip while session rules remain process-local."""
    manager = PermissionManager(tmp_path)
    persisted = PermissionRule(id="persisted", decision=Decision.ALLOW, tool="read_*")
    transient = PermissionRule(id="transient", decision=Decision.DENY, tool="danger")

    manager.set_default(Action.FILESYSTEM_CREATE, Decision.DENY)
    manager.set_default(Action.NETWORK_REQUEST, Decision.ASK)
    manager.add_rule(persisted)
    manager.add_rule(transient, scope=PolicyScope.SESSION)

    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.defaults[Action.FILESYSTEM_CREATE] is Decision.DENY
    assert loaded.configuration.defaults[Action.NETWORK_REQUEST] is Decision.ASK
    assert loaded.configuration.rules == [persisted]
    assert manager.remove_rule("transient", scope=PolicyScope.SESSION) is True
    assert manager.remove_rule("missing", scope=PolicyScope.SESSION) is False
    assert manager.remove_rule("persisted") is True
    assert manager.remove_rule("missing") is False


def test_preset_replacement_changes_the_selected_defaults_and_rule_layer(tmp_path):
    """A preset replaces selected-scope defaults and rules while preserving limits and overlays."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    manager.set_limit("allow_host_processes", True)
    manager.add_rule(PermissionRule(id="old-workspace", decision=Decision.DENY))
    manager.add_rule(
        PermissionRule(
            id="session-guard",
            decision=Decision.DENY,
            action=Action.PROCESS_EXECUTE,
        ),
        scope=PolicyScope.SESSION,
    )

    preview = manager.preview_preset_replacement(
        "workspace",
        scope=PolicyScope.WORKSPACE,
    )
    manager.replace_preset(preview)

    assert [rule.id for rule in preview.removed_rules] == ["old-workspace"]
    assert preview.removed_defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    assert not manager.persistent_rules
    assert manager.configuration.defaults[Action.FILESYSTEM_CREATE] is Decision.ALLOW
    assert manager.configuration.defaults[Action.FILESYSTEM_REPLACE] is Decision.ALLOW
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK
    assert manager.configuration.limits.allow_host_processes is True
    assert [rule.id for rule in manager.session_rules] == ["session-guard"]
    assert (
        manager.explain("write_text_file", Action.FILESYSTEM_CREATE, str(tmp_path / "new")).decision
        is Decision.ALLOW
    )
    assert not PermissionManager(tmp_path).session_rules


def test_preset_replacement_rejects_a_stale_preview_and_supports_session_scope(tmp_path):
    """Previews cannot overwrite later scoped changes and session profiles never persist."""
    manager = PermissionManager(tmp_path)
    preview = manager.preview_preset_replacement("locked", scope=PolicyScope.SESSION)
    manager.set_default(Action.FILESYSTEM_CREATE, Decision.DENY, scope=PolicyScope.SESSION)

    with pytest.raises(ValueError, match="stale"):
        manager.replace_preset(preview)

    replacement = manager.preview_preset_replacement("locked", scope=PolicyScope.SESSION)
    manager.replace_preset(replacement)

    assert set(manager.session_overrides.defaults.values()) == {Decision.DENY}
    assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.ALLOW
    assert not manager.session_rules
    assert not PermissionManager(tmp_path).persistent_rules


def test_preset_catalog_rejects_duplicate_custom_identifiers(tmp_path):
    """A caller cannot shadow a built-in preset with the same catalog identifier."""
    preset = PermissionPreset.model_validate(
        {
            "metadata": {
                "id": "observe",
                "revision": "custom",
                "title": "Duplicate",
                "description": "Duplicate identifier.",
            },
            "defaults": {action.value: "deny" for action in Action},
            "rules": [],
        }
    )

    with pytest.raises(ValueError, match="identifiers must be unique"):
        PermissionManager(tmp_path, presets=(preset,))


def test_preset_catalog_returns_copies_and_rejects_unknown_ids(tmp_path):
    """Preset lookup exposes isolated artifacts and reports missing catalog entries."""
    manager = PermissionManager(tmp_path)
    presets = manager.presets

    assert [preset.metadata.id for preset in presets] == sorted(
        preset.metadata.id for preset in presets
    )
    assert presets[0] is not manager.presets[0]
    assert manager.preset("workspace") == next(
        preset for preset in presets if preset.metadata.id == "workspace"
    )
    with pytest.raises(ValueError, match="Unknown permission preset 'missing'"):
        manager.preset("missing")


def test_preset_replacement_rejects_rule_id_collisions_with_other_scope(tmp_path):
    """A replacement cannot activate a preset rule already used by the other scope."""
    preset = PermissionPreset.model_validate(
        {
            "metadata": {
                "id": "collision",
                "revision": "1",
                "title": "Collision",
                "description": "Collision test.",
            },
            "defaults": {action.value: "deny" for action in Action},
            "rules": [{"id": "shared", "decision": "allow"}],
        }
    )
    manager = PermissionManager(tmp_path, presets=(preset,))
    manager.add_rule(
        PermissionRule(id="preset:workspace:collision:1:shared", decision=Decision.DENY),
        scope=PolicyScope.SESSION,
    )

    with pytest.raises(ValueError, match="already exists in session"):
        manager.replace_preset(
            manager.preview_preset_replacement("collision", scope=PolicyScope.WORKSPACE)
        )


def test_preset_requires_a_default_for_every_authority_action():
    """A preset cannot leave an action to the policy it is replacing."""
    with pytest.raises(ValueError, match="every known action"):
        PermissionPreset.model_validate(
            {
                "metadata": {
                    "id": "incomplete",
                    "revision": "1",
                    "title": "Incomplete",
                    "description": "Missing fallback decisions.",
                },
                "defaults": {Action.FILESYSTEM_READ.value: Decision.ALLOW.value},
                "rules": [],
            }
        )


def test_preset_rules_retain_the_selected_artifact_provenance(tmp_path):
    """Rules supplied by a complete preset retain its immutable diagnostic identity."""
    preset = PermissionPreset.model_validate(
        {
            "metadata": {
                "id": "custom",
                "revision": "1",
                "title": "Custom",
                "description": "Custom rule-bearing preset.",
            },
            "defaults": {action.value: "deny" for action in Action},
            "rules": [
                {
                    "id": "allow-read",
                    "decision": "allow",
                    "action": "filesystem.read",
                }
            ],
        }
    )
    manager = PermissionManager(tmp_path, presets=(preset,))

    manager.replace_preset(
        manager.preview_preset_replacement("custom", scope=PolicyScope.WORKSPACE)
    )

    installed = manager.persistent_rules[0]
    assert installed.source and installed.source.preset_id == "custom"
    assert "preset=custom@1" in manager.describe("workspace")


def test_rule_identifiers_are_unique_across_workspace_and_session_layers(tmp_path):
    """A rule identity always names exactly one active rule."""
    manager = PermissionManager(tmp_path)
    manager.add_rule(PermissionRule(id="unique", decision=Decision.ALLOW))

    with pytest.raises(ValueError, match="unique"):
        manager.add_rule(
            PermissionRule(id="unique", decision=Decision.DENY),
            scope=PolicyScope.SESSION,
        )

    with pytest.raises(ValueError, match="identifiers must be unique"):
        PermissionConfiguration(
            rules=[
                PermissionRule(id="duplicate", decision=Decision.ALLOW),
                PermissionRule(id="duplicate", decision=Decision.DENY),
            ]
        )


def test_limit_mutations_validate_names_deduplicate_and_support_in_memory_changes(tmp_path):
    """Limit APIs reject unknown fields and report collection changes accurately."""
    manager = PermissionManager(tmp_path)

    with pytest.raises(ValueError, match="Unknown boolean"):
        manager.set_limit("unknown", True)
    with pytest.raises(ValueError, match="Unknown collection"):
        manager.update_limit_values("unknown", "value", add=True)

    assert (
        manager.update_limit_values(
            "readable_roots", "workspace", add=True, scope=PolicyScope.SESSION
        )
        is False
    )
    assert (
        manager.update_limit_values(
            "readable_roots", str(tmp_path / "shared"), add=True, scope=PolicyScope.SESSION
        )
        is True
    )
    assert (
        manager.update_limit_values(
            "writable_roots", "relative", add=True, scope=PolicyScope.SESSION
        )
        is True
    )
    assert str(tmp_path / "relative") in manager.effective_configuration.limits.writable_roots
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    assert manager.effective_configuration.limits.deny_private_networks is False


def test_session_overrides_never_leak_into_later_workspace_saves(tmp_path):
    """Persisting workspace changes cannot serialize process-local policy state."""
    manager = PermissionManager(tmp_path)
    session_rule = PermissionRule(id="session-only", decision=Decision.DENY)

    manager.set_default(Action.FILESYSTEM_DELETE, Decision.ALLOW, scope=PolicyScope.SESSION)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.add_rule(session_rule, scope=PolicyScope.SESSION)
    manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)

    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK
    assert loaded.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY
    assert loaded.configuration.limits.allow_host_processes is False
    assert loaded.configuration.rules == []
    assert manager.effective_configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ALLOW
    assert manager.effective_configuration.limits.allow_host_processes is True
    assert manager.session_rules == (session_rule,)


def test_session_resets_restore_workspace_inheritance_without_changing_disk(tmp_path):
    """Per-field and whole-session resets reveal the underlying workspace policy."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY)
    manager.set_limit("allow_host_processes", True)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    manager.set_limit("allow_host_processes", False, scope=PolicyScope.SESSION)

    assert manager.reset_default(Action.PROCESS_EXECUTE) is True
    assert manager.reset_default(Action.PROCESS_EXECUTE) is False
    assert manager.reset_limit("allow_host_processes") is True
    assert manager.reset_limit("allow_host_processes") is False
    assert manager.effective_configuration.defaults[Action.PROCESS_EXECUTE] is Decision.DENY
    assert manager.effective_configuration.limits.allow_host_processes is True

    manager.add_rule(
        PermissionRule(id="temporary", decision=Decision.ASK),
        scope=PolicyScope.SESSION,
    )
    assert manager.reset_session() is True
    assert manager.reset_session() is False
    assert PermissionManager(tmp_path).configuration == manager.configuration


def test_workspace_resets_restore_application_bootstrap_values(tmp_path):
    """Workspace resets validate names and restore built-in defaults transactionally."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    manager.set_limit("deny_private_networks", False)

    assert manager.reset_default(Action.FILESYSTEM_READ, scope=PolicyScope.WORKSPACE) is True
    assert manager.reset_default(Action.FILESYSTEM_READ, scope=PolicyScope.WORKSPACE) is False
    assert manager.reset_limit("deny_private_networks", scope=PolicyScope.WORKSPACE) is True
    assert manager.reset_limit("deny_private_networks", scope=PolicyScope.WORKSPACE) is False
    with pytest.raises(ValueError, match="Unknown permission limit"):
        manager.reset_limit("unknown")


def test_policy_views_validate_names_and_render_sparse_session_overrides(tmp_path):
    """Each policy view is selectable and sparse session sections remain explicit."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)

    assert "Workspace policy:" in manager.describe("workspace")
    assert "Effective policy:" in manager.describe("effective")
    session = manager.describe("session")
    assert "Defaults:\n    none" in session
    assert "deny_private_networks: False" in session
    with pytest.raises(ValueError, match="Unknown permission policy view"):
        manager.describe("unknown")


def test_policy_mutations_are_transactional_when_persistence_fails(tmp_path, monkeypatch):
    """A failed atomic save leaves the active in-memory policy unchanged."""
    manager = PermissionManager(tmp_path)
    monkeypatch.setattr("pathlib.Path.write_text", Mock(side_effect=OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK


def test_explain_constructs_every_typed_target_without_prompting(tmp_path):
    """Effective-policy explanation handles absolute paths, URLs, processes, and session state."""
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True)),
    )

    assert (
        manager.explain("read", Action.FILESYSTEM_READ, str(tmp_path / "file.txt")).decision
        is Decision.ALLOW
    )
    assert (
        manager.explain("fetch", Action.NETWORK_REQUEST, "https://my-host.local:8443/file").decision
        is Decision.ASK
    )
    assert (
        manager.explain("fetch", Action.NETWORK_REQUEST, "https://my-host.local/file").decision
        is Decision.ASK
    )
    assert manager.explain("run", Action.PROCESS_EXECUTE, "git status").decision is Decision.ASK
    assert (
        manager.explain("skills", Action.SESSION_MUTATE, "activate:review").decision is Decision.ASK
    )
    with pytest.raises(ValueError, match="absolute HTTP"):
        manager.explain("fetch", Action.NETWORK_REQUEST, "relative")
    with pytest.raises(ValueError, match="non-empty command"):
        manager.explain("run", Action.PROCESS_EXECUTE, "")


def test_remove_rule_scans_past_nonmatching_rules():
    """Rule removal finds a requested identity beyond the first list entry."""
    manager = PermissionManager()
    manager.add_rule(PermissionRule(id="first", decision=Decision.ALLOW), scope=PolicyScope.SESSION)
    manager.add_rule(PermissionRule(id="second", decision=Decision.DENY), scope=PolicyScope.SESSION)

    assert manager.remove_rule("second", scope=PolicyScope.SESSION) is True
    result = manager.evaluate(
        (
            operation(
                Action.SESSION_MUTATE,
                target=SessionTarget(identifier="state"),
            ),
        )
    )
    assert result.sources == ("rule:session:first",)


def test_describe_exposes_defaults_boundaries_and_rule_identity(tmp_path):
    """Policy summaries expose every effective policy dimension."""
    manager = PermissionManager(tmp_path)
    manager.add_rule(
        PermissionRule(
            id="docs",
            decision=Decision.ALLOW,
            tool="read_*",
            action=Action.FILESYSTEM_READ,
            resource="*/docs/*",
        ),
        scope=PolicyScope.SESSION,
    )
    description = manager.describe()
    assert "Workspace policy:" in description
    assert "  Rules: none" in description
    assert "Session overrides:" in description

    manager.add_rule(
        PermissionRule(id="persisted", decision=Decision.DENY),
        scope=PolicyScope.SESSION,
    )
    configured = PermissionManager(
        configuration=PermissionConfiguration(
            rules=[PermissionRule(id="docs", decision=Decision.ALLOW)]
        )
    )

    description = configured.describe()
    assert "filesystem.read: allow" in description
    assert "readable_roots: workspace" in description
    assert "docs allow tool=*" in description


def test_in_memory_policy_cannot_be_saved():
    """Persistence without a local configuration path is rejected explicitly."""
    with pytest.raises(ValueError, match="cannot persist"):
        PermissionManager().save()


def test_empty_yaml_loads_as_default_policy(tmp_path):
    """An empty policy file is accepted as the supervised default."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("", "utf-8")

    assert PermissionManager(tmp_path).configuration == PermissionConfiguration()


def test_error_policy_raises_configuration_errors_and_preserves_active_policy(tmp_path):
    """Strict startup and reload expose typed errors without replacing valid active policy."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    path = tmp_path / ".loop" / "permissions.yaml"
    path.write_text("version: 2\n", "utf-8")

    with pytest.raises(PermissionConfigurationError) as raised:
        manager.reload()

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    assert raised.value.path == str(path)
    assert raised.value.__cause__ is not None
    with pytest.raises(PermissionConfigurationError):
        PermissionManager(tmp_path)


def test_auto_policy_reports_and_uses_defaults_or_last_known_good(tmp_path, caplog):
    """Automatic recovery reports startup defaults and retains valid policy on reload."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 2\n", "utf-8")

    manager = PermissionManager(tmp_path, load_policy=PermissionLoadPolicy.AUTO)

    assert manager.configuration == PermissionConfiguration()
    assert "automatic permission policy" in caplog.text
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    path.write_text("version: 2\n", "utf-8")
    interaction = Mock(spec=Interaction)
    manager.interaction = interaction

    assert manager.reload() is PermissionLoadResult.RETAINED
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()
    interaction.warning.assert_called_once()


def test_interactive_policy_retries_a_repaired_file(tmp_path):
    """Interactive recovery rereads a file repaired while the recovery prompt is active."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 2\n", "utf-8")
    interaction = Mock(spec=Interaction)

    def repair(*_args, **_kwargs):
        path.write_text("version: 1\ndefaults:\n  filesystem.delete: deny\n", "utf-8")
        return "retry"

    interaction.prompt.side_effect = repair
    manager = PermissionManager(tmp_path, interaction=interaction)

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()
    assert interaction.prompt.call_args.kwargs["index"] == {
        "retry": "R",
        "continue": "S",
        "reset": "A",
        "exit": "Q",
    }


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("continue", PermissionLoadResult.RETAINED), (None, PermissionLoadResult.RETAINED)],
)
def test_interactive_reload_can_retain_the_active_policy(tmp_path, choice, expected):
    """Interactive reload retains last-known-good state when the user continues."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    manager.configuration_path.write_text("version: 2\n", "utf-8")
    interaction.prompt.return_value = choice

    assert manager.reload() is expected
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.warning.assert_called_once_with("Keeping the current permission policy.")


def test_interactive_loading_requires_an_interaction(tmp_path):
    """Interactive loading cannot silently start or recover without an interaction."""
    with pytest.raises(ValueError, match="requires an Interaction"):
        PermissionManager(load_policy=PermissionLoadPolicy.INTERACTIVE)

    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.interaction = None
    manager.configuration_path.parent.mkdir(exist_ok=True)
    manager.configuration_path.write_text("version: 2\n", "utf-8")
    with pytest.raises(PermissionConfigurationError):
        manager.reload()


def test_preset_failures_raise_strictly_and_report_during_headless_auto_recovery(
    monkeypatch, caplog
):
    """Preset failures follow the same strict or reported recovery policy as configuration."""
    failure = PermissionLoadFailure(
        source="preset", path="presets/broken.yaml", message="invalid schema"
    )
    monkeypatch.setattr(
        PermissionPreset,
        "load_builtin_presets",
        classmethod(lambda cls: ((), (failure,))),
    )

    with pytest.raises(PermissionPresetError) as raised:
        PermissionManager()
    assert raised.value.failures == (failure,)

    manager = PermissionManager(load_policy=PermissionLoadPolicy.AUTO)
    assert not manager.presets
    assert "Excluded invalid permission preset" in caplog.text


def test_preset_error_requires_a_failure():
    """The aggregate preset error rejects an empty diagnostic collection."""
    with pytest.raises(ValueError, match="at least one failure"):
        PermissionPresetError(())


def test_reset_configuration_creates_defaults_when_no_policy_file_exists(tmp_path):
    """An explicit reset creates the default policy even before a policy file exists."""
    manager = PermissionManager(tmp_path)

    assert manager.reset_configuration() is None
    assert (tmp_path / ".loop" / "permissions.yaml").exists()


def test_in_memory_policy_cannot_be_reset():
    """Reset requires a local policy path to preserve its archival contract."""
    with pytest.raises(ValueError, match="cannot reset"):
        PermissionManager().reset_configuration()


def test_diagnostic_audit_failure_does_not_change_authorization(tmp_path, monkeypatch):
    """Unavailable diagnostic JSONL storage cannot turn a permit into a denial."""
    manager = PermissionManager(tmp_path)
    monkeypatch.setattr("pathlib.Path.open", Mock(side_effect=OSError("unavailable")))

    result = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))

    assert isinstance(result, AuthorizationResult)
    assert result.decision is Decision.ALLOW


def test_permission_manager_configuration_path_and_recorder(tmp_path):
    """The manager exposes its .loop policy path and a settable recorder sink."""
    recorder = Mock()
    workspace = PermissionManager(tmp_path)
    workspace.recorder = recorder
    assert workspace.recorder is recorder
    assert workspace.configuration_path.parent.parent == tmp_path

    in_memory = PermissionManager()
    assert in_memory.configuration_path is None


def test_approval_prompt_anchors_a_workspace_rooted_target(tmp_path):
    """A prompt for the workspace root renders an explicit workspace anchor."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_READ, Decision.ASK, scope=PolicyScope.SESSION)

    result = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path),))

    assert result.prompt is not None
    assert "workspace root:" in result.prompt


def test_approval_prompt_renders_session_targets_without_workspace():
    """A manager without a workspace renders session targets without a relative path."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(interaction=interaction)

    result = manager.authorize(
        (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="config")),)
    )

    assert result.prompt is not None
    assert "config" in result.prompt


def test_process_target_display_resolves_local_paths_to_workspace_virtual_paths(tmp_path):
    """Process target prompts render local workspace paths as VirtualPaths."""
    sub = tmp_path / "src" / "loop"
    sub.mkdir(parents=True)
    readme = tmp_path / "README.md"
    readme.touch()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    target = ProcessTarget(
        argv=("cat", str(readme)),
        cwd=str(sub),
        boundary=ProcessBoundary.HOST,
    )
    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    prompt = interaction.info.call_args.args[0]
    assert "cat /workspace/README.md" in prompt
    assert "(cwd: /workspace/src/loop)" in prompt
    assert "../" not in prompt


def test_process_target_display_preserves_workspace_relative_argv(tmp_path):
    """Relative argv that stays within workspace passes through cleanly."""
    target = ProcessTarget(
        argv=("cat", "README.md"),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    assert "cat README.md (cwd: /workspace)" in interaction.info.call_args.args[0]


def test_process_target_display_preserves_argument_boundaries(tmp_path):
    """Process prompts quote arguments so distinct argv remain visibly distinct."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    target = ProcessTarget(
        argv=("tool", "a b"),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    assert "tool 'a b' (cwd: /workspace)" in interaction.info.call_args.args[0]


def test_process_target_display_handles_temporary_virtual_paths(tmp_path):
    """Temporary directory paths are rendered below the temporary VirtualPath."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    scratch_file = manager.temporary_directory / "output.log"
    scratch_file.parent.mkdir(exist_ok=True)
    target = ProcessTarget(
        argv=("cat", str(scratch_file)),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    prompt = interaction.info.call_args.args[0]
    assert "/tmp/output.log" in prompt


def test_process_target_display_preserves_external_temporary_paths(tmp_path):
    """An external temporary path is not redacted in an approval prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    external_path = Path("/tmp/verify_review.py")
    target = ProcessTarget(
        argv=(".venv/bin/python", str(external_path)),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    prompt = interaction.info.call_args.args[0]
    assert str(external_path) in prompt
    assert "<external>" not in prompt
