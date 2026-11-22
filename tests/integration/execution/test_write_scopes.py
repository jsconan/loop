"""Exercise declared write capabilities through public tools and real native boundaries."""

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from loop import PermissionManager, ToolRegistry
from loop.configuration import ToolSettings
from loop.execution import CommandExecutionService
from loop.execution.sandbox import UnavailableSandboxBackend, select_sandbox_backend
from loop.interaction import ConsoleInteraction
from loop.permissions import (
    Action,
    ApprovalChoice,
    Decision,
    PermissionRule,
    PolicyScope,
    SandboxPathAccess,
)
from loop.sandbox_catalog import installed_paths
from loop.tooling import ToolRuntimeSettings
from loop.tools import BUILTIN_TOOLS

pytestmark = pytest.mark.integration
_NATIVE_PLATFORM = "MACOS" if sys.platform == "darwin" else "LINUX"
_SYSTEM_TOOL_PATH = os.pathsep.join(
    (*(str(path) for path in installed_paths() if path.name == "bin"), os.defpath)
)
_RG = shutil.which("rg", path=_SYSTEM_TOOL_PATH)
_RG = str(Path(_RG).resolve()) if _RG is not None else None
_GIT = shutil.which("git")
_CC = shutil.which("cc")
_CC = str(Path(_CC).resolve()) if _CC is not None else None
_NATIVE_MODE = "owned-volume" if sys.platform == "darwin" else "installed"
_NATIVE_ENABLED = os.environ.get(f"LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST") == _NATIVE_MODE


@pytest.fixture
def native_write_tool(native_directory, request):
    """Own a checkout, native service and explicit approval surface without host recovery."""
    if not _NATIVE_ENABLED:
        pytest.skip(
            f"Set LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST={_NATIVE_MODE} for native qualification"
        )
    workspace = native_directory("/private/tmp" if sys.platform == "darwin" else "/tmp")
    (workspace / "out").mkdir()
    (workspace / "ordinary").write_text("ordinary-canary")
    (workspace / "neighbor").write_text("unchanged")
    (workspace / ".env").write_text("hard-canary")
    backend = select_sandbox_backend()
    assert not isinstance(backend, UnavailableSandboxBackend), backend.capability_failure()
    manager = PermissionManager(workspace)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    manager.set_limit("deny_local_networks", True)
    host = Mock()
    host.run_host_command.side_effect = AssertionError("Unexpected host execution")
    service = CommandExecutionService(backend, host)
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.return_value = "approve"
    environment = ToolSettings(
        sandbox_environment=getattr(request, "param", {})
    ).sandbox_environment
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=service,
        settings=ToolRuntimeSettings(sandbox_environment=environment),
    )

    def run(source="", *, tool="run_command", asynchronous=False, **arguments):
        """Dispatch opaque source or poll its owned continuation through the public registry."""
        if tool == "run_command":
            arguments.update(command=source, cwd=str(workspace))
        encoded = json.dumps(arguments)
        output = (
            asyncio.run(registry.call_async(tool, encoded, interaction=interaction))
            if asynchronous
            else registry.call(tool, encoded, interaction=interaction)
        )
        return json.loads(output)

    try:
        yield workspace, manager, interaction, run
    finally:
        registry.close()
        manager.close()


def test_native_allowed_commands_read_and_use_scratch_without_approval(native_write_tool):
    """Shell children can read ordinary files and write private scratch without live writes."""
    workspace, _manager, interaction, run = native_write_tool
    result = run(
        'sh -c "cat ordinary"; '
        'printf scratch > "$TMPDIR/report"; cat "$TMPDIR/report"; '
        'printf output > "$LOOP_OUTPUT_DIR/status"; cat "$LOOP_OUTPUT_DIR/status"'
    )
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["exit_code"] == 0, result
    assert result["result"]["stdout"]["content"] == "ordinary-canaryscratchoutput"
    assert (workspace / "ordinary").read_text() == "ordinary-canary"
    interaction.prompt.assert_not_called()


def test_native_scratch_redirection_cannot_acquire_workspace_writes(native_write_tool):
    """Rebinding a scratch variable cannot extend native writes beyond private scratch."""
    workspace, _manager, interaction, run = native_write_tool
    result = run('TMPDIR="$WORKSPACE_ROOT"; printf changed > "$TMPDIR/neighbor"')
    assert not result["ok"], result
    assert result["problem"]["code"] == "process.nonzero_exit"
    assert (workspace / "neighbor").read_text() == "unchanged"
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize("exact_report", [False, True])
@pytest.mark.parametrize(
    "native_write_tool", [{"REPORT_FILE": {"output": "test report"}}], indirect=True
)
def test_native_children_keep_configured_report_and_sidecar_in_attempt_output(
    native_write_tool, exact_report
):
    """A launcher can place a report and its dynamically named sibling in private output."""
    workspace, manager, _interaction, run = native_write_tool
    report = workspace / "report"
    report.write_text("checkout-report")
    scope = {"write_files": ["$WORKSPACE_ROOT/report"]} if exact_report else {}
    result = run(
        "perl -e 'for my $path ($ENV{REPORT_FILE}, qq($ENV{REPORT_FILE}.worker)) { "
        "open my $report, q(>), $path or die $!; print $report q(output) }'",
        **scope,
    )

    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["write_scope"]["roots"] == []
    assert result["result"]["write_scope"]["files"] == scope.get("write_files", [])
    output = manager.temporary_directory / Path(result["result"]["scratch"]["output_path"]).name
    assert (output / "test report").read_text() == "output"
    assert (output / "test report.worker").read_text() == "output"
    assert report.read_text() == "checkout-report"
    assert not (workspace / "report.worker").exists()


@pytest.mark.parametrize(
    "source, fresh_review",
    [
        ("printf changed > neighbor", True),
        ("perl -e 'open my $f, q(>), q(neighbor) or die $!; print $f q(changed)'", False),
    ],
)
def test_native_undeclared_writes_are_denied_independent_of_shell_spelling(
    native_write_tool, source, fresh_review
):
    """Redirection and opaque interpreter writes cannot acquire an omitted live write root."""
    workspace, _manager, interaction, run = native_write_tool
    result = run(source)
    assert not result["ok"], result
    assert (workspace / "neighbor").read_text() == "unchanged", result
    assert result["problem"]["metadata"]["write_scope"]["roots"] == []
    assert "write_roots" in result["problem"]["metadata"]["write_scope"]["guidance"]
    assert interaction.prompt.call_count == int(fresh_review)


@pytest.mark.parametrize("decision", list(Decision))
def test_native_exact_write_root_obeys_policy_and_keeps_neighbors_denied(
    native_write_tool, decision
):
    """One declared directory follows allow/ask/deny without granting neighboring writes."""
    workspace, manager, interaction, run = native_write_tool
    for action in (Action.FILESYSTEM_CREATE, Action.FILESYSTEM_REPLACE, Action.FILESYSTEM_DELETE):
        manager.add_rule(
            PermissionRule(decision=decision, action=action, resource=str(workspace / "out"))
        )
    result = run(
        "printf approved > out/value; printf bad > neighbor; cat .env",
        write_roots=[str(workspace / "out")],
    )
    assert (workspace / "neighbor").read_text() == "unchanged", result
    assert (workspace / ".env").read_text() == "hard-canary"
    assert "hard-canary" not in json.dumps(result)
    if decision is Decision.DENY:
        assert not (workspace / "out/value").exists()
    else:
        assert (workspace / "out/value").read_text() == "approved", result
    assert interaction.prompt.call_count == (decision is not Decision.DENY)


def test_native_exact_write_file_remembers_only_the_approved_command(native_write_tool):
    """An exact command grant permits one report and other commands need separate approval."""
    workspace, manager, interaction, run = native_write_tool
    coverage = workspace / ".coverage"
    grants = {"write_files": ["$WORKSPACE_ROOT/.coverage"]}
    interaction.prompt.return_value = "session"
    prepare_coverage = "perl -e 'open my $f, q(>), q(.coverage) or die $!; print $f q(first)'"
    first = run(prepare_coverage, **grants)

    interaction.prompt.assert_called_once()
    assert interaction.prompt.call_args.kwargs["index"]["session"] == "S"
    assert first["ok"], json.dumps(first, indent=2)
    assert first["result"]["boundary"] == "sandbox"
    assert first["result"]["write_scope"]["roots"] == []
    assert first["result"]["write_scope"]["files"] == ["$WORKSPACE_ROOT/.coverage"]
    assert coverage.read_text() == "first"
    assert str(coverage) in interaction.info.call_args.args[0]
    assert not manager.sandbox_path_rules(ApprovalChoice.SESSION)
    assert [rule.source for rule in manager.sandboxed_command_rules(ApprovalChoice.SESSION)] == [
        prepare_coverage
    ]

    interaction.reset_mock()
    interaction.prompt.return_value = "deny"
    omitted = run(prepare_coverage.replace("q(first)", "q(unauthorized)"))
    assert not omitted["ok"], omitted
    assert coverage.read_text() == "first"
    interaction.prompt.return_value = "approve"
    second = run(
        "perl -e 'unlink q(.coverage) or die $!; "
        "open my $f, q(>), q(.coverage) or die $!; print $f q(replacement); "
        "close $f or die $!; "
        "for my $path (q(neighbor), q(.coverage.sibling)) { "
        "open my $blocked, q(>), $path and die q(unexpected write); }'",
        **grants,
    )

    assert second["ok"], json.dumps(second, indent=2)
    assert coverage.read_text() == "replacement"
    assert (workspace / "neighbor").read_text() == "unchanged"
    assert not (workspace / ".coverage.sibling").exists()
    interaction.prompt.assert_called_once()


def test_native_exact_write_file_cannot_authorize_a_protected_name(native_write_tool):
    """A request for one protected file fails without prompting or changing the protected data."""
    workspace, _manager, interaction, run = native_write_tool
    denied = run("printf forbidden > .env", write_files=["$WORKSPACE_ROOT/.env"])

    assert not denied["ok"], denied
    assert (workspace / ".env").read_text() == "hard-canary"
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize("decision", [Decision.DENY, Decision.ASK])
def test_native_nested_policy_refuses_encompassing_root_before_effects(native_write_tool, decision):
    """A withheld nested resource cannot be widened by approving its enclosing directory."""
    workspace, manager, interaction, run = native_write_tool
    manager.add_rule(
        PermissionRule(
            decision=decision,
            action=Action.FILESYSTEM_CREATE,
            resource=str(workspace / "out/withheld"),
        )
    )
    result = run("printf effect > out/marker", write_roots=[str(workspace / "out")])
    assert not result["ok"], result
    assert not (workspace / "out/marker").exists()
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize("workflow", ["search", "git", "compiler"])
def test_native_developer_workflows_need_no_live_write_approval(native_write_tool, workflow):
    """Search, Git inspection and scratch compilation work without workspace write grants."""
    workspace, _manager, interaction, run = native_write_tool
    if workflow == "search":
        if _RG is None:
            pytest.skip("ripgrep in a standard native runtime is required for the search workflow")
        source = f"{shlex.quote(_RG)} ordinary-canary ordinary"
        expected = "ordinary-canary"
    elif workflow == "git":
        if _GIT is None:
            pytest.skip("Git is required for the native status workflow")
        subprocess.run([_GIT, "init", "-q", str(workspace)], check=True, capture_output=True)
        source = f"{shlex.quote(_GIT)} --no-optional-locks status --short"
        expected = "ordinary"
    else:
        if _CC is None:
            pytest.skip("A system C compiler is required for the native build workflow")
        (workspace / "source.c").write_text("int main(void) { return 0; }\n")
        source = (
            f'{shlex.quote(_CC)} -c source.c -o "$TMPDIR/result.o" '
            '&& test -s "$TMPDIR/result.o" && printf compiled'
        )
        expected = "compiled"
    result = run(source)
    assert result["ok"], json.dumps(result, indent=2)
    assert expected in result["result"]["stdout"]["content"]
    assert result["result"]["write_scope"]["roots"] == []
    assert (workspace / "neighbor").read_text() == "unchanged"
    assert not (workspace / "result.o").exists()
    interaction.prompt.assert_not_called()


def test_native_runtime_baseline_deny_refuses_before_command_effects(native_write_tool):
    """A runtime file rule applies even when its native read root is implicit."""
    workspace, manager, interaction, run = native_write_tool
    manager.add_rule(
        PermissionRule(
            decision=Decision.DENY,
            action=Action.FILESYSTEM_READ,
            resource="/usr/bin/withheld-runtime-entry",
        )
    )
    result = run("printf effect > out/marker", write_roots=[str(workspace / "out")])
    assert not result["ok"], result
    assert not (workspace / "out/marker").exists()
    interaction.prompt.assert_not_called()


def test_native_private_scratch_deny_refuses_before_command_effects(native_write_tool):
    """A named scratch restriction cannot be bypassed through implicit temporary writes."""
    workspace, manager, interaction, run = native_write_tool
    manager.add_rule(
        PermissionRule(
            decision=Decision.DENY,
            action=Action.FILESYSTEM_CREATE,
            resource=str(manager.temporary_directory / "withheld"),
        )
    )
    result = run('printf effect > "$TMPDIR/withheld"')
    assert not result["ok"], result
    assert not (manager.temporary_directory / "withheld").exists()
    assert (workspace / "neighbor").read_text() == "unchanged"
    interaction.prompt.assert_not_called()


def test_native_default_process_ask_can_deny_without_effects(native_write_tool):
    """Declining the default process approval produces no native payload or host effects."""
    workspace, manager, interaction, run = native_write_tool
    manager.reset_default(Action.PROCESS_EXECUTE, scope=PolicyScope.WORKSPACE)
    interaction.prompt.return_value = "deny"
    result = run("printf changed > ordinary", write_roots=[str(workspace)])
    assert not result["ok"]
    assert (workspace / "ordinary").read_text() == "ordinary-canary"
    interaction.prompt.assert_called_once()


@pytest.mark.parametrize(
    "arguments", [{}, {"write_roots": [], "write_files": []}, {"write_roots": "scratch"}]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_command_review_prepares_checkout_writes(native_write_tool, arguments, asynchronous):
    """The single reviewed scope accommodates an opaque wrapper's dynamic outputs."""
    workspace, manager, interaction, run = native_write_tool
    manager.reset_default(Action.PROCESS_EXECUTE, scope=PolicyScope.WORKSPACE)
    interaction.supports_fresh_host_approval = True
    selected_writes = arguments.get("write_roots") != "scratch"

    def approve(*_args, choices, **_kwargs):
        """Approve the scope Loop disclosed for this invocation."""
        assert set(choices) <= {"deny", "approve", "session", "workspace", "user", "details"}
        assert (
            f'"workspace_write": {str(selected_writes).lower()}'
            in (interaction.info.call_args.args[0])
        )
        return "approve"

    interaction.prompt.side_effect = approve
    interaction.prompt_async = AsyncMock(side_effect=approve)
    (workspace / "wrapper.sh").write_text(
        'file="report-$$"; printf report > "$file" && printf companion > "$file.side" '
        "&& ! printf forbidden > .env\n"
    )
    result = run("sh ./wrapper.sh", asynchronous=asynchronous, **arguments)

    if asynchronous:
        interaction.prompt_async.assert_awaited_once()
    else:
        interaction.prompt.assert_called_once()
    assert not manager.configuration.workspace_command_writes
    assert (workspace / ".env").read_text() == "hard-canary"
    files = sorted(path.read_text() for path in workspace.glob("report-*"))
    if selected_writes:
        assert result["ok"], result
        assert result["result"]["write_scope"]["roots"] == ["$WORKSPACE_ROOT"]
        assert files == ["companion", "report"]
    else:
        assert result["problem"]["code"] == "process.nonzero_exit", result
        assert result["problem"]["metadata"]["write_scope"]["roots"] == []
        assert files == []


@pytest.mark.parametrize("background", [False, True])
def test_native_ordinary_workspace_write_preference_keeps_scope_and_approval(
    native_write_tool, native_directory, background
):
    """Native foreground and continued commands share normal approval and hard path boundaries."""
    workspace, manager, interaction, run = native_write_tool
    sibling = native_directory("/private/tmp" if sys.platform == "darwin" else "/tmp") / "sibling"
    sibling.write_text("outside")
    manager.reset_default(Action.PROCESS_EXECUTE, scope=PolicyScope.WORKSPACE)
    manager.set_workspace_command_writes(True)

    source = (
        "printf formatted > ordinary || exit; "
        "if cat .env || printf bad > .env || "
        f"printf bad > {shlex.quote(str(sibling))}; then exit 32; fi"
    )
    if background:
        source += (
            "; remaining=200; until test -f release; do "
            'remaining=$((remaining - 1)); test "$remaining" -gt 0 || exit 33; '
            "sleep 0.05; done"
        )
    interaction.prompt.return_value = "deny"
    refused = run(source)
    assert not refused["ok"], refused
    assert (workspace / "ordinary").read_text() == "ordinary-canary"
    interaction.prompt.assert_called_once()

    interaction.reset_mock()
    interaction.prompt.return_value = "approve"
    try:
        result = run(source, wait_seconds=0 if background else 10)
    finally:
        if background:
            (workspace / "release").touch()
    assert result["ok"], result
    interaction.prompt.assert_called_once()
    assert '"workspace_write": true' in interaction.info.call_args.args[0]
    outcome = result["result"]
    if background:
        assert outcome["status"] == "running", outcome
        outcome = run(tool="poll_command_job", handle=outcome["handle"], wait_seconds=10)["result"]
        assert outcome["status"] == "complete", outcome
    assert outcome["exit_code"] == 0, outcome
    assert (workspace / "ordinary").read_text() == "formatted"
    assert (workspace / ".env").read_text() == "hard-canary"
    assert sibling.read_text() == "outside"
    assert "hard-canary" not in json.dumps(outcome)


def test_native_workspace_write_preference_accepts_narrower_scope_and_revocation(native_write_tool):
    """An explicit narrow request overrides the default and stale consent cannot launch it."""
    workspace, manager, interaction, run = native_write_tool
    manager.set_workspace_command_writes(True)
    manager.update_sandbox_paths(
        ApprovalChoice.SESSION, {str(workspace / "out"): SandboxPathAccess.READ_ONLY}
    )
    denied = run("printf forbidden > ordinary")
    assert "read-only sandbox path" in denied["problem"]["detail"], denied
    interaction.prompt.assert_not_called()

    scratch_only = run("printf forbidden > ordinary", write_roots=[])
    assert not scratch_only["ok"], scratch_only
    narrow = run(
        "printf updated > ordinary; ! printf forbidden > neighbor", write_files=["ordinary"]
    )
    assert narrow["ok"], narrow
    assert (workspace / "ordinary").read_text() == "updated"
    assert (workspace / "neighbor").read_text() == "unchanged"

    manager.update_sandbox_paths(ApprovalChoice.SESSION, {str(workspace / "out"): None})

    def revoke_before_launch(*_args, **_kwargs):
        """Withdraw the default before returning the pending process decision."""
        manager.set_workspace_command_writes(False)
        return "approve"

    interaction.prompt.side_effect = revoke_before_launch
    revoked = run("printf stale > ordinary")
    assert not revoked["ok"], revoked
    assert (workspace / "ordinary").read_text() == "updated"
