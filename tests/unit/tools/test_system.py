"""Test the public native command tool boundary."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from loop import BUILTIN_TOOLS, PermissionManager, ToolContext, ToolRegistry
from loop.constants import MAX_OUTPUT_CHARS
from loop.execution import CommandExecutionService
from loop.execution.coordinator import HostCommandRequest, LocalHostCommandExecutor
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, UnavailableSandboxBackend
from loop.execution.sandbox.macos import MacOSSeatbeltBackend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.permissions import (
    CommandFinding,
    CommandInspection,
    CommandReviewStatus,
    Operation,
    ProcessBoundary,
    ProcessTarget,
)
from loop.tools.system import resolve_executable as resolve_executable_tool
from loop.tools.system import run_command as run_command_tool
from loop.utils import VirtualPath


class ApprovingInteraction:
    """Approve every explicit one-off native command prompt."""

    def info(self, message):
        """Accept informational output."""

    def prompt(self, message, **kwargs):
        """Approve the requested command."""
        return "deny" if "without the OS sandbox" in message else "approve"


class ApprovingHostInteraction(ApprovingInteraction):
    """Approve both separately presented sandbox and host prompts."""

    def prompt(self, message, **kwargs):
        """Approve the fresh one-off choice shown in each prompt."""
        return "approve"


class RememberingHostInteraction(ApprovingInteraction):
    """Record the first exact host choice and deny any later host prompt."""

    host_prompts: int

    def __init__(self):
        self.host_prompts = 0

    def prompt(self, message, **kwargs):
        """Choose a session rule once and expose unexpected reapproval requests."""
        if "without the OS sandbox" in message:
            self.host_prompts += 1
            return "session" if self.host_prompts == 1 else "deny"
        return "approve"


class CountingHostCommandExecutor:
    """Record approved host requests without starting a real shell."""

    _executor: MagicMock
    requests: list[HostCommandRequest]

    def __init__(self):
        self._executor = MagicMock()
        self._executor.run_host_command.return_value = CommandProcessResult(
            SandboxOutcome.COMPLETED, exit_code=0, stdout="startup-recovery"
        )
        self.requests = []

    def run_host_command(self, request: HostCommandRequest) -> CommandProcessResult:
        """Record one approved host request and return the configured process result."""
        self.requests.append(request)
        return self._executor.run_host_command(request)


class RecordingHostOfferInteraction(ApprovingInteraction):
    """Record the offer and host-launch count before selecting a retry decision."""

    _host_answer: str
    _host_executor: CountingHostCommandExecutor
    information: list[str]
    host_prompts: list[str]
    launches_before_host_prompt: list[int]

    def __init__(self, host_executor: CountingHostCommandExecutor, host_answer: str):
        self._host_answer = host_answer
        self._host_executor = host_executor
        self.information = []
        self.host_prompts = []
        self.launches_before_host_prompt = []

    def info(self, message):
        """Capture the warning presented with a host retry offer."""
        self.information.append(message)

    def prompt(self, message, **kwargs):
        """Choose the configured host decision and approve unrelated prompts."""
        if message and "without the OS sandbox" in message:
            self.host_prompts.append(message)
            self.launches_before_host_prompt.append(len(self._host_executor.requests))
            return self._host_answer
        return super().prompt(message, **kwargs)


@pytest.fixture
def registry():
    """Provide a registry with the native permission authority."""
    return native_registry()


def native_registry(*, permission_manager=None):
    """Compose the real execution facade for public tool tests."""
    return ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permission_manager or PermissionManager(),
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )


def call(registry, command, cwd=".", interaction=None, instructions_manager=None, **grants):
    """Dispatch one command and decode its result envelope."""
    output = registry.call(
        "run_command",
        json.dumps({"command": command, "cwd": cwd, **grants}),
        interaction=interaction or ApprovingInteraction(),
        instructions_manager=instructions_manager,
    )
    payload = json.loads(output)
    return payload.get("result"), payload.get("problem")


def lookup(registry, name, cwd=None, **context):
    """Dispatch one named host executable lookup through the public registry."""
    arguments = {"name": name}
    if cwd is not None:
        arguments["cwd"] = cwd
    return json.loads(registry.call("resolve_executable", json.dumps(arguments), **context))


def test_public_host_session_approval_reuses_across_fresh_scratch(tmp_path):
    """Public commands reuse one exact host approval and prompt for changed source."""
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="probe")
    host = MagicMock()
    host.run_host_command.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="host\n"
    )
    permissions = PermissionManager(tmp_path)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    interaction = RememberingHostInteraction()

    first, first_problem = call(registry, "printf host", cwd=str(tmp_path), interaction=interaction)
    second, second_problem = call(
        registry, "printf host", cwd=str(tmp_path), interaction=interaction
    )
    changed, changed_problem = call(
        registry, "printf changed", cwd=str(tmp_path), interaction=interaction
    )

    assert first_problem is None and second_problem is None
    assert first["boundary"] == second["boundary"] == "host"
    assert first["stdout"]["content"] == second["stdout"]["content"] == "host\n"
    assert changed is None and changed_problem["code"] == "sandbox.unavailable"
    assert interaction.host_prompts == 2
    assert host.run_host_command.call_count == 2
    assert len(permissions.host_command_rules()) == 1


def homebrew_fixture(tmp_path, monkeypatch, receipt=None):
    """Create a disposable Homebrew-shaped executable and optional dependency receipt."""
    prefix = tmp_path / "brew"
    package = prefix / "Cellar" / "sample" / "1.0"
    binaries = package / "bin"
    binaries.mkdir(parents=True)
    executable = binaries / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    opt = prefix / "opt"
    opt.mkdir()
    (opt / "sample").symlink_to(package, target_is_directory=True)
    if receipt is not None:
        (package / "INSTALL_RECEIPT.json").write_text(receipt, encoding="utf-8")
    monkeypatch.setenv("PATH", str(binaries))
    return prefix, package, opt


def test_resolve_executable_reports_plain_and_symlinked_tools(tmp_path, monkeypatch, registry):
    """A host PATH lookup finds files and symlinks without running or reading them."""
    prefix = tmp_path / "tool-prefix"
    binaries = prefix / "bin"
    support = prefix / "lib"
    binaries.mkdir(parents=True)
    support.mkdir()
    plain = binaries / "plain"
    plain.write_text("#!/bin/sh\nprintf executed > marker\n", encoding="utf-8")
    plain.chmod(0o755)
    target = support / "linked-target"
    target.write_text("#!/bin/sh\nprintf executed > marker\n", encoding="utf-8")
    target.chmod(0o755)
    (binaries / "linked").symlink_to(target)
    monkeypatch.setenv("PATH", str(binaries))

    plain_result = lookup(registry, "plain")["result"]
    linked_result = lookup(registry, "linked")["result"]

    assert plain_result["sandbox_access"] == "external_read_grant_required"
    assert plain_result["candidate_read_roots"] == [str(binaries)]
    assert linked_result["resolved_path"] == str(target)
    assert linked_result["candidate_read_roots"] == [str(support)]
    assert plain_result["grant_reference"] != linked_result["grant_reference"]
    assert "both references as a list" in plain_result["grant_note"]
    assert "executed" not in json.dumps((plain_result, linked_result))
    assert not (tmp_path / "marker").exists()


def test_executable_lookup_without_a_bound_service_fails_closed():
    """A registry missing native execution state cannot issue a reusable grant."""
    registry = ToolRegistry([resolve_executable_tool])
    assert lookup(registry, "true")["problem"]["code"] == "executable.invalid_lookup"


def test_resolve_executable_distinguishes_absent_and_invalid_names(tmp_path, monkeypatch, registry):
    """A missing named tool is distinct from an unsafe or oversized lookup."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert lookup(registry, "missing")["result"] == {
        "name": "missing",
        "status": "absent_from_host_path",
    }
    assert lookup(registry, "../secret")["problem"]["code"] == "executable.invalid_lookup"
    monkeypatch.setenv("PATH", ":".join([str(tmp_path)] * 129))
    assert lookup(registry, "missing")["problem"]["code"] == "executable.invalid_lookup"


def test_resolve_executable_skips_nonexecutables_and_preserves_path_order(
    tmp_path, monkeypatch, registry
):
    """Only the first executable regular file on PATH is reported."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "tool").write_text("not executable", encoding="utf-8")
    selected = second / "tool"
    selected.write_text("selected", encoding="utf-8")
    selected.chmod(0o755)
    monkeypatch.setenv("PATH", f"{first}:{second}")

    assert lookup(registry, "tool")["result"]["path"] == str(selected)


def test_resolve_executable_does_not_suggest_a_shared_temporary_parent(monkeypatch, registry):
    """A symlink across separate temporary roots recommends only their file directories."""
    with (
        tempfile.TemporaryDirectory(prefix="lookup-search-") as first_name,
        tempfile.TemporaryDirectory(prefix="lookup-target-") as second_name,
    ):
        first = Path(first_name).resolve()
        second = Path(second_name).resolve()
        target = second / "tool"
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        target.chmod(0o755)
        (first / "tool").symlink_to(target)
        monkeypatch.setenv("PATH", str(first))

        roots = lookup(registry, "tool")["result"]["candidate_read_roots"]

        assert roots == [str(second)]


@pytest.fixture
def reviewed_system_tool(tmp_path, monkeypatch):
    """Provide a reviewed executable without depending on installed host utilities."""
    binaries = tmp_path / "system-tools"
    binaries.mkdir()
    executable = binaries / "true"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries))
    monkeypatch.setattr(
        MacOSSeatbeltBackend, "system_read_roots", property(lambda self: (binaries,))
    )
    return executable


def test_resolve_executable_reports_system_tool_as_default_read(registry, reviewed_system_tool):
    """A reviewed system executable needs no external read grant."""

    result = lookup(registry, "true")["result"]

    assert result["status"] == "installed_on_host"
    assert result["sandbox_access"] == "default_read"
    assert "candidate_read_roots" not in result


def test_homebrew_lookup_binds_only_package_and_declared_support_roots(
    tmp_path, monkeypatch, registry
):
    """Installation metadata binds versioned dependencies and their exact opt aliases."""
    prefix, package, opt = homebrew_fixture(
        tmp_path,
        monkeypatch,
        json.dumps({"runtime_dependencies": [{"full_name": "support"}]}),
    )
    support = prefix / "Cellar" / "support" / "2.0"
    support.mkdir(parents=True)
    (opt / "support").symlink_to(support, target_is_directory=True)

    result = lookup(registry, "sampletool")["result"]

    assert result["candidate_read_roots"] == [str(package), str(support)]
    assert result["grant_reference"]


def test_homebrew_lookup_accepts_installed_package_without_opt_alias(
    tmp_path, monkeypatch, registry
):
    """A versioned package can use its canonical root without an optional opt link."""
    _, package, opt = homebrew_fixture(tmp_path, monkeypatch)
    (opt / "sample").unlink()

    result = lookup(registry, "sampletool")["result"]

    assert result["candidate_read_roots"] == [str(package)]


def test_lookup_under_incomplete_cellar_layout_keeps_executable_directory(
    tmp_path, monkeypatch, registry
):
    """A path named Cellar without a version is treated as a plain installation."""
    cellar = tmp_path / "Cellar"
    cellar.mkdir()
    executable = cellar / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(cellar))

    assert lookup(registry, "sampletool")["result"]["candidate_read_roots"] == [str(cellar)]


@pytest.mark.parametrize(
    "receipt",
    [
        "x" * 131_073,
        json.dumps({"runtime_dependencies": "wrong"}),
        json.dumps({"runtime_dependencies": [{}]}),
        json.dumps({"runtime_dependencies": [{"full_name": "../private"}]}),
        json.dumps({"runtime_dependencies": [{"full_name": "missing"}]}),
        "not json",
    ],
)
def test_homebrew_lookup_rejects_unsafe_or_unusable_dependency_metadata(
    tmp_path, monkeypatch, registry, receipt
):
    """An invalid package receipt never widens a sandboxed grant."""
    homebrew_fixture(tmp_path, monkeypatch, receipt)

    assert lookup(registry, "sampletool")["problem"]["code"] == "executable.invalid_lookup"


def test_homebrew_lookup_rejects_dependency_outside_its_package(tmp_path, monkeypatch, registry):
    """A receipt cannot turn an unrelated opt alias into a tool support grant."""
    _, _, opt = homebrew_fixture(
        tmp_path,
        monkeypatch,
        json.dumps({"runtime_dependencies": [{"full_name": "support"}]}),
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (opt / "support").symlink_to(unrelated, target_is_directory=True)

    assert lookup(registry, "sampletool")["problem"]["code"] == "executable.invalid_lookup"


def test_lookup_cannot_issue_a_grant_for_an_uninspectable_installation(
    tmp_path, monkeypatch, registry
):
    """A root identity failure leaves an installed tool without a usable grant."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    original = Path.lstat

    def inaccessible(path):
        """Simulate loss of metadata access at grant binding."""
        if path == package:
            raise OSError("unavailable")
        return original(path)

    monkeypatch.setattr(Path, "lstat", inaccessible)
    result = lookup(registry, "sampletool")["result"]
    assert result["sandbox_access"] == "external_grant_unavailable"
    assert "grant_reference" not in result


def test_homebrew_alias_change_invalidates_bound_reference(tmp_path, monkeypatch, registry):
    """Replacing a package opt alias cannot reuse a previous executable reference."""
    _, _, opt = homebrew_fixture(tmp_path, monkeypatch)
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    (opt / "sample").unlink()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (opt / "sample").symlink_to(replacement, target_is_directory=True)

    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": reference}),
            interaction=ApprovingInteraction(),
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"
    (opt / "sample").unlink()
    missing = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": reference}),
            interaction=ApprovingInteraction(),
        )
    )
    assert missing["problem"]["code"] == "sandbox.unavailable"


def test_lookup_reference_requires_unchanged_path_and_live_target(tmp_path, monkeypatch, registry):
    """A reference is rejected when its PATH snapshot, executable, or token is stale."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]

    def run(grant):
        """Send one public request using a previously returned reference."""
        return json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "sampletool", "executable_grant": grant}),
                interaction=ApprovingInteraction(),
            )
        )

    assert run("unknown")["problem"]["code"] == "sandbox.unavailable"
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert run(reference)["problem"]["code"] == "sandbox.unavailable"
    monkeypatch.setenv("PATH", str(package / "bin"))
    (package / "bin" / "sampletool").unlink()
    assert run(reference)["problem"]["code"] == "sandbox.unavailable"


def test_lookup_limits_live_references(tmp_path, monkeypatch, registry):
    """Old lookup references expire as the bounded service table fills."""
    homebrew_fixture(tmp_path, monkeypatch)
    first = lookup(registry, "sampletool")["result"]["grant_reference"]
    for _ in range(128):
        assert lookup(registry, "sampletool")["result"]["grant_reference"]
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": first}),
            interaction=ApprovingInteraction(),
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"


def test_lookup_never_grants_a_protected_external_tool(tmp_path, monkeypatch, registry):
    """An installed executable below a private directory is not requestable."""
    protected = tmp_path / ".ssh"
    protected.mkdir()
    executable = protected / "secret-tool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(protected))

    result = lookup(registry, "secret-tool")["result"]

    assert result["sandbox_access"] == "protected_external_path"
    assert "grant_reference" not in result


def test_external_lookup_reference_round_trips_through_public_virtual_boundary(
    tmp_path, monkeypatch
):
    """A model can request a bound tool grant without seeing a host path or starting a host retry."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "sampletool"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(installation))
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stderr=f"loaded {installation}/sampletool",
    )
    host = MagicMock()
    service = CommandExecutionService(backend, host)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(
            workspace, audit_path=tmp_path / "audit.db", workspace_id="workspace"
        ),
        execution_service=service,
    )

    looked_up = json.loads(
        registry.call(
            "resolve_executable",
            json.dumps({"name": "sampletool"}),
            instructions_manager=instructions,
        )
    )["result"]
    assert looked_up["sandbox_access"] == "external_read_grant_required"
    assert looked_up["path"] == "<external>"
    assert "candidate_read_roots" not in looked_up
    assert str(tmp_path) not in json.dumps(looked_up)
    payload = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": looked_up["grant_reference"]}),
            instructions_manager=instructions,
            interaction=interaction,
        )
    )
    assert payload["result"]["exit_code"] == 0
    assert str(installation) not in json.dumps(payload)
    assert "<external>/sampletool" in payload["result"]["stderr"]["content"]
    assert backend.run.call_args.args[0].read_roots == (installation,)
    assert backend.run.call_args.args[0].automatic_tool_reads == ()
    assert "Read the installed sampletool tool folder" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()
    host.run.assert_not_called()
    with closing(sqlite3.connect(tmp_path / "audit.db")) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records "
            "WHERE event_name = 'sandbox.permission_scope'"
        ).fetchone()
    assert json.loads(row[0])["automatic_tool_reads"] == []


def test_managed_toolchain_and_workspace_link_run_without_explicit_references(
    tmp_path, monkeypatch
):
    """The execution facade binds a named managed tool and its workspace PATH child."""
    workspace = tmp_path / "workspace"
    binaries = workspace / "bin"
    binaries.mkdir(parents=True)
    cellar = tmp_path / "brew" / "Cellar"
    roots = []
    for name in ("sampletool", "childtool"):
        package = cellar / name / "1.0"
        target = package / "bin" / name
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
        (package / "INSTALL_RECEIPT.json").write_text("{}", encoding="utf-8")
        roots.append(package)
        (binaries / name).symlink_to(target)
    monkeypatch.setenv("PATH", str(binaries))
    backend = MagicMock()
    backend.installed_tool_roots.side_effect = MacOSSeatbeltBackend.installed_tool_roots
    backend.managed_tool_root.side_effect = lambda root: (
        root.is_relative_to(cellar) and (root / "INSTALL_RECEIPT.json").is_file()
    )
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    host = MagicMock()
    interaction = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, host),
    )

    result, problem = call(registry, "sampletool", cwd=str(workspace), interaction=interaction)

    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert set(request.automatic_tool_reads) == set(roots)
    assert set(request.read_roots) == set(roots)
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_managed_toolchain_discovery_rejects_unsafe_or_unavailable_candidates(
    tmp_path, monkeypatch
):
    """Malformed shell source and unstable workspace PATH entries grant no authority."""
    workspace = tmp_path / "workspace"
    binaries = workspace / "bin"
    binaries.mkdir(parents=True)
    service = CommandExecutionService(MagicMock(), MagicMock())
    context = MagicMock()
    assert service._automatic_tool_references(context, "echo '", workspace, "/usr/bin") == ()
    assert service._automatic_tool_references(context, "env", workspace, "/usr/bin") == ()
    assert service._automatic_tool_references(context, "echo ok &&", workspace, "/usr/bin") == ()
    assert service._automatic_tool_references(context, "/bad/name", workspace, "/usr/bin") == ()

    original_iterdir = Path.iterdir

    def unreadable(self):
        """Simulate a PATH directory becoming unreadable during discovery."""
        if self == binaries:
            raise OSError("changed")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", unreadable)
    assert service._automatic_tool_references(context, "env", workspace, str(binaries)) == ()
    monkeypatch.setattr(Path, "iterdir", original_iterdir)
    for index in range(129):
        (binaries / f"entry-{index}").touch()
    assert service._automatic_tool_references(context, "env", workspace, str(binaries)) == ()

    monkeypatch.setattr(service, "resolve_executable", MagicMock(side_effect=ValueError("bad")))
    assert service._automatic_tool_references(context, "missing", workspace, "/usr/bin") == ()


def test_public_extra_read_contexts_and_protected_root_fail_closed(tmp_path):
    """Public requests distinguish narrow and broad reads and reject protected roots prelaunch."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    protected = external / ".ssh"
    protected.mkdir()
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )

    for root, expected in (
        (str(external), "Read an additional folder"),
        (str(tmp_path), "Read a broad area"),
    ):
        output = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "/usr/bin/true", "read_roots": [root]}),
                interaction=interaction,
                instructions_manager=instructions,
            )
        )
        assert "result" in output, output
        assert output["result"]["exit_code"] == 0
        assert expected in interaction.info.call_args.args[0]
        assert root not in interaction.info.call_args.args[0]
    before = backend.run.call_count
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "/usr/bin/true", "read_roots": [str(protected)]}),
            interaction=interaction,
            instructions_manager=instructions,
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"
    assert backend.run.call_count == before


def test_homebrew_grants_run_without_prompt_but_manual_read_still_prompts(tmp_path, monkeypatch):
    """Bound tool roots run in the sandbox while an additional data root still asks."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    (package / "INSTALL_RECEIPT.json").write_text("{}", encoding="utf-8")
    second = package / "bin" / "othertool"
    second.write_text("#!/bin/sh\n", encoding="utf-8")
    second.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    extra = tmp_path / "data"
    extra.mkdir()
    backend = MagicMock()
    backend.installed_tool_roots.side_effect = MacOSSeatbeltBackend.installed_tool_roots
    backend.managed_tool_root.side_effect = lambda root: (
        root.is_relative_to(package.parent.parent) and (root / "INSTALL_RECEIPT.json").is_file()
    )
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    first_ref = lookup(registry, "sampletool")["result"]["grant_reference"]
    second_ref = lookup(registry, "othertool")["result"]["grant_reference"]

    for grants, manual, expected, roots in (
        (first_ref, None, None, (package,)),
        ([first_ref, second_ref], None, None, (package,)),
        (
            first_ref,
            [str(extra)],
            "Use installed tools and read an additional folder",
            (package, extra),
        ),
    ):
        output = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": "sampletool",
                        "cwd": str(workspace),
                        "executable_grant": grants,
                        "read_roots": manual,
                    }
                ),
                interaction=interaction,
            )
        )
        assert "result" in output, output
        assert output["result"]["exit_code"] == 0
        if expected is None:
            interaction.info.assert_not_called()
            interaction.prompt.assert_not_called()
        else:
            assert expected in interaction.info.call_args.args[0]
            interaction.prompt.assert_called_once()
        assert backend.run.call_args.args[0].read_roots == roots
        assert backend.run.call_args.args[0].automatic_tool_reads == (package,)
        interaction.reset_mock()


def test_managed_tool_receipt_is_rechecked_before_native_launch(tmp_path, monkeypatch):
    """A stale package receipt cannot preserve prompt-free executable authority."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    receipt = package / "INSTALL_RECEIPT.json"
    receipt.write_text("{}", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = MagicMock()
    backend.installed_tool_roots.side_effect = MacOSSeatbeltBackend.installed_tool_roots
    backend.managed_tool_root.side_effect = lambda root: (
        root == package and (root / "INSTALL_RECEIPT.json").is_file()
    )
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    receipt.unlink()

    _, problem = call(registry, "sampletool", cwd=str(workspace), executable_grant=reference)

    assert problem["code"] == "sandbox.unavailable"
    backend.run.assert_not_called()


def test_unverified_cellar_name_needs_an_additional_read_decision(tmp_path, monkeypatch):
    """A Cellar-shaped directory without trusted provenance never receives automatic reads."""
    homebrew_fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = MagicMock()
    interaction.prompt.return_value = "deny"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    _, problem = call(
        registry,
        "sampletool",
        cwd=str(workspace),
        interaction=interaction,
        executable_grant=reference,
    )
    assert problem["code"] == "tool.denied"
    assert "Read the installed sampletool tool folder" in interaction.info.call_args.args[0]
    backend.run.assert_not_called()


def test_external_reference_requires_approval_even_for_unrelated_command(tmp_path, monkeypatch):
    """A bound arbitrary tool directory never becomes silent general read authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    executable = installation / "reviewtool"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    secret = installation / "unrelated.txt"
    secret.write_text("private marker", encoding="utf-8")
    monkeypatch.setenv("PATH", str(installation))
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    result = json.loads(
        registry.call(
            "resolve_executable",
            json.dumps({"name": "reviewtool"}),
            instructions_manager=instructions,
        )
    )["result"]
    assert str(installation) not in json.dumps(result)
    source = f"/bin/cat {secret}"
    request = {"command": source, "read_only": True, "executable_grant": result["grant_reference"]}
    denied = MagicMock()
    denied.prompt.return_value = "deny"
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps(request),
            instructions_manager=instructions,
            interaction=denied,
        )
    )
    assert output["problem"]["code"] == "tool.denied"
    denied.prompt.assert_called_once()
    backend.run.assert_not_called()

    approved = MagicMock()
    approved.prompt.return_value = "approve"
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps(request),
            instructions_manager=instructions,
            interaction=approved,
        )
    )
    assert output["result"]["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].automatic_tool_reads == ()
    assert backend.run.call_args.args[0].read_roots == (installation,)
    approved.prompt.assert_called_once()


def test_external_lookup_reference_rejects_changed_target(tmp_path, monkeypatch):
    """A changed executable cannot inherit approval from a prior host lookup."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(installation))
    backend = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    executable.rename(installation / "original-executable")
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": reference}),
            interaction=ApprovingInteraction(),
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"
    backend.run.assert_not_called()


def test_run_command_explains_external_tool_read_authority(registry):
    """The public schema tells the agent how to request access to installed tools."""
    definition = next(item for item in registry.definitions() if item.name == "run_command")

    description = definition.parameters["properties"]["read_roots"]["description"]
    assert "executable and support files" in description
    assert "PATH alone grants lookup metadata" in description


def test_run_command_accepts_opaque_shell_and_returns_bounded_result(monkeypatch, registry):
    """The public tool preserves shell source and returns a successful sandbox result."""
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="hello\n", stderr="warning\n"
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    result, problem = call(registry, "printf hello | cat; /bin/sh -c 'printf child'")

    assert problem is None
    assert result["exit_code"] == 0
    assert result["stdout"]["content"] == "hello\n"
    assert backend.run.call_args.args[0].source.endswith("'printf child'")
    environment = dict(backend.run.call_args.args[0].environment)
    assert environment.keys() == {"LANG", "PATH", "TMPDIR", "XDG_CACHE_HOME"}
    assert Path(environment["XDG_CACHE_HOME"]).parent == Path(environment["TMPDIR"])
    assert not Path(environment["TMPDIR"]).exists()


def test_read_only_command_uses_a_nonwritable_workspace_policy(monkeypatch, registry):
    """A default read-only command runs without approval or workspace writes."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"

    result, problem = call(registry, "cat file", interaction=interaction, read_only=True)

    assert problem is None
    assert result["boundary"] == "sandbox"
    assert result["scratch"]["path"] == "$TMPDIR"
    assert result["scratch"]["lifetime"] == "this command"
    assert backend.run.call_args.args[0].write_roots == ()
    environment = dict(backend.run.call_args.args[0].environment)
    assert Path(environment["COVERAGE_FILE"]).parent == Path(environment["TMPDIR"])
    assert environment["RUFF_NO_CACHE"] == "true"
    assert environment["UV_NO_SYNC"] == "1"
    interaction.prompt.assert_not_called()


def test_known_read_command_defaults_to_read_only(monkeypatch, registry):
    """An exact inspection command receives no automatic write restriction."""
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    result, problem = call(registry, "git diff --staged")
    assert problem is None
    assert result["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].write_roots


def test_read_only_cannot_request_git_write(tmp_path):
    """Contradictory read and Git write grants fail before execution."""
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), read_only=True, git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_public_git_write_asks_before_launch_without_explicit_flag(monkeypatch, registry, tmp_path):
    """A direct Git mutation requests its metadata grant before any sandbox child starts."""
    (tmp_path / ".git").mkdir()
    backend = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "deny"

    _, problem = call(registry, "git add file", cwd=str(tmp_path), interaction=interaction)

    assert problem["code"] == "tool.denied"
    assert "change Git metadata" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()
    backend.run.assert_not_called()


def test_public_command_uses_composed_inspection_for_grant_and_review(monkeypatch, tmp_path):
    """One injected matcher drives both Git authority planning and the approval prompt."""

    def match_special(command: list[str]) -> CommandFinding | None:
        """Classify the chosen test command as needing Git metadata authority."""
        if command != ["printf", "special"]:
            return None
        return CommandFinding(
            policy_id="special",
            status=CommandReviewStatus.FRESH,
            context="Review the special command",
            reason="run the special command",
            requests_git_write=True,
        )

    (tmp_path / ".git").mkdir()
    manager = PermissionManager(
        command_inspection=CommandInspection(()).with_matcher(match_special)
    )
    registry = native_registry(permission_manager=manager)
    backend = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "deny"

    _, problem = call(registry, "printf special", cwd=str(tmp_path), interaction=interaction)

    assert problem["code"] == "tool.denied"
    assert "change Git metadata" in interaction.info.call_args.args[0]
    assert "Review the special command" in interaction.info.call_args.args[0]
    assert "run the special command" in interaction.info.call_args.args[0]
    backend.run.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        "env rm -f victim",
        "env -i /bin/rm victim",
        "env -u HOME -- /bin/rm victim",
        "env --unset=HOME /bin/rm victim",
        "env -i /usr/bin/git clean -fd",
        "env -u HOME -- /usr/bin/git clean -fd",
        "env --unset=HOME /usr/bin/git clean -fd",
        "FOO=1 rm -f victim",
        ": > victim",
        "printf text > victim",
        "printf text &> victim",
        "printf text >& victim",
        "printf text 1>& victim",
        "printf text 2&> victim",
        "env git add victim",
        "FOO=1 git add victim; echo done",
    ],
)
def test_public_direct_destructive_commands_need_fresh_approval(
    monkeypatch, registry, tmp_path, source
):
    """Direct wrappers and truncation ask before launch and cannot reuse an earlier approval."""
    victim = tmp_path / "victim"
    victim.write_text("safe", encoding="utf-8")
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.side_effect = ["approve", "deny"]
    first, problem = call(registry, source, cwd=str(tmp_path), interaction=interaction)
    assert problem is None
    assert first["boundary"] == "sandbox"
    assert backend.run.call_count == 1
    _, problem = call(registry, source, cwd=str(tmp_path), interaction=interaction)
    assert problem["code"] == "tool.denied"
    assert backend.run.call_count == 1
    assert victim.read_text(encoding="utf-8") == "safe"
    assert interaction.prompt.call_count == 2
    assert all(
        "similar_session" not in item.kwargs["choices"]
        for item in interaction.prompt.call_args_list
    )


def test_git_grant_resolves_linked_worktree_metadata(monkeypatch, registry, tmp_path):
    """A linked worktree grant binds its real metadata directory for read and write."""
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "linked"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    common = tmp_path / "main" / ".git"
    (common / "HEAD").write_text("ref: refs/heads/main\n")
    (gitdir / "commondir").write_text("../..\n")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {gitdir}\n")
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    result, problem = call(registry, "git add file", cwd=str(linked), git_write=True)
    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.write_roots == (linked, gitdir, common)
    assert request.read_roots == (gitdir, common)


def test_git_grant_binds_absent_metadata_creation(monkeypatch, registry, tmp_path):
    """A fresh workspace can approve Git initialization without precreating `.git`."""
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    result, problem = call(registry, "git init", cwd=str(tmp_path), git_write=True)
    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.git_create is True
    assert request.write_roots == (tmp_path,)
    assert not (tmp_path / ".git").exists()


def test_git_creation_rejects_symlinked_metadata_directory(tmp_path):
    """A linked directory cannot replace the absent Git creation target."""
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / ".git").symlink_to(external, target_is_directory=True)
    registry = native_registry()
    _, problem = call(registry, "git init", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_linked_git_status_requests_metadata_read_grant(monkeypatch, registry, tmp_path):
    """A linked worktree status cannot silently read metadata outside the workspace."""
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "linked"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {gitdir}\n")
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    _, problem = call(registry, "git status", cwd=str(linked), interaction=interaction)
    assert problem is None
    request = backend.run.call_args.args[0]
    assert request.write_roots == (linked,)
    assert request.read_roots == (gitdir,)
    assert "read files outside this workspace" in interaction.info.call_args.args[0]
    assert str(gitdir) not in interaction.info.call_args.args[0]


@pytest.mark.parametrize("pointer", ["invalid", "gitdir: missing"])
def test_invalid_linked_worktree_grant_fails_before_approval(tmp_path, pointer):
    """An invalid linked pointer cannot silently grant an unrelated directory."""
    (tmp_path / ".git").write_text(pointer)
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_linked_worktree_grant_rejects_missing_head_and_symlink_pointer(tmp_path):
    """Git pointer names must be regular files and lead to usable metadata."""
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    pointer = tmp_path / ".git"
    pointer.write_text(f"gitdir: {gitdir}\n")
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    pointer.rename(tmp_path / "git-pointer")
    pointer.symlink_to(tmp_path / "git-pointer")
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


@pytest.mark.parametrize("common_pointer", ["missing", "no-head", "symlink"])
def test_linked_worktree_grant_rejects_invalid_common_metadata(tmp_path, common_pointer):
    """The shared Git metadata directory must be a real reviewed directory."""
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git").write_text(f"gitdir: {gitdir}\n")
    if common_pointer == "missing":
        (gitdir / "commondir").write_text("../missing\n")
    elif common_pointer == "no-head":
        (tmp_path / "common-empty").mkdir()
        (gitdir / "commondir").write_text("../common-empty\n")
    else:
        target = tmp_path / "common-target"
        target.mkdir()
        (target / "HEAD").write_text("ref: refs/heads/main\n")
        (gitdir / "commondir").symlink_to(target / "HEAD")
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_git_write_grant_uses_regular_metadata_directory(monkeypatch, registry, tmp_path):
    """A conventional repository keeps its Git grant under the workspace."""
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem is None
    assert backend.run.call_args.args[0].write_roots == (tmp_path, gitdir)


def test_run_command_preserves_nonzero_exit(monkeypatch, registry):
    """An ordinary shell nonzero exit does not offer host execution."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    SandboxOutcome.COMPLETED,
                    exit_code=17,
                )
            )
        ),
    )

    _, problem = call(registry, "exit 17")

    assert problem["code"] == "process.nonzero_exit"
    assert problem["metadata"]["exit_code"] == 17
    assert "host_offer" not in problem["metadata"]


def test_run_command_binds_empty_and_relative_path_entries_from_command_cwd(
    monkeypatch, registry, tmp_path
):
    """Public command binding resolves empty and relative PATH entries from its cwd."""
    workspace = tmp_path / "workspace"
    (workspace / "bin").mkdir(parents=True)
    external = tmp_path / "external-bin"
    external.mkdir()
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setenv("PATH", ":bin:../external-bin")

    result, problem = call(registry, ":", cwd=str(workspace))

    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.shell_environment()["PATH"] == ":bin:../external-bin"
    assert request.path_roots == (
        (workspace, workspace),
        (workspace / "bin", workspace / "bin"),
        (external, external),
    )


def test_relative_one_character_tool_requires_explicit_external_read(tmp_path, monkeypatch):
    """A one-character tool on relative PATH runs only after its bound read grant."""
    workspace = tmp_path / "workspace"
    cwd = workspace / "subdir"
    cwd.mkdir(parents=True)
    installation = tmp_path / "external-bin"
    installation.mkdir()
    executable = installation / "x"
    executable.write_text("#!/bin/sh\nprintf external-ok\n", encoding="utf-8")
    executable.chmod(0o755)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("PATH", "../../external-bin:/usr/bin:/bin")
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=1, stderr="blocked"
    )
    host = MagicMock()
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, host),
    )

    found = lookup(registry, "x", cwd="subdir", instructions_manager=instructions)["result"]
    assert found["status"] == "installed_on_host"
    assert found["sandbox_access"] == "external_read_grant_required"
    assert "candidate_read_roots" not in found
    assert str(installation) not in json.dumps(found)

    _, problem = call(
        registry,
        "x",
        cwd="subdir",
        read_only=True,
        interaction=interaction,
        instructions_manager=instructions,
    )
    assert problem["code"] == "process.nonzero_exit"
    assert backend.run.call_args.args[0].read_roots == ()
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()

    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="external-ok"
    )
    result, problem = call(
        registry,
        "x",
        cwd="subdir",
        read_only=True,
        executable_grant=found["grant_reference"],
        interaction=interaction,
        instructions_manager=instructions,
    )
    assert problem is None
    assert result["stdout"]["content"] == "external-ok"
    assert backend.run.call_args.args[0].read_roots == (installation,)
    assert "Read the installed x tool folder" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()
    host.run_host_command.assert_not_called()


def test_resolve_executable_binds_relative_lookup_cwd_and_rejects_missing_cwd(
    tmp_path, registry, reviewed_system_tool
):
    """Executable lookup resolves workspace cwd and returns typed failures for missing cwd."""
    workspace = tmp_path / "workspace"
    subdirectory = workspace / "subdirectory"
    subdirectory.mkdir(parents=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))

    service_context = ToolContext(
        interaction=None,
        tool_name="resolve_executable",
        instructions_manager=instructions,
    )
    result = registry._execution_service.resolve_executable(service_context, "true", "subdirectory")
    missing_without_context = lookup(registry, "true", cwd=str(tmp_path / "missing"))
    missing_with_context = lookup(
        registry, "true", cwd="missing", instructions_manager=instructions
    )

    assert result["status"] == "installed_on_host"
    assert result["sandbox_access"] == "default_read"
    assert missing_without_context["problem"]["code"] == "executable.invalid_lookup"
    assert missing_with_context["problem"]["code"] == "executable.invalid_lookup"


def test_request_binding_error_redacts_external_paths_without_instructions(
    monkeypatch, registry, tmp_path
):
    """A late request-binding failure keeps external grant paths out of model output."""
    external = tmp_path / "external-read"
    external.mkdir()
    monkeypatch.setattr(
        "loop.execution.facade.SandboxRequest.create",
        MagicMock(side_effect=ValueError(f"request failed at {external}")),
    )

    _, problem = call(registry, ":", read_roots=[str(external)])

    assert problem["code"] == "sandbox.unavailable"
    assert str(external) not in problem["detail"]
    assert "<external>" in problem["detail"]


def test_run_command_preserves_system_path_in_redacted_output(tmp_path):
    """Approved system tool text stays readable while external roots remain redacted."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, scratch))
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED, exit_code=0, stdout="/usr/bin/true\n"
            )
        )
    )
    backend.system_read_roots = (Path("/usr/bin"),)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    payload = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "printf /usr/bin/true", "read_roots": ["/usr/bin"]}),
            instructions_manager=instructions,
            interaction=ApprovingInteraction(),
        )
    )

    assert payload["result"]["stdout"]["content"] == "/usr/bin/true\n"


def test_observed_denial_preserves_exit_and_offers_fresh_retry(monkeypatch, registry):
    """A verified denial remains diagnostic while the shell exit stays authoritative."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=7,
                stderr="Operation not permitted: /outside/secret",
                observed_denial="file-read-data /outside/secret",
            )
        )
    )
    host = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    _, problem = call(registry, "cat outside || :; exit 7")

    assert problem["code"] == "process.nonzero_exit"
    assert problem["metadata"]["exit_code"] == 7
    assert problem["metadata"]["sandbox_diagnostic"] == "file-read-data /outside/secret"
    assert "may be unrelated" in problem["metadata"]["host_offer"]
    host.run_host_command.assert_not_called()


def test_exit_127_with_incidental_denial_stays_in_sandbox(monkeypatch, registry):
    """A missing executable suggests grants without offering an unrelated host retry."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=127,
                stderr="/bin/sh: tool: command not found\n",
                observed_denial="file-read-data /Library/Preferences/unrelated.plist",
            )
        )
    )
    host = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    _, problem = call(registry, "tool --version", read_only=True)

    assert problem["code"] == "process.nonzero_exit"
    metadata = problem["metadata"]
    assert metadata["exit_code"] == 127
    assert "resolve_executable" in metadata["executable_hint"]
    assert "host_offer" not in metadata
    assert "pass their references as a list" in metadata["executable_hint"]
    assert "caused" not in metadata["executable_hint"]
    host.run_host_command.assert_not_called()


def test_approved_observed_denial_retry_retains_sandbox_attempt(monkeypatch, registry):
    """A fresh approved host run reports the original sandbox exit and denial separately."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=1,
                stderr="Operation not permitted: /outside/secret",
                observed_denial="file-read-data /outside/secret",
            )
        )
    )
    host = MagicMock(
        run_host_command=MagicMock(
            return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
        )
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    result, problem = call(registry, "cat outside", interaction=ApprovingHostInteraction())

    assert problem is None
    assert result["boundary"] == "host"
    assert result["sandbox_attempt"] == {
        "exit_code": 1,
        "observed_denial": "file-read-data /outside/secret",
    }
    host.run_host_command.assert_called_once()


def test_run_command_reports_typed_sandbox_failure(monkeypatch, registry):
    """A backend launch failure is returned without host execution."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    SandboxOutcome.UNAVAILABLE, detail="profile rejected"
                )
            )
        ),
    )

    _, problem = call(registry, "printf MUST_NOT_RUN")

    assert problem["code"] == "sandbox.unavailable"
    assert problem["detail"] == "profile rejected"
    assert 'Command: "printf MUST_NOT_RUN"' in problem["metadata"]["host_offer"]


@pytest.mark.parametrize(("host_answer", "approved"), [("deny", False), ("approve", True)])
def test_public_startup_capability_failure_requires_separate_host_approval(
    tmp_path, host_answer, approved
):
    """An unavailable native backend offers recovery but never starts host before approval."""
    backend = UnavailableSandboxBackend("Operational sandbox probe failed.")
    backend.run = MagicMock(wraps=backend.run)
    host = CountingHostCommandExecutor()
    interaction = RecordingHostOfferInteraction(host, host_answer)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, host),
    )

    result, problem = call(
        registry,
        "printf startup-recovery",
        cwd=str(tmp_path),
        interaction=interaction,
    )

    assert backend.run.call_count == 0
    assert interaction.host_prompts == ["Run this exact command without the OS sandbox?"]
    assert interaction.launches_before_host_prompt == [0]
    assert len(interaction.information) == 1
    assert "The sandbox was unavailable." in interaction.information[0]
    assert str(tmp_path) not in interaction.information[0]
    if approved:
        assert problem is None
        assert result["boundary"] == "host"
        assert result["stdout"]["content"] == "startup-recovery"
        assert len(host.requests) == 1
        assert host.requests[0].source == "printf startup-recovery"
    else:
        assert result is None
        assert problem["code"] == "sandbox.unavailable"
        assert "host_offer" in problem["metadata"]
        assert host.requests == []


def test_run_command_warns_that_host_retry_may_repeat_effects(monkeypatch, registry):
    """The public host offer names repeat risk after a partially effective attempt."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    SandboxOutcome.DENIED, detail="file read denied", possible_effects=True
                )
            )
        ),
    )

    _, problem = call(registry, "printf changed > file; cat secret")

    assert problem["code"] == "sandbox.denied"
    assert "may repeat" in problem["metadata"]["host_offer"]


def test_run_command_host_retry_requires_a_second_prompt(monkeypatch, registry):
    """A distinct approved host retry is marked as unrestricted in the public result."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no profile")
        )
    )
    host = MagicMock(
        run_host_command=MagicMock(
            return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, stdout="host")
        )
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)
    interaction = ApprovingHostInteraction()

    result, problem = call(registry, "printf host", interaction=interaction)

    assert problem is None
    assert result["boundary"] == "host"
    assert result["stdout"]["content"] == "host"
    host.run_host_command.assert_called_once()


def test_run_command_keeps_host_timeout_distinct(monkeypatch, registry):
    """A separately approved host timeout is labeled as a host outcome."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no profile")
            )
        ),
    )
    monkeypatch.setattr(
        registry._execution_service,
        "_host_executor",
        MagicMock(
            run_host_command=MagicMock(return_value=CommandProcessResult(SandboxOutcome.TIMED_OUT))
        ),
    )

    _, problem = call(registry, "sleep 10", interaction=ApprovingHostInteraction())

    assert problem["code"] == "host_command.timed_out"
    assert problem["metadata"]["boundary"] == "host"


def test_run_command_timeout_does_not_offer_host_retry(monkeypatch, registry):
    """A timed-out sandbox attempt is not reclassified as a host offer."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.TIMED_OUT))),
    )

    _, problem = call(registry, "sleep 10")

    assert problem["code"] == "sandbox.timed_out"
    assert "host_offer" not in problem["metadata"]


def test_run_command_requires_bound_target_and_permission_manager(tmp_path):
    """Direct calls fail closed without registry-owned native authority."""
    context = ToolContext(ApprovingInteraction(), "run_command")
    assert run_command_tool(context, "printf no").code == "sandbox.unavailable"
    context = ToolContext(
        ApprovingInteraction(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action="process.execute",
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "printf no"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
    )
    assert (
        run_command_tool(context, "printf no").detail
        == "Native command execution service is unavailable."
    )


def test_run_command_fails_when_request_paths_are_invalid(tmp_path):
    """A missing working directory fails before approval or launch."""
    manager = MagicMock()
    context = ToolContext(
        ApprovingInteraction(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action="process.execute",
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", ":"),
                    cwd=str(Path(tmp_path) / "missing"),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=manager,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    result = run_command_tool(context, ":")
    assert result.code == "sandbox.unavailable"
    manager.authorize_sandboxed_command.assert_not_called()


def test_run_command_uses_available_temp_for_virtual_path_alias(monkeypatch, tmp_path):
    """Virtual path aliases use an accessible private temp root instead of /private/tmp."""
    original = tempfile.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        """Keep the alias on the disposable filesystem and reject a fixed temp path."""
        assert "dir" not in kwargs
        return (
            original(*args, dir=tmp_path, **kwargs)
            if kwargs.get("prefix") == "loop-vpath-"
            else original(*args, **kwargs)
        )

    monkeypatch.setattr("loop.execution.facade.tempfile.TemporaryDirectory", temporary_directory)
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    execution_service = CommandExecutionService(backend, LocalHostCommandExecutor())
    instructions = MagicMock(virtual_paths=VirtualPath(workspace=tmp_path))
    context = ToolContext(
        ApprovingInteraction(),
        "run_command",
        instructions_manager=instructions,
        operations=(
            Operation(
                tool_id="run_command",
                action="process.execute",
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "cat /workspace/file"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(),
        execution_service=execution_service,
    )

    result = run_command_tool(context, "cat /workspace/file")

    assert result["boundary"] == "sandbox"
    alias = backend.run.call_args.args[0].aliases[0][1]
    assert alias.parent.is_relative_to(tmp_path)


def test_run_command_honors_fresh_denial(monkeypatch, registry):
    """A denied native prompt starts no sandbox backend."""
    backend = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    registry.permission_manager.authorize_sandboxed_command = MagicMock(return_value=False)

    _, problem = call(registry, "printf no")

    assert problem["code"] == "tool.denied"
    backend.run.assert_not_called()


def test_run_command_spills_an_oversized_backend_preview(monkeypatch, registry):
    """Oversized backend text retains a continuation handle in the public result."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    SandboxOutcome.COMPLETED,
                    exit_code=0,
                    stdout="x" * MAX_OUTPUT_CHARS,
                    stdout_discarded=1,
                )
            )
        ),
    )
    result, _ = call(registry, "printf lots")
    assert result["stdout"]["truncated"] is True
    assert result["stdout"]["handle"]
