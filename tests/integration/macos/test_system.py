"""Verify public command dispatch, destructive review and native execution boundaries."""

import base64
import errno
import json
import os
import secrets
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from loop import (
    Action,
    ApprovalChoice,
    Decision,
    Loop,
    NetworkTarget,
    OpenAIBackend,
    Operation,
    PermissionManager,
    PolicyScope,
    ProcessBoundary,
    ProcessTarget,
    SessionManager,
    SQLiteSessionStore,
    ToolRegistry,
    ToolResult,
)
from loop.constants import MAX_OUTPUT_CHARS
from loop.execution import CommandExecutionService
from loop.execution.sandbox import SandboxOutcome, SandboxRequest
from loop.execution.sandbox.constants import TMPDIR_ENVIRONMENT_VARIABLE
from loop.execution.sandbox.macos import backend as macos_backend
from loop.permissions.managed import ManagedPolicyStore
from loop.permissions.models import SessionTarget
from loop.tooling import ToolRuntimeSettings
from loop.tools.files import read_text_file, write_text_file
from loop.tools.system import control_command_terminal, resolve_executable, run_command
from loop.tools.web import read_cached_content
from loop.utils import VirtualPath
from loop.utils import path as path_module
from tests.scheduling import receive_fifo_byte

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.macos]
_QUALIFICATION_UV = shutil.which("uv")
_SYSTEM_MAKE = shutil.which("make", path=os.defpath)
_SYSTEM_CAT = shutil.which("cat", path=os.defpath)
_SYSTEM_CC = shutil.which("cc", path=os.defpath)
_SYSTEM_PRINTF = shutil.which("printf", path=os.defpath)
_SYSTEM_SH = shutil.which("sh", path=os.defpath)
_SYSTEM_XCRUN = shutil.which("xcrun", path=os.defpath)
_SYSTEM_GETCONF = shutil.which("getconf", path=os.defpath)


def test_native_tool_schema_omits_unavailable_execution_capabilities():
    """An unqualified raw backend does not advertise optional native controls."""
    service = CommandExecutionService(macos_backend.MacOSSeatbeltBackend(), Mock())
    registry = ToolRegistry([run_command, control_command_terminal], execution_service=service)
    try:
        definitions = {item.name: item for item in registry.definitions()}
        properties = definitions["run_command"].parameters["properties"]
        assert "interactive" not in properties
        assert "egress_origins" not in properties
        assert "control_command_terminal" not in definitions
    finally:
        registry.close()


def test_live_system_probe_preserves_command_diagnostics_and_isolation(native_workspace):
    """Real startup diagnostics can adapt to the host without granting unrelated files."""
    backend = macos_backend.MacOSSeatbeltBackend()

    def execute(source):
        """Run a fresh command with only the disposable workspace authorized."""
        return backend.run(
            SandboxRequest.create(
                source=source,
                cwd=native_workspace,
                workspace=native_workspace,
                read_roots=(),
                write_roots=(native_workspace,),
                network=False,
                environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
                policy_version=backend.policy_version,
                deadline=time.monotonic() + 30,
                workspace_id="live-system-probe",
            )
        )

    try:
        ordinary = execute("exit 1")
        assert ordinary.outcome is SandboxOutcome.COMPLETED and ordinary.exit_code == 1
        observed_directory = backend._diagnostics.system_directory(time.monotonic() + 10)
        if observed_directory is not None:
            assert str(observed_directory) not in ordinary.observed_denial
        outside = native_workspace.parent / "outside/secret"
        denied = execute(f"cat {shlex.quote(str(outside))}")
        assert denied.outcome is SandboxOutcome.COMPLETED and denied.exit_code != 0
        assert "fake-outside-canary" not in denied.stdout
    finally:
        backend.close()


def test_direct_unbound_home_scratch_keeps_private_names_denied(native_workspace, monkeypatch):
    """A caller-supplied scratch under a home-based temporary root cannot create private data."""
    home = native_workspace.parent / "synthetic-home"
    home.mkdir(mode=0o700)
    temporary = home / "temporary"
    temporary.mkdir(mode=0o700)
    scratch = temporary / "loop-seatbelt-provided"
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(macos_backend, "_trusted_home", lambda: home)
    monkeypatch.setattr(macos_backend.tempfile, "tempdir", str(temporary))
    backend = macos_backend.MacOSSeatbeltBackend()
    try:
        result = backend.run(
            SandboxRequest.create(
                source='printf ordinary > "$TMPDIR/ordinary"; mkdir "$TMPDIR/.ssh"',
                cwd=native_workspace,
                workspace=native_workspace,
                read_roots=(),
                write_roots=(native_workspace,),
                network=False,
                environment={"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(scratch)},
                policy_version=backend.policy_version,
                deadline=time.monotonic() + 30,
                workspace_id="unbound-home-scratch",
            )
        )
    finally:
        backend.close()

    assert result.outcome is SandboxOutcome.COMPLETED, result
    assert result.exit_code != 0
    assert (scratch / "ordinary").read_text() == "ordinary"
    assert not (scratch / ".ssh").exists()


@pytest.mark.parametrize("protected_target", [False, True])
def test_live_silent_denial_is_reported_without_attributing_the_exit(
    native_workspace, protected_target
):
    """Real silent denials remain uncertain without disclosing protected target names."""
    backend = macos_backend.MacOSSeatbeltBackend()
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host execution")
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    try:
        if protected_target:
            protected = native_workspace.parent / "outside/.ssh"
            protected.mkdir()
            (protected / "fixture-private-target").write_text("synthetic-canary")
            (native_workspace / "opaque-link").symlink_to(protected / "fixture-private-target")
            command = "test -e opaque-link >/dev/null 2>&1; exit 7"
        else:
            command = "cat external-link >/dev/null 2>&1; exit 7"
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": command, "cwd": str(native_workspace)}),
                interaction=interaction,
            )
        )
        assert result["problem"]["code"] == "process.nonzero_exit"
        metadata = result["problem"]["metadata"]
        assert metadata["exit_code"] == 7
        assert metadata["stderr"]["content"] == ""
        diagnostics = metadata["sandbox_diagnostics"]
        if diagnostics["collection"] != "available":
            pytest.skip("The native diagnostic stream did not supply a denial record")
        if protected_target:
            assert "fixture-private-target" not in json.dumps(result)
            assert any(
                "<protected>" in item["observation"] for item in diagnostics["denial_candidates"]
            )
        else:
            assert any(
                "outside/secret" in item["observation"] for item in diagnostics["denial_candidates"]
            )
        assert all(item["attribution"] == "unknown" for item in diagnostics["denial_candidates"])
        assert diagnostics["cause"] == "unknown"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize("diagnostics", ["live", "unknown", "unavailable"])
def test_successful_command_keeps_buffered_denials_optional(
    public_command, native_workspace, monkeypatch, diagnostics
):
    """A successful silent command keeps buffered evidence without snapshot or replay."""
    denial = "Sandbox: cat(1) deny(1) file-read-data /arbitrary/system/cache"
    monitor = Mock()
    monitor.register.return_value = True
    monitor.denial.return_value = denial if diagnostics == "live" else None
    monkeypatch.setattr(
        macos_backend.SeatbeltDiagnosticService,
        "monitor",
        lambda *_args: None if diagnostics == "unavailable" else monitor,
    )
    snapshot = Mock(side_effect=AssertionError("successful command must not wait for logs"))
    monkeypatch.setattr(macos_backend, "_read_denial_from_log", snapshot)
    monkeypatch.setattr(macos_backend, "_denial_from_log", snapshot)

    result = public_command.call("printf x >> count; cat external-link >/dev/null 2>&1; true")

    assert result["ok"]
    output = result["result"]
    assert output["exit_code"] == 0
    assert output["stdout"]["content"] == output["stderr"]["content"] == ""
    evidence = output["sandbox_diagnostics"]
    assert evidence["collection"] == ("available" if diagnostics == "live" else diagnostics)
    assert evidence["observations_may_be_incomplete"]
    if diagnostics == "live":
        assert evidence["denial_candidates"][0]["attribution"] == "unknown"
        assert "do not establish failure" in evidence["candidate_note"]
    else:
        assert not evidence["denial_candidates"]
    assert (native_workspace / "count").read_text() == "x"
    snapshot.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("diagnostics", ["live", "delayed", "unknown", "unavailable"])
def test_public_command_reports_unverified_diagnostic_candidates(
    public_command, native_workspace, monkeypatch, diagnostics
):
    """Unverified records hide host paths and never replay an effectful command."""
    denial = "Sandbox: sh(1) deny(1) file-read-data /arbitrary/system/cache"
    monitor = Mock()
    monitor.register.return_value = True
    monitor.denial.return_value = denial if diagnostics == "live" else None
    monkeypatch.setattr(
        macos_backend.SeatbeltDiagnosticService,
        "monitor",
        lambda *_args: None if diagnostics == "unavailable" else monitor,
    )
    monkeypatch.setattr(
        macos_backend,
        "_denial_from_log",
        lambda *_args: denial if diagnostics == "delayed" else None,
    )
    monkeypatch.setattr(macos_backend, "_read_denial_from_log", lambda *_args: None)

    result = public_command.call("printf x >> count; printf 'invalid option\\n' >&2; exit 7")

    assert result["problem"]["code"] == "process.nonzero_exit"
    metadata = result["problem"]["metadata"]
    assert metadata["stderr"]["content"] == "invalid option\n"
    assert metadata["sandbox_diagnostics"]["observed_denial"] is None
    assert metadata["sandbox_diagnostics"]["source"] == (
        "macos-kernel-log" if diagnostics in {"live", "delayed"} else "unknown"
    )
    assert metadata["sandbox_diagnostics"]["collection"] == (
        "unavailable"
        if diagnostics == "unavailable"
        else "unknown"
        if diagnostics == "unknown"
        else "available"
    )
    assert "sandbox_diagnostic" not in metadata
    candidates = metadata["sandbox_diagnostics"]["denial_candidates"]
    if diagnostics in {"live", "delayed"}:
        assert len(candidates) == 1
        assert candidates[0]["observation"] == ("Sandbox: sh(1) deny(1) file-read-data <external>")
        assert candidates[0]["operation"] == "file-read-data"
        assert candidates[0]["target"] == "<external>"
        assert candidates[0]["attribution"] == "unknown"
        assert candidates[0]["classification"] == "unclassified"
    else:
        assert not candidates
    assert "/arbitrary/system/cache" not in json.dumps(metadata)
    assert (native_workspace / "count").read_text() == "x"
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.e2e
def test_public_denial_never_launches_unapproved_host_retry(backend, native_workspace):
    """Public permission, native denial and recovery never launch an unrestricted retry."""
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host execution")
    interaction = Mock()
    interaction.prompt.side_effect = lambda message, **kwargs: (
        "deny" if "without the OS sandbox" in message else "approve"
    )
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    try:
        output = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "cat external-link", "cwd": str(native_workspace)}),
                interaction=interaction,
            )
        )
        assert output["ok"] is False
        assert "fake-outside-canary" not in json.dumps(output)
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_yolo_scoped_external_read_uses_native_sandbox_without_saved_grant(
    backend, native_workspace
):
    """A YOLO scope runs in Seatbelt without a prompt and expires when YOLO is disabled."""
    outside = native_workspace.parent / "outside"
    permissions = PermissionManager(native_workspace, yolo=True)
    interaction = Mock()
    interaction.prompt.return_value = "deny"
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = json.dumps(
        {
            "command": "cat external-link",
            "cwd": str(native_workspace),
            "read_only": True,
            "read_roots": [str(outside)],
        }
    )
    try:
        allowed = json.loads(registry.call("run_command", arguments, interaction=interaction))
        assert allowed["ok"], allowed
        assert allowed["result"]["boundary"] == "sandbox"
        assert allowed["result"]["stdout"]["content"] == "fake-outside-canary"
        interaction.prompt.assert_not_called()
        host.run_host_command.assert_not_called()

        permissions.set_yolo(False)
        denied = json.loads(registry.call("run_command", arguments, interaction=interaction))
        assert denied["problem"]["code"] == "tool.denied", denied
        interaction.prompt.assert_called_once()
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_workspace_paths_are_usable_process_paths(public_command, native_workspace):
    """Shell expansion reaches workspace bytes and preserves the process path in output."""
    result = public_command.call(
        'cat "$WORKSPACE_ROOT/ordinary"; printf "%s" "$WORKSPACE_ROOT"', read_only=True
    )
    assert result["ok"] is True
    assert result["result"]["stdout"]["content"] == f"ordinary-data{native_workspace}"
    public_command.host.run_host_command.assert_not_called()


def test_public_command_and_file_tool_preserve_path_like_content(
    public_command, native_workspace, monkeypatch
):
    """Native output reaches model input faithfully except for marked credential filtering."""
    if _SYSTEM_CAT is None:
        pytest.skip("System file tool is required")
    physical = str(native_workspace)
    credential = "registered-credential-123"
    content = (
        json.dumps(
            {
                "workspace": physical,
                "sibling": physical + "-extra/file",
                "url": "https://example.test" + physical + "/file",
                "quoted": shlex.quote(physical),
                "unicode": "é ☃",
                "encoded": base64.b64encode(physical.encode()).decode(),
                "credential": credential,
            },
            ensure_ascii=False,
        )
        + "\r\n"
    )
    source = native_workspace / "path-content.json"
    source.write_text(content, encoding="utf-8", newline="")
    logical = f"{VirtualPath.WORKSPACE}/{source.name}"
    registry = ToolRegistry(
        [run_command, read_text_file],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(public_command.backend, public_command.host),
    )

    def dispatch(tool, arguments):
        """Call a public tool through the same virtual-path context."""
        return json.loads(
            registry.call(
                tool,
                json.dumps(arguments),
                interaction=public_command.interaction,
                instructions_manager=public_command.instructions,
            )
        )

    command = f'{shlex.quote(_SYSTEM_CAT)} "{logical}"'
    path_only = content.replace(credential, "synthetic-marker")
    quoted_source = "printf '%s\\n' '$WORKSPACE_ROOT/quoted'; " + shlex.join(
        ("/bin/sh", "-c", "printf %s '$TMPDIR/embedded'")
    )
    try:
        read = dispatch("read_text_file", {"path": logical})
        success = dispatch("run_command", {"command": command, "read_only": True})
        failure = dispatch(
            "run_command", {"command": f"{command}; {command} >&2; exit 7", "read_only": True}
        )
        literal = dispatch(
            "run_command", {"command": f"printf %s {shlex.quote(path_only)}", "read_only": True}
        )
        quoted = dispatch("run_command", {"command": quoted_source, "read_only": True})
    finally:
        registry.close()
    assert read["ok"] and read["result"]["path"] == logical
    assert read["result"]["content"] == content
    assert success["ok"] and success["result"]["stdout"]["content"] == content
    assert failure["problem"]["code"] == "process.nonzero_exit"
    assert all(
        failure["problem"]["metadata"][stream]["content"] == content
        for stream in ("stdout", "stderr")
    )
    assert literal["ok"] and literal["result"]["stdout"]["content"] == path_only
    assert quoted["ok"] and quoted["result"]["stdout"]["content"] == (
        "$WORKSPACE_ROOT/quoted\n$TMPDIR/embedded"
    )
    public_command.host.run_host_command.assert_not_called()
    sdk = Mock()
    sdk.responses.create.return_value = SimpleNamespace(
        status="completed", output=[], output_text="", usage=None, model="fixture-model"
    )
    monkeypatch.setattr("loop.backend.openai.OpenAI", lambda **_kwargs: sdk)
    model = OpenAIBackend(default_model="fixture-model", api_key=credential)
    model_input = [
        ToolResult(call_id=str(index), output=json.dumps(result))
        for index, result in enumerate((read, success, failure, literal, quoted))
    ]
    list(model.get_response(model_input))
    serialized = [
        json.loads(item["output"]) for item in sdk.responses.create.call_args.kwargs["input"]
    ]
    filtered = content.replace(credential, "<redacted:secret>")
    assert serialized[0]["result"]["content"] == filtered
    assert serialized[0]["result"]["payload_redacted"] is True
    for stream in (
        serialized[1]["result"]["stdout"],
        serialized[2]["problem"]["metadata"]["stdout"],
        serialized[2]["problem"]["metadata"]["stderr"],
    ):
        assert stream["content"] == filtered
        assert stream["payload_redacted"] is True
    assert serialized[3]["result"]["stdout"]["content"] == path_only
    assert "payload_redacted" not in serialized[3]["result"]["stdout"]
    assert serialized[4]["result"]["stdout"]["content"] == (
        "$WORKSPACE_ROOT/quoted\n$TMPDIR/embedded"
    )
    assert "payload_redacted" not in serialized[4]["result"]["stdout"]
    assert credential not in json.dumps(serialized)
    assert credential in model_input[0].output


@pytest.mark.parametrize("metadata_target", ["workspace", "outside", "private"])
def test_unrelated_environment_metadata_does_not_block_workspace_tool(
    public_command, native_workspace, monkeypatch, metadata_target
):
    """An unrelated metadata symlink does not affect a workspace command or outside denials."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    tools = native_workspace / "bin"
    tools.mkdir()
    executable = tools / "fixture-transformer"
    executable.write_text(f"#!{_SYSTEM_SH}\nprintf fixture-ok\n", encoding="utf-8")
    executable.chmod(0o755)
    config = native_workspace / "unrelated-config"
    config.write_text("unrelated project metadata", encoding="utf-8")
    outside = native_workspace.parent / "outside/secret"
    metadata = native_workspace / "pyvenv.cfg"
    metadata.symlink_to(
        config
        if metadata_target == "workspace"
        else outside
        if metadata_target == "outside"
        else native_workspace / ".ssh/id_ed25519"
    )
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))
    source = (
        f"fixture-transformer; if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(outside))} "
        ">/dev/null 2>&1; then exit 9; fi"
    )
    if metadata_target != "workspace":
        source += (
            f"; if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(metadata))} "
            ">/dev/null 2>&1; then exit 10; fi"
        )

    result = public_command.call(source, read_only=True)

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "fixture-ok"
    assert outside.read_text(encoding="utf-8") == "fake-outside-canary"
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_timeout_warns_about_effects_without_replay(backend, native_workspace, monkeypatch):
    """A fake timeout preserves one native command's effects without replay."""
    ready = native_workspace / "ready"
    gate = native_workspace / "gate"
    os.mkfifo(ready)
    os.mkfifo(gate)
    clock = [100.0]
    observed = False

    def expired_after_entry():
        """Advance the fake deadline after the sandboxed child reports its effects."""
        nonlocal observed
        if not observed:
            assert receive_fifo_byte(ready) == b"r"
            observed = True
            clock[0] = 200.0
        return clock[0]

    native_clock = SimpleNamespace(
        monotonic=lambda: clock[0],
        strftime=lambda _format, _value: "2000-01-01 00:00:00+0000",
        localtime=lambda: None,
    )
    monkeypatch.setattr("loop.execution.facade.time", native_clock)
    monkeypatch.setattr("loop.execution.coordinator.time", native_clock)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time", native_clock)
    monkeypatch.setattr("loop.utils.process.time", SimpleNamespace(monotonic=expired_after_entry))
    host = Mock()
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
        settings=ToolRuntimeSettings(command_timeout=2),
    )
    try:
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": "printf x >> count; printf ready; printf r > ready; read _ < gate",
                        "cwd": str(native_workspace),
                    }
                ),
                interaction=interaction,
            )
        )
        assert result["problem"]["code"] == "sandbox.timed_out"
        metadata = result["problem"]["metadata"]
        assert metadata["sandbox_diagnostics"]["possible_effects"] is True
        assert metadata["stdout"]["content"] == "ready"
        assert "not retried" in metadata["recovery_hint"]
        assert "may repeat effects" in metadata["recovery_hint"]
        assert "request_host_command" not in metadata["recovery_hint"]
        assert "host_offer" not in metadata
        assert (native_workspace / "count").read_text(encoding="utf-8") == "x"
        assert observed
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_automatic_discovery_keeps_an_existing_scoped_grant(backend, native_workspace, monkeypatch):
    """A full lookup cache cannot evict a scoped grant or expose an unrelated directory."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    tools = native_workspace.parent / "tools"
    tools.mkdir()
    sibling = tools / "support"
    sibling.write_text("approved-support", encoding="utf-8")
    outside = native_workspace.parent / "outside" / "secret"
    tool = tools / "fixture-builder"
    tool.write_text(
        f"#!{_SYSTEM_SH}\n{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(sibling))}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(outside))} >/dev/null 2>&1; "
        "then exit 9; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    host = Mock()
    registry = ToolRegistry(
        [run_command, resolve_executable],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    try:
        references = [
            json.loads(registry.call("resolve_executable", json.dumps({"name": tool.name})))[
                "result"
            ]["grant_reference"]
            for _ in range(128)
        ]
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": tool.name,
                        "cwd": str(native_workspace),
                        "read_only": True,
                        "executable_grant": references[0],
                    }
                ),
                interaction=interaction,
            )
        )
        assert result["ok"], result
        assert result["result"]["stdout"]["content"] == "approved-support"
        assert outside.read_text(encoding="utf-8") == "fake-outside-canary"
        interaction.prompt.assert_called_once()
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_project_selected_tools_keep_separate_scoped_grants(backend, native_workspace, monkeypatch):
    """Multiple requested tools gain neighboring files only after their roots are approved."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    directories = [native_workspace.parent / "tools-a", native_workspace.parent / "tools-b"]
    names = ("fixture-build", "fixture-format")
    for directory, name in zip(directories, names, strict=True):
        directory.mkdir()
        secret = directory / "sibling"
        secret.write_text("fixture-only", encoding="utf-8")
        tool = directory / name
        tool.write_text(
            f"#!{_SYSTEM_SH}\nif {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} "
            ">/dev/null 2>&1; "
            "then printf visible; else printf denied; fi\n",
            encoding="utf-8",
        )
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((*map(str, directories), os.defpath)))
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    host = Mock()
    registry = ToolRegistry(
        [run_command, resolve_executable],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    try:
        lookup = json.loads(registry.call("resolve_executable", json.dumps({"name": list(names)})))
        assert lookup["ok"], lookup
        found = lookup["result"]["executables"]
        assert [item["name"] for item in found] == list(names)
        references = [item["grant_reference"] for item in found]
        assert references[0] != references[1]
        arguments = {"command": "; ".join(names), "cwd": str(native_workspace), "read_only": True}
        without = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert without["ok"] and without["result"]["stdout"]["content"] == "denieddenied"
        approved = json.loads(
            registry.call(
                "run_command",
                json.dumps(arguments | {"executable_grant": references}),
                interaction=interaction,
            )
        )
        assert approved["ok"] and approved["result"]["stdout"]["content"] == "visiblevisible"
        assert interaction.prompt.called
        wrapper = native_workspace / "wrapper.sh"
        wrapper.write_text(
            f"#!{_SYSTEM_SH}\n" + " && ".join(map(shlex.quote, names)) + "\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        indirect = {"command": "./wrapper.sh", "cwd": str(native_workspace), "read_only": True}
        unavailable = json.loads(
            registry.call("run_command", json.dumps(indirect), interaction=interaction)
        )
        assert not unavailable["ok"]
        enabled = json.loads(
            registry.call(
                "run_command",
                json.dumps(indirect | {"executable_grant": references}),
                interaction=interaction,
            )
        )
        assert enabled["ok"] and enabled["result"]["stdout"]["content"] == "visiblevisible"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize("selection", ["path", "relative", "absolute", "parent", "grant"])
def test_selected_executable_follows_directory_alias_into_approved_root(
    backend, native_workspace, monkeypatch, selection
):
    """An approved installation works through an alias without exposing its siblings."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    installed = native_workspace.parent / "fixture-installed"
    links = native_workspace.parent / "fixture-links"
    outside = native_workspace.parent / "fixture-outside"
    for directory in (installed, links, outside):
        directory.mkdir()
    (installed / "support").write_text("approved-support", encoding="utf-8")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    sibling = links / "sibling"
    sibling.write_text("sibling-canary", encoding="utf-8")
    tool = installed / "actual-executable"
    tool.write_text(
        f"#!{_SYSTEM_SH}\n{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(installed / 'support'))}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(sibling))} >/dev/null 2>&1; "
        "then exit 10; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    lookup = native_workspace / "lookup"
    lookup.mkdir()
    target = alias / tool.name
    (lookup / "fixture-task").symlink_to(
        os.path.relpath(target, lookup) if selection == "relative" else target
    )
    monkeypatch.setenv("PATH", os.pathsep.join((str(lookup), os.defpath)))
    interaction = Mock()
    interaction.prompt.return_value = "deny"
    host = Mock()
    registry = ToolRegistry(
        [run_command, resolve_executable],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    command = {
        "path": "fixture-task",
        "relative": "./lookup/fixture-task",
        "absolute": shlex.quote(str(lookup / "fixture-task")),
        "parent": "./lookup/../lookup/fixture-task",
        "grant": "fixture-task",
    }[selection]
    arguments = {"command": command, "cwd": str(native_workspace), "read_only": True}
    try:
        denied = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert not denied["ok"]
        if selection == "parent":
            interaction.prompt.assert_not_called()
        else:
            interaction.prompt.assert_called_once()
        interaction.prompt.reset_mock()
        interaction.prompt.return_value = "approve"
        if selection == "grant":
            lookup_result = json.loads(
                registry.call(
                    "resolve_executable",
                    json.dumps({"name": "fixture-task"}),
                    interaction=interaction,
                )
            )
            assert lookup_result["ok"]
            arguments |= {"executable_grant": lookup_result["result"]["grant_reference"]}
        else:
            arguments |= {"read_roots": [str(installed)]}
        allowed = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert allowed["ok"], allowed
        assert allowed["result"]["stdout"]["content"] == "approved-support"
        interaction.prompt.assert_called_once()
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert sibling.read_text(encoding="utf-8") == "sibling-canary"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize("alias_mode", ["file", "directory"])
def test_approved_path_scope_supports_indirect_executables(
    backend, native_workspace, monkeypatch, alias_mode
):
    """A PATH alias reaches an approved tool without exposing an unrelated directory."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    installed = native_workspace.parent / "fixture-installed"
    lookup = native_workspace.parent / "fixture-lookup"
    links = native_workspace.parent / "fixture-links"
    outside = native_workspace.parent / "fixture-outside"
    for directory in (installed, lookup, links, outside):
        directory.mkdir()
    (installed / "support").write_text("approved-support", encoding="utf-8")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    sibling = links / "sibling"
    sibling.write_text("sibling-canary", encoding="utf-8")
    tool = installed / "actual-executable"
    tool.write_text(
        f"#!{_SYSTEM_SH}\n{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(installed / 'support'))}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(sibling))} >/dev/null 2>&1; "
        "then exit 10; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    name = "fixture-task" if alias_mode == "file" else "@fixture-加工"
    (lookup / name).symlink_to(tool if alias_mode == "file" else alias / tool.name)
    launcher = native_workspace / "launcher"
    launcher.write_text(f"#!{_SYSTEM_SH}\n{shlex.quote(name)}\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(lookup), os.defpath)))
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = {"command": "./launcher", "cwd": str(native_workspace), "read_only": True}
    try:
        denied = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert not denied["ok"]
        interaction.prompt.assert_not_called()
        allowed = json.loads(
            registry.call(
                "run_command",
                json.dumps(arguments | {"read_roots": [str(installed)]}),
                interaction=interaction,
            )
        )
        assert allowed["ok"], allowed
        assert allowed["result"]["stdout"]["content"] == "approved-support"
        interaction.prompt.assert_called_once()
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert sibling.read_text(encoding="utf-8") == "sibling-canary"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_public_writable_command_cannot_change_policy_coordination_files(
    public_command, native_workspace
):
    """Protect policy, lock, and reset-marker creation while ordinary writes succeed."""
    directory = native_workspace / ".loop"
    directory.mkdir()
    targets = [
        directory / "permissions.yaml",
        directory / "permissions.yaml.lock",
    ]
    for target in targets:
        target.write_text("preserve-policy-canary", encoding="utf-8")
    for target in targets:
        for command in ("printf changed > ", "rm "):
            result = public_command.call(command + shlex.quote(str(target)))
            assert not result["ok"], result
            assert target.read_text(encoding="utf-8") == "preserve-policy-canary"
    marker = directory / "permissions.yaml.reset-pending"
    denied_marker = public_command.call("printf changed > " + shlex.quote(str(marker)))
    assert not denied_marker["ok"], denied_marker
    assert not marker.exists()
    allowed = public_command.call("printf writable > ordinary-output")
    assert allowed["ok"], allowed
    assert (native_workspace / "ordinary-output").read_text(encoding="utf-8") == "writable"
    public_command.host.run_host_command.assert_not_called()


def test_external_environment_keeps_selection_with_narrow_read_scope(
    backend, native_workspace, monkeypatch
):
    """An approved external executable directory retains its environment without sibling reads."""
    runtime = Path(sys.base_prefix).resolve(strict=True)
    binary = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    environment = native_workspace.parent / "external-environment"
    tools = environment / "bin"
    tools.mkdir(parents=True)
    selected = tools / "fixture-runner"
    selected.symlink_to(binary)
    (environment / "pyvenv.cfg").write_text(
        f"home = {binary.parent}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    sibling = environment / "sibling-canary"
    sibling.write_text("outside-data", encoding="utf-8")
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    code = (
        "import pathlib, sys\n"
        "try:\n"
        f"    pathlib.Path({str(sibling)!r}).read_text()\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(9)\n"
        "print(sys.prefix)\n"
    )
    arguments = {
        "command": f"{shlex.quote(str(selected))} -c {shlex.quote(code)}",
        "cwd": str(native_workspace),
        "read_only": True,
        "read_roots": [str(tools), str(runtime)],
    }
    try:
        result = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert result["ok"], result
        assert Path(result["result"]["stdout"]["content"].strip()).resolve() == environment
        assert sibling.read_text(encoding="utf-8") == "outside-data"
        interaction.prompt.assert_called_once()
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize(
    "invocation",
    [
        "direct",
        "direct-file",
        "forwarded",
        "forwarded-chain",
        "forwarded-path",
        "forwarded-path-chain",
        "forwarded-path-earlier",
        "forwarded-path-retarget",
        "forwarded-path-displaced",
        "literal-forward",
        "literal-options",
        "literal-unicode",
        "literal-chain",
        "literal-direct-override",
        "literal-retarget",
        "literal-fixed-retarget",
        "literal-path-override",
        "forwarded-path-override",
        "not-forwarded-chain",
        "outside-relative",
        "path-override",
        "path-override-later",
        "path-assignment",
        "retarget",
        "covered-retarget",
        "quoted-assignment",
    ],
)
def test_workspace_executable_link_requests_installation_scope(
    backend, native_workspace, monkeypatch, native_execution_clock, invocation
):
    """A workspace-selected binary needs approval for its linked installation only."""
    if _SYSTEM_CC is None:
        pytest.skip("System compiler is required")
    installed = native_workspace.parent / "fixture-installed"
    links = native_workspace.parent / "fixture-links"
    outside = native_workspace.parent / "fixture-outside"
    for directory in (installed, links, outside):
        directory.mkdir()
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    secret = native_workspace / ".ssh/id_ed25519"
    program = installed / "actual"
    compiled = subprocess.run(
        [_SYSTEM_CC, "-x", "c", "-o", str(program), "-"],
        input=(
            "#include <stdio.h>\n"
            "int main(void) {\n"
            f'  if (fopen({json.dumps(str(canary))}, "r")) return 9;\n'
            f'  if (fopen({json.dumps(str(secret))}, "r")) return 10;\n'
            '  puts("selected-ok"); return 0;\n}\n'
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    assert subprocess.run([str(program)], check=False).returncode == 9
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    selected = native_workspace / (
        "fixture-加工" if invocation == "literal-unicode" else "fixture-tool"
    )
    selected.symlink_to(
        program
        if invocation
        in {
            "direct-file",
            "retarget",
            "covered-retarget",
            "forwarded-path-retarget",
            "forwarded-path-displaced",
            "literal-retarget",
            "literal-fixed-retarget",
        }
        else alias / program.name
    )
    launcher = native_workspace / "fixture-launcher"
    launcher.write_text(
        "#!/bin/sh\nexec ./fixture-tool\n"
        if invocation == "literal-fixed-retarget"
        else '#!/bin/sh\nexec ./fixture-tool -e "$@"\n'
        if invocation == "literal-options"
        else '#!/bin/sh\nexec ./fixture-加工 "$@"\n'
        if invocation == "literal-unicode"
        else '#!/bin/sh\nexec ./fixture-tool "$@"\n'
        if invocation in {"literal-forward", "literal-direct-override", "literal-retarget"}
        else '#!/bin/sh\nexec ./fixture-forwarder "$@"\n'
        if invocation == "literal-chain"
        else '#!/bin/sh\nexec fixture-tool "$@"\n'
        if invocation == "literal-path-override"
        else '#!/bin/sh\nexec "$1"\n'
        if invocation == "not-forwarded-chain"
        else '#!/bin/sh\nexec \t"$@"\n'
        if invocation in {"forwarded", "forwarded-path-retarget"}
        else '#!/bin/sh\nexec "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    forwarder = native_workspace / "fixture-forwarder"
    forwarder.write_text(
        '#!/bin/sh\nexec ./fixture-tool "$@"\n'
        if invocation == "literal-chain"
        else '#!/bin/sh\nexec "$@"\n',
        encoding="utf-8",
    )
    forwarder.chmod(0o755)
    earlier = native_workspace / "earlier"
    earlier.mkdir()
    if invocation == "forwarded-path-earlier":
        competing = earlier / "fixture-tool"
        competing.write_text('#!/bin/sh\nprintf "earlier-ok\\n"\n', encoding="utf-8")
        competing.chmod(0o755)
    elif invocation == "forwarded-path-displaced":
        (earlier / "fixture-tool").symlink_to(earlier / "missing")
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(earlier), str(native_workspace), os.defpath))
        if invocation
        in {
            "path-override",
            "path-override-later",
            "path-assignment",
            "forwarded-path",
            "forwarded-path-chain",
            "forwarded-path-earlier",
            "forwarded-path-retarget",
            "forwarded-path-displaced",
            "literal-path-override",
            "forwarded-path-override",
        }
        else os.defpath,
    )
    interaction = Mock()
    interaction.prompt.side_effect = ["deny", "approve"]
    if invocation in {
        "retarget",
        "covered-retarget",
        "forwarded-path-retarget",
        "forwarded-path-displaced",
        "literal-retarget",
        "literal-fixed-retarget",
    }:
        replacement = installed / "replacement"
        shutil.copyfile(program, replacement)
        replacement.chmod(0o755)

        def approve_changed_selection(*_args, **_kwargs):
            """Change the selected file while the original scope is being approved."""
            changed = (
                earlier / "fixture-tool" if invocation == "forwarded-path-displaced" else selected
            )
            changed.unlink()
            changed.symlink_to(replacement)
            return "approve"

        interaction.prompt.side_effect = approve_changed_selection
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = {
        "command": "PATH=/nonexistent ./fixture-launcher fixture-tool"
        if invocation == "forwarded-path-override"
        else "PATH=/nonexistent ./fixture-launcher"
        if invocation in {"literal-path-override", "literal-direct-override"}
        else "./fixture-launcher"
        if invocation
        in {
            "literal-forward",
            "literal-options",
            "literal-chain",
            "literal-retarget",
            "literal-fixed-retarget",
            "literal-unicode",
        }
        else "'FOO=bar' ./fixture-tool"
        if invocation == "quoted-assignment"
        else ":; PATH=/nonexistent fixture-tool"
        if invocation == "path-override-later"
        else "PATH=/nonexistent; fixture-tool"
        if invocation == "path-assignment"
        else "PATH=/nonexistent fixture-tool"
        if invocation == "path-override"
        else shlex.quote(os.path.relpath(alias / program.name, native_workspace))
        if invocation == "outside-relative"
        else "./fixture-tool"
        if invocation in {"direct", "direct-file", "retarget", "covered-retarget"}
        else "./fixture-launcher ./fixture-forwarder ./fixture-tool"
        if invocation in {"forwarded-chain", "not-forwarded-chain"}
        else "./fixture-launcher fixture-tool"
        if invocation
        in {
            "forwarded-path",
            "forwarded-path-earlier",
            "forwarded-path-retarget",
            "forwarded-path-displaced",
        }
        else "./fixture-launcher fixture-forwarder fixture-tool"
        if invocation == "forwarded-path-chain"
        else "./fixture-launcher ./fixture-tool",
        "cwd": str(native_workspace),
        "read_only": True,
    }
    if invocation == "covered-retarget":
        arguments["read_roots"] = [str(installed)]
    try:
        denied = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        if invocation in {
            "retarget",
            "covered-retarget",
            "forwarded-path-retarget",
            "forwarded-path-displaced",
            "literal-retarget",
            "literal-fixed-retarget",
        }:
            assert denied["problem"]["code"] == "sandbox.stale", denied
            interaction.prompt.assert_called_once()
            host.run_host_command.assert_not_called()
            return
        if invocation == "not-forwarded-chain":
            assert denied["ok"], denied
            assert denied["result"]["stdout"]["content"] == ""
            interaction.prompt.assert_not_called()
            host.run_host_command.assert_not_called()
            return
        if invocation == "forwarded-path-earlier":
            assert denied["ok"], denied
            assert denied["result"]["stdout"]["content"] == "earlier-ok\n"
            interaction.prompt.assert_not_called()
            host.run_host_command.assert_not_called()
            return
        if invocation in {
            "outside-relative",
            "path-override",
            "path-override-later",
            "path-assignment",
            "quoted-assignment",
            "literal-path-override",
            "forwarded-path-override",
        }:
            assert denied["problem"]["code"] == "process.nonzero_exit", denied
            assert denied["problem"]["metadata"]["exit_code"] not in {9, 10}
            interaction.prompt.assert_not_called()
            return
        assert denied["problem"]["code"] == "tool.denied", denied
        allowed = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert allowed["ok"], allowed
        assert allowed["result"]["stdout"]["content"] == "selected-ok\n"
        assert interaction.prompt.call_count == 2
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert secret.read_text(encoding="utf-8") == "fake-private-canary"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize(
    "env_argument",
    [
        "fixture-engine",
        "-S fixture-engine",
        "-S fixture-engine --fixture",
        "./fixture-tools/fixture-engine",
        "-S ./fixture-tools/fixture-engine --fixture",
    ],
)
@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_env_shebang_requests_selected_interpreter_without_host_retry(
    backend, native_workspace, monkeypatch, native_execution_clock, env_argument, decision
):
    """Native env dispatch observes scoped approval and retains private-file confinement."""
    if _SYSTEM_CC is None or _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell, compiler, and file tools are required")
    installation = native_workspace.parent / "fixture-installation"
    installation.mkdir()
    executable = installation / "actual"
    compiled = subprocess.run(
        [_SYSTEM_CC, "-x", "c", "-o", str(executable), "-"],
        input=(
            "#include <string.h>\n#include <unistd.h>\n"
            "int main(int argc, char **argv) {\n"
            "  if (argc < 2) return 1;\n"
            '  int script = strcmp(argv[1], "--fixture") == 0 ? 2 : 1;\n'
            "  if (argc <= script) return 1;\n"
            f"  execl({json.dumps(_SYSTEM_SH)}, {json.dumps(_SYSTEM_SH)}, argv[script], (char *)0);\n"
            "  return 127;\n}\n"
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    aliases = native_workspace.parent / "fixture-aliases"
    aliases.mkdir()
    alias = aliases / "runtime"
    alias.symlink_to(installation, target_is_directory=True)
    tools = native_workspace / "fixture-tools"
    tools.mkdir()
    (tools / "fixture-engine").symlink_to(alias / executable.name)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))
    outside = native_workspace.parent / "outside" / "secret"
    private = native_workspace / ".ssh" / "id_ed25519"
    task = native_workspace / "fixture-task"
    task.write_text(
        f"#!/usr/bin/env {env_argument}\n"
        "printf started > marker\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(outside))} >/dev/null 2>&1; "
        "then exit 9; fi\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(private))} >/dev/null 2>&1; "
        "then exit 10; fi\n"
        "printf fixture-ok\n",
        encoding="utf-8",
    )
    task.chmod(0o755)
    interaction = Mock()
    interaction.prompt.return_value = decision
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    try:
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "./fixture-task", "cwd": str(native_workspace)}),
                interaction=interaction,
            )
        )
    finally:
        registry.close()
    interaction.prompt.assert_called_once()
    host.run_host_command.assert_not_called()
    if decision == "deny":
        assert not result["ok"] and result["problem"]["code"] == "tool.denied"
        assert not (native_workspace / "marker").exists()
    else:
        assert result["ok"], result
        assert result["result"]["exit_code"] == 0
        assert result["result"]["stdout"]["content"] == "fixture-ok"
        assert (native_workspace / "marker").read_text() == "started"
    assert outside.read_text() == "fake-outside-canary"
    assert private.read_text() == "fake-private-canary"


def test_env_shebang_respects_workspace_path_precedence(
    public_command, native_workspace, monkeypatch
):
    """An earlier workspace interpreter prevents an unrelated installation approval."""
    first = native_workspace / "first-tools"
    first.mkdir()
    earlier = first / "fixture-engine"
    earlier.write_text("#!/bin/sh\nprintf selected-first\n", encoding="utf-8")
    earlier.chmod(0o755)
    installation = native_workspace.parent / "fixture-installation"
    installation.mkdir()
    other = installation / "other"
    other.write_text("#!/bin/sh\nprintf selected-later\n", encoding="utf-8")
    other.chmod(0o755)
    alias = native_workspace.parent / "fixture-alias"
    alias.symlink_to(installation, target_is_directory=True)
    second = native_workspace / "second-tools"
    second.mkdir()
    (second / "fixture-engine").symlink_to(alias / other.name)
    monkeypatch.setenv("PATH", os.pathsep.join((str(first), str(second), os.defpath)))
    task = native_workspace / "fixture-task"
    task.write_text("#!/usr/bin/env fixture-engine\n", encoding="utf-8")
    task.chmod(0o755)

    result = public_command.call("./fixture-task", read_only=True)

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "selected-first"
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_env_noexec_shell_does_not_request_a_forwarded_runtime(
    public_command, native_workspace, monkeypatch
):
    """A shell option that prevents execution does not infer an installation grant."""
    installation = native_workspace.parent / "fixture-installation"
    installation.mkdir()
    interpreter = installation / "actual"
    interpreter.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    interpreter.chmod(0o755)
    alias = native_workspace.parent / "fixture-alias"
    alias.symlink_to(installation, target_is_directory=True)
    selected = native_workspace / "fixture-engine"
    selected.symlink_to(alias / interpreter.name)
    task = native_workspace / "fixture-task"
    task.write_text(f"#!{selected}\n", encoding="utf-8")
    task.chmod(0o755)
    launcher = native_workspace / "fixture-launcher"
    launcher.write_text('#!/usr/bin/env -S fixture-shell -n\nexec "$1"\n', encoding="utf-8")
    launcher.chmod(0o755)
    tools = native_workspace / "fixture-shells"
    tools.mkdir()
    (tools / "fixture-shell").symlink_to("/bin/sh")
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))

    result = public_command.call("./fixture-launcher ./fixture-task", read_only=True)

    assert result["ok"], result
    assert result["result"]["exit_code"] == 0
    assert result["result"]["stdout"]["content"] == ""
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_env_wrapper_does_not_treat_an_unrelated_interpreter_as_a_shell(
    public_command, native_workspace, monkeypatch
):
    """A non-shell executable with a shell-like name cannot create a runtime grant."""
    tools = native_workspace / "fixture-tools"
    tools.mkdir()
    unrelated = tools / "fixture-shell"
    unrelated.write_text("#!/bin/sh\nprintf unrelated-interpreter\n", encoding="utf-8")
    unrelated.chmod(0o755)
    installation = native_workspace.parent / "fixture-installation"
    installation.mkdir()
    interpreter = installation / "actual"
    interpreter.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    interpreter.chmod(0o755)
    alias = native_workspace.parent / "fixture-alias"
    alias.symlink_to(installation, target_is_directory=True)
    selected = native_workspace / "fixture-engine"
    selected.symlink_to(alias / interpreter.name)
    task = native_workspace / "fixture-task"
    task.write_text(f"#!{selected}\n", encoding="utf-8")
    task.chmod(0o755)
    launcher = native_workspace / "fixture-launcher"
    launcher.write_text('#!/usr/bin/env fixture-shell\nexec "$1"\n', encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))

    result = public_command.call("./fixture-launcher ./fixture-task", read_only=True)

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "unrelated-interpreter"
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()
    assert not (native_workspace / "marker").exists()


@pytest.mark.parametrize(
    "invocation",
    [
        "direct",
        "launcher",
        "launcher-name",
        "forward",
        "forward-name",
        "env-forward",
        "env-split-forward",
        "env-safe-forward",
        "env-glob-forward",
        "env-multiple-forward",
        "env-end-forward",
        "direct-safe-forward",
        "literal-fixed",
    ],
)
@pytest.mark.parametrize("engine", ["shell", "reader"])
def test_workspace_script_uses_approved_off_path_interpreter(
    backend, native_workspace, monkeypatch, native_execution_clock, engine, invocation
):
    """An invoked script reaches an approved off-PATH interpreter without exposing canaries."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None or _SYSTEM_CC is None:
        pytest.skip("System shell, compiler, and file tools are required")
    installed = native_workspace.parent / "fixture-installed"
    links = native_workspace.parent / "fixture-links"
    outside = native_workspace.parent / "fixture-outside"
    for directory in (installed, links, outside):
        directory.mkdir()
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    secret = native_workspace / ".ssh/id_ed25519"
    interpreter = installed / "actual"
    system_engine = _SYSTEM_SH if engine == "shell" else _SYSTEM_CAT
    compiled = subprocess.run(
        [_SYSTEM_CC, "-x", "c", "-o", str(interpreter), "-"],
        input=(
            "#include <unistd.h>\n"
            "int main(int argc, char **argv) {\n"
            "  if (argc < 2) return 1;\n"
            f"  execl({json.dumps(system_engine)}, {json.dumps(system_engine)}, argv[1], (char *)0);\n"
            "  return 127;\n}\n"
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    host_probe = native_workspace / "fixture-host-probe"
    host_probe.write_text("printf host-positive\n", encoding="utf-8")
    host_result = subprocess.run(
        [str(interpreter), str(host_probe)], capture_output=True, text=True, check=False
    )
    assert host_result.returncode == 0, host_result.stderr
    assert host_result.stdout == (
        "host-positive" if engine == "shell" else "printf host-positive\n"
    )
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    environment = native_workspace / "off-path-tools"
    environment.mkdir()
    selected = environment / "fixture-engine"
    selected.symlink_to(alias / interpreter.name)
    script = native_workspace / "fixture-task"
    content = f"#!{selected}\nprintf offpath-ok\n"
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    launcher = native_workspace / "fixture-launcher"
    identified = invocation not in {"launcher", "launcher-name"}
    target = "./$1" if invocation in {"launcher-name", "forward-name"} else "$1"
    suffix = "" if invocation != "direct" and identified else "; :"
    header = {
        "env-forward": "#!/usr/bin/env fixture-shell\n",
        "env-split-forward": "#!/usr/bin/env -S fixture-shell\n",
        "env-safe-forward": "#!/usr/bin/env -S fixture-shell -eu\n",
        "env-glob-forward": "#!/usr/bin/env -S fixture-shell -f\n",
        "env-multiple-forward": "#!/usr/bin/env -S fixture-shell -e -u\n",
        "env-end-forward": "#!/usr/bin/env -S fixture-shell --\n",
        "direct-safe-forward": "#!/bin/sh -eu\n",
    }.get(invocation, "#!/bin/sh\n")
    body = "exec ./fixture-task" if invocation == "literal-fixed" else f'exec "{target}"{suffix}'
    launcher.write_text(f"{header}{body}\n", encoding="utf-8")
    launcher.chmod(0o755)
    if invocation.startswith("env-"):
        shell_directory = native_workspace / "fixture-shells"
        shell_directory.mkdir()
        (shell_directory / "fixture-shell").symlink_to(_SYSTEM_SH)
        monkeypatch.setenv("PATH", os.pathsep.join((str(shell_directory), os.defpath)))
    else:
        monkeypatch.setenv("PATH", os.defpath)
    invocation_source = (
        "./fixture-task"
        if invocation == "direct"
        else "./fixture-launcher fixture-task"
        if invocation in {"launcher-name", "forward-name"}
        else "./fixture-launcher ./ignored-task"
        if invocation == "literal-fixed"
        else "./fixture-launcher ./fixture-task"
    )
    command = (
        f"{invocation_source} || exit $?; "
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi; "
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
        "then exit 10; fi"
    )
    interaction = Mock()
    if identified:
        interaction.prompt.side_effect = ["deny", "approve"]
    else:
        interaction.prompt.return_value = "approve"
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = {"command": command, "cwd": str(native_workspace), "read_only": True}
    try:
        denied = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert not denied["ok"]
        if identified:
            assert denied["problem"]["code"] == "tool.denied"
            interaction.prompt.assert_called_once()
        else:
            interaction.prompt.assert_not_called()
        allowed = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    arguments if identified else arguments | {"read_roots": [str(installed)]}
                ),
                interaction=interaction,
            )
        )
        assert allowed["ok"], json.dumps(allowed)
        assert allowed["result"]["stdout"]["content"] == (
            "offpath-ok" if engine == "shell" else content
        )
        assert interaction.prompt.call_count == (2 if identified else 1)
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert secret.read_text(encoding="utf-8") == "fake-private-canary"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("requested", "spelling"),
    [
        (requested, spelling)
        for requested in ("none", "workspace", "extra")
        for spelling in ("plain", "single-space", "double")
    ]
    + [("none", "literal-dollar"), ("none", "dynamic-dollar"), ("none", "wrapped")],
)
def test_direct_script_requests_identified_interpreter_directory(
    backend, native_workspace, monkeypatch, native_execution_clock, requested, spelling
):
    """A selected interpreter directory requires approval and leaves other files confined."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None or _SYSTEM_CC is None:
        pytest.skip("System shell, compiler, and file tools are required")
    installed = native_workspace.parent / "fixture-installed"
    links = native_workspace.parent / "fixture-links"
    outside = native_workspace.parent / "fixture-outside"
    extra = native_workspace.parent / "fixture-extra"
    for directory in (installed, links, outside, extra):
        directory.mkdir()
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    secret = native_workspace / ".ssh/id_ed25519"
    engine = installed / "engine"
    compiled = subprocess.run(
        [_SYSTEM_CC, "-x", "c", "-o", str(engine), "-"],
        input=(
            "#include <unistd.h>\n"
            "int main(int argc, char **argv) {\n"
            "  if (argc < 2) return 1;\n"
            f"  execl({json.dumps(_SYSTEM_SH)}, {json.dumps(_SYSTEM_SH)}, argv[1], (char *)0);\n"
            "  return 127;\n}\n"
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    tools = native_workspace / "tools"
    tools.mkdir()
    interpreter = tools / "engine"
    interpreter.symlink_to(alias / engine.name)
    script_root = (
        native_workspace / "$NAME"
        if spelling in {"literal-dollar", "dynamic-dollar"}
        else native_workspace
    )
    script_root.mkdir(exist_ok=True)
    script = script_root / ("fixture task" if spelling == "single-space" else "fixture-task")
    script.write_text(
        f"#!{interpreter}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
        "then exit 10; fi\n"
        "printf runtime-ok\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    host_control = subprocess.run(
        [str(engine), str(script)], capture_output=True, text=True, check=False
    )
    assert host_control.returncode == 9, host_control.stderr
    monkeypatch.setenv("PATH", os.defpath)
    monkeypatch.delenv("NAME", raising=False)
    interaction = Mock()
    interaction.prompt.side_effect = ["deny", "approve"]
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    invocation = (
        "'./$NAME/fixture-task'"
        if spelling == "literal-dollar"
        else "exec ./fixture-task"
        if spelling == "wrapped"
        else '"./$NAME/fixture-task"'
        if spelling == "dynamic-dollar"
        else "'./fixture task'"
        if spelling == "single-space"
        else '"./fixture-task"'
        if spelling == "double"
        else "./fixture-task"
    )
    arguments = {
        "command": invocation
        if spelling in {"literal-dollar", "dynamic-dollar"}
        else f"{invocation} && :",
        "cwd": str(native_workspace),
        "read_only": True,
    }
    if requested != "none":
        arguments["read_roots"] = [str(native_workspace if requested == "workspace" else extra)]
    try:
        denied = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        if spelling == "dynamic-dollar":
            assert denied["problem"]["code"] == "process.nonzero_exit", denied
            interaction.prompt.assert_not_called()
            return
        assert denied["problem"]["code"] == "tool.denied", denied
        allowed = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert allowed["ok"], allowed
        assert allowed["result"]["stdout"]["content"] == "runtime-ok"
        assert interaction.prompt.call_count == 2
        assert "including neighboring files" in interaction.info.call_args.args[0]
        scope = json.loads(
            interaction.info.call_args.args[0].split("\nScope: ", 1)[1].split("\n", 1)[0]
        )
        assert scope["read_roots"] == (
            [str(extra), str(installed)] if requested == "extra" else [str(installed)]
        )
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert secret.read_text(encoding="utf-8") == "fake-private-canary"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("prepared", "environment_name", "selection"),
    [
        (False, ".venv", "project"),
        (True, ".venv", "project"),
        (True, ".venv", "project-no-bytecode"),
        (True, ".venv", "project-cd"),
        (True, ".venv", "project-timeout"),
        (True, ".venv", "project-timeout-impersonator"),
        (True, "chosen env's", "project"),
        ("retarget", ".venv", "project"),
        (False, ".venv", "direct"),
        (True, ".venv", "direct"),
        (False, ".venv", "script"),
        (True, ".venv", "script"),
        (True, ".venv", "script-virtual"),
        (False, ".venv", "standard-direct"),
        (True, ".venv", "standard-direct"),
    ],
)
def test_workspace_runtime_is_prepared_in_original_call(
    backend,
    native_workspace,
    monkeypatch,
    native_execution_clock,
    prepared,
    environment_name,
    selection,
):
    """A selected workspace runtime works off PATH after scoped approval."""
    if _QUALIFICATION_UV is None:
        pytest.skip("The qualification runner is required")
    timeout = shutil.which("timeout")
    if selection == "project-timeout" and timeout is None:
        pytest.skip("A timeout executable is required")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    if not configuration.is_file():
        pytest.skip("The qualification interpreter has no environment metadata")
    metadata = {
        key.strip(): value.strip()
        for line in configuration.read_text(encoding="utf-8").splitlines()
        for key, separator, value in [line.partition("=")]
        if separator
    }
    home = Path(metadata.get("home", ""))
    if not home.is_absolute() or metadata.get("implementation") != "CPython":
        pytest.skip("The qualification environment has no supported runtime metadata")
    try:
        interpreter = (Path(sys.prefix) / "bin/python").resolve(strict=True)
        runtime = home.resolve(strict=True).parent
    except (OSError, RuntimeError):
        pytest.skip("The qualification runtime is unavailable")
    if not interpreter.is_relative_to(runtime) or home.parent.resolve() == home.parent:
        pytest.skip("The qualification runtime has no selected installation alias")
    environment = native_workspace / environment_name
    commands = environment / "bin"
    if selection == "standard-direct":
        created = subprocess.run(
            [str(home / interpreter.name), "-m", "venv", "--without-pip", str(environment)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert created.returncode == 0, created.stderr
        (commands / "python").unlink()
    else:
        commands.mkdir(parents=True)
    (commands / "python").symlink_to(
        interpreter
        if selection in {"direct", "script", "standard-direct"}
        else home / interpreter.name
    )
    selected = (commands / "python").lstat()
    selected_link = (selected.st_dev, selected.st_ino, os.readlink(commands / "python"))
    if selection != "standard-direct":
        (environment / "pyvenv.cfg").write_text(
            configuration.read_text(encoding="utf-8"), encoding="utf-8"
        )
    (native_workspace / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0.0.0"\nrequires-python = ">=3.12"\n',
        encoding="utf-8",
    )
    canary = native_workspace.parent / "outside/secret"
    private = native_workspace / ".ssh/id_ed25519"
    program = (
        "import os,pathlib,sys\n"
        f"for target in ({str(canary)!r}, {str(private)!r}):\n"
        " try: pathlib.Path(target).read_text()\n"
        " except OSError: pass\n"
        " else: raise SystemExit(9)\n"
        "pathlib.Path(os.environ['TMPDIR'], 'entered').write_text('entered')\n"
        "print(sys.prefix)\n"
    )
    assert subprocess.run([str(interpreter), "-c", program], check=False).returncode == 9
    if selection == "project-timeout-impersonator":
        fake_bin = native_workspace / "fake-bin"
        fake_bin.mkdir()
        fake_timeout = fake_bin / "timeout"
        fake_timeout.write_text('#!/bin/sh\nshift\nexec "$@"\n', encoding="utf-8")
        fake_timeout.chmod(0o755)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            (str(Path(timeout).parent), str(Path(_QUALIFICATION_UV).parent), os.defpath)
            if selection == "project-timeout"
            else (str(fake_bin), str(Path(_QUALIFICATION_UV).parent), os.defpath)
            if selection == "project-timeout-impersonator"
            else (str(Path(_QUALIFICATION_UV).parent), os.defpath)
        ),
    )
    if not prepared:
        monkeypatch.setattr("loop.execution.facade.discover_workspace_runtimes", lambda *_args: ())
        monkeypatch.setattr(
            "loop.execution.facade.discover_workspace_interpreters", lambda *_args: ()
        )
    prefix = (
        ""
        if environment_name == ".venv"
        else f"UV_PROJECT_ENVIRONMENT={shlex.quote(environment_name)} "
    )
    if selection in {"script", "script-virtual"}:
        script = (
            commands / "pytest"
            if selection == "script-virtual"
            else native_workspace / "fixture-task"
        )
        script.write_text(f"#!{commands / 'python'}\n" + program, encoding="utf-8")
        script.chmod(0o755)
        command = (
            'PYTHONDONTWRITEBYTECODE=1 "$WORKSPACE_ROOT"/.venv/bin/pytest'
            if selection == "script-virtual"
            else "./fixture-task"
        )
    else:
        command = (
            "./.venv/bin/python -c "
            if selection in {"direct", "standard-direct"}
            else prefix
            + f"{shlex.quote(Path(_QUALIFICATION_UV).name)} run "
            + ("" if selection in {"project-no-bytecode", "project-cd"} else "--no-sync ")
            + "python -c "
        )
        if selection in {
            "project-no-bytecode",
            "project-cd",
            "project-timeout",
            "project-timeout-impersonator",
        }:
            pytest_script = commands / "pytest"
            pytest_script.write_text(f"#!{commands / 'python'}\n" + program, encoding="utf-8")
            pytest_script.chmod(0o755)
            command = f"PYTHONDONTWRITEBYTECODE=1 {shlex.quote(Path(_QUALIFICATION_UV).name)} --no-cache run pytest"
            if selection == "project-cd":
                command = 'cd "$WORKSPACE_ROOT" && ' + command
            if selection in {"project-timeout", "project-timeout-impersonator"}:
                command = command.removeprefix("PYTHONDONTWRITEBYTECODE=1 ")
                command = 'cd "$WORKSPACE_ROOT" && timeout 10 ' + command + " 2>&1 | tail -20"
    arguments = {
        "command": command
        if selection
        in {
            "script",
            "script-virtual",
            "project-no-bytecode",
            "project-cd",
            "project-timeout",
            "project-timeout-impersonator",
        }
        else command + shlex.quote(program),
        "cwd": str(native_workspace),
        "read_only": selection
        not in {
            "project-no-bytecode",
            "project-cd",
            "project-timeout",
            "project-timeout-impersonator",
        },
    }
    interaction = Mock()
    if prepared == "retarget":

        def approve_changed_selection(*_args, **_kwargs):
            """Change the selected interpreter while its scope is being approved."""
            (commands / "python").unlink()
            (commands / "python").symlink_to("/bin/sh")
            return "approve"

        interaction.prompt.side_effect = approve_changed_selection
    else:
        if prepared:
            interaction.prompt.side_effect = ["deny", "approve"]
        else:
            interaction.prompt.return_value = "approve"
    host = Mock()
    permissions = PermissionManager(native_workspace)
    marker = permissions.temporary_directory / "entered"
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    try:
        first = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        if not prepared:
            assert first["problem"]["code"] == "process.nonzero_exit", first
            assert interaction.prompt.call_count == int(selection != "project")
            assert not marker.exists()
            host.run_host_command.assert_not_called()
            return
        if prepared == "retarget":
            assert first["problem"]["code"] == "sandbox.stale", first
            interaction.prompt.assert_called_once()
            assert not marker.exists()
            host.run_host_command.assert_not_called()
            return
        assert first["problem"]["code"] == "tool.denied", first
        assert not marker.exists()
        allowed = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert allowed["ok"], allowed
        assert Path(allowed["result"]["stdout"]["content"].strip()) == environment
        assert interaction.prompt.call_count == 2
        assert marker.read_text(encoding="utf-8") == "entered"
        assert any(str(runtime) in str(call) for call in interaction.info.call_args_list)
        current = (commands / "python").lstat()
        assert (current.st_dev, current.st_ino, os.readlink(commands / "python")) == selected_link
        assert canary.read_text(encoding="utf-8") == "fake-outside-canary"
        assert private.read_text(encoding="utf-8") == "fake-private-canary"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize(
    "invocation", ["direct", "launcher", "direct-quoted", "forward", "executable"]
)
def test_workspace_script_uses_approved_off_path_virtual_environment(
    backend, native_workspace, monkeypatch, invocation
):
    """An existing off-PATH virtual environment stays selected under the approved runtime."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    if not interpreter.is_relative_to(runtime) or not interpreter.is_file():
        pytest.skip("The active interpreter is not contained in its base runtime")
    links = native_workspace.parent / "fixture-links"
    outside = native_workspace.parent / "fixture-outside"
    links.mkdir()
    outside.mkdir()
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    secret = native_workspace / ".ssh/id_ed25519"
    alias = links / "runtime"
    alias.symlink_to(runtime, target_is_directory=True)
    environment = native_workspace / "chosen-environment"
    commands = environment / "arbitrary-commands"
    commands.mkdir(parents=True)
    selected = commands / "fixture-engine"
    selected.symlink_to(alias / interpreter.relative_to(runtime))
    (environment / "pyvenv.cfg").write_text(
        f"home = {interpreter.parent}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    script = native_workspace / (
        "fixture task" if invocation == "direct-quoted" else "fixture-task"
    )
    script.write_text(
        f"#!{selected}\nimport sys\nprint('runtime-ok', sys.prefix)\n", encoding="utf-8"
    )
    script.chmod(0o755)
    launcher = native_workspace / "fixture-launcher"
    suffix = "" if invocation == "forward" else "; :"
    launcher.write_text(f'#!/bin/sh\nexec "$1"{suffix}\n', encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.defpath)
    invocation_source = (
        "./chosen-environment/arbitrary-commands/fixture-engine -c "
        + shlex.quote("import sys; print('runtime-ok', sys.prefix)")
        if invocation == "executable"
        else "'./fixture task'"
        if invocation == "direct-quoted"
        else "./fixture-task"
        if invocation == "direct"
        else "./fixture-launcher ./fixture-task"
    )
    command = (
        f"{invocation_source} || exit $?; "
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi; "
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
        "then exit 10; fi"
    )
    interaction = Mock()
    if invocation != "launcher":
        interaction.prompt.side_effect = ["deny", "approve"]
    else:
        interaction.prompt.return_value = "approve"
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = {"command": command, "cwd": str(native_workspace), "read_only": True}
    try:
        denied = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert not denied["ok"]
        if invocation != "launcher":
            assert denied["problem"]["code"] == "tool.denied"
            interaction.prompt.assert_called_once()
        else:
            interaction.prompt.assert_not_called()
        allowed = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    arguments
                    if invocation != "launcher"
                    else arguments | {"read_roots": [str(runtime)]}
                ),
                interaction=interaction,
            )
        )
        assert allowed["ok"], allowed
        assert allowed["result"]["stdout"]["content"] == f"runtime-ok {environment}\n"
        assert interaction.prompt.call_count == (1 if invocation == "launcher" else 2)
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert secret.read_text(encoding="utf-8") == "fake-private-canary"
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_indirect_alias_order_reuses_remembered_native_approval(
    backend, native_workspace, monkeypatch
):
    """Reordering PATH entries keeps the same scope and denies protected reads."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    installed = native_workspace.parent / "fixture-installed"
    lookup = native_workspace.parent / "fixture-lookup"
    links = native_workspace.parent / "fixture-links"
    for directory in (installed, lookup, links):
        directory.mkdir()
    for name in ("second", "first"):
        package = installed / name
        package.mkdir()
        tool = package / "actual"
        tool.write_text(f"#!{_SYSTEM_SH}\nprintf {name}\n", encoding="utf-8")
        tool.chmod(0o755)
        alias = links / name
        alias.symlink_to(package, target_is_directory=True)
        (lookup / name).symlink_to(alias / tool.name)
    protected = native_workspace / ".ssh/id_ed25519"
    launcher = native_workspace / "launcher"
    launcher.write_text(
        f"#!{_SYSTEM_SH}\nfirst\nsecond\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(protected))} >/dev/null 2>&1; "
        "then exit 9; fi\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(lookup), os.defpath)))
    interaction = Mock()
    interaction.prompt.return_value = "session"
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = {
        "command": "./launcher",
        "cwd": str(native_workspace),
        "read_only": True,
        "read_roots": [str(installed)],
    }
    original_scandir = os.scandir
    try:
        first = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )

        def reversed_scandir(path):
            """Reverse the selected PATH directory while leaving its contents intact."""
            entries = original_scandir(path)
            if isinstance(path, int) or Path(path) != lookup:
                return entries
            with entries:
                return nullcontext(iter(reversed(list(entries))))

        monkeypatch.setattr("loop.execution.facade.os.scandir", reversed_scandir)
        second = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        assert first["ok"] and second["ok"], (first, second)
        assert first["result"]["stdout"]["content"] == "firstsecond"
        assert second["result"]["stdout"]["content"] == "firstsecond"
        assert protected.read_text(encoding="utf-8") == "fake-private-canary"
        interaction.prompt.assert_called_once()
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_indirect_path_alias_does_not_skip_an_unapproved_earlier_executable(
    backend, native_workspace, monkeypatch
):
    """A later approved alias does not replace the first executable selected on PATH."""
    if _SYSTEM_SH is None:
        pytest.skip("A system shell is required")
    earlier = native_workspace.parent / "fixture-earlier"
    installed = native_workspace.parent / "fixture-installed"
    lookup = native_workspace.parent / "fixture-lookup"
    links = native_workspace.parent / "fixture-links"
    for directory in (earlier, installed, lookup, links):
        directory.mkdir()
    for directory, filename, output in (
        (earlier, "fixture-task", "first-choice"),
        (installed, "actual", "later-choice"),
    ):
        executable = directory / filename
        executable.write_text(f"#!{_SYSTEM_SH}\nprintf {output}\n", encoding="utf-8")
        executable.chmod(0o755)
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    (lookup / "fixture-task").symlink_to(alias / "actual")
    launcher = native_workspace / "launcher"
    launcher.write_text(f"#!{_SYSTEM_SH}\nfixture-task\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(earlier), str(lookup), os.defpath)))
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = {"command": "./launcher", "cwd": str(native_workspace), "read_only": True}
    try:
        denied = json.loads(
            registry.call(
                "run_command",
                json.dumps(arguments | {"read_roots": [str(installed)]}),
                interaction=interaction,
            )
        )
        assert not denied["ok"]
        assert denied["problem"]["metadata"]["stdout"]["content"] == ""
        allowed = json.loads(
            registry.call(
                "run_command",
                json.dumps(arguments | {"read_roots": [str(earlier), str(installed)]}),
                interaction=interaction,
            )
        )
        assert allowed["ok"], allowed
        assert allowed["result"]["stdout"]["content"] == "first-choice"
        assert interaction.prompt.call_count == 2
        host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize(
    "source",
    [
        'cat "$WORKSPACE_ROOT/external-link"',
        "/bin/sh -c 'cat ../outside/secret'",
        "printf unsafe > .git/config",
        "printf unsafe > ../outside/write",
    ],
)
def test_public_native_denial_never_retries_on_host(public_command, native_workspace, source):
    """A declined unrestricted retry preserves protected data after public native denial."""
    result = public_command.call(source)
    assert result["ok"] is False
    assert "fake-outside-canary" not in json.dumps(result)
    assert (native_workspace / ".git" / "config").read_text() == "preserve-git"
    assert (native_workspace.parent / "outside" / "write").read_text() == "preserve-outside"
    public_command.host.run_host_command.assert_not_called()


def test_public_nested_seatbelt_refusal_never_retries_on_host(public_command):
    """A rejected second Seatbelt profile remains a sandbox result without host escalation."""
    result = public_command.call(
        "/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' "
        "/bin/sh -c 'printf unexpected-inner-sandbox'"
    )

    assert result["problem"]["code"] == "process.nonzero_exit"
    metadata = result["problem"]["metadata"]
    assert metadata["boundary"] == "sandbox"
    assert "sandbox_apply: Operation not permitted" in metadata["stderr"]["content"]
    assert "unexpected-inner-sandbox" not in metadata["stdout"]["content"]
    public_command.host.run_host_command.assert_not_called()


def test_read_only_command_writes_only_owned_scratch(public_command, native_workspace):
    """A read-only command may use private scratch but cannot write workspace coverage data."""
    scratch = public_command.call(
        'printf scratch > "$TMPDIR/report"; cat "$TMPDIR/report"', read_only=True
    )
    assert scratch["ok"] is True
    assert scratch["result"]["stdout"]["content"] == "scratch"
    denied = public_command.call("printf unsafe > .coverage", read_only=True)
    assert denied["ok"] is False
    assert not (native_workspace / ".coverage").exists()
    assert not (native_workspace / "report").exists()
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    ("command", "content", "size", "loss"),
    [
        ("printf '\\377'; printf '\\357\\277\\275' >&2", "�", 3, "unknown"),
        ("printf '\\303\\251'; printf '\\303\\251' >&2", "é", 2, "none"),
        ("printf 'a\\rb\\r\\nc\\n'; printf 'a\\rb\\r\\nc\\n' >&2", "a\rb\r\nc\n", 7, "none"),
    ],
)
@pytest.mark.parametrize("exit_code", [0, 7])
def test_command_output_reports_decoding_loss(
    public_command, command, content, size, loss, exit_code
):
    """Successful and failed native output preserve newlines and report decoding loss."""
    result = public_command.call(f"{command}; exit {exit_code}", read_only=True)

    output = result["result"] if exit_code == 0 else result["problem"]["metadata"]
    if exit_code:
        assert result["problem"]["code"] == "process.nonzero_exit"
    for name in ("stdout", "stderr"):
        stream = output[name]
        assert stream["content"] == content
        assert stream["captured_bytes"] == size
        assert stream["capture_complete"] is True
        assert stream["decoding_loss"] == loss
    public_command.host.run_host_command.assert_not_called()


def test_literal_tmp_does_not_grant_host_temporary_files(public_command, native_directory):
    """Host /tmp stays inaccessible while the bound session scratch remains usable."""
    host_directory = native_directory("/tmp")
    marker = host_directory / "marker"
    marker.write_text("host-temporary-canary", encoding="utf-8")
    host_spelling = Path("/tmp") / host_directory.name
    read = public_command.call("cat " + shlex.quote(str(host_spelling / "marker")), read_only=True)
    assert read["ok"] is False
    assert "host-temporary-canary" not in json.dumps(read)
    write = public_command.call(
        "printf forbidden > " + shlex.quote(str(host_spelling / "output")), read_only=True
    )
    assert write["ok"] is False
    assert not (host_directory / "output").exists()
    private = public_command.call(
        'printf private > "$TMPDIR/output"; cat "$TMPDIR/output"', read_only=True
    )
    assert private["ok"] is True
    assert private["result"]["stdout"]["content"] == "private"
    assert marker.read_text(encoding="utf-8") == "host-temporary-canary"
    public_command.host.run_host_command.assert_not_called()


def test_shared_developer_cache_names_do_not_grant_host_temporary_files(
    public_command, native_file
):
    """Developer-cache-shaped host files remain outside sandbox authority."""
    if None in (_SYSTEM_GETCONF, _SYSTEM_CAT, _SYSTEM_PRINTF):
        pytest.skip("System temporary-directory and file tools are required")
    root = Path(
        subprocess.check_output([_SYSTEM_GETCONF, "DARWIN_USER_TEMP_DIR"], text=True).strip()
    ).resolve(strict=True)
    marker = native_file(root, name=f"xcrun_db-{secrets.token_hex(16)}")
    output = native_file(root, name=f"loop-test-{secrets.token_hex(16)}-{secrets.token_hex(3)}.o")
    output.unlink()
    marker.write_text("private-cache-canary", encoding="utf-8")
    read = public_command.call(
        f"{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(marker))}", read_only=True
    )
    assert not read["ok"]
    assert "private-cache-canary" not in json.dumps(read)
    write = public_command.call(
        f"{shlex.quote(_SYSTEM_PRINTF)} forbidden > {shlex.quote(str(output))}", read_only=True
    )
    assert not write["ok"]
    assert not output.exists()
    assert marker.read_text(encoding="utf-8") == "private-cache-canary"


@pytest.mark.parametrize("source", ["make test", 'cd "$WORKSPACE_ROOT"; make test 2>&1'])
@pytest.mark.parametrize("policy_case", ["new", "default_saved", "custom_saved"])
def test_make_prepares_a_literal_indirect_tool_in_one_call(
    public_command, native_workspace, monkeypatch, policy_case, source
):
    """An indirect build tool runs with one scope review unless a custom ceiling needs two."""
    if _SYSTEM_MAKE is None or _SYSTEM_SH is None:
        pytest.skip("System build tool and shell are required")
    root = native_workspace.parent / "make-tool"
    root.mkdir()
    canary = native_workspace.parent / "outside/secret"
    tool = root / "fixture-indirect-tool"
    tool.write_text(
        f"#!{_SYSTEM_SH}\n"
        f"if cat {shlex.quote(str(canary))} >/dev/null 2>&1; then exit 9; fi\n"
        "printf indirect-ready",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    (native_workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\t@fixture-indirect-tool\n", encoding="utf-8"
    )
    workspace_bin = native_workspace / ".venv/bin"
    workspace_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join((str(workspace_bin), str(root), os.defpath)))

    result = public_command.call(source, read_only=True)

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "indirect-ready"
    assert canary.read_text(encoding="utf-8") == "fake-outside-canary"
    assert public_command.interaction.prompt.call_count == (
        2 if policy_case == "custom_saved" else 1
    )
    if policy_case == "custom_saved":
        policy = yaml.safe_load(
            (native_workspace / ".loop/permissions.yaml").read_text(encoding="utf-8")
        )
        assert policy["limit_classifications"]["readable_roots"] == "approval"
        assert policy["limits"]["readable_roots"] == ["workspace"]
    public_command.host.run_host_command.assert_not_called()


def test_make_prepares_workspace_venv_runtime_in_one_call(public_command, native_workspace):
    """A literal recipe authorizes its selected workspace interpreter before launch."""
    if _SYSTEM_MAKE is None:
        pytest.skip("System make is required")
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(native_workspace / ".venv")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    (native_workspace / "Makefile").write_text(
        '.PHONY: test\ntest:\n\t@./.venv/bin/python -c "print(1)"\n'
    )
    result = public_command.call(f"{_SYSTEM_MAKE} test", read_only=True)
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "1\n"
    assert public_command.interaction.prompt.call_count <= 1
    public_command.host.run_host_command.assert_not_called()


def test_make_runs_unrelated_system_tools_in_one_call(public_command, native_workspace):
    """One build request runs separate Perl and awk programs under Seatbelt."""
    if (
        _SYSTEM_MAKE is None
        or not Path("/usr/bin/perl").is_file()
        or not Path("/usr/bin/awk").is_file()
    ):
        pytest.skip("System make, Perl and awk are required")
    (native_workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\t@perl -e 'print \"perl\"'\n\t@awk 'BEGIN { print 23 }'\n",
        encoding="utf-8",
    )
    result = public_command.call(f"{_SYSTEM_MAKE} test", read_only=True)
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "perl23\n"
    public_command.host.run_host_command.assert_not_called()


def test_make_recipe_can_use_session_scratch(public_command, native_workspace):
    """System build recipes use session scratch without accessing another session."""
    if _SYSTEM_MAKE is None or _SYSTEM_CAT is None or _SYSTEM_PRINTF is None:
        pytest.skip("System build and file tools are required")
    make = shlex.quote(_SYSTEM_MAKE)
    scratch = VirtualPath.TEMPORARY
    temporary = TMPDIR_ENVIRONMENT_VARIABLE
    make_args = f'{temporary}="{scratch}"'
    makefile = public_command.scratch / "fixture.mk"
    makefile.write_text(
        f'all:\n\t@{shlex.quote(_SYSTEM_PRINTF)} recipe > "$${temporary}/recipe.txt"\n',
        encoding="utf-8",
    )
    source = f'{make} -f "{scratch}/{makefile.name}"'
    default = public_command.call(source, read_only=True)
    assert default["ok"] is True, default
    assert (public_command.scratch / "recipe.txt").read_text(encoding="utf-8") == "recipe"
    result = public_command.call(f"{source} {make_args}", read_only=True)
    assert result["ok"] is True, result
    assert (public_command.scratch / "recipe.txt").read_text(encoding="utf-8") == "recipe"
    other = SessionManager(
        session_store=SQLiteSessionStore(
            native_workspace.parent / "other-session" / "sessions.db", workspace_id="other"
        ),
        workspace_id="other",
    )
    with other.execution():
        canary = other.temporary_directory.get() / "canary"
        canary.write_text("other-session-canary", encoding="utf-8")
        makefile.write_text(
            f"all:\n\t@{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))}\n", encoding="utf-8"
        )
        denied = public_command.call(source, read_only=True)
        assert denied["ok"] is False
        assert "other-session-canary" not in json.dumps(denied)
        assert canary.read_text(encoding="utf-8") == "other-session-canary"
    public_command.host.run_host_command.assert_not_called()


def test_developer_tool_shim_makefile_from_stdin_uses_session_scratch(public_command):
    """System build tools can create their own temporary input in private scratch."""
    if _SYSTEM_MAKE is None or _SYSTEM_PRINTF is None:
        pytest.skip("System build and output tools are required")
    source = (
        f"{shlex.quote(_SYSTEM_PRINTF)} 'all:\\n\\t@{shlex.quote(_SYSTEM_PRINTF)} private\\n' "
        f"| {shlex.quote(_SYSTEM_MAKE)} -f -"
    )
    result = public_command.call(source, read_only=True)
    assert result["ok"] is True, result
    assert result["result"]["stdout"]["content"] == "private"
    public_command.host.run_host_command.assert_not_called()


def test_system_compiler_uses_private_temporary_storage(public_command):
    """A discovered native compiler can build without shared temporary-cache access."""
    if _SYSTEM_CC is None:
        pytest.skip("A system compiler is required")
    source = public_command.scratch / "source.c"
    output = public_command.scratch / "result.o"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    command = (
        f'{shlex.quote(_SYSTEM_CC)} -c "{VirtualPath.TEMPORARY}/{source.name}" '
        f'-o "{VirtualPath.TEMPORARY}/{output.name}"'
    )
    result = public_command.call(command, read_only=True)
    assert result["ok"], result
    assert output.is_file() and output.stat().st_size > 0
    public_command.host.run_host_command.assert_not_called()


def test_developer_tool_discovery_preserves_scratch_for_stdin(public_command, native_workspace):
    """The selected developer tool uses private scratch and cannot read another session."""
    if None in (_SYSTEM_MAKE, _SYSTEM_PRINTF, _SYSTEM_XCRUN, _SYSTEM_CAT):
        pytest.skip("Developer tool discovery, a build tool, and an output tool are required")
    make_name = shlex.quote(Path(_SYSTEM_MAKE).name)
    finder = shlex.quote(_SYSTEM_XCRUN)
    printer = shlex.quote(_SYSTEM_PRINTF)
    makefile = shlex.quote(f'all:\n\t@{printer} private > "$$TMPDIR/output"\n')
    source = (
        f"set -e; tool=$({finder} --find {make_name}); "
        f'{printer} %s {makefile} | "$tool" -f -; '
        f'{shlex.quote(_SYSTEM_CAT)} "$TMPDIR/output"'
    )
    result = public_command.call(source, read_only=True)
    assert result["ok"] is True, result
    assert result["result"]["stdout"]["content"] == "private"
    other = SessionManager(
        session_store=SQLiteSessionStore(
            native_workspace.parent / "developer-tool-other" / "sessions.db", workspace_id="other"
        ),
        workspace_id="other",
    )
    with other.execution():
        canary = other.temporary_directory.get() / "canary"
        canary.write_text("other-session-canary", encoding="utf-8")
        makefile = shlex.quote(f"all:\n\t@{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))}\n")
        command = (
            f'set -e; tool=$({finder} --find {make_name}); {printer} %s {makefile} | "$tool" -f -'
        )
        denied = public_command.call(command, read_only=True)
        assert denied["ok"] is False
        assert "other-session-canary" not in json.dumps(denied)
        assert canary.read_text(encoding="utf-8") == "other-session-canary"
    public_command.host.run_host_command.assert_not_called()


def test_public_python_session_uses_scratch_without_ancestor_read_access(
    public_command, monkeypatch
):
    """Nested session ownership works while ancestor directory listings remain denied."""
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + ":/usr/bin:/bin")
    source = """
import os
from pathlib import Path
from unittest.mock import Mock
from loop import Interaction, SessionManager

manager = SessionManager(interaction=Mock(spec=Interaction))
with manager.execution():
    report = Path(manager.temporary_directory) / "report"
    report.write_text("session-ok")
    print(report.read_text())
try:
    os.listdir("/private")
except PermissionError:
    print("ancestor-denied")
else:
    raise AssertionError("ancestor listing was allowed")
"""
    result = public_command.call(
        "python -B -c " + shlex.quote(source),
        read_only=True,
        read_roots=[str(Path(sys.prefix).resolve()), str(Path(path_module.__file__).parents[2])],
    )

    assert result["ok"], result
    assert result["result"]["boundary"] == "sandbox"
    assert result["result"]["stdout"]["content"] == "session-ok\nancestor-denied\n"
    assert result["result"]["stderr"]["content"] == ""
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("system_path", [False, True])
def test_system_files_remain_readable_without_host_access(public_command, monkeypatch, system_path):
    """Ordinary system files remain available while unrelated host files stay confined."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/System" if system_path else "/usr/bin:/bin")
    system_version = Path("/System/Library/CoreServices/SystemVersion.plist")
    if not system_version.is_file():
        pytest.skip("This macOS installation does not expose the system version file")
    result = public_command.call(
        f"cat {shlex.quote(str(system_version))} >/dev/null && "
        'if cat "$WORKSPACE_ROOT/external-link" >/dev/null 2>&1; then exit 9; fi',
        read_only=True,
    )
    assert result["ok"], json.dumps(result, indent=2)
    public_command.host.run_host_command.assert_not_called()


def test_system_read_roots_keep_private_names_blocked_with_parent_on_path(
    public_command, native_workspace, monkeypatch
):
    """Overlapping system and PATH roots retain private-name protection."""
    parent = native_workspace.parent / "system-root"
    system = parent / "Library"
    (system / ".ssh").mkdir(parents=True)
    (system / "ordinary").write_text("ordinary-system")
    (system / ".ssh/secret").write_text("private-system-canary")
    monkeypatch.setattr(
        macos_backend, "_SYSTEM_READ_ROOTS", (*macos_backend._SYSTEM_READ_ROOTS, system)
    )
    monkeypatch.setenv("PATH", os.pathsep.join(("/usr/bin", "/bin", str(parent))))
    result = public_command.call(
        f"cat {shlex.quote(str(system / 'ordinary'))} && "
        f"if cat {shlex.quote(str(system / '.ssh/secret'))} 2>/dev/null; then exit 9; fi",
        read_only=True,
    )
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "ordinary-system"
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("scratch_script", [False, True])
def test_public_project_environment_loads_managed_runtime_dependencies(
    public_command, native_workspace, monkeypatch, scratch_script
):
    """A selected project interpreter can load its managed package dependencies in one call."""
    uv = _QUALIFICATION_UV
    environment = Path(sys.prefix)
    config = environment / "pyvenv.cfg"
    interpreter = (environment / "bin/python").resolve()
    if uv is None or not config.is_file():
        pytest.skip("An active uv-managed project environment is required")
    installation = public_command.backend.installed_tool_roots(interpreter)
    if installation is None:
        pytest.skip("The active interpreter has invalid package metadata")
    roots, _ = installation
    if len(roots) < 2 or not all(public_command.backend.managed_tool_root(root) for root in roots):
        pytest.skip("The active interpreter needs verified managed package dependencies")
    (native_workspace / "external-link").unlink()
    shutil.rmtree(native_workspace / ".ssh")
    commands = native_workspace / ".venv/bin"
    commands.mkdir(parents=True)
    (commands / "python").symlink_to(interpreter)
    (commands / "python3").symlink_to("python")
    shutil.copyfile(config, commands.parent / "pyvenv.cfg")
    (native_workspace / "pyproject.toml").write_text(
        '[project]\nname = "native-runtime"\nversion = "0.0.0"\n'
        "[tool.uv]\npackage = false\n"
        '[tool.pytest.ini_options]\naddopts = ["-p", "no:cacheprovider"]\n'
        'pythonpath = ["."]\n',
        encoding="utf-8",
    )
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = commands.parent / "lib" / version / "site-packages"
    shutil.copytree(
        environment / "lib" / version / "site-packages",
        site_packages,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_editable_impl_loop.pth"),
    )
    pytest_entry = commands / "pytest"
    pytest_entry.write_text(
        f"#!{commands / 'python'}\nimport sys\nfrom _pytest.config import _console_main\n"
        "sys.exit(_console_main())\n",
        encoding="utf-8",
    )
    pytest_entry.chmod(0o755)
    (native_workspace / "sample.py").write_text("def value():\n    return 42\n")
    (native_workspace / "tests/unit").mkdir(parents=True)
    (native_workspace / "tests/unit/test_sample.py").write_text(
        "import sqlite3\nfrom sample import value\n\ndef test_value():\n    assert value() == 42\n"
    )
    monkeypatch.setenv("PATH", os.pathsep.join((str(Path(uv).parent), os.defpath)))
    command = (
        "PYTHONDONTWRITEBYTECODE=1 uv --no-cache run --no-sync "
        "pytest tests/unit --cov=sample --cov-report=term-missing:skip-covered 2>&1"
    )
    if scratch_script:
        (public_command.scratch / "run_tests.sh").write_text(
            '#!/bin/bash\ncd "$WORKSPACE_ROOT"\n'
            + command
            + ' > "$TMPDIR/test_output.txt"\necho EXIT_CODE=$?\ncat "$TMPDIR/test_output.txt"\n'
        )
        command = 'bash "$TMPDIR/run_tests.sh"'
    else:
        command = 'cd "$WORKSPACE_ROOT"; ' + command
    result = public_command.call(command)
    assert result["ok"], json.dumps(result, indent=2)
    assert "1 passed" in result["result"]["stdout"]["content"]
    if scratch_script:
        assert "EXIT_CODE=0" in result["result"]["stdout"]["content"]
    assert public_command.interaction.prompt.call_count == 1
    assert any("sqlite" in str(call) for call in public_command.interaction.info.call_args_list)
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("separate_store", [False, True])
def test_public_runtime_dependencies_support_custom_package_prefix(
    public_command, native_workspace, separate_store
):
    """An explicitly approved selected runtime can use dependencies in a custom prefix."""
    prefix = native_workspace.parent / "custom-prefix"
    store = native_workspace.parent / "separate-packages" if separate_store else prefix / "Cellar"
    runtime = store / "runtime/1"
    dependency = store / "support/2"
    (runtime / "bin").mkdir(parents=True)
    dependency.mkdir(parents=True)
    opt = prefix / "opt"
    opt.mkdir(parents=True)
    (opt / "runtime").symlink_to(runtime, target_is_directory=True)
    (opt / "support").symlink_to(dependency, target_is_directory=True)
    (runtime / "INSTALL_RECEIPT.json").write_text(
        '{"runtime_dependencies": [{"full_name": "support"}]}'
    )
    (dependency / "INSTALL_RECEIPT.json").write_text("{}")
    (dependency / "data").write_text("custom-dependency")
    executable = runtime / "bin/engine"
    outside = native_workspace.parent / "outside/secret"
    executable.write_text(
        f"#!/bin/sh\nif /bin/cat {shlex.quote(str(outside))} >/dev/null 2>&1; "
        "then exit 9; fi\n"
        f"exec /bin/cat {shlex.quote(str(opt / 'support/data'))}\n"
    )
    executable.chmod(0o755)
    commands = native_workspace / ".venv/bin"
    commands.mkdir(parents=True)
    (commands / "python").symlink_to(opt / "runtime/bin/engine")
    (commands.parent / "pyvenv.cfg").write_text(
        f"home = {opt / 'runtime/bin'}\nimplementation = CPython\nversion_info = 9.8.7\n"
    )

    assert not public_command.backend.managed_tool_root(runtime)
    result = public_command.call("./.venv/bin/python -c pass")
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "custom-dependency"
    assert public_command.interaction.prompt.call_count == 1
    assert any(
        str(dependency) in str(call) for call in public_command.interaction.info.call_args_list
    )
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("timed", [False, True])
def test_piped_make_prepares_its_selected_tool(
    public_command, native_workspace, monkeypatch, timed
):
    """A piped build prepares its selected tool without weakening the sandbox."""
    uv = _QUALIFICATION_UV
    if uv is None or _SYSTEM_MAKE is None:
        pytest.skip("A managed tool and system build tool are required")
    timeout = shutil.which("timeout")
    if timed and timeout is None:
        pytest.skip("A timeout executable is required")
    (native_workspace / "Makefile").write_text(
        f".PHONY: test\ntest:\n\t{shlex.quote(Path(uv).name)} --version\n", encoding="utf-8"
    )
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            (str(Path(timeout).parent), str(Path(uv).parent), os.defpath)
            if timed
            else (str(Path(uv).parent), os.defpath)
        ),
    )
    wrapper = "timeout 10 " if timed else ""
    result = public_command.call(f'cd "$WORKSPACE_ROOT" && {wrapper}make test 2>&1 | tail -200')

    assert result["ok"], json.dumps(result, indent=2)
    assert "uv " in result["result"]["stdout"]["content"]
    outside = public_command.call('cat "$WORKSPACE_ROOT/external-link"')
    assert not outside["ok"]
    assert "fake-outside-canary" not in json.dumps(outside)
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "full_project",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.environ.get("LOOP_NATIVE_FULL_PROJECT_TEST") != "extended-native-check",
                reason="Run the full project test workflow explicitly.",
            ),
        ),
    ],
)
def test_public_session_runs_pytest_from_private_unusual_scratch(
    backend, native_workspace, monkeypatch, native_execution_clock, full_project
):
    """Public dispatch runs small or full project tests in durable session scratch."""
    uv = _QUALIFICATION_UV
    if uv is None:
        pytest.skip("uv is required to qualify the project test workflow")
    if _SYSTEM_MAKE is None:
        pytest.skip("A system build tool is required")
    uv_name = Path(uv).name
    python_name = Path(sys.executable).name
    make = shlex.quote(_SYSTEM_MAKE)
    (native_workspace / "external-link").unlink()
    shutil.rmtree(native_workspace / ".ssh")
    if full_project:
        project = Path(__file__).resolve().parents[3]
        for relative in ("src", "tests/unit", "tests/architecture"):
            shutil.copytree(
                project / relative,
                native_workspace / relative,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for relative in (
            "Makefile",
            "pyproject.toml",
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/scheduling.py",
        ):
            target = native_workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)
    else:
        (native_workspace / "pyproject.toml").write_text(
            '[project]\nname = "native-test-fixture"\nversion = "0.0.0"\n'
            "[tool.uv]\npackage = false\n",
            encoding="utf-8",
        )
        (native_workspace / "Makefile").write_text(
            ".PHONY: test test-architecture\n"
            "test:\n\tuv --no-cache run --no-sync pytest -p no:cacheprovider -q tests/unit "
            "--cov=sample --cov-report=term-missing:skip-covered\n"
            "test-architecture:\n\tuv --no-cache run --no-sync pytest -p no:cacheprovider -q "
            "tests/unit\n",
            encoding="utf-8",
        )
        (native_workspace / "sample.py").write_text("def value():\n    return 42\n")
        tests = native_workspace / "tests/unit"
        tests.mkdir(parents=True)
        (tests / "test_sample.py").write_text(
            "from sample import value\n\ndef test_value():\n    assert value() == 42\n"
        )
    database = native_workspace.parent / "Library" / "Application Support" / "sessions.db"
    sessions = SessionManager(
        session_store=SQLiteSessionStore(database, workspace_id="pytest-native"),
        workspace_id="pytest-native",
    )
    permissions = PermissionManager(native_workspace)
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host launch")
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        [run_command, resolve_executable, read_cached_content],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
        settings=ToolRuntimeSettings(command_timeout=120),
    )
    loop = Loop.create_default(
        backend=SimpleNamespace(default_model="test", get_context_window=lambda _: None),
        working_directory=native_workspace,
        session_manager=sessions,
        interaction=interaction,
        permission_manager=permissions,
        tool_registry=registry,
    )
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(Path(uv).parent), str(Path(sys.executable).parent), os.defpath)),
    )
    try:
        assert "Library/Application Support" in str(loop.permission_manager.temporary_directory)
        lookup = json.loads(
            loop.tool_registry.call(
                "resolve_executable",
                json.dumps({"name": [uv_name, python_name]}),
                interaction=interaction,
                instructions_manager=loop.instructions_manager,
            )
        )
        assert lookup["ok"], json.dumps(lookup, indent=2)
        environment = (
            f'PYTHONPATH="{VirtualPath.WORKSPACE}{"/src" if full_project else ""}" '
            f"UV_PROJECT_ENVIRONMENT={shlex.quote(str(Path(sys.prefix).resolve()))} "
            + ("UV_NO_SYNC=1 " if full_project else "")
        )
        references = [item["grant_reference"] for item in lookup["result"]["executables"]]
        ungranted = json.loads(
            loop.tool_registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": environment + f"{make} test-architecture",
                        "cwd": "$WORKSPACE_ROOT",
                        "read_roots": [str(Path(sys.prefix).resolve())],
                    }
                ),
                interaction=interaction,
                instructions_manager=loop.instructions_manager,
            )
        )
        assert not ungranted["ok"]
        assert ungranted["problem"]["code"] == "process.nonzero_exit"
        denial = ungranted["problem"]["metadata"]["stderr"]["content"]
        assert (uv_name if full_project else "Operation not permitted") in denial
        direct_test = (
            "tests/unit/agent/test_runner.py -k prepares_env_interpreter"
            if full_project
            else "tests/unit/test_sample.py"
        )
        commands = [
            (
                (
                    f"{shlex.quote(uv_name)} --no-cache run pytest "
                    f"-o addopts= -p no:cacheprovider -q -x --tb=short {direct_test}"
                ),
                " passed",
            ),
            (f"{make} test", "100.00%" if full_project else "100%"),
        ]
        if _SYSTEM_XCRUN is not None:
            commands.append(
                (
                    (
                        f'"$({shlex.quote(_SYSTEM_XCRUN)} --find '
                        f'{shlex.quote(Path(_SYSTEM_MAKE).name)})" test-architecture'
                    ),
                    "passed",
                )
            )
        for command, expected in commands:
            result = json.loads(
                loop.tool_registry.call(
                    "run_command",
                    json.dumps(
                        {
                            "command": environment + command,
                            "cwd": "$WORKSPACE_ROOT",
                            "read_roots": [str(Path(sys.prefix).resolve())],
                            "executable_grant": references,
                        }
                    ),
                    interaction=interaction,
                    instructions_manager=loop.instructions_manager,
                )
            )
            if result["ok"]:
                assert result["result"]["boundary"] == "sandbox"
                stream = result["result"]["stdout"]
            else:
                stream = result["problem"].get("metadata", {}).get("stdout", {})
            content = stream.get("content", "")
            cursor = stream.get("next_cursor")
            while cursor and (not result["ok"] or expected not in content):
                continuation = json.loads(
                    loop.tool_registry.call(
                        "read_cached_content",
                        json.dumps({"handle": stream["handle"], "cursor": cursor}),
                        interaction=interaction,
                        instructions_manager=loop.instructions_manager,
                    )
                )
                assert continuation["ok"], continuation
                page = continuation["result"]
                content += page["content"]
                cursor = page.get("next_cursor")
            assert result["ok"], json.dumps(result, indent=2) + "\nFull stdout:\n" + content
            assert expected in content
        host.run_host_command.assert_not_called()
    finally:
        loop.command_manager.close()
        registry.close()
        permissions.close()


@pytest.mark.parametrize("component", [".ssh", ".aws", ".gnupg", ".config", "Library", ".AWS"])
def test_public_writable_command_cannot_change_private_workspace_entries(
    public_command, native_workspace, component
):
    """Private credential components reject overwrite and creation through public execution."""
    directory = native_workspace / component
    directory.mkdir(exist_ok=True)
    existing = directory / "credential"
    existing.write_text("preserve-private-canary", encoding="utf-8")
    created = directory / "new-entry"

    overwritten = public_command.call(f"printf changed > {shlex.quote(str(existing))}")
    created_result = public_command.call(f"printf changed > {shlex.quote(str(created))}")

    assert overwritten["ok"] is False
    assert created_result["ok"] is False
    assert existing.read_text(encoding="utf-8") == "preserve-private-canary"
    assert not created.exists()
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("role", ["workspace", "temporary", "extra", "skill"])
@pytest.mark.parametrize("component", [".ssh", ".AWS", ".config", ".loop", "Library"])
@pytest.mark.parametrize("read_only", [False, True])
def test_public_granted_roots_keep_private_components_denied(
    public_command, native_workspace, role, component, read_only
):
    """Canonical and virtual grants protect private content at every depth and access mode."""
    if role == "workspace":
        root, virtual, reads = native_workspace, str(native_workspace), []
    elif role == "temporary":
        root, virtual, reads = public_command.scratch, "$TMPDIR", []
    elif role == "skill":
        public_command.instructions.activate_skill("fixture")
        root, virtual, reads = public_command.skill_root, str(public_command.skill_root), []
    else:
        root = native_workspace.parent / "extra"
        root.mkdir()
        virtual, reads = str(root), [str(root)]

    def shell_path(value):
        """Expand the session token while preserving literal path suffix characters."""
        return '"' + value + '"' if value.startswith("$TMPDIR") else shlex.quote(value)

    (root / "neighbor").write_text("ordinary-readable")
    for prefix in ("", "nested/"):
        directory = root / prefix / component
        directory.mkdir(parents=True, exist_ok=True)
        existing = directory / "credential"
        existing.write_text("private-root-canary")
        created = directory / "new-entry"
        for spelling in dict.fromkeys((str(existing), f"{virtual}/{prefix}{component}/credential")):
            result = public_command.call(
                f": {shell_path(virtual)}; cat {shell_path(spelling)}; "
                f"printf changed > {shell_path(spelling)}; "
                f"printf changed > {shell_path(spelling.rsplit('/', 1)[0] + '/new-entry')}",
                read_only=read_only,
                read_roots=reads,
            )
            assert result["ok"] is False, result
            assert "private-root-canary" not in json.dumps(result)
            assert existing.read_text() == "private-root-canary"
            assert not created.exists()
    ordinary = public_command.call(
        f"cat {shell_path(virtual + '/neighbor')}", read_only=read_only, read_roots=reads
    )
    assert ordinary["result"]["stdout"]["content"] == "ordinary-readable"
    public_command.host.run_host_command.assert_not_called()


def test_public_skill_root_shell_uses_physical_path_without_rewriting(public_command):
    """An active skill's physical path works while its typed virtual root stays literal in shell."""
    root = public_command.skill_root
    (root / "neighbor").write_text("ordinary-skill-data", encoding="utf-8")
    public_command.instructions.activate_skill("fixture")

    physical = public_command.call(f"cat {shlex.quote(str(root / 'neighbor'))}", read_only=True)
    assert physical["ok"], physical
    assert physical["result"]["stdout"]["content"] == "ordinary-skill-data"

    virtual = public_command.call(
        "printf '%s\\n' '/skills/fixture/neighbor'; cat /skills/fixture/neighbor", read_only=True
    )
    assert virtual["problem"]["code"] == "process.nonzero_exit", virtual
    assert virtual["problem"]["metadata"]["stdout"]["content"] == "/skills/fixture/neighbor\n"
    public_command.host.run_host_command.assert_not_called()


def test_public_skill_alias_rejects_private_hardlink(public_command, native_workspace):
    """An activated read-only skill cannot expose a private inode through a public alias."""
    root = public_command.skill_root
    private = root / ".ssh"
    private.mkdir()
    source = private / "key"
    source.write_text("private-skill-canary")
    (root / "public-alias").hardlink_to(source)
    public_command.instructions.activate_skill("fixture")

    result = public_command.call("cat " + shlex.quote(str(root / "public-alias")), read_only=True)

    assert result["problem"]["code"] == "sandbox.invalid", result
    assert "private-skill-canary" not in json.dumps(result)
    assert source.read_text() == "private-skill-canary"
    public_command.host.run_host_command.assert_not_called()


def test_public_session_scratch_denies_private_names(public_command):
    """Reused session storage enforces private-name restrictions on command-created paths."""
    for component in (".ssh", ".AWS", ".config", ".loop", "Library"):
        result = public_command.call(
            f'mkdir -p "$TMPDIR/nested"; mkdir "$TMPDIR/nested/{component}"; '
            f'printf fixture > "$TMPDIR/nested/{component}/key"; '
            f'cat "$TMPDIR/nested/{component}/key"',
            read_only=True,
        )
        assert result["problem"]["code"] == "process.nonzero_exit", result
        assert result["problem"]["metadata"]["stdout"]["content"] == ""
    ordinary = public_command.call('printf ordinary > "$TMPDIR/result"; cat "$TMPDIR/result"')
    assert ordinary["result"]["stdout"]["content"] == "ordinary"
    public_command.host.run_host_command.assert_not_called()


def test_public_writable_command_keeps_ordinary_workspace_writes_available(
    public_command, native_workspace
):
    """Private-component write denials leave ordinary writable workspace files available."""
    result = public_command.call("printf ordinary-updated > ordinary")

    assert result["ok"] is True
    assert (native_workspace / "ordinary").read_text(encoding="utf-8") == "ordinary-updated"
    public_command.host.run_host_command.assert_not_called()


def test_public_external_write_grant_stays_in_approved_directory(public_command, native_workspace):
    """Seatbelt permits one approved external write and denies sibling and control writes."""
    external = native_workspace.parent / "external-write"
    sibling = native_workspace.parent / "external-sibling"
    external.mkdir()
    sibling.mkdir()
    control = native_workspace / ".loop"
    control.mkdir()
    protected = control / "policy"
    protected.write_text("unchanged", encoding="utf-8")
    grants = {"write_roots": [str(external)]}

    approved = public_command.call(
        "printf approved > " + shlex.quote(str(external / "allowed")), **grants
    )
    assert approved["ok"], approved
    assert (external / "allowed").read_text(encoding="utf-8") == "approved"

    outside = public_command.call(
        "printf denied > " + shlex.quote(str(sibling / "blocked")), **grants
    )
    assert outside["problem"]["code"] == "process.nonzero_exit", outside
    assert not (sibling / "blocked").exists()

    control_write = public_command.call('printf denied > "$WORKSPACE_ROOT"/.loop/policy', **grants)
    assert control_write["problem"]["code"] == "process.nonzero_exit", control_write
    assert protected.read_text(encoding="utf-8") == "unchanged"
    public_command.host.run_host_command.assert_not_called()


def test_public_default_network_limit_refuses_general_grant(public_command, native_workspace):
    """The default hard private-network limit stops a broad grant before native launch."""
    result = public_command.call("printf unsafe > network-marker", network=True)

    assert result["problem"]["code"] == "sandbox.unavailable", result
    assert "private-network limit" in result["problem"]["detail"]
    assert not (native_workspace / "network-marker").exists()
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_public_attempt_local_unix_ipc_needs_no_approval_or_host_fallback(public_command):
    """One public command exchanges local Unix data under the default hard network limit."""
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        "my $path = qq($ENV{LOOP_LOCAL_SOCKET_DIR}/public.sock); "
        "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "bind($s, pack_sockaddr_un($path)) or die $!; listen($s, 1) or die $!; "
        "socket(my $c, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "connect($c, pack_sockaddr_un($path)) or die $!; "
        "defined accept(my $p, $s) or die $!; send($c, qq(local), 0) or die $!; "
        "my $data = qq(); defined recv($p, $data, 5, 0) or die $!; print $data;"
    )

    result = public_command.call(source, read_only=True)

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "local"
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_public_local_network_deny_removes_macos_socket_authority(public_command):
    """A user hard limit blocks local Unix sockets through public command dispatch."""
    public_command.registry.permission_manager.set_limit(
        "deny_local_networks", True, scope=PolicyScope.SESSION
    )
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "my $path = qq($ENV{TMPDIR}/denied.sock); "
        "print bind($s, pack_sockaddr_un($path)) ? qq(allowed) : 0+$!;"
    )

    result = public_command.call(source, read_only=True)

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == str(errno.EPERM)
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_managed_local_network_deny_removes_macos_socket_authority(
    public_command, native_workspace
):
    """An active managed deny reaches native Seatbelt without a host retry or prompt."""
    parent = native_workspace.parent / "system"
    directory = parent / "loop"
    directory.mkdir(parents=True)
    parent.chmod(0o755)
    directory.chmod(0o755)
    identity = "a" * 32
    expectation = parent / "loop.managed.expect.json"
    policy = directory / "managed-policy.json"
    expectation.write_text(
        json.dumps({"version": 1, "identity": identity, "minimum_revision": 1}),
        encoding="utf-8",
    )
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "identity": identity,
                "revision": 1,
                "rules": [],
                "deny_local_networks": True,
            }
        ),
        encoding="utf-8",
    )
    expectation.chmod(0o644)
    policy.chmod(0o644)
    manager = PermissionManager(
        native_workspace,
        managed_policy_store=ManagedPolicyStore(
            directory, trust_anchor=native_workspace.parent, owner_uid=os.getuid()
        ),
    )
    registry = ToolRegistry(
        [run_command],
        permission_manager=manager,
        execution_service=CommandExecutionService(public_command.backend, public_command.host),
    )
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "my $path = qq($ENV{TMPDIR}/denied.sock); "
        "print bind($s, pack_sockaddr_un($path)) ? qq(allowed) : 0+$!;"
    )
    try:
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": source, "cwd": "$WORKSPACE_ROOT", "read_only": True}),
                interaction=public_command.interaction,
                instructions_manager=public_command.instructions,
            )
        )
        assert result["ok"], result
        assert result["result"]["stdout"]["content"] == str(errno.EPERM)
        public_command.interaction.prompt.assert_not_called()
        public_command.host.run_host_command.assert_not_called()
    finally:
        registry.close()


@pytest.mark.skipif(
    os.environ.get("LOOP_NATIVE_MANAGED_POLICY_TEST") != "macos-root-fixture",
    reason="Exact root-owned macOS policy fixture must be provisioned separately",
)
def test_root_managed_policy_controls_macos_yolo_host_and_native_network(
    public_command, native_workspace
):
    """The fixed root-owned policy overrides YOLO and removes native local socket authority."""
    store = ManagedPolicyStore()
    assert store.directory == Path("/Library/Loop")
    assert store.expectation_path == Path("/Library/loop.managed.expect.json")
    snapshot = store.load()
    assert snapshot is not None
    assert snapshot.policy.identity == "0123456789abcdef0123456789abcdef"
    assert snapshot.policy.revision == 1
    assert snapshot.policy.forbid_host
    assert snapshot.policy.deny_local_networks
    assert {rule.id for rule in snapshot.policy.rules} == {
        "native-deny-session",
        "native-deny-egress",
    }
    for path in (store.expectation_path, store.directory / "managed-policy.json"):
        details = path.stat()
        assert details.st_uid == 0
        assert stat.S_IMODE(details.st_mode) == 0o644
        write_probe = subprocess.run(
            [sys.executable, "-c", "import os,sys; os.open(sys.argv[1],os.O_WRONLY)", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert write_probe.returncode != 0
        assert "PermissionError" in write_probe.stderr

    manager = public_command.registry.permission_manager
    manager.set_yolo(True)
    session = (
        Operation(
            tool_id="test",
            action=Action.SESSION_MUTATE,
            target=SessionTarget(identifier="native"),
        ),
    )
    host = (
        Operation(
            tool_id="request_host_command",
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=("/bin/sh", "-c", "printf native"),
                cwd=str(native_workspace),
                boundary=ProcessBoundary.HOST,
            ),
        ),
    )
    assert manager.evaluate(session).sources == ("managed:deny:native-deny-session",)
    assert manager.authorize(session).decision is Decision.DENY
    assert manager.evaluate(host).sources == ("managed:host_forbidden",)
    assert manager.authorize(host).decision is Decision.DENY

    egress = (
        Operation(
            tool_id="run_command",
            action=Action.NETWORK_REQUEST,
            target=NetworkTarget(url="http://11.0.0.2:18080", origin="http://11.0.0.2:18080"),
        ),
    )
    assert manager.evaluate(egress).sources == ("managed:deny:native-deny-egress",)
    assert manager.authorize(egress).decision is Decision.DENY

    denied_egress = public_command.call(
        "printf unexpected", egress_origins=["http://11.0.0.2:18080"]
    )
    assert not denied_egress["ok"], denied_egress
    assert denied_egress["problem"]["code"] == "tool.denied"
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "my $path = qq($ENV{TMPDIR}/denied.sock); "
        "print bind($s, pack_sockaddr_un($path)) ? qq(allowed) : 0+$!;"
    )
    denied_local = public_command.call(source, read_only=True)
    assert denied_local["ok"], denied_local
    assert denied_local["result"]["stdout"]["content"] == str(errno.EPERM)
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_public_mediated_origin_uses_scoped_approval_without_host_fallback(
    live_egress_clock, native_workspace
):
    """An approved macOS mediator keeps ordinary command effects inside the sandbox."""
    result = live_egress_clock.call(
        "printf unsafe > mediated-marker", egress_origins=["https://example.invalid"]
    )

    assert result["ok"], result
    assert (native_workspace / "mediated-marker").read_text() == "unsafe"
    live_egress_clock.host.run_host_command.assert_not_called()


def test_native_general_network_grant_reaches_host_loopback(backend, native_workspace):
    """Seatbelt's general network switch can reach a host listener when deliberately granted."""
    if not Path("/usr/bin/nc").is_file():
        pytest.skip("The system netcat client is unavailable")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        port = listener.getsockname()[1]
        request = SandboxRequest.create(
            source=f"printf granted | /usr/bin/nc -w 1 127.0.0.1 {port}",
            cwd=native_workspace,
            workspace=native_workspace,
            read_roots=(),
            write_roots=(native_workspace,),
            network=True,
            environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            policy_version=backend.policy_version,
            deadline=time.monotonic() + 10,
            workspace_id="native-general-network",
        )
        result = backend.run(request)
        assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
        with listener.accept()[0] as connection:
            assert connection.recv(32) == b"granted"


def test_public_user_enabled_general_network_is_an_explicit_alternative(public_command):
    """A trusted limit change permits separately approved general networking on macOS."""
    if not Path("/usr/bin/nc").is_file():
        pytest.skip("The system netcat client is unavailable")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        port = listener.getsockname()[1]
        public_command.registry.permission_manager.set_limit(
            "deny_private_networks", False, scope=PolicyScope.SESSION
        )
        result = public_command.call(
            f"printf granted | /usr/bin/nc -w 1 127.0.0.1 {port}", network=True
        )
        assert result["ok"], result
        assert result["result"]["boundary"] == "sandbox"
        with listener.accept()[0] as connection:
            assert connection.recv(32) == b"granted"
        public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        "printf text >/dev/null; printf done",
        "printf text >|/dev/null; printf done",
        "printf text 2>&-; printf done",
    ],
)
def test_public_safe_redirections_run_without_approval(public_command, source):
    """Null-device and managed temporary output preserve native enforcement without prompts."""
    result = public_command.call(source, read_only=True)
    assert result["ok"] is True
    public_command.interaction.prompt.assert_not_called()
    public_command.host.run_host_command.assert_not_called()


def test_public_safe_redirect_does_not_bypass_workspace_overwrite_review(
    public_command, native_workspace
):
    """A harmless output sink cannot hide a preceding workspace truncation."""
    interaction = Mock()
    interaction.prompt.return_value = "deny"
    result = public_command.call(
        "printf unsafe >ordinary >/dev/null", chosen_interaction=interaction
    )
    assert result["problem"]["code"] == "tool.denied"
    assert (native_workspace / "ordinary").read_text() == "ordinary-data"
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("code", [7, 17, 65, 71, 127])
def test_public_nonzero_is_not_a_host_authorization(public_command, code):
    """Preserve the real exit and reject unrestricted execution after forged permission text."""
    result = public_command.call(f"printf 'Operation not permitted' >&2; exit {code}")
    assert result["ok"] is False
    assert result["problem"]["metadata"]["exit_code"] == code
    assert "host_offer" not in result["problem"]["metadata"]
    assert ("executable_hint" in result["problem"]["metadata"]) is (code == 127)
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("exit_code", [7, 127])
def test_effectful_native_denial_is_not_replayed(public_command, native_workspace, exit_code):
    """A real protected read and forged denial text cannot replay an effectful command."""
    if _SYSTEM_CAT is None:
        pytest.skip("System file tool is required")
    canary = native_workspace.parent / "outside/secret"
    assert subprocess.check_output([_SYSTEM_CAT, str(canary)], text=True) == "fake-outside-canary"
    source = (
        "printf x >> count; "
        f'if {shlex.quote(_SYSTEM_CAT)} "$WORKSPACE_ROOT/external-link" >/dev/null 2>&1; '
        f"then exit 9; fi; printf 'Operation not permitted' >&2; exit {exit_code}"
    )

    result = public_command.call(source)

    assert result["problem"]["code"] == "process.nonzero_exit", result
    metadata = result["problem"]["metadata"]
    assert metadata["exit_code"] == exit_code
    assert metadata["sandbox_diagnostics"]["possible_effects"] is True
    assert metadata["sandbox_diagnostics"]["cause"] == "unknown"
    assert "host_offer" not in metadata
    if exit_code == 127:
        assert "indirect tools remain unidentified" in metadata["executable_hint"]
        assert "may already have had effects" in metadata["executable_hint"]
    else:
        assert "executable_hint" not in metadata
    assert (native_workspace / "count").read_text(encoding="utf-8") == "x"
    assert canary.read_text(encoding="utf-8") == "fake-outside-canary"
    public_command.host.run_host_command.assert_not_called()


def test_explicitly_declined_destructive_command_preserves_input(public_command, native_workspace):
    """Destructive review is enforced before native or unrestricted execution."""
    interaction = Mock()
    interaction.prompt.return_value = "deny"
    result = public_command.call("rm -f ordinary", chosen_interaction=interaction)
    assert result["problem"]["code"] == "tool.denied"
    assert (native_workspace / "ordinary").read_text() == "ordinary-data"
    public_command.host.run_host_command.assert_not_called()


def test_large_output_remains_bounded(public_command):
    """Public native output capture drains large output while retaining its configured limit."""
    result = public_command.call("/usr/bin/yes x | /usr/bin/head -c 100000", read_only=True)
    assert result["ok"] is True
    output = result["result"]["stdout"]
    assert len(output["content"]) <= MAX_OUTPUT_CHARS
    assert output["truncated"] is True
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "relative",
    [
        ".gitignore",
        ".agentignore",
        "nested/.gitignore",
        "nested/.agentignore",
        "nested/AGENTS.md",
        "nested/POLICY+.md",
        "nested/.agents/skills/demo/asset.txt",
    ],
)
def test_native_protected_controls_resist_opaque_overwrite(
    public_command, native_workspace, relative
):
    """Configured instructions, ignore files and nested skill assets survive opaque shell writes."""
    target = native_workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("preserve-control")
    result = public_command.call(f"eval 'printf changed > {relative}'")
    assert result["ok"] is False
    assert target.read_text() == "preserve-control"
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        "rm -rf ordinary",
        "env rm -f ordinary",
        "env -i /bin/rm ordinary",
        "env -u HOME -- /bin/rm ordinary",
        "env --unset=HOME /bin/rm ordinary",
        "FOO=1 rm -f ordinary",
        ": > ordinary",
        "printf changed > ordinary",
        "printf changed &> ordinary",
        "printf changed >& ordinary",
        "printf changed 1>& ordinary",
        "cp sample.txt ordinary",
        "mv sample.txt ordinary",
        "install sample.txt ordinary",
        "dd if=sample.txt of=ordinary",
        "tee ordinary < sample.txt",
        "sed -i.bak s/ordinary/changed/ ordinary",
        "perl -i -pe 's/ordinary/changed/' ordinary",
        "env cp sample.txt ordinary && echo done",
        "FOO=1 command mv sample.txt ordinary",
        "sh -c 'install sample.txt ordinary'",
        "git add sample.txt",
        "git checkout other-branch",
        "git clean -fd",
        "env -i /usr/bin/git clean -fd",
    ],
)
def test_destructive_forms_require_fresh_review(public_command, native_workspace, source):
    """Direct, wrapped and nested destructive forms cannot launch after review is declined."""
    (native_workspace / "sample.txt").write_text("replacement")
    interaction = Mock()
    interaction.prompt.return_value = "deny"
    result = public_command.call(source, chosen_interaction=interaction)
    assert result["problem"]["code"] == "tool.denied"
    assert (native_workspace / "ordinary").read_text() == "ordinary-data"
    interaction.prompt.assert_called_once()
    public_command.host.run_host_command.assert_not_called()


def test_renamed_tool_keeps_native_boundary(public_command):
    """A public registration alias preserves native command enforcement."""
    result = public_command.call("printf renamed", name="renamed_command")
    assert result["ok"] is True
    assert result["result"]["boundary"] == "sandbox"
    assert result["result"]["stdout"]["content"] == "renamed"


@pytest.mark.parametrize(
    "native_workspace",
    [
        'workspace"quote',
        "workspace\\slash",
        "workspace\nline",
        "workspace[0]+;$()",
        'workspace") (allow default) ;',
    ],
    indirect=True,
)
def test_public_special_workspace_names_preserve_policy(public_command, native_workspace):
    """Virtual paths with unusual host spellings retain protected and outside denies."""
    allowed = public_command.call('cat "$WORKSPACE_ROOT/ordinary"', read_only=True)
    assert allowed["ok"] and allowed["result"]["stdout"]["content"] == "ordinary-data"
    targets = ["AGENTS.md", 'POLICY"name.md', "POLICY\nname.md"]
    for name in targets:
        path = native_workspace / name
        path.write_text("preserve-instructions")
        denied = public_command.call(
            "eval " + shlex.quote("printf changed > " + shlex.quote(str(native_workspace / name)))
        )
        assert not denied["ok"]
        assert "host_offer" not in denied["problem"]["metadata"]
        assert path.read_text() == "preserve-instructions"
    for source in (
        'cat "$WORKSPACE_ROOT/.ssh/id_ed25519"',
        'cat "$WORKSPACE_ROOT/external-link"',
        "eval 'printf bad > ../outside/write'",
    ):
        denied = public_command.call(source)
        assert not denied["ok"]
        assert "host_offer" not in denied["problem"]["metadata"]
        assert "fake-private-canary" not in json.dumps(denied)
        assert "fake-outside-canary" not in json.dumps(denied)
    assert (native_workspace.parent / "outside/write").read_text() == "preserve-outside"
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("kind", ["script", "binary", "symlink"])
def test_public_exact_path_executable_does_not_expose_siblings(
    public_command, native_workspace, monkeypatch, kind
):
    """A safe exact PATH executable runs without authority over neighboring file data."""
    tools = native_workspace.parent / 'external"tools'
    tools.mkdir(mode=0o700)
    executable = tools / "reviewtool"
    if kind == "binary":
        shutil.copyfile("/usr/bin/true", executable)
    else:
        executable.write_text("#!/bin/sh\nprintf exact-file-ok\n")
    executable.chmod(0o700)
    if kind == "symlink":
        target = tools / "target"
        executable.rename(target)
        executable.symlink_to(target)
    sibling = tools / "sibling"
    sibling.write_text("outside-sibling-canary")
    monkeypatch.setenv("PATH", str(tools) + ":/usr/bin:/bin")
    for registration in ("run_command", "renamed_command"):
        allowed = public_command.call("reviewtool", name=registration, read_only=True)
        assert allowed["ok"] and allowed["result"]["boundary"] == "sandbox"
        assert allowed["result"]["stdout"]["content"] == (
            "" if kind == "binary" else "exact-file-ok"
        )
    public_command.interaction.prompt.assert_not_called()
    denied = public_command.call("cat " + shlex.quote(str(sibling)), read_only=True)
    assert not denied["ok"]
    assert "outside-sibling-canary" not in json.dumps(denied)
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("storage", ["loop", "Loop é ;$' (data)"])
def test_session_files_survive_commands_turns_and_runtime_recreation(
    backend, native_workspace, storage
):
    """TMPDIR and file tools reopen the same files after a saved session resumes."""
    database = native_workspace.parent / "Library" / storage / "workspace" / "sessions.db"
    runtimes = []
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host launch")
    interaction = Mock()
    interaction.prompt.side_effect = lambda message, **kwargs: (
        ApprovalChoice.ONCE
        if message == "Proceed?"
        else "deny"
        if "without the OS sandbox" in message
        else "approve"
    )

    def compose(identifier=None):
        """Recreate the runtime using a fresh durable store and the existing native backend."""
        manager = SessionManager(
            session=identifier,
            session_store=SQLiteSessionStore(database, workspace_id="workspace"),
            workspace_id="workspace",
        )
        permissions = PermissionManager(native_workspace)
        registry = ToolRegistry(
            [run_command, read_text_file, write_text_file],
            permission_manager=permissions,
            execution_service=CommandExecutionService(backend, host),
        )
        loop = Loop.create_default(
            backend=SimpleNamespace(default_model="test", get_context_window=lambda _: None),
            working_directory=native_workspace,
            session_manager=manager,
            interaction=interaction,
            permission_manager=permissions,
            tool_registry=registry,
        )
        runtimes.append(loop)
        return loop, manager

    def dispatch(loop, name, **arguments):
        """Invoke one public tool using the runtime's active virtual paths."""
        value = json.loads(
            loop.tool_registry.call(
                name,
                json.dumps(arguments),
                interaction=interaction,
                instructions_manager=loop.instructions_manager,
            )
        )
        assert value["ok"], json.dumps(value)
        return value["result"]

    try:
        loop, sessions = compose()
        sessions.add_user_message("prepare a report")
        identifier = sessions.session.id
        directory = loop.permission_manager.temporary_directory
        dispatch(
            loop,
            "run_command",
            command='printf draft > "$TMPDIR/report.txt"',
            cwd="$WORKSPACE_ROOT",
            read_only=True,
        )
        canonical = dispatch(loop, "run_command", command='cd "$TMPDIR"; pwd -P', read_only=True)
        assert canonical["stdout"]["content"].strip() == str(directory.resolve())
        dispatch(
            loop,
            "run_command",
            command='sqlite3 "$TMPDIR/state.db" "CREATE TABLE state(value); '
            "INSERT INTO state VALUES ('literal /tmp/row');\"",
            read_only=True,
        )
        read = dispatch(loop, "read_text_file", path="$TMPDIR/report.txt")
        assert read["content"] == "draft"
        sessions.add_user_message("revise the report")
        dispatch(loop, "write_text_file", path="$TMPDIR/revised.txt", content="revised")
        result = dispatch(
            loop,
            "run_command",
            command='cat "$TMPDIR/report.txt"; cat "$TMPDIR/revised.txt"',
            cwd="$WORKSPACE_ROOT",
            read_only=True,
        )
        assert result["stdout"]["content"] == "draftrevised"
        loop.command_manager.call("sessions", "new")
        assert loop.permission_manager.temporary_directory != directory
        dispatch(
            loop,
            "run_command",
            command='test ! -e "$TMPDIR/report.txt"',
            cwd="$WORKSPACE_ROOT",
            read_only=True,
        )
        for forbidden in (database.parent, database, directory):
            denied = json.loads(
                loop.tool_registry.call(
                    "run_command",
                    json.dumps({"command": "ls " + shlex.quote(str(forbidden)), "read_only": True}),
                    interaction=interaction,
                    instructions_manager=loop.instructions_manager,
                )
            )
            assert not denied["ok"], denied
            assert "draft" not in denied["problem"]["metadata"]["stdout"]["content"]
        second_directory = loop.permission_manager.temporary_directory
        escape = second_directory / "escape"
        escape.symlink_to(directory / "report.txt")
        denied = json.loads(
            loop.tool_registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": 'cat "$TMPDIR/escape"',
                        "cwd": "$WORKSPACE_ROOT",
                        "read_only": True,
                    }
                ),
                interaction=interaction,
                instructions_manager=loop.instructions_manager,
            )
        )
        assert denied["ok"] is False, denied
        assert "draft" not in json.dumps(denied)
        escape.unlink()
        escape.hardlink_to(directory / "report.txt")
        denied = json.loads(
            loop.tool_registry.call(
                "run_command",
                json.dumps({"command": ":", "cwd": "$WORKSPACE_ROOT", "read_only": True}),
                interaction=interaction,
                instructions_manager=loop.instructions_manager,
            )
        )
        assert denied["problem"]["code"] == "sandbox.invalid", denied
        assert (directory / "report.txt").read_text() == "draft"
        escape.unlink()
        loop.permission_manager.close()
        loop.command_manager.close()
        loop, _ = compose(identifier)
        assert loop.permission_manager.temporary_directory == directory
        result = dispatch(
            loop,
            "run_command",
            command='cat "$TMPDIR/report.txt"; cat "$TMPDIR/revised.txt"',
            cwd="$WORKSPACE_ROOT",
            read_only=True,
        )
        assert result["stdout"]["content"] == "draftrevised"
        state = dispatch(
            loop,
            "run_command",
            command='sqlite3 "$TMPDIR/state.db" "SELECT value FROM state;"',
            read_only=True,
        )
        assert state["stdout"]["content"].strip() == "literal /tmp/row"
        assert directory.is_dir()
        host.run_host_command.assert_not_called()
    finally:
        for loop in reversed(runtimes):
            loop.command_manager.close()
            loop.permission_manager.close()


@pytest.mark.parametrize("native_workspace", ["work space é"], indirect=True)
def test_session_paths_round_trip_through_scripts_subprocesses_and_file_tools(
    backend, native_workspace, monkeypatch
):
    """Fresh data paths stay identical across process boundaries while outside data is denied."""
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + ":/usr/bin:/bin")
    database = native_workspace.parent / "Library" / "Application Support" / "state.db"
    sessions = SessionManager(
        session_store=SQLiteSessionStore(database, workspace_id="paths"), workspace_id="paths"
    )
    permissions = PermissionManager(native_workspace)
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host launch")
    interaction = Mock()
    interaction.prompt.side_effect = lambda message, **kwargs: (
        ApprovalChoice.ONCE
        if message == "Proceed?"
        else "deny"
        if "without the OS sandbox" in message
        else "approve"
    )
    registry = ToolRegistry(
        [run_command, read_text_file, write_text_file],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    loop = Loop.create_default(
        backend=SimpleNamespace(default_model="test", get_context_window=lambda _: None),
        working_directory=native_workspace,
        session_manager=sessions,
        interaction=interaction,
        permission_manager=permissions,
        tool_registry=registry,
    )

    def call(name, **arguments):
        """Use public dispatch with the session's canonical path prefixes."""
        return json.loads(
            registry.call(
                name,
                json.dumps(arguments),
                interaction=interaction,
                instructions_manager=loop.instructions_manager,
            )
        )

    def command(source, **arguments):
        """Require successful native execution without permitting a host recovery."""
        value = call("run_command", command=source, cwd="$WORKSPACE_ROOT", **arguments)
        assert value["ok"], json.dumps(value)
        assert value["result"]["boundary"] == "sandbox"
        return value["result"]["stdout"]["content"]

    try:
        scratch = permissions.temporary_directory
        assert sessions.working_directory_allows_private_names()
        rendered = loop.instructions_manager.virtual_paths
        assert rendered.display(str(native_workspace)) == "$WORKSPACE_ROOT"
        assert rendered.display(str(scratch)) == "$TMPDIR"
        assert call("write_text_file", path="$WORKSPACE_ROOT/input.txt", content="workspace")["ok"]
        assert command('cat "$WORKSPACE_ROOT/input.txt"') == "workspace"
        assert (native_workspace / "input.txt").read_text() == "workspace"
        output = command(
            'printf command > "$TMPDIR/command.txt"; '
            'printf "%s\\n" "$WORKSPACE_ROOT" "$TMPDIR" "$TMP" "$TEMP"'
        )
        assert output.splitlines() == [str(native_workspace), *([str(scratch)] * 3)]
        read = call("read_text_file", path="$TMPDIR/command.txt")
        assert read["result"]["path"] == "$TMPDIR/command.txt"
        assert read["result"]["content"] == "command"
        shell = (
            'set -eu\ncd "$WORKSPACE_ROOT"\n'
            'test "$(cat "$TMPDIR/command.txt")" = command\n'
            'printf script > "$TMPDIR/script.txt"\n'
            'mkdir "$TMPDIR/.loop" "$TMPDIR/Library"\n'
        )
        assert call("write_text_file", path="$TMPDIR/script.sh", content=shell)["ok"]
        assert (scratch / "script.sh").read_text() == shell
        command('/bin/sh "$TMPDIR/script.sh"')
        assert call("write_text_file", path="$TMPDIR/.loop/data", content="tool")["ok"]
        assert call("read_text_file", path="$TMPDIR/.loop/data")["result"]["content"] == "tool"
        program = (
            "import json, os, pathlib, tempfile, subprocess\n"
            "s = pathlib.Path(os.environ['TMPDIR'])\n"
            "w = pathlib.Path(os.environ['WORKSPACE_ROOT'])\n"
            "assert pathlib.Path.cwd() == w\n"
            "assert tempfile.gettempdir() == str(s)\n"
            "assert subprocess.check_output(['/bin/cat', str(s/'script.txt')]) == b'script'\n"
            "assert subprocess.check_output(['/bin/cat', str(w/'input.txt')]) == b'workspace'\n"
            "assert subprocess.check_output(['/usr/bin/printf', '%s', '$TMPDIR/script.txt']) "
            "== b'$TMPDIR/script.txt'\n"
            "subprocess.run(['/bin/sh', '-c', 'test -f \"$TMPDIR/.loop/data\"'], check=True)\n"
            "(s/'paths.json').write_text(json.dumps(dict(workspace=str(w), scratch=str(s))))\n"
            "with tempfile.NamedTemporaryFile(delete=False) as f:\n"
            " f.write(b'subprocess'); print(f.name)\n"
        )
        assert call("write_text_file", path="$TMPDIR/probe.py", content=program)["ok"]
        filename = command('python -B -S "$TMPDIR/probe.py"').strip()
        assert Path(filename).parent == scratch
        physical_read = call("read_text_file", path=filename)
        assert physical_read["result"]["path"] == rendered.display(filename)
        assert physical_read["result"]["content"] == "subprocess"
        expected = {"workspace": str(native_workspace), "scratch": str(scratch)}
        assert json.loads((scratch / "paths.json").read_text()) == expected
        evidence = call("read_text_file", path="$TMPDIR/paths.json")["result"]["content"]
        assert json.loads(evidence) == expected
        assert command('cat "$TMPDIR/paths.json"') == evidence
        physical = str(native_workspace)
        variants = json.dumps(
            {
                "workspace": physical,
                "sibling": physical + "-extra/file",
                "url": "https://example.test" + physical + "/file",
                "quoted": shlex.quote(str(scratch)),
                "unicode": "é ☃",
                "encoded": base64.b64encode(physical.encode()).decode(),
            },
            ensure_ascii=True,
        )
        assert call("write_text_file", path="$TMPDIR/variants.json", content=variants)["ok"]
        read_variants = call("read_text_file", path="$TMPDIR/variants.json")["result"]
        assert read_variants["path"] == "$TMPDIR/variants.json"
        assert read_variants["content"] == variants
        assert command('cat "$TMPDIR/variants.json"', read_only=True) == variants
        lines = "one\rtwo\r\nthree\n"
        assert call("write_text_file", path="$TMPDIR/lines.txt", content=lines)["ok"]
        assert call("read_text_file", path="$TMPDIR/lines.txt")["result"]["content"] == lines
        assert command('cat "$TMPDIR/lines.txt"', read_only=True) == lines
        failed_lines = call(
            "run_command", command='cat "$TMPDIR/lines.txt"; exit 7', read_only=True
        )
        assert failed_lines["problem"]["metadata"]["stdout"]["content"] == lines
        failure = call("run_command", command='cat "$TMPDIR/variants.json"; exit 7', read_only=True)
        assert failure["problem"]["code"] == "process.nonzero_exit"
        assert failure["problem"]["metadata"]["stdout"]["content"] == variants
        command(
            'printf data > "$TMPDIR/.loop/library"; mv "$TMPDIR/.loop/library" "$TMPDIR/renamed"; '
            'rm "$TMPDIR/renamed"',
            read_only=True,
        )
        for target in (native_workspace / ".gitignore", native_workspace / ".ssh" / "new"):
            denied = call("run_command", command="printf bad > " + shlex.quote(str(target)))
            assert not denied["ok"], denied
            assert not target.exists()
        sessions.new_session()
        current = permissions.temporary_directory
        assert current != scratch
        (current / "escape").symlink_to(scratch, target_is_directory=True)
        for source in (
            'cat "$TMPDIR/escape/command.txt"',
            'printf bad > "$TMPDIR/escape/command.txt"',
            "cat " + shlex.quote(str(scratch / "command.txt")),
            'cat "$TMPDIR/../../"*/files/command.txt',
        ):
            denied = call("run_command", command=source, read_only=True)
            assert not denied["ok"], denied
            assert "command" not in denied["problem"]["metadata"]["stdout"]["content"]
        assert (scratch / "command.txt").read_text() == "command"
        assert not call("read_text_file", path="$TMPDIR/escape/command.txt")["ok"]
        assert not call("read_text_file", path=str(current / "escape" / "command.txt"))["ok"]
        assert not call("read_text_file", path=str(scratch / "command.txt"))["ok"]
        host.run_host_command.assert_not_called()
    finally:
        loop.command_manager.close()
        permissions.close()
