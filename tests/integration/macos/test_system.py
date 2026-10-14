"""Verify public command dispatch, destructive review and native execution boundaries."""

import json
import shlex
import shutil
from unittest.mock import Mock

import pytest

from loop import PermissionManager, ToolRegistry
from loop.constants import MAX_OUTPUT_CHARS
from loop.execution import CommandExecutionService
from loop.tools.system import run_command

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.macos]


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


def test_virtual_workspace_paths_remain_model_safe(public_command, native_workspace):
    """Public shell paths and output expose virtual workspace identities rather than host paths."""
    result = public_command.call("cat /workspace/ordinary; printf '%s' /workspace", read_only=True)
    assert result["ok"] is True
    assert "ordinary-data" in result["result"]["stdout"]["content"]
    assert "/workspace" in result["result"]["stdout"]["content"]
    assert str(native_workspace) not in json.dumps(result)
    public_command.host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        "cat /workspace/external-link",
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
        root, virtual, reads = native_workspace, "/workspace", []
    elif role == "temporary":
        root, virtual, reads = public_command.scratch, "/tmp", []
    elif role == "skill":
        public_command.instructions.activate_skill("fixture")
        root, virtual, reads = public_command.skill_root, "/skills/fixture", []
    else:
        root = native_workspace.parent / "extra"
        root.mkdir()
        virtual, reads = str(root), [str(root)]
    (root / "neighbor").write_text("ordinary-readable")
    for prefix in ("", "nested/"):
        directory = root / prefix / component
        directory.mkdir(parents=True, exist_ok=True)
        existing = directory / "credential"
        existing.write_text("private-root-canary")
        created = directory / "new-entry"
        for spelling in dict.fromkeys((str(existing), f"{virtual}/{prefix}{component}/credential")):
            result = public_command.call(
                f": {shlex.quote(virtual)}; cat {shlex.quote(spelling)}; "
                f"printf changed > {shlex.quote(spelling)}; "
                f"printf changed > {shlex.quote(spelling.rsplit('/', 1)[0] + '/new-entry')}",
                read_only=read_only,
                read_roots=reads,
            )
            assert result["ok"] is False, result
            assert "private-root-canary" not in json.dumps(result)
            assert existing.read_text() == "private-root-canary"
            assert not created.exists()
    ordinary = public_command.call(
        f"cat {shlex.quote(virtual + '/neighbor')}", read_only=read_only, read_roots=reads
    )
    assert ordinary["result"]["stdout"]["content"] == "ordinary-readable"
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

    result = public_command.call("cat /skills/fixture/public-alias", read_only=True)

    assert result["problem"]["code"] == "sandbox.invalid", result
    assert "private-skill-canary" not in json.dumps(result)
    assert source.read_text() == "private-skill-canary"
    public_command.host.run_host_command.assert_not_called()


def test_public_private_scratch_keeps_owned_fixture_paths_writable(public_command):
    """Fresh command scratch supports synthetic private-name fixtures without host-data access."""
    for component in (".ssh", ".AWS", ".config", ".loop", "Library"):
        result = public_command.call(
            f'mkdir "$TMPDIR/nested"; mkdir "$TMPDIR/nested/{component}"; '
            f'printf fixture > "$TMPDIR/nested/{component}/key"; '
            f'cat "$TMPDIR/nested/{component}/key"',
            read_only=True,
        )
        assert result["result"]["stdout"]["content"] == "fixture", result
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


@pytest.mark.parametrize(
    "source",
    [
        "printf text >/dev/null; printf done",
        "printf text >|/dev/null; printf done",
        "printf text 2>&-; printf done",
        "printf text >/tmp/report; cat /tmp/report",
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
    allowed = public_command.call("cat /workspace/ordinary", read_only=True)
    assert allowed["ok"] and allowed["result"]["stdout"]["content"] == "ordinary-data"
    targets = ["AGENTS.md", 'POLICY"name.md', "POLICY\nname.md"]
    for name in targets:
        path = native_workspace / name
        path.write_text("preserve-instructions")
        denied = public_command.call(
            "eval " + shlex.quote("printf changed > " + shlex.quote("/workspace/" + name))
        )
        assert not denied["ok"]
        assert "host_offer" not in denied["problem"]["metadata"]
        assert path.read_text() == "preserve-instructions"
    for source in (
        "cat /workspace/.ssh/id_ed25519",
        "cat /workspace/external-link",
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
