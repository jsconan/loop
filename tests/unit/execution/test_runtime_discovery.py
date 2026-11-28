"""Verify bounded discovery of selected shell and build entry points."""

import os
from pathlib import Path

import pytest

from loop.execution import runtime_discovery
from loop.execution.runtime_discovery import (
    discover_makefile_commands,
    discover_scratch_shell_source,
    selected_workspace_executable,
    timeout_child_source,
)
from loop.permissions import CommandInspection


def _allow_read(_path):
    """Allow synthetic discovery fixtures to be read by the planner."""
    return True


@pytest.fixture
def project(tmp_path):
    """Own a workspace for literal build selections."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.mark.parametrize(
    ("source", "selected"),
    [
        ('"$WORKSPACE_ROOT"/.venv/bin/pytest', True),
        ('"${WORKSPACE_ROOT}"/.venv/bin/pytest', True),
        ('"$WORKSPACE_ROOT/.venv/bin/pytest"', True),
        ('"${WORKSPACE_ROOT}/.venv/bin/pytest"', True),
        ("$WORKSPACE_ROOT/.venv/bin/pytest", True),
        ("${WORKSPACE_ROOT}/.venv/bin/pytest", True),
        ('PYTHONDONTWRITEBYTECODE=1 "$WORKSPACE_ROOT"/.venv/bin/pytest', True),
        ("'$WORKSPACE_ROOT'/.venv/bin/pytest", False),
        ('WORKSPACE_ROOT=/tmp "$WORKSPACE_ROOT"/.venv/bin/pytest', False),
        ('IFS=: "$WORKSPACE_ROOT"/.venv/bin/pytest', False),
        ('"$WORKSPACE_ROOT"/../outside', False),
        ('"$WORKSPACE_ROOT"/$TOOL', False),
        ('"$WORKSPACE_ROOT"/".venv"/bin/pytest', False),
        ('"$WORKSPACE_ROOT/.venv/bin/pytest"extra', False),
        ('"$WORKSPACE_ROOT/.venv/bin/$TOOL"', False),
        ('"$WORKSPACE_ROOT/../outside"', False),
        ("$OTHER/.venv/bin/pytest", False),
        ('"$WORKSPACE_ROOT"/', False),
        ('echo "$WORKSPACE_ROOT"/.venv/bin/pytest', False),
    ],
)
def test_selected_workspace_executable_requires_bound_literal_suffix(source, selected):
    """Only a direct bound workspace spelling resolves for interpreter discovery."""
    workspace = Path("/workspace")
    analysis = CommandInspection().analyze(source)
    expected = str(workspace / ".venv/bin/pytest") if selected else None
    assert selected_workspace_executable(source, analysis, workspace) == expected


@pytest.mark.parametrize("quoted", [False, True, "whole"])
def test_selected_workspace_executable_respects_unquoted_field_splitting(tmp_path, quoted):
    """An unquoted workspace expansion cannot select a path with shell splitting."""
    workspace = tmp_path / "space path"
    source = (
        '"$WORKSPACE_ROOT/.venv/bin/pytest"'
        if quoted == "whole"
        else '"$WORKSPACE_ROOT"/.venv/bin/pytest'
        if quoted
        else "$WORKSPACE_ROOT/.venv/bin/pytest"
    )
    expected = str(workspace / ".venv/bin/pytest") if quoted else None
    assert (
        selected_workspace_executable(source, CommandInspection().analyze(source), workspace)
        == expected
    )


@pytest.mark.parametrize("source", ["FLAG=1", "'unterminated"])
def test_selected_workspace_executable_rejects_stale_or_invalid_source(tmp_path, source):
    """A changed or malformed source cannot reuse an earlier executable analysis."""
    selected = '"$WORKSPACE_ROOT"/.venv/bin/pytest'
    analysis = CommandInspection().analyze(selected)
    assert selected_workspace_executable(source, analysis, tmp_path) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("timeout 180 uv run pytest | tail -20", "uv run pytest | tail -20"),
        ("gtimeout 1.5m make test 2>&1", "make test 2>&1"),
        ("'timeout' '2s' uv run pytest", "uv run pytest"),
        ("timeout --preserve-status 180 uv run pytest", None),
        ("timeout -- 180 uv run pytest", None),
        ("timeout $DEADLINE uv run pytest", None),
        ("timeout 1 PATH=/tmp uv run pytest", None),
        ("timeout 1 $COMMAND run pytest", None),
        ("timeout 1 /tmp/uv run pytest", None),
        ("timeout 1 ; uv run pytest", None),
        ("timeout 1", None),
        ("'unterminated", None),
    ],
)
def test_timeout_child_requires_a_simple_literal_invocation(source, expected):
    """Only an unambiguous timeout child becomes an advisory approval candidate."""
    assert timeout_child_source(source) == expected


@pytest.mark.parametrize("spelling", ["$TMPDIR/job.sh", '"$TMPDIR/job.sh"', '"${TMPDIR}/job.sh"'])
def test_scratch_script_discovers_its_original_shell_source(tmp_path, spelling):
    """A selected session shell script supplies literal commands without a host retry."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    content = "#!/bin/sh\nfixture-tool run\n"
    (scratch / "job.sh").write_text(content)
    assert (
        discover_scratch_shell_source(
            f"sh {spelling}",
            scratch,
            tmp_path,
            os.defpath,
            lambda _: True,
            may_read=_allow_read,
        )
        == "fixture-tool run\n"
    )


@pytest.mark.parametrize(
    "source",
    [
        "",
        "sh",
        "sh | cat",
        "echo $TMPDIR/job.sh",
        "sh '$TMPDIR/job.sh'",
        "sh $TMPDIR/../job.sh",
        "sh $TMPDIR/",
        "sh $TMPDIR/$OTHER",
        "sh $TMPDIR/*.sh",
        "sh $TMPDIR/.ssh/job.sh",
        "sh $OTHER/job.sh",
        "sh -c $TMPDIR/job.sh",
        "SHELL=sh sh $TMPDIR/job.sh",
    ],
)
def test_scratch_script_rejects_ambiguous_or_protected_selection(tmp_path, source):
    """Shell expansion, traversal, private paths and ambiguous interpreters grant nothing."""
    assert (
        discover_scratch_shell_source(
            source, tmp_path, tmp_path, os.defpath, lambda _: True, may_read=_allow_read
        )
        is None
    )


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "symlink",
        "large",
        "not-script",
        "unterminated",
        "different-shell",
        "existing-different-shell",
        "shebang-options",
        "invalid-utf8",
    ],
)
def test_scratch_script_requires_a_stable_matching_interpreter(tmp_path, kind):
    """Only a bounded regular script with the selected interpreter supplies candidates."""
    script = tmp_path / "job.sh"
    if kind == "symlink":
        target = tmp_path / "other"
        target.write_text("#!/bin/sh\ntrue\n")
        script.symlink_to(target)
    elif kind != "missing":
        content = {
            "large": b"x" * (runtime_discovery._MAX_METADATA_BYTES + 1),
            "not-script": b"echo test\n",
            "unterminated": b"#!/bin/sh",
            "different-shell": b"#!/bin/false\ntrue\n",
            "existing-different-shell": b"#!/usr/bin/env\ntrue\n",
            "shebang-options": b"#!/bin/sh -e\ntrue\n",
            "invalid-utf8": b"#!/bin/sh\n\xff\n",
        }[kind]
        script.write_bytes(content)
    assert (
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"',
            tmp_path,
            tmp_path,
            os.defpath,
            lambda _: True,
            may_read=_allow_read,
        )
        is None
    )


@pytest.mark.parametrize(
    "body",
    [
        "cat <<EOF\nfixture-tool\nEOF\n",
        "f() { fixture-tool; }\n",
        "echo $(fixture-tool)\n",
        "echo `fixture-tool`\n",
        "if true; then fixture-tool; fi\n",
    ],
)
def test_scratch_script_ignores_ambiguous_shell_constructs(tmp_path, body):
    """Data, substitutions and control structures do not authorize extra tools."""
    (tmp_path / "job.sh").write_text("#!/bin/sh\n" + body)
    assert (
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"',
            tmp_path,
            tmp_path,
            os.defpath,
            lambda _: True,
            may_read=_allow_read,
        )
        is None
    )


def test_scratch_script_requires_trusted_shell(tmp_path):
    """A selected untrusted shell cannot authorize tools from a purported script."""
    (tmp_path / "job.sh").write_text("#!/bin/sh\nfixture-tool\n")
    assert (
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"',
            tmp_path,
            tmp_path,
            os.defpath,
            lambda _: False,
            may_read=_allow_read,
        )
        is None
    )


def test_scratch_script_trust_errors_are_advisory(tmp_path):
    """A shell that cannot be verified supplies no automatic candidates."""

    def unavailable(_path):
        raise OSError("unavailable")

    assert (
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"',
            tmp_path,
            tmp_path,
            os.defpath,
            unavailable,
            may_read=_allow_read,
        )
        is None
    )


@pytest.mark.parametrize("exception", [PermissionError, ValueError])
def test_scratch_script_audit_veto_propagates(tmp_path, monkeypatch, exception):
    """An audit veto during script access cannot be silently bypassed."""

    def veto(*_args, **_kwargs):
        raise exception("veto")

    monkeypatch.setattr(runtime_discovery, "_read_workspace_file", veto)
    with pytest.raises(exception, match="veto"):
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"',
            tmp_path,
            tmp_path,
            os.defpath,
            lambda _: True,
            may_read=_allow_read,
        )


def test_scratch_script_shell_audit_veto_propagates(tmp_path):
    """A veto while authenticating the selected shell also prevents execution."""

    def veto(_path):
        raise PermissionError("shell veto")

    with pytest.raises(PermissionError, match="shell veto"):
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"', tmp_path, tmp_path, os.defpath, veto, may_read=_allow_read
        )


def test_makefile_discovers_only_selected_literal_phony_recipes(project, monkeypatch):
    """A simple selected build and its prerequisites suggest their literal commands."""
    workspace = project
    (workspace / "Makefile").write_text(
        ".PHONY: test check unused\n"
        "test: check\n\tPYTHONDONTWRITEBYTECODE=1 fixture-tool run tests\n"
        "check:\n\t  @fixture-tool check\n"
        "unused:\n\tother-tool run\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime_discovery, "_system_make_selected", lambda name, *_: Path(name).name == "make"
    )

    assert discover_makefile_commands(
        "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
    ) == (
        "fixture-tool check",
        "PYTHONDONTWRITEBYTECODE=1 fixture-tool run tests",
    )
    assert discover_makefile_commands(
        "/bin/make test", workspace, workspace, "/usr/bin", may_read=_allow_read
    ) == (
        "fixture-tool check",
        "PYTHONDONTWRITEBYTECODE=1 fixture-tool run tests",
    )


@pytest.mark.parametrize("withdraw_after_selection", [False, True])
def test_unreadable_makefile_cannot_supply_candidate_commands(
    project, monkeypatch, withdraw_after_selection
):
    """A denied recipe cannot contribute commands before or after descriptor binding."""
    makefile = project / "Makefile"
    makefile.write_text(".PHONY: test\ntest:\n\tfixture-tool check\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_args: True)
    decisions = iter((withdraw_after_selection, False))

    def may_read(path):
        """Return the current policy decision for the selected recipe."""
        assert path == makefile
        return next(decisions)

    assert (
        discover_makefile_commands("make test", project, project, "/usr/bin", may_read=may_read)
        == ()
    )


def test_makefile_discovers_quoted_parentheses_but_not_shell_operators(project, monkeypatch):
    """Quoted interpreter arguments remain literal while shell grouping stays ambiguous."""
    workspace = project
    makefile = workspace / "Makefile"
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_args: True)
    recipe = './.venv/bin/python -c "print(1)"'
    makefile.write_text(f".PHONY: test\ntest:\n\t{recipe}\n")
    assert discover_makefile_commands(
        "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
    ) == (recipe,)
    makefile.write_text(".PHONY: test\ntest:\n\t(fixture-tool)\n")
    assert (
        discover_makefile_commands(
            "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
        )
        == ()
    )


@pytest.mark.parametrize(
    "source",
    [
        'cd "$WORKSPACE_ROOT"; make test 2>&1',
        'cd "${WORKSPACE_ROOT}" && make test',
        "cd . && make test",
        "cd -- .; make test",
        "OTHER=1 make test",
        "make test > /dev/null",
        "make test;",
        "make test;\n",
        'make test 2>&1; echo "EXIT_CODE=$?"',
        "make test && printf finished",
        "make test || printf issue",
        "make test\nprintf 'later command'",
        'cd "$WORKSPACE_ROOT" && make test 2>&1 | tail -200',
        "make test | fixture-filter --quiet | fixture-summary",
    ],
)
def test_makefile_discovery_understands_shell_structure(project, monkeypatch, source):
    """A fixed initial build retains its tool selection in a simple shell command list."""
    workspace = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    monkeypatch.setattr(
        runtime_discovery, "_system_make_selected", lambda name, *_: Path(name).name == "make"
    )
    assert discover_makefile_commands(
        source, workspace, workspace, "/usr/bin", may_read=_allow_read
    ) == ("fixture-tool",)


@pytest.mark.parametrize(
    "source",
    [
        "cd /; make test",
        "cd '$WORKSPACE_ROOT'; make test",
        'cd "$OTHER"; make test',
        "make test $OTHER",
        "PATH=/other make test",
        "MAKEFLAGS=-n make test",
        "GNUMAKEFLAGS=-n make test",
        "MFLAGS=-n make test",
        "MAKEFILES=extra.mk make test",
        "MAKEFILES=$OTHER make test",
        "OTHER=${VAR:?oops} make test",
        'OTHER="$(touch Makefile)" make test',
        "make test > Makefile",
        "make test > $OUTPUT",
        "make test >",
        "make test 2>&",
        "make test <",
        "make test &&",
        "make test |",
        "make test | tail $COUNT",
        "make test | tail -2 > Makefile",
        "make test | tail && fixture-tool",
        "make test | tail |",
        "make test | | tail",
        "make test | if",
        "make test | PATH=/elsewhere tail",
        "make test '|' tail",
        r"make test \| tail",
        "make test; )",
        "make test; if",
        'make test; "$COMMAND"',
    ],
)
def test_makefile_discovery_preserves_ambiguous_shell_selection(project, monkeypatch, source):
    """Unknown working directories, arguments and PATH changes do not infer build tools."""
    workspace = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    assert not discover_makefile_commands(
        source, workspace, workspace, "/usr/bin", may_read=_allow_read
    )


@pytest.mark.parametrize("name", ["MAKEFLAGS", "GNUMAKEFLAGS", "MFLAGS", "MAKEFILES"])
def test_makefile_discovery_ignores_uninherited_host_make_controls(project, monkeypatch, name):
    """Host make settings are not inherited by the sandboxed command."""
    workspace = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    monkeypatch.setenv(name, "configured")
    assert discover_makefile_commands(
        "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
    ) == ("fixture-tool",)
    assert not discover_makefile_commands(
        "make test",
        workspace,
        workspace,
        "/usr/bin",
        environment_names={name},
        may_read=_allow_read,
    )


@pytest.mark.parametrize("source", ["cd -- . &&", "cd . word make test", 'cd . ";" make test'])
def test_workspace_command_source_preserves_incomplete_or_quoted_separators(project, source):
    """Incomplete commands and quoted separators are never treated as no-op prefixes."""
    workspace = project
    assert runtime_discovery.workspace_command_source(source, workspace, workspace) == source


@pytest.mark.parametrize("destination", ["-", '"-"', "-- -", "alias", "./alias"])
def test_workspace_command_source_does_not_guess_special_cd_operands(project, destination):
    """OLDPWD, CDPATH and retargeted links can change a directory-like operand."""
    workspace = project
    (workspace / "-").symlink_to(".", target_is_directory=True)
    (workspace / "alias").symlink_to(".", target_is_directory=True)
    source = f"cd {destination} && make test"
    assert runtime_discovery.workspace_command_source(source, workspace, workspace) == source
    assert not discover_makefile_commands(
        source, workspace, workspace, "/usr/bin", may_read=_allow_read
    )


@pytest.mark.parametrize(
    ("source", "makefile"),
    [
        ("make -s", ".PHONY: -s\n-s:\n\tfixture-tool\n"),
        ("make --version", ".PHONY: --version\n--version:\n\tfixture-tool\n"),
        ("make\ntest", ".PHONY: test\ntest:\n\tfixture-tool\n"),
        ("make test\r", ".PHONY: test\ntest:\n\tfixture-tool\n"),
        ("make test extra", ".PHONY: test\ntest:\n\tfixture-tool\n"),
        ("make $TARGET", ".PHONY: test\ntest:\n\tfixture-tool\n"),
        ("make 'test", ".PHONY: test\ntest:\n\tfixture-tool\n"),
        ("other test", ".PHONY: test\ntest:\n\tfixture-tool\n"),
        ("make test", "test:\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ntest: missing\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ntest: test\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ninclude other.mk\ntest:\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ntest:\n\tfixture-tool $(OTHER)\n"),
        ("make test", ".PHONY: test\ntest:\n\tfixture-tool; other\n"),
        ("make test", ".PHONY: test\ntest:\n\tfixture-tool 'unterminated\n"),
        ("make test", ".PHONY: test\ntest:\n\tfixture-tool\ntest:\n\tother\n"),
        ("make test", "\torphan\n.PHONY: test\ntest:\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ntest:\n\t   \n"),
        ("make test", ".PHONY: test\ntest:\n\t\n"),
        ("make test", ".PHONY: test\ntest: dependency%\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ntest: # comment\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ntest: \\\n dependency\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\n.SUFFIXES:\ntest:\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\nMakefile:\ntest:\n\tfixture-tool\n"),
        ("make test", ".PHONY: test\ninvalid?name:\ntest:\n\tfixture-tool\n"),
        ("make " + "x" * 300, ".PHONY: test\ntest:\n\tfixture-tool\n"),
    ],
)
def test_makefile_discovery_declines_ambiguous_selection(project, monkeypatch, source, makefile):
    """Options, dynamic recipes and ambiguous rule graphs cannot suggest tool authority."""
    workspace = project
    (workspace / "Makefile").write_text(makefile, encoding="utf-8")
    monkeypatch.setattr(
        runtime_discovery, "_system_make_selected", lambda name, *_: Path(name).name == "make"
    )

    assert (
        discover_makefile_commands(source, workspace, workspace, "/usr/bin", may_read=_allow_read)
        == ()
    )


def test_makefile_discovery_refuses_unbounded_selected_recipes(project, monkeypatch):
    """A workspace cannot force unbounded executable lookups during preparation."""
    workspace = project
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n"
        + "\tfixture-tool\n" * (runtime_discovery.constants.MAX_COMMAND_PATH_COUNT + 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)

    assert (
        discover_makefile_commands(
            "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
        )
        == ()
    )


@pytest.mark.parametrize("case", ["selected", "absent", "other", "error", "outside", "missing"])
def test_makefile_discovery_binds_system_make_and_workspace(project, monkeypatch, case):
    """Only a current system make selection and a workspace file produce candidates."""
    workspace = project
    if case != "missing":
        (workspace / "Makefile").write_text(
            ".PHONY: test\ntest:\n\tfixture-tool\n", encoding="utf-8"
        )
    selected = Path("/usr/bin/make").resolve()
    resolver = {
        "absent": None,
        "other": (workspace / "make", workspace / "make"),
    }.get(case, (selected, selected))
    if case == "error":

        def unavailable(*_args):
            """Represent an unusable executable selection."""
            raise ValueError("unavailable")

        monkeypatch.setattr(runtime_discovery, "resolve_selected_executable", unavailable)
    else:
        monkeypatch.setattr(
            runtime_discovery, "resolve_selected_executable", lambda *_args: resolver
        )
    cwd = workspace.parent if case == "outside" else workspace

    result = discover_makefile_commands(
        "/usr/bin/make test", workspace, cwd, "/usr/bin", may_read=_allow_read
    )

    assert result == (("fixture-tool",) if case == "selected" else ())


def test_makefile_discovery_rejects_higher_priority_makefile(project, monkeypatch):
    """A different makefile selected by make cannot authorize suggestions from Makefile."""
    workspace = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    (workspace / "GNUmakefile").write_text("test:\n\tother-tool\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)

    assert (
        discover_makefile_commands(
            "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
        )
        == ()
    )


@pytest.mark.parametrize("fault", ["inaccessible", "growing", "changed"])
def test_makefile_discovery_rejects_unstable_inputs(project, monkeypatch, fault):
    """An inaccessible override or changing recipe cannot authorize indirect tools."""
    workspace = project
    makefile = workspace / "Makefile"
    makefile.write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    if fault == "inaccessible":
        original_stat = Path.lstat

        def stat_path(path):
            """Deny probing the higher-priority recipe."""
            if path == workspace / "GNUmakefile":
                raise PermissionError("fixture access denied")
            return original_stat(path)

        monkeypatch.setattr(Path, "lstat", stat_path)
    else:
        original_read = os.read

        def read_file(descriptor, limit):
            """Simulate a recipe that no longer matches its original metadata."""
            if fault == "growing":
                return b"x" * limit
            content = original_read(descriptor, limit)
            makefile.write_bytes(content + b"\n")
            return content

        monkeypatch.setattr(runtime_discovery.os, "read", read_file)
    assert not discover_makefile_commands(
        "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
    )


def test_makefile_discovery_refuses_unverifiable_alternate(project, monkeypatch):
    """An ambiguous higher-priority spelling cannot establish the selected makefile."""
    workspace = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    original_exists = runtime_discovery._exists
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    monkeypatch.setattr(
        runtime_discovery,
        "_exists",
        lambda item: True if item.name == "makefile" else original_exists(item),
    )

    def unavailable(*_args):
        """Represent a failed identity check."""
        raise OSError("unavailable")

    monkeypatch.setattr(Path, "samefile", unavailable)
    assert (
        discover_makefile_commands(
            "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
        )
        == ()
    )


def test_makefile_discovery_refuses_shared_inode_and_propagates_open_veto(project, monkeypatch):
    """Shared makefiles are ignored and an explicit open veto prevents preparation."""
    workspace = project
    makefile = workspace / "Makefile"
    makefile.write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    hardlink = workspace / "shared"
    os.link(makefile, hardlink)
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    assert (
        discover_makefile_commands(
            "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
        )
        == ()
    )
    hardlink.unlink()

    def veto(*_args):
        """Raise an access veto while opening the selected makefile."""
        raise PermissionError("makefile veto")

    monkeypatch.setattr(runtime_discovery, "open_selected_file", veto)
    with pytest.raises(PermissionError, match="makefile veto"):
        discover_makefile_commands(
            "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
        )


@pytest.mark.parametrize("component", [".ssh", ".AWS", "Library"])
def test_makefile_discovery_does_not_read_private_workspace_directories(
    project, monkeypatch, component
):
    """A protected cwd, including a resolved alias, cannot suggest executable authority."""
    workspace = project
    private = workspace / component
    private.mkdir()
    (private / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    alias = workspace / "alias"
    alias.symlink_to(private, target_is_directory=True)
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)

    for cwd in (private, alias.resolve()):
        assert (
            discover_makefile_commands(
                "make test", workspace, cwd, "/usr/bin", may_read=_allow_read
            )
            == ()
        )


def test_makefile_discovery_rejects_non_utf8_and_supports_shared_dependencies(project, monkeypatch):
    """Invalid text is ignored and a shared prerequisite is suggested only once."""
    workspace = project
    makefile = workspace / "Makefile"
    makefile.write_bytes(b"\xff")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    assert (
        discover_makefile_commands(
            "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
        )
        == ()
    )
    makefile.write_text(
        "# comment\n\n.PHONY: test one two shared\n"
        "test: one two\n\tfixture-tool run\n"
        "one: shared\n\ttask-one\n"
        "two: shared\n\ttask-two\n"
        "shared:\n\ttask-shared\n"
    )

    assert discover_makefile_commands(
        "make test", workspace, workspace, "/usr/bin", may_read=_allow_read
    ) == (
        "task-shared",
        "task-one",
        "task-two",
        "fixture-tool run",
    )


@pytest.mark.parametrize(
    "case", ["eligible", "denied", "missing", "many-roots", "many-links", "many-files"]
)
def test_workspace_tool_links_are_bounded_and_stay_within_project(project, monkeypatch, case):
    """Only inventory real, approved bin directories, discarding partial results on overflow."""
    folder = project / "tools" / "bin"
    folder.mkdir(parents=True)
    (folder / "plain").touch()
    (folder / "linked").symlink_to("/unavailable")
    (project / "alias").symlink_to(folder.parent, target_is_directory=True)
    if case == "many-roots":
        monkeypatch.setattr(runtime_discovery.constants, "MAX_COMMAND_PATH_COUNT", 1)
    if case == "many-links":
        monkeypatch.setattr(runtime_discovery.constants, "MAX_COMMAND_PATH_COUNT", 2)
        for index in range(3):
            (folder / str(index)).symlink_to("/missing")
    if case == "many-files":
        for index in range(1025):
            (folder / str(index)).touch()
    if case == "missing":
        project = project / "missing"
    result = runtime_discovery.workspace_tool_links(
        project, lambda _path, _directory: case != "denied"
    )
    assert result == ((str(folder / "linked"),) if case == "eligible" else ())


@pytest.mark.parametrize(
    "case",
    [
        "script",
        "link",
        "directory",
        "unreadable",
        "not-executable",
        "changed",
        "binary",
        "relative",
    ],
)
def test_workspace_tool_links_selects_only_the_named_launcher_interpreter(
    project, monkeypatch, case
):
    """Bound launcher evidence excludes unreadable, indirect and unrelated runtime links."""
    directory = project / "tools" / "bin"
    directory.mkdir(parents=True)
    interpreter = directory / "runtime"
    interpreter.symlink_to("/external-runtime")
    (directory / "unrelated").symlink_to("/another-runtime")
    launcher = directory / "verify"
    if case == "directory":
        launcher.mkdir()
    elif case == "link":
        launcher.symlink_to("/external-launcher")
    else:
        body = {
            "relative": "#!runtime\n",
            "binary": "other format",
        }.get(case, f"#!{interpreter} -I\n")
        launcher.write_text(body)
        launcher.chmod(0o600 if case == "not-executable" else 0o700)

    if case == "changed":
        real_open = runtime_discovery.open_selected_file

        def changed(root, path, identity):
            """Simulate the exact launcher disappearing while its contents are pinned."""
            if path == launcher:
                raise FileNotFoundError(path)
            return real_open(root, path, identity)

        monkeypatch.setattr(runtime_discovery, "open_selected_file", changed)

    result = runtime_discovery.workspace_tool_links(
        project,
        lambda _path, directory: directory or case != "unreadable",
        launchers={"verify"},
    )
    assert result == {"script": (str(interpreter),), "link": (str(launcher),)}.get(case, ())


@pytest.mark.parametrize(
    ("command", "script", "allowed", "expected"),
    [
        ("./check", b"#!/bin/sh\nproject-tool --check", True, ("project-tool",)),
        (
            'cd "$WORKSPACE_ROOT" && RESULT="$TMPDIR/result" ./check',
            b"#!/bin/sh\nproject-tool",
            True,
            ("project-tool",),
        ),
        ("PATH=/unreviewed ./check", b"#!/bin/sh\nproject-tool", True, ()),
        ("sh ./check", b"#!/usr/bin/env bash\nproject-tool", True, ("project-tool",)),
        (
            'export PATH=/tools; export OUT="$TMPDIR/report"; bash ./check',
            b"#!/bin/sh\nchild",
            True,
            (),
        ),
        ('export OUT="$TMPDIR/report" && sh ./check', b"#!/bin/sh\nchild", True, ("child",)),
        ("export A=1\nsh ./check", b"#!/bin/sh\nchild", True, ("child",)),
        ("export; sh ./check", b"", True, None),
        ("export A=1", b"", True, None),
        ("export A=1 | sh ./check", b"", True, None),
        ("sh", b"", True, None),
        ("sh ;", b"", True, None),
        ('sh "$script"', b"", True, None),
        ("", b"", True, None),
        ("$wrapper", b"", True, None),
        (";", b"", True, None),
        ("command", b"", True, None),
        ("../check", b"", True, None),
        ("./.ssh/key", b"", True, None),
        ("./check", b"", False, None),
        ("./check", b"text data", True, None),
        ("./check", b"#!\n", True, None),
        ("./check", b"#!/bin/other\n", True, None),
        ("./check", b"#!/bin/sh\n\xff", True, None),
        ("./absent", b"", True, None),
    ],
)
def test_project_launcher_only_suggests_programs_from_readable_shell_source(
    project, command, script, allowed, expected
):
    """Use no-follow source selection and never execute shell text during tool discovery."""
    (project / "check").write_bytes(script)
    assert (
        runtime_discovery.workspace_shell_programs(command, project, project, lambda _path: allowed)
        == expected
    )


def test_linked_installations_skip_changed_or_shared_executables(project, monkeypatch):
    """A stale or multiply named external executable cannot propose an installation scope."""
    folder = project / "tools" / "bin"
    folder.mkdir(parents=True)
    install = project.parent / "installed"
    (install / "bin").mkdir(parents=True)
    (install / "lib").mkdir()
    program = install / "bin" / "program"
    program.write_text("program")
    program.chmod(0o700)
    (program.parent / "second").hardlink_to(program)
    (folder / "launcher").symlink_to(program)
    assert not runtime_discovery.linked_project_installations(
        project, (), lambda _path, _directory: True
    )

    def changed(*_args):
        """Simulate a disappearing executable during selection."""
        raise OSError("tool disappeared")

    monkeypatch.setattr(runtime_discovery, "resolve_selected_executable", changed)
    assert not runtime_discovery.linked_project_installations(
        project, (), lambda _path, _directory: True
    )
