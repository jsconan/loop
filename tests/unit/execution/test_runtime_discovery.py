"""Verify bounded discovery of selected workspace runtime candidates."""

import os
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution import runtime_discovery
from loop.execution.runtime_discovery import (
    RuntimeCandidate,
    discover_makefile_commands,
    discover_scratch_shell_source,
    discover_workspace_interpreters,
    discover_workspace_runtimes,
    selected_workspace_executable,
    timeout_child_source,
)
from loop.permissions import CommandInspection


@pytest.fixture
def project(tmp_path):
    """Own a project, a linked runtime, and an identity-bound selected tool."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = workspace / ".venv"
    commands = environment / "bin"
    commands.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    interpreter = runtime / "engine"
    interpreter.write_text("fixture", encoding="utf-8")
    interpreter.chmod(0o755)
    selected = commands / "python"
    selected.symlink_to(interpreter)
    (environment / "pyvenv.cfg").write_text(
        f"home = {runtime}\nimplementation = CPython\nversion_info = 3.14.2\n", encoding="utf-8"
    )
    (workspace / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0.0.0"\nrequires-python = ">=3.12"\n',
        encoding="utf-8",
    )
    tool = tmp_path / "uv"
    tool.write_text("fixture", encoding="utf-8")
    tool.chmod(0o755)
    details = tool.stat()
    identity = ("uv", tool, tool, details.st_dev, details.st_ino)
    return workspace, selected, interpreter, identity


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
            f"sh {spelling}", scratch, tmp_path, os.defpath, lambda _: True
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
        discover_scratch_shell_source(source, tmp_path, tmp_path, os.defpath, lambda _: True)
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
            'sh "$TMPDIR/job.sh"', tmp_path, tmp_path, os.defpath, lambda _: True
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
            'sh "$TMPDIR/job.sh"', tmp_path, tmp_path, os.defpath, lambda _: True
        )
        is None
    )


def test_scratch_script_requires_trusted_shell(tmp_path):
    """A selected untrusted shell cannot authorize tools from a purported script."""
    (tmp_path / "job.sh").write_text("#!/bin/sh\nfixture-tool\n")
    assert (
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"', tmp_path, tmp_path, os.defpath, lambda _: False
        )
        is None
    )


def test_scratch_script_trust_errors_are_advisory(tmp_path):
    """A shell that cannot be verified supplies no automatic candidates."""

    def unavailable(_path):
        raise OSError("unavailable")

    assert (
        discover_scratch_shell_source(
            'sh "$TMPDIR/job.sh"', tmp_path, tmp_path, os.defpath, unavailable
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
            'sh "$TMPDIR/job.sh"', tmp_path, tmp_path, os.defpath, lambda _: True
        )


def test_scratch_script_shell_audit_veto_propagates(tmp_path):
    """A veto while authenticating the selected shell also prevents execution."""

    def veto(_path):
        raise PermissionError("shell veto")

    with pytest.raises(PermissionError, match="shell veto"):
        discover_scratch_shell_source('sh "$TMPDIR/job.sh"', tmp_path, tmp_path, os.defpath, veto)


def test_makefile_discovers_only_selected_literal_phony_recipes(project, monkeypatch):
    """A simple selected build and its prerequisites suggest their literal commands."""
    workspace, _, _, _ = project
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

    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == (
        "fixture-tool check",
        "PYTHONDONTWRITEBYTECODE=1 fixture-tool run tests",
    )
    assert discover_makefile_commands("/bin/make test", workspace, workspace, "/usr/bin") == (
        "fixture-tool check",
        "PYTHONDONTWRITEBYTECODE=1 fixture-tool run tests",
    )


def test_makefile_discovers_quoted_parentheses_but_not_shell_operators(project, monkeypatch):
    """Quoted interpreter arguments remain literal while shell grouping stays ambiguous."""
    workspace, _, _, _ = project
    makefile = workspace / "Makefile"
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_args: True)
    recipe = './.venv/bin/python -c "print(1)"'
    makefile.write_text(f".PHONY: test\ntest:\n\t{recipe}\n")
    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == (recipe,)
    makefile.write_text(".PHONY: test\ntest:\n\t(fixture-tool)\n")
    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == ()


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
        'cd "$WORKSPACE_ROOT" && make test 2>&1 | tail -200',
        "make test | fixture-filter --quiet | fixture-summary",
    ],
)
def test_makefile_discovery_understands_shell_structure(project, monkeypatch, source):
    """A no-op directory change, assignments and redirections retain make selection."""
    workspace, _, _, _ = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    monkeypatch.setattr(
        runtime_discovery, "_system_make_selected", lambda name, *_: Path(name).name == "make"
    )
    assert discover_makefile_commands(source, workspace, workspace, "/usr/bin") == ("fixture-tool",)


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
        "make test && printf finished",
    ],
)
def test_makefile_discovery_preserves_ambiguous_shell_selection(project, monkeypatch, source):
    """Unknown working directories, arguments and PATH changes do not infer build tools."""
    workspace, _, _, _ = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    assert not discover_makefile_commands(source, workspace, workspace, "/usr/bin")


@pytest.mark.parametrize("name", ["MAKEFLAGS", "GNUMAKEFLAGS", "MFLAGS", "MAKEFILES"])
def test_makefile_discovery_ignores_uninherited_host_make_controls(project, monkeypatch, name):
    """Host make settings are not inherited by the sandboxed command."""
    workspace, _, _, _ = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    monkeypatch.setenv(name, "configured")
    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == (
        "fixture-tool",
    )


@pytest.mark.parametrize("source", ["cd -- . &&", "cd . word make test", 'cd . ";" make test'])
def test_workspace_command_source_preserves_incomplete_or_quoted_separators(project, source):
    """Incomplete commands and quoted separators are never treated as no-op prefixes."""
    workspace, _, _, _ = project
    assert runtime_discovery.workspace_command_source(source, workspace, workspace) == source


@pytest.mark.parametrize("destination", ["-", '"-"', "-- -", "alias", "./alias"])
def test_workspace_command_source_does_not_guess_special_cd_operands(project, destination):
    """OLDPWD, CDPATH and retargeted links can change a directory-like operand."""
    workspace, _, _, _ = project
    (workspace / "-").symlink_to(".", target_is_directory=True)
    (workspace / "alias").symlink_to(".", target_is_directory=True)
    source = f"cd {destination} && make test"
    assert runtime_discovery.workspace_command_source(source, workspace, workspace) == source
    assert not discover_makefile_commands(source, workspace, workspace, "/usr/bin")


@pytest.mark.parametrize("prefix", ["other", "1invalid=value"])
def test_uv_discovery_rejects_source_inconsistent_with_analysis(project, prefix):
    """A mismatched analysis cannot turn an arbitrary prefix into an assignment."""
    workspace, _, _, identity = project
    analysis = CommandInspection().analyze("uv run pytest")
    assert not runtime_discovery._uv_project_runtime(
        f"{prefix} uv run pytest", analysis, workspace, workspace, "/usr/bin", (identity,)
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
    workspace, _, _, _ = project
    (workspace / "Makefile").write_text(makefile, encoding="utf-8")
    monkeypatch.setattr(
        runtime_discovery, "_system_make_selected", lambda name, *_: Path(name).name == "make"
    )

    assert discover_makefile_commands(source, workspace, workspace, "/usr/bin") == ()


def test_makefile_discovery_refuses_unbounded_selected_recipes(project, monkeypatch):
    """A workspace cannot force unbounded executable lookups during preparation."""
    workspace, _, _, _ = project
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n"
        + "\tfixture-tool\n" * (runtime_discovery.constants.MAX_COMMAND_PATH_COUNT + 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)

    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == ()


@pytest.mark.parametrize("case", ["selected", "absent", "other", "error", "outside", "missing"])
def test_makefile_discovery_binds_system_make_and_workspace(project, monkeypatch, case):
    """Only a current system make selection and a workspace file produce candidates."""
    workspace, _, _, _ = project
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

    result = discover_makefile_commands("/usr/bin/make test", workspace, cwd, "/usr/bin")

    assert result == (("fixture-tool",) if case == "selected" else ())


def test_makefile_discovery_rejects_higher_priority_makefile(project, monkeypatch):
    """A different makefile selected by make cannot authorize suggestions from Makefile."""
    workspace, _, _, _ = project
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    (workspace / "GNUmakefile").write_text("test:\n\tother-tool\n")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)

    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == ()


def test_makefile_discovery_refuses_unverifiable_alternate(project, monkeypatch):
    """An ambiguous higher-priority spelling cannot establish the selected makefile."""
    workspace, _, _, _ = project
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
    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == ()


def test_makefile_discovery_refuses_shared_inode_and_propagates_open_veto(project, monkeypatch):
    """Shared makefiles are ignored and an explicit open veto prevents preparation."""
    workspace, _, _, _ = project
    makefile = workspace / "Makefile"
    makefile.write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    hardlink = workspace / "shared"
    os.link(makefile, hardlink)
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == ()
    hardlink.unlink()

    def veto(*_args):
        """Raise an access veto while opening the selected makefile."""
        raise PermissionError("makefile veto")

    monkeypatch.setattr(runtime_discovery, "open_selected_file", veto)
    with pytest.raises(PermissionError, match="makefile veto"):
        discover_makefile_commands("make test", workspace, workspace, "/usr/bin")


@pytest.mark.parametrize("component", [".ssh", ".AWS", "Library"])
def test_makefile_discovery_does_not_read_private_workspace_directories(
    project, monkeypatch, component
):
    """A protected cwd, including a resolved alias, cannot suggest executable authority."""
    workspace, _, _, _ = project
    private = workspace / component
    private.mkdir()
    (private / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    alias = workspace / "alias"
    alias.symlink_to(private, target_is_directory=True)
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)

    for cwd in (private, alias.resolve()):
        assert discover_makefile_commands("make test", workspace, cwd, "/usr/bin") == ()


def test_makefile_discovery_rejects_non_utf8_and_supports_shared_dependencies(project, monkeypatch):
    """Invalid text is ignored and a shared prerequisite is suggested only once."""
    workspace, _, _, _ = project
    makefile = workspace / "Makefile"
    makefile.write_bytes(b"\xff")
    monkeypatch.setattr(runtime_discovery, "_system_make_selected", lambda *_: True)
    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == ()
    makefile.write_text(
        "# comment\n\n.PHONY: test one two shared\n"
        "test: one two\n\tfixture-tool run\n"
        "one: shared\n\ttask-one\n"
        "two: shared\n\ttask-two\n"
        "shared:\n\ttask-shared\n"
    )

    assert discover_makefile_commands("make test", workspace, workspace, "/usr/bin") == (
        "task-shared",
        "task-one",
        "task-two",
        "fixture-tool run",
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("uv run --no-sync pytest", True),
        ("uv --no-cache run --no-sync arbitrary-task", True),
        ("PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/unit", True),
        ('cd "$WORKSPACE_ROOT" && PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest', True),
        ('cd "$WORKSPACE_ROOT" && uv run pytest', True),
        ("uv run --no-sync -- arbitrary-task", True),
        ("uv run --isolated pytest", False),
        ("uv run --python 3.12 pytest", False),
        ("uv --project other run pytest", False),
        ("uv run script.py", False),
        ("uv run ./script.py", False),
        ("uv run $TASK", False),
        ("uv", False),
        ("uv sync", False),
        ("uv run", False),
        ("uv run --no-sync --other pytest", False),
        ("OTHER=1 uv run pytest", True),
        ("ONE=1 TWO='two words' uv run pytest", True),
        ("UV_NO_CACHE=1 uv run pytest", True),
        ("UV_PYTHON=3.12 uv run pytest", False),
        ("UV_PROJECT=other uv run pytest", False),
        ("OTHER=$(command) uv run pytest", False),
        ("'uv' run pytest", True),
        ("PYTHONDONTWRITEBYTECODE=0 uv run pytest", True),
        ('cd "$OTHER" && uv run pytest', False),
        ("other run pytest", False),
    ],
)
def test_workspace_runtime_discovery_requires_unambiguous_invocation(project, command, expected):
    """Only a selected supported invocation can suggest an off-PATH interpreter."""
    workspace, selected, interpreter, identity = project
    details = interpreter.stat()
    result = discover_workspace_runtimes(
        command, CommandInspection().analyze(command), workspace, workspace, "/usr/bin", (identity,)
    )
    expected_identity = (str(selected), selected, interpreter, details.st_dev, details.st_ino)
    assert result == ((RuntimeCandidate(expected_identity, interpreter),) if expected else ())


@pytest.mark.parametrize(
    "case",
    [
        "other-cwd",
        "unbound-tool",
        "wrong-version",
        "override",
        "external-environment",
        "missing-project",
        "project-link",
        "project-hardlink",
        "project-too-large",
        "invalid-project",
        "non-project",
        "non-tool-table",
        "non-adapter-table",
        "configured-adapter",
        "duplicate-metadata",
        "missing-version",
        "invalid-version",
        "other-implementation",
        "bad-requirement",
        "numeric-requirement",
        "newer-requirement",
        "invalid-version-file",
        "version-file-link",
        "workspace-interpreter",
        "relative-home",
        "missing-home-target",
        "non-file-interpreter",
        "hardlinked-interpreter",
        "nonexecutable-interpreter",
    ],
)
def test_workspace_runtime_discovery_rejects_ambiguous_selection(project, tmp_path, case):
    """Unrelated or overridden environments do not add a candidate."""
    workspace, selected, interpreter, identity = project
    project_file = workspace / "pyproject.toml"
    config = workspace / ".venv/pyvenv.cfg"
    if case == "wrong-version":
        (workspace / ".python-version").write_text("3.12\n", encoding="utf-8")
    elif case == "override":
        (workspace / "uv.toml").write_text("python = 'other'\n", encoding="utf-8")
    elif case == "external-environment":
        selected.parent.parent.rename(tmp_path / "external")
        (workspace / ".venv").symlink_to(tmp_path / "external", target_is_directory=True)
    elif case == "missing-project":
        project_file.unlink()
    elif case == "project-link":
        project_file.unlink()
        project_file.symlink_to(interpreter)
    elif case == "project-hardlink":
        os.link(project_file, workspace / "linked-project")
    elif case == "project-too-large":
        project_file.write_bytes(b"x" * 65537)
    elif case == "invalid-project":
        project_file.write_bytes(b"\xff")
    elif case == "non-project":
        project_file.write_text('project = "fixture"\n', encoding="utf-8")
    elif case == "non-tool-table":
        project_file.write_text('tool = "fixture"\n[project]\n', encoding="utf-8")
    elif case == "non-adapter-table":
        project_file.write_text('[project]\n[tool]\nuv = "fixture"\n', encoding="utf-8")
    elif case == "configured-adapter":
        project_file.write_text('[project]\n[tool.uv]\npython = "other"\n', encoding="utf-8")
    elif case == "duplicate-metadata":
        config.write_text("version_info = 3.14\nversion_info = 3.12\n", encoding="utf-8")
    elif case == "missing-version":
        config.write_text("no-version\n", encoding="utf-8")
    elif case == "invalid-version":
        config.write_text("version_info = invalid\n", encoding="utf-8")
    elif case == "other-implementation":
        config.write_text("implementation = PyPy\nversion_info = 3.14\n", encoding="utf-8")
    elif case in {"bad-requirement", "numeric-requirement", "newer-requirement"}:
        requirement = {
            "bad-requirement": '"~=3.14"',
            "numeric-requirement": "3",
            "newer-requirement": '">=3.15"',
        }[case]
        project_file.write_text(f"[project]\nrequires-python = {requirement}\n", encoding="utf-8")
    elif case == "invalid-version-file":
        (workspace / ".python-version").write_text("invalid\n", encoding="utf-8")
    elif case == "version-file-link":
        (workspace / ".python-version").symlink_to(interpreter)
    elif case == "workspace-interpreter":
        selected.unlink()
        selected.write_text("fixture", encoding="utf-8")
        selected.chmod(0o755)
    elif case == "relative-home":
        config.write_text(
            "home = relative\nimplementation = CPython\nversion_info = 3.14\n", encoding="utf-8"
        )
    elif case == "missing-home-target":
        config.write_text(
            f"home = {tmp_path}\nimplementation = CPython\nversion_info = 3.14\n", encoding="utf-8"
        )
    elif case == "non-file-interpreter":
        selected.unlink()
        selected.symlink_to(interpreter.parent, target_is_directory=True)
        config.write_text(
            f"home = {tmp_path}\nimplementation = CPython\nversion_info = 3.14\n", encoding="utf-8"
        )
    elif case == "hardlinked-interpreter":
        os.link(interpreter, tmp_path / "hardlink")
    elif case == "nonexecutable-interpreter":
        interpreter.chmod(0o644)
    command = "uv run --no-sync pytest"
    assert not discover_workspace_runtimes(
        command,
        CommandInspection().analyze(command),
        workspace,
        tmp_path if case == "other-cwd" else workspace,
        "/usr/bin",
        () if case == "unbound-tool" else (identity,),
    )


@pytest.mark.parametrize("path_value", [".venv/bin", "/usr/bin"])
def test_workspace_runtime_discovery_includes_environment_on_path(project, path_value):
    """A PATH entry alone does not guarantee the selected interpreter is authorized."""
    workspace, selected, interpreter, identity = project
    if path_value == ".venv/bin":
        path_value = str(selected.parent)
    details = interpreter.stat()
    command = "uv run --no-sync pytest"
    assert discover_workspace_runtimes(
        command, CommandInspection().analyze(command), workspace, workspace, path_value, (identity,)
    ) == (
        RuntimeCandidate(
            (str(selected), selected, interpreter, details.st_dev, details.st_ino), interpreter
        ),
    )


@pytest.mark.parametrize("source", ["uv run --no-sync task", "uv --no-cache run --no-sync task"])
def test_workspace_runtime_discovery_accepts_matching_metadata(project, source):
    """Compatible project and version metadata retain the selected environment."""
    workspace, _, _, identity = project
    (workspace / ".python-version").write_text("3.14\n", encoding="utf-8")
    (workspace / "pyproject.toml").write_text(
        '[project]\n[tool.uv]\npackage = false\nexclude-newer = "7 days"\n', encoding="utf-8"
    )
    assert discover_workspace_runtimes(
        source, CommandInspection().analyze(source), workspace, workspace, "/usr/bin", (identity,)
    )


@pytest.mark.parametrize("selection", ["relative", "absolute", "special"])
def test_workspace_runtime_discovery_honors_explicit_project_environment(project, selection):
    """A literal workspace-local override selects its own environment."""
    workspace, _, interpreter, identity = project
    environment = workspace / ("chosen env's" if selection == "special" else "chosen-environment")
    (workspace / ".venv").rename(environment)
    selected = environment / "bin/python"
    value = environment.name if selection in {"relative", "special"} else str(environment)
    command = f"UV_PROJECT_ENVIRONMENT={shlex.quote(value)} uv run --no-sync arbitrary-task"
    details = interpreter.stat()

    assert discover_workspace_runtimes(
        command, CommandInspection().analyze(command), workspace, workspace, "/usr/bin", (identity,)
    ) == (
        RuntimeCandidate(
            (str(selected), selected, interpreter, details.st_dev, details.st_ino), interpreter
        ),
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "UV_PROJECT_ENVIRONMENT=../outside",
        "UV_PROJECT_ENVIRONMENT=/outside",
        "UV_PROJECT_ENVIRONMENT=$DYNAMIC",
        "UV_PROJECT_ENVIRONMENT=''",
        "UV_PROJECT_ENVIRONMENT=chosen OTHER=value",
    ],
)
def test_workspace_runtime_discovery_ignores_ambiguous_environment_overrides(project, prefix):
    """An ambiguous or external override cannot suggest a workspace runtime."""
    workspace, _, _, identity = project
    command = f"{prefix} uv run --no-sync arbitrary-task"

    assert not discover_workspace_runtimes(
        command, CommandInspection().analyze(command), workspace, workspace, "/usr/bin", (identity,)
    )


@pytest.mark.parametrize("case", ["unstable", "oversized-read", "unavailable", "inaccessible"])
def test_workspace_runtime_discovery_refuses_unstable_metadata(project, monkeypatch, case):
    """Changed, unreadable, or unavailable metadata cannot supply a runtime hint."""
    workspace, _, _, identity = project
    original_open = runtime_discovery.open_selected_file
    original_fstat = os.fstat
    original_read = os.read
    descriptors = {}

    def selected_file(*args):
        """Track a bound metadata descriptor."""
        if case == "unavailable":
            raise OSError("fixture unavailable")
        descriptor = original_open(*args)
        descriptors[descriptor] = 0
        return descriptor

    def changed_stat(descriptor):
        """Present a modification after the bounded metadata read."""
        details = original_fstat(descriptor)
        if descriptor in descriptors:
            descriptors[descriptor] += 1
            if case == "unstable" and descriptors[descriptor] == 2:
                return SimpleNamespace(
                    st_size=details.st_size,
                    st_mtime_ns=details.st_mtime_ns + 1,
                    st_ctime_ns=details.st_ctime_ns,
                )
        return details

    def oversized_read(descriptor, count):
        """Report excess bytes only for the selected fixture descriptor."""
        if case == "oversized-read" and descriptor in descriptors:
            return b"x" * count
        return original_read(descriptor, count)

    original_lstat = Path.lstat

    def inaccessible(path, *args, **kwargs):
        """Treat an unavailable optional configuration as ambiguous."""
        if case == "inaccessible" and path.name == "uv.toml":
            raise PermissionError("fixture denied")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(runtime_discovery, "open_selected_file", selected_file)
    monkeypatch.setattr(runtime_discovery.os, "fstat", changed_stat)
    monkeypatch.setattr(runtime_discovery.os, "read", oversized_read)
    monkeypatch.setattr(Path, "lstat", inaccessible)
    command = "uv run --no-sync pytest"
    assert not discover_workspace_runtimes(
        command, CommandInspection().analyze(command), workspace, workspace, "/usr/bin", (identity,)
    )


def test_workspace_runtime_discovery_requires_a_literal_initial_command(project):
    """Ambiguous source and command metadata never request a runtime."""
    workspace, _, _, identity = project
    analysis = CommandInspection().analyze("uv run pytest")
    for source, modified in [
        ("", analysis),
        ("UV_PROJECT_ENVIRONMENT='unterminated", analysis),
        ("uv run pytest", replace(analysis, first_executable_initial=False)),
        ("uv run pytest", replace(analysis, first_executable_literal=False)),
        ("uv run pytest", replace(analysis, first_path_inherited=False)),
        ("uv run pytest", replace(analysis, executables=())),
    ]:
        assert not discover_workspace_runtimes(
            source, modified, workspace, workspace, "/usr/bin", (identity,)
        )


@pytest.mark.parametrize(
    "case",
    [
        "selected",
        "arbitrary-name",
        "stale",
        "outside",
        "wrong-root",
        "invalid-config",
        "missing-config",
        "missing-version",
        "other-implementation",
        "missing-executable",
    ],
)
def test_workspace_interpreter_discovery_uses_selected_identity(project, tmp_path, case):
    """Only an identity-bound workspace interpreter can suggest its runtime."""
    workspace, selected, interpreter, _ = project
    if case == "arbitrary-name":
        selected = selected.parent / "fixture-engine"
        selected.symlink_to(interpreter)
    elif case == "outside":
        selected = tmp_path / "outside"
        selected.symlink_to(interpreter)
    elif case == "wrong-root":
        selected = workspace / "fixture-engine"
        selected.symlink_to(interpreter)
    elif case == "invalid-config":
        (selected.parent.parent / "pyvenv.cfg").write_bytes(b"\xff")
    elif case == "missing-config":
        (selected.parent.parent / "pyvenv.cfg").unlink()
    elif case == "missing-version":
        (selected.parent.parent / "pyvenv.cfg").write_text(
            "implementation = CPython\n", encoding="utf-8"
        )
    elif case == "other-implementation":
        (selected.parent.parent / "pyvenv.cfg").write_text(
            "implementation = PyPy\nversion_info = 3.14\n", encoding="utf-8"
        )
    elif case == "missing-executable":
        selected.unlink()
    details = interpreter.stat()
    identity = (
        str(selected),
        selected,
        interpreter,
        details.st_dev,
        details.st_ino + int(case == "stale"),
    )

    result = discover_workspace_interpreters(workspace, (identity,))

    expected = case in {"selected", "arbitrary-name"}
    assert result == ((RuntimeCandidate(identity, interpreter),) if expected else ())


def test_workspace_interpreter_discovery_bounds_the_identity_scan(project, tmp_path):
    """Later identities cannot cause unbounded environment inspection."""
    workspace, selected, interpreter, _ = project
    details = interpreter.stat()
    outside = tmp_path / "outside"
    ignored = (str(outside), outside, interpreter, details.st_dev, details.st_ino)
    selected_identity = (str(selected), selected, interpreter, details.st_dev, details.st_ino)

    assert not discover_workspace_interpreters(workspace, (ignored,) * 128 + (selected_identity,))


@pytest.mark.parametrize("kind", ["project", "interpreter"])
@pytest.mark.parametrize("matching", [True, False])
def test_standard_environment_metadata_requires_matching_executable(project, kind, matching):
    """A standard environment suggests a runtime only when its executable identity matches."""
    workspace, selected, interpreter, identity = project
    reference = interpreter if matching else interpreter.parent / "other"
    if not matching:
        reference.write_text("other", encoding="utf-8")
    (selected.parent.parent / "pyvenv.cfg").write_text(
        f"home = {interpreter.parent}\nversion = 3.14.2\nexecutable = {reference}\n",
        encoding="utf-8",
    )
    details = interpreter.stat()
    selected_identity = (str(selected), selected, interpreter, details.st_dev, details.st_ino)
    command = "uv run --no-sync pytest"

    result = (
        discover_workspace_runtimes(
            command,
            CommandInspection().analyze(command),
            workspace,
            workspace,
            "/usr/bin",
            (identity,),
        )
        if kind == "project"
        else discover_workspace_interpreters(workspace, (selected_identity,))
    )

    expected = (RuntimeCandidate(selected_identity, interpreter),) if matching else ()
    assert result == expected
