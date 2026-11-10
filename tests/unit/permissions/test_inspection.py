"""Tests for composable, advisory shell command findings."""

from pathlib import Path

import pytest

from loop.permissions import (
    CommandFinding,
    CommandInspection,
    CommandReviewStatus,
    literal_shell_word,
    match_destructive_command,
    match_git_mutation,
    shell_tokens,
)


def test_shell_inspection_exposes_exact_tokens_and_literal_classification():
    """Discovery can inspect source offsets and literals through the shared public boundary."""
    tokens = shell_tokens("tool 'two words' $VALUE > /dev/null")
    assert [(token.value, token.raw, token.operator) for token in tokens] == [
        ("tool", "tool", False),
        ("two words", "'two words'", False),
        ("$VALUE", "$VALUE", False),
        (">", ">", True),
        ("/dev/null", "/dev/null", False),
    ]
    assert literal_shell_word(tokens[1].raw)
    assert not literal_shell_word(tokens[2].raw)
    words, review = CommandInspection().strip_redirections(tokens, ())
    assert [token.value for token in words] == ["tool", "two words", "$VALUE"]
    assert not review
    with pytest.raises(ValueError, match="Unsupported or incomplete shell word"):
        shell_tokens("tool 'unfinished")


def test_command_inspection_collects_every_default_finding_from_nested_shell():
    """Default matchers classify nested Git and destructive effects in stable order."""
    findings = CommandInspection().inspect("sh -c 'rm file && git add file'")

    assert [finding.policy_id for finding in findings] == ["git_change", "destructive_command"]
    assert [finding.reason for finding in findings] == [
        "change repository state",
        "delete or overwrite files in this workspace",
    ]
    assert all(finding.status is CommandReviewStatus.FRESH for finding in findings)
    assert [finding.requests_git_write for finding in findings] == [True, False]


def test_command_analysis_retains_literal_path_arguments_without_granting_authority():
    """Literal path hints exclude redirects and paths requiring shell expansion."""
    analysis = CommandInspection().analyze(
        "./launcher task './tools/job with space' $WORKSPACE_ROOT/dynamic ./glob* ~/dynamic "
        "--option=./value > ./output; sh -c 'other ../nested/task'"
    )

    assert analysis.executables == ("./launcher", "sh", "other")
    assert analysis.path_arguments == ("./tools/job with space", "../nested/task", "task")


def test_command_analysis_prioritizes_explicit_paths_with_many_arguments():
    """Bare-name hints cannot crowd explicit paths out of bounded discovery."""
    names = " ".join(f"item{index}" for index in range(150))
    analysis = CommandInspection().analyze(f"launcher {names} ./selected ''")

    assert len(analysis.path_arguments) == 128
    assert analysis.path_arguments[0] == "./selected"
    assert "" not in analysis.path_arguments


@pytest.mark.parametrize(
    ("source", "literal"),
    [
        ("./tool", True),
        ("$NAME/tool", False),
        ('"$NAME/tool"', False),
        ("'$NAME/tool'", True),
        (r"\$NAME/tool", True),
        (r"./tool\ with\ space", True),
        ("./'tool with space'", True),
        (r"./to\?l", True),
        ("./to?l", False),
        ("'./to?l'", True),
        ('"./to?l"', True),
        (r'"./\$NAME/tool"', False),
        ("./{one,two}", False),
        ("~/tool", False),
        ("'~/tool'", True),
        ("exec './tool with space'", True),
        ("VALUE=literal command './tool with space'", True),
        (">/dev/null './tool with space'", True),
        ("2>/dev/null exec './tool with space'", True),
        ("VALUE=literal", False),
        ("$NAME/tool; '$NAME/tool'", False),
        ("'$NAME/tool'; $NAME/tool", True),
        ("'", False),
    ],
)
def test_command_analysis_identifies_literal_first_executable(source, literal):
    """Quoted literals and escaped characters differ from shell-expanded command words."""
    assert CommandInspection().analyze(source).first_executable_literal is literal


@pytest.mark.parametrize(
    ("source", "initial"),
    [
        ("exec ./task; :", True),
        ("OPTION=value ./task && :", True),
        ("OPTION='value with space' ./task && :", True),
        ("'OPTION=value' ./task", False),
        ("env ./task", False),
        (">/dev/null ./task; :", True),
        (":; ./task", True),
        ("; ./task", False),
        ("( ./task )", False),
        ("./task() { :; }", False),
        ("'", False),
    ],
)
def test_command_analysis_identifies_initial_simple_command(source, initial):
    """Only the initial simple command can drive first-call interpreter preparation."""
    assert CommandInspection().analyze(source).first_executable_initial is initial


@pytest.mark.parametrize(
    ("source", "argument"),
    [
        ("./launcher ./task", "./task"),
        ("./launcher './task with space'", "./task with space"),
        ("./launcher $TASK", None),
        ('./launcher "$TASK"', None),
        ("./launcher ./task*", None),
        ("./launcher '$TASK'", "$TASK"),
        ("./launcher >/dev/null ./task", "./task"),
        ("./launcher", None),
        ("'", None),
    ],
)
def test_command_analysis_retains_only_a_literal_first_argument(source, argument):
    """A first argument requiring shell expansion cannot identify a selected file."""
    assert CommandInspection().analyze(source).first_executable_argument == argument


@pytest.mark.parametrize(
    ("source", "arguments"),
    [
        ("./outer ./inner ./tool --flag", ("./inner", "./tool", "--flag")),
        ("./outer './inner with space' '$NAME'", ("./inner with space", "$NAME")),
        ("./outer ./inner $TOOL ./later", ("./inner",)),
        ("./outer $INNER ./tool", ()),
        ("./outer ./inner ./tool* ./later", ("./inner",)),
        ("./outer >/dev/null ./inner ./tool; other", ("./inner", "./tool")),
        ("./outer", ()),
        ("'", ()),
    ],
)
def test_command_analysis_retains_the_literal_argument_prefix(source, arguments):
    """Later words are not treated as positional arguments after an expansion."""
    assert CommandInspection().analyze(source).first_executable_arguments == arguments


def test_command_analysis_bounds_the_literal_argument_prefix():
    """Argument discovery remains bounded even for very long command lines."""
    analysis = CommandInspection().analyze("./outer " + " ".join(f"item{n}" for n in range(140)))

    assert analysis.first_executable_arguments == tuple(f"item{n}" for n in range(128))


@pytest.mark.parametrize(
    ("source", "inherited"),
    [
        ("fixture", True),
        ("VALUE=one fixture", True),
        ("exec fixture", True),
        ("PATH=/other fixture", False),
        ("VALUE=one PATH=/other fixture", False),
        ("env fixture", False),
        ("env -i fixture", False),
        ("'", False),
    ],
)
def test_command_analysis_identifies_inherited_path_lookup(source, inherited):
    """Only supported prefixes that preserve PATH can use inherited lookup for discovery."""
    assert CommandInspection().analyze(source).first_path_inherited is inherited


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("first; second", (("first", True), ("second", True))),
        ("PATH=/other first; second", (("second", True),)),
        ("first; PATH=/other second", (("first", True),)),
        ("PATH=/other; second", ()),
        ("export PATH=/other; second", ()),
        (': "${PATH:=/other}"; second', ()),
        ('VALUE="${PATH:=/other}"; second', ()),
        (': "$((PATH=1))"; second', ()),
        (': "$((PATH[0]=1))"; second', ()),
        (': "$((1+$((PATH=1))))"; second', ()),
        (': "$((PATH=1)"; second', ()),
        (': "$((1+2))"; second', (("second", True),)),
        (': "$((0xff+1))"; second', (("second", True),)),
        (': "$((16#ff+1))"; second', (("second", True),)),
        (': "$((VALUE))"; second', ()),
        (': "$((1+$VALUE))"; second', ()),
        (": '${PATH:=/other}'; second", (("second", True),)),
        (r': "\${PATH:=/other}"; second', (("second", True),)),
        (r': "\\${PATH:=/other}"; second', ()),
        ("printf -v PATH value; second", ()),
        ("set -e; second", (("second", True),)),
        ("set -k; second", ()),
        ("set -o keyword; second", ()),
        ("set -ek; second", ()),
        ("set -okeyword; second", ()),
        ("set +o keyword; second", (("second", True),)),
        ("set -o; second", (("second", True),)),
        ("set -; second", (("second", True),)),
        ("set +; second", (("second", True),)),
        ("set -- -k; second", (("second", True),)),
        ("set value -k; second", (("second", True),)),
        ("pushd somewhere; second", (("second", False),)),
        ("env -i first; second", (("second", True),)),
        ("cd somewhere; second", (("second", False),)),
        ("cd .; second", (("second", True),)),
        ("if first; then second; fi", ()),
        ("fixture() { :; }; fixture", ()),
        ("sh -c 'first; second'", (("sh", True), ("first", True), ("second", True))),
        ('sh -c "$SOURCE"', (("sh", True),)),
    ],
)
def test_command_analysis_tracks_inherited_path_candidates(source, expected):
    """Known PATH changes never authorize lookup using an unrelated inherited search path."""
    assert CommandInspection().analyze(source).inherited_path_executables == expected


@pytest.mark.parametrize(
    "source",
    [
        "rm -rf build",
        "cd $WORKSPACE_ROOT && /bin/rm -rf build",
        "command rm build",
        "exec rmdir empty",
        "unlink obsolete",
        "shred old-key",
        "truncate -s 0 report",
        "find . -name '*.tmp' -delete",
        "git reset --hard HEAD",
        "git clean -fd",
        "git branch -D obsolete",
        "git add file",
        "git commit -m change",
        "git checkout other-branch",
        "git switch other-branch",
        "git restore file",
        "git apply patch.diff",
        "git merge other-branch",
        "git rebase main",
        "git cherry-pick HEAD~1",
        "git stash pop",
        "git tag release",
        "git push origin main",
        "git diff --output=changes.patch",
        "git diff --ext-diff",
        "git grep --open-files-in-pager=vim needle",
        "command git add file",
        "cd $WORKSPACE_ROOT && env git add file",
        "env GIT_CONFIG_NOSYSTEM=1 git add file",
        "sh -c 'git add file'",
        "sh -c 'rm -rf build'",
        "env rm -f victim",
        "env -i /bin/rm victim",
        "env -u HOME -- /bin/rm victim",
        "env --unset=HOME /bin/rm victim",
        "env -i /usr/bin/git clean -fd",
        "env -u HOME -- /usr/bin/git clean -fd",
        "env --unset=HOME /usr/bin/git clean -fd",
        "env --unknown /bin/rm victim",
        "env -u",
        "FOO=1 rm -f victim",
        "cp source existing",
        "mv source existing",
        "install source existing",
        "dd if=source of=existing",
        "tee existing < source",
        "sed -i.bak s/old/new/ existing",
        "perl -i -pe 's/old/new/' existing",
        "env cp source existing && echo done",
        "FOO=1 command mv source existing",
        "sh -c 'install source existing'",
        ": > victim",
        "printf text > victim",
        "printf text &> victim",
        "printf text >& victim",
        "printf text 1>& victim",
        "printf text 2&> victim",
        "env FOO=1 git add file",
        "FOO=1 git add file; echo done",
    ],
)
def test_destructive_shell_syntax_is_classified_for_fresh_review(source):
    """Recognized shell spellings request a fresh Git or destructive decision."""
    findings = CommandInspection().inspect(source)
    assert findings
    assert all(finding.status is CommandReviewStatus.FRESH for finding in findings)


@pytest.mark.parametrize(
    "source",
    [
        "printf '%s' 'rm -rf build'",
        "printf text 2>&1",
        "printf text >&1",
        "; printf ok",
        "exec",
        "env",
        "echo '",
    ],
)
def test_non_destructive_or_invalid_syntax_has_no_finding(source):
    """Literal output and invalid shell text do not imply a file mutation."""
    assert not CommandInspection().inspect(source)


@pytest.mark.parametrize(
    "source",
    [
        "echo '>'",
        'echo ">"',
        r"echo \>",
        "echo 'a; rm victim'",
        "printf text 2>&-",
        "printf text <&0",
        "printf text >&'1'",
        "printf text >/dev/null",
        "printf text >|/dev/null",
        "printf text >>/dev/null",
        "printf text &>/dev/null",
        "printf text &>>/dev/null",
        "printf text <>/dev/null",
        "printf text 2> '/dev/null'",
        ">/dev/null printf text",
        "sh -c 'printf text >/dev/null'",
        "printf text # >victim",
        "printf text\n# >victim",
        "printf text # >victim\nprintf done",
        "\\\nprintf text >/dev/null",
        "printf \\\ntext >/dev/null",
        "printf te\\\nxt >/dev/null",
        "cat <input >/dev/null",
        (
            "ls src/loop/utils src/loop/tools 2>/dev/null | head -40; "
            'echo "=== counts"; find src/loop -name "*.py" -not -name "__init__.py" | wc -l; '
            'find tests/unit -name "test_*.py" | wc -l; echo "=== collect"; '
            "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest tests/unit -q --co 2>&1 | tail -5"
        ),
    ],
)
def test_safe_redirections_and_quoted_operators_require_no_review(source):
    """Literal output sinks and descriptor operations do not imply workspace mutation."""
    assert not CommandInspection().inspect(source)


@pytest.mark.parametrize(
    "source",
    [
        "printf text >/dev/null >victim",
        "printf text >victim >/dev/null",
        "printf text >>victim",
        "printf text >|victim",
        "printf text <>victim",
        "printf text &>>victim",
        'printf text >"$OUTPUT"',
        'printf text >"$TMPDIR/report"',
        "printf text >/tmp/report",
        "printf text >/dev/null-other",
        "printf text >",
        "printf text >&",
        "printf text >1>victim",
        "printf text >; rm victim",
        "rm victim >/dev/null",
        ">/dev/null rm victim",
        "2>/dev/null git add victim",
        "printf text\nrm victim",
    ],
)
def test_safe_redirect_cannot_hide_other_mutation_or_unknown_output(source):
    """Every output destination and direct executable retains its independent review gate."""
    assert CommandInspection().inspect(source)


@pytest.mark.parametrize("operator", [">", ">>", ">|", "<>", "&>", "&>>"])
def test_literal_managed_temporary_outputs_need_no_review(tmp_path, operator):
    """Bound temporary destinations are exempt for each destination-bearing write form."""
    inspection = CommandInspection()
    source = f"printf text {operator}'{tmp_path}/report'"
    assert not inspection.analyze(source, temporary_roots=(tmp_path,)).findings
    assert not inspection.inspect(f'sh -c "{source}"', temporary_roots=(tmp_path,))


@pytest.mark.parametrize("suffix", ["../victim", "link/victim", "report*", "$(echo report)"])
def test_temporary_exemption_rejects_escape_or_dynamic_destinations(tmp_path, suffix):
    """Traversal, symlinks outside scratch and runtime destinations retain output review."""
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    assert CommandInspection().inspect(
        f"printf text >{tmp_path}/{suffix}", temporary_roots=(tmp_path,)
    )


def test_destination_resolution_failure_retains_review(monkeypatch):
    """Unresolvable temporary paths do not receive an exemption."""

    def fail_resolution(*_args, **_kwargs):
        """Model an unreadable or cyclic destination path."""
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "resolve", fail_resolution)
    assert CommandInspection().inspect("printf text >/tmp/report", temporary_roots=(Path("/tmp"),))


def test_command_inspection_does_not_claim_safety_from_no_findings():
    """An inspection reports no finding for unrecognized, invalid, or incomplete shell text."""
    inspection = CommandInspection()

    assert not inspection.inspect("printf safe")
    assert not inspection.inspect("echo '")
    assert not inspection.inspect("sh -c")


def test_command_inspection_keeps_the_first_finding_from_repeated_nested_effects():
    """Nested findings do not replace a finding already produced by the same matcher."""
    findings = CommandInspection().inspect("git add file && sh -c 'git commit -m change'")

    assert [finding.policy_id for finding in findings] == ["git_change"]


def test_analysis_carries_git_reads_and_tool_names_from_nested_commands():
    """One inspection supplies linked Git reads and candidates across shell segments."""
    analysis = CommandInspection().analyze(
        "env FOO=1 git -C repo status --short && sh -c 'git log -1 --format=%s; rg x'"
    )

    assert analysis.git_read
    assert analysis.executables == ("git", "sh", "rg")
    assert not analysis.findings


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("git status", False),
        ("git status && echo done", False),
        ("cd $WORKSPACE_ROOT && git diff --cached", False),
        ("git --version", False),
        ("git", False),
        ("git -C repository add file", True),
        ("git add file", True),
        ("command git add file", True),
        ("cd $WORKSPACE_ROOT && env git add file", True),
        ("env GIT_CONFIG_NOSYSTEM=1 git add file", True),
        ("git commit -m change", True),
        ("git clean -fd", True),
        ("env -i /usr/bin/git clean -fd", True),
        ("env -u HOME -- /usr/bin/git clean -fd", True),
        ("env --unset=HOME /usr/bin/git clean -fd", True),
        ("env --unknown /usr/bin/git clean -fd", False),
        ("git diff --output=changes.patch", True),
        ("git diff --output changes.patch", True),
        ("git diff --ext-diff", True),
        ("git grep --open-files-in-pager=vim needle", True),
        ("sh -c 'git add file'", True),
        ("bash -c 'cd $WORKSPACE_ROOT && git commit -m change'", True),
        ("echo 'git add file'", False),
        ("echo '", False),
    ],
)
def test_git_write_intent_preserves_reads_and_flags_repository_changes(source, expected):
    """Direct Git changes request the write boundary while reads remain advisory."""
    analysis = CommandInspection().analyze(source)
    assert any(finding.requests_git_write for finding in analysis.findings) is expected


def test_public_matchers_return_findings_for_their_respective_command_types():
    """Each public default matcher exposes its own recognition behavior."""
    git_finding = match_git_mutation(["git", "add", "file"])
    destructive_finding = match_destructive_command(["rm", "file"])

    assert git_finding is not None
    assert git_finding.policy_id == "git_change"
    assert destructive_finding is not None
    assert destructive_finding.policy_id == "destructive_command"


def test_command_inspection_composes_and_replaces_matchers():
    """A caller can extend built-ins or supply an independent ordered matcher set."""

    def match_special(command: list[str]) -> CommandFinding | None:
        """Request reusable review for the chosen command."""
        if command != ["special"]:
            return None
        return CommandFinding(
            policy_id="custom",
            status=CommandReviewStatus.REVIEW,
            context="Review a special command",
            reason="run a special command",
        )

    default = CommandInspection()
    extended = default.with_matcher(match_special)
    assert [finding.policy_id for finding in default.inspect("git add file")] == ["git_change"]
    assert [finding.policy_id for finding in extended.inspect("git add file; special")] == [
        "git_change",
        "custom",
    ]
    assert [
        finding.policy_id for finding in CommandInspection((match_special,)).inspect("special")
    ] == ["custom"]
    assert not CommandInspection(()).inspect("git add file")
