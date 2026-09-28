"""Tests for composable, advisory shell command findings."""

import pytest

from loop.permissions import (
    CommandFinding,
    CommandInspection,
    CommandReviewStatus,
    match_destructive_command,
    match_git_mutation,
)


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
        ("cd /workspace && git diff --cached", False),
        ("git --version", False),
        ("git", False),
        ("git -C repository add file", True),
        ("git add file", True),
        ("command git add file", True),
        ("cd /workspace && env git add file", True),
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
        ("bash -c 'cd /workspace && git commit -m change'", True),
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
