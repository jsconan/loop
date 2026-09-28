"""Inspect recognizable shell effects without treating source inspection as enforcement."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .models import CommandFinding, CommandReviewStatus

type CommandMatcher = Callable[[list[str]], CommandFinding | None]


@dataclass(frozen=True)
class CommandAnalysis:
    """Hold advisory findings, Git-read intent, and executable candidates.

    Args:
        findings (tuple[CommandFinding, ...]): Recognized effects requiring review.
        git_read (bool): Whether a recognized Git command may read repository metadata.
        executables (tuple[str, ...]): Bounded command names for tool resolution.
    """

    findings: tuple[CommandFinding, ...]
    git_read: bool
    executables: tuple[str, ...]


def match_destructive_command(command: list[str]) -> CommandFinding | None:
    """Return a finding for a recognizable direct file mutation.

    Args:
        command (list[str]): Normalized shell command segment.

    Returns:
        CommandFinding | None: Fresh destructive-command finding when recognized, otherwise None.
    """
    name = Path(command[0]).name
    arguments = command[1:]
    if (
        name in {"rm", "rmdir", "unlink", "shred", "truncate"}
        or (name in {"cp", "mv", "install", "tee"} and arguments)
        or (name == "dd" and any(argument.startswith("of=") for argument in arguments))
        or (
            name in {"sed", "perl"}
            and any(
                argument == "--in-place" or argument.startswith(("-i", "--in-place="))
                for argument in arguments
            )
        )
        or (name == "find" and "-delete" in arguments)
    ):
        return CommandFinding(
            policy_id="destructive_command",
            status=CommandReviewStatus.FRESH,
            context="Review a destructive workspace command",
            reason="delete or overwrite files in this workspace",
        )
    return None


def match_git_mutation(command: list[str]) -> CommandFinding | None:
    """Return a finding unless a Git command is known to be read-only.

    Args:
        command (list[str]): Normalized shell command segment.

    Returns:
        CommandFinding | None: Fresh Git-change finding when recognized, otherwise None.
    """
    if Path(command[0]).name != "git":
        return None
    arguments = command[1:]
    while arguments and arguments[0] == "-C" and len(arguments) > 1:
        arguments = arguments[2:]
    if not arguments:
        return None
    read_only = {
        "--version",
        "version",
        "help",
        "status",
        "diff",
        "log",
        "show",
        "rev-parse",
        "ls-files",
        "ls-tree",
        "grep",
        "shortlog",
        "describe",
        "cat-file",
        "merge-base",
        "for-each-ref",
    }
    effect_options = {"--output", "--ext-diff", "--open-files-in-pager"}
    if arguments[0] not in read_only or any(
        argument in effect_options
        or any(argument.startswith(f"{option}=") for option in effect_options)
        for argument in arguments[1:]
    ):
        return CommandFinding(
            policy_id="git_change",
            status=CommandReviewStatus.FRESH,
            context="Review a Git state-changing command",
            reason="change repository state",
            requests_git_write=True,
        )
    return None


class CommandInspection:
    """Inspect shell source with registered finding-producing matchers.

    The default matchers recognize Git state changes and direct destructive workspace commands.
    Inspection is lexical and bounded to two nested ``sh -c`` calls. Use ``with_matcher`` to
    extend the built-ins or pass an explicit sequence to replace them.

    Args:
        matchers (Iterable[CommandMatcher]): Matchers to evaluate in caller-selected order.
            Defaults to the built-in Git and destructive-command matchers; an empty iterable
            disables them.
    """

    SHELLS = frozenset({"sh", "bash", "zsh"})
    SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")"})
    DESTRUCTIVE_REDIRECTIONS = frozenset({">", ">|", "<>", "&>"})
    DEFAULT_COMMAND_MATCHERS = (match_git_mutation, match_destructive_command)

    _matchers: tuple[CommandMatcher, ...]

    def __init__(self, matchers: Iterable[CommandMatcher] = DEFAULT_COMMAND_MATCHERS) -> None:
        self._matchers = tuple(matchers)

    def with_matcher(self, matcher: CommandMatcher) -> CommandInspection:
        """Return an inspection extended with one matcher.

        Args:
            matcher (CommandMatcher): Matcher to append after existing entries.

        Returns:
            CommandInspection: New inspection retaining the current matcher order.
        """
        return CommandInspection((*self._matchers, matcher))

    def inspect(self, source: str) -> tuple[CommandFinding, ...]:
        """Collect all applicable findings in matcher registration order.

        Args:
            source (str): Opaque POSIX shell source to inspect heuristically.

        Returns:
            tuple[CommandFinding, ...]: Applicable findings; absence does not prove safety.
        """
        return self.analyze(source).findings

    def analyze(self, source: str) -> CommandAnalysis:
        """Inspect source once for effects and candidate executables.

        Args:
            source (str): Opaque POSIX shell source to inspect heuristically.

        Returns:
            CommandAnalysis: Bounded advisory classification; absence never proves safety.
        """
        findings, git_read, executables = self._scan(source)
        return CommandAnalysis(
            tuple(finding for finding in findings if finding is not None),
            git_read,
            tuple(dict.fromkeys(executables))[:128],
        )

    def _scan(
        self, source: str, depth: int = 0
    ) -> tuple[list[CommandFinding | None], bool, list[str]]:
        """Scan one shell level and merge bounded nested command classifications."""
        findings = [None] * len(self._matchers)
        git_read = False
        executables = []
        try:
            lexer = shlex.shlex(source, posix=True, punctuation_chars=";&|()<>")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return findings, git_read, executables
        start = 0
        for index in range(len(tokens) + 1):
            if index < len(tokens) and tokens[index] not in self.SEPARATORS:
                continue
            command = tokens[start:index]
            start = index + 1
            if self._has_destructive_redirection(command):
                self._add_destructive_finding(findings)
            command, ambiguous = self._strip_shell_prefixes(command)
            if ambiguous:
                self._add_destructive_finding(findings)
                continue
            if not command:
                continue
            name = Path(command[0]).name
            executables.append(command[0])
            if name == "git" and len(command) > 1:
                git_read = git_read or command[1] not in {"--version", "version", "help"}
            for position, matcher in enumerate(self._matchers):
                if findings[position] is None and (finding := matcher(command)) is not None:
                    findings[position] = finding
            if Path(command[0]).name in self.SHELLS and depth < 2 and "-c" in command[1:]:
                arguments = command[1:]
                position = arguments.index("-c")
                if position + 1 < len(arguments):
                    nested_findings, nested_git_read, nested_executables = self._scan(
                        arguments[position + 1], depth + 1
                    )
                    git_read = git_read or nested_git_read
                    executables.extend(nested_executables)
                    for nested_position, finding in enumerate(nested_findings):
                        if findings[nested_position] is None:
                            findings[nested_position] = finding
        return findings, git_read, executables

    def _add_destructive_finding(self, findings: list[CommandFinding | None]) -> None:
        """Record a redirection or ambiguous-wrapper destructive finding when enabled."""
        for position, matcher in enumerate(self._matchers):
            if matcher is match_destructive_command and findings[position] is None:
                findings[position] = CommandFinding(
                    policy_id="destructive_command",
                    status=CommandReviewStatus.FRESH,
                    context="Review a destructive workspace command",
                    reason="delete or overwrite files in this workspace",
                )

    def _has_destructive_redirection(self, command: list[str]) -> bool:
        """Return whether a shell segment contains a destination-bearing redirection."""
        return any(
            token in self.DESTRUCTIVE_REDIRECTIONS
            or (
                token == ">&"
                and (position + 1 == len(command) or not command[position + 1].isdigit())
            )
            for position, token in enumerate(command)
        )

    def _strip_shell_prefixes(self, command: list[str]) -> tuple[list[str], bool]:
        """Remove supported shell prefixes; flag ambiguous wrappers for fresh review."""
        while command:
            token = command[0]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token) or token in {"command", "exec"}:
                command = command[1:]
            elif token == "env":
                command = command[1:]
                while command:
                    option = command[0]
                    if option == "--":
                        command = command[1:]
                        break
                    if option in {"-i", "--ignore-environment", "-0", "--null"}:
                        command = command[1:]
                    elif option in {"-u", "--unset"}:
                        if len(command) < 2 or not command[1] or command[1].startswith("-"):
                            return [], True
                        command = command[2:]
                    elif (
                        (option.startswith("--unset=") and option[8:])
                        or (option.startswith("-u") and len(option) > 2)
                        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", option)
                    ):
                        command = command[1:]
                    elif option.startswith("-"):
                        return [], True
                    else:
                        break
            else:
                break
        return command, False
