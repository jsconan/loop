"""Inspect recognizable shell effects without treating source inspection as enforcement."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .models import CommandFinding, CommandReviewStatus

type CommandMatcher = Callable[[list[str]], CommandFinding | None]

_SHELL_TOKEN = re.compile(
    r"(?P<operator>&>>|<<-|>>|>\||<>|>&|<&|&>|<<|&&|\|\||[;&|()<>\n])"
    r"|(?P<word>(?:[^\s;&|()<>'\"\\]+|'[^']*'|\"(?:\\.|[^\"\\])*\"|\\[\s\S])+)",
)


@dataclass(frozen=True)
class ShellToken:
    """Retain word quoting and operator identity for bounded lexical inspection."""

    value: str
    raw: str
    operator: bool
    start: int
    end: int


def _shell_tokens(source: str) -> list[ShellToken]:
    """Tokenize shell words without mistaking quoted punctuation for operators."""
    tokens = []
    position = 0
    while position < len(source):
        if source[position] in " \t\r":
            position += 1
            continue
        if source.startswith("\\\n", position):
            position += 2
            continue
        if source[position] == "#":
            position = source.find("\n", position)
            if position == -1:
                break
            continue
        match = _SHELL_TOKEN.match(source, position)
        if match is None:
            raise ValueError("Unsupported or incomplete shell word.")
        raw = match.group()
        operator = match.lastgroup == "operator"
        value = raw if operator else shlex.split(raw.replace("\\\n", ""), comments=False)[0]
        tokens.append(ShellToken(value, raw, operator, position, match.end()))
        position = match.end()
    return tokens


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
    WRITE_REDIRECTIONS = frozenset({">", ">|", ">>", "<>", "&>", "&>>", ">&"})
    REDIRECTIONS = WRITE_REDIRECTIONS | frozenset({"<", "<&", "<<", "<<-"})
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

    def inspect(
        self,
        source: str,
        *,
        temporary_roots: tuple[Path, ...] = (),
    ) -> tuple[CommandFinding, ...]:
        """Collect all applicable findings in matcher registration order.

        Args:
            source (str): Opaque POSIX shell source to inspect heuristically.
            temporary_roots (tuple[Path, ...]): Bound, managed temporary roots whose literal
                destinations need no destructive review. Defaults to no temporary exemptions.

        Returns:
            tuple[CommandFinding, ...]: Applicable findings; absence does not prove safety.
        """
        return self.analyze(source, temporary_roots=temporary_roots).findings

    def analyze(self, source: str, *, temporary_roots: tuple[Path, ...] = ()) -> CommandAnalysis:
        """Inspect source once for effects and candidate executables.

        Args:
            source (str): Opaque POSIX shell source to inspect heuristically.
            temporary_roots (tuple[Path, ...]): Bound, managed temporary roots whose literal
                destinations need no destructive review. Defaults to no temporary exemptions.

        Returns:
            CommandAnalysis: Bounded advisory classification; absence never proves safety.
        """
        findings, git_read, executables = self._scan(source, temporary_roots)
        return CommandAnalysis(
            tuple(finding for finding in findings if finding is not None),
            git_read,
            tuple(dict.fromkeys(executables))[:128],
        )

    def _scan(
        self,
        source: str,
        temporary_roots: tuple[Path, ...],
        depth: int = 0,
    ) -> tuple[list[CommandFinding | None], bool, list[str]]:
        """Scan one shell level and merge bounded nested command classifications."""
        findings = [None] * len(self._matchers)
        git_read = False
        executables = []
        try:
            tokens = _shell_tokens(source)
        except ValueError:
            return findings, git_read, executables
        start = 0
        for index in range(len(tokens) + 1):
            if index < len(tokens) and not (
                tokens[index].operator and tokens[index].value in self.SEPARATORS | {"\n"}
            ):
                continue
            segment = tokens[start:index]
            start = index + 1
            command, review = self._strip_redirections(segment, temporary_roots)
            if review:
                self._add_destructive_finding(findings, redirection=True)
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
                        arguments[position + 1], temporary_roots, depth + 1
                    )
                    git_read = git_read or nested_git_read
                    executables.extend(nested_executables)
                    for nested_position, finding in enumerate(nested_findings):
                        if findings[nested_position] is None:
                            findings[nested_position] = finding
        return findings, git_read, executables

    def _add_destructive_finding(
        self,
        findings: list[CommandFinding | None],
        *,
        redirection: bool = False,
    ) -> None:
        """Record a redirection or ambiguous-wrapper destructive finding when enabled."""
        for position, matcher in enumerate(self._matchers):
            if matcher is match_destructive_command and findings[position] is None:
                findings[position] = CommandFinding(
                    policy_id="destructive_command",
                    status=CommandReviewStatus.FRESH,
                    context=(
                        "Review an output destination" if redirection else "Review a shell wrapper"
                    ),
                    reason=(
                        "write to an output destination that may overwrite or modify files"
                        if redirection
                        else "run a shell wrapper whose effects could not be determined"
                    ),
                )

    def _strip_redirections(
        self,
        tokens: list[ShellToken],
        temporary_roots: tuple[Path, ...],
    ) -> tuple[list[str], bool]:
        """Separate command words from redirects, retaining review for unproven destinations."""
        words = []
        last_word_position = -1
        review = False
        position = 0
        while position < len(tokens):
            token = tokens[position]
            if not token.operator or token.value not in self.REDIRECTIONS:
                words.append(token.value)
                last_word_position = position
                position += 1
                continue
            if (
                last_word_position == position - 1
                and position > 0
                and tokens[position - 1].end == token.start
                and tokens[position - 1].raw.isdecimal()
            ):
                words.pop()
            position += 1
            destination = tokens[position] if position < len(tokens) else None
            if token.value in self.WRITE_REDIRECTIONS:
                descriptor = (
                    token.value == ">&"
                    and destination is not None
                    and (destination.value.isdecimal() or destination.value == "-")
                )
                review = review or not (
                    descriptor or self._managed_destination(destination, temporary_roots)
                )
            if destination is not None:
                position += 1
        return words, review

    @staticmethod
    def _managed_destination(token: ShellToken | None, temporary_roots: tuple[Path, ...]) -> bool:
        """Recognize only literal null-device or contained managed temporary destinations."""
        if token is None or token.operator or any(char in token.raw for char in "$`*?[\n"):
            return False
        path = Path(token.value)
        if token.value == "/dev/null":
            return True
        if not path.is_absolute() or ".." in path.parts:
            return False
        try:
            resolved = path.resolve(strict=False)
            return any(resolved.is_relative_to(root) for root in temporary_roots)
        except (OSError, RuntimeError):
            return False

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
