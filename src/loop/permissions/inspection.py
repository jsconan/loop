"""Inspect recognizable shell effects without treating source inspection as enforcement."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .models import CommandFinding, CommandReviewStatus

type CommandMatcher = Callable[[list[str]], CommandFinding | None]

_SHELL_TOKEN = re.compile(
    r"(?P<operator>&>>|<<-|>>|>\||<>|>&|<&|&>|<<|&&|\|\||[;&|()<>\n])"
    r"|(?P<word>(?:[^\s;&|()<>'\"\\]+|'[^']*'|\"(?:\\.|[^\"\\])*\"|\\[\s\S])+)",
)


@dataclass(frozen=True)
class ShellToken:
    """Retain one shell word or operator and its location in the original source.

    Args:
        value (str): Decoded word or literal operator.
        raw (str): Exact source spelling, including quotes and escapes.
        operator (bool): Whether the token is a shell operator.
        start (int): Inclusive source character offset.
        end (int): Exclusive source character offset.
    """

    value: str
    raw: str
    operator: bool
    start: int
    end: int


def shell_tokens(source: str) -> list[ShellToken]:
    """Tokenize shell words without mistaking quoted punctuation for operators.

    Args:
        source (str): Shell source to inspect without executing it.

    Returns:
        list[ShellToken]: Ordered tokens with their decoded and original spellings.

    Raises:
        ValueError: Source contains an unsupported or incomplete shell word.
    """
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


def literal_shell_word(raw: str) -> bool:
    """Identify a quoted or escaped shell word with no runtime expansion.

    Args:
        raw (str): Original word spelling as returned by ``shell_tokens``.

    Returns:
        bool: Whether the entire word is a literal without shell expansion.
    """
    quote = None
    index = 0
    while index < len(raw):
        char = raw[index]
        if quote == "'":
            if char == "'":
                quote = None
        elif char == "\\":
            if quote == '"' and raw[index + 1] not in '\\"\n':
                return False
            index += 1
        elif quote == '"':
            if char == '"':
                quote = None
            elif char in "$`":
                return False
        elif char in "'\"":
            quote = char
        elif char in "$`*?[{}" or (index == 0 and char == "~"):
            return False
        index += 1
    return True


def _path_mutating_expansion(raw: str) -> bool:
    """Detect unescaped shell expansions that can assign PATH before a command runs."""
    if "$" not in raw:
        return False
    quote = None
    index = 0
    while index < len(raw):
        char = raw[index]
        if quote == "'":
            if char == "'":
                quote = None
        elif char == "\\" and (quote is None or raw[index + 1] in '$`"\\\n'):
            index += 1
        elif char == '"':
            quote = None if quote == '"' else '"'
        elif char == "'" and quote is None:
            quote = "'"
        elif char == "$":
            if raw.startswith(("${PATH=", "${PATH:="), index):
                return True
            if raw.startswith("$((", index):
                end = raw.find("))", index + 3)
                if end < 0:
                    return True
                expression = raw[index + 3 : end]
                expression = re.sub(
                    r"(?<![A-Za-z0-9_])(?:0[xX][0-9a-fA-F]+|[0-9]+#[0-9A-Za-z@_]+)",
                    "0",
                    expression,
                )
                if re.fullmatch(r"[0-9\s+\-*/%<>=&|^~!?:(),]*", expression) is None:
                    return True
                index = end + 1
        index += 1
    return False


def _set_enables_keyword(arguments: list[str]) -> bool:
    """Recognize the shell option that changes where assignment words are applied."""
    position = 0
    while position < len(arguments):
        option = arguments[position]
        if option == "--" or not option.startswith(("-", "+")) or option in {"-", "+"}:
            break
        if option in {"-o", "+o"}:
            position += 1
            if position < len(arguments) and option == "-o" and arguments[position] == "keyword":
                return True
        elif option.startswith("-") and ("k" in option[1:] or option[1:].endswith("okeyword")):
            return True
        position += 1
    return False


@dataclass(frozen=True)
class CommandAnalysis:
    """Hold advisory findings, Git-read intent, and literal invocation candidates.

    Args:
        findings (tuple[CommandFinding, ...]): Recognized effects requiring review.
        git_read (bool): Whether a recognized Git command may read repository metadata.
        executables (tuple[str, ...]): Bounded command names for tool resolution.
        path_arguments (tuple[str, ...]): Bounded literal path and name hints, not authority.
        first_executable_literal (bool): Whether the first parsed executable is a literal word.
        first_executable_initial (bool): Whether it is in the initial simple command.
        first_executable_argument (str | None): Literal first argument, or None when unknown.
        first_path_inherited (bool): Whether supported prefixes preserve inherited PATH lookup.
        inherited_path_executables (tuple[tuple[str, bool], ...]): Literal executables with known
            inherited PATH, paired with whether their working directory is unchanged.
        first_executable_arguments (tuple[str, ...]): Literal argument prefix before any
            expansion, bounded independently of later shell commands.
    """

    findings: tuple[CommandFinding, ...]
    git_read: bool
    executables: tuple[str, ...]
    path_arguments: tuple[str, ...] = ()
    first_executable_literal: bool = False
    first_executable_initial: bool = False
    first_executable_argument: str | None = None
    first_path_inherited: bool = False
    inherited_path_executables: tuple[tuple[str, bool], ...] = ()
    first_executable_arguments: tuple[str, ...] = ()


class CommandScanResult(NamedTuple):
    """Carry one shell level's provisional findings and invocation hints.

    Args:
        findings (list[CommandFinding | None]): Matcher-aligned provisional findings.
        git_read (bool): Whether this level may read Git metadata.
        executables (list[str]): Candidate executable spellings.
        path_arguments (list[str]): Literal path hints without authority.
        first_literal (bool): Whether the first executable is literal.
        first_initial (bool): Whether it belongs to the initial simple command.
        first_argument (str | None): First literal argument, when known.
        first_path_inherited (bool): Whether the first command retains inherited PATH lookup.
        inherited_executables (list[tuple[str, bool]]): Literal inherited-PATH candidates and
            whether their working directory is unchanged.
        first_arguments (list[str]): Literal argument prefix of the first command.
    """

    findings: list[CommandFinding | None]
    git_read: bool
    executables: list[str]
    path_arguments: list[str]
    first_literal: bool
    first_initial: bool
    first_argument: str | None
    first_path_inherited: bool
    inherited_executables: list[tuple[str, bool]]
    first_arguments: list[str]


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
    PATH_CONTEXT_MUTATORS = frozenset(
        {
            ".",
            "builtin",
            "declare",
            "enable",
            "eval",
            "export",
            "fc",
            "getopts",
            "hash",
            "let",
            "local",
            "mapfile",
            "read",
            "readarray",
            "readonly",
            "source",
            "shopt",
            "trap",
            "typeset",
            "unset",
        }
    )
    SHELL_BUILTINS = frozenset(
        {
            ":",
            "[",
            "break",
            "cd",
            "command",
            "continue",
            "echo",
            "exec",
            "exit",
            "false",
            "getopts",
            "hash",
            "popd",
            "printf",
            "pushd",
            "pwd",
            "return",
            "set",
            "shift",
            "test",
            "true",
            "umask",
            "wait",
        }
    )
    SHELL_KEYWORDS = frozenset(
        {
            "if",
            "then",
            "else",
            "elif",
            "fi",
            "for",
            "while",
            "until",
            "do",
            "done",
            "case",
            "esac",
            "in",
            "{",
            "}",
        }
    )
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
        scan = self._scan(source, temporary_roots)
        return CommandAnalysis(
            tuple(finding for finding in scan.findings if finding is not None),
            scan.git_read,
            tuple(dict.fromkeys(scan.executables))[:128],
            tuple(sorted(dict.fromkeys(scan.path_arguments), key=lambda word: "/" not in word))[
                :128
            ],
            scan.first_literal,
            scan.first_initial,
            scan.first_argument,
            scan.first_path_inherited,
            tuple(dict.fromkeys(scan.inherited_executables))[:128],
            tuple(scan.first_arguments),
        )

    def _scan(
        self,
        source: str,
        temporary_roots: tuple[Path, ...],
        depth: int = 0,
    ) -> CommandScanResult:
        """Scan one shell level and merge bounded nested command classifications."""
        findings = [None] * len(self._matchers)
        git_read = False
        executables = []
        path_arguments = []
        first_literal = False
        first_initial = False
        first_argument = None
        first_arguments = []
        first_path_inherited = False
        inherited_executables = []
        path_stable = True
        cwd_stable = True
        try:
            tokens = shell_tokens(source)
        except ValueError:
            return CommandScanResult(
                findings,
                git_read,
                executables,
                path_arguments,
                first_literal,
                first_initial,
                None,
                first_path_inherited,
                inherited_executables,
                first_arguments,
            )
        start = 0
        for index in range(len(tokens) + 1):
            if index < len(tokens) and not (
                tokens[index].operator and tokens[index].value in self.SEPARATORS | {"\n"}
            ):
                continue
            segment = tokens[start:index]
            initial = start == 0
            start = index + 1
            word_tokens, review = self.strip_redirections(segment, temporary_roots)
            if review:
                self._add_destructive_finding(findings, redirection=True)
            command, ambiguous = self._strip_shell_prefixes([token.value for token in word_tokens])
            if ambiguous:
                self._add_destructive_finding(findings)
                path_stable = False
                continue
            expansion_changes_path = any(
                _path_mutating_expansion(token.raw) for token in segment if not token.operator
            )
            if expansion_changes_path:
                path_stable = False
            if not command:
                if any(token.value.startswith("PATH=") for token in word_tokens):
                    path_stable = False
                continue
            prefixes = word_tokens[: -len(command)]
            valid_prefixes = all(
                token.value in {"command", "exec"}
                or re.match(r"[A-Za-z_][A-Za-z0-9_]*=", token.raw) is not None
                for token in prefixes
            )
            inherited = (
                path_stable
                and valid_prefixes
                and not any(token.value.startswith("PATH=") for token in prefixes)
            )
            literal = literal_shell_word(word_tokens[-len(command)].raw)
            function_syntax = index < len(tokens) and tokens[index].value == "("
            if (
                inherited
                and literal
                and not function_syntax
                and command[0]
                not in self.PATH_CONTEXT_MUTATORS | self.SHELL_KEYWORDS | self.SHELL_BUILTINS
            ):
                inherited_executables.append((command[0], cwd_stable))
            if not executables:
                first_literal = literal
                first_initial = initial and not function_syntax and valid_prefixes
                first_path_inherited = inherited
                argument_start = len(word_tokens) - len(command) + 1
                for token in word_tokens[argument_start : argument_start + 128]:
                    if not literal_shell_word(token.raw):
                        break
                    first_arguments.append(token.value)
                first_argument = first_arguments[0] if first_arguments else None
            name = Path(command[0]).name
            executables.append(command[0])
            arguments = command[1:]
            shell_source = (
                arguments.index("-c") + 1 if name in self.SHELLS and "-c" in arguments else -1
            )
            path_arguments.extend(
                word
                for position, word in enumerate(arguments)
                if position != shell_source
                and word
                and not word.startswith(("-", "~"))
                and not any(char in word for char in "$`*?[")
            )
            if name == "git" and len(command) > 1:
                git_read = git_read or command[1] not in {"--version", "version", "help"}
            for position, matcher in enumerate(self._matchers):
                if findings[position] is None and (finding := matcher(command)) is not None:
                    findings[position] = finding
            if name in self.SHELLS and depth < 2 and 0 <= shell_source < len(arguments):
                nested = self._scan(arguments[shell_source], temporary_roots, depth + 1)
                git_read = git_read or nested.git_read
                executables.extend(nested.executables)
                path_arguments.extend(nested.path_arguments)
                if inherited and literal_shell_word(
                    word_tokens[-len(command) + 1 + shell_source].raw
                ):
                    inherited_executables.extend(
                        (name, cwd_stable and nested_cwd_stable)
                        for name, nested_cwd_stable in nested.inherited_executables
                    )
                for nested_position, finding in enumerate(nested.findings):
                    if findings[nested_position] is None:
                        findings[nested_position] = finding
            if command[0] in self.PATH_CONTEXT_MUTATORS | self.SHELL_KEYWORDS or function_syntax:
                path_stable = False
            if command[0] == "printf" and len(command) > 1 and command[1].startswith("-v"):
                path_stable = False
            if command[0] == "set" and _set_enables_keyword(command[1:]):
                path_stable = False
            if command[0] in {"cd", "pushd", "popd"} and command[1:] != ["."]:
                cwd_stable = False
        return CommandScanResult(
            findings,
            git_read,
            executables,
            path_arguments,
            first_literal,
            first_initial,
            first_argument,
            first_path_inherited,
            inherited_executables,
            first_arguments,
        )

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

    def strip_redirections(
        self,
        tokens: list[ShellToken],
        temporary_roots: tuple[Path, ...],
    ) -> tuple[list[ShellToken], bool]:
        """Separate command words from redirects and identify unproven write destinations.

        Args:
            tokens (list[ShellToken]): Tokens for one shell command segment.
            temporary_roots (tuple[Path, ...]): Canonical temporary directories approved for writes.

        Returns:
            tuple[list[ShellToken], bool]: Remaining words and whether a write needs review.
        """
        words = []
        last_word_position = -1
        review = False
        position = 0
        while position < len(tokens):
            token = tokens[position]
            if not token.operator or token.value not in self.REDIRECTIONS:
                words.append(token)
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
