"""Translate model-visible paths without filesystem-ignore or application dependencies."""

import json
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from .process import parse_command_line


class VirtualPath:
    """Translate canonical tool path prefixes to local filesystem roots.

    Args:
        workspace (Path | None): Physical workspace root addressed by ``$WORKSPACE_ROOT``.
        temporary_directory (Path | None): Physical session scratch addressed by ``$TMPDIR``.
        skill_roots (Mapping[str, Path] | None): Skill resource directories represented below
            ``/skills/<name>``.

    Typed paths resolve before filesystem operations. Shell source and saved file content
    remain unchanged; executed programs obtain the same roots from their environment.
    """

    WORKSPACE = "$WORKSPACE_ROOT"
    TEMPORARY = "$TMPDIR"
    SKILLS = "/skills"
    EXTERNAL = "<external>"

    _roots: dict[str, Path]

    def __init__(
        self,
        workspace: Path | None = None,
        temporary_directory: Path | None = None,
        skill_roots: Mapping[str, Path] | None = None,
    ) -> None:
        roots = {}
        if workspace is not None:
            roots[self.WORKSPACE] = workspace.absolute()
        if temporary_directory is not None:
            roots[self.TEMPORARY] = temporary_directory.absolute()
        roots.update(
            (f"{self.SKILLS}/{name}", root.absolute()) for name, root in (skill_roots or {}).items()
        )
        self._roots = roots

    def resolve(self, value: str) -> str:
        """Resolve a canonical prefix or workspace-relative tool path.

        Args:
            value (str): Model-supplied path within a configured root.

        Returns:
            str: Local lexical path, preserving leaf symlinks for mutation planners.

        Raises:
            ValueError: The path is outside a configured root or escapes its root.
        """
        if not value or "//" in value:
            raise ValueError("Path must be a virtual or workspace-relative path.")
        if value.startswith(("workspace:", "scratch:", "skill:")):  # Detect deprecated path aliases
            raise ValueError("Path aliases are not supported; use a VirtualPath.")
        if value.startswith(("/", "$")):
            for virtual_root, local_root in self._roots.items():
                if value == virtual_root or value.startswith(f"{virtual_root}/"):
                    return self._local_path(local_root, value[len(virtual_root) :].lstrip("/"))
            raise ValueError("Absolute paths must be inside a configured virtual root.")
        workspace = self._roots.get(self.WORKSPACE)
        if workspace is None:
            raise ValueError("Workspace-relative paths require a configured workspace.")
        return self._local_path(workspace, value)

    def display(self, value: str) -> str:
        """Render a local path using its configured public spelling.

        Args:
            value (str): Local path or relative metadata path.

        Returns:
            str: Canonical prefix, relative path, or ``<external>``.
        """
        path = Path(value)
        if not path.is_absolute():
            return value
        for virtual_root, local_root in sorted(
            self._roots.items(), key=lambda pair: len(str(pair[1])), reverse=True
        ):
            if path.is_relative_to(local_root):
                relative = path.relative_to(local_root).as_posix()
                return virtual_root if relative == "." else f"{virtual_root}/{relative}"
        return self.EXTERNAL

    def resolve_command(self, command: str) -> str:
        """Resolve VirtualPath arguments in one restricted command line.

        Args:
            command (str): Model-supplied shell-free command line.

        Returns:
            str: Equivalent command line with virtual path arguments replaced by local paths.

        Raises:
            ValueError: The command is malformed or a virtual argument escapes its root.
        """
        argv = parse_command_line(command)
        return shlex.join(tuple(self._resolve_command_argument(argument) for argument in argv))

    def command_roots(self) -> tuple[tuple[str, Path], ...]:
        """Return canonical root keys and physical targets for request binding.

        Returns:
            tuple[tuple[str, Path], ...]: Roots ordered from longest virtual name to shortest.
        """
        return tuple(sorted(self._roots.items(), key=lambda pair: len(pair[0]), reverse=True))

    @staticmethod
    def translate_shell_source(source: str, aliases: Mapping[str, Path]) -> str:
        """Replace declared literal virtual roots without reconstructing POSIX shell syntax.

        Args:
            source (str): Opaque model-authored POSIX shell source.
            aliases (Mapping[str, Path]): Configured virtual roots mapped to shell-safe aliases.

        Returns:
            str: Source with path-segment root occurrences replaced in one pass. A bare
                directory root gains a trailing slash so tools traverse the alias symlink.
            Environment-variable prefixes are excluded because the shell expands their values.
        """
        if not aliases:
            return source
        aliases = {root: path for root, path in aliases.items() if not root.startswith("$")}
        if not aliases:
            return source
        roots = "|".join(re.escape(root) for root in sorted(aliases, key=len, reverse=True))
        pattern = re.compile(rf"(?<![\w./~%-])({roots})(?=/|$|[^\w.-])")
        return pattern.sub(
            lambda match: (
                str(aliases[match.group(1)])
                + ("" if match.end() < len(source) and source[match.end()] == "/" else "/")
            ),
            source,
        )

    def metadata(self, value: object, fields: tuple[tuple[str, ...], ...]) -> object:
        """Render declared metadata fields using their configured public paths.

        Args:
            value (object): Parsed tool result or reference metadata.
            fields (tuple[tuple[str, ...], ...]): Declared key paths; ``"*"`` selects list items.

        Returns:
            object: Copy with declared metadata path values using public spellings.
        """

        def rewrite(item: object, path: tuple[str, ...]) -> object:
            if not path:
                return self.display(item) if isinstance(item, str) else item
            head, *tail = path
            if head == "*" and isinstance(item, list):
                return [rewrite(child, tuple(tail)) for child in item]
            if isinstance(item, dict) and head in item:
                return {**item, head: rewrite(item[head], tuple(tail))}
            return item

        prepared = value
        for field in fields:
            prepared = rewrite(prepared, field)
        return prepared

    def redact(
        self,
        value: str,
        *,
        roots: tuple[tuple[str, str], ...] | None = None,
    ) -> str:
        """Render configured roots as canonical prefixes in diagnostics.

        Args:
            value (str): Diagnostic text that may include one declared root.
            roots (tuple[tuple[str, str], ...] | None): Precomputed local and public root pairs.
                Defaults to resolving canonical root aliases on demand.

        Returns:
            str: Text with canonical prefixes and unrelated host ancestors sanitized.
        """
        selected_roots = self.redaction_roots() if roots is None else roots
        replacements = {}
        for local, virtual in selected_roots:
            for spelling in sorted(
                {
                    local,
                    shlex.quote(local),
                    re.sub(r"([^\w/.-])", r"\\\1", local),
                    local.replace("/", "\\/"),
                    json.dumps(local, ensure_ascii=True)[1:-1],
                    json.dumps(local, ensure_ascii=False)[1:-1],
                },
                key=len,
                reverse=True,
            ):
                replacements.setdefault(spelling, virtual)
        if not replacements:
            return value
        pattern = "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True))
        return re.sub(pattern, lambda match: replacements[match.group()], value)

    def redaction_roots(self) -> tuple[tuple[str, str], ...]:
        """Return lexical and canonical roots for public-path disclosure.

        Returns:
            tuple[tuple[str, str], ...]: Local path spellings paired with public roots,
                ordered from longest to shortest for safe replacement.
        """
        roots = {}
        for virtual, root in self._roots.items():
            for local in (root, root.resolve(strict=False)):
                for parent in local.parents:
                    if len(parent.parts) > 2:
                        roots.setdefault(str(parent), self.EXTERNAL)
                roots[str(local)] = virtual
                # Diagnostic tools commonly abbreviate an account's home with a tilde.
                for candidate in (local, *local.parents):
                    parts = candidate.parts
                    if len(parts) > 3 and parts[1] in {"Users", "home"}:
                        roots["~/" + "/".join(parts[3:])] = (
                            virtual if candidate == local else self.EXTERNAL
                        )
        return tuple(sorted(roots.items(), key=lambda pair: len(pair[0]), reverse=True))

    def _resolve_command_argument(self, argument: str) -> str:
        """Resolve an exact virtual argument or the value portion of an option."""
        value = argument
        prefix = ""
        if argument.startswith("-") and "=" in argument:
            prefix, value = argument.split("=", maxsplit=1)
            prefix += "="
        if any(value == root or value.startswith(f"{root}/") for root in self._roots):
            return prefix + self.resolve(value)
        return argument

    @staticmethod
    def _local_path(root: Path, suffix: str) -> str:
        """Join one virtual suffix to its local root without allowing traversal."""
        if ".." in PurePosixPath(suffix).parts:
            raise ValueError("Virtual path escapes its root.")
        candidate = Path(os.path.abspath(root / suffix))
        return str(candidate)
