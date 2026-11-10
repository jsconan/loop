"""Load an optional administrator policy from a fixed protected system directory."""

from __future__ import annotations

import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .. import constants
from ..utils import has_extended_access_acl, load_unique_json, sha256_digest
from .models import Decision, PermissionRule

_MAX_POLICY_BYTES = 65_536
_EXPECTATION_NAME = f"{constants.APP_NAME}.managed.expect.json"
_POLICY_NAME = "managed-policy.json"
_SYSTEM_DIRECTORIES = {
    "darwin": Path("/Library") / constants.APP_DISPLAY_NAME,
    "linux": Path("/etc") / constants.APP_NAME,
}
_OPEN_DIRECTORY = (
    getattr(os, "O_SEARCH", getattr(os, "O_PATH", os.O_RDONLY))
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | os.O_CLOEXEC
)
_OPEN_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


class ManagedPolicyError(ValueError):
    """Report an unsafe, missing, stale, or invalid configured administrator policy."""


class ManagedExpectation(BaseModel):
    """Bind a protected installation to one administrator policy identity and revision floor.

    Args:
        version (Literal[1]): Expectation schema version.
        identity (str): Stable administrator-selected policy identity.
        minimum_revision (int): Lowest acceptable policy revision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    identity: str = Field(pattern=r"^[0-9a-f]{32}$")
    minimum_revision: int = Field(ge=1)


class ManagedPolicy(BaseModel):
    """Hold restrictions that user policy and automatic approval cannot relax.

    Args:
        version (Literal[1]): Policy schema version.
        identity (str): Identity matching the protected installation expectation.
        revision (int): Monotonic administrator policy revision.
        require_sandbox (bool): Reject host process execution.
        forbid_host (bool): Reject host process execution independently of user limits.
        deny_local_networks (bool): Forbid sandbox-attempt loopback even when user policy allows it.
        forbid_yolo (bool): Prevent automatic no-prompt mode from becoming active.
        rules (tuple[PermissionRule, ...]): Deny or fresh-human rules for typed operations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    identity: str = Field(pattern=r"^[0-9a-f]{32}$")
    revision: int = Field(ge=1)
    require_sandbox: bool = False
    forbid_host: bool = False
    deny_local_networks: bool = False
    forbid_yolo: bool = False
    rules: tuple[PermissionRule, ...] = ()

    @model_validator(mode="after")
    def validate_rules(self) -> ManagedPolicy:
        """Reject ambiguous identifiers and decisions that could grant authority.

        Returns:
            ManagedPolicy: Validated restrictive policy.

        Raises:
            ValueError: A managed rule is ambiguous or attempts to allow an operation.
        """
        identifiers = [rule.id for rule in self.rules]
        if len(identifiers) != len(set(identifiers)) or any(
            rule.decision not in {Decision.DENY, Decision.ASK}
            or rule.source is not None
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", rule.id) is None
            for rule in self.rules
        ):
            raise ValueError("Managed rules require safe distinct IDs and deny or ask decisions.")
        return self


@dataclass(frozen=True)
class ManagedPolicySnapshot:
    """Bind one validated policy to the protected installation bytes.

    Args:
        policy (ManagedPolicy): Validated administrator restrictions.
        identity (tuple[object, ...]): Bound directory, file identities, and content digests.
    """

    policy: ManagedPolicy
    identity: tuple[object, ...]


class ManagedPolicyStore:
    """Read a fixed system policy through validated no-follow directory descriptors.

    The default location is selected by the platform, never from environment or workspace data.
    An explicit directory and trust anchor are for owned native test fixtures and trusted callers.

    Args:
        directory (Path | None): Trusted override of the platform installation directory.
        trust_anchor (Path): Already selected ancestor from which no-follow walking begins.
        owner_uid (int): Required owner of every opened path component and file.
    """

    _directory: Path | None
    _trust_anchor: Path
    _owner_uid: int
    _system_default: bool

    def __init__(
        self,
        directory: Path | None = None,
        *,
        trust_anchor: Path = Path("/"),
        owner_uid: int = 0,
    ) -> None:
        self._system_default = directory is None
        self._directory = (
            directory if directory is not None else _SYSTEM_DIRECTORIES.get(sys.platform)
        )
        self._trust_anchor = trust_anchor
        self._owner_uid = owner_uid

    @property
    def directory(self) -> Path | None:
        """Return the fixed installation directory on this platform.

        Returns:
            Path | None: Protected source location, or None on an unsupported platform.
        """
        return self._directory

    @property
    def expectation_path(self) -> Path | None:
        """Return the independent protected installation expectation path.

        Returns:
            Path | None: Fixed sibling expectation, or None on an unsupported platform.
        """
        return self._directory.with_name(_EXPECTATION_NAME) if self._directory else None

    def load(self) -> ManagedPolicySnapshot | None:
        """Read and validate the optional installed policy and protected expectation.

        Returns:
            ManagedPolicySnapshot | None: Bound policy, or None when the installation is absent.

        Raises:
            ManagedPolicyError: A configured source is unsafe, missing, stale, or malformed.
        """
        if self._directory is None:
            return None
        if self._system_default and not self.may_be_installed():
            return None
        try:
            parent, parent_ancestry = self._open_directory(self._directory.parent)
            if parent is None:
                raise ManagedPolicyError("Administrator expectation parent is unavailable.")
            try:
                try:
                    expectation_bytes, expectation_identity = self._read_file(
                        parent, _EXPECTATION_NAME
                    )
                except FileNotFoundError:
                    if self.may_be_installed():
                        raise ManagedPolicyError(
                            "Protected administrator policy expectation is missing."
                        ) from None
                    return None
            finally:
                os.close(parent)
            descriptor, ancestry = self._open_directory(self._directory)
            if descriptor is None:
                raise ManagedPolicyError("Expected administrator policy directory is missing.")
            try:
                policy_bytes, policy_identity = self._read_file(descriptor, _POLICY_NAME)
            finally:
                os.close(descriptor)
            current_parent, current_parent_ancestry = self._open_directory(self._directory.parent)
            if current_parent is None:
                raise ManagedPolicyError("Administrator expectation parent disappeared.")
            try:
                current_expectation, current_expectation_identity = self._read_file(
                    current_parent, _EXPECTATION_NAME
                )
            finally:
                os.close(current_parent)
            if (
                parent_ancestry != current_parent_ancestry
                or expectation_identity != current_expectation_identity
                or expectation_bytes != current_expectation
            ):
                raise ManagedPolicyError("Administrator policy expectation changed during loading.")
            current, current_ancestry = self._open_directory(self._directory)
            if current is None:
                raise ManagedPolicyError("Administrator policy directory disappeared.")
            try:
                current_policy, current_policy_identity = self._read_file(current, _POLICY_NAME)
            finally:
                os.close(current)
            if ancestry != current_ancestry:
                raise ManagedPolicyError("Administrator policy directory changed during loading.")
            if policy_identity != current_policy_identity or policy_bytes != current_policy:
                raise ManagedPolicyError("Administrator policy file changed during loading.")
            expectation = ManagedExpectation.model_validate(
                load_unique_json(
                    expectation_bytes, duplicate_message="Duplicate administrator policy field."
                )
            )
            raw_policy = load_unique_json(
                policy_bytes, duplicate_message="Duplicate administrator policy field."
            )
            if any("id" not in rule for rule in raw_policy.get("rules", ())):
                raise ManagedPolicyError("Managed rules require explicit stable identifiers.")
            policy = ManagedPolicy.model_validate(raw_policy)
            if (
                policy.identity != expectation.identity
                or policy.revision < expectation.minimum_revision
            ):
                raise ManagedPolicyError("Administrator policy identity or revision is stale.")
            return ManagedPolicySnapshot(
                policy,
                (
                    parent_ancestry,
                    ancestry,
                    expectation_identity,
                    sha256_digest(expectation_bytes),
                    policy_identity,
                    sha256_digest(policy_bytes),
                ),
            )
        except (OSError, ValueError, TypeError, AttributeError) as error:
            if isinstance(error, ManagedPolicyError):
                raise
            raise ManagedPolicyError(f"Administrator policy cannot be verified: {error}") from error

    def may_be_installed(self) -> bool:
        """Detect a present source without interpreting inspection failure as absence.

        Returns:
            bool: Whether either protected entry exists or cannot be safely inspected.
        """
        if self._directory is None:
            return False
        try:
            components = self._directory.relative_to(self._trust_anchor).parts[:-1]
        except ValueError:
            return True
        parent = self._trust_anchor
        for component in components:
            parent /= component
            try:
                details = parent.lstat()
            except PermissionError:
                return not self._verified_absence()
            except OSError:
                return True
            if (
                not stat.S_ISDIR(details.st_mode)
                or details.st_uid != self._owner_uid
                or details.st_mode & 0o7022
            ):
                return True
        for path in (self.expectation_path, self._directory):
            try:
                path.lstat()
            except PermissionError:
                return not self._verified_absence()
            except FileNotFoundError:
                continue
            except OSError:
                return True
            return True
        return not self._verified_absence()

    def _verified_absence(self) -> bool:
        """Prove both names absent relative to validated, searchable ancestry."""
        try:
            descriptor, ancestry = self._open_directory(self._directory.parent)
            if descriptor is None:
                return False
            try:
                for name in (_EXPECTATION_NAME, self._directory.name):
                    try:
                        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    return False
                current, current_ancestry = self._open_directory(self._directory.parent)
                if current is None:
                    return False
                os.close(current)
                return ancestry == current_ancestry
            finally:
                os.close(descriptor)
        except (OSError, ManagedPolicyError):
            return False

    def _open_directory(self, directory: Path) -> tuple[int | None, tuple[tuple[int, ...], ...]]:
        """Walk only real protected directories and return their bound identities."""
        if not directory.is_absolute() or ".." in directory.parts:
            raise ManagedPolicyError("Invalid administrator policy installation path.")
        try:
            components = directory.relative_to(self._trust_anchor).parts
        except ValueError as error:
            raise ManagedPolicyError("Policy directory escapes its trust anchor.") from error
        descriptor = os.open(self._trust_anchor, _OPEN_DIRECTORY)
        identities = []
        try:
            identities.append(self._validate_descriptor(descriptor, directory=True))
            for index, component in enumerate(components):
                try:
                    child = os.open(component, _OPEN_DIRECTORY, dir_fd=descriptor)
                except FileNotFoundError:
                    if index == len(components) - 1:
                        return None, tuple(identities)
                    raise ManagedPolicyError("Administrator policy path is incomplete.") from None
                try:
                    identities.append(self._validate_descriptor(child, directory=True))
                except BaseException:
                    os.close(child)
                    raise
                os.close(descriptor)
                descriptor = child
            result = descriptor
            descriptor = -1
            return result, tuple(identities)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _read_file(self, directory: int, name: str) -> tuple[bytes, tuple[int, ...]]:
        """Read bounded bytes and verify the opened file remains at the fixed name."""
        descriptor = os.open(name, _OPEN_FILE, dir_fd=directory)
        try:
            before = self._validate_descriptor(descriptor, directory=False)
            content = bytearray()
            while chunk := os.read(descriptor, _MAX_POLICY_BYTES + 1 - len(content)):
                content.extend(chunk)
                if len(content) > _MAX_POLICY_BYTES:
                    break
            if len(content) > _MAX_POLICY_BYTES:
                raise ManagedPolicyError("Administrator policy file exceeds its size limit.")
            after = self._validate_descriptor(descriptor, directory=False)
            named = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if before != after or before != self._identity(named):
                raise ManagedPolicyError("Administrator policy file changed during loading.")
            return bytes(content), before
        finally:
            os.close(descriptor)

    def _validate_descriptor(self, descriptor: int, *, directory: bool) -> tuple[int, ...]:
        """Reject non-root, writable, linked, ACL-bearing, or wrong-type authority."""
        details = os.fstat(descriptor)
        if (
            details.st_uid != self._owner_uid
            or details.st_mode & 0o7022
            or not (stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode))
            or (not directory and details.st_nlink != 1)
            or has_extended_access_acl(descriptor)
        ):
            raise ManagedPolicyError("Administrator policy path has unsafe ownership or access.")
        return self._identity(details)

    @staticmethod
    def _identity(details: os.stat_result) -> tuple[int, ...]:
        """Bind file type, owner, inode, content metadata, and change time."""
        return (
            details.st_dev,
            details.st_ino,
            details.st_mode,
            details.st_uid,
            details.st_nlink,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
        )
