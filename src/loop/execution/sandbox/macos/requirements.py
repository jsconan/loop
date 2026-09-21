"""Verify the closed macOS prerequisites for Loop's managed Lima runtime."""

from __future__ import annotations

import os
import platform
import plistlib
from dataclasses import dataclass
from pathlib import Path

from ...runtime.bootstrap import InstalledExecutable
from ...runtime.manifest import RuntimeManifest
from .candidate import macos_entitlement_keys
from .lima import LimaClient


class MacosRequirementError(RuntimeError):
    """Report a failed macOS substrate prerequisite without exposing host paths."""


@dataclass(frozen=True, slots=True)
class MacosRequirementEvidence:
    """Record the sanitized result of one successful macOS requirement probe.

    Args:
        entitlement_keys (tuple[str, ...]): Exact sealed entitlement keys observed on Lima.
        code_directory_hash (str | None): Informational signature identity when available.
    """

    entitlement_keys: tuple[str, ...]
    code_directory_hash: str | None


class MacosRequirements:
    """Verify immutable Lima and host predicates required before VZ lifecycle work.

    Args:
        runner (LimaClient): macOS client owning the fixed codesign invocation.
        minimum_macos_major (int): Minimum supported macOS major version.
        minimum_free_bytes (int): Minimum private runtime storage capacity.
        allowed_entitlements (frozenset[str]): Complete candidate-recorded Lima entitlement set.
    """

    runner: LimaClient
    minimum_macos_major: int
    minimum_free_bytes: int
    allowed_entitlements: frozenset[str]

    def __init__(
        self,
        runner: LimaClient,
        minimum_macos_major: int,
        minimum_free_bytes: int,
        allowed_entitlements: frozenset[str],
    ) -> None:
        if minimum_macos_major < 13 or minimum_free_bytes <= 0:
            raise ValueError("macOS requirement limits are invalid.")
        self.runner = runner
        self.minimum_macos_major = minimum_macos_major
        self.minimum_free_bytes = minimum_free_bytes
        self.allowed_entitlements = allowed_entitlements

    @classmethod
    def from_candidate(
        cls,
        runner: LimaClient,
        candidate: RuntimeManifest,
        minimum_macos_major: int,
        minimum_free_bytes: int,
    ) -> MacosRequirements:
        """Create requirement verification bound to one checked-in candidate.

        Args:
            runner (LimaClient): macOS client owning the fixed codesign invocation.
            candidate (RuntimeManifest): Closed candidate providing expected entitlements.
            minimum_macos_major (int): Minimum supported macOS major version.
            minimum_free_bytes (int): Minimum required private runtime storage.

        Returns:
            MacosRequirements: Requirement verifier constrained by ``candidate``.
        """
        return cls(
            runner,
            minimum_macos_major,
            minimum_free_bytes,
            macos_entitlement_keys(candidate),
        )

    def verify(self, executable: InstalledExecutable) -> MacosRequirementEvidence:
        """Verify the current host and one leased Lima executable.

        Args:
            executable (InstalledExecutable): Verified installed Lima executable.

        Returns:
            MacosRequirementEvidence: Sanitized signature-entitlement evidence.

        Raises:
            MacosRequirementError: If a host or signed-Lima requirement is not met.
        """
        version = platform.mac_ver()[0]
        if platform.system() != "Darwin" or platform.machine() != "arm64" or not version:
            raise MacosRequirementError("Managed macOS runtime requires Apple Silicon.")
        if int(version.split(".", maxsplit=1)[0]) < self.minimum_macos_major:
            raise MacosRequirementError("macOS version does not support the managed VZ runtime.")
        if not Path("/System/Library/Frameworks/Virtualization.framework").is_dir():
            raise MacosRequirementError("Virtualization.framework is unavailable.")
        if (
            os.statvfs(self.runner.application_data).f_bavail
            * os.statvfs(self.runner.application_data).f_frsize
            < self.minimum_free_bytes
        ):
            raise MacosRequirementError("Private runtime storage is insufficient.")
        try:
            result = self.runner.verify_codesign(executable)
            payload = plistlib.loads(result.stdout)
        except (RuntimeError, ValueError, plistlib.InvalidFileException) as error:
            raise MacosRequirementError("Installed Lima signature evidence is invalid.") from error
        if not isinstance(payload, dict) or not all(
            isinstance(value, bool) for value in payload.values()
        ):
            raise MacosRequirementError("Installed Lima signature evidence is invalid.")
        if set(payload) != self.allowed_entitlements:
            raise MacosRequirementError("Installed Lima entitlements do not match the candidate.")
        if payload.get("com.apple.security.virtualization") is not True:
            raise MacosRequirementError("Installed Lima lacks the virtualization entitlement.")
        return MacosRequirementEvidence(tuple(sorted(payload)), None)
