"""Test closed macOS managed-runtime requirement verification."""

from __future__ import annotations

import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.sandbox.macos import MacosRequirementError, MacosRequirements


class _Runner:
    """Return fixed sealed-codesign evidence without starting a process."""

    application_data: Path
    result: InfrastructureProcessResult

    def __init__(self, application_data: Path, payload: dict[str, bool]) -> None:
        self.application_data = application_data
        self.result = InfrastructureProcessResult(0, plistlib.dumps(payload), b"", False, False)

    def verify_codesign(self, executable: object) -> InfrastructureProcessResult:
        """Return the configured sealed evidence."""
        return self.result


class _FailingRunner:
    """Reject sealed signature inspection without disclosing its implementation detail."""

    application_data: Path

    def __init__(self, application_data: Path) -> None:
        self.application_data = application_data

    def verify_codesign(self, executable: object) -> InfrastructureProcessResult:
        """Fail the sealed signature boundary."""
        raise RuntimeError("untrusted native diagnostic")


def _requirements(tmp_path: Path, payload: dict[str, bool]) -> MacosRequirements:
    """Build requirements with one candidate-recorded entitlement set."""
    return MacosRequirements(_Runner(tmp_path, payload), 13, 1, frozenset(payload))  # type: ignore[arg-type]


def test_requirements_accept_sealed_apple_silicon_virtualization_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A current Apple-Silicon host accepts exactly the candidate entitlement set."""
    payload = {
        "com.apple.security.virtualization": True,
        "com.apple.security.network.client": True,
    }
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.mac_ver", lambda: ("26.7", ("", "", ""), ""))
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    evidence = _requirements(tmp_path, payload).verify(SimpleNamespace())
    assert evidence.entitlement_keys == tuple(sorted(payload))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "entitlements"),
        ({"com.apple.security.virtualization": False}, "virtualization"),
        (
            {"com.apple.security.virtualization": True, "unexpected": True},
            "entitlements",
        ),
    ],
)
def test_requirements_reject_invalid_entitlement_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, bool], message: str
):
    """Missing, false, and unexpected entitlements fail closed."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.mac_ver", lambda: ("26.7", ("", "", ""), ""))
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    candidate = {"com.apple.security.virtualization"}
    requirements = MacosRequirements(_Runner(tmp_path, payload), 13, 1, candidate)  # type: ignore[arg-type]
    with pytest.raises(MacosRequirementError, match=message):
        requirements.verify(SimpleNamespace())


def test_requirements_reject_unsupported_host_before_codesign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-Apple-Silicon and unsupported macOS versions fail before launching a probe."""
    requirements = _requirements(tmp_path, {"com.apple.security.virtualization": True})
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "amd64")
    monkeypatch.setattr("platform.mac_ver", lambda: ("", ("", "", ""), ""))
    with pytest.raises(MacosRequirementError, match="Apple Silicon"):
        requirements.verify(SimpleNamespace())


@pytest.mark.parametrize(("minimum_version", "free_bytes"), [(12, 1), (13, 0)])
def test_requirements_reject_invalid_configuration_limits(
    tmp_path: Path, minimum_version: int, free_bytes: int
) -> None:
    """The requirement verifier cannot be configured below its VZ/storage floor."""
    with pytest.raises(ValueError, match="limits"):
        MacosRequirements(
            _Runner(tmp_path, {"com.apple.security.virtualization": True}),
            minimum_version,
            free_bytes,
            frozenset({"com.apple.security.virtualization"}),
        )  # type: ignore[arg-type]


def test_requirements_rejects_unsupported_macos_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Apple-Silicon host below the candidate's VZ floor is unsupported."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.mac_ver", lambda: ("12.7", ("", "", ""), ""))
    with pytest.raises(MacosRequirementError, match="version"):
        _requirements(tmp_path, {"com.apple.security.virtualization": True}).verify(
            SimpleNamespace()
        )


@pytest.mark.parametrize(
    ("framework", "available_bytes", "message"),
    [(False, 2, "Virtualization"), (True, 0, "storage")],
)
def test_requirements_reject_missing_framework_or_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: bool,
    available_bytes: int,
    message: str,
) -> None:
    """Mandatory VZ and private-storage predicates fail closed before codesign."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.mac_ver", lambda: ("26.7", ("", "", ""), ""))
    monkeypatch.setattr(Path, "is_dir", lambda path: framework)
    monkeypatch.setattr(
        "os.statvfs", lambda path: SimpleNamespace(f_bavail=available_bytes, f_frsize=1)
    )
    requirements = _requirements(tmp_path, {"com.apple.security.virtualization": True})
    with pytest.raises(MacosRequirementError, match=message):
        requirements.verify(SimpleNamespace())


def test_requirements_sanitizes_sealed_signature_and_malformed_plist_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native verifier failures never disclose their raw diagnostic to callers."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.mac_ver", lambda: ("26.7", ("", "", ""), ""))
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    for runner in (
        _FailingRunner(tmp_path),
        _Runner(tmp_path, {"com.apple.security.virtualization": "not-a-boolean"}),
    ):
        requirements = MacosRequirements(
            runner, 13, 1, frozenset({"com.apple.security.virtualization"})
        )  # type: ignore[arg-type]
        with pytest.raises(MacosRequirementError, match="signature evidence") as error:
            requirements.verify(SimpleNamespace())
        assert "untrusted native diagnostic" not in str(error.value)
