"""Expose OCI preparation and attestation behind a fail-closed boundary."""

from __future__ import annotations

from dataclasses import dataclass

from ...contracts import BackendAttestation, SandboxExecutionRequest
from ...results import InfrastructureFailure
from .attestation import OciAttestationError, attest_inspection
from .spec import OciExecutionSpec, OciSpecCompiler


@dataclass(frozen=True, slots=True)
class OciPreparedAttempt:
    """Bind a compiled attempt to its expected private runtime evidence.

    Args:
        spec: Closed OCI invocation specification.
        namespace: Private namespace that must appear in inspection evidence.
        platform_os: Selected runtime operating system.
        platform_architecture: Selected runtime architecture.
    """

    spec: OciExecutionSpec
    namespace: str
    platform_os: str
    platform_architecture: str


class OciSandboxAdapter:
    """Prepare and attest OCI attempts without a host-execution route.

    This boundary only accepts a compiled attempt after closed inspection succeeds.

    Args:
        compiler: Compiler for the selected immutable OCI profile.
        namespace: Attested private containerd namespace.
        platform_os: Attested runtime operating system.
        platform_architecture: Attested runtime architecture.
    """

    _compiler: OciSpecCompiler
    _namespace: str
    _platform_os: str
    _platform_architecture: str

    def __init__(
        self,
        compiler: OciSpecCompiler,
        namespace: str,
        platform_os: str,
        platform_architecture: str,
    ) -> None:
        self._compiler = compiler
        self._namespace = namespace
        self._platform_os = platform_os
        self._platform_architecture = platform_architecture

    def prepare(
        self, request: SandboxExecutionRequest, workspace_generation: str
    ) -> OciPreparedAttempt:
        """Compile one authorized request into an OCI attempt.

        Args:
            request: Authorized virtual sandbox request.
            workspace_generation: Opaque immutable generation identity.

        Returns:
            OciPreparedAttempt: Closed settings awaiting typed lifecycle management.
        """
        return OciPreparedAttempt(
            self._compiler.compile(request, workspace_generation),
            self._namespace,
            self._platform_os,
            self._platform_architecture,
        )

    def attest(
        self, prepared: OciPreparedAttempt, evidence: bytes
    ) -> BackendAttestation | InfrastructureFailure:
        """Verify evidence or return the closed infrastructure-failure result.

        Args:
            prepared: Prepared attempt whose identity must be attested.
            evidence: Normalized pinned-version inspector JSON.

        Returns:
            BackendAttestation | InfrastructureFailure: Attestation or fail-closed result.
        """
        try:
            return attest_inspection(
                evidence,
                prepared.spec,
                prepared.namespace,
                prepared.platform_os,
                prepared.platform_architecture,
            )
        except OciAttestationError:
            return InfrastructureFailure(request_id=prepared.spec.request_id)
