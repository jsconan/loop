"""Test typed execution authority through the real command-tool flow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from loop.execution import Capability, ExecutionService
from loop.execution.command import SandboxCommandExecutor
from loop.execution.contracts import (
    DeltaEffect,
    ExecutionBoundary,
    JobHandle,
    JobOperation,
    NetworkConnectionLease,
    NetworkListenerLease,
    NetworkProtocol,
    SandboxExecutionRequest,
    SecretExposure,
    SecretMechanism,
)
from loop.execution.results import (
    CapabilityDenied,
    Completed,
    InfrastructureFailure,
    UnsupportedCapability,
)
from loop.execution.service import AttemptObserver
from loop.execution.vfs import CanonicalDelta, CanonicalDeltaEntry, ObjectKind
from loop.interaction import Interaction
from loop.permissions import (
    ApprovalChoice,
    CommandProfile,
    ExecutionPermissionAdapter,
    GrantAuthority,
    GrantConstraints,
    GrantDecision,
    GrantEffect,
    GrantScope,
    NetworkEndpoint,
    NetworkListener,
    OpaqueResource,
    PermissionConfiguration,
    PermissionManager,
    PolicyGrant,
    PolicyRequest,
    SecretUse,
    SubjectIdentity,
    UserPermissionConfiguration,
    VirtualPathTree,
    evaluate_grant,
)
from loop.permissions import execution_adapter as adapter_module
from loop.tools import create_default_tool_registry


def _subject() -> SubjectIdentity:
    """Return the authenticated ordinary-shell subject used by product tests."""
    return SubjectIdentity(
        tool_id="run_command",
        publisher="loop",
        profile_id="ordinary-shell",
        profile_version="1",
    )


def _request(**changes: object) -> PolicyRequest:
    """Build one exact workspace replacement request."""
    values: dict[str, object] = {
        "subject": _subject(),
        "boundary": ExecutionBoundary.SANDBOX,
        "effect": GrantEffect.FS_REPLACE,
        "resource": VirtualPathTree(
            root="/workspace/docs/readme", effects=frozenset({GrantEffect.FS_REPLACE})
        ),
        "workspace_id": "workspace",
        "policy_version": "2",
        "now_ns": 10,
        "scope_binding": "agent",
    }
    values.update(changes)
    return PolicyRequest(**values)


def _grant(identifier: str, decision: GrantDecision, **changes: object) -> PolicyGrant:
    """Build one matching typed grant."""
    values: dict[str, object] = {
        "grant_id": identifier,
        "subject": _subject(),
        "boundary": ExecutionBoundary.SANDBOX,
        "effect": GrantEffect.FS_REPLACE,
        "resource": VirtualPathTree(
            root="/workspace/docs/**", effects=frozenset({GrantEffect.FS_REPLACE})
        ),
        "decision": decision,
        "authority": GrantAuthority.USER_ALLOW,
        "scope": GrantScope.WORKSPACE,
        "workspace_binding": "workspace",
        "policy_version": "2",
        "issued_at_ns": 1,
        "origin": "user",
    }
    values.update(changes)
    return PolicyGrant(**values)


class _Verifier:
    """Accept one deterministic profile signature."""

    def verify(self, publisher: str, payload: bytes, signature: bytes) -> bool:
        """Return whether the expected trusted tuple was supplied."""
        return publisher == "loop" and bool(payload) and signature == b"signature"


class _SandboxAdapter:
    """Model stopped-overlay publication while retaining the real service boundary."""

    def __init__(self, permissions: ExecutionPermissionAdapter) -> None:
        self.permissions = permissions
        self.requests: list[SandboxExecutionRequest] = []
        self.host_starts = 0

    def execute(self, request, observer: AttemptObserver, cancellation):
        """Return output or authorize one exact observed write."""
        del cancellation
        self.requests.append(request)
        observer.backend_attested()
        observer.process_started()
        if request.script == "infrastructure-failure":
            return InfrastructureFailure(request_id=request.request_id, diagnostic_id="opaque")
        if request.script.startswith("write "):
            path = request.script.removeprefix("write ")
            delta = CanonicalDelta(
                entries=(
                    CanonicalDeltaEntry(
                        effect=DeltaEffect.REPLACE,
                        destination_path=path,
                        object_kind=ObjectKind.FILE,
                        mode=0o644,
                    ),
                )
            )
            if not self.permissions.authorize_publication(request, delta):
                return CapabilityDenied(
                    request_id=request.request_id,
                    capability=Capability.WORKSPACE_WRITE.value,
                )
        return Completed(request_id=request.request_id, exit_code=0, stdout=b"sandbox\n")


def _command_registry(
    tmp_path: Path,
    *,
    workspace_id: str = "workspace",
    interaction: Interaction | None = None,
    configuration: PermissionConfiguration | None = None,
):
    """Compose the actual command tool with canonical permissions and a sandbox service."""
    manager = PermissionManager(
        tmp_path,
        configuration_path=tmp_path / "permissions.yaml",
        workspace_id=workspace_id,
        interaction=interaction,
        configuration=configuration,
    )
    permissions = ExecutionPermissionAdapter(
        manager,
        _subject(),
        workspace_id,
        "2",
        "agent",
        clock_ns=lambda: 10,
    )
    adapter = _SandboxAdapter(permissions)
    executor = SandboxCommandExecutor(
        ExecutionService(adapter),
        permissions,
        workspace_id,
        "agent",
        "sha256:runtime",
        supports_network_effects=True,
        supports_secret_exposures=True,
    )
    return (
        create_default_tool_registry(
            interaction=interaction,
            permission_manager=manager,
            command_executor=executor,
        ),
        manager,
        adapter,
    )


def _run(registry, command: str) -> dict:
    """Call the registered public command interface and decode its envelope."""
    return json.loads(
        registry.call("run_command", json.dumps({"command": command}), call_id="request")
    )


def test_actual_command_flow_reads_without_prompt_and_uses_natural_shell_source(tmp_path: Path):
    """Default sandbox reads avoid prompts and preserve opaque shell syntax."""
    interaction = Mock(spec=Interaction)
    registry, _, adapter = _command_registry(tmp_path, interaction=interaction)

    result = _run(registry, "printf x | sed s/x/y/ > /tmp/result")

    assert result["result"]["stdout"]["content"] == "sandbox\n"
    assert adapter.requests[0].script == "printf x | sed s/x/y/ > /tmp/result"
    interaction.prompt.assert_not_called()
    assert adapter.host_starts == 0


def test_actual_command_flow_reuses_scoped_write_and_honors_revocation(tmp_path: Path):
    """Persist exact observed authority, reuse it, then require approval after revocation."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    registry, manager, _ = _command_registry(tmp_path, interaction=interaction)

    assert _run(registry, "write docs/readme")["ok"] is True
    assert interaction.prompt.call_count == 1
    grant_id = manager.execution_grants[0].grant_id
    assert _run(registry, "write docs/readme")["ok"] is True
    assert interaction.prompt.call_count == 1

    assert manager.revoke_execution_grant(grant_id)
    interaction.prompt.return_value = ApprovalChoice.DENY
    denied = _run(registry, "write docs/readme")

    assert denied["problem"]["code"] == "process.capability_denied"
    assert interaction.prompt.call_count == 2


def test_actual_command_flow_deny_precedence_clone_binding_and_no_host_conversion(tmp_path: Path):
    """Deny beats allow, copied authority cannot cross workspaces, and failures stay sandboxed."""
    allow = _grant("allow", GrantDecision.ALLOW)
    deny = _grant("deny", GrantDecision.DENY, authority=GrantAuthority.USER_DENY)
    configuration = PermissionConfiguration(version=2, execution_grants=[allow, deny])
    registry, _, adapter = _command_registry(tmp_path / "denied", configuration=configuration)

    assert _run(registry, "write docs/readme")["problem"]["code"] == ("process.capability_denied")
    assert _run(registry, "infrastructure-failure")["problem"]["code"] == (
        "process.infrastructure_failure"
    )
    assert adapter.host_starts == 0

    clone_registry, _, clone_adapter = _command_registry(
        tmp_path / "clone", workspace_id="clone", configuration=configuration
    )
    assert _run(clone_registry, "write docs/readme")["problem"]["code"] == (
        "process.capability_denied"
    )
    assert clone_adapter.host_starts == 0


def test_typed_matching_rejects_expiry_identity_constraints_and_prefers_narrow_rules():
    """Retain deny dominance and fail closed on every typed binding."""
    broad = _grant("broad", GrantDecision.ALLOW)
    exact = _grant(
        "exact",
        GrantDecision.ALLOW,
        resource=VirtualPathTree(
            root="/workspace/docs/readme", effects=frozenset({GrantEffect.FS_REPLACE})
        ),
    )
    assert evaluate_grant((broad, exact), _request()).matched_grant_id == "exact"
    assert (
        evaluate_grant(
            (_grant("expired", GrantDecision.ALLOW, expires_at_ns=10),), _request()
        ).decision
        is GrantDecision.PROMPT
    )
    assert (
        evaluate_grant((_grant("revoked", GrantDecision.ALLOW, revoked=True),), _request()).decision
        is GrantDecision.PROMPT
    )
    bounded = _grant("bounded", GrantDecision.ALLOW, constraints=GrantConstraints(max_bytes=3))
    assert (
        evaluate_grant((bounded,), _request(constraints=GrantConstraints(max_bytes=4))).decision
        is GrantDecision.PROMPT
    )


def test_profile_is_signed_advice_and_typed_models_reject_unsafe_shapes():
    """Authenticate advisory profile identity without granting its predicted effects."""
    profile = CommandProfile(
        profile_id="shell",
        version="1",
        publisher="loop",
        executable_digest="sha256:shell",
        argument_class="ordinary",
        predicted_effects=frozenset({GrantEffect.FS_READ}),
        signature="7369676e6174757265",
    )
    profile.verify(_Verifier())
    assert profile.subject("run_command").profile_id == "shell"
    assert profile.digest
    assert GrantEffect.NETWORK_CONNECT.capability is Capability.NETWORK_CONNECT
    assert GrantEffect.FS_CREATE.capability is None
    with pytest.raises(ValueError, match="verification"):
        profile.model_copy(update={"signature": "00"}).verify(_Verifier())
    with pytest.raises(ValueError, match="hexadecimal"):
        profile.model_copy(update={"signature": "nope"}).verify(_Verifier())
    with pytest.raises(ValidationError, match="normalized absolute"):
        VirtualPathTree(root="/workspace/../secret")
    with pytest.raises(ValidationError, match="needs an executable"):
        SubjectIdentity(tool_id="tool", publisher="loop")
    with pytest.raises(ValidationError, match="both id and version"):
        SubjectIdentity(
            tool_id="tool", publisher="loop", executable_digest="sha256:tool", profile_id="p"
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"workspace_binding": None}, "Workspace grants"),
        (
            {"scope": GrantScope.SESSION, "workspace_binding": None},
            "Process and session",
        ),
        (
            {
                "scope": GrantScope.USER_POLICY,
                "workspace_binding": None,
                "resource": VirtualPathTree(
                    root="/workspace/**", effects=frozenset({GrantEffect.FS_REPLACE})
                ),
            },
            "bare global workspace",
        ),
        ({"expires_at_ns": 0}, "expiry cannot predate"),
    ],
)
def test_grant_lifetimes_require_exact_bindings(changes, message):
    """Reject grant lifetimes that could replay outside their intended authority."""
    with pytest.raises(ValidationError, match=message):
        _grant("invalid", GrantDecision.ALLOW, **changes)


def test_typed_matching_covers_resource_identity_scope_and_constraint_mismatches():
    """Never match across selector kinds, subjects, sessions, or profile constraints."""
    request = _request()
    assert (
        evaluate_grant(
            (
                _grant(
                    "other-resource",
                    GrantDecision.ALLOW,
                    resource=adapter_module.OpaqueResource(name="other"),
                ),
            ),
            request,
        ).decision
        is GrantDecision.PROMPT
    )
    changed_subject = request.subject.model_copy(update={"tool_id": "other"})
    assert (
        evaluate_grant(
            (_grant("identity", GrantDecision.ALLOW),),
            request.model_copy(update={"subject": changed_subject}),
        ).decision
        is GrantDecision.PROMPT
    )
    session = _grant(
        "session",
        GrantDecision.ALLOW,
        scope=GrantScope.SESSION,
        workspace_binding=None,
        scope_binding="agent",
    )
    assert (
        evaluate_grant((session,), request.model_copy(update={"scope_binding": "other"})).decision
        is GrantDecision.PROMPT
    )
    constrained = _grant(
        "profile",
        GrantDecision.ALLOW,
        constraints=GrantConstraints(runtime_digest="sha256:runtime", max_count=1),
    )
    assert evaluate_grant((constrained,), request).decision is GrantDecision.PROMPT
    opaque = adapter_module.OpaqueResource(name="same")
    opaque_request = request.model_copy(update={"resource": opaque})
    opaque_grant = _grant("opaque", GrantDecision.ALLOW, resource=opaque)
    assert evaluate_grant((opaque_grant,), opaque_request).decision is GrantDecision.ALLOW


def test_execution_adapter_fails_closed_for_unclassified_capability_and_maps_rename(tmp_path):
    """Reject missing capability classification and authorize both ends of a rename."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    registry, manager, sandbox = _command_registry(tmp_path, interaction=interaction)
    del registry
    permissions = sandbox.permissions
    assert not permissions.authorize_capabilities(
        (Capability.NETWORK_CONNECT,),
        runtime_digest="sha256:runtime",
        duration_ns=1,
    )
    removed = adapter_module._CAPABILITY_EFFECTS.pop(Capability.PROCESS_SPAWN)
    try:
        assert not permissions.authorize_capabilities(
            (Capability.PROCESS_SPAWN,),
            runtime_digest="sha256:runtime",
            duration_ns=1,
        )
    finally:
        adapter_module._CAPABILITY_EFFECTS[Capability.PROCESS_SPAWN] = removed
    request = sandbox.requests[0] if sandbox.requests else None
    if request is None:
        executor = SandboxCommandExecutor(
            ExecutionService(sandbox), permissions, "workspace", "agent", "sha256:runtime"
        )
        executor.execute("read", "/workspace", 1, request_id="rename")
        request = sandbox.requests[-1]
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                source_path="old",
                destination_path="new",
                object_kind=ObjectKind.FILE,
                mode=0o644,
            ),
        )
    )
    assert permissions.authorize_publication(request, delta)
    assert interaction.prompt.call_count == 1
    prompt = interaction.info.call_args.args[0]
    assert "Approval required to save command changes" in prompt
    assert "Command: read" in prompt
    assert "Working directory: /workspace" in prompt
    assert "Purpose: Save the following changes" in prompt
    assert "Rename a file or directory: /workspace/old" in prompt
    assert "Rename a file or directory: /workspace/new" in prompt
    assert "Boundary: sandbox" in prompt
    assert "Future coverage:" in prompt
    assert '"kind":"virtual_path_tree"' not in prompt
    assert manager.execution_grants == ()


def test_execution_adapter_compiles_exact_network_listener_and_secret_requests() -> None:
    """Every broker lease becomes one bounded canonical pre-exposure request."""
    manager = Mock()
    manager.authorize_execution.return_value = SimpleNamespace(decision=GrantDecision.ALLOW)
    permissions = ExecutionPermissionAdapter(
        manager,
        _subject(),
        "workspace",
        "2",
        "agent",
        clock_ns=lambda: 10,
    )
    connection = NetworkConnectionLease(
        hostname="api.example",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    listener = NetworkListenerLease(port=18080, target_port=8080)
    secret = SecretExposure(
        secret_id="token",
        audience="api.example",
        mechanism=SecretMechanism.REQUEST_HEADER,
        target="Authorization",
    )
    assert permissions.authorize_capabilities(
        (
            Capability.WORKSPACE_READ,
            Capability.PROCESS_SPAWN,
            Capability.NETWORK_CONNECT,
            Capability.NETWORK_LISTEN,
            Capability.SECRET_USE,
        ),
        runtime_digest="sha256:runtime",
        duration_ns=100,
        network_connections=(connection,),
        network_listeners=(listener,),
        secret_exposures=(secret,),
    )
    requests = manager.authorize_execution.call_args.args[0]
    assert {request.effect for request in requests} == {
        GrantEffect.FS_READ,
        GrantEffect.PROCESS_SPAWN,
        GrantEffect.NETWORK_CONNECT,
        GrantEffect.NETWORK_LISTEN,
        GrantEffect.SECRET_USE,
    }


def test_command_executor_stops_before_service_when_launch_authority_is_denied():
    """Return a typed denial without invoking the sandbox service."""
    permissions = Mock()
    permissions.authorize_capabilities.return_value = False
    service = Mock()
    executor = SandboxCommandExecutor(service, permissions, "workspace", "agent", "sha256:runtime")

    result = executor.execute("read", "/workspace", 1)

    assert isinstance(result, CapabilityDenied)
    service.execute.assert_not_called()


def test_command_executor_lazily_ensures_and_caches_the_single_sandbox_image() -> None:
    """The first command prepares one image and later commands reuse its exact digest."""
    permissions = Mock()
    permissions.authorize_capabilities.return_value = True
    service = Mock()
    service.execute.return_value = Completed(request_id="request", exit_code=0)
    digest = "sha256:" + "a" * 64
    resolver = Mock(return_value=digest)
    executor = SandboxCommandExecutor(
        service,
        permissions,
        "workspace",
        "agent",
        "sha256:source",
        runtime_resolver=resolver,
    )

    assert isinstance(executor.execute("go version", "/workspace", 1), Completed)
    assert isinstance(executor.execute("python --version", "/workspace", 1), Completed)
    resolver.assert_called_once_with()
    assert service.execute.call_args.args[0].lease.runtime_digest == digest


@pytest.mark.parametrize(
    "resolver",
    [Mock(return_value="mutable"), Mock(side_effect=RuntimeError("private failure"))],
)
def test_command_executor_closes_sandbox_preparation_failures(resolver: Mock) -> None:
    """Invalid or failed managed preparation returns only an opaque infrastructure result."""
    permissions = Mock()
    service = Mock()
    executor = SandboxCommandExecutor(
        service,
        permissions,
        "workspace",
        "agent",
        "sha256:source",
        runtime_resolver=resolver,
    )

    result = executor.execute("go version", "/workspace", 1)

    assert isinstance(result, InfrastructureFailure)
    assert result.diagnostic_id.startswith("diag_")
    permissions.authorize_capabilities.assert_not_called()
    service.execute.assert_not_called()


def test_durable_job_closes_sandbox_preparation_failure() -> None:
    """Durable startup fails before permission or adapter access when preparation fails."""
    permissions = Mock()
    service = SimpleNamespace(adapter=Mock())
    executor = SandboxCommandExecutor(
        service,  # type: ignore[arg-type]
        permissions,
        "workspace",
        "agent",
        "sha256:source",
        runtime_resolver=Mock(side_effect=RuntimeError("private failure")),
    )

    result = executor.start_job("true", "/workspace", 1)

    assert isinstance(result, InfrastructureFailure)
    assert result.diagnostic_id.startswith("diag_")
    permissions.authorize_capabilities.assert_not_called()


def test_durable_job_activates_the_lazily_prepared_sandbox_image() -> None:
    """A valid local image identity replaces the source version before authorization."""
    permissions = Mock()
    permissions.authorize_capabilities.return_value = True
    platform = Mock()
    platform.start_job.return_value = JobHandle(job_id="job", token="a" * 64)
    digest = "sha256:" + "b" * 64
    executor = SandboxCommandExecutor(
        SimpleNamespace(adapter=platform),  # type: ignore[arg-type]
        permissions,
        "workspace",
        "agent",
        "sha256:bootstrap",
        runtime_resolver=Mock(return_value=digest),
    )

    assert isinstance(executor.start_job("true", "/workspace", 1), JobHandle)
    assert executor.runtime_digest == digest
    assert executor.runtime_ready is True


def test_command_executor_authorizes_and_preserves_exact_effect_leases() -> None:
    """Public network, ingress, and secret requests retain exact typed authority."""
    permissions = Mock()
    permissions.authorize_capabilities.return_value = True
    service = Mock()
    service.execute.return_value = Completed(request_id="request", exit_code=0)
    executor = SandboxCommandExecutor(
        service,
        permissions,
        "workspace",
        "agent",
        "sha256:runtime",
        supports_network_effects=True,
        supports_secret_exposures=True,
    )
    connection = NetworkConnectionLease(
        hostname="api.example",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    listener = NetworkListenerLease(port=18080, target_port=8080)
    secret = SecretExposure(
        secret_id="token",
        audience="api.example",
        mechanism=SecretMechanism.REQUEST_HEADER,
        target="Authorization",
    )

    result = executor.execute(
        "true",
        "/workspace",
        1,
        request_id="request",
        network_connections=(connection,),
        network_listeners=(listener,),
        secret_exposures=(secret,),
    )

    assert isinstance(result, Completed)
    request = service.execute.call_args.args[0]
    assert request.network_connections == (connection,)
    assert request.network_listeners == (listener,)
    assert request.secret_exposures == (secret,)
    assert {
        Capability.NETWORK_CONNECT,
        Capability.NETWORK_LISTEN,
        Capability.SECRET_USE,
    } <= request.lease.capabilities
    assert permissions.authorize_capabilities.call_args.kwargs["network_connections"] == (
        connection,
    )


def test_command_executor_rejects_uncomposed_effect_services_before_authorization() -> None:
    """Production defaults fail closed before launching unsupported network or secret effects."""
    permissions = Mock()
    service = Mock()
    executor = SandboxCommandExecutor(service, permissions, "workspace", "agent", "sha256:runtime")
    connection = NetworkConnectionLease(
        hostname="api.example",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    secret = SecretExposure(
        secret_id="token",
        audience="api.example",
        mechanism=SecretMechanism.REQUEST_HEADER,
        target="Authorization",
    )

    network = executor.execute("true", "/workspace", 1, network_connections=(connection,))
    secrets = executor.execute("true", "/workspace", 1, secret_exposures=(secret,))

    assert isinstance(network, UnsupportedCapability)
    assert network.capability == "network_byte_limits"
    assert isinstance(secrets, UnsupportedCapability)
    assert secrets.capability == "secret_authority"
    permissions.authorize_capabilities.assert_not_called()
    service.execute.assert_not_called()


def test_command_executor_reuses_the_composed_durable_lifecycle() -> None:
    """Authorize durable starts and issue operation-specific leases through one adapter."""
    manager = object()
    platform = Mock()
    platform.start_job.return_value = JobHandle(job_id="job", token="a" * 64)
    platform.job_manager.return_value = manager
    permissions = Mock()
    permissions.authorize_capabilities.return_value = True
    service = SimpleNamespace(adapter=platform)
    executor = SandboxCommandExecutor(
        service,  # type: ignore[arg-type]
        permissions,
        "workspace",
        lambda: "agent",
        "sha256:runtime",
    )

    handle = executor.start_job("sleep 1", "/workspace", 2, request_id="job")
    status_lease = executor.job_lease(JobOperation.STATUS, 1)
    signal_lease = executor.job_lease(JobOperation.SIGNAL, 1)

    assert isinstance(handle, JobHandle)
    request = platform.start_job.call_args.args[0]
    assert request.mode.value == "durable_job"
    assert request.terminal.value == "pty"
    assert request.lease.agent_run_id == "agent"
    assert status_lease.capabilities == frozenset({Capability.PROCESS_SPAWN})
    assert signal_lease.capabilities == frozenset({Capability.PROCESS_SIGNAL})
    assert executor.job_manager() is manager

    denied_permissions = Mock()
    denied_permissions.authorize_capabilities.return_value = False
    denied = SandboxCommandExecutor(
        service,  # type: ignore[arg-type]
        denied_permissions,
        "workspace",
        "agent",
        "sha256:runtime",
    ).start_job("true", "/workspace", 1)
    assert isinstance(denied, CapabilityDenied)

    unsupported_service = SimpleNamespace(adapter=object())
    unsupported = SandboxCommandExecutor(
        unsupported_service,  # type: ignore[arg-type]
        permissions,
        "workspace",
        "agent",
        "sha256:runtime",
    )
    result = unsupported.start_job("true", "/workspace", 1)
    assert isinstance(result, UnsupportedCapability)
    with pytest.raises(TypeError, match="unavailable"):
        unsupported.job_manager()


def test_permission_adapter_authorizes_exact_durable_job_operations() -> None:
    """Bind each durable operation to its job identity, effect, runtime, and lifetime."""
    manager = Mock()
    manager.authorize_execution.return_value = SimpleNamespace(decision=GrantDecision.ALLOW)
    permissions = ExecutionPermissionAdapter(
        manager,
        _subject(),
        "workspace",
        "2",
        "session",
    )
    lease = SimpleNamespace(expires_at_ns=10**30, runtime_digest="sha256:runtime")

    assert permissions.authorize_job_operation(lease, JobOperation.STATUS, "job")
    request = manager.authorize_execution.call_args.args[0][0]
    assert request.effect is GrantEffect.PROCESS_SPAWN
    assert request.resource.name == "durable-job:job:status"

    manager.authorize_execution.return_value = SimpleNamespace(decision=GrantDecision.DENY)
    assert not permissions.authorize_job_operation(lease, JobOperation.CANCEL, "job")
    request = manager.authorize_execution.call_args.args[0][0]
    assert request.effect is GrantEffect.PROCESS_SIGNAL


def test_execution_approval_scopes_persist_in_the_single_manager_store(tmp_path):
    """Store session and user grants in their canonical manager-owned partitions."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(
        tmp_path,
        user_configuration_path=tmp_path / "user.yaml",
        interaction=interaction,
        workspace_id="workspace",
    )
    request = _request()

    session = manager.authorize_execution((request,))
    assert session.approval_scope is GrantScope.SESSION
    session_id = session.installed_grant_ids[0]
    assert manager.revoke_execution_grant("missing") is False
    assert manager.revoke_execution_grant(session_id)

    interaction.prompt.return_value = ApprovalChoice.USER
    user = manager.authorize_execution((request,))
    assert user.approval_scope is GrantScope.USER_POLICY
    user_id = user.installed_grant_ids[0]
    assert manager.revoke_execution_grant(user_id)
    assert manager.revoke_execution_grant(user_id) is False


def test_execution_prompt_explains_every_resource_in_plain_language(tmp_path):
    """Describe privileged effects without exposing internal resource serialization."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction, workspace_id="workspace")
    resources = (
        _request(),
        _request(
            effect=GrantEffect.NETWORK_CONNECT,
            resource=NetworkEndpoint(
                host="example.com",
                port=443,
                protocol=NetworkProtocol.HTTPS,
                addresses=("192.0.2.1",),
            ),
        ),
        _request(
            effect=GrantEffect.NETWORK_LISTEN,
            resource=NetworkListener(port=8080, target_port=3000),
        ),
        _request(
            effect=GrantEffect.SECRET_USE,
            resource=SecretUse(secret_id="api-token", audience="example.com", mechanism="env"),
        ),
        _request(
            effect=GrantEffect.IPC_CONNECT,
            resource=OpaqueResource(name="managed-ipc-channel"),
        ),
    )

    result = manager.authorize_execution(resources)

    assert result.decision is GrantDecision.ALLOW
    prompt = interaction.info.call_args.args[0]
    assert "Reason: This operation needs permissions" in prompt
    assert "Resource: /workspace/docs/readme" in prompt
    assert "Resource: https://example.com:443" in prompt
    assert "Resource: local port 8080 to sandbox port 3000" in prompt
    assert "Resource: secret 'api-token' for 'example.com' via env" in prompt
    assert "Resource: managed-ipc-channel" in prompt
    assert "Boundary: sandbox" in prompt
    assert "Future coverage:" in prompt
    assert '"kind"' not in prompt


def test_execution_approval_failure_and_mixed_prompt_fail_closed(tmp_path, monkeypatch):
    """Deny persistence failures and omit already-authorized effects from prompt and grants."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        workspace_id="workspace",
    )
    read = _request(
        effect=GrantEffect.FS_READ,
        resource=VirtualPathTree(root="/workspace/**", effects=frozenset({GrantEffect.FS_READ})),
    )
    result = manager.authorize_execution((read, _request()))
    assert result.decision is GrantDecision.ALLOW
    assert len(result.installed_grant_ids) == 1
    assert "fs.read" not in interaction.info.call_args.args[0]

    monkeypatch.setattr(
        manager,
        "_remember_execution_approval",
        Mock(side_effect=OSError("read only")),
    )
    failed = manager.authorize_execution(
        (
            _request(
                resource=VirtualPathTree(
                    root="/workspace/other", effects=frozenset({GrantEffect.FS_REPLACE})
                )
            ),
        )
    )
    assert failed.decision is GrantDecision.DENY
    assert "persist" in failed.reason


def test_unavailable_execution_persistence_scopes_and_duplicate_documents_fail_closed(
    tmp_path, monkeypatch
):
    """Reject spoofed durable choices and ambiguous typed grant identifiers."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(
        configuration=PermissionConfiguration(),
        interaction=interaction,
        workspace_id="workspace",
    )
    monkeypatch.setattr(manager, "request_permission", Mock(return_value=ApprovalChoice.WORKSPACE))
    assert manager.authorize_execution((_request(),)).decision is GrantDecision.DENY
    monkeypatch.setattr(manager, "request_permission", Mock(return_value=ApprovalChoice.USER))
    assert manager.authorize_execution((_request(),)).decision is GrantDecision.DENY

    grant = _grant("same", GrantDecision.ALLOW)
    with pytest.raises(ValidationError, match="grant identifiers"):
        PermissionConfiguration(version=2, execution_grants=[grant, grant])
    with pytest.raises(ValidationError, match="Version 1"):
        PermissionConfiguration(execution_grants=[grant])
    user_grant = grant.model_copy(update={"scope": GrantScope.USER_POLICY})
    with pytest.raises(ValidationError, match="grant identifiers"):
        UserPermissionConfiguration(version=2, execution_grants=[user_grant, user_grant])
    with pytest.raises(ValidationError, match="Version 1"):
        UserPermissionConfiguration(execution_grants=[user_grant])


def test_command_executor_closes_only_adapters_with_cleanup() -> None:
    """Executor shutdown delegates only when the injected sandbox owns cleanup."""
    adapter = Mock()
    service = ExecutionService(adapter)
    executor = SandboxCommandExecutor(service, Mock(), "workspace", "agent", "sha256:runtime")

    executor.close()
    adapter.close.assert_called_once_with()

    service.adapter = object()  # type: ignore[assignment]
    executor.close()
