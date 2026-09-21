"""Test declarative effect-broker planning and trust-boundary validation."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from pydantic import ValidationError

from loop.execution.broker import BrokerPlanError, EffectBrokerPlanner
from loop.execution.contracts import (
    Capability,
    ExecutionLease,
    NetworkConnectionLease,
    NetworkListenerLease,
    NetworkProtocol,
    SecretExposure,
    SecretMechanism,
    ShellExecutionRequest,
)
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactRole,
    OciImageIdentity,
    PlatformSelector,
)


class _Secrets:
    """Resolve exact test secret bindings."""

    values: ClassVar[dict[tuple[str, str], bytes]] = {
        ("token", "api.example"): b"Bearer secret",
        ("env", "process"): b"environment-secret",
        ("file", "process"): b"file-secret",
    }

    def resolve(self, secret_id: str, audience: str) -> bytes:
        """Return exact fixture material or report an absent binding."""
        return self.values[(secret_id, audience)]


def _envoy() -> Artifact:
    """Return one fully pinned official broker image fixture."""
    return Artifact(
        artifact_id="envoy",
        version="1.39.1",
        role=ArtifactRole.ENVOY,
        platform=PlatformSelector(os="linux", architecture="arm64"),
        source="docker.io/envoyproxy/envoy:distroless-v1.39.1@sha256:" + "a" * 64,
        digest="sha256:" + "a" * 64,
        acquisition=AcquisitionKind.OCI,
        media_type="application/vnd.oci.image.index.v1+json",
        oci_identity=OciImageIdentity(
            index_digest="sha256:" + "a" * 64,
            manifest_digest="sha256:" + "b" * 64,
            config_digest="sha256:" + "c" * 64,
        ),
        capabilities=("network-broker",),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )


def _connection(
    hostname: str = "api.example",
    port: int = 443,
    protocol: NetworkProtocol = NetworkProtocol.HTTPS,
    addresses: tuple[str, ...] = ("93.184.216.34",),
) -> NetworkConnectionLease:
    """Return one exact public outbound lease."""
    return NetworkConnectionLease(
        hostname=hostname,
        port=port,
        protocol=protocol,
        addresses=addresses,
    )


def _request(
    *,
    connections: tuple[NetworkConnectionLease, ...] = (),
    listeners: tuple[NetworkListenerLease, ...] = (),
    secrets: tuple[SecretExposure, ...] = (),
) -> ShellExecutionRequest:
    """Return one request whose capabilities exactly match its broker leases."""
    capabilities = {Capability.WORKSPACE_READ, Capability.PROCESS_SPAWN}
    if connections:
        capabilities.add(Capability.NETWORK_CONNECT)
    if listeners:
        capabilities.add(Capability.NETWORK_LISTEN)
    if secrets:
        capabilities.add(Capability.SECRET_USE)
    return ShellExecutionRequest(
        request_id="request",
        lease=ExecutionLease(
            lease_id="lease",
            workspace_id="workspace",
            agent_run_id="agent",
            policy_version="2",
            runtime_digest="sha256:runtime",
            expires_at_ns=1,
            capabilities=frozenset(capabilities),
        ),
        script="true",
        network_connections=connections,
        network_listeners=listeners,
        secret_exposures=secrets,
    )


def test_offline_and_tls_plans_are_capability_proportional_and_static() -> None:
    """Offline plans start nothing, while TLS uses a no-route bridge and exact SNI chains."""
    planner = EffectBrokerPlanner()
    offline = planner.plan(_request(), _envoy(), _Secrets())
    assert offline.network_name is None
    assert offline.cni_configuration is None
    assert offline.envoy_configuration is None

    connections = (
        _connection("one.example", addresses=("93.184.216.34",)),
        _connection("two.example", addresses=("93.184.216.34",)),
    )
    plan = planner.plan(_request(connections=connections), _envoy(), _Secrets())
    cni = json.loads(plan.cni_configuration)
    assert cni["plugins"][0]["isGateway"] is False
    assert cni["plugins"][0]["ipMasq"] is False
    assert cni["plugins"][0]["ipam"]["routes"] == []
    envoy = json.loads(plan.envoy_configuration)
    assert "dynamic_resources" not in envoy
    chains = envoy["static_resources"]["listeners"][0]["filter_chains"]
    assert {chain["filter_chain_match"]["server_names"][0] for chain in chains} == {
        "one.example",
        "two.example",
    }
    assert all(
        "default_filter_chain" not in item for item in envoy["static_resources"]["listeners"]
    )
    assert plan.host_aliases == (
        ("one.example", plan.gateway),
        ("two.example", plan.gateway),
    )


def test_http_dns_tcp_ingress_and_secret_shapes_compile_without_fallback() -> None:
    """Every supported protocol and exposure produces only exact static resources."""
    connections = (
        _connection("tls.example", 443, NetworkProtocol.TLS),
        _connection("api.example", 8080, NetworkProtocol.HTTP),
        _connection("resolver.example", 53, NetworkProtocol.DNS, ("1.1.1.1",)),
        _connection("endpoint.example", 9000, NetworkProtocol.TCP),
    )
    secrets = (
        SecretExposure(
            secret_id="token",
            audience="api.example",
            mechanism=SecretMechanism.REQUEST_HEADER,
            target="Authorization",
        ),
        SecretExposure(
            secret_id="env",
            audience="process",
            mechanism=SecretMechanism.RAW_ENVIRONMENT,
            target="TOKEN",
        ),
        SecretExposure(
            secret_id="file",
            audience="process",
            mechanism=SecretMechanism.RAW_FILE,
            target="/run/secrets/token",
        ),
    )
    plan = EffectBrokerPlanner().plan(
        _request(
            connections=connections,
            listeners=(NetworkListenerLease(port=18080, target_port=8080),),
            secrets=secrets,
        ),
        _envoy(),
        _Secrets(),
    )
    envoy = json.loads(plan.envoy_configuration)
    listeners = envoy["static_resources"]["listeners"]
    assert (
        envoy["overload_manager"]["resource_monitors"][0]["typed_config"][
            "max_active_downstream_connections"
        ]
        == 80
    )
    assert all(
        cluster["circuit_breakers"]["thresholds"][0]["max_connections"] == 16
        for cluster in envoy["static_resources"]["clusters"]
    )
    http = next(item for item in listeners if item["name"] == "http_8080")
    route = http["filter_chains"][0]["filters"][0]["typed_config"]["route_config"]
    assert route["virtual_hosts"][0]["routes"][0]["route"]["cluster"] == "outbound_1"
    assert route["virtual_hosts"][0]["routes"][0]["request_headers_to_add"] == [
        {
            "header": {"key": "Authorization", "value": "Bearer secret"},
            "append_action": "OVERWRITE_IF_EXISTS_OR_ADD",
        }
    ]
    assert "request_headers_to_add" not in route["virtual_hosts"][0]["routes"][0]["route"]
    assert "Bearer secret" in json.dumps(http)
    dns = next(item for item in listeners if item["name"] == "dns_53")
    assert dns["address"]["socket_address"]["protocol"] == "UDP"
    assert any(item["name"] == "ingress_0" for item in listeners)
    assert plan.environment_file == b"TOKEN=environment-secret\n"
    assert plan.secret_files[0].destination == "/run/secrets/token"
    assert "file-secret" not in repr(plan)

    uncredentialed = EffectBrokerPlanner().plan(
        _request(connections=(_connection("public.example", 80, NetworkProtocol.HTTP),)),
        _envoy(),
        _Secrets(),
    )
    assert "request_headers_to_add" not in uncredentialed.envoy_configuration.decode()


@pytest.mark.parametrize(
    "value",
    ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fe80::1"],
)
def test_private_metadata_and_loopback_destinations_are_rejected(value: str) -> None:
    """No private, metadata, link-local, or loopback address reaches the planner."""
    with pytest.raises(ValidationError, match="not public"):
        _connection(addresses=(value,))


def test_broker_rejects_ambiguous_protocols_credentials_and_secret_material() -> None:
    """Unsupported sharing and malformed secret bytes fail before any exposure."""
    planner = EffectBrokerPlanner()
    mixed = (
        _connection("api.example", 443, NetworkProtocol.HTTPS),
        _connection("plain.example", 443, NetworkProtocol.HTTP),
    )
    with pytest.raises(BrokerPlanError, match="HTTP and opaque"):
        planner.plan(_request(connections=mixed), _envoy(), _Secrets())
    duplicate_dns = (
        _connection("one.example", 53, NetworkProtocol.DNS, ("1.1.1.1",)),
        _connection("two.example", 53, NetworkProtocol.DNS, ("8.8.8.8",)),
    )
    with pytest.raises(BrokerPlanError, match="dedicated UDP"):
        planner.plan(_request(connections=duplicate_dns), _envoy(), _Secrets())
    header = SecretExposure(
        secret_id="token",
        audience="api.example",
        mechanism=SecretMechanism.REQUEST_HEADER,
        target="Authorization",
    )
    with pytest.raises(BrokerPlanError, match="broker-terminated HTTP"):
        planner.plan(
            _request(connections=(_connection(),), secrets=(header,)),
            _envoy(),
            _Secrets(),
        )

    invalid_env = SecretExposure(
        secret_id="env",
        audience="process",
        mechanism=SecretMechanism.RAW_ENVIRONMENT,
        target="TOKEN",
    )
    invalid_header = SecretExposure(
        secret_id="token",
        audience="api.example",
        mechanism=SecretMechanism.REQUEST_HEADER,
        target="Authorization",
    )

    class InvalidSecrets:
        """Return one deliberately unsafe value selected by identifier."""

        def resolve(self, secret_id: str, audience: str) -> bytes:
            """Return fixture bytes that exercise bounded secret validation."""
            del audience
            return {
                "env": b"line\nbreak",
                "token": b"\xff",
                "empty": b"",
                "large": b"x" * (1024 * 1024 + 1),
                "delimiter": b"Bearer bad\rvalue",
            }[secret_id]

    with pytest.raises(BrokerPlanError, match="unsafe delimiter"):
        planner.plan(_request(secrets=(invalid_env,)), _envoy(), InvalidSecrets())
    with pytest.raises(BrokerPlanError, match="bounded ASCII"):
        planner.plan(
            _request(
                connections=(_connection(protocol=NetworkProtocol.HTTP),),
                secrets=(invalid_header,),
            ),
            _envoy(),
            InvalidSecrets(),
        )
    for secret_id in ("empty", "large"):
        exposure = invalid_env.model_copy(update={"secret_id": secret_id})
        with pytest.raises(BrokerPlanError, match="bounded shape"):
            planner.plan(_request(secrets=(exposure,)), _envoy(), InvalidSecrets())
    delimiter = invalid_header.model_copy(update={"secret_id": "delimiter"})
    with pytest.raises(BrokerPlanError, match="unsafe delimiter"):
        planner.plan(
            _request(
                connections=(_connection(protocol=NetworkProtocol.HTTP),),
                secrets=(delimiter,),
            ),
            _envoy(),
            InvalidSecrets(),
        )
    with pytest.raises(BrokerPlanError, match="uniquely leased"):
        mismatched = _request(
            connections=(_connection(protocol=NetworkProtocol.HTTP),),
            secrets=(invalid_header,),
        ).model_copy(
            update={
                "network_connections": (
                    _connection("other.example", protocol=NetworkProtocol.HTTP),
                )
            }
        )
        planner.plan(
            mismatched,
            _envoy(),
            InvalidSecrets(),
        )
    invalid_image = _envoy().model_copy(update={"role": ArtifactRole.OCI_PROFILE})
    with pytest.raises(BrokerPlanError, match="image identity"):
        planner.plan(_request(), invalid_image, _Secrets())

    shared_tcp = (
        _connection("one.example", 9000, NetworkProtocol.TCP),
        _connection("two.example", 9000, NetworkProtocol.TCP),
    )
    with pytest.raises(BrokerPlanError, match="dedicated listener"):
        planner.plan(_request(connections=shared_tcp), _envoy(), _Secrets())
    with pytest.raises(BrokerPlanError, match="unavailable"):
        planner.plan(
            _request(
                secrets=(
                    SecretExposure(
                        secret_id="missing",
                        audience="process",
                        mechanism=SecretMechanism.RAW_ENVIRONMENT,
                        target="TOKEN",
                    ),
                )
            ),
            _envoy(),
            _Secrets(),
        )


def test_contract_rejects_malformed_and_duplicated_broker_authority() -> None:
    """Typed requests reject unsafe paths, names, and duplicate exposures."""
    with pytest.raises(ValidationError):
        _connection("-bad.example")
    with pytest.raises(ValidationError, match="address is invalid"):
        _connection(addresses=("not-an-address",))
    with pytest.raises(ValidationError, match="DNS authority"):
        _connection("93.184.216.34")
    for mechanism, target in (
        (SecretMechanism.REQUEST_HEADER, "Cookie"),
        (SecretMechanism.RAW_ENVIRONMENT, "1TOKEN"),
    ):
        with pytest.raises(ValidationError):
            SecretExposure(
                secret_id="secret",
                audience="process",
                mechanism=mechanism,
                target=target,
            )
    with pytest.raises(ValidationError):
        SecretExposure(
            secret_id="secret",
            audience="process",
            mechanism=SecretMechanism.RAW_FILE,
            target="/run/secrets/value,option",
        )
    connection = _connection()
    with pytest.raises(ValidationError, match="unique"):
        _request(connections=(connection, connection))
    listener = NetworkListenerLease(port=18080, target_port=80)
    with pytest.raises(ValidationError, match="unique"):
        _request(listeners=(listener, listener))
    with pytest.raises(ValidationError, match="at most 64"):
        _request(
            connections=tuple(
                _connection(
                    f"endpoint-{index}.example",
                    10000 + index,
                    NetworkProtocol.TCP,
                )
                for index in range(65)
            )
        )

    for capability in (
        Capability.NETWORK_CONNECT,
        Capability.NETWORK_LISTEN,
        Capability.SECRET_USE,
    ):
        base = _request()
        with pytest.raises(ValidationError, match="authority"):
            ShellExecutionRequest(
                **base.model_dump(exclude={"lease"}),
                lease=base.lease.model_copy(
                    update={"capabilities": base.lease.capabilities | {capability}}
                ),
            )
    unmatched_header = SecretExposure(
        secret_id="token",
        audience="other.example",
        mechanism=SecretMechanism.REQUEST_HEADER,
        target="Authorization",
    )
    with pytest.raises(ValidationError, match="matching network lease"):
        _request(connections=(_connection(),), secrets=(unmatched_header,))
