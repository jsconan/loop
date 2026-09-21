"""Compile authorized network and secret leases into declarative broker state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .. import constants
from ..utils import sha256_digest
from .contracts import (
    NetworkConnectionLease,
    NetworkProtocol,
    SandboxExecutionRequest,
    SecretMechanism,
)
from .runtime.models import Artifact, ArtifactRole


class BrokerPlanError(ValueError):
    """Report a lease combination that cannot be represented safely."""


class SecretAuthority(Protocol):
    """Resolve audience-bound secret material without exposing a storage implementation."""

    def resolve(self, secret_id: str, audience: str) -> bytes:
        """Return secret bytes for one exact identifier and audience.

        Args:
            secret_id (str): Opaque secret identifier.
            audience (str): Exact authorized audience.

        Returns:
            bytes: Secret material owned by the caller until broker cleanup.

        Raises:
            KeyError: If no exact secret and audience binding exists.
        """


@dataclass(frozen=True, slots=True)
class BrokerSecretFile:
    """Carry one raw secret file only across the trusted staging boundary.

    Args:
        destination (str): Absolute process-private container destination.
        content (bytes): Bounded secret material, excluded from representations.
    """

    destination: str
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class BrokerPlan:
    """Describe complete pre-exposure broker state for one attempt.

    Args:
        lease_id (str): Immutable authorization lease identity.
        network_name (str | None): Dedicated CNI network name when networking is granted.
        gateway (str | None): Dedicated command-facing bridge gateway address.
        command_address (str | None): Fixed command address used by ingress clusters.
        cni_configuration (bytes | None): Canonical CNI configuration transferred as data.
        envoy_configuration (bytes | None): Canonical Envoy bootstrap transferred as data.
        host_aliases (tuple[tuple[str, str], ...]): Authorized hosts mapped only to the gateway.
        environment_file (bytes | None): Raw environment secrets for trusted guest staging.
        secret_files (tuple[BrokerSecretFile, ...]): Explicit raw file exposures.
    """

    lease_id: str
    network_name: str | None
    gateway: str | None
    command_address: str | None
    cni_configuration: bytes | None = field(repr=False)
    envoy_configuration: bytes | None = field(repr=False)
    host_aliases: tuple[tuple[str, str], ...]
    environment_file: bytes | None = field(repr=False)
    secret_files: tuple[BrokerSecretFile, ...] = field(repr=False)


class EffectBrokerPlanner:
    """Build data-only CNI, Envoy, and secret staging plans from closed leases."""

    def plan(
        self,
        request: SandboxExecutionRequest,
        envoy: Artifact,
        secrets: SecretAuthority,
    ) -> BrokerPlan:
        """Compile one request into a complete immutable broker plan.

        Args:
            request (SandboxExecutionRequest): Authorized sandbox request.
            envoy (Artifact): Digest-pinned official Envoy image.
            secrets (SecretAuthority): Exact audience-bound secret source.

        Returns:
            BrokerPlan: Data-only plan for platform-owned staging and supervision.

        Raises:
            BrokerPlanError: If a lease or secret cannot be represented safely.
        """
        if envoy.role is not ArtifactRole.ENVOY or envoy.oci_identity is None:
            raise BrokerPlanError("Network broker image identity is incomplete.")
        digest = sha256_digest(request.lease.lease_id)
        network_name = f"loop-{digest[:12]}"
        third_octet = 16 + int(digest[:2], 16) % 224
        subnet = f"10.240.{third_octet}.0/24"
        gateway = f"10.240.{third_octet}.1"
        command_address = f"10.240.{third_octet}.10"
        environment: list[bytes] = []
        files: list[BrokerSecretFile] = []
        headers: dict[str, dict[str, str]] = {}
        for exposure in request.secret_exposures:
            try:
                value = secrets.resolve(exposure.secret_id, exposure.audience)
            except KeyError as error:
                raise BrokerPlanError("Authorized secret material is unavailable.") from error
            if not value or len(value) > constants.MAX_SECRET_BYTES:
                raise BrokerPlanError("Secret material exceeds its bounded shape.")
            if exposure.mechanism is SecretMechanism.RAW_ENVIRONMENT:
                if b"\x00" in value or b"\n" in value or b"\r" in value:
                    raise BrokerPlanError("Environment secret contains an unsafe delimiter.")
                environment.append(exposure.target.encode() + b"=" + value)
            elif exposure.mechanism is SecretMechanism.RAW_FILE:
                files.append(BrokerSecretFile(exposure.target, value))
            else:
                lease = _connection_for_audience(request.network_connections, exposure.audience)
                if lease.protocol is not NetworkProtocol.HTTP:
                    raise BrokerPlanError(
                        "Header injection requires an explicit broker-terminated HTTP lease."
                    )
                try:
                    header = value.decode("ascii")
                except UnicodeDecodeError as error:
                    raise BrokerPlanError("Credential header is not bounded ASCII.") from error
                if any(character in header for character in "\r\n\x00"):
                    raise BrokerPlanError("Credential header contains an unsafe delimiter.")
                headers.setdefault(exposure.audience, {})[exposure.target] = header
        has_network = bool(request.network_connections or request.network_listeners)
        cni = _cni(network_name, subnet, gateway, command_address) if has_network else None
        envoy_config = (
            _envoy(
                request.network_connections,
                request.network_listeners,
                gateway,
                command_address,
                headers,
            )
            if has_network
            else None
        )
        aliases = tuple(
            sorted(
                (lease.hostname, gateway)
                for lease in request.network_connections
                if lease.protocol
                in {NetworkProtocol.HTTPS, NetworkProtocol.TLS, NetworkProtocol.HTTP}
            )
        )
        return BrokerPlan(
            request.lease.lease_id,
            network_name if has_network else None,
            gateway if has_network else None,
            command_address if has_network else None,
            cni,
            envoy_config,
            aliases,
            b"\n".join(environment) + (b"\n" if environment else b"") or None,
            tuple(files),
        )


def _connection_for_audience(
    connections: tuple[NetworkConnectionLease, ...], audience: str
) -> NetworkConnectionLease:
    """Return one unique audience lease or reject ambiguous credential injection."""
    matches = [lease for lease in connections if lease.hostname == audience]
    if len(matches) != 1:
        raise BrokerPlanError("Credential audience is not uniquely leased.")
    return matches[0]


def _cni(name: str, subnet: str, gateway: str, command_address: str) -> bytes:
    """Return one no-NAT, no-default-route CNI bridge configuration."""
    payload = {
        "cniVersion": "1.0.0",
        "name": name,
        "plugins": [
            {
                "type": "bridge",
                "bridge": f"br{name[-10:]}",
                "isGateway": False,
                "ipMasq": False,
                "hairpinMode": False,
                "ipam": {
                    "type": "host-local",
                    "ranges": [
                        [
                            {
                                "subnet": subnet,
                                "gateway": gateway,
                                "rangeStart": command_address,
                                "rangeEnd": command_address,
                            }
                        ]
                    ],
                    "routes": [],
                },
            },
            {"type": "loopback"},
        ],
    }
    return _json(payload)


def _envoy(connections, listeners, gateway: str, command_address: str, headers) -> bytes:
    """Return a static Envoy configuration with no dynamic discovery or fallback route."""
    clusters = []
    envoy_listeners = []
    by_port: dict[int, list[NetworkConnectionLease]] = {}
    for index, lease in enumerate(connections):
        name = f"outbound_{index}"
        clusters.append(_cluster(name, lease.addresses, lease.port, lease.max_connections))
        by_port.setdefault(lease.port, []).append(lease)
    for port, leases in sorted(by_port.items()):
        protocols = {lease.protocol for lease in leases}
        if NetworkProtocol.DNS in protocols:
            if protocols != {NetworkProtocol.DNS} or len(leases) != 1:
                raise BrokerPlanError("DNS leases require one dedicated UDP listener port.")
            index = connections.index(leases[0])
            envoy_listeners.append(_udp_listener(gateway, port, f"outbound_{index}"))
        elif NetworkProtocol.HTTP in protocols:
            if protocols != {NetworkProtocol.HTTP}:
                raise BrokerPlanError("HTTP and opaque streams cannot share one broker port.")
            envoy_listeners.append(_http_listener(gateway, port, leases, connections, headers))
        else:
            envoy_listeners.append(_stream_listener(gateway, port, leases, connections))
    for index, lease in enumerate(listeners):
        name = f"ingress_{index}"
        clusters.append(
            _cluster(name, (command_address,), lease.target_port, lease.max_connections)
        )
        envoy_listeners.append(
            {
                "name": name,
                "address": _socket(
                    "0.0.0.0" if lease.externally_visible else "127.0.0.1", lease.port
                ),
                "filter_chains": [_tcp_chain(name)],
                "per_connection_buffer_limit_bytes": 65536,
            }
        )
    return _json(
        {
            "static_resources": {"listeners": envoy_listeners, "clusters": clusters},
            "overload_manager": {
                "refresh_interval": "0.25s",
                "resource_monitors": [
                    {
                        "name": "envoy.resource_monitors.global_downstream_max_connections",
                        "typed_config": {
                            "@type": "type.googleapis.com/envoy.extensions.resource_monitors.downstream_connections.v3.DownstreamConnectionsConfig",
                            "max_active_downstream_connections": sum(
                                lease.max_connections for lease in (*connections, *listeners)
                            ),
                        },
                    }
                ],
            },
        }
    )


def _cluster(
    name: str,
    addresses: tuple[str, ...],
    port: int,
    max_connections: int,
) -> dict[str, object]:
    """Return one static cluster pinned to the approved address set."""
    return {
        "name": name,
        "type": "STATIC",
        "connect_timeout": "5s",
        "circuit_breakers": {
            "thresholds": [
                {
                    "priority": "DEFAULT",
                    "max_connections": max_connections,
                    "max_pending_requests": max_connections,
                    "max_requests": max_connections,
                }
            ]
        },
        "load_assignment": {
            "cluster_name": name,
            "endpoints": [
                {
                    "lb_endpoints": [
                        {"endpoint": {"address": _socket(address, port)}} for address in addresses
                    ]
                }
            ],
        },
    }


def _stream_listener(gateway, port, leases, all_connections) -> dict[str, object]:
    """Return exact SNI chains for TLS or one resolved-endpoint raw TCP chain."""
    chains = []
    for lease in leases:
        index = all_connections.index(lease)
        chain = _tcp_chain(f"outbound_{index}")
        if lease.protocol in {NetworkProtocol.HTTPS, NetworkProtocol.TLS}:
            chain["filter_chain_match"] = {
                "transport_protocol": "tls",
                "server_names": [lease.hostname],
            }
        elif len(leases) != 1 or lease.protocol is not NetworkProtocol.TCP:
            raise BrokerPlanError("Raw TCP leases require one dedicated listener port.")
        chains.append(chain)
    return {
        "name": f"outbound_{port}",
        "address": _socket(gateway, port),
        "listener_filters": [
            {
                "name": "envoy.filters.listener.tls_inspector",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.listener.tls_inspector.v3.TlsInspector"
                },
            }
        ],
        "filter_chains": chains,
        "per_connection_buffer_limit_bytes": min(65536, min(lease.max_bytes for lease in leases)),
    }


def _http_listener(gateway, port, leases, all_connections, headers) -> dict[str, object]:
    """Return authority-exact HTTP routes with optional broker-owned credentials."""
    virtual_hosts = []
    for local_index, lease in enumerate(leases):
        route: dict[str, object] = {
            "match": {"prefix": "/"},
            "route": {"cluster": f"outbound_{all_connections.index(lease)}"},
        }
        values = headers.get(lease.hostname, {})
        if values:
            route["request_headers_to_add"] = [
                {
                    "header": {"key": key, "value": value},
                    "append_action": "OVERWRITE_IF_EXISTS_OR_ADD",
                }
                for key, value in sorted(values.items())
            ]
        virtual_hosts.append(
            {
                "name": f"authority_{local_index}",
                "domains": [lease.hostname, f"{lease.hostname}:*"],
                "routes": [route],
            }
        )
    return {
        "name": f"http_{port}",
        "address": _socket(gateway, port),
        "filter_chains": [
            {
                "filters": [
                    {
                        "name": "envoy.filters.network.http_connection_manager",
                        "typed_config": {
                            "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
                            "stat_prefix": "broker_http",
                            "route_config": {
                                "name": "leased_routes",
                                "virtual_hosts": virtual_hosts,
                            },
                            "http_filters": [
                                {
                                    "name": "envoy.filters.http.router",
                                    "typed_config": {
                                        "@type": "type.googleapis.com/envoy.extensions.filters.http.router.v3.Router"
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        ],
        "per_connection_buffer_limit_bytes": min(65536, min(lease.max_bytes for lease in leases)),
    }


def _udp_listener(gateway: str, port: int, cluster: str) -> dict[str, object]:
    """Return one dedicated UDP proxy listener for an approved resolver."""
    address = _socket(gateway, port)
    address["socket_address"]["protocol"] = "UDP"
    return {
        "name": f"dns_{port}",
        "address": address,
        "udp_listener_config": {"downstream_socket_config": {"max_rx_datagram_size": 4096}},
        "listener_filters": [
            {
                "name": "envoy.filters.udp_listener.udp_proxy",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.udp.udp_proxy.v3.UdpProxyConfig",
                    "stat_prefix": cluster,
                    "matcher": {
                        "on_no_match": {
                            "action": {
                                "name": "route",
                                "typed_config": {
                                    "@type": "type.googleapis.com/envoy.extensions.filters.udp.udp_proxy.v3.Route",
                                    "cluster": cluster,
                                },
                            }
                        }
                    },
                    "idle_timeout": "5s",
                    "upstream_socket_config": {"max_rx_datagram_size": 4096},
                },
            }
        ],
    }


def _tcp_chain(cluster: str) -> dict[str, object]:
    """Return one bounded TCP proxy chain."""
    return {
        "filters": [
            {
                "name": "envoy.filters.network.tcp_proxy",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.network.tcp_proxy.v3.TcpProxy",
                    "stat_prefix": cluster,
                    "cluster": cluster,
                },
            }
        ]
    }


def _socket(address: str, port: int) -> dict[str, object]:
    """Return one TCP socket address."""
    return {"socket_address": {"address": address, "port_value": port, "protocol": "TCP"}}


def _json(value: object) -> bytes:
    """Encode canonical declarative broker data."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
