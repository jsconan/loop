"""Test lease-scoped macOS effect-broker staging and cleanup."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from loop.execution.broker import BrokerPlan, BrokerSecretFile
from loop.execution.contracts import NetworkListenerLease, NetworkProtocol
from loop.execution.sandbox.macos.broker import (
    MacosBrokerError,
    MacosEffectBroker,
    guest_owner_home,
)


def _result(stdout: bytes = b"") -> SimpleNamespace:
    """Return successful bounded infrastructure evidence."""
    return SimpleNamespace(
        exit_code=0,
        stdout=stdout,
        stderr=b"",
        stdout_truncated=False,
        stderr_truncated=False,
    )


class _Control:
    """Record exact trusted-plane broker operations."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.fail_operation: str | None = None
        self.fail_operations: set[str] = set()
        self.namespace_matches = True
        self.inspect_stdout = b'{"State":{"Running":true}}'
        self.publication_stdout = b"7\n"
        self.namespace_error = False

    def _record(self, operation: str, *values: object, stdout: bytes = b""):
        """Record one operation and optionally return failure evidence."""
        self.calls.append((operation, *values))
        if operation == self.fail_operation or operation in self.fail_operations:
            return SimpleNamespace(
                exit_code=1,
                stdout=b"",
                stderr=b"failed",
                stdout_truncated=False,
                stderr_truncated=False,
            )
        return _result(stdout)

    def write_broker_file(self, path: str, content: bytes):
        """Record staged bytes."""
        return self._record("write", path, content)

    def initialize_network(self, *values: object):
        """Record one direct CNI attachment."""
        return self._record("initialize", *values)

    def remove_network_attachment(self, *values: object):
        """Record direct CNI attachment cleanup."""
        return self._record("remove_attachment", *values)

    def harden_network_namespace(self):
        """Return exact forwarding and transparent-port hardening evidence."""
        return self._record(
            "harden",
            stdout=b"ipv4=0\nipv6=0\nunprivileged_port_start=0\n",
        )

    def start_broker(self, *values: object):
        """Record detached Envoy startup."""
        return self._record("start", *values)

    def inspect_container(self, container: str):
        """Report one running broker."""
        return self._record("inspect", container, stdout=self.inspect_stdout)

    def container_uses_rootless_network_namespace(self, container: str) -> bool:
        """Record and attest the broker's exact trusted namespace identity."""
        self.calls.append(("attest_namespace", container))
        if self.namespace_error:
            raise ValueError("bad namespace")
        return self.namespace_matches

    def publish_broker_port(self, address: str, port: int):
        """Return one RootlessKit publication identity."""
        return self._record("publish", address, port, stdout=self.publication_stdout)

    def remove_broker_port(self, identity: int):
        """Record publication revocation."""
        return self._record("unpublish", identity)

    def remove_container(self, container: str):
        """Record broker removal."""
        return self._record("remove_container", container)

    def remove_broker_bridge(self, bridge: str):
        """Record broker bridge reclamation."""
        return self._record("remove_bridge", bridge)

    def remove_broker_paths(self, paths: tuple[str, ...]):
        """Record exact guest-path cleanup."""
        return self._record("remove_paths", paths)


def _plan() -> BrokerPlan:
    """Return a complete network and raw-secret broker plan."""
    cni = {
        "cniVersion": "1.0.0",
        "name": "loop-network",
        "plugins": [
            {
                "type": "bridge",
                "isGateway": False,
                "ipam": {
                    "ranges": [
                        [
                            {
                                "gateway": "10.240.1.1",
                                "rangeStart": "10.240.1.10",
                                "rangeEnd": "10.240.1.10",
                            }
                        ]
                    ]
                },
            }
        ],
    }
    return BrokerPlan(
        lease_id="lease",
        network_name="loop-network",
        gateway="10.240.1.1",
        command_address="10.240.1.10",
        cni_configuration=json.dumps(cni).encode(),
        envoy_configuration=b'{"static_resources":{}}',
        host_aliases=(("api.example", "10.240.1.1"),),
        environment_file=b"TOKEN=value\n",
        secret_files=(BrokerSecretFile("/run/secrets/token", b"secret"),),
    )


def test_prepare_orders_bridge_hardening_before_exposure_and_reclaims_everything() -> None:
    """Bridge creation, forwarding disablement, final config, Envoy, and ingress are ordered."""
    control = _Control()
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    lease = broker.prepare(
        _plan(),
        frozenset({NetworkProtocol.HTTPS, NetworkProtocol.DNS}),
    )
    broker.activate(
        _plan(),
        lease,
        "registry.test/envoy@sha256:" + "a" * 64,
        (NetworkListenerLease(port=18080, target_port=8080),),
    )

    operations = [call[0] for call in control.calls]
    harden_index = operations.index("harden")
    final_cni_index = operations.index("write", harden_index)
    start_index = operations.index("start")
    assert harden_index < final_cni_index < start_index < operations.index("inspect")
    assert operations.index("inspect") < operations.index("attest_namespace")
    first_cni = json.loads(control.calls[0][2])
    final_cni = json.loads(control.calls[final_cni_index][2])
    assert first_cni["plugins"][0]["isGateway"] is True
    assert final_cni["plugins"][0]["isGateway"] is False
    assert lease.bindings.dns_servers == ("10.240.1.1",)
    assert lease.bindings.environment_file.endswith("/environment")
    assert lease.bindings.secret_mounts[0][1] == "/run/secrets/token"
    assert lease.bindings.start_gate.endswith("/start-gate")

    without_environment = broker.prepare(
        replace(_plan(), environment_file=None),
        frozenset(),
    )
    assert without_environment.bindings.environment_file is None
    without_environment.close()

    lease.close()
    lease.close()
    assert [call[0] for call in control.calls[-5:]] == [
        "unpublish",
        "remove_container",
        "remove_attachment",
        "remove_bridge",
        "remove_paths",
    ]


def test_prepare_failure_attempts_complete_partial_cleanup() -> None:
    """Any failed exposure reclaims already-created state before propagating failure."""
    control = _Control()
    control.fail_operation = "publish"
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    lease = broker.prepare(_plan(), frozenset({NetworkProtocol.HTTPS}))
    with pytest.raises(MacosBrokerError, match="publication"):
        broker.activate(
            _plan(),
            lease,
            "registry.test/envoy@sha256:" + "a" * 64,
            (NetworkListenerLease(port=18080, target_port=8080),),
        )
    assert "remove_container" in [call[0] for call in control.calls]
    assert control.calls[-1][0] == "remove_paths"

    control = _Control()
    control.namespace_matches = False
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    lease = broker.prepare(_plan(), frozenset({NetworkProtocol.HTTPS}))
    with pytest.raises(MacosBrokerError, match="running state"):
        broker.activate(
            _plan(),
            lease,
            "registry.test/envoy@sha256:" + "a" * 64,
            (),
        )
    assert "attest_namespace" in [call[0] for call in control.calls]
    assert control.calls[-1][0] == "remove_paths"


def test_broker_rejects_bad_owner_config_evidence_and_cleanup() -> None:
    """Guest identity, declarative config, running evidence, and cleanup all fail closed."""
    with pytest.raises(ValueError, match="attested guest owner"):
        MacosEffectBroker(_Control(), 0, "/home/loop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="owner home"):
        guest_owner_home("/var/lib/containerd")
    assert guest_owner_home("/home/loop/.local/share/containerd") == "/home/loop"

    control = _Control()
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    malformed = replace(_plan(), cni_configuration=b"{}")
    with pytest.raises(MacosBrokerError, match="malformed"):
        broker.prepare(
            malformed,
            frozenset(),
        )

    control = _Control()
    control.fail_operation = "remove_paths"
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    secret_only = BrokerPlan(
        "lease",
        None,
        None,
        None,
        None,
        None,
        (),
        b"TOKEN=value\n",
        (),
    )
    lease = broker.prepare(secret_only, frozenset())
    with pytest.raises(MacosBrokerError, match="cleanup"):
        lease.close()
    control.fail_operation = None
    lease.close()
    assert lease._closed


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("initialize", "initialization"),
        ("write", "staged"),
        ("harden", "hardened"),
        ("start", "running state"),
    ),
)
def test_broker_fails_closed_at_each_pre_exposure_boundary(failure: str, message: str) -> None:
    """Bridge, staging, hardening, and sidecar failures reclaim partial resources."""
    control = _Control()
    control.fail_operation = failure
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    if failure in {"initialize", "write"}:
        with pytest.raises(MacosBrokerError, match=message):
            broker.prepare(_plan(), frozenset())
    else:
        lease = broker.prepare(_plan(), frozenset())
        with pytest.raises(MacosBrokerError, match=message):
            broker.activate(
                _plan(),
                lease,
                "registry.test/envoy@sha256:" + "a" * 64,
                (),
            )
    assert control.calls[-1][0] == "remove_paths"


@pytest.mark.parametrize("publication", (b"bad", b"0"))
def test_broker_rejects_malformed_publication_identity(publication: bytes) -> None:
    """RootlessKit publication identities must be positive decimal values."""
    control = _Control()
    control.publication_stdout = publication
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    lease = broker.prepare(_plan(), frozenset())
    with pytest.raises(MacosBrokerError, match="publication evidence"):
        broker.activate(
            _plan(),
            lease,
            "registry.test/envoy@sha256:" + "a" * 64,
            (NetworkListenerLease(port=18080, target_port=8080),),
        )


@pytest.mark.parametrize(
    ("inspect_stdout", "inspect_failure", "namespace_error"),
    (
        (b"bad", False, False),
        (b'{"State":{"Running":false}}', False, False),
        (b'{"State":{"Running":true}}', True, False),
        (b'{"State":{"Running":true}}', False, True),
    ),
)
def test_broker_rejects_untrusted_running_evidence(
    inspect_stdout: bytes, inspect_failure: bool, namespace_error: bool
) -> None:
    """Malformed, stopped, failed, or unbound sidecar evidence never opens exposure."""
    control = _Control()
    control.inspect_stdout = inspect_stdout
    control.namespace_error = namespace_error
    if inspect_failure:
        control.fail_operation = "inspect"
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    lease = broker.prepare(_plan(), frozenset())
    with pytest.raises(MacosBrokerError, match="running state"):
        broker.activate(
            _plan(),
            lease,
            "registry.test/envoy@sha256:" + "a" * 64,
            (),
        )


def test_broker_swallows_secondary_cleanup_failure_while_preserving_primary_error() -> None:
    """Cleanup failure cannot replace the operation that originally failed closed."""
    control = _Control()
    control.fail_operations = {"write", "remove_paths"}
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    with pytest.raises(MacosBrokerError, match="staged"):
        broker.prepare(_plan(), frozenset())

    control = _Control()
    control.fail_operations = {"harden", "remove_paths"}
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    lease = broker.prepare(_plan(), frozenset())
    with pytest.raises(MacosBrokerError, match="hardened"):
        broker.activate(
            _plan(),
            lease,
            "registry.test/envoy@sha256:" + "a" * 64,
            (),
        )


@pytest.mark.parametrize(
    "failure",
    ("unpublish", "remove_container", "remove_attachment", "remove_bridge", "remove_paths"),
)
def test_broker_cleanup_retries_only_unreclaimed_resources(failure: str) -> None:
    """A transient cleanup failure remains retryable without revisiting reclaimed resources."""
    control = _Control()
    broker = MacosEffectBroker(control, 1000, "/home/loop")  # type: ignore[arg-type]
    lease = broker.prepare(_plan(), frozenset())
    broker.activate(
        _plan(),
        lease,
        "registry.test/envoy@sha256:" + "a" * 64,
        (NetworkListenerLease(port=18080, target_port=8080),),
    )
    control.fail_operation = failure
    with pytest.raises(MacosBrokerError, match="cleanup"):
        lease.close()
    first_attempt = [call[0] for call in control.calls]
    control.fail_operation = None
    lease.close()
    retry = [call[0] for call in control.calls[len(first_attempt) :]]
    assert retry == [failure]


@pytest.mark.parametrize("configuration", (b"not-json", b'{"plugins":[]}', b'{"plugins":[1]}'))
def test_broker_rejects_every_malformed_cni_bootstrap(configuration: bytes) -> None:
    """Malformed declarative CNI shapes fail before invoking a plugin."""
    broker = MacosEffectBroker(_Control(), 1000, "/home/loop")  # type: ignore[arg-type]
    with pytest.raises(MacosBrokerError, match="malformed"):
        broker.prepare(replace(_plan(), cni_configuration=configuration), frozenset())
