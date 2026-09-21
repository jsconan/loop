"""Prepare and reclaim lease-scoped effect brokers in the managed guest."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ....utils import sha256_digest
from ...broker import BrokerPlan
from ...contracts import NetworkListenerLease, NetworkProtocol
from ..oci.control import OciAttemptBindings
from .control_plane import MacosControlPlane

_LOGGER = logging.getLogger(__name__)


class MacosBrokerError(RuntimeError):
    """Report failure to prepare or fully reclaim a macOS effect broker."""


@dataclass(slots=True)
class MacosBrokerLease:
    """Own staged broker state, publication IDs, and a possible Envoy container.

    Args:
        bindings (OciAttemptBindings): Resources supplied to the command container.
        control_plane (MacosControlPlane): Attested guest control plane.
        broker_container (str | None): Lease-derived Envoy container identity.
        staged_paths (tuple[str, ...]): Exact guest paths to reclaim.
        publication_ids (tuple[int, ...]): RootlessKit port identities to revoke.
        cni_attachment (tuple[str, str, str, str] | None): Exact CNI attachment identity.
        bridge_name (str | None): Lease-derived bridge interface to reclaim.
    """

    bindings: OciAttemptBindings
    control_plane: MacosControlPlane
    broker_container: str | None
    staged_paths: tuple[str, ...]
    publication_ids: tuple[int, ...]
    cni_attachment: tuple[str, str, str, str] | None = None
    bridge_name: str | None = None
    _closed: bool = False

    def close(self) -> None:
        """Revoke publications and remove every lease-owned guest resource.

        Raises:
            MacosBrokerError: If any resource cannot be removed or attested absent.
        """
        if self._closed:
            return
        failed = False
        remaining_publications = []
        for publication_id in reversed(self.publication_ids):
            result = self.control_plane.remove_broker_port(publication_id)
            if _failed(result):
                failed = True
                remaining_publications.append(publication_id)
        self.publication_ids = tuple(reversed(remaining_publications))
        if self.broker_container is not None:
            result = self.control_plane.remove_container(self.broker_container)
            if _failed(result):
                failed = True
            else:
                self.broker_container = None
        if self.cni_attachment is not None:
            result = self.control_plane.remove_network_attachment(*self.cni_attachment)
            if _failed(result):
                failed = True
            else:
                self.cni_attachment = None
        if self.bridge_name is not None:
            result = self.control_plane.remove_broker_bridge(self.bridge_name)
            if _failed(result):
                failed = True
            else:
                self.bridge_name = None
        if self.staged_paths:
            result = self.control_plane.remove_broker_paths(self.staged_paths)
            if _failed(result):
                failed = True
            else:
                self.staged_paths = ()
        self._closed = not failed
        if failed:
            raise MacosBrokerError("Effect-broker cleanup could not be attested.")


class MacosEffectBroker:
    """Materialize one data-only broker plan through fixed trusted guest operations.

    Args:
        control_plane (MacosControlPlane): Attested managed guest boundary.
        owner_uid (int): Attested non-root guest runtime owner.
        owner_home (str): Attested guest owner home directory.
    """

    _control_plane: MacosControlPlane
    _owner_uid: int
    _owner_home: str

    def __init__(
        self,
        control_plane: MacosControlPlane,
        owner_uid: int,
        owner_home: str,
    ) -> None:
        if owner_uid <= 0 or re.fullmatch(r"/home/[A-Za-z0-9._-]+", owner_home) is None:
            raise ValueError("Effect broker requires an attested guest owner.")
        self._control_plane = control_plane
        self._owner_uid = owner_uid
        self._owner_home = owner_home

    def prepare(
        self,
        plan: BrokerPlan,
        protocols: frozenset[NetworkProtocol],
    ) -> MacosBrokerLease:
        """Stage broker state and return bindings without exposing untrusted code.

        Args:
            plan (BrokerPlan): Complete immutable broker configuration.
            protocols (frozenset[NetworkProtocol]): Granted outbound protocol set.

        Returns:
            MacosBrokerLease: Owned resources and command-container bindings.

        Raises:
            MacosBrokerError: If staging fails closed.
        """
        digest = sha256_digest(plan.lease_id)[:16]
        root = f"/run/user/{self._owner_uid}/loop/brokers/{digest}"
        staged: list[str] = []
        environment_file: str | None = None
        start_gate: str | None = None
        mounts: list[tuple[str, str]] = []
        cni_path: str | None = None
        cni_attachment: tuple[str, str, str, str] | None = None
        bridge_name: str | None = None
        try:
            if plan.cni_configuration is not None:
                assert plan.network_name is not None
                cni_path = (
                    f"{self._owner_home}/.config/cni/net.d/90-loop-{plan.network_name}.conflist"
                )
                self._write(cni_path, _bootstrap_configuration(plan.cni_configuration))
                staged.append(cni_path)
                bootstrap_path = f"{root}/bridge.json"
                self._write(
                    bootstrap_path,
                    _bootstrap_plugin_configuration(plan.cni_configuration),
                )
                staged.append(bootstrap_path)
                cni_attachment = (
                    f"loop-{digest}",
                    f"lb{digest[:10]}",
                    f"/run/user/{self._owner_uid}/loop/netns/{digest}",
                    bootstrap_path,
                )
                bridge_name = f"br{plan.network_name[-10:]}"
                initialized = self._control_plane.initialize_network(*cni_attachment)
                if _failed(initialized):
                    raise MacosBrokerError("Command bridge initialization failed.")
            if plan.environment_file is not None:
                environment_file = f"{root}/environment"
                self._write(environment_file, plan.environment_file)
                staged.append(environment_file)
            if plan.host_aliases:
                start_gate = f"{root}/start-gate"
                self._write(start_gate, b"wait\n")
                staged.append(start_gate)
            for index, secret in enumerate(plan.secret_files):
                source = f"{root}/secret-{index}"
                self._write(source, secret.content)
                staged.append(source)
                mounts.append((source, secret.destination))
            if plan.envoy_configuration is not None:
                configuration_path = f"{root}/envoy.json"
                self._write(configuration_path, plan.envoy_configuration)
                staged.append(configuration_path)
            gateway = plan.gateway
            bindings = OciAttemptBindings(
                plan.network_name,
                plan.command_address,
                plan.host_aliases,
                (gateway,) if gateway is not None and NetworkProtocol.DNS in protocols else (),
                environment_file,
                tuple(mounts),
                start_gate,
            )
            return MacosBrokerLease(
                bindings,
                self._control_plane,
                None,
                (*staged, root),
                (),
                cni_attachment,
                bridge_name,
            )
        except BaseException:
            lease = MacosBrokerLease(
                OciAttemptBindings(None, None),
                self._control_plane,
                None,
                (*staged, root),
                (),
                cni_attachment,
                bridge_name,
            )
            try:
                lease.close()
            except MacosBrokerError:
                pass
            raise

    def activate(
        self,
        plan: BrokerPlan,
        lease: MacosBrokerLease,
        envoy_reference: str,
        listeners: tuple[NetworkListenerLease, ...],
    ) -> None:
        """Harden an established command bridge before starting untrusted code.

        Args:
            plan (BrokerPlan): Complete immutable broker configuration.
            lease (MacosBrokerLease): Prepared lease that owns staged state.
            envoy_reference (str): Digest-pinned official Envoy image reference.
            listeners (tuple[NetworkListenerLease, ...]): Authorized ingress publications.

        Raises:
            MacosBrokerError: If hardening, startup, or publication fails closed.
        """
        digest = sha256_digest(plan.lease_id)[:16]
        root = f"/run/user/{self._owner_uid}/loop/brokers/{digest}"
        broker_container = f"loop-broker-{digest}"
        try:
            hardened = self._control_plane.harden_network_namespace()
            if (
                _failed(hardened)
                or hardened.stdout != b"ipv4=0\nipv6=0\nunprivileged_port_start=0\n"
            ):
                raise MacosBrokerError("Command broker namespace could not be hardened.")
            assert plan.network_name is not None
            cni_path = f"{self._owner_home}/.config/cni/net.d/90-loop-{plan.network_name}.conflist"
            assert plan.cni_configuration is not None
            self._write(cni_path, plan.cni_configuration)
            assert plan.envoy_configuration is not None
            started = self._control_plane.start_broker(
                broker_container,
                envoy_reference,
                f"{root}/envoy.json",
            )
            lease.broker_container = broker_container
            if _failed(started) or not self._broker_running(broker_container):
                raise MacosBrokerError("Envoy broker did not reach running state.")
            publications: list[int] = []
            for listener in listeners:
                address = "0.0.0.0" if listener.externally_visible else "127.0.0.1"
                published = self._control_plane.publish_broker_port(address, listener.port)
                if _failed(published):
                    raise MacosBrokerError("RootlessKit rejected broker publication.")
                try:
                    publication_id = int(published.stdout.strip())
                except ValueError as error:
                    raise MacosBrokerError(
                        "RootlessKit returned malformed publication evidence."
                    ) from error
                if publication_id <= 0:
                    raise MacosBrokerError("RootlessKit returned invalid publication evidence.")
                publications.append(publication_id)
                lease.publication_ids = tuple(publications)
        except BaseException:
            try:
                lease.close()
            except MacosBrokerError:
                pass
            raise

    def _write(self, path: str, content: bytes) -> None:
        """Stage one file and require complete bounded transfer evidence."""
        result = self._control_plane.write_broker_file(path, content)
        if _failed(result):
            raise MacosBrokerError("Effect-broker state could not be staged.")

    def _broker_running(self, container_id: str) -> bool:
        """Require immediate structured evidence that detached Envoy is running."""
        result = self._control_plane.inspect_container(container_id)
        if _failed(result):
            _LOGGER.error(
                "Envoy broker inspection failed: exit=%s stderr=%r truncated=(%s,%s).",
                result.exit_code,
                result.stderr,
                result.stdout_truncated,
                result.stderr_truncated,
            )
            return False
        try:
            value = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _LOGGER.error("Envoy broker inspection returned malformed structured evidence.")
            return False
        running = isinstance(value, dict) and value.get("State", {}).get("Running") is True
        if running:
            try:
                running = self._control_plane.container_uses_rootless_network_namespace(
                    container_id
                )
            except ValueError:
                return False
        if not running:
            state = value.get("State") if isinstance(value, dict) else None
            _LOGGER.error("Envoy broker did not remain running: state=%r.", state)
        return running


def guest_owner_home(state_path: str) -> str:
    """Derive the validated guest owner home from attested containerd state.

    Args:
        state_path (str): Attested rootless containerd state directory.

    Returns:
        str: Exact guest owner home directory.

    Raises:
        ValueError: If the state path is outside the supported guest layout.
    """
    match = re.fullmatch(r"(/home/[A-Za-z0-9._-]+)/\.local/share/containerd", state_path)
    if match is None:
        raise ValueError("Guest runtime state path has no supported owner home.")
    return match.group(1)


def _failed(result) -> bool:
    """Return whether fixed-shape infrastructure evidence is unsuccessful."""
    return bool(result.exit_code or result.stdout_truncated or result.stderr_truncated)


def _bootstrap_configuration(configuration: bytes) -> bytes:
    """Enable the gateway only for trusted bridge creation before exposure."""
    try:
        value = json.loads(configuration)
        plugins = value["plugins"]
        bridge = plugins[0]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise MacosBrokerError("CNI broker configuration is malformed.") from error
    if not isinstance(bridge, dict) or bridge.get("type") != "bridge":
        raise MacosBrokerError("CNI broker configuration is malformed.")
    bridge["isGateway"] = True
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _bootstrap_plugin_configuration(configuration: bytes) -> bytes:
    """Return one bridge-plugin ADD document with an initializer-only address."""
    value = json.loads(_bootstrap_configuration(configuration))
    bridge = value["plugins"][0]
    range_value = bridge["ipam"]["ranges"][0][0]
    prefix = range_value["gateway"].rsplit(".", 1)[0]
    range_value["rangeStart"] = f"{prefix}.11"
    range_value["rangeEnd"] = f"{prefix}.11"
    return json.dumps(
        {
            "cniVersion": value["cniVersion"],
            "name": value["name"],
            **bridge,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
