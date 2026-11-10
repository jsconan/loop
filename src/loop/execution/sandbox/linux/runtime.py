"""Own installed Linux request admission, transport and confined execution."""

from __future__ import annotations

import ast
import base64
import errno
import fcntl
import hashlib
import json
import os
import select
import socket
import stat
import struct
import subprocess
import tempfile
import termios
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING

from ....constants import (
    MAX_NATIVE_SETUP_BYTES,
    MAX_OUTPUT_CHARS,
    NATIVE_RESULT_DELIVERY_SECONDS,
    NATIVE_UNIT_PREFIX,
)
from ....utils.hashing import sha256_digest
from ....utils.json import load_unique_json
from ....utils.process import ProcessCaptureStatus
from .. import contracts as shared_contracts
from ..constants import (
    LC_ALL_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    SYSTEM_LOCALE,
    TMPDIR_ENVIRONMENT_VARIABLE,
)
from ..contracts import CommandProcessResult, SandboxOutcome, SandboxRequest
from . import constants as linux_constants
from .admission import admission_group, receive_request
from .compiler import CompilationCancelled, compile_payload
from .constants import (
    BOOT_FILENAME,
    CACHE_LOCK_FILENAME,
    NATIVE_ALIAS_PATH,
    NATIVE_CANCEL_FRAME,
    NATIVE_PAYLOAD_MARKER_PREFIX,
    NATIVE_POLICY_VERSION,
    NATIVE_PROFILE_GROUP_HASH_CHARS,
    NATIVE_TRACKING_PATH,
    PACKAGE_ARTIFACT_PREFIX,
    PACKAGE_NAMESPACE,
    PAYLOAD_FILENAME,
    RECEIPT_FILENAME,
    SYSTEM_PYTHON_PATH,
    SYSTEM_RUNTIME_DIRECTORY,
    SYSTEMD_SOCKET_UNIT_SUFFIX,
)
from .egress import EgressBroker
from .host import (
    bind_host_file,
    discover_tool,
    installation_files,
    is_managed_bubblewrap,
    require_fixed_bubblewrap,
    runtime_read_roots,
)
from .namespace import namespace_arguments
from .policy import payload_profile
from .records import read_generation
from .seccomp import local_network_filter
from .supervision import guard_client_disconnect, supervise_payload
from .transport import exchange_request

if TYPE_CHECKING:
    from .diagnostics import LinuxDenialObserver
    from .transport import EgressLease

# The retained constants module is Python source, so individual assignment shapes cannot prove
# its runtime values. Qualify only the exact reviewed source loaded with this client.
_REVIEWED_CONSTANTS_DIGEST = hashlib.sha256(
    files(PACKAGE_NAMESPACE).joinpath("execution/sandbox/linux/constants.py").read_bytes()
).digest()
_SECCOMP_ARTIFACT = linux_constants.LINUX_ARTIFACT_DIRECTORY + "seccomp.py"
_UNIX_ISOLATION_FEATURE = "NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1"
_SOCKET_FAMILY_FEATURE = "NATIVE_SCOPED_SOCKET_FAMILIES_V1"
_WIRE_ARTIFACT = f"{PACKAGE_ARTIFACT_PREFIX}execution/sandbox/contracts.py"
_WIRE_DECLARATIONS = frozenset(
    {
        "NATIVE_REQUEST_SCHEMAS",
        "NATIVE_REQUEST_FEATURES",
        "NATIVE_REQUEST_FIELDS",
        "NATIVE_REQUEST_OPTIONAL_FIELDS",
        "NATIVE_REQUEST_SCHEMA2_ABSENT_FIELDS",
        "NATIVE_POLICY_CAPABILITIES",
        "NATIVE_RESULT_SCHEMAS",
    }
)


class NativeRepairRequired(ValueError):
    """Identify a verified native installation requiring a specific checked repair."""


class NativeFeatureUnavailable(ValueError):
    """Identify a request that the authenticated installed wire contract cannot support."""


def _has_unix_isolation(installed, deadline: float | None) -> bool:
    """Require both uniquely declared IPC capabilities in protected retained code."""
    artifact = installed.artifacts.get(_SECCOMP_ARTIFACT)
    if artifact is None or artifact.identity[4] > MAX_NATIVE_SETUP_BYTES:
        return False
    with artifact.open_verified(deadline=deadline) as descriptor:
        source = os.pread(descriptor, artifact.identity[4] + 1, 0)
    if len(source) != artifact.identity[4]:
        return False
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    features = {_UNIX_ISOLATION_FEATURE, _SOCKET_FAMILY_FEATURE}
    shadow_names = features | {"*"}
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in features:
            writes.append(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(
                (alias.asname or alias.name.split(".", 1)[0]) in shadow_names
                for alias in node.names
            ):
                return False
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in features
        ):
            return False
    if len(writes) != len(features) or set(writes) != features:
        return False
    declared = {
        statement.targets[0].id
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id in features
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is True
    }
    return declared == features


def _wire_capabilities(installed, deadline: float | None) -> dict[str, tuple]:
    """Read bounded, uniquely declared protocol capabilities from retained protected code."""
    artifact = installed.artifacts.get(_WIRE_ARTIFACT)
    if artifact is None or artifact.identity[4] > MAX_NATIVE_SETUP_BYTES:
        raise NativeFeatureUnavailable("Installed Linux sandbox wire capabilities are unavailable.")
    with artifact.open_verified(deadline=deadline) as descriptor:
        source = os.pread(descriptor, artifact.identity[4] + 1, 0)
    try:
        if len(source) != artifact.identity[4]:
            raise ValueError("Incomplete wire capability read.")
        tree = ast.parse(source)
        writes = [
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in _WIRE_DECLARATIONS
        ]
        if len(writes) != len(_WIRE_DECLARATIONS) or set(writes) != _WIRE_DECLARATIONS:
            raise ValueError("Ambiguous wire capability declarations.")
        result = {}
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in _WIRE_DECLARATIONS:
                result[target.id] = ast.literal_eval(statement.value)
        if result.keys() != _WIRE_DECLARATIONS:
            raise ValueError("Incomplete wire capability declarations.")
        for name, values in result.items():
            if (
                not isinstance(values, tuple)
                or (
                    not values
                    and name
                    not in {
                        "NATIVE_REQUEST_FEATURES",
                        "NATIVE_REQUEST_OPTIONAL_FIELDS",
                        "NATIVE_REQUEST_SCHEMA2_ABSENT_FIELDS",
                    }
                )
                or len(values) > 64
            ):
                raise ValueError("Invalid wire capability list.")
            kind = int if name in {"NATIVE_REQUEST_SCHEMAS", "NATIVE_RESULT_SCHEMAS"} else str
            if any(
                not isinstance(value, kind) or isinstance(value, bool) for value in values
            ) or len(set(values)) != len(values):
                raise ValueError("Invalid wire capability values.")
            if kind is str and any(
                not value.isascii() or not 0 < len(value) <= 100 for value in values
            ):
                raise ValueError("Invalid wire feature name.")
            if kind is int and any(value < 1 or value > 65535 for value in values):
                raise ValueError("Invalid wire schema version.")
        if not set(result["NATIVE_RESULT_SCHEMAS"]) <= set(
            shared_contracts.NATIVE_ACCEPTED_RESULT_SCHEMAS
        ):
            raise NativeFeatureUnavailable("Installed Linux sandbox result format is unsupported.")
        if 1 not in result["NATIVE_REQUEST_SCHEMAS"]:
            raise NativeFeatureUnavailable(
                "Installed Linux sandbox lacks the baseline request schema."
            )
        if not {"linux-native-v1", "unix-host-isolation-v1"} <= set(
            result["NATIVE_POLICY_CAPABILITIES"]
        ):
            raise NativeFeatureUnavailable("Installed Linux sandbox enforcement is unsupported.")
        supported = set(result["NATIVE_REQUEST_FIELDS"])
        if not (
            set(result["NATIVE_REQUEST_OPTIONAL_FIELDS"]) <= supported
            and set(result["NATIVE_REQUEST_SCHEMA2_ABSENT_FIELDS"]) <= supported
        ):
            raise ValueError("Invalid declared request shapes.")
        return result
    except NativeFeatureUnavailable:
        raise
    except (SyntaxError, ValueError, TypeError, RecursionError) as error:
        raise NativeFeatureUnavailable(
            "Installed Linux sandbox wire capabilities are unavailable."
        ) from error


def _require_wire_capabilities(installed, payload: bytes, deadline: float | None) -> None:
    """Refuse a request or possible response unsupported by the authenticated receiver."""
    capabilities = _wire_capabilities(installed, deadline)
    try:
        envelope = load_unique_json(payload.decode("utf-8"), duplicate_message="Duplicate request.")
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"schema", "request"}
            or not isinstance(envelope.get("request"), dict)
        ):
            raise TypeError("Invalid request envelope.")
        schema = envelope.get("schema")
        if not isinstance(schema, int) or isinstance(schema, bool):
            raise TypeError("Invalid request schema.")
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise NativeFeatureUnavailable(
            "Linux sandbox request wire format is unavailable."
        ) from error
    if schema not in (1, 2) or schema not in capabilities["NATIVE_REQUEST_SCHEMAS"]:
        raise NativeFeatureUnavailable(
            "Installed Linux sandbox cannot support this request schema."
        )
    fields = set(envelope["request"])
    supported_fields = set(capabilities["NATIVE_REQUEST_FIELDS"])
    optional = set(capabilities["NATIVE_REQUEST_OPTIONAL_FIELDS"])
    absent = set(capabilities["NATIVE_REQUEST_SCHEMA2_ABSENT_FIELDS"]) if schema == 2 else set()
    if (
        not fields <= supported_fields - absent
        or not supported_fields - optional - absent <= fields
    ):
        raise NativeFeatureUnavailable(
            "Installed Linux sandbox request fields differ; use /sandbox refresh."
        )
    if envelope["request"].get("policy_version") != NATIVE_POLICY_VERSION:
        raise NativeFeatureUnavailable("Installed Linux sandbox policy revision is unsupported.")
    features = set(capabilities["NATIVE_REQUEST_FEATURES"])
    if (
        envelope["request"].get("local_network") is False
        and "sandbox-local-network-v1" not in features
    ):
        raise NativeFeatureUnavailable(
            "Installed Linux sandbox cannot enforce local-network denial."
        )
    if (
        "workspace_executable_identities" in fields
        and "workspace-executable-identities-v1" not in features
    ):
        raise NativeFeatureUnavailable("Installed Linux sandbox cannot bind workspace executables.")
    identities = envelope["request"].get("executable_identities")
    if not isinstance(identities, list) or any(
        not isinstance(identity, list)
        or len(identity) != 5
        or not isinstance(identity[1], str)
        or not isinstance(identity[2], str)
        for identity in identities
    ):
        raise NativeFeatureUnavailable("Linux sandbox executable identities are unavailable.")
    if "system-executable-alias-v1" not in features and any(
        identity[1] != identity[2] for identity in identities
    ):
        raise NativeFeatureUnavailable("Installed Linux sandbox cannot bind executable aliases.")
    if envelope["request"].get("egress_origins") and "mediated-egress-v1" not in features:
        raise NativeFeatureUnavailable("Installed Linux sandbox cannot mediate external egress.")
    if envelope["request"].get("terminal") and "owned-terminal-v1" not in features:
        raise NativeFeatureUnavailable("Installed Linux sandbox lacks owned terminal support.")
    if envelope["request"].get("stream_output") and "live-output-beyond-capture-v1" not in features:
        raise NativeFeatureUnavailable(
            "Installed Linux sandbox cannot continue live output; use /sandbox refresh."
        )


def _require_runtime_revision(installed, *, deadline: float | None = None) -> None:
    """Require the installed architecture, system helper, and interface to match this client."""
    if BOOT_FILENAME not in installed.artifacts:
        raise NativeRepairRequired("Installed Linux sandbox needs /sandbox refresh.")
    if (
        not is_managed_bubblewrap(installed.helper.digest)
        and discover_tool("bwrap", deadline=deadline).digest != installed.helper.digest
    ):
        raise NativeRepairRequired("System Bubblewrap changed; use /sandbox refresh.")
    try:
        require_fixed_bubblewrap(installed.helper, deadline=deadline)
    except (OSError, TimeoutError, ValueError) as error:
        raise NativeRepairRequired(
            "Installed Bubblewrap lacks the fixed setup-symlink boundary; install version 0.12.0 "
            "or newer and use /sandbox refresh."
        ) from error
    if not _has_unix_isolation(installed, deadline):
        raise NativeRepairRequired(
            "Installed Linux sandbox lacks required Unix socket isolation or scoped socket-family "
            "isolation; use /sandbox refresh."
        )
    constants = installed.artifacts.get(linux_constants.LINUX_ARTIFACT_DIRECTORY + "constants.py")
    if constants is None or constants.identity[4] > MAX_NATIVE_SETUP_BYTES:
        raise NativeFeatureUnavailable("Installed native interface is unavailable.")
    with constants.open_verified(deadline=deadline) as descriptor:
        retained = os.pread(descriptor, constants.identity[4] + 1, 0)
    if (
        len(retained) != constants.identity[4]
        or hashlib.sha256(retained).digest() != _REVIEWED_CONSTANTS_DIGEST
    ):
        raise NativeFeatureUnavailable("Installed native interface differs; use /sandbox refresh.")


def _cancelled_before_spawn(request: SandboxRequest, descriptor: int) -> bool:
    """Validate launch authority and observe control without mistaking half-close for cancel."""
    if not request.paths_are_current():
        raise ValueError("Execution request changed before launch.")
    if time.monotonic() >= request.deadline:
        raise TimeoutError("Installed execution preparation deadline expired.")
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
    events = poller.poll(0)
    if not events:
        return False
    if len(events) != 1 or events[0][0] != descriptor:
        raise OSError("Client cancellation authority returned unexpected events.")
    mask = events[0][1]
    if mask & (select.POLLERR | select.POLLNVAL):
        raise OSError("Client cancellation authority could not be observed.")
    if mask & select.POLLIN:
        control = os.read(descriptor, len(NATIVE_CANCEL_FRAME))
        if control:
            if control != NATIVE_CANCEL_FRAME:
                raise ValueError("Unrecognized client cancellation control.")
            return True
    if mask & select.POLLHUP:
        return True
    if mask != select.POLLIN:
        raise OSError("Client cancellation authority returned unexpected events.")
    return False


@contextmanager
def receive_installed_request(
    record: Path, deadline: float
) -> Iterator[tuple[bytes, socket.socket]]:
    """Authenticate original installation authority before accepting one ordinary bounded frame.

    Reuse the production receipt, peer/frame and protected membership owner. Require receipt
    admission bindings to equal the original saved record, then revalidate that record after
    receipt and frame validation. Receiving grants no execution or backend readiness authority.

    Args:
        record (Path): Fixed protected complete management record beside the installed receiver.
        deadline (float): Absolute monotonic admission deadline supplied by the fixed receiver.

    Yields:
        tuple[bytes, socket.socket]: The admitted opaque request and its retained control socket.

    Raises:
        ValueError: Original record, receipt, peer, frame or protected membership differs.
        TimeoutError: The admission deadline expires during authority verification.
        OSError: Original authority cannot be read or admission is unavailable.
    """
    installed, authority = read_generation(record, deadline=deadline)
    receipt = installation_files(record.parent / RECEIPT_FILENAME, deadline=deadline)
    if any(receipt[role] != bound for role, bound in installed.files.items()):
        raise ValueError("Receipt admission differs from original management authority.")
    with receive_request(installed.config, receipt, deadline) as (payload, connection):
        with authority.open_verified(deadline=deadline):
            pass
        yield payload, connection


def exchange_installed_request(
    record: Path,
    payload: bytes,
    deadline: float,
    *,
    cancellation_event: Event | None = None,
    observer: LinuxDenialObserver | None = None,
    egress_lease: EgressLease | None = None,
    output_observer: Callable[[str, str], None] | None = None,
    terminal_input: Callable[[], bytes | None] | None = None,
) -> bytes:
    """Exchange one opaque request under the original complete installation authority.

    Authenticate all saved runtime artifacts and require the receipt roles to match the
    original management record before the ordinary socket transport sends any content.
    Revalidate the original record after the bounded exchange. No elevation, retry, outcome
    interpretation or command authorization occurs at this transport boundary.

    Args:
        record (Path): Fixed protected complete management record for the installed generation.
        payload (bytes): Already authorized nonempty opaque request, at most 64 KiB.
        deadline (float): Original absolute monotonic execution deadline; result cleanup/delivery
            has a fixed additional five-second interval.
        cancellation_event (Event | None): Optional run signal passed through the authenticated
            transport, which waits for receiver cleanup before returning its result.
        observer (LinuxDenialObserver | None): Optional unprivileged denial observation owner.
        egress_lease (EgressLease | None): Exact active policy lease for a mediated attempt.
        output_observer (Callable[[str, str], None] | None): Bounded live output receiver.
        terminal_input (Callable[[], bytes | None] | None): Next bounded transient terminal
            control frame after payload-entry confirmation.

    Returns:
        bytes: Authenticated bounded attempt-bound native result envelope.

    Raises:
        ValueError: Original record, receipt, socket, peer or framing authority is invalid.
        TimeoutError: The execution or bounded cleanup/delivery deadline expires.
        OSError: Original authority or bounded native socket exchange fails.
    """
    if time.monotonic() >= deadline:
        raise TimeoutError("Installed request transport deadline expired.")
    installed, authority = read_generation(record, deadline=deadline)
    _require_runtime_revision(installed, deadline=deadline)
    _require_wire_capabilities(installed, payload, deadline)
    receipt = installation_files(record.parent / RECEIPT_FILENAME, deadline=deadline)
    if any(receipt[role] != bound for role, bound in installed.files.items()):
        raise ValueError("Receipt transport differs from original management authority.")
    exchange_arguments = (
        {"cancellation_event": cancellation_event} if cancellation_event is not None else {}
    )
    if observer is not None:
        exchange_arguments["observer"] = observer
    if egress_lease is not None:
        exchange_arguments["egress_lease"] = egress_lease
    if output_observer is not None:
        exchange_arguments["output_observer"] = output_observer
    if terminal_input is not None:
        exchange_arguments["terminal_input"] = terminal_input
    response = exchange_request(
        installed.config,
        receipt,
        payload,
        deadline,
        **exchange_arguments,
    )
    delivery_deadline = deadline + NATIVE_RESULT_DELIVERY_SECONDS
    if time.monotonic() >= delivery_deadline:
        raise TimeoutError("Installed request transport deadline expired.")
    with authority.open_verified(deadline=delivery_deadline):
        pass
    if time.monotonic() >= delivery_deadline:
        raise TimeoutError("Installed request transport deadline expired.")
    return response


def execute_installed_request(
    record: Path, payload: bytes, connection: socket.socket
) -> CommandProcessResult:
    """Launch one admitted request through original installed authority and verified cleanup.

    Restore the original shared snapshot, compile its concrete payload authority, retain source
    and control bindings through launch, and supervise only the owned private PID namespace.
    Fixed trusted setup alone receives namespace-local MAC_ADMIN/SETPCAP. A fixed shell wrapper
    emits an attempt-bound startup header only after irreversible enforcing payload entry; opaque
    source cannot manufacture that first header. Failed setup returns invalid without a host
    offer. The request's preparation ceiling covers restoration and compilation; the runtime
    budget starts at launch. No elevation, retry, client-selected control input or inferred kernel
    denial occurs. A cancellation already queued at the final pre-spawn check returns without
    launching a payload; a later cancellation
    remains subject to the process supervisor and its cleanup guarantees.

    Args:
        record (Path): Original protected installed management record beside the receiver.
        payload (bytes): Authenticated admitted request snapshot, never an approval token.
        connection (socket.socket): Retained socket authenticated during request admission.

    Returns:
        CommandProcessResult: Typed bounded result after authenticated descendant absence, or an
            invalid result when system repair or native namespace path preparation is required.

    Raises:
        ValueError: Original identity, installation, request or cleanup authority differs outside
            namespace path preparation.
        TimeoutError: Preparation expires or descendants cannot be proven absent.
        OSError: Original native authority, compilation, launch or supervision fails.
    """
    request = SandboxRequest.from_wire(payload)
    if request.policy_version != NATIVE_POLICY_VERSION:
        raise ValueError("Installed request has an incompatible native policy revision.")
    execution_deadline = request.deadline
    if request.runtime_timeout is not None:
        preparation_deadline = execution_deadline - request.runtime_timeout
        if time.monotonic() >= preparation_deadline:
            return CommandProcessResult(
                SandboxOutcome.TIMED_OUT,
                detail="Native command preparation exceeded its deadline.",
            )
        request = replace(request, deadline=preparation_deadline)
    installed, authority = read_generation(record, deadline=request.deadline)
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (
        installed.config.uid,
        installed.config.uid,
        installed.config.gid,
        installed.config.gid,
    ):
        raise ValueError("Execution requires the fixed ordinary generation identity.")
    try:
        _require_runtime_revision(installed, deadline=request.deadline)
    except NativeRepairRequired as error:
        return CommandProcessResult(SandboxOutcome.INVALID, detail=str(error))
    scratch = request.shell_environment().get(TMPDIR_ENVIRONMENT_VARIABLE)
    cache, mask = (installed.artifacts.get(name) for name in (CACHE_LOCK_FILENAME, "mask.empty"))
    entry = installed.artifacts.get(PAYLOAD_FILENAME)
    if scratch is None or cache is None or mask is None or entry is None or mask.identity[4] != 0:
        raise ValueError(
            "Execution requires original trusted setup, empty mask and private scratch."
        )
    receipt = installation_files(record.parent / RECEIPT_FILENAME, deadline=request.deadline)
    if any(receipt[role] != bound for role, bound in installed.files.items()):
        raise ValueError("Execution receipt differs from original management authority.")
    bootstrap = installed.artifacts.get("policies/root/bootstrap.bin")
    if bootstrap is None:
        raise ValueError("Execution lacks original compiled bootstrap authority.")
    python = bind_host_file(SYSTEM_PYTHON_PATH, deadline=request.deadline)
    with (
        ExitStack() as pins,
        tempfile.TemporaryFile() as stream,
        mask.open_verified(deadline=request.deadline) as empty_mask,
        ExitStack() as prepared_paths,
    ):
        broker = None
        bridge_child = None
        if isinstance(request.egress_origins, tuple) and request.egress_origins:
            bridge_host, bridge_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            pins.enter_context(bridge_host)
            pins.enter_context(bridge_child)
            broker = EgressBroker(request, bridge_host)
            pins.callback(broker.close)
        try:
            arguments, inherited = prepared_paths.enter_context(
                namespace_arguments(request, Path(scratch), empty_mask=empty_mask)
            )
        except ValueError as error:
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail=str(error),
                failure_context="Native sandbox path preparation was refused.",
            )
        except OSError as error:
            if error.errno not in (errno.EMFILE, errno.ENFILE):
                raise
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Native sandbox mount preparation exceeded the open-file limit.",
                failure_context="Native sandbox path preparation was refused.",
            )
        try:
            for bound in (
                authority,
                installed.helper,
                entry,
                python,
                cache,
                bootstrap,
                receipt["apparmor"],
            ):
                pins.enter_context(bound.open_verified(deadline=request.deadline))
            runtime_roots = runtime_read_roots()
            group = admission_group(installed.config)
            group_fd = os.open(group.parent, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            pins.callback(os.close, group_fd)
            group_info = os.fstat(group_fd)
            group_identity = (group_info.st_dev, group_info.st_ino)
            current_group = group.parent.stat()
            if (
                not stat.S_ISDIR(group_info.st_mode)
                or (current_group.st_dev, current_group.st_ino) != group_identity
            ):
                raise ValueError("Generation tracking directory changed before binding.")
            profile = (
                payload_profile(request, Path(scratch), cache.path, runtime_roots)
                + "_"
                + sha256_digest(group.name)[:NATIVE_PROFILE_GROUP_HASH_CHARS]
            )
            try:
                compiled = compile_payload(
                    request,
                    Path(scratch),
                    cache.path,
                    profile,
                    runtime_roots,
                    installed.compiler_lock,
                    cancelled=lambda: _cancelled_before_spawn(request, 0),
                )
            except CompilationCancelled:
                return CommandProcessResult(SandboxOutcome.CANCELLED)
            value = {
                "request": base64.b64encode(payload).decode("ascii"),
                "compiled": base64.b64encode(compiled).decode("ascii"),
                "profile": profile,
                "namespace": installed.config.generation,
                "uid": installed.config.uid,
                "tracking": [group_info.st_dev, group_info.st_ino, group.name],
                **({"egress_fd": bridge_child.fileno()} if bridge_child is not None else {}),
                **(
                    {
                        "local_network_filter": base64.b64encode(
                            local_network_filter(
                                mediated_egress=bool(request.egress_origins),
                                deadline=request.deadline,
                            )
                        ).decode("ascii")
                    }
                    if not request.local_network
                    else {}
                ),
                **{
                    role: bound.to_record(deadline=request.deadline)
                    for role, bound in (
                        ("apparmor", receipt["apparmor"]),
                        ("bootstrap", bootstrap),
                        ("cache_lock", cache),
                        ("reservation", installed.config.reservation),
                    )
                },
            }
            data = json.dumps(value, separators=(",", ":")).encode("ascii")
            if len(data) > MAX_NATIVE_SETUP_BYTES or stream.write(data) != len(data):
                raise ValueError(
                    "Trusted setup input exceeds its bound or was not written completely."
                )
            stream.seek(0)
            root_fd = os.open(record.parent, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            pins.callback(os.close, root_fd)
            security = Path("/sys/kernel/security/apparmor")
            security_fd = os.open(security, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            pins.callback(os.close, security_fd)
            with (
                authority.open_verified(deadline=request.deadline),
                mask.open_verified(deadline=request.deadline),
            ):
                pass
            current_group = group.parent.stat()
            if (
                admission_group(installed.config) != group
                or (current_group.st_dev, current_group.st_ino) != group_identity
            ):
                raise ValueError("Generation tracking authority changed before launch.")
        except OSError as error:
            if error.errno not in (errno.EMFILE, errno.ENFILE):
                raise
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Native sandbox preparation exceeded the open-file limit.",
                failure_context="Native sandbox path preparation was refused.",
            )
        with guard_client_disconnect(connection) as spawn_returned:
            if _cancelled_before_spawn(request, 0):
                return CommandProcessResult(SandboxOutcome.CANCELLED)
            if request.runtime_timeout is not None and time.monotonic() >= request.deadline:
                return CommandProcessResult(
                    SandboxOutcome.TIMED_OUT,
                    detail="Native command preparation exceeded its deadline.",
                )
            terminal_master = None
            terminal_slave = None
            if request.terminal:
                master, slave = os.openpty()
                terminal_master = pins.enter_context(
                    os.fdopen(master, "r", encoding="utf-8", errors="replace")
                )
                terminal_slave = pins.enter_context(os.fdopen(slave, "rb+", buffering=0))
                attributes = termios.tcgetattr(slave)
                attributes[1] &= ~termios.ONLCR
                termios.tcsetattr(slave, termios.TCSANOW, attributes)
                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
            process = subprocess.Popen(
                [
                    str(installed.helper.path),
                    *arguments,
                    "--ro-bind-fd",
                    str(root_fd),
                    str(record.parent),
                    "--bind-fd",
                    str(security_fd),
                    str(security),
                    "--dir",
                    str(NATIVE_ALIAS_PATH),
                    "--ro-bind-fd",
                    str(group_fd),
                    str(NATIVE_TRACKING_PATH),
                    "--cap-add",
                    "CAP_MAC_ADMIN",
                    "--cap-add",
                    "CAP_SETPCAP",
                    str(python.path),
                    "-I",
                    "-B",
                    str(entry.path),
                ],
                stdin=stream,
                stdout=terminal_slave if request.terminal else subprocess.PIPE,
                stderr=terminal_slave if request.terminal else subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd="/",
                env={
                    PATH_ENVIRONMENT_VARIABLE: MINIMAL_SYSTEM_PATH,
                    LC_ALL_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                },
                pass_fds=(
                    *inherited,
                    root_fd,
                    security_fd,
                    group_fd,
                    *((bridge_child.fileno(),) if bridge_child is not None else ()),
                ),
                start_new_session=True,
            )
            if terminal_slave is not None:
                terminal_slave.close()
                read_end, write_end = os.pipe()
                os.close(write_end)
                process.stdout = terminal_master
                process.stderr = pins.enter_context(
                    os.fdopen(read_end, "r", encoding="utf-8", errors="replace")
                )
            spawn_returned.set()
            if request.runtime_timeout is not None:
                request = replace(
                    request,
                    deadline=min(execution_deadline, time.monotonic() + request.runtime_timeout),
                )
            if bridge_child is not None:
                bridge_child.close()
                broker.start_lease()
                connection.sendall(b"E")
            output_lock = Lock() if request.stream_output is True else None
            output_marker = NATIVE_PAYLOAD_MARKER_PREFIX + request.attempt_id + "\n"
            pending_marker = ""
            pending_stderr: list[str] = []
            pending_stderr_chars = 0
            marker_checked = False
            marker_valid = False

            def send_output(stream: str, chunk: str) -> None:
                """Send one bounded frame under the shared authenticated writer lock."""
                if not chunk:
                    return
                encoded = chunk.encode("utf-8")
                if len(encoded) > 4 * 8192:
                    raise ValueError("Native output frame exceeds its bound.")
                frame = (
                    b"O"
                    + (b"1" if stream == "stdout" else b"2")
                    + len(encoded).to_bytes(4, "big")
                    + encoded
                )
                while frame:
                    remaining = request.deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Native output delivery deadline expired.")
                    try:
                        count = connection.send(frame, getattr(socket, "MSG_DONTWAIT", 0))
                    except BlockingIOError:
                        select.select([], [connection], [], min(0.05, remaining))
                        continue
                    if count <= 0:
                        raise OSError("Native output receiver disconnected.")
                    frame = frame[count:]

            def publish_output(stream: str, chunk: str) -> None:
                """Withhold both streams until the trusted payload marker is verified."""
                nonlocal pending_marker, pending_stderr_chars, marker_checked, marker_valid
                with output_lock:
                    if not marker_checked:
                        if stream == "stderr":
                            retained = chunk[: max(0, MAX_OUTPUT_CHARS - pending_stderr_chars)]
                            if retained:
                                pending_stderr.append(retained)
                                pending_stderr_chars += len(retained)
                            return
                        pending_marker += chunk
                        if len(pending_marker) < len(output_marker):
                            return
                        marker_checked = True
                        marker_valid = pending_marker.startswith(output_marker)
                        if not marker_valid:
                            pending_stderr.clear()
                            return
                        chunk = pending_marker[len(output_marker) :]
                        pending_marker = ""
                        if request.terminal:
                            connection.sendall(b"T")
                        for pending in pending_stderr:
                            send_output("stderr", pending)
                        pending_stderr.clear()
                    if marker_valid:
                        send_output(stream, chunk)

            captured = supervise_payload(
                process,
                installed.config,
                request.deadline,
                connection=0,
                **({"terminal_master": master} if request.terminal else {}),
                **({"mediator": broker} if broker is not None else {}),
                **(
                    {"output_observer": publish_output, "send_lock": output_lock}
                    if output_lock is not None
                    else {}
                ),
            )
            if broker is not None and not broker.close():
                return CommandProcessResult(
                    SandboxOutcome.INVALID,
                    detail="Mediated egress closure could not be confirmed.",
                    possible_effects=True,
                    cleanup_verified=True,
                )
    header = NATIVE_PAYLOAD_MARKER_PREFIX + request.attempt_id + "\n"
    entered = captured.stdout.startswith(header)
    invalid_entry = not entered and captured.status not in (
        ProcessCaptureStatus.TIMED_OUT,
        ProcessCaptureStatus.CANCELLED,
    )
    outcome = SandboxOutcome.INVALID if invalid_entry else SandboxOutcome(captured.status.value)
    output = captured.stdout[len(header) :] if entered else captured.stdout
    error_output = captured.stderr
    extra = max(0, len(output) - MAX_OUTPUT_CHARS)
    error_extra = max(0, len(error_output) - MAX_OUTPUT_CHARS)
    return CommandProcessResult(
        outcome,
        exit_code=captured.exit_code if outcome is SandboxOutcome.COMPLETED else None,
        stdout=output[:MAX_OUTPUT_CHARS],
        stderr=error_output[:MAX_OUTPUT_CHARS],
        stdout_discarded=None
        if captured.stdout_discarded is None
        else captured.stdout_discarded + extra,
        stderr_discarded=None
        if captured.stderr_discarded is None
        else captured.stderr_discarded + error_extra,
        detail="Trusted payload entry was not established."
        if invalid_entry
        else "Native command output capture failed."
        if outcome is SandboxOutcome.UNAVAILABLE
        else "",
        possible_effects=True,
        cleanup_verified=True,
        failure_context="Native trusted setup did not establish command entry."
        if invalid_entry
        else "Native sandbox execution could not start.",
    )


def execution_authority(record: Path) -> Path:
    """Authenticate complete installed runtime metadata for ordinary callers.

    Args:
        record (Path): Original protected installation management record.

    Returns:
        Path: Fixed generation socket path; the caller must independently verify readiness.

    Raises:
        ValueError: Original installed runtime or trusted entry authority differs.
        OSError: Required protected installed or kernel authority is unavailable.
    """
    installed, authority = read_generation(record)
    _require_runtime_revision(installed)
    _wire_capabilities(installed, None)
    if not {PAYLOAD_FILENAME, "policies/root/bootstrap.bin"} <= installed.artifacts.keys():
        raise ValueError("Installed generation has no trusted payload entry.")
    with authority.open_verified():
        return SYSTEM_RUNTIME_DIRECTORY / (
            NATIVE_UNIT_PREFIX + installed.config.generation + SYSTEMD_SOCKET_UNIT_SUFFIX
        )
