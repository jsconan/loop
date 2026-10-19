"""Own installed Linux request admission, transport and confined execution."""

import base64
import json
import os
import subprocess
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from threading import Event

from ....constants import (
    MAX_NATIVE_SETUP_BYTES,
    MAX_OUTPUT_CHARS,
    NATIVE_RESULT_DELIVERY_SECONDS,
)
from ....utils.hashing import sha256_digest
from ....utils.process import ProcessCaptureStatus
from ..constants import (
    LC_ALL_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    SYSTEM_LOCALE,
    TMPDIR_ENVIRONMENT_VARIABLE,
)
from ..contracts import CommandProcessResult, SandboxOutcome, SandboxRequest
from .admission import receive_request
from .compiler import compile_payload
from .constants import (
    CACHE_LOCK_FILENAME,
    NATIVE_ALIAS_PATH,
    NATIVE_PAYLOAD_MARKER_PREFIX,
    NATIVE_POLICY_VERSION,
    PAYLOAD_FILENAME,
    RECEIPT_FILENAME,
    SYSTEM_PYTHON_PATH,
)
from .host import bind_host_file, installation_files, runtime_read_roots
from .namespace import namespace_arguments
from .paths import payload_aliases
from .policy import payload_profile
from .records import read_generation
from .supervision import supervise_payload
from .transport import exchange_request


def _require_runtime_revision(installed) -> None:
    """Refuse retained namespace semantics from another native protocol revision."""
    constants = installed.artifacts.get("loop/execution/sandbox/linux/constants.py")
    if constants is None or constants.digest != sha256_digest(
        Path(__file__).with_name("constants.py").read_bytes()
    ):
        raise ValueError("Installed native runtime revision changed; use checked refresh.")


def receive_installed_request(record: Path, deadline: float) -> bytes:
    """Authenticate original installation authority before accepting one ordinary bounded frame.

    Reuse the production receipt, peer/frame and protected membership owner. Require receipt
    admission bindings to equal the original saved record, then revalidate that record after
    receipt and frame validation. Receiving grants no execution or backend readiness authority.

    Args:
        record (Path): Fixed protected complete management record beside the installed receiver.
        deadline (float): Absolute monotonic admission deadline supplied by the fixed receiver.

    Returns:
        bytes: One authenticated bounded opaque frame; never interpreted or executed here.

    Raises:
        ValueError: Original record, receipt, peer, frame or protected membership differs.
        TimeoutError: The admission deadline expires during authority verification.
        OSError: Original authority cannot be read or admission is unavailable.
    """
    installed, authority = read_generation(record, deadline=deadline)
    receipt = installation_files(record.parent / RECEIPT_FILENAME, deadline=deadline)
    if any(receipt[role] != bound for role, bound in installed.files.items()):
        raise ValueError("Receipt admission differs from original management authority.")
    payload = receive_request(installed.config, receipt, deadline)
    with authority.open_verified(deadline=deadline):
        pass
    return payload


def exchange_installed_request(
    record: Path,
    payload: bytes,
    deadline: float,
    *,
    cancellation_event: Event | None = None,
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
    _require_runtime_revision(installed)
    receipt = installation_files(record.parent / RECEIPT_FILENAME, deadline=deadline)
    if any(receipt[role] != bound for role, bound in installed.files.items()):
        raise ValueError("Receipt transport differs from original management authority.")
    exchange_arguments = (
        {"cancellation_event": cancellation_event} if cancellation_event is not None else {}
    )
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


def execute_installed_request(record: Path, payload: bytes) -> CommandProcessResult:
    """Launch one admitted request through original installed authority and verified cleanup.

    Restore the original shared snapshot, compile its concrete payload authority, retain source
    and control bindings through launch, and supervise only the owned private PID namespace.
    Fixed trusted setup alone receives namespace-local MAC_ADMIN/SETPCAP. A fixed shell wrapper
    emits an attempt-bound startup header only after irreversible enforcing payload entry; opaque
    source cannot manufacture that first header. Failed setup returns invalid without a host
    offer. The original budget covers restoration, compilation, launch and output. No elevation,
    retry, client-selected control input or inferred kernel denial occurs.

    Args:
        record (Path): Original protected installed management record beside the receiver.
        payload (bytes): Authenticated admitted request snapshot, never an approval token.

    Returns:
        CommandProcessResult: Typed bounded result after authenticated descendant absence, or an
            invalid result when native namespace path preparation refuses the request.

    Raises:
        ValueError: Original identity, installation, request or cleanup authority differs outside
            namespace path preparation.
        TimeoutError: Preparation expires or descendants cannot be proven absent.
        OSError: Original native authority, compilation, launch or supervision fails.
    """
    request = SandboxRequest.from_wire(payload)
    if request.policy_version != NATIVE_POLICY_VERSION:
        raise ValueError("Installed request has an incompatible native policy revision.")
    installed, authority = read_generation(record, deadline=request.deadline)
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (
        installed.config.uid,
        installed.config.uid,
        installed.config.gid,
        installed.config.gid,
    ):
        raise ValueError("Execution requires the fixed ordinary generation identity.")
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
        profile = payload_profile(request, Path(scratch), cache.path, runtime_roots)
        compiled = compile_payload(
            request,
            Path(scratch),
            cache.path,
            profile,
            runtime_roots,
            installed.compiler_lock,
        )
        value = {
            "request": base64.b64encode(payload).decode("ascii"),
            "compiled": base64.b64encode(compiled).decode("ascii"),
            "profile": profile,
            "namespace": installed.config.generation,
            "uid": installed.config.uid,
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
            raise ValueError("Trusted setup input exceeds its bound or was not written completely.")
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
        if not request.paths_are_current():
            raise ValueError("Execution request changed before launch.")
        if time.monotonic() >= request.deadline:
            raise TimeoutError("Installed execution preparation deadline expired.")
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd="/",
            env={
                PATH_ENVIRONMENT_VARIABLE: MINIMAL_SYSTEM_PATH,
                LC_ALL_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
            },
            pass_fds=(*inherited, root_fd, security_fd),
            start_new_session=True,
        )
        captured = supervise_payload(process, installed.config, request.deadline, connection=0)
    header = NATIVE_PAYLOAD_MARKER_PREFIX + request.attempt_id + "\n"
    entered = captured.stdout.startswith(header)
    invalid_entry = not entered and captured.status not in (
        ProcessCaptureStatus.TIMED_OUT,
        ProcessCaptureStatus.CANCELLED,
    )
    outcome = SandboxOutcome.INVALID if invalid_entry else SandboxOutcome(captured.status.value)
    output = captured.stdout[len(header) :] if entered else captured.stdout
    error_output = captured.stderr
    for host, native, _ in sorted(
        payload_aliases(request), key=lambda item: len(str(item[1])), reverse=True
    ):
        output = output.replace(str(native), str(host))
        error_output = error_output.replace(str(native), str(host))
    if request.aliases:
        parent = str(request.aliases[0][1].parent)
        output = output.replace(str(NATIVE_ALIAS_PATH), parent)
        error_output = error_output.replace(str(NATIVE_ALIAS_PATH), parent)
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
        detail="Trusted payload entry was not established." if invalid_entry else "",
        possible_effects=True,
        failure_context="Native trusted setup did not establish command entry."
        if invalid_entry
        else "Native sandbox execution could not start.",
    )


def execution_authority(record: Path) -> None:
    """Authenticate complete installed runtime metadata for ordinary callers.

    Args:
        record (Path): Original protected installation management record.

    Raises:
        ValueError: Original installed runtime or trusted entry authority differs.
        OSError: Required protected installed or kernel authority is unavailable.
    """
    installed, authority = read_generation(record)
    _require_runtime_revision(installed)
    if not {PAYLOAD_FILENAME, "policies/root/bootstrap.bin"} <= installed.artifacts.keys():
        raise ValueError("Installed generation has no trusted payload entry.")
    with authority.open_verified():
        pass
