"""Own exact package review and terminal setup enabling checked ordinary native admission."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path

from .... import constants
from ....constants import (
    MAX_NATIVE_SETUP_ERROR_CHARS,
    MAX_NATIVE_SETUP_ERROR_COUNT,
    NATIVE_INSTALLATION_DIRECTORY,
    NATIVE_MANAGEMENT_TIMEOUT_SECONDS,
    NATIVE_UNIT_PREFIX,
)
from .admission import GenerationAdmission
from .catalog import runtime_catalog
from .constants import (
    COMPILER_LOCK_FILENAME,
    INITIAL_USER_NAMESPACE_UID_MAP,
    MANAGEMENT_LOCK_FILENAME,
    MANAGEMENT_RECORD_NAME,
    NATIVE_RESERVATIONS_DIRECTORY_NAME,
    PROC_UID_MAP_PATH,
    RECEIVER_FILENAME,
    SYSTEM_PYTHON_PATH,
    SYSTEMD_SLICE_UNIT_SUFFIX,
    UNINSTALLER_FILENAME,
)
from .host import NATIVE_LOCK_SEEDS, HostFile, bind_host_file, discover_tool, protected_path
from .installation import (
    generation_limits,
    reserve_generation,
    stage_generation_files,
    validate_generation_catalog,
)
from .management import (
    clean_unrecorded_installation,
    install_generation,
    uninstall_authority,
    uninstall_generation,
    verify_uninstalled,
)
from .policy import runtime_policy_sources

_DIRECTORY = NATIVE_INSTALLATION_DIRECTORY
_SEEDS = NATIVE_LOCK_SEEDS
_MAX_SETUP_ERROR_REPORT_BYTES = 4_096
_SETUP_ACTION_TIMEOUT_SECONDS = 30.0
_MAX_SETUP_INPUT_BYTES = 34 * 1024 * 1024
_SETUP_AUTHENTICATION_TIMEOUT_SECONDS = 120


def _directory(path):
    """Create fixed protected ancestry without aliases or changes to existing permissions."""
    if protected_path(path.parent, directory=True) != path.parent:
        raise ValueError("Setup parent aliases other authority.")
    try:
        path.mkdir(mode=constants.PUBLIC_DIRECTORY_MODE)
    except FileExistsError:
        pass
    else:
        path.chmod(constants.PUBLIC_DIRECTORY_MODE)
    if protected_path(path, directory=True) != path:
        raise ValueError("Setup directory aliases other authority.")
    if (
        path.stat().st_mode & constants.READ_ONLY_EXECUTABLE_FILE_MODE
        != constants.READ_ONLY_EXECUTABLE_FILE_MODE
    ):
        raise ValueError("Setup directory lacks required runtime read/traversal authority.")


def install_reviewed(
    action: str,
    generation: str,
    uid: int,
    gid: int,
    record_digest: str,
    contents: dict[str, bytes],
) -> None:
    """Run one reviewed operation against fixed protected Linux installation authority.

    Install removes an authenticated partial generation before staging fresh files. Uninstall
    retries from the original record or verifies complete absence after its final removal.

    Args:
        action (str): Exact install or uninstall operation.
        generation (str): Reviewed generation identity, including a retained partial reservation.
        uid (int): Positive ordinary main-app user identity.
        gid (int): Positive ordinary main-app group identity.
        record_digest (str): Reviewed original record SHA256 when one exists, else empty.
        contents (dict[str, bytes]): Complete digest-authenticated packaged runtime catalog.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Reviewed input, original authority or fixed scaffolding differs.
        OSError: Setup, live state, contention or native lifecycle work fails.
        BaseExceptionGroup: Activated authority validation and checked admission closure both fail.
    """
    if os.getuid() != 0 or (
        tuple(PROC_UID_MAP_PATH.read_text(encoding="ascii").split())
        != INITIAL_USER_NAMESPACE_UID_MAP
    ):
        raise PermissionError("Setup requires initial-host administrator authentication.")
    if (
        action not in {"install", "uninstall"}
        or not isinstance(uid, int)
        or isinstance(uid, bool)
        or not isinstance(gid, int)
        or isinstance(gid, bool)
        or min(uid, gid) <= 0
    ):
        raise ValueError("Invalid reviewed setup identity or action.")
    if action == "uninstall":
        record = _DIRECTORY / MANAGEMENT_RECORD_NAME
        if not record.exists() and not record.is_symlink():
            if record_digest:
                raise ValueError("Reviewed uninstall authority disappeared.")
            reservations = _DIRECTORY / NATIVE_RESERVATIONS_DIRECTORY_NAME
            if reservations.exists() or reservations.is_symlink():
                if protected_path(reservations, directory=True) != reservations:
                    raise ValueError("Installation capacity directory aliases other authority.")
                retained = list(reservations.iterdir())
                if len(retained) > 1:
                    raise ValueError("Ambiguous partial installation capacity.")
                if retained:
                    if retained[0].name != generation:
                        raise ValueError("Reviewed partial generation changed.")
                    reservation = bind_host_file(retained[0])
                    generation_limits(reservation, generation)
                    validate_generation_catalog(
                        _DIRECTORY, contents, {RECEIVER_FILENAME, UNINSTALLER_FILENAME, "helper"}
                    )
                    helper = discover_tool("bwrap")
                    if hashlib.sha256(contents["helper"]).hexdigest() != helper.digest:
                        raise ValueError("Reviewed system helper changed.")
                    sources, source = runtime_policy_sources(_DIRECTORY, generation)
                    lock = bind_host_file(_DIRECTORY / MANAGEMENT_LOCK_FILENAME)
                    compiler = bind_host_file(_DIRECTORY / COMPILER_LOCK_FILENAME)
                    clean_unrecorded_installation(
                        GenerationAdmission(
                            generation,
                            uid,
                            gid,
                            _DIRECTORY / RECEIVER_FILENAME,
                            HostFile(
                                _DIRECTORY
                                / f"{NATIVE_UNIT_PREFIX}{generation}{SYSTEMD_SLICE_UNIT_SUFFIX}",
                                (),
                                "",
                            ),
                            reservation,
                        ),
                        lock,
                        compiler,
                        contents,
                        {RECEIVER_FILENAME, UNINSTALLER_FILENAME, "helper"},
                        sources,
                        source,
                        time.monotonic() + _SETUP_ACTION_TIMEOUT_SECONDS,
                    )
            verify_uninstalled(
                _DIRECTORY,
                time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS,
            )
            return
        authority, entry = uninstall_authority(record)
        if authority.digest != record_digest:
            raise ValueError("Reviewed uninstall authority changed.")
        with authority.open_verified(), entry.open_verified():
            uninstall_generation(
                record,
                time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS,
            )
        return
    placeholder = HostFile(
        _DIRECTORY / f"{NATIVE_UNIT_PREFIX}{generation}{SYSTEMD_SLICE_UNIT_SUFFIX}", (), ""
    )
    config = GenerationAdmission(
        generation, uid, gid, _DIRECTORY / RECEIVER_FILENAME, placeholder, placeholder
    )
    slice_sources = config.unit_sources()
    validate_generation_catalog(
        _DIRECTORY, contents, {RECEIVER_FILENAME, UNINSTALLER_FILENAME, "helper"}
    )
    helper = discover_tool("bwrap")
    if hashlib.sha256(contents["helper"]).hexdigest() != helper.digest:
        raise ValueError("Reviewed system helper changed.")
    sources, source = runtime_policy_sources(_DIRECTORY, generation)
    executables = {RECEIVER_FILENAME, UNINSTALLER_FILENAME, "helper"}
    deadline = time.monotonic() + _SETUP_ACTION_TIMEOUT_SECONDS
    record = _DIRECTORY / MANAGEMENT_RECORD_NAME
    if record.exists() or record.is_symlink():
        authority, entry = uninstall_authority(record)
        if authority.digest != record_digest:
            raise ValueError("Reviewed partial installation authority changed.")
        with authority.open_verified(), entry.open_verified():
            uninstall_generation(record, deadline)
    elif record_digest:
        raise ValueError("Fresh install must not supply replacement authority.")
    # Create only fixed missing directories under authenticated administrator ancestry.
    for directory in (_DIRECTORY.parent.parent, _DIRECTORY.parent, _DIRECTORY):
        _directory(directory)
    for name, seed in _SEEDS.items():
        path = _DIRECTORY / name
        if not path.exists() and not path.is_symlink():
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                constants.READ_ONLY_FILE_MODE,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(seed)
                stream.flush()
                os.fchmod(stream.fileno(), constants.READ_ONLY_FILE_MODE)
                os.fsync(stream.fileno())
        bound = bind_host_file(path)
        with bound.open_verified() as descriptor:
            if (
                bound.identity[2] & constants.WRITE_MODE_MASK
                or os.pread(descriptor, len(seed) + 1, 0) != seed
            ):
                raise ValueError("Setup shared lock differs from fixed authority.")
    reservations = _DIRECTORY / NATIVE_RESERVATIONS_DIRECTORY_NAME
    _directory(reservations)
    lock = bind_host_file(_DIRECTORY / MANAGEMENT_LOCK_FILENAME)
    compiler = bind_host_file(_DIRECTORY / COMPILER_LOCK_FILENAME)
    retained = list(reservations.iterdir())
    if retained:
        if len(retained) != 1 or retained[0].name != generation:
            raise ValueError("Partial installation reservation differs from reviewed generation.")
        original_reservation = bind_host_file(retained[0])
        generation_limits(original_reservation, generation)
        clean_unrecorded_installation(
            GenerationAdmission(
                generation, uid, gid, config.receiver, placeholder, original_reservation
            ),
            lock,
            compiler,
            contents,
            executables,
            sources,
            source,
            deadline,
        )
    elif {path.name for path in _DIRECTORY.iterdir()} - {
        *_SEEDS,
        NATIVE_RESERVATIONS_DIRECTORY_NAME,
    }:
        raise ValueError("Install found private files without reserved authority.")
    verify_uninstalled(_DIRECTORY, deadline)
    reservation = reserve_generation(reservations, lock, generation)
    catalog = {
        **contents,
        placeholder.path.name: slice_sources[placeholder.path.name].encode(),
        **_SEEDS,
    }
    with reservation.open_verified() as descriptor:
        catalog[reservation.path.relative_to(_DIRECTORY).as_posix()] = os.pread(
            descriptor, reservation.identity[4], 0
        )
    staged = stage_generation_files(_DIRECTORY, catalog, executables, lock)
    config = GenerationAdmission(
        generation, uid, gid, config.receiver, staged[placeholder.path.name], reservation
    )
    install_generation(config, lock, compiler, catalog, executables, sources, source, deadline)


def manage_setup(action: str, review: Callable[[str], bool], *, main_app: bool = False) -> bool:
    """Review and authenticate a fixed native sandbox setup action.

    Refuse library/headless calls before catalog reads or elevation. The main app presents a concise
    operation/effect summary; this owner binds the exact fixed action, package, loader, original
    record and command internally, then revalidates authority before execution. The standard-library
    loader authenticates the bounded whole input before importing any package code. Credentials stay
    on inherited terminal streams. Cancellation removes only the ordinary temporary input; native
    failures retain protected authority and never enable execution.

    Args:
        action (str): Install or uninstall at the fixed location.
        review (Callable[[str], bool]): Direct human review of the operation/effect summary; only
            literal True authorizes execution.
        main_app (bool): True only for direct main-app dispatch; False refuses elevation by default.

    Returns:
        bool: True after successful checked setup; False when the human declines.

    Raises:
        ValueError: Main-app terminal, action, ordinary identity or reviewed authority differs.
        OSError: Platform, resources, system authentication or checked native setup fails.
        ExceptionGroup: Bounded distinct privileged owner refusals and retry guidance.
    """
    if not main_app or not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("Administrator management requires the main app in a real terminal.")
    if sys.platform != "linux":
        raise OSError("Linux installation management is unavailable on this platform.")
    if action not in {"install", "uninstall"} or min(os.getuid(), os.getgid()) <= 0:
        raise ValueError("Setup requires an ordinary user and a fixed action.")
    record = _DIRECTORY / MANAGEMENT_RECORD_NAME
    has_record = record.exists() or record.is_symlink()
    original = uninstall_authority(record) if has_record else None
    generation = "native_" + uuid.uuid4().hex
    reservations = _DIRECTORY / NATIVE_RESERVATIONS_DIRECTORY_NAME
    if action in {"install", "uninstall"} and not has_record and reservations.exists():
        if protected_path(reservations, directory=True) != reservations:
            raise ValueError("Installation capacity directory aliases other authority.")
        retained = list(reservations.iterdir())
        if len(retained) > 1:
            raise ValueError("Ambiguous partial installation capacity.")
        if retained:
            reserved = bind_host_file(retained[0])
            generation_limits(reserved, retained[0].name)
            generation = retained[0].name
    contents, _, _, _ = runtime_catalog(_DIRECTORY, generation)
    # The temporary privileged import snapshot includes these orchestration owners; the retained
    # runtime remains the smaller catalog needed by its receiver and app-absent uninstaller.
    package = files("loop")
    snapshot = {**contents}
    for name in ("catalog.py", "setup.py"):
        snapshot["loop/execution/sandbox/linux/" + name] = package.joinpath(
            "execution/sandbox/linux/" + name
        ).read_bytes()
    data = json.dumps(
        {
            "contents": {name: value.hex() for name, value in snapshot.items()},
            "runtime": sorted(contents),
        },
        sort_keys=True,
    ).encode("ascii")
    if len(data) > _MAX_SETUP_INPUT_BYTES:
        raise ValueError("Reviewed setup input exceeds its bound.")
    digest = hashlib.sha256(data).hexdigest()
    loader = package.joinpath("execution/sandbox/linux/resources/setup_loader.py").read_text(
        encoding="utf-8"
    )
    sudo = bind_host_file(Path("/usr/bin/sudo"), system_elevation=True)
    python = bind_host_file(SYSTEM_PYTHON_PATH)
    with tempfile.TemporaryDirectory(prefix="loop-native-reviewed-") as temporary:
        path = Path(temporary) / "input.json"
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), constants.OWNER_READ_ONLY_FILE_MODE)
        argv = [
            str(sudo.path),
            "--",
            str(python.path),
            "-I",
            "-S",
            "-B",
            "-c",
            loader,
            str(path),
            digest,
            action,
            generation,
            str(os.getuid()),
            str(os.getgid()),
            original[0].digest if original else "",
        ]
        if action == "uninstall":
            description = (
                "Linux sandbox removal\n"
                "Removes the protected system setup. Sandboxed commands will stay unavailable "
                "until it is installed again.\n"
                "Administrator authentication may be requested next."
            )
        else:
            description = (
                "Linux sandbox installation\n"
                "Adds the protected system setup needed to run sandboxed commands. "
                "No system packages are installed.\n"
                "Administrator authentication may be requested next."
            )
        if review(description) is not True:
            return False
        if original and uninstall_authority(_DIRECTORY / MANAGEMENT_RECORD_NAME) != original:
            raise ValueError(f"Reviewed {action} authority changed after consent.")
        with ExitStack() as pins, tempfile.TemporaryFile() as outcome:
            for bound in (sudo, python, *(original or ())):
                pins.enter_context(bound.open_verified())
            try:
                result = subprocess.run(
                    argv,
                    check=False,
                    timeout=_SETUP_AUTHENTICATION_TIMEOUT_SECONDS,
                    stdout=outcome,
                )
            except subprocess.TimeoutExpired as error:
                raise OSError(
                    "Administrator authentication expired; inspect retained state."
                ) from error
            if result.returncode != 0:
                outcome.seek(0)
                data = outcome.read(_MAX_SETUP_ERROR_REPORT_BYTES + 1)
                try:
                    payload = (
                        json.loads(data) if len(data) <= _MAX_SETUP_ERROR_REPORT_BYTES else None
                    )
                except (ValueError, UnicodeError):
                    payload = None
                if (
                    isinstance(payload, dict)
                    and set(payload) == {"errors"}
                    and isinstance(payload["errors"], list)
                    and 1 <= len(payload["errors"]) <= MAX_NATIVE_SETUP_ERROR_COUNT
                    and all(
                        isinstance(line, str)
                        and line
                        and len(line) <= MAX_NATIVE_SETUP_ERROR_CHARS
                        and all(" " <= char <= "~" for char in line)
                        for line in payload["errors"]
                    )
                ):
                    raise ExceptionGroup(
                        f"Native {action} incomplete.",
                        [OSError(line) for line in dict.fromkeys(payload["errors"])]
                        + [
                            OSError(
                                "Inspect /sandbox status; resolve the refusal, then retry "
                                "the action."
                            )
                        ],
                    )
                raise OSError("Administrator setup failed; inspect /sandbox status, then retry.")
    return True
