"""Authenticate persisted Linux generation records and their complete original inventory."""

import os
import time
from pathlib import Path

from ....constants import MAX_NATIVE_INSTALLATION_ENTRIES
from ....utils.json import load_unique_json
from .admission import GenerationAdmission
from .host import HostFile, bind_host_file
from .installation import GenerationLifecycle

MANAGEMENT_RECORD_NAME = "management.json"
MAX_MANAGEMENT_RECORD_BYTES = 65_536


def _check_deadline(deadline: float | None) -> None:
    """Stop record reconstruction when its optional deadline expires."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Native installation record deadline expired.")


def _bound(value: dict, deadline: float | None) -> HostFile:
    """Refuse malformed or changed saved authority rather than adopting current files."""
    bound = HostFile.from_record(value)
    if bind_host_file(bound.path, deadline=deadline) != bound:
        raise ValueError("Saved installation authority changed; management refused.")
    return bound


def read_generation(
    record: Path,
    *,
    partial: bool = False,
    deadline: float | None = None,
) -> tuple[GenerationLifecycle, HostFile]:
    """Authenticate a persisted generation record and every original artifact binding.

    Partial mode is reserved for checked teardown: missing owned files remain represented by
    their original bindings, while every surviving file and shared management authority must
    still match. The optional deadline bounds file hashing and inventory traversal.

    Args:
        record (Path): Protected original management record for one fixed generation.
        partial (bool): Allow missing owned artifacts only for checked teardown; defaults to
            False.
        deadline (float | None): Optional absolute monotonic bound for authority verification.

    Returns:
        tuple[GenerationLifecycle, HostFile]: Reconstructed original lifecycle and its bound
            management record.

    Raises:
        ValueError: Record schema, inventory or saved authority is invalid or has changed.
        TimeoutError: The optional authority-verification deadline expires.
        OSError: Required record or original filesystem authority cannot be inspected.
    """
    _check_deadline(deadline)
    authority = bind_host_file(record, deadline=deadline)
    with authority.open_verified(deadline=deadline) as descriptor:
        data = os.pread(descriptor, MAX_MANAGEMENT_RECORD_BYTES + 1, 0)
    if (
        record.name != MANAGEMENT_RECORD_NAME
        or authority.identity[4] > MAX_MANAGEMENT_RECORD_BYTES
        or authority.identity[2] & 0o222
    ):
        raise ValueError("Invalid protected installation management record.")
    value = load_unique_json(data, duplicate_message="Duplicate installation management field.")
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema",
            "config",
            "files",
            "lock",
            "compiler_lock",
            "artifacts",
            "namespace",
            "policies",
            "helper",
            "compiled",
            "source",
        }
        or not isinstance(value["schema"], int)
        or isinstance(value["schema"], bool)
        or value["schema"] != 1
        or not isinstance(value["config"], dict)
        or set(value["config"]) != {"generation", "uid", "gid", "receiver", "slice", "reservation"}
        or not isinstance(value["files"], dict)
        or set(value["files"]) != {"receiver", "socket_unit", "service_unit"}
        or not isinstance(value["artifacts"], dict)
        or not 1 <= len(value["artifacts"]) <= MAX_NATIVE_INSTALLATION_ENTRIES
        or not isinstance(value["namespace"], list)
        or len(value["namespace"]) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in value["namespace"]
        )
        or not isinstance(value["policies"], dict)
    ):
        raise ValueError("Invalid installation management schema.")
    config_value = value["config"]
    if (
        not isinstance(config_value["generation"], str)
        or not isinstance(config_value["uid"], int)
        or isinstance(config_value["uid"], bool)
        or not isinstance(config_value["gid"], int)
        or isinstance(config_value["gid"], bool)
    ):
        # A malformed persisted record is a value error even when a field's type is wrong.
        raise ValueError("Invalid saved generation identity.")  # noqa: TRY004
    bound_record = HostFile.from_record if partial else lambda value: _bound(value, deadline)
    receiver = bound_record(config_value["receiver"])
    config = GenerationAdmission(
        config_value["generation"],
        config_value["uid"],
        config_value["gid"],
        receiver.path,
        bound_record(config_value["slice"]),
        bound_record(config_value["reservation"]),
    )
    config.unit_sources()
    if receiver.path.parent != record.parent:
        raise ValueError("Management record differs from its private generation directory.")
    artifacts = {name: bound_record(saved) for name, saved in value["artifacts"].items()}
    if any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or name == MANAGEMENT_RECORD_NAME
        or artifacts[name].path != record.parent / name
        for name in artifacts
    ):
        raise ValueError("Management artifact inventory differs from its private directory.")

    def contents(name: object) -> bytes:
        """Read only an originally bound artifact named by this protected record."""
        _check_deadline(deadline)
        if not isinstance(name, str) or name not in artifacts:
            raise ValueError("Management policy lacks its original artifact authority.")
        original = artifacts[name]
        if partial and not original.path.exists() and not original.path.is_symlink():
            return b""
        with original.open_verified(deadline=deadline) as descriptor:
            return os.pread(descriptor, original.identity[4] + 1, 0)

    if any(not isinstance(profiles, dict) for profiles in value["policies"].values()):
        raise ValueError("Invalid saved namespace policy inventory.")
    result = GenerationLifecycle(
        config,
        {role: bound_record(saved) for role, saved in value["files"].items()},
        bound_record(value["lock"]),
        bound_record(value["compiler_lock"]),
        artifacts,
        tuple(value["namespace"]),
        {
            namespace: {profile: contents(name) for profile, name in profiles.items()}
            for namespace, profiles in value["policies"].items()
        },
        bound_record(value["helper"]),
        contents(value["compiled"]),
        contents(value["source"]),
    )
    expected = {record, *(bound.path for bound in artifacts.values())}
    expected.update(
        bound.path
        for bound in (result.lock, result.compiler_lock, config.reservation)
        if bound.path.is_relative_to(record.parent)
    )
    expected.update(
        parent
        for path in tuple(expected)
        for parent in path.parents
        if parent.is_relative_to(record.parent)
    )
    for count, path in enumerate(record.parent.rglob("*"), 1):
        _check_deadline(deadline)
        if count > MAX_NATIVE_INSTALLATION_ENTRIES or path not in expected:
            raise ValueError("Unknown private management inventory; authority retained.")
    if partial:
        for bound in (result.lock, result.compiler_lock):
            if bind_host_file(bound.path, deadline=deadline) != bound:
                raise ValueError("Shared management authority changed during partial retirement.")
        for bound in (
            *result.artifacts.values(),
            *result.files.values(),
            result.config.slice_unit,
            result.config.reservation,
        ):
            _check_deadline(deadline)
            if not bound.path.exists() and not bound.path.is_symlink():
                continue
            current_bound = bind_host_file(bound.path, deadline=deadline)
            if (
                bound.path == result.helper.path
                and current_bound != bound
                and current_bound.identity[:2] == bound.identity[:2]
                and current_bound.identity[3:6] == bound.identity[3:6]
                and current_bound.digest == bound.digest
                and current_bound.identity[2] & 0o777 == 0o444
                and current_bound.identity[6] >= bound.identity[6]
            ):
                continue
            if current_bound != bound:
                raise ValueError("Saved partial installation authority changed.")
    with authority.open_verified(deadline=deadline):
        pass
    return result, authority
