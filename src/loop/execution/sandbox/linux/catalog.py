"""Assemble reviewed distribution bytes for an isolated Linux command installation.

This catalog is administrator review input, not authority to load untrusted package code. Its
receiver admits authenticated bounded frames and composes trusted setup with enforcing payload
entry. Catalog assembly grants no administrator consent or native qualification.
"""

import re
from importlib.resources import files
from pathlib import Path

from .constants import (
    BOOT_FILENAME,
    CACHE_LOCK_FILENAME,
    PACKAGE_ARTIFACT_PREFIX,
    PACKAGE_NAMESPACE,
    PAYLOAD_FILENAME,
    RECEIVER_FILENAME,
    UNINSTALLER_FILENAME,
)
from .installation import validate_generation_catalog
from .managed_bubblewrap import selected_bubblewrap_bytes
from .policy import runtime_policy_sources

_MODULES = (
    "constants.py",
    "utils/process.py",
    "utils/virtual_path.py",
    "utils/hashing.py",
    "utils/json.py",
    "permissions/protection.py",
    "execution/sandbox/contracts.py",
    "execution/sandbox/constants.py",
    "execution/sandbox/linux/constants.py",
    "execution/sandbox/linux/host.py",
    "execution/sandbox/linux/installation.py",
    "execution/sandbox/linux/mountinfo.py",
    "execution/sandbox/linux/admission.py",
    "execution/sandbox/linux/transport.py",
    "execution/sandbox/linux/egress.py",
    "execution/sandbox/linux/management.py",
    "execution/sandbox/linux/records.py",
    "execution/sandbox/linux/runtime.py",
    "execution/sandbox/linux/compiler.py",
    "execution/sandbox/linux/policy.py",
    "execution/sandbox/linux/paths.py",
    "execution/sandbox/linux/namespace.py",
    "execution/sandbox/linux/bootstrap.py",
    "execution/sandbox/linux/preflight.py",
    "execution/sandbox/linux/seccomp.py",
    "execution/sandbox/linux/supervision.py",
)
_BARRELS = (
    "",
    "utils/",
    "permissions/",
    "execution/",
    "execution/sandbox/",
    "execution/sandbox/linux/",
)
_RESOURCES = {
    BOOT_FILENAME: "boot.py",
    RECEIVER_FILENAME: "receiver.py",
    UNINSTALLER_FILENAME: "uninstall.py",
    PAYLOAD_FILENAME: "payload.py",
}


def runtime_member_names() -> tuple[str, ...]:
    """List every fixed private runtime file name without reading host authority.

    Returns:
        tuple[str, ...]: Relative file names installed by the current package catalog.
    """
    return (
        *(PACKAGE_ARTIFACT_PREFIX + name for name in _MODULES),
        *(PACKAGE_ARTIFACT_PREFIX + prefix + "__init__.py" for prefix in _BARRELS),
        *_RESOURCES,
        "helper",
        "mask.empty",
        CACHE_LOCK_FILENAME,
    )


def runtime_catalog(
    directory: Path,
    generation: str,
    *,
    helper_bytes: bytes | None = None,
    management_only: bool = False,
) -> tuple[dict[str, bytes], set[str], dict[str, dict[str, bytes]], bytes]:
    """Build a trusted runtime catalog from installed resources.

    Read packaged Python resources, never plans or developer paths. Use passive native package
    barrels so the retained uninstaller remains independent of application dependencies. Copy the
    authenticated fixed system or reviewed downloaded Bubblewrap bytes into a private helper.
    Do not modify a shared binary/profile.
    Include a fresh read-only cache-lock seed and fixed installed receiver/uninstaller resources.
    The administrator installer must authenticate the exact reviewed catalog before executing it.
    Trusted bootstrap grants precede separately compiled restrictive payload entry. Successful setup
    enables ordinary native admission; payload entry verifies enforcement before source starts.
    Setup does not certify release qualification.

    Args:
        directory (Path): Intended literal absolute private runtime root, without aliases or '..'.
        generation (str): New simple ASCII generation identity, at most 120 characters.
        helper_bytes (bytes | None): Already selected helper bytes, or None to select once here.
        management_only (bool): Omit the helper for recorded or absent uninstall authority.

    Returns:
        tuple[dict[str, bytes], set[str], dict[str, dict[str, bytes]], bytes]: Reviewed file
            catalog, executable names, fixed root bootstrap sources and literal global attachment.

    Raises:
        ValueError: Intended naming, catalog bounds or host helper authority is invalid or changed.
        OSError: A packaged resource or authenticated host Bubblewrap is unavailable.
    """
    if (
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,119}", generation) is None
        or re.fullmatch(r"/[A-Za-z0-9_./-]+", str(directory)) is None
        or ".." in directory.parts
        or directory == Path("/")
    ):
        raise ValueError("Invalid fixed runtime catalog location or generation.")

    package = files(PACKAGE_NAMESPACE)
    catalog = {
        PACKAGE_ARTIFACT_PREFIX + name: package.joinpath(name).read_bytes() for name in _MODULES
    }
    catalog.update({PACKAGE_ARTIFACT_PREFIX + prefix + "__init__.py": b"" for prefix in _BARRELS})
    resources = package.joinpath("execution/sandbox/linux/resources")
    catalog.update(
        {name: resources.joinpath(resource).read_bytes() for name, resource in _RESOURCES.items()}
    )
    if not management_only:
        catalog["helper"] = (
            helper_bytes if helper_bytes is not None else selected_bubblewrap_bytes()
        )

    catalog["mask.empty"] = b""
    catalog[CACHE_LOCK_FILENAME] = (generation + "\n").encode("ascii")
    executable = {RECEIVER_FILENAME, UNINSTALLER_FILENAME}
    if not management_only:
        executable.add("helper")
    validate_generation_catalog(directory, catalog, executable)
    sources, source = runtime_policy_sources(directory, generation)
    return catalog, executable, sources, source
