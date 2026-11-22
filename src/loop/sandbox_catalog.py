"""Read the installed backends’ declarations without loading native execution code."""

import os
import pwd
import sys
import tomllib
from importlib.resources import files
from pathlib import Path

from . import constants

_PRESETS = {}
for _package in files(__package__).joinpath("execution", "sandbox").iterdir():
    _declaration = _package.joinpath("policy.toml")
    if _declaration.is_file():
        with _declaration.open("rb") as _stream:
            _data = tomllib.load(_stream)
        _PRESETS[_data["platform"]] = _data

SANDBOX_OPTIONS = {
    key: detail for preset in _PRESETS.values() for key, detail in preset.get("options", {}).items()
}


def managed_directory(platform: str | None = None) -> Path | None:
    """Return the fixed administrator policy location for the given system.

    Args:
        platform (str | None): Operating system identifier. Defaults to the current system.

    Returns:
        Path | None: Trusted system location, or None for an unsupported platform.
    """
    directory = _PRESETS.get(sys.platform if platform is None else platform, {}).get(
        "managed_directory"
    )
    if directory is None:
        return None
    return Path(
        directory.format(app_name=constants.APP_NAME, app_display_name=constants.APP_DISPLAY_NAME)
    )


def sandbox_options(platform: str | None = None) -> dict[str, dict[str, str | bool]]:
    """Return the optional authorities declared by one installed backend.

    Args:
        platform (str | None): Operating system identifier. Defaults to the current system.

    Returns:
        dict[str, dict[str, str | bool]]: Option keys, notices and preparation settings.
    """
    return _PRESETS.get(sys.platform if platform is None else platform, {}).get("options", {})


def hardlink_source_paths(
    options: tuple[str, ...], platform: str | None = None
) -> tuple[Path, ...]:
    """Return backend-declared metadata sources for selected options under the account home.

    Args:
        options (tuple[str, ...]): Selected option keys.
        platform (str | None): Operating system identifier, defaulting to the current host.

    Returns:
        tuple[Path, ...]: Fixed host paths where the backend may verify other hardlink names.

    Raises:
        ValueError: The operating system cannot identify the account home.
    """
    declarations = sandbox_options(platform)
    sources = tuple(
        declarations[name]["hardlink_source"]
        for name in options
        if "hardlink_source" in declarations.get(name, {})
    )
    if not sources:
        return ()
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError as error:
        raise ValueError(
            "The account home for the selected sandbox access is unavailable."
        ) from error
    return tuple(home / source for source in sources)


def validate_sandbox_options(options: tuple[str, ...]) -> tuple[str, ...]:
    """Bind a deterministic, nonduplicated set of known backend option keys.

    Args:
        options (tuple[str, ...]): Requested backend options.

    Returns:
        tuple[str, ...]: Canonical ordering used in policy and approval signatures.

    Raises:
        ValueError: An option is unknown or appears more than once.
    """
    if len(set(options)) != len(options) or not SANDBOX_OPTIONS.keys() >= set(options):
        raise ValueError("Unknown or duplicate sandbox option.")
    return tuple(sorted(options))


def system_paths(platform: str | None = None) -> tuple[Path, ...]:
    """Return the native system runtime's path spellings.

    Args:
        platform (str | None): Platform name. None selects the current system.

    Returns:
        tuple[Path, ...]: Shipped system path spellings, in native preparation order.
    """
    return tuple(
        Path(path)
        for path in _PRESETS.get(sys.platform if platform is None else platform, {}).get(
            "system", ()
        )
    )


def installed_paths(platform: str | None = None) -> tuple[Path, ...]:
    """Return installed runtime preset candidates from the platform's known layouts.

    Args:
        platform (str | None): Platform name. None selects the current system.

    Returns:
        tuple[Path, ...]: Candidate read-only directories for present installation stores.
    """
    preset = _PRESETS.get(sys.platform if platform is None else platform, {})
    return (
        *(Path(path) for path in preset.get("read_only", ())),
        *(
            path
            for group in preset.get("installed", ())
            if (store := Path(group["store"])).is_dir() and store.resolve() == store
            for path in (store, *(Path(value) for value in group["paths"]))
        ),
    )


def package_stores(platform: str | None = None) -> tuple[Path, ...]:
    """Return the standard package stores described by the installed-runtime presets.

    Args:
        platform (str | None): Platform name. None selects the current system.

    Returns:
        tuple[Path, ...]: Supported package store spellings, including absent installations.
    """
    return tuple(
        Path(group["store"])
        for group in _PRESETS.get(sys.platform if platform is None else platform, {}).get(
            "installed", ()
        )
    )
