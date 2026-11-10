"""Fetch a reviewed fixed Bubblewrap before administrator sandbox setup."""

import fcntl
import hashlib
import io
import json
import lzma
import os
import platform
import re
import subprocess
import tarfile
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path

from .... import constants
from .constants import PACKAGE_NAMESPACE
from .host import (
    bind_host_file,
    discover_tool,
    require_fixed_bubblewrap,
    require_managed_bubblewrap_bytes,
)

_ARCHIVE = "https://archive.ubuntu.com/ubuntu/"
_RELEASE = "stonking"
_VERSION = "0.12.0-1"
_SIGNER = "F6ECB3762474EDA9D21B7022871920D1991BC93C"
_MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
_MAX_INDEX_BYTES = 80 * 1024 * 1024
_LINUX_F_ADD_SEALS = 1033
_LINUX_FULL_FILE_SEALS = 0xF
_ARCHITECTURES = {"x86_64": "amd64", "aarch64": "arm64"}


def _field(stanza: str, name: str) -> str:
    """Read exactly one non-continuation Debian control field."""
    values = [line[len(name) + 2 :] for line in stanza.splitlines() if line.startswith(name + ": ")]
    if len(values) != 1:
        raise ValueError(f"Ubuntu Bubblewrap metadata lacks one {name} field.")
    return values[0]


def _fetch(path: str) -> bytes:
    """Fetch a bounded exact HTTPS object without accepting a mirror redirect."""
    url = _ARCHIVE + path
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.url != url or response.status != 200:
            raise OSError("Ubuntu Bubblewrap archive location changed.")
        data = response.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(data) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Ubuntu Bubblewrap archive object exceeds its size limit.")
    return data


def _signed_release() -> str:
    """Authenticate the exact Ubuntu release index with the protected archive keyring."""
    release = _fetch(f"dists/{_RELEASE}/InRelease")
    verifier = bind_host_file(Path("/usr/bin/gpgv"))
    keyring = bind_host_file(Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg"))
    with tempfile.TemporaryDirectory(prefix=f"{constants.APP_NAME}-bwrap-release-") as directory:
        path = Path(directory) / "InRelease"
        path.write_bytes(release)
        with verifier.open_verified(), keyring.open_verified():
            result = subprocess.run(
                [
                    str(verifier.path),
                    "--status-fd",
                    "1",
                    "--keyring",
                    str(keyring.path),
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
    signers = {
        line.split()[2]
        for line in result.stdout.splitlines()
        if line.startswith("[GNUPG:] VALIDSIG ")
    }
    if result.returncode != 0 or signers != {_SIGNER}:
        raise ValueError("Ubuntu Bubblewrap release signature is invalid or unrecognized.")
    signed = release.decode("utf-8")
    if _field(signed, "Codename") != _RELEASE or _field(signed, "Version") != "26.10":
        raise ValueError("Ubuntu Bubblewrap release identity changed.")
    return signed


def _package_record(release: str, architecture: str) -> str:
    """Bind one package stanza through the signed release's SHA512 index entry."""
    index = f"main/binary-{architecture}/Packages.xz"
    section = release.split("SHA512:\n", 1)[1].split("\nAcquire-By-Hash:", 1)[0]
    entries = [line.split() for line in section.splitlines() if line.endswith(" " + index)]
    if len(entries) != 1 or len(entries[0]) != 3:
        raise ValueError("Ubuntu Bubblewrap package index is absent or ambiguous.")
    expected, size, name = entries[0]
    if name != index or re.fullmatch(r"[0-9a-f]{128}", expected) is None:
        raise ValueError("Ubuntu Bubblewrap package index digest is malformed.")
    data = _fetch(f"dists/{_RELEASE}/{index}")
    if len(data) != int(size) or hashlib.sha512(data).hexdigest() != expected:
        raise ValueError("Ubuntu signed Bubblewrap package index changed.")
    decoder = lzma.LZMADecompressor()
    contents = decoder.decompress(data, max_length=_MAX_INDEX_BYTES + 1)
    if len(contents) > _MAX_INDEX_BYTES or not decoder.eof:
        raise ValueError("Ubuntu Bubblewrap package index exceeds its size limit.")
    packages = contents.decode("utf-8")
    matches = [
        stanza
        for stanza in packages.split("\n\n")
        if stanza.startswith("Package: bubblewrap\n")
        and _field(stanza, "Version") == _VERSION
        and _field(stanza, "Architecture") == architecture
    ]
    if len(matches) != 1:
        raise ValueError("Ubuntu fixed Bubblewrap package is absent or ambiguous.")
    return matches[0]


def _extract_binary(package: bytes) -> bytes:
    """Extract one root-owned regular executable from a sealed reviewed Debian package."""
    tool = bind_host_file(Path("/usr/bin/dpkg-deb"))
    descriptor = os.memfd_create("loop-reviewed-bwrap-deb", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        if os.write(descriptor, package) != len(package):
            raise OSError("Incomplete reviewed Bubblewrap package staging.")
        fcntl.fcntl(descriptor, _LINUX_F_ADD_SEALS, _LINUX_FULL_FILE_SEALS)
        with tool.open_verified() as executable:
            result = subprocess.run(
                [
                    f"/proc/self/fd/{executable}",
                    "--fsys-tarfile",
                    f"/proc/self/fd/{descriptor}",
                ],
                pass_fds=(executable, descriptor),
                capture_output=True,
                timeout=10,
                check=False,
            )
    finally:
        os.close(descriptor)
    if result.returncode != 0 or len(result.stdout) > 4 * 1024 * 1024:
        raise ValueError("Reviewed Ubuntu Bubblewrap package cannot be extracted.")
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        matches = [member for member in archive if member.name == "./usr/bin/bwrap"]
        if (
            len(matches) != 1
            or not matches[0].isreg()
            or matches[0].uid != 0
            or matches[0].gid != 0
            or matches[0].mode != 0o755
        ):
            raise ValueError("Ubuntu Bubblewrap package executable ownership or mode changed.")
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError("Ubuntu Bubblewrap package executable is unavailable.")
        return stream.read()


def _download_managed_bubblewrap() -> bytes:
    """Fetch a signed, digest-pinned Ubuntu build matching this machine."""
    architecture = _ARCHITECTURES.get(platform.machine())
    if architecture is None:
        raise OSError("No managed Bubblewrap is qualified for this architecture.")
    manifest = json.loads(
        files(PACKAGE_NAMESPACE)
        .joinpath("execution/sandbox/linux/resources/bwrap/provenance.json")
        .read_text(encoding="utf-8")
    )
    if (
        manifest["release"] != _RELEASE
        or manifest["version"] != _VERSION
        or manifest["signers"] != [_SIGNER]
    ):
        raise ValueError("Managed Bubblewrap provenance manifest changed.")
    entry = manifest["binaries"][architecture]
    record = _package_record(_signed_release(), architecture)
    filename = _field(record, "Filename")
    if filename != entry["package_path"] or _field(record, "SHA512") != entry["package_sha512"]:
        raise ValueError("Ubuntu Bubblewrap package differs from reviewed provenance.")
    package = _fetch(filename)
    if (
        len(package) != int(_field(record, "Size"))
        or hashlib.sha512(package).hexdigest() != entry["package_sha512"]
    ):
        raise ValueError("Downloaded Ubuntu Bubblewrap package differs from reviewed bytes.")
    binary = _extract_binary(package)
    if (
        len(binary) != entry["elf_size"]
        or hashlib.sha256(binary).hexdigest() != entry["elf_sha256"]
    ):
        raise ValueError("Extracted Ubuntu Bubblewrap differs from reviewed bytes.")
    require_managed_bubblewrap_bytes(binary)
    return binary


def selected_bubblewrap_bytes() -> bytes:
    """Select a fixed protected system helper or fetch a reviewed Ubuntu 24.04 helper.

    Returns:
        bytes: Exact compatible Bubblewrap executable to copy into the protected catalog.

    Raises:
        ValueError: Host or archive authority, signature, artifact or ELF validation fails.
        OSError: No fixed host or downloadable qualified helper is available.
        TimeoutError: Native helper or archive operation exceeds its bound.
    """
    try:
        helper = discover_tool("bwrap")
    except OSError:
        helper = None
    if helper is not None:
        try:
            require_fixed_bubblewrap(helper)
        except ValueError:
            pass
        else:
            with helper.open_verified() as descriptor:
                data = os.pread(descriptor, helper.identity[4] + 1, 0)
            if len(data) != helper.identity[4] or hashlib.sha256(data).hexdigest() != helper.digest:
                raise ValueError("Authenticated helper changed during catalog read.")
            return data
    return _download_managed_bubblewrap()
