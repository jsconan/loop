"""Record Bubblewrap provenance from Ubuntu's signed archive metadata.

Run on an Ubuntu build host with ubuntu-archive-keyring, gpgv and dpkg-deb installed. This
script is a release preparation tool; Loop installation separately fetches verified packages
before administrator authentication and never embeds their executables in its distribution.
"""

import hashlib
import json
import lzma
import subprocess
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.12.0-1"
RELEASE = "stonking"
ARCHITECTURES = ("amd64", "arm64")
ARCHIVE = "https://archive.ubuntu.com/ubuntu/"
KEYRING = Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg")
SIGNERS = {
    "790BC7277767219C42C86F933B4FE6ACC0B21F32",
    "F6ECB3762474EDA9D21B7022871920D1991BC93C",
}
UPSTREAM_SOURCE_SHA256 = "9760d007363e3abba7c747489910f9f82d9fca53ba3bd3282e396fa3c97a3314"
DESTINATION = (
    Path(__file__).resolve().parents[1] / "src/loop/execution/sandbox/linux/resources/bwrap"
)


def _fetch(path: str) -> bytes:
    """Read one bounded HTTPS archive object."""
    with urllib.request.urlopen(ARCHIVE + path, timeout=30) as response:
        if response.url != ARCHIVE + path:
            raise ValueError("Ubuntu archive redirected a provenance input.")
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Ubuntu archive input exceeds its bound.")
    return data


def _field(stanza: str, name: str) -> str:
    """Extract one exact Debian control field."""
    values = [line[len(name) + 2 :] for line in stanza.splitlines() if line.startswith(name + ": ")]
    if len(values) != 1:
        raise ValueError(f"Expected one {name} field.")
    return values[0]


def main() -> None:
    """Verify signed package provenance and copy exact architecture-specific executables."""
    with tempfile.TemporaryDirectory(prefix="loop-bwrap-vendor-") as temporary:
        root = Path(temporary)
        release = root / "InRelease"
        release.write_bytes(_fetch(f"dists/{RELEASE}/InRelease"))
        result = subprocess.run(
            ["gpgv", "--status-fd", "1", "--keyring", str(KEYRING), str(release)],
            check=True,
            capture_output=True,
            text=True,
        )
        signers = {
            line.split()[2]
            for line in result.stdout.splitlines()
            if line.startswith("[GNUPG:] VALIDSIG ")
        }
        if not signers or not signers <= SIGNERS:
            raise ValueError("Ubuntu release signer is unrecognized.")
        signed = release.read_text(encoding="utf-8")
        if _field(signed, "Codename") != RELEASE:
            raise ValueError("Ubuntu signed release identity changed.")
        hashes = {}
        sha_lines = signed.split("SHA512:\n", 1)[1].split("\nAcquire-By-Hash:", 1)[0]
        for line in sha_lines.splitlines():
            parts = line.split()
            if len(parts) == 3:
                hashes[parts[2]] = (parts[0], int(parts[1]))
        source_path = "main/source/Sources.xz"
        source_data = _fetch(f"dists/{RELEASE}/{source_path}")
        source_digest, source_size = hashes[source_path]
        if (
            len(source_data) != source_size
            or hashlib.sha512(source_data).hexdigest() != source_digest
        ):
            raise ValueError("Ubuntu signed source index changed.")
        sources = lzma.decompress(source_data).decode("utf-8")
        source_matches = [
            stanza
            for stanza in sources.split("\n\n")
            if stanza.startswith("Package: bubblewrap\n") and f"Version: {VERSION}\n" in stanza
        ]
        if len(source_matches) != 1:
            raise ValueError("Expected one fixed Bubblewrap source package.")
        source = source_matches[0]
        source_tar = "bubblewrap_0.12.0.orig.tar.xz"
        source_checksums = source.split("Checksums-Sha512:\n", 1)[1].split("\nHomepage:", 1)[0]
        matching_source = [
            line.split() for line in source_checksums.splitlines() if source_tar in line
        ]
        if len(matching_source) != 1 or len(matching_source[0]) != 3:
            raise ValueError("Upstream source archive is absent from Ubuntu's signed index.")
        archive = _fetch(_field(source, "Directory") + "/" + source_tar)
        if (
            len(archive) != int(matching_source[0][1])
            or hashlib.sha512(archive).hexdigest() != matching_source[0][0]
            or hashlib.sha256(archive).hexdigest() != UPSTREAM_SOURCE_SHA256
        ):
            raise ValueError("Ubuntu source differs from the upstream release archive.")
        manifest = {
            "release": RELEASE,
            "version": VERSION,
            "signers": sorted(signers),
            "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
            "source_index_sha512": source_digest,
            "binaries": {},
        }
        for arch in ARCHITECTURES:
            package_path = f"main/binary-{arch}/Packages.xz"
            package_data = _fetch(f"dists/{RELEASE}/{package_path}")
            digest, size = hashes[package_path]
            if len(package_data) != size or hashlib.sha512(package_data).hexdigest() != digest:
                raise ValueError("Ubuntu signed package index changed.")
            packages = lzma.decompress(package_data).decode("utf-8")
            candidates = [
                stanza
                for stanza in packages.split("\n\n")
                if stanza.startswith("Package: bubblewrap\n")
                and _field(stanza, "Version") == VERSION
                and _field(stanza, "Architecture") == arch
            ]
            if len(candidates) != 1:
                raise ValueError("Expected one fixed Bubblewrap package.")
            stanza = candidates[0]
            filename = _field(stanza, "Filename")
            package = _fetch(filename)
            if len(package) != int(_field(stanza, "Size")) or hashlib.sha512(
                package
            ).hexdigest() != _field(stanza, "SHA512"):
                raise ValueError("Ubuntu Bubblewrap package changed.")
            archive_path = root / f"bubblewrap-{arch}.deb"
            archive_path.write_bytes(package)
            extracted = root / f"extracted-{arch}"
            subprocess.run(["dpkg-deb", "--extract", str(archive_path), str(extracted)], check=True)
            binary = (extracted / "usr/bin/bwrap").read_bytes()
            if (
                binary[:5] != b"\x7fELF\x02"
                or int.from_bytes(binary[18:20], "little") != {"amd64": 62, "arm64": 183}[arch]
            ):
                raise ValueError("Ubuntu Bubblewrap ELF architecture changed.")
            manifest["binaries"][arch] = {
                "package_sha512": hashlib.sha512(package).hexdigest(),
                "elf_sha256": hashlib.sha256(binary).hexdigest(),
                "elf_size": len(binary),
                "package_path": filename,
                "index_sha512": digest,
            }
        DESTINATION.mkdir(parents=True, exist_ok=True)
        (DESTINATION / "provenance.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
