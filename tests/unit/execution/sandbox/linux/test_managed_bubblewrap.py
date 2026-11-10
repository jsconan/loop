"""Verify signed live Bubblewrap selection without network or native processes."""

import hashlib
import io
import json
import lzma
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import managed_bubblewrap as managed


class Response:
    """Model one exact HTTPS response with a bounded read."""

    def __init__(self, url, data):
        self.url = url
        self.status = 200
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.data


def package_tar(binary=b"reviewed executable", *, mode=0o755, name="./usr/bin/bwrap", uid=0):
    """Create disposable package contents for verified extraction cases."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(binary)
        member.mode = mode
        member.uid = uid
        member.gid = 0
        archive.addfile(member, io.BytesIO(binary))
    return stream.getvalue()


@pytest.fixture
def archive(monkeypatch):
    """Provide a complete signed metadata chain and pinned package fixture."""
    package = b"reviewed Debian package"
    binary = b"reviewed executable"
    path = "pool/main/b/bubblewrap/bubblewrap_0.12.0-1_arm64.deb"
    index_path = "main/binary-arm64/Packages.xz"
    stanza = (
        "Package: bubblewrap\nVersion: 0.12.0-1\nArchitecture: arm64\n"
        f"Filename: {path}\nSize: {len(package)}\n"
        f"SHA512: {hashlib.sha512(package).hexdigest()}\n"
    )
    index = lzma.compress((stanza + "\n").encode())
    release = (
        "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA512\n\n"
        "Codename: stonking\nVersion: 26.10\nSHA512:\n"
        f" {hashlib.sha512(index).hexdigest()} {len(index)} {index_path}\n"
        "Acquire-By-Hash: yes\n"
    ).encode()
    objects = {
        "dists/stonking/InRelease": release,
        "dists/stonking/" + index_path: index,
        path: package,
    }
    manifest = {
        "release": "stonking",
        "version": "0.12.0-1",
        "signers": ["F6ECB3762474EDA9D21B7022871920D1991BC93C"],
        "binaries": {
            "arm64": {
                "package_path": path,
                "package_sha512": hashlib.sha512(package).hexdigest(),
                "elf_size": len(binary),
                "elf_sha256": hashlib.sha256(binary).hexdigest(),
            }
        },
    }
    resource = Mock()
    resource.joinpath.return_value.read_text.side_effect = lambda **_kwargs: json.dumps(manifest)
    monkeypatch.setattr(managed, "files", Mock(return_value=resource))
    monkeypatch.setattr(managed.platform, "machine", lambda: "aarch64")
    bound = Mock()
    bound.path = Path("/usr/bin/reviewed-tool")
    bound.open_verified.return_value = nullcontext(11)
    binding = Mock(return_value=bound)
    monkeypatch.setattr(managed, "bind_host_file", binding)
    verifier = SimpleNamespace(
        returncode=0,
        stdout="[GNUPG:] VALIDSIG F6ECB3762474EDA9D21B7022871920D1991BC93C\n",
    )
    extractor = SimpleNamespace(returncode=0, stdout=package_tar(binary))
    process = Mock(side_effect=[verifier, extractor])
    monkeypatch.setattr(managed.subprocess, "run", process)
    monkeypatch.setattr(managed.os, "MFD_CLOEXEC", 1, raising=False)
    monkeypatch.setattr(managed.os, "MFD_ALLOW_SEALING", 2, raising=False)
    monkeypatch.setattr(managed.os, "memfd_create", Mock(return_value=42), raising=False)
    monkeypatch.setattr(managed.os, "write", Mock(return_value=len(package)))
    real_close = managed.os.close
    closed = Mock()

    def close(descriptor):
        """Preserve temporary-file cleanup while observing the sealed package descriptor."""
        if descriptor == 42:
            closed(descriptor)
        else:
            real_close(descriptor)

    monkeypatch.setattr(managed.os, "close", close)
    monkeypatch.setattr(managed.fcntl, "fcntl", Mock())
    qualified = Mock()
    monkeypatch.setattr(managed, "require_managed_bubblewrap_bytes", qualified)

    def open_url(url, timeout):
        """Serve only the exact canonical archive objects."""
        assert timeout == 30
        return Response(url, objects[url.removeprefix("https://archive.ubuntu.com/ubuntu/")])

    monkeypatch.setattr(managed.urllib.request, "urlopen", open_url)
    return SimpleNamespace(
        objects=objects,
        manifest=manifest,
        package=package,
        binary=binary,
        path=path,
        index_path=index_path,
        verifier=verifier,
        extractor=extractor,
        process=process,
        binding=binding,
        qualified=qualified,
        closed=closed,
    )


def test_live_archive_selection_verifies_signature_package_and_executable(archive, monkeypatch):
    """Stock 0.9 selects the reviewed signed package before any privileged operation."""
    monkeypatch.setattr(managed.constants, "APP_NAME", "renamed-app")
    helper = Mock()
    monkeypatch.setattr(managed, "discover_tool", Mock(return_value=helper))
    monkeypatch.setattr(managed, "require_fixed_bubblewrap", Mock(side_effect=ValueError("0.9")))
    assert managed.selected_bubblewrap_bytes() == archive.binary
    archive.qualified.assert_called_once_with(archive.binary)
    managed.files.assert_called_once_with(managed.PACKAGE_NAMESPACE)
    assert archive.process.call_count == 2
    assert archive.binding.call_count == 3


def test_missing_system_helper_uses_reviewed_archive(archive, monkeypatch):
    """Absent system Bubblewrap uses the same signed managed path."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=OSError("missing")))
    assert managed.selected_bubblewrap_bytes() == archive.binary


def test_fixed_system_helper_is_copied_without_download(archive, monkeypatch):
    """A protected fixed helper retains its pinned-byte fast path."""
    helper = Mock(digest=hashlib.sha256(b"fixed").hexdigest(), identity=(0, 0, 0, 0, 5))
    helper.open_verified.return_value = nullcontext(11)
    monkeypatch.setattr(managed, "discover_tool", Mock(return_value=helper))
    monkeypatch.setattr(managed, "require_fixed_bubblewrap", Mock())
    monkeypatch.setattr(managed.os, "pread", Mock(return_value=b"fixed"))
    assert managed.selected_bubblewrap_bytes() == b"fixed"
    archive.process.assert_not_called()


@pytest.mark.parametrize("value", [b"short", b"other"])
def test_fixed_system_helper_refuses_changed_read(archive, monkeypatch, value):
    """Short or changed bytes never become the copied system helper."""
    helper = Mock(digest=hashlib.sha256(b"fixed").hexdigest(), identity=(0, 0, 0, 0, 5))
    helper.open_verified.return_value = nullcontext(11)
    monkeypatch.setattr(managed, "discover_tool", Mock(return_value=helper))
    monkeypatch.setattr(managed, "require_fixed_bubblewrap", Mock())
    monkeypatch.setattr(managed.os, "pread", Mock(return_value=value))
    with pytest.raises(ValueError, match="helper changed"):
        managed.selected_bubblewrap_bytes()
    archive.process.assert_not_called()


def test_unsafe_system_authority_does_not_trigger_download(archive, monkeypatch):
    """An ambiguous protected host path fails closed before network use."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=ValueError("unsafe path")))
    with pytest.raises(ValueError, match="unsafe path"):
        managed.selected_bubblewrap_bytes()
    archive.process.assert_not_called()


@pytest.mark.parametrize(
    "case", ["architecture", "manifest", "signature", "release", "package", "bytes", "elf"]
)
def test_live_archive_refuses_changed_provenance(archive, monkeypatch, case):
    """Each provenance boundary rejects an unsupported or altered download."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=OSError("missing")))
    if case == "architecture":
        monkeypatch.setattr(managed.platform, "machine", lambda: "riscv64")
    elif case == "manifest":
        archive.manifest["version"] = "0.9.0"
    elif case == "signature":
        archive.verifier.stdout = "[GNUPG:] VALIDSIG wrong\n"
    elif case == "release":
        archive.objects["dists/stonking/InRelease"] = archive.objects[
            "dists/stonking/InRelease"
        ].replace(b"Version: 26.10", b"Version: 24.04")
    elif case == "package":
        archive.manifest["binaries"]["arm64"]["package_sha512"] = "0" * 128
    elif case == "bytes":
        archive.objects[archive.path] = b"changed package"
    else:
        archive.manifest["binaries"]["arm64"]["elf_sha256"] = "0" * 64
    with pytest.raises(
        (OSError, ValueError), match="qualified|provenance|signature|identity|package|bytes"
    ):
        managed.selected_bubblewrap_bytes()


@pytest.mark.parametrize(
    "case", ["redirect", "oversize", "index", "index_hash", "index_limit", "stanza"]
)
def test_live_archive_refuses_bad_transport_or_index(archive, monkeypatch, case):
    """Redirects, oversized content, bad signed checksums and missing stanzas fail closed."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=OSError("missing")))
    if case == "redirect":

        def redirected(url, timeout):
            """Model a noncanonical response target."""
            assert timeout == 30
            response = Response("https://other.invalid/", b"")
            return response

        monkeypatch.setattr(managed.urllib.request, "urlopen", redirected)
    elif case == "oversize":
        monkeypatch.setattr(managed, "_MAX_DOWNLOAD_BYTES", 1)
    elif case == "index":
        archive.objects["dists/stonking/InRelease"] = (
            b"Codename: stonking\nVersion: 26.10\nSHA512:\nAcquire-By-Hash: yes\n"
        )
    elif case == "index_hash":
        archive.objects["dists/stonking/" + archive.index_path] = b"changed"
    elif case == "index_limit":
        monkeypatch.setattr(managed, "_MAX_INDEX_BYTES", 1)
    else:
        archive.objects["dists/stonking/" + archive.index_path] = lzma.compress(
            b"Package: other\n\n"
        )
        index = archive.objects["dists/stonking/" + archive.index_path]
        release = archive.objects["dists/stonking/InRelease"].decode()
        old = release.split("SHA512:\n", 1)[1].split("\nAcquire-By-Hash:", 1)[0]
        new = f" {hashlib.sha512(index).hexdigest()} {len(index)} {archive.index_path}"
        archive.objects["dists/stonking/InRelease"] = release.replace(old, new).encode()
    with pytest.raises((OSError, ValueError)):
        managed.selected_bubblewrap_bytes()


@pytest.mark.parametrize("case", ["extract", "output", "mode", "uid", "missing", "short"])
def test_live_archive_refuses_untrusted_extraction(archive, monkeypatch, case):
    """Only one complete root-owned executable with approved mode is extracted."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=OSError("missing")))
    if case == "extract":
        archive.extractor.returncode = 1
    elif case == "output":
        monkeypatch.setattr(managed, "_MAX_INDEX_BYTES", 80 * 1024 * 1024)
        archive.extractor.stdout = b"x" * (4 * 1024 * 1024 + 1)
    elif case == "mode":
        archive.extractor.stdout = package_tar(mode=0o777)
    elif case == "uid":
        archive.extractor.stdout = package_tar(uid=1000)
    elif case == "missing":
        archive.extractor.stdout = package_tar(name="./other")
    else:
        managed.os.write.return_value = 0
    with pytest.raises((ValueError, OSError), match="staging|extract|ownership"):
        managed.selected_bubblewrap_bytes()
    archive.closed.assert_called_once_with(42)


def test_signed_release_requires_complete_identity_fields(archive, monkeypatch):
    """A signed index without its one release identity cannot authorize a package."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=OSError("missing")))
    archive.objects["dists/stonking/InRelease"] = archive.objects[
        "dists/stonking/InRelease"
    ].replace(b"Codename: stonking\n", b"")
    with pytest.raises(ValueError, match="Codename field"):
        managed.selected_bubblewrap_bytes()


def test_signed_index_requires_well_formed_digest(archive, monkeypatch):
    """Malformed SHA512 text cannot identify a package index."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=OSError("missing")))
    index = archive.objects["dists/stonking/" + archive.index_path]
    archive.objects["dists/stonking/InRelease"] = archive.objects[
        "dists/stonking/InRelease"
    ].replace(hashlib.sha512(index).hexdigest().encode(), b"z" * 128)
    with pytest.raises(ValueError, match="digest is malformed"):
        managed.selected_bubblewrap_bytes()


def test_package_extraction_requires_readable_regular_member(archive, monkeypatch):
    """A listed executable with no readable data is rejected."""
    monkeypatch.setattr(managed, "discover_tool", Mock(side_effect=OSError("missing")))
    member = SimpleNamespace(name="./usr/bin/bwrap", uid=0, gid=0, mode=0o755, isreg=lambda: True)

    class MissingData:
        """Model a tar member whose content cannot be read."""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter((member,))

        def extractfile(self, _member):
            """Report a missing member stream."""
            return

    monkeypatch.setattr(managed.tarfile, "open", Mock(return_value=MissingData()))
    with pytest.raises(ValueError, match="executable is unavailable"):
        managed.selected_bubblewrap_bytes()
