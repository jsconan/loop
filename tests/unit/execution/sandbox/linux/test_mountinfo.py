"""Test shared mount inventory readers."""

import sys
from pathlib import Path

import pytest

from loop.execution.sandbox.linux import mountinfo


def test_read_mountinfo_splits_rows_without_reordering(monkeypatch):
    """Preserve kernel row order and fields while reading the current process table."""
    contents = "2 1 0:2 / /second rw - tmpfs tmpfs rw\n1 0 0:1 / /first ro - cgroup2 cgroup rw\n"

    def read_bytes(path):
        """Return only the owned test mount inventory."""
        assert path == mountinfo.MOUNTINFO_PATH
        return contents.encode()

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    assert mountinfo.read_mountinfo() == [
        ["2", "1", "0:2", "/", "/second", "rw", "-", "tmpfs", "tmpfs", "rw"],
        ["1", "0", "0:1", "/", "/first", "ro", "-", "cgroup2", "cgroup", "rw"],
    ]


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [(r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")],
)
def test_read_mountinfo_decodes_kernel_path_escapes(monkeypatch, encoded, decoded):
    """Root and mountpoint fields retain the exact names represented by the kernel."""
    content = f"2 1 0:2 /root{encoded}name /mount{encoded}name rw - tmpfs tmpfs rw\n"
    monkeypatch.setattr(Path, "read_bytes", lambda *args, **kwargs: content.encode())

    rows = mountinfo.read_mountinfo()

    assert rows[0][3:5] == [f"/root{decoded}name", f"/mount{decoded}name"]


def test_read_mountinfo_does_not_decode_literal_escape_twice(monkeypatch):
    """An escaped backslash followed by digits remains part of the literal filename."""
    content = r"2 1 0:2 / /mount\134040name rw - tmpfs tmpfs rw" + "\n"
    monkeypatch.setattr(Path, "read_bytes", lambda *args, **kwargs: content.encode())

    assert mountinfo.read_mountinfo()[0][4] == r"/mount\040name"


def test_read_mountinfo_preserves_unescaped_nonseparator_control_characters(monkeypatch):
    """Carriage returns in legal pathnames do not split a mount table record."""
    content = "2 1 0:2 / /mount\rname rw - tmpfs tmpfs rw\n"
    monkeypatch.setattr(Path, "read_bytes", lambda *args, **kwargs: content.encode())

    rows = mountinfo.read_mountinfo()

    assert len(rows) == 1
    assert rows[0][4] == "/mount\rname"


def test_read_mountinfo_preserves_undecodable_filesystem_bytes(monkeypatch):
    """Mount path bytes outside UTF-8 survive the filesystem encoding's lossless decoding."""
    content = b"2 1 0:2 / /mount\xffname rw - tmpfs tmpfs rw\n"
    monkeypatch.setattr(Path, "read_bytes", lambda *args, **kwargs: content)

    value = mountinfo.read_mountinfo()[0][4]

    assert value.encode(sys.getfilesystemencoding(), "surrogateescape") == b"/mount\xffname"


@pytest.mark.parametrize("encoded", [r"\000", r"\777", r"\04", "\\"])
def test_read_mountinfo_rejects_malformed_path_escapes(monkeypatch, encoded):
    """Ambiguous kernel path fields cannot authorize an incomplete mount inventory."""
    content = f"2 1 0:2 / /mount{encoded}name rw - tmpfs tmpfs rw\n"
    monkeypatch.setattr(Path, "read_bytes", lambda *args, **kwargs: content.encode())

    with pytest.raises(ValueError, match="Invalid mountinfo path escape"):
        mountinfo.read_mountinfo()


def test_read_descriptor_mount_ids_selects_only_mount_id_fields(monkeypatch):
    """Return every descriptor mount ID and ignore unrelated fdinfo fields."""
    contents = "pos:\t0\nflags:\t0100000\nmnt_id:\t71\nmnt_id: 72\n"

    def read_text(path, *, encoding, errors=None):
        """Return only the owned test descriptor inventory."""
        assert path == mountinfo.FDINFO_DIRECTORY / "321"
        assert encoding == "ascii"
        assert errors is None
        return contents

    monkeypatch.setattr(Path, "read_text", read_text)

    assert mountinfo.read_descriptor_mount_ids(321) == [["71"], ["72"]]
