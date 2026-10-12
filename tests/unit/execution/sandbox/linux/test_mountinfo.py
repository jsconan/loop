"""Test shared mount inventory readers."""

from pathlib import Path

from loop.execution.sandbox.linux import mountinfo


def test_read_mountinfo_splits_rows_without_reordering(monkeypatch):
    """Preserve kernel row order and fields while reading the current process table."""
    contents = "2 1 0:2 / /second rw - tmpfs tmpfs rw\n1 0 0:1 / /first ro - cgroup2 cgroup rw\n"

    def read_text(path, *, encoding, errors=None):
        """Return only the owned test mount inventory."""
        assert path == mountinfo.MOUNTINFO_PATH
        assert encoding is not None
        assert errors == "surrogateescape"
        return contents

    monkeypatch.setattr(Path, "read_text", read_text)

    assert mountinfo.read_mountinfo() == [
        ["2", "1", "0:2", "/", "/second", "rw", "-", "tmpfs", "tmpfs", "rw"],
        ["1", "0", "0:1", "/", "/first", "ro", "-", "cgroup2", "cgroup", "rw"],
    ]


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
