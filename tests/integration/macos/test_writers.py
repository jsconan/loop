"""Qualify detached native writers and private-directory ACL admission on macOS."""

import mmap
import os
import pwd
import select
import shlex
import sqlite3
import subprocess
import time
from contextlib import closing
from unittest.mock import Mock

import pytest

from loop import Interaction, SessionManager
from loop.execution.sandbox import SandboxOutcome, SandboxRequest
from loop.utils.path import ensure_private_directory

pytestmark = [pytest.mark.integration, pytest.mark.macos]


@pytest.fixture
def writer_executable(native_workspace):
    """Compile a self-bounded writer with native spawn, shared mappings and SQLite WAL."""
    source = native_workspace / "writer.c"
    executable = native_workspace / "writer"
    source.write_text(r"""
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static void require(int ok) {
    if (!ok) { perror("writer fixture"); _exit(2); }
}

static void sql(sqlite3 *db, const char *statement) {
    require(sqlite3_exec(db, statement, NULL, NULL, NULL) == SQLITE_OK);
}

static void child(const char *resource, const char *gate, int ready, const char *sibling) {
    alarm(10);
    const char *root = getenv("TMPDIR");
    char path[4096], escape[4096];
    snprintf(path, sizeof(path), "%s/state", root);
    snprintf(escape, sizeof(escape), "%s/../escape", root);
    int fd = -1;
    char *mapping = MAP_FAILED;
    sqlite3 *db = NULL;
    if (strcmp(resource, "mmap") == 0) {
        fd = open(path, O_CREAT | O_RDWR, 0600);
        require(fd >= 0 && ftruncate(fd, 4096) == 0);
        mapping = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        require(mapping != MAP_FAILED);
    } else {
        require(sqlite3_open(path, &db) == SQLITE_OK);
        sql(db, "PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;"
                "CREATE TABLE state(value); INSERT INTO state VALUES ('before');"
                "BEGIN IMMEDIATE; UPDATE state SET value='confined';");
    }
    char completion_path[4096];
    snprintf(completion_path, sizeof(completion_path), "%s/completed", root);
    int completed = open(completion_path, O_CREAT | O_WRONLY, 0600);
    require(completed >= 0);
    pid_t pid = getpid();
    require(write(ready, &pid, sizeof(pid)) == sizeof(pid));
    close(ready);
    close(STDIN_FILENO); close(STDOUT_FILENO); close(STDERR_FILENO);
    while (access(gate, F_OK) != 0) { usleep(10000); }
    int outside = open(escape, O_CREAT | O_WRONLY, 0600);
    int neighbor = open(sibling, O_RDONLY);
    int confined = outside < 0 && neighbor < 0;
    if (outside >= 0) close(outside);
    if (neighbor >= 0) close(neighbor);
    if (mapping != MAP_FAILED) {
        // A retired pathname cannot be recreated through the old leaf's authority.
        int recreated = mkdir(root, 0700);
        confined = confined && recreated < 0 && errno != EEXIST;
        strcpy(mapping, confined ? "confined" : "escaped");
        require(msync(mapping, 4096, MS_SYNC) == 0);
        require(munmap(mapping, 4096) == 0);
        close(fd);
    } else {
        if (!confined) sql(db, "UPDATE state SET value='escaped';");
        sql(db, "COMMIT;");
        require(sqlite3_close(db) == SQLITE_OK);
    }
    require(write(completed, "done", 4) == 4);
    close(completed);
    _exit(0);
}

int main(int argc, char **argv) {
    require(argc == 6);
    if (strcmp(argv[1], "child") == 0) {
        child(argv[2], argv[3], atoi(argv[4]), argv[5]);
    }
    alarm(10);
    int ready[2]; require(pipe(ready) == 0);
    pid_t pid;
    if (strcmp(argv[1], "double-fork") == 0) {
        pid = fork(); require(pid >= 0);
        if (pid == 0) {
            close(ready[0]); require(setsid() >= 0);
            pid_t grandchild = fork(); require(grandchild >= 0);
            if (grandchild > 0) _exit(0);
            child(argv[2], argv[3], ready[1], argv[5]);
        }
        require(waitpid(pid, NULL, 0) == pid);
    } else {
        char descriptor[32]; snprintf(descriptor, sizeof(descriptor), "%d", ready[1]);
        char *arguments[] = {argv[0], "child", argv[2], argv[3], descriptor, argv[5], NULL};
        posix_spawnattr_t attributes;
        require(posix_spawnattr_init(&attributes) == 0);
        short flags = strcmp(argv[1], "spawn-session") == 0
            ? POSIX_SPAWN_SETSID : POSIX_SPAWN_SETPGROUP;
        require(posix_spawnattr_setflags(&attributes, flags) == 0);
        if (flags == POSIX_SPAWN_SETPGROUP)
            require(posix_spawnattr_setpgroup(&attributes, 0) == 0);
        require(posix_spawn(&pid, argv[0], NULL, &attributes, arguments, environ) == 0);
        require(posix_spawnattr_destroy(&attributes) == 0);
    }
    close(ready[1]);
    pid_t writer; require(read(ready[0], &writer, sizeof(writer)) == sizeof(writer));
    close(ready[0]);
    printf("%d\n", writer);
    return 0;
}
""")
    compiled = subprocess.run(
        [
            "/usr/bin/clang",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-lsqlite3",
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "CLANG_MODULE_CACHE_PATH": str(native_workspace / "module-cache")},
    )
    assert compiled.returncode == 0, compiled.stderr
    assert compiled.stdout == compiled.stderr == ""
    return executable


@pytest.mark.parametrize("mechanism", ["double-fork", "spawn-session", "spawn-group"])
@pytest.mark.parametrize("resource", ["mmap", "wal"])
def test_detached_native_resources_remain_confined(
    backend, native_workspace, writer_executable, mechanism, resource
):
    """Retained mappings and WAL writers survive command exit without gaining new authority."""
    manager = SessionManager(interaction=Mock(spec=Interaction))
    original = manager.temporary_directory.get()
    sibling = SessionManager(interaction=Mock(spec=Interaction), session_store=manager.store)
    secret = sibling.temporary_directory.get() / "secret"
    secret.write_text("sibling-private-canary")
    alias = native_workspace / "session-alias"
    alias.symlink_to(original, target_is_directory=True)
    gate = native_workspace / "release"
    request = SandboxRequest.create(
        source=shlex.join(
            [str(writer_executable), mechanism, resource, str(gate), "unused", str(secret)]
        ),
        cwd=native_workspace,
        workspace=native_workspace,
        read_roots=(),
        write_roots=(),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(original)},
        policy_version=backend.policy_version,
        deadline=time.monotonic() + 5,
        workspace_id="detached-native-resources",
        aliases=(("$TMPDIR", alias, original),),
    )
    with manager.execution():
        result = backend.run(request)
    assert result.outcome is SandboxOutcome.COMPLETED, result
    assert result.exit_code == 0 and result.stderr == "", result
    pid = int(result.stdout.strip())
    with (
        closing(select.kqueue()) as queue,
        (original / "completed").open("rb") as completion,
    ):
        queue.control(
            [
                select.kevent(
                    pid,
                    filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT,
                )
            ],
            0,
            0,
        )
        exited = False
        try:
            if resource == "mmap":
                with (
                    (original / "state").open("r+b") as stream,
                    mmap.mmap(stream.fileno(), 4096) as retained,
                ):
                    manager.discard_working_directory()
                    replacement = manager.temporary_directory.get()
                    (replacement / "state").write_text("replacement-canary")
                    gate.write_text("release")
                    exited = bool(queue.control(None, 1, 5))
                    assert exited, "Detached writer did not exit."
                    assert retained[:9] == b"confined\0"
                    assert (replacement / "state").read_text() == "replacement-canary"
                assert not original.exists()
            else:
                assert (original / "state-wal").is_file()
                contender = SessionManager(
                    interaction=Mock(spec=Interaction),
                    session=manager.session,
                    session_store=manager.store,
                )
                with contender.execution():
                    with closing(sqlite3.connect(original / "state")) as reader:
                        assert reader.execute("SELECT value FROM state").fetchone() == ("before",)
                    gate.write_text("release")
                    exited = bool(queue.control(None, 1, 5))
                    assert exited, "Detached writer did not exit."
                    with closing(sqlite3.connect(original / "state")) as reader:
                        assert reader.execute("SELECT value FROM state").fetchone() == ("confined",)
            assert completion.read() == b"done"
            assert secret.read_text() == "sibling-private-canary"
            assert not (original.parent / "escape").exists()
        finally:
            gate.write_text("release")
            if not exited:
                assert queue.control(None, 1, 12), "Self-bounded writer did not terminate."


@pytest.fixture
def acl_directory(native_directory):
    """Own a traversable public control and an initially private directory with no ACL."""
    root = native_directory("/private/tmp")
    root.chmod(0o755)
    public = root / "public"
    public.write_text("public-control")
    public.chmod(0o644)
    private = root / "private"
    private.mkdir(mode=0o700)
    secret = private / "secret"
    secret.write_text("private-canary")
    secret.chmod(0o644)
    ensure_private_directory(private, create=False)
    return public, private, secret


def test_native_acl_cannot_be_hidden_by_private_mode_bits(acl_directory):
    """Reject a real extended ACL even when the private directory still has mode 0700."""
    _, private, _ = acl_directory
    added = subprocess.run(
        ["/bin/chmod", "+a", "user:nobody allow read,search", str(private)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert added.returncode == 0, added.stderr
    try:
        assert private.stat().st_mode & 0o777 == 0o700
        with pytest.raises(ValueError, match="private real directory"):
            ensure_private_directory(private, create=False)
    finally:
        subprocess.run(["/bin/chmod", "-N", str(private)], check=True, timeout=5)
    ensure_private_directory(private, create=False)


def test_distinct_account_access_matches_native_acl_admission(acl_directory):
    """Prove account isolation with a public control and an ACL that admission rejects."""
    account = pwd.getpwnam("nobody")
    if account.pw_uid == os.getuid():
        pytest.skip("Requires a distinct unprivileged nobody account.")
    identity = subprocess.run(
        ["/usr/bin/sudo", "-n", "-u", "nobody", "/usr/bin/id", "-u"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if identity.returncode != 0:
        if "a password is required" in identity.stderr:
            pytest.skip(
                "Distinct-account ACL qualification requires noninteractive sudo to nobody."
            )
        pytest.fail(f"Cannot establish distinct-account identity: {identity.stderr}")
    assert int(identity.stdout.strip()) == account.pw_uid
    public, private, secret = acl_directory

    def read_as_other(path):
        """Read only an owned synthetic fixture as the verified unprivileged account."""
        return subprocess.run(
            ["/usr/bin/sudo", "-n", "-u", "nobody", "/bin/cat", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    control = read_as_other(public)
    assert control.returncode == 0 and control.stdout == "public-control", control.stderr
    denied = read_as_other(secret)
    assert denied.returncode != 0 and denied.stdout == ""
    assert "Permission denied" in denied.stderr
    subprocess.run(
        ["/bin/chmod", "+a", "user:nobody allow read,search", str(private)],
        check=True,
        timeout=5,
    )
    try:
        allowed = read_as_other(secret)
        assert allowed.returncode == 0 and allowed.stdout == "private-canary", allowed.stderr
        with pytest.raises(ValueError, match="private real directory"):
            ensure_private_directory(private, create=False)
    finally:
        subprocess.run(["/bin/chmod", "-N", str(private)], check=True, timeout=5)
    denied = read_as_other(secret)
    assert denied.returncode != 0 and denied.stdout == ""
