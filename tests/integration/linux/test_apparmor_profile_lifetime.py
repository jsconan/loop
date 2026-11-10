"""Verify native AppArmor profile lifetime behavior on an explicitly disposable Linux host."""

import os
import select
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

from loop.execution.sandbox.linux.host import compile_policy_source

pytestmark = [
    pytest.mark.integration,
    pytest.mark.linux,
    pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux AppArmor"),
    pytest.mark.skipif(
        os.environ.get("LOOP_NATIVE_MANAGEMENT_TEST") != "disposable-linux-host",
        reason="Requires an explicitly designated disposable Linux host",
    ),
]


def _qualified_labels(namespace: str, profile: str, *, enforcing: bool) -> set[str]:
    """Accept the exact plain and stacked-unconfined kernel spellings."""
    suffix = f":{namespace}:{profile}" + (" (enforce)" if enforcing else "")
    return {suffix, "unconfined//&" + suffix}


def test_compiled_policy_matches_kernel_export():
    """The host compiler's bounded bytes match the enforcing kernel's exact policy export."""
    source = b"abi <abi/4.0>,\nprofile loop_export_probe { allow r /**, }\n"
    compiled = compile_policy_source(source, time.monotonic() + 5)
    code = """
import os,sys,uuid
from pathlib import Path
namespace=Path('/sys/kernel/security/apparmor/policy/namespaces')/('loop_export_'+uuid.uuid4().hex)
namespace.mkdir()
try:
    data=sys.stdin.buffer.read()
    descriptor=os.open(namespace/'.load',os.O_WRONLY)
    try: written=os.write(descriptor,data)
    finally: os.close(descriptor)
    entries=list((namespace/'profiles').iterdir())
    if written!=len(data) or len(entries)!=1:
        raise ValueError('Kernel policy load count or inventory differs')
    entry=entries[0]
    if (entry/'mode').read_bytes().strip()!=b'enforce' or (entry/'raw_data').read_bytes()!=data:
        raise ValueError('Kernel policy export or enforcement differs')
finally:
    namespace.rmdir()
"""
    subprocess.run(
        ["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "-c", code],
        input=compiled,
        capture_output=True,
        check=True,
        timeout=10,
    )


def test_foreign_process_can_enter_and_outlive_removed_profile():
    """Profile deletion unconfines an outside process that entered the same policy namespace."""
    namespace = "loop_native_foreign_" + uuid4().hex
    directory = Path("/sys/kernel/security/apparmor/policy/namespaces") / namespace
    profile = "payload_foreign_test"
    source = f"abi <abi/4.0>,\nprofile {profile} {{\n  allow r /**,\n  allow ix /**,\n}}\n"
    subprocess.run(["/usr/bin/sudo", "-n", "mkdir", str(directory)], check=True, timeout=10)
    process = None
    try:
        subprocess.run(
            ["/usr/bin/sudo", "-n", "/usr/sbin/apparmor_parser", "-a", "-K", "-n", namespace],
            input=source,
            text=True,
            check=True,
            timeout=10,
        )
        process = subprocess.Popen(
            ["/usr/bin/aa-exec", "-p", f":{namespace}:{profile}", "--", "/usr/bin/sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        current = Path(f"/proc/{process.pid}/attr/current")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Foreign profile test process exited before attachment.")
            if current.read_text(encoding="ascii").strip() in _qualified_labels(
                namespace, profile, enforcing=True
            ):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("Foreign profile test process did not enter the policy.")
        subprocess.run(
            ["/usr/bin/sudo", "-n", "/usr/bin/tee", str(directory / ".remove")],
            input=profile,
            text=True,
            stdout=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        assert current.read_text(encoding="ascii").strip() in _qualified_labels(
            namespace, "unconfined", enforcing=False
        )
        assert process.poll() is None
    finally:
        try:
            if process is not None:
                process.terminate()
                process.communicate(timeout=10)
        finally:
            subprocess.run(["/usr/bin/sudo", "-n", "rmdir", str(directory)], check=True, timeout=10)


def test_removal_unconfines_process_entering_from_another_enforcing_profile():
    """An outside confined process can lose its protection when a target profile is removed."""
    suffix = uuid4().hex
    namespace = "loop_native_foreign_" + suffix
    parent = "loop_native_parent_" + suffix
    directory = Path("/sys/kernel/security/apparmor/policy/namespaces") / namespace
    target = "payload_foreign_test"
    child_source = f"abi <abi/4.0>,\nprofile {target} {{\n  allow r /**,\n  allow ix /**,\n}}\n"
    parent_source = (
        f"abi <abi/4.0>,\nprofile {parent} {{\n"
        "  allow rwmix /**,\n"
        "  change_profile -> :loop_native_foreign_*:payload_*,\n"
        "}\n"
    )
    subprocess.run(["/usr/bin/sudo", "-n", "mkdir", str(directory)], check=True, timeout=10)
    process = None
    parent_loaded = False
    try:
        subprocess.run(
            ["/usr/bin/sudo", "-n", "/usr/sbin/apparmor_parser", "-a", "-K", "-n", namespace],
            input=child_source,
            text=True,
            check=True,
            timeout=10,
        )
        subprocess.run(
            ["/usr/bin/sudo", "-n", "/usr/sbin/apparmor_parser", "-a", "-K"],
            input=parent_source,
            text=True,
            check=True,
            timeout=10,
        )
        parent_loaded = True
        refused = subprocess.run(
            [
                "/usr/bin/aa-exec",
                "-p",
                parent,
                "--",
                "/usr/bin/aa-exec",
                "-p",
                "unconfined",
                "--",
                "/usr/bin/true",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        assert refused.returncode != 0
        assert "insufficient permissions to change to the profile 'unconfined'" in refused.stderr
        process = subprocess.Popen(
            [
                "/usr/bin/aa-exec",
                "-p",
                parent,
                "--",
                "/usr/bin/aa-exec",
                "-p",
                f":{namespace}:{target}",
                "--",
                "/usr/bin/sleep",
                "30",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        current = Path(f"/proc/{process.pid}/attr/current")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Confined profile test process exited before attachment.")
            if current.read_text(encoding="ascii").strip() in _qualified_labels(
                namespace, target, enforcing=True
            ):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("Confined profile test process did not enter the target policy.")
        subprocess.run(
            ["/usr/bin/sudo", "-n", "/usr/bin/tee", str(directory / ".remove")],
            input=target,
            text=True,
            stdout=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        assert current.read_text(encoding="ascii").strip() in _qualified_labels(
            namespace, "unconfined", enforcing=False
        )
        assert process.poll() is None
    finally:
        try:
            if process is not None:
                process.terminate()
                process.communicate(timeout=10)
        finally:
            try:
                if parent_loaded:
                    subprocess.run(
                        ["/usr/bin/sudo", "-n", "/usr/sbin/apparmor_parser", "-R", "-K"],
                        input=parent_source,
                        text=True,
                        check=True,
                        timeout=10,
                    )
            finally:
                subprocess.run(
                    ["/usr/bin/sudo", "-n", "rmdir", str(directory)], check=True, timeout=10
                )


def test_replacing_profile_changes_the_rules_for_existing_processes(tmp_path):
    """Reusing a profile name cannot safely widen the rules of an attached process."""
    namespace = "loop_native_replace_" + uuid4().hex
    directory = Path("/sys/kernel/security/apparmor/policy/namespaces") / namespace
    profile = "payload_replacement_test"
    protected = tmp_path / "protected"
    protected.write_text("private", encoding="ascii")
    restricted = (
        f'abi <abi/4.0>,\nprofile {profile} {{\n  allow rix /**,\n  deny r "{protected}",\n}}\n'
    )
    widened = f"abi <abi/4.0>,\nprofile {profile} {{\n  allow rix /**,\n}}\n"
    program = (
        "import sys\n"
        "for line in sys.stdin:\n"
        "    try:\n"
        "        open(sys.argv[1], encoding='ascii').read()\n"
        "    except PermissionError:\n"
        "        print('denied', flush=True)\n"
        "    else:\n"
        "        print('allowed', flush=True)\n"
    )
    subprocess.run(["/usr/bin/sudo", "-n", "mkdir", str(directory)], check=True, timeout=10)
    process = None
    try:
        subprocess.run(
            ["/usr/bin/sudo", "-n", "/usr/sbin/apparmor_parser", "-a", "-K", "-n", namespace],
            input=restricted,
            text=True,
            check=True,
            timeout=10,
        )
        process = subprocess.Popen(
            [
                "/usr/bin/aa-exec",
                "-p",
                f":{namespace}:{profile}",
                "--",
                "/usr/bin/python3",
                "-u",
                "-c",
                program,
                str(protected),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        process.stdin.write("check\n")
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 10)[0]
        assert process.stdout.readline().strip() == "denied"
        subprocess.run(
            ["/usr/bin/sudo", "-n", "/usr/sbin/apparmor_parser", "-r", "-K", "-n", namespace],
            input=widened,
            text=True,
            check=True,
            timeout=10,
        )
        process.stdin.write("check\n")
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 10)[0]
        assert process.stdout.readline().strip() == "allowed"
    finally:
        try:
            if process is not None:
                process.terminate()
                process.communicate(timeout=10)
        finally:
            subprocess.run(["/usr/bin/sudo", "-n", "rmdir", str(directory)], check=True, timeout=10)
