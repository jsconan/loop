"""Exercise direct-tool confidentiality and changed-name defenses on real filesystems."""

import json
import subprocess
import sys

import pytest

from loop import (
    BUILTIN_TOOLS,
    PermissionConfiguration,
    PermissionManager,
    PolicyLimits,
    ToolRegistry,
)

pytestmark = pytest.mark.integration


def test_direct_tools_admit_benign_links_and_refuse_protected_or_incomplete_sets(tmp_path):
    """Native inode accounting admits ordinary links without exposing outside or hard data."""
    source = tmp_path / "source"
    source.write_text("ordinary")
    alias = tmp_path / "alias"
    alias.hardlink_to(source)
    registry = ToolRegistry(
        BUILTIN_TOOLS, permission_manager=PermissionManager(tmp_path, yolo=True)
    )
    try:
        result = json.loads(registry.call("read_text_file", json.dumps({"path": str(alias)})))
        assert result["result"]["content"] == "ordinary"
        for tool, arguments in (
            ("edit_text_file", {"old_content": "ordinary", "new_content": "edited"}),
            ("write_text_file", {"content": "replacement"}),
            ("delete_path", {}),
        ):
            if not alias.exists() or alias.stat().st_nlink == 1:
                alias.unlink(missing_ok=True)
                alias.hardlink_to(source)
            result = json.loads(registry.call(tool, json.dumps({"path": str(alias), **arguments})))
            assert result["ok"] is True, result
            assert source.read_text() == "ordinary"
        for original in (tmp_path / ".env", tmp_path.parent / "outside"):
            original.write_text("hard-canary")
            alias.hardlink_to(original)
            result = registry.call("read_text_file", json.dumps({"path": str(alias)}))
            assert json.loads(result)["problem"]["code"] == "tool.denied"
            assert "hard-canary" not in result
            alias.unlink()
    finally:
        registry.close()


@pytest.mark.parametrize("name", [".env", ".env.local", ".env.production", ".env.example", "cache"])
def test_direct_tools_keep_hard_resources_out_of_results_and_mutations(tmp_path, name):
    """Built-in and exact designated data stay denied under automatic approval."""
    source = tmp_path / name
    source.write_text("hard-canary", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("cache\n", encoding="utf-8")
    designated = (str(source),) if name in {".env.example", "cache"} else ()
    manager = PermissionManager(
        tmp_path,
        yolo=True,
        configuration=PermissionConfiguration(limits=PolicyLimits(hard_denied_paths=designated)),
    )
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)
    try:
        for tool, arguments in (
            ("read_text_file", {}),
            ("write_text_file", {"content": "changed"}),
            ("edit_text_file", {"old_content": "hard", "new_content": "changed"}),
            ("delete_path", {}),
            ("search_text", {"query": "hard-canary"}),
        ):
            output = registry.call(tool, json.dumps({"path": str(source), **arguments}))
            assert json.loads(output)["problem"]["code"] == "tool.denied"
            assert "hard-canary" not in output
        for tool, arguments in (
            ("list_folder", {"recursive": True}),
            ("search_text", {"query": "hard-canary"}),
        ):
            output = registry.call(tool, json.dumps({"path": str(tmp_path), **arguments}))
            assert json.loads(output)["ok"] is True
            assert "hard-canary" not in output
            assert (
                name not in {entry["path"] for entry in json.loads(output)["result"]}
                if tool == "list_folder"
                else True
            )
        assert source.read_text(encoding="utf-8") == "hard-canary"
        public = tmp_path / ".env.production.sample"
        public.write_text("public control", encoding="utf-8")
        output = registry.call("read_text_file", json.dumps({"path": str(public)}))
        assert json.loads(output)["result"]["content"] == "public control"
    finally:
        registry.close()


@pytest.mark.parametrize("kind", ["symlink", "replacement", "hardlink"])
def test_direct_read_refuses_a_concurrent_changed_selection(tmp_path, kind):
    """A separately running writer cannot swap forbidden bytes into an authorized read."""
    source = tmp_path / "source"
    source.write_text("public", encoding="utf-8")
    secret = tmp_path / ".env"
    secret.write_text("forbidden-canary", encoding="utf-8")
    writer_source = """
import os, sys
from pathlib import Path
source, secret, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
print('ready', flush=True)
assert sys.stdin.readline() == 'swap\\n'
source.unlink()
if kind == 'symlink':
    source.symlink_to(secret)
elif kind == 'hardlink':
    os.link(secret, source)
else:
    source.write_text(secret.read_text())
print('swapped', flush=True)
"""
    writer = subprocess.Popen(
        [sys.executable, "-c", writer_source, str(source), str(secret), kind],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=PermissionManager(tmp_path))
    try:
        assert writer.stdout.readline() == "ready\n"

        def swap_before_execution():
            """Synchronize the external writer after authorization and before descriptor opening."""
            writer.stdin.write("swap\n")
            writer.stdin.flush()
            assert writer.stdout.readline() == "swapped\n"

        output = registry.call(
            "read_text_file",
            json.dumps({"path": str(source)}),
            execution_started=swap_before_execution,
        )
        assert json.loads(output)["ok"] is False
        assert "forbidden-canary" not in output
        assert secret.read_text(encoding="utf-8") == "forbidden-canary"
    finally:
        writer.stdin.close()
        writer.wait(timeout=10)
        writer.stdout.close()
        writer.stderr.close()
        registry.close()


@pytest.mark.parametrize("tool", ["write_text_file", "edit_text_file", "delete_path"])
def test_direct_mutation_refuses_a_concurrent_parent_swap(tmp_path, tool):
    """A real external parent replacement cannot redirect an approved mutation."""
    parent = tmp_path / "selected"
    parent.mkdir()
    source = parent / "file"
    source.write_text("original", encoding="utf-8")
    outside = tmp_path.parent / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside-canary", encoding="utf-8")
    registry = ToolRegistry(
        BUILTIN_TOOLS, permission_manager=PermissionManager(tmp_path, yolo=True)
    )

    def swap_parent():
        """Replace the selected parent in a separate process after authorization."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from pathlib import Path; p=Path(sys.argv[1]); "
                    'p.rename(p.with_name("retired")); '
                    "p.symlink_to(sys.argv[2], target_is_directory=True)"
                ),
                str(parent),
                str(outside),
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )

    arguments = {"path": str(source)}
    if tool == "write_text_file":
        arguments["content"] = "changed"
    elif tool == "edit_text_file":
        arguments.update(old_content="original", new_content="changed")
    try:
        output = registry.call(tool, json.dumps(arguments), execution_started=swap_parent)
        assert json.loads(output)["ok"] is False
        assert (outside / "file").read_text(encoding="utf-8") == "outside-canary"
        assert (tmp_path / "retired/file").read_text(encoding="utf-8") == "original"
    finally:
        registry.close()
