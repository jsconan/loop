"""Check repository-wide sync and async function naming contracts."""

import ast
from pathlib import Path


def test_source_definitions_follow_sync_async_naming_convention():
    """Async declarations end in ``_async`` and sync declarations use no boundary suffix."""
    source_root = Path(__file__).parents[2] / "src" / "loop"
    violations = []

    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and not node.name.endswith("_async"):
                violations.append(f"{path}:{node.lineno}: async {node.name}")
            elif isinstance(node, ast.FunctionDef) and node.name.endswith(("_async", "_sync")):
                violations.append(f"{path}:{node.lineno}: sync {node.name}")

    assert violations == []
