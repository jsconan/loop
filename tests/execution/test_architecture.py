"""Architecture gates for execution authority and process creation."""

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_SOURCE_ROOT = _ROOT / "src" / "loop"
_CLASSIFIED_SUBPROCESS_MODULES = {
    "execution/infrastructure/process.py",  # Managed-runtime process boundary.
    "execution/sandbox/oci/process.py",  # Attached OCI process/PTY owner.
}
_IMPLEMENTATION_HISTORY = re.compile(
    r"\b(?:Stage(?:-| )[A-Z](?:\d+(?:\.\d+)?)?|M0(?:\.\d+)?|WP\d+(?:\.\d+)?|"
    r"work[ -]package|milestone|task[ -]order)\b",
    re.IGNORECASE,
)


def _imports_subprocess(path: Path) -> bool:
    """Return whether one module imports Python's subprocess module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Import)
        and any(alias.name == "subprocess" for alias in node.names)
        or isinstance(node, ast.ImportFrom)
        and node.module == "subprocess"
        for node in ast.walk(tree)
    )


def test_sandbox_modules_cannot_import_host_execution_modules():
    """The ordinary execution package cannot acquire host-process authority by import."""
    execution_root = _SOURCE_ROOT / "execution"
    forbidden_modules = {"loop.execution.host"}
    for path in execution_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not imports & forbidden_modules
        if path.relative_to(execution_root).as_posix() not in {
            "infrastructure/process.py",
            "sandbox/oci/process.py",
        }:
            assert "subprocess" not in imports


def test_sandbox_modules_cannot_name_host_authority_contracts():
    """Sandbox adapters cannot acquire or construct explicit host authority types."""
    sandbox_root = _SOURCE_ROOT / "execution" / "sandbox"
    forbidden_names = {"HostExecutionLease", "HostExecutionRequest", "HostExecutionResult"}
    for path in sandbox_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not (names | imported_names) & forbidden_names
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (
                node.module == "host"
                or node.module.startswith("host.")
                or node.module.endswith(".host")
                or ".host." in node.module
            )
            for node in ast.walk(tree)
        )


def test_common_execution_modules_do_not_import_platform_adapters():
    """Platform-neutral execution code cannot depend on a concrete host adapter."""
    execution_root = _SOURCE_ROOT / "execution"
    for path in execution_root.rglob("*.py"):
        relative = path.relative_to(execution_root)
        if relative.parts[:2] in {("sandbox", "macos"), ("sandbox", "linux")}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any("sandbox.macos" in module or "sandbox.linux" in module for module in modules)
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.level
            and (
                node.module in {"macos", "linux"}
                or node.module is not None
                and node.module.endswith("sandbox")
                and any(alias.name in {"macos", "linux"} for alias in node.names)
            )
            for node in ast.walk(tree)
        )


def test_production_source_contains_no_implementation_history_identifiers():
    """Production behavior and diagnostics use domain terminology only."""
    offenders = {
        str(path.relative_to(_SOURCE_ROOT)): sorted(set(_IMPLEMENTATION_HISTORY.findall(text)))
        for path in _SOURCE_ROOT.rglob("*.py")
        if (text := path.read_text(encoding="utf-8")) and _IMPLEMENTATION_HISTORY.search(text)
    }
    assert offenders == {}


def test_every_current_subprocess_import_has_a_cutover_classification():
    """No unclassified host-process creation route enters the repository unnoticed."""
    importing_modules = {
        str(path.relative_to(_SOURCE_ROOT))
        for path in _SOURCE_ROOT.rglob("*.py")
        if _imports_subprocess(path)
    }

    assert importing_modules == _CLASSIFIED_SUBPROCESS_MODULES


def test_only_host_broker_can_reach_the_host_supervisor() -> None:
    """No production caller can bypass identity, permission, and lease creation."""
    offenders = []
    for path in _SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(_SOURCE_ROOT).as_posix()
        if relative in {"execution/host/broker.py", "execution/host/supervisor.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports_supervisor = any(
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (
                node.module.endswith("host.supervisor")
                or node.module == "supervisor"
                and "execution/host" in relative
            )
            for node in ast.walk(tree)
        )
        names_supervisor = any(
            isinstance(node, ast.Name)
            and node.id == "HostExecutionSupervisor"
            or isinstance(node, ast.Attribute)
            and node.attr == "HostExecutionSupervisor"
            or isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "HostExecutionSupervisor" for alias in node.names)
            for node in ast.walk(tree)
        )
        if imports_supervisor or names_supervisor:
            offenders.append(relative)

    assert offenders == []
