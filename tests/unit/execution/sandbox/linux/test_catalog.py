"""Verify packaged isolated runtime assembly, closed policy and exact helper byte authority."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop import constants as app_constants
from loop.execution.sandbox.linux import catalog as linux_catalog
from loop.execution.sandbox.linux import constants as linux_constants


@pytest.fixture
def helper(monkeypatch):
    """Provide selected helper bytes through their public source boundary."""
    selected = Mock(return_value=b"authenticated helper")
    monkeypatch.setattr(linux_catalog, "selected_bubblewrap_bytes", selected)
    return selected


def test_catalog_uses_distribution_resources_and_authenticated_helper(helper):
    """Retained code uses passive barrels, shipped resources and a private exact helper copy."""
    catalog, executable, sources, attachment = linux_catalog.runtime_catalog(
        Path("/usr/local/libexec/loop/native"), "native_test"
    )
    assert set(linux_catalog.runtime_member_names()) == set(catalog)
    helper.assert_called_once_with()
    assert catalog["helper"] == b"authenticated helper"
    assert executable == {"loop_receiver.py", "loop_uninstall.py", "helper"}
    for name in (
        "loop/__init__.py",
        "loop/utils/__init__.py",
        "loop/permissions/__init__.py",
        "loop/execution/__init__.py",
        "loop/execution/sandbox/__init__.py",
        "loop/execution/sandbox/linux/__init__.py",
    ):
        assert catalog[name] == b""
    assert b"receive_installed_request" in catalog["loop_receiver.py"]
    assert b"uninstall_generation" in catalog["loop_uninstall.py"]
    assert b"def supervise_process" in catalog["loop/utils/process.py"]
    assert b"def receive_request" in catalog["loop/execution/sandbox/linux/admission.py"]
    assert b"NATIVE_REQUEST_MAGIC" in catalog["loop/execution/sandbox/linux/constants.py"]
    assert b"def read_mountinfo" in catalog["loop/execution/sandbox/linux/mountinfo.py"]
    assert b"def exchange_request" in catalog["loop/execution/sandbox/linux/transport.py"]
    assert b"def from_wire" in catalog["loop/execution/sandbox/contracts.py"]
    assert b"protected_workspace_paths" in catalog["loop/permissions/protection.py"]
    assert b"execute_installed_request(\n" in catalog["loop_receiver.py"]
    assert b'entry.parent / "management.json", payload, connection' in catalog["loop_receiver.py"]
    assert b"def compile_payload" in catalog["loop/execution/sandbox/linux/compiler.py"]
    assert b"def payload_policy" in catalog["loop/execution/sandbox/linux/policy.py"]
    assert b"def mount_arguments" in catalog["loop/execution/sandbox/linux/paths.py"]
    assert b"def read_generation" in catalog["loop/execution/sandbox/linux/records.py"]
    assert b"def execute_installed_request" in catalog["loop/execution/sandbox/linux/runtime.py"]
    assert catalog["mask.empty"] == b""
    assert b"def namespace_arguments" in catalog["loop/execution/sandbox/linux/namespace.py"]
    assert set(sources) == {""}
    assert set(sources[""]) == {"bootstrap"}
    assert b"allow capability mac_admin" in sources[""]["bootstrap"]
    assert b"change_profile -> &payload_*" in sources[""]["bootstrap"]
    assert b'profile loopgen_native_test "/usr/local/libexec/loop/native/helper"' in attachment
    assert b"change_profile -> &:native_test:bootstrap" in attachment
    assert catalog["cache.lock"] == b"native_test\n"
    assert b"execute_payload" in catalog["loop_payload.py"]
    assert b"def supervise_payload" in catalog["loop/execution/sandbox/linux/supervision.py"]


def test_catalog_uses_preselected_helper_once(helper):
    """Reviewed installation reuses one pre-consent helper selection without another fetch."""
    catalog, executable, _, _ = linux_catalog.runtime_catalog(
        Path("/native"), "test", helper_bytes=b"selected"
    )
    assert catalog["helper"] == b"selected"
    assert "helper" in executable
    helper.assert_not_called()


def test_catalog_import_namespace_does_not_follow_app_name(helper, monkeypatch):
    """Renaming the application does not retarget reviewed Python code or entry filenames."""
    monkeypatch.setattr(app_constants, "APP_NAME", "nova")
    spec = importlib.util.spec_from_file_location(
        "loop.execution.sandbox.linux._renamed_constants", linux_constants.__file__
    )
    renamed_constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renamed_constants)

    catalog, _, _, _ = linux_catalog.runtime_catalog(
        Path("/native"), "test", helper_bytes=b"selected"
    )

    assert "loop/constants.py" in catalog
    assert "nova/constants.py" not in catalog
    for suffix, constant in (
        ("boot", "BOOT_FILENAME"),
        ("payload", "PAYLOAD_FILENAME"),
        ("receiver", "RECEIVER_FILENAME"),
        ("uninstall", "UNINSTALLER_FILENAME"),
    ):
        filename = f"{linux_constants.PACKAGE_NAMESPACE}_{suffix}.py"
        assert filename in catalog
        assert f"nova_{suffix}.py" not in catalog
        assert getattr(renamed_constants, constant) == filename
    assert renamed_constants.BOOT_SERVICE_NAME == "nova-native-boot.service"
    assert renamed_constants.NATIVE_SCRATCH_PATH == Path("/nova-private-scratch")


def test_recorded_uninstall_catalog_needs_no_helper_or_network(helper):
    """Authenticated recorded removal assembles its imports without a download."""
    catalog, executable, _, _ = linux_catalog.runtime_catalog(
        Path("/native"), "test", management_only=True
    )
    assert "helper" not in catalog
    assert "helper" not in executable
    helper.assert_not_called()


@pytest.mark.parametrize(
    "directory,generation",
    [
        ("relative", "test"),
        ("/root/../other", "test"),
        ("/root/sp ace", "test"),
        ("/", "test"),
        ("/native", ""),
        ("/native", "../escape"),
        ("/native", "a" * 121),
    ],
)
def test_catalog_invalid_literal_identity_refuses_before_reads(helper, directory, generation):
    """Invalid source identifiers and roots cannot select package or helper authority."""
    with pytest.raises(ValueError, match="Invalid fixed runtime catalog"):
        linux_catalog.runtime_catalog(Path(directory), generation)
    helper.assert_not_called()
