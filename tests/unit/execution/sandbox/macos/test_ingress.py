"""Verify the exact native connector contract without launching system programs."""

import json
from pathlib import Path

from loop.execution.sandbox.macos.ingress import connector_arguments


def test_connector_binds_exact_native_endpoint_and_foreground_identity():
    """The separately sandboxed connector has one Unix destination and authenticates before reads."""
    path = Path('/Volumes/owned/socket-"name')
    arguments = connector_arguments(path, 123)
    assert arguments[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert arguments[3:6] == ["/usr/bin/perl", "-e", arguments[5]]
    assert arguments[-2:] == [str(path), "123"]
    profile = arguments[2]
    assert "(deny default)" in profile
    assert profile.count("network-outbound") == 1
    assert f"(literal {json.dumps(str(path))})" in profile
    assert "AF_INET" not in profile and "network-bind" not in profile
    assert arguments[5].index("getsockopt") < arguments[5].index("sysread")
