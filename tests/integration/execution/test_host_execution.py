"""Exercise the complete explicit host workflow with a real child process."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from loop.execution.contracts import HostExecutionRequest
from loop.execution.host import HostCompleted, HostExecutionBroker
from loop.execution.host.models import HostAuditEvent, HostAuthorizationDecision


@dataclass
class _Audit:
    """Collect real workflow audit events."""

    events: list[HostAuditEvent] = field(default_factory=list)

    def record(self, event: HostAuditEvent) -> None:
        """Append one event."""
        self.events.append(event)


class _Approved:
    """Represent an already tested exact permission approval boundary."""

    def authorize(self, request, executable, prompt) -> HostAuthorizationDecision:
        """Approve the exact verified request for process-mechanics integration."""
        del request, executable, prompt
        return HostAuthorizationDecision(allowed=True)


def test_real_host_start_requires_request_identity_permission_and_lease(tmp_path: Path) -> None:
    """A real host child starts only through the audited broker and lease-bound supervisor."""
    executable = Path("/usr/bin/printf")
    if not executable.exists():
        pytest.skip("The integration host does not provide /usr/bin/printf.")
    audit = _Audit()
    broker = HostExecutionBroker(
        _Approved(),
        "workspace",
        "2",
        audit=audit,
        clock_ns=lambda: 10,
        id_factory=lambda: "lease",
    )
    request = HostExecutionRequest(
        request_id="request",
        executable=str(executable),
        argv=(str(executable), "host-ok"),
        cwd=str(tmp_path),
        display_cwd="workspace",
        resource_class="native tool",
        reason="Integration coverage for explicit host execution.",
    )

    result = broker.execute(request)

    assert isinstance(result, HostCompleted)
    assert result.stdout == b"host-ok"
    assert [event.type.value for event in audit.events] == [
        "execution.host.requested",
        "execution.host.authorized",
        "execution.host.started",
        "execution.host.terminal",
    ]


def test_real_host_git_reads_staged_changes_from_virtual_workspace(tmp_path: Path) -> None:
    """The explicit host Git path resolves `/workspace` to the authenticated project root."""
    executable = Path("/usr/bin/git")
    if not executable.exists():
        pytest.skip("The integration host does not provide /usr/bin/git.")
    subprocess.run((executable, "init", "-q", tmp_path), check=True)
    staged = tmp_path / "staged.txt"
    staged.write_text("staged", encoding="utf-8")
    subprocess.run((executable, "-C", tmp_path, "add", staged.name), check=True)
    audit = _Audit()
    broker = HostExecutionBroker(
        _Approved(),
        "workspace",
        "2",
        workspace_root=tmp_path,
        audit=audit,
    )
    request = HostExecutionRequest(
        request_id="git-staged",
        executable=str(executable),
        argv=(str(executable), "diff", "--cached", "--name-status"),
        cwd="/workspace",
        display_cwd="project workspace",
        resource_class="git repository (read-only)",
        reason="Read the authenticated repository index.",
    )

    result = broker.execute(request)

    assert isinstance(result, HostCompleted), audit.events
    assert result.exit_code == 0
    assert result.stdout == b"A\tstaged.txt\n"
