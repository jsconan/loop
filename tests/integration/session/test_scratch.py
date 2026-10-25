"""Verify durable session working files across independent application processes."""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_saved_working_files_survive_process_exit_and_resume(tmp_path):
    """A resumed session reopens its working files after the creating process has exited."""
    database = tmp_path / "storage" / "sessions.db"
    source = """
import json
import sys
from unittest.mock import Mock
from loop import Interaction, SessionManager, SQLiteSessionStore

manager = SessionManager(
    interaction=Mock(spec=Interaction),
    session=sys.argv[2] or None,
    session_store=SQLiteSessionStore(sys.argv[1], workspace_id="workspace"),
    workspace_id="workspace",
)
directory = manager.temporary_directory.get()
if not sys.argv[2]:
    (directory / "report.txt").write_text("retained", encoding="utf-8")
    manager.add_user_message("prepare a report")
print(json.dumps({
    "session_id": manager.session.id,
    "directory": str(directory),
    "content": (directory / "report.txt").read_text(encoding="utf-8"),
}))
"""

    def run(identifier):
        """Launch an isolated application process and collect its public storage result."""
        result = subprocess.run(
            [sys.executable, "-B", "-c", source, str(database), identifier],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        assert result.stderr == ""
        return json.loads(result.stdout)

    created = run("")
    resumed = run(created["session_id"])

    assert created["content"] == "retained"
    assert resumed == created
