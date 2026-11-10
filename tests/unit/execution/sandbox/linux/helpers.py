"""Share protected-record and mount inventory fixtures across Linux test suites."""

import json

from loop.execution.sandbox.linux import management as linux_management


def mountinfo_path(path):
    """Encode a synthetic mount path using the kernel's octal field escapes."""
    escapes = {" ": r"\040", "\t": r"\011", "\n": r"\012", "\\": r"\134"}
    return "".join(escapes.get(character, character) for character in str(path))


def publish(installation):
    """Publish a protected record through the owning management component."""
    _record, arguments = installation
    return linux_management.publish_generation_record(*arguments)


def rewrite(record, mutation):
    """Change a disposable protected record to exercise exact schema refusal."""
    value = json.loads(record.read_bytes())
    mutation(value)
    record.chmod(0o644)
    record.write_text(json.dumps(value))
    record.chmod(0o444)
