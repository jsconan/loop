"""Share record publication helpers across management-owned Linux test suites."""

import json

from loop.execution.sandbox.linux import management as linux_management


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
