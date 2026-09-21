"""Create durable, process-owned leases for immutable runtime versions."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ... import constants


@dataclass(frozen=True, slots=True)
class RuntimeLease:
    """Identify a live owner of one immutable artifact set.

    Args:
        path (Path): Durable lease record path.
        owner_id (str): Random process-owner identity.
        manifest_digest (str): Manifest identity protected by the lease.
        artifact_set_digest (str): Artifact-set identity protected by the lease.
        created (float): Original creation timestamp.
    """

    path: Path
    owner_id: str
    manifest_digest: str
    artifact_set_digest: str
    created: float

    def heartbeat(self, expiry: float) -> None:
        """Renew this durable lease through an atomic replace.

        Args:
            expiry: The new expiry timestamp for the lease.
        """
        payload = {
            "owner_id": self.owner_id,
            "manifest_digest": self.manifest_digest,
            "artifact_set_digest": self.artifact_set_digest,
            "created": self.created,
            "expiry": expiry,
        }
        if expiry <= time.time():
            raise ValueError("Runtime lease expiry must be in the future.")
        temporary = self.path.with_suffix(".new")
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)

    def close(self) -> None:
        """Release the owner lease without altering installed content."""
        self.path.unlink(missing_ok=True)


def create_lease(
    root: Path,
    manifest_digest: str,
    artifact_set_digest: str,
    lifetime: float,
) -> RuntimeLease:
    """Create an fsynced lease below the private runtime root.

    Args:
        root: The root directory under which the lease will be created.
        manifest_digest: The digest of the manifest associated with the lease.
        artifact_set_digest: The digest of the artifact set associated with the lease.
        lifetime: The desired lifetime of the lease in seconds.

    Returns:
        A RuntimeLease object representing the newly created lease.
    """
    owner_id = uuid.uuid4().hex
    if lifetime <= 0:
        raise ValueError("Runtime lease lifetime must be positive.")
    path = root / constants.RUNTIME_LEASES_DIRECTORY / f"{owner_id}.json"
    path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    created = time.time()
    lease = RuntimeLease(path, owner_id, manifest_digest, artifact_set_digest, created)
    lease.heartbeat(created + lifetime)
    return lease


def active_leases(root: Path, now: float | None = None) -> frozenset[str]:
    """Return artifact-set identities protected by unexpired valid leases.

    Args:
        root: The root directory under which leases are stored.
        now: The current timestamp used to determine lease validity.
            If None, the current system time is used.

    Returns:
        A frozenset of artifact set digests that are currently protected by valid leases.
    """
    timestamp = time.time() if now is None else now
    active: set[str] = set()
    for path in (root / constants.RUNTIME_LEASES_DIRECTORY).glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            digest = payload.get("artifact_set_digest")
            valid_digest = (
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            )
            if (
                valid_digest
                and isinstance(payload.get("expiry"), (int, float))
                and payload["expiry"] > timestamp
            ):
                active.add(payload["artifact_set_digest"])
            elif isinstance(payload.get("expiry"), (int, float)) and payload["expiry"] <= timestamp:
                path.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError):
            continue
    return frozenset(active)
