"""Inspect stopped macOS guest attempt layers without exposing guest paths upstream."""

from __future__ import annotations

import stat
import tarfile
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from re import fullmatch
from uuid import uuid4

from ....utils import sha256_digest
from ...contracts import DeltaEffect
from ...infrastructure import InfrastructureProcessResult
from ...vfs.agent_workspace import AgentWorkspaceContext, AgentWorkspaceManager
from ...vfs.coordinator import PublicationCoordinator
from ...vfs.delta import (
    ObservedDelta,
    ObservedDeltaEntry,
    ObservedObjectKind,
    UnrepresentableDeltaError,
    normalize_delta,
)
from ...vfs.manifest import SnapshotManifest
from ...vfs.materializer import GenerationView
from ...vfs.models import (
    AffectedPathIdentity,
    BaseSnapshotId,
    BranchId,
    CanonicalDelta,
    CanonicalDeltaEntry,
    ContentReference,
    DestinationFilesystemCapabilities,
    GenerationId,
    HostObjectIdentity,
    MetadataPreservation,
    ObjectKind,
)
from ...vfs.namespace import validate_relative_path
from ...vfs.representability import FilesystemRepresentabilityValidator
from ...vfs.staging import StagedContentStore
from .backend import PreparedMacosRuntime

_OPAQUE_XATTRS = frozenset(
    {"SCHILY.xattr.trusted.overlay.opaque", "SCHILY.xattr.user.overlay.opaque"}
)
_WHITEOUT_XATTRS = frozenset(
    {"SCHILY.xattr.trusted.overlay.whiteout", "SCHILY.xattr.user.overlay.whiteout"}
)
_METACOPY_XATTRS = frozenset(
    {"SCHILY.xattr.trusted.overlay.metacopy", "SCHILY.xattr.user.overlay.metacopy"}
)
_REDIRECT_XATTRS = frozenset(
    {"SCHILY.xattr.trusted.overlay.redirect", "SCHILY.xattr.user.overlay.redirect"}
)
_INTERNAL_OVERLAY_XATTRS = frozenset(
    {
        "SCHILY.xattr.trusted.overlay.impure",
        "SCHILY.xattr.trusted.overlay.origin",
        "SCHILY.xattr.trusted.overlay.uuid",
        "SCHILY.xattr.user.overlay.impure",
        "SCHILY.xattr.user.overlay.origin",
        "SCHILY.xattr.user.overlay.uuid",
    }
)
_IDENTIFIER_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"

_CREATE_OVERLAY_SCRIPT = r"""
set -eu
root=$1
lower=$2
target=$3
quota=$4
case "$target" in "$root"/branches/*|"$root"/attempts/*) ;; *) exit 64;; esac
case "$lower" in /run/loop/snapshots/*/tree|"$root"/branches/*/merged) ;; *) exit 64;; esac
test -d "$lower"
test ! -e "$target"
cleanup() {
  status=$?
  if test "$status" -ne 0; then
    if mountpoint -q "$target/merged"; then sudo -n umount "$target/merged"; fi
    if mountpoint -q "$target/layer"; then sudo -n umount "$target/layer"; fi
    sudo -n rm -rf -- "$target"
  fi
  exit "$status"
}
trap cleanup EXIT
mkdir -p "$target/layer" "$target/merged"
chmod 700 "$target" "$target/layer"
chmod 755 "$target/merged"
if test "$quota" -gt 0; then
  sudo -n mount -t tmpfs -o "size=$quota,nosuid,nodev,noexec,mode=700" tmpfs "$target/layer"
  sudo -n chown "$(id -u):$(id -g)" "$target/layer"
fi
mkdir -p "$target/layer/upper" "$target/layer/work"
chmod 700 "$target/layer/upper" "$target/layer/work"
if test "$quota" -gt 0; then
  user=$(id -un)
  uid_mapping=$(awk -F: -v user="$user" '$1 == user { if (found) exit 2; print $2 ":" $3; found=1 } END { if (!found) exit 1 }' /etc/subuid)
  gid_mapping=$(awk -F: -v user="$user" '$1 == user { if (found) exit 2; print $2 ":" $3; found=1 } END { if (!found) exit 1 }' /etc/subgid)
  uid_start=${uid_mapping%:*}; uid_length=${uid_mapping#*:}
  gid_start=${gid_mapping%:*}; gid_length=${gid_mapping#*:}
  case "$uid_start:$uid_length:$gid_start:$gid_length" in *[!0-9:]*) exit 64;; esac
  test "$uid_length" -gt 1000 && test "$gid_length" -gt 1000
  mapped_uid=$((uid_start + 999))
  mapped_gid=$((gid_start + 999))
  sudo -n chown "$mapped_uid:$mapped_gid" "$target/layer/upper" "$target/layer/work"
  sudo -n chmod 755 "$target/layer/upper"
fi
sudo -n mount -t overlay overlay -o "lowerdir=$lower,upperdir=$target/layer/upper,workdir=$target/layer/work,metacopy=on" "$target/merged"
trap - EXIT
""".strip()

_INITIALIZE_BRANCH_SCRIPT = r"""
set -eu
branch=$1
test -d "$branch/merged"
user=$(id -un)
uid_mapping=$(awk -F: -v user="$user" '$1 == user { if (found) exit 2; print $2 ":" $3; found=1 } END { if (!found) exit 1 }' /etc/subuid)
gid_mapping=$(awk -F: -v user="$user" '$1 == user { if (found) exit 2; print $2 ":" $3; found=1 } END { if (!found) exit 1 }' /etc/subgid)
uid_start=${uid_mapping%:*}; uid_length=${uid_mapping#*:}
gid_start=${gid_mapping%:*}; gid_length=${gid_mapping#*:}
case "$uid_start:$uid_length:$gid_start:$gid_length" in *[!0-9:]*) exit 64;; esac
test "$uid_length" -gt 1000 && test "$gid_length" -gt 1000
mapped_uid=$((uid_start + 999))
mapped_gid=$((gid_start + 999))
sudo -n chown -hR "$mapped_uid:$mapped_gid" "$branch/merged"
sudo -n chmod 755 "$branch/merged"
""".strip()

_RECOVER_BRANCH_SCRIPT = r"""
set -eu
root=$1
lower=$2
target=$3
case "$target" in "$root"/branches/*) ;; *) exit 64;; esac
test -d "$lower"
if test ! -e "$target"; then
  mkdir -p "$target/layer/upper" "$target/layer/work" "$target/merged"
  chmod 700 "$target" "$target/layer" "$target/layer/upper" "$target/layer/work"
  chmod 755 "$target/merged"
  sudo -n mount -t overlay overlay -o "lowerdir=$lower,upperdir=$target/layer/upper,workdir=$target/layer/work,metacopy=on" "$target/merged"
  printf 'reconstructed\n'
  exit 0
fi
test -d "$target/layer/upper" && test -d "$target/layer/work" && test -d "$target/merged"
chmod 755 "$target/merged"
if ! mountpoint -q "$target/merged"; then
  sudo -n mount -t overlay overlay -o "lowerdir=$lower,upperdir=$target/layer/upper,workdir=$target/layer/work,metacopy=on" "$target/merged"
fi
printf 'recovered\n'
""".strip()

_ARCHIVE_ATTEMPT_SCRIPT = r"""
set -eu
attempt=$1
archive=$2
test -d "$attempt/layer/upper"
test ! -e "$archive"
sudo -n tar --xattrs --acls --numeric-owner -C "$attempt/layer/upper" -cpf "$archive" .
sudo -n chown "$(id -u):$(id -g)" "$archive"
chmod 600 "$archive"
""".strip()

_DISPOSE_OVERLAY_SCRIPT = r"""
set -eu
root=$1
target=$2
case "$target" in "$root"/branches/*|"$root"/attempts/*) ;; *) exit 64;; esac
if mountpoint -q "$target/merged"; then sudo -n umount "$target/merged"; fi
if mountpoint -q "$target/layer"; then sudo -n umount "$target/layer"; fi
sudo -n rm -rf -- "$target"
""".strip()

_PREPARE_TRANSACTION_SCRIPT = r"""
set -eu
transaction=$1
archive=$2
mkdir -p "$transaction/payloads" "$transaction/renames"
chmod 700 "$transaction" "$transaction/payloads" "$transaction/renames"
if test -f "$archive"; then tar -C "$transaction/payloads" -xf "$archive"; rm -f -- "$archive"; fi
""".strip()

_STAGE_RENAME_SCRIPT = r"""
set -eu
view=$1
transaction=$2
index=$3
source=$4
stage="$transaction/renames/$index"
test -f "$transaction/renames.prepared" && exit 0
test -e "$stage" && exit 0
case "$source" in ''|/*|*'//'*) exit 64;; esac
old_ifs=$IFS; IFS=/; set -- $source; IFS=$old_ifs
current=$view
last=$#
i=1
for part do
  test "$part" != . && test "$part" != .. && test -n "$part" || exit 64
  if test "$i" -lt "$last"; then
    test -d "$current/$part" && test ! -L "$current/$part" || exit 65
    current="$current/$part"
  fi
  i=$((i + 1))
done
test -e "$view/$source" || test -L "$view/$source"
mv -- "$view/$source" "$stage"
sync -f "$transaction/renames"
""".strip()

_MARK_RENAMES_PREPARED_SCRIPT = r"""
set -eu
transaction=$1
: >"$transaction/renames.prepared"
sync -f "$transaction"
""".strip()

_APPLY_ENTRY_SCRIPT = r"""
set -eu
view=$1
transaction=$2
index=$3
effect=$4
destination=$5
source=$6
kind=$7
mode=$8
target=$9
marker="$transaction/$index.done"
test -f "$marker" && exit 0
case "$destination" in ''|/*|*'//'*) exit 64;; esac
old_ifs=$IFS; IFS=/; set -- $destination; IFS=$old_ifs
parent=$view
last=$#
i=1
name=
for part do
  test "$part" != . && test "$part" != .. && test -n "$part" || exit 64
  if test "$i" -lt "$last"; then
    test -d "$parent/$part" && test ! -L "$parent/$part" || exit 65
    parent="$parent/$part"
  else
    name=$part
  fi
  i=$((i + 1))
done
destination_path="$parent/$name"
remove_destination() {
  if test -d "$destination_path" && test ! -L "$destination_path"; then
    rmdir -- "$destination_path"
  else
    rm -f -- "$destination_path"
  fi
}
case "$effect" in
  delete)
    if test -e "$destination_path" || test -L "$destination_path"; then remove_destination; fi
    ;;
  rename)
    stage="$transaction/renames/$index"
    if test -e "$stage" || test -L "$stage"; then
      if test -e "$destination_path" || test -L "$destination_path"; then remove_destination; fi
      mv -- "$stage" "$destination_path"
    else
      test -e "$destination_path" || test -L "$destination_path"
    fi
    ;;
  metadata)
    chmod "$mode" "$destination_path"
    ;;
  create|replace)
    if test -e "$destination_path" || test -L "$destination_path"; then remove_destination; fi
    case "$kind" in
      directory) mkdir -- "$destination_path"; chmod "$mode" "$destination_path";;
      symlink) ln -s -- "$target" "$destination_path";;
      file)
        temporary="$parent/.loop-apply-$index"
        rm -f -- "$temporary"
        cp -- "$transaction/payloads/$index" "$temporary"
        chmod "$mode" "$temporary"
        mv -- "$temporary" "$destination_path"
        ;;
      *) exit 64;;
    esac
    ;;
  *) exit 64;;
esac
sync -f "$parent"
: >"$marker"
sync -f "$transaction"
""".strip()

_MARK_TRANSACTION_APPLIED_SCRIPT = r"""
set -eu
transaction=$1
: >"$transaction/applied"
rm -rf -- "$transaction/payloads"
sync -f "$transaction"
""".strip()


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    """Carry the portable state needed to classify one upper-layer object."""

    kind: ObjectKind
    identity: HostObjectIdentity | None
    content: ContentReference | None
    mode: int
    symlink_target: str | None


@dataclass(frozen=True, slots=True)
class MacosAttemptLayer:
    """Name one guest-private attempt overlay and its trusted export locations.

    Args:
        attempt_id (str): Validated attempt identity.
        branch_reference (str): Opaque owning branch reference.
        mount_source (str): Trusted guest merged mount passed only to OCI management.
        guest_directory (str): Trusted guest-private attempt directory.
        guest_archive (str): Trusted guest-private upper-layer archive.
        host_archive (Path): Private host destination for archive inspection.
    """

    attempt_id: str
    branch_reference: str
    mount_source: str
    guest_directory: str
    guest_archive: str
    host_archive: Path


@dataclass(frozen=True, slots=True)
class MacosLeasedWorkspace:
    """Bind a leased logical generation to one platform-private attempt overlay.

    Args:
        context (AgentWorkspaceContext): Immutable logical agent generation used by the attempt.
        layer (MacosAttemptLayer): Guest-private fresh writable attempt layer.
        generation_reference (str): Label-safe opaque OCI generation identity.
    """

    context: AgentWorkspaceContext
    layer: MacosAttemptLayer
    generation_reference: str


class MacosDurableWorkspaceLease:
    """Hold one logical generation and attempt overlay across job operations.

    Args:
        leased (MacosLeasedWorkspace): Exact logical generation and attempt layer.
        generation_lease (AbstractContextManager): Active logical generation lease.
        materializer (MacosWorkspaceMaterializer): Owner of the attempt overlay.
    """

    leased: MacosLeasedWorkspace
    _generation_lease: AbstractContextManager
    _materializer: MacosWorkspaceMaterializer
    _released: bool
    _discarded: bool

    def __init__(
        self,
        leased: MacosLeasedWorkspace,
        generation_lease: AbstractContextManager,
        materializer: MacosWorkspaceMaterializer,
    ) -> None:
        self.leased = leased
        self._generation_lease = generation_lease
        self._materializer = materializer
        self._released = False
        self._discarded = False

    def release(self) -> None:
        """Release the logical generation without discarding the stopped overlay."""
        if not self._released:
            self._generation_lease.__exit__(None, None, None)
            self._released = True

    def discard(self) -> None:
        """Release the generation and reclaim the durable attempt overlay exactly once."""
        self.release()
        if not self._discarded:
            self._materializer.discard_attempt(self.leased.layer)
            self._discarded = True


class MacosWorkspaceMaterializer:
    """Own guest OverlayFS branches, attempts, canonical replay, and cleanup.

    Args:
        prepared (PreparedMacosRuntime): Running attested managed runtime.
        content_store (StagedContentStore): Durable canonical file-content source.
        archive_directory (Path): Private host directory beneath managed instance state.
        attempt_write_bytes (int): Positive tmpfs quota for each fresh attempt upper layer.
    """

    prepared: PreparedMacosRuntime
    content_store: StagedContentStore
    archive_directory: Path
    guest_root: str
    attempt_write_bytes: int

    def __init__(
        self,
        prepared: PreparedMacosRuntime,
        content_store: StagedContentStore,
        archive_directory: Path,
        attempt_write_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """Bind materialization to one attested guest and private host archive root."""
        if attempt_write_bytes <= 0:
            raise ValueError("Attempt write quota must be positive.")
        state_path = prepared.endpoint.state_path
        suffix = "/.local/share/containerd"
        if not state_path.endswith(suffix):
            raise ValueError("Managed guest state path cannot locate its private home.")
        self.guest_root = state_path.removesuffix(suffix) + "/.local/share/loop/workspaces"
        context = prepared.instance.context
        archive_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            resolved = archive_directory.resolve(strict=True)
            resolved.relative_to(context.state_root)
        except (OSError, ValueError) as error:
            raise ValueError("Workspace archive root is outside private instance state.") from error
        if archive_directory.is_symlink() or not archive_directory.is_dir():
            raise ValueError("Workspace archive root must be a private directory.")
        archive_directory.chmod(0o700)
        self.prepared = prepared
        self.content_store = content_store
        self.archive_directory = resolved
        self.attempt_write_bytes = attempt_write_bytes

    def create_branch(self, base: BaseSnapshotId) -> BranchId:
        """Create one guest-private cumulative branch over an immutable snapshot.

        Args:
            base (BaseSnapshotId): Immutable snapshot available through read-only VirtioFS.

        Returns:
            BranchId: Opaque private branch identity.

        Raises:
            ValueError: The base identity is unsafe or guest setup fails.
        """
        _validate_identifier(base.value, "snapshot")
        branch = BranchId(value=f"branch-{uuid4().hex}")
        self._create_overlay(
            f"/run/loop/snapshots/{base.value}/tree",
            self._branch_directory(branch.value),
            "workspace.branch.create",
            0,
        )
        try:
            self._initialize_branch(branch)
        except BaseException:
            self.dispose_branch(branch)
            raise
        return branch

    def recover_branch(
        self,
        base: BaseSnapshotId,
        branch: BranchId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> None:
        """Remount or reconstruct one persisted branch after a VM restart.

        Args:
            base (BaseSnapshotId): Immutable snapshot beneath the branch.
            branch (BranchId): Persisted private branch identity.
            committed_transactions (tuple[tuple[str, CanonicalDelta], ...]): Ordered durable
                transactions used to reconstruct guest state lost with a replaced VM.

        Raises:
            ValueError: If either opaque identity is unsafe or guest recovery fails.
        """
        _validate_identifier(base.value, "snapshot")
        _validate_identifier(branch.value, "branch")
        result = self._run_guest(
            (
                "/bin/sh",
                "-c",
                _RECOVER_BRANCH_SCRIPT,
                "loop-workspace-recover",
                self.guest_root,
                f"/run/loop/snapshots/{base.value}/tree",
                self._branch_directory(branch.value),
            ),
            "workspace.branch.recover",
        )
        reconstructed = result.stdout == b"reconstructed\n"
        if not reconstructed and result.stdout != b"recovered\n":
            raise RuntimeError("Managed branch recovery returned invalid evidence.")
        try:
            self._initialize_branch(branch)
            if not reconstructed:
                return
            for transaction_id, delta in committed_transactions:
                self.apply_delta(branch, transaction_id, delta)
        except BaseException:
            if reconstructed:
                self.dispose_branch(branch)
            raise

    def generation_view(self, branch: BranchId, generation: GenerationId) -> GenerationView:
        """Return an opaque read-only reference for the current branch generation.

        Args:
            branch (BranchId): Owning private branch.
            generation (GenerationId): Current monotonically advancing generation.

        Returns:
            GenerationView: Opaque platform reference with no guest or host path.
        """
        _validate_identifier(branch.value, "branch")
        if not generation.value.isdigit():
            raise ValueError("Workspace generation identity is invalid.")
        return GenerationView(reference=branch.value)

    def begin_attempt(self, view: GenerationView, attempt_id: str) -> MacosAttemptLayer:
        """Create a fresh guest-private attempt overlay over one leased generation.

        Args:
            view (GenerationView): Opaque leased branch generation.
            attempt_id (str): Unique execution attempt identity.

        Returns:
            MacosAttemptLayer: Platform-private attempt mount and archive identities.

        Raises:
            ValueError: A reference is unsafe or guest setup fails.
        """
        _validate_identifier(view.reference, "branch")
        _validate_identifier(attempt_id, "attempt")
        guest_directory = f"{self.guest_root}/attempts/{attempt_id}"
        self._create_overlay(
            f"{self._branch_directory(view.reference)}/merged",
            guest_directory,
            "workspace.attempt.create",
            self.attempt_write_bytes,
        )
        archive_name = f"{attempt_id}-{uuid4().hex}.tar"
        return MacosAttemptLayer(
            attempt_id=attempt_id,
            branch_reference=view.reference,
            mount_source=f"{guest_directory}/merged",
            guest_directory=guest_directory,
            guest_archive=f"{guest_directory}/upper.tar",
            host_archive=self.archive_directory / archive_name,
        )

    def archive_attempt(self, attempt: MacosAttemptLayer) -> Path:
        """Freeze and copy one stopped attempt upper into private host state.

        Args:
            attempt (MacosAttemptLayer): Stopped platform-private attempt.

        Returns:
            Path: New private host archive ready for bounded inspection.

        Raises:
            RuntimeError: Guest archive or Lima copy fails.
        """
        _validate_identifier(attempt.attempt_id, "attempt")
        self._run_guest(
            (
                "/bin/sh",
                "-c",
                _ARCHIVE_ATTEMPT_SCRIPT,
                "loop-workspace-archive",
                attempt.guest_directory,
                attempt.guest_archive,
            ),
            "workspace.attempt.archive",
        )
        result = self.prepared.instance.runner.copy_from_guest(
            self.prepared.instance.context,
            attempt.guest_archive,
            attempt.host_archive,
        )
        _require_success(result, "Managed attempt archive copy failed.")
        return attempt.host_archive

    def discard_attempt(self, attempt: MacosAttemptLayer) -> None:
        """Unmount and reclaim one attempt plus its host archive.

        Args:
            attempt (MacosAttemptLayer): Attempt to reclaim after inspection or denial.
        """
        _validate_identifier(attempt.attempt_id, "attempt")
        self._dispose_overlay(attempt.guest_directory, "workspace.attempt.dispose")
        attempt.host_archive.unlink(missing_ok=True)

    def apply_delta(self, branch: BranchId, transaction_id: str, delta: CanonicalDelta) -> None:
        """Idempotently replay one canonical delta through a quiescent branch view.

        Args:
            branch (BranchId): Owning private branch.
            transaction_id (str): Durable transaction idempotency identity.
            delta (CanonicalDelta): Already-inspected canonical effects.

        Raises:
            ValueError: A branch, transaction, or effect is unsafe.
            RuntimeError: Guest replay or payload transfer fails.
        """
        _validate_identifier(branch.value, "branch")
        if not transaction_id:
            raise ValueError("Workspace transaction identity is invalid.")
        transaction_key = _transaction_key(transaction_id)
        transaction = f"{self.guest_root}/transactions/{branch.value}/{transaction_key}"
        if self._transaction_applied(transaction):
            return
        upload = self._build_payload_archive(transaction_key, delta)
        guest_archive = f"{transaction}/payloads.tar"
        try:
            self._run_guest(
                (
                    "/bin/sh",
                    "-c",
                    'set -eu; mkdir -p "$1"; chmod 700 "$1"',
                    "loop-workspace-transaction",
                    transaction,
                ),
                "workspace.transaction.create",
            )
            result = self.prepared.instance.runner.copy_to_guest(
                self.prepared.instance.context, upload, guest_archive
            )
            _require_success(result, "Managed transaction payload copy failed.")
            self._run_guest(
                (
                    "/bin/sh",
                    "-c",
                    _PREPARE_TRANSACTION_SCRIPT,
                    "loop-workspace-transaction",
                    transaction,
                    guest_archive,
                ),
                "workspace.transaction.prepare",
            )
            view = f"{self._branch_directory(branch.value)}/merged"
            for index, entry in enumerate(delta.entries):
                if entry.effect is DeltaEffect.RENAME:
                    self._run_guest(
                        (
                            "/usr/bin/sudo",
                            "-n",
                            "/bin/sh",
                            "-c",
                            _STAGE_RENAME_SCRIPT,
                            "loop-workspace-rename",
                            view,
                            transaction,
                            str(index),
                            entry.source_path or "",
                        ),
                        "workspace.transaction.stage_rename",
                    )
            self._run_guest(
                (
                    "/bin/sh",
                    "-c",
                    _MARK_RENAMES_PREPARED_SCRIPT,
                    "loop-workspace-renames-prepared",
                    transaction,
                ),
                "workspace.transaction.prepare_renames",
            )
            for index, entry in enumerate(delta.entries):
                self._apply_entry(view, transaction, index, entry)
            self._initialize_branch(branch)
            self._run_guest(
                (
                    "/bin/sh",
                    "-c",
                    _MARK_TRANSACTION_APPLIED_SCRIPT,
                    "loop-workspace-applied",
                    transaction,
                ),
                "workspace.transaction.complete",
            )
        finally:
            upload.unlink(missing_ok=True)

    def fork_branch(
        self,
        base: BaseSnapshotId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> BranchId:
        """Create an independent branch by replaying the parent's canonical journal.

        Args:
            base (BaseSnapshotId): Shared immutable starting snapshot.
            committed_transactions (tuple[tuple[str, CanonicalDelta], ...]): Ordered parent
                transaction identities and effects.

        Returns:
            BranchId: New private branch containing the replayed committed state.
        """
        branch = self.create_branch(base)
        try:
            for transaction_id, delta in committed_transactions:
                self.apply_delta(branch, transaction_id, delta)
        except BaseException:
            self.dispose_branch(branch)
            raise
        return branch

    def dispose_branch(self, branch: BranchId) -> None:
        """Unmount and reclaim one quiescent private branch.

        Args:
            branch (BranchId): Private branch whose leases have ended.
        """
        _validate_identifier(branch.value, "branch")
        self._dispose_overlay(self._branch_directory(branch.value), "workspace.branch.dispose")

    def _create_overlay(self, lower: str, target: str, operation: str, quota: int) -> None:
        """Create one fixed-shape trusted guest overlay mount."""
        self._run_guest(
            (
                "/bin/sh",
                "-c",
                _CREATE_OVERLAY_SCRIPT,
                "loop-workspace-overlay",
                self.guest_root,
                lower,
                target,
                str(quota),
            ),
            operation,
        )

    def _dispose_overlay(self, target: str, operation: str) -> None:
        """Unmount and remove one validated materializer-owned guest directory."""
        self._run_guest(
            (
                "/bin/sh",
                "-c",
                _DISPOSE_OVERLAY_SCRIPT,
                "loop-workspace-dispose",
                self.guest_root,
                target,
            ),
            operation,
        )

    def _run_guest(self, argv: tuple[str, ...], operation: str) -> InfrastructureProcessResult:
        """Run one fixed trusted guest-management operation and require bounded success."""
        result = self.prepared.instance.runner.run_guest(
            self.prepared.instance.context,
            argv,
            operation=operation,
            deadline_seconds=120.0,
        )
        _require_success(result, "Managed guest workspace operation failed.")
        return result

    def _branch_directory(self, branch: str) -> str:
        """Resolve one validated opaque branch into the private guest root."""
        return f"{self.guest_root}/branches/{branch}"

    def _initialize_branch(self, branch: BranchId) -> None:
        """Make one private branch accessible only to the sandbox command identity."""
        self._run_guest(
            (
                "/bin/sh",
                "-c",
                _INITIALIZE_BRANCH_SCRIPT,
                "loop-workspace-initialize",
                self._branch_directory(branch.value),
            ),
            "workspace.branch.initialize",
        )

    def _transaction_applied(self, transaction: str) -> bool:
        """Return whether a durable guest transaction marker already exists."""
        result = self.prepared.instance.runner.run_guest(
            self.prepared.instance.context,
            ("/usr/bin/test", "-f", f"{transaction}/applied"),
            operation="workspace.transaction.status",
        )
        if result.stdout_truncated or result.stderr_truncated or result.exit_code not in {0, 1}:
            raise RuntimeError("Managed transaction status failed.")
        return result.exit_code == 0

    def _build_payload_archive(self, transaction_key: str, delta: CanonicalDelta) -> Path:
        """Build one private file-only canonical replay payload archive."""
        archive = self.archive_directory / f"transaction-{transaction_key}-{uuid4().hex}.tar"
        try:
            with tarfile.open(archive, mode="x") as bundle:
                for index, entry in enumerate(delta.entries):
                    if (
                        entry.effect not in {DeltaEffect.CREATE, DeltaEffect.REPLACE}
                        or entry.object_kind is not ObjectKind.FILE
                        or entry.content is None
                    ):
                        continue
                    with self.content_store.open(entry.content) as source:
                        member = tarfile.TarInfo(str(index))
                        member.size = entry.content.size
                        member.mode = 0o600
                        bundle.addfile(member, source)
            archive.chmod(0o600)
            return archive
        except BaseException:
            archive.unlink(missing_ok=True)
            raise

    def _apply_entry(
        self,
        view: str,
        transaction: str,
        index: int,
        entry: CanonicalDeltaEntry,
    ) -> None:
        """Compile one canonical effect into the fixed guest maintenance command."""
        self._run_guest(
            (
                "/usr/bin/sudo",
                "-n",
                "/bin/sh",
                "-c",
                _APPLY_ENTRY_SCRIPT,
                "loop-workspace-apply",
                view,
                transaction,
                str(index),
                entry.effect.value,
                entry.destination_path,
                entry.source_path or "",
                entry.object_kind.value if entry.object_kind else "",
                format(entry.mode or entry.metadata.basic_mode or 0, "o"),
                entry.symlink_target or "",
            ),
            "workspace.transaction.apply",
        )


class MacosWorkspaceCoordinator:
    """Coordinate generation leases, attempt observation, validation, and publication.

    Args:
        manager (AgentWorkspaceManager): Durable logical agent-lineage owner.
        materializer (MacosWorkspaceMaterializer): Guest overlay and archive owner.
        inspector (MacosDeltaArchiveInspector): Trusted upper-layer archive inspector.
        publication (PublicationCoordinator): Journaled host and branch publication service.
        manifest_loader (Callable[[BaseSnapshotId], SnapshotManifest]): Authenticated immutable
            snapshot manifest loader.
        destination_capabilities (DestinationFilesystemCapabilities): Measured host destination
            filesystem facts.
    """

    manager: AgentWorkspaceManager
    materializer: MacosWorkspaceMaterializer
    inspector: MacosDeltaArchiveInspector
    publication: PublicationCoordinator
    manifest_loader: Callable[[BaseSnapshotId], SnapshotManifest]
    destination_capabilities: DestinationFilesystemCapabilities
    validator: FilesystemRepresentabilityValidator

    def __init__(
        self,
        manager: AgentWorkspaceManager,
        materializer: MacosWorkspaceMaterializer,
        inspector: MacosDeltaArchiveInspector,
        publication: PublicationCoordinator,
        manifest_loader: Callable[[BaseSnapshotId], SnapshotManifest],
        destination_capabilities: DestinationFilesystemCapabilities,
    ) -> None:
        """Bind one workspace's portable and macOS-specific transaction boundaries."""
        self.manager = manager
        self.materializer = materializer
        self.inspector = inspector
        self.publication = publication
        self.manifest_loader = manifest_loader
        self.destination_capabilities = destination_capabilities
        self.validator = FilesystemRepresentabilityValidator()

    @contextmanager
    def lease(
        self,
        workspace_id: str,
        agent_run_id: str,
        attempt_id: str,
    ) -> Generator[MacosLeasedWorkspace, None, None]:
        """Lease one generation and create a fresh attempt overlay for its lifetime.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.
            attempt_id (str): Unique execution attempt identity.

        Yields:
            MacosLeasedWorkspace: Logical context and platform-private fresh attempt.
        """
        with self.manager.lease_generation(workspace_id, agent_run_id) as (context, view):
            layer = self.materializer.begin_attempt(view, attempt_id)
            leased = MacosLeasedWorkspace(
                context=context,
                layer=layer,
                generation_reference=_generation_reference(context),
            )
            try:
                yield leased
            finally:
                self.materializer.discard_attempt(layer)

    def observe(self, leased: MacosLeasedWorkspace) -> CanonicalDelta:
        """Freeze, inspect, and destination-validate one stopped attempt upper.

        Args:
            leased (MacosLeasedWorkspace): Stopped attempt while its generation remains leased.

        Returns:
            CanonicalDelta: Complete representable canonical effects.

        Raises:
            UnrepresentableDeltaError: Inspection or destination representation is unsafe.
        """
        archive = self.materializer.archive_attempt(leased.layer)
        context = leased.context
        delta = self.inspector.inspect(
            leased.layer.attempt_id,
            archive,
            self.manifest_loader(context.base_snapshot_id),
            context.committed_deltas,
            context.affected_path_identities,
        )
        return self.validator.validate(delta, self.destination_capabilities)

    def lease_job(
        self,
        workspace_id: str,
        agent_run_id: str,
        attempt_id: str,
    ) -> MacosDurableWorkspaceLease:
        """Lease a generation and retain its fresh overlay for a durable job.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.
            attempt_id (str): Unique durable-job identity.

        Returns:
            MacosDurableWorkspaceLease: Explicitly released durable generation lease.
        """
        generation_lease = self.manager.lease_generation(workspace_id, agent_run_id)
        context, view = generation_lease.__enter__()
        try:
            layer = self.materializer.begin_attempt(view, attempt_id)
        except BaseException:
            generation_lease.__exit__(None, None, None)
            raise
        return MacosDurableWorkspaceLease(
            MacosLeasedWorkspace(
                context=context,
                layer=layer,
                generation_reference=_generation_reference(context),
            ),
            generation_lease,
            self.materializer,
        )

    def resume_job(
        self,
        workspace_id: str,
        agent_run_id: str,
        generation_reference: str,
        layer: MacosAttemptLayer,
    ) -> MacosDurableWorkspaceLease:
        """Reacquire one persisted job generation after application restart.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.
            generation_reference (str): Persisted opaque generation identity.
            layer (MacosAttemptLayer): Persisted private attempt-layer identity.

        Returns:
            MacosDurableWorkspaceLease: Reacquired exact generation lease.

        Raises:
            RuntimeError: If the agent generation no longer matches the job.
        """
        generation_lease = self.manager.lease_generation(workspace_id, agent_run_id)
        context, _view = generation_lease.__enter__()
        if _generation_reference(context) != generation_reference:
            generation_lease.__exit__(None, None, None)
            raise RuntimeError("Durable job workspace generation is stale.")
        return MacosDurableWorkspaceLease(
            MacosLeasedWorkspace(context, layer, generation_reference),
            generation_lease,
            self.materializer,
        )

    def publish(
        self,
        leased: MacosLeasedWorkspace,
        transaction_id: str,
        delta: CanonicalDelta,
    ) -> AgentWorkspaceContext:
        """Publish one approved observed delta after its generation lease has ended.

        Args:
            leased (MacosLeasedWorkspace): Attempt context that produced the delta.
            transaction_id (str): Stable publication idempotency identity.
            delta (CanonicalDelta): Fully observed and representability-validated effects.

        Returns:
            AgentWorkspaceContext: Advanced durable agent lineage.
        """
        self.manager.require_publishable(leased.context)
        return self.publication.publish(transaction_id, leased.context, delta)

    def deny(self, leased: MacosLeasedWorkspace) -> None:
        """Reclaim staged content for a denied attempt without creating a journal record.

        Args:
            leased (MacosLeasedWorkspace): Attempt whose observed effects were denied.
        """
        self.inspector.content_store.discard(leased.layer.attempt_id)


def _generation_reference(context: AgentWorkspaceContext) -> str:
    """Return the label-safe identity of one exact logical generation."""
    return (
        "generation-"
        + sha256_digest(f"{context.branch_id.value}:{context.generation_id.value}".encode())[:32]
    )


class MacosDeltaArchiveInspector:
    """Translate one trusted guest upper-layer tar into a portable canonical delta.

    Args:
        content_store (StagedContentStore): Durable host-private publication content store.
        maximum_entries (int): Maximum accepted archive members and observed effects.
        maximum_content_bytes (int): Maximum total staged regular-file bytes.
        maximum_archive_bytes (int): Maximum accepted archive size including metadata.
        ignore_path (Callable[[str], bool] | None): Workspace-relative publication exclusion
            policy. Excluded effects are discarded before content staging.
    """

    content_store: StagedContentStore
    maximum_entries: int
    maximum_content_bytes: int
    maximum_archive_bytes: int
    ignore_path: Callable[[str], bool]

    def __init__(
        self,
        content_store: StagedContentStore,
        *,
        maximum_entries: int = 10_000,
        maximum_content_bytes: int = 1 << 30,
        maximum_archive_bytes: int = 2 << 30,
        ignore_path: Callable[[str], bool] | None = None,
    ) -> None:
        """Configure bounded inspection of trusted guest-produced archives."""
        if min(maximum_entries, maximum_content_bytes, maximum_archive_bytes) < 0:
            raise ValueError("Workspace archive limits cannot be negative.")
        self.content_store = content_store
        self.maximum_entries = maximum_entries
        self.maximum_content_bytes = maximum_content_bytes
        self.maximum_archive_bytes = maximum_archive_bytes
        self.ignore_path = ignore_path or (lambda _path: False)

    def inspect(
        self,
        staging_id: str,
        archive: Path,
        base: SnapshotManifest,
        committed: tuple[CanonicalDelta, ...] = (),
        affected_identities: tuple[AffectedPathIdentity, ...] = (),
    ) -> CanonicalDelta:
        """Inspect and stage one stopped attempt's complete upper-layer archive.

        Args:
            staging_id (str): Stable owner of content staged for this attempt.
            archive (Path): Loop-private tar copied from the trusted guest plane.
            base (SnapshotManifest): Immutable starting workspace manifest.
            committed (tuple[CanonicalDelta, ...]): Agent-local committed delta journal.
            affected_identities (tuple[AffectedPathIdentity, ...]): Current host identities for
                paths affected after the base snapshot.

        Returns:
            CanonicalDelta: Complete normalized attempt effects using only virtual paths.

        Raises:
            UnrepresentableDeltaError: The archive or an observed effect is unsafe or lossy.
        """
        self._validate_archive(archive)
        current = _current_state(base, committed, affected_identities)
        upper: dict[str, WorkspaceEntry] = {}
        deletions: set[str] = set()
        opaque_directories: set[str] = set()
        staged_bytes = 0
        try:
            with tarfile.open(archive, mode="r:*") as bundle:
                members = bundle.getmembers()
                if len(members) > self.maximum_entries:
                    raise UnrepresentableDeltaError("Delta archive exceeds its entry quota.")
                for member in members:
                    metadata_headers = dict(member.pax_headers)
                    _validate_xattrs(metadata_headers)
                    path = _member_path(member.name)
                    if path is None:
                        continue
                    whiteout = _whiteout_path(member, path)
                    if whiteout is not None:
                        if whiteout and not self.ignore_path(whiteout):
                            deletions.add(whiteout)
                        elif not whiteout and not self.ignore_path(path):
                            parent = str(PurePosixPath(path).parent)
                            opaque_directories.add("" if parent == "." else parent)
                        continue
                    if self.ignore_path(path):
                        continue
                    if _truthy_xattr(metadata_headers, _OPAQUE_XATTRS):
                        opaque_directories.add(path)
                    if _truthy_xattr(metadata_headers, _WHITEOUT_XATTRS):
                        deletions.add(path)
                        continue
                    previous = current.get(path)
                    redirect = _xattr_value(metadata_headers, _REDIRECT_XATTRS)
                    if redirect is not None:
                        try:
                            validate_relative_path(redirect)
                        except ValueError as error:
                            raise UnrepresentableDeltaError(
                                "Overlay redirect metadata is unsafe."
                            ) from error
                        previous = current.get(redirect)
                        if previous is None:
                            raise UnrepresentableDeltaError(
                                "Overlay redirect has no visible lower object."
                            )
                    entry, consumed = self._entry(
                        staging_id,
                        bundle,
                        member,
                        previous,
                        _truthy_xattr(metadata_headers, _METACOPY_XATTRS),
                        self.maximum_content_bytes - staged_bytes,
                    )
                    staged_bytes += consumed
                    upper[path] = entry
        except (OSError, tarfile.TarError, ValueError) as error:
            if isinstance(error, UnrepresentableDeltaError):
                raise
            raise UnrepresentableDeltaError("Delta archive is malformed.") from error
        for directory in opaque_directories:
            prefix = f"{directory}/" if directory else ""
            visible = {path for path in upper if path.startswith(prefix)}
            deletions.update(
                path
                for path in current
                if path.startswith(prefix)
                and path not in visible
                and path != directory
                and not self.ignore_path(path)
            )
        return normalize_delta(
            ObservedDelta(
                entries=tuple(_observations(current, upper, deletions)),
                maximum_entries=self.maximum_entries,
                maximum_content_bytes=self.maximum_content_bytes,
            )
        )

    def _validate_archive(self, archive: Path) -> None:
        """Require one bounded regular archive without following a final symlink."""
        try:
            metadata = archive.lstat()
        except OSError as error:
            raise UnrepresentableDeltaError("Delta archive is unavailable.") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > self.maximum_archive_bytes
        ):
            raise UnrepresentableDeltaError("Delta archive is not a bounded regular file.")

    def _entry(
        self,
        staging_id: str,
        bundle: tarfile.TarFile,
        member: tarfile.TarInfo,
        previous: WorkspaceEntry | None,
        metacopy: bool,
        remaining_bytes: int,
    ) -> tuple[WorkspaceEntry, int]:
        """Translate one supported archive member and durably stage regular-file bytes."""
        mode = member.mode & 0o7777
        if member.isdir():
            return WorkspaceEntry(ObjectKind.DIRECTORY, None, None, mode, None), 0
        if member.issym():
            return WorkspaceEntry(ObjectKind.SYMLINK, None, None, mode, member.linkname), 0
        if member.islnk():
            raise UnrepresentableDeltaError("Delta changes hardlink topology.")
        if not member.isreg():
            raise UnrepresentableDeltaError("Delta contains an unsupported filesystem node type.")
        if metacopy:
            if previous is None or previous.kind is not ObjectKind.FILE:
                raise UnrepresentableDeltaError("Overlay metacopy has no regular lower object.")
            return replace(previous, identity=None, mode=mode), 0
        if member.size > remaining_bytes:
            raise UnrepresentableDeltaError("Delta exceeds its configured content quota.")
        source = bundle.extractfile(member)
        if source is None:
            raise UnrepresentableDeltaError("Delta regular file has no archive payload.")
        with source:
            content = self.content_store.stage(staging_id, source, maximum_bytes=remaining_bytes)
        return WorkspaceEntry(ObjectKind.FILE, None, content, mode, None), member.size


def _member_path(name: str) -> str | None:
    """Normalize one tar name into a workspace-relative virtual path."""
    while name.startswith("./"):
        name = name[2:]
    name = name.rstrip("/")
    if not name or name == ".":
        return None
    try:
        validate_relative_path(name)
    except ValueError as error:
        raise UnrepresentableDeltaError("Delta archive contains an unsafe path.") from error
    return name


def _whiteout_path(member: tarfile.TarInfo, path: str) -> str | None:
    """Return a deleted path, an empty opaque marker, or ``None`` for an ordinary member."""
    pure = PurePosixPath(path)
    if pure.name == ".wh..wh..opq":
        return ""
    if pure.name.startswith(".wh."):
        return str(pure.with_name(pure.name.removeprefix(".wh.")))
    if member.ischr() and member.devmajor == 0 and member.devminor == 0:
        return path
    return None


def _validate_xattrs(xattrs: dict[str, str]) -> None:
    """Reject ACL, security, and unrepresented extended-attribute changes."""
    known = (
        _OPAQUE_XATTRS
        | _WHITEOUT_XATTRS
        | _METACOPY_XATTRS
        | _REDIRECT_XATTRS
        | _INTERNAL_OVERLAY_XATTRS
    )
    for name in xattrs:
        if name in known:
            continue
        if "acl" in name.lower():
            raise UnrepresentableDeltaError("Delta changes ACL metadata.")
        if "xattr" in name.lower():
            if ".security." in name or ".system." in name:
                raise UnrepresentableDeltaError("Delta changes security xattrs.")
            raise UnrepresentableDeltaError("Delta changes unsupported extended attributes.")
        if name.startswith(("SCHILY.", "LIBARCHIVE.")):
            raise UnrepresentableDeltaError("Delta changes unsupported filesystem metadata.")


def _truthy_xattr(values: dict[str, str], names: frozenset[str]) -> bool:
    """Recognize the kernel and tar truth spellings used by overlay metadata."""
    return any(
        name in values and values[name] in {"", "y", "Y", "1", "true", "x"} for name in names
    )


def _xattr_value(values: dict[str, str], names: frozenset[str]) -> str | None:
    """Return the sole kernel metadata value from one supported spelling."""
    matches = [values[name] for name in names if name in values]
    if len(matches) > 1:
        raise UnrepresentableDeltaError("Overlay metadata has conflicting spellings.")
    return matches[0] if matches else None


def _current_state(
    base: SnapshotManifest,
    committed: tuple[CanonicalDelta, ...],
    affected: tuple[AffectedPathIdentity, ...],
) -> dict[str, WorkspaceEntry]:
    """Reconstruct one agent generation from its immutable base and canonical journal."""
    identities = {entry.path: entry.identity for entry in affected}
    state = {
        entry.path: WorkspaceEntry(
            entry.object_kind,
            identities.get(entry.path, entry.identity),
            entry.content,
            entry.mode,
            entry.symlink_target,
        )
        for entry in base.entries
    }
    for delta in committed:
        for entry in delta.entries:
            _apply_state_entry(state, entry, identities)
    return state


def _apply_state_entry(
    state: dict[str, WorkspaceEntry],
    entry: CanonicalDeltaEntry,
    identities: dict[str, HostObjectIdentity | None],
) -> None:
    """Apply one already-committed portable effect to an in-memory generation map."""
    destination = entry.destination_path
    if entry.effect is DeltaEffect.DELETE:
        _remove_subtree(state, destination)
        return
    if entry.effect is DeltaEffect.RENAME:
        source = entry.source_path or ""
        moved = {
            path: value
            for path, value in state.items()
            if path == source or path.startswith(source + "/")
        }
        _remove_subtree(state, source)
        _remove_subtree(state, destination)
        for path, value in moved.items():
            new_path = destination + path.removeprefix(source)
            state[new_path] = replace(value, identity=identities.get(new_path, value.identity))
        return
    previous = state.get(destination)
    if entry.effect is DeltaEffect.METADATA and previous is not None:
        state[destination] = replace(previous, mode=entry.mode or previous.mode)
        return
    if entry.object_kind is not ObjectKind.DIRECTORY:
        _remove_subtree(state, destination)
    state[destination] = WorkspaceEntry(
        entry.object_kind,
        identities.get(destination),
        entry.content,
        entry.mode or 0,
        entry.symlink_target,
    )


def _remove_subtree(state: dict[str, WorkspaceEntry], path: str) -> None:
    """Remove one object and every logical descendant from a generation map."""
    for candidate in tuple(state):
        if candidate == path or candidate.startswith(path + "/"):
            del state[candidate]


def _observations(
    current: dict[str, WorkspaceEntry],
    upper: dict[str, WorkspaceEntry],
    deletion_roots: set[str],
) -> list[ObservedDeltaEntry]:
    """Classify complete upper-layer effects and infer unambiguous leaf renames."""
    deleted_paths = {
        path
        for root in deletion_roots
        for path in current
        if path == root or path.startswith(root + "/")
    } - upper.keys()
    deletions = {
        path: ObservedDeltaEntry(
            effect=DeltaEffect.DELETE,
            destination_path=path,
            destination_identity=current[path].identity,
        )
        for path in deleted_paths
    }
    changes: dict[str, ObservedDeltaEntry] = {}
    for path, value in upper.items():
        previous = current.get(path)
        if previous is not None and _same_payload(previous, value) and previous.mode == value.mode:
            continue
        if previous is not None and _same_payload(previous, value) and previous.mode != value.mode:
            effect = DeltaEffect.METADATA
        else:
            effect = DeltaEffect.REPLACE if previous is not None else DeltaEffect.CREATE
        changes[path] = ObservedDeltaEntry(
            effect=effect,
            destination_path=path,
            object_kind=ObservedObjectKind(value.kind),
            destination_identity=previous.identity if previous else None,
            content=value.content,
            mode=value.mode,
            symlink_target=value.symlink_target,
            metadata=MetadataPreservation(basic_mode=value.mode),
        )
    candidates: dict[tuple[object, ...], list[str]] = {}
    for path in deletions:
        value = current[path]
        if value.kind is not ObjectKind.DIRECTORY:
            candidates.setdefault(_fingerprint(value), []).append(path)
    for path, change in tuple(changes.items()):
        if change.effect not in {DeltaEffect.CREATE, DeltaEffect.REPLACE}:
            continue
        value = upper[path]
        sources = candidates.get(_fingerprint(value), [])
        if len(sources) != 1:
            continue
        source = sources[0]
        changes[path] = change.model_copy(
            update={
                "effect": DeltaEffect.RENAME,
                "source_path": source,
                "source_identity": current[source].identity,
                "content": None,
            }
        )
        del deletions[source]
        del candidates[_fingerprint(value)]
    return [*deletions.values(), *changes.values()]


def _same_payload(left: WorkspaceEntry, right: WorkspaceEntry) -> bool:
    """Compare object kind and persistent payload without identity or mode."""
    return (
        left.kind,
        left.content.digest if left.content else None,
        left.symlink_target,
    ) == (
        right.kind,
        right.content.digest if right.content else None,
        right.symlink_target,
    )


def _fingerprint(entry: WorkspaceEntry) -> tuple[object, ...]:
    """Return the exact portable payload used for conservative rename inference."""
    return (
        entry.kind,
        entry.content.digest if entry.content else None,
        entry.mode,
        entry.symlink_target,
    )


def _validate_identifier(value: str, kind: str) -> None:
    """Reject an opaque identifier that could become guest path or option syntax."""
    if fullmatch(_IDENTIFIER_PATTERN, value) is None:
        raise ValueError(f"Workspace {kind} identity is invalid.")


def _transaction_key(transaction_id: str) -> str:
    """Map an arbitrary durable transaction identity to one guest-safe name."""
    return sha256_digest(transaction_id)


def _require_success(result: InfrastructureProcessResult, message: str) -> None:
    """Require complete successful output from one trusted management operation."""
    if result.exit_code or result.stdout_truncated or result.stderr_truncated:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{message} exit={result.exit_code!r}; stderr={detail!r}")
