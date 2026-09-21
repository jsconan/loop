"""Test automatic composition of the managed macOS sandbox substrate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import loop.execution.sandbox.macos.backend as backend_module
from loop import constants
from loop.execution.sandbox.macos import (
    LimaLifecycleState,
    MacosSandboxBackend,
    PreparedMacosRuntime,
    load_macos_runtime_candidate,
)


class _Lease:
    """Record release of one fake installed-runtime lease."""

    def __init__(self) -> None:
        self.closed = False
        self.expiry: float | None = None

    def heartbeat(self, expiry: float) -> None:
        """Record one renewed runtime lease expiry."""
        self.expiry = expiry

    def close(self) -> None:
        """Record lease release."""
        self.closed = True


class _Runtime:
    """Expose one fake Lima executable and lease."""

    def __init__(self) -> None:
        self.lease = _Lease()
        self.executable_value = SimpleNamespace()
        self.artifact_set_digest = "a" * 64

    def executable(self, artifact_id: str, relative_path: str) -> object:
        """Return the fake executable only for the manifest-owned Lima path."""
        assert (artifact_id, relative_path) == ("lima", "bin/limactl")
        return self.executable_value


class _Bootstrapper:
    """Record installation and activation requests."""

    def __init__(self) -> None:
        self.runtime = _Runtime()
        self.requirement = None
        self.activated = None

    def ensure(self, requirement: object, progress=lambda _: None) -> _Runtime:
        """Return one installed fake runtime."""
        del progress
        self.requirement = requirement
        return self.runtime

    def activate(self, runtime: object) -> None:
        """Record activation after complete attestation."""
        self.activated = runtime


class _Instance:
    """Model the public managed-instance state decisions."""

    state: LimaLifecycleState
    calls: list[str]
    recover_attest: bool
    obsolete: bool

    def __init__(
        self,
        state: LimaLifecycleState,
        recover_attest: bool = False,
        obsolete: bool = False,
    ) -> None:
        self.state = state
        self.calls: list[str] = []
        self.recover_attest = recover_attest
        self.obsolete = obsolete

    @property
    def state(self) -> LimaLifecycleState:
        """Return state unless a prior-release journal requires explicit recovery."""
        if self.obsolete:
            raise backend_module.LimaConfigurationError("obsolete release")
        return self._state

    @state.setter
    def state(self, value: LimaLifecycleState) -> None:
        self._state = value

    def reset(self) -> None:
        """Recover incomplete owned state to absent."""
        self.calls.append("reset")
        self.state = LimaLifecycleState.ABSENT

    def reset_obsolete(self) -> None:
        """Recover a validated prior-release instance to absent."""
        self.calls.append("reset_obsolete")
        self.obsolete = False
        self.state = LimaLifecycleState.ABSENT

    def create(self) -> None:
        """Create stopped state."""
        self.calls.append("create")
        self.state = LimaLifecycleState.STOPPED

    def start(self) -> None:
        """Start running state."""
        self.calls.append("start")
        self.state = LimaLifecycleState.RUNNING

    def attest_running(self) -> None:
        """Attest existing running state."""
        self.calls.append("attest")
        if self.recover_attest:
            self.recover_attest = False
            self.state = LimaLifecycleState.ABSENT
            raise backend_module.LimaConfigurationError("stale stopped instance")

    def stop(self, *, force: bool = False) -> None:
        """Record a stop operation."""
        del force
        self.calls.append("stop")

    def delete(self) -> None:
        """Record a delete operation."""
        self.calls.append("delete")


def _candidate():
    """Load the checked-in runtime manifest."""
    return load_macos_runtime_candidate(
        Path(__file__).parents[4] / "scripts/runtime-candidates/macos-arm64-v1.json"
    )


def _patch_composition(
    monkeypatch: pytest.MonkeyPatch,
    state: LimaLifecycleState,
    *,
    fail_image: bool = False,
    readiness_failures: int = 0,
    recover_attest: bool = False,
    obsolete: bool = False,
) -> tuple[_Instance, object]:
    """Replace native collaborators while retaining orchestration decisions."""
    instance = _Instance(state, recover_attest, obsolete)
    evidence = SimpleNamespace(entitlement_keys=("virtualization",), code_directory_hash=None)
    endpoint = SimpleNamespace()
    remaining_readiness_failures = [readiness_failures]
    monkeypatch.setattr(backend_module, "LimaClient", lambda process, root: SimpleNamespace())
    monkeypatch.setattr(
        backend_module.MacosRequirements,
        "from_candidate",
        lambda *args: SimpleNamespace(verify=lambda executable: evidence),
    )
    monkeypatch.setattr(
        backend_module.LimaInstanceContext,
        "create",
        lambda *args: SimpleNamespace(),
    )
    monkeypatch.setattr(backend_module, "ManagedLimaInstance", lambda *args: instance)

    class _Plane:
        def __init__(self, *args: object) -> None:
            pass

        def attest_runtime(self, candidate: object) -> object:
            if remaining_readiness_failures[0]:
                remaining_readiness_failures[0] -= 1
                raise backend_module.MacosRuntimeNotReadyError("not ready")
            return endpoint

    class _Builder:
        def __init__(
            self,
            plane: object,
            state_root: Path,
        ) -> None:
            del state_root
            assert isinstance(plane, _Plane)

        def prepare(self) -> object:
            if fail_image:
                raise RuntimeError("image failed")
            identity = SimpleNamespace(manifest_digest="sha256:" + "b" * 64)
            image = SimpleNamespace(
                reference="loop.local/sandbox@sha256:" + "a" * 64, identity=identity
            )
            return SimpleNamespace(image=image)

    monkeypatch.setattr(backend_module, "MacosControlPlane", _Plane)
    monkeypatch.setattr(backend_module, "MacosSandboxImageBuilder", _Builder)
    return instance, endpoint


@pytest.mark.parametrize(
    ("state", "calls"),
    [
        (LimaLifecycleState.ABSENT, ["create", "start"]),
        (LimaLifecycleState.STOPPED, ["start"]),
        (LimaLifecycleState.RUNNING, ["attest"]),
        (LimaLifecycleState.STARTING, ["reset", "create", "start"]),
    ],
)
def test_prepare_installs_recovers_reuses_and_attests_before_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: LimaLifecycleState,
    calls: list[str],
) -> None:
    """Preparation handles every durable state and activates only complete substrate evidence."""
    instance, endpoint = _patch_composition(monkeypatch, state)
    bootstrapper = _Bootstrapper()
    progress: list[str] = []
    backend = MacosSandboxBackend(
        _candidate(),
        bootstrapper,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace-id",
        minimum_free_bytes=1,
        progress=progress.append,
    )
    prepared = backend.prepare()
    assert instance.calls == calls
    assert prepared.endpoint is endpoint
    assert prepared.snapshot_store == backend.snapshot_store
    assert prepared.image_reference.startswith("loop.local/sandbox@sha256:")
    assert bootstrapper.requirement.capabilities == frozenset({"sandbox-management"})
    assert bootstrapper.activated is bootstrapper.runtime
    assert bootstrapper.runtime.lease.expiry is not None
    assert bootstrapper.runtime.lease.expiry > (
        backend_module.time.time() + constants.RUNTIME_PREPARATION_LEASE_SECONDS - 1
    )
    expected_progress = {
        LimaLifecycleState.ABSENT: [
            "Creating isolated command sandbox…",
            "Starting isolated command sandbox…",
            "Building sandbox command image…",
            "Isolated command sandbox is ready.",
        ],
        LimaLifecycleState.STOPPED: [
            "Starting isolated command sandbox…",
            "Building sandbox command image…",
            "Isolated command sandbox is ready.",
        ],
        LimaLifecycleState.RUNNING: [
            "Building sandbox command image…",
            "Isolated command sandbox is ready.",
        ],
        LimaLifecycleState.STARTING: [
            "Repairing isolated command sandbox…",
            "Creating isolated command sandbox…",
            "Starting isolated command sandbox…",
            "Building sandbox command image…",
            "Isolated command sandbox is ready.",
        ],
    }
    assert progress == expected_progress[state]

    prepared.renew(60)
    assert bootstrapper.runtime.lease.expiry is not None
    with pytest.raises(ValueError, match="positive"):
        prepared.renew(0)
    prepared.stop()
    prepared.close()
    assert instance.calls[-1] == "stop"
    assert bootstrapper.runtime.lease.closed


def test_prepare_retries_only_bounded_rootless_runtime_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient namespace-identity race converges without weakening other attestation."""
    _patch_composition(
        monkeypatch,
        LimaLifecycleState.ABSENT,
        readiness_failures=1,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(backend_module.time, "sleep", sleeps.append)
    prepared = MacosSandboxBackend(
        _candidate(),
        _Bootstrapper(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace",
        minimum_free_bytes=1,
    ).prepare()

    assert prepared.endpoint is not None
    assert sleeps == [0.05]


def test_prepare_recovers_stale_running_instance_before_returning_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exactly owned stale running journal is rebuilt within the original request."""
    instance, _ = _patch_composition(
        monkeypatch,
        LimaLifecycleState.RUNNING,
        recover_attest=True,
    )

    prepared = MacosSandboxBackend(
        _candidate(),
        _Bootstrapper(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace",
        minimum_free_bytes=1,
    ).prepare()

    assert prepared.endpoint is not None
    assert instance.calls == ["attest", "create", "start"]


def test_prepare_replaces_a_validated_prior_release_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest update removes its obsolete owned VM before creating the new instance."""
    instance, _ = _patch_composition(
        monkeypatch,
        LimaLifecycleState.RUNNING,
        obsolete=True,
    )

    prepared = MacosSandboxBackend(
        _candidate(),
        _Bootstrapper(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace",
        minimum_free_bytes=1,
    ).prepare()

    assert prepared.endpoint is not None
    assert instance.calls == ["reset_obsolete", "create", "start"]


def test_prepare_bounds_recovery_when_runtime_state_remains_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated stale-runtime evidence fails after one exact owned recovery attempt."""
    instance, _ = _patch_composition(monkeypatch, LimaLifecycleState.RUNNING, recover_attest=True)

    def remain_stale() -> None:
        instance.calls.append("start")
        instance.state = LimaLifecycleState.ABSENT
        raise backend_module.LimaConfigurationError("still stale")

    instance.start = remain_stale  # type: ignore[method-assign]
    backend = MacosSandboxBackend(
        _candidate(),
        _Bootstrapper(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace",
        minimum_free_bytes=1,
    )

    with pytest.raises(backend_module.LimaConfigurationError, match="still stale"):
        backend.prepare()

    assert instance.calls == ["attest", "create", "start", "delete"]


def test_prepare_fails_closed_when_rootless_runtime_never_stabilizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent namespace-identity disagreement exhausts the bounded readiness window."""
    instance, _ = _patch_composition(
        monkeypatch,
        LimaLifecycleState.ABSENT,
        readiness_failures=2,
    )
    bootstrapper = _Bootstrapper()
    times = iter((0.0, 0.0, backend_module._RUNTIME_READINESS_SECONDS + 1.0))
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: next(times))
    progress: list[str] = []

    def fail_delete() -> None:
        instance.calls.append("delete")
        raise RuntimeError("cleanup failed")

    instance.delete = fail_delete  # type: ignore[method-assign]

    with pytest.raises(backend_module.MacosRuntimeNotReadyError, match="not ready"):
        MacosSandboxBackend(
            _candidate(),
            bootstrapper,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            tmp_path / "application",
            "workspace",
            minimum_free_bytes=1,
            progress=progress.append,
        ).prepare()

    assert instance.calls[-1] == "delete"
    assert bootstrapper.runtime.lease.closed
    assert progress.count("Waiting for isolated command sandbox services…") == 1


def test_prepared_delete_releases_lease_even_when_cleanup_fails() -> None:
    """VM deletion cannot strand the caller's immutable-runtime lease."""
    runtime = _Runtime()

    class _FailingInstance(_Instance):
        def delete(self) -> None:
            raise RuntimeError("cleanup failed")

    prepared = PreparedMacosRuntime(
        runtime,  # type: ignore[arg-type]
        _FailingInstance(LimaLifecycleState.RUNNING),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(image=SimpleNamespace(reference="loop.local/sandbox@sha256:" + "a" * 64)),  # type: ignore[arg-type]
        Path("/private/snapshots"),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        prepared.delete()
    assert runtime.lease.closed


def test_prepared_runtime_resolves_only_the_single_image_manifest_digest() -> None:
    """Prepared runtimes expose only the ready single image to authorized leases."""
    image = SimpleNamespace(
        reference="loop.local/sandbox@sha256:" + "a" * 64,
        identity=SimpleNamespace(manifest_digest="sha256:" + "b" * 64),
    )
    prepared = PreparedMacosRuntime(
        _Runtime(),  # type: ignore[arg-type]
        _Instance(LimaLifecycleState.RUNNING),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(image=image),  # type: ignore[arg-type]
        Path("/private/snapshots"),
    )

    assert prepared.image_reference == image.reference
    assert prepared.image_for_digest("sha256:" + "b" * 64) is image
    with pytest.raises(ValueError, match="prepared sandbox image"):
        prepared.image_for_digest("sha256:" + "e" * 64)


def test_image_failure_preserves_a_fully_attested_new_instance_and_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Image failure retains complete VM substrate so a retry does not recreate it."""
    instance, _ = _patch_composition(monkeypatch, LimaLifecycleState.ABSENT, fail_image=True)
    bootstrapper = _Bootstrapper()
    backend = MacosSandboxBackend(
        _candidate(),
        bootstrapper,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace-id",
        minimum_free_bytes=1,
    )
    with pytest.raises(RuntimeError, match="image failed"):
        backend.prepare()
    assert instance.calls == ["create", "start"]
    assert bootstrapper.runtime.lease.closed


def test_prepare_failure_preserves_preexisting_warm_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed warm revalidation releases its lease without deleting shared owned state."""
    instance, _ = _patch_composition(monkeypatch, LimaLifecycleState.RUNNING, fail_image=True)
    bootstrapper = _Bootstrapper()
    backend = MacosSandboxBackend(
        _candidate(),
        bootstrapper,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace-id",
        minimum_free_bytes=1,
    )
    with pytest.raises(RuntimeError, match="image failed"):
        backend.prepare()
    assert instance.calls == ["attest"]
    assert bootstrapper.runtime.lease.closed


@pytest.mark.parametrize(
    ("delete", "state", "calls"),
    [
        (False, LimaLifecycleState.RUNNING, ["stop"]),
        (False, LimaLifecycleState.STOPPED, []),
        (False, LimaLifecycleState.STARTING, ["reset"]),
        (True, LimaLifecycleState.RUNNING, ["reset"]),
    ],
)
def test_backend_manages_existing_sandbox_without_starting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delete: bool,
    state: LimaLifecycleState,
    calls: list[str],
) -> None:
    """Management stops or resets journal-owned state without VM preparation."""
    instance, _ = _patch_composition(monkeypatch, state)
    bootstrapper = _Bootstrapper()
    application = tmp_path / "application"
    backend = MacosSandboxBackend(
        _candidate(),
        bootstrapper,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        application,
        "workspace-id",
        minimum_free_bytes=1,
    )
    binding = next(application.glob(".loop-*.workspace"))
    journal = application / "lima" / f"{binding.name[1:-10]}.loop.json"
    journal.parent.mkdir()
    journal.write_text("{}", encoding="utf-8")

    assert backend.manage(delete=delete) is True

    assert instance.calls == calls
    assert bootstrapper.runtime.lease.closed


def test_backend_management_reports_missing_sandbox_without_installing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent journal produces a cheap no-op cleanup result."""
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    bootstrapper = _Bootstrapper()
    backend = MacosSandboxBackend(
        _candidate(),
        bootstrapper,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace-id",
        minimum_free_bytes=1,
    )

    assert backend.manage(delete=True) is False

    assert bootstrapper.requirement is None


@pytest.mark.parametrize(
    ("state", "calls"),
    [
        (LimaLifecycleState.RUNNING, ["attest", "attest"]),
        (LimaLifecycleState.STOPPED, []),
    ],
)
def test_backend_status_and_inventory_validate_owned_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: LimaLifecycleState,
    calls: list[str],
) -> None:
    """Status and inventory validate journal state and attest a claimed-running VM."""
    instance, _ = _patch_composition(monkeypatch, state)
    bootstrapper = _Bootstrapper()
    application = tmp_path / "application"
    backend = MacosSandboxBackend(
        _candidate(),
        bootstrapper,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        application,
        "workspace-id",
        minimum_free_bytes=1,
    )
    binding = next(application.glob(".loop-*.workspace"))
    name = binding.name[1:-10]
    journal = application / "lima" / f"{name}.loop.json"
    journal.parent.mkdir()
    journal.write_text("{}", encoding="utf-8")

    assert backend.status() == (name, state)
    assert backend.list_sandboxes() == ((name, state),)

    assert instance.calls == calls
    assert bootstrapper.runtime.lease.closed


def test_backend_status_and_inventory_report_absence_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status and inventory report no ownership without installing a runtime."""
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    bootstrapper = _Bootstrapper()
    backend = MacosSandboxBackend(
        _candidate(),
        bootstrapper,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path / "application",
        "workspace-id",
        minimum_free_bytes=1,
    )

    assert backend.status() is None
    assert backend.list_sandboxes() == ()
    assert bootstrapper.requirement is None


def test_backend_rejects_missing_identity_and_linked_application_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workspace and private-root ambiguity fail before runtime acquisition."""
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    with pytest.raises(ValueError, match="workspace identity"):
        MacosSandboxBackend(
            _candidate(),
            object(),
            object(),
            tmp_path / "state",
            "",  # type: ignore[arg-type]
        )
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="private application state"):
        MacosSandboxBackend(
            _candidate(),
            object(),
            object(),
            link,
            "workspace",  # type: ignore[arg-type]
        )
    control_link = tmp_path / "control-link"
    control_link.symlink_to(target)
    with pytest.raises(ValueError, match="private short control state"):
        MacosSandboxBackend(
            _candidate(),
            object(),
            object(),
            tmp_path / "application",
            "workspace",  # type: ignore[arg-type]
            state_root=control_link,
        )


def test_backend_reuses_only_an_exact_durable_workspace_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated composition accepts its full identity and rejects altered binding state."""
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    application = tmp_path / "application"
    MacosSandboxBackend(
        _candidate(),
        object(),
        object(),
        application,
        "workspace",  # type: ignore[arg-type]
    )
    MacosSandboxBackend(
        _candidate(),
        object(),
        object(),
        application,
        "workspace",  # type: ignore[arg-type]
    )
    binding = next(application.glob("*.workspace"))
    binding.write_text("0" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="workspace binding"):
        MacosSandboxBackend(
            _candidate(),
            object(),
            object(),
            application,
            "workspace",  # type: ignore[arg-type]
        )
    binding.unlink()
    target = application / "binding-target"
    target.write_text("0" * 64, encoding="ascii")
    binding.symlink_to(target)
    with pytest.raises(ValueError, match="workspace binding"):
        MacosSandboxBackend(
            _candidate(),
            object(),
            object(),
            application,
            "workspace",  # type: ignore[arg-type]
        )


def test_backend_names_are_unique_across_private_control_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same workspace cannot collide on Lima identity across private installations."""
    names: list[str] = []
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    original_create = backend_module.LimaInstanceContext.create

    def capture_name(*args: object) -> object:
        configuration = args[2]
        names.append(configuration.instance_name)
        return original_create(*args)

    monkeypatch.setattr(backend_module.LimaInstanceContext, "create", capture_name)
    for suffix in ("first", "second"):
        MacosSandboxBackend(
            _candidate(),
            _Bootstrapper(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            tmp_path / suffix / "application",
            "workspace-id",
            state_root=tmp_path / suffix / "control",
            minimum_free_bytes=1,
        ).prepare()

    assert len(names) == 2
    assert names[0] != names[1]


def test_backend_wraps_workspace_binding_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed durable identity write aborts composition before runtime acquisition."""
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    original = Path.open

    def fail_binding(path: Path, *args: object, **kwargs: object):
        if path.name.endswith(".workspace"):
            raise OSError("storage failed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_binding)
    with pytest.raises(ValueError, match="could not be persisted"):
        MacosSandboxBackend(
            _candidate(),
            object(),
            object(),
            tmp_path / "application",
            "workspace",  # type: ignore[arg-type]
        )


def test_backend_detects_missing_and_replaced_snapshot_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshot-store removal, replacement, and permission widening fail before reuse."""
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    backend = MacosSandboxBackend(
        _candidate(),
        object(),
        object(),
        tmp_path / "missing" / "application",
        "workspace",  # type: ignore[arg-type]
    )
    backend.snapshot_store.rmdir()
    with pytest.raises(ValueError, match="identity changed"):
        _ = backend.snapshot_store

    backend = MacosSandboxBackend(
        _candidate(),
        object(),
        object(),
        tmp_path / "replaced" / "application",
        "workspace",  # type: ignore[arg-type]
    )
    store = backend.snapshot_store
    store.rename(store.with_name("old"))
    store.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="identity changed"):
        _ = backend.snapshot_store

    backend = MacosSandboxBackend(
        _candidate(),
        object(),
        object(),
        tmp_path / "permissions" / "application",
        "workspace",  # type: ignore[arg-type]
    )
    backend.snapshot_store.chmod(0o755)
    with pytest.raises(ValueError, match="identity changed"):
        _ = backend.snapshot_store


def test_backend_keeps_short_control_state_separate_and_identity_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lima control sockets use a distinct short root whose replacement fails closed."""
    _patch_composition(monkeypatch, LimaLifecycleState.ABSENT)
    control = tmp_path / "control"
    backend = MacosSandboxBackend(
        _candidate(),
        object(),
        object(),
        tmp_path / "durable" / "application",
        "workspace",  # type: ignore[arg-type]
        state_root=control,
    )

    assert backend._state_root == control
    control.chmod(0o755)
    with pytest.raises(ValueError, match="identity changed"):
        _ = backend.snapshot_store
