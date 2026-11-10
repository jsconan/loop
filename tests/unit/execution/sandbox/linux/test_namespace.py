"""Verify fixed namespace setup, exact grants and owned descriptor-filter lifetime."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import namespace as linux_namespace


@pytest.fixture
def preparation(monkeypatch):
    """Isolate native binding/filter/file collaborators with explicit resource lifetimes."""
    monkeypatch.setattr(linux_namespace, "runtime_read_roots", lambda: ())
    monkeypatch.setattr(linux_namespace, "runtime_read_files", lambda **_kwargs: ())
    monkeypatch.setattr(linux_namespace.os, "O_PATH", 1, raising=False)
    request = Mock(network=False, deadline=20, cwd=Path("/work/sub"), path_roots=())
    mounts = Mock(return_value=nullcontext((("--bind-fd", "4", "/work"), (4,))))
    monkeypatch.setattr(linux_namespace, "mount_arguments", mounts)
    monkeypatch.setattr(linux_namespace, "descriptor_filter", Mock(return_value=b"filter"))
    monkeypatch.setattr(linux_namespace.time, "monotonic", lambda: 10)
    stream = Mock()
    stream.write.return_value = 6
    stream.fileno.return_value = 7
    context = Mock()
    context.__enter__ = Mock(return_value=stream)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(linux_namespace.tempfile, "TemporaryFile", Mock(return_value=context))
    return request, mounts, stream, context


@pytest.mark.parametrize("network", [False, True])
def test_namespace_fixed_boundary_and_original_grants(preparation, network):
    """Only an explicit network grant preserves networking; all fixed isolation remains."""
    request, mounts, stream, context = preparation
    request.network = network
    with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as result:
        arguments, descriptors = result
        assert arguments == (
            "--unshare-user",
            "--uid",
            "0",
            "--gid",
            "0",
            "--unshare-pid",
            "--unshare-ipc",
            "--new-session",
            "--die-with-parent",
            "--cap-drop",
            "ALL",
            *(("--unshare-net",) if not network else ()),
            "--bind-fd",
            "4",
            "/work",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            "/work/sub",
            "--seccomp",
            "7",
        )
        assert descriptors == (4, 7)
        context.__exit__.assert_not_called()
    context.__exit__.assert_called_once()
    mounts.assert_called_once_with(
        request,
        Path("/scratch"),
        empty_mask=9,
        skip_path_roots=(),
        skip_executable_roots=(),
        runtime_roots=(),
    )
    stream.write.assert_called_once_with(b"filter")
    stream.seek.assert_called_once_with(0)


@pytest.mark.parametrize("failure", [False, True])
def test_namespace_owns_distinct_repeated_mount_sources(preparation, monkeypatch, failure):
    """Repeated bind sources receive distinct descriptors and are closed on every exit."""
    request, mounts, _, _ = preparation
    mounts.return_value = nullcontext(
        (("--bind-fd", "4", "/a", "--ro-bind-fd", "4", "/b", "--bind-fd", "4", "/c"), (4,))
    )
    duplicate = Mock(side_effect=[20, OSError("descriptor limit")] if failure else [20, 21])
    close = Mock()
    monkeypatch.setattr(linux_namespace.os, "dup", duplicate)
    monkeypatch.setattr(linux_namespace.os, "close", close)
    if failure:
        with (
            pytest.raises(OSError, match="descriptor limit"),
            linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9),
        ):
            pytest.fail("Incomplete descriptor preparation must not launch.")
        close.assert_called_once_with(20)
        return
    with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as (
        arguments,
        descriptors,
    ):
        assert arguments[arguments.index("--bind-fd") :][:9] == (
            "--bind-fd",
            "4",
            "/a",
            "--ro-bind-fd",
            "20",
            "/b",
            "--bind-fd",
            "21",
            "/c",
        )
        assert descriptors == (4, 7, 20, 21)
        close.assert_not_called()
    assert close.call_args_list == [((21,),), ((20,),)]


@pytest.mark.parametrize("failure", ["paths", "filter", "write", "deadline", "launch"])
def test_namespace_failure_closes_owned_resources(preparation, monkeypatch, failure):
    """Unsafe preparation, expiry and caller failure never leak the owned filter context."""
    request, mounts, stream, context = preparation
    if failure == "paths":
        mounts.side_effect = ValueError("unsafe paths")
    elif failure == "filter":
        linux_namespace.descriptor_filter.side_effect = OSError("filter refused")
    elif failure == "write":
        stream.write.return_value = 0
    elif failure == "deadline":
        monkeypatch.setattr(linux_namespace.time, "monotonic", lambda: 20)
    category = (
        ValueError if failure == "paths" else TimeoutError if failure == "deadline" else OSError
    )
    with (
        pytest.raises(
            category, match="unsafe paths|filter refused|incomplete|deadline expired|launch refused"
        ),
        linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9),
    ):
        raise OSError("launch refused")
    assert context.__exit__.call_count == 1


@pytest.mark.parametrize("failure", [None, "flag", "identity", "alias", "path"])
def test_runtime_sources_are_pinned_readonly_and_closed(preparation, monkeypatch, failure):
    """Authenticate and pin runtime aliases; reject replacement and close every source."""
    from types import SimpleNamespace

    request, _mounts, _stream, _context = preparation
    canonical = Mock()
    details = SimpleNamespace(st_dev=1, st_ino=2, st_mode=0o40755, st_uid=0, st_gid=0)
    canonical.stat.return_value = details
    roots = Mock(return_value=(Path("/bin"),))
    protected = Mock(return_value=canonical)
    descriptor = 1_000_000
    closed = Mock()
    original_open = linux_namespace.os.open
    original_fstat = linux_namespace.os.fstat
    original_close = linux_namespace.os.close

    def open_runtime(path, flags, *args, **kwargs):
        """Fake only the selected runtime source and open other paths normally."""
        return descriptor if path is canonical else original_open(path, flags, *args, **kwargs)

    bound = details

    def stat_runtime(fd):
        """Return selected runtime metadata without affecting other descriptors."""
        return bound if fd == descriptor else original_fstat(fd)

    def close_runtime(fd):
        """Record only the selected runtime close and preserve other closes."""
        if fd == descriptor:
            closed(fd)
        else:
            original_close(fd)

    monkeypatch.setattr(linux_namespace, "runtime_read_roots", roots)
    monkeypatch.setattr(linux_namespace, "protected_path", protected)
    monkeypatch.setattr(linux_namespace.os, "open", open_runtime)
    monkeypatch.setattr(linux_namespace.os, "close", close_runtime)
    monkeypatch.setattr(linux_namespace.os, "fstat", stat_runtime)
    if failure == "flag":
        monkeypatch.delattr(linux_namespace.os, "O_PATH")
    elif failure == "identity":
        bound = SimpleNamespace(st_dev=1, st_ino=3, st_mode=0o40755, st_uid=0, st_gid=0)
    elif failure == "alias":
        protected.side_effect = [canonical, Path("/changed")]
    elif failure == "path":
        canonical.stat.side_effect = [details, object()]
    if failure:
        error = OSError if failure == "flag" else ValueError
        message = (
            "Runtime mount preparation requires Linux O_PATH."
            if failure == "flag"
            else "Native runtime directory changed during binding."
        )
        with (
            pytest.raises(error, match=message),
            linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9),
        ):
            pytest.fail("Unsafe runtime was accepted")
        if failure != "flag":
            closed.assert_called_once_with(descriptor)
    else:
        with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as (
            arguments,
            descriptors,
        ):
            assert arguments[
                arguments.index("--ro-bind-fd") : arguments.index("--ro-bind-fd") + 6
            ] == (
                "--ro-bind-fd",
                str(descriptor),
                str(canonical),
                "--symlink",
                str(canonical),
                "/bin",
            )
            assert descriptors == (descriptor, 4, 7)
            closed.assert_not_called()
        closed.assert_called_once_with(descriptor)


def test_runtime_path_alias_is_recreated_when_request_mount_is_skipped(
    preparation, monkeypatch, tmp_path
):
    """A runtime PATH alias remains present when its duplicate request mount is skipped."""
    request, mounts, _, _ = preparation
    runtime = tmp_path / "runtime"
    canonical = runtime / "bin"
    canonical.mkdir(parents=True)
    spelling = tmp_path / "bin"
    spelling.symlink_to(canonical, target_is_directory=True)
    request.path_roots = ((spelling, canonical),)
    request.workspace = tmp_path / "workspace"
    request.read_roots = ()
    request.write_roots = ()
    request.read_aliases = ()
    details = canonical.stat()
    monkeypatch.setattr(linux_namespace, "runtime_read_roots", lambda: (spelling,))
    monkeypatch.setattr(
        linux_namespace,
        "protected_path",
        lambda path, **kwargs: canonical,
    )
    monkeypatch.setattr(linux_namespace.os, "open", Mock(return_value=12))
    monkeypatch.setattr(linux_namespace.os, "close", Mock())
    monkeypatch.setattr(linux_namespace.os, "fstat", lambda fd: details)
    with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as (
        arguments,
        _,
    ):
        assert arguments.count("--symlink") == 1
        index = arguments.index("--symlink")
        assert arguments[index : index + 3] == ("--symlink", str(canonical), str(spelling))
    mounts.assert_called_once_with(
        request,
        Path("/scratch"),
        empty_mask=9,
        skip_path_roots=(spelling,),
        skip_executable_roots=(canonical,),
        runtime_roots=(spelling, canonical),
    )


def test_namespace_recreates_the_workspace_python_runtime_alias(preparation, monkeypatch, tmp_path):
    """A bound Python runtime remains resolvable through the venv's configured home alias."""
    request, _mounts, _, _ = preparation
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "home/runtime/cpython-3.14.1"
    alias = tmp_path / "home/runtime/cpython-3.14"
    request.workspace = workspace
    request.read_roots = (runtime,)
    request.automatic_tool_reads = ()
    request.write_roots = (workspace,)
    request.path_roots = ()
    request.read_aliases = ()
    request.executable_identities = ()
    monkeypatch.setattr(
        linux_namespace,
        "python_runtime_mounts",
        lambda *_args, **_kwargs: ((runtime, alias),),
    )

    with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as result:
        arguments, _ = result

    assert arguments[arguments.index("--symlink") : arguments.index("--symlink") + 3] == (
        "--symlink",
        str(runtime),
        str(alias),
    )


def test_namespace_rejects_a_python_runtime_alias_overlapping_the_workspace(
    preparation, monkeypatch, tmp_path
):
    """A runtime alias cannot shadow the already approved writable workspace mount."""
    request, _, _, _ = preparation
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    request.workspace = workspace
    request.read_roots = (runtime,)
    request.automatic_tool_reads = (runtime,)
    request.write_roots = (workspace,)
    request.path_roots = ()
    request.read_aliases = ()
    request.executable_identities = ()
    monkeypatch.setattr(
        linux_namespace,
        "python_runtime_mounts",
        lambda *_args, **_kwargs: ((runtime, workspace),),
    )

    with (
        pytest.raises(ValueError, match="overlaps another mount"),
        linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9),
    ):
        pytest.fail("An overlapping Python runtime alias was accepted")


def test_namespace_keeps_a_canonical_python_runtime_path(preparation, monkeypatch, tmp_path):
    """A venv with a canonical Python home needs no synthetic path alias."""
    request, _, _, _ = preparation
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    request.workspace = workspace
    request.read_roots = (runtime,)
    request.automatic_tool_reads = (runtime,)
    request.write_roots = (workspace,)
    request.path_roots = ()
    request.read_aliases = ()
    monkeypatch.setattr(
        linux_namespace,
        "python_runtime_mounts",
        lambda *_args, **_kwargs: ((runtime, runtime),),
    )

    with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as result:
        arguments, _ = result

    assert "--symlink" not in arguments


def test_namespace_rejects_a_python_runtime_missing_approved_read_authority(
    preparation, monkeypatch, tmp_path
):
    """A derived venv alias cannot add a read root omitted from the approved request."""
    request, _, _, _ = preparation
    runtime = tmp_path / "runtime"
    request.workspace = tmp_path / "workspace"
    request.read_roots = ()
    request.automatic_tool_reads = ()
    request.write_roots = ()
    request.path_roots = ()
    request.read_aliases = ()
    monkeypatch.setattr(
        linux_namespace,
        "python_runtime_mounts",
        lambda *_args, **_kwargs: ((runtime, runtime),),
    )

    with (
        pytest.raises(ValueError, match="authority changed"),
        linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9),
    ):
        pytest.fail("An unapproved Python runtime was accepted")


def test_runtime_path_root_without_alias_adds_no_symlink(preparation, monkeypatch, tmp_path):
    """A canonical runtime root is mounted directly without a synthetic symlink."""
    request, _, _, _ = preparation
    canonical = tmp_path / "runtime"
    canonical.mkdir()
    details = canonical.stat()
    monkeypatch.setattr(linux_namespace, "runtime_read_roots", lambda: (canonical,))
    monkeypatch.setattr(linux_namespace, "protected_path", lambda *args, **kwargs: canonical)
    monkeypatch.setattr(linux_namespace.os, "open", Mock(return_value=12))
    monkeypatch.setattr(linux_namespace.os, "close", Mock())
    monkeypatch.setattr(linux_namespace.os, "fstat", lambda fd: details)

    with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as (
        arguments,
        _,
    ):
        assert "--symlink" not in arguments
        index = arguments.index("--ro-bind-fd")
        assert arguments[index : index + 3] == ("--ro-bind-fd", "12", str(canonical))


def test_runtime_path_root_recreates_live_protected_symlink_chain(tmp_path, monkeypatch):
    """A PATH root under runtime mounts retains a protected external symlink target."""
    runtime = tmp_path / "runtime"
    canonical = runtime / "tool-version/bin"
    canonical.mkdir(parents=True)
    alias_directory = runtime / "tool"
    alias_parent = tmp_path / "aliases"
    alias_parent.mkdir()
    alias_target = alias_parent / "tool"
    alias_target.symlink_to(canonical.parent, target_is_directory=True)
    alias_directory.symlink_to(alias_target, target_is_directory=True)
    spelling = alias_directory / "bin"
    request = Mock(
        path_roots=((spelling, canonical),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )
    monkeypatch.setattr(
        linux_namespace,
        "protected_path",
        lambda path, *, directory: Path(path).resolve(strict=directory),
    )

    covered, arguments = linux_namespace._runtime_path_alias_mounts(
        request,
        ((runtime, runtime),),
        (tmp_path / "etc/hosts",),
        tmp_path / "scratch",
    )

    assert covered == (spelling,)
    assert arguments == (
        "--dir",
        str(alias_parent),
        "--symlink",
        str(canonical.parent),
        str(alias_target),
    )


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "multihop",
        "approved",
        "untrusted",
        "stale",
        "private",
        "other-target",
        "wrong-authority",
        "cycle",
        "regular",
        "within-runtime",
        "outside",
        "retarget",
        "canonical-outside",
    ],
)
def test_selected_system_executable_alias_is_exact_and_bound(
    tmp_path, monkeypatch, case, logical_alias_private_names
):
    """Only a protected selected executable gets its exact missing alternatives link."""
    runtime = tmp_path / "runtime"
    binary = runtime / "bin"
    binary.mkdir(parents=True)
    alternatives = tmp_path / "alternatives"
    alternatives.mkdir()
    target = binary / "mawk"
    target.write_text("tool", encoding="utf-8")
    intermediate = alternatives / "awk"
    intermediate.symlink_to(target)
    spelling = binary / "awk"
    spelling.symlink_to(intermediate)
    if case == "multihop":
        spelling.unlink()
        middle = binary / "middle"
        middle.symlink_to(intermediate)
        spelling.symlink_to(middle)
    if case == "other-target":
        spelling.unlink()
        other = tmp_path / "other"
        other.symlink_to(target)
        spelling.symlink_to(other)
    if case == "regular":
        spelling.unlink()
        spelling = target
    if case == "within-runtime":
        spelling.unlink()
        spelling.symlink_to(target)
    if case == "cycle":
        spelling.unlink()
        spelling.symlink_to(spelling)
    details = target.stat()
    request = Mock(
        executable_identities=(("awk", spelling, target, details.st_dev, details.st_ino),),
        workspace=tmp_path / "workspace",
        read_roots=(alternatives,) if case == "approved" else (),
        write_roots=(),
        path_roots=((binary, binary),),
        read_aliases=(),
    )
    monkeypatch.setattr(linux_namespace, "_SYSTEM_ALTERNATIVES", alternatives)

    def protected(path, *, directory=False, allow_root_owned_hardlinks=False):
        """Model protected host authority with an explicit refusal for the intermediate."""
        if case == "untrusted" and path == intermediate:
            raise ValueError("untrusted ancestry")
        if case == "canonical-outside" and path == intermediate:
            return tmp_path / "outside-resolved"
        if case == "wrong-authority" and path == spelling:
            return tmp_path / "wrong"
        if case == "cycle" and path == spelling:
            return target
        if case == "retarget" and path == spelling:
            replacement = binary / "other-tool"
            replacement.write_text("other", encoding="utf-8")
            spelling.unlink()
            spelling.symlink_to(replacement)
            return target
        return path.resolve(strict=True)

    monkeypatch.setattr(linux_namespace, "protected_path", protected)
    if case == "private":
        monkeypatch.setattr(linux_namespace, "is_protected_external_read", lambda _: True)
    if case == "stale":
        replacement = binary / "replacement"
        replacement.write_text("replacement", encoding="utf-8")
        replacement.replace(target)
    if case in {
        "untrusted",
        "stale",
        "private",
        "wrong-authority",
        "cycle",
        "other-target",
        "retarget",
        "canonical-outside",
    }:
        with pytest.raises(ValueError):
            linux_namespace._runtime_executable_aliases(request, ((runtime, runtime),))
    else:
        roots = (
            ((tmp_path / "elsewhere", tmp_path / "elsewhere"),)
            if case == "outside"
            else ((runtime, runtime),)
        )
        aliases = linux_namespace._runtime_executable_aliases(request, roots)
        assert aliases == (
            ()
            if case in {"approved", "regular", "within-runtime", "outside"}
            else ((intermediate, target),)
        )
        if case in {"valid", "multihop"}:
            _, arguments = linux_namespace._runtime_path_alias_mounts(
                request, ((runtime, runtime),), (), tmp_path / "scratch", aliases
            )
            assert arguments == (
                "--dir",
                str(alternatives),
                "--symlink",
                str(target),
                str(intermediate),
            )


def test_system_executable_alias_ignores_missing_typed_identities():
    """Absent typed executable identities cannot create namespace aliases."""
    assert linux_namespace._runtime_executable_aliases(Mock(executable_identities=None), ()) == ()


def test_system_executable_alias_walks_symlinked_ancestors(
    tmp_path, monkeypatch, logical_alias_private_names
):
    """Intermediate directory links cannot hide an unmounted alternatives hop."""
    runtime = tmp_path / "runtime"
    binary = runtime / "bin"
    binary.mkdir(parents=True)
    package = runtime / "package"
    package.mkdir()
    target = package / "tool"
    target.write_text("tool", encoding="utf-8")
    alternatives = tmp_path / "alternatives"
    alternatives.mkdir()
    intermediate = alternatives / "package"
    intermediate.symlink_to(package, target_is_directory=True)
    alias_directory = runtime / "selected"
    alias_directory.symlink_to(intermediate, target_is_directory=True)
    spelling = binary / "tool"
    spelling.symlink_to(alias_directory / "tool")
    details = target.stat()
    request = Mock(
        executable_identities=(("tool", spelling, target, details.st_dev, details.st_ino),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
    )
    monkeypatch.setattr(linux_namespace, "_SYSTEM_ALTERNATIVES", alternatives)
    monkeypatch.setattr(
        linux_namespace, "protected_path", lambda path, **_kwargs: path.resolve(strict=True)
    )
    assert linux_namespace._runtime_executable_aliases(request, ((runtime, runtime),)) == (
        (intermediate, package),
    )


def test_system_alias_refuses_parent_traversal_across_unresolved_link(
    tmp_path, monkeypatch, logical_alias_private_names
):
    """Lexical parent collapse cannot conceal a missing intermediate alternatives link."""
    runtime = tmp_path / "runtime"
    binary = runtime / "bin"
    binary.mkdir(parents=True)
    library = runtime / "lib"
    library.mkdir()
    package = runtime / "container/pkg"
    package.mkdir(parents=True)
    target = package / "tool"
    target.write_text("tool", encoding="utf-8")
    child = runtime / "container/child"
    child.mkdir()
    alternatives = tmp_path / "alternatives"
    alternatives.mkdir()
    (alternatives / "selected").symlink_to(child, target_is_directory=True)
    (library / "redirect").symlink_to(alternatives / "selected", target_is_directory=True)
    spelling = binary / "tool"
    spelling.symlink_to(library / "redirect/../pkg/tool")
    details = target.stat()
    request = Mock(
        executable_identities=(("tool", spelling, target, details.st_dev, details.st_ino),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
    )
    monkeypatch.setattr(linux_namespace, "_SYSTEM_ALTERNATIVES", alternatives)
    monkeypatch.setattr(
        linux_namespace, "protected_path", lambda path, **_kwargs: path.resolve(strict=True)
    )
    assert spelling.resolve(strict=True) == target
    with pytest.raises(ValueError, match="unresolved symlink parent"):
        linux_namespace._runtime_executable_aliases(request, ((runtime, runtime),))


def test_system_alias_normalizes_safe_relative_parent(tmp_path):
    """A relative parent component without an intervening link keeps its intended target."""
    directory = tmp_path / "bin"
    directory.mkdir()
    assert linux_namespace._normalize_alias_target(directory / "tool", Path("../lib/tool")) == (
        tmp_path / "lib/tool"
    )


@pytest.mark.parametrize("hops", [40, 41])
def test_system_executable_alias_walk_matches_host_symlink_limit(tmp_path, monkeypatch, hops):
    """Forty protected links are accepted and a longer chain is refused."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "target"
    target.write_text("tool", encoding="utf-8")
    current = target
    for index in range(hops):
        link = runtime / f"link-{index}"
        link.symlink_to(current)
        current = link
    details = target.stat()
    request = Mock(
        executable_identities=(("tool", current, target, details.st_dev, details.st_ino),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
    )
    monkeypatch.setattr(linux_namespace, "protected_path", lambda *_args, **_kwargs: target)
    if hops == 40:
        assert linux_namespace._runtime_executable_aliases(request, ((runtime, runtime),)) == ()
    else:
        with pytest.raises(ValueError, match="exceeds its bound"):
            linux_namespace._runtime_executable_aliases(request, ((runtime, runtime),))


def test_runtime_path_root_keeps_request_mount_for_external_symlink_target(tmp_path):
    """A PATH symlink through an unmounted external directory is not runtime-covered."""
    runtime = tmp_path / "runtime"
    canonical = runtime / "tool-version/bin"
    canonical.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "bin").symlink_to(canonical, target_is_directory=True)
    spelling = runtime / "tool/bin"
    spelling.parent.symlink_to(external, target_is_directory=True)
    request = Mock(
        path_roots=((spelling, canonical),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )

    covered, arguments = linux_namespace._runtime_path_alias_mounts(
        request,
        ((runtime, runtime),),
        (),
        tmp_path / "scratch",
    )

    assert covered == ()
    assert arguments == ()


def test_runtime_path_roots_without_runtime_coverage_are_unchanged(tmp_path):
    """PATH directories outside authenticated runtime mounts remain request-owned."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    request = Mock(
        path_roots=((external, external),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )

    covered, arguments = linux_namespace._runtime_path_alias_mounts(
        request, ((runtime, runtime),), (), tmp_path / "scratch"
    )

    assert covered == ()
    assert arguments == ()


def test_runtime_path_root_uses_in_tree_symlink_without_extra_alias(tmp_path, monkeypatch):
    """A symlink already resolved within a runtime tree needs no namespace alias."""
    runtime = tmp_path / "runtime"
    canonical = runtime / "tool-version/bin"
    canonical.mkdir(parents=True)
    alias = runtime / "tool"
    alias.symlink_to(canonical.parent, target_is_directory=True)
    spelling = alias / "bin"
    request = Mock(
        path_roots=((spelling, canonical),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )
    monkeypatch.setattr(
        linux_namespace,
        "protected_path",
        lambda path, *, directory: Path(path).resolve(strict=directory),
    )

    covered, arguments = linux_namespace._runtime_path_alias_mounts(
        request, ((runtime, runtime),), (), tmp_path / "scratch"
    )

    assert covered == (spelling,)
    assert arguments == ()


def test_runtime_path_alias_rejects_symlink_drift(tmp_path, monkeypatch):
    """A link changed between readlink and resolution cannot become runtime authority."""
    runtime = tmp_path / "runtime"
    canonical = runtime / "tool-version/bin"
    canonical.mkdir(parents=True)
    other = runtime / "other-version"
    other.mkdir()
    alias_target = tmp_path / "aliases/tool"
    alias_target.parent.mkdir()
    alias_target.symlink_to(canonical.parent, target_is_directory=True)
    spelling = runtime / "tool/bin"
    spelling.parent.symlink_to(alias_target, target_is_directory=True)
    request = Mock(
        path_roots=((spelling, canonical),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )
    original_readlink = Path.readlink

    def changed_readlink(path):
        """Return a different destination than the link currently resolves to."""
        return other if path == spelling.parent else original_readlink(path)

    monkeypatch.setattr(Path, "readlink", changed_readlink)

    with pytest.raises(ValueError, match="PATH link changed"):
        linux_namespace._runtime_path_alias_mounts(
            request, ((runtime, runtime),), (), tmp_path / "scratch"
        )


def test_runtime_path_alias_rejects_unprotected_target(tmp_path, monkeypatch):
    """A discovered external target must match its authenticated canonical directory."""
    runtime = tmp_path / "runtime"
    canonical = runtime / "tool-version/bin"
    canonical.mkdir(parents=True)
    other = runtime / "other-version"
    other.mkdir()
    alias_target = tmp_path / "aliases/tool"
    alias_target.parent.mkdir()
    alias_target.symlink_to(canonical.parent, target_is_directory=True)
    spelling = runtime / "tool/bin"
    spelling.parent.symlink_to(alias_target, target_is_directory=True)
    request = Mock(
        path_roots=((spelling, canonical),),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )
    monkeypatch.setattr(linux_namespace, "protected_path", lambda *_args, **_kwargs: other)

    with pytest.raises(ValueError, match="target authority differs"):
        linux_namespace._runtime_path_alias_mounts(
            request, ((runtime, runtime),), (), tmp_path / "scratch"
        )


def test_runtime_path_alias_rejects_request_mount_overlap(tmp_path, monkeypatch):
    """A synthetic alias cannot be placed inside another approved request mount."""
    runtime = tmp_path / "runtime"
    canonical = runtime / "tool-version/bin"
    canonical.mkdir(parents=True)
    alias_target = tmp_path / "aliases/tool"
    alias_target.parent.mkdir()
    alias_target.symlink_to(canonical.parent, target_is_directory=True)
    spelling = runtime / "tool/bin"
    spelling.parent.symlink_to(alias_target, target_is_directory=True)
    request = Mock(
        path_roots=((spelling, canonical),),
        workspace=alias_target.parent,
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )
    monkeypatch.setattr(
        linux_namespace,
        "protected_path",
        lambda path, *, directory: Path(path).resolve(strict=directory),
    )

    with pytest.raises(ValueError, match="overlaps another mount"):
        linux_namespace._runtime_path_alias_mounts(
            request, ((runtime, runtime),), (), tmp_path / "scratch"
        )


def test_runtime_path_alias_rejects_conflicting_targets(tmp_path, monkeypatch):
    """A changing system alias cannot map one namespace path to two runtime directories."""
    runtime = tmp_path / "runtime"
    first = runtime / "tool-one/bin"
    second = runtime / "tool-two/bin"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    alias_target = tmp_path / "aliases/tool"
    alias_target.parent.mkdir()
    alias_target.symlink_to(first.parent, target_is_directory=True)
    first_spelling = runtime / "first/bin"
    second_spelling = runtime / "second/bin"
    first_spelling.parent.symlink_to(alias_target, target_is_directory=True)
    second_spelling.parent.symlink_to(alias_target, target_is_directory=True)
    request = Mock(
        path_roots=((first_spelling, first), (second_spelling, second)),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )
    original_resolve = Path.resolve
    active_target = None
    target_resolution = 0
    targets = (first.parent, second.parent)
    link_paths = (first_spelling.parent, second_spelling.parent)

    def changing_resolve(path, strict=False, **kwargs):
        """Model a link target changing between two individually revalidated discoveries."""
        nonlocal active_target, target_resolution
        if path in link_paths:
            active_target = targets[target_resolution]
            target_resolution += 1
            return active_target
        if path == alias_target:
            return active_target
        return original_resolve(path, strict=strict, **kwargs)

    monkeypatch.setattr(Path, "resolve", changing_resolve)
    monkeypatch.setattr(linux_namespace, "protected_path", lambda *_args, **_kwargs: active_target)

    with pytest.raises(ValueError, match="conflicting targets"):
        linux_namespace._runtime_path_alias_mounts(
            request, ((runtime, runtime),), (), tmp_path / "scratch"
        )


def test_runtime_path_aliases_reject_nested_destinations(tmp_path, monkeypatch):
    """Overlapping symlink destinations are refused before namespace construction."""
    runtime = tmp_path / "runtime"
    first = runtime / "tool-one/bin"
    second = runtime / "tool-two/bin"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    alias_parent = tmp_path / "aliases"
    alias_parent.mkdir()
    (alias_parent / "tool").symlink_to(first.parent, target_is_directory=True)
    (first.parent / "nested").symlink_to(second.parent, target_is_directory=True)
    first_spelling = runtime / "first/bin"
    second_spelling = runtime / "second/bin"
    first_spelling.parent.symlink_to(alias_parent / "tool", target_is_directory=True)
    second_spelling.parent.symlink_to(alias_parent / "tool/nested", target_is_directory=True)
    request = Mock(
        path_roots=((first_spelling, first), (second_spelling, second)),
        workspace=tmp_path / "workspace",
        read_roots=(),
        write_roots=(),
        read_aliases=(),
    )
    monkeypatch.setattr(
        linux_namespace,
        "protected_path",
        lambda path, *, directory: Path(path).resolve(strict=directory),
    )

    with pytest.raises(ValueError, match="aliases overlap"):
        linux_namespace._runtime_path_alias_mounts(
            request, ((runtime, runtime),), (), tmp_path / "scratch"
        )


def test_namespace_pins_system_data_at_exact_readonly_destinations(preparation, monkeypatch):
    """Resolver and TLS data use owned verified descriptors without directory grants."""
    request, _, _, _ = preparation
    authority = Mock()
    authority.open_verified.return_value = nullcontext(81)
    monkeypatch.setattr(
        linux_namespace,
        "runtime_read_files",
        lambda **_kwargs: ((Path("/etc/resolv.conf"), authority),),
    )
    with linux_namespace.namespace_arguments(request, Path("/scratch"), empty_mask=9) as result:
        arguments, descriptors = result
        assert arguments[arguments.index("--ro-bind-fd") :][:3] == (
            "--ro-bind-fd",
            "81",
            "/etc/resolv.conf",
        )
        assert descriptors == (81, 4, 7)
    authority.open_verified.assert_called_once_with(deadline=request.deadline)
