"""Verify bound AppArmor authority, protected names and literal policy rendering."""

from dataclasses import replace
from pathlib import Path

import pytest

from loop.execution.sandbox.contracts import SandboxRequest
from loop.execution.sandbox.linux import policy as linux_policy
from loop.execution.sandbox.linux.constants import NATIVE_SCRATCH_PATH
from loop.execution.sandbox.linux.policy import _literal, payload_policy, payload_profile


@pytest.fixture
def bound(monkeypatch, tmp_path):
    """Bind a disposable workspace and private scratch through the public request contract."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(
        "loop.execution.sandbox.contracts.tempfile.gettempdir", lambda: str(tmp_path)
    )

    def create(**changes):
        """Create a fresh approved authority with controlled environment and deadline."""
        values = {
            "source": "printf ok",
            "cwd": workspace,
            "workspace": workspace,
            "read_roots": (),
            "write_roots": (workspace,),
            "network": False,
            "environment": {"TMPDIR": str(scratch), "PATH": "/usr/bin:/bin"},
            "policy_version": "policy-test",
            "deadline": 1000,
            "workspace_id": "workspace-test",
        }
        values.update(changes)
        return SandboxRequest.create(**values)

    return create, workspace, scratch


def render(request, scratch, **changes):
    """Render a complete policy with a fixed trusted cache-lock binding."""
    return payload_policy(
        request, scratch, **{"cache_lock": Path("/run/cache.lock"), "profile": "payload", **changes}
    )


def test_payload_policy_retains_native_composition_denials(bound):
    """Rendering never grants capabilities, namespace mutation or cache-control access."""
    create, _, scratch = bound
    policy = render(create(), scratch)
    assert policy.startswith("abi <abi/4.0>,\nprofile payload flags=(attach_disconnected) {\n")
    assert policy.endswith("}\n")
    for rule in (
        "audit deny capability,",
        "audit deny userns,",
        "deny mount,",
        "deny umount,",
        "deny /sys/kernel/security/{,**} rwkl,",
        'deny "/\\x72\\x75\\x6e/\\x63\\x61\\x63\\x68\\x65\\x2e\\x6c\\x6f\\x63\\x6b" rwkl,',
    ):
        assert rule in policy
    assert "change_profile" not in policy
    assert "allow file" not in policy


def test_payload_policy_read_only_changes_only_authorized_workspace_writes(bound):
    """Read-only requests retain private scratch writes without granting workspace writes."""
    create, _, scratch = bound
    writable = render(create(), scratch)
    readonly = render(create(write_roots=()), scratch)
    writable_rules = [line for line in writable.splitlines() if "allow rwlkmix" in line]
    readonly_rules = [line for line in readonly.splitlines() if "allow rwlkmix" in line]
    assert len(writable_rules) == 4
    assert len(readonly_rules) == 2
    assert sum('/{,**/}*",' in line for line in writable_rules) == 2
    assert sum('/{,**/}*",' in line for line in readonly_rules) == 1
    assert writable.count("allow rmix") == readonly.count("allow rmix")


def test_payload_policy_deduplicates_only_exact_authorized_roots(bound):
    """Repeated roots do not widen authority or change the resulting policy bytes."""
    create, workspace, scratch = bound
    assert render(create(), scratch) == render(
        create(read_roots=(workspace,), write_roots=(workspace, workspace)), scratch
    )


def test_payload_profile_reuses_only_identical_policy_across_fresh_scratch(bound):
    """Fresh host scratch and command source reuse one exact stable-path policy profile."""
    create, workspace, scratch = bound
    second_scratch = scratch.parent / "second-scratch"
    second_scratch.mkdir(mode=0o700)
    cache = Path("/run/cache.lock")
    runtime_roots = (Path("/usr"),)
    first = create(source="printf first")
    second = create(
        source="printf second",
        environment={"TMPDIR": str(second_scratch), "PATH": "/usr/bin:/bin"},
    )

    first_profile = payload_profile(first, scratch, cache, runtime_roots)
    second_profile = payload_profile(second, second_scratch, cache, runtime_roots)

    assert first_profile == second_profile
    assert payload_policy(
        first, scratch, cache, first_profile, runtime_roots=runtime_roots
    ) == payload_policy(second, second_scratch, cache, second_profile, runtime_roots=runtime_roots)

    extra = workspace.parent / "extra-read"
    extra.mkdir()
    changed = create(read_roots=(extra,))
    assert payload_profile(changed, scratch, cache, runtime_roots) != first_profile


def test_payload_policy_grants_stable_namespace_scratch_not_host_spelling(bound):
    """A profile cannot read or write a command's unique host scratch path directly."""
    create, _, scratch = bound
    policy = render(create(), scratch)

    assert str(scratch) not in policy
    assert _literal("/.loop-private-scratch") in policy


def test_payload_profile_reuses_virtual_alias_authority_without_host_names(bound):
    """Fresh symlink spellings reuse policy only when their actual target grant is unchanged."""
    create, workspace, scratch = bound
    requests = []
    for name in ("first", "second"):
        parent = workspace.parent / name
        parent.mkdir(mode=0o700)
        alias = parent / "r0"
        alias.symlink_to(workspace)
        requests.append(create(aliases=(("$WORKSPACE_ROOT", alias, workspace),)))
    cache = Path("/run/cache.lock")
    profiles = [payload_profile(request, scratch, cache, (Path("/usr"),)) for request in requests]
    assert profiles[0] == profiles[1]
    for request in requests:
        policy = render(request, scratch)
        assert _literal(str(request.aliases[0][1])) not in policy
        assert _literal("/.loop-private-paths/r0") in policy
    extra = workspace.parent / "other-target"
    extra.mkdir()
    parent = workspace.parent / "third"
    parent.mkdir(mode=0o700)
    alias = parent / "r0"
    alias.symlink_to(extra)
    changed = create(read_roots=(extra,), aliases=(("$WORKSPACE_ROOT", alias, extra),))
    assert payload_profile(changed, scratch, cache, (Path("/usr"),)) != profiles[0]


def test_payload_profile_reuses_selected_workspace_with_fresh_private_backing(bound):
    """Selected snapshots share stable policy while retaining distinct host directories."""
    create, workspace, _ = bound
    profiles = []
    for name in ("first-search", "second-search"):
        scratch = workspace.parent / name
        scratch.mkdir(mode=0o700)
        selected = scratch / "selected"
        selected.mkdir()
        request = create(
            cwd=selected, workspace=selected, write_roots=(), environment={"TMPDIR": str(scratch)}
        )
        profiles.append(payload_profile(request, scratch, Path("/run/cache.lock"), (Path("/usr"),)))
        assert _literal(str(selected)) not in render(request, scratch)
        assert _literal("/.loop-private-scratch/selected") in render(request, scratch)
    assert profiles[0] == profiles[1]


def test_payload_profile_refuses_a_noncanonical_policy_template(bound, monkeypatch):
    """Profile naming fails closed if the renderer omits its exact placeholder header."""
    create, _, scratch = bound
    monkeypatch.setattr(linux_policy, "payload_policy", lambda *_args, **_kwargs: "profile other")

    with pytest.raises(ValueError, match="one exact profile name"):
        payload_profile(create(), scratch, Path("/run/cache.lock"), (Path("/usr"),))


def test_payload_policy_preserves_manual_reads_and_writes_separately(bound):
    """An additional approved read root cannot obtain a native write allowance."""
    create, workspace, scratch = bound
    external = workspace.parent / "external"
    external.mkdir()
    policy = render(create(read_roots=(external,)), scratch)
    assert policy.count("allow rmix") == 6
    assert policy.count("allow rwlkmix") == 4


def test_payload_policy_allows_only_the_bound_external_executable_file(bound):
    """A discovered PATH command gets exact-file execution without a directory read grant."""
    create, workspace, scratch = bound
    tools = workspace.parent / "tools"
    tools.mkdir()
    executable = tools / "uv"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    request = create(
        environment={"TMPDIR": str(scratch), "PATH": str(tools)},
        executable_identities=(("uv", executable, executable, details.st_dev, details.st_ino),),
    )

    policy = render(request, scratch)

    assert f'allow rmix "{_literal(str(executable))}",' in policy
    assert f'allow rmix "{_literal(str(tools))}/{{,**}}",' not in policy
    assert f'allow rwlkmix "{_literal(str(tools))}/{{,**}}",' not in policy


def test_payload_policy_does_not_duplicate_executable_inside_approved_root(bound):
    """A workspace executable uses its existing root grant without another file rule."""
    create, workspace, scratch = bound
    tools = workspace / "bin"
    tools.mkdir()
    executable = tools / "uv"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    request = create(
        environment={"TMPDIR": str(scratch), "PATH": str(tools)},
        executable_identities=(("uv", executable, executable, details.st_dev, details.st_ino),),
    )

    policy = render(request, scratch)

    assert f'allow rmix "{_literal(str(executable))}",' not in policy
    assert f'allow rmix "{_literal(str(workspace))}/{{,**}}",' in policy


def test_payload_policy_rejects_private_executable_identities(bound):
    """A private component cannot become executable authority through a forged identity."""
    create, workspace, scratch = bound
    tools = workspace.parent / ".ssh" / "bin"
    tools.mkdir(parents=True)
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    request = create(
        environment={"TMPDIR": str(scratch), "PATH": str(tools)},
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
    )

    with pytest.raises(ValueError, match="cannot be an executable"):
        render(request, scratch)


def test_payload_policy_preserves_bound_aliases_without_writable_parent_grants(bound):
    """Approved target and alias reads never grant the containing mapping directory writes."""
    create, workspace, scratch = bound
    target = workspace.parent / "target"
    target.mkdir()
    mappings = workspace.parent / "mappings"
    mappings.mkdir(mode=0o700)
    alias = mappings / "mapped"
    alias.symlink_to(target, target_is_directory=True)
    read_alias = mappings / "read"
    read_alias.symlink_to(target, target_is_directory=True)
    policy = render(
        create(
            read_roots=(target,),
            read_aliases=(read_alias,),
            aliases=(("/virtual", alias, target),),
        ),
        scratch,
    )
    assert policy.count("allow rmix") == 7
    assert policy.count("allow rwlkmix") == 4
    # Resolver/trust inputs add exact read-only files, never a containing-tree grant.
    assert policy.count('allow r "') == 11


@pytest.mark.parametrize("name", ['RULES [*]? "é"\nline.md', "} profile injected {", "@{HOME}.md"])
def test_payload_policy_encodes_instruction_names_without_policy_injection(bound, name):
    """Operator, newline, Unicode and variable syntax in filenames stays literal."""
    create, _, scratch = bound
    policy = render(create(protected_instruction_names=(name,)), scratch)
    assert name not in policy
    assert policy.count("profile ") == 1
    assert policy.count('" wkl,') == 11


@pytest.mark.parametrize("git_mode", ["none", "create", "write"])
def test_payload_policy_git_exception_preserves_nested_git_and_other_controls(bound, git_mode):
    """Only an explicit top-level Git grant changes that name's write protection."""
    create, workspace, scratch = bound
    git = workspace / ".git"
    if git_mode == "write":
        git.mkdir()
    request = create(
        git_create=git_mode == "create",
        write_roots=(workspace, git) if git_mode == "write" else (workspace,),
    )
    policy = render(request, scratch)
    prefix = "*/{,**/}" if git_mode != "none" else "{,**/}"
    assert prefix + "\\x2e\\x67\\x69\\x74" in policy
    assert "{,**/}\\x2e\\x6c\\x6f\\x6f\\x70" in policy
    assert "{,**/}\\x41\\x47\\x45\\x4e\\x54\\x53\\x2e\\x6d\\x64" in policy


def test_payload_policy_denies_private_names_in_every_granted_root(bound):
    """Runtime and request grants protect private data without blocking new scratch fixtures."""
    create, workspace, scratch = bound
    external = workspace.parent / "external"
    external.mkdir()
    runtime_roots = (Path("/usr"), Path("/usr/lib64"), Path("/lib64"))
    policy = render(
        create(read_roots=(external,), write_roots=(workspace, external)),
        scratch,
        runtime_roots=runtime_roots,
    )
    for root in (*runtime_roots, workspace, external):
        prefix = _literal(str(root))
        assert f'deny "{prefix}/{{,**/}}.[sS][sS][hH]" rwklmx,' in policy
        assert f'deny "{prefix}/{{,**/}}.[cC][oO][nN][fF][iI][gG]/{{,**}}" rwklmx,' in policy
    assert 'deny "/{,**/}.[sS][sS][hH]" rwklmx,' not in policy
    assert (
        f'deny "{_literal("/.loop-private-scratch")}/{{,**/}}.[sS][sS][hH]" rwklmx,' not in policy
    )
    assert _literal(str(scratch)) not in policy


def test_payload_policy_root_workspace_preserves_protected_name_scope(bound):
    """Root spelling cannot introduce a double separator that skips protected basenames."""
    create, _, scratch = bound
    policy = render(create(workspace=Path("/"), write_roots=()), scratch)
    assert 'deny "/{,**/}\\x41\\x47\\x45\\x4e\\x54\\x53\\x2e\\x6d\\x64" wkl,' in policy
    assert '"//{,**/}' not in policy


@pytest.mark.parametrize("root_kind", ["read", "write"])
def test_payload_policy_refuses_private_roots_before_rendering(bound, root_kind):
    """A protected private path cannot become an authorized policy subtree."""
    create, workspace, scratch = bound
    private = workspace / ".AWS"
    private.mkdir()
    request = create(**{root_kind + "_roots": (private,)})
    with pytest.raises(ValueError, match="Protected private data"):
        render(request, scratch)


@pytest.mark.parametrize("profile", ["", "bootstrap", "payload/child", "payload {", "é", "a" * 129])
def test_payload_policy_refuses_invalid_profile_names(bound, profile):
    """Names cannot inject attachment/profile syntax or reuse bootstrap identity."""
    create, _, scratch = bound
    with pytest.raises(ValueError, match="Invalid payload profile name"):
        render(create(), scratch, profile=profile)


def test_payload_policy_accepts_boundary_length_profile_name(bound):
    """The longest supported simple profile name renders without truncation."""
    create, _, scratch = bound
    profile = "p" * 128
    assert f"profile {profile} " in render(create(), scratch, profile=profile)


def test_payload_policy_refuses_replaced_authority(bound):
    """A directory replaced after binding prevents policy generation."""
    create, workspace, scratch = bound
    request = create()
    workspace.rename(workspace.with_name("old"))
    workspace.mkdir()
    with pytest.raises(ValueError, match="Policy request authority changed"):
        render(request, scratch)


def test_payload_policy_refuses_unbound_scratch(bound):
    """A caller cannot substitute an unapproved writable scratch directory."""
    create, _, scratch = bound
    with pytest.raises(ValueError, match="scratch is not bound"):
        render(create(), scratch.with_name("other"))


@pytest.mark.parametrize("lock", [Path("relative"), Path("/run/../cache.lock"), Path("/")])
def test_payload_policy_refuses_ambiguous_lock_names(bound, lock):
    """Cache-control denial requires an absolute unambiguous file spelling."""
    create, _, scratch = bound
    with pytest.raises(ValueError, match="cache lock must be absolute"):
        render(create(), scratch, cache_lock=lock)


def test_payload_policy_refuses_nul_in_trusted_paths(bound):
    """A malformed literal cannot terminate the parser's path token."""
    create, _, scratch = bound
    with pytest.raises(ValueError, match="cannot contain NUL"):
        render(create(), scratch, cache_lock=Path("/run/bad\x00lock"))


def test_payload_policy_ignores_source_environment_and_network_in_filename_authority(bound):
    """Opaque source, unrelated variables and network choices cannot inject file rules."""
    create, _, scratch = bound
    request = create()
    changed = replace(request, source="} profile injected {", network=True)
    assert render(request, scratch) == render(changed, scratch)


def test_payload_policy_accepts_bound_runtime_aliases_without_duplicates(bound):
    """The production runtime layout can include lib64 while retaining literal encoding."""
    create, _, scratch = bound
    policy = payload_policy(
        create(),
        scratch,
        Path("/cache"),
        "payload",
        runtime_roots=(Path("/usr"), Path("/lib64"), Path("/lib64")),
    )
    assert policy.count("allow rmix") == 4
    assert '"/\\x6c\\x69\\x62\\x36\\x34/{,**}"' in policy


@pytest.mark.parametrize(
    "roots", [(), (Path("/"),), (Path("relative"),), (Path("/usr/../home"),), (Path("/home/.ssh"),)]
)
def test_payload_policy_rejects_invalid_runtime_roots(bound, roots):
    """Malformed, whole-host and protected-private runtime grants cannot be rendered."""
    create, _, scratch = bound
    with pytest.raises(ValueError, match="Invalid native runtime roots"):
        payload_policy(create(), scratch, Path("/cache"), "payload", runtime_roots=roots)


@pytest.mark.parametrize(
    "directory,generation",
    [
        (Path("/"), "test"),
        (Path("/bad path"), "test"),
        (Path("/safe/../bad"), "test"),
        (Path("/safe"), "bad:name"),
        (Path("/safe"), "test"),
    ],
)
def test_runtime_policy_sources_bind_only_fixed_trusted_setup(directory, generation):
    """Trusted setup source has literal attachment and payload stacking, without loading policy."""
    from loop.execution.sandbox.linux.policy import runtime_policy_sources

    if directory != Path("/safe") or generation != "test":
        with pytest.raises(ValueError, match="trusted runtime"):
            runtime_policy_sources(directory, generation)
    else:
        policies, attachment = runtime_policy_sources(directory, generation)
        assert set(policies) == {""}
        assert set(policies[""]) == {"bootstrap"}
        assert b"change_profile -> &payload_*," in policies[""]["bootstrap"]
        assert b'profile loopgen_test "/safe/helper" flags=(attach_disconnected) {' in attachment
        assert b"change_profile -> &:test:bootstrap," in attachment


def test_session_scratch_policy_guards_future_private_names(bound, tmp_path):
    """Session scratch retains private-name denials even before those entries exist."""
    create, _, scratch = bound
    alias = tmp_path / "alias"
    alias.symlink_to(scratch, target_is_directory=True)
    value = create(aliases=(("$TMPDIR", alias, scratch),))
    policy = render(value, scratch)

    assert _literal(str(NATIVE_SCRATCH_PATH)) + "/{,**/}" in policy


def test_fresh_session_data_policy_keeps_private_guards_outside_scratch(bound):
    """Only the bound fresh scratch loses basename denials; workspace private rules remain."""
    create, workspace, scratch = bound
    value = create(scratch_root=scratch, scratch_data=True)
    policy = render(value, scratch)
    assert 'deny "' + _literal(str(NATIVE_SCRATCH_PATH)) + "/{,**/}" not in policy
    assert _literal(str(workspace)) + "/{,**/}[lL][iI][bB][rR][aA][rR][yY]" in policy
