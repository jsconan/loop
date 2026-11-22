"""Tests for typed operation and policy model behavior."""

import pytest
from pydantic import ValidationError

from loop import (
    Action,
    Decision,
    FileTarget,
    NetworkTarget,
    Operation,
    PermissionConfiguration,
    PolicyLimits,
    ProcessTarget,
    SessionTarget,
    UserPermissionConfiguration,
)
from loop import constants as app_constants
from loop.permissions import (
    HostCommandRule,
    NetworkAccessTarget,
    NetworkIngressTarget,
    PermissionPreset,
    PolicyLimitClassifications,
    SandboxPathAccess,
)


@pytest.mark.parametrize(
    "path, level, error",
    [
        ("/tmp/fixture", "ask", "read-only"),
        ("/", "allow", "Sandbox paths require an exact path"),
        ("//tmp", "deny", "Sandbox paths require an exact path"),
    ],
)
def test_sandbox_path_policy_requires_exact_access_level(path, level, error):
    """The persisted path map cannot express a broad allow or an ambiguous decision."""
    with pytest.raises(ValueError, match=error):
        PermissionConfiguration(sandbox_paths={path: level})


def test_sandbox_path_policy_preserves_legacy_read_grants():
    """Original scalar allow entries load as read grants and serialize canonically."""
    config = PermissionConfiguration(version=6, sandbox_paths={"/tools": "allow", "/a": "deny"})
    assert config.sandbox_paths == {"/tools": SandboxPathAccess.READ, "/a": SandboxPathAccess.DENY}
    assert config.model_dump(mode="json")["sandbox_paths"] == {"/tools": "read", "/a": "deny"}


def test_workspace_write_preference_requires_an_explicit_boolean():
    """Existing policies remain scratch-only and ambiguous new values are rejected."""
    assert not PermissionConfiguration().workspace_command_writes
    with pytest.raises(ValidationError, match="valid boolean"):
        PermissionConfiguration.model_validate({"workspace_command_writes": "off"})


@pytest.mark.parametrize(
    ("scopes", "error"),
    [
        (["user"], "require workspace scope"),
        (["workspace", "workspace"], "identifiers must be unique"),
    ],
)
def test_workspace_command_approvals_reject_invalid_scope_and_duplicate_ids(scopes, error):
    """Only uniquely identified workspace approvals can be stored in workspace policy."""
    rules = [
        {"id": "shared-id", "signature": "bound", "source": "printf safe", "scope": scope}
        for scope in scopes
    ]
    assert not PermissionConfiguration().sandboxed_command_rules
    with pytest.raises(ValidationError, match=error):
        PermissionConfiguration.model_validate({"sandboxed_command_rules": rules})


def test_workspace_backend_options_default_off_and_require_a_selection():
    """Workspace selection starts empty and cannot be set to an ambiguous boolean."""
    assert not PermissionConfiguration().backend_options
    with pytest.raises(ValidationError, match="valid tuple"):
        PermissionConfiguration.model_validate({"backend_options": True})


def test_hard_paths_reject_patterns_in_persisted_workspace_limits():
    """Workspace designation parsing enforces the exact-path native compiler contract."""
    with pytest.raises(ValidationError, match="normalized absolute exact paths"):
        PolicyLimits(hard_denied_paths=("/tmp/*.secret",))


@pytest.fixture(autouse=True)
def isolated_permission_catalog():
    """Exercise the actual packaged catalog loader in its owning model suite."""


def test_actions_map_to_distinct_prompt_icons():
    """Every authority-bearing action exposes its assigned prompt icon."""
    assert {action: action.icon for action in Action} == {
        Action.FILESYSTEM_LIST: "📂",
        Action.FILESYSTEM_READ: "📖",
        Action.FILESYSTEM_EXECUTE: "⚙️",
        Action.FILESYSTEM_CREATE: "📝",
        Action.FILESYSTEM_REPLACE: "✏️",
        Action.FILESYSTEM_DELETE: "🗑️",
        Action.NETWORK_REQUEST: "🌐",
        Action.NETWORK_LOCAL: "🔌",
        Action.NETWORK_BROAD: "🌐",
        Action.NETWORK_INGRESS: "🔌",
        Action.PROCESS_EXECUTE: "⚙️",
        Action.SESSION_MUTATE: "💾",
    }


def test_limit_classification_requires_complete_known_provenance():
    """Policy metadata cannot omit a classified limit or invent its source."""
    with pytest.raises(ValidationError, match="classification provenance is invalid"):
        PolicyLimitClassifications(sources={"readable_roots": "user"})
    with pytest.raises(ValidationError, match="classification provenance is invalid"):
        PolicyLimitClassifications(
            sources={
                "readable_roots": "unknown",
                "writable_roots": "builtin",
                "network_origins": "builtin",
            }
        )


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "file", "path": "/tmp/file", "unknown": True},
        {
            "kind": "network",
            "url": "https://example.test",
            "origin": "https://example.test",
            "unknown": True,
        },
        {"kind": "process", "argv": ["true"], "cwd": "/tmp", "unknown": True},
        {"kind": "session", "identifier": "state", "unknown": True},
        {
            "kind": "file",
            "path": "/tmp/file",
            "manifest": [
                {
                    "relative_path": ".",
                    "kind": "file",
                    "device": 1,
                    "inode": 1,
                    "size": 1,
                    "mtime_ns": 1,
                    "mode": 0,
                    "unknown": True,
                }
            ],
        },
    ],
)
def test_policy_rejects_unknown_fields_in_exact_targets(target):
    """Unknown target restrictions cannot disappear during policy validation."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PermissionConfiguration.model_validate(
            {"rules": [{"decision": "deny", "target": target}]}, extra="forbid"
        )


def test_policy_rejects_unknown_preset_source_fields():
    """Unknown provenance metadata cannot be silently discarded."""
    source = {
        "preset_id": "x",
        "revision": "1",
        "content_hash": "hash",
        "rule_id": "r",
        "unknown": True,
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PermissionConfiguration.model_validate(
            {"rules": [{"decision": "deny", "source": source}]}, extra="forbid"
        )


def test_operations_expose_stable_resource_representations():
    """Typed targets derive canonical strings only for rule matching and display."""
    operations = (
        Operation(tool_id="file", action=Action.FILESYSTEM_READ, target=FileTarget(path="/x")),
        Operation(
            tool_id="web",
            action=Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://my-host.local/a", origin="https://my-host.local"),
        ),
        Operation(
            tool_id="exec",
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(argv=("git", "status"), cwd="/x"),
        ),
        Operation(
            tool_id="skills",
            action=Action.SESSION_MUTATE,
            target=SessionTarget(identifier="activate:demo"),
        ),
    )

    assert tuple(operation.resource for operation in operations) == (
        "/x",
        "https://my-host.local/a",
        "git status",
        "activate:demo",
    )


def test_operations_require_the_target_type_associated_with_their_action():
    """An action cannot omit or disguise the concrete resource policy must evaluate."""
    with pytest.raises(ValidationError, match="requires a FileTarget"):
        Operation(
            tool_id="invalid",
            action=Action.FILESYSTEM_READ,
            target=SessionTarget(identifier="not-a-file"),
        )


def test_policy_configuration_has_a_versioned_complete_default():
    """The editable YAML schema exposes a version and every action fallback."""
    configuration = PermissionConfiguration()

    assert configuration.version == 10
    assert set(configuration.defaults) == set(Action)
    assert configuration.defaults[Action.FILESYSTEM_READ] is Decision.ALLOW


def test_durable_policy_rejects_missing_backend_option_revocation_marker():
    """A present but incomplete marker map cannot restore an older optional-service approval."""
    values = PermissionConfiguration(namespace_generation="1" * 32).model_dump()
    del values["backend_option_revisions"]["account_lookup"]
    with pytest.raises(ValueError, match="revocation markers are incomplete"):
        PermissionConfiguration.model_validate(values)


def test_user_permission_configuration_rejects_duplicate_rule_identifiers():
    """User-wide remembered approvals cannot use ambiguous rule identifiers."""
    with pytest.raises(ValueError, match="rule identifiers must be unique"):
        UserPermissionConfiguration.model_validate(
            {
                "rules": [
                    {"id": "same", "decision": "allow"},
                    {"id": "same", "decision": "allow"},
                ]
            }
        )


@pytest.mark.parametrize(
    "change",
    [
        {"cwd": "/Users/alice/project"},
        {"cwd": "$WORKSPACE_ROOT/../private"},
        {"source": "cat /Users/alice/private"},
        {"signature": "/Users/alice/project"},
    ],
)
def test_host_rule_rejects_real_paths_and_malformed_virtual_context(change):
    """Session host rule records contain virtual paths and a path-free digest only."""
    values = {"signature": "a" * 64, "cwd": "$WORKSPACE_ROOT", "source": "printf ok"}
    values.update(change)
    with pytest.raises(ValidationError, match="virtual context"):
        HostCommandRule.model_validate(values)


def test_user_policy_accepts_only_distinct_durable_host_grants():
    """Persisted user policy refuses session grants and duplicate host grant identities."""
    rule = {
        "id": "same",
        "signature": "a" * 64,
        "cwd": "$WORKSPACE_ROOT",
        "source": "<exact command>",
        "scope": "user",
    }
    with pytest.raises(ValueError, match="durable scopes"):
        UserPermissionConfiguration.model_validate(
            {"trusted_host_command_rules": [{**rule, "scope": "session"}]}
        )
    with pytest.raises(ValueError, match="distinct IDs"):
        UserPermissionConfiguration.model_validate({"trusted_host_command_rules": [rule, rule]})


def test_permission_presets_reject_duplicate_rule_identifiers():
    """A preset rule set cannot contain ambiguous rule identifiers."""
    with pytest.raises(ValueError, match="rule identifiers must be unique"):
        PermissionPreset.model_validate(
            {
                "metadata": {
                    "id": "duplicate-rules",
                    "revision": "1",
                    "title": "Duplicate rules",
                    "description": "Invalid preset.",
                },
                "defaults": {action.value: Decision.DENY.value for action in Action},
                "rules": [
                    {"id": "same", "decision": "allow"},
                    {"id": "same", "decision": "deny"},
                ],
            }
        )


def test_builtin_permission_presets_are_sorted_and_have_stable_hashes(monkeypatch):
    """Packaged presets ignore app branding and expose stable catalog hashes."""
    monkeypatch.setattr(app_constants, "APP_NAME", "renamed-app")
    presets = PermissionPreset.builtin_presets()

    assert [preset.metadata.id for preset in presets] == [
        "locked",
        "observe",
        "supervised",
        "unsupervised",
        "workspace",
    ]
    assert all(preset.content_hash.startswith("sha256:") for preset in presets)
    assert presets[0].content_hash == presets[0].model_copy(deep=True).content_hash
    assert next(
        preset for preset in presets if preset.metadata.id == "unsupervised"
    ).confirmation_warning


def test_builtin_preset_loading_excludes_invalid_artifacts_and_reports_them(monkeypatch):
    """Malformed packaged presets do not prevent valid catalog artifacts from loading."""

    class Resource:
        """Provide a single invalid preset artifact."""

        name = "invalid.yaml"

        @staticmethod
        def read_text(_encoding):
            """Return an invalid preset payload."""
            return "version: 2\n"

        def __str__(self):
            """Render a stable artifact path."""
            return "presets/invalid.yaml"

    class IgnoredResource:
        """Provide an unrelated package resource."""

        name = "README.md"

    class ResourcePackage:
        """Provide package-style access to the invalid preset artifact."""

        @staticmethod
        def joinpath(_name):
            """Return the preset directory."""
            return ResourcePackage()

        @staticmethod
        def iterdir():
            """Return the available preset artifact."""
            return (Resource(), IgnoredResource())

        def __str__(self):
            """Render a stable display path."""
            return "presets"

    monkeypatch.setattr("loop.permissions.models.files", lambda package: ResourcePackage())

    presets, failures = PermissionPreset.load_builtin_presets()

    assert not presets
    assert failures[0].source == "preset"
    assert failures[0].path == "presets/invalid.yaml"


def test_builtin_permission_presets_reject_duplicates_but_retains_one_for_recovery(
    monkeypatch, tmp_path
):
    """Strict loading rejects duplicate IDs while recovery retains one usable preset."""
    defaults = "\n".join(f"  {action.value}: deny" for action in Action)
    payload = (
        "version: 1\nkind: app.permission-preset\nmetadata:\n"
        "  id: duplicate\n  revision: '1'\n  title: Duplicate\n"
        "  description: Invalid\ndefaults:\n"
        f"{defaults}\nrules: []\n"
    )
    (tmp_path / "one.yaml").write_text(payload)
    (tmp_path / "two.yaml").write_text(payload)

    class ResourcePackage:
        """Expose the temporary preset directory as an importlib resource."""

        def joinpath(self, name):
            """Return the requested temporary resource directory."""
            assert name == "presets"
            return tmp_path

    monkeypatch.setattr("loop.permissions.models.files", lambda package: ResourcePackage())
    with pytest.raises(ValueError, match="Built-in permission preset identifiers"):
        PermissionPreset.builtin_presets()

    presets, failures = PermissionPreset.load_builtin_presets()

    assert [preset.metadata.id for preset in presets] == ["duplicate"]
    assert failures[0].message == "Built-in permission preset identifiers must be unique."


@pytest.mark.parametrize("missing", [False, True])
def test_preset_directory_failure_is_reported(monkeypatch, tmp_path, missing):
    """A missing or empty shipped catalog requires explicit recovery."""

    class ResourcePackage:
        """Expose the test directory as a package resource."""

        def joinpath(self, name):
            """Return the requested preset directory."""
            return tmp_path / "missing" if missing else tmp_path

    monkeypatch.setattr("loop.permissions.models.files", lambda package: ResourcePackage())
    presets, failures = PermissionPreset.load_builtin_presets()
    assert presets == ()
    assert len(failures) == 1
    assert "Cannot read" in failures[0].message if missing else "empty" in failures[0].message


@pytest.mark.parametrize("model", [PermissionConfiguration, UserPermissionConfiguration])
@pytest.mark.parametrize("version", [0, 999, "2", True, False, 1.0])
def test_permission_policy_rejects_unsupported_schema_versions(model, version):
    """Both persisted policy scopes reject unknown or malformed schema versions."""
    with pytest.raises(ValidationError):
        model.model_validate({"version": version})


def test_permission_preset_rejects_boolean_version():
    """Preset versions require an actual integer even when the value equals one."""
    values = PermissionPreset.builtin_presets()[0].model_dump()
    values["version"] = True
    with pytest.raises(ValidationError, match="version"):
        PermissionPreset.model_validate(values)


def test_legacy_defaults_retain_process_asks_and_reject_malformed_mappings():
    """Adding native exec policy preserves saved decisions and rejects corrupted defaults."""
    old = PermissionConfiguration.model_validate({"version": 3})
    assert old.defaults[Action.PROCESS_EXECUTE] is Decision.ASK
    assert old.defaults[Action.FILESYSTEM_EXECUTE] is Decision.ALLOW
    current = PermissionConfiguration()
    assert current.defaults[Action.PROCESS_EXECUTE] is Decision.ASK
    for decision in (Decision.ASK, Decision.DENY):
        retained = PermissionConfiguration.model_validate(
            {"version": 3, "defaults": {"process.execute": decision}}
        )
        assert retained.defaults[Action.PROCESS_EXECUTE] is decision
    with pytest.raises(ValidationError, match="defaults"):
        PermissionConfiguration.model_validate({"version": 3, "defaults": []})


@pytest.mark.parametrize("scope", ["local", "broad"])
def test_native_network_targets_keep_distinct_action_and_resource(scope):
    """Native capability targets cannot masquerade as HTTP destinations or another scope."""
    operation = Operation(
        tool_id="run_command",
        action=Action("network." + scope),
        target=NetworkAccessTarget(scope=scope),
    )
    assert operation.resource == "network:" + scope
    assert Operation.model_validate(operation.model_dump(exclude={"resource"})) == operation
    with pytest.raises(ValueError, match="action and capability scope must agree"):
        Operation(
            tool_id="run_command",
            action=Action.NETWORK_BROAD if scope == "local" else Action.NETWORK_LOCAL,
            target=NetworkAccessTarget(scope=scope),
        )


@pytest.mark.parametrize("model", [PermissionConfiguration, UserPermissionConfiguration])
def test_saved_unscoped_egress_allows_require_renewal(model):
    """Migration retains destinations and denials while making legacy unconstrained allows ask."""
    policy = model.model_validate(
        {
            "version": 4,
            "rules": [
                {"id": "wide", "action": "network.request", "decision": "allow"},
                {
                    "id": "exact",
                    "action": "network.request",
                    "decision": "allow",
                    "resource": "https://example.test/*",
                },
                {"id": "deny", "action": "network.request", "decision": "deny"},
            ],
        }
    )
    assert [rule.decision for rule in policy.rules] == [Decision.ASK, Decision.ALLOW, Decision.DENY]
    if isinstance(policy, PermissionConfiguration):
        assert policy.defaults[Action.NETWORK_LOCAL] is Decision.ALLOW
        assert policy.defaults[Action.NETWORK_BROAD] is Decision.ASK
    with pytest.raises(ValidationError, match="rules"):
        model.model_validate({"version": 4, "rules": {}})
    with pytest.raises(ValidationError, match="rules"):
        model.model_validate({"version": 4, "rules": [0]})


def test_legacy_network_default_is_reviewed_and_old_preset_is_complete():
    """Old HTTP defaults cannot silently widen native networking when separate scopes arrive."""
    policy = PermissionConfiguration.model_validate(
        {"version": 4, "defaults": {"network.request": "allow"}}
    )
    assert policy.defaults[Action.NETWORK_REQUEST] is Decision.ASK
    preset = PermissionPreset.builtin_presets()[0].model_dump(mode="json")
    del preset["defaults"]["network.local"]
    del preset["defaults"]["network.broad"]
    del preset["defaults"]["network.ingress"]
    migrated = PermissionPreset.model_validate(preset)
    assert migrated.defaults[Action.NETWORK_BROAD] is Decision.ASK
    assert migrated.defaults[Action.NETWORK_LOCAL] is Decision.ALLOW


def test_ingress_operation_resource_is_exact_host_loopback():
    """The independent ingress action matches only its canonical publication resource."""
    operation = Operation(
        tool_id="run_command", action=Action.NETWORK_INGRESS, target=NetworkIngressTarget(port=8123)
    )
    assert operation.resource == "tcp://127.0.0.1:8123"
