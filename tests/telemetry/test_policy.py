"""Tests for centralized observability disclosure policies."""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from loop.telemetry.policy import ModelInputPolicy, OperationalDisclosurePolicy, freeze, thaw


class Value(BaseModel):
    """Provide one structured value for normalization tests."""

    name: str


@dataclass
class DataclassValue:
    """Provide one dataclass value for normalization tests."""

    enabled: bool


def test_model_input_policy_redacts_only_high_confidence_credentials():
    """Model preparation removes registered, validated, provider, and private-key credentials."""
    reports = []
    alphabet = "aB3dE5fG7hI9jK1mN3pQ5rS7tU9vW1xY3zA5cD7eF9gH1iJ3kL5mN7pQ9rS1tU3vW5xY7zA9"
    openai_key = f"sk-proj-{alphabet[:40]}"
    github_token = f"ghp_{alphabet[:36]}"
    fine_grained_token = f"github_pat_{(alphabet * 2)[:82]}"
    policy = ModelInputPolicy(("registered-secret",), reporter=reports.append)
    source = {
        "authorization": "design notes",
        "nested": [
            (
                "registered-secret "
                "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0."
                "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8 "
                "Authorization: Basic dXNlcjpwYXNz "
                "Authorization: Bearer A1b2C3d4E5f6G7h8I9j0 "
                f"{openai_key} {github_token} {fine_grained_token}"
            ),
            "-----BEGIN PRIVATE KEY-----\nvalue\n-----END PRIVATE KEY-----",
        ],
        "tuple": ("safe",),
    }

    prepared = policy.apply(source)

    assert prepared["authorization"] == "design notes"
    assert "registered-secret" not in prepared["nested"][0]
    assert "Authorization: Bearer <redacted:secret>" in prepared["nested"][0]
    assert "Authorization: Basic <redacted:secret>" in prepared["nested"][0]
    assert openai_key not in prepared["nested"][0]
    assert github_token not in prepared["nested"][0]
    assert fine_grained_token not in prepared["nested"][0]
    assert "PRIVATE KEY" not in prepared["nested"][1]
    assert prepared["tuple"] == ("safe",)
    assert dict(reports[0]) == {
        "registered_exact": 1,
        "private_key": 1,
        "authorization": 3,
        "openai_api_key": 1,
        "github_token": 1,
        "github_fine_grained_token": 1,
    }


def test_model_input_policy_ignores_redaction_reporting_failures():
    """Redaction remains effective when optional observability reporting fails."""
    reporter = Mock(side_effect=RuntimeError("unavailable"))
    policy = ModelInputPolicy(("registered-secret",), reporter=reporter)

    assert policy.apply("contains registered-secret") == "contains <redacted:secret>"
    reporter.assert_called_once()


@pytest.mark.parametrize("secret", ["test", "key", "dummy", "local-api-key"])
def test_model_input_policy_preserves_weak_secrets_in_ordinary_text(secret):
    """Weak credentials never corrupt common prose or repository paths."""
    source = "Let me try the test directory. Reviewed 6 test files. Assessment: the test updates."

    assert ModelInputPolicy((secret,)).apply(source) == source


def test_model_input_policy_limits_short_registered_secrets_to_credential_context():
    """Short credentials are removed only from explicit assignments."""
    policy = ModelInputPolicy(("tiny-key", "test"))

    assert policy.apply(
        "The tiny-key adapter uses api_key='tiny-key', OPENAI_API_KEY=test, and "
        "Authorization: Bearer tiny-key. password=test."
    ) == (
        "The tiny-key adapter uses api_key='<redacted:secret>', "
        "OPENAI_API_KEY=<redacted:secret>, and Authorization: Bearer <redacted:secret>. "
        "password=<redacted:secret>."
    )


def test_model_input_policy_redacts_weak_proxy_authorization_assignments():
    """Short credentials remain protected in explicit proxy-header assignments."""
    policy = ModelInputPolicy(("test",))

    assert policy.apply("Proxy-Authorization = Bearer test") == (
        "Proxy-Authorization = Bearer <redacted:secret>"
    )


def test_model_input_policy_preserves_weak_secrets_in_unrelated_assignments():
    """Sensitive-word substrings do not turn ordinary assignments into credential contexts."""
    source = "tokenizer=test passwordless=test secretariat=test"

    assert ModelInputPolicy(("test",)).apply(source) == source


@pytest.mark.parametrize(
    "source",
    ["A basic test passes.", "The secret, test directory.", "A bearer test passes."],
)
def test_model_input_policy_preserves_weak_secrets_in_authentication_prose(source):
    """Weak registered words require an assignment rather than adjacent security vocabulary."""
    assert ModelInputPolicy(("test",)).apply(source) == source


@pytest.mark.parametrize(
    "source",
    [
        '{"api_key": "test"}',
        "{'password': 'test'}",
        '"Authorization": "Bearer test"',
        "api_key:=test",
        "api_key=>test",
        "api_key<-test",
    ],
)
def test_model_input_policy_redacts_weak_secrets_in_serialized_assignments(source):
    """Quoted JSON keys and multi-character assignment operators retain credential protection."""
    assert ModelInputPolicy(("test",)).apply(source) == source.replace("test", "<redacted:secret>")


def test_model_input_policy_redacts_weak_secrets_before_url_delimiters():
    """Short credentials are removed from URL query and path credential contexts."""
    policy = ModelInputPolicy(("test",))

    assert policy.apply(
        "https://example.test/?api_key=test&next=1; "
        "api_key=test#fragment; api_key=test/path; api_key=test?next=1"
    ) == (
        "https://example.test/?api_key=<redacted:secret>&next=1; "
        "api_key=<redacted:secret>#fragment; api_key=<redacted:secret>/path; "
        "api_key=<redacted:secret>?next=1"
    )


def test_model_input_policy_matches_registered_weak_secrets_case_sensitively():
    """Credential context matching preserves differently cased values."""
    policy = ModelInputPolicy(("test",))

    assert policy.apply("API_KEY=TEST; api_key=test") == ("API_KEY=TEST; api_key=<redacted:secret>")


def test_model_input_policy_preserves_weak_secrets_under_semantic_keys():
    """Arbitrary mapping keys do not turn weak values into known credential metadata."""
    source = {
        "OPENAI_API_KEY": "test",
        "nested": {"token": "test", "password": "field in an example schema"},
    }

    assert ModelInputPolicy(("test",)).apply(source) == source


@pytest.mark.parametrize("credential", ["A1b2C3d4E5f6G7h8I9j0", "A1b2C3d4E5f6G7h8I9j0=="])
def test_model_input_policy_redacts_high_entropy_sensitive_assignments(credential):
    """Unregistered opaque credentials require both sensitive assignment context and strength."""
    source = f'api_key = "{credential}"; object_id = {credential}'

    assert ModelInputPolicy().apply(source) == (
        f'api_key = "<redacted:secret>"; object_id = {credential}'
    )


def test_model_input_policy_preserves_ambiguous_patterns_and_semantic_keys():
    """Ordinary auth prose, loose prefixes, and semantic field names remain model-visible."""
    source = {
        "token": "a lexical token",
        "password": "field in an example schema",
        "authorization": "design notes",
        "text": (
            "Use basic authentication in a basic directory. Bearer abc. "
            "Bearer documentation2026. Bearer feature1234567890. "
            "Bearer abc.def.ghi! Bearer characterization. "
            "Bearer AbCdEfGhIjKlMnOp@example.com. "
            "Bearer ____.____.signature. "
            "Bearer abc..ghi. "
            "Bearer abc.def=.ghi. "
            "Basic dGVzdA==. Basic /w==. sk_abcdefghijklmnopqrstuvwxyz"
        ),
    }

    assert ModelInputPolicy().apply(source) == source


def test_model_input_policy_handles_long_dotted_bearer_prose_without_pathological_matching():
    """Long malformed bearer prose remains visible without pathological regex backtracking."""
    source = f"Bearer {'segment.' * 10_000}tail"

    assert ModelInputPolicy().apply(source) == source


def test_model_input_policy_preserves_strong_opaque_bearer_values_without_header_names():
    """Generic bearer-looking prose requires an explicit authorization-header context."""
    credential = "A1b2C3d4E5f6G7h8I9j0"

    source = f"Bearer {credential}"

    assert ModelInputPolicy().apply(source) == source


@pytest.mark.parametrize("header", ["Authorization", "Proxy-Authorization"])
@pytest.mark.parametrize("credential", ["A1b2C3d4E5f6G7h8I9j0", "A1b2C3d4E5f6G7h8I9j0==="])
def test_model_input_policy_redacts_opaque_bearer_values_in_explicit_headers(header, credential):
    """Explicit bearer headers protect opaque credentials that pass strength gates."""
    assert ModelInputPolicy().apply(f"{header}: Bearer {credential}") == (
        f"{header}: Bearer <redacted:secret>"
    )


def test_model_input_policy_preserves_sentence_punctuation_after_authorization_credentials():
    """Authorization redaction leaves adjacent sentence punctuation intact."""
    credential = "A1b2C3d4E5f6G7h8I9j0"

    assert ModelInputPolicy().apply(f"Authorization: Bearer {credential}.") == (
        "Authorization: Bearer <redacted:secret>."
    )


def test_model_input_policy_preserves_oversized_authorization_values():
    """Authorization candidates beyond the documented matching cap remain unchanged."""
    credential = "A1" * 1_025
    source = f"Authorization: Bearer {credential}"

    assert ModelInputPolicy().apply(source) == source


@pytest.mark.parametrize(
    "source",
    [
        'Authorization: "Bearer A1b2C3d4E5f6G7h8I9j0"',
        '{"Authorization": "Bearer A1b2C3d4E5f6G7h8I9j0"}',
        '{"Proxy-Authorization": "Basic dXNlcjpwYXNz"}',
    ],
)
def test_model_input_policy_redacts_credentials_in_quoted_authorization_headers(source):
    """Quoted and JSON authorization values preserve their enclosing header structure."""
    prepared = ModelInputPolicy().apply(source)

    assert "<redacted:secret>" in prepared
    assert "Authorization" in prepared


@pytest.mark.parametrize("credential", ["a", "abc===", "placeholder", "documentation2026"])
def test_model_input_policy_preserves_weak_unregistered_bearer_header_examples(credential):
    """Explicit bearer headers preserve weak unregistered documentation examples."""
    source = f"Authorization: Bearer {credential}"

    assert ModelInputPolicy().apply(source) == source


def test_model_input_policy_preserves_invalid_empty_bearer_credentials():
    """Authorization headers without bearer-token characters remain unchanged."""
    source = "Authorization: Bearer ="

    assert ModelInputPolicy().apply(source) == source


@pytest.mark.parametrize("credential", ["dXNlcjo=", "OnBhc3M=", "Og=="])
def test_model_input_policy_preserves_basic_placeholders_with_empty_components(credential):
    """Basic-auth examples with an empty username or password remain model-visible."""
    source = f"Basic {credential}"

    assert ModelInputPolicy().apply(source) == source


@pytest.mark.parametrize("credential", ["/w==", "abc", "dXNlcjpwYXNzd29yZA="])
def test_model_input_policy_preserves_malformed_basic_authorization_headers(credential):
    """Malformed Base64 Basic authorization headers remain model-visible."""
    source = f"Authorization: Basic {credential}"

    assert ModelInputPolicy().apply(source) == source


def test_model_input_policy_redacts_paddingless_basic_authorization_headers():
    """Valid unpadded Basic credentials retain protection in explicit headers."""
    source = "Authorization: Basic dXNlcjpwYXNzd29yZA"

    assert ModelInputPolicy().apply(source) == "Authorization: Basic <redacted:secret>"


@pytest.mark.parametrize("operator", ["=", ":=", "=>", "<-"])
def test_model_input_policy_redacts_bearer_authorization_assignments(operator):
    """Explicit authorization assignments protect validated bearer credentials."""
    credential = "A1b2C3d4E5f6G7h8I9j0"
    source = f"authorization {operator} Bearer {credential}"

    assert ModelInputPolicy().apply(source) == (
        f"authorization {operator} Bearer <redacted:secret>"
    )


@pytest.mark.parametrize(
    "credential",
    [
        "abc..ghi",
        "aA.e30.ABCD",
        "e30.e30.A",
        "eyJhbGciOiJub25lIn0.e30.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abc",
        "eyJhbGciOiJIUzI1NiJ9.e30.x",
        "eyJhbGciOiJIUzI1NiJ9.e30.eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHg",
        "eA.e30.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abc",
        "_w.e30.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abc",
    ],
)
def test_model_input_policy_preserves_malformed_unsigned_and_placeholder_jwts(credential):
    """Malformed, unsigned, truncated, and placeholder header JWTs remain visible."""
    source = f"Authorization: Bearer {credential}"

    assert ModelInputPolicy().apply(source) == source


def test_model_input_policy_preserves_nonsemantic_scalar_values():
    """Non-string model values remain structurally unchanged."""
    assert ModelInputPolicy().apply(0) == 0


def test_model_input_policy_requires_complete_credential_boundaries():
    """Credential prefixes inside longer values remain unchanged."""
    github_prefix = f"ghp_{'aB3dE5fG7hI9jK1mN3pQ5rS7tU9vW1xY'[:36]}"
    source = f"api_key='tiny-key-suffix' {github_prefix}-suffix"

    assert ModelInputPolicy(("tiny-key",)).apply(source) == source


def test_model_input_policy_prefers_longest_unique_registered_secret():
    """Overlapping registered values redact the complete longest credential once."""
    reports = []
    shorter = "abcdefghijklmnop"
    longer = f"{shorter}SECRETTAIL"
    policy = ModelInputPolicy((shorter, longer, longer), reporter=reports.append)

    assert policy.apply(longer) == "<redacted:secret>"
    assert dict(reports[0]) == {"registered_exact": 1}


def test_model_input_policy_rejects_low_entropy_provider_placeholders():
    """Provider-shaped repeated placeholders do not qualify as credentials."""
    source = f"ghp_{'x' * 36} github_pat_{'x' * 82}"

    assert ModelInputPolicy().apply(source) == source


@pytest.mark.parametrize("namespace", ["proj", "svcacct", "admin", "None", None])
def test_model_input_policy_redacts_supported_openai_key_namespaces(namespace):
    """Supported OpenAI key namespaces are redacted without fixed-length assumptions."""
    credential = (
        f"sk-{namespace + '-' if namespace else ''}aB3dE5fG7hI9jK1mN3pQ5rS7tU9vW1xY3zA5cD7eF9"
    )

    assert ModelInputPolicy().apply(credential) == "<redacted:secret>"


def test_operational_policy_minimizes_fields_and_sanitizes_lines():
    """Operational metadata excludes content and secrets while retaining safe scalar facts."""
    attributes = OperationalDisclosurePolicy().normalize(
        {
            "prompt": "private",
            "api-key": "credential",
            "component": "backend\nforged",
            "count": 2,
            "exception": RuntimeError("private"),
        }
    )

    assert dict(attributes) == {
        "component": "backend\\nforged",
        "count": 2,
        "exception": "RuntimeError",
    }


def test_freeze_and_thaw_support_models_dataclasses_and_nested_values():
    """Telemetry normalization creates immutable structures and reversible JSON values."""
    frozen = freeze(
        {
            "model": Value(name="value"),
            "dataclass": DataclassValue(enabled=True),
            "items": [1, None],
        }
    )

    assert thaw(frozen) == {
        "model": {"name": "value"},
        "dataclass": {"enabled": True},
        "items": [1, None],
    }


def test_freeze_rejects_arbitrary_objects():
    """Normalization never invokes arbitrary object string conversion."""
    with pytest.raises(TypeError, match="Unsupported telemetry value type"):
        freeze(object())
