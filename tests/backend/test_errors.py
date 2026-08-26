"""Tests for provider-neutral backend failures."""

from loop import BackendAuthenticationError, BackendRateLimitError


def test_backend_error_converts_all_safe_diagnostics_to_a_problem():
    """Backend problem conversion retains recovery and provider request identifiers."""
    error = BackendRateLimitError(
        "slow down",
        provider="example",
        operation="create_response",
        status_code=429,
        code="rate_limit",
        request_id="request-1",
    )

    problem = error.to_problem()

    assert problem.code == "backend.request_failed"
    assert problem.detail == "slow down"
    assert problem.severity == "error"
    assert problem.retryable is True
    assert problem.metadata == {
        "status_code": 429,
        "request_id": "request-1",
        "provider_code": "rate_limit",
    }


def test_backend_error_keeps_nonrecoverable_failures_as_errors():
    """Backend problem conversion keeps non-recoverable failures as errors."""
    error = BackendAuthenticationError(
        "invalid credentials",
        provider="example",
        operation="list_models",
    )

    problem = error.to_problem()

    assert problem.severity == "error"
    assert problem.retryable is False
