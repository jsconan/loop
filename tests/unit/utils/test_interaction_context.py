"""Verify interaction execution identity scoping."""

from uuid import UUID

from loop.utils import (
    InteractionCorrelation,
    current_interaction_correlation,
    interaction_operation,
    interaction_run,
)


def test_interaction_correlation_is_empty_outside_a_run():
    """Interactions outside an agent run have no session, run, or operation identity."""
    correlation = current_interaction_correlation()

    assert isinstance(correlation, InteractionCorrelation)
    assert correlation.session_id is None
    assert correlation.run_id is None
    assert correlation.operation_id is None


def test_interaction_run_reuses_nested_identity_and_restores_parent():
    """Nested work in one session shares its run and scope exit restores its parent."""
    before = current_interaction_correlation()

    with interaction_run("session-a"):
        run = current_interaction_correlation()
        assert run.session_id == "session-a"
        assert UUID(run.run_id).version in {4, 7}

        with interaction_run("session-a"):
            assert current_interaction_correlation().run_id == run.run_id

        assert current_interaction_correlation() == run

    assert current_interaction_correlation() == before


def test_interaction_run_starts_a_new_identity_for_a_different_session():
    """Changing session starts a fresh run while restoring the original run afterward."""
    with interaction_run("session-a"):
        parent = current_interaction_correlation()
        with interaction_run("session-b"):
            child = current_interaction_correlation()
            assert child.session_id == "session-b"
            assert child.run_id != parent.run_id
        assert current_interaction_correlation() == parent


def test_interaction_operation_preserves_and_restores_run_identity():
    """Operation scope adds a call identity without changing its session or run."""
    with interaction_run("session-a"):
        run = current_interaction_correlation()
        with interaction_operation("call-a"):
            operation = current_interaction_correlation()
            assert operation.session_id == run.session_id
            assert operation.run_id == run.run_id
            assert operation.operation_id == "call-a"
        assert current_interaction_correlation() == run


def test_interaction_operation_can_scope_requests_without_a_run():
    """Standalone operations retain absent session and run identities."""
    with interaction_operation("call-a"):
        correlation = current_interaction_correlation()

    assert correlation.session_id is None
    assert correlation.run_id is None
    assert correlation.operation_id == "call-a"
    assert current_interaction_correlation().operation_id is None
