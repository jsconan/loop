"""Verify command output spooling and process-local session ownership."""

import json
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from loop.errors import Problem
from loop.execution import jobs
from loop.utils.content import encode_content_cursor


@pytest.fixture
def workers(monkeypatch):
    """Retain job worker targets for deterministic unit scheduling."""
    created = []

    class Worker:
        """Represent one worker without launching a real thread."""

        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.joined = []
            created.append(self)

        def start(self):
            """Leave execution under the test's explicit control."""

        def join(self, timeout):
            """Record the requested bounded cleanup wait."""
            self.joined.append(timeout)

    monkeypatch.setattr(jobs.threading, "Thread", Worker)
    return created


def test_command_job_spools_and_pages_the_complete_output():
    """Large streams remain byte exact and cursor-readable beyond native capture ceilings."""
    from types import SimpleNamespace

    from loop.constants import MAX_OUTPUT_CHARS, MAX_TOOL_CONTENT_BYTES

    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    content = ("abé\0" * 210_000) + "tail"
    job.publish("stdout", content)
    job.publish("stderr", "err")
    job.capture_result_output(
        SimpleNamespace(stdout=content[:MAX_OUTPUT_CHARS], stderr="err"),
        {"stdout": 0, "stderr": 0},
    )
    stdout = job.stream_result("stdout", len(content) - MAX_OUTPUT_CHARS)
    stderr = job.stream_result("stderr", 0)
    job.complete({"exit_code": 0, "stdout": stdout, "stderr": stderr})

    assert stdout["capture_complete"] is True
    assert stdout["captured_bytes"] == len(content.encode("utf-8"))
    assert stdout["content"] != content
    assert stderr["content"] == "err"
    pages = []
    cursor = None
    while True:
        page = job.read_output("stdout", cursor, MAX_TOOL_CONTENT_BYTES)
        pages.append(page["content"])
        cursor = page.get("next_cursor")
        if cursor is None:
            break
    assert "".join(pages) == content
    assert page["job_complete"] is True
    assert page["capture_complete"] is True
    assert "\0" in pages[0]


def test_command_job_output_pages_resume_while_the_process_is_running():
    """A read returns the available page immediately and resumes from its exact byte cursor."""
    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    job.publish("stdout", "€first")
    first = job.read_output("stdout", None, 1)
    assert first["content"] == "€"
    assert first["included_bytes"] == 3
    assert first["job_complete"] is False
    assert first["capture_complete"] is False

    job.publish("stdout", "second")
    next_page = job.read_output("stdout", first["next_cursor"], 20)
    assert next_page["content"] == "firstsecond"
    assert next_page["start_byte"] == 3
    assert next_page["job_complete"] is False


def test_command_job_rejects_future_output_cursor():
    """A future byte offset cannot hide output that arrives later."""
    from loop.utils.content import encode_content_cursor

    job = jobs.CommandJob("session", 3600.0)
    job.publish("stdout", "first")
    cursor = encode_content_cursor(job._outputs["stdout"].handle, 6)
    with pytest.raises(ValueError, match="beyond the end"):
        job.read_output("stdout", cursor, 10)
    assert job.read_output("stdout", None, 10)["content"] == "first"


def test_fallback_capture_does_not_claim_capped_output_is_complete():
    """A native result fallback cannot masquerade as the complete live observer stream."""
    from types import SimpleNamespace

    from loop.constants import MAX_OUTPUT_CHARS

    job = jobs.CommandJob("session", 3600.0)
    job.capture_result_output(
        SimpleNamespace(stdout="x" * MAX_OUTPUT_CHARS, stderr=""),
        {"stdout": 0, "stderr": 0},
    )
    result = job.stream_result("stdout", 10)
    assert result["captured_bytes"] == MAX_OUTPUT_CHARS
    assert result["capture_complete"] is False
    assert result["discarded_characters"] is None


def test_job_poll_waits_for_completion_not_for_output(monkeypatch):
    """Published stream data does not wake a poll before timeout or job completion."""
    now = [0.0]
    monkeypatch.setattr(jobs.time, "monotonic", lambda: now[0])
    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    waits = []

    def publish_during_wait(timeout):
        """Publish output but leave the job active until the poll budget expires."""
        waits.append(timeout)
        job.publish("stdout", "available")
        now[0] += timeout
        return False

    monkeypatch.setattr(job._condition, "wait", publish_during_wait)
    status = job.poll(5)
    assert waits == [5]
    assert status["status"] == "running"
    assert status["stdout_bytes"] == len("available")
    assert status["stdout"]["content"] == "available"
    assert status["stdout"]["next_cursor"]
    assert not status["cancel_requested"]
    assert not job.cancellation_event.is_set()


def test_job_poll_returns_when_completion_wakes_it(monkeypatch):
    """A completion notification ends the wait and returns output with status."""
    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    waits = []

    def complete_during_wait(timeout):
        """Settle the command while poll is waiting on the completion condition."""
        waits.append(timeout)
        job.complete({"exit_code": 0})
        return False

    monkeypatch.setattr(job._condition, "wait", complete_during_wait)
    status = job.poll(5)
    assert waits == [5]
    assert status["status"] == "complete"
    assert status["exit_code"] == 0
    assert status["stdout"]["content"] == ""
    assert status["stderr"]["content"] == ""


def test_job_poll_cursors_return_only_new_output_pages():
    """Per-stream cursors continue after prior poll output without repeating text."""
    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    job.publish("stdout", "first")
    first = job.poll(0)
    cursor = first["stdout"]["next_cursor"]

    job.publish("stdout", "second")
    second = job.poll(0, stdout_cursor=cursor)

    assert first["stdout"]["content"] == "first"
    assert second["stdout"]["content"] == "second"
    assert second["stdout"]["start_byte"] == len("first")


def test_job_poll_pages_fit_the_tool_result_budget():
    """Poll pages shrink when escaped text would exceed the complete result budget."""
    from loop.constants import MAX_TOOL_RESULT_BYTES

    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    job.publish("stdout", "\x01" * 5000)
    job.publish("stderr", "\x02" * 5000)

    status = job.poll(0)

    assert len(json.dumps({"ok": True, "result": status}).encode("utf-8")) < MAX_TOOL_RESULT_BYTES
    assert status["stdout"]["included_bytes"] < status["stdout"]["captured_bytes"]
    assert status["stderr"]["included_bytes"] < status["stderr"]["captured_bytes"]
    assert status["stdout"]["next_cursor"]
    assert status["stderr"]["next_cursor"]


def test_job_poll_reduces_page_to_zero_when_the_envelope_budget_is_too_small(monkeypatch):
    """A zero-byte poll page keeps its cursor when no content fits the result envelope."""
    monkeypatch.setattr(jobs.constants, "MAX_TOOL_RESULT_BYTES", 128)
    monkeypatch.setattr(jobs.constants, "MAX_TOOL_CONTENT_BYTES", 1)
    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    job.publish("stdout", "x")
    job.complete({"exit_code": 0})

    status = job.status_with_output()

    assert status["stdout"]["content"] == ""
    assert status["stderr"]["content"] == ""
    assert status["stdout"]["included_bytes"] == 0
    assert status["stderr"]["included_bytes"] == 0
    assert status["stdout"]["next_cursor"]


def test_job_poll_rejects_cursor_beyond_current_output():
    """A cursor into future bytes is rejected before the poll starts waiting."""
    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    job.publish("stdout", "current")
    future = encode_content_cursor(job._outputs["stdout"].handle, 100)

    with pytest.raises(ValueError, match="beyond the end"):
        job.poll(5, stdout_cursor=future)


def test_job_poll_interrupt_stops_wait_without_cancelling_command(monkeypatch):
    """Cancelling a poll stops that caller's wait but leaves command ownership intact."""
    job = jobs.CommandJob("session", 3600.0)
    job.admitted()
    interrupted = Event()

    def interrupt_during_wait(_timeout):
        """Simulate a call interruption while the job itself remains active."""
        interrupted.set()
        return False

    monkeypatch.setattr(job._condition, "wait", interrupt_during_wait)
    status = job.poll(5, cancellation_event=interrupted)
    assert status["status"] == "running"
    assert not status["cancel_requested"]
    assert not job.cancellation_event.is_set()


def test_command_job_elapsed_time_stops_at_completion(monkeypatch):
    """Report time since admission and freeze it when the job settles."""
    now = [10.0]
    monkeypatch.setattr(jobs.time, "monotonic", lambda: now[0])
    job = jobs.CommandJob("session", 3600.0)
    assert job.inspect()["elapsed_seconds"] is None
    job.admitted()
    now[0] = 12.0
    assert job.inspect()["elapsed_seconds"] == 2.0
    now[0] = 15.0
    job.complete({"exit_code": 0})
    now[0] = 20.0
    assert job.inspect()["elapsed_seconds"] == 5.0


def test_cancellation_signal_transfers_origin_at_handoff():
    """Cancellation follows the origin before handoff and remains owned afterward."""
    origin = Event()
    job = jobs.CommandJob("session", 3600.0)
    signal = job.cancellation_signal(origin)
    assert not signal.is_set()
    origin.set()
    assert signal.is_set()
    job.handoff(origin)
    origin.clear()
    assert signal.is_set()
    assert job.cancellation_event.is_set()


def test_cancellation_arriving_during_handoff_is_retained():
    """A stop arriving between the origin check and transfer reaches the owned job."""
    origin = Mock()
    origin.is_set.side_effect = [False, True]
    job = jobs.CommandJob("session", 3600.0)
    signal = job.cancellation_signal(origin)

    job.handoff(origin)

    assert origin.is_set.call_count == 2
    assert job.cancellation_event.is_set()
    assert signal.is_set()


def test_cancellation_signal_ignores_later_origin_but_honors_job_cancellation():
    """After handoff, a later origin stop cannot cancel an independently owned job."""
    origin = Event()
    job = jobs.CommandJob("session", 3600.0)
    signal = job.cancellation_signal(origin)
    job.handoff(origin)
    origin.set()
    assert not signal.is_set()
    job.cancel()
    assert signal.is_set()
    assert signal.wait(0)


def test_cancellation_signal_wait_bounds_and_observes_origin(monkeypatch):
    """A timed wait expires without cancellation; an unbounded wait observes the origin."""
    now = [0.0]
    monkeypatch.setattr(jobs.time, "monotonic", lambda: now[0])
    origin = Event()
    job = jobs.CommandJob("session", 3600.0)
    signal = job.cancellation_signal(origin)
    waits = []

    def wait_for_signal(seconds):
        """Advance a controlled clock and expose each bounded wait interval."""
        waits.append(seconds)
        now[0] += seconds
        if len(waits) == 3:
            origin.set()
        return False

    monkeypatch.setattr(job.cancellation_event, "wait", wait_for_signal)
    assert not signal.wait(0)
    assert not signal.wait(jobs.JOB_READ_CANCELLATION_CHECK_SECONDS * 2)
    assert signal.wait()
    assert waits == pytest.approx([jobs.JOB_READ_CANCELLATION_CHECK_SECONDS] * 3)


def test_approval_interaction_cannot_start_after_handoff():
    """A transferred command cannot start another interaction on the old caller."""
    job = jobs.CommandJob("session", 3600.0)
    assert job.begin_interaction()
    job.end_interaction()
    job.handoff()
    assert not job.begin_interaction()


def test_job_delivers_unexpected_error_to_the_initial_caller_once(monkeypatch):
    """A settled failure reaches its initial caller and is not logged again on handoff."""
    logged = Mock()
    monkeypatch.setattr(jobs, "log_problem", logged)
    job = jobs.CommandJob("session", 3600.0)
    assert job.result() is None
    error = RuntimeError("synthetic failure")
    problem = Problem(code="command_job.failed", title="Failed", detail="Safe failure")
    job.complete(problem, error)
    with pytest.raises(RuntimeError, match="synthetic failure") as caught:
        job.result()
    assert caught.value is error
    assert job.result() is problem
    job.handoff()
    logged.assert_not_called()


def test_detached_job_reports_an_unexpected_failure_once(monkeypatch):
    """A failure after handoff is logged once and remains inspectable as a safe result."""
    logged = Mock()
    monkeypatch.setattr(jobs, "log_problem", logged)
    job = jobs.CommandJob("session", 3600.0)
    job.handoff()
    error = RuntimeError("synthetic failure")
    problem = Problem(code="command_job.failed", title="Failed", detail="Safe failure")
    job.complete(problem, error)
    assert job.result() is problem
    assert job.inspect()["outcome"] == problem.code
    logged.assert_called_once()
    assert logged.call_args.args[1:] == (problem, error)


def test_initial_job_wait_stops_at_deadline_or_origin_cancellation(monkeypatch):
    """Initial waits stop at their deadline or cancellation without stopping the job."""
    job = jobs.CommandJob("session", 3600.0)
    now = [0.0]
    monkeypatch.setattr(jobs.time, "monotonic", lambda: now[0])
    origin = Event()
    waits = []

    def wait_for_completion(seconds):
        """Advance time and make cancellation observable during the second wait."""
        waits.append(seconds)
        now[0] += seconds
        if len(waits) == 3:
            origin.set()
        return False

    monkeypatch.setattr(job.finished, "wait", wait_for_completion)
    job.wait(0, None)
    assert waits == [0]
    job.wait(jobs.JOB_READ_CANCELLATION_CHECK_SECONDS, origin)
    assert waits[-1] == jobs.JOB_READ_CANCELLATION_CHECK_SECONDS
    job.wait(10, origin)
    assert origin.is_set()
    assert len(waits) == 3
    assert not job.cancellation_event.is_set()


def test_completed_job_ignores_late_output():
    """A completed job cannot append output from a late pipe reader."""
    job = jobs.CommandJob("session", 3600.0)
    job.publish("stdout", "before")
    job.complete({"exit_code": 0})
    before = job.output_sizes()
    job.publish("stdout", "after")
    job.publish("stderr", "after")
    assert job.output_sizes() == before


def test_terminal_job_bounds_transient_controls_and_escapes_output():
    """An owned terminal queues exact controls and renders unsafe output visibly."""
    job = jobs.CommandJob("session", 3600.0, terminal=True)
    job.admitted()
    job.write_terminal("é\n")
    job.resize_terminal(30, 100)
    job.terminal_eof()
    job.terminal_interrupt()
    job.publish("stdout", "ready\x1b]52;c;secret\x07\r\n")
    output = job.read_output("stdout", None, 100)

    assert job.inspect()["terminal"]
    assert output["content"] == "ready\u241b]52;c;secret\u2407\u240d\n"
    assert job.next_terminal_frame() == b"I\x00\x03\xc3\xa9\n"
    assert job.next_terminal_frame() == b"Z\x00\x1e\x00d"
    assert job.next_terminal_frame() == b"F"
    assert job.next_terminal_frame() == b"X"
    assert job.next_terminal_frame() is None

    job.write_terminal("secret")
    job.complete({"exit_code": 0})
    assert job.next_terminal_frame() is None
    with pytest.raises(ValueError, match="unavailable"):
        job.terminal_eof()


@pytest.mark.parametrize("content", [None, "", "x" * 4097])
def test_terminal_job_refuses_invalid_input(content):
    """Only nonempty bounded UTF-8 text can enter a native terminal queue."""
    job = jobs.CommandJob("session", 3600.0, terminal=True)
    with pytest.raises((TypeError, ValueError), match="input"):
        job.write_terminal(content)


@pytest.mark.parametrize(("rows", "columns"), [(0, 80), (24, 501), (True, 80), (24, "80")])
def test_terminal_job_refuses_invalid_window_sizes(rows, columns):
    """Malformed dimensions cannot reach a native terminal ioctl."""
    job = jobs.CommandJob("session", 3600.0, terminal=True)
    with pytest.raises(ValueError, match="rows and columns"):
        job.resize_terminal(rows, columns)


def test_terminal_job_refuses_nonterminal_control_and_queue_overflow(monkeypatch):
    """Normal jobs have no control channel and terminal input applies bounded backpressure."""
    ordinary = jobs.CommandJob("session", 3600.0)
    with pytest.raises(ValueError, match="unavailable"):
        ordinary.terminal_interrupt()
    terminal = jobs.CommandJob("session", 3600.0, terminal=True)
    monkeypatch.setattr(jobs, "MAX_TERMINAL_PENDING_BYTES", 4)
    with pytest.raises(ValueError, match="backed up"):
        terminal.write_terminal("ab")
    terminal.cancel()
    assert terminal.next_terminal_frame() is None
    with pytest.raises(ValueError, match="unavailable"):
        terminal.write_terminal("a")


def test_settled_job_rejects_late_admission_completion_and_cancel():
    """A finished job cannot be relabeled or cancelled by a late callback."""
    job = jobs.CommandJob("session", 3600.0)
    job.complete({"exit_code": 0})
    job.admitted()
    job.complete({"exit_code": 7})
    job.cancel()
    assert job.inspect()["status"] == "complete"
    assert not job.inspect()["cancel_requested"]
    assert job.result() == {"exit_code": 0}


def test_output_capture_failure_is_explicit_and_stops_the_command(monkeypatch):
    """A failed spool write cancels execution and becomes an explicit capture problem."""
    logged = Mock()
    monkeypatch.setattr(jobs, "log_problem", logged)

    def fail_append(_spool, _content):
        """Represent an output filesystem that cannot retain another byte."""
        raise OSError("disk full")

    monkeypatch.setattr(jobs._ContentSpool, "append", fail_append)
    job = jobs.CommandJob("session", 3600.0)
    job.publish("stdout", "unretained")
    assert job.cancellation_event.is_set()
    job.complete({"exit_code": 0})
    result = job.result()
    assert isinstance(result, Problem)
    assert result.code == "command_output.capture_failed"
    assert result.metadata["capture_failed"] is True
    logged.assert_called_once()


def test_result_fallback_capture_failure_is_explicit_and_stops_the_command(monkeypatch):
    """A failed fallback write is reported through the same capture-failure path."""
    from types import SimpleNamespace

    logged = Mock()
    monkeypatch.setattr(jobs, "log_problem", logged)

    def fail_append(_spool, _content):
        """Represent a full filesystem when a backend result is retained."""
        raise OSError("disk full")

    monkeypatch.setattr(jobs._ContentSpool, "append", fail_append)
    job = jobs.CommandJob("session", 3600.0)
    job.capture_result_output(
        SimpleNamespace(stdout="unretained", stderr=""),
        {"stdout": 0, "stderr": 0},
    )

    assert job.cancellation_event.is_set()
    job.complete({"exit_code": 0})
    assert job.result().code == "command_output.capture_failed"
    logged.assert_called_once()


@pytest.mark.parametrize("stream", ["other", "", "STDOUT"])
def test_command_job_rejects_unknown_output_stream(stream):
    """Only the two authenticated process streams may enter a job spool."""
    with pytest.raises(ValueError, match="output stream"):
        jobs.CommandJob("session", 3600.0).publish(stream, "x")


@pytest.mark.parametrize("stream", ["other", "", "STDOUT"])
def test_command_job_rejects_unknown_output_read_stream(stream):
    """A read cannot address any stream outside stdout and stderr."""
    with pytest.raises(ValueError, match="output stream"):
        jobs.CommandJob("session", 3600.0).read_output(stream, None, 10)


def test_command_job_rejects_unknown_output_result_stream():
    """A final result cannot request metadata for an unowned stream."""
    with pytest.raises(ValueError, match="output stream"):
        jobs.CommandJob("session", 3600.0).stream_result("other", 0)


@pytest.mark.parametrize("cursor", ["invalid", "", "0"])
def test_command_job_rejects_invalid_output_cursor(cursor):
    """Malformed or cross-stream cursors cannot silently restart output."""
    with pytest.raises(ValueError, match="cursor"):
        jobs.CommandJob("session", 3600.0).read_output("stdout", cursor, 10)


@pytest.mark.parametrize("max_bytes", [True, 0, -1, "1"])
def test_command_job_rejects_invalid_output_page_size(max_bytes):
    """A page request must contain a positive byte ceiling."""
    with pytest.raises(ValueError, match="max_bytes"):
        jobs.CommandJob("session", 3600.0).read_output("stdout", None, max_bytes)


@pytest.mark.parametrize(
    "wait",
    [True, -1, float("inf"), float("nan"), "1", jobs.constants.MAX_COMMAND_WAIT_SECONDS + 1],
)
def test_command_job_rejects_unbounded_poll_wait(wait):
    """A poll wait must be finite and shorter than the hard per-call bound."""
    with pytest.raises(ValueError, match="poll"):
        jobs.CommandJob("session", 3600.0).poll(wait)


@pytest.mark.parametrize(
    "maximum",
    [True, -1, float("inf"), float("nan"), "1", jobs.constants.MAX_COMMAND_WAIT_SECONDS + 1],
)
def test_command_job_rejects_invalid_poll_maximum(maximum):
    """A configured maximum outside the supported wait range is rejected."""
    with pytest.raises(ValueError, match="maximum"):
        jobs.CommandJob("session", 3600.0).poll(0, max_wait_seconds=maximum)


def test_unreferenced_inline_output_is_discarded_but_exposed_output_survives():
    """Small inline results release their files while returned handles preserve full streams."""
    inline = jobs.CommandJob("session", 3600.0)
    inline.publish("stdout", "small")
    output = inline.stream_result("stdout", 0)
    inline.release_unreferenced_output({"stdout": output})
    with pytest.raises(ValueError, match="no longer retained"):
        inline.read_output("stdout", None, 10)

    exposed = jobs.CommandJob("session", 3600.0)
    exposed.publish("stdout", "x" * jobs.constants.MAX_TOOL_CONTENT_BYTES)
    result = exposed.stream_result("stdout", 0)
    exposed.release_unreferenced_output(
        Problem(code="test.failed", title="Failed", detail="Failed", metadata={"stdout": result})
    )
    assert exposed.read_output("stdout", None, 10)["content"] == "x" * 10


def test_exposed_command_output_survives_inline_result_cleanup():
    """Once status exposes output handles, result rendering cannot discard their spools."""
    job = jobs.CommandJob("session", 3600.0)
    job.publish("stdout", "retained")
    handles = job.output_handles()
    job.release_unreferenced_output({})
    assert handles["stdout"] == job._outputs["stdout"].handle
    assert job.read_output("stdout", None, 20)["content"] == "retained"


def test_command_jobs_create_distinct_workers_and_bind_handles_to_one_session(workers):
    """Each invocation owns a fresh handle and other sessions cannot inspect either job."""
    registry = jobs.CommandJobs()
    runner = Mock(return_value={"exit_code": 0})
    job = registry.start("session-a", runner, runtime_limit_seconds=3600.0)
    assert isinstance(job, jobs.CommandJob)
    second = registry.start("session-a", runner, runtime_limit_seconds=3600.0)
    assert isinstance(second, jobs.CommandJob)
    assert second.handle != job.handle
    assert len(workers) == 2
    assert isinstance(registry.get("session-b", job.handle), Problem)
    assert isinstance(registry.get("session-a", "unknown"), Problem)
    assert registry.inspect("session-b") == []

    job.admitted()
    workers[0].target()
    workers[1].target()
    assert runner.call_count == 2
    assert registry.get("session-a", job.handle) is job
    assert registry.inspect("session-a")[0]["status"] == "complete"
    assert registry.close()
    assert workers[0].joined
    assert isinstance(registry.start("session-a", runner, runtime_limit_seconds=3600.0), Problem)


def test_close_terminal_keeps_pending_cleanup_and_forgets_only_settled_owner(workers, monkeypatch):
    """Closing a terminal cancels first and preserves an unsettled handle for inspection."""
    registry = jobs.CommandJobs()
    ordinary = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    terminal = registry.start("owner", lambda _job: {}, terminal=True, runtime_limit_seconds=3600.0)
    assert isinstance(registry.close_terminal("other", terminal.handle), Problem)
    assert isinstance(registry.close_terminal("owner", ordinary.handle), Problem)
    monkeypatch.setattr(terminal.finished, "wait", lambda _seconds: False)
    pending = registry.close_terminal("owner", terminal.handle)
    assert pending["closed"] is False
    assert pending["cancel_requested"]
    assert registry.get("owner", terminal.handle) is terminal

    terminal.complete({"exit_code": 0})
    monkeypatch.setattr(terminal.finished, "wait", lambda _seconds: True)
    closed = registry.close_terminal("owner", terminal.handle)
    assert closed["closed"] is True
    assert isinstance(registry.get("owner", terminal.handle), Problem)
    replacement = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    assert isinstance(replacement, jobs.CommandJob)
    assert replacement.handle != terminal.handle


def test_command_jobs_cap_active_and_retained_work(workers):
    """A bounded registry refuses a fifth active process and evicts settled history first."""
    registry = jobs.CommandJobs()
    active = [
        registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0) for _ in range(4)
    ]
    assert all(isinstance(job, jobs.CommandJob) for job in active)
    assert isinstance(
        registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0), Problem
    )
    for index, job in enumerate(active):
        job.complete({"index": index})
    for _ in range(4):
        job = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
        job.complete({})
    newest = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    assert isinstance(newest, jobs.CommandJob)
    assert isinstance(registry.get("owner", active[0].handle), Problem)
    assert len(registry.inspect("owner")) == 8
    assert not registry.close()
    assert newest.cancellation_event.is_set()


def test_command_jobs_expire_without_resolving_old_handles(workers, monkeypatch):
    """Expired handles are unavailable while a new invocation receives a new handle."""
    clock = [0.0]
    monkeypatch.setattr(jobs.time, "monotonic", lambda: clock[0])
    registry = jobs.CommandJobs()
    job = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    job.complete({})
    clock[0] = jobs.JOB_RETENTION_SECONDS

    assert isinstance(registry.get("owner", job.handle), Problem)
    assert registry.inspect("owner") == []
    replacement = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    assert isinstance(replacement, jobs.CommandJob)
    assert replacement.handle != job.handle
    assert len(workers) == 2


def test_process_restart_invalidates_old_job_handles(workers):
    """A fresh registry cannot claim an old process's command or replay its output."""
    original = jobs.CommandJobs()
    job = original.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    job.complete({"exit_code": 0})

    restarted = jobs.CommandJobs()

    assert isinstance(restarted.get("owner", job.handle), Problem)
    assert restarted.inspect("owner") == []


def test_registry_eviction_handles_completion_at_zero_clock(workers, monkeypatch):
    """A zero-valued monotonic completion time remains a valid eviction priority."""
    monkeypatch.setattr(jobs.time, "monotonic", lambda: 0.0)
    registry = jobs.CommandJobs()
    original = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    original.complete({})
    for _ in range(jobs.MAX_RETAINED_JOBS):
        current = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
        current.complete({})
    assert isinstance(registry.get("owner", original.handle), Problem)
    replacement = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    assert isinstance(replacement, jobs.CommandJob)
    assert replacement.handle != original.handle
    assert len(workers) == jobs.MAX_RETAINED_JOBS + 2


def test_registry_reports_active_jobs(workers):
    """A completed job no longer counts as active even while its handle is retained."""
    registry = jobs.CommandJobs()
    assert not registry.has_active()
    job = registry.start("session", lambda _job: {}, runtime_limit_seconds=3600.0)
    assert registry.has_active()
    job.complete({"exit_code": 0})
    assert not registry.has_active()


def test_command_jobs_report_worker_failure_and_start_failure(workers, monkeypatch):
    """Worker exceptions settle the handle, and failed thread startup removes the reservation."""
    logged = Mock()
    monkeypatch.setattr(jobs, "log_problem", logged)
    registry = jobs.CommandJobs()

    def fail(_job):
        """Raise after the job has been reserved."""
        raise RuntimeError("sensitive detail")

    job = registry.start("owner", fail, runtime_limit_seconds=3600.0)
    workers[0].target()
    job.handoff()
    result = job.result()
    assert isinstance(result, Problem)
    assert result.code == "command_job.failed"
    assert "sensitive detail" not in result.detail

    def refused_start(_self):
        """Fail before a worker can own the attempt."""
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(type(workers[0]), "start", refused_start)
    rejected = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    assert isinstance(rejected, Problem)
    assert registry.inspect("owner") == [job.inspect()]
    assert logged.call_count == 2


def test_interrupted_worker_start_cannot_execute_after_reservation_is_removed(workers, monkeypatch):
    """A worker created before a failed handoff cannot run or duplicate effects."""
    registry = jobs.CommandJobs()
    effects = []

    def interrupted_start(_worker):
        """Simulate a started native thread whose start call was interrupted."""
        raise KeyboardInterrupt("thread start interrupted")

    monkeypatch.setattr(jobs.threading.Thread, "start", interrupted_start)
    monkeypatch.setattr(jobs, "log_problem", Mock())
    failed = registry.start(
        "owner", lambda _job: effects.append("effect") or {}, runtime_limit_seconds=3600.0
    )
    assert isinstance(failed, Problem)
    workers[0].target()
    assert effects == []

    monkeypatch.setattr(jobs.threading.Thread, "start", lambda _worker: None)
    retried = registry.start(
        "owner", lambda _job: effects.append("effect") or {}, runtime_limit_seconds=3600.0
    )
    assert isinstance(retried, jobs.CommandJob)
    workers[1].target()
    assert effects == ["effect"]


def test_native_worker_cannot_enter_after_interrupted_start(monkeypatch):
    """A real worker observes a failed start before it can enter the command."""
    registry = jobs.CommandJobs()
    effects = []
    started = []
    original_start = jobs.threading.Thread.start

    def interrupt_after_start(worker):
        """Interrupt the caller after the native thread has been created."""
        started.append(worker)
        original_start(worker)
        raise KeyboardInterrupt("thread start interrupted")

    monkeypatch.setattr(jobs.threading.Thread, "start", interrupt_after_start)
    monkeypatch.setattr(jobs, "log_problem", Mock())
    failed = registry.start(
        "owner", lambda _job: effects.append("effect") or {}, runtime_limit_seconds=3600.0
    )
    assert isinstance(failed, Problem)
    started[0].join(timeout=5)
    assert not started[0].is_alive()
    assert effects == []


def test_start_failure_and_close_during_thread_handoff(workers, monkeypatch):
    """Startup failure releases capacity and close tolerates a handoff race."""
    registry = jobs.CommandJobs()

    def refused_start(_self):
        """Simulate a worker that never acquired ownership."""
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(
        type(workers[0]) if workers else jobs.threading.Thread, "start", refused_start
    )
    monkeypatch.setattr(jobs, "log_problem", Mock())
    assert isinstance(
        registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0), Problem
    )
    assert registry.inspect("owner") == []
    monkeypatch.setattr(type(workers[0]), "start", lambda _self: None)
    active = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)
    active.thread = None
    assert not registry.close()
    assert active.cancellation_event.is_set()


def test_worker_starts_before_shutdown_can_snapshot_it(workers, monkeypatch):
    """A close cannot join a published command thread before its start call returns."""
    registry = jobs.CommandJobs()
    observed = []

    def start(_self):
        """Check that another thread cannot acquire the publication lock."""

        def inspect_lock():
            """Probe the lock from outside the starting thread."""
            acquired = registry._lock.acquire(blocking=False)
            observed.append(not acquired)
            if acquired:
                registry._lock.release()

        checker = Thread(target=inspect_lock)
        checker.start()
        checker.join()

    monkeypatch.setattr(type(workers[0]) if workers else jobs.threading.Thread, "start", start)
    job = registry.start("owner", lambda _job: {}, runtime_limit_seconds=3600.0)

    assert isinstance(job, jobs.CommandJob)
    assert observed == [True]
