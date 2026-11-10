"""Verify bounded command continuations and process-local session ownership."""

from threading import Thread
from unittest.mock import Mock

import pytest

from loop.errors import Problem
from loop.execution import jobs


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


def test_command_job_pages_ordered_output_and_exposes_settled_result():
    """A cursor advances over bounded events without reusing earlier output."""
    job = jobs.CommandJob("session", "call")
    job.admitted()
    job.publish("stdout", "a" * (jobs.MAX_OUTPUT_EVENT_CHARS + 1))
    job.publish("stderr", "err")

    first = job.read(0)
    second = job.read(1)
    assert first["status"] == "running"
    assert [event["cursor"] for event in first["events"]] == [0, 1, 2]
    assert first["next_cursor"] == 3
    assert [event["stream"] for event in second["events"]] == ["stdout", "stderr"]
    assert second["next_cursor"] == 3

    job.cancel()
    job.complete({"exit_code": 0})
    settled = job.read(3)
    assert settled["status"] == "complete"
    assert settled["cancel_requested"]
    assert settled["events"] == []
    assert settled["result"] == {"exit_code": 0}


def test_completed_job_ignores_late_output():
    """A completed job cannot acquire events from a late pipe reader."""
    job = jobs.CommandJob("session", "call")
    job.publish("stdout", "before")
    job.complete({"exit_code": 0})
    before = job.read(0)
    job.publish("stdout", "after")
    job.publish("stderr", "after")
    assert job.read(0) == before


def test_fragmented_output_coalesces_without_changing_delivered_events():
    """Small writes remain bounded and a cursor never skips later appended output."""
    job = jobs.CommandJob("session", None)
    for _ in range(10_000):
        job.publish("stdout", "x")
    first = job.read(0)
    assert len(first["events"]) == 4
    assert all(len(event["content"]) == jobs.MAX_OUTPUT_EVENT_CHARS for event in first["events"])
    cursor = first["next_cursor"]
    job.publish("stdout", "end")
    collected = "".join(event["content"] for event in first["events"])
    while cursor < job.inspect()["next_cursor"]:
        page = job.read(cursor)
        collected += "".join(event["content"] for event in page["events"])
        cursor = page["next_cursor"]
    assert collected == "x" * 10_000 + "end"
    job.publish("stdout", "after-read")
    page = job.read(cursor)
    assert page["events"][0]["content"] == "after-read"


def test_alternating_output_bounds_events_and_reports_loss(monkeypatch):
    """Alternating streams cannot bypass the event limit or silently lose output."""
    monkeypatch.setattr(jobs, "MAX_JOB_OUTPUT_EVENTS", 2)
    job = jobs.CommandJob("session", None)
    job.publish("stdout", "a")
    job.publish("stderr", "b")
    job.publish("stdout", "c")
    expired = job.read(0)
    assert expired.code == "command_job.cursor_expired"
    assert expired.metadata["oldest_cursor"] == 1
    assert job.inspect()["dropped_characters"] == 1
    assert [event["content"] for event in job.read(1)["events"]] == ["b", "c"]


@pytest.mark.parametrize("completed", [False, True])
def test_future_cursor_is_rejected_without_skipping_output(completed):
    """A future cursor cannot hide retained or subsequently published output."""
    job = jobs.CommandJob("session", None)
    job.publish("stdout", "first")
    if completed:
        job.complete({"exit_code": 0})
    with pytest.raises(ValueError, match="beyond the produced output"):
        job.read(2)
    page = job.read(0)
    assert page["events"][0]["content"] == "first"
    assert (page["result"] is not None) is completed
    if not completed:
        job.publish("stdout", "second")
        assert job.read(page["next_cursor"])["events"][0]["content"] == "second"


def test_settled_result_waits_until_all_retained_pages_are_delivered():
    """A completed job does not signal a final result before its output is paged."""
    job = jobs.CommandJob("session", None)
    for index in range(jobs.MAX_READ_EVENTS + 1):
        job.publish("stdout" if index % 2 == 0 else "stderr", str(index))
    job.complete({"exit_code": 0})
    first = job.read(0)
    assert first["status"] == "complete"
    assert first["result"] is None
    last = job.read(first["next_cursor"])
    assert last["result"] == {"exit_code": 0}
    assert last["events"][-1]["content"] == str(jobs.MAX_READ_EVENTS)


@pytest.mark.parametrize("failure", [False, True])
def test_settled_output_previews_are_available_only_on_request(failure):
    """Pages avoid repeating captured output while retaining explicit full-result access."""
    job = jobs.CommandJob("session", None)
    job.publish("stdout", "visible")
    output = {"stdout": {"content": "visible", "captured_bytes": 7, "tail": {"content": "e"}}}
    result = (
        Problem(code="test.failed", title="Failed", detail="Failed", metadata=output)
        if failure
        else output
    )
    job.complete(result)
    page = job.read(0)
    summary = page["result"]["metadata"] if failure else page["result"]
    assert summary["stdout"] == {"captured_bytes": 7}
    full = job.read(page["next_cursor"], include_output=True)["result"]
    full = full["metadata"] if failure else full
    assert full == output


def test_settled_result_preserves_non_mapping_metadata():
    """Results with no stream metadata remain available without transformation."""
    job = jobs.CommandJob("session", None)
    job.complete({"metadata": None})
    assert job.read(0)["result"] == {"metadata": None}


def test_terminal_job_bounds_transient_controls_and_escapes_output():
    """An owned terminal queues exact controls and renders unsafe output without retaining input."""
    job = jobs.CommandJob("session", "call", terminal=True)
    job.admitted()
    job.write_terminal("é\n")
    job.resize_terminal(30, 100)
    job.terminal_eof()
    job.terminal_interrupt()
    job.publish("stdout", "ready\x1b]52;c;secret\x07\r\n")

    assert job.inspect()["terminal"]
    assert job.read(0)["events"][0]["content"] == "ready\u241b]52;c;secret\u2407\u240d\n"
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
    job = jobs.CommandJob("session", None, terminal=True)
    with pytest.raises((TypeError, ValueError), match="input"):
        job.write_terminal(content)


@pytest.mark.parametrize(("rows", "columns"), [(0, 80), (24, 501), (True, 80), (24, "80")])
def test_terminal_job_refuses_invalid_window_sizes(rows, columns):
    """Malformed dimensions cannot reach a native terminal ioctl."""
    job = jobs.CommandJob("session", None, terminal=True)
    with pytest.raises(ValueError, match="rows and columns"):
        job.resize_terminal(rows, columns)


def test_terminal_job_refuses_nonterminal_control_and_queue_overflow(monkeypatch):
    """Normal jobs have no control channel and terminal input applies bounded backpressure."""
    ordinary = jobs.CommandJob("session", None)
    with pytest.raises(ValueError, match="unavailable"):
        ordinary.terminal_interrupt()
    terminal = jobs.CommandJob("session", None, terminal=True)
    monkeypatch.setattr(jobs, "MAX_TERMINAL_PENDING_BYTES", 4)
    with pytest.raises(ValueError, match="backed up"):
        terminal.write_terminal("ab")
    terminal.cancel()
    assert terminal.next_terminal_frame() is None
    with pytest.raises(ValueError, match="unavailable"):
        terminal.write_terminal("a")


def test_settled_job_rejects_late_admission_completion_and_cancel():
    """A finished job cannot be relabeled or cancelled by a late callback."""
    job = jobs.CommandJob("session", None)
    job.complete({"exit_code": 0})
    job.admitted()
    job.complete({"exit_code": 7})
    job.cancel()
    assert job.inspect()["status"] == "complete"
    assert not job.inspect()["cancel_requested"]
    assert job.read(0)["result"] == {"exit_code": 0}


def test_command_job_expired_cursor_and_structured_failure(monkeypatch):
    """A bounded ring reports lost earlier events and keeps a safe final failure."""
    monkeypatch.setattr(jobs, "MAX_JOB_OUTPUT_BYTES", 4)
    job = jobs.CommandJob("session", None)
    job.publish("stdout", "abcd")
    job.publish("stderr", "ef")
    expired = job.read(0)
    assert isinstance(expired, Problem)
    assert expired.code == "command_job.cursor_expired"
    assert expired.metadata == {"oldest_cursor": 1}
    assert job.inspect()["dropped_characters"] == 4

    failure = Problem(code="test.failed", title="Failed", detail="Safe failure")
    job.complete(failure)
    assert job.read(1)["result"] == failure.model_dump()


@pytest.mark.parametrize("stream", ["other", "", "STDOUT"])
def test_command_job_rejects_unknown_output_stream(stream):
    """Only the two authenticated process streams may enter a job page."""
    with pytest.raises(ValueError, match="output stream"):
        jobs.CommandJob("session", None).publish(stream, "x")


@pytest.mark.parametrize("cursor", [-1, True, "0"])
def test_command_job_rejects_invalid_cursor(cursor):
    """Malformed cursors cannot silently restart output from zero."""
    with pytest.raises(ValueError, match="cursor"):
        jobs.CommandJob("session", None).read(cursor)


@pytest.mark.parametrize("wait", [True, -1, 31, float("nan"), "1"])
def test_command_job_rejects_unbounded_wait(wait):
    """A caller cannot hold a read beyond the fixed wait budget."""
    with pytest.raises(ValueError, match="wait"):
        jobs.CommandJob("session", None).read(0, wait)


def test_command_jobs_bind_handles_and_model_calls_to_one_session(workers):
    """A duplicate call reuses one worker while another session cannot inspect it."""
    registry = jobs.CommandJobs()
    runner = Mock(return_value={"exit_code": 0})
    job = registry.start("session-a", "call-1", runner)
    assert isinstance(job, jobs.CommandJob)
    assert registry.start("session-a", "call-1", runner) is job
    assert len(workers) == 1
    assert isinstance(registry.get("session-b", job.handle), Problem)
    assert isinstance(registry.get("session-a", "unknown"), Problem)
    assert registry.inspect("session-b") == []

    job.admitted()
    workers[0].target()
    assert runner.call_count == 1
    assert registry.get("session-a", job.handle) is job
    assert registry.inspect("session-a")[0]["status"] == "complete"
    assert registry.close()
    assert workers[0].joined
    assert isinstance(registry.start("session-a", None, runner), Problem)


def test_close_terminal_keeps_pending_cleanup_and_forgets_only_settled_owner(workers, monkeypatch):
    """Closing a terminal cancels first and preserves an unsettled handle for inspection."""
    registry = jobs.CommandJobs()
    ordinary = registry.start("owner", None, lambda _job: {})
    terminal = registry.start("owner", "call", lambda _job: {}, terminal=True)
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
    assert isinstance(registry.start("owner", "call", lambda _job: {}), Problem)


def test_command_jobs_cap_active_and_retained_work(workers):
    """A bounded registry refuses a fifth active process and evicts settled history first."""
    registry = jobs.CommandJobs()
    active = [registry.start("owner", None, lambda _job: {}) for _ in range(4)]
    assert all(isinstance(job, jobs.CommandJob) for job in active)
    assert isinstance(registry.start("owner", None, lambda _job: {}), Problem)
    for index, job in enumerate(active):
        job.complete({"index": index})
    for _ in range(4):
        job = registry.start("owner", None, lambda _job: {})
        job.complete({})
    newest = registry.start("owner", None, lambda _job: {})
    assert isinstance(newest, jobs.CommandJob)
    assert isinstance(registry.get("owner", active[0].handle), Problem)
    assert len(registry.inspect("owner")) == 8
    assert not registry.close()
    assert newest.cancellation_event.is_set()


def test_command_jobs_expire_without_replaying_a_model_call(workers, monkeypatch):
    """An expired handle or repeated model call cannot launch source again."""
    clock = [0.0]
    monkeypatch.setattr(jobs.time, "monotonic", lambda: clock[0])
    registry = jobs.CommandJobs()
    job = registry.start("owner", "call", lambda _job: {})
    job.complete({})
    clock[0] = jobs.JOB_RETENTION_SECONDS

    assert isinstance(registry.get("owner", job.handle), Problem)
    assert isinstance(registry.start("owner", "call", lambda _job: {}), Problem)
    assert registry.inspect("owner") == []
    assert len(workers) == 1


def test_process_restart_invalidates_old_job_handles(workers):
    """A fresh registry cannot claim an old process's command or replay its output."""
    original = jobs.CommandJobs()
    job = original.start("owner", "call", lambda _job: {})
    job.complete({"exit_code": 0})

    restarted = jobs.CommandJobs()

    assert isinstance(restarted.get("owner", job.handle), Problem)
    assert restarted.inspect("owner") == []


def test_registry_eviction_handles_completion_at_zero_clock(workers, monkeypatch):
    """A zero-valued monotonic completion time remains a valid eviction priority."""
    monkeypatch.setattr(jobs.time, "monotonic", lambda: 0.0)
    registry = jobs.CommandJobs()
    original = registry.start("owner", "old", lambda _job: {})
    original.complete({})
    for _ in range(jobs.MAX_RETAINED_JOBS):
        current = registry.start("owner", None, lambda _job: {})
        current.complete({})
    assert isinstance(registry.get("owner", original.handle), Problem)
    assert isinstance(registry.start("owner", "old", lambda _job: {}), Problem)
    assert len(workers) == jobs.MAX_RETAINED_JOBS + 1


def test_command_jobs_bound_remembered_model_calls(workers, monkeypatch):
    """A bounded identity ledger refuses new calls without permitting old replay."""
    monkeypatch.setattr(jobs, "MAX_REMEMBERED_CALL_IDS", 1)
    registry = jobs.CommandJobs()
    original = registry.start("owner", "first", lambda _job: {})
    original.complete({})

    assert registry.start("owner", "first", lambda _job: {}) is original
    assert isinstance(registry.start("owner", "second", lambda _job: {}), Problem)
    assert len(workers) == 1


def test_command_jobs_report_worker_failure_and_start_failure(workers, monkeypatch):
    """Worker exceptions settle the handle, and failed thread startup removes the reservation."""
    logged = Mock()
    monkeypatch.setattr(jobs, "log_problem", logged)
    registry = jobs.CommandJobs()

    def fail(_job):
        """Raise after the job has been reserved."""
        raise RuntimeError("sensitive detail")

    job = registry.start("owner", None, fail)
    workers[0].target()
    assert job.read(0)["result"]["code"] == "command_job.failed"
    assert "sensitive detail" not in job.read(0)["result"]["detail"]

    def refused_start(_self):
        """Fail before a worker can own the attempt."""
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(type(workers[0]), "start", refused_start)
    rejected = registry.start("owner", "refused", lambda _job: {})
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
    failed = registry.start("owner", "call", lambda _job: effects.append("effect") or {})
    assert isinstance(failed, Problem)
    workers[0].target()
    assert effects == []

    monkeypatch.setattr(jobs.threading.Thread, "start", lambda _worker: None)
    retried = registry.start("owner", "call", lambda _job: effects.append("effect") or {})
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
    failed = registry.start("owner", "call", lambda _job: effects.append("effect") or {})
    assert isinstance(failed, Problem)
    started[0].join(timeout=5)
    assert not started[0].is_alive()
    assert effects == []


def test_start_failure_without_call_id_and_close_during_thread_handoff(workers, monkeypatch):
    """Anonymous startup failure releases capacity and close tolerates a handoff race."""
    registry = jobs.CommandJobs()

    def refused_start(_self):
        """Simulate a worker that never acquired ownership."""
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(
        type(workers[0]) if workers else jobs.threading.Thread, "start", refused_start
    )
    monkeypatch.setattr(jobs, "log_problem", Mock())
    assert isinstance(registry.start("owner", None, lambda _job: {}), Problem)
    assert registry.inspect("owner") == []
    monkeypatch.setattr(type(workers[0]), "start", lambda _self: None)
    active = registry.start("owner", None, lambda _job: {})
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
    job = registry.start("owner", None, lambda _job: {})

    assert isinstance(job, jobs.CommandJob)
    assert observed == [True]
