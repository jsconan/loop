"""Tests for passive tool declaration, validation, and dispatch."""

import asyncio
import gc
import importlib
import inspect
import json
import warnings
from contextvars import ContextVar
from dataclasses import FrozenInstanceError
from threading import Event, get_ident
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from loop import (
    Action,
    Operation,
    OperationPlan,
    Problem,
    ProblemException,
    SessionTarget,
    ToolRegistrationError,
    ToolRegistry,
    ToolResultPresentation,
    ToolResultPresentationSpec,
    tool,
)
from loop.tooling import (
    OwnedToolTask,
    OwnedToolTaskCancelled,
    PlatformArgumentCapability,
    Tool,
    ToolContext,
    ToolExecutionAdmissionRejected,
    ToolPreflightResult,
    ToolStatus,
)
from loop.tooling.models import ToolExecutionOutcomeUnknown
from loop.utils import await_value_async
from tests.scheduling import deliver_callbacks

tool_module = importlib.import_module("loop.tooling.tool")


class Arguments(BaseModel):
    """Validate the integer argument used by tool tests."""

    number: int


class DerivedCancellation(asyncio.CancelledError):
    """Represent a specialized cancellation delivered through the coroutine protocol."""


def make_tool(function=None) -> Tool:
    """Build a tool with the shared argument model."""
    if function is None:
        function = Mock(return_value=2)
    return Tool(
        function=function,
        name="calculate",
        description="Calculate a value.",
        arguments_model=Arguments,
    )


def test_owned_tool_task_rejects_invalid_or_started_coroutines():
    """Ownership can only be established before a native coroutine begins."""
    with pytest.raises(TypeError, match="native coroutine"):
        OwnedToolTask("not a coroutine")

    async def calculate():
        """Suspend to expose the started coroutine state."""
        await asyncio.sleep(0)

    coroutine = calculate()
    try:
        coroutine.send(None)
        with pytest.raises(ValueError, match="unstarted coroutine"):
            OwnedToolTask(coroutine)
    finally:
        coroutine.close()


def test_owned_tool_task_closes_unstarted_work():
    """Canceling an unstarted owned coroutine prevents it from running."""

    async def calculate():
        """Represent work that must remain unstarted."""
        return "unexpected"

    work = OwnedToolTask(calculate())
    work.cancel()
    assert work.task is None
    with asyncio.Runner() as runner, pytest.raises(RuntimeError, match="closed"):
        work.start(runner.get_loop())


def test_owned_task_cancelled_before_first_turn_closes_coroutine():
    """Cancellation before the first task turn releases the original coroutine."""

    async def scenario():
        async def calculate():
            return "unused"

        coroutine = calculate()
        work = OwnedToolTask(coroutine)
        task = work.start(asyncio.get_running_loop())
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await deliver_callbacks()
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED

    asyncio.run(scenario())


def test_sync_tool_refuses_running_loop_before_invocation():
    """Direct synchronous dispatch cannot start a task inside an active loop."""
    calls = []

    def calculate(number):
        calls.append(number)
        return asyncio.create_task(asyncio.sleep(0))

    async def scenario():
        result = make_tool(calculate).execute({"number": 3})
        assert json.loads(result.output)["problem"]["code"] == "tool.async_required"
        await deliver_callbacks()
        assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["native", "owned", "borrowed"])
def test_runner_failure_preserves_settled_tool_receipt(kind):
    """A recoverable runner failure does not discard an effect's settled result."""
    effects = []

    async def effect(number):
        try:
            asyncio.get_running_loop().stop()
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        effects.append(number)
        return "confirmed"

    with asyncio.Runner() as runner:

        def calculate(number):
            if kind == "owned":
                return OwnedToolTask(effect(number))
            return runner.get_loop().create_task(effect(number))

        function = effect if kind == "native" else calculate
        result = make_tool(function).execute({"number": 3}, async_runner=runner)

    assert json.loads(result.output)["result"] == "confirmed"
    assert effects == [3]


@pytest.mark.parametrize("adapter_factory", [False, True])
def test_sync_runner_preserves_declared_final_owned_receipt(adapter_factory):
    """Synchronous dispatch retains an explicitly final result after runner failure."""
    effects = []

    async def effect():
        """Interrupt the runner, then finish the owned operation."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            effects.append("committed")
            return "confirmed"

    async def calculate(number):
        """Declare the awaited operation as this tool's final outcome."""
        return await OwnedToolTask(effect(), final_result=True)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        if adapter_factory:

            def factory(owner_loop, coroutine, **kwargs):
                """Adapt runner dispatch through a distinct inner task."""

                async def adapter():
                    """Forward the inner task's completion."""
                    return await asyncio.Task(coroutine, loop=owner_loop)

                return asyncio.Task(adapter(), loop=owner_loop, **kwargs)

            loop.set_task_factory(factory)
        try:
            result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
        finally:
            loop.set_task_factory(None)
    assert json.loads(result.output)["result"] == "confirmed"
    assert effects == ["committed"]


def test_interrupted_sync_final_result_does_not_start_nested_work():
    """A sync runner failure cannot admit a deferred child returned during cleanup."""
    effects = []

    async def deferred():
        """Record incorrectly admitted work."""
        effects.append("deferred")
        return "unexpected"

    async def child():
        """Return unstarted owned work after the runner is interrupted."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return OwnedToolTask(deferred())

    async def calculate(number):
        """Declare the child as the tool's final outcome."""
        return await OwnedToolTask(child(), final_result=True)

    with asyncio.Runner() as runner, pytest.raises(ToolExecutionOutcomeUnknown):
        make_tool(calculate).execute({"number": 1}, async_runner=runner)
    assert effects == []


def test_interrupted_sync_result_does_not_start_nested_work():
    """A normally returned value cannot admit deferred work after interruption."""
    effects = []

    async def deferred():
        """Record incorrectly admitted work."""
        effects.append("deferred")
        return "unexpected"

    async def calculate(number):
        """Return unstarted owned work while settling interruption."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return OwnedToolTask(deferred())

    with asyncio.Runner() as runner, pytest.raises(ToolExecutionOutcomeUnknown):
        make_tool(calculate).execute({"number": 1}, async_runner=runner)
    assert effects == []


def test_interrupted_sync_final_result_preserves_started_coroutine_owner():
    """An already-started returned coroutine remains owned by its original task."""
    effects = []
    tasks = []

    async def child():
        """Start borrowed work, then interrupt the runner and return its coroutine."""
        release = asyncio.Event()
        started = asyncio.Event()

        async def borrowed():
            """Finish only after the original owner releases the task."""
            started.set()
            await release.wait()
            effects.append("borrowed complete")

        coroutine = borrowed()
        tasks.append((asyncio.create_task(coroutine), release))
        await started.wait()
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return coroutine

    async def calculate(number):
        """Declare the child's outcome as the entire tool result."""
        return await OwnedToolTask(child(), final_result=True)

    with asyncio.Runner() as runner:
        with pytest.raises(ToolExecutionOutcomeUnknown):
            make_tool(calculate).execute({"number": 1}, async_runner=runner)
        task, release = tasks[0]
        release.set()
        runner.run(await_value_async(task))
    assert effects == ["borrowed complete"]


def test_interrupted_sync_result_preserves_foreign_loop_owned_task():
    """An interrupted result cannot claim work on another event loop."""

    async def foreign():
        """Return a value to the separate owner."""
        return "foreign"

    with asyncio.Runner() as runner, asyncio.Runner() as owner:
        handle = OwnedToolTask(foreign())
        task = handle.start(owner.get_loop())

        async def calculate(number):
            """Return the foreign handle after interrupting this runner."""
            try:
                asyncio.get_running_loop().stop()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return handle

        with pytest.raises(ToolExecutionOutcomeUnknown):
            make_tool(calculate).execute({"number": 1}, async_runner=runner)
        assert not task.cancelled()
        assert owner.run(await_value_async(task)) == "foreign"


def test_keyboard_interrupt_remains_an_interruption_for_deferred_result():
    """A deferred result cannot turn Ctrl+C into a completed tool result."""
    effects = []

    async def deferred():
        """Record any incorrectly started deferred work."""
        effects.append("entered")

    async def child():
        """Return deferred work during cancellation of a final-result task."""
        try:

            def interrupt():
                """Interrupt the running owner loop."""
                raise KeyboardInterrupt

            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return OwnedToolTask(deferred())

    async def calculate(number):
        """Treat the child as the enclosing tool's final outcome."""
        return await OwnedToolTask(child(), final_result=True)

    with asyncio.Runner() as runner, pytest.raises(KeyboardInterrupt) as captured:
        make_tool(calculate).execute({"number": 1}, async_runner=runner)
    assert not hasattr(captured.value, "output")
    assert effects == []


def test_interrupted_sync_final_nested_admission_failure_is_execution_failure(caplog):
    """A nested failed future cannot make admitted work appear unstarted."""
    effects = []

    async def child():
        """Return a failed nested future after committing an effect."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            effects.append("committed")
            future = asyncio.get_running_loop().create_future()
            future.set_exception(ToolExecutionAdmissionRejected())
            return future

    async def calculate(number):
        """Declare the child as the final outcome."""
        return await OwnedToolTask(child(), final_result=True)

    with asyncio.Runner() as runner:
        result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
    assert effects == ["committed"]
    assert any(
        record.__dict__.get("error.type") == "tool.execution_failed" for record in caplog.records
    )


@pytest.mark.parametrize("distinct_task", [False, True])
def test_sync_runner_uses_adapter_override_after_final_cancellation(distinct_task):
    """A settled outer task's result takes precedence over its inner final marker."""

    async def child():
        """Finish with the inner result after stopping the loop."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "child result"

    async def calculate(number):
        """Await a declared final child."""
        return await OwnedToolTask(child(), final_result=True)

    def factory(loop, coroutine, **kwargs):
        """Override the inner cancellation at the runner boundary."""

        async def adapter():
            """Return the actual outer task's result."""
            try:
                return (
                    await asyncio.Task(coroutine, loop=loop) if distinct_task else await coroutine
                )
            except OwnedToolTaskCancelled:
                return "adapter result"

        return asyncio.Task(adapter(), loop=loop, **kwargs)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_task_factory(factory)
        try:
            result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
        finally:
            loop.set_task_factory(None)
    assert json.loads(result.output)["result"] == "adapter result"


@pytest.mark.parametrize("case", ["cycle", "foreign", "owned_cancelled", "owned_completed"])
def test_interrupted_final_result_settles_only_started_same_loop_work(case):
    """Interrupted final work cannot take authority from another loop."""
    events = []
    with asyncio.Runner() as runner, asyncio.Runner() as foreign:
        foreign_future = foreign.get_loop().create_future()

        async def child():
            """Return the requested nested value after interruption."""
            try:
                asyncio.get_running_loop().stop()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if case == "cycle":
                    cycle = asyncio.get_running_loop().create_future()
                    cycle.set_result(cycle)
                    return cycle
                if case == "foreign":
                    return foreign_future
                ready = asyncio.Event()

                async def background():
                    """Settle the already-started operation after cancellation."""
                    ready.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        events.append("cancelled")
                        if case == "owned_completed":
                            return "settled"
                        raise

                work = OwnedToolTask(background())
                work.start(asyncio.get_running_loop())
                await ready.wait()
                return work

        async def calculate(number):
            """Return a declared final operation."""
            return await OwnedToolTask(child(), final_result=True)

        try:
            if case == "owned_completed":
                result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
                payload = json.loads(result.output)
                assert payload["result"] == "settled"
            else:
                with pytest.raises(ToolExecutionOutcomeUnknown):
                    make_tool(calculate).execute({"number": 1}, async_runner=runner)
            if case.startswith("owned_"):
                assert events == ["cancelled"]
            assert not foreign_future.cancelled()
        finally:
            foreign_future.cancel()


@pytest.mark.parametrize("started", [False, True])
def test_sync_interruption_preserves_preclaimed_owned_work(started):
    """Interrupted dispatch cannot close or cancel another dispatcher's owned work."""
    effects = []
    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        release = asyncio.Event()
        ready = asyncio.Event()

        async def foreign():
            """Finish only after the original owner releases this work."""
            ready.set()
            await release.wait()
            effects.append("foreign completed")
            return "foreign result"

        handle = OwnedToolTask(foreign())
        if started:
            task = handle.claim(loop)
        else:
            handle.reserve()
            task = None

        async def calculate(number):
            """Return work already owned by a different dispatcher."""
            if started:
                await ready.wait()
            try:
                effects.append("tool entered")
                loop.stop()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return handle

        with pytest.raises(ToolExecutionOutcomeUnknown):
            make_tool(calculate).execute({"number": 1}, async_runner=runner)
        if task is None:
            task = handle.start(loop)
        assert not task.cancelling()
        release.set()
        assert runner.run(await_value_async(task)) == "foreign result"
    assert effects == ["tool entered", "foreign completed"]


@pytest.mark.parametrize("host", ["absent", "factory_failure", "runner_failure"])
def test_unavailable_sync_host_preserves_preclaimed_owned_coroutine(host):
    """An unavailable synchronous host cannot close another owner's unstarted work."""
    effects = []

    async def foreign():
        """Record execution when the original owner starts the coroutine."""
        effects.append("foreign completed")
        return "foreign result"

    handle = OwnedToolTask(foreign())
    handle.reserve()

    def failed_factory():
        """Represent an unavailable runner factory."""
        raise RuntimeError("runner unavailable")

    if host == "runner_failure":
        runner = Mock()
        runner.get_loop.side_effect = RuntimeError("runner unavailable")
        with pytest.raises(ToolExecutionOutcomeUnknown):
            make_tool().resolve_awaitable(handle, runner)
    else:
        with pytest.raises(ToolExecutionOutcomeUnknown):
            make_tool(lambda number: handle).execute(
                {"number": 1},
                async_runner_factory=failed_factory if host == "factory_failure" else None,
            )

    with asyncio.Runner() as runner:
        task = handle.start(runner.get_loop())
        assert runner.run(await_value_async(task)) == "foreign result"
    assert effects == ["foreign completed"]


@pytest.mark.parametrize("failure", [ValueError, ToolExecutionAdmissionRejected])
def test_sync_final_child_failure_survives_distinct_adapter(failure, caplog):
    """Settled final failures remain tool failures across an adapter task."""

    async def child():
        """Fail after interruption of the owned operation."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise failure("settled child failure")

    async def calculate(number):
        """Declare the child outcome as the final result."""
        return await OwnedToolTask(child(), final_result=True)

    def factory(loop, coroutine, **kwargs):
        """Use a distinct task to adapt the runner coroutine."""

        async def adapter():
            """Forward the inner task's terminal state."""
            return await asyncio.Task(coroutine, loop=loop)

        return asyncio.Task(adapter(), loop=loop, **kwargs)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_task_factory(factory)
        try:
            result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
        finally:
            loop.set_task_factory(None)
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
    assert any(
        record.__dict__.get("error.type") == "tool.execution_failed" for record in caplog.records
    )


def test_sync_adapter_admission_error_after_execution_is_not_an_admission_rejection(caplog):
    """A task adapter cannot relabel admitted work as unstarted."""

    async def child():
        """Settle a declared result after interruption."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "confirmed"

    async def calculate(number):
        """Await the declared final child."""
        return await OwnedToolTask(child(), final_result=True)

    def factory(loop, coroutine, **kwargs):
        """Convert the final cancellation to an adapter failure."""

        async def adapter():
            """Reject after admitted work has already finished."""
            try:
                return await coroutine
            except OwnedToolTaskCancelled as error:
                raise ToolExecutionAdmissionRejected from error

        return asyncio.Task(adapter(), loop=loop, **kwargs)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_task_factory(factory)
        try:
            result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
        finally:
            loop.set_task_factory(None)
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
    assert any(
        record.__dict__.get("error.type") == "tool.execution_failed" for record in caplog.records
    )


@pytest.mark.parametrize("outcome", ["success", "failure", "admission"])
def test_sync_returned_owned_task_retains_nested_final_outcome(outcome, caplog):
    """A returned owned task retains its nested final result or settled failure."""
    effects = []

    async def child():
        """Settle after the synchronous runner is interrupted."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            effects.append("completed")
            if outcome == "failure":
                raise ValueError("settled failure")
            if outcome == "admission":
                raise ToolExecutionAdmissionRejected("settled admission failure")
            return "confirmed"

    async def wrapper():
        """Transfer a nested final outcome through the returned task."""
        return await OwnedToolTask(child(), final_result=True)

    def calculate(number):
        """Return explicitly owned work to the synchronous runner."""
        return OwnedToolTask(wrapper())

    with asyncio.Runner() as runner:
        result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
    payload = json.loads(result.output)
    if outcome == "success":
        assert payload["result"] == "confirmed"
    else:
        assert payload["problem"]["code"] == "tool.execution_failed"
        assert any(
            record.__dict__.get("error.type") == "tool.execution_failed"
            for record in caplog.records
        )
    assert effects == ["completed"]


def test_sync_callback_cannot_promote_a_final_marker_without_invocation():
    """A callback without a tool invocation cannot publish a final tool receipt."""

    async def child():
        """Settle an otherwise declared final result after interruption."""
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "callback result"

    async def callback():
        """Await work outside a tool invocation."""
        return await OwnedToolTask(child(), final_result=True)

    with (
        asyncio.Runner() as runner,
        pytest.raises(RuntimeError, match="Event loop stopped before Future completed"),
    ):
        make_tool().resolve_awaitable(callback(), runner)


@pytest.mark.parametrize("kind", ["native", "returned_owned"])
def test_sync_runner_rejects_foreign_final_marker(kind, caplog):
    """A marker from a pre-existing task cannot authorize a new tool receipt."""

    async def child(started, release):
        """Produce a foreign marker after the runner interrupts the caller."""
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            return "foreign result"

    async def unrelated(started, release):
        """Declare a final outcome outside the tool's invocation."""
        return await OwnedToolTask(child(started, release), final_result=True)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        started, release = asyncio.Event(), asyncio.Event()
        if kind == "returned_owned":
            work = OwnedToolTask(unrelated(started, release))
            task = work.start(loop)
        else:
            work = None
            task = loop.create_task(unrelated(started, release))
        runner.run(started.wait())

        async def calculate(number):
            """Interrupt execution and await a pre-existing task."""
            loop.call_soon(loop.stop)
            return await task

        def returned_owned(number):
            """Interrupt execution and return a pre-existing owned handle."""
            loop.call_soon(loop.stop)
            return work

        try:
            result = make_tool(calculate if kind == "native" else returned_owned).execute(
                {"number": 1}, async_runner=runner
            )
            assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
            assert any(
                record.__dict__.get("error.type") == "tool.execution_failed"
                for record in caplog.records
            )
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    runner.run(await_value_async(task))


async def failing_admission_awaitable():
    """Raise an admission rejection from an awaited tool result."""
    raise ToolExecutionAdmissionRejected("admission not reached")


def test_async_awaited_admission_rejection_propagates_before_interruption():
    """An admission rejection without interruption remains a rejection."""

    async def calculate(number):
        """Return a rejection through an awaited child."""
        return await failing_admission_awaitable()

    with pytest.raises(ToolExecutionAdmissionRejected, match="admission not reached"):
        asyncio.run(make_tool(calculate).execute_async({"number": 1}))


@pytest.mark.parametrize("declared_final", [False, True])
def test_sync_keyboard_interrupt_retains_only_declared_final_receipt(declared_final):
    """A confirmed final outcome survives interruption without resuming the tool."""
    effects = []

    async def child():
        """Settle a side effect after the runner is interrupted."""
        try:

            def interrupt():
                """Interrupt on the event-loop thread."""
                raise KeyboardInterrupt

            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            effects.append("committed")
            return "confirmed"

    async def calculate(number):
        """Await the child, optionally declaring the whole tool result."""
        return await OwnedToolTask(child(), final_result=declared_final)

    with asyncio.Runner() as runner, pytest.raises(KeyboardInterrupt) as caught:
        make_tool(calculate).execute({"number": 1}, async_runner=runner)
    assert effects == ["committed"]
    if declared_final:
        assert type(caught.value) is not KeyboardInterrupt
        assert json.loads(caught.value.output)["result"] == "confirmed"
    else:
        assert not hasattr(caught.value, "output")


@pytest.mark.parametrize("structured", [False, True])
def test_sync_keyboard_interrupt_preserves_nested_settled_failure(structured, caplog):
    """A nested failure is reported without swallowing the user's interruption."""
    effects = []

    async def child():
        """Commit an effect and return a nested failed future after interruption."""
        try:

            def interrupt():
                """Interrupt the active synchronous runner."""
                raise KeyboardInterrupt

            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            effects.append("committed")
            future = asyncio.get_running_loop().create_future()
            if structured:
                future.set_exception(
                    ProblemException(
                        Problem(code="tool.nested", title="Nested", detail="Nested failure.")
                    )
                )
            else:
                future.set_exception(ValueError("nested failure"))
            return future

    async def calculate(number):
        """Treat the child as the registered tool's final outcome."""
        return await OwnedToolTask(child(), final_result=True)

    with asyncio.Runner() as runner, pytest.raises(KeyboardInterrupt) as caught:
        make_tool(calculate).execute({"number": 1}, async_runner=runner)
    payload = json.loads(caught.value.output)
    assert payload["problem"]["code"] == ("tool.nested" if structured else "tool.execution_failed")
    assert effects == ["committed"]
    if not structured:
        assert any(
            record.__dict__.get("error.type") == "tool.execution_failed"
            for record in caplog.records
        )


def test_owned_tool_task_retains_cancellation_history():
    """An owned task completes cleanup after clearing an earlier cancellation."""

    async def scenario():
        """Drive the owned task and its cancellation on one loop."""
        started, cleaning, release = (asyncio.Event() for _ in range(3))
        finished = []

        async def calculate():
            """Finish cleanup without accepting a repeated cancellation."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                asyncio.current_task().uncancel()
                cleaning.set()
                await release.wait()
                finished.append(True)
                return "done"

        work = OwnedToolTask(calculate())
        loop = asyncio.get_running_loop()
        task = work.start(loop)
        assert work.start(loop) is task
        await started.wait()
        task.cancel()
        await cleaning.wait()
        assert work.claim(loop) is task
        work.cancel()
        await deliver_callbacks()
        assert not task.done()
        release.set()
        assert await task == "done"
        assert finished == [True]
        work.cancel()
        with pytest.raises(RuntimeError, match="already claimed"):
            work.claim(loop)

    asyncio.run(scenario())


def test_owned_tool_task_awaits_once():
    """Awaiting a handle transfers ownership and returns its coroutine result."""

    async def scenario():
        """Await a handle and reject a second dispatcher."""

        async def calculate():
            """Return a known value."""
            return "done"

        work = OwnedToolTask(calculate())
        assert await work == "done"
        with pytest.raises(RuntimeError, match="already claimed"):
            await work

    asyncio.run(scenario())


def test_owned_tool_task_preserves_context_and_its_loop_factory():
    """Explicit owned work retains context without passing its task through an opaque factory."""
    correlation = ContextVar("owned_tool_correlation", default="missing")

    async def calculate():
        """Return the context present when the owned task starts."""
        return correlation.get()

    def factory(_loop, _coroutine, **_kwargs):
        """Reject use of the external factory for an explicitly owned inner task."""
        raise AssertionError("owned inner task reached the external factory")

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_task_factory(factory)
        token = correlation.set("owned-context")
        try:
            work = OwnedToolTask(calculate())
            task = work.start(loop)
            assert loop.get_task_factory() is factory
            loop.set_task_factory(None)
            assert runner.run(await_value_async(task)) == "owned-context"
        finally:
            correlation.reset(token)
            loop.set_task_factory(None)


def test_owned_tool_task_rejects_foreign_or_closed_loop(monkeypatch):
    """Owned work rejects the wrong loop and releases a coroutine on startup failure."""

    async def calculate():
        """Return a value if the owned task can start."""
        return "done"

    first = asyncio.new_event_loop()
    second = asyncio.new_event_loop()
    try:
        work = OwnedToolTask(calculate())
        task = work.start(first)
        with pytest.raises(ValueError, match="different event loop"):
            work.claim(second)
        assert work.claim(first) is task
        assert first.run_until_complete(task) == "done"
    finally:
        first.close()
        second.close()

    closed = asyncio.new_event_loop()
    closed.close()
    work = OwnedToolTask(calculate())
    with warnings.catch_warnings(record=True) as captured:
        with pytest.raises(RuntimeError, match="closed"):
            work.start(closed)
        gc.collect()
    assert captured == []

    other = asyncio.new_event_loop()
    work = OwnedToolTask(calculate())
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(other, "is_running", lambda: True)
            with pytest.raises(RuntimeError, match="event-loop thread"):
                work.start(other)
    finally:
        work.close()
        other.close()


def test_tool_returns_the_original_function_with_pure_defaults():
    """Bare declaration preserves callability and supplies pure registration metadata."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    declared = tool(calculate)
    registered = ToolRegistry([declared]).tools[0]

    assert declared is calculate
    assert calculate(3) == 3
    assert registered.name == "calculate"
    assert registered.actions == frozenset()


def test_plan_rejects_operations_outside_the_declared_action_bound():
    """A planner cannot silently expand the authority declared by its tool."""

    def planner(arguments):
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.SESSION_MUTATE,
                    target=SessionTarget(identifier="state"),
                ),
            ),
        )

    planned = Tool(
        function=Mock(),
        name="calculate",
        description="Calculate a value.",
        arguments_model=Arguments,
        operation_planner=planner,
    )

    with pytest.raises(ValueError, match="planned undeclared actions: session.mutate"):
        planned.plan({"number": 3})


def test_tool_options_preserve_an_explicitly_empty_capability_set():
    """Configured declarations retain names, descriptions, and empty capability collections."""

    @tool(name="selected", description="Selected tool.", actions=())
    def calculate(number: int) -> int:
        return number

    registered = ToolRegistry([calculate]).tools[0]

    assert registered.name == "selected"
    assert registered.description == "Selected tool."
    assert registered.actions == frozenset()


def test_tool_options_preserve_preflight_and_requiredness_metadata():
    """Declarations carry readiness behavior into each registered tool."""
    preflight = Mock(return_value=ToolPreflightResult(ToolStatus.READY))

    @tool(preflight=preflight, required=True)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registered = ToolRegistry([calculate]).tools[0]

    assert registered.preflight is preflight
    assert registered.required is True
    preflight.assert_called_once_with()


def test_tools_are_immutable():
    """Tool declarations cannot be modified after construction."""
    declaration = Tool(function=lambda: None)

    with pytest.raises(FrozenInstanceError):
        declaration.name = "changed"


def test_capability_bound_arguments_must_exist_in_the_registered_schema():
    """A misspelled conditional field cannot silently disappear from model definitions."""

    @tool(platform_argument_capabilities=(PlatformArgumentCapability("missing", "terminal"),))
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    with pytest.raises(ToolRegistrationError, match="unknown capability-bound arguments: missing"):
        ToolRegistry([calculate])


def test_platform_argument_capabilities_normalize_bare_pairs():
    """Registration converts valid positional pairs to named capability records."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registered = Tool(
        calculate, platform_argument_capabilities=(("number", "terminal"),)
    ).registered()

    assert registered.platform_argument_capabilities == (
        PlatformArgumentCapability("number", "terminal"),
    )
    assert isinstance(registered.platform_argument_capabilities[0], PlatformArgumentCapability)


@pytest.mark.parametrize(
    "item",
    [
        ("number",),
        ("number", 42),
        ["number", "terminal"],
        "no",
        ("", "terminal"),
        ("number", ""),
        ("number", "terminal", "Use this."),
    ],
)
def test_platform_argument_capabilities_reject_malformed_pairs(item):
    """Only malformed capability records fail registration."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    with pytest.raises(ToolRegistrationError, match="invalid platform argument capability"):
        Tool(calculate, platform_argument_capabilities=(item,)).registered()


def test_execution_selects_presentation_from_arguments_and_raw_result():
    """Dynamic presentation selectors receive canonical arguments and unserialized results."""
    selected = ToolResultPresentationSpec(kind=ToolResultPresentation.LIST)
    selector = Mock(return_value=selected)

    @tool(result_presentation=selector)
    def calculate(number: int) -> dict:
        """Calculate a structured value."""
        return {"values": [number]}

    registered = ToolRegistry([calculate]).tools[0]

    execution = registered.execute({"number": 3})

    assert json.loads(execution.output) == {"ok": True, "result": {"values": [3]}}
    assert execution.presentation is selected
    selector.assert_called_once_with({"number": 3}, {"values": [3]})


@pytest.mark.parametrize("selector", [Mock(side_effect=RuntimeError), Mock(return_value=None)])
def test_execution_falls_back_when_dynamic_presentation_selection_fails(selector):
    """Presentation metadata failures cannot turn successful tool calls into tool errors."""

    @tool(result_presentation=selector)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number * 2

    execution = ToolRegistry([calculate]).tools[0].execute({"number": 3})

    assert json.loads(execution.output) == {"ok": True, "result": 6}
    assert execution.presentation.kind is ToolResultPresentation.RAW


def test_passive_tools_require_registration_for_model_operations():
    """Passive definitions reject definition and argument-validation operations."""
    declaration = Tool(function=lambda: None)

    with pytest.raises(ValueError, match="must be registered"):
        declaration.definition()
    with pytest.raises(ValueError, match="must be registered"):
        declaration.validate_arguments("{}")


def test_tool_rejects_redeclaring_the_same_function():
    """A function has one unambiguous passive declaration."""

    @tool
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    with pytest.raises(ToolRegistrationError, match="already declared"):
        tool(name="other")(calculate)


@pytest.mark.parametrize("mode", ["action", "service", "sandbox"])
def test_fresh_host_requirement_rejects_incompatible_registration(mode):
    """Fresh host input cannot be attached to a tool without its separate process boundary."""

    @tool(
        actions=set() if mode == "action" else {Action.PROCESS_EXECUTE},
        operation_planner=lambda arguments: OperationPlan(arguments=arguments),
        requires_execution_service=mode != "service",
        execution_authorization=mode == "sandbox",
        requires_fresh_host_approval=True,
    )
    def command(context: ToolContext) -> None:
        """Request a command."""

    with pytest.raises(ToolRegistrationError, match="separately authorized host process plan"):
        ToolRegistry([command])


def test_service_dependency_requires_a_context_aware_tool():
    """A declared execution service cannot be injected into a context-free callable."""

    @tool(requires_execution_service=True)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    with pytest.raises(ToolRegistrationError, match="context-aware execution service"):
        ToolRegistry([calculate])


def test_definition_adapts_the_argument_model(monkeypatch):
    """Definition exposes neutral tool metadata and delegates schema adaptation."""
    adapt = Mock(return_value={"adapted": True})
    monkeypatch.setattr(tool_module, "get_tool_schema", adapt)

    definition = make_tool().definition()

    assert definition.name == "calculate"
    assert definition.description == "Calculate a value."
    assert definition.parameters == {"adapted": True}
    assert definition.strict is True
    adapt.assert_called_once_with(Arguments.model_json_schema())


def test_validate_arguments_prepares_call_arguments():
    """Validation converts serialized model input into typed call arguments."""
    arguments, error = make_tool().validate_arguments('{"number": 3}')

    assert arguments == {"number": 3}
    assert error is None


def test_call_invokes_and_serializes_validated_arguments(monkeypatch):
    """Synchronous dispatch invokes and serializes arguments prepared by validation."""
    function = Mock(return_value=4)
    serialize = Mock(return_value="serialized")
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr(tool_module, "serialize_tool_result", serialize)

    assert make_tool(function).call({"number": 3}) == "serialized"
    function.assert_called_once_with(number=3)
    serialize.assert_called_once_with(4)


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("async_function", [False, True])
def test_sync_execution_admits_at_callable_start(allowed, async_function):
    """Final admission controls both ordinary and coroutine tool bodies."""
    effects = []

    def sync_tool(number: int) -> int:
        """Record an admitted synchronous invocation."""
        effects.append(number)
        return number

    async def async_tool(number: int) -> int:
        """Record an admitted coroutine invocation."""
        effects.append(number)
        return number

    selected = make_tool(async_tool if async_function else sync_tool)
    with asyncio.Runner() as runner:
        if allowed:
            selected.execute({"number": 3}, async_runner=runner, execution_admission=lambda: True)
            assert effects == [3]
        else:
            with pytest.raises(ToolExecutionAdmissionRejected):
                selected.execute(
                    {"number": 3}, async_runner=runner, execution_admission=lambda: False
                )
            assert effects == []


@pytest.mark.parametrize(
    ("exception", "cancelled"),
    [
        (asyncio.CancelledError, True),
        (DerivedCancellation, True),
        (ValueError("synthetic"), False),
        (ValueError, False),
    ],
)
def test_tool_continuation_tracks_cancellation(exception, cancelled):
    """Coroutine delegation distinguishes cancellation from other exceptions."""

    async def calculate():
        """Suppress cancellation after one suspension."""
        try:
            await asyncio.sleep(0)
        except (asyncio.CancelledError, ValueError):
            return "settled"

    async def exercise():
        """Drive a continuation through its standard coroutine protocol."""
        continuation = tool_module.ToolContinuation(calculate())
        assert next(continuation) is None
        with pytest.raises(StopIteration) as stopped:
            continuation.throw(exception)
        assert stopped.value.value == "settled"
        assert continuation.cancellation_received is cancelled

    asyncio.run(exercise())


def test_closed_sync_runner_closes_unscheduled_coroutines(caplog):
    """A failed coroutine submission releases both unstarted awaitables without warnings."""
    effects = []

    async def calculate(number: int) -> int:
        """Record one admitted coroutine invocation."""
        effects.append(number)
        return number

    runner = asyncio.Runner()
    runner.get_loop()
    runner.close()
    with warnings.catch_warnings(record=True) as captured:
        result = make_tool(calculate).execute(
            {"number": 3}, async_runner=runner, execution_admission=lambda: True
        )
        gc.collect()
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
    assert effects == []
    assert captured == []
    assert len(caplog.records) == 1
    assert caplog.records[0].__dict__["error.type"] == "tool.execution_failed"
    assert caplog.records[0].__dict__["exception.type"] == "builtins.RuntimeError"


@pytest.mark.parametrize(
    "cleanup_outcome", ["success", "failure", "interrupt", "suppressed", "loop_stopped"]
)
def test_interrupted_sync_runner_preserves_async_cleanup(cleanup_outcome):
    """An interrupted started task settles its async cleanup on its owning loop."""
    cleanup = []

    def interrupt():
        """Interrupt the running loop while the tool is suspended."""
        raise KeyboardInterrupt

    async def calculate(number: int) -> int:
        """Suspend until interrupted, then finish asynchronous cleanup."""
        try:
            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if cleanup_outcome == "suppressed":
                return number
            raise
        finally:
            if cleanup_outcome == "interrupt":
                asyncio.get_running_loop().call_soon(interrupt)
            elif cleanup_outcome == "loop_stopped":
                loop = asyncio.get_running_loop()
                loop.call_soon(loop.stop)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            cleanup.append(number)
            if cleanup_outcome == "failure":
                raise ValueError("synthetic cleanup failure")
        return number

    with asyncio.Runner() as runner:
        with pytest.raises(KeyboardInterrupt) as interruption:
            make_tool(calculate).execute({"number": 3}, async_runner=runner)
        assert cleanup == [3]
        if cleanup_outcome == "failure":
            assert interruption.value.__notes__ == ["Tool cleanup also stopped with ValueError."]
        elif cleanup_outcome == "interrupt":
            assert interruption.value.__notes__ == [
                "Tool cleanup also stopped with KeyboardInterrupt."
            ]
        elif cleanup_outcome == "loop_stopped":
            assert interruption.value.__notes__ == ["Tool cleanup also stopped with RuntimeError."]
        assert runner.run(asyncio.sleep(0, result="ready")) == "ready"


def test_repeated_runner_interrupt_does_not_leak_cleanup_exception():
    """An abandoned cleanup shield cannot emit an unobserved task failure."""
    unhandled = []

    def interrupt():
        """Interrupt the owner loop at a controlled scheduling boundary."""
        raise KeyboardInterrupt

    async def calculate(number):
        """Fail cleanup after a second interruption."""
        try:
            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        finally:
            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            raise ValueError("cleanup failed")

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        with pytest.raises(KeyboardInterrupt):
            make_tool(calculate).execute({"number": 1}, async_runner=runner)
        runner.run(asyncio.sleep(0))
        gc.collect()
        assert unhandled == []


@pytest.mark.parametrize("admission", [False, True])
@pytest.mark.parametrize("clear_cancellation", [False, True])
@pytest.mark.parametrize("custom_task_factory", [False, True])
def test_interrupted_sync_runner_finishes_already_cancelling_cleanup(
    admission, clear_cancellation, custom_task_factory
):
    """A later interruption does not cancel cleanup already processing cancellation."""
    cleanup = []

    def interrupt():
        """Interrupt the loop while cancellation cleanup is suspended."""
        raise KeyboardInterrupt

    async def calculate(number: int) -> int:
        """Finish asynchronous cleanup after the first cancellation request."""
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        try:
            loop.call_soon(task.cancel)
            await asyncio.Event().wait()
        finally:
            cleanup.append("started")
            if clear_cancellation:
                task.uncancel()
            loop.call_soon(interrupt)
            await asyncio.sleep(0)
            cleanup.append(number)
        return number

    def task_factory(loop, coroutine, **kwargs):
        """Wrap a coroutine while preserving the caller's task factory."""
        assert inspect.iscoroutine(coroutine)

        async def invoke():
            """Await the coroutine through an additional task-factory layer."""
            return await coroutine

        return asyncio.Task(invoke(), loop=loop, **kwargs)

    with asyncio.Runner() as runner:
        if custom_task_factory:
            runner.get_loop().set_task_factory(task_factory)
        with pytest.raises(KeyboardInterrupt):
            make_tool(calculate).execute(
                {"number": 3},
                async_runner=runner,
                execution_admission=(lambda: True) if admission else None,
            )
        assert cleanup == ["started", 3]
        assert runner.run(asyncio.sleep(0, result="ready")) == "ready"


def test_interrupted_sync_runner_settles_nested_task_factory():
    """An interrupted runner settles its supervising task as well as the tool task."""
    events = []

    def interrupt():
        """Interrupt the loop while the tool is suspended."""
        raise KeyboardInterrupt

    async def calculate(number: int) -> int:
        """Finish tool cleanup before the runner releases control."""
        try:
            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        finally:
            events.append("tool cleanup")
        return number

    def task_factory(loop, coroutine, **kwargs):
        """Supervise a separate task with asynchronous cleanup."""

        async def supervise():
            """Retain factory cleanup until its child settles."""
            child = asyncio.Task(coroutine, loop=loop)
            try:
                return await child
            finally:
                events.append("factory cleanup started")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                events.append("factory cleanup finished")

        return asyncio.Task(supervise(), loop=loop, **kwargs)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_task_factory(task_factory)
        try:
            with pytest.raises(KeyboardInterrupt):
                make_tool(calculate).execute({"number": 3}, async_runner=runner)
            assert events == ["tool cleanup", "factory cleanup started", "factory cleanup finished"]
            assert not asyncio.all_tasks(loop)
        finally:
            loop.set_task_factory(None)


@pytest.mark.parametrize("factory_mode", ["task", "future", "opaque"])
def test_interrupted_sync_runner_does_not_start_factory_shielded_tool(factory_mode):
    """A pre-start interruption cannot admit work detached by a task factory."""
    effects = []

    def interrupt():
        """Interrupt before the tool coroutine gets a turn."""
        raise KeyboardInterrupt

    async def calculate(number: int) -> int:
        """Record a tool body that must not be entered."""
        effects.append(number)
        return number

    def task_factory(loop, coroutine, **kwargs):
        """Shield the factory-created child, with a task or future as its owner."""
        if factory_mode == "opaque":

            async def invoke():
                """Await the runner coroutine through an additional child."""
                return await coroutine

            child = asyncio.Task(invoke(), loop=loop, **kwargs)
        else:
            child = asyncio.Task(coroutine, loop=loop, **kwargs)
        if factory_mode == "future":
            return asyncio.shield(child)

        async def supervise():
            """Retain a separate task while shielding its child."""
            return await asyncio.shield(child)

        return asyncio.Task(supervise(), loop=loop, **kwargs)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_task_factory(task_factory)
        try:
            loop.call_soon(interrupt)
            with pytest.raises(KeyboardInterrupt):
                make_tool(calculate).execute({"number": 3}, async_runner=runner)
            loop.set_task_factory(None)
            runner.run(asyncio.sleep(0))
            assert effects == []
            assert not asyncio.all_tasks(loop)
        finally:
            loop.set_task_factory(None)


def test_interrupted_sync_runner_settles_returned_task_before_it_starts():
    """An interrupted submitted call cannot leave a returned task to start later."""
    effects = []

    def interrupt():
        """Interrupt before the returned task starts."""
        raise KeyboardInterrupt

    async def effect(number: int) -> int:
        """Record any execution of the returned task."""
        effects.append(number)
        return number

    with asyncio.Runner() as runner:
        loop = runner.get_loop()

        def calculate(number: int):
            """Return work owned by this call."""
            return OwnedToolTask(effect(number))

        loop.call_soon(interrupt)
        with pytest.raises(KeyboardInterrupt):
            make_tool(calculate).execute({"number": 3}, async_runner=runner)
        runner.run(asyncio.sleep(0))
        assert effects == []
        assert not asyncio.all_tasks(loop)


@pytest.mark.parametrize("clear_cancellation", [False, True])
def test_interrupted_sync_runner_preserves_returned_task_cleanup(clear_cancellation):
    """A returned task already cancelling finishes cleanup before interruption returns."""
    cleanup = []

    def interrupt():
        """Interrupt while the returned task is cleaning up."""
        raise KeyboardInterrupt

    async def effect(number: int) -> int:
        """Handle cancellation and finish asynchronous cleanup."""
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        try:
            loop.call_soon(task.cancel)
            await asyncio.Event().wait()
        finally:
            cleanup.append("started")
            if clear_cancellation:
                task.uncancel()
            loop.call_soon(interrupt)
            await asyncio.sleep(0)
            cleanup.append(number)
        return number

    with asyncio.Runner() as runner:
        loop = runner.get_loop()

        def calculate(number: int):
            """Return work owned by this call."""
            work = OwnedToolTask(effect(number))
            work.start(loop)
            return work

        with pytest.raises(KeyboardInterrupt):
            make_tool(calculate).execute({"number": 3}, async_runner=runner)
        assert cleanup == ["started", 3]
        assert not asyncio.all_tasks(loop)


def test_failed_submission_closes_unstarted_owned_work(caplog):
    """A failed submission cannot leave explicitly owned work to start later."""
    effects = []

    async def effect(number: int) -> int:
        """Record an effect that must not start."""
        effects.append(number)
        return number

    with asyncio.Runner() as runner:
        loop = runner.get_loop()

        def bad_factory(_loop, _coroutine, **_kwargs):
            """Reject the runner's task before it is submitted."""
            raise OSError("synthetic task factory failure")

        def calculate(number: int):
            """Declare ownership of work without starting it."""
            return OwnedToolTask(effect(number))

        loop.set_task_factory(bad_factory)
        try:
            output = json.loads(
                make_tool(calculate).execute({"number": 3}, async_runner=runner).output
            )
            assert output["problem"]["code"] == "tool.execution_failed"
            assert effects == []
        finally:
            loop.set_task_factory(None)
        runner.run(asyncio.sleep(0))
        assert effects == []
        assert not asyncio.all_tasks(loop)
    assert len(caplog.records) == 1


def test_failed_submission_settles_borrowed_task_before_reporting_result():
    """A borrowed task cannot start later after submission failure is reported."""
    effects = []

    async def effect(number: int) -> int:
        """Produce a known result from borrowed work."""
        effects.append(number)
        return number

    with asyncio.Runner() as runner:
        loop = runner.get_loop()

        def bad_factory(_loop, _coroutine, **_kwargs):
            """Reject the runner's task before it is submitted."""
            raise OSError("synthetic task factory failure")

        def calculate(number: int):
            """Return a borrowed task without passing it through the failing factory."""
            return asyncio.Task(effect(number), loop=loop)

        loop.set_task_factory(bad_factory)
        try:
            output = json.loads(
                make_tool(calculate).execute({"number": 3}, async_runner=runner).output
            )
            assert output == {"ok": True, "result": 3}
            assert effects == [3]
        finally:
            loop.set_task_factory(None)
        runner.run(asyncio.sleep(0))
        assert effects == [3]
        assert not asyncio.all_tasks(loop)


def test_failed_submission_settles_started_owned_task_before_reporting_result():
    """A started owned task produces its actual result if wrapper submission fails."""
    effects = []

    async def effect(number: int) -> int:
        """Produce one result from owned work."""
        effects.append(number)
        return number

    with asyncio.Runner() as runner:
        loop = runner.get_loop()

        def bad_factory(_loop, _coroutine, **_kwargs):
            """Reject the runner's task before it is submitted."""
            raise OSError("synthetic task factory failure")

        def calculate(number: int):
            """Start explicitly owned work on the same loop."""
            work = OwnedToolTask(effect(number))
            work.start(loop)
            return work

        loop.set_task_factory(bad_factory)
        try:
            output = json.loads(
                make_tool(calculate).execute({"number": 3}, async_runner=runner).output
            )
            assert output == {"ok": True, "result": 3}
            assert effects == [3]
        finally:
            loop.set_task_factory(None)
        assert not asyncio.all_tasks(loop)


@pytest.mark.parametrize("custom_factory", [False, True])
def test_sync_runner_preserves_task_factory_changes(custom_factory):
    """The configured factory is visible to the tool and its later changes are retained."""
    observed = []

    def factory(loop, coroutine, **kwargs):
        """Create a task with the supplied loop and context."""
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    def replacement(loop, coroutine, **kwargs):
        """Create tasks after the tool changes its factory."""
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    async def calculate(number: int) -> int:
        """Observe and change the runner's factory."""
        loop = asyncio.get_running_loop()
        observed.append(loop.get_task_factory())
        loop.set_task_factory(replacement)
        return number

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        original = factory if custom_factory else None
        loop.set_task_factory(original)
        try:
            result = make_tool(calculate).execute({"number": 3}, async_runner=runner)
            assert json.loads(result.output) == {"ok": True, "result": 3}
            assert observed == [original]
            assert loop.get_task_factory() is replacement
        finally:
            loop.set_task_factory(None)


def test_sync_runner_preserves_unrelated_factory_task():
    """Interrupting a tool does not cancel unrelated work created by its task factory."""
    events = []
    background = None
    creating = False
    release = asyncio.Event()

    def interrupt():
        """Interrupt the tool after the background task starts."""
        raise KeyboardInterrupt

    async def background_work():
        """Wait independently of the tool invocation."""
        events.append("background started")
        await release.wait()
        events.append("background finished")

    async def calculate(number: int) -> int:
        """Suspend until the tool is interrupted."""
        asyncio.get_running_loop().call_soon(interrupt)
        await asyncio.Event().wait()
        return number

    def factory(loop, coroutine, **kwargs):
        """Create one independent task in addition to the runner's task."""
        nonlocal background, creating
        if not creating:
            creating = True
            background = loop.create_task(background_work())
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.set_task_factory(factory)
        try:
            with pytest.raises(KeyboardInterrupt):
                make_tool(calculate).execute({"number": 3}, async_runner=runner)
            assert events == ["background started"]
            assert not background.done()
            release.set()
            runner.run(asyncio.sleep(0))
            assert events == ["background started", "background finished"]
        finally:
            release.set()
            loop.set_task_factory(None)


@pytest.mark.parametrize("fail_closure", [False, True])
def test_interrupted_sync_runner_closes_when_cleanup_loop_fails(monkeypatch, fail_closure):
    """A runner that cannot settle a tool cannot be reused for later dispatch."""
    runner = asyncio.Runner()
    loop = runner.get_loop()
    run_until_complete = loop.run_until_complete
    close = runner.close
    calls = 0

    def drive(future):
        """Fail the cleanup drive while leaving runner closure available."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic loop failure")
        return run_until_complete(future)

    def interrupt():
        """Interrupt the event loop during the tool."""
        raise KeyboardInterrupt

    async def calculate(number: int) -> int:
        """Wait until the loop is interrupted."""
        asyncio.get_running_loop().call_soon(interrupt)
        await asyncio.Event().wait()
        return number

    monkeypatch.setattr(loop, "run_until_complete", drive)
    if fail_closure:
        monkeypatch.setattr(runner, "close", Mock(side_effect=OSError("synthetic closure failure")))
    try:
        if fail_closure:
            with pytest.raises(SystemExit, match="could not be safely closed") as failure:
                make_tool(calculate).execute({"number": 3}, async_runner=runner)
            assert isinstance(failure.value.__cause__, KeyboardInterrupt)
        else:
            with pytest.raises(KeyboardInterrupt) as interruption:
                make_tool(calculate).execute({"number": 3}, async_runner=runner)
            assert interruption.value.__notes__ == ["Tool cleanup also stopped with OSError."]
            with pytest.raises(RuntimeError, match="Runner is closed"):
                runner.get_loop()
    finally:
        close()


def test_failed_submission_preserves_a_returned_future_owner(caplog):
    """A failed submission does not cancel a future owned by the caller's event loop."""
    with asyncio.Runner() as owner:
        future = owner.get_loop().create_future()
        runner = Mock(spec=asyncio.Runner)
        runner.get_loop.return_value = owner.get_loop()
        runner.run.side_effect = RuntimeError("synthetic submission failure")
        result = make_tool(lambda number: future).execute({"number": 3}, async_runner=runner)
        assert not future.done()
        future.cancel()
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
    assert len(caplog.records) == 1
    assert caplog.records[0].__dict__["error.type"] == "tool.execution_failed"
    assert caplog.records[0].__dict__["exception.type"] == "builtins.RuntimeError"


@pytest.mark.parametrize("owned", [False, True])
def test_direct_async_dispatch_settles_returned_cleanup_after_uncancel(monkeypatch, owned):
    """Borrowed and explicitly owned tasks retain cleanup after clearing cancellation."""

    async def scenario():
        """Interrupt a direct call while returned work is already cleaning up."""
        started, cleaning, attached, release = (asyncio.Event() for _ in range(4))
        completed = []
        shield = asyncio.shield

        async def effect():
            """Finish cleanup after clearing the first cancellation."""
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                asyncio.current_task().uncancel()
                cleaning.set()
                await release.wait()
                completed.append(True)

        handle = OwnedToolTask(effect()) if owned else None
        task = handle.start(asyncio.get_running_loop()) if owned else asyncio.create_task(effect())

        def observe_shield(value):
            """Expose when direct dispatch begins waiting for returned work."""
            if value is task:
                attached.set()
            return shield(value)

        monkeypatch.setattr(asyncio, "shield", observe_shield)

        def calculate(number: int):
            """Return either explicitly owned or borrowed work."""
            return handle if owned else task

        await started.wait()
        task.cancel()
        await cleaning.wait()
        dispatch = asyncio.create_task(make_tool(calculate).execute_async({"number": 3}))
        try:
            await attached.wait()
            dispatch.cancel()
            await deliver_callbacks()
            assert not task.done()
            assert not dispatch.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await dispatch
            assert completed == [True]
        finally:
            release.set()
            await asyncio.gather(dispatch, task, return_exceptions=True)

    asyncio.run(scenario())


def test_direct_async_worker_cancellation_closes_unstarted_owned_work():
    """Cancellation after a worker starts does not leak its returned owned coroutine."""
    effects = []
    entered = asyncio.Event()
    release = Event()
    owner_loop = None

    async def effect(number: int) -> int:
        """Record work that must remain unstarted."""
        effects.append(number)
        return number

    def calculate(number: int):
        """Return owned work only after the worker is released."""
        owner_loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=5)
        return OwnedToolTask(effect(number))

    async def scenario():
        """Cancel the caller while the worker is still running."""
        nonlocal owner_loop
        owner_loop = asyncio.get_running_loop()
        dispatch = asyncio.create_task(make_tool(calculate).execute_async({"number": 3}))
        try:
            await entered.wait()
            dispatch.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await dispatch
            assert effects == []
        finally:
            release.set()
            await asyncio.gather(dispatch, return_exceptions=True)

    with warnings.catch_warnings(record=True) as captured:
        asyncio.run(scenario())
        gc.collect()
    assert captured == []


@pytest.mark.parametrize("async_function", [False, True])
@pytest.mark.parametrize("allowed", [False, True])
def test_async_execution_admits_at_callable_start(allowed, async_function):
    """Async dispatch propagates admission refusal without presenting a tool failure."""
    effects = []

    def sync_tool(number: int) -> int:
        """Record an admitted synchronous invocation."""
        effects.append(number)
        return number

    async def async_tool(number: int) -> int:
        """Record an admitted coroutine invocation."""
        effects.append(number)
        return number

    selected = make_tool(async_tool if async_function else sync_tool)

    async def dispatch(invoke):
        """Run the synchronous invocation through a controlled dispatcher."""
        return invoke()

    async def exercise():
        """Check the observable result and side effect for this admission."""
        if allowed:
            await selected.execute_async(
                {"number": 3}, sync_dispatch=dispatch, execution_admission=lambda: True
            )
            assert effects == [3]
        else:
            with pytest.raises(ToolExecutionAdmissionRejected):
                await selected.execute_async(
                    {"number": 3}, sync_dispatch=dispatch, execution_admission=lambda: False
                )
            assert effects == []

    asyncio.run(exercise())


def test_async_worker_rechecks_admission_after_queuing():
    """A queued worker is not admitted using an earlier policy decision."""
    effects = []
    allowed = [True]

    def calculate(number: int) -> int:
        """Record a synthetic effect."""
        effects.append(number)
        return number

    async def exercise():
        """Change admission while dispatch is waiting to run the callable."""
        queued = asyncio.Event()
        release = asyncio.Event()

        async def dispatch(invoke):
            """Hold the callable until the owner changes its decision."""
            queued.set()
            await release.wait()
            return invoke()

        task = asyncio.create_task(
            make_tool(calculate).execute_async(
                {"number": 3}, sync_dispatch=dispatch, execution_admission=lambda: allowed[0]
            )
        )
        await queued.wait()
        allowed[0] = False
        release.set()
        with pytest.raises(ToolExecutionAdmissionRejected):
            await task
        assert effects == []

    asyncio.run(exercise())


@pytest.mark.parametrize("async_mode", [False, True])
def test_admission_check_failure_is_not_reported_as_tool_failure(async_mode, caplog):
    """An admission failure remains separate from the unstarted tool's result."""
    function = Mock(return_value=3)
    selected = make_tool(function)

    def fail():
        """Expose a failed policy authority check."""
        raise OSError("policy unavailable")

    with pytest.raises(ToolExecutionAdmissionRejected) as captured:
        if async_mode:
            asyncio.run(selected.execute_async({"number": 3}, execution_admission=fail))
        else:
            selected.execute({"number": 3}, execution_admission=fail)
    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "policy unavailable"
    function.assert_not_called()
    assert not caplog.records


def test_call_injects_required_context(monkeypatch):
    """Inject context during dispatch and report missing context as an execution error."""
    function = Mock(return_value="done")
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=True))
    monkeypatch.setattr(tool_module, "serialize_tool_result", lambda result: result)
    context = Mock(spec=ToolContext)
    tool = make_tool(function)

    assert tool.call({"number": 3}, context) == "done"
    function.assert_called_once_with(context, number=3)
    failure = json.loads(tool.call({"number": 3}))
    assert failure["ok"] is False
    assert failure["problem"]["code"] == "tool.execution_failed"
    assert failure["problem"]["detail"] == "Tool 'calculate' failed."


def test_validate_arguments_returns_errors_without_invoking(monkeypatch):
    """Invalid model arguments return structured details without reaching application code."""
    function = Mock()
    arguments, error = make_tool(function).validate_arguments("not json")

    assert arguments is None
    failure = json.loads(error)
    assert failure["problem"]["code"] == "tool.invalid_arguments"
    assert failure["problem"]["metadata"]["fields"]
    function.assert_not_called()


def test_call_rejects_coroutine_functions_in_sync_dispatch(monkeypatch):
    """Synchronous dispatch directs coroutine functions to the asynchronous API."""

    async def calculate(number: int) -> int:
        return number * 2

    failure = json.loads(make_tool(calculate).call({"number": 3}))
    assert failure["problem"]["code"] == "tool.async_required"


def test_execute_rejects_awaitables_without_a_close_method_inside_a_loop():
    """Synchronous execution safely rejects non-closeable awaitables in an active loop."""

    runner = asyncio.Runner()

    async def exercise():
        future = asyncio.get_running_loop().create_future()
        direct = make_tool(lambda number: future).execute({"number": 3})

        async def calculate(number: int) -> int:
            return number

        async_tool = make_tool(calculate).execute({"number": 3}, async_runner=runner)
        future.cancel()
        return json.loads(direct.output), json.loads(async_tool.output)

    try:
        direct, async_tool = asyncio.run(exercise())
    finally:
        runner.close()

    assert direct["problem"]["code"] == "tool.async_required"
    assert async_tool["problem"]["code"] == "tool.async_required"


def test_call_serializes_execution_failures(monkeypatch):
    """Synchronous application failures become model-readable execution errors."""
    function = Mock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))

    failure = json.loads(make_tool(function).call({"number": 3}))
    assert failure["problem"]["code"] == "tool.execution_failed"
    assert "boom" not in failure["problem"]["detail"]


def test_call_async_supports_sync_and_awaitable_results(monkeypatch):
    """Asynchronous dispatch serializes both immediate and awaitable application results."""

    async def calculate(number: int) -> int:
        return number * 2

    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr(tool_module, "serialize_tool_result", lambda result: str(result))

    assert asyncio.run(make_tool(Mock(return_value=4)).call_async({"number": 3})) == "4"
    assert asyncio.run(make_tool(calculate).call_async({"number": 3})) == "6"


def test_call_async_keeps_the_owner_loop_responsive_for_context_free_sync_tools():
    """Direct async dispatch runs context-free synchronous callbacks in a worker."""
    started = Event()
    release = Event()
    worker_threads = []

    def calculate(number: int) -> int:
        """Wait for the test owner before returning a value."""
        worker_threads.append(get_ident())
        started.set()
        release.wait(timeout=1)
        return number

    async def invoke():
        owner_thread = get_ident()
        pending = asyncio.create_task(make_tool(calculate).call_async({"number": 7}))
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 1), timeout=2)
        await deliver_callbacks()
        assert not pending.done()
        release.set()
        assert json.loads(await pending)["result"] == 7
        assert worker_threads != [owner_thread]

    asyncio.run(invoke())


def test_direct_async_tool_rejects_a_live_context_without_a_registry_relay():
    """A context-aware sync tool cannot receive live owner services through a direct call."""

    def inspect_context(context: ToolContext, number: int) -> int:
        """Return a value if the context was incorrectly exposed."""
        del context
        return number

    result = asyncio.run(
        make_tool(inspect_context).execute_async(
            {"number": 1}, ToolContext(interaction=None, tool_name="calculate")
        )
    )

    assert json.loads(result.output)["problem"]["code"] == "tool.context_relay_unavailable"


def test_call_async_handles_execution_failures(monkeypatch):
    """Asynchronous dispatch returns structured errors for application failures."""
    function = Mock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))
    tool = make_tool(function)

    failure = json.loads(asyncio.run(tool.call_async({"number": 3})))
    assert failure["problem"]["code"] == "tool.execution_failed"


def test_call_async_preserves_structured_tool_problems(monkeypatch):
    """Asynchronous dispatch preserves deliberate structured tool failures."""
    problem = Problem(code="tool.unavailable", title="Unavailable", detail="Try later.")
    function = Mock(side_effect=ProblemException(problem))
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))

    failure = json.loads(asyncio.run(make_tool(function).call_async({"number": 3})))

    assert failure["problem"]["code"] == "tool.unavailable"
    assert failure["problem"]["detail"] == "Try later."


def test_failed_runner_factory_closes_unstarted_owned_coroutine():
    """Runner startup failure closes work that has not begun."""

    async def effect():
        """Provide an unstarted owned operation."""
        return "unexpected"

    coroutine = effect()
    work = OwnedToolTask(coroutine)

    def calculate(number):
        """Return work before runner creation."""
        return work

    def failed_factory():
        """Represent a failed host runner."""
        raise RuntimeError("runner unavailable")

    result = make_tool(calculate).execute({"number": 1}, async_runner_factory=failed_factory)
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


def test_direct_async_cancellation_carries_settled_receipt():
    """Direct async dispatch propagates cancellation with the completed result."""

    async def scenario():
        """Interrupt an operation that commits during cancellation."""
        started = asyncio.Event()

        async def calculate(number):
            """Finish after cancellation arrives."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return number

        task = asyncio.create_task(make_tool(calculate).execute_async({"number": 7}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert json.loads(caught.value.output)["result"] == 7

    asyncio.run(scenario())


def test_direct_async_final_owned_receipt_survives_interruption():
    """Direct async dispatch carries the declared final result through cancellation."""

    async def scenario():
        """Interrupt a tool while its declared final owned operation settles."""
        started = asyncio.Event()

        async def effect():
            """Return the final result after interruption."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "confirmed"

        async def calculate(number):
            """Declare the awaited operation as the complete tool result."""
            return await OwnedToolTask(effect(), final_result=True)

        task = asyncio.create_task(make_tool(calculate).execute_async({"number": 1}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert json.loads(caught.value.output)["result"] == "confirmed"

    asyncio.run(scenario())


def test_failed_custom_awaitable_start_closes_only_the_wrapper(monkeypatch, caplog):
    """Failed wrapper creation leaves the borrowed custom awaitable untouched."""

    async def scenario():
        """Fail task creation at the native event-loop boundary."""
        events = []

        class BorrowedResult:
            """Record any entry or ownership-taking close."""

            def __await__(self):
                async def resolve():
                    """Record an incorrectly entered awaitable."""
                    events.append("entered")
                    return "unexpected"

                return resolve().__await__()

            def close(self):
                events.append("closed")

        async def dispatch(invoke):
            """Run the synchronous test tool on the current owner."""
            return invoke()

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "Task", Mock(side_effect=RuntimeError("task creation failed")))
            result = await make_tool(lambda number: BorrowedResult()).execute_async(
                {"number": 1}, sync_dispatch=dispatch
            )
        assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
        assert any(
            record.__dict__.get("error.type") == "tool.execution_failed"
            for record in caplog.records
        )
        assert events == []

    asyncio.run(scenario())


def test_async_tool_awaits_nested_returned_task():
    """A nested borrowed task settles before dispatch returns its receipt."""

    async def scenario():
        """Return a child task from a native async tool."""
        release = asyncio.Event()

        async def child():
            """Wait until the caller permits completion."""
            await release.wait()
            return "settled"

        async def calculate(number):
            """Return a newly scheduled task."""
            return asyncio.create_task(child())

        task = asyncio.create_task(make_tool(calculate).execute_async({"number": 1}))
        await deliver_callbacks()
        assert not task.done()
        release.set()
        assert json.loads((await task).output)["result"] == "settled"

    asyncio.run(scenario())


def test_sync_tool_awaits_nested_returned_task():
    """Synchronous runner dispatch waits for nested returned tasks."""
    effects = []

    async def child():
        """Complete after a scheduling boundary."""
        await asyncio.sleep(0)
        effects.append("done")
        return "confirmed"

    async def calculate(number):
        """Return a task from the async tool."""
        return asyncio.create_task(child())

    with asyncio.Runner() as runner:
        result = make_tool(calculate).execute({"number": 1}, async_runner=runner)
    assert json.loads(result.output)["result"] == "confirmed"
    assert effects == ["done"]


def test_direct_async_worker_cancellation_carries_settled_receipt():
    """Cancellation retains the completed direct worker's result."""
    started = Event()
    release = Event()

    def calculate(number):
        """Finish a blocking effect after the owner is interrupted."""
        started.set()
        release.wait(timeout=5)
        return number

    async def scenario():
        """Cancel and release the worker after it starts."""
        task = asyncio.create_task(make_tool(calculate).execute_async({"number": 7}))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert json.loads(caught.value.output)["result"] == 7

    asyncio.run(scenario())


def test_repeated_cancellation_preserves_nested_borrowed_receipt():
    """Repeated interruption retains a borrowed child's eventual result."""

    async def scenario():
        """Interrupt the owned operation and then the borrowed child wait."""
        started = asyncio.Event()
        release = asyncio.Event()
        child_started = asyncio.Event()

        async def child():
            """Return a result after release."""
            child_started.set()
            await release.wait()
            return "confirmed"

        async def calculate(number):
            """Return borrowed work while handling interruption."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return asyncio.create_task(child())

        task = asyncio.create_task(make_tool(calculate).execute_async({"number": 1}))
        await started.wait()
        task.cancel()
        await child_started.wait()
        task.cancel()
        await deliver_callbacks()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert json.loads(caught.value.output)["result"] == "confirmed"

    asyncio.run(scenario())


def test_owned_task_rejects_foreign_thread_without_consuming_claim():
    """A foreign-thread claim or cancellation leaves ownership and task state intact."""

    async def scenario():
        """Exercise operations from a worker and then from the owner loop."""

        async def effect():
            """Wait for the owner to cancel the task."""
            await asyncio.Event().wait()

        loop = asyncio.get_running_loop()
        work = OwnedToolTask(effect())
        with pytest.raises(RuntimeError, match="event-loop thread"):
            await asyncio.to_thread(work.claim, loop)
        task = work.claim(loop)
        await deliver_callbacks()
        with pytest.raises(RuntimeError, match="event-loop thread"):
            await asyncio.to_thread(work.start, loop)
        with pytest.raises(RuntimeError, match="event-loop thread"):
            await asyncio.to_thread(work.cancel)
        assert not task.cancelling()
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario(), debug=True)


@pytest.mark.parametrize("outcome", ["cancelled", "success", "failure"])
def test_await_owned_handle_settles_after_cancellation(outcome):
    """Awaiting an owned handle settles cancellation and records cleanup failures."""

    async def scenario():
        """Interrupt the handle while the owned coroutine is running."""
        started = asyncio.Event()

        async def effect():
            """Finish, fail, or propagate cancellation during cleanup."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if outcome == "failure":
                    raise ValueError("cleanup failed")
                if outcome == "success":
                    return "done"
                raise

        work = OwnedToolTask(effect())
        task = asyncio.ensure_future(work)
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        if outcome == "failure":
            assert any("ValueError" in note for note in caught.value.__notes__)
        assert work.task.done()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_final_owned_handle_carries_outcome_while_cancelled(failure):
    """A declared final handle propagates cancellation with its settled outcome."""

    async def scenario():
        """Interrupt owned work and inspect the explicitly declared outcome."""
        started = asyncio.Event()

        async def effect():
            """Settle after cancellation with a result or a failure."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if failure:
                    raise ValueError("settled failure")
                return "confirmed"

        task = asyncio.ensure_future(OwnedToolTask(effect(), final_result=True))
        await started.wait()
        task.cancel()
        with pytest.raises(OwnedToolTaskCancelled) as caught:
            await task
        assert caught.value.result == (None if failure else "confirmed")
        if failure:
            assert isinstance(caught.value.error, ValueError)
            assert str(caught.value.error) == "settled failure"
        else:
            assert caught.value.error is None

    asyncio.run(scenario())


def test_owned_start_rejects_different_loop():
    """An already-started handle cannot be adopted by a different event loop."""

    async def effect():
        """Return when the owner loop runs."""
        return "done"

    with asyncio.Runner() as first, asyncio.Runner() as second:
        work = OwnedToolTask(effect())
        task = work.start(first.get_loop())
        with pytest.raises(ValueError, match="different event loop"):
            work.start(second.get_loop())
        assert first.run(await_value_async(task)) == "done"


def test_sync_nested_self_referential_future_is_rejected():
    """A future returning itself is rejected instead of spinning forever."""
    with asyncio.Runner() as runner:
        future = runner.get_loop().create_future()
        future.set_result(future)
        result = make_tool(lambda number: future).execute({"number": 1}, async_runner=runner)
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"


def test_async_nested_self_referential_future_is_rejected():
    """Async dispatch rejects a completed future whose result is itself."""

    async def scenario():
        """Return a cyclic borrowed future from the tool."""
        future = asyncio.get_running_loop().create_future()
        future.set_result(future)
        result = await make_tool(lambda number: future).execute_async({"number": 1})
        assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["cycle", "started_coroutine"])
def test_interrupted_worker_returning_borrowed_work_has_no_completed_receipt(kind):
    """An unsettled borrowed result after interruption leaves the outcome unknown."""
    entered = asyncio.Event()
    release = Event()

    async def scenario():
        """Interrupt after the worker starts and let it return a cyclic future."""
        loop = asyncio.get_running_loop()
        if kind == "cycle":
            result = loop.create_future()
            result.set_result(result)
        else:

            async def other():
                """Suspend work owned outside the tool dispatcher."""
                await asyncio.sleep(0)

            result = other()
            result.send(None)

        def calculate(number: int):
            """Return borrowed work after cancellation."""
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5)
            return result

        dispatch = asyncio.create_task(make_tool(calculate).execute_async({"number": 1}))
        try:
            await entered.wait()
            dispatch.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await dispatch
            assert not hasattr(caught.value, "output")
        finally:
            release.set()
            await asyncio.gather(dispatch, return_exceptions=True)
            if kind == "started_coroutine":
                result.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["owned", "task"])
def test_runner_factory_failure_with_started_work_stops_dispatch(kind):
    """A failed runner factory cannot return control with started tool work."""

    async def effect():
        """Wait for cleanup after the failed factory."""
        await asyncio.Event().wait()

    with asyncio.Runner() as runner:
        if kind == "owned":
            result = OwnedToolTask(effect())
            task = result.start(runner.get_loop())
        else:
            task = runner.get_loop().create_task(effect())
            result = task

        def fail():
            """Fail before a runner is available for settlement."""
            raise RuntimeError("runner unavailable")

        try:
            with pytest.raises(SystemExit, match="cannot be safely settled"):
                make_tool(lambda number: result).execute({"number": 1}, async_runner_factory=fail)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                runner.run(await_value_async(task))


@pytest.mark.parametrize("owned", [False, True])
def test_failed_runner_get_loop_releases_unstarted_work(owned):
    """A failing runner cannot leak an unstarted coroutine."""

    async def effect():
        """Provide an unstarted coroutine."""
        return "done"

    coroutine = effect()
    value = OwnedToolTask(coroutine) if owned else coroutine
    runner = Mock()
    runner.get_loop.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        make_tool().resolve_awaitable(value, runner)
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.parametrize("owned", [False, True])
def test_failed_runner_get_loop_rejects_started_work(owned):
    """A failing runner cannot report a result while a started task remains."""

    async def effect():
        """Wait for test cleanup."""
        await asyncio.Event().wait()

    with asyncio.Runner() as real_runner:
        loop = real_runner.get_loop()
        work = OwnedToolTask(effect()) if owned else None
        task = work.start(loop) if owned else loop.create_task(effect())
        runner = Mock()
        runner.get_loop.side_effect = RuntimeError("unavailable")
        try:
            with pytest.raises(SystemExit, match="cannot be safely settled"):
                make_tool().resolve_awaitable(work if owned else task, runner)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                real_runner.run(await_value_async(task))


def test_sync_owned_work_without_runner_is_closed():
    """A direct sync invocation without a runner closes unstarted owned work."""

    async def effect():
        """Represent work that cannot be admitted."""
        return "unexpected"

    coroutine = effect()
    result = make_tool(lambda number: OwnedToolTask(coroutine)).execute({"number": 1})
    assert json.loads(result.output)["problem"]["code"] == "tool.async_required"
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


def test_sync_custom_awaitable_without_runner_remains_borrowed():
    """An unhosted custom awaitable stays unknown and retains its caller-owned future."""
    with asyncio.Runner() as owner:
        future = owner.get_loop().create_future()

        class BorrowedResult:
            """Make an incorrect close observable on the owner's future."""

            def __await__(self):
                return future.__await__()

            def close(self):
                future.cancel()

        with pytest.raises(ToolExecutionOutcomeUnknown):
            make_tool(lambda number: BorrowedResult()).execute({"number": 1})
        assert not future.cancelled()
        future.cancel()


@pytest.mark.parametrize("host", ["absent", "factory_failure", "runner_failure"])
def test_unhosted_borrowed_future_keeps_the_tool_outcome_unknown(host):
    """A runner failure cannot turn pending borrowed work into a completed tool result."""
    effects = []
    with asyncio.Runner() as owner:
        future = owner.get_loop().create_future()

        def effect(number):
            """Perform an effect and return work retained by another owner."""
            effects.append("entered")
            return future

        def fail():
            """Represent a runner factory that cannot provide a host."""
            raise RuntimeError("runner unavailable")

        runner = Mock() if host == "runner_failure" else None
        if runner is not None:
            runner.get_loop.side_effect = RuntimeError("runner unavailable")
        try:
            with pytest.raises(ToolExecutionOutcomeUnknown):
                make_tool(effect).execute(
                    {"number": 1},
                    async_runner=runner,
                    async_runner_factory=fail if host == "factory_failure" else None,
                )
            assert effects == ["entered"]
            assert not future.done()
        finally:
            future.cancel()


def test_unhosted_started_coroutine_remains_with_its_original_owner():
    """A running coroutine cannot become a completed result when its host is unavailable."""
    effects = []
    with asyncio.Runner() as owner:
        started = asyncio.Event()
        release = asyncio.Event()

        async def foreign():
            """Remain active on the original loop until that owner releases it."""
            started.set()
            await release.wait()
            effects.append("foreign")

        coroutine = foreign()
        task = owner.get_loop().create_task(coroutine)
        owner.run(started.wait())

        def effect(number):
            """Perform an effect and return the running foreign coroutine."""
            effects.append("tool")
            return coroutine

        def fail():
            """Represent an unavailable runner factory."""
            raise RuntimeError("runner unavailable")

        try:
            with pytest.raises(ToolExecutionOutcomeUnknown):
                make_tool(effect).execute({"number": 1}, async_runner_factory=fail)
            assert effects == ["tool"]
            assert not task.done()
            release.set()
            owner.run(await_value_async(task))
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    owner.run(await_value_async(task))
    assert effects == ["tool", "foreign"]


def test_unhosted_completed_owned_handle_cannot_certify_nested_work():
    """A completed handle can contain pending work owned by another task."""
    effects = []
    with asyncio.Runner() as owner:
        release = asyncio.Event()

        async def child():
            """Finish only when the original owner permits it."""
            await release.wait()
            effects.append("foreign")

        async def outer():
            """Return an independently owned pending child."""
            return owner.get_loop().create_task(child())

        handle = OwnedToolTask(outer())
        outer_task = handle.start(owner.get_loop())
        child_task = owner.run(await_value_async(outer_task))

        def fail():
            """Represent an unavailable runner factory."""
            raise RuntimeError("runner unavailable")

        try:
            with pytest.raises(ToolExecutionOutcomeUnknown):
                make_tool(lambda number: handle).execute({"number": 1}, async_runner_factory=fail)
            assert not child_task.done()
            release.set()
            owner.run(await_value_async(child_task))
        finally:
            if not child_task.done():
                child_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    owner.run(await_value_async(child_task))
    assert effects == ["foreign"]


@pytest.mark.parametrize("host", ["factory", "runner"])
def test_unhosted_preclaimed_work_preserves_control_interruption(host):
    """An interrupted host setup cannot replace control flow or close another owner's work."""

    async def foreign():
        """Return only when its original owner starts the work."""
        return "foreign"

    handle = OwnedToolTask(foreign())
    handle.reserve()

    def interrupt():
        """Interrupt before a runner can host the preclaimed work."""
        raise KeyboardInterrupt

    runner = Mock() if host == "runner" else None
    if runner is not None:
        runner.get_loop.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        make_tool(lambda number: handle).execute(
            {"number": 1},
            async_runner=runner,
            async_runner_factory=interrupt if host == "factory" else None,
        )
    with asyncio.Runner() as owner:
        assert owner.run(await_value_async(handle.start(owner.get_loop()))) == "foreign"


@pytest.mark.parametrize("host", ["factory", "runner"])
def test_unhosted_borrowed_future_preserves_control_interruption(host):
    """User interruption is not replaced when a borrowed future cannot be hosted."""
    with asyncio.Runner() as owner:
        future = owner.get_loop().create_future()

        def interrupt():
            """Interrupt before a runner can accept the borrowed future."""
            raise KeyboardInterrupt

        runner = Mock() if host == "runner" else None
        if runner is not None:
            runner.get_loop.side_effect = KeyboardInterrupt
        try:
            with pytest.raises(KeyboardInterrupt):
                make_tool(lambda number: future).execute(
                    {"number": 1},
                    async_runner=runner,
                    async_runner_factory=interrupt if host == "factory" else None,
                )
            assert not future.done()
        finally:
            future.cancel()


def test_direct_async_foreign_owned_task_is_rejected_without_cancellation():
    """Foreign work remains with its owner without establishing a tool outcome."""

    async def effect():
        """Wait until released by the original loop."""
        await asyncio.sleep(0)
        return "done"

    with asyncio.Runner() as foreign:
        work = OwnedToolTask(effect())
        task = work.start(foreign.get_loop())

        async def pending():
            """Leave pending foreign work with its original owner."""
            with pytest.raises(ToolExecutionOutcomeUnknown):
                await make_tool(lambda number: work).execute_async({"number": 1})
            assert not task.cancelling()

        asyncio.run(pending())
        assert foreign.run(await_value_async(task)) == "done"

        async def completed():
            """Do not infer a final result from an opaque foreign-loop task."""
            with pytest.raises(ToolExecutionOutcomeUnknown):
                await make_tool(lambda number: work).execute_async({"number": 1})

        asyncio.run(completed())


@pytest.mark.parametrize("owned", [False, True])
def test_direct_async_interruption_preserves_foreign_work_without_receipt(owned):
    """A foreign result after interruption stays unknown and remains with its owner."""
    started = Event()
    release = Event()
    effects = []

    async def foreign_effect():
        """Perform a later effect only when the original owner runs its loop."""
        effects.append("foreign")
        return "done"

    with asyncio.Runner() as foreign:
        handle = OwnedToolTask(foreign_effect())
        foreign_task = handle.start(foreign.get_loop())

        def calculate(number):
            """Commit an effect and return a result owned by the other loop."""
            started.set()
            assert release.wait(timeout=5)
            effects.append("tool")
            return handle if owned else foreign_task

        async def scenario():
            """Interrupt an entered worker before its foreign result is returned."""
            task = asyncio.create_task(make_tool(calculate).execute_async({"number": 1}))
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as captured:
                await task
            assert not hasattr(captured.value, "output")
            assert effects == ["tool"]
            assert not foreign_task.cancelled()

        try:
            asyncio.run(scenario())
            assert foreign.run(await_value_async(foreign_task)) == "done"
        finally:
            release.set()
        assert effects == ["tool", "foreign"]


def test_direct_async_worker_interruption_without_result_propagates():
    """An interrupted worker failure does not invent a completed receipt."""
    started = Event()
    release = Event()

    def calculate(number):
        """Fail after interruption without returning a value."""
        started.set()
        release.wait(timeout=5)
        raise ValueError("worker failed")

    async def scenario():
        """Interrupt the worker and wait for its failure."""
        task = asyncio.create_task(make_tool(calculate).execute_async({"number": 1}))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert any("ValueError" in note for note in caught.value.__notes__)
        assert not hasattr(caught.value, "output")

    asyncio.run(scenario())


def test_direct_async_tool_accepts_custom_awaitable():
    """Async dispatch resolves an ordinary custom awaitable."""

    class Result:
        """Represent a value with the native await protocol."""

        def __await__(self):
            async def value():
                """Produce the final application result."""
                return "confirmed"

            return value().__await__()

    async def scenario():
        """Resolve the custom awaitable in direct async dispatch."""
        result = await make_tool(lambda number: Result()).execute_async({"number": 1})
        assert json.loads(result.output)["result"] == "confirmed"

    asyncio.run(scenario())


def test_runner_factory_inspection_failure_closes_coroutine(monkeypatch):
    """Runner setup failure before owner creation releases the tool coroutine."""

    async def effect():
        """Represent work that must never begin."""
        return "unexpected"

    coroutine = effect()
    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        with monkeypatch.context() as patch:
            patch.setattr(loop, "get_task_factory", Mock(side_effect=RuntimeError("unavailable")))
            with pytest.raises(RuntimeError, match="unavailable"):
                make_tool().resolve_awaitable(coroutine, runner)
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


def test_interrupted_async_tool_cancels_started_nested_owned_work():
    """An interruption also cancels newly returned started owned work."""

    async def scenario():
        """Return an owned child while settling the parent's cancellation."""
        started = asyncio.Event()
        child_tasks = []

        async def child():
            """Wait for cancellation before doing any work."""
            await asyncio.Event().wait()

        async def calculate(number):
            """Start a child during cancellation cleanup."""
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                work = OwnedToolTask(child())
                child_tasks.append(work.start(asyncio.get_running_loop()))
                return work

        task = asyncio.create_task(make_tool(calculate).execute_async({"number": 1}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert child_tasks and child_tasks[0].done()
        assert child_tasks[0].cancelled()

    asyncio.run(scenario())


def test_failed_runner_loop_lookup_preserves_finished_task():
    """A finished task requires no emergency shutdown if runner lookup fails."""

    async def effect():
        """Return an already settled value."""
        return "done"

    with asyncio.Runner() as real_runner:
        task = real_runner.get_loop().create_task(effect())
        assert real_runner.run(await_value_async(task)) == "done"
        runner = Mock()
        runner.get_loop.side_effect = RuntimeError("unavailable")
        with pytest.raises(RuntimeError, match="unavailable"):
            make_tool().resolve_awaitable(task, runner)
    assert task.result() == "done"


def test_interrupted_runner_setup_cancels_prestarted_owned_task(monkeypatch):
    """Interruption before runner submission settles explicitly owned work."""

    async def effect():
        """Remain pending until canceled."""
        await asyncio.Event().wait()

    with asyncio.Runner() as runner:
        work = OwnedToolTask(effect())
        task = work.start(runner.get_loop())
        with monkeypatch.context() as patch:
            patch.setattr(runner, "run", Mock(side_effect=KeyboardInterrupt))
            with pytest.raises(KeyboardInterrupt):
                make_tool().resolve_awaitable(work, runner)
        assert task.cancelled()


def test_runner_failure_with_cancelled_tool_has_no_fabricated_receipt():
    """A failed runner cannot invent a result when its owned operation is cancelled."""

    async def effect(number):
        """Stop the loop and allow cancellation to propagate."""
        asyncio.get_running_loop().stop()
        await asyncio.sleep(0)
        return number

    with asyncio.Runner() as runner:
        result = make_tool(effect).execute({"number": 1}, async_runner=runner)
    assert json.loads(result.output)["problem"]["code"] == "tool.execution_failed"
