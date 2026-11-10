"""Provide event-loop and process synchronization without elapsed-time delays."""

import asyncio
import os


def await_condition(condition):
    """Wait for a scheduling event with a bounded number of non-timed yields."""
    for _ in range(1_000_000):
        if condition():
            return
        os.sched_yield()
    raise AssertionError("The expected scheduling event was not observed.")


def receive_fifo_byte(path):
    """Read one FIFO byte without waiting for elapsed time or an absent writer."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    received = []

    def available():
        """Capture data if a writer has reached the FIFO."""
        try:
            value = os.read(descriptor, 1)
        except BlockingIOError:
            return False
        if value:
            received.append(value)
            return True
        return False

    try:
        await_condition(available)
        return received[0]
    finally:
        os.close(descriptor)


async def deliver_callbacks():
    """Deliver callbacks already queued on the active loop without scheduling a timer."""
    loop = asyncio.get_running_loop()
    delivered = loop.create_future()
    loop.call_soon(delivered.set_result, None)
    await delivered
