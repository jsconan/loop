"""Provide explicit event-loop delivery without timers or elapsed-time delays."""

import asyncio


async def deliver_callbacks():
    """Deliver callbacks already queued on the active loop without scheduling a timer."""
    loop = asyncio.get_running_loop()
    delivered = loop.create_future()
    loop.call_soon(delivered.set_result, None)
    await delivered
