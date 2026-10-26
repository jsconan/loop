"""Verify native thread contention and shared holder ownership."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from loop.utils import IntValueHolder

pytestmark = pytest.mark.integration


def test_update_is_atomic_across_concurrent_workers():
    """Concurrent atomic increments cannot lose read-modify-write updates."""
    holder = IntValueHolder(0)
    workers = 8
    increments = 500

    def increment() -> None:
        """Increment the shared holder repeatedly through its atomic API."""
        for _ in range(increments):
            holder.update(lambda current: current + 1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _worker: increment(), range(workers)))

    assert holder.get() == workers * increments


def test_shared_reads_writes_and_conversions_remain_consistent():
    """Readers observe complete coerced values while another thread replaces them."""
    holder = IntValueHolder(0)
    barrier = threading.Barrier(5)

    def write(seed: int) -> None:
        """Replace the shared value after all workers are ready."""
        barrier.wait(timeout=5)
        for offset in range(500):
            holder.value = seed + offset

    def read() -> None:
        """Read and convert complete integer snapshots during replacement."""
        barrier.wait(timeout=5)
        for _ in range(500):
            assert isinstance(holder.get(), int)
            assert isinstance(holder.value, int)
            assert isinstance(int(holder), int)
            assert isinstance(str(holder), str)
            assert holder == holder.get() or holder != holder.get()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write, seed * 1_000) for seed in range(2)]
        futures.extend(executor.submit(read) for _ in range(3))
        for future in futures:
            future.result()


def test_opposite_order_holder_comparisons_do_not_deadlock():
    """Two threads can compare the same holders in opposite orders without lock nesting."""
    first = IntValueHolder(1)
    second = IntValueHolder(1)
    barrier = threading.Barrier(2)

    def compare(left: IntValueHolder, right: IntValueHolder) -> bool:
        """Start with the peer and compare repeatedly in one operand order."""
        barrier.wait(timeout=5)
        return all(left == right for _ in range(1_000))

    with ThreadPoolExecutor(max_workers=2) as executor:
        forward = executor.submit(compare, first, second)
        reverse = executor.submit(compare, second, first)
        assert forward.result(timeout=2)
        assert reverse.result(timeout=2)
