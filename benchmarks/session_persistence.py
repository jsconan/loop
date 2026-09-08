"""Measure SQLite snapshot persistence as session history grows."""

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from loop import Message, Session, SQLiteSessionStore


def measure_snapshot_saves(message_counts: tuple[int, ...]) -> list[tuple[int, int, float]]:
    """Measure one full-snapshot save at each requested history size.

    Args:
        message_counts (tuple[int, ...]): Increasing positive history sizes to measure.

    Returns:
        list[tuple[int, int, float]]: Message count, serialized bytes, and save seconds per sample.

    Raises:
        ValueError: If counts are not positive and strictly increasing.
    """
    if any(count <= 0 for count in message_counts) or tuple(sorted(set(message_counts))) != (
        message_counts
    ):
        raise ValueError("Message counts must be positive and strictly increasing.")

    with TemporaryDirectory() as directory:
        store = SQLiteSessionStore(Path(directory) / "sessions.db", workspace_id="benchmark")
        session = Session()
        results = []
        previous = 0
        for count in message_counts:
            session.messages.extend(
                Message(role="user", content=f"message {index}") for index in range(previous, count)
            )
            started = perf_counter()
            store.save(session)
            elapsed = perf_counter() - started
            results.append((count, len(session.serialize().encode()), elapsed))
            previous = count
        return results


def main() -> None:
    """Print default snapshot persistence measurements."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("counts", nargs="*", type=int, default=[10, 100, 1000, 5000])
    counts = tuple(parser.parse_args().counts)
    print("messages\tbytes\tseconds")
    for count, size, elapsed in measure_snapshot_saves(counts):
        print(f"{count}\t{size}\t{elapsed:.6f}")


if __name__ == "__main__":
    main()
