# Value holders

`ValueHolder` and its scalar and path subclasses are safe to share between threads. Individual reads and replacements through `.value`, `get()`, and `set()` are synchronized. Conversions, comparisons, and `PathHolder` operations use a snapshot of the value observed when the operation begins. Each holder in a comparison is snapshotted independently.

Use `update()` for an atomic read-modify-write operation:

```python
counter = IntValueHolder(0)
counter.update(lambda value: value + 1)
```

The callback executes while the holder lock is held, so it should be short and avoid slow I/O or unnecessary blocking. Separate reads and writes are not atomic; neither `counter.value += 1` nor `counter.set(counter.get() + 1)` is an atomic increment. `compare_and_set(expected, value)` offers an atomic conditional replacement without exposing the holder's lock.

`PathHolder` releases its lock after obtaining a path snapshot and before filesystem I/O or lazy iteration. For example, if one thread starts `logs.read_text()` while another retargets `logs` with `logs.set("/new/logs")`, the read continues against its original path snapshot and future operations use the new path.
