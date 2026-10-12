"""Check exported native BPF semantics and irreversible kernel enforcement in owned children."""

import json
import platform
import struct
import subprocess
import sys

import pytest

from loop.execution.sandbox.linux.seccomp import descriptor_filter

pytestmark = [pytest.mark.integration, pytest.mark.linux]

# Independent test expectations, never used to construct the production filter.
_ABIS = {
    "aarch64": (0xC00000B7, (212, 243, 425, 426, 427), 63),
    "arm64": (0xC00000B7, (212, 243, 425, 426, 427), 63),
    "x86_64": (0xC000003E, (47, 299, 425, 426, 427), 0),
    "amd64": (0xC000003E, (47, 299, 425, 426, 427), 0),
}


def evaluate(program, architecture, syscall):
    """Interpret exported classic BPF against a synthetic seccomp record without execution."""
    instructions = tuple(struct.iter_unpack("=HBBI", program))
    index = 0
    accumulator = 0
    for _ in range(len(instructions)):
        opcode, yes, no, value = instructions[index]
        if opcode == 0x20:
            assert value in (0, 4)
            accumulator = architecture if value == 4 else syscall
        elif opcode == 0x15:
            index += yes if accumulator == value else no
        elif opcode == 0x35:
            index += yes if accumulator >= value else no
        elif opcode == 0x25:
            index += yes if accumulator > value else no
        elif opcode == 0x45:
            index += yes if accumulator & value else no
        elif opcode == 0x05:
            index += value
        elif opcode == 0x06:
            return value
        else:
            pytest.fail(f"Unexpected exported BPF opcode: {opcode}")
        index += 1
    pytest.fail("Exported filter has no terminal verdict")


@pytest.fixture
def native_filter():
    """Require real protected libseccomp export on a supported native Linux ABI."""
    name = platform.machine().lower()
    if name not in _ABIS:
        pytest.skip("requires a native 64-bit ARM or x86 ABI")
    return descriptor_filter(), _ABIS[name]


def test_export_rejects_foreign_and_compatibility_abis(native_filter):
    """Actual exported BPF kills foreign audit architectures and x32 syscall aliases."""
    program, (architecture, numbers, ordinary) = native_filter
    for foreign in (0, 0x40000003, 0x40000028, 0xC000003E, 0xC00000B7):
        if foreign != architecture:
            for number in (*numbers, ordinary):
                assert evaluate(program, foreign, number) == 0x80000000
    if architecture == 0xC000003E:
        for number in (*numbers, ordinary):
            assert evaluate(program, architecture, 0x40000000 | number) == 0x80000000


def test_export_preserves_required_denials_and_other_calls(native_filter):
    """Real native export denies receivers and io_uring while retaining ordinary calls."""
    program, (architecture, numbers, ordinary) = native_filter
    for number, error in zip(numbers, (1, 1, 38, 38, 38), strict=True):
        assert evaluate(program, architecture, number) == 0x50000 | error
    assert evaluate(program, architecture, ordinary) == 0x7FFF0000
    assert evaluate(program, architecture, 424) == 0x7FFF0000
    assert evaluate(program, architecture, 428) == 0x7FFF0000


def test_kernel_enforces_filter_and_child_inheritance(native_filter, tmp_path):
    """A disposable ordinary child and its descendant receive exact kernel denial errors."""
    program, (_, numbers, _) = native_filter
    code = r"""
import ctypes, json, os, sys
class Instruction(ctypes.Structure):
    _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint32)]
class Program(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ushort), ('filter', ctypes.POINTER(Instruction))]
data = bytes.fromhex(sys.argv[1])
instructions = (Instruction * (len(data) // 8)).from_buffer_copy(data)
program = Program(len(instructions), instructions)
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.restype = ctypes.c_int
libc.syscall.restype = ctypes.c_long
if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'Filter installation failed')
numbers = json.loads(sys.argv[2])
def check():
    results = []
    for number in numbers:
        ctypes.set_errno(0)
        result = libc.syscall(number, -1, 0, 0, 0, 0, 0)
        results.append([result, ctypes.get_errno()])
    return results
print(json.dumps(check()), flush=True)
child = os.fork()
if child == 0:
    print(json.dumps(check()), flush=True)
    os._exit(0)
if os.waitpid(child, 0)[1] != 0:
    raise RuntimeError('Child failed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, program.hex(), json.dumps(numbers)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        timeout=5,
        check=False,
    )
    assert result.returncode == 0 and not result.stderr
    expected = [[-1, error] for error in (1, 1, 38, 38, 38)]
    assert [json.loads(line) for line in result.stdout.splitlines()] == [expected, expected]
