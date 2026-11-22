"""Relay accepted host clients only to one private namespace endpoint."""

from __future__ import annotations

import array
import ctypes
import os
import select
import socket
import threading

_MAX_CLIENTS = 64
_POLL_SECONDS = 0.1
_TIMEOUT_SECONDS = 0.5
_LOCALHOST = "127.0.0.1"


def start_namespace_ingress(descriptor: int, port: int) -> None:
    """Fork a sealed namespace worker and remove its control descriptor from the payload.

    Args:
        descriptor (int): Inherited authenticated sequenced-packet control socket.
        port (int): Approved private-loopback TCP port.

    Raises:
        OSError: Authority activation, nondumpability or worker creation fails.
    """
    with socket.socket(fileno=descriptor) as channel:
        channel.settimeout(2.0)
        if channel.recv(1) != b"G":
            raise OSError("Ingress authority did not activate.")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
            raise OSError("Ingress worker identity could not be sealed.")
        if os.fork() == 0:
            try:
                with open(os.devnull, "wb", buffering=0) as sink:
                    os.dup2(sink.fileno(), 1)
                    os.dup2(sink.fileno(), 2)
                _serve(channel, port)
            finally:
                os._exit(0)


def _serve(channel: socket.socket, port: int) -> None:
    """Accept only one connected descriptor and opaque correlation ID per handoff."""
    clients: set[socket.socket] = set()
    lock = threading.Lock()
    stopped = threading.Event()

    def relay(client: socket.socket, identifier: bytes) -> None:
        """Forward within the private namespace until control or either peer closes."""
        try:
            client.settimeout(_TIMEOUT_SECONDS)
            with socket.create_connection((_LOCALHOST, port), timeout=_TIMEOUT_SECONDS) as upstream:
                while not stopped.is_set():
                    ready, _, _ = select.select([client, upstream], [], [], _POLL_SECONDS)
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (upstream if source is client else client).sendall(data)
        except OSError:
            pass
        finally:
            client.close()
            with lock:
                clients.discard(client)
            try:
                channel.send(b"D" + identifier)
            except OSError:
                pass

    try:
        while True:
            if not select.select([channel], [], [], _POLL_SECONDS)[0]:
                continue
            data, controls, flags, _ = channel.recvmsg(
                5, socket.CMSG_SPACE(array.array("i").itemsize), socket.MSG_CMSG_CLOEXEC
            )
            handles = array.array("i")
            for level, kind, payload in controls:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    handles.frombytes(payload[: len(payload) - len(payload) % handles.itemsize])
            if (
                len(data) != 5
                or data[:1] != b"C"
                or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
                or len(handles) != 1
            ):
                for handle in handles:
                    os.close(handle)
                return
            client = socket.socket(fileno=handles[0])
            with lock:
                if (
                    len(clients) >= _MAX_CLIENTS
                    or client.family != socket.AF_INET
                    or client.type != socket.SOCK_STREAM
                    or client.getsockname()[0] != _LOCALHOST
                    or client.getpeername()[0] != _LOCALHOST
                ):
                    client.close()
                    return
                clients.add(client)
            threading.Thread(target=relay, args=(client, data[1:]), daemon=True).start()
    except (OSError, ValueError, RuntimeError):
        return
    finally:
        stopped.set()
        with lock:
            for client in clients:
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                client.close()
