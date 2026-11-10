"""Serve owned synthetic DNS answers for the macOS direct-resolver control."""

import argparse
import socket
import socketserver
import struct
from pathlib import Path

_DOMAIN = "loop-egress-owned.test"


class FixtureDnsHandler(socketserver.BaseRequestHandler):
    """Record and answer only synthetic names on the disposable VM."""

    def handle(self) -> None:
        """Reply to one bounded UDP question and retain its queried name."""
        packet, channel = self.request
        if len(packet) < 17 or len(packet) > 512:
            return
        offset = 12
        labels = []
        while offset < len(packet):
            size = packet[offset]
            offset += 1
            if size == 0:
                break
            if size > 63 or offset + size > len(packet):
                return
            labels.append(packet[offset : offset + size].decode("ascii", errors="ignore"))
            offset += size
        else:
            return
        if offset + 4 > len(packet):
            return
        name = ".".join(labels).lower()
        qtype, qclass = struct.unpack_from("!HH", packet, offset)
        if qclass != 1 or not (name == _DOMAIN or name.endswith("." + _DOMAIN)):
            return
        with (self.server.directory / "dns-queries.log").open("a", encoding="ascii") as output:
            output.write(f"{name} {qtype}\n")
        question = packet[12 : offset + 4]
        answer = (
            b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 0, 4) + socket.inet_aton("11.0.0.2")
            if qtype == 1
            else b""
        )
        response = (
            struct.pack(
                "!HHHHHH", struct.unpack_from("!H", packet)[0], 0x8180, 1, int(bool(answer)), 0, 0
            )
            + question
            + answer
        )
        channel.sendto(response, self.client_address)


def main() -> None:
    """Bind the private VM address and serve the owned DNS test suffix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    options = parser.parse_args()
    with socketserver.UDPServer((options.address, 53), FixtureDnsHandler) as server:
        server.directory = options.directory
        server.serve_forever()


if __name__ == "__main__":
    main()
