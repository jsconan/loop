"""Serve synthetic HTTP and HTTPS artifacts in an externally routed test namespace.

Run only inside an owned disposable network namespace with a non-loopback address. The
test controller creates the route, certificate, and fixture directory separately.
"""

import argparse
import base64
import hashlib
import http.server
import ssl
import subprocess
import time
import zipfile
from functools import partial
from pathlib import Path


def prepare(directory: Path) -> None:
    """Create only synthetic package-index and Git fixtures in the owned directory."""
    directory.mkdir(parents=True, exist_ok=True)
    wheel_name = "loop_egress_fixture-1.0.0-py3-none-any.whl"
    wheel = directory / "wheels" / wheel_name
    wheel.parent.mkdir(exist_ok=True)
    files = {
        "loop_egress_fixture/__init__.py": b'VALUE = "ok"\n',
        "loop_egress_fixture-1.0.0.dist-info/METADATA": (
            b"Metadata-Version: 2.3\nName: loop-egress-fixture\nVersion: 1.0.0\n"
        ),
        "loop_egress_fixture-1.0.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: loop-fixture\nRoot-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
    }
    records = []
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
            digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
            records.append(f"{name},sha256={digest.decode('ascii')},{len(payload)}")
        record_name = "loop_egress_fixture-1.0.0.dist-info/RECORD"
        archive.writestr(record_name, "\n".join((*records, f"{record_name},,")) + "\n")
    simple = directory / "simple" / "loop-egress-fixture"
    simple.mkdir(parents=True, exist_ok=True)
    (simple / "index.html").write_text(
        f'<a href="/wheels/{wheel_name}">{wheel_name}</a>\n', encoding="utf-8"
    )
    redirect = directory / "redirect"
    redirect.mkdir(exist_ok=True)
    (redirect / "index.html").write_text("redirect-ok\n", encoding="utf-8")
    source = directory / "git-source"
    source.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "README.md").write_text("Synthetic mediated egress fixture.\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    repository = directory / "repo.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(source), str(repository)], check=True)
    subprocess.run(["git", "--git-dir", str(repository), "update-server-info"], check=True)


class FixtureHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static package and Git data plus controlled redirect and stream responses."""

    def do_GET(self):
        """Return a deterministic resource or an owned static fixture file."""
        if self.path in {"/", "/redirect", "/redirect-forbidden", "/slow"}:
            if self.path.startswith("/redirect"):
                self.send_response(302)
                destination = (
                    "https://11.0.0.3:18443/"
                    if self.path == "/redirect-forbidden"
                    else "/redirect/"
                )
                self.send_header("Location", destination)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            first = (
                b"started\n"
                if self.path == "/slow"
                else b"egress-https-ok\n"
                if self.server.is_https
                else b"egress-fixture-ok\n"
            )
            last = b"finished\n" if self.path == "/slow" else b""
            self.send_response(200)
            self.send_header("Content-Length", str(len(first) + len(last)))
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            if last:
                time.sleep(8)
                self.wfile.write(last)
            return
        super().do_GET()

    def log_message(self, format, *args):
        """Keep the disposable fixture quiet without logging request paths."""
        del format, args


def main():
    """Bind the supplied isolated address and optionally wrap it with test TLS."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--address")
    parser.add_argument("--port", type=int)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--prepare", action="store_true")
    options = parser.parse_args()
    if options.prepare:
        prepare(options.directory)
        return
    if options.address is None or options.port is None:
        parser.error("--address and --port are required when serving")
    server = http.server.ThreadingHTTPServer(
        (options.address, options.port),
        partial(FixtureHandler, directory=str(options.directory)),
    )
    server.is_https = options.certificate is not None
    if options.certificate is not None and options.key is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(options.certificate, options.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
