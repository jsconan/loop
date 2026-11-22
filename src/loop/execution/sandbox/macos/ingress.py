"""Build a separately native-confined connector to one foreground Unix service."""

import json
from pathlib import Path

INGRESS_SOCKET_ENVIRONMENT_VARIABLE = "LOOP_INGRESS_SOCKET"
INGRESS_ENDPOINT_NOT_READY = 77
INGRESS_PEER_MISMATCH = 78
_CONNECTOR_FAILED = 79

_CONNECTOR = rf"""
use Socket;
use IO::Select;
use Errno qw(ENOENT ECONNREFUSED);
my ($path, $pid) = @ARGV;
socket(my $target, PF_UNIX, SOCK_STREAM, 0) or die $!;
connect($target, pack_sockaddr_un($path)) or
    exit(($! == ENOENT || $! == ECONNREFUSED) ? {INGRESS_ENDPOINT_NOT_READY} : {_CONNECTOR_FAILED});
my $peer = getsockopt($target, 0, 2);
(defined($peer) && unpack("i", $peer) == $pid) or exit {INGRESS_PEER_MISMATCH};
my $readers = IO::Select->new(\*STDIN, $target);
while (my @ready = $readers->can_read) {{
    for my $source (@ready) {{
        my $count = sysread($source, my $data, 65536);
        defined($count) or exit {_CONNECTOR_FAILED};
        if (!$count) {{
            if (fileno($source) == fileno(STDIN)) {{
                $readers->remove(\*STDIN); shutdown($target, 1); next;
            }}
            exit 0;
        }}
        my $output = fileno($source) == fileno(STDIN) ? $target : \*STDOUT;
        my $offset = 0;
        while ($offset < $count) {{
            my $written = syswrite($output, $data, $count - $offset, $offset);
            (defined($written) && $written > 0) or exit {_CONNECTOR_FAILED};
            $offset += $written;
        }}
    }}
}}
"""


def connector_arguments(endpoint: Path, process_id: int) -> list[str]:
    """Build an exact Unix-only connector that checks peer identity before forwarding bytes.

    Args:
        endpoint (Path): Fresh attempt socket on the checked local-IPC volume.
        process_id (int): Live, unreaped foreground service identity retained by the supervisor.

    Returns:
        list[str]: Trusted system launcher and connector arguments, with no IP grant.
    """
    profile = "\n".join(
        (
            "(version 1)",
            "(deny default)",
            "(allow process-exec sysctl-read)",
            '(allow file-read-data (literal "/"))',
            "(allow file-read-metadata)",
            (
                '(allow file-read* (subpath "/usr/bin") (subpath "/usr/lib") '
                '(subpath "/System/Library") (subpath "/dev"))'
            ),
            '(allow file-write-data (literal "/dev/null"))',
            "(allow system-socket (socket-domain AF_UNIX))",
            f"(allow network-outbound (remote unix-socket (literal {json.dumps(str(endpoint))})))",
        )
    )
    return [
        "/usr/bin/sandbox-exec",
        "-p",
        profile,
        "/usr/bin/perl",
        "-e",
        _CONNECTOR,
        str(endpoint),
        str(process_id),
    ]
