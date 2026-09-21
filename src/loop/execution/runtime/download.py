"""Acquire manifest-pinned file artifacts without ambient network authority."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import httpx

from ... import constants
from .models import Artifact


class AcquisitionError(RuntimeError):
    """Report a failed or unsafe artifact acquisition."""


class AcquisitionCancelled(AcquisitionError):
    """Report cancellation before an artifact became installed content."""


class ArtifactUnavailable(AcquisitionError):
    """Report an unavailable artifact, including an offline transport."""


class IntegrityFailure(AcquisitionError):
    """Report downloaded bytes that do not match trusted manifest identity."""


class QuotaExceeded(AcquisitionError):
    """Report a configured runtime storage quota exhaustion."""


class ArtifactTransport(Protocol):
    """Stream an HTTPS artifact using a reviewed transport implementation."""

    def stream(self, url: str) -> Iterator[tuple[str, Iterator[bytes]]]:
        """Yield final URL and response body chunks for one request.

        Args:
            url (str): HTTPS URL to request.

        Yields:
            tuple[str, Iterator[bytes]]: Final URL and response body chunks.

        Raises:
            AcquisitionError: If the request or redirect policy fails.
        """


class HttpxArtifactTransport:
    """Stream HTTPS downloads while preserving same-origin redirect authority."""

    def stream(self, url: str) -> Iterator[tuple[str, Iterator[bytes]]]:
        """Yield a bounded HTTPX response stream.

        Args:
            url (str): Manifest-pinned HTTPS source URL.

        Yields:
            tuple[str, Iterator[bytes]]: Final URL and binary chunks.

        Raises:
            AcquisitionError: If the request or redirect policy fails.
        """
        try:
            with (
                httpx.Client(follow_redirects=True, max_redirects=3, trust_env=False) as client,
                client.stream("GET", url) as response,
            ):
                response.raise_for_status()
                if any(
                    urlparse(str(item.url)).scheme != "https"
                    for item in (*response.history, response)
                ):
                    raise AcquisitionError("Artifact redirect used a non-HTTPS origin.")
                yield str(response.url), response.iter_bytes()
        except httpx.HTTPError as error:
            raise ArtifactUnavailable("Artifact is unavailable from its pinned source.") from error


def download_artifact(
    artifact: Artifact,
    directory: Path,
    transport: ArtifactTransport,
    configured_limit: int,
    cancelled: Callable[[], bool] = lambda: False,
) -> Path:
    """Download and verify one file artifact into a private temporary file.

    Args:
        artifact (Artifact): Manifest-selected file or archive artifact.
        directory (Path): Loop-private temporary directory.
        transport (ArtifactTransport): Injected reviewed artifact transport.
        configured_limit (int): Maximum permitted download bytes.
        cancelled (Callable[[], bool]): Cancellation predicate.

    Returns:
        Path: Fsynced verified temporary file owned by the caller.

    Raises:
        AcquisitionCancelled: If cancellation is requested.
        AcquisitionError: If download, size, or digest verification fails.
    """
    if artifact.size is None:
        raise AcquisitionError("OCI artifacts cannot use the file downloader.")
    if artifact.size > configured_limit:
        raise QuotaExceeded("Artifact exceeds the configured download quota.")
    directory.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    fd, raw_name = tempfile.mkstemp(prefix=constants.RUNTIME_DOWNLOAD_PREFIX, dir=directory)
    path = Path(raw_name)
    digest = hashlib.sha256()
    received = 0
    try:
        with os.fdopen(fd, "wb") as output:
            streamed = False
            for final_url, chunks in transport.stream(artifact.source):
                final = urlparse(final_url)
                origins = {
                    urlparse(artifact.source).netloc,
                    *(urlparse(origin).netloc for origin in artifact.allowed_redirect_origins),
                }
                if final.scheme != "https" or final.netloc not in origins:
                    raise AcquisitionError(
                        "Artifact transport returned an unapproved final origin."
                    )
                if streamed:
                    raise AcquisitionError("Artifact transport returned multiple responses.")
                streamed = True
                for chunk in chunks:
                    if cancelled():
                        raise AcquisitionCancelled("Artifact acquisition was cancelled.")
                    received += len(chunk)
                    if received > artifact.size or received > configured_limit:
                        raise QuotaExceeded("Artifact download exceeds its declared byte limit.")
                    digest.update(chunk)
                    output.write(chunk)
            if not streamed:
                raise AcquisitionError("Artifact transport returned no response.")
            output.flush()
            os.fsync(output.fileno())
        expected = artifact.digest.removeprefix(constants.SHA256_PREFIX)
        if received != artifact.size or digest.hexdigest() != expected:
            raise IntegrityFailure("Artifact size or SHA-256 digest did not match its manifest.")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
