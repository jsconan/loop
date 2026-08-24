"""Define provider-neutral backend failures."""

from typing import Any, ClassVar

from ..errors import Problem


class BackendError(Exception):
    """Describe a failure reported through a conversation backend.

    Args:
        message (str): Human-readable failure description.
        provider (str): Backend provider that reported the failure.
        operation (str): Backend operation that failed.
        status_code (int | None): HTTP response status when available.
        code (str | None): Provider error code when available.
        request_id (str | None): Provider request identifier when available.
        retry_after (float | None): Suggested retry delay in seconds when available.
        response_started (bool): Whether response events were emitted before the failure.
        details (Any | None): Opaque provider diagnostic details when available.
    """

    provider: str
    operation: str
    status_code: int | None
    code: str | None
    request_id: str | None
    retry_after: float | None
    response_started: bool
    details: Any | None
    recoverable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        operation: str,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        response_started: bool = False,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.operation = operation
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retry_after = retry_after
        self.response_started = response_started
        self.details = details

    def to_problem(self) -> Problem:
        """Return the safe application problem for this backend failure.

        Returns:
            Problem: Provider-neutral problem with available request diagnostics.
        """
        metadata = {}
        if self.status_code is not None:
            metadata["status_code"] = self.status_code
        if self.request_id:
            metadata["request_id"] = self.request_id
        if self.code:
            metadata["provider_code"] = self.code
        return Problem.from_exception(
            self,
            code="backend.request_failed",
            title="Backend request failed",
            retryable=self.recoverable,
            operation=self.operation,
            metadata=metadata,
        )


class BackendConnectionError(BackendError):
    """Indicate that a backend could not be reached."""

    recoverable = True


class BackendTimeoutError(BackendConnectionError):
    """Indicate that a backend operation timed out."""


class BackendResponseError(BackendError):
    """Indicate that a backend returned an invalid response."""


class BackendStatusError(BackendError):
    """Indicate that a backend rejected or failed an operation."""


class BackendBadRequestError(BackendStatusError):
    """Indicate that a backend rejected the request contents."""


class BackendAuthenticationError(BackendStatusError):
    """Indicate that backend authentication failed."""


class BackendPermissionDeniedError(BackendStatusError):
    """Indicate that the backend denied the requested operation."""


class BackendNotFoundError(BackendStatusError):
    """Indicate that a requested backend resource was not found."""


class BackendConflictError(BackendStatusError):
    """Indicate that the backend request conflicted with current state."""

    recoverable = True


class BackendRateLimitError(BackendStatusError):
    """Indicate that the backend rate limit was exceeded."""

    recoverable = True


class BackendServerError(BackendStatusError):
    """Indicate that the backend encountered a server-side failure."""

    recoverable = True
