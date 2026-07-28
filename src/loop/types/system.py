"""Define exceptions used to request controlled process shutdown."""


class ShutdownRequested(Exception):
    """Indicate that the process received a termination signal."""
