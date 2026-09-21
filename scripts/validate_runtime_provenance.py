"""Validate optional upstream SBOM and provenance evidence when supplied."""

from loop.execution.runtime.download import HttpxArtifactTransport
from loop.execution.runtime.manifest import load_embedded_release
from loop.execution.runtime.provenance import validate_release_provenance


def main() -> None:
    """Validate optional evidence for every artifact that declares it."""
    validate_release_provenance(load_embedded_release(), HttpxArtifactTransport())


if __name__ == "__main__":
    main()
