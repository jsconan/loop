# Optional runtime SBOM and provenance review

Release pages, project pages, checksum listings, source manifests, and license files are useful
documentation, but they are not SBOM or build-provenance artifacts. The `sbom` strings in the
current manifest are therefore treated as upstream documentation rather than release authority.

`make release-provenance` is an optional maintainer audit. When used, it requires every pinned
executable archive and OCI image to carry both:

- a digest-pinned CycloneDX or SPDX document; and
- a digest-pinned SLSA provenance statement whose subject digest exactly equals the pinned runtime
  archive or OCI descriptor.

The evidence record also validates a credential-free immutable HTTPS source, its media type, and
the SHA-256 digest of the evidence bytes. A tag, release page, mutable registry tag, checksum page,
or statement naming a different subject fails the optional audit. Evidence is never synthesized
by Loop.

At the current pins, authoritative upstream material confirms release identities but does not
provide the complete optional evidence set for every selected artifact. Lima 2.2.0 publishes the
archive SHA-256 and links an ephemeral build run, but no retained CycloneDX/SPDX document or SLSA
statement is attached to the exact Darwin arm64 archive. Nerdctl 2.2.0 publishes the full archive,
its component inventory, and checksum, but its exact arm64 archive has no retained SBOM or SLSA
statement; the repository's visible artifact attestation for 2.2.1 names different subjects and
cannot be reused. Envoy documents the official distroless image, but the selected platform
descriptor still lacks exact digest-bound SBOM and provenance subjects. The sandbox image is built
locally from the checked-in Containerfile, digest-pinned Debian base, immutable Debian snapshot,
and exact package inventory; it is not a separately promoted Loop artifact.

These gaps do not block installation, native qualification, or release. The release gate instead
validates pinned HTTPS origins, redirect allowlists, bounded sizes, SHA-256 or OCI digests, exact
package inputs, safe installation, and the live runtime boundary.

Authoritative references:

- [Lima v2.2.0 release and archive digest](https://github.com/lima-vm/lima/releases/tag/v2.2.0)
- [Nerdctl v2.2.0 release and component inventory](https://github.com/containerd/nerdctl/releases/tag/v2.2.0)
- [Nerdctl v2.2.1 attestation naming different subjects](https://github.com/containerd/nerdctl/attestations/15541687)
- [Envoy official distroless image documentation](https://github.com/envoyproxy/envoy/blob/main/ci/README.md#distroless-envoy-image)
- [Debian snapshot service](https://snapshot.debian.org/)
- [GitHub artifact-attestation verification](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/verifying-the-provenance-of-binary-artifacts)
