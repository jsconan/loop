# Runtime third-party notices

The closed runtime inventory is Lima, nerdctl-full (including its bundled containerd, runc,
RootlessKit, BuildKit, and CNI components), the pinned Ubuntu guest image, the digest-pinned Debian
base plus exact Debian snapshot packages used for the one locally built sandbox image, and Envoy
distroless. Each pinned artifact records its upstream source, immutable SHA-256 or OCI digest,
version, and notices reference in the embedded runtime manifest. No artifact is installed globally
or outside Loop application data.

The upstream notices shipped with or referenced by the pinned inputs remain authoritative:

- Lima 2.2.0 is distributed under Apache-2.0; its archive includes the upstream license and
  dependency notices.
- nerdctl-full 2.2.0 includes nerdctl, containerd, runc, RootlessKit, BuildKit, CNI plugins, and
  their upstream license files. The bundle is installed intact in Loop-private runtime storage.
- The Ubuntu guest image and Debian sandbox image retain their package copyright and license data.
  The exact Debian packages are fixed by `src/loop/execution/runtime/inventory.json`.
- Envoy distroless 1.39.1 is digest-pinned and its Apache-2.0 notice is linked by the manifest.

SBOMs, signatures, and provenance statements may be reviewed when upstreams publish exact
digest-bound evidence, but they are not substitutes for the pinned origin, size, digest, and live
runtime checks and are not release prerequisites.
