# Linux runtime support matrix

Linux cutover requires a clean, zero-remediation local qualification result. This checked-in matrix
is intentionally **not yet qualified**. A row is accepted when the native verifier and real public
`run_command` corpus prove every predicate below on a clean host with only Loop installed. CI may
repeat that verification, but external CI evidence is not required.

| Architecture | Clean host identity | Local native result | userns/subids | private RootlessKit under LSM | cgroup memory/pids | CPU mode | quota/bytes | seccomp | OverlayFS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amd64 | Pending qualification | Pending local verification | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| arm64 | Pending qualification | Pending local verification | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

No row may be accepted if it needs administrator action, package installation, `/etc/subuid` or
`/etc/subgid` edits, a sysctl/LSM change, a system path change, or a persistent service. The host
attestor must reject such a host as `UnsupportedCapability`; release qualification may promote a
manifest after the architecture being released has one accepted row. A pending architecture is not
advertised or selected. No provenance, SBOM, SLSA/in-toto statement, artifact signature, published
image, registry, or external CI record is an acceptance requirement.
