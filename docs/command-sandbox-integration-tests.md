# Command sandbox integration tests

Native sandbox validation is intentionally separate from the ordinary pytest suite. Unit tests
inject sandbox adapters, process handles, streams, readiness results, filesystem identities, and
interactive decisions; they must not launch native sandbox or host processes.

Provision dedicated, disposable CI workers for Linux x64 and ARM64, macOS x64 and ARM64, and
Windows x64 and ARM64. Each worker must install the platform wheel and its bundled native
components, then run the opt-in sandbox suite with no developer credentials and no shared working
directory. Preserve denial diagnostics as artifacts, but never command output or environment
values.

The sandbox suite must prove denial of reads and writes outside every granted root; `.loop` reads
and writes; writes to `.git`, `.gitignore`, and `.agentignore`; symlink, hard-link, mount, junction,
and reparse-point replacement escapes; public, private, localhost, IPv4, IPv6, Unix-socket,
named-pipe, Docker-socket, and VM-host networking; parent-process memory and inherited credentials;
Apple Events, LaunchServices, Win32 GUI calls, clipboard access, and equivalent process bridges;
detached and daemonized descendants; and CPU, memory, process-count, descriptor/handle, output,
file-size, core-dump, and wall-clock violations. Every denial test must also verify bounded cleanup
and that no descendant remains.

Race cases must repeatedly replace missing and existing protected paths during launch and exercise
symlink, junction, reparse-point, and mount transitions. Tests must inspect observed backend
behavior, not only generated profiles or launcher arguments. Linux jobs must run on kernels that
permit unprivileged user namespaces and test the x86-64 and AArch64 seccomp tables. macOS jobs must
verify Seatbelt inheritance. Windows jobs must verify AppContainer identity uniqueness, Bound File
System grants and denies, low integrity, Win32k disablement, UI limits, handle isolation, and the
kill-on-close Job Object.

A separate opt-in host suite must use an interactive test driver. It must prove that no process is
created before confirmation; approval applies once to the exact executable identity, argv, cwd,
environment, and incompatibility reason; rejection and headless calls create no process; existing
allow rules cannot suppress confirmation; sandbox failure never starts a host process
automatically; results and audit events identify `host`; and timeout cleanup removes the complete
host descendant tree while enforcing every applicable non-containment resource limit.

Native jobs fail closed when the required backend or capability is absent. They must never convert
an unavailable sandbox into an unrestricted execution test.
