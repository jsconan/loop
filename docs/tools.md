# Built-in tools and command sandbox

Loop registers these tools for the model by default:

| Tool | What it does |
| --- | --- |
| `list_folder` | List files and folders, optionally recursively. |
| `read_text_file` | Read bounded UTF-8 text by line range. |
| `search_text` | Search selected files and folders with structured matches. |
| `write_text_file` | Write a UTF-8 file after authorization. |
| `edit_text_file` | Replace exact UTF-8 text with ambiguity checks. |
| `delete_path` | Delete an authorized file, link, or folder tree. |
| `get_current_datetime` | Return the local date and time. |
| `fetch_content` | Fetch authorized HTTP(S) text into a bounded cache. |
| `read_cached_content` | Read cached text by line or cursor. |
| `resolve_executable` | Look up a host tool and request a bounded sandbox read grant. |
| `run_command` | Run shell commands inside the host OS sandbox. |
| `activate_skill` | Load matching skill instructions. |
| `manage_skills` | Manage active skills and their bounded resources. |

## Edit files while a model is running

File mutations requested by an agent are checked against the source versions supplied in that model request. If you save a different version while the model is generating, Loop refuses the stale mutation before requesting approval. The tool result tells the model to reread the current file, reconsider its proposed edit, and preserve the intervening changes. When a read is authorized and complete before and after content fits the available read budget, the warning includes a bounded diff. Changes elsewhere in the same file also invalidate an edit whose exact match remains unchanged. Snapshot checks precede exact-match validation, so removed or duplicated edit anchors still produce stale-context guidance.

`read_text_file` returns a bounded text range and the SHA-256 identity of the complete source, including for empty files. Capturing this identity scans the whole file with bounded memory. Exact edits require their matched text to have been supplied to the model. Whole-file replacements and `replace_all` require complete visible coverage; consecutive or overlapping pages of the same version can provide it. Overlapping bytes must agree, and complete assembled content must match its source digest. Inconsistent snapshot bytes require a new read. Search results provide discovery information and require a file read before mutation. If a previously observed file is deleted during generation, a stale write cannot recreate it. The rejected write carries a trusted absence receipt, allowing the model to reconsider creation in its next request without a separate missing-file read. An explicit `read_text_file` absence report provides the same observation. Neither report authorizes another write in the original generation. Reconsidered creation still requires normal authorization and exclusively creates the file, preserving any competing recreation.

User-attached file snapshots and their cached continuations use their existing immutable versions. Reading an old cached snapshot does not refresh it from disk. The backend prepares file-bearing context before snapshot observation, preserving canonical stored history. Redacted read results and redacted or native-byte references retain their revision and existence without establishing visible source-text coverage; a newer opaque reference invalidates coverage from an older revision. A read performed in the same batch as an edit becomes usable only after its result is supplied in another model request. Successful model writes and deletions advance the known state using registry receipts. Edit receipts identify the source version; if retained ranges no longer match that source, or historical mutation arguments were redacted, Loop retains the committed revision but requires a new read rather than inventing visible bytes. Redaction of unrelated diagnostic metadata does not invalidate unchanged source content. Interrupted tool batches recover the latest matching generation view and replay completed mutations without accepting same-batch reads. Missing or inconsistent recovery history requires a new generation before file mutations, including creation. Recovery rejects ambiguous call IDs, duplicate results, and results without a matching call within one response. It never replays mutations from a later response into an older pending request. Context replay consumes at most one result per recorded call, while allowing later calls to reuse provider identifiers. Observations are reconstructed separately for each request and session; compacted summaries and legacy results without verified read receipts require another read.

These checks apply to agent file tools. Direct user commands and standalone registry calls without model observations retain their existing authorization behavior. Arbitrary shell commands write directly to the workspace and do not provide file snapshot checks. Only saved disk content is observed; unsaved editor buffers and semantic dependencies in other files are outside this protection.

## Run shell commands

`run_command` uses the host shell and works in the live workspace. Its default execution deadline is 60 seconds, and captured stdout and stderr are limited. A timeout can still return output captured before the deadline.

Commands run offline by default. Recognized private components such as `.ssh`, `.aws`, `.config`, `.loop`, and `Library` remain protected inside granted workspace, managed temporary, skill, tool and system runtime directories, including case variants. Loop may ask for approval to:

- Read a directory outside the workspace.
- Use the network.
- Change or create Git metadata.
- Expand other sandbox access.

For a recognized safe executable named directly in a command, Loop verifies the executable found on `PATH` and can grant access to that file alone. Its directory and neighboring files remain inaccessible. Other tools may need their installation directories: `resolve_executable` identifies these directories and returns a reference for use with `run_command`. Verified package directories can be added automatically; other directories require separate read approval, which can include neighboring files. Commands still run inside the sandbox.

The sandbox writes directly to the checkout. Concurrent commands can change the same workspace and session working files, and Loop does not roll back shell changes. On macOS, approved roots containing hardlinked regular files are refused. On Linux, ordinary linked files remain readable and executable, but become read-only; large sets can make their containing directories read-only too. Links to protected or private files found in the approved roots are refused.

Each conversation has one private working directory shared by command `$TMPDIR`, file tools using the exact `$TMPDIR` or `$TMPDIR/` prefix, and its command cache. For example, write with `printf draft > "$TMPDIR/report.txt"` and read with `read_text_file(path="$TMPDIR/report.txt")`. Loop passes the native path through the subprocess environment and leaves shell and embedded program text unchanged. File tools perform no general environment expansion and reject traversal outside the selected root. `/tmp` is an operating-system pathname, outside the session-retention guarantee for programs that ignore `$TMPDIR`. Files persist across commands, turns, interruptions, and saved-session resumes on the same machine. Starting or switching conversations selects an isolated directory without removing earlier files. Saved-session working directories live in a database- and workspace-specific namespace beside the session database and are retained when Loop exits; in-memory library stores retain files only for the store's lifetime. Session exports do not include these working files. Library integrations must keep session storage outside shared workspace or approved read roots to retain isolation between conversations. Durable generation bindings prevent silent recreation when bound data is missing or unsafe; such sessions fail with an explicit recovery error. Valid existing directories are adopted in place without copying. Private-data restrictions and link checks also apply inside session working storage; place ordinary intermediate files there and copy deliverables into the workspace.

Native advisory ownership locks serialize cooperating harness operations on the same saved session; independent sessions can run concurrently. Locks close on exec and live outside the command-writable leaf. Ownership does not revoke old writable descriptors. Library callers can retire working files with `SessionManager.discard_working_directory()`; it commits a fresh generation before cleaning the validated retired root and reports incomplete cleanup. Child processes inherit the sandbox restrictions. Linux verifies descendant cleanup before returning a successful result. On macOS, detached children can outlive command completion and cleanup while retaining their restrictions. Neither deletion, reopening nor an ownership lock establishes descendant termination or an exclusive-writer guarantee. Working files are ordinary retained filesystem state, without an atomic snapshot of them and the conversation database.

Logical paths and output sanitization reduce accidental native-path disclosure before previews, continuation caching and session recording. Arbitrary commands know a usable native path and can encode it, so sanitization cannot guarantee absolute path secrecy. The sandbox shares the host kernel and provides less isolation than a virtual machine. It does not protect against a malicious program already running outside the sandbox under your account, or impose strict limits on command memory use or process counts. Offline commands can still disclose readable data through output sent to the model or permitted local communication. Approved network access permits general connections; it does not filter by domain.

Loop asks before running common direct destructive shell commands, but it cannot predict every effect of arbitrary shell source. Commands that have workspace write access can change existing files without a separate prompt for each file.

If the sandbox fails, Loop reports the failure in the command result. Some eligible failures can offer a separate host retry, which runs without the OS sandbox and always needs its own approval. When the sandboxed command may already have changed something, the prompt warns that retrying may repeat its effects. Declining keeps the original command result. A Linux installation that fails verification never offers an unrestricted retry.

| Platform | Enforcement |
| --- | --- |
| macOS | Startup checks that Seatbelt allows a workspace write and blocks a protected read and an outside write. |
| Linux | Requires an authenticated native installation and an enforcement probe. Loop checks the installation before each command. |

Sandbox availability depends on the operating system’s enforcement capabilities and a successful startup check. Loop refuses sandboxed execution when those requirements are not met.

## Manage the Linux sandbox

The interactive Linux app provides `/sandbox` commands. Check status before running commands that depend on native enforcement:

```text
/sandbox status
/sandbox status details=true
```

The first command gives a short installed, not installed, or needs attention result. The detailed form inventories managed paths, systemd units, and loaded AppArmor profiles. It is read-only and does not authorize removal. If Loop cannot safely inspect part of the installation, it reports that part as unknown.

| Command | Effect |
| --- | --- |
| `/sandbox install` | Set up a fresh installation, cleaning authenticated partial state first. |
| `/sandbox refresh` | Alias for reinstall. Reviews removal and installation separately. |
| `/sandbox uninstall` | Remove an authenticated, idle installation. |
| `/sandbox reinstall` | Run uninstall, then install, with separate consent for each step. |

These management commands require fresh consent, a real terminal, and administrator authentication. The responsive UI first requests exclusive terminal access; press Enter or use `/respond` to hand it over. The normal editor stays suspended until the management operation settles, and the operation still requires its own fresh consent. Library and headless calls cannot elevate, and setup does not install OS packages. You can retry after a failure; Loop verifies the current installation before proceeding. Loop refuses removal when descendants are running, native state is ambiguous, files have changed, or private inventory cannot be verified. An incomplete operation is reported as a management failure with attempted steps, a short reason, and current status; native execution remains unavailable until setup is healthy.

The Linux runtime needs Bubblewrap, enforcing AppArmor, libapparmor, libseccomp, Python 3, and systemd/cgroup v2, together with the host's policy parser and management tools. Install missing prerequisites through your operating system's administrator tools, then run `/sandbox install`. Loop does not install them automatically. Offline commands use a separate network namespace, and only selected system resolver and certificate files are exposed.

Removal refuses while sandboxed commands or their descendants are live. An attempted removal can still block new commands and invalidate results from commands already running. After commands finish, retry removal or refresh in the main app. Loop keeps those commands confined and does not retry them automatically.

Linux stores a separate sandbox profile for each distinct access configuration, reusing it for commands with the same access. Temporary contents remain isolated and search copies remain read-only. An installation holds at most 29 command profiles, including the startup check. Different approved roots or executable access can consume additional profiles; use `/sandbox refresh` when capacity is exhausted. An app update that changes the native execution protocol also requires a reviewed refresh before commands can run.

## Search files

`search_text` searches folder contents or selected files. It supports smart-case literal matching, regular expressions, case control, Git-style globs, neighboring lines, and a global result limit. Folder searches honor ignore rules, skip binary files and symbolic links, and return paths relative to the selected folder.

When trusted ripgrep is available, Loop runs it with fixed arguments inside a read-only native sandbox. The helper receives only the files selected and authorized by the file tool; it cannot browse other workspace or private files. Otherwise, Loop uses a bounded Python regex matcher. The two matchers have different regex dialects. Patterns are limited to 1,000 characters and searches to 30 seconds; long lines and results can be truncated. A changed native authority causes search to fail instead of switching to the Python matcher.

The optional `tools` extra installs ripgrep in the workspace virtual environment for shell commands. That workspace-writable binary is not trusted as a helper for public search.

## File and web results

Direct file tools use the permissions of the process running Loop. `list_folder` follows `.gitignore` and `.agentignore` when discovering files, but an explicit file request can name an ignored path. Text reads report source size, range, truncation, and continuation position. Results larger than 20 KiB are stored outside conversation history and returned as a preview with a `read_cached_content` handle.

Web fetches require authorization and do not follow redirects implicitly. Cached web handles keep their source metadata when a session resumes. After a handle expires, fetch the content again with new network authorization. Other cached artifacts stay local to the process.
