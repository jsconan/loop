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

## Run shell commands

`run_command` uses the host shell and works in the live workspace. Its default execution deadline is 60 seconds, and captured stdout and stderr are limited. A timeout can still return output captured before the deadline.

Commands run offline by default. Recognized private components such as `.ssh`, `.aws`, `.config`, `.loop`, and `Library` remain protected inside granted workspace, managed temporary, skill, tool and system runtime directories, including case variants. Loop may ask for approval to:

- Read a directory outside the workspace.
- Use the network.
- Change or create Git metadata.
- Expand other sandbox access.

For a recognized safe executable named directly in a command, Loop verifies the executable found on `PATH` and can grant access to that file alone. Its directory and neighboring files remain inaccessible. Other tools may need their installation directories: `resolve_executable` identifies these directories and returns a reference for use with `run_command`. Verified package directories can be added automatically; other directories require separate read approval, which can include neighboring files. Commands still run inside the sandbox.

The sandbox writes directly to the checkout. Concurrent commands can change the same files, and Loop does not roll back shell changes. On macOS, approved roots containing hardlinked regular files are refused. On Linux, ordinary linked files remain readable and executable, but become read-only; large sets can make their containing directories read-only too. Links to protected or private files found in the approved roots are refused.

Child processes inherit the sandbox restrictions. Linux verifies descendant cleanup before returning a successful result. On macOS, detached children can outlive cleanup while retaining their restrictions.

The sandbox shares the host kernel and provides less isolation than a virtual machine. It does not protect against a malicious program already running outside the sandbox under your account, or impose strict limits on command memory use or process counts. Offline commands can still disclose readable data through output sent to the model or permitted local communication. Approved network access permits general connections; it does not filter by domain.

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

These management commands require fresh consent, a real terminal, and administrator authentication. Library and headless calls cannot elevate, and setup does not install OS packages. You can retry after a failure; Loop verifies the current installation before proceeding. Loop refuses removal when descendants are running, native state is ambiguous, files have changed, or private inventory cannot be verified. An incomplete operation is reported as a management failure with attempted steps, a short reason, and current status; native execution remains unavailable until setup is healthy.

The Linux runtime needs Bubblewrap, enforcing AppArmor, libapparmor, libseccomp, Python 3, and systemd/cgroup v2, together with the host's policy parser and management tools. Install missing prerequisites through your operating system's administrator tools, then run `/sandbox install`. Loop does not install them automatically. Offline commands use a separate network namespace, and only selected system resolver and certificate files are exposed.

Removal refuses while sandboxed commands or their descendants are live. An attempted removal can still block new commands and invalidate results from commands already running. After commands finish, retry removal or refresh in the main app. Loop keeps those commands confined and does not retry them automatically.

Each command receives fresh private temporary and cache space through `$TMPDIR`. Commands can create synthetic private-name fixtures there for tests. Literal `/tmp` in shell source maps to Loop's file-tool temporary area, which retains private-data restrictions. Linux stores a separate sandbox profile for each distinct access configuration, reusing it for commands with the same access. Temporary contents remain isolated and search copies remain read-only. An installation holds at most 29 command profiles, including the startup check. Different approved roots or executable access can consume additional profiles; use `/sandbox refresh` when capacity is exhausted. An app update that changes the native execution protocol also requires a reviewed refresh before commands can run.

## Search files

`search_text` searches folder contents or selected files. It supports smart-case literal matching, regular expressions, case control, Git-style globs, neighboring lines, and a global result limit. Folder searches honor ignore rules, skip binary files and symbolic links, and return paths relative to the selected folder.

When trusted ripgrep is available, Loop runs it with fixed arguments inside a read-only native sandbox. The helper receives only the files selected and authorized by the file tool; it cannot browse other workspace or private files. Otherwise, Loop uses a bounded Python regex matcher. The two matchers have different regex dialects. Patterns are limited to 1,000 characters and searches to 30 seconds; long lines and results can be truncated. A changed native authority causes search to fail instead of switching to the Python matcher.

The optional `tools` extra installs ripgrep in the workspace virtual environment for shell commands. That workspace-writable binary is not trusted as a helper for public search.

## File and web results

Direct file tools use the permissions of the process running Loop. `list_folder` follows `.gitignore` and `.agentignore` when discovering files, but an explicit file request can name an ignored path. Text reads report source size, range, truncation, and continuation position. Results larger than 20 KiB are stored outside conversation history and returned as a preview with a `read_cached_content` handle.

Web fetches require authorization and do not follow redirects implicitly. Cached web handles keep their source metadata when a session resumes. After a handle expires, fetch the content again with new network authorization. Other cached artifacts stay local to the process.
