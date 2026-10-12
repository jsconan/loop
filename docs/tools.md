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

Commands run offline by default and cannot read protected paths. Loop may ask for approval to:

- Read a directory outside the workspace.
- Use the network.
- Change or create Git metadata.
- Expand other sandbox access.

The sandbox writes directly to the checkout. Concurrent commands can change the same files, and Loop does not roll back shell changes. A root that contains a hardlinked regular file is refused before launch. Child processes inherit the sandbox restrictions; detached children can outlive command cleanup but cannot escape those restrictions.

Loop asks before running common direct destructive shell commands, but it cannot predict every effect of arbitrary shell source. Commands that have workspace write access can change existing files without a separate prompt for each file.

If the sandbox fails, Loop reports a typed result. Some eligible failures can offer a separate host retry, which runs without the OS sandbox and always needs its own approval. When the sandboxed command may already have changed something, the prompt warns that retrying may repeat its effects. Declining keeps the original command result. Invalid Linux installation authority never offers an unrestricted retry.

| Platform | Enforcement |
| --- | --- |
| macOS | Startup checks that Seatbelt allows a workspace write and blocks a protected read and an outside write. |
| Linux | Requires an authenticated native installation and an enforcement probe. Loop checks the installation before each command. |

Linux support depends on host capabilities and trusted tools, not the distribution or CPU name.

## Manage the Linux sandbox

The interactive Linux app provides `/sandbox` commands. Check status before running commands that depend on native enforcement:

```text
/sandbox status
/sandbox status details=true
```

The first command gives a short installed, not installed, or needs attention result. The detailed form inventories managed paths, systemd units, and loaded AppArmor profiles. It is read-only and does not authorize removal. Inspection is limited to 512 private entries and 64 unit fragments; an inaccessible or larger inventory is reported as unknown.

| Command | Effect |
| --- | --- |
| `/sandbox install` | Set up a fresh installation, cleaning authenticated partial state first. |
| `/sandbox refresh` | Alias for reinstall. Reviews removal and installation separately. |
| `/sandbox uninstall` | Remove an authenticated, idle installation. |
| `/sandbox reinstall` | Run uninstall, then install, with separate consent for each step. |

These management commands require fresh consent, a real terminal, and administrator authentication. Library and headless calls cannot elevate, and setup does not install OS packages. Loop blocks unsafe removal when descendants are running, files have changed, or private inventory cannot be verified. Failed operations provide a short reason and point to detailed status when useful.

The Linux runtime uses Bubblewrap, AppArmor, seccomp, and systemd/cgroup v2. The host must provide these native features and trusted tools. Loop exposes only selected resolver, hosts, NSS, and CA files, and refuses unsafe or dangling inputs. Offline commands use a separate network namespace. An installation accepts at most 29 payload profiles, including operational probes; fresh scratch paths can also consume entries, so the limit may be reached before 29 commands. Refresh the installation when it is exhausted.

## Search files

`search_text` searches folder contents or selected files. It supports smart-case literal matching, regular expressions, case control, Git-style globs, neighboring lines, and a global result limit. Folder searches honor ignore rules, skip binary files and symbolic links, and return paths relative to the selected folder.

When trusted ripgrep is available, Loop runs it with fixed arguments inside a read-only native sandbox. Otherwise, Loop uses a bounded Python regex matcher. The two matchers have different regex dialects. Patterns are limited to 1,000 characters and searches to 30 seconds; long lines and results can be truncated. A changed native authority causes search to fail instead of switching to the Python matcher.

The optional `tools` extra installs ripgrep in the workspace virtual environment for shell commands. That workspace-writable binary is not trusted as a helper for public search.

## File and web results

Direct file tools use the permissions of the process running Loop. `list_folder` follows `.gitignore` and `.agentignore` when discovering files, but an explicit file request can name an ignored path. Text reads report source size, range, truncation, and continuation position. Results larger than 20 KiB are stored outside conversation history and returned as a preview with a `read_cached_content` handle.

Web fetches require authorization and do not follow redirects implicitly. Cached web handles keep their source metadata when a session resumes. After a handle expires, fetch the content again with new network authorization. Other cached artifacts stay local to the process.
