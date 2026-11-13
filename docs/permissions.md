# Permissions and approvals

Loop checks tool requests against a workspace policy before they run. It validates the tool's arguments, plans the effects it can identify, then applies one decision to the complete plan. If a tool discovers another effect while it runs, Loop checks that effect before allowing it.

The default **supervised** policy allows scoped file reads and asks before most changes, network requests, and session changes. Shell commands run inside a restricted workspace sandbox; routine offline commands can run there without a separate prompt. Requests for additional access need approval. When no interactive user is available, a request that needs approval is denied unless a trusted user enabled YOLO mode.

## Inspect a decision

Start with these commands:

```text
/permissions
/permissions help
/permissions show effective
/permissions explain run_command process.execute "git status"
```

`show effective` includes workspace settings and temporary session overrides. `explain` shows the decision and which default, rule, or boundary determined it. Every operation has an action, such as `filesystem.read`, `filesystem.delete`, `network.request`, or `process.execute`. A decision can be `allow`, `ask`, or `deny`.

| Action | Applies to |
| --- | --- |
| `filesystem.list`, `filesystem.read` | Listing and reading files. |
| `filesystem.create`, `filesystem.replace`, `filesystem.delete` | Creating, changing, and deleting files. |
| `network.request` | Outbound HTTP requests. |
| `process.execute` | Running a process. Shell commands also need the separate sandbox access described below. |
| `session.mutate` | Changing agent session state. |

## Change defaults and rules

Defaults set what happens when no rule matches. Rules apply to a tool, action, and canonical resource; use `*` to match any value. For example:

```text
/permissions default set workspace filesystem.delete deny
/permissions rule add workspace deny delete_path filesystem.delete "/project/build/*"
/permissions rule list workspace
/permissions rule remove workspace <rule-id>
```

Use `ask` to require approval, `deny` to reject without prompting, and `allow` to run matching operations without a prompt. Deny rules take priority. An exact-target allow can take priority over an `ask` rule; among glob rules, `ask` takes priority over `allow`. Creating a rule with a wildcard tool, action, or resource prompts for confirmation because it may match more operations than you intend.

Each change uses one of two scopes:

| Scope | Where it applies |
| --- | --- |
| `workspace` | Saved for this workspace. |
| `session` | Active until this Loop process exits. |

When a permission policy cannot be validated, Loop reports the affected artifact and reason and asks for recovery. You can repair the file and reload, halt, or continue with permission operations blocked. Where a reset can safely archive the captured invalid file, Loop also offers that reset. Administrator policy and durable namespace failures require repair or restoration because their restrictions cannot safely be inferred. Invalid built-in presets can be excluded for the current startup only after explicit confirmation.

Use `/permissions default reset <scope> <action>` to reset a decision. A workspace reset restores the supervised default; a session reset removes the temporary override so the workspace value applies. Use `/permissions reset` or `/permissions reset session` to clear every session override. The explicit `/permissions reset workspace` and `/permissions reset user` commands archive and replace the complete selected policy with built-in defaults; the user reset removes user-wide remembered approvals and trusted host grants. To edit the workspace YAML file directly, run `/permissions reload`; Loop validates it and keeps session overrides active. Existing permission files must contain a non-empty YAML mapping; an empty document, an empty mapping, or a scalar is rejected because it may be a truncated policy. If the workspace policy file is missing on reload, Loop retains the active workspace policy. A complete reset leaves a `<policy-file>.reset-pending` marker until publication completes. If that marker remains, file-based policy loading and reload refuse; inspect the policy and any reset archive, restore the intended policy, and remove the marker before retrying. Policy saves and resets do not explicitly synchronize file contents or directory entries to durable storage, so recovery after a power loss is not guaranteed. If the user-wide policy is missing or cannot be loaded during recovery, Loop retains active deny and ask rules and discards remembered user-wide approvals.

Permission YAML uses version 1.2 by default. An explicit YAML version directive determines the interpretation of that document. Without a directive, numeric values use YAML 1.2 semantics, including decimal numbers with leading zeros.

Workspace and user permission policies have an integer `version` field; the current version is 2, and version 1 remains supported for existing policies. Durable policy files carry a generation linked to the independent workspace registry. Loop automatically establishes built-in workspace and user policies when a registered namespace has no policy, reservation, backup, or pending reset; a newly discovered workspace needs no permission setup. A valid older policy without a generation migrates conservatively, retains its explicit limits and rules, and archives its original bytes before publication; a missing local-network limit becomes a hard deny. A complete, registry-marked version 1 policy is likewise archived and published as version 2 before admission. If its workspace policy lacks `deny_local_networks`, migration sets it to `true`: the missing key could also be a truncated hard deny, so a trusted user must run `/permissions limit set workspace local-network allow` to enable it afterward. An older user policy gains an empty host-grant collection. Existing values, rules, and generation are preserved. Missing policy data from an initialized namespace, conflicting migration backups, incomplete version 2 content, malformed content, and copied or mismatched generations require recovery and deny affected decisions. Interactive startup reports the specific recovery error and stops before model work; repair or restore the named policy and registry from a trusted backup, then restart Loop. The trusted `/permissions initialize <workspace|user> <identity>` command and headless `loop --initialize-policy <workspace|user> <identity>` remain available for explicit recovery when no policy or backup data exists; they refuse conflicting state and must run outside a sandbox. If the entire same-user policy store and its independent registry are lost, Loop cannot distinguish that loss from a new installation and starts from built-in defaults; restore a trusted backup to retain previous rules. In-memory library policies have only the lifetime of their manager and do not provide durable migration protection.

The recovery initialization commands are `/permissions initialize workspace <workspace-identity>` and `/permissions initialize user <user-identity>`. Copy the exact identity from `/permissions show`; quote an identity containing spaces. Completion offers the configured identity while Loop is idle. Initialization refuses an existing policy, backup, or incomplete reset. Ordinary first use does not require these commands.

Policy saves refuse pre-existing staging entries that are links or special files. Cooperating Loop managers use an advisory lock and reject writes and resets when the policy bytes have changed since that manager loaded them; reload before retrying. A busy lock also refuses the update, and a reset refuses an existing archive name rather than overwriting it. Each Loop save records a new revision in a YAML comment, even when policy values return to an earlier state. Durable first-use initialization requires a verifiable backup directory listing; an inspection veto requires recovery. Direct editors do not take the lock, so finish editing before saving through Loop and reload afterward. A running manager with a durable namespace rechecks the registry identity, generation, and both policy files before each admission; a changed namespace denies further requests until reload or recovery. A manager without a durable namespace continues using its loaded policy until reload, except that a pending reset marker denies admission until trusted recovery and reload. The policy and lock directory must be trusted: these locks do not prevent host processes with the same user identity from replacing files or ignoring the lock. On systems without no-follow file opening, any existing staging entry prevents a save.

## Optional administrator policy

An administrator can install an additional restrictive policy in a fixed system directory: `/etc/loop` on Linux or `/Library/Loop` on macOS. The independent protected expectation is `/etc/loop.managed.expect.json` on Linux or `/Library/loop.managed.expect.json` on macOS; the policy file is `managed-policy.json` inside the respective Loop directory. Loop never selects these locations from workspace, user, or environment configuration. Only simultaneous absence of the expectation and policy directory under verified protected ancestry means managed enforcement is unavailable. Once either exists, a missing or invalid counterpart denies policy admission until repaired and reloaded; removing the policy directory cannot turn an expected installation into an unmanaged one. The directory, its ancestors, and both files must be real paths owned by root, without group or other write access, extended ACLs, or unsafe links. Policy files must have one hard link and be at most 64 KiB. A sandboxed command cannot request access to either protected path or a write scope containing it. On Linux, search-only traversal needs access to the held descriptor through `/proc/self/fd` for ACL inspection; if that check is unavailable, admission requires recovery rather than assuming the installation is absent.

The protected expectation records the administrator-selected identity and minimum acceptable revision. For example, `loop.managed.expect.json` can contain `{"version":1,"identity":"0123456789abcdef0123456789abcdef","minimum_revision":1}`. A matching `managed-policy.json` can contain:

```json
{
  "version": 1,
  "identity": "0123456789abcdef0123456789abcdef",
  "revision": 1,
  "require_sandbox": true,
  "forbid_host": true,
  "deny_local_networks": true,
  "forbid_yolo": true,
  "rules": [
    {"id": "no-network", "decision": "deny", "action": "network.request"},
    {"id": "review-deletion", "decision": "ask", "action": "filesystem.delete"}
  ]
}
```

Managed rules use the same typed tool, action, resource, and exact-target match fields as user rules, but accept only `deny` and `ask`. Every rule needs a distinct stable ID of 1–64 ASCII letters, digits, dots, underscores, or hyphens. A managed `ask` requires a fresh one-time human decision for each matching request; saved approvals, automatic approval, and YOLO cannot satisfy it. Under YOLO, that request is denied without a prompt. `require_sandbox` and `forbid_host` each prohibit host process execution; `deny_local_networks` disables sandbox-attempt IP loopback on qualified Linux and attempt-local Unix sockets on qualified macOS. `forbid_yolo` blocks no-prompt activation and overrides saved and session settings. A changed or invalid managed source also suspends automatic approval until trusted reload; a later policy relaxation requires an explicit activation in the running process. A previously saved user setting remains saved and may activate on restart if policy permits it. Sandbox commands still need their normal native enforcement. User rules, limits, grants, and YOLO cannot override managed denies or host restrictions. A matching user deny remains a deny.

An administrator should publish files with root ownership and restrictive modes using protected staging and atomic replacement, update the expectation revision floor when advancing policy, and increase the policy revision whenever the policy bytes change. Loop binds the loaded file identities and content digests. A replacement, disappearance, unsafe path, malformed content, or stale identity or revision blocks subsequent admissions. `/permissions show managed` reports unavailable, active identity and revision, or recovery required. After a trusted repair or update, run `/permissions reload` to bind the new revision; a changed policy for the same identity must have a higher revision. Removing an installed managed source during the same Loop process remains a recovery error. A root administrator can remove both protected entries and restart Loop to return to unmanaged operation.

The protected files resist modification by an ordinary same-user host process, but a root-owned file alone does not protect Loop's same-user process or force that process to be used. Managed decisions are authoritative only in a trusted, correctly installed Loop enforcement path. This feature does not claim resistance to an administrator, a compromised OS, or unrestricted same-user code that modifies or replaces the enforcing Loop process. The system policy contains restrictions only; it is not a substitute for native sandbox enforcement.

## Boundaries

Limits restrict what a policy can reach. Rules cannot bypass them. By default, Loop operations can read and write inside the workspace and the active conversation's private working directory (`app-temp`), addressed as `$TMPDIR/...` by file tools and through the inherited `TMPDIR` environment variable by commands. `$WORKSPACE_ROOT/...` addresses the workspace in file tools, and commands inherit `WORKSPACE_ROOT` for the same directory. Fresh scratch generations allow ordinary data names; existing restricted generations preserve private-name restrictions until explicitly discarded. Workspace controls, host-private data, and other sessions remain outside this exception. Saved-session files persist across restarts; when available, Linux `/tmp` refers to a separate subtree of session scratch, not the whole working directory. Namespace parents and ownership locks remain outside command authority; native metadata traversal does not permit private ancestor listings. Host-process access is off, and private or local network addresses are blocked.

File tools cannot delete the active working-root directory or a containing tree, even with an allow rule or prior approval. They can delete ordinary descendants and remove symbolic links without deleting their targets. Replacing the working root requires the session manager's explicit retirement operation.

Use limits to grant narrowly scoped access:

```text
/permissions limit add workspace read-root system-temp
/permissions limit add workspace network-origin https://my-host.local
/permissions limit set workspace private-network allow
/permissions limit set session local-network deny
/permissions limit set session host-process allow
/permissions limit classify workspace read-root approval
/permissions limit list
```

The supported roots include `workspace`, `app-temp`, and `system-temp`. Add `system-temp` only when Loop needs to read across applications' temporary files. `/permissions limit list` shows each effective filesystem and network-origin limit's value, hard or approval-controlled classification, scope, and source. Built-in read-root limits, including unchanged legacy copies of those defaults, permit an explicitly approved sandbox request for an enforceable external root. A customized legacy hard read-root limit can be changed to approval-controlled through a trusted prompt in the same command call; the command's exact external read scope is then reviewed separately. Deliberately set hard roots remain hard until a trusted user runs `/permissions limit classify <workspace|session> <read-root|write-root|network-origin> approval`. Automatic reads bound to a validated executable do not count as requested external data roots. The classification change makes future scoped requests eligible for approval without adding roots to the saved limit. Network origins accept glob patterns. The default `*` is a ceiling rather than authorization; adding the first specific origin replaces `*`, so the limit becomes narrower. Private-network access has its own hard limit. Outside YOLO mode, host-process eligibility must be enabled before policy can approve a host process.

When only the user-changeable host-process limit blocks an explicit host request, an interactive user can enable session eligibility and decide the exact request in the same call. The denial also shows `/permissions limit set session host-process allow` for a separate trusted user action. Enabling eligibility does not override a policy denial or approve a command. An approved host command runs without Loop's sandbox restrictions, with the host user's authority subject to independent OS and outer-environment restrictions; it can change files and sandbox configuration outside the workspace. Loop rechecks policy immediately before host launch, but a concurrent revocation after that final check cannot stop a process already starting or running.

An explicit host request under [YOLO mode](configuration.md#edit-settings-while-loop-is-running) can pass the user-changeable eligibility gate without changing it or asking for fresh host approval. It still displays a host-boundary notice and is audited as a YOLO decision. An explicit host deny or unavailable policy or enforcement still blocks it. YOLO does not turn a sandbox failure into a host request, replay the command, or persist a host grant. Ordinary automatic approval cannot authorize host execution.

Shell commands have a separate sandbox approval path. Routine commands run within the workspace sandbox. A request to read or write outside the workspace, use the network, change Git metadata, or expand other sandbox access asks separately. A `process.execute` rule does not grant these permissions or authorize running a command outside the sandbox. On qualified Linux, ordinary commands can communicate over IPv4 and IPv6 loopback with descendants in the same attempt; on qualified macOS, they can use pathname Unix stream and datagram sockets beneath `LOOP_LOCAL_SOCKET_DIR`. Neither path needs a separate model argument or approval. `/permissions limit set <workspace|session> local-network deny` or a managed `deny_local_networks` restriction disables the applicable local path for subsequently admitted attempts; stop an already running command to end its existing local communication. A general `network: true` command grant cannot enforce a hard private-network or local-network deny or a restricted origin list. The default private-network limit therefore denies that request with policy-change guidance. On qualified Linux or macOS installations, a command can instead request exact HTTP(S) `egress_origins`; Loop checks each origin against the effective network permission and hard origin limit and mediates approved connections while keeping direct networking isolated. With local networking denied, the mediated path remains usable, but the payload cannot use its ordinary local-network capability. User and managed denies still take precedence over YOLO. A stronger origin rule that cannot be enforced through visible HTTP requests or opaque HTTPS tunnels is unavailable rather than widened. A trusted user can deliberately change the user-controlled private-network limit and request general networking again; the resulting grant can reach private host services. Scoped IP loopback remains unqualified in the current macOS backend. See [Built-in tools and command sandbox](tools.md) for supported protocols, command behavior, and Linux sandbox management.

Command approval summarizes the workspace and additional access; choose **Show full scope** to inspect the complete bound paths and executable identities, then **Show reduced scope** to return to the summary before deciding.

## Permission presets

Presets replace defaults and rules in one selected scope. They do not change enforcement limits or the other scope. Loop shows the proposed changes and asks before replacing the policy.

```text
/permissions preset list
/permissions preset show supervised
/permissions preset diff workspace observe
/permissions preset replace workspace observe
```

The built-in presets are:

| Preset | Behavior |
| --- | --- |
| `observe` | Allow reads; deny other operations. |
| `supervised` | Allow reads; ask before other operations. |
| `workspace` | Allow reads and file creation or replacement; ask before other operations. |
| `locked` | Deny every operation that requires authority. |
| `unsupervised` | Allow operations, subject to enforcement limits. |

## Review and revoke approvals

`run_command` approvals are separate from policy rules. List or remove remembered sandbox approvals with `/permissions sandbox list` and `/permissions sandbox remove <session|workspace|user> <id>`. If Loop offers a host retry after a no-effects prelaunch failure, the user can approve that exact retry for this session; list or revoke these approvals with `/permissions host list` and `/permissions host remove <id>`. At an explicit host approval prompt, a trusted human can allow one run or remember the exact command for the session, workspace, or user. The same scopes are available through `/permissions host trust <session|workspace|user> '<command>'` for workspace-root commands. List grants with `/permissions host trusted` and revoke them with `/permissions host untrust <session|workspace|user> <id>`. Session grants expire with this Loop process and bind session scratch; workspace and user grants persist in the validated user policy and survive restart. Workspace grants bind the registered workspace identity; user grants can match in another qualified workspace. All grants bind the exact shell source, working directory, process `PATH`, and explicit host tool. Ordinary workspace configuration cannot create a host grant. A trusted grant does not enable host-process eligibility or override a host policy deny. A new agent-requested attempt can repeat effects; the exact grant authorizes that new request without a prompt, while a session retry approval applies only to independently eligible no-effects prelaunch retries.

Ordinary permission decisions are recorded in the session timeline and application audit database. Explicit host admission, denial, grant changes and execution outcomes use minimized audit records containing a command digest and identifiers, without shell source, output or credentials. Canonical tool calls and complete tool traces retain their original command and result content. Export audit events as JSONL with `/app audit <destination>`. Replacing an existing destination requires `force=true`.
