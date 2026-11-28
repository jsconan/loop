# Permissions and approvals

Loop checks tool requests against a workspace policy before they run. It validates the tool's arguments, plans the effects it can identify, then applies one decision to the complete plan. If a tool discovers another effect while it runs, Loop checks that effect before allowing it.

The default **supervised** policy allows scoped file reads and asks before most changes, network requests, and session changes. Shell commands run inside a restricted workspace sandbox and default to `process.execute: ask`. Users can approve the reviewed command once or remember it for the session, workspace or user when policy permits. Resource permissions and hard denials still apply. When no interactive user is available, a request that needs approval is denied unless a trusted user enabled YOLO mode.

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
| `network.request` | Outbound HTTP requests and exact mediated HTTP(S) origins; default `ask`. |
| `network.local` | Attempt-local communication; default `allow`. macOS additionally requires `local_ipc: true` or ingress. |
| `network.broad` | General networking, including host and LAN reach; default `ask`, subject to hard destination ceilings. |
| `network.ingress` | Publishing one attempt service at an exact `tcp://127.0.0.1:<port>`; default `ask`. |
| `filesystem.execute` | Native executable access. Readable interpreter source is not a code-trust boundary. |
| `process.execute` | Running a process. Sandbox file capabilities and direct-host consent are separate. |
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

When a user or workspace permission policy cannot be validated at startup, Loop reports the affected artifact and reason and offers reload after repair, temporary built-in defaults, an archived reset to defaults, or exit. Temporary defaults leave the selected file intact; a later explicit permission edit/save archives its original bytes and replaces it with the current policy. Changed files require another reload before replacement. Other valid policy layers, administrator restrictions and built-in hard denials remain enforced. If a live reload fails, repair and retry it or restart Loop to open startup recovery. Administrator policy requires repair by its owner and cannot be bypassed with user defaults. Invalid built-in presets can be excluded for the current startup only after explicit confirmation.

Use `/permissions default reset <scope> <action>` to reset a decision. A workspace reset restores the supervised default; a session reset removes the temporary override so the workspace value applies. Use `/permissions reset` or `/permissions reset session` to clear every session override. The explicit `/permissions reset workspace` and `/permissions reset user` commands archive and replace the complete selected policy with built-in defaults; the user reset removes user-wide remembered approvals and trusted host grants. To edit the workspace YAML file directly, run `/permissions reload`; Loop validates it and keeps session overrides active. Existing permission files must contain a non-empty YAML mapping; an empty document, an empty mapping, or a scalar is rejected because it may be a truncated policy. The main application recreates a missing workspace policy on reload with defaults and notice; a standalone library manager without automatic initialization retains its active policy. A complete reset leaves a `<policy-file>.reset-pending` marker until publication completes. If that marker remains, normal file-based loading requires recovery; inspect the policy and its archives, then repair/reload or explicitly select temporary defaults or an archived reset. Saving selected recovery defaults archives the marker as well and clears it only after policy publication. Policy saves and resets do not explicitly synchronize file contents or directory entries to durable storage, so recovery after a power loss is not guaranteed. For standalone managers without automatic initialization, an unavailable user-wide policy retains active deny and ask rules and discards remembered user-wide approvals until recovery.

Permission YAML uses version 1.2 by default. An explicit YAML version directive determines the interpretation of that document. Without a directive, numeric values use YAML 1.2 semantics, including decimal numbers with leading zeros.

Workspace and user permission policies use integer schema version 10. Loop backs up and migrates valid versions 1 through 9 before use. Durable policy files carry a generation linked to the independent workspace registry. Loop creates missing workspace and user policies with built-in defaults, including when a previously initialized file has been deleted, and reports the creation. Existing backups remain available for restoration; a newly discovered workspace needs no permission setup. A valid older policy without a generation migrates conservatively, retains its explicit limits and rules, and archives its original bytes before publication; a missing local-network limit becomes a hard deny. A complete, registry-marked version 1 policy is likewise archived and published with the current version before admission. If its workspace policy lacks `deny_local_networks`, migration sets it to `true`: the missing key could also be a truncated hard deny, so a trusted user must run `/permissions limit set workspace local-network allow` to enable it afterward. An older user policy gains an empty host-grant collection. Migration removes the obsolete `allow_host_processes` field without transferring either value to the trusted application setting. Older host grants retain their inactive version-one approval semantics until the user renews them. Existing values, rules, and generation are preserved. The newly explicit `filesystem.execute` capability defaults to allow, retaining existing native execution behavior. Saved process asks and denies remain unchanged; ambiguous older defaults retain ask. `process.execute` defaults to `ask`. Existing explicit `allow` choices remain saved; `/permissions default set workspace process.execute ask` restores command approval for those policies. Versions 1 and 2 gain the previously absent empty `hard_denied_paths` collection while retaining their other limits. Conflicting migration backups, incomplete or malformed policy content, and copied or mismatched generations open startup recovery before model work. Select temporary defaults to continue without modifying the file, an archived reset to replace it, or repair and reload it. A damaged independent registry requires repair of the workspace identity before policy admission. The trusted `/permissions initialize <workspace|user> <identity>` command and headless `loop --initialize-policy <workspace|user> <identity>` remain available for explicit recovery when no policy or backup data exists; they refuse conflicting state and must run outside a sandbox. If the entire same-user policy store and its independent registry are lost, Loop cannot distinguish that loss from a new installation and starts from built-in defaults; restore a trusted backup to retain previous rules. In-memory library policies have only the lifetime of their manager and do not provide durable migration protection.

The recovery initialization commands are `/permissions initialize workspace <workspace-identity>` and `/permissions initialize user <user-identity>`. Copy the exact identity from `/permissions show`; quote an identity containing spaces. Completion offers the configured identity while Loop is idle. Initialization refuses an existing policy, backup, or incomplete reset. Ordinary first use does not require these commands.

Policy saves refuse pre-existing staging entries that are links or special files. Cooperating Loop managers use an advisory lock and reject writes and resets when the policy bytes have changed since that manager loaded them; reload before retrying. A busy lock also refuses the update, and a reset refuses an existing archive name rather than overwriting it. Each Loop save records a new revision in a YAML comment, even when policy values return to an earlier state. Direct editors do not take the lock, so finish editing before saving through Loop and reload afterward. A running manager with a durable namespace rechecks the registry identity, generation, and both policy files before each admission; a changed namespace denies further requests until reload or recovery. A manager without a durable namespace continues using its loaded policy until reload, except that a pending reset marker denies admission until trusted recovery and reload. The policy and lock directory must be trusted: these locks do not prevent host processes with the same user identity from replacing files or ignoring the lock. On systems without no-follow file opening, any existing staging entry prevents a save.

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
  "forbidden_backend_options": ["account_lookup", "process_pools"],
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

Limits restrict what a policy can reach. Rules cannot bypass them. By default, direct file tools can read and request writes inside the workspace and the active conversation's private working directory (`app-temp`), addressed as `$TMPDIR/...` by file tools and through the inherited `TMPDIR` environment variable by commands. `$WORKSPACE_ROOT/...` addresses the workspace in file tools, and commands inherit `WORKSPACE_ROOT` for the same directory. Fresh scratch generations allow ordinary data names; existing restricted generations preserve private-name restrictions until explicitly discarded. Workspace controls, host-private data, and other sessions remain outside this exception. Native application-state protection excludes only the bound agent-data scratch subtree, including when it lives beneath the platform application-data directory; explicit hard-denied paths still apply inside scratch. The installed Linux receiver must support this separation or commands require `/sandbox refresh`. Saved-session files persist across restarts; when available, Linux `/tmp` refers to a separate subtree of session scratch, not the whole working directory. Namespace parents and ownership locks remain outside command authority; native metadata traversal does not permit private ancestor listings. Direct-host execution is disabled by default and has a separate trusted gate and authorization boundary. Private or local network addresses remain blocked for restricted operations.

File tools cannot delete the active working-root directory or a containing tree, even with an allow rule or prior approval. They can delete ordinary descendants and remove symbolic links without deleting their targets. Replacing the working root requires the session manager's explicit retirement operation.

Native commands and direct file tools deny reads and mutations of private host data and environment-value names such as `.env`, `.env.local`, and `.env.production`. A final `.example`, `.sample`, or `.template` suffix makes an `.env.*` component an ordinary template, including a directory with ordinary descendants; a protected child such as `.env.local` remains private. Trusted policy can still designate the template itself private. Ignoring a file with `.gitignore` or `.agentignore` affects discovery, not its secrecy classification. Ordinary data in private session scratch remains accessible. Workspace control files retain their mutation guards, and direct tools also deny reading workspace `.git` data.

Trusted workspace policy may set `limits.hard_denied_paths` to a list of exact, normalized absolute file or directory paths; administrator policy has a top-level `hard_denied_paths` list. Each list accepts at most 256 entries, including names that do not yet exist. Relative paths, parent traversal, glob syntax, backslashes, NULs and line breaks are rejected. Directory designations cover descendants. Edit the trusted policy source and reload it through `/permissions reload`; ordinary allow rules, remembered approval and YOLO cannot remove a hard designation. Removing one trusted template designation changes only that designation and does not disable built-in environment-value denials. Cached file continuations retain private source provenance and recheck current policy, including after session restoration; older nonreloadable artifacts without provenance require a new source read. Policy changes cannot retract bytes already returned. Directory search and listing apply child restrictions before traversal or reading discovery metadata. A child that requires separate review is skipped during enumeration and can be requested directly.

Direct tools open selected files through no-follow descriptors and bind device, inode, change timestamp and link count before returning content. Mutations bind parent descriptors and approved entry identities. Multiply linked host files require complete, stable alias accounting within the effective directory roots, with every alias outside hard protection. Missing, protected or changing aliases are refused even after ordinary approval. Mutation previews bind that verdict to the selected inode, change timestamp and link count; atomic replacement changes only the selected name, preserving the other links. Native commands separately enforce the shared hard classification through their operating-system boundary.

Native admission binds the trusted hard-path snapshot and approving policy revision. A bounded no-follow census covers all admitted data roots, deduplicates overlapping roots, and compares every regular inode's link count with its observed names. An alias to protected data, a protected symlink, a changing tree, or more than 100,000 directory entries refuses that grant before command effects. On Linux, an incomplete alias set can be admitted only when every observed name is inside an exact trusted denied-directory mask. A request that requires a working directory, scratch, executable, read alias or read or write root inside a masked directory is refused. Any visible name in an incomplete alias set also refuses admission. A hard designation inside an authenticated system-runtime baseline refuses the encompassing baseline when its aliases cannot be accounted for. On macOS, existing aliases that span writable and read-only scopes refuse admission, and new links require source write authority. Native rules keep future protected names and protected ancestor relocation denied. Updating the installed Linux receiver requires `/sandbox refresh`.

The live checkout is shared with ordinary editors and separately authorized writers. Approved paths are not a transactional or immutable inode view: another writer can publish bytes or aliases after admission. Loop does not promise independent inode isolation between overlapping grants. Existing hard-resource rules and admission checks still apply; this limit does not authorize bypassing them.

A tightened hard policy cancels active native attempts owned by the current Loop manager and rejects stale admissions, including a change during final admission callbacks. Cancellation requests cleanup and preserves the backend's cleanup result; it does not retroactively restrict an existing process or descriptor, and macOS cannot generally prove detached-child retirement. Concurrent commands intentionally share checkout content: a writer approved to read a source and write the checkout can publish those bytes there. Independently launched host programs remain outside this boundary.

Use limits to grant narrowly scoped access:

```text
/permissions limit add workspace read-root system-temp
/permissions limit add workspace network-origin https://my-host.local
/permissions limit set workspace private-network allow
/permissions limit set session local-network deny
/permissions limit classify workspace read-root approval
/permissions limit list
```

The supported roots include `workspace`, `app-temp`, and `system-temp`. Add `system-temp` only when Loop needs to read across applications' temporary files. `/permissions limit list` shows each effective filesystem and network-origin limit's value, hard or approval-controlled classification, scope, and source. Built-in read-root limits, including unchanged legacy copies of those defaults, permit an explicitly approved sandbox request for an enforceable external root. A customized legacy hard read-root limit can be changed to approval-controlled through a trusted prompt in the same command call; the command's exact external read scope is then reviewed separately. Deliberately set hard roots remain hard until a trusted user runs `/permissions limit classify <workspace|session> <read-root|write-root|network-origin> approval`. Automatic reads bound to a validated executable do not count as requested external data roots. The classification change makes future scoped requests eligible for approval without adding roots to the saved limit. Network origins accept glob patterns. The default `*` is a ceiling rather than authorization; adding the first specific origin replaces `*`, so the limit becomes narrower. Private-network access has its own hard limit. Host eligibility is controlled only by trusted `app.allow_direct_host_execution`; no `/permissions limit host-process` setting exists.


Shell commands have a separate sandbox approval path. New policies require approval for sandbox process execution (`process.execute: ask`), and provide private session scratch. The prepared scope shows any additional workspace writes. Workspace reads do not require a separate default approval. A trusted user can explicitly set `process.execute` to `allow` for automatic sandbox commands; this changes command prompting only, not native confinement, filesystem grants, hard denials, or direct-host eligibility. Requested write roots are evaluated for create, replace and delete capabilities. Filesystem allow admits the scope, ask requests approval, and deny refuses it. Additional reads, network access and Git metadata authority receive their own policy checks. Implicit native runtime roots and private scratch are also evaluated against ordinary filesystem rules. Scratch does not need a default write approval, but explicit scratch restrictions and default denies still apply. A narrower deny or withheld ask under a directory capability refuses the encompassing request when it cannot be compiled into the native boundary; approving the parent never overrides that restriction. Native exec currently uses an encompassing capability: narrower exec restrictions refuse the command rather than pretending to restrict what readable interpreter source can do. A `process.execute` rule does not grant these permissions or authorize running a command outside the sandbox. On qualified Linux, ordinary commands can communicate over IPv4 and IPv6 loopback with descendants in the same attempt when `network.local` allows it. On macOS, `local_ipc: true` additionally requests pathname Unix stream and datagram sockets beneath `LOOP_LOCAL_SOCKET_DIR`; ordinary offline commands create no IPC attachment. A local `ask` resolves before setup, and `deny` creates no local socket resource. `/permissions limit set <workspace|session> local-network deny` or a managed `deny_local_networks` restriction disables the applicable local path for subsequently admitted attempts; stop an already running command to end its existing local communication. A general `network: true` command grant cannot enforce a hard private-network or local-network deny or a restricted origin list. The default private-network limit therefore denies that request with policy-change guidance. On qualified Linux or macOS installations, a command can instead request exact HTTP(S) `egress_origins`; Loop checks each origin against the effective network permission and hard origin limit and mediates approved connections while keeping direct networking isolated. With local networking denied, the mediated path remains usable, but the payload cannot use its ordinary local-network capability. User and managed denies still take precedence over YOLO. A stronger origin rule that cannot be enforced through visible HTTP requests or opaque HTTPS tunnels is unavailable rather than widened. A trusted user can deliberately change the user-controlled private-network limit and request general networking again; the resulting grant can reach private host services. Scoped IP loopback remains unqualified in the current macOS backend. See [Built-in tools and command sandbox](tools.md) for supported protocols, command behavior, and Linux sandbox management.

With `process.execute: ask`, review the command and its proposed filesystem, runtime and network access. Choose **Deny** to decline or **Allow once** to approve only this attempt. When eligible, **Allow for this session**, **Allow in this workspace** and **Allow for this user** remember the reviewed command scope. When the application stores trusted workspace policy outside the checkout, new workspace approvals are saved with that policy, independently of other workspaces. Existing workspace approvals saved in user policy remain available for review and revocation. **Show full scope** displays the bound paths and executable identities. **Show reduced scope** returns to the summary. A fresh-approval rule, destructive-command review or ingress request requires a separate decision for the current attempt. These limits also apply when a broader remembered rule exists.

### Ordinary command writes

Omitting `write_roots` or passing `write_roots: null` with no exact file grants uses the ordinary workspace policy. Its base scope includes private session scratch. The owner can include workspace writes in the base scope with `/permissions sandbox workspace-write on`. An empty `write_roots` list requests private scratch for directory writes. For an interactive command using the ordinary workspace policy, Loop keeps recognized reads and scratch operations narrow. When an operation may write to the workspace, Loop prepares that authority for review if policy permits. This includes launchers whose output paths are determined at runtime. Approving the command authorizes the displayed scope for the chosen lifetime. Automatic process approval keeps opaque commands at the workspace default when there is no other reason to review the command. Inferred workspace writes still require matching consent. Explicit directories, exact files, private scratch scopes and saved tasks retain their declared scopes. Process rules, filesystem policy, administrator limits and hard protections apply to every request. Recognized destructive commands still require fresh review. With scratch-only native writes, simple redirections to a fixed path beneath `$TMPDIR` or `$LOOP_OUTPUT_DIR` do not add a fresh review.

### Optional sandbox capabilities

The command includes any backend resources it needs in the same process approval. On an enrolled workspace the agent can request one or more advertised `runtime_services`. The prompt describes the prepared scope and its effects. An owner-enabled workspace service also applies to recognized project commands. Platform-specific mechanisms are implemented by the backend.

| Resource | Authority and limits |
| --- | --- |
| Read-only access to hard-linked files | A qualified Linux backend offers an account-home metadata lookup when a visible hardlink group has names outside the selected scope. It respects read policy and omits protected paths. The command sees verified shared files through its approved paths with read-only enforcement. |
| Look up system users and groups | The host directory service can return account and group metadata, including directory-supplied password fields, and contact directory servers while direct command networking is disabled. |
| Use process pools for parallel work | Stock runtimes can use host POSIX named semaphores. The native owner filter protects live unrelated objects. Names and capacity are shared, and forced termination can leave a same-user object that another command can open or unlink. |

Host account lookup and stock process pools default to off and can be requested independently. A protected administrator policy can prohibit any option key using `forbidden_backend_options`. The command tool advertises only services available under the current backend and enrollment. An imported checkout policy cannot enable them. Owners can inspect the workspace settings with `/permissions show` and revoke an option with `/permissions sandbox <backend-option> off`. Revocation invalidates past command approvals containing that option in the workspace and requests cancellation of affected owned jobs. A subsequent explicit request requires a new decision. Saved tasks use only the services in their owner-controlled definition.

### Sandbox paths

Use `/permissions path list` to inspect the system and installed runtime presets and your saved path rules. System and runtime presets supply default read and execution access. Saved grants and restrictions affect sandboxed commands and their descendants.

| Access | Effect |
| --- | --- |
| `read` | Make an existing directory readable and executable. Writes still require their own scope and approval. |
| `write` | Supply read access and a reusable grant for an explicit `write_roots` request at or beneath this directory. |
| `write-file` | Supply a reusable grant for an explicit `write_files` request at this exact workspace path. The parent must already exist. |
| `read-only` | Supply read and execution access and refuse any writable scope that overlaps this directory. |
| `deny` | Block an exact file or directory, its descendants and the current target of a denied symlink. |

Use canonical absolute paths for grants. Directory and exact-file grants retain their declared kind, even if the path changes later. Restrictions take precedence across session, workspace and user scopes. An enclosing write request that includes a `read-only` directory is refused so the command can request a smaller scope. An ordinary command uses the owner’s selected scope or supplies its own write paths. Process approval, explicit filesystem rules, hard resource limits and administrator restrictions still apply. A denial intersecting a required native system mount can make native execution unavailable until the conflict is resolved.

```text
/permissions path list
/permissions path set workspace read '/opt/my runtime'
/permissions path set workspace write '/path/to/build-out'
/permissions path set workspace write-file '/absolute/path/to/workspace/.coverage'
/permissions path set user read-only '/path/to/reference-data'
/permissions path set user deny '/opt/my runtime/private'
/permissions path remove user '/opt/my runtime/private'
```

Session rules last until the session policy resets or Loop exits. Workspace and user rules are stored in the corresponding policy's `sandbox_paths` mapping. Existing `allow` entries load as `read`. Removing a rule restores applicable presets and decisions from other scopes. Changes apply to new commands, and Loop requests cancellation for live attempts affected by withdrawn grants or new restrictions. To disable a shipped installation preset, deny its installation prefix.

A command approval covers only its displayed command and bound authority. A replaced directory, changed runtime alias or rekeyed workspace requires another command decision. Earlier approvals with weaker identity binding remain visible for removal. To grant or restrict a path for other commands, use the path commands above. Saved path access remains subject to process approval, administrator policy and protected-resource checks.

Policy schema 5 separates local, broad and ingress authority from HTTP egress. Older unscoped HTTP allow defaults and rules require review; scoped HTTP permissions and existing asks/denies are preserved. An old broad-network command approval does not authorize the new broad scope. An actionless allow rule cannot authorize local, broad or ingress networking. Explicitly selecting the unsupervised preset can allow those actions, subject to hard limits; upgrading an existing policy does not implicitly select it.

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
| `supervised` | Allow reads. Ask for sandbox process execution, writes and network access. |
| `workspace` | Allow reads and explicitly requested file creation or replacement. Ask for sandbox process execution, deletion and network access. |
| `locked` | Deny every operation that requires authority. |
| `unsupervised` | Allow operations, subject to enforcement limits. |

## Review and revoke approvals

`run_command` approvals are sandbox-only. List or remove them with `/permissions sandbox list` and `/permissions sandbox remove <session|workspace|user> <id>`. Host execution requires the separate trusted gate and exact host approval. Version-one session, workspace, and user host records remain inactive and are never converted into sandbox grants. Inspect them with `/permissions host list` or `/permissions host trusted`; remove them with `/permissions host remove <id>` or `/permissions host untrust <session|workspace|user> <id>`. Fixed trusted `/sandbox` management has its own authorization and cannot execute arbitrary agent shell source.

Ordinary permission decisions are recorded in the session timeline and application audit database. Denied host operations and removal of legacy host records use minimized audit records without shell source, output or credentials. Canonical tool calls and complete tool traces retain their original command and result content. Export audit events as JSONL with `/app audit <destination>`. Replacing an existing destination requires `force=true`.

## Direct-host authorization

`request_host_command` is visible only while trusted `app.allow_direct_host_execution` permits it and tool-wide host policy allows a request. Dispatch and final launch recheck eligibility and policy. The exact command and cwd, agent-supplied reason, unrestricted authority and possible repeated effects are shown before consent. Choose deny, allow once, or explicitly remember the exact command for the session, workspace or user. YOLO supplies one eligible approval without persisting it. Generic process allows and sandbox approvals never provide host consent.

Host grants bind exact shell source, relative workspace cwd and reviewed environment. Session and workspace grants also bind the workspace and working directory identities. User grants intentionally span workspaces at the same relative cwd. An explicit host `ask` requires one-time approval even when a remembered grant exists. Matching source does not freeze mutable scripts or dependencies. Older grants that lack the required identity remain inspectable and revocable until renewed. Use `/permissions host trusted` and `/permissions host untrust [session|workspace|user] <id>` for durable grants; `/permissions host list` and `/permissions host remove <id>` inspect and revoke session retry records. Disabling the trusted gate leaves every grant dormant.

A separately authorized host process has the user's unrestricted filesystem and network authority, subject to independent operating-system restrictions. Sandbox and direct-file hard denials do not cover that run. No native failure, timeout, or nonzero exit automatically requests consent or replays the command on the host. Inspect possible effects before issuing a separate host request.
