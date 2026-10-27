# Permissions and approvals

Loop checks tool requests against a workspace policy before they run. It validates the tool's arguments, plans the effects it can identify, then applies one decision to the complete plan. If a tool discovers another effect while it runs, Loop checks that effect before allowing it.

The default **supervised** policy allows scoped file reads and asks before most changes, network requests, and session changes. Shell commands run inside a restricted workspace sandbox; routine offline commands can run there without a separate prompt. Requests for additional access need approval. When no interactive user is available, a request that needs approval is denied.

## Inspect a decision

Start with these commands:

```text
/permissions
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

Use `/permissions default reset <scope> <action>` to reset a decision. A workspace reset restores the supervised default; a session reset removes the temporary override so the workspace value applies. Use `/permissions session reset` to clear every session override. To edit the workspace YAML file directly, run `/permissions reload`; Loop validates it and keeps session overrides active.

## Boundaries

Limits restrict what a policy can reach. Rules cannot bypass them. By default, Loop operations can read and write inside the workspace and the active conversation's private working directory (`loop-temp`), addressed as `$TMPDIR/...` by file tools and through the inherited `TMPDIR` environment variable by commands. `$WORKSPACE_ROOT/...` addresses the workspace in file tools, and commands inherit `WORKSPACE_ROOT` for the same directory. Fresh scratch generations allow ordinary data names; existing restricted generations preserve private-name restrictions until explicitly discarded. Workspace controls, host-private data, and other sessions remain outside this exception. Saved-session files persist across restarts; `/tmp` is an OS path and is not an alias for `loop-temp`. Namespace parents and ownership locks remain outside command authority; native metadata traversal does not permit private ancestor listings. Host-process access is off, and private or local network addresses are blocked.

File tools cannot delete the active working-root directory or a containing tree, even with an allow rule or prior approval. They can delete ordinary descendants and remove symbolic links without deleting their targets. Replacing the working root requires the session manager's explicit retirement operation.

Use limits to grant narrowly scoped access:

```text
/permissions limit add workspace read-root system-temp
/permissions limit add workspace network-origin https://my-host.local
/permissions limit set workspace private-network allow
/permissions limit set session host-process allow
/permissions limit list
```

The supported roots include `workspace`, `loop-temp`, and `system-temp`. Add `system-temp` only when Loop needs to read across applications' temporary files. Network origins accept glob patterns. The default `*` allows every origin; adding the first specific origin replaces `*`, so the limit becomes narrower. Private-network access has its own limit. Host-process access must be enabled before policy can approve a host process.

Shell commands have a separate sandbox approval path. Routine commands run within the workspace sandbox. A request to read outside the workspace, use the network, change Git metadata, or expand other sandbox access asks separately. A `process.execute` rule does not grant these permissions or authorize running a command outside the sandbox. See [Built-in tools and command sandbox](tools.md) for command behavior and Linux sandbox management.

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

`run_command` approvals are separate from policy rules. List or remove remembered sandbox approvals with `/permissions sandbox list` and `/permissions sandbox remove <session|workspace|user> <id>`. If Loop offers a host retry after a sandbox failure, that retry has its own approval. List or revoke exact host approvals for the current session with `/permissions host list` and `/permissions host remove <id>`.

Permission decisions are recorded in the session timeline and the application audit database. Export audit events as JSONL with `/app audit <destination>`. Replacing an existing destination requires `force=true`.
