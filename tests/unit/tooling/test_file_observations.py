"""Test file freshness and visibility bound to the context of a model request."""

import json
from unittest.mock import Mock

import pytest

from loop.models import ContextReference, Message, ToolCall, ToolResult
from loop.permissions import Action, FileTarget, Operation
from loop.tooling.file_observations import FileObservations
from loop.utils import VirtualPath, content_digest, sha256_digest


def observed_message(content, *, visible=None, start=0, handle=None, path="file.txt"):
    """Build an immutable reference with explicit delivered coverage."""
    visible = content if visible is None else visible
    return Message(
        role="user",
        content="edit",
        context=(
            ContextReference(
                kind="file",
                path=path,
                content=visible,
                version=content_digest(content),
                handle=handle,
                size_bytes=len(content.encode()),
                included_bytes=len(visible.encode()),
                truncated=visible != content,
                payload_start_bytes=start,
            ),
        ),
    )


def replacement(tmp_path, content, *, tool="edit_text_file", action=Action.FILESYSTEM_REPLACE):
    """Describe the current file state captured by authorized mutation planning."""
    return (
        Operation(
            tool_id=tool,
            action=action,
            target=FileTarget(
                path=str(tmp_path / "file.txt"),
                expected_exists=True,
                expected_digest=sha256_digest(content),
            ),
        ),
    )


def authorized_read(tmp_path):
    """Describe a read phase already authorized for the current invocation."""
    return (
        Operation(
            tool_id="read_text_file",
            action=Action.FILESYSTEM_READ,
            target=FileTarget(path=str(tmp_path / "file.txt")),
        ),
    )


@pytest.mark.parametrize("absence_report", [False, True])
@pytest.mark.parametrize("reissued", [False, True])
def test_context_replay_consumes_results_once_and_allows_reissued_call_ids(
    tmp_path, absence_report, reissued
):
    """A repeated result cannot erase a later write unless a new call establishes its provenance."""
    removal = ToolCall(
        call_id="remove",
        name="write_text_file" if absence_report else "delete_path",
        arguments="{}",
    )
    removed = ToolResult(
        call_id=removal.call_id,
        output=json.dumps(
            {
                "ok": False,
                "problem": {"code": "filesystem.stale_context"},
                "read_versions": {"$WORKSPACE_ROOT/file.txt": None},
            }
            if absence_report
            else {"ok": True, "file_versions": {"$WORKSPACE_ROOT/file.txt": None}}
        ),
    )
    restore = ToolCall(call_id="restore", name="write_text_file", arguments='{"content":"new"}')
    context = [
        observed_message("old"),
        removal,
        removed,
        restore,
        ToolResult(
            call_id=restore.call_id,
            output=json.dumps(
                {"ok": True, "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")}}
            ),
        ),
    ]
    if reissued:
        context.append(removal)
    context.append(removed)
    observations = FileObservations(context, VirtualPath(tmp_path))
    replacement_problem = observations.check(replacement(tmp_path, "new"), {"content": "next"})
    creation = Operation(
        tool_id="write_text_file",
        action=Action.FILESYSTEM_CREATE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
    )
    creation_problem = observations.check((creation,), {"content": "next"})
    if reissued:
        assert replacement_problem.code == "filesystem.stale_context"
        assert creation_problem is None
    else:
        assert replacement_problem is None
        assert creation_problem.code == "filesystem.stale_context"


def test_changes_outside_a_partial_read_invalidate_an_exact_edit(tmp_path):
    """A matching edit anchor cannot conceal divergence in unseen parts of its source."""
    observations = FileObservations(
        [observed_message("anchor\nold\n", visible="anchor\n")], VirtualPath(tmp_path)
    )
    problem = observations.check(replacement(tmp_path, "anchor\nnew\n"), {"old_content": "anchor"})

    assert problem.code == "filesystem.stale_context"
    assert problem.severity == "warning"
    assert problem.metadata == {"reread_required": True}
    assert "Nothing was written" in problem.detail


def test_untrusted_tool_output_cannot_refresh_a_file_observation(tmp_path):
    """Content retrieved by another tool cannot attest to the current on-disk source version."""
    before, current = "anchor\nold\n", "anchor\nnew\n"
    call = ToolCall(call_id="untrusted", name="fetch_document", arguments="{}")
    output = json.dumps(
        {
            "ok": True,
            "result": {
                "path": "$WORKSPACE_ROOT/file.txt",
                "version": content_digest(current),
                "size_bytes": len(current),
                "content": "anchor\n",
                "start_byte": 0,
            },
        }
    )
    observations = FileObservations(
        [
            observed_message(before),
            call,
            ToolResult(call_id=call.call_id, output=output),
        ],
        VirtualPath(tmp_path),
    )
    assert (
        observations.check(replacement(tmp_path, current), {"old_content": "anchor"}).code
        == "filesystem.stale_context"
    )


def test_redacted_read_results_retain_source_identity_without_visible_bytes(tmp_path):
    """An expanded redaction marker cannot supply source bytes or preserve an older revision."""
    call = ToolCall(call_id="read", name="read_text_file", arguments="{}")
    observations = FileObservations(
        [
            observed_message("old"),
            call,
            ToolResult(
                call_id=call.call_id,
                output=json.dumps(
                    {
                        "ok": True,
                        "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")},
                        "result": {
                            "path": "$WORKSPACE_ROOT/file.txt",
                            "version": content_digest("new"),
                            "size_bytes": 3,
                            "start_byte": 0,
                            "content": "<redacted:secret>",
                            "payload_redacted": True,
                        },
                    }
                ),
            ),
        ],
        VirtualPath(tmp_path),
    )
    for content in ("old", "new"):
        assert (
            observations.check(
                replacement(tmp_path, content), {"old_content": "<redacted:secret>"}
            ).code
            == "filesystem.unread_file"
        )


def test_conflict_reporting_cannot_read_without_current_read_authorization(tmp_path, monkeypatch):
    """A captured target digest alone grants no authority to reread bytes for a conflict diff."""
    observations = FileObservations([observed_message("old")], VirtualPath(tmp_path))
    reader = Mock(side_effect=AssertionError("read was not authorized"))
    monkeypatch.setattr("loop.tooling.file_observations.read_bounded_text", reader)

    problem = observations.check(replacement(tmp_path, "new"), {})
    assert problem.code == "filesystem.stale_context"
    assert "diff" not in problem.metadata
    reader.assert_not_called()


def test_stale_context_includes_a_bounded_diff_without_refreshing_the_view(tmp_path):
    """A retained full snapshot explains user changes while still requiring an explicit read."""
    target = tmp_path / "file.txt"
    target.write_text("new\n", encoding="utf-8")
    observations = FileObservations([observed_message("old\n")], VirtualPath(tmp_path))
    operations = replacement(tmp_path, "new\n")

    first = observations.check(operations, {}, authorized_reads=authorized_read(tmp_path))
    assert first.code == "filesystem.stale_context"
    assert "-old" in first.metadata["diff"]
    assert "+new" in first.metadata["diff"]
    assert "$WORKSPACE_ROOT/file.txt" in first.metadata["diff"]
    assert observations.check(operations, {}).code == "filesystem.stale_context"


@pytest.mark.parametrize("current", ["newer", "x\n" * 40_000, "binary\0content"])
def test_conflict_diffs_require_a_complete_matching_current_snapshot(tmp_path, current):
    """A second change, oversized source, or binary content cannot produce a misleading diff."""
    (tmp_path / "file.txt").write_text(current, encoding="utf-8")
    observations = FileObservations([observed_message("old")], VirtualPath(tmp_path))
    problem = observations.check(
        replacement(tmp_path, "new"), {}, authorized_reads=authorized_read(tmp_path)
    )
    assert problem.code == "filesystem.stale_context"
    assert "diff" not in problem.metadata


def test_partial_reads_allow_visible_exact_edits_but_require_full_replacement_coverage(tmp_path):
    """Full source identity does not imply the model received every source byte."""
    text = "first\nsecond\n"
    paths = VirtualPath(tmp_path)
    observations = FileObservations([observed_message(text, visible="first\n")], paths)
    operations = replacement(tmp_path, text)

    assert observations.check(operations, {"old_content": "first"}) is None
    assert (
        observations.check(operations, {"old_content": "second"}).code
        == "filesystem.unread_content"
    )
    assert (
        observations.check(operations, {"replace_all": True}).code == "filesystem.incomplete_read"
    )
    assert (
        observations.check(
            replacement(tmp_path, text, tool="write_text_file"), {"content": "new"}
        ).code
        == "filesystem.incomplete_read"
    )

    complete = FileObservations(
        [
            observed_message(text, visible="first\n"),
            observed_message(text, visible="second\n", start=6),
            observed_message(text, visible="first\n"),
        ],
        paths,
    )
    assert complete.check(operations, {"replace_all": True}) is None


def test_new_revisions_do_not_inherit_old_range_coverage(tmp_path):
    """A partial reread of a new revision cannot make stale omitted content visible."""
    observations = FileObservations(
        [
            observed_message("first\nold\n"),
            observed_message("first\nnew\n", visible="new\n", start=6),
        ],
        VirtualPath(tmp_path),
    )
    assert (
        observations.check(replacement(tmp_path, "first\nnew\n"), {"old_content": "first"}).code
        == "filesystem.unread_content"
    )


def test_empty_and_overlapping_snapshot_ranges_retain_precise_coverage(tmp_path):
    """An empty file and overlapping pages can both constitute complete observations."""
    paths = VirtualPath(tmp_path)
    assert (
        FileObservations([observed_message("")], paths).check(
            replacement(tmp_path, "", tool="write_text_file"), {"content": "created"}
        )
        is None
    )
    observations = FileObservations(
        [
            observed_message("abcdef", visible="abcd"),
            observed_message("abcdef", visible="cdef", start=2),
        ],
        paths,
    )
    assert observations.check(replacement(tmp_path, "abcdef"), {"old_content": "bcde"}) is None


def test_cached_continuations_keep_the_original_immutable_reference_version(tmp_path):
    """Reading a retained artifact observes its saved version, never the current disk file."""
    call = ToolCall(call_id="cached", name="read_cached_content", arguments='{"handle":"snapshot"}')
    result = ToolResult(
        call_id=call.call_id,
        output=json.dumps(
            {
                "ok": True,
                "read_versions": {"snapshot": None},
                "result": {
                    "handle": "snapshot",
                    "content": "old",
                    "start_byte": 0,
                },
            }
        ),
    )
    observations = FileObservations(
        [
            observed_message("old", visible="", handle="snapshot"),
            call,
            result,
        ],
        VirtualPath(tmp_path),
    )
    assert observations.check(replacement(tmp_path, "old"), {"old_content": "old"}) is None
    assert observations.check(replacement(tmp_path, "new"), {"old_content": "old"}).code == (
        "filesystem.stale_context"
    )


def test_deduplicated_references_reuse_only_previously_delivered_snapshot_bytes(tmp_path):
    """A reused immutable artifact can bind another path without inventing unseen content."""
    original = observed_message("same", path="original.txt")
    reused = (
        observed_message("same")
        .context[0]
        .model_copy(
            update={
                "content": "",
                "payload_start_bytes": 4,
                "reused": True,
            }
        )
    )
    message = Message(role="user", content="same attachment", context=(reused,))
    paths = VirtualPath(tmp_path)
    assert (
        FileObservations(
            [observed_message("different", path="other.txt"), original, message], paths
        ).check(replacement(tmp_path, "same"), {"old_content": "same"})
        is None
    )
    assert (
        FileObservations([message], paths).check(replacement(tmp_path, "same"), {}).code
        == "filesystem.unread_file"
    )


@pytest.mark.parametrize(
    "visible,other_visible,other_start,complete",
    [("", "same", 0, True), ("sa", "same", 0, True), ("sa", "me", 2, True), ("", "me", 2, False)],
)
@pytest.mark.parametrize("recovering", [False, True])
def test_reused_references_combine_all_previously_visible_views(
    tmp_path, visible, other_visible, other_start, complete, recovering
):
    """Reused observations unite delivered ranges across paths without inventing omitted bytes."""
    reference = (
        observed_message("same")
        .context[0]
        .model_copy(update={"content": "", "payload_start_bytes": 4, "reused": True})
    )
    context = [
        observed_message("same", visible=visible),
        observed_message("same", visible=other_visible, start=other_start, path="other.txt"),
        Message(role="user", content="edit", context=(reference,)),
    ]
    paths = VirtualPath(tmp_path)
    if recovering:
        pending = ToolCall(call_id="pending", name="write_text_file", arguments="{}")
        observations = FileObservations.for_recovery([*context, pending], paths, [pending])
    else:
        observations = FileObservations(context, paths)
    problem = observations.check(replacement(tmp_path, "same"), {"content": "new"})
    if complete:
        assert problem is None
    else:
        assert problem.code == "filesystem.incomplete_read"


@pytest.mark.parametrize(
    "output",
    [
        "invalid",
        "[]",
        '{"ok":false}',
        '{"ok":true,"result":"legacy"}',
        '{"ok":true,"result":{}}',
        '{"ok":true,"result":{"path":"/external","version":"old","size_bytes":1,"start_byte":0,"content":"a"}}',
        '{"ok":true,"result":{"path":"file.txt","version":"old","size_bytes":1,"start_byte":0,"content":null}}',
        '{"ok":true,"result":{"path":"file.txt","version":null,"size_bytes":1,"start_byte":0,"content":"a"}}',
        '{"ok":true,"result":{"path":"file.txt","version":"old","size_bytes":null,"start_byte":0,"content":"a"}}',
        '{"ok":true,"result":{"path":"file.txt","version":"old","size_bytes":1,"start_byte":null,"content":"a"}}',
        '{"ok":true,"result":{"path":"file.txt","version":"old","size_bytes":1,"start_byte":-1,"content":"a"}}',
        '{"ok":true,"result":{"path":"file.txt","version":"old","size_bytes":0,"start_byte":0,"content":"a"}}',
    ],
)
def test_incomplete_or_legacy_results_cannot_authorize_an_edit(tmp_path, output):
    """Unavailable provenance requires another read instead of trusting guessed file state."""
    if output.startswith('{"ok":true,"result":{"path"'):
        payload = json.loads(output)
        source = payload["result"]
        payload["read_versions"] = {source["path"]: source["version"]}
        output = json.dumps(payload)
    call = ToolCall(call_id="read", name="read_text_file", arguments="{}")
    observations = FileObservations(
        [
            ToolResult(call_id="unmatched", output="{}"),
            call,
            ToolResult(call_id=call.call_id, output=output),
        ],
        VirtualPath(tmp_path),
    )
    assert observations.check(replacement(tmp_path, "a"), {}).code == "filesystem.unread_file"


def test_successful_model_edits_advance_only_the_content_the_model_knows(tmp_path):
    """Own writes update the expected revision without blessing unseen file ranges."""
    observations = FileObservations(
        [
            observed_message("old\nhidden\n", visible="old\n"),
            observed_message("old\nhidden\n", visible="hidden\n", start=4),
        ],
        VirtualPath(tmp_path),
    )
    edit = ToolCall(
        call_id="edit",
        name="edit_text_file",
        arguments=json.dumps(
            {
                "old_content": "old",
                "new_content": "longer",
            }
        ),
    )
    output = json.dumps(
        {
            "ok": True,
            "result": "edited",
            "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("old\nhidden\n")},
            "file_versions": {
                "$WORKSPACE_ROOT/file.txt": content_digest("longer\nhidden\n"),
            },
        }
    )
    observations.observe_result(edit, output, include_reads=False)
    assert (
        observations.check(replacement(tmp_path, "longer\nhidden\n"), {"old_content": "hidden"})
        is None
    )

    write = ToolCall(call_id="write", name="write_text_file", arguments='{"content":"entire"}')
    observations.observe_result(
        write,
        json.dumps(
            {
                "ok": True,
                "result": "written",
                "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("entire")},
            }
        ),
    )
    assert observations.check(replacement(tmp_path, "entire"), {"old_content": "entire"}) is None


def test_undelivered_reads_do_not_refresh_the_generating_model_view(tmp_path):
    """A same-batch reread cannot validate an edit generated before that read returned."""
    observations = FileObservations([observed_message("old")], VirtualPath(tmp_path))
    read = ToolCall(call_id="reread", name="read_text_file", arguments="{}")
    output = json.dumps(
        {
            "ok": True,
            "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")},
            "result": {
                "path": "$WORKSPACE_ROOT/file.txt",
                "version": content_digest("new"),
                "size_bytes": 3,
                "content": "new",
                "start_byte": 0,
            },
        }
    )
    observations.observe_result(read, output, include_reads=False)
    assert observations.check(replacement(tmp_path, "new"), {}).code == "filesystem.stale_context"
    observations.observe_result(read, output)
    assert observations.check(replacement(tmp_path, "new"), {}) is None


def test_observations_are_scoped_to_each_sessions_delivered_context(tmp_path):
    """A newer read in another session cannot refresh or authorize this session's edits."""
    paths = VirtualPath(tmp_path)
    first = FileObservations([observed_message("old")], paths)
    second = FileObservations([observed_message("new")], paths)
    assert first.check(replacement(tmp_path, "new"), {}).code == "filesystem.stale_context"
    assert second.check(replacement(tmp_path, "new"), {}) is None
    assert (
        FileObservations([], paths).check(replacement(tmp_path, "new"), {}).code
        == "filesystem.unread_file"
    )


def test_missing_or_redacted_reference_bytes_do_not_establish_visibility(tmp_path):
    """Opaque references and transport-redacted text do not constitute a source read."""
    message = observed_message("source")
    reference = message.context[0]
    message = message.model_copy(
        update={
            "context": (
                reference.model_copy(update={"kind": "directory"}),
                reference.model_copy(update={"version": None}),
                reference.model_copy(update={"content": b"source"}),
                reference.model_copy(update={"payload_redacted": True}),
            )
        }
    )
    assert (
        FileObservations([message], VirtualPath(tmp_path))
        .check(replacement(tmp_path, "source"), {})
        .code
        == "filesystem.unread_file"
    )


def test_unresolvable_references_and_unknown_cached_handles_require_a_read(tmp_path):
    """External provenance and unrelated cached content cannot authorize workspace edits."""
    call = ToolCall(call_id="cached", name="read_cached_content", arguments="{}")
    observations = FileObservations(
        [
            observed_message("source", path="/unavailable/file.txt"),
            call,
            ToolResult(
                call_id="cached",
                output=json.dumps(
                    {
                        "ok": True,
                        "read_versions": {"unknown": None},
                        "result": {
                            "handle": "unknown",
                            "content": "source",
                            "start_byte": 0,
                        },
                    }
                ),
            ),
        ],
        VirtualPath(tmp_path),
    )
    assert observations.check(replacement(tmp_path, "source"), {}).code == "filesystem.unread_file"


def test_an_edit_receipt_without_its_source_cannot_establish_visible_content(tmp_path):
    """A compacted or legacy edit receipt alone cannot reconstruct omitted source bytes."""
    call = ToolCall(
        call_id="edited", name="edit_text_file", arguments='{"old_content":"a","new_content":"b"}'
    )
    observations = FileObservations(
        [
            call,
            ToolResult(
                call_id=call.call_id,
                output=json.dumps(
                    {
                        "ok": True,
                        "result": "edited",
                        "file_versions": {
                            "$WORKSPACE_ROOT/file.txt": content_digest("b"),
                        },
                    }
                ),
            ),
        ],
        VirtualPath(tmp_path),
    )
    assert observations.check(replacement(tmp_path, "b"), {}).code == "filesystem.unread_file"


@pytest.mark.parametrize(
    "versions,arguments",
    [
        (None, {}),
        ({"$WORKSPACE_ROOT/file.txt": content_digest("b")}, {"content": 123}),
    ],
)
def test_incompatible_mutation_receipts_require_a_fresh_read(tmp_path, versions, arguments):
    """Malformed imported result metadata does not crash context preparation or grant visibility."""
    call = ToolCall(call_id="write", name="write_text_file", arguments=json.dumps(arguments))
    observations = FileObservations(
        [
            call,
            ToolResult(
                call_id="write",
                output=json.dumps(
                    {
                        "ok": True,
                        "result": "written",
                        "file_versions": versions,
                    }
                ),
            ),
        ],
        VirtualPath(tmp_path),
    )
    assert observations.check(replacement(tmp_path, "b"), {}).code == "filesystem.unread_file"


@pytest.mark.parametrize("code", ["filesystem.path_missing", "filesystem.stale_context"])
def test_missing_reads_refresh_existence_only_when_delivered(tmp_path, code):
    """A trusted absence report permits creation only in a later generation."""
    paths = VirtualPath(tmp_path)
    observations = FileObservations([observed_message("old")], paths)
    creation = (
        Operation(
            tool_id="write_text_file",
            action=Action.FILESYSTEM_CREATE,
            target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
        ),
    )
    call = ToolCall(call_id="missing", name="read_text_file", arguments="{}")
    output = json.dumps(
        {
            "ok": False,
            "problem": {"code": code},
            "read_versions": {
                "$WORKSPACE_ROOT/file.txt": None,
                "$WORKSPACE_ROOT/other.txt": "ignored",
            },
        }
    )
    observations.observe_result(call, output, include_reads=False)
    assert observations.check(creation, {"content": "new"}).code == "filesystem.stale_context"
    observations.observe_result(call, output)
    assert observations.check(creation, {"content": "new"}) is None
    assert observations.check(replacement(tmp_path, "user recreated"), {}).code == (
        "filesystem.stale_context"
    )
    observations.observe_result(
        call,
        json.dumps(
            {
                "ok": True,
                "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")},
                "result": {
                    "path": "$WORKSPACE_ROOT/file.txt",
                    "version": content_digest("new"),
                    "size_bytes": 3,
                    "start_byte": 0,
                    "content": "new",
                },
            }
        ),
    )
    assert observations.check(replacement(tmp_path, "new"), {}) is None


@pytest.mark.parametrize("prior_generation", [False, True])
def test_deletion_receipts_preserve_generation_recovery_and_session_boundaries(
    tmp_path, prior_generation
):
    """Only delivered absence permits recreation, including after interrupted-batch recovery."""
    paths = VirtualPath(tmp_path)
    original = observed_message("old")
    observations = FileObservations([original], paths)
    creation = Operation(
        tool_id="write_text_file",
        action=Action.FILESYSTEM_CREATE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
    )
    receipts = {}
    problem = observations.check((creation,), {"content": "stale"}, absence_receipts=receipts)
    assert receipts == {"$WORKSPACE_ROOT/file.txt": None}
    assert problem.metadata == {"reread_required": False, "regeneration_required": True}
    assert "deleted" in problem.detail
    assert observations.check((creation,), {"content": "same batch"}).code == (
        "filesystem.stale_context"
    )
    rejected = ToolCall(call_id="rejected", name="write_text_file", arguments="{}")
    result = ToolResult(
        call_id=rejected.call_id,
        output=json.dumps(
            {"ok": False, "problem": problem.model_dump(), "read_versions": receipts}
        ),
    )
    pending = ToolCall(call_id="pending", name="write_text_file", arguments="{}")
    context = (
        [original, rejected, result, pending]
        if prior_generation
        else [original, rejected, pending, result]
    )
    recovered = FileObservations.for_recovery(context, paths, [pending])
    verdict = recovered.check((creation,), {"content": "reconsidered"})
    if prior_generation:
        assert verdict is None
    else:
        assert verdict.code == "filesystem.stale_context"
    assert FileObservations([original], paths).check((creation,), {}).code == (
        "filesystem.stale_context"
    )
    incomplete = FileObservations.for_recovery([original, rejected, result], paths, [pending])
    incomplete_receipts = {}
    assert (
        incomplete.check((creation,), {}, absence_receipts=incomplete_receipts).metadata[
            "regeneration_required"
        ]
        is True
    )
    assert incomplete_receipts == {}
    assert recovered.check(replacement(tmp_path, "external recreation"), {}).code == (
        "filesystem.stale_context"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "problem": {
                "code": "filesystem.stale_context",
                "metadata": {"read_versions": {"file.txt": None}},
            }
        },
        {"problem": {"code": "filesystem.stale_context"}, "read_versions": []},
        {"problem": {"code": "filesystem.stale_context"}, "read_versions": {"file.txt": "invalid"}},
        {"problem": {"code": "filesystem.read_failed"}, "read_versions": {"file.txt": None}},
        {
            "problem": {"code": "filesystem.stale_context"},
            "read_versions": {"/outside/file.txt": None},
        },
    ],
)
def test_untrusted_or_malformed_absence_cannot_authorize_recreation(tmp_path, payload):
    """Only valid outer absence receipts on recognized failures can replace observed existence."""
    call = ToolCall(call_id="failed", name="write_text_file", arguments="{}")
    observations = FileObservations(
        [
            observed_message("old"),
            call,
            ToolResult(call_id=call.call_id, output=json.dumps(payload)),
        ],
        VirtualPath(tmp_path),
    )
    creation = Operation(
        tool_id=call.name,
        action=Action.FILESYSTEM_CREATE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
    )
    assert observations.check((creation,), {}).code == "filesystem.stale_context"


@pytest.mark.parametrize("code", ["filesystem.path_missing", "filesystem.stale_context"])
@pytest.mark.parametrize("status", ["missing", None, 0, "false", [], {}])
def test_absence_receipts_require_an_explicit_failed_envelope(tmp_path, code, status):
    """Malformed result status cannot replace a source snapshot with an absence observation."""
    call = ToolCall(call_id="failed", name="write_text_file", arguments="{}")
    payload = {
        "problem": {"code": code},
        "read_versions": {"$WORKSPACE_ROOT/file.txt": None},
    }
    if status != "missing":
        payload["ok"] = status
    observations = FileObservations(
        [
            observed_message("old"),
            call,
            ToolResult(call_id=call.call_id, output=json.dumps(payload)),
        ],
        VirtualPath(tmp_path),
    )
    creation = Operation(
        tool_id=call.name,
        action=Action.FILESYSTEM_CREATE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
    )
    assert observations.check((creation,), {}).code == "filesystem.stale_context"
    assert observations.check(replacement(tmp_path, "old"), {"content": "new"}) is None


def test_successful_removals_allow_creation_and_detect_external_recreation(tmp_path):
    """A committed model deletion advances existence without trusting later external writes."""
    observations = FileObservations([observed_message("old")], VirtualPath(tmp_path))
    delete = ToolCall(call_id="delete", name="delete_path", arguments="{}")
    observations.observe_result(
        delete,
        json.dumps({"ok": True, "file_versions": {"$WORKSPACE_ROOT/file.txt": None}}),
        include_reads=False,
    )
    creation = (
        Operation(
            tool_id="write_text_file",
            action=Action.FILESYSTEM_CREATE,
            target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
        ),
    )
    assert observations.check(creation, {"content": "new"}) is None
    assert observations.check(replacement(tmp_path, "external"), {}).code == (
        "filesystem.stale_context"
    )
    write = ToolCall(call_id="write", name="write_text_file", arguments='{"content":"new"}')
    observations.observe_result(
        write,
        json.dumps(
            {"ok": True, "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")}}
        ),
        include_reads=False,
    )
    assert observations.check(replacement(tmp_path, "new"), {}) is None


def test_recovery_uses_prior_generation_and_replays_completed_own_writes(tmp_path):
    """Recovery retains prior reads and successful writes while excluding interrupted rereads."""
    paths = VirtualPath(tmp_path)
    prior = ToolCall(call_id="prior", name="passive", arguments="{}")
    write = ToolCall(call_id="write", name="write_text_file", arguments='{"content":"new"}')
    pending = ToolCall(call_id="pending", name="edit_text_file", arguments="{}")
    context = [
        observed_message("old"),
        prior,
        ToolResult(call_id="prior", output='{"ok":true,"result":"done"}'),
        Message(role="assistant", content="Changing the file"),
        write,
        pending,
        ToolResult(
            call_id="write",
            output=json.dumps(
                {"ok": True, "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")}}
            ),
        ),
    ]
    observations = FileObservations.for_recovery(context, paths, [pending])
    assert observations.check(replacement(tmp_path, "new"), {}) is None
    assert (
        FileObservations.for_recovery(context, paths, []).check(replacement(tmp_path, "new"), {})
        is None
    )
    assert (
        FileObservations.for_recovery(context[:1], paths, [pending])
        .check(replacement(tmp_path, "old"), {})
        .code
        == "filesystem.stale_context"
    )
    assert (
        FileObservations.for_recovery([pending], paths, [pending])
        .check(replacement(tmp_path, "old"), {})
        .code
        == "filesystem.unread_file"
    )


@pytest.mark.parametrize("completed_write", [False, True])
def test_recovery_prepares_only_the_preceding_generation_context(tmp_path, completed_write):
    """Backend preparation filters delivered history without redacting the interrupted own write."""
    original = observed_message("secret")
    prior = ToolCall(call_id="prior", name="passive", arguments="{}")
    prefix = [original, prior, ToolResult(call_id=prior.call_id, output='{"ok":true}')]
    pending = ToolCall(call_id="pending", name="write_text_file", arguments='{"content":"secret"}')
    write = ToolCall(call_id="write", name="write_text_file", arguments='{"content":"new secret"}')
    context = [*prefix, *((write,) if completed_write else ()), pending]
    if completed_write:
        context.append(
            ToolResult(
                call_id=write.call_id,
                output=json.dumps(
                    {
                        "ok": True,
                        "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new secret")},
                    }
                ),
            )
        )
    redacted = original.model_copy(
        update={
            "context": (
                original.context[0].model_copy(
                    update={"content": "<redacted:secret>", "payload_redacted": True}
                ),
            )
        }
    )
    prepare = Mock(return_value=[redacted])
    observations = FileObservations.for_recovery(
        context, VirtualPath(tmp_path), [pending], prepare_context=prepare
    )
    prepare.assert_called_once_with(prefix)
    verdict = observations.check(
        replacement(tmp_path, "new secret" if completed_write else "secret"), {"content": "next"}
    )
    if completed_write:
        assert verdict is None
    else:
        assert verdict.code == "filesystem.unread_file"
    prepare.reset_mock()
    observations = FileObservations.for_recovery(
        prefix, VirtualPath(tmp_path), [], prepare_context=prepare
    )
    prepare.assert_called_once_with(prefix)
    assert observations.check(replacement(tmp_path, "secret"), {}).code == "filesystem.unread_file"


@pytest.mark.parametrize("receipts", [None, [], {"unknown": "wrong"}])
def test_invalid_read_receipts_do_not_authorize_cached_content(tmp_path, receipts):
    """Malformed or mismatched read attestations cannot establish cached file provenance."""
    call = ToolCall(call_id="cached", name="read_cached_content", arguments="{}")
    output = json.dumps(
        {
            "ok": True,
            "read_versions": receipts,
            "result": {"handle": "unknown", "content": "a", "start_byte": 0},
        }
    )
    observations = FileObservations(
        [call, ToolResult(call_id="cached", output=output)], VirtualPath(tmp_path)
    )
    assert observations.check(replacement(tmp_path, "a"), {}).code == "filesystem.unread_file"


def test_targets_without_content_preconditions_keep_existing_authorization(tmp_path):
    """Non-content operations remain governed by their declared authorization capabilities."""
    operation = Operation(
        tool_id="delete_path",
        action=Action.FILESYSTEM_DELETE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=True),
    )
    observations = FileObservations([observed_message("old")], VirtualPath(tmp_path))
    assert observations.check((operation,), {}) is None


def test_recovery_uses_the_latest_batch_when_a_backend_reuses_call_ids(tmp_path):
    """A repeated provider identifier cannot recover an older generation's observations."""
    pending = ToolCall(call_id="reused", name="edit_text_file", arguments="{}")
    context = [
        observed_message("old"),
        pending,
        ToolResult(call_id="reused", output='{"ok":true,"result":"done"}'),
        observed_message("new"),
        pending,
    ]
    observations = FileObservations.for_recovery(context, VirtualPath(tmp_path), [pending])
    assert observations.check(replacement(tmp_path, "new"), {}) is None
    assert observations.check(replacement(tmp_path, "old"), {}).code == "filesystem.stale_context"


def test_cached_rereads_cannot_supply_the_source_of_a_same_batch_edit_receipt(tmp_path):
    """An edit receipt must not relabel bytes from an older cached revision as its own result."""
    paths = VirtualPath(tmp_path)
    cached = ToolCall(call_id="cached", name="read_cached_content", arguments="{}")
    edit = ToolCall(
        call_id="edit",
        name="edit_text_file",
        arguments='{"old_content":"anchor","new_content":"model"}',
    )
    result = json.dumps(
        {
            "ok": True,
            "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("anchor newest")},
            "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("model newest")},
            "result": "edited",
        }
    )
    observations = FileObservations(
        [
            observed_message("anchor old", visible="anchor", handle="snapshot"),
            observed_message("anchor newest"),
            cached,
            edit,
            ToolResult(
                call_id="cached",
                output=json.dumps(
                    {
                        "ok": True,
                        "read_versions": {"snapshot": None},
                        "result": {"handle": "snapshot", "content": "anchor old", "start_byte": 0},
                    }
                ),
            ),
            ToolResult(call_id="edit", output=result),
        ],
        paths,
    )
    assert observations.check(
        replacement(tmp_path, "model newest"), {"content": "replace"}
    ).code == ("filesystem.unread_file")
    assert observations.check(
        replacement(tmp_path, "model newest"), {"old_content": "old"}
    ).code == ("filesystem.unread_file")


@pytest.mark.parametrize(
    "field,value", [("name", "another_tool"), ("arguments", '{"different":true}')]
)
def test_recovery_rejects_a_pending_request_that_does_not_match_history(tmp_path, field, value):
    """A matching identifier cannot substitute for the pending request's actual identity."""
    pending = ToolCall(call_id="pending", name="write_text_file", arguments="{}")
    recorded = pending.model_copy(update={field: value})
    observations = FileObservations.for_recovery(
        [observed_message("old"), recorded], VirtualPath(tmp_path), [pending]
    )
    creation = Operation(
        tool_id="write_text_file",
        action=Action.FILESYSTEM_CREATE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
    )
    problem = observations.check((creation,), {"content": "stale"})
    assert problem.code == "filesystem.stale_context"
    assert problem.metadata["regeneration_required"] is True


@pytest.mark.parametrize(
    "boundary",
    [ToolResult(call_id="first", output="{}"), Message(role="user", content="another turn")],
)
def test_recovery_refuses_pending_calls_from_different_generations(tmp_path, boundary):
    """A recovery batch must belong to a single uninterrupted model response."""
    first = ToolCall(call_id="first", name="edit_text_file", arguments="{}")
    last = ToolCall(call_id="last", name="edit_text_file", arguments="{}")
    observations = FileObservations.for_recovery(
        [observed_message("old"), first, boundary, last], VirtualPath(tmp_path), [first, last]
    )
    assert (
        observations.check(replacement(tmp_path, "old"), {}).metadata["regeneration_required"]
        is True
    )


@pytest.mark.parametrize(
    "original,visible,arguments,result",
    [
        ("a", "a", {"content": "guessed"}, "actual"),
        ("a", "a", {"old_content": "", "new_content": "b"}, "b"),
        ("a hidden", "a", {"old_content": "hidden", "new_content": "b"}, "a b"),
        (
            "a hidden",
            "a",
            {"old_content": "a", "new_content": "b", "replace_all": True},
            "b hidden",
        ),
        ("a", "a", {"old_content": "a", "new_content": "b"}, "unexpected"),
    ],
)
def test_inconsistent_mutation_receipts_discard_visible_coverage(
    tmp_path, original, visible, arguments, result
):
    """Unsupported or inconsistent reconstruction never makes guessed bytes visible."""
    observations = FileObservations(
        [observed_message(original, visible=visible)], VirtualPath(tmp_path)
    )
    call = ToolCall(call_id="write", name="mutate", arguments=json.dumps(arguments))
    observations.observe_result(
        call,
        json.dumps(
            {
                "ok": True,
                "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(original)},
                "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(result)},
            }
        ),
    )
    assert (
        observations.check(replacement(tmp_path, result), {"content": "replace"}).code
        == "filesystem.unread_file"
    )
    reread = ToolCall(call_id="read", name="read_text_file", arguments="{}")
    observations.observe_result(
        reread,
        json.dumps(
            {
                "ok": True,
                "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(result)},
                "result": {
                    "path": "$WORKSPACE_ROOT/file.txt",
                    "version": content_digest(result),
                    "size_bytes": len(result),
                    "start_byte": 0,
                    "content": result,
                },
            }
        ),
    )
    assert observations.check(replacement(tmp_path, result), {"content": "replace"}) is None


@pytest.mark.parametrize("recovering", [False, True])
def test_redacted_edit_arguments_cannot_invent_visible_replacement_bytes(tmp_path, recovering):
    """Prepared edit arguments cannot turn a hidden literal marker into an observed edit anchor."""
    original = "anchor\nhidden <redacted:secret>\n"
    updated = "abcdefghijklmnop\nhidden <redacted:secret>\n"
    edit = ToolCall(
        call_id="edit",
        name="edit_text_file",
        arguments='{"old_content":"anchor","new_content":"<redacted:secret>"}',
        arguments_redacted=True,
    )
    context = [
        observed_message(original, visible="anchor\n"),
        edit,
        ToolResult(
            call_id=edit.call_id,
            output=json.dumps(
                {
                    "ok": True,
                    "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(original)},
                    "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(updated)},
                }
            ),
        ),
    ]
    paths = VirtualPath(tmp_path)
    if recovering:
        pending = ToolCall(call_id="pending", name="edit_text_file", arguments="{}")
        observations = FileObservations.for_recovery([*context, pending], paths, [pending])
    else:
        observations = FileObservations(context, paths)
    problem = observations.check(
        replacement(tmp_path, updated), {"old_content": "<redacted:secret>"}
    )
    assert problem is not None
    assert problem.code == "filesystem.unread_file"


def test_partial_mutation_receipts_preserve_utf8_ranges_and_offsets(tmp_path):
    """A verified exact edit advances known ranges while leaving omitted bytes unseen."""
    original = "é hidden tail"
    updated = "longer hidden tail"
    observations = FileObservations(
        [
            observed_message(original, visible="é"),
            observed_message(original, visible="tail", start=len("é hidden ".encode())),
        ],
        VirtualPath(tmp_path),
    )
    edit = ToolCall(
        call_id="edit",
        name="edit_text_file",
        arguments='{"old_content":"é","new_content":"longer"}',
    )
    observations.observe_result(
        edit,
        json.dumps(
            {
                "ok": True,
                "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(original)},
                "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(updated)},
            }
        ),
        include_reads=False,
    )
    assert observations.check(replacement(tmp_path, updated), {"old_content": "tail"}) is None
    assert (
        observations.check(replacement(tmp_path, updated), {"content": "overwrite"}).code
        == "filesystem.incomplete_read"
    )


def test_recovery_does_not_replay_mutations_from_a_later_generation(tmp_path):
    """An older pending request cannot adopt committed versions from another response."""
    pending = ToolCall(call_id="pending", name="write_text_file", arguments='{"content":"stale"}')
    newer = ToolCall(call_id="newer", name="write_text_file", arguments='{"content":"new"}')
    observations = FileObservations.for_recovery(
        [
            observed_message("old"),
            pending,
            ToolResult(call_id="unrelated", output='{"ok":true}'),
            newer,
            ToolResult(
                call_id="newer",
                output=json.dumps(
                    {
                        "ok": True,
                        "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")},
                    }
                ),
            ),
        ],
        VirtualPath(tmp_path),
        [pending],
    )
    problem = observations.check(replacement(tmp_path, "new"), {"content": "stale"})
    assert problem.code == "filesystem.stale_context"
    assert problem.metadata["regeneration_required"] is True


@pytest.mark.parametrize("same_arguments", [False, True])
def test_recovery_refuses_ambiguous_pending_call_identifiers(tmp_path, same_arguments):
    """Duplicate unresolved identifiers cannot silently collapse into a single request."""
    first = ToolCall(call_id="same", name="write_text_file", arguments='{"content":"old"}')
    second = (
        first if same_arguments else first.model_copy(update={"arguments": '{"content":"new"}'})
    )
    observations = FileObservations.for_recovery(
        [observed_message("old"), first, second], VirtualPath(tmp_path), [first, second]
    )
    problem = observations.check(replacement(tmp_path, "old"), {})
    assert problem.code == "filesystem.stale_context"
    assert problem.metadata["regeneration_required"] is True


@pytest.mark.parametrize(
    "pages",
    [
        [(0, "wrong!")],
        [(0, "abc"), (3, "xyz")],
        [(0, "abcd"), (2, "XX")],
    ],
)
def test_read_ranges_must_agree_with_each_other_and_the_complete_digest(tmp_path, pages):
    """Conflicting ranges cannot establish visibility merely by claiming the same version."""
    source = "abcdef"
    context = []
    for index, (offset, content) in enumerate(pages):
        call = ToolCall(call_id=str(index), name="read_text_file", arguments="{}")
        context.extend(
            [
                call,
                ToolResult(
                    call_id=call.call_id,
                    output=json.dumps(
                        {
                            "ok": True,
                            "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(source)},
                            "result": {
                                "path": "$WORKSPACE_ROOT/file.txt",
                                "version": content_digest(source),
                                "content": content,
                                "start_byte": offset,
                                "size_bytes": len(source),
                            },
                        }
                    ),
                ),
            ]
        )
    observations = FileObservations(context, VirtualPath(tmp_path))
    problem = observations.check(replacement(tmp_path, source), {"old_content": pages[0][1]})
    assert problem.code == "filesystem.unread_file"


def test_recovery_refuses_duplicate_recorded_calls_before_the_pending_request(tmp_path):
    """Ambiguous completed siblings cannot be replayed into a pending mutation's view."""
    duplicate = ToolCall(call_id="same", name="write_text_file", arguments='{"content":"old"}')
    pending = ToolCall(call_id="pending", name="write_text_file", arguments='{"content":"new"}')
    observations = FileObservations.for_recovery(
        [observed_message("old"), duplicate, duplicate, pending], VirtualPath(tmp_path), [pending]
    )
    problem = observations.check(replacement(tmp_path, "old"), {})
    assert problem.code == "filesystem.stale_context"
    assert problem.metadata["regeneration_required"] is True


@pytest.mark.parametrize("result_id", ["remove", "unmatched"])
def test_recovery_refuses_duplicate_or_unmatched_results(tmp_path, result_id):
    """Ambiguous results cannot undo a completed recreation or establish a recovery baseline."""
    remove = ToolCall(call_id="remove", name="delete_path", arguments="{}")
    restore = ToolCall(call_id="restore", name="write_text_file", arguments='{"content":"new"}')
    pending = ToolCall(call_id="pending", name="write_text_file", arguments='{"content":"next"}')
    removal = json.dumps({"ok": True, "file_versions": {"$WORKSPACE_ROOT/file.txt": None}})
    observations = FileObservations.for_recovery(
        [
            observed_message("old"),
            remove,
            restore,
            pending,
            ToolResult(call_id=remove.call_id, output=removal),
            ToolResult(
                call_id=restore.call_id,
                output=json.dumps(
                    {
                        "ok": True,
                        "file_versions": {"$WORKSPACE_ROOT/file.txt": content_digest("new")},
                    }
                ),
            ),
            ToolResult(call_id=result_id, output=removal),
        ],
        VirtualPath(tmp_path),
        [pending],
    )
    problem = observations.check(replacement(tmp_path, "new"), {"content": "next"})
    assert problem is not None
    assert problem.metadata.get("regeneration_required") is True
    creation = Operation(
        tool_id=pending.name,
        action=Action.FILESYSTEM_CREATE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
    )
    receipts = {}
    problem = observations.check((creation,), {"content": "next"}, absence_receipts=receipts)
    assert problem is not None
    assert problem.metadata.get("regeneration_required") is True
    assert receipts == {}


def test_a_verified_reread_restores_coverage_after_inconsistent_snapshot_bytes(tmp_path):
    """Rejected bytes do not prevent a later complete verified snapshot from authorizing edits."""
    source = "abcdef"
    observations = FileObservations(
        [observed_message(source, visible="wrong!")], VirtualPath(tmp_path)
    )
    assert observations.check(replacement(tmp_path, source), {}).code == "filesystem.unread_file"
    call = ToolCall(call_id="read", name="read_text_file", arguments="{}")
    observations.observe_result(
        call,
        json.dumps(
            {
                "ok": True,
                "read_versions": {"$WORKSPACE_ROOT/file.txt": content_digest(source)},
                "result": {
                    "path": "$WORKSPACE_ROOT/file.txt",
                    "version": content_digest(source),
                    "content": source,
                    "start_byte": 0,
                    "size_bytes": len(source),
                },
            }
        ),
    )
    assert observations.check(replacement(tmp_path, source), {"content": "replacement"}) is None


@pytest.mark.parametrize("opaque", [b"new", "<redacted:secret>"])
@pytest.mark.parametrize("recovering", [False, True])
def test_opaque_references_replace_the_identity_of_an_older_visible_revision(
    tmp_path, opaque, recovering
):
    """Opaque payloads retain their revision without borrowing visible bytes from an older one."""
    latest = (
        observed_message("new")
        .context[0]
        .model_copy(update={"content": opaque, "payload_redacted": isinstance(opaque, str)})
    )
    context = [observed_message("old"), Message(role="user", content="edit", context=(latest,))]
    paths = VirtualPath(tmp_path)
    if recovering:
        pending = ToolCall(call_id="pending", name="write_text_file", arguments="{}")
        observations = FileObservations.for_recovery([*context, pending], paths, [pending])
    else:
        observations = FileObservations(context, paths)
    assert observations.check(replacement(tmp_path, "old"), {"content": "stale"}).code == (
        "filesystem.unread_file"
    )
    assert observations.check(replacement(tmp_path, "new"), {"content": "stale"}).code == (
        "filesystem.unread_file"
    )


@pytest.mark.parametrize("opaque", [b"new", "<redacted:secret>"])
def test_opaque_references_prevent_recreating_a_file_deleted_during_generation(tmp_path, opaque):
    """Observed existence survives an opaque attachment so a stale create cannot bypass it."""
    latest = (
        observed_message("new")
        .context[0]
        .model_copy(update={"content": opaque, "payload_redacted": isinstance(opaque, str)})
    )
    observations = FileObservations(
        [Message(role="user", content="edit", context=(latest,))], VirtualPath(tmp_path)
    )
    operation = Operation(
        tool_id="write_text_file",
        action=Action.FILESYSTEM_CREATE,
        target=FileTarget(path=str(tmp_path / "file.txt"), expected_exists=False),
    )
    assert observations.check((operation,), {"content": "stale"}).code == "filesystem.stale_context"


def test_opaque_reference_preserves_bytes_already_delivered_for_the_same_revision(tmp_path):
    """Repeated opaque metadata does not erase valid earlier coverage of the same source."""
    original = observed_message("source")
    opaque = original.context[0].model_copy(
        update={"content": "<redacted:secret>", "payload_redacted": True}
    )
    observations = FileObservations(
        [original, Message(role="user", content="edit", context=(opaque,))], VirtualPath(tmp_path)
    )
    assert observations.check(replacement(tmp_path, "source"), {"content": "new"}) is None
