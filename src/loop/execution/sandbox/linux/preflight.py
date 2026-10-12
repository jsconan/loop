"""Attest loaded unit authority against the fixed installed configuration.

These checks do not enable a backend or replace protected admission/lifecycle qualification.
The owning installer supplies fixed unit expectations; requests cannot choose this authority.
"""

import re
import subprocess
import time
from contextlib import ExitStack

from ....constants import NATIVE_MANAGEMENT_TIMEOUT_SECONDS
from ....utils.process import ProcessCaptureStatus, supervise_process
from .constants import SYSTEM_COMMAND_PATH, SYSTEM_LOCALE, SYSTEMCTL_PATH
from .host import HostFile, bind_host_file

_MAX_PREFLIGHT_STDOUT_BYTES = 65_536
_MAX_PREFLIGHT_STDERR_BYTES = 4_096


def _run(argv: list[str], deadline: float):
    """Supervise one fixed preflight operation without interactive system authentication."""
    if time.monotonic() >= deadline:
        raise TimeoutError("Native preflight deadline expired.")
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/",
        env={
            "PATH": SYSTEM_COMMAND_PATH,
            "LC_ALL": SYSTEM_LOCALE,
            "SYSTEMD_PAGER": "",
            "SYSTEMD_COLORS": "0",
        },
        close_fds=True,
        start_new_session=True,
    )
    result = supervise_process(
        process,
        deadline,
        stdout_limit=_MAX_PREFLIGHT_STDOUT_BYTES,
        stderr_limit=_MAX_PREFLIGHT_STDERR_BYTES,
    )
    if (
        result.status is not ProcessCaptureStatus.COMPLETED
        or result.exit_code != 0
        or result.stderr
        or result.stdout_discarded
        or result.stderr_discarded
    ):
        raise OSError("Native unit attestation failed.")
    return result.stdout


def attest_units(
    files: dict[str, HostFile],
    expectations: dict[str, dict[str, str]],
    deadline: float,
) -> None:
    """Compare the actual system manager's loaded unit properties with fixed setup authority.

    Expectations come from the trusted installer, never request/model data. Use actual socket
    and concrete service instance names as keys. Both must correspond to the receipt's unit
    fragment names (a service template may bind one concrete instance). Mandatory protected
    cgroup/no-delegation/control-group kill properties supplement the installer's exact ExecStart,
    user, capability and namespace expectations. Missing/extra properties, aliases, drop-ins,
    transient units and pending reloads refuse. Compare exact ExecStart command authority while
    ignoring only its manager-reported timestamps, PID and exit metadata; fixed MainPID may
    be attested separately for a running receiver. Revalidate installed bytes after the query.
    Report mismatched fixed property names without values or unknown property names.
    This authenticates loaded configuration, not runtime descendant immutability or admission.

    Args:
        files (dict[str, HostFile]): Full installation_files result from initial host preparation.
        expectations (dict[str, dict[str, str]]): Exactly two fixed units and their expected
            manager properties; service must include ExecStart, User and Group, socket Listen.
        deadline (float): Absolute monotonic preparation deadline, bounded internally to five
            seconds.

    Raises:
        ValueError: Unit names, expectations, loaded properties or bound authority are invalid.
        TimeoutError: Preparation budget expired before manager inspection.
        OSError: Required protected manager tooling or bounded inspection fails.
    """
    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    tool = bind_host_file(SYSTEMCTL_PATH, deadline=deadline)
    roles = {"socket": files["socket_unit"], "service": files["service_unit"]}
    if len(expectations) != 2:
        raise ValueError("Expected exactly the installed socket and service units.")
    with ExitStack() as inputs:
        inputs.enter_context(tool.open_verified(deadline=deadline))
        for bound in files.values():
            inputs.enter_context(bound.open_verified(deadline=deadline))
        for kind, bound in roles.items():
            candidates = [name for name in expectations if name.endswith("." + kind)]
            if len(candidates) != 1:
                raise ValueError("Invalid installed unit roles.")
            name = candidates[0]
            fragment = name
            if kind == "service" and "@" in name:
                fragment = name.split("@", 1)[0] + "@.service"
            if (
                re.fullmatch(r"[A-Za-z][A-Za-z0-9_@.-]{0,191}", name) is None
                or bound.path.name != fragment
            ):
                raise ValueError("Unit does not match installed fragment authority.")
            expected = expectations[name]
            required = {"ExecStart", "User", "Group"} if kind == "service" else {"Listen"}
            if not required <= expected.keys() or any(
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key)
                or not isinstance(value, str)
                or "\n" in value
                or "\x00" in value
                for key, value in expected.items()
            ):
                raise ValueError("Incomplete fixed unit enforcement expectations.")
            invariant = {
                "Id": name,
                "LoadState": "loaded",
                "FragmentPath": str(bound.path),
                "DropInPaths": "",
                "NeedDaemonReload": "no",
                "Transient": "no",
            }
            if kind == "service":
                invariant.update(
                    ProtectControlGroups="yes", Delegate="no", KillMode="control-group"
                )
            if any(key in expected and expected[key] != value for key, value in invariant.items()):
                raise ValueError("Fixed unit expectations weaken required invariants.")
            wanted = expected | invariant
            output = _run(
                [
                    str(tool.path),
                    "--system",
                    "--no-pager",
                    "show",
                    "--property=" + ",".join(sorted(wanted)),
                    "--",
                    name,
                ],
                deadline,
            )
            actual = {}
            for line in output.splitlines():
                key, separator, value = line.partition("=")
                if not separator or key in actual:
                    raise ValueError("Ambiguous loaded unit properties.")
                actual[key] = value
            # ExecStart reports per-process timestamps and exit metadata as well as authority.
            # Compare the complete fixed command while normalizing only that manager suffix.
            if "ExecStart" in actual:
                actual["ExecStart"] = re.sub(
                    r" ; start_time=\[[^\]\n]*\] ; stop_time=\[[^\]\n]*\]"
                    r" ; pid=[0-9]+ ; code=\([^\)\n]*\) ; status=[0-9]+/[0-9]+ }$",
                    " ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }",
                    actual["ExecStart"],
                )
            if actual != wanted:
                differences = sorted(key for key in wanted if actual.get(key) != wanted[key])
                if actual.keys() - wanted.keys():
                    differences.append("<unexpected>")
                raise ValueError(
                    "Loaded unit differs from installed enforcement authority: "
                    + ", ".join(differences)
                    + "."
                )
        for bound in (tool, *files.values()):
            inputs.enter_context(bound.open_verified(deadline=deadline))
