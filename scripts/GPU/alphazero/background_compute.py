"""THE BACKGROUND-COMPUTE CHECK -- is BOINC on this machine? Read-only: one `ps` listing, nothing stopped, suspended or
restored. Nothing runs on import; it loads no MLX.

WHY (ledger entry 39, #101): the strength comparison's attempt 1 timed out VOID after a BOINC client started 3.8 h in
and took the machine whenever it was user-idle; the idle gate checks the launch only. Attempt 2 (amendment
`docs/superpowers/2026-10-27-mirror-halving-strength-attempt2-amendment.md`) runs with BOINC quit throughout, and this
check is how the run sees that: the wrapper refuses to launch, and the script ends the attempt VOID, unless it passes.

A process is BOINC when its EXECUTABLE PATH contains "boinc" (any case), or it runs as a BOINC account
(`boinc_master`, `boinc_project`) from outside macOS's system directories -- so the client, the Manager and project
workers count, and the per-user daemons macOS leaves running under those accounts (cfprefsd, lsd, ...) do not. The
executable, never the command line: a shell or `pgrep` whose ARGUMENTS mention BOINC is not BOINC (a command-line
match flagged this check's own caller). EXEMPT (the user's decision, 2026-10-27): BOINC's podman VM
(`/opt/podman/bin/` gvproxy, krunkit) does none of the work that slowed attempt 1; it is listed in every observation
as exempt, never as a hit. This process is excluded. A `ps` failure or an empty listing is a FAILED check, never a
pass.

    python -B -m scripts.GPU.alphazero.background_compute check
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

BOINC_USERS = ("boinc_master", "boinc_project")
SYSTEM_DIRS = ("/usr/sbin/", "/usr/libexec/", "/usr/bin/", "/sbin/", "/bin/", "/System/")
EXEMPT_DIRS = ("/opt/podman/bin/",)                 # BOINC's podman VM: exempt by the user's decision
SHOWN = 5                                              # hits kept in an observation
EXIT_CLEAR, EXIT_DETECTED, EXIT_FAILED, EXIT_USAGE = 0, 1, 3, 2

Proc = Tuple[int, str, str]                            # (pid, user, executable path)


def listing() -> List[Proc]:
    """The real machine: `ps -Aww -o pid=,user=,comm=`, this process excluded. Raises if ps fails."""
    out = subprocess.run(["ps", "-Aww", "-o", "pid=,user=,comm="], capture_output=True, text=True, check=True,
                         timeout=30).stdout
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) != os.getpid():
            procs.append((int(parts[0]), parts[1], parts[2]))
    return procs


def is_exempt(exe: str) -> bool:
    return exe.startswith(EXEMPT_DIRS)


def is_boinc(user: str, exe: str) -> bool:
    if is_exempt(exe):
        return False
    return "boinc" in exe.lower() or (user in BOINC_USERS and not exe.startswith(SYSTEM_DIRS))


def observe(listing_fn: Callable[[], List[Proc]] = listing) -> Dict[str, Any]:
    """{"ok", "n_procs", "hits", "n_hits", "exempt", "error"}: ok only if the listing worked, was not empty, and holds
    no BOINC process; exempt processes are listed, never counted."""
    try:
        procs = listing_fn()
    except Exception as e:                             # a failed check, never a pass
        return {"ok": False, "n_procs": 0, "hits": [], "n_hits": 0, "exempt": [], "error": f"{type(e).__name__}: {e}"}
    if not procs:
        return {"ok": False, "n_procs": 0, "hits": [], "n_hits": 0, "exempt": [], "error": "the process listing is empty"}
    hits = [f"{pid} {user} {exe[:120]}" for pid, user, exe in procs if is_boinc(user, exe)]
    exempt = [f"{pid} {user} {exe[:120]}" for pid, user, exe in procs if is_exempt(exe)]
    return {"ok": not hits, "n_procs": len(procs), "hits": hits[:SHOWN], "n_hits": len(hits), "exempt": exempt[:SHOWN],
            "error": None}


def describe(obs: Dict[str, Any]) -> str:
    if obs["error"]:
        return f"CHECK FAILED: {obs['error']}"
    exempt = f" (exempt: {'; '.join(obs['exempt'])})" if obs["exempt"] else ""
    if obs["hits"]:
        return f"BOINC DETECTED: {obs['n_hits']} of {obs['n_procs']} processes; " + "; ".join(obs["hits"]) + exempt
    return f"CLEAR: no BOINC process among {obs['n_procs']}" + exempt


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv != ["check"]:
        print("usage: python -B -m scripts.GPU.alphazero.background_compute check", file=sys.stderr)
        return EXIT_USAGE
    obs = observe()
    print(describe(obs), flush=True)
    return EXIT_CLEAR if obs["ok"] else (EXIT_FAILED if obs["error"] else EXIT_DETECTED)


if __name__ == "__main__":
    sys.exit(main())
