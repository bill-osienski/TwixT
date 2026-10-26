"""THE IDLE GATE -- a launch precondition for GPU runs, enforced by the wrappers that use it (first: the mirror-averaging
+ root-halving screen and strength comparison, `docs/superpowers/evidence/mirror-halving-{screen,strength}-run/
01_wrapper.sh`). Nothing runs on import; it loads no MLX.

WHY (ledger entry 39, #93): C1 launched at load 4.0 with Spotlight near 313% CPU because the wrapper's caller PRINTED
the load and launched regardless. Here the check decides: the wrapper launches only if `wait` exits 0.

`wait`: sample every INTERVAL_S seconds; the machine is IDLE when the 1-minute load average is below LOAD_LIMIT and no
other process is above CPU_LIMIT percent (ps's %CPU; this process excluded). It must stay idle for HOLD_S seconds in a
row (any busy sample restarts the hold). Exit 0 once held; exit 1 if MAX_WAIT_S passes first (the wrapper then refuses,
before the run starts). Every sample is printed, so the wrapper's record keeps the evidence.

    python -B -m scripts.GPU.alphazero.idle_gate wait
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Callable, List, Sequence, Tuple

LOAD_LIMIT = 3.0
CPU_LIMIT = 50.0
HOLD_S = 60.0
INTERVAL_S = 5.0
MAX_WAIT_S = 1800.0
EXIT_IDLE, EXIT_BUSY, EXIT_USAGE = 0, 1, 2

Sample = Tuple[float, List[Tuple[float, str]]]          # (1-minute load, [(%cpu, command)] of the other processes)


def sample() -> Sample:
    """The real machine: os.getloadavg and `ps -Ao pid=,pcpu=,comm=`, this process excluded."""
    out = subprocess.run(["ps", "-Ao", "pid=,pcpu=,comm="], capture_output=True, text=True, check=True).stdout
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and int(parts[0]) != os.getpid():
            procs.append((float(parts[1]), parts[2]))
    return os.getloadavg()[0], procs


def busy_reasons(s: Sample) -> List[str]:
    load, procs = s
    reasons = [f"load {load:.2f} >= {LOAD_LIMIT}"] if load >= LOAD_LIMIT else []
    reasons += [f"{cmd} at {cpu:.1f}% CPU > {CPU_LIMIT}%" for cpu, cmd in procs if cpu > CPU_LIMIT]
    return reasons


def wait(sample_fn: Callable[[], Sample] = sample, clock: Callable[[], float] = time.monotonic,
         sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = print) -> int:
    """EXIT_IDLE once the machine has been idle for HOLD_S in a row; EXIT_BUSY if MAX_WAIT_S passes first."""
    start = clock()
    idle_since = None
    while True:
        now = clock()
        reasons = busy_reasons(sample_fn())
        if reasons:
            idle_since = None
            say(f"t={now - start:7.1f}s BUSY: " + "; ".join(reasons[:5]))
        else:
            idle_since = now if idle_since is None else idle_since
            say(f"t={now - start:7.1f}s idle for {now - idle_since:.1f}s")
            if now - idle_since >= HOLD_S:
                say(f"IDLE: held {HOLD_S:.0f}s (load < {LOAD_LIMIT}, no process > {CPU_LIMIT}% CPU)")
                return EXIT_IDLE
        if now - start >= MAX_WAIT_S:
            say(f"BUSY: not idle for {HOLD_S:.0f}s in a row within {MAX_WAIT_S:.0f}s -- do not launch")
            return EXIT_BUSY
        sleep(INTERVAL_S)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv != ["wait"]:
        print("usage: python -B -m scripts.GPU.alphazero.idle_gate wait", file=sys.stderr)
        return EXIT_USAGE
    return wait(say=lambda m: print(m, flush=True))


if __name__ == "__main__":
    sys.exit(main())
