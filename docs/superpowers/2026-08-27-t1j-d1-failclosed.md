# D1 — Fail-Closed Machinery and Its Controls

**Status:** IMPLEMENTATION AND TESTS ONLY. **D1 has not run.** No model was
loaded, no JVM started, no T1j query issued, no seed registered or drawn, no
position queried, no game played, nothing trained, nothing pushed.

**Gate:** `d1_probe.D1_EXECUTION_AUTHORIZED = False`, read at **both** public
entry points — `run_d1` before compilation or probing, and `main` before argument
handling. An earlier version of this card said "exactly once, in the CLI"; that
described the design **before** review found that gating only the CLI protected
nothing, and it was wrong to leave standing. Opening D1 is a reviewed one-line change **plus**
a separate authorization. Nothing reads an environment variable, flag, config
file or import hook to reach it.

**Limits enforced**, frozen in plan §12.10: **120 s per T1j
query** · **90 minutes whole run**, monotonic, started
before compilation · **1,135 queries** ·
`[202614000, 202614227)`, unregistered.

---

## 1. What the tests prove

46 tests in `tests/test_d1_probe.py`.

| Requirement | How it is proved |
|---|---|
| Every T1j call receives `timeout_s=120` **at the subprocess boundary** | `subprocess.run` itself is patched and its `timeout` kwarg asserted. Not the call site: three hops (`make_agent_factory` → `T1jAgent` → `query`) each default to `None`, and one unforwarded hop silently restores unbounded waiting while raising nothing |
| Duplicate queries are separate `repeats=1` JVM calls | The adapter puts the mode in **argv** — `repeats=1` emits `query`, `repeats>1` emits `determinism` — so the prohibition is observable at the boundary, not asserted about a kwarg |
| The 90-minute monotonic deadline starts before compilation and yields `VOID` | A compile spy records `deadline.started` and `elapsed()` at the moment it is called |
| Forced query timeout produces no partial analysis | `subprocess.run` raises `TimeoutExpired`; the run raises `D1VoidError` and **no report file exists** afterwards |
| Forced deadline breach produces no partial analysis | Two separate tests — one tripping a check inside the position loop, one reaching **only** the final pre-write check |
| The 1,135-query cap is enforced | The budget refuses the query that would exceed it, the refused spend is not counted, and probing stops rather than overrunning |
| Retained prefixes | A missing prefix, a malformed prefix, and a prefix whose length disagrees with its ply are each `VOID` — a digest cannot be replayed, and a wrong-length prefix replays the **wrong position** |
| The seed interval is enforced and unregistered | Seeds either side of the interval, zero, negative, `True`, a string and `None` are each refused; and every seed in the interval is asserted absent from all four registries |

## 2. Controls — 19 injected defects, 19 rejected

A test that has never failed has not been shown to bind. Each defect was applied
to the source, the test that must catch it was run, and the test was **required
to fail**. `__pycache__` is purged with `PYTHONDONTWRITEBYTECODE=1` so a
same-length edit cannot leave a stale `.pyc` that makes the test import the old
module and pass vacuously. Restoration happens in a `finally`.

| Injected defect | Result |
|---|---|
| timeout dropped at the last hop | REJECTED |
| same-JVM `determinism` mode instead of two processes | REJECTED |
| only one invocation per depth | REJECTED |
| deadline started **after** compilation | REJECTED |
| wall-clock limit widened | REJECTED |
| query cap loosened by one | REJECTED |
| seed-interval check disabled | REJECTED |
| timeout no longer converted to `VOID` | REJECTED |
| final pre-write deadline check removed | REJECTED |
| gate flipped open | REJECTED |

🔴 **One of these failed on the first pass, and it mattered.** Deleting the
pre-write deadline check changed nothing, because the existing breach test tripped
an **earlier** check inside the position loop — so the final guard was never
exercised by any test. That is the same unexercised-abort defect §12.7 had to
correct, reappearing one layer down. A test was added that runs with no positions,
so the loop cannot fire and only the last check can catch the breach. The control
then rejected the defect.

## 3. What is deliberately absent

No position selection, no incumbent evaluator, no model load path, no seed
registration, and no D1 execution. `_default_compile` raises rather than
compiling. The runner is the **T1j side's** fail-closed skeleton; the incumbent
side, the selection implementation and the registration of
`[202614000, 202614227)` all belong to the execution
authorization.

## 4. Evidence

| File | Contents |
|---|---|
| `01_injected_defect_controls.py.txt` | the control harness, as run |
| `02_injected_defect_controls.txt` | its output: 10/10 rejected, source restored |
| `03_full_suite.txt` | full repository suite |

---

## 5. Review round 2 — four guards that did not bind

| Defect | Fix |
|---|---|
| **`run_d1` was ungated** — only the CLI checked. A direct Python caller bypassed the gate entirely | `run_d1` reads the gate before compilation or probing. Machinery moved below it into `_run_d1_unguarded`, so tests exercise internals **without lifting the gate in a fixture** |
| **The AST "own gate" test counted the assignment** — it passed with every guard read stripped | Counts only `Load`-context references, requires **≥2** (runner + CLI), with a negative control proving it fails when the reads are removed |
| **`probe_position` was public and ungated** — the same hole one level down, found by the new structural test | Renamed `_probe_position`. A test enumerates every public function that queries or compiles and asserts each reads the gate |
| **Identical invalid replies passed** — agreement was checked, validity never was | `_validate_reply` checks each reply alone: null sentinel, legality, completion, requested **and** completed depth, real move |
| **Two empty dumps compared equal** — absence read as agreement; the fixture emitted no dump, which is what hid it | Missing dump is `VOID`; the dump's final ply must equal the retained prefix length. Fixture now emits a complete reply with a 576-bit legal map |
| **The deadline was cooperative** — checks ran between stages and could not interrupt a hung one | External `SIGALRM` supervisor wrapping the whole run, **failing closed** if it cannot arm. The blocking-stage test went from 30 s to under 1 s |

🔴 **A stale control reported as a miss.** The first expanded run showed 18/19
with a `SKIP`: restructuring `run_d1` had broken that control's anchor, so it
**silently never executed** and was counted as "not caught". A control that does
not run proves nothing. Retargeted, and a stale anchor now reports loudly with an
explicit `ALL CONTROLS RAN` line instead of blending into the tally.

## 6. 🔴 The T1j toolchain is GONE — suite is 41 red for that reason alone

**3390 passed, 41 failed, 4 skipped.** Every one of the 41 failures is
`[jdk] pinned JDK component missing: bin/java` (48 occurrences), in exactly
the two files that hardcode a path into a **previous session's scratchpad**:

```
/private/tmp/claude-501/-Users-bill-projects-TwixT-Game/d037040d-…/scratchpad
    e2/jdk/home/jdk-17.0.20.1+1/Contents/Home   ← GONE
    e1/acq/release/t1j.jar                      ← GONE
```

**Zero failures mention `d1_probe`**, and the immediately preceding committed run
of the same suite, at `47493e7`, was 3,419 passed / 0 failed. The tests are behaving
**correctly** — failing closed because a pinned artifact is absent. What broke is
the environment, not a guard.

**Consequence, which outranks everything else here:** D1 execution is blocked
regardless of authorization. There is no JDK and no `t1j.jar`. Both must be
re-acquired and re-verified against E1's pinned hashes before any T1j work, and
the qualification suite is not reproducible until then. The checkpoint survives
because it lives in the repository; the toolchain did not, because it never did.
