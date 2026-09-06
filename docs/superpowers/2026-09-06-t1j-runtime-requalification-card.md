# Java-Only Runtime Requalification of E4Preflight — CARD, RUNNER BUILT, NOT AUTHORIZED, NOT RUN

**Status: FROZEN CARD + GATED RUNNER + MOCKED TESTS. Nothing has run.** No JVM
started, no helper executed, no prefix replayed, no T1j query, no incumbent
loaded, no seed drawn or registered, no gate opened, nothing played, nothing
pushed. The runner is `scripts/GPU/alphazero/runtime_requalification.py`; its
gate `RUNTIME_REQUAL_AUTHORIZED = False` is its **own** constant, read at all
three public entries (`main`, `worker_main`, `run_requalification`). It must not
read any other experiment's gate, and a test asserts none is named in the file.

**Scope its eventual authorization may permit: Java and the pinned toolchain
only** — one `javac`, eight E3bDump replays, eight E4Preflight queries. Still
barred, by construction: **no incumbent model load, no seed action of any kind,
no game, no D1 retry, no push.** It needs no seed interval: it never invokes the
incumbent and never derives an RNG stream.

---

## 1. The question, and what an answer would establish

**Does the edited `E4Preflight` (sha `fa662339…`, compile-only verified
2026-09-06 @ `89e66ad`) bind, search depth 6, return a legal move, re-bind its
searched position, and report a clean preference surface *with its own
before/after values*, on the eight frozen screen openings — including the exact
position on which the 2026-09-05 H1 match VOIDed?**

The 2026-09-05 edit added four fields to the helper's POSTCOND line
(`prefs_before/prefs_after/count_before/count_after`) so that a failed
preference check carries what the failing JVM actually compared. It has been
**compiled** but has never **executed**. This sample is the runtime step between
"it builds" and "a match may run through it".

🔴 **A PASS establishes OPERATION of this build on these eight cases, once each.
It does NOT explain the 2026-09-05 failure, and does NOT prove it cannot
recur.** The failing check compares a directory shared with every process on the
machine; eight clean samples do not rule out a race. The report says so in its
own `establishes` field.

## 2. What is frozen — the exact prefixes

`evidence/2026-09-06-t1j-runtime-requalification/01_frozen_prefixes.json`
sha256 `86b9e73801b6d643625845aa9b6b82052c1993436c691b34d2ad53401f09a305`

Derived read-only by `00_build_frozen_prefixes.py.txt` from **one already-frozen
source**: the sha256-pinned E4 screen plan's eight openings
(`06_endpoint_screen_plan.json`, sha `10cd8c31…`, verified by
`h1_viability_plan.load_source_plan` before a value is read). **Every opening is
taken, in plan order, at the opening ply (6). Nothing is selected.** The loader
re-verifies the file's hash **and** compares it with the plan: two frozen
sources, one statement; a disagreement is a refusal.

| # | opening | prefix (our coordinates) | digest | note |
|--:|---|---|---|---|
| 1 | `o1_center` | `(11,11) (12,13) (13,12) (10,13) (12,10) (14,14)` | `0ae621381af1` | |
| 2 | `o2_offcenter` | `(10,12) (13,10) (12,13) (11,9) (14,12) (9,11)` | `4829a23c205d` | |
| 3 | `o3_low` | `(15,11) (12,12) (13,10) (14,14) (11,11) (16,13)` | `0dc546b435a2` | **the H1 failure**: task `h1match-060-strong6-o3_low-t1j_red-r4`, query at ply 6, zero ply records |
| 4 | `o4_high` | `(8,12) (11,11) (10,13) (9,9) (12,12) (13,14)` | `5fdb19f000a0` | |
| 5 | `o5_wide_left` | `(11,7) (12,10) (13,8) (10,11) (15,9) (14,12)` | `216282d7224b` | |
| 6 | `o6_wide_right` | `(11,16) (12,13) (13,15) (10,12) (15,14) (14,11)` | `e635a4ce4a63` | |
| 7 | `o7_diagonal` | `(9,9) (12,12) (11,10) (14,13) (13,11) (16,14)` | `a6603d7fcb09` | |
| 8 | `o8_contact` | `(12,11) (11,12) (13,13) (12,14) (14,12) (10,10)` | `cd944495ebcd` | |

All eight are red-to-move at ply 6 — the position T1j searches first in the
`t1j_red` arm, which is where the match VOIDed. ⚠ **This says nothing about
ply-7 positions (the `t1j_black` arm's first query) or about mid-game plies.**

## 3. Depth, invocations, and the process budget — 8 / 16 / 17

| count | value | what it is |
|---|---:|---|
| depth | **6** | the match's `mdPly`; the depth of the failed query. One depth, deliberately: this is a sample of operation, not an endpoint screen |
| invocations per prefix | **1** | cross-process agreement was qualified by E3a; re-proving it is not this card's question |
| **T1j queries** | **8** | 8 prefixes × 1. **This is the budget** (`QUERY_CAP`), and the public entry requires exactly the frozen eight in the frozen order |
| replay launches | **8** | one E3bDump replay per prefix, for the E3b binding |
| **helper launches** | **16** | 8 + 8 — every process that runs the T1j jar |
| **Java processes** | **17** | 16 + **one `javac`**, compiling the four committed sources once, up front |

🔑 The three counts are different units and are not interchangeable — the
low-ply card's correction, inherited. `repeats>1` is prohibited (a test asserts
no `determinism` mode is ever requested).

## 4. Cost basis and frozen limits — 120 s per call, 900 s inner, 960 s outer

Measured `wall_ms` per depth-6 query from the E4 preflight's own record
(`2026-08-25-t1j-e4-preflight-attempt4/03_results.jsonl`, 6 rows, JVM startup
included), recomputed 2026-09-06: **min 202.6 · median 681.2 · max 2734.5**.
Indicative T1j subtotal: 8 × 2.73 s ≈ **22 s**, excluding the 8 replays
(unmeasured anywhere), one `javac` (0.47 s on 2026-09-06), output and fsync.

| limit | value | mechanism |
|---|---|---|
| per-call timeout | **120 s** | the qualified E4 limit, passed at **every** `subprocess.run` (replay and query); a test asserts every call carries it. `subprocess.run(timeout=)` kills the JVM, which is its direct child |
| inner whole-run deadline | **900 s** | D1's `Deadline` + SIGALRM supervisor, reused not restated: one clock, one origin, armed from the remaining time. A breach is a **VOID** reported by the worker (exit 3) |
| **outer process-tree cap** | **960 s** = 900 + 60 grace | `main` runs the stages in a **worker subprocess started in its own session**; on expiry it `SIGTERM`s the **process group**, waits 5 s, `SIGKILL`s the group, then **probes the group until empty** (`killpg(…, 0)` → ESRCH) and records whether it cleared. Exit **6** |

🔴 **Why the outer cap exists (review of the compile-only driver, 2026-09-06):**
`subprocess.run(timeout=)` on a Python worker kills the worker, not necessarily
its `javac`/`java` child. The worker here is a process-group leader, so the
kill reaches every JDK descendant. A mocked test proves it with a Python
grandchild (`sleep 300`) that must be dead after the supervisor fires. The
60-second grace lets the inner supervisor report a deadline VOID as a VOID
before the outer one turns it into a kill.

## 5. What is verified about the new POSTCOND fields — through the REAL path

The query goes through the real `T1jAgent` (the object the match queries with),
whose `_query` seam is used only to **keep** the raw reply; it still calls
`A.query`. Consequences, each pinned by a mocked test:

- A clean reply's record carries the four observation values from
  `parse_postconds` **and** the diagnostic reader `postcond_prefs_observation`;
  the two must agree or the run VOIDs.
- **A reply WITHOUT the observation is a VOID**, not a legacy reading: the
  compiled class emits the fields on every line, so their absence means a
  different class ran. A **partial** observation is a VOID (unreadable
  instrument).
- **`prefs_ok=false` is a recorded FAIL** on that prefix; the run continues.
  The reply arrives as the same `AbortError`-from-`HelperOutputError` chain the
  H1 match's diagnostic reads, and the record is built with the H1 runner's own
  `_bounded_excerpt` / `_helper_text`: the test pushes POSTCOND past the 800-char
  excerpt with eleven FAIL lines and asserts the full observation still lands in
  `helper_prefs_observed`.
- **Preference checks are intact.** `prefs_ok` is the helper's verdict, `clean`
  requires it, and nothing here relaxes or reinterprets it.

## 6. Output location — create-only

`docs/superpowers/evidence/2026-09-06-t1j-runtime-requalification/`

| file | content |
|---|---|
| `00_build_frozen_prefixes.py.txt` | **exists** — the derivation script |
| `01_frozen_prefixes.json` | **exists** — the frozen input above |
| `02_prerun_verification.txt` | gates, toolchain hashes, limits, input sha256 (at run time) |
| `03_run_command.txt` | the exact invocation |
| `04_stdout.txt` / `05_stderr.txt` / `06_exit.txt` | the run's own streams and **exit code** |
| `07_requal_records.json` | the record — **create-only, fsynced, written once**; a VOID writes nothing |
| `07_requal_records.json.t1j_classes/` | the classes compiled for the run |
| `08_full_suite_after_gate_restored.txt` | the suite, gate back to `False` |
| `A_MANIFEST.sha256.txt` | hashes of all of the above |

## 7. Outcome vocabulary and exit codes — one meaning each

| exit | outcome | meaning |
|--:|---|---|
| **0** | `PASS` | all 8 prefixes bind; all 8 queries complete depth 6, return a move legal in *both* engines, re-bind the searched position, present a clean surface with the exact reflection count **and** a complete observation |
| **2** | `FAIL` | any of those does not hold on ≥ 1 prefix. **A RESULT, not an abort** — the run completes and records every reply; `h1_failed_prefix` in the report says what happened on `o3_low` without searching |
| **3** | `VOID` | the instrument: identity mismatch, per-call timeout, inner deadline breach, unparseable output, a reply that could not have come from the compiled class, parser/reader disagreement. **Nothing is written** |
| **4** | `UNEXPECTED` | an unnamed exception, reported rather than escaping as a traceback |
| **5** | `UNAUTHORIZED` | the gate, at whichever entry was reached; nothing spawned, nothing written |
| **6** | `TIMEOUT` | the outer supervisor killed the worker's process group; whether it cleared is on stderr |
| **7** | `REFUSED` | a precondition refusal by the harness (wrong prefixes, unreadable input, existing output) |

🔴 **A `FAIL` exits 2, not 0.** The low-ply runner exited 0 on FAIL ("a
result"), and the 2026-09-05 match wrapper exited 0 unconditionally, so its VOID
was invisible to anything that read only the status. Here a pass and a result
are different numbers, and a mocked test pins every mapping.

## 8. What this card does NOT decide

It does not authorize itself. It does not reserve or register a seed interval,
does not amend any preregistration, does not touch E3bDump (which still emits
the earlier POSTCOND shape) or D1's finally-VOID. A PASS would make the edited
helper **operational on these cases**; whether that suffices for a fresh H1
interval is a separate decision, on the results.
