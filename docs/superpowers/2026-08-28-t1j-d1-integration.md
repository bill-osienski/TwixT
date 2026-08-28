# D1 — Integration: Selection, Incumbent, E3b Binding, Seed Registration

**Status:** IMPLEMENTATION AND TESTS ONLY. **D1 has not run.** No model was
loaded, no JVM started, no T1j query issued, no seed registered or drawn, no
game played, nothing trained, nothing pushed. Every test mocks the process
boundary (`subprocess.run`) or the evaluator; the registries are read and never
written.

**Gates:** all three still `False` — `d1_probe.D1_EXECUTION_AUTHORIZED`,
`l0_match_command.L0_EXECUTION_AUTHORIZED`, `e4_screen_command.SCREEN_AUTHORIZED`.

**A SECOND BARRIER now stands beside D1's gate.** `[202614000, 202614227)` is
absent from `ACCOUNTED_SEED_INTERVALS`, and `_check_seed_registration` refuses
the run on that ground alone — before the deadline starts and before anything is
compiled. Opening the gate would not be enough; registering the block is a
separate reviewed edit belonging to the D1 execution authorization. Neither
barrier can be opened by opening the other.

---

## 1. The four pieces

| Piece | Reused unchanged | New |
|---|---|---|
| **Selection** (§12.1–12.3) | `d0_postmortem.bind_record` / `game_features` / `game_moves` / `phase_of`; the discovery/confirmation split | `d1_selection.py` — the rule, the matched controls, §12.2's digest, the seed assignment |
| **Incumbent** (§12.6, §5.2) | `load_reference_evaluator`, `e4_screen_reference.build`, `SeededReferenceAgent`, `eval_readout.select`, `eval_replay.ply_record`, `frozen_settings` | an opt-in capture seam; the raw-policy rank; `_Incumbent` |
| **E3b binding** (§5.5) | `T1jRuntime`, `IntegrationContext`, `make_binder`, `compare_state`, `check_postcond` | the replay timeout, explicit `ply_cap`, `AbortError` → `D1VoidError` |
| **Seed registration** (§12.5) | the four registries, `validate_task_executable`, `rng_stream_seeds` | `_check_seed_registration` — a read-only precondition |

## 2. The frozen §12.1 table, reproduced from the record

Recomputed read-only from `06_l0_match_results.jsonl` after digest binding
(`04_selection_recomputation.txt`):

| signature | role | n | cells |
|---|---|---:|---:|
| `mover_fragmentation` | position | **101** | **36** |
| `mover_fragmentation` | control | **60** | 24 |
| `created_threat` | position | **30** | **12** |
| `created_threat` | control | **36** | 12 |

**227 positions, 1,135 queries.** The rule as written reproduces §12.1's frozen
counts exactly; `select_all` refuses any departure from them rather than
reporting whatever the record yields.

### 🔴 D0's per-ply rows were never persisted

The D0 evidence package holds identity, inventory, aggregates, gate and
by-system — `run_d0` returns `rows` and `main` writes none of them. Selection
therefore **recomputes** the rows from the bound record. That is zero-inference
and costs about 40 s. *(Scope: the top-level keys of all five JSON artifacts in
`2026-08-27-t1j-d0-postmortem/`, plus a grep for the column name across
`docs/superpowers/evidence/**/*.json`.)*

### 🔴 24 positions are the same board state as another 24

227 retained positions cover **203 distinct states**. §12.1 and §12.3 deduplicate
**within** a cohort and nothing in §12 deduplicates across them, so a state can be
one signature's position and the other's control at once — with two different
seeds. **This is what was frozen, not a departure from it:** cross-cohort
deduplication would have totalled 203, not 227.

The consequence is recorded here so it is not discovered during a run: those 24
states get **four** JVM invocations per depth rather than two, and §12.7's
determinism check compares only **within** a pair. The second pair is never
compared against the first — a free cross-check the design does not take.

### One choice §12 does not make

**Which position gets which seed.** §12.5 fixes the interval and "one per
position" and stops. `SEED_ASSIGNMENT_ORDER` names the order used — the
signature table's rows, positions before controls, `(task_id, ply)` within a
group — so it is recorded rather than left implicit.

## 3. Two holes closed, and where the assertions sit

**`t1j_adapter.replay` had no timeout parameter at all** (§12.9 recorded it).
`timeout_s` is now a required keyword that refuses `None`, exactly as `ply_cap`
does, and `T1jRuntime` carries it. The assertion is made **at
`subprocess.run`**, never at the call site, with a dropped-last-hop negative
control — `subprocess.run(timeout=None)` waits forever and a single unforwarded
hop restores that silently while raising nothing.

**The binder's refusals were not D1 refusals.** `make_binder` raises
`e4_screen_runner.AbortError`, which is not a `D1Error`; untranslated it escapes
`main`'s handlers and a fully understood refusal reports as UNEXPECTED, exit 4,
instead of VOID, exit 3. The same defect shipped once already in
`l0_match_command`, which is why `_delegate` exists there.

🔑 **A test found a hole I had not: the binder's replay could time out and was
the one T1j call whose timeout was not a VOID.** `subprocess.TimeoutExpired` is
now translated too.

**Two other commands were touched.** `e4_screen_command` and `l0_match_command`
construct `T1jRuntime`, so both now pass the qualified 120 s
(`e4_screen_command.T1J_TIMEOUT_S`, sourced from the E4 preflight and reused by
L0 the way `JAR_SHA256` already is). ⚠ **Recorded, not fixed here:**
`l0_match_command` still calls `make_agent_factory` without `t1j_timeout_s`, so
L0's T1j *queries* remain unbounded. That is a different experiment's wiring and
outside this authorization.

## 4. The capture seam

`SeededReferenceAgent.__call__` computed `counts`, `root_value`, `root` and
`top2` and **discarded all four**, returning only the move — and every §5.2
observable lives in those four values. `capture` is opt-in, off by default, and
exposes only what was already computed: no extra search, no extra draw, the same
contract `play_eval_game(capture=True)` keeps.

Proved at the boundary that matters: with capture on and off, the selected moves
**and the state of both generators** are identical — with a negative control that
injects one stray `readout_rng.random()` and requires **both** detectors to fire.

The record is built by `eval_replay.ply_record`, the existing single definition
of visit rank, top-1 share, root-value perspective and top-2 with root-perspective
Q. The raw legal-move policy comes from `root.priors_raw`, and its rank uses the
**same** tie-break `ply_record` uses so the two ranks are comparable — a policy
rank, not a second visit rank. A root whose `priors` differ from its `priors_raw`
is a `VOID`: §12.6 specifies `add_noise=False`, which leaves them identical.

## 5. Controls — 30 injected defects, 30 rejected

Each defect was applied to the source, the test that must catch it was run, and
that test was **required to fail**. `__pycache__` purged with
`PYTHONDONTWRITEBYTECODE=1`; every one of the six touched files restored in a
`finally`. **0 stale anchors — `ALL CONTROLS RAN: True`.**

Covered: the registration check (disabled, endpoint-only, run after compilation),
the digest re-check, prefix legality, the binder never called, both
error-translation paths, `ply_cap`, the runtime timeout, the adapter's
unbounded-wait guard and its last hop, the capture seam (drawing, defaulted on,
not forwarded), the noise check, evaluator reload, the query spend, the rank
tie-break, the typed-instead-of-read incumbent identity, the digest's hash and
payload, the per-cell cap, deduplication, the incumbent-to-move filter, control
matching, in-place seed assignment, the count reconciliation, and the extracted
mover cut.

### 🔴 Three defects were NOT caught on the first pass

All three were my error, and two exposed real gaps:

1. **"replay timeout defaulted back to None" — a mis-specified control, not a
   defect.** Adding a signature default leaves the body's `if timeout_s is None:
   raise` still binding, so behaviour was unchanged. Replaced with the real
   switch-off: removing that body guard.
2. **"T1jRuntime accepts an unbounded timeout" — the control named a test in the
   wrong file.** An adapter test cannot see a `T1jRuntime` defect, and **no test
   asserted the runtime's own refusal**. `t1j_adapter.replay` refuses `None` too,
   so an adapter-level test passes whether or not the runtime carries its own
   guard. Added a case that reaches it alone.
3. **"frozen-count reconciliation removed" — the test asserted the counts
   directly**, so it passed with the internal check gone. Added a case that
   drives the reconciliation by declaring an expectation the record cannot meet.

🔑 **All three are the same lesson, and it is the one this workstream keeps
relearning: when several guards can catch a condition, a test proves only the
first one. Each later guard needs a case that reaches it alone.** Points 2 and 3
were guards no test could see; point 1 was a control that could never have
failed.

## 6. Two qualified modules edited, under the ruling

**`d0_postmortem.moved_by`** — the incumbent-vs-T1j cut, extracted verbatim from
`by_system` where it was inline. The regression control compares `by_system`'s
whole output against `05_by_system.json`, written by the pre-extraction code on
2026-08-27.

**`SeededReferenceAgent(capture=…)`** — threaded through `build_reference_agent`
and `e4_screen_reference.build`, defaulting off at every hop.

## 7. The structural gate test had a hole the size of this change

It scanned only module-level `FunctionDef`, so **a public class whose methods
execute was invisible to it**, and its "executes" markers named only `A.query`,
`compile_fn` and `_default_compile` — so a public entry point that loaded the
incumbent checkpoint and ran a 400-simulation search would have passed it. Both
widened, with a negative control over a synthetic module proving the scanner
finds an ungated function *and* an ungated method.

## 8. Suite, and "no model was loaded" as an observation

**3,512 passed, 4 skipped, 53 deselected, 0 failed** — from
`3,447 / 4 / 53` at `0817fc2`, so **+65 tests**, none failing. Run with all
three gates `False`.

`model_iter_0001.safetensors` is present in this tree, so a test that reached
`_default_load_evaluator` would have read a real model and the run would still
have been green. **Asserting that no model was loaded is not the same as
observing it.** A pytest plugin makes the load fatal; a negative control proves
the plugin fires on a real load; the nine test files this work touches then pass
**425/425** under it. *(Scope: those nine files, the only ones that can reach the
loader through this change — not the whole suite.)*

## 9. Deliberately absent

No D1 run. No seed registered. `_default_compile` still raises. The §5.4
per-position comparisons are analysis, not capture, and are not built — the
record persists the full raw policy and the full visit distribution, which is
what those comparisons need.
