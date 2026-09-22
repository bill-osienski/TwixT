# H4 §4B — ACCEPTANCE-MODE QUALIFICATION CARD

**Written 2026-09-22. DESIGN ONLY.** Nothing is implemented, no gate exists,
nothing runs, no seed is reserved, no game is played.

🔴 **THIS CARD IS THE AUTHORITY FOR §4B.** The replacement card
(`2026-09-21-t1j-h4-replacement-card.md`) is **preserved unedited** as the
historical design. Its §4B-relevant clauses that are now stale are superseded
**here, clause by clause** (§1). Outside §4B — the scientific core, the pilot,
the study, persistence — the replacement card stands unchanged. Implementation
either **conforms** to this card or returns here for an explicit amendment; a
card completed from its implementation only ratifies what was built.

> **What §4B must establish.** That the **production adapter**, in an opt-in H4
> mode whose default stays **OFF**, accepts exactly the replies the repair
> qualification established as legitimate — native-initial and searched — at the
> frozen positions, refuses everything else, names which routine answered, and
> records the process identity of every JVM; while **every existing caller keeps
> the behaviour it was qualified with**.

---

## 0. What is established, and what is not

**Established** — the H4 repair qualification, **CLEAN** @ `adcd48f`: through the
opt-in `h4query` helper mode, at the 16 frozen positions, depth 6, 72/72
subprocesses, every reply was a legal, board-coherent move; plies 0–3 answered
natively (`native_initial_first` ×5, `native_initial_second_to_fourth` ×45),
plies 4–5 searched (×6); one `PROC` per JVM, `refl_n` 4 / 1, `MatchData` read
back as frozen with `identity=true`.

**Not established:**

* 🔴 that the **production adapter** accepts any of it — **today it refuses every
  native reply**, at three separate sites (§2);
* that `native_initial_fifth_or_more` can occur at all — **never observed** in
  this programme, at any position;
* anything about games, the H4 runner, the pilot, or strength.

---

## 1. The replacement card's stale clauses — SUPERSEDED here, preserved there

| # | replacement card | what it says | superseded by |
|---:|---|---|---|
| 1 | §3.1, "search" row | `T1jAgent.__call__ → A.query(...)` — the **default** `query` mode | H4 mode queries through the opt-in **`h4query`** mode (`inject_matchdata=True`), because the default mode **throws at ply 0** (§4A). The lifecycle is unchanged: a fresh JVM per T1j move (§7) |
| 2 | §4.1 | *"THERE ARE TWO REFUSAL POINTS, NOT ONE"* | there are **three** sites that refuse a legitimate native reply, and a **fourth** condition that refuses **every** `h4query` reply (§2) |
| 3 | §4.1 | *"Only the completion condition is relaxed"*, and `check_postcond` *"must stay exactly as it is"* | `check_postcond` judges `PostCond.clean`, which includes **`failures == 0`**, and expects `refl_n` **3**. Followed literally, §4.1 would leave every native reply refused. In H4 mode the safety surface is judged field by field and `failures` goes to the classifier (§4) |
| 4 | §4B.1 item 2 | `QUERY_REFL_N = 3` | opt-in `h4query` = **4** (`h4_repair_qualification.QUERY_REFL_N_OPTIN`); the **default stays 3**; replay stays **1** |
| 5 | §4B.1 | no `MatchData` check | **mandatory** readback: exactly one `MATCHDATA` line, `pieRule=false`, 24×24, `ystarts=true`, **`identity=true`** (§4 step 4) |
| 6 | §4B.1 item 5, §4B.2 item 5 | no ply restriction on a completed search | **a completed search is impossible at plies 0, 1 and 2** — the repair card's §4 amendment, enforced by the shared classifier (§4 step 7) |
| 7 | §4B.1 | *"Both existing refusal sites must consume the SAME classification result"* | **all three** sites; and the one classification is `h4_repair_qualification.classify_reply` **itself** (§4) |
| 8 | §4B.3 | five negative controls, no positive baseline | the full control set of §6, including one **constructed real-entry-point baseline** for `native_initial_fifth_or_more` |
| 9 | §4C | *"Whether the helper accepts a zero-length position is unqualified in both paths"* | settled: `E3bDump` accepts it (§4A); `E4Preflight` **throws** on the default path and **answers** on `h4query` (repair qualification, 5/5). **Arm B exists in this form ONLY through the opt-in path** |
| 10 | §7.1 table | *"Java emits PROC per JVM — exists"* | untrue on the **replay** path when written — §4A counted **0 PROC on 10 replays**; true on both paths only since the repair's Change 2. A production caller is still **none** — §5 wires it |
| 11 | §10 | *"§4A … is the NEXT AUTHORIZABLE ACTION"*; *"§4B … is not written until §4A reports"* | §4A reported `STOP_ZERO_LENGTH_REFUSED`; the repair was designed, built and qualified CLEAN; **this card is §4B** |
| 12 | §4A.2 | *"prefer … `move_count=0`"* | already **retired** by §4A's `10_correction.md` §2; restated so no reader revives it |

**Still binding, unchanged:** the §3.1 lifecycle (fresh search JVM per T1j move,
separate fresh replay JVM per binder call, no reuse, classes compiled once);
§4B.2 items 1–7 as refined here; §4B.3's ply ≥ 6 control; the requirement that a
predicate be driven through its **real entry point** and a CLI qualified as a
**fresh subprocess**.

---

## 2. 🔴 THE REFUSAL SITES — read from source: THREE, not two

A legitimate native-initial reply through `h4query` arrives as
`exit 3 · failures 1 · completed false · usealphabeta false · currentMaxPly 0 ·
refl_n 4`. `T1jAgent.__call__` (`e4_screen_integration.py`) refuses it at:

| # | site | where | why it refuses a legitimate native reply |
|---:|---|---|---|
| 1 | `if rc != 0 or len(recs) != 1:` | `:251` | exit 3 |
| 2 | `check_postcond(...)` → `if not p.clean:` | `:263` → `:80`; `PostCond.clean` ends `and self.failures == 0` (`t1j_adapter.py:557-559`) | failures 1 |
| 3 | `if not r.completed or r.requested_depth != self.depth:` | `:277` | completed false |

and one more condition refuses **every** `h4query` reply, native or searched:

| # | site | where | why |
|---:|---|---|---|
| 4 | `check_postcond(out, expected_refl=QUERY_REFL_N, …)` → `if p.refl_n != expected_refl:` | `:263` → `:82` | `h4query` reports 4; `QUERY_REFL_N` is 3 |

🔴 **The replacement card named sites 1 and 3.** Site 2 is a trap this programme
has already fallen into once: §4A's own runner judged `clean`, found that it made
every non-searching reply an "instrument failure", and fixed it **in its runner**
— but that fix never reached the adapter. A mode that relaxed only the
completion condition, as §4.1 prescribes, would **still refuse every native
reply at site 2**, and at site 4 **every reply of any kind**.

---

## 3. THE IMPLEMENTATION BOUNDARY — frozen

### 3.1 One switch, default OFF

* **`T1jRuntime(..., h4_acceptance: bool = False)` is the ONLY switch.** The
  binder and the agent both read it from the runtime they share, so an H4 agent
  can **never** be paired with a default binder, or the reverse. No agent
  keyword, no binder keyword, no environment variable, no configuration file.
* The default, `False`, **is** today's fail-closed behaviour. It keeps every
  guard on; it switches nothing off.
* In H4 mode, **construction refuses**:
  * a depth other than **6** — the qualified classifier is depth-6 specific
    (`DEPTH = 6`, `SEARCHED_CURRENT_MAX_PLY = 7`);
  * a query timeout of **`None`**. `T1jAgent` and `make_agent_factory` default
    `timeout_s` to `None` (`:231`, `:292`), an **unbounded wait**. That stays as
    qualified for default callers and is **refused** in H4 mode.

### 3.2 What may change — at implementation, separately authorized

| file | change |
|---|---|
| `e4_screen_integration.py` | the switch on `T1jRuntime`; an **additive** H4 branch in `T1jAgent.__call__` and `make_binder`; a per-task process-record list on `IntegrationContext` |
| `h4_4b_acceptance_qualification.py` | **new** — the gated runner |
| `tests/test_h4_4b_acceptance_qualification.py` | **new** — every §4B test |
| `tests/test_gate_inventory.py` | `EXPECTED_GATES` 13 → **14**, in one deliberate edit with its stale-count control |
| `injected_defect_controls.py` | the §4B controls of §6.5, **appended** (data only) |

🔴 **The default path's statements stay byte-identical** — in particular the four
lines existing injected-defect controls anchor on in this module: the binder's
and the agent's failure messages, the chained-transcript `raise … from
A.HelperOutputError(message, out)`, and `T1jRuntime`'s `timeout_s is None`
refusal. (At `adcd48f` the harness holds **783** controls by this card's count,
**26** anchored in adapter code; the harness's own derived count is the
authority.)

### 3.3 What must NOT change

* **`t1j_adapter.py`** — `query(inject_matchdata=…)`, `parse_procs` and
  `parse_matchdata` already exist; **`PostCond.clean` stays exactly as it is**,
  because every default caller is qualified against it.
* **`h4_repair_qualification.py`** — `classify_reply`, `DEPTH`,
  `QUERY_REFL_N_OPTIN`, `derive_matrix`, `verify_matrix` are **consumed, never
  copied or edited**.
* **The Java helpers** — §4B changes no helper.
* **Every existing caller** — D1, E4, L0, H1, H2, H3, the low-ply
  qualification, runtime requalification, §4A and the repair qualification. None
  opts in.
* `QUERY_REFL_N = 3`, `REPLAY_REFL_N = 1`.

### 3.4 Out of scope — named so it cannot be smuggled in

The H4 runner and game loop · the empty-board state factory (`opening_bound =
0`) · per-game persistence and replay export (replacement §7) · removing
`h3_study_analysis`'s duplicate-pair rule (replacement §1.3 — **still owed before
any pilot**) · seeds · the pilot · the incumbent.

🔑 §4B drives the production agent and binder **directly over the frozen
matrix** (§7) — **never through the game loop** — so no game exists.

---

## 4. THE H4-MODE AGENT PATH — frozen order

Every step **fails closed** with `AbortError(PHASE_MOVE, …)`, and every refusal
**chains the full transcript** as its cause (`from A.HelperOutputError(message,
out)`), so a STOP's durable record carries the raw stdout — the gap that once
left a real D1 abort unexplained.

0. **The existing pre-checks, unchanged** — colour to move; move log length ==
   ply.
1. **Query through `h4query`** (`inject_matchdata=True`), a fresh JVM, a finite
   timeout.
2. **`PROC`** — exactly one line; recorded (§5).
3. **`POSTCOND`** — exactly one line. The **safety surface is judged field by
   field**: `no_throw`, `windows == 0`, `frames == 0`, `headless`, `prefs_ok`,
   `refl_ok`, and **`refl_n == QUERY_REFL_N_OPTIN` (4)**. 🔴 **`failures` is NOT
   judged here** — not directly and not through `clean`. It is recorded and
   passed to step 7.
4. **`MATCHDATA`** — exactly one line; `pieRule=false`, `xsize=24`, `ysize=24`,
   `ystarts=true`, **`identity=true`**.
5. **The QUERY line describes the position sent** — exactly one record; `q ==
   1`; `moveNr == state.ply`; `to_move` == our side; `requested_depth == 6`.
6. **The searched position** — exactly one same-process dump; `compare_state`
   finds **no divergence**.
7. 🔑 **THE SHARED CLASSIFICATION** —
   `classify_reply(ply=state.ply, rec=r, exit_status=rc, failures=post.failures)`,
   called **once**. **In H4 mode this is the ONLY place `rc`, `failures` and
   `completed` are judged:** sites 1–3 of §2 are not relaxed one at a time — they
   are **replaced** by this one result. It accepts, or it refuses (its
   `H4RQStop` becomes `AbortError(PHASE_MOVE)` carrying the classifier's own
   message). It enforces the repair card as amended: a native reply only at
   plies 0–5, with exit 3 + failures 1, named by routine; a completed search
   only at plies ≥ 3, with exit 0 + failures 0; **everything else refused**.
8. **A usable move** — not the null sentinel, not `None`, legal in T1j's own
   report, and in **our** `state.legal_moves()`.
9. **Record** the per-call record (§5); return the move.

🔴 **The classifier must be `h4_repair_qualification.classify_reply` itself** —
the object the CLEAN qualification exercised — **imported at call time**
(`h4_repair_qualification` imports `e4_screen_integration` at module level, so a
module-level import back would be a cycle). It is **never re-implemented, never
copied, and never wrapped in logic that re-judges its inputs**.

**The binder in H4 mode:** every existing replay check unchanged, plus exactly
one `PROC`, recorded.

⚠ **Naming.** `ctx.bump("searched_binds")` (`:274`) counts **dump re-binds
whatever routine answered**. In H4 mode the per-call **`source`** is the
authority on whether a search ran; the counter keeps its name because default
callers read it.

---

## 5. PRODUCTION PROC PLUMBING — H4 mode only

Replacement §7.1 found the parser written, tested, and **unused in production**.
In H4 mode the adapter appends **one record per subprocess** to the
`IntegrationContext`, for the current task:

| field | query | replay |
|---|:---:|:---:|
| role (`query` / `replay`) | ✓ | ✓ |
| ordinal — per task, monotonic, **shared by both roles** | ✓ | ✓ |
| `task_id`, board ply, return code | ✓ | ✓ |
| parsed `PROC` — `pid`, `java_version`, `vm`, `headless`, `prefs_factory` | ✓ | ✓ |
| postcondition fields, **including `failures` and `refl_n`** | ✓ | ✓ |
| `source` — which routine answered | ✓ | — |
| `MatchData` readback | ✓ | — |
| move and telemetry — `usealphabeta`, `currentMaxPly`, `completed`, `completed_depth`, `moveNr` | ✓ | — |

* **Anything other than exactly one `PROC` per JVM is a refusal**, on both
  paths.
* Pair and arm identity are the H4 runner's to attach; the adapter records
  `task_id`.
* 🔴 **Default callers do not parse `PROC`.** A new refusal on their path would be
  a behaviour change none of them was qualified with.
* Persisting these records into H4's canonical record is the runner's job
  (out of scope). **§4B makes them exist, and proves they are complete.**

---

## 6. CONTROLS — each proven, through the REAL entry point

**The real entry point** is `T1jAgent.__call__` on an agent built by
**`make_agent_factory`** from an H4-mode `T1jRuntime`, and the binder from
**`make_binder`** on the same runtime — stubbed at the **process boundary**
(`subprocess.run`), **never at the `_query` seam**, so the adapter's real argv
construction and parsing run. A permissive double hides the seam.

### 6.1 Negative controls — each must REJECT

| control | what it must reject |
|---|---|
| **search at ply 0** / **ply 1** / **ply 2** | a clean completed search (exit 0, failures 0) — **three separate controls**, so dropping any one ply fails a control of its own |
| native claim at **ply ≥ 6** | native telemetry where `initialMove()` returns null by dispatch — **synthetic**, the matrix ends at 5 |
| native signature, wrong exit/failures | `(0,1)`, `(3,0)`, `(0,0)`, `(3,2)` |
| searched signature, any failure | `(3,0)`, `(0,1)`, `(3,1)` |
| telemetry outside the table | e.g. `usealphabeta=true`, `currentMaxPly=3`, incomplete |
| unusable move | null sentinel · `None` · T1j `legal=false` · illegal in **our** engine |
| wrong board | a dump that diverges from the position sent · 0 or 2 dumps |
| incoherent QUERY line | `q ≠ 1` · `moveNr ≠ ply` · wrong `to_move` · requested depth ≠ 6 |
| `MatchData` | missing line · two lines · `pieRule=true` · size ≠ 24 · `ystarts=false` · **`identity=false`** |
| reflection drift | opt-in `refl_n` 3 or 5 |
| `PROC` | 0 lines · 2 lines — on the **query** path **and** the **replay** path |
| dirty safety surface | a throw · windows · frames · not headless · prefs disturbed · `refl_ok=false` |
| H4-mode construction | depth ≠ 6 · query timeout `None` |

### 6.2 Clean baselines — each must ACCEPT, individually

One per accepted classification, **not one aggregate valid record**:

| baseline | must reach |
|---|---|
| ply 0, exit 3 / failures 1, native telemetry | `native_initial_first` |
| plies 1, 2, 3, same | `native_initial_second_to_fourth` |
| 🔴 **plies 4, 5, same — CONSTRUCTED** | **`native_initial_fifth_or_more`** |
| plies 3, 4, 5, exit 0 / failures 0, completed depth 6 | `searched` — also the control that the plies 0–2 rule does not spread |
| an H4-mode replay with exactly one `PROC` | binds, and is recorded |

⚠ **`native_initial_fifth_or_more` has never been observed**, so its baseline is
driven with **constructed telemetry** at the process boundary and says so. It
proves the production path **can** accept that routine's reply; it does not prove
the routine ever answers.

### 6.3 The shared classification — proven, not asserted

* **Identity:** the classifier the adapter calls **`is`**
  `h4_repair_qualification.classify_reply`.
* **Called exactly once** per H4-mode agent call.
* **A spy that ACCEPTS** a reply the real classifier refuses (e.g. exit 3 on
  searched telemetry) → the adapter **accepts**. That proves no site re-judges
  `rc`, `failures` or `completed` on its own.
* **A spy that REFUSES** a clean searched reply → the adapter **refuses**. The
  verdict binds in both directions.

### 6.4 Default callers unchanged — each proven

* a default runtime's query argv is **byte-identical** to before §4B (`query`,
  never `h4query`);
* a default agent **refuses a native reply** at site 1, with the pre-§4B message
  and phase;
* a default agent still expects `refl_n` **3**; a default binder still binds
  **without** a `PROC` line;
* **no existing caller file opts in** — an AST walk over executable code, with a
  clean-baseline control proving the walker sees an opt-in when one is planted;
* the T1j-adjacent regression failure set is **identical** to the recorded
  baseline: the 30 files matching
  `tests/test_*{h1,h2,h3,h4,l0,e4,t1j,gate,control}*.py` under
  `/opt/homebrew/bin/python3` with `--continue-on-collection-errors` — **30
  failing ids**, all pre-existing.

### 6.5 The injected-defect harness

* **Every pre-existing control still matches its anchor and still rejects** —
  count derived by the harness before and after, never hand-typed. A control
  whose anchor no longer matches has **not run**, and a moved line is how that
  happens silently.
* **New controls, appended** — each must be caught by the named test, for its
  stated reason:

| injected defect | caught by |
|---|---|
| the H4 path re-checks `rc != 0` on its own | the native baselines (§6.2) |
| the H4 path judges `PostCond.clean` | the native baselines |
| the H4 path keeps `not r.completed` | the native baselines |
| the H4 path expects `refl_n` 3 | every H4 baseline |
| the H4 path passes a constant exit status to the classifier | the wrong-exit controls (§6.1) |
| the H4 path skips the `identity` check | the `identity=false` control |
| the H4 path skips the `PROC` count | the `PROC` controls, both paths |
| the H4 path queries without `inject_matchdata` | the argv control and the missing-`MATCHDATA` control |
| the switch defaults to `True` | the default-caller controls (§6.4) |
| the classifier is a local copy | the identity control (§6.3) |

### 6.6 The runner

* **Closed gate:** both public entries refuse, and the CLI refuses as a **fresh
  subprocess** — exit 5, no class directory and no record created.
* 🔴 **At least one test drives the runner's PUBLIC entry with the gate
  monkeypatched OPEN**, the process boundary stubbed, to **CLEAN** — and the
  stub answers at least one ply-4/5 position natively, so
  `native_initial_fifth_or_more` is reached **end to end through the production
  adapter** (constructed). A suite that only drives the unguarded body never
  runs the combination that will actually execute.
* The matrix-pin and nesting controls, an occupied destination refused
  **before any subprocess**, a create-only class directory, and a STOP that
  leaves a durable record.

---

## 7. THE QUALIFICATION RUN — frozen

| | |
|---|---|
| positions | the repair card's pinned derived matrix — **16 rows, `3cc14ca9…d22f27cd`** — re-derived and verified by **importing** `derive_matrix` / `verify_matrix`, never copied |
| per position | `ctx.reset(task_id, prefix)`; **one binder call** (a replay JVM); then **agent calls** — 5 at plies 0–3, 1 at plies 4–5 |
| caps | 56 agent calls + 16 binder calls = **72**, **derived** from the matrix |
| agents | from `make_agent_factory`, each built for `state.to_move` — red at plies 0/2/4, black at 1/3/5 |
| depth · ply cap | **6** · `l0_match_rules.PLY_CAP` (280) |
| timeouts | **120 s** per call · **1,800 s** whole run, SIGALRM supervisor |
| lifecycle | a fresh JVM per call; helpers compiled **once**, through the verified compile (jar and JDK pins re-checked) into a **create-only** directory outside the repository; the identity recorded |

**Reported, never judged:** the classification by ply, realized moves, distinct
pids. Either classification at plies 3–5 is legitimate. The absence of
`native_initial_fifth_or_more` stops nothing.

**Execution procedure**, when separately authorized — the one that ran the
repair qualification (`adcd48f`): a pre-run file recomputing every pin; the
**exact command run once with the gate still closed** (it must exit 5 and create
nothing); a gate-open commit that is **the one line alone**; one launch through a
wrapper whose EXIT trap restores the gate from the reviewed commit; evidence
files create-only; the evidence and the restoration committed together.

---

## 8. STOP RULES — frozen before any output is seen

**STOP (exit 2)** — a RESULT about the production adapter, and it ends §4B:

* **any `AbortError` from the production agent or binder at any matrix call**,
  whatever its reason. 🔴 In particular, **a reply refused here that the repair
  qualification established as legitimate at these positions is a STOP about the
  ADAPTER** — the acceptance mode refusing what it must accept — **never a
  finding about T1j**;
* a process record missing for any subprocess, anything other than **72**
  records, or anything other than **72 distinct pids**;
* subprocesses spent ≠ the derived **72**.

**VOID (exit 3)** — the instrument is unreadable: a per-call timeout,
unparseable helper output, a toolchain or compilation failure, the whole-run
deadline.

**Refused before starting**, nothing written: the gate closed (exit 5); a
matrix-pin or nesting failure; an occupied destination.

🔴 **A STOP or VOID ends §4B under this card: no repair, no retry, no
reinterpretation, no second run.** A STOP writes a durable record — the reason,
the failing position, the raw stdout, and the observations completed before it,
marked **not a partial qualification**.

---

## 9. FROZEN NAMES

```text
gate         H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED   (created CLOSED; the FOURTEENTH gate)
runner       scripts/GPU/alphazero/h4_4b_acceptance_qualification.py
tests        tests/test_h4_4b_acceptance_qualification.py
destination  docs/superpowers/evidence/2026-09-22-t1j-h4-4b-acceptance-qualification/
record       05_qualification.json   (claimed O_EXCL before compilation and before any JVM)
```

The gate is read at **both** public entries (`run_qualification` and `main`),
offers **no override** of any kind — not argv, not the environment, not a
configuration file, not an import hook — and reads no other experiment's gate.

---

## 10. What a CLEAN §4B establishes — and what it does not

**Establishes:** at the 16 frozen positions, depth 6, the production adapter in
H4 mode accepts the helper's replies, refuses none of them, names the routine
that answered, and records one `PROC` per JVM; and every default caller is
proven unchanged.

**Does NOT establish:** that real output from `fifthOrMoreMove()` is accepted
(constructed only); anything outside the matrix; anything about games, the H4
runner, the pilot or strength. It may not be pooled with anything.

**A CLEAN §4B authorizes nothing by itself.** Before any pilot, each separately
authorized: the H4 runner (replacement §2, §7), the §1.3 duplicate-rule removal
with its negative control, and a fresh, collision-proved seed block.

---

## 11. What this card does not do

It writes no code, opens no gate, runs no helper, reserves no seed, plays no
game, aggregates nothing, and pushes nothing. It does not edit the replacement
card.

**The next separately authorized action is §4B implementation plus tests, with
the gate created CLOSED and no helper execution.** Execution is a separate
authorization after that; push is separate again.
