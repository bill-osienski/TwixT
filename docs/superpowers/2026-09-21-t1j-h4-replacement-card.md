# H4 REPLACEMENT CARD — empty-board direct play under the frozen argmax agent

**Written 2026-09-20. DESIGN ONLY.** No seeds are chosen or reserved, no code is
written, no gate exists, nothing runs. This card **supersedes
`2026-09-21-t1j-h4-pilot-card.md`**, which is marked VOID in place and is
retained unedited as the correction trail.

> **The question H4 asks.** From an empty board, under the frozen argmax
> incumbent, frozen native T1j behavior, and a preregistered
> execution/randomization protocol, which model has the higher expected
> colour-balanced score?

"Execution/randomization protocol" is deliberate and replaces the superseded
card's "search-seed distribution": **T1j contributes native process-local
entropy that our seed does not control.** §3 enumerates every source.

---

## 0. The correction trail — what the superseded card got wrong

The superseded card's §0 headline was *"under argmax both agents are
deterministic, so an empty board yields 2 distinct games, not 592"*, and its §1
recommended abandoning argmax on that basis. **Every premise failed against this
programme's own records.** Nothing in that card's §0, §1 or §7 option framing may
be cited.

| superseded card claimed | the record says |
|---|---|
| "692 games across 16 groups, 46 each" | 736 scheduled, **692 completed, VOID**. **15** complete cells; 14 held 1 distinct transcript; `o8_contact/t1j_black` held **2** games (`08_accounting_from_records.txt`) |
| "the 274-ply cap" | `PLY_CAP = 280` **total** plies (`l0_match_rules.py:109`). 274 was H3's *post-opening budget*, mislabelled as the cap |
| the 46-distinct capped cell was "noise rather than diversity" | `12_correction.md`: it **CLEARED** the floor of 42. It lacked **outcome** information, not trajectory variation — "distinctness is necessary … not sufficient" |
| "a deterministic argmax readout repeats one game per cell" | **that exact sentence is RETRACTED** by `12_correction.md`: argmax removes the **readout's** randomness, not the **search's**. Why seeded search variation was insufficient there "is not determined by this run" |
| "T1j is deterministic — E3a, 25 queries" | E3a covered **one position at one ply**. At board-plies 1 and 3 the two independent JVMs **disagree in 5 of 12** (depth × prefix) cells; **0 of 6** at ply 5. One `o3_low` ply-1 prefix returned **three** distinct moves across four invocations. The record names the condition: `signature: "mover_fragmentation"` |
| "the entropy lives in the selection mode, not the board" | H1 and H2 differ in readout **and** seed block (`[202617000,…)` vs `[202622000,…)`). H2 card §8 forbids pooling and rate attribution across them; the same confound applies to a diversity comparison by identical construction |

🔑 **The root failure.** The card quoted `08_accounting_from_records.txt` while
`12_correction.md` — retracting that very sentence — sat **four files later in
the same directory**. Evidence here is create-only *by design*, so a wrong record
is never edited in place; it is superseded by a later sibling. Citing file *n*
without reading files *> n* is citing a draft, not the record.

---

## 1. The scientific core — FROZEN before any qualification or pilot result

The pilot (§5) may affect **feasibility and logistics only**. It may not alter
anything in this section.

**✅ FROZEN 2026-09-20.** The three items this card originally left open — δ,
the cap policy and the process lifecycle — are affirmed in §1.5, §1.6 and §3.1,
and three arithmetic/factual corrections were applied at the same time (§1.2's
support, §1.5's N, §1.3's account of what H3 actually dropped). Each correction
is marked in place rather than quietly overwritten.

### 1.1 Estimand

The expected **colour-balanced pair score** for the incumbent, over independent
pair bundles drawn from the §3 execution/randomization protocol, with games
played from the empty board.

### 1.2 The observation is a pair bundle

| | |
|---|---|
| Arm A | incumbent **Red** (moves first), T1j Black |
| Arm B | T1j **Red** (moves first), incumbent Black |
| points (per game) | win 1, loss 0, **cap 0.5** |
| pair score | incumbent's points over the two games ÷ 2 ∈ **{0, 0.25, 0.5, 0.75, 1}** |

⚠ **Corrected 2026-09-20.** An earlier draft wrote the support as {0, 0.5, 1},
which is the cap-free range and contradicted this table's own cap rule one line
above. Two games at {0, 0.5, 1} each give total points {0, 0.5, 1, 1.5, 2},
hence five pair scores. The `[0,1]` Hoeffding bound is unaffected.

🔴 **This is a COLOUR-BALANCED pair, not a within-position pair, and no
common-random-number control is claimed.** H3's pair was two views of one
position. H4's two arms are different games sharing only a bundle index.
Matching nominal seeds across different positions and colours does not create
meaningful CRN coupling, and **T1j's randomness cannot be coupled at all** (§3).
The pairing buys balance of the first-move advantage. That is all it buys, and
any reuse of H3's analysis code must not import the stronger meaning.

### 1.3 Repeated trajectories are RETAINED — unequivocally

* **Every valid pair stays in the primary analysis with its original
  multiplicity.**
* Identical transcripts are **never** excluded, collapsed, or reweighted.
* Repeated trajectories are **scientific observations about probability mass
  under the execution protocol**, not duplicates.
* **No minimum-uniqueness gate determines whether observations count.** H2's
  `MIN_DISTINCT_PER_CELL = 42` has no analogue here and must not be ported.

Integrity faults are a different category and remain fatal: duplicate
`task_id`, a repeated seed assignment, or mutually inconsistent records.

🔴 **INHERITANCE HAZARD — and it is worse than a stale design note.** The
superseded card listed H3's analysis as "directly reusable". It is not.
`h3_study_analysis.py` **drops the observation**:

```
ident = pair_identity(rows)
if ident in seen_identity:
    duplicates += 1
    continue          # ← the pair never reaches `scored`
```

and `pair_identity` is **`(opening_digest, red_digest, black_digest)`**.

**H4 has no opening digest** — the board is empty, so that field is constant
across every pair. The identity therefore **degenerates to pure trajectory
identity**, and a rule that in H3 could only ever have caught a coincidentally
repeated *opening* would in H4 **silently delete every repeated trajectory from
the primary estimate** — precisely the mass this estimand exists to measure.
Under a concentrated protocol it could discard most of the sample.

⚠ **What H3 actually did, stated exactly.** The rule was **capable** of dropping
observations; it **dropped none**. Every H3 report records `duplicate_pairs: 0`
(`09_combined_report.json`, and each segment's `09_report.json`). An earlier
draft implied H3 had dropped "a handful" — it did not, and the hazard is
entirely prospective. That is what makes it dangerous: the rule is inert in H3's
records and would become load-bearing on H4's first concentrated run.

**The rule must be removed, and its removal proven with a negative control**
that feeds identical pairs through the estimator and asserts they are all
retained with their multiplicity. H3's `MAX_DUPLICATE_PAIRS` ceiling must not be
ported either: in H4 a repeated trajectory is **data**, not a harness fault.

### 1.4 A fully collapsed distribution is a RESULT

If the protocol places most of its mass on a few trajectories, that is the
finding, reported as such. It is **not** grounds for retrospectively replacing
`n` with the number of unique transcripts, and it does not invalidate the
interval: a Hoeffding bound over *n* independent bounded draws is valid
regardless of how concentrated the draws turn out to be.

### 1.5 Precision and the decision threshold — ✅ AFFIRMED δ = 0.08

The half-width is **derived from the smallest advantage worth acting on (δ)**,
not inherited because H3 used it.

**✅ δ = 0.08, affirmed 2026-09-20.** Its justification for H4 is its own:
H4 declares a winner only for a **reasonably substantial direct-play
difference**, rather than treating a modest edge as actionable. With δ = 0.08 an
observed mean ≥ 0.58 or ≤ 0.42 yields an interval excluding parity.

Two-sided nominal 95% Hoeffding on pair scores bounded in [0,1]:
`h(n) = sqrt(ln(2/0.05) / 2n)`.

🔴 **δ ALONE GIVES N ≥ 289, NOT 296.** Recomputed:

| n | h(n) | |
|---:|---|---|
| 288 | 0.0800269 | **too wide** |
| **289** | 0.0798883 | the true minimum from δ |
| **296** | 0.078938 | requires a **second, stated** constraint |

An earlier draft said "N = 296 follows from that δ and nothing else." **That was
false.** 296 is reachable only with an additional operational constraint, and it
must be stated rather than smuggled in:

* **296 = 4 × 74 pairs = 4 segments × 148 games**, which is exactly H3's proven
  segment shape.
* **Freeze the four equal segments and N = 296 follows.** Until the operational
  segmentation is chosen, the card carries **N ≥ 289** and nothing tighter.

### 1.6 Cap policy — ✅ AFFIRMED, 280 total plies

Caps score 0.5, so the cap is part of the estimand, not a runtime detail.

**✅ `PLY_CAP = 280` total plies, affirmed 2026-09-20.** H4 uses the
**established total-game boundary**. H3's games spent 6 of that on the opening
and had 274 plies of play; an H4 game from the empty board gets the full 280.

🔑 **The asymmetry is deliberate and is the right way round.** Reducing H4 to
274 merely to equal H3's post-opening continuation budget would **truncate a
complete game six plies earlier** for no scientific reason. The boundary is a
property of a whole game, not of a continuation.

Caps score **0.5** in the primary; the **cap-free sensitivity is retained**
beside it. The cap *rate* is a §5.4 feasibility rule, never a validity gate.

### 1.7 Forbidden interpretations — frozen now, before any number exists

* **No pooling** with H1, H2, H3 or L0 — different population, different
  protocol, different question.
* **No carry-over of H3's verdict.** H4 is a new study. H3's 0.7348 is not
  evidence for or against anything here.
* **No causal attribution to opening selection** from a T1j win (§9).
* **No claim about human play**, about any other configuration, or about either
  engine outside this protocol.
* **No sizing of the full study from pilot outcomes** (§5).

---

## 2. What is unchanged from the frozen agent, read from source

Named so the card and the code cannot drift apart. Every value below is read,
never retyped from memory:

| constant | value | source |
|---|---|---|
| `SELECTION_MODE` | `"argmax"` | `h2_match_rules.py:52` |
| `MCTS_SIMS` | `400` | `h2_match_rules.py:57` |
| `T1J_MDPLY` | `6` (T1j **search depth**, with `mdFixedPly=True`) | `l0_match_rules.py` |
| `PLY_CAP` | `280` total plies | `l0_match_rules.py:109` |
| reference | name + sha1, as H3 records them | `h3_study_rules.py` |

⚠ **Two different sixes.** `T1J_MDPLY = 6` is T1j's requested search depth.
H3's `OPENING_PLIES = 6` is a board-ply count. They are unrelated and the card
must never let one stand for the other.

**What is genuinely new to build** (not built here): the empty-board runner path
(`opening_bound = 0`), the adapter acceptance path of §4, full move persistence,
and the replay export.

---

## 3. The randomization protocol — every source named

A source that is not listed here is a source that was not controlled.

| source | controlled by us? | recorded? |
|---|---|---|
| incumbent search stream, Arm A | **yes** — distinct, domain-separated per arm | yes, via seed bundle |
| incumbent search stream, Arm B | **yes** — distinct, domain-separated per arm | yes, via seed bundle |
| **T1j native process-local entropy** | 🔴 **NO** | **no — we neither set nor observe its stream** |
| T1j JVM/process lifecycle (fresh vs reused, per game or per query) | **yes** — one frozen protocol, declared | yes |
| coupling between the two arms | **none claimed** (§1.2) | n/a |
| opening position | none — the board is empty by construction | n/a |

🔴 **Consequence, stated plainly: an H4 game is NOT reproducible from our seed
alone.** T1j's low-ply path returned different moves across independent JVMs in
5 of 12 measured cells. This is why §7's complete move persistence is
load-bearing rather than hygienic: **the persisted move list is the
reproducibility record**, not the seed.

### 3.1 ✅ The process lifecycle — AFFIRMED, and it is the EXISTING production path

Frozen 2026-09-20. **This is not a new design**: it is what the production path
already does, read from source rather than described from memory.

| | frozen behaviour | source |
|---|---|---|
| agent object | one Python `T1jAgent` **per T1j colour per game** | `e4_screen_integration.py`, `make_agent_factory` → `agent_factory(task, mover)` |
| search | a **fresh JVM for every T1j move**, reconstructing the full history | `T1jAgent.__call__` → `A.query(...)` → `subprocess.run` (`t1j_adapter.py:391`) |
| replay/bind | a **separate fresh JVM for every binder call** — the opening plus every completed ply | `make_binder` → `A.replay(...)` → `subprocess.run` (`t1j_adapter.py:261`) |
| reuse | **none** — no JVM is reused across moves, games, arms or pairs | follows from the two above |
| helper classes | **compiled once per run** | `compile_helper()` (`t1j_adapter.py:202`) |

🔑 **Why this lifecycle and not a cheaper one.** It is expensive — two JVM
spawns per ply in the worst case. Moving to a persistent JVM **would change
T1j's realized behaviour and its entropy source**, which is the very thing under
measurement. The §4 qualification and H4 itself use **this exact lifecycle**;
qualifying one lifecycle and running another qualifies nothing.

### 3.2 The lifecycle must be TESTABLE, not merely described

Asserted from recorded counts, not from reading the code:

* **search-process count == number of T1j moves**;
* **replay-process count == final ply count + 1** — the binder runs once for the
  opening (`move=None`) and once per applied ply, and already asserts
  `len(plies) == state.ply + 1` internally;
* every subprocess **emits and records its `PROC` identity and ordinal**;
* **no process survives its call**;
* query telemetry records the **fallback/search classification** (§4.2.5);
* any **Zobrist fingerprint is observational only** and must not alter
  initialization — if recording it would perturb init, it is not recorded, and
  §7's statement about non-reproducibility stands instead.

---

## 4. STAGE 1 — Adapter qualification, separately reported

**This is a harness qualification. It produces NO strength evidence and no part
of it may be read as one.** It is reported on its own, before any pilot.

### 4.1 Why it is required, with the mechanism named

T1j **does** return a legal move at low ply. **The qualified adapter path
refuses to deliver it.**

🔴 **THERE ARE TWO REFUSAL POINTS, NOT ONE**, and both are in
`T1jAgent.__call__` (`e4_screen_integration.py`). A change addressing only the
first would still abort:

1. **`if rc != 0 or len(recs) != 1:`** — the helper's own exit. A non-completing
   search sets `exit_status: 3`, and `E4Preflight` exits 3 exactly when
   `req(completed, …)` fails.
2. **`if not r.completed or r.requested_depth != self.depth:`** — the adapter's
   **own** completion check, which raises `AbortError` **even at `rc == 0`**.

This is not theoretical. **H3 generation attempt 2 VOIDed at board-ply 1** on
`FAIL q1: requested depth 6 completed … failures=1`, spending a seed range and a
destination. Every H4 game passes through board-plies 1 and 3 in **every**
configuration, so this blocks H4 outright until it is deliberately changed.

🔑 **The change is narrower than "an acceptance path" suggests.** Every other
guard in `__call__` already enforces something §4.2 requires and **must stay
exactly as it is**: `compare_state(state, dumps[0], …)` re-binds the searched
position, `r.null_sentinel / r.move is None / not r.legal` rejects unusable
moves, `r.move not in state.legal_moves()` rejects moves illegal in our engine,
and `check_postcond` enforces the postcondition surface. **Only the completion
condition is relaxed, and only under the qualified signature.**

⚠ The H3 full-study design's heading "T1j CANNOT MOVE AT THE PLIES THE PROTOCOL
NEEDS" is loose; its body is accurate ("never enters alpha-beta", "known not to
search"). The precise statement is: **the engine moves, the adapter refuses.**

### 4.2 What the qualification must establish

**An explicit H4 acceptance mode whose default remains OFF for existing
callers.** A default that switches the guard off is the defect class this
programme keeps hitting; the mode is opt-in, and every current caller —
E4, L0, H1, H2, H3, D1 — keeps the present fail-closed behaviour untouched and
is proven to.

The adapter must distinguish a **legitimate native fallback** from a **broken
search**:

1. coverage of **both T1j colour roles** and the board states it will encounter
   at **board-plies 0–5**;
2. the **same JVM/process lifecycle** intended for H4 (§3);
3. a **legal, non-null** move returned;
4. **mover, board and postcondition coherence** — the returned move is legal for
   the side to move on the board actually sent;
5. explicit telemetry distinguishing **`searched`** from
   **`native_low_ply_fallback`**;
6. acceptance **only** for the qualified fallback signature;
7. **continued fail-closed behavior for genuine incomplete searches elsewhere.**

### 4.3 Negative controls — the acceptance path must be proven to REJECT

Requirement 7 is the one that fails silently if it is only asserted. The
qualification must show rejection of at least: a non-completing search **outside**
the qualified low-ply signature; an illegal or null move; a move legal on a
different board than the one sent; a postcondition surface that is not clean;
and a fallback claimed at a ply where search is known to complete (ply 5).

A check that greps source is not a test. The acceptance predicate must be
driven through its **real entry point** in the state it will run in, and the CLI
must be qualified as a **fresh subprocess**.

### 4.4 The empty board is genuinely unqualified

The lowest observed prefix in the low-ply record is **one stone**
(`"ply": 1, "prefix": [[11,11]]`). The empty board was never queried.

* **Arm A** — incumbent moves first, T1j answers at 1 stone: **observed**.
* **Arm B** — T1j moves first from **0 stones**: **not observed**.

The qualification must cover ply 0 explicitly. If T1j cannot produce an
acceptable move there, **Arm B does not exist in this form** and H4 must be
redesigned rather than patched.

---

## 5. STAGE 2 — One small, outcome-blinded feasibility pilot

**Pilot games NEVER enter the confirmatory estimate.**

### 5.1 Size — ✅ 16 complete pairs / 32 games

Frozen 2026-09-20, chosen from runtime and qualification needs — **never from
score precision**. It is large enough to exercise **both arms and the per-move
JVM lifecycle repeatedly**, and **clearly too small to masquerade as strength
evidence** (at n=16 the nominal Hoeffding half-width is ≈0.34, which settles
nothing and is not to be computed anyway, per §5.3).

### 5.2 Purpose — feasibility only

Replay integrity · runtime distribution · cap rate · process behavior ·
empirical trajectory concentration (§8).

### 5.3 The blind, and how it is actually enforced

Per-game `winner` **is persisted** to the durable record — integrity requires
it. The blind is on the **decision-maker**, not the disk:

* the **feasibility report contains no winner, no score, no per-arm split and no
  interval** — it reports `terminal_reason` categories (natural vs cap) only,
  which is what cap feasibility needs and carries no competitive outcome;
* the aggregation step is **not authorized** until the proceed/stop decision is
  **committed durably**;
* proceeding to the full study does not release the pilot into it.

### 5.4 Preregistered before launch

* **integrity failures: ZERO TOLERANCE.** Complete move persistence and replay
  verification are pass/fail at zero. This is a different category from the
  operational rate below and is not traded against it.
* a **maximum operational failure rate** (launch/timeout/process faults), stated
  as a number before launch;
* runtime and cap feasibility rules;
* trajectory concentration is **reported**, and may inform the **economic**
  proceed/stop judgment — whether the full study is worth its cost. It may
  **never** invalidate, deduplicate or reweight an observation (§1.3);
* that winner-based score, model advantage and confidence intervals **will not
  guide the full design**.

### 5.5 No two-game "sanity run"

The superseded card proposed one. It is **withdrawn**: seeing those outcomes
before the design is frozen is avoidable sequential-design contamination, and
§4's adapter qualification already serves the legitimate operational purpose
without revealing a competitive outcome.

---

## 6. STAGE 3 — The full study

Proceeds **only** if the §5.4 preregistered feasibility rules clear. If
feasibility fails, that is reported and the design revisited — **the study is
never shrunk and thresholds are never relaxed after seeing results.**

| | |
|---|---|
| size | **N ≥ 289** from δ alone; **296 pair bundles / 592 games** once four equal 74-pair segments are frozen (§1.5) |
| interval | two-sided **nominal** 95% Hoeffding — h(289) = **0.0798883**, h(296) = **0.078938** |
| caps | score **0.5**; a **cap-free** sensitivity reported beside the primary |
| seeds | a **fresh, collision-proved** block, ACCOUNTED before use |
| execution | segmented, one-shot per segment, supervised gate restoration |

🔴 **The interval is nominal under a declared independence model, and the model
must now declare BOTH**: independent seed bundles **and** isolated T1j process
realizations. The design does not establish that model, and no screen tests it.

Reusable from H3 once §1.3's collapse rule is removed and the removal is
controlled: the segment runner and per-segment seed isolation, the supervised
wrapper with unconditional gate restoration, create-only destinations with
atomic durable installs, the seed registries and collision proof,
`combine_segments`, and the final-state cross-check.

---

## 7. Persistence — one canonical record, viewer files derived

The evidence record is the **authority**; every viewer artifact is derived from
it and hash-bound to it.

Per game:

* **complete ordered move list**;
* **player and frozen configuration identity by colour**;
* **pair, arm and seed-bundle identity**;
* **per move**: actor, elapsed time, `searched` vs `native_low_ply_fallback`,
  and the relevant T1j telemetry;
* `winner`, `terminal_reason`, total timing, task result;
* **transcript digest recomputed from the moves plus the terminal state** —
  never read back from its own field;
* **replay verification**: mover parity, move legality, natural-win detection,
  and cap termination;
* **hash-bound viewer-compatible replay export** carrying its own
  `evidence_note`.

**T1j state fingerprint.** If a digest or fingerprint of T1j's initialized
Zobrist state can be recorded **without changing its behavior**, record it.
Otherwise the card states plainly: **exact regeneration from the incumbent seed
is impossible, and the persisted moves are the reproducibility record.**

This corrects H3, which persisted **no moves**. Read from
`h3_study_runner.py`: the `transcript` record carries `task_id`, `n_plies` and
`opening_bound`; the `task_result` record carries digests, `winner`, `plies` and
timing. **Neither carries a move list**, so no H3 game can ever be replayed and
no H3 digest can be recomputed from the play it claims to summarise.

---

## 8. Concentration reporting — descriptive, never a gate

Reported for both the pilot and the full study, and never used to include or
exclude an observation:

* unique complete transcripts, by arm;
* maximum transcript frequency;
* first-move and first-*k*-ply prefix frequencies;
* unique outcome patterns;
* frequency of identical **pair-level trajectory tuples**;
* **fallback-versus-search frequency by ply and colour** — this one also reads
  directly on §4's qualification holding up in production.

---

## 9. Interpretation limits — frozen with the design

| outcome | what may be said |
|---|---|
| **incumbent stronger** | stronger in empty-board direct play **under this joint execution protocol** |
| **T1j stronger** | T1j is stronger under this protocol. 🔴 **H4 alone cannot localize the cause** to opening selection or early planning |
| **near parity** | direct play is unresolved or near parity. It does **not** explain why H3 differed |
| **high cap rate** | the protocol often fails to resolve games |
| **high trajectory concentration** | the execution distribution places substantial mass on a small number of games. It does **not** reduce the formal sample to the number of unique transcripts |

⚠ In Arm B, T1j moves first at exactly the plies where it is measured not to
search. A T1j loss may reflect its unsearched opening play rather than general
weakness, and **this design cannot separate the two.** That is the mirror of the
reading that a T1j win would implicate our opening selection, and it belongs in
the claim either way.

---

## 10. What this card does not do

It reserves no seeds, writes no code, opens no gate and runs nothing. It does
not qualify the adapter, does not establish that T1j can move from an empty
board, and does not carry H3's verdict across.

**Temperature play and *k*-ply temperature sampling are OUT OF SCOPE.** They
define a *different agent* and a *different question*, and are recorded here as
separate future hypotheses — **not** as repairs to H4.

### ✅ The three open items are RESOLVED — the scientific core is FROZEN

Affirmed 2026-09-20, after the three corrections above were applied:

1. ✅ **δ = 0.08** (§1.5) — with **N ≥ 289**, and 296 only alongside the frozen
   four-segment shape.
2. ✅ **`PLY_CAP = 280` total plies** (§1.6) — caps 0.5 in the primary,
   cap-free sensitivity retained.
3. ✅ **The existing production process lifecycle** (§3.1) — per-move search
   JVM, per-bind replay JVM, no reuse, classes compiled once per run — with the
   §3.2 assertions making it testable.

### What is still owed before implementation

* the **four-segment operational shape**, if N = 296 rather than 289 (§1.5);
* the preregistered **numbers** for §5.4's operational failure rate and the
  runtime/cap feasibility rules;
* the **§4 adapter qualification itself**, which must run and be reported before
  any pilot.

**No implementation begins until those are recorded.**
