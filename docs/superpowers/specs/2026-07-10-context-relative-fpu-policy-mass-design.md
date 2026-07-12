# Context-Relative FPU (Parent-Relative + Explored-Policy-Mass) — Design

**Status:** APPROVED design, protocol frozen (2026-07-10). Tooling-only branch. The heavy MCTS phases (geometry scan, corpus generation, coefficient sweep, selection, frozen-split check, held-out validation) are **operator phases** and are NOT run by this implementation.

## 0. Objective & context

v16a rejected an **absolute** `fpu_value = -0.20` as a global 400-sim setting (late new-collapse 15.48%; postmortem: a fixed floor commits flat-prior, near-even, high-branching roots to a network-dispreferred move — 12/15 collapses onto priors ranked #16–#207). See `[[a-signal-search-artifact-fpu]]` and `docs/superpowers/specs/2026-07-10-v16a-fpu-reject-postmortem.md`.

The successor makes the FPU reduction **context-relative** so it never parks a constant penalty near the value of contested positions. A constant-reduction parent-relative rule (`Q_parent − const`) is *also* rejected (at a near-even node it recreates the absolute floor). The reduction must be scaled by search/policy geometry. **This is a reject-unsafe-candidates rung, not the strength benchmark;** the decisive endpoint remains a same-checkpoint, same-400-sim, balanced-color, statistically-significant strength gain.

## 1. §A — The rule

At the single unvisited/pending-child site in `_select_child` (`mcts.py`, currently `q = self.config.fpu_value`), when the policy-mass mode is enabled:

```
P_explored = Σ prior(a)  over parent children with a COMPLETED (backed-up) visit   # one pass per _select_child call
FPU        = Q_parent − r · sqrt( clamp(P_explored, 0, 1) )
```

- `Q_parent` = parent's value in the **mover's perspective** (`node.q_value`; `q = -child.q_value` confirms each node's `q_value` is its own to-move perspective; `assert root.q_value == root_value_stm` confirms it for the root). FPU is in the mover's perspective, matching `fpu_value`.
- `P_explored` uses the normalized legal-move priors on the node (`node.priors`); clamped `[0,1]`.
- `r` (single coefficient) = reduction at full explored mass; **finite and ≥ 0**.

### Completed-visits-only (safeguard 6a)

`P_explored` counts a child only when `child.visit_count > 0` is a completed, backed-up visit — never virtual/pending. Confirmed in current code: `_select_child`'s virtual penalty is a **local** var (`child_visits += pending_virtual_visits`) and does not mutate `child.visit_count`; pending leaves aren't backed up until eval completes, so their `visit_count` stays 0. A test asserts a pending/virtual child contributes 0 to `P_explored`; if a future batching change folds virtual/pending into `visit_count`, `P_explored` must switch to an explicit completed-visit indicator.

### Opt-in field + mutual-exclusion guard

```python
fpu_policy_mass_reduction: float | None = None
```
- `None` (default) → existing `fpu_value` path, **byte-identical**.
- Not `None` → policy-mass formula. **`0.0` is an enabled mode** (`FPU = Q_parent`) — not equivalent to `None`; tested.

Fail-loud at `MCTSConfig` construction: raise if (`fpu_policy_mass_reduction is not None` and `fpu_value != 0.0`); raise if (`fpu_policy_mass_reduction is not None` and not finite or `< 0`).

### Pure helper (safeguard 6b)

```python
def policy_mass_fpu(parent_q: float, explored_mass: float, r: float) -> float:
    if not (math.isfinite(parent_q) and math.isfinite(explored_mass) and math.isfinite(r)):
        raise ValueError("policy_mass_fpu requires finite inputs")   # NaN would pass both clamp comparisons
    m = 0.0 if explored_mass < 0.0 else (1.0 if explored_mass > 1.0 else explored_mass)
    return parent_q - r * math.sqrt(m)
```

### Byte-identical-off proof

Full existing suite passes unchanged; plus an `old==new` selection-trace check on a fixed synthetic tree with both new features disabled (`fpu_policy_mass_reduction=None`, observer `None`).

## 2. §B — Development corpus (geometry, not failure-selected)

**Source:** `logs/eval/0379_vs_calib020_0001_800g_w4_seed20116_replay_games.jsonl` (same matchup, different games/seed). Membership decided ONLY by the fpu-off `calib020_0001` anchor + raw policy — never any candidate-FPU result.

### Bounded two-stage scan (edit 3)

1. **Ply enumeration (cheap, stored replay records):** per game (ascending `game_idx`), among plies with stored `n_legal ≥ 200`, take **the 1st, 5th, 9th, … qualifying ply in ascending ply order** ("every fourth eligible ply"), capped at **6 pre-anchor candidates per game**.
2. **Raw-policy pre-filter (cheap, our net):** reconstruct via trusted `position_state`, one raw-policy forward pass; keep target (`normalized_entropy = H(prior)/log(n_legal) ≥ 0.90` **and** `top1_prior ≤ 0.025`) or control (concentrated: `normalized_entropy < 0.85` **or** `top1_prior ≥ 0.05`) candidates.
3. **Anchor confirm (expensive, survivors only):** 400-sim fpu-off anchor only on raw-policy survivors; keep `|root_value_stm| ≤ 0.25`.
4. Continue deterministically until each band quota is filled **plus a ≥2× reserve pool**.

Recorded per row: `source_corpus_id, game_idx, position_ply, side, game_result, total_plies, n_legal, root_value_stm, normalized_entropy, top1_prior, top4_mass, top8_mass, canonical_position_sha1, ply_bucket, branching_band, split, role(target|control)`.

**Target eligibility:** `n_legal ≥ 200 ∧ |root_value_stm| ≤ 0.25 ∧ normalized_entropy ≥ 0.90 ∧ top1_prior ≤ 0.025`.

**Ply buckets** (for the ≤50% cap, reused from v16a): opening 1–15, early_mid 16–40, midgame 41–90, late 91+. **Branching bands:** 200–299 / 300–399 / 400+.

**Composition — 240 = 180 target + 60 controls:**
- Target: 60 per branching band.
- Controls: 20 per branching band (matched branching, concentrated policy).

Sampling: game-first round-robin; ≤ 2 per game; ≥ 12-ply separation within a game; ~50/50 side; cap any ply bucket ≤ 50%; include state-cap/unknown games; deterministic under a fixed seed.

### 2.2 Split — hard-isolated, deterministic allocation (edits 1, 4)

Split **by whole game** (a game's positions never straddle splits). Every row carries `split ∈ {tuning, frozen_check}`. Target allocation (goal; whole-game constraint is hard, deterministic greedy matches it, any deviation is logged):

| | 200–299 | 300–399 | 400+ | total |
|---|---:|---:|---:|---:|
| target · tuning | 40 | 40 | 40 | 120 |
| target · frozen | 20 | 20 | 20 | 60 |
| control · tuning | 13 | 13 | 14 | 40 |
| control · frozen | 7 | 7 | 6 | 20 |

`tuning` = 160 (120 target + 40 control); `frozen_check` = 80 (60 + 20). Coefficient selection uses only `tuning`; `frozen_check` runs once after `r` is frozen, never retuned.

### 2.3 Complete-state canonical hash + disjointness (edits 4, 5)

`canonical_position_sha1 = sha1(canonical_serialization(state))` over the **complete future-play-relevant game state**, not merely the visible board — include everything that can change legality or evaluation: board size, red-peg set, black-peg set, side-to-move, terminal status, and any rule-state flag not derivable from pegs+side (swap/pie availability if supported, move-count-dependent rule state). Exclude caches and move history that do not affect future play. A move-prefix hash is insufficient (transpositions). **Test:** any two states with equal hash have identical side-to-move, legal-move set, terminal result, and network-input representation (this test is the completeness guarantee; the implementer enumerates the exact `TwixtState` fields it requires).

**Collision handling (edit 4):** during candidate/reserve construction, **discard** any position whose hash collides with selected-A, the consumed v16a positions, or an already-kept dev row, and continue deterministically (the reserve absorbs these drops). On the completed 240-row manifest, **assert** zero cross-corpus collisions (vs selected-A ∪ v16a) and zero internal duplicate hashes; fail if violated.

## 3. §C — Discovery diagnostic + modes

**Configurations:** `absolute_off` (`None`, shipped default — byte-identical ref + **production baseline**); `r = 0.0` (parent-relative control, `FPU = Q_parent`); candidates `r ∈ {0.10, 0.20, 0.35, 0.50, 0.75}`.

### Reference binding — dual reference (edit 1)

`r = 0.0` is **not** production. Every paired metric is reported against **both** references. Safety is gated against **both**:

- **Control-qualification (prerequisite):** before any nonzero `r` is run or interpreted, `r = 0.0` must pass the **complete §6.2 development-safety table relative to `absolute_off`**. If `r = 0.0` fails, **reject the parent-relative formula family** — do not run/interpret the nonzero grid.
- **Per-candidate:** each nonzero `r` reports (a) incremental effect vs `r = 0.0` and (b) **total effect vs `absolute_off`**, and must pass new-collapse, low-prior lock-in, and matched-control harm against **both** references (§6.2).
- **Selected-A gate:** vs `absolute_off` (production) — §6.3.
- **Byte-identity:** `absolute_off`/observer-off vs pre-branch — §6.4.

### Enforced modes (edit 1)

```
--mode tuning:        reads ONLY split==tuning rows (+ selected-A)
                      permits: absolute_off, r=0.0, and the five-r grid; A participates in the A gate
--mode frozen_check:  reads ONLY split==frozen_check rows
                      permits: absolute_off, r=0.0, and EXACTLY ONE nonzero (frozen) r
                      rejects a grid or >1 nonzero candidate; excludes selected-A from pass/fail
```
The script fails loudly if a mode is handed rows from the wrong split (validates every row's `split` against `--mode`).

**Per-position metrics** (paired, both references): selected-move prior + prior rank; effective-children Δ; opponent-reply-count Δ; top-share Δ; root-value (mover) Δ; new/resolved collapse; top-move flip; plus the trace dynamics (§4). **Central check:** does the candidate commit early to a very low-prior move while most policy mass is unexplored (lock-in event, §6.1)?

## 4. Trace collector — exact, per-completed-simulation (edit 4 + edge case)

Optional observer on the single real 400-sim run. Interface:

```python
observer.on_root_simulation(completed_simulation_count: int,
                            root,
                            updated_root_move: int | None,
                            current_root_leader_move: int | None)
```

Fires **once per completed root-simulation backup** (once per backed-up leaf, in backup order — not once per batch flush covering several sims), passing the monotonic completed count, the root move on that simulation's path, and the canonical visit-leader move for **incremental** updates (no 200–500-child rescan/sim).

- **Edge case:** the first completed simulation may expand/back up the root without traversing a root move → `updated_root_move` may be `None`; the observer ignores `None` for first-visit/mass bookkeeping but still advances the completed-simulation counter.
- `completed_simulation_count` counts completed sims, never attempted/pending; explored mass excludes virtual/pending visits.
- Leader determination **reuses the actual MCTS leader helper/comparator** (the `_best_child` max-visit, lowest-encoded-move tie-break) rather than duplicating tie-break logic.
- Records: first-visit sim per root move; explored mass when the eventual final top move is first visited; leader-change timeline; final leader's **last-takeover** sim (stabilization); first sim crossing 25%/50%/75% explored mass; explored mass when the final move first becomes leader; selected move's prior + rank.
- **Isolation:** defaults `None`; no effect on selection/eval/RNG/batching/backup; `mcts.py` holds only the optional reference + the single guarded callback; observer logic lives in the diagnostic module; an observer exception **aborts the diagnostic**. Observer-off is part of the byte-identical proof.

## 5. §D — Pre-registered coefficient-selection protocol

0. Run `absolute_off` and `r = 0.0` on the tuning split. **`r = 0.0` must pass the full §6.2 table vs `absolute_off`** (control-qualification). If it fails → reject the formula family; stop. Record `r0_target_lockin_count` and `absoff_target_lockin_count` (§6.1) — the only values learned from the control runs; substitute into the already-frozen caps.
1. For each nonzero `r`: reject if it fails any §6.2 gate vs **either** reference on the 160 tuning split or its controls.
2. Among survivors, require the §6.3 selected-A gate (vs `absolute_off`).
3. Choose the **smallest `r`** satisfying both.
4. If none pass → reject the formula. Do not interpolate, extend the grid, or inspect `frozen_check`.

## 6. Pre-registered NUMERIC gates (edits 2, 3) — FROZEN

**All thresholds below are frozen now** (committed before any nonzero-`r` result). The **only** values learned from the control run are `r0_target_lockin_count` and `absoff_target_lockin_count`, substituted into the already-frozen `baseline + 2` lock-in caps. No threshold may be edited after any control result is observed.

### 6.0 Definitions

- **prior rank** of move `a`: `rank(a) = 1 + |{legal b : prior(b) > prior(a)}|` (strictly-greater count; encoded move id orders exact ties only).
- **root top share**: `top_child.visit_count / root.visit_count` (the v16a quantity).
- **selected-A progress**: with `V_off` = absolute_off A mean, `V_ref = −0.0451` (exact 6400-sim reference mean), `V_r` = candidate A mean: `progress = (V_off − V_r) / (V_off − V_ref)`.
- **reply reduction** (config `X` vs reference): `1 − replies_X / replies_ref`.

### 6.1 Early low-prior lock-in event (per target position, boolean)

Qualifies iff **all**:
```
selected_move_prior_rank > 10
selected_move_prior      < 0.01
explored_mass_at_final_leader_stabilization < 0.25
final_leader_stabilization_sim              ≤ 100
final_root_top_share     ≥ 0.90
```
`r0_target_lockin_count` / `absoff_target_lockin_count` = the count of these events on the tuning target set under `r=0.0` / `absolute_off`.

### 6.2 Development-safety gates — evaluated vs BOTH references (120 target + 40 controls, tuning)

For reference `X ∈ {r0, absolute_off}`, REJECT the coefficient if any:
- **New-collapse:** target new-collapse-vs-`X` rate ≥ **5%**, OR any branching band (n ≥ 20) ≥ **10%**.
- **Lock-in:** target lock-in count > `Xtarget_lockin_count + 2`.
- **Broad distortion:** target p95 `|mover-value Δ vs X|` ≥ **0.35** (inspect ≥ 0.20).
- **Compound concentration:** mean effective-children reduction vs `X` ≥ **50%** AND mean top-share increase vs `X` ≥ **+0.15**.
- **Control harm (40 controls):** top-move-flip-to-lower-prior rate ≥ **10%**, OR control p95 `|mover-value Δ vs X|` ≥ **0.35**.

(`r=0.0` is gated with `X = absolute_off` as the control-qualification prerequisite, §5 step 0.) Top-move flip rate on target is **reported** (inspection), not a standalone reject — flips can be the intended correction.

### 6.3 Selected-A mechanism gate (vs `absolute_off`, 30 roots) — REQUIRE all

- `progress ≥ 0.50` (§6.0).
- reply reduction `1 − replies_r/replies_off ≥ 0.50`.
- new-collapse count on A ≤ **2**.
- mean top-share increase on A ≤ **+0.15**.

### 6.4 Byte-identity

`absolute_off` (config `None`) + observer-off reproduce the pre-branch code byte-for-byte; full repository suite green.

## 7. Branch scope (mirrors v16a)

**Ships (verified):** `MCTSConfig.fpu_policy_mass_reduction` + validation + guard; `_select_child` policy-mass branch + one-pass completed-visit `P_explored`; pure `policy_mass_fpu` helper; optional read-only per-completed-simulation trace hook + observer (reusing the MCTS leader helper); geometry-corpus builder (pure sampler + two-stage confirm-under-anchor + complete-state hashing + disjointness); discovery diagnostic with enforced tuning/frozen_check modes + dual-reference metrics/gates + schemas; unit tests (fakes/synthetic trees, no MCTS); `old==new` trace proof with both features disabled; full repository suite; this design note.

**Does NOT run (operator phases):** seed20116 anchor geometry scan; final corpus generation; 5-coefficient sweep; coefficient selection; frozen 80-position check; cross-matchup / fresh held-out validation.

## 8. File structure

- **Modify** `scripts/GPU/alphazero/mcts.py` — `MCTSConfig` field + validation; `_select_child` policy-mass branch + one-pass `P_explored`; single guarded optional `on_root_simulation` callback; `policy_mass_fpu` helper; expose/share the leader comparator for observer reuse.
- **Create** `scripts/GPU/alphazero/build_fpu_dev_corpus.py` — pure sampler (round-robin/cap/gap/band-quota/split-allocation) + two-stage confirm-under-anchor shell + complete-state hashing + disjointness (operator-run).
- **Create** `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — discovery diagnostic (mode-enforced; dual-reference; selected-A + dev corpus × configs), trace observer, metric/gate + schema emission (operator-run).
- **Create** tests: `tests/test_fpu_policy_mass_rule.py` (helper incl. nonfinite reject; guard; `0.0`≠`None`; completed-visit-only `P_explored`; byte-identical selection trace), `tests/test_fpu_dev_corpus.py` (determinism/round-robin/band+split allocation/eligibility/controls/complete-state hash equivalence + disjointness + collision-discard), `tests/test_fpu_trace_observer.py` (per-completed-sim events incl. `None` root-move edge case, leader timeline/stabilization via the shared comparator, mass thresholds), `tests/test_fpu_diagnostic_modes.py` (tuning/frozen_check isolation + config-set validation + wrong-split failure + dual-reference numeric gates incl. progress/reply/rank/top-share formulas).
- **Create** this design note.

## 9. Non-goals / do-not-change

No self-play adoption: `SIMS_TABLE`, `self_play.py`, trainer, network, promotion, calibration manifests, value-adapter/projection untouched. Adoption is a later commitment gated on the strength match. The v16a manifest/outputs remain frozen and are not used here.

## 10. Progression

```
seed20116 geometry-confirmed dev corpus (complete-state disjointness-proven)
  → run absolute_off + r=0.0 ; r=0.0 must qualify vs absolute_off
  → 5-coefficient sweep on the 160 tuning split (dual-reference gates) → smallest-safe-passing r → FREEZE
  → 80-position frozen_check (once)
  → pooled cross-matchup robustness → fresh game-held-out safety corpus (new decision table)
  → selected-A diagnostic + B/C/D guardrails
  → same-checkpoint, same-400-sim, balanced-color strength match  ← decisive
  → (only then) controlled self-play pilot
```

## 11. AMENDMENT (2026-07-11) — seed20116 retired as the dev-corpus source; feasibility preflight added

The first real operator corpus build (`build_fpu_dev_corpus` on `0379_vs_calib020_0001_800g_w4_seed20116`) hard-stopped at `assign_split: cell ('target','b200_299') capacity 0 < demand 60`. Read-only geometry analysis (stored/reconstructed `n_legal`, no evaluator) established the following. **These findings are frozen; the rejected variants below must not be retried.**

### 11.1 Corpus geometry (verified)
- `n_legal ≈ 528 − ply`, near-deterministic; **red and black branching are identical** (median 496 vs 497). So on this corpus **branching band ≡ ply window**: `400+` ↔ ply 0–131, `300–399` ↔ ply ~131–233, `200–299` ↔ ply ~233+. Game lengths: min 27, **median 53**, max 280.
- Low-branching positions exist only in the rare long/marathon games: `300–399` in **27** distinct games, `200–299` in **21**.

### 11.2 What was rejected, and why (do not retry)
1. **Original 200/300/400 protocol — INFEASIBLE on seed20116.** Under the frozen `≤2/game` cap, `b300_399` tops out at **52** and `b200_299` at **42** realizable positions (21–27 distinct source games) vs the demand of **80** per band — before the raw-policy eligibility filters. No enumeration change can fix a source that lacks the distinct low-band games. (The `assign_split` capacity precheck correctly detected genuine infeasibility; this is not a tooling bug.)
2. **Original stride-4 enumeration — INDEPENDENTLY INVALID.** Because `n_legal` is monotone in ply, the qualifying-ply set is a contiguous prefix, so `qualifying[::4]` aliases ply-parity and selects a **single side** (measured: original global stride-4 → 4800 red / 0 black). It cannot satisfy the side-balance gate on any such corpus. (Masked previously only because the band-capacity stop fired first.)
3. **Revised high-band protocol (450–499 / 500–524 / 525–549, per-band stride-1 cap-6) — REJECTED.** It is *feasible for count and side* (red≈black), but it is **deliberately phase-confounded**: the bands map to plies 0–3 / 4–28 / 29–78 (all early game). `b525_549` is intrinsically the opening (plies 0–3, ~80 rows = 33% of the corpus). Stride-1 puts `b500_524` in the opening too → opening 160/240 > the ≤50% ply-bucket cap (120). An odd-stride workaround only reaches ~120 opening with **no feasibility margin** after eligibility filters. **The ply-bucket ≤50% cap is the safeguard against exactly this confounding**, so a corpus that violates it cannot screen the known **late/midgame collateral collapse mode** that v16a flagged. Rejected.

### 11.3 Decision (Option 3)
- **Retire seed20116** as the development-corpus source for this protocol. Obtain a source corpus with enough **distinct games across the intended 200–399 and later-ply geometry**.
- **Preserve the original protocol unchanged:** `200–299 / 300–399 / 400+` bands; original quotas, whole-game split, side balance, 12-ply spacing, **≤50% ply-bucket cap**; all §2 membership criteria and §6 safety gates. (The stride-4 enumeration is a rejected variant per 11.2.2 — the enumeration rule is re-validated per corpus by the preflight below, not frozen to stride-4.)
- **Add a pure feasibility preflight (§11.4).**

### 11.4 Pure feasibility preflight (REQUIRED before any operator MCTS build)
Before evaluator loading, `build_fpu_dev_corpus` must run a **pure, replay-geometry-only** preflight (no NN, no MCTS, no raw-policy) that proves the (source corpus, enumeration) pair can **jointly** satisfy the four structural constraints, or stops:
1. **Band quotas** — ≥ the per-band total (80 = 60 target + 20 control) selectable per band. (Role/target-vs-control is evaluator-dependent and remains an operator-runtime check under the stop-don't-retune rule — out of the geometric preflight's scope.)
2. **≤2/game capacity** — under `MAX_PER_GAME` and `MIN_PLY_GAP`.
3. **Side balance** — `|red − black| ≤ SIDE_TOL` per split is achievable.
4. **Ply-bucket ≤50% cap.**
"Jointly" = a single selection satisfying all four simultaneously (independent per-constraint capacity is necessary but not sufficient — see 11.2.3). If the preflight cannot prove feasibility from replay geometry, **stop before evaluator loading** and report which constraint bound.

**Pre-registration:** once a source + enumeration pass the preflight and the operator build runs, no boundary/quota/cap/stride may be retuned in response to a downstream eligibility-filter shortfall — record the shortfall and stop, as in 11.2.1.

## 12. AMENDMENT (2026-07-11) — evidence-chain hardening (required before any renewed corpus build)

Re-review of the executed plan confirmed the MCTS core (§1: None-vs-0.0, formula, bounds, completed-visit mass, observer-off golden, frozen grid), the dual-reference §6 gates + boundary/p95 semantics, and whole-game split isolation + exact quotas + fail-loud sampler are correct. The following operator/evidence-chain additions are REQUIRED before renewed implementation or corpus build.

- **12.1 Feasibility preflight (Tasks 5–6 gap) — DONE** as §11.4 (`build_fpu_dev_corpus`, commit on branch `fpu-corpus-feasibility-preflight`): pure joint witness over real replay geometry, gates `main()` before evaluator loading. Validated on seed20116 (INFEASIBLE, binding `band-capacity:b200_299`).

- **12.2 Immutable frozen-check coefficient (Task 7).** The `frozen_check` candidate must equal the `smallest_safe_r` selected on the tuning split: `run_candidates_stage(--mode frozen_check)` loads the tuning `candidates_result.json` artifact, validates its fingerprint matches, and REJECTS unless the single frozen `r` is byte-exactly that selected grid coefficient. "One nonzero r" (current §3/Task 7) is insufficient — it permits an arbitrary value.

- **12.3 Selected-A is tuning-only (Task 7).** Require selected-A presence/participation for `--mode tuning` candidates (the §6.3 mechanism gate); REJECT its presence in `--mode frozen_check`. frozen_check is a held-out dev screen only.

- **12.4 Complete persisted candidate artifacts (Task 7).** Persist, not just pass/fail reasons: per-position `candidate-vs-absolute_off` AND `candidate-vs-r0` rows (joinable by `canonical_sha1`); selected-A case rows; full NUMERIC §6.2/§6.3 gate summaries (every computed metric, both references); and the selected-coefficient record. **Join-vs-recompute:** the persisted `controls_cases.csv` omits `top_move`/`top_move_prior` needed by the low-prior-flip gate, so the candidate stage recomputes references. Either (a) persist those fields and make the candidate stage TRULY join the controls rows, or (b) explicitly define + persist the recomputation provenance (that it re-derives, and the fingerprint that guarantees recomputed == persisted). Pick one and make it explicit.

- **12.5 Strengthened fingerprints (Task 7 + builder).** A git commit alone does not detect uncommitted edits, and a replay PATH does not fingerprint replay CONTENTS. Every fingerprint must include: clean-worktree assertion (or hashes of the effective source files), source-index hash + a deterministic replay-DATA hash (contents, not paths), checkpoint hash, the FULL effective MCTS config (not a subset), and an explicit `add_noise=False` record.

- **12.6 Decisive-match preregistration (FUTURE — before the strength phase, not now).** The plan correctly stops before strength testing. Before that later operator phase, a SEPARATE preregistration is required: game count or sequential-testing rule, confidence-interval method, effect-size threshold, color/seed pairing, and stop criteria. No code in this branch.
