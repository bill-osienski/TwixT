# Context-Relative FPU (Parent-Relative + Explored-Policy-Mass) — Design

**Status:** APPROVED design with required protocol edits (2026-07-10). Tooling-only branch. The heavy MCTS phases (geometry scan, corpus generation, coefficient sweep, selection, frozen-split check, held-out validation) are **operator phases** and are NOT run by this implementation.

## 0. Objective & context

v16a rejected an **absolute** `fpu_value = -0.20` as a global 400-sim setting (late new-collapse 15.48%; postmortem: a fixed floor commits flat-prior, near-even, high-branching roots to a network-dispreferred move — 12/15 collapses onto priors ranked #16–#207). See `[[a-signal-search-artifact-fpu]]` and `docs/superpowers/specs/2026-07-10-v16a-fpu-reject-postmortem.md`.

The successor makes the FPU reduction **context-relative** so it never parks a constant penalty near the value of contested positions. A constant-reduction parent-relative rule (`Q_parent − const`) is *also* rejected: at a near-even node it recreates the absolute floor. The reduction must be scaled by search/policy geometry.

**This is a reject-unsafe-candidates rung, not the strength benchmark.** The decisive endpoint remains a same-checkpoint, same-400-sim, balanced-color, statistically-significant strength gain.

## 1. §A — The rule

At the single unvisited/pending-child site in `_select_child` (`mcts.py`, currently `q = self.config.fpu_value`), when the policy-mass mode is enabled:

```
P_explored = Σ prior(a)  over parent children with a COMPLETED (backed-up) visit   # one pass per _select_child call
FPU        = Q_parent − r · sqrt( clamp(P_explored, 0, 1) )
```

- `Q_parent` = the parent node's value in the **mover's perspective** (`node.q_value`; `q = -child.q_value` confirms each node's `q_value` is in its own to-move perspective, and `assert root.q_value == root_value_stm` confirms it for the root). FPU is in the mover's perspective, matching `fpu_value` semantics.
- `P_explored` uses the **normalized legal-move priors already on the node** (`node.priors`); clamped to `[0,1]`.
- `r` (single coefficient) = the reduction reached at full explored mass; **finite and ≥ 0**.

### Completed-visits-only (safeguard 6a)

`P_explored` counts a child **only when `child.visit_count > 0` reflects a completed, backed-up visit** — never a virtual or pending visit. This holds in the current code: `_select_child`'s virtual penalty is a **local** adjustment (`child_visits += self.config.pending_virtual_visits`) and does **not** mutate `child.visit_count`; pending leaves are not backed up until evaluation completes, so their `visit_count` stays 0. The implementation must re-confirm this invariant (a test asserts a pending/virtual child contributes 0 to `P_explored`); if a future batching change ever folds virtual/pending into `visit_count`, `P_explored` must switch to an explicit completed-visit indicator.

### Opt-in field + mutual-exclusion guard

```python
fpu_policy_mass_reduction: float | None = None
```

- `None` (default) → the **existing** `fpu_value` path, **byte-identical**.
- Not `None` → the policy-mass formula. **`0.0` is an enabled mode** (`FPU = Q_parent`, zero reduction) — **not** equivalent to `None`; tested explicitly.

Fail-loud validation at `MCTSConfig` construction:
- `fpu_policy_mass_reduction is not None` **and** `fpu_value != 0.0` → raise (no ambiguous absolute+relative combination).
- `fpu_policy_mass_reduction is not None` and not finite, or `< 0` → raise.

### Pure helper (safeguard 6b — reject nonfinite inputs)

```python
def policy_mass_fpu(parent_q: float, explored_mass: float, r: float) -> float:
    if not (math.isfinite(parent_q) and math.isfinite(explored_mass) and math.isfinite(r)):
        raise ValueError("policy_mass_fpu requires finite inputs")   # NaN would pass both clamp comparisons
    m = 0.0 if explored_mass < 0.0 else (1.0 if explored_mass > 1.0 else explored_mass)
    return parent_q - r * math.sqrt(m)
```

`_select_child` computes `P_explored` once and calls this in the unvisited branch when the mode is enabled.

### Byte-identical-off proof

Full existing suite passes unchanged (every MCTS/self-play/eval test exercises the `None` path), plus an `old==new` selection-trace check on a fixed synthetic tree with **both** new features disabled (`fpu_policy_mass_reduction=None`, observer `None`).

## 2. §B — Development corpus (geometry, not failure-selected)

**Source:** `logs/eval/0379_vs_calib020_0001_800g_w4_seed20116_replay_games.jsonl` — same matchup as selected-A/v16a, different games/seed.

**Membership is decided ONLY by the fpu-off `calib020_0001` anchor + raw policy — never any candidate-FPU result.**

### Bounded two-stage scan (edit 3)

Running a 400-sim anchor on every ply would dominate the experiment. Deterministic procedure:

1. **Ply enumeration (cheap, from stored replay records):** for each game (ascending `game_idx`), consider plies whose stored `n_legal ≥ 200`, take **every 4th** such ply, capped at **6 pre-anchor candidates per game**. (Fixed stride + cap so the builder does not silently keep whichever positions it meets first.)
2. **Raw-policy pre-filter (cheap, our net):** reconstruct each candidate via the trusted `position_state` path, run one raw-policy forward pass, keep those with `normalized_entropy = H(prior)/log(n_legal) ≥ 0.90` **and** `top1_prior ≤ 0.025` (target) — or the concentrated control filter (below).
3. **Anchor confirm (expensive, survivors only):** run the 400-sim fpu-off anchor **only** on raw-policy survivors; keep those with `|root_value_stm| ≤ 0.25`.
4. Continue in deterministic order until each branching-band quota is filled **plus a deterministic reserve pool** (≥ 2× quota retained) to absorb later dedup/disjointness drops.

Recorded per position: `source_corpus_id`, `game_idx`, `position_ply`, `side`, `game_result`, `total_plies`, `n_legal`, `root_value_stm` (anchor), `normalized_entropy`, `top1_prior`, `top4_mass`, `top8_mass`, `canonical_position_sha1` (see §2.3), and `split` (see §2.2).

**Target eligibility (broad — the mechanism, not the 15 rejected cases):**
```
n_legal ≥ 200  ∧  |root_value_stm| ≤ 0.25  ∧  normalized_entropy ≥ 0.90  ∧  top1_prior ≤ 0.025
```
**Matched controls:** comparable branching/value but **concentrated** policy (`normalized_entropy < 0.85` **or** `top1_prior ≥ 0.05`) — verify the rule preserves efficient concentration when the net has a real preference.

**Composition — 240 positions:** 180 target (60 per band: `n_legal` 200–299 / 300–399 / 400+) + 60 matched controls. Game-first round-robin; ≤ 2 per game; ≥ 12-ply separation within a game; ~50/50 side; ply descriptive only (cap any ply bucket ≤ 50%); include state-cap / winner-unknown games; deterministic under a fixed seed.

### 2.2 Split — hard-isolated in code (edit 1)

Split **by whole game** (not position). Every manifest row carries `split ∈ {tuning, frozen_check}`, preserving composition:
- `tuning`: 120 target + 40 controls (160).
- `frozen_check`: 60 target + 20 controls (80).

The diagnostic enforces modes (§3). Coefficient selection uses only `tuning`; `frozen_check` is run once after `r` is frozen (never retuned on).

### 2.3 Canonical-state disjointness proof (edit 5)

"Different seed ⇒ disjoint" is demonstrated, not assumed. For every dev-corpus position, compute `canonical_position_sha1 = sha1(canonical_state)` where the canonical state is built from **board size, sorted red-peg locations, sorted black-peg locations, side-to-move** (a move-prefix hash is insufficient — transpositions reach the same board via different move orders). The builder **fails** if any dev-corpus hash collides with the hash set of selected-A **or** the consumed v16a positions (both hashed identically). This guarantees position-level disjointness across the discovery/validation ladder.

## 3. §C — Discovery diagnostic + modes

**Configurations:** `absolute_off` (`None`, shipped default — byte-identical ref + current-behavior baseline); `r = 0.0` (parent-relative control, `FPU = Q_parent`); candidates `r ∈ {0.10, 0.20, 0.35, 0.50, 0.75}`.

Every paired metric is reported against **both** references (`r = 0.0` and `absolute_off`). Pre-registered binding (§6): dev-corpus safety gates → `r = 0.0`; selected-A improvement → `absolute_off`; byte-identity → `absolute_off`/observer-off vs pre-branch.

### Enforced modes (edit 1)

```
--mode tuning:
   reads ONLY split==tuning rows (+ selected-A)
   permits configs: absolute_off, r=0.0, and the five-r grid
   selected-A participates in the A gate
--mode frozen_check:
   reads ONLY split==frozen_check rows
   permits configs: absolute_off, r=0.0, and EXACTLY ONE nonzero (frozen) r
   rejects a grid or >1 nonzero candidate
   does NOT include selected-A in its pass/fail
```
The script **fails loudly** if a mode is handed rows from the wrong split (validates the `split` column of every row against `--mode`).

**Per-position metrics** (paired, vs both references): selected-move prior + prior rank; effective-children Δ; opponent-reply-count Δ; top-share Δ; root-value (mover) Δ; new-collapse / resolved-collapse; top-move flip; plus the exact dynamics from the trace observer (§4).

**Central acceptance check:** *does the candidate commit early to a very low-prior move while most policy mass is still unexplored?* (Operationalized as the lock-in event, §6.)

## 4. Trace collector — exact, per-completed-simulation (edit 4)

An optional observer attached to the single real 400-sim run (no budget snapshots — they diverge under batching and lack exact events). Interface:

```python
observer.on_root_simulation(completed_simulation_count, root, updated_root_move)
```

Fires **once per completed root-simulation backup** (once per backed-up leaf, in backup order — **not** once per batch flush covering several sims), passing the monotonic completed count and the root move on that simulation's path so the observer updates **incrementally** (no 200–500-child rescan per sim).

Requirements:
- `completed_simulation_count` counts **completed** simulations, never attempted/pending.
- Explored mass excludes any **virtual/pending** visit (consistent with §1 completed-visits-only).
- The current leader uses the **same tie-break as MCTS** (max visit; ties → lowest encoded move id).
- The final leader and its **stabilization point** are computed from the recorded **leader-change timeline** (the final leader's last takeover simulation).
- The observer records: first-visit sim per root move; explored mass when the eventual final top move is first visited; leader-change timeline; first sim crossing **25% / 50% / 75%** explored mass; explored mass when the final move first becomes leader; selected move's prior + rank.
- **Isolation:** observer defaults `None`; must not affect selection, evaluation, RNG consumption, batching, or backup. `mcts.py` holds only the optional reference + the single guarded callback; observer logic lives in the diagnostic module. An observer **exception aborts the diagnostic** (never silently alters search). Observer-off is part of the byte-identical proof.

## 5. §D — Pre-registered coefficient-selection protocol

`r = 0.0` is the control, not a candidate.

1. **Reject** any `r` failing a dev-corpus safety gate (§6) on the **160-position tuning split or its controls**.
2. Among survivors, **require** the selected-A mechanism gate (§6).
3. Choose the **smallest `r`** satisfying both.

If none pass: **reject the formula.** Do not interpolate, extend the grid, or inspect the `frozen_check` split to choose another coefficient.

## 6. Pre-registered NUMERIC acceptance gates (edit 2)

Qualitative terms are replaced by exact gates. **Freezing procedure:** the control-relative numbers below may be finalized *after* running only `absolute_off` and `r = 0.0` (to read the control distribution), and **must be committed to this section before any nonzero-`r` result is produced.** The fixed numbers stand as pre-registered defaults now.

### 6.1 Early low-prior lock-in event (per target position, boolean)

Qualifies iff **all**:
```
selected_move_prior_rank > 10
selected_move_prior      < 0.01
explored_mass_at_final_leader_stabilization < 0.25      # committed early, before 25% mass
final_leader_stabilization_sim              ≤ 100
final_root_top_share     ≥ 0.90
```

### 6.2 Dev-corpus safety gates (reference `r = 0.0`; 120 target + 40 controls in tuning)

REJECT the coefficient if any:
- **New-collapse:** target new-collapse rate ≥ **5%**, OR any branching band (n ≥ 20) ≥ **10%** (mirrors the v16a reject line).
- **Lock-in:** early-low-prior-lock-in event count on target > **control count + 2** (control = the `r=0.0` lock-in count; committed after the control run).
- **Broad distortion:** target p95 `|mover-value Δ|` ≥ **0.35** (inspect at ≥ 0.20).
- **Compound concentration:** mean effective-children reduction ≥ **50%** AND mean top-share increase ≥ **+0.15** (v16a compound; narrowing alone is the intended mechanism and is not a reject on its own).
- **Control harm:** on the 40 controls — any top-move flip to a **lower-prior** move rate ≥ **10%**, OR control p95 `|mover-value Δ|` ≥ **0.35** (the rule must not wreck positions where the net has a genuine preference).

Top-move flip rate is **reported** (inspection) but is not a standalone reject on target — flips can be the intended correction.

### 6.3 Selected-A mechanism gate (reference `absolute_off`; 30 roots)

REQUIRE **all**:
- Mean opponent-reply-count (`top_child_n_visited_children`) reduced by ≥ **50%** vs `absolute_off` (mechanism engaged; the rejected −0.20 achieved ~95%, so 50% is a floor).
- Mean A metric (`root_mcts_black_value`) moves ≥ **50% of the way** from `absolute_off` toward the 6400-sim reference (**−0.045**).
- New-collapse count on A ≤ **2** (improvement not achieved via collapse).
- Mean top-share increase on A ≤ **+0.15** (not degenerate concentration).

### 6.4 Byte-identity

`absolute_off` (config `None`) and observer-off reproduce the pre-branch code byte-for-byte; full repository suite green.

## 7. Branch scope (mirrors v16a)

**Ships (verified):** `MCTSConfig.fpu_policy_mass_reduction` + validation + mutual-exclusion guard; `_select_child` policy-mass branch + one-pass completed-visit `P_explored`; pure `policy_mass_fpu` helper; optional read-only per-completed-simulation trace hook + observer; geometry-corpus builder (pure sampler + confirm-under-anchor shell + canonical-hash disjointness); discovery diagnostic with enforced tuning/frozen_check modes + output schemas + metric computations; unit tests (fakes / synthetic trees, no MCTS); `old==new` trace proof with both features disabled; full repository suite; this design note.

**Does NOT run (operator phases):** seed20116 anchor geometry scan; final corpus generation; 5-coefficient sweep; coefficient selection; frozen 80-position check; cross-matchup / fresh held-out validation.

## 8. File structure

- **Modify** `scripts/GPU/alphazero/mcts.py` — `MCTSConfig` field + validation; `_select_child` policy-mass branch + one-pass `P_explored`; single guarded optional `on_root_simulation` callback; `policy_mass_fpu` helper.
- **Create** `scripts/GPU/alphazero/build_fpu_dev_corpus.py` — pure sampler (round-robin/cap/gap/split/eligibility/band quotas) + two-stage confirm-under-anchor shell + canonical-hash disjointness + meta (operator-run).
- **Create** `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — discovery diagnostic (mode-enforced; selected-A + dev corpus × configs), the trace observer, metric/gate + schema emission (operator-run).
- **Create** tests: `tests/test_fpu_policy_mass_rule.py` (helper incl. nonfinite reject; guard; `0.0`≠`None`; completed-visit-only `P_explored`; byte-identical selection trace), `tests/test_fpu_dev_corpus.py` (sampler determinism/round-robin/split-composition/eligibility/controls/canonical-hash + disjointness), `tests/test_fpu_trace_observer.py` (per-completed-sim events, leader timeline/stabilization, mass thresholds on synthetic trees), `tests/test_fpu_diagnostic_modes.py` (tuning/frozen_check row isolation + config-set validation + wrong-split failure + numeric-gate computations).
- **Create** this design note.

## 9. Non-goals / do-not-change

No self-play adoption: `SIMS_TABLE`, `self_play.py`, trainer, network, promotion, calibration manifests, value-adapter/projection untouched. Adoption is a later commitment gated on the strength match. The v16a manifest/outputs remain frozen and are not used here.

## 10. Progression

```
seed20116 geometry-confirmed dev corpus (disjointness-proven)
  → 5-coefficient sweep on the 160 tuning split → smallest-safe-passing r → FREEZE
  → 80-position frozen_check (once)
  → pooled cross-matchup robustness (option-2 corpora)
  → fresh game-held-out safety corpus (new decision table)
  → selected-A diagnostic + B/C/D guardrails
  → same-checkpoint, same-400-sim, balanced-color strength match  ← decisive
  → (only then) controlled self-play pilot
```
