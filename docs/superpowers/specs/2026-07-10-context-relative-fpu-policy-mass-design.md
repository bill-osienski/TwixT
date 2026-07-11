# Context-Relative FPU (Parent-Relative + Explored-Policy-Mass) — Design

**Status:** APPROVED design (2026-07-10). Tooling-only branch. The heavy MCTS phases (geometry scan, corpus generation, coefficient sweep, selection, frozen-split check, held-out validation) are **operator phases** and are NOT run by this implementation.

## 0. Objective & context

v16a rejected an **absolute** `fpu_value = -0.20` as a global 400-sim setting (late new-collapse 15.48%; postmortem: a fixed floor commits flat-prior, near-even, high-branching roots to a network-dispreferred move — 12/15 collapses onto priors ranked #16–#207). See `[[a-signal-search-artifact-fpu]]` and `docs/superpowers/specs/2026-07-10-v16a-fpu-reject-postmortem.md`.

The successor must make the FPU reduction **context-relative** so it does not park a constant penalty near the value of contested positions. A plain constant-reduction parent-relative rule (`Q_parent − const`) is *also* rejected: at a near-even node it recreates the absolute floor. The reduction must be scaled by search/policy geometry.

**This is a reject-unsafe-candidates rung, not the strength benchmark.** The decisive endpoint remains a same-checkpoint, same-400-sim, balanced-color, statistically-significant strength gain. Matching the deeper-search reference and passing these screens is mechanistic evidence only.

## 1. §A — The rule

At the single unvisited/pending-child site in `_select_child` (`mcts.py`, currently `q = self.config.fpu_value`), when the policy-mass mode is enabled the assumed value becomes:

```
P_explored = Σ prior(a)  over root/parent children with visit_count > 0   # computed once per _select_child call
FPU        = Q_parent − r · sqrt( clamp(P_explored, 0, 1) )
```

- `Q_parent` = the parent node's value in the **mover's perspective** (`node.q_value`; the existing `q = -child.q_value` line confirms every node's `q_value` is in its own to-move perspective, and `assert root.q_value == root_value_stm` in the v15 diagnostic confirms it for the root). The FPU value is in the mover's perspective, matching the current `fpu_value` semantics.
- `P_explored` uses the **normalized legal-move priors already stored on the node** (`node.priors`); only children with `visit_count > 0` contribute; clamped to `[0, 1]` defensively.
- `r` is the single coefficient (the reduction reached at full explored mass). It must be **finite and ≥ 0**.

### Opt-in field + mutual-exclusion guard

New `MCTSConfig` field:

```python
fpu_policy_mass_reduction: float | None = None
```

- `None` (default) → the **existing** `fpu_value` path, unchanged and **byte-identical**.
- Not `None` → the policy-mass formula above. **`0.0` is an enabled mode** yielding `FPU = Q_parent` (parent-relative, zero reduction); it is **not** equivalent to `None` (which keeps the absolute-FPU path). This distinction is tested explicitly.

Validation at config construction (fail loud):
- If `fpu_policy_mass_reduction is not None` **and** `fpu_value != 0.0` → raise (ambiguous combined absolute+relative behavior is disallowed).
- If `fpu_policy_mass_reduction is not None` and it is not finite or `< 0` → raise.

### Pure helper (unit-testable without MCTS)

```python
def policy_mass_fpu(parent_q: float, explored_mass: float, r: float) -> float:
    m = 0.0 if explored_mass < 0.0 else (1.0 if explored_mass > 1.0 else explored_mass)
    return parent_q - r * math.sqrt(m)
```

`_select_child` computes `P_explored` once (a single pass over `node.priors`/`node.children`) and calls this helper in the unvisited branch when the mode is enabled.

### Byte-identical-off proof

- Full existing repository suite passes unchanged (every MCTS/self-play/eval test exercises the `None` path).
- An `old==new` selection-trace check: with both new features disabled (`fpu_policy_mass_reduction=None`, observer off), a fixed synthetic tree produces byte-identical `_select_child` decisions and search output vs the pre-branch code.

## 2. §B — Development corpus (geometry, not failure-selected)

**Source:** `logs/eval/0379_vs_calib020_0001_800g_w4_seed20116_replay_games.jsonl` — same matchup as selected-A/v16a, different games/seed (zero position overlap; holds opponent pair, checkpoint strength, generation policy, and game distribution constant so responses attribute to the rule, not matchup composition).

**Membership is decided ONLY by the fpu-off `calib020_0001` anchor + raw policy — never by any candidate-FPU result.** For each candidate position (reconstructed through the trusted `position_state` path): run the anchor at 400 sims fpu-off, run the raw policy forward pass at the root, and record `n_legal`, `root_value_stm`, `normalized_entropy = H(prior)/log(n_legal)`, `top1_prior`, `top4_mass`, `top8_mass`, `game_idx`, `ply`, `side`, `game_result`, `total_plies`.

**Target eligibility (the mechanism, deliberately broad — not a copy of the 15 rejected cases):**
```
n_legal ≥ 200  ∧  |root_value_stm| ≤ 0.25  ∧  normalized_entropy ≥ 0.90  ∧  top1_prior ≤ 0.025
```

**Composition — 240 positions:**
- 180 target-geometry: 60 with `n_legal` 200–299, 60 with 300–399, 60 with 400+.
- 60 matched controls: comparable branching and near-even value, but **concentrated** policy (`normalized_entropy < 0.85` **or** `top1_prior ≥ 0.05`). These verify the rule preserves efficient concentration when the network genuinely has a strong preference.

**Sampling:** game-first round-robin; ≤ 2 positions per game; ≥ 12-ply separation within a game; ~50/50 side to move; ply is descriptive only (cap any single ply bucket at ≤ 50% of the corpus); include state-cap / winner-unknown games. Deterministic under a fixed seed.

**Split by whole game (not by position):** 160 tuning / 80 internal frozen check. Coefficient selection uses only the 160. The 80 is run once after `r` is frozen; if it fails, reject the rule or build a new corpus — never retune on the 80.

## 3. §C — Discovery diagnostic + metrics

Runs two position sets under each configuration and emits paired metrics:
- **Selected-A** (the 30 discovery roots) — to confirm the mechanism still works.
- **Development corpus** (240) — to confirm no lock-in and to choose `r`.

**Configurations:** `absolute_off` (`fpu_policy_mass_reduction=None`, the shipped default — byte-identical reference and current-behavior baseline); `r = 0.0` (parent-relative control, `FPU = Q_parent`); candidates `r ∈ {0.10, 0.20, 0.35, 0.50, 0.75}`. Every paired metric is reported against **both** references (`r = 0.0` and `absolute_off`); the pre-registered gates below then bind each criterion to its natural reference:

- **Dev-corpus safety** (small reduction at low explored mass; no low-prior lock-in; new-collapse and top-move churn "near control"): referenced to **`r = 0.0`** — the control per "include r=0.0 as the control," which isolates the coefficient's effect since only the reduction varies.
- **Selected-A mechanism improvement** (opponent-reply scanning decreases; root moves toward the deeper-search reference): referenced to **`absolute_off`** — "fpu-off" here means the shipped absolute default, i.e. the behavior whose first-touch scan we are fixing.
- **Byte-identity** (criterion 6): `absolute_off` (config `None`) and observer-off vs the pre-branch code.

(Reporting both references means the reviewer can re-bind a criterion without re-running; the assignment above is the pre-registered default.)

**Per-position metrics** (paired): selected-move prior + prior rank; effective-children Δ; opponent-reply-count Δ; top-share Δ; root-value (mover) Δ; new-collapse / resolved-collapse; top-move flip. Plus the **dynamics** from the trace collector (§4).

**Central acceptance check:** *does the candidate commit early to a very low-prior move while most policy mass is still unexplored?* If yes on the target geometry, the candidate has reproduced the v16a failure in a new form and is rejected.

## 4. Trace collector (opt-in, read-only, exact — on the single 400-sim run)

A strictly optional observer attached to one real 400-sim search (no separate budget runs — those diverge under batching/pending/stall-flush and cost ~2× without exact events). It records **transitions and threshold crossings**, not per-sim tree copies:

- Simulation index when each root move is first visited.
- Explored policy mass when the **eventual final top move** is first visited.
- Leader changes; the final leader's **last takeover** simulation (its stabilization point).
- First simulation where explored policy mass crosses **25% / 50% / 75%**.
- Explored policy mass when the final move first becomes leader.
- Selected move's prior and prior rank.

**Isolation guarantees:** the observer defaults to `None`; it must not influence selection, evaluation, RNG consumption, batching, or backup. `mcts.py` holds only an optional reference and a single guarded, read-only callback (e.g. after each backup/flush, passed the root + a monotonic sim counter); the observer logic lives in the diagnostic module. Observer-off (`None`) is included in the byte-identical proof alongside FPU-off.

## 5. Pre-registered coefficient-selection protocol

`r = 0.0` is the control, not a candidate. To choose one coefficient:

1. **Reject** any `r` that produces unsafe collapse, low-prior exploration-order lock-in, or broad distortion on the **160-position development split or its matched controls**.
2. Among survivors, **require** meaningful selected-A mechanism improvement: opponent-reply scanning decreases; root behavior moves toward the deeper-search reference; and the improvement is **not** produced through collapse.
3. Choose the **smallest `r`** satisfying both (prevents picking the most aggressive coefficient just because it makes A look best).

If none pass: **reject the formula.** Do not interpolate between grid values, extend the grid, or inspect the frozen 80-position split to choose another coefficient.

## 6. Pre-registered acceptance criteria (discovery)

1. Selected-A opponent-reply scanning drops materially from fpu-off.
2. Selected-A root behavior moves toward the deeper-search reference (not via collapse).
3. On high-branching, flat-prior development positions, early explored mass produces only a small reduction.
4. No low-prior exploration-order lock-in on the target geometry.
5. New-collapse and top-move churn remain near control.
6. Default-off (and observer-off) behavior remains byte-identical.

## 7. Branch scope (mirrors v16a)

**Ships (this branch, all verified):**
- `MCTSConfig.fpu_policy_mass_reduction` field, formula in `_select_child`, config validation + mutual-exclusion guard, pure `policy_mass_fpu` helper.
- Optional read-only search trace hook + observer.
- Geometry-corpus builder (pure sampler + confirm-under-anchor shell).
- Discovery diagnostic + output schemas + metric computations.
- Unit tests with fakes / synthetic trees (no MCTS in unit tests).
- `old==new` trace proof with both new features disabled; full repository suite green.
- This design note (frozen protocol).

**Does NOT run (operator phases, after tooling lands):**
- seed20116 anchor geometry scan; final corpus generation.
- 5-coefficient discovery sweep; coefficient selection.
- Frozen 80-position check.
- Cross-matchup (pooled) or fresh held-out validation.

## 8. File structure

- **Modify** `scripts/GPU/alphazero/mcts.py` — `MCTSConfig` field + validation; `_select_child` policy-mass branch + one-pass `P_explored`; single guarded optional trace callback; `policy_mass_fpu` helper.
- **Create** `scripts/GPU/alphazero/build_fpu_dev_corpus.py` — pure geometry sampler (round-robin/cap/gap/split/eligibility) + confirm-under-anchor shell (operator-run) + meta.
- **Create** `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — discovery diagnostic (selected-A + dev corpus × configs), the trace observer, metric + schema emission (operator-run).
- **Create** tests: `tests/test_fpu_policy_mass_rule.py` (helper, guard, `0.0`≠`None`, byte-identical selection trace), `tests/test_fpu_dev_corpus.py` (sampler determinism/round-robin/split/eligibility/controls), `tests/test_fpu_trace_observer.py` (event logic on synthetic trees).
- **Create** this design note.

## 9. Non-goals / do-not-change

No self-play adoption: `SIMS_TABLE`, `self_play.py`, trainer, network, promotion, calibration manifests, value-adapter/projection are untouched. This branch does not thread FPU into self-play; adoption is a separate, later commitment gated on the strength match. The v16a manifest/outputs remain frozen and are not used here.

## 10. Progression

```
seed20116 geometry-confirmed dev corpus
  → 5-coefficient discovery sweep (160 tuning) → smallest-safe-passing r → FREEZE
  → 80-position internal check (once)
  → pooled cross-matchup robustness (option 2 corpora)
  → fresh game-held-out safety corpus (new decision table)
  → selected-A diagnostic + B/C/D guardrails
  → same-checkpoint, same-400-sim, balanced-color strength match  ← decisive
  → (only then) controlled self-play pilot
```
