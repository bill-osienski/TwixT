# FPU First-Play-Value Knob + Sweep (Design)

**Status:** APPROVED (2026-07-10). One opt-in `MCTSConfig` field (byte-identical at default) plus one read-only diagnostic. No trainer, network, manifest, value-row, prior-pruning, promotion, or self-play-adoption change.

## 0. The claim under test

The c_puct falsification test upheld the mechanism: the 400-sim A metric is driven by the opponent's *first-touch* evaluation of unvisited replies. At a depth-1 node (opponent to move), `_select_child` scores an unvisited child `q = 0.0` (an even game, in the mover's perspective) plus a large exploration term, so the opponent expands hundreds of distinct replies once each instead of revisiting good ones. Each such reply is an optimistic single-visit blunder evaluation, and `root.q_value` is the unweighted mean of those leaf evaluations, so the metric inflates. Measured confirmation: across the c_puct sweep, `top_child_n_visited_children` and `root_mcts_black_value` correlate at r = +0.943, and lowering c_puct (which funnels more sims into that first-touch scan) *raised* the metric monotonically.

**This design tests the direct lever.** If the inflation is first-touch optimism, then making an unvisited child's assumed value *pessimistic for the mover* — First-Play Urgency — should make the opponent revisit its good replies instead of scanning bad ones, and should pull the 400-sim metric toward the 6400-sim reference (mean −0.045, over 10.0%, severe 3.3%).

## 1. The change

`_select_child` (`mcts.py`) currently scores an unvisited or pending child:

```python
else:
    q = 0.0
    child_visits = 0
```

Add one `MCTSConfig` field and read it here:

```python
# in MCTSConfig, immediately after c_puct:
fpu_value: float = 0.0   # First-Play Urgency: assumed Q for an unvisited child,
                         # in the MOVER's perspective. 0.0 reproduces the prior
                         # hardcoded value exactly. Negative => pessimistic =>
                         # the mover revisits known-good children before scanning
                         # unexplored ones.

# in _select_child's unvisited branch:
else:
    q = self.config.fpu_value
    child_visits = 0
```

**Byte-identical guarantee at the default.** `fpu_value` defaults to `0.0`, is read at exactly this one site, and `q = self.config.fpu_value` yields the same float `0.0` as `q = 0.0`. Both `_select_child` callers (the synchronous path at `mcts.py:694` and the batched path at `mcts.py:529`) route through this single function, so both are covered and both are unchanged at the default. The `else` branch covers both never-created children (`child is None`) and created-but-unvisited/pending children (`visit_count == 0`); FPU applies to both, which is correct — neither has a backed-up value yet. `asdict` telemetry in the codebase dumps `EvalConfig`, not `MCTSConfig`, so no telemetry JSON gains a field.

**Proof obligations, discharged by the plan:** (a) the full authoritative suite stays at its current count with zero failures — every MCTS-, self-play-, and eval-dependent test exercises the default path; (b) the diagnostic's `fpu_value=0.0` integrity check reproduces the Phase-0 concentration CSV per-case within 1e-6 (a real 400-sim search over 30 roots).

## 2. FPU form — constant, and why that is adequate here (and only here)

`fpu_value` is an **absolute constant** in the mover's perspective. On the A roots this is adequate: every A root is a contested/losing-for-black midgame position, so a constant negative FPU never has to suppress exploration at a genuinely winning node. That failure mode — a constant FPU wrongly discouraging exploration when the mover is already ahead — is real in general play, which is why production engines use parent-relative FPU *reduction* (`parent_q − c·√Σπ_visited`).

**Therefore a positive result on this diagnostic does not pre-commit the constant form into self-play.** If FPU survives to a stage-3 strength eval, the ship-form (constant vs. parent-relative reduction) is a separate decision, and the strength eval must use whichever form is intended to ship. This diagnostic answers only: *does reducing unvisited-child optimism pull the 400-sim A metric toward 6400?*

## 3. The diagnostic

`scripts/GPU/alphazero/diagnose_fpu_sweep.py`, the same shape as `diagnose_cpuct_sweep.py`: build the gate's config via `cfg_from(EvalConfig(mcts_sims=400, mcts_eval_batch_size=14, mcts_stall_flush_sims=48))`, then `dataclasses.replace(base, fpu_value=x)` so `fpu_value` is the only difference. One evaluator, reused across all values, via an explicit factory (`_make_search_fn`) so late binding is structurally impossible.

**FPU values:** `0.0, -0.05, -0.10, -0.20, -0.35, -0.50`. No more negative until the shape is seen.

**Mandatory integrity check:** at `fpu_value == 0.0`, every case's `root_mcts_black_value` reproduces `logs/eval/v15prep_a_continuation_concentration.csv` within 1e-6, or the run aborts. `--fpu-values` must include `0.0` or the run aborts (it is the only check that the per-value config binding took effect and the sweep reproduces the gate).

**Case-level CSV** (`logs/eval/fpu_check/a_predrop_fpu_sweep_cases.csv`):
```
fpu_value, case_id, root_mcts_black_value, gate_over_ge_0_25, gate_severe_ge_0_50,
root_n_visited_children, top_child_move, top_child_visit_share, top_child_q_black,
top_child_n_visited_children
```

**Summary CSV** (`logs/eval/fpu_check/a_predrop_fpu_sweep_summary.csv`):
```
fpu_value, n, mean_black_value, over_pct_ge_0_25, severe_pct_ge_0_50, positive_pct_gt_0,
root_children_mean, top_child_children_mean, top_child_visit_share_mean, min, max
```

Gate thresholds `>= 0.25` / `>= 0.50` (never `> 0`); `positive_pct_gt_0` is its own separate column.

## 4. Decision rule (the operator's)

FPU is promising only if it moves the 400-sim A metric toward the 6400 reference:

- `mean_black_value` falls materially below +0.2570
- `over_pct_ge_0_25` falls materially below 50.0%
- `severe_pct_ge_0_50` falls materially below 43.3%
- `top_child_n_visited_children` falls materially (the mechanism actually changing)
- **and** it does not merely hide the inflation by wrecking root move choice — watch `top_child_visit_share` and `top_child_move` for degenerate collapse

If FPU helps on this selected A set, validation before any adoption is, in order: an **unbiased position sample** (these 30 roots were selected by the very statistic under test); then **B/C/D gates** under the same FPU; then a **head-to-head strength eval** if still promising — using the ship-form FPU, per §2.

## 5. Out of scope

No `trainer.py`, `network.py`, manifest, value-row, prior-pruning, top-k, promotion-match, or self-play-adoption change. `fpu_value` is opt-in; nothing sets it but the diagnostic. `self_play.py:881`'s `c_puct=cfg.c_puct` is a telemetry helper (`build_root_child_details`), not search-config reconstruction, and is left untouched; if FPU is ever adopted into self-play, that adoption is a separate change with its own audit (per the surface-descriptor rule, a field-by-field config copy is exactly where a new knob silently fails to propagate).
