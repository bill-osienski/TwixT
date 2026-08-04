# Atlas Stage 3 — Delivered Interfaces and Qualification (Stage 4 Handoff)

**Status:** Stage 3 LANDED and QUALIFIED, 2026-08-04, at clean tree `705b46e`.
**No reservoir generated, no checkpoint loaded, no MLX executed, no measurement run.**

**Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §2b, §4, §8
(EXECUTION-FROZEN). **Plan:** `2026-08-04-atlas-stage3-warm-replay-ladder.md`.

## Qualification

```text
full suite   2435 passed / 4 skipped / 53 deselected / 0 failed   (REAL_EXIT=0)
             6m57s, exit code read from the process, never a pipe
Stage 3 new  38 tests   (warm-replay 32, integration 6)
```

`2397 + 38 = 2435` exactly — **no pre-existing test changed behaviour**, and none was
modified, skipped or deleted. The predicted count was corrected before the run, which
is what let the delta function as a check rather than decoration.

## Delivered interfaces — `scripts/GPU/alphazero/warm_prefix_replay.py`

```python
BOUNDARY_THRESHOLD = 320
LEG_INCREMENTS = (400, 1200, 1600, 3200)      # additive
NOMINAL_B      = (400, 1600, 3200, 6400)      # cumulative

replay_seed_for(meta, base_seed) -> int        # VERIFIES against the sidecar

PrefixStep(ply, forced_move, forced_child_visits, inheritance_reset,
           zero_effective_inheritance, state_agrees)
PrefixResult(root, inherited_I, steps, reset_count, reset_rate,
             last_reset_ply, cache_clears)
replay_prefix(mcts, meta, move_history, target_ply, active_size) -> PrefixResult

BoundaryRecord(N_actual, overshoot, remaining, flush_type)
BatchSafeBoundaryObserver(inherited_I, threshold=320, leg_B=400, tracer=None)
    .record                          -> Optional[BoundaryRecord]
    .tracer_snapshot_at_boundary     -> Optional[dict]

LegResult(nominal_B, inherited_I, effective, root_value, selected_move,
          selected_move_prior_rank, top_share, top_two_margin,
          effective_children, n_visited_children, visit_counts)
_root_summary(root, visit_counts, selected_move) -> dict
run_additive_ladder(mcts, root, inherited_I, ply, boundary_observer=None,
                    target_tracer=None, increments=LEG_INCREMENTS)
    -> tuple[list[LegResult], dict]            # dict: at_boundary, at_400

project_runtime(rows, mean_prefix_plies, tracer_overhead=0.010) -> dict
```

`scripts/GPU/alphazero/run_atlas_ladder.py` — `emit-plan`, `project-runtime`. Runs
no measurement.

## Invariants Stage 4 may rely on

- **Every rung's evidence is frozen before the tree advances past it.** The ladder is
  additive on ONE tree, so after leg 4 the 400/1,600/3,200 states exist nowhere.
  `LegResult` already carries the visit distribution, `effective_children`,
  `top_share`, `top_two_margin` and the selected move's prior **rank** — §5 and §7 are
  computable without re-searching.
- **Two frozen §8 snapshots**: `at_boundary` (taken by the observer at the quiescent
  instant) and `at_400`. `run_additive_ladder` **refuses** a tracer that is non-empty
  or is not the MCTS's selection observer, so a prefix-contaminated snapshot cannot
  reach Stage 4.
- **Canonical leader everywhere.** `visit_leader_move` (ties by lowest encoded move id),
  never `max()` over dict order.
- **`N_actual = root.visit_count − I`**, asserted `320 ≤ N_actual ≤ 400`, now exercised
  with genuinely nonzero `I`.
- **One RNG per row**, continued across prefix and all four legs, with a non-vacuity
  control proving that reseeding changes the result.
- **Prefix asserts legality and canonical state agreement** (`to_move`, `pegs`,
  `bridges`) after every advance; cache clears counted one per advance.

## Cost, measured not assumed

`project-runtime --rows 240 --mean-prefix-plies 69`:

| term | sims/row |
|---|---:|
| prefix replay | 27,600 |
| additive ladder | 6,400 |
| **dominant** | **prefix_replay (4.3×)** |

Confirms §4's prediction. `mean_prefix_plies` has **no default** and must come from
the corpus's observed per-phase ply supply — never a smoke.

## What Stage 4 must still close

1. **Scale.** Everything is qualified at `active_size=6`, `FakeEvaluator`, 3–5-ply
   prefixes and tiny legs (one real 400-simulation leg in the boundary-timing test).
   Throughput at board 24 against a real checkpoint is unmeasured.
2. **`remaining` distribution.** §4's preregistered deployability rule fails Read-out A
   if the **median `remaining` is zero**. Only a single-row value has been observed.
3. **Reset rate on real games.** `inheritance_reset` fires when a recorded move was
   never expanded. Its rate under shipped generation is unknown, and a high rate would
   mean the warm start is not actually warming.

## Standing authorization boundary

Corpus generation and every GPU measurement run remain **unauthorized**. Stages 4–5
are unplanned; Stage 4 is planned only against these interfaces once accepted.
