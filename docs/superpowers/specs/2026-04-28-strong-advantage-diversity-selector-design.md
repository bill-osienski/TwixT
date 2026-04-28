# Strong-Advantage Probe Tier — Diversity-Aware Selector Design

**Date:** 2026-04-28
**Status:** approved design, pending implementation plan
**Parent spec:** [2026-04-28-strong-advantage-probe-tier-design.md](./2026-04-28-strong-advantage-probe-tier-design.md)
**Operator doc:** [../../probe-suite-generation.md](../../probe-suite-generation.md)
**Touch points:** `scripts/build_probe_suite.py::_run_strong_advantage`, `scripts/GPU/alphazero/probe_eval.py::extract_strong_advantage_candidates`, `tests/probes/strong_advantage_probes.json`, `tests/probes/candidates_strong_advantage.json`

---

## 1. Motivation

The first production run of `scripts/build_probe_suite.py --tier strong_advantage` produced 30 admitted probes from only 16 unique source games. One game (`iter_0058_game_040`) contributed 5 consecutive plies (50–54), all admitted with near-identical Phase-1 features. Effective independence was much lower than `n=30` suggested; ~half the suite was clustered. Manual thinning — drop 8 redundant probes, cap at 2 per game — was required to ship a defensible 22-probe committed suite.

**Root cause** (`scripts/build_probe_suite.py:380`):
```python
admitted = admitted[: args.max_probes]
```
There is no per-game cap, no near-ply dedupe, and no category-aware fill. The order of `admitted` is whatever Phase 1 produced — sorted by `(-iter, -source_ply, source_game)` — so consecutive plies of one game cluster at the top of the list and the slice greedily takes them.

**Compounding factor — Phase-1 K-loop fan-out** (`scripts/GPU/alphazero/probe_eval.py:801`):
```python
for k in range(k_plies_range[0], k_plies_range[1] + 1):  # default (3, 8) → 6 plies per game
```
Each game can contribute up to 6 candidate plies. When a game has a strong winning chain by ply T-8, all 6 K-shifts typically pass Phase-1 thresholds because the chain only grows monotonically. The audit confirms this: at least 6 of the most-clustered games have all 6 K's surviving Phase 1, and at the production run's level several had all 6 surviving Phase 2 as well.

**Audit double-counting** (discovered while inspecting the audit):
- Phase 1 writes `audit.append({**base, "reason": "admitted"})` for every Phase-1 survivor (`probe_eval.py:855`).
- Phase 2 also writes one audit row per labeled candidate, with `reason="admitted"` if it passes the admission filter (`build_probe_suite.py:345-351`).
- So each Phase-2-admitted candidate appears twice in the audit; each Phase-2-rejected candidate appears as one Phase-1 `admitted` plus one rejection row.
- The 527 `admitted` reasons in the production audit reverse-engineer to ~323 Phase-1 admits and ~204 Phase-2 admits, **not** 527.

This makes diversity-drop diagnostics unreliable unless the schema is fixed in the same change.

## 2. Goals

- **Diverse across games:** no single game dominates the suite.
- **Diverse across categories:** the 4-category structure the tier is meant to measure is preserved when categories are populated.
- **Deterministic and byte-reproducible:** same inputs → same output, independent of dict iteration order.
- **Audit-visible:** every diversity-driven drop has a specific, queryable reason.
- **Doesn't sacrifice yield from sparse categories:** when categories are empty (currently common for `edge_*`), backfill from non-empty categories is the expected path, not a fallback.
- **`reason="admitted"` count is honest:** one canonical audit row per probe in the final suite.

## 3. Non-goals

- **No Phase-1 changes for diversity.** Phase-2 labels are useful for choosing among siblings; the cap stays post-admission. Phase-2 MCTS labeling cost is acceptable at current scale.
- **No multi-mode CLI.** One selector behavior. No `--selection-mode`, `--prefer-quality-key`, `--allow-category-overfill`. Defer until a real second mode appears.
- **No tunable quality key.** The ranking order is hard-coded; tuning is a code change, not a CLI flag.
- **No changes to the `forced` tier.** Its red/black interleave is unrelated and stays as-is.

## 4. Design

### 4.1 Pipeline

Replace the single line `admitted = admitted[: args.max_probes]` with a call to a new helper:

```python
admitted = _select_diverse_admitted_candidates(
    admitted,
    audit,
    max_probes=args.max_probes,
    max_probes_per_game=args.max_probes_per_game,
)
```

`_select_diverse_admitted_candidates` mutates `audit` in place (appending diversity-drop rows for evicted candidates) and returns the kept candidates in deterministic order.

Module constant in `build_probe_suite.py`:
```python
MIN_PLY_SEPARATION_SAME_GAME = 3
```
Not a CLI flag — see §6.

### 4.2 Pipeline (single round-robin walk)

The selector does one round-robin walk over categories. For each candidate considered, three suppression rules are checked in precedence order; the first match drops the candidate with that reason. Otherwise the candidate is added to the kept set.

#### Stage 1 — Group by category

Bucket admitted candidates into:
- `chain_advantage_central_red`
- `chain_advantage_central_black`
- `chain_advantage_edge_red`
- `chain_advantage_edge_black`

#### Stage 2 — Rank within each category

Sort candidates within a category by this total-order key:

1. `phase1_features.cc_size` desc
2. `phase1_features.axis_span_margin` desc
3. `phase1_features.cc_axis_span` desc
4. `phase2_label.min_top1_share` desc
5. `phase2_label.value_stability` asc
6. Existing `_sort_key`: `(-iter_num, -source_ply, source_game)`

Rationale (settled in design discussion):
- **Structural-first** because the suite tests *future* model iterations against fixed positions; structural features are model-invariant signals of "this position is genuinely a strong-advantage state."
- **`min_top1_share` over `abs(mean_root_value)`** because the latter saturates near 0.99 for admitted candidates (admission floor 0.45, observed cluster near 1.0); `min_top1_share` has wider observed range (0.22–0.74) and tests model decisiveness about *which move* to play, not just position value.
- **Existing `_sort_key` last** ensures total-order determinism without dominating the substantive ranking.

#### Stage 3 — Round-robin walk with suppression rules

Iterate categories in a fixed canonical order:
1. `chain_advantage_central_red`
2. `chain_advantage_central_black`
3. `chain_advantage_edge_red`
4. `chain_advantage_edge_black`

Skip categories with no remaining (un-considered) candidates. Each pass takes the next-best candidate (per Stage-2 rank) from the current category and applies these three suppression rules in order. **The first match wins**, drops the candidate with the corresponding audit reason, and walks the round-robin to the next category.

**Rule A — Near-duplicate.** The candidate is a near-duplicate if any already-kept candidate has all of:
- same `source_game`
- same `category`
- `|Δcc_size| < 2`
- `|Δaxis_span_margin| < 0.05`

Drop with `reason="diversity_near_duplicate"`. `kept_instead_source_ply` = source_ply of the matching keeper (smallest source_ply if multiple match, deterministically).

**Rule B — Ply-too-close.** The candidate is too close to a kept sibling if any already-kept candidate has:
- same `source_game`
- `|Δsource_ply| < MIN_PLY_SEPARATION_SAME_GAME`

(Note: Rule B does not require same category — a game's plies form a single time-series regardless of which category each ply lands in.)

Drop with `reason="diversity_ply_too_close"`. `kept_instead_source_ply` = source_ply of the closest kept sibling (smallest source_ply if multiple are equidistant, deterministically).

**Rule C — Per-game cap.** The candidate exceeds the cap if:
- `count(kept candidates with same source_game) >= max_probes_per_game`

(Counted total across all categories, per §5.2.)

Drop with `reason="diversity_per_game_cap"`. `kept_instead_source_ply` = smallest source_ply among the keepers from that game (deterministic).

**Else: admit.** Add the candidate to the kept set. The audit gets no new row for this candidate (its `reason="admitted"` row is already there from Phase 2).

Continue round-robin passes until either:
- Total kept count == `max_probes`, or
- All categories are exhausted (every candidate considered).

#### Why the rules are sequenced this way

- Rule A (near-duplicate) is the most specific: it identifies a structurally redundant probe, not just one over a count budget. Catching it first gives the most informative audit reason.
- Rule B (ply-too-close) is more specific than the cap: it tells the operator the suite would have benefited from broader separation in the K-window, not just a smaller cap.
- Rule C (per-game cap) is the catch-all: it fires only when the candidate isn't otherwise dropped. This means a game can fill its cap with structurally distinct, well-separated probes — which is the desired behavior.

The duplicate definition's "same category" requirement is deliberate: a same-game pair landing in different categories is treating different parts of the structure (the centroid crossed the cheb=6 → cheb=9 boundary), and the suite explicitly wants to preserve both category buckets. Cross-category same-game pairs are rare in practice (cheb rarely jumps 3+ cells between consecutive plies), but the rule should not silently dedupe them.

Folding ply-separation into the duplicate rule was considered and rejected: keeping them separate gives cleaner audit diagnostics ("dropped because near-duplicate" vs. "dropped because too close in plies").

### 4.3 Determinism

- Stage 2's intra-category sort is total-order (the existing `_sort_key` final tiebreak guarantees this).
- Stage 3's round-robin walks categories in a fixed canonical order, not by yield or alphabetical.
- Within a category, candidates are visited in Stage-2 rank order.
- Suppression-rule checks compare against an insertion-ordered keepers list, so traversal is deterministic.
- `kept_instead_source_ply` always picks the smallest source_ply among matching keepers — deterministic when ties exist.

Same input → byte-identical output, independent of Python dict iteration order.

## 5. Resolved design questions

### 5.1 Why post-Phase-2, not Phase-1?

Phase-2 labels (`min_top1_share`, `value_stability`, `mean_root_value`) are useful for breaking ties among siblings and would be unavailable if the cap moved earlier. Saving Phase-2 MCTS labeling cost on cap-evicted candidates is real (each evicted candidate avoids ~10K MCTS sims × repeats), but at current scale (~600 Phase-1 candidates per run, 50–60% labeled in minutes) the cost is acceptable in exchange for richer selection signals.

### 5.2 Per-game cap is total across categories, not per `(game, category)`

The goal is "no game dominates the suite." A rare cross-category same-game pair (one central, one edge) is genuinely diverse but still consumes both of that game's slots under `max_probes_per_game=2`. This is intentional.

### 5.3 Ply-separation is a fixed module constant, not a CLI flag

`MIN_PLY_SEPARATION_SAME_GAME = 3` is hard-coded. Reasons:
- Smaller API surface; one new flag (`--max-probes-per-game`) is enough.
- Tied to the current K-range `[3, 8]`. With span 5 and separation 3, max admissible same-game pairs = 2, which matches the cap default. If K-range expands later, this becomes a code edit, which is acceptable for a single-operator workflow.
- Avoids exposing a knob with no concrete tuning use case.

### 5.4 Category iteration order is a fixed canonical 4-tuple

Order: `central_red, central_black, edge_red, edge_black`. Skip empty buckets. This is preferred over:
- **Alphabetical** — introduces accidental color/centrality bias.
- **Yield-based** — non-deterministic across runs as yields shift; harder to reason about.
- **Dynamic by category size** — same problem.

The pre-thinning production draft was 21 central_red / 9 central_black / 0 edge — empty edge buckets are the **expected** common case, not exceptional. Stage 3's "skip empty" path is the primary path on current data.

### 5.5 Quality key is structural-first, not value-first

Rejected: `abs(mean_root_value)` desc as the primary discriminator.
- Admitted candidates already passed `magnitude_threshold ≥ 0.45`; observed values cluster near 0.99 with low variance among admitted.
- Sensitive to MCTS labeling noise across re-runs.
- Conflates "model thinks this position is more decisively winning" with "the position is more clearly a strong-advantage state for future evaluation."

Accepted: structural fields first (`cc_size`, `axis_span_margin`, `cc_axis_span`), `min_top1_share` and `value_stability` as Phase-2 tiebreaks, existing `_sort_key` as final determinism guarantee.

### 5.6 Near-duplicate rule includes "same category"

A same-game pair is a near-duplicate **only if also same category**. The category boundary (cheb=6 vs cheb=9 → central vs edge) is part of the structural distinction the suite is meant to preserve, so a cross-category same-game pair should not be deduped even if its `cc_size` and `axis_span_margin` deltas would otherwise qualify.

Cross-category same-game pairs are rare in practice (the centroid rarely jumps 3+ cells between consecutive plies of one game), but the rule should not silently drop them when they do occur.

The "same category" requirement applies only to Rule A (near-duplicate). Rule B (ply-too-close) and Rule C (per-game cap) operate on the whole game irrespective of category.

## 6. CLI surface

Add to `scripts/build_probe_suite.py` argparse (in the strong_advantage block):

```python
ap.add_argument("--max-probes-per-game", type=int, default=2,
                help="Maximum number of admitted probes from any single source game. "
                     "Counts total across all 4 categories. Default 2.")
```

No other new flags. No changes to existing flags.

`meta.selection_rules` in the output payload gains:
```json
{
  "max_probes_per_game": 2,
  "min_ply_separation_same_game": 3,
  "category_iteration_order": [
    "chain_advantage_central_red",
    "chain_advantage_central_black",
    "chain_advantage_edge_red",
    "chain_advantage_edge_black"
  ],
  "diversity_quality_key_order": [
    "phase1_features.cc_size desc",
    "phase1_features.axis_span_margin desc",
    "phase1_features.cc_axis_span desc",
    "phase2_label.min_top1_share desc",
    "phase2_label.value_stability asc",
    "default_sort_key"
  ]
}
```

These keys make the selector configuration self-documenting in the committed file and let an audit-time reviewer verify which version of the policy produced the suite.

## 7. Audit changes

### 7.1 Eliminate the Phase-1 `admitted` row

In `scripts/GPU/alphazero/probe_eval.py:855`, remove the line:

```python
audit.append({**base_audit, "reason": "admitted"})
```

After this change:
- Phase 1 writes audit rows **only** for rejected candidates (the `phase1_*` reasons).
- Phase 2 is the only writer of `reason="admitted"` rows.
- The diversity selector writes `reason="diversity_*"` rows for post-Phase-2 evictions.
- `reason="admitted"` ⇔ probe is in the final committed suite. Exactly one such row per output probe.

This is a minimal change to `probe_eval.py` (one line removed). All Phase-1 *rejection* paths are unchanged.

### 7.2 New diversity drop reasons

| Reason | Trigger | Rule | Extra fields |
|---|---|---|---|
| `diversity_near_duplicate` | Same-game **and same-category** sibling has `\|Δcc_size\| < 2 AND \|Δaxis_span_margin\| < 0.05` | A | `kept_instead_source_ply` |
| `diversity_ply_too_close` | Same-game sibling within `MIN_PLY_SEPARATION_SAME_GAME` plies (any category) | B | `kept_instead_source_ply` |
| `diversity_per_game_cap` | Game already has `max_probes_per_game` keepers (any category) | C | `kept_instead_source_ply` |

All diversity-drop rows carry the full `phase2_label` (since by definition they survived Phase 2).

### 7.3 Audit row schema after this change

Existing fields are unchanged. The post-Phase-2 audit row schema becomes:

```json
{
  "source_game": "iter_NNNN_game_MMM",
  "source_ply": 42,
  "phase1_features": { ... },
  "phase2_label": { ... },
  "reason": "admitted | sign_mismatch | magnitude_below_threshold | low_top1_share | unstable_value | position_already_forced | diversity_near_duplicate | diversity_ply_too_close | diversity_per_game_cap | mcts_error | replay_error",
  "kept_instead_source_ply": 50    // present only for diversity_* reasons
}
```

Phase-1 rejection rows continue to lack `phase2_label` (they were never labeled). Their reasons are unchanged: `phase1_cc_size`, `phase1_axis_span`, `phase1_axis_span_margin`, `phase1_no_goal_touch`, `phase1_already_forced`, `category_midband`.

## 8. Edge cases

- **Empty category:** silently skipped in Stage 3 round-robin. No warning beyond the existing per-category yield warning at `probe_eval.py:866-880`.
- **Single non-empty category:** Stage 3 degenerates to "take top-N from that category" with all suppression rules still applied.
- **Game contributes one near-duplicate pair, then a third structurally distinct ply:** Rule A drops the rank-2 of the duplicate pair; the third ply is kept (subject to Rules B and C).
- **Game has 6 candidates, all near-duplicates of each other (same category):** Rule A keeps only the rank-1; the rest fail Rule A. Game contributes 1 probe.
- **All admitted candidates are near-duplicates of each other across games:** Rule A is per-game-and-category, so cross-game candidates are kept independently. (This is intentional — the goal is per-game diversity, not global feature diversity.)
- **`max_probes` not reached:** emit fewer probes. No additional warning beyond the existing yield warning. Operator broadens `--source-iter-range` if needed.
- **`max_probes_per_game` higher than admitted-per-game:** no-op for that game; Rule C doesn't bind.
- **Same-game probes in different categories:** counted under one game-cap budget. If `max_probes_per_game=2` and a game has one admitted central + one admitted edge, both are eligible; Rule A does not dedupe them (different categories); Rule B may still drop one if their plies are too close; Rule C lets both through if Rules A and B did.

## 9. Test plan

Tests live in `tests/test_build_probe_suite.py` (extend existing or add new file). Each constructs a synthetic `admitted` list of candidates with controlled features and asserts on the selector's output and audit deltas.

1. **`test_per_game_cap`** — 5 probes from one game (no near-dupes, well-separated plies), 1 probe from another. `max_probes_per_game=2`. Assert: 2 from clustered game survive (the rank-top 2), 3 dropped with `diversity_per_game_cap`, the 1 from the other game survives.

2. **`test_near_duplicate_suppression`** — 3 same-game **same-category** probes with `cc_size = (20, 21, 25)` and `axis_span_margin = (0.20, 0.21, 0.40)`. Assert: probes with `cc_size=20` and `cc_size=21` are duplicates; the rank-2 of those is dropped with `diversity_near_duplicate` and `kept_instead_source_ply` set; `cc_size=25` is kept as structurally distinct.

3. **`test_ply_separation`** — 3 same-game probes, structurally distinct (no near-dupes), `source_ply ∈ {50, 51, 54}`, `max_probes_per_game=3` (so cap doesn't bind). Assert: 50 and 54 kept; 51 dropped with `diversity_ply_too_close` and `kept_instead_source_ply=50`.

4. **`test_category_round_robin_canonical_order`** — admitted across 4 categories with counts (4, 4, 4, 4), `max_probes=8`. Assert: output contains 2 from each of central_red and central_black (chosen by canonical order), 2 from each of edge_red and edge_black if `max_probes` allows; iteration order is the canonical 4-tuple.

5. **`test_sparse_category_backfill`** — only `central_red` populated (10 candidates from 5 games), all others empty. `max_probes=10`, `max_probes_per_game=2`. Assert: all 10 kept (2 per game × 5 games); no error; round-robin gracefully skips empties.

6. **`test_edge_categories_empty`** — `central_red` and `central_black` populated, `edge_*` empty. Assert: round-robin alternates central_red ↔ central_black; output respects `max_probes`; no spurious warnings about edge categories beyond the existing yield warning.

7. **`test_drop_reason_precedence`** — synthetic candidate that triggers both `diversity_near_duplicate` AND `diversity_per_game_cap`. Assert: audit reason is `diversity_near_duplicate` (precedence rule).

8. **`test_determinism_under_input_shuffle`** — call selector twice with the same admitted list in different orders. Assert: byte-identical output payloads (including audit row order, modulo selector-appended rows which should themselves be deterministic).

9. **`test_audit_admitted_count_canonical`** — synthetic input through the full pipeline (Phase 1 + Phase 2 + selector). Assert: `count(audit rows with reason="admitted") == count(probes in payload)` (no double-counting after the §7.1 fix).

10. **`test_audit_kept_instead_field`** — diversity-dropped rows carry `kept_instead_source_ply` set to a real keeper's `source_ply`; regular admitted/Phase-2 rejection rows do not have this field.

11. **`test_quality_key_structural_priority`** — two same-game same-category candidates: one with higher `cc_size` but lower `min_top1_share`, the other with lower `cc_size` but higher `min_top1_share`, structurally far enough apart that the near-duplicate rule does not fire (e.g., `cc_size = (15, 25)`). With `max_probes_per_game=1`, assert the higher `cc_size` wins (structural beats Phase-2 fields).

12. **`test_cross_category_same_game_not_deduped`** — two same-game probes that satisfy `|Δcc_size| < 2 AND |Δaxis_span_margin| < 0.05` BUT land in different categories (one central, one edge). Assert: both are kept (Rule A's same-category requirement prevents dedupe); per-game cap allows both since `max_probes_per_game=2`.

13. **`test_meta_selection_rules_recorded`** — output payload's `meta.selection_rules` includes `max_probes_per_game`, `min_ply_separation_same_game`, `category_iteration_order`, `diversity_quality_key_order` with the expected values.

## 10. Regeneration plan for the committed suite

The current committed `tests/probes/strong_advantage_probes.json` is a hand-thinned 22-probe snapshot. It will be regenerated under the new selector as part of this change.

Steps (executed by the operator, not the implementer):

1. Implement selector + audit cleanup; tests pass.
2. Re-run the original generator invocation on the same source range and label checkpoint:
   ```
   scripts/build_probe_suite.py --tier strong_advantage \
       --source-iter-range 57 58 \
       --label-checkpoint <same path as original promotion> \
       --force
   ```
   Record the exact invocation in the regeneration commit message.
3. Inspect the new draft:
   - Per-category counts (`central_red / central_black / edge_red / edge_black`).
   - Per-game distribution (target: max 2 per game, well-separated plies).
   - Audit drop reasons (sanity-check that `diversity_*` reasons fire as expected).
4. Light review pass (operator), same criteria as the original promotion. No formal gate review required — this remains a `bootstrap_rule_selected` tier.
5. Promote: `scripts/build_probe_suite.py --tier strong_advantage --promote --reviewer <name> --force`.
6. Commit the regenerated `strong_advantage_probes.json` with a message that explicitly states:
   - The previous committed file was a manually thinned snapshot.
   - The new file is reproducible from the recorded invocation under the diversity-aware selector.
   - Source range and label checkpoint are unchanged.
   - The hand-thinned file is preserved in git history for provenance.

This is a controlled regeneration, not a magic match. The new file will almost certainly contain different probes than the old hand-thinned one; that is expected and acceptable for a `bootstrap_rule_selected` light-reviewed artifact.

## 11. Backward compatibility

- **Existing draft → promote workflow** unchanged.
- **Existing CLI flags** unchanged. One new optional flag (`--max-probes-per-game`) with a default that does not require operator intervention.
- **`forced` tier** completely unaffected (separate code path).
- **Audit consumers:** the only audit consumer today is the operator inspecting the JSON manually. New `reason` values are additive; existing values that survive (`admitted`, all Phase-1 rejections, all Phase-2 rejections) keep their meaning. The semantic shift is that `reason="admitted"` now has cleaner meaning (no double-counting).
- **`meta.selection_rules`** is an open dict; new keys are non-breaking.
- **Trainer-side probe scoring** does not parse the audit and is unaffected.

## 12. Implementation scope

Rough LOC estimates:

- `scripts/build_probe_suite.py`: ~80 LOC for the new `_select_diverse_admitted_candidates` helper, ~5 LOC for the argparse addition and the call site, ~10 LOC for `meta.selection_rules` enrichment.
- `scripts/GPU/alphazero/probe_eval.py`: 1 LOC removed (the Phase-1 `admitted` audit append).
- `tests/test_build_probe_suite.py`: ~250 LOC for the 12 unit tests.
- `docs/probe-suite-generation.md`: ~30 LOC documenting the new flag, the diversity rules, the audit reasons, and the regeneration that occurred.

Total: ~375 LOC change, ~50% of which is tests.

This is intentionally small. The architecture decisions here are load-bearing; the code surface is not.
