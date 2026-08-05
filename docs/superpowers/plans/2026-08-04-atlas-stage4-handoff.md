# Atlas Stage 4 — Delivered Interfaces and Qualification (Stage 5 Handoff)

**Status:** Stage 4 LANDED and QUALIFIED, 2026-08-04, at clean tree `f375136`.
**No reservoir generated, no checkpoint loaded, no MLX executed, no measurement run.**

**Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §3, §5, §6,
§6a (amendments 3 and 4), §7, §8 — EXECUTION-FROZEN.
**Plan:** `2026-08-04-atlas-stage4-readouts.md`, revision 4.

## Qualification

```text
full suite   2556 passed / 4 skipped / 53 deselected / 0 failed   (REAL_EXIT=0)
             7m10s, exit code read from the process, never a pipe
Stage 4 new  119 tests
```

### The delta arithmetic, and the baseline correction it forced

The prediction was 2554, from a Stage 3 baseline of 2435. The measured total is 2556.
**The +2 is entirely in the baseline.** Commit `5dbcfe9` ("event-weighted excluded mass
in the shipped tracer") landed *after* Stage 3's qualification run and added two test
functions to `tests/test_selection_tracer.py` (10 → 12). Collecting at `b85ae19` — the
last commit before any Stage 4 code — gives 2441 selected = **2437 passed + 4 skipped**.

```text
2437  true pre-Stage-4 baseline
+119  Stage 4, recounted from `def test_` on disk
=2556 measured
```

**The delta is exactly 119, so no pre-existing test changed behaviour.** The check
earned its keep here: the arithmetic failed first against a stale baseline, and running
it down found a gap in the record rather than a defect in the work. **Stage 5 must
baseline against 2556, measured, not against any number quoted in a document.**

| file | tests |
|---|---:|
| `tests/test_atlas_producer_closure.py` | 13 |
| `tests/test_atlas_labelling.py` | 10 |
| `tests/test_atlas_readout_a.py` | 27 |
| `tests/test_atlas_readout_b.py` | 14 |
| `tests/test_atlas_readout_c.py` | 30 |
| `tests/test_atlas_artifact.py` | 14 |
| `tests/test_atlas_readout_chain.py` | 11 |
| **total** | **119** |

### Standing invariants, verified

- `git diff b85ae19..HEAD -- mcts.py` is **empty**. Stage 1's scoped observer exception
  already delivered every hook; Stage 4 added nothing to it.
- No Stage 4 module imports MLX. Every test uses synthetic dicts or `FakeEvaluator` at
  `active_size=6`.
- `tests/test_build_atlas_corpus_cli.py` is unchanged at 13, so the additive `_jsonable`
  branch is byte-identical for every pre-existing caller.

## Delivered interfaces

### `warm_prefix_replay.py` — Task 0 additions

```python
capture_tree_state(root) -> dict          # D3, root_visits, leader_breadth, entropy, ...
capture_parent_visits(root, max_depth=2) -> dict[tuple[int, ...], int]
deep_reference_line(root) -> {"edges": [...], "moves": [...]}
DEPTH_NAMES = ("root", "reply", "two_ply")
merge_reference_lines(line_3200, line_6400) -> {"required_edges": [...], "agreement": {...}}
check_backup_invariant(d3_start, d3_boundary, n_actual) -> bool   # raises on violation
```

An **edge** is `{parent_path, move, depth, parent_priors}` plus `sources` after merging.
`agreement[depth]["state"]` is one of `agree` / `disagree` / `single_line` /
`absent_both`; the last two are outside the denominator and are counted separately.

`BatchSafeBoundaryObserver` now also freezes `capture_at_boundary` and
`parent_visits_at_boundary` at its own instant. `run_additive_ladder`'s `snapshots`
keeps its Stage 3 keys and gains:

```python
{"at_boundary": ..., "at_400": ...,                       # Stage 3, unchanged
 "captures":       {"at_start", "at_boundary", "at_400"},
 "parent_visits":  {"at_boundary", "at_400"},
 "reference_lines": {"at_3200", "at_6400", "merged"}}
```

Deep rungs are selected by **leg index** (`leg_idx == 2` and `3`), never `running_B`.

### `selection_tracer.py`

`BATCH_LAG = 14`, and every cell carries `lagged_first_touch_outside_events`, counted
online in the same pass as the unlagged one.

### The four pure modules

```python
atlas_labelling:  stable_reference, classify_row, class_counts, size_from_pilot,
                  final_capacity_gate
atlas_readout_a:  FEATURE_NAMES, collect_features, LABEL_TO_Y, prepare_rows,
                  standardize, fit_ridge_logistic, auc, bootstrap_auc_lower_bound,
                  evaluate_detector, deployability,
                  FEATURE_SETS, AUTHORITATIVE_FEATURE_SET, INSUFFICIENCY_VERDICTS,
                  evaluate_detector_both
atlas_readout_b:  gate_triggers, closes_half, convergent, compound_narrowing,
                  calibrate_gate, natural_convergence_report, by_stratum_summary
atlas_readout_c:  INSTANTS, GATING_INSTANT, STRATA, STABLE_REFERENCE_LABELS,
                  REQUIRED_RATES, static_retention, edge_retention,
                  intervention_from_snapshots, classify_strata, classify_edge_strata,
                  aggregate_shape, validation_verdict, select_shape,
                  select_on_discovery_validate_on_selected
atlas_artifact:   ROW_SCHEMA_VERSION, build_row, validate_provenance, emit
```

### The row is one object

`build_row(...)` returns a dict that is **simultaneously** a Read-out A row (`label`,
`features_at_boundary`, `features_at_400`), a Read-out B row (`legs`, `phase`,
`flat_policy`, `near_even`) and a Read-out C row (`snapshots`, `label`, `phase`,
`flat_policy`, `near_even`). There is no translation layer between producer and
consumers, and the chain test drives all three read-outs from a real `build_row` result.

`legs` and `boundary` hold **dataclass objects**, not flattened dicts, because Read-out B
and `atlas_labelling` address them by attribute. `_jsonable` converts dataclasses and
tuple keys at the JSON boundary; the root path `()` emits as the empty-string key `""`.

`emit` refuses any run whose provenance does not validate, checking hexadecimal digests
rather than length alone, and validates *before* serializing so a payload defect still
raises `TypeError`.

## Invariants Stage 5 may rely on

- **Retention never reads a 6,400-era visit count.** The producer does not emit one, so
  the defect is unreachable rather than merely discouraged.
- **Floors pass at both instants or not at all**: the hoisted retention numbers are the
  worse of `at_boundary` / `at_400`, and `None` if either is undefined.
- **The bars are gated on `at_400`**; the boundary intervention is reported beside it.
- **Only stable-reference-eligible rows feed retention**, via the allow-list
  `STABLE_REFERENCE_LABELS`. Selection-event counters still cover every row, and
  `retention_rows` / `rows_without_stable_reference` say how many were set aside.
- **Cohort counters pool by summing** numerators and denominators, never by averaging
  per-row rates.
- **Undefined is `None` everywhere** — every rate, median, quartile and boolean gate.
- **`LATE_ONLY_SEPARATION` needs boundary `FAIL` + `B=400` `PASS` + zero boundary
  missing-feature rejections in either split.** A blocked lateness reading reports as the
  boundary's own `FAIL` with `lateness_blocked_by` set.

## What Stage 5 must still close

1. **`flat_policy` and `near_even` are SUPPLIED, not derived.** Stage 4 accepts both as
   caller-supplied booleans and every test hardcodes them, so the one seam Stage 4
   cannot qualify is the one that computes them. Read-out B's `flat_policy` / `near_even`
   strata and Read-out C's `root_flat` / `near_even` strata are only as good as that
   unqualified producer. Derive from already-frozen measured fields and qualify the
   producer → `build_row` seam:

   | field | frozen definition | available from |
   |---|---|---|
   | `flat_policy` | normalized policy entropy ≥ `0.90` **and** top prior ≤ `0.025` (§8) | `captures[*]["policy_entropy"]` and the root priors — reuse `atlas_readout_c._is_flat`, do not write a second predicate |
   | `near_even` | `\|V_stm\| ≤ 0.30` (§8) | `LegResult.root_value` at nominal `B = 400` |
   | `phase` | ply bounds 0–30 / 31–60 / 61–90 / 91+ (§3) | `corpus_geometry.phase_for_ply(target_ply)` |

2. **Scale.** Everything is qualified at `active_size=6`, `FakeEvaluator`, 3–5-ply
   prefixes. Throughput at board 24 against a real checkpoint is unmeasured, and at
   `active_size=6` the admitted set clamps to `n_legal`, so no Read-out C *number* from
   the chain test means anything — only that the path runs.
3. **The three distribution gaps remain operator/pilot measurements**: real-scale
   throughput, the `remaining` distribution (§4's preregistered deployability rule fails
   Read-out A if the median is zero), and the inheritance-reset rate.
4. **The end-to-end runner does not exist.** There is no module that walks a corpus row
   list, drives prefix → ladder → read-outs → artifact, and writes the run document.
   Stage 4 delivered the parts; nothing composes them over a corpus.

## Standing authorization boundary

Corpus generation and every GPU measurement run remain **unauthorized**. Stage 5 is
planned only against these interfaces once they are accepted, and the pilot is a
separate written authorization after Stage 5 qualifies.
