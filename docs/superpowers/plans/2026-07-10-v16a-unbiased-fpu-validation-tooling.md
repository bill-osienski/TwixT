# v16a Stratified, Game-Held-Out, Non-Selected FPU Collateral-Damage Screen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling + manifest to *screen* whether the **frozen** candidate `MCTSConfig.fpu_value = -0.20` (selected on the A discovery set) causes obvious collateral damage — degenerate collapse, top-move churn, search-shape distortion, wrong-signed mover-value shifts — on ordinary positions from games that were **never** part of A discovery. Capability only; the FPU sweep is **not** run.

**Hierarchy (do not conflate).**
- **Root goal:** a stronger 400-sim MCTS and better self-play search targets.
- **Decisive benchmark:** a statistically significant, same-checkpoint, equal-400-sim, balanced-color head-to-head strength gain (FPU off vs ship-form).
- **v16a:** *only* a held-out collateral-damage screen for the frozen `-0.20` candidate. Passing it is a necessary gate to proceed toward B/C/D and the strength match — it is **not** evidence of improvement.

**Architecture:** `diagnose_fpu_sweep.py` already reconstructs positions through the generic trusted path (`load_csv_manifest` → `search_for_row` → `position_state`) and touches no A-specific case field except `case_id`, so no reconstruction change is needed. Work is: (a) a config-resolution layer (CLI aliases; conditional integrity; **frozen-protocol fpu defaults**; mode-scoped output paths; strict mode detection); (b) a delta-based, mover-perspective, search-shape output selected automatically for neutral manifests. A new builder samples a deterministic, game-held-out, stratified, side-balanced, spaced set of positions.

**Tech Stack:** Python 3 stdlib only (argparse, csv, json, random, dataclasses, statistics, math). No numpy. pytest via the repo interpreter `.venv/bin/python -m pytest`. No MLX/GPU/MCTS in unit tests.

## Global Constraints

- **Do NOT modify** `mcts.py` (beyond the shipped FPU hook — i.e. not at all), `trainer.py`, `network.py`, calibration manifests, value-adapter/projection code, prior-pruning, top-k pruning, promotion logic, or `SIMS_TABLE`.
- **Reconstruct positions ONLY through the trusted path** (`load_csv_manifest` → `search_for_row` → `position_state`). No second reconstruction path.
- **`fpu_value` default stays `0.0`, byte-identical off**; configs built only via `dataclasses.replace`.
- **Selected-A path is byte-identical** for a manifest without `ply_bucket` (legacy mode): identical case + summary CSV bytes, identical integrity behavior (a bare legacy run still checks against the default Phase-0 CSV). Proven by golden-bytes tests + an old-vs-new diff, not by unchanged constants.
- **Mover (side-to-move) perspective is the PRIMARY value signal.** Black-perspective deltas cancel across colors on a mixed set and would falsely read as harmless. Report both; lead with mover.
- **Game-level holdout:** exclude *entire games* named in the A manifest (default), sourced from the single canonical `DEFAULT_A_MANIFEST` constant. Discovery and validation share no games.
- **Frozen protocol is enforced in code:** a neutral run defaults `--fpu-values` to `0.0,-0.20`; any other values on a neutral manifest require `--allow-non-protocol-fpu` and print a warning. The tool stays generic; the safe protocol is the default.
- **Neutral runs never write into the A output directory:** output paths resolve beside the manifest in neutral mode.
- **Winner-null games are INCLUDED by default** labeled `game_result="unknown"` (measured: all 19 are `state_cap` 280-ply marathons that reconstruct cleanly — the most search-stressed, highest-value late samples). A winner-null row that fails reconstruction is dropped + logged; a winner-having row that fails is a hard error.
- **No A/drop-derived labels** in neutral rows.
- **Determinism:** fixed seed → byte-identical manifest.
- **Do NOT run the FPU sweep** in v16a. The design note must not contain an executable next-run command.
- Tests pure where possible; no MCTS in unit tests. Run from repo root with `.venv/bin/python -m pytest`.

---

## Design decisions (with measured data)

**Naming.** Manifest uses canonical loader names (`game_idx`, `position_ply`, `replay_path`); the spec's `game_id`/`ply`/`top_move` appear as *sweep-output* column names (mapped at write time). Mode auto-detected via `ply_bucket`.

**Stratified, game-held-out, non-selected — not "unbiased."** Equal bucket quotas are deliberately not the natural distribution.

**Measured corpus** (`calib020_0001_vs_0379_800g_w4_seed20115_replay_games.jsonl`, 800 games): 30 A-games excluded, **770 held-out** (results: red 421 / black 330 / unknown 19). Sampling (cap 2, min-gap 8, round-robin, side-balanced):
- opening (1–15): 40/40, sides 20/20.
- early-mid (16–40): 100/100, sides 50/50.
- midgame (41–90): 100/100, sides 50/50.
- **late (91+): 84/100, sides 42/42, across 48 games — 38 of them from the 19 `state_cap` 280-ply marathons.** (Without the marathons it is 47.) The quota is a ceiling; the 84/100 shortfall is logged (data-limited, not a bug).
- Total ≈ 324 rows.

**Single top-share denominator.** Top-1 share and the `>=0.95` collapse flag both use `top.visit_count / root.visit_count` (continuity with the selected-A diagnostic); entropy normalizes over the child-visit total. There is one root top-share column (`top_child_visit_share`); the redundant `root_top1_visit_share` is not emitted.

**Search-shape deltas are paired, not levels.** The screen reports change-vs-fpu-0 for entropy, effective-children (`exp(entropy)`), visited-children, reply count (all + stable-top), and top-share, plus `new_collapse`/`resolved_collapse` accounting — because breadth count alone can stay flat while concentration moves (the c_puct result), and a lower reply count could merely reflect a different (smaller-subtree) root move.

**Opening dedup is prefix dedup.** `opening_prefix_key` hashes the ordered move prefix (TwixT has no captures, so identical prefixes reach identical states). Transpositions (same pegs via different order) are not merged — named "opening-prefix dedup," not "board-state dedup."

---

## File structure

- **Modify** `scripts/GPU/alphazero/diagnose_fpu_sweep.py`
- **Create** `scripts/GPU/alphazero/build_v16a_neutral_position_manifest.py`
- **Create** `tests/test_fpu_sweep_v16a.py`, `tests/test_v16a_neutral_manifest.py`
- **Create** `tests/golden/fpu_sweep_legacy_cases.csv`, `tests/golden/fpu_sweep_legacy_summary.csv` (generated + committed in Task 4)
- **Create** `logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv` (+ `.meta.json`) — Task 7, force-added
- **Create** `docs/superpowers/specs/2026-07-10-v16a-unbiased-fpu-validation-design.md` — Task 7

---

## Task 1: Config-resolution layer (CLI, integrity, frozen fpu, output paths, strict mode)

Covers spec §1, the arg/behavior half of §3, and review points 1, 2, 3(mode), 4. Adds pure resolvers + strict mode detection + the CLI, and wires them into `main()`'s front matter. The compute/write body is finalized in Task 4; between here and Task 4 the module imports and all unit tests pass (no test calls `main()` until Task 4).

**Files:** Modify `diagnose_fpu_sweep.py`; Test `tests/test_fpu_sweep_v16a.py` (create).

**Interfaces (produces):** `manifest_is_neutral(cases) -> bool` (strict); `resolve_integrity_csv(integrity_csv, skip, neutral, default_csv) -> str|None`; `resolve_fpu_values(fpu_values_arg, neutral, allow_non_protocol) -> list[float]`; `resolve_output_paths(out, summary_out, strata_out, manifest, neutral) -> tuple[str,str,str]`; `_parse_fpu_list(s) -> list[float]`; constants `PROTOCOL_FPUS = [0.0, -0.20]`, `DEFAULT_STRATA_SUMMARY_OUT`; `_parse_args` with `manifest`, `integrity_csv` (None), `skip_integrity_check`, `fpu_values` (None), `allow_non_protocol_fpu`, `out`/`summary_out`/`strata_summary_out` (None).

- [ ] **Step 1: Write the failing tests** — create `tests/test_fpu_sweep_v16a.py`:

```python
import pytest
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    _parse_args, manifest_is_neutral, resolve_integrity_csv, resolve_fpu_values,
    resolve_output_paths, PROTOCOL_FPUS, DEFAULT_A_MANIFEST, DEFAULT_PHASE0_CSV,
    DEFAULT_OUT, DEFAULT_FPUS)


def test_cli_aliases_and_none_sentinels():
    assert _parse_args(["--manifest", "m"]).manifest == "m"
    assert _parse_args(["--a-manifest", "m"]).manifest == "m"
    assert _parse_args(["--integrity-csv", "i"]).integrity_csv == "i"
    assert _parse_args(["--phase0-csv", "i"]).integrity_csv == "i"
    a = _parse_args([])
    assert a.integrity_csv is None and a.fpu_values is None
    assert a.out is None and a.summary_out is None and a.strata_summary_out is None
    assert a.skip_integrity_check is False and a.allow_non_protocol_fpu is False


def test_manifest_is_neutral_is_strict():
    assert manifest_is_neutral([{"ply_bucket": "midgame"}, {"ply_bucket": "late"}]) is True
    assert manifest_is_neutral([{"case_id": "c"}, {"case_id": "d"}]) is False
    with pytest.raises(ValueError):
        manifest_is_neutral([{"ply_bucket": "late"}, {"case_id": "d"}])   # mixed
    with pytest.raises(ValueError):
        manifest_is_neutral([])                                           # empty


def test_resolve_integrity_conditional():
    assert resolve_integrity_csv(None, False, False, DEFAULT_PHASE0_CSV) == DEFAULT_PHASE0_CSV
    assert resolve_integrity_csv(None, False, True, DEFAULT_PHASE0_CSV) is None
    assert resolve_integrity_csv("x", False, True, DEFAULT_PHASE0_CSV) == "x"
    assert resolve_integrity_csv(None, True, False, DEFAULT_PHASE0_CSV) is None


def test_resolve_fpu_values_frozen_protocol():
    assert resolve_fpu_values(None, True, False) == PROTOCOL_FPUS       # neutral default
    assert resolve_fpu_values(None, False, False) == [float(x) for x in DEFAULT_FPUS.split(",")]
    assert resolve_fpu_values("0.0,-0.20", True, False) == [0.0, -0.20]
    with pytest.raises(SystemExit):
        resolve_fpu_values("0.0,-0.10,-0.20", True, False)             # non-protocol, no override
    assert resolve_fpu_values("0.0,-0.10,-0.20", True, True) == [0.0, -0.10, -0.20]
    with pytest.raises(SystemExit):
        resolve_fpu_values("-0.20", True, True)                        # missing baseline 0.0


def test_resolve_output_paths_mode_scoped():
    lo = resolve_output_paths(None, None, None, "a/x.csv", False)
    assert lo[0] == DEFAULT_OUT                                         # legacy -> A defaults
    ne = resolve_output_paths(None, None, None, "logs/eval/v16a/m.csv", True)
    assert ne == ("logs/eval/v16a/neutral_fpu_sweep_cases.csv",
                  "logs/eval/v16a/neutral_fpu_sweep_summary.csv",
                  "logs/eval/v16a/neutral_fpu_sweep_by_stratum.csv")
    assert resolve_output_paths("o", "s", "t", "m.csv", True) == ("o", "s", "t")  # explicit wins
```

- [ ] **Step 2: Run tests to verify they fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py -q`. Expected: FAIL (import errors).

- [ ] **Step 3: Add constants + resolvers.** In `diagnose_fpu_sweep.py`, add `from pathlib import Path` if not present, and near the defaults:

```python
DEFAULT_STRATA_SUMMARY_OUT = "logs/eval/fpu_check/a_predrop_fpu_sweep_summary_by_stratum.csv"
PROTOCOL_FPUS = [0.0, -0.20]   # frozen v16a validation candidate set (0.0 = control)


def manifest_is_neutral(cases) -> bool:
    """Neutral / v16a iff EVERY row has a non-empty ply_bucket; legacy iff none
    do. A mixed or empty manifest is a construction error -> raise (a first-row
    check would silently misclassify a malformed manifest)."""
    if not cases:
        raise ValueError("empty manifest: cannot determine legacy/neutral mode")
    flags = [bool(c.get("ply_bucket")) for c in cases]
    if all(flags):
        return True
    if not any(flags):
        return False
    raise ValueError(
        f"mixed manifest: {sum(flags)}/{len(flags)} rows carry ply_bucket; "
        "all-or-none is required")


def resolve_integrity_csv(integrity_csv, skip, neutral, default_csv):
    """fpu=0.0 exact-reproduction baseline, or None to skip. --skip wins; explicit
    csv next; neutral-unspecified skips; legacy-unspecified uses default_csv (so a
    bare legacy run still checks)."""
    if skip:
        return None
    if integrity_csv:
        return integrity_csv
    return None if neutral else default_csv


def _parse_fpu_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def resolve_fpu_values(fpu_values_arg, neutral, allow_non_protocol):
    """Frozen-protocol default: a neutral run with no explicit values uses
    PROTOCOL_FPUS (0.0, -0.20). Any other value set on a neutral manifest needs
    --allow-non-protocol-fpu (screening extra candidates on the holdout is tuning
    on the holdout). 0.0 must be present (delta baseline + integrity)."""
    if fpu_values_arg is None:
        values = list(PROTOCOL_FPUS) if neutral else _parse_fpu_list(DEFAULT_FPUS)
    else:
        values = _parse_fpu_list(fpu_values_arg)
    if BASELINE_FPU not in values:
        raise SystemExit(f"--fpu-values must include the baseline {BASELINE_FPU}")
    if neutral and set(values) != set(PROTOCOL_FPUS) and not allow_non_protocol:
        raise SystemExit(
            f"neutral (held-out) manifest: the frozen v16a protocol is "
            f"{PROTOCOL_FPUS} only. Screening other values on the holdout is "
            f"tuning on the holdout. Pass --allow-non-protocol-fpu to override.")
    return values


def resolve_output_paths(out, summary_out, strata_out, manifest, neutral):
    """Legacy -> the exact A defaults. Neutral -> beside the manifest, so held-out
    results never land in the selected-A directory. Explicit paths always win."""
    if not neutral:
        return (out or DEFAULT_OUT, summary_out or DEFAULT_SUMMARY_OUT,
                strata_out or DEFAULT_STRATA_SUMMARY_OUT)
    base = Path(manifest).parent
    return (out or str(base / "neutral_fpu_sweep_cases.csv"),
            summary_out or str(base / "neutral_fpu_sweep_summary.csv"),
            strata_out or str(base / "neutral_fpu_sweep_by_stratum.csv"))
```

- [ ] **Step 4: Rewrite `_parse_args`** (keep description) so args read:

```python
    ap.add_argument("--manifest", "--a-manifest", dest="manifest",
                    default=DEFAULT_A_MANIFEST)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--integrity-csv", "--phase0-csv", dest="integrity_csv",
                    default=None)
    ap.add_argument("--skip-integrity-check", action="store_true")
    ap.add_argument("--fpu-values", default=None,
                    help="comma list; neutral manifests default to the frozen "
                         "protocol 0.0,-0.20.")
    ap.add_argument("--allow-non-protocol-fpu", action="store_true",
                    help="permit non-protocol --fpu-values on a neutral manifest.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--strata-summary-out", default=None)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--limit-cases", type=int, default=None)
    return ap.parse_args(argv)
```

- [ ] **Step 5: Wire `main()` front matter** (leave the existing compute/write body; Task 4 replaces it). Replace the top of `main()` (manifest load through `search_fns = …`) with:

```python
    args = _parse_args(argv)
    cases = load_csv_manifest(args.manifest)["cases"]
    if args.limit_cases is not None:
        cases = cases[:args.limit_cases]
    neutral = manifest_is_neutral(cases)
    fpus = resolve_fpu_values(args.fpu_values, neutral, args.allow_non_protocol_fpu)
    if neutral and set(fpus) != set(PROTOCOL_FPUS):
        print(f"[fpu] WARNING: non-protocol values {fpus} on a held-out manifest "
              f"(--allow-non-protocol-fpu); this is not the frozen v16a protocol.")
    out_path, summary_path, strata_path = resolve_output_paths(
        args.out, args.summary_out, args.strata_summary_out, args.manifest, neutral)
    resolved_integrity = resolve_integrity_csv(
        args.integrity_csv, args.skip_integrity_check, neutral, DEFAULT_PHASE0_CSV)
    run_integrity = resolved_integrity is not None
    baseline = _phase0_baseline(resolved_integrity) if run_integrity else {}
    if not run_integrity:
        why = ("--skip-integrity-check" if args.skip_integrity_check
               else "neutral manifest, no baseline" if neutral else "no baseline")
        print(f"[fpu] integrity check SKIPPED ({why}); fpu=0.0 remains the delta baseline.")
    search_fns = _search_fns(args.checkpoint, fpus, args.eval_batch_size,
                             args.stall_flush_sims)
```

In the existing body below, guard both integrity sites with `run_integrity and x == BASELINE_FPU:`, replace `args.phase0_csv` with `resolved_integrity` in messages, and change the two final write calls to use `out_path` / `summary_path` instead of `args.out` / `args.summary_out`. (This body is superseded in Task 4.)

- [ ] **Step 6: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py -q`. Expected: PASS. And legacy: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep.py -q`. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_fpu_sweep.py tests/test_fpu_sweep_v16a.py
git commit -m "feat(fpu-sweep): config-resolution layer — aliases, conditional integrity, frozen fpu, mode-scoped outputs, strict mode"
```

---

## Task 2: Mover-perspective value + search-shape metrics + delta enrichment

Covers spec §6 and review points 3, 5, 6. Per-position: mover-perspective root value, visit entropy, effective children, single-denominator top share, collapse flag. Enrichment: mover + black value deltas, all paired search-shape deltas, and `new_collapse`/`resolved_collapse`.

**Files:** Modify `diagnose_fpu_sweep.py`; Test `tests/test_fpu_sweep_v16a.py`.

**Interfaces (produces):** `visit_entropy(visit_counts) -> float`; `_num_delta(a, b)`; `enrich_with_deltas(rows) -> list[dict]`; constant `GENERIC_CASE_FIELDNAMES`. Rich-row keys consumed: `fpu_value, case_id, root_mcts_stm_value, root_mcts_black_value, top_child_move, root_n_visited_children, top_child_n_visited_children, top_child_visit_share, root_effective_children, root_visit_entropy, root_collapsed_ge_0_95`. Keys added by enrich: `root_value_delta_stm_vs_fpu0, root_value_delta_black_vs_fpu0, top_move_changed_vs_fpu0, root_children_delta_vs_fpu0, top_child_children_delta_vs_fpu0, top_child_visit_share_delta_vs_fpu0, root_effective_children_delta_vs_fpu0, root_visit_entropy_delta_vs_fpu0, new_collapse_vs_fpu0, resolved_collapse_vs_fpu0`.

- [ ] **Step 1: Write the failing tests** — append:

```python
import math
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    visit_entropy, enrich_with_deltas, GENERIC_CASE_FIELDNAMES)


def test_visit_entropy():
    assert abs(visit_entropy([5, 5, 5, 5]) - math.log(4)) < 1e-12
    assert visit_entropy([10]) == 0.0 and visit_entropy([]) == 0.0


def _rich(fpu, cid, stm, blk, top, rootc, topc, share, eff, ent, col):
    return {"fpu_value": fpu, "case_id": cid, "root_mcts_stm_value": stm,
            "root_mcts_black_value": blk, "top_child_move": top,
            "root_n_visited_children": rootc, "top_child_n_visited_children": topc,
            "top_child_visit_share": share, "root_effective_children": eff,
            "root_visit_entropy": ent, "root_collapsed_ge_0_95": col}


def test_enrich_mover_black_shape_and_collapse_deltas():
    rows = [_rich(0.0, "A", 0.20, -0.20, "3:4", 5, 200, 0.60, 6.0, 1.8, False),
            _rich(-0.2, "A", 0.05, -0.05, "3:4", 8, 120, 0.97, 2.0, 0.3, True)]
    enrich_with_deltas(rows)
    c = rows[1]
    assert abs(c["root_value_delta_stm_vs_fpu0"] - (-0.15)) < 1e-12
    assert abs(c["root_value_delta_black_vs_fpu0"] - 0.15) < 1e-12
    assert c["root_children_delta_vs_fpu0"] == 3
    assert c["top_child_children_delta_vs_fpu0"] == -80
    assert abs(c["root_effective_children_delta_vs_fpu0"] - (-4.0)) < 1e-12
    assert abs(c["root_visit_entropy_delta_vs_fpu0"] - (-1.5)) < 1e-12
    assert c["new_collapse_vs_fpu0"] is True and c["resolved_collapse_vs_fpu0"] is False
    assert rows[0]["new_collapse_vs_fpu0"] is False


def test_enrich_resolved_collapse_and_blank_share():
    rows = [_rich(0.0, "A", 0.2, -0.2, "", 5, 0, "", 1.0, 0.0, True),
            _rich(-0.2, "A", 0.2, -0.2, "9:9", 5, 200, 0.6, 3.0, 1.0, False)]
    enrich_with_deltas(rows)
    assert rows[1]["resolved_collapse_vs_fpu0"] is True
    assert rows[1]["top_move_changed_vs_fpu0"] is True
    assert rows[1]["top_child_visit_share_delta_vs_fpu0"] == ""    # baseline blank


def test_generic_case_fieldnames_no_redundant_top1_share():
    assert "root_top1_visit_share" not in GENERIC_CASE_FIELDNAMES
    for k in ("root_mcts_stm_value", "top_child_visit_share", "root_collapsed_ge_0_95",
              "root_value_delta_stm_vs_fpu0", "new_collapse_vs_fpu0"):
        assert k in GENERIC_CASE_FIELDNAMES
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py -q -k "entropy or enrich or fieldnames"`. Expected: FAIL.

- [ ] **Step 3: Implement.** Add `import math`. Add the constant + functions:

```python
GENERIC_CASE_FIELDNAMES = [
    "fpu_value", "case_id", "game_id", "ply", "ply_bucket", "side_to_move",
    "root_mcts_stm_value", "root_mcts_black_value", "top_move",
    "top_child_visit_share", "root_visit_entropy", "root_effective_children",
    "root_collapsed_ge_0_95", "root_n_visited_children",
    "top_child_n_visited_children",
    "root_value_delta_stm_vs_fpu0", "root_value_delta_black_vs_fpu0",
    "top_move_changed_vs_fpu0", "root_children_delta_vs_fpu0",
    "top_child_children_delta_vs_fpu0", "top_child_visit_share_delta_vs_fpu0",
    "root_effective_children_delta_vs_fpu0", "root_visit_entropy_delta_vs_fpu0",
    "new_collapse_vs_fpu0", "resolved_collapse_vs_fpu0",
]


def visit_entropy(visit_counts) -> float:
    """Shannon entropy (nats) of the root children's visit distribution.
    exp(entropy) = effective children; it falls as search concentrates even when
    the raw visited-children COUNT stays flat (the c_puct result). Empty -> 0."""
    total = sum(visit_counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in visit_counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _num_delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    return ""


def enrich_with_deltas(rows):
    """Attach fpu=0.0-relative deltas per case_id, in place. Mover (_stm_) and
    black (_black_) value deltas both recorded (summaries lead with mover, which
    does not cancel across colors). Paired search-shape deltas + collapse
    accounting (new = candidate collapses where baseline did not; resolved =
    the reverse)."""
    baseline = {r["case_id"]: r for r in rows if r["fpu_value"] == BASELINE_FPU}
    for r in rows:
        b = baseline.get(r["case_id"])
        if b is None:
            raise ValueError(f"no fpu={BASELINE_FPU} baseline for {r['case_id']!r}")
        r["root_value_delta_stm_vs_fpu0"] = r["root_mcts_stm_value"] - b["root_mcts_stm_value"]
        r["root_value_delta_black_vs_fpu0"] = r["root_mcts_black_value"] - b["root_mcts_black_value"]
        r["top_move_changed_vs_fpu0"] = r["top_child_move"] != b["top_child_move"]
        r["root_children_delta_vs_fpu0"] = r["root_n_visited_children"] - b["root_n_visited_children"]
        r["top_child_children_delta_vs_fpu0"] = r["top_child_n_visited_children"] - b["top_child_n_visited_children"]
        r["top_child_visit_share_delta_vs_fpu0"] = _num_delta(
            r["top_child_visit_share"], b["top_child_visit_share"])
        r["root_effective_children_delta_vs_fpu0"] = _num_delta(
            r["root_effective_children"], b["root_effective_children"])
        r["root_visit_entropy_delta_vs_fpu0"] = r["root_visit_entropy"] - b["root_visit_entropy"]
        r["new_collapse_vs_fpu0"] = bool(r["root_collapsed_ge_0_95"]) and not bool(b["root_collapsed_ge_0_95"])
        r["resolved_collapse_vs_fpu0"] = bool(b["root_collapsed_ge_0_95"]) and not bool(r["root_collapsed_ge_0_95"])
    return rows
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_fpu_sweep.py tests/test_fpu_sweep_v16a.py
git commit -m "feat(fpu-sweep): mover-perspective value + paired search-shape + collapse-accounting deltas"
```

---

## Task 3: Stratified, paired-delta summaries (overall / bucket / side / bucket×side)

Covers spec §7 and review points 3, 5. One metric function (paired-primary, mover-primary, collapse counts, stable-top paired) + a grouping driver.

**Files:** Modify `diagnose_fpu_sweep.py`; Test `tests/test_fpu_sweep_v16a.py`.

**Interfaces (produces):** `_percentile(values, q)`; `_delta_metrics(rows) -> dict`; `summarize_grouped(rows, group_kind) -> list[dict]`; constants `_METRIC_FIELDS`, `GENERIC_SUMMARY_FIELDNAMES`, `STRATA_SUMMARY_FIELDNAMES`, `BUCKET_ORDER`.

- [ ] **Step 1: Write the failing tests** — append:

```python
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    _percentile, _delta_metrics, summarize_grouped,
    GENERIC_SUMMARY_FIELDNAMES, STRATA_SUMMARY_FIELDNAMES)


def test_percentile():
    assert abs(_percentile([0, 1, 2, 3, 4], 90) - 3.6) < 1e-12
    assert _percentile([7.0], 95) == 7.0


def _e(cid, bucket, side, stm, blk, changed, rootc, topc, share, eff, ent, col,
       ecd=0.0, entd=0.0, tccd=0, newc=False, resc=False, rcd=0, tcsd=0.0):
    return {"fpu_value": -0.2, "case_id": cid, "ply_bucket": bucket,
            "side_to_move": side, "root_value_delta_stm_vs_fpu0": stm,
            "root_value_delta_black_vs_fpu0": blk, "top_move_changed_vs_fpu0": changed,
            "root_n_visited_children": rootc, "top_child_n_visited_children": topc,
            "top_child_visit_share": share, "root_effective_children": eff,
            "root_visit_entropy": ent, "root_collapsed_ge_0_95": col,
            "root_effective_children_delta_vs_fpu0": ecd,
            "root_visit_entropy_delta_vs_fpu0": entd,
            "top_child_children_delta_vs_fpu0": tccd,
            "root_children_delta_vs_fpu0": rcd,
            "top_child_visit_share_delta_vs_fpu0": tcsd,
            "new_collapse_vs_fpu0": newc, "resolved_collapse_vs_fpu0": resc}


def test_black_cancels_mover_preserved():
    rows = [_e(f"c{i}", "midgame", s, -0.10, (-0.10 if s == "black" else 0.10),
               False, 6, 100, 0.5, 5, 1.5, False)
            for i, s in enumerate(["black", "red", "black", "red"])]
    m = _delta_metrics(rows)
    assert abs(m["mean_root_value_delta_black_vs_fpu0"]) < 1e-12       # cancels
    assert abs(m["mean_root_value_delta_stm_vs_fpu0"] - (-0.10)) < 1e-12  # preserved


def test_paired_shape_deltas_and_collapse_counts_and_stable_top():
    rows = [_e("a", "late", "black", -0.2, -0.2, False, 6, 100, 0.96, 4, 1.0, True,
               ecd=-2.0, tccd=-50, newc=True),
            _e("b", "late", "red", 0.1, -0.1, True, 4, 300, 0.5, 8, 2.0, False,
               ecd=1.0, tccd=+30, resc=True)]
    m = _delta_metrics(rows)
    assert m["new_collapse_count"] == 1 and m["resolved_collapse_count"] == 1
    assert abs(m["new_collapse_rate"] - 0.5) < 1e-12
    assert abs(m["mean_root_effective_children_delta_vs_fpu0"] - (-0.5)) < 1e-12
    # stable-top paired reply delta uses only the unchanged-top row (a): -50
    assert abs(m["mean_top_child_children_delta_stable_top_vs_fpu0"] - (-50)) < 1e-12
    assert abs(m["mean_top_child_children_delta_vs_fpu0"] - (-10)) < 1e-12


def test_summarize_grouped_strata():
    rows = [_e("a", "midgame", "black", -0.2, -0.2, True, 6, 100, 0.5, 5, 1.5, False),
            _e("b", "midgame", "red", 0.0, 0.0, False, 4, 200, 0.5, 6, 1.6, False),
            _e("c", "late", "black", 0.4, 0.4, True, 2, 300, 0.7, 3, 1.0, False)]
    assert [g["group"] for g in summarize_grouped(rows, "bucket")] == ["midgame", "late"]
    assert {g["group"] for g in summarize_grouped(rows, "side")} == {"black", "red"}
    assert {g["group"] for g in summarize_grouped(rows, "bucket_x_side")} == {
        "midgame|black", "midgame|red", "late|black"}
    assert STRATA_SUMMARY_FIELDNAMES[:3] == ["fpu_value", "group_kind", "group"]
    assert "mean_root_value_delta_stm_vs_fpu0" in GENERIC_SUMMARY_FIELDNAMES
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py -q -k "percentile or cancels or paired or grouped"`. Expected: FAIL.

- [ ] **Step 3: Implement.** Add `from statistics import median`. Add:

```python
BUCKET_ORDER = ["opening", "early_mid", "midgame", "late"]
_METRIC_FIELDS = [
    "num_positions",
    "mean_root_value_delta_stm_vs_fpu0", "median_abs_root_value_delta_stm_vs_fpu0",
    "p90_abs_root_value_delta_stm_vs_fpu0", "p95_abs_root_value_delta_stm_vs_fpu0",
    "mean_root_value_delta_black_vs_fpu0", "top_move_flip_rate_vs_fpu0",
    "mean_root_visit_entropy_delta_vs_fpu0",
    "mean_root_effective_children_delta_vs_fpu0",
    "mean_root_children_delta_vs_fpu0", "mean_top_child_children_delta_vs_fpu0",
    "mean_top_child_children_delta_stable_top_vs_fpu0",
    "mean_top_child_visit_share_delta_vs_fpu0",
    "new_collapse_count", "new_collapse_rate", "resolved_collapse_count",
    "mean_root_effective_children", "mean_root_n_visited_children",
    "mean_top_child_n_visited_children", "mean_top_child_visit_share",
    "collapsed_ge_0_95_rate",
]
GENERIC_SUMMARY_FIELDNAMES = ["fpu_value"] + _METRIC_FIELDS
STRATA_SUMMARY_FIELDNAMES = ["fpu_value", "group_kind", "group"] + _METRIC_FIELDS


def _percentile(values, q):
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    if n == 1:
        return float(xs[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    if lo + 1 >= n:
        return float(xs[-1])
    return float(xs[lo] + (rank - lo) * (xs[lo + 1] - xs[lo]))


def _mean_num(rows, key):
    vals = [r[key] for r in rows if isinstance(r[key], (int, float))
            and not isinstance(r[key], bool)]
    return sum(vals) / len(vals) if vals else 0.0


def _delta_metrics(rows):
    """Metric dict over ENRICHED rows of one group. Mover deltas are primary;
    black mean is continuity (cancels across colors). Search-shape and reply
    metrics are PAIRED (vs fpu=0.0); reply is also reported over unchanged-top
    rows. Collapse is counted as newly-introduced vs resolved."""
    n = len(rows)
    if n == 0:
        raise ValueError("no rows to summarize")
    stm = [r["root_value_delta_stm_vs_fpu0"] for r in rows]
    abs_stm = [abs(d) for d in stm]
    stable = [r for r in rows if not r["top_move_changed_vs_fpu0"]]
    stable_d = [r["top_child_children_delta_vs_fpu0"] for r in stable
                if isinstance(r["top_child_children_delta_vs_fpu0"], (int, float))]
    new_c = sum(1 for r in rows if r["new_collapse_vs_fpu0"])
    return {
        "num_positions": n,
        "mean_root_value_delta_stm_vs_fpu0": sum(stm) / n,
        "median_abs_root_value_delta_stm_vs_fpu0": median(abs_stm),
        "p90_abs_root_value_delta_stm_vs_fpu0": _percentile(abs_stm, 90),
        "p95_abs_root_value_delta_stm_vs_fpu0": _percentile(abs_stm, 95),
        "mean_root_value_delta_black_vs_fpu0":
            sum(r["root_value_delta_black_vs_fpu0"] for r in rows) / n,
        "top_move_flip_rate_vs_fpu0":
            sum(1 for r in rows if r["top_move_changed_vs_fpu0"]) / n,
        "mean_root_visit_entropy_delta_vs_fpu0": _mean_num(rows, "root_visit_entropy_delta_vs_fpu0"),
        "mean_root_effective_children_delta_vs_fpu0": _mean_num(rows, "root_effective_children_delta_vs_fpu0"),
        "mean_root_children_delta_vs_fpu0": _mean_num(rows, "root_children_delta_vs_fpu0"),
        "mean_top_child_children_delta_vs_fpu0": _mean_num(rows, "top_child_children_delta_vs_fpu0"),
        "mean_top_child_children_delta_stable_top_vs_fpu0":
            (sum(stable_d) / len(stable_d) if stable_d else ""),
        "mean_top_child_visit_share_delta_vs_fpu0": _mean_num(rows, "top_child_visit_share_delta_vs_fpu0"),
        "new_collapse_count": new_c,
        "new_collapse_rate": new_c / n,
        "resolved_collapse_count": sum(1 for r in rows if r["resolved_collapse_vs_fpu0"]),
        "mean_root_effective_children": _mean_num(rows, "root_effective_children"),
        "mean_root_n_visited_children": _mean_num(rows, "root_n_visited_children"),
        "mean_top_child_n_visited_children": _mean_num(rows, "top_child_n_visited_children"),
        "mean_top_child_visit_share": _mean_num(rows, "top_child_visit_share"),
        "collapsed_ge_0_95_rate": sum(1 for r in rows if r["root_collapsed_ge_0_95"]) / n,
    }


def _ordered(values, canonical):
    return [v for v in canonical if v in values] + sorted(v for v in values if v not in canonical)


def summarize_grouped(rows, group_kind):
    if group_kind == "all":
        groups = [("all", rows)]
    elif group_kind == "bucket":
        groups = [(b, [r for r in rows if r["ply_bucket"] == b])
                  for b in _ordered({r["ply_bucket"] for r in rows}, BUCKET_ORDER)]
    elif group_kind == "side":
        groups = [(s, [r for r in rows if r["side_to_move"] == s])
                  for s in _ordered({r["side_to_move"] for r in rows}, ["red", "black"])]
    elif group_kind == "bucket_x_side":
        groups = []
        for b in _ordered({r["ply_bucket"] for r in rows}, BUCKET_ORDER):
            for s in ["red", "black"]:
                sub = [r for r in rows if r["ply_bucket"] == b and r["side_to_move"] == s]
                if sub:
                    groups.append((f"{b}|{s}", sub))
    else:
        raise ValueError(f"unknown group_kind {group_kind!r}")
    out = []
    for gname, grows in groups:
        if not grows:
            continue
        m = _delta_metrics(grows)
        m["group_kind"], m["group"] = group_kind, gname
        out.append(m)
    return out
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_fpu_sweep.py tests/test_fpu_sweep_v16a.py
git commit -m "feat(fpu-sweep): stratified paired-delta summaries (bucket/side/bucket×side) + collapse accounting"
```

---

## Task 4: `main()` compute/write rewrite + routing + legacy golden regression

Covers spec §2/§3 and the write half of §6/§7, plus review point 7. Rewrites the compute/write body: one rich per-case row set (fpu outer, case inner — original order), per-position shape metrics in the loop, then legacy or generic projection. Adds golden-bytes tests (unit + end-to-end no-MCTS) proving the selected-A output is unchanged.

**Files:** Modify `diagnose_fpu_sweep.py`; Test `tests/test_fpu_sweep_v16a.py`; Create `tests/golden/fpu_sweep_legacy_{cases,summary}.csv`.

**Interfaces (produces):** `_legacy_case_row(r)`; `_generic_case_row(r)`; `_write_csv(path, fieldnames, rows)`.

- [ ] **Step 1: Write the failing unit tests (projections + golden bytes)** — append:

```python
import csv as _csv
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    _legacy_case_row, _generic_case_row, _write_csv, FIELDNAMES)


def test_legacy_projection_exact_columns():
    rich = {k: 0 for k in FIELDNAMES}
    rich.update({"fpu_value": 0.0, "case_id": "A", "extra_key": 99})
    assert list(_legacy_case_row(rich).keys()) == FIELDNAMES     # no extras -> DictWriter safe


def test_legacy_case_csv_golden_bytes(tmp_path):
    r = {"fpu_value": 0.0, "case_id": "game_x", "root_mcts_black_value": 0.5,
         "gate_over_ge_0_25": True, "gate_severe_ge_0_50": False,
         "root_n_visited_children": 3, "top_child_move": "12:8",
         "top_child_visit_share": 0.75, "top_child_q_black": -0.25,
         "top_child_n_visited_children": 42}
    p = tmp_path / "c.csv"
    _write_csv(str(p), FIELDNAMES, [_legacy_case_row(r)])
    expected = (
        "fpu_value,case_id,root_mcts_black_value,gate_over_ge_0_25,"
        "gate_severe_ge_0_50,root_n_visited_children,top_child_move,"
        "top_child_visit_share,top_child_q_black,top_child_n_visited_children\r\n"
        "0.0,game_x,0.5,True,False,3,12:8,0.75,-0.25,42\r\n")
    assert p.read_bytes() == expected.encode()
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py -q -k "projection or golden"`. Expected: FAIL.

- [ ] **Step 3: Add projection + writer helpers.**

```python
def _legacy_case_row(r):
    return {k: r[k] for k in FIELDNAMES}


def _generic_case_row(r):
    m = {"fpu_value": r["fpu_value"], "case_id": r["case_id"],
         "game_id": r["game_idx"], "ply": r["position_ply"],
         "ply_bucket": r["ply_bucket"], "side_to_move": r["side_to_move"],
         "root_mcts_stm_value": r["root_mcts_stm_value"],
         "root_mcts_black_value": r["root_mcts_black_value"],
         "top_move": r["top_child_move"],
         "top_child_visit_share": r["top_child_visit_share"],
         "root_visit_entropy": r["root_visit_entropy"],
         "root_effective_children": r["root_effective_children"],
         "root_collapsed_ge_0_95": r["root_collapsed_ge_0_95"],
         "root_n_visited_children": r["root_n_visited_children"],
         "top_child_n_visited_children": r["top_child_n_visited_children"]}
    for k in ("root_value_delta_stm_vs_fpu0", "root_value_delta_black_vs_fpu0",
              "top_move_changed_vs_fpu0", "root_children_delta_vs_fpu0",
              "top_child_children_delta_vs_fpu0", "top_child_visit_share_delta_vs_fpu0",
              "root_effective_children_delta_vs_fpu0", "root_visit_entropy_delta_vs_fpu0",
              "new_collapse_vs_fpu0", "resolved_collapse_vs_fpu0"):
        m[k] = r[k]
    return m


def _write_csv(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
```

- [ ] **Step 4: Replace the `main()` compute/write body** (everything after the `search_fns = …` line from Task 1):

```python
    all_rows = []
    for x in fpus:
        rows = []
        for case in cases:
            cid = case["case_id"]
            _state, side, root_value_stm, root = search_for_row(
                case, search_fns[x],
                pos_base_seed=args.position_probe_base_seed,
                goal_base_seed=args.goal_line_base_seed)
            if root.visit_count != SIMS:
                raise SystemExit(
                    f"fpu_value={x} {cid}: {root.visit_count} sims != {SIMS}")
            black = to_black(root_value_stm, side)
            if run_integrity and x == BASELINE_FPU:
                if cid not in baseline:
                    raise SystemExit(f"{cid} missing from {resolved_integrity}")
                if abs(black - baseline[cid]) > TOLERANCE:
                    raise SystemExit(
                        f"INTEGRITY CHECK FAILED at fpu=0.0 on {cid}: "
                        f"{black:+.6f} != Phase-0 {baseline[cid]:+.6f}")
            over, severe = gate_flags(black)
            top = _best_child(root)
            top_share = "" if top is None else top.visit_count / root.visit_count
            child_visits = [c.visit_count for c in root.children.values() if c.visit_count > 0]
            entropy = visit_entropy(child_visits)
            rows.append({
                "fpu_value": x, "case_id": cid,
                "game_idx": case["game_idx"], "position_ply": case["position_ply"],
                "ply_bucket": case.get("ply_bucket", ""), "side_to_move": side,
                "root_mcts_stm_value": root_value_stm, "root_mcts_black_value": black,
                "gate_over_ge_0_25": over, "gate_severe_ge_0_50": severe,
                "root_n_visited_children": n_visited_children(root),
                "root_visit_entropy": entropy,
                "root_effective_children": math.exp(entropy) if child_visits else 0.0,
                "root_collapsed_ge_0_95": isinstance(top_share, float) and top_share >= 0.95,
                "top_child_move": "" if top is None else "{}:{}".format(*decode_move(top.move)),
                "top_child_visit_share": top_share,
                "top_child_q_black": "" if top is None else to_black(top.q_value, top.state.to_move),
                "top_child_n_visited_children": 0 if top is None else n_visited_children(top),
            })
        if run_integrity and x == BASELINE_FPU:
            print(f"[fpu] integrity check PASSED at fpu=0.0 on {len(rows)} cases")
        all_rows.extend(rows)

    if neutral:
        enrich_with_deltas(all_rows)
        _write_csv(out_path, GENERIC_CASE_FIELDNAMES, [_generic_case_row(r) for r in all_rows])
        overall = []
        for x in fpus:
            g = summarize_grouped([r for r in all_rows if r["fpu_value"] == x], "all")[0]
            g["fpu_value"] = x
            overall.append({k: g[k] for k in GENERIC_SUMMARY_FIELDNAMES})
        _write_csv(summary_path, GENERIC_SUMMARY_FIELDNAMES, overall)
        strata = []
        for x in fpus:
            xr = [r for r in all_rows if r["fpu_value"] == x]
            for kind in ("bucket", "side", "bucket_x_side"):
                for g in summarize_grouped(xr, kind):
                    g["fpu_value"] = x
                    strata.append({k: g[k] for k in STRATA_SUMMARY_FIELDNAMES})
        _write_csv(strata_path, STRATA_SUMMARY_FIELDNAMES, strata)
        for row in overall:
            print(f"[fpu] fpu={row['fpu_value']:<6} mover_dmean={row['mean_root_value_delta_stm_vs_fpu0']:+.4f} "
                  f"flip={row['top_move_flip_rate_vs_fpu0']*100:.1f}% "
                  f"new_collapse={row['new_collapse_rate']*100:.1f}% "
                  f"eff_child_d={row['mean_root_effective_children_delta_vs_fpu0']:+.2f}")
        print(f"\nwrote {len(all_rows)} case rows -> {out_path}")
        print(f"wrote {len(overall)} overall + {len(strata)} stratified summary rows")
    else:
        _write_csv(out_path, FIELDNAMES, [_legacy_case_row(r) for r in all_rows])
        summary_rows = []
        for x in fpus:
            s = summarize([r for r in all_rows if r["fpu_value"] == x])
            s["fpu_value"] = x
            summary_rows.append(s)
        _write_csv(summary_path, SUMMARY_FIELDNAMES, summary_rows)
        for s in summary_rows:
            print(f"[fpu] fpu={s['fpu_value']:<6} mean={s['mean_black_value']:+.4f} "
                  f"over={s['over_pct_ge_0_25']:.1f}% severe={s['severe_pct_ge_0_50']:.1f}%")
        print(f"\nwrote {len(all_rows)} case rows -> {out_path}")
        print(f"wrote {len(summary_rows)} summary rows -> {summary_path}")
    return 0
```

Delete the leftover old single-pass body. Keep `_phase0_baseline`, `summarize`, `gate_flags`, `n_visited_children`, `_search_fns`, `_make_search_fn` unchanged.

- [ ] **Step 5: Write the end-to-end legacy regression test (no MCTS, monkeypatched).** Append to `tests/test_fpu_sweep_v16a.py`:

```python
import types
from pathlib import Path as _P
import scripts.GPU.alphazero.diagnose_fpu_sweep as sweep
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move


def _fake_root(root_stm, children):
    """children: [(rc, visits, q_value, to_move)]. Root is a stub with the given
    children; visit_count = SIMS so the sim-count guard passes."""
    root = MCTSNode(state=types.SimpleNamespace(to_move="black"), visit_count=sweep.SIMS)
    for rc, v, q, tm in children:
        ch = MCTSNode(state=types.SimpleNamespace(to_move=tm), parent=root,
                      move=encode_move(*rc), visit_count=v, value_sum=q * v)
        root.children[ch.move] = ch
    return root, root_stm


# per (case_id, fpu-marker) canned search output; fpu marker is the float itself
_FAKE = {
    ("game_000005_ply_020", 0.0): _fake_root(0.5, [((12, 8), 300, -0.25, "red"), ((1, 1), 100, 0.1, "red")]),
    ("game_000005_ply_020", -0.20): _fake_root(0.3, [((12, 8), 260, -0.10, "red"), ((1, 1), 140, 0.2, "red")]),
}


def _fake_search_for_row(case, fn, **kw):
    root, stm = _FAKE[(case["case_id"], fn)]
    return None, case["side_to_move"], stm, root


def _legacy_manifest(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text("game_idx,case_id,replay_path,position_ply,side_to_move\n"
                 "5,game_000005_ply_020,r.json,20,black\n")
    return str(p)


def test_main_legacy_end_to_end_matches_golden(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "_search_fns", lambda *a, **k: {x: x for x in a[1]})
    monkeypatch.setattr(sweep, "search_for_row", _fake_search_for_row)
    # integrity baseline that matches the fpu=0.0 fake (black == stm since black to move)
    monkeypatch.setattr(sweep, "_phase0_baseline", lambda p: {"game_000005_ply_020": 0.5})
    out, summ = tmp_path / "cases.csv", tmp_path / "summary.csv"
    rc = sweep.main(["--manifest", _legacy_manifest(tmp_path), "--fpu-values", "0.0,-0.20",
                     "--integrity-csv", "dummy", "--out", str(out), "--summary-out", str(summ)])
    assert rc == 0
    golden = _P("tests/golden")
    assert out.read_bytes() == (golden / "fpu_sweep_legacy_cases.csv").read_bytes()
    assert summ.read_bytes() == (golden / "fpu_sweep_legacy_summary.csv").read_bytes()


def test_main_legacy_integrity_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "_search_fns", lambda *a, **k: {x: x for x in a[1]})
    monkeypatch.setattr(sweep, "search_for_row", _fake_search_for_row)
    monkeypatch.setattr(sweep, "_phase0_baseline", lambda p: {"game_000005_ply_020": 0.99})
    with pytest.raises(SystemExit):
        sweep.main(["--manifest", _legacy_manifest(tmp_path), "--fpu-values", "0.0,-0.20",
                    "--integrity-csv", "dummy", "--out", str(tmp_path / "o.csv"),
                    "--summary-out", str(tmp_path / "s.csv")])
```

- [ ] **Step 6: Generate + verify the golden, then prove new == pre-branch.**

Generate (writes into the tmp dir, then copy to the committed golden and eyeball):
```bash
mkdir -p tests/golden
.venv/bin/python - <<'PY'
# reuse the test's fakes to emit the golden into tests/golden/
import types, scripts.GPU.alphazero.diagnose_fpu_sweep as s
from tests.test_fpu_sweep_v16a import _fake_search_for_row, _FAKE
import tests.test_fpu_sweep_v16a as t, pathlib
s._search_fns = lambda *a, **k: {x: x for x in a[1]}
s.search_for_row = _fake_search_for_row
s._phase0_baseline = lambda p: {"game_000005_ply_020": 0.5}
m = pathlib.Path("tests/golden/_m.csv"); m.write_text(
    "game_idx,case_id,replay_path,position_ply,side_to_move\n5,game_000005_ply_020,r.json,20,black\n")
s.main(["--manifest", str(m), "--fpu-values", "0.0,-0.20", "--integrity-csv", "d",
        "--out", "tests/golden/fpu_sweep_legacy_cases.csv",
        "--summary-out", "tests/golden/fpu_sweep_legacy_summary.csv"])
m.unlink()
PY
cat tests/golden/fpu_sweep_legacy_cases.csv   # eyeball: 2 rows (0.0, -0.20), legacy columns
```
Prove equivalence to the original code (true regression proof):
```bash
git show HEAD~3:scripts/GPU/alphazero/diagnose_fpu_sweep.py > /tmp/old_sweep.py
# NOTE for the implementer: import /tmp/old_sweep.py as a module, monkeypatch the
# same _search_fns/search_for_row/_phase0_baseline fakes, run its main() on the
# same 1-case manifest, and assert its case+summary CSV bytes equal the golden
# just generated. The old legacy columns/values must match exactly. Record the
# result in the commit message. (One-time check; not kept in the suite.)
```

- [ ] **Step 7: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep_v16a.py tests/test_fpu_sweep.py -q`. Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_fpu_sweep.py tests/test_fpu_sweep_v16a.py tests/golden/
git commit -m "feat(fpu-sweep): main routing + per-position shape metrics; legacy golden + old==new regression proof"
```

---

## Task 5: Neutral manifest builder — pure sampler

Covers spec §4/§5 and review points 5-context (side balance), 8. Game-first round-robin, per-game cap, min-ply-gap, ~50/50 side balance, buckets incl. opening, injectable opening-prefix dedup.

**Files:** Create `scripts/GPU/alphazero/build_v16a_neutral_position_manifest.py`; Test `tests/test_v16a_neutral_manifest.py`.

**Interfaces (produces):** `side_to_move_for_ply`, `bucket_for_ply`, `candidate_positions`, `_pick_ply`, `sample_bucket(pool_by_game, *, quota, cap, min_gap, seed, side_of=..., state_key_fn=None)`, `sample_neutral_rows(game_records, *, base_seed, source_replay, buckets=BUCKETS, quotas=DEFAULT_QUOTAS, per_game_caps=DEFAULT_PER_GAME_CAPS, min_ply_gap=DEFAULT_MIN_PLY_GAP, state_key_fn_by_bucket=None)`; constants `BUCKETS`, `BUCKET_ORDER`, `DEFAULT_QUOTAS`, `DEFAULT_PER_GAME_CAPS`, `DEFAULT_MIN_PLY_GAP`, `NEUTRAL_FIELDNAMES`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_v16a_neutral_manifest.py`:

```python
from collections import Counter
from scripts.GPU.alphazero.build_v16a_neutral_position_manifest import (
    side_to_move_for_ply, bucket_for_ply, candidate_positions, sample_bucket,
    sample_neutral_rows, NEUTRAL_FIELDNAMES)


def _games(specs):
    return [{"game_idx": i, "n_moves": n, "winner": w,
             "replay_path": f"replays/game_{i:06d}.json"} for i, n, w in specs]


def test_side_parity_and_buckets():
    assert side_to_move_for_ply(0) == "red" and side_to_move_for_ply(91) == "black"
    assert bucket_for_ply(0) is None and bucket_for_ply(1) == "opening"
    assert bucket_for_ply(15) == "opening" and bucket_for_ply(16) == "early_mid"
    assert bucket_for_ply(90) == "midgame" and bucket_for_ply(91) == "late"


def test_candidate_positions_range():
    c = candidate_positions(_games([(1, 45, "red")]))
    assert c["midgame"][1][-1] == 44 and 1 not in c["late"]


def test_round_robin_covers_games_first():
    pool = {g: list(range(16, 41)) for g in range(10)}
    sel, _ = sample_bucket(pool, quota=10, cap=2, min_gap=8, seed=1)
    assert len({g for g, _ in sel}) == 10


def test_min_gap_and_cap():
    sel, _ = sample_bucket({1: list(range(16, 41))}, quota=5, cap=2, min_gap=8, seed=1)
    plies = sorted(p for _, p in sel)
    assert len(sel) == 2 and plies[1] - plies[0] >= 8


def test_side_balance():
    sel, sc = sample_bucket({g: list(range(16, 41)) for g in range(100)},
                            quota=100, cap=2, min_gap=8, seed=2)
    assert abs(sc["red"] - sc["black"]) <= 2


def test_prefix_dedup_via_injected_key():
    sel, _ = sample_bucket({g: list(range(1, 16)) for g in range(20)},
                           quota=20, cap=1, min_gap=0, seed=3,
                           state_key_fn=lambda g, p: g % 3)
    assert len(sel) == 3


def test_deterministic_same_seed():
    recs = _games([(i, 60, "red" if i % 2 else "black") for i in range(200)])
    assert (sample_neutral_rows(recs, base_seed=42, source_replay="s")
            == sample_neutral_rows(recs, base_seed=42, source_replay="s"))


def test_no_dup_quota_and_shortfall():
    recs = _games([(i, 60, "red" if i % 2 else "black") for i in range(300)])
    rows, stats = sample_neutral_rows(recs, base_seed=1, source_replay="s")
    keys = [(r["game_idx"], r["position_ply"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert stats["early_mid"]["achieved"] == 100 and stats["midgame"]["achieved"] == 100
    assert stats["late"]["achieved"] == 0 and stats["late"]["requested"] == 100


def test_row_schema_no_a_labels_and_result_passthrough():
    recs = _games([(i, 60, "red" if i % 2 else "black") for i in range(200)]
                  + [(999, 280, "unknown")])
    rows, _ = sample_neutral_rows(recs, base_seed=5, source_replay="src")
    r = rows[0]
    assert set(r.keys()) == set(NEUTRAL_FIELDNAMES)
    assert not any(k in r for k in ("drop_ply", "largest_a_value_drop", "case_rank"))
    assert any(x["game_result"] == "unknown" for x in rows)     # null-winner passthrough
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v16a_neutral_manifest.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement the pure sampler** — create the module docstring + pure section:

```python
"""Build a deterministic, GAME-HELD-OUT, NON-A-SELECTED neutral position manifest
for the v16a FPU collateral-damage screen.

Samples ordinary positions across mixed games/plies/sides/outcomes, EXCLUDING
entire games named in the A discovery manifest, so discovery and validation share
no games. Output conforms to the canonical position_probe_cases schema and flows
through the SAME trusted path (load_csv_manifest -> search_for_row ->
position_state), unmodified. Stratified (opening/early-mid/midgame/late),
game-first round-robin, per-game capped, min-ply-gap separated, ~50/50 side
balanced per bucket. Winner-null games are kept (game_result="unknown"): in the
default corpus all such games are state_cap 280-ply marathons -- the most
search-stressed, highest-value late samples.

READ-ONLY on replays; writes one CSV + sidecar meta JSON. No MCTS/network/train.
Building/running THIS builder is in scope for v16a; running the FPU sweep is NOT.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .goal_line_trigger_probe_cases import position_state

BUCKETS = (("opening", 1, 15), ("early_mid", 16, 40),
           ("midgame", 41, 90), ("late", 91, None))
BUCKET_ORDER = [name for name, _, _ in BUCKETS]
DEFAULT_QUOTAS = {"opening": 40, "early_mid": 100, "midgame": 100, "late": 100}
DEFAULT_PER_GAME_CAPS = {"opening": 1, "early_mid": 2, "midgame": 2, "late": 2}
DEFAULT_MIN_PLY_GAP = 8

NEUTRAL_FIELDNAMES = [
    "case_id", "game_idx", "replay_path", "position_ply", "side_to_move",
    "ply_bucket", "game_result", "total_game_plies", "source_replay", "sample_seed",
]


def side_to_move_for_ply(ply: int) -> str:
    """Ply 0 is red's opener; TwixtState alternates each ply. position_state
    asserts exactly this parity -> a mismatch fails loud at validation."""
    return "red" if ply % 2 == 0 else "black"


def bucket_for_ply(ply: int, buckets=BUCKETS):
    for name, lo, hi in buckets:
        if ply >= lo and (hi is None or ply <= hi):
            return name
    return None


def candidate_positions(game_records, buckets=BUCKETS):
    """{bucket -> {game_idx -> [valid plies asc]}}; valid iff in bucket AND
    0 <= ply < n_moves (position_state range)."""
    out = {name: {} for name, _, _ in buckets}
    for r in sorted(game_records, key=lambda x: x["game_idx"]):
        n = r["n_moves"]
        for name, lo, hi in buckets:
            hi_eff = (n - 1) if hi is None else min(hi, n - 1)
            plies = list(range(lo, hi_eff + 1))
            if plies:
                out[name][r["game_idx"]] = plies
    return out


def _pick_ply(cand, per_game_selected, min_gap, side_count, side_of):
    ok = [p for p in cand if all(abs(p - q) >= min_gap for q in per_game_selected)]
    if not ok:
        return None
    behind = min(side_count, key=lambda s: (side_count[s], s))
    for p in ok:
        if side_of(p) == behind:
            return p
    return ok[0]


def sample_bucket(pool_by_game, *, quota, cap, min_gap, seed,
                  side_of=side_to_move_for_ply, state_key_fn=None):
    """Game-first round-robin: pass 1 takes <=1 ply/game (covers every game),
    later passes add up to `cap`/game, each >= min_gap from that game's picks;
    side balanced toward 50/50; optional state_key_fn de-dupes across games.
    Deterministic. Returns (selected [(game,ply)], side_count)."""
    rng = random.Random(seed)
    games = sorted(pool_by_game)
    rng.shuffle(games)
    plies = {g: pool_by_game[g][:] for g in games}
    for g in games:
        rng.shuffle(plies[g])
    picked, selected, seen = {}, [], set()
    side_count = {"red": 0, "black": 0}
    progress = True
    while len(selected) < quota and progress:
        progress = False
        for g in games:
            if len(selected) >= quota:
                break
            if len(picked.get(g, [])) >= cap:
                continue
            cand = plies[g]
            if state_key_fn is not None:
                cand = [p for p in cand if state_key_fn(g, p) not in seen]
            chosen = _pick_ply(cand, picked.get(g, []), min_gap, side_count, side_of)
            if chosen is None:
                continue
            picked.setdefault(g, []).append(chosen)
            plies[g].remove(chosen)
            if state_key_fn is not None:
                seen.add(state_key_fn(g, chosen))
            selected.append((g, chosen))
            side_count[side_of(chosen)] += 1
            progress = True
    return selected, side_count


def sample_neutral_rows(game_records, *, base_seed, source_replay,
                        buckets=BUCKETS, quotas=DEFAULT_QUOTAS,
                        per_game_caps=DEFAULT_PER_GAME_CAPS,
                        min_ply_gap=DEFAULT_MIN_PLY_GAP,
                        state_key_fn_by_bucket=None):
    recs = {r["game_idx"]: r for r in game_records}
    pools = candidate_positions(game_records, buckets)
    key_by_bucket = state_key_fn_by_bucket or {}
    rows, stats = [], {}
    for offset, (name, _lo, _hi) in enumerate(buckets):
        sel, side_count = sample_bucket(
            pools[name], quota=quotas[name], cap=per_game_caps[name],
            min_gap=min_ply_gap, seed=base_seed + offset,
            state_key_fn=key_by_bucket.get(name))
        for game_idx, ply in sel:
            rec = recs[game_idx]
            rows.append({
                "case_id": f"neutral_game_{game_idx:06d}_ply_{ply:03d}",
                "game_idx": game_idx, "replay_path": rec["replay_path"],
                "position_ply": ply, "side_to_move": side_to_move_for_ply(ply),
                "ply_bucket": name, "game_result": rec["winner"],
                "total_game_plies": rec["n_moves"], "source_replay": source_replay,
                "sample_seed": base_seed,
            })
        stats[name] = {"requested": quotas[name], "achieved": len(sel),
                       "games_used": len({g for g, _ in sel}),
                       "eligible_games": len(pools[name]), "side_balance": side_count}
    return rows, stats
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v16a_neutral_manifest.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_v16a_neutral_position_manifest.py tests/test_v16a_neutral_manifest.py
git commit -m "feat(v16a): game-first round-robin, side-balanced, min-gap neutral sampler"
```

---

## Task 6: Builder shell — game holdout, opening-prefix dedup, validation, meta, CLI

Covers spec §4/§5 I/O and review points 2, 8, recommendations A+B. Keeps winner-null games; excludes A-games via the shared `DEFAULT_A_MANIFEST` constant; records requested vs matched exclusions; opening-prefix dedup; graceful winner-null validation drop.

**Files:** Modify `build_v16a_neutral_position_manifest.py`; Test `tests/test_v16a_neutral_manifest.py`.

**Interfaces (produces):** `load_game_index(jsonl_path, *, require_winner=False)`; `load_excluded_game_ids(paths)`; `opening_prefix_key(moves, ply)`; `make_opening_key_fn(records_by_idx)`; `validate_row(row)`; `write_manifest`, `write_meta`, `main(argv=None)`. Constants: `DEFAULT_SOURCE_JSONL`, `DEFAULT_OUT`, `DEFAULT_SEED = 20260710`.

- [ ] **Step 1: Write the failing tests** — append:

```python
import json
from scripts.GPU.alphazero.build_v16a_neutral_position_manifest import (
    load_game_index, load_excluded_game_ids, opening_prefix_key, write_manifest, write_meta)


def test_load_game_index_keeps_null_winner_as_unknown_and_sorts(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"game_idx": 5, "n_moves": 60, "winner": "red", "replay_path": "a"},
        {"game_idx": 2, "n_moves": 40, "winner": "black", "replay_path": "b"},
        {"game_idx": 9, "n_moves": 280, "winner": None, "replay_path": "c"}]) + "\n")
    recs, dropped = load_game_index(str(p))                 # require_winner False default
    assert dropped == 0 and [r["game_idx"] for r in recs] == [2, 5, 9]
    assert recs[2]["winner"] == "unknown"


def test_load_excluded_game_ids(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("game_idx,case_id\n347,x\n631,y\n347,z\n")
    assert load_excluded_game_ids([str(p)]) == {347, 631}


def test_opening_prefix_key():
    moves = [{"row": 1, "col": 2}, {"row": 3, "col": 4}, {"row": 5, "col": 6}]
    assert opening_prefix_key(moves, 2) == ((1, 2), (3, 4))


def test_write_roundtrip(tmp_path):
    import csv
    out = tmp_path / "s" / "n.csv"
    write_manifest([{"case_id": "c", "game_idx": 1, "replay_path": "r",
                     "position_ply": 20, "side_to_move": "red", "ply_bucket": "early_mid",
                     "game_result": "red", "total_game_plies": 60,
                     "source_replay": "s", "sample_seed": 9}], str(out))
    write_meta(str(out), {"base_seed": 9})
    with open(out, newline="") as f:
        assert list(csv.DictReader(f))[0]["case_id"] == "c"
    assert json.loads((tmp_path / "s" / "n.csv.meta.json").read_text())["base_seed"] == 9
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v16a_neutral_manifest.py -q -k "load_game or excluded or prefix or roundtrip"`. Expected: FAIL.

- [ ] **Step 3: Implement the shell + CLI** — append:

```python
DEFAULT_SOURCE_JSONL = ("logs/eval/"
                        "calib020_0001_vs_0379_800g_w4_seed20115_replay_games.jsonl")
DEFAULT_OUT = "logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv"
DEFAULT_SEED = 20260710


def load_game_index(jsonl_path, *, require_winner=False):
    """Read the replay-eval JSONL INDEX (per line: game_idx, n_moves, winner,
    replay_path -- not the moves). Winner-null games are KEPT with
    winner='unknown' (require_winner=False); they reconstruct like any other and
    are the most search-stressed samples. Returns (records sorted, dropped)."""
    recs, dropped = [], 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            w = g.get("winner")
            if w not in ("red", "black"):
                if require_winner:
                    dropped += 1
                    continue
                w = "unknown"
            recs.append({"game_idx": int(g["game_idx"]), "n_moves": int(g["n_moves"]),
                         "winner": w, "replay_path": g["replay_path"]})
    recs.sort(key=lambda r: r["game_idx"])
    return recs, dropped


def load_excluded_game_ids(paths):
    out = set()
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                out.add(int(r["game_idx"]))
    return out


def opening_prefix_key(moves, ply):
    """Ordered move-prefix key. TwixT has no captures, so identical prefixes reach
    identical states; transpositions (same pegs, different order) are NOT merged
    -- this is opening-PREFIX dedup, not full board-state dedup."""
    return tuple((m["row"], m["col"]) for m in moves[:ply])


def make_opening_key_fn(records_by_idx):
    cache = {}

    def keyfn(game_idx, ply):
        moves = cache.get(game_idx)
        if moves is None:
            moves = json.loads(Path(records_by_idx[game_idx]["replay_path"]).read_text())["moves"]
            cache[game_idx] = moves
        return opening_prefix_key(moves, ply)

    return keyfn


def validate_row(row):
    replay = json.loads(Path(row["replay_path"]).read_text())
    position_state(replay, int(row["position_ply"]), row["side_to_move"])


def write_manifest(rows, out_csv):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NEUTRAL_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def write_meta(out_csv, meta):
    Path(str(out_csv) + ".meta.json").write_text(json.dumps(meta, indent=2))


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Build a deterministic, game-held-out, non-A-selected neutral "
                    "position manifest for the v16a FPU collateral screen. "
                    "READ-ONLY; writes one CSV + meta JSON. Does NOT run the sweep.")
    ap.add_argument("--source-jsonl", default=DEFAULT_SOURCE_JSONL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--exclude-manifest", action="append", default=None,
                    help="CSV(s) whose game_idx values are held out (repeatable). "
                         "The A probe manifest is added unless --no-default-exclude.")
    ap.add_argument("--no-default-exclude", action="store_true")
    ap.add_argument("--exclude-winnerless", action="store_true",
                    help="drop winner-null games (default: keep as game_result=unknown).")
    ap.add_argument("--min-ply-gap", type=int, default=DEFAULT_MIN_PLY_GAP)
    ap.add_argument("--quota-opening", type=int, default=DEFAULT_QUOTAS["opening"])
    ap.add_argument("--quota-early-mid", type=int, default=DEFAULT_QUOTAS["early_mid"])
    ap.add_argument("--quota-midgame", type=int, default=DEFAULT_QUOTAS["midgame"])
    ap.add_argument("--quota-late", type=int, default=DEFAULT_QUOTAS["late"])
    ap.add_argument("--no-validate", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    # DEFAULT_A_MANIFEST is the SINGLE source of truth for the discovery set;
    # import (deferred: diagnose_fpu_sweep pulls mcts) so the holdout cannot drift.
    from .diagnose_fpu_sweep import DEFAULT_A_MANIFEST
    args = _parse_args(argv)
    quotas = {"opening": args.quota_opening, "early_mid": args.quota_early_mid,
              "midgame": args.quota_midgame, "late": args.quota_late}

    records, dropped = load_game_index(args.source_jsonl,
                                       require_winner=args.exclude_winnerless)
    excludes = list(args.exclude_manifest or [])
    if not args.no_default_exclude:
        excludes.append(DEFAULT_A_MANIFEST)
    requested_ids = load_excluded_game_ids(excludes) if excludes else set()
    corpus_ids = {r["game_idx"] for r in records}
    matched_ids = requested_ids & corpus_ids
    held = [r for r in records if r["game_idx"] not in matched_ids]
    print(f"[v16a] {len(records)} games (dropped winnerless {dropped}); excluded "
          f"{len(matched_ids)}/{len(requested_ids)} requested A-games -> {len(held)} held-out")

    records_by_idx = {r["game_idx"]: r for r in held}
    rows, stats = sample_neutral_rows(
        held, base_seed=args.seed, source_replay=args.source_jsonl,
        quotas=quotas, min_ply_gap=args.min_ply_gap,
        state_key_fn_by_bucket={"opening": make_opening_key_fn(records_by_idx)})

    for name in BUCKET_ORDER:
        st = stats[name]
        flag = ("  <-- SHORTFALL (data-limited)" if st["achieved"] < st["requested"] else "")
        print(f"[v16a] {name:10s} {st['achieved']:3d}/{st['requested']:<3d} across "
              f"{st['games_used']} games  sides={st['side_balance']}{flag}")

    validate_dropped = 0
    if not args.no_validate:
        kept = []
        for r in rows:
            try:
                validate_row(r)
                kept.append(r)
            except Exception:
                if r["game_result"] == "unknown":     # tolerate odd winner-null games
                    validate_dropped += 1
                    continue
                raise                                  # winner-having failure is a real bug
        rows = kept
        print(f"[v16a] validated {len(rows)} rows (dropped {validate_dropped} "
              f"unreconstructable winner-null rows)")

    write_manifest(rows, args.out)
    write_meta(args.out, {
        "source_jsonl": args.source_jsonl, "base_seed": args.seed,
        "buckets": {n: [lo, hi] for n, lo, hi in BUCKETS}, "quotas": quotas,
        "per_game_caps": DEFAULT_PER_GAME_CAPS, "min_ply_gap": args.min_ply_gap,
        "excluded_manifests": excludes,
        "requested_excluded_game_count": len(requested_ids),
        "matched_excluded_game_count": len(matched_ids),
        "matched_excluded_game_ids": sorted(matched_ids),
        "winnerless_dropped": dropped, "validate_dropped": validate_dropped,
        "num_rows": len(rows), "per_bucket_stats": stats,
        "fieldnames": NEUTRAL_FIELDNAMES,
        "sample_kind": "stratified_game_held_out_non_selected",
    })
    print(f"[v16a] wrote {len(rows)} rows -> {args.out}  (+ .meta.json)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v16a_neutral_manifest.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_v16a_neutral_position_manifest.py tests/test_v16a_neutral_manifest.py
git commit -m "feat(v16a): builder shell — A-game holdout (single-SoT), winner-null kept, opening-prefix dedup, validation, meta"
```

---

## Task 7: Generate the artifact + design/protocol note with a pre-registered decision table

Runs the builder (no MCTS), force-adds the artifact, and writes the design note — including a pre-registered decision table and **no executable sweep command**.

**Files:** Create the manifest CSV + meta; create the design note.

- [ ] **Step 1: Generate** — `.venv/bin/python -m scripts.GPU.alphazero.build_v16a_neutral_position_manifest`. Expected: 770 held-out, opening ≈40, early_mid 100, midgame 100, **late ≈84/100 with SHORTFALL**, sides ≈50/50, all rows validate.

- [ ] **Step 2: Sanity-check**

```bash
.venv/bin/python -c "
import csv, collections
rows=list(csv.DictReader(open('logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv')))
print('rows', len(rows), dict(collections.Counter(r['ply_bucket'] for r in rows)))
print('sides', dict(collections.Counter(r['side_to_move'] for r in rows)))
print('results', dict(collections.Counter(r['game_result'] for r in rows)))
print('games', len({r['game_idx'] for r in rows}))
assert not any(k in rows[0] for k in ('drop_ply','largest_a_value_drop','case_rank'))
print('OK')"
```

- [ ] **Step 3: Neutral-mode + zero-leak checks**

```bash
.venv/bin/python -c "
from scripts.GPU.alphazero.position_probe_cases import load_csv_manifest
from scripts.GPU.alphazero.diagnose_fpu_sweep import manifest_is_neutral, DEFAULT_A_MANIFEST
import csv
m=load_csv_manifest('logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv')
print('neutral?', manifest_is_neutral(m['cases']), 'n', m['num_cases'])
a={int(r['game_idx']) for r in csv.DictReader(open(DEFAULT_A_MANIFEST))}
n={c['game_idx'] for c in m['cases']}
assert not (a & n), f'LEAK {a&n}'
print('OK: 0 games shared with A discovery')"
```

- [ ] **Step 4: Write the design/protocol note** — create `docs/superpowers/specs/2026-07-10-v16a-unbiased-fpu-validation-design.md` covering: objective (first rung of the FPU validation ladder from `[[a-signal-search-artifact-fpu]]`); the naming/mode/holdout/mover-perspective/single-denominator/opening-prefix decisions; measured bucket reality (late 84 incl. 280-ply marathons); the hierarchy (root goal / decisive benchmark / v16a rung). Include the **Protocol** and **pre-registered decision table** verbatim:

```
Tool capability:      diagnose_fpu_sweep accepts arbitrary --fpu-values.
v16a validation run:  FROZEN control 0.0 vs candidate -0.20 ONLY (enforced in
                      code: neutral default = 0.0,-0.20; others need
                      --allow-non-protocol-fpu). Screening extra candidates on
                      the holdout is tuning on the holdout.

The held-out MCTS sweep is an operator-run phase and is deliberately not
specified or executed by this implementation plan.

PRE-REGISTERED DECISION TABLE (confirm BEFORE running the sweep; never after
seeing results). Candidate = -0.20 vs control 0.0, over the held-out set.

  STATISTICAL RULE: only strata with num_positions >= 20 participate in any
  "stratum" pass/fail below; smaller strata are descriptive, inspection-only.
  NARROWING IS THE INTENDED MECHANISM: effective-children and reply-count
  reductions are NOT failures on their own -- they matter only alongside new
  collapse, extreme concentration, or broad value disruption.

  AUTOMATIC REJECT if any:
    - overall new_collapse_rate >= 0.05, OR
    - any n>=20 stratum new_collapse_rate >= 0.10, OR
    - overall median_abs_root_value_delta_stm_vs_fpu0 >= 0.20, OR
    - overall p95_abs_root_value_delta_stm_vs_fpu0 >= 0.60, OR
    - effective-children reduction >= 50% AND overall
      mean_top_child_visit_share_delta_vs_fpu0 >= +0.15
      [reduction = -mean_root_effective_children_delta_vs_fpu0 / mean_root_effective_children(@0.0)], OR
    - integrity, pairing, deterministic-reproduction, or manifest-holdout failure.

  MANDATORY INSPECTION (case-level review; not auto-reject) if any:
    - new_collapse_rate > 0 but < 0.05, OR
    - overall top_move_flip_rate_vs_fpu0 >= 0.25, OR
    - any n>=20 stratum top_move_flip_rate_vs_fpu0 >= 0.35, OR
    - effective-children reduction >= 30% [same formula as above], OR
    - any n>=20 stratum |mean_root_value_delta_stm_vs_fpu0| >= 0.10, OR
    - overall p95_abs_root_value_delta_stm_vs_fpu0 >= 0.35, OR
    - overall mean_top_child_visit_share_delta_vs_fpu0 >= 0.10, OR
    - stable-top opponent-reply reduction >= 50%
      [= -mean_top_child_children_delta_stable_top_vs_fpu0 / mean_top_child_n_visited_children(@0.0)].

  SAFE-TO-ADVANCE (to B/C/D + the equal-budget strength match) only if ALL hold:
    - integrity + paired-row checks pass, AND
    - new_collapse_count == 0, AND
    - overall top_move_flip_rate_vs_fpu0 < 0.25, AND
    - no n>=20 stratum top_move_flip_rate_vs_fpu0 >= 0.35, AND
    - overall median_abs_root_value_delta_stm_vs_fpu0 < 0.10, AND
    - overall p95_abs_root_value_delta_stm_vs_fpu0 < 0.35, AND
    - no n>=20 stratum |mean_root_value_delta_stm_vs_fpu0| >= 0.10, AND
    - overall mean_top_child_visit_share_delta_vs_fpu0 < 0.10, AND
    - no n>=20 stratum mean_top_child_visit_share_delta_vs_fpu0 >= 0.15.
    (Do NOT require any effective-children or reply-count reduction to advance;
    narrowing is the mechanism and is acceptable absent collapse, extreme
    concentration, or broad value disruption.)

  "Advance" means "no obvious collateral damage," NOT "improvement." Improvement
  is decided only by the equal-budget, balanced-color, statistically-significant
  head-to-head strength match.
```

- [ ] **Step 5: Force-add + commit**

```bash
git add -f logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv \
           logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv.meta.json
git add docs/superpowers/specs/2026-07-10-v16a-unbiased-fpu-validation-design.md
git commit -m "feat(v16a): emit game-held-out neutral manifest + design note w/ pre-registered decision table (sweep NOT run)"
```

- [ ] **Step 6: Focused regression suite** (the v16a-touching files)

`.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_sweep.py tests/test_fpu_sweep_v16a.py tests/test_v16a_neutral_manifest.py tests/test_fpu_value.py tests/test_cpuct_sweep.py -q`. Expected: PASS.

- [ ] **Step 7: Authoritative full suite** (must pass before declaring done)

`.venv/bin/python -m pytest tests/ -q`. Expected: PASS (or only pre-existing, unrelated failures — record any in the final report).

---

## Self-review — spec + review coverage

| Requirement | Task(s) |
|---|---|
| §1 aliases + conditional integrity | Task 1 |
| §2 tolerant reconstruction | Already satisfied; verified Task 7 Step 3 |
| §3 selected-A byte-identical + regression | Task 1 + Task 4 (unit golden + **end-to-end golden** + **old==new proof**) |
| §4/§5 builder | Tasks 5, 6, 7 |
| §6 case output (mover + shape + deltas) | Tasks 2, 4 |
| §7 stratified summaries | Tasks 3, 4 |
| §8 tests | Tasks 1/2/3/5 (pure) + Task 4 (e2e) |
| §9 do-not-change | Global Constraints |
| Review 1 conditional integrity + golden bytes | Task 1 + Task 4 |
| Review 2 game-level holdout + requested/matched counts | Task 6 |
| Review 3 mover-primary + full stratification + cancellation test | Tasks 2, 3 |
| Review 4 frozen protocol enforced in code | Task 1 (`resolve_fpu_values`, `--allow-non-protocol-fpu`) + Task 7 note |
| Review 5 paired shape deltas + collapse counts + stable-top paired | Tasks 2, 3 |
| Review 6 single top-share denominator; drop redundant column | Tasks 2, 4 |
| Review 7 end-to-end legacy regression + fixed fixture perspective | Task 4 |
| Review 8 rename to opening-prefix dedup | Tasks 5, 6, Design decisions |
| Rec A include winner-null (state_cap) games | Task 6 (`require_winner=False`, graceful drop) |
| Rec B A-manifest single SoT + requested/matched counts | Task 6 (deferred import of `DEFAULT_A_MANIFEST`) |
| Testing correction: `.venv` interpreter + full suite | all Task test steps + Task 7 Step 7 |
| Protocol gap: pre-registered decision table | Task 7 Step 4 |
| No executable next-run command | Task 7 Step 4 |

**Placeholder scan:** none. **Type consistency:** rich-row keys (Task 4) consumed unchanged by enrich/metrics/projections; `game_record` shape flows `load_game_index` → `sample_neutral_rows`; `NEUTRAL_FIELDNAMES` single-sources the manifest columns; `DEFAULT_A_MANIFEST` single-sources the holdout.
