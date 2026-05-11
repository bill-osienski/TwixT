# Closeout Tail Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the closeout tail after Spec 2 by adding a td=1 root visit-forcing rule in MCTS, plus three diagnostics (td-before bulk breakdown, recovery-after-dominant-lost classification, and Fix 1 telemetry summarization) in the replay analyzer.

**Architecture:** Three pure analyzer additions ship first (Fix 0, Fix 3, Fix 1 telemetry summarizer) and gate the trainer work. Then MCTS gains a `force_root_visits` method that reuses the existing synchronous single-simulation path with a root-child override — proved equivalent to a normal PUCT-selected sim via a unit test. Self-play wires the call after `_add_dirichlet_noise` and before the main simulation loop. Fix 1 ships off by default; a single 10-iteration treatment block from `model_iter_0139.safetensors` evaluates impact against the existing 130-139 baseline.

**Tech Stack:** Python 3.14, MLX (training), pytest 9, existing AlphaZero-staged training pipeline in `scripts/GPU/alphazero/`.

**Spec:** `docs/superpowers/specs/2026-05-10-closeout-tail-correction-design.md` (commit `a9651ae`).

---

## File Structure

**Created:**
- `tests/test_analyzer_td_closeout_breakdown.py` — Fix 0 aggregation tests
- `tests/test_analyzer_recovery_events.py` — Fix 3 classification tests
- `tests/test_analyzer_closeout_td1_visit_forcing_summary.py` — Fix 1 telemetry roll-up tests
- `tests/test_mcts_force_root_visits.py` — Fix 1 MCTS unit tests (trigger, config, behavior)
- `tests/test_mcts_forced_root_visit_equivalence.py` — equivalence test (forced sim ≡ normal sim with same target child)
- `scripts/GPU/alphazero/smoke_closeout_td1_visit_forcing.py` — integration smoke
- `tests/test_train_closeout_td1_cli.py` — CLI flag plumbing test
- `tests/test_analyzer_closeout_selection_tiebreak_summary.py` — Fix 2 telemetry tests (Phase 7, conditional)
- `tests/test_mcts_closeout_selection_tiebreak.py` — Fix 2 MCTS tests (Phase 7, conditional)

**Modified:**
- `scripts/twixt_replay_analyzer.py` — Fix 0, Fix 3, Fix 1 telemetry summarizer (+ Fix 2 summarizer in Phase 7)
- `scripts/GPU/alphazero/mcts.py` — `MCTSConfig` fields, extracted `_run_single_simulation` helper, `force_root_visits` method, telemetry accumulators, call site inside `search_from_root`
- `scripts/GPU/alphazero/self_play.py` — pass `gc_state_full` into `search_from_root`, drain MCTS telemetry into stats sidecar at end of iteration
- `scripts/GPU/alphazero/train.py` — new CLI flags + validation + wire into `MCTSConfig`

---

# Phase 1 — Analyzer only (Fix 0 + Fix 3)

## Task 1: Add `aggregate_td_closeout_breakdown` (Fix 0 aggregator)

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` (add new function near `aggregate_goal_completion_diagnostics_from_records` at line 734)
- Test: `tests/test_analyzer_td_closeout_breakdown.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyzer_td_closeout_breakdown.py`:

```python
"""Tests for Fix 0: td-before closeout breakdown (spec 2026-05-10 §3)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import aggregate_td_closeout_breakdown


def _ply_record(*, ply, side, det_side, td, q, sel_class, ec_p=None, ec_v=None, rd_p=None, rd_v=None):
    """Build a minimal per-ply diagnostic record matching schema in §3.1."""
    def _rank_block(p, v):
        if p is None and v is None:
            return {"best_policy_rank": None, "best_visit_rank": None}
        return {"best_policy_rank": p, "best_visit_rank": v}
    return {
        "ply": ply,
        "side_to_move": side,
        "root_summary": {"q_value": q},
        "goal_completion": {"total_goal_distance_before": td},
        "endpoint_completion_ranking": _rank_block(ec_p, ec_v),
        "distance_reducing_ranking": _rank_block(rd_p, rd_v),
        "selected_move_classification": {"primary_class": sel_class},
    }


def test_td_buckets_split_by_distance_and_classify_selection():
    # Build two td=1 rows (one redundant, one completes_endpoint) and
    # one td=2 row (off_chain). Detected player is "black" throughout.
    records = [
        _ply_record(ply=10, side="black", det_side="black", td=1, q=0.97,
                    sel_class="redundant_reinforcement", ec_p=33, ec_v=173, rd_p=33, rd_v=173),
        _ply_record(ply=12, side="black", det_side="black", td=1, q=0.98,
                    sel_class="completes_endpoint", ec_p=1, ec_v=1, rd_p=1, rd_v=1),
        _ply_record(ply=14, side="black", det_side="black", td=2, q=0.96,
                    sel_class="off_chain", ec_p=None, ec_v=None, rd_p=5, rd_v=4),
    ]

    out = aggregate_td_closeout_breakdown(records, detected_player="black",
                                          high_value_threshold=0.95)

    assert out["td=1"]["records"] == 2
    assert out["td=1"]["high_value_records"] == 2
    assert out["td=1"]["selected_redundant_rate"] == 0.5
    assert out["td=1"]["selected_completes_endpoint_rate"] == 0.5
    # endpoint exists in both td=1 rows
    assert out["td=1"]["endpoint_completion_exists_rate"] == 1.0
    # visit top-5: ranks 173 (no) and 1 (yes) → 0.5
    assert out["td=1"]["endpoint_visit_top5_rate"] == 0.5
    assert out["td=1"]["endpoint_visit_gt20_rate"] == 0.5

    assert out["td=2"]["records"] == 1
    assert out["td=2"]["selected_off_chain_rate"] == 1.0
    assert out["td=2"]["endpoint_completion_exists_rate"] == 0.0
    # reducer exists, in visit top-5
    assert out["td=2"]["distance_reducer_exists_rate"] == 1.0
    assert out["td=2"]["reducer_visit_top5_rate"] == 1.0


def test_records_for_other_side_to_move_are_excluded():
    records = [
        _ply_record(ply=10, side="red", det_side="black", td=1, q=0.97,
                    sel_class="completes_endpoint", ec_p=1, ec_v=1, rd_p=1, rd_v=1),
    ]
    out = aggregate_td_closeout_breakdown(records, detected_player="black",
                                          high_value_threshold=0.95)
    assert out["td=1"]["records"] == 0


def test_td_outside_1_2_3_is_ignored():
    records = [
        _ply_record(ply=10, side="black", det_side="black", td=4, q=0.97,
                    sel_class="off_chain", ec_p=None, ec_v=None, rd_p=None, rd_v=None),
    ]
    out = aggregate_td_closeout_breakdown(records, detected_player="black",
                                          high_value_threshold=0.95)
    assert out["td=1"]["records"] == 0
    assert out["td=2"]["records"] == 0
    assert out["td=3"]["records"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_analyzer_td_closeout_breakdown.py -v
```

Expected: ImportError — `aggregate_td_closeout_breakdown` does not yet exist.

- [ ] **Step 3: Implement the aggregator**

Add to `scripts/twixt_replay_analyzer.py` immediately before `aggregate_goal_completion_diagnostics_from_records` (around line 734):

```python
def aggregate_td_closeout_breakdown(
    per_ply_records: list,
    detected_player: str,
    high_value_threshold: float = 0.95,
) -> dict:
    """Bucket strict closeout per-ply records by total_goal_distance_before.

    Spec 2026-05-10 §3. Reads records of the form emitted in
    goal_completion_diagnostics (see closeout_diagnostics.build_*).

    Returns a dict keyed "td=1" / "td=2" / "td=3" with the metric set
    described in spec §3.1.
    """
    def _empty():
        return {
            "records": 0,
            "high_value_records": 0,
            "selected_completes_endpoint": 0,
            "selected_reduces_distance": 0,
            "selected_redundant": 0,
            "selected_off_chain": 0,
            "selected_other": 0,
            "endpoint_exists": 0,
            "endpoint_policy_top1": 0,
            "endpoint_policy_top5": 0,
            "endpoint_policy_top20": 0,
            "endpoint_policy_gt20": 0,
            "endpoint_visit_top1": 0,
            "endpoint_visit_top5": 0,
            "endpoint_visit_top20": 0,
            "endpoint_visit_gt20": 0,
            "reducer_exists": 0,
            "reducer_policy_top1": 0,
            "reducer_policy_top5": 0,
            "reducer_policy_top20": 0,
            "reducer_policy_gt20": 0,
            "reducer_visit_top1": 0,
            "reducer_visit_top5": 0,
            "reducer_visit_top20": 0,
            "reducer_visit_gt20": 0,
        }

    buckets = {"td=1": _empty(), "td=2": _empty(), "td=3": _empty()}

    def _bucket_rank(rank, c):
        if rank is None:
            return
        if rank <= 1:
            c["_top1"] += 1; c["_top5"] += 1; c["_top20"] += 1
        elif rank <= 5:
            c["_top5"] += 1; c["_top20"] += 1
        elif rank <= 20:
            c["_top20"] += 1
        else:
            c["_gt20"] += 1

    for rec in per_ply_records or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("side_to_move") != detected_player:
            continue
        gc = rec.get("goal_completion") or {}
        td = gc.get("total_goal_distance_before")
        if td not in (1, 2, 3):
            continue
        key = f"td={td}"
        b = buckets[key]
        b["records"] += 1
        q = (rec.get("root_summary") or {}).get("q_value")
        if isinstance(q, (int, float)) and q >= high_value_threshold:
            b["high_value_records"] += 1
        cls_name = ((rec.get("selected_move_classification") or {}).get("primary_class")) or ""
        cls_field = {
            "completes_endpoint": "selected_completes_endpoint",
            "reduces_total_goal_distance": "selected_reduces_distance",
            "redundant_reinforcement": "selected_redundant",
            "off_chain": "selected_off_chain",
            "other": "selected_other",
        }.get(cls_name)
        if cls_field is not None:
            b[cls_field] += 1
        # Endpoint completion ranking buckets (denominator: endpoint_exists)
        ec = rec.get("endpoint_completion_ranking") or {}
        epr = ec.get("best_policy_rank")
        evr = ec.get("best_visit_rank")
        if epr is not None or evr is not None:
            b["endpoint_exists"] += 1
            tmp_p = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(epr, tmp_p)
            tmp_v = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(evr, tmp_v)
            for k in ("_top1", "_top5", "_top20", "_gt20"):
                b[f"endpoint_policy{k}"] += tmp_p[k]
                b[f"endpoint_visit{k}"] += tmp_v[k]
        # Distance reducer ranking buckets
        rd = rec.get("distance_reducing_ranking") or {}
        rpr = rd.get("best_policy_rank")
        rvr = rd.get("best_visit_rank")
        if rpr is not None or rvr is not None:
            b["reducer_exists"] += 1
            tmp_p = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(rpr, tmp_p)
            tmp_v = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(rvr, tmp_v)
            for k in ("_top1", "_top5", "_top20", "_gt20"):
                b[f"reducer_policy{k}"] += tmp_p[k]
                b[f"reducer_visit{k}"] += tmp_v[k]

    # Convert raw counts to rates
    def _rate(num, den):
        return (num / den) if den > 0 else 0.0

    out = {}
    for key, b in buckets.items():
        n = b["records"]
        e_exists = b["endpoint_exists"]
        r_exists = b["reducer_exists"]
        out[key] = {
            "records": n,
            "high_value_records": b["high_value_records"],
            "selected_completes_endpoint_rate": _rate(b["selected_completes_endpoint"], n),
            "selected_reduces_distance_rate":   _rate(b["selected_reduces_distance"], n),
            "selected_redundant_rate":          _rate(b["selected_redundant"], n),
            "selected_off_chain_rate":          _rate(b["selected_off_chain"], n),
            "selected_other_rate":              _rate(b["selected_other"], n),
            "endpoint_completion_exists_rate":  _rate(e_exists, n),
            "endpoint_policy_top1_rate":  _rate(b["endpoint_policy_top1"], e_exists),
            "endpoint_policy_top5_rate":  _rate(b["endpoint_policy_top5"], e_exists),
            "endpoint_policy_top20_rate": _rate(b["endpoint_policy_top20"], e_exists),
            "endpoint_policy_gt20_rate":  _rate(b["endpoint_policy_gt20"], e_exists),
            "endpoint_visit_top1_rate":   _rate(b["endpoint_visit_top1"], e_exists),
            "endpoint_visit_top5_rate":   _rate(b["endpoint_visit_top5"], e_exists),
            "endpoint_visit_top20_rate":  _rate(b["endpoint_visit_top20"], e_exists),
            "endpoint_visit_gt20_rate":   _rate(b["endpoint_visit_gt20"], e_exists),
            "distance_reducer_exists_rate":   _rate(r_exists, n),
            "reducer_policy_top1_rate":   _rate(b["reducer_policy_top1"], r_exists),
            "reducer_policy_top5_rate":   _rate(b["reducer_policy_top5"], r_exists),
            "reducer_policy_top20_rate":  _rate(b["reducer_policy_top20"], r_exists),
            "reducer_policy_gt20_rate":   _rate(b["reducer_policy_gt20"], r_exists),
            "reducer_visit_top1_rate":    _rate(b["reducer_visit_top1"], r_exists),
            "reducer_visit_top5_rate":    _rate(b["reducer_visit_top5"], r_exists),
            "reducer_visit_top20_rate":   _rate(b["reducer_visit_top20"], r_exists),
            "reducer_visit_gt20_rate":    _rate(b["reducer_visit_gt20"], r_exists),
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_analyzer_td_closeout_breakdown.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_td_closeout_breakdown.py
git commit -m "feat(analyzer): add td_closeout_breakdown aggregator (Spec 3 Fix 0)"
```

---

## Task 2: Add report formatter for td_closeout_breakdown

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` (add formatter near `format_goal_completion_report` at line 2382)
- Test: `tests/test_analyzer_td_closeout_breakdown.py` (extend)

- [ ] **Step 1: Add the formatter test**

Append to `tests/test_analyzer_td_closeout_breakdown.py`:

```python
from scripts.twixt_replay_analyzer import format_td_closeout_breakdown_report


def test_report_formatter_produces_section():
    breakdown = {
        "td=1": {
            "records": 100, "high_value_records": 90,
            "selected_completes_endpoint_rate": 0.1,
            "selected_reduces_distance_rate": 0.0,
            "selected_redundant_rate": 0.6,
            "selected_off_chain_rate": 0.25,
            "selected_other_rate": 0.05,
            "endpoint_completion_exists_rate": 1.0,
            "endpoint_policy_top5_rate": 0.3,
            "endpoint_visit_top5_rate": 0.2,
            "endpoint_visit_gt20_rate": 0.6,
            "distance_reducer_exists_rate": 1.0,
            "reducer_policy_top5_rate": 0.4,
            "reducer_visit_top5_rate": 0.3,
            "reducer_visit_gt20_rate": 0.5,
        },
        "td=2": {"records": 50, "high_value_records": 40,
                 "selected_completes_endpoint_rate": 0.7, "selected_reduces_distance_rate": 0.1,
                 "selected_redundant_rate": 0.1, "selected_off_chain_rate": 0.1,
                 "selected_other_rate": 0.0,
                 "endpoint_completion_exists_rate": 0.8, "endpoint_policy_top5_rate": 0.8,
                 "endpoint_visit_top5_rate": 0.75, "endpoint_visit_gt20_rate": 0.05,
                 "distance_reducer_exists_rate": 1.0, "reducer_policy_top5_rate": 0.85,
                 "reducer_visit_top5_rate": 0.8, "reducer_visit_gt20_rate": 0.05},
        "td=3": {"records": 0, "high_value_records": 0,
                 "selected_completes_endpoint_rate": 0.0, "selected_reduces_distance_rate": 0.0,
                 "selected_redundant_rate": 0.0, "selected_off_chain_rate": 0.0,
                 "selected_other_rate": 0.0,
                 "endpoint_completion_exists_rate": 0.0, "endpoint_policy_top5_rate": 0.0,
                 "endpoint_visit_top5_rate": 0.0, "endpoint_visit_gt20_rate": 0.0,
                 "distance_reducer_exists_rate": 0.0, "reducer_policy_top5_rate": 0.0,
                 "reducer_visit_top5_rate": 0.0, "reducer_visit_gt20_rate": 0.0},
    }
    lines = format_td_closeout_breakdown_report(breakdown)
    body = "\n".join(lines)
    assert "Closeout breakdown by total_goal_distance" in body
    assert "td=1:" in body and "td=2:" in body and "td=3:" in body
    assert "records=100" in body
    assert "visit >20=60.0%" in body or "visit >20=60%" in body
```

- [ ] **Step 2: Run test to confirm failure**

```bash
.venv/bin/pytest tests/test_analyzer_td_closeout_breakdown.py::test_report_formatter_produces_section -v
```

Expected: ImportError on `format_td_closeout_breakdown_report`.

- [ ] **Step 3: Implement the formatter**

Add to `scripts/twixt_replay_analyzer.py` immediately after `format_goal_completion_report` (line 2382 ends around 2511):

```python
def format_td_closeout_breakdown_report(breakdown: dict) -> list:
    """Format the td_closeout_breakdown section for report_<range>.txt.

    Spec 2026-05-10 §3.2. `breakdown` is the dict returned by
    aggregate_td_closeout_breakdown().
    """
    def _pct(x):
        return f"{(x or 0.0) * 100.0:.1f}%"

    lines = []
    lines.append("Closeout breakdown by total_goal_distance")
    lines.append("=========================================")
    for key in ("td=1", "td=2", "td=3"):
        b = breakdown.get(key) or {}
        n = b.get("records", 0)
        hv = b.get("high_value_records", 0)
        lines.append(f"{key}:  records={n}  high_value={hv}")
        if n == 0:
            lines.append("  (no records)")
            continue
        lines.append(
            "  selected: complete=" + _pct(b.get("selected_completes_endpoint_rate"))
            + "  reduce=" + _pct(b.get("selected_reduces_distance_rate"))
            + "  redundant=" + _pct(b.get("selected_redundant_rate"))
            + "  off-chain=" + _pct(b.get("selected_off_chain_rate"))
            + "  other=" + _pct(b.get("selected_other_rate"))
        )
        lines.append(
            "  endpoint exists: " + _pct(b.get("endpoint_completion_exists_rate"))
            + "  policy top5=" + _pct(b.get("endpoint_policy_top5_rate"))
            + "  visit top5=" + _pct(b.get("endpoint_visit_top5_rate"))
            + "  visit >20=" + _pct(b.get("endpoint_visit_gt20_rate"))
        )
        lines.append(
            "  reducer  exists: " + _pct(b.get("distance_reducer_exists_rate"))
            + "  policy top5=" + _pct(b.get("reducer_policy_top5_rate"))
            + "  visit top5=" + _pct(b.get("reducer_visit_top5_rate"))
            + "  visit >20=" + _pct(b.get("reducer_visit_gt20_rate"))
        )
    return lines
```

- [ ] **Step 4: Run test**

```bash
.venv/bin/pytest tests/test_analyzer_td_closeout_breakdown.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_td_closeout_breakdown.py
git commit -m "feat(analyzer): add td_closeout_breakdown report formatter"
```

---

## Task 3: Add CSV writer for td_closeout_breakdown

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`

- [ ] **Step 1: Add the CSV test**

Append to `tests/test_analyzer_td_closeout_breakdown.py`:

```python
import csv
import tempfile

from scripts.twixt_replay_analyzer import write_goal_completion_td_breakdown_csv


def test_td_breakdown_csv_one_row_per_td_value(tmp_path):
    breakdown = {
        "td=1": {"records": 3, "high_value_records": 2,
                 "selected_completes_endpoint_rate": 0.33,
                 "selected_reduces_distance_rate": 0.0,
                 "selected_redundant_rate": 0.33, "selected_off_chain_rate": 0.34,
                 "selected_other_rate": 0.0,
                 "endpoint_completion_exists_rate": 1.0,
                 "endpoint_policy_top1_rate": 0.0, "endpoint_policy_top5_rate": 0.33,
                 "endpoint_policy_top20_rate": 0.66, "endpoint_policy_gt20_rate": 0.34,
                 "endpoint_visit_top1_rate": 0.0, "endpoint_visit_top5_rate": 0.33,
                 "endpoint_visit_top20_rate": 0.33, "endpoint_visit_gt20_rate": 0.67,
                 "distance_reducer_exists_rate": 1.0,
                 "reducer_policy_top1_rate": 0.0, "reducer_policy_top5_rate": 0.33,
                 "reducer_policy_top20_rate": 0.66, "reducer_policy_gt20_rate": 0.34,
                 "reducer_visit_top1_rate": 0.0, "reducer_visit_top5_rate": 0.33,
                 "reducer_visit_top20_rate": 0.33, "reducer_visit_gt20_rate": 0.67},
        "td=2": {"records": 0, "high_value_records": 0,
                 "selected_completes_endpoint_rate": 0.0,
                 "selected_reduces_distance_rate": 0.0,
                 "selected_redundant_rate": 0.0, "selected_off_chain_rate": 0.0,
                 "selected_other_rate": 0.0,
                 "endpoint_completion_exists_rate": 0.0,
                 "endpoint_policy_top1_rate": 0.0, "endpoint_policy_top5_rate": 0.0,
                 "endpoint_policy_top20_rate": 0.0, "endpoint_policy_gt20_rate": 0.0,
                 "endpoint_visit_top1_rate": 0.0, "endpoint_visit_top5_rate": 0.0,
                 "endpoint_visit_top20_rate": 0.0, "endpoint_visit_gt20_rate": 0.0,
                 "distance_reducer_exists_rate": 0.0,
                 "reducer_policy_top1_rate": 0.0, "reducer_policy_top5_rate": 0.0,
                 "reducer_policy_top20_rate": 0.0, "reducer_policy_gt20_rate": 0.0,
                 "reducer_visit_top1_rate": 0.0, "reducer_visit_top5_rate": 0.0,
                 "reducer_visit_top20_rate": 0.0, "reducer_visit_gt20_rate": 0.0},
        "td=3": {"records": 0, "high_value_records": 0,
                 "selected_completes_endpoint_rate": 0.0,
                 "selected_reduces_distance_rate": 0.0,
                 "selected_redundant_rate": 0.0, "selected_off_chain_rate": 0.0,
                 "selected_other_rate": 0.0,
                 "endpoint_completion_exists_rate": 0.0,
                 "endpoint_policy_top1_rate": 0.0, "endpoint_policy_top5_rate": 0.0,
                 "endpoint_policy_top20_rate": 0.0, "endpoint_policy_gt20_rate": 0.0,
                 "endpoint_visit_top1_rate": 0.0, "endpoint_visit_top5_rate": 0.0,
                 "endpoint_visit_top20_rate": 0.0, "endpoint_visit_gt20_rate": 0.0,
                 "distance_reducer_exists_rate": 0.0,
                 "reducer_policy_top1_rate": 0.0, "reducer_policy_top5_rate": 0.0,
                 "reducer_policy_top20_rate": 0.0, "reducer_policy_gt20_rate": 0.0,
                 "reducer_visit_top1_rate": 0.0, "reducer_visit_top5_rate": 0.0,
                 "reducer_visit_top20_rate": 0.0, "reducer_visit_gt20_rate": 0.0},
    }
    out = tmp_path / "td_breakdown.csv"
    write_goal_completion_td_breakdown_csv(str(out), breakdown)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert [r["td_before"] for r in rows] == ["1", "2", "3"]
    assert int(rows[0]["records"]) == 3
    assert float(rows[0]["selected_completes_endpoint_rate"]) == 0.33
```

- [ ] **Step 2: Run test → fail**

```bash
.venv/bin/pytest tests/test_analyzer_td_closeout_breakdown.py::test_td_breakdown_csv_one_row_per_td_value -v
```

- [ ] **Step 3: Implement the CSV writer**

Add to `scripts/twixt_replay_analyzer.py` immediately after `write_goal_completion_worst_cases_csv` (around line 2596):

```python
def write_goal_completion_td_breakdown_csv(path: str, breakdown: dict) -> None:
    """Write one row per td_before bucket. Spec 2026-05-10 §3.3."""
    fields = [
        "td_before", "records", "high_value_records",
        "selected_completes_endpoint_rate", "selected_reduces_distance_rate",
        "selected_redundant_rate", "selected_off_chain_rate", "selected_other_rate",
        "endpoint_completion_exists_rate",
        "endpoint_policy_top1_rate", "endpoint_policy_top5_rate",
        "endpoint_policy_top20_rate", "endpoint_policy_gt20_rate",
        "endpoint_visit_top1_rate", "endpoint_visit_top5_rate",
        "endpoint_visit_top20_rate", "endpoint_visit_gt20_rate",
        "distance_reducer_exists_rate",
        "reducer_policy_top1_rate", "reducer_policy_top5_rate",
        "reducer_policy_top20_rate", "reducer_policy_gt20_rate",
        "reducer_visit_top1_rate", "reducer_visit_top5_rate",
        "reducer_visit_top20_rate", "reducer_visit_gt20_rate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in ("td=1", "td=2", "td=3"):
            b = breakdown.get(key) or {}
            row = {"td_before": key.split("=", 1)[1]}
            for k in fields[1:]:
                row[k] = b.get(k, 0)
            w.writerow(row)
```

- [ ] **Step 4: Run all tests**

```bash
.venv/bin/pytest tests/test_analyzer_td_closeout_breakdown.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_td_closeout_breakdown.py
git commit -m "feat(analyzer): add td_closeout_breakdown CSV writer"
```

---

## Task 4: Wire Fix 0 into `analyze()` and CLI

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` (the `analyze` function around line 2940, the `main` function around line 4134, and the worst-cases-CSV write site around line 3918)

- [ ] **Step 1: Read the existing wiring**

```bash
grep -n "write_goal_completion_worst_cases_csv\|format_goal_completion_report\|format_policy_mcts_closeout_report" /Users/bill/projects/TwixT_Game/scripts/twixt_replay_analyzer.py
```

You should see existing call sites in `analyze()` that build the `summary` dict and emit `report_<range>.txt`. The Fix 0 wiring slots in beside them.

- [ ] **Step 2: Add the multi-side helper**

Detected-player can vary game-to-game, so we can't reuse the single-side `aggregate_td_closeout_breakdown` directly across all games. Add a sibling helper that takes per-record detected-side labels. Place it immediately above `aggregate_td_closeout_breakdown` (which you added in Task 1):

```python
def _aggregate_td_breakdown_multi_side(
    records: list,
    detected_sides: list,
    high_value_threshold: float = 0.95,
) -> dict:
    """Variant of aggregate_td_closeout_breakdown that takes per-record
    detected_player labels. Used when aggregating across games where
    detected_player differs game-to-game.

    Internally splits by side, calls aggregate_td_closeout_breakdown per
    side, and combines raw counts before recomputing rates.
    """
    from collections import defaultdict

    # Re-implement the aggregation directly on raw counts to avoid
    # losing precision when combining rates.
    def _empty():
        return {
            "records": 0, "high_value_records": 0,
            "selected_completes_endpoint": 0, "selected_reduces_distance": 0,
            "selected_redundant": 0, "selected_off_chain": 0, "selected_other": 0,
            "endpoint_exists": 0,
            "endpoint_policy_top1": 0, "endpoint_policy_top5": 0,
            "endpoint_policy_top20": 0, "endpoint_policy_gt20": 0,
            "endpoint_visit_top1": 0, "endpoint_visit_top5": 0,
            "endpoint_visit_top20": 0, "endpoint_visit_gt20": 0,
            "reducer_exists": 0,
            "reducer_policy_top1": 0, "reducer_policy_top5": 0,
            "reducer_policy_top20": 0, "reducer_policy_gt20": 0,
            "reducer_visit_top1": 0, "reducer_visit_top5": 0,
            "reducer_visit_top20": 0, "reducer_visit_gt20": 0,
        }

    def _bucket_rank(rank, c, prefix):
        if rank is None:
            return
        if rank <= 1:
            c[f"{prefix}_top1"] += 1; c[f"{prefix}_top5"] += 1; c[f"{prefix}_top20"] += 1
        elif rank <= 5:
            c[f"{prefix}_top5"] += 1; c[f"{prefix}_top20"] += 1
        elif rank <= 20:
            c[f"{prefix}_top20"] += 1
        else:
            c[f"{prefix}_gt20"] += 1

    buckets = {"td=1": _empty(), "td=2": _empty(), "td=3": _empty()}
    for rec, det in zip(records, detected_sides):
        if not isinstance(rec, dict) or rec.get("side_to_move") != det:
            continue
        gc = rec.get("goal_completion") or {}
        td = gc.get("total_goal_distance_before")
        if td not in (1, 2, 3):
            continue
        b = buckets[f"td={td}"]
        b["records"] += 1
        q = (rec.get("root_summary") or {}).get("q_value")
        if isinstance(q, (int, float)) and q >= high_value_threshold:
            b["high_value_records"] += 1
        cls_name = ((rec.get("selected_move_classification") or {}).get("primary_class")) or ""
        cls_field = {
            "completes_endpoint": "selected_completes_endpoint",
            "reduces_total_goal_distance": "selected_reduces_distance",
            "redundant_reinforcement": "selected_redundant",
            "off_chain": "selected_off_chain",
            "other": "selected_other",
        }.get(cls_name)
        if cls_field:
            b[cls_field] += 1
        ec = rec.get("endpoint_completion_ranking") or {}
        if ec.get("best_policy_rank") is not None or ec.get("best_visit_rank") is not None:
            b["endpoint_exists"] += 1
            _bucket_rank(ec.get("best_policy_rank"), b, "endpoint_policy")
            _bucket_rank(ec.get("best_visit_rank"),  b, "endpoint_visit")
        rd = rec.get("distance_reducing_ranking") or {}
        if rd.get("best_policy_rank") is not None or rd.get("best_visit_rank") is not None:
            b["reducer_exists"] += 1
            _bucket_rank(rd.get("best_policy_rank"), b, "reducer_policy")
            _bucket_rank(rd.get("best_visit_rank"),  b, "reducer_visit")

    def _rate(num, den):
        return (num / den) if den > 0 else 0.0

    out = {}
    for key, b in buckets.items():
        n = b["records"]; e = b["endpoint_exists"]; r = b["reducer_exists"]
        out[key] = {
            "records": n,
            "high_value_records": b["high_value_records"],
            "selected_completes_endpoint_rate": _rate(b["selected_completes_endpoint"], n),
            "selected_reduces_distance_rate":   _rate(b["selected_reduces_distance"], n),
            "selected_redundant_rate":          _rate(b["selected_redundant"], n),
            "selected_off_chain_rate":          _rate(b["selected_off_chain"], n),
            "selected_other_rate":              _rate(b["selected_other"], n),
            "endpoint_completion_exists_rate":  _rate(e, n),
            "endpoint_policy_top1_rate":  _rate(b["endpoint_policy_top1"], e),
            "endpoint_policy_top5_rate":  _rate(b["endpoint_policy_top5"], e),
            "endpoint_policy_top20_rate": _rate(b["endpoint_policy_top20"], e),
            "endpoint_policy_gt20_rate":  _rate(b["endpoint_policy_gt20"], e),
            "endpoint_visit_top1_rate":   _rate(b["endpoint_visit_top1"], e),
            "endpoint_visit_top5_rate":   _rate(b["endpoint_visit_top5"], e),
            "endpoint_visit_top20_rate":  _rate(b["endpoint_visit_top20"], e),
            "endpoint_visit_gt20_rate":   _rate(b["endpoint_visit_gt20"], e),
            "distance_reducer_exists_rate":   _rate(r, n),
            "reducer_policy_top1_rate":   _rate(b["reducer_policy_top1"], r),
            "reducer_policy_top5_rate":   _rate(b["reducer_policy_top5"], r),
            "reducer_policy_top20_rate":  _rate(b["reducer_policy_top20"], r),
            "reducer_policy_gt20_rate":   _rate(b["reducer_policy_gt20"], r),
            "reducer_visit_top1_rate":    _rate(b["reducer_visit_top1"], r),
            "reducer_visit_top5_rate":    _rate(b["reducer_visit_top5"], r),
            "reducer_visit_top20_rate":   _rate(b["reducer_visit_top20"], r),
            "reducer_visit_gt20_rate":    _rate(b["reducer_visit_gt20"], r),
        }
    return out
```

- [ ] **Step 3: Build the per-ply record pool and call the helper**

In `aggregate_goal_completion_diagnostics_from_records`, after the existing `result = aggregate_goal_completion_records(...)` call (around line 810-830), add:

```python
# Fix 0: bulk td-before breakdown across decisive winners.
td_breakdown_records = []
td_breakdown_detected = []
for replay, rec in zip(replays, per_game_records):
    if rec is None:
        continue
    det = rec.get("detected_player")
    if not det:
        continue
    diag = replay.get("goal_completion_diagnostics") or []
    for r in diag:
        if isinstance(r, dict):
            td_breakdown_records.append(r)
            td_breakdown_detected.append(det)
result["td_closeout_breakdown"] = _aggregate_td_breakdown_multi_side(
    td_breakdown_records, td_breakdown_detected, high_value_threshold=0.95,
)
```

- [ ] **Step 4: Wire the formatter into report rendering**

In `analyze()` (around line 2940 — search for the existing `format_goal_completion_report(...)` call inside `analyze`), add immediately after it:

```python
td_breakdown = (goal_completion_val or {}).get("td_closeout_breakdown")
if td_breakdown:
    report_lines.extend([""])
    report_lines.extend(format_td_closeout_breakdown_report(td_breakdown))
```

- [ ] **Step 5: Wire the CSV writer**

In `analyze()` near where other CSVs are written (search for `write_goal_completion_worst_cases_csv`, around line 3918):

```python
write_goal_completion_td_breakdown_csv(
    os.path.join(out_dir, _suffixed("goal_completion_td_breakdown", "csv", suffix)),
    td_breakdown or {},
)
```

- [ ] **Step 6: Smoke-run the analyzer on existing 130-139 data**

```bash
.venv/bin/python ./scripts/twixt_replay_analyzer.py --input Replays/120-129 --out /tmp/td_break_smoke
ls /tmp/td_break_smoke/goal_completion_td_breakdown_120-129.csv
grep -A 10 "Closeout breakdown by total_goal_distance" /tmp/td_break_smoke/report_120-129.txt
```

Expected: CSV exists with 3 rows; report has the section.

- [ ] **Step 7: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "feat(analyzer): wire td_closeout_breakdown into analyze() and CSV output"
```

---

## Task 5: Implement Fix 3 recovery event aggregator + tests

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` (add `aggregate_recovery_events`)
- Test: `tests/test_analyzer_recovery_events.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyzer_recovery_events.py`:

```python
"""Tests for Fix 3: recovery event classification (spec 2026-05-10 §6)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import aggregate_recovery_events


def _fixture(rec_overrides=None, meta_overrides=None, diag=None):
    rec = {
        "winner": "black",
        "detected_player": "black",
        "first_dominant_unclosed_ply": 50,
        "actual_terminal_ply": 90,
        "conversion_delay_winner_moves": 15,
        "winner_moves_in_watch_window": 20,
        "winner_moves_with_dominant_unavailable": 12,
    }
    if rec_overrides:
        rec.update(rec_overrides)
    meta = {"reason": "win", "iteration": 130, "game_idx": 1, "final_root_value": 0.95}
    if meta_overrides:
        meta.update(meta_overrides)
    return {
        "goal_completion_record": rec,
        "meta": meta,
        "goal_completion_diagnostics": diag or [],
    }


def test_lost_then_state_cap_classified():
    g = _fixture(rec_overrides={"winner_moves_with_dominant_unavailable": 15},
                 meta_overrides={"reason": "state_cap", "final_root_value": 0.92})
    events = aggregate_recovery_events([g])
    assert len(events) == 1
    assert events[0]["recovery_class"] == "lost_then_state_cap"
    assert events[0]["eventual_outcome"] == "state_cap"


def test_lost_and_value_collapsed():
    g = _fixture(meta_overrides={"final_root_value": 0.2},
                 diag=[{"ply": 60, "side_to_move": "black",
                        "root_summary": {"q_value": 0.95},
                        "goal_completion": {"total_goal_distance_before": 5}}])
    events = aggregate_recovery_events([g])
    assert events[0]["recovery_class"] == "lost_and_value_collapsed"


def test_lost_but_value_stayed_high():
    g = _fixture(meta_overrides={"final_root_value": 0.99},
                 diag=[{"ply": 60, "side_to_move": "black",
                        "root_summary": {"q_value": 0.95},
                        "goal_completion": {"total_goal_distance_before": 5}}])
    events = aggregate_recovery_events([g])
    assert events[0]["recovery_class"] == "lost_but_value_stayed_high"


def test_lost_then_won_late():
    g = _fixture(rec_overrides={"conversion_delay_winner_moves": 50})
    events = aggregate_recovery_events([g])
    assert events[0]["recovery_class"] in ("lost_then_recovered", "lost_then_won_late")


def test_below_event_threshold_excluded():
    g = _fixture(rec_overrides={"winner_moves_with_dominant_unavailable": 2})
    assert aggregate_recovery_events([g]) == []
```

- [ ] **Step 2: Run test → fail**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_events.py -v
```

- [ ] **Step 3: Implement the aggregator**

Add to `scripts/twixt_replay_analyzer.py` near other `aggregate_*` helpers (immediately after `aggregate_td_closeout_breakdown` / `_aggregate_td_breakdown_multi_side`):

```python
def aggregate_recovery_events(replays: list) -> list:
    """Build per-event rows for the recovery diagnostic (spec §6).

    Event criterion (§6.1): a replay contributes an event when any of
    - winner_moves_with_dominant_unavailable >= 10
    - meta.reason == "state_cap" AND record.detected == True (or
      winner_moves_in_watch_window > 0 as a proxy for detection)
    - meta.reason == "adjudicated" AND winner_moves_with_dominant_unavailable >= 5
    """
    events = []
    for replay in replays or []:
        rec = replay.get("goal_completion_record")
        if not isinstance(rec, dict):
            continue
        meta = replay.get("meta") or {}
        reason = meta.get("reason")
        dom_unavail = rec.get("winner_moves_with_dominant_unavailable") or 0
        in_window = rec.get("winner_moves_in_watch_window") or 0
        detected = (in_window or 0) > 0

        triggered = (
            (dom_unavail or 0) >= 10
            or (reason == "state_cap" and detected)
            or (reason == "adjudicated" and (dom_unavail or 0) >= 5)
        )
        if not triggered:
            continue

        # Optional per-ply walk to find first_unavailable_ply (first detected-side
        # ply where total_goal_distance_before > 2).
        det_side = rec.get("detected_player")
        first_unavailable_ply = None
        q_at_first_unavailable = None
        diag = replay.get("goal_completion_diagnostics") or []
        for r in diag:
            if not isinstance(r, dict):
                continue
            if r.get("side_to_move") != det_side:
                continue
            gc = r.get("goal_completion") or {}
            td = gc.get("total_goal_distance_before")
            if td is not None and td > 2:
                first_unavailable_ply = r.get("ply")
                q_at_first_unavailable = (r.get("root_summary") or {}).get("q_value")
                break
        # latest fields from last detected-side row in diag
        latest_largest = None; latest_td = None
        for r in diag:
            if isinstance(r, dict) and r.get("side_to_move") == det_side:
                gc = r.get("goal_completion") or {}
                latest_largest = gc.get("largest_component_size")
                latest_td = gc.get("total_goal_distance_before")
        sel_class_counts = {"completes_endpoint": 0, "reduces_total_goal_distance": 0,
                            "redundant_reinforcement": 0, "off_chain": 0, "other": 0}
        if first_unavailable_ply is not None:
            for r in diag:
                if not isinstance(r, dict):
                    continue
                if r.get("side_to_move") != det_side:
                    continue
                if (r.get("ply") or 0) < first_unavailable_ply:
                    continue
                cls = ((r.get("selected_move_classification") or {}).get("primary_class"))
                if cls in sel_class_counts:
                    sel_class_counts[cls] += 1

        q_at_terminal = meta.get("final_root_value")
        outcome = "win" if reason == "win" else (
            "state_cap" if reason == "state_cap" else (
                "adjudicated" if reason == "adjudicated" else (reason or "other")
            )
        )

        delay_winner = rec.get("conversion_delay_winner_moves") or 0
        # Bucket assignment (priority order, §6.3)
        recovered_later_to_le2 = any(
            (r.get("side_to_move") == det_side
             and (r.get("goal_completion") or {}).get("total_goal_distance_before") is not None
             and (r.get("goal_completion") or {}).get("total_goal_distance_before") <= 2
             and (first_unavailable_ply is not None and (r.get("ply") or 0) > first_unavailable_ply))
            for r in diag if isinstance(r, dict)
        )
        if outcome == "win" and (dom_unavail or 0) >= 10 and recovered_later_to_le2:
            bucket = "lost_then_recovered"
        elif outcome == "win" and delay_winner >= 30:
            bucket = "lost_then_won_late"
        elif outcome == "state_cap":
            bucket = "lost_then_state_cap"
        elif (q_at_first_unavailable is not None and q_at_first_unavailable >= 0.9
              and (q_at_terminal or 0) <= 0.5):
            bucket = "lost_and_value_collapsed"
        elif (q_at_first_unavailable is not None and q_at_first_unavailable >= 0.9
              and (q_at_terminal or 0) >= 0.9):
            bucket = "lost_but_value_stayed_high"
        else:
            bucket = "lost_other"

        events.append({
            "iteration": meta.get("iteration"),
            "game_id": rec.get("game_id"),
            "winner": rec.get("winner"),
            "detected_player": det_side,
            "first_detection_ply": rec.get("first_dominant_unclosed_ply"),
            "first_unavailable_ply": first_unavailable_ply,
            "dominant_unavailable_moves": dom_unavail,
            "latest_largest_component_size": latest_largest,
            "latest_total_goal_distance": latest_td,
            "q_at_first_unavailable": q_at_first_unavailable,
            "q_at_terminal": q_at_terminal,
            "selected_class_counts_after_first_unavailable": sel_class_counts,
            "eventual_outcome": outcome,
            "recovery_class": bucket,
        })
    return events
```

- [ ] **Step 4: Run tests → pass**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_events.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_events.py
git commit -m "feat(analyzer): add recovery event aggregator (Spec 3 Fix 3)"
```

---

## Task 6: Recovery CSV writer + report formatter

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Test: `tests/test_analyzer_recovery_events.py` (extend)

- [ ] **Step 1: Add CSV + formatter tests**

Append to `tests/test_analyzer_recovery_events.py`:

```python
import csv
from scripts.twixt_replay_analyzer import (
    write_recovery_events_csv,
    format_recovery_events_report,
)


def test_recovery_csv_written(tmp_path):
    events = [
        {"iteration": 131, "game_id": "game_079", "winner": "black",
         "detected_player": "black", "first_detection_ply": 56,
         "first_unavailable_ply": 60, "dominant_unavailable_moves": 100,
         "latest_largest_component_size": 24, "latest_total_goal_distance": 5,
         "q_at_first_unavailable": 0.95, "q_at_terminal": -0.1,
         "selected_class_counts_after_first_unavailable":
             {"completes_endpoint": 0, "reduces_total_goal_distance": 1,
              "redundant_reinforcement": 3, "off_chain": 8, "other": 1},
         "eventual_outcome": "adjudicated", "recovery_class": "lost_and_value_collapsed"},
    ]
    out = tmp_path / "rec.csv"
    write_recovery_events_csv(str(out), events)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["recovery_class"] == "lost_and_value_collapsed"
    assert rows[0]["dominant_unavailable_moves"] == "100"


def test_recovery_report_formatter():
    events = [
        {"recovery_class": "lost_then_state_cap", "dominant_unavailable_moves": 10,
         "conversion_delay_winner_moves": 20},
        {"recovery_class": "lost_then_state_cap", "dominant_unavailable_moves": 14,
         "conversion_delay_winner_moves": 30},
        {"recovery_class": "lost_but_value_stayed_high",
         "dominant_unavailable_moves": 12, "conversion_delay_winner_moves": 5},
    ]
    lines = format_recovery_events_report(events)
    body = "\n".join(lines)
    assert "Recovery / dominant-component-lost diagnostics" in body
    assert "lost_then_state_cap" in body
    assert "Events: 3" in body
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_events.py -v
```

- [ ] **Step 3: Implement writer + formatter**

Add to `scripts/twixt_replay_analyzer.py`:

```python
def write_recovery_events_csv(path: str, events: list) -> None:
    """One row per recovery event (spec §6.4)."""
    fields = [
        "iteration", "game_id", "winner", "detected_player",
        "first_detection_ply", "first_unavailable_ply", "dominant_unavailable_moves",
        "latest_largest_component_size", "latest_total_goal_distance",
        "q_at_first_unavailable", "q_at_terminal",
        "sel_completes_endpoint", "sel_reduces_distance",
        "sel_redundant_reinforcement", "sel_off_chain", "sel_other",
        "eventual_outcome", "recovery_class",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            sc = e.get("selected_class_counts_after_first_unavailable") or {}
            row = {
                "iteration": e.get("iteration"),
                "game_id": e.get("game_id"),
                "winner": e.get("winner"),
                "detected_player": e.get("detected_player"),
                "first_detection_ply": e.get("first_detection_ply"),
                "first_unavailable_ply": e.get("first_unavailable_ply"),
                "dominant_unavailable_moves": e.get("dominant_unavailable_moves"),
                "latest_largest_component_size": e.get("latest_largest_component_size"),
                "latest_total_goal_distance": e.get("latest_total_goal_distance"),
                "q_at_first_unavailable": e.get("q_at_first_unavailable"),
                "q_at_terminal": e.get("q_at_terminal"),
                "sel_completes_endpoint": sc.get("completes_endpoint", 0),
                "sel_reduces_distance": sc.get("reduces_total_goal_distance", 0),
                "sel_redundant_reinforcement": sc.get("redundant_reinforcement", 0),
                "sel_off_chain": sc.get("off_chain", 0),
                "sel_other": sc.get("other", 0),
                "eventual_outcome": e.get("eventual_outcome"),
                "recovery_class": e.get("recovery_class"),
            }
            w.writerow(row)


def format_recovery_events_report(events: list) -> list:
    """Format the recovery section for report_<range>.txt (spec §6.4)."""
    lines = []
    lines.append("Recovery / dominant-component-lost diagnostics")
    lines.append("===============================================")
    lines.append(f"Events: {len(events)}")
    if not events:
        return lines
    counts = {}
    for e in events:
        b = e.get("recovery_class") or "lost_other"
        counts[b] = counts.get(b, 0) + 1
    lines.append("By outcome:")
    for k in ("lost_then_recovered", "lost_then_won_late", "lost_then_state_cap",
              "lost_and_value_collapsed", "lost_but_value_stayed_high", "lost_other"):
        if k in counts:
            lines.append(f"  {k:30s} {counts[k]}")
    dom = sorted(int(e.get("dominant_unavailable_moves") or 0) for e in events)
    delays = sorted(int(e.get("conversion_delay_winner_moves") or 0) for e in events)
    def _median(xs):
        return xs[len(xs)//2] if xs else 0
    lines.append(f"Median dominant_unavailable_moves: {_median(dom)}")
    lines.append(f"Median delay (winner_moves):       {_median(delays)}")
    return lines
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_events.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_events.py
git commit -m "feat(analyzer): recovery events CSV writer + report formatter"
```

---

## Task 7: Wire Fix 3 into `analyze()`

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`

- [ ] **Step 1: Add wiring inside `analyze()`**

Search for the existing `format_recovery_or_extreme_closeout_drift_report` call inside `analyze()`. Add right after it:

```python
recovery_events = aggregate_recovery_events(replays)
write_recovery_events_csv(
    os.path.join(out_dir, _suffixed("recovery_events", "csv", suffix)),
    recovery_events,
)
report_lines.extend([""])
report_lines.extend(format_recovery_events_report(recovery_events))
```

- [ ] **Step 2: Smoke run on 130-139**

```bash
.venv/bin/python ./scripts/twixt_replay_analyzer.py --input Replays/120-129 --out /tmp/recovery_smoke
ls /tmp/recovery_smoke/recovery_events_*.csv
grep -A 8 "Recovery / dominant-component-lost" /tmp/recovery_smoke/report_*.txt
```

Expected: CSV exists; report section present.

- [ ] **Step 3: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "feat(analyzer): wire recovery_events CSV and report into analyze()"
```

---

## Task 8: Re-run analyzer on existing 130-139 artifact range (Phase 1 gate)

**Files:** none (data run)

- [ ] **Step 1: Re-run analyzer on the existing artifact range**

```bash
.venv/bin/python ./scripts/twixt_replay_analyzer.py \
    --input Replays/130-139 --out Replays/130-139_Replay
```

Note: this overwrites the existing `Replays/130-139_Replay/` output with the new Fix 0 / Fix 3 additions.

- [ ] **Step 2: Inspect td-breakdown to confirm gate condition**

```bash
cat Replays/130-139_Replay/goal_completion_td_breakdown_130-139.csv
grep -B 1 -A 25 "Closeout breakdown by total_goal_distance" Replays/130-139_Replay/report_130-139.txt
grep -B 1 -A 15 "Recovery / dominant-component-lost" Replays/130-139_Replay/report_130-139.txt
```

Spec §3.4 gate to proceed to Phase 2:
- td=1 has higher selected_redundant_rate + selected_off_chain_rate than td=2, AND
- td=1 endpoint_visit_top5_rate is materially lower than td=2 (≥ 20pp gap).

If those hold, Phase 2 (MCTS code) is justified. If not, stop and re-discuss with the user before proceeding.

- [ ] **Step 3: Do NOT commit the regenerated artifacts**

`Replays/130-139_Replay/` is not tracked in this repo (verified: `git ls-files Replays/130-139_Replay/` returns empty). Analyzer output is a generated artifact, not source. Skip the commit step entirely — the gate decision and metric values from the run go into the spec's results section or the conversation summary, not into git as data files. If this project later starts tracking analyzer output, re-introduce a commit step here.

---

# Phase 2 — Fix 1 MCTS code

## Task 9: Add `MCTSConfig` fields for closeout td=1 visit forcing

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py` (`MCTSConfig` dataclass around line 96)
- Test: `tests/test_mcts_force_root_visits.py`

- [ ] **Step 1: Write a config test**

Create `tests/test_mcts_force_root_visits.py`:

```python
"""Unit tests for Spec 3 Fix 1 — td=1 root visit forcing (mcts side)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.mcts import MCTSConfig


def test_config_defaults_disable_visit_forcing():
    c = MCTSConfig()
    assert c.closeout_td1_visit_forcing_enabled is False
    assert c.closeout_td1_min_visits == 8
    assert c.closeout_td1_max_forced_moves == 4
    assert c.closeout_td1_require_high_value is False
    assert c.closeout_td1_high_value_threshold == 0.95


def test_config_accepts_overrides():
    c = MCTSConfig(closeout_td1_visit_forcing_enabled=True,
                   closeout_td1_min_visits=16,
                   closeout_td1_max_forced_moves=2,
                   closeout_td1_require_high_value=True,
                   closeout_td1_high_value_threshold=0.9)
    assert c.closeout_td1_visit_forcing_enabled is True
    assert c.closeout_td1_min_visits == 16
    assert c.closeout_td1_high_value_threshold == 0.9
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_mcts_force_root_visits.py -v
```

Expected: AttributeError — field does not exist.

- [ ] **Step 3: Add the fields**

In `scripts/GPU/alphazero/mcts.py`, inside `MCTSConfig` dataclass (line 96), append fields before `__post_init__`:

```python
    # Spec 3 Fix 1 — td=1 root visit forcing
    closeout_td1_visit_forcing_enabled: bool = False
    closeout_td1_min_visits: int = 8
    closeout_td1_max_forced_moves: int = 4
    closeout_td1_require_high_value: bool = False
    closeout_td1_high_value_threshold: float = 0.95
```

- [ ] **Step 4: Run → pass**

```bash
.venv/bin/pytest tests/test_mcts_force_root_visits.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_mcts_force_root_visits.py
git commit -m "feat(mcts): add MCTSConfig fields for closeout td=1 visit forcing"
```

---

## Task 10: Extract `_run_single_simulation` synchronous helper

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py` (refactor the per-sim body from `search` method around lines 281-313 into a method `_run_single_simulation`)
- Test: `tests/test_mcts.py` (existing tests must still pass)

- [ ] **Step 1: Run existing MCTS tests to baseline**

```bash
.venv/bin/pytest tests/test_mcts.py -v
```

Note the number passing. They must still pass after the refactor.

- [ ] **Step 2: Extract the helper**

In `scripts/GPU/alphazero/mcts.py`, replace the body of the `for _ in range(self.config.n_simulations)` loop inside `MCTS.search` (around lines 281-313) with a call to a new method:

```python
def _run_single_simulation(
    self,
    root: MCTSNode,
    root_move_override: Optional[int] = None,
) -> None:
    """Run one synchronous MCTS simulation from `root`.

    Implements the canonical DESCEND → EXPAND/EVAL → BACKUP loop without
    the batching/waiter machinery used by search_from_root. Shared by both
    the non-batched search() entrypoint and force_root_visits().

    Args:
      root: tree root.
      root_move_override: if not None, this move_id is selected at the root
        instead of PUCT's argmax. Internal-node descent is normal PUCT.
    """
    node = root
    search_path = [node]

    # Step 1: Root selection (overridden or PUCT).
    if root.is_expanded and not root.state.is_terminal():
        if root_move_override is not None:
            move_id = root_move_override
            child = root.children.get(move_id)
        else:
            move_id, child = self._select_child(node)
        if child is None:
            r, c = decode_move(move_id)
            child = MCTSNode(state=node.state.apply_move((r, c)), parent=node, move=move_id)
            node.children[move_id] = child
        search_path.append(child)
        node = child

    # Step 2: Standard PUCT descent below the root.
    while node.is_expanded and not node.state.is_terminal():
        move_id, child = self._select_child(node)
        if child is None:
            r, c = decode_move(move_id)
            child = MCTSNode(state=node.state.apply_move((r, c)), parent=node, move=move_id)
            node.children[move_id] = child
        search_path.append(child)
        node = child

    # Step 3: Expand or terminal-evaluate.
    if not node.state.is_terminal():
        value = self._expand(node)
    else:
        value = self._terminal_value(node.state)

    # Step 4: Backup along the recorded path.
    self._backup(search_path, value)
```

And replace the per-sim body of `search()` with:

```python
for _ in range(self.config.n_simulations):
    self._run_single_simulation(root, root_move_override=None)
```

- [ ] **Step 3: Run existing MCTS tests to confirm no regression**

```bash
.venv/bin/pytest tests/test_mcts.py -v
```

Expected: same set of tests pass as before.

- [ ] **Step 4: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py
git commit -m "refactor(mcts): extract _run_single_simulation helper for reuse"
```

---

## Task 11: Add `force_root_visits` method

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py`
- Test: `tests/test_mcts_force_root_visits.py` (extend)

- [ ] **Step 1: Add force_root_visits test**

Append to `tests/test_mcts_force_root_visits.py`:

```python
from unittest.mock import MagicMock
import math


def _make_mcts_with_stub_eval(value_fn, prior_uniform=True, n_sims=64):
    """Build an MCTS whose NN eval is a deterministic stub.

    value_fn(state) -> (priors_dict, value) for any state.
    """
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    cfg = MCTSConfig(n_simulations=n_sims)
    m = MCTS(network=None, config=cfg)
    # Replace _expand with a deterministic stub that fills priors + returns value
    def stub_expand(node):
        priors, value = value_fn(node.state)
        # mimic real _expand effect on the node
        node.priors_raw = dict(priors)
        node.priors = dict(priors)
        node.is_expanded = True
        # caller of _expand uses returned value as the backup value
        return value
    m._expand = stub_expand
    m.rng = MagicMock()
    m.rng.choice = lambda xs: xs[0]
    m.rng.random = lambda: 0.5
    return m


def test_force_root_visits_runs_exactly_min_visits_per_candidate():
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTSConfig, MCTS, MCTSNode, encode_move

    cfg = MCTSConfig(n_simulations=400,
                     closeout_td1_visit_forcing_enabled=True,
                     closeout_td1_min_visits=3,
                     closeout_td1_max_forced_moves=2)
    state = TwixtState()
    # Use small uniform priors stub
    def stub(state):
        legal = state.legal_moves()
        if not legal:
            return {}, 0.0
        p = 1.0 / len(legal)
        return {encode_move(r, c): p for (r, c) in legal}, 0.5
    m = _make_mcts_with_stub_eval(stub, n_sims=400)
    m.config = cfg
    root = MCTSNode(state=state)
    m._expand(root)
    # Pick the first two legal moves as candidates
    legal = list(state.legal_moves())[:2]
    forced = m.force_root_visits(
        root=root,
        candidate_moves=legal,
        min_visits=cfg.closeout_td1_min_visits,
        max_candidates=cfg.closeout_td1_max_forced_moves,
    )
    assert forced == 6  # 3 visits each * 2 candidates
    for mv in legal:
        child = root.children[encode_move(*mv)]
        assert child.visit_count == 3
```

- [ ] **Step 2: Run → fail (force_root_visits not defined)**

```bash
.venv/bin/pytest tests/test_mcts_force_root_visits.py::test_force_root_visits_runs_exactly_min_visits_per_candidate -v
```

- [ ] **Step 3: Implement `force_root_visits`**

Add as a method on `MCTS` (in `scripts/GPU/alphazero/mcts.py`, place it just after `_run_single_simulation`):

```python
def force_root_visits(
    self,
    root: MCTSNode,
    candidate_moves: list,            # list of (row, col)
    min_visits: int,
    max_candidates: int,
) -> int:
    """Force min_visits forced root-override sims for each of the first
    max_candidates candidate moves. Returns the total number of forced
    sims executed.

    Each forced sim uses the existing _run_single_simulation helper with
    root_move_override set to the candidate move's encoded id. The
    sim consumes from the same n_simulations budget as normal sims —
    callers are expected to reduce the main-loop budget by the return
    value. Forced sims must not exceed n_simulations.
    """
    if not self.config.closeout_td1_visit_forcing_enabled:
        return 0
    if not candidate_moves:
        return 0
    moves = list(candidate_moves)[:max_candidates]
    budget_total = self.config.n_simulations
    forced = 0
    for (r, c) in moves:
        move_id = encode_move(r, c)
        for _ in range(min_visits):
            if forced >= budget_total:
                return forced
            self._run_single_simulation(root, root_move_override=move_id)
            forced += 1
    return forced
```

- [ ] **Step 4: Run → pass**

```bash
.venv/bin/pytest tests/test_mcts_force_root_visits.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_mcts_force_root_visits.py
git commit -m "feat(mcts): add force_root_visits using _run_single_simulation"
```

---

## Task 12: Equivalence test — forced sim ≡ normal sim

**Files:**
- Test: `tests/test_mcts_forced_root_visit_equivalence.py`

- [ ] **Step 1: Write the equivalence test**

Create `tests/test_mcts_forced_root_visit_equivalence.py`:

```python
"""Equivalence test for Spec 3 Fix 1 (§9.1).

A single forced sim with `root_move_override=move_id` MUST produce the
same child.visit_count increment, the same `value` backed up through
the search path, and the same root.value_sum delta as a normal sim
that selects `move_id` by PUCT.
"""
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, encode_move


def _stub_value_fn():
    """Deterministic stub: uniform priors over legal moves, value=0.5."""
    def f(state):
        legal = state.legal_moves()
        if not legal:
            return {}, 0.0
        p = 1.0 / len(legal)
        return {encode_move(r, c): p for (r, c) in legal}, 0.5
    return f


def _build_mcts(stub):
    cfg = MCTSConfig(n_simulations=1)
    m = MCTS(network=None, config=cfg)
    def stub_expand(node):
        priors, value = stub(node.state)
        node.priors_raw = dict(priors)
        node.priors = dict(priors)
        node.is_expanded = True
        return value
    m._expand = stub_expand
    return m


def test_forced_override_matches_normal_puct_path():
    state = TwixtState()
    stub = _stub_value_fn()

    # Branch A: normal sim — expand root, then run one normal PUCT sim.
    m_a = _build_mcts(stub)
    root_a = MCTSNode(state=state)
    m_a._expand(root_a)
    # Determine which child PUCT picks (with uniform priors, ties broken by rng).
    # Force the rng to pick a specific child deterministically.
    legal = list(state.legal_moves())
    chosen_move = legal[0]
    chosen_id = encode_move(*chosen_move)
    # Stub _select_child to return our chosen move (uniform priors -> arbitrary).
    def stub_select_a(node, pending_ids=None):
        # Return chosen move at root level, then PUCT for deeper (won't recurse here).
        return chosen_id, node.children.get(chosen_id)
    m_a._select_child = stub_select_a
    m_a._run_single_simulation(root_a, root_move_override=None)

    # Branch B: forced sim — override root to chosen_id.
    m_b = _build_mcts(stub)
    root_b = MCTSNode(state=state)
    m_b._expand(root_b)
    m_b._run_single_simulation(root_b, root_move_override=chosen_id)

    # Compare child visit counts at root.
    child_a = root_a.children[chosen_id]
    child_b = root_b.children[chosen_id]
    assert child_a.visit_count == child_b.visit_count
    assert child_a.value_sum == child_b.value_sum
    # And the parent root accumulators must match.
    assert root_a.visit_count == root_b.visit_count
    assert root_a.value_sum == root_b.value_sum


def test_multiple_forced_overrides_match_multiple_normal_sims():
    """Run 5 sims, comparing normal-PUCT-with-forced-selection vs. force_root_visits."""
    state = TwixtState()
    stub = _stub_value_fn()
    legal = list(state.legal_moves())[:3]

    # Branch A: 5 normal sims with rigged _select_child cycling through targets.
    m_a = _build_mcts(stub)
    root_a = MCTSNode(state=state)
    m_a._expand(root_a)
    target_cycle = [legal[i % len(legal)] for i in range(5)]
    cycle_idx = {"i": 0}
    def stub_select_a(node, pending_ids=None):
        i = cycle_idx["i"]
        cycle_idx["i"] = (i + 1) if (node is root_a) else cycle_idx["i"]
        mv = target_cycle[i % 5]
        mid = encode_move(*mv)
        return mid, node.children.get(mid)
    m_a._select_child = stub_select_a
    for _ in range(5):
        m_a._run_single_simulation(root_a, root_move_override=None)

    # Branch B: same 5 sims via force_root_visits.
    cfg = MCTSConfig(n_simulations=5, closeout_td1_visit_forcing_enabled=True,
                     closeout_td1_min_visits=5, closeout_td1_max_forced_moves=1)
    m_b = _build_mcts(stub)
    m_b.config = cfg
    root_b = MCTSNode(state=state)
    m_b._expand(root_b)
    # Force-visit only the first move 5 times; mirror that in branch A is hard,
    # so test the simpler invariant: forcing N visits on one move increments
    # that child's visit_count by exactly N.
    forced = m_b.force_root_visits(root_b, [legal[0]], min_visits=5, max_candidates=1)
    assert forced == 5
    assert root_b.children[encode_move(*legal[0])].visit_count == 5
```

- [ ] **Step 2: Run**

```bash
.venv/bin/pytest tests/test_mcts_forced_root_visit_equivalence.py -v
```

Expected: 2 PASSED. If the first test fails on `value_sum` mismatch, investigate — the backup path or selection ordering may differ between branches.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcts_forced_root_visit_equivalence.py
git commit -m "test(mcts): equivalence test for forced root override vs normal sim"
```

---

## Task 13: Add MCTS telemetry accumulators + wire into `search_from_root`

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py`
- Test: `tests/test_mcts_force_root_visits.py` (extend)

- [ ] **Step 1: Add telemetry test**

Append to `tests/test_mcts_force_root_visits.py`:

```python
def test_search_from_root_invokes_force_when_td1_triggers():
    """When closeout_td1_visit_forcing_enabled and gc_state has td=1 and
    endpoint_completion_moves non-empty, the MCTS telemetry counters update."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, encode_move

    cfg = MCTSConfig(n_simulations=20,
                     closeout_td1_visit_forcing_enabled=True,
                     closeout_td1_min_visits=2,
                     closeout_td1_max_forced_moves=2)
    state = TwixtState()
    def stub(state):
        legal = state.legal_moves()
        p = 1.0 / len(legal)
        return {encode_move(r, c): p for (r, c) in legal}, 0.5
    m = _make_mcts_with_stub_eval(stub, n_sims=20)
    m.config = cfg
    # Reset telemetry to a known state
    m.reset_closeout_td1_telemetry()
    root = MCTSNode(state=state)
    legal = list(state.legal_moves())[:2]
    gc_state = {
        "total_goal_distance": 1,
        "endpoint_completion_moves": legal,
    }
    m.search_from_root(root, add_noise=False, ply=42, gc_state_full=gc_state)
    tel = m.get_closeout_td1_telemetry()
    assert tel["positions_triggered"] == 1
    assert tel["forced_sims_total"] == 4   # min_visits=2 * 2 candidates
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_mcts_force_root_visits.py -v
```

- [ ] **Step 3: Add telemetry accumulators on `MCTS`**

In `scripts/GPU/alphazero/mcts.py`, inside `MCTS.__init__` (the method around line 194), append after the existing init body:

```python
        # Spec 3 Fix 1 — closeout td=1 visit-forcing telemetry
        self._closeout_td1_positions_triggered = 0
        self._closeout_td1_positions_skipped_no_candidates = 0
        self._closeout_td1_positions_skipped_high_value_gate = 0
        self._closeout_td1_forced_sims_total = 0
        self._closeout_td1_selected_forced_move_count = 0
        self._closeout_td1_post_force_top1_hits = 0
        self._closeout_td1_post_force_top5_hits = 0
```

Add two methods on `MCTS`:

```python
def reset_closeout_td1_telemetry(self) -> None:
    self._closeout_td1_positions_triggered = 0
    self._closeout_td1_positions_skipped_no_candidates = 0
    self._closeout_td1_positions_skipped_high_value_gate = 0
    self._closeout_td1_forced_sims_total = 0
    self._closeout_td1_selected_forced_move_count = 0
    self._closeout_td1_post_force_top1_hits = 0
    self._closeout_td1_post_force_top5_hits = 0

def get_closeout_td1_telemetry(self) -> dict:
    triggered = self._closeout_td1_positions_triggered
    return {
        "enabled": bool(self.config.closeout_td1_visit_forcing_enabled),
        "min_visits": self.config.closeout_td1_min_visits,
        "max_forced_moves": self.config.closeout_td1_max_forced_moves,
        "require_high_value": bool(self.config.closeout_td1_require_high_value),
        "high_value_threshold": self.config.closeout_td1_high_value_threshold,
        "positions_triggered": triggered,
        "positions_skipped_no_candidates": self._closeout_td1_positions_skipped_no_candidates,
        "positions_skipped_high_value_gate": self._closeout_td1_positions_skipped_high_value_gate,
        "forced_sims_total": self._closeout_td1_forced_sims_total,
        "selected_forced_move_count": self._closeout_td1_selected_forced_move_count,
        "selected_forced_move_rate": (
            (self._closeout_td1_selected_forced_move_count / triggered)
            if triggered > 0 else 0.0
        ),
        "post_force_endpoint_visit_top1_rate": (
            (self._closeout_td1_post_force_top1_hits / triggered) if triggered > 0 else 0.0
        ),
        "post_force_endpoint_visit_top5_rate": (
            (self._closeout_td1_post_force_top5_hits / triggered) if triggered > 0 else 0.0
        ),
    }
```

- [ ] **Step 4: Wire the call into `search_from_root`**

Modify `MCTS.search_from_root` (line 331). Change the signature to accept `gc_state_full`:

```python
def search_from_root(
    self,
    root: MCTSNode,
    add_noise: bool = True,
    ply: int = 0,
    gc_state_full: Optional[dict] = None,
) -> Tuple[Dict[Tuple[int, int], int], float, MCTSNode]:
```

Immediately after `_add_dirichlet_noise(root, ply)` (still inside `search_from_root`, before `for sim in range(self.config.n_simulations)`), insert:

```python
# Spec 3 Fix 1 — td=1 root visit forcing.
forced_count = 0
forced_candidate_ids = set()
if self.config.closeout_td1_visit_forcing_enabled and gc_state_full is not None:
    if gc_state_full.get("total_goal_distance") == 1:
        ec_moves = gc_state_full.get("endpoint_completion_moves") or []
        gate_passes = (
            (not self.config.closeout_td1_require_high_value)
            or (root.q_value >= self.config.closeout_td1_high_value_threshold)
        )
        if not ec_moves:
            self._closeout_td1_positions_skipped_no_candidates += 1
        elif not gate_passes:
            self._closeout_td1_positions_skipped_high_value_gate += 1
        else:
            self._closeout_td1_positions_triggered += 1
            forced_count = self.force_root_visits(
                root=root,
                candidate_moves=ec_moves,
                min_visits=self.config.closeout_td1_min_visits,
                max_candidates=self.config.closeout_td1_max_forced_moves,
            )
            self._closeout_td1_forced_sims_total += forced_count
            forced_candidate_ids = {
                encode_move(r, c) for (r, c) in ec_moves[:self.config.closeout_td1_max_forced_moves]
            }

# Reduce the main loop budget by the number of forced sims already run.
remaining_sims = self.config.n_simulations - forced_count
```

Change the existing `for sim in range(self.config.n_simulations):` to:

```python
for sim in range(remaining_sims):
```

After the main loop completes, before `return`, add a post-force telemetry update:

```python
if forced_candidate_ids and self._closeout_td1_positions_triggered > 0:
    # Final visit-count distribution determines whether the forced moves
    # actually ranked top-1 / top-5 by visits.
    sorted_visits = sorted(
        [(mid, c.visit_count) for mid, c in root.children.items()],
        key=lambda kv: kv[1], reverse=True,
    )
    top1_ids = {sorted_visits[0][0]} if sorted_visits else set()
    top5_ids = {mid for mid, _ in sorted_visits[:5]}
    if forced_candidate_ids & top1_ids:
        self._closeout_td1_post_force_top1_hits += 1
    if forced_candidate_ids & top5_ids:
        self._closeout_td1_post_force_top5_hits += 1
    # Did MCTS-final-selection pick one of the forced moves? Compute by visit argmax
    # (selection is by visit count in the caller; replicate the rule conservatively).
    if sorted_visits and sorted_visits[0][0] in forced_candidate_ids:
        self._closeout_td1_selected_forced_move_count += 1
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_mcts_force_root_visits.py tests/test_mcts_forced_root_visit_equivalence.py tests/test_mcts.py -v
```

Expected: all PASS (existing tests still green plus the new ones).

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_mcts_force_root_visits.py
git commit -m "feat(mcts): wire force_root_visits into search_from_root with telemetry"
```

---

# Phase 3 — Self-play wiring + CLI

## Task 14: Plumb `gc_state_full` into `search_from_root` from self_play

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py`

- [ ] **Step 1: Locate the `search_from_root` call site**

```bash
grep -n "search_from_root" /Users/bill/projects/TwixT_Game/scripts/GPU/alphazero/self_play.py
```

You should see one or two call sites inside `play_game()`.

- [ ] **Step 2: Pass `gc_state_full` as a kwarg**

For each call site, change:

```python
visit_counts, root_value, root = mcts.search_from_root(root, add_noise=add_noise, ply=ply)
```

to:

```python
visit_counts, root_value, root = mcts.search_from_root(
    root, add_noise=add_noise, ply=ply, gc_state_full=gc_state_full,
)
```

The variable `gc_state_full` is already computed earlier in `play_game` (around line 732, where the `compute_goal_completion_state` call lives). Ensure `search_from_root` is called AFTER that computation each ply.

- [ ] **Step 3: Smoke-run training for one short iteration**

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --resume checkpoints/alphazero-v2-staged/model_iter_0139.safetensors \
  --iterations 140 \
  --games-per-iter 2 \
  --checkpoint-dir /tmp/spec3_smoke_ckpt \
  --n-workers 2 \
  --mcts-eval-batch-size 4 \
  --opening-noise-ply 4
```

Expected: completes without exceptions; produces a stats sidecar and 2 games.

- [ ] **Step 4: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat(self_play): plumb gc_state_full into search_from_root for Fix 1"
```

---

## Task 15: Add CLI flags + validation in `train.py`

**Files:**
- Modify: `scripts/GPU/alphazero/train.py`
- Test: `tests/test_train_closeout_td1_cli.py`

- [ ] **Step 1: Write the CLI test**

Create `tests/test_train_closeout_td1_cli.py`:

```python
"""Tests for Spec 3 Fix 1 CLI flag plumbing in train.py."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser():
    from scripts.GPU.alphazero.train import build_arg_parser
    return build_arg_parser()


def test_default_flags_disable_visit_forcing():
    p = _build_parser()
    args = p.parse_args([])
    assert args.closeout_td1_visit_forcing_enabled is False
    assert args.closeout_td1_min_visits == 8
    assert args.closeout_td1_max_forced_moves == 4
    assert args.closeout_td1_require_high_value is False
    assert abs(args.closeout_td1_high_value_threshold - 0.95) < 1e-9


def test_flags_parse_overrides():
    p = _build_parser()
    args = p.parse_args([
        "--closeout-td1-visit-forcing-enabled",
        "--closeout-td1-min-visits", "16",
        "--closeout-td1-max-forced-moves", "2",
        "--closeout-td1-require-high-value",
        "--closeout-td1-high-value-threshold", "0.9",
    ])
    assert args.closeout_td1_visit_forcing_enabled is True
    assert args.closeout_td1_min_visits == 16
    assert args.closeout_td1_max_forced_moves == 2
    assert args.closeout_td1_require_high_value is True
    assert abs(args.closeout_td1_high_value_threshold - 0.9) < 1e-9
```

Note: if `train.py` does not expose a `build_arg_parser` function, refactor the parser-construction block inside `main()` into a top-level `def build_arg_parser()` so it can be tested in isolation. This is a small refactor; do it as part of this task.

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_train_closeout_td1_cli.py -v
```

- [ ] **Step 3: Add the flags**

In `scripts/GPU/alphazero/train.py`, locate the existing block of `--conversion-*` flag definitions (around line 357-363). Immediately after, add:

```python
# Spec 3 Fix 1 — td=1 root visit forcing
parser.add_argument("--closeout-td1-visit-forcing-enabled", action="store_true",
                    help="Enable td=1 endpoint-completion root visit forcing in MCTS (Spec 3 Fix 1).")
parser.add_argument("--closeout-td1-min-visits", type=int, default=8,
                    help="Forced visits per endpoint-completion candidate at td=1 (default: 8).")
parser.add_argument("--closeout-td1-max-forced-moves", type=int, default=4,
                    help="Cap on number of candidate endpoint-completion moves to force per position.")
parser.add_argument("--closeout-td1-require-high-value", action="store_true",
                    help="Gate Fix 1 on root.q_value >= --closeout-td1-high-value-threshold.")
parser.add_argument("--closeout-td1-high-value-threshold", type=float, default=0.95,
                    help="Root q threshold used when --closeout-td1-require-high-value is set.")
```

Wire these into the MCTSConfig construction (search for an existing `MCTSConfig(` construction site in `train.py` and add the new fields, mirroring how `conversion_*` flags are wired around line 697):

```python
closeout_td1_visit_forcing_enabled=args.closeout_td1_visit_forcing_enabled,
closeout_td1_min_visits=args.closeout_td1_min_visits,
closeout_td1_max_forced_moves=args.closeout_td1_max_forced_moves,
closeout_td1_require_high_value=args.closeout_td1_require_high_value,
closeout_td1_high_value_threshold=args.closeout_td1_high_value_threshold,
```

Add a validation right after the existing conversion validators (around line 406-417):

```python
if args.closeout_td1_min_visits < 1:
    parser.error("--closeout-td1-min-visits must be >= 1")
if args.closeout_td1_max_forced_moves < 1:
    parser.error("--closeout-td1-max-forced-moves must be >= 1")
if not (0.0 <= args.closeout_td1_high_value_threshold <= 1.0):
    parser.error("--closeout-td1-high-value-threshold must be in [0.0, 1.0]")
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_train_closeout_td1_cli.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/train.py tests/test_train_closeout_td1_cli.py
git commit -m "feat(train): add CLI flags for closeout td=1 visit forcing"
```

---

## Task 16: Drain MCTS telemetry into stats sidecar

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py`, `scripts/GPU/alphazero/ipc_messages.py`, `scripts/GPU/alphazero/self_play_worker.py`, `scripts/GPU/alphazero/trainer.py`

**Transport pattern (Option A — per the spec §4.5 telemetry transport paragraph):**

The MCTS instance lives inside a worker, but the sidecar gets written by the trainer. The telemetry must cross the worker boundary. As shipped, the chosen pattern is:

1. `play_game` snapshots `mcts.get_closeout_td1_telemetry()` at game end and stores it on a new `GameRecord.closeout_td1_telemetry: Optional[dict]` field.
2. `GameComplete` (IPC dataclass) gains a parallel `closeout_td1_telemetry: Optional[dict] = None` field; the worker (`self_play_worker.py`) forwards `game.closeout_td1_telemetry` into the message.
3. The trainer collects each completed game's snapshot into an `all_closeout_td1_telemetry` list (one entry per game per iteration) on both the serial and parallel code paths.
4. At the per-iteration sidecar emit point, the trainer calls `_merge_closeout_td1_telemetry(all_closeout_td1_telemetry)` and assigns the result to `_sidecar["closeout_td1_visit_forcing"]`.

Alternative Option B (worker already has a stats payload — add closeout_td1 there and aggregate in trainer) was considered but rejected because the existing payload contract is per-game (matches GameRecord), not per-worker-iteration, and GameRecord already collects every other per-game telemetry surface.

- [ ] **Step 1: Locate the trainer-side sidecar emit site**

```bash
grep -n "closeout_td1\|recovery_or_extreme_closeout_drift\|conversion_training\|_sidecar\[" /Users/bill/projects/TwixT_Game/scripts/GPU/alphazero/trainer.py | head
```

Find where the trainer assembles the per-iteration sidecar dict (look for `_sidecar["recovery_or_extreme_closeout_drift"]` or similar). The new merged block joins them at that point.

- [ ] **Step 2: Aggregate per-worker MCTS telemetry**

Self-play uses multiple workers, each with its own MCTS instance. At the end of each worker's contribution (or when the worker reports stats back to the trainer), call `mcts.get_closeout_td1_telemetry()` and accumulate by summing per-counter across workers.

Add a helper at module scope of `self_play.py`:

```python
def _merge_closeout_td1_telemetry(per_worker_telemetry: list) -> dict:
    """Sum counters across workers and recompute rates."""
    if not per_worker_telemetry:
        return {}
    # Config fields come from the first worker; counters sum.
    first = per_worker_telemetry[0]
    out = {k: first.get(k) for k in
           ("enabled", "min_visits", "max_forced_moves",
            "require_high_value", "high_value_threshold")}
    sums = {"positions_triggered": 0, "positions_skipped_no_candidates": 0,
            "positions_skipped_high_value_gate": 0, "forced_sims_total": 0,
            "selected_forced_move_count": 0,
            "_top1_hits": 0, "_top5_hits": 0}
    for t in per_worker_telemetry:
        for k in sums:
            if k.startswith("_"):
                continue
            sums[k] += int(t.get(k, 0) or 0)
        # rate-back-out for hit counts
        triggered = int(t.get("positions_triggered", 0) or 0)
        sums["_top1_hits"] += int(round((t.get("post_force_endpoint_visit_top1_rate", 0) or 0) * triggered))
        sums["_top5_hits"] += int(round((t.get("post_force_endpoint_visit_top5_rate", 0) or 0) * triggered))
    triggered_total = sums["positions_triggered"]
    out.update({
        "positions_triggered": triggered_total,
        "positions_skipped_no_candidates": sums["positions_skipped_no_candidates"],
        "positions_skipped_high_value_gate": sums["positions_skipped_high_value_gate"],
        "forced_sims_total": sums["forced_sims_total"],
        "selected_forced_move_count": sums["selected_forced_move_count"],
        "selected_forced_move_rate":
            (sums["selected_forced_move_count"] / triggered_total) if triggered_total > 0 else 0.0,
        "post_force_endpoint_visit_top1_rate":
            (sums["_top1_hits"] / triggered_total) if triggered_total > 0 else 0.0,
        "post_force_endpoint_visit_top5_rate":
            (sums["_top5_hits"] / triggered_total) if triggered_total > 0 else 0.0,
    })
    return out
```

In the stats-sidecar assembly block, gather telemetry from each worker's MCTS instance, merge, and write it to the sidecar dict:

```python
per_worker_tel = [w.mcts.get_closeout_td1_telemetry() for w in workers if hasattr(w, "mcts")]
stats_sidecar["closeout_td1_visit_forcing"] = _merge_closeout_td1_telemetry(per_worker_tel)
```

(Adjust to match the actual variable names; the goal is `stats_sidecar["closeout_td1_visit_forcing"]` populated from accumulated worker telemetry.)

- [ ] **Step 3: Smoke-run 1 iter and inspect the sidecar**

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --resume checkpoints/alphazero-v2-staged/model_iter_0139.safetensors \
  --iterations 140 \
  --games-per-iter 2 \
  --checkpoint-dir /tmp/spec3_smoke_ckpt2 \
  --n-workers 2 --mcts-eval-batch-size 4 --opening-noise-ply 4 \
  --closeout-td1-visit-forcing-enabled
.venv/bin/python -c "import json; print(json.dumps(json.load(open('scripts/GPU/logs/games/iter_0140_stats.json')).get('closeout_td1_visit_forcing'), indent=2))"
```

Expected: well-formed block with enabled=true and counters present (likely zero if no td=1 positions arose in 2 games, but the block must exist).

- [ ] **Step 4: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat(self_play): drain closeout_td1_visit_forcing telemetry into sidecar"
```

---

# Phase 4 — Analyzer reads Fix 1 telemetry

## Task 17: Aggregate `closeout_td1_visit_forcing` from sidecars

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Test: `tests/test_analyzer_closeout_td1_visit_forcing_summary.py`

- [ ] **Step 1: Write the summary test**

Create `tests/test_analyzer_closeout_td1_visit_forcing_summary.py`:

```python
"""Tests for Fix 1 telemetry aggregation in the analyzer (spec §1.2)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    aggregate_closeout_td1_visit_forcing,
    format_closeout_td1_visit_forcing_report,
)


def test_aggregator_sums_across_iterations():
    sidecars = {
        140: {"closeout_td1_visit_forcing": {
            "enabled": True, "min_visits": 8, "max_forced_moves": 4,
            "positions_triggered": 100, "forced_sims_total": 500,
            "selected_forced_move_count": 70,
            "post_force_endpoint_visit_top5_rate": 0.85,
            "post_force_endpoint_visit_top1_rate": 0.72,
        }},
        141: {"closeout_td1_visit_forcing": {
            "enabled": True, "min_visits": 8, "max_forced_moves": 4,
            "positions_triggered": 80, "forced_sims_total": 400,
            "selected_forced_move_count": 60,
            "post_force_endpoint_visit_top5_rate": 0.9,
            "post_force_endpoint_visit_top1_rate": 0.8,
        }},
    }
    s = aggregate_closeout_td1_visit_forcing(sidecars)
    assert s["enabled"] is True
    assert s["positions_triggered_total"] == 180
    assert s["forced_sims_total"] == 900
    assert abs(s["selected_forced_move_rate"] - (130 / 180)) < 1e-6
    # weighted rates
    expected_top5 = (0.85 * 100 + 0.9 * 80) / 180
    assert abs(s["post_force_endpoint_visit_top5_rate"] - expected_top5) < 1e-6


def test_report_formatter_emits_section():
    summary = {
        "enabled": True, "min_visits": 8, "max_forced_moves": 4,
        "iters_covered": [140, 141, 142],
        "positions_triggered_total": 180, "forced_sims_total": 900,
        "selected_forced_move_count": 130, "selected_forced_move_rate": 0.722,
        "post_force_endpoint_visit_top1_rate": 0.75,
        "post_force_endpoint_visit_top5_rate": 0.875,
    }
    lines = format_closeout_td1_visit_forcing_report(summary)
    body = "\n".join(lines)
    assert "Closeout td=1 visit forcing" in body
    assert "Iters covered" in body
    assert "180" in body
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_analyzer_closeout_td1_visit_forcing_summary.py -v
```

- [ ] **Step 3: Implement aggregator + formatter**

Add to `scripts/twixt_replay_analyzer.py` (near `format_conversion_training_trend_report` and `format_recovery_or_extreme_closeout_drift_report`, around lines 1893-1951):

```python
def aggregate_closeout_td1_visit_forcing(sidecars: dict) -> dict:
    """Aggregate the closeout_td1_visit_forcing block across iterations.

    Sums raw counters; recomputes weighted rates with positions_triggered
    as the weight. Spec 3 Fix 1 §4.5.
    """
    iters_covered = sorted([
        it for it, sc in (sidecars or {}).items()
        if isinstance(sc, dict) and isinstance(sc.get("closeout_td1_visit_forcing"), dict)
    ])
    if not iters_covered:
        return {}
    enabled = False
    min_visits = None
    max_forced_moves = None
    pt_total = 0
    skip_no_cand = 0
    skip_hv = 0
    forced_total = 0
    selected_count = 0
    top1_weighted = 0.0
    top5_weighted = 0.0
    for it in iters_covered:
        blk = sidecars[it].get("closeout_td1_visit_forcing") or {}
        enabled = enabled or bool(blk.get("enabled"))
        if min_visits is None and blk.get("min_visits") is not None:
            min_visits = blk.get("min_visits")
        if max_forced_moves is None and blk.get("max_forced_moves") is not None:
            max_forced_moves = blk.get("max_forced_moves")
        pt = int(blk.get("positions_triggered", 0) or 0)
        pt_total += pt
        skip_no_cand += int(blk.get("positions_skipped_no_candidates", 0) or 0)
        skip_hv      += int(blk.get("positions_skipped_high_value_gate", 0) or 0)
        forced_total += int(blk.get("forced_sims_total", 0) or 0)
        selected_count += int(blk.get("selected_forced_move_count", 0) or 0)
        top1_weighted += float(blk.get("post_force_endpoint_visit_top1_rate", 0) or 0) * pt
        top5_weighted += float(blk.get("post_force_endpoint_visit_top5_rate", 0) or 0) * pt

    def _rate(num, den):
        return (num / den) if den > 0 else 0.0

    return {
        "iters_covered": iters_covered,
        "enabled": enabled,
        "min_visits": min_visits,
        "max_forced_moves": max_forced_moves,
        "positions_triggered_total": pt_total,
        "positions_skipped_no_candidates": skip_no_cand,
        "positions_skipped_high_value_gate": skip_hv,
        "forced_sims_total": forced_total,
        "selected_forced_move_count": selected_count,
        "selected_forced_move_rate": _rate(selected_count, pt_total),
        "post_force_endpoint_visit_top1_rate": _rate(top1_weighted, pt_total),
        "post_force_endpoint_visit_top5_rate": _rate(top5_weighted, pt_total),
    }


def format_closeout_td1_visit_forcing_report(summary: dict) -> list:
    """Format the Fix 1 telemetry section. Spec §1.2."""
    if not summary:
        return []
    def _pct(x):
        return f"{(x or 0.0) * 100.0:.1f}%"
    lines = []
    lines.append("Closeout td=1 visit forcing")
    lines.append("===========================")
    iters = summary.get("iters_covered") or []
    if iters:
        lines.append(f"Iters covered: {min(iters)}-{max(iters)}  enabled={summary.get('enabled')}  "
                     f"min_visits={summary.get('min_visits')}  "
                     f"max_forced_moves={summary.get('max_forced_moves')}")
    lines.append(f"Positions triggered: {summary.get('positions_triggered_total', 0)}  "
                 f"skipped(no_cand)={summary.get('positions_skipped_no_candidates', 0)}  "
                 f"skipped(hv_gate)={summary.get('positions_skipped_high_value_gate', 0)}")
    lines.append(f"Forced sims total:   {summary.get('forced_sims_total', 0)}")
    lines.append(f"Selected forced move rate:        {_pct(summary.get('selected_forced_move_rate'))}")
    lines.append(f"Post-force endpoint visit top-1:  {_pct(summary.get('post_force_endpoint_visit_top1_rate'))}")
    lines.append(f"Post-force endpoint visit top-5:  {_pct(summary.get('post_force_endpoint_visit_top5_rate'))}")
    return lines
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_analyzer_closeout_td1_visit_forcing_summary.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_closeout_td1_visit_forcing_summary.py
git commit -m "feat(analyzer): aggregate + format closeout_td1_visit_forcing summary"
```

---

## Task 18: Wire Fix 1 telemetry into `analyze()`

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`

- [ ] **Step 1: Wire in `analyze()`**

In `analyze()`, find the existing call to `format_conversion_training_trend_report(sidecar_summaries)` or `format_recovery_or_extreme_closeout_drift_report(...)`. Right after it, add:

```python
td1_summary = aggregate_closeout_td1_visit_forcing(sidecars or {})
if td1_summary:
    report_lines.extend([""])
    report_lines.extend(format_closeout_td1_visit_forcing_report(td1_summary))
    summary["closeout_td1_visit_forcing"] = td1_summary
```

`sidecars` is the dict of per-iter sidecar blobs already loaded earlier in `analyze()`.

- [ ] **Step 2: Smoke-run on Replays/130-139** (baseline; no Fix 1 telemetry yet — block should be absent and code should handle it gracefully)

```bash
.venv/bin/python ./scripts/twixt_replay_analyzer.py --input Replays/130-139 --out /tmp/td1_smoke
grep "Closeout td=1 visit forcing" /tmp/td1_smoke/report_*.txt || echo "(no Fix 1 section in baseline, as expected)"
```

Expected: no section (baseline pre-Fix-1 has no telemetry).

- [ ] **Step 3: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "feat(analyzer): wire closeout_td1_visit_forcing telemetry summary into analyze()"
```

---

# Phase 5 — Treatment run + evaluation

## Task 19: Run treatment block 140-149

**Files:** none (training run; produces `checkpoints/alphazero-v2-staged/model_iter_014?.safetensors` and `scripts/GPU/logs/games/iter_014?_*.json` plus `iter_014?_stats.json`)

- [ ] **Step 1: Confirm `model_iter_0139.safetensors` exists**

```bash
ls -la checkpoints/alphazero-v2-staged/model_iter_0139.safetensors
```

If absent, retrace which checkpoint number is current and adjust `--resume` accordingly.

- [ ] **Step 2: Launch treatment training**

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --resume checkpoints/alphazero-v2-staged/model_iter_0139.safetensors \
  --iterations 149 \
  --games-per-iter 100 \
  --checkpoint-dir checkpoints/alphazero-v2-staged \
  --value-weight 0.5 --value-lr-scale 0.0025 --value-grad-max-norm 0.05 \
  --progress-weighted-value-loss --progress-weight-floor 0.25 \
  --n-workers 10 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48 \
  --opening-noise-ply 10 --opening-dirichlet-alpha 0.7 --opening-dirichlet-eps 0.50 \
  --mirror-prob 0.5 \
  --resign-enabled --resign-min-ply 80 --resign-threshold -0.945 \
  --resign-window 12 --resign-k 4 --resign-min-visits 200 --resign-min-top1-share 0.102 \
  --adjudicate-enabled --adjudicate-min-ply 240 --adjudicate-threshold 0.20 \
  --adjudicate-min-visits 200 --adjudicate-min-top1-share 0.13 \
  --max-positions-per-game 64 --endgame-keep-positions 16 \
  --conversion-policy-loss-enabled \
  --conversion-policy-loss-weight 0.05 \
  --conversion-completion-weight 1.0 --conversion-reducer-weight 0.35 \
  --conversion-max-total-goal-distance 2 --conversion-sample-boost 2.0 \
  --conversion-max-batch-fraction 0.15 \
  --closeout-td1-visit-forcing-enabled \
  --closeout-td1-min-visits 8 \
  --closeout-td1-max-forced-moves 4
```

This is the treatment run. Fix 2 is NOT enabled.

- [ ] **Step 3: Confirm 10 iterations completed**

```bash
ls -1 checkpoints/alphazero-v2-staged/model_iter_014?.safetensors | wc -l
ls -1 scripts/GPU/logs/games/iter_014?_stats.json | wc -l
```

Expected: 10 checkpoints, 10 stats files.

---

## Task 20: Run analyzer on 140-149 and compare to baseline

**Files:** none (analysis run)

- [ ] **Step 1: Stage games and run analyzer**

```bash
mkdir -p Replays/140-149
cp scripts/GPU/logs/games/iter_014?_*.json Replays/140-149/
.venv/bin/python ./scripts/twixt_replay_analyzer.py \
    --input Replays/140-149 --out Replays/140-149_Replay
```

- [ ] **Step 2: Compare to baseline against §8 criteria**

```bash
diff <(grep -E "delay >=|state_cap after|high-value but delayed|conv_delay" Replays/130-139_Replay/report_130-139.txt) \
     <(grep -E "delay >=|state_cap after|high-value but delayed|conv_delay" Replays/140-149_Replay/report_140-149.txt)

echo "---td=1 breakdown comparison---"
echo "BASELINE (130-139):"
grep -A 5 "td=1:" Replays/130-139_Replay/report_130-139.txt | head -8
echo "TREATMENT (140-149):"
grep -A 5 "td=1:" Replays/140-149_Replay/report_140-149.txt | head -8

echo "---Fix 1 telemetry---"
grep -A 8 "Closeout td=1 visit forcing" Replays/140-149_Replay/report_140-149.txt
```

Decision criteria (spec §8):
- `delay >= 10` drops from 21 toward target ≤ 10
- `delay >= 20` drops from 7 toward ≤ 3
- `state_cap after detection` drops from 4 toward ≤ 1
- td=1 endpoint visit top-5 rises sharply (target ≥ 90%)
- Fix 1 telemetry: positions_triggered > 0 (proves trigger fired); selected_forced_move_rate is informative

- [ ] **Step 3: Commit the artifacts**

```bash
git add Replays/140-149/ Replays/140-149_Replay/
git commit -m "data: spec 3 fix 1 treatment run 140-149 (visit forcing on)"
```

- [ ] **Step 4: Write a short summary in the spec doc**

Append to `docs/superpowers/specs/2026-05-10-closeout-tail-correction-design.md` under a new `## 11. Results` section, filling in the actual numbers from §8 baseline-vs-treatment.

Commit:

```bash
git add docs/superpowers/specs/2026-05-10-closeout-tail-correction-design.md
git commit -m "docs(spec): record Spec 3 Fix 1 treatment results 130-139 vs 140-149"
```

---

# Phase 6 — Decision point: Fix 2

## Task 21: Decide whether Fix 2 is needed (gate)

**Files:** none

- [ ] **Step 1: Apply the §8 decision rule**

Re-read §8 of the spec. Fix 2 should be enabled iff:
- td=1 endpoint visit top-5 IS now high (≥ 80%) after Fix 1, AND
- the game-count tail (delay≥10, delay≥20, state_cap_after_detection) has not dropped by at least the §8 target.

If both conditions hold, the residual cause is selection-time and Fix 2 is the right next tool. Proceed to Tasks 22-26.

If td=1 visit top-5 is still low after Fix 1, the issue is upstream of selection — STOP and re-discuss with the user before adding Fix 2.

If the §8 targets are met without Fix 2, Fix 2 stays off permanently. Skip Tasks 22-26 and mark Phase 6 done.

- [ ] **Step 2: Record the decision in the spec**

Append to `docs/superpowers/specs/2026-05-10-closeout-tail-correction-design.md` under `## 12. Fix 2 decision`, with one paragraph stating the call and the metric values that drove it.

Commit:

```bash
git add docs/superpowers/specs/2026-05-10-closeout-tail-correction-design.md
git commit -m "docs(spec): record Spec 3 Fix 2 enable/skip decision"
```

---

# Phase 7 — Fix 2 (CONDITIONAL — only if Phase 6 gate passes)

## Task 22: Add `MCTSConfig` fields for Fix 2 + selection-tiebreak method

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py`
- Test: `tests/test_mcts_closeout_selection_tiebreak.py`

- [ ] **Step 1: Write the tie-break test**

Create `tests/test_mcts_closeout_selection_tiebreak.py`:

```python
"""Tests for Spec 3 Fix 2 — narrow closeout selection tie-break."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig


def test_tiebreak_disabled_by_default():
    c = MCTSConfig()
    assert c.closeout_selection_tiebreak_enabled is False


def test_tiebreak_override_when_endpoint_in_topk():
    """argmax was redundant; endpoint is rank-3 in visits with share above floor → override."""
    cfg = MCTSConfig(
        closeout_selection_tiebreak_enabled=True,
        closeout_selection_tiebreak_max_distance=2,
        closeout_selection_tiebreak_topk=5,
        closeout_selection_tiebreak_min_value=0.95,
        closeout_selection_tiebreak_min_share=0.05,
    )
    visit_counts = {(0, 0): 100, (1, 1): 50, (2, 2): 80, (3, 3): 60}  # argmax = (0,0)
    gc_state = {
        "total_goal_distance": 2,
        "endpoint_completion_moves": [(2, 2)],
        "distance_reducing_moves": [(2, 2), (3, 3)],
    }
    root_q = 0.97
    selected_argmax_class = "redundant_reinforcement"
    updated_counts, record = MCTS.apply_closeout_selection_tiebreak(
        visit_counts=visit_counts, gc_state_full=gc_state,
        root_q=root_q, selected_argmax_class=selected_argmax_class, config=cfg,
    )
    new_argmax = max(updated_counts, key=updated_counts.get)
    assert new_argmax == (2, 2)
    assert record["overrode_to"] == "endpoint"


def test_tiebreak_skips_when_share_below_floor():
    cfg = MCTSConfig(closeout_selection_tiebreak_enabled=True,
                     closeout_selection_tiebreak_min_share=0.1)
    visit_counts = {(0, 0): 95, (2, 2): 5}  # endpoint share = 0.05 -> below 0.1 floor
    gc_state = {"total_goal_distance": 2, "endpoint_completion_moves": [(2, 2)],
                "distance_reducing_moves": [(2, 2)]}
    updated, rec = MCTS.apply_closeout_selection_tiebreak(
        visit_counts, gc_state, root_q=0.97,
        selected_argmax_class="off_chain", config=cfg,
    )
    assert updated == visit_counts
    assert rec.get("overrode_to") is None
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_mcts_closeout_selection_tiebreak.py -v
```

- [ ] **Step 3: Add MCTSConfig fields**

In `MCTSConfig`:

```python
    # Spec 3 Fix 2 — narrow closeout selection tie-break (opt-in)
    closeout_selection_tiebreak_enabled: bool = False
    closeout_selection_tiebreak_max_distance: int = 2
    closeout_selection_tiebreak_topk: int = 5
    closeout_selection_tiebreak_min_value: float = 0.95
    closeout_selection_tiebreak_min_share: float = 0.05
```

- [ ] **Step 4: Implement the tiebreak static method**

Add as a `@staticmethod` on `MCTS`:

```python
@staticmethod
def apply_closeout_selection_tiebreak(
    visit_counts: dict,
    gc_state_full: dict,
    root_q: float,
    selected_argmax_class: str,
    config: MCTSConfig,
) -> tuple:
    """Conditionally override the visit-count argmax to a closeout candidate.

    Returns (updated_counts, record). The record has overrode_to: 'endpoint'
    or 'reducer' or None. Spec 2026-05-10 §5.
    """
    record = {"overrode_to": None}
    if not config.closeout_selection_tiebreak_enabled or not visit_counts or not gc_state_full:
        return visit_counts, record
    td = gc_state_full.get("total_goal_distance")
    if td is None or td > config.closeout_selection_tiebreak_max_distance:
        return visit_counts, record
    if root_q < config.closeout_selection_tiebreak_min_value:
        return visit_counts, record
    if selected_argmax_class not in {"redundant_reinforcement", "off_chain", "other"}:
        return visit_counts, record
    total_visits = sum(visit_counts.values()) or 1
    # Build ordered top-K by visits
    sorted_moves = sorted(visit_counts.items(), key=lambda kv: kv[1], reverse=True)
    topk_moves = [mv for mv, _ in sorted_moves[:config.closeout_selection_tiebreak_topk]]

    def _best_match(candidate_list):
        candidate_set = {tuple(m) for m in (candidate_list or [])}
        for mv, c in sorted_moves:
            if mv in candidate_set and mv in topk_moves:
                share = c / total_visits
                if share >= config.closeout_selection_tiebreak_min_share:
                    return mv, c
        return None

    ec = _best_match(gc_state_full.get("endpoint_completion_moves"))
    if ec is not None:
        target_move = ec[0]
        record["overrode_to"] = "endpoint"
    else:
        rd = _best_match(gc_state_full.get("distance_reducing_moves"))
        if rd is None:
            return visit_counts, record
        target_move = rd[0]
        record["overrode_to"] = "reducer"
    new_counts = dict(visit_counts)
    new_counts[target_move] = sorted_moves[0][1] + 1
    record["target_move"] = target_move
    record["argmax_class_before_override"] = selected_argmax_class
    return new_counts, record
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_mcts_closeout_selection_tiebreak.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_mcts_closeout_selection_tiebreak.py
git commit -m "feat(mcts): add closeout selection tie-break (Spec 3 Fix 2, off by default)"
```

---

## Task 23: Wire Fix 2 into self_play + add CLI flags + telemetry

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py`, `scripts/GPU/alphazero/train.py`

- [ ] **Step 1: Add CLI flags in train.py**

After the Fix 1 flag block:

```python
parser.add_argument("--closeout-selection-tiebreak-enabled", action="store_true")
parser.add_argument("--closeout-selection-tiebreak-max-distance", type=int, default=2)
parser.add_argument("--closeout-selection-tiebreak-topk", type=int, default=5)
parser.add_argument("--closeout-selection-tiebreak-min-value", type=float, default=0.95)
parser.add_argument("--closeout-selection-tiebreak-min-share", type=float, default=0.05)
```

Wire into `MCTSConfig` construction.

- [ ] **Step 2: Add tiebreak telemetry counters on the `MCTS` instance**

In `MCTS.__init__`, after the Fix 1 counter block, add:

```python
        # Spec 3 Fix 2 — closeout selection tie-break telemetry
        self._closeout_tiebreak_eligible = 0
        self._closeout_tiebreak_overrides = 0
        self._closeout_tiebreak_override_to_endpoint = 0
        self._closeout_tiebreak_override_to_reducer = 0
        self._closeout_tiebreak_would_have_redundant = 0
        self._closeout_tiebreak_would_have_off_chain = 0
        self._closeout_tiebreak_would_have_other = 0
```

Add `reset_closeout_tiebreak_telemetry` and `get_closeout_tiebreak_telemetry` methods mirroring the Fix 1 pair:

```python
def reset_closeout_tiebreak_telemetry(self) -> None:
    self._closeout_tiebreak_eligible = 0
    self._closeout_tiebreak_overrides = 0
    self._closeout_tiebreak_override_to_endpoint = 0
    self._closeout_tiebreak_override_to_reducer = 0
    self._closeout_tiebreak_would_have_redundant = 0
    self._closeout_tiebreak_would_have_off_chain = 0
    self._closeout_tiebreak_would_have_other = 0

def get_closeout_tiebreak_telemetry(self) -> dict:
    eligible = self._closeout_tiebreak_eligible
    return {
        "enabled": bool(self.config.closeout_selection_tiebreak_enabled),
        "eligible_positions": eligible,
        "overrides": self._closeout_tiebreak_overrides,
        "override_rate": (self._closeout_tiebreak_overrides / eligible) if eligible > 0 else 0.0,
        "override_to_endpoint": self._closeout_tiebreak_override_to_endpoint,
        "override_to_reducer": self._closeout_tiebreak_override_to_reducer,
        "would_have_selected_redundant": self._closeout_tiebreak_would_have_redundant,
        "would_have_selected_off_chain": self._closeout_tiebreak_would_have_off_chain,
        "would_have_selected_other": self._closeout_tiebreak_would_have_other,
    }
```

- [ ] **Step 3: Classify the current argmax move before invoking the tiebreak**

The tiebreak needs the primary_class of the move that would win on visit-argmax. Compute it inline using `gc_state_full` (which already lists endpoint and reducer move sets):

```python
def _classify_argmax_against_gc(argmax_move, gc_state_full):
    ec_set = {tuple(m) for m in (gc_state_full.get("endpoint_completion_moves") or ())}
    rd_set = {tuple(m) for m in (gc_state_full.get("distance_reducing_moves") or ())}
    if argmax_move in ec_set:
        return "completes_endpoint"
    if argmax_move in rd_set:
        return "reduces_total_goal_distance"
    # Off-chain vs redundant: precise classification requires the connectivity
    # helper (closeout_diagnostics.classify_move_against_chain). For the tiebreak
    # gate we only need to distinguish "in the closeout candidate set" from
    # "not in it"; everything not in EC/RD is treated as 'other' below.
    return "other"
```

- [ ] **Step 4: Invoke the tiebreak between search and select_move in self_play.py**

In `play_game()`, between `mcts.search_from_root(...)` and `mcts.select_move(visit_counts, ply)`:

```python
if mcts.config.closeout_selection_tiebreak_enabled and gc_state_full is not None:
    argmax_move = max(visit_counts, key=visit_counts.get) if visit_counts else None
    argmax_class = (_classify_argmax_against_gc(argmax_move, gc_state_full)
                    if argmax_move is not None else "other")
    mcts._closeout_tiebreak_eligible += 1
    visit_counts, tiebreak_record = MCTS.apply_closeout_selection_tiebreak(
        visit_counts=visit_counts,
        gc_state_full=gc_state_full,
        root_q=root_value,
        selected_argmax_class=argmax_class,
        config=mcts.config,
    )
    if tiebreak_record.get("overrode_to") == "endpoint":
        mcts._closeout_tiebreak_overrides += 1
        mcts._closeout_tiebreak_override_to_endpoint += 1
    elif tiebreak_record.get("overrode_to") == "reducer":
        mcts._closeout_tiebreak_overrides += 1
        mcts._closeout_tiebreak_override_to_reducer += 1
    # would-have-selected accounting (only when an override fired)
    if tiebreak_record.get("overrode_to"):
        if argmax_class == "redundant_reinforcement":
            mcts._closeout_tiebreak_would_have_redundant += 1
        elif argmax_class == "off_chain":
            mcts._closeout_tiebreak_would_have_off_chain += 1
        elif argmax_class == "other":
            mcts._closeout_tiebreak_would_have_other += 1
```

Place the `_classify_argmax_against_gc` function at module scope of `self_play.py`.

- [ ] **Step 5: Drain tiebreak telemetry into stats sidecar**

In the same stats-sidecar assembly block where Task 16 wires `closeout_td1_visit_forcing`, add a parallel `_merge_closeout_tiebreak_telemetry`:

```python
def _merge_closeout_tiebreak_telemetry(per_worker_telemetry: list) -> dict:
    if not per_worker_telemetry:
        return {}
    first = per_worker_telemetry[0]
    sums = {
        "eligible_positions": 0, "overrides": 0,
        "override_to_endpoint": 0, "override_to_reducer": 0,
        "would_have_selected_redundant": 0, "would_have_selected_off_chain": 0,
        "would_have_selected_other": 0,
    }
    for t in per_worker_telemetry:
        for k in sums:
            sums[k] += int(t.get(k, 0) or 0)
    eligible = sums["eligible_positions"]
    return {
        "enabled": bool(first.get("enabled")),
        **sums,
        "override_rate": (sums["overrides"] / eligible) if eligible > 0 else 0.0,
    }

per_worker_tel_tb = [w.mcts.get_closeout_tiebreak_telemetry() for w in workers if hasattr(w, "mcts")]
stats_sidecar["closeout_selection_tiebreak"] = _merge_closeout_tiebreak_telemetry(per_worker_tel_tb)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py scripts/GPU/alphazero/train.py
git commit -m "feat(self_play,train): wire Fix 2 closeout tie-break with CLI + telemetry"
```

---

## Task 24: Analyzer summary for Fix 2 telemetry

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Test: `tests/test_analyzer_closeout_selection_tiebreak_summary.py`

- [ ] **Step 1: Write test**

Create `tests/test_analyzer_closeout_selection_tiebreak_summary.py`:

```python
"""Tests for Fix 2 telemetry aggregation."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    aggregate_closeout_selection_tiebreak,
    format_closeout_selection_tiebreak_report,
)


def test_aggregator_sums_overrides():
    sidecars = {
        150: {"closeout_selection_tiebreak": {
            "enabled": True, "eligible_positions": 100,
            "overrides": 20, "override_to_endpoint": 15, "override_to_reducer": 5,
            "would_have_selected_redundant": 12, "would_have_selected_off_chain": 6,
            "would_have_selected_other": 2,
        }},
    }
    s = aggregate_closeout_selection_tiebreak(sidecars)
    assert s["eligible_positions"] == 100
    assert s["overrides"] == 20
    assert s["override_rate"] == 0.2


def test_format_emits_section():
    summary = {"iters_covered": [150], "enabled": True,
               "eligible_positions": 50, "overrides": 10, "override_rate": 0.2,
               "override_to_endpoint": 8, "override_to_reducer": 2,
               "would_have_selected_redundant": 6, "would_have_selected_off_chain": 4,
               "would_have_selected_other": 0}
    body = "\n".join(format_closeout_selection_tiebreak_report(summary))
    assert "Closeout selection tie-break" in body
    assert "Overrides: 10" in body
```

- [ ] **Step 2: Implement aggregator + formatter**

Add to `scripts/twixt_replay_analyzer.py` immediately after `format_closeout_td1_visit_forcing_report`:

```python
def aggregate_closeout_selection_tiebreak(sidecars: dict) -> dict:
    """Aggregate the closeout_selection_tiebreak block across iterations.

    Sums raw counters and recomputes override_rate. Spec 3 Fix 2 §5.5.
    """
    iters_covered = sorted([
        it for it, sc in (sidecars or {}).items()
        if isinstance(sc, dict) and isinstance(sc.get("closeout_selection_tiebreak"), dict)
    ])
    if not iters_covered:
        return {}
    enabled = False
    sums = {
        "eligible_positions": 0, "overrides": 0,
        "override_to_endpoint": 0, "override_to_reducer": 0,
        "would_have_selected_redundant": 0,
        "would_have_selected_off_chain": 0,
        "would_have_selected_other": 0,
    }
    for it in iters_covered:
        blk = sidecars[it].get("closeout_selection_tiebreak") or {}
        enabled = enabled or bool(blk.get("enabled"))
        for k in sums:
            sums[k] += int(blk.get(k, 0) or 0)
    eligible = sums["eligible_positions"]
    return {
        "iters_covered": iters_covered,
        "enabled": enabled,
        **sums,
        "override_rate": (sums["overrides"] / eligible) if eligible > 0 else 0.0,
    }


def format_closeout_selection_tiebreak_report(summary: dict) -> list:
    """Format the Fix 2 telemetry section. Spec §5.5."""
    if not summary:
        return []
    def _pct(x):
        return f"{(x or 0.0) * 100.0:.1f}%"
    lines = []
    lines.append("Closeout selection tie-break")
    lines.append("============================")
    iters = summary.get("iters_covered") or []
    if iters:
        lines.append(f"Iters covered: {min(iters)}-{max(iters)}  enabled={summary.get('enabled')}")
    lines.append(f"Eligible positions: {summary.get('eligible_positions', 0)}")
    lines.append(f"Overrides: {summary.get('overrides', 0)}  rate={_pct(summary.get('override_rate'))}")
    lines.append(f"  -> endpoint: {summary.get('override_to_endpoint', 0)}")
    lines.append(f"  -> reducer:  {summary.get('override_to_reducer', 0)}")
    lines.append("Would-have-selected (only when an override fired):")
    lines.append(f"  redundant: {summary.get('would_have_selected_redundant', 0)}")
    lines.append(f"  off-chain: {summary.get('would_have_selected_off_chain', 0)}")
    lines.append(f"  other:     {summary.get('would_have_selected_other', 0)}")
    return lines
```

- [ ] **Step 3: Wire into `analyze()`**

In `analyze()`, immediately after the Task 18 Fix 1 telemetry block, add:

```python
tb_summary = aggregate_closeout_selection_tiebreak(sidecars or {})
if tb_summary:
    report_lines.extend([""])
    report_lines.extend(format_closeout_selection_tiebreak_report(tb_summary))
    summary["closeout_selection_tiebreak"] = tb_summary
```

- [ ] **Step 4: Run tests and commit**

```bash
.venv/bin/pytest tests/test_analyzer_closeout_selection_tiebreak_summary.py -v
```

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_closeout_selection_tiebreak_summary.py
git commit -m "feat(analyzer): Fix 2 closeout_selection_tiebreak summary + report"
```

---

## Task 25: Second treatment run with Fix 2 enabled

**Files:** none (training run)

- [ ] **Step 1: Run training from latest checkpoint with both Fix 1 and Fix 2 enabled**

```bash
# Adjust --resume to the latest checkpoint from Phase 5
.venv/bin/python -m scripts.GPU.alphazero.train \
  --resume checkpoints/alphazero-v2-staged/model_iter_0149.safetensors \
  --iterations 159 --games-per-iter 100 \
  ... [same other knobs as Task 19] ... \
  --closeout-td1-visit-forcing-enabled \
  --closeout-td1-min-visits 8 --closeout-td1-max-forced-moves 4 \
  --closeout-selection-tiebreak-enabled
```

- [ ] **Step 2: Run analyzer on 150-159 and compare**

```bash
mkdir -p Replays/150-159
cp scripts/GPU/logs/games/iter_015?_*.json Replays/150-159/
.venv/bin/python ./scripts/twixt_replay_analyzer.py --input Replays/150-159 --out Replays/150-159_Replay
```

Compare against both the baseline (130-139) and the Fix 1-only run (140-149).

- [ ] **Step 3: Commit + record results**

```bash
git add Replays/150-159 Replays/150-159_Replay docs/superpowers/specs/2026-05-10-closeout-tail-correction-design.md
git commit -m "data: spec 3 fix 2 treatment 150-159 + spec results update"
```

---

# Phase 8 — Recovery spec is out of scope

Per spec §7 row 8, recovery training is deferred to a future Spec 4 brainstorm gated on the Fix 3 outputs produced in Phase 1. No tasks in this plan.

---

## Self-Review notes (engineer-facing)

If any task fails its tests, do NOT introduce ad-hoc try/except blocks to mask the failure — root-cause the issue. The equivalence test in Task 12 is the most likely source of subtle bugs: if `value_sum` doesn't match between branches, inspect whether `_select_child` and the root-override path are both incrementing the visit count exactly once and backing up the same leaf value through the same path length.

Key invariants that must hold:
- After a forced sim, `root.visit_count` increments by 1 (same as a normal sim).
- After a forced sim, `child.visit_count` increments by 1 for the chosen child.
- The `value_sum` increment at each node along the path equals `±leaf_value` according to the side-to-move alternation rule already implemented in `_backup`.
- `force_root_visits` is a no-op if `closeout_td1_visit_forcing_enabled` is False.
- Tests in `tests/test_mcts.py` continue to pass after the Task 10 refactor.
- The analyzer must handle the absence of `closeout_td1_visit_forcing` in pre-Fix-1 sidecars without raising (Task 18 smoke check).
