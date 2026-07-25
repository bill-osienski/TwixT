"""v17 Task 1 steps 3-7 -- the pre-change provenance record must stay true.

Guards `logs/eval/fpu_v17_baseline_policy_mass/prechange_baseline.json`, which
freezes the A/B/C/D acceptance baselines and source identities that design §9
makes a Stage-4 validity gate. Every number here is recomputed from the
canonical CSVs rather than copied, so silent drift in either the artifacts or
the record fails the suite.

Also mechanically enforces the freeze: the design SHA-1 recorded at Task 0 must
still be the design's SHA-1. Editing an APPROVED - FROZEN protocol breaks this.
"""
import csv
import hashlib
import json
import os
import statistics

import pytest

RECORD_PATH = "logs/eval/fpu_v17_baseline_policy_mass/prechange_baseline.json"
FROZEN_DESIGN_SHA1 = "944f358c0e3ef66503d2cbb56e31dabd145bafc2"
DESIGN = ("docs/superpowers/specs/"
          "2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md")

# Aggregates as written in the frozen design §9. Compared at the design's own
# 1e-6 reproduction tolerance, which absorbs the documented C/D one-ULP gap.
DESIGN_S9 = {
    "A": (30, +0.25702582687976244, 15, 13),
    "B": (18, -0.24424638776811966, 1, 0),
    "C": (30, +0.09857376916756039, 10, 4),
    "D": (30, -0.18752245797826617, 4, 0),
}
TOL = 1e-6
OVER, SEVERE = 0.25, 0.50


def _record():
    with open(RECORD_PATH) as f:
        return json.load(f)


def _sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


# The four canonical probe CSVs are historical run outputs under the gitignored
# `logs/` tree and are NOT tracked, so the checks that reopen them skip when
# absent -- same pattern as
# `tests/test_goal_line_trigger_probe_cases.py::CANON_*`. The frozen record
# itself, the goldens, the design, and every source module ARE tracked, so the
# checks above never skip.
def _probe_csvs_present():
    if not os.path.exists(RECORD_PATH):
        return False
    return all(os.path.exists(g["canonical_source"])
               for g in _record()["abcd_frozen_baseline"].values())


requires_probe_csvs = pytest.mark.skipif(
    not _probe_csvs_present(),
    reason="canonical A/B/C/D probe CSVs absent (untracked logs/ artifacts)")


def test_frozen_design_has_not_been_edited():
    assert _sha1(DESIGN) == FROZEN_DESIGN_SHA1
    assert _record()["frozen_documents"][DESIGN] == FROZEN_DESIGN_SHA1


def test_recorded_source_sha1s_still_match_the_files():
    rec = _record()["source_sha1s"]
    for group in ("production_result_determining_set", "v17_pure_dependencies",
                  "identity_basis"):
        for path, expected in rec[group].items():
            assert _sha1(path) == expected, path


@requires_probe_csvs
def test_abcd_baselines_recompute_from_the_canonical_sources():
    for gate, frozen in _record()["abcd_frozen_baseline"].items():
        path = frozen["canonical_source"]
        assert _sha1(path) == frozen["source_sha1"], gate
        rows = [r for r in csv.DictReader(open(path))
                if r["checkpoint"] == frozen["checkpoint_filter"]]
        vals = [float(r["probe_black_root_value"]) for r in rows]
        assert len(rows) == frozen["n"], gate
        assert repr(statistics.fmean(vals)) == frozen["mean_repr"], gate
        assert sum(v >= OVER for v in vals) == frozen["over"], gate
        assert sum(v >= SEVERE for v in vals) == frozen["severe"], gate
        assert len(frozen["cases"]) == frozen["n"], gate


def test_frozen_baselines_agree_with_design_section_9():
    for gate, (n, mean, over, severe) in DESIGN_S9.items():
        frozen = _record()["abcd_frozen_baseline"][gate]
        assert frozen["n"] == n, gate
        assert frozen["over"] == over, gate
        assert frozen["severe"] == severe, gate
        assert abs(float(frozen["mean_repr"]) - mean) < TOL, gate


@requires_probe_csvs
def test_csv_over_severe_flags_agree_with_the_threshold_convention():
    """Design §9 states the over/severe convention for A and D; this proves the
    CSVs' own recorded flags agree with it for all four gates, so the frozen
    counts do not depend on which of the two sources you read."""
    for gate, frozen in _record()["abcd_frozen_baseline"].items():
        rows = [r for r in csv.DictReader(open(frozen["canonical_source"]))
                if r["checkpoint"] == frozen["checkpoint_filter"]]
        truthy = ("true", "1")
        recorded_over = sum(r["black_overvalue"].strip().lower() in truthy for r in rows)
        recorded_sev = sum(r["severe_black_overvalue"].strip().lower() in truthy
                           for r in rows)
        assert (recorded_over, recorded_sev) == (frozen["over"], frozen["severe"]), gate


def test_golden_sha1_recorded_and_current():
    g = _record()["goldens"]
    assert _sha1(g["path"]) == g["sha1"]
    assert g["batching_triple"] == [14, 48, 8]


@requires_probe_csvs
def test_selected_move_gap_is_recorded_not_silently_dropped():
    """Design §9 asks Stage 4 to match 'exact selected moves', but no canonical
    CSV carries one. The gap must stay visible in the record until resolved."""
    rec = _record()
    assert "selected_moves" in rec["known_gap"]
    move_identity = ("selected_move", "chosen_move", "best_move", "top_move",
                     "top1_move", "move_id", "pv_move", "principal_variation")
    for frozen in rec["abcd_frozen_baseline"].values():
        cols = set(next(csv.DictReader(open(frozen["canonical_source"]))))
        assert not any(tok in c for c in cols for tok in move_identity), sorted(cols)
        # `side_to_move` is the only 'move'-named column and records whose turn
        # it is, not which move the search picked.
        assert "side_to_move" in cols
