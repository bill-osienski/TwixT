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


# Modules the FROZEN PLAN authorizes implementation to edit. Everything else in
# the recorded snapshot must stay byte-identical, which is what turns this
# record into a guard against out-of-scope source edits.
AUTHORIZED_TO_CHANGE = {
    "scripts/GPU/alphazero/mcts.py",                     # Tasks 2, 3
    "scripts/GPU/alphazero/diagnose_fpu_policy_mass.py",  # Task 5 (parameterize only)
    "scripts/GPU/alphazero/eval_runner.py",              # Task 6
    "scripts/GPU/alphazero/eval_checkpoint_match.py",    # Task 6
}


# What actually determines the golden's CONTENT: the fixture that builds the
# signatures, the search harness under it, and the golden file itself. The
# recorded `identity_basis` also lists the golden's CONSUMER test module, which
# is not a determinant -- later tasks legitimately append assertions to it
# (Task 3 added the search-level exact-zero proofs) without moving the basis.
#
# Not listed, and covered behaviourally instead:
# `tests/test_fpu_policy_mass_rule.py` supplies the pinned synthetic tree the
# sweep walks. If that tree changes, `synthetic_selection_trace()` changes and
# `test_fpu_v17_prechange_golden.py` fails on re-derivation, which is a
# stronger check than a hash pin.
GOLDEN_PRODUCERS = (
    "tests/fpu_search_fixture.py",
    "tests/fpu_v17_prechange_fixture.py",
    "tests/golden/fpu_v17_prechange_search.json",
)


def test_golden_producers_are_unchanged():
    """A change here means the identity basis moved and the golden must be
    re-derived deliberately, not silently."""
    basis = _record()["source_sha1s"]["identity_basis"]
    for path in GOLDEN_PRODUCERS:
        assert _sha1(path) == basis[path], path


def test_no_out_of_scope_source_module_was_edited():
    """The record is a PRE-CHANGE snapshot, so modules the plan authorizes
    (mcts.py at Tasks 2-3, etc.) legitimately drift from it. Every other
    recorded module must still match exactly."""
    rec = _record()["source_sha1s"]
    drifted = set()
    for group in ("production_result_determining_set", "v17_pure_dependencies"):
        for path, expected in rec[group].items():
            if _sha1(path) != expected:
                drifted.add(path)
    assert drifted <= AUTHORIZED_TO_CHANGE, sorted(drifted - AUTHORIZED_TO_CHANGE)


def test_snapshot_was_genuinely_taken_before_any_mcts_edit():
    """Independent of later edits: the recorded mcts.py hash must be the one the
    authenticated v16 production run used, which is what proves Task 1's
    'no MCTS source edit before goldens exist' gate held."""
    rec = _record()["source_sha1s"]["production_result_determining_set"]
    assert rec["scripts/GPU/alphazero/mcts.py"] == \
        "11fa989d7cd521fab69c3801898ee7282e00fc10"


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
def test_canonical_csvs_still_carry_no_selected_move():
    """The reason the supplemental capture exists. If a canonical CSV ever grows
    a selected-move column, this fails and the supplement should be revisited."""
    move_identity = ("selected_move", "chosen_move", "best_move", "top_move",
                     "top1_move", "move_id", "pv_move", "principal_variation")
    for frozen in _record()["abcd_frozen_baseline"].values():
        cols = set(next(csv.DictReader(open(frozen["canonical_source"]))))
        assert not any(tok in c for c in cols for tok in move_identity), sorted(cols)
        # `side_to_move` is the only 'move'-named column and records whose turn
        # it is, not which move the search picked.
        assert "side_to_move" in cols


# ---------------------------------------------------------------------------
# Supplemental shipped-baseline selected moves (design §9 "with exact selected
# moves"). Captured at Task 1 from the same harness, checkpoint, batching
# triple and per-case seeds that produced the frozen CSVs.
# ---------------------------------------------------------------------------

def _selected_moves():
    rec = _record()["selected_moves"]
    with open(rec["artifact"]) as f:
        return rec, json.load(f)


def test_selected_move_artifact_is_authenticated_and_unmodified():
    rec, art = _selected_moves()
    assert rec["status"].startswith("RESOLVED")
    assert _sha1(rec["artifact"]) == rec["artifact_sha1"]
    assert _sha1(rec["capture_tool"]) == rec["capture_tool_sha1"]
    assert art["authentication"]["mismatches"] == 0
    assert art["authentication"]["cases_checked"] == 108
    assert art["mcts"]["batching_triple"] == [14, 48, 8]
    assert art["mcts"]["n_simulations"] == 400
    assert art["mcts"]["add_noise"] is False
    # the checkpoint the whole v17 experiment is pinned to
    assert art["checkpoint_sha1"] == "209cf2d4fd24a48553d259dd71b4954867b9473e"


def test_selected_moves_cover_every_frozen_case_exactly_once():
    _rec, art = _selected_moves()
    frozen = _record()["abcd_frozen_baseline"]
    assert set(art["gates"]) == set(frozen)
    for gate, g in art["gates"].items():
        assert g["n"] == frozen[gate]["n"], gate
        ids = [c["case_id"] for c in g["cases"]]
        assert len(set(ids)) == len(ids), gate
        assert set(ids) == {c["case_id"] for c in frozen[gate]["cases"]}, gate
        for case in g["cases"]:
            move = case["selected_move"]
            assert len(move) == 2 and all(isinstance(x, int) for x in move), case
            assert case["selected_move_visits"] > 0, case


def test_selected_move_capture_reproduced_the_frozen_values_exactly():
    """The authentication that makes these moves usable as a §9 baseline: the
    re-run reproduced every frozen probe_black_root_value bit-for-bit, so the
    moves come from the same search regime as the frozen values."""
    _rec, art = _selected_moves()
    frozen = _record()["abcd_frozen_baseline"]
    for gate, g in art["gates"].items():
        assert float(g["max_abs_delta_vs_frozen_repr"]) == 0.0, gate
        by_id = {c["case_id"]: c for c in frozen[gate]["cases"]}
        for case in g["cases"]:
            assert (case["recomputed_black_value_repr"]
                    == case["frozen_black_value_repr"]
                    == by_id[case["case_id"]]["probe_black_root_value_repr"]), case


@requires_probe_csvs
def test_selected_move_top_share_agrees_with_the_canonical_csv():
    """Independent cross-check on a column the capture did not authenticate
    against: the re-run's top share must match the CSV's probe_top1_share."""
    _rec, art = _selected_moves()
    for gate, g in art["gates"].items():
        rows = {r["case_id"]: r for r in
                csv.DictReader(open(g["cases_source"]))
                if r["checkpoint"] == "0001"}
        for case in g["cases"]:
            assert abs(float(case["top_share_repr"])
                       - float(rows[case["case_id"]]["probe_top1_share"])) < TOL, case


def test_top_visit_ties_are_reported_not_hidden():
    """A tie makes the §7.0 canonical comparator load-bearing for that case, so
    the count must stay visible rather than being silently resolved."""
    _rec, art = _selected_moves()
    reported = art["mcts"]["positions_with_a_tie_at_the_top"]
    counted = sum(1 for g in art["gates"].values()
                  for c in g["cases"] if c["tied_with_top"] > 0)
    assert reported == counted
    assert "canonical move order" in art["mcts"]["tie_break"]
