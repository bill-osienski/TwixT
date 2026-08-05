"""Atlas Stage 4, Task 6 -- artifact schema, provenance and emission.

The Stage 3 producer document is stored ONCE, undivided, under `snapshots`, so
an artifact row is directly consumable by Read-outs A, B and C. Emission is
fail-closed on provenance.
"""
import json

import pytest

from scripts.GPU.alphazero.atlas_artifact import (
    ROW_SCHEMA_VERSION, build_row, emit, validate_provenance,
)
from scripts.GPU.alphazero.warm_prefix_replay import LegResult

# Valid synthetic provenance. Emission is fail-closed, so serialization tests
# must supply one rather than proving that an empty object gets through.
PROV = {"git_head": "a" * 40, "worktree_clean": True,
        "checkpoint_sha1": "0" * 40}


def _snapshots(**over):
    """The Stage 3 producer document, stored ONCE and undivided."""
    base = {"at_boundary": {"by_shape": {}}, "at_400": {"by_shape": {}},
            "captures": {"at_start": {}, "at_boundary": {}, "at_400": {}},
            "parent_visits": {"at_boundary": {(): 463}, "at_400": {(): 537}},
            "reference_lines": {"at_3200": {"moves": [7]},
                                "at_6400": {"moves": [7]},
                                "merged": {"required_edges": [],
                                           "agreement": {}}}}
    base.update(over)
    return base


def _kw(**over):
    base = dict(
        game_idx=3, replay_seed=20400003, target_ply=95, phase="late",
        side="black", split="validation", inherited_I=137,
        reset_count=1, reset_rate=0.02, last_reset_ply=44,
        boundary={"N_actual": 326, "overshoot": 6, "remaining": 74,
                  "flush_type": "full"},
        legs=[{"nominal_B": 400}], label="misleading",
        features_at_boundary={"one_visit_backup_share": 0.4},
        features_at_400={"one_visit_backup_share": 0.3},
        snapshots=_snapshots(),
        flat_policy=False, near_even=True)
    base.update(over)
    return base


def _real_legs():
    """A single genuine LegResult. `_kw`'s default `[{"nominal_B": 400}]` is a
    bare dict that predates load_run, and rehydrating it would fabricate the
    fields it lacks."""
    return [LegResult(nominal_B=400, inherited_I=137, effective=537,
                      root_value=0.05, selected_move=3,
                      selected_move_prior_rank=1, top_share=0.5,
                      top_two_margin=0.2, effective_children=12.0,
                      n_visited_children=20, visit_counts={3: 100})]


def _tracer_snapshots():
    """Tracer snapshots shaped like SelectionTracer.snapshot() emits.

    `_snapshots`'s default `{"by_shape": {}}` is enough for emission but Read-out
    C indexes `by_shape["c4a05"]`, so the round trip needs the real shape.
    """
    from scripts.GPU.alphazero.selection_tracer import WIDENING_SHAPES
    cell = {"eligible_events": 200, "outside_events": 30,
            "first_touch_events": 100, "first_touch_outside_events": 15,
            "lagged_first_touch_outside_events": 12,
            "excluded_prior_mass": 80.0, "outside_rate": 0.15,
            "first_touch_outside_rate": 0.15,
            "mean_excluded_prior_mass": 0.4}
    block = {**{k: dict(cell) for k in ("overall", "0", "1", "2+")},
             "forced_root_bypass_events": 0,
             "forced_root_bypass_outside_events": 0,
             "forced_root_bypass_outside_rate": None,
             "meaningfully_affected": True}
    snap = {"by_shape": {n: dict(block) for n, _c, _a in WIDENING_SHAPES},
            "within_forced_events": 0}
    return {"at_boundary": snap, "at_400": dict(snap)}


def test_row_carries_BOTH_feature_captures():
    """Section 6a: B=400 supplies the required 400-tree diagnostic contrast, so
    it must survive into the artifact, not just the boundary capture."""
    r = build_row(**_kw())
    assert r["features_at_boundary"]["one_visit_backup_share"] == 0.4
    assert r["features_at_400"]["one_visit_backup_share"] == 0.3


def test_the_row_stores_ONE_undivided_producer_document():
    """Amendment 4's output is kept whole, under the key Read-out C consumes.
    The tree is gone by the time anything re-reads this row, so a map that is
    not carried is a permanently missing measurement -- and a map carried in
    two overlapping places is how the two copies drift."""
    r = build_row(**_kw())
    assert set(r["snapshots"]["reference_lines"]) == {"at_3200", "at_6400",
                                                      "merged"}
    assert set(r["snapshots"]["parent_visits"]) == {"at_boundary", "at_400"}
    # No duplicated copies and no abolished singular field.
    for gone in ("reference_line", "reference_lines", "parent_visits",
                 "tracer_snapshots"):
        assert gone not in r


def test_an_artifact_row_IS_a_readout_row_for_both_consumers():
    """The seam is the contract: no translation layer exists to drift."""
    r = build_row(**_kw())
    assert {"snapshots", "label", "phase", "flat_policy", "near_even"} <= set(r)
    assert {"label", "features_at_boundary", "features_at_400"} <= set(r)


def test_the_rows_NATIVE_shapes_survive_emission():
    """The row holds native Python and `_jsonable` normalizes at the boundary.

    Tuple KEYS join with "|", so the root path () becomes "" -- deterministic,
    but surprising enough that a reader must not mistake it for an absent
    entry. Dataclasses convert too, which is what lets `legs` stay a list of
    LegResult objects that Read-out B can read by ATTRIBUTE.
    """
    leg = LegResult(nominal_B=400, inherited_I=137, effective=537,
                    root_value=0.25, selected_move=7,
                    selected_move_prior_rank=1, top_share=0.5,
                    top_two_margin=0.2, effective_children=12.0,
                    n_visited_children=20, visit_counts={7: 100})
    r = build_row(**_kw(legs=[leg], snapshots=_snapshots(
        parent_visits={"at_boundary": {(): 463, (7, 3): 12}})))
    back = json.loads(emit({"rows": [r], "provenance": PROV}))["rows"][0]
    assert back["snapshots"]["parent_visits"]["at_boundary"] == {"": 463,
                                                                "7|3": 12}
    assert back["legs"][0]["nominal_B"] == 400
    assert back["legs"][0]["root_value"] == 0.25


def test_row_carries_resets_remaining_strata_and_the_schema_version():
    r = build_row(**_kw())
    assert r["schema_version"] == ROW_SCHEMA_VERSION
    assert r["reset_count"] == 1 and r["reset_rate"] == 0.02
    assert r["last_reset_ply"] == 44 and r["boundary"]["remaining"] == 74
    assert r["near_even"] is True and r["flat_policy"] is False


def test_undefined_values_stay_None_through_emission():
    r = build_row(**_kw(reset_rate=None, last_reset_ply=None, boundary=None,
                        features_at_400=None))
    back = json.loads(emit({"rows": [r], "provenance": PROV}))["rows"][0]
    assert back["reset_rate"] is None and back["last_reset_ply"] is None
    assert back["boundary"] is None and back["features_at_400"] is None


def test_a_row_missing_the_boundary_is_flagged_not_defaulted():
    assert build_row(**_kw(boundary=None))["boundary_missing"] is True


def test_emission_goes_through_jsonable():
    run = {"rows": [], "provenance": PROV,
           "cells": {("discovery", "late", "red"): 12}}
    assert json.loads(emit(run))["cells"] == {"discovery|late|red": 12}


def test_emission_REJECTS_an_unserializable_payload():
    """No default=str: it would stringify a schema defect into a
    plausible-looking value instead of failing. Provenance is VALID here, so
    the TypeError proves the serializer refused -- not the provenance gate."""
    with pytest.raises(TypeError):
        emit({"rows": [{"bad": object()}], "provenance": PROV})


def test_emission_REFUSES_a_run_whose_provenance_does_not_validate():
    """A fail-closed check nothing invokes is decoration. Emission is the one
    place every artifact passes through, so the gate belongs here."""
    for bad in ({}, {"git_head": "a" * 40, "worktree_clean": False,
                     "checkpoint_sha1": "0" * 40}):
        with pytest.raises(ValueError, match="provenance"):
            emit({"rows": [], "provenance": bad})


def test_provenance_fails_closed_on_a_dirty_tree():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": False,
                             "checkpoint_sha1": "0" * 40})
    assert r["verdict"] == "PROVENANCE_FAILURE" and "worktree_clean" in r["problems"]


def test_provenance_requires_a_checkpoint_digest():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": True,
                             "checkpoint_sha1": ""})
    assert "checkpoint_sha1" in r["problems"]


def test_a_forty_character_non_hexadecimal_digest_is_rejected():
    """Length alone is not a SHA-1. Checking only `len == 40` accepts a
    placeholder, a truncated path, or a typo'd branch name."""
    r = validate_provenance({"git_head": "z" * 40, "worktree_clean": True,
                             "checkpoint_sha1": "not-a-hash" + "x" * 30})
    assert r["verdict"] == "PROVENANCE_FAILURE"
    assert set(r["problems"]) == {"git_head", "checkpoint_sha1"}


def test_valid_provenance_passes():
    r = validate_provenance(PROV)
    assert r["verdict"] == "OK" and r["problems"] == []
    # Upper case is still hexadecimal.
    assert validate_provenance({**PROV, "git_head": "A" * 40})["verdict"] == "OK"


# -- Stage 5, Task 1: the artifact must be RELOADABLE ------------------------

from scripts.GPU.alphazero.atlas_artifact import load_run
from scripts.GPU.alphazero.warm_prefix_replay import BoundaryRecord


def test_load_run_is_the_inverse_of_emit_for_every_lossy_type():
    leg = LegResult(nominal_B=400, inherited_I=137, effective=537,
                    root_value=0.25, selected_move=7,
                    selected_move_prior_rank=1, top_share=0.5,
                    top_two_margin=0.2, effective_children=12.0,
                    n_visited_children=20, visit_counts={7: 100})
    deep_edge = {"parent_path": (7,), "move": 3, "depth": 1,
                 "parent_priors": {3: 0.7, 4: 0.3}}
    snaps = _snapshots(
        parent_visits={"at_boundary": {(): 463, (7, 3): 12},
                       "at_400": {(): 537}},
        reference_lines={
            "at_3200": {"edges": [dict(deep_edge)], "moves": [3]},
            "at_6400": {"edges": [dict(deep_edge)], "moves": [3]},
            "merged": {"required_edges": [
                {"parent_path": (), "move": 7, "depth": 0,
                 "parent_priors": {7: 0.6, 8: 0.4},
                 "sources": (3200, 6400)}], "agreement": {}}})
    row = build_row(**_kw(legs=[leg], boundary=None, snapshots=snaps))
    back = load_run(emit({"rows": [row], "provenance": PROV}))["rows"][0]

    # Dataclasses, by ATTRIBUTE -- Read-out B and atlas_labelling need this.
    assert back["legs"][0].nominal_B == 400
    assert back["legs"][0].visit_counts == {7: 100}        # int keys, not "7"
    # Tuple paths, including the empty root path.
    pv = back["snapshots"]["parent_visits"]["at_boundary"]
    assert pv[()] == 463 and pv[(7, 3)] == 12
    # Edge identity: tuple path, int-keyed priors, tuple sources.
    lines = back["snapshots"]["reference_lines"]
    edge = lines["merged"]["required_edges"][0]
    assert edge["parent_path"] == () and edge["sources"] == (3200, 6400)
    assert edge["parent_priors"] == {7: 0.6, 8: 0.4}
    # BOTH DEEP LINES too: `merged` uses `required_edges`, the deep lines use
    # `edges`, and rehydrating only one leaves the other list-pathed and
    # string-keyed -- an inverse in name only.
    for rung in ("at_3200", "at_6400"):
        e = lines[rung]["edges"][0]
        assert e["parent_path"] == (7,)
        assert e["parent_priors"] == {3: 0.7, 4: 0.3}


def test_load_run_refuses_a_truncated_document():
    """A file that lost its rows must not read as a valid zero-row run."""
    doc = json.loads(emit({"rows": [], "provenance": PROV}))
    del doc["rows"]
    with pytest.raises(ValueError, match="rows"):
        load_run(json.dumps(doc))
    with pytest.raises(ValueError, match="rows"):
        load_run(json.dumps({**doc, "rows": {"0": {}}}))   # not a list


def test_a_boundary_record_rehydrates_or_stays_None():
    # REAL legs: load_run rehydrates LegResult and must not be made tolerant of
    # a partial one, so the fixture supplies what the production path emits.
    row = build_row(**_kw(legs=_real_legs(),
                          boundary=BoundaryRecord(N_actual=326, overshoot=6,
                                                  remaining=74,
                                                  flush_type="full")))
    back = load_run(emit({"rows": [row], "provenance": PROV}))["rows"][0]
    assert back["boundary"].remaining == 74
    none_row = build_row(**_kw(legs=_real_legs(), boundary=None))
    back = load_run(emit({"rows": [none_row], "provenance": PROV}))["rows"][0]
    assert back["boundary"] is None and back["boundary_missing"] is True


def test_load_run_AUTHENTICATES_rather_than_merely_parsing():
    """A hand-edited or truncated artifact must not be consumable."""
    good = emit({"rows": [], "provenance": PROV})
    load_run(good)                                   # baseline: accepted
    doc = json.loads(good)
    doc["provenance"]["worktree_clean"] = False
    with pytest.raises(ValueError, match="provenance"):
        load_run(json.dumps(doc))
    doc = json.loads(good)
    doc["rows"] = [dict(json.loads(emit({"rows": [build_row(**_kw(
        legs=_real_legs()))], "provenance": PROV}))["rows"][0],
                        schema_version=999)]
    with pytest.raises(ValueError, match="schema_version"):
        load_run(json.dumps(doc))


def test_the_ROUND_TRIP_feeds_all_three_readouts(tmp_path):
    """emit -> DISK -> load -> A, B and C. The two-stage protocol is exactly
    this path, so it is qualified as one."""
    from scripts.GPU.alphazero.atlas_readout_a import evaluate_detector_both
    from scripts.GPU.alphazero.atlas_readout_b import calibrate_gate
    from scripts.GPU.alphazero.atlas_readout_c import aggregate_shape

    def _four_rungs():
        """All four frozen rungs -- labelling and Read-out B index every one."""
        return [LegResult(nominal_B=b, inherited_I=10, effective=10 + b,
                          root_value=0.05, selected_move=3,
                          selected_move_prior_rank=1, top_share=0.5,
                          top_two_margin=0.2, effective_children=12.0,
                          n_visited_children=20, visit_counts={3: 100})
                for b in (400, 1600, 3200, 6400)]

    rows = [build_row(**_kw(legs=_four_rungs(), label=lbl,
                            snapshots=_snapshots(**_tracer_snapshots())))
            for lbl in ("misleading", "stable_negative")]
    p = tmp_path / "pilot_artifact.json"
    p.write_text(emit({"rows": rows, "provenance": PROV}))

    back = load_run(p)["rows"]
    # B reads legs by attribute; a dict would raise AttributeError here.
    assert calibrate_gate(back, "top_share_increase")["verdict"] in {
        "needs review", "no finding"}
    # C indexes parent_visits by tuple and ranks int-keyed priors.
    agg = aggregate_shape(back, ("c4a05", 4.0, 0.5))
    assert agg["gated_on"] == "at_400"
    # A reads the two feature dicts off the row.
    r = evaluate_detector_both(back, back, replicates=8)
    assert r["authoritative"] == "features_at_boundary"
