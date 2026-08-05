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
