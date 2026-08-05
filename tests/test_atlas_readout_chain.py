"""Real Stage 3 output -> real Read-outs A, B and C. No surrogates.

Every read-out below consumes a REAL `build_row` result. The producer/consumer
seam is the thing under test, and a chain test that rebuilds its own row shape
tests everything except the seam.

CPU-only: FakeEvaluator at active_size=6. No reservoir, no checkpoint, no MLX.
"""
import json
import random

import pytest

from scripts.GPU.alphazero.atlas_artifact import build_row, emit
from scripts.GPU.alphazero.atlas_labelling import class_counts, classify_row
from scripts.GPU.alphazero.atlas_readout_a import (
    collect_features, deployability, evaluate_detector, evaluate_detector_both,
)
from scripts.GPU.alphazero.atlas_readout_b import (
    by_stratum_summary, calibrate_gate, natural_convergence_report,
)
from scripts.GPU.alphazero.atlas_readout_c import (
    aggregate_shape, classify_strata, intervention_from_snapshots,
    select_on_discovery_validate_on_selected, validation_verdict,
)
from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from scripts.GPU.alphazero.warm_prefix_replay import (
    BatchSafeBoundaryObserver, replay_prefix, run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6
SHAPE = ("c4a05", 4.0, 0.5)
PROV = {"git_head": "a" * 40, "worktree_clean": True,
        "checkpoint_sha1": "0" * 40}


def _history(n, size=SIZE):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=size, to_move="red")
    out = []
    for _ in range(n):
        lm = s.legal_moves()
        if not lm:
            break
        out.append(lm[0])
        s = s.apply_move(lm[0])
    return out


@pytest.fixture(scope="module")
def real_row():
    """ONE shared fixture at the FROZEN increments -- 6,400 FakeEvaluator
    simulations at active_size=6, CPU-only."""
    hist = _history(4)
    meta = GameMeta(game_id=0, seed=BASE, n_moves=len(hist), start_player="red")
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=1, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(BASE))
    pre = replay_prefix(m, meta, hist, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, tracer=tracer)
    legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                      boundary_observer=obs,
                                      target_tracer=tracer)   # frozen defaults
    return {"meta": meta, "pre": pre, "legs": legs, "snaps": snaps, "obs": obs}


@pytest.fixture(scope="module")
def artifact_row(real_row):
    """A REAL `build_row` result.

    Every downstream read-out below consumes THIS, never a hand-written
    stand-in. The producer/consumer seam is the thing under test.
    """
    pre, legs, snaps, obs = (real_row["pre"], real_row["legs"],
                             real_row["snaps"], real_row["obs"])
    caps = snaps["captures"]
    return build_row(
        game_idx=real_row["meta"].game_id, replay_seed=real_row["meta"].seed,
        target_ply=2, phase="opening", side="red", split="discovery",
        inherited_I=pre.inherited_I, reset_count=pre.reset_count,
        reset_rate=pre.reset_rate, last_reset_ply=pre.last_reset_ply,
        # Dataclasses, not vars(): the read-outs address these by attribute and
        # `_jsonable` flattens them at emission.
        boundary=obs.record, legs=legs, label=classify_row(legs),
        features_at_boundary=collect_features(caps["at_start"],
                                              caps["at_boundary"],
                                              obs.record.N_actual),
        features_at_400=collect_features(caps["at_start"], caps["at_400"], 400),
        # SUPPLIED facts in this synthetic chain -- see the Stage 5 handoff note
        # in the plan's completion criteria. Stage 5 must DERIVE them.
        snapshots=snaps, flat_policy=False, near_even=True)


def test_the_fixture_uses_the_frozen_rungs(real_row):
    assert [l.nominal_B for l in real_row["legs"]] == [400, 1600, 3200, 6400]


def test_real_legs_classify_and_count(real_row):
    label = classify_row(real_row["legs"])
    assert label in {"misleading", "stable_negative", "ambiguous",
                     "no_stable_reference"}
    assert set(class_counts([real_row["legs"]])) >= {"misleading",
                                                     "stable_negative"}


def test_readout_A_features_come_from_the_frozen_captures(real_row):
    caps, obs = real_row["snaps"]["captures"], real_row["obs"]
    f = collect_features(caps["at_start"], caps["at_boundary"],
                         obs.record.N_actual)
    assert set(f) == {"one_visit_backup_share", "depth3plus_backup_fraction",
                      "leader_visit_margin", "root_policy_entropy",
                      "leader_breadth"}
    f400 = collect_features(caps["at_start"], caps["at_400"], 400)
    assert f400 is not None            # the 400-tree diagnostic contrast


def test_readout_A_detector_runs_end_to_end_on_real_features(real_row):
    caps, obs = real_row["snaps"]["captures"], real_row["obs"]
    f = collect_features(caps["at_start"], caps["at_boundary"],
                         obs.record.N_actual)
    rows = ([{"label": "misleading", "features": f} for _ in range(20)]
            + [{"label": "stable_negative", "features": f} for _ in range(25)])
    r = evaluate_detector(discovery=rows, validation=rows, replicates=64)
    # Identical features cannot separate; the point is that the REAL path runs
    # and reports a verdict rather than raising. `replicates` is cut from the
    # frozen 10,000 because this row set reaches the bootstrap: 10,000 x 500
    # rank comparisons is several seconds of pure Python for no added signal.
    assert r["verdict"] in {"PASS", "FAIL", "INSUFFICIENT_CLASSES",
                            "INSUFFICIENT_DISCOVERY_CLASSES"}
    d = deployability([real_row["obs"].record.remaining])
    assert d["verdict"] in {"DEPLOYABLE", "NOT_DEPLOYABLE"}


def test_readout_B_consumes_the_ARTIFACT_ROW_directly(artifact_row):
    """A Read-out B row is an artifact row too: `legs`, `phase`, `flat_policy`
    and `near_even` are all already there, and `legs` holds LegResult OBJECTS
    so the attribute access in `_by_b` works on the real row."""
    assert calibrate_gate([artifact_row], "top_share_increase")["verdict"] in {
        "needs review", "no finding"}
    nc = natural_convergence_report([artifact_row])
    assert nc["transition"] == "400->6400" and nc["is_causal_evidence"] is False
    assert "overall" in by_stratum_summary([artifact_row], "top_share_increase")


def test_the_real_ladder_freezes_both_deep_lines_and_both_visit_maps(real_row):
    """The producer half of amendment 4, driven by the real ladder rather than
    by a hand-written stand-in for it."""
    snaps = real_row["snaps"]
    assert set(snaps["reference_lines"]) == {"at_3200", "at_6400", "merged"}
    assert set(snaps["parent_visits"]) == {"at_boundary", "at_400"}
    merged = snaps["reference_lines"]["merged"]
    assert set(merged["agreement"]) == {"root", "reply", "two_ply"}
    # Every edge carries its own parent's priors, and no deep-rung visit count.
    for e in merged["required_edges"]:
        assert e["parent_priors"] and "parent_effective_visits" not in e


def test_readout_C_consumes_the_ARTIFACT_ROW_directly(artifact_row):
    """No surrogate row: `aggregate_shape` is handed `build_row`'s output.

    At active_size=6 the admitted set clamps to n_legal, so these numbers
    cannot be interesting -- the point is that the real seam holds and the
    real path reports rates rather than raising.
    """
    assert isinstance(classify_strata(artifact_row), set)
    for instant in ("at_boundary", "at_400"):
        iv = intervention_from_snapshots(artifact_row["snapshots"], "c4a05",
                                         instant=instant)
        assert iv["verdict"] in {"OK", "INCONCLUSIVE", "NO_EVENTS",
                                 "NO_SNAPSHOT"}
    agg = aggregate_shape([artifact_row], SHAPE)
    assert agg["gated_on"] == "at_400"
    assert set(agg["instants"]) == {"at_boundary", "at_400"}
    assert set(agg) >= {"root_retention", "misleading_intervention",
                        "inconclusive", "counters", "agreement",
                        "retention_rows", "rows_without_stable_reference"}
    assert validation_verdict(agg)["verdict"] in {"FAIL", "INCONCLUSIVE", "PASS"}


def test_readout_C_selection_and_verdict_run_on_real_artifact_rows(artifact_row):
    """Labels are forced, because whatever the FakeEvaluator ladder happens to
    classify this position as must not decide whether the test exercises the
    intervention denominators."""
    rows = [{**artifact_row, "label": "misleading"},
            {**artifact_row, "label": "stable_negative"}]
    r = select_on_discovery_validate_on_selected(rows, rows)
    assert r["selected_on"] == "discovery"
    assert set(r["validated"]) <= {r["selected"]}
    if r["selected"] is None:
        assert r["validation_verdict"] is None
    else:
        assert r["validation_verdict"]["verdict"] in {"FAIL", "INCONCLUSIVE",
                                                      "PASS"}


def test_readout_A_dual_pipeline_consumes_the_artifact_row(artifact_row):
    """The detector row IS the artifact row -- no translation layer to drift."""
    rows = ([{**artifact_row, "label": "misleading"} for _ in range(20)]
            + [{**artifact_row, "label": "stable_negative"} for _ in range(25)])
    r = evaluate_detector_both(rows, rows, replicates=32)
    assert r["authoritative"] == "features_at_boundary"
    assert r["verdict"] in {"PASS", "FAIL", "LATE_ONLY_SEPARATION",
                            "INSUFFICIENT_CLASSES",
                            "INSUFFICIENT_DISCOVERY_CLASSES"}
    assert r["row_overlap"]["discovery"]["identical"] is True
    assert r["row_overlap"]["validation"]["identical"] is True


def test_a_real_row_survives_the_artifact_boundary(artifact_row, real_row):
    back = json.loads(emit({"rows": [artifact_row],
                            "provenance": PROV}))["rows"][0]
    assert back["inherited_I"] == real_row["pre"].inherited_I
    assert len(back["legs"]) == 4
    assert back["features_at_boundary"] is not None
    assert back["features_at_400"] is not None
    # Amendment 4's producer output survives the JSON boundary, tuple keys and
    # all: the root path () emits as the empty-string key.
    assert set(back["snapshots"]["reference_lines"]) == {"at_3200", "at_6400",
                                                         "merged"}
    assert "" in back["snapshots"]["parent_visits"]["at_400"]


def test_emission_of_a_real_run_still_fails_closed_on_provenance(artifact_row):
    """The gate is not bypassed by a row that is otherwise perfectly valid."""
    with pytest.raises(ValueError, match="provenance"):
        emit({"rows": [artifact_row], "provenance": {}})
