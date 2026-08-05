"""Stage 5 composition -- CPU only, FakeEvaluator, no reservoir or checkpoint."""
import json

import pytest

from scripts.GPU.alphazero.atlas_labelling import ALLOWED_N, class_counts, \
    classify_row, size_from_pilot
from scripts.GPU.alphazero.atlas_readout_a import FEATURE_NAMES
from scripts.GPU.alphazero.atlas_row_facts import derive_row_facts
from scripts.GPU.alphazero.atlas_run import (
    RowOutcome, _early_widening_check, combine_final_runs, pilot_rows,
    run_corpus, run_final, run_pilot, verify_assignment, verify_pilot,
)
from scripts.GPU.alphazero.corpus_geometry import (
    PILOT_GAMES, PILOT_PER_CELL, GameMeta, assign_corpus, phase_for_ply,
    pilot_geometry_gate, side_for_ply,
)
from scripts.GPU.alphazero.warm_prefix_replay import BoundaryRecord, LegResult

from tests.eval_fakes import FakeEvaluator

BASE = 20500000
SAMPLING_SEED = 20260805
_PROV = {"git_head": "a" * 40, "worktree_clean": True,
         "checkpoint_sha1": "0" * 40}

# The LADDER must stay frozen: labelling and Read-out B index nominal_B at
# 400/1,600/3,200/6,400, so shrinking the increments produces rungs they
# rightly reject. Only the PREFIX budget is reduced.
#
# Row-level tests run at active_size=6, which is ~10x cheaper per simulation
# and needs no late ply. The PILOT must use board 24: a late cell needs ply
# 91+, and a 6x6 fixture game terminates around 29 moves.
_SMALL = dict(prefix_sims=2, active_size=6)
_BOARD24 = dict(prefix_sims=2)


def _history(n, size=24):
    """A real board-24 history. `legal_moves()[0]` walks a single file and can
    connect early, so the stride spreads the moves out."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s, out = TwixtState(active_size=size, to_move="red"), []
    while len(out) < n:
        if s.is_terminal():
            break
        lm = s.legal_moves()
        if not lm:
            break
        mv = lm[(len(out) * 37) % len(lm)]
        out.append(mv)
        s = s.apply_move(mv)
    return out


_HIST_CACHE = {}


def _late_history(min_plies=92):
    """A REAL board-24 history long enough to serve a late cell.

    A 6x6 fixture game terminates after roughly 29 legal moves, so
    `active_size=6` can never reach ply 91+. `replay_prefix` also asserts
    `meta.n_moves == len(move_history)`, so the length must be DERIVED.
    """
    if "hist" not in _HIST_CACHE:
        hist = _history(min_plies + 8)
        assert len(hist) >= min_plies, (
            f"fixture produced only {len(hist)} plies; a late cell needs 91+")
        _HIST_CACHE["hist"] = hist
    return _HIST_CACHE["hist"]


def _small_history():
    if "small" not in _HIST_CACHE:
        _HIST_CACHE["small"] = _history(4, size=6)
    return _HIST_CACHE["small"]


def _meta(game_id=0, n_moves=None):
    n = len(_small_history()) if n_moves is None else n_moves
    return GameMeta(game_id=game_id, seed=BASE + game_id, n_moves=n,
                    start_player="red")


def _assigned(game_id=0, split="discovery", ply=2):
    return {"game_id": game_id, "seed": BASE + game_id, "split": split,
            "phase": phase_for_ply(ply), "side": side_for_ply(ply, "red"),
            "ply": ply}


@pytest.fixture(scope="module")
def one_row():
    from scripts.GPU.alphazero.atlas_run import run_row
    return run_row(FakeEvaluator(value=0.0), _meta(), _assigned(),
                   move_history=_small_history(), base_seed=BASE, **_SMALL)


def test_a_row_produces_a_complete_artifact_row(one_row):
    assert isinstance(one_row, RowOutcome)
    assert one_row.ok is True and one_row.failure is None
    row = one_row.row
    # The artifact row is simultaneously an A row, a B row and a C row.
    assert {"snapshots", "label", "phase", "flat_policy", "near_even"} <= set(row)
    assert {"features_at_boundary", "features_at_400"} <= set(row)
    assert len(row["legs"]) == 4


def test_the_row_facts_are_DERIVED_not_supplied(one_row):
    """The Stage 4 gap: these were hardcoded booleans everywhere."""
    row = one_row.row
    assert row["phase"] == "opening" and row["side"] == "red"
    assert row["flat_policy"] in (True, False, None)
    assert row["near_even"] in (True, False, None)
    assert "row_facts_undefined" in row


def test_ONE_evaluator_is_shared_across_every_row():
    """Section 2b: construct the evaluator once for the whole run. Rebuilding a
    compiled evaluator per unit of work is the documented MLX trap."""
    ev, seen = FakeEvaluator(value=0.0), []
    run_corpus(ev, [_meta(0), _meta(1)], [_assigned(0), _assigned(1)],
               base_seed=BASE,
               move_histories={0: _small_history(), 1: _small_history()},
               provenance=_PROV, _on_row=lambda m: seen.append(m), **_SMALL)
    assert len(seen) == 2
    assert seen[0].evaluator is seen[1].evaluator is ev      # identity


def test_each_row_gets_its_OWN_mcts_seeded_from_the_verified_replay_seed():
    ev, seen = FakeEvaluator(value=0.0), []
    run_corpus(ev, [_meta(0), _meta(1)], [_assigned(0), _assigned(1)],
               base_seed=BASE,
               move_histories={0: _small_history(), 1: _small_history()},
               provenance=_PROV, _on_row=lambda m: seen.append(m), **_SMALL)
    assert seen[0] is not seen[1]            # a fresh MCTS per row
    assert seen[0].rng is not seen[1].rng


def test_a_sidecar_seed_mismatch_fails_the_row_rather_than_being_assumed():
    """replay_seed_for verifies base_seed + game_idx against the sidecar."""
    from scripts.GPU.alphazero.atlas_run import run_row
    bad = GameMeta(game_id=0, seed=BASE + 99, n_moves=len(_small_history()),
                   start_player="red")
    out = run_row(FakeEvaluator(value=0.0), bad, _assigned(),
                  move_history=_small_history(), base_seed=BASE, **_SMALL)
    assert out.ok is False and "seed" in out.failure


def test_a_backup_invariant_violation_FAILS_the_row_and_is_recorded():
    """Section 6a: a violation means the accounting is wrong and the row must
    fail, not be recorded."""
    from scripts.GPU.alphazero.atlas_run import run_row
    out = run_row(FakeEvaluator(value=0.0), _meta(), _assigned(),
                  move_history=_small_history(), base_seed=BASE,
                  _corrupt_d3=True, **_SMALL)
    assert out.ok is False and "backup accounting" in out.failure
    assert out.row is None                   # not a half-recorded row


def test_a_row_whose_boundary_never_fired_is_FAILED_not_defaulted():
    """A missing boundary is a missing measurement, not an N_actual of zero."""
    from scripts.GPU.alphazero.atlas_run import run_row
    out = run_row(FakeEvaluator(value=0.0), _meta(), _assigned(),
                  move_history=_small_history(), base_seed=BASE,
                  prefix_sims=2, active_size=6,
                  threshold=10_000)              # no flush can reach it
    assert out.ok is False and "boundary" in out.failure


def test_inheritance_resets_KEEP_the_row(one_row):
    """Section 2b: never top up, drop, substitute or resample a row."""
    assert one_row.ok is True
    assert "reset_count" in one_row.row and "reset_rate" in one_row.row


# -- the completeness condition ----------------------------------------------

def _mixed_corpus():
    metas = [_meta(0), GameMeta(game_id=1, seed=BASE + 99,
                                n_moves=len(_small_history()),
                                start_player="red")]
    return run_corpus(FakeEvaluator(value=0.0), metas,
                      [_assigned(0), _assigned(1)], base_seed=BASE,
                      move_histories={0: _small_history(), 1: _small_history()},
                      provenance=_PROV, **_SMALL)


def test_ANY_row_failure_makes_the_whole_run_ABORTED_and_non_authoritative():
    """The frozen corpus is exactly N assigned positions. A run that measured
    N-k of them is not the atlas, however good the surviving rows look.

    A COMPLETENESS condition, not a statistical threshold: it compares assigned
    against measured and introduces no number. One failure is disqualifying, so
    no "maximum tolerable failures" knob exists or can exist.
    """
    doc = _mixed_corpus()
    assert doc["verdict"] == "ABORTED"
    assert doc["authoritative"] is False
    assert doc["assigned"] == 2 and doc["measured"] == 1


def test_an_aborted_run_still_retains_its_diagnostics():
    doc = _mixed_corpus()
    assert len(doc["rows"]) == 1                    # the row that succeeded
    assert len(doc["failed_rows"]) == 1
    assert doc["failed_rows"][0]["game_id"] == 1
    assert doc["readout_a"] is not None
    assert doc["readout_a_authoritative"] is False


def test_a_complete_run_is_OK_and_authoritative():
    doc = run_corpus(FakeEvaluator(value=0.0), [_meta(0), _meta(1)],
                     [_assigned(0), _assigned(1)], base_seed=BASE,
                     move_histories={0: _small_history(), 1: _small_history()},
                     provenance=_PROV, **_SMALL)
    assert doc["verdict"] == "OK" and doc["authoritative"] is True
    assert doc["assigned"] == doc["measured"] == 2


def test_no_maximum_failure_threshold_is_invented():
    """Completeness is binary. A tolerance would be a new number, and there is
    nothing to tune: one unmeasured assigned position already disqualifies."""
    import inspect
    from scripts.GPU.alphazero.atlas_run import run_row
    for fn in (run_corpus, run_row):
        params = set(inspect.signature(fn).parameters)
        for invented in ("max_failures", "failure_rate", "tolerance",
                         "allow_partial", "min_measured"):
            assert invented not in params


def test_run_corpus_composes_all_three_readouts_into_one_document():
    doc = run_corpus(FakeEvaluator(value=0.0),
                     [_meta(i) for i in range(4)],
                     [_assigned(0, "discovery"), _assigned(1, "discovery"),
                      _assigned(2, "validation"), _assigned(3, "validation")],
                     base_seed=BASE,
                     move_histories={i: _small_history() for i in range(4)},
                     provenance=_PROV, **_SMALL)
    assert set(doc) >= {"rows", "provenance", "readout_a", "readout_b",
                        "readout_c", "class_counts", "capacity",
                        "failed_rows", "row_facts_undefined"}
    assert doc["splits"] == {"discovery": 2, "validation": 2}
    assert doc["readout_a"]["authoritative"] == "features_at_boundary"
    assert doc["readout_c"]["selected_on"] == "discovery"


def test_the_run_document_survives_emission_with_valid_provenance():
    from scripts.GPU.alphazero.atlas_artifact import emit
    doc = run_corpus(FakeEvaluator(value=0.0), [_meta()], [_assigned()],
                     base_seed=BASE, move_histories={0: _small_history()},
                     provenance=_PROV, **_SMALL)
    back = json.loads(emit(doc))
    assert back["provenance"]["worktree_clean"] is True
    assert "" in back["rows"][0]["snapshots"]["parent_visits"]["at_400"]


def test_run_corpus_cannot_build_an_evaluator():
    """The MLX trap is unreachable by construction: there is no factory
    parameter to pass, only an already-built evaluator."""
    import inspect
    params = set(inspect.signature(run_corpus).parameters)
    assert "evaluator" in params
    for forbidden in ("evaluator_factory", "checkpoint", "checkpoint_path"):
        assert forbidden not in params


# -- section 3's chronology: pilot first, then final --------------------------

def _pilot_metas(n_moves=None):
    n = len(_late_history()) if n_moves is None else n_moves
    return [GameMeta(game_id=i, seed=BASE + i, n_moves=n, start_player="red")
            for i in range(PILOT_GAMES)]


def _pilot_assignment():
    gate = pilot_geometry_gate(_pilot_metas(), SAMPLING_SEED)
    assert gate["verdict"] == "PASS"
    return gate["assignment"]


def test_the_pilot_fixture_can_actually_serve_a_late_cell():
    """The fixture is load-bearing: if it cannot reach ply 91+, the geometry
    gate fails and every pilot test below is vacuous."""
    assert len(_late_history()) >= 92
    rows = pilot_rows(_pilot_metas(), SAMPLING_SEED)
    assert max(r["ply"] for r in rows) >= 91
    assert {r["phase"] for r in rows} == {"opening", "early_mid", "midgame",
                                          "late"}


def test_pilot_rows_are_the_24_fixed_discovery_rows():
    rows = pilot_rows(_pilot_metas(), SAMPLING_SEED)
    assert len(rows) == PILOT_GAMES
    cells = {}
    for r in rows:
        cells[(r["phase"], r["side"])] = cells.get((r["phase"], r["side"]), 0) + 1
    assert set(cells.values()) == {PILOT_PER_CELL}
    assert len(cells) == 8
    assert len({r["game_id"] for r in rows}) == PILOT_GAMES
    assert all(isinstance(r["ply"], int) for r in rows)


def test_pilot_rows_are_discovery_only_and_never_validation():
    """Section 3: included in DISCOVERY only and never eligible for
    validation."""
    assert {r["split"] for r in pilot_rows(_pilot_metas(), SAMPLING_SEED)} == {
        "discovery"}


def test_pilot_rows_fail_closed_when_the_geometry_gate_did_not_pass():
    """A no-go here costs nothing but the pilot block, which is the point."""
    short = [GameMeta(game_id=i, seed=BASE + i, n_moves=8, start_player="red")
             for i in range(PILOT_GAMES)]          # no game reaches ply 91
    with pytest.raises(ValueError, match="PHASE_GEOMETRY_NO_GO"):
        pilot_rows(short, SAMPLING_SEED)


# -- synthetic rows, for the gates that must reject before any measurement ----

def _four_rung_legs(label_as="stable_negative"):
    """All four frozen rungs, shaped to CLASSIFY as `label_as`.

    `class_counts` re-derives labels from the LEGS, so a fixture cannot simply
    claim a label: a pilot of 24 stable-negatives has p_m = 0 and its own
    sizing rule returns PROJECTED_CAPACITY_NO_GO, whatever the artifact says.
    """
    v400 = {"misleading": 0.90, "ambiguous": 0.20,
            "stable_negative": 0.06}[label_as]
    values = {400: v400, 1600: 0.10, 3200: 0.05, 6400: 0.05}
    return [LegResult(nominal_B=b, inherited_I=137, effective=137 + b,
                      root_value=values[b], selected_move=3,
                      selected_move_prior_rank=1, top_share=0.5,
                      top_two_margin=0.2, effective_children=12.0,
                      n_visited_children=20, visit_counts={3: 100})
            for b in (400, 1600, 3200, 6400)]


# 8 misleading + 9 stable-negative of 24 is the frozen formula's own worked
# case: max(60/(8/24), 75/(9/24)) = max(180, 200) = 200, already a multiple of
# 40. The remaining 7 are ambiguous, which section 5 keeps and counts.
_PILOT_MIX = (["misleading"] * 8 + ["stable_negative"] * 9 + ["ambiguous"] * 7)


def test_the_pilot_mix_really_produces_N_200():
    """If the fixture's own class counts do not yield 200, every run-final test
    built on it is asserting against an impossible artifact."""
    counts = class_counts([_four_rung_legs(m) for m in _PILOT_MIX])
    assert counts["misleading"] == 8 and counts["stable_negative"] == 9
    assert size_from_pilot(counts) == {"p_m": 8 / 24, "p_s": 9 / 24,
                                       "verdict": "OK", "N": 200,
                                       "required": 200}
    assert 200 in ALLOWED_N


def _populated_snapshots():
    """Snapshots Read-out C can aggregate, and derive_row_facts can read."""
    priors = {i: (1.0 if i == 3 else 0.5 - i * 1e-4) for i in range(500)}
    edge = {"parent_path": (), "move": 3, "depth": 0,
            "parent_priors": priors, "sources": (3200, 6400)}
    agree = {d: {"in_3200": True, "in_6400": True, "state": "agree"}
             for d in ("root", "reply", "two_ply")}
    return {"at_boundary": None, "at_400": None,
            "captures": {"at_start": {}, "at_boundary": {}, "at_400": {}},
            "parent_visits": {"at_boundary": {(): 463}, "at_400": {(): 537}},
            "reference_lines": {"at_3200": {"edges": [dict(edge)]},
                                "at_6400": {"edges": [dict(edge)]},
                                "merged": {"required_edges": [edge],
                                           "agreement": agree}}}


def _row_for(assigned, label_as="stable_negative", start_player="red"):
    """A schema-valid row FOR A SPECIFIC ASSIGNED ROW.

    Every identifying field is carried through, so a stub built from these
    cannot silently substitute a different corpus for the one the assignment
    selected. The row facts and the label are DERIVED exactly as production
    derives them, because `verify_pilot` re-derives both: a fixture that cannot
    survive the validation it is used to test qualifies nothing.
    """
    from scripts.GPU.alphazero.atlas_artifact import build_row
    legs, snaps = _four_rung_legs(label_as), _populated_snapshots()
    facts = derive_row_facts(legs, snaps, assigned["ply"], start_player,
                             assigned_phase=assigned["phase"],
                             assigned_side=assigned["side"])
    assert classify_row(legs) == label_as
    return build_row(
        game_idx=assigned["game_id"], replay_seed=assigned["seed"],
        target_ply=assigned["ply"], phase=facts["phase"], side=facts["side"],
        split=assigned["split"], inherited_I=137, reset_count=0,
        reset_rate=0.0, last_reset_ply=None,
        boundary=BoundaryRecord(N_actual=326, overshoot=6, remaining=74,
                                flush_type="full"),
        legs=legs, label=label_as,
        features_at_boundary={k: 0.5 for k in FEATURE_NAMES},
        features_at_400={k: 0.5 for k in FEATURE_NAMES},
        snapshots=snaps, flat_policy=facts["flat_policy"],
        near_even=facts["near_even"])


def _measured_pilot_rows():
    assigned = pilot_rows(_pilot_metas(), SAMPLING_SEED)
    return [_row_for(a, m) for a, m in zip(assigned, _PILOT_MIX)]


def _complete_pilot_doc(n=200):
    """What `run_pilot` writes: `sampling_seed` and the measured rows, and
    NOTHING else about the assignment."""
    return {"mode": "pilot", "verdict": "OK", "authoritative": True,
            "rows": _measured_pilot_rows(),
            "failed_rows": [], "assigned": 24, "measured": 24,
            "sampling_seed": SAMPLING_SEED,
            "sizing": {"p_m": 8 / 24, "p_s": 9 / 24, "verdict": "OK",
                       "N": n, "required": n}}


def _pilot_stub(n=200, rows=24, verdict="OK", authoritative=True,
                sizing_verdict="OK"):
    """A pilot DOCUMENT, not a pilot run: these gates must reject before any
    measurement is paid for, so they are tested without one."""
    return {"mode": "pilot", "verdict": verdict, "authoritative": authoritative,
            "rows": [{"split": "discovery"} for _ in range(rows)],
            "sampling_seed": SAMPLING_SEED,
            "sizing": {"verdict": sizing_verdict,
                       "N": n if sizing_verdict == "OK" else None}}


def _final_kw(**over):
    base = dict(pilot_games=_pilot_metas(), continuation_games=[],
                assignment_rows=[], base_seed=BASE, move_histories={},
                provenance=_PROV)
    base.update(over)
    return base


def test_run_final_takes_N_ONLY_from_the_pilot_sizing():
    """There is no n_target parameter to supply an invented value through."""
    import inspect
    assert "n_target" not in inspect.signature(run_final).parameters


def test_run_final_requires_the_continuation_to_be_exactly_N_minus_24():
    with pytest.raises(ValueError, match="exactly"):
        run_final(FakeEvaluator(value=0.0), pilot_doc=_pilot_stub(n=200),
                  **_final_kw(assignment_rows=[_assigned(100)] * 3))


def test_run_final_refuses_a_non_authoritative_or_unsized_pilot():
    for stub in (_pilot_stub(verdict="ABORTED", authoritative=False),
                 _pilot_stub(sizing_verdict="PROJECTED_CAPACITY_NO_GO"),
                 _pilot_stub(sizing_verdict="UNAVAILABLE"),
                 _pilot_stub(rows=23)):            # not the fixed 24
        with pytest.raises(ValueError):
            run_final(FakeEvaluator(value=0.0), pilot_doc=stub, **_final_kw())


def test_run_final_revalidates_the_pilots_OWN_sizing():
    """A stored N that the carried rows do not produce is the difference
    between "N=200" and "N=200 because 8 of 24 were misleading"."""
    pilot = _complete_pilot_doc(n=200)
    lied = dict(pilot, sizing={**pilot["sizing"], "N": 400})
    with pytest.raises(ValueError, match="sizing"):
        verify_pilot(lied, _pilot_metas())
    assert verify_pilot(pilot, _pilot_metas())


def test_verify_pilot_rejects_rows_that_are_not_the_recomputed_assignment():
    pilot = _complete_pilot_doc(n=200)
    tampered = dict(pilot, rows=[dict(pilot["rows"][0], target_ply=7)]
                    + pilot["rows"][1:])
    with pytest.raises(ValueError, match="recomputed"):
        verify_pilot(tampered, _pilot_metas())


def test_verify_pilot_compares_the_ARTIFACT_field_names_not_the_assignment_ones():
    """`build_row` stores game_idx / replay_seed / target_ply; `assign_corpus`
    emits game_id / seed / ply. A comparison that reads the assignment names
    off an artifact row raises KeyError on every honest call -- so the happy
    path is the test that catches it."""
    pilot = _complete_pilot_doc(n=200)
    assert "game_id" not in pilot["rows"][0]          # the trap, made explicit
    assert verify_pilot(pilot, _pilot_metas())
    bad_seed = dict(pilot, rows=[dict(pilot["rows"][0], replay_seed=BASE + 999)]
                    + pilot["rows"][1:])
    with pytest.raises(ValueError, match="recomputed"):
        verify_pilot(bad_seed, _pilot_metas())


def test_verify_pilot_refuses_a_row_whose_STORED_LABEL_was_edited():
    """Sizing reads the legs, but Read-out A takes its classes and Read-out C
    its intervention denominators from the stored `label`.

    The replacement is derived from the row's OWN label: `_PILOT_MIX` begins
    with eight misleading rows, so a hardcoded `label="misleading"` on row 0 is
    not a tamper at all.
    """
    pilot = _complete_pilot_doc(n=200)
    row = pilot["rows"][0]
    other = "stable_negative" if row["label"] != "stable_negative" else "misleading"
    with pytest.raises(ValueError, match="label"):
        verify_pilot(dict(pilot, rows=[dict(row, label=other)]
                          + pilot["rows"][1:]), _pilot_metas())


def test_verify_pilot_rederives_the_stratum_facts():
    """Each wrong value is the NEGATION of what the row actually carries.
    Writing literals here would assert nothing the moment the fixture derives
    those same values -- which is exactly what happened once already."""
    pilot = _complete_pilot_doc(n=200)
    row = pilot["rows"][0]
    for field in ("flat_policy", "near_even"):
        assert row[field] in (True, False)          # a negation must be real
        with pytest.raises(ValueError, match=field):
            verify_pilot(dict(pilot, rows=[dict(row, **{field: not row[field]})]
                              + pilot["rows"][1:]), _pilot_metas())


# -- the assignment is recomputed, never trusted -----------------------------

def _continuation_metas(n_games, start_index=24):
    """G_total - 24 games -- the COMPLETE authorized block, larger than the
    N - 24 rows the assignment selects from it."""
    n = len(_late_history())
    return [GameMeta(game_id=start_index + i, seed=BASE + start_index + i,
                     n_moves=n, start_player="red") for i in range(n_games)]


def _assigned_176():
    games = _continuation_metas(216)                 # G_total - 24
    return assign_corpus(_pilot_assignment(), games, 200,
                         SAMPLING_SEED)["rows"]


def test_the_continuation_split_is_96_discovery_and_80_validation():
    """8 x (3N/40 - 3) = 96 and 8 x N/20 = 80 at N = 200. With the pilot's 24
    counted as discovery the corpus is 120/80 -- the frozen 60/40."""
    rows = _assigned_176()
    assert len(rows) == 176
    assert sum(1 for r in rows if r["split"] == "discovery") == 96
    assert sum(1 for r in rows if r["split"] == "validation") == 80


def test_run_final_recomputes_the_assignment_rather_than_trusting_it():
    games, good = _continuation_metas(216), _assigned_176()
    args = (_pilot_metas(), _pilot_assignment(), SAMPLING_SEED, 200, games)
    verify_assignment(*args, good)                   # baseline: accepted
    tampered = [dict(good[0], ply=good[0]["ply"] + 2)] + good[1:]
    with pytest.raises(ValueError, match="recomputed"):
        verify_assignment(*args, tampered)


def test_the_continuation_BLOCK_is_larger_than_the_selected_rows():
    """G_total - 24 = 216 games supply N - 24 = 176 rows at N = 200. A fixture
    that made them equal would pass an assignment that selected everything."""
    games, rows = _continuation_metas(216), _assigned_176()
    assert len(games) == 216 and len(rows) == 176
    with pytest.raises(ValueError, match="frozen sizing"):
        verify_assignment(_pilot_metas(), _pilot_assignment(), SAMPLING_SEED,
                          200, _continuation_metas(176), rows)


# -- the successful composition, qualified without GPU work ------------------

def _complete_continuation_doc(assigned_rows):
    """Built FROM the assigned rows the caller received. A stub that invented
    its own ids would let the success seam pass even if measurement had
    substituted a different corpus."""
    rows = [_row_for(a) for a in assigned_rows]
    return {"verdict": "OK", "authoritative": True, "rows": rows,
            "failed_rows": [], "assigned": len(rows), "measured": len(rows),
            "splits": {"discovery": sum(1 for r in rows
                                        if r["split"] == "discovery"),
                       "validation": sum(1 for r in rows
                                         if r["split"] == "validation")}}


def test_combine_final_runs_is_PURE_and_recomposes_over_the_whole_corpus():
    """The SUCCESSFUL final composition, qualified on CPU at the real frozen
    N -- no ladders, no budget override, no CLI flag."""
    doc = combine_final_runs(_complete_pilot_doc(n=200),
                             _complete_continuation_doc(_assigned_176()),
                             provenance=_PROV)
    assert doc["verdict"] == "OK" and doc["authoritative"] is True
    assert len(doc["rows"]) == 200
    assert doc["pilot_rows_carried"] == 24
    assert all(r["split"] == "discovery" for r in doc["rows"][:24])
    for key in ("readout_a", "readout_b", "readout_c"):
        assert doc[key] is not None
    assert doc["readout_a_authoritative"] is True


def test_combine_inherits_incompleteness_from_EITHER_half():
    pilot, rows = _complete_pilot_doc(n=200), _assigned_176()
    doc = combine_final_runs(dict(pilot, authoritative=False),
                             _complete_continuation_doc(rows),
                             provenance=_PROV)
    assert doc["verdict"] == "ABORTED" and doc["authoritative"] is False
    cont = dict(_complete_continuation_doc(rows[:-1]), authoritative=False,
                measured=175, failed_rows=[{"game_id": 9, "failure": "seed"}])
    doc = combine_final_runs(pilot, cont, provenance=_PROV)
    assert doc["verdict"] == "ABORTED" and doc["authoritative"] is False


# -- the early widening check uses the frozen precedence ---------------------

def _row_with_no_reference_edges():
    return {"snapshots": {"at_boundary": None, "at_400": None,
                          "parent_visits": {"at_boundary": {}, "at_400": {}},
                          "reference_lines": {"merged": {
                              "required_edges": [],
                              "agreement": {d: {"state": "absent_both"}
                                            for d in ("root", "reply",
                                                      "two_ply")}}}},
            "label": "ambiguous", "phase": "late",
            "flat_policy": None, "near_even": None}


def test_INCONCLUSIVE_widening_evidence_is_not_a_failure():
    """A None retention rate means the rate is UNDEFINED, not that the shape
    failed. Closing progressive widening on an absence of evidence is exactly
    the mistake the verdict precedence exists to prevent."""
    ew = _early_widening_check([_row_with_no_reference_edges() for _ in range(3)])
    assert ew["c4a05"]["verdict"] == "INCONCLUSIVE"
    assert ew["c13a03"]["verdict"] == "INCONCLUSIVE"
    assert ew["both_fail"] is False               # NOT closed


# -- run_pilot ---------------------------------------------------------------

_PILOT_CACHE = {}


def _tiny_pilot():
    """One cached 24-row pilot at board 24 with prefix_sims=2.

    The BOARD is not reduced: it is a frozen production setting and the block
    manifest would reject anything else.
    """
    if "doc" not in _PILOT_CACHE:
        metas = _pilot_metas()
        _PILOT_CACHE["doc"] = run_pilot(
            FakeEvaluator(value=0.0), metas, sampling_seed=SAMPLING_SEED,
            base_seed=BASE,
            move_histories={m.game_id: _late_history() for m in metas},
            provenance=_PROV, **_BOARD24)
    return _PILOT_CACHE["doc"]


def test_run_pilot_sizes_from_ITS_OWN_class_counts():
    """N is not an input to the pilot -- it is the pilot's output."""
    doc = _tiny_pilot()
    assert doc["verdict"] == "OK" and doc["mode"] == "pilot"
    assert doc["sampling_seed"] == SAMPLING_SEED
    assert doc["sizing"]["verdict"] in {"OK", "PROJECTED_CAPACITY_NO_GO"}
    if doc["sizing"]["verdict"] == "OK":
        assert doc["sizing"]["N"] in ALLOWED_N
    else:
        assert doc["sizing"]["N"] is None          # None, never a default
    assert doc["splits"] == {"discovery": len(doc["rows"])}


def test_run_pilot_reports_the_early_static_widening_check():
    doc = _tiny_pilot()
    ew = doc["early_widening_check"]
    assert set(ew) == {"c4a05", "c13a03", "both_fail"}
    for shape in ("c4a05", "c13a03"):
        assert ew[shape]["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert doc["early_widening_check_authoritative"] is True


def test_an_ABORTED_pilot_does_not_size_and_does_not_close_widening():
    """Sizing an incomplete pilot would set N from a class frequency measured
    over fewer than 24 rows, and closing progressive widening on it would end a
    read-out on evidence that was never gathered."""
    metas = _pilot_metas()
    # Only ONE history is supplied, so 23 rows fail at replay_prefix's
    # n_moves check before any search runs and one row is genuinely measured.
    # Partial-but-aborted is the state under test, reached in seconds.
    doc = run_pilot(FakeEvaluator(value=0.0), metas,
                    sampling_seed=SAMPLING_SEED, base_seed=BASE,
                    move_histories={0: _late_history()},
                    provenance=_PROV, **_BOARD24)
    assert doc["verdict"] == "ABORTED" and doc["authoritative"] is False
    assert doc["sizing"]["verdict"] == "UNAVAILABLE"
    assert doc["sizing"]["N"] is None
    assert doc["early_widening_check_authoritative"] is False
    assert doc["rows"] and doc["failed_rows"]
