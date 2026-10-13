"""Stage B pure pieces (card docs/superpowers/2026-10-12-search-distillation-stage-b-card.md). CPU-only fixtures."""
import dataclasses
import math
import random

import numpy as np
import pytest

from scripts.GPU.alphazero import search_distillation_stage_b as B
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from tests.fpu_search_fixture import FakeEvaluator

MOVES = [(5, 5), (10, 3), (7, 6), (12, 8), (9, 7), (3, 14), (11, 8), (14, 10), (13, 9), (16, 11)]


def test_split_is_every_fifth_game_and_repeat_slots_cover_400_to_439():
    held = [g for g in range(200) if B.is_heldout(g)]
    assert len(held) == 40 and held[0] == 4 and held[-1] == 199
    assert [B.repeat_slot(g) for g in held] == list(range(400, 440))
    with pytest.raises(ValueError):
        B.repeat_slot(3)


def test_replay_state_carries_the_corpus_ply_cap():
    s = B.replay_state(MOVES[:3])
    assert (s.max_plies_limit, s.ply, s.active_size, s.to_move) == (280, 3, 24, "black")
    assert TwixtState.from_moves(MOVES[:3]).max_plies_limit is None      # why replay_state exists


def test_digest_covers_every_constructor_field():
    s = B.replay_state(MOVES)
    variants = [dataclasses.replace(s, to_move="red" if s.to_move == "black" else "black"),
                dataclasses.replace(s, ply=s.ply + 1),
                dataclasses.replace(s, max_plies_limit=None),
                dataclasses.replace(s, active_size=23),
                dataclasses.replace(s, board_size=25),
                dataclasses.replace(s, pegs={**s.pegs, (20, 20): "red"}),
                dataclasses.replace(s, bridges=set(list(s.bridges)[1:]))]
    assert s.bridges, "the fixture must have a bridge to remove"
    digests = {B.state_digest(v) for v in variants}
    assert len(digests) == 7 and B.state_digest(s) not in digests


def test_digest_ignores_insertion_order():
    s = B.replay_state(MOVES)
    t = dataclasses.replace(s, pegs=dict(reversed(list(s.pegs.items()))), bridges=set(reversed(list(s.bridges))))
    assert B.state_digest(t) == B.state_digest(s)


def test_digest_refuses_a_changed_field_set(monkeypatch):
    monkeypatch.setattr(B, "STATE_FIELDS", tuple(f for f in B.STATE_FIELDS if f != "bridges"))
    with pytest.raises(RuntimeError):
        B.state_digest(B.replay_state(MOVES[:2]))

def test_select_plies_draws_up_to_four_from_1_to_n_minus_1():
    for seed in range(1000, 1050):
        plies = B.select_plies(30, seed)
        assert len(plies) == 4 and plies == sorted(plies) and all(1 <= p <= 29 for p in plies)
        assert plies == sorted(random.Random(seed).sample(range(1, 30), 4))
    assert B.select_plies(3, 1000) == [1, 2]
    assert B.select_plies(1, 1000) == [] and B.select_plies(0, 1000) == []


def test_select_game_hashes_the_capped_rebuilt_state():
    picked = B.select_game(MOVES, 1007)
    assert [p["ply"] for p in picked] == B.select_plies(len(MOVES), 1007)
    for p in picked:
        assert p["digest"] == B.state_digest(B.replay_state(MOVES[:p["ply"]]))


def _selection(shared=None):
    sel = {g: [{"ply": i + 1, "digest": f"g{g}p{i}"} for i in range(4)] for g in range(200)}
    for (g, i), digest in (shared or {}).items():
        sel[g][i] = {"ply": i + 1, "digest": digest}
    return sel


def test_leakage_drops_held_out_positions_seen_in_training_and_can_empty_a_game():
    shared = {(4, 0): "g0p0"}                                   # one held-out position also in training
    shared.update({(9, i): f"g1p{i}" for i in range(4)})       # held-out game 9 entirely in training
    shared[(14, 1)] = "g19p2"                                   # duplicate of ANOTHER held-out game: kept
    kept, dropped = B.leakage_filter(_selection(shared))
    assert sorted(kept) == [g for g in range(200) if g % 5 == 4]
    assert dropped == 5
    assert [p["digest"] for p in kept[4]] == ["g4p1", "g4p2", "g4p3"]
    assert kept[9] == []                                        # an EMPTY game
    assert len(kept[14]) == 4


def test_leakage_requires_all_200_games_selected():
    sel = _selection()
    del sel[123]
    with pytest.raises(ValueError):
        B.leakage_filter(sel)

def test_full_counts_restores_every_legal_move_including_zero_visits():
    s = B.replay_state([(5, 5)])
    legal = [tuple(m) for m in s.legal_moves()]
    sha = B.legal_digest(legal)
    sparse = B.sparse_label({legal[0]: 3, legal[7]: 1, legal[9]: 0}, 4)
    assert sparse == sorted([[*legal[0], 3], [*legal[7], 1]])               # zeros are not stored
    got_legal, counts = B.full_counts(s, sparse, 4, sha)
    assert got_legal == legal and len(counts) == len(legal) == 527
    assert counts[0] == 3 and counts[7] == 1 and sum(counts) == 4 and counts.count(0) == 525


@pytest.mark.parametrize("sparse, sims, wrong_sha", [
    ([[5, 5, 4]], 4, False),                # an occupied square: not a legal move
    ([[1, 1, 2], [1, 1, 2]], 4, False),     # the same move twice
    ([[1, 1, 3]], 4, False),                # visits do not sum to sims
    ([[1, 1, 0], [1, 2, 4]], 4, False),     # a stored zero
    ([[1, 1, 4]], 4, True),                 # the rebuilt legal list is not the labelled one
])
def test_full_counts_refuses_a_label_that_does_not_fit_the_state(sparse, sims, wrong_sha):
    # (1, 1) and (1, 2) are legal for black after red (5, 5); row 0 is not, so it would refuse for the wrong reason
    s = B.replay_state([(5, 5)])
    sha = B.legal_digest([tuple(m) for m in s.legal_moves()])
    with pytest.raises(ValueError):
        B.full_counts(s, sparse, sims, ("0" if sha[0] != "0" else "1") + sha[1:] if wrong_sha else sha)


def test_sparse_label_refuses_a_visit_sum_other_than_sims():
    with pytest.raises(ValueError):
        B.sparse_label({(0, 1): 3}, 4)


def test_tv_and_kl_match_hand_values():
    assert B.tv([1, 0, 0], [0, 0, 1]) == 1.0
    assert B.tv([0.5, 0.5, 0], [0.25, 0.25, 0.5]) == 0.5
    assert math.isclose(B.kl([0.5, 0.5, 0], [0.25, 0.25, 0.5]), math.log(2), rel_tol=1e-12)


def test_kl_uses_the_full_support_a_zero_visit_move_still_takes_probability():
    t = B.probs([3, 1, 0])
    assert math.isclose(B.kl(t, B.softmax([0.0, 0.0, 0.0])), 0.5362771440493014, rel_tol=1e-12)
    assert math.isclose(B.kl(t[:2], B.softmax([0.0, 0.0])), 0.13081203594113697, rel_tol=1e-12)   # if dropped


def test_top1_breaks_ties_to_the_lowest_index_and_argmax_set_keeps_all_maxima():
    assert B.top1([1.0, 3.0, 3.0]) == 1
    assert B.argmax_set([2, 5, 5, 0]) == {1, 2}


def test_q_game_means_and_empty_game():
    pos = [{"t1600": [2, 2, 0], "t400": [1, 0, 0], "repeat": [2, 2, 0]},     # D 0.5, R 0
           {"t1600": [4, 0, 0], "t400": [4, 0, 0], "repeat": [0, 4, 0]}]     # D 0,   R 1
    assert B.q_game(pos) == (0.25, 0.5)
    assert B.q_game([]) is None


def test_game_measures_per_arm_and_empty_game():
    flat, sharp = [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]
    pos = [{"target": [0, 4, 0], "logits": {"student": sharp, "control": flat, "temperature": flat, "incumbent": flat}}]
    m = B.game_measures(pos)
    assert m["top1"] == {"student": 1.0, "control": 0.0, "temperature": 0.0, "incumbent": 0.0}
    assert math.isclose(m["kl"]["control"], math.log(3), rel_tol=1e-12)
    assert m["kl"]["student"] < m["kl"]["control"]
    assert B.game_measures([]) is None

WIN, TIE = (0.5, 0.25), (0.0, 0.0)      # TIE adds nothing to Q2's sums: the Q1 boundary is isolated (ratio 0.5)


def test_q_passes_at_26_of_40_and_fails_at_25():
    ok = B.gate_q([WIN] * 26 + [TIE] * 14)
    assert ok["q1_wins"] == 26 and ok["failed"] == [] and ok["verdict"] == "QUALIFIED"
    bad = B.gate_q([WIN] * 25 + [TIE] * 15)
    assert bad["failed"] == ["Q1"] and bad["verdict"] == "QUALIFICATION NON-PASS"


def test_q2_ratio_exactly_half_passes_and_just_above_fails():
    assert B.gate_q([WIN] * 40)["ratio"] == 0.5 and B.gate_q([WIN] * 40)["failed"] == []
    assert B.gate_q([(0.5, 0.25 + 2 ** -10)] * 40)["failed"] == ["Q2"]


def test_q_zero_total_d_and_ties_fail_both_conditions():
    r = B.gate_q([(0.0, 0.0)] * 40)
    assert r["ratio"] is None and r["failed"] == ["Q1", "Q2"]


def test_q_empty_games_fail_q1_and_are_left_out_of_q2_sums():
    r = B.gate_q([WIN] * 25 + [None] * 15)
    assert r["failed"] == ["Q1"] and r["sum_d"] == 12.5 and r["ratio"] == 0.5
    assert B.gate_q([None] * 40)["failed"] == ["Q1", "Q2"]


def test_q_requires_exactly_40_games():
    with pytest.raises(ValueError):
        B.gate_q([WIN] * 39)


def _g(s, c, t, i, top=(1.0, 0.0, 0.0, 0.0)):
    return {"kl": dict(zip(B.ARMS, (s, c, t, i))), "top1": dict(zip(B.ARMS, top))}


def test_learning_pass_at_26_and_non_pass_at_25():
    win, loss = _g(1.0, 2.0, 2.0, 2.0), _g(3.0, 2.0, 2.0, 2.0)
    assert B.learning_verdict([win] * 26 + [loss] * 14)["verdict"] == "PASS"
    r = B.learning_verdict([win] * 25 + [loss] * 15)
    assert r["verdict"] == "NON-PASS" and r["failed"] == ["L1", "L2"]


def test_l2_needs_the_student_below_both_incumbent_and_temperature():
    beats_temp_not_incumbent = _g(1.5, 2.0, 2.0, 1.0)
    r = B.learning_verdict([beats_temp_not_incumbent] * 40)
    assert r["l1_wins"] == 40 and r["l2_wins"] == 0 and r["failed"] == ["L2"]


def test_equal_kl_is_a_student_loss():
    r = B.learning_verdict([_g(2.0, 2.0, 2.0, 2.0)] * 40)
    assert r["l1_wins"] == 0 and r["l2_wins"] == 0


def test_l3_needs_strictly_higher_agreement_than_control_and_incumbent():
    tied_control = _g(1.0, 2.0, 2.0, 2.0, top=(0.5, 0.5, 0.0, 0.0))
    assert B.learning_verdict([tied_control] * 40)["failed"] == ["L3"]


def test_empty_games_lose_l1_l2_and_score_zero_top1_for_every_arm():
    win = _g(1.0, 2.0, 2.0, 2.0, top=(1.0, 1.0, 1.0, 1.0))
    r = B.learning_verdict([win] * 26 + [None] * 14)
    assert r["l1_wins"] == 26 and r["l2_wins"] == 26
    assert r["top1"] == {a: 26 / 40 for a in B.ARMS}           # empty games count 0 for EVERY arm
    assert r["failed"] == ["L3"]                                  # equal agreement is not strictly above

BASE = 1000
HELD = [g for g in range(200) if g % 5 == 4]


def ev(name, games):
    return [{"event": name, "game": g} for g in games]


def test_stop_after_position_selection_exposes_game_and_label_slots_only():
    events = ev("game_start", range(200)) + ev("select_start", range(200))
    assert B.closing_intervals(BASE, events) == ([(BASE, BASE + 400)], (BASE, BASE + 441))


def test_stop_part_way_through_selection_exposes_the_used_label_slots_not_a_prefix_of_everything():
    events = ev("game_start", range(200)) + ev("select_start", [0, 1, 2, 7])
    assert B.closing_intervals(BASE, events)[0] == [(BASE, BASE + 203), (BASE + 207, BASE + 208)]


def test_qualification_non_pass_exposes_repeats_but_not_the_training_seed():
    events = (ev("game_start", range(200)) + ev("select_start", range(200)) + ev("label_start", HELD)
              + ev("repeat_start", HELD))
    assert B.closing_intervals(BASE, events)[0] == [(BASE, BASE + 440)]


def test_full_run_exposes_the_whole_block_and_noise_events_are_ignored():
    events = (ev("game_start", range(200)) + ev("game_end", range(200)) + ev("select_start", range(200))
              + ev("label_start", range(200)) + ev("repeat_start", HELD) + [{"event": "train_start"}])
    assert B.closing_intervals(BASE, events) == ([(BASE, BASE + 441)], (BASE, BASE + 441))


def test_stop_during_corpus_exposes_started_games_only_and_retires_all():
    assert B.closing_intervals(BASE, ev("game_start", range(3))) == ([(BASE, BASE + 3)], (BASE, BASE + 441))
    assert B.closing_intervals(BASE, []) == ([], (BASE, BASE + 441))


@pytest.mark.parametrize("bad", [{"event": "repeat_start", "game": 3}, {"event": "game_start", "game": 200},
                                 {"event": "select_start", "game": -1}])
def test_exposure_refuses_an_event_outside_the_schedule(bad):
    with pytest.raises(ValueError):
        B.used_slots([bad])


# ─── checkpoint 2A: the label adapter on the REAL search (CPU fake evaluator), stable KL, soft beta ───

class CentralFakeEvaluator(FakeEvaluator):
    """Deterministic CPU fake: priors favour the centre and are mirror-symmetric, so exact ties occur; value 0."""

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        w = move_mask * np.exp(-((move_rows - 11.5) ** 2 + (move_cols - 11.5) ** 2) / 32.0)
        s = w.sum(axis=1, keepdims=True)
        priors = np.divide(w, s, out=np.zeros_like(w), where=s > 0)
        return priors.astype(np.float32), np.zeros(len(move_rows), dtype=np.float32)


SEED = 1000
EV = CentralFakeEvaluator()
STATE = B.replay_state(MOVES[:2])          # 24x24, ply cap 280, red to move


@pytest.fixture(scope="module")
def runs():
    """THE ONLY SEARCHES IN THIS FILE: 1,600 + 400 + 400 + 400 = 2,800 simulations (the plan's ceiling)."""
    m = B.label_mcts(EV, 1600, SEED)
    orig, calls, snap = m._run_single_simulation, [0], {}

    def counted(root, root_move_override=None):         # test-side instrumentation; MCTS is not modified
        orig(root, root_move_override=root_move_override)
        calls[0] += 1
        if calls[0] == 400:
            snap.update({mid: ch.visit_count for mid, ch in root.children.items() if ch.visit_count})

    m._run_single_simulation = counted
    v1600, _, _ = m.search_with_root(STATE, add_noise=False)
    v400, _, root400 = B.label_mcts(EV, 400, SEED).search_with_root(STATE, add_noise=False)

    real, seen = B.label_mcts, {}

    def capturing(evaluator, sims, seed):               # test-side: record the full output of label's own search
        mc = real(evaluator, sims, seed)
        search = mc.search_with_root

        def recorded(state, add_noise=True):
            out = search(state, add_noise=add_noise)
            seen["visits"] = out[0]
            return out

        mc.search_with_root = recorded
        return mc

    B.label_mcts = capturing
    try:
        sparse = B.label(EV, STATE, 400, SEED)
    finally:
        B.label_mcts = real
    other, _, _ = B.label_mcts(EV, 400, SEED + 1).search_with_root(STATE, add_noise=False)
    return {"v1600": v1600, "calls1600": calls[0], "snap400": snap, "v400": v400, "again": seen["visits"],
            "label": sparse, "other": other,
            "root400": {mid: ch.visit_count for mid, ch in root400.children.items() if ch.visit_count}}


def test_label_construction_is_the_frozen_call():
    m = B.label_mcts(EV, 400, SEED)
    assert (m.config.n_simulations, m.config.eval_batch_size, m.config.stall_flush_sims) == (400, 14, 48)
    assert m.rng.getstate() == random.Random(SEED ^ 0xA5A5A5).getstate()


def test_completed_visits_sum_to_sims_at_400_and_1600(runs):
    assert sum(runs["v400"].values()) == 400 and runs["calls1600"] == 1600 and sum(runs["v1600"].values()) == 1600
    assert set(runs["v400"]) == set(STATE.legal_moves())                     # every legal move, zeros included
    assert 0 in runs["v400"].values()


def test_same_seed_repeats_exactly_through_the_adapter(runs, capsys):
    differ = sum(runs["other"][m] != runs["v400"][m] for m in runs["v400"])
    with capsys.disabled():                                                  # descriptive only: no assertion
        print(f"\nDESCRIPTIVE another-seed 400: counts differ = {differ > 0} ({differ} moves differ)")
    assert runs["again"] == runs["v400"]


def test_400_search_is_the_first_400_simulations_of_the_1600_search(runs):
    assert runs["snap400"] == runs["root400"] and sum(runs["snap400"].values()) == 400


def test_label_output_rebuilds_its_own_search_exactly(runs):
    legal = [tuple(m) for m in STATE.legal_moves()]
    got_legal, counts = B.full_counts(STATE, runs["label"], 400, B.legal_digest(legal))
    assert got_legal == legal and counts == [runs["again"][m] for m in legal]


def test_kl_from_logits_is_stable_where_softmax_underflows():
    assert B.kl_logits([1.0, 0.0], [-1000.0, 0.0]) == pytest.approx(1000.0)


def test_kl_from_logits_matches_the_probability_form_on_benign_logits():
    t, z = B.probs([3, 1, 0]), [0.3, -0.2, 1.1]
    assert math.isclose(B.kl_logits(t, z), B.kl(t, B.softmax(z)), rel_tol=1e-12)


def test_game_measures_survives_underflowing_temperature_logits():
    pos = [{"target": [1, 0], "logits": {a: [-1000.0, 0.0] for a in B.ARMS}}]
    m = B.game_measures(pos)
    assert all(m["kl"][a] == pytest.approx(1000.0) for a in B.ARMS)


def test_soft_beta_hand_solution_two_moves():
    # logits [0, 1], target [0.25, 0.75]: softmax(beta z) matches the target at sigma(beta) = 0.75, beta = ln 3;
    # derivative g(beta) = sigma(beta) - 0.75 (= sum p z - sum T z), so g(0) = -0.25 exactly.
    logits, targets = [[0.0, 1.0]], [[0.25, 0.75]]
    assert B.soft_beta_derivative(logits, targets, 0.0) == -0.25
    fit = B.fit_beta_soft(logits, targets)
    assert abs(fit["beta"] - math.log(3)) < 1e-6 and abs(fit["derivative"]) < 1e-6
    assert B.soft_beta_derivative(logits, targets, math.log(3) - 0.1) < 0 < B.soft_beta_derivative(logits, targets, math.log(3) + 0.1)


def test_soft_beta_accepts_the_zero_endpoint_for_a_uniform_target():
    fit = B.fit_beta_soft([[0.0, 1.0]], [[0.5, 0.5]])
    assert fit["beta"] == 0.0 and fit["bracket"] == [0.0, 0.0]


def test_soft_beta_refuses_a_target_it_can_never_reach():
    with pytest.raises(ValueError):
        B.fit_beta_soft([[0.0, 1.0]], [[0.0, 1.0]])     # CE still decreasing at beta = 100: not a minimum
