"""Focused CPU checks for the mirror-averaging + root-halving combination's shared code
(`scripts/GPU/alphazero/mirror_halving_combination.py`; card
`docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` §2, §4, §5 B).

The prototypes run for real (MirrorAveragingEvaluator, RootSequentialHalvingMCTS, MCTS, the copied game loop) over
GPU-free fakes: no model, no MLX device, no seed accounting. The search RNG is a fresh random.Random(s) per search with
s from the test-only band, the two arms of every comparison sharing it."""
import inspect
import math
import pathlib
import random
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import eval_runner as ER
from scripts.GPU.alphazero import mirror_halving_combination as K
from scripts.GPU.alphazero import root_sequential_halving as RSH
from scripts.GPU.alphazero.external_learning_stage3 import PROTOCOL
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.mcts import MCTS, encode_move
from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ER.EvalConfig(**PROTOCOL)
SEED = REF.TEST_ONLY_SEED_INTERVALS[0][0]


class Fake:
    """30 zero channels (what the mirror needs); priors proportional to 1 + col, NOT mirror-symmetric; value 0.1."""

    def __init__(self):
        self.calls, self.network = 0, object()

    def build_input_tensor(self, state):
        return np.zeros((30, state.active_size, state.active_size), dtype=np.float32)

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        self.calls += 1
        w = (1.0 + move_cols.astype(np.float32)) * move_mask
        return (w / w.sum(axis=1, keepdims=True)).astype(np.float32), np.full(boards.shape[0], 0.1, np.float32)


def played(plies):
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for i in range(plies):
        moves = s.legal_moves()
        s = s.apply_move(moves[(i * 97) % len(moves)])
    assert not s.is_terminal()
    return s


def test_THE_CHECK_SEED_IS_IN_THE_TEST_ONLY_BAND():
    assert REF.seed_is_test_only(SEED)


# ─────────────────────────────── the candidate, frozen ───────────────────────────────

def test_THE_LOOP_IS_THE_ROOT_HALVING_MATCH_S_UNCHANGED():
    src = (ROOT / "docs/superpowers/evidence/root-halving-match/01_match.py").read_text()
    for f in (K.play_game, K.play_and_build):
        assert inspect.getsource(f) in src, f.__name__


def test_THE_CANDIDATE_AND_ITS_OPPONENT_ARE_THE_CARD_S():
    assert (K.CANDIDATE_SIMS, K.INCUMBENT_SIMS) == (200, 400)
    assert K.GUARDS == {"evaluations": 1.05, "search_seconds": 1.10}
    assert K.SCREEN == {"pairs": 32, "stop_below": 0.40}
    assert K.STRENGTH == {"pairs": 509, "alpha": 0.05, "null": 0.5}
    assert K.check_variant(RSH) == K.VARIANT == {"FROM_PLY": 20, "CANDIDATES": 16, "C_VISIT": 50.0, "C_SCALE": 1.0,
                                                 "PRIOR_FLOOR": 1e-12, "BOARD": 24}
    base = Fake()
    c = K.candidate_search(base, CFG, random.Random(SEED))
    assert type(c) is RSH.RootSequentialHalvingMCTS and type(c.evaluator) is MirrorAveragingEvaluator
    assert c.evaluator.base is base and c.config.n_simulations == 200
    i = K.incumbent_search(base, CFG, random.Random(SEED))
    assert type(i) is MCTS and i.evaluator is base and i.config.n_simulations == 400


def test_A_CHANGED_HALVING_CONSTANT_IS_REFUSED(monkeypatch):
    monkeypatch.setattr(RSH, "CANDIDATES", 8)
    with pytest.raises(ValueError, match="not the card's"):
        K.check_variant(RSH)


def test_HALVING_S_FROZEN_ALLOCATION_AT_200_SIMULATIONS():
    """3 / 6 / 12 / 28 more visits per round: 8 moves end at 3, 4 at 9, 2 at 21 and the final 2 at 49."""
    m, counts, _v, root, _s = K.timed_search(lambda r: K.candidate_search(Fake(), CFG, r), played(25), SEED)
    assert root.visit_count == 200 and [len(x) for x in m.last_rounds] == [16, 8, 4, 2]
    assert sorted(v for v in counts.values() if v) == [3] * 8 + [9] * 4 + [21] * 2 + [49] * 2
    assert m.last_choice in [tuple(x) for x in m.last_rounds[-1]]


# ─────────────────────────────── composition and the identity controls ───────────────────────────────

class Spy(MirrorAveragingEvaluator):
    def __init__(self, base):
        super().__init__(base)
        self.outputs = []

    def infer(self, *a):
        out = super().infer(*a)
        self.outputs.append(out)
        return out


def test_HALVING_RECEIVES_THE_WRAPPER_S_AVERAGED_PRIORS_AND_VALUES():
    base, s = Fake(), played(25)
    spy = Spy(base)
    m = RSH.RootSequentialHalvingMCTS(spy, ER.cfg_from(CFG, 200), random.Random(SEED))
    _counts, _v, root = m.search_with_root(s, add_noise=False)
    priors, values = spy.outputs[0]                          # the root's expansion, batch 1
    legal = s.legal_moves()
    assert root.priors_raw == {encode_move(r, c): float(priors[0, j]) for j, (r, c) in enumerate(legal)}
    assert root.nn_value == float(values[0])
    plain = Fake().infer(np.zeros((1, 24, 24, 30), np.float32), np.array([[r for r, _ in legal]], np.int32),
                         np.array([[c for _, c in legal]], np.int32), np.ones((1, len(legal)), np.float32), 24)[0]
    assert not np.allclose(priors[0, :len(legal)], plain[0]), "the fake must make averaging change the priors"
    assert base.calls == 2 * len(spy.outputs)                # two underlying calls per averaged call


def test_BEFORE_PLY_20_THE_CANDIDATE_EQUALS_AVERAGING_ALONE():
    base, s = Fake(), played(10)
    a = K.timed_search(lambda r: K.candidate_search(base, CFG, r), s, SEED)
    b = K.timed_search(lambda r: MCTS(MirrorAveragingEvaluator(base), ER.cfg_from(CFG, 200), r), s, SEED)
    assert a[1] == b[1] and a[2] == b[2] and a[0].last_choice is None


def test_FROM_PLY_20_HALVING_OVER_A_PASS_THROUGH_EQUALS_PLAIN_HALVING():
    base, s = Fake(), played(25)
    a = K.timed_search(lambda r: RSH.RootSequentialHalvingMCTS(K.PassThroughEvaluator(base), ER.cfg_from(CFG, 200),
                                                              r), s, SEED)
    b = K.timed_search(lambda r: RSH.RootSequentialHalvingMCTS(base, ER.cfg_from(CFG, 200), r), s, SEED)
    assert a[1] == b[1] and a[2] == b[2] and a[0].last_choice == b[0].last_choice is not None


def test_THE_PASS_THROUGH_DELEGATES_EVERY_CALL():
    base = Fake()
    p = K.PassThroughEvaluator(base)
    s = played(3)
    assert p.network is base.network and np.array_equal(p.build_input_tensor(s), base.build_input_tensor(s))
    args = (np.zeros((1, 24, 24, 30), np.float32), np.array([[0, 1]], np.int32), np.array([[2, 3]], np.int32),
            np.ones((1, 2), np.float32), 24)
    assert all(np.array_equal(x, y) for x, y in zip(p.infer(*args), base.infer(*args)))


# ─────────────────────────────── a controlled last_choice case ───────────────────────────────

def test_CONTROLLED_A_LAST_CHOICE_THAT_DIFFERS_FROM_THE_READOUT_IS_THE_MOVE_PLAYED():
    """A search whose visit leader is what the seeded readout picks, while last_choice names a different, less-visited
    move. The copied loop must play and record last_choice; with no last_choice it plays the readout's move."""
    cfg = ER.EvalConfig(**{**PROTOCOL, "max_moves": 1, "mcts_sims": 16})
    rd = ER.readout_from_eval_config(cfg)
    seen = []

    class Forced(MCTS):
        def search_with_root(self, state, add_noise=True):
            counts, value, root = super().search_with_root(state, add_noise=add_noise)
            readout = K.readout_choice(counts, root, state.ply, rd, random.Random(SEED ^ 0xC3C3C3))  # red's stream
            self.last_choice = next(m for m in state.legal_moves() if counts[m] < counts[readout])
            seen.append((readout, self.last_choice))
            return counts, value, root

    _w, _r, plies, records = K.play_game(Fake(), Fake(), cfg, SEED, capture=True, red_cls=Forced, black_cls=Forced)
    (readout, choice), = seen
    assert plies == 1 and choice != readout
    assert (records[0]["row"], records[0]["col"]) == choice and records[0]["readout_overrode_leader"] is False
    _w, _r, _p, plain = K.play_game(Fake(), Fake(), cfg, SEED, capture=True)
    assert (plain[0]["row"], plain[0]["col"]) == readout     # the readout really would have played the other move


# ─────────────────────────────── cost accounting (card §4) ───────────────────────────────

def test_INSTRUMENT_COUNTS_UNDERLYING_POSITIONS_AND_PASSES_OUTPUTS_THROUGH():
    base, cost = Fake(), K.new_cost()
    expected = base.infer(np.zeros((3, 24, 24, 30), np.float32), np.zeros((3, 2), np.int32),
                          np.ones((3, 2), np.int32), np.ones((3, 2), np.float32), 24)
    K.instrument(base, cost)
    got = MirrorAveragingEvaluator(base).infer(np.zeros((3, 24, 24, 30), np.float32), np.zeros((3, 2), np.int32),
                                               np.ones((3, 2), np.int32), np.ones((3, 2), np.float32), 24)
    assert cost["network_infer_calls"] == 2 and cost["network_positions"] == 6 and cost["network_seconds"] > 0
    assert got[1].tolist() == pytest.approx(expected[1].tolist())


def test_GUARDS_ARE_RATIOS_OF_TOTALS_NOT_MEANS_OF_RATIOS():
    # per-search ratios 2.0 and 0.5 (mean 1.25), but the totals give 1100 / 1100 = 1.0
    cand = {"network_positions": 1000 + 100, "search_seconds": 10.0}
    inc = {"network_positions": 500 + 600, "search_seconds": 10.0}
    g = K.guards(cand, inc)
    assert g["evaluations_ratio"] == 1.0 and g["evaluations_within"] and g["search_seconds_within"]


@pytest.mark.parametrize("ev, sec, ok", [(1.05, 1.10, (True, True)), (1.0501, 1.0, (False, True)),
                                         (1.0, 1.1001, (True, False))])
def test_THE_GUARDS_ARE_INCLUSIVE_AT_1_05_AND_1_10(ev, sec, ok):
    g = K.guards({"network_positions": ev * 10000, "search_seconds": sec * 100}, {"network_positions": 10000,
                                                                                 "search_seconds": 100})
    assert (g["evaluations_within"], g["search_seconds_within"]) == ok


def test_AN_EMPTY_INCUMBENT_TOTAL_IS_AN_ERROR_NOT_A_PASS():
    with pytest.raises(ValueError):
        K.guards({"network_positions": 1, "search_seconds": 1}, {"network_positions": 0, "search_seconds": 1})


# ─────────────────────────────── scores and verdicts (card §5) ───────────────────────────────

def _result(g, cand_red, cand_score):
    red = cand_score if cand_red else 1 - cand_score
    return SimpleNamespace(game_idx=g, red_agent_id=K.CANDIDATE if cand_red else K.INCUMBENT, red_score=red,
                           black_score=1 - red)


def test_PAIR_SCORES_FOLLOW_THE_CANDIDATE_ACROSS_THE_COLOUR_SWAP():
    rs = [_result(0, True, 1.0), _result(1, False, 0.5), _result(2, True, 0.0), _result(3, False, 1.0)]
    assert K.pair_scores(rs, 2) == [0.75, 0.5]
    with pytest.raises(ValueError, match="incomplete"):
        K.pair_scores(rs[:3], 2)


def test_THE_SCREEN_STOPS_ONLY_BELOW_0_40():
    assert K.screen_verdict([0.40] * 32)["verdict"] == "CONTINUE"
    assert K.screen_verdict([0.375] * 32)["verdict"] == "STOP"
    with pytest.raises(ValueError):
        K.screen_verdict([0.5] * 31)


def test_THE_STRENGTH_VERDICT_USES_THE_EXACT_HOEFFDING_FORMULA():
    pairs = [1.0] * 290 + [0.0] * 219                        # mean 0.5697...
    expected = math.fsum(pairs) / 509 - math.sqrt(math.log(2 / 0.05) / (2 * 509))
    assert K.hoeffding_lower(pairs, 0.05) == expected
    ok = {"evaluations_within": True, "search_seconds_within": True}
    v = K.strength_verdict(pairs, ok)
    assert v["hoeffding_lower"] == expected and v["verdict"] == "PASS"
    assert K.strength_verdict(pairs, {**ok, "search_seconds_within": False})["verdict"] == "NOT_COMPUTE_MATCHED"
    with pytest.raises(ValueError):
        K.strength_verdict(pairs[:508], ok)


def test_THE_THRESHOLD_IS_THE_FORMULA_NOT_THE_ROUNDED_0_56020():
    """Just above and just below the exact threshold: the rounded figure must not decide."""
    half = math.sqrt(math.log(40) / (2 * 509))
    ok = {"evaluations_within": True, "search_seconds_within": True}
    for delta, want in ((+1e-9, "PASS"), (-1e-9, "NON_PASS")):
        mean = 0.5 + half + delta
        pairs = [mean] * 509
        assert K.strength_verdict(pairs, ok)["verdict"] == want, (mean, want)
    assert abs((0.5 + half) - 0.56020) < 1e-5
