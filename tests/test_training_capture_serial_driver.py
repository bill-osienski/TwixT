"""The minimal serial capture driver (training_capture_serial.py): REAL MCTS.search_from_root inside the real
play_game, driven by the repo's deterministic CPU FakeEvaluator (uniform priors, value 0; no network inference).
A small random-init AlphaZeroNetwork exists only so its parameters can be hashed -- it never evaluates a position.

Checks: capture changes no move, learner row or RNG draw (same seeds, capture off vs on); the reopened targets are
the learner rows' own visit counts and outcomes; the model digest covers BN running statistics and a model change
withholds the completion marker; unsupported setups refuse before anything is written. Synthetic games only: no
claim about target quality, throughput or strength."""
import random

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import self_play as sp  # noqa: E402
from scripts.GPU.alphazero import training_capture_serial as S  # noqa: E402
from scripts.GPU.alphazero import training_replay_capture as C  # noqa: E402
from scripts.GPU.alphazero.mcts import MCTSConfig  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402
from tests.fpu_search_fixture import FakeEvaluator  # noqa: E402

SEEDS = {0: 11, 1: 12}
CFG = MCTSConfig(n_simulations=8)
DIAG_OFF = {"goal_completion_emit_enabled": False, "goal_completion_record_enabled": False}   # both sides alike


def _net():
    mx.random.seed(0)
    net = AlphaZeroNetwork()
    net.policy_head.bn.running_mean = mx.array([0.3, -0.2])
    net.eval()
    return net


def _evaluator(net):
    ev = FakeEvaluator()
    ev.network = net
    return ev


def _context(net, seeds=SEEDS, cfg=CFG, digest=None):
    return S.build_context(run_id="test:run", phase_id="test:serial", seeds=seeds, model_id="test:model",
                           model_digest=digest or S.network_state_digest(net),
                           model_source={"kind": "random-init test network"}, mcts_config=cfg,
                           code={"fixture": "not a source closure"}, phase_bytes=64 << 20, play_kwargs=DIAG_OFF)


def _signature(game):
    return (list(game.move_history), game.winner, game.draw_reason,
            [(p.ply, p.to_move, list(p.legal_moves), list(p.visit_counts), p.outcome, p.board_tensor.tobytes())
             for p in game.positions])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The same two seeded games, capture OFF (play_game as train()'s serial loop calls it) and ON (the driver)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(sp, "_MIRROR_PROB", 0.0)
    net = _net()
    off, off_rng = [], []
    for gid, seed in SEEDS.items():
        rng = random.Random(seed)
        start = "red" if rng.random() < 0.5 else "black"
        off.append(sp.play_game(_evaluator(net), mcts_config=CFG, rng=rng, max_moves=280, add_noise=True,
                                active_size=24, start_player=start, game_id=gid, **DIAG_OFF))
        off_rng.append(rng.getstate())
    on_rng = []

    def spy(evaluator, *, rng, **kw):
        game = sp.play_game(evaluator, rng=rng, **kw)
        on_rng.append(rng.getstate())
        return game

    d = tmp_path_factory.mktemp("serial") / "cap"
    out = S.run_serial_capture(evaluator=_evaluator(net), network=net, capture_dir=str(d), context=_context(net),
                               mcts_config=CFG, play_game=spy)
    mp.undo()
    return {"net": net, "off": off, "off_rng": off_rng, "on": out["games"], "on_rng": on_rng, "dir": d,
            "stats": out["stats"]}


def test_CAPTURE_CHANGES_NO_MOVE_LEARNER_ROW_OR_RNG_DRAW(runs):
    assert [_signature(g) for g in runs["on"]] == [_signature(g) for g in runs["off"]]
    assert runs["on_rng"] == runs["off_rng"]
    assert all(len(g.positions) == g.n_moves > 0 for g in runs["off"])
    print("serial-driver-games", [(g.n_moves, g.winner, g.draw_reason) for g in runs["off"]], runs["stats"])


def test_THE_REOPENED_TARGETS_ARE_THE_LEARNER_ROWS(runs):
    ctx, envs = C.load_phase(str(runs["dir"]))
    assert ctx["models"]["test:model"]["state_digest"] == S.network_state_digest(runs["net"])
    for gid, env, game in zip(SEEDS, envs, runs["on"]):
        rng = random.Random(SEEDS[gid])
        rng.random()
        assert env["rng_at_entry"] == C.encode_rng_state(rng)                  # the seed, after train()'s draw
        assert env["start_player"] == ctx["games"][str(gid)]["start_player"]
        assert [tuple(m) for m in env["move_history"]] == list(game.move_history)
        assert len(env["rows"]) == game.n_moves == len(game.positions)
        for p in game.positions:
            row = env["rows"][p.ply]
            assert (row["to_move"], row["model_id"], row["learner_side"]) == (p.to_move, "test:model", True)
            assert [tuple(m) for m in row["legal_moves"]] == list(p.legal_moves)
            assert row["child_visits_after"] == list(p.visit_counts)            # the training target itself
            assert row["selection_counts"] == row["child_visits_after"] and row["selection_adjustment"] is None
            assert row["z"] == p.outcome
            assert row["root_visits_after"] - row["root_visits_before"] == CFG.n_simulations
        assert [m["source_row_id"] for m in env["buffer_membership"]] == [f"{gid}:{p.ply}" for p in game.positions]
        # Tree reuse carries the played child's own count into the next root. At 8 simulations over ~528 uniform
        # priors that child has no visited children, so inherited CHILD mass is not exercised here (the real-search
        # probe covers it); root count != child sum is.
        assert any(r["root_visits_before"] > sum(r["child_visits_before"]) for r in env["rows"])
        assert env["outcome"]["winner"] == game.winner and env["outcome"]["final_ply"] == game.n_moves


def test_THE_MODEL_DIGEST_COVERS_BN_STATISTICS_and_a_changed_model_withholds_the_marker(tmp_path, monkeypatch):
    net = _net()
    before = S.network_state_digest(net)
    net.policy_head.bn.running_var = net.policy_head.bn.running_var + 1e-3      # a BN buffer only
    assert S.network_state_digest(net) != before
    net = _net()
    monkeypatch.setattr(sp, "_MIRROR_PROB", 0.0)

    def drifting(evaluator, **kw):
        game = sp.play_game(evaluator, **kw)
        evaluator.network.policy_head.bn.running_mean = mx.array([0.31, -0.2])   # the model changes mid-phase
        return game

    d = tmp_path / "cap"
    with pytest.raises(C.CaptureError, match="no longer matches"):
        S.run_serial_capture(evaluator=_evaluator(net), network=net, capture_dir=str(d),
                             context=_context(_net(), seeds={0: 11}), mcts_config=CFG, play_game=drifting)
    assert (d / "games" / "game_0.json").exists() and not (d / "finalized.json").exists()
    with pytest.raises(C.CaptureError, match="missing"):
        C.load_phase(str(d))


REFUSALS = {"other_network": "does not serve", "mirror_on": "mirror augmentation is on",
            "wrong_digest": "does not match the context's model digest", "sims_mismatch": "n_simulations",
            "dir_exists": "already exists", "play_kwarg": "play_kwargs may only",
            "c_puct_mismatch": "recorded MCTS configuration", "recorded_mcts_edit": "recorded MCTS configuration",
            "recorded_mirror": "noise, mirror or position-cap", "td1_forcing": "forced visits and the closeout",
            "tiebreak": "forced visits and the closeout"}     # each case must hit ITS refusal, not a neighbour's


@pytest.mark.parametrize("case", ["other_network", "mirror_on", "wrong_digest", "sims_mismatch", "dir_exists",
                                  "play_kwarg", "c_puct_mismatch", "recorded_mcts_edit", "recorded_mirror",
                                  "td1_forcing", "tiebreak"])
def test_UNSUPPORTED_SETUPS_REFUSE_BEFORE_ANYTHING_IS_WRITTEN(tmp_path, monkeypatch, case):
    net = _net()
    monkeypatch.setattr(sp, "_MIRROR_PROB", 0.5 if case == "mirror_on" else 0.0)
    ev, ctx, cfg = _evaluator(net), _context(net), CFG
    if case == "other_network":
        ev = _evaluator(_net())
    if case == "wrong_digest":
        ctx = _context(net, digest="0" * 64)
    if case == "sims_mismatch":
        cfg = MCTSConfig(n_simulations=9)
    if case == "play_kwarg":
        ctx["configuration"]["play_kwargs"] = {"resign_enabled": False}
    if case == "c_puct_mismatch":                       # same N, different search: the targets would be misattributed
        cfg = MCTSConfig(n_simulations=8, c_puct=2.0)
    if case == "recorded_mcts_edit":
        ctx["configuration"]["mcts"]["dirichlet_eps"] = 0.0
    if case == "recorded_mirror":
        ctx["configuration"]["mirror_prob"] = 0.5
    if case in ("td1_forcing", "tiebreak"):             # recorded consistently, still refused before the context
        cfg = MCTSConfig(n_simulations=8, **{("closeout_td1_visit_forcing_enabled" if case == "td1_forcing"
                                               else "closeout_selection_tiebreak_enabled"): True})
        ctx = _context(net, cfg=cfg)
    d = tmp_path / "cap"
    if case == "dir_exists":
        d.mkdir()
    calls = []
    with pytest.raises(C.CaptureError, match=REFUSALS[case]):
        S.run_serial_capture(evaluator=ev, network=net, capture_dir=str(d), context=ctx, mcts_config=cfg,
                             play_game=lambda *a, **k: calls.append(1))
    assert calls == [] and (case == "dir_exists" or not d.exists())
    assert case != "dir_exists" or list(d.iterdir()) == []
