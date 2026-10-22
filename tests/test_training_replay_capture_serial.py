"""Serial-wiring checkpoint: play_game driven by a SCRIPTED agent over the fixed 25-ply history. The evaluator
is a tripwire; no search, model, MLX, worker or generated history. Capture off vs on, and reopened targets vs
the observations made at the capture boundaries."""
import ast
import inspect
import random

import pytest

from scripts.GPU.alphazero import self_play as sp
from scripts.GPU.alphazero import training_replay_capture as C
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, decode_move, encode_move
from tests.test_training_replay_capture import HISTORY, _context, _scripted, write_phase


class _TripwireEvaluator:
    def build_input_tensor(self, state):
        raise AssertionError("the evaluator was touched")

    def infer(self, *a, **k):
        raise AssertionError("the evaluator was touched")


def _agent_class(log):
    class ScriptedMCTS(MCTS):
        """Replays the first checkpoint's scripted counts; records what each boundary saw."""

        def search_from_root(self, root, add_noise=True, ply=0, gc_state_full=None):
            s = root.state
            legal = s.legal_moves()
            _before, after, _alt = _scripted(ply, legal, HISTORY[ply])
            obs = {"ply": ply, "root_before": root.visit_count,
                   "children_before": {decode_move(k): c.visit_count for k, c in root.children.items()}}
            for m, c in after.items():
                mid = encode_move(*m)
                if mid not in root.children:
                    root.children[mid] = MCTSNode(state=s.apply_move(m))
                root.children[mid].visit_count = c
            root.visit_count += 4
            counts = {m: (root.children[encode_move(*m)].visit_count if encode_move(*m) in root.children else 0)
                      for m in legal}
            value = 0.1 * (ply % 5)
            obs.update(counts=dict(counts), root_after=root.visit_count, value=value)
            log.append(obs)
            return counts, value, root

        def select_move(self, visit_counts, ply):
            log[-1]["selection"] = dict(visit_counts)
            self.rng.random()                          # stands in for temperature sampling's single draw
            return HISTORY[ply]

        def advance_root(self, root, move):
            nxt = root.state.apply_move(move)
            node = MCTSNode(state=nxt)
            if nxt.ply < len(HISTORY) and not nxt.is_terminal():
                before, _after, _alt = _scripted(nxt.ply, nxt.legal_moves(), HISTORY[nxt.ply])
                for m, c in before.items():                 # inherited visits, as tree reuse would carry
                    node.children[encode_move(*m)] = MCTSNode(state=nxt.apply_move(m), visit_count=c)
                node.visit_count = sum(before.values())
            return node

    return ScriptedMCTS


def _selfplay_context():
    ctx = _context()
    ctx["games"]["0"].update(learner_player=None,
                             models_by_colour={"red": "fixture:model-A", "black": "fixture:model-A"})
    return ctx, C.sha256_hex(C.canonical_bytes(ctx))


def _play(monkeypatch, rng, capture=None, log=None, **kw):
    log = [] if log is None else log
    monkeypatch.setattr(sp, "MCTS", _agent_class(log))
    monkeypatch.setattr(sp, "_MIRROR_PROB", kw.pop("mirror_prob", 0.0))
    args = dict(mcts_config=MCTSConfig(n_simulations=4), rng=rng, max_moves=280, active_size=24,
                start_player="red", goal_completion_emit_enabled=False, goal_completion_record_enabled=False,
                training_capture=capture)
    args.update(kw)
    return sp.play_game(_TripwireEvaluator(), **args), log


def _signature(game):
    return (list(game.move_history), game.winner,
            [(p.ply, p.to_move, list(p.legal_moves), list(p.visit_counts), p.outcome, p.board_tensor.tobytes())
             for p in game.positions])


def test_capture_off_vs_on_changes_no_learner_row_move_or_rng(monkeypatch):
    rng_off = random.Random(7)
    off, log_off = _play(monkeypatch, rng_off)
    ctx, sha = _selfplay_context()
    rng_on = random.Random(7)
    cap = C.DecisionCapture(context=ctx, context_sha256=sha, game_id=0, game_rng=rng_on)
    on, log_on = _play(monkeypatch, rng_on, capture=cap)
    assert list(off.move_history) == HISTORY and off.winner == "red"
    assert _signature(off) == _signature(on)
    assert rng_off.getstate() == rng_on.getstate()
    assert log_off == log_on


def test_reopened_targets_equal_the_boundary_observations(monkeypatch, tmp_path):
    ctx, sha = _selfplay_context()
    rng = random.Random(7)
    cap = C.DecisionCapture(context=ctx, context_sha256=sha, game_id=0, game_rng=rng)
    game, log = _play(monkeypatch, rng, capture=cap)
    stats = write_phase(str(tmp_path / "cap"), ctx, [cap.envelope])
    _, (env,) = C.load_phase(str(tmp_path / "cap"))
    assert len(env["rows"]) == len(log) == len(HISTORY)
    for row, obs in zip(env["rows"], log):
        legal = [tuple(m) for m in row["legal_moves"]]
        assert legal == list(obs["counts"]), obs["ply"]                    # the search's own legal order
        assert row["child_visits_before"] == [obs["children_before"].get(m, 0) for m in legal], obs["ply"]
        assert row["child_visits_after"] == [obs["counts"][m] for m in legal], obs["ply"]
        assert row["selection_counts"] == [obs["selection"][m] for m in legal], obs["ply"]
        assert (row["root_visits_before"], row["root_visits_after"], row["root_value"]) == \
            (obs["root_before"], obs["root_after"], obs["value"]), obs["ply"]
        assert row["selected_move"] == list(HISTORY[obs["ply"]])
    assert [m["source_row_id"] for m in env["buffer_membership"]] == [f"0:{p.ply}" for p in game.positions]
    assert env["outcome"]["winner"] == game.winner == "red"
    print("serial-checkpoint-measurements", stats)


@pytest.mark.parametrize("bad", [
    dict(max_moves=200), dict(resign_enabled=True), dict(adjudicate_enabled=True), dict(mirror_prob=0.5),
    dict(max_positions_per_game=10), dict(learner_player="red"),
    dict(active_size=16),                                                   # refused before any board exists
    dict(mcts_config=MCTSConfig(n_simulations=4, closeout_td1_visit_forcing_enabled=True)),
    dict(mcts_config=MCTSConfig(n_simulations=4, closeout_selection_tiebreak_enabled=True)),
], ids=["max_moves", "resign", "adjudicate", "mirror", "position_cap", "learner_mismatch", "active_size",
        "forced_visits", "closeout_tiebreak"])
def test_unsupported_settings_refuse_before_any_search(monkeypatch, bad):
    ctx, sha = _selfplay_context()
    rng = random.Random(7)
    cap = C.DecisionCapture(context=ctx, context_sha256=sha, game_id=0, game_rng=rng)
    before, log = rng.getstate(), []
    with pytest.raises(C.CaptureError, match="unsupported"):
        _play(monkeypatch, rng, capture=cap, log=log, **bad)
    assert rng.getstate() == before and log == [] and cap.rows == []


def test_hooks_are_guarded_last_and_at_the_boundaries():
    params = list(inspect.signature(sp.play_game).parameters.values())
    assert params[-1].name == "training_capture" and params[-1].default is None
    tree = ast.parse(inspect.getsource(sp.play_game))
    hooks = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "training_capture"]
    guarded = {id(c) for g in ast.walk(tree) if isinstance(g, ast.If)
               and "training_capture is not None" in ast.unparse(g.test) for c in ast.walk(g)}
    assert hooks and all(id(h) in guarded for h in hooks)
    line = lambda name: min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                            and ast.unparse(n.func).endswith(name))
    hook = lambda name: min(h.lineno for h in hooks if h.func.attr == name)
    assert line("CaptureError") < line("MCTS")
    assert hook("begin_decision") < line("search_from_root") < hook("after_search")
    assert hook("before_selection") < line("select_move") < hook("after_selection") < line("advance_root")
    assert line("apply_game_position_cap") < hook("finish")
