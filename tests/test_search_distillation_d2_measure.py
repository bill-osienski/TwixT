"""Search distillation D2 Stage 1 (search_distillation_d2_measure.py): the record boundary and its stops, through
the TEST seams with real MCTS, the repo's CPU FakeEvaluator and a small random network that is only hashed (8
simulations, not the card's 1,600). The public entry is checked with its gate closed and with a wrong card; it is
never run for real here."""
import json
import os
import signal

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import self_play as sp  # noqa: E402
from scripts.GPU.alphazero import training_replay_capture as C  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402
from tests.fpu_search_fixture import FakeEvaluator  # noqa: E402

SPEC = {"games": 2, "n_simulations": 8, "phase_bytes": 64 << 20}


def _net():
    mx.random.seed(0)
    net = AlphaZeroNetwork(hidden=16, n_blocks=1)
    net.policy_head.bn.running_mean = mx.array([0.3, -0.2])
    return net                                               # left in TRAINING mode: the run must call eval()


def _evaluator(net):
    ev = FakeEvaluator()
    ev.network = net
    return ev


def _seams(**over):
    base = dict(read_plan=lambda: {"base": 11, "n_slots": 2}, seed_ok=lambda s: True, mirror_prob=lambda: 0.0,
                checkpoint_sha1=lambda: M.CHECKPOINT_SHA1, load_network=_net, make_evaluator=_evaluator,
                loaded_modules=lambda: {"fixture": "not a closure"}, play_game=sp.play_game)
    base.update(over)
    return M.Seams(**base)


@pytest.fixture(autouse=True)
def _mirror_off(monkeypatch):
    monkeypatch.setattr(sp, "_MIRROR_PROB", 0.0)


def _progress(out):
    return [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]


def test_A_COMPLETED_RUN_records_both_games_the_capture_and_the_stage0_reopen(tmp_path):
    out = tmp_path / "m"
    code, rec = M.conclude(SPEC, _seams(), out, head="f" * 40)
    assert code == M.EXIT_DONE and rec["status"] == "COMPLETED", rec.get("void")
    assert json.loads((out / "record.json").read_text()) == json.loads(json.dumps(rec))
    assert [g["seed"] for g in rec["games"]] == [11, 12] and all(g["plies"] > 0 for g in rec["games"])
    assert rec["model_digest_before"] == rec["model_digest_after"]
    assert rec["modes_at_load"]["modules_in_training_mode"] == 0 == rec["modes_at_end"]["modules_in_training_mode"]
    assert rec["stage0_reader"]["records"] == rec["totals"]["plies"] == sum(g["plies"] for g in rec["games"])
    assert rec["stage0_reader"]["aligned_with_learner_rows"] is True and rec["stage0_reader"]["rows"] == rec["totals"]["plies"]
    assert rec["configuration"]["n_simulations"] == 8 and rec["configuration"]["mirror_prob"] == 0.0
    assert rec["configuration"]["play_kwargs"] == M.DIAGNOSTICS_OFF and rec["head"] == "f" * 40
    ctx, envs = C.load_phase(str(out / "capture"))
    assert ctx["models"][M.MODEL_ID]["state_digest"] == rec["model_digest_before"] and len(envs) == 2
    events = [e["event"] for e in _progress(out)]
    assert events == ["start", "loaded", "game_start", "game_end", "game_start", "game_end", "end"]


def test_THE_FROZEN_SPEC_IS_THE_CARD_S():
    assert M.FROZEN == {"games": 2, "n_simulations": 1600, "phase_bytes": 20 << 20}
    cfg = M.mcts_config_for(M.FROZEN)
    assert (cfg.n_simulations, cfg.eval_batch_size, cfg.pending_virtual_visits, cfg.stall_flush_sims) == \
           (1600, 14, 8, 16)
    assert not cfg.closeout_td1_visit_forcing_enabled and not cfg.closeout_selection_tiebreak_enabled


def test_SIGTERM_MID_GAME_is_a_VOID_with_a_record_and_no_completion_marker(tmp_path):
    def play(evaluator, **kw):
        if kw["game_id"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return sp.play_game(evaluator, **kw)

    out = tmp_path / "m"
    previous = signal.getsignal(signal.SIGTERM)
    code, rec = M.conclude(SPEC, _seams(play_game=play), out)
    assert signal.getsignal(signal.SIGTERM) == previous                  # the handler is restored on the way out
    assert code == M.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == "terminated"
    assert json.loads((out / "record.json").read_text())["void"]["step"] == "terminated"
    assert (out / "capture" / "games" / "game_0.json").exists() and not (out / "capture" / "finalized.json").exists()
    assert len(rec["games"]) == 1 and _progress(out)[-1] == {"event": "end", "status": "VOID",
                                                             "t": _progress(out)[-1]["t"]}


@pytest.mark.parametrize("case, step", [("seeds", "seeds"), ("plan", "seeds"), ("mirror", "mirror"),
                                        ("checkpoint", "checkpoint")])
def test_A_STOP_BEFORE_PLAY_writes_no_capture(tmp_path, case, step):
    over = {"seeds": dict(seed_ok=lambda s: s != 12), "plan": dict(read_plan=lambda: {"base": 11, "n_slots": 3}),
            "mirror": dict(mirror_prob=lambda: 0.5), "checkpoint": dict(checkpoint_sha1=lambda: "0" * 40)}[case]
    out = tmp_path / "m"
    code, rec = M.conclude(SPEC, _seams(**over), out)
    assert code == M.EXIT_VOID and rec["void"]["step"] == step and not (out / "capture").exists()


def test_A_CAPTURE_FAILURE_STOPS_THE_RUN_model_drift_withholds_the_marker(tmp_path):
    def drifting(evaluator, **kw):
        game = sp.play_game(evaluator, **kw)
        evaluator.network.policy_head.bn.running_mean = mx.array([0.31, -0.2])
        return game

    out = tmp_path / "m"
    code, rec = M.conclude(SPEC, _seams(play_game=drifting), out)
    assert code == M.EXIT_VOID and rec["void"]["step"] == "exception" and "no longer matches" in rec["void"]["detail"]
    assert not (out / "capture" / "finalized.json").exists()


def test_A_MODULE_IN_TRAINING_MODE_is_refused():
    net = _net()
    with pytest.raises(M.Void, match="evaluation mode"):
        M._require_eval(net, "check")
    net.eval()
    assert M._require_eval(net, "check")["modules_in_training_mode"] == 0


def test_THE_PUBLIC_ENTRY_refuses_with_its_gate_closed_and_with_an_unpinned_card(monkeypatch, tmp_path, capsys):
    assert M.GATE.SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED is False
    existed = (M.ROOT / M.EVIDENCE_DIR).exists()             # after the real run it exists
    ckpt = tmp_path / "x.safetensors"
    ckpt.write_bytes(b"")
    assert M.main(["run", "--checkpoint", str(ckpt)]) == M.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err
    monkeypatch.setattr(M.GATE, "SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED", True)
    monkeypatch.setattr(M, "CARD_SHA256", "1" * 64)
    assert M.main(["run", "--checkpoint", str(ckpt)]) == M.EXIT_REFUSED
    assert "not the pinned card" in capsys.readouterr().err
    assert M.main(["run"]) == M.EXIT_USAGE
    assert (M.ROOT / M.EVIDENCE_DIR).exists() == existed                # nothing created or removed


def test_THE_CARD_IS_PINNED():
    import hashlib
    assert hashlib.sha256((M.ROOT / M.CARD).read_bytes()).hexdigest() == M.CARD_SHA256


def test_A_REBUILT_RECORD_THAT_DIFFERS_FROM_ITS_LEARNER_ROW_is_a_VOID(tmp_path, monkeypatch):
    from scripts.GPU.alphazero import search_distillation_d2 as D
    real = D.load_records

    def shifted(capture_dir):
        ctx, records, ids = real(capture_dir)
        records[3].visit_counts = records[3].visit_counts[::-1]
        return ctx, records, ids

    monkeypatch.setattr(D, "load_records", shifted)
    code, rec = M.conclude(SPEC, _seams(), tmp_path / "m")
    assert code == M.EXIT_VOID and rec["void"]["step"] == "alignment"
