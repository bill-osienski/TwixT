"""Search distillation D2 Stage 2 (search_distillation_d2_generate.py): phases, split, eligibility and stops through
the TEST seams -- real MCTS, the repo's CPU FakeEvaluator, a small network that is only hashed, 8 simulations and
two phases of five games (not the card's 12 x 10 at 1,600). The public entry is checked only with its gate closed
and with an unpinned card."""
import hashlib
import json
import os
import signal

import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2 as D  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_generate as G  # noqa: E402
from scripts.GPU.alphazero import self_play as sp  # noqa: E402
from scripts.GPU.alphazero import training_replay_capture as C  # noqa: E402
from tests.test_search_distillation_d2_measure import _seams  # noqa: E402

SPEC = {"phases": 2, "games_per_phase": 5, "n_simulations": 8, "phase_bytes": 64 << 20}
PLAN = {"base": 100, "n_slots": 10}


@pytest.fixture(autouse=True)
def _mirror_off(monkeypatch):
    monkeypatch.setattr(sp, "_MIRROR_PROB", 0.0)


def _progress(out):
    return [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(sp, "_MIRROR_PROB", 0.0)
    root = tmp_path_factory.mktemp("gen")
    code, rec = G.conclude(SPEC, _seams(read_plan=lambda: PLAN), root / "ev", root / "corpus", head="e" * 40)
    mp.undo()
    return code, rec, root


def test_A_COMPLETED_RUN_is_eligible_and_its_corpus_manifest_binds_every_phase(completed):
    code, rec, root = completed
    assert code == G.EXIT_DONE and rec["status"] == "COMPLETED" and rec["eligible_for_training"] is True, rec.get("void")
    corpus = json.loads((root / "ev" / "corpus.json").read_text())
    assert corpus["eligible_for_training"] is True and [ph["games"] for ph in corpus["phases"]] == [[0, 1, 2, 3, 4],
                                                                                                    [5, 6, 7, 8, 9]]
    assert corpus["held_out_games"] == [9] and corpus["training_games"] == list(range(9))
    for ph in corpus["phases"]:
        d = root / "corpus" / ph["dir"]
        assert hashlib.sha256((d / "finalized.json").read_bytes()).hexdigest() == ph["finalized_sha256"]
        for f in C.PHASE_FILES:                                      # the evidence copies are the corpus's bytes
            assert (root / "ev" / "phases" / ph["dir"] / f).read_bytes() == (d / f).read_bytes()
        _ctx, records, _ids = D.load_records(str(d))
        assert len(records) == ph["records"]
    assert [g["seed"] for g in rec["games"]] == [100 + s for s in range(10)]
    assert [g["held_out"] for g in rec["games"]] == [False] * 9 + [True]
    assert rec["model_digest_before"] == rec["model_digest_after"] and rec["totals"]["games"] == 10
    events = [e["event"] for e in _progress(root / "ev")]
    assert events.count("game_start") == 10 and events.count("phase_end") == 2 and events[-1] == "end"


def test_THE_FROZEN_SPEC_AND_SPLIT_ARE_THE_CARD_S():
    assert G.FROZEN == {"phases": 12, "games_per_phase": 10, "n_simulations": 1600, "phase_bytes": 84 << 20}
    assert G.FROZEN["phase_bytes"] >= 10 * C.ENVELOPE_MAX_BYTES
    slots = [s for p in range(12) for s in G.phase_slots(G.FROZEN, p)]
    assert slots == list(range(120))
    assert [s for s in slots if G.held_out(s)] == [10 * p + 9 for p in range(12)]
    assert sum(not G.held_out(s) for s in slots) == 108
    assert G.CORPUS_PARENT.is_relative_to(os.path.expanduser("~")) and not G.CORPUS_PARENT.is_relative_to(G.ROOT)


def test_SIGTERM_in_phase_1_is_a_VOID_not_eligible_and_both_phases_are_kept(tmp_path):
    def play(evaluator, **kw):
        if kw["game_id"] == 7:
            os.kill(os.getpid(), signal.SIGTERM)
        return sp.play_game(evaluator, **kw)

    code, rec = G.conclude(SPEC, _seams(read_plan=lambda: PLAN, play_game=play), tmp_path / "ev", tmp_path / "c")
    assert code == G.EXIT_VOID and rec["void"]["step"] == "terminated" and rec["eligible_for_training"] is False
    assert (tmp_path / "c" / "phase_00" / "finalized.json").exists()                    # completed phase kept
    assert (tmp_path / "c" / "phase_01" / "games" / "game_6.json").exists()             # partial phase kept
    assert not (tmp_path / "c" / "phase_01" / "finalized.json").exists()
    assert not (tmp_path / "ev" / "corpus.json").exists() and len(rec["phases"]) == 1


@pytest.mark.parametrize("case, step", [("plan", "seeds"), ("seed", "seeds"), ("mirror", "mirror"),
                                        ("checkpoint", "checkpoint")])
def test_A_STOP_BEFORE_PLAY_writes_no_phase(tmp_path, case, step):
    over = {"plan": dict(read_plan=lambda: {"base": 100, "n_slots": 9}),
            "seed": dict(read_plan=lambda: PLAN, seed_ok=lambda s: s != 109),
            "mirror": dict(read_plan=lambda: PLAN, mirror_prob=lambda: 0.5),
            "checkpoint": dict(read_plan=lambda: PLAN, checkpoint_sha1=lambda: "0" * 40)}[case]
    code, rec = G.conclude(SPEC, _seams(**over), tmp_path / "ev", tmp_path / "c")
    assert code == G.EXIT_VOID and rec["void"]["step"] == step and list((tmp_path / "c").iterdir()) == []


def test_A_CAPTURE_FAILURE_stops_at_once_without_retry(tmp_path):
    calls = []

    def drifting(evaluator, **kw):
        calls.append(kw["game_id"])
        game = sp.play_game(evaluator, **kw)
        if kw["game_id"] == 1:
            evaluator.network.policy_head.bn.running_mean = mx.array([0.31, -0.2])
        return game

    code, rec = G.conclude(SPEC, _seams(read_plan=lambda: PLAN, play_game=drifting), tmp_path / "ev", tmp_path / "c")
    assert code == G.EXIT_VOID and "no longer matches" in rec["void"]["detail"] and calls == [0, 1, 2, 3, 4]
    assert not (tmp_path / "c" / "phase_00" / "finalized.json").exists() and not (tmp_path / "c" / "phase_01").exists()


def test_THE_PUBLIC_ENTRY_refuses_with_its_gate_closed_and_with_an_unpinned_card(monkeypatch, tmp_path, capsys):
    assert G.GATE.SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED is False
    existed = (G.ROOT / G.EVIDENCE_DIR).exists()             # after the real run it exists
    ckpt = tmp_path / "x.safetensors"
    ckpt.write_bytes(b"")
    assert G.main(["run", "--checkpoint", str(ckpt)]) == G.EXIT_REFUSED and "is closed" in capsys.readouterr().err
    monkeypatch.setattr(G.GATE, "SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED", True)
    monkeypatch.setattr(G, "CARD_SHA256", "1" * 64)
    assert G.main(["run", "--checkpoint", str(ckpt)]) == G.EXIT_REFUSED
    assert "not the pinned card" in capsys.readouterr().err
    assert G.main(["run"]) == G.EXIT_USAGE and (G.ROOT / G.EVIDENCE_DIR).exists() == existed


def test_THE_CARD_IS_PINNED():
    assert hashlib.sha256((G.ROOT / G.CARD).read_bytes()).hexdigest() == G.CARD_SHA256
