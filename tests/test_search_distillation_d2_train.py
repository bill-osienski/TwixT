"""Search distillation D2 Stage 3 (search_distillation_d2_train.py): corpus verification, split, the single training
pass, save/reload, the criteria and the stops, through the TEST seams. The corpus comes from Stage 2's own
`conclude` (real MCTS, the repo's CPU FakeEvaluator, 8 simulations, two phases of five games) and the networks are
the small random one Stage 1's tests use -- not the card's corpus, recipe or incumbent. The public entry is checked
only with its gate closed and with an unpinned card."""
import dataclasses
import hashlib
import json
import math
import os
import signal

import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2 as D  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_generate as G  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_train as T  # noqa: E402
from scripts.GPU.alphazero import self_play as sp  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402
from tests.test_search_distillation_d2_measure import _net, _seams as _gen_seams  # noqa: E402

GEN_SPEC = {"phases": 2, "games_per_phase": 5, "n_simulations": 8, "phase_bytes": 64 << 20}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(sp, "_MIRROR_PROB", 0.0)
    root = tmp_path_factory.mktemp("d2train")
    code, rec = G.conclude(GEN_SPEC, _gen_seams(read_plan=lambda: {"base": 100, "n_slots": 10}), root / "gen",
                           root / "corpus", head="e" * 40)
    mp.undo()
    assert code == G.EXIT_DONE, rec.get("void")
    raw = (root / "gen" / "corpus.json").read_bytes()
    held = sum(g["plies"] for g in rec["games"] if g["held_out"])
    train = rec["totals"]["records"] - held
    spec = {**T.FROZEN, "corpus_sha256": hashlib.sha256(raw).hexdigest(), "teacher_digest": D.state_digest(_net()),
            "training_games": 9, "held_out_games": 1, "training_records": train, "held_out_records": held,
            "passes": 1, "steps": math.ceil(train / 32), "batch_size": 32, "learning_rate": 3e-3,
            "value_abs_mean_max": 1.0, "value_abs_p95_max": 1.0}
    return {"root": root, "raw": raw, "spec": spec, "gen": rec}


def _seams(corpus, **over):
    base = dict(read_corpus_json=lambda: corpus["raw"], checkpoint_sha1=lambda: M.CHECKPOINT_SHA1,
                load_network=_net, make_network_for=lambda path: AlphaZeroNetwork(hidden=16, n_blocks=1),
                loaded_modules=lambda: {"fixture": "not a closure"})
    base.update(over)
    return T.Seams(**base)


def _progress(out):
    return [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def passed(corpus, tmp_path_factory):
    d = tmp_path_factory.mktemp("pass")
    code, rec = T.conclude(corpus["spec"], _seams(corpus), d / "ev", d / "student.safetensors", head="f" * 40)
    return code, rec, d


def test_A_PASSING_RUN_trains_once_saves_reloads_and_judges_on_the_held_out_records(passed, corpus):
    code, rec, d = passed
    assert rec["status"] == "COMPLETED" and rec["verdict"] == "PASS" and code == T.EXIT_PASS, \
        (rec.get("void"), rec.get("criteria"))
    assert json.loads((d / "ev" / "record.json").read_text()) == json.loads(json.dumps(rec))
    spec = corpus["spec"]
    assert rec["corpus"]["training_records"] == spec["training_records"] and rec["corpus"]["held_out_games"] == [9]
    assert len(rec["training_log"]) == spec["steps"] and rec["training_failure"] is None
    assert rec["training"]["checks"]["surface_tensors_changed"] > 0
    assert rec["teacher_digest_before"] == rec["teacher_digest_after"] == spec["teacher_digest"]
    cand = rec["candidate"]
    assert cand["path"] == str(d / "student.safetensors") and os.path.exists(cand["path"])
    assert hashlib.sha256(open(cand["path"], "rb").read()).hexdigest() == cand["sha256"]
    assert cand["reloaded_state_digest"] == cand["state_digest"] == rec["training"]["student_digest"]
    assert cand["state_digest"] != spec["teacher_digest"] and cand["failed_training"] is False
    assert rec["final_held_out"]["policy_ce"] < rec["baseline_held_out"]["policy_ce"]
    assert rec["baseline_held_out"]["value_abs_max"] == 0.0              # the teacher against itself
    assert all(rec["criteria"].values()) and rec["stop_reasons"] == []
    events = [e["event"] for e in _progress(d / "ev")]
    assert events[:4] == ["start", "corpus_verified", "loaded", "baseline"]
    assert events[-4:] == ["trained", "saved", "judged", "end"]
    assert events.count("step") == len(range(0, spec["steps"], T.STEP_LOG_EVERY)) + \
        ((spec["steps"] - 1) % T.STEP_LOG_EVERY != 0)


def test_THE_RECORD_HOLDS_ADAM_S_RESOLVED_DEFAULTS_and_the_recipe(passed, corpus):
    _, rec, _ = passed
    assert {k: rec["optimizer"][k] for k in T.ADAM_DEFAULTS} == T.ADAM_DEFAULTS
    assert rec["optimizer"]["name"] == "Adam" and math.isclose(rec["optimizer"]["learning_rate"], 3e-3, rel_tol=1e-6)
    assert rec["recipe"] == dataclasses.asdict(T.recipe_for(corpus["spec"]))


def test_THE_HELD_OUT_RECORDS_ARE_READ_ONLY_FOR_THE_BASELINE_AND_THE_FINAL_CHECK(corpus, tmp_path, monkeypatch):
    training, held_out, _ = T.read_corpus(corpus["spec"], corpus["raw"])
    held = {id(r) for r in held_out}
    seen = []
    real_forward, real_train = D.forward, D.train_student

    def forward(network, records, chunk=64):
        if any(id(r) in held for r in records):
            seen.append("held_out")
        return real_forward(network, records, chunk)

    def train(student, teacher, records, recipe, **kw):
        assert not any(id(r) in held for r in records)
        seen.append("train")
        return real_train(student, teacher, records, recipe, **kw)

    monkeypatch.setattr(T, "read_corpus", lambda spec, raw: (training, held_out, {"fixture": True}))
    monkeypatch.setattr(D, "forward", forward)
    monkeypatch.setattr(D, "train_student", train)
    code, rec = T.conclude(corpus["spec"], _seams(corpus), tmp_path / "ev", tmp_path / "s.safetensors")
    assert rec["status"] == "COMPLETED", rec.get("void")
    # teacher values + baseline before training; the final check after it -- nothing on held-out in between
    assert seen == ["held_out", "held_out", "train", "held_out"]


def test_A_FAILED_CRITERION_is_a_STOP_and_the_candidate_is_kept(corpus, tmp_path):
    spec = {**corpus["spec"], "value_abs_mean_max": 0.0}
    code, rec = T.conclude(spec, _seams(corpus), tmp_path / "ev", tmp_path / "s.safetensors")
    assert code == T.EXIT_STOP and rec["status"] == "COMPLETED" and rec["verdict"] == "STOP"
    assert rec["stop_reasons"] == ["value_abs_mean_within_limit"]
    assert os.path.exists(tmp_path / "s.safetensors") and rec["candidate"]["reloaded_state_digest"]


def test_NON_FINITE_TRAINING_is_a_STOP_and_the_failed_candidate_is_kept(corpus, tmp_path):
    spec = {**corpus["spec"], "learning_rate": 1e38, "clip_norm": 1e38}
    code, rec = T.conclude(spec, _seams(corpus), tmp_path / "ev", tmp_path / "s.safetensors")
    assert code == T.EXIT_STOP and rec["verdict"] == "STOP", rec.get("void")
    assert "non-finite" in rec["training_failure"] and rec["candidate"]["failed_training"] is True
    assert {"training_finished_with_every_check", "metrics_finite"} <= set(rec["stop_reasons"])
    assert os.path.exists(tmp_path / "s.safetensors")


def test_CE_NOT_STRICTLY_LOWER_is_a_STOP():
    base = {"policy_ce": 2.0, "policy_kl": 0.1, "value_abs_mean": 0.0, "value_abs_p95": 0.0, "value_abs_max": 0.0}
    same = {**base, "value_abs_mean": 0.05, "value_abs_p95": 0.15}
    c = T.judge(T.FROZEN, base, same, None)
    assert c["held_out_policy_ce_strictly_lower"] is False and c["value_abs_mean_within_limit"] is True
    assert c["value_abs_p95_within_limit"] is True                       # the limits are inclusive (<=)
    assert not T.judge(T.FROZEN, base, {**same, "policy_ce": 1.9, "value_abs_p95": 0.1500001}, None)[
        "value_abs_p95_within_limit"]
    assert all(T.judge(T.FROZEN, base, {**same, "policy_ce": 1.9}, None).values())
    assert T.judge(T.FROZEN, base, {**same, "policy_ce": math.nan}, None)["metrics_finite"] is False
    assert T.judge(T.FROZEN, base, None, "boom") == {k: False for k in c}


@pytest.mark.parametrize("case, step", [("corpus_hash", "corpus"), ("split_count", "split"),
                                        ("record_count", "split"), ("teacher_pin", "corpus"),
                                        ("checkpoint", "checkpoint"), ("other_network", "networks"),
                                        ("loaded_not_pinned", "networks"), ("recipe", "recipe"),
                                        ("adam_defaults", "optimizer")])
def test_A_BROKEN_INPUT_OR_IDENTITY_is_a_VOID_before_any_training_and_saves_nothing(corpus, tmp_path, monkeypatch,
                                                                                    case, step):
    spec, over = dict(corpus["spec"]), {}
    if case == "corpus_hash":
        over["read_corpus_json"] = lambda: corpus["raw"] + b" "
    elif case == "split_count":
        spec["held_out_games"] = 2
    elif case == "record_count":
        spec["training_records"] += 1
        spec["steps"] = math.ceil(spec["training_records"] / spec["batch_size"])
    elif case == "teacher_pin":
        spec["teacher_digest"] = "0" * 64
    elif case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    elif case == "other_network":
        over["load_network"] = lambda: AlphaZeroNetwork(hidden=16, n_blocks=1)
    elif case == "loaded_not_pinned":                  # two equal copies, matching the corpus pin, but not this state
        def other():
            net = _net()
            net.policy_head.bn.running_mean = mx.array([0.31, -0.2])
            return net
        over["load_network"] = other
    elif case == "recipe":
        spec["steps"] += 1
    elif case == "adam_defaults":
        monkeypatch.setattr(T, "ADAM_DEFAULTS", {**T.ADAM_DEFAULTS, "eps": 1e-7})
    code, rec = T.conclude(spec, _seams(corpus, **over), tmp_path / "ev", tmp_path / "s.safetensors")
    assert code == T.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert rec["verdict"] is None and not os.path.exists(tmp_path / "s.safetensors") and "training_log" not in rec


def test_A_TAMPERED_GAME_FILE_is_a_VOID(corpus, tmp_path):
    import shutil
    copy = tmp_path / "corpus"
    shutil.copytree(corpus["root"] / "corpus", copy)
    manifest = json.loads(corpus["raw"])
    manifest["corpus_dir"] = str(copy)
    raw = json.dumps(manifest).encode()
    game = sorted((copy / "phase_01" / "games").iterdir())[0]
    game.chmod(0o644)                                        # the capture writes its files read-only
    game.write_bytes(game.read_bytes().replace(b'"z"', b'"z" ', 1))
    spec = {**corpus["spec"], "corpus_sha256": hashlib.sha256(raw).hexdigest()}
    code, rec = T.conclude(spec, _seams(corpus, read_corpus_json=lambda: raw), tmp_path / "ev",
                           tmp_path / "s.safetensors")
    assert code == T.EXIT_VOID and rec["void"]["step"] == "exception" and "differ from the receipt" in \
        rec["void"]["detail"]


def test_A_PHASE_THAT_IS_NOT_THE_MANIFEST_S_is_a_VOID_even_when_it_is_internally_consistent(corpus, tmp_path):
    manifest = json.loads(corpus["raw"])
    manifest["phases"][1]["finalized_sha256"] = "0" * 64
    raw = json.dumps(manifest).encode()
    spec = {**corpus["spec"], "corpus_sha256": hashlib.sha256(raw).hexdigest()}
    code, rec = T.conclude(spec, _seams(corpus, read_corpus_json=lambda: raw), tmp_path / "ev",
                           tmp_path / "s.safetensors")
    assert code == T.EXIT_VOID and rec["void"]["step"] == "corpus"
    assert rec["void"]["detail"] == "phase_01/finalized.json differs from the manifest"


def test_A_RELOAD_THAT_DOES_NOT_REPRODUCE_THE_STATE_is_a_VOID(corpus, tmp_path):
    def wrong(path):
        net = AlphaZeroNetwork(hidden=16, n_blocks=1)
        real = net.load_weights

        def load_weights(p, strict=True):
            real(p, strict=strict)
            net.policy_head.fc.bias = net.policy_head.fc.bias + 1.0
        net.load_weights = load_weights
        return net

    code, rec = T.conclude(corpus["spec"], _seams(corpus, make_network_for=wrong), tmp_path / "ev",
                           tmp_path / "s.safetensors")
    assert code == T.EXIT_VOID and rec["void"]["step"] == "reload" and os.path.exists(tmp_path / "s.safetensors")


def test_SIGTERM_DURING_TRAINING_is_a_VOID_with_a_record(corpus, tmp_path, monkeypatch):
    real = D.train_student

    def train(*a, on_step=None, **kw):
        def step(entry):
            if entry["step"] == 1:
                os.kill(os.getpid(), signal.SIGTERM)
            on_step(entry)
        return real(*a, on_step=step, **kw)

    monkeypatch.setattr(D, "train_student", train)
    code, rec = T.conclude(corpus["spec"], _seams(corpus), tmp_path / "ev", tmp_path / "s.safetensors")
    assert code == T.EXIT_VOID and rec["void"]["step"] == "terminated" and rec["verdict"] is None
    assert json.loads((tmp_path / "ev" / "record.json").read_text())["status"] == "VOID"


def test_AN_EXISTING_CANDIDATE_is_never_overwritten(corpus, tmp_path):
    (tmp_path / "s.safetensors").write_bytes(b"keep")
    code, rec = T.conclude(corpus["spec"], _seams(corpus), tmp_path / "ev", tmp_path / "s.safetensors")
    assert code == T.EXIT_VOID and "exists" in rec["void"]["detail"]
    assert (tmp_path / "s.safetensors").read_bytes() == b"keep"


def test_THE_FROZEN_SPEC_IS_THE_CARD_S():
    r = T.recipe_for(T.FROZEN)
    assert (r.steps, r.batch_size, r.learning_rate, r.anchor_weight, r.l2_weight, r.clip_norm, r.seed) == \
           (401, 64, 1e-4, 1.0, 1e-4, 1.0, 0)
    assert math.ceil(4 * 6403 / 64) == 401 == T.FROZEN["steps"]
    assert (T.FROZEN["training_games"], T.FROZEN["held_out_games"]) == (108, 12)
    assert (T.FROZEN["training_records"], T.FROZEN["held_out_records"]) == (6403, 617)
    assert (T.FROZEN["value_abs_mean_max"], T.FROZEN["value_abs_p95_max"]) == (0.05, 0.15)
    assert T.FROZEN["corpus_sha256"] == hashlib.sha256((T.ROOT / T.CORPUS_JSON).read_bytes()).hexdigest()
    gen = json.loads((T.ROOT / G.EVIDENCE_DIR / "record.json").read_text())
    assert gen["corpus_sha256"] == T.FROZEN["corpus_sha256"] and gen["status"] == "COMPLETED"
    assert gen["model_digest_before"] == T.FROZEN["teacher_digest"] == gen["model_digest_after"]
    assert T.ADAM_DEFAULTS == {"betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": False}
    assert T.CANDIDATE_PARENT.is_relative_to(os.path.expanduser("~")) and not T.CANDIDATE_PARENT.is_relative_to(T.ROOT)


def test_THE_PUBLIC_ENTRY_refuses_with_its_gate_closed_and_with_an_unpinned_card(monkeypatch, tmp_path, capsys):
    assert T.GATE.SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED is False
    existed = (T.ROOT / T.EVIDENCE_DIR).exists()
    ckpt = tmp_path / "x.safetensors"
    ckpt.write_bytes(b"")
    assert T.main(["run", "--checkpoint", str(ckpt)]) == T.EXIT_REFUSED and "is closed" in capsys.readouterr().err
    monkeypatch.setattr(T.GATE, "SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED", True)
    monkeypatch.setattr(T, "CARD_SHA256", "1" * 64)
    assert T.main(["run", "--checkpoint", str(ckpt)]) == T.EXIT_REFUSED
    assert "not the pinned card" in capsys.readouterr().err
    assert T.main(["run"]) == T.EXIT_USAGE and (T.ROOT / T.EVIDENCE_DIR).exists() == existed


def test_THE_CARD_IS_PINNED():
    assert hashlib.sha256((T.ROOT / T.CARD).read_bytes()).hexdigest() == T.CARD_SHA256
