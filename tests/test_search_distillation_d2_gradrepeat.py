"""Search distillation D2 reproducibility diagnostic (search_distillation_d2_gradrepeat.py): three passes over a
prefix of updates, the digests, the immediate retention of every observation, the post-pass measurements and the
VOIDs, through the TEST seams. The corpus, the Stage 3 reference and the gradient-diagnostic trace are the gradient
diagnostic's test fixtures (Stage 2's own `conclude` on the CPU FakeEvaluator, a Stage 3 `conclude` on it, and a
gradient-diagnostic `conclude` on that) -- not the card's corpus, references, recipe or incumbent. Prefix 4 updates,
snapshot at update 3, pass B attributing updates 2 and 3. The public entry is checked only with its gate closed and
with an unpinned card."""
import hashlib
import json
import math
import os
import signal

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2 as D  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_gradrepeat as G  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_gradtrace as R  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_train as T  # noqa: E402
from scripts.GPU.alphazero import trainer  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402
from tests.test_search_distillation_d2_gradtrace import ref, reproduced  # noqa: E402,F401
from tests.test_search_distillation_d2_measure import _net  # noqa: E402
from tests.test_search_distillation_d2_train import corpus  # noqa: E402,F401

PLAN = [{"name": "A", "attribution": []}, {"name": "B", "attribution": [2, 3]}, {"name": "C", "attribution": []}]


@pytest.fixture(scope="module")
def rig(corpus, ref, reproduced):                                      # noqa: F811
    _, _, out = reproduced
    trace = (out / "trace.jsonl").read_bytes()
    spec = {**G.FROZEN, **{k: ref["spec"][k] for k in ("corpus_sha256", "teacher_digest", "reference_sha256",
                                                      "reference_student_digest", "training_ids_sha256",
                                                      "training_games", "held_out_games", "training_records",
                                                      "passes", "steps", "batch_size", "learning_rate")},
            "replay1_trace_sha256": hashlib.sha256(trace).hexdigest(), "prefix_updates": 4, "snapshot_update": 3,
            "plan": PLAN}
    return {"corpus": corpus, "ref": ref, "trace": trace, "spec": spec}


def _seams(rig, **over):
    base = dict(read_corpus_json=lambda: rig["corpus"]["raw"], read_reference=lambda: rig["ref"]["raw"],
                read_replay1_trace=lambda: rig["trace"], checkpoint_sha1=lambda: M.CHECKPOINT_SHA1,
                load_network=_net, loaded_modules=lambda: {"fixture": "not a closure"})
    base.update(over)
    return G.Seams(**base)


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def done(rig, tmp_path_factory):
    d = tmp_path_factory.mktemp("repeat")
    code, rec = G.conclude(rig["spec"], _seams(rig), d / "ev", head="f" * 40)
    return code, rec, d / "ev"


def test_THREE_PASSES_COMPLETE_and_on_one_device_they_agree_to_the_bit(done, rig):
    code, rec, out = done
    assert code == G.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert json.loads((out / "record.json").read_text()) == json.loads(json.dumps(rec))
    ups = _lines(out / "updates.jsonl")
    assert [(u["pass"], u["update"]) for u in ups] == [(p, s) for p in "ABC" for s in range(4)]
    log = rig["ref"]["rec"]["training_log"]
    for u in ups:
        assert {f: u[f] for f in G.TRACE_FIELDS} == {f: log[u["update"]][f] for f in G.TRACE_FIELDS} == u["stage3"]
    c = rec["comparison"]
    for n in "ABC":
        assert c["vs_stage3"][n] == c["vs_gradient_diagnostic"][n] == {"updates_differing_bitwise": [],
                                                                       "updates_outside_tolerance": []}
    for pair in ("A-B", "A-C", "B-C"):
        p = c["between_passes"][pair]
        assert p["updates_differing_bitwise"] == [] and p["measurements_equal"] is True
        assert p["before_update"] == {"model": True, "adam": True, "inputs": True}
        assert p["before_attribution"] == {"model": True, "adam": True, "inputs": True}
        assert p["after_update"] == {"model": True, "adam": True}
    assert c["attribution_left_state_unchanged"] == {"A": True, "B": True, "C": True}
    assert rec["student_saved"] is False and rec["causal_claims"] is None and rec["held_out_records_read"] == 0
    assert not [p for p in out.rglob("*") if p.suffix == ".safetensors"]
    assert rec["corpus"]["held_out_games_never_opened"] == [9]
    assert not [f for f in rec["corpus"]["files_opened"] if f.endswith("game_9.json")]
    assert rec["teacher_digest_after"] == rig["spec"]["teacher_digest"]


def test_THE_DIGESTS_AT_THE_SNAPSHOT_UPDATE_are_written_for_each_pass_and_point(done):
    _, rec, out = done
    ds = _lines(out / "digests.jsonl")
    assert [(d["pass"], d["point"]) for d in ds] == [(p, q) for p in "ABC"
                                                     for q in ("before_attribution", "before_update", "after_update")]
    for d in ds:
        assert d["update"] == 3 and len(d["model"]) == len(d["adam"]) == 64
        if d["point"] != "after_update":
            assert set(d["per_array"]) == {"boards", "move_rows", "move_cols", "move_mask", "target",
                                           "teacher_values"} and len(d["batch"]) == 32
    before, after = ds[1], ds[2]
    assert before["model"] != after["model"] and before["adam"] != after["adam"]       # update 3 moved both


def test_PASS_B_S_ATTRIBUTION_and_EVERY_PASS_S_MEASUREMENTS_are_files_of_their_own(done):
    _, rec, out = done
    assert sorted(rec["files"]["written"]) == ["attribution/pass_B_update_2.json", "attribution/pass_B_update_3.json",
                                               "measurements/pass_A.json", "measurements/pass_B.json",
                                               "measurements/pass_C.json"]
    ups = {(u["pass"], u["update"]): u for u in _lines(out / "updates.jsonl")}
    for s in (2, 3):
        a = json.loads((out / f"attribution/pass_B_update_{s}.json").read_text())
        assert a["batch"] == ups[("B", s)]["batch"] and a["components_sum_vs_total_relative_gap"] < 1e-4
        assert R.close(a["total_grad"]["global_norm"], ups[("B", s)]["grad_norm"], {"rel_tol": 1e-5, "abs_tol": 1e-8})
    for n in "ABC":
        m = json.loads((out / f"measurements/pass_{n}.json").read_text())
        assert m["manual_logits_max_abs_diff"] == 0.0 and m["bn"]["manual_encoder_max_abs_diff"] == 0.0
        pr = m["per_record"]
        assert len(pr["logit_spread"]) == len(pr["gathered_abs_max"]) == 32
        assert all(math.isclose(s, hi - lo) for s, hi, lo in zip(pr["logit_spread"], pr["logit_max"], pr["logit_min"]))
        assert math.isclose(np.mean(pr["policy_ce"]), ups[(n, 3)]["policy"], rel_tol=1e-5)
        for f in G.TRACE_FIELDS:                        # the recomputation at the pre-update state is update 3's
            assert math.isclose(m["recomputed"][f], ups[(n, 3)][f], rel_tol=1e-5), f


def test_THE_PLAIN_PASSES_RUN_NOTHING_BETWEEN_THEIR_UPDATES(rig, tmp_path, monkeypatch):
    events = []
    real_attr, real_scale, real_update = R.attribute, G.scale, G._Sink.update

    def attribute(*a, **kw):
        events.append("attribute")
        return real_attr(*a, **kw)

    def scale(*a, **kw):
        events.append("measure")
        return real_scale(*a, **kw)

    def update(self, name, step, idx, values):
        events.append(f"{name}{step}")
        return real_update(self, name, step, idx, values)

    monkeypatch.setattr(R, "attribute", attribute)
    monkeypatch.setattr(G, "scale", scale)
    monkeypatch.setattr(G._Sink, "update", update)
    code, rec = G.conclude(rig["spec"], _seams(rig), tmp_path / "ev")
    assert code == G.EXIT_COMPLETED, rec.get("void")
    assert events == ["A0", "A1", "A2", "A3", "measure",
                      "B0", "B1", "attribute", "B2", "attribute", "B3", "measure",
                      "C0", "C1", "C2", "C3", "measure"]


def test_AN_ATTRIBUTION_SURVIVES_A_LATER_STOP_in_the_same_pass(done, rig, tmp_path, monkeypatch):
    """The fix for the gradient diagnostic's defect: pass B's update 3 goes non-finite AFTER its attribution was
    computed -- the run stops (VOID), and both attributions are on disk, equal to an unstopped run's."""
    real, calls = trainer.clip_grad_norm, []

    def clip(grads, max_norm=1.0):
        calls.append(1)
        g, n = real(grads, max_norm)
        return (g, n * mx.array(float("nan"))) if len(calls) == 8 else (g, n)    # A: 4 calls; B's 4th = update 3

    monkeypatch.setattr(trainer, "clip_grad_norm", clip)
    code, rec = G.conclude(rig["spec"], _seams(rig), tmp_path / "ev")
    assert code == G.EXIT_VOID and rec["void"]["step"] == "non_finite" and "pass B update 3" in rec["void"]["detail"]
    _, _, ok = done
    for s in (2, 3):
        name = f"attribution/pass_B_update_{s}.json"
        assert json.loads((tmp_path / "ev" / name).read_text()) == json.loads((ok / name).read_text())
    assert rec["files"]["written"] == ["measurements/pass_A.json", "attribution/pass_B_update_2.json",
                                       "attribution/pass_B_update_3.json"]
    ups = _lines(tmp_path / "ev" / "updates.jsonl")
    assert (ups[-1]["pass"], ups[-1]["update"]) == ("B", 3) and math.isnan(ups[-1]["grad_norm"])


def test_AN_ATTRIBUTION_SURVIVES_SIGTERM_right_after_it_is_written(rig, tmp_path, monkeypatch):
    real = G._Sink.attribution

    def attribution(self, name, step, data):
        real(self, name, step, data)
        if step == 3:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(G._Sink, "attribution", attribution)
    previous = signal.getsignal(signal.SIGTERM)
    code, rec = G.conclude(rig["spec"], _seams(rig), tmp_path / "ev")
    assert signal.getsignal(signal.SIGTERM) == previous
    assert code == G.EXIT_VOID and rec["void"]["step"] == "terminated"
    assert (tmp_path / "ev" / "attribution" / "pass_B_update_3.json").exists()
    assert json.loads((tmp_path / "ev" / "record.json").read_text())["status"] == "VOID"


def test_A_DIFFERENCE_FROM_STAGE_3_IS_AN_OBSERVATION_and_every_pass_still_runs(rig, tmp_path):
    bad = json.loads(rig["ref"]["raw"])
    bad["training_log"][2]["grad_norm"] *= 1 + 1e-3
    raw = json.dumps(bad).encode()
    spec = {**rig["spec"], "reference_sha256": hashlib.sha256(raw).hexdigest()}
    code, rec = G.conclude(spec, _seams(rig, read_reference=lambda: raw), tmp_path / "ev")
    assert code == G.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert set(rec["passes"]) == {"A", "B", "C"}
    for n in "ABC":
        assert rec["comparison"]["vs_stage3"][n] == {"updates_differing_bitwise": [2], "updates_outside_tolerance": [2]}
        assert rec["comparison"]["vs_gradient_diagnostic"][n]["updates_differing_bitwise"] == []


@pytest.mark.parametrize("case, step", [("corpus_hash", "corpus"), ("reference_hash", "reference"),
                                        ("replay1_hash", "replay1"), ("replay1_short", "replay1"),
                                        ("ids_digest", "corpus"), ("checkpoint", "checkpoint"),
                                        ("teacher_not_pinned", "networks"), ("student_not_pinned", "networks"),
                                        ("plan", "plan"), ("recipe", "recipe"), ("adam_defaults", "optimizer")])
def test_A_BROKEN_INPUT_OR_IDENTITY_is_a_VOID(rig, tmp_path, monkeypatch, case, step):
    spec, over = dict(rig["spec"]), {}
    if case == "corpus_hash":
        over["read_corpus_json"] = lambda: rig["corpus"]["raw"] + b" "
    elif case == "reference_hash":
        over["read_reference"] = lambda: rig["ref"]["raw"] + b" "
    elif case == "replay1_hash":
        over["read_replay1_trace"] = lambda: rig["trace"] + b" "
    elif case == "replay1_short":
        short = b"".join(rig["trace"].splitlines(keepends=True)[:3])
        spec["replay1_trace_sha256"] = hashlib.sha256(short).hexdigest()
        over["read_replay1_trace"] = lambda: short
    elif case == "ids_digest":
        spec["training_ids_sha256"] = "0" * 64
    elif case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    elif case == "teacher_not_pinned":
        over["load_network"] = lambda: AlphaZeroNetwork(hidden=16, n_blocks=1)
    elif case == "student_not_pinned":
        loads = []

        def load():
            loads.append(1)
            return _net() if len(loads) == 1 else AlphaZeroNetwork(hidden=16, n_blocks=1)
        over["load_network"] = load
    elif case == "plan":
        spec["plan"] = [PLAN[1], PLAN[0], PLAN[2]]
    elif case == "recipe":
        spec["steps"] += 1
    elif case == "adam_defaults":
        monkeypatch.setattr(G, "ADAM_DEFAULTS", {**G.ADAM_DEFAULTS, "eps": 1e-7})
    code, rec = G.conclude(spec, _seams(rig, **over), tmp_path / "ev")
    assert code == G.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    if case != "student_not_pinned":
        assert not (tmp_path / "ev" / "updates.jsonl").exists()
    else:
        assert (tmp_path / "ev" / "updates.jsonl").read_text() == ""


def test_THE_MEASUREMENT_REFUSES_A_STATE_OR_BATCH_THAT_IS_NOT_THE_PASS_S(rig):
    training, _, _ = T.read_corpus(rig["corpus"]["spec"], rig["corpus"]["raw"])
    net = _net()
    net.eval()
    snapshot = D.snapshot(net)
    idx = list(range(32))
    recipe = T.recipe_for(rig["spec"])
    tv = np.zeros(len(training), dtype=np.float32)
    batch = D._batch([training[i] for i in idx])
    good = {"model": D.state_digest(net), **G.input_digests(batch, mx.array(tv[idx]), idx)}
    seams = _seams(rig)
    assert G.measure(seams, snapshot, good, training, tv, idx, recipe, rig["spec"])["manual_logits_max_abs_diff"] == 0
    with pytest.raises(G.Void, match="pre-update state"):
        G.measure(seams, {**snapshot, "policy_head.fc.bias": snapshot["policy_head.fc.bias"] + 1}, good, training,
                  tv, idx, recipe, rig["spec"])
    with pytest.raises(G.Void, match="rebuilt batch"):
        G.measure(seams, snapshot, good, training, tv, idx[::-1], recipe, rig["spec"])


def test_THE_SCALE_MEASUREMENTS_ARE_THE_POLICY_HEAD_S_OWN_NUMBERS(rig):
    training, _, _ = T.read_corpus(rig["corpus"]["spec"], rig["corpus"]["raw"])
    net = _net()
    net.eval()
    recs = training[:6]
    batch = D._batch(recs)
    m = G.scale(net, batch, mx.zeros((6,)), T.recipe_for(rig["spec"]), 1e-6)
    logits, values, _ = net.forward_padded(*batch[:4], active_size=D.BOARD)
    lg, mask = np.array(logits), np.array(batch[3]) > 0.5
    for i, r in enumerate(recs):
        z = lg[i][mask[i]]
        assert m["per_record"]["n_legal"][i] == len(r.legal_moves)
        assert math.isclose(m["per_record"]["logit_spread"][i], float(z.max() - z.min()), rel_tol=1e-6)
        assert math.isclose(m["per_record"]["logit_abs_max"][i], float(np.abs(z).max()), rel_tol=1e-6)
    assert np.allclose(m["values"], np.array(values), rtol=0, atol=0)
    assert m["per_record"]["gathered_abs_max"][0][0] >= 0 and len(m["per_record"]["gathered_abs_max"][0]) == 2


def test_THE_FROZEN_SPEC_IS_THE_CARD_S():
    shared = ("corpus_sha256", "teacher_digest", "reference_sha256", "reference_student_digest",
              "training_ids_sha256", "training_records", "steps", "batch_size", "learning_rate", "anchor_weight",
              "l2_weight", "clip_norm", "seed", "low_variance_max")
    assert {k: G.FROZEN[k] for k in shared} == {k: R.FROZEN[k] for k in shared}
    assert (G.FROZEN["prefix_updates"], G.FROZEN["snapshot_update"]) == (22, 21)
    assert G.FROZEN["plan"] == [{"name": "A", "attribution": []}, {"name": "B", "attribution": [20, 21]},
                                {"name": "C", "attribution": []}]
    raw = (G.ROOT / G.REPLAY1_TRACE).read_bytes()
    assert G.FROZEN["replay1_trace_sha256"] == hashlib.sha256(raw).hexdigest()
    lines = G.read_replay1(G.FROZEN, raw)
    assert len(lines) == 22 and round(lines[21]["grad_norm"], 2) == 12854.44
    rec = json.loads((G.ROOT / "docs/superpowers/evidence/search-distillation-d2-gradtrace/record.json").read_text())
    assert rec["trace"] == "MISMATCH" and rec["mismatch"]["step"] == 21
    assert T.recipe_for(G.FROZEN) == T.recipe_for(T.FROZEN)


def test_THE_PUBLIC_ENTRY_refuses_with_its_gate_closed_and_with_an_unpinned_card(monkeypatch, tmp_path, capsys):
    assert G.GATE.SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED is False
    existed = (G.ROOT / G.EVIDENCE_DIR).exists()
    ckpt = tmp_path / "x.safetensors"
    ckpt.write_bytes(b"")
    assert G.main(["run", "--checkpoint", str(ckpt)]) == G.EXIT_REFUSED and "is closed" in capsys.readouterr().err
    monkeypatch.setattr(G.GATE, "SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED", True)
    monkeypatch.setattr(G, "CARD_SHA256", "1" * 64)
    assert G.main(["run", "--checkpoint", str(ckpt)]) == G.EXIT_REFUSED
    assert "not the pinned card" in capsys.readouterr().err
    assert G.main(["run"]) == G.EXIT_USAGE and (G.ROOT / G.EVIDENCE_DIR).exists() == existed


def test_THE_CARD_IS_PINNED():
    assert hashlib.sha256((G.ROOT / G.CARD).read_bytes()).hexdigest() == G.CARD_SHA256
