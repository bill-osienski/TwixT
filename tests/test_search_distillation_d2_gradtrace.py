"""Search distillation D2 gradient diagnostic (search_distillation_d2_gradtrace.py): the training-only reader, the
instrumented replay against a Stage 3 reference, the tolerance stop and the VOIDs, through the TEST seams. The corpus
is Stage 3's test corpus (Stage 2's own `conclude`, real MCTS, the CPU FakeEvaluator, two phases of five games) and
the reference is a Stage 3 `conclude` on it -- not the card's corpus, reference, recipe or incumbent. The public
entry is checked only with its gate closed and with an unpinned card."""
import hashlib
import json
import math
import os
import shutil
import signal

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2 as D  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_gradtrace as R  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_train as T  # noqa: E402
from scripts.GPU.alphazero import training_replay_capture as C  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402
from tests.test_search_distillation_d2_measure import _net  # noqa: E402
from tests.test_search_distillation_d2_train import _seams as _train_seams, corpus  # noqa: E402,F401

ATTRIBUTED = [1, 3]


def _training_ids(manifest):
    """Independently of the module: the training row ids in corpus order, through the strict whole-phase reader."""
    held, ids = set(manifest["held_out_games"]), []
    for ph in manifest["phases"]:
        _, envs = C.load_phase(os.path.join(manifest["corpus_dir"], ph["dir"]))
        for env in envs:
            if env["game_id"] not in held:
                ids += D.records_from_envelopes([env])[1]
    return ids


@pytest.fixture(scope="module")
def ref(corpus, tmp_path_factory):                                     # noqa: F811
    """A Stage 3 run on the fixture corpus: its record is the reference trace."""
    d = tmp_path_factory.mktemp("stage3")
    code, rec = T.conclude(corpus["spec"], _train_seams(corpus), d / "ev", d / "student.safetensors")
    assert rec["status"] == "COMPLETED", rec.get("void")
    raw = (d / "ev" / "record.json").read_bytes()
    manifest = json.loads(corpus["raw"])
    ids = _training_ids(manifest)
    spec = {**R.FROZEN, **{k: corpus["spec"][k] for k in ("corpus_sha256", "teacher_digest", "training_games",
                                                          "held_out_games", "training_records", "passes", "steps",
                                                          "batch_size", "learning_rate")},
            "reference_sha256": hashlib.sha256(raw).hexdigest(),
            "reference_student_digest": rec["training"]["student_digest"],
            "training_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "attribution_steps": ATTRIBUTED}
    assert spec["steps"] > max(ATTRIBUTED)
    return {"raw": raw, "rec": rec, "spec": spec, "ids": ids}


def _seams(corpus, ref, **over):                                       # noqa: F811
    base = dict(read_corpus_json=lambda: corpus["raw"], read_reference=lambda: ref["raw"],
                checkpoint_sha1=lambda: M.CHECKPOINT_SHA1, load_network=_net,
                loaded_modules=lambda: {"fixture": "not a closure"})
    base.update(over)
    return R.Seams(**base)


def _trace(out):
    return [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def reproduced(corpus, ref, tmp_path_factory):                         # noqa: F811
    d = tmp_path_factory.mktemp("replay")
    code, rec = R.conclude(ref["spec"], _seams(corpus, ref), d / "ev", head="f" * 40)
    return code, rec, d / "ev"


def test_THE_REPLAY_REPRODUCES_STAGE_3_S_TRACE_bitwise_and_saves_nothing(reproduced, ref, corpus):  # noqa: F811
    code, rec, out = reproduced
    assert code == R.EXIT_REPRODUCED and rec["status"] == "COMPLETED" and rec["trace"] == "REPRODUCED", \
        (rec.get("void"), rec.get("mismatch"))
    assert json.loads((out / "record.json").read_text()) == json.loads(json.dumps(rec))
    trace, log = _trace(out), ref["rec"]["training_log"]
    assert len(trace) == len(log) == ref["spec"]["steps"]
    for t, e in zip(trace, log):                    # on one device the replay is the reference, to the bit
        assert {f: t[f] for f in R.TRACE_FIELDS} == {f: e[f] for f in R.TRACE_FIELDS} == t["reference"]
    assert rec["final_state_bitwise_equal_to_reference"] is True
    assert rec["replay"]["student_digest"] == ref["rec"]["candidate"]["state_digest"]
    assert rec["teacher_digest_before"] == rec["teacher_digest_after"] == ref["spec"]["teacher_digest"]
    assert rec["student_saved"] is False and rec["held_out_records_read"] == 0 and rec["causal_claims"] is None
    assert not [p for p in out.rglob("*") if p.suffix == ".safetensors"]
    assert rec["corpus"]["training_ids_sha256"] == ref["spec"]["training_ids_sha256"]
    assert len(rec["training_identities"]) == ref["spec"]["training_records"]


def test_EACH_STEP_RECORDS_ITS_BATCH_CLIPPING_AND_APPLIED_UPDATE(reproduced, ref):
    _, rec, out = reproduced
    spec, n = ref["spec"], len(rec["replay"]["surface_names"])
    rng, order = np.random.default_rng(spec["seed"]), []
    for t in _trace(out):
        if len(order) < spec["batch_size"]:
            order += rng.permutation(spec["training_records"]).tolist()
        assert t["batch"] == order[:spec["batch_size"]]
        order = order[spec["batch_size"]:]
        assert len(t["raw_per_tensor"]) == len(t["update_l2_per_tensor"]) == len(t["update_max_abs_per_tensor"]) == n
        assert math.isclose(math.sqrt(sum(v * v for v in t["raw_per_tensor"])), t["grad_norm"], rel_tol=1e-4)
        assert math.isclose(t["clipped_global_norm"], t["grad_norm"] * t["clip_scale"], rel_tol=1e-4)
        assert t["clip_scale"] <= 1.0 and any(v > 0 for v in t["update_l2_per_tensor"])
    assert rec["replay"]["checks"]["surface_tensors"] == n


def test_ATTRIBUTION_ONLY_AT_THE_PRESELECTED_STEPS_and_it_adds_up(reproduced, ref):
    _, rec, out = reproduced
    att, trace = rec["attribution"], _trace(out)
    assert sorted(att) == [str(s) for s in ATTRIBUTED]
    for s in ATTRIBUTED:
        a = att[str(s)]
        assert a["batch"] == trace[s]["batch"] and a["step_grad_norm"] == trace[s]["grad_norm"]
        assert a["total_grad_matches_step"] is True
        assert a["components_sum_vs_total_relative_gap"] < 1e-4
        assert a["policy_records_sum_vs_batch_relative_gap"] < 1e-4
        assert a["anchor_records_sum_vs_batch_relative_gap"] < 1e-4 or a["anchor_grad"]["global_norm"] == 0.0
        pr = a["per_record"]
        assert len(pr["policy_ce"]) == len(pr["policy_grad_norm"]) == len(pr["anchor_grad_per_tensor"]) == 32
        assert math.isclose(sum(pr["policy_ce"]) / 32, trace[s]["policy"], rel_tol=1e-5)
        assert a["bn"]["manual_encoder_max_abs_diff"] == 0.0 and len(a["bn"]["layers"]) == 4


def test_THE_INSTRUMENTATION_DOES_NOT_ALTER_THE_UPDATES(corpus, ref, tmp_path):    # noqa: F811
    """train_student, the bare replay and the replay attributing EVERY step leave the same student, to the bit."""
    training, _, _ = T.read_corpus(corpus["spec"], corpus["raw"])
    recipe = T.recipe_for(ref["spec"])
    teacher = _net()
    teacher.eval()
    tv = D.forward(teacher, training)[1]
    plain = _net()
    plain.eval()
    out = D.train_student(plain, teacher, training, recipe, teacher_values=tv)
    log = out["steps"]
    for i, steps in enumerate(([], list(range(ref["spec"]["steps"])))):
        student = _net()
        student.eval()
        with open(tmp_path / f"t{i}.jsonl", "x") as fh:
            got = R.replay(student, teacher, training, recipe, tv, log, {**ref["spec"], "attribution_steps": steps},
                           fh)
        assert got["student_digest"] == out["student_digest"] == ref["spec"]["reference_student_digest"]
        assert [{f: t[f] for f in R.TRACE_FIELDS} for t in _trace_file(tmp_path / f"t{i}.jsonl")] == \
               [{f: e[f] for f in R.TRACE_FIELDS} for e in log]
        assert sorted(got["attribution"], key=int) == [str(s) for s in steps]


def _trace_file(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_WHEN_THE_CLIP_BINDS_the_raw_and_the_clipped_norms_are_both_recorded(corpus, ref, tmp_path):  # noqa: F811
    """A clip far below the fixture's norms: every update is clipped, the per-tensor norms stay PRE-clip."""
    training, teacher, tv = _fresh(corpus)
    spec = {**ref["spec"], "clip_norm": 1e-5, "attribution_steps": []}
    recipe = T.recipe_for(spec)
    plain = _net()
    plain.eval()
    log = D.train_student(plain, teacher, training, recipe, teacher_values=tv)["steps"]
    student = _net()
    student.eval()
    with open(tmp_path / "t.jsonl", "x") as fh:
        R.replay(student, teacher, training, recipe, tv, log, spec, fh)
    for t in _trace_file(tmp_path / "t.jsonl"):
        assert t["grad_norm"] > 10 * spec["clip_norm"] and t["clip_scale"] < 0.1
        assert math.isclose(math.sqrt(sum(v * v for v in t["raw_per_tensor"])), t["grad_norm"], rel_tol=1e-4)
        assert math.isclose(t["clipped_global_norm"], spec["clip_norm"], rel_tol=1e-4)


def test_NO_HELD_OUT_GAME_FILE_IS_OPENED(corpus, ref, tmp_path):        # noqa: F811
    copy = tmp_path / "corpus"
    shutil.copytree(corpus["root"] / "corpus", copy)
    manifest = json.loads(corpus["raw"])
    manifest["corpus_dir"] = str(copy)
    held = []
    for ph in manifest["phases"]:
        for g in ph["games"]:
            if g in manifest["held_out_games"]:
                held.append(f"{ph['dir']}/{C._game_path(g)}")
                (copy / ph["dir"] / C._game_path(g)).unlink()      # opening one now raises
    assert held
    raw = json.dumps(manifest).encode()
    spec = {**ref["spec"], "corpus_sha256": hashlib.sha256(raw).hexdigest()}
    records, identities, summary = R.read_training(spec, raw)
    assert [i["row_id"] for i in identities] == ref["ids"] and len(records) == spec["training_records"]
    assert not set(held) & set(summary["files_opened"]) and summary["held_out_games_never_opened"] == [9]
    with pytest.raises(Exception):                                  # Stage 3's reader does open them
        T.read_corpus({**corpus["spec"], "corpus_sha256": spec["corpus_sha256"]}, raw)


@pytest.mark.parametrize("field, scale, code", [("grad_norm", 1 + 2e-5, R.EXIT_MISMATCH),
                                                ("grad_norm", 1 + 5e-6, R.EXIT_REPRODUCED),
                                                ("loss", 1 + 2e-5, R.EXIT_MISMATCH),
                                                ("policy", 1 + 2e-5, R.EXIT_MISMATCH),
                                                ("anchor", 1 + 2e-5, R.EXIT_MISMATCH)])
def test_A_STEP_OUTSIDE_THE_TOLERANCE_STOPS_THE_REPLAY_there(corpus, ref, tmp_path, field, scale,  # noqa: F811
                                                             code):
    k = 2
    bad = json.loads(ref["raw"])
    assert bad["training_log"][k][field] > 1e-3                      # the relative term is the one tested
    bad["training_log"][k][field] *= scale
    raw = json.dumps(bad).encode()
    spec = {**ref["spec"], "reference_sha256": hashlib.sha256(raw).hexdigest()}
    got, rec = R.conclude(spec, _seams(corpus, ref, read_reference=lambda: raw), tmp_path / "ev")
    assert got == code and rec["status"] == "COMPLETED", rec.get("void")
    if code == R.EXIT_MISMATCH:
        assert rec["trace"] == "MISMATCH" and rec["mismatch"]["step"] == k and rec["mismatch"]["fields"] == [field]
        assert len(_trace(tmp_path / "ev")) == k + 1 and "attribution" not in rec
        assert rec["mismatch"]["reference"][field] == bad["training_log"][k][field]
    else:
        assert rec["trace"] == "REPRODUCED"


def _fresh(corpus):                                                    # noqa: F811
    training, _, _ = T.read_corpus(corpus["spec"], corpus["raw"])
    teacher = _net()
    teacher.eval()
    return training, teacher, D.forward(teacher, training)[1].astype(np.float32)


def test_THE_FIRST_STEP_S_TENSORS_ARE_THE_GRADIENT_AND_AFTER_MINUS_BEFORE(reproduced, corpus, ref):  # noqa: F811
    """Step 0 recomputed independently: per-tensor pre-clip gradient norms, and the applied update from a
    one-update train_student run, tensor by tensor under the record's names."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from scripts.GPU.alphazero.trainer import MainModule, compute_masked_log_probs
    _, rec, out = reproduced
    names, line = rec["replay"]["surface_names"], _trace(out)[0]
    training, teacher, tv = _fresh(corpus)
    recipe = T.recipe_for(ref["spec"])
    student = _net()
    student.eval()
    before = D.snapshot(student)
    surface = MainModule(student.encoder, student.policy_head)
    boards, rows, cols, mask, target = D._batch([training[i] for i in line["batch"]])

    def loss_fn(_s):
        logits, values, _ = student.forward_padded(boards, rows, cols, mask, active_size=D.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        anchor = mx.mean((values - mx.array(tv[line["batch"]])) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(_s.trainable_parameters()))
        return policy + recipe.anchor_weight * anchor + recipe.l2_weight * l2

    _, g = nn.value_and_grad(surface, loss_fn)(surface)
    grads = dict(tree_flatten(g))
    for n, v in zip(names, line["raw_per_tensor"]):
        assert math.isclose(float(np.linalg.norm(np.array(grads[n]).ravel())), v, rel_tol=1e-4, abs_tol=1e-9), n
    one = _net()
    one.eval()
    D.train_student(one, teacher, training, D.Recipe(**{**recipe.__dict__, "steps": 1}), teacher_values=tv)
    after = D.snapshot(one)
    for n, v, m in zip(names, line["update_l2_per_tensor"], line["update_max_abs_per_tensor"]):
        d = after[n].astype(np.float64) - before[n].astype(np.float64)
        assert math.isclose(float(np.linalg.norm(d.ravel())), v, rel_tol=1e-4, abs_tol=1e-12), n
        assert math.isclose(float(np.abs(d).max()), m, rel_tol=1e-4, abs_tol=1e-12), n


def test_PER_RECORD_ATTRIBUTION_IS_THAT_RECORD_S_OWN_GRADIENT(corpus):    # noqa: F811
    """Record j's attributed gradient, times the batch size, is the gradient of a batch holding only record j
    (twice, to stay on the batched path)."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from scripts.GPU.alphazero.trainer import MainModule, compute_masked_log_probs
    training, teacher, tv = _fresh(corpus)
    recipe = T.recipe_for(corpus["spec"])
    student = _net()
    student.eval()
    student.encoder.conv1.weight = student.encoder.conv1.weight * 1.1      # not the teacher: the anchor is nonzero
    surface = MainModule(student.encoder, student.policy_head)
    names = [k for k, _ in tree_flatten(surface.trainable_parameters())]
    idx = [5, 17, 2, 40]
    batch = D._batch([training[i] for i in idx])
    a = R.attribute(student, surface, recipe, batch, mx.array(tv[idx]), names, 1e-6)
    for j in (0, 2):
        b2 = D._batch([training[idx[j]]] * 2)
        t2 = mx.array(tv[[idx[j]] * 2])
        for which in ("policy", "anchor"):
            def f(_s):
                logits, values, _ = student.forward_padded(*b2[:4], active_size=D.BOARD)
                if which == "policy":
                    return mx.mean(-mx.sum(b2[4] * compute_masked_log_probs(logits, b2[3]), axis=1))
                return recipe.anchor_weight * mx.mean((values - t2) ** 2)
            _, g = nn.value_and_grad(surface, f)(surface)
            want = math.sqrt(sum(float(mx.sum(v ** 2)) for _, v in tree_flatten(g)))
            got = a["per_record"][f"{which}_grad_norm"][j] * len(idx)
            assert want > 0 and math.isclose(got, want, rel_tol=1e-3), (which, j, got, want)


def test_THE_TOLERANCE_is_inclusive_and_never_passes_NaN():
    s = {"rel_tol": 1e-5, "abs_tol": 1e-8}
    assert R.close(1.0, 1.0, s) and R.close(0.0, 0.0, s) and R.close(1e-8, 0.0, s) and not R.close(3e-8, 0.0, s)
    assert R.close(100.0 + 1e-3, 100.0, s) and not R.close(100.0 + 1.01e-3, 100.0, s)
    assert not R.close(math.nan, 1.0, s) and not R.close(1.0, math.nan, s) and not R.close(math.inf, 1.0, s)


@pytest.mark.parametrize("case, step", [("corpus_hash", "corpus"), ("reference_hash", "reference"),
                                        ("reference_not_completed", "reference"), ("reference_short", "reference"),
                                        ("reference_recipe", "reference"), ("reference_student", "reference"),
                                        ("ids_digest", "corpus"), ("split_count", "split"),
                                        ("record_count", "split"), ("checkpoint", "checkpoint"),
                                        ("other_network", "networks"), ("loaded_not_pinned", "networks"),
                                        ("recipe", "recipe"), ("adam_defaults", "optimizer")])
def test_A_BROKEN_INPUT_OR_IDENTITY_is_a_VOID_before_any_replay(corpus, ref, tmp_path, monkeypatch,  # noqa: F811
                                                                case, step):
    spec, over = dict(ref["spec"]), {}

    def reference(edit):
        r = json.loads(ref["raw"])
        edit(r)
        raw = json.dumps(r).encode()
        spec["reference_sha256"] = hashlib.sha256(raw).hexdigest()
        over["read_reference"] = lambda: raw

    if case == "corpus_hash":
        over["read_corpus_json"] = lambda: corpus["raw"] + b" "
    elif case == "reference_hash":
        over["read_reference"] = lambda: ref["raw"] + b" "
    elif case == "reference_not_completed":
        reference(lambda r: r.update(status="VOID"))
    elif case == "reference_short":
        reference(lambda r: r["training_log"].pop())
    elif case == "reference_recipe":
        reference(lambda r: r["recipe"].update(anchor_weight=2.0))
    elif case == "reference_student":
        spec["reference_student_digest"] = "0" * 64
    elif case == "ids_digest":
        spec["training_ids_sha256"] = "0" * 64
    elif case == "split_count":
        spec["held_out_games"] = 2
    elif case == "record_count":
        spec["training_records"] += 1
        spec["steps"] = math.ceil(spec["training_records"] / spec["batch_size"])
    elif case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    elif case == "other_network":
        over["load_network"] = lambda: AlphaZeroNetwork(hidden=16, n_blocks=1)
    elif case == "loaded_not_pinned":
        def other():
            net = _net()
            net.policy_head.bn.running_mean = mx.array([0.31, -0.2])
            return net
        over["load_network"] = other
    elif case == "recipe":
        spec["steps"] += 1
    elif case == "adam_defaults":
        monkeypatch.setattr(R, "ADAM_DEFAULTS", {**R.ADAM_DEFAULTS, "eps": 1e-7})
    code, rec = R.conclude(spec, _seams(corpus, ref, **over), tmp_path / "ev")
    assert code == R.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert rec["trace"] is None and not (tmp_path / "ev" / "trace.jsonl").exists()


def test_A_TAMPERED_TRAINING_GAME_FILE_is_a_VOID(corpus, ref, tmp_path):     # noqa: F811
    copy = tmp_path / "corpus"
    shutil.copytree(corpus["root"] / "corpus", copy)
    manifest = json.loads(corpus["raw"])
    manifest["corpus_dir"] = str(copy)
    raw = json.dumps(manifest).encode()
    game = copy / "phase_01" / C._game_path(manifest["phases"][1]["games"][0])
    game.chmod(0o644)                                        # the capture writes its files read-only
    game.write_bytes(game.read_bytes().replace(b'"z"', b'"z" ', 1))
    spec = {**ref["spec"], "corpus_sha256": hashlib.sha256(raw).hexdigest()}
    code, rec = R.conclude(spec, _seams(corpus, ref, read_corpus_json=lambda: raw), tmp_path / "ev")
    assert code == R.EXIT_VOID and rec["void"]["step"] == "corpus" and "differ from the receipt" in \
        rec["void"]["detail"]


def test_A_PHASE_FILE_THAT_IS_NOT_THE_MANIFEST_S_is_a_VOID(corpus, ref, tmp_path):     # noqa: F811
    manifest = json.loads(corpus["raw"])
    manifest["phases"][1]["context_sha256"] = "0" * 64
    raw = json.dumps(manifest).encode()
    spec = {**ref["spec"], "corpus_sha256": hashlib.sha256(raw).hexdigest()}
    code, rec = R.conclude(spec, _seams(corpus, ref, read_corpus_json=lambda: raw), tmp_path / "ev")
    assert code == R.EXIT_VOID and rec["void"] == {"step": "corpus",
                                                   "detail": "phase_01/context.json differs from the manifest"}


def test_SIGTERM_DURING_THE_REPLAY_is_a_VOID_with_a_record(corpus, ref, tmp_path, monkeypatch):    # noqa: F811
    real = R.replay

    def replay(*a, **kw):
        on_step = a[-1]

        def step(line):
            if line["step"] == 1:
                os.kill(os.getpid(), signal.SIGTERM)
            on_step(line)
        return real(*a[:-1], step, **kw)

    monkeypatch.setattr(R, "replay", replay)
    previous = signal.getsignal(signal.SIGTERM)
    code, rec = R.conclude(ref["spec"], _seams(corpus, ref), tmp_path / "ev")
    assert signal.getsignal(signal.SIGTERM) == previous
    assert code == R.EXIT_VOID and rec["void"]["step"] == "terminated" and rec["trace"] is None
    assert json.loads((tmp_path / "ev" / "record.json").read_text())["status"] == "VOID"
    assert len(_trace(tmp_path / "ev")) == 2                       # the partial trace is kept


def test_THE_BATCHNORM_TAPS_split_low_variance_channels():
    mx.random.seed(1)
    net = AlphaZeroNetwork(hidden=8, n_blocks=1)
    var = np.ones(8, dtype=np.float32)
    var[[2, 5]] = 0.0
    net.encoder.bn1.running_var = mx.array(var)
    net.eval()
    rng = np.random.default_rng(0)
    boards = mx.array(rng.random((3, 24, 24, net.encoder.conv1.weight.shape[-1])).astype(np.float32))
    rows = mx.array(np.tile(np.arange(4, dtype=np.int32), (3, 1)))
    mask = mx.ones((3, 4))
    taps = R.bn_taps(net, boards, rows, rows, mask, 1e-6)
    assert taps["manual_encoder_max_abs_diff"] == 0.0
    layer = taps["layers"]["encoder.bn1"]
    assert layer["low_variance_channels"] == 2 and set(layer["argmax_channel_low"]) <= {2, 5}
    from scripts.GPU.alphazero.network import canonicalize_batch
    pre = np.array(net.encoder.conv1(canonicalize_batch(boards, rows, rows, mask, 24)[0]))
    xh = np.abs((pre - np.array(net.encoder.bn1.running_mean)) / np.sqrt(var + net.encoder.bn1.eps))
    assert np.allclose(layer["max_abs_low"], xh[..., [2, 5]].max(axis=(1, 2, 3)), rtol=1e-6)
    assert np.allclose(layer["max_abs_other"], np.delete(xh, [2, 5], axis=3).max(axis=(1, 2, 3)), rtol=1e-6)
    assert taps["layers"]["encoder.blocks.0.bn1"]["low_variance_channels"] == 0


def test_THE_SUMMARY_names_the_largest_tensor_per_update():
    trace = [{"reference": {"grad_norm": 2e4}, "raw_per_tensor": [1.0, 3.0]},
             {"reference": {"grad_norm": 2.0}, "raw_per_tensor": [2.0, 1.0]},
             {"reference": {"grad_norm": 3e4}, "raw_per_tensor": [0.0, 4.0]}]
    s = R.summarize(trace, ["a", "b"], {"spike_norm": 1e4})
    assert s["spike"] == {"updates": 2, "largest_tensor_counts": {"b": 2}, "largest_tensor_share_min": 0.9,
                          "largest_tensor_share_median": 1.0}
    assert s["other"]["largest_tensor_counts"] == {"a": 1} and s["other"]["largest_tensor_share_min"] == 0.8


def test_THE_FROZEN_SPEC_IS_THE_CARD_S_and_Stage_3_s():
    shared = ("corpus_sha256", "teacher_digest", "training_games", "held_out_games", "training_records", "passes",
              "steps", "batch_size", "learning_rate", "anchor_weight", "l2_weight", "clip_norm", "seed")
    assert {k: R.FROZEN[k] for k in shared} == {k: T.FROZEN[k] for k in shared}
    assert R.FROZEN["reference_sha256"] == hashlib.sha256((R.ROOT / R.REFERENCE).read_bytes()).hexdigest()
    stage3 = json.loads((R.ROOT / R.REFERENCE).read_text())
    assert R.FROZEN["reference_student_digest"] == stage3["candidate"]["state_digest"] == \
        stage3["training"]["student_digest"]
    assert R.FROZEN["attribution_steps"] == [20, 21, 361]
    norms = [e["grad_norm"] for e in stage3["training_log"]]
    assert norms[20] < 10 and norms[21] > R.FROZEN["spike_norm"] and norms[361] == max(norms)
    assert (R.FROZEN["rel_tol"], R.FROZEN["abs_tol"], R.FROZEN["low_variance_max"]) == (1e-5, 1e-8, 1e-6)
    assert R.ADAM_DEFAULTS == T.ADAM_DEFAULTS and R.CORPUS_JSON == T.CORPUS_JSON
    assert R.TRACE_FIELDS == ("loss", "policy", "anchor", "grad_norm")
    assert "held" not in R.REFERENCE and R.EVIDENCE_DIR != T.EVIDENCE_DIR


def test_THE_PUBLIC_ENTRY_refuses_with_its_gate_closed_and_with_an_unpinned_card(monkeypatch, tmp_path, capsys):
    assert R.GATE.SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED is False
    existed = (R.ROOT / R.EVIDENCE_DIR).exists()
    ckpt = tmp_path / "x.safetensors"
    ckpt.write_bytes(b"")
    assert R.main(["run", "--checkpoint", str(ckpt)]) == R.EXIT_REFUSED and "is closed" in capsys.readouterr().err
    monkeypatch.setattr(R.GATE, "SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED", True)
    monkeypatch.setattr(R, "CARD_SHA256", "1" * 64)
    assert R.main(["run", "--checkpoint", str(ckpt)]) == R.EXIT_REFUSED
    assert "not the pinned card" in capsys.readouterr().err
    assert R.main(["run"]) == R.EXIT_USAGE and (R.ROOT / R.EVIDENCE_DIR).exists() == existed


def test_THE_CARD_IS_PINNED():
    assert hashlib.sha256((R.ROOT / R.CARD).read_bytes()).hexdigest() == R.CARD_SHA256
