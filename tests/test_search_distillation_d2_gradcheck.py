"""Search distillation D2 gradient correctness check (search_distillation_d2_gradcheck.py): the NumPy reference and
its finite-difference validation, the device comparison, the stage localization, fixtures, incremental retention and
the VOIDs, through the TEST seams. Both device labels run on the CPU here ("gpu" is remapped), so every honest
comparison agrees to the bit and every disagreement is an injected fault. The corpus and references are the earlier
diagnostics' fixtures; the "S21" state here is the pre-update-3 state of a 4-update fixture prefix. The public entry
is checked only with its gate closed and with an unpinned card."""
import hashlib
import json
import math
import os
import signal

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2_gradcheck as K  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_gradtrace as R  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import trainer  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402
from tests.test_search_distillation_d2_gradrepeat import done, rig  # noqa: E402,F401
from tests.test_search_distillation_d2_gradtrace import ref, reproduced  # noqa: E402,F401
from tests.test_search_distillation_d2_measure import _net  # noqa: E402
from tests.test_search_distillation_d2_train import corpus  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _both_labels_on_the_cpu(monkeypatch):
    monkeypatch.setattr(K, "DEVICES", {"cpu": mx.cpu, "gpu": mx.cpu})


@pytest.fixture(scope="module")
def kit(rig, done):                                                    # noqa: F811
    _, _, out = done
    rec, dig = (out / "record.json").read_bytes(), (out / "digests.jsonl").read_bytes()
    spec = {**K.FROZEN, **{k: rig["spec"][k] for k in ("corpus_sha256", "teacher_digest", "reference_sha256",
                                                      "reference_student_digest", "training_ids_sha256",
                                                      "training_games", "held_out_games", "training_records",
                                                      "passes", "steps", "batch_size", "learning_rate")},
            "repeat_record_sha256": hashlib.sha256(rec).hexdigest(),
            "repeat_digests_sha256": hashlib.sha256(dig).hexdigest(),
            "states": [{"name": "S0", "prefix_updates": 0, "batch_update": 0},
                       {"name": "S21", "prefix_updates": 3, "batch_update": 3}], "repeats": 2}
    return {"rig": rig, "rec": rec, "dig": dig, "spec": spec}


def _seams(kit, **over):
    rig = kit["rig"]
    base = dict(read_corpus_json=lambda: rig["corpus"]["raw"], read_reference=lambda: rig["ref"]["raw"],
                read_repeat_record=lambda: kit["rec"], read_repeat_digests=lambda: kit["dig"],
                checkpoint_sha1=lambda: M.CHECKPOINT_SHA1, load_network=_net,
                loaded_modules=lambda: {"fixture": "not a closure"})
    base.update(over)
    return K.Seams(**base)


def _go(kit, tmp_path, spec=None, **over):
    return K.conclude(spec or kit["spec"], _seams(kit, **over), tmp_path / "ev", tmp_path / "fixtures")


@pytest.fixture(scope="module")
def clean(kit, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(K, "DEVICES", {"cpu": mx.cpu, "gpu": mx.cpu})
    d = tmp_path_factory.mktemp("gradcheck")
    try:
        code, rec = K.conclude(kit["spec"], _seams(kit), d / "ev", d / "fixtures", head="f" * 40)
    finally:
        mp.undo()
    return code, rec, d


def _j(path):
    return json.loads(path.read_text())


def test_A_CLEAN_RUN_COMPLETES_and_every_honest_comparison_agrees(clean, kit):
    code, rec, d = clean
    ev = d / "ev"
    assert code == K.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert json.loads((ev / "record.json").read_text()) == json.loads(json.dumps(rec))
    assert rec["states"]["S0"]["is_the_incumbent"] is True
    assert rec["states"]["S21"]["equals_pass_A_recorded_digest"] is True                  # the CPU repeats itself
    assert rec["teacher_values"]["equals_reproducibility_diagnostic_s"] is True
    for name in ("S0", "S21"):
        assert rec["results"][name]["first_disagreement"] == {"cpu": {"forward": None, "backward": None},
                                                             "gpu": {"forward": None, "backward": None}}
        assert rec["results"][name]["reference_disagrees"] == {"cpu": False, "gpu": False}
        full = _j(ev / "full" / f"{name}.json")
        for dev in ("cpu", "gpu"):
            g = full["devices"][dev]
            assert all(all(v["bitwise"] for v in e.values()) for e in g["grads_vs_reference"])
            assert all(v["within_tol"] for v in g["additivity"].values())
    for case in ("repeated_indices", "batch_members", "padding", "weighted_additivity"):
        assert rec["synthetic"][case]["disagrees"] == {"cpu": False, "gpu": False}
        dirc = rec["synthetic"][case]["directional"]
        assert dirc["skipped"] is False and dirc["analytic_ok"] and dirc["mlx_cpu_ok"] and dirc["mlx_gpu_ok"]
    assert rec["files"]["fixtures"] == [] and not (d / "fixtures").exists()
    assert rec["full_state_saved"] is False and rec["causal_claims"] is None and rec["held_out_records_read"] == 0
    assert not [p for p in d.rglob("*") if p.suffix == ".safetensors"]
    lines = [json.loads(x) for x in (ev / "reconstruction.jsonl").read_text().splitlines()]
    assert [x["update"] for x in lines] == [0, 1, 2]
    assert all({f: x[f] for f in R.TRACE_FIELDS} == x["stage3"] for x in lines)


def test_THE_STAGES_ARE_LOCALIZED_FROM_IDENTICAL_INPUTS(clean):
    _, rec, d = clean
    st = _j(d / "ev" / "stages" / "S21.json")
    names = ["encoder", "policy_conv", "policy_bn", "policy_relu", "gather", "fc", "fc_relu", "out", "head_mask",
             "log_softmax", "policy_ce", "value_head", "anchor"]
    for dev in ("cpu", "gpu"):
        assert set(st["forward"][dev]) == set(names) == set(st["backward"][dev])          # JSON keys are sorted
        assert set(st["backward"][dev]["policy_conv"]["tensors"]) == {"weight", "bias"}
        assert set(st["backward"][dev]["encoder"]["tensors"]) >= {"conv1.weight", "bn1.weight"}
    assert all(all(v) for v in st["canonicalize"]["gpu"].values())
    for name in ("S0", "S21"):                      # the decomposition reproduces autodiff's tensor gradients
        chain = _j(d / "ev" / "stages" / f"{name}.json")["chain_vs_autodiff"]
        full = _j(d / "ev" / "full" / f"{name}.json")["devices"]["cpu"]["grads_vs_reference"][0]
        assert set(chain) == set(full) and all(v.get("within_tol") for v in chain.values()), name


def test_AN_ENCODER_STAGE_FIXTURE_HOLDS_NO_ENCODER_WEIGHTS(kit, tmp_path, monkeypatch):
    real = K.stage_list

    def stages(net, ex):
        out = real(net, ex)
        name, module, fn = out[0]
        assert name == "encoder"
        return [(name, module, lambda a: fn(a) * (1.01 if K.CURRENT["device"] == "gpu" else 1.0))] + out[1:]

    monkeypatch.setattr(K, "stage_list", stages)
    code, rec = _go(kit, tmp_path)
    assert code == K.EXIT_COMPLETED, rec.get("void")
    assert rec["results"]["S0"]["first_disagreement"]["gpu"]["forward"] == "encoder"
    fx = {json.loads(x)["name"]: json.loads(x) for x in (tmp_path / "ev" / "fixtures.jsonl").read_text().splitlines()}
    f = fx["stage_S0_encoder"]
    assert "never saved" in f["weights"] and not [k for k in f["arrays"] if k.startswith("param.")]
    assert not [k for k in np.load(tmp_path / "fixtures" / "stage_S0_encoder.npz").files if k.startswith("param.")]


def test_THE_NUMPY_REFERENCE_MATCHES_CENTRAL_DIFFERENCES_COORDINATE_BY_COORDINATE():
    """The analytic derivatives, independently of MLX: every policy-head tensor element and F at a repeated gather
    index, against float64 central differences."""
    rng = np.random.default_rng(3)
    p = {"conv.weight": rng.normal(0, 1, (2, 1, 1, 3)), "conv.bias": rng.normal(0, .1, 2),
         "bn.weight": rng.uniform(.5, 1.5, 2), "bn.bias": rng.uniform(.5, 1, 2), "fc.weight": rng.normal(0, 1, (64, 2)),
         "fc.bias": rng.normal(0, .1, 64), "out.weight": rng.normal(0, 1, (1, 64)), "out.bias": rng.normal(0, .1, 1),
         "bn.running_mean": rng.normal(0, .3, 2), "bn.running_var": rng.uniform(.5, 2, 2), "bn.eps": np.array(1e-5)}
    F = rng.normal(0, 1, (2, 24, 24, 3))
    rows = np.array([[3, 3, 0, 5], [3, 7, 0, 0]])
    cols = np.array([[4, 4, 0, 6], [4, 1, 0, 0]])
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=float)
    t = rng.uniform(.1, 1, (2, 4)) * mask
    t /= t.sum(axis=1, keepdims=True)
    w = np.array([0.7, 1.3])
    _, g, _ = K.np_policy_head(p, F, rows, cols, mask, t, w)
    eps = 1e-6
    for k in K.PH_KEYS:
        for i in range(p[k].size):
            q1, q2 = {**p, k: p[k].copy()}, {**p, k: p[k].copy()}
            q1[k].flat[i] += eps
            q2[k].flat[i] -= eps
            fd = (K.np_policy_head(q1, F, rows, cols, mask, t, w, grads=False)[0] -
                  K.np_policy_head(q2, F, rows, cols, mask, t, w, grads=False)[0]) / (2 * eps)
            assert abs(fd - g[k].flat[i]) <= 1e-6 * (abs(g[k].flat[i]) + 1), (k, i)
    for idx in [(0, 3, 4, 1), (1, 3, 4, 2), (0, 0, 0, 0), (1, 0, 0, 1), (0, 10, 10, 0)]:   # (3,4): three uses
        F1, F2 = F.copy(), F.copy()
        F1[idx] += eps
        F2[idx] -= eps
        fd = (K.np_policy_head(p, F1, rows, cols, mask, t, w, grads=False)[0] -
              K.np_policy_head(p, F2, rows, cols, mask, t, w, grads=False)[0]) / (2 * eps)
        assert abs(fd - g["F"][idx]) <= 1e-6 * (abs(g["F"][idx]) + 1), idx
    assert g["F"][0, 10, 10].tolist() == [0.0, 0.0, 0.0]                  # never gathered: no gradient


def test_A_FORWARD_FAULT_IN_ONE_STAGE_IS_LOCATED_THERE_and_its_fixture_saved(kit, tmp_path, monkeypatch):
    real = K._gather

    def gather(x, rows, cols):
        out = real(x, rows, cols)
        return out * 1.01 if K.CURRENT["device"] == "gpu" else out

    monkeypatch.setattr(K, "_gather", gather)
    code, rec = _go(kit, tmp_path)
    assert code == K.EXIT_COMPLETED, rec.get("void")
    for name in ("S0", "S21"):
        fd = rec["results"][name]["first_disagreement"]
        assert fd["cpu"] == {"forward": None, "backward": None}
        assert fd["gpu"]["forward"] == "gather" and fd["gpu"]["backward"] == "gather"
    fx = [json.loads(x) for x in (tmp_path / "ev" / "fixtures.jsonl").read_text().splitlines()]
    assert sorted(f["name"] for f in fx) == ["stage_S0_gather", "stage_S21_gather"]
    for f in fx:
        path = tmp_path / "fixtures" / f"{f['name']}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"] and f["component"] == "stage gather"
        z = np.load(path)
        assert {"input", "cotangent", "reference_output", "rows_c", "cols_c"} <= set(z.files)
        assert not [k for k in z.files if k.startswith("param.")]          # the gather reads no tensor
        assert f["tolerances"]["cross_rel_tol"] == K.FROZEN["cross_rel_tol"]


def test_A_FAULT_IN_THE_MASKED_LOG_SOFTMAX_shows_in_the_full_model_the_stage_and_the_reference(kit, tmp_path,
                                                                                               monkeypatch):
    real = trainer.compute_masked_log_probs

    def lp(logits, mask):
        out = real(logits, mask)
        return out * 1.01 if K.CURRENT["device"] == "gpu" else out

    monkeypatch.setattr(trainer, "compute_masked_log_probs", lp)
    code, rec = _go(kit, tmp_path)
    assert code == K.EXIT_COMPLETED, rec.get("void")
    full = _j(tmp_path / "ev" / "full" / "S21.json")
    assert all(all(v["bitwise"] for v in e.values()) for e in full["devices"]["cpu"]["grads_vs_reference"])
    assert not all(all(v["within_tol"] for v in e.values()) for e in full["devices"]["gpu"]["grads_vs_reference"])
    fd = rec["results"]["S21"]["first_disagreement"]["gpu"]
    assert fd["forward"] == "log_softmax" and fd["backward"] == "log_softmax"
    assert rec["results"]["S21"]["reference_disagrees"] == {"cpu": False, "gpu": True}
    names = {json.loads(x)["name"] for x in (tmp_path / "ev" / "fixtures.jsonl").read_text().splitlines()}
    assert {"stage_S21_log_softmax", "reference_real_S21"} <= names
    z = np.load(tmp_path / "fixtures" / "reference_real_S21.npz")
    assert {k for k in z.files if k.startswith("param.")} <= {f"param.{k}" for k in K.PH_KEYS} | {
        "param.bn.running_mean", "param.bn.running_var", "param.bn.eps"}            # the policy head only


def test_A_NON_REPEATABLE_STAGE_IS_FLAGGED_even_within_the_tolerance(kit, tmp_path, monkeypatch):
    real, calls = K._gather, []

    def gather(x, rows, cols):
        calls.append(1)
        out = real(x, rows, cols)
        return out * (1 + 1e-7 * len(calls)) if K.CURRENT["device"] == "gpu" else out

    monkeypatch.setattr(K, "_gather", gather)
    code, rec = _go(kit, tmp_path)
    assert code == K.EXIT_COMPLETED, rec.get("void")
    st = _j(tmp_path / "ev" / "stages" / "S0.json")
    g = st["forward"]["gpu"]["gather"]
    assert all(v["within_tol"] for v in g["vs_reference"]) and g["repeats_bitwise"] == [False]
    assert rec["results"]["S0"]["first_disagreement"]["gpu"]["forward"] == "gather"


def test_EARLIER_RESULTS_SURVIVE_A_LATER_NON_FINITE_STOP(kit, tmp_path, monkeypatch):
    real = K.synthetic_cases

    def cases(seed):
        out = real(seed)
        name, ph, F, *rest = out[-1]
        F = F.copy()
        F[0, 0, 0, 0] = np.nan
        return out[:-1] + [("poisoned", ph, F, *rest)]

    monkeypatch.setattr(K, "synthetic_cases", cases)
    code, rec = _go(kit, tmp_path)
    assert code == K.EXIT_VOID and rec["void"]["step"] == "non_finite" and "poisoned" in rec["void"]["detail"]
    ev = tmp_path / "ev"
    for rel in ("states.json", "full/S0.json", "full/S21.json", "stages/S0.json", "stages/S21.json",
                "reference/real_S0.json", "reference/real_S21.json", "reference/padding.json"):
        assert (ev / rel).exists(), rel
    assert set(rec["files"]["written"]) >= {"full/S0.json", "reference/padding.json"}


def test_SIGTERM_is_a_VOID_with_a_record_and_the_files_so_far(kit, tmp_path, monkeypatch):
    real = K.localize

    def localize(state_name, *a, **kw):
        out = real(state_name, *a, **kw)
        if state_name == "S0":
            os.kill(os.getpid(), signal.SIGTERM)
        return out

    monkeypatch.setattr(K, "localize", localize)
    previous = signal.getsignal(signal.SIGTERM)
    code, rec = _go(kit, tmp_path)
    assert signal.getsignal(signal.SIGTERM) == previous
    assert code == K.EXIT_VOID and rec["void"]["step"] == "terminated"
    assert (tmp_path / "ev" / "stages" / "S0.json").exists() and (tmp_path / "ev" / "record.json").exists()


@pytest.mark.parametrize("case, step", [("corpus_hash", "corpus"), ("reference_hash", "reference"),
                                        ("repeat_record", "repeat"), ("repeat_digests", "repeat"),
                                        ("ids_digest", "corpus"), ("checkpoint", "checkpoint"),
                                        ("not_pinned", "networks"), ("plan", "plan"), ("devices", "plan"),
                                        ("adam_defaults", "optimizer")])
def test_A_BROKEN_INPUT_OR_IDENTITY_is_a_VOID(kit, tmp_path, monkeypatch, case, step):
    spec, over = dict(kit["spec"]), {}
    if case == "corpus_hash":
        over["read_corpus_json"] = lambda: kit["rig"]["corpus"]["raw"] + b" "
    elif case == "reference_hash":
        over["read_reference"] = lambda: kit["rig"]["ref"]["raw"] + b" "
    elif case == "repeat_record":
        over["read_repeat_record"] = lambda: kit["rec"] + b" "
    elif case == "repeat_digests":
        over["read_repeat_digests"] = lambda: kit["dig"] + b" "
    elif case == "ids_digest":
        spec["training_ids_sha256"] = "0" * 64
    elif case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    elif case == "not_pinned":
        over["load_network"] = lambda: AlphaZeroNetwork(hidden=16, n_blocks=1)
    elif case == "plan":
        spec["states"] = [spec["states"][1], spec["states"][0]]
    elif case == "devices":
        spec["devices"] = ["gpu", "cpu"]
    elif case == "adam_defaults":
        monkeypatch.setattr(K, "ADAM_DEFAULTS", {**K.ADAM_DEFAULTS, "eps": 1e-7})
    code, rec = _go(kit, tmp_path, spec, **over)
    assert code == K.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert not (tmp_path / "ev" / "full").exists() and not (tmp_path / "fixtures").exists()


def test_AN_EXISTING_FIXTURE_DIRECTORY_IS_NEVER_REUSED(kit, tmp_path):
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "keep").write_text("keep")
    with pytest.raises(K.Void, match="exists"):
        _go(kit, tmp_path)
    assert (tmp_path / "fixtures" / "keep").read_text() == "keep"


def test_AGREE_is_bitwise_and_tolerance_aware():
    s = {"cross_rel_tol": 1e-3, "cross_abs_tol": 1e-6}
    b = np.array([1.0, -2.0, 0.5])
    assert K.agree(b.copy(), b, s) == {"bitwise": True, "max_abs": 0.0, "scale": 2.0, "within_tol": True}
    assert K.agree(b + 2e-3, b, s)["within_tol"] is True and K.agree(b + 2e-3, b, s)["bitwise"] is False
    assert K.agree(b + 2.1e-3, b, s)["within_tol"] is False
    assert K.agree(b[:2], b, s)["within_tol"] is False


def test_THE_FROZEN_SPEC_IS_THE_CARD_S():
    shared = ("corpus_sha256", "teacher_digest", "reference_sha256", "reference_student_digest",
              "training_ids_sha256", "training_records", "steps", "batch_size", "learning_rate", "anchor_weight",
              "l2_weight", "clip_norm", "seed")
    assert {k: K.FROZEN[k] for k in shared} == {k: R.FROZEN[k] for k in shared}
    assert K.FROZEN["repeat_record_sha256"] == hashlib.sha256((K.ROOT / K.REPEAT_RECORD).read_bytes()).hexdigest()
    assert K.FROZEN["repeat_digests_sha256"] == hashlib.sha256((K.ROOT / K.REPEAT_DIGESTS).read_bytes()).hexdigest()
    assert K.FROZEN["states"] == [{"name": "S0", "prefix_updates": 0, "batch_update": 0},
                                  {"name": "S21", "prefix_updates": 21, "batch_update": 21}]
    assert (K.FROZEN["devices"], K.FROZEN["repeats"]) == (["cpu", "gpu"], 3)
    assert (K.FROZEN["cross_rel_tol"], K.FROZEN["cross_abs_tol"], K.FROZEN["fd_eps"], K.FROZEN["fd_rel_tol"],
            K.FROZEN["fd_mlx_rel_tol"]) == (1e-3, 1e-6, 1e-5, 1e-6, 1e-3)
    assert not K.FIXTURE_PARENT.is_relative_to(K.ROOT)


def test_THE_PUBLIC_ENTRY_refuses_with_its_gate_closed_and_with_an_unpinned_card(monkeypatch, tmp_path, capsys):
    assert K.GATE.SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED is False
    existed = (K.ROOT / K.EVIDENCE_DIR).exists()
    ckpt = tmp_path / "x.safetensors"
    ckpt.write_bytes(b"")
    assert K.main(["run", "--checkpoint", str(ckpt)]) == K.EXIT_REFUSED and "is closed" in capsys.readouterr().err
    monkeypatch.setattr(K.GATE, "SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED", True)
    monkeypatch.setattr(K, "CARD_SHA256", "1" * 64)
    assert K.main(["run", "--checkpoint", str(ckpt)]) == K.EXIT_REFUSED
    assert "not the pinned card" in capsys.readouterr().err
    assert K.main(["run"]) == K.EXIT_USAGE and (K.ROOT / K.EVIDENCE_DIR).exists() == existed


def test_THE_CARD_IS_PINNED():
    assert hashlib.sha256((K.ROOT / K.CARD).read_bytes()).hexdigest() == K.CARD_SHA256
