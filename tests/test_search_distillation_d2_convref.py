"""The convolution-reference diagnostic (`scripts/GPU/alphazero/search_distillation_d2_convref.py`): the float64
reference's correctness (a brute-force loop, MLX's own convolution, the adjoint self-check), the capture's fidelity
(the instrumentation adds nothing and restores everything; a capture that omits the L2 contribution or perturbs the
graph is caught), each stop, the classification, retention of every earlier result, the refusals, and the public entry
plus the capture and reference in the REAL state (the real corpus through the training-only reader, the real incumbent
and entry 32's real evidence, all read-only). Both device labels run on the CPU here ("gpu" is remapped), so every
honest comparison agrees to the bit and every disagreement is an injected fault, on the networks built under the "gpu"
label. The rig tests use the earlier diagnostics' fixture rig (a small 1-block network).

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture remaps the devices and
redirects the evidence AND the fixture parent for every in-process test. The checks of the REAL setting and of MLX
already being loaded need a fresh process; those subprocesses redirect both and replace `conclude` with a stub, so
even a mutant that removes every check before it cannot reach a real run."""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from scripts.GPU.alphazero import search_distillation_d2_convref as C  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_convref_authorization as GATE  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_gradcheck as K  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_tf32path as E  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_train as T  # noqa: E402
from tests.test_search_distillation_d2_gradcheck import kit  # noqa: E402,F401
from tests.test_search_distillation_d2_gradrepeat import done, rig  # noqa: E402,F401
from tests.test_search_distillation_d2_gradtrace import ref, reproduced  # noqa: E402,F401
from tests.test_search_distillation_d2_measure import _net  # noqa: E402
from tests.test_search_distillation_d2_train import corpus  # noqa: E402,F401

RIG_TARGET = "encoder.blocks.0.conv2.weight"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(K, "DEVICES", {"cpu": mx.cpu, "gpu": mx.cpu})
    monkeypatch.setattr(C, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setattr(C, "FIXTURE_PARENT", tmp_path / "fixture-parent")
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(C, "_mlx_loaded", lambda: False)          # this process has MLX; the subprocesses test it
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED", True)
    return isolated


def _checkpoint() -> pathlib.Path:
    """The real incumbent, from the main checkout (the parent of the shared git directory), as the wrapper finds it."""
    common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=C.ROOT,
                                     text=True).strip()
    return pathlib.Path(common).parent / M.CHECKPOINT


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


# ─────────────────────────────── the float64 reference ───────────────────────────────

def _brute_wgrad(X, G, kshape):
    O, kh, kw, I = kshape
    B, H, W, _ = X.shape
    dW = np.zeros(kshape)
    for o in range(O):
        for a in range(kh):
            for b in range(kw):
                for c in range(I):
                    s = 0.0
                    for n in range(B):
                        for y in range(H):
                            for x in range(W):
                                yy, xx = y + a - kh // 2, x + b - kw // 2
                                if 0 <= yy < H and 0 <= xx < W:
                                    s += G[n, y, x, o] * X[n, yy, xx, c]
                    dW[o, a, b, c] = s
    return dW


def _brute_forward(X, Wt):
    O, kh, kw, I = Wt.shape
    B, H, W, _ = X.shape
    out = np.zeros((B, H, W, O))
    for n in range(B):
        for y in range(H):
            for x in range(W):
                for a in range(kh):
                    for b in range(kw):
                        yy, xx = y + a - kh // 2, x + b - kw // 2
                        if 0 <= yy < H and 0 <= xx < W:
                            out[n, y, x] += Wt[:, a, b, :] @ X[n, yy, xx, :]
    return out


def _tiny(seed=0, B=2, H=4, W=5, I=3, O=2):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((B, H, W, I)), rng.standard_normal((B, H, W, O)),
            rng.standard_normal((O, 3, 3, I)))


def test_THE_REFERENCE_EQUALS_A_BRUTE_FORCE_LOOP_for_the_weight_gradient_and_the_forward():
    X, G, Wt = _tiny()
    assert np.allclose(C.np_conv_weight_grad(X, G, Wt.shape), _brute_wgrad(X, G, Wt.shape), rtol=1e-12, atol=1e-12)
    assert np.allclose(C.np_conv_forward(X, Wt), _brute_forward(X, Wt), rtol=1e-12, atol=1e-12)
    b = np.array([0.5, -1.0])
    assert np.allclose(C.np_conv_forward(X, Wt, b), _brute_forward(X, Wt) + b, rtol=1e-12, atol=1e-12)


def test_THE_REFERENCE_AGREES_WITH_MLX_s_CONVOLUTION_on_the_CPU_in_its_layouts():
    """nn.Conv2d (padding 1): its output against the float64 forward, and its weight VJP (the stage VJP helper the run
    uses) against the float64 weight gradient, in float32 precision."""
    X, G, _ = _tiny(1, B=3, H=6, W=6, I=4, O=5)
    with K.on("cpu"):
        mx.random.seed(3)
        conv = nn.Conv2d(4, 5, kernel_size=3, padding=1)
        Wt, bias = np.array(conv.weight), np.array(conv.bias)
        y = np.array(conv(mx.array(X.astype(np.float32))))
        loc = np.array(C._local(conv, mx.array(X.astype(np.float32)), mx.array(G.astype(np.float32))))
    assert np.abs(y - C.np_conv_forward(X.astype(np.float32), Wt, bias)).max() < 1e-5
    ref64 = C.np_conv_weight_grad(X.astype(np.float32), G.astype(np.float32), Wt.shape)
    assert np.abs(loc - ref64).max() < 1e-5 * np.abs(ref64).max()


def test_THE_ADJOINT_SELF_CHECK_HOLDS_and_catches_a_flipped_or_transposed_reference():
    X, G, Wt = _tiny(2)
    spec = C.FROZEN
    dW = C.np_conv_weight_grad(X, G, Wt.shape)
    assert C.adjoint_check(X, G, dW, spec)["ok"]
    assert not C.adjoint_check(X, G, dW[:, ::-1, ::-1, :], spec)["ok"]          # a flipped kernel
    assert not C.adjoint_check(X, G, dW * (1 + 1e-6), spec)["ok"]               # a scale error far below 1e-3


# ─────────────────────────────── the instrumentation ───────────────────────────────

def test_THE_INSTRUMENTATION_ADDS_NO_PARAMETER_AND_RESTORES_THE_CLASS_even_on_an_exception():
    with K.on("cpu"):
        net = _net()
        convs = [net.encoder.blocks[0].conv2]
        before = sorted(k for k, _ in __import__("mlx.utils").utils.tree_flatten(net.parameters()))
        with pytest.raises(RuntimeError):
            with C.instrumented(convs) as slots:
                assert type(convs[0]).__name__ == "ProbedConv2d" and isinstance(convs[0], nn.Conv2d)
                assert sorted(k for k, _ in __import__("mlx.utils").utils.tree_flatten(net.parameters())) == before
                slots[0]["probe"] = mx.zeros((1, 24, 24, 16))
                convs[0](mx.zeros((1, 24, 24, 16)))
                assert slots[0]["x"].shape == (1, 24, 24, 16)
                raise RuntimeError("boom")
        assert type(convs[0]) is nn.Conv2d and "_slot" not in vars(convs[0])
        assert sorted(k for k, _ in __import__("mlx.utils").utils.tree_flatten(net.parameters())) == before


# ─────────────────────────────── the rig ───────────────────────────────

def _spec(kit, **over):                                                 # noqa: F811
    shared = ("corpus_sha256", "teacher_digest", "training_ids_sha256", "training_games", "held_out_games",
              "training_records", "passes", "steps", "batch_size", "learning_rate")
    return {**C.FROZEN, **{k: kit["spec"][k] for k in shared}, "targets": [RIG_TARGET],
            "reproduce_failing": [], "repeats": 2, **over}


@pytest.fixture
def rigged(kit, monkeypatch):                                           # noqa: F811
    """Entry 32's record.json as the rig sees it: the rig's own update-0 draw, pinned by its own hash."""
    spec = _spec(kit)
    draw = K.draw_batches(spec["training_records"], T.recipe_for(spec), 1)[0]
    real = json.loads((C.ROOT / C.ENTRY32 / "record.json").read_text())
    fake = json.dumps({**real, "batches": {"0": draw}}).encode()
    monkeypatch.setitem(C.PINNED, f"{C.ENTRY32}/record.json", M._sha(fake))
    return {"kit": kit, "record": fake}


def _seams(rigged, make=_net, **over):
    rig = rigged["kit"]["rig"]

    def read_repo(rel):
        return rigged["record"] if rel == f"{C.ENTRY32}/record.json" else (C.ROOT / rel).read_bytes()

    base = dict(read_corpus_json=lambda: rig["corpus"]["raw"], checkpoint_sha1=lambda: M.CHECKPOINT_SHA1,
                load_network=make, loaded_modules=lambda: {"fixture": "not a closure"}, read_repo=read_repo)
    base.update(over)
    return C.Seams(**base)


def _go(rigged, tmp_path, spec=None, make=_net, **over):
    return C.conclude(spec or _spec(rigged["kit"]), _seams(rigged, make, **over), tmp_path / "ev", tmp_path / "fx")


def _on_gpu_label(make, apply):
    """A network factory that applies `apply` to networks built under the "gpu" label."""
    def load():
        net = make()
        if K.CURRENT["device"] == "gpu":
            apply(net)
        return net
    return load


class _ScaledWeightGrad(nn.Conv2d):
    """A LOCAL fault: the same forward, bitwise, but the weight's gradient scaled by 1.01."""
    def __call__(self, x):
        w = self.weight
        y = mx.conv2d(x, w + 0.01 * (w - mx.stop_gradient(w)), self.stride, self.padding, self.dilation, self.groups)
        return y + self.bias


class _ScaledCotangent(nn.BatchNorm):
    """An INHERITED fault: bn2's output unchanged, the cotangent passing back through it scaled by 1.01."""
    def __call__(self, x):
        y = super().__call__(x)
        return y + 0.01 * (y - mx.stop_gradient(y))


def _swap(attr, cls):
    def apply(net):
        getattr(net.encoder.blocks[0], attr).__class__ = cls
    return apply


def test_A_CLEAN_RIG_RUN_CAPTURES_FAITHFULLY_and_classifies_NEITHER_ALONE(rigged, tmp_path):
    code, rec = _go(rigged, tmp_path)
    ev, fx = tmp_path / "ev", tmp_path / "fx"
    assert code == C.EXIT_COMPLETED and rec["status"] == "COMPLETED" and rec["outcome"] == "CLASSIFIED", rec.get("void")
    assert json.loads((ev / "record.json").read_text()) == json.loads(json.dumps(rec, sort_keys=True))
    assert rec["step1"]["ok"] and rec["step2"] == {"cpu": {"faithful": True}, "gpu": {"faithful": True}}
    for dev in ("cpu", "gpu"):
        cap = json.loads((ev / "capture" / f"{dev}.json").read_text())
        assert cap["uninstrumented"]["ok"] and cap["instrumented"]["ok"]
        t = cap["targets"][RIG_TARGET]
        assert t["recon_bitwise"] and t["recon_ok"] and t["l2_max_abs"] > 0
        assert t["local_alone_max_abs"] > C.FROZEN["recon_rel_tol"] * t["orig_scale"]     # the L2 term is needed
    r = json.loads((ev / "reference" / f"{RIG_TARGET}.json").read_text())
    assert r["class"] == "NEITHER_ALONE" and r["flags"] == dict.fromkeys(("local_gpu", "local_cpu", "inherited"), False)
    assert r["self_check"]["layout"]["within_tol"] and all(a["ok"] for a in r["self_check"]["adjoint"].values())
    assert r["identical_operands"]["cpu"]["cpu_equals_capture"] and r["identical_operands"]["gpu"]["gpu_equals_capture"]
    assert r["native"]["original_gpu_vs_cpu"]["bitwise"] and r["failed_elements"]["count"] == 0
    assert rec["step3"]["overall"] == "NEITHER_ALONE"
    names = sorted(p.name for p in fx.iterdir())
    assert names == [f"cpu_{RIG_TARGET}.npz", f"gpu_{RIG_TARGET}.npz"]
    entries = _lines(ev / "fixtures.jsonl")
    assert [e["name"] for e in entries] == [f"cpu_{RIG_TARGET}", f"gpu_{RIG_TARGET}"] and rec["fixtures"]
    with np.load(fx / f"cpu_{RIG_TARGET}.npz") as z:                    # the fixture reproduces the local gradient
        assert sorted(z.files) == ["G", "X", "local"]
        with K.on("cpu"):
            conv = _net().encoder.blocks[0].conv2
            again = np.array(C._local(conv, mx.array(z["X"]), mx.array(z["G"])))
        assert np.array_equal(again, z["local"])
    assert rec["optimizer_updates"] == 0 and rec["held_out_records_read"] == 0 and rec["model_state_saved"] is False


def test_A_LOCAL_GPU_FAULT_REPRODUCES_AND_IS_CLASSIFIED_LOCAL_GPU(rigged, tmp_path):
    spec = _spec(rigged["kit"], reproduce_failing=[RIG_TARGET])
    code, rec = _go(rigged, tmp_path, spec, make=_on_gpu_label(_net, _swap("conv2", _ScaledWeightGrad)))
    assert rec["outcome"] == "CLASSIFIED" and rec["step1"]["ok"], (rec.get("void"), rec.get("step1"))
    assert rec["step2"] == {"cpu": {"faithful": True}, "gpu": {"faithful": True}}
    r = json.loads((tmp_path / "ev" / "reference" / f"{RIG_TARGET}.json").read_text())
    assert r["flags"] == {"local_gpu": True, "local_cpu": False, "inherited": False} and r["class"] == "LOCAL_GPU"
    assert r["native"]["G_gpu_vs_cpu"]["bitwise"] and r["native"]["X_gpu_vs_cpu"]["bitwise"]
    assert r["failed_elements"]["count"] > 0 and r["failed_elements"]["recorded"][0]["abs_terms_cpu_ops"] > 0


def test_AN_UPSTREAM_FAULT_IS_CLASSIFIED_INHERITED(rigged, tmp_path, monkeypatch):
    """It changes every tensor upstream, so entry 32's failure set cannot reproduce; the reproduction rule is
    replaced HERE ONLY, to reach the classification."""
    real = C.reproduced
    monkeypatch.setattr(C, "reproduced", lambda blk, spec: {**real(blk, spec), "ok": True})
    code, rec = _go(rigged, tmp_path, make=_on_gpu_label(_net, _swap("bn2", _ScaledCotangent)))
    assert rec["outcome"] == "CLASSIFIED", rec.get("void")
    r = json.loads((tmp_path / "ev" / "reference" / f"{RIG_TARGET}.json").read_text())
    assert r["flags"] == {"local_gpu": False, "local_cpu": False, "inherited": True} and r["class"] == "INHERITED"
    assert not r["native"]["G_gpu_vs_cpu"]["within_tol"] and r["native"]["X_gpu_vs_cpu"]["bitwise"]


def test_WITHOUT_THE_DISCREPANCY_THE_RUN_STOPS_NOT_REPRODUCED_before_any_capture(rigged, tmp_path):
    code, rec = _go(rigged, tmp_path, _spec(rigged["kit"], reproduce_failing=[RIG_TARGET]))
    assert code == C.EXIT_COMPLETED and rec["outcome"] == "NOT_REPRODUCED" and not rec["step1"]["ok"]
    assert "step2" not in rec and "step3" not in rec and not (tmp_path / "fx").exists()
    assert not list((tmp_path / "ev" / "capture").iterdir()) and (tmp_path / "ev" / "full" / "S0_b0.json").exists()


@pytest.mark.parametrize("fault", ["no_l2", "nonzero_probe"])
def test_AN_UNFAITHFUL_CAPTURE_STOPS_before_the_reference(rigged, tmp_path, monkeypatch, fault):
    """Omitting the separately calculated L2 contribution fails the reconstruction; a probe that is not zero changes
    the instrumented gradients. Either stops at CAPTURE_NOT_FAITHFUL, with the fixtures already saved kept."""
    if fault == "no_l2":
        monkeypatch.setattr(C, "_l2_term", lambda w, recipe: mx.zeros_like(w))
    else:
        real = C.instrumented

        def shifted(convs):
            import contextlib

            @contextlib.contextmanager
            def cm():
                with real(convs) as slots:
                    class S(dict):
                        def __setitem__(self, k, v):
                            super().__setitem__(k, v + 1e-3 if k == "probe" else v)
                    for i, s in enumerate(slots):
                        slots[i] = S(s)
                        object.__setattr__(convs[i], "_slot", slots[i])
                    yield slots
            return cm()
        monkeypatch.setattr(C, "instrumented", shifted)
    code, rec = _go(rigged, tmp_path)
    assert code == C.EXIT_COMPLETED and rec["outcome"] == "CAPTURE_NOT_FAITHFUL", rec.get("void")
    assert "step3" not in rec and not list((tmp_path / "ev" / "reference").iterdir())
    cap = json.loads((tmp_path / "ev" / "capture" / "cpu.json").read_text())
    if fault == "no_l2":
        assert cap["uninstrumented"]["ok"] and cap["instrumented"]["ok"] and not cap["targets"][RIG_TARGET]["recon_ok"]
    else:
        assert not cap["instrumented"]["ok"]
    assert len(list((tmp_path / "fx").iterdir())) == 2


def test_A_LAYOUT_ERROR_IN_THE_REFERENCE_IS_A_VOID(rigged, tmp_path, monkeypatch):
    real = C.np_conv_forward
    monkeypatch.setattr(C, "np_conv_forward", lambda X, Wt, bias=None: real(X, Wt[:, ::-1, ::-1, :], bias))
    code, rec = _go(rigged, tmp_path)
    assert code == C.EXIT_VOID and rec["void"]["step"] == "reference" and rec["outcome"] is None


def test_SIGTERM_DURING_THE_REFERENCE_IS_A_VOID_and_the_capture_and_fixtures_survive(rigged, tmp_path, monkeypatch):
    def term(*a, **k):
        raise C.Terminated("SIGTERM")
    monkeypatch.setattr(C, "np_conv_weight_grad", term)
    code, rec = _go(rigged, tmp_path)
    ev = tmp_path / "ev"
    assert code == C.EXIT_VOID and rec["void"]["step"] == "terminated" and rec["outcome"] is None
    assert json.loads((ev / "record.json").read_text())["status"] == "VOID"
    assert (ev / "capture" / "cpu.json").exists() and (ev / "capture" / "gpu.json").exists()
    assert len(_lines(ev / "fixtures.jsonl")) == 2 and len(list((tmp_path / "fx").iterdir())) == 2
    assert [x["event"] for x in _lines(ev / "progress.jsonl")][-3:] == ["capture", "capture", "end"]


@pytest.mark.parametrize("case, step", [("draw", "batch"), ("checkpoint", "checkpoint"), ("entry32", "entry32"),
                                        ("target", "plan")])
def test_A_BROKEN_INPUT_IS_A_VOID_BEFORE_ANY_NETWORK_IS_LOADED(rigged, tmp_path, monkeypatch, case, step):
    loads = []
    over, spec = {}, None
    if case == "draw":
        monkeypatch.setitem(C.PINNED, f"{C.ENTRY32}/record.json", M._sha(b'{"batches": {"0": [1]}}'))
        rigged = {**rigged, "record": b'{"batches": {"0": [1]}}'}
    elif case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    elif case == "entry32":
        monkeypatch.setitem(C.PINNED, f"{C.ENTRY32}/full.jsonl", "0" * 64)
    else:
        spec = _spec(rigged["kit"], targets=["encoder.blocks.0.conv1.weight"])
    code, rec = _go(rigged, tmp_path, spec, make=lambda: loads.append(1) or _net(), **over)
    assert code == C.EXIT_VOID and rec["void"]["step"] == step and loads == []


def test_THE_CLASSIFICATION_TABLE_is_the_card_s():
    want = {(True, False, False): "LOCAL_GPU", (False, True, False): "LOCAL_CPU", (False, False, True): "INHERITED",
            (True, False, True): "LOCAL_GPU_AND_INHERITED", (False, True, True): "LOCAL_CPU_AND_INHERITED",
            (True, True, False): "BOTH_KERNELS", (True, True, True): "BOTH_KERNELS",
            (False, False, False): "NEITHER_ALONE"}
    for (g, c, i), name in want.items():
        assert C.classify({"local_gpu": g, "local_cpu": c, "inherited": i}) == name
    assert C.overall({"a": "LOCAL_GPU", "b": "LOCAL_GPU"}) == "LOCAL_GPU"
    assert C.overall({"a": "LOCAL_GPU", "b": "INHERITED"}) == "MIXED"


def test_THE_REPRODUCTION_RULE_wants_exactly_the_expected_failures_and_no_CPU_failure_of_its_own():
    spec = {"devices": ["cpu", "gpu"], "repeats": 3, "reproduce_failing": C.TARGETS}
    six = [f"evaluation {i}: {t}" for i in range(3) for t in C.TARGETS]

    def blk(gpu, cpu=()):
        return {"devices": {"gpu": {"failures": list(gpu)}, "cpu": {"failures": list(cpu)}}}
    assert C.reproduced(blk(six), spec)["ok"]
    assert not C.reproduced(blk(six[:-1]), spec)["ok"]
    assert not C.reproduced(blk(six + ["evaluation 0: values"]), spec)["ok"]
    assert not C.reproduced(blk(six, ["repeat 1: logits"]), spec)["ok"]
    assert C.reproduced(blk(six, ["evaluation 0: x"]), spec)["ok"]         # the CPU against itself is not judged


# ─────────────────────────────── refusals ───────────────────────────────

def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED is False
    assert C.main(["run", "--checkpoint", str(_checkpoint())]) == C.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err and not (isolated / "evidence").exists()


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_DEVICES_EVIDENCE_OR_FIXTURES(isolated):
    assert K.DEVICES == {"cpu": mx.cpu, "gpu": mx.cpu}
    assert C.EVIDENCE_DIR.startswith(str(isolated)) and str(C.FIXTURE_PARENT).startswith(str(isolated))


def test_ANYTHING_BUT_run_WITH_A_CHECKPOINT_IS_USAGE():
    assert C.main([]) == C.main(["run"]) == C.main(["go", "--checkpoint", "x"]) == C.EXIT_USAGE


def test_THE_PINS_ARE_THE_FILES_ON_DISK_and_the_tolerances_are_frozen():
    assert M._sha((C.ROOT / C.CARD).read_bytes()) == C.CARD_SHA256
    for rel, sha in C.PINNED.items():
        assert M._sha((C.ROOT / rel).read_bytes()) == sha, rel
    assert set(C.CODE.values()) <= set(C.PINNED)
    f = C.FROZEN
    assert (f["cross_rel_tol"], f["cross_abs_tol"], f["recon_rel_tol"], f["adjoint_rel_tol"]) == (1e-3, 1e-6, 1e-6, 1e-9)
    assert f["targets"] == f["reproduce_failing"] == ["encoder.blocks.1.conv2.weight", "encoder.blocks.2.conv2.weight"]
    assert (f["devices"], f["repeats"], f["state"], f["batch"]) == (["cpu", "gpu"], 3, "S0", 0)
    assert all(f[k] == E.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "batch_size", "l2_weight", "seed"))
    old = json.loads((C.ROOT / C.ENTRY32 / "full" / "S0_b0.json").read_text())["devices"]["gpu"]["failures"]
    assert sorted(old) == sorted(f"evaluation {i}: {t}" for i in range(3) for t in C.TARGETS)   # entry 32's six


@pytest.mark.parametrize("pin", [C.CODE["tf32path"], f"{C.ENTRY32}/full.jsonl", f"{C.ENTRY32}/record.json"])
def test_A_FILE_THAT_IS_NOT_THE_PINNED_BYTES_REFUSES(opened, monkeypatch, pin, capsys):
    monkeypatch.setitem(C.PINNED, pin, "0" * 64)
    assert C.main(["run", "--checkpoint", str(_checkpoint())]) == C.EXIT_REFUSED
    assert pin in capsys.readouterr().err


def test_AN_UNPINNED_CARD_REFUSES(opened, monkeypatch, capsys):
    monkeypatch.setattr(C, "CARD_SHA256", "0" * 64)
    assert C.main(["run", "--checkpoint", str(_checkpoint())]) == C.EXIT_REFUSED
    assert "pinned card" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "1", "", " 0", "false"])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gate_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert C.main(["run", "--checkpoint", str(_checkpoint())]) == C.EXIT_REFUSED
    assert "MLX_ENABLE_TF32" in capsys.readouterr().err


def test_MLX_ALREADY_LOADED_REFUSES_even_with_the_gate_open(opened, monkeypatch, capsys):
    monkeypatch.setattr(C, "_mlx_loaded", lambda: True)
    assert C.main(["run", "--checkpoint", str(_checkpoint())]) == C.EXIT_REFUSED
    assert "imported" in capsys.readouterr().err


def test_AN_EXISTING_EVIDENCE_DIRECTORY_OR_A_RELATIVE_CHECKPOINT_REFUSE(opened, monkeypatch, capsys):
    assert C.main(["run", "--checkpoint", "checkpoints/x.safetensors"]) == C.EXIT_REFUSED
    assert "absolute" in capsys.readouterr().err
    (opened / "evidence").mkdir()
    assert C.main(["run", "--checkpoint", str(_checkpoint())]) == C.EXIT_REFUSED
    assert "one-shot" in capsys.readouterr().err


PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import search_distillation_d2_convref as C
from scripts.GPU.alphazero import search_distillation_d2_convref_authorization as GATE
print("MLX_AFTER_IMPORT", C._mlx_loaded(), flush=True)
_tmp = pathlib.Path(tempfile.mkdtemp())
C.EVIDENCE_DIR = str(_tmp / "evidence")
C.FIXTURE_PARENT = _tmp / "fixture-parent"
def stub(spec, seams, out, fixture_dir, head=None):     # never a real run, whatever a mutant does above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB", "head": head}
C.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(C.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    argv = repr(["run", "--checkpoint", str(_checkpoint())])
    return subprocess.run([sys.executable, "-c", PRELUDE + body + f"\nsys.exit(C.main({argv}))\n"], cwd=C.ROOT,
                          env=env, capture_output=True, text=True, timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_and_the_gate_open_every_real_check_passes():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


# ─────────────────────────────── the REAL state ───────────────────────────────

def test_THE_PUBLIC_ENTRY_IN_THE_REAL_STATE_STOPS_NOT_REPRODUCED_when_gpu_is_the_cpu(opened, capsys):
    """The gate open, the frozen spec, the pinned card, code and evidence, the real corpus (training-only reader) and
    the real incumbent. "gpu" is the CPU here, so entry 32's discrepancy cannot reproduce and the run stops before any
    capture; the CPU's step-1 lines repeat entry 32's exactly (prediction 1's CPU half)."""
    assert C.main(["run", "--checkpoint", str(_checkpoint())]) == C.EXIT_COMPLETED
    ev = opened / "evidence"
    rec = json.loads((ev / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["outcome"] == "NOT_REPRODUCED", rec.get("void")
    assert rec["spec"] == json.loads(json.dumps(C.FROZEN)) and rec["checkpoint"]["sha1"] == M.CHECKPOINT_SHA1
    assert rec["step1"]["gpu_failures"] == [] and rec["step1"]["cpu_own_failures"] == []
    assert rec["entry32"]["cpu"] == {"compared": 3, "lines_equal": True}
    assert rec["batch_draw"] == json.loads((C.ROOT / C.ENTRY32 / "record.json").read_text())["batches"]["0"]
    assert rec["held_out_records_read"] == 0 and not (opened / "fixture-parent").exists()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["outcome"] == "NOT_REPRODUCED" and out["fixture_dir"].startswith(str(opened / "fixture-parent"))


def test_THE_CAPTURE_AND_REFERENCE_IN_THE_REAL_STATE_are_faithful_and_self_checked(opened):
    """The real sources and incumbent through `conclude`, with the reproduction expectation emptied HERE ONLY ("gpu" is
    the CPU, so nothing fails): both targets captured bitwise-faithfully on each label, the L2 contribution large
    enough that omitting it would fail the reconstruction, and the float64 reference passing its self-checks on the
    real operands."""
    spec = {**C.FROZEN, "reproduce_failing": []}
    code, rec = C.conclude(spec, C._real_seams(_checkpoint()), opened / "ev", opened / "fx")
    assert rec["outcome"] == "CLASSIFIED", rec.get("void")
    for dev in ("cpu", "gpu"):
        cap = json.loads((opened / "ev" / "capture" / f"{dev}.json").read_text())
        assert cap["uninstrumented"]["ok"] and cap["instrumented"]["ok"] and cap["instrumented"]["grads_compared"] == 60
        for t in C.TARGETS:
            x = cap["targets"][t]
            assert x["recon_bitwise"] and x["X"]["shape"] == x["G"]["shape"] == [64, 24, 24, 128]
            assert x["local_alone_max_abs"] > C.FROZEN["recon_rel_tol"] * x["orig_scale"]
    for t in C.TARGETS:
        r = json.loads((opened / "ev" / "reference" / f"{t}.json").read_text())
        assert r["self_check"]["layout"]["within_tol"] and all(a["ok"] for a in r["self_check"]["adjoint"].values())
        assert r["class"] == "NEITHER_ALONE" and r["kshape"] == [128, 3, 3, 128]
    assert sorted(p.name for p in (opened / "fx").iterdir()) == sorted(f"{d}_{t}.npz" for d in ("cpu", "gpu")
                                                                       for t in C.TARGETS)
