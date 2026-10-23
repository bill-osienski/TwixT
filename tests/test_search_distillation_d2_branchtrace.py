"""The encoder branch trace (`scripts/GPU/alphazero/search_distillation_d2_branchtrace.py`): the float64 references'
correctness (the transposed convolution and BatchNorm against a brute-force loop and MLX, the adjoint self-check), the
instrumentation's fidelity (it adds no parameter and restores every class; a perturbing trace is caught), each
outcome (a local derivative fault, a local forward fault, a branch change explained by the accounting), each stop,
retention, the refusals, and the public entry plus a full trace in the REAL state (the real corpus through the
training-only reader, the real incumbent and entries 32-33's real evidence, read-only). Both device labels run on the
CPU here ("gpu" is remapped), so honest comparisons agree to the bit and every disagreement is an injected fault on the
networks built under the "gpu" label. The rig tests use the earlier diagnostics' 1-block fixture network.

Reproduction is against entry 33's digests; on the rig (and for the full real-state trace, where "gpu" is the CPU) a
first pass records the digests this rig produces and a second pass is served entry-33 files holding them, pinned by
their own hashes. That substitution is in these tests only.

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture remaps the devices and
redirects the evidence AND the fixture parent for every in-process test; the fresh-process checks redirect both and
replace `conclude` with a stub."""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from scripts.GPU.alphazero import search_distillation_d2_branchtrace as B  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_branchtrace_authorization as GATE  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_convref as C  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_gradcheck as K  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
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
    monkeypatch.setattr(B, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setattr(B, "FIXTURE_PARENT", tmp_path / "fixture-parent")
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(B, "_mlx_loaded", lambda: False)
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED", True)
    return isolated


def _checkpoint() -> pathlib.Path:
    common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=B.ROOT,
                                     text=True).strip()
    return pathlib.Path(common).parent / M.CHECKPOINT


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


# ─────────────────────────────── the float64 references ───────────────────────────────

def _brute_input_grad(G, Wt):
    O, kh, kw, I = Wt.shape
    Bn, H, W, _ = G.shape
    gx = np.zeros((Bn, H, W, I))
    for n in range(Bn):
        for y in range(H):
            for x in range(W):
                for a in range(kh):
                    for b in range(kw):
                        i, j = y + a - kh // 2, x + b - kw // 2
                        if 0 <= i < H and 0 <= j < W:
                            gx[n, i, j] += G[n, y, x] @ Wt[:, a, b, :]
    return gx


def test_THE_TRANSPOSED_CONVOLUTION_EQUALS_A_BRUTE_FORCE_LOOP_and_MLX_s_input_gradient():
    rng = np.random.default_rng(0)
    G, Wt = rng.standard_normal((2, 4, 5, 3)), rng.standard_normal((3, 3, 3, 2))
    assert np.allclose(B.np_conv_input_grad(G, Wt), _brute_input_grad(G, Wt), rtol=1e-12, atol=1e-12)
    X = rng.standard_normal((3, 6, 6, 4)).astype(np.float32)
    Gy = rng.standard_normal((3, 6, 6, 5)).astype(np.float32)
    with K.on("cpu"):
        mx.random.seed(1)
        conv = nn.Conv2d(4, 5, kernel_size=3, padding=1)
        _, gx, _ = K._vjp(conv, lambda a: conv(a), mx.array(X), mx.array(Gy))
        gx, Wn = np.array(gx), np.array(conv.weight)
    ref = B.np_conv_input_grad(Gy, Wn)
    assert np.abs(gx - ref).max() < 1e-5 * np.abs(ref).max()


def test_THE_FLOAT64_BATCHNORM_EQUALS_MLX_s_IN_EVALUATION_MODE_and_its_backward_is_the_scale():
    rng = np.random.default_rng(2)
    with K.on("cpu"):
        bn = nn.BatchNorm(4)
        bn.running_mean = mx.array(rng.standard_normal(4).astype(np.float32))
        bn.running_var = mx.array((rng.random(4) + 0.5).astype(np.float32))
        bn.weight = mx.array(rng.standard_normal(4).astype(np.float32))
        bn.bias = mx.array(rng.standard_normal(4).astype(np.float32))
        bn.eval()
        z = rng.standard_normal((2, 3, 3, 4)).astype(np.float32)
        g = rng.standard_normal((2, 3, 3, 4)).astype(np.float32)
        y = np.array(bn(mx.array(z)))
        _, (gz,) = mx.vjp(lambda a: bn(a), [mx.array(z)], [mx.array(g)])
        c = {"mean": np.array(bn.running_mean), "var": np.array(bn.running_var), "weight": np.array(bn.weight),
             "bias": np.array(bn.bias), "eps": float(bn.eps)}
    assert np.abs(y - B.np_bn(z, c)).max() < 1e-5
    assert np.abs(np.array(gz) - g * B.bn_scale(c)).max() < 1e-5


def test_THE_ADJOINT_SELF_CHECK_HOLDS_and_catches_a_flipped_transposed_convolution(monkeypatch):
    rng = np.random.default_rng(3)
    X, G, Wt = rng.standard_normal((2, 5, 5, 3)), rng.standard_normal((2, 5, 5, 4)), rng.standard_normal((4, 3, 3, 3))
    assert B.adjoint_input_check(X, G, Wt, B.FROZEN)["ok"]
    real = B.np_conv_input_grad
    monkeypatch.setattr(B, "np_conv_input_grad", lambda G, Wt: real(G, Wt[:, ::-1, ::-1, :]))
    assert not B.adjoint_input_check(X, G, Wt, B.FROZEN)["ok"]


def test_MLX_s_RELU_DERIVATIVE_IS_ONE_ONLY_ABOVE_ZERO_as_the_bitwise_check_assumes():
    with K.on("cpu"):
        x = mx.array([-1.0, -0.0, 0.0, 1e-30, 2.0])
        _, (g,) = mx.vjp(lambda a: nn.relu(a), [x], [mx.full((5,), 3.0)])
    assert np.array(g).tolist() == [0.0, 0.0, 0.0, 3.0, 3.0]


# ─────────────────────────────── the instrumentation ───────────────────────────────

def test_THE_INSTRUMENTATION_ADDS_NO_PARAMETER_AND_RESTORES_EVERY_CLASS_even_on_an_exception():
    from mlx.utils import tree_flatten
    with K.on("cpu"):
        net = _net()
        enc = net.encoder
        before = sorted(k for k, _ in tree_flatten(net.parameters()))
        classes = [type(enc), *map(type, enc.blocks)]
        with pytest.raises(RuntimeError):
            with B.instrumented(enc) as traces:
                assert type(enc).__name__.startswith("Traced") and len(traces) == 1 + len(enc.blocks)
                assert sorted(k for k, _ in tree_flatten(net.parameters())) == before
                raise RuntimeError("boom")
        assert [type(enc), *map(type, enc.blocks)] == classes and "_trace" not in vars(enc)
        assert all("_trace" not in vars(b) for b in enc.blocks)
        assert sorted(k for k, _ in tree_flatten(net.parameters())) == before


# ─────────────────────────────── the rig ───────────────────────────────

def _spec(kit, **over):                                                 # noqa: F811
    shared = ("corpus_sha256", "teacher_digest", "training_ids_sha256", "training_games", "held_out_games",
              "training_records", "passes", "steps", "batch_size", "learning_rate")
    s = {**B.FROZEN, **{k: kit["spec"][k] for k in shared}, "targets": [RIG_TARGET], **over}
    s["focus"] = [0, 0, 0, 0]
    return s


def _fake_evidence(draw, got=None, teacher=None):
    """Entries 32-33's files as a rig run sees them: the rig's draw, and (second pass) the digests it produced."""
    got = got or {d: {"grad_digest": "x", "terms": [0.0, 0.0, 0.0], "logits_digest": "x", "values_digest": "x"}
                  for d in ("cpu", "gpu")}
    files = {f"{B.ENTRY32}/record.json": json.dumps({"batches": {"0": draw}}).encode(),
             f"{B.ENTRY33}/record.json": json.dumps({"teacher_values": {"digest": teacher or "x"}}).encode(),
             f"{B.ENTRY33}/full.jsonl": "".join(json.dumps({"device": d, "repeat": 0, **{k: got[d][k] for k in (
                 "logits_digest", "values_digest")}}) + "\n" for d in ("cpu", "gpu")).encode()}
    for d in ("cpu", "gpu"):
        files[f"{B.ENTRY33}/capture/{d}.json"] = json.dumps({"uninstrumented": {
            "grad_digest": got[d]["grad_digest"], "terms": got[d]["terms"]}}).encode()
    return files


def _serve(monkeypatch, base_read, files):
    for rel, raw in files.items():
        monkeypatch.setitem(B.PINNED, rel, M._sha(raw))
    return lambda rel: files[rel] if rel in files else base_read(rel)


def _seams(kit, read_repo, make=_net, **over):                          # noqa: F811
    rig = kit["rig"]
    base = dict(read_corpus_json=lambda: rig["corpus"]["raw"], checkpoint_sha1=lambda: M.CHECKPOINT_SHA1,
                load_network=make, loaded_modules=lambda: {"fixture": "not a closure"}, read_repo=read_repo)
    base.update(over)
    return C.Seams(**base)


def _run(kit, tmp_path, monkeypatch, spec=None, make=_net, reproducing=True, **over):  # noqa: F811
    """One rig run; if `reproducing`, a first pass records this rig's digests and the second is served them."""
    spec = spec or _spec(kit)
    draw = K.draw_batches(spec["training_records"], T.recipe_for(spec), 1)[0]
    base_read = lambda rel: (B.ROOT / rel).read_bytes()                  # noqa: E731
    files = _fake_evidence(draw)
    if reproducing:
        _, first = B.conclude(spec, _seams(kit, _serve(monkeypatch, base_read, files), make, **over),
                              tmp_path / "pass1", tmp_path / "fx1")
        assert first["outcome"] == "NOT_REPRODUCED", first.get("void")
        files = _fake_evidence(draw, {d: first["step1"][d]["got"] for d in ("cpu", "gpu")},
                               first["teacher_values"]["digest"])
    return B.conclude(spec, _seams(kit, _serve(monkeypatch, base_read, files), make, **over), tmp_path / "ev",
                      tmp_path / "fx")


def _on_gpu_label(make, apply):
    def load():
        net = make()
        if K.CURRENT["device"] == "gpu":
            apply(net)
        return net
    return load


class _ScaledBN2Backward(nn.BatchNorm):
    """A LOCAL DERIVATIVE fault: the same forward, the gradient through BN2 scaled by 1.01."""
    def __call__(self, x):
        y = super().__call__(x)
        return y + 0.01 * (y - mx.stop_gradient(y))


class _ShiftedBN2Forward(nn.BatchNorm):
    """A LOCAL FORWARD fault: BN2's output shifted by a constant; its gradient unchanged."""
    def __call__(self, x):
        return super().__call__(x) + 0.05


class _NoisyConv2(nn.Conv2d):
    """Different OPERANDS for the outer ReLU (a fixed tiny perturbation of conv2's output, gradient unchanged), as two
    backends' rounding would give: the ReLU's branches flip only where its input is near zero."""
    def __call__(self, x):
        y = super().__call__(x)
        noise = mx.array(np.random.default_rng(7).standard_normal(y.shape).astype(np.float32))
        return y + 2e-4 * mx.stop_gradient(mx.abs(y).max()) * noise


def _swap(attr, cls):
    def apply(net):
        getattr(net.encoder.blocks[0], attr).__class__ = cls
    return apply


def test_A_CLEAN_RIG_TRACE_IS_FAITHFUL_EVERY_LOCAL_CHECK_PASSES_and_nothing_flips(kit, tmp_path, monkeypatch):  # noqa
    code, rec = _run(kit, tmp_path, monkeypatch)
    ev = tmp_path / "ev"
    assert code == B.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert rec["step1"]["ok"] and rec["step2"] == {"cpu": {"faithful": True}, "gpu": {"faithful": True}}
    assert rec["outcome"] == "NOT_EXPLAINED" and rec["self_check"]["adjoint_input_grad"]["ok"]
    assert rec["local_checks"] == {d: {"forward_failures": [], "backward_failures": []} for d in ("cpu", "gpu")}
    assert all(v["count"] == 0 for v in rec["flips"].values()) and rec["walk_entry"] is None
    for dev in ("cpu", "gpu"):
        t = json.loads((ev / "trace" / f"{dev}.json").read_text())
        assert t["fidelity"]["ok"] and t["fidelity"]["grads_compared"] > 0
    assert sorted(p.name for p in (ev / "stages").iterdir()) == [
        "blocks.0.relu_inner.json", "blocks.0.relu_outer.json", "stem.relu.json"]
    for f in ("walk.json", "focus.json", "accounting.json", "fixtures.jsonl", "record.json"):
        assert (ev / f).exists(), f
    with np.load(tmp_path / "fx" / "cpu_record1.npz") as z:
        assert "grad.blocks.0.s" in z.files and "blocks.0.x_in" in z.files and "input" in z.files
    assert rec["optimizer_updates"] == 0 and rec["held_out_records_read"] == 0 and rec["elapsed_is_a_benchmark"] is False


def test_A_LOCAL_DERIVATIVE_FAULT_IS_NAMED(kit, tmp_path, monkeypatch):  # noqa: F811
    code, rec = _run(kit, tmp_path, monkeypatch, make=_on_gpu_label(_net, _swap("bn2", _ScaledBN2Backward)))
    assert rec["outcome"] == "LOCAL_DERIVATIVE_ERROR", rec.get("void")
    assert rec["local_checks"]["gpu"]["backward_failures"] == ["blocks.0.bn2"]
    assert rec["local_checks"]["cpu"]["backward_failures"] == []


def test_A_LOCAL_FORWARD_FAULT_IS_NAMED(kit, tmp_path, monkeypatch):  # noqa: F811
    code, rec = _run(kit, tmp_path, monkeypatch, make=_on_gpu_label(_net, _swap("bn2", _ShiftedBN2Forward)))
    assert rec["outcome"] == "LOCAL_FORWARD_ERROR", rec.get("void")
    assert rec["local_checks"]["gpu"] == {"forward_failures": ["blocks.0.s"], "backward_failures": []}


def test_DIFFERENT_OPERANDS_AT_A_RELU_ARE_A_BRANCH_CHANGE_that_the_accounting_explains(kit, tmp_path, monkeypatch):  # noqa
    code, rec = _run(kit, tmp_path, monkeypatch, make=_on_gpu_label(_net, _swap("conv2", _NoisyConv2)))
    ev = tmp_path / "ev"
    assert rec["local_checks"]["gpu"] == {"forward_failures": [], "backward_failures": []}, rec.get("void")
    assert rec["flips"]["blocks.0.relu_outer"]["count"] > 0
    st = json.loads((ev / "stages" / "blocks.0.relu_outer.json").read_text())
    f = st["recorded"][0]
    assert f["cpu"]["mask"] != f["gpu"]["mask"] and f["cpu"]["float64_sign_agrees"] and f["gpu"]["float64_sign_agrees"]
    acc = json.loads((ev / "accounting.json").read_text())["targets"][RIG_TARGET]
    assert acc["with_flips"]["within_tol"] and not acc["without_flips"]["within_tol"], acc
    assert rec["outcome"] == "BRANCH_CHANGE"
    assert rec["walk_entry"]["point"] in ("blocks.0.s", "blocks.0.out")


def test_WITHOUT_THE_DIGESTS_THE_RUN_STOPS_NOT_REPRODUCED_before_any_trace(kit, tmp_path, monkeypatch):  # noqa
    code, rec = _run(kit, tmp_path, monkeypatch, reproducing=False)
    assert code == B.EXIT_COMPLETED and rec["outcome"] == "NOT_REPRODUCED" and "step2" in rec and rec["step2"] == {}
    assert not list((tmp_path / "ev" / "trace").iterdir()) and not (tmp_path / "fx").exists()


def test_A_PERTURBING_TRACE_STOPS_TRACE_NOT_FAITHFUL(kit, tmp_path, monkeypatch):  # noqa: F811
    real = B._traced_block_call

    def shifted(self, x):
        return real(self, x) + 1e-3
    monkeypatch.setattr(B, "_TRACED", {})
    monkeypatch.setattr(B, "_traced_block_call", shifted)
    code, rec = _run(kit, tmp_path, monkeypatch)
    assert rec["outcome"] == "TRACE_NOT_FAITHFUL", rec.get("void")
    assert not json.loads((tmp_path / "ev" / "trace" / "cpu.json").read_text())["fidelity"]["ok"]
    assert not list((tmp_path / "ev" / "stages").iterdir()) and len(list((tmp_path / "fx").iterdir())) == 2


def test_A_FAILED_ADJOINT_SELF_CHECK_IS_A_VOID(kit, tmp_path, monkeypatch):  # noqa: F811
    real = B.np_conv_input_grad
    monkeypatch.setattr(B, "np_conv_input_grad", lambda G, Wt: real(G, Wt[:, ::-1, ::-1, :]))
    code, rec = _run(kit, tmp_path, monkeypatch)
    assert code == B.EXIT_VOID and rec["void"]["step"] == "reference" and rec["outcome"] is None


def test_SIGTERM_DURING_THE_ANALYSIS_IS_A_VOID_and_the_traces_and_fixtures_survive(kit, tmp_path, monkeypatch):  # noqa
    def term(*a, **k):
        raise B.Terminated("SIGTERM")
    monkeypatch.setattr(B, "local_checks", term)
    code, rec = _run(kit, tmp_path, monkeypatch)
    ev = tmp_path / "ev"
    assert code == B.EXIT_VOID and rec["void"]["step"] == "terminated"
    assert (ev / "trace" / "cpu.json").exists() and (ev / "trace" / "gpu.json").exists()
    assert len(_lines(ev / "fixtures.jsonl")) == 2 and json.loads((ev / "record.json").read_text())["status"] == "VOID"


@pytest.mark.parametrize("case, step", [("checkpoint", "checkpoint"), ("pin", "evidence"), ("focus", "plan")])
def test_A_BROKEN_INPUT_IS_A_VOID_BEFORE_ANY_NETWORK_IS_LOADED(kit, tmp_path, monkeypatch, case, step):  # noqa
    loads, over, spec = [], {}, None
    if case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    elif case == "pin":
        monkeypatch.setitem(B.PINNED, f"{B.ENTRY33}/full.jsonl", "0" * 64)
    spec = _spec(kit)
    if case == "focus":
        spec = {**spec, "focus": [1, 2]}
    spec_draw = K.draw_batches(spec["training_records"], T.recipe_for(spec), 1)[0]
    files = _fake_evidence(spec_draw)
    if case == "pin":
        files.pop(f"{B.ENTRY33}/full.jsonl")
    read = _serve(monkeypatch, lambda rel: (B.ROOT / rel).read_bytes(), files)
    if case == "pin":
        monkeypatch.setitem(B.PINNED, f"{B.ENTRY33}/full.jsonl", "0" * 64)
    code, rec = B.conclude(spec, _seams(kit, read, lambda: loads.append(1) or _net(), **over), tmp_path / "ev",
                           tmp_path / "fx")
    assert code == B.EXIT_VOID and rec["void"]["step"] == step and loads == []


def test_THE_OUTCOME_ORDER_is_the_card_s():
    spec = {"targets": ["encoder.blocks.1.conv2.weight", "encoder.blocks.2.conv2.weight"]}
    ok = {"forward_failures": [], "backward_failures": []}
    flips = {"stem.relu": {"count": 3, "block": -1}, "blocks.0.relu_outer": {"count": 2, "block": 0},
             "blocks.2.relu_outer": {"count": 1, "block": 2}}
    yes, no = {"both_explained": True}, {"both_explained": False}
    assert B.outcome({"cpu": ok, "gpu": {**ok, "backward_failures": ["x"]}}, flips, yes, spec) == "LOCAL_DERIVATIVE_ERROR"
    assert B.outcome({"cpu": {**ok, "forward_failures": ["x"]}, "gpu": ok}, flips, yes, spec) == "LOCAL_FORWARD_ERROR"
    assert B.outcome({"cpu": ok, "gpu": ok}, flips, yes, spec) == "BRANCH_CHANGE"
    assert B.outcome({"cpu": ok, "gpu": ok}, flips, no, spec) == "NOT_EXPLAINED"
    below = {k: v for k, v in flips.items() if v["block"] < 1}           # flips only below the targets do not count
    assert B.outcome({"cpu": ok, "gpu": ok}, below, yes, spec) == "NOT_EXPLAINED"


# ─────────────────────────────── refusals ───────────────────────────────

def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED is False
    assert B.main(["run", "--checkpoint", str(_checkpoint())]) == B.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err and not (isolated / "evidence").exists()


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_DEVICES_EVIDENCE_OR_FIXTURES(isolated):
    assert K.DEVICES == {"cpu": mx.cpu, "gpu": mx.cpu}
    assert B.EVIDENCE_DIR.startswith(str(isolated)) and str(B.FIXTURE_PARENT).startswith(str(isolated))


def test_ANYTHING_BUT_run_WITH_A_CHECKPOINT_IS_USAGE():
    assert B.main([]) == B.main(["run"]) == B.main(["go", "--checkpoint", "x"]) == B.EXIT_USAGE


def test_THE_PINS_ARE_THE_FILES_ON_DISK_and_the_plan_is_frozen():
    assert M._sha((B.ROOT / B.CARD).read_bytes()) == B.CARD_SHA256
    for rel, sha in B.PINNED.items():
        assert M._sha((B.ROOT / rel).read_bytes()) == sha, rel
    assert set(B.CODE.values()) <= set(B.PINNED)
    f = B.FROZEN
    assert (f["cross_rel_tol"], f["cross_abs_tol"], f["adjoint_rel_tol"]) == (1e-3, 1e-6, 1e-9)
    assert f["focus"] == [1, 13, 4, 42] and f["targets"] == C.TARGETS and (f["state"], f["batch"]) == ("S0", 0)
    cap = json.loads((B.ROOT / B.ENTRY33 / "reference" / "encoder.blocks.2.conv2.weight.json").read_text())
    assert cap["native"]["G_gpu_vs_cpu"]["worst"]["index"] == f["focus"]          # entry 33's focus element


@pytest.mark.parametrize("pin", [B.CODE["convref"], f"{B.ENTRY33}/capture/gpu.json"])
def test_A_FILE_THAT_IS_NOT_THE_PINNED_BYTES_REFUSES(opened, monkeypatch, pin, capsys):
    monkeypatch.setitem(B.PINNED, pin, "0" * 64)
    assert B.main(["run", "--checkpoint", str(_checkpoint())]) == B.EXIT_REFUSED
    assert pin in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "1", "", " 0"])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gate_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert B.main(["run", "--checkpoint", str(_checkpoint())]) == B.EXIT_REFUSED
    assert "MLX_ENABLE_TF32" in capsys.readouterr().err


def test_MLX_LOADED_AN_UNPINNED_CARD_AN_EXISTING_EVIDENCE_DIR_OR_A_RELATIVE_CHECKPOINT_REFUSE(opened, monkeypatch,
                                                                                              capsys):
    monkeypatch.setattr(B, "_mlx_loaded", lambda: True)
    assert B.main(["run", "--checkpoint", str(_checkpoint())]) == B.EXIT_REFUSED
    assert "imported" in capsys.readouterr().err
    monkeypatch.setattr(B, "_mlx_loaded", lambda: False)
    assert B.main(["run", "--checkpoint", "checkpoints/x.safetensors"]) == B.EXIT_REFUSED
    assert "absolute" in capsys.readouterr().err
    monkeypatch.setattr(B, "CARD_SHA256", "0" * 64)
    assert B.main(["run", "--checkpoint", str(_checkpoint())]) == B.EXIT_REFUSED
    assert "pinned card" in capsys.readouterr().err
    monkeypatch.undo()
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED", True)
    monkeypatch.setattr(B, "EVIDENCE_DIR", str(opened / "evidence"))
    monkeypatch.setattr(B, "_mlx_loaded", lambda: False)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    (opened / "evidence").mkdir()
    assert B.main(["run", "--checkpoint", str(_checkpoint())]) == B.EXIT_REFUSED
    assert "one-shot" in capsys.readouterr().err


PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import search_distillation_d2_branchtrace as B
from scripts.GPU.alphazero import search_distillation_d2_branchtrace_authorization as GATE
print("MLX_AFTER_IMPORT", B._mlx_loaded(), flush=True)
_tmp = pathlib.Path(tempfile.mkdtemp())
B.EVIDENCE_DIR = str(_tmp / "evidence")
B.FIXTURE_PARENT = _tmp / "fixture-parent"
def stub(spec, seams, out, fixture_dir, head=None):     # never a real run, whatever a mutant does above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB", "head": head}
B.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(B.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    argv = repr(["run", "--checkpoint", str(_checkpoint())])
    return subprocess.run([sys.executable, "-c", PRELUDE + body + f"\nsys.exit(B.main({argv}))\n"], cwd=B.ROOT,
                          env=env, capture_output=True, text=True, timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_and_the_gate_open_every_real_check_passes():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


# ─────────────────────────────── the REAL state ───────────────────────────────

def test_THE_PUBLIC_ENTRY_IN_THE_REAL_STATE_STOPS_NOT_REPRODUCED_when_gpu_is_the_cpu(opened, capsys):
    """The gate open, the frozen spec, the pinned card, code and evidence, the real corpus and incumbent. "gpu" is the
    CPU here, so the teacher values come from the CPU and the gradients cannot repeat entry 33's: the run stops before
    any trace. The CPU's logits and values digests do repeat entry 33's."""
    assert B.main(["run", "--checkpoint", str(_checkpoint())]) == B.EXIT_COMPLETED
    rec = json.loads((opened / "evidence" / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["outcome"] == "NOT_REPRODUCED", rec.get("void")
    assert rec["spec"] == json.loads(json.dumps(B.FROZEN)) and not rec["teacher_values"]["equals_entry33"]
    cpu = rec["step1"]["cpu"]
    assert cpu["got"]["logits_digest"] == cpu["expected"]["logits_digest"]
    assert cpu["got"]["values_digest"] == cpu["expected"]["values_digest"]
    assert cpu["got"]["grad_digest"] != cpu["expected"]["grad_digest"]
    assert rec["held_out_records_read"] == 0 and not (opened / "fixture-parent").exists()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["outcome"] == "NOT_REPRODUCED"


def test_A_FULL_TRACE_IN_THE_REAL_STATE_is_faithful_and_every_local_check_passes(opened, monkeypatch):
    """The real sources and incumbent through `conclude`, the frozen spec; entry 33's digests replaced HERE ONLY by the
    ones this CPU-only process produces (a first pass), so the trace runs: bitwise faithful on both labels, the adjoint
    self-check and every local forward and backward check passing on the real operands, and nothing flips."""
    seams = C._real_seams(_checkpoint())
    _, first = B.conclude(B.FROZEN, seams, opened / "pass1", opened / "fx1")
    assert first["outcome"] == "NOT_REPRODUCED", first.get("void")
    files = {f"{B.ENTRY33}/record.json": json.dumps({"teacher_values": first["teacher_values"]}).encode(),
             f"{B.ENTRY33}/full.jsonl": "".join(json.dumps({"device": d, "repeat": 0, **{
                 k: first["step1"][d]["got"][k] for k in ("logits_digest", "values_digest")}}) + "\n"
                 for d in ("cpu", "gpu")).encode()}
    for d in ("cpu", "gpu"):
        files[f"{B.ENTRY33}/capture/{d}.json"] = json.dumps({"uninstrumented": {
            k: first["step1"][d]["got"][k] for k in ("grad_digest", "terms")}}).encode()
    seams = C.Seams(**{**seams.__dict__, "read_repo": _serve(monkeypatch, seams.read_repo, files)})
    code, rec = B.conclude(B.FROZEN, seams, opened / "ev", opened / "fx")
    assert rec["step1"]["ok"] and rec["step2"] == {"cpu": {"faithful": True}, "gpu": {"faithful": True}}, \
        rec.get("void")
    assert rec["self_check"]["adjoint_input_grad"]["ok"]
    assert rec["local_checks"] == {d: {"forward_failures": [], "backward_failures": []} for d in ("cpu", "gpu")}
    assert len(rec["flips"]) == 13 and all(v["count"] == 0 for v in rec["flips"].values())
    assert rec["outcome"] == "NOT_EXPLAINED"
    t = json.loads((opened / "ev" / "trace" / "gpu.json").read_text())
    assert t["fidelity"]["grads_compared"] == 60
    foc = json.loads((opened / "ev" / "focus.json").read_text())
    assert foc["element"] == [1, 13, 4, 42] and len(foc["cpu"]["blocks"]) == 6
    assert sorted(p.name for p in (opened / "fx").iterdir()) == ["cpu_record1.npz", "gpu_record1.npz"]
