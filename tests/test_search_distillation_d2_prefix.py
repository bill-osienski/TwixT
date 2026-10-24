"""The 21-update diagnostic prefix (`scripts/GPU/alphazero/search_distillation_d2_prefix.py`): the prefix loop equal to
gradcheck.reconstruct's and its per-update digests catching a changed path that the scalars miss; the float64 parameter
and policy-head references (each gating, L2 included); each criterion of card §6 produced by an injected fault; the
explanation rule; the stops; the refusals; and the public entry plus a full run in the REAL state (the real corpus
through the training-only reader, the real incumbent and entry 32's real evidence, read-only). Both device labels run
on the CPU here ("gpu" is remapped), so honest comparisons agree to the bit and every disagreement is an injected
fault on the networks built under the "gpu" label (the prefix's own loads are outside any label, so no fault touches
it). The rig tests use the earlier diagnostics' 1-block fixture network, with a 3-update prefix.

Reproduction is against entry 32's digests; on the rig (and for the full real-state run, where "gpu" is the CPU) a
first pass records the digests this rig produces and a second pass is served entry-32 files holding them, pinned by
their own hashes. That substitution is in these tests only.

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture remaps the devices and
redirects the evidence for every in-process test; the fresh-process checks redirect it and replace `conclude` with a
stub."""
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
from scripts.GPU.alphazero import search_distillation_d2_gradcheck as K  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_gradtrace as R  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_prefix as P  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_prefix_authorization as GATE  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_train as T  # noqa: E402
from tests.test_search_distillation_d2_gradcheck import kit  # noqa: E402,F401
from tests.test_search_distillation_d2_gradrepeat import done, rig  # noqa: E402,F401
from tests.test_search_distillation_d2_gradtrace import ref, reproduced  # noqa: E402,F401
from tests.test_search_distillation_d2_measure import _net  # noqa: E402
from tests.test_search_distillation_d2_train import corpus  # noqa: E402,F401

RIG_UPDATES = 3


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(K, "DEVICES", {"cpu": mx.cpu, "gpu": mx.cpu})
    monkeypatch.setattr(P, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED", True)
    return isolated


def _checkpoint() -> pathlib.Path:
    common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=P.ROOT,
                                     text=True).strip()
    return pathlib.Path(common).parent / M.CHECKPOINT


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


# ─────────────────────────────── the rig ───────────────────────────────

def _spec(kit, **over):                                                 # noqa: F811
    shared = ("corpus_sha256", "teacher_digest", "reference_sha256", "reference_student_digest",
              "training_ids_sha256", "training_games", "held_out_games", "training_records", "passes", "steps",
              "batch_size", "learning_rate")
    return {**P.FROZEN, **{k: kit["spec"][k] for k in shared}, "batches": [0, RIG_UPDATES],
            "prefix_updates": RIG_UPDATES, "focus": [0, 0, 0, 0], **over}


def _fake32(spec, got=None, teacher="x"):
    """Entry 32's files as a rig run sees them: the rig's draws, and (second pass) the digests it produced."""
    draws = K.draw_batches(spec["training_records"], T.recipe_for(spec), max(spec["batches"]) + 1)
    rec = {"batches": {str(b): draws[b] for b in spec["batches"]}, "teacher_values": {"digest": teacher}}
    lines = []
    for b in spec["batches"]:
        for dev in ("cpu", "gpu"):
            g = (got or {}).get(f"S0_b{b}", {}).get(dev, {}).get("got") or {
                "terms": [0.0, 0.0, 0.0], "grad_digest": "x", "logits_digest": "x", "values_digest": "x"}
            lines.append({"block": f"S0_b{b}", "device": dev, "repeat": 0, "loss": g["terms"][0],
                          "policy": g["terms"][1], "anchor": g["terms"][2],
                          **{k: g[k] for k in ("grad_digest", "logits_digest", "values_digest")}})
    return {f"{P.ENTRY32}/record.json": json.dumps(rec).encode(),
            f"{P.ENTRY32}/full.jsonl": "".join(json.dumps(x) + "\n" for x in lines).encode()}


def _serve(monkeypatch, base_read, files):
    for rel, raw in files.items():
        if rel in P.PINNED:
            monkeypatch.setitem(P.PINNED, rel, M._sha(raw))
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
    base_read = lambda rel: (P.ROOT / rel).read_bytes()                  # noqa: E731
    files = {R.REFERENCE: kit["rig"]["ref"]["raw"], **_fake32(spec)}
    if reproducing:
        _, first = P.conclude(spec, _seams(kit, _serve(monkeypatch, base_read, files), make, **over),
                              tmp_path / "pass1")
        assert first["outcome"] == "NOT_REPRODUCED", first.get("void")
        got = json.loads((tmp_path / "pass1" / "reproduce.json").read_text())["blocks"]
        files.update(_fake32(spec, got, first["teacher_values"]["digest"]))
    return P.conclude(spec, _seams(kit, _serve(monkeypatch, base_read, files), make, **over), tmp_path / "ev")


def _on_gpu_label(make, apply):
    def load():
        net = make()
        if K.CURRENT["device"] == "gpu":
            apply(net)
        return net
    return load


def _swap(path, cls):
    def apply(net):
        m = net
        for p in path.split("."):
            m = m[int(p)] if p.isdigit() else getattr(m, p)
        m.__class__ = cls
    return apply


class _ScaledConvWeightGrad(nn.Conv2d):
    """A PARAMETER-GRADIENT fault: the same forward and input gradient; the weight's gradient scaled by 1.01."""
    def __call__(self, x):
        w = self.weight + 0.01 * (self.weight - mx.stop_gradient(self.weight))
        y = mx.conv2d(x, w, self.stride, self.padding, self.dilation, self.groups)
        return y + self.bias if "bias" in self else y


class _ScaledLinearWeightGrad(nn.Linear):
    """The same, for the policy head's fc."""
    def __call__(self, x):
        w = self.weight + 0.01 * (self.weight - mx.stop_gradient(self.weight))
        return x @ w.T + self.bias if "bias" in self else x @ w.T


class _ScaledInputGrad(nn.Conv2d):
    """An ENCODER-OUTPUT-GRADIENT fault (the policy head's conv): the same forward and parameter gradients; the
    gradient it passes back to the encoder scaled by 1.01."""
    def __call__(self, x):
        return super().__call__(x + 0.01 * (x - mx.stop_gradient(x)))


class _NoisyConv2(nn.Conv2d):
    """A perturbation of the WHOLE forward (2e-4 of scale on every element of conv2's output, gradient unchanged): its
    ReLU flips come with a changed encoder output and heads, which no internal flip can excuse."""
    def __call__(self, x):
        y = super().__call__(x)
        noise = mx.array(np.random.default_rng(7).standard_normal(y.shape).astype(np.float32))
        return y + 2e-4 * mx.stop_gradient(mx.abs(y).max()) * noise


def _flip_near_zero(frac):
    """Different OPERANDS only where BN1's output (the inner ReLU's input) is within `frac` of its scale of zero:
    there its sign is reversed (gradient unchanged), as two backends' rounding would flip a branch."""
    class _FlipNearZero(nn.BatchNorm):
        def __call__(self, x):
            y = super().__call__(x)
            eps = frac * mx.stop_gradient(mx.abs(y).max())
            return y + mx.stop_gradient(mx.where(mx.abs(y) < eps, -2 * y, 0))
    return _FlipNearZero


class _ShiftedBN2Forward(nn.BatchNorm):
    """A FORWARD fault: BN2's output shifted by a constant."""
    def __call__(self, x):
        return super().__call__(x) + 0.05


class _NaNConv2(nn.Conv2d):
    def __call__(self, x):
        return super().__call__(x) * mx.array(float("nan"))


# ─────────────────────────────── the prefix ───────────────────────────────

def _training(kit):                                                     # noqa: F811
    spec = _spec(kit)
    training, _, _ = R.read_training(spec, kit["rig"]["corpus"]["raw"])
    return spec, T.recipe_for(spec), training


class _Lines:
    def __init__(self):
        self.lines = []

    def line(self, rel, obj):
        self.lines.append(obj)


def test_THE_PREFIX_LOOP_IS_RECONSTRUCT_S_bitwise_and_records_both_digests_every_update(kit):  # noqa: F811
    spec, recipe, training = _training(kit)
    tv = np.linspace(-0.5, 0.5, len(training)).astype(np.float32)
    reference = [{f: 0.0 for f in R.TRACE_FIELDS}] * RIG_UPDATES
    with K.on("cpu"):
        a = _net()
        a.eval()
        base = K.reconstruct(a, training, recipe, tv, RIG_UPDATES, spec, _Lines(), reference)
        b = _net()
        b.eval()
        sink, n = _Lines(), {"updates": 0}
        state, rows = P.prefix(b, training, recipe, tv, RIG_UPDATES, spec, sink, reference, "P1", n)
    assert K._digest(state) == K._digest(base) and n["updates"] == RIG_UPDATES
    assert [r["update"] for r in rows] == list(range(RIG_UPDATES)) and sink.lines == rows
    assert len({r["model_digest"] for r in rows}) == RIG_UPDATES and len({r["adam_digest"] for r in rows}) == RIG_UPDATES
    assert rows[-1]["model_digest"] == K._digest(state)


def test_A_CHANGED_ADAM_IS_CAUGHT_BY_THE_DIGESTS_where_update_0_s_scalars_agree(kit, monkeypatch):  # noqa: F811
    spec, recipe, training = _training(kit)
    tv = np.linspace(-0.5, 0.5, len(training)).astype(np.float32)
    reference = [{f: 0.0 for f in R.TRACE_FIELDS}] * RIG_UPDATES
    runs = []
    real = P._adam
    for eps in (None, 1e-4):
        monkeypatch.setattr(P, "_adam", real if eps is None else (lambda r: real(r).__class__(
            learning_rate=r.learning_rate, eps=eps)))
        with K.on("cpu"):
            net = _net()
            net.eval()
            runs.append(P.prefix(net, training, recipe, tv, RIG_UPDATES, spec, _Lines(), reference, "P",
                                 {"updates": 0})[1])
    cmp = P.prefixes_equal(*runs)
    assert not cmp["equal"] and cmp["first_difference"]["update"] == 0
    assert "model_digest" in cmp["first_difference"]["fields"]
    assert all(runs[0][0][f] == runs[1][0][f] for f in R.TRACE_FIELDS)   # update 0's scalars alone would pass


# ─────────────────────────────── the explanation rule ───────────────────────────────

def test_THE_EXPLANATION_RULE_needs_every_condition():
    assert P.explained(True, True, True, True, False)
    for i in range(5):
        args = [True, True, True, True, False]
        args[i] = not args[i]
        assert not P.explained(*args), i                # the encoder-output gradient (index 1) included


def test_THE_PARAMETER_TABLE_COVERS_EVERY_ENCODER_TENSOR_and_its_path_starts_after_it():
    with K.on("cpu"):
        net = _net()
        net.eval()
        from mlx.utils import tree_flatten
        from scripts.GPU.alphazero.trainer import MainModule
        names = sorted(k for k, _ in tree_flatten(MainModule(net.encoder, net.policy_head).trainable_parameters())
                       if k.startswith("encoder."))
        table = P.param_table(P.B._constants(net))
    assert sorted(table) == names
    assert table["encoder.blocks.0.conv2.weight"]["path"] == ["blocks.0.relu_outer"]
    assert table["encoder.blocks.0.bn1.bias"]["path"] == ["blocks.0.relu_inner", "blocks.0.relu_outer"]
    assert table["encoder.conv1.weight"]["path"][0] == "stem.relu"


# ─────────────────────────────── each outcome on the rig ───────────────────────────────

def test_A_CLEAN_RIG_RUN_IS_QUALIFIED_with_every_check_passing_and_nothing_flipping(kit, tmp_path, monkeypatch):  # noqa
    code, rec = _run(kit, tmp_path, monkeypatch)
    ev = tmp_path / "ev"
    assert code == P.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert rec["outcome"] == "QUALIFIED_AT_U21" and rec["reason"] is None, rec.get("failures")
    assert rec["reproduce"]["ok"] and rec["optimizer_updates"] == 2 * RIG_UPDATES
    assert rec["prefix"]["comparison"]["equal"] and rec["prefix"]["saved"] is False
    assert len(_lines(ev / "prefix.jsonl")) == 2 * RIG_UPDATES
    for b in ("0", str(RIG_UPDATES)):
        r = rec["batches"][b]
        assert r["fidelity"] == {d: {"trace": True, "pretanh_capture": True} for d in ("cpu", "gpu")}
        for d in ("cpu", "gpu"):
            assert r[d]["parameters_checked"] == 12 and not any(
                r[d][k] for k in ("backward_failures", "forward_failures", "parameter_failures",
                                  "policy_head_failures"))
        assert r["common_input_failures"] == [] and all(v == 0 for v in r["flips"].values())
        assert r["native_failing"] == [] and r["encoder_output_gradient"]["bitwise"]
        chk = json.loads((ev / "checks" / f"b{b}_gpu.json").read_text())
        assert len(chk["policy_head_native"]["tensors"]) == 8 and chk["fidelity"]["pretanh_capture"]["ok"]
    assert sorted(p.name for p in (ev / "full").iterdir()) == ["U21_b0.json", f"U21_b{RIG_UPDATES}.json"]
    assert not list((ev / "accounting").iterdir())
    assert rec["training_qualified"] is False and rec["candidate_checkpoint"] is None and not rec["model_state_saved"]


def test_A_PARAMETER_GRADIENT_FAULT_IS_A_LOCAL_DERIVATIVE_ERROR_caught_only_by_the_parameter_reference(
        kit, tmp_path, monkeypatch):  # noqa: F811
    make = _on_gpu_label(_net, _swap("encoder.blocks.0.conv2", _ScaledConvWeightGrad))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert rec["outcome"] == "NOT_QUALIFIED" and rec["reason"] == "local_derivative_error", rec.get("void")
    g = rec["batches"]["0"]["gpu"]
    assert g["parameter_failures"] == ["encoder.blocks.0.conv2.weight"] and g["backward_failures"] == []
    assert rec["batches"]["0"]["cpu"]["parameter_failures"] == []


def test_A_POLICY_HEAD_GRADIENT_FAULT_IS_CAUGHT_BY_THE_NATIVE_CHECK_not_the_common_input_one(
        kit, tmp_path, monkeypatch):  # noqa: F811
    make = _on_gpu_label(_net, _swap("policy_head.fc", _ScaledLinearWeightGrad))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert rec["reason"] == "local_derivative_error", rec.get("void")
    r = rec["batches"]["0"]
    assert r["gpu"]["policy_head_failures"] == ["policy_head.fc.weight"] and r["cpu"]["policy_head_failures"] == []
    assert r["common_input_failures"] == []             # its MLX side rebuilds the head from clean parameters


def test_A_GRADIENT_ENTERING_FROM_THE_ENCODER_OUTPUT_IS_UNEXPLAINED(kit, tmp_path, monkeypatch):  # noqa: F811
    make = _on_gpu_label(_net, _swap("policy_head.conv", _ScaledInputGrad))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert rec["reason"] == "unexplained_gradient", (rec.get("void"), rec.get("failures"))
    r = rec["batches"]["0"]
    assert not r["encoder_output_gradient"]["within_tol"] and r["unexplained"]
    acc = json.loads((tmp_path / "ev" / "accounting" / "b0.json").read_text())
    assert all(not v["explained"] for v in acc["tensors"].values())


def test_NEAR_ZERO_BRANCH_FLIPS_ARE_EXPLAINED_BY_THE_ACCOUNTING_and_qualify(kit, tmp_path, monkeypatch):  # noqa
    make = _on_gpu_label(_net, _swap("encoder.blocks.0.bn1", _flip_near_zero(1e-4)))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert rec["outcome"] == "QUALIFIED_AT_U21", (rec.get("void"), rec.get("reason"), rec.get("failures"))
    r = rec["batches"]["0"]
    assert r["flips"]["blocks.0.relu_inner"] > 0 and r["encoder_output_gradient"]["within_tol"]
    assert "encoder.blocks.0.conv1.weight" in r["native_failing"] and r["explained"] == r["native_failing"]
    acc = json.loads((tmp_path / "ev" / "accounting" / "b0.json").read_text())["tensors"]
    w = acc["encoder.blocks.0.conv1.weight"]
    assert w["with_flips"]["within_tol"] and not w["without_flips"]["within_tol"]
    assert w["flips_on_path"] == {"blocks.0.relu_inner": r["flips"]["blocks.0.relu_inner"]}


def test_FLIPS_BIG_ENOUGH_TO_SHOW_IN_THE_FORWARD_ARE_A_LOCAL_FORWARD_ERROR(kit, tmp_path, monkeypatch):  # noqa
    make = _on_gpu_label(_net, _swap("encoder.blocks.0.bn1", _flip_near_zero(1e-3)))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert rec["reason"] == "local_forward_error" and rec["failures"] == ["U21_b0 gpu: blocks.0.a1"], rec.get("void")


def test_A_WHOLE_FORWARD_PERTURBATION_IS_UNEXPLAINED_despite_its_flips(kit, tmp_path, monkeypatch):  # noqa: F811
    make = _on_gpu_label(_net, _swap("encoder.blocks.0.conv2", _NoisyConv2))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert rec["reason"] == "unexplained_gradient", (rec.get("void"), rec.get("failures"))
    r = rec["batches"][str(RIG_UPDATES)]
    assert r["flips"]["blocks.0.relu_outer"] > 0 and not r["encoder_output_gradient"]["within_tol"]
    assert any(k.startswith("policy_head.") for k in r["unexplained"])
    assert "encoder.blocks.0.conv2.weight" in r["unexplained"]           # flips on its path, but the output moved


def test_A_FORWARD_FAULT_STOPS_FORWARD_DISAGREEMENT_before_any_trace(kit, tmp_path, monkeypatch):  # noqa: F811
    make = _on_gpu_label(_net, _swap("encoder.blocks.0.bn2", _ShiftedBN2Forward))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert rec["reason"] == "forward_disagreement", rec.get("void")
    assert "batches" not in rec and not list((tmp_path / "ev" / "checks").iterdir())


def test_A_NON_FINITE_RESULT_IS_NOT_QUALIFIED_not_a_void(kit, tmp_path, monkeypatch):  # noqa: F811
    after, real = {"prefix": False}, P.prefix

    def prefix(*a, **k):
        out = real(*a, **k)
        after["prefix"] = True
        return out
    monkeypatch.setattr(P, "prefix", prefix)
    nan = _swap("encoder.blocks.0.conv2", _NaNConv2)
    make = _on_gpu_label(_net, lambda net: nan(net) if after["prefix"] else None)     # at U21 only
    code, rec = _run(kit, tmp_path, monkeypatch, make=make)
    assert code == P.EXIT_COMPLETED and rec["outcome"] == "NOT_QUALIFIED" and rec["reason"] == "non_finite", \
        rec.get("void")


def test_A_NON_FINITE_RESULT_AT_THE_INCUMBENT_IS_NOT_REPRODUCED(kit, tmp_path, monkeypatch):  # noqa: F811
    make = _on_gpu_label(_net, _swap("encoder.blocks.0.conv2", _NaNConv2))
    code, rec = _run(kit, tmp_path, monkeypatch, make=make, reproducing=False)
    assert rec["outcome"] == "NOT_REPRODUCED" and rec["optimizer_updates"] == 0, rec.get("void")
    rep = json.loads((tmp_path / "ev" / "reproduce.json").read_text())["blocks"]["S0_b0"]["gpu"]
    assert "non_finite" in rep["got"] and not rep["ok"]


def test_A_DIFFERENT_SECOND_PREFIX_STOPS_NOT_REPEATABLE_before_any_evaluation(kit, tmp_path, monkeypatch):  # noqa
    real, calls = P._adam, []

    def adam(r):
        calls.append(1)
        return real(r) if len(calls) != 4 else real(r).__class__(learning_rate=r.learning_rate, eps=1e-4)
    monkeypatch.setattr(P, "_adam", adam)   # call 1: pass 1's settings check; then 2: settings, 3: P1, 4: P2
    code, rec = _run(kit, tmp_path, monkeypatch)
    assert rec["reason"] == "not_repeatable", rec.get("void")
    assert rec["prefix"]["comparison"]["first_difference"]["update"] == 0
    assert "native" not in rec and rec["optimizer_updates"] == 2 * RIG_UPDATES


def test_A_PERTURBING_TRACE_STOPS_TRACE_NOT_FAITHFUL(kit, tmp_path, monkeypatch):  # noqa: F811
    real = P.B._traced_block_call

    def shifted(self, x):
        return real(self, x) + 1e-3
    monkeypatch.setattr(P.B, "_TRACED", {})
    monkeypatch.setattr(P.B, "_traced_block_call", shifted)
    code, rec = _run(kit, tmp_path, monkeypatch)
    assert rec["reason"] == "trace_not_faithful", rec.get("void")
    assert rec["batches"]["0"]["fidelity"]["cpu"] == {"trace": False, "pretanh_capture": True}


def test_WITHOUT_THE_DIGESTS_THE_RUN_STOPS_NOT_REPRODUCED_before_any_update(kit, tmp_path, monkeypatch):  # noqa
    code, rec = _run(kit, tmp_path, monkeypatch, reproducing=False)
    assert code == P.EXIT_COMPLETED and rec["outcome"] == "NOT_REPRODUCED" and rec["optimizer_updates"] == 0
    assert not (tmp_path / "ev" / "prefix.jsonl").exists() and (tmp_path / "ev" / "reproduce.json").exists()


def test_SIGTERM_DURING_THE_PREFIX_IS_A_VOID_and_the_lines_survive(kit, tmp_path, monkeypatch):  # noqa: F811
    def term(*a, **k):
        raise P.Terminated("SIGTERM")
    monkeypatch.setattr(P, "prefixes_equal", term)
    code, rec = _run(kit, tmp_path, monkeypatch)
    ev = tmp_path / "ev"
    assert code == P.EXIT_VOID and rec["void"]["step"] == "terminated" and rec["outcome"] is None
    assert len(_lines(ev / "prefix.jsonl")) == 2 * RIG_UPDATES and rec["optimizer_updates"] == 2 * RIG_UPDATES
    assert json.loads((ev / "record.json").read_text())["status"] == "VOID"


@pytest.mark.parametrize("case, step", [("checkpoint", "checkpoint"), ("pin", "evidence"), ("plan", "plan")])
def test_A_BROKEN_INPUT_IS_A_VOID_BEFORE_ANY_NETWORK_IS_LOADED(kit, tmp_path, monkeypatch, case, step):  # noqa
    loads, over = [], {}
    spec = _spec(kit)
    if case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    if case == "plan":
        spec = {**spec, "prefixes": 1}
    files = {R.REFERENCE: kit["rig"]["ref"]["raw"], **_fake32(_spec(kit))}
    read = _serve(monkeypatch, lambda rel: (P.ROOT / rel).read_bytes(), files)
    if case == "pin":
        monkeypatch.setitem(P.PINNED, f"{P.ENTRY32}/full.jsonl", "0" * 64)
    code, rec = P.conclude(spec, _seams(kit, read, lambda: loads.append(1) or _net(), **over), tmp_path / "ev")
    assert code == P.EXIT_VOID and rec["void"]["step"] == step and loads == []


# ─────────────────────────────── refusals ───────────────────────────────

def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED is False
    assert P.main(["run", "--checkpoint", str(_checkpoint())]) == P.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err and not (isolated / "evidence").exists()


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_DEVICES_OR_EVIDENCE(isolated):
    assert K.DEVICES == {"cpu": mx.cpu, "gpu": mx.cpu} and P.EVIDENCE_DIR.startswith(str(isolated))


def test_ANYTHING_BUT_run_WITH_A_CHECKPOINT_IS_USAGE():
    assert P.main([]) == P.main(["run"]) == P.main(["go", "--checkpoint", "x"]) == P.EXIT_USAGE


def test_THE_PINS_ARE_THE_FILES_ON_DISK_and_the_plan_is_frozen():
    assert M._sha((P.ROOT / P.CARD).read_bytes()) == P.CARD_SHA256
    for rel, sha in P.PINNED.items():
        assert M._sha((P.ROOT / rel).read_bytes()) == sha, rel
    assert set(P.CODE.values()) <= set(P.PINNED)
    f = P.FROZEN
    assert (f["cross_rel_tol"], f["cross_abs_tol"], f["adjoint_rel_tol"]) == (1e-3, 1e-6, 1e-9)
    assert (f["batches"], f["prefix_updates"], f["prefixes"], f["repeats"]) == ([0, 21], 21, 2, 3)
    assert (f["l2_weight"], f["clip_norm"], f["batch_size"], f["learning_rate"]) == (1e-4, 1.0, 64, 1e-4)


@pytest.mark.parametrize("pin", ["scripts/GPU/alphazero/search_distillation_d2_branchtrace.py",
                                 "docs/superpowers/evidence/search-distillation-d2-tf32path/full.jsonl"])
def test_A_FILE_THAT_IS_NOT_THE_PINNED_BYTES_REFUSES(opened, monkeypatch, pin, capsys):
    monkeypatch.setitem(P.PINNED, pin, "0" * 64)
    assert P.main(["run", "--checkpoint", str(_checkpoint())]) == P.EXIT_REFUSED
    assert pin in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "1", "", " 0"])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gate_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert P.main(["run", "--checkpoint", str(_checkpoint())]) == P.EXIT_REFUSED
    assert "MLX_ENABLE_TF32" in capsys.readouterr().err


def test_MLX_LOADED_AN_UNPINNED_CARD_AN_EXISTING_EVIDENCE_DIR_OR_A_RELATIVE_CHECKPOINT_REFUSE(opened, monkeypatch,
                                                                                              capsys):
    monkeypatch.setattr(P, "_mlx_loaded", lambda: True)
    assert P.main(["run", "--checkpoint", str(_checkpoint())]) == P.EXIT_REFUSED
    assert "imported" in capsys.readouterr().err
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    assert P.main(["run", "--checkpoint", "checkpoints/x.safetensors"]) == P.EXIT_REFUSED
    assert "absolute" in capsys.readouterr().err
    monkeypatch.setattr(P, "CARD_SHA256", "0" * 64)
    assert P.main(["run", "--checkpoint", str(_checkpoint())]) == P.EXIT_REFUSED
    assert "pinned card" in capsys.readouterr().err
    monkeypatch.undo()
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED", True)
    monkeypatch.setattr(P, "EVIDENCE_DIR", str(opened / "evidence"))
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    (opened / "evidence").mkdir()
    assert P.main(["run", "--checkpoint", str(_checkpoint())]) == P.EXIT_REFUSED
    assert "one-shot" in capsys.readouterr().err


PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import search_distillation_d2_prefix as P
from scripts.GPU.alphazero import search_distillation_d2_prefix_authorization as GATE
print("MLX_AFTER_IMPORT", P._mlx_loaded(), flush=True)
_tmp = pathlib.Path(tempfile.mkdtemp())
P.EVIDENCE_DIR = str(_tmp / "evidence")
def stub(spec, seams, out, head=None):     # never a real run, whatever a mutant does above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB", "head": head}
P.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(P.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    argv = repr(["run", "--checkpoint", str(_checkpoint())])
    return subprocess.run([sys.executable, "-c", PRELUDE + body + f"\nsys.exit(P.main({argv}))\n"], cwd=P.ROOT,
                          env=env, capture_output=True, text=True, timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_and_the_gate_open_every_real_check_passes():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


# ─────────────────────────────── the REAL state ───────────────────────────────

def test_THE_PUBLIC_ENTRY_IN_THE_REAL_STATE_STOPS_NOT_REPRODUCED_when_gpu_is_the_cpu(opened, capsys):
    """The gate open, the frozen spec, the pinned card, code and evidence, the real corpus and incumbent. "gpu" is the
    CPU here, so the teacher values come from the CPU and cannot repeat entry 32's: the run stops before any update.
    The CPU's logits and values digests do repeat entry 32's on both batches."""
    assert P.main(["run", "--checkpoint", str(_checkpoint())]) == P.EXIT_COMPLETED
    ev = opened / "evidence"
    rec = json.loads((ev / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["outcome"] == "NOT_REPRODUCED", rec.get("void")
    assert rec["spec"] == json.loads(json.dumps(P.FROZEN)) and not rec["teacher_values"]["equals_entry32"]
    rep = json.loads((ev / "reproduce.json").read_text())["blocks"]
    for b in ("S0_b0", "S0_b21"):
        cpu = rep[b]["cpu"]
        assert cpu["got"]["logits_digest"] == cpu["expected"]["logits_digest"], b
        assert cpu["got"]["values_digest"] == cpu["expected"]["values_digest"], b
    assert rec["optimizer_updates"] == 0 and rec["held_out_records_read"] == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["outcome"] == "NOT_REPRODUCED"


def test_A_FULL_RUN_IN_THE_REAL_STATE_is_qualified_when_both_labels_are_the_cpu(opened, monkeypatch):
    """The real sources and incumbent through `conclude`, the frozen spec; entry 32's digests replaced HERE ONLY by the
    ones this CPU-only process produces (a first pass), so the prefix runs: two 21-update prefixes bitwise equal at
    every update, and at U21 every local, parameter and policy-head reference passing on the real operands, with
    nothing flipping. It proves the references hold on the real model; it says nothing about the GPU."""
    seams = C._real_seams(_checkpoint())
    _, first = P.conclude(P.FROZEN, seams, opened / "pass1")
    assert first["outcome"] == "NOT_REPRODUCED", first.get("void")
    got = json.loads((opened / "pass1" / "reproduce.json").read_text())["blocks"]
    files = _fake32(P.FROZEN, got, first["teacher_values"]["digest"])
    rec32 = json.loads(seams.read_repo(f"{P.ENTRY32}/record.json"))
    assert json.loads(files[f"{P.ENTRY32}/record.json"])["batches"] == {k: rec32["batches"][k] for k in ("0", "21")}
    seams = C.Seams(**{**seams.__dict__, "read_repo": _serve(monkeypatch, seams.read_repo, files)})
    code, rec = P.conclude(P.FROZEN, seams, opened / "ev")
    assert rec["outcome"] == "QUALIFIED_AT_U21", (rec.get("void"), rec.get("reason"), rec.get("failures"))
    assert rec["optimizer_updates"] == 42 and rec["prefix"]["comparison"]["updates_compared"] == 21
    for b in ("0", "21"):
        r = rec["batches"][b]
        assert all(r[d]["parameters_checked"] == 52 for d in ("cpu", "gpu"))
        assert all(v == 0 for v in r["flips"].values()) and len(r["flips"]) == 13
    lines = _lines(opened / "ev" / "prefix.jsonl")
    assert [x["update"] for x in lines] == list(range(21)) * 2 and all("stage3" in x for x in lines)
