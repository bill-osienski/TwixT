"""Search distillation D2 linear-layer reproducer (search_distillation_d2_linrepro.py): operands from fixtures (hash
checked), the NumPy expectation, the three implementations, the gross/tolerance/repeatability judgement split by
legal and padded rows, the gating of the targeted cases, retention and the VOIDs, through the TEST seams. Both device
labels run on the CPU here ("gpu" is remapped), so every disagreement is an injected fault. The fixtures are small
synthetic ones written in the gradient check's own format -- not the real fixtures. The public entry is checked only
with its gate closed and with an unpinned card."""
import hashlib
import io
import json
import os
import signal

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2_gradcheck as K  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_linrepro as L  # noqa: E402

ROWS = [4, 8, 12, 16, 20]


@pytest.fixture(autouse=True)
def _both_labels_on_the_cpu(monkeypatch):
    monkeypatch.setattr(L, "DEVICES", {"cpu": mx.cpu, "gpu": mx.cpu})


def _npz_bytes(arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


@pytest.fixture(scope="module")
def fx():
    """Synthetic fixtures for S0 and S21: a reference_real_<S> (8-channel head, 4 records, 6 move slots with padding)
    and a stage_<S>_out (its input, cotangent zero on padded slots, weight and bias)."""
    rng = np.random.default_rng(5)
    files = {}
    for st in ("S0", "S21"):
        _, ph, F, rows, cols, _m, _t, _w = K.synthetic_cases(int(rng.integers(100)))[1]
        mask = np.ones((4, 6), np.float32)
        mask[1, 4:] = 0
        mask[3, 2:] = 0
        t = rng.uniform(0.1, 1, (4, 6)).astype(np.float32) * mask
        t /= t.sum(axis=1, keepdims=True)
        p = K.policy_head_params(ph)
        files[f"reference_real_{st}"] = _npz_bytes({"F": F, "rows": rows, "cols": cols, "mask": mask, "target": t,
                                                   "w": np.ones(4, np.float32),
                                                   **{f"param.{k}": v for k, v in p.items()}})
        ct = (rng.normal(0, 0.01, (4, 6)) * mask).astype(np.float32)
        files[f"stage_{st}_out"] = _npz_bytes({"input": np.maximum(rng.normal(0, 1, (24, 64)), 0).astype(np.float32),
                                              "cotangent": ct, "mask": mask,
                                              "param.weight": rng.normal(0, .2, (1, 64)).astype(np.float32),
                                              "param.bias": np.array([0.01], np.float32)})
    index = "".join(json.dumps({"name": n, "sha256": hashlib.sha256(b).hexdigest()}) + "\n" for n, b in files.items())
    spec = {**L.FROZEN, "fixture_index_sha256": hashlib.sha256(index.encode()).hexdigest(), "row_counts": ROWS}
    return {"files": files, "index": index.encode(), "spec": spec}


def _seams(fx, **over):
    base = dict(read_fixture_index=lambda: fx["index"], read_fixture=lambda name: fx["files"][name])
    base.update(over)
    return L.Seams(**base)


def _go(fx, tmp_path, spec=None, **over):
    return L.conclude(spec or fx["spec"], _seams(fx, **over), tmp_path / "ev", tmp_path / "saved")


def _cases(tmp_path):
    return {c["id"]: c for c in map(json.loads, (tmp_path / "ev" / "cases.jsonl").read_text().splitlines())}


def test_A_CLEAN_RUN_reproduces_nothing_and_runs_no_targeted_case(fx, tmp_path):
    code, rec = _go(fx, tmp_path)
    assert code == L.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert json.loads((tmp_path / "ev" / "record.json").read_text()) == json.loads(json.dumps(rec))
    assert rec["isolation_reproduces"] is False
    assert rec["original_reproduces"] == {"fc_S0_original": False, "fc_S21_original": False,
                                          "out_S0_original": False, "out_S21_original": False}
    cases = _cases(tmp_path)
    assert list(cases) == rec["case_list"] and len(cases) == 4 + 4 * 7
    assert all(c.get("skipped") for c in cases.values() if c["targeted"])
    for cid in rec["original_reproduces"]:
        c = cases[cid]
        for impl in L.FROZEN["implementations"]:
            gpu = c["implementations"][impl]["gpu"]
            assert gpu["failing_outputs"] == [] and gpu["gross"] is False
            assert all(all(v) for v in gpu["repeats_bitwise"].values()) and len(gpu["outputs"]["Y"]) == 3
            assert len(c["implementations"][impl]["cpu"]["outputs"]["Y"]) == 1
            y = gpu["outputs"]["Y"][0]
            assert y["legal"]["rows"] == 18 and y["padded"]["rows"] == 6         # 24 rows, 6 padded
    assert rec["network_loaded"] is False and rec["causal_claims"] is None
    assert sorted(s["name"] for s in rec["saved"]) == ["operands_fc_S0", "operands_fc_S21"]   # derived operands only
    assert rec["operands"]["out_S0"]["source"] == "stage_S0_out (as saved)"


def test_THE_EXPECTATION_IS_THE_FLOAT64_FORMULA_and_every_implementation_meets_it():
    rng = np.random.default_rng(0)
    o = {"X": rng.normal(0, 1, (300, 2)).astype(np.float32), "W": rng.normal(0, 1, (64, 2)).astype(np.float32),
         "b": rng.normal(0, 1, 64).astype(np.float32), "C": rng.normal(0, 1, (300, 64)).astype(np.float32),
         "legal": np.arange(300) < 250}
    e = L.expected(o)
    X, W, C = (o[k].astype(np.float64) for k in ("X", "W", "C"))
    assert np.array_equal(e["Y"], X @ W.T + o["b"].astype(np.float64)) and np.array_equal(e["dW"], C.T @ X)
    for impl in L.FROZEN["implementations"]:
        for layout in ("contiguous", "x_transposed_view"):
            for context in ("joint_eval", "separate_eval"):
                got = L.compute(impl, o, layout, context)
                for k in ("Y", "dX", "dW", "db"):
                    j = L.judge(got[k], e[k], o["legal"], L.FROZEN)
                    assert j["outside_tol"] == 0 and j["gross"] == 0, (impl, layout, context, k, j["max_abs"])


def _fault(monkeypatch, fn):
    real = L.compute

    def compute(impl, o, layout="contiguous", context="joint_eval"):
        got = real(impl, o, layout, context)
        return fn(impl, o, got) if L.CURRENT["device"] == "gpu" else got

    monkeypatch.setattr(L, "compute", compute)


def test_A_GROSS_FAULT_ON_PADDED_ROWS_reproduces_runs_the_targeted_cases_and_saves_the_elements(fx, tmp_path,
                                                                                                 monkeypatch):
    def fault(impl, o, got):
        if impl == "explicit_products" and o["X"].shape[1] == 64:            # out layer only
            dX = got["dX"].copy()
            dX[~o["legal"]] += 1.0
            return {**got, "dX": dX}
        return got

    _fault(monkeypatch, fault)
    code, rec = _go(fx, tmp_path)
    assert code == L.EXIT_COMPLETED, rec.get("void")
    assert rec["original_reproduces"] == {"fc_S0_original": False, "fc_S21_original": False,
                                          "out_S0_original": True, "out_S21_original": True}
    cases = _cases(tmp_path)
    for st in ("S0", "S21"):
        assert all(c.get("skipped") for cid, c in cases.items() if cid.startswith(f"fc_{st}_") and c["targeted"])
        for suffix in [f"rows_{n}" for n in ROWS] + ["layout", "context"]:
            assert "skipped" not in cases[f"out_{st}_{suffix}"], suffix
        c = cases[f"out_{st}_original"]["implementations"]["explicit_products"]["gpu"]
        assert c["failing_outputs"] == ["dX"] and c["gross"] is True
        j = c["outputs"]["dX"][0]
        assert j["legal"]["outside_tol"] == 0 and j["padded"]["rows_with_gross"] == 6 and j["first_failing"]
        assert cases[f"out_{st}_original"]["implementations"]["linear_autodiff"]["gpu"]["failing_outputs"] == []
    assert cases["out_S0_rows_4"]["operands"]["X"]["shape"] == [4, 64]
    assert cases["out_S0_layout"]["operands"]["X"]["layout"].startswith("transposed view")
    lines = (tmp_path / "ev" / "artifacts.jsonl").read_text().splitlines()
    saved = {json.loads(x)["name"]: json.loads(x) for x in lines}
    z = np.load(tmp_path / "saved" / "out_S0_original.npz")
    assert {"operand.X", "operand.C", "operand.legal", "expected.dX", "explicit_products.gpu.0.dX"} <= set(z.files)
    assert hashlib.sha256((tmp_path / "saved" / "out_S0_original.npz").read_bytes()).hexdigest() == \
        saved["out_S0_original"]["sha256"]
    assert not os.access(tmp_path / "saved" / "out_S0_original.npz", os.W_OK)


def test_A_NON_REPEATABLE_RESULT_reproduces_even_within_the_tolerance(fx, tmp_path, monkeypatch):
    calls = []

    def fault(impl, o, got):
        calls.append(1)
        return {**got, "Y": got["Y"] * np.float32(1 + 1e-7 * len(calls))}

    _fault(monkeypatch, fault)
    code, rec = _go(fx, tmp_path)
    assert code == L.EXIT_COMPLETED and rec["original_reproduces"]["fc_S0_original"] is True
    c = _cases(tmp_path)["fc_S0_original"]["implementations"]["linear_autodiff"]["gpu"]
    assert c["repeats_bitwise"]["Y"] == [False, False] and all(j["outside_tol"] == 0 for j in c["outputs"]["Y"])


def test_A_TOLERANCE_LEVEL_DIFFERENCE_is_recorded_but_does_not_reproduce(fx, tmp_path, monkeypatch):
    _fault(monkeypatch, lambda impl, o, got: {**got, "dW": got["dW"] * np.float32(1 + 1e-3)})
    code, rec = _go(fx, tmp_path)
    assert code == L.EXIT_COMPLETED and rec["isolation_reproduces"] is False
    c = _cases(tmp_path)["fc_S0_original"]
    assert c["outside_tol_anywhere"] is True and c["reproduces"] is False
    assert c["implementations"]["explicit_products"]["gpu"]["failing_outputs"] == ["dW"]
    assert all(x.get("skipped") for x in _cases(tmp_path).values() if x["targeted"])


def test_THE_FC_OPERANDS_ARE_THE_REFERENCE_S_GATHER_AND_COTANGENT(fx):
    ops = L.operands(fx["spec"], _seams(fx))
    z = L._npz(fx["files"]["reference_real_S0"])
    p = {k[6:]: v for k, v in z.items() if k.startswith("param.")}
    _, g, inter = K.np_policy_head(p, z["F"], z["rows"], z["cols"], z["mask"], z["target"], z["w"])
    o = ops[("fc", "S0")]
    assert np.array_equal(o["X"], inter["gathered"].reshape(-1, 2).astype(np.float32))
    # the reference's own fc gradients follow from these operands: dW = C^T X, db = sum C
    e = L.expected({**o, "X": inter["gathered"].reshape(-1, 2), "C": o["C"].astype(np.float64)})
    assert np.allclose(e["db"], g["fc.bias"], rtol=1e-6, atol=1e-9)
    assert np.allclose(e["dW"], g["fc.weight"], rtol=1e-5, atol=1e-9)
    assert o["legal"].sum() == 18 and ops[("out", "S0")]["C"].shape == (24, 1)


@pytest.mark.parametrize("case", ["index_hash", "fixture_hash", "non_finite", "shapes"])
def test_A_BAD_FIXTURE_is_a_VOID_before_any_case(fx, tmp_path, case):
    files, over, spec = dict(fx["files"]), {}, fx["spec"]
    if case == "index_hash":
        over["read_fixture_index"] = lambda: fx["index"] + b" "
    elif case == "fixture_hash":
        files["stage_S0_out"] = files["stage_S0_out"] + b"x"
    else:
        z = L._npz(files["stage_S21_out"])
        if case == "non_finite":
            z["input"][0, 0] = np.nan
        else:
            z["param.weight"] = np.zeros((1, 63), np.float32)
        files["stage_S21_out"] = _npz_bytes(z)
        idx = [json.loads(x) for x in fx["index"].decode().splitlines()]
        for e in idx:
            e["sha256"] = hashlib.sha256(files[e["name"]]).hexdigest()
        raw = "".join(json.dumps(e) + "\n" for e in idx).encode()
        over["read_fixture_index"] = lambda: raw
        spec = {**spec, "fixture_index_sha256": hashlib.sha256(raw).hexdigest()}
    over.setdefault("read_fixture", lambda name: files[name])
    code, rec = _go(fx, tmp_path, spec, **over)
    want = {"index_hash": "fixtures", "fixture_hash": "fixtures", "non_finite": "non_finite", "shapes": "fixtures"}
    assert code == L.EXIT_VOID and rec["void"]["step"] == want[case], rec.get("void")
    assert not (tmp_path / "ev" / "cases.jsonl").exists()


def test_EARLIER_CASES_SURVIVE_SIGTERM(fx, tmp_path, monkeypatch):
    real = L.run_case

    def run_case(case, o, spec, sink):
        out = real(case, o, spec, sink)
        if case["id"] == "fc_S21_original":
            os.kill(os.getpid(), signal.SIGTERM)
        return out

    monkeypatch.setattr(L, "run_case", run_case)
    previous = signal.getsignal(signal.SIGTERM)
    code, rec = _go(fx, tmp_path)
    assert signal.getsignal(signal.SIGTERM) == previous
    assert code == L.EXIT_VOID and rec["void"]["step"] == "terminated"
    assert list(_cases(tmp_path)) == ["fc_S0_original", "fc_S21_original"]


def test_AN_EXISTING_OUTPUT_DIRECTORY_IS_NEVER_REUSED(fx, tmp_path):
    (tmp_path / "saved").mkdir()
    with pytest.raises(L.Void, match="exists"):
        _go(fx, tmp_path)


def test_THE_FROZEN_SPEC_AND_CASE_LIST_ARE_THE_CARD_S():
    assert L.FROZEN["fixture_index_sha256"] == hashlib.sha256((L.ROOT / L.FIXTURE_INDEX).read_bytes()).hexdigest()
    assert (L.FROZEN["rel_tol"], L.FROZEN["abs_tol"], L.FROZEN["gross_rel"]) == (1e-4, 1e-6, 1e-2)
    assert L.FROZEN["row_counts"] == [512, 2048, 8192, 16384, 32768]
    assert (L.FROZEN["gpu_repeats"], L.FROZEN["cpu_repeats"]) == (3, 1)
    ids = [c["id"] for c in L.case_list(L.FROZEN)]
    assert ids[:4] == ["fc_S0_original", "fc_S21_original", "out_S0_original", "out_S21_original"]
    assert len(ids) == len(set(ids)) == 32
    assert sum(1 for c in L.case_list(L.FROZEN) if c["layout"] != "contiguous" and c["context"] != "joint_eval") == 0
    assert not L.OUTPUT_PARENT.is_relative_to(L.ROOT) and not L.FIXTURE_DIR.is_relative_to(L.ROOT)


def test_THE_PUBLIC_ENTRY_refuses_with_its_gate_closed_and_with_an_unpinned_card(monkeypatch, capsys):
    assert L.GATE.SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED is False
    existed = (L.ROOT / L.EVIDENCE_DIR).exists()
    assert L.main(["run"]) == L.EXIT_REFUSED and "is closed" in capsys.readouterr().err
    monkeypatch.setattr(L.GATE, "SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED", True)
    monkeypatch.setattr(L, "CARD_SHA256", "1" * 64)
    assert L.main(["run"]) == L.EXIT_REFUSED and "not the pinned card" in capsys.readouterr().err
    assert L.main(["run", "--checkpoint", "x"]) == L.EXIT_USAGE and (L.ROOT / L.EVIDENCE_DIR).exists() == existed


def test_THE_CARD_IS_PINNED():
    assert hashlib.sha256((L.ROOT / L.CARD).read_bytes()).hexdigest() == L.CARD_SHA256


def test_THE_ALTERNATIVE_LAYOUT_AND_CONTEXT_ARE_REALLY_USED(monkeypatch):
    """The transposed-view layout builds X from an (in x N) array; the separate context evaluates each output alone."""
    rng = np.random.default_rng(1)
    o = {"X": rng.normal(0, 1, (40, 2)).astype(np.float32), "W": rng.normal(0, 1, (64, 2)).astype(np.float32),
         "b": rng.normal(0, 1, 64).astype(np.float32), "C": rng.normal(0, 1, (40, 64)).astype(np.float32),
         "legal": np.ones(40, bool)}
    shapes, evals = [], []
    real_array, real_eval = mx.array, mx.eval
    monkeypatch.setattr(mx, "array", lambda a, *k, **kw: (shapes.append(np.shape(a)), real_array(a, *k, **kw))[1])
    monkeypatch.setattr(mx, "eval", lambda *a: (evals.append(len(a)), real_eval(*a))[1])
    L.compute("explicit_products", o, "x_transposed_view", "joint_eval")
    assert (2, 40) in shapes and (40, 2) not in shapes and evals == [4]
    shapes.clear(), evals.clear()
    L.compute("explicit_products", o, "contiguous", "separate_eval")
    assert (40, 2) in shapes and evals == [1, 1, 1, 1]


def test_A_NON_FINITE_RESULT_is_a_VOID_and_the_cases_before_it_are_kept(fx, tmp_path, monkeypatch):
    def fault(impl, o, got):
        if o["X"].shape[1] == 64:
            return {**got, "Y": got["Y"] * np.float32(np.nan)}
        return got

    _fault(monkeypatch, fault)
    code, rec = _go(fx, tmp_path)
    assert code == L.EXIT_VOID and rec["void"]["step"] == "non_finite" and "out_S0_original" in rec["void"]["detail"]
    assert list(_cases(tmp_path)) == ["fc_S0_original", "fc_S21_original"]
