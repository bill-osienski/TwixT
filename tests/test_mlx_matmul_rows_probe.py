"""The standalone MLX matmul row-count probe (`scripts/GPU/alphazero/mlx_matmul_rows_probe.py`).

Every test runs on the CPU: the "gpu" label is remapped to the CPU device, and faults are injected keyed on the
label a computation runs under (`P.CURRENT["device"]`). Most tests use a reduced SPEC (small row counts) so the
judgement, ranges, example rule, stops and create-only outputs can be seen exactly; the frozen SPEC itself runs end
to end in the gated entry's test."""
import ast
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest

from scripts.GPU.alphazero import mlx_matmul_rows_probe as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMALL = dict(P.SPEC, row_counts=[256, 300, 301, 333])


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import mlx.core as mx
    monkeypatch.setattr(P, "DEVICES", {"gpu": mx.cpu, "cpu": mx.cpu})
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _inject(monkeypatch, fault):
    """fault(o, out, device, call_index) mutates `out` in place; every call of compute() goes through it."""
    real, calls = P.compute, {"n": 0}

    def compute(o):
        out = real(o)
        fault(o, out, P.CURRENT["device"], calls["n"])
        calls["n"] += 1
        return out
    monkeypatch.setattr(P, "compute", compute)


def _leading(name, shape_w, boundary):
    """The observed failure's shape: on the 'gpu', rows [0, n - boundary) of `name` wrong, for n > boundary + 10."""
    def fault(o, out, dev, _):
        n = o["X"].shape[0]
        if dev == "gpu" and o["W"].shape == shape_w and n > boundary + 10:
            out[name][: n - boundary] = o["b"] if name == "Y" else 0.0
    return fault


def _run(tmp_path, spec=SMALL, example=True):
    out, ex = tmp_path / "out", (tmp_path / "ex" if example else tmp_path / "out")
    code, rec = P.execute(out, ex, spec)
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()] \
        if (out / "cases.jsonl").exists() else []
    return code, rec, cases, out, ex


# ─────────────────────────────── frozen, standalone ───────────────────────────────

def test_THE_SPEC_IS_THE_FROZEN_ONE():
    s = P.SPEC
    assert s["layers"] == {"fc": [2, 64], "out": [64, 1], "wide": [64, 64]}
    assert s["row_counts"] == [32704, 32736, 32737, 32767, 32768, 32769, 32800, 33600, 33728]
    assert (s["gpu_repeats"], s["cpu_repeats"]) == (3, 1)
    assert (s["rel_tol"], s["abs_tol"], s["gross_rel"]) == (1e-4, 1e-6, 1e-2)
    assert s["outputs"] == ["Y", "dX", "dW", "db"] and s["row_outputs"] == ["Y", "dX"]
    assert s["seed"] == 20261023


def test_ONLY_THE_STANDARD_LIBRARY_NUMPY_AND_MLX_ARE_IMPORTED():
    tree = ast.parse(pathlib.Path(P.__file__).read_text())
    allowed = {"__future__", "argparse", "hashlib", "json", "os", "pathlib", "platform", "signal", "sys",
               "threading", "time", "typing", "numpy", "mlx"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import ties the probe to this repository"
            seen.add(node.module.split(".")[0])
    assert seen <= allowed, seen - allowed


def test_THE_SCRIPT_LOADS_BY_ITSELF_outside_the_repository(tmp_path):
    """A copy run with -I (no repository on the path, no current-directory imports) gets as far as its argument
    check: a missing --out is USAGE (2), before MLX is touched."""
    copy = tmp_path / "solo" / "mlx_matmul_rows_probe.py"
    copy.parent.mkdir()
    shutil.copy(P.__file__, copy)
    r = subprocess.run([sys.executable, "-I", str(copy)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert not (tmp_path / "out").exists()


# ─────────────────────────────── operands and the expected result ───────────────────────────────

def test_OPERANDS_ARE_DETERMINISTIC_DISTINCT_PER_CASE_AND_THE_DB_SUM_CANCELS():
    a, b = P.operands("fc", 301), P.operands("fc", 301)
    assert all(a[k].tobytes() == b[k].tobytes() and a[k].dtype == np.float32 for k in a)
    c = P.operands("fc", 300)
    assert a["X"][:300].tobytes() != c["X"].tobytes()
    assert P.operands("wide", 301)["C"].tobytes() != P.operands("out", 301)["C"].tobytes()
    for layer in P.SPEC["layers"]:
        C = P.operands(layer, 333)["C"].astype(np.float64)
        assert np.abs(C.sum(axis=0)).max() <= 1e-4 * np.abs(C).sum(axis=0).max()


def test_THE_EXPECTED_RESULT_IS_AN_INDEPENDENT_FLOAT64_SUM():
    o = P.operands("fc", 5)
    X, W, b, C = (o[k] for k in ("X", "W", "b", "C"))
    ref = P.expected(o)
    assert all(v.dtype == np.float64 for v in ref.values())
    for i in range(5):
        for j in range(64):
            y = float(b[j]) + sum(float(X[i, k]) * float(W[j, k]) for k in range(2))
            assert abs(ref["Y"][i, j] - y) <= 1e-12
        for k in range(2):
            assert abs(ref["dX"][i, k] - sum(float(C[i, j]) * float(W[j, k]) for j in range(64))) <= 1e-12
    for j in range(64):
        assert abs(ref["db"][j] - sum(float(C[i, j]) for i in range(5))) <= 1e-12
        for k in range(2):
            assert abs(ref["dW"][j, k] - sum(float(C[i, j]) * float(X[i, k]) for i in range(5))) <= 1e-12


def test_RANGES_ARE_THE_TRUE_RUNS():
    assert P.ranges(np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], bool)) == [[0, 2], [4, 5], [6, 9]]
    assert P.ranges(np.zeros(4, bool)) == []
    assert P.ranges(np.ones(3, bool)) == [[0, 3]]


def test_THE_JUDGEMENT_SEPARATES_TOLERANCE_FROM_GROSS():
    o = P.operands("fc", 20)
    ref = P.expected(o)
    sc = P.scales(o, ref)
    got = ref["Y"].astype(np.float32).copy()
    got[3, 4] += 5 * (1e-4 * sc["Y"] + 1e-6)              # outside the tolerance, far from gross
    got[7, 1] += 2 * 1e-2 * sc["Y"]                       # gross
    j = P.judge("Y", got, ref["Y"], sc["Y"], o)
    assert j["outside_tol"] == 2 and j["gross"] == 1 and j["rows_gross"] == 1 and j["gross_row_ranges"] == [[7, 8]]
    assert j["rows_outside_tol"] == 2


def test_THE_DB_SCALE_IS_ABSOLUTE_so_cancellation_cannot_make_a_gross_verdict():
    o = P.operands("out", 333)
    ref = P.expected(o)
    sc = P.scales(o, ref)
    assert sc["db"] == pytest.approx(np.abs(o["C"].astype(np.float64)).sum(axis=0).max())
    assert sc["db"] > 1e4 * np.abs(ref["db"]).max()       # the cancellation the absolute scale exists for
    near = ref["db"] + 1e-3                                # a float32-sized summation error
    assert P.judge("db", near.astype(np.float32), ref["db"], sc["db"], o)["gross"] == 0
    far = ref["db"] + 2e-2 * sc["db"]                      # and a real error still is gross
    assert P.judge("db", far.astype(np.float32), ref["db"], sc["db"], o)["gross"] == 1


# ─────────────────────────────── runs ───────────────────────────────

def test_A_CLEAN_RUN_COMPLETES_with_nothing_gross_and_no_example(tmp_path):
    code, rec, cases, out, ex = _run(tmp_path)
    assert (code, rec["status"]) == (0, "COMPLETED")
    assert [c["id"] for c in cases] == rec["case_list"] == [f"{L}_{n}" for L in ("fc", "out", "wide")
                                                           for n in SMALL["row_counts"]]
    assert rec["cases_with_gpu_gross"] == rec["cases_with_cpu_gross"] == rec["cases_gpu_nonrepeatable"] == []
    assert rec["example"] is None and not ex.exists() and not (out / "artifacts.jsonl").exists()
    for c in cases:
        for dev, reps in (("gpu", 3), ("cpu", 1)):
            d = c["devices"][dev]
            assert all(len(d["outputs"][k]) == reps for k in P.SPEC["outputs"])
            assert all(j["gross"] == 0 for k in P.SPEC["outputs"] for j in d["outputs"][k])
    env = json.loads((out / "environment.json").read_text())
    assert env["mlx"] and env["numpy"] == np.__version__
    assert env["generator_sha256"] == P._sha(pathlib.Path(P.__file__).read_bytes())
    assert json.loads((out / "record.json").read_text())["status"] == "COMPLETED"


def test_INJECTED_LEADING_ROW_CORRUPTION_is_located_exactly_and_the_example_is_fc_Y(tmp_path, monkeypatch):
    _inject(monkeypatch, _leading("Y", (64, 2), 290))
    code, rec, cases, out, ex = _run(tmp_path)
    assert code == 0 and rec["cases_with_gpu_gross"] == ["fc_301", "fc_333"] and rec["cases_with_cpu_gross"] == []
    by = {c["id"]: c for c in cases}
    for n in (301, 333):
        ys = by[f"fc_{n}"]["devices"]["gpu"]["outputs"]["Y"]
        assert [j["gross_row_ranges"] for j in ys] == [[[0, n - 290]]] * 3
        assert ys[0]["rows_gross"] == n - 290 == ys[0]["gross_rows_near_bias"]
        assert by[f"fc_{n}"]["devices"]["cpu"]["outputs"]["Y"][0]["rows_gross"] == 0
    assert by["fc_300"]["devices"]["gpu"]["gross_outputs"] == []
    e = rec["example"]
    assert (e["case"], e["output"], e["operands_full"]) == ("fc_301", "Y", True)
    path = ex / "compact_example.npz"
    assert oct(path.stat().st_mode & 0o777) == "0o444"
    line = json.loads((out / "artifacts.jsonl").read_text())
    assert line["sha256"] == P._sha(path.read_bytes()) == e["sha256"]
    z = np.load(path)
    k = 11 + P.SPEC["example_margin_rows"]
    assert z["rows"].tolist() == list(range(k)) and z["operand.X"].shape == (301, 2)
    assert np.array_equal(z["observed_gpu_repeat0.Y"][:11], np.broadcast_to(z["operand.b"], (11, 64)))
    assert z["seed_entropy"].tolist() == [P.SPEC["seed"], 0, 301]
    regenerated = P.operands("fc", 301)
    assert np.array_equal(z["operand.X"], regenerated["X"])


def test_WITHOUT_AN_FC_FORWARD_FAILURE_the_example_falls_back_to_the_first_gross_row_output(tmp_path, monkeypatch):
    _inject(monkeypatch, _leading("dX", (1, 64), 300))
    code, rec, cases, out, ex = _run(tmp_path)
    assert code == 0 and rec["cases_with_gpu_gross"] == ["out_333"]
    e = rec["example"]
    assert (e["case"], e["output"]) == ("out_333", "dX")
    z = np.load(ex / "compact_example.npz")
    assert set(z.files) == {"operand.C", "operand.W", "rows", "expected.dX", "observed_gpu_repeat0.dX",
                            "seed_entropy"}
    assert len(z["rows"]) == 33 + P.SPEC["example_margin_rows"]


def test_THE_EXAMPLE_IS_CAPPED(tmp_path, monkeypatch):
    spec = dict(SMALL, example_rows_cap=16, example_operand_cap_bytes=100)
    _inject(monkeypatch, _leading("Y", (64, 2), 290))
    code, rec, cases, out, ex = _run(tmp_path, spec)
    z = np.load(ex / "compact_example.npz")
    assert len(z["rows"]) == 16 and z["operand.X"].shape == (16, 2) and rec["example"]["operands_full"] is False


def test_A_NONREPEATABLE_GPU_OUTPUT_is_reported_with_its_size(tmp_path, monkeypatch):
    def fault(o, out, dev, i):
        if dev == "gpu" and o["W"].shape == (64, 64) and o["X"].shape[0] == 300:
            out["dW"][0, 0] += 0.5 * (i % 3 == 2)          # the third GPU repeat only
    _inject(monkeypatch, fault)
    code, rec, cases, out, ex = _run(tmp_path)
    c = {c["id"]: c for c in cases}["wide_300"]["devices"]["gpu"]
    assert c["nonrepeatable_outputs"] == ["dW"] and c["repeats_bitwise"]["dW"] == [True, False]
    assert c["repeats_max_abs_diff"]["dW"][1] == pytest.approx(0.5, rel=1e-3)
    assert rec["cases_gpu_nonrepeatable"] == ["wide_300"]


def test_A_NON_FINITE_RESULT_VOIDS_after_the_earlier_cases_are_written(tmp_path, monkeypatch):
    def fault(o, out, dev, _):
        if o["W"].shape == (1, 64) and o["X"].shape[0] == 300:
            out["db"][0] = np.nan
    _inject(monkeypatch, fault)
    code, rec, cases, out, ex = _run(tmp_path)
    assert code == 3 and rec["status"] == "VOID" and rec["void"]["step"] == "non_finite"
    assert [c["id"] for c in cases] == ["fc_256", "fc_300", "fc_301", "fc_333", "out_256", "out_300"]
    assert "devices" in cases[4] and cases[5]["void"]["step"] == "non_finite"
    assert json.loads((out / "record.json").read_text())["cases_done"] == ["fc_256", "fc_300", "fc_301", "fc_333",
                                                                          "out_256"]


def test_SIGTERM_VOIDS_with_its_own_record(tmp_path):
    """In a child process, so a SIGTERM the probe failed to handle kills the child, not this test run."""
    child = f"""
import os, signal, mlx.core as mx
mx.set_default_device(mx.cpu)
from scripts.GPU.alphazero import mlx_matmul_rows_probe as P
P.DEVICES.update({{"gpu": mx.cpu, "cpu": mx.cpu}})
real, calls = P.compute, [0]
def compute(o):
    calls[0] += 1
    if calls[0] == 10:
        os.kill(os.getpid(), signal.SIGTERM)
    return real(o)
P.compute = compute
from pathlib import Path
code, rec = P.execute(Path({str(tmp_path / "out")!r}), Path({str(tmp_path / "ex")!r}),
                      dict(P.SPEC, row_counts=[256, 300, 301, 333]))
raise SystemExit(code)
"""
    r = subprocess.run([sys.executable, "-c", child], cwd=ROOT, env=dict(os.environ, PYTHONPATH="."),
                       capture_output=True, text=True)
    assert r.returncode == 3, (r.returncode, r.stderr[-500:])
    rec = json.loads((tmp_path / "out" / "record.json").read_text())
    assert rec["status"] == "VOID" and rec["void"]["step"] == "terminated"
    assert len((tmp_path / "out" / "cases.jsonl").read_text().splitlines()) == 2   # 4 computations per case


def test_WHEN_FC_Y_AND_DX_BOTH_FAIL_the_example_is_the_forward_Y(tmp_path, monkeypatch):
    y, dx = _leading("Y", (64, 2), 290), _leading("dX", (64, 2), 290)
    _inject(monkeypatch, lambda o, out, dev, i: (y(o, out, dev, i), dx(o, out, dev, i)))
    code, rec, cases, out, ex = _run(tmp_path)
    assert (rec["example"]["case"], rec["example"]["output"]) == ("fc_301", "Y")
    assert {c["id"]: c for c in cases}["fc_301"]["devices"]["gpu"]["gross_outputs"] == ["Y", "dX"]


def test_THE_OUTPUTS_ARE_CREATE_ONLY(tmp_path, capsys):
    (tmp_path / "out").mkdir()
    assert P.main(["--out", str(tmp_path / "out")]) == 4
    assert list((tmp_path / "out").iterdir()) == []
    (tmp_path / "ex").mkdir()
    assert P.main(["--out", str(tmp_path / "new"), "--example-dir", str(tmp_path / "ex")]) == 4
    assert not (tmp_path / "new").exists()
    with pytest.raises(FileExistsError):
        P.execute(tmp_path / "out", tmp_path / "ex2", SMALL)
    assert P.main([]) == 2
