#!/usr/bin/env python3
"""MLX MATMUL ROW-COUNT PROBE -- STANDALONE: imports only the Python standard library, NumPy and MLX.

Copy this one file anywhere and run it:

    python mlx_matmul_rows_probe.py --out NEW_DIR [--example-dir NEW_DIR]

What it checks. A linear layer's four float32 operations, on deterministic synthetic operands, against NumPy:
    Y = X @ W.T + b      dX = C @ W      dW = C.T @ X      db = sum(C, axis=0)
for three layers (in -> out features): fc 2 -> 64 and out 64 -> 1 (the shapes that failed in a TwixT policy head
under MLX 0.30.3 on an Apple M5 Max GPU), and one wider control, 64 -> 64; each at a fixed list of row counts N
around 32,736 and 32,768 plus the two failing sizes, 33,600 and 33,728. No adaptive search: the case list is SPEC.

Each case runs on the GPU three times and on the CPU once. Every output is evaluated (mx.eval) before it is copied to
NumPy. The expected result is NumPy's float64 accumulation of the same float32 operands; it does not use MLX.

Judgement (per output, per element; frozen in SPEC):
  * scale: max|expected| for Y, dX and dW. For db, whose cotangent columns are centred so that the sum nearly
    cancels, the scale is ABSOLUTE: max over columns of sum_rows |C|, the magnitude of the terms being summed. A
    near-zero expected value therefore cannot make a small error look gross.
  * outside the tolerance: |got - expected| > rel_tol * scale + abs_tol; GROSS: > gross_rel * scale.
  * for Y and dX (one row per row of X), the rows holding a gross element are reported as [start, stop) ranges,
    with the largest absolute error inside and outside those rows.
  * GPU repeats are compared bitwise with the first; the largest difference is reported.

Outputs, create-only (both directories must not exist):
  OUT_DIR: environment.json (written first), progress.jsonl and cases.jsonl (one line per case, flushed and fsynced
           as each finishes), artifacts.jsonl, record.json (on every exit it survives; SIGTERM -> VOID).
  EXAMPLE_DIR (default OUT_DIR): compact_example.npz, read-only -- ONE compact failing example, chosen by a fixed
           rule: the first case (in case order) whose GPU fc forward Y has gross rows; failing that, the first case
           with gross rows in any GPU Y or dX. Its operands and the expected and observed rows [0, last gross row +
           margin), capped. Nothing else is saved: full outputs are not written.

Exit: 0 COMPLETED | 3 VOID (non-finite operand or result, exception, SIGTERM) | 4 REFUSED (an output directory
exists) | 2 USAGE. A difference from the expected result is an observation; it never stops the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

RECORD_KIND = "mlx-matmul-rows-probe-record/1"
SPEC: Dict[str, Any] = {
    "seed": 20261023,
    "layers": {"fc": [2, 64], "out": [64, 1], "wide": [64, 64]},       # name -> [in_features, out_features]
    "row_counts": [32704, 32736, 32737, 32767, 32768, 32769, 32800, 33600, 33728],
    "outputs": ["Y", "dX", "dW", "db"],
    "row_outputs": ["Y", "dX"],
    "formulas": {"Y": "X @ W.T + b", "dX": "C @ W", "dW": "C.T @ X", "db": "sum(C, axis=0)"},
    "gpu_repeats": 3, "cpu_repeats": 1,
    "rel_tol": 1e-4, "abs_tol": 1e-6, "gross_rel": 1e-2,
    "db_scale": "max_j sum_i |C_ij|",
    "failing_index_cap": 20, "range_cap": 64,
    "example_margin_rows": 8, "example_rows_cap": 4096, "example_operand_cap_bytes": 4 * 1024 * 1024,
}
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = 0, 2, 3, 4
DEVICES: Dict[str, Any] = {}                       # name -> MLX device; filled at first use (tests may remap)
CURRENT = {"device": None}                         # the device label a computation is running under


class Void(Exception):
    def __init__(self, step: str, detail: str):
        super().__init__(f"{step}: {detail}")
        self.step, self.detail = step, detail


class Terminated(BaseException):
    """SIGTERM arrived (the caller's ceiling)."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ─────────────────────────────── operands and the expected result ───────────────────────────────

def operands(layer: str, n: int, spec=SPEC) -> Dict[str, np.ndarray]:
    """Deterministic float32 operands for one case, from the seed, the layer's index and n. C's columns are centred
    in float64 before the cast, so db = sum(C) nearly cancels, as a softmax cotangent's does."""
    fin, fout = spec["layers"][layer]
    rng = np.random.default_rng([spec["seed"], list(spec["layers"]).index(layer), n])
    X = rng.standard_normal((n, fin)).astype(np.float32)
    W = (rng.standard_normal((fout, fin)) / np.sqrt(fin)).astype(np.float32)
    b = rng.standard_normal(fout).astype(np.float32)
    C = rng.standard_normal((n, fout))
    C = (C - C.mean(axis=0)).astype(np.float32)
    return {"X": X, "W": W, "b": b, "C": C}


def expected(o) -> Dict[str, np.ndarray]:
    """NumPy float64 accumulation of the float32 operands."""
    X, W, b, C = (o[k].astype(np.float64) for k in ("X", "W", "b", "C"))
    return {"Y": X @ W.T + b, "dX": C @ W, "dW": C.T @ X, "db": C.sum(axis=0)}


def scales(o, ref) -> Dict[str, float]:
    """max|expected| for Y, dX, dW; for db the absolute magnitude of its terms, max_j sum_i |C_ij|."""
    out = {k: float(np.abs(ref[k]).max()) for k in ("Y", "dX", "dW")}
    out["db"] = float(np.abs(o["C"].astype(np.float64)).sum(axis=0).max())
    return out


# ─────────────────────────────── the MLX computation ───────────────────────────────

class _On:
    def __init__(self, device: str):
        self.device = device

    def __enter__(self):
        import mlx.core as mx
        if not DEVICES:
            DEVICES.update({"cpu": mx.cpu, "gpu": mx.gpu})
        self.previous = mx.default_device()
        mx.set_default_device(DEVICES[self.device])
        CURRENT["device"] = self.device

    def __exit__(self, *exc):
        import mlx.core as mx
        mx.set_default_device(self.previous)
        CURRENT["device"] = None


def compute(o) -> Dict[str, np.ndarray]:
    """The four operations on the current device, evaluated jointly, copied to float32 NumPy."""
    import mlx.core as mx
    X, W, b, C = (mx.array(o[k]) for k in ("X", "W", "b", "C"))
    outs = {"Y": X @ W.T + b, "dX": C @ W, "dW": C.T @ X, "db": mx.sum(C, axis=0)}
    mx.eval(*outs.values())
    return {k: np.array(v) for k, v in outs.items()}


# ─────────────────────────────── judging ───────────────────────────────

def ranges(rows: np.ndarray) -> List[List[int]]:
    """A 1-D boolean array -> its True runs as [start, stop)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], rows.astype(np.int8), [0]])))
    return edges.reshape(-1, 2).tolist()


def judge(name: str, got: np.ndarray, ref: np.ndarray, scale: float, o, spec=SPEC) -> Dict[str, Any]:
    if got.shape != ref.shape:
        return {"shape_mismatch": [list(got.shape), list(ref.shape)], "outside_tol": -1, "gross": -1}
    err = np.abs(got.astype(np.float64) - ref)
    tol, gross_at = spec["rel_tol"] * scale + spec["abs_tol"], spec["gross_rel"] * scale
    out: Dict[str, Any] = {"scale": scale, "tol": tol, "gross_threshold": gross_at, "max_abs": float(err.max()),
                           "outside_tol": int((err > tol).sum()), "gross": int((err > gross_at).sum())}
    if name in spec["row_outputs"]:
        gross_rows = (err > gross_at).any(axis=1)
        r = ranges(gross_rows)
        out.update(rows=int(err.shape[0]), rows_outside_tol=int((err > tol).any(axis=1).sum()),
                   rows_gross=int(gross_rows.sum()), gross_row_ranges=r[:spec["range_cap"]],
                   gross_row_range_count=len(r),
                   max_abs_in_gross_rows=float(err[gross_rows].max()) if gross_rows.any() else 0.0,
                   max_abs_in_other_rows=float(err[~gross_rows].max()) if (~gross_rows).any() else 0.0,
                   gross_rows_near_zero=int((np.abs(got[gross_rows]).max(axis=1) <= gross_at).sum())
                   if gross_rows.any() else 0)
        if name == "Y":
            out["gross_rows_near_bias"] = int((np.abs(got[gross_rows] - o["b"]).max(axis=1) <= gross_at).sum()) \
                if gross_rows.any() else 0
    bad = np.argwhere(err > tol)
    out["first_failing"] = [[int(i) for i in ix] + [float(got[tuple(ix)]), float(ref[tuple(ix)])]
                            for ix in bad[:spec["failing_index_cap"]]]
    return out


# ─────────────────────────────── one case ───────────────────────────────

def run_case(layer: str, n: int, spec=SPEC) -> tuple:
    """(the case's result, (operands, expected, GPU repeat 0) for the example rule)."""
    o = operands(layer, n, spec)
    for k, v in o.items():
        if not np.all(np.isfinite(v)):
            raise Void("non_finite", f"{layer} n={n}: operand {k} is not finite")
    ref = expected(o)
    sc = scales(o, ref)
    fin, fout = spec["layers"][layer]
    result: Dict[str, Any] = {
        "id": f"{layer}_{n}", "layer": layer, "rows": n, "in_features": fin, "out_features": fout,
        "products": {"Y": [[n, fin], [fin, fout]], "dX": [[n, fout], [fout, fin]], "dW": [[fout, n], [n, fin]],
                     "db": [[n, fout], "sum over rows"]},
        "operand_sha256": {k: _sha(np.ascontiguousarray(v).tobytes()) for k, v in o.items()},
        "scales": sc, "devices": {}}
    runs_by_dev = {}
    for dev, reps in (("gpu", spec["gpu_repeats"]), ("cpu", spec["cpu_repeats"])):
        runs = []
        with _On(dev):
            for i in range(reps):
                got = compute(o)
                for k, v in got.items():
                    if not np.all(np.isfinite(v)):
                        result["non_finite"] = {"device": dev, "repeat": i, "output": k}
                        raise Void("non_finite", f"{result['id']} on {dev}, repeat {i}: {k} is not finite")
                runs.append(got)
        runs_by_dev[dev] = runs
        result["devices"][dev] = {
            "outputs": {k: [judge(k, r[k], ref[k], sc[k], o, spec) for r in runs] for k in spec["outputs"]},
            "repeats_bitwise": {k: [bool(np.array_equal(r[k], runs[0][k])) for r in runs[1:]]
                                for k in spec["outputs"]},
            "repeats_max_abs_diff": {k: [float(np.abs(r[k].astype(np.float64) - runs[0][k]).max()) for r in runs[1:]]
                                     for k in spec["outputs"]}}
        d = result["devices"][dev]
        d["gross_outputs"] = [k for k in spec["outputs"] if any(j["gross"] for j in d["outputs"][k])]
        d["outside_tol_outputs"] = [k for k in spec["outputs"] if any(j["outside_tol"] for j in d["outputs"][k])]
        d["nonrepeatable_outputs"] = [k for k in spec["outputs"] if not all(d["repeats_bitwise"][k])]
    return result, (o, ref, runs_by_dev["gpu"][0])


def example_arrays(case, src, spec=SPEC) -> Optional[Dict[str, Any]]:
    """The compact example for a case with gross GPU rows in Y (preferred) or dX: the operands that product needs
    (in full if they fit the cap, so the example runs by itself; otherwise only the rows shown, and the seed
    regenerates the rest), the row indices shown, and the expected and observed (GPU repeat 0) rows."""
    o, ref, got = src
    first = case["devices"]["gpu"]["outputs"]
    name = next((k for k in ("Y", "dX") if first[k][0].get("rows_gross")), None)
    if name is None:
        return None
    stop = first[name][0]["gross_row_ranges"][-1][1] if first[name][0]["gross_row_range_count"] <= spec["range_cap"] \
        else case["rows"]
    k = min(stop + spec["example_margin_rows"], case["rows"], spec["example_rows_cap"])
    needs = ("X", "W", "b") if name == "Y" else ("C", "W")
    full = sum(o[x].nbytes for x in needs) <= spec["example_operand_cap_bytes"]
    arrays = {f"operand.{x}": (o[x] if full or o[x].shape[0] != case["rows"] else o[x][:k]) for x in needs}
    arrays.update({"rows": np.arange(k, dtype=np.int64), f"expected.{name}": ref[name][:k],
                   f"observed_gpu_repeat0.{name}": got[name][:k],
                   "seed_entropy": np.array([spec["seed"], list(spec["layers"]).index(case["layer"]), case["rows"]],
                                            dtype=np.int64)})
    return {"output": name, "operands_full": full, "arrays": arrays}


# ─────────────────────────────── the run ───────────────────────────────

def environment() -> Dict[str, Any]:
    import mlx.core as mx
    env: Dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "mac_ver": platform.mac_ver()[0],
                           "machine": platform.machine(), "numpy": np.__version__,
                           "mlx": getattr(mx, "__version__", None), "executable": sys.executable,
                           "generator": os.path.abspath(__file__),
                           "generator_sha256": _sha(pathlib.Path(__file__).read_bytes()),
                           "default_device": str(mx.default_device()), "loadavg": os.getloadavg()}
    try:
        env["metal_device_info"] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                                    for k, v in mx.metal.device_info().items()}
    except Exception as e:                                    # recorded, never fatal
        env["metal_device_info"] = f"unavailable: {type(e).__name__}: {e}"
    return env


class _Writer:
    def __init__(self, out_dir: pathlib.Path):
        self.out_dir, self.handles = out_dir, {}

    def line(self, rel, obj):
        fh = self.handles.get(rel) or self.handles.setdefault(rel, open(self.out_dir / rel, "x"))
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def new_json(self, rel, obj):
        with open(self.out_dir / rel, "x") as fh:
            json.dump(obj, fh, indent=1)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

    def close(self):
        for fh in self.handles.values():
            fh.close()


def _summary(case, spec=SPEC) -> Dict[str, Any]:
    out = {}
    for dev, d in case["devices"].items():
        out[dev] = {k: {"gross": [j["gross"] for j in d["outputs"][k]],
                        "max_abs": [j["max_abs"] for j in d["outputs"][k]],
                        "max_abs_over_scale": max(j["max_abs"] / (j["scale"] or 1.0) for j in d["outputs"][k]),
                        **({"gross_row_ranges": [j["gross_row_ranges"] for j in d["outputs"][k]]}
                           if k in spec["row_outputs"] else {})}
                    for k in spec["outputs"]}
        out[dev]["nonrepeatable_outputs"] = d["nonrepeatable_outputs"]
    return out


def _save_example(ex, case, example_dir: pathlib.Path, writer: _Writer) -> Dict[str, Any]:
    example_dir.mkdir(parents=True, exist_ok=True)
    path = example_dir / "compact_example.npz"
    tmp = example_dir / f".compact_example.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **ex["arrays"])
    try:
        os.link(tmp, path)                                    # create-only
    finally:
        os.unlink(tmp)
    os.chmod(path, 0o444)
    entry = {"name": "compact_example", "path": str(path), "sha256": _sha(path.read_bytes()),
             "bytes": path.stat().st_size, "case": case["id"], "output": ex["output"],
             "operands_full": ex["operands_full"],
             "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in ex["arrays"].items()}}
    writer.line("artifacts.jsonl", entry)
    return entry


def execute(out_dir: pathlib.Path, example_dir: pathlib.Path, spec=SPEC) -> tuple:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    writer = _Writer(out_dir)
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "status": None, "spec": spec, "out_dir": str(out_dir),
                           "example_dir": str(example_dir), "cases_done": [], "example": None}

    def log(event, **kw):
        writer.line("progress.jsonl", {"event": event, "t": round(time.monotonic() - t0, 3), **kw})

    def on_term(signum, _frame):
        raise Terminated(signal.Signals(signum).name)

    code = EXIT_VOID
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main_thread else None
    try:
        log("start")
        rec["environment"] = environment()
        writer.new_json("environment.json", rec["environment"])
        cases = [(layer, n) for layer in spec["layers"] for n in spec["row_counts"]]
        rec["case_list"] = [f"{layer}_{n}" for layer, n in cases]
        fallback = None
        summary = {}
        for layer, n in cases:
            try:
                case, src = run_case(layer, n, spec)
            except Void as v:
                writer.line("cases.jsonl", {"id": f"{layer}_{n}", "layer": layer, "rows": n,
                                            "void": {"step": v.step, "detail": v.detail}})
                raise
            writer.line("cases.jsonl", case)
            rec["cases_done"].append(case["id"])
            summary[case["id"]] = _summary(case, spec)
            log("case", id=case["id"], gpu_gross=case["devices"]["gpu"]["gross_outputs"],
                cpu_gross=case["devices"]["cpu"]["gross_outputs"],
                gpu_nonrepeatable=case["devices"]["gpu"]["nonrepeatable_outputs"])
            if rec["example"] is None:
                ex = example_arrays(case, src, spec)
                if ex is not None and ex["output"] == "Y" and layer == "fc":
                    rec["example"] = _save_example(ex, case, example_dir, writer)
                elif ex is not None and fallback is None:
                    fallback = (ex, case)
        if rec["example"] is None and fallback is not None:
            rec["example"] = _save_example(fallback[0], fallback[1], example_dir, writer)
        rec["summary"] = summary
        rec["cases_with_gpu_gross"] = [c for c, s in summary.items() if any(any(s["gpu"][k]["gross"])
                                                                            for k in spec["outputs"])]
        rec["cases_with_cpu_gross"] = [c for c, s in summary.items() if any(any(s["cpu"][k]["gross"])
                                                                            for k in spec["outputs"])]
        rec["cases_gpu_nonrepeatable"] = [c for c, s in summary.items() if s["gpu"]["nonrepeatable_outputs"]]
        rec["status"] = "COMPLETED"
        code = EXIT_COMPLETED
    except Terminated as t:
        rec.update(status="VOID", void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"])
            writer.close()
            with open(out_dir / "record.json", "x") as fh:
                json.dump(rec, fh, indent=1)
                fh.write("\n")
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MLX matmul row-count probe (standalone).")
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--example-dir", type=pathlib.Path, default=None)
    try:
        a = p.parse_args(argv)
    except SystemExit:
        return EXIT_USAGE
    example_dir = a.example_dir if a.example_dir is not None else a.out
    for d in {a.out, example_dir}:
        if os.path.lexists(d):
            print(f"REFUSED: {d} exists (create-only)", file=sys.stderr)
            return EXIT_REFUSED
    code, rec = execute(a.out, example_dir)
    print(json.dumps({"status": rec["status"], "void": rec.get("void"),
                      "cases_with_gpu_gross": rec.get("cases_with_gpu_gross"),
                      "cases_with_cpu_gross": rec.get("cases_with_cpu_gross"),
                      "cases_gpu_nonrepeatable": rec.get("cases_gpu_nonrepeatable"),
                      "example": (rec.get("example") or {}).get("sha256")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
