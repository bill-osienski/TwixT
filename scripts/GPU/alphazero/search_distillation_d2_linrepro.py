"""SEARCH DISTILLATION D2 -- LINEAR-LAYER REPRODUCER: the policy head's `fc` (2->64) and `out` (64->1) operations,
isolated from the saved gradient-check fixtures, against NumPy. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-linear-reproducer-card.md`, pinned below by sha256. Nothing
runs on import. `run` refuses unless
`search_distillation_d2_linrepro_authorization.SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED` is open, read at call time.

The gradient check (ledger entry 26) found, from identical stage inputs, the GPU's `fc` forward and input gradient
and `out` input gradient outside tolerance and not repeatable. This run asks whether ONE linear operation, alone,
reproduces that. No network is loaded, nothing is trained, no model is replayed: the operands come from the saved
fixtures only (each verified against the gradient check's recorded hash).
  OPERANDS (float32). out: the `stage_<S>_out` fixtures' own input, weight, bias and cotangent. fc: no fixture holds
    fc's input, so it is DERIVED from the `reference_real_<S>` fixtures by the NumPy reference (gather output and the
    analytic cotangent of fc's output, float64, cast to float32); the derived operands are saved with their hashes.
  EXPECTED: NumPy float64 accumulation of the same float32 operands: Y = X W^T + b, dX = C W, dW = C^T X, db = sum C.
  IMPLEMENTATIONS, each on the GPU three times (and the CPU once, a comparison only): nn.Linear with mx.vjp;
    explicit `x @ w.T + b` with mx.vjp; explicit products (the forward and the three gradient formulas as direct
    matrix products, no autodiff). Every output is evaluated (mx.eval) before it is copied to NumPy.
  TARGETED CASES, ONLY for a (layer, state) whose original shape REPRODUCES the failure: five predeclared row counts
    (the first n rows), one alternative layout (X as a transposed view of a contiguous array) and one alternative
    evaluation context (each output evaluated separately), at the original size. No product of these, no search.
  If no (layer, state) reproduces, that is recorded (the failure needs the surrounding context) and nothing more runs.
Every case result is written as it exists; failing elements are reported for legal and padded rows separately and
the outputs of any failing case are saved outside the repository with their hashes. A difference is an observation;
invalid inputs, a non-finite result, an exception or SIGTERM (the wrapper's 10-minute ceiling) are a VOID. No retry.

    python -m scripts.GPU.alphazero.search_distillation_d2_linrepro run
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import search_distillation_d2_gradcheck as K
from . import search_distillation_d2_linrepro_authorization as GATE
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-linrepro-record/1"
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-linear-reproducer-card.md"
CARD_SHA256 = "5ddc3136de055a2f4077c7fe528289e218c0eab51fc3f1bff0d56c0a8b63d5a4"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-linrepro"
FIXTURE_INDEX = "docs/superpowers/evidence/search-distillation-d2-gradcheck/fixtures.jsonl"
FIXTURE_DIR = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2" / "gradcheck-fixtures-20261023T034401Z"
OUTPUT_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"          # outside the repository
FROZEN = {"fixture_index_sha256": "5a5e16e84a4c4f709bce65f26746481504a4127dbe9909fe63311ac226aeb055",
          "states": ["S0", "S21"], "layers": ["fc", "out"],
          "implementations": ["linear_autodiff", "explicit_autodiff", "explicit_products"],
          "outputs": ["Y", "dX", "dW", "db"],
          "gpu_repeats": 3, "cpu_repeats": 1,
          "rel_tol": 1e-4, "abs_tol": 1e-6, "gross_rel": 1e-2,
          "row_counts": [512, 2048, 8192, 16384, 32768],
          "alternative_layout": "x_transposed_view", "alternative_context": "separate_eval",
          "failing_index_cap": 20}
EXIT_COMPLETED, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated = M.Void, M.Terminated
DEVICES: Dict[str, Any] = {}                       # name -> MLX device; filled at first use (tests may remap)
CURRENT = {"device": None}                         # the device label a computation is running under


class Refused(Exception):
    """The run never started: gate closed, card not the pinned bytes, an output exists, or a bad argument."""


def _require_gate() -> None:
    if GATE.SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED is closed: the reproducer is not authorized")


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_fixture_index: Callable[[], bytes]
    read_fixture: Callable[[str], bytes]                 # name -> the .npz bytes


@contextlib.contextmanager
def on(device: str):
    import mlx.core as mx
    if not DEVICES:
        DEVICES.update({"cpu": mx.cpu, "gpu": mx.gpu})
    previous = mx.default_device()
    mx.set_default_device(DEVICES[device])
    CURRENT["device"] = device
    try:
        yield
    finally:
        mx.set_default_device(previous)
        CURRENT["device"] = None


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _array_sha(a: np.ndarray) -> str:
    return _sha(np.ascontiguousarray(a).tobytes()) + f":{a.dtype.str}:{list(a.shape)}"


# ─────────────────────────────── operands from the fixtures ───────────────────────────────

def _npz(raw: bytes) -> Dict[str, np.ndarray]:
    import io
    with np.load(io.BytesIO(raw)) as z:
        return {k: z[k] for k in z.files}


def operands(spec, seams: Seams) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(layer, state): {"X", "W", "b", "C", "legal", "source"}}, float32 (legal: per row of X). Every fixture is
    checked against the gradient check's recorded hash before it is read."""
    raw_index = seams.read_fixture_index()
    if _sha(raw_index) != spec["fixture_index_sha256"]:
        raise Void("fixtures", "the gradient check's fixture index is not the pinned bytes")
    index = {e["name"]: e for e in map(json.loads, raw_index.decode().splitlines())}

    def fixture(name):
        raw = seams.read_fixture(name)
        if name not in index or _sha(raw) != index[name]["sha256"]:
            raise Void("fixtures", f"{name}.npz is not the recorded fixture")
        return _npz(raw)

    out = {}
    for st in spec["states"]:
        z = fixture(f"stage_{st}_out")
        X, C = z["input"].astype(np.float32), z["cotangent"].reshape(-1, 1).astype(np.float32)
        out[("out", st)] = {"X": X, "W": z["param.weight"].astype(np.float32), "b": z["param.bias"].astype(np.float32),
                            "C": C, "legal": z["mask"].reshape(-1) > 0.5, "source": f"stage_{st}_out (as saved)"}
        r = fixture(f"reference_real_{st}")
        p = {k[len("param."):]: v for k, v in r.items() if k.startswith("param.")}
        _, _, inter = K.np_policy_head(p, r["F"], r["rows"], r["cols"], r["mask"], r["target"], r["w"], grads=False)
        m = r["mask"] > 0.5
        B = r["F"].shape[0]
        dlp = -(r["w"].astype(np.float64)[:, None] * r["target"]) / B          # the analytic chain down to fc's output
        dzm = dlp - np.exp(inter["lp"]) * dlp.sum(axis=1, keepdims=True)
        dz = np.where(m, dzm, 0.0)
        da1 = (dz[..., None] * p["out.weight"][0]) * (inter["a1"] > 0)
        out[("fc", st)] = {"X": inter["gathered"].reshape(-1, 2).astype(np.float32),
                           "W": p["fc.weight"].astype(np.float32), "b": p["fc.bias"].astype(np.float32),
                           "C": da1.reshape(-1, da1.shape[-1]).astype(np.float32), "legal": m.reshape(-1),
                           "source": f"derived from reference_real_{st} by the NumPy reference"}
    for (layer, st), o in out.items():
        if not (o["X"].shape[0] == o["C"].shape[0] == o["legal"].shape[0] and o["W"].shape == (o["C"].shape[1],
                                                                                              o["X"].shape[1])):
            raise Void("fixtures", f"{layer} {st}: operand shapes do not fit a linear layer")
        for k in ("X", "W", "b", "C"):
            if not np.all(np.isfinite(o[k])):
                raise Void("non_finite", f"{layer} {st}: operand {k} is not finite")
    return out


def expected(o) -> Dict[str, np.ndarray]:
    """NumPy float64 accumulation of the float32 operands."""
    X, W, b, C = (o[k].astype(np.float64) for k in ("X", "W", "b", "C"))
    return {"Y": X @ W.T + b, "dX": C @ W, "dW": C.T @ X, "db": C.sum(axis=0)}


# ─────────────────────────────── the three implementations ───────────────────────────────

def _inputs(o, layout: str):
    import mlx.core as mx
    if layout == "x_transposed_view":
        Xm = mx.array(np.ascontiguousarray(o["X"].T)).T
    else:
        Xm = mx.array(o["X"])
    return Xm, mx.array(o["W"]), mx.array(o["b"]), mx.array(o["C"])


def compute(impl: str, o, layout: str = "contiguous", context: str = "joint_eval") -> Dict[str, np.ndarray]:
    """One implementation on the current device: Y, dX, dW, db as float32 NumPy (each evaluated first)."""
    import mlx.core as mx
    import mlx.nn as nn

    Xm, Wm, bm, Cm = _inputs(o, layout)
    if impl == "linear_autodiff":
        lin = nn.Linear(o["W"].shape[1], o["W"].shape[0])
        lin.update({"weight": Wm, "bias": bm})

        def f(w, b, x):
            lin.update({"weight": w, "bias": b})
            return lin(x)

        (Y,), (dW, db, dX) = mx.vjp(f, [Wm, bm, Xm], [Cm])
    elif impl == "explicit_autodiff":
        (Y,), (dW, db, dX) = mx.vjp(lambda w, b, x: x @ w.T + b, [Wm, bm, Xm], [Cm])
    elif impl == "explicit_products":
        Y, dX, dW, db = Xm @ Wm.T + bm, Cm @ Wm, Cm.T @ Xm, mx.sum(Cm, axis=0)
    else:
        raise ValueError(impl)
    outs = {"Y": Y, "dX": dX, "dW": dW, "db": db}
    if context == "separate_eval":
        for k in ("Y", "dX", "dW", "db"):
            mx.eval(outs[k])
            mx.synchronize()
    else:
        mx.eval(*outs.values())
    return {k: np.array(v) for k, v in outs.items()}


def describe(o, layout: str) -> Dict[str, Any]:
    return {"X": {"shape": list(o["X"].shape), "dtype": "float32",
                  "layout": "row-major from NumPy" if layout == "contiguous" else
                  "transposed view of a row-major (in x N) array"},
            "W": {"shape": list(o["W"].shape), "dtype": "float32", "layout": "row-major from NumPy"},
            "b": {"shape": list(o["b"].shape), "dtype": "float32"},
            "C": {"shape": list(o["C"].shape), "dtype": "float32", "layout": "row-major from NumPy"},
            "Y_formula": "X @ W.T + b", "rows_legal": int(o["legal"].sum()),
            "rows_padded": int((~o["legal"]).sum())}


# ─────────────────────────────── judging one output ───────────────────────────────

def judge(got: np.ndarray, ref: np.ndarray, legal, spec) -> Dict[str, Any]:
    """Elements outside rel_tol * max|ref| + abs_tol, and GROSS ones beyond gross_rel * max|ref|; for row-wise
    outputs (one row per X row), split into legal and padded rows."""
    if got.shape != ref.shape:
        return {"shape_mismatch": [list(got.shape), list(ref.shape)], "outside_tol": -1, "gross": -1}
    err = np.abs(got.astype(np.float64) - ref)
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    tol, gross = spec["rel_tol"] * scale + spec["abs_tol"], spec["gross_rel"] * scale
    out = {"scale": scale, "max_abs": float(err.max()) if err.size else 0.0, "tol": tol,
           "outside_tol": int((err > tol).sum()), "gross": int((err > gross).sum())}
    if legal is not None and err.ndim == 2 and err.shape[0] == len(legal):
        for name, rows in (("legal", legal), ("padded", ~legal)):
            e = err[rows]
            out[name] = {"rows": int(rows.sum()), "max_abs": float(e.max()) if e.size else 0.0,
                         "outside_tol": int((e > tol).sum()), "gross": int((e > gross).sum()),
                         "rows_with_gross": int((e > gross).any(axis=1).sum())}
    bad = np.argwhere(err > tol)
    out["first_failing"] = [[int(i) for i in ix] + [float(got[tuple(ix)]), float(ref[tuple(ix)])]
                            for ix in bad[:spec["failing_index_cap"]]]
    return out


class _Sink:
    def __init__(self, out_dir: pathlib.Path, save_dir: pathlib.Path):
        self.out_dir, self.save_dir = out_dir, save_dir
        self.handles: Dict[str, Any] = {}
        self.saved: List[Dict[str, Any]] = []

    def line(self, rel, obj):
        fh = self.handles.get(rel) or self.handles.setdefault(rel, open(self.out_dir / rel, "x"))
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def save(self, name, arrays: Mapping[str, np.ndarray], meta) -> Dict[str, Any]:
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        path = self.save_dir / f"{name}.npz"
        tmp = self.save_dir / f".{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        try:
            os.link(tmp, path)
        finally:
            os.unlink(tmp)
        os.chmod(path, 0o444)
        entry = {"name": name, "path": str(path), "sha256": _sha(path.read_bytes()),
                 "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}, **meta}
        self.saved.append(entry)
        self.line("artifacts.jsonl", entry)
        return {"name": name, "sha256": entry["sha256"]}

    def close(self):
        for fh in self.handles.values():
            if not fh.closed:
                fh.close()


def run_case(case: Mapping[str, Any], o, spec, sink) -> Dict[str, Any]:
    """One case: every implementation, the GPU `gpu_repeats` times and the CPU `cpu_repeats` times, judged against
    NumPy; repeats compared bitwise; a failing case's outputs and operands saved."""
    sub = o if case["rows"] is None else {**o, **{k: o[k][:case["rows"]] for k in ("X", "C", "legal")}}
    ref = expected(sub)
    result = {**case, "operands": describe(sub, case["layout"]), "implementations": {}}
    failing = {}
    for impl in spec["implementations"]:
        per = {}
        for dev, n in (("gpu", spec["gpu_repeats"]), ("cpu", spec["cpu_repeats"])):
            runs = []
            with on(dev):
                for _ in range(n):
                    got = compute(impl, sub, case["layout"], case["context"])
                    for k, v in got.items():
                        if not np.all(np.isfinite(v)):
                            raise Void("non_finite", f"{case['id']} {impl} on {dev}: {k} is not finite")
                    runs.append(got)
            per[dev] = {"outputs": {k: [judge(r[k], ref[k], sub["legal"], spec) for r in runs]
                                    for k in spec["outputs"]},
                        "repeats_bitwise": {k: [bool(np.array_equal(r[k], runs[0][k])) for r in runs[1:]]
                                            for k in spec["outputs"]}}
            fails = [k for k in spec["outputs"] if any(j["outside_tol"] for j in per[dev]["outputs"][k])
                     or not all(per[dev]["repeats_bitwise"][k])]
            per[dev]["failing_outputs"] = fails
            per[dev]["gross"] = any(j["gross"] for k in spec["outputs"] for j in per[dev]["outputs"][k]) or \
                not all(all(v) for v in per[dev]["repeats_bitwise"].values())
            for k in fails:
                for i, r in enumerate(runs):
                    failing[f"{impl}.{dev}.{i}.{k}"] = r[k]
        result["implementations"][impl] = per
    result["reproduces"] = any(result["implementations"][i]["gpu"]["gross"] for i in spec["implementations"])
    result["outside_tol_anywhere"] = any(result["implementations"][i][d]["failing_outputs"]
                                         for i in spec["implementations"] for d in ("gpu", "cpu"))
    if failing:
        result["saved"] = sink.save(case["id"], {**{f"operand.{k}": np.asarray(sub[k]) for k in ("X", "W", "b", "C",
                                                                                                  "legal")},
                                                 **{f"expected.{k}": v for k, v in ref.items()}, **failing},
                                    {"case": case["id"], "rel_tol": spec["rel_tol"], "abs_tol": spec["abs_tol"],
                                     "gross_rel": spec["gross_rel"]})
    sink.line("cases.jsonl", result)
    return result


def case_list(spec) -> List[Dict[str, Any]]:
    """The complete case list, frozen: the original shapes first; the targeted cases follow, each run only if its
    (layer, state) reproduces at the original shape."""
    cases = [{"id": f"{L}_{S}_original", "layer": L, "state": S, "rows": None, "layout": "contiguous",
              "context": "joint_eval", "targeted": False} for L in spec["layers"] for S in spec["states"]]
    for L in spec["layers"]:
        for S in spec["states"]:
            cases += [{"id": f"{L}_{S}_rows_{n}", "layer": L, "state": S, "rows": n, "layout": "contiguous",
                       "context": "joint_eval", "targeted": True} for n in spec["row_counts"]]
            cases.append({"id": f"{L}_{S}_layout", "layer": L, "state": S, "rows": None,
                          "layout": spec["alternative_layout"], "context": "joint_eval", "targeted": True})
            cases.append({"id": f"{L}_{S}_context", "layer": L, "state": S, "rows": None, "layout": "contiguous",
                          "context": spec["alternative_context"], "targeted": True})
    return cases


def _execute(spec, seams: Seams, out_dir: pathlib.Path, save_dir: pathlib.Path, rec, log) -> None:
    import mlx.core as mx
    rec["mlx"] = getattr(mx, "__version__", None)
    ops = operands(spec, seams)
    rec["operands"] = {f"{L}_{S}": {"source": o["source"], **describe(o, "contiguous"),
                                    "hashes": {k: _array_sha(np.asarray(o[k])) for k in ("X", "W", "b", "C")}}
                       for (L, S), o in ops.items()}
    sink = _Sink(out_dir, save_dir)
    try:
        for (L, S), o in ops.items():
            if o["source"].startswith("derived"):
                sink.save(f"operands_{L}_{S}", {k: np.asarray(o[k]) for k in ("X", "W", "b", "C", "legal")},
                          {"case": f"operands {L} {S}", "derived_from": o["source"]})
        log("operands", **{k: v["source"] for k, v in rec["operands"].items()})
        cases = case_list(spec)
        rec["case_list"] = [c["id"] for c in cases]
        results: Dict[str, Any] = {}
        for c in cases:
            if c["targeted"] and not results[f"{c['layer']}_{c['state']}_original"]["reproduces"]:
                results[c["id"]] = {"skipped": "its original shape did not reproduce the failure"}
                sink.line("cases.jsonl", {**c, **results[c["id"]]})
                continue
            r = run_case(c, ops[(c["layer"], c["state"])], spec, sink)
            results[c["id"]] = r
            log("case", id=c["id"], reproduces=r["reproduces"], outside_tol=r["outside_tol_anywhere"])
        originals = {k: v["reproduces"] for k, v in results.items() if k.endswith("_original")}
        rec["original_reproduces"] = originals
        rec["isolation_reproduces"] = any(originals.values())
        rec["summary"] = {
            cid: ({"skipped": True} if "skipped" in r else
                  {impl: {dev: {"failing_outputs": r["implementations"][impl][dev]["failing_outputs"],
                                "gross": r["implementations"][impl][dev]["gross"],
                                "max_abs_over_scale": {k: max(j["max_abs"] / (j["scale"] or 1.0)
                                                              for j in r["implementations"][impl][dev]["outputs"][k])
                                                       for k in spec["outputs"]}}
                          for dev in ("gpu", "cpu")} for impl in spec["implementations"]})
            for cid, r in results.items()}
    finally:
        sink.close()
        rec["saved"] = [{k: e[k] for k in ("name", "path", "sha256")} for e in sink.saved]


def conclude(spec, seams: Seams, out_dir: pathlib.Path, save_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    if os.path.lexists(save_dir):
        raise Void("outputs", f"{save_dir} exists")
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "written_by": "script", "diagnostic": True,
                           "network_loaded": False, "save_dir": str(save_dir), "causal_claims": None,
                           "environment": {"python": sys.version, "platform": platform.platform(),
                                           "executable": sys.executable},
                           "loadavg_before": os.getloadavg()}
    log_fh = open(out_dir / "progress.jsonl", "x")

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - t0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    def on_term(signum, _frame):
        raise Terminated(signal.Signals(signum).name)

    code = EXIT_VOID
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main_thread else None
    try:
        log("start")
        _execute(spec, seams, out_dir, save_dir, rec, log)
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
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _real_seams() -> Seams:
    return Seams(read_fixture_index=lambda: (ROOT / FIXTURE_INDEX).read_bytes(),
                 read_fixture=lambda name: (FIXTURE_DIR / f"{name}.npz").read_bytes())


def run() -> int:
    """THE PUBLIC ENTRY. No seam: the frozen spec, the saved fixtures, the pinned card."""
    _require_gate()
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save = OUTPUT_PARENT / f"linrepro-{stamp}"
    if not OUTPUT_PARENT.is_dir() or os.path.lexists(save):
        raise Refused(f"{OUTPUT_PARENT} is not a directory, or {save} exists")
    code, rec = conclude(FROZEN, _real_seams(), out, save, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "head": rec["head"],
                      "original_reproduces": rec.get("original_reproduces"), "saved": len(rec.get("saved", []))}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv != ["run"]:
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_linrepro run", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
