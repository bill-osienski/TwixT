"""BATCH-NORM MODE, EVENT-LOGGED -- the narrower replacement for 06_bn_mode.py (which aborted with exit 134 and left
no record). (prepared 2026-10-15; the run needs approval)

Same scope as 06 minus every running-statistics read: P1 only, two SEPARATE loads of the pinned incumbent through
eval_runner's factory, two ordinary infer calls on copy A and two fixed-shape (14 x 528, sliced back) calls on
copy B. It reuses 06's mode, padding, output and comparison helpers; none of them reads a running statistic.
Compilation and module modes are left exactly as the factory returns them; nothing calls train() or eval().

Progress goes to DIR/events.jsonl as it happens, one JSON line per event, flushed and fsynced before the next
step, so an abort leaves its location bounded: before/after each load, modes after each load, before/after each
infer (with the outputs after a successful call). DIR/record.json is written at the end with the comparisons
(A1 vs A2, B1 vs B2, A1 vs B1). The running-statistics question is left unresolved by design. No timing verdict.

  --out DIR   DIR must exist (the launch creates it) and hold neither record.json nor events.jsonl.

Exit 0 COMPLETED / 3 STOP_ERROR / 4 REFUSED.
"""
import importlib.util
import json
import os
import pathlib
import platform
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("search_profile_06", HERE / "06_bn_mode.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)                      # definitions only; it loads 01_profile.py as M.P
P = M.P
EXIT = {"COMPLETED": 0, "STOP_ERROR": 3, "REFUSED": 4}


class Events:
    def __init__(self, path):
        self.fh, self.t0, self.n = open(path, "x"), time.perf_counter(), 0

    def __call__(self, event, **data):
        self.fh.write(json.dumps({"seq": self.n, "t_s": round(time.perf_counter() - self.t0, 4), "event": event,
                                  **data}, sort_keys=True) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.n += 1


def run(rec, state, log):
    import numpy as np

    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory

    evs = {}
    for copy in ("A", "B"):
        log("load_begin", copy=copy)
        ev = _default_evaluator_factory(str(P.CKPT))
        log("load_end", copy=copy, evaluator=type(ev).__name__, compile=ev._use_compile)
        if type(ev) is not LE.LocalGPUEvaluator or ev._use_compile is not True:
            raise P.Refused("the factory did not return the production compile=True evaluator")
        evs[copy] = ev
        rec.setdefault("modes_at_load", {})[copy] = M.modes(ev.network)
        log("modes_after_load", copy=copy, **rec["modes_at_load"][copy])
    if evs["A"].network is evs["B"].network:
        raise P.Refused("the two loads returned the same network object")

    legal = state.legal_moves()
    board = np.transpose(evs["A"].build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[r for r, _ in legal]], dtype=np.int32)
    cols = np.array([[c for _, c in legal]], dtype=np.int32)
    mask = np.ones((1, len(legal)), dtype=np.float32)
    rec["input"] = {"fixture": "P1", "boards": 1, "moves": len(legal), "padded_to": [M.PAD_B, M.PAD_M]}
    log("input_built", **rec["input"])

    res, rec["calls"] = {}, []
    for label, copy, kind in (("A1", "A", "ordinary"), ("A2", "A", "ordinary"),
                              ("B1", "B", "fixed_shape"), ("B2", "B", "fixed_shape")):
        ev = evs[copy]
        log("infer_begin", label=label, copy=copy, kind=kind)
        if kind == "ordinary":
            priors, values = ev.infer(board, rows, cols, mask, state.active_size)
        else:
            priors, values = M.padded_infer(ev, board, rows, cols, mask, state.active_size)
        res[label] = (np.asarray(priors), np.asarray(values))
        call = {"label": label, "copy": copy, "kind": kind, **M.output(*res[label], legal),
                "modules_in_training_mode_after": M.modes(ev.network)["modules_in_training_mode"]}
        rec["calls"].append(call)
        log("infer_end", **call)
    rec["comparisons"] = {"A1_vs_A2": M.compare(res["A1"], res["A2"]), "B1_vs_B2": M.compare(res["B1"], res["B2"]),
                          "A1_vs_B1": M.compare(res["A1"], res["B1"])}
    log("comparisons", **rec["comparisons"])


def main(argv):
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists() or (out / "events.jsonl").exists():
        print("REFUSED: --out must name an existing directory without record.json or events.jsonl", file=sys.stderr)
        return EXIT["REFUSED"]
    log = Events(out / "events.jsonl")
    rec = {"kind": "search-bn-mode-events/1", "script": "docs/superpowers/evidence/search-profile/07_bn_mode_events.py",
           "running_statistics": "not read (unresolved by design)", "strength_verdict": None, "timing_verdict": None}
    log("start", script=rec["script"])
    try:
        states, _sd, sha1 = P.preflight()          # fixtures and checkpoint hash, before MLX
        log("preflight_ok", checkpoint_sha1=sha1)
        import mlx.core
        import numpy
        rec.update({"head": P.git("rev-parse", "HEAD"), "dirty_paths": P.git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(P.CKPT), "sha1_verified": sha1},
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "executable": sys.executable, "mlx": mlx.core.__version__,
                                    "numpy": numpy.__version__}, "loadavg_before": os.getloadavg()})
        log("environment", mlx=mlx.core.__version__, numpy=numpy.__version__)
        run(rec, states["P1"], log)
        rec["verdict"], code = "COMPLETED", EXIT["COMPLETED"]
    except P.Refused as e:
        rec["verdict"], rec["error"], code = "REFUSED", str(e), EXIT["REFUSED"]
        print(f"REFUSED: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        rec["verdict"], rec["error"], code = "STOP_ERROR", traceback.format_exc(), EXIT["STOP_ERROR"]
        print(f"STOP_ERROR: {e!r}", file=sys.stderr)
    log("end", verdict=rec["verdict"])
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if code == 0:
        m, c = rec["modes_at_load"]["A"], rec["comparisons"]["A1_vs_B1"]
        print(f"COMPLETED: network training={m['network_training']}, {sum(x['training'] for x in m['batchnorm'])}/"
              f"{len(m['batchnorm'])} BatchNorm in training mode; A1 vs B1 priors identical={c['priors_identical']}, "
              f"max diff {c['priors_max_abs_diff']:.3g}, value diff {c['value_abs_diff']:.3g}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
