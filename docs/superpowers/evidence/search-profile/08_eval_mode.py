"""EVAL-MODE CHECK -- with the factory calling net.eval(), is P1's prediction independent of batch padding?
(prepared 2026-10-15; the run needs approval)

One load of the pinned incumbent through eval_runner's factory, which now puts the network in evaluation mode
before the evaluator is built. This script never calls train() or eval() itself and never reads a running
statistic. Four infer calls on P1's root, all on that one evaluator: two ordinary (1 board x 522 moves) and two
fixed-shape (padded to 14 x 528 and sliced back), using 06's padding, output and comparison helpers. Progress goes
to DIR/events.jsonl, flushed and fsynced per event, as in 07.

Checks, frozen in advance (verdict PASS only if all hold):
* after the load: the network and every BatchNorm layer report training == False;
* repeats: A1 vs A2 and B1 vs B2 bit-identical (priors and value);
* ordinary vs padded (A1 vs B1): max |prior difference| <= 1e-5, |value difference| <= 1e-5, same top prior move.
The training-mode values from bn-mode-events/record.json are copied in for description only.
No searches, games or timing verdict.

  --out DIR   DIR must exist (the launch creates it) and hold neither record.json nor events.jsonl.

Exit 0 PASS / 2 FAIL / 3 STOP_ERROR / 4 REFUSED.
"""
import importlib.util
import json
import os
import pathlib
import platform
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("search_profile_07", HERE / "07_bn_mode_events.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)                      # definitions only; it loads 06 as E.M and 01 as E.P
M, P = E.M, E.P

TOLERANCE = 1e-5
TRAINING_MODE = HERE / "bn-mode-events/record.json"
EXIT = {"PASS": 0, "FAIL": 2, "STOP_ERROR": 3, "REFUSED": 4}


def run(rec, state, log):
    import numpy as np

    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory

    log("load_begin")
    ev = _default_evaluator_factory(str(P.CKPT))
    log("load_end", evaluator=type(ev).__name__, compile=ev._use_compile)
    if type(ev) is not LE.LocalGPUEvaluator or ev._use_compile is not True:
        raise P.Refused("the factory did not return the production compile=True evaluator")
    rec["modes_at_load"] = M.modes(ev.network)
    log("modes_after_load", **rec["modes_at_load"])

    legal = state.legal_moves()
    board = np.transpose(ev.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[r for r, _ in legal]], dtype=np.int32)
    cols = np.array([[c for _, c in legal]], dtype=np.int32)
    mask = np.ones((1, len(legal)), dtype=np.float32)
    rec["input"] = {"fixture": "P1", "boards": 1, "moves": len(legal), "padded_to": [M.PAD_B, M.PAD_M]}
    log("input_built", **rec["input"])

    res, rec["calls"] = {}, []
    for label, kind in (("A1", "ordinary"), ("A2", "ordinary"), ("B1", "fixed_shape"), ("B2", "fixed_shape")):
        log("infer_begin", label=label, kind=kind)
        if kind == "ordinary":
            priors, values = ev.infer(board, rows, cols, mask, state.active_size)
        else:
            priors, values = M.padded_infer(ev, board, rows, cols, mask, state.active_size)
        res[label] = (np.asarray(priors), np.asarray(values))
        call = {"label": label, "kind": kind, **M.output(*res[label], legal),
                "modules_in_training_mode_after": M.modes(ev.network)["modules_in_training_mode"]}
        rec["calls"].append(call)
        log("infer_end", **call)
    cmp = {"A1_vs_A2": M.compare(res["A1"], res["A2"]), "B1_vs_B2": M.compare(res["B1"], res["B2"]),
           "A1_vs_B1": M.compare(res["A1"], res["B1"])}
    rec["comparisons"] = cmp
    log("comparisons", **cmp)

    m = rec["modes_at_load"]
    checks = {"network_eval_mode": m["network_training"] is False,
              "all_batchnorm_eval_mode": bool(m["batchnorm"]) and not any(x["training"] for x in m["batchnorm"]),
              "no_module_training_after_calls": all(c["modules_in_training_mode_after"] == 0 for c in rec["calls"]),
              "A_repeat_identical": cmp["A1_vs_A2"]["priors_identical"] and cmp["A1_vs_A2"]["value_identical"],
              "B_repeat_identical": cmp["B1_vs_B2"]["priors_identical"] and cmp["B1_vs_B2"]["value_identical"],
              "padded_priors_within_tolerance": cmp["A1_vs_B1"]["priors_max_abs_diff"] <= TOLERANCE,
              "padded_value_within_tolerance": cmp["A1_vs_B1"]["value_abs_diff"] <= TOLERANCE,
              "padded_same_top_move": cmp["A1_vs_B1"]["same_top_prior_move"]}
    rec["checks"] = checks
    log("checks", **checks)
    return all(checks.values())


def main(argv):
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists() or (out / "events.jsonl").exists():
        print("REFUSED: --out must name an existing directory without record.json or events.jsonl", file=sys.stderr)
        return EXIT["REFUSED"]
    log = E.Events(out / "events.jsonl")
    rec = {"kind": "search-eval-mode/1", "script": "docs/superpowers/evidence/search-profile/08_eval_mode.py",
           "tolerance": {"priors_max_abs": TOLERANCE, "value_abs": TOLERANCE, "repeats": "bit-identical"},
           "running_statistics": "not read", "strength_verdict": None, "timing_verdict": None}
    log("start", script=rec["script"])
    try:
        states, _sd, sha1 = P.preflight()          # fixtures and checkpoint hash, before MLX
        tm = json.loads(TRAINING_MODE.read_text())
        rec["training_mode_reference"] = {"path": str(TRAINING_MODE.relative_to(P.ROOT)),
                                          "calls": {c["label"]: {"value": c["value"], "top5": c["top5"]}
                                                    for c in tm["calls"]}}
        log("preflight_ok", checkpoint_sha1=sha1)
        import mlx.core
        import numpy
        rec.update({"head": P.git("rev-parse", "HEAD"), "dirty_paths": P.git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(P.CKPT), "sha1_verified": sha1},
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "executable": sys.executable, "mlx": mlx.core.__version__,
                                    "numpy": numpy.__version__}, "loadavg_before": os.getloadavg()})
        log("environment", mlx=mlx.core.__version__, numpy=numpy.__version__)
        rec["verdict"] = "PASS" if run(rec, states["P1"], log) else "FAIL"
        code = EXIT[rec["verdict"]]
    except P.Refused as e:
        rec["verdict"], rec["error"], code = "REFUSED", str(e), EXIT["REFUSED"]
        print(f"REFUSED: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        rec["verdict"], rec["error"], code = "STOP_ERROR", traceback.format_exc(), EXIT["STOP_ERROR"]
        print(f"STOP_ERROR: {e!r}", file=sys.stderr)
    log("end", verdict=rec["verdict"])
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if code in (0, 2):
        c = rec["comparisons"]["A1_vs_B1"]
        print(f"{rec['verdict']}: failed checks {[k for k, v in rec['checks'].items() if not v]}; A1 vs B1 max prior "
              f"diff {c['priors_max_abs_diff']:.3g}, value diff {c['value_abs_diff']:.3g}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
