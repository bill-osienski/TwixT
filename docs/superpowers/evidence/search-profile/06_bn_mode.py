"""BATCH-NORM MODE DIAGNOSTIC -- in what mode does the production loader leave the network, and does zero-padding
a batch change the real position's outputs? (prepared 2026-10-15; the run needs approval)

Production files are unchanged; nothing here calls train() or eval() on a module. P1 only, no searches:
* two SEPARATE loads of the pinned incumbent through eval_runner's factory (copies A and B);
* record the network's and every BatchNorm layer's training mode, and every BatchNorm's running statistics;
* copy A: two ordinary infer calls on P1's root (1 board x 522 moves, as _expand_batch packs it);
* copy B: two fixed-shape calls on the same input, padded to 14 boards x 528 slots (zero boards; zero rows and
  cols with mask 0) and sliced back to the real row and slots -- the padding 05_fixed_shape.py would use,
  restated here so this record does not depend on that uncommitted file;
* compare the real position's priors and value (A1 vs A2, B1 vs B2, A1 vs B1) and record whether the running
  statistics change after each call.
Four infer calls in all. No timing verdict, no STOP condition: it records what it finds.

  --out DIR   DIR must exist (the launch creates it) and hold no record.json; writes DIR/record.json.

Exit 0 COMPLETED / 3 STOP_ERROR / 4 REFUSED.
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("search_profile_01", HERE / "01_profile.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)                      # definitions only; it puts the repository root on sys.path

PAD_B, PAD_M = 14, 528
EXIT = {"COMPLETED": 0, "STOP_ERROR": 3, "REFUSED": 4}


def modes(net):
    import mlx.nn as nn

    mods = net.named_modules()
    bn = [(name, m) for name, m in mods if isinstance(m, nn.BatchNorm)]
    return {"network_training": bool(net.training), "modules": len(mods),
            "modules_in_training_mode": sum(bool(m.training) for _, m in mods),
            "dropout_layers": [name for name, m in mods if "Dropout" in type(m).__name__],
            "batchnorm": [{"name": name, "training": bool(m.training), "track_running_stats": m.track_running_stats}
                          for name, m in bn]}


def stats(net):
    """Every BatchNorm's running mean and variance as NumPy, or the error reading them."""
    import mlx.nn as nn
    import numpy as np

    out = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm) and m.track_running_stats:
            try:
                out[name] = (np.array(m.running_mean), np.array(m.running_var))
            except Exception as e:  # noqa: BLE001 -- recorded, not hidden
                out[name] = repr(e)
    return out


def stats_digest(s):
    h = hashlib.sha256()
    for name in sorted(s):
        h.update(name.encode())
        v = s[name]
        h.update(v.encode() if isinstance(v, str) else v[0].tobytes() + v[1].tobytes())
    return h.hexdigest()


def stats_change(before, after):
    import numpy as np

    errors = sorted(k for k in after if isinstance(after[k], str) or isinstance(before.get(k), str))
    deltas = [float(max(np.abs(after[k][0] - before[k][0]).max(), np.abs(after[k][1] - before[k][1]).max()))
              for k in after if k not in errors]
    return {"digest_after": stats_digest(after), "changed": stats_digest(after) != stats_digest(before),
            "layers_changed": sum(d > 0 for d in deltas), "max_abs_delta": max(deltas, default=0.0),
            "read_errors": {k: after[k] if isinstance(after[k], str) else before[k] for k in errors}}


def padded_infer(ev, board, rows, cols, mask, active_size):
    import numpy as np

    b, m = board.shape[0], rows.shape[1]
    pb = np.zeros((PAD_B,) + board.shape[1:], board.dtype)
    pr, pc = np.zeros((PAD_B, PAD_M), rows.dtype), np.zeros((PAD_B, PAD_M), cols.dtype)
    pm = np.zeros((PAD_B, PAD_M), mask.dtype)
    pb[:b], pr[:b, :m], pc[:b, :m], pm[:b, :m] = board, rows, cols, mask
    priors, values = ev.infer(pb, pr, pc, pm, active_size)
    return priors[:b, :m], values[:b]


def output(priors, values, legal):
    p = priors[0]
    top = sorted(range(len(p)), key=lambda j: (-p[j], legal[j]))[:5]
    return {"value": repr(float(values[0])), "prior_sum": float(p.sum()),
            "top5": [[list(legal[j]), float(p[j])] for j in top],
            "priors_sha256": hashlib.sha256(p.tobytes()).hexdigest()}


def compare(x, y):
    import numpy as np

    return {"priors_identical": bool(np.array_equal(x[0], y[0])), "value_identical": bool(x[1][0] == y[1][0]),
            "priors_max_abs_diff": float(np.abs(x[0] - y[0]).max()),
            "priors_total_variation": float(0.5 * np.abs(x[0] - y[0]).sum()),
            "value_abs_diff": float(abs(x[1][0] - y[1][0])),
            "same_top_prior_move": int(np.argmax(x[0])) == int(np.argmax(y[0]))}


def run(rec, state):
    import numpy as np

    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory

    a = _default_evaluator_factory(str(P.CKPT))
    b = _default_evaluator_factory(str(P.CKPT))
    for ev in (a, b):
        if type(ev) is not LE.LocalGPUEvaluator or ev._use_compile is not True:
            raise P.Refused("the factory did not return the production compile=True evaluator")
    if a.network is b.network:
        raise P.Refused("the two loads returned the same network object")
    rec["modes_at_load"] = {"A": modes(a.network), "B": modes(b.network)}
    s0 = {"A": stats(a.network), "B": stats(b.network)}
    rec["running_stats_at_load"] = {"A": stats_digest(s0["A"]), "B": stats_digest(s0["B"]),
                                    "copies_identical": stats_digest(s0["A"]) == stats_digest(s0["B"])}

    legal = state.legal_moves()
    board = np.transpose(a.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[r for r, _ in legal]], dtype=np.int32)
    cols = np.array([[c for _, c in legal]], dtype=np.int32)
    mask = np.ones((1, len(legal)), dtype=np.float32)
    rec["input"] = {"fixture": "P1", "boards": 1, "moves": len(legal), "padded_to": [PAD_B, PAD_M]}

    res, rec["calls"] = {}, []
    for copy, ev, kind in (("A", a, "ordinary"), ("A", a, "ordinary"), ("B", b, "fixed_shape"), ("B", b, "fixed_shape")):
        before = stats(ev.network)
        if kind == "ordinary":
            priors, values = ev.infer(board, rows, cols, mask, state.active_size)
        else:
            priors, values = padded_infer(ev, board, rows, cols, mask, state.active_size)
        label = f"{copy}{sum(c['copy'] == copy for c in rec['calls']) + 1}"
        res[label] = (np.asarray(priors), np.asarray(values))
        rec["calls"].append({"label": label, "copy": copy, "kind": kind, **output(*res[label], legal),
                             "running_stats_change": stats_change(before, stats(ev.network)),
                             "modes_after": modes(ev.network)["modules_in_training_mode"]})
    rec["comparisons"] = {"A1_vs_A2": compare(res["A1"], res["A2"]), "B1_vs_B2": compare(res["B1"], res["B2"]),
                          "A1_vs_B1": compare(res["A1"], res["B1"])}


def main(argv):
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        return EXIT["REFUSED"]
    rec = {"kind": "search-bn-mode/1", "script": "docs/superpowers/evidence/search-profile/06_bn_mode.py",
           "strength_verdict": None, "timing_verdict": None}
    try:
        states, _sd, sha1 = P.preflight()          # fixtures and checkpoint hash, before MLX
        import mlx.core
        import numpy
        rec.update({"head": P.git("rev-parse", "HEAD"), "dirty_paths": P.git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(P.CKPT), "sha1_verified": sha1},
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "executable": sys.executable, "mlx": mlx.core.__version__,
                                    "numpy": numpy.__version__}, "loadavg_before": os.getloadavg()})
        run(rec, states["P1"])
        rec["verdict"], code = "COMPLETED", EXIT["COMPLETED"]
    except P.Refused as e:
        rec["verdict"], rec["error"], code = "REFUSED", str(e), EXIT["REFUSED"]
        print(f"REFUSED: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every failure
        rec["verdict"], rec["error"], code = "STOP_ERROR", traceback.format_exc(), EXIT["STOP_ERROR"]
        print(f"STOP_ERROR: {e!r}", file=sys.stderr)
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
