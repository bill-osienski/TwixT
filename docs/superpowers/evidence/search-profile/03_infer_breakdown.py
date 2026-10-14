"""INFER BREAKDOWN -- where inside LocalGPUEvaluator.infer does play-path time go?
(prepared 2026-10-14; the run needs approval)

Production files are unchanged. The instrumented evaluator is LocalGPUEvaluator with `infer` rebuilt from the
production source by inserting five timing marks, each at an anchor that must occur exactly once. Nothing else
changes: the source with the marks removed must equal the production source, so both synchronization calls
stay as they are. The marks split each call into four consecutive segments:

    input     NumPy -> MLX conversion (the four mx.array calls)
    forward   compiled-forward lookup and graph, through the FIRST mx.eval (policy logits, values)
    softmax   masked softmax graph, through the SECOND mx.eval (priors)
    output    tolist / np.array conversion back to NumPy and the array releases, up to the return

These locate wall time on the CPU thread. They cannot separate GPU execution from waiting for it: time spent
inside an mx.eval is both.

The run reuses the frozen setup of 01_profile.py and the checks of 02_sync_compare.py: the same three
fixtures, the pinned incumbent, the test-only seeds and 400-simulation MCTS.search_with_root(state,
add_noise=False). It goes as follows:
* one network load through eval_runner's factory; the instrumented evaluator shares that network;
* three explicit warm-up infers, one per fixture root, before any timed search (their segments are recorded
  apart and are not in the totals);
* one search per fixture: 3 searches, 1,200 simulations.

Every search must EXACTLY match the committed baseline (run/record.json, SHA-256 pinned), as in
02_sync_compare.py. Any mismatch is a STOP. No threshold: this run measures, it does not screen.

  --out DIR   DIR must exist (the launch creates it) and hold no record.json; writes DIR/record.json.

Exit 0 COMPLETED / 2 STOP (behaviour mismatch) / 3 STOP_ERROR / 4 REFUSED.
"""
import difflib
import importlib.util
import inspect
import json
import os
import pathlib
import platform
import sys
import textwrap
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("search_profile_02", HERE / "02_sync_compare.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)                      # definitions only; it loads 01_profile.py as C.P
P = C.P

SEGMENTS = ("input", "forward", "softmax", "output")
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}
MARK = "        _MARK(_NS())\n"
# (anchor, where the mark goes). Mark k closes segment k-1 and opens segment k.
ANCHORS = (("        # Convert numpy to MLX arrays\n", "before"),
           ("        move_mask_mx = mx.array(move_mask)\n", "after"),
           ("        mx.eval(policy_logits, values_mx)\n", "after"),
           ("        mx.eval(priors_mx)\n", "after"),
           ("        return priors_np, values_np\n", "before"))
TICKS = []                                      # perf_counter_ns marks, five per infer call


def instrumented_class(LE):
    """LocalGPUEvaluator with `infer` rebuilt from its own source plus the five marks, compiled against a COPY of
    the module's namespace (the module itself gains no names)."""
    src = inspect.getsource(LE.LocalGPUEvaluator.infer)
    new = src
    for anchor, where in ANCHORS:
        if src.count(anchor) != 1:
            raise P.Refused(f"the production infer does not contain {anchor.strip()!r} exactly once")
        new = new.replace(anchor, MARK + anchor if where == "before" else anchor + MARK)
    if new.count(MARK) != len(ANCHORS) or new.replace(MARK, "") != src:
        raise P.Refused("the instrumented infer is not the production infer plus the marks")
    g = dict(vars(LE), _MARK=TICKS.append, _NS=time.perf_counter_ns)
    ns = {}
    exec(compile(textwrap.dedent(new), "<instrumented infer>", "exec"), g, ns)  # noqa: S102
    diff = "".join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), "production", "instrumented"))
    return type("InstrumentedEvaluator", (LE.LocalGPUEvaluator,), {"infer": ns["infer"]}), diff


def take_calls():
    """Per-call segment durations (ns) from the ticks so far, then clear them."""
    if len(TICKS) % len(ANCHORS):
        raise RuntimeError(f"{len(TICKS)} marks is not a whole number of infer calls")
    calls = [[TICKS[i + k + 1] - TICKS[i + k] for k in range(len(SEGMENTS))]
             for i in range(0, len(TICKS), len(ANCHORS))]
    TICKS.clear()
    return calls


def mark_ns():
    """One mark's own cost, on this process. An ESTIMATE: each segment carries about one."""
    n, f, g = 200_000, TICKS.append, time.perf_counter_ns
    t = g()
    for _ in range(n):
        f(g())
    el = g() - t
    TICKS.clear()
    return el / n


def summary(calls):
    out = {}
    for k, seg in enumerate(SEGMENTS):
        xs = [c[k] for c in calls]
        out[seg] = {"total_ms": sum(xs) / 1e6, "median_ms": P.pct(xs, 0.5) / 1e6, "p95_ms": P.pct(xs, 0.95) / 1e6,
                    "max_ms": max(xs) / 1e6}
    return out


def run(rec, states, sd, base):
    t = time.perf_counter()
    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory
    rec["phases_s"] = {"imports": time.perf_counter() - t}
    cls, rec["instrumentation_diff"] = instrumented_class(LE)
    t = time.perf_counter()
    loaded = _default_evaluator_factory(str(P.CKPT))
    rec["phases_s"]["factory"] = time.perf_counter() - t
    if type(loaded) is not LE.LocalGPUEvaluator or loaded._use_compile is not True:
        raise P.Refused("the factory did not return the production compile=True evaluator")
    ev = cls(loaded.network, compile=True)

    TICKS.clear()
    rec["warmups"] = []
    for n in P.FIXTURES:
        seconds = P.warmup(ev, states[n])
        rec["warmups"].append({"fixture": n, "seconds": seconds,
                               "segments_ms": dict(zip(SEGMENTS, (x / 1e6 for x in take_calls()[0])))})
    rec["searches"], every = [], []
    for n in P.FIXTURES:
        r = C.search(ev, n, "instrumented", states[n], sd[n], base["positions"][n]["baseline"])
        calls = take_calls()
        every += calls
        seg = summary(calls)
        rec["searches"].append({"fixture": n, "matched_baseline": True, **r, "infer_calls": len(calls),
                                "segments": seg,
                                "infer_segments_ms": sum(s["total_ms"] for s in seg.values())})
    seg = summary(every)
    infer_ms = sum(s["total_ms"] for s in seg.values())
    wall = sum(s["wall_ms"] for s in rec["searches"])
    m = mark_ns()
    rec["combined"] = {"simulations": P.SIMS * len(rec["searches"]), "search_wall_ms": wall,
                       "infer_calls": len(every), "infer_segments_ms": infer_ms,
                       "infer_share_of_search_wall": infer_ms / wall, "segments": seg,
                       "share_of_infer": {k: v["total_ms"] / infer_ms for k, v in seg.items()},
                       "share_of_search_wall": {k: v["total_ms"] / wall for k, v in seg.items()},
                       "mark_estimate": {"ns_per_mark": m, "estimated_ms": m * len(ANCHORS) * len(every) / 1e6,
                                         "note": "an estimate from a loop calibration, not a measurement of the run"}}


def main(argv):
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        return EXIT["REFUSED"]
    rec = {"kind": "search-infer-breakdown/1",
           "script": "docs/superpowers/evidence/search-profile/03_infer_breakdown.py",
           "segments": SEGMENTS, "strength_verdict": None,
           "limitation": "CPU-thread wall time; time inside mx.eval is GPU execution and waiting, not separable"}
    t0 = time.perf_counter()
    try:
        states, sd, sha1 = P.preflight()           # fixtures, test-only seeds and checkpoint hash, before MLX
        base = C.load_baseline()
        if base["seeds"]["values"] != sd:
            raise P.Refused("the baseline seeds are not this run's seeds")
        import mlx.core
        import numpy
        rec.update({"head": P.git("rev-parse", "HEAD"), "dirty_paths": P.git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(P.CKPT), "sha1_verified": sha1},
                    "baseline": {"path": str(C.BASELINE.relative_to(P.ROOT)), "sha256": C.BASELINE_SHA256},
                    "seeds": sd, "loadavg_before": os.getloadavg(),
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "executable": sys.executable, "mlx": mlx.core.__version__,
                                    "numpy": numpy.__version__}})
        run(rec, states, sd, base)
        rec["loadavg_after"] = os.getloadavg()
        rec["verdict"], code = "COMPLETED", EXIT["COMPLETED"]
    except P.Refused as e:
        rec["verdict"], rec["error"], code = "REFUSED", str(e), EXIT["REFUSED"]
        print(f"REFUSED: {e}", file=sys.stderr)
    except C.Stop as e:
        rec["verdict"], rec["error"], rec["mismatch"], code = "STOP", str(e), e.details, EXIT["STOP"]
        print(f"STOP: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every failure
        rec["verdict"], rec["error"], code = "STOP_ERROR", traceback.format_exc(), EXIT["STOP_ERROR"]
        print(f"STOP_ERROR: {e!r}", file=sys.stderr)
    rec["total_seconds"] = time.perf_counter() - t0
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if code == 0:
        m = rec["combined"]
        print(f"COMPLETED: {m['simulations']} simulations, all 3 searches matched the baseline; infer "
              f"{m['infer_segments_ms']:.0f} of {m['search_wall_ms']:.0f} ms; "
              + ", ".join(f"{k} {v:.1%}" for k, v in m["share_of_infer"].items()))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
