"""SYNC COMPARISON -- does ONE GPU sync in infer, in place of two, make play-path search faster?
(prepared 2026-10-14; the run needs approval)

Production files are unchanged. The prototype is LocalGPUEvaluator with `infer` rebuilt from the production
source by ONE exact text replacement (refused if the production text is not found exactly once):

    mx.eval(policy_logits, values_mx)                                     priors_mx = softmax(logits, mask)
    priors_mx = softmax(logits, mask)                    becomes          mx.eval(priors_mx, values_mx)
    mx.eval(priors_mx)

The network, the softmax mathematics, the precision and the search are unchanged. The record holds the diff.

The run reuses the frozen setup of 01_profile.py: the same three fixtures, the pinned incumbent, the test-only
seeds and 400-simulation MCTS.search_with_root(state, add_noise=False). It goes as follows:
* one network load through eval_runner's factory (the control evaluator); the prototype evaluator shares
  that network;
* six fixed warm-up infers, before any timed search: per fixture, the control and then the prototype, each
  on the fixture's root;
* per fixture: control, prototype, prototype, control. That is 12 searches, 4,800 simulations.

Every search must EXACTLY match the committed baseline (run/record.json, SHA-256 pinned): the full legal
visit counts and their digest, the root value repr, the most-visited move and the RNG-state digest. Any
mismatch is a STOP. Screening threshold, fixed in advance: the prototype's combined search time is at least
10% below the control's. That is a practical screen, not statistical proof.

  --out DIR   DIR must exist (the launch creates it) and hold no record.json; writes DIR/record.json.

Exit 0 COMPLETED (whatever the screen) / 2 STOP (behaviour mismatch) / 3 STOP_ERROR / 4 REFUSED.
"""
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import platform
import random
import sys
import textwrap
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("search_profile_01", HERE / "01_profile.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)                      # definitions only; it puts the repository root on sys.path

from scripts.GPU.alphazero import search_distillation_stage_b as B  # noqa: E402
from scripts.GPU.alphazero.eval_runner import EvalConfig, cfg_from  # noqa: E402
from scripts.GPU.alphazero.mcts import MCTS  # noqa: E402

BASELINE = HERE / "run/record.json"
BASELINE_SHA256 = "0fd099ed2e929847f44ce0b5c312316eea2e6e8a9cd4dd6eabe976627e624e57"
ORDER = ("control", "prototype", "prototype", "control")
THRESHOLD = 0.10
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}

OLD = ("        # Sync GPU (known stable at eval_batch <= 14)\n"
       "        mx.eval(policy_logits, values_mx)\n"
       "\n"
       "        # Stable softmax with mask and renormalization\n"
       "        priors_mx = self._masked_softmax_batch(policy_logits, move_mask_mx)\n"
       "        mx.eval(priors_mx)\n")
NEW = ("        # Stable softmax with mask and renormalization\n"
       "        priors_mx = self._masked_softmax_batch(policy_logits, move_mask_mx)\n"
       "        mx.eval(priors_mx, values_mx)  # PROTOTYPE: one sync in place of two\n")


class Stop(Exception):
    """A behaviour mismatch. `details` (fixture, arm, differing fields, computed and baseline values) goes into
    the record, so a mismatch can be diagnosed without another run."""

    def __init__(self, msg, details=None):
        super().__init__(msg)
        self.details = details


def prototype_class(LE):
    """LocalGPUEvaluator with `infer` rebuilt from its own source, OLD -> NEW, compiled in the module's namespace."""
    src = inspect.getsource(LE.LocalGPUEvaluator.infer)
    if src.count(OLD) != 1:
        raise P.Refused("the production infer does not contain the two-sync block exactly once")
    new = src.replace(OLD, NEW)
    ns = {}
    exec(compile(textwrap.dedent(new), "<prototype infer>", "exec"), vars(LE), ns)  # noqa: S102
    diff = "".join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), "production", "prototype"))
    return type("PrototypeEvaluator", (LE.LocalGPUEvaluator,), {"infer": ns["infer"]}), diff


def load_baseline():
    raw = BASELINE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
        raise P.Refused("run/record.json is not the pinned baseline")
    base = json.loads(raw)
    if base["verdict"] != "COMPLETED" or base["fixtures"] != P.FIXTURES or base["sims_per_position"] != P.SIMS:
        raise P.Refused("the baseline is not a COMPLETED profile of these fixtures at this budget")
    return base


def search(ev, name, arm, state, seed, want):
    mcts = MCTS(ev, cfg_from(EvalConfig(**B.PROTOCOL), P.SIMS), random.Random(seed))
    w0, c0 = time.perf_counter_ns(), time.process_time_ns()
    counts, value, root = mcts.search_with_root(state, add_noise=False)
    wall, cpu = time.perf_counter_ns() - w0, time.process_time_ns() - c0
    legal = state.legal_moves()
    if root.visit_count != P.SIMS or sum(counts.values()) != P.SIMS or sorted(counts) != sorted(legal):
        raise Stop(f"{name}/{arm}: not {P.SIMS} completed simulations over the full legal list",
                   {"fixture": name, "arm": arm, "fields": ["simulations"], "root_visits": root.visit_count,
                    "counted_visits": sum(counts.values()), "visit_keys_equal_legal": sorted(counts) == sorted(legal)})
    full = [[r, c, counts[(r, c)]] for r, c in sorted(legal)]
    top = max(counts.values())
    got = {"visit_counts": full, "visit_counts_sha256": B._sha(full), "root_value": repr(value),
           "most_visited": sorted([r, c] for (r, c), v in counts.items() if v == top)[0],
           "rng_state_sha256": hashlib.sha256(repr(mcts.rng.getstate()).encode()).hexdigest()}
    bad = [k for k in got if got[k] != want[k]]
    if bad:
        raise Stop(f"{name}/{arm}: behaviour differs from the baseline in {bad}",
                   {"fixture": name, "arm": arm, "fields": bad, "computed": {k: got[k] for k in bad},
                    "baseline": {k: want[k] for k in bad}})
    return {"wall_ms": wall / 1e6, "process_cpu_ms": cpu / 1e6}


def run(rec, states, sd, base):
    t = time.perf_counter()
    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory
    rec["phases_s"] = {"imports": time.perf_counter() - t}
    proto_cls, rec["prototype_diff"] = prototype_class(LE)
    t = time.perf_counter()
    control = _default_evaluator_factory(str(P.CKPT))
    rec["phases_s"]["factory"] = time.perf_counter() - t
    if type(control) is not LE.LocalGPUEvaluator or control._use_compile is not True:
        raise P.Refused("the factory did not return the production compile=True evaluator")
    arms = {"control": control, "prototype": proto_cls(control.network, compile=True)}

    rec["warmups_s"] = [{"fixture": n, "arm": a, "seconds": P.warmup(arms[a], states[n])}
                        for n in P.FIXTURES for a in ("control", "prototype")]
    rec["searches"] = []
    for n in P.FIXTURES:
        for a in ORDER:
            r = search(arms[a], n, a, states[n], sd[n], base["positions"][n]["baseline"])
            rec["searches"].append({"index": len(rec["searches"]), "fixture": n, "arm": a, "matched_baseline": True, **r})

    def total(arm, fixture=None):
        return sum(s["wall_ms"] for s in rec["searches"] if s["arm"] == arm and fixture in (None, s["fixture"]))

    c, p = total("control"), total("prototype")
    rec["combined"] = {"control_ms": c, "prototype_ms": p, "reduction": 1 - p / c, "threshold": THRESHOLD,
                       "simulations": P.SIMS * len(rec["searches"]),
                       "per_fixture_reduction": {n: 1 - total("prototype", n) / total("control", n)
                                                 for n in P.FIXTURES}}
    rec["screen"] = "SCREEN_PASS" if 1 - p / c >= THRESHOLD else "SCREEN_NON_PASS"


def main(argv):
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        return EXIT["REFUSED"]
    rec = {"kind": "search-sync-compare/1", "script": "docs/superpowers/evidence/search-profile/02_sync_compare.py",
           "order_per_fixture": ORDER, "strength_verdict": None}
    t0 = time.perf_counter()
    try:
        states, sd, sha1 = P.preflight()           # fixtures, test-only seeds and checkpoint hash, before MLX
        base = load_baseline()
        if base["seeds"]["values"] != sd:
            raise P.Refused("the baseline seeds are not this run's seeds")
        import mlx.core
        import numpy
        rec.update({"head": P.git("rev-parse", "HEAD"), "dirty_paths": P.git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(P.CKPT), "sha1_verified": sha1},
                    "baseline": {"path": str(BASELINE.relative_to(P.ROOT)), "sha256": BASELINE_SHA256},
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
    except Stop as e:
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
        print(f"COMPLETED: {m['simulations']} simulations, all 12 searches matched the baseline; control "
              f"{m['control_ms']:.0f} ms, prototype {m['prototype_ms']:.0f} ms, reduction {m['reduction']:.1%} "
              f"-> {rec['screen']}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
