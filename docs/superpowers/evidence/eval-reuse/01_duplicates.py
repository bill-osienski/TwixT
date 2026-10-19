"""EVALUATION-REUSE MEASUREMENT -- do consecutive searches repeat exact neural-evaluator inputs? Read-only counting:
every inference runs normally; nothing is cached or skipped. No game, training or parameter change.

Positions: game 0 of the self-play diagnostic corpus after 20, 21 and 22 moves (red, black, red to move), applied
to a fresh 24x24 state (280-ply cap). One load of calib020_0001 through eval_runner's factory (evaluation mode
checked, SHA-1 pinned); one ordinary 400-simulation search per position through search_with_root with cfg_from(Stage
B PROTOCOL) and no root noise -- 1,200 simulations. Each side has its own MCTS as in eval_runner.play_eval_game
(masks 0xA5A5A5 red / 0x5A5A5A black) over one shared evaluator; RNG seed TEST_ONLY_SEED_INTERVALS[0][0] (a
diagnostic search, not a game).
Every evaluator.infer call is keyed by the sha256 of its exact inputs -- the board tensor, move-row and move-column
arrays, mask (bytes, shapes and dtypes) and active size -- and timed. A DUPLICATE is a call whose key was seen
before, in the same search or an earlier one (reported apart). Search time = time inside search_with_root minus
the instrumentation's own hashing time.
FROZEN DECISION: saving = inference seconds spent on duplicates / search seconds. saving < 10% -> STOP (this idea
on this sample); >= 10% -> justifies only a small prototype. Optimistic: cache overhead is excluded.

  --checkpoint PATH (absolute)   --out DIR (exists, without record.json)
Exit 0 COMPLETED / 2 STOP / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 50` (60 s in all), in a clean tree.
"""
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY = ROOT / "docs/superpowers/evidence/self-play-diagnostic/games/game_000000.json"
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
CUTS, BAR = (20, 21, 22), 0.10
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def key(boards, rows, cols, mask, active_size):
    h = hashlib.sha256()
    for a in (boards, rows, cols, mask):
        h.update(f"{a.shape}{a.dtype}".encode())
        h.update(a.tobytes())
    h.update(str(int(active_size)).encode())
    return h.hexdigest()


def main(rec, argv):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS
    from scripts.GPU.alphazero.search_distillation_stage_b import PROTOCOL

    moves = [(m["row"], m["col"]) for m in json.loads(REPLAY.read_text())["moves"]]
    states, s = {}, TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for i, m in enumerate(moves[:max(CUTS)]):
        s = s.apply_move(m)
        if i + 1 in CUTS:
            states[i + 1] = s
    if any(states[c].is_terminal() for c in CUTS) or [states[c].to_move for c in CUTS] != ["red", "black", "red"]:
        raise Refused("the three positions are not live red/black/red positions")
    seed = REF.TEST_ONLY_SEED_INTERVALS[0][0]
    if not REF.seed_is_test_only(seed):
        raise Refused("the diagnostic seed is not test-only")
    if hashlib.sha1(ckpt.read_bytes()).hexdigest() != SHA1:
        raise Refused("the checkpoint is not the pinned calib020_0001")
    ev = ER._default_evaluator_factory(str(ckpt))
    rec["modules_in_training_mode"] = sum(bool(x.training) for _, x in ev.network.named_modules())
    if ev.network.training or rec["modules_in_training_mode"]:
        raise Stop("the network is not in evaluation mode")
    cfg = ER.cfg_from(ER.EvalConfig(**PROTOCOL))
    if (cfg.n_simulations, cfg.c_puct, cfg.fpu_value) != (400, 1.5, 0.0):
        raise Refused("the search settings are not 400 sims, c_puct 1.5, absolute FPU 0.0")

    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if (n == "__main__" or n.split(".")[0] == "scripts") and getattr(m, "__file__", None)}
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Refused("a loaded repository module is uncommitted or differs from HEAD")
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}

    seen, calls, cur = {}, [], {"search": None, "hash_s": 0.0}
    real = ev.infer

    def infer(boards, rows, cols, mask, active_size):
        t = time.perf_counter()
        k = key(boards, rows, cols, mask, active_size)
        cur["hash_s"] += time.perf_counter() - t
        t = time.perf_counter()
        out = real(boards, rows, cols, mask, active_size)
        dt = time.perf_counter() - t
        dup = k in seen                       # seen BEFORE this call, in this search or an earlier one
        first = seen.setdefault(k, cur["search"])
        calls.append({"search": cur["search"], "batch": int(boards.shape[0]), "seconds": dt,
                      "duplicate": dup, "first_search": first})
        return out

    ev.infer = infer
    mcts = {"red": MCTS(ev, cfg, random.Random(seed ^ 0xA5A5A5)), "black": MCTS(ev, cfg, random.Random(seed ^ 0x5A5A5A))}
    rec["searches"] = []
    for c in CUTS:
        st = states[c]
        cur["search"], h0 = c, cur["hash_s"]
        t = time.perf_counter()
        counts, value, root = mcts[st.to_move].search_with_root(st, add_noise=False)
        wall = time.perf_counter() - t
        if root.visit_count != cfg.n_simulations:
            raise Stop(f"position {c}: root visits {root.visit_count}, not {cfg.n_simulations}")
        mine = [x for x in calls if x["search"] == c]
        rec["searches"].append({"moves_applied": c, "to_move": st.to_move, "root_visits": root.visit_count,
                                "root_value": value, "infer_calls": len(mine),
                                "positions": sum(x["batch"] for x in mine),
                                "infer_seconds": sum(x["seconds"] for x in mine),
                                "hash_seconds": cur["hash_s"] - h0,
                                "search_seconds": wall - (cur["hash_s"] - h0)})
    dup = [x for x in calls if x["duplicate"]]
    within = [x for x in dup if x["first_search"] == x["search"]]
    search_s = sum(x["search_seconds"] for x in rec["searches"])
    rec["totals"] = {"infer_calls": len(calls), "distinct_inputs": len(seen), "duplicates": len(dup),
                     "duplicates_within_search": len(within), "duplicates_from_earlier_search": len(dup) - len(within),
                     "infer_seconds": sum(x["seconds"] for x in calls),
                     "duplicate_infer_seconds": sum(x["seconds"] for x in dup),
                     "search_seconds": search_s, "hash_seconds": cur["hash_s"]}
    t = rec["totals"]
    if t["infer_calls"] - t["duplicates"] != t["distinct_inputs"]:
        raise Stop("duplicate accounting does not reconcile with the distinct-input count")
    t["saving_fraction_upper_bound"] = t["duplicate_infer_seconds"] / search_s
    rec["answer"] = "STOP" if t["saving_fraction_upper_bound"] < BAR else "JUSTIFIES_SMALL_PROTOTYPE"
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "eval-reuse-duplicates/1", "script": "docs/superpowers/evidence/eval-reuse/01_duplicates.py",
           "head": git("rev-parse", "HEAD"), "bar": BAR}
    t0 = time.monotonic()
    try:
        verdict = main(rec, sys.argv)
    except Refused as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Stop as e:
        verdict, rec["reason"] = "STOP", str(e)
    except Exception:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        verdict, rec["reason"] = "STOP_ERROR", traceback.format_exc()
    rec["verdict"], rec["seconds"] = verdict, round(time.monotonic() - t0, 3)
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if verdict == "COMPLETED":
        t = rec["totals"]
        print(f"COMPLETED: {rec['answer']}; {t['duplicates']} duplicates of {t['infer_calls']} calls "
              f"({t['duplicates_within_search']} within a search, {t['duplicates_from_earlier_search']} from an "
              f"earlier one); duplicate inference {t['duplicate_infer_seconds']:.3f} s of {t['search_seconds']:.3f} s "
              f"search = {100 * t['saving_fraction_upper_bound']:.2f}% (bar 10%); {rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
