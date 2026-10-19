"""EVALUATION-REUSE CACHE PROTOTYPE -- one bounded timing screen. The cache exists only in this script; production is
unchanged. No game, training or parameter change.

Same positions, checkpoint and settings as 01_duplicates.py: game 0 of the self-play diagnostic corpus after 20, 21
and 22 moves (red, black, red), calib020_0001 through eval_runner's factory (evaluation mode checked, SHA-1
pinned), cfg_from(Stage B PROTOCOL), 400 simulations, no root noise, each colour its own MCTS over one shared
network evaluator (masks 0xA5A5A5 red / 0x5A5A5A black, seed TEST_ONLY_SEED_INTERVALS[0][0]).
A SEQUENCE = the three searches in order, with fresh MCTS objects and RNG streams (and, if cached, fresh caches).
Order: ORDINARY -> CACHED -> CACHED -> ORDINARY (4,800 search simulations). Before the first sequence, an identical
warm-up shared by both paths: one plain inference of each of the three root positions (no search).
CACHED: each colour's MCTS gets its own CachedEvaluator, retained across that colour's searches in the sequence,
never shared with the other colour. Key = sha256 of the exact inputs (board tensor, move rows, move columns, mask --
bytes, shapes, dtypes -- and active size), as in 01. A miss runs the network and stores copies; a hit returns copies.
CORRECTNESS (any mismatch STOPs at once): every search of every sequence equals the first ORDINARY sequence's in
every move's visit count, the root value (exact), the most-visited move (ties broken by (row, col)) and, after the
sequence, both colours' final search-RNG states.
TIMING: wall time of each whole search_with_root call, hashing, lookup, storage and copying included. Reported: per
search and per sequence, cache hits and misses by colour, entries and stored bytes (array nbytes + key bytes).
FROZEN SCREEN: reduction = 1 - (cached sequences' combined search seconds) / (ordinary sequences' combined search
seconds). reduction >= 10% with every check passing -> QUALIFIES for further consideration; less ->
STOP_CONFIGURATION. A small timing screen, not deployment or strength evidence.

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
ORDER = ("ordinary", "cached", "cached", "ordinary")
MASK = {"red": 0xA5A5A5, "black": 0x5A5A5A}
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def key(boards, rows, cols, mask, active_size):
    """01_duplicates.py's key, unchanged."""
    h = hashlib.sha256()
    for a in (boards, rows, cols, mask):
        h.update(f"{a.shape}{a.dtype}".encode())
        h.update(a.tobytes())
    h.update(str(int(active_size)).encode())
    return h.hexdigest()


class CachedEvaluator:
    """Exact-input prediction cache over one evaluator, for ONE colour."""

    def __init__(self, base):
        self.base, self.network, self.store = base, base.network, {}
        self.hits = self.misses = 0

    def build_input_tensor(self, state):
        return self.base.build_input_tensor(state)

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        k = key(boards, move_rows, move_cols, move_mask, active_size)
        hit = self.store.get(k)
        if hit is not None:
            self.hits += 1
            return hit[0].copy(), hit[1].copy()
        self.misses += 1
        priors, values = self.base.infer(boards, move_rows, move_cols, move_mask, active_size)
        self.store[k] = (priors.copy(), values.copy())
        return priors, values

    def stored_bytes(self):
        return sum(p.nbytes + v.nbytes + len(k) for k, (p, v) in self.store.items())


def main(rec, argv):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    sys.path.insert(0, str(ROOT))
    import numpy as np
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

    for c in CUTS:                                           # the shared warm-up: one plain inference per root
        st = states[c]
        lm = st.legal_moves()
        ev.infer(np.transpose(ev.build_input_tensor(st), (1, 2, 0))[None].astype(np.float32),
                 np.array([[r for r, _ in lm]], dtype=np.int32), np.array([[q for _, q in lm]], dtype=np.int32),
                 np.ones((1, len(lm)), dtype=np.float32), st.active_size)

    reference, rec["sequences"] = None, []
    for n, path in enumerate(ORDER):
        evs = {col: CachedEvaluator(ev) if path == "cached" else ev for col in MASK}
        mcts = {col: MCTS(evs[col], cfg, random.Random(seed ^ MASK[col])) for col in MASK}
        searches, observed = [], []
        for c in CUTS:
            st = states[c]
            t = time.perf_counter()
            counts, value, root = mcts[st.to_move].search_with_root(st, add_noise=False)
            secs = time.perf_counter() - t
            if root.visit_count != cfg.n_simulations:
                raise Stop(f"sequence {n} position {c}: root visits {root.visit_count}")
            top = min(counts, key=lambda m: (-counts[m], tuple(m)))
            observed.append({"counts": sorted((tuple(m), v) for m, v in counts.items()), "value": value,
                             "top": tuple(top)})
            searches.append({"moves_applied": c, "to_move": st.to_move, "seconds": secs, "root_value": value,
                             "most_visited": list(top), "most_visited_count": counts[top]})
        observed.append({col: repr(mcts[col].rng.getstate()) for col in MASK})
        if reference is None:
            reference = observed
        else:
            for i, c in enumerate(CUTS):
                for field in ("counts", "value", "top"):
                    if observed[i][field] != reference[i][field]:
                        raise Stop(f"sequence {n} ({path}) position {c}: {field} differs from the first ordinary")
            if observed[-1] != reference[-1]:
                raise Stop(f"sequence {n} ({path}): final search RNG state differs from the first ordinary")
        seq = {"index": n, "path": path, "searches": searches, "search_seconds": sum(x["seconds"] for x in searches)}
        if path == "cached":
            seq["cache"] = {col: {"hits": evs[col].hits, "misses": evs[col].misses, "entries": len(evs[col].store),
                                  "stored_bytes": evs[col].stored_bytes()} for col in MASK}
        rec["sequences"].append(seq)
    rec["checks"] = "every search equal to the first ordinary sequence (visit counts, root value, most-visited " \
                    "move); final RNG states equal"
    tot = {p: sum(q["search_seconds"] for q in rec["sequences"] if q["path"] == p) for p in ("ordinary", "cached")}
    rec["combined_search_seconds"] = tot
    rec["reduction"] = 1 - tot["cached"] / tot["ordinary"]
    rec["answer"] = "QUALIFIES" if rec["reduction"] >= BAR else "STOP_CONFIGURATION"
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "eval-reuse-cache-screen/1", "script": "docs/superpowers/evidence/eval-reuse/02_cache_screen.py",
           "head": git("rev-parse", "HEAD"), "bar": BAR, "order": list(ORDER)}
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
        t = rec["combined_search_seconds"]
        print(f"COMPLETED: {rec['answer']}; combined search ordinary {t['ordinary']:.3f} s, cached {t['cached']:.3f} s, "
              f"reduction {100 * rec['reduction']:.2f}% (bar 10%); all checks passed; {rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
