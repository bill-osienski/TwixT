"""MIRROR-AVERAGING PROTOTYPE -- one bounded correctness check. No search, game, seed or training.

The four diagnostic positions of self-play-diagnostic/07_mirror_probe.py (first 20 moves of games 0 and 2, 21 of
games 1 and 3; the reflection replays the whole history with (r, c) -> (r, 23 - c) through the engine). One load of
calib020_0001 through eval_runner's factory (eval mode checked, SHA-1 pinned). Per position, at batch 1 as mcts.py
evaluates a leaf:
* PROTOTYPE: MirrorAveragingEvaluator(base).infer on the original position (2 underlying calls: the original and
  its tensor-mirrored board);
* REFERENCE, independent of the prototype's mirror: base.infer on the original state and on the ENGINE-REBUILT
  reflected state (2 calls), the reflected priors mapped back by move, then averaged; values averaged.
Underlying base.infer calls are counted: exactly 16, or STOP. Every loaded repository module must be committed and
unmodified (run from a clean tree). PASS iff for every position: |prior difference| and |value difference| <=
1e-5, all outputs finite, the prototype's priors cover exactly the legal moves, are >= 0 and sum to 1 within 1e-5.

  --checkpoint PATH (absolute)   --out DIR (exists, without record.json)
Exit 0 COMPLETED (verdict in the record) / 2 STOP / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 50`.
"""
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GAMES = ROOT / "docs/superpowers/evidence/self-play-diagnostic/games"
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
CUTS = {0: 20, 1: 21, 2: 20, 3: 21}
SIZE, TOL, CALLS = 24, 1e-5, 16
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def mirror(m):
    return (m[0], SIZE - 1 - m[1])


def evaluate(ev, state, np):
    """mcts.py's leaf evaluation for one state (batch 1)."""
    moves = [tuple(m) for m in state.legal_moves()]
    board = np.transpose(ev.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[r for r, _ in moves]], dtype=np.int32)
    cols = np.array([[c for _, c in moves]], dtype=np.int32)
    priors, values = ev.infer(board, rows, cols, np.ones((1, len(moves)), dtype=np.float32), state.active_size)
    return moves, [float(x) for x in priors[0]], float(values[0])


def main(rec, argv):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator

    positions = []
    for g, cut in CUTS.items():
        moves = [(m["row"], m["col"]) for m in json.loads((GAMES / f"game_{g:06d}.json").read_text())["moves"]]
        s, sm = (TwixtState(active_size=SIZE, to_move="red", max_plies_limit=280) for _ in range(2))
        for m in moves[:cut]:
            s, sm = s.apply_move(m), sm.apply_move(mirror(m))
        if s.is_terminal() or sm.is_terminal() or s.to_move != sm.to_move or \
                sorted(mirror(m) for m in s.legal_moves()) != sorted(map(tuple, sm.legal_moves())):
            raise Refused(f"game {g}: the reflected position does not correspond")
        positions.append((g, cut, s, sm))

    if hashlib.sha1(ckpt.read_bytes()).hexdigest() != SHA1:
        raise Refused("the checkpoint is not the pinned calib020_0001")
    base = ER._default_evaluator_factory(str(ckpt))
    rec["modules_in_training_mode"] = sum(bool(x.training) for _, x in base.network.named_modules())
    if base.network.training or rec["modules_in_training_mode"]:
        raise Stop("the network is not in evaluation mode")
    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if (n == "__main__" or n.split(".")[0] == "scripts") and getattr(m, "__file__", None)}
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Stop("a loaded repository module is uncommitted or differs from HEAD")
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}
    count, real_infer = [0], base.infer

    def counted(*a, **k):
        count[0] += 1
        return real_infer(*a, **k)

    base.infer = counted
    proto = MirrorAveragingEvaluator(base)

    rows = []
    for g, cut, s, sm in positions:
        moves, a_p, a_v = evaluate(proto, s, np)
        _, p, v = evaluate(base, s, np)
        m_moves, q, vm = evaluate(base, sm, np)
        qmap = {mirror(m): x for m, x in zip(m_moves, q)}
        ref_p = [(x + qmap[m]) / 2 for m, x in zip(moves, p)]
        ref_v = (v + vm) / 2
        prior_diff = max(abs(x - y) for x, y in zip(a_p, ref_p))
        rows.append({"game": g, "moves_applied": cut, "to_move": s.to_move, "legal": len(moves),
                     "prototype_value": a_v, "reference_value": ref_v, "value_abs_diff": abs(a_v - ref_v),
                     "prior_max_abs_diff": prior_diff, "prototype_prior_sum": sum(a_p),
                     "prototype_prior_min": min(a_p), "prototype_priors_cover_legal": len(a_p) == len(moves),
                     "finite": all(math.isfinite(x) for x in [a_v, ref_v, *a_p, *ref_p]),
                     "value_original": v, "value_reflected": vm})
    rec["underlying_inference_calls"] = count[0]
    if count[0] != CALLS:
        raise Stop(f"{count[0]} underlying inference calls, not {CALLS}")
    rec["positions"] = rows
    ok = all(r["value_abs_diff"] <= TOL and r["prior_max_abs_diff"] <= TOL and r["finite"] and
             r["prototype_priors_cover_legal"] and r["prototype_prior_min"] >= 0 and
             abs(r["prototype_prior_sum"] - 1) <= TOL for r in rows)
    rec["answer"] = "PASS" if ok else "FAIL"
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "mirror-averaging-check/1", "script": "docs/superpowers/evidence/mirror-averaging/01_check.py",
           "head": git("rev-parse", "HEAD"), "tolerance": TOL}
    t = time.monotonic()
    try:
        verdict = main(rec, sys.argv)
    except Refused as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Stop as e:
        verdict, rec["reason"] = "STOP", str(e)
    except Exception:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        verdict, rec["reason"] = "STOP_ERROR", traceback.format_exc()
    rec["verdict"], rec["seconds"] = verdict, round(time.monotonic() - t, 3)
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if verdict == "COMPLETED":
        r = rec["positions"]
        print(f"COMPLETED: {rec['answer']}; max value diff {max(x['value_abs_diff'] for x in r):.3g}, max prior "
              f"diff {max(x['prior_max_abs_diff'] for x in r):.3g}; {rec['underlying_inference_calls']} calls; "
              f"{rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
