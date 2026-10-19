"""SELF-PLAY DIAGNOSTIC -- MIRROR CONSISTENCY: does calib020_0001 predict the same for a position and its
left-right reflection? Eight inference calls, no search, no game, no seed, no training.

Positions (the four committed diagnostic replays only): the first 20 moves of games 0 and 2, the first 21 of
games 1 and 3, applied to a fresh 24x24 state (280-ply cap). The reflection replays the COMPLETE history with
every move (r, c) -> (r, 23 - c), so the engine rebuilds pegs and links itself; refused unless the side to move
matches and the legal sets correspond exactly under the map.
One load through eval_runner's factory (evaluation mode checked: 0 modules in training mode, SHA-1 pinned). Each
position is evaluated alone (batch 1) exactly as mcts.py does: build_input_tensor, (C,H,W) -> (H,W,C), legal
moves in state.legal_moves() order, evaluator.infer. Reflected priors are mapped back before comparing.
Reported per position: both values (side to move) and |difference|; policy total variation; max |prior
difference|; top move of each (the reflected one mapped back) and whether they agree. Verdict WITHIN_TOLERANCE iff
every |value difference|, total variation and max prior difference <= 1e-5 and every top move agrees; otherwise
DISCREPANT. Descriptive only.

  --checkpoint PATH (absolute)   --out DIR (exists, without record.json)
Exit 0 COMPLETED / 2 STOP / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 50` (60 s in all).
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
CUTS = {0: 20, 1: 21, 2: 20, 3: 21}
SIZE, TOL = 24, 1e-5
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def mirror(m):
    return (m[0], SIZE - 1 - m[1])


def evaluate(ev, state, np):
    """mcts.py's leaf evaluation for one state (batch 1)."""
    moves = state.legal_moves()
    board = np.transpose(ev.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[r for r, _ in moves]], dtype=np.int32)
    cols = np.array([[c for _, c in moves]], dtype=np.int32)
    mask = np.ones((1, len(moves)), dtype=np.float32)
    priors, values = ev.infer(board, rows, cols, mask, state.active_size)
    return {tuple(m): float(p) for m, p in zip(moves, priors[0])}, float(values[0])


def main(rec, argv):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    positions = []
    for g, cut in CUTS.items():
        moves = [(m["row"], m["col"]) for m in json.loads((HERE / f"games/game_{g:06d}.json").read_text())["moves"]]
        s, sm = (TwixtState(active_size=SIZE, to_move="red", max_plies_limit=280) for _ in range(2))
        for m in moves[:cut]:
            s, sm = s.apply_move(m), sm.apply_move(mirror(m))
        if s.is_terminal() or sm.is_terminal() or s.to_move != sm.to_move or \
                sorted(mirror(m) for m in s.legal_moves()) != sorted(map(tuple, sm.legal_moves())):
            raise Refused(f"game {g}: the reflected position does not correspond")
        positions.append((g, cut, s, sm))

    if hashlib.sha1(ckpt.read_bytes()).hexdigest() != SHA1:
        raise Refused("the checkpoint is not the pinned calib020_0001")
    ev = ER._default_evaluator_factory(str(ckpt))
    rec["modules_in_training_mode"] = sum(bool(x.training) for _, x in ev.network.named_modules())
    if ev.network.training or rec["modules_in_training_mode"]:
        raise Stop("the network is not in evaluation mode")

    rows, calls = [], 0
    for g, cut, s, sm in positions:
        p, v = evaluate(ev, s, np)
        q_raw, vm = evaluate(ev, sm, np)
        calls += 2
        q = {mirror(m): x for m, x in q_raw.items()}               # reflected priors mapped back
        diffs = [abs(p[m] - q[m]) for m in p]
        top_p = min(p, key=lambda m: (-p[m], m))
        top_q = min(q, key=lambda m: (-q[m], m))
        rows.append({"game": g, "moves_applied": cut, "to_move": s.to_move, "legal": len(p),
                     "value": v, "value_reflected": vm, "value_abs_diff": abs(v - vm),
                     "policy_total_variation": 0.5 * sum(diffs), "policy_max_abs_diff": max(diffs),
                     "top_move": list(top_p), "top_prior": p[top_p],
                     "top_move_reflected_mapped_back": list(top_q), "top_prior_reflected": q[top_q],
                     "top_moves_agree": top_p == top_q})
    rec["inference_calls"] = calls
    rec["positions"] = rows
    rec["max"] = {k: max(r[k] for r in rows) for k in ("value_abs_diff", "policy_total_variation",
                                                       "policy_max_abs_diff")}
    rec["answer"] = "WITHIN_TOLERANCE" if all(x <= TOL for x in rec["max"].values()) and \
        all(r["top_moves_agree"] for r in rows) else "DISCREPANT"
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "self-play-diagnostic-mirror-probe/1",
           "script": "docs/superpowers/evidence/self-play-diagnostic/07_mirror_probe.py",
           "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "tolerance": TOL}
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
        print(f"COMPLETED: {rec['answer']}; max {rec['max']}; {rec['inference_calls']} calls; {rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
