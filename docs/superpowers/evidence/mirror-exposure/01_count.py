"""MIRROR-DEFECT EXPOSURE -- rules-only count: on the four committed self-play diagnostic replays, how often would
the OLD and the CORRECTED training mirror produce different tensors? No model, search, game, seed or training.

Positions: every pre-move position of games 0-3 (ply 0 to the last move), applied to a fresh 24x24 state (280-ply
cap), tensor built as self_play.play_game records it (state.to_tensor(), (C,H,W) -> (H,W,C)).
CORRECTED = self_play._mirror_position_lr (with 0c417882's 27 <-> 28 swap). OLD = CORRECTED with channels 27 and 28
swapped back (the swap is its own inverse, so this is the pre-fix helper exactly). A position is EXPOSED iff OLD !=
CORRECTED anywhere, i.e. its mirrored training sample would have carried wrong black edge-connectivity labels.
Integrity check (any failure STOPs): CORRECTED equals the tensor the engine builds from the reflected history
((r, c) -> (r, 23 - c)), at every position.
Reported per position: ply, side to move, black pegs, cells set in 27 / 28, exposed; totals per game and overall.
Measures exposure only, not benefit; descriptive, no decision rule. Self-play mirrors each position with
probability 0.5, so the expected exposed share of a buffer is (exposed / positions) / 3.

  --out DIR (exists, without record.json)
Exit 0 COMPLETED / 2 STOP / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 50` (60 s in all), in a clean tree.
"""
import json
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GAMES = ROOT / "docs/superpowers/evidence/self-play-diagnostic/games"
SIZE = 24
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def main(rec):
    sys.path.insert(0, str(ROOT))
    import hashlib
    import numpy as np
    from scripts.GPU.alphazero.game.twixt_state import (CHANNEL_BLACK_CONN_LEFT as L, CHANNEL_BLACK_CONN_RIGHT as R,
                                                        TwixtState)
    from scripts.GPU.alphazero.self_play import _mirror_position_lr

    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if (n == "__main__" or n.split(".")[0] == "scripts") and getattr(m, "__file__", None)}
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Refused("a loaded repository module is uncommitted or differs from HEAD")
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}

    def hwc(state):
        return np.transpose(state.to_tensor(), (1, 2, 0))

    rec["games"] = []
    for g in range(4):
        moves = [(m["row"], m["col"]) for m in json.loads((GAMES / f"game_{g:06d}.json").read_text())["moves"]]
        s, sm = (TwixtState(active_size=SIZE, to_move="red", max_plies_limit=280) for _ in range(2))
        rows = []
        for ply, m in enumerate(moves):
            if s.is_terminal() or sm.is_terminal() or s.to_move != sm.to_move:
                raise Refused(f"game {g} ply {ply}: not a live, corresponding position")
            board = hwc(s)
            corrected = _mirror_position_lr(board, [], [], SIZE)[0]
            if not np.array_equal(corrected, hwc(sm)):
                bad = sorted({int(c) for c in np.nonzero(corrected != hwc(sm))[2]})
                raise Stop(f"game {g} ply {ply}: corrected mirror differs from the engine reflection on {bad}")
            old = corrected.copy()
            old[:, :, [L, R]] = corrected[:, :, [R, L]]
            rows.append({"ply": ply, "to_move": s.to_move, "black_pegs": int(board[:, :, 1].sum()),
                         "conn_left_cells": int(board[:, :, L].sum()), "conn_right_cells": int(board[:, :, R].sum()),
                         "exposed": not np.array_equal(old, corrected)})
            s, sm = s.apply_move(m), sm.apply_move((m[0], SIZE - 1 - m[1]))
        exposed = [r["ply"] for r in rows if r["exposed"]]
        rec["games"].append({"game": g, "positions": len(rows), "exposed": len(exposed),
                             "first_exposed_ply": exposed[0] if exposed else None, "rows": rows})
    n = sum(x["positions"] for x in rec["games"])
    e = sum(x["exposed"] for x in rec["games"])
    rec["totals"] = {"positions": n, "exposed": e, "exposed_fraction": e / n,
                     "expected_exposed_buffer_share_at_mirror_prob_0.5": e / n / 3}
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "mirror-exposure-count/1", "script": "docs/superpowers/evidence/mirror-exposure/01_count.py",
           "head": git("rev-parse", "HEAD")}
    t0 = time.monotonic()
    try:
        verdict = main(rec)
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
        per = ", ".join(f"g{x['game']} {x['exposed']}/{x['positions']}" for x in rec["games"])
        print(f"COMPLETED: {t['exposed']}/{t['positions']} positions exposed ({100 * t['exposed_fraction']:.1f}%; "
              f"{per}); integrity check passed; {rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
