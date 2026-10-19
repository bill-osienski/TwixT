"""SELF-PLAY DIAGNOSTIC -- RULES ENGINE ONLY: after the save (21,6), can red force a win within three further plies
(red, black, red)? No model, no search, no seed.

The position is 05_probe.py's (game 3's first 41 moves, black to move; refused unless red's immediate wins and the
saving set are exactly 04_scan.json's), then (21,6). Exact checks only, from d0_postmortem and h5_weakness_review:
* ply 1: red's immediate wins (D0.one_ply_threats) -- any is a proof;
* else, for EVERY legal red move R: black's immediate wins after R (any refutes R); red's threats T after R
  (none refutes R); black's saving replies (RV.saving_moves_of, the H5 exact reduction to T). R FORCES a win iff
  black has no immediate win, T is non-empty and no black reply saves.
For a forcing R: every legal black reply with a red move that wins against it, each re-verified by applying it.
For a refuted R: one black reply and a FULL scan (one_ply_threats, no reduction) showing red has no immediate win.
Verdict FORCED_WIN_WITHIN_3 (with the forcing moves) or NOT_FORCED_WITHIN_3: no red move wins within these
three plies. The second is bounded: it does not show that black can save the game.

  --out DIR (exists, without record.json)
Exit 0 COMPLETED / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 50` (60 s in all).
"""
import json
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY, PLY, SAVE = HERE / "games/game_000003.json", 41, (21, 6)
EXIT = {"COMPLETED": 0, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


def position(RV, D0, TwixtState):
    """05_probe.py's position(), unchanged."""
    g = json.loads(REPLAY.read_text())
    scan = json.loads((HERE / "04_scan.json").read_text())["mistakes"]
    [m] = [m for m in scan if m["replay"].endswith(REPLAY.name) and m["ply_index"] == PLY]
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for mv in g["moves"][:PLY]:
        s = s.apply_move((mv["row"], mv["col"]))
    T = D0.one_ply_threats(D0._flip(s))
    saving = RV.saving_moves_of(s, T)
    if s.to_move != "black" or D0.one_ply_threats(s) or [list(t) for t in T] != m["opponent_threats"] or \
            saving != [SAVE] or [g["moves"][PLY]["row"], g["moves"][PLY]["col"]] != m["played"]:
        raise Refused("the rebuilt position is not 04_scan.json's missed defence")
    return s, {"ply_index": PLY, "red_threats_before_save": [list(t) for t in T], "save": list(SAVE)}


def main(rec):
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import d0_postmortem as D0
    from scripts.GPU.alphazero import h5_weakness_review as RV
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    s0, rec["position"] = position(RV, D0, TwixtState)
    s1 = s0.apply_move(SAVE)
    if s1.to_move != "red" or s1.is_terminal():
        raise Refused("after (21,6) it is not red to move in a live game")
    W1 = D0.one_ply_threats(s1)
    rec["ply1_red_immediate_wins"] = [list(w) for w in W1]
    forcing, refuted = [], []
    if not W1:
        for R in s1.legal_moves():
            s2 = s1.apply_move(R)
            if s2.is_terminal():
                raise Refused(f"red {R} ends the game without winning")
            black_wins = D0.one_ply_threats(s2)
            T = D0.one_ply_threats(D0._flip(s2))
            if black_wins:
                refuted.append({"red": list(R), "why": "BLACK_WINS_IMMEDIATELY", "black": list(black_wins[0])})
                continue
            saving = RV.saving_moves_of(s2, T) if T else None
            if T and not saving:
                table = []
                for B in s2.legal_moves():
                    s3 = s2.apply_move(B)
                    win = next((t for t in T if s3.is_valid_placement(*t) and s3.apply_move(t).winner() == "red"),
                               None)
                    if win is None:
                        raise Refused(f"red {R}, black {B}: no red win in T although no reply saves")
                    table.append({"black": list(B), "red_wins_with": list(win)})
                forcing.append({"red": list(R), "red_threats": [list(t) for t in T], "replies": table})
                continue
            B = saving[0] if saving else s2.legal_moves()[0]
            left = D0.one_ply_threats(s2.apply_move(B))            # full scan, no reduction
            if left:
                raise Refused(f"red {R}, black {B}: the reduction said saving, the full scan found {left}")
            refuted.append({"red": list(R), "why": "BLACK_SAVES" if T else "NO_RED_THREAT",
                            "red_threats": [list(t) for t in T], "black": list(B),
                            "saving_replies": len(saving) if saving else None})
        rec["red_moves_checked"] = len(s1.legal_moves())
    rec["forcing_red_moves"] = forcing
    rec["refuted_red_moves"] = refuted
    rec["refutations_by_reason"] = {k: sum(r["why"] == k for r in refuted)
                                    for k in ("BLACK_WINS_IMMEDIATELY", "BLACK_SAVES", "NO_RED_THREAT")}
    rec["answer"] = "FORCED_WIN_WITHIN_3" if (W1 or forcing) else "NOT_FORCED_WITHIN_3"
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "self-play-diagnostic-forced-loss/1",
           "script": "docs/superpowers/evidence/self-play-diagnostic/06_forced_loss.py",
           "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "horizon": "red, black, red after (21,6)"}
    t = time.monotonic()
    try:
        verdict = main(rec)
    except Refused as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Exception:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        verdict, rec["reason"] = "STOP_ERROR", traceback.format_exc()
    rec["verdict"], rec["seconds"] = verdict, round(time.monotonic() - t, 3)
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if verdict == "COMPLETED":
        print(f"COMPLETED: {rec['answer']}; forcing red moves {[f['red'] for f in rec['forcing_red_moves']]}; "
              f"refuted {rec['refutations_by_reason']}; {rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
