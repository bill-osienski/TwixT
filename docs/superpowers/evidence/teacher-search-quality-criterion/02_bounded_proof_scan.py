"""READ-ONLY: bounded exact values (our rules, protocol semantics) at every decision of the stage-1
TRAINING games (families 0-3). Held-out families 4-5 are never opened. Writes nothing.
WIN3(m): m wins at once, or leaves the opponent UNDEFENDABLE.  LOSS4(m): after m the opponent wins at
once (OWN_WIN) or has a reply leaving the mover UNDEFENDABLE.  Only the CHOSEN move's LOSS4 is resolved
in full; for a live position we look for one alternative that is not LOSS4."""
import collections, json, pathlib, sys, time
from scripts.GPU.alphazero import h5_incumbent_tactics as T
D = pathlib.Path("docs/superpowers/evidence/external-engine-learning-stage1/games")
P = "protocol"

def cls(s):
    c = T.classify(s, semantics=P)
    return None if c is None else c["class"]

def win3(s, m):
    a = s.apply_move(m)
    return a.winner() == s.to_move or cls(a) == "UNDEFENDABLE"

def loss4(s, m):
    a = s.apply_move(m)
    if a.winner() is not None or a.ply >= T.PLY_CAP:
        return False
    if cls(a) == "OWN_WIN":
        return True
    return any(cls(a.apply_move(r)) == "UNDEFENDABLE" for r in a.legal_moves()
               if a.apply_move(r).winner() is None)

out = collections.Counter(); rows = []; t0 = time.time()
for fam in (0, 1, 2, 3):
    for arm in ("teacher_red", "teacher_black"):
        g = json.loads((D / f"f{fam}-{arm}.json").read_text()); assert g["split"] == "train"
        moves = [m["move"] for m in g["moves"]]
        for i, m in enumerate(g["moves"]):
            if m["mover"] == "opening":
                continue
            s = T.replay(moves[:i], semantics=P); ch = tuple(m["move"]); legal = s.legal_moves()
            W = [x for x in legal if win3(s, x)]
            cl = loss4(s, ch)
            alt = None
            if cl:
                alt = next((x for x in legal if x != ch and not loss4(s, x)), None)
            key = (m["mover"], "win3_avail" if W else "-", "chosen_win3" if ch in W else "-",
                   "chosen_LOSS4" if cl else "-", "alt_ok" if alt else "-")
            out[key] += 1
            if W or cl:
                rows.append((g["game_id"], i, m["mover"], len(W), ch in W, cl, alt))
        print(g["game_id"], round(time.time() - t0), file=sys.stderr, flush=True)
for k, v in sorted(out.items()): print(k, v)
for r in rows: print(r)
print("seconds", round(time.time() - t0))
