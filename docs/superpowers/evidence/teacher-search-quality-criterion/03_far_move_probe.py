"""READ-ONLY trap probe at every decision of the stage-1 TRAINING games (families 0-3; 4-5 never opened).
LOSS4(m): after m the opponent wins at once or has a reply leaving the mover UNDEFENDABLE (exact, our rules).
A position is CONFIRMED LIVE when one of the PROBES -- the 3 legal moves farthest from every peg (pass-like)
-- is LOSS4. No hit is NOT a proof of a dead position (a cheap null-move proof is unsound here: an own peg
can pre-empt an own later link, since own crossings are forbidden). Writes nothing."""
import collections, json, pathlib, sys, time
from scripts.GPU.alphazero import h5_incumbent_tactics as T
D = pathlib.Path("docs/superpowers/evidence/external-engine-learning-stage1/games")
P = "protocol"

def cls(s):
    c = T.classify(s, semantics=P)
    return None if c is None else c["class"]

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
            s = T.replay(moves[:i], semantics=P)
            pegs = list(s.pegs)
            probes = sorted(s.legal_moves(), key=lambda x: (-min(abs(x[0] - p[0]) + abs(x[1] - p[1])
                                                                 for p in pegs), x))[:3]
            hit = next((x for x in probes if loss4(s, x)), None)
            out[(m["mover"], "LIVE" if hit else "no_hit")] += 1
            if hit:
                rows.append((g["game_id"], i, m["mover"], hit))
        print(g["game_id"], round(time.time() - t0), file=sys.stderr, flush=True)
for k, v in sorted(out.items()): print(k, v)
for r in rows: print(r)
print("seconds", round(time.time() - t0))
