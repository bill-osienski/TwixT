"""READ-ONLY liveness at every decision of the stage-1 TRAINING games (families 0-3; 4-5 never opened).
A move m is LOSS4 if after it the opponent wins at once or has a reply r leaving the mover UNDEFENDABLE.
A mover's own peg only occupies a cell and blocks crossings, so it never adds an opponent option and never
removes a mover defence: P+m+r UNDEFENDABLE => P+r (mover to move) UNDEFENDABLE. So with the OPPONENT to
move instead (a null move), T0 = its immediate wins, R0 = its replies leaving the mover UNDEFENDABLE;
T0, R0 empty => NO move is LOSS4 (proved). Non-empty => candidate; confirmed by one LOSS4 move found.
Writes nothing."""
import collections, json, pathlib, sys, time
from scripts.GPU.alphazero import h5_incumbent_tactics as T
from scripts.GPU.alphazero.game.twixt_state import TwixtState
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
            opp = "black" if s.to_move == "red" else "red"
            n = TwixtState(board_size=s.board_size, active_size=s.active_size, to_move=opp,
                           pegs=dict(s.pegs), bridges=set(s.bridges), ply=s.ply)
            T0 = [r for r in n.legal_moves() if n.apply_move(r).winner() == opp]
            R0 = [r for r in n.legal_moves() if r not in T0 and cls(n.apply_move(r)) == "UNDEFENDABLE"]
            if not (T0 or R0):
                out[(m["mover"], "dead")] += 1
                continue
            # confirm: a move far from everything is the first tried; then the rest in order
            trap = next((x for x in sorted(s.legal_moves(), key=lambda x: -min(
                [abs(x[0] - y[0]) + abs(x[1] - y[1]) for y in T0 + R0])) if loss4(s, x)), None)
            out[(m["mover"], "LIVE" if trap else "candidate_unconfirmed")] += 1
            rows.append((g["game_id"], i, m["mover"], len(T0), len(R0), trap))
        print(g["game_id"], round(time.time() - t0), file=sys.stderr, flush=True)
for k, v in sorted(out.items()): print(k, v)
for r in rows: print(r)
print("seconds", round(time.time() - t0))
