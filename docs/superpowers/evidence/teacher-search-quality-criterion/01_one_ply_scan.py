"""READ-ONLY: exact one-ply class of every decision in the stage-1 TRAINING games (families 0-3 only).
Held-out families 4-5 are never opened. Writes nothing."""
import collections, json, pathlib, sys, time
from scripts.GPU.alphazero import h5_incumbent_tactics as T
from scripts.GPU.alphazero import tactical_candidate as TC
D = pathlib.Path("docs/superpowers/evidence/external-engine-learning-stage1/games")
tally = collections.Counter(); events = []; t0 = time.time(); xc = 0
for fam in (0, 1, 2, 3):
    for arm in ("teacher_red", "teacher_black"):
        g = json.loads((D / f"f{fam}-{arm}.json").read_text())
        assert g["split"] == "train"
        moves = [m["move"] for m in g["moves"]]
        for i, m in enumerate(g["moves"]):
            if m["mover"] == "opening":
                continue
            s = T.replay(moves[:i], semantics="protocol")
            c = T.classify(s, semantics="protocol")
            if i % 4 == 0:
                xc += 1; assert TC.cross_check(s)["agree"], (g["game_id"], i)
            cls = c["class"]
            ok = tuple(m["move"]) in T.admissible(c, s.legal_moves())
            tally[(m["mover"], cls, ok)] += 1
            if cls != "QUIET":
                events.append((g["game_id"], i, m["mover"], cls, ok, len(c["W"] or c["D"] or []), len(c["T"] or [])))
        print(g["game_id"], g["plies"], g["winner"], g["teacher_colour"], file=sys.stderr)
for k, v in sorted(tally.items()): print(k, v)
for e in events: print(e)
print("cross-checks", xc, "all agree; seconds", round(time.time() - t0, 1))
