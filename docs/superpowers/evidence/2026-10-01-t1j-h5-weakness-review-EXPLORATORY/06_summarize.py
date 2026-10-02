"""EXPLORATORY -- derived tables for the report, recomputed from this directory's run
outputs (01-04) and the committed H5 replay exports. Read-only; create-only output
(06_summary.json). Every table is descriptive: no interval, test or p-value.

Tables marked HYPOTHESIS-SUPPORT use a stated heuristic (the ladder detector, the
largest-component goal profile) and support a hypothesis; they demonstrate nothing.
"""
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.GPU.alphazero import d0_postmortem as D0                      # noqa: E402
from scripts.GPU.alphazero.game.twixt_state import TwixtState               # noqa: E402

KNIGHT = {(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)}
THREAT_LABELS = ("DEFENDED", "MISSED_DEFENCE", "UNDEFENDABLE")


def load(name):
    return json.loads((HERE / name).read_text())


def replay_moves(rel):
    f = json.loads((ROOT / rel).read_text())
    return f, [(m["row"], m["col"]) for m in f["moves"]]


def adjacent(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def ladders(moves):
    """HEURISTIC, stated: maximal runs of ONE side's moves on alternate plies where each
    is a knight step from that side's previous move and the opponent's move in between
    is Chebyshev-adjacent to either. -> [(side parity, steps, last ply index)]."""
    out = []
    for parity in (0, 1):
        j, steps = parity, 0
        while j + 2 < len(moves):
            a, d, b = moves[j], moves[j + 1], moves[j + 2]
            if (b[0] - a[0], b[1] - a[1]) in KNIGHT and (adjacent(d, a) or adjacent(d, b)):
                steps += 1
            else:
                if steps:
                    out.append((parity, steps, j))
                steps = 0
            j += 2
        if steps:
            out.append((parity, steps, j))
    return out


def main():
    checks = load("03_mechanical_checks.json")
    caps = load("04_caps.json")
    tables = load("02_prefix_tables.json")
    by = collections.defaultdict(list)
    for h in checks["hits"]:
        by[h["digest"]].append(h)
    trajs = {}          # digest -> (arm, outcome, frequency, replays) for every trajectory with a hit
    for d, hs in by.items():
        trajs[d] = (hs[0]["arm"], hs[0]["outcome"], hs[0]["frequency"], hs[0]["replays"])
    out = {"label": "EXPLORATORY"}

    # (a) presence per distinct trajectory and per game
    pres = collections.defaultdict(lambda: {"distinct_trajectories": 0, "games": 0})
    for d, hs in by.items():
        arm, outcome, freq, _ = trajs[d]
        for eng, lab in {(h["engine"], h["label"]) for h in hs}:
            cell = pres[f"{arm}/{outcome}/{eng}/{lab}"]
            cell["distinct_trajectories"] += 1
            cell["games"] += freq
    out["presence_by_trajectory"] = dict(sorted(pres.items()))

    # (b) missed defences punished on the very next ply
    out["missed_defence_punished_next_ply"] = {
        eng: {"positions": sum(1 for h in checks["hits"] if h["engine"] == eng
                               and h["label"] == "MISSED_DEFENCE" and h["opponent_won_next_ply"]),
              "distinct_trajectories": len({h["digest"] for h in checks["hits"] if h["engine"] == eng
                                            and h["label"] == "MISSED_DEFENCE"
                                            and h["opponent_won_next_ply"]})}
        for eng in ("incumbent", "t1j")}

    # (c) the LOSER's first threatened turn, for incumbent losses and for incumbent wins
    first = {}
    for d, hs in by.items():
        arm, outcome, freq, reps = trajs[d]
        if outcome not in ("loss", "win"):
            continue
        loser = "incumbent" if outcome == "loss" else "t1j"
        turns = sorted((h for h in hs if h["engine"] == loser and h["label"] in THREAT_LABELS),
                       key=lambda h: h["ply"])
        if not turns:
            key = "no threatened turn"
        else:
            t = turns[0]
            key = (f"{'multi' if len(t['T']) >= 2 else 'single'}-hole/{t['label']}/"
                   f"{'ended next ply' if t['distance_to_terminal'] == 1 else 'continued'}")
        cell = first.setdefault(f"{arm}/{loser} loses", {}).setdefault(
            key, {"distinct_trajectories": 0, "games": 0, "examples": []})
        cell["distinct_trajectories"] += 1
        cell["games"] += freq
        if turns and len(cell["examples"]) < 3:
            cell["examples"].append({"replay": reps[0], "ply": turns[0]["ply"],
                                     "board_threats": turns[0]["T"]})
    out["loser_first_threatened_turn"] = first

    # (d) single-hole UNDEFENDABLE endings: is the hole on the attacker's own goal edge?
    single = collections.Counter()
    for d, hs in by.items():
        arm, outcome, freq, _ = trajs[d]
        if outcome != "loss":
            continue
        t = min((h for h in hs if h["engine"] == "incumbent" and h["label"] in THREAT_LABELS),
                key=lambda h: h["ply"])
        if t["label"] == "UNDEFENDABLE" and len(t["T"]) == 1:
            r, c = t["T"][0]
            attacker_edge = (c in (0, 23)) if arm == "A" else (r in (0, 23))   # A: T1j black
            single[f"{arm}: hole on T1j's own goal edge={attacker_edge}"] += 1
    out["single_hole_undefendable_location_distinct"] = dict(single)

    # (e) caps: the Arm-B loop and the largest-component goal profile (HYPOTHESIS-SUPPORT)
    loop = {h["digest"] for h in checks["hits"] if h["engine"] == "t1j" and h["label"] == "MISSED_WIN"}
    capsum = {}
    for arm, inc in (("A", "red"), ("B", "black")):
        t1j = "black" if inc == "red" else "red"
        prof = collections.Counter()
        for t in caps[arm]["trajectories"]:
            _, moves = replay_moves(t["replays"][0])
            s = TwixtState(active_size=24, to_move="red")
            for m in moves:
                s = s.apply_move(m)

            def largest(colour):
                c = max(D0.component_boundary_distances(s, colour), key=lambda c: c["size"])
                return f"goal1={c['to_goal1'] == 0},goal2={c['to_goal2'] == 0}"
            prof[f"{'loop' if t['digest'] in loop else 'no loop'} | incumbent largest "
                 f"{largest(inc)} | T1j largest {largest(t1j)}"] += t["frequency"]
        capsum[arm] = {"distinct_capped": len(caps[arm]["trajectories"]),
                       "with_loop": sum(t["digest"] in loop for t in caps[arm]["trajectories"]),
                       "profiles_HYPOTHESIS_SUPPORT": dict(prof.most_common())}
    out["caps"] = capsum

    # (f) ladders (HYPOTHESIS-SUPPORT), over every won game, both engines as the winner
    lad = collections.defaultdict(lambda: {"games": 0, "ladder_ge3": 0, "win_extends_ladder": 0,
                                           "distinct": set(), "distinct_ladder": set(),
                                           "distinct_extends": set()})
    for k in range(4):
        for p in sorted((ROOT / f"docs/superpowers/evidence/t1j-h5-study-segment{k}/replays").glob("*.json")):
            f, moves = replay_moves(p.relative_to(ROOT))
            if f["meta"]["reason"] != "win":
                continue
            par = (len(moves) - 1) % 2
            L = [x for x in ladders(moves) if x[0] == par]
            has = any(s >= 3 for _, s, _ in L)
            ext = any(s >= 3 and (moves[-1][0] - moves[e][0], moves[-1][1] - moves[e][1]) in KNIGHT
                      for _, s, e in L)
            loser = "incumbent" if f["meta"]["players"][f["winner"]].startswith("T1j") else "t1j"
            c = lad[f"{f['meta']['arm']}/{loser} loses"]
            c["games"] += 1
            c["ladder_ge3"] += has
            c["win_extends_ladder"] += ext
            c["distinct"].add(tuple(moves))
            if has:
                c["distinct_ladder"].add(tuple(moves))
            if ext:
                c["distinct_extends"].add(tuple(moves))
    out["ladders_HYPOTHESIS_SUPPORT"] = {
        k: {"games": v["games"], "winner_ran_ladder_ge3": v["ladder_ge3"],
            "winning_move_extends_ladder": v["win_extends_ladder"],
            "distinct": len(v["distinct"]), "distinct_ladder_ge3": len(v["distinct_ladder"]),
            "distinct_winning_move_extends": len(v["distinct_extends"])}
        for k, v in sorted(lad.items())}

    # (g) Arm B: T1j's first move (+ the incumbent's fixed reply) -> outcomes
    out["armB_depth2_outcomes"] = [
        {"prefix": r["prefix"], "occurrence": r["occurrence"], "distinct": r["distinct"],
         "recurs_in_losses": r["recurs_in_losses"]} for r in tables["B"]["2"]]

    # (h) Arm A losses: when the incumbent placed pegs on its own goal rows
    goal_rows = {}
    for d, (arm, outcome, freq, reps) in trajs.items():
        if arm == "A" and outcome == "loss":
            f, moves = replay_moves(reps[0])
            goal_rows[reps[0]] = {"plies": len(moves), "frequency": freq,
                                  "incumbent_goal_row_pegs_at_ply": [
                                      i + 1 for i, (r, c) in enumerate(moves)
                                      if i % 2 == 0 and r in (0, 23)]}
    out["armA_loss_goal_row_pegs"] = goal_rows

    with open(HERE / "06_summary.json", "x", encoding="utf-8") as fh:       # create-only
        json.dump(out, fh, indent=1, sort_keys=True)
    print("06_summary.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
