"""EXPLORATORY -- where the winning holes sit at the LOSER's first threatened turn.

Geometric and one-ply only (demonstrated): for every distinct decided trajectory, the
board threats T at the loser's first threatened turn are classified as all on the
WINNER's own goal edge (cells the loser may not occupy), none there, or mixed.
Read-only; create-only output 06b_edge_holes.json.
"""
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
THREAT_LABELS = ("DEFENDED", "MISSED_DEFENCE", "UNDEFENDABLE")


def on_winner_edge(hole, winner_colour):
    r, c = hole
    return r in (0, 23) if winner_colour == "red" else c in (0, 23)


def main():
    hits = json.loads((HERE / "03_mechanical_checks.json").read_text())["hits"]
    by = collections.defaultdict(list)
    for h in hits:
        by[h["digest"]].append(h)
    out = collections.defaultdict(lambda: collections.Counter())
    for d, hs in by.items():
        arm, outcome, freq = hs[0]["arm"], hs[0]["outcome"], hs[0]["frequency"]
        if outcome not in ("win", "loss"):
            continue
        loser = "incumbent" if outcome == "loss" else "t1j"
        turns = sorted((h for h in hs if h["engine"] == loser and h["label"] in THREAT_LABELS),
                       key=lambda h: h["ply"])
        if not turns:
            continue
        t = turns[0]
        winner_colour = "black" if t["mover"] == "red" else "red"
        edge = [on_winner_edge(x, winner_colour) for x in t["T"]]
        where = "all on the winner's own goal edge" if all(edge) else \
                "none on it" if not any(edge) else "mixed"
        key = f"{arm}/{loser} loses/{t['label']}/{'multi' if len(t['T']) >= 2 else 'single'}-hole"
        out[key][f"{where} (distinct)"] += 1
        out[key][f"{where} (games)"] += freq
    result = {"label": "EXPLORATORY", "tables": {k: dict(v) for k, v in sorted(out.items())}}
    with open(HERE / "06b_edge_holes.json", "x", encoding="utf-8") as fh:  # create-only
        json.dump(result, fh, indent=1, sort_keys=True)
    for k, v in result["tables"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
