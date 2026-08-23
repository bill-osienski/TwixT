#!/usr/bin/env python3
"""G2 — determinism, for the twixtbot anchor pilot. FAIL-CLOSED.

Card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md, gate G2.
One fixed position; trials=100, temperature=0, add_noise=0, rotation=ROT_OFF.
20 queries in one process + 5 queries in 5 fresh processes. All 25 must return
the identical move AND identical top-three visit counts.

This probe ASSERTS the criterion and exits non-zero on any mismatch. It does not
print results for a human to compare: near-identical output reads as identical.

  exit 0 = all 25 agree      exit 1 = mismatch (G2 FAIL)      exit 2 = cannot run

usage: g2_probe.py <clone> all | g2_probe.py <clone> one
"""
import json, os, subprocess, sys

TRIALS, TEMPERATURE, ADD_NOISE = 100, 0, 0
MOVES = [(12, 12), (10, 11), (13, 10), (11, 13)]   # same fixed position as G1
N_IN_PROCESS, N_SUBPROCESS = 20, 5


def make_result(player, twixt, Point):
    """One query under a FRESH Player. NeuralMCTS caches self.root between
    calls, so reusing a player would accumulate a tree and this would test
    caching, not determinism."""
    game = twixt.Game(allow_scl=False)
    for x, y in MOVES:
        game.play(Point(x, y))
    resp = player().pick_move(game)
    return {
        "move": [int(resp["moves"][0].x), int(resp["moves"][0].y)],
        "top3_moves": [[int(m.x), int(m.y)] for m in resp["moves"][:3]],
        "top3_visits": [int(v) for v in resp["Y"][:3]],
        "proven": bool(resp["proven"]),
    }


def build(clone):
    sys.path.insert(0, clone)
    os.chdir(clone)                      # NNEvaluater joins os.getcwd()
    from src.backend import nnmplayer, twixt          # noqa
    from src.backend.point import Point               # noqa
    from src import constants as ct                   # noqa
    kw = dict(model="model/pb", trials=TRIALS, temperature=TEMPERATURE,
              add_noise=ADD_NOISE, rotation=ct.ROT_OFF, allow_swap=0)
    shared = nnmplayer.Player(**kw).evaluator          # one TF session, reused
    return (lambda: nnmplayer.Player(evaluator=shared, **kw)), twixt, Point


def disagreements(results):
    """Pure. Returns [] when every result matches the first on all three fields."""
    if not results:
        return ["no results collected"]
    ref, bad = results[0], []
    for i, r in enumerate(results):
        for f in ("move", "top3_moves", "top3_visits"):
            if r[f] != ref[f]:
                bad.append(f"result[{i}].{f} = {r[f]!r} != result[0].{f} = {ref[f]!r}")
    return bad


def self_test():
    """A comparator that has never rejected anything is not known to bind."""
    base = {"move": [1, 2], "top3_moves": [[1, 2], [3, 4], [5, 6]],
            "top3_visits": [50, 30, 20], "proven": False}
    checks = [("identical", [base, dict(base)], True)]
    for f, bad in [("move", [9, 9]), ("top3_moves", [[9, 9], [3, 4], [5, 6]]),
                   ("top3_visits", [51, 30, 20])]:
        checks.append((f, [base, {**base, f: bad}], False))
    out = []
    for label, rs, expect_ok in checks:
        ok = not disagreements(rs)
        out.append((label, "ok" if ok == expect_ok else "COMPARATOR BROKEN"))
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1]); return 2
    clone, mode = sys.argv[1], sys.argv[2]
    if not os.path.isdir(clone):
        print(f"FAIL: clone not found at {clone}"); return 2
    probe = os.path.abspath(__file__)

    if mode == "one":
        player, twixt, Point = build(clone)
        print(json.dumps(make_result(player, twixt, Point)))
        return 0

    if mode != "all":
        print(f"FAIL: unknown mode {mode}"); return 2

    print("comparator self-test (a mismatch in any field must be caught):")
    for label, verdict in self_test():
        print(f"  {verdict:18} {label}")
        if verdict != "ok":
            print("\nFAIL: the comparator does not bind."); return 1

    player, twixt, Point = build(clone)
    results = [make_result(player, twixt, Point) for _ in range(N_IN_PROCESS)]
    print(f"\ncollected {len(results)} in-process results")

    for i in range(N_SUBPROCESS):
        p = subprocess.run([sys.executable, probe, clone, "one"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(f"FAIL: subprocess {i} exited {p.returncode}\n{p.stderr[-2000:]}")
            return 1
        results.append(json.loads(p.stdout.strip().splitlines()[-1]))
    print(f"collected {N_SUBPROCESS} fresh-process results; {len(results)} total")

    expected = N_IN_PROCESS + N_SUBPROCESS
    if len(results) != expected:
        print(f"FAIL: {len(results)} results, expected {expected}"); return 1

    bad = disagreements(results)
    json.dump(results, open(os.path.join(os.path.dirname(probe),
                                         "g2_raw_results.json"), "w"), indent=1)
    if bad:
        print(f"\nG2 FAIL — {len(bad)} disagreement(s):")
        for b in bad[:20]:
            print("  " + b)
        return 1

    r = results[0]
    print(f"\nall {len(results)} identical:")
    print(f"  move         {r['move']}")
    print(f"  top3 moves   {r['top3_moves']}")
    print(f"  top3 visits  {r['top3_visits']}")
    print(f"  proven       {r['proven']}")
    print("\nG2 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
