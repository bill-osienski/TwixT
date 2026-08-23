#!/usr/bin/env python3
"""G2 (attempt 2) — determinism, twixtbot anchor pilot. FAIL-CLOSED.

Card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md, gate G2.
Frozen settings UNCHANGED from attempt 1: trials=100, temperature=0,
add_noise=0, rotation=ROT_OFF, same fixed position.

Attempt 1 was BLOCKED: nnmcts.send_message() dereferences `window` with no None
guard and fires every MCTS_TRIAL_CHUNK=20 trials, so window=None raises for any
trials>=20. Authorized resolution: inject a RECORDING progress sink through the
existing `window` parameter. The engine is NOT modified.

Asserts its criterion internally; catches engine exceptions and reports them.
  exit 0 = all 25 agree   exit 1 = mismatch or engine error   exit 2 = cannot run

usage: g2_probe.py <clone> all | g2_probe.py <clone> one
"""
import json, os, subprocess, sys, traceback

TRIALS, TEMPERATURE, ADD_NOISE = 100, 0, 0
MOVES = [(12, 12), (10, 11), (13, 10), (11, 13)]
N_IN_PROCESS, N_SUBPROCESS = 20, 5
HERE = os.path.dirname(os.path.abspath(__file__))


class ProgressSink:
    """The ONLY thing the engine asks of `window`: write_event_value(key, value).

    Recording, not silent: a silent stub would discard the evidence that the
    injection changed nothing. Records the shape of each progress event, never
    returns anything the engine consumes (send_message's result is discarded),
    and holds no reference the engine can read back.
    """

    def __init__(self):
        self.events = []

    def write_event_value(self, key, value):
        self.events.append({
            "key": key,
            "status": value.get("status"),
            "current": value.get("current"),
            "max": value.get("max"),
            "proven": value.get("proven"),
            "n_moves": len(value.get("moves", [])),
        })


def one_result(make_player, twixt, Point):
    """One query: fresh Game, fresh Player, fresh sink."""
    game = twixt.Game(allow_scl=False)
    for x, y in MOVES:
        game.play(Point(x, y))
    sink = ProgressSink()
    resp = make_player().pick_move(game, window=sink)
    return {
        "move": [int(resp["moves"][0].x), int(resp["moves"][0].y)],
        "top3_moves": [[int(m.x), int(m.y)] for m in resp["moves"][:3]],
        "top3_visits": [int(v) for v in resp["Y"][:3]],
        "proven": bool(resp["proven"]),
        "progress_events": len(sink.events),
        "progress_currents": [e["current"] for e in sink.events],
        "progress_sample": sink.events[:2],
    }


def build(clone):
    sys.path.insert(0, clone)
    os.chdir(clone)
    from src.backend import nnmplayer, twixt
    from src.backend.point import Point
    from src import constants as ct
    kw = dict(model="model/pb", trials=TRIALS, temperature=TEMPERATURE,
              add_noise=ADD_NOISE, rotation=ct.ROT_OFF, allow_swap=0)
    shared = nnmplayer.Player(**kw).evaluator
    return (lambda: nnmplayer.Player(evaluator=shared, **kw)), twixt, Point


COMPARED = ("move", "top3_moves", "top3_visits")


def disagreements(results):
    if not results:
        return ["no results collected"]
    ref, bad = results[0], []
    for i, r in enumerate(results):
        for f in COMPARED:
            if r[f] != ref[f]:
                bad.append(f"result[{i}].{f} = {r[f]!r} != result[0].{f} = {ref[f]!r}")
    return bad


def self_test():
    base = {"move": [1, 2], "top3_moves": [[1, 2], [3, 4], [5, 6]],
            "top3_visits": [50, 30, 20]}
    out = [("identical", not disagreements([base, dict(base)]) is False)]
    out = [("identical", "ok" if not disagreements([base, dict(base)]) else "BROKEN")]
    for f, bad in [("move", [9, 9]), ("top3_moves", [[9, 9], [3, 4], [5, 6]]),
                   ("top3_visits", [51, 30, 20])]:
        rejected = bool(disagreements([base, {**base, f: bad}]))
        out.append((f, "ok" if rejected else "BROKEN"))
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: g2_probe.py <clone> all|one"); return 2
    clone, mode = sys.argv[1], sys.argv[2]
    if not os.path.isdir(clone):
        print(f"FAIL: clone not found at {clone}"); return 2

    if mode == "one":
        try:
            make_player, twixt, Point = build(clone)
            print(json.dumps(one_result(make_player, twixt, Point)))
            return 0
        except Exception:
            print("SUBPROCESS_ENGINE_ERROR\n" + traceback.format_exc(), file=sys.stderr)
            return 1
    if mode != "all":
        print(f"FAIL: unknown mode {mode}"); return 2

    print("comparator self-test (each field mismatch must be caught):")
    for label, verdict in self_test():
        print(f"  {verdict:8} {label}")
        if verdict != "ok":
            print("\nFAIL: comparator does not bind."); return 1

    results = []
    try:
        make_player, twixt, Point = build(clone)
        for i in range(N_IN_PROCESS):
            results.append(one_result(make_player, twixt, Point))
    except Exception:
        print(f"\nG2 FAIL — engine raised on in-process query {len(results) + 1}:")
        print(traceback.format_exc())
        return 1
    print(f"\n{len(results)} in-process results collected")

    for i in range(N_SUBPROCESS):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), clone, "one"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(f"\nG2 FAIL — fresh process {i} exited {p.returncode}:")
            print(p.stderr[-3000:])
            return 1
        results.append(json.loads(p.stdout.strip().splitlines()[-1]))
    print(f"{N_SUBPROCESS} fresh-process results collected; {len(results)} total")

    json.dump(results, open(os.path.join(HERE, "g2_raw_results.json"), "w"), indent=1)

    if len(results) != N_IN_PROCESS + N_SUBPROCESS:
        print(f"\nG2 FAIL — {len(results)} results, expected 25"); return 1

    bad = disagreements(results)
    if bad:
        print(f"\nG2 FAIL — {len(bad)} disagreement(s):")
        for b in bad[:20]:
            print("  " + b)
        return 1

    r = results[0]
    print(f"\nall {len(results)} identical on {COMPARED}:")
    print(f"  move          {r['move']}")
    print(f"  top3 moves    {r['top3_moves']}")
    print(f"  top3 visits   {r['top3_visits']}")
    print(f"  proven        {r['proven']}")
    print(f"\nprogress sink (recorded, NOT a pass criterion):")
    print(f"  events/query  {sorted({x['progress_events'] for x in results})}")
    print(f"  currents      {r['progress_currents']}")
    print("\nG2 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
