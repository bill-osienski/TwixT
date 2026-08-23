#!/usr/bin/env python3
"""Pre-G3 qualification of the twixtbot adapter. FAIL-CLOSED.

Card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md.
Asserts every criterion; exits non-zero on ANY divergence or engine exception.
No complete games, no G3, no calibration, no reserved seeds.

usage: qualify_adapter.py <clone> <repo>
  exit 0 = qualified   exit 1 = divergence/failure   exit 2 = cannot run
"""
import json, os, sys, traceback

TRIALS = 100
RESULTS = {"checks": [], "failures": []}


def check(name, fn):
    """Every check reports pass/fail; a raised exception is a failure, not a crash."""
    try:
        detail = fn()
        RESULTS["checks"].append({"name": name, "ok": True, "detail": detail})
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        RESULTS["checks"].append({"name": name, "ok": False, "detail": msg})
        RESULTS["failures"].append(f"{name}: {msg}")
        print(f"  FAIL  {name} — {msg}")
        if not isinstance(e, AssertionError):
            print(traceback.format_exc())
        return False


def main():
    if len(sys.argv) != 3:
        print("usage: qualify_adapter.py <clone> <repo>"); return 2
    clone, repo = sys.argv[1], sys.argv[2]
    for p in (clone, repo):
        if not os.path.isdir(p):
            print(f"FAIL: not a directory: {p}"); return 2

    sys.path.insert(0, clone)
    sys.path.insert(0, repo)
    os.chdir(clone)                       # NNEvaluater joins os.getcwd()

    from src.backend import twixt, naf, nnmplayer
    from src.backend.point import Point
    from src import constants as ct
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero import twixtbot_adapter as A

    B = A.BOARD_SIZE

    # ---- 1. index mapping: round trips, exclusions, padded cells, both colours
    def mapping_roundtrip():
        n = 0
        for colour in (A.RED, A.BLACK):
            seen = set()
            for i in range(A.POLICY_N):
                r, c = A.policy_index_to_rc(colour, i)
                assert A.is_playable(colour, r, c), f"{colour} index {i} -> unplayable {(r,c)}"
                assert A.policy_index(colour, r, c) == i, f"{colour} round trip broke at {i}"
                seen.add((r, c))
                n += 1
            assert len(seen) == A.POLICY_N, f"{colour}: {len(seen)} distinct cells, expected 528"
        return f"{n} index round trips over both colours"

    def mapping_matches_engine():
        """Our mapping must equal twixtbot's OWN naf.policy_index_point."""
        for colour, tb in ((A.RED, A.TB_WHITE), (A.BLACK, A.TB_BLACK)):
            for i in range(A.POLICY_N):
                p = naf.policy_index_point(tb, i)
                assert A.policy_index_to_rc(colour, i) == A.xy_to_rc(p.x, p.y), \
                    f"{colour} index {i}: ours={A.policy_index_to_rc(colour,i)} engine={(p.y,p.x)}"
        return "528 x 2 agree with naf.policy_index_point"

    def exclusions_and_padding():
        for colour in (A.RED, A.BLACK):
            ex = A.excluded_cells(colour)
            assert len(ex) == A.PADDED_N - A.POLICY_N == 48, f"{colour}: {len(ex)} excluded, expected 48"
            for r, c in ex:
                try:
                    A.policy_index(colour, r, c)
                except A.AdapterError:
                    pass
                else:
                    raise AssertionError(f"{colour}: excluded {(r,c)} got a policy index")
        # padded index is total and bijective over the whole board
        idx = {A.padded_index(r, c) for r in range(B) for c in range(B)}
        assert idx == set(range(A.PADDED_N)), "padded index is not a bijection over 576"
        for i in range(A.PADDED_N):
            r, c = A.padded_index_to_rc(i)
            assert A.padded_index(r, c) == i
        return "48 excluded per colour; 576 padded bijection"

    def exclusions_match_our_engine():
        """Excluded cells must be exactly those our engine refuses."""
        for colour in (A.RED, A.BLACK):
            st = TwixtState(to_move=colour)
            ours = {(r, c) for r in range(B) for c in range(B) if st.is_valid_placement(r, c)}
            mine = {(r, c) for r in range(B) for c in range(B) if A.is_playable(colour, r, c)}
            assert ours == mine, f"{colour}: differs on {sorted(ours ^ mine)[:6]}"
        return "playable sets equal our is_valid_placement for both colours"

    ok = True
    print("index mapping:")
    for nm, fn in [("round trips", mapping_roundtrip),
                   ("agrees with naf.policy_index_point", mapping_matches_engine),
                   ("edge exclusions and padded cells", exclusions_and_padding),
                   ("exclusions match our engine", exclusions_match_our_engine)]:
        ok &= check(nm, fn)

    # ---- 2. per-ply state equivalence over scripted positions
    SCRIPTS = {
        "opening, no bridges": [(12, 12), (11, 10), (13, 9), (10, 13)],
        "automatic bridges": [(12, 12), (5, 5), (10, 11), (7, 6), (11, 13), (9, 7)],
        "boundary moves (red row 0/23, black col 0/23)":
            [(0, 5), (5, 0), (23, 6), (6, 23), (1, 6), (7, 1)],
        "dense centre, crossing pressure":
            [(10, 10), (10, 12), (11, 12), (11, 8), (12, 11), (9, 11),
             (12, 9), (13, 11), (9, 9), (12, 13)],
    }

    def make_ply_check(label, moves):
        def run():
            our = TwixtState()
            tb = twixt.Game(allow_scl=False)
            d = A.state_divergences(our, tb, twixt)
            assert not d, f"divergence before any move: {d}"
            for i, (r, c) in enumerate(moves):
                assert (r, c) in our.legal_moves(), f"ply {i}: {(r,c)} illegal in ours"
                our = our.apply_move((r, c))
                x, y = A.rc_to_xy(r, c)
                tb.play(Point(x, y))
                d = A.state_divergences(our, tb, twixt)
                assert not d, f"ply {i} after {(r,c)}: {d}"
            return f"{len(moves)} plies, all fields equal at every ply"
        return run

    print("\nper-ply state equivalence:")
    for label, moves in SCRIPTS.items():
        ok &= check(label, make_ply_check(label, moves))

    # ---- 3. blocked crossing: a bridge our engine refuses must be refused by both
    def blocked_crossing():
        our = TwixtState()
        tb = twixt.Game(allow_scl=False)
        seq = [(10, 10), (11, 12), (12, 11), (9, 11)]
        for r, c in seq:
            our = our.apply_move((r, c))
            tb.play(Point(*A.rc_to_xy(r, c)))
            d = A.state_divergences(our, tb, twixt)
            assert not d, f"after {(r,c)}: {d}"
        nb = sum(len(v) for v in A.tb_bridges(tb, twixt).values())
        assert nb == len(our.bridges), f"bridge totals differ: ours={len(our.bridges)} theirs={nb}"
        return f"{len(our.bridges)} bridges agree under crossing pressure"

    def terminal_state():
        """A red knight-ladder from row 0 to row 23. Both engines must agree.

        Consecutive red pegs differ by (2,+-1), a knight move, so each placement
        bridges to the previous one automatically. Black plays far away at col 2
        with (2,0) gaps, which form no bridges and cross nothing.
        """
        red = [(2 * i, 10 if i % 2 == 0 else 11) for i in range(12)] + [(23, 13)]
        # rows 1..12 at col 2: legal for black (row not in {0,23}), 12 fillers for
        # red's 13 moves, mutually non-knight so they form no bridges, and far
        # from red's col 10-13 corridor. An earlier version ran to row 23, which
        # black may never play -- the assert caught it rather than skipping.
        black = [(r, 2) for r in range(1, 13)]
        seq = []
        for i, m in enumerate(red):
            seq.append(m)
            if i < len(black):
                seq.append(black[i])

        our = TwixtState()
        tb = twixt.Game(allow_scl=False)
        for i, (rr, cc) in enumerate(seq):
            if our.is_terminal():
                break
            assert (rr, cc) in our.legal_moves(), f"ply {i}: {(rr,cc)} illegal in ours"
            our = our.apply_move((rr, cc))
            tb.play(Point(*A.rc_to_xy(rr, cc)))
            d = A.state_divergences(our, tb, twixt)
            assert not d, f"ply {i} after {(rr,cc)}: {d}"
            if our.winner():
                break
        w = our.winner()
        assert w == A.RED, f"scripted ladder produced winner={w}, expected red"
        assert A.tb_winner(tb) == w, f"winner disagrees: ours={w} theirs={A.tb_winner(tb)}"
        assert our.is_terminal(), "our engine does not call the win terminal"
        return f"winner={w}, agreed by both engines, terminal"

    print("\nbridges, crossings and terminal:")
    ok &= check("blocked crossing agrees", blocked_crossing)
    ok &= check("terminal state and winner agree", terminal_state)

    # ---- 4. end-to-end move smoke (ONE fixed position, no game played)
    def end_to_end():
        moves = [(12, 12), (11, 10), (13, 9), (10, 13)]
        our, tb = A.build_pair(moves, twixt, Point, TwixtState)
        assert not A.state_divergences(our, tb, twixt), "pair diverged before the query"
        kw = dict(model="model/pb", trials=TRIALS, temperature=0, add_noise=0,
                  rotation=ct.ROT_OFF, allow_swap=0)
        sink = A.ProgressSink()
        resp = nnmplayer.Player(**kw).pick_move(tb, window=sink)
        rc = A.move_from_response(resp, our)          # validates legality in OUR engine
        our2 = our.apply_move(rc)
        tb.play(Point(*A.rc_to_xy(*rc)))
        d = A.state_divergences(our2, tb, twixt)
        assert not d, f"engines diverged after applying twixtbot's move: {d}"
        # three-class head -> probability, via twixtbot's OWN naf.three_to_one
        ev = nnmplayer.Player(**kw).evaluator
        raw_pwin, _ = ev.eval_one(naf.NetInputs(tb))
        import numpy as _np
        flat = _np.asarray(raw_pwin).reshape(-1)
        assert flat.size == 3, f"pwin has {flat.size} components, expected 3"
        value = A.pwin_to_value(raw_pwin, naf)
        assert -1.0 <= value <= 1.0, f"three_to_one gave {value}, outside [-1,1]"
        # The card requires the ENGINE's conversion, not our own: assert we
        # delegate to naf.three_to_one rather than reimplementing it.
        assert value == float(_np.asarray(naf.three_to_one(flat)).reshape(-1)[0]), \
            "pwin_to_value does not match naf.three_to_one"
        assert value not in [float(x) for x in flat], \
            "converted value equals a raw logit; conversion looks like a no-op"
        RESULTS["end_to_end"] = {
            "chosen_move_rc": list(rc),
            "legal_in_ours": True,
            "progress_events": sink.events,
            "n_progress_events": len(sink.events),
            "top3_visits": [int(v) for v in resp["Y"][:3]],
        }
        return (f"move {rc} legal and states identical after; "
                f"{len(sink.events)} progress events preserved in full, "
                f"pwin(3-class)->value {value:+.4f}")

    print("\nend-to-end move smoke (one fixed position, no game played):")
    ok &= check("twixtbot move is legal and leaves states identical", end_to_end)

    # ---- 5. the checker must bind
    def checker_binds():
        our, tb = A.build_pair([(12, 12), (11, 10)], twixt, Point, TwixtState)
        assert not A.state_divergences(our, tb, twixt), "premise: pair agrees"
        tb.play(Point(*A.rc_to_xy(4, 4)))            # advance ONLY twixtbot
        d = A.state_divergences(our, tb, twixt)
        assert d, "an extra twixtbot peg was NOT detected — the checker does not bind"
        fields = " ".join(d)
        for expect in ("pegs[", "side to move", "legal moves"):
            assert expect in fields, f"divergence report never mentions {expect}"
        return f"{len(d)} divergences detected on a deliberately desynced pair"

    print("\nnegative control:")
    ok &= check("state checker detects a real divergence", checker_binds)

    RESULTS["qualified"] = bool(ok)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "qualification_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)

    n_ok = sum(1 for c in RESULTS["checks"] if c["ok"])
    print(f"\n{n_ok}/{len(RESULTS['checks'])} checks passed")
    if not ok:
        print("\nADAPTER QUALIFICATION FAILED")
        for f_ in RESULTS["failures"]:
            print("  " + f_)
        return 1
    print("\nADAPTER QUALIFICATION PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("UNHANDLED ERROR\n" + traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
