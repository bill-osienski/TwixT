#!/usr/bin/env python3
"""G3 PREFLIGHT. FAIL-CLOSED. No games, no seed consumption.

Card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md.
Asserts every criterion; exits non-zero on any failure.
  exit 0 = preflight passed   1 = failed   2 = cannot run
usage: g3_preflight.py <clone> <repo>
"""
import hashlib, json, os, resource, sys, time, traceback

R = {"checks": [], "failures": []}
T0 = time.time()


def check(name, fn):
    try:
        d = fn()
        R["checks"].append({"name": name, "ok": True, "detail": d})
        print(f"  PASS  {name}" + (f" — {d}" if d else ""))
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        R["checks"].append({"name": name, "ok": False, "detail": msg})
        R["failures"].append(f"{name}: {msg}")
        print(f"  FAIL  {name} — {msg}")
        if not isinstance(e, AssertionError):
            print(traceback.format_exc())
        return False


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_file(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3:
        print("usage: g3_preflight.py <clone> <repo>"); return 2
    clone, repo = sys.argv[1], sys.argv[2]
    for p in (clone, repo):
        if not os.path.isdir(p):
            print(f"FAIL: not a directory: {p}"); return 2
    sys.path.insert(0, clone); sys.path.insert(0, repo)
    os.chdir(clone)

    ok = True

    # ---- 1. COMBINED RUNTIME: TF + MLX + both pinned models in ONE process
    print("combined runtime (one process):")
    state = {}

    def load_everything():
        import tensorflow as tf
        import mlx.core as mx
        import numpy as np
        from src.backend import twixt, naf, nneval
        from src.backend.point import Point
        from src import constants as ct
        from scripts.GPU.alphazero.game.twixt_state import TwixtState
        from scripts.GPU.alphazero import twixtbot_adapter as A
        from scripts.GPU.alphazero import twixtbot_g3_schedule as S
        from scripts.GPU.alphazero import twixtbot_g3_harness as H
        state.update(tf=tf, mx=mx, np=np, twixt=twixt, naf=naf, nneval=nneval,
                     Point=Point, ct=ct, TwixtState=TwixtState, A=A, S=S, H=H)
        return f"tensorflow {tf.__version__}, mlx {mx.__version__}, numpy {np.__version__}"

    ok &= check("TensorFlow, MLX and both engines import together", load_everything)
    if not ok:
        print("\nPREFLIGHT FAILED at import"); return 1
    A, S, H = state["A"], state["S"], state["H"]
    twixt, Point, ct, naf = state["twixt"], state["Point"], state["ct"], state["naf"]
    TwixtState, np, mx = state["TwixtState"], state["np"], state["mx"]

    def anchor_eval():
        """One fixed-position evaluation through the ANCHOR engine (TensorFlow)."""
        ev = state["nneval"].NNEvaluater("model/pb")
        g = twixt.Game(allow_scl=False)
        for r, c in ((12, 12), (11, 10), (13, 9), (10, 13)):
            g.play(Point(*A.rc_to_xy(r, c)))
        pwin, ml = ev.eval_one(naf.NetInputs(g))
        pw, mlv = np.asarray(pwin), np.asarray(ml)
        assert pw.reshape(-1).size == 3, f"pwin size {pw.reshape(-1).size}"
        assert mlv.shape[-1] == 528, f"movelogits width {mlv.shape[-1]}"
        assert np.all(np.isfinite(pw)) and np.all(np.isfinite(mlv)), "non-finite anchor output"
        v = A.pwin_to_value(pwin, naf)
        assert -1.0 <= v <= 1.0
        state["anchor_value"] = v
        return f"pwin{list(pw.shape)} movelogits{list(mlv.shape)} -> value {v:+.4f}"

    def reference_eval():
        """One fixed-position evaluation through OUR engine (MLX), both checkpoints."""
        from scripts.GPU.alphazero.probe_eval import load_network_for_scoring
        st = TwixtState()
        for m in ((12, 12), (11, 10), (13, 9), (10, 13)):
            st = st.apply_move(m)
        board = mx.array(st.to_tensor()[None].transpose(0, 2, 3, 1))
        out = {}
        for name, meta in S.REFERENCE_CHECKPOINTS.items():
            path = os.path.join(repo, meta["path"])
            net, *_ = load_network_for_scoring(path)
            net.eval()
            priors, value = net.evaluate(board, st.legal_moves(), active_size=24)
            pa = np.asarray(priors)
            assert pa.size == len(st.legal_moves()), f"{name}: {pa.size} priors vs {len(st.legal_moves())} moves"
            assert np.all(np.isfinite(pa)) and np.isfinite(value), f"{name}: non-finite"
            assert -1.0 <= value <= 1.0, f"{name}: value {value}"
            out[name] = round(float(value), 4)
        state["reference_values"] = out
        return f"{len(out)} checkpoints evaluated in-process: {out}"

    ok &= check("anchor (TensorFlow) fixed-position evaluation", anchor_eval)
    ok &= check("reference (MLX) fixed-position evaluation, both checkpoints", reference_eval)

    # ---- 2. PINNED HASHES
    print("\npinned artifact hashes:")

    def anchor_hashes():
        m = {"model/pb/variables/variables.data-00000-of-00001": S.ANCHOR["weights_sha256"],
             "model/pb/saved_model.pb": S.ANCHOR["saved_model_sha256"],
             "model/pb/variables/variables.index": S.ANCHOR["variables_index_sha256"]}
        for rel, want in m.items():
            got = sha256_file(os.path.join(clone, rel))
            assert got == want, f"{rel}: {got} != pinned {want}"
        return f"{len(m)} anchor artifacts match their pins"

    def reference_hashes():
        for name, meta in S.REFERENCE_CHECKPOINTS.items():
            got = sha1_file(os.path.join(repo, meta["path"]))
            assert got == meta["sha1"], f"{name}: {got} != pinned {meta['sha1']}"
        return "both reference checkpoints match their pinned SHA-1"

    ok &= check("anchor model artifacts match pinned sha256", anchor_hashes)
    ok &= check("reference checkpoints match pinned sha1", reference_hashes)

    # ---- 3. OPENINGS: legal, bound-equal in both engines, hash-stable
    print("\nfrozen openings:")

    def openings_valid():
        rows = []
        for oid, moves in S.OPENINGS:
            our = TwixtState(max_plies_limit=S.PLY_CAP)
            tb = twixt.Game(allow_scl=False)
            d = A.state_divergences(our, tb, twixt, ply_cap=S.PLY_CAP)
            assert not d, f"{oid}: diverged before any move: {d}"
            for i, (r, c) in enumerate(moves):
                assert (r, c) in our.legal_moves(), f"{oid} ply {i}: {(r,c)} illegal in ours"
                our = our.apply_move((r, c))
                tb.play(Point(*A.rc_to_xy(r, c)))
                d = A.state_divergences(our, tb, twixt, ply_cap=S.PLY_CAP)
                assert not d, f"{oid} ply {i} after {(r,c)}: {d}"
            assert our.ply == len(moves), f"{oid}: ply {our.ply} != {len(moves)}"
            assert our.winner() is None, f"{oid}: opening already decided"
            rows.append({"opening_id": oid, "moves": [list(m) for m in moves],
                         "sha256": S.opening_hash(moves), "ply": our.ply,
                         "legal_moves_after": len(our.legal_moves()),
                         "to_move_after": our.to_move})
        assert len({r["sha256"] for r in rows}) == len(rows), "two openings share a hash"
        R["openings"] = rows
        return f"{len(rows)} openings replay legally and bind identically at every ply"

    def openings_colour_neutral():
        """The SAME sequence serves both colour arms: no per-arm variant exists."""
        tasks = S.enumerate_tasks()
        for oid, moves in S.OPENINGS:
            used = {tuple(map(tuple, t["opening_moves"])) for t in tasks if t["opening_id"] == oid}
            assert used == {tuple(moves)}, f"{oid} used with {len(used)} different sequences"
            arms = {t["colour_arm"] for t in tasks if t["opening_id"] == oid}
            assert arms == set(S.COLOUR_ARMS), f"{oid} missing a colour arm"
        return "each opening is byte-identical across both colour arms"

    ok &= check("openings replay legally and bind in both engines", openings_valid)
    ok &= check("openings are colour-arm neutral", openings_colour_neutral)

    # ---- 4. SCHEDULE DRY RUN (identities only; no game, no seed consumed)
    print("\nschedule dry run (128 task identities, no game played):")

    def schedule_ok():
        tasks = S.enumerate_tasks()
        bad = S.schedule_invariants(tasks)
        assert not bad, "; ".join(bad)
        R["tasks"] = tasks
        return f"{len(tasks)} tasks, 0 invariant violations"

    def seed_formula():
        tasks = S.enumerate_tasks()
        for t_i, tr in enumerate(S.TRIALS_LADDER):
            for r_i, rf in enumerate(S.REFERENCES):
                for o_i, (oid, _) in enumerate(S.OPENINGS):
                    for c_i, arm in enumerate(S.COLOUR_ARMS):
                        want = 202611000 + (((t_i * 2 + r_i) * 8 + o_i) * 2 + c_i)
                        got = [t for t in tasks if t["trials"] == tr and t["reference"] == rf
                               and t["opening_id"] == oid and t["colour_arm"] == arm]
                        assert len(got) == 1, f"{tr}/{rf}/{oid}/{arm}: {len(got)} tasks"
                        assert got[0]["seed"] == want, f"seed {got[0]['seed']} != {want}"
        seeds = sorted(t["seed"] for t in tasks)
        assert seeds == list(range(202611000, 202611128)), "seeds are not exactly the first 128"
        lo, hi = S.RESERVED_SEEDS
        unused = (hi - lo) - len(seeds)
        return f"128 seeds = [202611000, 202611128) exactly; {unused} reserved seeds remain unused"

    def counts():
        tasks = S.enumerate_tasks()
        per_trials = {tr: len([t for t in tasks if t["trials"] == tr]) for tr in S.TRIALS_LADDER}
        assert set(per_trials.values()) == {32}, per_trials
        pr = {(tr, rf): len([t for t in tasks if t["trials"] == tr and t["reference"] == rf])
              for tr in S.TRIALS_LADDER for rf in S.REFERENCES}
        assert set(pr.values()) == {16}, pr
        per_arm = {a: len([t for t in tasks if t["colour_arm"] == a]) for a in S.COLOUR_ARMS}
        assert set(per_arm.values()) == {64}, per_arm
        return "32 per trials, 16 per trials/reference, 2 per opening, 64 per colour arm"

    ok &= check("schedule invariants hold", schedule_ok)
    ok &= check("seed formula matches the frozen expression", seed_formula)
    ok &= check("task counts are balanced", counts)

    # ---- 5. HARNESS STRUCTURE (no game played)
    print("\nharness structure:")

    def harness_binds_cap():
        import inspect
        src = inspect.getsource(H.play_game)
        assert "ply_cap=ply_cap" in src, "play_game does not pass ply_cap explicitly"
        assert "is_terminal" in src, "play_game does not use our terminality"
        p = inspect.signature(A.state_divergences).parameters["ply_cap"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
        return "play_game passes ply_cap explicitly; the parameter is keyword-only with no default"

    def harness_aborts_fail_closed():
        """Abort behaviour, tested BEHAVIOURALLY.

        An earlier version grepped the harness source for "retry" and failed on
        the docstring sentence that FORBIDS retrying -- an assertion reading its
        own documentation. Replaced with observation: drive play_game with fake
        engines that fail, and count how many times the failing seam is called.

        This plays no G3 game: the task is synthetic (task_index/seed -1, never
        one of the 128), both engines are fakes, no model is loaded and no
        reserved seed is touched. It aborts at the first agent move.
        """
        assert set(H.ABORT_REASONS) >= {
            "state_divergence", "illegal_move", "resignation_or_swap",
            "engine_exception", "malformed_output"}, H.ABORT_REASONS
        try:
            H.HarnessAbort("not_a_reason", "x", 0)
        except ValueError:
            pass
        else:
            raise AssertionError("HarnessAbort accepted an unknown reason")

        our0 = TwixtState(max_plies_limit=S.PLY_CAP)
        for bad, want in ((lambda: {"nope": 1}, "malformed_output"),
                          (lambda: (_ for _ in ()).throw(RuntimeError("boom")), "engine_exception")):
            class P:
                def pick_move(self, g, window=None):
                    return bad()
            try:
                H.anchor_move(lambda: P(), twixt.Game(allow_scl=False), our0, 0)
            except H.HarnessAbort as e:
                assert e.reason == want, f"expected {want}, got {e.reason}"
            else:
                raise AssertionError(f"{want} was not raised")

        # NO RETRY, observed: the failing seam must be entered exactly once.
        calls = {"n": 0}

        def exploding_reference(_state):
            calls["n"] += 1
            raise RuntimeError("reference exploded")

        synthetic = {
            "task_index": -1, "seed": -1, "trials": 0, "reference": "SYNTHETIC",
            "opening_id": "SYNTHETIC", "colour_arm": "anchor_red",
            "anchor_colour": "red",
            # red opens, so after one opening ply it is BLACK to move: the
            # reference seam is the first thing called, and it raises.
            "opening_moves": [[12, 12]],
        }

        class NeverCalled:
            def pick_move(self, g, window=None):
                raise AssertionError("anchor was called after the reference aborted")

        rec = H.play_game(task=synthetic, twixt=twixt, Point=Point, ct=ct,
                          TwixtState=TwixtState, player_factory=lambda: NeverCalled(),
                          reference_fn=exploding_reference, ply_cap=S.PLY_CAP)
        assert rec["aborted"] is True, f"aborted flag is {rec['aborted']}"
        assert rec["abort_reason"] == "engine_exception", rec["abort_reason"]
        assert calls["n"] == 1, f"failing seam called {calls['n']} times; expected exactly 1 (no retry)"
        assert rec["result"] is None and rec["winner"] is None, "an aborted game carries a result"
        assert rec["plies"] == 1, f"aborted at ply {rec['plies']}, expected 1"
        return (f"{len(H.ABORT_REASONS)} reasons; unknown rejected; malformed and exception both "
                f"abort; failing seam entered exactly once (no retry); aborted game keeps no result")

    def harness_preserves_full_visits():
        import inspect
        src = inspect.getsource(H.anchor_move)
        assert '"visits": [int(v) for v in resp["Y"]]' in src, "visits are not the FULL array"
        assert "[:3]" not in src and "[:2]" not in src, "a head slice is being stored"
        return "full visit array stored per move, no head slice"

    ok &= check("harness passes ply_cap explicitly", harness_binds_cap)
    ok &= check("harness aborts fail closed and never retries", harness_aborts_fail_closed)
    ok &= check("harness preserves the FULL visit array", harness_preserves_full_visits)

    # ---- 6. no game was played, no seed consumed
    print("\nscope:")

    def no_g3_game_played():
        """State the scope ACCURATELY.

        play_game IS invoked once above, with fake engines on a synthetic task
        (task_index/seed -1), aborting at ply 1 to observe the no-retry
        behaviour. That is a harness unit test, not a G3 game: no model plays, no
        scheduled task runs, no reserved seed is used and no result is produced.
        An earlier version of this check claimed "play_game was never called",
        which stopped being true the moment the abort test was added.
        """
        tasks = S.enumerate_tasks()
        scheduled = {t["seed"] for t in tasks}
        assert scheduled == set(range(202611000, 202611128)), "scheduled seeds moved"
        # nothing in this preflight produced a game result against a scheduled task
        assert "games" not in R and "results" not in R, "a game/result record exists"
        for c in R["checks"]:
            assert "result" not in str(c.get("detail", "")).lower() or "no result" in str(c["detail"]).lower(), \
                f"a check reported a game result: {c['name']}"
        lo, hi = S.RESERVED_SEEDS
        return (f"no scheduled task was run and no reserved seed consumed; the {hi-lo} seeds in "
                f"[{lo}, {hi}) are untouched. play_game was invoked ONCE on a synthetic task "
                f"(seed -1) with fake engines to observe abort behaviour.")

    ok &= check("no G3 game played, no reserved seed consumed", no_g3_game_played)

    R["timing_seconds"] = round(time.time() - T0, 2)
    R["max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)
    R["passed"] = bool(ok)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "g3_preflight_results.json"), "w") as f:
        json.dump(R, f, indent=1)

    n = sum(1 for c in R["checks"] if c["ok"])
    print(f"\n{n}/{len(R['checks'])} checks passed")
    print(f"elapsed {R['timing_seconds']}s, max RSS {R['max_rss_mib']} MiB")
    if not ok:
        print("\nG3 PREFLIGHT FAILED")
        for f_ in R["failures"]:
            print("  " + f_)
        return 1
    print("\nG3 PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("UNHANDLED\n" + traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
