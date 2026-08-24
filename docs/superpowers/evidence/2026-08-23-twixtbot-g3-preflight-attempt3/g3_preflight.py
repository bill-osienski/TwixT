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
                        want = S.SEED_BASE + (((t_i * 2 + r_i) * 8 + o_i) * 2 + c_i)
                        got = [t for t in tasks if t["trials"] == tr and t["reference"] == rf
                               and t["opening_id"] == oid and t["colour_arm"] == arm]
                        assert len(got) == 1, f"{tr}/{rf}/{oid}/{arm}: {len(got)} tasks"
                        assert got[0]["seed"] == want, f"seed {got[0]['seed']} != {want}"
        seeds = sorted(t["seed"] for t in tasks)
        assert seeds == list(range(*S.SCHEDULE_SEEDS)), \
            f"seeds are not exactly {S.SCHEDULE_SEEDS}"
        assert not (set(seeds) & set(S.CONSUMED_SEEDS)), "schedule reuses a consumed seed"
        lo, hi = S.RESERVED_SEEDS
        unused = (hi - lo) - len(seeds) - len(S.CONSUMED_SEEDS)
        return (f"128 seeds = [{S.SCHEDULE_SEEDS[0]}, {S.SCHEDULE_SEEDS[1]}) exactly, disjoint from "
                f"{len(S.CONSUMED_SEEDS)} consumed; {unused} reserved seeds remain unused")

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

    # ---- 5. EXECUTION PATHS (real engines, one move each; no game, no seed)
    print("\nexecution paths:")
    from scripts.GPU.alphazero import twixtbot_g3_reference as RF
    from src.backend import nnmplayer

    def trials0_one_move():
        """trials=0 is the RAW-POLICY path and returns NO Y array.

        Requiring Y there aborted all 32 trials=0 tasks on the anchor's first
        move. One real anchor move at trials=0, through the harness.
        """
        our = TwixtState(max_plies_limit=S.PLY_CAP)
        tb = twixt.Game(allow_scl=False)
        for r, c in S.OPENINGS[0][1]:
            our = our.apply_move((r, c)); tb.play(Point(*A.rc_to_xy(r, c)))
        pl = nnmplayer.Player(**H.anchor_player_kwargs(0, ct))
        rc, rec = H.anchor_move(pl, tb, our, our.ply, 0)
        assert rc in our.legal_moves(), f"{rc} illegal"
        assert rec["visits"] is None and rec["visits_available"] is False, "visits claimed at trials=0"
        assert rec["policy"] is not None and len(rec["policy"]) == 528, \
            f"policy array is {None if rec['policy'] is None else len(rec['policy'])}, expected 528"
        assert len(rec["moves_order"]) == 528, "move order truncated"
        return f"trials=0 move {rc}; visits unavailable, full 528 policy array preserved"

    def trials_positive_one_move():
        """trials>0 must still yield a complete visit array."""
        our = TwixtState(max_plies_limit=S.PLY_CAP)
        tb = twixt.Game(allow_scl=False)
        for r, c in S.OPENINGS[0][1]:
            our = our.apply_move((r, c)); tb.play(Point(*A.rc_to_xy(r, c)))
        pl = nnmplayer.Player(**H.anchor_player_kwargs(100, ct))
        rc, rec = H.anchor_move(pl, tb, our, our.ply, 100)
        assert rec["visits_available"] is True and rec["visits"] is not None
        assert len(rec["visits"]) == 528, f"visit array {len(rec['visits'])}, expected 528"
        assert sum(rec["visits"]) > 0, "no visits recorded"
        assert rec["policy"] is None, "policy stored on a search path"
        state["visit_concentration_probe"] = {
            "trials": 100, "top3": sorted(rec["visits"], reverse=True)[:3],
            "nonzero": sum(1 for v in rec["visits"] if v),
        }
        return (f"trials=100 move {rc}; full 528 visit array, "
                f"{state['visit_concentration_probe']['nonzero']} non-zero")

    def one_player_per_game():
        """ONE Player must serve the whole game, so root reuse is preserved."""
        import inspect
        src = inspect.getsource(H.play_game)
        assert src.count("player_factory()") == 1, "player_factory called more than once per game"
        assert "anchor_move(player," in src, "play_game does not pass the single Player instance"
        assert "player_factory" not in inspect.getsource(H.anchor_move), \
            "anchor_move still constructs its own Player"
        # behaviourally: two moves from ONE player, the second sees a warm root
        our = TwixtState(max_plies_limit=S.PLY_CAP)
        tb = twixt.Game(allow_scl=False)
        for r, c in S.OPENINGS[0][1]:
            our = our.apply_move((r, c)); tb.play(Point(*A.rc_to_xy(r, c)))
        pl = nnmplayer.Player(**H.anchor_player_kwargs(100, ct))
        rc1, _ = H.anchor_move(pl, tb, our, our.ply, 100)
        our = our.apply_move(rc1); tb.play(Point(*A.rc_to_xy(*rc1)))
        assert pl.nm.root is not None, "root was not retained on the Player after a move"
        rc2, _ = H.anchor_move(pl, tb, our, our.ply, 100)
        assert rc2 in our.legal_moves()
        return f"one Player served two moves ({rc1}, {rc2}); NeuralMCTS root retained between them"

    # SYNTHETIC seed, deliberately OUTSIDE the reserved interval. Attempt 2 used
    # task 0's scheduled seed for a real 400-sim move and burnt 202611000; no
    # preflight may touch a reserved seed again.
    SYNTHETIC_SEED = 999000001

    def synthetic_task(reference="0379", anchor_colour="red", trials=0):
        return {"task_index": -1, "seed": SYNTHETIC_SEED, "trials": trials,
                "reference": reference,
                "reference_sha1": S.REFERENCE_CHECKPOINTS[reference]["sha1"],
                "opening_id": "SYNTHETIC", "opening_moves": [list(m) for m in S.OPENINGS[0][1]],
                "colour_arm": "anchor_red" if anchor_colour == "red" else "anchor_black",
                "anchor_colour": anchor_colour, "ply_cap": S.PLY_CAP, "workers": 1}

    def synthetic_seed_is_outside_reservation():
        lo, hi = S.RESERVED_SEEDS
        assert not (lo <= SYNTHETIC_SEED < hi), "the preflight seed is inside the reservation"
        assert SYNTHETIC_SEED not in {t["seed"] for t in S.enumerate_tasks()}
        return f"preflight seed {SYNTHETIC_SEED} is outside [{lo}, {hi}) and unscheduled"

    def reference_agent_binds_everything():
        """The reference must bind seed, checkpoint identity, colour and config."""
        cfg = RF.eval_config()
        assert (cfg.mcts_sims, cfg.mcts_eval_batch_size, cfg.mcts_stall_flush_sims) == (400, 14, 48)
        assert cfg.selection_mode == "opening_temperature" and cfg.max_moves == 280
        task = synthetic_task(reference="0379", anchor_colour="red")
        ev = RF.load_reference_evaluator("0379", repo)      # verifies sha1, compile=True
        assert getattr(ev, "_use_compile", None) is True, "evaluator is not compiled"
        ag = RF.build_reference_agent(task=task, evaluator=ev, colour="black")
        assert ag.seed == SYNTHETIC_SEED, f"seed {ag.seed}"
        import random
        assert ag.mcts.rng.getstate() == random.Random(SYNTHETIC_SEED ^ 0x5A5A5A).getstate(), \
            "search RNG is not the eval_runner black stream for this seed"
        assert ag.readout_rng.getstate() == random.Random(SYNTHETIC_SEED ^ 0x3C3C3C).getstate(), \
            "readout RNG is not the eval_runner black stream for this seed"
        st = TwixtState(max_plies_limit=S.PLY_CAP)
        try:
            ag(st)                       # red to move, agent is black
        except RF.ReferenceError:
            pass
        else:
            raise AssertionError("agent played out of turn")
        state["reference_bound"] = {"seed": ag.seed, "sims": ag.mcts.config.n_simulations}
        return ("synthetic seed bound; both RNG streams match eval_runner's masks; "
                "compile=True; 400/14/48/opening_temperature; out-of-turn refused")

    def negative_bindings():
        ev0379 = RF.load_reference_evaluator("0379", repo)
        # 1. wrong checkpoint for the task
        t = synthetic_task(reference="calib020_0001", anchor_colour="red")
        try:
            RF.build_reference_agent(task=t, evaluator=ev0379, colour="black")
        except RF.ReferenceError:
            pass
        else:
            raise AssertionError("calib020 task accepted the 0379 evaluator")
        # 2. untagged evaluator (not built by the loader)
        class Bare: pass
        try:
            RF.build_reference_agent(task=synthetic_task(), evaluator=Bare(), colour="black")
        except RF.ReferenceError:
            pass
        else:
            raise AssertionError("an untagged evaluator was accepted")
        # 3. non-frozen config
        from scripts.GPU.alphazero.eval_runner import EvalConfig
        try:
            RF.build_reference_agent(task=synthetic_task(), evaluator=ev0379, colour="black",
                                     config=EvalConfig(mcts_sims=200))
        except RF.ReferenceError:
            pass
        else:
            raise AssertionError("a non-frozen EvalConfig was accepted")
        # 4. colour contradicting anchor_colour
        try:
            RF.build_reference_agent(task=synthetic_task(anchor_colour="red"),
                                     evaluator=ev0379, colour="red")
        except RF.ReferenceError:
            pass
        else:
            raise AssertionError("reference colour contradicting the anchor was accepted")
        # 5. an already-consumed seed
        t5 = synthetic_task(); t5["seed"] = S.CONSUMED_SEEDS[0]
        try:
            RF.build_reference_agent(task=t5, evaluator=ev0379, colour="black")
        except RF.ReferenceError:
            pass
        else:
            raise AssertionError("a consumed seed was accepted")
        # 6. anchor settings: wrong trials must be refused
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        kw = H.anchor_player_kwargs(100, ct)
        try:
            RUN.assert_anchor_settings(kw, synthetic_task(trials=400), ct)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("anchor trials mismatch was accepted")
        bad = dict(kw); bad["temperature"] = 1.0
        try:
            RUN.assert_anchor_settings(bad, synthetic_task(trials=100), ct)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("anchor temperature drift was accepted")
        return "6 negative bindings all refused (checkpoint, untagged, config, colour, consumed seed, anchor settings)"

    def reference_one_move():
        """One real reference move at 400 sims, on a SYNTHETIC seed."""
        task = synthetic_task(reference="0379", anchor_colour="red")
        ev = RF.load_reference_evaluator("0379", repo)
        ag = RF.build_reference_agent(task=task, evaluator=ev, colour="black")
        our = TwixtState(max_plies_limit=S.PLY_CAP)
        our = our.apply_move(S.OPENINGS[0][1][0])          # red opens -> black to move
        t0 = time.time()
        rc, rec = H.reference_move(ag, our, our.ply)
        dt = time.time() - t0
        assert rc in our.legal_moves(), f"{rc} illegal"
        assert ag.moves_made == 1
        state["reference_move_seconds"] = round(dt, 2)
        return f"reference chose {rc} in {dt:.2f}s at 400 sims"

    def unexpected_exception_is_structured():
        """A failure in a STATE-UPDATE seam must still yield an aborted record."""
        class ExplodingState(TwixtState):
            def apply_move(self, move):
                raise RuntimeError("apply_move exploded")
        synthetic = {"task_index": -1, "seed": -1, "trials": 0, "reference": "SYNTHETIC",
                     "opening_id": "SYNTHETIC", "colour_arm": "anchor_red",
                     "anchor_colour": "red", "opening_moves": [[12, 12]]}

        class Ref:
            seed = -1
            def __call__(self, s): raise AssertionError("not reached")

        rec = H.play_game(task=synthetic, twixt=twixt, Point=Point, ct=ct,
                          TwixtState=ExplodingState, player_factory=lambda: None,
                          reference_agent=Ref(), ply_cap=S.PLY_CAP)
        assert rec["aborted"] is True, "state-update failure produced no aborted record"
        assert rec["abort_reason"] == "engine_exception", rec["abort_reason"]
        assert "apply_move exploded" in rec["abort_detail"], rec["abort_detail"]
        assert rec["result"] is None and rec["winner"] is None
        # and an unbound reference is refused rather than silently accepted
        class NoSeed:
            def __call__(self, s): raise AssertionError("not reached")
        rec2 = H.play_game(task=synthetic, twixt=twixt, Point=Point, ct=ct,
                           TwixtState=TwixtState, player_factory=lambda: None,
                           reference_agent=NoSeed(), ply_cap=S.PLY_CAP)
        assert rec2["aborted"] and rec2["abort_reason"] == "malformed_output", rec2["abort_reason"]
        return "state-update exception recorded as engine_exception; unbound reference refused"

    def runner_binds_and_scores():
        """The concrete runner: construction errors are structured, and scoring
        refuses an incomplete run."""
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        tasks = [synthetic_task()]
        tasks[0]["task_index"] = 0
        # a binding failure OUTSIDE play_game must still be recorded
        class BadPlayerModule:
            @staticmethod
            def Player(**kw):
                raise RuntimeError("player construction exploded")
        out = RUN.run_g3(tasks=[dict(t, seed=S.CONSUMED_SEEDS[0]) for t in tasks],
                         twixt=twixt, Point=Point, ct=ct, TwixtState=TwixtState,
                         nnmplayer=BadPlayerModule, repo_root=repo,
                         play_game=lambda **kw: {"aborted": False},
                         cleanup=lambda: None)
        assert out["completed"] is False, "a consumed seed was allowed to run"
        assert out["results"][0]["aborted"] and out["results"][0]["abort_reason"] == "engine_exception"
        assert "consumed" in out["results"][0]["abort_detail"], out["results"][0]["abort_detail"]
        assert out["summary"] is None, "an incomplete run produced a summary"
        # summarise refuses a set containing an abort
        try:
            RUN.summarise([{"trials": 0, "reference": "0379", "result": "anchor", "aborted": True}])
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("summarise scored a run containing an abort")
        # and computes the two rates + lowest passing trials on a clean set
        synth = []
        for tr, (a, b) in ((0, (1.0, 1.0)), (100, (0.5, 0.5)), (400, (0.5, 0.5))):
            for ref, rate in (("0379", a), ("calib020_0001", b)):
                for i in range(4):
                    synth.append({"trials": tr, "reference": ref, "aborted": False,
                                  "result": "anchor" if i < rate * 4 else "reference"})
        sm = RUN.summarise(synth)
        assert sm["per_trials"][0]["non_saturated"] is False, "1.0 counted as unsaturated"
        assert sm["per_trials"][100]["non_saturated"] is True
        assert sm["selected_trials"] == 100, f"selected {sm['selected_trials']}, expected the lowest passing"
        assert "descriptive" in sm["per_trials"][100]["ordering_note"]
        assert "not a gate" in sm["pass_condition"]
        return ("binding failure recorded as a structured abort; no summary on an incomplete run; "
                f"two rates per setting; lowest passing trials = {sm['selected_trials']}")

    def runner_stops_on_first_abort():
        tasks = S.enumerate_tasks()[:5]
        seen = []
        def play(t):
            seen.append(t["task_index"])
            return {"task_index": t["task_index"], "aborted": t["task_index"] == 2,
                    "abort_reason": "state_divergence" if t["task_index"] == 2 else None}
        out = H.run_schedule(tasks, play)
        assert out["completed"] is False and out["stopped_at_task_index"] == 2, out
        assert out["n_played"] == 3 and out["n_remaining"] == 2, out
        assert seen == [0, 1, 2], f"runner continued past the abort: {seen}"
        clean = H.run_schedule(tasks[:2], lambda t: {"task_index": t["task_index"], "aborted": False})
        assert clean["completed"] is True and clean["n_remaining"] == 0
        return "stops at the first aborted task, leaving the rest unplayed"

    ok &= check("trials=0 raw-policy path yields a move", trials0_one_move)
    ok &= check("trials>0 yields a complete visit array", trials_positive_one_move)
    ok &= check("one Player per GAME, root retained", one_player_per_game)
    ok &= check("preflight seed is outside the reservation", synthetic_seed_is_outside_reservation)
    ok &= check("reference agent binds seed/checkpoint/settings", reference_agent_binds_everything)
    ok &= check("negative bindings are all refused", negative_bindings)
    ok &= check("reference agent plays one real move", reference_one_move)
    ok &= check("unexpected exceptions are structured aborts", unexpected_exception_is_structured)
    ok &= check("concrete runner binds, aborts structurally, and scores", runner_binds_and_scores)
    ok &= check("runner stops on the first aborted task", runner_stops_on_first_abort)

    def harness_preserves_full_visits():
        import inspect
        src = inspect.getsource(H.anchor_move)
        assert '[int(v) for v in resp["Y"]]' in src, "visits are not the FULL array"
        assert "[:3]" not in src and "[:2]" not in src, "a head slice is being stored"
        return "full visit array stored per move, no head slice"

    def harness_binds_cap():
        import inspect
        src = inspect.getsource(H.play_game)
        assert "ply_cap=ply_cap" in src, "play_game does not pass ply_cap explicitly"
        p = inspect.signature(A.state_divergences).parameters["ply_cap"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
        return "play_game passes ply_cap explicitly; parameter keyword-only with no default"

    print("\nharness contract:")
    ok &= check("harness passes ply_cap explicitly", harness_binds_cap)
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
        assert scheduled == set(range(*S.SCHEDULE_SEEDS)), "scheduled seeds moved"
        # nothing in this preflight produced a game result against a scheduled task
        assert "games" not in R and "results" not in R, "a game/result record exists"
        for c in R["checks"]:
            assert "result" not in str(c.get("detail", "")).lower() or "no result" in str(c["detail"]).lower(), \
                f"a check reported a game result: {c['name']}"
        lo, hi = S.RESERVED_SEEDS
        return (f"no scheduled task was run and no reserved seed consumed; the {hi-lo} seeds in "
                f"[{lo}, {hi}) are untouched BY THIS ATTEMPT. Every real move here used the "
                f"synthetic seed {SYNTHETIC_SEED}, outside the reservation. "
                f"202611000 remains recorded as consumed by attempt 2, and the "
                f"schedule was re-frozen onto [202611128, 202611256) to avoid it.")

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
