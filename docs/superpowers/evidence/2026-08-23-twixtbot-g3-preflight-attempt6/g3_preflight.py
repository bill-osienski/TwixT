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
    RUNNER_EXIT = {'GATE_PASSED': 0, 'GATE_FAILED': 1, 'RUN_ABORTED': 2, 'PRECONDITION': 3}

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
    #: reserved but never scheduled; an RNG is constructed from it and never drawn
    RNG_ONLY_SEED = 202611300

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

    def production_entry_has_no_switches():
        """run_g3 must take ONLY paths: no engines, no seams, no task subset."""
        import inspect, tempfile
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        params = set(inspect.signature(RUN.run_g3).parameters)
        assert params == {"clone_root", "repo_root", "results_path"}, params
        for forbidden in ("tasks", "play_game", "cleanup", "twixt", "Point", "ct",
                          "TwixtState", "nnmplayer", "require_canonical"):
            assert forbidden not in params, f"run_g3 still accepts {forbidden}"
        for bad, why in ((None, "None"), ("", "empty")):
            try:
                RUN._prepare_results_path(bad)
            except RUN.RunnerError:
                pass
            else:
                raise AssertionError(f"{why} results_path accepted")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            existing = f.name
        try:
            RUN._prepare_results_path(existing)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("an EXISTING results_path was accepted")
        finally:
            os.unlink(existing)
        return "run_g3 takes only clone_root/repo_root/results_path; persistence mandatory and new"

    def anchor_runtime_identity_is_verified():
        """The pinned clone is proven before anything runs, and fakes are refused."""
        import subprocess, tempfile
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        ident = RUN.verify_anchor_runtime(clone)
        assert ident["commit"] == S.ANCHOR["commit"], ident
        assert ident["model_artifacts"] == 3, ident
        # a non-checkout is refused
        try:
            RUN.verify_anchor_runtime(tempfile.mkdtemp())
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("a directory with no .git was accepted as the anchor clone")
        # a checkout at the WRONG commit is refused
        fake = tempfile.mkdtemp()
        subprocess.run(["git", "-C", fake, "init", "-q"], check=True)
        subprocess.run(["git", "-C", fake, "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", fake, "config", "user.name", "t"], check=True)
        open(os.path.join(fake, "x"), "w").write("x")
        subprocess.run(["git", "-C", fake, "add", "-A"], check=True)
        subprocess.run(["git", "-C", fake, "commit", "-qm", "x"], check=True)
        try:
            RUN.verify_anchor_runtime(fake)
        except RUN.RunnerError as e:
            assert "pinned commit" in str(e), str(e)
        else:
            raise AssertionError("a clone at the wrong commit was accepted")
        # module origin is checked, and cwd is set to the clone root
        rt = RUN._import_anchor_runtime(clone, repo)
        assert os.path.realpath(os.getcwd()) == os.path.realpath(clone), os.getcwd()
        for name in ("twixt", "nnmplayer", "nneval"):
            f = os.path.realpath(rt[name].__file__ if name in rt else
                                 __import__("src.backend." + name, fromlist=["x"]).__file__)
            assert f.startswith(os.path.realpath(clone) + os.sep), f"{name} from {f}"
        assert os.path.realpath(rt["TwixtState"].__module__ and
                                __import__("scripts.GPU.alphazero.game.twixt_state",
                                           fromlist=["x"]).__file__).startswith(
            os.path.realpath(repo) + os.sep), "TwixtState not from the repo"
        return (f"clone verified at {ident['commit'][:7]}, 3 model artifacts hashed; "
                f"non-checkout and wrong-commit refused; module origins and cwd bound")

    def shared_anchor_evaluator():
        """The production path reuses ONE anchor evaluator, not one per game."""
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        from src.backend import nnmplayer as NP
        shared = object()
        seen = []

        class FakeModule:
            @staticmethod
            def Player(**kw):
                seen.append(kw.get("evaluator"))
                return "player"

        # build_task_bindings requires a seed inside the reservation BY DESIGN --
        # the gate refused 999000002, correctly. Use a reserved-but-UNSCHEDULED
        # seed: inside [202611000,202611400), outside the schedule block
        # [202611128,202611256), and not in CONSUMED_SEEDS. It only ever has a
        # random.Random constructed from it; nothing is drawn, no search runs and
        # no move is produced.
        t = synthetic_task(reference="0379", anchor_colour="red")
        t["seed"] = RNG_ONLY_SEED
        ev = RF.load_reference_evaluator("0379", repo)
        cache = {"0379": ev}
        f1, _ = RUN.build_task_bindings(task=t, ct=ct, nnmplayer=FakeModule,
                                        evaluator_cache=cache, repo_root=repo,
                                        anchor_evaluator=shared)
        f1(); f1()
        assert seen == [shared, shared], "the shared anchor evaluator was not passed through"
        # and the frozen settings check still sees only frozen keys
        RUN.assert_anchor_settings(H.anchor_player_kwargs(t["trials"], ct), t, ct)
        return "one shared anchor evaluator is passed to every Player; frozen-key check unaffected"

    def gate_is_durable_and_exit_bound():
        """Verdict mapping and the terminal record (in-process unit level)."""
        import tempfile
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        assert (RUN.EXIT_GATE_PASSED, RUN.EXIT_GATE_FAILED,
                RUN.EXIT_RUN_ABORTED, RUN.EXIT_PRECONDITION) == (0, 1, 2, 3)
        for out, want in (({"completed": False}, "ABORTED"),
                          ({"completed": True, "summary": {"selected_trials": None}}, "FAILED"),
                          ({"completed": True, "summary": {"selected_trials": 100}}, "PASSED")):
            assert RUN.gate_verdict(out) == want, (out, want)
        assert RUN.exit_code_for({"completed": False}) == 2
        assert RUN.exit_code_for({"completed": True, "summary": {"selected_trials": None}}) == 1
        assert RUN.exit_code_for({"completed": True, "summary": {"selected_trials": 400}}) == 0
        path = os.path.join(tempfile.mkdtemp(), "g3.jsonl")
        RUN._write_terminal_record(path, {"completed": True, "games_played": 128,
                                          "tasks_remaining": 0,
                                          "summary": {"selected_trials": None},
                                          "anchor_identity": {"commit": "abc"}})
        rec = json.loads(open(path).read().strip())
        assert rec["record_type"] == "terminal" and rec["gate"] == "FAILED", rec
        assert rec["anchor_identity"]["commit"] == "abc", "terminal record omits the identity"
        return "PASSED/FAILED/ABORTED -> 0/1/2; terminal record carries the verdict AND the identity"

    def command_runs_as_a_module():
        """The qualified command must be REACHABLE, tested as a fresh subprocess.

        main() existed but the module had no __main__ guard, so
        `python3 -m ...twixtbot_g3_runner --help` printed nothing and exited 0.
        Calling main() in-process would never have caught that.
        """
        import subprocess, tempfile
        env = dict(os.environ, PYTHONPATH=repo)
        MOD = "scripts.GPU.alphazero.twixtbot_g3_runner"

        r = subprocess.run([sys.executable, "-m", MOD, "--help"],
                           capture_output=True, text=True, cwd=repo, env=env)
        assert r.returncode == 0, f"--help exited {r.returncode}"
        assert "--clone-root" in r.stdout and "--results-path" in r.stdout, r.stdout[:200]
        assert "usage" in r.stdout.lower(), "no usage text"

        r = subprocess.run([sys.executable, "-m", MOD], capture_output=True, text=True,
                           cwd=repo, env=env)
        assert r.returncode != 0, "missing required arguments exited 0"

        # a refused precondition must be exit 3, not a traceback
        out = os.path.join(tempfile.mkdtemp(), "g3.jsonl")
        r = subprocess.run([sys.executable, "-m", MOD, "--clone-root", "/nonexistent",
                            "--repo-root", repo, "--results-path", out],
                           capture_output=True, text=True, cwd=repo, env=env)
        assert r.returncode == RUNNER_EXIT["PRECONDITION"], f"exit {r.returncode}: {r.stderr[-300:]}"
        assert "Traceback" not in r.stderr, "a precondition failure raised instead of classifying"
        assert not os.path.exists(out), "a refused precondition still created the results file"
        return "module runs via -m: --help exits 0 with usage, missing args non-zero, bad clone exits 3"

    def unexpected_failures_are_classified():
        """An infrastructure failure must never be mistaken for a failed gate.

        PHASE BOUNDARY: everything up to and including the durable run header is
        PRECONDITION -- nothing was announced, nothing started. After that the run
        HAS started, so a failure is ABORTED. Exit 1 is reserved for a COMPLETED
        gate in which no setting passed.

        COVERAGE NOTE, stated plainly: the pre-run branch is exercised end-to-end
        as a fresh SUBPROCESS below. The post-start branch cannot be reached in a
        subprocess without playing a scheduled game, which is unauthorized, so it
        is covered at unit level via classify_failure. That is a real limit of
        this preflight, not a claim of end-to-end coverage.
        """
        import subprocess, stat, tempfile
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        env = dict(os.environ, PYTHONPATH=repo)
        MOD = "scripts.GPU.alphazero.twixtbot_g3_runner"

        # the classifier itself, both branches
        assert RUN.classify_failure(False) == ("PRECONDITION", RUNNER_EXIT["PRECONDITION"])
        assert RUN.classify_failure(True) == ("ABORTED", RUNNER_EXIT["RUN_ABORTED"])
        assert RUNNER_EXIT["GATE_FAILED"] not in (RUN.classify_failure(False)[1],
                                                  RUN.classify_failure(True)[1]), \
            "an infrastructure failure maps to the failed-gate exit code"

        # SUBPROCESS: a repo root that cannot supply our modules -> pre-run -> 3
        bad_repo = tempfile.mkdtemp()
        out1 = os.path.join(tempfile.mkdtemp(), "g3.jsonl")
        r1 = subprocess.run([sys.executable, "-m", MOD, "--clone-root", clone,
                             "--repo-root", bad_repo, "--results-path", out1],
                            capture_output=True, text=True, cwd=repo, env=env)
        assert r1.returncode == RUNNER_EXIT["PRECONDITION"], \
            f"pre-run failure exited {r1.returncode} ({r1.stdout[-200:]})"
        assert "Traceback" not in r1.stderr, "pre-run failure raised instead of classifying"

        # SUBPROCESS: an unwritable output directory is caught as a PRECONDITION
        # by the writability probe, before anything is announced.
        ro = tempfile.mkdtemp()
        out2 = os.path.join(ro, "g3.jsonl")
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        try:
            r2 = subprocess.run([sys.executable, "-m", MOD, "--clone-root", clone,
                                 "--repo-root", repo, "--results-path", out2],
                                capture_output=True, text=True, cwd=repo, env=env)
        finally:
            os.chmod(ro, stat.S_IRWXU)
        assert r2.returncode == RUNNER_EXIT["PRECONDITION"], \
            f"unwritable output exited {r2.returncode}, expected 3 ({r2.stdout[-200:]})"
        assert "not writable" in r2.stdout, r2.stdout[-200:]
        assert not os.path.exists(out2), "the unwritable path produced a file"
        assert r2.returncode != RUNNER_EXIT["GATE_FAILED"]
        return ("classifier maps started/not-started to 2/3 and never to 1; two pre-run "
                "subprocess failures exit 3 with no traceback; post-start branch unit-covered")

    def run_header_records_identity():
        """The durable evidence must say WHICH clone and model produced it."""
        import inspect
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        # The header is written by PHASE 1 (preconditions), which necessarily
        # completes before phase 2 (execute) runs any task.
        pre = inspect.getsource(RUN.preconditions)
        assert '"record_type": "run_header"' in pre, "preconditions writes no run header"
        assert "_run_tasks(" not in pre, "preconditions runs tasks"
        assert "_run_tasks(" in inspect.getsource(RUN.execute), "execute runs no tasks"
        top = inspect.getsource(RUN.run_g3)
        assert top.index("preconditions(") < top.index("execute(") or "execute(preconditions(" in top, \
            "run_g3 does not run preconditions before execute"
        # behavioural: the header lands on disk during phase 1 alone
        import tempfile
        out = os.path.join(tempfile.mkdtemp(), "g3.jsonl")
        prepared = RUN.preconditions(clone_root=clone, repo_root=repo, results_path=out)
        lines = open(out).read().strip().splitlines()
        assert len(lines) == 1, f"{len(lines)} records after phase 1, expected the header"
        hdr = json.loads(lines[0])
        assert hdr["record_type"] == "run_header", hdr
        assert hdr["anchor_commit"] == S.ANCHOR["commit"], hdr
        assert hdr["anchor_artifact_sha256"]["model/pb/saved_model.pb"] == \
            S.ANCHOR["saved_model_sha256"], hdr
        assert hdr["n_tasks"] == 128 and hdr["schedule_seed_block"] == list(S.SCHEDULE_SEEDS), hdr
        assert prepared["path"] == out
        ident = RUN.verify_anchor_runtime(clone)
        assert set(ident["artifact_sha256"]) == {
            "model/pb/variables/variables.data-00000-of-00001",
            "model/pb/saved_model.pb", "model/pb/variables/variables.index"}, ident
        assert ident["artifact_sha256"]["model/pb/saved_model.pb"] == S.ANCHOR["saved_model_sha256"]
        assert ident["artifact_sha256"]["model/pb/variables/variables.data-00000-of-00001"] \
            == S.ANCHOR["weights_sha256"]
        return ("run header written before the first task, carrying commit, clone root and all "
                "three ACTUAL artifact hashes")

    def counts_are_games_not_records():
        """A cleanup stop must not inflate the progress counts."""
        import tempfile
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        tasks = S.enumerate_tasks()[:3]
        path = os.path.join(tempfile.mkdtemp(), "g3.jsonl")
        out = RUN._run_tasks(
            tasks=tasks, twixt=twixt, Point=Point, ct=ct, TwixtState=TwixtState,
            nnmplayer=None, repo_root=repo, results_path=path,
            play_game=lambda **kw: dict(kw["task"], aborted=False, result="draw"),
            cleanup=(lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
            require_canonical=False)
        assert out["games_played"] == 1, f"games_played={out['games_played']}, expected 1"
        assert out["tasks_attempted"] == 1, f"tasks_attempted={out['tasks_attempted']}"
        assert out["tasks_remaining"] == 2, f"tasks_remaining={out['tasks_remaining']}, expected 2"
        assert "n_played" not in out and "n_remaining" not in out, "old ambiguous counters remain"
        return "one game played, two tasks remaining; the run-stop record is not counted as a game"

    def summarise_rejects_partial():
        """Only the canonical 128-identity result set may be scored."""
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        tasks = S.enumerate_tasks()

        def complete(res_for):
            return [dict(t, aborted=False, result=res_for(t)) for t in tasks]

        # the exact failure the review reproduced: one draw vs 0379 alone
        one = [dict(tasks[0], aborted=False, result="draw")]
        try:
            RUN.summarise(one)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("a single-game result set was scored")
        # missing one task
        try:
            RUN.summarise(complete(lambda t: "draw")[:-1])
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("a 127-game result set was scored")
        # duplicate identity
        dup = complete(lambda t: "draw"); dup[1] = dict(dup[0])
        try:
            RUN.summarise(dup)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("duplicate identities were scored")
        # unexpected reference / trials
        alien = complete(lambda t: "draw"); alien[0] = dict(alien[0], reference="ALIEN")
        try:
            RUN.summarise(alien)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("an unexpected reference was scored")
        # malformed result value
        mal = complete(lambda t: "draw"); mal[0] = dict(mal[0], result="победа")
        try:
            RUN.summarise(mal)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("a malformed result value was scored")
        # an aborted record anywhere
        ab = complete(lambda t: "draw"); ab[5] = dict(ab[5], aborted=True, abort_reason="x")
        try:
            RUN.summarise(ab)
        except RUN.RunnerError:
            pass
        else:
            raise AssertionError("a set containing an abort was scored")
        # and a genuinely complete set scores
        def res(t):
            if t["trials"] == 0:
                return "anchor"                       # saturated 1.0
            return "draw" if t["task_index"] % 2 else ("anchor" if t["trials"] == 100 else "reference")
        sm = RUN.summarise(complete(res))
        assert sm["per_trials"][0]["non_saturated"] is False
        assert sm["selected_trials"] in (100, 400, 1000)
        assert "not a gate" in sm["pass_condition"]
        return ("partial, 127-game, duplicate, alien-reference, malformed and aborted sets all "
                f"refused; a complete 128 set scores, lowest passing = {sm['selected_trials']}")

    def cleanup_failure_is_a_run_abort():
        """A failing cleanup must stop the run with a durable record, not raise."""
        import tempfile
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        tasks = S.enumerate_tasks()[:3]
        path = os.path.join(tempfile.mkdtemp(), "g3.jsonl")
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise RuntimeError("clear_cache exploded")

        out = RUN._run_tasks(
            tasks=tasks, twixt=twixt, Point=Point, ct=ct, TwixtState=TwixtState,
            nnmplayer=None, repo_root=repo, results_path=path,
            play_game=lambda **kw: dict(kw["task"], aborted=False, result="draw"),
            cleanup=boom, require_canonical=False,
        )
        # binding runs first and will fail on nnmplayer=None only when it constructs
        # a Player -- it does not, so the game record is produced, then cleanup blows.
        assert out["completed"] is False, "the run completed despite a cleanup failure"
        assert out["stopped_reason"] == "engine_exception", out["stopped_reason"]
        assert "cleanup" in (out["stopped_detail"] or ""), out["stopped_detail"]
        assert out["summary"] is None, "a summary was produced after a cleanup failure"
        assert calls["n"] == 1, f"cleanup entered {calls['n']} times"
        lines = open(path).read().strip().splitlines()
        assert len(lines) == 2, f"{len(lines)} durable records, expected the game + the stop"
        last = json.loads(lines[-1])
        assert last["aborted"] and last.get("cleanup_failure") is True, last
        return "cleanup failure stops the run, is persisted durably, and blocks the summary"

    def runner_binds_and_scores():
        """Construction errors outside play_game are still structured aborts."""
        from scripts.GPU.alphazero import twixtbot_g3_runner as RUN
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "g3.jsonl")
        t = dict(S.enumerate_tasks()[0], seed=S.CONSUMED_SEEDS[0])
        out = RUN._run_tasks(tasks=[t], twixt=twixt, Point=Point, ct=ct,
                             TwixtState=TwixtState, nnmplayer=None, repo_root=repo,
                             results_path=path,
                             play_game=lambda **kw: {"aborted": False},
                             cleanup=lambda: None, require_canonical=False)
        assert out["completed"] is False, "a consumed seed was allowed to run"
        assert out["results"][0]["aborted"] and "consumed" in out["results"][0]["abort_detail"]
        assert out["summary"] is None
        assert out["games_played"] == 0, out["games_played"]
        assert len(open(path).read().strip().splitlines()) == 1, "the abort was not persisted"
        return "a binding failure is recorded, persisted, and stops the run without a summary"

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
    ok &= check("production entry point has no switches", production_entry_has_no_switches)
    ok &= check("anchor runtime identity is verified", anchor_runtime_identity_is_verified)
    ok &= check("one shared anchor evaluator", shared_anchor_evaluator)
    ok &= check("gate is durable and exit-bound", gate_is_durable_and_exit_bound)
    ok &= check("the command runs as a module (subprocess)", command_runs_as_a_module)
    ok &= check("unexpected failures are classified (subprocess)", unexpected_failures_are_classified)
    ok &= check("run header records the verified identity", run_header_records_identity)
    ok &= check("counts are games, not records", counts_are_games_not_records)
    ok &= check("summarise rejects partial/duplicate/malformed sets", summarise_rejects_partial)
    ok &= check("cleanup failure is a run-level abort", cleanup_failure_is_a_run_abort)
    ok &= check("binding failure is a structured, persisted abort", runner_binds_and_scores)
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
        """State the scope by ENUMERATION, not by scanning text.

        A previous version scanned every check's message for the word "result"
        and tripped on "results_path is required" -- a text scan standing in for
        a fact. What matters is which seeds actually drove an engine or an RNG,
        so those are listed explicitly here.
        """
        # Every seed this preflight put through a real engine or RNG stream:
        SEEDS_DRIVEN = {SYNTHETIC_SEED}          # the one real reference agent
        # Reserved, unscheduled, and only ever used to CONSTRUCT a random.Random
        # that was never drawn from: no search, no move, no game.
        SEEDS_RNG_CONSTRUCTED_ONLY = {RNG_ONLY_SEED}
        # Seeds appearing only as inert task DATA in fake-seam tests (no engine,
        # no RNG, no move): the canonical schedule, plus the already-consumed
        # 202611000 used to prove the runner refuses it at binding.
        scheduled = set(range(*S.SCHEDULE_SEEDS))
        assert not (SEEDS_DRIVEN & scheduled), \
            f"a scheduled seed drove an engine: {sorted(SEEDS_DRIVEN & scheduled)}"
        lo, hi = S.RESERVED_SEEDS
        assert not any(lo <= x < hi for x in SEEDS_DRIVEN), \
            f"a reserved seed drove an engine: {sorted(SEEDS_DRIVEN)}"
        assert not (SEEDS_RNG_CONSTRUCTED_ONLY & scheduled), \
            "a SCHEDULED seed had an RNG constructed from it"
        assert "games" not in R and "results" not in R, "a game/result record exists"
        assert set(S.CONSUMED_SEEDS) == {202611000}, "the consumed-seed record changed"
        return (f"no scheduled seed was driven or even seeded an RNG; the only seed driven "
                f"through an engine was {SYNTHETIC_SEED} (outside the reservation). "
                f"{RNG_ONLY_SEED} is reserved-but-unscheduled and only had a random.Random "
                f"constructed from it, never drawn. Scheduled task dicts appear as inert DATA "
                f"tests (no engine, no RNG, no move). 202611000 stays recorded as consumed "
                f"by attempt 2; the schedule sits on [{S.SCHEDULE_SEEDS[0]}, {S.SCHEDULE_SEEDS[1]}).")

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
