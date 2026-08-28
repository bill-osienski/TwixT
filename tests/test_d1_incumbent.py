"""D1's incumbent side: the capture seam and the readout record. NO EXECUTION.

No checkpoint is loaded and no MLX evaluator is built. The searches here run
against a deterministic stub evaluator on a small board, so the REAL MCTS,
readout and RNG code all execute while nothing touches a model file.

The capture seam is asserted at the boundary that matters: not "capture returned
something", but that turning it on changes NEITHER the selected move NOR the
state of either generator. A capture that quietly drew a random number would
still look correct in a one-move test.
"""
import random

import numpy as np
import pytest

from scripts.GPU.alphazero import eval_readout as RO
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from scripts.GPU.alphazero.eval_runner import EvalConfig
from scripts.GPU.alphazero.game.twixt_state import TwixtState

SIMS = 8
CFG = EvalConfig(mcts_sims=SIMS, mcts_eval_batch_size=2, mcts_stall_flush_sims=2,
                 selection_mode="opening_temperature")
SEED = 90009042          # TEST_ONLY_SEED_INTERVALS; no schedule may contain it


class StubEvaluator:
    """Deterministic priors and values. Real MCTS, no model."""

    def __init__(self):
        self.calls = 0

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        self.calls += 1
        b, m = move_mask.shape
        priors = move_mask.astype(np.float32).copy()
        # A non-uniform but fixed shape, so ranks are meaningful and stable.
        priors *= (1.0 + np.arange(m, dtype=np.float32)[None, :] % 7)
        priors /= np.maximum(priors.sum(axis=1, keepdims=True), 1e-9)
        return priors, np.zeros(b, dtype=np.float32)


def _state(n=4):
    st = TwixtState(active_size=8)
    for mv in [(2, 2), (3, 4), (4, 3), (5, 5)][:n]:
        st = st.apply_move(mv)
    return st


def _agent(**kw):
    return G3.SeededReferenceAgent(evaluator=StubEvaluator(), colour="red",
                                   seed=SEED, config=CFG, **kw)


def _play(agent, plies=3):
    st = _state()
    moves = []
    for _ in range(plies):
        mv = agent(st)
        moves.append(mv)
        st = st.apply_move(mv)
        st = st.apply_move(next(m for m in st.legal_moves()))
    return moves


# ───────────────────────── the seam is OPT-IN and off by default ─────────────

def test_capture_is_off_by_default_and_exposes_nothing():
    agent = _agent()
    assert agent.capture is False
    _play(agent, 1)
    assert agent.last_capture is None


def test_capture_on_exposes_the_values_the_readout_actually_used(monkeypatch):
    """IDENTITY, not equality: the captured objects must be the ones passed to
    `select`, or the record describes a second computation nobody made."""
    seen = {}
    real = RO.select
    monkeypatch.setattr(RO, "select", lambda counts, *a, **k: (
        seen.setdefault("counts", counts), real(counts, *a, **k))[1])
    agent = _agent(capture=True)
    _play(agent, 1)
    cap = agent.last_capture
    assert cap is not None
    assert cap["counts"] is seen["counts"]
    assert cap["root"].visit_count == SIMS
    assert cap["top2"] and all(isinstance(s, RO.ChildStat) for s in cap["top2"])


# ─────────────── capture must not perturb the move or either stream ──────────

def test_capture_preserves_the_selected_moves():
    assert _play(_agent(capture=False)) == _play(_agent(capture=True))


def test_capture_preserves_the_state_of_both_generators():
    """Stronger than move equality: a stray draw can leave the move unchanged on
    a short game and still desynchronise every later ply."""
    off, on = _agent(capture=False), _agent(capture=True)
    _play(off), _play(on)
    assert off.readout_rng.getstate() == on.readout_rng.getstate()
    assert off.mcts.rng.getstate() == on.mcts.rng.getstate()


def test_the_no_perturbation_checks_catch_a_capture_that_draws(monkeypatch):
    """NEGATIVE CONTROL. Make capture draw one number and both checks must fail;
    otherwise they prove only that the code happens to work today."""
    real = G3.SeededReferenceAgent.__call__

    def drawing(self, state):
        move = real(self, state)
        if self.capture:
            self.readout_rng.random()
        return move

    off = _agent(capture=False)
    baseline = _play(off)
    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", drawing)
    on = _agent(capture=True)
    # BOTH detectors, separately. An `or` here would let one silently rot.
    assert _play(on) != baseline, "the move check did not notice the injected draw"
    assert off.readout_rng.getstate() != on.readout_rng.getstate(), \
        "the generator check did not notice the injected draw"


def test_the_capture_flag_threads_through_the_qualified_builder():
    """`build_reference_agent` is the ONE construction path; a seam it cannot
    reach would force D1 to build agents some other way."""
    ev = StubEvaluator()
    ev._g3_reference = "calib020_0001"
    ev._g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
    task = {"seed": SEED, "reference": "calib020_0001",
            "reference_sha1": "209cf2d4fd24a48553d259dd71b4954867b9473e",
            "anchor_colour": "black"}
    agent = G3.build_reference_agent(task=task, evaluator=ev, colour="red",
                                     config=G3.eval_config(), capture=True)
    assert agent.capture is True
    assert G3.build_reference_agent(task=task, evaluator=ev, colour="red",
                                    config=G3.eval_config()).capture is False


# ═════════════ D1's incumbent side: 12.6 settings, 5.2 capture ═══════════════
#
# The searches below are the REAL frozen path -- 400 simulations, noise off at
# the call site, `eval_readout.select` -- driven by the stub evaluator above. No
# checkpoint is read and no MLX buffer is allocated.

from scripts.GPU.alphazero import d1_probe as D1                # noqa: E402
from scripts.GPU.alphazero import e4_screen_reference as REF    # noqa: E402
from scripts.GPU.alphazero import eval_replay as RPL            # noqa: E402
from scripts.GPU.alphazero import l0_match_plan as PLAN         # noqa: E402

PREFIX = [(11, 11), (12, 13), (13, 12), (10, 13), (12, 10), (14, 14)]


def _board(prefix=PREFIX):
    st = TwixtState(active_size=24, to_move="red")
    for mv in prefix:
        st = st.apply_move(tuple(mv))
    return st


def _tagged_evaluator(identity):
    ev = StubEvaluator()
    ev._g3_reference = identity["reference"]
    ev._g3_sha1 = identity["reference_sha1"]
    return ev


@pytest.fixture(scope="module")
def identity():
    return D1.frozen_incumbent_identity()


@pytest.fixture
def incumbent(identity):
    loads = []

    def load(repo_root):
        loads.append(repo_root)
        return _tagged_evaluator(identity)

    inc = D1._Incumbent(repo_root=".", _load=load)
    inc._loads = loads                       # the test's counter, not the class's
    return inc


def _pos(seed=None, ply=None):
    return {"task_id": "t", "ply": len(PREFIX) if ply is None else ply,
            "seed": D1.SEED_INTERVAL[0] if seed is None else seed}


# ───────────────── 12.6: the incumbent is read, never retyped ────────────────

def test_the_incumbent_identity_comes_from_the_frozen_l0_plan(identity):
    plan = PLAN.load_l0_plan()
    assert {(t["reference"], t["reference_sha1"]) for t in plan["tasks"]} == \
        {(identity["reference"], identity["reference_sha1"])}
    assert identity["plan_sha256"] == PLAN.L0_PLAN_SHA256


def test_the_recorded_settings_are_the_frozen_ones(identity):
    assert identity["eval_config"]["mcts_sims"] == 400
    assert identity["noise_suppression"] == "search_with_root(state, add_noise=False)"
    assert identity["readout_path"] == "eval_readout.select, never mcts.select_move"
    assert identity["search_mask"] == {"red": 0xA5A5A5, "black": 0x5A5A5A}


# ─────────────── 12.6: ONE evaluator for the whole run, never reloaded ───────

def test_one_evaluator_is_loaded_for_the_entire_run(incumbent):
    budget = D1.QueryBudget()
    for i in range(3):
        incumbent(pos=_pos(seed=D1.SEED_INTERVAL[0] + i), state=_board(), budget=budget)
    assert incumbent._loads == ["."], incumbent._loads


def test_a_fresh_agent_per_position_carries_only_that_positions_seed(incumbent):
    budget = D1.QueryBudget()
    a = incumbent(pos=_pos(seed=D1.SEED_INTERVAL[0]), state=_board(), budget=budget)
    b = incumbent(pos=_pos(seed=D1.SEED_INTERVAL[0] + 1), state=_board(), budget=budget)
    assert a["seed"] != b["seed"]
    assert a["streams"] != b["streams"]
    # Same position, different seed: the wrapper is stateless wrt the evaluator,
    # so the readout depends on the seed and nothing carried over.
    assert a["streams"]["search_seed"] == \
        a["seed"] ^ REF.SeededReferenceAgent.SEARCH_MASK[_board().to_move]


def test_the_incumbent_spends_exactly_one_query_per_position(incumbent):
    budget = D1.QueryBudget()
    incumbent(pos=_pos(), state=_board(), budget=budget)
    assert budget.spent == 1, "12.4 funds ONE incumbent readout per position"


# ──────────────────── 5.2: the record, on the ONE definition ────────────────

def test_the_record_reuses_ply_record_rather_than_restating_it(incumbent):
    rec = incumbent(pos=_pos(), state=_board(), budget=D1.QueryBudget())
    for key in ("root_value", "root_top1_share", "selected_visit_rank",
                "selected_visit_count", "root_total_visits", "n_legal", "top2",
                "readout_overrode_leader"):
        assert key in rec, key
    assert rec["root_total_visits"] == 400
    assert rec["n_legal"] == len(rec["root_visits"])


def test_the_recorded_rank_matches_the_one_ply_record_computes(incumbent):
    rec = incumbent(pos=_pos(), state=_board(), budget=D1.QueryBudget())
    counts = {tuple(int(x) for x in k.split(",")): v
              for k, v in rec["root_visits"].items()}
    direct = RPL.ply_record(rec["ply"], rec["player"], (rec["row"], rec["col"]),
                            counts, rec["root_value"])
    assert rec["selected_visit_rank"] == direct["selected_visit_rank"]
    assert rec["root_top1_share"] == direct["root_top1_share"]


def test_the_raw_policy_covers_every_legal_move_and_ranks_the_chosen_one(incumbent):
    rec = incumbent(pos=_pos(), state=_board(), budget=D1.QueryBudget())
    assert set(rec["raw_policy"]) == set(rec["root_visits"])
    assert len(rec["raw_policy"]) == rec["n_legal"] == len(_board().legal_moves())
    key = f"{rec['row']},{rec['col']}"
    assert rec["selected_policy_mass"] == rec["raw_policy"][key]
    better = sum(1 for k, v in rec["raw_policy"].items()
                 if (-v, tuple(int(x) for x in k.split(","))) <
                    (-rec["raw_policy"][key], (rec["row"], rec["col"])))
    assert rec["selected_policy_rank"] == better + 1


def test_a_root_policy_that_carries_noise_is_a_VOID(incumbent, monkeypatch):
    """NEGATIVE CONTROL for 12.6's "noise off". `add_noise=False` leaves
    `priors` and `priors_raw` the same object; anything else means the root this
    record describes is not the root the frozen settings specify."""
    real = G3.SeededReferenceAgent.__call__

    def noisy(self, state):
        move = real(self, state)
        root = self.last_capture["root"]
        root.priors = {k: v + 0.001 for k, v in root.priors_raw.items()}
        return move

    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", noisy)
    with pytest.raises(D1.D1VoidError, match="noise"):
        incumbent(pos=_pos(), state=_board(), budget=D1.QueryBudget())


# ──────────────── the seed still has to survive the qualified checks ─────────

def test_an_exposed_seed_is_refused_by_the_qualified_builder(incumbent, monkeypatch):
    """`REF.build` asks the EXECUTABLE question, so a block that was spent
    between preregistration and execution cannot be replayed."""
    monkeypatch.setattr(REF, "EXPOSED_SEED_INTERVALS",
                        REF.EXPOSED_SEED_INTERVALS + (D1.SEED_INTERVAL,))
    with pytest.raises(REF.E4ReferenceError, match="EXPOSED"):
        incumbent(pos=_pos(), state=_board(), budget=D1.QueryBudget())


def test_the_agent_moves_the_colour_that_is_to_move(incumbent):
    st = _board(PREFIX + [(16, 16)])          # black to move
    assert st.to_move == "black"
    rec = incumbent(pos=_pos(ply=len(PREFIX) + 1), state=st, budget=D1.QueryBudget())
    assert rec["player"] == "black"


# ═════════════ order, and one position end to end with everything mocked ═════

def _replay_stdout(prefix):
    from scripts.GPU.alphazero import t1j_adapter as A
    post = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true "
            f"refl_ok=true refl_n={D1.INT.REPLAY_REFL_N} failures=0")
    st, moves, out = TwixtState(active_size=24, to_move="red"), [], []
    for mv in list(prefix) + [None]:
        pegs, bridges = A.our_snapshot(st)
        legal = {A.to_t1j(r, c) for (r, c) in st.legal_moves()}
        bits = "".join("1" if (i // A.BOARD_N, i % A.BOARD_N) in legal else "0"
                       for i in range(A.LEGAL_BITS))
        hist = " ".join(f"{x},{y}" for x, y in (A.to_t1j(*m) for m in moves))
        out.append(f"PLY {st.ply} moveNr={st.ply} next={A.PLAYER_TO_T1J[st.to_move]} "
                   f"termY=false termX=false\n  PEGS {' '.join(sorted(pegs))}\n"
                   f"  BRIDGES {' '.join(sorted(bridges))}\n  HIST {hist}\n"
                   f"  LEGAL {bits}\n")
        if mv is None:
            break
        moves.append(tuple(mv))
        st = st.apply_move(tuple(mv))
    return "".join(out) + post + "\n"


@pytest.fixture
def boundary(monkeypatch):
    """The process boundary. Serves replays and query replies; spawns nothing."""
    import subprocess
    from scripts.GPU.alphazero import t1j_adapter as A
    calls = []
    x, y = A.to_t1j(15, 15)

    def fake_run(args, **kw):
        calls.append({"args": args, "kw": kw})
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, _replay_stdout(PREFIX), "")
        depth = int(args[args.index("query") + 1])
        hist = " ".join(f"{a},{b}" for a, b in (A.to_t1j(*m) for m in PREFIX))
        legal = {A.to_t1j(r, c) for (r, c) in _board().legal_moves()}
        bits = "".join("1" if (i // A.BOARD_N, i % A.BOARD_N) in legal else "0"
                       for i in range(A.LEGAL_BITS))
        return subprocess.CompletedProcess(args, 0, (
            f"QUERY q=1 requested_depth={depth} move_x={x} move_y={y} to_move=Y "
            f"usealphabeta=true currentMaxPly={depth} completed_depth={depth} "
            f"completed=true legal=true null_sentinel=false moveNr={len(PREFIX)} "
            f"eval_regime=fixed elapsed_us=1000\n"
            f"PLY {len(PREFIX)} moveNr={len(PREFIX)} next=X termY=false termX=false\n"
            f"  PEGS \n  BRIDGES \n  HIST {hist}\n  LEGAL {bits}\n"), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_no_evaluator_is_loaded_while_the_seed_block_is_unregistered(identity, tmp_path):
    """ORDER, asserted at the effect. An unregistered block must cost nothing --
    not a compile, and not a checkpoint read either."""
    loads = []
    inc = D1._Incumbent(".", _load=lambda r: loads.append(r) or _tagged_evaluator(identity))
    with pytest.raises(D1.D1Error, match="not registered"):
        D1._run_d1_unguarded(
            positions=[{"task_id": "t", "ply": len(PREFIX), "prefix": PREFIX,
                        "seed": D1.SEED_INTERVAL[0], "digest": "x" * 64}],
            paths=D1.T1jPaths(java="j", jar="j", classes="c", ply_cap=280),
            out_path=str(tmp_path / "r.json"), _compile=lambda d: None, _incumbent=inc)
    assert loads == [], "the incumbent checkpoint was read before the block was registered"


def test_one_position_end_to_end_records_both_sides(identity, boundary, monkeypatch,
                                                    tmp_path):
    """Selection output -> E3b binding -> incumbent readout -> T1j, all mocked.

    Five queries, exactly as 12.4 funds them: one incumbent readout plus two
    invocations at each of the two depths.
    """
    from scripts.GPU.alphazero import d1_selection as SEL
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + (D1.SEED_INTERVAL,))
    inc = D1._Incumbent(".", _load=lambda r: _tagged_evaluator(identity))
    out = tmp_path / "r.json"
    report = D1._run_d1_unguarded(
        positions=[{"task_id": "t", "ply": len(PREFIX), "prefix": PREFIX,
                    "seed": D1.SEED_INTERVAL[0],
                    "digest": SEL.canonical_digest(_board())}],
        paths=D1.T1jPaths(java="j", jar="j", classes="c", ply_cap=280),
        out_path=str(out), _compile=lambda d: None, _incumbent=inc)

    assert report["queries_spent"] == 5 == 1 + 2 * 2
    assert report["incumbent_identity"]["reference"] == identity["reference"]
    pos = report["positions"][0]
    assert pos["incumbent"]["root_total_visits"] == 400
    assert [d["depth"] for d in pos["depths"]] == [3, 6]
    assert all(d["invocations"] == 2 for d in pos["depths"])
    assert len([c for c in boundary if "replay" in c["args"]]) == 1
    assert len([c for c in boundary if "query" in c["args"]]) == 4
    assert out.exists()


def test_the_end_to_end_report_is_create_only(identity, boundary, monkeypatch, tmp_path):
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + (D1.SEED_INTERVAL,))
    out = tmp_path / "r.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        D1._run_d1_unguarded(positions=[], paths=D1.T1jPaths(java="j", jar="j",
                                                             classes="c", ply_cap=280),
                             out_path=str(out), _compile=lambda d: None,
                             _incumbent=lambda **kw: {})
    assert out.read_text(encoding="utf-8") == "{}", "an existing record was overwritten"
