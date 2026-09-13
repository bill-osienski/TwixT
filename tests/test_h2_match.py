"""H2 — the deterministic-readout head-to-head, implemented CODE-AND-TEST ONLY
against the card frozen at 5aa5db4.

🔴 NOTHING HERE RUNS A GAME. No model is loaded, no JVM started, no query issued,
no seed registered or drawn, and no gate opened. The play seam is a fixture; the
gate is asserted CLOSED in the real repository, never patched open.
"""
import inspect
import os
import json

import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import h2_match_command as CMD
from scripts.GPU.alphazero import h2_match_plan as PLAN
from scripts.GPU.alphazero import h2_match_rules as R
from scripts.GPU.alphazero import h2_match_runner as RUN


# ────────────────────────── the frozen design numbers ───────────────────────

def test_the_card_numbers_are_the_module_numbers():
    assert (R.N_REPS, R.N_OPENINGS, R.N_ARMS, R.N_GAMES) == (46, 8, 2, 736)
    assert R.H2_SEED_BLOCK == (202622000, 202622736)           # attempt 3
    assert R.H2_ATTEMPT2_SEED_BLOCK == (202620000, 202620736)  # spent by the INCIDENT
    assert R.H2_ATTEMPT1_SEED_BLOCK == (202618000, 202618736)  # spent, retired whole
    assert len({R.H2_SEED_BLOCK, R.H2_ATTEMPT1_SEED_BLOCK,
                R.H2_ATTEMPT2_SEED_BLOCK}) == 3, "three attempts, three blocks"
    assert R.H2_SEED_BLOCK[1] - R.H2_SEED_BLOCK[0] == R.N_GAMES
    assert R.PARITY == 0.50
    assert R.MIN_DISTINCT_PER_CELL == 42
    assert R.SELECTION_MODE == "argmax"
    assert R.T1J_MDPLY == 6 and R.MCTS_SIMS == 400
    assert R.PLY_CAP == 280
    assert R.CAP_NO_RATE_THRESHOLD == 368


def test_the_decisive_bands_are_DERIVED_from_the_interval_not_typed():
    lo, hi = R.decisive_bands()
    assert round(lo, 4) == 0.4499 and round(hi, 4) == 0.5501
    assert round(hi - R.PARITY, 7) == 0.0500603


def test_the_terminal_reasons_are_exactly_win_and_cap_with_NO_resignation():
    assert R.TERMINAL_REASONS == ("win", "cap")
    assert "resignation" not in R.TERMINAL_REASONS


def test_the_shared_constants_are_BOUND_to_L0_not_retyped():
    from scripts.GPU.alphazero import l0_match_rules as L0
    assert R.PLY_CAP is L0.PLY_CAP and R.TERMINAL_REASONS is L0.TERMINAL_REASONS
    assert R.WINNERS is L0.WINNERS and R.ALPHA is L0.ALPHA


def test_there_is_no_early_stop():
    assert R.EARLY_STOP is None
    assert R.may_stop_early(anything=1) is False


# ───────────────────────────── the parity verdict ───────────────────────────

@pytest.mark.parametrize("lo,hi,want", [
    (0.5001, 0.60, "T1J_STRONGER"),
    (0.40, 0.4999, "NOT_STRONGER"),
    (0.45, 0.55, "INCONCLUSIVE"),
    (0.50, 0.60, "INCONCLUSIVE"),     # AT parity is not above it
    (0.40, 0.50, "INCONCLUSIVE"),     # AT parity is not below it
])
def test_the_parity_rule_at_and_around_the_threshold(lo, hi, want):
    assert R.parity_verdict(lo, hi) == want


def test_the_0_75_threshold_is_carried_but_is_NOT_the_rule():
    assert R.H1_INVESTMENT_THRESHOLD == 0.75
    src = inspect.getsource(R.parity_verdict)
    assert "0.75" not in src and "H1_INVESTMENT_THRESHOLD" not in src


# ───────────────────────── the frozen transcript identity ───────────────────

#: 🔴 THE MOVERS COME FROM THE PRODUCTION STATE MACHINE, NOT FROM `colour_at_ply`.
#: My first fixtures called the same helper the transcript uses, so the test and
#: the code shared one bug and agreed with each other -- and the bug was real: the
#: helper derived the mover from the ARM, which is wrong for every `t1j_black`
#: task. `TwixtState` is the authority on turn order.
def _engine_movers(n, first):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    st, out = TwixtState(), {}
    while st.ply < first + n - 1:
        mover = st.to_move
        st = st.apply_move(sorted(st.legal_moves())[0])
        out[st.ply] = mover
    return out


def _plies(n=4, *, first=7, flip=False):
    movers = _engine_movers(n, first)
    out = []
    for k in range(n):
        ply = first + k
        mover = movers[ply]
        if flip:
            mover = "black" if mover == "red" else "red"
        out.append({"ply": ply, "mover": mover, "move": [ply, k]})
    return out


def _result(n=4, *, first=7, reason="win", winner="red"):
    return {"plies": first + n - 1, "terminal_reason": reason, "winner": winner}


def _t(plies=None, result=None, *, bound=6):
    return R.transcript(plies if plies is not None else _plies(),
                        result if result is not None else _result(),
                        opening_bound=bound)


def test_a_transcript_is_the_moves_the_reason_and_the_winner_and_NOTHING_ELSE():
    t = _t()
    assert t == (("red", 7, 0), ("black", 8, 1), ("red", 9, 2), ("black", 10, 3),
                 ("terminal", "win", "red"))


def test_two_games_with_DIFFERENT_seeds_and_ids_but_IDENTICAL_PLAY_are_ONE_transcript():
    """🔑 THE VACUITY THIS DEFINITION EXISTS TO PREVENT. Hash the whole record and
    `seed`, `task_id` and `rep` make every game unique even when the gameplay is
    byte-identical."""
    a = _plies()
    b = [dict(p, task_id="other", rep=45, seed=999, elapsed_s=1.25) for p in a]
    assert R.transcript_digest(_t(a)) == R.transcript_digest(_t(b))


def test_changing_ONE_played_move_creates_a_DISTINCT_transcript():
    a = _plies()
    b = [dict(p) for p in a]
    b[2]["move"] = [21, 21]
    assert R.transcript_digest(_t(a)) != R.transcript_digest(_t(b))


def test_a_different_terminal_reason_or_winner_creates_a_distinct_transcript():
    base = R.transcript_digest(_t())
    assert R.transcript_digest(_t(result=_result(reason="cap", winner=None))) != base
    assert R.transcript_digest(_t(result=_result(winner="black"))) != base


def test_the_ply_sequence_must_be_EXACTLY_opening_bound_plus_one_to_result_plies():
    R.transcript(_plies(), _result(), opening_bound=6)
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(_plies(), _result(), opening_bound=5)


def test_REMOVING_THE_FIRST_ply_REFUSES_though_the_rest_stays_contiguous():
    """🔴 Contiguity is a property of the INTERIOR: a sequence missing its first
    record is still contiguous from whatever survives."""
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(_plies()[1:], _result(), opening_bound=6)


def test_REMOVING_THE_FINAL_ply_REFUSES_though_the_rest_stays_contiguous():
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(_plies()[:-1], _result(), opening_bound=6)


def test_a_DUPLICATED_or_OUT_OF_ORDER_ply_REFUSES(  ):
    p = _plies()
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(p[:2] + [p[1]] + p[2:], _result(), opening_bound=6)
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript([p[1], p[0]] + p[2:], _result(), opening_bound=6)


def test_FLIPPING_EVERY_MOVER_refuses_even_though_alternation_is_preserved():
    """🔴 The case an alternation check cannot see: every mover swapped keeps the
    moves alternating while swapping which side played each one."""
    flipped = _plies(flip=True)
    movers = [p["mover"] for p in flipped]
    assert all(a != b for a, b in zip(movers, movers[1:])), "still alternating"
    with pytest.raises(R.H2RulesError, match="PLY PARITY"):
        R.transcript(flipped, _result(), opening_bound=6)


def test_THE_MOVER_FOLLOWS_PLY_PARITY_AND_NOT_THE_ARM_checked_against_the_engine():
    """🔴 THE DEFECT THIS CLOSES, and it would have VOIDed half the schedule. Red
    moves first in every game; `colour_arm` only says which SYSTEM plays which
    colour. The expectation is taken from `TwixtState` itself, so this test cannot
    agree with a wrong helper the way my first one did."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    st = TwixtState()
    assert st.to_move == R.STARTING_COLOUR == "red"
    for _ in range(8):
        mover = st.to_move
        st = st.apply_move(sorted(st.legal_moves())[0])
        assert R.colour_at_ply(st.ply) == mover, (st.ply, mover)
    assert R.colour_at_ply(7) == "red" and R.colour_at_ply(8) == "black"
    assert not hasattr(R, "expected_mover"), "the arm-derived helper must be gone"


def test_a_BLACK_ARM_transcript_is_accepted_with_the_SAME_parity_movers():
    """The half of the schedule the old rule would have refused."""
    t = R.transcript(_plies(), _result(), opening_bound=6)
    assert t[0][0] == "red", "ply 7 is red in BOTH arms"


def test_a_terminal_reason_outside_the_two_is_REFUSED():
    with pytest.raises(R.H2RulesError, match="no resignation"):
        R.transcript(_plies(), _result(reason="resignation"), opening_bound=6)


@pytest.mark.parametrize("bad", ["7", 7.0, True])
def test_a_ply_index_that_is_not_an_INT_is_refused(bad):
    p = _plies()
    p[0] = dict(p[0], ply=bad)
    with pytest.raises(R.H2RulesError):
        R.transcript(p, _result(), opening_bound=6)


def test_a_move_whose_coordinates_are_STRINGS_is_refused():
    p = _plies()
    p[1] = dict(p[1], move=["8", "1"])
    with pytest.raises(R.H2RulesError, match="int is required"):
        R.transcript(p, _result(), opening_bound=6)


def test_the_excluded_fields_are_DECLARED_not_merely_absent():
    for f in ("task_id", "rep", "seed", "elapsed_s", "root_visits"):
        assert f in R.TRANSCRIPT_EXCLUDES


# ──────────────────────── the per-cell degeneracy screen ────────────────────

CELLS = [(f"o{i}", arm) for i in range(1, 9) for arm in ("t1j_red", "t1j_black")]


def _per_game(distinct_per_cell, *, task_ids=None, reps=None):
    """One entry per game; `distinct_per_cell` maps a cell to how many distinct
    transcripts it should hold. Task ids are carried because the screen BINDS its
    input to the canonical task list."""
    out = []
    for c, cell in enumerate(CELLS):
        k = distinct_per_cell.get(cell, R.N_REPS)
        for i in range(reps if reps is not None else R.N_REPS):
            out.append({"task_id": f"t-{c}-{i}", "opening": cell[0],
                        "colour_arm": cell[1],
                        "transcript_digest": f"{cell}-{min(i, k - 1)}"})
    if task_ids is not None:
        for row, tid in zip(out, task_ids):
            row["task_id"] = tid
    return out


def test_a_fully_distinct_design_passes_and_reports_ALL_SIXTEEN_cells():
    s = R.degeneracy_screen(_per_game({}))
    assert s["passes"] is True
    assert s["n_cells"] == 16 and len(s["cells"]) == 16
    assert all(c["n_distinct"] == 46 and c["n_games"] == 46 for c in s["cells"])


def test_ONE_WHOLLY_COLLAPSED_CELL_FAILS_though_the_GLOBAL_rate_is_above_90_percent():
    """🔴 THE COUNTEREXAMPLE THE CARD NAMES. 691/736 = 93.9% distinct globally, and
    a sixteenth of the design carries nothing."""
    s = R.degeneracy_screen(_per_game({CELLS[0]: 1}))
    assert s["passes"] is False
    assert s["failing_cells"] == [list(CELLS[0])]
    assert s["global_distinct_rate_NOT_THE_RULE"] > 0.93


@pytest.mark.parametrize("k,passes", [(46, True), (42, True), (41, False), (1, False)])
def test_the_per_cell_threshold_accepts_AT_42_and_refuses_one_below(k, passes):
    s = R.degeneracy_screen(_per_game({CELLS[3]: k}))
    assert s["passes"] is passes


def test_a_missing_cell_is_REFUSED_now_that_the_screen_binds_its_input():
    """It used to return `passes: False`; binding the input makes it a REFUSAL,
    which is stronger: a vector that is not the design cannot be screened at all."""
    games = [g for g in _per_game({}) if (g["opening"], g["colour_arm"]) != CELLS[0]]
    with pytest.raises(R.H2RulesError, match="not 736|rows for"):
        R.degeneracy_screen(games)


def test_a_SHORT_VECTOR_of_42_per_cell_is_REFUSED_not_passed():
    """🔴 THE DEFECT THIS CLOSES. 672 rows -- exactly 42 per cell -- satisfied "16
    cells and at least 42 distinct" while missing 64 games entirely."""
    games = _per_game({}, reps=42)
    assert len(games) == 672
    with pytest.raises(R.H2RulesError, match="not 736|rows for"):
        R.degeneracy_screen(games)


def test_A_FULL_LENGTH_vector_with_UNEQUAL_CELLS_is_REFUSED():
    """🔴 The short-vector test could not see this: 736 rows in total, but 47 games
    in one cell and 45 in another. Only the per-cell game count catches it."""
    games = _per_game({})
    games[0] = dict(games[0], opening=CELLS[1][0], colour_arm=CELLS[1][1])
    assert len(games) == 736
    with pytest.raises(R.H2RulesError, match="exactly 46 games"):
        R.degeneracy_screen(games)


def test_a_vector_whose_TASK_IDS_are_not_the_canonical_ones_is_REFUSED():
    games = _per_game({})
    with pytest.raises(R.H2RulesError, match="not exactly theirs|rows for"):
        R.degeneracy_screen(games, canonical_task_ids=[f"other-{i}" for i in range(736)])


# ─────────────────────── the report, and the ORDER of its checks ────────────

def _tasks():
    return PLAN.build_tasks(PLAN.load_source_plan())


def _results(tasks, *, t1j_wins):
    rows = []
    for i, t in enumerate(tasks):
        t1j_won = i < t1j_wins
        anchor = t["anchor_colour"]
        winner = anchor if t1j_won else ("black" if anchor == "red" else "red")
        rows.append({"task_id": t["task_id"], "winner": winner,
                     "terminal_reason": "win",
                     "t1j_points": 1.0 if t1j_won else 0.0,
                     "plies": 42, "seed": t["seed"]})
    return rows


def test_a_FAILING_SCREEN_PREVENTS_THE_INTERVAL_FROM_BEING_COMPUTED():
    """🔑 A screen that runs after the number it guards is decoration."""
    tasks = _tasks()
    ids = [t["task_id"] for t in tasks]
    out = R.h2_report(_results(tasks, t1j_wins=700), tasks,
                      _per_game({CELLS[0]: 1}, task_ids=ids),
                      task_digest=R.H2_TASK_DIGEST)
    assert out["reported"] is False
    assert out["outcome"] == "INCONCLUSIVE — DEGENERATE DESIGN"
    assert out["interval"] is None and out["score"] is None
    assert out["degeneracy_screen"]["failing_cells"] == [list(CELLS[0])]


@pytest.mark.parametrize("wins,want", [
    (736, "T1J_STRONGER"), (0, "NOT_STRONGER"), (368, "INCONCLUSIVE"),
])
def test_the_verdict_follows_the_parity_rule_end_to_end(wins, want):
    tasks = _tasks()
    out = R.h2_report(_results(tasks, t1j_wins=wins), tasks,
                      _per_game({}, task_ids=[t["task_id"] for t in tasks]),
                      task_digest=R.H2_TASK_DIGEST)
    assert out["reported"] is True
    assert out["outcome"] == want == out["verdict"]


def test_the_report_carries_the_interval_STANDING_and_all_sixteen_counts():
    tasks = _tasks()
    out = R.h2_report(_results(tasks, t1j_wins=400), tasks,
                      _per_game({}, task_ids=[t["task_id"] for t in tasks]),
                      task_digest=R.H2_TASK_DIGEST)
    assert "NOMINAL UNDER THE INDEPENDENCE MODEL" in out["interval_standing"]
    assert len(out["degeneracy_screen"]["cells"]) == 16
    assert out["selection_mode"] == "argmax"
    assert out["reported_for_continuity_only"]["h1_investment_threshold"] == 0.75


def test_the_report_FORBIDS_the_claims_the_card_forbids():
    joined = " ".join(R.FORBIDDEN_CLAIMS).lower()
    for phrase in ("pooling", "confounded", "attributable to the readout",
                   "strongest deterministic", "coverage guarantee", "training"):
        assert phrase in joined


# ────────────────────────────── the frozen plan ─────────────────────────────

def test_the_plan_builds_the_frozen_736_and_matches_the_PINNED_digest():
    plan = PLAN.build_plan()
    assert plan["n_tasks"] == 736 and plan["cells"] == 16 and plan["reps_per_cell"] == 46
    assert plan["task_digest"] == R.H2_TASK_DIGEST


def test_every_row_carries_its_POSITIONAL_seed_and_the_readout_mode():
    tasks = _tasks()
    lo = R.H2_SEED_BLOCK[0]
    assert [t["seed"] for t in tasks] == list(range(lo, lo + 736))
    assert all(t["selection_mode"] == "argmax" for t in tasks)


@pytest.mark.parametrize("mutate,match", [
    (lambda ts: ts[:-1], "expected exactly"),
    # 🔑 A SWAP, not a shift. Shifting one seed DUPLICATES another and the
    # structural check catches it first, so the positional rule would never be
    # exercised. Swapping keeps the set identical and injective: only "row i
    # carries lo + i" can see it.
    (lambda ts: [dict(ts[i], seed=ts[6]["seed"]) if i == 5 else
                 dict(ts[i], seed=ts[5]["seed"]) if i == 6 else t
                 for i, t in enumerate(ts)], "bound POSITIONALLY"),
    (lambda ts: [{k: v for k, v in t.items() if k != "selection_mode"} if i == 0 else t
                 for i, t in enumerate(ts)], "selection_mode"),
    (lambda ts: [dict(t, t1j_mdPly=3) if i == 0 else t for i, t in enumerate(ts)],
     "mdPly"),
])
def test_the_schedule_validator_refuses_a_broken_design(mutate, match):
    with pytest.raises(PLAN.H2PlanError, match=match):
        PLAN.validate_h2_schedule(mutate(_tasks()))


# ───────────────────────────── the three barriers ───────────────────────────

def test_THE_GATE_IS_SHUT_IN_THE_REAL_REPOSITORY():
    assert RUN.H2_EXECUTION_AUTHORIZED is False
    with pytest.raises(RUN.H2Error, match="UNAUTHORIZED"):
        RUN.check_gate()


def test_the_public_entry_READS_THE_GATE_before_anything_else(tmp_path):
    out = tmp_path / "r.jsonl"
    with pytest.raises(RUN.H2Error, match="UNAUTHORIZED"):
        RUN.run_h2(results_path=str(out), trace_path=str(tmp_path / "t.jsonl"),
                   report_path=str(tmp_path / "rep.json"))
    assert not out.exists()


def test_the_public_entry_TAKES_ONLY_THE_OUTPUT_PATHS():
    """🔴 THE DEFECT THIS CLOSES. It used to accept `tasks`, `play`, `identity` and
    `deadline_s`, so opening the gate would have authorized CALLER-SUPPLIED
    GAMEPLAY through the API while the CLI could not run the real match at all."""
    assert list(inspect.signature(RUN.run_h2).parameters) == \
        ["results_path", "trace_path", "report_path"]
    seams = list(inspect.signature(RUN._run_h2_unguarded).parameters)
    assert {"play", "identity", "deadline_s", "tasks"} <= set(seams), \
        "the seams must survive on the PRIVATE entry, for tests"


def _seam_tree():
    import ast, textwrap
    outer = ast.parse(textwrap.dedent(inspect.getsource(RUN._production_play))).body[0]
    return outer, next(n for n in outer.body
                       if isinstance(n, ast.FunctionDef) and n.name == "play")


def test_the_production_play_seam_EXISTS_and_is_built_lazily():
    """A refusing stub is not a production path. The factory must construct
    nothing effectful, and the seam must not refuse before doing its work."""
    import ast
    play = RUN._production_play("/tmp/h2-never-written")
    assert callable(play) and play._state is None
    _outer, fn = _seam_tree()
    raises = [n for n in fn.body if isinstance(n, ast.Raise)]
    assert not raises, "the seam refuses before playing: that is a stub, not a path"


def test_the_seam_WIRES_THE_HARNESS_GAME_LOOP():
    """🔴 A control that inserted a refusal above the loop went unseen, because the
    test only read the source. The call itself is asserted, by AST."""
    import ast
    _outer, fn = _seam_tree()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert any(getattr(c.func, "attr", "") == "play_task" for c in calls), \
        "the seam must call e4_screen_runner.play_task"


def test_the_seam_BUILDS_AN_ARGMAX_CONFIG_rather_than_reusing_the_frozen_one():
    """🔴 `argmax_cfg = cfg` passed a source check that only looked for the NAME.
    The assignment must be a CALL that sets selection_mode."""
    import ast
    _outer, fn = _seam_tree()
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "argmax_cfg" for t in n.targets)]
    assert len(assigns) == 1, "exactly one argmax config is built"
    value = assigns[0].value
    assert isinstance(value, ast.Call), "the config must be REBUILT, not aliased"
    assert "selection_mode" in ast.dump(value), "and it must set selection_mode"


def test_THE_RETRY_BLOCK_IS_EXPOSED_383_AND_RETIRED_WHOLE():
    """🔴 INVERTED 2026-09-12, the day after registration, by the INCIDENT: an
    injected-defect control deleted `check_gate()` from `run_h2` and the harness
    played 383 of the 736 games before it was killed.

    * ACCOUNTED, from its registration.
    * EXPOSED for the FIRST 383 seeds ONLY -- 202620000..202620382 -- each of which
      has a `task_result` and ply records in the preserved rogue output. Zero-based
      task 383 reached only a trace `task_start`: no `opening_bound`, no ply, no
      result, so seed 202620383 is NOT claimed as a confirmed draw.
    * RETIRED WHOLE, all 736. The one-shot rule does not care how far a started
      schedule got, and the uncertainty about task 383 is covered by the
      whole-block retirement rather than by a claim either way.

    ⚠ The 383 preserved games are INCIDENT EVIDENCE. They are not an H2 verdict:
    they are an unauthorized partial run of a design that requires all 736.
    """
    lo, hi = R.H2_ATTEMPT2_SEED_BLOCK
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                 "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS"):
        assert getattr(REF, name), f"vacuous: {name} is empty"
    for seed in range(lo, hi):
        st = REF.seed_status(seed)
        assert st["accounted"] and st["retired"], (seed, st)
        assert st["exposed"] is (seed < lo + 383), (seed, st)
        assert not st["test_only"], (seed, st)
    # the FIRST seed of the interrupted game: retired, but NOT claimed as drawn
    assert REF.seed_status(lo + 383)["exposed"] is False
    assert REF.seed_status(lo + 383)["retired"] is True
    assert RUN.H2_EXECUTION_AUTHORIZED is False
    with pytest.raises(RUN.H2Error, match="UNAUTHORIZED"):
        RUN.run_h2(results_path="/dev/null/x", trace_path="/dev/null/y",
                   report_path="/dev/null/z")


def test_THE_RETRY_BLOCK_CANNOT_BE_SCHEDULED_AGAIN_but_STAYS_ACCOUNTED():
    """Which barrier refuses, and which does not. `check_seed_registration` asks only
    whether the block is ACCOUNTED -- retirement does not un-account it, so that
    barrier still PASSES, by design. Availability is a different question, and
    `validate_schedule_executable` is the one that now refuses.

    🔑 Asserting the wrong barrier would have hidden the real protection: my first
    version of this test expected `check_seed_registration` to raise, and it does
    not. The refusal that matters is the per-task one."""
    RUN.check_seed_registration()                       # still satisfied
    lo = R.H2_ATTEMPT2_SEED_BLOCK[0]
    tasks = [dict(t, seed=lo + i, rng_streams=REF.rng_stream_seeds(dict(t, seed=lo + i)))
             for i, t in enumerate(PLAN.build_tasks(PLAN.load_source_plan()))]
    with pytest.raises(REF.E4ReferenceError, match="EXPOSED|RETIRED|retired"):
        REF.validate_schedule_executable(tasks)


def test_ATTEMPT_ONES_BLOCK_IS_STILL_SPENT_and_the_retirement_STANDS():
    """INVERTED AGAIN 2026-09-11, hours after the registration, because the single
    authorized match VOIDed at task 0.

    * ACCOUNTED, from its registration.
    * NOT EXPOSED: `build_reference_agent` refused before any agent was built, so
      zero ply records exist and no seed was ever drawn.
    * RETIRED WHOLE: a one-shot schedule was STARTED and did not complete.

    ⚠ The retirement was REVIEWED AND KEPT on 2026-09-11 rather than overruled, so
    this must keep asserting it: the retry uses a FRESH block, and this one stays
    spent.
    """
    lo, hi = R.H2_ATTEMPT1_SEED_BLOCK
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                 "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS"):
        assert getattr(REF, name), f"vacuous: {name} is empty"
    for seed in range(lo, hi):
        st = REF.seed_status(seed)
        assert st["accounted"] and st["retired"], (seed, st)
        assert not (st["exposed"] or st["test_only"]), (seed, st)
        assert seed not in REF.CONSUMED_SEEDS, seed
    RUN.check_seed_registration()                     # the barrier is still down
    assert RUN.H2_EXECUTION_AUTHORIZED is False       # and the gate is shut again
    with pytest.raises(RUN.H2Error, match="UNAUTHORIZED"):
        RUN.run_h2(results_path="/dev/null/x", trace_path="/dev/null/y",
                   report_path="/dev/null/z")


def test_THE_SPENT_BLOCK_CANNOT_BE_SCHEDULED_AGAIN():
    """Retirement that does not refuse the next schedule is a comment, not a state."""
    lo, hi = R.H2_ATTEMPT1_SEED_BLOCK
    for seed in (lo, hi - 1):
        task = {"seed": seed, "reference": "calib020_0001",
                "reference_sha1": "209cf2d4fd24a48553d259dd71b4954867b9473e",
                "anchor_colour": "black"}
        REF.validate_task_structure(task)              # well formed, forever
        with pytest.raises(REF.E4ReferenceError, match="RETIRED|EXPOSED"):
            REF.validate_task_executable(task)         # but no longer runnable


def test_ATTEMPT_ONES_SCHEDULE_IS_REFUSED_FOR_EXECUTION_though_it_still_PARSES():
    """The voided attempt's plan stays loadable AS A RECORD and is refused for
    EXECUTION by the registry -- never by a digest mismatch alone. Rebuilt here with
    attempt 1's seeds, which is what its records hold."""
    lo = R.H2_ATTEMPT1_SEED_BLOCK[0]
    tasks = [dict(t, seed=lo + i, rng_streams=REF.rng_stream_seeds(dict(t, seed=lo + i)))
             for i, t in enumerate(PLAN.build_tasks(PLAN.load_source_plan()))]
    assert R.L0.l0_task_digest(tasks) == R.H2_ATTEMPT1_TASK_DIGEST, \
        "attempt 1's digest must still verify, or its record is unverifiable"
    assert R.h2_full_task_digest(tasks) == R.H2_ATTEMPT1_FULL_TASK_DIGEST
    with pytest.raises(Exception, match="RETIRED|EXPOSED|retired|exposed"):
        REF.validate_schedule_executable(tasks)


def test_THE_RETRY_SCHEDULE_STILL_MATCHES_BOTH_PINS_BUT_IS_NOW_SPENT():
    """Attempt 2's schedule stays verifiable AS THE INCIDENT'S RECORD -- its 383
    preserved games must remain checkable against the schedule they came from --
    and the registry REFUSES it for execution. Rebuilt with attempt 2's seeds,
    which is what those records hold."""
    lo = R.H2_ATTEMPT2_SEED_BLOCK[0]
    tasks = [dict(t, seed=lo + i, rng_streams=REF.rng_stream_seeds(dict(t, seed=lo + i)))
             for i, t in enumerate(PLAN.build_tasks(PLAN.load_source_plan()))]
    assert R.L0.l0_task_digest(tasks) == R.H2_ATTEMPT2_TASK_DIGEST, \
        "attempt 2's digest must still verify, or the incident's record is unverifiable"
    assert R.h2_full_task_digest(tasks) == R.H2_ATTEMPT2_FULL_TASK_DIGEST
    assert R.H2_ATTEMPT2_TASK_DIGEST != R.H2_ATTEMPT1_TASK_DIGEST
    with pytest.raises(REF.E4ReferenceError, match="EXPOSED|RETIRED|retired"):
        REF.validate_schedule_executable(tasks)


def test_THE_THIRD_BLOCK_IS_ACCOUNTED_ONLY_and_the_barrier_is_SATISFIED():
    """THE SEED-PREPARATION STEP, and only that. Registering is bookkeeping, not
    permission: the block is ACCOUNTED so the registration barrier is satisfied,
    and NOT exposed and NOT retired because a reservation is not a draw. The gate
    is the separate review and it is still shut."""
    lo, hi = R.H2_SEED_BLOCK
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                 "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS"):
        assert getattr(REF, name), f"vacuous: {name} is empty"
    for seed in range(lo, hi):
        st = REF.seed_status(seed)
        assert st["accounted"], (seed, st)
        assert not (st["exposed"] or st["retired"] or st["test_only"]), (seed, st)
        assert seed not in REF.CONSUMED_SEEDS, seed
    RUN.check_seed_registration()                     # the barrier is down
    assert RUN.H2_EXECUTION_AUTHORIZED is False, (
        "registering a block ALSO opened the execution gate -- registration is "
        "bookkeeping, and permission is a separate review")
    with pytest.raises(RUN.H2Error, match="UNAUTHORIZED"):
        RUN.run_h2(results_path="/dev/null/x", trace_path="/dev/null/y",
                   report_path="/dev/null/z")


def test_THE_THIRD_SCHEDULE_MATCHES_BOTH_PINS_and_IS_EXECUTABLE_by_the_registry():
    """The other half of the registration: a block that is registered but whose
    schedule the registry still refuses would be a reservation that buys nothing.
    Both pins are recomputed from the BUILT schedule -- a pinned digest never
    checked against the artifact it pins is decoration."""
    tasks = PLAN.build_tasks(PLAN.load_source_plan())
    summary = PLAN.validate_h2_schedule(tasks)
    assert summary["seed_block"] == list(R.H2_SEED_BLOCK)
    assert summary["task_digest"] == R.H2_TASK_DIGEST
    assert R.h2_full_task_digest(tasks) == R.H2_FULL_TASK_DIGEST
    assert R.H2_TASK_DIGEST not in (R.H2_ATTEMPT1_TASK_DIGEST,
                                    R.H2_ATTEMPT2_TASK_DIGEST), "a NEW schedule"
    assert [t["seed"] for t in tasks] == list(range(*R.H2_SEED_BLOCK))
    REF.validate_schedule_executable(tasks)           # raises nothing


def _registry_without_h2():
    """The real ACCOUNTED tuple, MINUS H2's block, with the strip asserted.

    A negative control that quietly stops removing anything is a control that has
    stopped controlling -- the D1 round's lesson, in the opposite direction."""
    out = tuple(i for i in REF.ACCOUNTED_SEED_INTERVALS
                if tuple(i) != tuple(R.H2_SEED_BLOCK))
    assert len(out) < len(REF.ACCOUNTED_SEED_INTERVALS), \
        "nothing was stripped: H2's block is NOT registered, so this controls nothing"
    return out


def test_an_UNREGISTERED_block_is_still_refused(monkeypatch):
    """NEGATIVE CONTROL. Strip the block and the barrier must refuse again."""
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS", _registry_without_h2())
    with pytest.raises(RUN.H2Error, match="not registered"):
        RUN.check_seed_registration()


def test_the_registration_barrier_checks_EVERY_seed_not_the_endpoints(monkeypatch):
    """A PARTIAL registration must still refuse: all but the last seed."""
    lo, hi = R.H2_SEED_BLOCK
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        _registry_without_h2() + ((lo, hi - 1),))
    with pytest.raises(RUN.H2Error, match="not registered"):
        RUN.check_seed_registration()


def test_the_output_paths_are_CREATE_ONLY_and_must_be_two_files(tmp_path):
    r, t = tmp_path / "r.jsonl", tmp_path / "t.jsonl"
    RUN.check_output_paths(str(r), str(t), str(tmp_path / "rep.json"))
    with pytest.raises(RUN.H2Error, match="requires a trace"):
        RUN.check_output_paths(str(r), None, str(tmp_path / "rep.json"))
    with pytest.raises(RUN.H2Error, match="ONE file"):
        RUN.check_output_paths(str(r), str(r), str(tmp_path / "rep.json"))
    r.write_text("x")
    with pytest.raises(RUN.H2Error, match="already exists"):
        RUN.check_output_paths(str(r), str(t), str(tmp_path / "rep.json"))


# ───────────────────────── the incumbent identity binding ───────────────────

def test_the_frozen_identity_carries_ARGMAX_and_records_the_inert_settings():
    i = RUN.frozen_incumbent_identity()
    assert i["eval_config"]["selection_mode"] == "argmax"
    assert set(i["inert_under_argmax"]) == set(R.INERT_UNDER_ARGMAX)
    for k in R.INERT_UNDER_ARGMAX:
        assert k not in i["eval_config"], "an inert setting must not look active"


def test_an_identity_that_still_says_opening_temperature_VOIDS():
    i = RUN.frozen_incumbent_identity()
    i["eval_config"] = dict(i["eval_config"], selection_mode="opening_temperature")
    # 🔴 The generic recursive comparison ALSO reports a selection_mode mismatch, so
    # matching that word could not tell the two apart and a control on the early
    # check went unseen. The early check exists for its MESSAGE; assert it.
    with pytest.raises(RUN.H2VoidError, match="H2 IS the readout change"):
        RUN.check_incumbent_identity(i)


@pytest.mark.parametrize("field", ["reference", "reference_sha1", "plan_sha256"])
def test_an_identity_naming_another_model_VOIDS(field):
    i = RUN.frozen_incumbent_identity()
    i[field] = "not-the-frozen-one"
    with pytest.raises(RUN.H2VoidError, match=field):
        RUN.check_incumbent_identity(i)


def test_an_identity_with_a_different_simulation_budget_VOIDS():
    i = RUN.frozen_incumbent_identity()
    i["eval_config"] = dict(i["eval_config"], mcts_sims=800)
    with pytest.raises(RUN.H2VoidError, match="mcts_sims"):
        RUN.check_incumbent_identity(i)


# ───────────────────────────── the schedule barrier ─────────────────────────

def test_the_schedule_must_BE_the_frozen_736_by_digest(monkeypatch):
    monkeypatch.setattr(R, "H2_TASK_DIGEST", "0" * 64)
    with pytest.raises(RUN.H2Error, match="different schedule"):
        RUN.check_schedule(_tasks())


def test_a_SHORT_schedule_is_refused_because_a_budget_bounds_nothing_below():
    with pytest.raises(PLAN.H2PlanError, match="expected exactly"):
        RUN.check_schedule(_tasks()[:700])


# ──────────────────────── the run, on a mocked play seam ────────────────────

@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + (R.H2_SEED_BLOCK,))
    monkeypatch.setattr(REF, "validate_schedule_executable", lambda tasks: None)


def _play_factory(*, distinct=True, wins=368):
    n = {"i": 0}

    def play(*, task, identity, timeout_s):
        i = n["i"]
        n["i"] += 1
        anchor = task["anchor_colour"]
        t1j_won = i < wins
        winner = anchor if t1j_won else ("black" if anchor == "red" else "red")
        tag = i if distinct else 0
        plies = [{"ply": 7 + k, "mover": R.colour_at_ply(7 + k),
                  "move": [7 + k, tag]} for k in range(4)]
        return {"result": {"task_id": task["task_id"], "winner": winner,
                           "terminal_reason": "win",
                           "t1j_points": 1.0 if t1j_won else 0.0,
                           "plies": 10, "seed": task["seed"]},
                "plies": plies, "opening_bound": 6}
    return play


def test_a_complete_mocked_run_reports_a_parity_verdict(registered, tmp_path):
    out = RUN._run_h2_unguarded(
        tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
        trace_path=str(tmp_path / "t.jsonl"),
                          report_path=str(tmp_path / "rep.json"), play=_play_factory(wins=736))
    assert out["reported"] is True and out["outcome"] == "T1J_STRONGER"
    assert out["degeneracy_screen"]["passes"] is True


def test_a_run_whose_games_are_all_IDENTICAL_is_DEGENERATE_not_a_verdict(registered,
                                                                        tmp_path):
    out = RUN._run_h2_unguarded(
        tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
        trace_path=str(tmp_path / "t.jsonl"), report_path=str(tmp_path / "rep.json"),
        play=_play_factory(distinct=False, wins=736))
    assert out["outcome"] == "INCONCLUSIVE — DEGENERATE DESIGN"
    assert out["interval"] is None
    assert len(out["degeneracy_screen"]["failing_cells"]) == 16


def test_the_deadline_VOIDS_mid_run_and_reports_no_partial_rate(registered, tmp_path):
    """🔴 The COOPERATIVE check is what this exercises, so the supervisor is stood
    down: with the real one a non-positive remaining refuses FIRST and masks it --
    a control went NOT CAUGHT proving exactly that."""
    with pytest.raises(RUN.H2VoidError, match="deadline expired"):
        RUN._run_h2_unguarded(
            tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
            trace_path=str(tmp_path / "t.jsonl"),
                          report_path=str(tmp_path / "rep.json"), play=_play_factory(),
            deadline_s=-1, _supervisor=_NoSupervisor)


def test_the_REAL_supervisor_refuses_a_deadline_with_nothing_remaining(registered,
                                                                       tmp_path):
    """And the supervisor's own refusal is asserted separately, so standing it down
    above does not lose it: arming `setitimer` with a non-positive remaining would
    DISABLE the timer while appearing to arm it."""
    with pytest.raises(Exception) as e:
        RUN._run_h2_unguarded(
            tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
            trace_path=str(tmp_path / "t.jsonl"),
                          report_path=str(tmp_path / "rep.json"), play=_play_factory(), deadline_s=-1)
    assert "deadline" in str(e.value).lower()


def test_the_trace_and_the_results_are_both_written_and_fsynced(registered, tmp_path):
    r, t = tmp_path / "r.jsonl", tmp_path / "t.jsonl"
    RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(r), trace_path=str(t),
                          report_path=str(tmp_path / "rep.json"), play=_play_factory())
    trace = [json.loads(l) for l in t.read_text().splitlines()]
    assert trace[0]["event"] == "run_start" and trace[-1]["event"] == "run_end"
    assert trace[-1]["verdict"] == "OK" and trace[-1]["games_completed"] == 736
    rows = [json.loads(l) for l in r.read_text().splitlines()]
    assert rows[0]["record_type"] == "header"
    assert rows[0]["selection_mode"] == "argmax"
    assert sum(1 for x in rows if x["record_type"] == "task_result") == 736


def test_a_malformed_ply_record_VOIDS_the_run_rather_than_being_counted(registered,
                                                                        tmp_path):
    def play(*, task, identity, timeout_s):
        anchor = task["anchor_colour"]
        return {"result": {"task_id": task["task_id"], "winner": anchor,
                           "terminal_reason": "win", "t1j_points": 1.0,
                           "plies": 10, "seed": task["seed"]},
                "plies": [{"ply": 8, "mover": R.colour_at_ply(8),
                           "move": [8, 1]}],       # the FIRST ply is missing
                "opening_bound": 6}
    with pytest.raises(RUN.H2VoidError):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(tmp_path / "t.jsonl"),
                          report_path=str(tmp_path / "rep.json"), play=play)


# ────────────────────────────────── the wrapper ─────────────────────────────

def test_the_wrapper_refuses_with_the_shut_gate_and_verifies_it_closed(capsys):
    assert CMD.gate_is_open() is False
    assert CMD.main([]) == CMD.EXIT_UNAUTHORIZED
    assert "NOT AUTHORIZED" in capsys.readouterr().err


def test_THE_OUTPUT_DESTINATION_IS_NOT_A_SPENT_ATTEMPTS_DIRECTORY():
    """🔴 THE PRE-RUN VERIFICATION FOR ATTEMPT 3 FOUND THIS. `OUT_DIR` still pointed
    at attempt 1's directory, where two of the three outputs already exist -- that
    VOIDed run's records. Create-only would have refused the launch BEFORE spawning,
    so the cost was an authorization spent on a run that could not start rather than
    a lost file. Nothing bound the destination, which is why it went unnoticed
    through a registration and a re-proof.

    Three invariants, none of which depends on what exists on disk right now: the
    three outputs are DISTINCT, none of them sits in a directory holding a spent
    attempt's records, and the spent list is not empty."""
    import os
    defaults = (CMD.DEFAULT_RESULTS, CMD.DEFAULT_TRACE, CMD.DEFAULT_REPORT)
    assert len(set(defaults)) == 3, defaults
    assert CMD.SPENT_OUT_DIRS, "vacuous: no spent directory is named"
    for spent in CMD.SPENT_OUT_DIRS:
        assert os.path.isdir(spent), f"{spent} is named as spent but does not exist"
        for d in defaults:
            assert not d.startswith(spent.rstrip("/") + "/"), (d, spent)
    for d in defaults:
        assert d.startswith(CMD.OUT_DIR.rstrip("/") + "/"), (d, CMD.OUT_DIR)


def test_the_wrapper_has_NO_runner_source_flag():
    """🔴 H1's lesson: an overridable path lets a decoy be restored while the real
    gate stays open, and the exit code then reports success."""
    with pytest.raises(SystemExit):
        CMD._parser().parse_args(["--runner-source", "/tmp/decoy.py"])
    assert "_runner_source" in inspect.signature(CMD.main).parameters


def test_restore_gate_is_TRUE_when_the_gate_is_already_closed():
    assert CMD.restore_gate() is True


def test_restore_gate_REWRITES_an_open_gate_and_verifies_it(tmp_path):
    decoy = tmp_path / "runner.py"
    decoy.write_text("x = 1\nH2_EXECUTION_AUTHORIZED = True\ny = 2\n")
    assert CMD.restore_gate(str(decoy)) is True
    assert "H2_EXECUTION_AUTHORIZED = False" in decoy.read_text()
    assert "= True" not in decoy.read_text()


def test_restore_gate_is_FALSE_when_it_cannot_verify(tmp_path):
    missing = tmp_path / "nope.py"
    assert CMD.restore_gate(str(missing)) is False
    two = tmp_path / "two.py"
    two.write_text("H2_EXECUTION_AUTHORIZED = True\nH2_EXECUTION_AUTHORIZED = True\n")
    assert CMD.restore_gate(str(two)) is False


def test_a_failed_restoration_becomes_the_wrappers_OWN_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(CMD, "restore_gate", lambda *a, **k: False)
    assert CMD.main([]) == CMD.EXIT_GATE_NOT_RESTORED
    assert "BY HAND" in capsys.readouterr().err


def test_THE_FINALLY_PATH_also_restores_and_reports_its_own_failure(
        monkeypatch, capsys, tmp_path):
    """🔴 A control proved the earlier test blind here: with the gate shut, `main`
    returns on the UNAUTHORIZED branch and never reaches the `finally`, so
    disabling the finally's check changed nothing. This drives the OTHER path --
    gate open, restoration failing -- which is the one that matters, because it is
    the path a real run takes.

    🔑 HERMETIC, and it was not always. With the gate forced open this reaches the
    output precheck, and it used to be REFUSED there because the frozen defaults
    pointed at attempt 1's directory, where two outputs already existed -- so it
    never reached `supervise` and the whole test silently depended on a destination
    being occupied. Repointing the defaults at a FREE directory made it spawn a
    real supervised worker instead. Its subject is the `finally`, not the
    destination and not the supervisor, so both are supplied here: tmp outputs and
    a stubbed `supervise`. Otherwise this test's path changes the day a match runs.
    """
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "restore_gate", lambda *a, **k: False)
    monkeypatch.setattr(CMD, "supervise", lambda *a, **k: {
        "exit_code": 0, "timed_out": False, "interrupted": False,
        "group_cleared": True})
    argv = ["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl"),
            "--report", str(tmp_path / "rep.json")]
    assert CMD.main(argv) == CMD.EXIT_GATE_NOT_RESTORED
    assert "BY HAND" in capsys.readouterr().err
    assert not list(tmp_path.iterdir()), "the stubbed supervisor wrote nothing"


def test_the_finally_path_restores_a_REAL_open_gate_after_a_refusal(monkeypatch, tmp_path):
    """The wrapper's own promise: restoration runs after EVERY exit, including the
    refusal path, and the file is left closed."""
    decoy = tmp_path / "runner.py"
    decoy.write_text("H2_EXECUTION_AUTHORIZED = True\n")
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "supervise",
                        lambda *a, **k: {"exit_code": CMD.EXIT_REFUSED, "timed_out": False,
                                         "interrupted": False, "group_cleared": True})
    code = CMD.main(["--results", str(tmp_path / "r.jsonl"),
                     "--trace", str(tmp_path / "t.jsonl")], _runner_source=str(decoy))
    assert code == CMD.EXIT_REFUSED
    assert decoy.read_text() == "H2_EXECUTION_AUTHORIZED = False\n"


def test_the_worker_is_SUPERVISED_in_its_own_group_under_an_OUTER_cap(monkeypatch,
                                                                     tmp_path):
    """🔴 THE DEADLINE IS POLLED BETWEEN GAMES, so one blocked game could overrun
    it indefinitely. The wrapper caps the WORKER and kills its whole group."""
    seen = {}

    def fake_supervise(cmd, *, timeout_s, kill_grace_s, interrupt_grace_s,
                       pass_fds=()):
        seen.update(cmd=cmd, timeout_s=timeout_s, pass_fds=list(pass_fds))
        return {"exit_code": 0, "timed_out": False, "interrupted": False,
                "group_cleared": True}
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "supervise", fake_supervise)
    decoy = tmp_path / "runner.py"
    decoy.write_text("H2_EXECUTION_AUTHORIZED = True\n")
    CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl")],
             _runner_source=str(decoy))
    assert "--worker" in seen["cmd"] and CMD.MODULE in seen["cmd"]
    assert seen["timeout_s"] == RUN.RUN_DEADLINE_S + CMD.SUPERVISOR_GRACE_S


@pytest.mark.parametrize("r,want", [
    ({"exit_code": 0, "timed_out": True, "interrupted": False, "group_cleared": True},
     "EXIT_TIMEOUT"),
    ({"exit_code": 0, "timed_out": False, "interrupted": True, "group_cleared": True},
     "EXIT_INTERRUPTED"),
    ({"exit_code": 0, "timed_out": False, "interrupted": False, "group_cleared": False},
     "EXIT_CLEANUP_FAILED"),
])
def test_a_timeout_an_interrupt_and_a_SURVIVING_DESCENDANT_each_get_their_own_code(
        monkeypatch, tmp_path, r, want):
    """A surviving child must never accompany a success -- the runtime
    requalification's own lesson."""
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "supervise", lambda *a, **k: r)
    decoy = tmp_path / "runner.py"
    decoy.write_text("H2_EXECUTION_AUTHORIZED = True\n")
    code = CMD.main(["--results", str(tmp_path / "r.jsonl"),
                     "--trace", str(tmp_path / "t.jsonl")], _runner_source=str(decoy))
    assert code == getattr(CMD, want)


def test_restore_gate_is_FALSE_when_the_READBACK_disagrees(monkeypatch, tmp_path):
    """🔴 The final `return` was unreachable from every earlier test: a missing
    file and a two-gate file both refuse before the write. Here the write is
    silently dropped, so the file still holds an OPEN gate afterwards -- only
    reading it back can tell."""
    decoy = tmp_path / "runner.py"
    decoy.write_text("H2_EXECUTION_AUTHORIZED = True\n")
    monkeypatch.setattr(CMD.os, "replace", lambda *a, **k: None)
    assert CMD.restore_gate(str(decoy)) is False
    assert "= True" in decoy.read_text(), "the write was meant to be dropped"


# ─────────────────────── nothing here can execute anything ──────────────────

@pytest.mark.parametrize("mod", [R, PLAN, RUN, CMD])
def test_no_H2_module_WRITES_a_registry_or_runs_a_bare_subprocess(mod):
    import pathlib
    src = pathlib.Path(mod.__file__).read_text()
    for forbidden in ("ACCOUNTED_SEED_INTERVALS = ", "subprocess.run("):
        assert forbidden not in src, (mod.__name__, forbidden)


def test_NOTHING_EFFECTFUL_IS_IMPORTED_AT_MODULE_LEVEL():
    """🔑 The production seam names the evaluator loader, the toolchain and the
    compile step -- it must, or there is no production path. What matters is that
    importing the module touches NONE of them: every such import sits inside a
    function, checked by AST rather than by grepping."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(RUN.__file__).read_text())
    # 🔴 `from . import e4_screen_runner` has module=None and the NAME in `names`,
    # so reading only `.module` missed an eager import entirely -- a control proved
    # it. Both are collected now, plus plain `import x`.
    top = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            top.add(n.module or "")
            top.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            top.update(a.name for a in n.names)
    for effectful in ("d1_probe", "e4_screen_runner", "e4_screen_command",
                      "t1j_toolchain", "e4_screen_integration",
                      "twixtbot_g3_reference"):
        assert not any(effectful in m for m in top), (effectful, sorted(top))


# ─────────────── the four production defects the review reproduced ──────────

def test_a_DANGLING_SYMLINK_at_an_output_path_is_REFUSED(tmp_path):
    """🔴 H1's own correction, which I had repeated as a defect. `os.path.exists`
    FOLLOWS the link, so a dangling symlink reads as absent; `O_EXCL` then fails on
    the link itself and a create-only guarantee becomes a mid-run error."""
    r, t = tmp_path / "r.jsonl", tmp_path / "t.jsonl"
    r.symlink_to(tmp_path / "nowhere")
    assert not r.exists() and r.is_symlink(), "the fixture must be a DANGLING link"
    with pytest.raises(RUN.H2Error, match="already exists"):
        RUN.check_output_paths(str(r), str(t), str(tmp_path / "rep.json"))


def test_INVALID_RESULTS_are_REFUSED_not_reported_as_degenerate():
    """🔴 Screening first let a schedule of INVALID results be reported as
    `DEGENERATE DESIGN` -- a data-integrity failure wearing a design-validity
    name. The bind now speaks first."""
    tasks = _tasks()
    bad = _results(tasks, t1j_wins=368)
    for row in bad:
        row["t1j_points"] = 0.25            # a score no rule can produce
    out = R.h2_report(bad, tasks,
                      _per_game({CELLS[0]: 1}, task_ids=[t["task_id"] for t in tasks]),
                      task_digest=R.H2_TASK_DIGEST)
    assert out["reported"] is False
    assert out["outcome"] == "REFUSED", out["outcome"]
    assert out["degeneracy_screen"] is None, "the screen must not have spoken"


def test_THE_PLIES_AND_DIGESTS_ARE_PERSISTED_so_the_screen_can_be_recomputed(registered,
                                                                             tmp_path):
    """🔴 Only the task result was written, so the reported diversity could not be
    recomputed by anyone who was not there."""
    r = tmp_path / "r.jsonl"
    RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(r),
                          trace_path=str(tmp_path / "t.jsonl"),
                          report_path=str(tmp_path / "rep.json"), play=_play_factory())
    rows = [json.loads(l) for l in r.read_text().splitlines()]
    kinds = {x["record_type"] for x in rows}
    assert {"header", "task_result", "ply", "transcript"} <= kinds, kinds
    tr = [x for x in rows if x["record_type"] == "transcript"]
    assert len(tr) == 736
    assert all(len(x["transcript_digest"]) == 64 for x in tr)
    # the persisted digests reproduce the screen's own counts, independently
    recomputed = R.degeneracy_screen(
        [{"task_id": x["task_id"], "opening": x["opening"],
          "colour_arm": x["colour_arm"], "transcript_digest": x["transcript_digest"]}
         for x in tr],
        canonical_task_ids=[t["task_id"] for t in _tasks()])
    assert recomputed["passes"] is True
    assert all(c["n_distinct"] == 46 for c in recomputed["cells"])


def test_ANY_mid_run_failure_still_writes_run_end_VOID(registered, tmp_path):
    """🔴 Only the deadline wrote one, so a crash left a trace that stopped
    mid-sentence and could not say the run was void."""
    def exploding(*, task, identity, timeout_s):
        raise RuntimeError("the instrument fell over")
    t = tmp_path / "t.jsonl"
    with pytest.raises(RUN.H2VoidError):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(t), report_path=str(tmp_path / "rep.json"), play=exploding)
    last = json.loads(t.read_text().splitlines()[-1])
    assert last == {"error": "RuntimeError", "event": "run_end",
                    "games_completed": 0, "verdict": "VOID"}


class _NoSupervisor:
    """A no-op stand-in, so the COOPERATIVE deadline check is what is exercised.
    🔴 With the real supervisor a non-positive remaining refuses FIRST, masking the
    between-games check entirely -- a control proved it by going NOT CAUGHT."""
    def __init__(self, deadline): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False


def test_the_DEADLINE_void_also_leaves_a_run_end_VOID(registered, tmp_path):
    t = tmp_path / "t.jsonl"
    with pytest.raises(RUN.H2VoidError, match="deadline expired"):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(t), report_path=str(tmp_path / "rep.json"), play=_play_factory(), deadline_s=-1,
                              _supervisor=_NoSupervisor)
    last = json.loads(t.read_text().splitlines()[-1])
    assert last["event"] == "run_end" and last["verdict"] == "VOID"


def test_a_forged_reference_sha256_is_REFUSED_though_the_DIGEST_still_matches(registered):
    """🔴 The design digest covers the dimensions only: a forged pin or forged
    rng_streams kept the same digest and passed."""
    tasks = [dict(t) for t in _tasks()]
    tasks[9]["reference_sha256"] = "f" * 64
    with pytest.raises(RUN.H2Error, match="full-field digest|reference_sha256"):
        RUN.check_schedule(tasks)


def test_THE_FULL_FIELD_DIGEST_ITSELF_distinguishes_a_forged_field():
    """🔴 A control that projected the digest down to `task_id` went NOT CAUGHT:
    any change to the function breaks the pin comparison, so the schedule was
    refused for the wrong reason. The DIGEST is exercised directly here."""
    tasks = _tasks()
    forged = [dict(t) for t in tasks]
    forged[9]["reference_sha256"] = "f" * 64
    assert R.h2_full_task_digest(tasks) != R.h2_full_task_digest(forged)
    streams = [dict(t) for t in tasks]
    streams[3]["rng_streams"] = {"search": 1, "readout": 2}
    assert R.h2_full_task_digest(tasks) != R.h2_full_task_digest(streams)
    assert R.h2_full_task_digest(tasks) == R.H2_FULL_TASK_DIGEST


def test_the_WORKER_persists_the_report_and_returns_its_mapped_code(monkeypatch,
                                                                    tmp_path, capsys):
    """🔴 A control that made `worker_main` print-and-return-0 went NOT CAUGHT,
    because the persistence test called `_persist_and_classify` directly. The
    worker path itself is driven here."""
    cap = CMD._make_capability()
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(RUN, "run_h2",
                        lambda **kw: {"reported": False, "outcome": "REFUSED",
                                      "reason": "bad rows"})
    report = tmp_path / "09_report.json"
    report.write_text('{"reported": false, "outcome": "REFUSED"}')   # the runner's
    code = CMD.worker_main(["--worker", "--capability-fd", str(cap),
                            "--results", str(tmp_path / "r.jsonl"),
                            "--trace", str(tmp_path / "t.jsonl"),
                            "--report", str(report)])
    assert code == CMD.EXIT_REFUSED
    assert json.loads(report.read_text())["outcome"] == "REFUSED"


def test_forged_rng_streams_are_REFUSED(registered):
    tasks = [dict(t) for t in _tasks()]
    tasks[3]["rng_streams"] = {"search": 1, "readout": 2}
    with pytest.raises(RUN.H2Error, match="full-field digest|rng_streams"):
        RUN.check_schedule(tasks)


# ══════ the production setup, DRIVEN through mocked effectful boundaries ═════
# 🔴 AST inspection proved the seam was WIRED; it could not prove the wiring
# WORKS. These tests replace only the effectful boundaries -- toolchain, compile,
# evaluator, factories, harness, cleanup -- and then RUN the seam, so the clock,
# the argmax config, the cleanup and the game loop are reached together. No JVM
# starts, no model loads, no game is played.

@pytest.fixture
def seam_boundaries(monkeypatch, tmp_path):
    from scripts.GPU.alphazero import d1_probe as D1
    from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
    from scripts.GPU.alphazero import e4_screen_integration as INT
    from scripts.GPU.alphazero import e4_screen_runner as HARNESS
    from scripts.GPU.alphazero import t1j_toolchain as TC
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3

    seen = {"compile_deadlines": [], "configs": [], "cleanups": 0, "play_calls": [],
            "loads": 0, "real_build": G3.build_reference_agent}

    monkeypatch.setattr(TC, "verified_paths",
                        lambda: {"jar": str(tmp_path / "t1j.jar"),
                                 "jdk_home": str(tmp_path / "jdk")})
    monkeypatch.setattr(D1, "_default_compile",
                        lambda deadline, paths: seen["compile_deadlines"].append(deadline))
    monkeypatch.setattr(INT, "T1jRuntime", lambda **kw: ("runtime", kw))
    monkeypatch.setattr(INT, "IntegrationContext", lambda: "ctx")
    monkeypatch.setattr(INT, "make_state_factory", lambda openings, ctx: "state_factory")
    monkeypatch.setattr(INT, "make_binder", lambda runtime, ctx: "binder")

    def fake_agent_factory(*, runtime, ctx, evaluator, t1j_timeout_s, reference_build):
        seen["reference_build"] = reference_build
        return "agent_factory"
    monkeypatch.setattr(INT, "make_agent_factory", fake_agent_factory)
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator",
                        lambda root: seen.__setitem__("loads", seen["loads"] + 1) or "evaluator")
    monkeypatch.setattr(SCREEN_CMD, "_default_cleanup",
                        lambda: seen.__setitem__("cleanups", seen["cleanups"] + 1))
    monkeypatch.setattr(G3, "build_reference_agent",
                        lambda **kw: seen["configs"].append(kw.get("config")) or "agent")

    def fake_play_task(*, task, agent_for, state_factory, binder, rec, ply_cap):
        seen["play_calls"].append({"task_id": task["task_id"], "agent_for": agent_for,
                                   "state_factory": state_factory, "binder": binder,
                                   "ply_cap": ply_cap})
        rec.emit({"record_type": "opening_bound", "task_id": task["task_id"], "ply": 6})
        for k in range(4):
            ply = 7 + k
            rec.emit({"record_type": "ply", "task_id": task["task_id"], "ply": ply,
                      "mover": R.colour_at_ply(ply), "move": [ply, 0]})
        return {"winner": task["anchor_colour"], "terminal_reason": "win",
                "plies": 10, "t1j_points": 1.0, "agents_built": 2}
    monkeypatch.setattr(HARNESS, "play_task", fake_play_task)
    # the test may put the real builder back; monkeypatch restores it either way
    monkeypatch.setattr(G3, "build_reference_agent", G3.build_reference_agent)

    # 🔑 THE ONE PLACE A SEAM TEST RELAXES AUTHORIZATION, and it is safe only
    # BECAUSE of the mocks above. The seam re-reads the gate (the second check added
    # after the 2026-09-12 incident), so a seam test cannot run without opening it --
    # and the line between this fixture and the control that played 383 real games is
    # that every production act here has been replaced. That is ASSERTED, not
    # assumed: `assert_production_acts_are_inert` raises if any of the four is still
    # the genuine function, so this fixture cannot become a way to reach production.
    SCREEN_CMD.assert_production_acts_are_inert("the H2 seam fixture", (
        (TC, "verified_paths"), (D1, "_default_compile"),
        (SCREEN_CMD, "_default_load_evaluator"), (HARNESS, "play_task")))
    monkeypatch.setattr(RUN, "H2_EXECUTION_AUTHORIZED", True)
    return seen


def _started_deadline():
    from scripts.GPU.alphazero import d1_probe as D1
    return D1.Deadline(RUN.RUN_DEADLINE_S).start()


def test_the_seam_HANDS_COMPILE_THE_RUNS_OWN_STARTED_CLOCK(seam_boundaries, tmp_path):
    """🔴 THE DEFECT THIS CLOSES. A fresh `Deadline` has no origin, so the first
    production setup was guaranteed to refuse -- after creating the classes
    directory -- with "the run deadline was never started"."""
    d = _started_deadline()
    play = RUN._production_play(str(tmp_path / "r.jsonl"), d)
    play(task=_tasks()[0], identity={}, timeout_s=120)
    assert seam_boundaries["compile_deadlines"] == [d]
    assert d.started is True


def test_the_seam_REFUSES_an_unstarted_or_absent_clock(seam_boundaries, tmp_path):
    from scripts.GPU.alphazero import d1_probe as D1
    for bad in (None, D1.Deadline(RUN.RUN_DEADLINE_S)):        # absent, then unstarted
        play = RUN._production_play(str(tmp_path / "r.jsonl"), bad)
        with pytest.raises(RUN.H2Error, match="STARTED deadline"):
            play(task=_tasks()[0], identity={}, timeout_s=120)
    assert seam_boundaries["compile_deadlines"] == [], "nothing compiled"


def test_the_incumbent_IS_ACTUALLY_BUILT_WITH_AN_ARGMAX_CONFIG(seam_boundaries, tmp_path):
    """The whole gameplay change, reached rather than read: the config handed to
    the incumbent's builder must carry `selection_mode = "argmax"`."""
    play = RUN._production_play(str(tmp_path / "r.jsonl"), _started_deadline())
    play(task=_tasks()[0], identity={}, timeout_s=120)
    build = seam_boundaries["reference_build"]
    build(_tasks()[0], "evaluator")                 # what the harness would call
    assert len(seam_boundaries["configs"]) == 1
    cfg = seam_boundaries["configs"][0]
    assert cfg.selection_mode == "argmax"
    assert cfg.mcts_sims == 400, "only the READOUT changes"


def test_THE_CLEANUP_RUNS_AFTER_EVERY_GAME_including_a_failed_one(seam_boundaries,
                                                                  tmp_path):
    """🔴 H1 clears MLX state after every game; H2 plays 736 and called it NEVER."""
    from scripts.GPU.alphazero import e4_screen_runner as HARNESS
    play = RUN._production_play(str(tmp_path / "r.jsonl"), _started_deadline())
    for t in _tasks()[:3]:
        play(task=t, identity={}, timeout_s=120)
    assert seam_boundaries["cleanups"] == 3 == play.cleanups

    def exploding(**kw):
        raise RuntimeError("the game fell over")
    HARNESS.play_task = exploding
    with pytest.raises(RuntimeError):
        play(task=_tasks()[3], identity={}, timeout_s=120)
    assert seam_boundaries["cleanups"] == 4, "a failed game must still clean up"


def test_the_evaluator_is_loaded_ONCE_for_the_whole_run(seam_boundaries, tmp_path):
    play = RUN._production_play(str(tmp_path / "r.jsonl"), _started_deadline())
    for t in _tasks()[:5]:
        play(task=t, identity={}, timeout_s=120)
    assert seam_boundaries["loads"] == 1
    assert len(seam_boundaries["play_calls"]) == 5


def test_the_seam_RETURNS_the_shape_the_transcript_needs(seam_boundaries, tmp_path):
    play = RUN._production_play(str(tmp_path / "r.jsonl"), _started_deadline())
    out = play(task=_tasks()[0], identity={}, timeout_s=120)
    assert out["opening_bound"] == 6 and len(out["plies"]) == 4
    t = R.transcript(out["plies"], out["result"], opening_bound=out["opening_bound"])
    assert len(R.transcript_digest(t)) == 64


# ───────────── the report is PERSISTED, and the outcome is MAPPED ────────────

@pytest.mark.parametrize("report,want", [
    ({"reported": True, "outcome": "T1J_STRONGER"}, "EXIT_COMPLETED"),
    ({"reported": True, "outcome": "INCONCLUSIVE"}, "EXIT_COMPLETED"),
    ({"reported": False, "outcome": "INCONCLUSIVE — DEGENERATE DESIGN",
      "reason": "one cell"}, "EXIT_DEGENERATE"),
    ({"reported": False, "outcome": "CAP_SATURATED_NO_RATE", "reason": "caps"},
     "EXIT_NO_RATE"),
    ({"reported": False, "outcome": "REFUSED", "reason": "bad rows"}, "EXIT_REFUSED"),
])
def test_every_outcome_gets_ITS_OWN_exit_code(tmp_path, report, want):
    """🔴 A refusal once exited 0 as COMPLETED. Each outcome is now mapped."""
    path = tmp_path / "09_report.json"
    path.write_text(json.dumps(report))          # the RUNNER wrote it
    assert CMD._persist_and_classify(report, str(path)) == getattr(CMD, want)


def test_a_MISSING_report_file_is_VOID_not_a_completed_run(tmp_path):
    """The classifier verifies its own artifact: a verdict with no durable record
    is a claim, not a result."""
    code = CMD._persist_and_classify({"reported": True, "outcome": "T1J_STRONGER"},
                                     str(tmp_path / "absent.json"))
    assert code == CMD.EXIT_VOID


def test_THE_RUNNER_WRITES_THE_REPORT_BEFORE_THE_TERMINAL_OK(registered, tmp_path):
    """🔴 `run_end/OK` was committed BEFORE the report existed, so a report failure
    exited VOID while the durable trace said OK."""
    rep, tr = tmp_path / "rep.json", tmp_path / "t.jsonl"
    out = RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                                trace_path=str(tr), report_path=str(rep),
                                play=_play_factory(wins=736))
    assert json.loads(rep.read_text())["outcome"] == out["outcome"]
    last = json.loads(tr.read_text().splitlines()[-1])
    assert last["event"] == "run_end" and last["verdict"] == "OK"
    assert last["outcome"] == out["outcome"], "the trace names the verdict it committed"


def test_A_REPORT_THAT_CANNOT_BE_WRITTEN_LEAVES_NO_OK_TRACE(registered, tmp_path):
    """The ordering, asserted at the EFFECT, through the only case the preflight
    cannot cover: the report path is free when the run starts and OCCUPIED by the
    time the run finishes. Six hours pass in production, so this is the real race
    -- and the trace must then say VOID, never OK.
    """
    rep, tr = tmp_path / "rep.json", tmp_path / "t.jsonl"
    base = _play_factory(wins=736)

    def play(*, task, identity, timeout_s):
        out = base(task=task, identity=identity, timeout_s=timeout_s)
        if task["task_id"].startswith("h2match-735"):
            rep.write_text("{}")          # someone else got there first
        return out

    with pytest.raises(RUN.H2VoidError):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(tr), report_path=str(rep), play=play)
    assert rep.read_text() == "{}", "an existing report was overwritten"
    # 🔴 NO `OK` ANYWHERE, not merely a VOID at the end. A control that emitted an
    # early OK and then let the failure append its VOID satisfied a last-line
    # assertion while the trace carried both verdicts -- which is the disagreement
    # this test exists to forbid.
    ends = [json.loads(l) for l in tr.read_text().splitlines()
            if json.loads(l).get("event") == "run_end"]
    assert [e["verdict"] for e in ends] == ["VOID"], ends


def test_an_EXISTING_report_is_refused_by_the_PREFLIGHT_before_any_game(registered,
                                                                       tmp_path):
    """And the ordinary case is caught before a single game is played."""
    rep = tmp_path / "rep.json"
    rep.write_text("{}")
    with pytest.raises(RUN.H2Error, match="report path already exists"):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(tmp_path / "t.jsonl"), report_path=str(rep),
                              play=lambda **kw: pytest.fail("a game was played"))


def test_ALL_THREE_OUTPUTS_are_preflighted_before_any_game(tmp_path):
    """🔴 THE DEFECT THIS CLOSES. The report was not checked, so an existing or
    aliased report path was discovered only AFTER 736 games -- spending the whole
    seed block for a knowable path error."""
    r, t, rep = (tmp_path / "r.jsonl"), (tmp_path / "t.jsonl"), (tmp_path / "rep.json")
    RUN.check_output_paths(str(r), str(t), str(rep))
    with pytest.raises(RUN.H2Error, match="requires a report path"):
        RUN.check_output_paths(str(r), str(t), None)
    rep.write_text("{}")
    with pytest.raises(RUN.H2Error, match="report path already exists"):
        RUN.check_output_paths(str(r), str(t), str(rep))


@pytest.mark.parametrize("pair", [("results", "report"), ("trace", "report")])
def test_NO_TWO_OUTPUTS_MAY_BE_THE_SAME_FILE(tmp_path, pair):
    paths = {"results": str(tmp_path / "r.jsonl"), "trace": str(tmp_path / "t.jsonl"),
             "report": str(tmp_path / "rep.json")}
    paths[pair[1]] = "./" + os.path.relpath(paths[pair[0]])   # an ALIAS of the other
    with pytest.raises(RUN.H2Error, match="name ONE file"):
        RUN.check_output_paths(paths["results"], paths["trace"], paths["report"])


# ─────────────────── the worker is not a public bypass ──────────────────────

def test_WORKER_REFUSES_without_the_supervisors_CAPABILITY(monkeypatch, capsys):
    """🔴 `--worker` ran the match in-process, unbounded, outside the gate
    restoration boundary. The first fix used `H2_SUPERVISED_WORKER=1`, which any
    caller could set -- no provenance at all, and it contradicted this module's own
    claim that no environment variable reaches the path."""
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(RUN, "run_h2", lambda **kw: pytest.fail("the match RAN"))
    assert CMD.worker_main(["--worker"]) == CMD.EXIT_REFUSED
    assert "not a way to run H2 by hand" in capsys.readouterr().err


def test_NO_ENVIRONMENT_VARIABLE_reaches_the_worker_path():
    """The claim in the source, checked: nothing here reads os.environ."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(CMD.__file__).read_text())
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and n.attr in ("environ", "getenv")]
    assert not reads, "an environment variable is being read"


def test_THE_CAPABILITY_IS_A_PIPE_AND_IS_SINGLE_USE():
    """Single-use because the pipe is DRAINED, not because the descriptor is closed.

    ⚠ A control that duplicated the descriptor instead of closing it went NOT
    CAUGHT, and it was right to: a second read finds EOF and an empty token either
    way. The write end is closed at creation, so the token can be read exactly
    once -- that is the property, and it is what this asserts.
    """
    import stat
    fd = CMD._make_capability()
    assert stat.S_ISFIFO(os.fstat(fd).st_mode), "it must be an anonymous pipe"
    dup = os.dup(fd)
    assert CMD._consume_capability(fd) is True
    assert CMD._consume_capability(dup) is False, "the token is drained, not reusable"


def test_the_capability_is_RANDOM_and_carries_no_path():
    a, b = CMD._make_capability(), CMD._make_capability()
    ta = os.fdopen(os.dup(a)).read()
    tb = os.fdopen(os.dup(b)).read()
    assert ta != tb and len(ta) == CMD.CAPABILITY_BYTES * 2
    assert all(c in "0123456789abcdef" for c in ta)
    for fd in (a, b):
        CMD._consume_capability(fd)


@pytest.mark.parametrize("bad", ["", "short", "x" * 10, "g" * 64, "A" * 64])
def test_a_FORGED_capability_is_refused_INCLUDING_one_of_the_RIGHT_LENGTH(bad):
    """🔴 THE DEFECT THIS CLOSES. The old test tried only empty and short values, so
    it missed the real forgery: ANY file of exactly 64 characters was accepted,
    because the token was never compared with anything. Length alone is not a
    check -- the content must be what the supervisor writes."""
    r_fd, w_fd = os.pipe()
    with os.fdopen(w_fd, "w") as fh:
        fh.write(bad)
    assert CMD._consume_capability(r_fd) is False, f"{bad!r} was accepted"


def test_NOTHING_IS_EVER_DELETED_by_the_capability_check(tmp_path):
    """🔴 THE DESTRUCTIVE FAULT. The old check UNLINKED whatever path it was handed,
    valid or not, so `--worker --capability <ordinary-file>` deleted that file while
    the gate was shut. There is no path now -- and the flag that took one is gone."""
    victim = tmp_path / "important.txt"
    victim.write_text("x" * 64)          # exactly the length the old check accepted
    flags = [s for a in CMD._parser()._actions for s in a.option_strings]
    assert "--capability" not in flags, flags
    assert "--capability-fd" in flags
    with pytest.raises(SystemExit):      # the path form cannot even be expressed
        CMD._parser().parse_args(["--capability", str(victim)])
    assert victim.exists() and victim.read_text() == "x" * 64


def test_the_capability_check_TAKES_A_DESCRIPTOR_not_a_path():
    import ast
    import inspect as _i
    import textwrap
    assert list(_i.signature(CMD._consume_capability).parameters) == ["fd"]
    # 🔴 BY AST, NOT BY GREP: the docstring SAYS "unlinked" on purpose, describing
    # the fault it closes, and a substring check failed on its own explanation.
    fn = ast.parse(textwrap.dedent(_i.getsource(CMD._consume_capability))).body[0]
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert not [c for c in calls if getattr(c.func, "attr", "") in ("unlink", "remove",
                                                                    "rmdir", "rmtree")]


def test_a_capability_descriptor_that_does_not_exist_is_refused():
    assert CMD._consume_capability(9999) is False
    assert CMD._consume_capability(None) is False


def test_the_supervisor_HANDS_the_child_a_capability_and_leaves_none_behind(monkeypatch,
                                                                           tmp_path):
    seen = {}

    def fake_supervise(cmd, **kw):
        seen["cmd"] = list(cmd)
        i = cmd.index("--capability-fd")
        seen["fd"] = int(cmd[i + 1])
        seen["passed"] = list(kw.get("pass_fds", ()))
        seen["token"] = os.fdopen(os.dup(seen["fd"])).read()
        return {"exit_code": 0, "timed_out": False, "interrupted": False,
                "group_cleared": True}
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "supervise", fake_supervise)
    decoy = tmp_path / "runner.py"
    decoy.write_text("H2_EXECUTION_AUTHORIZED = True\n")
    CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl"),
              "--report", str(tmp_path / "rep.json")], _runner_source=str(decoy))
    assert seen["passed"] == [seen["fd"]], "the descriptor must be INHERITED"
    assert len(seen["token"]) == CMD.CAPABILITY_BYTES * 2
    with pytest.raises(OSError):
        os.fstat(seen["fd"])             # closed after the run; none outlives it


# ──────────────── the interrupt contract: OK / VOID / INTERRUPTED ───────────

def test_an_OPERATOR_INTERRUPT_records_INTERRUPTED_not_VOID(registered, tmp_path):
    """🔴 Every BaseException wrote VOID, so the durable trace said the instrument
    failed while the wrapper's exit code said the operator stopped it."""
    def interrupting(*, task, identity, timeout_s):
        raise KeyboardInterrupt()
    t = tmp_path / "t.jsonl"
    with pytest.raises(KeyboardInterrupt):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(t), report_path=str(tmp_path / "rep.json"), play=interrupting)
    last = json.loads(t.read_text().splitlines()[-1])
    assert last["verdict"] == "INTERRUPTED", last
    assert last["error"] == "KeyboardInterrupt"


def test_the_run_ARMS_the_deadline_so_a_BLOCKED_game_can_be_cut_off(registered,
                                                                    tmp_path):
    """The cooperative check runs BETWEEN games; the supervisor arms SIGALRM from
    the SAME started deadline so a hung query is cut off by the run's own clock."""
    armed = {}

    class _Sup:
        def __init__(self, deadline):
            armed["deadline"] = deadline
        def __enter__(self):
            armed["entered"] = True
        def __exit__(self, *a):
            armed["exited"] = True
            return False
    RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                          trace_path=str(tmp_path / "t.jsonl"),
                          report_path=str(tmp_path / "rep.json"), play=_play_factory(),
                          _supervisor=_Sup)
    assert armed["entered"] and armed["exited"]
    assert armed["deadline"].started is True


def test_A_PARTIAL_OUTPUT_ACQUISITION_REFUSES_CLEANLY(registered, tmp_path,
                                                      monkeypatch):
    """A failure on the SECOND output must VOID, play nothing, and leave the
    descriptor it already took registered for close.

    ⚠ WHAT THIS CANNOT SHOW, stated rather than implied: whether the first
    descriptor was registered with the ExitStack is NOT observable here, because
    CPython's refcounting closes the file object as soon as it goes out of scope.
    The stack makes the close DETERMINISTIC and exception-safe, which is why the
    code does it; no control can distinguish it, so none pretends to.
    """
    real_open = os.open
    opened = []

    def failing_open(path, flags, mode=0o777):
        if str(path).endswith("r.jsonl"):
            raise OSError(28, "no space left on device")
        fd = real_open(path, flags, mode)
        opened.append(str(path))
        return fd
    monkeypatch.setattr(RUN.os, "open", failing_open)
    with pytest.raises(RUN.H2VoidError):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(tmp_path / "t.jsonl"),
                              report_path=str(tmp_path / "rep.json"),
                              play=lambda **kw: pytest.fail("a game was played"))
    assert opened == [str(tmp_path / "t.jsonl")], opened
    assert not (tmp_path / "r.jsonl").exists()
    assert not (tmp_path / "rep.json").exists()


def test_the_degeneracy_refusal_CLAIMS_NO_DEPENDENCE():
    """🔴 P2 WORDING. The refusal said the repetitions "are not independent plays",
    which the screen cannot establish: independent seeded games can produce
    identical transcripts, and §3.1 says the screen does not test independence."""
    tasks = _tasks()
    out = R.h2_report(_results(tasks, t1j_wins=368), tasks,
                      _per_game({CELLS[0]: 1}, task_ids=[t["task_id"] for t in tasks]),
                      task_digest=R.H2_TASK_DIGEST)
    reason = out["reason"]
    assert "DIVERSITY screen failed" in reason
    assert "does NOT establish" in reason and "non-independent" in reason
    assert "are not independent plays" not in reason


# ═══════ THE BLOCKER THE RUN FOUND: the builder refuses a non-frozen config ══

# 🔴 TWO TESTS STOOD HERE AND ARE GONE, AS THEY PROMISED THEY WOULD.
# `test_THE_QUALIFIED_BUILDER_REFUSES_H2s_ARGMAX_CONFIG` pinned the defect that
# VOIDed the match at task 0 -- the builder refusing any non-frozen config -- and
# its own docstring said: "When the readout change is made admissible, this test
# must be inverted, and its failure is the reminder." The repair made it fail, on
# cue. It and its frozen-config partner are replaced by the section below, which
# asserts the REPAIRED behaviour on both sides: argmax constructs, everything else
# still refuses. The history lives in the evidence and the commit record, which are
# not rewritten.


# ═════ the qualified builder, REPAIRED to admit H2's readout and nothing else ═

def _stub_evaluator():
    class _Eval:
        _g3_reference = "calib020_0001"
        _g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
    return _Eval()


def _task(anchor="red"):
    return {"seed": R.H2_SEED_BLOCK[0], "reference": "calib020_0001",
            "reference_sha1": "209cf2d4fd24a48553d259dd71b4954867b9473e",
            "anchor_colour": anchor}


def _cfg(**changes):
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    base = G3.eval_config()
    return base.__class__(**{**base.__dict__, **changes})


def test_THE_BUILDER_CONSTRUCTS_AN_AGENT_WITH_H2s_ARGMAX_CONFIG():
    """🔴 THE REPAIR. The builder refused any difference from the frozen research
    configuration, and H2 IS a one-field difference -- so the match VOIDed at task
    0 before its first move. The readout may now vary, within an admitted set, and
    NOTHING else may.

    This constructs through the REAL builder, not a mock: mocking the collaborator
    that enforces the constraint is exactly how the constraint stayed invisible.
    """
    from scripts.GPU.alphazero import eval_readout as RO
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    agent = G3.build_reference_agent(task=_task(), evaluator=_stub_evaluator(),
                                     colour="black", config=_cfg(selection_mode="argmax"),
                                     capture=True)
    assert agent.config.selection_mode == "argmax"
    assert agent.readout.mode == RO.MODE_ARGMAX, "the readout must BE argmax, not merely named"
    assert agent.config.mcts_sims == 400, "only the readout changes"


def test_the_UNCHANGED_frozen_configuration_still_constructs():
    """The reference case, unchanged: the repair must not have loosened the path
    every earlier study used."""
    from scripts.GPU.alphazero import eval_readout as RO
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    agent = G3.build_reference_agent(task=_task(), evaluator=_stub_evaluator(),
                                     colour="black", config=G3.eval_config(), capture=True)
    assert agent.config == G3.eval_config()
    assert agent.readout.mode == RO.MODE_OPENING_TEMPERATURE


def test_omitting_the_config_still_uses_the_frozen_one():
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    agent = G3.build_reference_agent(task=_task(), evaluator=_stub_evaluator(),
                                     colour="black", config=None, capture=True)
    assert agent.config == G3.eval_config()


@pytest.mark.parametrize("field,value", [
    ("mcts_sims", 800), ("board_size", 19), ("mcts_eval_batch_size", 7),
    ("mcts_stall_flush_sims", 1), ("opening_temp_plies", 4), ("temp_high", 0.5),
    ("temp_low", 0.9), ("max_moves", 100),
])
def test_EVERY_UNRELATED_FIELD_is_still_REFUSED(field, value):
    """🔑 THE OTHER HALF OF THE REPAIR. Admitting the readout must not admit
    anything else -- one field may vary, and it is named."""
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    with pytest.raises(G3.ReferenceError, match=field):
        G3.build_reference_agent(task=_task(), evaluator=_stub_evaluator(),
                                 colour="black", config=_cfg(**{field: value}),
                                 capture=True)


def test_a_field_that_differs_ALONGSIDE_the_readout_is_still_refused():
    """The combination a lax check would wave through: the admitted change plus an
    unrelated one."""
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    with pytest.raises(G3.ReferenceError, match="mcts_sims"):
        G3.build_reference_agent(
            task=_task(), evaluator=_stub_evaluator(), colour="black",
            config=_cfg(selection_mode="argmax", mcts_sims=800), capture=True)


@pytest.mark.parametrize("mode", ["hoeffding_lcb", "ARGMAX", "", None, 1])
def test_an_UNADMITTED_readout_mode_is_REFUSED(mode):
    """Only the modes a preregistered study has named are admissible. `hoeffding_lcb`
    is a real qualified mode and is refused here ANYWAY: admitting a field is not
    admitting every value of it, and a study that wants it must say so."""
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    with pytest.raises(G3.ReferenceError, match="selection_mode"):
        G3.build_reference_agent(task=_task(), evaluator=_stub_evaluator(),
                                 colour="black", config=_cfg(selection_mode=mode),
                                 capture=True)


def test_the_ADMITTED_MODES_are_DECLARED_and_are_exactly_two():
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    assert G3.ADMISSIBLE_SELECTION_MODES == ("opening_temperature", "argmax")
    assert G3.eval_config().selection_mode in G3.ADMISSIBLE_SELECTION_MODES


@pytest.mark.parametrize("field,value", [("mcts_sims", 400.0), ("temp_high", 1),
                                         ("board_size", 24.0), ("max_moves", 280.0)])
def test_a_field_whose_VALUE_MATCHES_but_whose_TYPE_DIFFERS_is_refused(field, value):
    """🔑 TYPE-STRICT, like every other comparison in this programme. `400 == 400.0`
    and `1 == 1.0`, so a loose check waves these through -- and a control proved the
    earlier tests could not see it, because every one of them changed the VALUE too.
    """
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    frozen = G3.eval_config()
    assert getattr(frozen, field) == value, "the fixture must differ only in TYPE"
    assert type(getattr(frozen, field)) is not type(value)
    with pytest.raises(G3.ReferenceError, match=field):
        G3.build_reference_agent(task=_task(), evaluator=_stub_evaluator(),
                                 colour="black", config=_cfg(**{field: value}),
                                 capture=True)


def test_a_config_of_the_WRONG_TYPE_is_refused():
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    with pytest.raises(G3.ReferenceError):
        G3.build_reference_agent(task=_task(), evaluator=_stub_evaluator(),
                                 colour="black", config={"selection_mode": "argmax"},
                                 capture=True)


def test_the_H2_SEAM_and_THE_BUILDER_now_agree(seam_boundaries, tmp_path):
    """END TO END ACROSS THE SEAM, WITH THE BUILDER UNMOCKED -- the exact path that
    VOIDed at task 0, ply 7.

    🔑 The fixture keeps a reference to the REAL builder precisely so this test can
    put it back. Every other effectful boundary stays mocked, so no JVM starts and
    no model loads; the one collaborator that ENFORCES the constraint is real,
    because mocking it is what hid the defect for four review rounds.
    """
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    play = RUN._production_play(str(tmp_path / "r.jsonl"), _started_deadline())
    play(task=_tasks()[0], identity={}, timeout_s=120)
    build = seam_boundaries["reference_build"]      # the seam's OWN closure

    # put the REAL builder back, then drive the seam's closure through it: this is
    # the call the harness makes at ply 7, and the one that aborted the match.
    G3.build_reference_agent = seam_boundaries["real_build"]
    task = dict(_tasks()[0])                        # a t1j_red task: we play black
    agent = build(task, _stub_evaluator())
    assert agent.config.selection_mode == "argmax"
    assert agent.config.mcts_sims == 400
    from scripts.GPU.alphazero import eval_readout as RO
    assert agent.readout.mode == RO.MODE_ARGMAX


def test_THE_EXACT_RETRY_TASK_CONSTRUCTS_THROUGH_THE_REAL_BUILDER():
    """🔴 THE CHECK THE VOID MADE NECESSARY, on the retry's OWN first task.

    Not a synthetic task and not a mock: task 0 of the retry schedule, with its
    fresh seed, handed to the real `build_reference_agent` with the config the
    production seam builds. This is the call that aborted at ply 7 on attempt 1.
    """
    from scripts.GPU.alphazero import eval_readout as RO
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    task = PLAN.build_tasks(PLAN.load_source_plan())[0]
    assert task["seed"] == R.H2_SEED_BLOCK[0] == 202622000
    assert task["colour_arm"] == "t1j_red" and task["anchor_colour"] == "red"
    assert task["selection_mode"] == "argmax"

    cfg = G3.eval_config()
    argmax = cfg.__class__(**{**cfg.__dict__, "selection_mode": R.SELECTION_MODE})
    agent = G3.build_reference_agent(task=task, evaluator=_stub_evaluator(),
                                    colour=REF.reference_colour(task),
                                    config=argmax, capture=True)
    assert agent.readout.mode == RO.MODE_ARGMAX
    assert agent.config.mcts_sims == 400 and agent.config.board_size == 24
    assert agent.seed == task["seed"], "the SCHEDULED seed, not another"


def test_EVERY_CELLS_FIRST_TASK_constructs_through_the_real_builder():
    """Both colour arms, all eight openings: our side plays red in half of them, and
    `build_reference_agent` checks the colour against the arm. One task would have
    proved only one arm."""
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    argmax = cfg.__class__(**{**cfg.__dict__, "selection_mode": "argmax"})
    seen = set()
    for task in PLAN.build_tasks(PLAN.load_source_plan()):
        key = (task["opening"], task["colour_arm"])
        if key in seen:
            continue
        seen.add(key)
        agent = G3.build_reference_agent(task=task, evaluator=_stub_evaluator(),
                                         colour=REF.reference_colour(task),
                                         config=argmax, capture=True)
        assert agent.config.selection_mode == "argmax"
    assert len(seen) == 16, seen


def test_THE_BUILDER_STILL_REFUSES_THE_WRONG_COLOUR_FOR_THE_ARM():
    """Our side plays the colour T1j does not. `build_reference_agent` checks that
    against `anchor_colour`, and the repair must not have loosened it.

    🔴 A control that made the check vacuous went NOT CAUGHT, because every existing
    test passed the CORRECT colour -- there was no mismatch to detect. This passes
    the wrong one deliberately.
    """
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    argmax = cfg.__class__(**{**cfg.__dict__, "selection_mode": "argmax"})
    task = PLAN.build_tasks(PLAN.load_source_plan())[0]      # t1j_red: we play black
    right = REF.reference_colour(task)
    wrong = "red" if right == "black" else "black"
    with pytest.raises(G3.ReferenceError, match="contradicts anchor_colour"):
        G3.build_reference_agent(task=task, evaluator=_stub_evaluator(), colour=wrong,
                                 config=argmax, capture=True)
