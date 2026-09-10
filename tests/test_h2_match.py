"""H2 — the deterministic-readout head-to-head, implemented CODE-AND-TEST ONLY
against the card frozen at 5aa5db4.

🔴 NOTHING HERE RUNS A GAME. No model is loaded, no JVM started, no query issued,
no seed registered or drawn, and no gate opened. The play seam is a fixture; the
gate is asserted CLOSED in the real repository, never patched open.
"""
import inspect
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
    assert R.H2_SEED_BLOCK == (202618000, 202618736)
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

def _plies(n=4, *, first=7, anchor="red", flip=False):
    out = []
    for k in range(n):
        ply = first + k
        mover = R.expected_mover(ply, anchor)
        if flip:
            mover = "black" if mover == "red" else "red"
        out.append({"ply": ply, "mover": mover, "move": [ply, k]})
    return out


def _result(n=4, *, first=7, reason="win", winner="red"):
    return {"plies": first + n - 1, "terminal_reason": reason, "winner": winner}


def _t(plies=None, result=None, *, bound=6, anchor="red"):
    return R.transcript(plies if plies is not None else _plies(anchor=anchor),
                        result if result is not None else _result(),
                        opening_bound=bound, anchor_colour=anchor)


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
    R.transcript(_plies(), _result(), opening_bound=6, anchor_colour="red")
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(_plies(), _result(), opening_bound=5, anchor_colour="red")


def test_REMOVING_THE_FIRST_ply_REFUSES_though_the_rest_stays_contiguous():
    """🔴 Contiguity is a property of the INTERIOR: a sequence missing its first
    record is still contiguous from whatever survives."""
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(_plies()[1:], _result(), opening_bound=6, anchor_colour="red")


def test_REMOVING_THE_FINAL_ply_REFUSES_though_the_rest_stays_contiguous():
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(_plies()[:-1], _result(), opening_bound=6, anchor_colour="red")


def test_a_DUPLICATED_or_OUT_OF_ORDER_ply_REFUSES(  ):
    p = _plies()
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript(p[:2] + [p[1]] + p[2:], _result(), opening_bound=6,
                     anchor_colour="red")
    with pytest.raises(R.H2RulesError, match="declare exactly"):
        R.transcript([p[1], p[0]] + p[2:], _result(), opening_bound=6,
                     anchor_colour="red")


def test_FLIPPING_EVERY_MOVER_refuses_even_though_alternation_is_preserved():
    """🔴 The case an alternation check cannot see: every mover swapped keeps the
    moves alternating while swapping which side played each one."""
    flipped = _plies(flip=True)
    movers = [p["mover"] for p in flipped]
    assert all(a != b for a, b in zip(movers, movers[1:])), "still alternating"
    with pytest.raises(R.H2RulesError, match="bound to the ARM"):
        R.transcript(flipped, _result(), opening_bound=6, anchor_colour="red")


def test_the_mover_is_derived_from_the_ARM_so_both_arms_expect_opposite_colours():
    assert R.expected_mover(7, "red") == "red" and R.expected_mover(8, "red") == "black"
    assert R.expected_mover(7, "black") == "black" and R.expected_mover(8, "black") == "red"


def test_a_terminal_reason_outside_the_two_is_REFUSED():
    with pytest.raises(R.H2RulesError, match="no resignation"):
        R.transcript(_plies(), _result(reason="resignation"), opening_bound=6,
                     anchor_colour="red")


@pytest.mark.parametrize("bad", ["7", 7.0, True])
def test_a_ply_index_that_is_not_an_INT_is_refused(bad):
    p = _plies()
    p[0] = dict(p[0], ply=bad)
    with pytest.raises(R.H2RulesError):
        R.transcript(p, _result(), opening_bound=6, anchor_colour="red")


def test_a_move_whose_coordinates_are_STRINGS_is_refused():
    p = _plies()
    p[1] = dict(p[1], move=["8", "1"])
    with pytest.raises(R.H2RulesError, match="int is required"):
        R.transcript(p, _result(), opening_bound=6, anchor_colour="red")


def test_the_excluded_fields_are_DECLARED_not_merely_absent():
    for f in ("task_id", "rep", "seed", "elapsed_s", "root_visits"):
        assert f in R.TRANSCRIPT_EXCLUDES


# ──────────────────────── the per-cell degeneracy screen ────────────────────

CELLS = [(f"o{i}", arm) for i in range(1, 9) for arm in ("t1j_red", "t1j_black")]


def _per_game(distinct_per_cell):
    """One entry per game; `distinct_per_cell` maps a cell to how many distinct
    transcripts it should hold."""
    out = []
    for cell in CELLS:
        k = distinct_per_cell.get(cell, R.N_REPS)
        for i in range(R.N_REPS):
            out.append({"opening": cell[0], "colour_arm": cell[1],
                        "transcript_digest": f"{cell}-{min(i, k - 1)}"})
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


def test_a_missing_cell_fails_the_screen_rather_than_passing_vacuously():
    games = [g for g in _per_game({}) if (g["opening"], g["colour_arm"]) != CELLS[0]]
    assert R.degeneracy_screen(games)["passes"] is False


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
    out = R.h2_report(_results(tasks, t1j_wins=700), tasks,
                      _per_game({CELLS[0]: 1}), task_digest=R.H2_TASK_DIGEST)
    assert out["reported"] is False
    assert out["outcome"] == "INCONCLUSIVE — DEGENERATE DESIGN"
    assert out["interval"] is None and out["score"] is None
    assert out["degeneracy_screen"]["failing_cells"] == [list(CELLS[0])]


@pytest.mark.parametrize("wins,want", [
    (736, "T1J_STRONGER"), (0, "NOT_STRONGER"), (368, "INCONCLUSIVE"),
])
def test_the_verdict_follows_the_parity_rule_end_to_end(wins, want):
    tasks = _tasks()
    out = R.h2_report(_results(tasks, t1j_wins=wins), tasks, _per_game({}),
                      task_digest=R.H2_TASK_DIGEST)
    assert out["reported"] is True
    assert out["outcome"] == want == out["verdict"]


def test_the_report_carries_the_interval_STANDING_and_all_sixteen_counts():
    tasks = _tasks()
    out = R.h2_report(_results(tasks, t1j_wins=400), tasks, _per_game({}),
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
        RUN.run_h2(tasks=[], results_path=str(out), trace_path=str(tmp_path / "t.jsonl"),
                   play=lambda **kw: pytest.fail("a game was played behind a shut gate"))
    assert not out.exists()


def test_THE_SEED_BLOCK_IS_REGISTERED_NOWHERE_and_the_barrier_says_so():
    lo, hi = R.H2_SEED_BLOCK
    assert not any(REF.seed_is_accounted(s) for s in (lo, hi - 1))
    with pytest.raises(RUN.H2Error, match="not registered"):
        RUN.check_seed_registration()


def test_the_registration_barrier_checks_EVERY_seed_not_the_endpoints(monkeypatch):
    lo, hi = R.H2_SEED_BLOCK
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + ((lo, hi - 1),))
    with pytest.raises(RUN.H2Error, match="not registered"):
        RUN.check_seed_registration()


def test_a_fully_registered_block_satisfies_the_barrier_and_NOT_the_gate(monkeypatch):
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + (R.H2_SEED_BLOCK,))
    RUN.check_seed_registration()
    assert RUN.H2_EXECUTION_AUTHORIZED is False


def test_the_output_paths_are_CREATE_ONLY_and_must_be_two_files(tmp_path):
    r, t = tmp_path / "r.jsonl", tmp_path / "t.jsonl"
    RUN.check_output_paths(str(r), str(t))
    with pytest.raises(RUN.H2Error, match="requires a trace"):
        RUN.check_output_paths(str(r), None)
    with pytest.raises(RUN.H2Error, match="ONE file"):
        RUN.check_output_paths(str(r), str(r))
    r.write_text("x")
    with pytest.raises(RUN.H2Error, match="already exists"):
        RUN.check_output_paths(str(r), str(t))


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
    with pytest.raises(RUN.H2VoidError, match="selection_mode"):
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
        plies = [{"ply": 7 + k, "mover": R.expected_mover(7 + k, anchor),
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
        trace_path=str(tmp_path / "t.jsonl"), play=_play_factory(wins=736))
    assert out["reported"] is True and out["outcome"] == "T1J_STRONGER"
    assert out["degeneracy_screen"]["passes"] is True


def test_a_run_whose_games_are_all_IDENTICAL_is_DEGENERATE_not_a_verdict(registered,
                                                                        tmp_path):
    out = RUN._run_h2_unguarded(
        tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
        trace_path=str(tmp_path / "t.jsonl"),
        play=_play_factory(distinct=False, wins=736))
    assert out["outcome"] == "INCONCLUSIVE — DEGENERATE DESIGN"
    assert out["interval"] is None
    assert len(out["degeneracy_screen"]["failing_cells"]) == 16


def test_the_deadline_VOIDS_mid_run_and_reports_no_partial_rate(registered, tmp_path):
    with pytest.raises(RUN.H2VoidError, match="deadline"):
        RUN._run_h2_unguarded(
            tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
            trace_path=str(tmp_path / "t.jsonl"), play=_play_factory(),
            deadline_s=-1)


def test_the_trace_and_the_results_are_both_written_and_fsynced(registered, tmp_path):
    r, t = tmp_path / "r.jsonl", tmp_path / "t.jsonl"
    RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(r), trace_path=str(t),
                          play=_play_factory())
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
                "plies": [{"ply": 8, "mover": R.expected_mover(8, anchor),
                           "move": [8, 1]}],       # the FIRST ply is missing
                "opening_bound": 6}
    with pytest.raises(RUN.H2VoidError):
        RUN._run_h2_unguarded(tasks=_tasks(), results_path=str(tmp_path / "r.jsonl"),
                              trace_path=str(tmp_path / "t.jsonl"), play=play)


# ────────────────────────────────── the wrapper ─────────────────────────────

def test_the_wrapper_refuses_with_the_shut_gate_and_verifies_it_closed(capsys):
    assert CMD.gate_is_open() is False
    assert CMD.main([]) == CMD.EXIT_UNAUTHORIZED
    assert "UNAUTHORIZED" in capsys.readouterr().err


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


def test_THE_FINALLY_PATH_also_restores_and_reports_its_own_failure(monkeypatch, capsys):
    """🔴 A control proved the earlier test blind here: with the gate shut, `main`
    returns on the UNAUTHORIZED branch and never reaches the `finally`, so
    disabling the finally's check changed nothing. This drives the OTHER path --
    gate open, restoration failing -- which is the one that matters, because it is
    the path a real run takes."""
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "restore_gate", lambda *a, **k: False)
    assert CMD.main([]) == CMD.EXIT_GATE_NOT_RESTORED
    assert "BY HAND" in capsys.readouterr().err


def test_the_finally_path_restores_a_REAL_open_gate_after_a_refusal(monkeypatch, tmp_path):
    """The wrapper's own promise: restoration runs after EVERY exit, including the
    refusal path, and the file is left closed."""
    decoy = tmp_path / "runner.py"
    decoy.write_text("H2_EXECUTION_AUTHORIZED = True\n")
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    code = CMD.main([], _runner_source=str(decoy))
    assert code in (CMD.EXIT_REFUSED, CMD.EXIT_VOID, CMD.EXIT_UNEXPECTED)
    assert decoy.read_text() == "H2_EXECUTION_AUTHORIZED = False\n"


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
def test_no_H2_module_imports_gameplay_or_registry_machinery(mod):
    import pathlib
    src = pathlib.Path(mod.__file__).read_text()
    for forbidden in ("ACCOUNTED_SEED_INTERVALS = ", "compile_helper",
                      "_default_load_evaluator", "subprocess.run("):
        assert forbidden not in src, (mod.__name__, forbidden)
