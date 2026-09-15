"""H3 FULL STUDY — the gated segment runner and the gated generator.

Every gate here is SHUT and must stay shut. Nothing in this file plays a game,
loads a model, starts a JVM, generates stratum B or draws a match seed.
"""
import ast
import inspect

import pytest

from scripts.GPU.alphazero import h3_study_rules as R
from scripts.GPU.alphazero import h3_study_runner as RUN
from scripts.GPU.alphazero import h3_study_generator as GEN

FRESH = (777000000, 777000000 + R.N_GAMES)


# ───────────────────────── BOTH gates are shut ─────────────────────────────

def test_BOTH_GATES_ARE_SHUT_IN_THE_REAL_REPOSITORY():
    assert RUN.H3_STUDY_EXECUTION_AUTHORIZED is False
    assert GEN.H3_GENERATION_AUTHORIZED is False


def test_THEY_ARE_SEPARATE_GATES_and_neither_implies_the_other():
    """🔑 Generating stratum B is a RUN with its own authorization (card §1.5).
    One switch for both would let a generation approval authorize a match."""
    assert RUN.__dict__ is not GEN.__dict__
    run_src = open(RUN.__file__, encoding="utf-8").read()
    gen_src = open(GEN.__file__, encoding="utf-8").read()
    assert "H3_GENERATION_AUTHORIZED" not in run_src
    assert "H3_STUDY_EXECUTION_AUTHORIZED" not in gen_src


def test_NO_SEED_BLOCK_IS_RESERVED():
    assert RUN.STUDY_SEED_BLOCK is None
    with pytest.raises(RUN.H3StudyRunError, match="NO SEED BLOCK"):
        RUN.check_seed_registration()


def test_the_entry_READS_THE_GATE_FIRST_by_AST():
    fn = ast.parse(inspect.getsource(RUN.run_segment)).body[0]
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    first = next(c for c in calls if getattr(c.func, "id", ""))
    assert getattr(first.func, "id", "") == "check_gate", ast.dump(first)


def test_the_generator_entry_READS_ITS_GATE_FIRST_by_AST():
    fn = ast.parse(inspect.getsource(GEN.generate_co_produced)).body[0]
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    first = next(c for c in calls if getattr(c.func, "id", ""))
    assert getattr(first.func, "id", "") == "check_gate", ast.dump(first)


def test_neither_entry_can_be_reached_with_the_gates_shut(tmp_path):
    with pytest.raises(RUN.H3StudyRunError, match="NOT AUTHORIZED"):
        RUN.run_segment(segment=0, results_path=str(tmp_path / "r"),
                        trace_path=str(tmp_path / "t"),
                        report_path=str(tmp_path / "p"))
    with pytest.raises(GEN.H3GenerationError, match="NOT AUTHORIZED"):
        GEN.generate_co_produced(out_path=str(tmp_path / "o.json"))
    assert not list(tmp_path.iterdir())


def test_THERE_IS_NO_CLI_OVERRIDE_FOR_EITHER_GATE():
    from scripts.GPU.alphazero import h3_study_command as CMD
    help_text = CMD._parser().format_help()
    for flag in ("--authorize", "--force", "--gate", "--runner-source"):
        assert flag not in help_text, flag
    src = open(CMD.__file__, encoding="utf-8").read()
    assert "os.environ" not in src, "no environment variable reaches the worker"


def test_NOTHING_EFFECTFUL_IS_IMPORTED_AT_MODULE_LEVEL():
    for mod in (RUN, GEN):
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        top = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
        for effectful in ("e4_screen_runner", "e4_screen_integration",
                          "t1j_toolchain", "d1_probe", "twixtbot_g3_reference",
                          "e4_screen_command"):
            assert effectful not in {str(t) for t in top}, (mod.__name__, effectful)


# ───────────────────────── containment ─────────────────────────────────────

def test_THE_SEAM_CHECKS_THE_GATE_ITSELF_not_only_the_entry():
    play = RUN._production_play("/tmp/h3-study-never-written")
    with pytest.raises(RUN.H3StudyRunError, match="NOT AUTHORIZED"):
        play(task={}, identity={}, timeout_s=1)


def test_THE_BOUNDARY_REFUSES_INSIDE_A_TEST_PROCESS_even_with_the_gate_OPEN(
        monkeypatch):
    """The 2026-09-12 incident's exact state: the gate forced open."""
    monkeypatch.setattr(RUN, "H3_STUDY_EXECUTION_AUTHORIZED", True)
    RUN.check_gate()
    play = RUN._production_play("/tmp/h3-study-never-written")
    with pytest.raises(RUN.H3StudyContainmentError):
        play(task={}, identity={}, timeout_s=1)


def test_the_boundary_refuses_with_EVERY_authorization_check_removed(monkeypatch):
    monkeypatch.setattr(RUN, "H3_STUDY_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(RUN, "check_gate", lambda: None)
    play = RUN._production_play("/tmp/h3-study-never-written")
    with pytest.raises(RUN.H3StudyContainmentError):
        play(task={}, identity={}, timeout_s=1)


def test_THE_GENERATOR_HAS_THE_SAME_BOUNDARY(monkeypatch, tmp_path):
    """🔑 The generator loads a model and starts a JVM, so it needs the seam
    boundary exactly as the match does — a generation run is still a run."""
    monkeypatch.setattr(GEN, "H3_GENERATION_AUTHORIZED", True)
    monkeypatch.setattr(GEN, "check_gate", lambda: None)
    with pytest.raises(GEN.H3GenerationContainmentError):
        GEN.generate_co_produced(out_path=str(tmp_path / "o.json"))
    assert not list(tmp_path.iterdir())


def test_the_seam_checks_BEFORE_anything_effectful():
    src = inspect.getsource(RUN._production_play)
    import textwrap
    outer = ast.parse(textwrap.dedent(src)).body[0]
    fn = next(n for n in ast.walk(outer)
              if isinstance(n, ast.FunctionDef) and n.name == "play")
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name:
                lines[name] = min(lines.get(name, node.lineno), node.lineno)
    for guard in ("check_gate", "assert_production_acts_are_inert"):
        assert guard in lines, guard
    for effect in ("verified_paths", "_default_compile", "_default_load_evaluator"):
        assert effect in lines, effect
        assert lines["check_gate"] < lines[effect], f"gate after {effect}"
        assert lines["assert_production_acts_are_inert"] < lines[effect], effect


def test_the_gate_is_checked_for_EVERY_GAME_not_only_the_first():
    assert inspect.getsource(RUN._production_play).count("check_gate()") >= 2


# ───────────────────────── the config object, threaded ─────────────────────

def test_the_argmax_config_has_ONE_construction_and_the_seam_CARRIES_it():
    cfg = RUN.frozen_argmax_config()
    assert cfg.selection_mode == "argmax"
    play = RUN._production_play("/tmp/x", None, None, cfg)
    assert play.config is cfg, "the seam holds THE object, not an equal one"


def test_the_identity_is_read_off_the_SEAM_S_OWN_config():
    cfg = RUN.frozen_argmax_config()
    ident = RUN.frozen_incumbent_identity(cfg)
    assert ident["eval_config"]["selection_mode"] == "argmax"
    assert ident["design"] == "H3_FULL_STUDY"
    for f, v in ident["argmax_config"].items():
        got = getattr(cfg, f)
        assert got == v and type(got) is type(v), f


def test_a_DRIFTED_config_object_is_REFUSED():
    real = RUN.frozen_argmax_config()
    drifted = real.__class__(**{**real.__dict__, "mcts_sims": 401})
    with pytest.raises(RUN.H3StudyRunError, match="mcts_sims|disagrees"):
        RUN.check_incumbent_identity(RUN.frozen_incumbent_identity(drifted), drifted)


def test_a_RECORD_describing_another_object_is_REFUSED():
    import copy
    cfg = RUN.frozen_argmax_config()
    bad = copy.deepcopy(RUN.frozen_incumbent_identity(cfg))
    bad["argmax_config"]["mcts_sims"] = 401
    with pytest.raises(RUN.H3StudyRunError, match="mcts_sims"):
        RUN.check_incumbent_identity(bad, cfg)


def test_a_seam_that_declares_NO_config_is_REFUSED():
    play = RUN._production_play("/tmp/x")
    assert play.config is None
    with pytest.raises(RUN.H3StudyRunError, match="no configuration object"):
        RUN._require_seam_config(play)


# ───────────────────────── the segment schedule ────────────────────────────

@pytest.fixture(scope="module")
def tasks():
    return R.build_tasks(R.stub_opening_set(), seed_interval=FRESH)


def test_a_SEGMENT_schedule_is_its_own_148_games(tasks):
    for k in range(R.N_SEGMENTS):
        seg = RUN.segment_schedule(tasks, k)
        assert len(seg) == R.GAMES_PER_SEGMENT
        assert {t["segment"] for t in seg} == {k}
        assert len({t["pair_id"] for t in seg}) == R.PAIRS_PER_SEGMENT


def test_a_SEGMENT_INDEX_outside_the_plan_is_REFUSED(tasks):
    for bad in (-1, R.N_SEGMENTS, 1.0, True):
        with pytest.raises(RUN.H3StudyRunError, match="segment"):
            RUN.segment_schedule(tasks, bad)


def test_the_segment_digest_is_ITS_OWN_and_differs_between_segments(tasks):
    digests = [RUN.segment_digest(tasks, k) for k in range(R.N_SEGMENTS)]
    assert len(set(digests)) == R.N_SEGMENTS
    assert all(len(d) == 64 for d in digests)


def test_check_segment_schedule_REFUSES_a_tampered_schedule(tasks):
    seg = [dict(t) for t in RUN.segment_schedule(tasks, 0)]
    want = RUN.segment_digest(tasks, 0)
    assert RUN.check_segment_schedule(seg, 0, want)["n_tasks"] == 148
    seg[0]["ply_cap"] = seg[0]["ply_cap"] + 1
    with pytest.raises(RUN.H3StudyRunError, match="different schedule"):
        RUN.check_segment_schedule(seg, 0, want)


def test_check_segment_schedule_ALSO_ASKS_THE_REGISTRY(tasks):
    from scripts.GPU.alphazero import e4_screen_reference as REF
    import unittest.mock as _m
    seg = RUN.segment_schedule(tasks, 0)
    want = RUN.segment_digest(tasks, 0)
    assert RUN.check_segment_schedule(seg, 0, want)["n_tasks"] == 148
    spent = REF.EXPOSED_SEED_INTERVALS + ((FRESH[0], FRESH[1]),)
    with _m.patch.object(REF, "EXPOSED_SEED_INTERVALS", spent):
        with pytest.raises(RUN.H3StudyRunError, match="EXPOSED|registry"):
            RUN.check_segment_schedule(seg, 0, want)


def test_a_SEEDLESS_schedule_is_REFUSED():
    tasks = R.build_tasks(R.stub_opening_set())
    seg = RUN.segment_schedule(tasks, 0)
    with pytest.raises(RUN.H3StudyRunError, match="carry no seed"):
        RUN.check_segment_schedule(seg, 0, RUN.segment_digest(tasks, 0))


# ───────────────────────── outputs ─────────────────────────────────────────

def test_the_output_paths_are_CREATE_ONLY_and_must_be_THREE_files(tmp_path):
    p = tmp_path / "x"
    p.write_text("")
    with pytest.raises(RUN.H3StudyRunError, match="already exists"):
        RUN.check_output_paths(str(p), str(tmp_path / "b"), str(tmp_path / "c"))
    with pytest.raises(RUN.H3StudyRunError, match="THREE"):
        RUN.check_output_paths(str(tmp_path / "a"), str(tmp_path / "a"),
                               str(tmp_path / "c"))


def test_a_DANGLING_SYMLINK_at_an_output_path_is_REFUSED(tmp_path):
    import os
    link = tmp_path / "dangling"
    os.symlink(str(tmp_path / "nonexistent"), str(link))
    assert not os.path.exists(link) and os.path.lexists(link)
    with pytest.raises(RUN.H3StudyRunError, match="already exists"):
        RUN.check_output_paths(str(link), str(tmp_path / "b"), str(tmp_path / "c"))


def test_EVERY_SEGMENT_WRITES_TO_ITS_OWN_DIRECTORY():
    dirs = [RUN.segment_out_dir(k) for k in range(R.N_SEGMENTS)]
    assert len(set(dirs)) == R.N_SEGMENTS
    from scripts.GPU.alphazero import h3_study_command as CMD
    for d in dirs:
        assert not any(d.startswith(spent.rstrip("/") + "/") or d == spent
                       for spent in CMD.SPENT_OUT_DIRS), d


# ───────────────────────── optional stopping (card §5.5) ───────────────────

def test_A_WITHHELD_SEGMENT_AFTER_INSPECTION_PREVENTS_A_VERDICT():
    """🔴 The four runs are separately authorized OPERATIONALLY, which is what
    makes a late failure survivable and is also the opening for optional
    stopping."""
    out = RUN.combine_segments([
        {"segment": 0, "status": "completed", "outcomes_inspected": True},
        {"segment": 1, "status": "withheld", "outcomes_inspected": True},
    ])
    assert out["verdict_permitted"] is False
    assert "inspect" in out["why"].lower()


def test_a_segment_withheld_WITHOUT_inspection_still_permits_a_verdict():
    out = RUN.combine_segments([
        {"segment": 0, "status": "completed", "outcomes_inspected": False},
        {"segment": 1, "status": "withheld", "outcomes_inspected": False},
    ])
    assert out["verdict_permitted"] is True


def test_a_VOIDED_segment_FLAGS_the_primary_and_blocks_an_automatic_verdict():
    out = RUN.combine_segments([
        {"segment": 0, "status": "completed", "outcomes_inspected": False},
        {"segment": 1, "status": "void", "outcomes_inspected": False},
    ])
    assert out["verdict_permitted"] is False
    assert out["flagged"] is True
    assert "review" in out["why"].lower()


def test_the_report_names_EVERY_segment_and_whether_it_was_inspected():
    out = RUN.combine_segments([
        {"segment": k, "status": "completed", "outcomes_inspected": False}
        for k in range(R.N_SEGMENTS)])
    assert out["verdict_permitted"] is True
    assert len(out["segments"]) == R.N_SEGMENTS
    assert all("outcomes_inspected" in s for s in out["segments"])


# ═══════════ the REAL builder accepts the study's own tasks ════════════════
# 🔴 THE CHECK H2'S ATTEMPT 1 VOID MADE NECESSARY, and which the pilot then found
# a second way to fail: the tasks must carry the identity fields UNPATCHED, and
# the agent factory must route on the task's OWN reference_colour.

def _stub_evaluator():
    class _Eval:
        _g3_reference = "calib020_0001"
        _g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
    return _Eval()


def test_EVERY_SEGMENT_S_TASKS_construct_through_the_REAL_builder(tasks):
    """One pair from every segment and both strata, through the real builder with
    the SEAM'S OWN config — not a fifth expression of the same override."""
    from scripts.GPU.alphazero import eval_readout as RO
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import h2_match_rules as H2R
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    argmax = RUN.frozen_argmax_config()
    seen = set()
    picked = []
    for t in tasks:
        key = (t["segment"], t["stratum"], t["incumbent_colour"])
        if key not in seen:
            seen.add(key)
            picked.append(t)
    assert len(picked) == R.N_SEGMENTS * len(R.STRATA) * 2 == 16
    for task in picked:                      # 🔑 UNPATCHED
        assert REF.reference_colour(task) == task["incumbent_colour"]
        agent = G3.build_reference_agent(
            task=task, evaluator=_stub_evaluator(),
            colour=REF.reference_colour(task), config=argmax, capture=True)
        assert agent.readout.mode == RO.MODE_ARGMAX
        assert agent.config.mcts_sims == H2R.MCTS_SIMS
        assert agent.seed == task["seed"], "the SCHEDULED seed, not another"


def test_the_builder_REFUSES_the_WRONG_COLOUR_for_the_arm(tasks):
    """The negative half: if it accepted any colour the test above proves nothing."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    task = tasks[0]
    wrong = "red" if REF.reference_colour(task) == "black" else "black"
    with pytest.raises(Exception):
        G3.build_reference_agent(task=task, evaluator=_stub_evaluator(),
                                 colour=wrong, config=RUN.frozen_argmax_config(),
                                 capture=True)


def test_THE_REAL_AGENT_FACTORY_ROUTES_BY_THE_TASKS_OWN_reference_colour(tasks):
    """🔴 `make_agent_factory` subscripts `task["reference_colour"]` DIRECTLY. The
    pilot's tasks lacked it and every game would have died at ply 6."""
    from scripts.GPU.alphazero import e4_screen_integration as INT
    built = []
    factory = INT.make_agent_factory(
        runtime=object(), ctx=INT.IntegrationContext(), evaluator="EVAL",
        reference_build=lambda t, evaluator: built.append(t["task_id"]) or "REF")
    for t in tasks[:2]:
        other = "black" if t["incumbent_colour"] == "red" else "red"
        assert factory(t, t["incumbent_colour"]) == "REF"
        assert isinstance(factory(t, other), INT.T1jAgent)
    assert len(built) == 2


def test_the_registry_admits_a_FRESH_schedule_for_EXECUTION(tasks):
    """The executable question, not the structural one."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    out = REF.validate_schedule_executable(list(tasks))
    assert out["n_tasks"] == R.N_GAMES
    assert out["distinct_seeds"] == R.N_GAMES
    assert out["distinct_stream_pairs"] == R.N_GAMES


# ═══════════ the generator's walk, driven with INERT movers ════════════════

def _inert_movers(config=None):
    """Movers that play the first legal move, deterministically — but READ the
    arguments the real ones read, so a contract change is visible here."""
    calls = {"incumbent": 0, "t1j": 0, "seeds": []}

    def incumbent(state, *, seed, colour):
        # 🔑 IT USES THE SEED, because the real one does: the incumbent is the
        # ONLY source of entropy in this stratum. A fixture that ignored it would
        # stand in for a player that cannot exist under this protocol.
        calls["incumbent"] += 1
        calls["seeds"].append(seed)
        assert colour in ("red", "black")
        legal = sorted(state.legal_moves())
        return legal[seed % len(legal)]

    def t1j(state, *, colour):
        calls["t1j"] += 1
        assert colour in ("red", "black")
        return sorted(state.legal_moves())[-1 - (calls["t1j"] % 3)]

    return {"incumbent": incumbent, "t1j": t1j,
            "config": config or R.generation_config(), "calls": calls}


def test_the_alternating_protocol_gives_each_engine_THREE_of_SIX_plies():
    for order in (R.ORDER_INCUMBENT_FIRST, R.ORDER_T1J_FIRST):
        movers = [GEN.mover_at_ply(order, p) for p in range(1, 7)]
        assert movers.count("incumbent") == movers.count("t1j") == 3, order
    assert GEN.mover_at_ply(R.ORDER_INCUMBENT_FIRST, 1) == "incumbent"
    assert GEN.mover_at_ply(R.ORDER_T1J_FIRST, 1) == "t1j"
    for bad in (0, 7, 1.0, True):
        with pytest.raises(GEN.H3GenerationError, match="ply"):
            GEN.mover_at_ply(R.ORDER_INCUMBENT_FIRST, bad)


def test_the_order_allocation_TOTALS_are_the_declared_ones():
    orders = [GEN.order_for_index(i) for i in range(R.PAIRS_PER_STRATUM)]
    assert orders.count(R.ORDER_INCUMBENT_FIRST) == sum(R.INCUMBENT_FIRST_PER_SEGMENT)
    assert orders.count(R.ORDER_T1J_FIRST) == sum(R.T1J_FIRST_PER_SEGMENT)


def test_generate_one_WALKS_BOTH_ENGINES_and_records_its_attempt_seed():
    movers = _inert_movers()
    op = GEN.generate_one(index=0, order=R.ORDER_INCUMBENT_FIRST, movers=movers)
    assert len(op["moves"]) == R.OPENING_PLIES
    assert op["stratum"] == R.STRATUM_CO_PRODUCED
    assert op["order"] == R.ORDER_INCUMBENT_FIRST
    assert op["stub"] is False
    assert movers["calls"]["incumbent"] == 3 and movers["calls"]["t1j"] == 3
    assert op["seed"] == R.attempt_seed(R.GEN_SEED_CO_PRODUCED, 0, op["attempts"] - 1)
    assert not op["state"].is_terminal()


def test_the_incumbent_MOVER_GETS_A_SEED_and_t1j_does_not():
    """🔑 The asymmetry IS the finding: only the incumbent supplies entropy."""
    import inspect as _i
    assert "seed" in _i.signature(_inert_movers()["incumbent"]).parameters
    assert "seed" not in _i.signature(_inert_movers()["t1j"]).parameters


def test_A_DETERMINISTIC_INCUMBENT_PRODUCES_DUPLICATES_AND_IS_REFUSED(tmp_path):
    """🔴 THE ENTROPY FINDING, BOUND BEHAVIOURALLY.

    If the incumbent ignores its seed — which is exactly what `argmax` does, its
    move being a function of the position alone — then every opening of one order
    is the SAME opening. The walk does not quietly emit 74 copies: the
    distinctness guard refuses. This is the failure the card's §1.7.2 exists to
    prevent, reproduced in one test.
    """
    movers = _inert_movers()
    movers["incumbent"] = lambda state, *, seed, colour: sorted(
        state.legal_moves())[0]                      # deterministic: no seed
    with pytest.raises(GEN.H3GenerationError, match="duplicates an accepted"):
        GEN._generate_unguarded(movers=movers, out_path=str(tmp_path / "o.json"),
                                trace_path=str(tmp_path / "t.jsonl"), n=2)


def test_an_ILLEGAL_move_from_either_engine_is_REFUSED():
    movers = _inert_movers()
    movers["incumbent"] = lambda state, *, seed, colour: (99, 99)
    with pytest.raises(GEN.H3GenerationError, match="illegal"):
        GEN.generate_one(index=0, order=R.ORDER_INCUMBENT_FIRST, movers=movers)


def test_exhausting_the_attempts_ABORTS_generation(monkeypatch):
    movers = _inert_movers()
    monkeypatch.setattr(R, "MAX_ATTEMPTS", 0)
    with pytest.raises(GEN.H3GenerationError, match="MAX_ATTEMPTS"):
        GEN.generate_one(index=0, order=R.ORDER_INCUMBENT_FIRST, movers=movers)


def test_the_walk_PINS_ITS_ARTIFACT_and_names_what_the_stratum_IS(tmp_path):
    import json
    movers = _inert_movers()
    cleanups = []
    doc = GEN._generate_unguarded(
        movers=movers, out_path=str(tmp_path / "o.json"),
        trace_path=str(tmp_path / "t.jsonl"),
        cleanup=lambda: cleanups.append(1), n=3)
    assert doc["n"] == 3 and len(doc["openings"]) == 3
    assert doc["opening_set_digest"] == R.opening_set_digest(doc["openings"])
    note = doc["generation_note"].lower()
    assert "not neutral" in note and "only the incumbent" in note
    assert "state" not in doc["openings"][0], "the engine state is not serialisable"
    assert all("attempts" in o and "seed" in o for o in doc["openings"])
    assert len(cleanups) == 3, "cleanup between EVERY opening -- no tree carried"
    trace = [json.loads(l) for l in open(tmp_path / "t.jsonl")]
    assert trace[0]["event"] == "generation_start"
    assert trace[-1]["event"] == "generation_end"
    assert trace[0]["selection_mode"] == "opening_temperature"


def test_the_artifact_is_CREATE_ONLY(tmp_path):
    (tmp_path / "o.json").write_text("{}")
    with pytest.raises(FileExistsError):
        GEN._generate_unguarded(movers=_inert_movers(),
                                out_path=str(tmp_path / "o.json"),
                                trace_path=str(tmp_path / "t.jsonl"), n=1)
