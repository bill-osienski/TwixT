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
from scripts.GPU.alphazero import h3_generation_preflight as PF

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


def test_THE_BUILDER_RECEIVES_THE_SEAM_S_OWN_CONFIG_OBJECT(monkeypatch):
    """🔴 THE PILOT'S STRONGEST TEST, AND I DROPPED IT. `play.config is cfg` says
    only what the seam HOLDS; this says what the builder GETS.

    Every production act is inert -- the line the containment boundary draws --
    and the configuration is captured at `build_reference_agent`, the call the
    agent factory actually makes. `is`, not `==`: an equal object built by a
    second call is exactly the defect.
    """
    from scripts.GPU.alphazero import d1_probe as D1
    from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
    from scripts.GPU.alphazero import e4_screen_runner as HARNESS
    from scripts.GPU.alphazero import t1j_toolchain as TC
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3

    captured = {}

    def fake_build(*, task, evaluator, colour, config, capture=False):
        captured["config"] = config
        return "AGENT"

    monkeypatch.setattr(TC, "verified_paths",
                        lambda: {"jdk_home": "/nonexistent", "jar": "/none.jar"})
    monkeypatch.setattr(D1, "_default_compile", lambda deadline, paths: None)
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator",
                        lambda root: _stub_evaluator())
    monkeypatch.setattr(HARNESS, "play_task", lambda **kw: (_ for _ in ()).throw(
        AssertionError("no game is played by this test")))
    monkeypatch.setattr(G3, "build_reference_agent", fake_build)
    monkeypatch.setattr(RUN, "H3_STUDY_EXECUTION_AUTHORIZED", True)

    openings = R.stub_opening_set()
    seeded = R.build_tasks(openings, seed_interval=FRESH)
    deadline = D1.Deadline(60)
    deadline.start()
    cfg = RUN.frozen_argmax_config()
    play = RUN._production_play("/tmp/h3-study-never-written", deadline,
                                openings, cfg)
    with pytest.raises(AssertionError, match="no game is played"):
        play(task=seeded[0], identity={}, timeout_s=1.0)
    factory = play._state["agent_factory"]
    assert factory(seeded[0], seeded[0]["reference_colour"]) == "AGENT"
    assert captured["config"] is cfg, (
        "the builder must receive THE object the identity was read off, not an "
        "equal one")


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

class _Ctx:
    """The move log a T1jAgent requires — and `reset`, because the REAL
    `IntegrationContext` has it and the walk must call it before either agent can
    move. A double without `reset` would have hidden the very defect that VOIDed
    the one authorized generation attempt."""

    def __init__(self):
        self.moves = []
        self.task_id = None
        self.stats = {}

    def reset(self, task_id, opening):
        assert isinstance(task_id, str) and task_id, task_id
        self.task_id = task_id
        self.moves = [tuple(m) for m in opening]
        self.stats.setdefault(task_id, {"binds": 0, "t1j_queries": 0,
                                        "searched_binds": 0})


def _inert_movers(config=None):
    """FACTORIES, not move functions — because the card gives each opening ONE
    agent whose streams advance across its plies, and a per-move function cannot
    have streams at all."""
    calls = {"built_incumbent": 0, "built_t1j": 0, "moves": 0, "seeds": [],
             "ctxs": []}

    def incumbent_agent(*, seed, colour):
        calls["built_incumbent"] += 1
        calls["seeds"].append(seed)
        assert colour in ("red", "black")
        state = {"n": 0}

        def agent(st):
            # 🔑 STREAM-LIKE: the move depends on the seed AND on how many moves
            # this agent has already made, exactly as a persisting generator does.
            calls["moves"] += 1
            state["n"] += 1
            legal = sorted(st.legal_moves())
            return legal[(seed + state["n"]) % len(legal)]

        return agent

    def t1j_agent(*, colour, ctx):
        calls["built_t1j"] += 1
        calls["ctxs"].append(ctx)
        assert colour in ("red", "black")

        def agent(st):
            calls["moves"] += 1
            assert len(ctx.moves) == st.ply, (len(ctx.moves), st.ply)
            return sorted(st.legal_moves())[-1]

        return agent

    return {"incumbent_agent": incumbent_agent, "t1j_agent": t1j_agent,
            "new_context": _Ctx,
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
    assert movers["calls"]["moves"] == R.OPENING_PLIES
    assert op["seed"] == R.attempt_seed(R.GEN_SEED_CO_PRODUCED, 0, op["attempts"] - 1)
    assert not op["state"].is_terminal()


def test_ONE_AGENT_PER_OPENING_not_one_per_ply():
    """🔴 CARD §1.7.4. My first version built a FRESH agent every ply, so no
    stream advanced across an opening — and, to differentiate the plies, it
    offset the seed by `ply * 7919`, which pushed the seeds 47,514 BEYOND the
    declared generation range. That overrun is larger than the 40,800 gap between
    the two strata's ranges, so the collision proof over the declared ranges
    would have missed it and the two strata would have shared seeds."""
    movers = _inert_movers()
    GEN.generate_one(index=0, order=R.ORDER_INCUMBENT_FIRST, movers=movers)
    assert movers["calls"]["built_incumbent"] == 1, "ONE incumbent agent"
    assert movers["calls"]["built_t1j"] == 1, "ONE T1j agent"
    assert movers["calls"]["moves"] == R.OPENING_PLIES


def test_THE_AGENT_IS_SEEDED_WITH_THE_ATTEMPT_SEED_EXACTLY():
    """No offset, no derivation: the seed the registry accounts for is the seed
    the agent gets, so `generation_seed_range` describes what is touched."""
    movers = _inert_movers()
    op = GEN.generate_one(index=7, order=R.ORDER_T1J_FIRST, movers=movers)
    assert movers["calls"]["seeds"] == [op["seed"]]
    lo, hi = R.generation_seed_range(R.GEN_SEED_CO_PRODUCED)
    assert lo <= op["seed"] < hi


def test_the_INCUMBENT_PLAYS_ONE_COLOUR_THROUGHOUT_AN_OPENING():
    """Plies 1/3/5 are all red and 2/4/6 all black, so the alternating order
    fixes each engine's colour for the whole opening — which is what lets one
    agent serve all three of its moves."""
    for order, want in ((R.ORDER_INCUMBENT_FIRST, "red"),
                        (R.ORDER_T1J_FIRST, "black")):
        assert GEN.incumbent_colour(order) == want
        plies = [p for p in range(1, 7) if GEN.mover_at_ply(order, p) == "incumbent"]
        assert plies == ([1, 3, 5] if want == "red" else [2, 4, 6])


def test_the_T1J_AGENT_GETS_A_MOVE_LOG_THAT_TRACKS_THE_WALK():
    """T1jAgent refuses when the log length and the state's ply disagree, so the
    walk must maintain it — the inert agent asserts the same thing."""
    movers = _inert_movers()
    GEN.generate_one(index=0, order=R.ORDER_T1J_FIRST, movers=movers)
    ctx = movers["calls"]["ctxs"][0]
    assert len(ctx.moves) == R.OPENING_PLIES


def test_the_incumbent_FACTORY_TAKES_A_SEED_and_t1j_does_not():
    """🔑 The asymmetry IS the finding: only the incumbent supplies entropy."""
    import inspect as _i
    m = _inert_movers()
    assert "seed" in _i.signature(m["incumbent_agent"]).parameters
    assert "seed" not in _i.signature(m["t1j_agent"]).parameters


def test_A_DETERMINISTIC_INCUMBENT_PRODUCES_DUPLICATES_AND_IS_REFUSED(tmp_path):
    """🔴 THE ENTROPY FINDING, BOUND BEHAVIOURALLY.

    If the incumbent ignores its seed — which is exactly what `argmax` does, its
    move being a function of the position alone — then every opening of one order
    is the SAME opening. The walk does not quietly emit 74 copies: the
    distinctness guard refuses. This is the failure the card's §1.7.2 exists to
    prevent, reproduced in one test.
    """
    movers = _inert_movers()
    movers["incumbent_agent"] = lambda *, seed, colour: (
        lambda st: sorted(st.legal_moves())[0])      # deterministic: no seed
    with pytest.raises(GEN.H3GenerationError, match="duplicates an accepted"):
        GEN._generate_unguarded(movers=movers, out_path=str(tmp_path / "o.json"),
                                trace_path=str(tmp_path / "t.jsonl"), n=2)


def test_an_ILLEGAL_move_from_either_engine_is_REFUSED():
    movers = _inert_movers()
    movers["incumbent_agent"] = lambda *, seed, colour: (lambda st: (99, 99))
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
    # 3 between openings (no tree carried across one) + 1 unconditional teardown
    # in the `finally`. Two cleanups for two different reasons; dropping either is
    # its own defect, and each has its own control.
    assert len(cleanups) == 4, (
        f"expected 3 BETWEEN openings + 1 final teardown, got {cleanups}")
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


# ═══════════ PREPARATION for the generation run (card §1.7.7) ══════════════

def test_THE_OPENING_SET_DIGEST_IS_UNSET_AND_REFUSES():
    """🔴 A pin invented before the artifact exists pins nothing — it is either a
    guess the real set must match, or a value the generator is tempted to
    reproduce."""
    assert R.OPENING_SET_DIGEST is None
    with pytest.raises(R.H3StudyError, match="does not exist yet|AUTHORIZED"):
        R.expected_opening_set_digest()


def test_ATTEMPT_1s_DIRECTORY_AND_RANGE_ARE_BOTH_SPENT():
    """🔴 RECORDED BY THE VOID OF 2026-09-16. It produced no opening, and it is
    spent all the same: the first attempt BUILT AN AGENT on 20261200000 and put a
    query to T1j from it. A drawn seed is drawn whether or not an opening
    survived, and the directory holds the receipt and trace that prove it."""
    from scripts.GPU.alphazero import h3_generation_command as C
    failed = "docs/superpowers/evidence/2026-09-15-t1j-h3-study-openings"
    assert failed in C.SPENT_OUT_DIRS
    assert (20261200000, 20261259200) in R.SPENT_GENERATION_RANGES
    assert R.is_spent_generation_range(20261200000) is True
    with pytest.raises(R.H3StudyError, match="SPENT"):
        R.attempt_seed(20261200000, 0, 0)
    # and attempt 2's range is NOT spent
    assert R.is_spent_generation_range(R.GEN_SEED_CO_PRODUCED) is False


def test_THE_DESTINATION_IS_ABSENT_AND_OUTSIDE_EVERY_SPENT_DIRECTORY():
    """It is marked spent only AFTER an attempted run consumes it."""
    import os
    from scripts.GPU.alphazero import h3_study_command as CMD
    assert not os.path.lexists(GEN.OUT_DIR)
    assert not os.path.lexists(GEN.DEFAULT_OUT)
    assert not os.path.lexists(GEN.DEFAULT_TRACE)
    for spent in CMD.SPENT_OUT_DIRS:
        assert GEN.OUT_DIR != spent
        assert not GEN.OUT_DIR.startswith(spent.rstrip("/") + "/")
    assert GEN.OUT_DIR not in CMD.SPENT_OUT_DIRS, (
        "the destination is marked spent only after a run consumes it")


def test_the_ARTIFACT_SCHEMA_is_frozen_and_enforced():
    doc = {"design": "H3_FULL_STUDY_OPENINGS", "stratum": R.STRATUM_CO_PRODUCED,
           "n": 0, "selection_mode": "opening_temperature",
           "generation_note": "x", "config_pins": {}, "toolchain": {},
           "openings": [], "opening_set_digest": R.opening_set_digest([])}
    assert GEN.validate_artifact(doc)["n"] == 0
    for k in GEN.ARTIFACT_KEYS:
        short = {x: v for x, v in doc.items() if x != k}
        with pytest.raises(GEN.H3GenerationError, match="missing"):
            GEN.validate_artifact(short)


def test_the_artifact_REFUSES_an_ARGMAX_provenance():
    doc = {"design": "d", "stratum": R.STRATUM_CO_PRODUCED, "n": 0,
           "selection_mode": "argmax", "generation_note": "x",
           "config_pins": {}, "toolchain": {}, "openings": [],
           "opening_set_digest": R.opening_set_digest([])}
    with pytest.raises(GEN.H3GenerationError, match="no entropy"):
        GEN.validate_artifact(doc)


def test_the_artifact_REFUSES_a_STUB_opening_and_an_EDITED_digest():
    base = {"design": "d", "stratum": R.STRATUM_CO_PRODUCED, "n": 1,
            "selection_mode": "opening_temperature", "generation_note": "x",
            "config_pins": {}, "toolchain": {}}
    op = {"index": 0, "stratum": R.STRATUM_CO_PRODUCED, "order": "incumbent_first",
          "stub": True, "moves": [], "digest": "a" * 64, "seed": 1, "attempts": 1}
    doc = {**base, "openings": [op],
           "opening_set_digest": R.opening_set_digest([op])}
    with pytest.raises(GEN.H3GenerationError, match="STUB"):
        GEN.validate_artifact(doc)
    doc2 = {**base, "openings": [{**op, "stub": False}],
            "opening_set_digest": "b" * 64}
    with pytest.raises(GEN.H3GenerationError, match="edited"):
        GEN.validate_artifact(doc2)


def test_THE_PREFLIGHT_BUILDS_BOTH_MOVERS_AND_NEVER_MOVES():
    """🔑 REAL: verified_paths, a real T1jRuntime, a real T1jAgent, and the REAL
    `build_reference_agent`. NOT real: no model, no compile, no JVM, NO MOVE."""
    out = PF.preflight_movers()
    assert len(out["built"]) == 2
    orders = {b["order"] for b in out["built"]}
    assert orders == {R.ORDER_INCUMBENT_FIRST, R.ORDER_T1J_FIRST}
    for b in out["built"]:
        assert b["incumbent_seed"] == b["seed"], "the ATTEMPT seed, unoffset"
        assert b["t1j_depth"] == R.t1j_depth() == 6
        assert b["t1j_colour"] != b["incumbent_colour"]
        assert b["moves_made"] == 0, "NO MOVE WAS REQUESTED"
        # 🔑 the entropy fix, confirmed on the REAL builder's own agent
        assert b["readout"] == "opening_temperature" != "argmax"
    assert out["config_pins"]["selection_mode"] == "opening_temperature"
    assert out["toolchain"]["reference"]["name"] == "calib020_0001"


def test_the_preflight_agents_HOLD_THE_ONE_RUNTIME():
    out = PF.preflight_movers()
    ctx = out["movers"]["new_context"]()
    agent = out["movers"]["t1j_agent"](colour="red", ctx=ctx)
    assert agent.runtime is out["runtime"], "an equal runtime is not the runtime"


def test_the_preflight_REFUSES_if_the_generating_config_became_argmax(monkeypatch):
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    real = G3.eval_config()
    monkeypatch.setattr(G3, "eval_config",
                        lambda: real.__class__(**{**real.__dict__,
                                                  "selection_mode": "argmax"}))
    with pytest.raises(R.H3StudyError, match="no entropy|two positions"):
        PF.preflight_movers()


def test_the_toolchain_identity_names_the_VERIFIED_jar_and_jdk():
    t = GEN.toolchain_identity()
    assert t["jar"].endswith(".jar") and t["jdk_home"]
    assert t["verified"], "verified_paths reports what it checked"
    assert t["t1j_mdPly"] == R.t1j_depth()


# ═══════════ THE SUPERVISED GENERATION WRAPPER ═════════════════════════════
# 🔴 Generation had NO launch path: the public entry drove the production
# collaborators directly, with no outer deadline, no process group, nothing to
# stop a surviving JVM and no unconditional gate restoration.

from scripts.GPU.alphazero import h3_generation_command as GCMD


def test_the_generation_wrapper_REFUSES_with_the_gate_shut_and_verifies_it(capsys):
    assert GCMD.main([]) == GCMD.EXIT_UNAUTHORIZED
    assert "NOT AUTHORIZED" in capsys.readouterr().err
    assert GEN.H3_GENERATION_AUTHORIZED is False
    assert GCMD.restore_gate() is True


def test_the_generation_wrapper_has_NO_gate_or_source_override():
    help_text = GCMD._parser().format_help()
    for flag in ("--authorize", "--force", "--gate", "--runner-source"):
        assert flag not in help_text, flag
    assert "os.environ" not in open(GCMD.__file__, encoding="utf-8").read()


def test_restore_gate_REWRITES_an_open_gate_and_VERIFIES_it(tmp_path):
    src = tmp_path / "gen.py"
    src.write_text("x = 1\nH3_GENERATION_AUTHORIZED = True\ny = 2\n")
    assert GCMD.restore_gate(str(src)) is True
    back = src.read_text()
    assert "H3_GENERATION_AUTHORIZED = False" in back
    assert "H3_GENERATION_AUTHORIZED = True" not in back


def test_restore_gate_is_FALSE_when_it_cannot_verify(tmp_path):
    missing = tmp_path / "nope.py"
    assert GCMD.restore_gate(str(missing)) is False
    two = tmp_path / "two.py"
    two.write_text("H3_GENERATION_AUTHORIZED = True\nH3_GENERATION_AUTHORIZED = True\n")
    assert GCMD.restore_gate(str(two)) is False


def test_a_FAILED_RESTORATION_becomes_the_wrappers_OWN_exit_code(monkeypatch, capsys):
    """🔑 IT SUPERSEDES EVERY OTHER CODE, including a refusal: an open gate is the
    larger fact."""
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: False)
    assert GCMD.main([]) == GCMD.EXIT_GATE_NOT_RESTORED
    assert "BY HAND" in capsys.readouterr().err


@pytest.mark.parametrize("r,want", [
    ({"timed_out": False, "interrupted": False, "group_cleared": True,
      "exit_code": 0}, 0),
    ({"timed_out": True, "interrupted": False, "group_cleared": True,
      "exit_code": 6}, 6),
    ({"timed_out": False, "interrupted": True, "group_cleared": True,
      "exit_code": 9}, 9),
    # 🔴 A SURVIVING DESCENDANT OUTRANKS EVERYTHING, even exit 0
    ({"timed_out": False, "interrupted": False, "group_cleared": False,
      "exit_code": 0}, 8),
    ({"timed_out": True, "interrupted": False, "group_cleared": False,
      "exit_code": 6}, 8),
])
def test_every_supervisor_outcome_gets_ITS_OWN_exit_code(monkeypatch, tmp_path,
                                                         r, want):
    # 🔴 THE RECEIPT PATH MUST BE REDIRECTED. Without this the test wrote a real
    # 00_launch_receipt.json into the RUN'S OWN DESTINATION -- a test artifact
    # sitting exactly where the authorized generation is meant to write, which
    # would then have refused the real launch as "already exists".
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: dict(r))
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    assert code == want


def test_the_wrapper_REFUSES_BEFORE_SPAWNING_when_the_destination_exists(
        monkeypatch, tmp_path):
    spawned = []
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "supervise",
                        lambda *a, **k: spawned.append(1) or {
                            "timed_out": False, "interrupted": False,
                            "group_cleared": True, "exit_code": 0})
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    (tmp_path / "o.json").write_text("{}")
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    assert code == GCMD.EXIT_REFUSED
    assert spawned == [], "nothing may be spawned once the destination is taken"


def test_the_destination_INSIDE_A_SPENT_DIRECTORY_is_refused(monkeypatch):
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    spent = GCMD.SPENT_OUT_DIRS[0]
    with pytest.raises(GEN.H3GenerationError, match="SPENT"):
        GCMD._check_destination(f"{spent}/o.json", f"{spent}/t.jsonl")


def test_THE_WORKER_REFUSES_without_the_supervisors_CAPABILITY(capsys):
    assert GCMD.worker_main(["--worker"]) == GCMD.EXIT_REFUSED
    assert "UNSUPERVISED" in capsys.readouterr().err


def test_the_capability_is_a_PIPE_and_DELETES_NOTHING(tmp_path):
    fd = GCMD._make_capability()
    assert GCMD._consume_capability(fd) is True
    assert GCMD._consume_capability(None) is False
    victim = tmp_path / "precious.txt"
    victim.write_text("x" * (GCMD.CAPABILITY_BYTES * 2))
    assert GCMD._consume_capability(str(victim)) is False
    assert victim.exists(), "the capability check must never delete a file"


# ── the generator's own terminal semantics, driven with INERT collaborators

def _gen_paths(tmp_path):
    return str(tmp_path / "o.json"), str(tmp_path / "t.jsonl")


def _trace(path):
    import json
    return [json.loads(l) for l in open(path)]


def test_CLEANUP_RUNS_EVEN_WHEN_THE_WALK_RAISES(tmp_path):
    """🔴 It used to run only after an ACCEPTED opening, so a raise anywhere --
    agent construction, a move, validation, Ctrl-C -- skipped it and could leave
    production collaborators alive."""
    cleanups = []
    movers = _inert_movers()
    movers["incumbent_agent"] = lambda *, seed, colour: (_ for _ in ()).throw(
        RuntimeError("agent construction failed"))
    out, trace = _gen_paths(tmp_path)
    with pytest.raises(RuntimeError, match="agent construction"):
        GEN._generate_unguarded(movers=movers, out_path=out, trace_path=trace,
                                cleanup=lambda: cleanups.append(1), n=2)
    assert cleanups == [1], "the finally tore down exactly once"
    end = _trace(trace)[-1]
    assert end["event"] == "generation_end" and end["verdict"] == "VOID"
    assert "agent construction failed" in end["failure"]


def test_AN_INTERRUPT_IS_A_TERMINAL_STATUS_and_still_cleans_up(tmp_path):
    cleanups = []
    movers = _inert_movers()
    movers["incumbent_agent"] = lambda *, seed, colour: (_ for _ in ()).throw(
        KeyboardInterrupt())
    out, trace = _gen_paths(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        GEN._generate_unguarded(movers=movers, out_path=out, trace_path=trace,
                                cleanup=lambda: cleanups.append(1), n=2)
    assert cleanups == [1]
    end = _trace(trace)[-1]
    assert end["verdict"] == "INTERRUPTED"


def test_A_CLEANUP_FAILURE_IS_ITS_OWN_TERMINAL_OUTCOME(tmp_path):
    """Even when the body SUCCEEDED: a population made and a JVM left alive is
    not a success."""
    def boom():
        raise OSError("the JVM would not die")
    out, trace = _gen_paths(tmp_path)
    with pytest.raises(GEN.H3GenerationCleanupError, match="would not die"):
        GEN._generate_unguarded(movers=_inert_movers(), out_path=out,
                                trace_path=trace, cleanup=boom, n=2)
    end = _trace(trace)[-1]
    assert end["verdict"] == "CLEANUP_FAILED" and end["cleanup_ok"] is False


def test_A_DEADLINE_CAPS_THE_WHOLE_LOOP_and_writes_NO_artifact(tmp_path):
    """🔴 The only deadline used to be the COMPILE's, which said nothing about a
    generation that walks for ever. And a partial population is not a population:
    the artifact is not written at all."""
    import os as _os

    class _Expired:
        started = True

        def start(self):
            pass

        def elapsed(self):
            return 1e9

    out, trace = _gen_paths(tmp_path)
    with pytest.raises(GEN.H3GenerationError, match="deadline"):
        GEN._generate_unguarded(movers=_inert_movers(), out_path=out,
                                trace_path=trace, cleanup=lambda: None,
                                deadline=_Expired(), deadline_s=1.0, n=3)
    assert not _os.path.lexists(out), "no partial population is ever written"
    end = _trace(trace)[-1]
    assert end["verdict"] == "TIMEOUT" and end["timed_out"] is True


def test_EVERY_TERMINAL_RECORD_NAMES_THE_WHOLE_RANGE_AS_RETIRED(tmp_path):
    """🔴 Attempts consume generation seeds whether or not the opening was
    accepted, so there is no partial retirement to argue about."""
    out, trace = _gen_paths(tmp_path)
    GEN._generate_unguarded(movers=_inert_movers(), out_path=out,
                            trace_path=trace, cleanup=lambda: None, n=2)
    end = _trace(trace)[-1]
    assert end["verdict"] == "OK"
    assert end["retires"] == list(R.generation_seed_range(R.GEN_SEED_CO_PRODUCED))
    assert "WHOLE RANGE" in end["retirement_rule"]
    # and on a failure path too
    out2, trace2 = str(tmp_path / "o2.json"), str(tmp_path / "t2.jsonl")
    movers = _inert_movers()
    movers["incumbent_agent"] = lambda *, seed, colour: (_ for _ in ()).throw(
        RuntimeError("x"))
    with pytest.raises(RuntimeError):
        GEN._generate_unguarded(movers=movers, out_path=out2, trace_path=trace2,
                                cleanup=lambda: None, n=1)
    assert _trace(trace2)[-1]["retires"] == end["retires"]


def test_THE_FINALLY_PATH_also_restores_the_generation_gate(monkeypatch, tmp_path):
    """🔑 A DIFFERENT PATH FROM THE REFUSAL'S. With the gate SHUT the wrapper
    returns before the try/finally, so only a run that gets past the gate
    exercises the `finally`'s restoration — and a control that removes it is
    invisible to the refusal test."""
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: {
        "timed_out": False, "interrupted": False, "group_cleared": True,
        "exit_code": 0})
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: False)
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    assert code == GCMD.EXIT_GATE_NOT_RESTORED, (
        "a failed restoration SUPERSEDES the worker's own exit 0")


# ═══════════ the GENERATION DEADLINE, preregistered on its own ═════════════

def test_GENERATION_HAS_ITS_OWN_PREREGISTERED_DEADLINE():
    """🔴 It was reusing SEGMENT_DEADLINE_S — a cap frozen for 148 GAMES at the
    pilot's 41.22 s/game, which says nothing about 148 six-ply generation walks.
    Same number, different quantity: a bound borrowed from something else."""
    assert R.GENERATION_DEADLINE_S == 10800
    assert "GENERATION_DEADLINE_S" in open(
        GEN.__file__, encoding="utf-8").read()
    src = open(GCMD.__file__, encoding="utf-8").read()
    assert "GENERATION_DEADLINE_S" in src
    assert "SEGMENT_DEADLINE_S" not in src, (
        "the generation wrapper must not borrow the match segment's cap")


def test_the_generation_deadline_RATIONALE_is_recorded_and_arithmetically_sound():
    """~0.81 s/ply from the pilot -> ~4.85 s per 6-ply attempt -> ~718 s for 148
    at one attempt each, so 10,800 s allows ~15 attempts per opening."""
    per_ply = 41.22 / 51
    one_pass = R.PAIRS_PER_STRATUM * R.OPENING_PLIES * per_ply
    assert 700 < one_pass < 740, one_pass
    assert 14 < R.GENERATION_DEADLINE_S / one_pass < 16
    doc = open("scripts/GPU/alphazero/h3_study_rules.py", encoding="utf-8").read()
    i = doc.index("GENERATION_DEADLINE_S = ")
    rationale = doc[max(0, i - 1600):i]
    assert "CHOSEN" in rationale and "MAY EXPIRE" in rationale
    assert "RETIRES THE WHOLE GENERATION RANGE" in rationale


# ═══════════ the PARENT-OWNED DURABLE RECEIPT ══════════════════════════════
# 🔴 The worker writes `generation_end` in its own `finally` -- but the outer
# supervisor can kill its process group before that runs, and the outcomes that
# SUPERSEDE the worker's are decided by the PARENT afterwards.

def _launch(monkeypatch, tmp_path, *, sup=None, restore=True, gate=True):
    """Drive the wrapper's parent path with the supervisor stubbed."""
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "00_launch_receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: gate)
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: restore)
    if sup is not None:
        monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: dict(sup))
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    import json
    path = tmp_path / "00_launch_receipt.json"
    doc = json.loads(path.read_text()) if path.exists() else None
    return code, doc


OK_SUP = {"timed_out": False, "interrupted": False, "group_cleared": True,
          "exit_code": 0}


def test_NO_RECEIPT_WHEN_THE_GATE_WAS_SHUT(monkeypatch, tmp_path):
    """Nothing was attempted and nothing consumed; a receipt would claim a launch
    that never happened."""
    code, doc = _launch(monkeypatch, tmp_path, gate=False)
    assert code == GCMD.EXIT_UNAUTHORIZED and doc is None


def test_THE_RECEIPT_EXISTS_EVEN_WHEN_THE_WORKER_WROTE_NO_TRACE(monkeypatch,
                                                                tmp_path):
    """🔑 THE CASE THE WORKER'S OWN `finally` CANNOT COVER: killed before it ever
    opened its trace."""
    code, doc = _launch(monkeypatch, tmp_path,
                        sup={"timed_out": True, "interrupted": False,
                             "group_cleared": True, "exit_code": -9})
    assert code == GCMD.EXIT_TIMEOUT
    assert doc is not None
    assert doc["outcome"] == "TIMEOUT" and doc["timed_out"] is True
    assert doc["trace_exists"] is False and doc["artifact_exists"] is False
    assert doc["worker_exit"] == -9


@pytest.mark.parametrize("sup,restore,outcome,code_name", [
    (OK_SUP, True, "COMPLETED", "EXIT_COMPLETED"),
    ({**OK_SUP, "timed_out": True, "exit_code": 6}, True, "TIMEOUT", "EXIT_TIMEOUT"),
    ({**OK_SUP, "interrupted": True, "exit_code": 9}, True, "INTERRUPTED",
     "EXIT_INTERRUPTED"),
    ({**OK_SUP, "group_cleared": False}, True, "CLEANUP_FAILED",
     "EXIT_CLEANUP_FAILED"),
    (OK_SUP, False, "GATE_NOT_RESTORED", "EXIT_GATE_NOT_RESTORED"),
])
def test_THE_RECEIPT_RECORDS_THE_SUPERSEDING_OUTCOME(monkeypatch, tmp_path, sup,
                                                     restore, outcome, code_name):
    code, doc = _launch(monkeypatch, tmp_path, sup=sup, restore=restore)
    assert code == getattr(GCMD, code_name)
    assert doc["outcome"] == outcome
    assert doc["exit_code"] == code
    assert doc["gate_restored"] is restore
    assert doc["group_cleared"] is sup["group_cleared"]


def test_A_SURVIVING_DESCENDANT_SUPERSEDES_THE_WORKERS_OWN_SUCCESS(monkeypatch,
                                                                   tmp_path):
    """The worker exited 0; a descendant lived. The receipt says CLEANUP_FAILED
    and keeps the worker's 0 beside it, so neither fact is lost."""
    code, doc = _launch(monkeypatch, tmp_path,
                        sup={**OK_SUP, "group_cleared": False, "exit_code": 0})
    assert code == GCMD.EXIT_CLEANUP_FAILED
    assert doc["outcome"] == "CLEANUP_FAILED"
    assert doc["worker_exit"] == 0, "the worker's own result is still recorded"


def test_THE_RECEIPT_READS_THE_GATE_BACK_FROM_THE_FILE(monkeypatch, tmp_path):
    """🔑 Not `restore_gate`'s opinion of itself: a receipt that records its own
    say-so records nothing."""
    code, doc = _launch(monkeypatch, tmp_path, sup=OK_SUP)
    assert doc["gate_readback"] == "False"
    assert GCMD._gate_readback(str(tmp_path / "absent.py")) == "UNREADABLE"
    open_src = tmp_path / "open.py"
    open_src.write_text("H3_GENERATION_AUTHORIZED = True\n")
    assert GCMD._gate_readback(str(open_src)) == "True"


def test_THE_RECEIPT_NAMES_THE_WHOLE_RANGE_AS_RETIRED(monkeypatch, tmp_path):
    for sup in (OK_SUP, {**OK_SUP, "timed_out": True, "exit_code": 6}):
        d = tmp_path / str(sup["exit_code"])
        d.mkdir()
        _, doc = _launch(monkeypatch, d, sup=sup)
        assert doc["retires"] == list(
            R.generation_seed_range(R.GEN_SEED_CO_PRODUCED))
        assert "WHOLE RANGE" in doc["retirement_rule"]
        assert doc["deadline_s"] == R.GENERATION_DEADLINE_S
        assert doc["outer_cap_s"] == (R.GENERATION_DEADLINE_S
                                      + GCMD.SUPERVISOR_GRACE_S), (
            "the OUTER cap must exceed the worker's own deadline, or the "
            "supervisor kills a worker that was about to stop cleanly")


def test_a_REFUSAL_BEFORE_SPAWNING_still_leaves_a_receipt(monkeypatch, tmp_path):
    """An authorization was spent on a run that could not start -- H2's lesson,
    where exactly that went unrecorded."""
    (tmp_path / "o.json").write_text("{}")
    code, doc = _launch(monkeypatch, tmp_path, sup=OK_SUP)
    assert code == GCMD.EXIT_REFUSED
    assert doc["outcome"] == "REFUSED" and doc["supervised"] is False
    assert "already exists" in doc["note"]


def test_THE_RECEIPT_IS_CREATE_ONLY_and_a_SECOND_LAUNCH_is_refused(monkeypatch,
                                                                   tmp_path):
    code, doc = _launch(monkeypatch, tmp_path, sup=OK_SUP)
    assert code == GCMD.EXIT_COMPLETED and doc is not None
    # the receipt now occupies the destination: a second launch must refuse
    code2, _ = _launch(monkeypatch, tmp_path, sup=OK_SUP)
    assert code2 == GCMD.EXIT_REFUSED


def test_NO_TEST_MAY_WRITE_INTO_THE_RUNS_OWN_DESTINATION():
    """🔴 A test DID. `test_every_supervisor_outcome_gets_ITS_OWN_exit_code` did
    not redirect `RECEIPT`, so running the suite created a real
    `00_launch_receipt.json` in the destination the authorized generation is
    meant to write — which would then have refused the real launch as "already
    exists". An authorization spent because a TEST occupied the destination is
    exactly H2's defect, arriving by a new road.

    Every wrapper test that reaches the receipt must patch the path first.
    """
    import ast
    import inspect as _i
    src = open(__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        body = ast.get_source_segment(src, fn) or ""
        # only a test that FORCES THE GATE OPEN can reach the receipt at all:
        # with the gate shut the wrapper returns before the try/finally.
        reaches_receipt = ("GCMD.main(" in body
                           and 'GCMD, "gate_is_open", lambda: True' in body)
        redirects = 'GCMD, "RECEIPT"' in body or "_launch(" in body
        if reaches_receipt and not redirects:
            offenders.append(fn.name)
    assert offenders == [], offenders
    import os
    assert not os.path.lexists(GEN.OUT_DIR), (
        f"{GEN.OUT_DIR} exists: a test wrote into the run's destination")


def test_O_EXCL_refuses_the_receipt_EVEN_WITH_THE_PRECHECK_DISABLED(monkeypatch,
                                                                    tmp_path):
    """🔑 THE PRECHECK CATCHES THE SECOND LAUNCH FIRST, so the create-only flag on
    the write itself is never exercised by that path — and a control removing
    `O_EXCL` went NOT CAUGHT. This reaches it alone: with `_check_destination`
    neutralised, the write must STILL refuse rather than overwrite a receipt that
    is the durable record of an earlier launch."""
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"outcome": "AN EARLIER LAUNCH"}')
    monkeypatch.setattr(GCMD, "RECEIPT", str(receipt))
    monkeypatch.setattr(GCMD, "_check_destination", lambda *a, **k: None)
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: dict(OK_SUP))
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    import json
    assert json.loads(receipt.read_text())["outcome"] == "AN EARLIER LAUNCH", (
        "the earlier launch's receipt was OVERWRITTEN")
    assert code == GCMD.EXIT_UNEXPECTED, (
        "a receipt that could not be written must not report success")


# ═══════════ THE CONTEXT RESET — the defect that VOIDed the one attempt ════

def _t1j_record(move, depth):
    class _R:
        completed = True
        requested_depth = depth
        completed_depth = depth
        null_sentinel = False
        legal = True
    r = _R()
    r.move = move
    return r


def _drive_t1j_agent(monkeypatch, ctx, *, depth=6):
    """The FIRST REAL `T1jAgent.__call__`, with an inert runtime: no JVM, no
    process, no toolchain — but the real control flow, including the counter."""
    from scripts.GPU.alphazero import e4_screen_integration as INT
    st = R._fresh_state()
    move = sorted(st.legal_moves())[0]
    monkeypatch.setattr(INT, "check_postcond", lambda *a, **k: None)
    monkeypatch.setattr(INT, "compare_state", lambda *a, **k: [])
    agent = INT.T1jAgent(
        runtime=INT.T1jRuntime(java="/nonexistent", jar="/none.jar",
                               classes="/none", ply_cap=280, timeout_s=1.0),
        ctx=ctx, depth=depth, colour=st.to_move, timeout_s=1.0,
        _query=lambda *a, **k: ([_t1j_record(move, depth)], [object()], 0, ""))
    return agent, agent(st), move


def test_A_BARE_CONTEXT_RAISES_ON_THE_FIRST_T1J_MOVE(monkeypatch):
    """🔴 THE DEFECT ITSELF, bound as a test. It cost the one authorized
    generation attempt: 0 of 148 openings, VOID in 3 seconds.

    Construction succeeds — which is why a preflight that stops at construction
    cannot see it. `bump` runs during the MOVE.
    """
    from scripts.GPU.alphazero import e4_screen_integration as INT
    bare = INT.IntegrationContext()
    assert bare.task_id is None and bare.stats == {}
    with pytest.raises(KeyError):
        _drive_t1j_agent(monkeypatch, bare)


def test_THE_FIRST_REAL_T1J_CALL_COUNTS_ITS_QUERY(monkeypatch):
    """🔑 `t1j_queries == 1`, through the real `__call__`, not construction."""
    from scripts.GPU.alphazero import e4_screen_integration as INT
    ctx = INT.IntegrationContext()
    ctx.reset("h3gen-000-a000-s20261400000", [])
    agent, got, want = _drive_t1j_agent(monkeypatch, ctx)
    assert got == want
    assert ctx.stats["h3gen-000-a000-s20261400000"]["t1j_queries"] == 1
    assert ctx.stats["h3gen-000-a000-s20261400000"]["searched_binds"] == 1
    assert agent.moves_made == 1


def test_THE_WALK_RESETS_ITS_CONTEXT_BEFORE_EITHER_AGENT_MOVES():
    """The reset must happen between creating the context and building the
    agents — an agent built first could be called before the reset."""
    import ast
    import inspect as _i
    import textwrap
    src = textwrap.dedent(_i.getsource(GEN.generate_one))
    fn = ast.parse(src).body[0]
    lines = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        # `movers["new_context"]()` is a SUBSCRIPT call, not an attribute or a
        # name -- my first version looked only at the latter two and saw nothing.
        name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
        if isinstance(node.func, ast.Subscript):
            key = getattr(node.func.slice, "value", None)
            name = key if isinstance(key, str) else name
        if name in ("new_context", "reset", "incumbent_agent", "t1j_agent"):
            lines.setdefault(name, node.lineno)
    for needed in ("new_context", "reset", "incumbent_agent", "t1j_agent"):
        assert needed in lines, (needed, lines)
    assert lines["new_context"] < lines["reset"], lines
    assert lines["reset"] < lines["incumbent_agent"], lines
    assert lines["reset"] < lines["t1j_agent"], lines


def test_EVERY_ATTEMPT_GETS_ITS_OWN_CONTEXT_IDENTITY():
    """🔴 NOT A SHARED CONSTANT. `stats` is keyed by task_id and never cleared,
    so one shared identity would pool every attempt's counters into one bucket —
    the very defect `IntegrationContext` documents."""
    ids = []

    class _Ctx2:
        def __init__(self):
            self.moves = []
            self.task_id = None
            self.stats = {}

        def reset(self, task_id, opening):
            ids.append(task_id)
            self.task_id = task_id
            self.moves = list(opening)
            self.stats.setdefault(task_id, {"binds": 0, "t1j_queries": 0,
                                            "searched_binds": 0})

    movers = _inert_movers()
    movers["new_context"] = _Ctx2
    GEN.generate_one(index=3, order=R.ORDER_INCUMBENT_FIRST, movers=movers)
    GEN.generate_one(index=4, order=R.ORDER_T1J_FIRST, movers=movers)
    assert len(ids) == len(set(ids)) == 2, ids
    assert all(i and isinstance(i, str) for i in ids)
    assert "003" in ids[0] and "004" in ids[1], ids
    assert all(str(R.GEN_SEED_CO_PRODUCED) in i or "-s" in i for i in ids)
