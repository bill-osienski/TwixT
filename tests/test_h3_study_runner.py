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
#: four blocks now, one per segment -- the study no longer uses a
#: single interval, so a test schedule must not either.
FRESH_BLOCKS = tuple(
    (777000000 + R.GAMES_PER_SEGMENT * k,
     777000000 + R.GAMES_PER_SEGMENT * (k + 1))
    for k in range(R.N_SEGMENTS))


# ───────────────────────── BOTH gates are shut ─────────────────────────────













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






























# ═══════════ PREPARATION for the generation run (card §1.7.7) ══════════════

def test_THE_OPENING_SET_DIGEST_IS_THE_FROZEN_ARTIFACTS_OWN(monkeypatch):
    """🔴 A pin invented before the artifact exists pins nothing — it is either a
    guess the real set must match, or a value the generator is tempted to
    reproduce. It is now SET, and set to the artifact's own value.

    The refusal it replaced is still tested, by unsetting the pin: that path has
    to keep working, because it is what stops a study running over an unpinned
    population.
    """
    import json
    assert R.OPENING_SET_DIGEST == (
        "35932b3fabd9c6463d615b0b3af380134dadd700e2ca1e882a0c863faf772e46")
    doc = json.load(open(RUN.OPENING_SET_PATH, encoding="utf-8"))
    assert doc["opening_set_digest"] == R.OPENING_SET_DIGEST
    assert R.expected_opening_set_digest() == R.OPENING_SET_DIGEST

    monkeypatch.setattr(R, "OPENING_SET_DIGEST", None)
    with pytest.raises(R.H3StudyError, match="not PINNED"):
        R.expected_opening_set_digest()




def test_THE_FROZEN_DESTINATION_IS_SEPARATE_FROM_EVERY_SPENT_RUN():
    """The "absent" half is spent: the freeze CONSUMED this destination on
    2026-09-17 and it now holds the population. The half that still binds is
    that it is its own place — no earlier run's directory, and no segment's."""
    import os
    from scripts.GPU.alphazero import h3_study_command as CMD
    assert os.path.lexists(GEN.DEFAULT_OUT) and os.path.lexists(GEN.DEFAULT_TRACE)
    for spent in CMD.SPENT_OUT_DIRS:
        assert GEN.OUT_DIR != spent
        assert not GEN.OUT_DIR.startswith(spent.rstrip("/") + "/")
        assert not spent.startswith(GEN.OUT_DIR.rstrip("/") + "/")
    for k in range(R.N_SEGMENTS):
        d = RUN.segment_out_dir(k)
        assert d != GEN.OUT_DIR and not d.startswith(GEN.OUT_DIR.rstrip("/") + "/"), (
            "a segment would write into the frozen population's directory")
















# ═══════════ THE SUPERVISED GENERATION WRAPPER ═════════════════════════════
# 🔴 Generation had NO launch path: the public entry drove the production
# collaborators directly, with no outer deadline, no process group, nothing to
# stop a surviving JVM and no unconditional gate restoration.

from scripts.GPU.alphazero import h3_generation_command as GCMD






















# ── the generator's own terminal semantics, driven with INERT collaborators

def _gen_paths(tmp_path):
    return str(tmp_path / "o.json"), str(tmp_path / "t.jsonl")


def _trace(path):
    import json
    return [json.loads(l) for l in open(path)]














# ═══════════ the GENERATION DEADLINE, preregistered on its own ═════════════





# ═══════════ the PARENT-OWNED DURABLE RECEIPT ══════════════════════════════
# 🔴 The worker writes `generation_end` in its own `finally` -- but the outer
# supervisor can kill its process group before that runs, and the outcomes that
# SUPERSEDE the worker's are decided by the PARENT afterwards.



OK_SUP = {"timed_out": False, "interrupted": False, "group_cleared": True,
          "exit_code": 0}






















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






@pytest.fixture(scope="module")
def tasks():
    """The 592 tasks over the real in-memory population -- no stub anywhere.

    🔑 THE STUB IS GONE AND THAT IS THE POINT. Under the two-stratum design the
    schedule could only be tested against a PLACEHOLDER, because half the
    population came from a run that had not happened. Uniform generation is
    engine-free, so the schedule is now built over the REAL openings.
    """
    return R.build_tasks(GEN.build_population(), seed_blocks=FRESH_BLOCKS)


def test_THE_GATE_AND_THE_BARRIER_ARE_BOTH_SHUT_IN_THE_REAL_REPOSITORY():
    assert RUN.H3_STUDY_EXECUTION_AUTHORIZED is False
    assert GEN.H3_POPULATION_FREEZE_AUTHORIZED is False


def test_THE_GATE_AND_THE_BARRIER_ARE_SEPARATE_and_neither_implies_the_other():
    """🔑 Generating stratum B is a RUN with its own authorization (card §1.5).
    One switch for both would let a generation approval authorize a match."""
    assert RUN.__dict__ is not GEN.__dict__
    run_src = open(RUN.__file__, encoding="utf-8").read()
    gen_src = open(GEN.__file__, encoding="utf-8").read()
    assert "H3_GENERATION_AUTHORIZED" not in run_src
    assert "H3_STUDY_EXECUTION_AUTHORIZED" not in gen_src


def test_the_entry_READS_THE_GATE_FIRST_by_AST():
    fn = ast.parse(inspect.getsource(RUN.run_segment)).body[0]
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    first = next(c for c in calls if getattr(c.func, "id", ""))
    assert getattr(first.func, "id", "") == "check_gate", ast.dump(first)


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

    openings = GEN.build_population()
    seeded = R.build_tasks(openings, seed_blocks=FRESH_BLOCKS)
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


def test_a_SEEDLESS_schedule_is_REFUSED():
    tasks = R.build_tasks(GEN.build_population())
    seg = RUN.segment_schedule(tasks, 0)
    with pytest.raises(RUN.H3StudyRunError, match="carry no seed"):
        RUN.check_segment_schedule(seg, 0, RUN.segment_digest(tasks, 0))




def test_A_RETIRED_SEGMENT_DOES_NOT_BLOCK_A_LATER_ONE(monkeypatch):
    """🔴 THE COUPLING THIS REPAIR REMOVED. Segment 0's quarter was retired on
    its VOID and the whole study became unlaunchable -- segments 1-3 included,
    though their seeds were untouched."""
    retired = RUN.RETIRED_SEGMENT_BLOCKS[0]
    monkeypatch.setattr(RUN, "SEGMENT_SEED_BLOCKS",
                        (retired,) + RUN.SEGMENT_SEED_BLOCKS[1:])
    with pytest.raises(RUN.H3StudyRunError, match="RETIRED"):
        RUN.check_segment_seeds(0)
    for k in (1, 2, 3):
        assert RUN.check_segment_seeds(k)["segment"] == k


def test_THE_RETIRED_BLOCK_CAN_NEVER_BE_RELAUNCHED():
    """The decoupling did not make a spent block revivable."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    for lo, hi in RUN.RETIRED_SEGMENT_BLOCKS:
        assert all(REF.seed_status(s)["retired"] for s in range(lo, hi))
        assert not any(REF.seed_status(s)["exposed"] for s in range(lo, hi))
        for a, b in RUN.SEGMENT_SEED_BLOCKS:
            assert hi <= a or b <= lo, "a live block overlaps the retired one"


def test_THE_RETIRED_OVERLAP_CHECK_IS_ITS_OWN_GUARD(monkeypatch):
    """🔴 TWO GUARDS REFUSE THE RETIRED BLOCK, and the harness showed a test that
    could not tell them apart.

    `check_segment_seeds` refuses on (a) overlapping RETIRED_SEGMENT_BLOCKS and
    (b) the seeds' registry STATUS. The retired quarter trips BOTH, so removing
    either one still produced a refusal whose message matched "RETIRED" -- and
    the injections were NOT CAUGHT.

    Each is now driven alone: a block that overlaps the retired range but whose
    seeds are NOT registry-retired, and a block that is registry-spent but
    overlaps nothing.
    """
    # (a) overlaps the retired range, seeds NOT retired in the registry
    monkeypatch.setattr(RUN, "RETIRED_SEGMENT_BLOCKS",
                        ((202_628_000, 202_628_148),))       # segment 0's LIVE block
    with pytest.raises(RUN.H3StudyRunError, match="may never be revived"):
        RUN.check_segment_seeds(0)

    # (b) registry-spent, overlapping no declared retired range
    monkeypatch.setattr(RUN, "RETIRED_SEGMENT_BLOCKS", ())
    monkeypatch.setattr(RUN, "SEGMENT_SEED_BLOCKS",
                        ((202_626_000, 202_626_148),) + RUN.SEGMENT_SEED_BLOCKS[1:])
    with pytest.raises(RUN.H3StudyRunError, match="is SPENT"):
        RUN.check_segment_seeds(0)


def test_REGISTRATION_IS_ASKED_OF_ONE_SEGMENT_WHEN_ONE_IS_NAMED(monkeypatch):
    """🔴 `check_seed_registration(segment)` must look at THAT segment only.

    The harness caught this: with every block registered, widening the loop back
    to all four changed nothing observable. It needs a segment whose block is
    UNREGISTERED while the others are fine.
    """
    unreg = (909_090_000, 909_090_148)
    monkeypatch.setattr(RUN, "SEGMENT_SEED_BLOCKS",
                        (unreg,) + RUN.SEGMENT_SEED_BLOCKS[1:])
    with pytest.raises(RUN.H3StudyRunError, match="not registered"):
        RUN.check_seed_registration(0)
    for k in (1, 2, 3):
        RUN.check_seed_registration(k)          # unaffected -- the repair
    with pytest.raises(RUN.H3StudyRunError, match="not registered"):
        RUN.check_seed_registration()           # None = all four, so it still fires


def test_run_segment_CHECKS_THE_SEGMENTS_SEEDS_BEFORE_ANYTHING_ELSE():
    """🔴 THE LAUNCH CHECK MUST BE ON THE LAUNCH PATH.

    Removing `check_segment_seeds(segment)` from `run_segment` was NOT CAUGHT:
    no test drove that entry. It is checked structurally here -- it must be
    called, and before the openings are loaded.
    """
    import ast
    src = open(RUN.__file__, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "run_segment")
    body = ast.get_source_segment(src, fn) or ""
    assert "check_segment_seeds(segment)" in body
    assert body.index("check_segment_seeds(segment)") < body.index("load_opening_set")


def test_run_segment_REFUSES_A_SEGMENT_WHOSE_SEEDS_ARE_SPENT(monkeypatch):
    """…and behaviourally: with the gate forced open, a spent segment refuses
    before any output is touched."""
    monkeypatch.setattr(RUN, "H3_STUDY_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(RUN, "SEGMENT_SEED_BLOCKS",
                        ((202_626_000, 202_626_148),) + RUN.SEGMENT_SEED_BLOCKS[1:])
    with pytest.raises(RUN.H3StudyRunError, match="SPENT|may never be revived"):
        RUN.run_segment(segment=0, results_path="/dev/null/r",
                        trace_path="/dev/null/t", report_path="/dev/null/p")


def test_EVERY_SEGMENT_BLOCK_IS_REGISTERED_AND_ONLY_SEGMENT_0_IS_SPENT(monkeypatch):
    """🔴 INVERTED AFTER SEGMENT 0 RAN (2026-09-18, 148/148, exit 0).

    Registration is a PLANNING question and still holds for all four. Whether a
    block is RUNNABLE is a LAUNCH question, and segment 0's answer changed: its
    148 seeds are exposed and retired, so it refuses. Segments 1-3 are untouched
    and still launch -- which is the whole point of the isolation repair, now
    demonstrated by a real consumption rather than by a monkeypatch.
    """
    assert len(RUN.SEGMENT_SEED_BLOCKS) == R.N_SEGMENTS == 4
    RUN.check_seed_registration()                 # all four still REGISTERED

    with pytest.raises(RUN.H3StudyRunError, match="is SPENT"):
        RUN.check_segment_seeds(0)
    for k in (1, 2, 3):
        assert RUN.check_segment_seeds(k)["n"] == R.GAMES_PER_SEGMENT

    monkeypatch.setattr(RUN, "SEGMENT_SEED_BLOCKS", ())
    with pytest.raises(RUN.H3StudyRunError, match="NO SEED BLOCKS"):
        RUN.check_seed_registration()


def test_SEGMENT_0_CANNOT_BE_RELAUNCHED():
    """One-shot. Its block is spent and its destination is occupied; either alone
    must stop a second launch."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    lo, hi = RUN.SEGMENT_SEED_BLOCKS[0]
    st = [REF.seed_status(s) for s in range(lo, hi)]
    assert all(x["exposed"] and x["retired"] for x in st)
    with pytest.raises(RUN.H3StudyRunError, match="is SPENT"):
        RUN.check_segment_seeds(0)
    import os
    from scripts.GPU.alphazero import h3_study_command as CMD
    assert any(os.path.lexists(p) for p in CMD.default_paths(0)), (
        "the destination holds the completed run's artifacts")
    with pytest.raises(RUN.H3StudyRunError, match="already exists"):
        RUN.check_output_paths(*CMD.default_paths(0))
