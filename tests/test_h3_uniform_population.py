"""AMENDMENT 3 — the UNIFORM-ONLY population, and what must stay true of it.

These are the invariants the amendment turns on. Each one is here because
removing a stratum is the kind of change that leaves residue: a constant nothing
reads, a field nothing validates, a path nothing walks but that still resolves.
"""
import ast
import json
import os
import pathlib
import tempfile

import pytest

from scripts.GPU.alphazero import h3_study_analysis as A
from scripts.GPU.alphazero import h3_study_command as CMD
from scripts.GPU.alphazero import h3_study_generator as GEN
from scripts.GPU.alphazero import h3_study_rules as R
from scripts.GPU.alphazero import h3_study_runner as RUN

ALPHAZERO = pathlib.Path(RUN.__file__).resolve().parent

#: the modules an execution actually walks. The retired engine path must be
#: unreachable from every one of them.
ACTIVE = ("h3_study_rules", "h3_study_generator", "h3_study_analysis",
          "h3_study_runner", "h3_study_command", "h3_study_prerun_verification")
RETIRED = ("h3_coproduced_generator_retired", "h3_generation_command",
           "h3_generation_preflight")


# ═══════════════════════ one population ════════════════════════════════════
def test_THE_POPULATION_IS_296_UNIFORM_OPENINGS():
    ops = GEN.build_population()
    assert len(ops) == R.N_PAIRS == 296
    assert {o["stratum"] for o in ops} == {R.STRATUM_UNIFORM}
    assert len({o["digest"] for o in ops}) == 296, "distinct up to symmetry"
    assert R.check_opening_set(ops)["n"] == 296


def test_THE_SIZE_STILL_MEETS_THE_PRECISION_TARGET_AMENDMENT_3_DID_NOT_TOUCH():
    """Dropping a stratum must not disturb the arithmetic that fixed N.

    Hoeffding takes no variance and no effect size, so nothing about the
    POPULATION can enter the sample size. 296 was chosen to satisfy h <= 0.08
    and it still does; it divided by stratum AND segment, and losing the stratum
    constraint leaves the segment one untouched.
    """
    assert R.N_PAIRS == 296 and R.N_GAMES == 592
    assert R.half_width(R.N_PAIRS) <= R.PRECISION_TARGET
    assert round(R.half_width(R.N_PAIRS), 5) == 0.07894
    assert R.PAIRS_PER_SEGMENT * R.N_SEGMENTS == R.N_PAIRS
    assert R.GAMES_PER_SEGMENT * R.N_SEGMENTS == R.N_GAMES


def test_FOUR_SEGMENTS_OF_74_PAIRS_WITH_NO_COMPOSITION_TO_BALANCE():
    tasks = R.build_tasks(GEN.build_population())
    for k in range(R.N_SEGMENTS):
        seg = [t for t in tasks if t["segment"] == k]
        assert len(seg) == 148
        assert len({t["pair_id"] for t in seg}) == 74
        assert {t["stratum"] for t in seg} == {R.STRATUM_UNIFORM}


@pytest.mark.parametrize("gone", [
    "STRATUM_CO_PRODUCED", "PAIRS_PER_STRATUM", "STRATUM_PAIRS_PER_SEGMENT",
    "INCUMBENT_FIRST_PER_SEGMENT", "T1J_FIRST_PER_SEGMENT",
    "ORDER_INCUMBENT_FIRST", "ORDER_T1J_FIRST", "generation_config",
    "stub_opening_set"])
def test_NO_CO_PRODUCED_CONSTANT_SURVIVES_IN_THE_LIVE_RULES(gone):
    """A dead constant beside live ones is one the next reader cannot classify."""
    assert not hasattr(R, gone), f"{gone} is still in h3_study_rules"


def test_NO_ORDER_FIELD_ANYWHERE_IN_THE_SCHEDULE_OR_THE_OPENINGS():
    ops = GEN.build_population()
    assert not any("order" in o for o in ops)
    assert not any("order" in t for t in R.build_tasks(ops))
    assert "order" not in GEN.OPENING_KEYS


# ═══════════════════════ the seed allocation ═══════════════════════════════
def test_THE_RANGE_COVERS_EVERY_CANDIDATE_NOT_EVERY_OPENING():
    """🔴 THE DEFECT AMENDMENT 3 REPAIRED. The old range was 148 x 400 = 59,200
    and the study needs 296 x 400 = 118,400 -- short by exactly half."""
    lo, hi = R.generation_seed_range(R.GEN_SEED_UNIFORM)
    assert hi - lo == R.N_PAIRS * R.MAX_ATTEMPTS == 118_400
    assert (lo, hi) == (20_261_600_000, 20_261_718_400)


def test_EVERY_SEED_THE_GENERATOR_ACTUALLY_USES_LIES_INSIDE_THE_RANGE():
    """The range is a claim; this checks the generator honours it.

    🔑 THE PROGRAMME HAS BEEN BITTEN HERE. A previous generator seeded a fresh
    agent per ply as `seed + ply*7919` and overran its declared range by 47,514
    -- more than the gap to the next one. A declared range nothing checks is a
    comment.
    """
    lo, hi = R.generation_seed_range(R.GEN_SEED_UNIFORM)
    used = [o["seed"] for o in GEN.build_population()]
    assert min(used) >= lo and max(used) < hi
    assert len(set(used)) == len(used), "no seed is reused"


@pytest.mark.parametrize("retired_lo", [20_261_000_000, 20_261_200_000,
                                        20_261_400_000])
def test_EVERY_RETIRED_RANGE_IS_REFUSED(retired_lo):
    assert R.is_spent_generation_range(retired_lo) is True
    with pytest.raises(R.H3StudyError, match="SPENT"):
        R.attempt_seed(retired_lo, 0, 0)


def test_THE_LIVE_RANGE_IS_NOT_REFUSED_OR_THE_CHECK_PROVES_NOTHING():
    """NEGATIVE CONTROL for the three above: a check that refuses everything
    would pass them all and block the study too."""
    assert R.is_spent_generation_range(R.GEN_SEED_UNIFORM) is False
    assert R.attempt_seed(R.GEN_SEED_UNIFORM, 0, 0) == R.GEN_SEED_UNIFORM


def test_THE_LIVE_RANGE_CLEARS_EVERY_RETIRED_ONE_BY_THE_GAP_FLOOR():
    """The floor is the candidate's OWN size -- 118,400. Disjoint is not
    separated: v11 rejected two ranges that were disjoint but 40,800 apart."""
    lo, hi = R.generation_seed_range(R.GEN_SEED_UNIFORM)
    floor = R.N_PAIRS * R.MAX_ATTEMPTS
    for r_lo, r_hi, _ in R.RETIRED_GENERATION_RANGES:
        assert hi <= r_lo or r_hi <= lo, "overlap"
        assert (lo - r_hi if lo >= r_hi else r_lo - hi) >= floor


def test_EVERY_RETIRED_RANGE_CARRIES_ITS_REASON():
    for lo, hi, why in R.RETIRED_GENERATION_RANGES:
        assert isinstance(why, str) and len(why) > 40, (lo, hi)


# ═══════════════════════ exclusions ════════════════════════════════════════
def test_THE_EXCLUSION_SET_IS_28_POSITIONS_AND_THE_VOIDS_ADD_NONE():
    """🔑 THE CO-PRODUCED EXCLUSION IS VACUOUS AS TO POSITIONS, and the count
    says so. Both attempts accepted ZERO openings, so there is nothing of theirs
    to exclude; their residue is the spent RANGES, not a position set. Claiming
    30 would be claiming coverage the study does not have."""
    assert len(R.excluded_digests()) == 28


def test_NO_STUDY_OPENING_IS_A_PILOT_OR_H1_H2_POSITION():
    assert not ({o["digest"] for o in GEN.build_population()}
                & R.excluded_digests())


# ═══════════════════════ the artifact ══════════════════════════════════════
def test_THE_ARTIFACT_CARRIES_COMPLETE_ATTEMPT_PROVENANCE():
    doc = GEN.artifact_document(GEN.build_population())
    for o in doc["openings"]:
        assert type(o["attempts"]) is int and o["attempts"] >= 1
        assert type(o["seed"]) is int
        assert len(o["moves"]) == R.OPENING_PLIES
    assert doc["generator"]["engine_free"] is True
    assert doc["generator"]["max_attempts"] == R.MAX_ATTEMPTS


@pytest.mark.parametrize("break_it,match", [
    (lambda d: d.update(stratum="co_produced"), "stratum"),
    (lambda d: d.update(n=148), "296"),
    (lambda d: d.update(claim=d["claim"].replace("NOTHING", "little")), "claim"),
    (lambda d: d.update(claim=""), "claim"),
    (lambda d: d["openings"][0].update(order="incumbent_first"), "order"),
    (lambda d: d["openings"][0].update(attempts=0), "attempts"),
    # 🔑 the EXACT-SCHEMA check now subsumes the stub-specific one and refuses
    # EARLIER, on the unknown key. The stub branch survives in
    # `check_opening_set`, which does no key check -- see the test below.
    (lambda d: d["openings"][0].update(stub=True), "unknown keys"),
    (lambda d: d["openings"][0].update(stratum="co_produced"), "stratum"),
    (lambda d: d.update(opening_set_digest="0" * 64), "edited"),
])
def test_THE_ARTIFACT_VALIDATOR_REFUSES(break_it, match):
    import copy
    doc = copy.deepcopy(GEN.artifact_document(GEN.build_population()))
    break_it(doc)
    with pytest.raises(GEN.H3GenerationError, match=match):
        GEN.validate_artifact(doc)


def test_A_GOOD_ARTIFACT_PASSES_OR_THE_REFUSALS_PROVE_NOTHING():
    doc = GEN.artifact_document(GEN.build_population())
    assert GEN.validate_artifact(doc)["n"] == R.N_PAIRS


# ═══════════════════════ the freeze barrier ════════════════════════════════
def test_GENERATING_IS_FREE_AND_FREEZING_IS_NOT():
    """The distinction the amendment rests on: no gate on generation, a barrier
    on declaring one population THE population."""
    assert not hasattr(GEN, "H3_GENERATION_AUTHORIZED"), "no execution gate"
    assert GEN.H3_POPULATION_FREEZE_AUTHORIZED is False
    GEN.build_population()                         # ungated, no exception




def test_THE_BARRIER_IS_READ_FIRST_BY_AST():
    """Structural, not behavioural: the refusal must be the FIRST statement, so
    no future edit can slip an effect above it."""
    src = (ALPHAZERO / "h3_study_generator.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "freeze_population")
    body = [st for st in fn.body if not (isinstance(st, ast.Expr)
                                         and isinstance(st.value, ast.Constant))]
    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert first.value.func.id == "check_freeze_barrier"


def test_write_artifact_HAS_NO_DEFAULT_DESTINATION():
    """🔑 A DEFAULT HERE WOULD PUT EVERY TEST ONE MISSING ARGUMENT AWAY FROM
    FREEZING THE POPULATION. Only `freeze_population` knows the official path."""
    src = (ALPHAZERO / "h3_study_generator.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "write_artifact")
    for name, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if name.arg in ("out_path", "trace_path"):
            assert default is None, f"{name.arg} has a default"


def test_WRITING_UNDER_A_TEMPORARY_PATH_WORKS_AND_IS_CREATE_ONLY():
    with tempfile.TemporaryDirectory() as d:
        ops = GEN.build_population()
        doc = GEN.write_artifact(openings=ops, out_path=f"{d}/a.json",
                                 trace_path=f"{d}/t.jsonl")
        assert doc["n"] == R.N_PAIRS
        with pytest.raises(GEN.H3GenerationError, match="CREATE-ONLY"):
            GEN.write_artifact(openings=ops, out_path=f"{d}/a.json",
                               trace_path=f"{d}/t2.jsonl")


def test_A_DANGLING_SYMLINK_IS_REFUSED_NOT_WRITTEN_THROUGH():
    with tempfile.TemporaryDirectory() as d:
        os.symlink(f"{d}/nowhere", f"{d}/a.json")
        assert not os.path.exists(f"{d}/a.json") and os.path.lexists(f"{d}/a.json")
        with pytest.raises(GEN.H3GenerationError, match="CREATE-ONLY"):
            GEN.write_artifact(openings=GEN.build_population(),
                               out_path=f"{d}/a.json", trace_path=f"{d}/t.jsonl")
        assert not os.path.exists(f"{d}/nowhere"), "written through the link"


def test_A_TERMINAL_RECORD_IS_WRITTEN_ON_THE_REFUSAL_PATH_TOO():
    import json
    with tempfile.TemporaryDirectory() as d:
        bad = [dict(o) for o in GEN.build_population()]
        bad[0]["attempts"] = 0                      # fails validation
        with pytest.raises(GEN.H3GenerationError):
            GEN.write_artifact(openings=bad, out_path=f"{d}/a.json",
                               trace_path=f"{d}/t.jsonl")
        end = [json.loads(l) for l in open(f"{d}/t.jsonl")][-1]
        assert end["event"] == "generation_end" and end["verdict"] == "VOID"
        assert end["failure"] and not os.path.lexists(f"{d}/a.json")


# ═══════════════════════ the official destination ══════════════════════════


def test_BOTH_VOIDED_DESTINATIONS_ARE_SPENT_AND_STAY_THAT_WAY():
    for d in ("docs/superpowers/evidence/2026-09-15-t1j-h3-study-openings",
              "docs/superpowers/evidence/2026-09-16-t1j-h3-study-openings-attempt2"):
        assert d in CMD.SPENT_OUT_DIRS
        assert os.path.lexists(d), "the VOID evidence must still be there"


def test_THE_RUNNER_READS_THE_DESTINATION_RATHER_THAN_REPEATING_IT():
    """🔴 IT WAS A LITERAL AND IT WENT STALE. `OPENING_SET_PATH` named attempt
    1's directory; when the destination moved, the runner silently pointed at a
    SPENT directory and every test still passed, because nothing compared them."""
    assert RUN.OPENING_SET_PATH == GEN.DEFAULT_OUT
    src = (ALPHAZERO / "h3_study_runner.py").read_text(encoding="utf-8")
    assert "OPENING_SET_PATH = GEN.DEFAULT_OUT" in src


# ═══════════════════════ the retired path is unreachable ═══════════════════
def _imports_of(module_name):
    src = (ALPHAZERO / f"{module_name}.py").read_text(encoding="utf-8")
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.add(a.name)
            if n.module:
                out.add(n.module.rsplit(".", 1)[-1])
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.rsplit(".", 1)[-1])
    return out


@pytest.mark.parametrize("active", ACTIVE)
def test_NO_ACTIVE_MODULE_REACHES_THE_RETIRED_ENGINE_PATH(active):
    """🔴 THE RETIREMENT MUST BIND. The engine generator, its JVM supervisor and
    its preflight are preserved for history; a live module importing one would
    make "retired" a label rather than a fact."""
    reached = _imports_of(active) & set(RETIRED)
    assert not reached, f"{active} imports {reached}"


def test_THE_RETIRED_MODULES_DO_EXIST_OR_THE_CHECK_IS_VACUOUS():
    """NEGATIVE CONTROL: if the files were simply deleted, the parametrized test
    above would pass for the wrong reason."""
    for m in RETIRED:
        assert (ALPHAZERO / f"{m}.py").exists(), m


def test_THE_RETIRED_GENERATOR_REFUSES_UNCONDITIONALLY():
    from scripts.GPU.alphazero import h3_coproduced_generator_retired as OLD
    with pytest.raises(OLD.H3GenerationError, match="RETIRED"):
        OLD.check_gate()


# ═══════════════════════ the narrowed claim ════════════════════════════════
def test_THE_CLAIM_NAMES_THE_POPULATION_AND_DISCLAIMS_PLAY():
    c = GEN.CLAIM
    assert "UNIFORMLY AT RANDOM" in c
    assert "NOTHING" in c and "realistic play" in c
    assert "not a position anyone plays" in c


def test_THE_ANALYSIS_READS_THE_CLAIM_AND_NEVER_RETYPES_IT():
    assert A.CLAIM is GEN.CLAIM or A.CLAIM == GEN.CLAIM
    src = (ALPHAZERO / "h3_study_analysis.py").read_text(encoding="utf-8")
    assert "uniformly random" not in src.lower().split("def _claim")[0], (
        "a second copy of the claim; the copy is what drifts")


@pytest.mark.parametrize("report,match", [
    ({}, "no claim"),
    ({"population": {}}, "no claim"),
    ({"population": {"claim": ""}}, "altered"),
    ({"population": {"claim": "T1j is stronger."}}, "altered"),
])
def test_A_REPORT_WITHOUT_THE_VERBATIM_CLAIM_IS_REFUSED(report, match):
    with pytest.raises(A.H3StudyAnalysisError, match=match):
        A.check_report_claim(report)


def test_THE_VERBATIM_CLAIM_PASSES_OR_THE_REFUSALS_PROVE_NOTHING():
    assert A.check_report_claim({"population": {"claim": GEN.CLAIM}}) == GEN.CLAIM


# ═══════════════════════ analysis refuses foreign records ══════════════════
def _rec(**kw):
    base = {"task_id": "t", "seed": 1, "pair_id": 0, "stratum": R.STRATUM_UNIFORM,
            "incumbent_colour": "red", "opening_digest": "a" * 64,
            "transcript_digest": "b" * 64, "terminal_reason": "win",
            "winner": "red", "plies": 30, "elapsed_s": 1.0}
    base.update(kw)
    return base


@pytest.mark.parametrize("bad,match", [
    ({"stratum": "co_produced"}, "uniform"),
    ({"stratum": None}, "uniform"),
    ({"order": "incumbent_first"}, "order"),
])
def test_THE_ANALYSIS_REFUSES_A_RECORD_FROM_ANOTHER_POPULATION(bad, match):
    with pytest.raises(A.H3StudyAnalysisError, match=match):
        A._validate([_rec(**bad)])


def test_A_UNIFORM_RECORD_IS_ACCEPTED_OR_THE_REFUSALS_PROVE_NOTHING():
    A._validate([_rec()])


def test_THE_REPORT_HAS_NO_BY_STRATUM_TABLE():
    """One population means a one-row table, which invites a comparison there is
    nothing to compare.

    🔑 CHECKED IN THE CODE, NOT THE PROSE. The first version grepped the file and
    flagged the COMMENT explaining the absence -- a guard that cannot tell code
    from an explanation of code. Narrowing a guard until it matches nothing is
    how a guard dies, so it parses instead, and the control below proves it
    still bites.
    """
    assert "by_stratum" not in _string_constants("h3_study_analysis")


def _string_constants(module_name):
    """Every string literal in the module's CODE. Comments are not literals, so
    an explanation of a removed field cannot trip a check on the field."""
    src = (ALPHAZERO / f"{module_name}.py").read_text(encoding="utf-8")
    return {n.value for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_THE_BY_STRATUM_CHECK_IS_NOT_VACUOUS():
    """NEGATIVE CONTROL: it must still see a real key, and must still ignore a
    comment that merely names one."""
    import textwrap
    code = textwrap.dedent('''
        # by_stratum was removed
        REPORT = {"by_stratum": {}}
    ''')
    lits = {n.value for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "by_stratum" in lits, "the check cannot see a real key"
    comment_only = {n.value for n in ast.walk(ast.parse("# by_stratum gone\nX = 1\n"))
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "by_stratum" not in comment_only, "the check trips on a comment"


# ═══════════════════════ nothing is pinned or reserved yet ═════════════════


# ═══════ gaps the injected-defect harness found in THESE tests ═════════════
# 🔴 Four controls were NOT CAUGHT on the first harness pass. Each injected a
# real defect that the tests above could not see, because each test exercised a
# path that routed around the guard it was meant to bind. The tests below close
# those four, and the reason is recorded next to each.

def test_THE_GENERATORS_DEFAULT_POPULATION_SIZE_IS_296():
    """NOT CAUGHT: `n: int = N_PAIRS` -> `n: int = 148`.

    Every test above called `build_population()`, which passes `n` EXPLICITLY, so
    the DEFAULT was never exercised and could be changed freely. A default nobody
    calls is a switch-off waiting to happen.
    """
    import inspect
    sig = inspect.signature(R.generate_uniform_openings)
    assert sig.parameters["n"].default == R.N_PAIRS == 296
    assert sig.parameters["seed"].default == R.GEN_SEED_UNIFORM
    assert len(R.generate_uniform_openings()) == 296      # the default, called


def test_assemble_opening_set_REFUSES_A_SHORT_POPULATION():
    """NOT CAUGHT: `if len(uniform) != N_PAIRS:` -> `if False:`.

    Nothing above ever handed it a short set, so removing the guard changed no
    observable behaviour. A guard no test feeds is a guard no test binds.
    """
    ops = GEN.build_population()
    with pytest.raises(R.H3StudyError, match="expected 296"):
        R.assemble_opening_set(ops[:148])
    with pytest.raises(R.H3StudyError, match="expected 296"):
        R.assemble_opening_set(ops + ops[:1])
    assert len(R.assemble_opening_set(ops)) == 296        # positive control


def test_check_opening_set_REFUSES_A_STALE_ORDER_FIELD():
    """NOT CAUGHT: the control injected into `check_opening_set`, but the test it
    named exercised `validate_artifact` -- a different function with its own copy
    of the rule. Two guards, one test, and the control found the gap."""
    ops = [dict(o) for o in GEN.build_population()]
    ops[7]["order"] = "incumbent_first"
    with pytest.raises(R.H3StudyError, match="order"):
        R.check_opening_set(ops)


def test_THE_FREEZE_BARRIER_REFUSES_A_TRUTHY_NON_TRUE_VALUE(monkeypatch):
    """NOT CAUGHT: `is not True` -> `not`.

    The tests above only ever saw `False`, which is falsy under both spellings.
    The difference appears for a truthy non-True value -- `1`, `"yes"`, a stray
    object -- and `is not True` is the spelling that refuses them.
    """
    for truthy in (1, "True", "yes", [1], object()):
        monkeypatch.setattr(GEN, "H3_POPULATION_FREEZE_AUTHORIZED", truthy)
        with pytest.raises(GEN.H3GenerationError, match="NOT AUTHORIZED"):
            GEN.check_freeze_barrier()
    monkeypatch.setattr(GEN, "H3_POPULATION_FREEZE_AUTHORIZED", True)
    GEN.check_freeze_barrier()                             # positive control


def test_check_opening_set_STILL_REFUSES_A_STUB():
    """The stub guard moved, it did not die.

    `validate_artifact` refuses a `stub` key on the EXACT-SCHEMA rule, before it
    can reach a stub-specific branch -- so that branch would be dead there and
    was removed. `check_opening_set` does no key check, so its stub guard is the
    reachable one and is tested here rather than assumed.
    """
    ops = [dict(o) for o in GEN.build_population()]
    ops[3]["stub"] = True
    with pytest.raises(R.H3StudyError, match="STUB"):
        R.check_opening_set(ops)


# ═══════════ P1 REPAIRS (2026-09-17 review) ════════════════════════════════
from scripts.GPU.alphazero import h3_freeze_command as FCMD    # noqa: E402


# ── P1-1: the barrier is ONE-SHOT ──────────────────────────────────────────
def test_freeze_population_ACCEPTS_NO_PATHS():
    """🔴 IT USED TO TAKE `out_path` AND `trace_path`. With the barrier open a
    caller could write any number of different 'official' populations to any
    number of destinations. An entry that takes a destination is an entry whose
    authorization does not name what it authorizes."""
    import inspect
    sig = inspect.signature(GEN.freeze_population)
    assert list(sig.parameters) == [], sig


def test_THE_FREEZE_COMMAND_RESTORES_THE_BARRIER_ON_EVERY_OUTCOME(monkeypatch,
                                                                  tmp_path):
    """The `finally` must run after success, refusal, timeout and interrupt."""
    src = tmp_path / "gen.py"
    calls = []

    def outcome(exc):
        src.write_text("H3_POPULATION_FREEZE_AUTHORIZED = True\n", encoding="utf-8")
        monkeypatch.setattr(FCMD, "GENERATOR_SOURCE", str(src))
        monkeypatch.setattr(FCMD, "barrier_is_open", lambda: True)

        def boom():
            calls.append(exc)
            if exc is None:
                return {"n": 296, "opening_set_digest": "d" * 64}
            raise exc
        monkeypatch.setattr(GEN, "freeze_population", boom)
        return FCMD.main(["--run"])

    for exc, want in ((None, FCMD.EXIT_OK),
                      (GEN.H3GenerationDeadline("slow"), FCMD.EXIT_TIMEOUT),
                      (KeyboardInterrupt(), FCMD.EXIT_INTERRUPTED),
                      (GEN.H3GenerationError("no"), FCMD.EXIT_FAILED)):
        assert outcome(exc) == want, exc
        assert src.read_text(encoding="utf-8").strip() == \
            "H3_POPULATION_FREEZE_AUTHORIZED = False", f"not restored after {exc!r}"
    assert len(calls) == 4


def test_A_FAILED_RESTORATION_SUPERSEDES_EVEN_A_SUCCESSFUL_FREEZE(monkeypatch,
                                                                  tmp_path):
    """🔴 A POPULATION WRITTEN WITH THE BARRIER LEFT OPEN IS NOT A COMPLETED
    FREEZE -- the next invocation would freeze again. So this outcome overrides
    the worker's own success, and gets its own exit code."""
    src = tmp_path / "gen.py"
    src.write_text("H3_POPULATION_FREEZE_AUTHORIZED = True\n", encoding="utf-8")
    monkeypatch.setattr(FCMD, "GENERATOR_SOURCE", str(src))
    monkeypatch.setattr(FCMD, "barrier_is_open", lambda: True)
    monkeypatch.setattr(FCMD, "restore_barrier", lambda *a, **k: False)
    monkeypatch.setattr(GEN, "freeze_population",
                        lambda: {"n": 296, "opening_set_digest": "d" * 64})
    assert FCMD.main(["--run"]) == FCMD.EXIT_BARRIER_NOT_RESTORED


def test_THE_RESTORATION_IS_VERIFIED_FROM_THE_FILE_NOT_FROM_MEMORY(tmp_path):
    """The imported module still holds the pre-rewrite value; reporting that
    would report what we hoped rather than what is on disk."""
    src = tmp_path / "gen.py"
    src.write_text("x = 1\nH3_POPULATION_FREEZE_AUTHORIZED = True\ny = 2\n",
                   encoding="utf-8")
    assert FCMD.barrier_readback(str(src)) == "True"
    assert FCMD.restore_barrier(str(src)) is True
    assert FCMD.barrier_readback(str(src)) == "False"
    assert "H3_POPULATION_FREEZE_AUTHORIZED = False" in src.read_text(encoding="utf-8")
    # already closed -> True, and still closed
    assert FCMD.restore_barrier(str(src)) is True
    # unreadable / absent -> False, never an optimistic True
    assert FCMD.restore_barrier(str(tmp_path / "nope.py")) is False
    assert FCMD.barrier_readback(str(tmp_path / "nope.py")) is None


def test_THE_COMMAND_NEVER_OPENS_ITS_OWN_BARRIER():
    """A command that could open its own barrier makes the barrier a formality."""
    src = pathlib.Path(FCMD.__file__).read_text(encoding="utf-8")
    assert "H3_POPULATION_FREEZE_AUTHORIZED = True" not in src.replace(
        '_BARRIER_OPEN = re.compile(r"^H3_POPULATION_FREEZE_AUTHORIZED = True$", re.M)', "")
    assert FCMD.main([]) == FCMD.EXIT_REFUSED            # --run is required
    assert FCMD.main(["--run"]) == FCMD.EXIT_NOT_AUTHORIZED   # and the barrier binds




# ── P1-2: generation happens INSIDE the deadline and the trace ─────────────
def test_THE_DEADLINE_IS_CHECKED_INSIDE_THE_CANDIDATE_LOOP():
    """🔴 THE REPAIRED DEFECT. `write_artifact(openings=build_population())`
    evaluated the whole walk BEFORE the trace opened and the clock started, so a
    hang inside generation left no terminal record and could not trip the guard.
    A zero deadline must now fire at opening 0, attempt 0 -- before any opening
    is accepted."""
    import json
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(GEN.H3GenerationDeadline, match="opening 0, attempt 0"):
            GEN.write_artifact(out_path=f"{d}/a.json", trace_path=f"{d}/t.jsonl",
                               build=GEN.build_population, deadline_s=0)
        end = [json.loads(l) for l in open(f"{d}/t.jsonl")][-1]
        assert end["event"] == "generation_end"
        assert end["verdict"] == "TIMEOUT", "a timeout must not be reported as VOID"
        assert end["accepted"] == 0
        assert not os.path.lexists(f"{d}/a.json"), "a partial population is not one"


def test_THE_TRACE_IS_OPEN_BEFORE_ANY_GENERATION_HAPPENS():
    """A failure DURING generation must still leave a terminal record."""
    import json
    with tempfile.TemporaryDirectory() as d:
        def explode(check_deadline=None):
            raise RuntimeError("boom during the walk")
        # 🔑 THE ORIGINAL EXCEPTION PROPAGATES, deliberately: wrapping a
        # RuntimeError from inside the walk in an H3GenerationError would lose
        # the traceback that says WHERE. What matters here is that the terminal
        # record exists anyway.
        with pytest.raises(RuntimeError, match="boom during the walk"):
            GEN.write_artifact(out_path=f"{d}/a.json", trace_path=f"{d}/t.jsonl",
                               build=explode)
        events = [json.loads(l) for l in open(f"{d}/t.jsonl")]
        assert [e["event"] for e in events] == ["generation_start", "generation_end"]
        assert events[-1]["verdict"] == "VOID" and events[-1]["failure"]


def test_build_population_THREADS_THE_HOOK_WITH_NO_LAMBDA_IN_BETWEEN():
    """🔑 `lambda cd: build_population()` -- taking the hook and dropping it -- is
    how the guard ends up checked only after the walk it was meant to bound. The
    hook is positional and first so `build=build_population` works directly."""
    import inspect
    params = list(inspect.signature(GEN.build_population).parameters)
    assert params[0] == "check_deadline"
    seen = []
    R.generate_uniform_openings(n=3, check_deadline=lambda i, a, acc:
                                seen.append((i, a)))
    assert seen == [(0, 0), (1, 0), (2, 0)], seen
    # and build_population passes it through rather than swallowing it
    got = []
    GEN.build_population(lambda i, a, acc: got.append(i))
    assert len(got) == R.N_PAIRS


def test_THE_CLOCK_IS_MONOTONIC():
    """`time.time()` steps backwards over an NTP correction, silently extending
    the window a runaway guard exists to bound."""
    src = (ALPHAZERO / "h3_study_generator.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "write_artifact")
    # 🔑 THE CALLS, not the file text. The docstring EXPLAINS why time.time() is
    # wrong, and a grep cannot tell an explanation from a use.
    calls = {f"{n.func.value.id}.{n.func.attr}" for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name)}
    assert "time.monotonic" in calls
    assert "time.time" not in calls


def test_THE_ARTIFACT_IS_FSYNCED_BEFORE_OK_IS_RECORDED():
    """`OK` in the trace asserts the artifact is ON DISK. Without the fsync the
    process could report OK and lose the file to a crash."""
    src = (ALPHAZERO / "h3_study_generator.py").read_text(encoding="utf-8")
    body = src[src.index("def write_artifact("):src.index("def freeze_population(")]
    fsync_at = body.index("os.fsync(fh.fileno())")
    assert fsync_at < body.index('verdict = "OK"')


def test_write_artifact_REFUSES_BOTH_OR_NEITHER_SOURCE():
    with tempfile.TemporaryDirectory() as d:
        for kw in ({}, {"build": GEN.build_population,
                        "openings": GEN.build_population()}):
            with pytest.raises(GEN.H3GenerationError, match="exactly one"):
                GEN.write_artifact(out_path=f"{d}/a.json",
                                   trace_path=f"{d}/t.jsonl", **kw)


# ── P1-3: the digest BINDS the moves ───────────────────────────────────────
def test_ALTERED_MOVES_WITH_AN_UNTOUCHED_DIGEST_ARE_REFUSED():
    """🔴 THE DEFECT, EXACTLY. Both checks trusted each row's DECLARED digest and
    neither replayed the moves, so an opening could be rewritten while its digest
    -- and `OPENING_SET_DIGEST` -- stayed valid, and the runner would play the
    altered position."""
    import copy
    doc = copy.deepcopy(GEN.artifact_document(GEN.build_population()))
    before = doc["opening_set_digest"]
    doc["openings"][5]["moves"][0] = [3, 4]          # digest untouched
    assert RULES_set_digest(doc) == before, "the set digest is deliberately unchanged"
    with pytest.raises(GEN.H3GenerationError, match="replaying its moves"):
        GEN.validate_artifact(doc)


def RULES_set_digest(doc):
    return R.opening_set_digest(doc["openings"])


@pytest.mark.parametrize("break_it,match", [
    (lambda d: d["openings"][4]["moves"].pop(), "moves must be a list"),
    (lambda d: d["openings"][4]["moves"].append([1, 1]), "moves must be a list"),
    (lambda d: d["openings"][4].__setitem__("moves", [[True, 1]] * 6), "not int"),
    (lambda d: d["openings"][4].__setitem__("moves", [[1.0, 1]] * 6), "not int"),
    (lambda d: d["openings"][4].__setitem__("moves", [[99, 1]] * 6), "off a"),
    (lambda d: d["openings"][4].__setitem__("index", 7), "study order is positional"),
    (lambda d: d["openings"][4].__setitem__("segment", 3), "frozen plan"),
    (lambda d: d["openings"][4].__setitem__("seed", 20_261_600_000), "disagree"),
    (lambda d: d["openings"][4].__setitem__("attempts", 2), "disagree"),
    (lambda d: d.__setitem__("extra", 1), "unknown keys"),
    (lambda d: d["openings"][4].__setitem__("note", "x"), "unknown keys"),
    (lambda d: d["generator"].__setitem__("bit_generator", "MT19937"), "disagrees"),
    (lambda d: d["generator"].__setitem__("gen_seed_base", 1), "disagrees"),
    (lambda d: d["generator"]["source_pins"].__setitem__(
        "h3_study_rules.py", "0" * 64), "disagrees"),
    (lambda d: d.__setitem__("generation_note", "engine-assisted"), "generation note"),
    (lambda d: d.__setitem__("generator", None), "no generator identity"),
])
def test_THE_BOUND_VALIDATOR_REFUSES(break_it, match):
    import copy
    doc = copy.deepcopy(GEN.artifact_document(GEN.build_population()))
    break_it(doc)
    doc["opening_set_digest"] = R.opening_set_digest(doc["openings"])   # re-hash!
    with pytest.raises(GEN.H3GenerationError, match=match):
        GEN.validate_artifact(doc)


def test_A_SWAPPED_PAIR_OF_OPENINGS_IS_REFUSED():
    """Reordering leaves every digest valid and changes the set digest only in
    order -- but index and segment then disagree with position."""
    import copy
    doc = copy.deepcopy(GEN.artifact_document(GEN.build_population()))
    doc["openings"][0], doc["openings"][1] = doc["openings"][1], doc["openings"][0]
    doc["opening_set_digest"] = R.opening_set_digest(doc["openings"])
    with pytest.raises(GEN.H3GenerationError, match="study order is positional"):
        GEN.validate_artifact(doc)


def test_verify_candidate_REDERIVES_EVERY_OPENING_FROM_ITS_PROVENANCE():
    """The binding, end to end: seed -> walk -> moves -> digest, for all 296."""
    for o in GEN.build_population():
        moves, digest = R.verify_candidate(R.GEN_SEED_UNIFORM, o["index"],
                                           o["attempts"])
        assert moves == o["moves"] and digest == o["digest"], o["index"]
        assert o["seed"] == R.attempt_seed(R.GEN_SEED_UNIFORM, o["index"],
                                           o["attempts"] - 1)


def test_THE_LOADER_VALIDATES_AND_DOES_NOT_REIMPLEMENT():
    """🔴 `load_opening_set` never called `validate_artifact` at all. It now
    delegates: a loader with its own copy of the rules checks the copy."""
    src = (ALPHAZERO / "h3_study_runner.py").read_text(encoding="utf-8")
    body = src[src.index("def load_opening_set("):src.index("def run_segment(")]
    assert "GEN.validate_artifact(doc)" in body
    assert "expected_opening_set_digest()" in body, "must also be THE pinned set"
    assert "co-produced" not in body, "the stale co-produced message must be gone"


def test_THE_LOADER_REFUSES_AN_ALTERED_ARTIFACT(tmp_path, monkeypatch):
    import json
    doc = GEN.artifact_document(GEN.build_population())
    monkeypatch.setattr(R, "OPENING_SET_DIGEST", doc["opening_set_digest"])
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(doc), encoding="utf-8")
    assert len(RUN.load_opening_set(str(good))) == R.N_PAIRS   # positive control

    doc["openings"][9]["moves"][0] = [5, 5]
    doc["opening_set_digest"] = R.opening_set_digest(doc["openings"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RUN.H3StudyRunError, match="FAILED validation"):
        RUN.load_opening_set(str(bad))


def test_THE_LOADER_REFUSES_A_SELF_CONSISTENT_BUT_UNPINNED_POPULATION(tmp_path,
                                                                      monkeypatch):
    """A population can be perfectly re-derivable and still not be THE one."""
    import json
    doc = GEN.artifact_document(GEN.build_population())
    monkeypatch.setattr(R, "OPENING_SET_DIGEST", "f" * 64)
    f = tmp_path / "a.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RUN.H3StudyRunError, match="different population"):
        RUN.load_opening_set(str(f))


# ── the generator identity now pins what reproduces the walk ───────────────
def test_THE_GENERATOR_IDENTITY_PINS_THE_TOOLCHAIN():
    """🔴 'NO ENGINE' IS NOT 'NO TOOLCHAIN'. The first version recorded the
    DESIGN -- seed base, attempt ceiling, filter names -- and nothing that would
    let anyone reproduce the walk."""
    i = GEN.generator_identity()
    assert i["bit_generator"] == "PCG64"
    assert i["python"].count(".") == 2 and i["numpy"]
    # 🔴 THE PROTOCOL, NOT THE RULES. `h3_study_rules.py` holds
    # OPENING_SET_DIGEST and must stay OUT of the pinned set, or the freeze
    # sequence invalidates its own artifact on the pin edit.
    assert set(i["source_pins"]) == {"h3_generation_protocol.py",
                                     "game/twixt_state.py", "d1_selection.py"}
    assert i["excluded_digest_set"] and len(i["excluded_digest_set"]) == 64
    assert all(len(v) == 64 for v in i["source_pins"].values())
    assert i["commit"] is None or len(i["commit"]) == 40


def test_THE_COMMIT_IS_RECORDED_BUT_NOT_ENFORCED():
    """A frozen population stays valid across later commits that do not touch the
    walk. `source_pins` refuses exactly when the walk changed, which is the
    honest version of the same check."""
    assert "commit" not in GEN.IDENTITY_MUST_MATCH
    assert "source_pins" in GEN.IDENTITY_MUST_MATCH
    assert "commit" in GEN.generator_identity()


def test_A_SUBSTITUTED_OPENING_IS_CAUGHT_ONLY_BY_THE_WALK():
    """🔴 THE CASE THE PRNG RE-DERIVATION EXISTS FOR, and the only one that
    reaches it.

    Put opening 7's moves AND its digest into opening 4's row, leaving index,
    segment, seed and attempts alone, then re-hash the set. Every earlier check
    passes: the moves replay to the digest recorded beside them, and the seed
    still matches its own attempt count. Only re-running the walk from that seed
    shows the row now holds a position the generator did not produce there.

    The control harness found this gap: injecting `if False` over the walk check
    changed nothing, because no test reached it.
    """
    import copy
    doc = copy.deepcopy(GEN.artifact_document(GEN.build_population()))
    donor = copy.deepcopy(doc["openings"][7])
    doc["openings"][4]["moves"] = donor["moves"]
    doc["openings"][4]["digest"] = donor["digest"]
    doc["opening_set_digest"] = R.opening_set_digest(doc["openings"])

    # every earlier check is satisfied -- shown, not assumed
    st = R._replay([tuple(m) for m in doc["openings"][4]["moves"]])
    from scripts.GPU.alphazero import d1_selection as SEL
    assert SEL.canonical_digest(st) == doc["openings"][4]["digest"]
    assert doc["openings"][4]["seed"] == R.attempt_seed(
        R.GEN_SEED_UNIFORM, 4, doc["openings"][4]["attempts"] - 1)

    with pytest.raises(GEN.H3GenerationError,
                       match="does NOT produce the recorded moves"):
        GEN.validate_artifact(doc)


def test_A_LYING_RESTORATION_IS_CAUGHT_BY_THE_READBACK(monkeypatch, tmp_path):
    """🔴 `restore_barrier` RETURNING TRUE IS NOT EVIDENCE THE FILE CLOSED.

    The other restoration test forces `restore_barrier` False, so `not restored`
    fires first and the READBACK is never consulted -- injecting a constant
    "False" readback changed nothing. Here the rewrite claims success while the
    file still says True, which is the case the readback exists for.
    """
    src = tmp_path / "gen.py"
    src.write_text("H3_POPULATION_FREEZE_AUTHORIZED = True\n", encoding="utf-8")
    monkeypatch.setattr(FCMD, "GENERATOR_SOURCE", str(src))
    monkeypatch.setattr(FCMD, "barrier_is_open", lambda: True)
    monkeypatch.setattr(FCMD, "restore_barrier", lambda *a, **k: True)   # lies
    monkeypatch.setattr(GEN, "freeze_population",
                        lambda: {"n": 296, "opening_set_digest": "d" * 64})
    assert FCMD.main(["--run"]) == FCMD.EXIT_BARRIER_NOT_RESTORED
    assert FCMD.barrier_readback(str(src)) == "True", "the file really is open"


# ═══════ THE FREEZE/PIN SEQUENCE MUST NOT INVALIDATE ITSELF ════════════════
# 🔴 THE BUG THIS SECTION EXISTS FOR. `generator_identity` pinned the whole of
# `h3_study_rules.py` -- which also holds `OPENING_SET_DIGEST`. Freezing recorded
# that file's hash with the constant still None; the next required step edits the
# constant; the hash moves; `load_opening_set` then refuses the population it had
# just frozen, for source drift caused by its own procedure. Nothing could ever
# have been played.

import hashlib                                                    # noqa: E402
from scripts.GPU.alphazero import h3_generation_protocol as PROTO  # noqa: E402

ALL_PINNED = ("h3_generation_protocol.py", "game/twixt_state.py",
              "d1_selection.py")


def _frozen_doc():
    return GEN.artifact_document(GEN.build_population())




def test_NO_OUTPUT_CONSTANT_LIVES_INSIDE_A_PINNED_SOURCE():
    """The rule, checked structurally: a pinned input may not contain an output.

    🔑 THIS IS THE GENERAL FORM OF THE BUG. `OPENING_SET_DIGEST` was the instance
    that bit; the retired seed ranges and the destination paths are the same
    shape -- each moves after a run, and pinning any of them would make a routine
    retirement invalidate every frozen population.
    """
    import ast
    here = ALPHAZERO
    FORBIDDEN = {"OPENING_SET_DIGEST", "RETIRED_GENERATION_RANGES",
                 "SPENT_GENERATION_RANGES", "OUT_DIR", "DEFAULT_OUT",
                 "DEFAULT_TRACE", "SPENT_OUT_DIRS", "STUDY_SEED_BLOCK"}
    for rel in ALL_PINNED:
        tree = ast.parse((here / rel).read_text(encoding="utf-8"))
        names = {t.id for n in tree.body if isinstance(n, ast.Assign)
                 for t in n.targets if isinstance(t, ast.Name)}
        names |= {n.target.id for n in tree.body if isinstance(n, ast.AnnAssign)
                  and isinstance(n.target, ast.Name)}
        clash = names & FORBIDDEN
        assert not clash, f"{rel} holds output constant(s) {clash}"


def test_THE_PINNED_SET_IS_NOT_VACUOUS():
    """NEGATIVE CONTROL: the check above would pass trivially over an empty set,
    and the pins must actually cover the three walk-determining sources."""
    pins = GEN.generator_identity()["source_pins"]
    assert set(pins) == set(ALL_PINNED) and len(pins) == 3
    for rel in ALL_PINNED:
        assert (ALPHAZERO / rel).exists(), rel
        assert len(pins[rel]) == 64


@pytest.mark.parametrize("rel,find,repl,why", [
    ("h3_generation_protocol.py", "rng = np.random.Generator(np.random.PCG64(seed))",
     "rng = np.random.Generator(np.random.PCG64(seed + 1))", "the walk"),
    ("h3_generation_protocol.py", "return base + index * MAX_ATTEMPTS + attempt",
     "return base + index * MAX_ATTEMPTS + attempt + 1", "the seed allocation"),
    ("h3_generation_protocol.py", "    if st.is_terminal():\n        return False",
     "    if False:\n        return False", "a structural filter"),
    ("h3_generation_protocol.py", "OPENING_PLIES = 6", "OPENING_PLIES = 4",
     "the ply count"),
    ("game/twixt_state.py", "                    moves.append((row, col))",
     "                    moves.insert(0, (row, col))", "the legal-move order"),
])
def test_CHANGING_A_WALK_DETERMINING_SOURCE_STILL_INVALIDATES(rel, find, repl, why):
    """🔴 NARROWING THE PIN MUST NOT HAVE BLUNTED IT.

    Each edit below would change the population. The pin must move for every one
    of them, or the repair traded a self-invalidating artifact for one that
    validates under code that produces something else.

    The files are read and hashed, never written: this asks whether the PIN would
    notice, not whether the suite survives a mutated tree.
    """
    src = (ALPHAZERO / rel).read_text(encoding="utf-8")
    assert src.count(find) >= 1, f"anchor gone from {rel}: {find!r}"
    mutated = src.replace(find, repl, 1)
    assert mutated != src
    before = hashlib.sha256(src.encode()).hexdigest()
    after = hashlib.sha256(mutated.encode()).hexdigest()
    assert before != after, why
    assert GEN.generator_identity()["source_pins"][rel] == before, (
        f"{rel} is not pinned at its current content")


def test_CHANGING_THE_CANONICAL_DIGEST_STILL_INVALIDATES():
    """`d1_selection.canonical_digest` decides which candidates count as
    distinct, so the file is pinned whole."""
    rel = "d1_selection.py"
    src = (ALPHAZERO / rel).read_text(encoding="utf-8")
    assert "def canonical_digest" in src
    assert GEN.generator_identity()["source_pins"][rel] == \
        hashlib.sha256(src.encode()).hexdigest()


def test_CHANGING_THE_EXCLUSION_SET_STILL_INVALIDATES(monkeypatch):
    """The exclusions are pinned BY VALUE, so a changed set refuses even though
    no pinned source moved."""
    doc = _frozen_doc()
    monkeypatch.setattr(R, "excluded_digest_set_pin", lambda: "0" * 64)
    with pytest.raises(GEN.H3GenerationError, match="excluded_digest_set"):
        GEN.validate_artifact(doc)


def test_A_REFACTOR_THAT_LEAVES_THE_EXCLUSIONS_IDENTICAL_DOES_NOT_INVALIDATE():
    """…and the converse, which is why the set is pinned by value rather than by
    pinning `h3_pilot_rules.py`: the population depends on WHICH digests are
    excluded, not on the code that computed them."""
    a = R.excluded_digest_set_pin()
    assert a == hashlib.sha256(
        "\n".join(sorted(R.excluded_digests())).encode()).hexdigest()
    assert len(R.excluded_digests()) == 28
    assert R.excluded_digest_set_pin() == a          # stable across calls


# ── the enforced / recorded-only split is a STATED policy ──────────────────
def test_EVERY_IDENTITY_FIELD_IS_CLASSIFIED():
    """🔑 NO FIELD MAY BE UNCLASSIFIED. `python` and `numpy` were recorded and
    silently unenforced while the contract said identity was checked exactly and
    only `commit` was informational. A field in neither set is a claim nobody
    made."""
    fields = set(GEN.generator_identity())
    enforced = set(GEN.IDENTITY_MUST_MATCH)
    recorded = set(GEN.IDENTITY_RECORDED_ONLY)
    assert enforced & recorded == set(), "a field cannot be both"
    assert enforced | recorded == fields, (
        f"unclassified: {fields - enforced - recorded}")


def test_NUMPY_IS_ENFORCED_BECAUSE_GENERATOR_STREAMS_ARE_NOT_GUARANTEED():
    """NumPy guarantees stream compatibility for legacy RandomState and
    explicitly NOT for Generator/PCG64, so a version bump may change every
    opening. Refusing is fail-closed."""
    assert "numpy" in GEN.IDENTITY_MUST_MATCH
    doc = _frozen_doc()
    doc["generator"]["numpy"] = "0.0.1"
    with pytest.raises(GEN.H3GenerationError, match="numpy"):
        GEN.validate_artifact(doc)


def test_PYTHON_AND_COMMIT_ARE_RECORDED_ONLY_AND_DO_NOT_REFUSE():
    """Stated policy, tested: neither changes what a seed produces."""
    assert set(GEN.IDENTITY_RECORDED_ONLY) == {"python", "commit"}
    doc = _frozen_doc()
    doc["generator"]["python"] = "1.2.3"
    doc["generator"]["commit"] = "0" * 40
    assert GEN.validate_artifact(doc)["n"] == R.N_PAIRS
    assert doc["generator"]["python"] and doc["generator"]["commit"]   # still recorded


def test_attempt_seed_DELEGATES_THE_ARITHMETIC_TO_THE_PINNED_PROTOCOL():
    """🔴 THE ALLOCATION MUST LIVE IN THE PINNED MODULE.

    A copy of the formula in `h3_study_rules` would compute the same seeds today
    and be unpinned tomorrow -- so no behavioural test can tell the two apart,
    and the harness reported exactly that (injecting an identical inline formula
    was NOT CAUGHT). The check is therefore structural: the guard is here, the
    arithmetic is there.
    """
    import ast
    src = (ALPHAZERO / "h3_study_rules.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "attempt_seed")
    calls = {f"{n.func.value.id}.{n.func.attr}" for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name)}
    assert "PROTO.seed_for" in calls, "the arithmetic is not delegated"
    # …and no arithmetic of its own: no BinOp anywhere in the body
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.BinOp)], (
        "attempt_seed computes a seed itself; the allocation must be pinned")
    # the values still agree with the protocol's
    for i, a in ((0, 0), (7, 3), (R.N_PAIRS - 1, R.MAX_ATTEMPTS - 1)):
        assert R.attempt_seed(R.GEN_SEED_UNIFORM, i, a) == \
            PROTO.seed_for(R.GEN_SEED_UNIFORM, i, a)


@pytest.mark.parametrize("retired_lo", [20_261_000_000, 20_261_200_000,
                                        20_261_400_000])
def test_GENERATION_ITSELF_REFUSES_A_RETIRED_BASE(retired_lo):
    """🔴 THE GUARD MUST BIND WHERE GENERATION HAPPENS, not only where a caller
    asks for one seed.

    The existing test called `attempt_seed` directly, so removing the guard the
    LOOP injects was invisible -- NOT CAUGHT. This drives the loop.
    """
    with pytest.raises(R.H3StudyError, match="SPENT"):
        R.generate_uniform_openings(seed=retired_lo, n=1)


def test_GENERATION_ACCEPTS_THE_LIVE_BASE_OR_THE_REFUSALS_PROVE_NOTHING():
    assert len(R.generate_uniform_openings(seed=R.GEN_SEED_UNIFORM, n=2)) == 2


# ═══════ AFTER THE FREEZE AND THE PIN (2026-09-17 / 2026-09-18) ════════════
# 🔴 FIVE TESTS HERE ASSERTED THE PRE-FREEZE STATE: destination absent, digest
# unset, a refusal creating no directory. Those preconditions are now legitimately
# gone -- the population IS frozen and IS pinned. Each is replaced by the invariant
# it actually protected, re-expressed for the state that now holds. NONE of them
# is weakened: what "nothing was touched" means simply changed from "the directory
# does not exist" to "the frozen artifact is byte-identical".

def _artifact_fingerprint():
    """Every byte of the frozen destination, so "unchanged" is checkable."""
    d = pathlib.Path(GEN.OUT_DIR)
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(d.iterdir())} if d.exists() else {}


def test_THE_POPULATION_IS_FROZEN_AND_PINNED():
    """The state this study now stands on, asserted once and plainly."""
    assert R.OPENING_SET_DIGEST == (
        "35932b3fabd9c6463d615b0b3af380134dadd700e2ca1e882a0c863faf772e46")
    assert os.path.lexists(GEN.DEFAULT_OUT) and os.path.lexists(GEN.DEFAULT_TRACE)
    doc = json.loads(pathlib.Path(GEN.DEFAULT_OUT).read_text(encoding="utf-8"))
    assert doc["opening_set_digest"] == R.OPENING_SET_DIGEST, (
        "the pin must be the artifact's OWN digest, not a recomputed one")
    assert R.opening_set_digest(doc["openings"]) == R.OPENING_SET_DIGEST


def test_THE_PIN_IS_READ_FROM_THE_ARTIFACT_NOT_RECOMPUTED_AT_START_UP():
    """🔑 THE WHOLE POINT OF PINNING A DETERMINISTIC POPULATION. The set can be
    rebuilt at will, so the runner must play the FROZEN one -- and a rebuild that
    happens to match is not evidence, it is a coincidence the pin exists to stop
    mattering."""
    src = (ALPHAZERO / "h3_study_runner.py").read_text(encoding="utf-8")
    body = src[src.index("def load_opening_set("):src.index("def run_segment(")]
    assert "build_population" not in body and "generate_uniform" not in body


def test_RESERVED_BUT_NOTHING_IS_OPEN():
    """What registration did NOT authorize.

    The block is now reserved -- ACCOUNTED only -- and that changed nothing about
    permission. A reservation is not a draw and it is not a gate.
    """
    assert RUN.STUDY_SEED_BLOCK == (202_626_000, 202_626_592)
    assert RUN.H3_STUDY_EXECUTION_AUTHORIZED is False
    assert GEN.H3_POPULATION_FREEZE_AUTHORIZED is False
    from scripts.GPU.alphazero import gate_inventory as INV
    assert INV.gate_count() == 10 and INV.open_gates() == []


def test_THE_FROZEN_DESTINATION_IS_SPENT_AND_A_SECOND_FREEZE_REFUSES():
    """🔴 THE NEW INVARIANT THE FREEZE CREATED. The destination used to have to be
    ABSENT; now it holds the population, and what must hold is that nothing can
    write over it. Create-only is what makes the frozen artifact final."""
    before = _artifact_fingerprint()
    assert before, "the frozen artifact must be there"
    with pytest.raises(GEN.H3GenerationError, match="CREATE-ONLY"):
        GEN.write_artifact(openings=GEN.build_population(),
                           out_path=GEN.DEFAULT_OUT, trace_path=GEN.DEFAULT_TRACE)
    assert _artifact_fingerprint() == before, "the frozen artifact was modified"


def test_THE_BARRIER_STILL_BARS_AND_A_REFUSAL_CHANGES_NOTHING():
    """The barrier closed behind the one authorized freeze. A refusal must leave
    the frozen artifact byte-identical -- which is what "nothing durable was
    touched" means now that the destination legitimately exists."""
    before = _artifact_fingerprint()
    with pytest.raises(GEN.H3GenerationError, match="NOT AUTHORIZED"):
        GEN.freeze_population()
    assert _artifact_fingerprint() == before
    assert FCMD.barrier_readback(FCMD.GENERATOR_SOURCE) == "False"
    assert FCMD.main(["--run"]) == FCMD.EXIT_NOT_AUTHORIZED


def test_THE_PIN_EDIT_DID_NOT_INVALIDATE_THE_ARTIFACT():
    """🔴 THE BINDING TEST, NOW RUN AGAINST THE REAL SEQUENCE RATHER THAN A
    SIMULATED ONE.

    It used to apply the digest edit to the file text and re-hash. The edit has
    now actually been made, so the check is stronger: the artifact frozen BEFORE
    it still validates against the code AFTER it. Before the provenance repair
    this is exactly where the population would have been refused.
    """
    doc = json.loads(pathlib.Path(GEN.DEFAULT_OUT).read_text(encoding="utf-8"))
    assert "h3_study_rules.py" not in doc["generator"]["source_pins"]
    assert GEN.validate_artifact(doc)["n"] == R.N_PAIRS
    # and the file really did move
    live = hashlib.sha256(
        (ALPHAZERO / "h3_study_rules.py").read_bytes()).hexdigest()
    assert live not in doc["generator"]["source_pins"].values()
    # …while every pinned source is still exactly what the artifact recorded
    for rel, pin in doc["generator"]["source_pins"].items():
        assert hashlib.sha256((ALPHAZERO / rel).read_bytes()).hexdigest() == pin, rel


def test_THE_RUNNER_REFUSES_A_POPULATION_THAT_IS_NOT_THE_PINNED_ONE(tmp_path,
                                                                    monkeypatch):
    """Now that a pin exists, it must actually exclude. A self-consistent set
    generated from a different base is valid and is NOT this study's."""
    other = GEN.artifact_document(GEN.build_population())
    other["openings"] = list(reversed(other["openings"]))
    for i, o in enumerate(other["openings"]):
        o["index"], o["segment"] = i, R.segment_of(i)
    other["opening_set_digest"] = R.opening_set_digest(other["openings"])
    assert other["opening_set_digest"] != R.OPENING_SET_DIGEST
    f = tmp_path / "other.json"
    f.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(RUN.H3StudyRunError):
        RUN.load_opening_set(str(f))


def test_THE_COMMITTED_ARTIFACT_LOADS_THROUGH_THE_REAL_RUNNER():
    """End to end, on the real path, against the real file."""
    ops = RUN.load_opening_set(RUN.OPENING_SET_PATH)
    assert len(ops) == R.N_PAIRS == 296
    assert len({o["digest"] for o in ops}) == 296
    assert [o["index"] for o in ops] == list(range(296))
    for o in ops:
        mv, dg = R.verify_candidate(R.GEN_SEED_UNIFORM, o["index"], o["attempts"])
        assert [tuple(m) for m in mv] == [tuple(m) for m in o["moves"]]
        assert dg == o["digest"]
        assert o["seed"] == R.attempt_seed(R.GEN_SEED_UNIFORM, o["index"],
                                           o["attempts"] - 1)


# ═══════ THE MATCH SEED BLOCK (registered 2026-09-18) ══════════════════════
BLOCK = (202_626_000, 202_626_592)


def test_THE_BLOCK_IS_592_SEEDS_ONE_PER_GAME():
    assert RUN.STUDY_SEED_BLOCK == BLOCK
    lo, hi = BLOCK
    assert hi - lo == R.N_GAMES == 592 == R.N_PAIRS * 2


def test_EVERY_SEED_IS_ACCOUNTED_AND_NOT_EXPOSED_RETIRED_OR_TEST_ONLY():
    """🔑 A RESERVATION IS NOT A DRAW. Registering the block did not spend it."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    lo, hi = BLOCK
    st = [REF.seed_status(s) for s in range(lo, hi)]
    assert all(x["accounted"] for x in st)
    assert not any(x["exposed"] or x["retired"] or x["test_only"] for x in st)
    assert not any(s in REF.CONSUMED_SEEDS for s in range(lo, hi))
    RUN.check_seed_registration()


def test_THE_BLOCK_OVERLAPS_NOTHING_DIRECTLY_OR_THROUGH_ITS_STREAMS():
    """Collision proof v14, re-run here so the suite owns it too.

    🔴 THE FOUR GENERATION RANGES ARE IN NO REGISTRY and are added by hand. A
    registry-only enumeration would call an overlapping block clean.
    """
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero.twixtbot_g3_reference import SeededReferenceAgent
    masks = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                    *SeededReferenceAgent.READOUT_MASK.values()})
    assert len(masks) == 4
    lo, hi = BLOCK
    ours = set(range(lo, hi))

    prior = set()
    for t in (REF.ACCOUNTED_SEED_INTERVALS, REF.EXPOSED_SEED_INTERVALS,
              REF.RETIRED_SEED_INTERVALS, REF.TEST_ONLY_SEED_INTERVALS):
        prior |= {s for a, b in t for s in range(a, b)}
    prior -= ours                       # 🔑 excluded BY IDENTITY: it is registered
    prior |= set(REF.CONSUMED_SEEDS)
    gen = [(a, b) for a, b, _ in R.RETIRED_GENERATION_RANGES]
    gen.append(R.generation_seed_range(R.GEN_SEED_UNIFORM))
    assert len(gen) == 4, "all four generation ranges, none of them in a registry"
    for a, b in gen:
        prior |= set(range(a, b))

    assert not (ours & prior), "direct overlap"
    derive = lambda ss: {v for s in ss for v in (s, *(s ^ m for m in masks))}
    mine, theirs = derive(ours), derive(prior)
    assert not (mine & theirs), "derived-stream collision"
    assert len(mine) == len(ours) * 5, "derivations not injective"


def test_THE_GAP_FLOOR_IS_THE_BLOCKS_OWN_SIZE_AND_IT_CLEARS_IT():
    """592, and the ACTUAL nearest distance is checked, not just the floor, so a
    narrow choice could not hide behind a small threshold."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    lo, hi = BLOCK
    floor = hi - lo
    assert floor == 592
    intervals = [iv for t in (REF.ACCOUNTED_SEED_INTERVALS,
                              REF.EXPOSED_SEED_INTERVALS,
                              REF.RETIRED_SEED_INTERVALS,
                              REF.TEST_ONLY_SEED_INTERVALS) for iv in t]
    intervals += [(a, b) for a, b, _ in R.RETIRED_GENERATION_RANGES]
    intervals += [R.generation_seed_range(R.GEN_SEED_UNIFORM)]
    gaps = [a - hi if a >= hi else lo - b
            for a, b in intervals
            if (a, b) != BLOCK and not (a < hi and lo < b)]
    # 🔑 BY IDENTITY IN THE GAP CHECK TOO. Without excluding BLOCK itself the
    # registered block measures its own distance as zero and fails its own floor.
    assert gaps and min(gaps) >= floor, min(gaps)
    assert min(gaps) == 1960, "the nearest actual boundary"


def test_SEEDS_ARE_POSITIONAL_ACROSS_FOUR_CONTIGUOUS_QUARTERS():
    ops = RUN.load_opening_set(RUN.OPENING_SET_PATH)
    tasks = R.build_tasks(ops, seed_interval=RUN.STUDY_SEED_BLOCK)
    lo, _ = BLOCK
    assert [t["seed"] for t in tasks] == list(range(lo, lo + R.N_GAMES)), (
        "row i must bind to lo + i")
    for k in range(R.N_SEGMENTS):
        seg = [t for t in tasks if t["segment"] == k]
        ss = [t["seed"] for t in seg]
        assert len(seg) == R.GAMES_PER_SEGMENT == 148
        assert ss == list(range(lo + 148 * k, lo + 148 * (k + 1))), k
        assert len({t["pair_id"] for t in seg}) == R.PAIRS_PER_SEGMENT == 74
    # the quarters partition the block exactly, with no seed in two of them
    quarters = [{t["seed"] for t in tasks if t["segment"] == k} for k in range(4)]
    assert set().union(*quarters) == set(range(lo, lo + 592))
    assert sum(len(q) for q in quarters) == 592, "a seed is in two quarters"


def test_A_PAIRS_TWO_ARMS_SIT_ADJACENT_AND_SHARE_A_SEGMENT():
    """The colour-reversed pair is the unit of analysis; splitting one across
    segments would make a pair unscoreable if a segment VOIDed."""
    ops = RUN.load_opening_set(RUN.OPENING_SET_PATH)
    tasks = R.build_tasks(ops, seed_interval=RUN.STUDY_SEED_BLOCK)
    for i in range(0, len(tasks), 2):
        a, b = tasks[i], tasks[i + 1]
        assert a["pair_id"] == b["pair_id"] and a["segment"] == b["segment"]
        assert {a["incumbent_colour"], b["incumbent_colour"]} == {"red", "black"}
        assert b["seed"] == a["seed"] + 1


# ── the schedule pins ──────────────────────────────────────────────────────
def test_THE_SEEDED_SCHEDULE_REPRODUCES_ITS_PINS():
    ops = RUN.load_opening_set(RUN.OPENING_SET_PATH)
    tasks = R.build_tasks(ops, seed_interval=RUN.STUDY_SEED_BLOCK)
    assert RUN.check_schedule_digest(tasks) == RUN.SCHEDULE_DIGEST
    assert len(RUN.SEGMENT_DIGESTS) == R.N_SEGMENTS == 4
    assert len(set(RUN.SEGMENT_DIGESTS)) == 4, "each segment's pin is its OWN"
    for k in range(4):
        assert RUN.segment_digest(tasks, k) == RUN.SEGMENT_DIGESTS[k]


@pytest.mark.parametrize("tamper", [
    lambda t: t[0].update(seed=t[0]["seed"] + 1),
    lambda t: t[0].update(incumbent_colour="black"),
    lambda t: t[0].update(ply_cap=t[0]["ply_cap"] + 1),
    lambda t: t[0].update(opening_digest="0" * 64),
    lambda t: t[0].update(mcts_sims=401),
])
def test_THE_SCHEDULE_PIN_REFUSES_A_TAMPERED_SCHEDULE(tamper):
    ops = RUN.load_opening_set(RUN.OPENING_SET_PATH)
    tasks = [dict(t) for t in R.build_tasks(ops, seed_interval=RUN.STUDY_SEED_BLOCK)]
    tamper(tasks)
    with pytest.raises(RUN.H3StudyRunError, match="different experiment"):
        RUN.check_schedule_digest(tasks)


def test_THE_SEGMENT_PIN_IS_NOT_COMPUTED_FROM_THE_TASKS_IT_CHECKS():
    """🔴 THE DEFECT THE PIN REPLACED. `run_segment` passed
    `want_digest=segment_digest(tasks, segment)` -- the digest computed from the
    very tasks it then handed to the checker. One source, two sides, so the
    comparison agreed unconditionally: a check that existed and did not bind."""
    import ast
    src = (ALPHAZERO / "h3_study_runner.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "run_segment")
    body = ast.get_source_segment(src, fn) or ""
    assert "want_digest=SEGMENT_DIGESTS[segment]" in body
    assert "want_digest=segment_digest(" not in body
    assert "check_schedule_digest(tasks)" in body


def test_THE_UNSEEDED_SCHEDULE_HAS_A_DIFFERENT_DIGEST():
    """The pin is of the SEEDED plan. An unseeded one is a design identity and
    must not satisfy it."""
    ops = RUN.load_opening_set(RUN.OPENING_SET_PATH)
    with pytest.raises(RUN.H3StudyRunError):
        RUN.check_schedule_digest(R.build_tasks(ops))


def test_REGISTERING_THE_BLOCK_OPENED_NOTHING():
    """The whole point of ACCOUNTED-only. Ten gates shut, barrier shut."""
    from scripts.GPU.alphazero import gate_inventory as INV
    assert INV.gate_count() == 10 and INV.open_gates() == []
    assert RUN.H3_STUDY_EXECUTION_AUTHORIZED is False
    assert GEN.H3_POPULATION_FREEZE_AUTHORIZED is False
    for k in range(R.N_SEGMENTS):
        from scripts.GPU.alphazero import h3_study_command as CMD
        assert not any(os.path.lexists(p) for p in CMD.default_paths(k)), k
