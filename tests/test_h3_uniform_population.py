"""AMENDMENT 3 — the UNIFORM-ONLY population, and what must stay true of it.

These are the invariants the amendment turns on. Each one is here because
removing a stratum is the kind of change that leaves residue: a constant nothing
reads, a field nothing validates, a path nothing walks but that still resolves.
"""
import ast
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
    (lambda d: d["openings"][0].update(stub=True), "STUB"),
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


def test_THE_BARRIER_IS_READ_BEFORE_ANYTHING_DURABLE_IS_TOUCHED():
    assert not os.path.lexists(GEN.OUT_DIR)
    with pytest.raises(GEN.H3GenerationError, match="NOT AUTHORIZED"):
        GEN.freeze_population()
    assert not os.path.lexists(GEN.OUT_DIR), (
        "a refusal left a directory behind; the barrier is read too late")


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
def test_THE_OFFICIAL_DESTINATION_IS_ABSENT_AND_OUTSIDE_EVERY_SPENT_DIRECTORY():
    """Repointed by Amendment 3. It named attempt 2's directory, which that run
    CONSUMED on 2026-09-16 -- so this test was correctly red until the
    destination moved."""
    assert not os.path.lexists(GEN.OUT_DIR)
    assert not os.path.lexists(GEN.DEFAULT_OUT)
    assert not os.path.lexists(GEN.DEFAULT_TRACE)
    for spent in CMD.SPENT_OUT_DIRS:
        assert GEN.OUT_DIR != spent
        assert not GEN.OUT_DIR.startswith(spent.rstrip("/") + "/")
    assert GEN.OUT_DIR not in CMD.SPENT_OUT_DIRS, (
        "a destination is marked spent only after a run consumes it")


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
def test_NOTHING_IS_PINNED_RESERVED_OR_OPEN():
    assert R.OPENING_SET_DIGEST is None
    assert RUN.STUDY_SEED_BLOCK is None
    assert RUN.H3_STUDY_EXECUTION_AUTHORIZED is False
    assert GEN.H3_POPULATION_FREEZE_AUTHORIZED is False
    with pytest.raises(R.H3StudyError, match="not FROZEN yet"):
        R.expected_opening_set_digest()
