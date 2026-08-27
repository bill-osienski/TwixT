"""L0 larger match: preregistration and design checks. NO EXECUTION.

No model, no jvm, no generator, no move, no game, and NO DRAW FROM ANY SEED.

POST-RUN. The canonical match executed once on 2026-08-27 and played all 64 games,
so [202613000, 202613064) is now EXPOSED and RETIRED, and its schedule may never
be executed again. What these tests check is that the READ path survived: the
plan still loads and verifies, and the durable results still reclassify to the
identical verdict. The last test asserts the seed registries are unchanged by this
file -- not that the block is unspent, which it no longer is.

The frozen plan's own "RESERVED, UNSPENT" and "NOT EXECUTED" strings are asserted
deliberately and must NOT be updated: they record the state at preregistration,
and their survival is the evidence that the plan was not rewritten after the
results were seen.
"""
import copy
import json
import math

import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import e4_screen_rules as SCREEN
from scripts.GPU.alphazero import e4_screen_runner as H
from scripts.GPU.alphazero import l0_match_plan as P
from scripts.GPU.alphazero import l0_match_rules as L

from scripts.GPU.alphazero.twixtbot_g3_schedule import CONSUMED_SEEDS


@pytest.fixture(scope="module")
def tasks():
    return P.build_tasks(P.load_source_plan())


# --- the design ------------------------------------------------------------

def test_the_design_is_8x2x4_and_the_arithmetic_is_not_asserted_by_hand():
    assert L.N_OPENINGS * L.N_ARMS * L.N_REPS == L.N_GAMES == 64


def test_every_opening_colour_cell_holds_exactly_four_repetitions(tasks):
    cells = {}
    for t in tasks:
        cells.setdefault((t["opening"], t["colour_arm"]), []).append(t["rep"])
    assert len(cells) == 16
    for key, reps in cells.items():
        assert sorted(reps) == [0, 1, 2, 3], key


def test_both_colour_arms_are_balanced_and_every_game_is_at_mdPly_6(tasks):
    arms = {}
    for t in tasks:
        arms[t["colour_arm"]] = arms.get(t["colour_arm"], 0) + 1
        assert t["t1j_mdPly"] == 6 and t["t1j_mdFixedPly"] is True
        assert t["endpoint"] == "strong"
    assert arms == {"t1j_red": 32, "t1j_black": 32}


def test_the_anchor_and_reference_colours_are_opposite_in_every_task(tasks):
    for t in tasks:
        assert t["anchor_colour"] == ("red" if t["colour_arm"] == "t1j_red" else "black")
        assert t["reference_colour"] == REF.reference_colour(t)
        assert t["reference_colour"] != t["anchor_colour"]


def test_parameters_are_READ_from_the_frozen_screen_plan_not_retyped(tasks):
    """'Unchanged' is only checkable if there is one source for the values."""
    src = P.load_source_plan()
    assert P.SOURCE_PLAN_SHA256 == H.CANONICAL_PLAN_SHA256
    plan = P.load_l0_plan()
    for key in ("openings", "opening_plies", "reference", "our_settings", "ply_cap",
                "scoring", "per_ply_binding", "durable_result_format"):
        assert plan[key] == src[key], key
    # abort_rules is deliberately NOT in that list: L0 carries its own, and
    # test_the_frozen_plan_carries_L0s_OWN_abort_rules_not_the_screens binds it.
    assert plan["our_settings"]["eval_config"]["mcts_sims"] == 400
    assert plan["ply_cap"] == L.PLY_CAP == 280


def test_the_builder_actually_READS_the_source_plan_it_is_given(tasks):
    """Not just that the frozen file matches -- that build_tasks consumes it.

    Comparing the frozen plan against the source proves the two files agree; it
    does NOT prove the builder read anything, and a builder with the openings
    retyped inline passed that check unchanged. Feeding a modified source and
    watching the output move is what binds it.
    """
    src = P.load_source_plan()
    renamed = json.loads(json.dumps(src))
    renamed["openings"] = {f"zz{i}_probe": v for i, v in
                           enumerate(src["openings"].values())}
    renamed["reference"] = dict(src["reference"], name="calib020_0001")
    built = P.build_tasks(renamed)
    assert [t["opening"] for t in built[:4]] == ["zz0_probe"] * 4
    assert {t["opening"] for t in built} == set(renamed["openings"])
    assert {t["opening"] for t in built} != {t["opening"] for t in tasks}
    # and the reference identity travels from the source too
    assert built[0]["reference_sha1"] == renamed["reference"]["sha1"]


def test_the_builder_refuses_a_source_with_the_wrong_number_of_openings():
    src = P.load_source_plan()
    short = json.loads(json.dumps(src))
    short["openings"] = dict(list(src["openings"].items())[:7])
    with pytest.raises(P.L0PlanError, match="openings"):
        P.build_tasks(short)


def test_a_tampered_source_plan_is_refused(tmp_path):
    bad = tmp_path / "src.json"
    bad.write_text(json.dumps({"openings": {}}))
    with pytest.raises(P.L0PlanError, match="sha256"):
        P.load_source_plan(str(bad))


# --- the seed block --------------------------------------------------------

def test_the_seed_block_remains_disjoint_from_every_OTHER_experiment():
    """The block now appears in its OWN exposed and retired entries, by design.

    What must still hold -- and is the property the pre-reservation proof
    established -- is that it collides with NO OTHER workstream's seeds. So its
    own two entries are excluded and everything else is checked.
    """
    def spread(iv, skip=()):
        return {s for lo, hi in iv if (lo, hi) not in skip for s in range(lo, hi)}

    block = set(range(*P.L0_SEED_BLOCK))
    own = (P.L0_SEED_BLOCK,)
    assert len(block) == 64
    assert not (block & spread(REF.EXPOSED_SEED_INTERVALS, own))
    assert not (block & spread(REF.RETIRED_SEED_INTERVALS, own))
    assert not (block & spread(REF.TEST_ONLY_SEED_INTERVALS))
    assert not (block & set(CONSUMED_SEEDS))
    src = P.load_source_plan()
    assert not (block & spread([tuple(x) for x in src["seed_block"]["prior_intervals"]]))
    assert not (block & spread([tuple(src["seed_block"]["interval"])]))
    # and it IS recorded in its own entries -- the run happened
    assert P.L0_SEED_BLOCK in REF.EXPOSED_SEED_INTERVALS
    assert P.L0_SEED_BLOCK in REF.RETIRED_SEED_INTERVALS


def test_the_seed_block_is_disjoint_from_every_derived_rng_stream():
    """A shared stream would correlate an L0 game with an already-played one.

    XOR only: deriving these integers constructs no generator and draws nothing.
    """
    def spread(iv):
        return {s for lo, hi in iv for s in range(lo, hi)}

    def streams(seed, anchor):
        t = {"seed": seed, "anchor_colour": anchor}
        d = REF.rng_stream_seeds(t)
        return (d["search_seed"], d["readout_seed"])

    block = set(range(*P.L0_SEED_BLOCK))
    others = (spread(REF.ACCOUNTED_SEED_INTERVALS) | spread(REF.EXPOSED_SEED_INTERVALS)
              | spread(REF.RETIRED_SEED_INTERVALS) | spread(REF.TEST_ONLY_SEED_INTERVALS)
              | set(CONSUMED_SEEDS)) - block
    mine = {v for s in block for a in ("red", "black") for v in streams(s, a)}
    theirs = {v for s in others for a in ("red", "black") for v in streams(s, a)}
    assert not (mine & theirs)


def test_all_64_seeds_are_now_EXPOSED_and_RETIRED():
    """POST-RUN STATE. The match ran once on 2026-08-27 and played all 64 games.

    L0 has no early stop, so every seed in the block drove real generators --
    unlike the E4 screen, where 8 of 32 were skipped and are retired without ever
    having been drawn. Here exposure and retirement cover the same range, and they
    still mean different things: drawn, and withdrawn.
    """
    for seed in range(*P.L0_SEED_BLOCK):
        st = REF.seed_status(seed)
        assert st == {"exposed": True, "retired": True, "accounted": True,
                      "test_only": False}, seed
    assert not REF.seed_is_accounted(P.L0_SEED_BLOCK[0] - 1)
    assert not REF.seed_is_accounted(P.L0_SEED_BLOCK[1])
    assert not REF.seed_is_exposed(P.L0_SEED_BLOCK[0] - 1)
    assert not REF.seed_is_exposed(P.L0_SEED_BLOCK[1])


def test_seeds_map_to_task_order_and_are_all_inside_the_block(tasks):
    lo, hi = P.L0_SEED_BLOCK
    assert [t["seed"] for t in tasks] == list(range(lo, hi))
    assert len({t["seed"] for t in tasks}) == 64


def test_the_spent_schedule_may_never_be_executed_again(tasks):
    """The one-shot completed, so execution is refused -- permanently."""
    with pytest.raises(REF.E4ReferenceError, match="EXPOSED"):
        REF.validate_schedule_executable(tasks)
    for t in tasks[:4]:
        with pytest.raises(REF.E4ReferenceError):
            REF.validate_task_executable(t)


def test_the_spent_plan_REMAINS_STRUCTURALLY_VALID(tasks):
    """Spent stops EXECUTION, not parsing, verification or classification."""
    plan = P.load_l0_plan()                      # still loads
    assert plan["n_tasks"] == 64
    assert P.l0_task_digest(plan["tasks"]) == P.L0_TASK_DIGEST
    summary = REF.validate_schedule_structure(plan["tasks"])
    assert summary["n_tasks"] == 64 and summary["search_readout_disjoint"]
    P.validate_l0_schedule(plan["tasks"])        # design rules still hold


def test_the_canonical_RESULTS_RECLASSIFY_IDENTICALLY_from_evidence():
    """The whole point of keeping the read path open: the verdict survives.

    Recomputed from the durable JSONL with the frozen reporter, after the seeds
    are gone -- score, BOTH intervals and every descriptive cell.
    """
    run = ("docs/superpowers/evidence/2026-08-27-t1j-l0-canonical-match/"
           "06_l0_match_results.jsonl")
    rows = [json.loads(l) for l in open(run)]
    results = [r for r in rows if r["record_type"] == "task_result"]
    recorded = [r for r in rows if r["record_type"] == "match_report"][-1]
    assert len(results) == 64

    again = L.match_report(results, P.load_l0_plan()["tasks"])
    assert again["reported"] is True
    assert again["overall"]["t1j_score"] == recorded["overall"]["t1j_score"] == 38.0
    assert again["overall"]["t1j_rate"] == pytest.approx(0.59375)
    assert again["overall"]["ci95_hoeffding"] == recorded["overall"]["ci95_hoeffding"]
    assert again["overall"]["ci95_wilson"] == recorded["overall"]["ci95_wilson"]
    assert again["by_opening"] == recorded["by_opening"]
    assert again["by_colour_arm"] == recorded["by_colour_arm"]
    assert again["overall"]["cap_terminations"] == 0

    # BOTH INTERVALS INCLUDE 0.5. A higher point estimate is not evidence that
    # T1j is stronger, and the claim discipline forbids saying otherwise.
    lo_h, hi_h = again["overall"]["ci95_hoeffding"]
    lo_w, hi_w = again["overall"]["ci95_wilson"]
    assert lo_h < 0.5 < hi_h, "the Hoeffding bound includes parity"
    assert lo_w < 0.5 < hi_w, "the nominal Wilson interval includes parity"


def test_no_witness_may_be_taken_from_the_L0_block(tasks):
    """The allowlist covers L0 too: a witness here would spend the match."""
    for t in tasks[:4]:
        with pytest.raises(REF.E4ReferenceError, match="would strike it off"):
            REF.rng_witness(t)


# --- the digest binds ------------------------------------------------------

def test_the_frozen_plan_loads_and_its_digest_binds(tasks):
    plan = P.load_l0_plan()
    assert plan["n_tasks"] == 64
    assert P.l0_task_digest(plan["tasks"]) == P.L0_TASK_DIGEST


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda ts: ts[::-1], id="reversed"),
    pytest.param(lambda ts: ts[:-1], id="one_removed"),
    pytest.param(lambda ts: ts + [copy.deepcopy(ts[0])], id="one_added"),
    pytest.param(lambda ts: [dict(t, seed=t["seed"] + 1) if i == 5 else t
                             for i, t in enumerate(ts)], id="seed_edited"),
    pytest.param(lambda ts: [dict(t, t1j_mdPly=3) if i == 0 else t
                             for i, t in enumerate(ts)], id="depth_edited"),
    pytest.param(lambda ts: [dict(t, rep=0) if i == 1 else t
                             for i, t in enumerate(ts)], id="rep_edited"),
    pytest.param(lambda ts: [dict(t, colour_arm="t1j_red") if i == 44 else t
                             for i, t in enumerate(ts)], id="arm_edited"),
])
def test_the_digest_refuses_every_kind_of_edit(tasks, mutate):
    """Each control must first PROVE it changed the projected payload.

    Index 40 was `t1j_red` already, so setting it to `t1j_red` was a no-op and the
    control passed vacuously -- caught by this guard, which is why it is here and
    not just in the comment.
    """
    original = copy.deepcopy(list(tasks))
    bad = mutate(copy.deepcopy(list(tasks)))
    project = lambda ts: [[t[k] for k in P.L0_TASK_DIMENSIONS] for t in ts]
    assert project(bad) != project(original), "the tamper changed nothing to detect"
    assert P.l0_task_digest(bad) != P.L0_TASK_DIGEST


def test_a_tampered_frozen_plan_is_refused(tmp_path, tasks):
    plan = json.loads(open(P.L0_PLAN_REL).read())
    plan["tasks"] = list(reversed(plan["tasks"]))
    bad = tmp_path / "l0.json"
    bad.write_text(json.dumps(plan, indent=1, sort_keys=True))
    with pytest.raises(P.L0PlanError, match="sha256"):
        P.load_l0_plan(str(bad))


def test_the_task_digest_binds_even_when_the_file_hash_has_been_RE_PINNED(tmp_path):
    """Isolates the digest check, which the sha256 check would otherwise mask.

    A control that reverses the tasks is caught by the file hash first, so it
    proves nothing about the digest. The realistic failure is different: someone
    regenerates the plan, updates L0_PLAN_SHA256 to match, and forgets the task
    digest. Re-pinning the hash here reproduces exactly that, leaving the digest
    as the only gate standing.
    """
    plan = json.loads(open(P.L0_PLAN_REL).read())
    plan["tasks"] = list(reversed(plan["tasks"]))
    raw = json.dumps(plan, indent=1, sort_keys=True).encode()
    bad = tmp_path / "l0.json"
    bad.write_bytes(raw)
    import hashlib
    repinned = hashlib.sha256(raw).hexdigest()
    with mock_attr(P, "L0_PLAN_SHA256", repinned):
        with pytest.raises(P.L0PlanError, match="task digest"):
            P.load_l0_plan(str(bad))


import contextlib


@contextlib.contextmanager
def mock_attr(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@pytest.mark.parametrize("mutate,match", [
    (lambda ts: ts[:63], "expected exactly 64"),
    (lambda ts: [dict(t, seed=999) if i == 0 else t for i, t in enumerate(ts)], "outside"),
    (lambda ts: [dict(t, t1j_mdPly=3) if i == 0 else t for i, t in enumerate(ts)], "mdPly"),
    (lambda ts: [dict(t, t1j_mdFixedPly=False) if i == 0 else t for i, t in enumerate(ts)],
     "mdFixedPly"),
    (lambda ts: [dict(t, rep=0) if i == 1 else t for i, t in enumerate(ts)], "repetitions"),
    (lambda ts: [dict(t, task_id=ts[0]["task_id"]) if i == 1 else t
                 for i, t in enumerate(ts)], "duplicate task_id"),
])
def test_validate_refuses_a_broken_design(tasks, mutate, match):
    with pytest.raises(P.L0PlanError, match=match):
        P.validate_l0_schedule(mutate(copy.deepcopy(list(tasks))))


def test_validate_accepts_the_real_design(tasks):
    """The control: a validator that refuses everything proves nothing."""
    out = P.validate_l0_schedule(tasks)
    assert out["n_tasks"] == 64 and out["cells"] == 16 and out["reps_per_cell"] == 4


# --- the protocol: no early stop, one run ----------------------------------

def test_there_is_no_early_stop_and_may_stop_early_is_constant():
    assert L.EARLY_STOP is None
    for args in [(), (0, 0, 64), (7.0, 8, 64, [0.05, 0.95]), (64.0, 64, 64)]:
        assert L.may_stop_early(*args) is False


def test_the_frozen_plan_names_the_screen_rules_that_must_not_be_used():
    plan = P.load_l0_plan()
    banned = plan["protocol"]["screen_rules_must_not_be_used"]
    assert "early_in_band_forced" in banned
    for name in banned:
        assert hasattr(SCREEN, name), f"{name} must exist to be banned meaningfully"
    assert plan["protocol"]["runs"] == 1
    assert plan["protocol"]["early_stop"] is None


def test_a_partial_match_is_not_reported(tasks):
    ids = [t["task_id"] for t in tasks]
    rows = [{"task_id": i, "t1j_points": 1.0, "terminal_reason": "win", "plies": 40,
             "colour_arm": "t1j_red", "opening": "o1_center"} for i in ids[:63]]
    rep = L.match_report(rows, ids)
    assert rep["reported"] is False and "unplayed" in rep["reason"]


def test_duplicate_and_alien_results_are_refused(tasks):
    ids = [t["task_id"] for t in tasks]
    row = lambda i: {"task_id": i, "t1j_points": 1.0, "terminal_reason": "win",
                     "plies": 40, "colour_arm": "t1j_red", "opening": "o1_center"}
    assert L.match_report([row(ids[0])] * 64, ids)["reported"] is False
    assert L.match_report([row(i) for i in ids[:63]] + [row("alien")], ids)["reported"] is False


# --- the interval ------------------------------------------------------------

def _binom_pmf(k, n, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def _exact_coverage(interval, p, n=64):
    """Exact repeated-sampling coverage under iid Bernoulli(p)."""
    return sum(_binom_pmf(k, n, p) for k in range(n + 1)
               if interval(float(k), n)[0] <= p <= interval(float(k), n)[1])


def test_WILSON_IS_NOT_CONSERVATIVE_and_the_module_no_longer_says_it_is():
    """The claim this module used to make, disproved by its own arithmetic.

    A variance inequality is not a coverage statement. Wilson's exact coverage at
    n=64 dips below nominal on its own, and the previous "provably conservative"
    wording was wrong rather than merely unproven.
    """
    assert _exact_coverage(L.wilson_interval, 0.002) == pytest.approx(0.8797, abs=5e-4)
    assert _exact_coverage(L.wilson_interval, 0.5) < 0.95
    below = [p / 200 for p in range(1, 200)
             if _exact_coverage(L.wilson_interval, p / 200) < 0.95]
    assert below, "if Wilson were conservative the primary interval could be Wilson"

    doc = L.wilson_interval.__doc__
    assert "NOMINAL" in doc and "APPROXIMATE" in doc
    assert "NOT a guaranteed 95% interval" in doc
    # NOT a raw grep for "provably conservative": the module now QUOTES that wrong
    # claim in the passage explaining why it was wrong, so a text search would
    # match the correction itself -- the same trap as grepping for os.environ and
    # hitting the comment that says it is never read. Check the BAN instead.
    assert any("conservative" in c for c in L.FORBIDDEN_CLAIMS)
    assert L.match_report.__doc__.count("NOMINAL") >= 1


def test_hoeffding_is_the_primary_and_matches_its_closed_form():
    n, alpha = L.N_GAMES, L.ALPHA
    t = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    for score in (0.0, 16.0, 32.0, 56.0, 64.0):
        lo, hi = L.hoeffding_interval(score, n)
        p = score / n
        assert lo == pytest.approx(max(0.0, p - t), abs=1e-12)
        assert hi == pytest.approx(min(1.0, p + t), abs=1e-12)
    assert t == pytest.approx(0.16976268946757744, abs=1e-15)


def test_hoeffding_actually_covers_at_least_95_percent_where_wilson_does_not():
    """The point of the swap: the primary interval keeps its promise."""
    for p in (0.002, 0.5, 0.875):
        assert _exact_coverage(L.hoeffding_interval, p) >= 0.95
    worst = min(_exact_coverage(L.hoeffding_interval, p / 100) for p in range(1, 100))
    assert worst >= 0.95


def test_hoeffding_needs_no_iid_assumption_and_holds_for_bounded_scores():
    """Independent, bounded in [0,1], NOT identically distributed: 8 openings.

    Simulated deterministically: each opening has its own fixed win probability,
    and coverage is computed exactly by convolving the 8 distributions rather than
    by sampling -- no generator anywhere.
    """
    per_opening = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    dist = {0.0: 1.0}
    for prob in per_opening:                       # 8 games per opening
        for _ in range(8):
            nxt = {}
            for s, w in dist.items():
                nxt[s + 1] = nxt.get(s + 1, 0.0) + w * prob
                nxt[s] = nxt.get(s, 0.0) + w * (1 - prob)
            dist = nxt
    mu = sum(per_opening) * 8 / 64
    covered = sum(w for s, w in dist.items()
                  if L.hoeffding_interval(float(s), 64)[0] <= mu
                  <= L.hoeffding_interval(float(s), 64)[1])
    assert covered >= 0.95
    assert sum(dist.values()) == pytest.approx(1.0)


def test_neither_interval_constructs_a_generator(monkeypatch):
    import random
    made = []

    class _Trap(random.Random):
        def __init__(self, *a, **k):
            made.append(a)
            super().__init__(*a, **k)

    monkeypatch.setattr(random, "Random", _Trap)
    L.wilson_interval(56.0, 64)
    L.hoeffding_interval(56.0, 64)
    assert made == [], "an interval must not construct a generator"


def test_wilson_endpoints_are_exact_at_zero_and_full():
    lo, hi = L.wilson_interval(0.0, 64)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = L.wilson_interval(64.0, 64)
    assert hi == 1.0 and 0.9 < lo < 1.0


def test_both_intervals_contain_the_point_estimate_and_are_monotone():
    for interval in (L.wilson_interval, L.hoeffding_interval):
        prev = (-1.0, -1.0)
        for half in range(0, 129):
            s = half / 2.0
            lo, hi = interval(s, 64)
            assert lo <= s / 64 <= hi
            assert lo >= prev[0] and hi >= prev[1]
            prev = (lo, hi)


def test_the_variance_identity_is_true_but_is_only_DESCRIPTIVE():
    """Kept and demoted: the identity holds; the coverage conclusion never did."""
    n = 64
    for wins in range(0, n + 1, 7):
        for draws in range(0, n - wins + 1, 5):
            xs = [1.0] * wins + [0.5] * draws + [0.0] * (n - wins - draws)
            p = sum(xs) / n
            var = sum(x * x for x in xs) / n - p * p
            assert var == pytest.approx(p * (1 - p) - 0.25 * draws / n, abs=1e-12)
    assert "DESCRIPTIVE ONLY" in L.variance_deficit.__doc__
    assert "not a coverage" in L.variance_deficit.__doc__


# --- results must be results, not names ------------------------------------

_DEFAULT = object()


def _row(t, pts=1.0, plies=40, reason="win", winner=_DEFAULT):
    """`winner=None` must mean None -- `winner or anchor` silently made caps win."""
    return {"task_id": t["task_id"], "seed": t["seed"], "plies": plies,
            "terminal_reason": reason,
            "winner": t["anchor_colour"] if winner is _DEFAULT else winner,
            "t1j_points": pts}


def _valid(tasks, wins_every=8):
    out = []
    for i, t in enumerate(tasks):
        win = i % wins_every != 0
        other = "black" if t["anchor_colour"] == "red" else "red"
        out.append(_row(t, 1.0 if win else 0.0,
                        winner=t["anchor_colour"] if win else other))
    return out


def test_the_reviewers_payload_is_refused(tasks):
    """64 canonical IDs with t1j_points=0.25, plies=-7 and invented cells.

    The earlier reporter accepted exactly this and produced a rate and an
    interval, because it checked names and nothing else.
    """
    bogus = [dict(_row(t), t1j_points=0.25, plies=-7, opening="nonsense",
                  colour_arm="nonsense") for t in tasks]
    rep = L.match_report(bogus, tasks)
    assert rep["reported"] is False
    assert "plies" in rep["reason"]


def test_a_fully_valid_match_is_reported(tasks):
    """The control: a reporter that refuses everything proves nothing."""
    rep = L.match_report(_valid(tasks), tasks)
    assert rep["reported"] is True
    o = rep["overall"]
    assert o["games"] == 64 and o["t1j_score"] == 56.0
    assert o["t1j_rate"] == pytest.approx(0.875)
    assert o["ci95_hoeffding"][0] == pytest.approx(0.7052, abs=1e-4)
    assert o["ci95_wilson"][0] == pytest.approx(0.7723, abs=1e-4)
    assert "PRIMARY" in o["ci95_hoeffding_method"]
    assert "NOMINAL" in o["ci95_wilson_method"]
    assert o["cap_warning"] is False
    assert len(rep["by_opening"]) == 8 and len(rep["by_colour_arm"]) == 2
    assert all(v["games"] == 8 for v in rep["by_opening"].values())
    assert all(v["games"] == 32 for v in rep["by_colour_arm"].values())


@pytest.mark.parametrize("break_it,match", [
    (lambda r, t: dict(r, t1j_points=0.25), "t1j_points"),
    (lambda r, t: dict(r, t1j_points=0.0), "t1j_points"),
    (lambda r, t: dict(r, plies=-7), "plies"),
    (lambda r, t: dict(r, plies=0), "plies"),
    (lambda r, t: dict(r, plies=281), "plies"),
    (lambda r, t: dict(r, plies=40.5), "not an integer"),
    (lambda r, t: dict(r, winner="t1j"), "winner"),
    (lambda r, t: dict(r, terminal_reason="resign"), "terminal_reason"),
    (lambda r, t: dict(r, terminal_reason="cap"), "only at ply"),
    (lambda r, t: dict(r, seed=t["seed"] + 500), "seed"),
    (lambda r, t: {k: v for k, v in r.items() if k != "plies"}, "missing"),
])
def test_every_result_field_is_validated(tasks, break_it, match):
    rows = _valid(tasks)
    rows[3] = break_it(rows[3], tasks[3])
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is False, match
    assert match in rep["reason"], rep["reason"]


def test_a_result_cannot_misreport_its_own_cell(tasks):
    """opening/colour_arm come from the CANONICAL TASK, never from the row."""
    rows = [dict(r, opening="zzz", colour_arm="zzz") for r in _valid(tasks)]
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is True
    assert set(rep["by_opening"]) == {t["opening"] for t in tasks}
    assert "zzz" not in rep["by_opening"]


def test_the_recorded_vocabulary_matches_the_screens_own_records(tasks):
    """`winner` is a COLOUR. An earlier score_game compared it to the string
    't1j', which no recorded row has ever contained."""
    run = ("docs/superpowers/evidence/2026-08-26-t1j-e4-canonical-screen/"
           "07_e4_screen_results.jsonl")
    rows = [json.loads(l) for l in open(run)]
    res = [r for r in rows if r["record_type"] == "task_result"]
    assert {r["winner"] for r in res} <= set(L.WINNERS)
    assert {r["terminal_reason"] for r in res} <= set(L.TERMINAL_REASONS)
    for r in res:
        anchor = "red" if "t1j_red" in r["task_id"] else "black"
        assert L.score_game(r["winner"], anchor, r["terminal_reason"]) == r["t1j_points"]


def test_score_game_uses_colours_and_rejects_nonsense():
    assert L.score_game("red", "red", "win") == 1.0
    assert L.score_game("black", "red", "win") == 0.0
    assert L.score_game(None, "red", "cap") == 0.5
    assert L.score_game("red", "red", "cap") == 0.5
    for bad in (lambda: L.score_game("t1j", "red", "win"),
                lambda: L.score_game("red", "green", "win"),
                lambda: L.score_game("red", "red", "resign"),
                lambda: L.score_game(None, "red", "win")):
        with pytest.raises(ValueError):
            bad()


def test_a_partial_match_is_not_reported(tasks):
    rep = L.match_report(_valid(tasks)[:63], tasks)
    assert rep["reported"] is False and "unplayed" in rep["reason"]


def test_duplicate_and_alien_results_are_refused(tasks):
    rows = _valid(tasks)
    assert L.match_report([rows[0]] * 64, tasks)["reported"] is False
    assert L.match_report(rows[:63] + [dict(rows[0], task_id="alien")],
                          tasks)["reported"] is False


# --- the cap policy --------------------------------------------------------

def test_caps_never_stop_the_match_and_the_threshold_is_preregistered():
    assert L.CAP_NO_RATE_THRESHOLD == 32
    assert L.may_stop_early(64, 64, 64) is False
    assert any("cap" in r for r in L.NOT_ABORT_RULES)
    assert not any("cap" in r.lower() for r in L.L0_ABORT_RULES)


def test_cap_heavy_results_return_a_preregistered_NO_RATE_outcome(tasks):
    rows = _valid(tasks)
    for i in range(33):                        # 33 > 32
        rows[i] = _row(tasks[i], 0.5, plies=L.PLY_CAP, reason="cap", winner=None)
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is False
    assert rep["outcome"] == "CAP_SATURATED_NO_RATE"
    assert rep["cap_terminations"] == 33 and rep["games"] == 64


def test_a_few_caps_still_report_but_carry_a_warning(tasks):
    rows = _valid(tasks)
    for i in range(3):
        rows[i] = _row(tasks[i], 0.5, plies=L.PLY_CAP, reason="cap", winner=None)
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is True
    assert rep["overall"]["cap_warning"] is True
    assert rep["overall"]["cap_terminations"] == 3
    assert rep["overall"]["variance_deficit_descriptive"] > 0


def test_the_frozen_plan_carries_L0s_OWN_abort_rules_not_the_screens():
    plan = P.load_l0_plan()
    assert list(plan["abort_rules"]) == list(L.L0_ABORT_RULES)
    assert not any("cap" in r.lower() for r in plan["abort_rules"])
    cap = plan["protocol"]["cap_policy"]
    assert cap["threshold"] == L.CAP_NO_RATE_THRESHOLD
    assert cap["caps_never_stop_the_match"] is True
    assert "CAP_SATURATED_NO_RATE" in cap["above_threshold"]
    src = P.load_source_plan()
    assert plan["abort_rules"] != src["abort_rules"], (
        "importing the screen's list gives a runner two conflicting authorities")


def test_the_plan_reports_hoeffding_as_primary_and_wilson_as_nominal():
    plan = P.load_l0_plan()
    r = plan["reporting"]
    assert "hoeffding" in r["primary"].lower()
    assert "nominal" in r["secondary"].lower()
    assert "not a guarantee" in r["secondary"].lower()
    # NOT a blanket grep for "conservative": the section deliberately QUOTES the
    # withdrawn claim in `corrected_claim` and BANS the word in `forbidden`, so a
    # text search matches the correction itself. Third time this trap has bitten
    # in this workstream; check meaning, not characters.
    assert "does not follow" in r["corrected_claim"].lower()
    assert "an earlier version" in r["corrected_claim"].lower()
    assert any("conservative" in f for f in r["forbidden"])
    assert "at least nominal by construction" in r["primary"].lower()


# --- the schedule handed to the reporter must BE the frozen one -------------

def test_the_reporter_refuses_a_schedule_that_is_not_the_frozen_design(tasks):
    """A renamed opening with matching results used to report on an invented cell.

    The digest lived above the reporter, so the reporter could not check it -- and
    a reporter that cannot check the schedule will report on any schedule.
    """
    bad = [dict(t) for t in tasks]
    for t in bad:
        if t["opening"] == "o1_center":
            t["opening"] = "invented_cell"
    rep = L.match_report(_valid(bad), bad)
    assert rep["reported"] is False
    assert "not the frozen L0 design" in rep["reason"]
    assert "invented_cell" not in json.dumps(rep)


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda ts: [dict(t, seed=t["seed"] + 1000) if i == 0 else t
                             for i, t in enumerate(ts)], id="seed"),
    pytest.param(lambda ts: [dict(t, rep=(t["rep"] + 1) % 4) if i == 0 else t
                             for i, t in enumerate(ts)], id="rep"),
    pytest.param(lambda ts: [dict(t, t1j_mdPly=3) if i == 0 else t
                             for i, t in enumerate(ts)], id="mdPly"),
    pytest.param(lambda ts: ts[::-1], id="reordered"),
])
def test_every_schedule_edit_is_refused_by_the_reporter(tasks, mutate):
    bad = mutate([dict(t) for t in tasks])
    rep = L.match_report(_valid(bad), bad)
    assert rep["reported"] is False and "frozen L0 design" in rep["reason"]


def test_the_digest_lives_in_the_rules_layer_so_the_reporter_can_verify_it():
    """Structural: the reporter must not depend on the module that imports it."""
    import inspect
    assert L.L0_TASK_DIGEST == P.L0_TASK_DIGEST
    assert L.l0_task_digest is P.l0_task_digest
    # AST, not a text search: the rules module's own comment explains why the
    # digest lives there "rather than in l0_match_plan", and a grep matches that
    # prose. Fourth time this trap has bitten in this workstream.
    import ast
    tree = ast.parse(inspect.getsource(L))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
    assert not any("l0_match_plan" in m for m in imported), (
        f"the rules layer must not import the plan; found {sorted(imported)}")
    assert any("l0_match_rules" in m for m in _module_imports(P))


def _module_imports(mod):
    import ast
    import inspect
    out = set()
    for node in ast.walk(ast.parse(inspect.getsource(mod))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
            out.update(a.name for a in node.names)
    return out


# --- the cap/win vocabulary matches the production loop ---------------------

def test_a_natural_win_ON_the_cap_ply_is_a_WIN_not_a_cap(tasks):
    """e4_screen_runner checks state.winner() BEFORE the ply cap.

    So a win on ply 280 records terminal_reason='win'. An earlier validator
    equated plies == 280 with 'cap' and rejected this legitimate outcome.
    """
    rows = _valid(tasks)
    rows[0] = _row(tasks[0], 1.0, plies=L.PLY_CAP, reason="win",
                   winner=tasks[0]["anchor_colour"])
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is True
    assert rep["overall"]["plies_max"] == L.PLY_CAP
    assert rep["overall"]["cap_terminations"] == 0


def test_a_cap_must_be_at_the_cap_ply_and_have_no_winner(tasks):
    rows = _valid(tasks)
    rows[0] = _row(tasks[0], 0.5, plies=L.PLY_CAP, reason="cap", winner=None)
    assert L.match_report(rows, tasks)["reported"] is True      # the control

    rows[0] = _row(tasks[0], 0.5, plies=L.PLY_CAP, reason="cap", winner="red")
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is False and "no winner" in rep["reason"]

    rows[0] = _row(tasks[0], 0.5, plies=100, reason="cap", winner=None)
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is False and "only at ply" in rep["reason"]


class _EndState:
    """A synthetic terminal state at the ply cap. No board, no engine, no agent."""

    def __init__(self, ply, winner):
        self.ply = ply
        self._winner = winner
        self.to_move = "red"

    def winner(self):
        return self._winner

    def legal_moves(self):                       # never reached: the loop breaks first
        raise AssertionError("play_task asked for moves at a terminal state")

    def apply_move(self, move):
        raise AssertionError("play_task tried to move at a terminal state")


class _NullRecorder:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _play(ply, winner, anchor="red"):
    """Drive the REAL play_task. Any agent construction is an immediate failure."""
    from scripts.GPU.alphazero import e4_screen_runner as RUNNER

    def no_agents(task, mover):
        raise AssertionError("an agent was constructed at a terminal state")

    task = {"task_id": "behavioural-probe", "anchor_colour": anchor, "opening": "o1_center"}
    rec = _NullRecorder()
    return RUNNER.play_task(task=task, agent_for=no_agents,
                            state_factory=lambda t: _EndState(ply, winner),
                            binder=lambda *a, **k: None, rec=rec,
                            ply_cap=L.PLY_CAP), rec


def test_the_production_loop_records_a_WIN_on_the_cap_ply(tasks):
    """BEHAVIOURAL, not a substring search over the runner's source.

    An earlier version of this control compared the character offsets of two
    strings in `play_task`'s source, so matching prose would have satisfied it --
    the same defect class the AST check above avoids. This drives the real loop
    instead: a state at ply 280 whose winner() returns a colour must be recorded
    as a WIN, with no agent constructed.
    """
    out, rec = _play(L.PLY_CAP, "red", anchor="red")
    assert out["terminal_reason"] == "win"
    assert out["winner"] == "red"
    assert out["plies"] == L.PLY_CAP
    assert out["t1j_points"] == 1.0
    assert out["agents_built"] == 0
    # and the reporter accepts what the runner produced
    row = dict(out, task_id=tasks[0]["task_id"], seed=tasks[0]["seed"])
    L.validate_result(row, dict(tasks[0], anchor_colour="red"))


def test_the_production_loop_records_a_CAP_when_the_ply_cap_has_no_winner(tasks):
    out, rec = _play(L.PLY_CAP, None, anchor="red")
    assert out["terminal_reason"] == "cap"
    assert out["winner"] is None, "the runner records winner=None for a cap"
    assert out["plies"] == L.PLY_CAP
    assert out["t1j_points"] == 0.5
    assert out["agents_built"] == 0
    row = dict(out, task_id=tasks[0]["task_id"], seed=tasks[0]["seed"])
    L.validate_result(row, dict(tasks[0], anchor_colour="red"))


def test_the_production_loop_scores_a_loss_at_the_cap_ply_for_the_other_colour():
    out, _ = _play(L.PLY_CAP, "black", anchor="red")
    assert out["terminal_reason"] == "win" and out["t1j_points"] == 0.0


def test_both_runner_outcomes_round_trip_through_the_L0_validator(tasks):
    """Whatever play_task can emit at the cap, validate_result must accept."""
    for winner, reason, pts in (("red", "win", 1.0), ("black", "win", 0.0),
                                (None, "cap", 0.5)):
        out, _ = _play(L.PLY_CAP, winner, anchor="red")
        assert (out["terminal_reason"], out["t1j_points"]) == (reason, pts)
        row = dict(out, task_id=tasks[0]["task_id"], seed=tasks[0]["seed"])
        L.validate_result(row, dict(tasks[0], anchor_colour="red"))


# --- the estimand and the independence model -------------------------------

def test_the_estimand_is_defined_and_the_design_makes_it_the_plain_mean(tasks):
    assert "equally weighted mean" in L.ESTIMAND
    assert "ENGINE RANDOMNESS ONLY" in L.ESTIMAND
    assert "not a sample from a population of openings" in L.ESTIMAND
    cells = {}
    for t in tasks:
        cells[(t["opening"], t["colour_arm"])] = cells.get(
            (t["opening"], t["colour_arm"]), 0) + 1
    assert set(cells.values()) == {L.N_REPS}, (
        "the plain mean equals the equally weighted cell mean only if balanced")


def test_independence_is_reported_as_a_MODEL_not_a_measurement(tasks):
    rep = L.match_report(_valid(tasks), tasks)
    o = rep["overall"]
    assert o["estimand"] == L.ESTIMAND
    assert o["independence_is_modelled_not_measured"] == L.INDEPENDENCE_CAVEAT
    assert "MODELLED as independent" in o["independence_is_modelled_not_measured"]
    assert "not a measurement" in o["independence_is_modelled_not_measured"]
    assert "UNDER AN INDEPENDENCE MODEL" in o["ci95_hoeffding_method"]
    assert any("ARE independent" in c for c in L.FORBIDDEN_CLAIMS)


def test_the_plan_states_independence_as_a_model_too():
    plan = P.load_l0_plan()
    r = plan["reporting"]
    assert r["estimand"] == L.ESTIMAND
    assert r["independence_note"] == L.INDEPENDENCE_CAVEAT
    assert "MODELLED" in r["independence_note"]
    assert "rules out accidental" in r["independence_note"]


def test_the_plan_no_longer_claims_abort_rules_come_from_the_screen():
    plan = P.load_l0_plan()
    src = plan["parameter_source"]
    assert "abort_rules" not in src["read_from_source"]
    assert "abort_rules" in src["NOT_read_from_source"]
    assert list(plan["abort_rules"]) == list(L.L0_ABORT_RULES)


# --- L0 spends nothing -----------------------------------------------------

def test_zzz_the_registries_are_intact_after_this_file():
    """Runs last. Catches any test that left the seed registries modified."""
    assert (202613000, 202613064) in REF.ACCOUNTED_SEED_INTERVALS
    assert (202613000, 202613064) in REF.EXPOSED_SEED_INTERVALS
    assert (202613000, 202613064) in REF.RETIRED_SEED_INTERVALS
    for seed in (202613000, 202613031, 202613063):
        assert REF.seed_is_exposed(seed) and REF.seed_is_retired(seed)
    # the FROZEN plan is unchanged: it still describes the pre-run state, which is
    # what a preregistration is. The run's outcome lives in the run's evidence.
    plan = P.load_l0_plan()
    assert plan["seed_block"]["status"].startswith("RESERVED, UNSPENT")
    assert "NOT EXECUTED" in plan["status"]


@pytest.mark.parametrize("pts,why", [
    (True, "is a bool"),
    (False, "is a bool"),
    ("1.0", "not a JSON number"),
    (None, "not a JSON number"),
    ([1.0], "not a JSON number"),
    ({"v": 1.0}, "not a JSON number"),
    (float("nan"), "not finite"),
    (float("inf"), "not finite"),
    (float("-inf"), "not finite"),
])
def test_t1j_points_is_type_safe_and_refuses_rather_than_raising(tasks, pts, why):
    """float() accepted True and "1.0" as wins and raised on None or a list.

    A malformed record must produce a REFUSED REPORT, never a traceback.
    """
    rows = _valid(tasks)
    rows[0] = _row(tasks[0], pts)
    rep = L.match_report(rows, tasks)            # must not raise
    assert rep["reported"] is False
    assert why in rep["reason"], rep["reason"]


@pytest.mark.parametrize("pts", [1.0, 1, 0, 0.0])
def test_valid_numeric_scores_are_still_accepted(tasks, pts):
    """The control: int and float both remain legal JSON numbers."""
    rows = _valid(tasks)
    win = pts in (1, 1.0)
    other = "black" if tasks[0]["anchor_colour"] == "red" else "red"
    rows[0] = _row(tasks[0], pts,
                   winner=tasks[0]["anchor_colour"] if win else other)
    assert L.match_report(rows, tasks)["reported"] is True


def test_the_reporter_never_raises_on_a_hostile_record(tasks):
    """Fail-closed backstop: every field replaced by something absurd."""
    import itertools
    hostile = [True, "x", None, [1], {"a": 1}, float("nan"), -1, 10 ** 9]
    for field, value in itertools.product(
            ("t1j_points", "plies", "winner", "terminal_reason", "seed"), hostile):
        rows = _valid(tasks)
        rows[0] = dict(rows[0], **{field: value})
        rep = L.match_report(rows, tasks)        # must not raise, ever
        assert rep["reported"] is False, (field, value)
        assert isinstance(rep["reason"], str) and rep["reason"]


def test_the_fail_closed_backstop_converts_an_UNFORESEEN_exception_to_a_refusal(
        tasks, monkeypatch):
    """The specific type checks cover everything the hostile-record test throws,
    so nothing in that test ever reaches the backstop -- and a gate no test
    reaches is not a gate that has been shown to work. This reaches it directly.
    """
    def explode(row, task):
        raise KeyError("something nobody anticipated")

    monkeypatch.setattr(L, "validate_result", explode)
    rep = L.match_report(_valid(tasks), tasks)       # must not raise
    assert rep["reported"] is False
    assert "malformed result rejected" in rep["reason"]
    assert "KeyError" in rep["reason"]


def test_the_backstop_does_not_swallow_a_normal_validation_message(tasks):
    """It is a last resort, not a blanket: real ValueErrors keep their own text."""
    rows = _valid(tasks)
    rows[0] = _row(tasks[0], 1.0, plies=-7)
    rep = L.match_report(rows, tasks)
    assert rep["reported"] is False
    assert "plies -7 outside" in rep["reason"]
    assert "malformed result rejected" not in rep["reason"]


# --- the fail-closed boundary wraps the WHOLE binding path ------------------

@pytest.mark.parametrize("build,match", [
    (lambda rows, ts: ([None] + rows[1:], ts), "result 0 is NoneType"),
    (lambda rows, ts: (rows, ["not a dict"] + list(ts[1:])), "schedule entry 0 is str"),
    (lambda rows, ts: (rows, [None] + list(ts[1:])), "schedule entry 0 is NoneType"),
    (lambda rows, ts: ([dict(rows[0], task_id=None),
                        dict(rows[1], task_id="alien")] + rows[2:], ts),
     "task_id None"),
    (lambda rows, ts: ([dict(rows[0], task_id=[1])] + rows[1:], ts), "task_id [1]"),
    (lambda rows, ts: (["x"] + rows[1:], ts), "result 0 is str"),
    (lambda rows, ts: ({"a": 1}, ts), "results are dict"),
    (lambda rows, ts: (rows, None), "schedule is NoneType"),
    (lambda rows, ts: (rows, {"a": 1}), "schedule is dict"),
    (lambda rows, ts: (rows, [dict(ts[0], task_id=7)] + list(ts[1:])),
     "schedule entry 0 has task_id 7"),
])
def test_malformed_shapes_are_refused_before_any_comprehension(tasks, build, match):
    """The three escapes the guard used to miss, plus their neighbours.

    Each of these raised a traceback out of `bind_results` before reaching the
    per-result guard: AttributeError at r.get(), TypeError building by_id, and
    TypeError sorting a set containing both None and a string.
    """
    rows, ts = build(_valid(tasks), list(tasks))
    rep = L.match_report(rows, ts)               # must not raise
    assert rep["reported"] is False
    assert match in rep["reason"], rep["reason"]


def test_the_outer_boundary_catches_what_the_shape_checks_do_not(tasks, monkeypatch):
    """The blanket is the last resort, and it must actually be reachable."""
    def explode(results, tasks):
        raise RuntimeError("something structural nobody anticipated")

    monkeypatch.setattr(L, "_bind_results", explode)
    rep = L.match_report(_valid(tasks), tasks)
    assert rep["reported"] is False
    assert "malformed input rejected" in rep["reason"]
    assert "RuntimeError" in rep["reason"]


def test_the_reporter_never_raises_on_any_hostile_container(tasks):
    """Fuzz over shapes, not just field values. No call may raise."""
    good = _valid(tasks)
    hostile = [None, "x", 7, [1], {"a": 1}, (), True, float("nan")]
    for value in hostile:
        for rows, ts in (
            ([value] + good[1:], list(tasks)),
            (good, [value] + list(tasks[1:])),
            (value, list(tasks)),
            (good, value),
            ([dict(good[0], task_id=value)] + good[1:], list(tasks)),
            (good, [dict(tasks[0], task_id=value)] + list(tasks[1:])),
        ):
            rep = L.match_report(rows, ts)       # must not raise, ever
            assert rep["reported"] is False, (value, type(value))
            assert isinstance(rep["reason"], str) and rep["reason"]


def test_a_valid_match_still_reports_after_all_that(tasks):
    """The control: the boundary must not have become a blanket refusal."""
    rep = L.match_report(_valid(tasks), tasks)
    assert rep["reported"] is True
    assert rep["overall"]["games"] == 64
