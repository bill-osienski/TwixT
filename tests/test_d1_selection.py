"""D1 position selection -- plan 12.1-12.3, the FROZEN rule. NO EXECUTION.

No model is loaded, no JVM started, no seed drawn or registered, no T1j query
issued, no game played. Everything here reads the published L0 record through
D0's digest-verified binding and recomputes deterministic board facts, exactly
as D0 itself does.
"""
import dataclasses
import hashlib
import json

import pytest

from scripts.GPU.alphazero import d0_postmortem as D0
from scripts.GPU.alphazero import d1_selection as SEL
from scripts.GPU.alphazero import fpu_state_hash as FSH
from scripts.GPU.alphazero.game.twixt_state import TwixtState

RECORD = "docs/superpowers/evidence/2026-08-27-t1j-l0-canonical-match/06_l0_match_results.jsonl"
PLAN_JSON = "docs/superpowers/evidence/2026-08-26-t1j-l0-larger-match/01_l0_match_plan.json"


def _state(moves):
    st = TwixtState()
    for m in moves:
        st = st.apply_move(m)
    return st


# ══════════════════ 12.2: the canonical digest, implemented LITERALLY ═════════

def test_the_digest_is_sha256_not_the_existing_sha1_helper():
    """12.2 freezes SHA-256. `fpu_state_hash` offers a SHA-1 over a SUPERSET key
    whose extra fields happen to be constant across this cohort -- so it would
    dedupe identically and still not be the digest the preregistration named."""
    st = _state([(11, 11), (12, 13)])
    d = SEL.canonical_digest(st)
    assert len(d) == 64 and int(d, 16) >= 0
    assert d != FSH.canonical_state_sha1(st)


def test_the_digest_payload_is_exactly_the_three_frozen_fields():
    """Recomputed here from the frozen wording, not from the implementation."""
    st = _state([(11, 11), (12, 13)])
    pegs = sorted((r, c, p) for (r, c), p in st.pegs.items())
    payload = json.dumps((st.to_move, pegs, sorted(st.bridges)), sort_keys=True)
    assert SEL.canonical_digest(st) == hashlib.sha256(payload.encode()).hexdigest()


def test_a_field_outside_the_frozen_three_does_not_enter_the_digest():
    """NEGATIVE CONTROL on the payload. `max_plies_limit` is in the SHA-1
    helper's key and is NOT one of 12.2's three fields, so it must not move the
    digest -- otherwise the payload silently grew."""
    st = _state([(11, 11), (12, 13)])
    other = dataclasses.replace(st, max_plies_limit=99)
    assert other.max_plies_limit != st.max_plies_limit
    assert SEL.canonical_digest(other) == SEL.canonical_digest(st)
    assert FSH.canonical_state_sha1(other) != FSH.canonical_state_sha1(st)


@pytest.mark.parametrize("mutate", ["to_move", "pegs", "bridges"])
def test_each_frozen_field_changes_the_digest(mutate):
    st = _state([(11, 11), (12, 13), (13, 12)])
    if mutate == "to_move":
        other = dataclasses.replace(st, to_move="red" if st.to_move == "black" else "black")
    elif mutate == "pegs":
        other = st.apply_move(next(m for m in st.legal_moves()))
    else:
        other = dataclasses.replace(st, bridges=set())
        assert st.bridges != other.bridges, "the fixture grew no bridge to remove"
    assert SEL.canonical_digest(other) != SEL.canonical_digest(st)


def test_transpositions_collapse_to_one_digest():
    """The digest is a DEDUPLICATION LABEL: two move orders, one position."""
    a = _state([(11, 11), (12, 13), (13, 12), (10, 13)])
    b = _state([(13, 12), (12, 13), (11, 11), (10, 13)])
    assert a.pegs == b.pegs and a.bridges == b.bridges and a.to_move == b.to_move
    assert SEL.canonical_digest(a) == SEL.canonical_digest(b)


# ═══════════════ 12.1: the frozen selection rule, applied to the record ══════

@pytest.fixture(scope="module")
def bound():
    return D0.bind_record(RECORD, PLAN_JSON)


@pytest.fixture(scope="module")
def plies(bound):
    return SEL.discovery_plies(bound)


@pytest.fixture(scope="module")
def selection(bound):
    return SEL.select_all(bound)


@pytest.fixture(scope="module")
def frozen(bound, plies):
    """§12.1-12.3's output, BEFORE §13's exclusion. The §12 facts below are
    properties of the frozen rule and must keep being asserted against it."""
    return {(c["signature"], c["role"]): c for c in SEL.cohorts(plies)}


@pytest.fixture(scope="module")
def seeded(bound):
    """A prospective assignment over a hypothetical fresh interval. §13 reserves
    none, so nothing here reserves, registers or draws anything -- it is
    arithmetic over numbers that are not in any registry."""
    return SEL.select_all(bound, seed_interval=(300000000, 300000221))


def test_the_frozen_column_names_still_exist_in_ply_features():
    """12.1 asks for exactly this check: a frozen name that drifts from the code
    is a preregistration that no longer binds anything."""
    produced = D0.recomputable_columns()
    declared = {s["column"] for s in D0.CANDIDATE_SIGNATURES}
    for sig in SEL.SIGNATURES:
        assert sig["column"] in produced, sig
        assert sig["column"] in declared, sig


def test_selection_never_reaches_the_confirmation_half(plies):
    assert plies, "no discovery plies were produced; every check below is vacuous"
    assert {r["rep"] for r in plies} == set(D0.DISCOVERY_REPS)


def test_every_ply_carries_a_prefix_whose_length_is_its_own_ply(plies):
    for r in plies:
        assert len(r["prefix"]) == r["ply"], r["task_id"]


def test_the_carried_prefix_replays_to_the_recorded_digest(plies):
    """12.7 aborts on a prefix that does not replay to its digest, so the
    selection must not hand one over that cannot."""
    for r in plies[::97]:                       # every 97th: the whole set is ~3k
        assert SEL.canonical_digest(_state(r["prefix"])) == r["digest"], r["task_id"]


def test_the_frozen_counts_are_reproduced(frozen):
    """The numbers 12.1 and 12.4 froze, recomputed from the record. Asserted
    against §12's OWN output: §13 excludes afterwards and must not be able to
    disguise a drifted rule."""
    got = {k: len(c["rows"]) for k, c in frozen.items()}
    assert got == {("mover_fragmentation", "position"): 101,
                   ("mover_fragmentation", "control"): 60,
                   ("created_threat", "position"): 30,
                   ("created_threat", "control"): 36}
    assert sum(got.values()) == SEL.N_POSITIONS == 227


def test_the_frozen_cell_counts_are_reproduced(selection):
    cells = {c["signature"]: c["n_cells"] for c in selection["cohorts"]
             if c["role"] == "position"}
    assert cells == {"mover_fragmentation": 36, "created_threat": 12}


def test_every_selected_position_has_our_incumbent_to_move(selection):
    for c in selection["cohorts"]:
        assert c["rows"], c["signature"]
        for r in c["rows"]:
            assert D0.moved_by(r["colour_arm"], r["mover"]) == "ours", r["task_id"]


def test_positions_hold_the_signature_and_controls_negate_it(selection):
    for c in selection["cohorts"]:
        want = c["role"] == "position"
        for r in c["rows"]:
            assert bool(r[c["column"]]) is want, (c["signature"], c["role"], r["ply"])


def test_no_cohort_retains_two_positions_with_one_digest(selection):
    for c in selection["cohorts"]:
        digests = [r["digest"] for r in c["rows"]]
        assert len(set(digests)) == len(digests), c["signature"]


def test_no_cell_exceeds_the_cap_of_three(selection):
    for c in selection["cohorts"]:
        counts: dict = {}
        for r in c["rows"]:
            k = SEL.cell(r)
            counts[k] = counts.get(k, 0) + 1
        assert counts and max(counts.values()) <= SEL.PER_CELL_CAP == 3, c["signature"]


def test_each_cohort_is_in_the_frozen_total_order(selection):
    for c in selection["cohorts"]:
        keys = [(r["task_id"], r["ply"]) for r in c["rows"]]
        assert keys == sorted(keys), c["signature"]


def test_controls_come_only_from_cells_that_hold_a_selected_position(selection):
    """12.3: matched by construction, not by post-hoc pairing."""
    by_sig = {}
    for c in selection["cohorts"]:
        by_sig.setdefault(c["signature"], {})[c["role"]] = c
    for sig, roles in by_sig.items():
        cells = {SEL.cell(r) for r in roles["position"]["rows"]}
        assert cells, sig
        for r in roles["control"]["rows"]:
            assert SEL.cell(r) in cells, (sig, r["task_id"], r["ply"])


# ────────────────────────────── the seed assignment ──────────────────────────

def test_every_position_draws_from_inside_a_supplied_interval(seeded):
    lo, hi = 300000000, 300000221
    seeds = [r["seed"] for c in seeded["cohorts"] for r in c["rows"]]
    assert len(seeds) == 221
    assert all(lo <= s < hi for s in seeds)


def test_a_supplied_interval_is_assigned_injectively_and_exhausted(seeded):
    seeds = [r["seed"] for c in seeded["cohorts"] for r in c["rows"]]
    assert sorted(seeds) == list(range(300000000, 300000221))


def test_the_seed_assignment_is_deterministic(bound):
    a = SEL.select_all(bound, seed_interval=(300000000, 300000221))
    b = SEL.select_all(bound, seed_interval=(300000000, 300000221))
    key = lambda s: [(r["task_id"], r["ply"], r["seed"]) for c in s["cohorts"] for r in c["rows"]]
    assert key(a) == key(b)


def test_the_block_is_ACCOUNTED_and_RETIRED_after_the_VOID():
    """Registered for the execution authorization, then RETIRED when the single
    authorized run VOIDED. Accounted records the reservation; retired records
    that it may not be used again. NOT exposed: how many of the 227 were actually
    drawn is undetermined, and claiming all of them would overstate the record.

    Each registry is asserted NON-EMPTY first; a check over an empty collection
    passes vacuously, which would make this test decorative.
    """
    from scripts.GPU.alphazero import e4_screen_reference as REF
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                 "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS"):
        assert getattr(REF, name), f"vacuous: {name} is empty"
    for seed in range(*SEL.RETIRED_SEED_INTERVAL):
        st = REF.seed_status(seed)
        assert st["accounted"] and st["retired"], (seed, st)
        assert not (st["exposed"] or st["test_only"]), (seed, st)


def test_the_frozen_rule_retains_the_same_position_in_two_cohorts(frozen):
    """RECORDED, NOT HIDDEN, and PINNED so it cannot drift silently.

    12.1 and 12.3 deduplicate WITHIN a cohort; nothing in 12 deduplicates ACROSS
    them. So a board state can be a `mover_fragmentation` position and a
    `created_threat` control at once, and 227 retained positions cover fewer
    distinct states than that. The frozen counts already contain this -- cohorts
    that shared a digest would total 203, not 227 -- so it is what was frozen,
    not a departure from it.

    The consequence is real and belongs in the record: those states are queried
    TWICE per depth per cohort, i.e. four independent JVMs rather than two, and
    12.7's determinism check compares only within a pair. The extra pair is not
    compared against the first.
    """
    digests = [r["digest"] for c in frozen.values() for r in c["rows"]]
    assert len(digests) == 227
    assert len(set(digests)) == 203, "the cross-cohort overlap changed"
    rows_by_digest: dict = {}
    for c in frozen.values():
        for r in c["rows"]:
            rows_by_digest.setdefault(r["digest"], []).append(r)
    shared = {d: rs for d, rs in rows_by_digest.items() if len(rs) > 1}
    assert len(shared) == 24
    assert all(len(rs) == 2 for rs in shared.values())
    # Every duplicate is the SAME ply of the SAME game seen from two cohorts.
    for digest, rows in shared.items():
        assert len({(r["task_id"], r["ply"]) for r in rows}) == 1, rows[0]["task_id"]


def test_a_cohort_that_departs_from_the_frozen_table_is_refused(bound, monkeypatch):
    """The reconciliation inside `select_all`, REACHED ALONE.

    The frozen-count test above asserts the counts directly, so it passes
    whether or not `select_all` also checks them -- an injected-defect control
    proved that removing the internal check changed nothing it could see. This
    drives the check itself by declaring an expectation the record cannot meet,
    which is the case where it is the only thing standing between a drifted rule
    and a silently different cohort.
    """
    bad = tuple(dict(s, positions=s["positions"] + 1) for s in SEL.SIGNATURES)
    monkeypatch.setattr(SEL, "SIGNATURES", bad)
    with pytest.raises(SEL.D1SelectionError, match="12.1 froze"):
        SEL.select_all(bound)


# ══════════════════════ the run's INPUT, projected from the selection ════════

def test_the_run_manifest_carries_what_the_probe_and_5_4_both_need(selection):
    m = SEL.run_manifest(selection)
    assert len(m) == SEL.N_POSITIONS_AFTER_EXCLUSION == 221
    required = {"task_id", "ply", "seed", "digest", "prefix", "signature", "role",
                "opening", "colour_arm", "phase",
                "mover_more_fragmented", "created_threat"}
    for row in m:
        assert required <= set(row), sorted(required - set(row))
        assert len(row["prefix"]) == row["ply"] and row["prefix"]
        assert all(isinstance(x, list) and len(x) == 2 for x in row["prefix"])


def test_every_manifest_row_carries_its_cohort_label(selection):
    """5.4 records the D0 structural signature and the matched-control label; a
    row that cannot say which cohort it came from cannot be compared to one."""
    m = SEL.run_manifest(selection)
    seen = {(r["signature"], r["role"]) for r in m}
    assert seen == set(SEL.SEED_ASSIGNMENT_ORDER)
    for row in m:
        want = row["role"] == "position"
        assert bool(row[dict((s["name"], s["column"]) for s in SEL.SIGNATURES)[
            row["signature"]]]) is want


def test_the_manifest_is_json_round_trippable_and_deterministic(bound):
    a = SEL.run_manifest(SEL.select_all(bound))
    b = SEL.run_manifest(SEL.select_all(bound))
    assert a == b
    assert json.loads(json.dumps(a)) == a


# ══════════════ §13 Amendment 2: a post-selection exclusion ══════════════════
#
# §12.1-12.3 are applied EXACTLY as frozen and only then are six already-retained
# rows removed. It is an ENUMERATED set, not the rule `ply >= 5`: the low-ply
# qualification supports excluding these observed prefixes and no wider claim.
# There is no reselection, backfill, re-deduplication or re-capping.

PLAN = "docs/superpowers/plans/2026-08-27-t1j-sparring-postmortem-opponent-ladder.md"
LOWPLY = ("docs/superpowers/evidence/2026-08-28-t1j-lowply-qualification/"
          "19_lowply_records.json")


def test_the_excluded_digests_are_exactly_the_ones_the_plan_enumerates():
    """Code and plan must not drift: the plan is the authority."""
    import re
    plan = open(PLAN, encoding="utf-8").read()
    sec = plan[plan.index("## 13. Amendment 2"):]
    assert set(re.findall(r"`([0-9a-f]{64})`", sec)) == set(SEL.AMENDMENT2_EXCLUDED)
    assert len(SEL.AMENDMENT2_EXCLUDED) == 6


def test_the_excluded_digests_are_exactly_the_rows_that_FAILED_qualification():
    """Not 'the low plies' -- the rows OBSERVED to fail. A passing ply-5 row must
    survive, or the exclusion would be a rule invented after the result."""
    rec = json.load(open(LOWPLY, encoding="utf-8"))
    failed = {p["digest"] for p in rec["prefixes"] if not p["passed"]}
    passed = {p["digest"] for p in rec["prefixes"] if p["passed"]}
    assert set(SEL.AMENDMENT2_EXCLUDED) == failed
    assert not (set(SEL.AMENDMENT2_EXCLUDED) & passed)


def test_the_frozen_12_counts_are_still_verified_BEFORE_the_exclusion(selection):
    """§13 applies §12 exactly as frozen and excludes afterwards. If the §12
    counts were checked only after removal, a drifted rule could hide inside the
    exclusion."""
    sel = selection
    assert sel["n_positions_frozen"] == 227
    got = {(c["signature"], c["role"]): c["n_frozen"] for c in sel["cohorts"]}
    assert got == {("mover_fragmentation", "position"): 101,
                   ("mover_fragmentation", "control"): 60,
                   ("created_threat", "position"): 30,
                   ("created_threat", "control"): 36}


def test_the_prospective_counts_after_exclusion(selection):
    sel = selection
    assert sel["n_positions"] == SEL.N_POSITIONS_AFTER_EXCLUSION == 221
    assert sel["n_excluded"] == 6
    got = {(c["signature"], c["role"]): len(c["rows"]) for c in sel["cohorts"]}
    assert got == {("mover_fragmentation", "position"): 101,
                   ("mover_fragmentation", "control"): 54,
                   ("created_threat", "position"): 30,
                   ("created_threat", "control"): 36}


def test_no_excluded_row_survives_and_nothing_replaced_it(selection, frozen):
    """NO BACKFILL. `frozen` is §12's INDEPENDENT output, not derived from the
    kept set -- deriving it would make both assertions below vacuously true,
    which is what an earlier version of this test did."""
    kept = {r["digest"] for c in selection["cohorts"] for r in c["rows"]}
    frozen_digests = {r["digest"] for c in frozen.values() for r in c["rows"]}
    assert len(frozen_digests) == 203 and len(kept) < len(frozen_digests)
    assert not (kept & set(SEL.AMENDMENT2_EXCLUDED))
    assert kept <= frozen_digests, "a row appeared that frozen §12 did not retain"
    assert frozen_digests - kept == set(SEL.AMENDMENT2_EXCLUDED), \
        "the difference between frozen and kept must be EXACTLY the six excluded"


def test_the_exclusion_records_the_matched_control_consequence(selection):
    """§13.3 states it rather than repairing it: backfill would be reselection."""
    sel = selection
    imb = sel["control_imbalance"]
    assert imb["mover_fragmentation"]["control_cells_before"] == 24
    assert imb["mover_fragmentation"]["control_cells_after"] == 21
    assert imb["mover_fragmentation"]["position_cells_without_control_before"] == 12
    assert imb["mover_fragmentation"]["position_cells_without_control_after"] == 15
    assert imb["mover_fragmentation"]["position_cells"] == 36


def test_the_lowest_retained_ply_is_five_and_every_low_ply_row_was_tested(selection):
    sel = selection
    assert min(r["ply"] for r in sel["positions"]) == 5
    low = [r for r in sel["positions"] if r["ply"] <= 5]
    rec = {p["digest"] for p in json.load(open(LOWPLY, encoding="utf-8"))["prefixes"]}
    assert all(r["digest"] in rec for r in low), "an untested low-ply row survived"


# ─────────────────────────── seeds: none reserved ───────────────────────────

def test_no_seed_is_assigned_because_no_interval_is_reserved(selection):
    """§13 reserves nothing. The retired block may not be revived, so positions
    carry no seed until a fresh interval is separately authorized."""
    sel = selection
    assert all(r.get("seed") is None for r in sel["positions"])
    assert sel["seed_interval"] is None


def test_an_explicitly_supplied_interval_must_match_the_cohort_size(bound):
    with pytest.raises(SEL.D1SelectionError, match="221"):
        SEL.select_all(bound, seed_interval=(300000000, 300000100))


def test_a_supplied_interval_of_the_right_size_is_assigned_injectively(bound):
    sel = SEL.select_all(bound, seed_interval=(300000000, 300000221))
    seeds = [r["seed"] for r in sel["positions"]]
    assert sorted(seeds) == list(range(300000000, 300000221))


def test_the_retired_block_is_refused_as_a_seed_interval(bound):
    """It was consumed administratively by the D1 VOID and retired whole."""
    with pytest.raises(SEL.D1SelectionError, match="retired"):
        SEL.select_all(bound, seed_interval=(202614000, 202614221))


def test_the_frozen_signature_table_sums_to_the_frozen_total():
    """The cross-check the deleted runtime branch was really doing, as the static
    fact it is: 12.1's per-cohort numbers must add up to 12.4's 227."""
    assert sum(sig["positions"] + sig["controls"] for sig in SEL.SIGNATURES) \
        == SEL.N_POSITIONS == 227
    assert SEL.N_POSITIONS - len(SEL.AMENDMENT2_EXCLUDED) \
        == SEL.N_POSITIONS_AFTER_EXCLUSION == 221
    assert sum(SEL.AMENDMENT2_COUNTS.values()) == SEL.N_POSITIONS_AFTER_EXCLUSION


def test_a_post_exclusion_count_that_departs_from_13_3_is_refused(bound, monkeypatch):
    """§13.3's reconciliation, REACHED ALONE. The count test above asserts the
    numbers directly, so it passes whether or not `select_all` also checks them --
    an injected-defect control proved exactly that."""
    bad = dict(SEL.AMENDMENT2_COUNTS)
    bad[("mover_fragmentation", "control")] = 55
    monkeypatch.setattr(SEL, "AMENDMENT2_COUNTS", bad)
    with pytest.raises(SEL.D1SelectionError, match="13.3 froze 55"):
        SEL.select_all(bound)


def test_a_12_1_count_that_departs_from_the_record_is_refused(bound, monkeypatch):
    """The §12 reconciliation, reached alone and BEFORE any exclusion."""
    bad = tuple(dict(sig, controls=sig["controls"] + 1) for sig in SEL.SIGNATURES)
    monkeypatch.setattr(SEL, "SIGNATURES", bad)
    with pytest.raises(SEL.D1SelectionError, match="12.1 froze"):
        SEL.select_all(bound)


def test_selection_still_REFUSES_the_retired_block(bound):
    """Requirement 3 of the handoff, written as a LITERAL.

    🔑 An injected-defect control proved the first version of this test could not
    fail: it passed `SEL.RETIRED_SEED_INTERVAL`, so when the defect MOVED that
    constant the test moved with it and kept passing. A test that reads the
    constant it is checking cannot see the constant change.

    The interval below is 221 seeds -- the size the cohort wants -- so it clears
    the length check and reaches the RETIREMENT guard alone. The exact 227-seed
    block would be refused by length first, and would prove only that.
    """
    with pytest.raises(SEL.D1SelectionError, match="retired"):
        SEL.select_all(bound, seed_interval=(202614000, 202614221))
    with pytest.raises(SEL.D1SelectionError, match="retired"):
        SEL.select_all(bound, seed_interval=(202614000, 202614227))


def test_selection_assigned_seeds_ALL_pass_the_RUNTIME_seed_check(bound):
    """Requirement 4, end to end: the seeds selection writes onto positions are
    the seeds `d1_probe._check_seed` admits. Two modules, one block."""
    from scripts.GPU.alphazero import d1_probe as D1
    sel = SEL.select_all(bound, seed_interval=SEL.SEED_INTERVAL)
    seeds = [p["seed"] for p in sel["positions"]]
    assert len(seeds) == 221 and len(set(seeds)) == 221
    for s in seeds:
        assert D1._check_seed(s) == s
