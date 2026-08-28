"""Test-review of the low-ply T1j qualification CARD. NOTHING RUNS.

No JVM, no model, no seed, no D1 retry. Every test here reads committed files
and recomputes; the qualification itself has no runner and is not authorized.

WHAT THIS FILE IS FOR. A card's numbers are worth exactly as much as the check
that they were computed rather than typed. §12's frozen counts earned that with
a test; these do too, and the same way: recompute from the source artifact and
compare, then assert the card's PROSE carries the same figures, so the document
and the frozen data cannot drift apart silently.
"""
import hashlib
import json

import pytest

CARD = "docs/superpowers/2026-08-28-t1j-lowply-qualification-card.md"
FROZEN = "docs/superpowers/evidence/2026-08-28-t1j-lowply-qualification/01_frozen_prefixes.json"
D1_INPUT = "docs/superpowers/evidence/2026-08-28-t1j-d1-execution/02_positions.json"
E4 = "docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/03_results.jsonl"

DEPTHS = (3, 6)
INVOCATIONS_PER_DEPTH = 2


@pytest.fixture(scope="module")
def card():
    return open(CARD, encoding="utf-8").read()


@pytest.fixture(scope="module")
def frozen():
    return json.load(open(FROZEN, encoding="utf-8"))


@pytest.fixture(scope="module")
def d1_rows():
    return json.load(open(D1_INPUT, encoding="utf-8"))


# ───────────────── the frozen list is DERIVED, not typed ────────────────────

def test_the_frozen_list_is_bound_to_the_d1_input_it_came_from(frozen):
    got = hashlib.sha256(open(D1_INPUT, "rb").read()).hexdigest()
    assert frozen["source"] == D1_INPUT
    assert frozen["source_sha256"] == got, "the frozen list names a different input"


def test_the_frozen_list_recomputes_from_the_d1_input(frozen, d1_rows):
    """The whole rule, re-applied here: ply <= 5, dedup by digest, earliest by
    (task_id, ply). If the card's list were hand-edited this fails."""
    low = [r for r in d1_rows if r["ply"] <= 5]
    seen, uniq = set(), []
    for r in sorted(low, key=lambda r: (r["task_id"], r["ply"])):
        if r["digest"] in seen:
            continue
        seen.add(r["digest"])
        uniq.append(r)
    assert [p["digest"] for p in frozen["prefixes"]] == [r["digest"] for r in uniq]
    assert [p["prefix"] for p in frozen["prefixes"]] == [r["prefix"] for r in uniq]
    assert frozen["n_rows_at_ply_le_5"] == len(low)
    assert frozen["n_retained"] == len(uniq)


def test_nothing_was_re_selected_only_filtered(frozen, d1_rows):
    """A qualification that invented its own positions would be a second
    selection rule competing with the frozen one."""
    by_digest = {r["digest"]: r for r in d1_rows}
    for p in frozen["prefixes"]:
        assert p["digest"] in by_digest, p["digest"]
        src = by_digest[p["digest"]]
        for field in ("task_id", "ply", "prefix", "opening", "colour_arm",
                      "signature", "role"):
            assert p[field] == src[field], (p["digest"], field)


def test_every_frozen_prefix_is_replayable_and_low_ply(frozen):
    for p in frozen["prefixes"]:
        assert 1 <= p["ply"] <= 5, p
        assert len(p["prefix"]) == p["ply"], p
        assert p["prefix"], "an empty prefix cannot be replayed"
        assert all(isinstance(m, list) and len(m) == 2 for m in p["prefix"])


def test_the_digests_are_distinct_so_the_dedup_was_a_no_op(frozen):
    digests = [p["digest"] for p in frozen["prefixes"]]
    assert len(set(digests)) == len(digests)
    assert frozen["dedup_removed"] == 0, "recorded as a no-op; it is not one"


def test_only_odd_plies_appear_and_the_card_explains_why(frozen, card):
    """12.1 selects plies where OUR INCUMBENT moves; in the t1j_red arm T1j is
    red and moves at even plies. The absence of plies 2 and 4 is a consequence
    of the frozen rule, not a gap in this list."""
    assert sorted({p["ply"] for p in frozen["prefixes"]}) == [1, 3, 5]
    assert {p["colour_arm"] for p in frozen["prefixes"]} == {"t1j_red"}
    assert "no even plies" in card and "our incumbent is to move" in card
    assert "says nothing about plies 2 and 4" in card


# ─────────────────── the card's arithmetic, recomputed ──────────────────────

def test_the_card_states_the_query_count_the_frozen_list_implies(frozen, card):
    n = frozen["n_retained"]
    queries = n * len(DEPTHS) * INVOCATIONS_PER_DEPTH
    invocations = queries + n                     # + one replay per prefix
    assert queries == 36 and invocations == 45, (queries, invocations)
    assert f"{queries} queries" in card
    assert f"{invocations} JVM invocations" in card
    assert "9 prefixes × 2 depths" in card   # the literal the card carries


def test_the_cost_basis_matches_the_E4_record_the_card_cites(card):
    """The card claims maxima from E4's own results. Recompute them."""
    rows = [json.loads(l) for l in open(E4, encoding="utf-8") if l.strip()]
    q = [r for r in rows if r["record_type"] == "query"]
    mx = {d: max(r["wall_ms"] for r in q if r["depth"] == d) for d in DEPTHS}
    assert round(mx[3], 1) == 215.7 and round(mx[6], 1) == 2734.5, mx
    assert "215.7" in card and "2734.5" in card


def test_the_indicative_subtotal_is_arithmetic_not_assertion(frozen, card):
    rows = [json.loads(l) for l in open(E4, encoding="utf-8") if l.strip()]
    q = [r for r in rows if r["record_type"] == "query"]
    mx = {d: max(r["wall_ms"] for r in q if r["depth"] == d) for d in DEPTHS}
    per = sum(INVOCATIONS_PER_DEPTH * mx[d] for d in DEPTHS)
    total = frozen["n_retained"] * per
    assert 53_000 < total < 53_200, total
    assert f"{round(total):,} ms" in card, f"{round(total):,} ms"


def test_the_two_limits_are_NOT_redundant_and_the_card_shows_it(frozen, card):
    """The same argument 12.10 had to make: a per-call timeout does not bound a
    run, and a run cap does not localise a hung call."""
    queries = frozen["n_retained"] * len(DEPTHS) * INVOCATIONS_PER_DEPTH
    assert queries * 120 == 4320                       # seconds
    assert 4320 > 900, "the per-call timeout would bound the run; the claim is wrong"
    assert "4,320 s" in card and "72 minutes" in card
    assert "900 s (15 min)" in card


def test_the_card_records_the_12_9_discrepancy_rather_than_reusing_it(card):
    """§12.9 lists depth 3 = 121 (the MINIMUM) and depth 6 = 2749 (absent from
    the record). The card must say so rather than quietly inherit it."""
    rows = [json.loads(l) for l in open(E4, encoding="utf-8") if l.strip()]
    walls = [r["wall_ms"] for r in rows if "wall_ms" in r]
    assert not any(round(w) == 2749 for w in walls), "2749 IS in the record after all"
    q3 = [r["wall_ms"] for r in rows if r["record_type"] == "query" and r["depth"] == 3]
    # 121 is the minimum TRUNCATED, not rounded: min is 121.65, and round() is 122.
    assert int(min(q3)) == 121, min(q3)
    assert round(min(q3)) != 121, "then 121 could have been a rounded value after all"
    assert "appears nowhere in the" in card and "minimum" in card


# ──────────────── the boundary the card is not allowed to cross ─────────────

def test_the_card_declares_its_OWN_gate_and_reads_no_other(card):
    assert "LOWPLY_QUALIFICATION_AUTHORIZED = False" in card
    assert "must not read" in card and "D1_EXECUTION_AUTHORIZED" in card


@pytest.mark.parametrize("barred", [
    "no incumbent model load", "no seed action", "no D1 retry",
    "no confirmation data", "no training", "no push",
])
def test_the_card_states_every_barred_action(card, barred):
    assert barred in card, barred


def test_the_card_says_it_needs_no_seed_interval_and_why(card):
    assert "no seed interval at all" in card
    assert "never invokes the incumbent" in card


def test_the_card_separates_FAIL_from_VOID(card):
    """D1 conflated them: a T1j reply that did not complete its depth aborted the
    whole run, so the observation could not be recorded. Here it IS the
    measurement."""
    assert "is a RESULT, not an abort" in card
    assert "T1j failing to complete is **the\nmeasurement**" in card.replace("\r", "")


def test_the_card_prohibits_the_same_jvm_repeats_mode(card):
    assert "`repeats>1` is PROHIBITED" in card
    assert "agrees by construction" in card


def test_the_card_does_not_claim_the_qualification_ran(card):
    assert "NOT AUTHORIZED, NOT RUN" in card
    assert "Nothing has run" in card
    for forbidden in ("PASS —", "we observed", "the run showed"):
        assert forbidden not in card, forbidden
