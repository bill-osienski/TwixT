"""D1″ — does our SEARCH lose what our POLICY already found?

The frozen plan is `docs/superpowers/2026-09-08-t1j-d1second-search-readout-proposal.md`
(amended at 0b2934c). D1′ asked whether T1j's move sits OUTSIDE our raw policy's
top five more often at fragmentation positions than at matched controls, and
returned NO_GO: a small positive but unstable excess. That constrains the
fragmentation-associated EXCESS and says nothing about the level the two roles
share, so it licenses no claim about how often our policy ranks T1j's move highly
overall.

D1″ asks the complementary question, on the rows where the policy DID rank it
highly:

    ss := rank_raw(t1j_move_6) <= 5  AND  rank_visit(t1j_move_6) > 5

-- our policy put T1j's move in its top five and, after 400 simulations, the
visit distribution did not. `ss` and `lprd` are DISJOINT BY CONSTRUCTION, which
is what makes this a new question rather than a second test of the old one, and
`check_disjoint` asserts it on the real rows at execution time.

🔴 THIS MODULE ACQUIRES NOTHING. No model, no JVM, no query, no game, no seed, no
registry, no gate. It is arithmetic over a D1 record that already exists, and it
refuses any record that is not a completed D1 run by our model on the pinned
toolchain -- `d1prime_analysis.check_report_contract` does that work and is
reused, not reimplemented.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from . import d1prime_analysis as DP

Move = Tuple[int, int]


class D1SecondError(DP.D1PrimeError):
    """A D1″ refusal. Subclasses D1′'s so a caller catching the analysis family
    catches both, and so the shared contract's refusals keep their own name."""


#: EVERY SHARED CONSTANT IS BOUND, NOT RETYPED -- the same object as D1′'s, so a
#: change to one cannot leave the other describing a different rule. Only the
#: bootstrap seed differs, and it differs on purpose (below).
K_SS = DP.K_LPRD                              # 5, unchanged: same K, other side
T_THRESHOLD = DP.T_THRESHOLD                  # 0.15, DELIBERATELY unchanged
B_REPLICATES = DP.B_REPLICATES
FLOOR = DP.FLOOR
PRIMARY_COHORT = DP.PRIMARY_COHORT
SECONDARY_COHORT = DP.SECONDARY_COHORT

#: The per-row boolean this analysis compares between roles.
INDICATOR = "ss"

#: 🔑 A DIFFERENT ANALYSIS-ONLY SEED. Reusing D1′'s 20260907 over the same games
#: would make D1″'s replicate draw a deterministic repeat of D1′'s, so the two
#: stability intervals would share every resampling accident. In no registry,
#: drawn from no experimental block.
BOOTSTRAP_SEED = 20260908


def rank_visit(visits: Mapping[Any, float]) -> Dict[Move, int]:
    """Rank every legal move by descending root visit count, 1 = most visited.

    TIES BROKEN BY THE CANONICAL MOVE ORDER (row, col) -- and by nothing else.
    🔑 THIS FUNCTION TAKES THE VISITS AND NO OTHER INPUT. If it could see the
    policy, the visit ranking would inherit the ordering it is being compared
    against and `ss` would measure agreement with itself.

    Zero-visit moves are RANKED, not dropped: at 400 simulations most legal moves
    have none, and discarding them would leave `rank_visit` undefined exactly
    where the metric needs it.
    """
    counts = {DP._move(k): float(v) for k, v in visits.items()}
    if not counts:
        raise D1SecondError("the root carries no legal move, so nothing can be ranked")
    order = sorted(counts, key=lambda m: (-counts[m], m[0], m[1]))
    return {m: i + 1 for i, m in enumerate(order)}


def suppression_row(pos: Mapping[str, Any]) -> Dict[str, Any]:
    """One D1 per-position record -> one D1″ analysis row.

    Built ON TOP of `d1prime_analysis.position_row`, so both analyses carry
    identical labels, identical binding fields and identical `lprd` -- and the
    disjointness claim is checkable on the same row rather than across two
    differently-built ones.
    """
    base = DP.position_row(pos)
    inc = pos["incumbent"]
    raw_visits = inc.get("root_visits") or {}
    # 🔴 A GUARD FOR THE EMPTY CASE STOOD HERE AND WAS DELETED. It raised its own
    # message, but `rank_visit` below already refuses an empty root, so an
    # injected-defect control that removed the guard was NOT CAUGHT: the refusal
    # happened anyway, one line later. A branch no control can distinguish is a
    # branch that proves nothing, so the refusal now has ONE owner and the test
    # asserts ITS message.
    ranks_visit = rank_visit(raw_visits)
    ranks_raw = DP.rank_raw(inc["raw_policy"])
    # D1 VOIDs a record whose policy and visits cover different legal sets. The
    # analysis refuses one too rather than ranking over a set it did not score.
    if set(ranks_visit) != set(ranks_raw):
        raise D1SecondError(
            f"{pos.get('task_id')}: the raw policy covers {len(ranks_raw)} moves and the "
            f"root visits {len(ranks_visit)}; they must be the same legal set")
    t1j = base["t1j_move_6"]
    rank_raw_t1j, rank_visit_t1j = ranks_raw[t1j], ranks_visit[t1j]
    visits_t1j = float({DP._move(k): float(v) for k, v in raw_visits.items()}[t1j])
    row = dict(base)
    row.update({
        "rank_visit_t1j": rank_visit_t1j,
        "visits_t1j": visits_t1j,
        # THE PRIMARY INDICATOR.
        "ss": rank_raw_t1j <= K_SS and rank_visit_t1j > K_SS,
        # The tie-free strict variant: reported, never decisive.
        "ss0": rank_raw_t1j <= K_SS and visits_t1j == 0.0,
        "n_legal": len(ranks_raw),
    })
    return row


def rows_from_d1_report(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [suppression_row(p) for p in report["positions"]]


def check_disjoint(rows: Iterable[Mapping[str, Any]]) -> None:
    """EXECUTION-TIME PRECONDITION: no row may score both `ss` and `lprd`.

    `ss` requires `rank_raw <= 5` and `lprd` requires `rank_raw > 5`, so a row
    carrying both did not come from these definitions -- the record, the labels
    or the arithmetic is wrong, and the statistic would be comparing a mixture.
    Asserted on the REAL rows every time the analysis runs, not only in tests:
    a property that holds by construction is exactly the kind that stops holding
    silently.
    """
    bad = [r.get("task_id") for r in rows if r.get("ss") and r.get("lprd")]
    if bad:
        raise D1SecondError(
            f"{len(bad)} row(s) score BOTH ss and lprd, which the definitions make "
            f"disjoint (first {bad[0]!r}). The rows did not come from "
            f"`suppression_row`, or the record they describe is wrong.")


def _readout_summary(rows: Sequence[Mapping[str, Any]], *, cohort: str) -> Dict[str, Any]:
    """DESCRIPTIVE, per role: the OTHER mechanism -- the readout declining the
    move the search preferred. Deliberately not the primary metric: one frozen
    decision rule, one metric. Nothing here is consulted by the decision.
    """
    out: Dict[str, Any] = {}
    for role in DP.ROLES:
        sel = [r for r in rows if r["signature"] == cohort and r["role"] == role]
        if not sel:
            out[role] = {"n": 0, "overrode_leader_rate": None, "ss_rate": None}
            continue
        for field in ("overrode_leader", "ss"):
            if any(field not in r for r in sel):
                raise D1SecondError(
                    f"a {role} row carries no {field!r}; the readout summary would "
                    f"describe a measurement that is absent")
        out[role] = {
            "n": len(sel),
            "overrode_leader_rate": sum(bool(r["overrode_leader"]) for r in sel) / len(sel),
            "ss_rate": sum(bool(r["ss"]) for r in sel) / len(sel),
        }
    return out


def _suppression_decision(rows: Sequence[Mapping[str, Any]],
                          strata: Sequence[Mapping[str, Any]], *,
                          cohort: str = PRIMARY_COHORT, B: int = B_REPLICATES,
                          seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """PRIVATE KERNEL -- the frozen §3-§5 rule, computed from whatever rows it is
    given.

    🔴 IT BINDS NOTHING, exactly like D1′'s kernel: it exists so the statistic can
    be exercised on synthetic cohorts. `analyse_search_suppression` is what
    production calls, and it binds the record, the cohort and the design FIRST.
    The eligibility floor is checked BEFORE any replicate is drawn; every
    secondary report is computed but cannot produce GO.
    """
    primary = DP.matched_statistic(rows, cohort=cohort, indicator=INDICATOR)
    met, detail = DP.floors_met(primary)
    secondary = {
        "ss0": DP.matched_statistic(rows, cohort=cohort, indicator="ss0"),
        "created_threat": DP.matched_statistic(
            rows, cohort=SECONDARY_COHORT if cohort == PRIMARY_COHORT else PRIMARY_COHORT,
            indicator=INDICATOR),
        "readout": _readout_summary(rows, cohort=cohort),
    }
    common = {"floor": detail, "statistic": primary, "secondary": secondary,
              "indicator": INDICATOR}
    if not met:
        return {"outcome": "NO_GO — insufficient support", "T": primary["T"],
                "interval": None, **common}
    stab = DP.stability_interval(rows, strata, cohort=cohort, B=B, seed=seed,
                                 indicator=INDICATOR)
    return {"outcome": DP._decide(stab), "T": stab["T"], "interval": stab["interval"],
            "stability": stab, **common,
            "meaning": ("GO = sufficient effect and resampling stability to justify "
                        "PROPOSING ONE confirmation acquisition; not proof of a "
                        "population effect, and not authorization to acquire anything")}


#: What the numbers mean, carried in the OUTPUT and not only in the plan -- a
#: claim that lives only in a document is one a reader of the result never sees.
CLAIM = ("T describes matched search suppression in the frozen selected cohort. The "
         "stratified cluster resampling measures T's sensitivity to reweighting the "
         "observed games. Its central 95% interval is an empirical stability interval, "
         "NOT an established confidence interval for repeated L0 designs. GO means enough "
         "effect and stability to justify proposing ONE confirmation acquisition, not "
         "statistical proof of a population effect.")


def analyse_search_suppression(d1_report: Mapping[str, Any]) -> Dict[str, Any]:
    """THE production entry. Takes the D1 report and NOTHING ELSE.

    🔴 EVERY OTHER INPUT IS RESOLVED HERE, so none can be supplied: the cohort and
    the design from `d1prime_analysis.resolve_canonical_cohort`, the repetitions
    from the development half, B and the PRNG seed from this module's frozen
    constants. D1′ learned this the hard way -- an entry accepting `rows`,
    `frozen_cohort`, `tasks`, `reps`, `B` and `seed` lets forged rows with a
    matching forged manifest produce a GO that looks frozen.

    The confirmation half is not opened, selected from, counted or described.
    """
    canon = DP.resolve_canonical_cohort()
    acquisition = DP.check_report_contract(d1_report, canon["rows"])
    rows = rows_from_d1_report(d1_report)
    check_disjoint(rows)                       # precondition, on the REAL rows
    out = DP._analyse(rows, frozen_cohort=canon["rows"], tasks=canon["tasks"],
                      reps=canon["reps"], kernel=_suppression_decision,
                      B=B_REPLICATES, seed=BOOTSTRAP_SEED)
    out["resolved"] = {k: canon[k] for k in ("record", "plan", "cohort_source", "n_positions")}
    out["acquisition"] = acquisition
    out["prng"] = {"bit_generator": "PCG64", "seed": BOOTSTRAP_SEED, "B": B_REPLICATES}
    out["claim"] = CLAIM
    return out
