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

import math
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


#: Every incumbent observable the analysis or its summary reads. Required BY
#: NAME: a field that is merely absent would otherwise be scored as a value.
REQUIRED_OBSERVABLES = ("raw_policy", "root_visits", "n_legal", "root_total_visits",
                        "selected_visit_rank", "selected_visit_count",
                        "selected_policy_rank", "selected_policy_mass",
                        "root_top1_share", "readout_overrode_leader")


#: LISTED IN THE PLAN'S SCHEMA INVENTORY AND DELIBERATELY NOT VALIDATED, because
#: D1″ reads it nowhere. The inventory says what the record CARRIES; this says
#: what the analysis USES. Declared rather than merely absent, so the promise the
#: plan makes and the promise this module keeps are the same one -- an unlisted
#: omission is how a documented guarantee quietly stops existing. Validating a
#: field no statistic reads would add a refusal path with nothing behind it, and
#: the writer may legitimately record `top2` as None ("not captured").
UNUSED_OBSERVABLES = ("top2",)


def frozen_sim_budget() -> int:
    """The search budget every root must have spent, READ from the frozen L0
    plan's configuration through `d1_probe.frozen_incumbent_identity` -- never
    retyped here, so this module cannot drift from the search that ran.

    🔑 WHY THE TOTAL IS BOUND AT ALL. The hypothesis is about what the search does
    to a move AFTER the frozen simulation budget is spent. A root that ran a
    different budget is not evidence about that search, and every internal
    consistency check passes on a forged map totalling one as long as
    `root_total_visits` is changed to agree.

    ⚠ A REAL RECORD THAT DISAGREES IS A REFUSAL TO REVIEW, never a tolerance to
    widen after the fact.
    """
    from . import d1_probe as D1P
    sims = D1P.frozen_incumbent_identity()["eval_config"]["mcts_sims"]
    if type(sims) is not int or sims <= 0:
        raise D1SecondError(f"the frozen configuration's mcts_sims is {sims!r}")
    return sims


def _real(x: Any) -> bool:
    """A real number, TYPE-STRICTLY: `True` is not 1 and "3" is not three.

    🔑 THE SINGLE OWNER of that rule. `type(x) in (int, float)` already excludes
    `bool`, so the callers below carried a redundant `isinstance(x, bool)` clause
    that no control could reach -- and a control that deleted one was NOT CAUGHT
    because the duplicate still refused. The clause is gone; this function is
    where bools are rejected, and a control that weakens it is observable.
    """
    return type(x) in (int, float) and math.isfinite(x)


def _validate_position(pos: Mapping[str, Any], where: str) -> None:
    """One position's observables: present, well typed, and AGREEING with the
    maps they claim to describe."""
    inc = pos.get("incumbent")
    if not isinstance(inc, dict):
        raise D1SecondError(f"{where}: the position carries no incumbent record")
    missing = [f for f in REQUIRED_OBSERVABLES if f not in inc]
    if missing:
        raise D1SecondError(
            f"{where}: the incumbent record is missing {missing}, which the analysis "
            f"and its frozen summary read; `eval_replay.ply_record` and `d1_probe` "
            f"write every one of them.")

    policy = inc["raw_policy"]
    if not isinstance(policy, dict) or not policy:
        raise D1SecondError(f"{where}: raw_policy is empty or not a mapping")
    for k, v in policy.items():
        if not _real(v) or v < 0:
            raise D1SecondError(
                f"{where}: raw_policy[{k!r}] is {v!r} ({type(v).__name__}); every mass "
                f"must be a finite non-negative real, and a bool is not one.")
    if sum(float(v) for v in policy.values()) <= 0:
        raise D1SecondError(
            f"{where}: raw_policy sums to zero, so its ranking would be pure tie-break "
            f"over every legal move")

    visits = inc["root_visits"]
    if not isinstance(visits, dict) or not visits:
        raise D1SecondError(f"{where}: root_visits is empty or not a mapping")
    for k, v in visits.items():
        if type(v) is not int or v < 0:
            raise D1SecondError(
                f"{where}: root_visits[{k!r}] is {v!r} ({type(v).__name__}); a visit "
                f"count must be a non-negative int, and `True` is not one visit.")
    if {DP._move(k) for k in policy} != {DP._move(k) for k in visits}:
        raise D1SecondError(
            f"{where}: raw_policy covers {len(policy)} moves and root_visits "
            f"{len(visits)}; they must be the same legal set")

    counts = {DP._move(k): int(v) for k, v in visits.items()}
    masses = {DP._move(k): float(v) for k, v in policy.items()}
    total = sum(counts.values())
    budget = frozen_sim_budget()
    if total != budget:
        raise D1SecondError(
            f"{where}: root_total_visits is {total}, but the frozen configuration spends "
            f"{budget} simulations. A self-consistent map that totals something else "
            f"describes a search this hypothesis is not about.")
    ours = (pos["incumbent"].get("row"), pos["incumbent"].get("col"))
    if type(ours[0]) is not int or type(ours[1]) is not int or ours not in counts:
        raise D1SecondError(
            f"{where}: the selected move {ours!r} is not a legal move of this root")

    # PRESENCE IS NOT AGREEMENT: each claim is recomputed and compared.
    expect = {
        "n_legal": len(counts),
        "root_total_visits": total,
        "selected_visit_count": counts[ours],
        "selected_visit_rank": rank_visit(visits)[ours],
        "selected_policy_rank": DP.rank_raw(policy)[ours],
    }
    for field, want in expect.items():
        got = inc[field]
        if type(got) is not int or got != want:
            raise D1SecondError(
                f"{where}: {field} is {got!r} but the record's own maps give {want!r}; "
                f"an observable that contradicts what it describes is not evidence.")
    # 🔴 EXACT, NOT APPROXIMATE. These were compared with `math.isclose`, which
    # admitted an altered record: both values are written FROM the very maps the
    # record preserves, so recomputing them reproduces the same float bit for bit
    # and no tolerance is needed. A tolerance nobody froze is a tolerance an
    # edited value can hide inside.
    for field, want in (("selected_policy_mass", masses[ours]),
                        ("root_top1_share", max(counts.values()) / total)):
        got = inc[field]
        if not _real(got) or got != want:
            raise D1SecondError(
                f"{where}: {field} is {got!r} but the record's own maps give {want!r}; "
                f"they are computed from the same maps, so they must agree exactly.")
    if type(inc["readout_overrode_leader"]) is not bool:
        raise D1SecondError(
            f"{where}: readout_overrode_leader is {inc['readout_overrode_leader']!r} "
            f"({type(inc['readout_overrode_leader']).__name__}); the writer writes a bool, "
            f"and a truthy int would be read as one.")

    # 🔴 THE CONTAINER AND EVERY ENTRY FIRST, THEN THE SELECTION. `int(d.get(
    # "depth", -1))` coerced: "6" and 6.0 passed a type-strict contract, and a
    # non-mapping entry escaped as AttributeError instead of a named refusal.
    depths = pos.get("depths")
    if not isinstance(depths, list) or not depths:
        raise D1SecondError(f"{where}: depths is {depths!r}; it must be a non-empty list")
    for j, d in enumerate(depths):
        if not isinstance(d, dict):
            raise D1SecondError(f"{where}: depth record {j} is {d!r}, not a mapping")
        if "depth" not in d:
            raise D1SecondError(f"{where}: depth record {j} carries no 'depth'")
        if type(d["depth"]) is not int:
            raise D1SecondError(
                f"{where}: depth record {j} has depth {d['depth']!r} "
                f"({type(d['depth']).__name__}); a depth is an int, and '6' is not 6.")
    depth6 = [d for d in depths if d["depth"] == 6]
    if len(depth6) != 1:
        raise D1SecondError(f"{where}: expected exactly one depth-6 record, got {len(depth6)}")
    mv = depth6[0].get("move")
    if (not isinstance(mv, (list, tuple)) or len(mv) != 2
            or any(type(x) is not int for x in mv)):
        raise D1SecondError(
            f"{where}: the depth-6 move is {mv!r}; it must be a pair of ints")
    if tuple(mv) not in counts:
        raise D1SecondError(
            f"{where}: the depth-6 move {tuple(mv)} is not a legal move of this root")


def validate_record(d1_report: Mapping[str, Any]) -> Dict[str, int]:
    """EVERY position's observables, checked BEFORE any row is built.

    🔑 A FULL PASS, NOT A PER-ROW CHECK INTERLEAVED WITH THE ARITHMETIC. A record
    that fails at row 200 must fail before row 0 is scored, so a
    partially-computed analysis never exists to be mistaken for a result.

    Type-strict and consistency-checking: coercing a value with `float()` and
    ranking immediately would let a boolean, a negative count, a NaN or a total
    that contradicts its own map reach the metric.
    """
    positions = d1_report.get("positions")
    if not isinstance(positions, list) or not positions:
        raise D1SecondError("the report carries no positions to validate")
    for i, pos in enumerate(positions):
        if not isinstance(pos, dict):
            raise D1SecondError(f"position {i} is not a record")
        _validate_position(pos, f"position {i} ({pos.get('task_id')!r})")
    return {"positions_validated": len(positions),
            "observables_per_position": len(REQUIRED_OBSERVABLES)}



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
        # The two fields the frozen readout summary histograms. Carried onto the
        # row so the summary reads what the RECORD says rather than recomputing
        # it from a different map.
        "selected_visit_rank": inc["selected_visit_rank"],
        "selected_policy_rank": inc["selected_policy_rank"],
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


def _histogram(rows: Sequence[Mapping[str, Any]], field: str) -> List[List[int]]:
    """`[[value, count], ...]` ascending by value, values with no rows omitted.

    🔑 A COUNT HISTOGRAM, NOT A MEAN, and the shape is frozen in the plan before
    any real value was seen. A mean rank cannot distinguish "usually the visit
    leader, occasionally something odd" from "drifts everywhere"; this can.
    """
    counts: Dict[int, int] = {}
    for r in rows:
        v = r[field]
        if type(v) is not int or isinstance(v, bool):
            raise D1SecondError(f"{field} is {v!r} ({type(v).__name__}); a rank is an int")
        counts[v] = counts.get(v, 0) + 1
    return [[v, counts[v]] for v in sorted(counts)]


def _readout_summary(rows: Sequence[Mapping[str, Any]], *, cohort: str) -> Dict[str, Any]:
    """DESCRIPTIVE, per role: the OTHER mechanism -- the readout declining the
    move the search preferred. Deliberately not the primary metric: one frozen
    decision rule, one metric. Nothing here is consulted by the decision.
    """
    out: Dict[str, Any] = {}
    fields = ("overrode_leader", "ss", "selected_visit_rank", "selected_policy_rank",
              "rank_t1j")
    for role in DP.ROLES:
        sel = [r for r in rows if r["signature"] == cohort and r["role"] == role]
        if not sel:
            out[role] = {"n": 0, "overrode_leader": [[False, 0], [True, 0]],
                         "selected_visit_rank": [], "selected_policy_rank": [],
                         "ss_rate": None, "complement_rank_raw_le_k": [0, None]}
            continue
        for field in fields:
            if any(field not in r for r in sel):
                raise D1SecondError(
                    f"a {role} row carries no {field!r}; the readout summary would "
                    f"describe a measurement that is absent")
        n_true = sum(1 for r in sel if bool(r["overrode_leader"]))
        # 🔑 THE COMPLEMENT, REPORTED AND DECIDING NOTHING. The eligibility floors
        # count rows in common-support cells over the UNCONDITIONAL denominator;
        # they never count rows with rank_raw <= K. A small complement is not
        # "insufficient support" -- it MECHANICALLY limits how large ss can be,
        # which is an ordinary NO_GO. The reader needs the number to see that.
        n_elig = sum(1 for r in sel if int(r["rank_t1j"]) <= K_SS)
        out[role] = {
            "n": len(sel),
            "overrode_leader": [[False, len(sel) - n_true], [True, n_true]],
            "selected_visit_rank": _histogram(sel, "selected_visit_rank"),
            "selected_policy_rank": _histogram(sel, "selected_policy_rank"),
            "ss_rate": sum(bool(r["ss"]) for r in sel) / len(sel),
            "complement_rank_raw_le_k": [n_elig, n_elig / len(sel)],
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
    validated = validate_record(d1_report)          # EVERY position, BEFORE any row
    rows = rows_from_d1_report(d1_report)
    check_disjoint(rows)                       # precondition, on the REAL rows
    out = DP._analyse(rows, frozen_cohort=canon["rows"], tasks=canon["tasks"],
                      reps=canon["reps"], kernel=_suppression_decision,
                      B=B_REPLICATES, seed=BOOTSTRAP_SEED)
    out["resolved"] = {k: canon[k] for k in ("record", "plan", "cohort_source", "n_positions")}
    out["acquisition"] = acquisition
    out["prng"] = {"bit_generator": "PCG64", "seed": BOOTSTRAP_SEED, "B": B_REPLICATES}
    out["validated"] = validated
    out["claim"] = CLAIM
    return out
