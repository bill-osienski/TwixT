"""D1' -- the FROZEN prospective analysis of the D1 same-position interrogation.
NOTHING HERE EXECUTES ANYTHING.

Frozen by `docs/superpowers/2026-09-07-t1j-d1prime-diagnostic-proposal.md`
(revision 4, committed 5b1680d). This module reads D1 records (or synthetic
rows shaped like them) and computes: the low-policy-rank disagreement (LPRD)
per position, the matched statistic T over common-support cells, the empirical
stability interval from a stratified whole-game resampling, the eligibility
floors, and the development and confirmation decision rules. It also implements
the confirmation SELECTION order (filter first, then select once) over feature
rows supplied to it.

WHAT IT DOES NOT DO. It loads no model, launches no JVM, reads no seed
registry, opens no confirmation game and runs no D1: it has no gate because it
has nothing to gate. The only randomness is the plan's analysis-only PRNG.
Access to the confirmation half's diagnostics is NOT provided here -- the plan
requires a separately reviewed function for that, and none exists.

THE CLAIM THIS CODE SUPPORTS, and no wider one (plan §4.2):
  T describes matched LPRD in the frozen selected cohort. The stratified cluster
  resampling measures T's sensitivity to reweighting the observed games. Its
  central 95% interval is an EMPIRICAL STABILITY INTERVAL, not an established
  confidence interval for repeated L0 designs. GO means enough effect and
  stability to justify ONE confirmation run -- not proof of a population effect,
  and not that copying T1j's move would improve play.

INTERPRETATIONS RECORDED (the plan's text admits more than one reading; each
choice is named here and asserted by a test):
  * "contributing games" in the eligibility floor = games holding >= 1 row of
    EITHER role in a common-support cell; unmatched rows make no game contribute.
  * the resampling strata are the DESIGN's games (two per opening x arm,
    including games that contributed no selected row), taken from a plan's
    task list, never inferred from the rows.
  * the confirmation ceiling of 240 "positions" is 240 SELECTED ROWS of both
    roles -- 1,200 queries = 240 x 5, one readout and four T1j queries per row.
  * per-arm T in confirmation is T over that arm's common-support cells with
    that arm's own weights; the both-arm rule reads the point value only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .runtime_requalification import _same

Cell = Tuple[str, str, str]                 # (opening, colour_arm, phase)
Move = Tuple[int, int]

# ───────────────────────────── frozen constants ─────────────────────────────

K_LPRD = 5                                   # lprd := rank_raw(t1j_move_6) > 5
T_THRESHOLD = 0.15                           # a judgement threshold, plan §5
B_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260907                    # analysis-only PRNG; in no registry
FLOOR = {"cells": 8, "positions": 40, "controls": 30, "games": 12}   # count rule, not power
CONFIRMATION_CEILING = 240                   # selected rows (both roles) -> 1,200 queries
CONFIRMATION_MIN_PLY = 5                     # rows at ply < 5 ineligible, both roles
PER_CELL_CAP = 3                             # plan §12.1, unchanged
PRIMARY_COHORT = "mover_fragmentation"
SECONDARY_COHORT = "created_threat"
COHORT_COLUMNS = {PRIMARY_COHORT: "mover_more_fragmented", SECONDARY_COHORT: "created_threat"}
ARMS = ("t1j_red", "t1j_black")
ROLES = ("position", "control")

#: The fields that bind an analysis row to its frozen manifest row, type-strictly.
COHORT_BINDING_FIELDS = ("task_id", "ply", "digest", "signature", "role",
                         "opening", "colour_arm", "phase")


class D1PrimeError(Exception):
    """A refusal by the analysis. Never a statement about the engines."""


class D1PrimeRefused(D1PrimeError):
    """A frozen ceiling or precondition refused the step before anything ran."""


# ───────────────────────────── LPRD per position ────────────────────────────

def _move(key: Any) -> Move:
    if isinstance(key, str):
        r, c = key.split(",")
        return (int(r), int(c))
    return (int(key[0]), int(key[1]))


def rank_raw(policy: Mapping[Any, float]) -> Dict[Move, int]:
    """Rank every legal move by raw-policy mass, 1 = leader.

    TIES BROKEN BY THE CANONICAL MOVE ORDER (row, col) -- and by nothing else.
    This function takes the policy and no other input, so visits cannot enter.
    """
    masses = {_move(k): float(v) for k, v in policy.items()}
    order = sorted(masses, key=lambda m: (-masses[m], m[0], m[1]))
    return {m: i + 1 for i, m in enumerate(order)}


def position_row(pos: Mapping[str, Any]) -> Dict[str, Any]:
    """One D1 per-position record -> one analysis row.

    `lprd` and `agree` are computed SEPARATELY; neither logically implies the
    other. Our readout may select a move the raw policy ranks below fifth and
    T1j may select that same move (lprd True, agree True).
    """
    inc = pos["incumbent"]
    depth6 = [d for d in pos["depths"] if int(d["depth"]) == 6]
    if len(depth6) != 1:
        raise D1PrimeError(f"{pos.get('task_id')}: expected exactly one mdPly-6 record, "
                           f"got {len(depth6)}")
    t1j = _move(depth6[0]["move"])
    ours = (int(inc["row"]), int(inc["col"]))
    ranks = rank_raw(inc["raw_policy"])
    if t1j not in ranks:
        raise D1PrimeError(f"{pos.get('task_id')}: T1j's move {t1j} is not in the raw policy "
                           f"({len(ranks)} legal moves); the position or the record is wrong")
    visits = {_move(k): float(v) for k, v in (inc.get("root_visits") or {}).items()}
    total = sum(visits.values())
    row = {k: pos.get(k) for k in COHORT_BINDING_FIELDS}
    row.update({
        "t1j_move_6": t1j, "our_move": ours,
        "rank_t1j": ranks[t1j],
        "lprd": ranks[t1j] > K_LPRD,
        "agree": ours == t1j,
        "rank_ours": ranks.get(ours),
        "visit_share_t1j": (visits.get(t1j, 0.0) / total) if total else None,
        "overrode_leader": inc.get("overrode_leader"),
    })
    return row


def rows_from_d1_report(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [position_row(p) for p in report["positions"]]


# ─────────────────────── the matched statistic over common support ──────────

def _cell(r: Mapping[str, Any]) -> Cell:
    return (r["opening"], r["colour_arm"], r["phase"])


def matched_statistic(rows: Iterable[Mapping[str, Any]], *, cohort: str,
                      arm: Optional[str] = None) -> Dict[str, Any]:
    """T over the cohort's common-support cells, Mantel-Haenszel weighted.

    A cell is in common support iff BOTH roles are present. Weights
    w_c = n_pos * n_ctl / (n_pos + n_ctl) come from the cohort's OWN counts.
    Rows in cells without both roles are RETAINED and reported descriptively;
    they enter no statistic. T is None when no cell has both roles.
    """
    per_cell: Dict[Cell, Dict[str, List[bool]]] = defaultdict(lambda: {"position": [], "control": []})
    games_by_cell: Dict[Cell, set] = defaultdict(set)
    for r in rows:
        if r["signature"] != cohort:
            continue
        if arm is not None and r["colour_arm"] != arm:
            continue
        per_cell[_cell(r)][r["role"]].append(bool(r["lprd"]))
        games_by_cell[_cell(r)].add(r["task_id"])

    cells_out: List[Dict[str, Any]] = []
    num = den = 0.0
    n_pos = n_ctl = 0
    games_cs: set = set()
    unmatched_pos: List[bool] = []
    unmatched_ctl: List[bool] = []
    for cell in sorted(per_cell):
        pos, ctl = per_cell[cell]["position"], per_cell[cell]["control"]
        if pos and ctl:
            w = len(pos) * len(ctl) / (len(pos) + len(ctl))
            diff = sum(pos) / len(pos) - sum(ctl) / len(ctl)
            cells_out.append({"cell": cell, "n_positions": len(pos), "n_controls": len(ctl),
                              "lprd_positions": sum(pos) / len(pos),
                              "lprd_controls": sum(ctl) / len(ctl),
                              "difference": diff, "weight": w})
            num += w * diff
            den += w
            n_pos += len(pos)
            n_ctl += len(ctl)
            games_cs |= games_by_cell[cell]
        else:
            unmatched_pos += pos
            unmatched_ctl += ctl
    return {
        "cohort": cohort, "arm": arm,
        "T": (num / den) if den else None,
        "cells": cells_out, "n_cs_cells": len(cells_out),
        "n_positions_cs": n_pos, "n_controls_cs": n_ctl,
        "games_cs": len(games_cs),
        "unmatched": {"n_positions": len(unmatched_pos), "n_controls": len(unmatched_ctl),
                      "lprd_rate": (sum(unmatched_pos) / len(unmatched_pos)) if unmatched_pos else None},
    }


def floors_met(stat: Mapping[str, Any]) -> Tuple[bool, Dict[str, int]]:
    """The eligibility floor: a COUNT rule, not power. Accepts AT the floor."""
    detail = {"cells": stat["n_cs_cells"], "positions": stat["n_positions_cs"],
              "controls": stat["n_controls_cs"], "games": stat["games_cs"]}
    return all(detail[k] >= FLOOR[k] for k in FLOOR), detail


# ─────────────── stratified whole-game resampling, frozen algorithm ─────────

def design_strata(tasks: Sequence[Mapping[str, Any]], *, reps: Sequence[int],
                  openings: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """The 16 opening x colour-arm strata of a design half, in FROZEN order
    (opening order, then `ARMS`), each holding exactly the design's games for
    `reps` -- including games that contributed no selected row."""
    if openings is None:
        seen: List[str] = []
        for t in tasks:
            if t["opening"] not in seen:
                seen.append(t["opening"])
        openings = seen
    out = []
    for o in openings:
        for a in ARMS:
            games = [t["task_id"] for t in tasks
                     if t["opening"] == o and t["colour_arm"] == a and int(t["rep"]) in set(reps)]
            games = sorted(games, key=lambda g: [int(t["rep"]) for t in tasks if t["task_id"] == g][0])
            if len(games) != 2:
                raise D1PrimeError(f"stratum {(o, a)} holds {len(games)} games; the design has "
                                   f"exactly 2 per stratum in a half")
            out.append({"key": (o, a), "games": games})
    return out


def resample_once(rows: Sequence[Mapping[str, Any]], strata: Sequence[Mapping[str, Any]],
                  rng: np.random.Generator) -> Dict[str, Any]:
    """ONE replicate: per stratum, draw 2 games with replacement from its 2
    (`rng.integers(0, 2, size=2)`, strata in frozen order); every drawn game
    brings ALL its rows -- both roles, all phases, both cohorts -- and a game
    drawn twice contributes them TWICE. No re-dedup, no re-cap, no re-selection."""
    by_game: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_game[r["task_id"]].append(r)
    games: List[str] = []
    for s in strata:
        i, j = rng.integers(0, 2, size=2).tolist()
        games += [s["games"][i], s["games"][j]]
    rep_rows: List[Mapping[str, Any]] = []
    for g in games:
        rep_rows += by_game.get(g, [])
    return {"games": games, "rows": rep_rows}


def replicates(rows: Sequence[Mapping[str, Any]], strata: Sequence[Mapping[str, Any]], *,
               B: int = B_REPLICATES, seed: int = BOOTSTRAP_SEED) -> List[Dict[str, Any]]:
    """B replicates in sequence from ONE Generator (PCG64, `default_rng(seed)`)."""
    rng = np.random.default_rng(seed)
    if type(rng.bit_generator).__name__ != "PCG64":
        raise D1PrimeError("the plan pins PCG64")
    return [resample_once(rows, strata, rng) for _ in range(B)]


def central_interval(values: Sequence[float]) -> Tuple[float, float]:
    """The empirical 2.5% / 97.5% quantiles, `numpy.quantile(method="linear")`
    (Hyndman-Fan type 7). No bias correction, no acceleration."""
    lo, hi = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975], method="linear")
    return (float(lo), float(hi))


def stability_interval(rows: Sequence[Mapping[str, Any]], strata: Sequence[Mapping[str, Any]], *,
                       cohort: str, B: int = B_REPLICATES, seed: int = BOOTSTRAP_SEED,
                       arm: Optional[str] = None) -> Dict[str, Any]:
    """T on the cohort and its EMPIRICAL STABILITY INTERVAL across B whole-game
    reweightings. A replicate whose T is undefined (no cell with both roles) is
    COUNTED and KEPT; it is never dropped, redrawn or imputed, and if any exists
    the interval is UNDEFINED."""
    full = matched_statistic(rows, cohort=cohort, arm=arm)
    reps = replicates(rows, strata, B=B, seed=seed)
    values: List[Optional[float]] = []
    defined_cells: Counter = Counter()
    for rep in reps:
        st = matched_statistic(rep["rows"], cohort=cohort, arm=arm)
        values.append(st["T"])
        defined_cells[st["n_cs_cells"]] += 1
    undefined = sum(1 for v in values if v is None)
    interval: Any = "UNDEFINED" if undefined else central_interval([v for v in values])
    return {"T": full["T"], "interval": interval, "replicates": values,
            "undefined": undefined, "defined_cells": dict(defined_cells),
            "prng": {"bit_generator": "PCG64", "seed": seed, "B": B},
            "statistic": full}


# ─────────────────────────────── decision rules ─────────────────────────────

def _decide(stab: Mapping[str, Any]) -> str:
    if stab["undefined"]:
        return "NO_GO — bootstrap undefined"
    T, (lo, _hi) = stab["T"], stab["interval"]
    return "GO" if (T >= T_THRESHOLD and lo > 0) else "NO_GO"


def development_decision(rows: Sequence[Mapping[str, Any]], strata: Sequence[Mapping[str, Any]], *,
                         cohort: str = PRIMARY_COHORT, B: int = B_REPLICATES,
                         seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Plan §4.2-§4.3. The eligibility floor is checked BEFORE any replicate is
    drawn; the secondary cohort is reported and cannot produce GO."""
    primary = matched_statistic(rows, cohort=cohort)
    met, detail = floors_met(primary)
    secondary = matched_statistic(rows, cohort=SECONDARY_COHORT if cohort == PRIMARY_COHORT
                                  else PRIMARY_COHORT)
    if not met:
        return {"outcome": "NO_GO — insufficient support", "T": primary["T"], "interval": None,
                "floor": detail, "statistic": primary, "secondary": secondary}
    stab = stability_interval(rows, strata, cohort=cohort, B=B, seed=seed)
    return {"outcome": _decide(stab), "T": stab["T"], "interval": stab["interval"],
            "floor": detail, "statistic": primary, "stability": stab, "secondary": secondary,
            "meaning": ("GO = sufficient effect and resampling stability to justify ONE "
                        "confirmation run; not proof of a population effect")}


def confirmation_decision(rows: Sequence[Mapping[str, Any]], strata: Sequence[Mapping[str, Any]], *,
                          cohort: str = PRIMARY_COHORT, B: int = B_REPLICATES,
                          seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Plan §6.5: the same statistic on a DIFFERENT cohort, plus the both-arm
    rule. An arm with no common-support cell has NO T: the outcome is
    `NO_GO — arm undefined`, never zero, never skipped."""
    dev = development_decision(rows, strata, cohort=cohort, B=B, seed=seed)
    arms = {a: matched_statistic(rows, cohort=cohort, arm=a)["T"] for a in ARMS}
    out = dict(dev, arms=arms)
    if dev["outcome"].startswith("NO_GO"):
        return out
    if any(arms[a] is None for a in ARMS):
        out["outcome"] = "NO_GO — arm undefined"
        return out
    if not all(arms[a] > 0 for a in ARMS):
        out["outcome"] = "NO_GO"
    return out


# ───────────── confirmation selection: filter first, then select once ───────

def _order(r: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(r["task_id"]), int(r["ply"]))


def confirmation_select(candidates: Iterable[Mapping[str, Any]], *, seen_digests: Iterable[str],
                        cap: int = PER_CELL_CAP, ceiling: int = CONFIRMATION_CEILING) -> Dict[str, Any]:
    """Plan §6.2-§6.3 over FEATURE ROWS supplied by the caller (this module
    computes no features and opens no game).

    FILTER FIRST: incumbent to move; ply >= CONFIRMATION_MIN_PLY (both roles);
    within-half dedup by digest, earliest by (task_id, ply); cross-half digests
    (development cohort + the six excluded) removed. THEN SELECT ONCE: per
    signature, positions = column True capped at `cap` per cell, earliest;
    controls = column False in the selected positions' cells, same cap. The
    ceiling refuses -- it never drops.
    """
    seen = set(seen_digests)
    removed = {"not_incumbent_to_move": 0, "ply_lt_5": 0, "within_half_dup": 0,
               "seen_in_development": 0}
    pool = []
    for r in sorted(candidates, key=_order):
        if not r.get("incumbent_to_move", True):
            removed["not_incumbent_to_move"] += 1
            continue
        if int(r["ply"]) < CONFIRMATION_MIN_PLY:
            removed["ply_lt_5"] += 1
            continue
        pool.append(r)
    deduped, have = [], set()
    for r in pool:                                   # earliest (task_id, ply) wins
        if r["digest"] in have:
            removed["within_half_dup"] += 1
            continue
        have.add(r["digest"])
        deduped.append(r)
    eligible = []
    for r in deduped:
        if r["digest"] in seen:
            removed["seen_in_development"] += 1
            continue
        eligible.append(r)

    selected: List[Dict[str, Any]] = []
    counts: Dict[str, Any] = {}
    for sig, col in COHORT_COLUMNS.items():
        per_cell: Dict[Cell, int] = Counter()
        positions = []
        for r in eligible:
            if r.get(col) is True and per_cell[_cell(r)] < cap:
                per_cell[_cell(r)] += 1
                positions.append(dict(r, signature=sig, role="position"))
        cells = {_cell(p) for p in positions}
        per_cell = Counter()
        controls = []
        for r in eligible:
            if r.get(col) is False and _cell(r) in cells and per_cell[_cell(r)] < cap:
                per_cell[_cell(r)] += 1
                controls.append(dict(r, signature=sig, role="control"))
        selected += positions + controls
        counts[sig] = {"positions": len(positions), "controls": len(controls), "cells": len(cells)}
    counts["total"] = len(selected)
    if len(selected) > ceiling:
        raise D1PrimeRefused(
            f"the confirmation selection yields {len(selected)} rows, above the frozen ceiling of "
            f"{ceiling} (1,200 queries). REFUSED before any seed or query; there is no drop rule "
            f"-- the overflow is resolved by a written amendment fixing a stricter cap first.")
    return {"rows": selected, "removed": removed, "counts": counts}


# ──────────────────────────── cohort binding ────────────────────────────────

def check_cohort(rows: Sequence[Mapping[str, Any]], frozen: Sequence[Mapping[str, Any]]) -> None:
    """The analysis refuses any cohort other than the frozen one: same count,
    same order, every binding field TYPE-STRICTLY equal (False != 0, 10 != "10")."""
    if len(rows) != len(frozen):
        raise D1PrimeError(f"{len(rows)} rows, the frozen cohort has {len(frozen)}: not the frozen cohort")
    for i, (got, want) in enumerate(zip(rows, frozen)):
        for k in COHORT_BINDING_FIELDS:
            if k not in got or not _same(got[k], want[k]):
                raise D1PrimeError(f"row {i}: field {k!r} {got.get(k, '<missing>')!r} != frozen "
                                   f"{want[k]!r}: not the frozen cohort")
