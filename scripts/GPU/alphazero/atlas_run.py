"""Atlas composition -- assigned row to authenticated artifact.

Section 3's chronology, executable:

    pilot_geometry_gate(24 games)  -> assignment, 3 per phase x side cell
    run-pilot: 24 rows, ALL discovery, on the full ladder
    size_from_pilot(pilot class counts) -> N
    assign_corpus(...)             -> exactly N - 24 continuation rows
    run-final: the pilot's 24 + the continuation's N - 24 == exactly N

Pure orchestration over an INJECTED evaluator. Imports no MLX and never
constructs one: section 2b requires a single long-lived evaluator, and
rebuilding a compiled one per unit of work is the documented MLX trap, so there
is deliberately no factory or checkpoint parameter to pass.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .atlas_artifact import build_row
from .atlas_labelling import class_counts, classify_row, final_capacity_gate, \
    size_from_pilot
from .atlas_readout_a import collect_features, deployability, \
    evaluate_detector_both
from .atlas_readout_b import GATE_NAMES, by_stratum_summary, \
    natural_convergence_report
from .atlas_readout_c import aggregate_shape, \
    select_on_discovery_validate_on_selected, validation_verdict
from .atlas_row_facts import derive_row_facts
from .corpus_geometry import PILOT_GAMES, assign_corpus, pilot_geometry_gate, \
    select_ply, size_continuation
from .mcts import MCTS, MCTSConfig
from .selection_tracer import WIDENING_SHAPES, SelectionTracer
from .warm_prefix_replay import BOUNDARY_THRESHOLD, BatchSafeBoundaryObserver, \
    LEG_INCREMENTS, capture_tree_state, check_backup_invariant, replay_prefix, \
    replay_seed_for, run_additive_ladder

# Section 2b's frozen ladder regime, named once instead of at each call site.
LADDER_BATCHING = (14, 48, 8)          # eval_batch_size, stall_flush, virtual
PREFIX_SIMS = 400                      # section 2b: "frozen 400-sim searches"
LEG_INCREMENTS_DEFAULT = LEG_INCREMENTS
BOUNDARY_THRESHOLD_DEFAULT = BOUNDARY_THRESHOLD
LEG_B_DEFAULT = 400
ACTIVE_SIZE = 24                       # section 3: the only size played


def ladder_config(n_simulations: int) -> MCTSConfig:
    eb, stall, virt = LADDER_BATCHING
    return MCTSConfig(n_simulations=n_simulations, eval_batch_size=eb,
                      stall_flush_sims=stall, pending_virtual_visits=virt)


@dataclass
class RowOutcome:
    ok: bool
    row: Optional[Dict[str, Any]]
    failure: Optional[str]
    game_id: int


def run_row(evaluator, meta, assigned, *, move_history, base_seed,
            active_size=None, prefix_sims=None, increments=None,
            threshold=None, leg_B=None, _on_row=None,
            _corrupt_d3=False) -> RowOutcome:
    """ONE row: verified seed -> one MCTS -> prefix -> ladder -> facts -> row.

    Every failure path returns a RowOutcome carrying the reason rather than
    raising, so one bad row does not discard the diagnostics already paid for by
    the rows before it. It does NOT make the run survivable: `run_corpus` turns
    any failure into an ABORTED, non-authoritative run.

    The budget parameters default to `None` and resolve against the MODULE
    globals at call time, so a test can lower them by patching the module. A
    signature default would bind at import and could not be patched.
    """
    active_size = ACTIVE_SIZE if active_size is None else active_size
    prefix_sims = PREFIX_SIMS if prefix_sims is None else prefix_sims
    increments = LEG_INCREMENTS_DEFAULT if increments is None else increments
    threshold = BOUNDARY_THRESHOLD_DEFAULT if threshold is None else threshold
    leg_B = LEG_B_DEFAULT if leg_B is None else leg_B

    try:
        seed = replay_seed_for(meta, base_seed)          # verifies the sidecar
    except ValueError as e:
        return RowOutcome(False, None, str(e), meta.game_id)

    # ONE MCTS per row, carrying ITS OWN frozen stream, continued across the
    # prefix and all four legs and never reseeded (section 2b). The evaluator is
    # the caller's and is never rebuilt. The prefix runs at prefix_sims; the
    # ladder overrides n_simulations per leg and restores it in a finally.
    mcts = MCTS(evaluator, ladder_config(prefix_sims), random.Random(seed))
    if _on_row is not None:
        _on_row(mcts)

    try:
        pre = replay_prefix(mcts, meta, move_history, assigned["ply"],
                            active_size)
        tracer = SelectionTracer()
        mcts._selection_observer = tracer
        obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I,
                                        threshold=threshold, leg_B=leg_B,
                                        tracer=tracer)
        legs, snaps = run_additive_ladder(mcts, pre.root, pre.inherited_I,
                                          ply=assigned["ply"],
                                          boundary_observer=obs,
                                          target_tracer=tracer,
                                          increments=increments)
    except (ValueError, AssertionError) as e:
        return RowOutcome(False, None, f"replay/ladder failed: {e}",
                          meta.game_id)

    if obs.record is None:
        # A missing boundary is a missing measurement, not an N_actual of zero.
        return RowOutcome(False, None,
                          "no boundary flush was captured during leg 1",
                          meta.game_id)

    caps = snaps["captures"]
    start, boundary = caps["at_start"], caps["at_boundary"]
    if _corrupt_d3:                                   # test-only fault injector
        boundary = {**boundary, "D3": start["D3"] - 1}
    try:
        check_backup_invariant(start["D3"], boundary["D3"], obs.record.N_actual)
        features_b = collect_features(start, boundary, obs.record.N_actual)
        features_4 = collect_features(start, caps["at_400"], leg_B)
    except ValueError as e:
        return RowOutcome(False, None, str(e), meta.game_id)

    try:
        facts = derive_row_facts(legs, snaps, assigned["ply"],
                                 meta.start_player,
                                 assigned_phase=assigned["phase"],
                                 assigned_side=assigned["side"])
    except ValueError as e:
        return RowOutcome(False, None, str(e), meta.game_id)

    row = build_row(
        game_idx=meta.game_id, replay_seed=seed, target_ply=assigned["ply"],
        phase=facts["phase"], side=facts["side"], split=assigned["split"],
        inherited_I=pre.inherited_I, reset_count=pre.reset_count,
        reset_rate=pre.reset_rate, last_reset_ply=pre.last_reset_ply,
        boundary=obs.record, legs=legs, label=classify_row(legs),
        features_at_boundary=features_b, features_at_400=features_4,
        snapshots=snaps, flat_policy=facts["flat_policy"],
        near_even=facts["near_even"])
    row["row_facts_undefined"] = facts["undefined"]
    return RowOutcome(True, row, None, meta.game_id)


def _readouts(rows: Sequence[Dict[str, Any]], complete: bool) -> Dict[str, Any]:
    """All three read-outs over `rows`, each marked with whether the corpus
    they describe was measured completely."""
    discovery = [r for r in rows if r["split"] == "discovery"]
    validation = [r for r in rows if r["split"] == "validation"]
    counts = class_counts([r["legs"] for r in rows])
    remaining = deployability([r["boundary"].remaining for r in rows
                               if r["boundary"] is not None])
    return {
        "class_counts": counts,
        "capacity": final_capacity_gate(
            class_counts([r["legs"] for r in validation])),
        "deployability": remaining,
        "readout_a": evaluate_detector_both(discovery, validation),
        "readout_b": {g: by_stratum_summary(rows, g) for g in GATE_NAMES},
        "natural_convergence": natural_convergence_report(rows),
        "readout_c": select_on_discovery_validate_on_selected(discovery,
                                                              validation),
        "readout_a_authoritative": complete,
        "readout_b_authoritative": complete,
        "readout_c_authoritative": complete,
    }


def run_corpus(evaluator, metas: Sequence[Any],
               assigned_rows: Sequence[Dict[str, Any]], *, base_seed: int,
               move_histories: Dict[int, Sequence[Tuple[int, int]]],
               provenance: Dict[str, Any], **kw) -> Dict[str, Any]:
    """Every assigned row through the full chain, then all three read-outs.

    `evaluator` is built ONCE by the caller and shared by every row. There is
    deliberately no factory or checkpoint parameter: section 2b requires one
    long-lived evaluator, and rebuilding a compiled one per unit of work is the
    documented MLX trap, so the trap is unreachable rather than discouraged.

    COMPLETENESS: the corpus is exactly the assigned positions, so ANY row
    failure makes the run ABORTED and non-authoritative. The read-outs still
    run -- a half-measured corpus is worth diagnosing -- but nothing computed
    from a partial corpus may be called authoritative. This counts assigned
    against measured and introduces no number, so there is no failure-tolerance
    knob and nothing to tune.
    """
    by_id = {g.game_id: g for g in metas}
    measured: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for assigned in assigned_rows:
        gid = assigned["game_id"]
        meta = by_id.get(gid)
        if meta is None:
            failures.append({"game_id": gid,
                             "failure": "no GameMeta for the assigned game"})
            continue
        out = run_row(evaluator, meta, assigned,
                      move_history=move_histories.get(gid, ()),
                      base_seed=base_seed, **kw)
        if out.ok:
            measured.append(out.row)
        else:
            failures.append({"game_id": gid, "failure": out.failure})

    complete = not failures
    doc = {
        "verdict": "OK" if complete else "ABORTED",
        "authoritative": complete,
        "assigned": len(assigned_rows), "measured": len(measured),
        "rows": measured, "failed_rows": failures,
        "provenance": provenance,
        "splits": {s: sum(1 for r in measured if r["split"] == s)
                   for s in sorted({r["split"] for r in measured})},
        "row_facts_undefined": sum(len(r.get("row_facts_undefined", ()))
                                   for r in measured),
    }
    doc.update(_readouts(measured, complete))
    return doc


# ---------------------------------------------------------------------------
# Section 3's two modes
# ---------------------------------------------------------------------------

def pilot_rows(pilot_games: Sequence[Any],
               sampling_seed: int) -> List[Dict[str, Any]]:
    """The 24 FIXED pilot rows, in `assign_corpus`'s row shape.

    Fails closed on PHASE_GEOMETRY_NO_GO rather than quietly running a smaller
    pilot: the gate exists to stop before the pilot ladder is paid for, and a
    downgraded pilot would silently change the sizing denominator.
    """
    gate = pilot_geometry_gate(pilot_games, sampling_seed)
    if gate["verdict"] != "PASS":
        raise ValueError(f"PHASE_GEOMETRY_NO_GO: {gate['unmet']}")
    by_id = {g.game_id: g for g in pilot_games}
    rows = []
    for gid in sorted(gate["assignment"]):
        split, phase, side = gate["assignment"][gid]      # split is discovery
        rows.append({"game_id": gid, "seed": by_id[gid].seed, "split": split,
                     "phase": phase, "side": side,
                     "ply": select_ply(by_id[gid], split, phase, side,
                                       sampling_seed)})
    return rows


def _early_widening_check(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 8's early static check, on the pilot only.

    Each shape gets the FROZEN three-way precedence, because a `None` retention
    rate means the rate is UNDEFINED -- not that the shape failed. Scoring an
    undefined rate as a failure would let a sparse pilot close progressive
    widening on an ABSENCE of evidence, which is the exact mistake
    `validation_verdict`'s precedence exists to prevent.

    `both_fail` therefore fires only on two genuine FAILs. Reported to the
    operator; never acted on automatically.
    """
    per = {s[0]: validation_verdict(aggregate_shape(rows, s))
           for s in WIDENING_SHAPES}
    return {**per,
            "both_fail": all(v["verdict"] == "FAIL" for v in per.values())}


def run_pilot(evaluator, pilot_games, *, sampling_seed, base_seed,
              move_histories, provenance, **kw) -> Dict[str, Any]:
    """Section 3's pilot: 24 rows, ALL discovery, on the full ladder.

    N is this function's OUTPUT, not its input.
    """
    doc = run_corpus(evaluator, pilot_games,
                     pilot_rows(pilot_games, sampling_seed),
                     base_seed=base_seed, move_histories=move_histories,
                     provenance=provenance, **kw)
    doc["mode"] = "pilot"
    # The ONLY assignment input the artifact carries. `pilot_games` and the
    # gate's assignment are deliberately NOT stored: `emit` would flatten
    # GameMeta objects to dicts and `load_run` does not rehydrate them, so
    # run-final re-derives both from the verified pilot block plus this seed.
    doc["sampling_seed"] = sampling_seed
    if not doc["authoritative"]:
        # An incomplete pilot must not size: N would come from a class
        # frequency measured over fewer than 24 rows, and closing progressive
        # widening on it would end a read-out on evidence never gathered.
        # Partial diagnostics are preserved; the conclusions are not offered.
        doc["sizing"] = {"verdict": "UNAVAILABLE", "N": None,
                         "reason": "the pilot did not measure all 24 assigned "
                                   "positions; sizing needs a complete pilot"}
        doc["early_widening_check"] = _early_widening_check(doc["rows"])
        doc["early_widening_check_authoritative"] = False
        return doc
    doc["sizing"] = size_from_pilot(doc["class_counts"])
    doc["early_widening_check"] = _early_widening_check(doc["rows"])
    doc["early_widening_check_authoritative"] = True
    return doc


# An ARTIFACT row and an ASSIGNMENT row name the same facts differently:
# build_row stores game_idx / replay_seed / target_ply, while assign_corpus
# emits game_id / seed / ply. Mapping them explicitly is the difference between
# a comparison and a KeyError.
_AS_ASSIGNED = {"game_idx": "game_id", "replay_seed": "seed",
                "target_ply": "ply", "split": "split", "phase": "phase",
                "side": "side"}


def verify_pilot(pilot_doc, pilot_games) -> Dict[int, Tuple[str, str, str]]:
    """Recompute and cross-check everything the pilot claims.

    `pilot_games` comes from the VERIFIED pilot block, not from the artifact:
    `emit` would have flattened `GameMeta` objects to dicts and `load_run` does
    not rehydrate them, so the artifact stores only `sampling_seed` and the
    measured rows. The geometry gate, the assignment and the sizing are all
    re-derived here from the block plus that one seed.

    Returns the recomputed pilot assignment, which the continuation assignment
    needs as its input.
    """
    seed = pilot_doc["sampling_seed"]
    gate = pilot_geometry_gate(pilot_games, seed)
    if gate["verdict"] != "PASS":
        raise ValueError(f"PHASE_GEOMETRY_NO_GO on the pilot block: "
                         f"{gate['unmet']}")
    expected = pilot_rows(pilot_games, seed)
    measured = [{dst: r[src] for src, dst in _AS_ASSIGNED.items()}
                for r in pilot_doc["rows"]]
    if measured != [{k: e[k] for k in _AS_ASSIGNED.values()} for e in expected]:
        raise ValueError("the pilot artifact's rows do not match a recomputed "
                         "pilot assignment; it is stale, edited or mis-seeded")

    # Sizing is re-derived from the CARRIED rows, not trusted. A stored N that
    # its own class counts do not produce is the difference between "N=200" and
    # "N=200 because 8 of 24 were misleading".
    counts = class_counts([r["legs"] for r in pilot_doc["rows"]])
    if size_from_pilot(counts) != pilot_doc["sizing"]:
        raise ValueError(
            f"stored sizing {pilot_doc['sizing']} is not what the carried "
            f"pilot rows produce ({size_from_pilot(counts)})")

    # The STORED label and row facts are re-derived too. Sizing reads the legs,
    # but Read-out A takes its classes and Read-out C its intervention
    # denominators from `label`, and both strata sets from `flat_policy` /
    # `near_even` -- so an edited label silently moves a row between classes
    # while every leg-derived check still passes.
    by_id = {g.game_id: g for g in pilot_games}
    for row in pilot_doc["rows"]:
        if row["label"] != classify_row(row["legs"]):
            raise ValueError(
                f"row {row['game_idx']}: stored label {row['label']!r} is not "
                f"what its own legs classify as ({classify_row(row['legs'])!r})")
        facts = derive_row_facts(row["legs"], row["snapshots"],
                                 row["target_ply"],
                                 by_id[row["game_idx"]].start_player)
        for field in ("phase", "side", "flat_policy", "near_even"):
            if row[field] != facts[field]:
                raise ValueError(
                    f"row {row['game_idx']}: stored {field} {row[field]!r} != "
                    f"re-derived {facts[field]!r}")
    return gate["assignment"]


def verify_assignment(pilot_games, pilot_assignment, sampling_seed, n_target,
                      continuation_games, assignment_rows) -> None:
    """Assignment is DETERMINISTIC, so recompute it rather than trusting the
    artifact. A hand-edited, stale or mis-seeded file would otherwise become
    the corpus.

    `continuation_games` is the COMPLETE authorized block -- `G_total - 24`,
    which is larger than the `N - 24` rows selected from it. Passing the
    selected rows here instead would make any selection look correct.
    """
    sizing = size_continuation(pilot_games, n_target)
    if sizing.get("verdict") != "OK":
        raise ValueError(f"continuation sizing is {sizing.get('verdict')!r}, "
                         f"so there is no G_total to check against")
    if len(continuation_games) != sizing["G_total"] - PILOT_GAMES:
        raise ValueError(
            f"continuation block holds {len(continuation_games)} games but the "
            f"frozen sizing requires {sizing['G_total'] - PILOT_GAMES}")
    recomputed = assign_corpus(pilot_assignment, continuation_games,
                               n_target, sampling_seed)
    if recomputed["verdict"] != "OK" or recomputed["rows"] != assignment_rows:
        raise ValueError("the assignment artifact does not match a recomputed "
                         "assignment; it is stale, edited, or mis-seeded")


def combine_final_runs(pilot_doc, continuation_doc, *,
                       provenance) -> Dict[str, Any]:
    """PURE. Carry the pilot rows, recompose all three read-outs over the
    combined corpus, and inherit the completeness condition from both halves.

    Separated from `run_final` on purpose: the expensive part is the ladders,
    and the part most likely to be wrong is this one. Splitting them lets the
    SUCCESSFUL final composition be qualified on CPU with a synthetic complete
    continuation result -- at the real frozen N, with no budget override and no
    ladder -- so the first production run stays evidence rather than becoming a
    disposable qualification run.
    """
    rows = list(pilot_doc["rows"]) + list(continuation_doc["rows"])
    complete = bool(pilot_doc["authoritative"]
                    and continuation_doc["authoritative"])
    failures = (list(pilot_doc.get("failed_rows", ()))
                + list(continuation_doc.get("failed_rows", ())))
    n_target = pilot_doc["sizing"]["N"]
    doc = {
        "mode": "final",
        "verdict": "OK" if complete else "ABORTED",
        "authoritative": complete,
        "n_target": n_target,
        "assigned": n_target,
        "measured": len(rows),
        "pilot_rows_carried": len(pilot_doc["rows"]),
        "rows": rows, "failed_rows": failures,
        "provenance": provenance,
        "sampling_seed": pilot_doc["sampling_seed"],
        "splits": {s: sum(1 for r in rows if r["split"] == s)
                   for s in sorted({r["split"] for r in rows})},
        "row_facts_undefined": sum(len(r.get("row_facts_undefined", ()))
                                   for r in rows),
    }
    doc.update(_readouts(rows, complete))
    return doc


def run_final(evaluator, *, pilot_doc, pilot_games, continuation_games,
              assignment_rows, base_seed, move_histories, provenance,
              **kw) -> Dict[str, Any]:
    """The pilot's 24 discovery rows PLUS the continuation's N-24.

    The contract, in order:

        1. load pilot games from the VERIFIED pilot block
        2. recompute and cross-check the pilot assignment AND its sizing
        3. derive the selected continuation rows from the VERIFIED complete
           continuation block, by recomputing the assignment
        4. measure exactly those assigned rows

    N comes from `pilot_doc["sizing"]` and from NOWHERE else -- there is no
    `n_target` parameter, so an invented or out-of-set value cannot be supplied.
    """
    if pilot_doc.get("verdict") != "OK" or not pilot_doc.get("authoritative"):
        raise ValueError("refusing to run: the pilot artifact is not an "
                         "authoritative, complete pilot")
    sizing = pilot_doc.get("sizing") or {}
    if sizing.get("verdict") != "OK" or sizing.get("N") is None:
        raise ValueError(f"refusing to run: pilot sizing is "
                         f"{sizing.get('verdict')!r}, so there is no N")
    carried = pilot_doc["rows"]
    if len(carried) != PILOT_GAMES:
        raise ValueError(f"pilot artifact holds {len(carried)} rows, not the "
                         f"fixed {PILOT_GAMES}")
    n_target = sizing["N"]
    if len(carried) + len(assignment_rows) != n_target:
        raise ValueError(
            f"corpus must contain exactly N={n_target} positions: "
            f"{len(carried)} pilot + {len(assignment_rows)} continuation")

    pilot_assignment = verify_pilot(pilot_doc, pilot_games)
    verify_assignment(pilot_games, pilot_assignment,
                      pilot_doc["sampling_seed"], n_target,
                      continuation_games, assignment_rows)
    # Measure exactly the assigned rows, whose metadata comes from the VERIFIED
    # continuation block rather than from the assignment file.
    by_id = {g.game_id: g for g in continuation_games}
    metas = [by_id[r["game_id"]] for r in assignment_rows]
    cont = run_corpus(evaluator, metas, assignment_rows, base_seed=base_seed,
                      move_histories=move_histories, provenance=provenance,
                      **kw)
    return combine_final_runs(pilot_doc, cont, provenance=provenance)
