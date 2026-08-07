"""Frozen preflight for the Candidate 2 readout rule.

Descriptive analysis of already-captured replays. The rule and every gate
below are FROZEN (design spec section 7.4, 2026-08-06) and MUST NOT be revised
in response to what this reports.

Population (frozen): post-opening turns belonging to the named agent only.
All such turns are in the denominator; an ineligible turn counts as
"no override", it does NOT disappear.

The stored `readout_overrode_leader` flag is deliberately NOT compared against
the recomputed rule. On a Candidate 1 (argmax) agent the stored flag is always
False by construction while the recomputed rule may fire, so a mismatch is
EXPECTED there and would be a meaningless alarm. The frozen rule is the
authority for these statistics; the flag records what was actually played.

NO GPU RUN IS AUTHORIZED BY THIS TOOL, and no Candidate telemetry exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys

from .eval_readout import MIN_CHILD_VISITS, ChildStat, lcb_override

REQUIRED_SCHEMA_VERSION = 2

# --- FROZEN gates -----------------------------------------------------------
OVERRIDE_RATE_FLOOR = 0.005     # below: not enough reach to justify the spend
OVERRIDE_RATE_CEILING = 0.15    # above: no longer a conservative occasional rule
SINGLE_GAME_SHARE_CEILING = 0.50


def _stat_from_dict(d):
    return ChildStat(
        move=(d["row"], d["col"]),
        visits=d["completed_visit_count"],
        q_child=d["q_value_child_perspective"],
        q_root=d["q_value_root_perspective"],
    )


def _agent_colour(replay, agent_id):
    """Which colour the named agent played in this game, or None."""
    if replay.get("red_agent_id") == agent_id:
        return "red"
    if replay.get("black_agent_id") == agent_id:
        return "black"
    return None


def _ply_bucket(ply):
    """Coarse phase label for descriptive reporting only."""
    if ply < 40:
        return "20-39"
    if ply < 70:
        return "40-69"
    if ply < 110:
        return "70-109"
    return "110+"


def preflight_stats(replays, agent_id, opening_temp_plies):
    """Compute the frozen population statistics over loaded replay dicts.

    Fails closed: a wrong schema, or ANY selected replay lacking the agent
    under analysis, is an error rather than a silent skip.
    """
    population = 0
    eligible = 0
    overrides = 0
    undefined_q = 0
    per_game = {}
    colour_counts = {"red": 0, "black": 0}
    by_bucket = {}
    challenger_visits_at_override = []
    matched_replays = 0
    missing_agent = []

    for replay in replays:
        if replay.get("schema_version") != REQUIRED_SCHEMA_VERSION:
            raise ValueError(
                f"game {replay.get('game_idx')}: schema_version "
                f"{replay.get('schema_version')!r}, need {REQUIRED_SCHEMA_VERSION}; "
                f"top-two telemetry is absent and cannot be inferred")
        colour = _agent_colour(replay, agent_id)
        if colour is None:
            # Collected, then raised on below. A replay selected for this
            # analysis that does not contain the named agent is an identity
            # fault, not something to skip: scoring the remainder would report
            # a clean-looking result over a population that silently lost
            # games.
            missing_agent.append(replay.get("game_idx"))
            continue
        matched_replays += 1
        gid = replay.get("game_idx")
        for rec in replay.get("moves", []):
            if rec["ply"] < opening_temp_plies or rec["player"] != colour:
                continue
            population += 1
            bucket = _ply_bucket(rec["ply"])
            slot = by_bucket.setdefault(bucket, {"plies": 0, "overrides": 0})
            slot["plies"] += 1

            # FAIL CLOSED on telemetry. B2 defines top2=None as "NOT
            # CAPTURED", so treating it as "no override" would silently lower
            # the override rate and could close Candidate 2 on absent data --
            # the floor gate is the one this would push us toward.
            n_legal = rec.get("n_legal")
            if not isinstance(n_legal, int) or n_legal < 1:
                raise ValueError(
                    f"game {gid} ply {rec['ply']}: n_legal is {n_legal!r}; "
                    f"cannot verify top-two telemetry completeness")
            top2_raw = rec.get("top2")
            if not isinstance(top2_raw, list):
                raise ValueError(
                    f"game {gid} ply {rec['ply']}: top2 is {top2_raw!r}, not a "
                    f"list; replay schema 2 records None only when telemetry "
                    f"was NOT CAPTURED, and absent telemetry must never be "
                    f"scored as 'no override'")
            expected = min(2, n_legal)
            if len(top2_raw) != expected:
                raise ValueError(
                    f"game {gid} ply {rec['ply']}: top2 has {len(top2_raw)} "
                    f"entries, expected {expected} for n_legal={n_legal}")
            if len(top2_raw) < 2:
                # Exactly one legal move: the rule needs a challenger, so this
                # turn is LEGITIMATELY ineligible. It stays in the denominator.
                continue
            top2 = [_stat_from_dict(d) for d in top2_raw]
            # A None mean on a VISITED child, or any non-finite value, is
            # corrupt telemetry -- not an undefined mean.
            corrupt = any(
                s.visits > 0 and (
                    s.q_root is None or s.q_child is None
                    or not math.isfinite(s.q_root)
                    or not math.isfinite(s.q_child))
                for s in top2)
            if corrupt:
                undefined_q += 1
                continue
            if any(s.q_root is None for s in top2):
                continue    # undefined mean on an unvisited child: ineligible
            if all(s.visits >= MIN_CHILD_VISITS for s in top2):
                eligible += 1
            # The frozen rule is the authority; the stored flag is not trusted.
            if lcb_override(top2) is not None:
                overrides += 1
                slot["overrides"] += 1
                per_game[gid] = per_game.get(gid, 0) + 1
                colour_counts[colour] += 1
                challenger_visits_at_override.append(top2[1].visits)

    if missing_agent:
        raise ValueError(
            f"agent {agent_id!r} is absent from {len(missing_agent)} of "
            f"{len(replays)} selected replays (game_idx "
            f"{sorted(x for x in missing_agent if x is not None)[:20]}); "
            f"every selected replay must contain the agent under analysis")

    max_share = (max(per_game.values()) / overrides) if overrides else None
    total_colour = colour_counts["red"] + colour_counts["black"]
    colour_split = (
        {k: v / total_colour for k, v in colour_counts.items()}
        if total_colour else None
    )
    cv = sorted(challenger_visits_at_override)
    return {
        "agent_id": agent_id,
        "replays_total": len(replays),
        "replays_matched": matched_replays,
        "population_plies": population,
        "eligible_plies": eligible,
        "overrides": overrides,
        "override_rate": (overrides / population) if population else None,
        "undefined_q_plies": undefined_q,
        "max_single_game_share": max_share,
        "games_with_overrides": len(per_game),
        # --- DESCRIPTIVE ONLY, frozen as non-gating (spec section 7.4) ------
        "colour_split": colour_split,
        "override_rate_by_ply_bucket": {
            b: {**v, "rate": (v["overrides"] / v["plies"]) if v["plies"] else None}
            for b, v in sorted(by_bucket.items())
        },
        "challenger_visits_at_override": {
            "n": len(cv),
            "min": cv[0] if cv else None,
            "median": cv[len(cv) // 2] if cv else None,
            "max": cv[-1] if cv else None,
        },
        "per_game_override_counts": dict(sorted(per_game.items())),
    }


def evaluate_gates(stats):
    """Apply the FROZEN stop rules. Returns {passed, failed_gates, thresholds}.

    Comparisons are strict, so a statistic exactly at a threshold is inside
    the band.
    """
    failed = []
    rate = stats.get("override_rate")
    if rate is None:
        failed.append("empty_population")
    else:
        if rate < OVERRIDE_RATE_FLOOR:
            failed.append("override_rate_floor")
        if rate > OVERRIDE_RATE_CEILING:
            failed.append("override_rate_ceiling")
    share = stats.get("max_single_game_share")
    if share is not None and share > SINGLE_GAME_SHARE_CEILING:
        failed.append("single_game_concentration")
    if stats.get("undefined_q_plies"):
        failed.append("undefined_q")
    return {
        "passed": not failed,
        "failed_gates": failed,
        "thresholds": {
            "override_rate_floor": OVERRIDE_RATE_FLOOR,
            "override_rate_ceiling": OVERRIDE_RATE_CEILING,
            "single_game_share_ceiling": SINGLE_GAME_SHARE_CEILING,
        },
    }


def nonleader_selection_report(replays, agent_id, opening_temp_plies):
    """Non-leader selection rate before and after the opening boundary.

    Required by spec section 7.3 so a Candidate 1 null can be attributed:
    all-ply argmax changes BOTH the opening and post-opening play, and this
    split says which half moved. Purely DESCRIPTIVE -- it gates nothing.

    Uses `selected_visit_rank`, which ply_record already emits; rank > 1 means
    the played move was not the visit leader.
    """
    buckets = {"opening": {"plies": 0, "nonleader": 0},
               "post_opening": {"plies": 0, "nonleader": 0}}
    for replay in replays:
        colour = _agent_colour(replay, agent_id)
        if colour is None:
            continue
        for rec in replay.get("moves", []):
            if rec["player"] != colour:
                continue
            key = "opening" if rec["ply"] < opening_temp_plies else "post_opening"
            buckets[key]["plies"] += 1
            rank = rec.get("selected_visit_rank")
            if rank is None:
                raise ValueError(
                    f"game {replay.get('game_idx')} ply {rec['ply']}: "
                    f"selected_visit_rank missing; cannot report non-leader rate")
            if rank > 1:
                buckets[key]["nonleader"] += 1
    return {
        k: {**v, "rate": (v["nonleader"] / v["plies"]) if v["plies"] else None}
        for k, v in buckets.items()
    }


def load_replays(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            out.append(json.load(fh))
    if not out:
        raise ValueError(f"no replays matched {pattern!r}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Frozen preflight over captured readout replays.")
    ap.add_argument("--replay-glob", required=True)
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    replays = load_replays(args.replay_glob)
    stats = preflight_stats(replays, args.agent_id, args.opening_temp_plies)
    gates = evaluate_gates(stats)
    report = {
        "stats": stats,
        "gates": gates,
        # Descriptive, gates nothing. Spec section 7.3 attribution aid.
        "nonleader_selection": nonleader_selection_report(
            replays, args.agent_id, args.opening_temp_plies),
    }
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    # Exit 0 = pass, 2 = a frozen gate closed the candidate.
    return 0 if gates["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
