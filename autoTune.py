#!/usr/bin/env python3
"""
Utility CLI for automating heuristic tuning and validation workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import signal


PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
SWEEP_LOG = LOGS_DIR / "sweep-results.json"
VALIDATION_LOG = LOGS_DIR / "validation-results.json"
NEXT_SWEEP_PATH = LOGS_DIR / "next-sweep.json"
STATE_PATH = LOGS_DIR / "autoTune-state.json"
PENDING_VALIDATION_PATH = LOGS_DIR / "pending-validation.json"
SEARCH_PATH = PROJECT_ROOT / "assets/js/ai/search.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def stable_serialize(value: Any) -> str:
    if isinstance(value, dict):
        items = sorted(value.items())
        return "{" + ",".join(f"{json.dumps(k)}:{stable_serialize(v)}" for k, v in items) + "}"
    if isinstance(value, list):
        return "[" + ",".join(stable_serialize(v) for v in value) + "]"
    return json.dumps(value, separators=(",", ":"))


OPTIONAL_OFFENSE_KEYS = [
    "connectorBonus",
    "finishThreshold",
    "finishBonusBase",
    "connectorTargetBonus",
    "doubleCoverageBase",
    "finishGapSlope",
    "nearFinishBonus",
    "redFinishExtra",
    "redGapDecayMultiplier",
]

HASH_KEYS = [
    "firstEdgeTouchRed",
    "firstEdgeTouchBlack",
    "finishPenaltyBase",
    "redFinishPenaltyFactor",
    "blackFinishScaleMultiplier",
    "redSpanGainMultiplier",
    "blackSpanGainMultiplier",
    "redDoubleCoverageBonus",
    "blackDoubleCoverageScale",
] + OPTIONAL_OFFENSE_KEYS

TREND_SLOPE_THRESHOLD = 0.05
TREND_CORRELATION_THRESHOLD = 0.2
UNDER_SAMPLED_COUNT = 2
PARITY_THRESHOLD = 3
DRAW_THRESHOLD = 6
SUCCESS_VALIDATION_STREAK = 2
OVERALL_STALL_CYCLES = 5

def compute_config_hash(combo: Dict[str, Any]) -> str:
    payload = {key: combo[key] for key in HASH_KEYS}
    serialized = stable_serialize(payload)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(ts: str) -> str:
    # ISO 8601 strings compare lexicographically when uniformly formatted.
    return ts


def load_search_config() -> Dict[str, Any]:
    if not SEARCH_PATH.exists():
        raise SystemExit(f"Missing baseline config at {SEARCH_PATH}")
    return load_json(SEARCH_PATH, {})


def save_search_config(config: Dict[str, Any]) -> None:
    write_json(SEARCH_PATH, config)


def ensure_offense(config: Dict[str, Any]) -> Dict[str, Any]:
    rewards = config.setdefault("rewards", {})
    edge = rewards.setdefault("edge", {})
    return edge.setdefault("offense", {})


def offense_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    offense = (
        config.get("rewards", {})
        .get("edge", {})
        .get("offense", {})
    )
    snapshot = {
        "blackFinishScaleMultiplier": offense.get("blackFinishScaleMultiplier"),
        "spanGainBase": offense.get("spanGainBase"),
        "gapDecay": offense.get("gapDecay"),
    }
    for key in OPTIONAL_OFFENSE_KEYS:
        snapshot[key] = offense.get(key)
    return snapshot


def extract_combo_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    offense = (
        config.get("rewards", {})
        .get("edge", {})
        .get("offense", {})
    )
    combo = {
        "firstEdgeRed": offense.get("firstEdgeTouchRed"),
        "firstEdgeBlack": offense.get("firstEdgeTouchBlack"),
        "finishPenalty": offense.get("finishPenaltyBase"),
        "redFinishPenaltyFactor": offense.get("redFinishPenaltyFactor"),
        "blackFinishScaleMultiplier": offense.get("blackFinishScaleMultiplier"),
        "redSpanGainMultiplier": offense.get("redSpanGainMultiplier"),
        "blackSpanGainMultiplier": offense.get("blackSpanGainMultiplier"),
        "redDoubleCoverageBonus": offense.get("redDoubleCoverageBonus"),
        "blackDoubleCoverageScale": offense.get("blackDoubleCoverageScale"),
    }
    for key in OPTIONAL_OFFENSE_KEYS:
        combo[key] = offense.get(key)
    return combo


def combo_from_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    combo = {
        "firstEdgeRed": entry.get("firstEdgeRed"),
        "firstEdgeBlack": entry.get("firstEdgeBlack"),
        "finishPenalty": entry.get("finishPenalty"),
        "gapDecay": entry.get("gapDecay"),
        "redFinishPenaltyFactor": entry.get("redFinishPenaltyFactor"),
        "blackFinishScaleMultiplier": entry.get("blackFinishScaleMultiplier"),
        "spanGainBase": entry.get("spanGainBase") or entry.get("spanBase"),
        "redSpanGainMultiplier": entry.get("redSpanGainMultiplier") or entry.get("redSpanMult"),
        "blackSpanGainMultiplier": entry.get("blackSpanGainMultiplier") or entry.get("blackSpanMult"),
        "redDoubleCoverageBonus": entry.get("redDoubleCoverageBonus") or entry.get("redDoubleCov"),
        "blackDoubleCoverageScale": entry.get("blackDoubleCoverageScale") or entry.get("blackDoubleCovScale"),
    }
    combo["connectorBonus"] = entry.get("connectorBonus")
    combo["finishThreshold"] = entry.get("finishThreshold")
    combo["finishBonusBase"] = entry.get("finishBonusBase")
    combo["connectorTargetBonus"] = entry.get("connectorTargetBonus")
    combo["doubleCoverageBase"] = entry.get("doubleCoverageBase")
    combo["finishGapSlope"] = entry.get("finishGapSlope")
    combo["nearFinishBonus"] = entry.get("nearFinishBonus")
    combo["redFinishExtra"] = entry.get("redFinishExtra")
    combo["redGapDecayMultiplier"] = entry.get("redGapDecayMultiplier")
    return combo


def render_combo(combo: Dict[str, Any]) -> str:
    parts = [
        f"R:{combo['firstEdgeRed']}",
        f"B:{combo['firstEdgeBlack']}",
        f"F:{combo['finishPenalty']}",
        f"rPen:{combo['redFinishPenaltyFactor']}",
        f"rSpan:{combo['redSpanGainMultiplier']}",
        f"bSpan:{combo['blackSpanGainMultiplier']}",
        f"rCov:{combo['redDoubleCoverageBonus']}",
        f"bCov:{combo['blackDoubleCoverageScale']}",
    ]
    return " ".join(parts)


@dataclass
class KnobSpec:
    name: str
    values: List[Any]


@dataclass
class KnobValueStats:
    count: int
    avg_score: float
    best_score: float


@dataclass
class KnobTrend:
    slope: float
    correlation: float
    value_stats: Dict[Any, KnobValueStats]
    top_values: List[Any]
    under_sampled: List[Any]

def build_values(min_val: float, max_val: float, step: float) -> List[Any]:
    count = int(round((max_val - min_val) / step))
    result: List[Any] = []
    step_is_integer = float(step).is_integer()
    for idx in range(count + 1):
        val = min_val + idx * step
        if isinstance(min_val, int) and isinstance(max_val, int) and step_is_integer:
            result.append(int(round(val)))
        else:
            result.append(round(val, 4))
    return result


KNOB_SPECS: Dict[str, KnobSpec] = {
    "firstEdgeRed": KnobSpec("firstEdgeRed", build_values(410, 440, 5)),
    "firstEdgeBlack": KnobSpec("firstEdgeBlack", build_values(445, 465, 5)),
    "finishPenalty": KnobSpec("finishPenalty", build_values(1161, 1221, 20)),
    "redFinishPenaltyFactor": KnobSpec("redFinishPenaltyFactor", build_values(0.15, 0.8, 0.05)),
    "blackFinishScaleMultiplier": KnobSpec("blackFinishScaleMultiplier", build_values(0.85, 1.1, 0.05)),
    "redSpanGainMultiplier": KnobSpec("redSpanGainMultiplier", build_values(0.9, 1.3, 0.05)),
    "blackSpanGainMultiplier": KnobSpec("blackSpanGainMultiplier", build_values(0.7, 1.1, 0.05)),
    "redDoubleCoverageBonus": KnobSpec("redDoubleCoverageBonus", build_values(0, 1800, 100)),
    "blackDoubleCoverageScale": KnobSpec("blackDoubleCoverageScale", build_values(0.55, 1.0, 0.05)),
}


def nearest_value(values: Sequence[Any], target: Any) -> Any:
    if not values:
        raise ValueError("Empty value domain")
    def score(val: Any) -> float:
        return abs(float(val) - float(target))
    return min(values, key=score)


def compute_regression(values: List[float], scores: List[float]) -> Tuple[float, float]:
    if not values or len(values) != len(scores):
        return 0.0, 0.0
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean_x = sum(values) / n
    mean_y = sum(scores) / n
    numerator = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for x, y in zip(values, scores):
        dx = x - mean_x
        dy = y - mean_y
        numerator += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    slope = numerator / denom_x if denom_x else 0.0
    if denom_x and denom_y:
        correlation = numerator / math.sqrt(denom_x * denom_y)
    else:
        correlation = 0.0
    return slope, correlation


def mutate_combo(base: Dict[str, Any], rng: random.Random, active_knobs: Sequence[str]) -> Dict[str, Any]:
    mutated = dict(base)
    knobs = [knob for knob in KNOB_SPECS.keys() if knob in active_knobs]
    if not knobs:
        return mutated
    change_count = rng.randint(1, min(3, len(knobs)))
    for knob in rng.sample(knobs, change_count):
        spec = KNOB_SPECS[knob]
        values = spec.values
        current = mutated.get(knob, values[0])
        if current not in values:
            current = nearest_value(values, current)
        idx = values.index(current)
        span = max(1, len(values) // 12)
        shift = rng.choice([-1, 1]) * rng.randint(1, span)
        new_idx = max(0, min(len(values) - 1, idx + shift))
        mutated[knob] = values[new_idx]
    return mutated


def random_combo(rng: random.Random, base: Dict[str, Any], active_knobs: Sequence[str]) -> Dict[str, Any]:
    result = dict(base)
    if not active_knobs:
        return result
    for knob in active_knobs:
        spec = KNOB_SPECS[knob]
        result[knob] = rng.choice(spec.values)
    return result


def combo_with_defaults(combo: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(combo)
    for key in OPTIONAL_OFFENSE_KEYS:
        if result.get(key) is None:
            default_value = defaults.get(key)
            if default_value is not None:
                result[key] = default_value
    if result.get("blackFinishScaleMultiplier") is None:
        result["blackFinishScaleMultiplier"] = defaults.get("blackFinishScaleMultiplier", 1.0)
    if result.get("spanGainBase") is None:
        result["spanGainBase"] = defaults.get("spanGainBase", 180)
    if result.get("gapDecay") is None:
        result["gapDecay"] = defaults.get("gapDecay", 23)
    return result


def combo_has_all_knobs(combo: Dict[str, Any]) -> bool:
    for knob in KNOB_SPECS.keys():
        if combo.get(knob) is None:
            return False
    return True


def analyze_knob_trends(entries: List[Dict[str, Any]]) -> Dict[str, KnobTrend]:
    trends: Dict[str, KnobTrend] = {}
    for knob, spec in KNOB_SPECS.items():
        value_bucket: Dict[Any, List[float]] = {}
        numeric_values: List[float] = []
        numeric_scores: List[float] = []
        for entry in entries:
            combo = entry.get("combo") or {}
            score = entry.get("score")
            if score is None:
                continue
            value = combo.get(knob)
            if value is None:
                continue
            score_f = float(score)
            numeric_scores.append(score_f)
            numeric_values.append(float(value))
            value_bucket.setdefault(value, []).append(score_f)
        if not value_bucket:
            continue
        if len(set(numeric_values)) > 1:
            slope, corr = compute_regression(numeric_values, numeric_scores)
        else:
            slope, corr = 0.0, 0.0
        value_stats: Dict[Any, KnobValueStats] = {}
        for value, scores in value_bucket.items():
            avg_score = sum(scores) / len(scores)
            best_score = min(scores)
            value_stats[value] = KnobValueStats(count=len(scores), avg_score=avg_score, best_score=best_score)
        top_values = sorted(
            value_stats.keys(),
            key=lambda v: (value_stats[v].avg_score, -value_stats[v].count)
        )
        under_sampled = [val for val, stat in value_stats.items() if stat.count <= UNDER_SAMPLED_COUNT]
        unseen = [val for val in spec.values if val not in value_stats]
        under_sampled.extend(unseen)
        trends[knob] = KnobTrend(
            slope=slope,
            correlation=corr,
            value_stats=value_stats,
            top_values=top_values,
            under_sampled=under_sampled
        )
    return trends


def generate_trend_variants(base_combo: Dict[str, Any], trends: Dict[str, KnobTrend], rng: random.Random, max_count: int, active_knobs: Sequence[str]) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = []
    sorted_trends = sorted(
        trends.items(),
        key=lambda item: abs(item[1].slope),
        reverse=True
    )
    for knob, data in sorted_trends:
        if knob not in active_knobs:
            continue
        if len(variants) >= max_count:
            break
        if abs(data.slope) < TREND_SLOPE_THRESHOLD and abs(data.correlation) < TREND_CORRELATION_THRESHOLD:
            continue
        spec = KNOB_SPECS[knob]
        current = base_combo.get(knob, spec.values[0])
        if current not in spec.values:
            current = nearest_value(spec.values, current)
        idx = spec.values.index(current)
        direction = -1 if data.slope > 0 else 1
        candidate_indices = []
        for step in (1, 2):
            target_idx = idx + direction * step
            if 0 <= target_idx < len(spec.values) and target_idx not in candidate_indices:
                candidate_indices.append(target_idx)
        if not candidate_indices:
            continue
        rng.shuffle(candidate_indices)
        for target_idx in candidate_indices:
            if len(variants) >= max_count:
                break
            candidate = dict(base_combo)
            candidate[knob] = spec.values[target_idx]
            variants.append((candidate, f"trend:{knob}"))
    return variants


def generate_best_value_variants(base_combo: Dict[str, Any], trends: Dict[str, KnobTrend], max_count: int, active_knobs: Sequence[str]) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = []
    ranking: List[Tuple[float, int, str, Any]] = []
    for knob, data in trends.items():
        if knob not in active_knobs:
            continue
        if not data.top_values:
            continue
        best_value = data.top_values[0]
        stats = data.value_stats.get(best_value)
        avg_score = stats.avg_score if stats else float("inf")
        sample_count = stats.count if stats else 0
        ranking.append((avg_score, -sample_count, knob, best_value))
    ranking.sort()
    for _, _, knob, value in ranking:
        if len(variants) >= max_count:
            break
        if base_combo.get(knob) == value:
            continue
        candidate = dict(base_combo)
        candidate[knob] = value
        variants.append((candidate, f"best:{knob}"))
    return variants


def generate_underexplored_variants(base_combo: Dict[str, Any], trends: Dict[str, KnobTrend], rng: random.Random, max_count: int, active_knobs: Sequence[str]) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = []
    knob_order = [knob for knob in KNOB_SPECS.keys() if knob in active_knobs]
    rng.shuffle(knob_order)
    for knob in knob_order:
        if len(variants) >= max_count:
            break
        spec = KNOB_SPECS[knob]
        data = trends.get(knob)
        candidates: List[Any] = []
        if data:
            candidates.extend([val for val in data.under_sampled if val is not None])
        else:
            candidates.extend(spec.values)
        if not candidates:
            continue
        rng.shuffle(candidates)
        for value in candidates:
            if len(variants) >= max_count:
                break
            candidate = dict(base_combo)
            candidate[knob] = value
            variants.append((candidate, f"explore:{knob}"))
    return variants


def to_hash_payload(combo: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "firstEdgeTouchRed": combo["firstEdgeRed"],
        "firstEdgeTouchBlack": combo["firstEdgeBlack"],
        "finishPenaltyBase": combo["finishPenalty"],
        "redFinishPenaltyFactor": combo["redFinishPenaltyFactor"],
        "blackFinishScaleMultiplier": combo["blackFinishScaleMultiplier"],
        "redSpanGainMultiplier": combo["redSpanGainMultiplier"],
        "blackSpanGainMultiplier": combo["blackSpanGainMultiplier"],
        "redDoubleCoverageBonus": combo["redDoubleCoverageBonus"],
        "blackDoubleCoverageScale": combo["blackDoubleCoverageScale"],
    }
    for key in OPTIONAL_OFFENSE_KEYS:
        payload[key] = combo.get(key)
    return payload


def attach_hash(combo: Dict[str, Any]) -> Dict[str, Any]:
    payload = to_hash_payload(combo)
    hash_value = compute_config_hash(payload)
    enriched = dict(combo)
    enriched["configHash"] = hash_value
    return enriched


def load_sweeps() -> List[Dict[str, Any]]:
    data = load_json(SWEEP_LOG, {"sweeps": []})
    sweeps: List[Dict[str, Any]] = data.get("sweeps", [])
    sweeps.sort(key=lambda entry: entry.get("timestamp", ""))
    return sweeps


def flatten_sweeps(sweeps: Iterable[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for sweep in sweeps:
        timestamp = sweep.get("timestamp")
        for combo in sweep.get("combos", []):
            original_hash = combo.get("configHash")
            combo_data = combo_from_log(combo)
            normalized_hash = original_hash
            if defaults is not None:
                combo_data = combo_with_defaults(combo_data, defaults)
                normalized_hash = compute_config_hash(to_hash_payload(combo_data))
            flattened.append(
                {
                    "timestamp": timestamp,
                    "configHash": normalized_hash or original_hash,
                    "originalConfigHash": original_hash,
                    "score": combo.get("score"),
                    "evaluation": combo.get("evaluation", {}),
                    "combo": combo_data,
                }
            )
    return flattened


def load_validations() -> List[Dict[str, Any]]:
    data = load_json(VALIDATION_LOG, {"runs": []})
    runs: List[Dict[str, Any]] = data.get("runs", [])
    runs.sort(key=lambda entry: entry.get("timestamp", ""))
    return runs


def compute_metrics(combo_entry: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = combo_entry.get("evaluation") or {}
    def depth_stats(depth: Dict[str, Any]) -> Dict[str, Any]:
        red = int(depth.get("red", 0) or 0)
        black = int(depth.get("black", 0) or 0)
        draw = int(depth.get("draw", 0) or 0)
        total = red + black + draw
        diff = abs(red - black)
        rate = (red / total) if total else 0.0
        return {
            "red": red,
            "black": black,
            "draw": draw,
            "total": total,
            "imbalance": diff,
            "redRate": rate,
        }

    depth2 = depth_stats(evaluation.get("depth2", {}))
    depth3 = depth_stats(evaluation.get("depth3", {}))
    total_red = depth2["red"] + depth3["red"]
    total_black = depth2["black"] + depth3["black"]
    total_draw = depth2["draw"] + depth3["draw"]
    total_games = total_red + total_black + total_draw
    return {
        "score": combo_entry.get("score"),
        "totalGames": total_games,
        "totalRed": total_red,
        "totalBlack": total_black,
        "totalDraw": total_draw,
        "parity": abs(total_red - total_black),
        "depth2": depth2,
        "depth3": depth3,
    }


def load_state() -> Dict[str, Any]:
    return load_json(STATE_PATH, {})


def save_state(state: Dict[str, Any]) -> None:
    write_json(STATE_PATH, state)


def select_base_combo(
    baseline_combo: Dict[str, Any],
    baseline_defaults: Dict[str, Any],
    history_entries: List[Dict[str, Any]],
    validations: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    aggregated_validations = aggregate_validation_runs(validations)
    best_hash: Optional[str] = None
    best_metric: Optional[Tuple[int, int]] = None
    for config_hash, bucket in aggregated_validations.items():
        games = bucket.get("games", 0)
        if games <= 0:
            continue
        parity = abs(bucket.get("red", 0) - bucket.get("black", 0))
        metric = (parity, -games)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_hash = config_hash
    if best_hash:
        record = find_combo_by_hash(best_hash)
        if record:
            combo = combo_with_defaults(combo_from_log(record), baseline_defaults)
            if combo_has_all_knobs(combo):
                return combo, best_hash
    scored_entries = [entry for entry in history_entries if entry.get("score") is not None]
    scored_entries.sort(key=lambda entry: (entry.get("score", float("inf")), entry.get("timestamp", "")))
    for entry in scored_entries:
        combo = combo_with_defaults(entry["combo"], baseline_defaults)
        if combo_has_all_knobs(combo):
            return combo, entry.get("configHash")
    fallback = combo_with_defaults(baseline_combo, baseline_defaults)
    return fallback, None


def command_suggest(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed if args.seed is not None else random.randrange(1 << 30))
    sweeps = load_sweeps()
    search_config = load_search_config()
    baseline_defaults = offense_snapshot(search_config)
    flattened = flatten_sweeps(sweeps, baseline_defaults)
    state = load_state()
    knob_stats = state.get("knobStats", {})
    frozen_knobs = {knob for knob, stats in knob_stats.items() if stats.get("frozen")}
    frozen_values = {knob: stats.get("bestValue") for knob, stats in knob_stats.items() if stats.get("frozen") and stats.get("bestValue") is not None}
    active_knobs = [knob for knob in KNOB_SPECS.keys() if knob not in frozen_knobs and len(KNOB_SPECS[knob].values) > 1]
    baseline_combo_raw = extract_combo_from_config(search_config)
    for knob, value in frozen_values.items():
        baseline_combo_raw[knob] = value
    validations = load_validations()
    requested = args.count
    wishlist: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    validation_streaks = compute_validation_streaks(validations, baseline_defaults)
    validated_hashes: set[str] = set(validation_streaks.keys())
    validated_hashes.update(
        run.get("configHash")
        for run in validations
        if run.get("configHash")
    )
    history_entries = flattened
    analytics_entries = [entry for entry in history_entries if entry.get("score") is not None]
    trends = analyze_knob_trends(analytics_entries)
    base_combo, base_hash = select_base_combo(baseline_combo_raw, baseline_defaults, history_entries, validations)
    for knob, value in frozen_values.items():
        base_combo[knob] = value

    def append_candidate(candidate_combo: Dict[str, Any], origin: str, *, score: Optional[float] = None, source: Optional[str] = None) -> bool:
        nonlocal wishlist, seen_hashes
        materialized = combo_with_defaults(candidate_combo, baseline_defaults)
        for knob, value in frozen_values.items():
            materialized[knob] = value
        if not combo_has_all_knobs(materialized):
            return False
        enriched = attach_hash(materialized)
        hash_value = enriched["configHash"]
        if hash_value in validated_hashes:
            return False
        if hash_value in seen_hashes and not args.allow_duplicates:
            return False
        if hash_value in seen_hashes and args.allow_duplicates:
            origin = f"{origin}-dup"
        enriched["origin"] = origin
        if score is not None:
            enriched["score"] = score
        else:
            enriched.setdefault("score", None)
        enriched["sourceSweep"] = source
        wishlist.append(enriched)
        seen_hashes.add(hash_value)
        return True

    requested = args.count
    exploitation_target = max(0, min(args.exploit, requested))
    flattened_sorted = sorted(
        history_entries,
        key=lambda entry: (
            entry.get("score", float("inf")),
            entry.get("timestamp", "")
        )
    )
    for entry in flattened_sorted:
        if len(wishlist) >= exploitation_target:
            break
        hash_value = entry.get("configHash")
        if not hash_value or hash_value in validated_hashes or hash_value in seen_hashes:
            continue
        combo = entry["combo"]
        append_candidate(combo, "exploit", score=entry.get("score"), source=entry.get("timestamp"))

    remaining = requested - len(wishlist)
    if remaining > 0:
        trend_quota = max(0, min(remaining, max(2, requested // 4)))
        for combo, origin in generate_trend_variants(base_combo, trends, rng, trend_quota * 2, active_knobs):
            if len(wishlist) >= requested or trend_quota <= 0:
                break
            if append_candidate(combo, origin):
                trend_quota -= 1

    remaining = requested - len(wishlist)
    if remaining > 0:
        best_quota = max(0, min(remaining, max(2, requested // 4)))
        best_variants = generate_best_value_variants(base_combo, trends, best_quota * 2, active_knobs)
        for combo, origin in best_variants:
            if len(wishlist) >= requested or best_quota <= 0:
                break
            if append_candidate(combo, origin):
                best_quota -= 1
        if best_variants:
            ranked_choices: List[Tuple[float, int, str, Any]] = []
            for knob, data in trends.items():
                if knob not in active_knobs:
                    continue
                if not data.top_values:
                    continue
                best_value = data.top_values[0]
                stats = data.value_stats.get(best_value)
                avg_score = stats.avg_score if stats else float("inf")
                sample_count = stats.count if stats else 0
                ranked_choices.append((avg_score, -sample_count, knob, best_value))
            ranked_choices.sort()
            combined = dict(base_combo)
            changed = False
            for _, _, knob, value in ranked_choices[:3]:
                if combined.get(knob) != value:
                    combined[knob] = value
                    changed = True
            if changed:
                append_candidate(combined, "best:mixed")

    remaining = requested - len(wishlist)
    if remaining > 0:
        explore_quota = max(0, min(remaining, max(2, requested // 4)))
        for combo, origin in generate_underexplored_variants(base_combo, trends, rng, explore_quota * 2, active_knobs):
            if len(wishlist) >= requested or explore_quota <= 0:
                break
            if append_candidate(combo, origin):
                explore_quota -= 1

    filler_attempts = 0
    while len(wishlist) < requested:
        if not active_knobs:
            break
        if rng.random() < 0.6:
            candidate = mutate_combo(base_combo, rng, active_knobs)
            origin = "mutate"
        else:
            candidate = random_combo(rng, base_combo, active_knobs)
            origin = "random"
        if append_candidate(candidate, origin):
            continue
        filler_attempts += 1
        if filler_attempts >= requested * 5:
            break

    wishlist = wishlist[:requested]
    payload = {
        "generatedAt": iso_now(),
        "seed": getattr(args, "seed", None),
        "requested": requested,
        "combos": wishlist,
    }
    write_json(NEXT_SWEEP_PATH, payload)

    state["lastSuggested"] = payload["generatedAt"]
    state["plannedSweep"] = {
        "hashes": [c["configHash"] for c in wishlist],
        "generatedAt": payload["generatedAt"],
    }
    save_state(state)

    print(f"Wrote {len(wishlist)} combos to {NEXT_SWEEP_PATH.relative_to(PROJECT_ROOT)}")
    for idx, combo in enumerate(wishlist, 1):
        print(f"[{idx:02d}] {combo['origin']:>7} {combo['configHash'][:8]}  {render_combo(combo)}")


def command_sweep(_args: argparse.Namespace) -> None:
    cmd = ["node", "scripts/tuneBaseline.js"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)


def recommend_for_validation(entries: List[Dict[str, Any]], limit: int, validated_hashes: set[str]) -> List[Dict[str, Any]]:
    filtered = [
        entry for entry in entries
        if entry["configHash"] and entry["configHash"] not in validated_hashes
    ]
    filtered.sort(key=lambda entry: (entry.get("score", 1_000_000), entry["timestamp"]))
    recommendations: List[Dict[str, Any]] = []
    for entry in filtered:
        metrics = compute_metrics(entry)
        if metrics["totalGames"] <= 0:
            continue
        recommendations.append(
            {
                "configHash": entry["configHash"],
                "sweepTimestamp": entry["timestamp"],
                "score": entry.get("score"),
                "metrics": metrics,
                "combo": entry["combo"],
            }
        )
        if len(recommendations) >= limit:
            break
    return recommendations


def command_update(args: argparse.Namespace) -> None:
    sweeps = load_sweeps()
    state = load_state()
    last_processed = state.get("lastProcessedSweep")
    new_sweeps = [
        sweep for sweep in sweeps
        if not last_processed or parse_iso(sweep.get("timestamp", "")) > parse_iso(last_processed)
    ]
    if not new_sweeps:
        print("No new sweep results to process.")
        return

    latest_timestamp = new_sweeps[-1].get("timestamp")
    baseline_defaults = offense_snapshot(load_search_config())
    flattened = flatten_sweeps(new_sweeps, baseline_defaults)
    validated_runs = load_validations()
    streaks = compute_validation_streaks(validated_runs, baseline_defaults)
    normalized_validated_hashes = set(streaks.keys())
    original_validated_hashes = {run.get("configHash") for run in validated_runs if run.get("configHash")}
    successful_hashes = {hash_value for hash_value, streak in streaks.items() if streak >= SUCCESS_VALIDATION_STREAK}
    skip_hashes = normalized_validated_hashes.union(original_validated_hashes).union(successful_hashes)
    prune_validated_pending(skip_hashes)

    knob_best_entries: Dict[str, Dict[str, Any]] = {}
    for entry in flattened:
        score = entry.get("score")
        if not isinstance(score, (int, float)):
            continue
        combo = entry.get("combo") or {}
        for knob in KNOB_SPECS.keys():
            value = combo.get(knob)
            if value is None:
                continue
            best = knob_best_entries.get(knob)
            if best is None or score < best["score"]:
                knob_best_entries[knob] = {"score": score, "value": value}

    best_score_cycle = None
    for entry in flattened:
        score = entry.get("score")
        if isinstance(score, (int, float)):
            best_score_cycle = score if best_score_cycle is None else min(best_score_cycle, score)
    current_best_streak = max(streaks.values()) if streaks else 0

    knob_stats = state.get("knobStats", {})
    newly_frozen: List[str] = []
    for knob in KNOB_SPECS.keys():
        stats = knob_stats.get(knob, {"bestScore": None, "bestValue": None, "cyclesSinceImprovement": 0, "frozen": False})
        if stats.get("bestValue") is None:
            stats["bestValue"] = (knob_best_entries.get(knob) or {}).get("value")
        if stats.get("bestValue") is None:
            stats["bestValue"] = baseline_defaults.get(knob)
        was_frozen = stats.get("frozen")
        spec = KNOB_SPECS[knob]
        if len(spec.values) <= 1:
            stats["frozen"] = True
        if not stats.get("frozen"):
            best_entry = knob_best_entries.get(knob)
            if best_entry and (stats.get("bestScore") is None or best_entry["score"] < stats["bestScore"]):
                stats["bestScore"] = best_entry["score"]
                stats["bestValue"] = best_entry["value"]
                stats["cyclesSinceImprovement"] = 0
            else:
                stats["cyclesSinceImprovement"] = stats.get("cyclesSinceImprovement", 0) + 1
                if stats["cyclesSinceImprovement"] >= KNOB_STALL_CYCLES:
                    stats["frozen"] = True
        knob_stats[knob] = stats
        if not was_frozen and stats.get("frozen"):
            newly_frozen.append(knob)
    state["knobStats"] = knob_stats
    if newly_frozen:
        print("\nKnobs frozen due to plateau:", ", ".join(newly_frozen))

    state["validationStreaks"] = streaks
    state["successfulConfigs"] = {hash_value: streaks[hash_value] for hash_value in successful_hashes}

    prev_best_score = state.get("bestScore")
    if best_score_cycle is not None:
        if prev_best_score is None or best_score_cycle < prev_best_score:
            state["bestScore"] = best_score_cycle
            state["cyclesSinceScoreImprovement"] = 0
        else:
            state["cyclesSinceScoreImprovement"] = state.get("cyclesSinceScoreImprovement", 0) + 1

    prev_best_streak = state.get("bestStreak", 0)
    if current_best_streak > prev_best_streak:
        state["bestStreak"] = current_best_streak
        state["cyclesSinceStreakImprovement"] = 0
    else:
        state["cyclesSinceStreakImprovement"] = state.get("cyclesSinceStreakImprovement", 0) + 1

    recommendations = recommend_for_validation(flattened, args.limit, skip_hashes)
    pending_payload = {
        "generatedAt": iso_now(),
        "recommendations": recommendations,
    }
    write_json(PENDING_VALIDATION_PATH, pending_payload)

    state["lastProcessedSweep"] = latest_timestamp
    state["pendingValidation"] = [entry["configHash"] for entry in recommendations]
    save_state(state)

    print(f"Processed {len(new_sweeps)} new sweep batch(es). Latest timestamp: {latest_timestamp}")
    for entry in flattened:
        metrics = compute_metrics(entry)
        display = (
            f"{entry['timestamp']} {entry['configHash'][:8]} "
            f"score={entry.get('score')} red={metrics['totalRed']} "
            f"black={metrics['totalBlack']} draw={metrics['totalDraw']} "
            f"parity={metrics['parity']}"
        )
        print(display)

    if successful_hashes:
        print("\nConfigs meeting validation goal:")
        for hash_value in sorted(successful_hashes):
            print(f"  {hash_value[:8]} streak={streaks.get(hash_value, 0)}")

    if recommendations:
        print("\nRecommended for validation:")
        for idx, rec in enumerate(recommendations, 1):
            metrics = rec["metrics"]
            print(
                f" {idx}. {rec['configHash'][:8]} score={rec['score']} "
                f"red={metrics['totalRed']} black={metrics['totalBlack']} "
                f"draw={metrics['totalDraw']} parity={metrics['parity']}"
            )
    else:
        print("No outstanding configs require validation.")


def apply_combo_to_search(combo: Dict[str, Any], config: Dict[str, Any]) -> None:
    offense = ensure_offense(config)
    offense["firstEdgeTouchRed"] = combo["firstEdgeRed"]
    offense["firstEdgeTouchBlack"] = combo["firstEdgeBlack"]
    offense["finishPenaltyBase"] = combo["finishPenalty"]
    offense["redFinishPenaltyFactor"] = combo["redFinishPenaltyFactor"]
    offense["blackFinishScaleMultiplier"] = combo["blackFinishScaleMultiplier"]
    offense["redSpanGainMultiplier"] = combo["redSpanGainMultiplier"]
    offense["blackSpanGainMultiplier"] = combo["blackSpanGainMultiplier"]
    offense["redDoubleCoverageBonus"] = combo["redDoubleCoverageBonus"]
    offense["blackDoubleCoverageScale"] = combo["blackDoubleCoverageScale"]
    for key in OPTIONAL_OFFENSE_KEYS:
        value = combo.get(key)
        if value is not None:
            offense[key] = value


def find_combo_by_hash(config_hash: str) -> Optional[Dict[str, Any]]:
    sweeps = load_sweeps()
    for sweep in reversed(sweeps):
        for entry in sweep.get("combos", []):
            if entry.get("configHash") == config_hash:
                result = dict(entry)
                result["timestamp"] = sweep.get("timestamp")
                return result
    return None


def remove_pending_hash(config_hash: str) -> None:
    state = load_state()
    pending = state.get("pendingValidation", [])
    if config_hash in pending:
        pending = [hash_value for hash_value in pending if hash_value != config_hash]
        state["pendingValidation"] = pending
        save_state(state)
    data = load_json(PENDING_VALIDATION_PATH, {})
    recs = data.get("recommendations", [])
    recs = [entry for entry in recs if entry.get("configHash") != config_hash]
    if recs:
        data["recommendations"] = recs
        write_json(PENDING_VALIDATION_PATH, data)
    elif PENDING_VALIDATION_PATH.exists():
        PENDING_VALIDATION_PATH.unlink()


def prune_validated_pending(validated_hashes: set[str]) -> None:
    if not validated_hashes:
        return

    state = load_state()
    pending_list = state.get("pendingValidation")
    if isinstance(pending_list, list):
        filtered = [hash_value for hash_value in pending_list if hash_value not in validated_hashes]
        if filtered != pending_list:
            state["pendingValidation"] = filtered
            save_state(state)

    if PENDING_VALIDATION_PATH.exists():
        data = load_json(PENDING_VALIDATION_PATH, {})
        recs = data.get("recommendations")
        if isinstance(recs, list):
            filtered_recs = [entry for entry in recs if entry.get("configHash") not in validated_hashes]
            if filtered_recs:
                if filtered_recs != recs:
                    data["recommendations"] = filtered_recs
                    write_json(PENDING_VALIDATION_PATH, data)
            else:
                PENDING_VALIDATION_PATH.unlink()


def command_validate(args: argparse.Namespace) -> None:
    config_hash = args.hash
    if not config_hash:
        pending = load_json(PENDING_VALIDATION_PATH, {}).get("recommendations", [])
        if not pending:
            raise SystemExit("No pending validation recommendations. Specify --hash explicitly.")
        config_hash = pending[0].get("configHash")
        print(f"Using first pending recommendation: {config_hash}")

    record = find_combo_by_hash(config_hash)
    if not record:
        raise SystemExit(f"Config hash {config_hash} not found in sweep history.")

    combo = combo_from_log(record)
    baseline_config = load_search_config()
    original_config = json.loads(json.dumps(baseline_config))
    apply_combo_to_search(combo, baseline_config)
    save_search_config(baseline_config)

    cmd = ["node", "scripts/runValidation.js"]
    if args.depth_config:
        cmd.append(f"--depth-config={args.depth_config}")
    if args.workers:
        cmd.append(f"--workers={args.workers}")
    if args.log:
        cmd.append(f"--log={args.log}")

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        save_search_config(original_config)
        raise SystemExit(exc.returncode)

    if not args.persist:
        save_search_config(original_config)
        print("search.json restored to the previous configuration.")

    remove_pending_hash(config_hash)


def aggregate_validation_runs(runs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        config_hash = run.get("configHash")
        if not config_hash:
            continue
        bucket = aggregated.setdefault(
            config_hash,
            {
                "games": 0,
                "red": 0,
                "black": 0,
                "draw": 0,
                "depth2": {"red": 0, "black": 0, "draw": 0},
                "depth3": {"red": 0, "black": 0, "draw": 0},
                "runs": 0,
            },
        )
        bucket["games"] += run.get("gamesRecorded", 0)
        wins = run.get("wins", {})
        bucket["red"] += wins.get("red", 0)
        bucket["black"] += wins.get("black", 0)
        bucket["draw"] += wins.get("draw", 0)
        for depth_key in ("depth2", "depth3"):
            depth_data = run.get(depth_key, {})
            for side in ("red", "black", "draw"):
                bucket[depth_key][side] += depth_data.get(side, 0)
        bucket["runs"] += 1
    return aggregated


def validation_run_meets_goal(run: Dict[str, Any]) -> bool:
    depth2 = run.get("depth2") or {}
    depth3 = run.get("depth3") or {}
    draws2 = depth2.get("draw", 0) or 0
    draws3 = depth3.get("draw", 0) or 0
    parity2 = abs((depth2.get("red", 0) or 0) - (depth2.get("black", 0) or 0))
    parity3 = abs((depth3.get("red", 0) or 0) - (depth3.get("black", 0) or 0))
    return (
        parity2 <= PARITY_THRESHOLD
        and parity3 <= PARITY_THRESHOLD
        and draws2 <= DRAW_THRESHOLD
        and draws3 <= DRAW_THRESHOLD
    )


def compute_validation_streaks(runs: List[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    streaks: Dict[str, int] = {}
    for run in runs:
        config_hash = run.get("configHash")
        normalized_hash = config_hash
        config_snapshot = run.get("config") or {}
        if config_snapshot:
            payload = {}
            for key in HASH_KEYS:
                value = config_snapshot.get(key)
                if value is None and defaults is not None:
                    value = defaults.get(key)
                payload[key] = value
            normalized_hash = compute_config_hash(payload)
        if not normalized_hash:
            continue
        if validation_run_meets_goal(run):
            streaks[normalized_hash] = streaks.get(normalized_hash, 0) + 1
        else:
            streaks[normalized_hash] = 0
    return streaks


def command_loop(args: argparse.Namespace) -> None:
    stop_requested = False

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            print("\n[loop] Second interrupt received. Exiting immediately.")
            raise SystemExit(130)
        stop_requested = True
        print("\n[loop] Stop requested. Finishing current cycle before exiting...")

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)

    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"\n=== Auto-tune cycle {cycle} ===")

            seed_value = None
            if args.seed is not None:
                seed_value = args.seed + cycle - 1

            suggest_args = argparse.Namespace(
                count=args.count,
                exploit=args.exploit,
                seed=seed_value,
                allow_duplicates=False,
            )
            try:
                command_suggest(suggest_args)
            except SystemExit as exc:
                print(f"[loop] Suggest stage failed with exit code {exc.code}. Stopping loop.")
                return

            sweep_args = argparse.Namespace()
            try:
                command_sweep(sweep_args)
            except SystemExit as exc:
                print(f"[loop] Sweep stage failed with exit code {exc.code}. Stopping loop.")
                return

            update_args = argparse.Namespace(limit=args.limit)
            try:
                command_update(update_args)
            except SystemExit as exc:
                print(f"[loop] Update stage failed with exit code {exc.code}. Stopping loop.")
                return

            pending_payload = load_json(PENDING_VALIDATION_PATH, {})
            pending = pending_payload.get("recommendations", []) or []

            if pending:
                for idx, entry in enumerate(pending, 1):
                    hash_value = entry.get("configHash")
                    if not hash_value:
                        continue
                    log_name = f"{args.log_prefix}-{cycle:04d}-{idx:02d}.log"
                    validate_args = argparse.Namespace(
                        hash=hash_value,
                        depth_config=args.depth_config,
                        workers=args.workers,
                        log=log_name,
                        persist=args.persist,
                    )
                    try:
                        command_validate(validate_args)
                    except SystemExit as exc:
                        print(f"[loop] Validation for {hash_value[:8]} failed with exit code {exc.code}. Stopping loop.")
                        return
            else:
                print("[loop] No pending validations after update.")

            state = load_state()
            successful_configs = state.get("successfulConfigs", {})
            if successful_configs:
                print("\n[loop] Validation goal reached by:")
                for hash_value, streak in successful_configs.items():
                    print(f"  {hash_value[:8]} streak={streak}")
                break

            stall_score = state.get("cyclesSinceScoreImprovement", 0)
            stall_streak = state.get("cyclesSinceStreakImprovement", 0)
            best_streak = state.get("bestStreak", 0)
            if best_streak < SUCCESS_VALIDATION_STREAK and max(stall_score, stall_streak) >= OVERALL_STALL_CYCLES:
                print(
                    f"\n[loop] No improvement in sweep score or validation streak for {OVERALL_STALL_CYCLES} cycles. "
                    "Consider expanding knob ranges or adjusting the tuning strategy."
                )
                break

            if args.max_cycles is not None and cycle >= args.max_cycles:
                print(f"[loop] Reached max cycles ({args.max_cycles}). Exiting.")
                break

            if stop_requested:
                print("[loop] Stop requested. Exiting after completing current cycle.")
                break
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def command_report(_args: argparse.Namespace) -> None:
    sweeps = load_sweeps()
    baseline_defaults = offense_snapshot(load_search_config())
    flattened = flatten_sweeps(sweeps, baseline_defaults)
    validations = load_validations()
    aggregated_validations = aggregate_validation_runs(validations)
    streaks = compute_validation_streaks(validations, baseline_defaults)
    successful_hashes = {hash_value for hash_value, streak in streaks.items() if streak >= SUCCESS_VALIDATION_STREAK}

    if not flattened:
        print("No sweep data available.")
    else:
        scored_entries: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for entry in flattened:
            metrics = compute_metrics(entry)
            if metrics["totalGames"] <= 0:
                continue
            scored_entries.append((entry, metrics))
        if scored_entries:
            print("Top sweep configs by score:")
            scored_entries.sort(
                key=lambda item: (
                    item[0]["score"] if isinstance(item[0].get("score"), (int, float)) else float("inf"),
                    item[0].get("timestamp", ""),
                )
            )
            top_entries = scored_entries[: min(5, len(scored_entries))]
            for entry, metrics in top_entries:
                hash_prefix = entry["configHash"][:8] if entry["configHash"] else "--------"
                validated = "yes" if entry["configHash"] in aggregated_validations else "no"
                goal = "yes" if entry["configHash"] in successful_hashes else "no"
                print(
                    f" {hash_prefix} score={entry.get('score')} "
                    f"red={metrics['totalRed']} black={metrics['totalBlack']} "
                    f"draw={metrics['totalDraw']} parity={metrics['parity']} validated={validated} goal={goal}"
                )
        else:
            print("Sweep results lack evaluation data; run a sweep to populate metrics.")

    if aggregated_validations:
        print("\nValidation balance by config:")
        for config_hash, bucket in aggregated_validations.items():
            total = bucket["red"] + bucket["black"] + bucket["draw"]
            parity = abs(bucket["red"] - bucket["black"])
            streak = streaks.get(config_hash, 0)
            goal_flag = "yes" if streak >= SUCCESS_VALIDATION_STREAK else "no"
            print(
                f" {config_hash[:8]} runs={bucket['runs']} games={total} "
                f"red={bucket['red']} black={bucket['black']} draw={bucket['draw']} parity={parity} "
                f"streak={streak} goal={goal_flag}"
            )
    else:
        print("No validation runs logged yet.")

    pending = load_json(PENDING_VALIDATION_PATH, {}).get("recommendations", [])
    if pending:
        print("\nPending validation queue:")
        for idx, entry in enumerate(pending, 1):
            metrics = entry.get("metrics", {})
            print(
                f" {idx}. {entry['configHash'][:8]} "
                f"score={entry.get('score')} parity={metrics.get('parity')} "
                f"timestamp={entry.get('sweepTimestamp')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TwixT heuristic auto tuner")
    subparsers = parser.add_subparsers(dest="command")

    suggest_parser = subparsers.add_parser("suggest", help="Generate next sweep combos")
    suggest_parser.add_argument("--count", type=int, default=24, help="Number of combos to propose (default: 24)")
    suggest_parser.add_argument("--exploit", type=int, default=8, help="Number of top historical combos to repeat (default: 8)")
    suggest_parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    suggest_parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate config hashes in suggestions")
    suggest_parser.set_defaults(func=command_suggest)

    subparsers.add_parser("sweep", help="Run the baseline sweep script").set_defaults(func=command_sweep)

    update_parser = subparsers.add_parser("update", help="Process latest sweep results")
    update_parser.add_argument("--limit", type=int, default=3, help="Number of configs to queue for validation (default: 3)")
    update_parser.set_defaults(func=command_update)

    validate_parser = subparsers.add_parser("validate", help="Run validation for a config")
    validate_parser.add_argument("--hash", help="Config hash to validate (defaults to first pending recommendation)")
    validate_parser.add_argument("--depth-config", dest="depth_config", help="Depth config string (e.g. 2:60,3:60)")
    validate_parser.add_argument("--workers", help="Worker count passed to runValidation.js")
    validate_parser.add_argument("--log", help="Custom log file name")
    validate_parser.add_argument("--persist", action="store_true", help="Keep validated config in search.json after run")
    validate_parser.set_defaults(func=command_validate)

    subparsers.add_parser("report", help="Summarize sweep and validation status").set_defaults(func=command_report)

    loop_parser = subparsers.add_parser("loop", help="Run continuous auto-tuning cycle")
    loop_parser.add_argument("--count", type=int, default=24, help="Combos per sweep suggestion (default: 24)")
    loop_parser.add_argument("--exploit", type=int, default=8, help="History combos to retain per sweep (default: 8)")
    loop_parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic suggestions")
    loop_parser.add_argument("--limit", type=int, default=3, help="Validation queue size per sweep (default: 3)")
    loop_parser.add_argument("--depth-config", default="2:60,3:60", help="Validation depth config (default: 2:60,3:60)")
    loop_parser.add_argument("--workers", default="10", help="Validation worker count (default: 10)")
    loop_parser.add_argument("--log-prefix", default="loop-validation", help="Prefix for validation log files")
    loop_parser.add_argument("--persist", action="store_true", help="Keep last validated config in search.json")
    loop_parser.add_argument("--max-cycles", type=int, default=None, help="Optional cap on loop iterations")
    loop_parser.set_defaults(func=command_loop)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
