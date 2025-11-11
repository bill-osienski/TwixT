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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
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
        return (
            "{"
            + ",".join(f"{json.dumps(k)}:{stable_serialize(v)}" for k, v in items)
            + "}"
        )
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
VALIDATION_SCORE_CUTOFF = 2
PLATEAU_SCORE_CUTOFF = 2
VALIDATION_MAX_PER_CYCLE = 8
VALIDATION_PARALLEL_WORKERS = 10
PLATEAU_VALIDATION_WORKERS = 10
KNOB_STALL_CYCLES = 5
DEPTH3_PARITY_WEIGHT = 2.0
DEPTH3_VALIDATION_MAX_PARITY = 6
VALIDATION_TREND_WEIGHT = 3.0
VALIDATION_DRAW_WEIGHT = 0.25
SWEEP_TREND_DRAW_WEIGHT = VALIDATION_DRAW_WEIGHT
SOFT_BEST_POOL_SIZE = 12
SOFT_BEST_MIN_VARIANCE = 0.05
SOFT_BEST_RECENT_SWEEPS = 6
SOFT_BEST_RECENT_TARGET = 6
SOFT_BEST_LEGEND_TARGET = 6
SOFT_BEST_MIN_SWEEP_GAMES = 40
NICHE_DISTANCE_THRESHOLD = 0.08
NICHE_TOP_CHAMPIONS = 6
NICHE_HILL_ATTEMPTS = 5
NICHE_SLOT_RETRIES = 6
NICHE_MAX_NORMALIZED_DELTA = 0.05
COARSE_NICHE_KNOBS = {
    "firstEdgeRed",
    "firstEdgeBlack",
    "redFinishPenaltyFactor",
    "redSpanGainMultiplier",
    "blackSpanGainMultiplier",
}
BEST_MAX_COUNT = 4
BUCKET_NAMES = [
    "soft-best",
    "niche",
    "trend",
    "best",
    "explore",
    "mutate",
    "anchor",
    "other",
]
MUTATE_MIN_COUNT = 3
CATEGORY_WEIGHTS: List[Tuple[str, int]] = [
    ("soft_best", 5),
    ("niche", 4),
    ("best", 4),
    ("trend", 4),
    ("explore", 4),
    ("filler", 3),
]

HASH_STATUS_UNTESTED = "UNTESTED"
HASH_STATUS_SHORTLIST = "SHORTLIST"
HASH_STATUS_VALIDATING = "VALIDATING"
HASH_STATUS_STABLE = "STABLE"
HASH_STATUS_RETIRED = "RETIRED"
FINAL_STATUSES = {
    HASH_STATUS_STABLE,
    HASH_STATUS_RETIRED,
    HASH_STATUS_VALIDATING,
}

SHORTLIST_SCORE_THRESHOLD = 2
MAX_TEN10_SWEEPS_PER_SHORTLIST = 3
RETIRE_SCORE_THRESHOLD = 6
RETIRE_AFTER_SWEEPS = 3
ANCHOR_SLOTS_PER_SWEEP = 2
ANCHOR_COOLOFF_SWEEPS = 4
MUTATION_MAX_ATTEMPTS = 8


def compute_config_hash(combo: Dict[str, Any]) -> str:
    payload = {key: combo[key] for key in HASH_KEYS}
    serialized = stable_serialize(payload)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def default_hash_record() -> Dict[str, Any]:
    return {
        "status": HASH_STATUS_UNTESTED,
        "ten10_sweeps": 0,
        "shortlist_sweeps": 0,
        "score_last": None,
        "score_best": None,
        "games_d2": 0,
        "games_d3": 0,
        "last_sweep_id": 0,
        "last_score_timestamp": None,
        "validations_requested": 0,
        "validation_runs": 0,
    }


def ensure_hash_registry(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    registry = state.get("hashRegistry")
    if not isinstance(registry, dict):
        registry = {}
        state["hashRegistry"] = registry
    return registry


def evaluation_game_counts(evaluation: Dict[str, Any]) -> Tuple[int, int]:
    depth2 = evaluation.get("depth2") or {}
    depth3 = evaluation.get("depth3") or {}
    total_d2 = sum(depth2.get(side, 0) or 0 for side in ("red", "black", "draw"))
    total_d3 = sum(depth3.get(side, 0) or 0 for side in ("red", "black", "draw"))
    return total_d2, total_d3


def set_hash_status(record: Dict[str, Any], status: str) -> None:
    record["status"] = status


def promote_to_shortlist(record: Dict[str, Any]) -> None:
    if record.get("status") != HASH_STATUS_SHORTLIST:
        record["shortlist_sweeps"] = 0
    record["status"] = HASH_STATUS_SHORTLIST


def increment_shortlist_sweep(record: Dict[str, Any]) -> None:
    record["shortlist_sweeps"] = record.get("shortlist_sweeps", 0) + 1


def should_schedule_hash(
    hash_value: Optional[str],
    bucket: str,
    registry: Dict[str, Dict[str, Any]],
    sweep_seen: Set[str],
    allow_duplicates: bool,
    sweep_id: int,
) -> bool:
    if not hash_value:
        return True
    if not allow_duplicates and hash_value in sweep_seen:
        return False
    record = registry.get(hash_value)
    if not record:
        return True
    status = record.get("status", HASH_STATUS_UNTESTED)
    if status == HASH_STATUS_RETIRED:
        return False
    if status == HASH_STATUS_VALIDATING:
        return False
    if status == HASH_STATUS_STABLE:
        if bucket in {"soft-best", "best", "anchor"}:
            last_seen = record.get("last_sweep_id", 0)
            if sweep_id - last_seen >= ANCHOR_COOLOFF_SWEEPS:
                return True
        return False
    if status == HASH_STATUS_SHORTLIST:
        if record.get("shortlist_sweeps", 0) >= MAX_TEN10_SWEEPS_PER_SHORTLIST:
            return False
    return True


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


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
    offense = config.get("rewards", {}).get("edge", {}).get("offense", {})
    snapshot = {
        "blackFinishScaleMultiplier": offense.get("blackFinishScaleMultiplier"),
        "spanGainBase": offense.get("spanGainBase"),
        "gapDecay": offense.get("gapDecay"),
    }
    for key in OPTIONAL_OFFENSE_KEYS:
        snapshot[key] = offense.get(key)
    return snapshot


def extract_combo_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    offense = config.get("rewards", {}).get("edge", {}).get("offense", {})
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


def _prefer_keys(entry: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = entry.get(key)
        if value is not None:
            return value
    return None


def combo_from_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    combo = {
        "firstEdgeRed": _prefer_keys(entry, "firstEdgeRed"),
        "firstEdgeBlack": _prefer_keys(entry, "firstEdgeBlack"),
        "finishPenalty": _prefer_keys(entry, "finishPenalty"),
        "gapDecay": _prefer_keys(entry, "gapDecay"),
        "redFinishPenaltyFactor": _prefer_keys(entry, "redFinishPenaltyFactor"),
        "blackFinishScaleMultiplier": _prefer_keys(entry, "blackFinishScaleMultiplier"),
        "spanGainBase": _prefer_keys(entry, "spanGainBase", "spanBase"),
        "redSpanGainMultiplier": _prefer_keys(
            entry, "redSpanGainMultiplier", "redSpanMult"
        ),
        "blackSpanGainMultiplier": _prefer_keys(
            entry, "blackSpanGainMultiplier", "blackSpanMult"
        ),
        "redDoubleCoverageBonus": _prefer_keys(
            entry, "redDoubleCoverageBonus", "redDoubleCov"
        ),
        "blackDoubleCoverageScale": _prefer_keys(
            entry, "blackDoubleCoverageScale", "blackDoubleCovScale"
        ),
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


def combo_from_validation_config(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    combo = {
        "firstEdgeRed": snapshot.get("firstEdgeTouchRed"),
        "firstEdgeBlack": snapshot.get("firstEdgeTouchBlack"),
        "finishPenalty": snapshot.get("finishPenaltyBase"),
        "redFinishPenaltyFactor": snapshot.get("redFinishPenaltyFactor"),
        "blackFinishScaleMultiplier": snapshot.get("blackFinishScaleMultiplier"),
        "redSpanGainMultiplier": snapshot.get("redSpanGainMultiplier"),
        "blackSpanGainMultiplier": snapshot.get("blackSpanGainMultiplier"),
        "redDoubleCoverageBonus": snapshot.get("redDoubleCoverageBonus"),
        "blackDoubleCoverageScale": snapshot.get("blackDoubleCoverageScale"),
    }
    for key in OPTIONAL_OFFENSE_KEYS:
        combo[key] = snapshot.get(key)
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
    count: float
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


def normalize_knob_value(knob: str, value: Any) -> float:
    spec = KNOB_SPECS.get(knob)  # type: ignore[name-defined]
    if not spec or not spec.values:
        return 0.0
    target = value
    if target not in spec.values:
        target = nearest_value(spec.values, target)  # type: ignore[name-defined]
    if len(spec.values) == 1:
        return 0.0
    idx = spec.values.index(target)
    return idx / (len(spec.values) - 1)


def combo_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    diffs: List[float] = []
    for knob in KNOB_SPECS.keys():  # type: ignore[name-defined]
        if knob not in a or knob not in b:
            continue
        diffs.append(
            abs(
                normalize_knob_value(knob, a[knob])
                - normalize_knob_value(knob, b[knob])
            )
        )
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)


def entry_total_games(entry: Dict[str, Any]) -> int:
    evaluation = entry.get("evaluation") or {}
    depth2 = evaluation.get("depth2") or {}
    depth3 = evaluation.get("depth3") or {}
    return (
        int(depth2.get("red", 0) or 0)
        + int(depth2.get("black", 0) or 0)
        + int(depth2.get("draw", 0) or 0)
        + int(depth3.get("red", 0) or 0)
        + int(depth3.get("black", 0) or 0)
        + int(depth3.get("draw", 0) or 0)
    )


def soft_best_eligible(
    entry: Dict[str, Any], validation_counts: Dict[str, int]
) -> bool:
    games = entry_total_games(entry)
    if games >= SOFT_BEST_MIN_SWEEP_GAMES:
        return True
    cfg_hash = entry.get("configHash")
    return bool(cfg_hash and validation_counts.get(cfg_hash, 0) > 0)


def score_weight(score: Optional[float]) -> float:
    if score is None:
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(score)))


def allocate_category_quota(remaining: int) -> Dict[str, int]:
    if remaining <= 0:
        return {name: 0 for name, _ in CATEGORY_WEIGHTS}
    total_weight = sum(weight for _, weight in CATEGORY_WEIGHTS)
    quotas: Dict[str, int] = {}
    accumulated = 0
    for name, weight in CATEGORY_WEIGHTS:
        share = (remaining * weight) // total_weight
        quotas[name] = share
        accumulated += share
    leftover = remaining - accumulated
    for name, _ in CATEGORY_WEIGHTS:
        if leftover <= 0:
            break
        quotas[name] += 1
        leftover -= 1
    return quotas


KNOB_SPECS: Dict[str, KnobSpec] = {
    "firstEdgeRed": KnobSpec("firstEdgeRed", build_values(410, 440, 5)),
    "firstEdgeBlack": KnobSpec("firstEdgeBlack", build_values(445, 465, 5)),
    "finishPenalty": KnobSpec("finishPenalty", build_values(1161, 1221, 20)),
    "redFinishPenaltyFactor": KnobSpec(
        "redFinishPenaltyFactor", build_values(0.15, 0.8, 0.05)
    ),
    "blackFinishScaleMultiplier": KnobSpec(
        "blackFinishScaleMultiplier", build_values(0.85, 1.1, 0.05)
    ),
    "redSpanGainMultiplier": KnobSpec(
        "redSpanGainMultiplier", build_values(0.9, 1.3, 0.05)
    ),
    "blackSpanGainMultiplier": KnobSpec(
        "blackSpanGainMultiplier", build_values(0.7, 1.1, 0.05)
    ),
    "redDoubleCoverageBonus": KnobSpec(
        "redDoubleCoverageBonus", build_values(0, 1800, 100)
    ),
    "blackDoubleCoverageScale": KnobSpec(
        "blackDoubleCoverageScale", build_values(0.55, 1.0, 0.05)
    ),
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


def mutate_combo(
    base: Dict[str, Any], rng: random.Random, active_knobs: Sequence[str]
) -> Dict[str, Any]:
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


def random_combo(
    rng: random.Random, base: Dict[str, Any], active_knobs: Sequence[str]
) -> Dict[str, Any]:
    result = dict(base)
    if not active_knobs:
        return result
    for knob in active_knobs:
        spec = KNOB_SPECS[knob]
        result[knob] = rng.choice(spec.values)
    return result


def get_bucket_from_origin(origin: Optional[str]) -> str:
    parts = (origin or "").strip().split()
    token = parts[0] if parts else ""
    for bucket in BUCKET_NAMES:
        if not token and bucket == "other":
            return bucket
        if token == bucket or token.startswith(f"{bucket}:"):
            return bucket
    return "other"


def new_bucket_counter() -> Dict[str, Any]:
    return {
        "total": 0,
        "top10": 0,
        "top25": 0,
        "wins": 0,
        "sum_rank": 0,
        "best_rank": None,
        "evalSamples": 0,
        "sumDepth2Parity": 0.0,
        "sumDepth3Parity": 0.0,
        "sumDraws": 0.0,
    }


def ensure_bucket_stats(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stats = state.get("bucketStats")
    if not isinstance(stats, dict):
        stats = {}
    for bucket in BUCKET_NAMES:
        bucket_stats = stats.get(bucket)
        if not isinstance(bucket_stats, dict):
            stats[bucket] = new_bucket_counter()
        else:
            template = new_bucket_counter()
            for key, value in template.items():
                bucket_stats.setdefault(key, value)
    state["bucketStats"] = stats
    return stats


def combo_with_defaults(
    combo: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    result = dict(combo)
    for key in OPTIONAL_OFFENSE_KEYS:
        if result.get(key) is None:
            default_value = defaults.get(key)
            if default_value is not None:
                result[key] = default_value
    for knob in KNOB_SPECS.keys():
        if result.get(knob) is None and defaults.get(knob) is not None:
            result[knob] = defaults[knob]
    if result.get("blackFinishScaleMultiplier") is None:
        result["blackFinishScaleMultiplier"] = defaults.get(
            "blackFinishScaleMultiplier", 1.0
        )
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


def analyze_knob_trends(
    entries: List[Dict[str, Any]],
    validation_samples: Optional[List[Tuple[Dict[str, Any], float]]] = None,
) -> Dict[str, KnobTrend]:
    combined_samples: List[Tuple[Dict[str, Any], float, float]] = []
    for entry in entries:
        combo = entry.get("combo") or {}
        score = sweep_trend_metric(entry)
        if score is None:
            continue
        combined_samples.append((combo, float(score), 1.0))
    if validation_samples:
        for combo, metric in validation_samples:
            combined_samples.append((combo, float(metric), VALIDATION_TREND_WEIGHT))
    trends: Dict[str, KnobTrend] = {}
    for knob, spec in KNOB_SPECS.items():
        value_bucket: Dict[Any, List[Tuple[float, float]]] = {}
        numeric_values: List[float] = []
        numeric_scores: List[float] = []
        for combo, score_f, weight in combined_samples:
            value = combo.get(knob)
            if value is None:
                continue
            repeat = max(1, int(round(weight)))
            for _ in range(repeat):
                numeric_scores.append(score_f)
                numeric_values.append(float(value))
            value_bucket.setdefault(value, []).append((score_f, weight))
        if not value_bucket:
            continue
        if len(set(numeric_values)) > 1:
            slope, corr = compute_regression(numeric_values, numeric_scores)
        else:
            slope, corr = 0.0, 0.0
        value_stats: Dict[Any, KnobValueStats] = {}
        for value, samples in value_bucket.items():
            total_weight = sum(weight for _, weight in samples)
            if not total_weight:
                continue
            avg_score = sum(score * weight for score, weight in samples) / total_weight
            best_score = min(score for score, _ in samples)
            value_stats[value] = KnobValueStats(
                count=total_weight, avg_score=avg_score, best_score=best_score
            )
        top_values = sorted(
            value_stats.keys(),
            key=lambda v: (value_stats[v].avg_score, -value_stats[v].count),
        )
        under_sampled = [
            val
            for val, stat in value_stats.items()
            if stat.count <= UNDER_SAMPLED_COUNT
        ]
        unseen = [val for val in spec.values if val not in value_stats]
        under_sampled.extend(unseen)
        trends[knob] = KnobTrend(
            slope=slope,
            correlation=corr,
            value_stats=value_stats,
            top_values=top_values,
            under_sampled=under_sampled,
        )
    return trends


def generate_trend_variants(
    base_combo: Dict[str, Any],
    trends: Dict[str, KnobTrend],
    rng: random.Random,
    max_count: int,
    active_knobs: Sequence[str],
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = []
    sorted_trends = sorted(
        trends.items(), key=lambda item: abs(item[1].slope), reverse=True
    )
    for knob, data in sorted_trends:
        if knob not in active_knobs:
            continue
        if len(variants) >= max_count:
            break
        if (
            abs(data.slope) < TREND_SLOPE_THRESHOLD
            and abs(data.correlation) < TREND_CORRELATION_THRESHOLD
        ):
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
            if (
                0 <= target_idx < len(spec.values)
                and target_idx not in candidate_indices
            ):
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


def generate_best_value_variants(
    base_combo: Dict[str, Any],
    trends: Dict[str, KnobTrend],
    max_count: int,
    active_knobs: Sequence[str],
) -> List[Tuple[Dict[str, Any], str]]:
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


def _pick_ordered_pool(
    entries: Sequence[Dict[str, Any]],
    target: int,
    taken_hashes: Set[str],
    validation_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    for entry in entries:
        if len(picked) >= target:
            break
        cfg_hash = entry.get("configHash")
        if not cfg_hash or cfg_hash in taken_hashes:
            continue
        if not soft_best_eligible(entry, validation_counts):
            continue
        picked.append(entry)
        taken_hashes.add(cfg_hash)
    return picked


def generate_soft_best_variants(
    base_combo: Dict[str, Any],
    history_entries: Sequence[Dict[str, Any]],
    validation_counts: Dict[str, int],
    recent_timestamps: Set[str],
    rng: random.Random,
    max_count: int,
    active_knobs: Sequence[str],
) -> List[Tuple[Dict[str, Any], str]]:
    scored = [entry for entry in history_entries if entry.get("score") is not None]
    if not scored or not active_knobs:
        return []
    scored.sort(
        key=lambda entry: (entry.get("score", float("inf")), entry.get("timestamp", ""))
    )
    taken_hashes: Set[str] = set()
    recent_entries = [
        entry for entry in scored if entry.get("timestamp") in recent_timestamps
    ]
    pool: List[Dict[str, Any]] = []
    if recent_entries:
        pool.extend(
            _pick_ordered_pool(
                recent_entries, SOFT_BEST_RECENT_TARGET, taken_hashes, validation_counts
            )
        )
    pool.extend(
        _pick_ordered_pool(
            scored,
            SOFT_BEST_LEGEND_TARGET,
            taken_hashes,
            validation_counts,
        )
    )
    if len(pool) < max(SOFT_BEST_POOL_SIZE, max_count):
        pool.extend(
            _pick_ordered_pool(
                scored,
                max(SOFT_BEST_POOL_SIZE, max_count) - len(pool),
                taken_hashes,
                validation_counts,
            )
        )
    if not pool:
        return []
    distributions: Dict[str, Dict[Any, float]] = {}
    for entry in pool:
        combo = entry.get("combo") or {}
        weight = score_weight(entry.get("score"))
        if weight <= 0:
            weight = 0.1
        for knob in active_knobs:
            value = combo.get(knob)
            if value is None:
                continue
            bucket = distributions.setdefault(knob, {})
            bucket[value] = bucket.get(value, 0.0) + weight
    normalized: Dict[str, Dict[Any, float]] = {}
    for knob, counts in distributions.items():
        if not counts:
            continue
        spec = KNOB_SPECS[knob]
        if len(counts) == 1 and len(spec.values) > 1:
            only_value = next(iter(counts))
            base_weight = counts[only_value]
            idx = spec.values.index(nearest_value(spec.values, only_value))
            neighbors: List[int] = []
            if idx > 0:
                neighbors.append(idx - 1)
            if idx < len(spec.values) - 1:
                neighbors.append(idx + 1)
            for neighbor_idx in neighbors:
                neighbor_value = spec.values[neighbor_idx]
                counts.setdefault(
                    neighbor_value,
                    max(base_weight * SOFT_BEST_MIN_VARIANCE, 0.05),
                )
        normalized[knob] = counts
    if not normalized:
        return []
    variants: List[Tuple[Dict[str, Any], str]] = []
    attempts = 0
    max_attempts = max_count * 4
    while len(variants) < max_count and attempts < max_attempts:
        attempts += 1
        candidate = dict(base_combo)
        for knob, counts in normalized.items():
            values = list(counts.keys())
            weights = [counts[val] for val in values]
            total = sum(weights)
            if total <= 0:
                continue
            choice = rng.choices(values, weights=weights, k=1)[0]
            candidate[knob] = choice
        variants.append((candidate, "soft-best"))
    return variants


def hill_climb_step(
    source_combo: Dict[str, Any],
    trends: Dict[str, KnobTrend],
    rng: random.Random,
    active_knobs: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not active_knobs:
        return None
    candidate = dict(source_combo)
    change_count = 1
    changed = False
    for knob in rng.sample(list(active_knobs), change_count):
        spec = KNOB_SPECS[knob]
        values = spec.values
        if not values:
            continue
        current = candidate.get(knob, values[0])
        if current not in values:
            current = nearest_value(values, current)
        idx = values.index(current)
        trend = trends.get(knob)
        direction = 0
        if trend and abs(trend.slope) >= TREND_SLOPE_THRESHOLD:
            direction = -1 if trend.slope > 0 else 1
        if direction == 0:
            direction = rng.choice([-1, 1])
        new_idx = idx + direction
        if not (0 <= new_idx < len(values)):
            continue
        new_value = values[new_idx]
        if new_value == current:
            continue
        normalized_current = normalize_knob_value(knob, current)
        normalized_new = normalize_knob_value(knob, new_value)
        delta = abs(normalized_new - normalized_current)
        if delta > NICHE_MAX_NORMALIZED_DELTA and knob not in COARSE_NICHE_KNOBS:
            continue
        candidate[knob] = new_value
        changed = True
    if not changed:
        return None
    return candidate


def generate_niche_hill_climb_variants(
    champions: Sequence[Dict[str, Any]],
    trends: Dict[str, KnobTrend],
    rng: random.Random,
    required_count: int,
    slot_retries: int,
    active_knobs: Sequence[str],
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = []
    if not champions or not active_knobs or required_count <= 0:
        return variants
    pool = [champ for champ in champions if champ.get("combo")]
    if not pool:
        return variants
    rng.shuffle(pool)
    niche_centers: List[Dict[str, Any]] = []
    champion_index = 0
    while len(variants) < required_count and pool:
        champion = pool[champion_index % len(pool)]
        champion_index += 1
        source_combo = champion.get("combo") or {}
        success = False
        attempts = 0
        while attempts < slot_retries:
            attempts += 1
            candidate = hill_climb_step(source_combo, trends, rng, active_knobs)
            if not candidate:
                continue
            if any(
                combo_distance(candidate, center) < NICHE_DISTANCE_THRESHOLD
                for center in niche_centers
            ):
                continue
            variants.append((candidate, f"niche:{champion.get('configHash', '')[:4]}"))
            niche_centers.append(candidate)
            success = True
            break
        if not success:
            pool.remove(champion)
    return variants


def generate_underexplored_variants(
    base_combo: Dict[str, Any],
    trends: Dict[str, KnobTrend],
    rng: random.Random,
    max_count: int,
    active_knobs: Sequence[str],
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = []
    knob_order = [knob for knob in KNOB_SPECS.keys() if knob in active_knobs]
    if not knob_order:
        return variants
    attempts = 0
    max_attempts = max_count * 4
    while len(variants) < max_count and attempts < max_attempts:
        attempts += 1
        prioritized = [
            knob
            for knob in knob_order
            if trends.get(knob) and trends[knob].under_sampled
        ]
        pool = prioritized if prioritized else knob_order
        change_knob_count = min(len(pool), rng.choice([1, 2]))
        selected_knobs = rng.sample(pool, change_knob_count)
        candidate = dict(base_combo)
        changed = False
        for knob in selected_knobs:
            spec = KNOB_SPECS[knob]
            data = trends.get(knob)
            candidates: List[Any] = []
            if data and data.under_sampled:
                candidates.extend(
                    [val for val in data.under_sampled if val is not None]
                )
            else:
                candidates.extend(spec.values)
            if not candidates:
                continue
            rng.shuffle(candidates)
            for value in candidates:
                if candidate.get(knob) == value:
                    continue
                candidate[knob] = value
                changed = True
                break
        if not changed:
            continue
        label = "explore:" + ",".join(selected_knobs)
        variants.append((candidate, label))
    return variants


def update_bucket_stats(state: Dict[str, Any], sweep: Dict[str, Any]) -> None:
    planned_origins = state.get("plannedOrigins") or {}
    combos = sweep.get("combos", [])
    scored = [combo for combo in combos if isinstance(combo.get("score"), (int, float))]
    if not scored:
        return
    scored.sort(
        key=lambda combo: (
            combo.get("score", float("inf")),
            combo.get("configHash") or "",
        )
    )
    bucket_stats = ensure_bucket_stats(state)
    total = len(scored)
    top10_cut = max(1, math.ceil(0.10 * total))
    top25_cut = max(1, math.ceil(0.25 * total))
    for idx, combo in enumerate(scored, start=1):
        origin = combo.get("origin")
        if not origin and combo.get("configHash"):
            origin = planned_origins.get(combo["configHash"])
        bucket = get_bucket_from_origin(origin)
        stats = bucket_stats.setdefault(bucket, new_bucket_counter())
        stats["total"] += 1
        stats["sum_rank"] += idx
        if idx <= top10_cut:
            stats["top10"] += 1
        if idx <= top25_cut:
            stats["top25"] += 1
        if idx == 1:
            stats["wins"] += 1
        best_rank = stats.get("best_rank")
        if best_rank is None or idx < best_rank:
            stats["best_rank"] = idx
        evaluation = combo.get("evaluation")
        if evaluation:
            metrics = compute_metrics(
                {
                    "evaluation": evaluation,
                    "score": combo.get("score"),
                }
            )
            if metrics["totalGames"] > 0:
                stats["evalSamples"] = stats.get("evalSamples", 0) + 1
                stats["sumDepth2Parity"] = (
                    stats.get("sumDepth2Parity", 0.0) + metrics["depth2Parity"]
                )
                stats["sumDepth3Parity"] = (
                    stats.get("sumDepth3Parity", 0.0) + metrics["depth3Parity"]
                )
                stats["sumDraws"] = stats.get("sumDraws", 0.0) + metrics["totalDraw"]


def rebuild_bucket_stats(state: Dict[str, Any], sweeps: List[Dict[str, Any]]) -> None:
    original_planned = state.get("plannedOrigins")
    state["bucketStats"] = {bucket: new_bucket_counter() for bucket in BUCKET_NAMES}
    state["plannedOrigins"] = {}
    for sweep in sweeps:
        update_bucket_stats(state, sweep)
    if original_planned is not None:
        state["plannedOrigins"] = original_planned
    else:
        state.pop("plannedOrigins", None)


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


def flatten_sweeps(
    sweeps: Iterable[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
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
    depth2_parity = depth2["imbalance"]
    depth3_parity = depth3["imbalance"]
    return {
        "score": combo_entry.get("score"),
        "totalGames": total_games,
        "totalRed": total_red,
        "totalBlack": total_black,
        "totalDraw": total_draw,
        "parity": abs(total_red - total_black),
        "depth2Parity": depth2_parity,
        "depth3Parity": depth3_parity,
        "weightedParity": depth2_parity + DEPTH3_PARITY_WEIGHT * depth3_parity,
        "depth2": depth2,
        "depth3": depth3,
    }


def sweep_trend_metric(entry: Dict[str, Any]) -> Optional[float]:
    metrics = compute_metrics(entry)
    if metrics["totalGames"] > 0:
        return (
            metrics["depth2Parity"]
            + DEPTH3_PARITY_WEIGHT * metrics["depth3Parity"]
            + SWEEP_TREND_DRAW_WEIGHT * metrics["totalDraw"]
        )
    score = entry.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def validation_parity_metric(run: Dict[str, Any]) -> float:
    depth2 = run.get("depth2", {}) or {}
    depth3 = run.get("depth3", {}) or {}
    parity2 = abs(int(depth2.get("red", 0) or 0) - int(depth2.get("black", 0) or 0))
    parity3 = abs(int(depth3.get("red", 0) or 0) - int(depth3.get("black", 0) or 0))
    draws = int(depth2.get("draw", 0) or 0) + int(depth3.get("draw", 0) or 0)
    return parity2 + DEPTH3_PARITY_WEIGHT * parity3 + VALIDATION_DRAW_WEIGHT * draws


def build_validation_trend_samples(
    validations: List[Dict[str, Any]],
    defaults: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], float]]:
    samples: List[Tuple[Dict[str, Any], float]] = []
    for run in validations:
        snapshot = run.get("config")
        if not snapshot:
            continue
        combo = combo_from_validation_config(snapshot)
        combo = combo_with_defaults(combo, defaults)
        if not combo_has_all_knobs(combo):
            continue
        metric = validation_parity_metric(run)
        samples.append((combo, metric))
    return samples


def load_state() -> Dict[str, Any]:
    return load_json(STATE_PATH, {})


def save_state(state: Dict[str, Any]) -> None:
    write_json(STATE_PATH, state)


def reset_stall_state(
    state: Dict[str, Any], *, thaw_knobs: bool = False
) -> Dict[str, Any]:
    state["cyclesSinceScoreImprovement"] = 0
    state["cyclesSinceStreakImprovement"] = 0
    if thaw_knobs:
        knob_stats = state.get("knobStats", {})
        for knob, stats in knob_stats.items():
            stats["cyclesSinceImprovement"] = 0
            stats["frozen"] = False
        state["knobStats"] = knob_stats
        planned = state.get("plannedSweep", {})
        planned["hashes"] = []
        planned["generatedAt"] = None
        state["plannedSweep"] = planned
    save_state(state)
    return state


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
    scored_entries = [
        entry for entry in history_entries if entry.get("score") is not None
    ]
    scored_entries.sort(
        key=lambda entry: (entry.get("score", float("inf")), entry.get("timestamp", ""))
    )
    for entry in scored_entries:
        combo = combo_with_defaults(entry["combo"], baseline_defaults)
        if combo_has_all_knobs(combo):
            return combo, entry.get("configHash")
    fallback = combo_with_defaults(baseline_combo, baseline_defaults)
    return fallback, None


def command_suggest(args: argparse.Namespace) -> None:
    dry_run = getattr(args, "dry_run", False)
    rng = random.Random(
        args.seed if args.seed is not None else random.randrange(1 << 30)
    )
    sweeps = load_sweeps()
    search_config = load_search_config()
    baseline_defaults = offense_snapshot(search_config)
    flattened = flatten_sweeps(sweeps, baseline_defaults)
    state = load_state()
    knob_stats = state.get("knobStats", {})
    frozen_knobs = {knob for knob, stats in knob_stats.items() if stats.get("frozen")}
    frozen_values = {
        knob: stats.get("bestValue")
        for knob, stats in knob_stats.items()
        if stats.get("frozen") and stats.get("bestValue") is not None
    }
    active_knobs = [
        knob
        for knob in KNOB_SPECS.keys()
        if knob not in frozen_knobs and len(KNOB_SPECS[knob].values) > 1
    ]
    baseline_combo_raw = extract_combo_from_config(search_config)
    for knob, value in frozen_values.items():
        baseline_combo_raw[knob] = value
    validations = load_validations()
    validation_samples = build_validation_trend_samples(validations, baseline_defaults)
    requested = args.count
    wishlist: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    mutate_success = 0
    hash_registry = ensure_hash_registry(state)
    next_sweep_id = state.get("lastSweepId", 0) + 1
    validation_counts = compute_validation_counts(validations, baseline_defaults)
    history_entries = flattened
    analytics_entries = [
        entry for entry in history_entries if entry.get("score") is not None
    ]
    trends = analyze_knob_trends(analytics_entries, validation_samples)
    base_combo, base_hash = select_base_combo(
        baseline_combo_raw, baseline_defaults, history_entries, validations
    )
    for knob, value in frozen_values.items():
        base_combo[knob] = value

    def normalize_combo(
        raw_combo: Dict[str, Any],
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        materialized = combo_with_defaults(raw_combo, baseline_defaults)
        for knob, value in frozen_values.items():
            materialized[knob] = value
        if not combo_has_all_knobs(materialized):
            return None
        return materialized, compute_config_hash(materialized)

    parent_norm = normalize_combo(base_combo)
    if not parent_norm:
        raise SystemExit("Baseline combo missing required knobs after normalization.")
    parent_materialized, parent_hash = parent_norm

    def append_candidate(
        candidate_combo: Dict[str, Any],
        origin: str,
        *,
        score: Optional[float] = None,
        source: Optional[str] = None,
    ) -> bool:
        nonlocal wishlist, seen_hashes, mutate_success
        norm = normalize_combo(candidate_combo)
        if not norm:
            return False
        materialized, combo_hash = norm
        enriched = attach_hash(materialized)
        hash_value = enriched["configHash"]
        bucket = get_bucket_from_origin(origin)
        if not should_schedule_hash(
            hash_value,
            bucket,
            hash_registry,
            seen_hashes,
            bool(args.allow_duplicates),
            next_sweep_id,
        ):
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
        if origin.startswith("mutate"):
            mutate_success += 1
        return True

    def schedule_anchor_configs(limit: int) -> None:
        if limit <= 0:
            return
        candidates: List[Tuple[float, str]] = []
        for cfg_hash, record in hash_registry.items():
            if record.get("status") != HASH_STATUS_STABLE:
                continue
            last_seen = record.get("last_sweep_id", 0)
            if next_sweep_id - last_seen < ANCHOR_COOLOFF_SWEEPS:
                continue
            best_score = record.get("score_best")
            metric = (
                best_score if isinstance(best_score, (int, float)) else float("inf")
            )
            candidates.append((metric, cfg_hash))
        candidates.sort()
        for _, cfg_hash in candidates:
            if len(wishlist) >= requested or limit <= 0:
                break
            entry = find_combo_by_hash(cfg_hash)
            if not entry:
                continue
            combo = combo_with_defaults(combo_from_log(entry), baseline_defaults)
            if not combo_has_all_knobs(combo):
                continue
            if append_candidate(
                combo,
                "anchor",
                score=entry.get("score"),
                source=entry.get("timestamp"),
            ):
                limit -= 1

    requested = args.count
    exploitation_target = max(0, min(args.exploit, requested))
    flattened_sorted = sorted(
        history_entries,
        key=lambda entry: (
            entry.get("score", float("inf")),
            entry.get("timestamp", ""),
        ),
    )
    for entry in flattened_sorted:
        if len(wishlist) >= exploitation_target:
            break
        hash_value = entry.get("configHash")
        if not hash_value or hash_value in seen_hashes:
            continue
        combo = entry["combo"]
        append_candidate(
            combo, "exploit", score=entry.get("score"), source=entry.get("timestamp")
        )

    anchor_quota = min(ANCHOR_SLOTS_PER_SWEEP, requested - len(wishlist))
    if anchor_quota > 0:
        schedule_anchor_configs(anchor_quota)

    remaining = requested - len(wishlist)
    category_quota = allocate_category_quota(remaining)
    champion_entries = [
        entry for entry in flattened_sorted if entry.get("score") is not None
    ][:NICHE_TOP_CHAMPIONS]

    timestamp_values = sorted(
        {
            entry.get("timestamp")
            for entry in history_entries
            if entry.get("timestamp") is not None
        }
    )
    recent_timestamps = set(timestamp_values[-SOFT_BEST_RECENT_SWEEPS:])

    bucket_counts = {bucket: 0 for bucket in BUCKET_NAMES}

    soft_quota = category_quota.get("soft_best", 0)
    if soft_quota > 0:
        soft_variants = generate_soft_best_variants(
            base_combo,
            analytics_entries,
            validation_counts,
            recent_timestamps,
            rng,
            soft_quota * 3,
            active_knobs,
        )
        for combo, origin in soft_variants:
            if len(wishlist) >= requested or soft_quota <= 0:
                break
            if append_candidate(combo, origin):
                bucket_counts["soft-best"] += 1
                soft_quota -= 1
        category_quota["soft_best"] = soft_quota

    niche_quota = category_quota.get("niche", 0)
    if niche_quota > 0:
        niche_variants = generate_niche_hill_climb_variants(
            champion_entries,
            trends,
            rng,
            niche_quota,
            NICHE_SLOT_RETRIES,
            active_knobs,
        )
        for combo, origin in niche_variants:
            if len(wishlist) >= requested or niche_quota <= 0:
                break
            if append_candidate(combo, origin):
                bucket_counts["niche"] += 1
                niche_quota -= 1
        category_quota["niche"] = niche_quota

    trend_quota = category_quota.get("trend", 0)
    if trend_quota > 0:
        for combo, origin in generate_trend_variants(
            base_combo, trends, rng, trend_quota * 2, active_knobs
        ):
            if len(wishlist) >= requested or trend_quota <= 0:
                break
            if append_candidate(combo, origin):
                bucket_counts["trend"] += 1
                trend_quota -= 1
        category_quota["trend"] = trend_quota

    best_quota = min(category_quota.get("best", 0), BEST_MAX_COUNT)
    best_added = 0
    if best_quota > 0:
        best_variants = generate_best_value_variants(
            base_combo, trends, best_quota * 2, active_knobs
        )
        for combo, origin in best_variants:
            if (
                len(wishlist) >= requested
                or best_quota <= 0
                or best_added >= BEST_MAX_COUNT
            ):
                break
            if append_candidate(combo, origin):
                best_quota -= 1
                best_added += 1
                bucket_counts["best"] += 1
        category_quota["best"] = best_quota
        if best_variants and best_added < BEST_MAX_COUNT:
            ranked_choices: List[Tuple[float, int, str, Any]] = []
            for knob, data in trends.items():
                if knob not in active_knobs or not data.top_values:
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
                if best_added < BEST_MAX_COUNT and append_candidate(
                    combined, "best:mixed"
                ):
                    bucket_counts["best"] += 1
                    best_added += 1

    explore_quota = category_quota.get("explore", 0)
    if explore_quota > 0:
        for combo, origin in generate_underexplored_variants(
            base_combo, trends, rng, explore_quota * 2, active_knobs
        ):
            if len(wishlist) >= requested or explore_quota <= 0:
                break
            if append_candidate(combo, origin):
                bucket = get_bucket_from_origin(origin)
                bucket_counts[bucket] += 1
                explore_quota -= 1
        category_quota["explore"] = explore_quota

    forced_attempts = 0
    while (
        mutate_success < MUTATE_MIN_COUNT
        and len(wishlist) < requested
        and active_knobs
        and forced_attempts < MUTATE_MIN_COUNT * 5
    ):
        forced_attempts += 1
        appended = False
        for _ in range(MUTATION_MAX_ATTEMPTS):
            candidate = mutate_combo(base_combo, rng, active_knobs)
            norm = normalize_combo(candidate)
            if not norm:
                continue
            _, candidate_hash = norm
            if candidate_hash == parent_hash:
                continue
            if append_candidate(candidate, "mutate"):
                bucket_counts["mutate"] += 1
                appended = True
                break
        if not appended:
            continue

    filler_attempts = 0
    mutate_quota = max(
        category_quota.get("filler", 0), MUTATE_MIN_COUNT - mutate_success
    )
    while len(wishlist) < requested:
        if not active_knobs:
            break
        use_mutate = mutate_quota > 0 or rng.random() < 0.6
        if use_mutate:
            appended = False
            for _ in range(MUTATION_MAX_ATTEMPTS):
                candidate = mutate_combo(base_combo, rng, active_knobs)
                norm = normalize_combo(candidate)
                if not norm:
                    continue
                _, candidate_hash = norm
                if candidate_hash == parent_hash:
                    continue
                if append_candidate(candidate, "mutate"):
                    bucket_counts["mutate"] += 1
                    appended = True
                    break
            if appended:
                if mutate_quota > 0:
                    mutate_quota -= 1
                continue
        else:
            candidate = random_combo(rng, base_combo, active_knobs)
            origin = "random"
            if append_candidate(candidate, origin):
                bucket = get_bucket_from_origin(origin)
                bucket_counts[bucket] += 1
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
    planned_origins = {
        combo["configHash"]: combo.get("origin")
        for combo in wishlist
        if combo.get("configHash")
    }
    if dry_run:
        print(
            f"[dry-run] Would write {len(wishlist)} combos to {NEXT_SWEEP_PATH.relative_to(PROJECT_ROOT)}"
        )
    else:
        write_json(NEXT_SWEEP_PATH, payload)
        state["lastSuggested"] = payload["generatedAt"]
        state["plannedSweep"] = {
            "hashes": [c["configHash"] for c in wishlist],
            "generatedAt": payload["generatedAt"],
        }
        state["plannedOrigins"] = planned_origins
        save_state(state)
        print(
            f"Wrote {len(wishlist)} combos to {NEXT_SWEEP_PATH.relative_to(PROJECT_ROOT)}"
        )
    for idx, combo in enumerate(wishlist, 1):
        print(
            f"[{idx:02d}] {combo['origin']:>7} {combo['configHash'][:8]}  {render_combo(combo)}"
        )


def command_sweep(args: Optional[argparse.Namespace] = None) -> bool:
    dry_run = getattr(args, "dry_run", False) if args else False
    stop_flag = getattr(args, "stop_requested", False) if args else False
    if dry_run:
        print("[dry-run] Skipping tuneBaseline.js")
        return True
    cmd = ["node", "scripts/tuneBaseline.js"]
    print("Running:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode == 0:
            return True
        if stop_flag or result.returncode == 1:
            print("[loop] Sweep aborted before completion.")
            return False
        raise subprocess.CalledProcessError(result.returncode, cmd)
    except subprocess.CalledProcessError as exc:
        if exc.returncode < 0 or stop_flag:
            print(f"[loop] Sweep aborted (return code {exc.returncode}).")
            return False
        raise SystemExit(exc.returncode)


def recommend_for_validation(
    all_entries: List[Dict[str, Any]],
    limit: int,
    validated_hashes: set[str],
    streaks: Dict[str, int],
    validation_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    best_entries: Dict[str, Dict[str, Any]] = {}
    for entry in all_entries:
        cfg_hash = entry.get("configHash")
        score = entry.get("score")
        timestamp = entry.get("timestamp", "")
        if not cfg_hash or score is None:
            continue
        if cfg_hash in validated_hashes:
            continue
        if score > VALIDATION_SCORE_CUTOFF:
            continue
        metrics = compute_metrics(entry)
        current = best_entries.get(cfg_hash)
        if (
            current is None
            or score < current["entry"].get("score", float("inf"))
            or (
                score == current["entry"].get("score")
                and timestamp > current["entry"].get("timestamp", "")
            )
        ):
            best_entries[cfg_hash] = {"entry": entry, "metrics": metrics}

    ranked = sorted(
        best_entries.items(),
        key=lambda item: (
            0 if streaks.get(item[0], 0) > 0 else 1,
            item[1]["metrics"]["depth3Parity"],
            item[1]["metrics"]["depth2Parity"],
            item[1]["metrics"]["weightedParity"],
            item[1]["entry"].get("score", float("inf")),
            item[1]["entry"].get("timestamp", ""),
        ),
    )

    recommendations: List[Dict[str, Any]] = []
    for cfg_hash, data in ranked:
        entry = data["entry"]
        metrics = data["metrics"]
        if streaks.get(cfg_hash, 0) == 0 and validation_counts.get(cfg_hash, 0) > 0:
            continue
        if metrics["depth3Parity"] > DEPTH3_VALIDATION_MAX_PARITY:
            continue
        if metrics["totalGames"] <= 0 and streaks.get(cfg_hash, 0) == 0:
            continue
        recommendations.append(
            {
                "configHash": cfg_hash,
                "sweepTimestamp": entry["timestamp"],
                "score": entry.get("score"),
                "metrics": metrics,
                "combo": entry["combo"],
            }
        )
        if len(recommendations) >= limit:
            break

    return recommendations


def compute_validation_counts(
    validations: List[Dict[str, Any]],
    defaults: Dict[str, Any],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for run in validations:
        config_hash = run.get("configHash")
        if not config_hash:
            continue
        normalized_hash = config_hash
        config_snapshot = run.get("config") or {}
        if config_snapshot:
            payload = {}
            for key in HASH_KEYS:
                value = config_snapshot.get(key)
                if value is None:
                    value = defaults.get(key)
                payload[key] = value
            normalized_hash = compute_config_hash(payload)
        if normalized_hash:
            counts[normalized_hash] = counts.get(normalized_hash, 0) + 1
    return counts


def outstanding_validation_hashes(
    score_cutoff: int,
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    defaults = offense_snapshot(load_search_config())
    validations = load_validations()
    streaks = compute_validation_streaks(validations, defaults)
    counts = compute_validation_counts(validations, defaults)
    sweeps = load_sweeps()
    flattened = flatten_sweeps(sweeps, defaults)
    best_by_hash: Dict[str, Dict[str, Any]] = {}
    for entry in flattened:
        score = entry.get("score")
        config_hash = entry.get("configHash")
        timestamp = entry.get("timestamp")
        if config_hash is None or score is None:
            continue
        if score > score_cutoff:
            continue
        current = best_by_hash.get(config_hash)
        if (
            current is None
            or score < current["score"]
            or (score == current["score"] and timestamp > current["timestamp"])
        ):
            best_by_hash[config_hash] = {"score": score, "timestamp": timestamp}
    outstanding: List[Tuple[int, float, str, str]] = []
    for cfg_hash, info in best_by_hash.items():
        streak = streaks.get(cfg_hash, 0)
        if streak >= SUCCESS_VALIDATION_STREAK:
            continue
        if counts.get(cfg_hash, 0) > 0 and streak == 0:
            continue
        outstanding.append(
            (
                -streak,
                info["score"],
                info["timestamp"],
                cfg_hash,
            )
        )
    outstanding.sort()
    ordered = [cfg_hash for _, _, _, cfg_hash in outstanding]
    return ordered, streaks, counts


def command_update(args: argparse.Namespace) -> None:
    dry_run = getattr(args, "dry_run", False)
    sweeps = load_sweeps()
    state = load_state()
    rebuild_requested = getattr(args, "rebuild_telemetry", False)
    if rebuild_requested:
        rebuild_bucket_stats(state, sweeps)
        print("Bucket telemetry rebuilt from full sweep history.")
    last_processed = state.get("lastProcessedSweep")
    new_sweeps = [
        sweep
        for sweep in sweeps
        if not last_processed
        or parse_iso(sweep.get("timestamp", "")) > parse_iso(last_processed)
    ]
    baseline_defaults = offense_snapshot(load_search_config())
    if not new_sweeps:
        print("No new sweep results to process.")
        if rebuild_requested:
            save_state(state)
        return

    latest_timestamp = new_sweeps[-1].get("timestamp")
    flattened = flatten_sweeps(new_sweeps, baseline_defaults)
    all_flattened = flatten_sweeps(sweeps, baseline_defaults)
    hash_registry = ensure_hash_registry(state)
    next_sweep_id = state.get("lastSweepId", 0)
    sweep_replay_summaries: List[Tuple[str, int, int, Counter[str]]] = []
    for sweep in new_sweeps:
        sweep_timestamp = sweep.get("timestamp")
        next_sweep_id += 1
        combos = sweep.get("combos", [])
        unique_count = 0
        reused_count = 0
        reused_by_status: Counter[str] = Counter()
        for combo in combos:
            cfg_hash = combo.get("configHash")
            if not cfg_hash:
                continue
            record = hash_registry.setdefault(cfg_hash, default_hash_record())
            prev_sweeps = record.get("ten10_sweeps", 0)
            if prev_sweeps > 0:
                reused_count += 1
                reused_by_status[record.get("status", HASH_STATUS_UNTESTED)] += 1
            else:
                unique_count += 1
            record["ten10_sweeps"] = record.get("ten10_sweeps", 0) + 1
            record["last_sweep_id"] = next_sweep_id
            record["last_score_timestamp"] = sweep_timestamp
            score = combo.get("score")
            if isinstance(score, (int, float)):
                record["score_last"] = score
                best = record.get("score_best")
                if best is None or score < best:
                    record["score_best"] = score
            evaluation = combo.get("evaluation") or {}
            games_d2, games_d3 = evaluation_game_counts(evaluation)
            record["games_d2"] = record.get("games_d2", 0) + games_d2
            record["games_d3"] = record.get("games_d3", 0) + games_d3
            prev_status = record.get("status", HASH_STATUS_UNTESTED)
            if prev_status not in FINAL_STATUSES:
                strong_score = isinstance(score, (int, float)) and (
                    score <= SHORTLIST_SCORE_THRESHOLD
                )
                if strong_score and prev_status != HASH_STATUS_SHORTLIST:
                    promote_to_shortlist(record)
                elif not strong_score:
                    best_value = record.get("score_best")
                    best_value = best_value if best_value is not None else float("inf")
                    if (
                        record.get("ten10_sweeps", 0) >= RETIRE_AFTER_SWEEPS
                        and best_value >= RETIRE_SCORE_THRESHOLD
                    ):
                        set_hash_status(record, HASH_STATUS_RETIRED)
            current_status = record.get("status", HASH_STATUS_UNTESTED)
            if current_status == HASH_STATUS_SHORTLIST:
                if prev_status == HASH_STATUS_SHORTLIST:
                    increment_shortlist_sweep(record)
                elif prev_status != HASH_STATUS_SHORTLIST:
                    record["shortlist_sweeps"] = 1
        update_bucket_stats(state, sweep)
        planned = state.get("plannedOrigins")
        if isinstance(planned, dict):
            for combo in sweep.get("combos", []):
                cfg_hash = combo.get("configHash")
                if cfg_hash and cfg_hash in planned:
                    planned.pop(cfg_hash, None)
        sweep_replay_summaries.append(
            (
                sweep_timestamp or "(unknown)",
                unique_count,
                reused_count,
                reused_by_status,
            )
        )
    validated_runs = load_validations()
    streaks = compute_validation_streaks(validated_runs, baseline_defaults)
    counts = compute_validation_counts(validated_runs, baseline_defaults)
    successful_hashes = {
        hash_value
        for hash_value, streak in streaks.items()
        if streak >= SUCCESS_VALIDATION_STREAK
    }
    failed_hashes = {
        hash_value
        for hash_value, total in counts.items()
        if total > 0 and streaks.get(hash_value, 0) == 0
    }
    skip_hashes = successful_hashes.union(failed_hashes)
    for hash_value, total in counts.items():
        record = hash_registry.setdefault(hash_value, default_hash_record())
        record["validation_runs"] = total
    for hash_value in successful_hashes:
        record = hash_registry.setdefault(hash_value, default_hash_record())
        set_hash_status(record, HASH_STATUS_STABLE)
    for hash_value in failed_hashes:
        record = hash_registry.setdefault(hash_value, default_hash_record())
        set_hash_status(record, HASH_STATUS_RETIRED)
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
            best_score_cycle = (
                score if best_score_cycle is None else min(best_score_cycle, score)
            )
    current_best_streak = max(streaks.values()) if streaks else 0

    knob_stats = state.get("knobStats", {})
    newly_frozen: List[str] = []
    for knob in KNOB_SPECS.keys():
        stats = knob_stats.get(
            knob,
            {
                "bestScore": None,
                "bestValue": None,
                "cyclesSinceImprovement": 0,
                "frozen": False,
            },
        )
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
            if best_entry and (
                stats.get("bestScore") is None
                or best_entry["score"] < stats["bestScore"]
            ):
                stats["bestScore"] = best_entry["score"]
                stats["bestValue"] = best_entry["value"]
                stats["cyclesSinceImprovement"] = 0
            else:
                stats["cyclesSinceImprovement"] = (
                    stats.get("cyclesSinceImprovement", 0) + 1
                )
                if stats.get("bestScore") == 0:
                    stats["cyclesSinceImprovement"] = 0
                if stats["cyclesSinceImprovement"] >= KNOB_STALL_CYCLES:
                    stats["frozen"] = True
        knob_stats[knob] = stats
        if not was_frozen and stats.get("frozen"):
            newly_frozen.append(knob)
    state["knobStats"] = knob_stats
    if newly_frozen:
        print("\nKnobs frozen due to plateau:", ", ".join(newly_frozen))

    state["validationStreaks"] = streaks
    state["successfulConfigs"] = {
        hash_value: streaks[hash_value] for hash_value in successful_hashes
    }

    prev_best_score = state.get("bestScore")
    prev_best_hashes = set(state.get("bestScoreHashes", []))
    if best_score_cycle is not None:
        current_best_hashes = {
            entry.get("configHash")
            for entry in flattened
            if entry.get("configHash") and entry.get("score") == best_score_cycle
        }
        improvement = False
        if prev_best_score is None or best_score_cycle < prev_best_score:
            improvement = True
        elif current_best_hashes and current_best_hashes - prev_best_hashes:
            improvement = True
        elif prev_best_score == best_score_cycle == 0:
            improvement = True
        if improvement:
            state["bestScore"] = best_score_cycle
            state["bestScoreHashes"] = sorted(current_best_hashes)
            state["cyclesSinceScoreImprovement"] = 0
        else:
            state["cyclesSinceScoreImprovement"] = (
                state.get("cyclesSinceScoreImprovement", 0) + 1
            )
    else:
        state.setdefault("bestScoreHashes", sorted(prev_best_hashes))

    prev_best_streak = state.get("bestStreak", 0)
    if current_best_streak > prev_best_streak:
        state["bestStreak"] = current_best_streak
        state["cyclesSinceStreakImprovement"] = 0
    else:
        state["cyclesSinceStreakImprovement"] = (
            state.get("cyclesSinceStreakImprovement", 0) + 1
        )

    validation_counts = counts

    eligible_hashes = {
        hash_value
        for hash_value, streak in streaks.items()
        if streak >= SUCCESS_VALIDATION_STREAK
    }
    failed_hashes = {
        hash_value
        for hash_value, count in validation_counts.items()
        if count > 0 and streaks.get(hash_value, 0) == 0
    }
    skip_hashes = eligible_hashes.union(successful_hashes).union(failed_hashes)

    validation_limit = min(VALIDATION_MAX_PER_CYCLE, max(args.limit, 0))
    recommendations = recommend_for_validation(
        all_flattened, validation_limit, skip_hashes, streaks, counts
    )
    for rec in recommendations:
        cfg_hash = rec.get("configHash")
        if not cfg_hash:
            continue
        record = hash_registry.setdefault(cfg_hash, default_hash_record())
        if record.get("status") != HASH_STATUS_STABLE:
            set_hash_status(record, HASH_STATUS_VALIDATING)
        record["validations_requested"] = record.get("validations_requested", 0) + 1
    pending_payload = {
        "generatedAt": iso_now(),
        "recommendations": recommendations,
    }
    if dry_run:
        generated = pending_payload.get("generatedAt")
        print(
            f"[dry-run] Would update pending-validation.json at {generated} with {len(recommendations)} entries"
        )
    else:
        write_json(PENDING_VALIDATION_PATH, pending_payload)

    if dry_run:
        print("[dry-run] Would update autoTune-state with new sweep metadata")
    else:
        state["lastProcessedSweep"] = latest_timestamp
        state["pendingValidation"] = [entry["configHash"] for entry in recommendations]
        state["lastSweepId"] = next_sweep_id
        state["hashRegistry"] = hash_registry
        save_state(state)

    if sweep_replay_summaries:
        print("\nSweep replay summary:")
        for ts, unique_count, reused_count, reused_by_status in sweep_replay_summaries:
            detail = ", ".join(
                f"{status}:{count}"
                for status, count in sorted(reused_by_status.items())
                if count
            )
            if not detail:
                detail = "—"
            print(f"  {ts}: unique={unique_count} reused={reused_count} ({detail})")

    print(
        f"Processed {len(new_sweeps)} new sweep batch(es). Latest timestamp: {latest_timestamp}"
    )
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

    bucket_stats = state.get("bucketStats", {})
    if bucket_stats:
        print("\nBucket promotion summary (lifetime):")
        for bucket in BUCKET_NAMES:
            stats = bucket_stats.get(bucket)
            if not stats or not stats.get("total"):
                continue
            total = stats["total"]
            top10_rate = stats["top10"] / total * 100
            top25_rate = stats["top25"] / total * 100
            avg_rank = stats["sum_rank"] / total
            best_rank = stats.get("best_rank")
            eval_samples = stats.get("evalSamples", 0)
            avg_d2 = (
                stats.get("sumDepth2Parity", 0.0) / eval_samples
                if eval_samples
                else None
            )
            avg_d3 = (
                stats.get("sumDepth3Parity", 0.0) / eval_samples
                if eval_samples
                else None
            )
            avg_draws = (
                stats.get("sumDraws", 0.0) / eval_samples if eval_samples else None
            )
            d2_text = f"{avg_d2:4.1f}" if avg_d2 is not None else " -- "
            d3_text = f"{avg_d3:4.1f}" if avg_d3 is not None else " -- "
            draw_text = f"{avg_draws:4.1f}" if avg_draws is not None else " -- "
            print(
                f"  {bucket:<8} top10={top10_rate:5.1f}% "
                f"top25={top25_rate:5.1f}% wins={stats['wins']:>3} "
                f"best={best_rank or '-'} avg_rank={avg_rank:.1f} "
                f"d2={d2_text} d3={d3_text} draw={draw_text}"
            )


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
    baseline_defaults = offense_snapshot(load_search_config())
    for sweep in reversed(sweeps):
        for entry in sweep.get("combos", []):
            stored_hash = entry.get("configHash")
            if stored_hash == config_hash:
                result = dict(entry)
                result["timestamp"] = sweep.get("timestamp")
                return result
            combo_data = combo_with_defaults(combo_from_log(entry), baseline_defaults)
            normalized_hash = compute_config_hash(to_hash_payload(combo_data))
            if normalized_hash == config_hash:
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
        filtered = [
            hash_value
            for hash_value in pending_list
            if hash_value not in validated_hashes
        ]
        if filtered != pending_list:
            state["pendingValidation"] = filtered
            save_state(state)

    if PENDING_VALIDATION_PATH.exists():
        data = load_json(PENDING_VALIDATION_PATH, {})
        recs = data.get("recommendations")
        if isinstance(recs, list):
            filtered_recs = [
                entry
                for entry in recs
                if entry.get("configHash") not in validated_hashes
            ]
            if filtered_recs:
                if filtered_recs != recs:
                    data["recommendations"] = filtered_recs
                    write_json(PENDING_VALIDATION_PATH, data)
            else:
                PENDING_VALIDATION_PATH.unlink()


def command_validate(args: argparse.Namespace) -> None:
    dry_run = getattr(args, "dry_run", False)
    stop_flag = getattr(args, "stop_requested", False)
    config_hash = args.hash
    if not config_hash:
        pending = load_json(PENDING_VALIDATION_PATH, {}).get("recommendations", [])
        if not pending:
            raise SystemExit(
                "No pending validation recommendations. Specify --hash explicitly."
            )
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

    if dry_run:
        print(
            f"[dry-run] Would run validation for {config_hash[:8]} with command: {' '.join(cmd)}"
        )
        if not args.persist:
            print("[dry-run] Would restore previous search.json after validation.")
        print("[dry-run] Would remove hash from pending queue.")
        return

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        save_search_config(original_config)
        if exc.returncode < 0 or stop_flag:
            print(
                f"[loop] Validation aborted for {config_hash[:8]} (return code {exc.returncode})."
            )
            return
        raise SystemExit(exc.returncode)

    if not args.persist:
        save_search_config(original_config)
        print("search.json restored to the previous configuration.")

    remove_pending_hash(config_hash)


def aggregate_validation_runs(
    runs: Iterable[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        original_hash = run.get("configHash")
        if not original_hash:
            continue
        config_snapshot = run.get("config") or {}
        normalized_hash = original_hash
        if config_snapshot:
            payload = {}
            for key in HASH_KEYS:
                value = config_snapshot.get(key)
                if value is None and defaults is not None:
                    value = defaults.get(key)
                payload[key] = value
            normalized_hash = compute_config_hash(payload)
        bucket = aggregated.setdefault(
            normalized_hash,
            {
                "games": 0,
                "red": 0,
                "black": 0,
                "draw": 0,
                "depth2": {"red": 0, "black": 0, "draw": 0},
                "depth3": {"red": 0, "black": 0, "draw": 0},
                "runs": 0,
                "originalHashes": set(),
            },
        )
        bucket["originalHashes"].add(original_hash)
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
    # Convert sets to sorted lists for stable output
    final: Dict[str, Dict[str, Any]] = {}
    for key, value in aggregated.items():
        value["originalHashes"] = sorted(value["originalHashes"])
        final[key] = value
    return final


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


def compute_validation_streaks(
    runs: List[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
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
    dry_run = getattr(args, "dry_run", False)

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            print("\n[loop] Second interrupt received. Exiting immediately.")
            raise SystemExit(130)
        stop_requested = True
        print("\n[loop] Stop requested. Attempting to cancel the current sweep...")

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)
    if getattr(args, "reset_stall", False):
        state = load_state()
        reset_stall_state(state, thaw_knobs=True)
        print("[loop] Stall counters reset and knobs thawed.")

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
                dry_run=dry_run,
            )
            try:
                command_suggest(suggest_args)
            except SystemExit as exc:
                print(
                    f"[loop] Suggest stage failed with exit code {exc.code}. Stopping loop."
                )
                return

            sweep_args = argparse.Namespace(
                dry_run=dry_run, stop_requested=stop_requested
            )
            try:
                sweep_ok = command_sweep(sweep_args)
            except SystemExit as exc:
                print(
                    f"[loop] Sweep stage failed with exit code {exc.code}. Stopping loop."
                )
                return
            if not sweep_ok:
                break

            update_args = argparse.Namespace(limit=args.limit, dry_run=dry_run)
            try:
                command_update(update_args)
            except SystemExit as exc:
                print(
                    f"[loop] Update stage failed with exit code {exc.code}. Stopping loop."
                )
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
                        workers=str(VALIDATION_PARALLEL_WORKERS),
                        log=log_name,
                        persist=args.persist,
                        dry_run=dry_run,
                        stop_requested=stop_requested,
                    )
                    try:
                        command_validate(validate_args)
                    except SystemExit as exc:
                        print(
                            f"[loop] Validation for {hash_value[:8]} failed with exit code {exc.code}. Stopping loop."
                        )
                        return
                    if stop_requested:
                        break
            else:
                print("[loop] No pending validations after update.")

            if dry_run:
                print("[dry-run] Completed simulated cycle.")
                break

            state = load_state()
            successful_configs = state.get("successfulConfigs", {})
            if successful_configs:
                print("\n[loop] Validation goal reached by:")
                winning_hashes = sorted(successful_configs.keys())
                for hash_value in winning_hashes:
                    print(f"  {hash_value[:8]} streak={successful_configs[hash_value]}")
                # Persist the first winning configuration to search.json
                winner = winning_hashes[0]
                record = find_combo_by_hash(winner)
                if record:
                    combo = combo_with_defaults(
                        combo_from_log(record), offense_snapshot(load_search_config())
                    )
                    config = load_search_config()
                    apply_combo_to_search(combo, config)
                    save_search_config(config)
                    print(
                        f"\n[loop] Persisted winning config {winner[:8]} to {SEARCH_PATH.relative_to(PROJECT_ROOT)}"
                    )
                    print("Knob values:")
                    for key, value in combo.items():
                        if key in HASH_KEYS:
                            print(f"  {key} = {value}")
                else:
                    print(
                        "[loop] Warning: could not locate winning combo details to persist."
                    )
                break

            stall_score = state.get("cyclesSinceScoreImprovement", 0)
            stall_streak = state.get("cyclesSinceStreakImprovement", 0)
            best_streak = state.get("bestStreak", 0)
            if (
                best_streak < SUCCESS_VALIDATION_STREAK
                and max(stall_score, stall_streak) >= OVERALL_STALL_CYCLES
            ):
                plateau_hashes, _, _ = outstanding_validation_hashes(
                    PLATEAU_SCORE_CUTOFF
                )
                plateau_hashes = [
                    h
                    for h in plateau_hashes
                    if h not in state.get("successfulConfigs", {})
                ]
                if plateau_hashes:
                    print(
                        f"\n[loop] Plateau detected after {OVERALL_STALL_CYCLES} cycles without improvement.\n"
                        "Validating remaining high-scoring configurations before exiting..."
                    )
                    while (
                        plateau_hashes
                        and not state.get("successfulConfigs")
                        and not stop_requested
                    ):
                        target_hash = plateau_hashes.pop(0)
                        log_name = f"{args.log_prefix}-plateau-{target_hash[:8]}.log"
                        validate_args = argparse.Namespace(
                            hash=target_hash,
                            depth_config=args.depth_config,
                            workers=str(PLATEAU_VALIDATION_WORKERS),
                            log=log_name,
                            persist=args.persist,
                            dry_run=dry_run,
                            stop_requested=stop_requested,
                        )
                        try:
                            command_validate(validate_args)
                        except SystemExit as exc:
                            print(
                                f"[loop] Validation for {target_hash[:8]} failed with exit code {exc.code}. Stopping loop."
                            )
                            return
                        state = load_state()
                        plateau_hashes, _, _ = outstanding_validation_hashes(
                            VALIDATION_SCORE_CUTOFF
                        )
                        plateau_hashes = [
                            h
                            for h in plateau_hashes
                            if h not in state.get("successfulConfigs", {})
                        ]
                    if not state.get("successfulConfigs"):
                        print("[loop] Plateau validation backlog cleared. Exiting.")
                else:
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
    aggregated_validations = aggregate_validation_runs(validations, baseline_defaults)
    streaks = compute_validation_streaks(validations, baseline_defaults)
    successful_hashes = {
        hash_value
        for hash_value, streak in streaks.items()
        if streak >= SUCCESS_VALIDATION_STREAK
    }

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
                    (
                        item[0]["score"]
                        if isinstance(item[0].get("score"), (int, float))
                        else float("inf")
                    ),
                    item[0].get("timestamp", ""),
                )
            )
            top_entries = scored_entries[: min(5, len(scored_entries))]
            for entry, metrics in top_entries:
                hash_prefix = (
                    entry["configHash"][:8] if entry["configHash"] else "--------"
                )
                validated = (
                    "yes" if entry["configHash"] in aggregated_validations else "no"
                )
                goal = "yes" if entry["configHash"] in successful_hashes else "no"
                print(
                    f" {hash_prefix} score={entry.get('score')} "
                    f"red={metrics['totalRed']} black={metrics['totalBlack']} "
                    f"draw={metrics['totalDraw']} parity={metrics['parity']} validated={validated} goal={goal}"
                )
        else:
            print(
                "Sweep results lack evaluation data; run a sweep to populate metrics."
            )

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
    suggest_parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="Number of combos to propose (default: 24)",
    )
    suggest_parser.add_argument(
        "--exploit",
        type=int,
        default=8,
        help="Number of top historical combos to repeat (default: 8)",
    )
    suggest_parser.add_argument(
        "--seed", type=int, default=None, help="Optional RNG seed"
    )
    suggest_parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate config hashes in suggestions",
    )
    suggest_parser.set_defaults(func=command_suggest)

    subparsers.add_parser("sweep", help="Run the baseline sweep script").set_defaults(
        func=command_sweep
    )

    update_parser = subparsers.add_parser("update", help="Process latest sweep results")
    update_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of configs to queue for validation (default: 5)",
    )
    update_parser.add_argument(
        "--rebuild-telemetry",
        action="store_true",
        help="Recompute bucket telemetry from the full sweep history before processing new results",
    )
    update_parser.set_defaults(func=command_update)

    validate_parser = subparsers.add_parser(
        "validate", help="Run validation for a config"
    )
    validate_parser.add_argument(
        "--hash",
        help="Config hash to validate (defaults to first pending recommendation)",
    )
    validate_parser.add_argument(
        "--depth-config",
        dest="depth_config",
        help="Depth config string (e.g. 2:60,3:60)",
    )
    validate_parser.add_argument(
        "--workers", help="Worker count passed to runValidation.js"
    )
    validate_parser.add_argument("--log", help="Custom log file name")
    validate_parser.add_argument(
        "--persist",
        action="store_true",
        help="Keep validated config in search.json after run",
    )
    validate_parser.set_defaults(func=command_validate)

    subparsers.add_parser(
        "report", help="Summarize sweep and validation status"
    ).set_defaults(func=command_report)

    loop_parser = subparsers.add_parser("loop", help="Run continuous auto-tuning cycle")
    loop_parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="Combos per sweep suggestion (default: 24)",
    )
    loop_parser.add_argument(
        "--exploit",
        type=int,
        default=8,
        help="History combos to retain per sweep (default: 8)",
    )
    loop_parser.add_argument(
        "--seed", type=int, default=None, help="Seed for deterministic suggestions"
    )
    loop_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Validation queue size per sweep (default: 5)",
    )
    loop_parser.add_argument(
        "--depth-config",
        default="2:60,3:60",
        help="Validation depth config (default: 2:60,3:60)",
    )
    loop_parser.add_argument(
        "--workers", default="10", help="Validation worker count (default: 10)"
    )
    loop_parser.add_argument(
        "--log-prefix",
        default="loop-validation",
        help="Prefix for validation log files",
    )
    loop_parser.add_argument(
        "--persist",
        action="store_true",
        help="Keep last validated config in search.json",
    )
    loop_parser.add_argument(
        "--max-cycles", type=int, default=None, help="Optional cap on loop iterations"
    )
    loop_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the loop without running sweeps or validations",
    )
    loop_parser.add_argument(
        "--reset-stall",
        action="store_true",
        help="Reset stall counters before starting the loop",
    )
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
