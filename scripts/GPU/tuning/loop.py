from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config.knobs import clamp_to_core, normalize_to_allowed
from ..selfplay.engine import TwixtSimulator
from ..selfplay.parallel import DEFAULT_WORKERS, run_games
from ..tuning.hasher import config_hash
from ..tuning.ridge import fit_ridge, predicted_bias_gate
from ..tuning.state import TuningState, load_state, norm_status, save_state
from ..tuning.sweep import generate_sweep
from ..tuning.validation import combine_depths, validate_depths
from ..utils.jsonl import iter_jsonl


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "scripts" / "GPU" / "logs"

    @property
    def games_dir(self) -> Path:
        return self.logs / "games"

    @property
    def state_json(self) -> Path:
        return self.logs / "tuning_state.json"

    @property
    def sweep_results_jsonl(self) -> Path:
        return self.logs / "sweep-results.jsonl"

    @property
    def validation_results_jsonl(self) -> Path:
        return self.logs / "validation-results.jsonl"

    @property
    def pending_validation_json(self) -> Path:
        return self.logs / "pending-validation.json"


def default_depth_weights(depths: Sequence[int]) -> Dict[int, float]:
    return {d: float(i + 1) for i, d in enumerate(sorted(depths))}


def init_logs(paths: Paths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.games_dir.mkdir(parents=True, exist_ok=True)
    if not paths.pending_validation_json.exists():
        paths.pending_validation_json.write_text(json.dumps({"queue": []}, indent=2) + "\n", encoding="utf-8")
    if not paths.state_json.exists():
        save_state(paths.state_json, TuningState())


def sweep_cycle(
    *,
    paths: Paths,
    base_knobs: Dict[str, float],
    depths: Sequence[int],
    games: int,
    total: int,
    fixed: int,
    mutate: int,
    seed: Optional[int],
    pred_gate: bool,
    max_pred_bias: float,
    min_r2: float,
    board: int = 24,
    workers: int = 0,
) -> None:
    sim = TwixtSimulator(board_size=board)
    st = load_state(paths.state_json)

    base_knobs = clamp_to_core(normalize_to_allowed(base_knobs))

    run_seed = int(seed) if seed is not None else int(time.time())
    candidates = generate_sweep(base_knobs, total=total, seed=run_seed, fixed_slots=fixed, mutate_slots=mutate)

    feature_names = sorted(base_knobs.keys())
    model = fit_ridge(st.samples, feature_names=feature_names) if st.samples else None

    for cand in candidates:
        knobs = cand.knobs
        h = config_hash(knobs)
        entry = st.get(h)
        if norm_status(entry.status) in ("RETIRED", "STABLE"):
            continue

        ok, pred = predicted_bias_gate(model, knobs, max_abs_bias=max_pred_bias, min_r2=min_r2)
        if pred_gate and not ok:
            knobs2 = clamp_to_core(knobs)
            ok2, pred2 = predicted_bias_gate(model, knobs2, max_abs_bias=max_pred_bias, min_r2=min_r2)
            if not ok2:
                continue
            knobs = knobs2
            pred = pred2

        actual_workers = workers if workers > 0 else DEFAULT_WORKERS
        for d in depths:
            run_games(
                sim=sim,
                knobs=knobs,
                depth=int(d),
                games=int(games),
                seed=run_seed + int(d) * 100_000,
                tag=cand.tag,
                results_jsonl=paths.sweep_results_jsonl,
                games_dir=paths.games_dir,
                predicted_bias=pred,
                workers=actual_workers,
            )

        entry.sweep_runs += 1
        st.mark_status(h, "SHORTLIST")

        # Lightweight training sample (we replace this with real measured sweep bias after aggregation)
        st.samples.append({**knobs, "bias": float(pred or 0.0), "weight": 2.0 if cand.tag == "fixed_probe" else 0.25})

    st.iteration += 1
    save_state(paths.state_json, st)


def rank_from_sweeps(
    *,
    paths: Paths,
    depths: Sequence[int],
    top: int,
) -> List[Dict[str, Any]]:
    depth_weights = default_depth_weights(depths)

    # Aggregate by (hash, depth)
    by_hd: Dict[str, Dict[int, Dict[str, int]]] = {}
    knobs_by_hash: Dict[str, Dict[str, float]] = {}

    for r in iter_jsonl(paths.sweep_results_jsonl):
        h = str(r.get("hash"))
        d = int(r.get("depth"))
        if d not in depths:
            continue
        by_hd.setdefault(h, {}).setdefault(d, {"games": 0, "red": 0, "black": 0, "draws": 0})
        agg = by_hd[h][d]
        agg["games"] += int(r.get("games", 0))
        agg["red"] += int(r.get("red", 0))
        agg["black"] += int(r.get("black", 0))
        agg["draws"] += int(r.get("draws", 0))
        knobs_by_hash[h] = dict(r.get("knobs", knobs_by_hash.get(h, {})))

    scored: List[Tuple[float, str]] = []
    for h, per_depth in by_hd.items():
        summaries: List[Tuple[int, Any]] = []
        ok = True
        for d in depths:
            if d not in per_depth:
                ok = False
                break
            a = per_depth[d]
            # Reuse GameSummary shape
            from ..selfplay.results import GameSummary

            summaries.append((d, GameSummary(games=a["games"], red=a["red"], black=a["black"], draws=a["draws"])))
        if not ok:
            continue
        score, _bias = combine_depths([(d, s) for d, s in summaries], depth_weights)
        scored.append((score, h))

    scored.sort(key=lambda t: t[0])
    queue = [{"hash": h, "score": s, "knobs": knobs_by_hash.get(h, {})} for s, h in scored[:top]]
    paths.pending_validation_json.write_text(json.dumps({"queue": queue}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return queue


def validate_queue(
    *,
    paths: Paths,
    depths: Sequence[int],
    games: int,
    pass_score: float,
    streak_needed: int,
    retire_after: int,
    board: int = 24,
    workers: int = 0,
) -> Optional[str]:
    """Validate candidates with per-depth pass requirements.

    Key improvement: Each depth must pass INDEPENDENTLY.
    A config with d2=+5% and d3=-5% will NOT pass, even though they average to 0%.
    """
    sim = TwixtSimulator(board_size=board)
    st = load_state(paths.state_json)
    actual_workers = workers if workers > 0 else DEFAULT_WORKERS

    raw = json.loads(paths.pending_validation_json.read_text(encoding="utf-8")) if paths.pending_validation_json.exists() else {"queue": []}
    queue = list(raw.get("queue", []))
    if not queue:
        return None

    for item in queue:
        knobs = dict(item.get("knobs", {}))
        if not knobs:
            continue
        h = config_hash(knobs)
        entry = st.get(h)
        if norm_status(entry.status) in ("RETIRED", "STABLE"):
            continue

        summaries = []
        base_seed = int(time.time())
        for d in depths:
            summ = run_games(
                sim=sim,
                knobs=knobs,
                depth=int(d),
                games=int(games),
                seed=base_seed + int(d) * 100_000,
                tag="validation",
                results_jsonl=paths.validation_results_jsonl,
                games_dir=paths.games_dir,
                workers=actual_workers,
            )
            summaries.append((int(d), summ))

        # Use new per-depth validation
        result = validate_depths(summaries, pass_threshold=pass_score)
        entry.validation_runs += 1
        entry.last_score = float(result.worst_bias)
        entry.last_bias = float(result.combined_bias)

        # Store per-depth results for debugging
        entry.per_depth_bias = {r.depth: r.bias for r in result.per_depth}

        # Only increment streak if ALL depths pass independently
        if result.all_passed:
            entry.streak60 += 1
            st.mark_status(h, "VALIDATING")
            print(f"  {h}: PASS (all depths) - worst={result.worst_bias:.3f}, streak={entry.streak60}")
        else:
            entry.streak60 = 0
            failed_depths = [r for r in result.per_depth if not r.passed]
            print(f"  {h}: FAIL - depths {[r.depth for r in failed_depths]} exceeded threshold")
            for r in result.per_depth:
                status = "PASS" if r.passed else "FAIL"
                print(f"    d{r.depth}: bias={r.bias:+.3f} ({status})")

        if entry.validation_runs >= retire_after and entry.streak60 == 0:
            st.mark_status(h, "RETIRED")
            entry.retired_reason = f"validation_fail_{retire_after}"
            print(f"  {h}: RETIRED after {retire_after} failures")

        if entry.streak60 >= streak_needed and result.all_passed:
            st.mark_status(h, "STABLE")
            st.active_hash = h
            save_state(paths.state_json, st)
            print(f"  {h}: STABLE! Streak of {streak_needed} achieved")
            return h

    save_state(paths.state_json, st)
    return None
