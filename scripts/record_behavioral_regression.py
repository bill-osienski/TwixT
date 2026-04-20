#!/usr/bin/env python3
"""Record behavioral regression deltas against a curated position suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move
from scripts.GPU.ai.search import choose_move
from scripts.GPU.ai.heuristics import DEFAULT_KNOBS

ORACLE = PROJECT_ROOT / "tests" / "js_oracle" / "search_oracle.js"


def _check_node_available() -> bool:
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_state(entry: Dict) -> GameState:
    state = GameState(board_size=int(entry.get("board_size", 24)))
    moves = entry.get("moves", [])
    if moves:
        state.to_move = moves[0]["player"]
    for mv in moves:
        state.to_move = mv["player"]
        state = apply_move(state, mv["row"], mv["col"])
    if "current_player" in entry:
        state.to_move = entry["current_player"]
    return state


def _state_to_oracle_payload(state: GameState, depth: int) -> Dict:
    pegs = [{"row": r, "col": c, "player": p} for (r, c), p in state.pegs.items()]
    bridges = [
        {"from": {"row": a[0], "col": a[1]}, "to": {"row": b[0], "col": b[1]}}
        for a, b in state.bridges
    ]
    return {
        "boardSize": state.board_size,
        "pegs": pegs,
        "bridges": bridges,
        "currentPlayer": state.to_move,
        "depth": depth,
    }


def _call_js_oracle(payload: Dict) -> Dict:
    result = subprocess.run(
        ["node", str(ORACLE)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"JS oracle failed: {result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("JS oracle returned empty output")
    for line in lines:
        if not line.lstrip().startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("JS oracle returned no JSON payload")


def _python_candidates(state: GameState, depth: int, top_n: int) -> List[Dict]:
    res = choose_move(
        state,
        knobs=DEFAULT_KNOBS,
        depth=depth,
        top_n=top_n,
        use_value_model=False,
        temperature=0.0,
        rng=None,
        mode="debug",
    )
    return res.candidates


def run_suite(positions_path: Path, top_k: int, score_delta: float) -> Dict:
    payload = json.loads(positions_path.read_text(encoding="utf-8"))
    positions = payload.get("positions", [])
    results = []
    pass_count = 0
    fail_count = 0

    for entry in positions:
        name = entry.get("name", "unnamed")
        depth = int(entry.get("depth", 2))
        state = _build_state(entry)

        candidates = _python_candidates(state, depth, top_n=20)
        if not candidates:
            results.append({"name": name, "error": "no python candidates"})
            fail_count += 1
            continue

        best_score = candidates[0]["score"]
        top_set = {(c["row"], c["col"]) for c in candidates[:top_k]}
        score_map = {(c["row"], c["col"]): c["score"] for c in candidates}

        js_result = _call_js_oracle(_state_to_oracle_payload(state, depth))
        if js_result.get("error"):
            results.append({"name": name, "error": js_result["error"]})
            fail_count += 1
            continue

        js_move = (js_result.get("row"), js_result.get("col"))
        js_score = score_map.get(js_move)
        if js_score is None:
            results.append({"name": name, "error": f"js move not in python candidates: {js_move}"})
            fail_count += 1
            continue

        in_top_k = js_move in top_set
        within_delta = js_score >= best_score - score_delta
        ok = in_top_k or within_delta

        results.append(
            {
                "name": name,
                "depth": depth,
                "js_move": {"row": js_move[0], "col": js_move[1]},
                "js_score": js_score,
                "best_score": best_score,
                "delta": best_score - js_score,
                "in_top_k": in_top_k,
                "within_delta": within_delta,
                "ok": ok,
            }
        )
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "score_delta": score_delta,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record behavioral regression deltas.")
    parser.add_argument(
        "--positions",
        default=str(PROJECT_ROOT / "tests" / "behavioral_positions.json"),
        help="Path to curated position suite JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "logs" / "behavioral-regression.json"),
        help="Output JSON file (appended).",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--score-delta", type=float, default=None)
    args = parser.parse_args()

    if not _check_node_available():
        raise SystemExit("Node.js not available; cannot run JS oracle.")
    if not ORACLE.exists():
        raise SystemExit("search_oracle.js not found.")

    positions_path = Path(args.positions).resolve()
    suite = json.loads(positions_path.read_text(encoding="utf-8"))
    top_k = int(args.top_k if args.top_k is not None else suite.get("top_k", 5))
    score_delta = float(
        args.score_delta if args.score_delta is not None else suite.get("score_delta", 500)
    )

    summary = run_suite(positions_path, top_k=top_k, score_delta=score_delta)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or "runs" not in existing:
            existing = {"runs": []}
    else:
        existing = {"runs": []}

    existing["runs"].append(summary)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote regression summary -> {out_path}")


if __name__ == "__main__":
    main()
