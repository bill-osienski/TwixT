#!/usr/bin/env python3
"""Behavioral regression tests: JS move must align with Python top-K."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move
from scripts.GPU.ai.search import choose_move
from scripts.GPU.ai.heuristics import DEFAULT_KNOBS

ORACLE = PROJECT_ROOT / "tests" / "js_oracle" / "search_oracle.js"
POSITIONS = PROJECT_ROOT / "tests" / "behavioral_positions.json"


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


def _load_positions() -> Dict:
    return json.loads(POSITIONS.read_text(encoding="utf-8"))


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


pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not _check_node_available(), reason="Node.js not available"),
    pytest.mark.skipif(not ORACLE.exists(), reason="search_oracle.js not found"),
]


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


def test_js_move_within_python_topk() -> None:
    config = _load_positions()
    top_k = int(config.get("top_k", 5))
    score_delta = float(config.get("score_delta", 500))

    for entry in config.get("positions", []):
        name = entry.get("name", "unnamed")
        depth = int(entry.get("depth", 2))
        state = _build_state(entry)

        candidates = _python_candidates(state, depth, top_n=20)
        if not candidates:
            pytest.fail(f"{name}: no python candidates")

        best_score = candidates[0]["score"]
        top_set = {(c["row"], c["col"]) for c in candidates[:top_k]}
        score_map = {(c["row"], c["col"]): c["score"] for c in candidates}

        payload = _state_to_oracle_payload(state, depth)
        js_result = _call_js_oracle(payload)
        if js_result.get("error"):
            pytest.fail(f"{name}: JS oracle error: {js_result['error']}")

        js_move = (js_result.get("row"), js_result.get("col"))
        if js_move not in score_map:
            pytest.fail(f"{name}: JS move {js_move} not in Python candidates")

        js_score = score_map[js_move]
        in_top_k = js_move in top_set
        within_delta = js_score >= best_score - score_delta

        assert (
            in_top_k or within_delta
        ), f"{name}: JS move {js_move} score={js_score:.2f} best={best_score:.2f} delta={score_delta}"
