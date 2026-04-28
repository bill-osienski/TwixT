# Strong-Advantage Probe Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `strong_advantage` probe tier (deep-MCTS labeled, light-reviewed) so the value head's behavior on dominant-but-not-forced positions can be measured per-iter, separately from the existing forced tier.

**Architecture:** Tier-parameterized probe generator (`scripts/build_probe_suite.py`) replacing the current `build_bootstrap_probe_suite.py` (kept as a backward-compat shim). The new tier is mined from decisive games (Phase 1 heuristic), labeled by 10k-sim MCTS × 3 repeats with strict admission filter (Phase 2), and operator-promoted from a draft (Phase 3). Trainer telemetry emits a tier-keyed `probe_summary` block alongside the legacy `forced_probe_summary`; analyzer aggregation is parameterized over tier names so adding tiers doesn't duplicate code.

**Tech Stack:** Python 3.14, MLX, pytest, existing `scripts/GPU/alphazero/probe_eval.py` and `scripts/twixt_replay_analyzer.py` infrastructure.

**Spec:** [`docs/superpowers/specs/2026-04-28-strong-advantage-probe-tier-design.md`](../specs/2026-04-28-strong-advantage-probe-tier-design.md).

**Scope:** Migration steps 1–4 from the spec. Step 5 (legacy `forced_probe_summary` removal after one release cycle of dual-emit) is deferred to a follow-up plan.

---

## Step 1 — Generator refactor + parity test

The forced tier already produces `tests/probes/twixt_probes.json`. Step 1 extracts that logic into a tier-parameterized `build_probe_suite.py` while a byte-identical parity test guarantees no regression.

### Task 1.1 — Add forced parity test (failing)

**Files:**
- Create: `tests/test_probe_suite_forced_parity.py`

- [ ] **Step 1: Write the failing test**

```python
"""Parity guard: regenerating --tier forced must produce a byte-identical
output to the committed tests/probes/twixt_probes.json. This is the safety
gate for the build_probe_suite.py refactor — if it fails after a refactor,
the refactor is wrong.

Reads selection_rules from the committed file's meta block, so the test
follows whatever args the committed suite used (not pinned to a literal
iter range).

Assumed stable inputs: scripts/GPU/logs/games/iter_NNNN_game_MMM.json for
the iter range in meta.selection_rules.source_iter_range. If those are
moved/edited, this test will fail and the committed suite must be
regenerated against the new replay set with a deliberate commit.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_SUITE = PROJECT_ROOT / "tests" / "probes" / "twixt_probes.json"


def test_tier_forced_byte_identical_to_committed_suite(tmp_path):
    committed_bytes = COMMITTED_SUITE.read_bytes()
    meta = json.loads(committed_bytes)["meta"]
    rules = meta["selection_rules"]
    src_min, src_max = rules["source_iter_range"]

    out_path = tmp_path / "regenerated_twixt_probes.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_probe_suite.py"),
        "--tier", "forced",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", str(src_min), str(src_max),
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"build_probe_suite.py exited {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )

    regenerated_bytes = out_path.read_bytes()
    if regenerated_bytes != committed_bytes:
        # Surface the first diff for debuggability.
        from difflib import unified_diff
        diff = "\n".join(unified_diff(
            committed_bytes.decode().splitlines(),
            regenerated_bytes.decode().splitlines(),
            fromfile="committed/twixt_probes.json",
            tofile="regenerated/twixt_probes.json",
            lineterm="",
            n=3,
        ))
        raise AssertionError(
            "Regenerated forced suite differs from committed suite.\n"
            "First 50 lines of diff:\n"
            + "\n".join(diff.splitlines()[:50])
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_probe_suite_forced_parity.py -v`
Expected: FAIL because `scripts/build_probe_suite.py` does not exist yet (subprocess will exit non-zero with "No such file").

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_probe_suite_forced_parity.py
git commit -m "test(probes): add forced-tier parity gate (currently failing)

Guards the upcoming build_probe_suite.py refactor. Reads selection_rules
from the committed suite's meta block so the test follows whatever args
produced the committed file.
"
```

### Task 1.2 — Create build_probe_suite.py with full forced-tier logic

**Files:**
- Create: `scripts/build_probe_suite.py`

- [ ] **Step 1: Write the new generator**

```python
"""Tier-parameterized probe suite generator.

Replaces scripts/build_bootstrap_probe_suite.py as the real implementation
(that script is kept as a thin --tier forced shim for muscle memory and
existing CI/cron commands).

Tiers:
  --tier forced            Bootstrap forced suite (existing behavior,
                           writes tests/probes/twixt_probes.json by default).
  --tier strong_advantage  Bootstrap strong-advantage suite (deep-MCTS
                           labeled, light-reviewed). Phases 1/2/3 per
                           docs/superpowers/specs/2026-04-28-...

Both tiers produce byte-identical output for identical inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# --- Tier dispatch ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--tier", choices=["forced", "strong_advantage"], required=True)
    ap.add_argument("--input", default="scripts/GPU/logs/games")
    ap.add_argument("--source-iter-range", nargs=2, type=int,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--out", default=None,
                    help="Output path. Defaults: forced -> tests/probes/twixt_probes.json, "
                         "strong_advantage -> tests/probes/strong_advantage_probes.json")
    ap.add_argument("--samples-per-bucket", type=int, default=12)
    ap.add_argument("--max-probes", type=int, default=30)

    # strong_advantage-specific flags (ignored for forced)
    ap.add_argument("--label-checkpoint", default=None)
    ap.add_argument("--label-mcts-sims", type=int, default=10000)
    ap.add_argument("--label-mcts-repeats", type=int, default=3)
    ap.add_argument("--magnitude-threshold", type=float, default=0.45)
    ap.add_argument("--top1-share-floor", type=float, default=0.15)
    ap.add_argument("--stability-cap", type=float, default=0.15)
    ap.add_argument("--promote", action="store_true",
                    help="Promote *.draft.json to committed file")
    ap.add_argument("--reviewer", default=None,
                    help="Reviewer name, required with --promote")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing draft or committed file")

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.tier == "forced":
        return _run_forced(args)
    elif args.tier == "strong_advantage":
        return _run_strong_advantage(args)
    else:
        print(f"[probe_suite] ERROR: unknown tier {args.tier}", file=sys.stderr)
        return 2


# --- Forced tier (lifted verbatim from build_bootstrap_probe_suite.py) ---

def _run_forced(args) -> int:
    if args.out is None:
        args.out = "tests/probes/twixt_probes.json"
    if args.source_iter_range is None:
        print("[probe_suite] ERROR: --source-iter-range required for --tier forced",
              file=sys.stderr)
        return 2

    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games

    min_iter, max_iter = args.source_iter_range
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[probe_suite] ERROR: --input path is not a directory: {input_dir}",
              file=sys.stderr)
        return 2

    games: list[dict] = []
    for fp in sorted(input_dir.glob("iter_*_game_*.json")):
        with open(fp) as f:
            try:
                g = json.load(f)
            except json.JSONDecodeError:
                continue
        iteration = (g.get("meta") or {}).get("iteration")
        if iteration is None or not (min_iter <= iteration <= max_iter):
            continue
        games.append(g)

    probes = extract_forced_probes_from_games(
        games,
        active_size=24,
        k_plies=2,
        winner_reasons=frozenset({"win"}),
        dedupe_exact=True,
        dedupe_mirror=True,
        max_probes=None,
    )

    def _sort_key(p: dict) -> tuple:
        basename = p["source_game"]
        try:
            iter_num = int(basename.split("_")[1])
        except (IndexError, ValueError):
            iter_num = 0
        return (-iter_num, -p["source_ply"], basename)

    red = [p for p in probes if p["category"] == "near_win_red"]
    black = [p for p in probes if p["category"] == "near_win_black"]

    balanced: list[dict] = []
    ri = bi = 0
    red_count = black_count = 0
    while len(balanced) < args.max_probes:
        can_red = ri < len(red) and red_count + 1 <= 2 * max(black_count, 1)
        can_black = bi < len(black) and black_count + 1 <= 2 * max(red_count, 1)
        if not can_red and not can_black:
            break
        if can_red and can_black:
            if _sort_key(red[ri]) <= _sort_key(black[bi]):
                balanced.append(red[ri]); ri += 1; red_count += 1
            else:
                balanced.append(black[bi]); bi += 1; black_count += 1
        elif can_red:
            balanced.append(red[ri]); ri += 1; red_count += 1
        else:
            balanced.append(black[bi]); bi += 1; black_count += 1

    balanced.sort(key=_sort_key)

    payload = {
        "meta": {
            "type": "bootstrap_rule_selected",
            "not_gate_suite": True,
            "note": ("Rule-selected bootstrap suite for trainer-side inline "
                     "telemetry and practical regression monitoring. NOT the "
                     "spec §7 review-curated gate suite — see "
                     "tests/probes/README.md for the distinction."),
            "generator": "scripts/build_bootstrap_probe_suite.py",
            "generator_version": 1,
            "selection_rules": {
                "board_size": 24,
                "winner_reasons": ["win"],
                "k_plies_from_terminal": 2,
                "dedup": "exact + 4-form-mirror-canonical",
                "source_iter_range": [min_iter, max_iter],
            },
        },
        "probes": balanced,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"[probe_suite] wrote {len(balanced)} forced probes to {out_path}")
    return 0


# --- Strong-advantage tier (filled in during Step 2 of the plan) ---

def _run_strong_advantage(args) -> int:
    raise NotImplementedError(
        "Strong-advantage tier added in Step 2 of the implementation plan. "
        "See docs/superpowers/plans/2026-04-28-strong-advantage-probe-tier.md."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
```

Note: the forced tier writes `"generator": "scripts/build_bootstrap_probe_suite.py"` in the meta block — same string as before — so existing committed suite stays byte-identical. The fact that the actual code now lives in `build_probe_suite.py` is hidden in the meta to keep parity.

- [ ] **Step 2: Run parity test to verify it now passes**

Run: `.venv/bin/pytest tests/test_probe_suite_forced_parity.py -v`
Expected: PASS (regenerated bytes equal committed bytes).

If it FAILS, the most likely cause is a divergence in the lifted code. Read the diff section the test prints; compare against `scripts/build_bootstrap_probe_suite.py` line-for-line. Don't proceed to Task 1.3 until it passes.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_probe_suite.py
git commit -m "feat(probes): add tier-parameterized build_probe_suite.py

Lifts the forced-tier generation logic from build_bootstrap_probe_suite.py
into a tier-parameterized entrypoint. Forced tier output is byte-identical
(parity test passes). Strong-advantage tier is a NotImplementedError stub
filled in by Step 2.
"
```

### Task 1.3 — Convert build_bootstrap_probe_suite.py to a shim

**Files:**
- Modify: `scripts/build_bootstrap_probe_suite.py` (replace contents)

- [ ] **Step 1: Replace the file with a shim**

```python
"""Backward-compatibility shim.

The real implementation now lives in scripts/build_probe_suite.py. This
shim preserves the existing CLI/cron invocation
(`build_bootstrap_probe_suite.py --source-iter-range MIN MAX`) by injecting
`--tier forced` and forwarding to the new entrypoint.

DO NOT add new flags here. Add them to build_probe_suite.py instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    real = Path(__file__).resolve().parent / "build_probe_suite.py"
    args = [sys.executable, str(real), "--tier", "forced", *sys.argv[1:]]
    import os
    os.execv(sys.executable, args)
```

- [ ] **Step 2: Verify the shim works**

Pull the iter range from the committed suite's meta (same approach as the parity test in Task 1.1) so this smoke doesn't drift if the committed file is regenerated against a different range:

```bash
RANGE=$(.venv/bin/python -c "import json; r=json.load(open('tests/probes/twixt_probes.json'))['meta']['selection_rules']['source_iter_range']; print(r[0], r[1])")
.venv/bin/python scripts/build_bootstrap_probe_suite.py \
    --source-iter-range $RANGE \
    --out /tmp/shim_smoke_output.json
```

Expected: writes `/tmp/shim_smoke_output.json`, prints `[probe_suite] wrote N forced probes to /tmp/shim_smoke_output.json`.

Then verify byte-identical:
```bash
diff /tmp/shim_smoke_output.json tests/probes/twixt_probes.json && echo "OK: shim produces identical output"
```
Expected: no diff output, prints `OK: shim produces identical output`.

- [ ] **Step 3: Re-run the parity test**

Run: `.venv/bin/pytest tests/test_probe_suite_forced_parity.py -v`
Expected: PASS (still passes — the shim doesn't affect the test which calls `build_probe_suite.py` directly).

- [ ] **Step 4: Commit**

```bash
git add scripts/build_bootstrap_probe_suite.py
git commit -m "refactor(probes): convert build_bootstrap_probe_suite.py to a shim

Existing CLI/cron commands continue to work; they now dispatch to
build_probe_suite.py --tier forced. Forced output remains byte-identical
(parity test green).
"
```

---

## Step 2 — Strong-advantage tier

Step 2 fills in `_run_strong_advantage` end-to-end: candidate mining (Phase 1), MCTS labeling (Phase 2), promotion workflow (Phase 3), tests (mocked + opt-in live smoke). Operator runs the generator manually after the code lands to produce the actual committed `strong_advantage_probes.json`.

### Task 2.1 — Phase-1 structural feature computation

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py` (append new functions)
- Create: `tests/test_strong_advantage_probe_suite.py`

- [ ] **Step 1: Write failing tests for structural features**

Append to `tests/test_strong_advantage_probe_suite.py` (create the file if it doesn't exist):

```python
"""Tests for the strong_advantage probe tier: structural features,
admission filter, ID determinism, category assignment, and the promotion
workflow.

Labeling is mocked: tests inject a stub labeler. The opt-in live smoke
test lives separately in tests/test_strong_advantage_smoke_live.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_state(moves, starting_player="red"):
    """Build a TwixtState by applying the given (row, col) moves in order."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=24, to_move=starting_player)
    for r, c in moves:
        s = s.apply_move((r, c))
    return s


def test_phase1_features_red_chain_top_to_mid_board():
    """Red builds a knight-connected chain from row 0 down through the
    middle. cc_size, cc_axis_span, cc_touches_own_goal must reflect the
    chain.
    """
    from scripts.GPU.alphazero.probe_eval import compute_phase1_features

    # Red knight chain: (0,12) -> (2,11) -> (4,12) -> (6,11) -> (8,12)
    # Black filler so plies alternate; black pegs placed away from red's chain.
    moves = [
        (0, 12), (1, 0),
        (2, 11), (1, 1),
        (4, 12), (1, 2),
        (6, 11), (1, 3),
        (8, 12), (1, 4),
    ]
    state = _make_state(moves)
    feats = compute_phase1_features(state, winner="red")
    assert feats["cc_size"] >= 5
    assert feats["cc_axis_span"] >= 0.30  # spans rows 0..8 of 23
    assert feats["cc_touches_own_goal"] is True  # (0, 12) touches row 0
    assert feats["forced_within_2"] is False
    # axis_span_margin = winner_span - loser_span; loser is black with no chain
    assert feats["axis_span_margin"] >= 0.20
    # centroid around row 4, col ~12 -> very central
    assert feats["centroid_chebyshev_from_center"] <= 6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py::test_phase1_features_red_chain_top_to_mid_board -v`
Expected: FAIL with `ImportError: cannot import name 'compute_phase1_features'`.

- [ ] **Step 3: Implement `compute_phase1_features`**

Append to `scripts/GPU/alphazero/probe_eval.py`:

```python
# ============================================================
# Strong-advantage probe tier — Phase 1 structural features
# ============================================================

def compute_phase1_features(state, winner: str) -> dict:
    """Compute Phase-1 structural features for the eventual winner of a game.

    Used by the strong_advantage probe-suite generator to filter candidate
    positions before deep-MCTS labeling. See spec
    docs/superpowers/specs/2026-04-28-strong-advantage-probe-tier-design.md
    Phase 1.

    Args:
        state: TwixtState at the candidate position (winner has not yet won).
        winner: "red" or "black" — the side that wins the source game.

    Returns:
        dict with keys:
          cc_size: int — size of the largest same-color connected component
            for `winner`.
          cc_axis_span: float — fraction of `winner`'s goal axis the largest
            CC spans. Red goal axis is rows; black is cols. Range [0, 1].
          cc_touches_own_goal: bool — True if the largest CC touches at
            least one of `winner`'s two goal edges.
            (red: row 0 or row 23; black: col 0 or col 23).
          axis_span_margin: float — winner_cc_axis_span - loser_cc_axis_span.
            Negative if the loser is more advanced.
          centroid_chebyshev_from_center: int — Chebyshev distance of the
            winner's CC centroid from the board center (11.5, 11.5).
          forced_within_2: bool — True if `winner` has a forced win within
            2 plies (delegates to the existing forced detector).
    """
    from .game.twixt_state import TwixtState  # noqa: F401  (type hint context)

    loser = "black" if winner == "red" else "red"
    winner_pegs = _collect_pegs(state, winner)
    loser_pegs = _collect_pegs(state, loser)

    win_cc, win_span = _largest_connected_component(state, winner_pegs, winner)
    _, lose_span = _largest_connected_component(state, loser_pegs, loser)

    if not win_cc:
        return {
            "cc_size": 0,
            "cc_axis_span": 0.0,
            "cc_touches_own_goal": False,
            "axis_span_margin": -lose_span,
            "centroid_chebyshev_from_center": 23,
            "forced_within_2": False,
        }

    # Goal-touching: does the largest CC touch a goal-axis edge for winner?
    if winner == "red":
        touches = any(r == 0 or r == 23 for r, _ in win_cc)
    else:
        touches = any(c == 0 or c == 23 for _, c in win_cc)

    # Centroid Chebyshev distance from board center (11.5, 11.5).
    avg_r = sum(r for r, _ in win_cc) / len(win_cc)
    avg_c = sum(c for _, c in win_cc) / len(win_cc)
    cheb = int(round(max(abs(avg_r - 11.5), abs(avg_c - 11.5))))

    return {
        "cc_size": len(win_cc),
        "cc_axis_span": round(win_span, 4),
        "cc_touches_own_goal": touches,
        "axis_span_margin": round(win_span - lose_span, 4),
        "centroid_chebyshev_from_center": cheb,
        "forced_within_2": is_forced_within_k(state, winner, k=2),
    }


def _collect_pegs(state, color: str) -> list:
    """Return [(r, c), ...] of all pegs of `color` on the board."""
    pegs = []
    # Iterate every cell; cheap on 24x24.
    for r in range(state.active_size):
        for c in range(state.active_size):
            if state.peg_at(r, c) == color:
                pegs.append((r, c))
    return pegs


def _largest_connected_component(state, pegs: list, color: str) -> tuple:
    """Return (cc_cells, axis_span) for the largest knight-bridged component
    of `color`. axis_span is the fraction of `color`'s goal axis the
    component spans (red: row range / 23; black: col range / 23).
    """
    if not pegs:
        return [], 0.0
    bridges = _bridges_for_color(state, color)
    adj = {p: set() for p in pegs}
    for a, b in bridges:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)

    # Connected components via flood fill.
    seen = set()
    components = []
    for p in pegs:
        if p in seen:
            continue
        stack = [p]; comp = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x)
            stack.extend(adj[x] - seen)
        components.append(comp)

    largest = max(components, key=len)
    if color == "red":
        rows = [r for r, _ in largest]
        span = (max(rows) - min(rows)) / 23.0
    else:
        cols = [c for _, c in largest]
        span = (max(cols) - min(cols)) / 23.0
    return largest, span


def _bridges_for_color(state, color: str) -> list:
    """Return [(p1, p2), ...] of every knight-bridge currently held by
    `color` on `state`.
    """
    out = []
    # state.bridges (or the equivalent accessor) returns "r1,c1-r2,c2" keys
    # paired with the placing color via the peg endpoint. Mirror the same
    # iteration pattern as TwixtState.to_tensor() (probe_eval.py existing
    # helpers also use this pattern).
    for bkey in state.bridges:
        p1_str, p2_str = bkey.split("-")
        r1, c1 = (int(x) for x in p1_str.split(","))
        r2, c2 = (int(x) for x in p2_str.split(","))
        if state.peg_at(r1, c1) == color:
            out.append(((r1, c1), (r2, c2)))
    return out


def is_forced_within_k(state, player: str, k: int = 1) -> bool:
    """True if `player` (whose turn it is, or hypothetically) can force a
    win within k plies of play.

    Conservative implementation: only does a 1-ply lookahead — returns
    True iff `player` has any legal move that immediately wins. For k>1
    this is a lower bound (under-reports forced positions), which is
    safe for the strong_advantage filter ("exclude already-forced") —
    we'd rather over-admit a not-quite-forced candidate (Phase 2 MCTS
    will filter it) than under-admit a genuinely strong-advantage one.

    A future tightening can extend to a true negamax k-ply search; the
    interface accepts k for forward-compat.
    """
    if state.to_move != player:
        # Conservative: if it's not `player`'s turn, we can't say they have
        # a forced win on this ply. (A true k>=2 search would consider
        # opponent responses; the conservative version returns False.)
        return False
    for move in state.legal_moves():
        try:
            next_state = state.apply_move(move)
        except Exception:
            continue
        if next_state.is_terminal() and next_state.winner() == player:
            return True
    return False
```

The `forced_within_2` field name is preserved on the returned dict for spec consistency, but populated via `is_forced_within_k(state, winner, k=2)`. The k=2 argument is a hint for future implementation; today's body returns the same result for any k>=1.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py::test_phase1_features_red_chain_top_to_mid_board -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_strong_advantage_probe_suite.py
git commit -m "feat(probes): Phase-1 structural features for strong_advantage tier

compute_phase1_features() returns cc_size, cc_axis_span,
cc_touches_own_goal, axis_span_margin, centroid_chebyshev_from_center,
and forced_within_2 for a candidate position. Used by the upcoming
strong_advantage suite generator to filter candidates before deep-MCTS
labeling.
"
```

### Task 2.2 — Phase-1 candidate extraction with category assignment

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py` (add `extract_strong_advantage_candidates`)
- Modify: `tests/test_strong_advantage_probe_suite.py` (add tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_strong_advantage_probe_suite.py`:

```python
def _make_decisive_game_dict(winner_color, terminal_ply, moves):
    """Build the minimal game-record dict that probe_eval ingests."""
    return {
        "meta": {"iteration": 70},
        "winner": winner_color,
        "winner_reason": "win",
        "moves": [{"row": r, "col": c} for r, c in moves],
        "starting_player": "red",
    }


def test_extract_strong_advantage_candidates_drops_midband():
    """Mid-band centroid (Chebyshev 7-8) candidates are excluded with the
    category_midband audit reason; central and edge candidates survive.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    # Three synthetic decisive games — one with a clearly central winning
    # chain, one with a clearly edge winning chain, one mid-band.
    games = [
        _make_decisive_game_dict("red", 30, _central_red_chain()),
        _make_decisive_game_dict("red", 30, _edge_red_chain()),
        _make_decisive_game_dict("red", 30, _midband_red_chain()),
    ]
    candidates, audit = extract_strong_advantage_candidates(
        games, k_plies_range=(3, 8), category_min_count=0
    )
    cats = sorted(c["category"] for c in candidates)
    assert "chain_advantage_central_red" in cats
    assert "chain_advantage_edge_red" in cats
    assert all("midband" not in c["category"] for c in candidates)
    midband_drops = [a for a in audit if a["reason"] == "category_midband"]
    assert len(midband_drops) >= 1


def test_extract_strong_advantage_candidates_drops_low_axis_span_margin():
    """A candidate where the loser's chain is as long as the winner's is
    rejected via axis_span_margin < 0.10.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    games = [_make_decisive_game_dict("red", 20, _both_strong_chain())]
    candidates, audit = extract_strong_advantage_candidates(
        games, k_plies_range=(3, 8), category_min_count=0
    )
    assert candidates == []
    assert any(a["reason"] == "phase1_axis_span_margin" for a in audit)


# --- Synthetic chain helpers (kept brief; real positions are richer) ---

def _central_red_chain():
    # Red knight chain stays in cols 11-13; centroid Chebyshev <= 6 from center.
    base = [(0, 12), (2, 11), (4, 12), (6, 11), (8, 12), (10, 11), (12, 12),
            (14, 11), (16, 12), (18, 11), (20, 12), (22, 11)]
    return _interleave_with_filler(base, filler_col=22)


def _edge_red_chain():
    # Red knight chain hugs col 1; centroid Chebyshev >= 9 from center.
    base = [(0, 1), (2, 0), (4, 1), (6, 0), (8, 1), (10, 0), (12, 1),
            (14, 0), (16, 1), (18, 0), (20, 1), (22, 0)]
    return _interleave_with_filler(base, filler_col=15)


def _midband_red_chain():
    # Red chain clustered around col 7-8; centroid Chebyshev in 7..8 from center.
    base = [(0, 7), (2, 8), (4, 7), (6, 8), (8, 7), (10, 8)]
    return _interleave_with_filler(base, filler_col=18)


def _both_strong_chain():
    # Red chain rows 0-12, black chain cols 0-12 — both spans >= 0.50.
    red = [(0, 5), (2, 4), (4, 5), (6, 4), (8, 5), (10, 4), (12, 5)]
    black = [(0, 0), (1, 2), (2, 1), (3, 3), (4, 2), (5, 4), (6, 3),
             (7, 5), (8, 4), (9, 6), (10, 5), (11, 7), (12, 6)]
    out = []
    for i in range(max(len(red), len(black))):
        if i < len(red):
            out.append(red[i])
        if i < len(black):
            out.append(black[i])
    return out


def _interleave_with_filler(red_moves, filler_col):
    """Interleave red moves with throwaway black moves so plies alternate."""
    out = []
    for i, rm in enumerate(red_moves):
        out.append(rm)
        out.append((1 + (i % 22), filler_col))  # black filler in safe column
    return out
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py -v -k extract_strong_advantage`
Expected: FAIL with `ImportError: cannot import name 'extract_strong_advantage_candidates'`.

- [ ] **Step 3: Implement `extract_strong_advantage_candidates`**

Append to `scripts/GPU/alphazero/probe_eval.py`:

```python
def extract_strong_advantage_candidates(
    games: list,
    *,
    k_plies_range: tuple = (3, 8),
    min_cc_size: int = 10,
    min_cc_axis_span: float = 0.55,
    min_axis_span_margin: float = 0.10,
    require_cc_touches_own_goal: bool = True,
    exclude_forced_within_2: bool = True,
    category_min_count: int = 5,
) -> tuple:
    """Phase-1 candidate mining for the strong_advantage probe tier.

    Walks each decisive game, samples positions at terminal_ply - K for K
    in k_plies_range (inclusive on both ends), computes structural features
    on each, and applies the Phase-1 admission gate. See spec Phase 1.

    Args:
        games: list of game-record dicts (must contain `moves`, `winner`,
            `winner_reason`, optionally `starting_player`).
        k_plies_range: (min_K, max_K) plies before terminal to sample.
        min_cc_size, min_cc_axis_span, min_axis_span_margin: Phase-1
            heuristic thresholds.
        require_cc_touches_own_goal, exclude_forced_within_2: gate flags.
        category_min_count: warning threshold; if any of the 4 categories
            ends up with fewer surviving candidates than this, a warning
            is printed (the candidate list is returned regardless).

    Returns:
        (candidates, audit) where:
          candidates: list of dicts with `move_history`, `ply`, `winner`,
            `category`, `phase1_features`, `source_game`, `source_ply`,
            `starting_player`. Sorted by (-iter, -source_ply, source_game)
            for deterministic order.
          audit: list of dicts with `source_game`, `source_ply`, `reason`,
            and `phase1_features`. One entry per dropped candidate; the
            audit row is also written for ADMITTED candidates with reason
            "admitted" so the audit captures the full provenance.
    """
    from .game.twixt_state import TwixtState

    candidates = []
    audit = []

    for game in games:
        if game.get("winner_reason") != "win":
            continue
        winner = game.get("winner")
        if winner not in ("red", "black"):
            continue
        moves_list = game.get("moves") or []
        if not moves_list:
            continue
        terminal_ply = len(moves_list)
        starting_player = game.get("starting_player", "red")
        source_game = game.get("source_game") or _derive_source_game_basename(game)

        for k in range(k_plies_range[0], k_plies_range[1] + 1):
            target_ply = terminal_ply - k
            if target_ply < 1:
                continue

            state = TwixtState(active_size=24, to_move=starting_player)
            for i in range(target_ply):
                m = moves_list[i]
                state = state.apply_move((m["row"], m["col"]))

            feats = compute_phase1_features(state, winner=winner)
            base_audit = {
                "source_game": source_game,
                "source_ply": target_ply,
                "phase1_features": feats,
            }

            if feats["cc_size"] < min_cc_size:
                audit.append({**base_audit, "reason": "phase1_cc_size"})
                continue
            if feats["cc_axis_span"] < min_cc_axis_span:
                audit.append({**base_audit, "reason": "phase1_axis_span"})
                continue
            if feats["axis_span_margin"] < min_axis_span_margin:
                audit.append({**base_audit, "reason": "phase1_axis_span_margin"})
                continue
            if require_cc_touches_own_goal and not feats["cc_touches_own_goal"]:
                audit.append({**base_audit, "reason": "phase1_no_goal_touch"})
                continue
            if exclude_forced_within_2 and feats["forced_within_2"]:
                audit.append({**base_audit, "reason": "phase1_already_forced"})
                continue

            cheb = feats["centroid_chebyshev_from_center"]
            if 7 <= cheb <= 8:
                audit.append({**base_audit, "reason": "category_midband"})
                continue

            if cheb <= 6:
                category = f"chain_advantage_central_{winner}"
            else:  # cheb >= 9
                category = f"chain_advantage_edge_{winner}"

            cand = {
                "move_history": [(m["row"], m["col"]) for m in moves_list[:target_ply]],
                "ply": target_ply,
                "winner": winner,
                "category": category,
                "phase1_features": feats,
                "source_game": source_game,
                "source_ply": target_ply,
                "starting_player": starting_player,
            }
            candidates.append(cand)
            audit.append({**base_audit, "reason": "admitted"})

    # Deterministic sort: most-recent iter first, then deepest source_ply, then name.
    def _sort_key(c: dict) -> tuple:
        try:
            iter_num = int(c["source_game"].split("_")[1])
        except (IndexError, ValueError):
            iter_num = 0
        return (-iter_num, -c["source_ply"], c["source_game"])

    candidates.sort(key=_sort_key)

    # Falling-short warning per category.
    for cat in [
        "chain_advantage_central_red",
        "chain_advantage_central_black",
        "chain_advantage_edge_red",
        "chain_advantage_edge_black",
    ]:
        n = sum(1 for c in candidates if c["category"] == cat)
        if n < category_min_count:
            import sys
            print(
                f"[probe_suite] WARNING: category {cat} has {n} candidates "
                f"(< {category_min_count}); broaden --source-iter-range or "
                f"relax thresholds.",
                file=sys.stderr,
            )

    return candidates, audit


def _derive_source_game_basename(game: dict) -> str:
    """Best-effort recovery of the source_game basename from a game dict."""
    # Used when a game dict lacks an explicit source_game field (e.g. test
    # fixtures). Real games loaded from disk should already carry this.
    meta = game.get("meta") or {}
    iteration = meta.get("iteration", 0)
    game_idx = meta.get("game_index", 0)
    return f"iter_{iteration:04d}_game_{game_idx:03d}"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py -v -k extract_strong_advantage`
Expected: PASS for both `test_extract_strong_advantage_candidates_drops_midband` and `test_extract_strong_advantage_candidates_drops_low_axis_span_margin`.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_strong_advantage_probe_suite.py
git commit -m "feat(probes): Phase-1 candidate extraction for strong_advantage tier

extract_strong_advantage_candidates() walks decisive games, samples
positions at terminal_ply - K for K in [3, 8], applies the Phase-1
admission gate, and assigns categories (central/edge × red/black,
mid-band 7-8 chebyshev dropped to keep separation crisp). Returns
(candidates, audit) for downstream MCTS labeling.
"
```

### Task 2.3 — MCTS labeler with mock-injectable interface

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py` (add `label_candidate_with_mcts` and a `MctsLabeler` protocol-style callable signature)
- Modify: `tests/test_strong_advantage_probe_suite.py` (add a labeler-integration test using a stub)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_strong_advantage_probe_suite.py`:

```python
def test_label_candidate_with_mcts_uses_injected_labeler():
    """The labeler signature must be (state, sims, seed) -> (root_value,
    top1_share). Test that a stub labeler produces the expected aggregate
    (mean_root_value, value_per_run, value_stability, min_top1_share).
    """
    from scripts.GPU.alphazero.probe_eval import label_candidate_with_mcts

    state = _make_state([(0, 12), (1, 0), (2, 11)])

    canned = [(0.6, 0.30), (0.7, 0.25), (0.5, 0.40)]
    calls = []

    def stub_labeler(state, sims, seed):
        calls.append((sims, seed))
        return canned[len(calls) - 1]

    label = label_candidate_with_mcts(
        state, sims=10000, repeats=3,
        rng_seed_base=12345, labeler=stub_labeler,
    )
    assert calls == [(10000, 12345 ^ 0), (10000, 12345 ^ 1), (10000, 12345 ^ 2)]
    assert label["mean_root_value"] == pytest.approx(0.6)
    assert label["value_per_run"] == [0.6, 0.7, 0.5]
    assert label["value_stability"] == pytest.approx(0.2)
    assert label["min_top1_share"] == pytest.approx(0.25)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py::test_label_candidate_with_mcts_uses_injected_labeler -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `label_candidate_with_mcts`**

Append to `scripts/GPU/alphazero/probe_eval.py`:

```python
def label_candidate_with_mcts(
    state,
    *,
    sims: int,
    repeats: int,
    rng_seed_base: int,
    labeler=None,
) -> dict:
    """Phase-2 deep-MCTS labeling for one candidate position.

    Runs MCTS at `sims` simulations × `repeats` repeats with different RNG
    seeds per repeat. Aggregates per-run results.

    Args:
        state: TwixtState at the candidate position.
        sims: simulations per MCTS run.
        repeats: number of repeated MCTS runs.
        rng_seed_base: integer seed; per-run seed = rng_seed_base ^ repeat_idx.
        labeler: optional callable (state, sims, seed) -> (root_value,
            top1_share). If None, uses the production deep-MCTS labeler
            from `_default_mcts_labeler`. Tests inject a stub here.

    Returns:
        dict with mean_root_value, value_per_run, value_stability,
        min_top1_share, label_mcts_sims, label_mcts_repeats, rng_seed_base.
    """
    if labeler is None:
        labeler = _default_mcts_labeler

    values = []
    top1_shares = []
    for repeat_idx in range(repeats):
        seed = rng_seed_base ^ repeat_idx
        v, t1 = labeler(state, sims, seed)
        values.append(v)
        top1_shares.append(t1)

    return {
        "mean_root_value": round(sum(values) / len(values), 6),
        "value_per_run": [round(v, 6) for v in values],
        "value_stability": round(max(values) - min(values), 6),
        "min_top1_share": round(min(top1_shares), 6),
        "label_mcts_sims": sims,
        "label_mcts_repeats": repeats,
        "rng_seed_base": rng_seed_base,
    }


def _default_mcts_labeler(state, sims, seed):
    """Production deep-MCTS labeler. Loads the network passed via the
    closure-captured `--label-checkpoint` path (set by the generator
    entrypoint) and runs MCTS at the given sim count and seed.

    Returns (root_value_from_stm_perspective, top1_visit_share).

    Implementation note: the closure variable is set by the caller in
    build_probe_suite.py:_run_strong_advantage via _set_default_labeler_ckpt.
    Tests should pass an explicit `labeler=` rather than relying on this.
    """
    if _DEFAULT_LABELER_NETWORK is None:
        raise RuntimeError(
            "Default MCTS labeler called without a registered network. "
            "Either pass labeler= explicitly or call "
            "_set_default_labeler_network() first."
        )
    # Lazy import to avoid pulling MLX into pure-test paths.
    from .local_evaluator import LocalGPUEvaluator
    from .mcts import MCTS, MCTSConfig
    import random

    evaluator = LocalGPUEvaluator(_DEFAULT_LABELER_NETWORK)
    cfg = MCTSConfig(c_puct=1.5, n_simulations=sims)
    mcts = MCTS(evaluator, cfg, rng=random.Random(seed))
    result = mcts.search(state)
    # `result` is expected to be { 'root_value': float (stm POV),
    #   'visit_counts': {move_key: count}, 'total_visits': int, ... }
    visits = result.get("visit_counts") or {}
    total = sum(visits.values()) or 1
    top1 = max(visits.values()) if visits else 0
    return result.get("root_value", 0.0), top1 / total


_DEFAULT_LABELER_NETWORK = None


def _set_default_labeler_network(network) -> None:
    """Register the production network for `_default_mcts_labeler`."""
    global _DEFAULT_LABELER_NETWORK
    _DEFAULT_LABELER_NETWORK = network
```

Note: the actual MCTS interface in `scripts/GPU/alphazero/mcts.py` may differ slightly (e.g. `MCTSConfig` may be named differently, or `search()` may return a tuple). Inspect with `grep -n "class MCTS\|class MCTSConfig\|def search" scripts/GPU/alphazero/mcts.py` and adjust the labeler accordingly. The test stub doesn't exercise this path so the test will pass regardless.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py::test_label_candidate_with_mcts_uses_injected_labeler -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_strong_advantage_probe_suite.py
git commit -m "feat(probes): MCTS labeler interface for strong_advantage tier

label_candidate_with_mcts() runs N=repeats MCTS searches at the given
sim budget with deterministic per-run seeds and aggregates per-run
results. Production labeler is a thin wrapper over the existing
LocalGPUEvaluator + MCTS infra; tests inject a stub via the labeler= arg.
"
```

### Task 2.4 — Phase-2 admission filter (clause-by-clause)

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py` (add `apply_admission_filter`)
- Modify: `tests/test_strong_advantage_probe_suite.py` (add 5 clause-rejection tests)

- [ ] **Step 1: Add the failing tests (one per clause)**

Append to `tests/test_strong_advantage_probe_suite.py`:

```python
@pytest.fixture
def passing_candidate():
    """A candidate whose default Phase-2 evaluation passes every clause."""
    return {
        "winner": "red",
        "phase1_features": {
            "cc_size": 14,
            "cc_axis_span": 0.74,
            "cc_touches_own_goal": True,
            "axis_span_margin": 0.20,
            "centroid_chebyshev_from_center": 4,
            "forced_within_2": False,
        },
        "phase2_label": {
            "mean_root_value": 0.62,
            "value_per_run": [0.60, 0.65, 0.61],
            "value_stability": 0.05,
            "min_top1_share": 0.25,
            "label_mcts_sims": 10000,
            "label_mcts_repeats": 3,
            "rng_seed_base": 1,
        },
    }


def test_admission_passes_when_all_clauses_satisfied(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    admitted, reason = apply_admission_filter(
        passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
    )
    assert admitted is True
    assert reason == "admitted"


def test_admission_rejects_sign_mismatch(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["mean_root_value"] = -0.62
    passing_candidate["phase2_label"]["value_per_run"] = [-0.60, -0.65, -0.61]
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "sign_mismatch"


def test_admission_rejects_low_magnitude(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["mean_root_value"] = 0.30
    passing_candidate["phase2_label"]["value_per_run"] = [0.28, 0.32, 0.30]
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "magnitude_below_threshold"


def test_admission_rejects_low_top1_share(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["min_top1_share"] = 0.10
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "low_top1_share"


def test_admission_rejects_unstable_value(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["value_stability"] = 0.30
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "unstable_value"


def test_admission_rejects_already_forced(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase1_features"]["forced_within_2"] = True
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "position_already_forced"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py -v -k admission`
Expected: 6 FAIL with `ImportError: cannot import name 'apply_admission_filter'`.

- [ ] **Step 3: Implement `apply_admission_filter`**

Append to `scripts/GPU/alphazero/probe_eval.py`:

```python
def apply_admission_filter(
    candidate: dict,
    *,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
) -> tuple:
    """Phase-2 admission gate. Returns (admitted: bool, reason: str).

    Reason is one of:
      "admitted", "sign_mismatch", "magnitude_below_threshold",
      "low_top1_share", "unstable_value", "position_already_forced".

    Order of checks matters only for the audit reason — first failing
    clause is reported. Sign-match is checked first because it's the
    cross-check against the source-game winner.
    """
    label = candidate["phase2_label"]
    feats = candidate["phase1_features"]
    winner = candidate["winner"]
    expected_sign = 1 if winner == "red" else -1

    if (label["mean_root_value"] >= 0) != (expected_sign == 1):
        return False, "sign_mismatch"
    if abs(label["mean_root_value"]) < magnitude_threshold:
        return False, "magnitude_below_threshold"
    if label["min_top1_share"] < top1_share_floor:
        return False, "low_top1_share"
    if label["value_stability"] > stability_cap:
        return False, "unstable_value"
    if feats.get("forced_within_2"):
        return False, "position_already_forced"
    return True, "admitted"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py -v -k admission`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_strong_advantage_probe_suite.py
git commit -m "feat(probes): Phase-2 admission filter for strong_advantage tier

apply_admission_filter() applies all 5 admission clauses (sign-match with
source winner, magnitude, top1-share, stability, not-already-forced) and
returns the first failing clause as the audit reason. One test per clause.
"
```

### Task 2.5a — Wire `--tier strong_advantage` generation path (mine + label + filter + draft)

**Scope:** CLI validation, game loading, Phase 1 + Phase 2 wiring, admission filtering, draft + audit serialization. Promotion (`--promote`) and the helpers (`_probe_id_for`, `_stm_at_ply`, `main_with_args`) land in Task 2.5b.

**Files:**
- Modify: `scripts/build_probe_suite.py` (replace `_run_strong_advantage` stub; introduce `main_with_args` early because the test needs it)
- Modify: `tests/test_strong_advantage_probe_suite.py` (integration test for the generation path)

**Important — checkpoint architecture limitation.** This generator currently supports ONLY labeling checkpoints built with the `create_network` default architecture (`hidden=128`, `n_blocks=6`). `load_network_for_scoring` auto-detects input channels (24 vs 30) but does NOT auto-detect `hidden` or `n_blocks` — those fall back to defaults. The bundled checkpoint (`model_iter_0059`) is 128 / 6 so this works today. To label against any other architecture, the operator MUST first extend this generator with `--hidden` / `--blocks` flags (a follow-up task, not in this plan); otherwise weight-loading raises a tensor-shape mismatch and the generator exits non-zero. This is a deliberately narrow scope — the alternative (auto-detect from safetensors metadata) is out of scope for the current plan.

- [ ] **Step 1: Add the integration test**

Append to `tests/test_strong_advantage_probe_suite.py`:

```python
def test_run_strong_advantage_writes_draft_with_admitted_candidates(tmp_path, monkeypatch):
    """End-to-end on the generation path: mock candidate-extraction +
    labeler + checkpoint loader so the test runs without disk I/O. Two
    synthetic candidates feed in: one is admitted (high magnitude), the
    other is rejected (low magnitude). Assert the draft lists only the
    admitted probe.
    """
    import json
    import unittest.mock as _mock
    import scripts.build_probe_suite as bps

    sample_central = {
        "move_history": [(0, 12), (1, 0), (2, 11), (1, 1)],
        "ply": 4, "winner": "red",
        "category": "chain_advantage_central_red",
        "phase1_features": {
            "cc_size": 12, "cc_axis_span": 0.65, "cc_touches_own_goal": True,
            "axis_span_margin": 0.20, "centroid_chebyshev_from_center": 4,
            "forced_within_2": False,
        },
        "source_game": "iter_0070_game_001", "source_ply": 4,
        "starting_player": "red",
    }
    sample_edge = {
        "move_history": [(0, 1), (1, 22), (2, 0), (1, 21)],
        "ply": 4, "winner": "red",
        "category": "chain_advantage_edge_red",
        "phase1_features": {
            "cc_size": 11, "cc_axis_span": 0.60, "cc_touches_own_goal": True,
            "axis_span_margin": 0.15, "centroid_chebyshev_from_center": 10,
            "forced_within_2": False,
        },
        "source_game": "iter_0070_game_002", "source_ply": 4,
        "starting_player": "red",
    }

    def fake_extract(games, **kw):
        return [sample_central, sample_edge], []

    # The labeler is invoked via label_candidate_with_mcts(...). When the
    # generator passes a TwixtState already advanced through the candidate's
    # move_history, the central candidate has a red peg at (0, 12) and the
    # edge candidate has a red peg at (0, 1). Use that to discriminate.
    def fake_labeler(state, sims, seed):
        # Note: TwixtState has no peg_at() accessor — use pegs.get() instead.
        # See Task 2.1's implementer note for the verified API surface.
        if state.pegs.get((0, 12)) == "red":
            return (0.65, 0.30)
        return (0.20, 0.30)  # below magnitude_threshold

    # Replace candidate extraction with our synthetic pair.
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval.extract_strong_advantage_candidates",
        fake_extract,
    )
    # Replace the production labeler with the discriminating fake.
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval._default_mcts_labeler",
        fake_labeler,
    )
    # Critical: the generator calls load_network_for_scoring(str(label_ckpt)).
    # We mock it to return a no-op network; otherwise it tries to read
    # /dev/null (or whatever placeholder we passed) as a real safetensors
    # file and the test fails before the fake labeler is reached.
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval.load_network_for_scoring",
        lambda *_a, **_kw: (_mock.MagicMock(), 30, 128, 6),
    )
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval._set_default_labeler_network",
        lambda _net: None,
    )
    # The generator also computes sha256 of the checkpoint file. Mock the
    # path so the file appears to exist with stable contents.
    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")

    out_path = tmp_path / "strong_advantage_probes.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--magnitude-threshold", "0.45",
        "--out", str(out_path),
    ])
    assert rc == 0
    draft = out_path.with_suffix(".draft.json")
    assert draft.exists(), f"expected draft at {draft}; out_path was {out_path}"

    payload = json.loads(draft.read_text())
    assert payload["meta"]["tier"] == "strong_advantage"
    assert len(payload["probes"]) == 1
    assert payload["probes"][0]["category"] == "chain_advantage_central_red"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py::test_run_strong_advantage_writes_draft_with_admitted_candidates -v`
Expected: FAIL — `_run_strong_advantage` is still the `NotImplementedError` stub from Task 1.2 (and `main_with_args` doesn't exist yet).

- [ ] **Step 3: Replace the `_run_strong_advantage` stub with the generation path**

In `scripts/build_probe_suite.py`, replace the `_run_strong_advantage(args)` stub with the implementation below, AND add `main_with_args` and the two small helpers (`_probe_id_for`, `_stm_at_ply`) the generation path needs. (The `_run_promote` helper lands in Task 2.5b — for now `_run_strong_advantage` rejects `--promote` with a clear error message.)

```python
def _run_strong_advantage(args) -> int:
    if args.out is None:
        args.out = "tests/probes/strong_advantage_probes.json"

    if args.promote:
        print("[probe_suite] ERROR: --promote not implemented yet "
              "(lands in Task 2.5b).", file=sys.stderr)
        return 2

    if args.label_checkpoint is None:
        print("[probe_suite] ERROR: --label-checkpoint required for "
              "--tier strong_advantage (when not --promote).", file=sys.stderr)
        return 2
    if args.source_iter_range is None:
        print("[probe_suite] ERROR: --source-iter-range required for "
              "--tier strong_advantage.", file=sys.stderr)
        return 2

    label_ckpt = Path(args.label_checkpoint)
    if not label_ckpt.exists():
        print(f"[probe_suite] ERROR: --label-checkpoint not found: {label_ckpt}",
              file=sys.stderr)
        return 2

    out_path = Path(args.out)
    draft_path = out_path.with_suffix(".draft.json")
    audit_path = out_path.parent / "candidates_strong_advantage.json"
    if draft_path.exists() and not args.force:
        print(f"[probe_suite] ERROR: draft already exists: {draft_path}\n"
              f"  Pass --force to overwrite, or delete the existing draft.",
              file=sys.stderr)
        return 2

    from scripts.GPU.alphazero.probe_eval import (
        extract_strong_advantage_candidates,
        label_candidate_with_mcts,
        apply_admission_filter,
        _set_default_labeler_network,
        load_network_for_scoring,
    )
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    # Phase 1: load games, mine candidates.
    min_iter, max_iter = args.source_iter_range
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[probe_suite] ERROR: --input not a directory: {input_dir}",
              file=sys.stderr)
        return 2
    games = []
    for fp in sorted(input_dir.glob("iter_*_game_*.json")):
        with open(fp) as f:
            try:
                g = json.load(f)
            except json.JSONDecodeError:
                continue
        iteration = (g.get("meta") or {}).get("iteration")
        if iteration is None or not (min_iter <= iteration <= max_iter):
            continue
        g["source_game"] = fp.stem
        games.append(g)

    candidates, audit = extract_strong_advantage_candidates(games)
    print(f"[probe_suite] Phase 1: {len(candidates)} candidates from "
          f"{len(games)} games")

    # Phase 2: load network, label each candidate, apply admission filter.
    # IMPORTANT: this generator currently supports ONLY labeling checkpoints
    # built with create_network defaults (hidden=128, n_blocks=6).
    # load_network_for_scoring auto-detects input channels (24 vs 30) but
    # does NOT auto-detect hidden/n_blocks. To label against a checkpoint
    # with a different architecture, this generator must first be extended
    # with --hidden/--blocks flags (follow-up); the call below will
    # otherwise raise a tensor-shape mismatch and abort the run.
    network, _ic, _h, _nb = load_network_for_scoring(str(label_ckpt))
    network.eval()
    _set_default_labeler_network(network)

    admitted = []
    for cand in candidates:
        state = TwixtState(active_size=24, to_move=cand["starting_player"])
        for r, c in cand["move_history"]:
            state = state.apply_move((r, c))

        # Stable seed: SHA-256 of probe ID, first 4 bytes as big-endian int.
        # Python's built-in hash() is process-randomized and would break
        # byte-reproducibility across runs.
        import hashlib
        seed_base = int.from_bytes(
            hashlib.sha256(_probe_id_for(cand).encode("utf-8")).digest()[:4],
            "big",
        )

        try:
            label = label_candidate_with_mcts(
                state,
                sims=args.label_mcts_sims,
                repeats=args.label_mcts_repeats,
                rng_seed_base=seed_base,
            )
        except Exception as exc:
            print(f"[probe_suite] WARN: MCTS error on {cand['source_game']} "
                  f"ply {cand['source_ply']}: {exc}", file=sys.stderr)
            audit.append({
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "reason": "mcts_error",
            })
            continue

        cand["phase2_label"] = label
        ok, reason = apply_admission_filter(
            cand,
            magnitude_threshold=args.magnitude_threshold,
            top1_share_floor=args.top1_share_floor,
            stability_cap=args.stability_cap,
        )
        cand["phase2_label"]["label_checkpoint"] = label_ckpt.name
        audit.append({
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
            "reason": reason,
        })
        if ok:
            admitted.append(cand)

    if not admitted:
        from collections import Counter
        reason_counts = Counter(a["reason"] for a in audit if a["reason"] != "admitted")
        msg = ", ".join(f"{r}: {n}" for r, n in reason_counts.most_common())
        print(f"[probe_suite] ERROR: 0 admitted probes overall.\n"
              f"  Drop reasons: {msg}", file=sys.stderr)
        return 1

    admitted = admitted[: args.max_probes]

    probes_out = []
    for cand in admitted:
        probes_out.append({
            "id": _probe_id_for(cand),
            "category": cand["category"],
            "confidence": "strong_advantage",
            "side_to_move": _stm_at_ply(cand),
            "expected_value_sign": 1 if cand["winner"] == "red" else -1,
            "active_size": 24,
            "ply": cand["ply"],
            "move_history": cand["move_history"],
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "starting_player": cand["starting_player"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
        })

    import hashlib
    ckpt_hash = hashlib.sha256(label_ckpt.read_bytes()).hexdigest()
    payload = {
        "meta": {
            "type": "bootstrap_rule_selected",
            "tier": "strong_advantage",
            "not_gate_suite": True,
            "review_mode": "draft",
            "reviewer": None,
            "reviewed_at_utc": None,
            "generator": "scripts/build_probe_suite.py",
            "generator_version": 1,
            "selection_rules": {
                "board_size": 24,
                "winner_reasons": ["win"],
                "k_plies_from_terminal_range": [3, 8],
                "phase1_thresholds": {
                    "min_cc_size": 10,
                    "min_cc_axis_span": 0.55,
                    "min_axis_span_margin": 0.10,
                    "require_cc_touches_own_goal": True,
                    "exclude_forced_within_2": True,
                },
                "phase2_thresholds": {
                    "label_mcts_sims": args.label_mcts_sims,
                    "label_mcts_repeats": args.label_mcts_repeats,
                    "min_magnitude": args.magnitude_threshold,
                    "min_top1_share": args.top1_share_floor,
                    "max_value_stability": args.stability_cap,
                    "require_sign_match_source_winner": True,
                },
                "label_checkpoint": str(label_ckpt),
                "label_checkpoint_sha256": ckpt_hash,
                "source_iter_range": [min_iter, max_iter],
                "dedup": "exact + 4-form-mirror-canonical",
                "category_min_count": 5,
            },
        },
        "probes": probes_out,
    }

    draft_path.parent.mkdir(parents=True, exist_ok=True)
    with open(draft_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    with open(audit_path, "w") as f:
        json.dump({"audit": audit}, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"[probe_suite] wrote {len(probes_out)} candidates to draft "
          f"{draft_path}\n  audit: {audit_path}\n"
          f"  Next: review the draft, then run --promote --reviewer NAME "
          f"(lands in Task 2.5b).")
    return 0


def _probe_id_for(cand: dict) -> str:
    """Deterministic probe ID: iter_NNNN_game_MMM_plyNNN_<category>."""
    return (
        f"{cand['source_game']}_ply{cand['source_ply']:03d}_{cand['category']}"
    )


def _stm_at_ply(cand: dict) -> str:
    """Whose turn it is at the candidate position (the side ABOUT to move)."""
    plies_played = cand["source_ply"]
    starting = cand["starting_player"]
    if plies_played % 2 == 0:
        return starting
    return "black" if starting == "red" else "red"


def main_with_args(argv: list) -> int:
    """Test entrypoint: invokes main() with explicit args (sys.argv-style)."""
    saved = sys.argv
    sys.argv = ["build_probe_suite.py", *argv]
    try:
        return main() or 0
    finally:
        sys.argv = saved
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py::test_run_strong_advantage_writes_draft_with_admitted_candidates -v`
Expected: PASS.

- [ ] **Step 5: Verify forced parity test still passes**

Run: `.venv/bin/pytest tests/test_probe_suite_forced_parity.py -v`
Expected: PASS (the strong_advantage path doesn't touch the forced path).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_probe_suite.py tests/test_strong_advantage_probe_suite.py
git commit -m "feat(probes): wire --tier strong_advantage generation path

Phase 1 (mine) + Phase 2 (label + filter) wired end-to-end. Writes
*.draft.json plus a candidates_strong_advantage.json audit row per
candidate (admitted or dropped, with reason).

Seed for MCTS labeling is SHA-256(probe_id)[:4] (big-endian) so output
is byte-reproducible across processes — built-in Python hash() would
not be (process-randomized).

Promotion (--promote) lands in Task 2.5b; for now --promote returns
ERROR with a pointer to that follow-up task.
"
```

### Task 2.5b — `--promote` workflow + serialization helpers

**Scope:** the `_run_promote` function, reviewer/timestamp handling, overwrite rules. Delivers the operator's draft → committed transition that Task 2.5a deferred.

**Files:**
- Modify: `scripts/build_probe_suite.py` (add `_run_promote`, replace the `--promote` ERROR stub in `_run_strong_advantage`)

The promotion-workflow tests (Task 2.6) also exercise this code, so this task only delivers the implementation; the tests come immediately after.

- [ ] **Step 1: Add `_run_promote` to `scripts/build_probe_suite.py`**

```python
def _run_promote(args) -> int:
    """Promote a *.draft.json to the committed file.

    Stamps meta.review_mode="light_review", meta.reviewer, and
    meta.reviewed_at_utc. Refuses to overwrite an existing committed
    file unless --force is passed.
    """
    if not args.reviewer:
        print("[probe_suite] ERROR: --reviewer required with --promote",
              file=sys.stderr)
        return 2
    out_path = Path(args.out)
    draft_path = out_path.with_suffix(".draft.json")
    if not draft_path.exists():
        print(f"[probe_suite] ERROR: no draft to promote at {draft_path}",
              file=sys.stderr)
        return 2
    if out_path.exists() and not args.force:
        print(f"[probe_suite] ERROR: committed file exists: {out_path}\n"
              f"  Pass --force to overwrite (deliberate re-promotion).",
              file=sys.stderr)
        return 2

    import datetime as _dt
    payload = json.loads(draft_path.read_text())
    payload["meta"]["review_mode"] = "light_review"
    payload["meta"]["reviewer"] = args.reviewer
    payload["meta"]["reviewed_at_utc"] = (
        _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"[probe_suite] promoted {draft_path} -> {out_path} "
          f"(reviewer={args.reviewer})")
    return 0
```

- [ ] **Step 2: Replace the `--promote` ERROR stub in `_run_strong_advantage`**

Find this block in `_run_strong_advantage` (added in Task 2.5a):

```python
    if args.promote:
        print("[probe_suite] ERROR: --promote not implemented yet "
              "(lands in Task 2.5b).", file=sys.stderr)
        return 2
```

Replace with:

```python
    if args.promote:
        return _run_promote(args)
```

- [ ] **Step 3: Smoke-check the promote path with a hand-built draft**

```bash
mkdir -p /tmp/promote_smoke
cat > /tmp/promote_smoke/x.draft.json <<'JSON'
{"meta": {"tier": "strong_advantage", "review_mode": "draft"}, "probes": []}
JSON

.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage \
    --promote \
    --reviewer "smoke" \
    --out /tmp/promote_smoke/x.json
```

Expected: writes `/tmp/promote_smoke/x.json` with `meta.review_mode == "light_review"`, `meta.reviewer == "smoke"`, `meta.reviewed_at_utc` populated.

Verify:
```bash
.venv/bin/python -c "import json; m=json.load(open('/tmp/promote_smoke/x.json'))['meta']; print(m['review_mode'], m['reviewer'], m['reviewed_at_utc'])"
```
Expected: `light_review smoke 2026-...Z`.

Run again (without `--force`):
```bash
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage --promote --reviewer "smoke" \
    --out /tmp/promote_smoke/x.json
```
Expected: ERROR exit 2 ("committed file exists").

- [ ] **Step 4: Verify the generation-path test still passes**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py::test_run_strong_advantage_writes_draft_with_admitted_candidates -v`
Expected: PASS (the `--promote` path is unrelated to the generation path).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_probe_suite.py
git commit -m "feat(probes): --promote workflow for strong_advantage tier

Promotes *.draft.json to the committed file with reviewer + UTC
timestamp + review_mode='light_review' stamped into meta. Refuses to
overwrite an existing committed file unless --force is passed
(deliberate re-promotion).
"
```

### Task 2.6 — Promotion workflow tests

**Files:**
- Modify: `tests/test_strong_advantage_probe_suite.py` (add 3 promotion tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_strong_advantage_probe_suite.py`:

```python
def test_promote_errors_with_no_draft(tmp_path):
    import scripts.build_probe_suite as bps
    out = tmp_path / "x.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "tester",
        "--out", str(out),
    ])
    assert rc != 0


def test_promote_writes_committed_with_reviewer_and_timestamp(tmp_path):
    import json, datetime
    import scripts.build_probe_suite as bps
    out = tmp_path / "x.json"
    draft = out.with_suffix(".draft.json")
    draft.write_text(json.dumps({
        "meta": {"tier": "strong_advantage", "review_mode": "draft"},
        "probes": [],
    }))
    rc = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "alice",
        "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["meta"]["review_mode"] == "light_review"
    assert payload["meta"]["reviewer"] == "alice"
    ts = payload["meta"]["reviewed_at_utc"]
    # Round-trips as ISO 8601 UTC.
    assert ts.endswith("Z")
    datetime.datetime.fromisoformat(ts[:-1])


def test_promote_refuses_overwrite_without_force(tmp_path):
    import json
    import scripts.build_probe_suite as bps
    out = tmp_path / "x.json"
    draft = out.with_suffix(".draft.json")
    draft.write_text(json.dumps({
        "meta": {"tier": "strong_advantage", "review_mode": "draft"},
        "probes": [],
    }))
    out.write_text("{}")  # pre-existing committed file
    rc1 = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "alice", "--out", str(out),
    ])
    assert rc1 != 0  # refused
    # With --force it succeeds
    rc2 = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "alice", "--force", "--out", str(out),
    ])
    assert rc2 == 0
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py -v -k promote`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_strong_advantage_probe_suite.py
git commit -m "test(probes): promotion workflow coverage for strong_advantage tier

Three tests: --promote errors without a draft, --promote writes the
committed file with reviewer + UTC timestamp, --promote refuses to
overwrite an existing committed file without --force.
"
```

### Task 2.7 — Opt-in live smoke test

**Files:**
- Create: `tests/test_strong_advantage_smoke_live.py`

- [ ] **Step 1: Write the marker-gated live test**

```python
"""Opt-in live smoke test for the strong_advantage labeling path.

Goal: confirm the labeling code path runs end-to-end without crashing —
checkpoint load, candidate replay, MCTS label call, admission filter,
draft output. NOT a label-correctness test.

Marker-gated; run with:
    .venv/bin/pytest -m slow_live tests/test_strong_advantage_smoke_live.py

Requires checkpoints/alphazero-v2-staged/model_iter_0059.safetensors on
disk and at least one decisive game in scripts/GPU/logs/games for the
hard-coded iter range below.
"""
from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT = PROJECT_ROOT / "checkpoints" / "alphazero-v2-staged" / "model_iter_0059.safetensors"


@pytest.mark.slow_live
def test_strong_advantage_smoke_live(tmp_path):
    if not CKPT.exists():
        pytest.skip(f"checkpoint not present: {CKPT}")

    import scripts.build_probe_suite as bps

    out_path = tmp_path / "smoke.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "25", "26",
        "--label-checkpoint", str(CKPT),
        "--label-mcts-sims", "200",
        "--label-mcts-repeats", "1",
        "--max-probes", "3",
        "--out", str(out_path),
    ])
    # rc 0 (admitted some probes) OR rc 1 (zero admitted — drop reasons logged)
    # are both acceptable plumbing-wise. rc 2 is a usage/config error.
    assert rc in (0, 1), f"smoke run errored at config/usage stage: rc={rc}"

    if rc == 0:
        draft = out_path.with_suffix(".draft.json")
        assert draft.exists()
```

- [ ] **Step 2: Register the marker in pytest config**

Modify `pyproject.toml` (or `pytest.ini`) to register `slow_live`. Search for an existing `[tool.pytest.ini_options]` section. If not present, add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow_live: opt-in live tests that hit real models or long-running pipelines",
]
```

If a markers list already exists, append to it. (Without registration, pytest emits a warning per run but the test still runs.)

- [ ] **Step 3: Verify the test is discoverable but not run by default**

Run: `.venv/bin/pytest tests/test_strong_advantage_smoke_live.py -v --collect-only`
Expected: collected `test_strong_advantage_smoke_live` listed.

Run: `.venv/bin/pytest tests/test_strong_advantage_smoke_live.py -v`
Expected: 1 deselected (because no `-m slow_live`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_strong_advantage_smoke_live.py pyproject.toml
git commit -m "test(probes): opt-in live smoke for strong_advantage labeling

Marker-gated (slow_live). Runs the full pipeline at sims=200 repeats=1
for 1-3 probes. Goal is plumbing coverage — checkpoint load, candidate
replay, MCTS label call, admission filter, draft output — not label
quality. Skipped if the checkpoint isn't on disk.
"
```

### Task 2.8 — Operator step: produce the first committed `strong_advantage_probes.json`

This task is run by an operator AFTER the code lands. It is not a CI test.

- [ ] **Step 1: Run the generator**

```bash
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage \
    --input scripts/GPU/logs/games \
    --source-iter-range 50 80 \
    --label-checkpoint checkpoints/alphazero-v2-staged/model_iter_0059.safetensors \
    --label-mcts-sims 10000 \
    --label-mcts-repeats 3 \
    --max-probes 30 \
    --out tests/probes/strong_advantage_probes.json
```

Expected: writes `tests/probes/strong_advantage_probes.draft.json` and `tests/probes/candidates_strong_advantage.json`. Wallclock: minutes (depends on hardware; ~30 candidates × 10k sims × 3 repeats).

If the generator errors with "0 admitted probes overall", widen `--source-iter-range` or relax `--magnitude-threshold` (e.g. 0.40) and retry.

- [ ] **Step 2: Eyeball the draft**

Open `tests/probes/strong_advantage_probes.draft.json` and skim each probe (~10–20 min for 30 probes). For each suspicious entry, replay the move history mentally or via the analyzer; remove anything that looks like a tactical trap rather than a genuine strong-advantage position.

If any probe is removed, edit the draft file directly (delete the entry from `probes[]`).

- [ ] **Step 3: Promote**

```bash
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage \
    --promote \
    --reviewer "$(git config user.name)" \
    --out tests/probes/strong_advantage_probes.json
```

Expected: writes `tests/probes/strong_advantage_probes.json` with `meta.review_mode = "light_review"`, `meta.reviewer`, `meta.reviewed_at_utc` populated.

- [ ] **Step 4: Verify schema**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py -v -k schema`

(The schema test is added in Task 2.9 if not yet present; if absent, skip this step until that task lands.)

- [ ] **Step 5: Commit the committed file (NOT the draft or audit)**

```bash
# Add the committed file but NOT the draft (which is intermediate) and
# NOT the audit (which is gitignored).
git add tests/probes/strong_advantage_probes.json
echo "tests/probes/strong_advantage_probes.draft.json" >> .gitignore
echo "tests/probes/candidates_strong_advantage.json" >> .gitignore
git add .gitignore
git commit -m "data(probes): seed committed strong_advantage_probes.json

Generated via build_probe_suite.py --tier strong_advantage from iters
50-80 with model_iter_0059 as label authority (10k MCTS sims × 3 repeats).
Reviewer: $(git config user.name).
"
```

### Task 2.9 — Schema validation test for the committed strong_advantage file

**Files:**
- Modify: `tests/test_strong_advantage_probe_suite.py` (add schema + meta tests against the committed file)

- [ ] **Step 1: Add the schema tests**

Append to `tests/test_strong_advantage_probe_suite.py`:

```python
COMMITTED_STRONG_SUITE = PROJECT_ROOT / "tests" / "probes" / "strong_advantage_probes.json"


def _load_committed_suite():
    if not COMMITTED_STRONG_SUITE.exists():
        pytest.skip("committed strong_advantage_probes.json not present yet")
    import json
    return json.loads(COMMITTED_STRONG_SUITE.read_text())


def test_committed_meta_block_well_formed():
    suite = _load_committed_suite()
    meta = suite["meta"]
    assert meta["tier"] == "strong_advantage"
    assert meta["not_gate_suite"] is True
    assert meta["review_mode"] == "light_review"
    assert isinstance(meta["reviewer"], str) and meta["reviewer"]
    assert meta["reviewed_at_utc"].endswith("Z")
    sha = meta["selection_rules"]["label_checkpoint_sha256"]
    assert isinstance(sha, str) and len(sha) == 64 and all(
        c in "0123456789abcdef" for c in sha
    )


def test_committed_probes_have_required_fields():
    suite = _load_committed_suite()
    valid_categories = {
        "chain_advantage_central_red", "chain_advantage_central_black",
        "chain_advantage_edge_red", "chain_advantage_edge_black",
    }
    for p in suite["probes"]:
        assert p["confidence"] == "strong_advantage"
        assert p["category"] in valid_categories
        assert p["side_to_move"] in ("red", "black")
        assert p["expected_value_sign"] in (-1, 1)
        assert isinstance(p["move_history"], list)
        # phase1_features: 5 keys
        feats = p["phase1_features"]
        assert set(feats.keys()) == {
            "cc_size", "cc_axis_span", "cc_touches_own_goal",
            "axis_span_margin", "centroid_chebyshev_from_center",
            "forced_within_2",
        }
        # phase2_label: 8 keys
        label = p["phase2_label"]
        assert set(label.keys()) >= {
            "mean_root_value", "value_per_run", "value_stability",
            "min_top1_share", "label_checkpoint", "label_mcts_sims",
            "label_mcts_repeats", "rng_seed_base",
        }
        assert isinstance(label["rng_seed_base"], int)
```

- [ ] **Step 2: Run the schema tests**

Run: `.venv/bin/pytest tests/test_strong_advantage_probe_suite.py -v -k committed`
Expected: 2 PASS (or 2 SKIP if Task 2.8 hasn't been run yet).

- [ ] **Step 3: Commit**

```bash
git add tests/test_strong_advantage_probe_suite.py
git commit -m "test(probes): schema validation for committed strong_advantage suite

Two tests: committed meta block is well-formed (tier, review_mode,
reviewer, sha256), and every probe has the required top-level fields,
correct phase1_features keys, and correct phase2_label keys. Skips
gracefully if the committed file isn't present yet (operator step).
"
```

---

## Step 3 — Trainer telemetry

The trainer already writes a `forced_probe_summary` block per-iter sidecar (`scripts/GPU/alphazero/trainer.py:2883`). Step 3 adds a sibling tier-keyed `probe_summary.{forced,strong_advantage}` block. The legacy field stays in place for one release cycle (deprecation handled in the deferred Step 5 plan).

### Task 3.1 — Add tier-keyed `probe_summary` emission

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (around line 2883, where sidecar is emitted)
- Create: `tests/test_trainer_probe_summary_emission.py`

- [ ] **Step 1: Write the failing test**

```python
"""Verify the trainer's per-iter sidecar carries probe_summary keyed by
tier alongside the legacy forced_probe_summary field.

Doesn't run training; uses a synthetic iter dict and the trainer's
sidecar serialization helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_sidecar_contains_tiered_probe_summary(tmp_path):
    """The sidecar dict assembled by the trainer for each iter must contain
    both `forced_probe_summary` (legacy) and `probe_summary.forced`
    populated with the same payload. probe_summary.strong_advantage may be
    null when the strong_advantage probes file is absent.
    """
    # The trainer assembles the sidecar dict inline; we synthesize the
    # field-build path by reading the function and exercising the
    # post-build helper directly. The helper is added in Step 3.1.
    from scripts.GPU.alphazero.trainer import build_probe_summary_block

    forced_payload = {
        "n": 28, "sign_correct": 25, "sign_correct_pct": 0.893,
        "median_abs_v": 0.61, "delta_sign_correct_pct": 0.02,
        "delta_median_abs_v": 0.01, "rolling5_sign_correct_pct": 0.86,
        "rolling5_median_abs_v": 0.59, "n_skipped_size": 0,
    }

    block = build_probe_summary_block(
        forced_summary=forced_payload,
        strong_advantage_summary=None,
    )
    assert block == {"forced": forced_payload, "strong_advantage": None}

    sa_payload = dict(forced_payload, n=20, sign_correct=14)
    block2 = build_probe_summary_block(
        forced_summary=forced_payload,
        strong_advantage_summary=sa_payload,
    )
    assert block2 == {"forced": forced_payload, "strong_advantage": sa_payload}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_trainer_probe_summary_emission.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_probe_summary_block'`.

- [ ] **Step 3: Add the helper to trainer.py**

Find a location near the existing forced_probe handling in `scripts/GPU/alphazero/trainer.py` (around the function that contains line 2883). At the top level of the module (above the function that builds the sidecar), add:

```python
def build_probe_summary_block(
    forced_summary,
    strong_advantage_summary,
):
    """Assemble the tier-keyed probe_summary block for the per-iter sidecar.

    Each value is the per-tier summary payload (same shape as the existing
    forced_probe_summary), or None when that tier didn't run this iter
    (probes file absent or inline eval disabled).
    """
    return {
        "forced": forced_summary,
        "strong_advantage": strong_advantage_summary,
    }
```

- [ ] **Step 4: Use the helper in the sidecar build**

In `scripts/GPU/alphazero/trainer.py`, find the dict literal that contains `"forced_probe_summary": forced_probe_summary,` (around line 2883). Add a sibling field immediately after it:

```python
                "forced_probe_summary": forced_probe_summary,
                "probe_summary": build_probe_summary_block(
                    forced_summary=forced_probe_summary,
                    strong_advantage_summary=None,  # populated when the
                                                    # strong_advantage inline
                                                    # eval lands (out of scope
                                                    # for this plan).
                ),
```

The `strong_advantage_summary=None` is intentional: the inline eval for the strong_advantage tier (analogous to `run_forced_probes_inline`) is not part of this plan's scope. Plumbing the field as `None` for now means the analyzer sees it consistently; populating it in a future plan only requires changing the kwarg.

- [ ] **Step 5: Run the test**

Run: `.venv/bin/pytest tests/test_trainer_probe_summary_emission.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_trainer_probe_summary_emission.py
git commit -m "feat(trainer): emit tier-keyed probe_summary alongside legacy field

Per-iter sidecar now carries:
  - forced_probe_summary: <payload>     (legacy, kept for compat)
  - probe_summary.forced: <same payload>
  - probe_summary.strong_advantage: null  (placeholder; inline eval lands
                                            in a follow-up plan)

Analyzer (Step 4) prefers the tiered structure; legacy field readers
continue to see what they expect.
"
```

---

## Step 4 — Analyzer aggregation

The analyzer currently aggregates `forced_probe_summary` from per-iter sidecars into `agg["forced_probe_by_iter"]` / `agg["forced_probe_latest"]` and emits a `forced_probe` block in `summary.json`, a `forced_probe_by_iter.csv`, and a section in `report.txt`. Step 4 generalizes this over a list of tier names so adding `strong_advantage` doesn't duplicate code.

### Task 4.1 — Tier-parameterized aggregation helper

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` (around lines 432-446, where forced is aggregated)

- [ ] **Step 1: Locate the forced aggregation block**

Read `scripts/twixt_replay_analyzer.py:430-450`:

```python
        fps = sc.get("forced_probe_summary")
        if fps:
            agg["forced_probe_by_iter"].append({...})
        ...
        if fps:
            agg["forced_probe_latest"] = fps
```

- [ ] **Step 2: Replace with the generalized helper**

Add this helper near the top of the module (above the function that contains the aggregation):

```python
TIER_NAMES = ("forced", "strong_advantage")


def _read_tier_summary(sc: dict, tier: str):
    """Read a per-iter sidecar's summary for `tier`. Prefers the new
    `probe_summary.<tier>` shape; falls back to the legacy
    `forced_probe_summary` field for tier == "forced".
    """
    ps = sc.get("probe_summary") or {}
    if tier in ps and ps[tier] is not None:
        return ps[tier]
    if tier == "forced":
        return sc.get("forced_probe_summary")
    return None
```

In the aggregation function (the one that currently has `fps = sc.get("forced_probe_summary")`), replace the block that does the by_iter append + the latest assignment with a tier-parameterized loop:

```python
        # Phase 2: per-iter inline probes, parameterized over all known tiers.
        for tier in TIER_NAMES:
            tps = _read_tier_summary(sc, tier)
            if not tps:
                continue
            agg.setdefault(f"{tier}_probe_by_iter", []).append({
                "iteration": it,
                "n": tps.get("n"),
                "n_skipped_size": tps.get("n_skipped_size"),
                "sign_correct": tps.get("sign_correct"),
                "sign_correct_pct": tps.get("sign_correct_pct"),
                "median_abs_v": tps.get("median_abs_v"),
                "delta_sign_correct_pct": tps.get("delta_sign_correct_pct"),
                "delta_median_abs_v": tps.get("delta_median_abs_v"),
                "rolling5_sign_correct_pct": tps.get("rolling5_sign_correct_pct"),
                "rolling5_median_abs_v": tps.get("rolling5_median_abs_v"),
            })
            if it == latest_it:
                agg[f"{tier}_probe_latest"] = tps
```

Also initialize the aggregate dict to include the new keys. Find the block that initializes `agg["forced_probe_by_iter"]` and `agg["forced_probe_latest"]` (around line 374) and parameterize:

```python
    for tier in TIER_NAMES:
        agg[f"{tier}_probe_by_iter"] = []
        agg[f"{tier}_probe_latest"] = {}
```

(Replace the two existing forced-only init lines.)

- [ ] **Step 3: Update the summary.json emission**

Find the existing `"forced_probe": {...}` block in the `summary.json` payload assembly (around line 1925). Replace with a parameterized loop:

```python
        **{
            f"{tier}_probe": {
                "by_iter": sc_agg.get(f"{tier}_probe_by_iter", []) if use_sidecar else [],
                "latest": sc_agg.get(f"{tier}_probe_latest", {}) if use_sidecar else {},
            }
            for tier in TIER_NAMES
        },
```

- [ ] **Step 4: Update the CSV emission**

Find the block that writes `forced_probe_by_iter.csv` (around line 1988). Replace with:

```python
        for tier in TIER_NAMES:
            tier_rows = sc_agg.get(f"{tier}_probe_by_iter", [])
            if tier_rows:
                tier_csv_path = os.path.join(
                    out_dir,
                    _suffixed(f"{tier}_probe_by_iter", "csv", suffix),
                )
                with open(tier_csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(tier_rows[0].keys()))
                    w.writeheader()
                    for row in tier_rows:
                        w.writerow(row)
                print(f"[OK] wrote: {tier_csv_path}")
```

- [ ] **Step 5: Add the report.txt section**

Find `format_forced_probe_report` (around line 1006). Generalize the function signature:

```python
def format_tier_probe_report(tier: str, by_iter: list, latest: dict) -> List[str]:
    """Render the per-iter inline probe section for `tier` (forced or
    strong_advantage). Same shape as the previous forced-only formatter.
    """
    title = {
        "forced": "Forced-Tier Probe Sign-Agree (Phase 2)",
        "strong_advantage": "Strong-Advantage Probe Sign-Agree (deep-MCTS labeled)",
    }.get(tier, f"{tier} Probe Sign-Agree")

    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    if not by_iter and not latest:
        lines.append(f"  (not available - no probe_summary.{tier} data in sidecars;")
        lines.append("   either probes file absent or inline eval disabled)")
        lines.append("")
        return lines

    n = (latest or {}).get("n")
    sc = (latest or {}).get("sign_correct")
    sc_pct = (latest or {}).get("sign_correct_pct")
    mv = (latest or {}).get("median_abs_v")
    r5_pct = (latest or {}).get("rolling5_sign_correct_pct")
    r5_mv = (latest or {}).get("rolling5_median_abs_v")
    lines.append("  Latest iter:")
    if n and n > 0:
        sc_pct_s = f"{sc_pct:.1%}" if sc_pct is not None else "n/a"
        mv_s = f"{mv:.3f}" if mv is not None else "n/a"
        lines.append(f"    n={n}, sign_correct={sc}/{n} ({sc_pct_s}), median |v|={mv_s}")
        if r5_pct is not None:
            r5_mv_s = f"{r5_mv:.3f}" if r5_mv is not None else "n/a"
            lines.append(f"    rolling(5 prior): sign={r5_pct:.1%}, median |v|={r5_mv_s}")
    else:
        lines.append(f"    n=0 (no probes matched active_size at this iter)")

    if by_iter:
        trend = by_iter[-10:]
        lines.append("")
        lines.append(f"  Trend (last {len(trend)} iters):")
        lines.append(f"    {'iter':>5} {'n':>4} {'sc':>4} {'sc%':>8} {'|v|':>8} {'delta_sc%':>10} {'rolling5_sc%':>13}")
        for row in trend:
            def _p(x, pct=False):
                if x is None: return "n/a"
                return f"{x:.1%}" if pct else f"{x:.3f}"
            def _d(x):
                if x is None: return "n/a"
                return f"{x*100:+.1f}pp"
            lines.append(
                f"    {row['iteration']:>5} "
                f"{row.get('n') or 0:>4} "
                f"{row.get('sign_correct') or 0:>4} "
                f"{_p(row.get('sign_correct_pct'), pct=True):>8} "
                f"{_p(row.get('median_abs_v')):>8} "
                f"{_d(row.get('delta_sign_correct_pct')):>10} "
                f"{_p(row.get('rolling5_sign_correct_pct'), pct=True):>13}"
            )
        lines.append(f"  ... full per-iter table: {tier}_probe_by_iter.csv")
    lines.append("")
    return lines


# Keep the old name as a forced-only shim for any external callers.
def format_forced_probe_report(by_iter: list, latest: dict) -> List[str]:
    return format_tier_probe_report("forced", by_iter, latest)
```

Find the call site (around line 2179):

```python
        lines.extend(format_forced_probe_report(
            sc_agg.get("forced_probe_by_iter", []),
            sc_agg.get("forced_probe_latest", {}),
        ))
```

Replace with:

```python
        for tier in TIER_NAMES:
            lines.extend(format_tier_probe_report(
                tier,
                sc_agg.get(f"{tier}_probe_by_iter", []),
                sc_agg.get(f"{tier}_probe_latest", {}),
            ))
```

- [ ] **Step 6: Smoke-run the analyzer**

Run the analyzer on whatever Replays directory is convenient (e.g. the latest):

```bash
.venv/bin/python scripts/twixt_replay_analyzer.py --out /tmp/analyzer_smoke
```

Expected: no crash; `summary.json` contains both `forced_probe` and `strong_advantage_probe` keys (the latter with empty `by_iter` and `latest` if no sidecars carry the field yet).

- [ ] **Step 7: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "refactor(analyzer): tier-parameterize probe aggregation

One TIER_NAMES list, one aggregation loop, one CSV-write loop, one
report-format loop. forced behavior unchanged. strong_advantage rows
flow through the same code path; previously this would have required
copy-pasting four blocks. format_forced_probe_report kept as a shim.
"
```

### Task 4.2 — Analyzer aggregation tests with synthetic sidecar fixtures

**Files:**
- Create: `tests/test_strong_advantage_analyzer_aggregation.py`

- [ ] **Step 1: Write the test file**

```python
"""Verify the analyzer correctly aggregates probe_summary.strong_advantage
from per-iter sidecars and emits the right summary.json/CSV/report rows.

Includes backward- and forward-compat assertions for the dual-emit
deprecation window.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_sidecar(dir_path: Path, iteration: int, payload: dict) -> Path:
    sc_path = dir_path / f"iter_{iteration:04d}.json"
    sc_path.write_text(json.dumps(payload))
    return sc_path


def _make_minimal_sidecar(iteration: int, **probe_blocks) -> dict:
    """Build the minimum sidecar shape the analyzer expects, with optional
    probe_summary / forced_probe_summary blocks.
    """
    return {
        "iteration": iteration,
        # ... minimal required fields. The analyzer is tolerant of missing
        # fields when use_sidecar is set; this fixture mirrors the shape
        # of a real sidecar enough to exercise the probe-aggregation path.
        **probe_blocks,
    }


def test_aggregation_collects_strong_advantage_rows(tmp_path):
    """Three sidecars with probe_summary.strong_advantage produce three
    by_iter rows in iter order; latest_iter populates strong_advantage_probe_latest.
    """
    from scripts.twixt_replay_analyzer import _aggregate_sidecars  # adjust if name differs

    sidecars_dir = tmp_path / "sidecars"
    sidecars_dir.mkdir()
    for it, sc in enumerate([60, 65, 70]):
        _write_sidecar(sidecars_dir, sc, _make_minimal_sidecar(
            iteration=sc,
            probe_summary={
                "forced": None,
                "strong_advantage": {
                    "n": 28, "sign_correct": 19 + it,
                    "sign_correct_pct": 0.679 + 0.02 * it,
                    "median_abs_v": 0.41,
                    "delta_sign_correct_pct": 0.02,
                    "delta_median_abs_v": 0.01,
                    "rolling5_sign_correct_pct": 0.65,
                    "rolling5_median_abs_v": 0.40,
                    "n_skipped_size": 0,
                },
            },
        ))
    agg = _aggregate_sidecars(sidecars_dir)
    rows = agg["strong_advantage_probe_by_iter"]
    assert [r["iteration"] for r in rows] == [60, 65, 70]
    latest = agg["strong_advantage_probe_latest"]
    assert latest["sign_correct"] == 21


def test_aggregation_backward_compat_legacy_field(tmp_path):
    """A sidecar with only forced_probe_summary (no probe_summary) still
    aggregates into forced_probe_by_iter via the fallback in
    _read_tier_summary().
    """
    from scripts.twixt_replay_analyzer import _aggregate_sidecars

    sidecars_dir = tmp_path / "sidecars"
    sidecars_dir.mkdir()
    _write_sidecar(sidecars_dir, 50, _make_minimal_sidecar(
        iteration=50,
        forced_probe_summary={
            "n": 30, "sign_correct": 28, "sign_correct_pct": 0.933,
            "median_abs_v": 0.55, "delta_sign_correct_pct": None,
            "delta_median_abs_v": None, "rolling5_sign_correct_pct": None,
            "rolling5_median_abs_v": None, "n_skipped_size": 0,
        },
    ))
    agg = _aggregate_sidecars(sidecars_dir)
    assert len(agg["forced_probe_by_iter"]) == 1
    assert agg["forced_probe_by_iter"][0]["sign_correct"] == 28
    # No strong_advantage in this sidecar.
    assert agg["strong_advantage_probe_by_iter"] == []


def test_aggregation_forward_compat_strong_only(tmp_path):
    """A sidecar with probe_summary.strong_advantage but no forced_*
    aggregates strong_advantage; forced stays empty.
    """
    from scripts.twixt_replay_analyzer import _aggregate_sidecars

    sidecars_dir = tmp_path / "sidecars"
    sidecars_dir.mkdir()
    _write_sidecar(sidecars_dir, 70, _make_minimal_sidecar(
        iteration=70,
        probe_summary={
            "forced": None,
            "strong_advantage": {
                "n": 25, "sign_correct": 20, "sign_correct_pct": 0.80,
                "median_abs_v": 0.50, "delta_sign_correct_pct": None,
                "delta_median_abs_v": None, "rolling5_sign_correct_pct": None,
                "rolling5_median_abs_v": None, "n_skipped_size": 0,
            },
        },
    ))
    agg = _aggregate_sidecars(sidecars_dir)
    assert agg["forced_probe_by_iter"] == []
    assert agg["strong_advantage_probe_by_iter"][0]["sign_correct"] == 20


def test_summary_json_has_both_tier_blocks(tmp_path):
    """summary.json emission produces both forced_probe and
    strong_advantage_probe blocks (parameterized over TIER_NAMES).
    """
    from scripts.twixt_replay_analyzer import _build_summary_json

    sc_agg = {
        "forced_probe_by_iter": [{"iteration": 50, "n": 30}],
        "forced_probe_latest": {"n": 30},
        "strong_advantage_probe_by_iter": [{"iteration": 50, "n": 25}],
        "strong_advantage_probe_latest": {"n": 25},
    }
    summary = _build_summary_json(sc_agg, use_sidecar=True)
    assert "forced_probe" in summary
    assert "strong_advantage_probe" in summary
    assert summary["forced_probe"]["latest"]["n"] == 30
    assert summary["strong_advantage_probe"]["latest"]["n"] == 25


def test_csv_emission_writes_strong_advantage_file(tmp_path):
    """When sc_agg has strong_advantage rows, the analyzer writes
    strong_advantage_probe_by_iter.csv.
    """
    from scripts.twixt_replay_analyzer import _write_tier_csvs

    sc_agg = {
        "strong_advantage_probe_by_iter": [
            {"iteration": 50, "n": 25, "sign_correct": 20,
             "sign_correct_pct": 0.80, "median_abs_v": 0.50,
             "delta_sign_correct_pct": None, "delta_median_abs_v": None,
             "rolling5_sign_correct_pct": None, "rolling5_median_abs_v": None,
             "n_skipped_size": 0},
        ],
    }
    _write_tier_csvs(sc_agg, out_dir=str(tmp_path), suffix="")
    csv_path = tmp_path / "strong_advantage_probe_by_iter.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "iteration,n,n_skipped_size,sign_correct" in content
```

Note: this test references analyzer-internal helpers (`_aggregate_sidecars`, `_build_summary_json`, `_write_tier_csvs`) that will likely need to be extracted from inline code into named functions during Task 4.1's edits. If the analyzer's structure makes these awkward to extract, adapt the tests to the actual shape — the important thing is that the seven assertions (3 aggregation paths + summary block + CSV emission + 2 compat) all pass.

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/pytest tests/test_strong_advantage_analyzer_aggregation.py -v`
Expected: 5 PASS.

If any test fails because the analyzer's internal helper isn't exposed, refactor `scripts/twixt_replay_analyzer.py` to extract the relevant inline code into named module-level functions (e.g. `_aggregate_sidecars`, `_build_summary_json`, `_write_tier_csvs`) and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_strong_advantage_analyzer_aggregation.py scripts/twixt_replay_analyzer.py
git commit -m "test(analyzer): aggregation coverage for strong_advantage tier

Five tests covering: per-iter aggregation, latest-iter snapshot,
backward-compat (legacy forced_probe_summary only), forward-compat
(probe_summary.strong_advantage only), summary.json shape, and CSV
emission. Helpers extracted from inline analyzer code for testability.
"
```

### Task 4.3 — Update README

**Files:**
- Modify: `tests/probes/README.md`

- [ ] **Step 1: Add the strong_advantage section**

After the existing "Bootstrap vs. formal gate suite" section in `tests/probes/README.md`, add:

```markdown
## Strong-advantage tier (`strong_advantage_probes.json`)

A second bootstrap-quality probe tier that complements the forced tier by
covering positions with a strong but not-yet-forced chain advantage. The
file `tests/probes/strong_advantage_probes.json` is produced by the same
generator (`scripts/build_probe_suite.py --tier strong_advantage`) and
follows the same schema with two added per-probe blocks:

- `phase1_features` — structural-dominance heuristics computed at
  candidate-mining time (CC size, axis span, goal-touch, span margin,
  centroid).
- `phase2_label` — deep-MCTS label results (mean root value, per-run
  values, value stability, top-1 visit share, label checkpoint, sim/repeat
  budget, RNG seed).

The committed `meta` block carries `tier: "strong_advantage"`,
`review_mode: "light_review"`, `reviewer`, `reviewed_at_utc`, and the
SHA-256 of the label checkpoint. This is **bootstrap-quality with light
operator review**, NOT the spec §7 review-curated formal gate suite.

### Generating / regenerating

```bash
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage \
    --input scripts/GPU/logs/games \
    --source-iter-range MIN MAX \
    --label-checkpoint checkpoints/<path>/<file>.safetensors \
    --label-mcts-sims 10000 \
    --label-mcts-repeats 3 \
    --magnitude-threshold 0.45 \
    --top1-share-floor 0.15 \
    --stability-cap 0.15 \
    --max-probes 30 \
    --out tests/probes/strong_advantage_probes.json
```

Writes `tests/probes/strong_advantage_probes.draft.json` plus an audit
file `tests/probes/candidates_strong_advantage.json` (gitignored). After
eyeball review, promote with:

```bash
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage \
    --promote \
    --reviewer "$(git config user.name)" \
    --out tests/probes/strong_advantage_probes.json
```

`--promote` refuses to overwrite an existing committed file unless
`--force` is passed.

### Telemetry path

Per-iter trainer sidecars carry both:

- `forced_probe_summary: {...}` — legacy field, kept for one release cycle
  for backward compatibility with downstream readers.
- `probe_summary: { forced: {...}, strong_advantage: {...} }` — the
  forward path. Analyzer prefers this when present and falls through to
  the legacy field for the `forced` tier.

The replay analyzer surfaces both tiers in `summary.json`
(`forced_probe`, `strong_advantage_probe` blocks), `report.txt` (parallel
sections), and per-tier `<tier>_probe_by_iter.csv` files.

### Forced parity test

`tests/test_probe_suite_forced_parity.py` is the safety gate that protects
the existing forced suite from drift when the generator is refactored. It
regenerates the forced tier from the same args recorded in the committed
suite's `meta.selection_rules` and asserts byte-equality with the
committed file. **Assumes** the source replay JSONs in
`scripts/GPU/logs/games/` for the recorded `source_iter_range` remain
on disk and unchanged. If those move/edit/delete, regenerate the
committed suite as a deliberate commit.
```

- [ ] **Step 2: Commit**

```bash
git add tests/probes/README.md
git commit -m "docs(probes): document strong_advantage tier and telemetry path

Adds a section covering the new probe file, generator/promote workflow,
trainer + analyzer plumbing, and the forced-tier parity test's
assumed-stable-replays constraint.
"
```

---

## Self-review

After writing, scan the plan checking against the spec.

**Spec coverage (every section/requirement → task):**

- File layout (spec §"File layout") → Tasks 1.2, 1.3, 2.1-2.5b, 2.7-2.9, 3.1, 4.1, 4.3.
- Telemetry data flow (spec §"Telemetry data flow end-to-end") → Tasks 3.1, 4.1.
- Phase 1 candidate mining → Tasks 2.1, 2.2.
- Phase 2 deep-MCTS labeling → Tasks 2.3, 2.4.
- Phase 3 light review (`--promote`) → Tasks 2.5b (impl), 2.6 (tests), 2.8 (operator run).
- Categories (central / edge / mid-band drop) → Task 2.2 (test + impl).
- Determinism (RNG seed as resolved integer; SHA-256-based not Python hash()) → Tasks 2.3, 2.5a.
- Schema (probe entry + meta) → Task 2.5a (impl), Task 2.9 (test).
- Edge cases (missing checkpoint, zero-admitted, draft/promote overwrite) → Tasks 2.5a, 2.5b (impl), Task 2.6 (tests).
- Edge case: checkpoint architecture mismatch → documented in Task 2.5a as an explicit limitation (hidden/blocks not auto-detected; weight-load failure surfaces operator error).
- Tests: forced parity → Task 1.1; schema/IDs/admission/promotion → Tasks 2.1-2.6, 2.9; analyzer aggregation → Task 4.2; live smoke → Task 2.7. ✓
- Migration steps 1–4 → Sections 1–4. Step 5 (legacy removal) explicitly deferred per scope. ✓

**Type/name consistency:**
- `compute_phase1_features`, `extract_strong_advantage_candidates`, `label_candidate_with_mcts`, `apply_admission_filter`, `_set_default_labeler_network`, `is_forced_within_k` — all referenced consistently across tasks. ✓
- `_probe_id_for`, `_stm_at_ply`, `main_with_args`, `_run_promote` — all introduced in Task 2.5a or 2.5b with consistent signatures used by Task 2.6 tests and the Task 2.8 operator run. ✓
- `build_probe_summary_block` (Task 3.1) signature stable across test and impl. ✓
- `TIER_NAMES`, `_read_tier_summary`, `format_tier_probe_report` (Task 4.1) referenced consistently in aggregation/CSV/report passes. ✓

**Placeholders:**
- No "TBD"/"TODO"/"implement later".
- One acknowledged uncertainty: Task 2.3 notes the actual MCTS interface in `mcts.py` may differ slightly from the example labeler code, with a grep command to verify and a rationale for why tests pass either way.
- Task 4.2 notes that some analyzer internal helpers may need to be extracted for testability — this is genuine refactor work, not deferred.
- Task 2.1 implements `is_forced_within_k` as a 1-ply lookahead (conservative lower bound for k>1) — documented inline as a known limitation that's safe for the admission-filter use, with an interface that admits a future tightening to true negamax search.

**Scope:**
- Plan covers steps 1–4 only. Step 5 (`forced_probe_summary` removal) explicitly deferred per user instruction.
- Each task is one component or one tightly-coupled change set; commits are at task boundaries.
- Task 2.5 split into 2.5a (generation path: validate + load + Phase 1 + Phase 2 + draft + audit) and 2.5b (promotion + serialization helpers) so each task's surface area stays reviewable.
