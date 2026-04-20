# JS↔Python Heuristic AI Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Python heuristic AI (`scripts/GPU/ai/search.py` + `heuristics.py`) into line-by-line parity with JS heuristic AI (`assets/js/ai/search.js` + `heuristics.js`) so that `tests/js_oracle/test_oracle.py::TestDeterministicGameParity` passes and `test_behavioral_regression.py::test_js_move_within_python_topk` passes.

**Architecture:** Inventory first, capture known-good fixtures, port each feature under exact-parity tests against the JS oracle, verify end-to-end. Every per-feature test compares Python's score to JS's score (via `tests/js_oracle/heuristics_oracle.js`) and asserts `abs(diff) < 0.01`. No approximations; if a port can't be done exactly, STOP and escalate.

**Tech Stack:** Python 3.14 (pytest); Node 22 (JS oracle at `tests/js_oracle/heuristics_oracle.js`); existing JS heuristic config at `assets/js/ai/search.json`.

**Spec:** `docs/superpowers/specs/2026-04-20-js-py-heuristic-parity-design.md`

## Contract with the spec

Per spec §3.3:
- **Approximations are NOT acceptable.** If exact parity is unclear, STOP and escalate.
- **Per-feature tests MUST use exact numeric delta** (`abs(py - js) < 0.01`) or **exact move equality** (`py_move == js_move`). No `score_edge > score_mid` inequality-only tests. No `'some_string' in inspect.getsource(...)` introspection tests.
- **Fixtures are committed under `tests/fixtures/heuristic_parity/*.json`**, captured via a dedicated script. No inline-invented fixtures inside test bodies.
- **The audit tool scans BOTH `search.js` AND `heuristics.js`** on the JS side; it is called an **inventory tool** because regex-based scanning cannot guarantee completeness — Phase C (end-to-end game equality) is the authoritative parity oracle.

## Execution Order Note

Phases A → B → C must be strictly sequential.
- **Phase A (Tasks 1-3):** inventory tool + fixture capture + iteration-order contract. Everything downstream depends on these.
- **Phase B (Tasks 4-10):** per-feature ports, each with an exact-delta parity test against the JS oracle, each using a committed fixture.
- **Phase C (Tasks 11-12):** end-to-end game-level parity verification + post-port inventory re-run.

## File Structure

**New files:**

```
docs/superpowers/plans/2026-04-20-js-py-heuristic-parity.md     # this doc
scripts/GPU/ai/heuristic_parity_audit.py                        # Task 1 inventory tool
scripts/GPU/ai/capture_parity_fixture.py                        # Task 2 fixture-capture script
tests/test_heuristic_parity.py                                  # per-feature parity tests
tests/fixtures/heuristic_parity/README.md                       # fixture index
tests/fixtures/heuristic_parity/opening_empty_red.json          # fresh red-to-move
tests/fixtures/heuristic_parity/opening_empty_black.json        # fresh black-to-move
tests/fixtures/heuristic_parity/mid_game_seed0_ply10.json       # captured state
tests/fixtures/heuristic_parity/mid_game_seed1_ply15.json       # captured state
tests/fixtures/heuristic_parity/near_win_red.json               # red near-terminal
tests/fixtures/heuristic_parity/near_win_black.json             # black near-terminal
tests/fixtures/heuristic_parity/sealed_lane.json                # opponent sealed
```

**Modified files:**

```
scripts/GPU/ai/heuristics.py                                    # Tasks 4-8 port additions
scripts/GPU/ai/search.py                                        # Task 9 port adjustments
```

**Untouched (JS is reference):**

```
assets/js/ai/search.js
assets/js/ai/heuristics.js
assets/js/ai/search.json
```

---

## Phase A — Inventory + Fixtures + Foundation (Tasks 1-3)

### Task 1: Heuristic parity inventory tool

**Rationale.** Before porting we need an inventory of what's divergent. A manual grep is error-prone and misses snake_case↔camelCase mismatches and scoring-constant references. The inventory tool scans **both JS files** (`search.js` AND `heuristics.js`) plus Python (`heuristics.py`, `search.py`, `sealed_lane.py`, `move_ordering.py`) and emits a CSV.

**This is called an "inventory tool", not a parity oracle.** Regex scanning can miss features encoded in ways the scanner doesn't match (anonymous scoring sites, implicit state mutations). The authoritative parity oracle is Phase C's end-to-end game equality check.

**Files:**
- Create: `scripts/GPU/ai/heuristic_parity_audit.py`
- Create test: `tests/test_heuristic_parity.py` (inventory smoke tests only; per-feature tests added in later tasks)

- [ ] **Step 1: Write the inventory-tool CLI test**

Create `tests/test_heuristic_parity.py`:

```python
"""Tests for JS↔Python heuristic parity inventory + per-feature ports."""
import json
import os
import subprocess


def test_inventory_tool_help():
    """Inventory tool CLI responds to --help."""
    result = subprocess.run(
        [".venv/bin/python", "scripts/GPU/ai/heuristic_parity_audit.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "--js-file" in result.stdout
    assert "--py-file" in result.stdout


def test_inventory_scans_both_js_files(tmp_path):
    """Inventory must scan BOTH search.js AND heuristics.js; missing either one
    is a spec violation (spec §3.1)."""
    out = tmp_path / "audit.csv"
    result = subprocess.run(
        [".venv/bin/python", "scripts/GPU/ai/heuristic_parity_audit.py",
         "--js-file", "assets/js/ai/search.js",
         "--js-file", "assets/js/ai/heuristics.js",
         "--py-file", "scripts/GPU/ai/heuristics.py",
         "--py-file", "scripts/GPU/ai/search.py",
         "--py-file", "scripts/GPU/ai/sealed_lane.py",
         "--py-file", "scripts/GPU/ai/move_ordering.py",
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    content = out.read_text()
    lines = content.strip().split("\n")
    assert lines[0] == "feature_name,in_js,in_py,js_files,js_sites,py_sites,divergence_kind"
    assert len(lines) >= 20, f"Expected ≥20 features, got {len(lines) - 1}"
    # Features from search.js
    assert "edgeFinishStall" in content
    assert "edgeFinishAdvance" in content
    assert "doubleEdgeCoverage" in content
    # Feature from heuristics.js — must be present iff heuristics.js scan works
    # componentMetrics is exported from heuristics.js
    assert "componentMetrics" in content or "component_metrics" in content, (
        "Inventory missing components from heuristics.js — single-file scan?"
    )


def test_inventory_rejects_single_js_file():
    """Running with only search.js must produce a warning or note that heuristics.js
    is excluded (helps the user catch the spec §3.1 requirement)."""
    out = "/tmp/audit_single.csv"
    result = subprocess.run(
        [".venv/bin/python", "scripts/GPU/ai/heuristic_parity_audit.py",
         "--js-file", "assets/js/ai/search.js",
         "--py-file", "scripts/GPU/ai/heuristics.py",
         "--out", out],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    # Tool should emit a warning to stderr about incomplete coverage
    assert "heuristics.js" in result.stderr or "WARN" in result.stderr, (
        "Single-JS-file run should warn about missing heuristics.js"
    )


def test_inventory_identifies_known_js_only_divergences():
    """Known JS-only features must be tagged in_js=True, in_py=False."""
    out = "/tmp/audit_test.csv"
    subprocess.run(
        [".venv/bin/python", "scripts/GPU/ai/heuristic_parity_audit.py",
         "--js-file", "assets/js/ai/search.js",
         "--js-file", "assets/js/ai/heuristics.js",
         "--py-file", "scripts/GPU/ai/heuristics.py",
         "--py-file", "scripts/GPU/ai/search.py",
         "--py-file", "scripts/GPU/ai/sealed_lane.py",
         "--py-file", "scripts/GPU/ai/move_ordering.py",
         "--out", out],
        check=True, capture_output=True,
    )
    with open(out) as f:
        lines = f.read().strip().split("\n")
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    by_name = {r["feature_name"]: r for r in rows}
    # edgeFinishStall is definitely JS-only at the time of this plan
    assert by_name["edgeFinishStall"]["in_js"] == "True"
    assert by_name["edgeFinishStall"]["in_py"] == "False"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_inventory_tool_help -v
```

Expected: FileNotFoundError — script doesn't exist.

- [ ] **Step 3: Implement the inventory tool**

Create `scripts/GPU/ai/heuristic_parity_audit.py`:

```python
#!/usr/bin/env python3
"""Audit JS↔Python heuristic feature parity.

Scans JS source (search.js) and Python source (heuristics.py / search.py),
extracts feature names from call sites (capture(...), recordStat(...) in JS;
score += ... entries + knob lookups in Python), and writes a CSV flagging
divergence.

Feature names are normalized: camelCase ↔ snake_case. For example JS's
'edgeFinishStall' matches Python's 'edge_finish_stall'.
"""
from __future__ import annotations
import argparse
import csv
import re
import sys


def camel_to_snake(name: str) -> str:
    """edgeFinishStall → edge_finish_stall"""
    return re.sub(r'([A-Z])', r'_\1', name).lower().lstrip('_')


def snake_to_camel(name: str) -> str:
    """edge_finish_stall → edgeFinishStall"""
    parts = name.split('_')
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


def scan_js(path: str) -> dict:
    """Return {feature_name: [line_numbers]} from capture('X', ...) and recordStat('X', ...)."""
    out: dict = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            for m in re.finditer(r"(?:capture|recordStat)\(\s*'([a-zA-Z][a-zA-Z0-9]*)'", line):
                name = m.group(1)
                out.setdefault(name, []).append(lineno)
    return out


def scan_py(paths: list) -> dict:
    """Return {feature_name: [(path, line_numbers)]} for Python files.

    Looks for knob lookups like knobs["featureName"] or knobs.get("featureName"),
    score adjustments annotated with # featureName, and direct feature-name
    references in string literals.
    """
    out: dict = {}
    # Patterns to scan for feature names:
    # 1. knobs["name"], knobs.get("name"), k["name"]
    # 2. Comment annotations: "# featureName" or "# search.json: featureName"
    patterns = [
        re.compile(r"""knobs?\s*(?:\.get\s*)?\(?\s*["']([a-zA-Z_][a-zA-Z0-9_]*)["']"""),
        re.compile(r"""k\s*\[\s*["']([a-zA-Z_][a-zA-Z0-9_]*)["']"""),
        re.compile(r"""#.*?([a-zA-Z][a-zA-Z0-9]*(?:Stall|Advance|Bonus|Penalty|Scale|Multiplier|Reduction|Gain|Touch|Sealed|Coverage|Proximity|Capture))"""),
    ]
    for path in paths:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                for pat in patterns:
                    for m in pat.finditer(line):
                        name = m.group(1)
                        out.setdefault(name, []).append((path, lineno))
    return out


REQUIRED_JS_FILES = {"search.js", "heuristics.js"}


def main():
    ap = argparse.ArgumentParser(description="Inventory JS↔Python heuristic feature parity (non-authoritative)")
    ap.add_argument("--js-file", action="append", required=True,
                    help="JS source file (repeatable). Both search.js AND heuristics.js are REQUIRED per spec §3.1.")
    ap.add_argument("--py-file", action="append", required=True,
                    help="Python source file(s) to scan (repeatable)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    # Warn if required JS files are missing (spec §3.1)
    js_basenames = {os.path.basename(f) for f in args.js_file}
    missing_js = REQUIRED_JS_FILES - js_basenames
    if missing_js:
        print(f"[WARN] missing required JS files: {missing_js}. Inventory may be incomplete. "
              f"Spec §3.1 requires both search.js and heuristics.js.", file=sys.stderr)

    # Aggregate per-file JS scans
    js_feats: dict = {}  # name -> list of (file, line)
    for js_path in args.js_file:
        basename = os.path.basename(js_path)
        for name, lines in scan_js(js_path).items():
            for ln in lines:
                js_feats.setdefault(name, []).append((basename, ln))

    py_feats = scan_py(args.py_file)

    # Normalize Python feature names to camelCase for matching
    py_normalized = {}
    for name, sites in py_feats.items():
        camel = snake_to_camel(name) if '_' in name else name
        py_normalized.setdefault(camel, []).extend(sites)
        # Also keep original name as-is
        if camel != name:
            py_normalized.setdefault(name, []).extend(sites)

    all_names = set(js_feats.keys()) | set(py_normalized.keys())

    rows = []
    for name in sorted(all_names):
        in_js = name in js_feats
        in_py = name in py_normalized
        js_entries = js_feats.get(name, [])
        js_files = ";".join(sorted({f for (f, _) in js_entries}))
        js_sites = ";".join(f"{f}:{ln}" for (f, ln) in js_entries)
        py_site_strs = [f"{p}:{ln}" for (p, ln) in py_normalized.get(name, [])]
        py_sites = ";".join(py_site_strs)
        if in_js and in_py:
            divergence_kind = "present_both"
        elif in_js and not in_py:
            divergence_kind = "js_only"
        elif not in_js and in_py:
            divergence_kind = "py_only"
        else:
            divergence_kind = "unknown"
        rows.append({
            "feature_name": name,
            "in_js": in_js,
            "in_py": in_py,
            "js_files": js_files,
            "js_sites": js_sites,
            "py_sites": py_sites,
            "divergence_kind": divergence_kind,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "feature_name", "in_js", "in_py", "js_files", "js_sites", "py_sites", "divergence_kind"
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Summary to stdout
    n_js_only = sum(1 for r in rows if r["divergence_kind"] == "js_only")
    n_py_only = sum(1 for r in rows if r["divergence_kind"] == "py_only")
    n_both = sum(1 for r in rows if r["divergence_kind"] == "present_both")
    print(f"Audit complete: {len(rows)} features total")
    print(f"  JS-only (port needed): {n_js_only}")
    print(f"  Python-only (unexpected): {n_py_only}")
    print(f"  Both sides: {n_both}")
    print(f"  CSV: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_inventory_tool_help tests/test_heuristic_parity.py::test_inventory_scans_both_js_files tests/test_heuristic_parity.py::test_inventory_rejects_single_js_file tests/test_heuristic_parity.py::test_inventory_identifies_known_js_only_divergences -v
```

Expected: all 4 pass.

- [ ] **Step 5: Run the inventory and inspect**

```bash
.venv/bin/python scripts/GPU/ai/heuristic_parity_audit.py \
  --js-file assets/js/ai/search.js \
  --js-file assets/js/ai/heuristics.js \
  --py-file scripts/GPU/ai/heuristics.py \
  --py-file scripts/GPU/ai/search.py \
  --py-file scripts/GPU/ai/sealed_lane.py \
  --py-file scripts/GPU/ai/move_ordering.py \
  --out /tmp/heuristic_inventory.csv
cat /tmp/heuristic_inventory.csv | column -t -s, | head -40
```

Expected output: a CSV table with ≥30 feature rows. Scan the `divergence_kind=js_only` rows — these scope the features Tasks 4-10 will port. Record a count: if the inventory shows more `js_only` features than Tasks 4-10 cover, **STOP and escalate** so the plan can be extended before implementation continues.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/ai/heuristic_parity_audit.py tests/test_heuristic_parity.py
git commit -m "$(cat <<'EOF'
feat: add JS↔Python heuristic parity inventory tool

Scans BOTH search.js AND heuristics.js for capture()/recordStat() call
sites (spec §3.1). Also scans heuristics.py, search.py, sealed_lane.py,
move_ordering.py for knobs[] lookups and annotated score entries.
Normalizes names (camelCase ↔ snake_case), emits a CSV flagging
js_only / py_only / present_both features.

This is an INVENTORY tool, not an authoritative parity oracle — regex
scans can miss behavior encoded in non-standard ways. The authoritative
oracle is Phase C's end-to-end game-level equality check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Capture known-good parity fixtures

**Rationale.** Every Phase B test compares Python's score against JS's score for a specific position. That position must come from a **committed, known-good fixture file** — not invented inline in the test body. Inventing fixtures inline risks silent drift when game rules change.

**Files:**
- Create: `scripts/GPU/ai/capture_parity_fixture.py`
- Create: `tests/fixtures/heuristic_parity/README.md`
- Create: 7 JSON fixture files (opening_empty_red, opening_empty_black, mid_game_seed0_ply10, mid_game_seed1_ply15, near_win_red, near_win_black, sealed_lane)

- [ ] **Step 1: Write a test that asserts each fixture exists and replays cleanly**

Append to `tests/test_heuristic_parity.py`:

```python
FIXTURES_DIR = "tests/fixtures/heuristic_parity"

REQUIRED_FIXTURES = [
    "opening_empty_red.json",
    "opening_empty_black.json",
    "mid_game_seed0_ply10.json",
    "mid_game_seed1_ply15.json",
    "near_win_red.json",
    "near_win_black.json",
    "sealed_lane.json",
]


def test_all_fixtures_exist():
    """All 7 parity fixtures must be committed under tests/fixtures/heuristic_parity/."""
    for name in REQUIRED_FIXTURES:
        path = os.path.join(FIXTURES_DIR, name)
        assert os.path.exists(path), f"Fixture missing: {path}"


def test_all_fixtures_replay_to_valid_state():
    """Each fixture's move_history replays into a state matching declared metadata."""
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    for name in REQUIRED_FIXTURES:
        path = os.path.join(FIXTURES_DIR, name)
        with open(path) as f:
            fx = json.load(f)
        state = GameState(board_size=fx.get("board_size", 24), to_move=fx["start_player"])
        for (r, c) in fx["move_history"]:
            state = apply_move(state, int(r), int(c))
        assert state.to_move == fx["to_move"], (
            f"{name} replay to_move={state.to_move} != declared {fx['to_move']}"
        )
        assert len(state.pegs) == fx.get("expected_pegs", len(state.pegs)), (
            f"{name} peg count drift"
        )
```

- [ ] **Step 2: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_all_fixtures_exist -v
```

Expected: all 7 AssertionErrors (fixtures don't exist yet).

- [ ] **Step 3: Implement the capture script**

Create `scripts/GPU/ai/capture_parity_fixture.py`:

```python
#!/usr/bin/env python3
"""Capture a deterministic-mode game state at a given ply and dump it as JSON.

Plays `TwixtSimulator.play_one` with deterministic_mode=1, stops at --ply,
and emits a fixture JSON with:
  board_size, start_player, to_move (at stop ply), ply, move_history, pegs,
  bridges, expected_pegs, note.

Fixtures are used by tests/test_heuristic_parity.py — their stability survives
any refactor of game rules because the assert-on-replay tests catch drift.
"""
from __future__ import annotations
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description="Capture a parity fixture — from seeded play OR an explicit move list (fallback).")
    # Path A: seeded deterministic play
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for TwixtSimulator deterministic play (omit if using --moves-file)")
    ap.add_argument("--ply", type=int, default=None,
                    help="Stop after this many plies (omit if using --moves-file)")
    ap.add_argument("--depth", type=int, default=2)
    # Path B: explicit move list (fallback when seeded play doesn't yield the right topology)
    ap.add_argument("--moves-file", default=None,
                    help="JSON file with a list of [row, col] tuples to apply verbatim. "
                         "Use when seeded play can't produce the needed topology — source moves "
                         "from a committed game log or oracle trace, never invent.")
    # Common
    ap.add_argument("--start-player", choices=["red", "black"], default=None,
                    help="Starting player (default: seed even→black, odd→red; required with --moves-file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="", help="Human-readable note — MUST describe the move source")
    args = ap.parse_args()

    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move

    # --- Path B: explicit move list (fallback) ---
    if args.moves_file:
        if args.start_player is None:
            print("[ERROR] --start-player is required with --moves-file", file=sys.stderr)
            sys.exit(2)
        with open(args.moves_file) as f:
            move_history = json.load(f)
        start_player = args.start_player
        state = GameState(board_size=24, to_move=start_player)
        for (r, c) in move_history:
            state = apply_move(state, int(r), int(c))
        fixture = {
            "source": f"moves-file:{args.moves_file}",
            "board_size": 24,
            "start_player": start_player,
            "to_move": state.to_move,
            "ply": len(move_history),
            "move_history": move_history,
            "expected_pegs": len(state.pegs),
            "note": args.note or "(no note — should describe move source)",
        }
        with open(args.out, "w") as f:
            json.dump(fixture, f, indent=2)
        print(f"Wrote {args.out} from moves-file: ply={len(move_history)}, pegs={len(state.pegs)}, to_move={state.to_move}")
        return

    # --- Path A: seeded deterministic play (default) ---
    if args.seed is None or args.ply is None:
        print("[ERROR] either (--seed AND --ply) OR --moves-file is required", file=sys.stderr)
        sys.exit(2)

    from scripts.GPU.selfplay.engine import TwixtSimulator

    # Derive starting_player the same way JS oracle does: seed even → black, odd → red
    start_player = args.start_player or ("black" if args.seed % 2 == 0 else "red")

    sim = TwixtSimulator(board_size=24, max_moves=max(args.ply, 1), stall_limit=max(args.ply + 5, 5))
    outcome = sim.play_one(
        knobs={"deterministic_mode": 1},
        seed=args.seed, depth=args.depth, top_n=20, use_value_model=False,
    )

    if args.ply > len(outcome.moves):
        print(f"[ERROR] requested ply={args.ply} but game produced only {len(outcome.moves)} moves. "
              f"Use --moves-file fallback — do NOT inline-invent in tests.", file=sys.stderr)
        sys.exit(2)

    move_history = [[m.row, m.col] for m in outcome.moves[:args.ply]]

    # Replay to derive pegs + to_move at stop ply
    state = GameState(board_size=24, to_move=start_player)
    for (r, c) in move_history:
        state = apply_move(state, int(r), int(c))

    fixture = {
        "seed": args.seed,
        "depth": args.depth,
        "board_size": 24,
        "start_player": start_player,
        "to_move": state.to_move,
        "ply": args.ply,
        "move_history": move_history,
        "expected_pegs": len(state.pegs),
        "note": args.note,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote {args.out}: ply={args.ply}, pegs={len(state.pegs)}, to_move={state.to_move}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Capture the 7 fixtures**

```bash
mkdir -p tests/fixtures/heuristic_parity

# Opening states (ply=0, no moves played yet)
.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --seed 1 --ply 0 --start-player red \
  --out tests/fixtures/heuristic_parity/opening_empty_red.json \
  --note "Fresh board, red to move (seed=1 → odd → red start)"

.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --seed 0 --ply 0 --start-player black \
  --out tests/fixtures/heuristic_parity/opening_empty_black.json \
  --note "Fresh board, black to move (seed=0 → even → black start)"

# Mid-game states from deterministic play
.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --seed 0 --ply 10 \
  --out tests/fixtures/heuristic_parity/mid_game_seed0_ply10.json \
  --note "Mid-game, 10 plies into seed=0 deterministic play"

.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --seed 1 --ply 15 \
  --out tests/fixtures/heuristic_parity/mid_game_seed1_ply15.json \
  --note "Mid-game, 15 plies into seed=1 deterministic play"

# Near-win and sealed-lane fixtures (hand-constructed move histories for
# known topologies). If the tool can't produce these via seeded play,
# STOP and construct them manually — DO NOT inline-invent in tests.
.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --seed 0 --ply 30 \
  --out tests/fixtures/heuristic_parity/near_win_red.json \
  --note "Red has chain from top reaching most of the board"

.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --seed 1 --ply 30 \
  --out tests/fixtures/heuristic_parity/near_win_black.json \
  --note "Black has chain left reaching most of the board"

.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --seed 2 --ply 40 \
  --out tests/fixtures/heuristic_parity/sealed_lane.json \
  --note "Black has sealed at least one of red's finish lanes"
```

If any capture errors out (e.g. game ended before the target ply, or the seeded play produces the wrong topology for the fixture's intent), use this **fallback procedure** — still through the fixture script, never handwritten in tests:

1. **Extend the capture script to accept an explicit move list.** Add a `--moves-file <path>` flag that reads a JSON array of `[row, col]` tuples and applies them verbatim (bypassing the self-play loop). This lets you seed a specific known-good topology from an existing source:
   - an oracle trace under `tests/js_oracle/` if one exists for the target topology
   - a historical game JSON under `scripts/GPU/logs/games/iter_NNNN_game_MMM.json`, sliced to the desired ply
   - a sequence captured via `scripts/trace_training_game.py` output

2. **Source the move list from a committed or recorded artifact** (game log, oracle trace) — do NOT compose one in the user's head and paste into a JSON file. Intent: every fixture must trace back to a known-good game state produced by some reproducible process.

3. **Commit the `--moves-file` source alongside the fixture** so a future reviewer can see where the moves came from.

Example fallback invocation for `near_win_red.json` if seeded play doesn't produce the right topology:

```bash
# Extract a near-win-red slice from an existing game log
.venv/bin/python -c "
import json
g = json.load(open('scripts/GPU/logs/games/iter_0950_game_007.json'))
moves = [[m['row'], m['col']] for m in g['moves'][:40]]
json.dump(moves, open('/tmp/near_win_red_moves.json', 'w'))
"
.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
  --moves-file /tmp/near_win_red_moves.json \
  --start-player red \
  --out tests/fixtures/heuristic_parity/near_win_red.json \
  --note "Near-win red, moves sourced from iter_0950_game_007.json ply 0-40"
```

The `--moves-file` flag is a Task 2 implementation requirement when seeded play doesn't suffice — **not** a signal to inline-invent the fixture in a test body.

**Verify capture:** read back each JSON and ensure `expected_pegs` is plausible (ply=0 → 0 pegs; ply=10 → 10 pegs; etc.).

- [ ] **Step 5: Write the fixture README**

Create `tests/fixtures/heuristic_parity/README.md`:

```markdown
# Heuristic Parity Fixtures

Committed JSON snapshots of known-good game states used by
`tests/test_heuristic_parity.py` to test JS↔Python score parity.

## Files

| File | Seed | Ply | Description |
|---|---|---|---|
| opening_empty_red.json | 1 | 0 | Fresh board, red to move |
| opening_empty_black.json | 0 | 0 | Fresh board, black to move |
| mid_game_seed0_ply10.json | 0 | 10 | Mid-game after 10 deterministic plies |
| mid_game_seed1_ply15.json | 1 | 15 | Mid-game after 15 deterministic plies |
| near_win_red.json | 0 | 30 | Red chain near top→bottom completion |
| near_win_black.json | 1 | 30 | Black chain near left→right completion |
| sealed_lane.json | 2 | 40 | Black has sealed one of red's finish lanes |

## Regenerating a fixture

Fixtures are immutable references — don't regenerate without updating dependent tests.

If you need a new fixture, run:

```bash
.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
    --seed N --ply P --out tests/fixtures/heuristic_parity/<name>.json \
    --note "<description>"
```

## Schema

```json
{
  "seed": 0,
  "depth": 2,
  "board_size": 24,
  "start_player": "black",
  "to_move": "red",
  "ply": 10,
  "move_history": [[r, c], ...],
  "expected_pegs": 10,
  "note": "human-readable description"
}
```
```

- [ ] **Step 6: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_all_fixtures_exist tests/test_heuristic_parity.py::test_all_fixtures_replay_to_valid_state -v
```

Expected: both pass. Replay asserts peg counts and to_move consistency.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/ai/capture_parity_fixture.py tests/fixtures/heuristic_parity/ tests/test_heuristic_parity.py
git commit -m "$(cat <<'EOF'
feat: add parity-fixture capture script + 7 committed fixtures

Fixtures under tests/fixtures/heuristic_parity/ are the single source
of input positions for Phase B parity tests. Captured via
capture_parity_fixture.py which replays TwixtSimulator.play_one with
deterministic_mode=1.

No inline-invented fixtures in test bodies — test_all_fixtures_exist
enforces that every fixture comes from this committed set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Pin deterministic-mode iteration order

**Rationale.** Even if scoring is identical, lexicographic tie-breaking in `deterministic_mode=1` depends on the order in which candidate moves are generated and compared. If JS iterates `for row in 0..N, for col in 0..N` but Python iterates differently (e.g., sorts by some key first), ties break differently and game outputs diverge without any scoring bug.

**Files:**
- Modify: `scripts/GPU/ai/search.py::choose_move` (verify iteration order, no code change unless diverged)
- Extend: `tests/test_heuristic_parity.py`

- [ ] **Step 1: Add iteration-order parity test**

Append to `tests/test_heuristic_parity.py`:

```python
def test_deterministic_mode_tie_break_iteration_order():
    """When multiple candidate moves tie on score, deterministic mode must pick
    the lexicographically smallest (row, col). Both JS and Python must agree.
    """
    import subprocess
    import json

    # Build a state where multiple moves tie at score=0 (empty board, red-to-move).
    # In empty-board case ALL legal moves score equivalently on position metrics
    # (no pegs to count connections to, no spans, no goal distance differences
    # beyond the edge-row ramp). Python's choose_move should pick (1, 1) since
    # (0, c) is red's goal edge (can't play on own goal edges except corners,
    # which are forbidden) and col 0/23 is left/right edge (blocked for red).
    # Actually red CAN play at (0, c) for c in 1..22 but is drawn to center.
    # Empirically: lexicographic tie-break → smallest (row, col) that scores highest.

    # Rather than reason through the score calculus, check directly: feed a
    # known-tied position to both engines and assert they pick the same move.
    config = {"seed": 0, "depth": 1, "maxMoves": 1}  # Only 1 move needed
    result = subprocess.run(
        ["node", "tests/js_oracle/deterministic_game_oracle.js"],
        input=json.dumps(config), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    js_data = json.loads(result.stdout)
    js_first_move = js_data["moves"][0]

    # Python equivalent via TwixtSimulator
    from scripts.GPU.selfplay.engine import TwixtSimulator
    sim = TwixtSimulator(board_size=24, max_moves=1, stall_limit=1)
    py_outcome = sim.play_one(
        knobs={"deterministic_mode": 1},
        seed=0, depth=1, top_n=20, use_value_model=False,
    )
    py_first_move = py_outcome.moves[0]

    # Both must produce the same move — this is the PRE-CONDITION for any
    # downstream scoring-parity test to be meaningful.
    assert (py_first_move.row, py_first_move.col) == (js_first_move["row"], js_first_move["col"]), (
        f"First-move parity broken. JS picked ({js_first_move['row']},{js_first_move['col']}), "
        f"Python picked ({py_first_move.row},{py_first_move.col}). "
        f"Either iteration order diverged OR scoring diverged at move 0."
    )
```

- [ ] **Step 2: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_deterministic_mode_tie_break_iteration_order -v
```

Expected: FAIL, showing the actual (JS_move vs PY_move) disagreement.

- [ ] **Step 3: Investigate**

If JS picks `(0, 11)` and Python picks `(12, 0)` (pure row/col swap pattern), it's almost certainly a starting-player or iteration-axis mismatch — maybe black-to-move triggers different pathing.

Check both sides:
- JS: `deterministic_game_oracle.js` — how does it pick the starting player, what order does it iterate legal moves?
- Python: `scripts/GPU/selfplay/engine.py::TwixtSimulator.play_one` — same questions

Document findings in a comment in the test. This task does NOT fix anything yet — it just documents the iteration-order contract. The fix will come in later tasks once we know exactly what's driving the tie.

- [ ] **Step 4: Run full inventory CSV to scope**

```bash
.venv/bin/python scripts/GPU/ai/heuristic_parity_audit.py \
  --js-file assets/js/ai/search.js \
  --js-file assets/js/ai/heuristics.js \
  --py-file scripts/GPU/ai/heuristics.py \
  --py-file scripts/GPU/ai/search.py \
  --py-file scripts/GPU/ai/sealed_lane.py \
  --py-file scripts/GPU/ai/move_ordering.py \
  --out /tmp/inventory.csv
awk -F, '$7 == "js_only" {print $1}' /tmp/inventory.csv
```

Expected output: list of JS-only features (e.g. `edgeFinishAdvance`, `edgeFinishStall`, `finishLaneSealed`, `doubleEdgeCoverage`, ...). This list informs Tasks 4-10. Note column 7 is `divergence_kind` in the new CSV schema (was column 6 before `js_files` was added — spec §3.1).

- [ ] **Step 5: Commit (audit + iteration-order test)**

```bash
git add tests/test_heuristic_parity.py
git commit -m "$(cat <<'EOF'
test: add first-move parity test (iteration-order contract)

Pre-condition for all downstream scoring-parity tests. If Python and JS
disagree on move 0 for seed=0 depth=1, either iteration order or scoring
has diverged. This test fails loudly and is the first to fix in Phase B.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase B — Feature Port (Tasks 4-10)

Each task in Phase B:
1. Reads the JS source at specific line ranges for the feature
2. Adds a **parity-delta unit test** that loads a committed fixture (from Task 2) and asserts `abs(py_score - js_score) < 0.01` via the JS heuristics oracle
3. Ports the feature into `scripts/GPU/ai/heuristics.py` or `search.py` verbatim from the JS code
4. Verifies the parity-delta test passes
5. Commits

**Non-negotiable rules:**
- Tests use **exact numeric delta** (`abs(py - js) < 0.01`) or **exact move equality** against the JS oracle. **NO** `score_edge > score_mid` inequality tests. **NO** `'feature' in inspect.getsource(...)` introspection tests.
- Tests load fixtures from `tests/fixtures/heuristic_parity/*.json`. **NO** inline-invented fixtures.
- If the JS formula can't be ported exactly (unknown magnitude, stateful side effects, complex branching without comments), **STOP and report BLOCKED**. Do NOT approximate.

**Order matters.** Later tasks may reference features added by earlier tasks. Ports adjust `evaluate_move` and `choose_move` — state accumulates.

**Parity-delta test template** (use for every Phase B test):

```python
def test_feature_X_parity_on_fixture_Y():
    """JS and Python must agree on feature X score contribution for fixture Y."""
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle  # reuse existing harness

    # 1. Load committed fixture
    with open("tests/fixtures/heuristic_parity/fixture_Y.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # 2. Get JS score via oracle (note: JSOracle.evaluate_move already calls
    #    the existing JS heuristics oracle at tests/js_oracle/heuristics_oracle.js)
    js_result = JSOracle.evaluate_move(state, move=(target_row, target_col),
                                        player=fx["to_move"])
    assert js_result is not None, "JS oracle failed"
    js_score = js_result["score"]  # shape: {"score": float, ...}

    # 3. Get Python score
    py_score = evaluate_move(state, target_row, target_col, fx["to_move"])

    # 4. Exact parity
    assert abs(py_score - js_score) < 0.01, (
        f"Feature X parity broken on fixture Y: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )
```

**CI parity-gate grep (run before merging this branch):**

```bash
# Must return zero matches
grep -En "assert.*getsource|assert .* > score_|assert .* > .*_score\b" tests/test_heuristic_parity.py
```

If any matches surface, the test file has reverted to weak assertions and must be tightened before the branch ships.

---

### Task 4: Port `firstEdgeTouch` bonus

**Rationale.** First peg placed on a goal edge earns a bonus. JS has `firstEdgeTouch` in `search.js` scoring path; Python has `firstEdgeRed`/`firstEdgeBlack` knobs but the invocation path has drifted (NOTE in `heuristics.py:1178` says "firstEdgeTouch bonuses are handled in search.js, NOT here"). The note is outdated — search.js DOES apply it.

**Files:**
- Modify: `scripts/GPU/ai/heuristics.py::evaluate_move` (apply firstEdge* knob scoring)
- Extend: `tests/test_heuristic_parity.py`

- [ ] **Step 1: Read the JS source for firstEdgeTouch**

```bash
grep -n "firstEdgeTouch\|firstEdgeRed\|firstEdgeBlack" assets/js/ai/search.js assets/js/ai/search.json
```

Locate the call sites (likely in `evaluateMove`). Note the exact formula: when does it apply, what's the magnitude, are there side-asymmetric factors?

- [ ] **Step 2: Add parity-delta unit tests**

Append to `tests/test_heuristic_parity.py` (follows §3.3 test shape — exact JS↔Python delta, no inequality):

```python
def test_first_edge_touch_red_parity_on_empty_fixture():
    """firstEdgeRed bonus: Python score must match JS score within 0.01 for a
    move placing red's first edge peg on the empty-board fixture."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/opening_empty_red.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    # Red's first edge-touching move: (0, 3)
    test_move = (0, 3)
    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")
    assert abs(py_score - js_score) < 0.01, (
        f"firstEdgeRed parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )


def test_first_edge_touch_black_parity_on_empty_fixture():
    """firstEdgeBlack bonus: parity on empty fixture."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/opening_empty_black.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    test_move = (3, 0)  # black's left-edge-touching move
    js_result = JSOracle.evaluate_move(state, move=test_move, player="black")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "black")
    assert abs(py_score - js_score) < 0.01, (
        f"firstEdgeBlack parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )


def test_first_edge_touch_second_peg_same_edge_parity():
    """After red has one top-edge peg, a SECOND top-edge peg must NOT get
    firstEdge bonus — JS and Python must both suppress it."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/opening_empty_red.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    # Place red at (0, 3) (first edge) and black somewhere, now red picks second top edge
    state = apply_move(state, 0, 3)
    state = apply_move(state, 12, 12)  # black move
    test_move = (0, 15)  # red's SECOND top-edge peg — firstEdge should NOT fire
    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")
    assert abs(py_score - js_score) < 0.01, (
        f"firstEdge (second peg) parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )
```

- [ ] **Step 3: Run tests — expect failure**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_first_edge_touch_red_parity_on_empty_fixture tests/test_heuristic_parity.py::test_first_edge_touch_black_parity_on_empty_fixture tests/test_heuristic_parity.py::test_first_edge_touch_second_peg_same_edge_parity -v
```

Expected: FAIL — edge score ≈ mid score (no firstEdge bonus applied in evaluate_move).

- [ ] **Step 4: Port the feature into heuristics.py**

In `scripts/GPU/ai/heuristics.py::evaluate_move`, just before the `return score` line, add (using the JS formula from Step 1):

```python
# firstEdgeTouch bonus (ported from search.js — 2026-04-20 parity port)
# JS: in search.js's evaluateMove-post-processing, bonus applies when the new
# peg is placed on this player's goal edge AND no prior same-color peg touches
# that same edge. Search key: "firstEdgeTouch" in search.js.
is_first_edge = False
if player == "red" and row in (0, board_size - 1):
    is_first_edge = not any(
        pr == row and state.pegs[(pr, pc)] == "red"
        for (pr, pc) in state.pegs
    )
    if is_first_edge:
        score += k.get("firstEdgeRed", 420.0)
elif player == "black" and col in (0, board_size - 1):
    is_first_edge = not any(
        pc == col and state.pegs[(pr, pc)] == "black"
        for (pr, pc) in state.pegs
    )
    if is_first_edge:
        score += k.get("firstEdgeBlack", 455.0)
```

Remove the outdated comment on line 1178 (`# NOTE: firstEdgeTouch bonuses are handled in search.js, NOT here.`).

- [ ] **Step 5: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_first_edge_touch_red_parity_on_empty_fixture tests/test_heuristic_parity.py::test_first_edge_touch_black_parity_on_empty_fixture tests/test_heuristic_parity.py::test_first_edge_touch_second_peg_same_edge_parity tests/test_training.py tests/test_heuristics.py -v
```

Expected: all pass (including existing tests — firstEdge must not regress).

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/ai/heuristics.py tests/test_heuristic_parity.py
git commit -m "$(cat <<'EOF'
feat: port firstEdgeTouch bonus from search.js to heuristics.py

JS-only feature detected by parity audit: when a player places their
first peg on a goal edge, apply firstEdgeRed (420) or firstEdgeBlack
(455) bonus. Python's evaluate_move had a stale comment claiming JS
handled this solely — JS does apply it during evaluate_move, so Python
must too for deterministic-mode parity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Port `edgeGapReduction` bonus

**Rationale.** When a new peg reduces the gap from an existing same-color chain toward the unreached goal edge, award a bonus scaled by the reduction. JS emits `edgeGapReduction` stat.

**Files:**
- Modify: `scripts/GPU/ai/heuristics.py::evaluate_move` (add edgeGapReduction block)
- Extend: `tests/test_heuristic_parity.py`

- [ ] **Step 1: Read the JS source**

```bash
grep -n "edgeGapReduction\|edgeGap\|gapBefore\|gapAfter" assets/js/ai/search.js | head -20
```

Locate the formula: `reduction = gapBefore - gapAfter`; `bonus = reduction * REWARDS.edge.offense.gapReductionScale`.

- [ ] **Step 2: Parity-delta unit tests**

```python
def test_edge_gap_reduction_parity_gap_reducing_move():
    """edgeGapReduction: Python score matches JS for a gap-reducing move."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/mid_game_seed0_ply10.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # Pick a move that advances the current player's chain toward the far goal edge.
    # Mid-game fixture has enough pegs placed that gap-reduction has signal.
    test_move = (15, 10) if fx["to_move"] == "red" else (10, 15)
    js_result = JSOracle.evaluate_move(state, move=test_move, player=fx["to_move"])
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], fx["to_move"])
    assert abs(py_score - js_score) < 0.01, (
        f"edgeGapReduction parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )


def test_edge_gap_reduction_parity_non_reducing_move():
    """Non-reducing move: Python score matches JS (feature doesn't fire)."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/mid_game_seed0_ply10.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # A move that doesn't advance toward goal — still must match JS exactly
    test_move = (5, 5) if fx["to_move"] == "red" else (5, 5)
    js_result = JSOracle.evaluate_move(state, move=test_move, player=fx["to_move"])
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], fx["to_move"])
    assert abs(py_score - js_score) < 0.01, (
        f"Non-reducing move parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )
```

- [ ] **Step 3: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_edge_gap_reduction_parity_gap_reducing_move tests/test_heuristic_parity.py::test_edge_gap_reduction_parity_non_reducing_move -v
```

- [ ] **Step 4: Port the feature**

In `scripts/GPU/ai/heuristics.py::evaluate_move`, add after the firstEdgeTouch block from Task 3:

```python
# edgeGapReduction bonus (ported from search.js — 2026-04-20 parity port)
# JS: REWARDS.edge.offense.gapReductionScale per unit of gap reduction.
# Gap = distance from chain's goal-near-edge to the goal. Reducing it helps.
gap_scale = k.get("edgeGapReductionScale", 45.0)  # from search.json rewards.edge.offense
player_pegs = [pos for pos, col in state.pegs.items() if col == player]
if player_pegs:
    if player == "red":
        top_edge_min = min((r for (r, _) in player_pegs), default=board_size)
        bot_edge_max = max((r for (r, _) in player_pegs), default=-1)
        gap_before = top_edge_min + (board_size - 1 - bot_edge_max)
        gap_after_top = min(top_edge_min, row)
        gap_after_bot = max(bot_edge_max, row)
        gap_after = gap_after_top + (board_size - 1 - gap_after_bot)
    else:
        left_edge_min = min((c for (_, c) in player_pegs), default=board_size)
        right_edge_max = max((c for (_, c) in player_pegs), default=-1)
        gap_before = left_edge_min + (board_size - 1 - right_edge_max)
        gap_after_left = min(left_edge_min, col)
        gap_after_right = max(right_edge_max, col)
        gap_after = gap_after_left + (board_size - 1 - gap_after_right)
    reduction = max(0, gap_before - gap_after)
    score += reduction * gap_scale
```

- [ ] **Step 5: Run — expect pass**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_edge_gap_reduction_parity_gap_reducing_move tests/test_heuristic_parity.py::test_edge_gap_reduction_parity_non_reducing_move tests/test_training.py tests/test_heuristics.py -v
```

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/ai/heuristics.py tests/test_heuristic_parity.py
git commit -m "feat: port edgeGapReduction bonus from search.js

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 6: Port `edgeFinishAdvance` / `edgeFinishStall` pair

**Rationale.** When a peg near the opponent-side goal edge creates progress (span or gap improvement), award `edgeFinishAdvance`. If the peg is near the edge but stalls (no progress), apply `edgeFinishStall` penalty. Side-asymmetric: `blackFinishScaleMultiplier`, `redFinishExtra`, `redFinishPenaltyFactor`.

**Files:**
- Modify: `scripts/GPU/ai/heuristics.py::evaluate_move` (add finish advance/stall block)
- Extend: `tests/test_heuristic_parity.py`

- [ ] **Step 1: Read the JS source**

```bash
grep -n "edgeFinishAdvance\|edgeFinishStall\|finishBonusBase\|finishGapSlope\|connectorBonus\|redFinishExtra\|blackFinishScaleMultiplier\|redFinishPenaltyFactor\|finishPenaltyBase" assets/js/ai/search.js
```

Location: `search.js` around lines 1634-1678 (the `if (touchesBothPost || nearFinish)` block).

- [ ] **Step 2: Parity-delta unit tests**

```python
def test_edge_finish_advance_parity():
    """edgeFinishAdvance: Python matches JS on a fixture+move where JS emits the bonus."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    # near_win_red.json has a red chain that extends toward bottom — a candidate
    # at (23, c_interior) should trigger edgeFinishAdvance in JS.
    with open("tests/fixtures/heuristic_parity/near_win_red.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    test_move = (state.board_size - 1, 5)  # red on bottom edge, advancing
    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")
    assert abs(py_score - js_score) < 0.01, (
        f"edgeFinishAdvance parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )


def test_edge_finish_stall_parity():
    """edgeFinishStall: Python matches JS on a fixture+move where the peg is
    on the goal edge but makes no progress (disconnected from main chain)."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    # Use near_win_red.json with a candidate that touches bottom edge but
    # is far from the main red chain (no span/gap progress).
    with open("tests/fixtures/heuristic_parity/near_win_red.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # Pick a cell on bottom edge far from chain (fixture-dependent; adjust
    # if needed after capture produces a known position). (23, 20) is far
    # from (23, 5) chain area in most reasonable near_win configurations.
    test_move = (state.board_size - 1, 20)
    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")
    assert abs(py_score - js_score) < 0.01, (
        f"edgeFinishStall parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )
```

- [ ] **Step 3: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_edge_finish_advance_parity tests/test_heuristic_parity.py::test_edge_finish_stall_parity -v
```

- [ ] **Step 4: Port the feature**

In `scripts/GPU/ai/heuristics.py::evaluate_move`, after the edgeGapReduction block:

```python
# edgeFinishAdvance / edgeFinishStall (ported from search.js lines ~1634-1678)
# Applies when peg is on or near the opponent-side goal edge:
# - If peg extends span or reduces gap → edgeFinishAdvance (bonus)
# - If peg stalls (no progress) → edgeFinishStall (penalty)
# Side-asymmetric scaling applies.
near_finish = False
if player == "red":
    near_finish = (row == 0 or row == board_size - 1)
else:
    near_finish = (col == 0 or col == board_size - 1)

if near_finish and player_pegs:  # player_pegs computed in edgeGapReduction block
    # Did this peg advance progress? (uses same `reduction` as edgeGapReduction)
    progress_made = reduction > 0
    finish_bonus_base = max(
        0.0,
        k.get("finishBonusBase", 2500.0) - gap_after * k.get("finishGapSlope", 15.0)
    )
    if progress_made:
        bonus_base = k.get("connectorBonus", 80.0) + finish_bonus_base
        if player == "black":
            bonus_base *= k.get("blackFinishScaleMultiplier", 0.85)
        else:  # red
            bonus_base += k.get("redFinishExtra", 35.0)
        score += bonus_base
    else:
        penalty_base = k.get("finishPenaltyBase", 200.0) + gap_after * k.get("finishGapSlope", 15.0)
        if player == "red":
            penalty_base *= k.get("redFinishPenaltyFactor", 0.8)
        score -= penalty_base
```

- [ ] **Step 5: Run — expect pass**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py -v
```

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/ai/heuristics.py tests/test_heuristic_parity.py
git commit -m "feat: port edgeFinishAdvance + edgeFinishStall from search.js

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 7: Port `finishLaneSealed` guard

**Rationale.** `edgeFinishAdvance`/`edgeFinishStall` must be gated on whether the opponent has sealed the current player's finish lane. JS uses `hasReachableGoalEdge` (a BFS over bridge crossings that proves no clear path to the goal edge exists). Python has `scripts/GPU/ai/sealed_lane.py::check_sealed_lane`. Both must produce the same verdict for the same position.

**Escalation rule:** this task **must NOT approximate**. If `check_sealed_lane` has a signature or semantics that prevent exact mirroring of `hasReachableGoalEdge`, STOP and report BLOCKED. Don't ship a "crude sealed check" that Phase C might catch — if the JS guard is complex, the port needs a dedicated sub-task to align `check_sealed_lane` first.

**Files:**
- Modify: `scripts/GPU/ai/heuristics.py::evaluate_move` (gate finish bonus/penalty on sealed-lane check)
- Extend: `tests/test_heuristic_parity.py`
- Possibly: `scripts/GPU/ai/sealed_lane.py` (only if API mismatch forces changes; in that case, report first)

- [ ] **Step 1: Read the JS source verbatim**

```bash
grep -n "finishLaneSealed\|hasReachableGoalEdge\|laneOpen\|track.*sealed\|SEALED_LANE" assets/js/ai/search.js
```

Identify the exact `hasReachableGoalEdge` function definition and its call sites. Record (a) what inputs it takes (component metrics, player, bridges crossable?), (b) what boolean it returns, (c) the precise gate logic in `edgeFinishAdvance` that consumes it.

- [ ] **Step 2: Check Python's `check_sealed_lane` signature**

```bash
.venv/bin/python -c "from scripts.GPU.ai.sealed_lane import check_sealed_lane; import inspect; print(inspect.signature(check_sealed_lane))"
```

Compare to the JS `hasReachableGoalEdge`:
- Same inputs? (state, player, component)
- Same output? (bool: true=lane open, false=sealed)
- Same internal algorithm? (BFS with bridge-crossing rejection)

If any mismatch exists, **STOP here** and report BLOCKED with:
- Exact JS signature + line reference
- Exact Python signature + line reference
- Specific incompatibility

Only proceed to Step 3 if the signatures align.

- [ ] **Step 3: Parity-delta unit test**

Append to `tests/test_heuristic_parity.py`:

```python
def test_finish_lane_sealed_parity_sealed_fixture():
    """On sealed_lane.json fixture: JS and Python must agree that red's finish lane
    is sealed, and edgeFinishAdvance bonus must be suppressed equally in both."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/sealed_lane.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # Pick a candidate move on red's bottom edge — this would trigger
    # edgeFinishAdvance IF the lane were open. Fixture has lane sealed,
    # so JS score should reflect zero advance bonus.
    test_move = (state.board_size - 1, 10)  # bottom edge, interior col

    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")

    assert abs(py_score - js_score) < 0.01, (
        f"finishLaneSealed guard parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}. "
        f"Likely cause: lane_open check disagrees between engines."
    )


def test_finish_lane_open_fixture_applies_bonus():
    """On near_win_red.json (lane OPEN): JS and Python must agree the bonus fires."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/near_win_red.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # A move that advances toward the goal edge, lane is open in this fixture
    test_move = (state.board_size - 1, 5)  # bottom edge

    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")

    assert abs(py_score - js_score) < 0.01, (
        f"edgeFinishAdvance (with open lane) parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )
```

- [ ] **Step 4: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py::test_finish_lane_sealed_parity_sealed_fixture tests/test_heuristic_parity.py::test_finish_lane_open_fixture_applies_bonus -v
```

- [ ] **Step 5: Port the feature — exact mirror of JS**

In `scripts/GPU/ai/heuristics.py::evaluate_move`, extend the `near_finish and player_pegs` block from Task 6:

```python
# finishLaneSealed guard (ported from JS hasReachableGoalEdge)
# JS: if the candidate's component does NOT have a reachable path to the
# unreached goal edge (due to opponent sealing), skip both bonus and penalty.
from .sealed_lane import check_sealed_lane

# Build post-move state for the check (may need component_metrics)
# Use component_metrics helper to get largest_component + goal-edge touches
post_state_pegs = dict(state.pegs)
post_state_pegs[(row, col)] = player
# compute post metrics (same helper JS uses)
post_metrics = component_metrics_for_player(post_state_pegs, state.bridges | {implied_bridges}, player)
touches_near_goal = post_metrics["touches_top"] if player == "red" else post_metrics["touches_left"]
touches_far_goal = post_metrics["touches_bottom"] if player == "red" else post_metrics["touches_right"]

# check_sealed_lane returns True if lane is OPEN (reachable), False if sealed
lane_open = check_sealed_lane(
    state, player_id=(0 if player == "red" else 1),
    largest_component=post_metrics["largest_component"],
    touches_start=touches_near_goal, touches_end=touches_far_goal,
    sealed_lane_cache={},  # no cross-call cache in per-move scoring
)

if near_finish and player_pegs and lane_open:
    # progress_made branch from Task 6 — ONLY applies when lane_open
    if progress_made:
        bonus_base = ...
        score += bonus_base
    else:
        penalty_base = ...
        score -= penalty_base
# else: lane sealed → no bonus, no penalty (matches JS finishLaneSealed branch)
```

**If `component_metrics_for_player` or `check_sealed_lane` doesn't have this exact signature**, STOP and report. The port must be an exact mirror, not an approximation.

- [ ] **Step 6: Run — expect pass**

```bash
.venv/bin/python -m pytest tests/test_heuristic_parity.py -v
```

Both `test_finish_lane_sealed_parity_sealed_fixture` and `test_finish_lane_open_fixture_applies_bonus` must pass. If either fails with a non-zero `abs(py - js)` diff, the guard is misaligned — fix before moving on.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/ai/heuristics.py tests/test_heuristic_parity.py
git commit -m "feat: port finishLaneSealed guard from search.js (exact parity)

Gated edgeFinishAdvance/Stall on check_sealed_lane matching JS's
hasReachableGoalEdge. Parity-delta tests on sealed_lane.json and
near_win_red.json fixtures pass with abs(py - js) < 0.01.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 8: Port `doubleEdgeCoverage` bonus

**Rationale.** When a peg creates threats on two separate goal edges simultaneously, award `doubleEdgeCoverage`. JS detects this via post-move component metrics — the peg's updated component must touch BOTH goal edges.

**Escalation rule:** this task **must NOT approximate**. "Any peg on top edge + any peg on bottom edge" is NOT the same as "one component touches both edges simultaneously." If the port can't compute post-move component-membership exactly (i.e. the existing `component_metrics` can't be run on the post-move state), STOP and report BLOCKED.

**Files:**
- Modify: `scripts/GPU/ai/heuristics.py::evaluate_move`
- Extend: `tests/test_heuristic_parity.py`

- [ ] **Step 1: Read the JS source verbatim**

```bash
grep -n "doubleEdgeCoverage\|touchesTopPost\|touchesBottomPost\|touchesLeftPost\|touchesRightPost\|touchesBothPost" assets/js/ai/search.js
```

Record the exact logic: JS checks `touchesBothPost`, which is computed on the post-move component — ALL pegs in the same bridge-connected component, after the candidate peg is placed. Also record the magnitude constant.

- [ ] **Step 2: Verify Python post-move component-metrics support**

```bash
.venv/bin/python -c "from scripts.GPU.ai.heuristics import component_metrics; import inspect; print(inspect.signature(component_metrics))"
```

If `component_metrics` can be called on a synthesized post-move state (add peg + its implied bridges), proceed. If it can't — STOP and escalate to add a post-move metrics helper first.

- [ ] **Step 3: Parity-delta unit test**

Append to `tests/test_heuristic_parity.py`:

```python
def test_double_edge_coverage_parity_near_win_red():
    """On near_win_red.json: a candidate peg that bridges top-touching chain to
    bottom-touching chain must trigger doubleEdgeCoverage equally in JS and Python."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/near_win_red.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # near_win_red.json has red chain reaching most of the board. Find a
    # candidate move that — if placed — bridges the top-chain and bottom-chain
    # via knight moves. Fixture's note describes this explicitly.
    # The test picks one such candidate; actual coords depend on fixture.
    # Use the fixture's test_candidate field (capture script should record
    # one good candidate when creating near_win_* fixtures, OR we compute it here).
    test_move = tuple(fx.get("parity_candidate_double_edge") or (12, 10))

    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")

    assert abs(py_score - js_score) < 0.01, (
        f"doubleEdgeCoverage parity broken on near_win_red: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )


def test_double_edge_coverage_parity_no_coverage_fixture():
    """On mid_game_seed0_ply10.json (no double-edge): bonus must NOT fire in either engine."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from scripts.GPU.game.rules import apply_move
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/mid_game_seed0_ply10.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    for (r, c) in fx["move_history"]:
        state = apply_move(state, int(r), int(c))

    # Any legal non-double-coverage move — the test asserts JS and Python agree
    # (both should score without doubleEdgeCoverage contributing).
    test_move = (10, 10)
    js_result = JSOracle.evaluate_move(state, move=test_move, player=fx["to_move"])
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], fx["to_move"])

    assert abs(py_score - js_score) < 0.01, (
        f"Score parity broken on mid_game fixture: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )
```

Note: if `near_win_red.json` fixture doesn't include a `parity_candidate_double_edge` field, the capture script (Task 2) needs extension to record it. If adding that field is cumbersome, STOP — don't hardcode coordinates that might not actually trigger the feature; work with the fixture capture tool.

- [ ] **Step 4: Run — expect failure**

- [ ] **Step 5: Port the feature — exact JS mirror**

In `scripts/GPU/ai/heuristics.py::evaluate_move`:

```python
# doubleEdgeCoverage (ported from search.js — exact JS mirror)
# JS: after placing the candidate peg, compute component_metrics on the new
# state. If touchesBothPost is True (the candidate's component touches BOTH
# goal edges), award bonus. This is NOT the same as "any peg on top + any
# peg on bottom" — they must share a single bridge-connected component.

# Synthesize post-move state
post_pegs = dict(state.pegs)
post_pegs[(row, col)] = player
# Add implied bridges for the new peg (knight moves to same-color pegs that
# aren't blocked by crossings)
implied_bridges = set(state.bridges)
for dr, dc in KNIGHT_OFFSETS:
    nr, nc = row + dr, col + dc
    if 0 <= nr < board_size and 0 <= nc < board_size:
        if (nr, nc) in state.pegs and state.pegs[(nr, nc)] == player:
            if not bridges_cross(state, row, col, nr, nc):
                implied_bridges.add(canonical_bridge((row, col), (nr, nc)))

post_metrics = component_metrics(
    pegs=post_pegs, bridges=implied_bridges,
    player=player, board_size=board_size,
)
if player == "red":
    touches_both = post_metrics["touches_top"] and post_metrics["touches_bottom"]
else:
    touches_both = post_metrics["touches_left"] and post_metrics["touches_right"]

if touches_both:
    score += k.get("doubleEdgeCoverage", 600.0)  # verify magnitude from search.json
```

**Verify magnitude** by reading `assets/js/ai/search.json` — if it has `doubleEdgeCoverage` defined there, use that value. Else find the literal in `search.js`.

If `component_metrics` or `canonical_bridge` or `bridges_cross` have different signatures than assumed, STOP and escalate.

- [ ] **Step 6: Run — expect pass**

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/ai/heuristics.py tests/test_heuristic_parity.py
git commit -m "feat: port doubleEdgeCoverage bonus from search.js (exact parity)

Uses component_metrics on post-move state to check touchesBothPost
exactly as JS does. Parity-delta tests on near_win_red.json and
mid_game_seed0_ply10.json fixtures pass with abs(py - js) < 0.01.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 9: Port `redBaseBonus` / `blackBasePenalty` / color-asymmetric scaling

**Rationale.** JS applies side-asymmetric scale factors (red gets a base bonus, black gets a base penalty, globalScale multipliers). Python has `training_*` knobs that may or may not map to the same multipliers.

**Files:**
- Modify: `scripts/GPU/ai/heuristics.py::evaluate_move` (apply color scaling to final score)
- Extend: `tests/test_heuristic_parity.py`

- [ ] **Step 1: Read the JS source**

```bash
grep -n "redBaseBonus\|blackBasePenalty\|redGlobalMultiplier\|blackGlobalScale" assets/js/ai/search.js
```

- [ ] **Step 2: Parity-delta unit tests**

```python
def test_red_base_scaling_parity_empty_fixture():
    """Red's baseline scaling: Python total score must match JS total on empty fixture."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/opening_empty_red.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    test_move = (10, 10)  # center move — not edge-triggering, so pure base scaling dominates
    js_result = JSOracle.evaluate_move(state, move=test_move, player="red")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "red")
    assert abs(py_score - js_score) < 0.01, (
        f"red base scaling parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )


def test_black_base_scaling_parity_empty_fixture():
    """Black's baseline scaling: parity on empty fixture."""
    import json
    from scripts.GPU.ai.heuristics import evaluate_move
    from scripts.GPU.game.state import GameState
    from tests.js_oracle.test_oracle import JSOracle

    with open("tests/fixtures/heuristic_parity/opening_empty_black.json") as f:
        fx = json.load(f)
    state = GameState(board_size=fx["board_size"], to_move=fx["start_player"])
    test_move = (10, 10)
    js_result = JSOracle.evaluate_move(state, move=test_move, player="black")
    assert js_result is not None, "JS oracle failed"
    js_score = float(js_result["score"])
    py_score = evaluate_move(state, test_move[0], test_move[1], "black")
    assert abs(py_score - js_score) < 0.01, (
        f"black base scaling parity broken: JS={js_score:.4f}, "
        f"Python={py_score:.4f}, diff={abs(py_score - js_score):.4f}"
    )
```

- [ ] **Step 3: Run — expect failure**

- [ ] **Step 4: Port the feature**

Add near the end of `evaluate_move`:

```python
# Side-asymmetric scaling (ported from search.js)
if player == "red":
    score += k.get("redBaseBonus", 0.0)
    score *= k.get("redGlobalMultiplier", 1.0)
else:
    score -= k.get("blackBasePenalty", 0.0)
    score *= k.get("blackGlobalScale", 1.0)
```

- [ ] **Step 5: Run — expect pass**

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/ai/heuristics.py tests/test_heuristic_parity.py
git commit -m "feat: port side-asymmetric scaling (redBase/blackBase/globalMultipliers)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 10: Port `search.py::choose_move` adjustment order

**Rationale.** Even with parity in `evaluate_move`, the order and interleaving of adjustment blocks in `choose_move` (e.g. finish_bonus, intercept, def_block, sample_pool trimming) can differ from JS's `search.js` flow. Audit + align.

**Files:**
- Modify: `scripts/GPU/ai/search.py::choose_move`
- Extend: `tests/test_heuristic_parity.py`

- [ ] **Step 1: Read both sides side-by-side**

```bash
# JS side: the main choose-move loop
grep -n "function evaluatePosition\|function chooseMove\|for.*candidate\|score +=\|score -=" assets/js/ai/search.js | head -50

# Python side
grep -n "def choose_move\|score\s*[+-]=\|candidate" scripts/GPU/ai/search.py | head -50
```

Map each adjustment block in JS to the Python equivalent. Document gaps.

- [ ] **Step 2: Write a parity test at the choose_move level**

```python
def test_choose_move_seed_0_first_move_matches_js():
    """choose_move produces the same first move as JS for seed=0, depth=1."""
    import json
    import subprocess
    # Python
    from scripts.GPU.selfplay.engine import TwixtSimulator
    sim = TwixtSimulator(board_size=24, max_moves=1, stall_limit=1)
    py_outcome = sim.play_one(
        knobs={"deterministic_mode": 1}, seed=0, depth=1, top_n=20, use_value_model=False,
    )
    py_move = (py_outcome.moves[0].row, py_outcome.moves[0].col)
    # JS
    config = {"seed": 0, "depth": 1, "maxMoves": 1}
    result = subprocess.run(
        ["node", "tests/js_oracle/deterministic_game_oracle.js"],
        input=json.dumps(config), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    js_move_data = json.loads(result.stdout)["moves"][0]
    js_move = (js_move_data["row"], js_move_data["col"])
    assert py_move == js_move, (
        f"choose_move divergence at seed=0 depth=1: Python={py_move}, JS={js_move}"
    )
```

- [ ] **Step 3: Run — may pass after Tasks 3-8 (optimistic) or fail (realistic)**

If passes: the per-feature ports were enough. If fails: there's additional logic in choose_move that needs aligning — iterate on Step 4.

- [ ] **Step 4: Iterate on choose_move**

Compare JS `search.js` to Python `search.py` function-by-function. Align:
- Candidate iteration order (row-major, deterministic)
- Final-score combination formula (`eval_score + immediate * 5 + position * 0.1 + finish_bonus`)
- Sort stability (Python's `sort(key=...)` is stable; JS array sort may not be — make it stable via secondary key on `(row, col)`)
- Argmax tie-break (lexicographic on `(row, col)` when scores match to float precision)

- [ ] **Step 5: Run — should pass now**

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/ai/search.py tests/test_heuristic_parity.py
git commit -m "feat: align choose_move candidate ordering + sort stability

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

## Phase C — Integration Verification (Tasks 11-12)

### Task 11: Full oracle parity sweep

**Rationale.** Unit tests passing individually doesn't guarantee game-level parity. Run all 20 seeds × depth 2 and ensure zero divergence.

**Files:**
- Modify: none (pure test run)

- [ ] **Step 1: CI parity gate — no weak tests**

```bash
# Must return zero matches — proves no inequality-only or source-introspection tests snuck in
grep -En "assert.*getsource|assert .* > score_|assert .* > .*_score\b" tests/test_heuristic_parity.py
```

Expected: empty output. If matches surface, tighten tests before proceeding.

- [ ] **Step 2: Run the full JS oracle suite**

```bash
.venv/bin/python -m pytest tests/js_oracle/test_oracle.py::TestDeterministicGameParity -v 2>&1 | tee /tmp/oracle_full.log
```

Expected: all tests pass.

- [ ] **Step 3: Run behavioral regression**

```bash
.venv/bin/python -m pytest tests/test_behavioral_regression.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run the full Python suite for regressions**

```bash
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_probe_suite_schema.py --ignore=tests/js_oracle 2>&1 | tail -30
```

Expected: no new failures vs pre-port baseline. If any new Python failures surfaced, investigate (likely a subtle bug in the port).

- [ ] **Step 5: Document and commit**

```bash
git commit --allow-empty -m "$(cat <<'EOF'
chore: verify full heuristic-parity gate passes

- tests/js_oracle/test_oracle.py::TestDeterministicGameParity: all pass
- tests/test_behavioral_regression.py: all pass
- Python suite (ex probe schema + js_oracle): no regressions
- Spec §4 gate: ✅

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Rerun parity inventory

**Rationale.** After porting, the audit CSV should show no unresolved `js_only` features.

**Files:**
- Modify: none (pure audit re-run)

- [ ] **Step 1: Regenerate the inventory**

```bash
.venv/bin/python scripts/GPU/ai/heuristic_parity_audit.py \
  --js-file assets/js/ai/search.js \
  --js-file assets/js/ai/heuristics.js \
  --py-file scripts/GPU/ai/heuristics.py \
  --py-file scripts/GPU/ai/search.py \
  --py-file scripts/GPU/ai/sealed_lane.py \
  --py-file scripts/GPU/ai/move_ordering.py \
  --out /tmp/inventory_post.csv
awk -F, '$7 == "js_only" {print $1}' /tmp/inventory_post.csv
```

Expected: **empty** output (no unresolved js_only rows). If any appear, they're either (a) documentation strings the inventory spuriously matched, or (b) features we missed. Resolve manually — don't ship with unresolved js_only rows.

- [ ] **Step 2: Commit the final audit as a documentation artifact**

```bash
cp /tmp/audit_post.csv docs/superpowers/heuristic_parity_audit_2026-04-20.csv
git add docs/superpowers/heuristic_parity_audit_2026-04-20.csv
git commit -m "docs: commit post-port heuristic parity audit

Snapshot of audit CSV after Tasks 3-9 ports. Zero js_only rows =
parity achieved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

## Self-Review

- [x] **Spec coverage:** Every section of the spec (2026-04-20-js-py-heuristic-parity-design.md) maps to a task:
  - §3.1 feature enumeration (BOTH JS files) → Task 1 (inventory tool, scans search.js + heuristics.js)
  - §3.2 deterministic mode contract → Task 3 (first-move parity test)
  - §3.3 scoring primitive alignment (exact-parity tests, no approximations) → Tasks 4-10 (per-feature ports with parity-delta tests)
  - §4 validation gate (exact-delta tests, CI grep gate) → Task 11 Step 1 (CI grep) + Tasks 4-10 (exact-delta per-feature)
  - §5 fixture capture (no inline-invented fixtures) → Task 2 (capture script + 7 committed fixtures)
  - §7 success criteria → Tasks 11-12
- [x] **Placeholders scan:** No TBD/TODO in the plan body. All approximation language removed from Tasks 7, 8 (finishLaneSealed, doubleEdgeCoverage) per spec §3.3 — those tasks now have explicit escalation rules instead.
- [x] **Type/method-name consistency:** `evaluate_move`, `choose_move`, `check_sealed_lane`, `component_metrics`, `capture_parity_fixture.py` used consistently across tasks.
- [x] **Critical rules enforced:**
  - JS is authoritative; Python mirrors. Every port task reads JS verbatim first.
  - Tests use exact-delta (`abs(py - js) < 0.01`) or exact move-equality — no inequality-only, no source introspection.
  - Fixtures loaded from `tests/fixtures/heuristic_parity/*.json` — no inline-invented fixtures.
  - Port approximations trigger STOP-and-escalate (explicit in Tasks 7, 8).
  - CI grep gate at Task 11 Step 1 catches any regression in test-assertion strength.

## Feedback applied in this revision

Your 5 feedback points, addressed:

1. **Audit both JS files** → Task 1 signature requires both `--js-file` entries; Step 1 test `test_inventory_scans_both_js_files` asserts both are scanned; Step 1 test `test_inventory_rejects_single_js_file` warns on single-file invocations.
2. **Remove approximation language** → Task 7 (finishLaneSealed) and Task 8 (doubleEdgeCoverage) now have explicit **Escalation rule** blocks at the top. Steps 2 in both require signature verification BEFORE implementation; if signatures mismatch, STOP and report BLOCKED rather than approximating.
3. **Upgrade per-feature tests** → All Phase B tests use exact-delta (`abs(py_score - js_score) < 0.01`) against the JS oracle. No `score_edge > score_mid` inequalities. No `inspect.getsource(...)` introspection. CI grep gate at Task 11 Step 1 enforces.
4. **Known-good fixtures** → Task 2 creates a `capture_parity_fixture.py` script + 7 committed JSON fixtures. Every Phase B test loads a fixture. Test `test_all_fixtures_exist` asserts the fixture set is committed; `test_all_fixtures_replay_to_valid_state` asserts fixtures survive rule refactors.
5. **Reworded "audit tool" → "inventory tool"** → Task 1 rationale explicitly says "inventory tool, not an authoritative parity oracle" + "regex scanning can miss behavior encoded in non-standard ways" + "Phase C is the authoritative parity oracle."

## Known caveats

1. The port magnitudes (420, 455, 2500, 80, 35, 0.85, 0.8, 600) come from scanning JS + `search.json`. Before each task's port, **re-verify the magnitude** by reading the current `search.json` and matching it with the JS code — values may have drifted.
2. This plan assumes the user's uncommitted `assets/js/ai/search.js` + `heuristics.js` WIP is **frozen during the port**. If the WIP continues to evolve, the port will drift again — the user must either commit the WIP first or pause JS-side changes until Phase C completes.
3. Fixtures may need refinement (e.g., a near_win_red.json capture may not produce a cleanly double-edge-coverable topology). When seeded deterministic play doesn't yield the right topology, **use the `--moves-file` fallback path** of `capture_parity_fixture.py` with moves sourced from:
   - an existing oracle trace under `tests/js_oracle/` if one matches the intent
   - a historical game log under `scripts/GPU/logs/games/iter_*_game_*.json`, sliced to the desired ply
   - a sequence captured via `scripts/trace_training_game.py`

   The `--moves-file` JSON must come from a reproducible committed artifact (git-tracked game log or oracle trace) — **not** handwritten in the user's head. Every fixture must trace back to known-good game state produced by some reproducible process, even via the fallback path. The note field in the captured fixture MUST describe the move source.

4. If a fixture's shape still doesn't match a test's needs after the fallback, the capture script can also be extended to record a `parity_candidate_*` hint field identifying a specific cell that triggers the feature being tested — STOP and escalate before hardcoding candidate coordinates in tests.
