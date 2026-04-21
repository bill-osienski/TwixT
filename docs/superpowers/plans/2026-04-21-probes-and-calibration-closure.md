# Probes & Calibration Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close both empty-output gaps in one round — (a) trainer-side `forced_probe_summary` (currently `None` because no probes file is committed) and (b) analyzer-side `value_calibration` (currently a `{"status": "not_implemented"}` stub) — by adding shared extraction/scoring primitives, committing a bootstrap rule-selected probe suite, wiring the analyzer to produce real snapshot data, and gating both paths behind tests.

**Architecture:** Two signals with distinct semantics. Trainer per-iter: stable committed probe suite, existing code path lights up when the file arrives. Analyzer end-of-chunk: replay-derived probes + stratified calibration sampling against an auto-discovered checkpoint, producing `replay_probe_scoring` and `value_calibration` top-level keys in `summary_*.json`. Three new primitives shared between callers: `extract_forced_probes_from_games`, `load_network_for_scoring`, `score_samples_against_checkpoint`.

**Tech Stack:** Python 3.14, MLX (Apple Silicon NN framework), pytest (with existing markers in `pytest.ini`), numpy. All changes additive — no migrations, no breaking API changes, user workflow unchanged.

**Spec:** `docs/superpowers/specs/2026-04-21-probes-and-calibration-closure-design.md`

---

## File Structure

### Created files
| Path | Responsibility |
|---|---|
| `scripts/build_bootstrap_probe_suite.py` | One-shot-but-reusable generator that produces `tests/probes/twixt_probes.json` from historical game replays. Natural-wins-only filter, K=2, size-24, dedup + balance rules per spec §5. |
| `tests/probes/twixt_probes.json` | Committed bootstrap rule-selected probe suite (20–30 probes). Generated output of the script above. Checked into git. |
| `tests/test_value_calibration_sampling.py` | Unit tests for new `score_samples_against_checkpoint`. |
| `tests/test_bootstrap_probe_suite.py` | Unit tests for `build_bootstrap_probe_suite.py` — determinism, schema, no wall-clock fields. |
| `tests/test_analyzer_replay_probe_scoring_end_to_end.py` | End-to-end test exercising the analyzer's full probe + calibration pipeline. |
| `tests/test_trainer_forced_probe_live.py` | `@pytest.mark.integration` test. Runs 2 real training iters with minimal config. Opt-in locally, required in CI. |

### Modified files
| Path | Responsibility |
|---|---|
| `scripts/GPU/alphazero/probe_eval.py` | Add `extract_forced_probes_from_games()` and `load_network_for_scoring()` (public wrapper over existing `_load_network`). |
| `scripts/GPU/alphazero/value_calibration.py` | Add `score_samples_against_checkpoint()`. Existing helpers (`classify_position`, `aggregate_calibration`, `compute_calibration_bins`) unchanged. |
| `scripts/twixt_replay_analyzer.py` | Add new CLI flags, checkpoint auto-discovery, replay probe scoring integration, calibration integration (replaces current stub), CSV writers, report formatters. |
| `tests/test_inline_probe_observability.py` | Add 4 new test cases for `extract_forced_probes_from_games`. |
| `tests/probes/README.md` | Add section distinguishing the committed bootstrap suite from the spec §7 curated gate suite. |
| `pytest.ini` | Add `integration` marker (per §7.2 opt-in policy). |

### Unchanged
- `scripts/GPU/alphazero/trainer.py` — no training-behavior changes. The existing inline probe code path (`trainer.py:1899-1921` loader, `trainer.py:2627-2699` per-iter call) is already wired correctly; it becomes active when `tests/probes/twixt_probes.json` lands in commit #2.
- `scripts/build_probe_candidates.py` — stays as-is for future curated-gate-suite work.
- Existing per-game sidecars in `scripts/GPU/logs/games/`.

### Commit grouping
Four ordered commits per spec §8.3. Each commit passes `pytest` standalone (integration test in commit 4 only under opt-in marker):

1. **Commit 1 (Tasks 1–7):** Shared primitives + their unit tests.
2. **Commit 2 (Tasks 8–11):** Bootstrap generator, committed suite, README, generator tests.
3. **Commit 3 (Tasks 12–18):** Analyzer wiring + end-to-end test.
4. **Commit 4 (Tasks 19–20):** Integration test + pytest marker config.

Code-changing tasks produce exactly one commit each. Pure verification tasks (Tasks 7 and 18) may produce no commit when all verification steps pass cleanly — they add commits only if a regression surfaces and needs a fix.

---

## Commit 1 — Shared primitives

### Task 1: Add `load_network_for_scoring` (public wrapper)

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py`
- Test: `tests/test_inline_probe_observability.py`

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_inline_probe_observability.py`:

```python
# ---------- load_network_for_scoring (public wrapper over _load_network) ----------

def test_load_network_for_scoring_public_symbol_exists():
    """The public wrapper is importable under the expected name."""
    from scripts.GPU.alphazero.probe_eval import load_network_for_scoring
    assert callable(load_network_for_scoring)


def test_load_network_for_scoring_matches_private_loader(tmp_path):
    """Public wrapper delegates to _load_network and returns the same shape."""
    from scripts.GPU.alphazero.probe_eval import (
        load_network_for_scoring, _load_network,
    )
    # Build a tiny network and save it to disk as a safetensors fixture.
    from scripts.GPU.alphazero.network import create_network
    net = create_network(in_channels=30, hidden=8, n_blocks=1)
    weights_path = tmp_path / "fixture.safetensors"
    net.save_weights(str(weights_path))

    pub = load_network_for_scoring(str(weights_path), verbose=False)
    priv = _load_network(str(weights_path), hidden=8, n_blocks=1, verbose=False)
    # Both return 4-tuples: (net, in_channels, hidden, n_blocks)
    assert len(pub) == 4
    assert pub[1] == priv[1] == 30   # in_channels
    assert pub[2] == priv[2]         # hidden
    assert pub[3] == priv[3]         # n_blocks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inline_probe_observability.py::test_load_network_for_scoring_public_symbol_exists -v`
Expected: `ImportError: cannot import name 'load_network_for_scoring'`

- [ ] **Step 3: Add the public wrapper**

Append to `scripts/GPU/alphazero/probe_eval.py` (after the existing `_load_network` function, around line 96):

```python
def load_network_for_scoring(weights_path: str, verbose: bool = False):
    """Public wrapper over _load_network.

    Provides a stable import symbol for the trainer, analyzer, and bootstrap
    generator to share. Returns (network, in_channels, hidden, n_blocks) with
    auto-detection of 24-channel vs 30-channel checkpoints.

    See _load_network for full docstring.
    """
    return _load_network(weights_path, hidden=None, n_blocks=None, verbose=verbose)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inline_probe_observability.py -k "load_network_for_scoring" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_inline_probe_observability.py
git commit -m "feat(probe_eval): add load_network_for_scoring public wrapper

Stable import symbol over _load_network so analyzer and bootstrap
generator can load checkpoints without reaching through underscore-
prefix internals. Zero behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `extract_forced_probes_from_games` — basic extraction + category-from-winner

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py`
- Test: `tests/test_inline_probe_observability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inline_probe_observability.py`:

```python
# ---------- extract_forced_probes_from_games ----------

def _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red",
                   reason="win", board_size=24, moves=None):
    """Minimal parsed-game-JSON shape matching what load_replays produces.

    Moves list: [{"player": "red"/"black", "move": [r, c]}, ...]
    """
    if moves is None:
        # Generate n_moves of alternating moves at arbitrary (but legal-looking) cells.
        moves = []
        for i in range(n_moves):
            player = "red" if i % 2 == 0 else "black"
            moves.append({"player": player, "move": [(i * 3) % board_size, (i * 5) % board_size]})
    return {
        "id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "meta": {
            "board_size": board_size,
            "iteration": iteration,
            "game_idx": game_idx,
            "reason": reason,
            "n_moves": n_moves,
            "starting_player": "red",
        },
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }


def test_extract_forced_probes_basic_two_per_game():
    """A single natural-win game at size 24 yields 2 probes (K=2: plies n_moves-1 and n_moves-2)."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([game], active_size=24, k_plies=2)
    assert len(probes) == 2
    plies = sorted(p["ply"] for p in probes)
    assert plies == [38, 39]  # n_moves-2 and n_moves-1


def test_extract_forced_probes_category_from_winner_not_side_to_move():
    """Category is based on eventual winner, independent of side_to_move."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    # Red wins, 40 moves. At ply 38 (even index after starting red), side-to-move
    # alternates. Category must be 'near_win_red' for BOTH probes regardless of stm.
    game = _make_game_dict(iteration=29, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([game], active_size=24)
    assert all(p["category"] == "near_win_red" for p in probes)
    # Now black wins: both probes are near_win_black.
    game_b = _make_game_dict(iteration=29, n_moves=40, winner="black")
    probes_b = extract_forced_probes_from_games([game_b], active_size=24)
    assert all(p["category"] == "near_win_black" for p in probes_b)


def test_extract_forced_probes_deterministic_ids():
    """IDs have the form {basename}_ply{ply:03d}_{winner} and are stable across reruns."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game = _make_game_dict(iteration=29, game_idx=42, n_moves=40, winner="red")
    probes_1 = extract_forced_probes_from_games([game], active_size=24)
    probes_2 = extract_forced_probes_from_games([game], active_size=24)
    ids_1 = sorted(p["id"] for p in probes_1)
    ids_2 = sorted(p["id"] for p in probes_2)
    assert ids_1 == ids_2
    assert "iter_0029_game_042_ply038_red" in ids_1
    assert "iter_0029_game_042_ply039_red" in ids_1


def test_extract_forced_probes_confidence_field_forced():
    """Every emitted probe has confidence='forced'."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game = _make_game_dict(iteration=29, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([game], active_size=24)
    assert all(p["confidence"] == "forced" for p in probes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_inline_probe_observability.py -k "extract_forced_probes" -v`
Expected: FAIL with `ImportError: cannot import name 'extract_forced_probes_from_games'`

- [ ] **Step 3: Implement the function — minimal version passing the 4 tests**

**Perspective convention (important — do not skip):**

All value numbers in the probe pipeline are stored in **red-perspective**:
- `expected_value_sign = +1` means "red wins from red's point of view"
- `expected_value_sign = -1` means "black wins from red's point of view"
- The network's raw `nn_value` is naturally in `state.to_move` perspective; the existing `run_forced_probes_inline` (at `scripts/GPU/alphazero/probe_eval.py:264-266`) negates it when `state.to_move == "black"` so the sign-match comparison against `expected_value_sign` is apples-to-apples in red-perspective.
- The existing `_eval_probe` function at `probe_eval.py:153-157` does the identical conversion, so the bootstrap generator, inline trainer calls, and the analyzer's replay_probe_scoring all share the same perspective contract.

Therefore the implementation sets `expected_value_sign = +1 if winner == "red" else -1`, and the comment in the code reflects the convention explicitly so future readers don't have to rediscover it.

Append to `scripts/GPU/alphazero/probe_eval.py` (after `run_forced_probes_inline`):

```python
def extract_forced_probes_from_games(
    games: list[dict],
    active_size: int = 24,
    k_plies: int = 2,
    winner_reasons: frozenset = frozenset({"win"}),
    dedupe_exact: bool = True,
    dedupe_mirror: bool = True,
    max_probes: int | None = None,
) -> list[dict]:
    """Extract near-terminal forced probes from parsed game JSONs.

    See spec §4.1 for full semantics. This is the shared primitive used by
    both the analyzer (replay-derived probe scoring) and the bootstrap
    probe suite generator.
    """
    probes: list[dict] = []

    for game in games:
        meta = game.get("meta") or {}
        if meta.get("board_size") != active_size:
            continue
        if meta.get("reason") not in winner_reasons:
            continue
        winner = game.get("winner")
        if winner not in ("red", "black"):
            continue

        moves_list = game.get("moves") or []
        n_moves = len(moves_list)
        if n_moves < k_plies + 1:
            continue

        # Build move_history as list of [r, c] pairs (ply-ordered).
        move_history = [list(m["move"]) for m in moves_list]
        source_game_basename = game.get("id") or f"iter_{meta.get('iteration', 0):04d}_game_{meta.get('game_idx', 0):03d}"
        source_iteration = meta.get("iteration", 0)
        category = f"near_win_{winner}"

        # Perspective convention (shared with run_forced_probes_inline and
        # _eval_probe): expected_value_sign is stored in RED-PERSPECTIVE.
        #   +1 = red wins from red's point of view
        #   -1 = black wins from red's point of view
        # The scoring code converts raw nn_value to red-perspective by negating
        # it when state.to_move == "black" (probe_eval.py:264-266), so this
        # comparison is apples-to-apples.
        expected_value_sign = +1 if winner == "red" else -1
        starting_player = game.get("starting_player") or "red"

        # Emit K probes at plies n_moves-1, n_moves-2, ..., n_moves-k_plies (down to 0).
        for k in range(1, k_plies + 1):
            ply = n_moves - k
            if ply < 0:
                continue
            # Side-to-move at this ply: starting_player if ply%2==0 else the other.
            side_to_move = starting_player if ply % 2 == 0 else ("black" if starting_player == "red" else "red")
            probe = {
                "id": f"{source_game_basename}_ply{ply:03d}_{winner}",
                "category": category,
                "confidence": "forced",
                "side_to_move": side_to_move,
                "expected_value_sign": expected_value_sign,
                "active_size": active_size,
                "ply": ply,
                "move_history": move_history[:ply],
                "source_game": source_game_basename,
                "source_ply": ply,
                "_source_iteration": source_iteration,  # sort-only; stripped before return
            }
            probes.append(probe)

    # Dedup (exact + mirror) will be added in Task 4.
    # Sort + truncate will be added in Task 5.

    # Strip internal sort-only keys before returning.
    for p in probes:
        p.pop("_source_iteration", None)
    return probes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inline_probe_observability.py -k "extract_forced_probes" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_inline_probe_observability.py
git commit -m "feat(probe_eval): add extract_forced_probes_from_games — basic extraction

Winner-based category (near_win_red / near_win_black), deterministic
{basename}_ply{NNN}_{winner} IDs, confidence='forced', K=2 positions
per game (plies n_moves-1 and n_moves-2). Dedup and sort semantics
added in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add filtering — natural wins only, active_size, invalid winners

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py`
- Test: `tests/test_inline_probe_observability.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_inline_probe_observability.py`:

```python
def test_extract_forced_probes_natural_wins_only():
    """Resigns, adjudicated games, draws, and timeouts produce zero probes."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    for bad_reason in ("resign", "adjudicated", "timeout", "board_full", "state_cap", "unknown"):
        game = _make_game_dict(iteration=29, n_moves=40, winner="red", reason=bad_reason)
        probes = extract_forced_probes_from_games([game], active_size=24)
        assert probes == [], f"reason={bad_reason!r} should yield zero probes"


def test_extract_forced_probes_active_size_filter():
    """Games at wrong board size produce zero probes."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game_16 = _make_game_dict(iteration=29, n_moves=40, winner="red", board_size=16)
    assert extract_forced_probes_from_games([game_16], active_size=24) == []
    # But size 24 games still work.
    game_24 = _make_game_dict(iteration=29, n_moves=40, winner="red", board_size=24)
    assert len(extract_forced_probes_from_games([game_24], active_size=24)) == 2


def test_extract_forced_probes_invalid_winner_skipped():
    """Games with winner None, 'draw', or any unexpected value are skipped."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    for bad_winner in (None, "draw", "", "unknown"):
        game = _make_game_dict(iteration=29, n_moves=40, winner=bad_winner)
        probes = extract_forced_probes_from_games([game], active_size=24)
        assert probes == [], f"winner={bad_winner!r} should yield zero probes"
```

- [ ] **Step 2: Run tests to verify they pass (filters already present from Task 2)**

Run: `pytest tests/test_inline_probe_observability.py -k "extract_forced_probes" -v`
Expected: PASS — the filters (`board_size`, `reason in winner_reasons`, `winner in ('red','black')`) were already added to the Task 2 implementation. These tests confirm the existing behavior.

- [ ] **Step 3: Commit**

```bash
git add tests/test_inline_probe_observability.py
git commit -m "test(probe_eval): lock in filter semantics for extract_forced_probes_from_games

Covers: natural-wins-only filter rejecting resign/adjudicated/timeout/
board_full/state_cap/unknown; active_size filter rejecting mismatched
board sizes; invalid-winner skip (None, 'draw', '', unknown).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add deduplication — exact + 4-form mirror canonical

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py`
- Test: `tests/test_inline_probe_observability.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_inline_probe_observability.py`:

```python
def test_extract_forced_probes_exact_dedupe():
    """Two games with byte-identical move_history yield probes dedup'd on exact match."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    # Same moves, different game id — the move_history is what gets dedup'd.
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % 24, (i * 3) % 24]}
             for i in range(40)]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=moves)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24,
                                              dedupe_exact=True, dedupe_mirror=False)
    # Only 2 survive (one copy of each ply), not 4.
    assert len(probes) == 2


def test_extract_forced_probes_mirror_dedupe_horizontal():
    """A game and its horizontal mirror (c → N-1-c) collapse to one copy."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    mirrored = [{"player": m["player"], "move": [m["move"][0], N - 1 - m["move"][1]]}
                for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=mirrored)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24,
                                              dedupe_exact=True, dedupe_mirror=True)
    assert len(probes) == 2, f"horizontal mirror should collapse, got {len(probes)} probes"


def test_extract_forced_probes_mirror_dedupe_vertical():
    """Vertical mirror (r → N-1-r) also collapses."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    mirrored = [{"player": m["player"], "move": [N - 1 - m["move"][0], m["move"][1]]}
                for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=mirrored)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24, dedupe_mirror=True)
    assert len(probes) == 2


def test_extract_forced_probes_mirror_dedupe_180_rotation():
    """180° rotation (r → N-1-r AND c → N-1-c) also collapses."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    mirrored = [{"player": m["player"], "move": [N - 1 - m["move"][0], N - 1 - m["move"][1]]}
                for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=mirrored)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24, dedupe_mirror=True)
    assert len(probes) == 2


def test_extract_forced_probes_transpose_NOT_deduped():
    """Transpose (r,c)→(c,r) swaps red/black goals and must NOT dedup."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    transposed = [{"player": m["player"], "move": [m["move"][1], m["move"][0]]}
                  for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=transposed)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24, dedupe_mirror=True)
    assert len(probes) == 4, "transpose must not dedupe — goals differ"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_inline_probe_observability.py -k "dedupe" -v`
Expected: FAIL — current implementation has no dedup logic; the exact test and mirror tests will each return 4 probes instead of 2.

- [ ] **Step 3: Add dedup logic to `extract_forced_probes_from_games`**

Replace the placeholder comment in `scripts/GPU/alphazero/probe_eval.py` (the `# Dedup ... will be added in Task 4.` line) with this block, inserted just before the `# Strip internal sort-only keys` line:

```python
    # Dedup: exact + 4-form mirror canonical (spec §4.1).
    if dedupe_exact or dedupe_mirror:
        seen_keys: set = set()
        deduped: list[dict] = []

        def _canon_key(move_history: list[list[int]], N: int, use_mirror: bool) -> tuple:
            # Always include the identity form.
            forms = [tuple(tuple(m) for m in move_history)]
            if use_mirror:
                # Horizontal: (r, c) → (r, N-1-c)
                forms.append(tuple((r, N - 1 - c) for (r, c) in move_history))
                # Vertical: (r, c) → (N-1-r, c)
                forms.append(tuple((N - 1 - r, c) for (r, c) in move_history))
                # 180°: (r, c) → (N-1-r, N-1-c)
                forms.append(tuple((N - 1 - r, N - 1 - c) for (r, c) in move_history))
            return min(forms)  # lex-smallest is the canonical form

        N = active_size
        for p in probes:
            key = _canon_key(p["move_history"], N, dedupe_mirror)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(p)
        probes = deduped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inline_probe_observability.py -k "extract_forced_probes" -v`
Expected: PASS (9 tests total at this point — 4 from Task 2, 3 from Task 3, 5 from Task 4, minus overlap)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_inline_probe_observability.py
git commit -m "feat(probe_eval): add exact + 4-form mirror dedupe to extract_forced_probes

Canonical key is lex-smallest of (identity, horizontal, vertical, 180°)
move_history forms. Transpose is excluded — it would swap red/black
goal orientations and conflate near_win_red with near_win_black.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add sort order + `max_probes` truncation

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py`
- Test: `tests/test_inline_probe_observability.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_inline_probe_observability.py`:

```python
def test_extract_forced_probes_sort_order():
    """Sort: source_iteration desc, source_ply desc, source_game basename asc."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    # Mix two iterations; verify iter 30 comes first, then within iter, later ply first.
    g_old = _make_game_dict(iteration=25, game_idx=0, n_moves=40, winner="red")
    g_new = _make_game_dict(iteration=30, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([g_old, g_new], active_size=24,
                                              dedupe_exact=False, dedupe_mirror=False)
    # Expected order: (iter=30, ply=39), (iter=30, ply=38), (iter=25, ply=39), (iter=25, ply=38)
    assert probes[0]["source_game"].startswith("iter_0030")
    assert probes[0]["source_ply"] == 39
    assert probes[1]["source_game"].startswith("iter_0030")
    assert probes[1]["source_ply"] == 38
    assert probes[2]["source_game"].startswith("iter_0025")
    assert probes[2]["source_ply"] == 39
    assert probes[3]["source_game"].startswith("iter_0025")
    assert probes[3]["source_ply"] == 38


def test_extract_forced_probes_max_probes_truncation():
    """max_probes=N truncates after sorting."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    g_old = _make_game_dict(iteration=25, game_idx=0, n_moves=40, winner="red")
    g_new = _make_game_dict(iteration=30, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([g_old, g_new], active_size=24,
                                              dedupe_exact=False, dedupe_mirror=False,
                                              max_probes=2)
    assert len(probes) == 2
    # Should keep the two most-recent-iter probes.
    assert all(p["source_game"].startswith("iter_0030") for p in probes)


def test_extract_forced_probes_max_probes_none_returns_all_sorted():
    """max_probes=None returns all probes, still in sorted order."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    g_old = _make_game_dict(iteration=25, game_idx=0, n_moves=40, winner="red")
    g_new = _make_game_dict(iteration=30, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([g_old, g_new], active_size=24,
                                              dedupe_exact=False, dedupe_mirror=False,
                                              max_probes=None)
    assert len(probes) == 4
    # Still sorted: first probe is from iter 30.
    assert probes[0]["source_game"].startswith("iter_0030")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_inline_probe_observability.py -k "sort_order or max_probes" -v`
Expected: FAIL — no sort applied, iteration order depends on input order.

- [ ] **Step 3: Add sort + truncation logic**

In `scripts/GPU/alphazero/probe_eval.py`, modify `extract_forced_probes_from_games` — insert this block **before** the existing "Strip internal sort-only keys" block:

```python
    # Sort: source_iteration desc, source_ply desc, source_game basename asc
    # (tiebreaker). Always applied, even when max_probes is None, so downstream
    # consumers (per-probe CSV) see deterministic order.
    probes.sort(key=lambda p: (
        -p["_source_iteration"],
        -p["source_ply"],
        p["source_game"],
    ))

    # Truncate to max_probes after sorting.
    if max_probes is not None and len(probes) > max_probes:
        probes = probes[:max_probes]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inline_probe_observability.py -k "extract_forced_probes" -v`
Expected: PASS (all extract_forced_probes tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_inline_probe_observability.py
git commit -m "feat(probe_eval): add deterministic sort + max_probes truncation

Sort key: source_iteration desc, source_ply desc, source_game basename
asc. Applied unconditionally so downstream per-probe CSVs see stable
order regardless of max_probes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Add `score_samples_against_checkpoint` — stratified sampling + scoring

**Files:**
- Modify: `scripts/GPU/alphazero/value_calibration.py`
- Create: `tests/test_value_calibration_sampling.py`

- [ ] **Step 1: Create the test file with failing tests**

Create `tests/test_value_calibration_sampling.py`:

```python
"""Tests for score_samples_against_checkpoint (stratified calibration sampling).

Covers:
- Stratified budget: per-bucket caps honored, stable alphabetical ordering
  when max_total binds
- natural_distribution reports counts across the full pool
- 24-channel checkpoint smoke: no crash, real buckets populated
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.GPU.alphazero.network import create_network


def _make_game_for_calibration(n_moves=40, winner="red", board_size=24):
    """Parsed-game JSON with enough moves for classify_position to produce
    a mix of buckets across plies."""
    moves = []
    for i in range(n_moves):
        player = "red" if i % 2 == 0 else "black"
        # Scatter moves across the board so replayed states have varied connectivity.
        moves.append({"player": player, "move": [(i * 7) % board_size, (i * 11) % board_size]})
    return {
        "id": f"iter_0029_game_{n_moves:03d}",
        "meta": {"board_size": board_size, "iteration": 29, "reason": "win", "n_moves": n_moves},
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }


@pytest.fixture
def tiny_30ch_network(tmp_path):
    net = create_network(in_channels=30, hidden=8, n_blocks=1)
    path = tmp_path / "tiny_30ch.safetensors"
    net.save_weights(str(path))
    return net, str(path)


def test_score_samples_natural_distribution_reported(tiny_30ch_network):
    """natural_distribution counts every position in the full pool."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    # 3 games × 40 moves → 120 positions total.
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(3)]
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=5, max_total=2000
    )
    # Sum across buckets should equal total positions across all games.
    total_natural = sum(result["natural_distribution"].values())
    total_positions = sum(len(g["moves"]) for g in replays)
    assert total_natural == total_positions


def test_score_samples_stratified_per_bucket_caps(tiny_30ch_network):
    """For each bucket, sampled_count <= min(samples_per_bucket, natural_count)."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(5)]
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=3, max_total=2000
    )
    for bucket, sampled in result["sampled_distribution"].items():
        natural = result["natural_distribution"][bucket]
        assert sampled <= min(3, natural), (
            f"bucket={bucket!r} sampled={sampled} exceeds min(cap=3, natural={natural})"
        )


def test_score_samples_stratified_flag_and_note(tiny_30ch_network):
    """Output advertises itself as stratified."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(2)]
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=5, max_total=2000
    )
    assert result["stratified"] is True
    assert "stratified" in result["overall_note"].lower()
    assert "aggregate" in result  # carries the existing aggregate_calibration schema


def test_score_samples_max_total_binds_alphabetical_halt(tiny_30ch_network):
    """When max_total binds, later-alphabetical buckets get sampled=0."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    # Enough games to populate multiple buckets heavily.
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(20)]
    # Set max_total so it binds after ~2 buckets worth of samples.
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=200, max_total=50
    )
    total_sampled = sum(result["sampled_distribution"].values())
    assert total_sampled <= 50
    # At least one bucket should have sampled=0 given the tight cap.
    zero_buckets = [b for b, n in result["sampled_distribution"].items() if n == 0]
    # Note: if natural distribution is lopsided, zero_buckets could be empty only
    # if all 50 samples fit in the first alphabetical bucket. Assert the ordering
    # invariant: any zero-bucket must be alphabetically AFTER the last nonzero bucket.
    sampled_names_sorted = sorted(result["sampled_distribution"].keys())
    saw_nonzero = False
    saw_zero_after_nonzero = False
    for name in sampled_names_sorted:
        n = result["sampled_distribution"][name]
        if n > 0:
            if saw_zero_after_nonzero:
                pytest.fail(f"bucket ordering violated: {name!r} nonzero after a zero bucket")
            saw_nonzero = True
        elif saw_nonzero:
            saw_zero_after_nonzero = True


def test_score_samples_24ch_checkpoint_no_crash(tmp_path):
    """A 24-channel checkpoint still produces real bucket stats (structural
    classification is state-based, independent of network channel count)."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net_24 = create_network(in_channels=24, hidden=8, n_blocks=1)
    replays = [_make_game_for_calibration(n_moves=30) for _ in range(3)]
    result = score_samples_against_checkpoint(
        replays, network=net_24, samples_per_bucket=3, max_total=100
    )
    # Real buckets populate — not just 'unknown'.
    buckets = list(result["natural_distribution"].keys())
    assert any(b != "unknown" for b in buckets)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_value_calibration_sampling.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_samples_against_checkpoint'`

- [ ] **Step 3: Implement `score_samples_against_checkpoint`**

Append to `scripts/GPU/alphazero/value_calibration.py`:

```python
def score_samples_against_checkpoint(
    replays: List[dict],
    network,
    samples_per_bucket: int = 200,
    max_total: int = 2000,
    min_size: int = 8,
) -> dict:
    """Phase-stratified calibration scoring.

    See spec §4.3 for full semantics. Pre-pass classifies every position in
    the replay pool; sample pass fills per-bucket caps in alphabetical
    order, halting when max_total binds; score pass runs NN forward for each
    sampled position and feeds the results to aggregate_calibration.
    """
    import random
    import numpy as np
    from .local_evaluator import LocalGPUEvaluator
    from .game.twixt_state import TwixtState

    # ---- Pre-pass: enumerate & classify every position ----
    # Positions are (game_idx, ply) pairs. classify_position needs TwixtState
    # + ply + game_n_moves.
    by_bucket_positions: dict[str, list[tuple[int, int]]] = {}
    natural_distribution: dict[str, int] = {}

    for g_idx, game in enumerate(replays):
        moves = game.get("moves") or []
        n_moves = len(moves)
        # Reconstruct state ply-by-ply so we classify each intermediate state.
        state = TwixtState(active_size=game.get("meta", {}).get("board_size", 24))
        for ply in range(n_moves):
            bucket = classify_position(state, ply, n_moves, min_size=min_size)
            natural_distribution[bucket] = natural_distribution.get(bucket, 0) + 1
            by_bucket_positions.setdefault(bucket, []).append((g_idx, ply))
            # Advance state for next ply.
            mv = moves[ply]["move"]
            state = state.apply_move((int(mv[0]), int(mv[1])))

    # ---- Sample pass: per-bucket caps, stable alphabetical order, halt on max_total ----
    rng = random.Random(42)  # deterministic sampling for reproducibility
    sampled_distribution: dict[str, int] = {b: 0 for b in natural_distribution}
    sampled_positions: list[tuple[str, int, int]] = []  # (bucket, game_idx, ply)
    cumulative = 0

    for bucket in sorted(natural_distribution.keys()):
        if cumulative >= max_total:
            break  # budget exhausted — remaining buckets get sampled=0
        bucket_pool = by_bucket_positions[bucket]
        cap = min(samples_per_bucket, len(bucket_pool), max_total - cumulative)
        if cap <= 0:
            continue
        chosen = rng.sample(bucket_pool, cap)
        for g_idx, ply in chosen:
            sampled_positions.append((bucket, g_idx, ply))
        sampled_distribution[bucket] = cap
        cumulative += cap

    # ---- Score pass: forward-pass each sampled position ----
    evaluator = LocalGPUEvaluator(network)
    samples: list[dict] = []

    for bucket, g_idx, ply in sampled_positions:
        game = replays[g_idx]
        moves = game["moves"]
        state = TwixtState(active_size=game.get("meta", {}).get("board_size", 24))
        for i in range(ply):
            mv = moves[i]["move"]
            state = state.apply_move((int(mv[0]), int(mv[1])))
        tensor = evaluator.build_input_tensor(state)
        tensor = np.transpose(tensor, (1, 2, 0))
        boards_np = np.expand_dims(tensor.astype(np.float32), axis=0)
        legal = state.legal_moves()
        move_rows_np = np.array([[m[0] for m in legal]], dtype=np.int32)
        move_cols_np = np.array([[m[1] for m in legal]], dtype=np.int32)
        move_mask_np = np.ones((1, len(legal)), dtype=np.float32)
        _, values_np = evaluator.infer(
            boards_np, move_rows_np, move_cols_np, move_mask_np, state.active_size
        )
        nn_value = float(values_np[0])
        # Red-perspective convention.
        if state.to_move == "black":
            nn_value = -nn_value
        # Outcome in red-perspective: +1 red wins, -1 black wins, 0 draw.
        winner = game.get("winner")
        if winner == "red":
            outcome = 1.0
        elif winner == "black":
            outcome = -1.0
        else:
            outcome = 0.0
        samples.append({"bucket": bucket, "nn_value": nn_value, "outcome": outcome})

    aggregate = aggregate_calibration(samples, n_bins=5)

    return {
        "samples_per_bucket_target": samples_per_bucket,
        "max_total": max_total,
        "natural_distribution": natural_distribution,
        "sampled_distribution": sampled_distribution,
        "stratified": True,
        "overall_note": "stratified aggregate, not population-weighted",
        "aggregate": aggregate,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_value_calibration_sampling.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/value_calibration.py tests/test_value_calibration_sampling.py
git commit -m "feat(value_calibration): add score_samples_against_checkpoint

Phase-stratified sampling over classify_position buckets. Per-bucket
cap honored, stable alphabetical ordering when max_total binds, no
redistribution across buckets. Emits natural_distribution alongside
sampled_distribution so readers see the underlying pool size.
Structural classification is state-based and works identically for
24ch and 30ch checkpoints.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Commit-1 sanity check — full test suite passes

**Files:** (none modified — verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/test_inline_probe_observability.py tests/test_value_calibration_sampling.py -v`
Expected: PASS for all tests added in Tasks 1–6.

- [ ] **Step 2: Run broader regression check**

Run: `pytest -x`
Expected: No regressions — all previously-passing tests still pass. If any fail unrelated to this plan, investigate whether they were pre-existing.

- [ ] **Step 3: Verify no linting regressions in modified files**

Run: `python -m py_compile scripts/GPU/alphazero/probe_eval.py scripts/GPU/alphazero/value_calibration.py`
Expected: No syntax errors (exit 0, no output).

- [ ] **Step 4: Commit a checkpoint marker if any drift noticed, else skip**

If steps 1–3 all pass cleanly, no additional commit is needed — Task 7 is pure verification. If any issue is found, fix it in a new commit before proceeding to Commit 2.

---

## Commit 2 — Bootstrap generator + committed suite + README

### Task 8: Create `scripts/build_bootstrap_probe_suite.py` — CLI skeleton

**Files:**
- Create: `scripts/build_bootstrap_probe_suite.py`
- Create: `tests/test_bootstrap_probe_suite.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_bootstrap_probe_suite.py`:

```python
"""Tests for the bootstrap probe suite generator.

Covers:
- CLI --help responds
- Deterministic byte-identical output on rerun
- No wall-clock fields in meta
- Only natural wins emitted
- Schema matches tests/probes/README.md expectations
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_bootstrap_cli_help():
    """Bootstrap generator responds to --help."""
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "--source-iter-range" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bootstrap_probe_suite.py::test_bootstrap_cli_help -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Create skeleton script**

Create `scripts/build_bootstrap_probe_suite.py`:

```python
"""Bootstrap rule-selected forced-probe suite generator.

Produces tests/probes/twixt_probes.json from historical game replays
using strict rule-based selection (no human review). See the spec at
docs/superpowers/specs/2026-04-21-probes-and-calibration-closure-design.md
§5 for selection rules.

The output is a rule-selected bootstrap suite, NOT the spec §7 review-
curated gate suite. See tests/probes/README.md for the distinction.

Reruns with identical --source-iter-range produce byte-identical output
(deterministic probe IDs, deterministic dedup canonicalization, stable
sort keys, no wall-clock fields).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--input", default="scripts/GPU/logs/games",
                    help="Directory containing iter_NNNN_game_MMM.json files.")
    ap.add_argument("--source-iter-range", nargs=2, type=int, required=True,
                    metavar=("MIN", "MAX"),
                    help="Inclusive iteration range to sample from (e.g., 25 30).")
    ap.add_argument("--out", default="tests/probes/twixt_probes.json",
                    help="Output path.")
    ap.add_argument("--samples-per-bucket", type=int, default=12,
                    help="Per winner class, before dedup.")
    ap.add_argument("--max-probes", type=int, default=30,
                    help="Final cap on probe count.")
    args = ap.parse_args()

    # Real implementation in Task 9.
    print(f"[bootstrap] input={args.input}")
    print(f"[bootstrap] source-iter-range={args.source_iter_range}")
    print(f"[bootstrap] out={args.out}")
    raise NotImplementedError("Generation logic lands in Task 9.")


if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bootstrap_probe_suite.py::test_bootstrap_cli_help -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_bootstrap_probe_suite.py tests/test_bootstrap_probe_suite.py
git commit -m "feat(scripts): add build_bootstrap_probe_suite.py skeleton

CLI parsing + help text. Generation logic lands in a follow-up commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Implement bootstrap generation logic — filter, balance, serialize

**Files:**
- Modify: `scripts/build_bootstrap_probe_suite.py`
- Modify: `tests/test_bootstrap_probe_suite.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_bootstrap_probe_suite.py`:

```python
def _write_fake_game(dir_path: Path, iteration: int, game_idx: int,
                     n_moves: int = 40, winner: str = "red",
                     reason: str = "win", board_size: int = 24):
    """Write a synthetic iter_NNNN_game_MMM.json matching the analyzer's
    replay format."""
    moves = [{"player": "red" if i % 2 == 0 else "black",
              "move": [(i * 7 + iteration * 3 + game_idx) % board_size,
                       (i * 11 + iteration * 5 + game_idx) % board_size]}
             for i in range(n_moves)]
    path = dir_path / f"iter_{iteration:04d}_game_{game_idx:03d}.json"
    path.write_text(json.dumps({
        "id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "meta": {"board_size": board_size, "iteration": iteration,
                 "game_idx": game_idx, "reason": reason, "n_moves": n_moves,
                 "starting_player": "red"},
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }))
    return path


def test_bootstrap_deterministic_rerun(tmp_path):
    """Two consecutive runs with identical inputs produce byte-identical output."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # 10 natural-win red games + 10 black at iter 30.
    for i in range(10):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
        _write_fake_game(games_dir, iteration=30, game_idx=100 + i, winner="black")

    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    for out in (out1, out2):
        result = subprocess.run(
            [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
             "--input", str(games_dir),
             "--source-iter-range", "30", "30",
             "--out", str(out),
             "--max-probes", "20"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    assert out1.read_bytes() == out2.read_bytes(), "byte-identity broken"


def test_bootstrap_no_wall_clock_fields(tmp_path):
    """Output meta contains no generated_at / timestamp / created_at etc."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(5):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
        _write_fake_game(games_dir, iteration=30, game_idx=100 + i, winner="black")

    out = tmp_path / "out.json"
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    meta = data.get("meta", {})
    for forbidden in ("generated_at", "timestamp", "created_at", "generation_time", "datetime"):
        assert forbidden not in meta, f"wall-clock field {forbidden!r} leaked into meta"


def test_bootstrap_only_natural_wins(tmp_path):
    """Resign/adjudicated/draw/timeout games produce zero probes."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # Mix of natural-win and other termination reasons.
    for i in range(5):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red", reason="win")
    for i, bad in enumerate(("resign", "adjudicated", "timeout", "board_full")):
        _write_fake_game(games_dir, iteration=30, game_idx=200 + i,
                         winner="red", reason=bad)

    out = tmp_path / "out.json"
    subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    sources = {p["source_game"] for p in data["probes"]}
    # Only natural-win source games (0..4) appear.
    for bad_idx in range(200, 204):
        assert f"iter_0030_game_{bad_idx:03d}" not in sources


def test_bootstrap_schema_fields(tmp_path):
    """Output conforms to tests/probes/README.md schema."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(10):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
        _write_fake_game(games_dir, iteration=30, game_idx=100 + i, winner="black")

    out = tmp_path / "out.json"
    subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    assert "meta" in data and "probes" in data
    assert data["meta"]["type"] == "bootstrap_rule_selected"
    assert data["meta"]["not_gate_suite"] is True
    for p in data["probes"]:
        for required in ("id", "category", "confidence", "side_to_move",
                         "expected_value_sign", "active_size", "ply",
                         "move_history", "source_game", "source_ply"):
            assert required in p, f"probe missing {required!r}"
        assert p["confidence"] == "forced"
        assert p["active_size"] == 24
        assert p["category"] in ("near_win_red", "near_win_black")


def test_bootstrap_balance_ratio(tmp_path):
    """Majority class is capped to ≤ 2:1 vs minority class."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # 50 red wins, 5 black wins — should truncate red to ~10 (2*5).
    for i in range(50):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
    for i in range(5):
        _write_fake_game(games_dir, iteration=30, game_idx=200 + i, winner="black")

    out = tmp_path / "out.json"
    subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out),
         "--max-probes", "100"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    red_count = sum(1 for p in data["probes"] if p["category"] == "near_win_red")
    black_count = sum(1 for p in data["probes"] if p["category"] == "near_win_black")
    assert red_count <= 2 * max(black_count, 1), (
        f"balance violated: red={red_count} black={black_count}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bootstrap_probe_suite.py -v`
Expected: FAIL with NotImplementedError from the skeleton.

- [ ] **Step 3: Replace the skeleton body with the real implementation**

Replace the `main()` function in `scripts/build_bootstrap_probe_suite.py`:

```python
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--input", default="scripts/GPU/logs/games")
    ap.add_argument("--source-iter-range", nargs=2, type=int, required=True,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--out", default="tests/probes/twixt_probes.json")
    ap.add_argument("--samples-per-bucket", type=int, default=12)
    ap.add_argument("--max-probes", type=int, default=30)
    args = ap.parse_args()

    # Add project root to sys.path so this script can import scripts.GPU.alphazero.*
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games

    min_iter, max_iter = args.source_iter_range
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[bootstrap] ERROR: --input path is not a directory: {input_dir}",
              file=sys.stderr)
        return 2

    # 1. Scan for iter_NNNN_game_MMM.json in range.
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

    # 2. Extract probes via shared helper (filters: size=24, natural wins,
    #    K=2, dedup exact+mirror).
    probes = extract_forced_probes_from_games(
        games,
        active_size=24,
        k_plies=2,
        winner_reasons=frozenset({"win"}),
        dedupe_exact=True,
        dedupe_mirror=True,
        max_probes=None,  # we balance and truncate below
    )

    # 3. Split by category and enforce ≤ 2:1 balance.
    red = [p for p in probes if p["category"] == "near_win_red"]
    black = [p for p in probes if p["category"] == "near_win_black"]
    if len(red) > 2 * max(len(black), 1):
        red = red[: 2 * max(len(black), 1)]
    if len(black) > 2 * max(len(red), 1):
        black = black[: 2 * max(len(red), 1)]
    balanced = red + black

    # 4. Re-sort balanced set (per spec §4.1 order) and truncate to max_probes.
    # extract_forced_probes_from_games already returned in sort order, but
    # concatenation of red+black may disrupt it — re-sort by the same keys.
    def _sort_key(p: dict) -> tuple:
        # Extract iteration from source_game basename 'iter_NNNN_game_MMM'.
        basename = p["source_game"]
        try:
            iter_num = int(basename.split("_")[1])
        except (IndexError, ValueError):
            iter_num = 0
        return (-iter_num, -p["source_ply"], basename)

    balanced.sort(key=_sort_key)
    if len(balanced) > args.max_probes:
        balanced = balanced[: args.max_probes]

    # 5. Serialize (no wall-clock fields).
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

    print(f"[bootstrap] wrote {len(balanced)} probes to {out_path}")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bootstrap_probe_suite.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_bootstrap_probe_suite.py tests/test_bootstrap_probe_suite.py
git commit -m "feat(scripts): implement bootstrap probe suite generation

Filters: board_size==24, reason=='win', natural wins only. Uses the
shared extract_forced_probes_from_games helper for extraction, then
applies ≤2:1 red/black balance and max_probes truncation. Output is
byte-identical across reruns (deterministic sort, no wall-clock
fields). Covered by determinism, schema, balance, and natural-wins
tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Generate and commit the bootstrap probes file

**Files:**
- Create: `tests/probes/twixt_probes.json`

- [ ] **Step 1: Pre-flight — confirm source iters exist**

Run: `ls scripts/GPU/logs/games/iter_002{5,6,7,8,9}_game_*.json scripts/GPU/logs/games/iter_0030_game_*.json 2>/dev/null | wc -l`
Expected: A count > 0 (user has games for iters 25–30; per earlier analysis ~600 games exist).

If zero, use a different `--source-iter-range` matching what's actually present. Run `ls scripts/GPU/logs/games/ | head` to inspect.

- [ ] **Step 2: Generate the probe file**

Run:
```bash
.venv/bin/python scripts/build_bootstrap_probe_suite.py \
    --input scripts/GPU/logs/games \
    --source-iter-range 25 30 \
    --out tests/probes/twixt_probes.json \
    --max-probes 30
```

Expected output: `[bootstrap] wrote N probes to tests/probes/twixt_probes.json` where `20 <= N <= 30`.

- [ ] **Step 3: Inspect the output**

Run: `python -c "import json; d=json.load(open('tests/probes/twixt_probes.json')); print('probes:', len(d['probes'])); print('meta:', d['meta']); print('categories:', {c: sum(1 for p in d['probes'] if p['category']==c) for c in ('near_win_red','near_win_black')})"`

Expected: probes count in [20, 30], meta shows `type=bootstrap_rule_selected`, red/black roughly balanced (≤ 2:1).

- [ ] **Step 4: Commit the file**

```bash
git add tests/probes/twixt_probes.json
git commit -m "chore(probes): commit bootstrap rule-selected probe suite

Generated via scripts/build_bootstrap_probe_suite.py from size-24
natural-win games in iters 25-30. Labeled not_gate_suite=true in its
meta block — this is a bootstrap suite for trainer-side telemetry and
regression monitoring, NOT the spec §7 review-curated gate suite. See
tests/probes/README.md.

Side effect: the trainer's existing inline probe path (trainer.py:
1899-1921 loader, 2627-2699 per-iter call) now has a file to consume.
Starting next training iter, forced_probe_summary in the per-iter
sidecar will populate with real values (n > 0) on active_size=24 iters.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Update `tests/probes/README.md` — bootstrap vs. gate-suite distinction

**Files:**
- Modify: `tests/probes/README.md`

- [ ] **Step 1: Read the current README to locate the insertion point**

Run: `head -40 tests/probes/README.md`
Expected: sections for Files, Categories, Confidence Tiers, Schema.

- [ ] **Step 2: Add a new section after the "Files" section**

Insert this block into `tests/probes/README.md` immediately after the "Files" section (before "Categories"):

```markdown
## Bootstrap vs. formal gate suite

The committed `twixt_probes.json` in this directory is a **bootstrap
rule-selected forced probe suite** (flagged `"type": "bootstrap_rule_selected"`
and `"not_gate_suite": true` in its `meta` block). It was produced
programmatically by `scripts/build_bootstrap_probe_suite.py` from
historical size-24 natural-win games using strict rule-based selection
— **no human review**.

This is **distinct** from the spec §7 *formal gate suite*, which requires
two-reviewer curation, per-probe category validation, and explicit
disagreement handling. The formal gate suite has not yet been produced.

**Implications:**

- The bootstrap suite is suitable for:
  - Trainer-side inline per-iter telemetry (`forced_probe_summary`)
  - Practical regression monitoring during training
  - Quick sanity checks on new checkpoints

- The bootstrap suite is **NOT** suitable as a substitute for the formal
  spec §7 gate evaluation. A 94% pass rate on the bootstrap suite is
  **not** equivalent to passing the ≥95% formal gate on a curated
  `confidence='forced'` tier — the two use different probe-selection
  methodologies and measure related-but-distinct quantities.

- Any gate decision promoting a training lineage under the formal spec §7
  criteria still requires the reviewed curated suite. The bootstrap suite
  can be replaced in place by the curated suite with no code changes
  (same file, same schema, trainer loader is methodology-agnostic).

To refresh the bootstrap suite against a newer training range, re-run
the generator:

```bash
.venv/bin/python scripts/build_bootstrap_probe_suite.py \
    --input scripts/GPU/logs/games \
    --source-iter-range <MIN> <MAX> \
    --out tests/probes/twixt_probes.json
```

Reruns with identical inputs produce byte-identical output (deterministic
probe IDs, deterministic dedup canonicalization, no wall-clock fields).
```

- [ ] **Step 3: Commit**

```bash
git add tests/probes/README.md
git commit -m "docs(probes): distinguish bootstrap suite from formal gate suite

Makes the boundary explicit so readers don't treat the committed
bootstrap probe file as equivalent to the spec §7 review-curated gate
suite. Documents the refresh workflow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Commit 3 — Analyzer wiring

### Task 12: Add checkpoint auto-discovery helper to analyzer

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Test: `tests/test_analyzer_replay_probe_scoring_end_to_end.py` (create; full suite lands in Task 18, but we add the first test now)

- [ ] **Step 1: Create the test file with the first failing test**

Create `tests/test_analyzer_replay_probe_scoring_end_to_end.py`:

```python
"""End-to-end tests for the analyzer's new replay_probe_scoring and
value_calibration integrations, plus the checkpoint auto-discovery helper.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _write_fake_replay(dir_path: Path, iteration: int, game_idx: int,
                      n_moves: int = 30, winner: str = "red", reason: str = "win"):
    moves = [{"player": "red" if i % 2 == 0 else "black",
              "move": [(i * 7 + game_idx) % 24, (i * 11 + game_idx) % 24]}
             for i in range(n_moves)]
    path = dir_path / f"iter_{iteration:04d}_game_{game_idx:03d}.json"
    path.write_text(json.dumps({
        "id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "meta": {"board_size": 24, "iteration": iteration, "game_idx": game_idx,
                 "reason": reason, "n_moves": n_moves, "starting_player": "red"},
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }))
    return path


def _write_fake_checkpoint(dir_path: Path, iteration: int, in_channels: int = 30):
    from scripts.GPU.alphazero.network import create_network
    net = create_network(in_channels=in_channels, hidden=8, n_blocks=1)
    path = dir_path / f"model_iter_{iteration:04d}.safetensors"
    net.save_weights(str(path))
    return path


# ---------- Checkpoint auto-discovery helper ----------

def test_resolve_checkpoint_explicit_weights_wins(tmp_path):
    """When --weights is passed, it takes precedence over auto-discovery."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    explicit = tmp_path / "explicit.safetensors"
    explicit.write_bytes(b"fake")
    # Use a minimal args-like object.
    class Args:
        weights = str(explicit)
        calibrate_weights = None
        checkpoint_dir = None
    replays = [{"meta": {"iteration": 29}}]
    assert _resolve_checkpoint_path(Args(), replays) == str(explicit)


def test_resolve_checkpoint_auto_discover_from_max_iter(tmp_path):
    """Auto-discovery maps max(meta.iteration) → model_iter_{N+1}.safetensors."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = _write_fake_checkpoint(ckpt_dir, iteration=30)
    class Args:
        weights = None
        calibrate_weights = None
        checkpoint_dir = str(ckpt_dir)
    replays = [{"meta": {"iteration": i}} for i in (27, 28, 29)]
    resolved = _resolve_checkpoint_path(Args(), replays)
    assert resolved == str(ckpt_path)


def test_resolve_checkpoint_not_found_returns_none(tmp_path):
    """When nothing is found, return None (analyzer will skip dependent sections)."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    class Args:
        weights = None
        calibrate_weights = None
        checkpoint_dir = str(tmp_path / "nonexistent")
    replays = [{"meta": {"iteration": 29}}]
    assert _resolve_checkpoint_path(Args(), replays) is None


def test_resolve_checkpoint_legacy_calibrate_weights_fallback(tmp_path):
    """Legacy --calibrate-weights path is honored when --weights not set."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    legacy = tmp_path / "legacy.safetensors"
    legacy.write_bytes(b"fake")
    class Args:
        weights = None
        calibrate_weights = str(legacy)
        checkpoint_dir = None
    replays = [{"meta": {"iteration": 29}}]
    assert _resolve_checkpoint_path(Args(), replays) == str(legacy)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py::test_resolve_checkpoint_explicit_weights_wins -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_checkpoint_path'`

- [ ] **Step 3: Add the helper to analyzer**

Add to `scripts/twixt_replay_analyzer.py`, placed near the top of the file right after imports (around line 95 in the existing file, before `_derive_out_suffix`):

```python
def _resolve_checkpoint_path(args, replays: List[dict]) -> Optional[str]:
    """Resolve a checkpoint path for probe scoring + calibration.

    Resolution order (spec §6.2):
      1. args.weights if given
      2. args.calibrate_weights (legacy fallback)
      3. Auto-discover model_iter_{max(meta.iteration) + 1:04d}.safetensors in:
         a. args.checkpoint_dir if given
         b. checkpoints/<single-subdir>/ if exactly one subdir exists
         c. current working directory
      4. Return None if nothing found.
    """
    import os

    # 1. Explicit --weights wins.
    explicit = getattr(args, "weights", None)
    if explicit:
        return explicit if os.path.exists(explicit) else None

    # 2. Legacy --calibrate-weights fallback.
    legacy = getattr(args, "calibrate_weights", None)
    if legacy:
        return legacy if os.path.exists(legacy) else None

    # 3. Auto-discover from replays.
    if not replays:
        return None
    iters = [r.get("meta", {}).get("iteration") for r in replays
             if isinstance(r.get("meta", {}).get("iteration"), int)]
    if not iters:
        return None
    max_iter = max(iters)
    target_name = f"model_iter_{max_iter + 1:04d}.safetensors"

    candidate_dirs: List[str] = []
    explicit_dir = getattr(args, "checkpoint_dir", None)
    if explicit_dir:
        candidate_dirs.append(explicit_dir)
    else:
        # Single-subdir convention.
        ckpt_root = Path("checkpoints")
        if ckpt_root.is_dir():
            subdirs = [p for p in ckpt_root.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                candidate_dirs.append(str(subdirs[0]))
        candidate_dirs.append(".")  # cwd fallback

    for d in candidate_dirs:
        full = os.path.join(d, target_name)
        if os.path.exists(full):
            return full
    return None
```

Ensure `from typing import Optional` is imported at the top of the file (check existing imports; add if absent).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py -k "resolve_checkpoint" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_replay_probe_scoring_end_to_end.py
git commit -m "feat(analyzer): add _resolve_checkpoint_path auto-discovery helper

Order: explicit --weights, legacy --calibrate-weights, auto-derive from
max(meta.iteration) in replays + single-subdir convention under
checkpoints/. Returns None when nothing found; downstream consumers
skip dependent sections.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Add new CLI flags to the analyzer

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`

- [ ] **Step 1: Locate the existing argparse block**

Run: `grep -n "ap.add_argument" scripts/twixt_replay_analyzer.py | tail -10`
Expected: lines showing the existing flags (`--probes`, `--calibrate`, `--calibrate-weights`, etc.) around line 1875–1940.

- [ ] **Step 2: Insert new flags**

In `scripts/twixt_replay_analyzer.py`, after the existing `--no-connectivity` flag (around line 1936), insert:

```python
    # Probes / calibration — spec §6.1 new flags.
    ap.add_argument("--weights", default=None,
                    help="Explicit checkpoint path for probe scoring + "
                         "calibration. Skips auto-discovery. When omitted, "
                         "the analyzer auto-discovers model_iter_{max+1}"
                         ".safetensors in --checkpoint-dir or "
                         "checkpoints/<single-subdir>/.")
    ap.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None,
                    help="Directory to search for auto-discovered checkpoint. "
                         "When omitted, uses checkpoints/<single-subdir>/ if "
                         "exactly one subdirectory exists under checkpoints/.")
    ap.add_argument("--probe-scoring-disable", dest="probe_scoring_disable",
                    action="store_true", default=False,
                    help="Skip replay_probe_scoring entirely.")
    ap.add_argument("--calibration-disable", dest="calibration_disable",
                    action="store_true", default=False,
                    help="Skip value_calibration entirely.")
    ap.add_argument("--calibration-samples-per-bucket",
                    dest="calibration_samples_per_bucket", type=int, default=200,
                    help="Target samples per phase-stratified bucket (spec §4.3).")
    ap.add_argument("--calibration-max-total", dest="calibration_max_total",
                    type=int, default=2000,
                    help="Safety cap on total calibration forward passes.")
```

- [ ] **Step 3: Update help text on legacy flags**

Find the existing `--calibrate` and `--calibrate-weights` flags (around line 1921–1927) and append a note. The line for `--calibrate-weights` becomes:

```python
    ap.add_argument("--calibrate-weights", dest="calibrate_weights", default=None,
                    help="Explicit weights path for --calibrate. Superseded "
                         "by --weights + auto-discovery; retained for "
                         "backwards compatibility with existing scripts.")
```

- [ ] **Step 4: Verify argparse accepts the flags**

Run: `.venv/bin/python scripts/twixt_replay_analyzer.py --help | grep -E "weights|checkpoint-dir|probe-scoring-disable|calibration-disable|calibration-samples-per-bucket|calibration-max-total"`
Expected: All six new flags listed in the help output.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "feat(analyzer): add CLI flags for weights / checkpoint-dir / probe-calibration opt-outs

All new flags optional. --calibrate and --calibrate-weights remain
accepted for backwards compatibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Wire replay probe scoring into analyzer main flow

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Modify: `tests/test_analyzer_replay_probe_scoring_end_to_end.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_analyzer_replay_probe_scoring_end_to_end.py`:

```python
def test_analyzer_emits_replay_probe_scoring(tmp_path):
    """End-to-end: analyzer with auto-discovered checkpoint populates
    summary['replay_probe_scoring'] with real counts."""
    import subprocess

    games_dir = tmp_path / "Replays" / "test_range"
    games_dir.mkdir(parents=True)
    for i in range(6):
        _write_fake_replay(games_dir, iteration=29, game_idx=i, winner="red")
        _write_fake_replay(games_dir, iteration=29, game_idx=100 + i, winner="black")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_fake_checkpoint(ckpt_dir, iteration=30, in_channels=30)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [".venv/bin/python", "scripts/twixt_replay_analyzer.py",
         "--input", str(games_dir),
         "--out", str(out_dir),
         "--checkpoint-dir", str(ckpt_dir),
         "--calibration-disable",  # focus this test on probe scoring
         "--no-plots"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    suffix = "test_range"  # derived from --out basename minus _Replay suffix
    summary_path = out_dir / f"summary_{suffix}.json"
    assert summary_path.exists(), f"summary not produced: {list(out_dir.iterdir())}"
    summary = json.loads(summary_path.read_text())

    rps = summary.get("replay_probe_scoring")
    assert rps is not None, "replay_probe_scoring missing from summary"
    assert rps["source"] == "replay_derived"
    assert rps["probe_count"] > 0
    assert rps["n"] == rps["probe_count"]
    assert 0.0 <= rps["sign_correct_pct"] <= 1.0
    assert "by_category" in rps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py::test_analyzer_emits_replay_probe_scoring -v`
Expected: FAIL — `replay_probe_scoring` not in summary.

- [ ] **Step 3: Wire the scoring call into the analyzer**

In `scripts/twixt_replay_analyzer.py`, locate the current `if _HAS_PHASE1_DIAG and calibrate:` block (around line 1422–1427). Replace it with:

```python
    # Spec §6.2: resolve checkpoint once, shared across probe scoring + calibration.
    resolved_weights = _resolve_checkpoint_path(args, replays)
    shared_network = None
    if resolved_weights is not None:
        try:
            from scripts.GPU.alphazero.probe_eval import load_network_for_scoring
            shared_network, _in_ch, _h, _nb = load_network_for_scoring(
                resolved_weights, verbose=False
            )
        except Exception as _e:
            print(f"[analyzer] WARNING: failed to load checkpoint {resolved_weights}: {_e}",
                  file=sys.stderr)
            resolved_weights = None
            shared_network = None

    # Spec §6.3: replay-derived probe scoring.
    replay_probe_scoring: dict = {}
    if (resolved_weights is not None
            and shared_network is not None
            and not args.probe_scoring_disable):
        from scripts.GPU.alphazero.probe_eval import (
            extract_forced_probes_from_games, run_forced_probes_inline,
        )
        probes_for_scoring = extract_forced_probes_from_games(
            replays,
            active_size=24,
            k_plies=2,
            winner_reasons=frozenset({"win"}),
            dedupe_exact=True,
            dedupe_mirror=True,
            max_probes=None,
        )
        if not probes_for_scoring:
            replay_probe_scoring = {
                "source": "replay_derived",
                "weights": os.path.abspath(resolved_weights),
                "probe_count": 0,
                "skipped_reason": "no_natural_wins",
            }
        else:
            scoring_result = run_forced_probes_inline(
                shared_network, probes_for_scoring, active_size=24
            )
            n = scoring_result["n"]
            sign_correct = scoring_result["sign_correct"]
            # Category breakdown.
            by_cat: dict[str, dict] = {}
            nn_values_iter = iter(scoring_result["nn_values"])
            exp_signs_iter = iter(scoring_result["expected_signs"])
            for p in probes_for_scoring:
                v = next(nn_values_iter)
                s = next(exp_signs_iter)
                correct = int((s > 0 and v > 0) or (s < 0 and v < 0)
                              or (s == 0 and abs(v) < 0.1))
                cat = p["category"]
                cat_bucket = by_cat.setdefault(cat, {
                    "n": 0, "sign_correct": 0, "abs_v_sum": 0.0,
                })
                cat_bucket["n"] += 1
                cat_bucket["sign_correct"] += correct
                cat_bucket["abs_v_sum"] += abs(v)
            by_category = {
                cat: {
                    "n": b["n"],
                    "sign_correct_pct": round(b["sign_correct"] / b["n"], 4)
                                       if b["n"] else None,
                    "median_abs_v": round(b["abs_v_sum"] / b["n"], 4)
                                   if b["n"] else None,
                }
                for cat, b in by_cat.items()
            }
            replay_probe_scoring = {
                "source": "replay_derived",
                "weights": os.path.abspath(resolved_weights),
                "checkpoint_in_channels": _in_ch,
                "selection_rules": {
                    "k_plies": 2,
                    "winner_reasons": ["win"],
                    "dedup": "exact + 4-form-mirror-canonical",
                },
                "probe_count": len(probes_for_scoring),
                "n": n,
                "sign_correct": sign_correct,
                "sign_correct_pct": scoring_result["sign_correct_pct"],
                "median_abs_v": scoring_result["median_abs_v"],
                "by_category": by_category,
            }
```

Also find where the summary dict is assembled (around line 1595–1597) and add the new field. Look for `"value_calibration": value_calibration_summary if calibrate else {},` and add before it:

```python
        "replay_probe_scoring": replay_probe_scoring,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py::test_analyzer_emits_replay_probe_scoring -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_replay_probe_scoring_end_to_end.py
git commit -m "feat(analyzer): wire replay_probe_scoring into summary output

Resolves checkpoint via _resolve_checkpoint_path, loads network once
via load_network_for_scoring, extracts probes via shared helper,
scores via run_forced_probes_inline. Adds by_category breakdown.
Emits replay_probe_scoring top-level key in summary_*.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Wire value calibration — replace the stub

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Modify: `tests/test_analyzer_replay_probe_scoring_end_to_end.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_analyzer_replay_probe_scoring_end_to_end.py`:

```python
def test_analyzer_emits_value_calibration(tmp_path):
    """Value calibration is populated (not stub) when checkpoint is available."""
    import subprocess

    games_dir = tmp_path / "Replays" / "test_range"
    games_dir.mkdir(parents=True)
    for i in range(8):
        _write_fake_replay(games_dir, iteration=29, game_idx=i, winner="red")
        _write_fake_replay(games_dir, iteration=29, game_idx=100 + i, winner="black")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_fake_checkpoint(ckpt_dir, iteration=30, in_channels=30)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [".venv/bin/python", "scripts/twixt_replay_analyzer.py",
         "--input", str(games_dir),
         "--out", str(out_dir),
         "--checkpoint-dir", str(ckpt_dir),
         "--probe-scoring-disable",
         "--calibration-samples-per-bucket", "5",
         "--calibration-max-total", "100",
         "--no-plots"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((out_dir / "summary_test_range.json").read_text())
    cal = summary.get("value_calibration", {})
    assert cal.get("stratified") is True
    assert "natural_distribution" in cal
    assert "sampled_distribution" in cal
    assert "aggregate" in cal
    # Not the old stub.
    assert cal.get("status") != "not_implemented"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py::test_analyzer_emits_value_calibration -v`
Expected: FAIL — `stratified` key missing; current stub returns `{}` when `--calibrate` not passed.

- [ ] **Step 3: Replace the calibration stub**

In `scripts/twixt_replay_analyzer.py`, find the block that currently contains:

```python
    if _HAS_PHASE1_DIAG and calibrate:
        value_calibration_summary = {
            "status": "not_implemented",
            "note": "Phase 1 scaffold — full scoring loop to be added in a follow-up",
            "weights": calibrate_weights,
        }
```

Replace with:

```python
    # Spec §6.4: real calibration scoring (replaces former stub).
    if (_HAS_PHASE1_DIAG
            and resolved_weights is not None
            and shared_network is not None
            and not args.calibration_disable):
        from scripts.GPU.alphazero.value_calibration import (
            score_samples_against_checkpoint,
        )
        try:
            cal = score_samples_against_checkpoint(
                replays,
                network=shared_network,
                samples_per_bucket=args.calibration_samples_per_bucket,
                max_total=args.calibration_max_total,
                min_size=args.winning_structure_min_size,
            )
            cal["weights"] = os.path.abspath(resolved_weights)
            value_calibration_summary = cal
        except Exception as _e:
            print(f"[analyzer] WARNING: calibration scoring failed: {_e}",
                  file=sys.stderr)
            value_calibration_summary = {}
```

Also find the summary assembly line `"value_calibration": value_calibration_summary if calibrate else {},` and change it to:

```python
        "value_calibration": value_calibration_summary,
```

(The `if calibrate` gate was the legacy trigger; the new gate lives inside the block above, based on resolved weights.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py::test_analyzer_emits_value_calibration -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_replay_probe_scoring_end_to_end.py
git commit -m "feat(analyzer): replace value_calibration stub with real scoring

Calls score_samples_against_checkpoint against the shared network
loaded by Task 14. Gates on resolved_weights + shared_network +
--calibration-disable. Removes the legacy 'if calibrate' gate; the
new trigger is successful checkpoint resolution.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Add new CSV outputs

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Modify: `tests/test_analyzer_replay_probe_scoring_end_to_end.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analyzer_replay_probe_scoring_end_to_end.py`:

```python
def test_analyzer_emits_replay_probe_per_probe_csv(tmp_path):
    """A CSV per-probe is written alongside summary."""
    import subprocess, csv

    games_dir = tmp_path / "Replays" / "test_range"
    games_dir.mkdir(parents=True)
    for i in range(4):
        _write_fake_replay(games_dir, iteration=29, game_idx=i, winner="red")
        _write_fake_replay(games_dir, iteration=29, game_idx=100 + i, winner="black")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_fake_checkpoint(ckpt_dir, iteration=30)

    out_dir = tmp_path / "out"
    subprocess.run(
        [".venv/bin/python", "scripts/twixt_replay_analyzer.py",
         "--input", str(games_dir),
         "--out", str(out_dir),
         "--checkpoint-dir", str(ckpt_dir),
         "--calibration-disable",
         "--no-plots"],
        capture_output=True, text=True, check=True,
    )
    csv_path = out_dir / "replay_probe_per_probe_test_range.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) > 0
    expected = {"id", "category", "source_game", "source_ply",
                "expected_value_sign", "nn_value", "sign_correct", "nn_magnitude"}
    assert expected.issubset(set(rows[0].keys()))


def test_analyzer_emits_value_calibration_by_bucket_csv(tmp_path):
    """A per-bucket CSV is written for calibration."""
    import subprocess, csv

    games_dir = tmp_path / "Replays" / "test_range"
    games_dir.mkdir(parents=True)
    for i in range(6):
        _write_fake_replay(games_dir, iteration=29, game_idx=i, winner="red")
        _write_fake_replay(games_dir, iteration=29, game_idx=100 + i, winner="black")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_fake_checkpoint(ckpt_dir, iteration=30)

    out_dir = tmp_path / "out"
    subprocess.run(
        [".venv/bin/python", "scripts/twixt_replay_analyzer.py",
         "--input", str(games_dir),
         "--out", str(out_dir),
         "--checkpoint-dir", str(ckpt_dir),
         "--probe-scoring-disable",
         "--calibration-samples-per-bucket", "5",
         "--calibration-max-total", "100",
         "--no-plots"],
        capture_output=True, text=True, check=True,
    )
    csv_path = out_dir / "value_calibration_by_bucket_test_range.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    expected = {"bucket", "natural_count", "sampled_count", "sign_agree",
                "mse", "pred_mean", "outcome_mean"}
    assert expected.issubset(set(rows[0].keys()))
    # Should have at least one row per represented bucket.
    assert len(rows) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py -k "csv" -v`
Expected: FAIL — CSV files not produced.

- [ ] **Step 3: Add CSV writers**

In `scripts/twixt_replay_analyzer.py`, find where existing CSVs are written (search for `write_replay_cap_by_iter_csv` calls, or the `_suffixed(...)` pattern around line 1700–1800). Add two new writer functions in an appropriate location (near other CSV writers), and call-sites:

```python
def write_replay_probe_per_probe_csv(
    out_dir: str, suffix: str, probes: list, scoring_result: dict,
) -> None:
    """Emit replay_probe_per_probe_<suffix>.csv (one row per probe)."""
    import csv
    path = os.path.join(out_dir, _suffixed("replay_probe_per_probe", "csv", suffix))
    nn_values = scoring_result.get("nn_values") or []
    expected_signs = scoring_result.get("expected_signs") or []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "category", "source_game", "source_ply",
            "expected_value_sign", "nn_value", "sign_correct", "nn_magnitude",
        ])
        w.writeheader()
        for p, v, s in zip(probes, nn_values, expected_signs):
            correct = int((s > 0 and v > 0) or (s < 0 and v < 0)
                          or (s == 0 and abs(v) < 0.1))
            w.writerow({
                "id": p["id"],
                "category": p["category"],
                "source_game": p["source_game"],
                "source_ply": p["source_ply"],
                "expected_value_sign": s,
                "nn_value": round(v, 4),
                "sign_correct": correct,
                "nn_magnitude": round(abs(v), 4),
            })


def write_value_calibration_by_bucket_csv(
    out_dir: str, suffix: str, cal_summary: dict,
) -> None:
    """Emit value_calibration_by_bucket_<suffix>.csv (one row per bucket)."""
    import csv
    path = os.path.join(out_dir, _suffixed("value_calibration_by_bucket", "csv", suffix))
    natural = cal_summary.get("natural_distribution") or {}
    sampled = cal_summary.get("sampled_distribution") or {}
    buckets_stats = (cal_summary.get("aggregate") or {}).get("buckets") or {}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bucket", "natural_count", "sampled_count",
            "sign_agree", "mse", "pred_mean", "outcome_mean",
        ])
        w.writeheader()
        for bucket in sorted(natural.keys()):
            stats = buckets_stats.get(bucket, {})
            w.writerow({
                "bucket": bucket,
                "natural_count": natural.get(bucket, 0),
                "sampled_count": sampled.get(bucket, 0),
                "sign_agree": stats.get("sign_agree", ""),
                "mse": stats.get("mse", ""),
                "pred_mean": stats.get("pred_mean", ""),
                "outcome_mean": stats.get("outcome_mean", ""),
            })
```

Then add call-sites right after the existing CSV writer calls (search for `write_replay_cap_by_iter_csv(` and add these after):

```python
    # Spec §6.5: new CSVs for replay probe scoring + value calibration.
    if replay_probe_scoring and replay_probe_scoring.get("probe_count", 0) > 0:
        write_replay_probe_per_probe_csv(
            out_dir, out_suffix,
            probes=probes_for_scoring,
            scoring_result=scoring_result,
        )
    if value_calibration_summary:
        write_value_calibration_by_bucket_csv(
            out_dir, out_suffix, value_calibration_summary
        )
```

**Note:** `probes_for_scoring` and `scoring_result` must be in scope at the call-site. Ensure both are named in the scoring block from Task 14 so they survive out of the conditional (initialize them to empty / empty-dict defaults at module-flow level before the conditional block, so the call-sites below can reference them unconditionally).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py -k "csv" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_replay_probe_scoring_end_to_end.py
git commit -m "feat(analyzer): emit per-probe and per-bucket calibration CSVs

replay_probe_per_probe_<suffix>.csv: one row per scored probe with
nn_value, sign_correct, nn_magnitude.
value_calibration_by_bucket_<suffix>.csv: one row per bucket with
natural_count, sampled_count, sign_agree, mse, pred_mean, outcome_mean.
Suffix follows the existing _suffixed() naming pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: Add report formatters for new sections

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Modify: `tests/test_analyzer_replay_probe_scoring_end_to_end.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_analyzer_replay_probe_scoring_end_to_end.py`:

```python
def test_analyzer_report_contains_new_sections(tmp_path):
    """report_<suffix>.txt contains populated (not '(not available)') sections
    for both replay_probe_scoring and value_calibration."""
    import subprocess

    games_dir = tmp_path / "Replays" / "test_range"
    games_dir.mkdir(parents=True)
    for i in range(6):
        _write_fake_replay(games_dir, iteration=29, game_idx=i, winner="red")
        _write_fake_replay(games_dir, iteration=29, game_idx=100 + i, winner="black")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_fake_checkpoint(ckpt_dir, iteration=30)

    out_dir = tmp_path / "out"
    subprocess.run(
        [".venv/bin/python", "scripts/twixt_replay_analyzer.py",
         "--input", str(games_dir),
         "--out", str(out_dir),
         "--checkpoint-dir", str(ckpt_dir),
         "--calibration-samples-per-bucket", "5",
         "--calibration-max-total", "100",
         "--no-plots"],
        capture_output=True, text=True, check=True,
    )
    report = (out_dir / "report_test_range.txt").read_text()
    assert "Replay-Derived Probe Scoring" in report
    assert "Value Head Calibration by Position Type" in report
    # New report must NOT contain the old "(not available)" placeholders for these.
    rps_section_idx = report.find("Replay-Derived Probe Scoring")
    cal_section_idx = report.find("Value Head Calibration by Position Type")
    assert "(not available" not in report[rps_section_idx:rps_section_idx + 500]
    assert "(not available" not in report[cal_section_idx:cal_section_idx + 500]
    # Stratified disclaimer present in calibration header.
    assert "stratified" in report[cal_section_idx:cal_section_idx + 600].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py::test_analyzer_report_contains_new_sections -v`
Expected: FAIL — report lacks "Replay-Derived Probe Scoring" section; calibration section still shows stub text.

- [ ] **Step 3: Update report formatters**

In `scripts/twixt_replay_analyzer.py`, find `format_value_calibration_report` (around line 907). Replace it:

```python
def format_value_calibration_report(summary: dict) -> List[str]:
    """Render the value-calibration section for the text report."""
    lines = []
    lines.append("Value Head Calibration by Position Type (Phase 1)")
    lines.append("=" * 50)
    if not summary:
        lines.append("  (not available — pass --weights <path> or place checkpoint")
        lines.append("   under checkpoints/<subdir>/ matching max(meta.iteration)+1)")
        lines.append("")
        return lines
    lines.append(f"  Weights: {summary.get('weights', '?')}")
    lines.append(f"  Stratified: True (per-bucket target N={summary.get('samples_per_bucket_target', '?')})")
    lines.append("  NOTE: per-bucket calibration is phase-stratified; 'overall' row is a")
    lines.append("        stratified aggregate, NOT population-weighted.")
    lines.append("")
    lines.append("  Natural vs. sampled distribution:")
    natural = summary.get("natural_distribution") or {}
    sampled = summary.get("sampled_distribution") or {}
    lines.append(f"    {'bucket':<34}{'natural':>10}{'sampled':>10}")
    for bucket in sorted(natural.keys()):
        lines.append(f"    {bucket:<34}{natural[bucket]:>10}{sampled.get(bucket, 0):>10}")
    lines.append("")
    lines.append("  Per-bucket stats:")
    buckets_stats = (summary.get("aggregate") or {}).get("buckets") or {}
    lines.append(f"    {'bucket':<34}{'n':>6}{'sign_agree':>12}{'mse':>10}")
    for bucket in sorted(buckets_stats.keys()):
        s = buckets_stats[bucket]
        n = s.get("n", 0)
        sa = s.get("sign_agree", "")
        mse = s.get("mse", "")
        lines.append(f"    {bucket:<34}{n:>6}{sa!s:>12}{mse!s:>10}")
    lines.append("")
    return lines


def format_replay_probe_scoring_report(summary: dict) -> List[str]:
    """Render the replay_probe_scoring section for the text report."""
    lines = []
    lines.append("Replay-Derived Probe Scoring (end-of-chunk snapshot)")
    lines.append("=" * 50)
    if not summary:
        lines.append("  (not available — pass --weights <path> or place checkpoint")
        lines.append("   under checkpoints/<subdir>/ matching max(meta.iteration)+1)")
        lines.append("")
        return lines
    if summary.get("probe_count", 0) == 0:
        lines.append(f"  (no probes extracted — {summary.get('skipped_reason', 'unknown')})")
        lines.append("")
        return lines
    lines.append(f"  Source: {summary.get('source', '?')} (NOT spec §7 curated gate suite)")
    lines.append(f"  Weights: {summary.get('weights', '?')}")
    lines.append(f"  Checkpoint in_channels: {summary.get('checkpoint_in_channels', '?')}")
    lines.append(f"  Probe count: {summary.get('probe_count', 0)}")
    n = summary.get("n", 0)
    sc = summary.get("sign_correct", 0)
    sc_pct = summary.get("sign_correct_pct", 0.0)
    mv = summary.get("median_abs_v", None)
    mv_s = f"{mv:.3f}" if mv is not None else "n/a"
    lines.append(f"  Overall: sign_correct={sc}/{n} ({sc_pct:.1%}), median |v|={mv_s}")
    lines.append("")
    lines.append("  By category:")
    for cat in sorted((summary.get("by_category") or {}).keys()):
        c = summary["by_category"][cat]
        pct = c.get("sign_correct_pct") or 0.0
        lines.append(f"    {cat:<20} n={c.get('n',0):>5}  sign_correct={pct:.1%}  median |v|={c.get('median_abs_v','?')}")
    lines.append("")
    return lines
```

Then find where the report lines are assembled (search for `format_value_calibration_report(` invocation around line 1855). Add a call to the new formatter right after it:

```python
    if _HAS_PHASE1_DIAG:
        lines.extend(format_value_calibration_report(value_calibration_summary))
        lines.extend(format_replay_probe_scoring_report(replay_probe_scoring))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py::test_analyzer_report_contains_new_sections -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_replay_probe_scoring_end_to_end.py
git commit -m "feat(analyzer): populated report sections for replay_probe_scoring + value_calibration

Replaces the '(not available)' placeholders with real populated
sections. Calibration section flags stratified origin and emits the
natural-vs-sampled distribution table alongside per-bucket stats.
Probe section emits per-category breakdown and labels the source as
replay-derived (NOT spec §7 gate suite).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: Commit-3 end-to-end sanity check

**Files:** (verification only)

- [ ] **Step 1: Run the full analyzer test suite**

Run: `pytest tests/test_analyzer_replay_probe_scoring_end_to_end.py -v`
Expected: All Tasks 12–17 tests PASS.

- [ ] **Step 2: Run broader regression check**

Run: `pytest -x --ignore=tests/test_trainer_forced_probe_live.py`
Expected: No regressions (integration test file doesn't exist yet — that's Task 19).

- [ ] **Step 3: Run the analyzer against real replays (if available)**

Manual verification, not a test assertion:
```bash
.venv/bin/python scripts/twixt_replay_analyzer.py \
    --input Replays/21-30 \
    --out /tmp/verify-21-30 \
    --no-plots
```

Expected:
- Runs to completion (exit 0)
- `/tmp/verify-21-30/summary_21-30.json` contains populated `replay_probe_scoring` and `value_calibration` blocks
- Neither contains `{"status": "not_implemented"}`
- Report text contains both new sections

If the real-data run surfaces any issue, fix it in a new commit before proceeding to Commit 4.

- [ ] **Step 4: Skip commit if step 1-3 clean; otherwise commit fixes**

Task 18 is verification only — no commit required if everything passes.

---

## Commit 4 — Integration test + pytest marker

### Task 19: Add `@pytest.mark.integration` test for live trainer path

**Files:**
- Create: `tests/test_trainer_forced_probe_live.py`

- [ ] **Step 1: Write the test**

Create `tests/test_trainer_forced_probe_live.py`:

```python
"""Live integration test for the trainer's forced-probe inline path.

Marked @pytest.mark.integration. Opt-in locally (pytest -m integration);
required in CI.

Runs 2 minimal training iterations against the committed bootstrap
probes and asserts:
  - forced_probe_summary lands in each iter's sidecar with n > 0
  - rolling-5 math populates correctly on iter 2 (None on iter 1)
  - sanity_by_connectivity is dict with winning/no_winning keys
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
def test_trainer_writes_forced_probe_summary_to_sidecar(tmp_path):
    """Full train() call with minimal config, 2 iterations, real probes file."""
    # Locate the most recent canonical checkpoint to resume from.
    ckpt_root = Path("checkpoints")
    if not ckpt_root.is_dir():
        pytest.skip("no checkpoints/ directory — cannot exercise live trainer path")
    # Pick the newest non-partial safetensors under any subdir.
    candidates = sorted(ckpt_root.glob("*/model_iter_*.safetensors"))
    candidates = [c for c in candidates if "_partial" not in c.name]
    if not candidates:
        pytest.skip("no canonical checkpoint available to resume from")
    resume = candidates[-1]
    # Extract the absolute iter number from the filename.
    iter_num = int(resume.stem.split("_")[-1])
    target_iter = iter_num + 2

    # Verify the committed probes file is present.
    probes_path = Path("tests/probes/twixt_probes.json")
    assert probes_path.exists(), (
        "Bootstrap probes file missing — Task 10 must land before this test. "
        "Run scripts/build_bootstrap_probe_suite.py --source-iter-range ..."
    )

    # Run 2 minimal iterations.
    import subprocess
    result = subprocess.run(
        [
            ".venv/bin/python", "-m", "scripts.GPU.alphazero.train",
            "--resume", str(resume),
            "--iterations", str(target_iter),
            "--games-per-iter", "2",
            "--simulations", "20",
            "--n-workers", "1",
            "--mcts-eval-batch-size", "2",
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--probes-path", str(probes_path),
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"train() failed:\n{result.stderr[-2000:]}"

    # Locate the written per-iter stats sidecars.
    games_dir = Path("scripts/GPU/logs/games")
    sidecar_1 = games_dir / f"iter_{iter_num:04d}_stats.json"
    sidecar_2 = games_dir / f"iter_{iter_num + 1:04d}_stats.json"
    assert sidecar_1.exists(), f"iter {iter_num} sidecar missing"
    assert sidecar_2.exists(), f"iter {iter_num + 1} sidecar missing"

    s1 = json.loads(sidecar_1.read_text())
    s2 = json.loads(sidecar_2.read_text())

    # Assertion 1: forced_probe_summary populated with n > 0 on both iters.
    fps1 = s1.get("forced_probe_summary")
    fps2 = s2.get("forced_probe_summary")
    assert fps1 is not None and fps1.get("n", 0) > 0, f"iter {iter_num}: {fps1}"
    assert fps2 is not None and fps2.get("n", 0) > 0, f"iter {iter_num+1}: {fps2}"

    # Assertion 2: rolling-5 / delta math — None on iter 1, float on iter 2.
    assert fps1.get("rolling5_sign_correct_pct") is None, \
        f"iter 1 rolling5 should be None, got {fps1.get('rolling5_sign_correct_pct')}"
    assert fps1.get("delta_sign_correct_pct") is None, \
        f"iter 1 delta should be None, got {fps1.get('delta_sign_correct_pct')}"
    assert isinstance(fps2.get("rolling5_sign_correct_pct"), float), \
        f"iter 2 rolling5 should be float, got {type(fps2.get('rolling5_sign_correct_pct'))}"
    assert isinstance(fps2.get("delta_sign_correct_pct"), float), \
        f"iter 2 delta should be float, got {type(fps2.get('delta_sign_correct_pct'))}"

    # Assertion 3: sanity_by_connectivity structure.
    sbc1 = s1.get("sanity_by_connectivity")
    assert isinstance(sbc1, dict), f"sbc missing or wrong type: {type(sbc1)}"
    assert "winning_structure" in sbc1 or "no_winning_structure" in sbc1, \
        f"sbc keys unexpected: {list(sbc1.keys())}"
```

- [ ] **Step 2: Try running without the integration marker opt-in**

Run: `pytest tests/test_trainer_forced_probe_live.py -v`
Expected: Test is collected but skipped with a "deselected" or marker message (exact wording depends on pytest config — see Task 20 where we make the opt-in explicit).

- [ ] **Step 3: Try running with the integration marker**

Run: `pytest -m integration tests/test_trainer_forced_probe_live.py -v`
Expected: PASS (takes ~30–90s; actually exercises the live trainer).

- [ ] **Step 4: Commit**

```bash
git add tests/test_trainer_forced_probe_live.py
git commit -m "test(trainer): add integration test for live forced_probe path

Runs 2 minimal training iterations (games_per_iter=2, sims=20, n_workers=1)
against the committed bootstrap probes. Asserts forced_probe_summary
populates with n>0, rolling-5 / delta math transitions correctly from
None (iter 1) to float (iter 2), and sanity_by_connectivity is a dict
with the expected keys.

Marked @pytest.mark.integration. Opt-in locally via 'pytest -m
integration'. CI enables the marker as a required merge gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 20: Configure `integration` marker in `pytest.ini`

**Files:**
- Modify: `pytest.ini`

- [ ] **Step 1: Inspect current pytest.ini**

Run: `cat pytest.ini`
Expected: existing markers (`slow, oracle, bridge, sealed_lane, phase1, phase2`) but no `integration` marker yet, and no default filter.

- [ ] **Step 2: Add the integration marker + exclude by default**

Edit `pytest.ini` — in the `[pytest]` section:

- Add to the `markers` list:
  ```
  integration: marks tests as integration (opt-in via -m integration; excluded by default)
  ```

- Modify `addopts` to exclude integration tests by default:
  ```
  addopts = -v --tb=short -m "not integration"
  ```

Final `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short -m "not integration"
filterwarnings =
    ignore::DeprecationWarning
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    oracle: marks tests that require Node.js for JS oracle
    bridge: marks bridge crossing tests
    sealed_lane: marks sealed lane tests
    phase1: marks Phase 1 geometry tests
    phase2: marks Phase 2 bitmask tests
    integration: marks tests as integration (opt-in via -m integration; excluded by default)
```

- [ ] **Step 3: Verify default run excludes integration**

Run: `pytest tests/test_trainer_forced_probe_live.py`
Expected: `1 test deselected` — integration test skipped by default filter.

- [ ] **Step 4: Verify explicit opt-in still works**

Run: `pytest -m integration tests/test_trainer_forced_probe_live.py`
Expected: Test runs (PASS assuming all previous commits land correctly).

- [ ] **Step 5: Verify the broader default run still includes non-integration tests**

Run: `pytest`
Expected: All tests from Tasks 1–18 run and PASS; the one integration test from Task 19 is deselected.

- [ ] **Step 6: Commit**

```bash
git add pytest.ini
git commit -m "chore(pytest): add integration marker + default exclusion

Default 'pytest' invocation skips @pytest.mark.integration tests to
keep local iteration fast. Opt in explicitly via 'pytest -m
integration'. CI config enables the marker as a merge gate (per spec
§7.2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

### 1. Spec coverage

Mapping spec sections to tasks:

| Spec section | Task(s) |
|---|---|
| §1 Problem (both gaps) | Closed by §§1–11 collectively |
| §2 Approach (stable trainer + dynamic analyzer) | Task 10 (probes file drops in → trainer path lights up), Tasks 14-15 (analyzer dynamic path) |
| §3.1 Key invariants (no training-behavior changes, workflow unchanged, distinct keys) | Task 10 does not modify trainer code; Tasks 13–15 preserve workflow; §14 + §15 emit distinct top-level keys |
| §4.1 `extract_forced_probes_from_games` (full schema) | Tasks 2–5 (winner-based category, deterministic IDs, filters, dedup, sort/truncate) |
| §4.2 `load_network_for_scoring` | Task 1 |
| §4.3 `score_samples_against_checkpoint` (stratified, natural distribution reported) | Task 6 |
| §5 Bootstrap generator (CLI, filters, balance, serialize, byte-identical) | Tasks 8, 9, 10 |
| §6.1 New CLI flags | Task 13 |
| §6.2 Checkpoint auto-discovery | Task 12 |
| §6.3 replay_probe_scoring schema | Task 14 |
| §6.4 value_calibration (replaces stub) | Task 15 |
| §6.5 New CSVs | Task 16 |
| §6.6 New report sections | Task 17 |
| §6.7 Graceful degradation | Task 14 (empty probe set → skipped_reason), Task 15 (exception → empty dict), Task 12 (not-found → None) |
| §7.1 Unit tests | Tasks 1–7 (probe_eval + value_calibration), Tasks 8–11 (bootstrap generator) |
| §7.2 Integration test + marker policy | Tasks 19, 20 |
| §7.3 End-to-end analyzer test | Tasks 12, 14, 15, 16, 17 each add to `test_analyzer_replay_probe_scoring_end_to_end.py`; Task 18 runs the full suite |
| §7.4 Post-merge operational smoke | Documented in this plan's outer scope; no task needed (manual operation) |
| §8 Rollout (4 commits, strict order) | Tasks grouped into 4 commit clusters matching §8.3 ordering |
| §8.2 Documentation updates | Task 11 (README), inline docstrings in Tasks 1, 6, 8, 12 |

All spec sections covered.

### 2. Placeholder scan

- No "TBD", "TODO", "implement later" strings
- Every step contains the actual code/command
- Reference to earlier tasks for type consistency is via concrete function names, not vague "similar to Task N"
- Post-merge smoke (spec §7.4) is referenced but is explicitly operational, not a plan task — that's consistent with the spec's framing

### 3. Type consistency

Cross-task name checks:
- `extract_forced_probes_from_games` — signature consistent across Tasks 2, 4, 5, 8, 9, 14
- `load_network_for_scoring` — signature consistent across Tasks 1, 14
- `score_samples_against_checkpoint` — signature consistent across Tasks 6, 15
- `_resolve_checkpoint_path` — signature `(args, replays)` consistent across Tasks 12, 14, 15
- Output schema keys: `forced_probe_summary` (trainer, unchanged), `replay_probe_scoring` (Task 14), `value_calibration` (Task 15) — distinct top-level keys
- `sign_correct` (not `sign_correct_nn`) used consistently in per-probe CSV (Task 16) and aggregate JSON (Task 14) — matches spec clarification
- Probe ID format `{basename}_ply{ply:03d}_{winner}` consistent across Tasks 2, 9

No type or name mismatches identified.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-21-probes-and-calibration-closure.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
