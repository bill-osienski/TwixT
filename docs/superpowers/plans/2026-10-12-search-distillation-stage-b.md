# Search Distillation Stage B — Lean Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage B pilot's pure, CPU-only decision pieces first. These are the split, position selection, the state digest, leakage filtering, full legal support, TV/KL, gate Q, the L1–L3 verdict, and seed exposure. Each piece is checked against hand-computed fixtures. After that, review the minimum label and training wiring before any of it is written.

**Architecture:** There is one new module of pure functions, `scripts/GPU/alphazero/search_distillation_stage_b.py`. It has no model, no search, no file I/O and no seed literal. One test file goes with it. Later phase scripts (P, L1, L2, T, analysis, closing) will import it. Search, loss, head training, batching and the surface check are reused from existing modules, not rewritten.

**Tech Stack:** Python 3.14 stdlib (`random`, `hashlib`, `json`, `math`, `dataclasses`), the engine's `TwixtState`, and pytest. No MLX in checkpoint 1.

**Spec:** `docs/superpowers/2026-10-12-search-distillation-stage-b-card.md` (rev 2, committed `a9e16fc9`). Read §4–§9 before starting.

## Global Constraints

- Approved so far: this plan only. **Do not** do any of the following without separate approval:
  - seed accounting;
  - model loading;
  - generated games;
  - labels;
  - training.
- Checkpoint 1 is CPU-only. Tests construct fixtures in memory and touch no checkpoint, no evaluator and no file outside `tmp_path`.
- No seed-band literal (a nine-digit number starting `2026`) in `scripts/`, `tests/` or `docs/superpowers/`. Fixtures use the synthetic base `1000`. Check with the pinned scanner:
  `docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py` `S.scan`.
- Frozen constants from the card:
  - 200 games;
  - held out iff `g % 5 == 4`;
  - 4 positions per game from plies `1 … n−1`;
  - sign-test threshold 26 of 40;
  - Q2 ratio ≤ 0.5;
  - slots `[0,200)` game, `[200,400)` label, `[400,440)` repeat, `440` training; 441 in all.
- Keep the capture work out of every commit:
  - `self_play.py` modification;
  - `training_replay_capture.py`;
  - the capture spec and plan;
  - the three capture test files.
- Commit only when the user approves. Never push.
- Test command, used everywhere below:
  `timeout 60 env PYTHONPATH=. .venv/bin/python -B -m pytest tests/test_search_distillation_stage_b.py -q -p no:cacheprovider`

## Review Focus

- **A held-out game emptied by leakage.** It must count as a loss in Q1, L1 and L2, add nothing to Q2's sums, and give **0** top-1 agreement for every arm. Pinned in Task 3 and Task 4.
- **A zero-visit legal move dropped from the support.** It changes KL. With target `[3,1,0]` and uniform logits, KL is 0.536277 with the zero-visit move kept and 0.130812 with it dropped. Pinned in Task 3.
- **A rebuilt state missing the 280-ply cap.** `TwixtState.from_moves` sets no `max_plies_limit`, while the corpus games had 280. Searching or hashing the wrong state is silent. `replay_state` builds the state as `play_eval_game` does. Pinned in Task 1.
- **Exposure counted as a prefix.** A run that stops after phase P exposes every game and label slot it used, and no repeat or training slot. Pinned in Task 5.
- **Exact boundaries.** 26 of 40 passes and 25 fails. A ratio of exactly 0.5 passes. Ties are losses. Equal top-1 agreement is not "strictly above". Pinned in Task 4.

---

# Checkpoint 1 — CPU-only pure pieces (Tasks 1–5), then STOP for review

**Execution as approved (2026-10-12).** The per-task red and green steps below show the intended order of the
tests and the code. The approved execution is shorter:
1. Write the complete test suite from Tasks 1–5.
2. Run it **once red** with the module absent.
3. Write the module.
4. Run it **once green**.

Both runs use the test command under `timeout 60`, and both exit statuses are captured. Any unexpected failure
stops for review: no extra runs. Afterward the module and tests stay uncommitted.

### Task 1: Split, state rebuild and the complete state digest

**Files:**
- Create: `scripts/GPU/alphazero/search_distillation_stage_b.py`
- Create: `tests/test_search_distillation_stage_b.py`

**Interfaces:**
- Produces:
  - `is_heldout(g: int) -> bool`
  - `repeat_slot(g: int) -> int`
  - `replay_state(moves) -> TwixtState`
  - `state_digest(s: TwixtState) -> str`
  - `_sha(obj) -> str`
  - constants `N_GAMES, PER_GAME, N_HELDOUT, SIGN_K, Q2_MAX, GAME, LABEL, REPEAT, TRAIN, N_SLOTS, STATE_FIELDS, ARMS`

- [ ] **Step 1: Write the failing tests**

```python
"""Stage B pure pieces (card docs/superpowers/2026-10-12-search-distillation-stage-b-card.md). CPU-only fixtures."""
import dataclasses
import math
import random

import pytest

from scripts.GPU.alphazero import search_distillation_stage_b as B
from scripts.GPU.alphazero.game.twixt_state import TwixtState

MOVES = [(5, 5), (10, 3), (7, 6), (12, 8), (9, 7), (3, 14), (11, 8), (14, 10), (13, 9), (16, 11)]


def test_split_is_every_fifth_game_and_repeat_slots_cover_400_to_439():
    held = [g for g in range(200) if B.is_heldout(g)]
    assert len(held) == 40 and held[0] == 4 and held[-1] == 199
    assert [B.repeat_slot(g) for g in held] == list(range(400, 440))
    with pytest.raises(ValueError):
        B.repeat_slot(3)


def test_replay_state_carries_the_corpus_ply_cap():
    s = B.replay_state(MOVES[:3])
    assert (s.max_plies_limit, s.ply, s.active_size, s.to_move) == (280, 3, 24, "black")
    assert TwixtState.from_moves(MOVES[:3]).max_plies_limit is None      # why replay_state exists


def test_digest_covers_every_constructor_field():
    s = B.replay_state(MOVES)
    variants = [dataclasses.replace(s, to_move="red" if s.to_move == "black" else "black"),
                dataclasses.replace(s, ply=s.ply + 1),
                dataclasses.replace(s, max_plies_limit=None),
                dataclasses.replace(s, active_size=23),
                dataclasses.replace(s, board_size=25),
                dataclasses.replace(s, pegs={**s.pegs, (20, 20): "red"}),
                dataclasses.replace(s, bridges=set(list(s.bridges)[1:]))]
    assert s.bridges, "the fixture must have a bridge to remove"
    digests = {B.state_digest(v) for v in variants}
    assert len(digests) == 7 and B.state_digest(s) not in digests


def test_digest_ignores_insertion_order():
    s = B.replay_state(MOVES)
    t = dataclasses.replace(s, pegs=dict(reversed(list(s.pegs.items()))), bridges=set(reversed(list(s.bridges))))
    assert B.state_digest(t) == B.state_digest(s)


def test_digest_refuses_a_changed_field_set(monkeypatch):
    monkeypatch.setattr(B, "STATE_FIELDS", tuple(f for f in B.STATE_FIELDS if f != "bridges"))
    with pytest.raises(RuntimeError):
        B.state_digest(B.replay_state(MOVES[:2]))
```

- [ ] **Step 2: Run them to verify they fail**

Run the test command.
Expected: collection error, `ModuleNotFoundError: ... search_distillation_stage_b`.

- [ ] **Step 3: Write the minimal module**

```python
"""SEARCH DISTILLATION STAGE B -- the pilot's pure pieces: split, position selection, state digest, leakage,
full legal support, TV/KL, gate Q, the L1-L3 verdict and seed exposure. CPU-only: no model, no search, no
file I/O, no seed literal. Card: docs/superpowers/2026-10-12-search-distillation-stage-b-card.md (a9e16fc9).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .external_learning_stage3 import PROTOCOL
from .game.twixt_state import TwixtState

N_GAMES, PER_GAME, N_HELDOUT, SIGN_K, Q2_MAX = 200, 4, 40, 26, 0.5
GAME, LABEL, REPEAT, TRAIN, N_SLOTS = 0, 200, 400, 440, 441
STATE_FIELDS = ("active_size", "board_size", "bridges", "max_plies_limit", "pegs", "ply", "to_move")
ARMS = ("student", "control", "temperature", "incumbent")


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def is_heldout(g: int) -> bool:
    return g % 5 == 4


def repeat_slot(g: int) -> int:
    if not is_heldout(g):
        raise ValueError(f"game {g} is not held out: it has no repeat seed")
    return REPEAT + g // 5


def replay_state(moves: Sequence[Sequence[int]]) -> TwixtState:
    """The state a corpus game reached after `moves`, built as play_eval_game builds it -- with its ply cap."""
    s = TwixtState(active_size=PROTOCOL["board_size"], to_move="red", max_plies_limit=PROTOCOL["max_moves"])
    for r, c in moves:
        s = s.apply_move((int(r), int(c)))
    return s


def state_digest(s: TwixtState) -> str:
    """Card §4: every constructor field. Refuses if TwixtState's field set has changed."""
    names = tuple(sorted(f.name for f in dataclasses.fields(TwixtState) if f.init))
    if names != STATE_FIELDS:
        raise RuntimeError(f"TwixtState fields {names} are not the frozen {STATE_FIELDS}: the digest would be incomplete")
    return _sha({"board_size": s.board_size, "active_size": s.active_size, "to_move": s.to_move, "ply": s.ply,
                 "max_plies_limit": s.max_plies_limit,
                 "pegs": sorted([r, c, colour] for (r, c), colour in s.pegs.items()),
                 "bridges": sorted([r1, c1, r2, c2] for (r1, c1), (r2, c2) in s.bridges)})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the test command. Expected: 5 passed.
If `MOVES` makes no bridge, `test_digest_covers_every_constructor_field` fails on its own assertion. In that case replace `MOVES` with a sequence that does make a bridge (a red knight-move pair: red plays `(5,5)` and later `(7,6)`). Do not weaken the assertion.

### Task 2: Position selection (phase P) and the leakage filter

**Interfaces:**
- Consumes: `replay_state`, `state_digest` and `is_heldout` from Task 1.
- Produces:
  - `select_plies(n_moves: int, label_seed: int) -> List[int]`
  - `select_game(moves, label_seed) -> List[{"ply": int, "digest": str}]`
  - `leakage_filter(selected: Mapping[int, list]) -> (Dict[int, list], int)`, which returns the kept held-out positions per held-out game and the number dropped.

- [ ] **Step 1: Write the failing tests** (append to the test file)

```python
def test_select_plies_draws_up_to_four_from_1_to_n_minus_1():
    for seed in range(1000, 1050):
        plies = B.select_plies(30, seed)
        assert len(plies) == 4 and plies == sorted(plies) and all(1 <= p <= 29 for p in plies)
        assert plies == sorted(random.Random(seed).sample(range(1, 30), 4))
    assert B.select_plies(3, 1000) == [1, 2]
    assert B.select_plies(1, 1000) == [] and B.select_plies(0, 1000) == []


def test_select_game_hashes_the_capped_rebuilt_state():
    picked = B.select_game(MOVES, 1007)
    assert [p["ply"] for p in picked] == B.select_plies(len(MOVES), 1007)
    for p in picked:
        assert p["digest"] == B.state_digest(B.replay_state(MOVES[:p["ply"]]))


def _selection(shared=None):
    sel = {g: [{"ply": i + 1, "digest": f"g{g}p{i}"} for i in range(4)] for g in range(200)}
    for (g, i), digest in (shared or {}).items():
        sel[g][i] = {"ply": i + 1, "digest": digest}
    return sel


def test_leakage_drops_held_out_positions_seen_in_training_and_can_empty_a_game():
    shared = {(4, 0): "g0p0"}                                   # one held-out position also in training
    shared.update({(9, i): f"g1p{i}" for i in range(4)})       # held-out game 9 entirely in training
    shared[(14, 1)] = "g19p2"                                   # duplicate of ANOTHER held-out game: kept
    kept, dropped = B.leakage_filter(_selection(shared))
    assert sorted(kept) == [g for g in range(200) if g % 5 == 4]
    assert dropped == 5
    assert [p["digest"] for p in kept[4]] == ["g4p1", "g4p2", "g4p3"]
    assert kept[9] == []                                        # an EMPTY game
    assert len(kept[14]) == 4


def test_leakage_requires_all_200_games_selected():
    sel = _selection()
    del sel[123]
    with pytest.raises(ValueError):
        B.leakage_filter(sel)
```

- [ ] **Step 2: Run them to verify they fail**

Run the test command. Expected: `AttributeError: ... has no attribute 'select_plies'`.

- [ ] **Step 3: Implement** (append to the module)

```python
def select_plies(n_moves: int, label_seed: int) -> List[int]:
    """Card §4: up to PER_GAME plies from 1..n-1 (never the empty board, never the final state), sorted.
    The caller logs `select_start` for this game's label slot BEFORE calling (the draw uses L_g)."""
    if n_moves < 2:
        return []
    return sorted(random.Random(label_seed).sample(range(1, n_moves), min(PER_GAME, n_moves - 1)))


def select_game(moves: Sequence[Sequence[int]], label_seed: int) -> List[Dict[str, Any]]:
    return [{"ply": p, "digest": state_digest(replay_state(moves[:p]))} for p in select_plies(len(moves), label_seed)]


def leakage_filter(selected: Mapping[int, Sequence[Mapping[str, Any]]]) -> Tuple[Dict[int, List[Mapping[str, Any]]], int]:
    """Held-out positions whose digest occurs among TRAINING positions are dropped (they stay in training)."""
    if sorted(selected) != list(range(N_GAMES)):
        raise ValueError("the selection must cover all 200 games before leakage can be filtered")
    train = {p["digest"] for g, ps in selected.items() if not is_heldout(g) for p in ps}
    kept = {g: [p for p in ps if p["digest"] not in train] for g, ps in selected.items() if is_heldout(g)}
    return kept, sum(len(selected[g]) - len(kept[g]) for g in kept)
```

- [ ] **Step 4: Run the tests.** Expected: 9 passed.

### Task 3: Full legal support, TV/KL, top-1 and per-game measures

**Interfaces:**
- Produces:
  - `legal_digest(legal) -> str`
  - `sparse_label(visits: Mapping[(r,c), int], sims) -> List[[r,c,v]]`
  - `full_counts(state, sparse, sims, legal_sha) -> (legal, counts)`
  - `probs(counts)`, `softmax(logits)`, `tv(p, q)`, `kl(t, p)`, `top1(logits) -> int`, `argmax_set(counts) -> set`
  - `q_game(positions) -> Optional[(D̄, R̄)]`, where each position is `{"t1600", "t400", "repeat"}` counts over the full support
  - `game_measures(positions) -> Optional[{"kl": {arm: float}, "top1": {arm: float}}]`, where each position is `{"target": counts, "logits": {arm: [...]}}`
  - Both return `None` for an empty game.

- [ ] **Step 1: Write the failing tests**

```python
def test_full_counts_restores_every_legal_move_including_zero_visits():
    s = B.replay_state([(5, 5)])
    legal = [tuple(m) for m in s.legal_moves()]
    sha = B.legal_digest(legal)
    sparse = B.sparse_label({legal[0]: 3, legal[7]: 1, legal[9]: 0}, 4)
    assert sparse == sorted([[*legal[0], 3], [*legal[7], 1]])               # zeros are not stored
    got_legal, counts = B.full_counts(s, sparse, 4, sha)
    assert got_legal == legal and len(counts) == len(legal) == 527
    assert counts[0] == 3 and counts[7] == 1 and sum(counts) == 4 and counts.count(0) == 525


@pytest.mark.parametrize("sparse, sims, wrong_sha", [
    ([[5, 5, 4]], 4, False),                # an occupied square: not a legal move
    ([[1, 1, 2], [1, 1, 2]], 4, False),     # the same move twice
    ([[1, 1, 3]], 4, False),                # visits do not sum to sims
    ([[1, 1, 0], [1, 2, 4]], 4, False),     # a stored zero
    ([[1, 1, 4]], 4, True),                 # the rebuilt legal list is not the labelled one
])
def test_full_counts_refuses_a_label_that_does_not_fit_the_state(sparse, sims, wrong_sha):
    # (1, 1) and (1, 2) are legal for black after red (5, 5); row 0 is not, so it would refuse for the wrong reason
    s = B.replay_state([(5, 5)])
    sha = B.legal_digest([tuple(m) for m in s.legal_moves()])
    with pytest.raises(ValueError):
        B.full_counts(s, sparse, sims, ("0" if sha[0] != "0" else "1") + sha[1:] if wrong_sha else sha)


def test_sparse_label_refuses_a_visit_sum_other_than_sims():
    with pytest.raises(ValueError):
        B.sparse_label({(0, 1): 3}, 4)


def test_tv_and_kl_match_hand_values():
    assert B.tv([1, 0, 0], [0, 0, 1]) == 1.0
    assert B.tv([0.5, 0.5, 0], [0.25, 0.25, 0.5]) == 0.5
    assert math.isclose(B.kl([0.5, 0.5, 0], [0.25, 0.25, 0.5]), math.log(2), rel_tol=1e-12)


def test_kl_uses_the_full_support_a_zero_visit_move_still_takes_probability():
    t = B.probs([3, 1, 0])
    assert math.isclose(B.kl(t, B.softmax([0.0, 0.0, 0.0])), 0.5362771440493014, rel_tol=1e-12)
    assert math.isclose(B.kl(t[:2], B.softmax([0.0, 0.0])), 0.13081203594113697, rel_tol=1e-12)   # if dropped


def test_top1_breaks_ties_to_the_lowest_index_and_argmax_set_keeps_all_maxima():
    assert B.top1([1.0, 3.0, 3.0]) == 1
    assert B.argmax_set([2, 5, 5, 0]) == {1, 2}


def test_q_game_means_and_empty_game():
    pos = [{"t1600": [2, 2, 0], "t400": [1, 0, 0], "repeat": [2, 2, 0]},     # D 0.5, R 0
           {"t1600": [4, 0, 0], "t400": [4, 0, 0], "repeat": [0, 4, 0]}]     # D 0,   R 1
    assert B.q_game(pos) == (0.25, 0.5)
    assert B.q_game([]) is None


def test_game_measures_per_arm_and_empty_game():
    flat, sharp = [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]
    pos = [{"target": [0, 4, 0], "logits": {"student": sharp, "control": flat, "temperature": flat, "incumbent": flat}}]
    m = B.game_measures(pos)
    assert m["top1"] == {"student": 1.0, "control": 0.0, "temperature": 0.0, "incumbent": 0.0}
    assert math.isclose(m["kl"]["control"], math.log(3), rel_tol=1e-12)
    assert m["kl"]["student"] < m["kl"]["control"]
    assert B.game_measures([]) is None
```

- [ ] **Step 2: Run them to verify they fail.** Expected: `AttributeError: ... 'legal_digest'`.

- [ ] **Step 3: Implement**

```python
def legal_digest(legal: Sequence[Sequence[int]]) -> str:
    return _sha([[int(r), int(c)] for r, c in legal])


def sparse_label(visits: Mapping[Tuple[int, int], int], sims: int) -> List[List[int]]:
    """Card §5 storage: the nonzero (row, col, visits); the visits must sum to sims."""
    if sum(visits.values()) != sims:
        raise ValueError(f"visits sum to {sum(visits.values())}, not {sims}")
    return sorted([int(r), int(c), int(v)] for (r, c), v in visits.items() if v > 0)


def full_counts(state: TwixtState, sparse: Sequence[Sequence[int]], sims: int, legal_sha: str
                ) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Card §5: restore the FULL legal support (zero-visit moves included) from a sparse label."""
    legal = [tuple(m) for m in state.legal_moves()]
    if legal_digest(legal) != legal_sha:
        raise ValueError("the rebuilt legal list is not the one the label was made on")
    index = {m: i for i, m in enumerate(legal)}
    counts = [0] * len(legal)
    for r, c, v in sparse:
        i = index.get((r, c))
        if i is None or v <= 0 or counts[i]:
            raise ValueError(f"label entry {(r, c, v)} is not a single positive count on a legal move")
        counts[i] = v
    if sum(counts) != sims:
        raise ValueError(f"label visits sum to {sum(counts)}, not {sims}")
    return legal, counts


def probs(counts: Sequence[float]) -> List[float]:
    s = sum(counts)
    return [c / s for c in counts]


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    e = [math.exp(z - m) for z in logits]
    s = sum(e)
    return [x / s for x in e]


def tv(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("distributions over different supports")
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def kl(t: Sequence[float], p: Sequence[float]) -> float:
    if len(t) != len(p):
        raise ValueError("distributions over different supports")
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(t, p) if a > 0)


def top1(logits: Sequence[float]) -> int:
    """Argmax; ties to the lowest index = the lowest encoded move id (legal order is row-major)."""
    return max(range(len(logits)), key=lambda i: (logits[i], -i))


def argmax_set(counts: Sequence[float]) -> set:
    m = max(counts)
    return {i for i, c in enumerate(counts) if c == m}


def q_game(positions: Sequence[Mapping[str, Sequence[int]]]) -> Optional[Tuple[float, float]]:
    """Card §6 per held-out game: (mean D, mean R); None for an empty game."""
    if not positions:
        return None
    d = [tv(probs(p["t1600"]), probs(p["t400"])) for p in positions]
    r = [tv(probs(p["t1600"]), probs(p["repeat"])) for p in positions]
    return sum(d) / len(d), sum(r) / len(r)


def game_measures(positions: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
    """Card §8 per held-out game: each arm's mean KL(T1600 || softmax) and top-1 share; None for an empty game.
    Logits are over the full legal support; the temperature arm's logits arrive already scaled by beta."""
    if not positions:
        return None
    n = len(positions)
    return {"kl": {a: sum(kl(probs(p["target"]), softmax(p["logits"][a])) for p in positions) / n for a in ARMS},
            "top1": {a: sum(top1(p["logits"][a]) in argmax_set(p["target"]) for p in positions) / n for a in ARMS}}
```

- [ ] **Step 4: Run the tests.** Expected: 21 passed (9 + 12; the parametrized test counts 5).

### Task 4: Gate Q and the L1–L3 verdict, with exact boundaries

**Interfaces:**
- Consumes: the output of `q_game` and `game_measures`, one entry per held-out game (`None` = empty).
- Produces:
  - `gate_q(games: Sequence[Optional[(D̄, R̄)]]) -> dict`, with keys `q1_wins, sum_d, sum_r, ratio, failed, verdict`
  - `learning_verdict(games: Sequence[Optional[measures]]) -> dict`, with keys `l1_wins, l2_wins, top1, failed, verdict`

- [ ] **Step 1: Write the failing tests**

```python
WIN, TIE = (0.5, 0.25), (0.0, 0.0)      # TIE adds nothing to Q2's sums: the Q1 boundary is isolated (ratio 0.5)


def test_q_passes_at_26_of_40_and_fails_at_25():
    ok = B.gate_q([WIN] * 26 + [TIE] * 14)
    assert ok["q1_wins"] == 26 and ok["failed"] == [] and ok["verdict"] == "QUALIFIED"
    bad = B.gate_q([WIN] * 25 + [TIE] * 15)
    assert bad["failed"] == ["Q1"] and bad["verdict"] == "QUALIFICATION NON-PASS"


def test_q2_ratio_exactly_half_passes_and_just_above_fails():
    assert B.gate_q([WIN] * 40)["ratio"] == 0.5 and B.gate_q([WIN] * 40)["failed"] == []
    assert B.gate_q([(0.5, 0.25 + 2 ** -10)] * 40)["failed"] == ["Q2"]


def test_q_zero_total_d_and_ties_fail_both_conditions():
    r = B.gate_q([(0.0, 0.0)] * 40)
    assert r["ratio"] is None and r["failed"] == ["Q1", "Q2"]


def test_q_empty_games_fail_q1_and_are_left_out_of_q2_sums():
    r = B.gate_q([WIN] * 25 + [None] * 15)
    assert r["failed"] == ["Q1"] and r["sum_d"] == 12.5 and r["ratio"] == 0.5
    assert B.gate_q([None] * 40)["failed"] == ["Q1", "Q2"]


def test_q_requires_exactly_40_games():
    with pytest.raises(ValueError):
        B.gate_q([WIN] * 39)


def _g(s, c, t, i, top=(1.0, 0.0, 0.0, 0.0)):
    return {"kl": dict(zip(B.ARMS, (s, c, t, i))), "top1": dict(zip(B.ARMS, top))}


def test_learning_pass_at_26_and_non_pass_at_25():
    win, loss = _g(1.0, 2.0, 2.0, 2.0), _g(3.0, 2.0, 2.0, 2.0)
    assert B.learning_verdict([win] * 26 + [loss] * 14)["verdict"] == "PASS"
    r = B.learning_verdict([win] * 25 + [loss] * 15)
    assert r["verdict"] == "NON-PASS" and r["failed"] == ["L1", "L2"]


def test_l2_needs_the_student_below_both_incumbent_and_temperature():
    beats_temp_not_incumbent = _g(1.5, 2.0, 2.0, 1.0)
    r = B.learning_verdict([beats_temp_not_incumbent] * 40)
    assert r["l1_wins"] == 40 and r["l2_wins"] == 0 and r["failed"] == ["L2"]


def test_equal_kl_is_a_student_loss():
    r = B.learning_verdict([_g(2.0, 2.0, 2.0, 2.0)] * 40)
    assert r["l1_wins"] == 0 and r["l2_wins"] == 0


def test_l3_needs_strictly_higher_agreement_than_control_and_incumbent():
    tied_control = _g(1.0, 2.0, 2.0, 2.0, top=(0.5, 0.5, 0.0, 0.0))
    assert B.learning_verdict([tied_control] * 40)["failed"] == ["L3"]


def test_empty_games_lose_l1_l2_and_score_zero_top1_for_every_arm():
    win = _g(1.0, 2.0, 2.0, 2.0, top=(1.0, 1.0, 1.0, 1.0))
    r = B.learning_verdict([win] * 26 + [None] * 14)
    assert r["l1_wins"] == 26 and r["l2_wins"] == 26
    assert r["top1"] == {a: 26 / 40 for a in B.ARMS}           # empty games count 0 for EVERY arm
    assert r["failed"] == ["L3"]                                  # equal agreement is not strictly above
```

- [ ] **Step 2: Run them to verify they fail.** Expected: `AttributeError: ... 'gate_q'`.

- [ ] **Step 3: Implement**

```python
def _forty(games: Sequence[Any]) -> None:
    if len(games) != N_HELDOUT:
        raise ValueError(f"{len(games)} held-out games, not {N_HELDOUT}")


def gate_q(games: Sequence[Optional[Tuple[float, float]]]) -> Dict[str, Any]:
    """Card §6. Q1: R < D in >= 26 of 40 (empty games and ties fail). Q2: sum R / sum D <= 0.5 over non-empty
    games; a zero (or absent) sum D fails Q2."""
    _forty(games)
    present = [g for g in games if g is not None]
    wins = sum(1 for d, r in present if r < d)
    sum_d, sum_r = sum(d for d, _ in present), sum(r for _, r in present)
    ratio = sum_r / sum_d if sum_d > 0 else None
    failed = [name for name, ok in (("Q1", wins >= SIGN_K), ("Q2", ratio is not None and ratio <= Q2_MAX)) if not ok]
    return {"q1_wins": wins, "sum_d": sum_d, "sum_r": sum_r, "ratio": ratio, "failed": failed,
            "verdict": "QUALIFIED" if not failed else "QUALIFICATION NON-PASS"}


def learning_verdict(games: Sequence[Optional[Mapping[str, Mapping[str, float]]]]) -> Dict[str, Any]:
    """Card §8. L1: KL_s < KL_control; L2: KL_s < min(KL_incumbent, KL_temperature); each in >= 26 of 40, ties and
    empty games are losses. L3: student top-1 agreement (mean over all 40; empty = 0 for every arm) strictly
    above both the control's and the incumbent's."""
    _forty(games)
    present = [g["kl"] for g in games if g is not None]
    l1 = sum(1 for k in present if k["student"] < k["control"])
    l2 = sum(1 for k in present if k["student"] < min(k["incumbent"], k["temperature"]))
    top = {a: sum(g["top1"][a] for g in games if g is not None) / N_HELDOUT for a in ARMS}
    l3 = top["student"] > top["control"] and top["student"] > top["incumbent"]
    failed = [name for name, ok in (("L1", l1 >= SIGN_K), ("L2", l2 >= SIGN_K), ("L3", l3)) if not ok]
    return {"l1_wins": l1, "l2_wins": l2, "top1": top, "failed": failed,
            "verdict": "PASS" if not failed else "NON-PASS"}
```

- [ ] **Step 4: Run the tests.** Expected: 31 passed.

  Before trusting `test_empty_games_lose_l1_l2_and_score_zero_top1_for_every_arm`, check its float equality by hand: `sum` of 26 ones is exactly 26.0, so `26.0 / 40 == 26 / 40`.

### Task 5: Seed exposure as the union of used slots across phases

**Interfaces:**
- Consumes: `repeat_slot` and the slot constants.
- Produces:
  - `used_slots(events) -> List[int]`
  - `closing_intervals(base: int, events) -> (List[(lo, hi)], (lo, hi))`, returning the merged EXPOSED intervals and the RETIRED interval
- Event names, which the phase scripts log **before** each use:
  - `game_start {game}`
  - `select_start {game}`
  - `label_start {game}`
  - `repeat_start {game}`
  - `train_start`

- [ ] **Step 1: Write the failing tests**

```python
BASE = 1000
HELD = [g for g in range(200) if g % 5 == 4]


def ev(name, games):
    return [{"event": name, "game": g} for g in games]


def test_stop_after_position_selection_exposes_game_and_label_slots_only():
    events = ev("game_start", range(200)) + ev("select_start", range(200))
    assert B.closing_intervals(BASE, events) == ([(BASE, BASE + 400)], (BASE, BASE + 441))


def test_stop_part_way_through_selection_exposes_the_used_label_slots_not_a_prefix_of_everything():
    events = ev("game_start", range(200)) + ev("select_start", [0, 1, 2, 7])
    assert B.closing_intervals(BASE, events)[0] == [(BASE, BASE + 203), (BASE + 207, BASE + 208)]


def test_qualification_non_pass_exposes_repeats_but_not_the_training_seed():
    events = (ev("game_start", range(200)) + ev("select_start", range(200)) + ev("label_start", HELD)
              + ev("repeat_start", HELD))
    assert B.closing_intervals(BASE, events)[0] == [(BASE, BASE + 440)]


def test_full_run_exposes_the_whole_block_and_noise_events_are_ignored():
    events = (ev("game_start", range(200)) + ev("game_end", range(200)) + ev("select_start", range(200))
              + ev("label_start", range(200)) + ev("repeat_start", HELD) + [{"event": "train_start"}])
    assert B.closing_intervals(BASE, events) == ([(BASE, BASE + 441)], (BASE, BASE + 441))


def test_stop_during_corpus_exposes_started_games_only_and_retires_all():
    assert B.closing_intervals(BASE, ev("game_start", range(3))) == ([(BASE, BASE + 3)], (BASE, BASE + 441))
    assert B.closing_intervals(BASE, []) == ([], (BASE, BASE + 441))


@pytest.mark.parametrize("bad", [{"event": "repeat_start", "game": 3}, {"event": "game_start", "game": 200},
                                 {"event": "select_start", "game": -1}])
def test_exposure_refuses_an_event_outside_the_schedule(bad):
    with pytest.raises(ValueError):
        B.used_slots([bad])
```

- [ ] **Step 2: Run them to verify they fail.** Expected: `AttributeError: ... 'closing_intervals'`.

- [ ] **Step 3: Implement**

```python
def _game(e: Mapping[str, Any]) -> int:
    g = e["game"]
    if not (isinstance(g, int) and 0 <= g < N_GAMES):
        raise ValueError(f"event {e} names no scheduled game")
    return g


_SLOT_OF = {"game_start": lambda e: GAME + _game(e), "select_start": lambda e: LABEL + _game(e),
            "label_start": lambda e: LABEL + _game(e), "repeat_start": lambda e: repeat_slot(_game(e)),
            "train_start": lambda e: TRAIN}


def used_slots(events: Sequence[Mapping[str, Any]]) -> List[int]:
    """Card §9: the union of slots any phase logged as used; other events are ignored."""
    return sorted({_SLOT_OF[e["event"]](e) for e in events if e["event"] in _SLOT_OF})


def closing_intervals(base: int, events: Sequence[Mapping[str, Any]]) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """EXPOSED = the used slots as merged half-open intervals; RETIRED = the whole block, whatever happened."""
    runs: List[List[int]] = []
    for s in used_slots(events):
        if runs and runs[-1][1] == base + s:
            runs[-1][1] += 1
        else:
            runs.append([base + s, base + s + 1])
    return [tuple(r) for r in runs], (base, base + N_SLOTS)
```

- [ ] **Step 4: Run the tests.** Expected: 39 passed.

- [ ] **Step 5: Seed scan, then commit (only on approval)**

```bash
.venv/bin/python -B - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("S", "docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
for f in ("scripts/GPU/alphazero/search_distillation_stage_b.py", "tests/test_search_distillation_stage_b.py"):
    print(f, S.scan(open(f).read()))      # each must print {}
EOF
git add -- scripts/GPU/alphazero/search_distillation_stage_b.py tests/test_search_distillation_stage_b.py
git commit -m "Stage B checkpoint 1: pure CPU pieces (split, selection, digest, leakage, support, TV/KL, Q, L1-L3, exposure)"
```

**STOP at checkpoint 1.** Report the test output (count, duration and captured exit status) and the scan results. Wait for review.

---

# Checkpoint 2 — Minimum label and training wiring (FOR REVIEW; not yet planned task-by-task)

This is the smallest wiring the pilot needs. It is listed so the reviewer can cut it down before any code is written. After review, each item becomes a TDD task like the ones above. Nothing below loads a model or runs a search until separately approved.

**Reused unchanged:**
- **Search:** `MCTS(evaluator, cfg_from(EvalConfig(**PROTOCOL), sims), random.Random(seed ^ 0xA5A5A5)).search_with_root(state, add_noise=False)`, from `eval_runner.cfg_from`, `EvalConfig` and `external_learning_stage3.PROTOCOL`.
- **Evaluator and checkpoint loading:** exactly as the Stage A run script does: the main checkout's checkpoint by absolute path, with its SHA-1 verified.
- **Batching and loss:** `external_learning_stage2.batch_of` and `chosen_ce`. Fed `PositionRecord`s built as `external_learning.position_record` builds them, but with `visit_counts` = the full counts from `full_counts`. `make_padded_batch` normalizes the counts, and `batch_of` already checks that the mass is kept.
- **Head training:** `external_learning_stage2.head_only` and `train_head(network, records, schedule)`, where `schedule = numpy.random.Generator(PCG64(T)).integers(0, n_train, size=(160, 32))`. The same array is used for both arms. `LEARNING_RATE` (1e-3) and `CLIP_NORM` (1.0) are already the card's values.
- **Held-out logits:** `forward_rows`.
- **Candidate files:** `save_candidate` and `surface_check`.

**New, minimal:**

1. **A label function.** `label(evaluator, state, sims, seed) -> sparse`. It calls the search once and returns `sparse_label(visits, sims)`.
   - It needs one CPU test with a stub evaluator (uniform priors, zero value) driving the real `search_with_root`. The test confirms two things:
     - the root children's visits sum to exactly `sims`; this is still unverified;
     - a repeated seed gives identical counts (the D check).
   - The common-random-numbers prefix property gets a CPU test of its own: the first 400 simulations of a 1,600 search are identical to a 400 search with the same seed.
2. **The soft-target β fit.** About 25 lines: #53's convex bisection with derivative `Σ p·z − Σ T·z`, reusing `external_learning._padded` and `scaled_probs`.
   - It needs one CPU fixture check: the fitted β has a zero derivative; a known two-move case gives a hand-computable β.
3. **Phase scripts, each in the evidence directory as with Stage A.** Each one:
   - writes create-only outputs;
   - writes `progress.jsonl` with the Task 5 events logged **before** each use;
   - runs under a `timeout -k 10` wrapper at its card cap (G 12 h, P 5 min, L1 85 min, L2 3.5 h, T 30 min);
   - records the outer status and survivors.

   The phases:
   - **G:** the Stage A runner at 400/400 with 200 slots.
   - **P:** `select_game` for all 200 games, then `leakage_filter`.
   - **L1:** held-out labels, T′ and D. Then `gate_q` on the committed L1 output.
   - **L2:** training labels and D.
   - **T:** both arms, β, the surface checks, `game_measures`, then `learning_verdict`.

   Survey, proof and closing copies come from Stage A's, with `N_SLOTS` 441 and `closing_intervals` in place of the prefix rule.

**Deferred** (not built for this pilot):
- workers or parallel labeling;
- general replay infrastructure (P reads only the move lists in Stage-A-style `record.json` and replay files);
- resumption;
- any failure check not named in the card.
