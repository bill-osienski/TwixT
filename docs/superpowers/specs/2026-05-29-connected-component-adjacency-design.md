# Connected-Component Adjacency Cache — Design

- **Date:** 2026-05-29
- **Branch:** `perf/connected-component-adjacency`
- **Status:** Approved (design), pending spec review → implementation plan
- **Author:** bill-osienski (with Claude)

## Problem

`_get_connected_component` performs a connectivity BFS by scanning the **entire**
`bridges` set for every popped peg (`scripts/GPU/alphazero/game/twixt_state.py:411`).
That is O(V·E) per call: at a dense late-game position (~140 own-color pegs,
~500–1000 bridges) it is tens of thousands of tuple comparisons per call, and the
function is on the hottest path in the engine — it backs `_check_win`, `winner()`,
and `connectivity_masks()`, which run on essentially every MCTS node and every
leaf evaluation.

This surfaced as a self-play worker stalling 40+ minutes on a dense ply-280 game
(iter 0330, worker 6, ~5.77× straggler vs peers). It is not a hang — the BFS
provably terminates — it is the O(V·E) cost amplified by MCTS, biting reliably
now that the trained network reaches dense positions. Two amplifiers compound it:
`_check_win` recomputes a component per goal-edge peg (no dedup), and
`connectivity_masks` (tensor channels 24–29) pays the same cost on every leaf
evaluation.

The same algorithm exists, by deliberate transliteration, in the deployed JS
engine (`server/gameLogic.js` `_getConnectedComponent`), which the Node-side MCTS
(`server/mcts.js`) uses to serve the live AI player. So the inefficiency affects
both training throughput (Python) and live-AI move latency (JS).

## Goals

- Reduce `_get_connected_component` from O(V·E) to O(V+E) amortized per call.
- Apply the **identical** change to both engines (Python `TwixtState` and JS
  `gameLogic.js`) so the transliteration mirror is preserved — "both engines, one
  unit."
- Preserve byte-identical observable behavior: `winner()` verdicts and
  `connectivity_masks()` / tensor outputs must not change.

## Non-Goals (YAGNI)

- **No `seen` dedup in `_check_win`.** Once the adjacency cache exists, `_check_win`
  is a handful of cheap traversals sharing one built adjacency; the multiplier is
  negligible. Adding `seen` is extra divergence-from-current (and would need
  mirroring to JS) for no meaningful gain. Revisit only if profiling shows it hot.
- **No MCTS node-level caching** of `isTerminal()`/`winner()` in `mcts.py`/`mcts.js`.
  Real, possibly larger lever, but orthogonal and a separate change.
- **No replacement of `bridges`** as the source of truth (rules out a DSU-canonical
  or bitmask-canonical store). `bridges` has 12+ external readers
  (`probe_eval.py`, `sealed_lane.py`, `tensor_repr.py`, oracle helpers, …) that
  must keep working unchanged.

## Approach: surgical lazy adjacency cache (mirrored)

Add a private, lazily-built adjacency map `peg → [bridge-connected neighbor pegs]`
(one map per state, not per player) to each state.
`_get_connected_component` builds it once from `bridges` if absent,
then walks `_adj[pos]` instead of scanning all bridges. `winner()`, `_check_win()`,
and `connectivity_masks()` are **untouched** — they keep calling
`_get_connected_component`, which is now O(V+E) amortized. This:

- keeps the diff confined to *how neighbors are enumerated* (not the graph), so
  output-equivalence is trivially arguable;
- preserves the documented invariant that `winner()` and `connectivity_masks()`
  share one connectivity path and "can never drift"
  (`twixt_state.py:440`, `gameLogic.js:450`);
- mirrors 1:1 into JS, keeping the engines identical.

### Alternatives considered

- **Port the sibling engine's pattern** (`find_connected_components` +
  `_adj_cache` + `cc_revision` + `invalidate_cc_cache` from `scripts/GPU/game`,
  tested in `tests/test_cc_optimization.py`). Proven code, but built for a
  different class/API, forces a larger refactor of `winner`/`_check_win`/
  `connectivity_masks`, and has **no JS counterpart** — it would break the
  Python↔JS mirror and create a third variant. Rejected.
- **Incremental adjacency** (build in `apply_move`, deep-copy in `copy()`).
  Correct and mirror-able, but pays a dict-of-lists copy on every `copy()` (MCTS
  copies a lot) and couples bookkeeping to the mutation site for no gain over
  lazy. Rejected as heavier.

## Detailed design

### Data structure

- **Python:** new dataclass field
  `_adj: Optional[Dict[Pos, List[Pos]]] = field(default=None, init=False, compare=False, repr=False)`.
  `init=False` keeps it internal (not a constructor arg); `compare=False`/`repr=False`
  document that it is derived state. It must not participate in `__hash__`/`__eq__`
  (already custom and keyed on `pegs`+`bridges`).
- **JS:** `this._adj = null;` in the `TwixtState` constructor.

### Build

In `_get_connected_component` (and only there), if `_adj is None`, build it with a
single pass over `bridges`: for each bridge, append each endpoint to the other's
neighbor list. The map is per state, not per player — bridges are same-player by
construction, and `_get_connected_component` already rejects off-color pegs at pop
time (today's `twixt_state.py:404`). So per-player traversal stays correct and a
stray cross-player or orphan bridge is ignored exactly as now (it gets
enqueued-then-rejected at pop instead of never-enqueued — identical component
result). Then BFS using `_adj.get(pos, [])` for neighbors.

### Invalidation: null-on-copy

- `copy()` does **not** carry `_adj`; the child starts `None` and rebuilds lazily
  on first query. This is safe because `apply_move` is `copy()` → mutate
  `bridges`/`pegs` → return, so the child's cache is always built against final
  state (`apply_move`, `twixt_state.py:316`).
- Add `_invalidate_adj()` (sets `_adj = None`) for any code path that mutates
  `bridges`/`pegs` on an existing state directly (tests/tools, mirroring how the
  sibling engine's tests call `invalidate_cc_cache()` after manual `bridges.add`).
  Production mutates only via `apply_move`, so this is a safety/robustness hook.

### Untouched

`winner()`, `_check_win()`, `connectivity_masks()`, `is_terminal()`, the `bridges`
set and all its external readers, serialization (`to_dict`/`from_dict`),
`__hash__`/`__eq__`. The JS change mirrors exactly the same surface.

## Correctness & parity invariants (must hold)

1. **Same graph:** `_adj` mirrors `bridges` edge-for-edge (same-player only), so
   BFS reachability — and therefore every component, every `winner()` verdict, and
   every `connectivity_masks` bit — is identical to today.
2. **Shared path preserved:** `winner()` and `connectivity_masks()` still both go
   through `_get_connected_component`; they cannot drift from each other.
3. **Output parity is the contract, not algorithm parity.** The deployed JS game
   is unaffected because consumers (win adjudication, the trained network's input
   features) observe only outputs. Editing Python cannot change JS runtime
   behavior; editing both identically keeps them in lockstep.

## Testing strategy

### Python equivalence test (new)

New test file (e.g. `tests/test_twixt_state_cc_adjacency.py`) mirroring
`tests/test_cc_optimization.py::TestEquivalence`:

- Keep a reference "legacy" full-bridge-scan `_get_connected_component` in the
  test. Assert the optimized engine produces identical **component sets**,
  identical `winner()`, and identical `connectivity_masks()` over a corpus of:
  - **(a) real replays:** every position from replaying the completed games in
    `Replays/` (sparse → realistically dense);
  - **(b) synthetic dense games:** seeded random legal games played to ply 280,
    to force the dense regime the bug lives in;
  - **(c) fixtures:** empty board, single peg, orphan bridge (endpoints lack
    pegs), mismatched-player bridge, multi-component graphs.
- **Cache-behavior tests:** lazy build on first query; `copy()` yields a child
  with `_adj is None` that rebuilds correctly; `_invalidate_adj()` forces a
  correct rebuild after a manual `bridges`/`pegs` mutation.

### JS / cross-engine (existing, must stay green)

- `tests/test_game_rules_parity.py` — Python vs JS `winner()`/`is_terminal()` over
  full random games to terminal.
- `tests/test_js_py_tensor_parity.py` — Python `to_tensor()` vs JS
  `buildStateTensor` for all 30 channels (incl. connectivity 24–29) at 1e-6.
- `tests/js_oracle/` deterministic game oracle.
- Transitive guarantee: new-Python == legacy (equivalence test) and JS == Python
  (parity tests) ⇒ JS == legacy.

### Optional (not default)

Regenerate exact iter-0330-density positions by running self-play with the
`model_iter_0330` checkpoint and adding them to corpus (b). Heavier (loads the
model); skip unless extra realism is wanted. The lost worker-6 game
(`iter_0330_game_099`) was never saved — the worker was killed mid-game.

## Implementation-time checks (not blockers)

- Confirm JS `gameLogic.js` is copy-forward (a fresh state per move, like Python)
  so null-on-copy applies; `server/mcts.js` walking a `node.state` tree strongly
  implies it. If it mutates in place anywhere, route that through
  `_invalidate_adj()`.
- Verify `_adj` cannot leak into `__hash__`/`__eq__`/serialization.

## Risks & mitigations

- **Silent output drift (the real risk):** a subtle bug where adjacency BFS
  disagrees with full-scan on some dense state would skew training labels/features
  vs the deployed engine. *Mitigation:* the equivalence test over the dense corpus
  + the existing cross-engine parity tests; these gate the merge.
- **Stale cache from manual mutation:** mitigated by null-on-copy + `_invalidate_adj()`
  and by the fact that production mutates only via `apply_move`.

## Files affected

- `scripts/GPU/alphazero/game/twixt_state.py` — `_adj` field, lazy build in
  `_get_connected_component`, `_invalidate_adj()`, `copy()` leaves `_adj` unset.
- `server/gameLogic.js` — identical mirror.
- `tests/test_twixt_state_cc_adjacency.py` — new equivalence + cache-behavior tests.
- (No change, must stay green) `tests/test_game_rules_parity.py`,
  `tests/test_js_py_tensor_parity.py`, `tests/js_oracle/*`.

## Success criteria

- `_get_connected_component` is O(V+E) amortized; dense-position component calls
  are dramatically faster (target: the previously-stalling regime completes in
  seconds, not tens of minutes).
- New equivalence test passes over the full corpus.
- All existing cross-engine parity/oracle tests stay green.
- Both engines carry the identical change; no divergence.
