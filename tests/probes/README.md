# Twixt Probe Suite

Curated Twixt positions used as a regression gate for value-head
behavior. Versioned in git; evaluated against every candidate checkpoint.

## Files

- `twixt_probes.json` — the committed curated suite (50–80 probes)
- `candidates.json` — (gitignored) intermediate output of the sampler
- `baselines/` — immutable baseline scoring artifacts per checkpoint

## Categories

| Category | Description | Min | Max |
|---|---|---:|---:|
| `near_win_red` | Red is 1–3 moves from winning | 10 | 15 |
| `near_win_black` | Black is 1–3 moves from winning | 10 | 15 |
| `blocked_or_trap` | One side has many pegs but no goal-touching component | 8 | 10 |
| `false_positive_connectivity` | Looks connected but globally isn't | 5 | 10 |
| `dense_but_disconnected` | Similar, either color, different heuristic | 8 | 10 |
| `central_win` | Winning chain primarily in board interior | 8 | 10 |
| `edge_corner_legitimate` | Edge/corner placement legitimately good | 5 | 10 |
| `symmetric_sanity` | Mirror-pair probes to check symmetry | 5 | 10 |

## Confidence Tiers

- `forced` — unambiguously winning/losing (1–2 moves from terminal or
  obvious structural lock). Gate requires **≥95% sign-correct** on this tier.
- `strong_advantage` — clearly better but not forced. Gate requires
  **≥80% sign-correct**.
- `unclear_do_not_use` — reviewer couldn't decide; discarded from final suite.

**Reviewer-disagreement rule:** if two reviewers disagree on a candidate's
tier, default to `unclear_do_not_use`. Do not force resolution.

## Schema

```json
{
  "id": "near_win_red-001",
  "category": "near_win_red",
  "confidence": "forced",
  "side_to_move": "black",
  "expected_value_sign": 1,
  "expected_value_min": 0.75,
  "expected_value_max": null,
  "active_size": 24,
  "ply": 42,
  "move_history": [[0, 3], [23, 20], ...],
  "source_game": "scripts/GPU/logs/games/iter_0820_game_014.json",
  "source_ply": 42,
  "peg_counts": {"red": 22, "black": 19},
  "mirror_of": null,
  "evaluation_modes": ["nn_only", "mcts"],
  "note": "Red has a chain reaching row 0 to row 21, one bridge from bottom"
}
```

### Field semantics

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | stable identifier, unique |
| `category` | yes | one of the categories above |
| `confidence` | yes | `forced` or `strong_advantage` (never `unclear_do_not_use` in the committed suite) |
| `side_to_move` | yes | whose turn it is in the replayed state |
| `expected_value_sign` | yes | +1 = red winning, -1 = black winning, 0 = balanced. Always evaluated from `side_to_move` perspective (flip sign if needed) |
| `expected_value_min` | optional | gate's magnitude check: `|nn_value| >= this` |
| `expected_value_max` | optional | upper bound on magnitude |
| `active_size` | yes | curriculum size; 24 for production probes |
| `ply` | optional | length of move_history (cross-check) |
| `move_history` | yes | canonical state — replayed from empty board |
| `source_game` | yes | where this probe was sampled from |
| `source_ply` | yes | ply offset in the source game |
| `peg_counts` | optional | convenience metadata |
| `mirror_of` | optional | id of the probe this mirrors, for `symmetric_sanity` |
| `evaluation_modes` | optional | which gate metrics use this probe (`nn_only`, `mcts`, or both) |
| `note` | optional | human annotation |

## Adding a new probe

1. Run sampler to extract candidates: `python scripts/build_probe_candidates.py --out tests/probes/candidates.json`
2. Review candidates manually: assign `confidence`, edit `note`, discard `unclear_do_not_use`
3. Append curated candidates to `twixt_probes.json` (preserving `id` uniqueness)
4. Run the baseline scoring script against iter-0999 to re-score
5. Commit with an ADR-style note describing what was added/changed

## Running the evaluator

For a formal gate-comparison run:

```bash
python -m scripts.GPU.alphazero.probe_eval \
  --weights checkpoints/alphazero-v2-staged/model_iter_0150.safetensors \
  --probes tests/probes/twixt_probes.json \
  --sims 200 \
  --out probe_eval_iter_0150.csv
```

The `--weights` path is **required** for formal runs. Passing it ensures the
output is traceable to a specific checkpoint and not to an implicit "latest."
