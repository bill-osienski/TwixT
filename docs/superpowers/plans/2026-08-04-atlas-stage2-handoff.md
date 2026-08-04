# Atlas Stage 2 — Delivered Interfaces and Qualification (Stage 3 Handoff)

**Status:** Stage 2 LANDED and QUALIFIED, 2026-08-04, at clean tree `bef2a96`.
**No reservoir was generated and no GPU work was performed.**

Qualification is complete: the full-suite `REAL_EXIT=0` landed and is recorded
below, read from the process rather than a pipe. Stage 2 is no longer
provisional.

**Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §3
(EXECUTION-FROZEN). **Plan:** `2026-08-04-atlas-stage2-corpus-geometry.md`.

## Qualification

```text
full suite   2397 passed / 4 skipped / 53 deselected / 0 failed   (REAL_EXIT=0)
             7m30s, exit code read from the process, never a pipe
Stage 2 new  62 tests   (geometry 33, producer 16, CLI 13)
```

Arithmetic: Stage 1's true final is **2335**, not the 2334 reported at the time —
its timing smoke was added after that measurement, and the suite was not re-run.
`2335 + 62 = 2397`, so **zero pre-existing tests changed behaviour**.

Every Stage 2 test uses synthetic `GameMeta` or `FakeEvaluator`. Nothing here
touches MLX, a checkpoint, or a reservoir.

### Test delta

| | tests |
|---|---:|
| `tests/test_corpus_geometry.py` | 33 |
| `tests/test_generate_atlas_reservoir.py` | 16 |
| `tests/test_build_atlas_corpus_cli.py` | 13 |
| **Stage 2 total** | **62** |

Suite at Stage 1 completion **2335** → Stage 2 **2397**. Delta **+62**, entirely
new tests; **no pre-existing test changed behaviour**, and none was modified,
skipped or deleted.

### No reservoir, no GPU — confirmed

No reservoir block was generated. No checkpoint was loaded. No MLX code ran: every
Stage 2 test uses synthetic `GameMeta` or `FakeEvaluator`, and the CLI's
generation subcommands only *print* commands. The `--checkpoint` arguments in
tests point at temporary byte files created solely so a SHA-1 exists to compare.

## Defects found and fixed during Stage 2

### Tuple keys are not JSON-serializable — reached the failure path

`json.dumps` cannot use **tuple keys**, and `default=` rescues unserializable
*values* only. `pilot_geometry_gate` returns `unmet` keyed by
`(split, phase, side)`, so `pilot-gate` raised `TypeError` and exited **1** where
the protocol specifies **3** — that is, the `PHASE_GEOMETRY_NO_GO` path, the one
an operator hits when the corpus is infeasible. The success path was unaffected
and green throughout.

**Why the unit tests could not catch it.** They call `pilot_geometry_gate`
directly and never serialize its result. Only running the CLI as a real
subprocess exercised the boundary. This is the producer/consumer-seam lesson
again, this time at the serialization boundary rather than the data one, and it
is the third distinct instance in this line of work.

**Fix, deliberately scoped:** a `_jsonable` converter in `build_atlas_corpus.py`,
normalizing only at the JSON boundary. The geometry module keeps `(split, phase,
side)` tuples, which are the natural type there and which every downstream
consumer indexes with. Stage 3 must apply the same normalization — reuse
`_jsonable`, do not re-key the geometry module.

## Delivered interfaces

### `scripts/GPU/alphazero/corpus_geometry.py` — pure, reservoir-free

```python
PHASES, SIDES, SPLITS, ALLOWED_N, MAX_SEED_RANGE_GAMES=480, PILOT_GAMES=24,
PILOT_PER_CELL=3, SIZING_MARGIN=1.20

phase_for_ply(ply) -> str
side_for_ply(ply, start_player) -> str
GameMeta(game_id, seed, n_moves, start_player)          # frozen dataclass
eligible_plies(meta, phase, side) -> list[int]
eligible_cells(meta) -> set[(phase, side)]
stable_key(sampling_seed, game_id, split, phase, side, ply) -> str   # SHA-1

MatchResult(assignment, achieved_flow, demanded_flow, unmet,
            min_cut_games, min_cut_cells)               # .complete property
match_games_to_cells(games, demands, sampling_seed) -> MatchResult
pilot_geometry_gate(pilot_games, sampling_seed) -> dict
size_continuation(pilot_games, n_target) -> dict
final_demands(n_target) -> dict[(split, phase, side), int]
select_ply(meta, split, phase, side, sampling_seed) -> int
assign_corpus(pilot_assignment, continuation_games, n_target, sampling_seed) -> dict
```

Cells are `(split, phase, side)` **tuples**. Stage 3 must convert them before any
`json.dumps` — tuple keys are not serializable and `default=` rescues only values.
`build_atlas_corpus._jsonable` is the working converter.

### `scripts/GPU/alphazero/generate_atlas_reservoir.py`

```python
MANIFEST = "block_manifest.json"
PRODUCTION_SETTINGS = {active_size:24, n_simulations:400, max_moves:280,
                       batching:[14,48,8], add_noise:True}

seed_for_index(base_seed, game_idx) -> int              # base_seed + game_idx
validate_source_provenance(porcelain, git_head, checkpoint_path,
                           checkpoint_sha1) -> dict     # PURE; raises
preflight_source_provenance(checkpoint_path) -> dict    # calls git, then above
generate_block(evaluator, base_seed, start_index, n_games, out_dir,
               provenance, n_simulations=400, max_moves=280,
               active_size=24) -> list[dict]
load_manifest(block_dir) -> dict
assert_blocks_agree(pilot_manifest, cont_manifest) -> None
load_block(block_dir, base_seed, start_index, n_games,
           *, require_production=True) -> list[GameMeta]
game_meta_from_sidecar(d) -> GameMeta
```

Sidecar row: `game_idx, seed, start_player, n_moves, winner, draw_reason,
move_history`.

### `scripts/GPU/alphazero/build_atlas_corpus.py` — generates nothing

Six subcommands in staged order. Exit codes **0** OK, **2** usage/validation,
**3** `PHASE_GEOMETRY_NO_GO`, **4** `ASSIGNMENT_SHORTFALL`.

```text
emit-protocol              --base-seed --sampling-seed --checkpoint
emit-pilot-command         + --out-dir
pilot-gate                 --sidecar-dir --base-seed --sampling-seed
size                       --sidecar-dir --base-seed --n-target
emit-continuation-command  --base-seed --pilot-dir --n-target --size-artifact
                           --checkpoint --out-dir
assign                     --pilot-dir --continuation-dir --base-seed
                           --n-target --sampling-seed
```

## Invariants Stage 3 may rely on

- `game_seed = base_seed + game_idx`, exactly. A single index reproduces
  independently of the block it was produced in — proven by
  `test_a_single_index_reproduces_exactly`.
- Source provenance is a **preflight**: a dirty tree or unidentifiable checkpoint
  costs nothing, because it is checked before evaluator construction.
- Blocks fail closed on a pre-existing directory; pilot and continuation each own
  one.
- `load_block` verifies manifest agreement, production settings, clean provenance,
  valid checkpoint digest, exact filenames, duplicate-index detection, exact index
  coverage and the seed identity — **before** any `GameMeta` exists.
- The continuation interval is **derived**, never read from the block being
  validated. An oversized block is rejected as an unauthorized top-up.
- `N` is never accepted pre-pilot; `G_total` is always recomputed, never trusted.

## Worked sizing values (full-coverage pilot, `q_S = 1.0`)

| `N` | `d_c` | `g_cont` | `G_total` | continuation | demand |
|---:|---:|---:|---:|---:|---:|
| 200 | 22 | 185 | 240 | 216 | 176 |
| 240 | 27 | 227 | 280 | 256 | 216 |

The slack between continuation and demand **is** the 20% margin, not spare
capacity.

## What Stage 3 must still close

1. **Scale.** The producer is qualified at 3–6 games, 8 simulations,
   `max_moves=6`, `active_size=6`, `FakeEvaluator`. Throughput, memory over a
   480-game range, and the `q_S` a real ply distribution yields are **unmeasured**.
   The geometry is proven; the supply it will be handed is not.
2. **Nonzero `I`.** `N_actual = root.visit_count − I` is never exercised with
   inherited visits — every Stage 1 search starts from a fresh root, so `I = 0`.
   Stage 3 owns the boundary consumer and must pin it against a real
   `advance_root`-inherited root where `320 ≤ N_actual ≤ 400` has content.
3. **Cache clearing at a real `advance_root`.** `SelectionTracer.clear_node_cache()`
   is exercised only by direct unit call, so the `id()`-reuse hazard it exists to
   prevent is never reproduced.
4. **`within_forced_simulation` inside a warm replay.** Only observed on
   synchronous forced simulations so far.

## Standing authorization boundary

Corpus generation and every GPU measurement run remain **unauthorized**. Stages 3–5
are unplanned; Stage 3 is planned only against these interfaces once they are
accepted.
