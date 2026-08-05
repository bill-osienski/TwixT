# Atlas Stage 5 — Delivered Interfaces and Qualification (Operator Handoff)

**Status:** Stage 5 LANDED and QUALIFIED, 2026-08-05, at clean tree `04bc8ec`.
**No reservoir generated, no checkpoint loaded, no MLX executed, no measurement run.**

**Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §2b, §3, §4,
§8, §9 — EXECUTION-FROZEN.
**Plan:** `2026-08-04-atlas-stage5-composition-runbook.md`, revision 6.

Stage 5 is the last tooling stage. It hands the operator one launchable protocol; it
does not launch it, and **nothing in this document authorizes GPU work.**

## Qualification

```text
full suite   2624 passed / 4 skipped / 53 deselected / 0 failed   (REAL_EXIT=0)
             9m30s, exit code read from the process, never a pipe
Stage 5 new  68 tests
```

| file | tests |
|---|---:|
| `tests/test_atlas_row_facts.py` | 8 |
| `tests/test_atlas_artifact.py` (Stage 4's 14 + 5) | 5 |
| `tests/test_atlas_run.py` | 37 |
| `tests/test_run_atlas_cli.py` | 18 |
| **Stage 5 total** | **68** |

`2556 + 68 = 2624` exactly. The baseline was **measured** before starting
(`--collect-only`: 2560 selected = 2556 + 4 skipped), not taken from a document — the
lesson from Stage 4, where a stale quoted baseline made a correct delta look wrong. **No
pre-existing test changed behaviour.**

### Standing invariants, verified

- `git diff cec2810..HEAD -- mcts.py` is **empty**. Stage 1's scoped observer exception
  remains the only change ever made to it.
- No Stage 5 module imports MLX. Every test uses synthetic dicts or `FakeEvaluator`.
- `tests/test_atlas_readout_c.py` (30) still green after `_is_flat` → `is_flat`.

## Delivered interfaces

### `atlas_row_facts.py` — the seam Stage 4 could not qualify

```python
NEAR_EVEN_ABS_VALUE = 0.30        # section 8's existing definition, verbatim
derive_row_facts(legs, snapshots, target_ply, start_player,
                 assigned_phase=None, assigned_side=None) -> dict
```

`phase` / `side` are derived from the frozen ply bounds **and cross-checked** against the
assignment — a disagreement fails the row rather than letting either side win silently.
`flat_policy` applies `atlas_readout_c.is_flat` (now public, one implementation shared
with `classify_edge_strata`) to the merged deep line's **root-edge priors**, because
`capture_tree_state` records `policy_entropy` and `n_legal` but **not the top prior**, so
the frozen two-part predicate is not computable from a capture. `near_even` is
`|V_stm| ≤ 0.30` at nominal `B = 400`. An undefined fact is `None` and named in
`undefined`, never `False`.

### `atlas_artifact.load_run` — the authenticated inverse of `emit`

```python
load_run(path_or_text) -> dict
```

`emit` is lossy for exactly the types the read-outs need. Every loss is **silently wrong**
rather than loudly broken — a string-keyed prior map still sorts and still yields a rank,
just not the right one:

| written | returns as | who breaks |
|---|---|---|
| `LegResult` / `BoundaryRecord` | `dict` | Read-out B and `atlas_labelling` read by attribute |
| `parent_visits` key `()` / `(7,3)` | `""` / `"7\|3"` | `edge_retention` looks up tuples |
| `parent_priors` / `visit_counts` int keys | strings | `static_retention` ranks int move ids |
| `parent_path` / `sources` tuples | lists | edge identity and dedup |

`load_run` restores **both** edge lists — the deep lines carry `edges`, only `merged`
carries `required_edges` — and authenticates rather than parsing: provenance must
validate, `schema_version` must match, and a missing or non-list `rows` is refused
instead of reading as an empty run. **Nothing may consume an artifact except through it.**

### `atlas_run.py` — the composition

```python
LADDER_BATCHING = (14, 48, 8)   PREFIX_SIMS = 400   ACTIVE_SIZE = 24
LEG_INCREMENTS_DEFAULT   BOUNDARY_THRESHOLD_DEFAULT   LEG_B_DEFAULT
ladder_config(n_simulations) -> MCTSConfig
RowOutcome(ok, row, failure, game_id)
run_row(evaluator, meta, assigned, *, move_history, base_seed, ...) -> RowOutcome
run_corpus(evaluator, metas, assigned_rows, *, base_seed, move_histories,
           provenance, ...) -> dict
pilot_rows(pilot_games, sampling_seed) -> list[dict]
run_pilot(evaluator, pilot_games, *, sampling_seed, base_seed,
          move_histories, provenance, ...) -> dict
verify_pilot(pilot_doc, pilot_games) -> assignment
verify_assignment(pilot_games, pilot_assignment, sampling_seed, n_target,
                  continuation_games, assignment_rows) -> None
combine_final_runs(pilot_doc, continuation_doc, *, provenance) -> dict
run_final(evaluator, *, pilot_doc, pilot_games, continuation_games,
          assignment_rows, base_seed, move_histories, provenance, ...) -> dict
```

**One evaluator per run, one seeded `MCTS` per row.** `run_corpus` has no factory or
checkpoint parameter, so the compiled-evaluator MLX trap is unreachable rather than
discouraged.

**The completeness condition.** The corpus is exactly the assigned positions, so **any**
row failure makes the run `ABORTED` and `authoritative: false`. Read-outs still run and
are still written — a half-measured corpus is worth diagnosing — but nothing computed
from a partial corpus is called authoritative. This compares assigned against measured
and introduces no number, so **no failure-tolerance knob exists or can exist.**

**§3's chronology, executable.** `run_pilot` measures the 24 fixed discovery rows and
**outputs** `N`; an aborted pilot does not size (`UNAVAILABLE`, `N: None`) and its
early-widening result is non-authoritative. `run_final` takes `N` only from
`pilot_doc["sizing"]`, re-derives the pilot gate / assignment / sizing / labels / row
facts from the **verified block**, re-derives the continuation selection from the
**complete** `G_total − 24` block, and requires `24 + (N − 24) == N`.

### `run_atlas.py` — the operator CLI

```python
EXIT_OK=0  EXIT_USAGE=2  EXIT_PROVENANCE=3  EXIT_ABORTED=5
STOP_CONDITIONS            # verdict, owner, action, exit_code
measure_provenance(checkpoint, *, pilot_dir, continuation_dir, pilot_artifact)
launch_wrapper(argv, *, out_dir) -> str
write_status_sidecar(path, *, verdict, exit_code, **extra)
main(argv=None) -> int     # preflight | emit-runbook | run-pilot | run-final
```

**Verdicts are results; exit codes are process outcomes.** A `CAPACITY_FAILURE`,
`NO_SHAPE_PASSES` or `NOT_DEPLOYABLE` exits **0** — the run did what it was asked. Only
usage, provenance and `ABORTED` are nonzero.

**Provenance is measured, and symmetric.** The tree, HEAD and checkpoint are measured
before any evaluator exists, then **both** the digest and the HEAD must match the block
manifests and the pilot artifact. A mismatch means regeneration or requalification.

**Two sidecars.** `launch_wrapper` captures `rc` first, writes it, then re-raises it;
`status.json` is written after the artifact. Together they distinguish: both present
(trust the verdict), wrapper only (python died before reporting), neither (nothing ran).
The parser exposes **no** frozen parameter — no `--active-size`, `--prefix-sims`,
`--increments` or `--n-target`.

## Implementation findings

1. **The ladder cannot be shrunk for CPU tests.** Labelling and Read-out B index
   `nominal_B` at 400/1,600/3,200/6,400, so the planned `increments=(80,80,80,80)`
   produced rungs they correctly rejected — 13 tests failed on `missing rungs`. Only the
   **prefix** budget is reducible. One frozen ladder at board 24 costs **4.6s**, which is
   what makes the 24-row pilot fixture ~110s.
2. **Board 24 is required only where late cells are.** A 6×6 fixture game terminates
   around 29 moves and can never reach ply 91+, and `replay_prefix` asserts
   `n_moves == len(move_history)`. Row-level tests run at `active_size=6`; only the pilot
   pays for board 24. The aborted-pilot fixture supplies one history so 23 rows fail at
   the `n_moves` check before any search.
3. **Budget defaults must resolve at CALL time.** `prefix_sims=PREFIX_SIMS` as a
   signature default binds at import, so `monkeypatch.setattr(ar, "PREFIX_SIMS", 2)`
   would silently do nothing and the CLI tests would run the full budget while appearing
   to inject a small one. They resolve against the module globals inside the function.
4. **Two Stage 4 fixtures could not reach the new code** — `_kw`'s bare
   `{"nominal_B": 400}` legs and `_snapshots`' empty `by_shape`. `load_run` was **not**
   made tolerant of them: a leg without `visit_counts` is not a `LegResult`, and
   rehydrating one would fabricate. The fixtures now supply what production emits.
5. Two self-inflicted: `run_corpus`'s docstring contained "tolerance" while a test
   grepped the source for it (the test now inspects signature parameters, which is what
   it meant); and the runbook named `wait $!` inside its own warning against it, which
   the guard could not distinguish from an instruction.

## Remaining unknowns — measured only by a real run

| unknown | why it matters | closed by |
|---|---|---|
| Real-scale throughput | the runtime projection is unvalidated at board 24 against a real checkpoint; §4 forbids scaling a smoke | the pilot |
| The `remaining` distribution | §4's preregistered rule **fails Read-out A's controller-deployability claim if the median is zero** | the pilot |
| The inheritance-reset rate | a high rate means the warm start is not actually warming | the pilot |
| `q_S` and the real ply supply | the sizing formula's denominator, and the prefix cost driver | the pilot |
| Pilot class frequencies `p_m`, `p_s` | **`N` itself** — the pilot's output, not an input | the pilot |
| The early static widening result | `both_fail` closes progressive widening without inventing another shape | the pilot |

Everything in that table is a **pilot** measurement. Nothing about the continuation or
the final run can be known before the pilot reports.

## Standing authorization boundary

Corpus generation and every GPU measurement run remain **unauthorized by this document**.
A separate written authorization — `docs/superpowers/2026-08-05-atlas-pilot-authorization.md`
— scopes the pilot alone. Continuation generation and `run-final` remain unauthorized
until the pilot produces a valid `N` and its early-widening result.
