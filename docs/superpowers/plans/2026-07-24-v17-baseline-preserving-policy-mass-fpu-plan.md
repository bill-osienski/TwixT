# v17 Baseline-Preserving Policy-Mass FPU — Execution Plan

**Status:** DRAFT FOR REVIEW — AMENDED AFTER ZERO-GPU PREFLIGHT (2026-07-24).
This plan implements
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`.
No v17 implementation or new experimental generation begins until that design
is explicitly approved and frozen. GPU/operator tasks require separate
authorization at each gate. Do not commit or push without explicit user
authorization.

## Working rules

- Preserve all existing user documentation changes.
- Keep the v16 production root, diagnostics, manifest, screen, replays, and
  control artifacts immutable.
- Do not modify the retired v16 formula or reinterpret its field.
- New outputs go only under
  `logs/eval/fpu_v17_baseline_policy_mass/`.
- TDD for every code change.
- Freeze every v17 effective MCTS config to
  `(eval_batch_size, stall_flush_sims, pending_virtual_visits) = (14, 48, 8)`.
- Canonical, timestamp-free diagnostic artifacts.
- File-scoped staging only if the user later authorizes a commit.
- Stop immediately on any failed scientific prerequisite.

## Gate map

```text
design approval/freeze
  → TDD + exact-zero proof
  → asymmetric same-checkpoint support + full software review
  → 32-game full-chain tooling smoke
  → tooling-only replay smoke from the chain's two selected controls
  → 8-game same-checkpoint runner tooling smoke
  → fresh 1,600-game development reservoir
  → fresh 32-position coefficient selection
  → freeze smallest passing coefficient
  → fresh 2,200-game / 56-position held-out collateral
  → established A/B/C/D
  → 800-game same-checkpoint strength
  → paired contemporary 0379 validation
  → separate self-play adoption decision
```

## Completed pre-freeze work

The design-only, zero-GPU preflights authorized by design §1 are complete:

- Shipped selected-A application-mass reach artifact:
  `logs/eval/fpu_v17_baseline_policy_mass/preflight/selected_a_shipped_application_mass.json`,
  SHA-1 `8854e02f33f210ed381e0cd44e8e1de0fb7f4b7b`.
- Shipped late-flat root-safety exposure artifact:
  `logs/eval/fpu_v17_baseline_policy_mass/preflight/late_flat_root_shipped_mass.json`,
  SHA-1 `9a18c7ff90ef27e4afa91b8bbf32d676ee655fad`.
- Development exact witness/sizing SHA-1s:
  `548422b3e1067f52c5ab6981bdb94d737c7f0d0b` and
  `b41089e2d5c34a551e6b6b9ebe95299afc8cf6bc`.
- Held-out exact witness/sizing SHA-1s:
  `ca8a795ce42dfcc3011ba5bc1468532bd3615c7d` and
  `35c0f2eb37d42492e768ee365a1d460c576c35c1`.

These establish only grid reach, late-root safety exposure, and reservoir
sizing. They do not authorize Task 1, select a coefficient, or count as v17
scientific evidence.

## Task 0 — Freeze the preregistration

**Files:** the v17 design and this plan only.

1. Review every item in design §14.
2. Resolve amendments before any code or new evidence.
3. Change design status from `DRAFT FOR REVIEW` to `APPROVED — FROZEN` only
   after explicit user approval.
4. Record the frozen design and execution-plan SHA-1s in the v17 provenance
   root.
5. Stop. Implementation is a separate authorization.

**Gate:** exact approved design bytes exist before Task 1.

## Task 1 — Capture pre-change identity goldens

**Files:** create a v17-specific test/golden module; do not alter v16 artifacts.

1. Add a fixed synthetic-tree search signature capturing selected move, visit
   counts, root value, child Q/visits, and callback sequence under shipped FPU
   with effective batching triple `(14, 48, 8)`.
2. Add a fixed CPU search signature with observer off and the same batching
   triple.
3. Record and require the current pre-change full-suite baseline:
   `2,274 passed / 0 failed`. A different count or any failure stops Task 1;
   do not manufacture the expected count by silently stashing or excluding
   `tests/test_fpu_policy_mass_postmortem.py`.
4. Record SHA-1s for `mcts.py`, the relevant config/eval modules, and frozen
   design.
5. Freeze the exact shipped A/B/C/D per-case artifacts, aggregate baselines,
   case counts, selected moves, and SHA-1s for the Stage-4 validity check:
   - A:
     `logs/eval/calib020_0001_black_loss_post_opening_predrop_probe/position_probe_cases.csv`
   - B:
     `logs/eval/black_predrop_calib010_goal_line/goal_line_trigger_probe_cases.csv`
   - C:
     `logs/eval/calib020_post_opening_sweep/position_probe_cases.csv`
   - D:
     `logs/eval/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv`
6. Freeze only rows with `checkpoint == "0001"` from each canonical source;
   exclude any other checkpoint rows in multi-checkpoint CSVs.
7. Record that A/D `over` and `severe` use black-perspective thresholds
   `>=+0.25` and `>=+0.50`, and that the C/D one-ULP mean representations
   (C: `...6037` versus `...6039`; D: `...619` versus `...617`) are inside
   the frozen `1e-6` tolerance.

**Tests:** scientific-result goldens reproduce twice byte-identically. Config
and provenance labels are compared separately.

**Gate:** no MCTS source edit before goldens exist.

## Task 2 — Add the v17 config field and reuse the pure helper

**Files:**

- Modify `scripts/GPU/alphazero/mcts.py`.
- Create `tests/test_fpu_baseline_policy_mass_rule.py`.

TDD sequence:

1. Failing tests for:
   - Existing `policy_mass_fpu` formula values, clamp, and nonfinite behavior
     remain unchanged.
   - New field default `None`.
   - `0.0` accepted.
   - Negative/nonfinite rejected.
   - New and retired fields mutually exclusive.
   - A non-`None` new field is rejected unless `fpu_value == 0.0`.
2. Add `fpu_shipped_policy_mass_reduction`.
3. Confirm through the focused tests that the existing `policy_mass_fpu`
   helper requires no formula change; do not add a duplicate helper or wire
   the selection site yet.
4. Run focused config/formula suites.

**Gate:** pure tests green; no selection wiring yet.

## Task 3 — Wire the exact-zero selection branch

**Files:** `mcts.py`, v17 rule tests, existing FPU/MCTS tests.

TDD sequence:

1. Failing discriminator: a positive v17 coefficient changes an unvisited
   child's score on a synthetic tree.
2. Failing identity: `None` and `0.0` match the pre-change golden exactly.
3. Implement the branch before explored-mass calculation, calling
   `policy_mass_fpu(config.fpu_value, explored_policy_mass(node), r)` only in
   the positive branch.
4. Prove neither `None` nor `0.0` calls `policy_mass_fpu` or scans explored
   mass.
5. Run all MCTS, v16 formula, observer, and v17 tests.

**Gate:** v16 tests remain byte-identical and exact-zero proof passes.

## Task 4 — Add v17 provenance/config schemas

**Files:** create new v17 modules rather than modifying the frozen producer:

- `scripts/GPU/alphazero/fpu_v17_provenance.py`
- `scripts/GPU/alphazero/fpu_v17_protocol.py`
- `tests/test_fpu_v17_protocol.py`

Requirements:

- Versioned schemas.
- Formula ID and exact grid.
- Exact batching triple `(14, 48, 8)` and rejection of all overrides.
- Frozen design SHA-1.
- Checkpoint/config/source/replay hashes.
- Seed-range and run-kind validation.
- Refuse v16 roots as output targets.
- Refuse tooling-smoke artifacts in scientific modes.
- Re-derive-and-byte-compare configs.
- Atomic write and refuse-overwrite-different behavior.
- `abcd` is a valid scientific run kind; every scientific run kind requires a
  clean worktree.

**Gate:** protocol/config emit twice byte-identically and tamper tests fail
before evaluator load.

## Task 5 — Add the paired v17 diagnostic

**Files:**

- Create `scripts/GPU/alphazero/diagnose_fpu_baseline_policy_mass.py` for the
  v17 CLI/protocol layer.
- Modify `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` only to
  parameterize existing pure gates with defaults preserving v16 behavior.
- Create `tests/test_fpu_v17_diagnostic.py` and extend existing diagnostic
  regression tests.

Required modes:

- `tooling_smoke`
- `development`
- `held_out`
- `abcd`

Required behavior:

- Exact config sets per mode.
- Reject any diagnostic config whose effective batching triple is not
  `(14, 48, 8)`, including free CLI flag overrides.
- `r=0` identity prerequisite before positive interpretation.
- Per-position selected moves, priors/ranks, final parent/root values, top
  share, effective children, replies, collapse, lock-in, explored mass, and
  stabilization.
- Complete paired rows versus shipped.
- Pure gate functions implementing design §§7–9 exactly.
- Smallest-passing coefficient selector.
- Held-out mode accepts exactly one frozen coefficient.
- No A/B/C/D participation in coefficient selection.
- Import, rather than copy, the existing constants and pure definitions:
  `prior_rank`, `reply_reduction`, `top_share`, `lock_in_event`,
  `_percentile`, `selected_a_verdict`, and existing thresholds.
- Parameterize `dev_safety_verdict` with keyword defaults that preserve exact
  v16 behavior: v17 disables the per-band gate and uses lock-in margin one.
  Pin byte-identical legacy fixtures before and after the change.

**Gate:** fabricated end-to-end fixtures pin every threshold boundary and
wrong-mode failure.

## Task 6 — Add asymmetric same-checkpoint match support

**Files:** make minimal backward-compatible extensions:

- `scripts/GPU/alphazero/eval_runner.py`
- `scripts/GPU/alphazero/eval_checkpoint_match.py`
- `scripts/GPU/alphazero/eval_summary.py` (added by amendment
  `task6-agent-identity-scope-v1`, 2026-07-25 — see below)
- `tests/test_eval_search_config_match.py`

**Amendment `task6-agent-identity-scope-v1` (2026-07-25).**
Implementation-scope correction only; it changes no scientific decision, no
frozen design section, no grid, no sizing, and no gate threshold.

Adds `scripts/GPU/alphazero/eval_summary.py` to this task's permitted file
scope. Reason: scoring two agents that share one checkpoint requires
aggregation by agent identity, and `summarize_match` is where the
`a_ckpt == b_ckpt` self-match classification is made — the branch that nulls
`a_score`, `a_score_rate`, `elo_estimate`, `elo_ci95`, `score_rate_ci95` and
`verdict`. Left unamended, the v17 strength endpoint would be unobtainable
because every same-checkpoint match reports `None` for each of those fields.
The change is additive and gated on explicit agent identity, so the
checkpoint-based path and its artifact bytes are unchanged.

Authorized by the operator during Task 6 review, 2026-07-25. The as-of-freeze
plan SHA-1 recorded in
`logs/eval/fpu_v17_baseline_policy_mass/frozen_preregistration.json` is a
historical record of the version reviewed at freeze and is deliberately left
unmodified; that record already documents this plan as an execution document
outside the frozen protocol.

Requirements:

- Two agents may share identical checkpoint bytes but have different full MCTS
  configs.
- Add optional per-agent MCTS/search config arguments to
  `play_eval_game` and pairing-task construction; default `None` must preserve
  current behavior and artifact bytes.
- Keep `_make_cache` as an NN-evaluator cache; do not add a search-config cache
  key because it does not cache MCTS configuration.
- Exact color balance and deterministic tasks.
- No side-specific config leakage.
- Fabricated tests prove task/config symmetry, default-path identity, and
  correct color/config swapping. They do not infer playing strength.
- Complete match provenance records both effective configs.
- Both per-agent configs explicitly carry `(14, 48, 8)`; construction asserts
  equality with the diagnostic base config before evaluator load.

The exact real runner smoke is deferred to Task 8C, after the complete
software review. No scientific result may exist before this task and Task 7
are complete.

**Gate:** focused tests green before the full software review.

## Task 7 — Full software review before GPU

Run:

```bash
.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_fpu_baseline_policy_mass_rule.py \
  tests/test_fpu_v17_protocol.py \
  tests/test_fpu_v17_diagnostic.py \
  tests/test_fpu_policy_mass_rule.py \
  tests/test_fpu_diagnostic_modes.py \
  tests/test_fpu_trace_observer.py \
  tests/test_eval_search_config_match.py

.venv/bin/python -m pytest -p no:cacheprovider
```

Review:

- New field cannot activate accidentally.
- Retired field semantics unchanged.
- Exact-zero branch is structural.
- No import-time MLX in pure modules.
- No output path points into v16.
- Source hashes cover every result-determining module.
- Diagnostic and both match-agent batching triples equal `(14, 48, 8)`.
- Default match-runner behavior/artifact bytes remain unchanged, while
  per-agent configs are distinct, correctly color-swapped, and fully recorded.

**Gate:** focused and full suites green. Present results; request tooling-smoke
authorization.

## Task 8 — Full-chain, replay, and match-runner tooling smoke

### 8A — real 32-game CLI chain

After separate operator authorization, run the design §5.2 batch exactly:

1. Create and check the 32-game protocol with seeds
   `[20309000,20309032)`.
2. Generate with board 24, 400 simulations, four workers, replay capture, and
   the frozen match settings, including batching triple `(14, 48, 8)`.
3. Run qualification, qualification recheck, screen, post-screen
   qualification, and the actual selector CLI.
4. Select exactly one opening control and one early-mid control under the
   smoke profile.
5. Exercise one controlled failure, verify nonzero exit/no scientific
   artifact, then verify idempotent successful rerun hashes.

Every artifact must say `run_kind=tooling_smoke` and
`scientific_interpretation_forbidden=true`.

### 8B — replay diagnostic from tracked chain output

Use exactly the opening and early-mid controls selected by 8A, with configs
`{shipped, r=0, r=0.35}`. Do not read the untracked `smoke_v1` directory.

Required checks:

- `run_kind=tooling_smoke`.
- `scientific_interpretation_forbidden=true`.
- Shipped and `r=0` scientific-result payloads, tree signatures, and
  `search_result_sha1` byte-identical; only config/provenance labels differ.
- Two full reruns produce identical artifact hashes.
- Positive plumbing completes.
- No v16 production artifact changes.

### 8C — same-checkpoint match-runner smoke

Run the exact design §5.4 blocks:

1. Shipped versus shipped: four games, seeds `[20309100,20309104)`, exactly
   2/2 colors.
2. Shipped versus `r=0.35`: four games, seeds `[20309104,20309108)`, exactly
   2/2 colors/config assignments.
3. Use `calib020_0001` on both sides, 400 simulations, and
   `add_noise=false`, with both agent configs fixed to `(14, 48, 8)`.
4. Require correct effective configs, deterministic tasks, color/config
   swapping, and complete provenance. Do not require or interpret a 50% score.
5. Stamp every output `run_kind=tooling_smoke` and
   `scientific_interpretation_forbidden=true`.

**Amendment `task8-artifact-labelling-scope-v1` (2026-07-26).**
Implementation-scope correction only; it changes no scientific decision, no
frozen design section, no grid, no sizing, and no gate threshold.

The Task 8 artifact audit found emitted outputs missing `run_kind` and/or
`scientific_interpretation_forbidden`: the match summary, every per-game JSONL
row, all 32 replay sidecars, and the qualification report. Labelling them at
the point of emission — required so the stamp precedes qualification and
hashing — adds these files to Task 8's permitted scope:

- `scripts/GPU/alphazero/eval_replay.py` — builds the replay sidecar dict;
  the only place a per-replay label can be applied.
- `scripts/GPU/alphazero/eval_checkpoint_match.py` — summary + per-game rows,
  and the `--run-kind` / `--scientific-interpretation-forbidden` CLI flags.
- `scripts/GPU/alphazero/eval_runner.py` — threads labels to the replay writer
  and defines `ARTIFACT_LABEL_KEYS` once.
- `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py` — stamps the
  qualification report, derives the stamping flags in `emit-gen-command`, and
  tolerates the label keys in row reconstruction and summary binding.
- `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — `--mode select` returns the
  defined usage/IO exit code 2 for a missing or corrupt screen sidecar instead
  of an uncaught `FileNotFoundError` (exit 1, outside the exit-code contract);
  `V2Config` gains an OPTIONAL `scientific_interpretation_forbidden`
  (default `None`) so a schema-3 config round-trips through the re-derivation
  byte-compare, and the post-screen qualification report states the flag,
  derived from its run kind.
- `scripts/GPU/alphazero/fpu_v17_match_smoke.py` (new) — the §5.4 driver,
  promoted from a scratch script because it is result-determining; its SHA-1 is
  recorded in the protocol's `source_files`.
- `scripts/GPU/alphazero/fpu_v17_protocol.py` — config schema 1 → 2, adding an
  explicit `scientific_interpretation_forbidden` rather than leaving it implied
  by `scientific`. (Listed for completeness: this module postdates the Task 1
  snapshot, so the out-of-scope guard does not track it.)

All label stamping is opt-in and omitted by default, so unlabelled artifact
bytes are unchanged. Authorized by the operator during Task 8 review,
2026-07-26.

**Follow-up `task8-reservoir-schema-3-v1` (2026-07-26), authorized separately.**
Reservoir protocol schema 3, strictly additive:
`PROTOCOL_SCHEMA_KEYS` and `PROTOCOL_SCHEMA_KEYS_V2` are NOT modified;
`PROTOCOL_SCHEMA_KEYS_V3 = V2 + ("scientific_interpretation_forbidden",)`.
Schema 1/2 parsing, validation, derivation and artifact bytes are preserved
exactly, pinned against the real frozen v16 protocols (`smoke_v1`, `smoke_v2`,
`reservoir_v1`, and the 4,000-game production reservoir), each of which must
re-derive to its own bytes and must not gain the new key. Schema 3 requires an
exact boolean derived from `run_kind`; a contradictory value is refused. Also
in this follow-up: label validation at the `run_match` boundary, a required
`--scientific-interpretation {forbidden,allowed}` alongside `--run-kind`, and a
complete no-output assertion in the negative call-site proof.

Operator note: the config is bound to the source hashes recorded at
qualification, so ALL source edits must land BEFORE `qualify`. Editing a
recorded module afterwards makes `screen` refuse with a `source_file_sha1s`
mismatch, and the chain must be re-qualified from the protocol down.

**Gate:** tooling integrity only. Present results; do not discuss candidate
quality or select `r`. The 1,600-game protocol is blocked until 8A–8C all
pass.

## Task 9 — Build the fresh development protocol

After separate authorization:

1. Refuse to emit a scientific protocol unless the worktree is clean. Task 1
   and tooling smoke may have recorded a dirty tree; Stage 1 may not.
2. Emit protocol for exactly 1,600 games, seed range
   `[20310000,20311600)`, four workers, board 24, 400 simulations, replay
   capture, `add_noise=false`, batching triple `(14, 48, 8)`, no top-up.
3. Run preflight and `--check`.
4. Persist exact generation command.
5. Authenticate source and checkpoint identities.

**Amendment `task9-stage-identity-and-gate-b-v1` (2026-07-26).**
Implementation-scope correction only; it changes no scientific decision, no
frozen design section, no grid, no sizing, and no gate threshold.

1. **Stage identity.** The reservoir/corpus profile run kinds were
   `("production", "tooling_smoke")` only, so a scientific v17 reservoir could
   not carry its own stage identity. Emitting `production` instead is not a
   second legitimate axis: the Task 5 consumer refuses a selector config whose
   `run_kind` is not the diagnostic mode, so such a corpus would be unreadable
   AFTER the GPU cost was paid. Schema 3 therefore admits `development` and
   `held_out` in addition (`profile_run_kinds_for`); schemas 1 and 2 keep the
   original pair exactly, so no frozen v16 artifact can reach the widened set
   and none of their bytes change. Pinned by a producer-to-consumer test: a
   chain emitted for a stage authenticates under `authenticate_qualification`
   for that same stage, and a `production` chain is refused by the development
   consumer.
2. **Gate B pre-selection exclusion.** `forbidden_manifests` omitted gate B
   because its canonical cases CSV carries no `replay_path` column. Deferring B
   to the diagnostic could let the deterministic selector choose a forbidden B
   position and cause an avoidable Stage-1 gate failure. The loadable
   equivalent `logs/eval/tvc_v2_gate_B_goal_line_manifest.csv` (SHA-1
   `b678e4ed34816e2daeeb009e8deb274191248dc1`, 18 canonical hashes equal to the
   frozen B set) is added to the forbidden set. The diagnostic's independent
   canonical-B authentication is retained as the backstop.

**Amendment `task9-batching-completeness-v1` (2026-07-26).**
Implementation-scope correction only; no scientific decision, frozen design
section, grid, sizing or gate threshold changes.

Design §2.4 requires the COMPLETE batching triple to be explicitly derived,
recorded in every artifact and validated before evaluator loading. Through
schema 3 only two elements were explicit: `pending_virtual_visits` appeared in
no protocol, no command and no recorded config, and its value of 8 came solely
from `MCTSConfig`'s default. A change to that default would have silently
altered every v17 result.

Reservoir protocol/config schema 4, additive over schema 3 (schemas 1-3,
including the accepted Task 8 artifacts, keep their exact key sets and bytes):

- the protocol records `mcts_pending_virtual_visits`, validated as a
  non-negative int (`bool` refused);
- `gen_command` emits `--mcts-pending-virtual-visits` AND
  `--require-batching-triple`, so the command asserts its own complete triple;
- `EvalConfig` gains an optional `mcts_pending_virtual_visits`, passed
  explicitly into `MCTSConfig` by `cfg_from`; when unset the key is omitted
  from recorded config entirely, so legacy artifact bytes are unchanged;
- `eval_checkpoint_match` refuses, BEFORE checkpoint resolution and any
  evaluator load, if the effective triple differs in any element, and refuses
  `--require-batching-triple` without an explicit
  `--mcts-pending-virtual-visits` -- comparing the effective triple alone
  cannot distinguish a stated 8 from an inherited one.

**Gate:** clean worktree and protocol qualification pass before generation.

## Task 10 — Generate and qualify the development reservoir

Operator task:

1. Generate exactly the frozen 1,600 games.
2. Qualify match summary, JSONL, replay hashes, seeds, colors, and source.
3. Run the shipped-only raw-policy/anchor screen.
4. Run exact-selector feasibility for the design §6.2 32-row corpus.
5. Select only if feasible.

**Stop:** any mismatch, missing replay, shortfall, or selector failure.

**Gate:** exact 32-row manifest, 16 targets/16 controls, exactly 16/16 sides
overall, at most two positions per game with 12-ply spacing, all disjointness
checks green. Report role side counts; do not require unsupported role-local
balance.

## Task 11 — Run fresh coefficient selection

Operator task:

1. Run shipped and `r=0`.
2. Verify exact scientific-result and tree-signature identity. Stop on any
   mismatch other than the required config/provenance label.
3. Assert the diagnostic base config's batching triple is exactly
   `(14, 48, 8)`.
4. Run exactly `{0.15,0.20,0.25,0.35,0.45}`.
5. Persist all paired rows and gate metrics.
6. Select the smallest coefficient passing design §§7.2–7.3.

**Stop:** if none pass, reject v17. Do not generate held-out evidence.

**Gate:** immutable `selected_coefficient.json` bound to the complete
selection context. Present the complete result and request separate held-out
authorization.

## Task 12 — Build and run fresh held-out collateral

Only after Task 11 passes:

1. Require the worktree to remain clean and emit a separate 2,200-game
   protocol with seed range `[20312000,20314200)` and the same frozen
   `(14, 48, 8)` batching triple.
2. Do not inspect held-out outputs before the selected-coefficient artifact is
   finalized.
3. Generate, qualify, screen, and select the exact 56-row held-out corpus:
   24 targets and 32 controls, with the same two-position/12-ply constraint.
4. Run only shipped and the frozen coefficient.
5. Apply design §8.2 gates.

**Stop:** any collateral or mechanism-confirmation failure rejects v17.

## Task 13 — Run A/B/C/D once

Only after Task 12 passes:

1. Authenticate the fixed A/B/C/D manifests.
2. Authenticate the pre-change shipped goldens captured in Task 1.
3. Run shipped and the frozen coefficient with identical seeds/settings,
   including batching triple `(14, 48, 8)`.
4. Require the contemporary shipped result to reproduce every frozen case
   within `1e-6` with exact moves/counts; mismatch invalidates the stage.
5. Apply the exact design §9 candidate gates.
6. Persist per-case and aggregate comparisons.

**Stop:** any one gate failure rejects v17. Do not run a strength match.

## Task 14 — Primary same-checkpoint strength match

Run exactly 800 games with seeds `[20320000,20320800)` and 400 games per color.
Use the frozen opening-temperature and move-cap settings from design §10,
`add_noise=false` for both agents, batching triple `(14, 48, 8)` for both
agents, and the exact even/odd color schedule there. Assert equality with the
diagnostic base config before evaluator load.

Pass only if candidate score-rate CI95 lower bound is strictly greater than
`0.5`, using `score_ci_trinomial(w, d, l, z=1.96)`. Report score, W/D/L,
Elo/CI, termination reasons, and by-color results.

**Stop:** no top-up, sequential extension, or reinterpretation.

## Task 15 — Contemporary `0379` validation

Only after Task 14 passes:

1. Run frozen candidate vs `0379`, 800 games, seeds
   `[20330000,20330800)`, balanced colors, `add_noise=false`, and batching
   triple `(14, 48, 8)` for both agents.
2. Run shipped FPU vs `0379` with the identical seed/color schedule.
3. Before interpreting the candidate, require the shipped control's CI95 to
   overlap `0.588875`, its point score rate to be within `0.05` of that value,
   and its CI95 lower bound to exceed `0.5`; otherwise invalidate Stage 6.
4. Compute the deterministic 100,000-replicate, seed-`20260724` paired
   two-game-block bootstrap from design §11.
5. Apply design §11 candidate gates.

**Stop:** failure keeps shipped FPU.

## Task 16 — Final decision; no automatic adoption

Present:

- Every diagnostic/collateral/ABCD/strength result.
- Exact commands and source identities.
- Artifact paths and hashes.
- Determinism evidence.
- Deviations (any deviation invalidates preregistered inference).
- Limitations.

Even after every gate passes, changing self-play defaults requires a new,
explicit adoption decision. This plan never edits self-play or training
configuration automatically.

## Operator cost envelope

The estimated total is approximately 48–53 GPU-hours, extrapolated from the
approximately 30-hour 4,000-game production run:

- Task 8: about 0.30 hour.
- Tasks 9–11: about 12 hours plus diagnostics.
- Task 12: about 16.5 hours plus diagnostics.
- Tasks 14–15: about 18 hours for three 800-game matches.
- Task 13 and overhead: remaining margin.

Each operator authorization covers only the next listed block. Most spend
occurs after a coefficient has passed fresh development.
