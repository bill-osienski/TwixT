# FPU Dev-Corpus v2 (Phase-Primary) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the *tooling* for the v2 phase-primary FPU development-corpus qualification pipeline — a phase-aware proposal enumerator, a **role-agnostic geometric preflight**, a separate **`screen`** stage (evaluator; persists every proposal), a separate **pure `select`** stage (identity hard-match + **post-screen role/floor qualification** + deterministic selection), a phase-stratified sampler enforcing the late coverage floors, a required versioned config, and a stratum-parameterized new-collapse gate (default `band`; v2 opts into `ply_bucket`) — all verified. The 4,800-game reservoir and all evaluator/MCTS runs are operator phases, NOT run here.

**Architecture:** One new module `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` (constants + pure enumerator/sampler/geometric-preflight/qualification/select + operator `screen` `main` + config loader) that IMPORTS v1 helpers (classifiers, witness machinery, provenance, hash) rather than duplicating them; plus one localized, **back-compatible** edit to `diagnose_fpu_policy_mass.py` (`dev_safety_verdict` stratum-key param, default `"band"`). Feasibility is proven in two stages — geometry (pre-screen, role-agnostic) then post-screen qualification (role-aware floors) — because target/control depends on the evaluator. All heavy MCTS work stays behind the operator `screen` entry point, exercised only via pure functions + synthetic geometry in tests.

**Tech Stack:** Python 3 stdlib (dataclasses, json, csv, hashlib, argparse) + numpy. Tests via `.venv/bin/python -m pytest -p no:cacheprovider`. Frozen spec: `docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md`.

## Global Constraints

- **New versioned pipeline, not a patch.** New module `fpu_dev_corpus_v2.py`; do NOT alter the v1 `build_fpu_dev_corpus.py` behavior except by importing its pure helpers. The v1 builder/preflight remain as-is.
- **Reuse, don't duplicate (DRY):** import `ply_bucket_of, band_of, side_to_move_for_ply, raw_policy_role, anchor_eligible, per_ply_n_legal, _policy_features_from_priors, load_forbidden_hashes, assert_disjoint` from `build_fpu_dev_corpus`; `canonical_state_sha1` from `fpu_state_hash`; `source_file_sha1s, replay_data_sha1, file_sha1, git_commit, worktree_clean, runtime_provenance` from `fpu_provenance`.
- **Frozen allocation:** `SPLIT_ALLOC_V2` per phase — target {tuning:30, frozen_check:15}, control {tuning:10, frozen_check:5}; 4 phases → 240 = 180 target + 60 control, 160 tuning / 80 frozen.
- **Late coverage floors:** among the 45 late TARGET rows, ≥12 `n_legal 300–399` AND ≥12 `n_legal 200–299`. Coverage floors — NOT strata, NOT rate-gate denominators.
- **Enumerator:** no global stride; per (game, proposal-cell) a side-opposed pair ≥12 plies apart, cap 2/cell/game; late split into 3 cells (b400_plus/b300_399/b200_299). Final sampler enforces global ≤2/game.
- **Separate `screen` and `select` stages — never combined in one invocation.** `screen` (evaluator): cheap filters (collision + `raw_policy_role`) BEFORE the 400-sim anchor; persists EVERY proposal (kept + excluded + ineligible) with an explicit `anchor_run: bool` (`false` + null anchor fields for `ineligible_role`/`collision`; `true` for `kept`/`ineligible_anchor`). Screening never stops early on a filled reserve. `select` (PURE, no evaluator): validates identities, qualifies, selects.
- **Two-stage feasibility.** (a) **Geometric preflight** (pure, pre-screen, role-AGNOSTIC): phase *candidate* capacity, late *candidate* availability per band cell, ≤2/game, ≥12-gap, side balance, whole-game split — via a constructive witness (`feasible=True` never without a witness). It does NOT prove target-role floors and does NOT prove disjointness. (b) **Post-screen qualification** (pure, in `select`): from the screen's `kept` rows, prove the exact 45/15 phase ROLES + the ≥12/≥12 late-TARGET floors; disjointness enforced here.
- **Config file required** — no default source/stride. `select` **hard-matches** the screen `.meta.json` identities (config hash, source-index + replay-data hash, checkpoint hash, source-file hashes, forbidden-manifest hashes) and aborts on any mismatch.
- **Diagnostic change (§1.4):** `dev_safety_verdict(..., *, stratum_key="band")` — **DEFAULT stays `"band"` (v1 byte-compatible)**; v2 opts in via config `new_collapse_stratum="ply_bucket"`; v2 manifests/candidate rows carry BOTH `band` AND `ply_bucket`. **10% threshold + n≥20 UNCHANGED**; all other §6 thresholds/gates/evidence-chain byte-identical.
- **Import-pure:** `fpu_dev_corpus_v2` imports without GPU/MLX. No `main()`/MCTS/operator run in tests; synthetic geometry + fabricated screens only.
- Tests always `-p no:cacheprovider` from repo root. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. File-scoped commits.

---

## Task 0: Impossibility regression test (the WHY, and a guard)

**Files:** Test `tests/test_fpu_dev_corpus_v2.py` (create).

- [ ] **Step 1: Failing test** — prove the v1 200/300/400 + ≤50%-bucket design is impossible on a 24-board, from the `n_legal ≥ 528 − ply` invariant (pure; no evaluator). Assert: empty-board `len(TwixtState(active_size=24,to_move="red").legal_moves()) == 528` (both sides); for a range of plies, any position with `n_legal ≤ 399` has `ply ≥ 129` (i.e. `528 − 399`), so it is in `ply_bucket_of == "late"`; therefore the two low bands (80+80=160 rows) exceed the 120-row ≤50% cap. Encode as a `test_v1_bands_impossible_on_24board()` asserting `528 - 399 == 129 >= 91` and `80 + 80 > int(0.5*240)`.
- [ ] **Step 2: Run → fail** (test file absent). **Step 3:** it's a pure arithmetic/geometry assertion — no source needed. **Step 4: Run → pass. Step 5: Commit** — `test(fpu-v2): regression proving v1 branching-band+50%-cap corpus is impossible on a 24 board`.

---

## Task 1: v2 constants + classifiers

**Files:** Create `scripts/GPU/alphazero/fpu_dev_corpus_v2.py`; Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces (Produces):** `PHASES=("opening","early_mid","midgame","late")`; `SPLIT_ALLOC_V2`; `LATE_TARGET_FLOORS={"b300_399":12,"b200_299":12}`; `PROPOSAL_CELLS` (list of `(phase, band_or_None)`: the three non-late phases with band `None`, plus `("late","b400_plus")`,`("late","b300_399")`,`("late","b200_299")`); `MAX_PER_CELL_PER_GAME=2`, `MIN_PLY_GAP=12`, `MAX_PER_GAME=2`, `SIDE_TOL=2`, `CORPUS_SIZE=240`; `proposal_cell_of(phase, n_legal)`.

- [ ] **Step 1: Failing tests** — `SPLIT_ALLOC_V2` sums to 240/180-60/160-80 (assert each derived total); `LATE_TARGET_FLOORS` values; `PROPOSAL_CELLS` has exactly the 6 cells; `proposal_cell_of("opening",520)==("opening",None)`, `proposal_cell_of("late",520)==("late","b400_plus")`, `proposal_cell_of("late",350)==("late","b300_399")`, `proposal_cell_of("late",250)==("late","b200_299")`. Import `ply_bucket_of, band_of` from `build_fpu_dev_corpus` and assert they're reused (e.g. `band_of` boundaries unchanged).
- [ ] **Step 2: Run → fail. Step 3: Implement** the constants + `proposal_cell_of` (uses `band_of` for the late split). **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): phase-primary constants, SPLIT_ALLOC_V2, late floors, proposal cells`.

---

## Task 2: Phase-aware proposal enumerator

**Files:** Modify `fpu_dev_corpus_v2.py`; Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces:** `enumerate_v2_proposals(replay) -> List[dict]` — per game, per `PROPOSAL_CELLS` cell, select a **side-opposed pair** (one red-to-move ply + one black-to-move ply, via `side_to_move_for_ply`), both eligible for that cell (`ply_bucket_of(ply)==phase` and, for late cells, `band_of(n_legal)==band`; `n_legal>=200`), **≥`MIN_PLY_GAP` apart**, capped at `MAX_PER_CELL_PER_GAME` proposals per cell per game. Each proposal dict: `game_idx, ply, side, phase, n_legal, band, proposal_cell`. Deterministic (choose the earliest-satisfying pair per cell in ascending ply).

- [ ] **Step 1: Failing tests** — synthetic replay (`{"moves":[{"n_legal":...}]}`, `game_idx`): a cell yields a red+black pair ≥12 apart (assert the two sides differ and `|Δply|>=12`); ≤2/cell/game (a cell with many eligible plies still yields ≤2); a cell with no valid opposed-gap pair yields 0 for that cell; late cells only select plies whose `band_of(n_legal)` matches the cell's band (so `late×b300_399` picks a 300–399 ply, reachable only at ply≥129 — build the synthetic n_legal accordingly); determinism (same replay → same proposals).
- [ ] **Step 2: Run → fail. Step 3: Implement.** **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): phase-aware side-opposed proposal enumerator (no global stride; late band cells)`.

---

## Task 3: v2 phase-stratified sampler + late floors

**Files:** Modify `fpu_dev_corpus_v2.py`; Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces:** `sample_v2_rows(kept, *, seed) -> (rows, stats)` — `kept` = the KEPT screen rows (each carrying `game_idx, role, phase, band, side, ply, canonical_sha1`). Deterministically select exactly `SPLIT_ALLOC_V2` per `(role, phase, split)` under whole-game split isolation, `MAX_PER_GAME`≤2, `MIN_PLY_GAP`≥12, per-split side balance ≤`SIDE_TOL`, no duplicate `canonical_sha1`, AND the late target rows satisfy `LATE_TARGET_FLOORS` (≥12 in b300_399, ≥12 in b200_299). Raise `ValueError` on any shortfall (incl. an unmet floor).

- [ ] **Step 1: Failing tests** — with an abundant fabricated `kept` pool: exact per-`(role,phase,split)` composition (Counter equals `SPLIT_ALLOC_V2`); totals 160 tuning/80 frozen, 180 target/60 control, 240; both late floors met (≥12/≥12 among late target); whole-game split isolation; ≤2/game; ≥12-gap; per-split side balance ≤2; no dup hash; determinism. Shortfall tests: an insufficient pool AND a pool that meets phase quotas but CANNOT meet a late floor each raise `ValueError` (the floor is a hard requirement). Reuse the v1 `_abundant_pool`/greedy patterns as a model (they were mutation-verified); adapt to phases + floors.
- [ ] **Step 2: Run → fail. Step 3: Implement** (mirror v1 `assign_split`/`sample_dev_rows` structure — whole-game assignment + round-robin — extended to phase strata and a floor-satisfaction pass on the late-target cell). **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): phase-stratified sampler with hard late coverage floors`.

---

## Task 4: v2 GEOMETRIC preflight (role-agnostic; + witness)

**Files:** Modify `fpu_dev_corpus_v2.py`; Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces:** `v2_geometry_feasibility(proposals_by_game, *, ...) -> V2PreflightReport(feasible, binding_constraint, diagnostics)` and `v2_preflight_source(records) -> V2PreflightReport` (I/O wrapper reading replays → `enumerate_v2_proposals`). Prove JOINTLY, **role-AGNOSTIC**, via a constructive witness (extend the v1 `_build_witness` pattern to phase cells): per-phase **candidate** capacity (60 selectable per phase), late **candidate** availability per band cell (enough `late×b300_399` and `late×b200_299` proposals to *potentially* meet the ≥12/≥12 floors, IGNORING role), ≤2/game, ≥12-gap, per-split side balance, whole-game split. **Does NOT prove target-role floors and does NOT prove disjointness** — both are the post-screen `select` step (Task 6). `feasible=True` ONLY via a completed witness (never false-feasible); report the binding constraint + diagnostics on failure.

- [ ] **Step 1: Failing tests** — synthetic proposal geometry: (a) a phase short of 60 candidates → infeasible, binding names the phase; (b) a late **candidate** band unreachable (zero `late×b200_299` proposals) → infeasible, binding names that candidate cell; (c) one-side-only in a cell → infeasible (side); (d) ≤2/game exceeded → infeasible; (e) feasible geometry → `feasible=True` with a witness satisfying all constraints (assert on the witness); (f) soundness: all per-constraint necessary checks pass but jointly infeasible → `feasible=False`; (g) determinism. **Explicitly assert the geometric preflight does NOT claim target floors or disjointness** (a geometry with ample late candidates but where roles/disjointness would fail still passes here — that's Task 6's job).
- [ ] **Step 2: Run → fail. Step 3: Implement.** **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): role-agnostic geometric preflight (phase capacity + late candidate availability, witness)`.

---

## Task 5: `screen` stage (operator) + schema

**Files:** Modify `fpu_dev_corpus_v2.py`; Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces:** `SCREEN_FIELDNAMES` = `game_idx, ply, side, phase, n_legal, band, ply_bucket, proposal_cell, normalized_entropy, top1_prior, top4_mass, top8_mass, raw_policy_role, anchor_run, root_value_stm, anchor_eligible, canonical_sha1, exclusion_status` (`phase == ply_bucket`; rows carry BOTH `band` and `ply_bucket`). Pure `classify_exclusion(*, collided, role, anchor_eligible_val) -> (exclusion_status, anchor_run)` and `screen_row(proposal, *, feats, role, anchor_run, root_value_stm, anchor_eligible, canonical_sha1, exclusion_status) -> dict`. Operator `run_screen(config)` / `main --mode screen`: per proposal from `enumerate_v2_proposals` over the reservoir → reconstruct state → `canonical_state_sha1` → **cheap filters BEFORE the anchor**: if `sha1 ∈ forbidden ∪ kept` → `collision` (`anchor_run=false`, null anchor fields); else raw-policy pass (`_policy_features_from_priors`) + `raw_policy_role` → if `None` → `ineligible_role` (`anchor_run=false`, null anchor); else run the **400-sim fpu-off anchor** (`search_with_root`, `MCTSConfig(fpu_policy_mass_reduction=None)`) → `anchor_run=true`, `anchor_eligible(root_value_stm)` → `kept` or `ineligible_anchor`. **Persist EVERY proposal**, never stopping early; write `fpu_dev_source_screen.csv` + `.meta.json` (config hash + §1.8 fingerprints: `source_file_sha1s, replay_data_sha1, source_index_sha1, checkpoint hash, forbidden-manifest hashes, runtime_provenance`).

- [ ] **Step 1: Failing tests (pure only)** — `classify_exclusion`: `collided=True → ("collision", anchor_run=False)`; `collided=False, role=None → ("ineligible_role", False)`; `role set, anchor_eligible_val=False → ("ineligible_anchor", True)`; `role set, anchor_eligible_val=True → ("kept", True)`. `screen_row` yields the full `SCREEN_FIELDNAMES` schema with BOTH `band` and `ply_bucket`, and **null `root_value_stm`/`anchor_eligible` when `anchor_run=False`**. Do NOT invoke `main`/MCTS. Confirm `import scripts.GPU.alphazero.fpu_dev_corpus_v2` loads with `mlx` NOT in sys.modules.
- [ ] **Step 2: Run → fail. Step 3: Implement** (operator `run_screen`/`main` mirrors the v1 shell's lazy checkpoint/evaluator; never run here). **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): screen stage — cheap filters before anchor, anchor_run+nullable fields, persists every proposal`.

---

## Task 6: `select` stage (PURE) — hard-match identities + post-screen qualification + selection

**Files:** Modify `fpu_dev_corpus_v2.py`; Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces (all pure — no evaluator):**
- `load_v2_config(path) -> V2Config` — raise if missing any required key (source reservoir + seed range, selection seed, allocation, floors, enumerator params, `new_collapse_stratum`, expected fingerprints).
- `validate_screen_identities(screen_meta, config, *, forbidden_paths) -> None` — **hard-match and raise on ANY mismatch:** config hash, `source_index_sha1`, `replay_data_sha1`, checkpoint hash, `source_file_sha1s`, forbidden-manifest hashes.
- `post_screen_qualification(kept_rows) -> None` — prove (raise on failure) the exact `SPLIT_ALLOC_V2` **roles** (45 target / 15 control per phase) AND the ≥12/≥12 late-**target** floors are satisfiable from the screen's `kept` rows (role + `anchor_eligible` known). This is where role-dependent floor feasibility is proven (the geometric preflight could not).
- `select_final_manifest(screen_rows, screen_meta, config, *, forbidden) -> (manifest_rows, stats)` — `validate_screen_identities` → filter `kept` → `post_screen_qualification` → `sample_v2_rows(kept, seed=config.selection_seed)` → `assert_disjoint(hashes, forbidden)` → manifest rows carry BOTH `band` and `ply_bucket`.
- Operator `main --mode select` REQUIRES `--config` + `--screen`; **`screen` and `select` are never the same invocation.**

- [ ] **Step 1: Failing tests (pure)** — `load_v2_config` raises on a missing required key, loads a complete one. `validate_screen_identities` raises when ANY of the six identity hashes differs, passes when all match (fabricated meta+config). **`post_screen_qualification` RAISES when a late-target floor is unmeetable even though geometry was fine** — the correction-1 test: a `kept` screen with ample `late×b200_299` *candidates* but too few classified `target` (so <12 late-target b200_299) → raise; and passes on a qualifying screen. `select_final_manifest` is **deterministic** (same screen+seed → identical manifest = screen-cache reproducibility), yields the exact v2 composition + floors, and refuses (raises) on failed identities OR failed qualification before any selection. `main --mode select` rejects a missing `--config`/`--screen`.
- [ ] **Step 2: Run → fail. Step 3: Implement.** **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): pure select stage — identity hard-match, post-screen role/floor qualification, deterministic selection`.

---

## Task 7: Diagnostic new-collapse gate — band→phase stratum

**Files:** Modify `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py`; Test `tests/test_fpu_evidence_chain.py` (extend).

**Interfaces:** `dev_safety_verdict(rows, ref, r0_lockin, absoff_lockin, *, stratum_key="band")` — the stratified new-collapse sub-gate iterates `by[stratum_key]` (was hardcoded `r["band"]`), keeping `DEV_NEW_COLLAPSE_BAND=0.10` + `DEV_BAND_MIN_N=20` unchanged. **The DEFAULT stays `"band"` so v1 stays byte-compatible** (v1 rows carry only `band`; existing tests unaffected). **v2 opts in explicitly**: the diagnostic reads the stratum from the v2 config's `new_collapse_stratum: "ply_bucket"` and passes `stratum_key="ply_bucket"`; v2 manifests/candidate rows carry BOTH `band` and `ply_bucket`. Bands still recorded in `metrics` for reporting under either stratum.

- [ ] **Step 1: Failing tests** — **default (`band`) is byte-identical:** pin an existing case's `rejected`/`reasons` (unchanged) and confirm the existing evidence-chain/diagnostic tests stay green with no call-site changes. **Opt-in (`stratum_key="ply_bucket"`)**: a per-phase new-collapse case (an n≥20 phase with rate ≥0.10) rejects with a `ply_bucket[...]_new_collapse` reason; the **10% threshold + n≥20 boundary are unchanged** (0.10 rejects / 0.0999 passes; n=20 active / n=19 inactive). A row set lacking the requested stratum key raises a clear error (rather than silently skipping the gate).
- [ ] **Step 2: Run → fail. Step 3: Implement** (parameterize the one loop; default `"band"`; thresholds + everything else identical). **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): stratum-parameterized new-collapse gate (default band; v2 opts into ply_bucket)`.

---

## Task 8: Full suite + integration; stop for review

**Files:** none (verification).

- [ ] **Step 1** — `tests/test_fpu_dev_corpus_v2.py` + `tests/test_fpu_evidence_chain.py` green; confirm all six regression areas covered: **old impossibility** (T0), **v2 phase quotas** (T3), **late floors** (T3 sampler enforces + T6 qualification proves, incl. the *geometry-passes-but-role-fails-the-floor* case), **proposal side/gap behavior** (T2), **whole-game split** (T3), **screen-cache reproducibility** (T6 deterministic `select`). Also confirm the v1 diagnostic default (`stratum_key="band"`) left existing tests byte-identical (T7).
- [ ] **Step 2** — `import scripts.GPU.alphazero.fpu_dev_corpus_v2` with `mlx` not loaded; byte-compile.
- [ ] **Step 3** — full suite `.venv/bin/python -m pytest -p no:cacheprovider tests/ -q` green vs the current baseline.
- [ ] **Step 4** — STOP for review before the operator generates the 4,800-game reservoir. Do NOT run any operator MCTS phase.

## Self-review — spec coverage

| Spec §1 decision | Task |
|---|---|
| 1.1 reservoir / stop-don't-topup (no silent top-up) | `select`/`screen` require the config + identity hard-match (T5/T6); operator generates the reservoir (not here) |
| 1.2 phase allocation 30/10/15/5 | T1 constants, T3 sampler |
| 1.3 late floors ≥12/≥12 (coverage, not strata) | T3 sampler **enforces** (hard, raises); T6 **post-screen qualification proves feasible**; T4 only proves late *candidate* availability |
| 1.4 stratum-parameterized new-collapse gate (default `band`; v2 opts into `ply_bucket`; rows carry both) | T7 (+ T5/T6 rows carry both `band` and `ply_bucket`) |
| 1.5 proposal enumerator (opposed pair, 6 cells incl. late band split, no stride) | T1 cells, T2 enumerator |
| 1.6 `screen`/`select` separate; screen persists ALL; `anchor_run` + null anchor fields; cheap filters before anchor | T5 (screen), T6 (select) |
| 1.7a geometric preflight (role-agnostic, witness; NOT floors, NOT disjointness) | T4 |
| 1.7b post-screen qualification (exact 45/15 roles + late-target floors) + disjointness | T6 |
| 1.8 required versioned config; `select` hard-matches all identity hashes | T6 |
| impossibility regression (v1 bands + 50% cap on a 24-board) | T0 |
