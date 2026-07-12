# FPU Dev-Corpus v2 (Phase-Primary) — Design

**Status:** APPROVED, decisions frozen (2026-07-12). Successor to the v1 branching-band corpus, which is **corpus-independently impossible on a 24×24 board** (see the v1 design §11/§12 and §0 below). New versioned pipeline `fpu-dev-corpus-v2-phase` — NOT a patch to the v1 builder. The 4,800-game reservoir generation and all evaluator/MCTS phases are **operator phases**, NOT run by this implementation.

## 0. Why v2 (the impossibility that retired v1)

On a 24×24 board each side's legal region is exactly **528** cells and each placed peg removes at most one, so `n_legal ≥ 528 − ply` (verified: `min(n_legal+ply)=528` across 800 games, 0 violations). Therefore `n_legal ≤ 399 ⟹ ply ≥ 129` and `n_legal ≤ 299 ⟹ ply ≥ 229` — both low branching-bands lie **entirely in the late ply-bucket (ply ≥ 91)**. The v1 frozen composition needs 80+80 = 160 late rows, but the ≤50% ply-bucket cap on 240 rows permits ≤120. **Mathematically incompatible on any 24-board source, independent of seed/length/enumeration/FPU.** Raw `n_legal ≈ 528 − pegs` is essentially a restatement of game phase, so a *branching*-stratified corpus is degenerate. v2 makes **phase the primary stratum** and demotes branching to a recorded covariate + explicit late coverage floors.

## 1. Frozen decisions

### 1.1 Reservoir (operator-generated later)
4,800 fresh replay games, same matchup `calib020_0001` vs `0379`, a **predeclared seed range**, replay capture enabled. Treated as ONE fixed reservoir. **If qualification (screen + preflight) fails, STOP and version a new source protocol — never silently top up** the reservoir. Generation is an operator phase; this pipeline only consumes it.

### 1.2 Primary allocation — phase-stratified 240
Four phase strata = the existing ply buckets (`opening` 1–15, `early_mid` 16–40, `midgame` 41–90, `late` 91+). Each phase: **45 target + 15 control**, split **30/10 tuning + 15/5 frozen**:

`SPLIT_ALLOC_V2` (per phase p ∈ PHASES):
- `("target", p)`: `{"tuning": 30, "frozen_check": 15}`
- `("control", p)`: `{"tuning": 10, "frozen_check": 5}`

Totals: target 180 (45×4), control 60 (15×4); tuning 160 ((30+10)×4), frozen 80 ((15+5)×4); grand **240**. Preserves 180/60, 160/80, whole-game split, ≤2/game, ≥12-ply separation, side balance. The ≤50%-bucket cap is **subsumed** (each phase = 60/240 = 25%), and is dropped as an independent constraint in v2. `n_legal ≥ 200` remains an eligibility floor and is recorded, but is **not** an independent quota stratum.

### 1.3 Late coverage floors
Among the **45 late target rows**, require **≥ 12 with `n_legal ∈ 300–399`** AND **≥ 12 with `n_legal ∈ 200–299`**. `LATE_TARGET_FLOORS = {"b300_399": 12, "b200_299": 12}`. These are **coverage floors** — not independent selection strata and **not rate-gate denominators**. They exist so the v1 late-collapse geometry does not vanish now that phase is primary.

### 1.4 Safety stratification (the one v1-diagnostic change; opt-in, back-compatible)
The §6.2 stratified new-collapse sub-gate becomes stratum-parameterized: `dev_safety_verdict(..., *, stratum_key="band")`, iterating `by[stratum_key]` with the **frozen 10% threshold + n ≥ 20 rule UNCHANGED**. **The DEFAULT stays `"band"` (v1 byte-compatible — existing v1 rows/tests unaffected).** v2 **opts in explicitly**: the v2 config sets `new_collapse_stratum: "ply_bucket"`, and v2 manifests + candidate rows carry **BOTH `band` AND `ply_bucket`** so the diagnostic can stratify by phase while still recording bands for reporting. Every other §6 threshold, gate, the controls→candidates→frozen flow, the selected-A gate, and the evidence chain are unchanged.

### 1.5 Proposal enumerator (no global stride)
Deterministic, phase-aware. A **proposal cell** is where a side-opposed pair is drawn:
- non-late phases: one cell per phase (`opening`, `early_mid`, `midgame`);
- late phase: **three distinct cells** by `n_legal` range — `late×b400_plus`, `late×b300_399`, `late×b200_299` — so the low-branching floors are actually reachable.

Per (game, proposal-cell): draw a **side-opposed pair** (one red-to-move ply + one black-to-move ply) **≥ 12 plies apart within that cell**, **capped at 2 proposals per cell per game**. No global stride; `--stride 4` is not a default anywhere. The final sampler still enforces the global ≤2/game rule across all cells.

### 1.6 Two-artifact source workflow — separate `screen` and `select` stages
1. The fixed reservoir (§1.1).
2. `fpu_dev_source_screen` — a complete, **fingerprinted** artifact persisting **every proposed root**. It is the **reusable evidence artifact, reviewable before final selection.**

**`screen` and `select` are SEPARATE operator invocations — `screen` never auto-selects a manifest.**

- **`screen` (evaluator/MCTS):** for each proposal, reconstruct the state, compute `canonical_sha1`, then apply the **cheap filters BEFORE the expensive anchor** — collision (`sha1 ∈ forbidden ∪ kept`) and `raw_policy_role`; only proposals that pass BOTH receive the **400-sim fpu-off anchor**. Persist EVERY proposal with `exclusion_status ∈ {kept, ineligible_role, collision, ineligible_anchor}` and an explicit **`anchor_run: bool`** — `false` for `ineligible_role`/`collision` (rejected pre-anchor; their `root_value_stm`/`anchor_eligible` fields are **null**), `true` for `kept`/`ineligible_anchor`. Screen row schema: `game_idx, ply, side, phase, n_legal, band, ply_bucket, proposal_cell, normalized_entropy, top1_prior, top4_mass, top8_mass, raw_policy_role, anchor_run, root_value_stm(nullable), anchor_eligible(nullable), canonical_sha1, exclusion_status`. Writes `fpu_dev_source_screen.csv` + `.meta.json` with the §1.8 fingerprints. **Screening never stops early because a reserve filled** — every proposal is screened and recorded.
- **`select` (PURE — no evaluator):** hard-match the screen `.meta.json` identities against the config + inputs (§1.7), run the post-screen qualification (§1.7), then deterministically select the final manifest via the sampler and `assert_disjoint`. Re-runnable and reviewable from the persisted screen alone.

### 1.7 Two-stage feasibility (geometric preflight + post-screen qualification)
Role (target/control) depends on raw policy + the fpu-off anchor, so it is **not** provable from geometry. Feasibility is proven in two stages:

- **Geometric preflight (PURE, pre-screen, role-AGNOSTIC):** over the exact v2 proposal set, prove **jointly** via a constructive witness: per-phase **candidate** capacity (60/phase), late **candidate** availability per band cell (enough `late×b300_399` and `late×b200_299` proposals to *potentially* meet the floors, ignoring role), ≤2/game, ≥12-gap, per-split side balance, whole-game split. Emits witness + binding constraint; gates before any evaluator loads. It does **NOT** prove target-role floors and does **NOT** prove disjointness (both need per-state work done at screen/select).
- **Post-screen qualification (PURE, in `select`):** from the screen's `kept` rows (role + `anchor_eligible` now known), prove the exact `SPLIT_ALLOC_V2` **roles** (45 target / 15 control per phase) AND the **≥12 / ≥12 late-target floors** are satisfiable, before manifest selection. Disjointness is enforced here (via the screen's `canonical_sha1` + `assert_disjoint`). A geometry that passes the geometric preflight can still fail qualification (e.g. enough late-`b200_299` *candidates* but too few classify as *target*) — that is caught here, not in the preflight.

### 1.8 Versioned config file (required)
An explicit `fpu_dev_corpus_v2_config.json` names: source reservoir path + seed range, selection seed, phase allocation, late floors, proposal-enumerator policy parameters, and all fingerprints (source-index hash, replay-data hash, source-file hashes, runtime provenance via `fpu_provenance`). **The builder REQUIRES this file** — no default source, no default stride.

## 2. Architecture & reuse (DRY — import, don't duplicate)

New module `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` holds the v2-specific constants, the phase-aware enumerator, the phase-stratified sampler (with late floors), the v2 preflight, the source-screen persistence, the config loader, and the operator `main`. It **imports and reuses**:
- from `build_fpu_dev_corpus`: `ply_bucket_of`, `band_of`, `side_to_move_for_ply`, `raw_policy_role`, `anchor_eligible`, `per_ply_n_legal`, `_policy_features_from_priors`, `load_forbidden_hashes`, `assert_disjoint`, and the witness machinery pattern (`_build_witness`/`geometry_feasibility`) as the base the v2 preflight extends;
- from `fpu_state_hash`: `canonical_state_sha1`;
- from `fpu_provenance`: `source_file_sha1s`, `replay_data_sha1`, `file_sha1`, `git_commit`, `worktree_clean`, `runtime_provenance`;
- the fpu-off anchor + raw-policy forward pass mirror the v1 shell (lazy imports; import-pure without GPU/MLX).

The one edit outside the new module: `diagnose_fpu_policy_mass.py` `dev_safety_verdict` stratum-key parameterization (§1.4).

## 3. Non-goals / unchanged
The FPU rule/observer/hash, the controls→candidates→frozen diagnostic flow, the selected-A gate, all §6 frozen thresholds, and the evidence chain are unchanged (except §1.4's stratum swap). No self-play adoption. No reservoir generation or evaluator run in this branch.

## 4. Progression
```
predeclare seed range → generate 4,800-game reservoir (operator)
  → enumerate v2 proposal set → build fpu_dev_source_screen (operator; persists ALL proposals)
  → v2 preflight (pure) MUST pass jointly incl. late floors; else STOP + version a new source
  → deterministic final-manifest selection from the screen (v2 sampler)
  → controls→candidates→frozen sweep (diagnostic UNCHANGED except per-phase new-collapse gate)
  → ... (unchanged downstream: frozen check → strength match)
```
