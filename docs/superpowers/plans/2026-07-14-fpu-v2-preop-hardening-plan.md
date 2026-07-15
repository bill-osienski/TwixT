# FPU v2 Pre-Operator Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task, one fresh subagent per task with two-stage review between tasks. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the two pre-reservoir gaps found in the post-merge audit of the merged `fpu-dev-corpus-v2-phase` tooling — (Group 1) wire the FPU diagnostic's phase-stratum knob end to end, and (Group 2) make `fpu_dev_corpus_v2_config.json` *enforce* the frozen reservoir decisions via a zero-GPU protocol → qualify → immutable-config lifecycle — so the pipeline is ready for the operator to generate the 4,800-game reservoir.

**Architecture:** Group 1 is a bounded edit to `diagnose_fpu_policy_mass.py` (consumes the config). Group 2 is a new pure module `fpu_dev_reservoir_protocol.py` (produces the config) plus a `run_screen` precheck hook and additive `V2Config` fields. Two clearly-separated phases on one branch, separately reviewed, ending in one combined pre-operator audit. Frozen spec: `docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md`.

**Tech Stack:** Python 3 stdlib (dataclasses, json, csv, hashlib, argparse) + numpy. Tests via `.venv/bin/python -m pytest -p no:cacheprovider`. Branch `fpu-v2-preop-hardening` off main @ `c92da41` (code baseline 053650a; full suite **1777 passed / 4 skipped / 0 failed**).

## Global Constraints

- **TOOLING ONLY.** No evaluator/MCTS/GPU/checkpoint, and **no reservoir generation**. `qualify`/`emit-*`/`run_screen` must EXIST and be unit-tested via pure functions + fabricated data + injected fakes, never invoked on real data. STOP before the reservoir.
- **v1 byte-identical.** `build_fpu_dev_corpus.py` and `build_fpu_dev_corpus._CORPUS_SOURCES` stay byte-identical. The v1 diagnostic path (no `--dev-corpus-config`) stays byte-identical: existing `dev_safety_verdict` default = `"band"`, existing `build_run_fingerprint` output unchanged (new fingerprint keys **omitted entirely** for v1, never `None`), existing persisted artifacts (`controls_cases.csv`, `candidate_dev_rows.csv`, `controls_gate.json`) unchanged.
- **Config is `derive(protocol, reservoir)`** — a pure deterministic function. The pre-screen/select config-tamper check is **re-derive + byte-compare**, not embedded-hash rechecks (spec §5).
- **Only GATE-FAIL condition** is a protocol-faithful reservoir with infeasible geometry (spec §4.2). Every structural defect (colors, linkage, contiguity, seed, MCTS config, summary mismatch, path derivation, provenance) is a **MISMATCH**.
- **Exit codes:** `0` ok · `3` MISMATCH · `4` GATE-FAIL · `2` usage/IO.
- **Frozen values:** `new_collapse_stratum == "ply_bucket"` for v2; anchor = checkpoint-A = calib020_0001; seed range half-open `[base_seed, base_seed + games)`, `seed == base_seed + game_idx`; games protocol-declared (production 4,800, tests 6); the ten result-determining match knobs = `board_size, mcts_sims, mcts_eval_batch_size, mcts_stall_flush_sims, selection_mode, opening_temp_plies, temp_high, temp_low, max_moves, base_seed`; `DEV_NEW_COLLAPSE_BAND=0.10`, `DEV_BAND_MIN_N=20` unchanged.
- **Circular import (spec §6, PINNED):** `run_screen` lazily imports the new module inside the function body; the new module top-level-imports `fpu_dev_corpus_v2` only for the shared config schema-key constant and treats the config as a duck-typed mapping otherwise.
- **Import-purity:** importing `fpu_dev_corpus_v2` and `fpu_dev_reservoir_protocol` must NOT load GPU/MLX.
- **Canonical JSON** for all emitted artifacts: sorted keys, fixed numeric formatting, no timestamps/RNG → byte-reproducible.
- **File-scoped commits only** (`git add <exact files>`; never `-A`). Trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Run tests `-p no:cacheprovider` from repo root.

## File Structure

- **Create** `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py` — pure protocol/qualification/config-derivation + CLI (Group 2).
- **Modify** `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — Group 1 (ply_bucket propagation, `--dev-corpus-config`, stratum threading, fingerprint keys).
- **Modify** `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — Group 2 (extend `V2Config` + `load_v2_config`; add the qualification module to `_V2_CORPUS_SOURCES`; the `run_screen` precheck hook; extend the select recheck).
- **Test** `tests/test_fpu_evidence_chain.py` (Group 1 fingerprint), `tests/test_fpu_diagnostic_modes.py` (Group 1 stratum wiring), `tests/test_fpu_dev_reservoir_protocol.py` (create; Group 2), `tests/test_fpu_dev_corpus_v2.py` (Group 2 config/precheck/select).

---

# PHASE A — Group 1: diagnostic stratum wiring + evidence-chain fingerprint

## Task A1: propagate `ply_bucket` into the dev rows (v2-gated)

**Files:** Modify `diagnose_fpu_policy_mass.py` (`_dev_target_row`@742, `_dev_control_row`@755, `_dev_rows_vs`@955, `_candidate_dev_records`@984, `CANDIDATE_DEV_ROW_FIELDNAMES`@977, the manifest row loader @~944); Test `tests/test_fpu_diagnostic_modes.py`.

**Interfaces (Produces):** `_dev_target_row(band, cand, ref, *, ply_bucket=None)` and `_dev_control_row(cand, ref, *, ply_bucket=None)` — when `ply_bucket` is not None, the returned gate row carries a `"ply_bucket"` key; when None, the row is byte-identical to today (no `ply_bucket` key). `_dev_rows_vs(target_control_rows, cand_by_sha, ref_by_sha, *, carry_ply_bucket=False)` — passes each source row's `ply_bucket` through iff `carry_ply_bucket`. The persisted `_candidate_dev_records` / `CANDIDATE_DEV_ROW_FIELDNAMES` gain `ply_bucket` **only** in v2 mode (a mode flag), leaving the v1 CSV schema unchanged.

- [ ] **Step 1: Failing tests** — (a) `_dev_rows_vs(rows, ..., carry_ply_bucket=True)` where each source row has `ply_bucket="late"` → every returned gate row has `row["ply_bucket"]=="late"`; (b) default `carry_ply_bucket=False` (v1) → returned rows have **no** `ply_bucket` key (assert `"ply_bucket" not in row`), and the dict equals today's output for a fixed input; (c) `_candidate_dev_records` in v1 mode emits exactly `CANDIDATE_DEV_ROW_FIELDNAMES` (no `ply_bucket`).
- [ ] **Step 2: Run → fail. Step 3: Implement** per spec §0/§9 — the source manifest rows already carry `ply_bucket` (v1 + v2 manifests both have the column); gate the propagation on the mode flag so v1 stays byte-identical. **Step 4: Run → pass** (+ run `tests/test_fpu_evidence_chain.py` to prove the v1 gate/persist path is untouched). **Step 5: Commit** — `feat(fpu-v2): propagate ply_bucket through the dev rows (v2-gated; v1 byte-identical)`.

## Task A2: `--dev-corpus-config` option + config identity checks + stratum threading

**Files:** Modify `diagnose_fpu_policy_mass.py` (`_parse_args`@~1314 add `--dev-corpus-config`; the 3 `dev_safety_verdict` call sites @1046/@1218/@1221; a new `_resolve_v2_stratum(args) -> str` helper); Test `tests/test_fpu_diagnostic_modes.py`.

**Interfaces (Consumes):** `load_v2_config` from `fpu_dev_corpus_v2` (lazy import — keeps the diagnostic import-light). **Produces:** `_resolve_v2_stratum(args)` → returns `config.new_collapse_stratum` when `--dev-corpus-config` is set (after the checks below), else `"band"`. All three production `dev_safety_verdict(...)` calls pass `stratum_key=<resolved>`.

- [ ] **Step 1: Failing tests** — with a fabricated v2 config + manifest meta: (a) `config.select_out != args.dev_manifest` raises; (b) `config.source_index_path != args.source_jsonl` raises; (c) the manifest `.meta.json`'s recorded `config_sha1` (in its `provenance`) ≠ `file_sha1(--dev-corpus-config)` raises; (d) meta `new_collapse_stratum` ≠ config's raises; (e) `config.new_collapse_stratum != "ply_bucket"` raises; (f) all agree → `_resolve_v2_stratum` returns `"ply_bucket"`; (g) no `--dev-corpus-config` → returns `"band"` and reads no config. Assert all three call sites receive the resolved stratum (spy/patch `dev_safety_verdict`).
- [ ] **Step 2: Run → fail. Step 3: Implement** per spec §0 + Group-1 bullets; each check raises before any evaluator work. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): --dev-corpus-config gate (5 identity checks) + stratum threading to the 3 safety-verdict calls`.

## Task A3: fingerprint the config SHA1 + effective stratum (omit for v1)

**Files:** Modify `diagnose_fpu_policy_mass.py` (`build_run_fingerprint`@641 — add optional `new_collapse_stratum=None, dev_corpus_config_sha1=None`; the caller @~1022 threads them); Test `tests/test_fpu_evidence_chain.py`.

**Interfaces (Produces):** `build_run_fingerprint(..., *, new_collapse_stratum=None, dev_corpus_config_sha1=None)` — when **both** are None (v1), `selection_context` is byte-identical to today (**no** new keys). When set (v2), `selection_context` gains exactly `new_collapse_stratum` and `dev_corpus_config_sha1`.

- [ ] **Step 1: Failing tests** — (a) v1 call (no kwargs) → `selection_context` has the exact current key set (pin it; assert `"new_collapse_stratum" not in selection_context` and `"dev_corpus_config_sha1" not in selection_context`); (b) v2 call → both keys present with the passed values; (c) an existing evidence-chain fingerprint test stays green (byte-identity).
- [ ] **Step 2: Run → fail. Step 3: Implement** (conditional key insertion; keys absent, never `None`, for v1) per spec §5/§9. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): fingerprint config sha1 + stratum in selection_context (omitted entirely for v1)`.

## Task A4: Group 1 integration — phase-gated operator verdict + pre-evaluator mismatch refusal

**Files:** Test only `tests/test_fpu_diagnostic_modes.py` / `tests/test_fpu_evidence_chain.py` (no new source — exercises A1–A3 end to end).

- [ ] **Step 1: Failing tests** — (a) **phase-vs-band fixture:** a dev-row set where band-stratification would pass but a phase (`ply_bucket`) has an n≥20 new-collapse rate ≥0.10 → with `stratum_key="ply_bucket"` the verdict rejects with a `ply_bucket[...]_new_collapse` reason, while `stratum_key="band"` on the same rows does not (proves the wiring changes the operator verdict); (b) **controls/candidates config/stratum mismatch refusal:** a candidates-stage fingerprint whose `selection_context` stratum or `dev_corpus_config_sha1` differs from the persisted controls fingerprint is rejected by the existing `validate_controls_fingerprint` **before** any evaluator/search call (assert the refusal path runs no search).
- [ ] **Step 2: Run → fail** (if any wiring is incomplete). **Step 3:** no source change expected — if a test fails, fix the responsible A1–A3 task and re-run. **Step 4: Run → pass. Step 5: Commit** — `test(fpu-v2): Group 1 integration — phase-gated verdict + pre-evaluator stratum-mismatch refusal`.

---

# PHASE B — Group 2: reservoir protocol → qualify → immutable config

## Task B1: protocol schema + `emit-protocol` (canonical JSON, atomic, immutable)

**Files:** Create `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py`; Test `tests/test_fpu_dev_reservoir_protocol.py` (create).

**Interfaces (Produces):** `PROTOCOL_SCHEMA_KEYS` (the spec §2.1 field set); `canonical_json_bytes(obj) -> bytes` (sorted keys, fixed formatting, `ensure_ascii`, trailing newline); `write_atomic(path, data_bytes)` (temp+rename; refuse-overwrite-different; idempotent on byte-identical → returns a status enum); `build_protocol(params) -> dict`; `emit_protocol(params, out_path, *, check=False) -> int`.

- [ ] **Step 1: Failing tests** — `canonical_json_bytes` is byte-stable across dict-insertion-order permutations; `write_atomic` writes when absent, is a no-op success on byte-identical, **raises** on overwrite-different; `build_protocol` includes every `PROTOCOL_SCHEMA_KEYS` field (matchup+anchor, `games`, `base_seed`, the 10 knobs, `save_eval_replays`, `workers`, output+report paths, generation provenance) and rejects a missing param; `emit_protocol(..., check=True)` never writes. **Import-purity:** `import fpu_dev_reservoir_protocol` loads with `mlx` not in `sys.modules`.
- [ ] **Step 2: Run → fail. Step 3: Implement** per spec §2.1/§3/§8. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): reservoir protocol schema + emit-protocol (canonical JSON, atomic, refuse-overwrite-different)`.

## Task B2: `emit-gen-command` — exact generation command from the frozen protocol

**Files:** Modify `fpu_dev_reservoir_protocol.py`; Test `tests/test_fpu_dev_reservoir_protocol.py`.

**Interfaces (Produces):** `gen_command(protocol) -> List[str]` — the exact `eval_checkpoint_match` argv derived from the protocol (`--checkpoint-a/-b`, `--games`, `--base-seed`, the 10 knobs, `--save-eval-replays`, `--replay-dir`, `--output`, `--workers`).

- [ ] **Step 1: Failing tests** — `gen_command` maps each protocol field to the right flag with the right value (assert the full argv for a fixed protocol); it is deterministic; `source_index_path` is **not** a flag (it's derived by the generator) but the emitted `--output` stem implies it. **Step 2: Run → fail. Step 3: Implement** per spec §2.1/§3. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): emit-gen-command derives the exact eval_checkpoint_match argv from the protocol`.

## Task B3: `qualify` protocol conformance (pure core, injected preflight)

**Files:** Modify `fpu_dev_reservoir_protocol.py`; Test `tests/test_fpu_dev_reservoir_protocol.py`.

**Interfaces (Produces):** `check_protocol_conformance(protocol, jsonl_rows, sidecars_by_idx, summary) -> ConformanceResult` (pure; returns ok or the first MISMATCH reason). Covers spec §4.1 **except** the summary-recompute (B4) and preflight (B5): game count == `protocol.games`; `game_idx` contiguous `0..games-1`; sidecar count; seed `== base_seed + game_idx`; matchup identities (name+sha1); model-color parity (even→A-red / odd→B-red); replay linkage (`replay_path` exists, sidecar `game_idx`/`seed`/colors/`board_size` match the row); the ten match knobs == protocol; `workers` == protocol; output-path derivation (`source_index_path == <summary stem>_games.jsonl`, `replay_dir`); within-game move-player parity; generation provenance (`generation_source_sha1s` over the §2.1 13-module list + `generation_git_commit`).

- [ ] **Step 1: Failing tests** — a fabricated **feasible-shaped but small** reservoir (fixtures fabricate rows/sidecars/summary dicts): the clean case returns ok; **each** defect returns MISMATCH naming the failing check — wrong seed, wrong matchup, wrong game count, each of the 10 knobs wrong, non-contiguous `game_idx`, missing/broken `replay_path`, wrong color parity, wrong workers, wrong JSONL-vs-summary path, a mutated generation-source hash. **Step 2: Run → fail. Step 3: Implement** per spec §4.1. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): qualify protocol-conformance checks (all MISMATCH cases)`.

## Task B4: `qualify` summary binding by reconstruction

**Files:** Modify `fpu_dev_reservoir_protocol.py`; Test `tests/test_fpu_dev_reservoir_protocol.py`.

**Interfaces (Consumes):** `eval_summary.summarize_match`, `eval_runner.EvalGameResult` (both pure). **Produces:** `check_summary_binding(protocol, jsonl_rows, summary) -> ConformanceResult` — reconstruct `EvalGameResult` rows from the JSONL, call the real `summarize_match(results, a, b, pairing_id, config)`, require its **complete** output == the supplied summary **excluding** `generated_at` + `git_commit`; separately require `summary["git_commit"] == protocol.generation_git_commit`. `reason_histogram(jsonl_rows) -> dict` for the report.

- [ ] **Step 1: Failing tests** — reconstructed-and-recomputed summary equals a faithful summary (clean); a summary from a *different* run (one flipped winner / different score) → MISMATCH; a `git_commit` ≠ protocol → MISMATCH; `generated_at`/`git_commit` differences alone (with matching body) do **not** trip the body compare; `reason_histogram` counts `win`/`state_cap`/`board_full` from the JSONL. **Step 2: Run → fail. Step 3: Implement** per spec §4.1 (amendments 3, 5). **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): summary<->JSONL binding by EvalGameResult reconstruction + full summarize_match compare`.

## Task B5: geometric preflight integration + exit-code classification

**Files:** Modify `fpu_dev_reservoir_protocol.py`; Test `tests/test_fpu_dev_reservoir_protocol.py`.

**Interfaces (Consumes):** `v2_preflight_source` from `fpu_dev_corpus_v2` (the real preflight; injected as a parameter defaulting to it). **Produces:** `qualify_core(protocol, reservoir, *, preflight=v2_preflight_source) -> QualifyResult{status: OK|MISMATCH|GATE_FAIL, reason, report}` — conformance (B3) → summary binding (B4) → `preflight(records)`; feasible ⇒ OK; infeasible ⇒ **GATE_FAIL**; a conformance/binding failure ⇒ **MISMATCH** (preflight not reached).

- [ ] **Step 1: Failing tests** — with an **injected fake** preflight: clean small reservoir + fake-feasible ⇒ OK; fake-infeasible ⇒ GATE_FAIL; any conformance defect ⇒ MISMATCH regardless of the fake. **Separately**, with the **real** `v2_preflight_source`: a genuinely-feasible larger synthetic reservoir (sized to clear the 240-row/4-phase quotas) ⇒ OK, and a protocol-faithful-but-infeasible one ⇒ GATE_FAIL. **Step 2: Run → fail. Step 3: Implement** per spec §4.2/§6 (injection point). **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): qualify geometric-preflight stage + OK/MISMATCH/GATE_FAIL classification (injected preflight for tests)`.

## Task B6: config derivation + emit + report state machine

**Files:** Modify `fpu_dev_reservoir_protocol.py`; Test `tests/test_fpu_dev_reservoir_protocol.py`.

**Interfaces (Produces):** `derive_config(protocol, reservoir_measurements) -> dict` — the canonical `fpu_dev_corpus_v2_config.json` per spec §2.2 (carried decisions + measured identities, incl. the three checkpoint identities, `eval_batch_size`/`stall_flush_sims`, `report_out`); `write_report(path, qualify_result) -> None` (deterministic; MISMATCH replaceable; PASS/GATE-FAIL terminal-immutable + GATE-FAIL records retirement); `is_retired(report_path) -> bool`; `run_qualify(protocol_path, *, check=False) -> int` (the operator entry: load protocol → measure reservoir → `qualify_core` → on OK derive+emit config atomically + PASS report; on GATE_FAIL write retirement report + refuse config; on MISMATCH write replaceable report; refuse if already retired; exit 0/3/4).

- [ ] **Step 1: Failing tests** — `derive_config` is byte-deterministic and equals a golden for a fixed (protocol, measurements); it is a **pure function** (same inputs → identical bytes); `write_report` refuses to overwrite a PASS/GATE-FAIL report but replaces a MISMATCH; `is_retired` true after a GATE-FAIL; `run_qualify` refuses (exit 4, no config) on an already-retired protocol. **Step 2: Run → fail. Step 3: Implement** per spec §2.2/§3. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): derive(protocol,reservoir) config + report state machine (MISMATCH replaceable; PASS/GATE-FAIL terminal)`.

## Task B7: extend `V2Config` + `load_v2_config`; add qualification module to the v2 source set

**Files:** Modify `fpu_dev_corpus_v2.py` (`_V2_CONFIG_REQUIRED_KEYS`@~1810, `V2Config`@1826, `load_v2_config`@1885, `_V2_CORPUS_SOURCES`@2152); Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces (Produces):** `V2Config` gains the spec §2.2 required fields (`config_schema_version`, `protocol_path`, `match_summary_path`, `replay_dir`, `report_out`, plus the extended `expected_fingerprints` sub-keys); `load_v2_config` requires them for the current `config_schema_version` (no silent omission). `_V2_CORPUS_SOURCES` gains `fpu_dev_reservoir_protocol.py`.

- [ ] **Step 1: Failing tests** — `load_v2_config` on a config missing any new required key raises a clear error; loads a complete one; `V2Config` round-trips the new fields; `_V2_CORPUS_SOURCES` contains the qualification module path (and every path exists on disk); **v1 `build_fpu_dev_corpus._CORPUS_SOURCES` is byte-unchanged** (assert the tuple equals its prior contents). Import-purity holds. **Step 2: Run → fail. Step 3: Implement** per spec §2.2/§6/§9 (the qualification module is result-determining for the corpus → v2 set only). **Fixture seam (spec §9):** this task makes the new fields required, so **every existing test that fabricates a `V2Config` / config dict** (the Task-6 select tests + any Group-1 A2 fixtures in `tests/test_fpu_dev_corpus_v2.py`) must be updated in the same commit — introduce/extend a single shared `_v2_config_fixture(**overrides)` helper so the schema lives in one place. Run the full `tests/test_fpu_dev_corpus_v2.py` to confirm no fabricated-config test regressed. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): extend V2Config schema (required) + qualification module in _V2_CORPUS_SOURCES + shared config fixture`.

## Task B8: `run_screen` pre-evaluator precheck (re-derive + byte-compare + defensive preflight)

**Files:** Modify `fpu_dev_corpus_v2.py` (`run_screen` — add the precheck before the evaluator loads; lazy import of the new module); `fpu_dev_reservoir_protocol.py` (`precheck_before_screen(config, *, preflight=v2_preflight_source) -> None`); Test `tests/test_fpu_dev_corpus_v2.py` + `tests/test_fpu_dev_reservoir_protocol.py`.

**Interfaces (Produces):** `precheck_before_screen(config, ...)` — (1) recompute every pinned hash from its pinned path and hard-match; (2) **re-derive the canonical config from (protocol, reservoir) and byte-compare** against the supplied config; (3) verify the config binds this protocol (`protocol_sha1`); (4) run the geometric preflight defensively. Raises on any failure. `run_screen` calls it (lazy import) before any checkpoint/evaluator work.

- [ ] **Step 1: Failing tests (pure; no evaluator)** — `precheck_before_screen` on a faithful (config, protocol, reservoir) passes; a byte-changed reservoir/protocol/summary/source-file → raises (hash recheck); **an edited non-hashed config field (`selection_seed`, `select_out`) → raises via the re-derive byte-compare** (the load-bearing case); a config binding a different protocol → raises; an infeasible reservoir → raises (defensive preflight). Statically confirm `run_screen` calls the precheck **before** its lazy evaluator import (source-order/wiring assertion; do NOT run the evaluator). **Step 2: Run → fail. Step 3: Implement** per spec §5/§6. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): run_screen pre-evaluator precheck — re-derive+byte-compare config + hash recheck + defensive preflight`.

## Task B9: extend the select recheck with the protocol + summary identities

**Files:** Modify `fpu_dev_corpus_v2.py` (`validate_screen_identities` / `select_final_manifest` — add the protocol + match_summary identities and the re-derive byte-compare to the select-time chain); Test `tests/test_fpu_dev_corpus_v2.py`.

**Interfaces (Consumes):** the Task B8 re-derive helper. **Produces:** select-time verification now covers the `protocol_sha1` + `match_summary_sha1` identities and re-derives + byte-compares the config, in addition to the existing 7-identity + screen-artifact chain (Task 6).

- [ ] **Step 1: Failing tests (pure)** — a tampered `protocol_sha1`/`match_summary_sha1` at select time → raises; an edited non-hashed config field → raises via re-derive; the existing Task-6 identity/forgery tests stay green (no regression to the shipped select chain). **Step 2: Run → fail. Step 3: Implement** per spec §5. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): extend select recheck — protocol/summary identities + re-derive byte-compare`.

## Task B10: CLI wiring + exit codes

**Files:** Modify `fpu_dev_reservoir_protocol.py` (`main(argv) -> int`); Test `tests/test_fpu_dev_reservoir_protocol.py`.

**Interfaces (Produces):** `main` with subcommands `emit-protocol`, `emit-gen-command`, `qualify` (each supporting `--check` where applicable), exit codes `0/2/3/4`. `qualify` never launches generation (zero-GPU). No `--mode select`/`screen` here — those stay in `fpu_dev_corpus_v2.main`.

- [ ] **Step 1: Failing tests (pure; argparse only)** — each subcommand parses its required args and rejects missing ones (exit 2); `qualify` returns the `qualify_core` status as the process exit code; `--check` paths write nothing. Do NOT invoke generation/MCTS. **Step 2: Run → fail. Step 3: Implement** per spec §3. **Step 4: Run → pass. Step 5: Commit** — `feat(fpu-v2): reservoir-protocol CLI (emit-protocol / emit-gen-command / qualify) + exit codes`.

---

# PHASE C — combined pre-operator audit

## Task C1: full suite, byte-identity, import-purity; STOP before the reservoir

**Files:** none (verification).

- [ ] **Step 1** — the combined test matrix (spec §11) is green: Group 1 (phase-gated verdict, v1 fingerprint byte-identity, config/stratum mismatch refusal) + Group 2 (MISMATCH/GATE-FAIL/precheck/reproducibility/immutability). Confirm the six regression areas of the v2 corpus tooling still pass.
- [ ] **Step 2** — `import scripts.GPU.alphazero.fpu_dev_corpus_v2` and `...fpu_dev_reservoir_protocol` with `mlx` not loaded; byte-compile both changed modules.
- [ ] **Step 3** — **v1 byte-identity:** `git diff main -- scripts/GPU/alphazero/build_fpu_dev_corpus.py` is empty; `build_fpu_dev_corpus._CORPUS_SOURCES` unchanged; the v1 diagnostic default path reproduces its prior fingerprints/artifacts (the existing evidence-chain suite green with no call-site changes).
- [ ] **Step 4** — full suite `.venv/bin/python -m pytest -p no:cacheprovider tests/ -q` green vs the **1777** baseline (report the delta = the new tests, zero regressions).
- [ ] **Step 5** — **STOP for review before the operator generates the 4,800-game reservoir.** Do NOT run any operator MCTS/screen/select/qualify-on-real-data phase.

## Self-review — spec coverage

| Spec section | Task(s) |
|---|---|
| §0 Group 1 seam (ply_bucket propagation, `--dev-corpus-config`, 5 checks, stratum threading, fingerprint keys) | A1, A2, A3, A4 |
| §2.1 protocol schema (matchup+anchor, 10 knobs, workers, paths, 13-module provenance) | B1 |
| §2.2 config schema (extends `_V2_CONFIG_REQUIRED_KEYS`; 3 checkpoint identities; eval_batch_size/stall_flush_sims/report_out; qualification module in v2 source set) | B6, B7 |
| §3 CLI stages, exit codes, immutability ops, report state machine | B1, B6, B10 |
| §4.1 protocol conformance (incl. workers, path derivation, provenance) | B3 |
| §4.1 summary binding by reconstruction + reason histogram | B4 |
| §4.2 geometric preflight + GATE-FAIL-only classification | B5 |
| §5 pre-screen re-derive+byte-compare + select recheck | B8, B9 |
| §6 circular-import resolution + preflight injection | B5, B7, B8 |
| §7 no-top-up & versioning (protocol_version, retirement) | B6 |
| §8 determinism & reviewability (canonical JSON, `--check`) | B1, B6, B10 |
| §9 backward compat (v1 byte-identical; required v2 schema) | A1, A3, B7, C1 |
| §10 trust boundaries (documented; no enforcement claimed) | (doc only) |
| §11 test matrix | A4, B3, B4, B5, B6, B8, C1 |
