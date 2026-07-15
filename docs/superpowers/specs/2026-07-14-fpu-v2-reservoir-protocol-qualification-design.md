# FPU v2 Reservoir Protocol, Qualification & Config Generation — Design

**Status:** APPROVED (conditional on the amendments below, all incorporated), 2026-07-14.
**Context:** Pre-operator hardening before the 4,800-game v2 reservoir is generated. Two gaps
surfaced in a post-merge audit of the `fpu-dev-corpus-v2-phase` tooling (merged to main @053650a):
(1) the FPU diagnostic's phase-stratum knob was parameterized but never wired to the operator
stages, and (2) `fpu_dev_corpus_v2_config.json` merely *records* the frozen reservoir decisions —
nothing *enforces* them. This spec covers the **Group 2** subsystem that closes gap (2). **Group 1**
(the bounded diagnostic-stratum + evidence-chain fix that closes gap (1)) is specified separately and
implemented as its own reviewed change; its seam to Group 2 is documented in §0.

**TOOLING ONLY.** No evaluator/MCTS/GPU/checkpoint and no reservoir generation runs in this work.
The qualifier is zero-GPU; the operator spends the GPU-hours generating the reservoir.

---

## 0. Scope & the Group 1 seam

- **Group 1 (bounded, separate change):** `diagnose_fpu_policy_mass.py` — propagate `ply_bucket` into
  the dev rows (in-memory gate rows *and* the persisted `candidate_dev_rows.csv`, **v2-gated** so v1
  stays byte-identical); add optional `--dev-corpus-config` (load via `load_v2_config`; require
  `config.select_out == --dev-manifest`, `config.source_index_path == --source-jsonl`, the manifest
  `.meta.json` config-identity + `new_collapse_stratum` agree, and `new_collapse_stratum == "ply_bucket"`);
  pass the configured `stratum_key` to all three `dev_safety_verdict` production calls; fingerprint the
  config SHA1 + effective stratum in `build_run_fingerprint`'s `selection_context`, **omitting both keys
  entirely for legacy v1** (no `None` placeholder).
- **Group 2 (this spec):** `reservoir_protocol.json` → emit exact generation command → operator
  generates once → `qualify` (zero-GPU: protocol conformance + geometric preflight) → emits the
  immutable `fpu_dev_corpus_v2_config.json` → `run_screen` precheck → screen → select recheck.
- **The seam:** the config is the shared artifact. Group 2 *produces* it; Group 1's diagnostic *consumes*
  its already-existing fields (`select_out`, `source_index_path`, `new_collapse_stratum`) and fingerprints
  its `config_sha1`. Group 2 adds `protocol`/`match_summary` path+hash identities to the pre-registered
  identity chain that the screen precheck and select recheck verify; Group 1's `config_sha1` fingerprint
  transitively covers them (the config binds the protocol hash). No rework between the groups — a clean
  producer/consumer seam. A **combined pre-operator audit** runs both groups' suites together before the
  reservoir is authorized.

## 1. Lifecycle (two immutable artifacts)

```
emit protocol → emit exact command → generate once (operator, GPU) →
qualify + geometric preflight (zero-GPU) → emit immutable config →
screen precheck → screen → select recheck
```

`reservoir_protocol.json` is frozen *before* generation and declares intent. `fpu_dev_corpus_v2_config.json`
is emitted *only after* a generated reservoir passes qualification, and is **born immutable**. The config is
never mutated; the protocol is never revisited (a failure regenerates or versions — §7).

## 2. Artifacts & schemas (canonical JSON — sorted keys, fixed numeric formatting, no timestamps/RNG)

### 2.1 `reservoir_protocol.json` (frozen pre-generation)
- `protocol_version` (monotonic int/string), `no_top_up: true`.
- Matchup: `checkpoint_a` / `checkpoint_b` — each a `{path, identity}` where identity = `name:sha1`.
- `games` — **protocol-declared** game count (first frozen protocol = 4,800; a later version may declare
  9,600). The qualifier validates against *this* number, never a hard-coded constant (amendment 7).
- `base_seed` — the half-open seed range is `[base_seed, base_seed + games)`, one seed per game,
  `seed == base_seed + game_idx` (verified against `eval_runner.build_pairing_tasks`, single pairing,
  offset 0).
- The ten result-determining match knobs (amendment 4): `board_size`, `mcts_sims`,
  `mcts_eval_batch_size`, `mcts_stall_flush_sims`, `selection_mode`, `opening_temp_plies`, `temp_high`,
  `temp_low`, `max_moves`, `base_seed`. (`workers` is non-determining → recorded-only.)
- Output paths: `source_index_path` (the `*_games.jsonl`), `replay_dir`, `match_summary_path`.
- Generation provenance (amendment 8): `generation_git_commit` + `generation_source_sha1s` over the
  result-determining generation modules (`eval_checkpoint_match`, `eval_runner`, `mcts`, plus the
  state/eval deps the games depend on). The exact module list is pinned in the implementation plan,
  mirroring how the corpus's `RESULT_DETERMINING_SOURCES` was fixed. See the trust-boundary limitation in §10.

### 2.2 `fpu_dev_corpus_v2_config.json` (emitted by `qualify`, immutable; amendment 2)
Schema-versioned; every field below is **required** for the current `config_schema_version` — a newly
generated v2 config may not silently omit them ("backward compatible" must not mean "may lack the new
identities").
- `config_schema_version`.
- `protocol_path` + `protocol_sha1`; `match_summary_path` + `match_summary_sha1`;
  `source_index_path` + `source_index_sha1`; `replay_dir` + `replay_data_sha1` (deterministic hash of the
  sidecar *contents*); `source_file_sha1s` (the corpus result-determining code); `checkpoint_identity`
  (the anchor/matchup checkpoints — carried through from the existing select identity set, not dropped).
- Forbidden manifests + their hashes.
- Selection settings: `selection_seed`, allocation (must equal `SPLIT_ALLOC_V2`), late floors,
  enumerator params, `new_collapse_stratum: "ply_bucket"`.
- `screen_out`, `select_out`.

**Paths as well as hashes (amendment 2):** hashes alone can't be recomputed pre-screen/select — the path
is needed to re-read the file. Every pinned identity carries both.

## 3. CLI stages, exit codes & operational immutability (amendment 5)

New module CLI (`fpu_dev_reservoir_protocol.py`):
- `emit-protocol` — freeze a `reservoir_protocol.json` from declared params.
- `emit-gen-command` — print the exact `eval_checkpoint_match` command derived from the frozen protocol
  (zero-GPU; so generation cannot drift from the frozen decisions).
- `qualify` — protocol conformance (§4.1) → geometric preflight (§4.2) → **emit the config only if every
  check passes**.

**Exit codes:** `0` ok · `3` MISMATCH · `4` GATE-FAIL · `2` usage/IO.

**Both emitters must:** write **atomically** (temp file + rename); **refuse to overwrite an existing
*different* artifact**; treat an existing **byte-identical** artifact as success (idempotent); emit a
**deterministic qualification report** on pass *or* fail; emit the config **only after all conformance +
preflight checks pass**. `--check` recomputes and diffs but **never writes**.

## 4. Qualification (zero-GPU)

### 4.1 Protocol conformance → any failure is **MISMATCH (exit 3)**, "regenerate under the same protocol"
- **Game count** == `protocol.games`; exactly that many JSONL rows **and** that many replay sidecars.
- **Contiguity:** `game_idx` runs `0..games-1` with no gaps/dupes.
- **Seed range:** the half-open `[base_seed, base_seed + games)`; each game's recorded `seed == base_seed +
  game_idx`.
- **Matchup:** the summary and every JSONL row's `red_checkpoint`/`black_checkpoint` resolve to the
  protocol's `checkpoint_a`/`checkpoint_b` identities (name + sha1).
- **Model color parity (amendment 4, corrected):** color alternates **between games** by `game_idx`
  parity — even `game_idx` → checkpoint-A red, odd → checkpoint-B red.
- **Replay linkage / capture:** every `replay_path` exists and its sidecar's `game_idx`, `seed`,
  colors, and `board_size` match the JSONL row (this existence+linkage check is what *requires* replay
  capture — there is no separate capture flag to gate).
- **Result-determining match config (amendment 4):** all ten knobs recorded in the summary's config equal
  the protocol. `workers` recorded-only.
- **Summary ↔ JSONL binding (amendment 3):** recompute from the JSONL — game count, win/loss/state-cap
  counts, termination `reason` counts (`state_cap`/`board_full`/`win`), red/black-win counts, per-checkpoint
  winner counts — and require they equal the summary's recorded aggregates. (Otherwise a summary from a
  *different* run with the same settings could be paired with this reservoir.)
- **Generation provenance (amendment 8):** recompute `generation_source_sha1s` + `generation_git_commit`
  and match the protocol (subject to the §10 trust boundary).
- **Within-game move-player parity** is validated **separately** (red on even ply within each replay) —
  distinct from the between-games model-color parity above.

### 4.2 Geometric preflight (amendment 1)
After protocol conformance passes, call the existing zero-GPU `v2_preflight_source(records)` (Task 4).
- **Feasible** → proceed to emit the config.
- **Infeasible** — the reservoir faithfully matches its protocol but its geometry cannot support the v2
  corpus → **GATE-FAIL (exit 4)**: retire this protocol version, version a new one.
- A corrupt/incomplete output that breaks preflight's *inputs* is a **MISMATCH (exit 3)**, not a gate
  failure.

**The only GATE-FAIL condition is a protocol-faithful reservoir with infeasible geometry.** Every
structural defect (colors, linkage, contiguity, seed, MCTS config, summary mismatch) is a MISMATCH.

## 5. Fingerprint check split (amendment 2; decision #6)
- `qualify` pins every §2.2 path+hash identity into the config.
- **Before screen** — `run_screen` startup recomputes every pinned hash from its pinned path and
  hard-matches; verifies the config binds *this* protocol (`protocol_sha1`); and **repeats the geometric
  preflight defensively** (amendment 1). Any staleness (edited code, byte-changed reservoir, tampered
  config/protocol/summary) aborts **before the evaluator loads** — so an hours-long screen never starts on
  stale inputs.
- **During select** — the existing Task 6 chain re-verifies the same reservoir/config identities, extended
  to include the `protocol` and `match_summary` identities, **plus** the screen artifact (`screen_csv_sha1`
  + the row cross-check). Reservoir/config identities are checked **twice** (pre-GPU and at select); the
  screen output once (at select).

## 6. Module boundary & circular-import resolution (amendment 6 — PINNED)
New module `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py` (pure protocol/qualification/config-emit +
CLI). The only edit to `fpu_dev_corpus_v2.py` is the `run_screen` precheck hook (§5) plus the additive
required `V2Config` fields (§2.2).

**Cycle risk:** `fpu_dev_corpus_v2` (run_screen) → new module → `fpu_dev_corpus_v2.V2Config`.
**Pinned resolution (option a):** `run_screen` **lazily imports** the new module's precheck function inside
the function body (consistent with run_screen's existing lazy-import discipline for heavy deps); the new
module top-level-imports `fpu_dev_corpus_v2` **only for the shared config schema-key constant** (single
source of truth, no schema duplication) and treats a config otherwise as a **duck-typed mapping**. No import
cycle exists, and the new module stays importable and unit-testable in isolation. *(Option b — extracting
`V2Config` into a separate pure schema module — is rejected: it would move the just-shipped screen/select
import sites and churn the audited evidence chain for no functional gain.)*

`load_v2_config` **requires** the §2.2 fields for the current `config_schema_version` (no silent omission).

## 7. No-top-up & versioning (decisions #4, #5)
- `protocol_version` + `no_top_up: true`; one protocol ⇒ one reservoir ⇒ `protocol.games` games.
- **MISMATCH** → regenerate under the **same** protocol (an operator/generation mistake never burns a
  version). **GATE-FAIL** → **retire** the version.
- To scale later (e.g. 9,600), declare a **new** protocol version with a new seed range, generate a fresh
  independent reservoir, and qualify → a new config. **Never append to or re-qualify an existing reservoir.**

## 8. Determinism & reviewability (decision #7)
Canonical JSON everywhere ⇒ re-emitting from the same inputs is byte-identical. `--check` recomputes and
diffs, never writes. `qualify` writes a deterministic report artifact (pass or fail) so a reviewer can
independently reproduce the verdict. No `Date.now`/RNG in any emitted artifact.

## 9. Backward compatibility (decision #9)
`build_fpu_dev_corpus.py` (v1) stays byte-identical. The v1 diagnostic path (no `--dev-corpus-config`) stays
byte-identical — v1 fingerprints omit the new keys entirely (Group 1). The new module is additive. The
`V2Config` field additions are **required** for the new schema version but change no v1 artifact. Fixture
seam: Group 2 owns the `V2Config` schema bump; shared config fixtures (including any Group 1 fabricates) are
updated when Group 2 lands.

## 10. Trust boundaries (honest limitations)
- **Generator provenance (amendment 8):** the protocol records the generation git commit + generation
  source hashes, and `qualify` recomputes and matches them. This proves the sources **as they exist at
  qualify time** equal the protocol's declaration; it does **not** prove those exact bytes **executed**
  during generation (an uncommitted edit reverted after the run would still pass). The only complete fix is
  snapshotting the generator source at execution time, which is out of scope. Stated plainly rather than
  implying the post-hoc hashes attest execution — mirroring the unsigned-meta residual disclosed in the
  select stage.
- **Config/protocol are unsigned:** an actor who rewrites *both* a pinned file and its recorded hash forges
  the chain; the identity set is tamper-*evident*, not tamper-*proof*. Signing is out of scope (as in select).

## 11. Test matrix (all zero-GPU; fabricated ~6-game reservoirs; game count is protocol-declared)
- **MISMATCH (exit 3):** wrong seed / wrong matchup / wrong game count / any of the ten MCTS knobs wrong /
  non-contiguous indices / broken replay linkage / wrong model colors / summary↔JSONL aggregate mismatch /
  generation-provenance mismatch.
- **GATE-FAIL (exit 4):** a protocol-faithful reservoir whose `v2_preflight_source` geometry is infeasible.
- **Screen precheck:** a stale fingerprint (edited code / byte-changed reservoir / tampered config or
  protocol or summary) is rejected **before evaluator load**; the defensive preflight re-runs.
- **Config bootstrap reproducibility:** byte-identical re-emit; `--check` diff; refuse-overwrite-different;
  idempotent-on-byte-identical; `--check` never writes.
- **Immutability ops:** atomic write verified; deterministic report on pass and on fail.
- **Combined audit (with Group 1):** v1 byte-identity (build + diagnostic) and v2 phase-gated operator
  behavior.

## 12. Progression
```
emit protocol → emit exact command → generate once → qualify + geometric preflight →
emit immutable config → screen precheck → screen → select recheck
```
Downstream (unchanged): controls → candidates → frozen sweep (per-phase new-collapse gate via Group 1) →
frozen check → strength match.
