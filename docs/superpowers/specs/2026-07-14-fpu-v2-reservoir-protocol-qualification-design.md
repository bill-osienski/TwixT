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

**Derivability invariant (amendments 2 + 3):** the config is a **pure deterministic function of
`(protocol, reservoir)`** — `config = derive(protocol, reservoir)`. Therefore the protocol carries *every*
declared decision (reservoir params **and** selection settings **and** output/report paths), and the config
adds only the *measured* identities (reservoir/code/checkpoint hashes). This is what makes the pre-screen
"re-derive and byte-compare" tamper check (§5) complete rather than partial.

### 2.1 `reservoir_protocol.json` (frozen pre-generation) — the single source of all declared decisions
- **Identity:** `protocol_version` (monotonic), `no_top_up: true`, `config_schema_version` (the schema the
  emitted config will use).
- **Matchup + anchor (amendment 1 disambiguation):** `checkpoint_a` and `checkpoint_b` — the two reservoir
  players, each `{path, identity=name:sha1}` — **plus** `anchor: "checkpoint_a" | "checkpoint_b"` pinning
  which one is the *single* screen anchor (the fpu-off value-head evaluator, `config.checkpoint`). These are
  **three distinct roles**: reservoir player A, reservoir player B, and the anchor (= one of A/B). The first
  frozen protocol pins `checkpoint_a = calib020_0001`, `checkpoint_b = 0379`, `anchor = "checkpoint_a"` — so
  the anchor is **calib020_0001 = A** (verified: `diagnose_fpu_sweep.DEFAULT_CHECKPOINT` is calib020_0001).
- **Reservoir params:** `games` (protocol-declared: first = 4,800, a later version may declare 9,600 —
  validated against *this* number, never a constant, amendment 7); `base_seed` (half-open seed range
  `[base_seed, base_seed + games)`, `seed == base_seed + game_idx`, verified against
  `eval_runner.build_pairing_tasks`, single pairing, offset 0); the ten result-determining match knobs
  (amendment 4): `board_size`, `mcts_sims`, `mcts_eval_batch_size`, `mcts_stall_flush_sims`,
  `selection_mode`, `opening_temp_plies`, `temp_high`, `temp_low`, `max_moves`, `base_seed`; `save_eval_replays:
  true` (capture is mandatory). `workers` is **not** a frozen decision (non-determining).
- **Output relationships (amendment 3):** `match_summary_path` (= the generator's `--output`);
  `source_index_path` — declared **and** required to equal `<match_summary stem>_games.jsonl` (the exact rule
  in `eval_checkpoint_match._write_outputs`); `replay_dir` — `--replay-dir` or `<stem>_replays`
  (`replay_dir_for`); `config_out`; `report_out`. Every explicit generation flag needed to reconstruct the
  command is present (matchup, games, base_seed, the ten knobs, `--save-eval-replays`, `--replay-dir`,
  `--output`).
- **Selection settings** (so the config is derivable): `selection_seed`, `phase_allocation` (= `SPLIT_ALLOC_V2`),
  `late_floors`, `enumerator_params`, `new_collapse_stratum: "ply_bucket"`, `forbidden_manifests`, `screen_out`,
  `select_out`.
- **Generation provenance (amendment 8):** `generation_git_commit` + `generation_source_sha1s` over the
  **enumerated** result-determining generation modules (pinned here, not deferred): `eval_checkpoint_match.py`,
  `eval_runner.py`, `mcts.py`, `game/twixt_state.py`, `eval_replay.py`, `probe_eval.py`
  (`load_network_for_scoring`), `network.py`, `local_evaluator.py`. Rationale: these are the modules whose
  bytes, given the checkpoint bytes, determine the generated games (task/seed/color assembly, the search, the
  board rules, the replay serialization, and network load/forward). Trust boundary in §10.

### 2.2 `fpu_dev_corpus_v2_config.json` (emitted by `qualify`, immutable; amendments 1, 2)
**This EXTENDS the current `_V2_CONFIG_REQUIRED_KEYS`** (`source_index_path`, `seed_range`, `selection_seed`,
`phase_allocation`, `late_floors`, `enumerator_params`, `new_collapse_stratum`, `checkpoint`,
`forbidden_manifests`, `screen_out`, `select_out`, `expected_fingerprints`; plus `eval_batch_size` with a
default). It does **not** replace them. Complete final schema — every field **required** for the current
`config_schema_version` (a newly generated v2 config may not silently omit them):

- **Carried from the protocol (the derivable decisions):** `source_index_path`, `seed_range`, `selection_seed`,
  `phase_allocation`, `late_floors`, `enumerator_params`, `new_collapse_stratum` (= `"ply_bucket"`),
  `checkpoint` (= the **anchor** path, singular), `forbidden_manifests`, `screen_out`, `select_out`.
- **New top-level (paths — amendment 2):** `config_schema_version`, `protocol_path`, `match_summary_path`,
  `replay_dir`.
- **`expected_fingerprints` (extended)** — the measured identities: `protocol_sha1`, `source_index_sha1`,
  `replay_data_sha1` (hash of sidecar *contents*), `match_summary_sha1`, `source_file_sha1s` (the corpus
  result-determining code), `forbidden_manifest_sha1s`, and **three distinct checkpoint identities**
  (amendment 1) — `reservoir_checkpoint_a_identity`, `reservoir_checkpoint_b_identity`,
  `anchor_checkpoint_identity` (= A). The prior single `checkpoint_identity` could not name all three; it is
  replaced by these three keys.

**Paths *and* hashes (amendment 2):** hashes alone can't be recomputed pre-screen/select — the path is needed
to re-read the file. Every pinned identity carries both.

## 3. CLI stages, exit codes & operational immutability (amendment 5)

New module CLI (`fpu_dev_reservoir_protocol.py`):
- `emit-protocol` — freeze a `reservoir_protocol.json` from declared params.
- `emit-gen-command` — print the exact `eval_checkpoint_match` command derived from the frozen protocol
  (zero-GPU; so generation cannot drift from the frozen decisions).
- `qualify` — protocol conformance (§4.1) → geometric preflight (§4.2) → **emit the config only if every
  check passes**.

**Exit codes:** `0` ok · `3` MISMATCH · `4` GATE-FAIL · `2` usage/IO.

**Both emitters must:** write **atomically** (temp file + rename); **refuse to overwrite an existing
*different* artifact**; treat an existing **byte-identical** artifact as success (idempotent); emit the config
**only after all conformance + preflight checks pass**. `--check` recomputes and diffs but **never writes**.

**Report lifecycle (amendment 3).** `qualify` always writes a **deterministic report** to the protocol's
`report_out`, on pass *and* on fail, recording every check's outcome plus the computed reason histogram
(§4.1). The report distinguishes the two failure classes: a **GATE-FAIL** report records the protocol version
as **RETIRED** (a durable retirement marker); `qualify` refuses to re-qualify a protocol whose `report_out`
already carries a retirement record. A **MISMATCH** report records the mismatch but **must not** retire the
protocol — regeneration under the same protocol stays allowed (so an operator/generation mistake never burns
a version). A pass report is the qualification evidence the config's `report_out` points back to.

## 4. Qualification (zero-GPU)

### 4.1 Protocol conformance → any failure is **MISMATCH (exit 3)**, "regenerate under the same protocol"
- **Game count** == `protocol.games`; exactly that many JSONL rows **and** that many replay sidecars.
- **Contiguity:** `game_idx` runs `0..games-1` with no gaps/dupes.
- **Seed range:** the half-open `[base_seed, base_seed + games)`; each game's recorded `seed == base_seed +
  game_idx`.
- **Output-path relationships (amendment 3):** `protocol.source_index_path` equals
  `<protocol.match_summary_path stem>_games.jsonl` (the `eval_checkpoint_match._write_outputs` rule); the
  `replay_dir` equals `--replay-dir` or `<stem>_replays`; the files exist at those paths.
- **Matchup:** the summary's `checkpoint_a`/`checkpoint_b` and every JSONL row's
  `red_checkpoint`/`black_checkpoint` resolve to the protocol's `checkpoint_a`/`checkpoint_b` identities
  (name + sha1).
- **Model color parity (amendment 4, corrected):** model color alternates **between games** by `game_idx`
  parity — even `game_idx` → checkpoint-A red, odd → checkpoint-B red.
- **Replay linkage / capture:** every `replay_path` exists and its sidecar's `game_idx`, `seed`,
  colors, and `board_size` match the JSONL row (this existence+linkage check is what *requires* replay
  capture — there is no separate capture flag to gate).
- **Result-determining match config (amendment 4):** all ten knobs recorded in the summary's `config` equal
  the protocol. `workers` recorded-only.
- **Summary ↔ JSONL binding (amendments 3, 5) — only against fields the summary *actually* records:**
  recompute from the JSONL and require equality with the summary's real aggregates — `games`, `state_caps`,
  `board_full`, `a_wins`, `b_wins`, `color_bias.red_win_rate_decisive`, `avg_plies`, `checkpoint_a`/`_b`,
  `selection_mode` (verified against `eval_summary.summarize_match`'s return). Do **not** assert against a
  nonexistent field. The summary has **no** complete termination-reason histogram, so `qualify` **computes the
  full reason histogram from the JSONL and records it in the report** (§3) rather than comparing it to a
  summary field. (Binding the summary to the JSONL prevents pairing a summary from a *different* run with the
  same settings.)
- **Generation provenance (amendment 8):** recompute `generation_source_sha1s` (over the §2.1 enumerated
  module list) + `generation_git_commit` and match the protocol (subject to the §10 trust boundary).
- **Within-game move-player parity** is validated **separately** (players alternate by ply within each
  replay) — distinct from the between-games model-color parity above.

### 4.2 Geometric preflight (amendment 1)
After protocol conformance passes, call the existing zero-GPU `v2_preflight_source(records)` (Task 4).
- **Feasible** → proceed to emit the config.
- **Infeasible** — the reservoir faithfully matches its protocol but its geometry cannot support the v2
  corpus → **GATE-FAIL (exit 4)**: retire this protocol version, version a new one.
- A corrupt/incomplete output that breaks preflight's *inputs* is a **MISMATCH (exit 3)**, not a gate
  failure.

**The only GATE-FAIL condition is a protocol-faithful reservoir with infeasible geometry.** Every
structural defect (colors, linkage, contiguity, seed, MCTS config, summary mismatch) is a MISMATCH.

## 5. Pre-screen & select verification (amendment 2; decision #6)

**Config-tamper detection is by re-derivation, not embedded-hash rechecks (amendment 2).** Recomputing the
hashes *embedded in* the config only proves the reservoir/protocol/code bytes are unchanged — it cannot catch
an edit to a config field that isn't itself hashed (`selection_seed`, `select_out`, `screen_out`, a floor).
Because the config is `derive(protocol, reservoir)` (§2), the complete check is to **re-derive the canonical
config from the pinned protocol + reservoir and byte-compare it against the supplied config** — the *same*
derivation `qualify --check` uses. Any tampered field, hashed or not, fails the byte-comparison.

- `qualify` pins every §2.2 path+hash identity into the config and is the sole producer of the canonical
  `derive(protocol, reservoir)`.
- **Before screen** — `run_screen` startup, **before the evaluator loads**: (1) recompute every pinned hash
  from its pinned path and hard-match (catches a changed reservoir/protocol/summary/code); (2) **re-derive the
  canonical config from `(protocol, reservoir)` and byte-compare** against the supplied config (catches *any*
  edited config field — this is the real config-tamper check, not the hash rechecks); (3) verify the config
  binds *this* protocol (`protocol_sha1`); (4) **repeat the geometric preflight defensively** (amendment 1).
  Any failure aborts before the hours-long screen starts.
- **During select** — the existing Task 6 chain re-verifies the same reservoir/config identities, extended to
  include the `protocol` and `match_summary` identities, and likewise **re-derives + byte-compares** the
  config, **plus** verifies the screen artifact (`screen_csv_sha1` + row cross-check). Reservoir/config are
  checked **twice** (pre-GPU and at select); the screen output once (at select).

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

**Preflight injection for tests (test clarification).** The pure qualification core accepts the preflight as
an **injected dependency** (default = the real `v2_preflight_source`); the CLI always wires the real one. This
lets protocol-conformance tests use a small fabricated reservoir with a fake preflight, while a *separate*
test exercises the real `v2_preflight_source` on a genuinely feasible (larger) synthetic reservoir — because a
6-game fixture cannot satisfy the frozen 240-row/4-phase geometric quotas (see §11).

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
- **`no_top_up` is procedural, not enforceable (amendment 3):** `qualify` sees only the *final* directory of
  `protocol.games` games — it cannot determine whether they were generated in one session or filled
  incrementally across runs. The half-open seed range, contiguity, and per-game seed checks make an
  *inconsistent* top-up detectable (a filled-in game with the wrong seed/params fails MISMATCH), but a
  disciplined incremental fill that happens to match every declared parameter is indistinguishable from a
  single-session generation. `no_top_up` is therefore an operator-discipline rule the protocol *declares*, not
  a property qualification *proves*.

## 11. Test matrix (all zero-GPU; game count is protocol-declared)

**Fixture sizing (test clarification).** Protocol-conformance and report/immutability tests use a small
fabricated reservoir (a 6-game protocol) with the preflight **injected as a fake** (§6) — a 6-game reservoir
cannot satisfy the frozen 240-row/4-phase geometric quotas, so it must never reach the *real* preflight. The
geometric-feasibility paths are covered by a **separate** test that drives the **real `v2_preflight_source`**
on a genuinely feasible (larger) synthetic reservoir sized to clear the quotas. The CLI always wires the real
preflight.

- **MISMATCH (exit 3):** wrong seed / wrong matchup / wrong game count / any of the ten MCTS knobs wrong /
  non-contiguous indices / broken replay linkage / wrong model colors / summary↔JSONL aggregate mismatch /
  wrong JSONL-vs-summary path derivation / generation-provenance mismatch. *(fake preflight → feasible.)*
- **GATE-FAIL (exit 4):** a protocol-faithful reservoir whose **real** `v2_preflight_source` geometry is
  infeasible (larger synthetic reservoir); the report records **retirement**, and re-qualifying the retired
  protocol is refused.
- **Screen precheck:** a stale fingerprint (edited code / byte-changed reservoir / tampered config or protocol
  or summary) *and* an edited non-hashed config field (`selection_seed`, `select_out`) are each rejected
  **before evaluator load** — the latter caught by the re-derive-and-byte-compare (§5); the defensive preflight
  re-runs.
- **Config bootstrap reproducibility:** byte-identical re-emit; `--check` diff; refuse-overwrite-different;
  idempotent-on-byte-identical; `--check` never writes.
- **Immutability ops:** atomic write verified; deterministic report on pass and on fail; MISMATCH report does
  **not** retire (regeneration still allowed) while GATE-FAIL does.
- **Combined audit (with Group 1):** v1 byte-identity (build + diagnostic) and v2 phase-gated operator
  behavior.

## 12. Progression
```
emit protocol → emit exact command → generate once → qualify + geometric preflight →
emit immutable config → screen precheck → screen → select recheck
```
Downstream (unchanged): controls → candidates → frozen sweep (per-phase new-collapse gate via Group 1) →
frozen check → strength match.
