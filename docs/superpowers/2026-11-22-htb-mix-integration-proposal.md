# H-MX integration proposal: serve the tested export `627c8d774df838a8` as `hard` (2026-11-22)

**Status:** PROPOSAL, revision 0, for review. Nothing here is built, copied, loaded, run or deployed.
`HARD_MODEL_ID` is still `78c16a31c3dba37f`.

**Authorized (the user, 2026-11-22):** writing this proposal and its ledger record. The read-only inspection it rests
on: the server source, the export record, the match evidence, and a model-free scratch probe of the seed preload (§5.2).

**Not authorized:** copying the model into `models/`, editing `HARD_MODEL_ID` or any server file, loading a model,
starting a server, playing or replaying any move, deploying, pushing. The stale H-OP test fix
(`tests/test_htb_onpolicy.py::test_EVERY_ENTRY_REFUSES_WHILE_ITS_GATE_IS_CLOSED_AND_WRITES_NOTHING`) is a separate
item and is not touched here.

**Links:** ledger Post-closure entries 76 (export attempt 2, PARITY_PASS), 77 (smoke) and 78 (match, PASS); card
`docs/superpowers/2026-11-18-htb-mix-distillation-card.md` §13 ("PASS → proposing an integration decision (export
pin, G2/G3-style checks), and nothing more"); the H-TB integration proposal
(`docs/superpowers/2026-11-06-htb-integration-proposal.md`) §7–§8, whose G2 checks and rollback this follows.

## 1. Scope and recommendation

**One change:** `hard` (under `HARD_MODE=protocol`, the default) serves `627c8d774df838a8` instead of
`78c16a31c3dba37f`. Nothing else in the served product changes:

| | today | proposed |
|---|---|---|
| `hard` model | `78c16a31c3dba37f` | **`627c8d774df838a8`** (the exact tested export) |
| `hard` evaluator | `MirrorAveragingInference(AlphaZeroInference)` | unchanged |
| `hard` search | `PROTOCOL_HARD`: 200 simulations, `search: 'protocol'` | unchanged |
| opening (plies 0–19) | `ProtocolMCTS` + temperature-1.0 sampling (`moveTemp` 1.0) | unchanged |
| plies ≥ 20 | root sequential halving (`HALVING`: from ply 20, 16 candidates, `c_visit` 50, `c_scale` 1.0) | unchanged |
| per-request seeds | two fresh 48-bit streams (`protocol_move.js` `freshSeed`) | unchanged |
| `easy`, `medium`, analysis endpoints | `c34b7ff3297c785a`, plain MCTS | unchanged |
| `HARD_MODE=legacy` | `c34b7ff3297c785a`, 800 simulations, argmax | unchanged |
| runtime | node v24.22.0, onnxruntime-node 1.23.2 | unchanged |

**Why this is only a pin.** The export's graph signature (ops, inputs, outputs, initializer shapes) equals
`78c16a31c3dba37f`'s (entry 76). It passed S1–S4 and V3 parity on the product's runtime. The H-MX match (entry 78)
played it **in-process through the product's own modules**: `protocolMove`, `PROTOCOL_HARD`, `MirrorAveragingInference`
over `AlphaZeroInference`, loaded through the product's `loadManifest` and `validateArtifact`, exactly as
`server/index.js` builds `hard` (§2). The match's opponent was today's `hard` itself. So no new match is proposed; the
checks below confirm that the served server reproduces the tested player and that the rollback restores today's.

**Recommendation:**
1. Track the three export files, byte-identical, in `models/627c8d774df838a8/` (§3).
2. Change one line, `HARD_MODEL_ID`, in its own commit together with its test pins (§4), so that one `git revert`
   undoes the whole swap.
3. Run the checks of §5 on a staged branch before anything reaches `htb-integration-g1`.
4. Keep `models/78c16a31c3dba37f/` tracked and byte-identical: the rollback is a restart with
   `HARD_MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json` (§6).

## 2. What was tested, exactly

| item | value | source |
|---|---|---|
| challenger | `627c8d774df838a8` | export attempt 2 record, `docs/superpowers/evidence/htb-mix-export-2/record.json` |
| `model.onnx` | 84,233 bytes, sha256 `3beaeed15a97f28c69e639b6c5ab2466f961a1aa75f232755e581d592fcb55af` | export record; match record `challenger.graph_sha256` |
| `model.onnx.data` | 7,493,120 bytes, sha256 `cf78b17f7e9368e5be850ded8b41d3c13921f32bf54c3fab0273cfcb8489822f` | same |
| `manifest.json` | 3,206 bytes, sha256 `abf3931d920f788de146cf8dafbd213581f682a754b65df1520d42eae954071d` | `~/twixt-cache/htb-mix/export/627c8d774df838a8/` (read 2026-11-22) |
| model id | sha256(graph_sha + ':' + data_sha)[:16] = `627c8d774df838a8` | `computeModelId` |
| source | `challenger.safetensors` sha256 `5dafde11…`, state digest `adba50fb…` (entry 74) | manifest provenance |
| match | 400 colour-reversed pairs, 200 simulations each side, `MirrorAveragingInference(AlphaZeroInference), both`, protocol search, `moveTemp` 1 | `docs/superpowers/evidence/htb-mix-match/record.json` (`6bfa8bb6…`) |
| result | PASS: decisive pairs 31–15, exact one-sided p = 0.012948; mean pair score 0.519375; compute ratio 1.00323 | `analysis.json` (`2a58660e…`), entry 78 |
| construction | `tests/htb_integration/mix_match.mjs:276–292`: `loadManifest` + `validateArtifact` on the export directory, `new AlphaZeroInference(graphPath)`, `load()`, `new MirrorAveragingInference(base)` | the same calls as `server/index.js:674–704` |

**Scope of the evidence:** PASS shows that the challenger wins more decisive pairs than it loses against today's
`hard`, over fresh 4-ply openings under this 200-simulation protocol. It does not estimate a size of advantage,
and it says nothing about `easy`, `medium`, human opponents or twixtbot (§8).

## 3. Proposed file changes

The build has two commits so that the swap can be reverted alone.

**Commit (i): track the tested artifact (no behaviour change).**

| file | change |
|---|---|
| `models/627c8d774df838a8/model.onnx` | new; byte copy of the export (`3beaeed1…`) |
| `models/627c8d774df838a8/model.onnx.data` | new; byte copy (`cf78b17f…`) |
| `models/627c8d774df838a8/manifest.json` | new; byte copy (`abf3931d…`), **unedited** (decision D1) |
| `server/test_hard_model.js` | the retained-pin loop gains `627c8d774df838a8` beside `c34b7ff…`, `1d64027…` and `HARD_MODEL_ID`; a new test checks the three files against the export record's hashes and the match record's `challenger` block, and recomputes the id |

The `.gitignore` exception (`!models/*/*.onnx*`) already admits the files. Commit (i) changes no execution-surface
file, so no golden or binding pin moves.

**Commit (ii): the swap (one product line).**

| file | change |
|---|---|
| `server/model_manifest.js` | `HARD_MODEL_ID = '627c8d774df838a8'`; the doc comment above it names entry 76 (parity) and entry 78 (match PASS) and keeps `78c16a31c3dba37f` as the named rollback pin |
| `server/test_hard_model.js` | `:112–113` and `:167–169` re-pointed to `627c8d774df838a8` and source sha256 `5dafde11…`; a new test that `HARD_MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json` resolves `78c16a31c3dba37f` under protocol, including in a fixture tree with `models/627c8d774df838a8/` absent (the rollback works when the new artifact is missing or damaged) |
| `tests/mcts_golden/cases.mjs` (+ `test_capture.mjs`, `test_falsify.mjs`) | a new frozen stage `protocol-hard-hmx`, not re-measured, exactly the `06b52aaa` pattern: identical to `protocol-hard-pinned` except `HARD_MODEL_ID`; plain search files byte-identical; the current-HEAD capture and attribution tests re-pointed; earlier stages keep their digests |
| `tests/htb_integration/test_g3.mjs` | "the committed binding equals a fresh computation" would fail, because the spent G3 binding (`g3_binding.json`) lists `server/model_manifest.js`. Proposed: the test asserts that the only difference is that one file's digest, and the binding itself stays as committed (decision D3) |

**New check tooling (model-free; used only by §5's run):**

| file | purpose |
|---|---|
| `tests/htb_integration/hmx_seed_preload.mjs` (~50 lines) | a test-only `node --import` preload (§5.2); never imported by product code |
| `tests/htb_integration/hmx_integration_check.mjs` (~400 lines) | the §5 driver: starts servers as child processes, replays the recorded rows, captures and compares; writes the record |
| `tests/htb_integration/hmx_integration_authorization.mjs` | one new gate, `HMX_INTEGRATION_CHECK_AUTHORIZED = false`; the gate inventory test derives the new count |
| `docs/superpowers/evidence/hmx-integration-check-run/01_wrapper.sh` | derived from the H-MX match wrapper (the idle gate, the BOINC check, the one-line gate commit, the ONNX Runtime shutdown test, the 1.23.2 pin and a real `node_modules`, the perl process-group supervisor, `outer.json`, gate restoration), with this stage's paths, driver and ceiling |
| `tests/htb_integration/test_hmx_integration.mjs` | model-free tests of the preload, the replay selection, the comparators and the record admission, each against a clean control |

**Not changed:** every other `server/` file (`index.js`, `protocol_move.js`, `protocol_mcts.js`, `root_halving.js`,
`py_random.js`, `mirror_inference.js`, `readout_policy.js`, `mcts.js`, `inference.js`, `gameLogic.js`,
`cache.js`), `models/78c16a31c3dba37f/`, `models/c34b7ff3297c785a/`, `models/1d64027db521a50f/`, `DEFAULT_MODEL_ID`,
`package.json`/lockfile, the client, and every H-MX, H-OP and H-TB research file and evidence. The spent match
drivers (`mix_match.mjs`, `onpolicy_match.mjs`, `teacher_bench.mjs`, `diag_o.mjs`, `g3_match.mjs`) resolve the
current `hard` through `resolveHardModel` and pin `78c16a31c3dba37f`. After the swap they would refuse to run, which
is the intended state for spent stages. They are not edited.

## 4. Build acceptance (model-free; the build approval)

- **B1. Artifact identity:** the three files equal the export directory and the hashes of §2. `loadManifest` and
  `validateArtifact` accept them in place. `computeModelId` gives `627c8d774df838a8`.
- **B2. Diff allowlist:**
  - Commit (ii)'s change to `server/` is exactly the `HARD_MODEL_ID` line and its comment, checked by a test on
    `git diff -U0`.
  - Every other `server/` file is byte-identical to the base commit.
  - `PROTOCOL_HARD` (200, `'protocol'`, 1.0), `LEGACY_HARD` (800, `'plain'`, 0), `HALVING` and the easy and medium
    entries are asserted unchanged by value.
- **B3. Pins:** the `test_hard_model.js` changes of §3, including the rollback-setting test with the new artifact
  absent.
- **B4. Golden stage:** `protocol-hard-hmx` digest computed and frozen; earlier stages unchanged.
- **B5. Seed preload, against controls:**
  - it delivers a queued pair in order to `protocolMove`'s `freshSeed`;
  - it fails closed when the queue is exhausted;
  - it passes every other `randomInt` call through;
  - it refuses to load without its environment flag;
  - a grep guard shows no product file imports it.
- **B6. Suites:** server, parity, golden, match and the `tests/test_htb_*` research suites, with counts recorded,
  plus lint. The four known stale H-OP failures are listed and left alone; any other failure stops the build for
  review.
- **B7. Wrapper:** derived byte-for-byte by a spec module, as the match wrappers are, and refused-path tests run on
  the real wrapper in a throwaway repo (the export-2 pattern).

## 5. The checks: G2/G3-style, one gated run (loads models; replays recorded positions; plays no new game)

### 5.1 Trees and procedure

- **Staged tree `<W>`:** a clean linked worktree at commit (ii), on branch `hmx-integration-check-launch`, with
  `node_modules` APFS-cloned (`cp -c -R`). Node, run from `<W>`, must resolve `onnxruntime-node` at 1.23.2 from
  `<W>`'s own copy. The tree must be clean before the one-line gate commit.
- **Reference tree `<W>-ref`:** detached at the base commit (today's product: `HARD_MODEL_ID = 78c16a31c3dba37f`),
  with `node_modules` cloned and verified the same way.
- **Revert tree `<W>-revert`:** a throwaway branch with `git revert <commit (ii)>` committed, `node_modules` cloned
  and verified. Its `server/`, `models/` and `tests/mcts_golden/` must equal commit (i)'s tree, which the driver
  checks before starting it.
- Then: the wrapper runs once in the background, with console output in the scratchpad. The closing commit uses exact
  paths and is verified, and the fast-forward is its own step. The ledger entry follows.
- **Branch order:** on `hmx-integration` the commits run (i) → tooling → build record → **(ii) last**. So
  `htb-integration-g1` can be fast-forwarded to the build without the swap. The gate, evidence and ledger commits
  stack on (ii) and stay on `hmx-integration` until the deployment decision (§7, step 4).
  - If deployment is approved, that branch is fast-forwarded.
  - If it is declined, a `git revert` of (ii) is committed on top first, so the evidence lands linearly without the
    swap.
- **Ceiling:** TERM 14,390 s, KILL 14,400 s (4 h). The estimate is about 2.5 h (§5.6). A timeout or failure is
  VOID, with no automatic retry.

### 5.2 Fixed seeds through the real server: the test-only preload

`server/index.js` never passes seeds (`protocol_move.js:9–11`), so no request can choose a stream. To make
transport answers comparable with recorded rows **without editing product code**, the check starts servers as
`node --import tests/htb_integration/hmx_seed_preload.mjs server/index.js`. The preload works like this:

- **Seed source:** it replaces `crypto.randomInt` only for the call `randomInt(0, 2**48 - 1)`, which is
  `freshSeed`'s. It serves that call from a seed file named by the environment, then calls
  `module.syncBuiltinESMExports()`, so `protocol_move.js`'s `import { randomInt } from 'node:crypto'` sees the
  queue.
- **Ordering:** each `hard` request consumes two values, search then readout (the object-literal order at
  `protocol_move.js:58`). Requests are sent strictly one at a time, each awaited.
- **Fail-closed:**
  - an exhausted queue throws;
  - every consumed value is logged with its index to stderr;
  - the driver checks that log against the queue at the end, so drift cannot pass silently.
- **Scratch probe (2026-11-22, model-free, node v24.22.0):** a toy module importing `randomInt` from `node:crypto`
  received the queued values in order and threw on exhaustion, and without the preload it drew fresh values. B5
  repeats this against the real `protocol_move.js` with a fake inference.
- **Limit, stated:** with the preload, the checks cover everything downstream of the seeds. Unseeded serving, the
  `crypto.randomInt` draw itself, is covered by S2's live requests (legality and fields), not by exact comparison.

### 5.3 The staged server serves the pinned export (S1, S2)

- **S1. Startup:**
  - `<W>`'s server starts with no model environment (HARD_MODE, MODEL_MANIFEST and HARD_MODEL_MANIFEST cleared).
  - The banner shows `Hard mode: protocol`, `Hard model id: 627c8d774df838a8` and
    `Model id: c34b7ff3297c785a`.
  - `/api/model-info` reports `hard_model_id: '627c8d774df838a8'`, `hard_mode: 'protocol'` and
    `model_id: 'c34b7ff3297c785a'`.
  - The session contract is asserted at load; the driver records the banner.
- **S2. Live REST and WS:** G2's `live_check.mjs` checks L1–L9, carried into the new driver with the expected hard id
  as a parameter (`627c8d774df838a8`). The spent tool hard-codes `78c16a31c3dba37f` and is not edited:
  - protocol fields, legality, visit totals and `valueKind` on both sides of ply 20;
  - value perspective and progress messages;
  - REST `cached: false` with the cache untouched, against the medium cache control;
  - conflicting overrides refused with one message on both transports;
  - won positions and 12×12 boards refused;
  - cancel and latest-wins;
  - easy and medium key sets;
  - per-move timing.

  PASS requires every check.

### 5.4 Fixed-seed moves agree with the match harness (S3)

- **Rows replayed (fixed now, before any run):**
  - every challenger row of the 40 match games in pairs p ≡ 0 (mod 20), i.e. pairs 0, 20, …, 380: 561 moves, 241 of
    them at ply ≥ 20;
  - plus every challenger row of the longest match game (`pair_0153_game_1`, 572 plies): 284 moves, which reach the
    nearly full board, where halving has fewer than 16 candidates.
  - **845 rows** in all.
  - Each row gives the position (the game's move list up to that ply), the recorded seeds `{search, readout}`, the
    recorded move, `nSimulations` 200 and root `visits` 200.
- **A0, in-process:** `resolveHardModel({})` in `<W>` (no environment) gives `627c8d774df838a8`, built as
  `server/index.js` builds it. `protocolMove` is called with the recorded seeds. This separates any construction
  difference from a transport difference.
- **A1, WS:** the staged server under the preload; one WS `move` per row with `difficulty: 'hard'` and
  `includeVisits`. Rows are sent in order, and the seed file holds the rows' seeds in the same order.
- **A2, REST:** the same, via `POST /api/move`, with a fresh server and seed file.
- **Pass rule, per path:**
  - the move equals the recorded move on all 845 rows;
  - `nSimulations` is 200 and the visits sum to 200;
  - `search` is `'protocol'` and `modelId` is `627c8d774df838a8`;
  - **no tolerance:** one mismatch fails that path.
- Expected exact: the same code, runtime, Node binary, machine and seeds as the match. Entry 60 found value
  numerics differing about 1e-7 across Node versions, but the Node version here is the match's own.

### 5.5 Easy and medium unchanged; rollback restores today's `hard` (S4, R1, R2)

**Positions:** the 142 positions of G2's `rollback_check.mjs`: the 126 non-terminal Phase 2 parity-corpus positions
and the 16 golden positions.

**Hard seeds for these captures (fixed now):** for position `id`, search = the first 12 hex digits of
sha256(`hmx-integration/20261122/<id>/search`), read as a 48-bit integer, and likewise for readout. The captures use
WS, one difficulty at a time, `includeVisits`.

- **Reference:** `<W>-ref` (today's product), under the preload. For each position it records:
  - `hard`: move, visits, `rootValue`, `moveValue`, `moveVisits`, `valueRed`, `valueKind`, `nSimulations`,
    `search`, `modelId` and the key set;
  - `easy` and `medium`: visits, `rootValue`, `valueRed`, `nSimulations` and the key set. Their moves are sampled
    with `Math.random` and are recorded but not compared, as in G2.
- **S4. Easy and medium retain today's behaviour:**
  - The staged `<W>` server's easy and medium search outputs equal the reference **exactly** on all 142 positions.
  - Its `hard` answers carry `modelId` `627c8d774df838a8`, `search: 'protocol'` and 200 simulations, on all 142.
    They are expected to differ from the reference, which is a different model.
- **R1. Rollback by restart (the operational rollback of §6):**
  - Setup: a throwaway copy of `<W>` with `models/627c8d774df838a8/` removed, started under protocol with
    `HARD_MODEL_MANIFEST=<copy>/models/78c16a31c3dba37f/manifest.json`.
  - The banner shows `Hard model id: 78c16a31c3dba37f`.
  - (a) Every `hard` field of the 142-position seeded capture, and every easy and medium search output, equals the
    reference **exactly**.
  - (b) Fixed-seed replay of the match's **current-`hard`** rows from the same 41 games: 844 rows, all moves equal,
    over WS.
  - This shows the restart restores the `hard` the match actually played against, even with the new artifact
    absent.
- **R2. Rollback by code revert:**
  - `<W>-revert`, with no environment, serves `78c16a31c3dba37f` by default.
  - (a) and (b) as in R1, exact.
  - The server, parity and golden suites pass in `<W>-revert`. The golden current-HEAD test passes against
    `protocol-hard-pinned` again, because the revert restores `HARD_MODEL_ID`.
  - The throwaway branch and tree are then removed, not merged.

### 5.6 Timing and the browser (G3-style; no new match)

- **T. Per-move time:** recorded from A1 for plies < 20 and ≥ 20, against the match's recorded `ms` for the same
  rows. This is informational: the same graph signature and budget should give the same cost, and the match's
  compute ratio was 1.00323.
- **V. Browser:** one game in the browser client against the staged server, on localhost:
  - the banner and `/api/model-info`;
  - every AI move arrives;
  - the MCTS bar updates across ply 20;
  - no client error;
  - a normal game end.

  This is a qualitative smoke, as in entry 62. It runs after the wrapper, outside the ceiling; you or I can drive it.
- **Estimate:** reference and staged captures about 5 min each; S2 about 2 min; A0, A1 and A2 about 21 min each
  (845 moves at the match's 1.49 s mean); R1 and R2 about 26 min each, plus R2's suites about 10 min. That is about
  2.5 h in all, under the 4 h ceiling.

### 5.7 Outcomes (fixed before the run)

| outcome | reading | what it permits |
|---|---|---|
| any check FAIL | the served product does not reproduce the tested player, or the rollback does not restore today's `hard` | nothing; a reviewed decision; no silent repair or retry |
| VOID (ceiling, crash, interruption) | no result | nothing; no automatic retry |
| **all of S1–S4, A0–A2, R1, R2 PASS** (T informational; V a smoke) | the staged server serves the exact tested export through both transports, reproducing the match harness move for move; easy and medium are unchanged; both rollbacks restore today's `hard` exactly | proposing deployment (§7, step 4), with its own approval |

**Prediction on record:** all checks PASS, about 90%. The likeliest failure is a harness defect in the new driver,
not in the product: preload ordering, or the WS request shape for a mid-game position. Next likeliest is an
environment-numerics difference like entry 59's, which a same-Node, same-machine replay should not have.

## 6. Rollback procedure (restores `78c16a31c3dba37f` in protocol mode, today's `hard`)

**Kept for it, byte-identical and tracked:** `models/78c16a31c3dba37f/`:

| file | sha256 |
|---|---|
| `manifest.json` | `81bbf545ec332d5a56541fcb79c95a0ab1b94bb6176118e21fcd73733ccfd0d7` |
| `model.onnx` | `bc68a2742803a8466856fa353958b168becff1044dd387da001ddcaeb7e54f24` |
| `model.onnx.data` | `317d7a7c0f704c5ab5527ff82599031911558dd04b522165b4bc4260b29f7f48` |

`test_hard_model.js` re-hashes them in every server test run (B3).

**Step 1: immediate, by restart (no edit; tested by R1).**
1. Stop the server.
2. Start it with `HARD_MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json npm start` (relative to the repo root;
   `HARD_MODE` unset, so protocol).
3. Confirm that the banner shows `Hard mode: protocol` and `Hard model id: 78c16a31c3dba37f`, and that
   `/api/model-info` reports `hard_model_id: '78c16a31c3dba37f'`.

The override never reads `models/627c8d774df838a8/`, so it works even if that directory is missing or damaged.
`easy` and `medium` are unaffected either way.

**Step 2: durable, by code revert (tested by R2).**
1. Run `git revert <commit (ii)>`. It restores `HARD_MODEL_ID = '78c16a31c3dba37f'`, the test pins and the golden
   current-HEAD stage in one commit.
2. Commit (i)'s files stay tracked and harmless.
3. Then restart with no override and confirm the banner as above.

**Not the rollback:** `HARD_MODE=legacy`. That serves `c34b7ff3297c785a` at 800 simulations with argmax, the
pre-H-TB `hard`, not today's. It remains the deeper fallback, unchanged.

## 7. Approvals, in order (none given)

1. **This proposal**, with decisions D1–D4.
2. **Build:** commits (i) and (ii) and the tooling of §3, on branch `hmx-integration` from the then-current
   `htb-integration-g1`, with B1–B7, in the order of §5.1. The swap commit (ii) **does not reach
   `htb-integration-g1`**. Only commit (i), the tooling and the build record are fast-forwarded. All gates stay
   closed. Returned for launch review.
3. **The check run** of §5 (one wrapper run, 4 h ceiling), then the browser smoke V. Evidence and ledger entry
   follow. Returned for review.
4. **Deployment:**
   - fast-forward `hmx-integration` (commit (ii) with the check evidence) into `htb-integration-g1`;
   - restart your server with no override;
   - confirm the banner shows `627c8d774df838a8`.

   Pushing remains a separate decision.

## 8. Product behaviour outside the tested evidence

1. **Strength scope:** PASS over fresh 4-ply openings, at 200 simulations, against today's `hard` only. The
   product's `hard` starts from the empty board. Its plies 0–3 come from the same T 1.0 opening readout that the
   match's openings replaced. There is no evidence against humans, `legacy` or twixtbot, and no size of advantage.
2. **Seat effect:** Red scored about 0.84–0.88 in both seatings (entry 78). The players' colour asymmetry is
   unchanged and not addressed by this swap.
3. **Game end:** the match ran under the product's own game-end rule (no 280 cap), as the server does.
4. **Displays and records:** same fields, same `valueKind` definitions; only `modelId` changes in responses and the
   banner.
5. **Caching:** protocol requests bypass the cache as before. The REST cache key for plain searches is unaffected.
6. **Latency:** expected unchanged (same graph signature); measured by T.

## 9. Decisions for the user

- **D1. The manifest file:**
  - (a) **Byte-identical copy (recommended).** It pins exactly the files the match tested, and its sha256 is the one
    already on record. Its `description` and `claims` say "held outside the product manifest … never served as hard
    by this record" and "no … serving-status claim". Both are true of that record. Serving status is recorded in
    `model_manifest.js`'s comment and the ledger, as the claims field directs. Its `provenance_note` says
    "`htb_mix_export run export`"; the run was attempt 2 (`run export-2`, the same export body). That slip is noted
    here and in the ledger and left as written.
  - (b) A product-written manifest with the same file entries and new wording. The model id is unchanged, but the
    manifest hash is new and untested by the match. Not recommended.
- **D2. No new match.** Recommended, because the match already ran the product modules against today's `hard`
  (§1). The alternative is a product-stack match like H-TB's G3 V8b, against `legacy` or today's `hard`, at about 7 h
  under its own card.
- **D3. The spent G3 binding test:** (a) assert that the only drift is `server/model_manifest.js`'s digest
  (recommended), or (b) compute the fresh binding from the G3 launch commit's git objects.
- **D4. Browser smoke V:** I drive it through Claude in Chrome on localhost, or you play it yourself.

## 10. Effort

| step | work |
|---|---|
| build | 3 copied files; 1 product line + comment; ~60 test lines (`test_hard_model.js`); golden stage ~20 lines + 2 test files (the `06b52aaa` pattern); G3 test ~5 lines; preload ~50 lines; driver ~400 lines; wrapper spec + tests ~300 lines |
| check run | ~2.5 h CPU under a 4 h ceiling; browser smoke ~10 min |
| deployment | a fast-forward and a restart |
