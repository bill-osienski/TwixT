# H-MX integration proposal: serve the tested export `627c8d774df838a8` as `hard` (2026-11-22)

**Status:** PROPOSAL, revision 1 (2026-11-23), returned for build approval. Nothing here is built, copied, loaded, run
or deployed. `HARD_MODEL_ID` is still `78c16a31c3dba37f`. Revision 0 (sha256 `b72593fa…`, commit `423374f6`) is
superseded; what changed is listed in §11.

**Authorized (the user, 2026-11-22):** writing this proposal and its ledger record. The read-only inspection it rests
on: the server source, the export record, the match evidence, and a model-free scratch probe of the seed preload (§5.2).

**Review of revision 0 (the user, 2026-11-23):** D1 and D2 accepted; D3 accepted with a precise binding assertion;
D4: I drive the browser smoke against the staged server with its normal seed source, and it is a required check.
Also: each hash cites the record that actually holds it, with the manifest pinned explicitly for packaging; the build
stays model-free, with exact commands, and the model-loading suites move to the gated run. Revision 1 does this.
Revision 1 is authorized as a document only; the build needs its own approval.

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

Each hash below cites the committed record that actually holds it. "Export record" is
`docs/superpowers/evidence/htb-mix-export-2/record.json`; "match record" is
`docs/superpowers/evidence/htb-mix-match/record.json` (`6bfa8bb6…`).

| item | value | evidence |
|---|---|---|
| challenger id | `627c8d774df838a8` | export record `challenger.model_id`; match record `challenger.model_id` |
| `model.onnx` | sha256 `3beaeed15a97f28c69e639b6c5ab2466f961a1aa75f232755e581d592fcb55af` | export record `challenger.graph_sha256`; match record `challenger.graph_sha256` (and `onnx_sha256`). Both agree (checked by the user and by me, 2026-11-23) |
| `model.onnx.data` | sha256 `cf78b17f7e9368e5be850ded8b41d3c13921f32bf54c3fab0273cfcb8489822f` | export record `challenger.external_data_sha256`; match record `challenger.external_data_sha256`. Both agree |
| sizes | 84,233 and 7,493,120 bytes | **in neither record.** They come from the manifest and are checked against the files themselves by `validateArtifact` |
| `manifest.json` | 3,206 bytes, sha256 `abf3931d920f788de146cf8dafbd213581f682a754b65df1520d42eae954071d` | **in neither record, and not tested evidence.** It is a **packaging pin**, set explicitly here, from the bytes in `~/twixt-cache/htb-mix/export/627c8d774df838a8/` (the export record's `challenger.dir`), read 2026-11-22 and re-read 2026-11-23. The match loaded the challenger through this directory's manifest (`mix_match.mjs:276`) but recorded no hash of it. So what makes it the tested artifact is that it names the qualified graph and data (B1), not its own hash |
| model id | sha256(graph_sha + ':' + data_sha)[:16] = `627c8d774df838a8` | `computeModelId`, recomputed from the two record hashes |
| source | `challenger.safetensors` sha256 `5dafde1152bf83af…` | export record `admission.challenger.sha256`; match record `challenger.source_sha256`; manifest `provenance.source_checkpoint_sha256` |
| state digest | `adba50fb…` | manifest `provenance.source_state_digest`; entry 74 |
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
| `models/627c8d774df838a8/manifest.json` | new; byte copy (`abf3931d…`), **unedited** (D1, accepted) |
| `server/test_hard_model.js` | the retained-pin loop gains `627c8d774df838a8` beside `c34b7ff…`, `1d64027…` and `HARD_MODEL_ID`; a new test does B1 (below) |

The `.gitignore` exception (`!models/*/*.onnx*`) already admits the files. Commit (i) changes no execution-surface
file, so no golden or binding pin moves.

**Commit (ii): the swap (one product line).**

| file | change |
|---|---|
| `server/model_manifest.js` | exactly the diff below: `HARD_MODEL_ID = '627c8d774df838a8'`, and the doc comment above it names entry 76 (parity), entry 78 (match PASS) and the rollback pin `78c16a31c3dba37f`. The file's sha256 goes from `ecde544b03b4e50f673edaf89f73434c878bfbda0a147341adb9197d788fc511` to **`054f9434071d1502482550fb91ed34b57462ba5a29a02ab9cbcc559dab06f3c2`** (computed on a scratch copy, 2026-11-23; the repository file is untouched) |
| `server/test_hard_model.js` | `:112–113` and `:167–169` re-pointed to `627c8d774df838a8` and source sha256 `5dafde11…`; a new test that `HARD_MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json` resolves `78c16a31c3dba37f` under protocol, including in a fixture tree with `models/627c8d774df838a8/` absent (the rollback works when the new artifact is missing or damaged) |
| `tests/mcts_golden/cases.mjs` (+ `test_capture.mjs`, `test_falsify.mjs`) | a new frozen stage `protocol-hard-hmx`, not re-measured, exactly the `06b52aaa` pattern: identical to `protocol-hard-pinned` except `HARD_MODEL_ID`; plain search files byte-identical; the current-HEAD capture and attribution tests re-pointed; earlier stages keep their digests |
| `tests/htb_integration/test_g3.mjs` | "the committed binding equals a fresh computation" (`:346–353`) would fail, because the spent G3 binding (`g3_binding.json`) hashes `server/model_manifest.js` among its 11 `driver_files`. That one test is rewritten to the precise assertion of §3.1 (D3, accepted). `g3_binding.json` and all G3 evidence stay byte-identical |

**The exact change to `server/model_manifest.js`** (lines 61–67 today):

```diff
@@ -61,5 +61,7 @@
- * The validated H-TB student's export, pinned after its export parity passed
- * (V2-R, ledger Post-closure entry 58, under the approved amendment
- * docs/superpowers/2026-11-09-htb-student-parity-amendment.md). Changing it is a
- * tracked, reviewable diff, as for DEFAULT_MODEL_ID. HARD_MODE=legacy never
- * reads it.
+ * The H-MX challenger's export, pinned after its export parity passed (ledger
+ * Post-closure entry 76) and its 400-pair match against 78c16a31c3dba37f passed
+ * (entry 78), under docs/superpowers/2026-11-22-htb-mix-integration-proposal.md.
+ * Rollback pin: 78c16a31c3dba37f, by a restart with
+ * HARD_MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json or a revert of this
+ * line. Changing it is a tracked, reviewable diff, as for DEFAULT_MODEL_ID.
+ * HARD_MODE=legacy never reads it.
@@ -67 +69 @@
-export const HARD_MODEL_ID = '78c16a31c3dba37f';
+export const HARD_MODEL_ID = '627c8d774df838a8';
```

### 3.1 The G3 binding assertion (D3)

The committed binding `tests/htb_integration/g3_binding.json` (file sha256 `ef25d4da65f1d7d2…`, aggregate
`digest` `da0b86bf4dcbdf4364d26a5f4a483fa97360b41c60ed3abae15778f7a636c04a`) is the H-TB G3 match's historical
binding. It is kept, unedited. The aggregate is `sha256(JSON.stringify(body))` over the binding without its `digest`
key (`g3_match.mjs:330–352`). The rewritten test asserts all of the following, each as a literal:

1. **The historical binding is intact.** `g3_binding.json` hashes to `ef25d4da…`, its `digest` is `da0b86bf…`,
   recomputing that digest from its own body gives `da0b86bf…`, and `A.selfCheck(committed)` is `[]`.
2. **Exactly one bound file changed.** `bindingDiff(committed, computeBinding())` equals exactly
   `['driver_files.server/model_manifest.js', 'digest']`. Every other path is identical, including the other
   10 driver files, the analyser files, the 9 extra files, both model manifests, the Node executable and every file of
   the two runtime packages.
3. **The old and new hashes are the expected ones.** The committed entry is
   `ecde544b03b4e50f673edaf89f73434c878bfbda0a147341adb9197d788fc511`, and the fresh entry is
   `054f9434071d1502482550fb91ed34b57462ba5a29a02ab9cbcc559dab06f3c2`.
4. **The resulting aggregate digest is the expected one.** Substituting only the new entry into the committed body
   and hashing gives `aa04f04c36cbcdb9a9db41e46ab401eff423625c579a9b902116c3e2d80fedae`, and the fresh
   `computeBinding().digest` equals it.
5. **Kept as before:** the gate is not in `fresh.driver_files`, and the `match_seed` control still reports
   `['match_seed']`.

Points 3 and 4 were computed on 2026-11-23 from the committed binding and a scratch copy of `model_manifest.js`
carrying exactly the diff above. Both are model-free, and no repository file was changed. If the build's file differs
by one byte, point 3 fails, and the build stops for review; the literals are not adjusted to fit. G3 itself is spent:
its driver would now refuse to run, which is intended, and its evidence and record are not touched. A revert of
(ii) restores today's test with its `[]` assertion (R2).

**New check tooling (model-free; used only by §5's run):**

| file | purpose |
|---|---|
| `tests/htb_integration/hmx_seed_preload.mjs` (~50 lines) | a test-only `node --import` preload (§5.2); never imported by product code |
| `tests/htb_integration/hmx_integration_check.mjs` (~400 lines) | the §5 driver: starts servers as child processes, replays the recorded rows, captures and compares; writes the record |
| `tests/htb_integration/hmx_integration_authorization.mjs` | one new gate, `HMX_INTEGRATION_CHECK_AUTHORIZED = false`; the gate inventory test derives the new count |
| `docs/superpowers/evidence/hmx-integration-check-run/01_wrapper.sh` | derived from the H-MX match wrapper (the idle gate, the BOINC check, the one-line gate commit, the ONNX Runtime shutdown test, the 1.23.2 pin and a real `node_modules`, the perl process-group supervisor, `outer.json`, gate restoration), with this stage's paths, driver and ceiling |
| `tests/hmx_integration_wrapper_spec.py` | the wrapper's derivation spec (exact-once substitutions on the match wrapper), as `tests/htb_mix_match_wrapper_spec.py` is for the match |
| `tests/test_hmx_integration_wrapper.py` | the derivation test and the refused-path tests on the real wrapper in a throwaway repo (B7) |
| `tests/htb_integration/test_hmx_integration.mjs` | model-free tests of the preload, the replay selection, the comparators and the record admission, each against a clean control |

**Not changed:** every other `server/` file (`index.js`, `protocol_move.js`, `protocol_mcts.js`, `root_halving.js`,
`py_random.js`, `mirror_inference.js`, `readout_policy.js`, `mcts.js`, `inference.js`, `gameLogic.js`,
`cache.js`), `models/78c16a31c3dba37f/`, `models/c34b7ff3297c785a/`, `models/1d64027db521a50f/`, `DEFAULT_MODEL_ID`,
`package.json`/lockfile, the client, and every H-MX, H-OP and H-TB research file and evidence. The spent match
drivers (`mix_match.mjs`, `onpolicy_match.mjs`, `teacher_bench.mjs`, `diag_o.mjs`, `g3_match.mjs`) resolve the
current `hard` through `resolveHardModel` and pin `78c16a31c3dba37f`. After the swap they would refuse to run, which
is the intended state for spent stages. They are not edited.

## 4. Build acceptance (model-free; the build approval)

**Model-free** here means: no process creates an ONNX Runtime session on any graph, real or synthetic; none runs a
search on a real network; and none starts a server. Reading model files to hash them and parsing manifests is
allowed, because that is B1 itself. Importing `server/inference.js`, which loads the `onnxruntime-node` library but
no model, is also allowed.

**Where:** a clean linked worktree `/Users/bill/projects/TwixT_Game-hmx-build` on branch `hmx-integration`, from the
then-current `htb-integration-g1`. Its `node_modules` is APFS-cloned (`cp -c -R`), and Node, run from the worktree,
must resolve `onnxruntime-node` 1.23.2 from that copy before any test runs. Lint and every suite that imports
`server/inference.js` need it.

**Commit order on `hmx-integration`:** (i) → tooling → **(ii)** → build record. `htb-integration-g1` does not move
during the build or the check run (§5.1, §7).

- **B1. Artifact identity (in `server/test_hard_model.js`):**
  - The three tracked files are byte-equal to the export directory.
  - Graph `3beaeed1…` and data `cf78b17f…` equal **both** records' `challenger.graph_sha256` and
    `challenger.external_data_sha256`, read from the committed record files at test time.
  - **Packaging pin:** `manifest.json` hashes to the literal `abf3931d920f788de146cf8dafbd213581f682a754b65df1520d42eae954071d`.
  - **The manifest names the qualified pair:**
    - `model_id` is `627c8d774df838a8` and equals both records' `challenger.model_id`;
    - `graph.filename` is `model.onnx`, and `graph.sha256` equals the records' graph hash;
    - `external_data.filename` is `model.onnx.data`, and `external_data.sha256` equals the records' data hash;
    - `computeModelId(graph.sha256, external_data.sha256)` is `627c8d774df838a8`;
    - `provenance.source_checkpoint_sha256` equals the export record's `admission.challenger.sha256` and the match
      record's `challenger.source_sha256` (`5dafde11…`).
  - `loadManifest` and `validateArtifact` accept the directory in place, which checks both sizes and both file
    hashes against the manifest.
  - Each assertion has a tampered-copy control (one byte of the manifest, one hash field, one filename), which must be
    refused.
- **B2. Diff allowlist:**
  - Between the base commit and (ii), the changed paths are exactly those of §3.
  - (ii)'s change to `server/` is exactly the diff of §3, checked against `git diff -U0`, and the resulting
    `server/model_manifest.js` hashes to `054f9434…`.
  - Every other `server/` file is byte-identical to the base commit.
  - `PROTOCOL_HARD` (200, `'protocol'`, 1.0), `LEGACY_HARD` (800, `'plain'`, 0), `HALVING` and the easy and medium
    entries are asserted unchanged by value.
  - `models/78c16a31c3dba37f/`, `models/c34b7ff3297c785a/` and `models/1d64027db521a50f/` are byte-identical.
- **B3. Pins:** the `test_hard_model.js` changes of §3, including the rollback-setting test with the new artifact
  absent.
- **B4. Golden stage and G3 binding:**
  - The `protocol-hard-hmx` digest is computed and frozen, and earlier stages are unchanged.
  - The §3.1 assertion holds, all five points.
- **B5. Seed preload, against controls:**
  - it delivers a queued pair in order to `protocolMove`'s `freshSeed`;
  - it fails closed when the queue is exhausted;
  - it passes every other `randomInt` call through;
  - it refuses to load without its environment flag;
  - a grep guard shows no product file imports it.
- **B6. Suites: the exact model-free commands.** These run from the build worktree root, twice: first at the
  pre-swap head (commit (i) plus tooling), then at (ii). Counts are recorded each time, and any failure stops the
  build for review.

  ```sh
  # server (npm run test:server minus test_model_manifest.js and test_ort_conv_canary.js)
  node --test server/test_server.js server/test_readout_policy.js server/test_mcts.js \
    server/test_alpha_zero_client.js server/test_py_random.js server/test_mirror_inference.js \
    server/test_protocol_mcts.js server/test_protocol_move.js server/test_hard_model.js
  # parity corpus and comparator (JSON only)
  npm run test:parity
  # product match (npm run test:match minus test_harness.mjs and test_analyse.mjs)
  node --test tests/product_match/test_openings.mjs tests/product_match/test_timing.mjs
  # golden: dry-run only (tests/mcts_golden/worker.mjs:24, "a dry-run process structurally cannot load a model")
  npm run test:golden
  # integration tests on fakes; spent drivers are exercised only with their gates closed
  node --test tests/htb_integration/test_g3.mjs tests/htb_integration/test_mix_match.mjs \
    tests/htb_integration/test_onpolicy_match.mjs tests/htb_integration/test_teacher_bench.mjs \
    tests/htb_integration/test_diag_o.mjs tests/htb_integration/test_hmx_integration.mjs
  npm test
  npm run lint
  # the new wrapper (B7)
  .venv/bin/python -m pytest -q tests/test_hmx_integration_wrapper.py
  ```

  **Moved to the gated run (§5.3, G0), because they load models:**

  | suite | why it is not model-free |
  |---|---|
  | `server/test_model_manifest.js` | creates a session on a real graph (`:332–334`), and starts `server/index.js` and `scripts/startServer.js` (`:851–925`) |
  | `server/test_ort_conv_canary.js` | creates ORT sessions on synthetic graphs (`:128`) |
  | `server/test_ort_shutdown.js` | spawns a probe that loads a 4-element ONNX model (`:7`, `:20`). Not in `test:server` |
  | `tests/product_match/test_harness.mjs` | `loadModel` on two committed artifacts (`:91–92`) |
  | `tests/product_match/test_analyse.mjs` | runs a real-model match through `runMatchWithExplicitP` (`:227`) |

  `server/test_parity.js` (loads `server/test_model.onnx`) is in no npm script and is not run, as before.

  **Not run in either step:** the Python research suites (`tests/test_htb_*.py`). The build changes no Python file,
  and no Python test reads a file the build changes. A grep found no Python test reading `server/model_manifest.js`,
  the golden files, `test_g3.mjs` or `g3_binding.json`. The one hit, `tests/htb_onpolicy_v3_spec.py`, rewrites an
  import line in a copied driver and does not read the file. No test enumerates `models/`. B2's allowlist re-checks
  this at build time. The four known stale H-OP failures are therefore not exercised, and they stay a separate,
  unapproved item.
- **B7. Wrapper:** derived byte-for-byte by `tests/hmx_integration_wrapper_spec.py`, as the match wrappers are, and
  refused-path tests run on the real wrapper in a throwaway repo (the export-2 pattern).

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
- **Branch order (revised):** on `hmx-integration`, (i) → tooling → (ii) → build record and its ledger entry → gate →
  evidence → ledger. **`htb-integration-g1` does not move** until the deployment decision (§7, step 4). Nothing is
  committed on it meanwhile, so the fast-forward stays possible. Revision 0 put (ii) after the build record, but the
  build record has to report the B-checks run at (ii), so it now follows (ii).
  - If deployment is approved, `htb-integration-g1` is fast-forwarded to `hmx-integration`.
  - If it is declined, a `git revert` of (ii) is committed on top first, and then the fast-forward, so the build, the
    evidence and the ledger land linearly without the swap.
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

### 5.3 The staged server serves the pinned export (G0, S1, S2)

- **G0. Model-loading suites (moved here from B6):** in `<W>`, after the gate commit, before any server starts:

  ```sh
  node --test server/test_model_manifest.js server/test_ort_conv_canary.js server/test_ort_shutdown.js
  node --test tests/product_match/test_harness.mjs tests/product_match/test_analyse.mjs
  ```

  These run again in `<W>-revert` under R2. Each file must pass, with counts recorded. I read the suites for hard-id
  pins and found none; they use the committed default and comparison artifacts and synthetic graphs. So they should
  pass unchanged on both trees.
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
  - In `<W>-revert`, the B6 model-free commands and the G0 model-loading suites all pass. The golden current-HEAD
    test passes against `protocol-hard-pinned` again, because the revert restores `HARD_MODEL_ID`. `test_g3.mjs`
    passes with today's `[]` assertion, and `model_manifest.js` hashes to `ecde544b…` again.
  - The throwaway branch and tree are then removed, not merged.

### 5.6 Timing and the browser (G3-style; no new match)

- **T. Per-move time:** recorded from A1 for plies < 20 and ≥ 20, against the match's recorded `ms` for the same
  rows. This is informational: the same graph signature and budget should give the same cost, and the match's
  compute ratio was 1.00323.
- **V. Browser smoke: required (D4, accepted).** I drive one game through Claude in Chrome against the staged
  server, on localhost, with its **normal seed source**:
  - The server is `<W>`'s, started by `node server/index.js` with no `--import` preload, no `NODE_OPTIONS`, and
    `HARD_MODE`, `MODEL_MANIFEST` and `HARD_MODEL_MANIFEST` unset, so each `hard` request draws fresh seeds from
    `crypto.randomInt`. Its stdout and stderr go to `docs/superpowers/evidence/hmx-integration-check/v_server.log`.
    Its argv and environment keys (`ps -o args=`, values not recorded) go to the V record, as proof that no preload
    was present.
  - It runs after the wrapper has exited, with the wrapper's process group empty and the gate restored, outside the
    ceiling. It is the only server running.
  - I play as one colour against `hard`, in a new tab of my own, from the empty board to the end of the game. The
    game must reach ply 21 or later.
  - **PASS requires all of:**
    - the banner and `/api/model-info` show `hard_mode: 'protocol'`, `hard_model_id: '627c8d774df838a8'` and
      `model_id: 'c34b7ff3297c785a'`;
    - every AI move arrives, is legal and is drawn;
    - the MCTS progress display updates on both sides of ply 20;
    - no error appears in the client console (read with a pattern filter) or in `v_server.log`;
    - the game ends through the client's normal end-of-game display.
  - **Record:** `docs/superpowers/evidence/hmx-integration-check/v_record.json`, written from what I observe: the
    model-info JSON, the move list, the ply count, the result, console-error count and the server argv. I also
    record a GIF of the game, `hmx_browser_smoke.gif`.
  - A V failure is a check FAIL (§5.7), not a VOID.
- **Estimate:** reference and staged captures about 5 min each; G0 about 5 min per tree; S2 about 2 min; A0, A1 and
  A2 about 21 min each (845 moves at the match's 1.49 s mean); R1 and R2 about 26 min each, plus R2's suites about
  15 min. That is about 2.7 h in all, under the 4 h ceiling. V adds about 15 min after it.

### 5.7 Outcomes (fixed before the run)

| outcome | reading | what it permits |
|---|---|---|
| any check FAIL | the served product does not reproduce the tested player, or the rollback does not restore today's `hard` | nothing; a reviewed decision; no silent repair or retry |
| VOID (ceiling, crash, interruption) | no result | nothing; no automatic retry |
| **all of G0, S1–S4, A0–A2, R1, R2 and V PASS** (T informational) | the staged server serves the exact tested export through both transports, reproducing the match harness move for move; easy and medium are unchanged; both rollbacks restore today's `hard` exactly; the unseeded server plays a full browser game normally | recommending deployment (§7, step 4), with its own approval. **No deployment is recommended without V's PASS** |

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

## 7. Approvals, in order

1. **This proposal.** Revision 0 was reviewed on 2026-11-23: D1–D4 decided (§9). Revision 1 is returned for the
   build approval.
2. **Build:** commits (i) and (ii), the tooling of §3, the build record and its ledger entry, on branch
   `hmx-integration` in the build worktree, in the order of §4, with B1–B7 (model-free only). **`htb-integration-g1`
   does not move.** All gates stay closed. Returned for launch review.
3. **The check run** of §5 (one wrapper run, 4 h ceiling), then the required browser smoke V. Evidence and ledger
   entry follow, on `hmx-integration`. Returned for review, with a deployment recommendation only if every check,
   V included, passed.
4. **Deployment:**
   - fast-forward `htb-integration-g1` to `hmx-integration` (the build, commit (ii) and the check evidence);
   - restart your server with no override;
   - confirm the banner shows `627c8d774df838a8`.

   If deployment is declined, commit a revert of (ii) on `hmx-integration` first, then fast-forward. Pushing remains
   a separate decision.

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

## 9. Decisions (decided by the user, 2026-11-23)

| | decision | in revision 1 |
|---|---|---|
| D1 | **(a) accepted:** byte-identical manifest | §3 commit (i); B1 pins `abf3931d…` as a packaging pin and checks that it names the qualified graph and data; the `provenance_note` slip is corrected in the ledger (proposal note, revision 1), not in the file |
| D2 | **accepted:** no new strength match | §1 unchanged |
| D3 | **(a) accepted, made precise:** only `server/model_manifest.js` changes, from `ecde544b…` to `054f9434…`; aggregate `da0b86bf…` → `aa04f04c…`; the historical binding and evidence stay intact | §3.1, B4 |
| D4 | **I drive V** against the staged server with its normal seed source; V is required before recommending deployment | §5.6, §5.7, §7 |

**The `provenance_note` correction (D1):** the export's manifest says "Exported once, to scratch, by `htb_mix_export
run export`". The string is fixed text in `scripts/GPU/alphazero/htb_mix_export.py:57`, written the same way by both
subcommands. The manifest at `abf3931d…` was written by export attempt 2, `htb_mix_export run export-2`
(`docs/superpowers/evidence/htb-mix-export-2-run/01_wrapper.sh`, entry 76), under export amendment 1. The file stays
byte-identical, because the hash is the pin. The correction is recorded in the ledger as dated text.

The options as revision 0 put them, kept for the record:

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
| build | 3 copied files; 1 product line + comment; ~80 test lines (`test_hard_model.js`, B1 with controls); golden stage ~20 lines + 2 test files (the `06b52aaa` pattern); G3 test ~25 lines (§3.1); preload ~50 lines; driver ~400 lines; wrapper spec + tests ~300 lines; B6 run twice, about 15 min each |
| check run | ~2.7 h CPU under a 4 h ceiling; browser smoke ~15 min after it |
| deployment | a fast-forward and a restart |

## 11. Revision log

**Revision 1 (2026-11-23), after the user's review of revision 0:**

- §2: each hash cites the record field that holds it. The sizes and the manifest hash are in neither record. The
  manifest hash `abf3931d…` is a packaging pin, not tested evidence.
- §3: the exact `model_manifest.js` diff, with its resulting hash `054f9434…`. §3.1 gives the precise G3 binding
  assertion, D3. The wrapper spec and test files are named.
- §4: model-free is defined. B1 checks that the manifest names the qualified pair against both records. B2 adds the
  file hash and the unchanged model directories. B4 adds §3.1. B6 lists exact model-free commands, moves five
  model-loading suites to G0, and leaves the Python research suites out of both steps. The build worktree with
  cloned `node_modules` is named.
- §5: the branch order is fixed: (ii) before the build record, and `htb-integration-g1` does not move until
  deployment. G0 is added. R2 runs every suite. V is required, run by me, with the normal seed source and a written
  record. The estimate is 2.7 h.
- §7 and §9: approvals and the decided D1–D4.
- The prediction is unchanged: all checks PASS, about 90%. G0's suites carry no hard-id pins, and V exercises the
  path the match and S2 already exercise.
