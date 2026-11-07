# H-TB integration proposal — the validated student in the served product (2026-11-06)

**Status:** PROPOSAL, for review. Nothing here is implemented, converted, loaded, benchmarked or played.

**Authorized (the user, 2026-11-06):** read-only source inspection, and writing this proposal and its ledger note.
Every step in §6 needs its own approval.

**Not authorized:** implementation, ONNX export or conversion, loading any model, training, benchmarks, games,
deployment, pushing.

## 1. Summary and recommendation

The product already serves the **reference** of the confirmation test. `DEFAULT_MODEL_ID = 'c34b7ff3297c785a'`
(`server/model_manifest.js:49`) is the ONNX export of `calib020_0001` (SHA-1 `209cf2d4…`;
`models/c34b7ff3297c785a/manifest.json`). The student has calib020_0001's architecture: 128 hidden, 6 blocks,
30 input channels, 576 moves, and a byte-identical file size (7,524,333). **Swapping the network is therefore
an export of the same kind as August's, with no format change.**

**The search is the real gap.** The product's `hard` level does not run the tested protocol:

| | tested protocol (entry 54) | product `hard` today |
|---|---|---|
| network | the whole student, frozen | calib020_0001 |
| evaluator | **mirror averaging**: two evaluations per leaf, averaged | one evaluation per leaf |
| simulations | **200** | 800 |
| plies 0–19 | plain PUCT search; **sampled at temperature 1.0** from visit counts | plain PUCT; argmax (deterministic) |
| ply ≥ 20 | **root sequential halving** (16 candidates); its choice is played | plain PUCT; argmax |
| game cap | 280 plies (a draw) | 600 plies |

**Recommendation, minimal:**
- Export the student once with the unmodified exporter, into a new content-addressed `models/<id>/`.
- Add two new server modules, a mirror-averaging inference wrapper and a root-halving search. Both are
  line-by-line transcriptions of the reviewed Python, used only by a new `protocol` search mode.
- Route **only `hard`** to that mode with the student (decision D1).
- Leave `easy`, `medium` and the analysis endpoints on the current model and search.
- Keep `c34b7ff3297c785a` byte-identical for rollback.

## 2. What was validated, exactly

Source of truth: `scripts/GPU/alphazero/htb_strength.py` (the run), `mirror_halving_combination.py`,
`mirror_averaging.py`, `root_sequential_halving.py`, `mcts.py` and `eval_readout.py`. The protocol is
`external_learning_stage3.PROTOCOL` with 200 simulations per side.

- **Network:** `~/twixt-cache/htb-pilot/student.safetensors`, sha256 `0e1cb621…`, state digest `586dca4f…`,
  in eval mode (BatchNorm uses running statistics).
- **Evaluator** (`MirrorAveragingEvaluator`):
  - Each leaf is evaluated twice, on the position and on its left-right mirror (`self_play._mirror_position_lr`).
  - The mirror flips columns, permutes the 8 link-direction channels of each colour by (dr, dc) → (dr, −dc),
    and swaps channels 21↔22 and 27↔28. Legal moves map to (r, 23 − c) in the same order.
  - Priors are the mean of the two softmaxed distributions, with no renormalization. The value is the mean of
    the two side-to-move values.
- **Search, plies 0–19:** `MCTS.search_with_root`. This is **serial**: one synchronous simulation at a time
  (`mcts.py:540`, "NOT search_from_root's batched waiter path"); the batch and virtual-visit settings are unused.
  - PUCT with c = 1.5, the √(N+1) convention, and FPU 0. No root noise, no prior penalties.
  - Ties within 1e-8 are broken by a per-game seeded random choice.
- **Readout, plies 0–19:** `eval_readout.select`, MODE_OPENING_TEMPERATURE, temperature 1.0. It samples ∝
  count^(1/T) with the 1e-8 floor and max-subtraction, the same formula as the product's `selectMove`.
- **Search and move, ply ≥ 20** (`RootSequentialHalvingMCTS`, constants checked by `check_variant`): FROM_PLY 20,
  CANDIDATES 16, C_VISIT 50, C_SCALE 1.0, PRIOR_FLOOR 1e-12, 24×24 only, no noise.
  - Candidates: the 16 highest-prior legal moves, ties broken by (row, col).
  - Rounds: R = ⌈log₂ m⌉. Each round but the last gives every survivor N // (R × survivors) visits, round-robin
    in rank order, then keeps the top ⌈survivors / 2⌉. The last round splits the remainder evenly, with any
    extra visits going one each in rank order.
  - Score: score = log(max(prior, 1e-12)) + (50 + max visits) × (q + 1) / 2, with q the candidate's mean from
    the root player's side. Ties go to the higher prior, then (row, col).
  - The **top-ranked final survivor is played**, bypassing the readout.
  - At N = 200 and m = 16 the rounds give 3 / 6 / 12 / 28 visits per survivor (cumulative 3 / 9 / 21 / 49).
  - Descent below the root is the ordinary PUCT, through the root-override seam
    (`_run_single_simulation(root, root_move_override=m)`).
- **Game:** the empty 24×24 board, red first, with no swap rule. The cap is 280 plies, and the search treats
  ply 280 as terminal (a draw). The input-feature move number is ply / 600 in both stacks.
- **Opponent and result:** both colours, against the original proven combination (calib020_0001 under the
  same evaluator and search). 217 wins, 2 losses and 1 draw over 220 games, within the declared compute
  allowances.

## 3. The product today (read-only inspection)

| concern | where | behaviour |
|---|---|---|
| model selection | `server/model_manifest.js` | `DEFAULT_MODEL_ID`, or the `MODEL_MANIFEST` override; validates both files by size and SHA-256 and binds the ONNX contract; no fallback |
| artifacts | `models/<16-hex id>/{model.onnx, model.onnx.data, manifest.json}` | tracked in git as a deliberate `.gitignore` exception (`!models/*/*.onnx*`); id = sha256(graph_sha + ':' + data_sha)[:16] |
| inference | `server/inference.js` | onnxruntime-node, CPU float32, batch 1; HWC→NCHW conversion; softmax over legal moves; value clamped to [−1, 1] |
| encoding | `server/gameLogic.js` | 30-channel `toTensorHWC`; channel constants at lines 54–70; MAX_PLIES 600; Phase 2 S1 proved exact equality with Python |
| search | `server/mcts.js` | serial PUCT, c 1.5, √(N+1), unvisited q 0, **lexicographic exact tie-break**, lazy children, abort and progress hooks |
| readout | `server/readout_policy.js` | easy 100 sims / T 1.0; **medium 400 / T 0.5 (DEFAULT_DIFFICULTY)**; hard 800 / T 0 (argmax); client `temperature` and `deterministicMode` overrides |
| transports | `server/index.js` | REST `/api/move` (search output cached under `${modelPath}\|${nSims}`), WS `computeBestMove` (uncached, cancellable, progress), `/api/evaluate` (raw network value for the win bar), `/api/analyze-position` (plain MCTS, 10–1000 sims, used by the game recorder), a model-info endpoint (hard-codes 128×6) |
| client | `assets/js/ai/alphaZeroClient.js`, `assets/js/game/twixtGame.js` | WS move with difficulty; board size always 24; no swap rule |
| test bindings | `tests/mcts_golden/cases.mjs` `STAGES`; `tests/product_match/p_decision.mjs` `EXECUTION_SURFACE_FILES` (10 files, including `server/{gameLogic,inference,mcts,model_manifest,readout_policy}.js`) | any edit to those files moves the surface digest; at the August switch, 17 tests failed from one pin until a new stage was added |

**Already the same as the protocol:** PUCT form and c_puct, FPU 0, no noise, terminal values, legal-move
order, input encoding (S1 exact), the temperature-sampling formula, and serial simulation.

**Different:** the evaluator (no mirror), the root rule (no halving), the simulation counts, the hard readout
(argmax rather than opening temperature then halving), the tie-break (lexicographic exact vs seeded random
within 1e-8), the RNG streams (`Math.random` vs Python `random`), the 600- vs 280-ply cap, and the runtime
(ORT CPU vs MLX GPU, bounded by Phase 2's tolerances).

## 4. Proposed changes

The validated artifact stays unchanged: the export reads it, and its sha256 is checked before and after.

### 4.1 Packaging: one ONNX export (no code change)

- **Export once**, to scratch, with `scripts/GPU/alphazero/export_onnx.py` **unmodified**, in exactly
  August's invocation (`models/c34b7ff3297c785a/manifest.json` `provenance.exporter_config`): `--weights
  ~/twixt-cache/htb-pilot/student.safetensors`; hidden 128, blocks 6, in_channels 30, max_moves 576,
  opset 18, BatchNorm folded.
- **Check the student's sha256** is `0e1cb621…` immediately before and after the export, and record its SHA-1:
  the manifest schema requires `provenance.source_checkpoint_sha1`.
- **Place the files** at `models/<id>/`, with the id computed by `computeModelId`. The new `manifest.json` has
  the same shape as c34b7ff's: identity and provenance only, no strength or serving claim. It names the source
  path, SHA-1 and sha256, and points to ledger entry 54 for the evidence.
- If the export is defective, that is the finding. There is no alternate export (Phase 2 §2's rule).

### 4.2 New: `server/mirror_inference.js` (~70 lines)

- `class MirrorAveragingInference` wraps an `AlphaZeroInference` and keeps the same
  `evaluate(boardTensorHWC, moves)` interface. It runs one evaluation on the board and one on its mirror, then
  returns index-aligned averaged priors and the mean value.
- The mirror is a literal transcription of `_mirror_position_lr` for the full 24×24 board:
  - flip columns;
  - permute each colour's 8 link-direction channels with the perm built from gameLogic's direction map by
    (dr, dc) → (dr, −dc);
  - swap channels 21↔22 and 27↔28;
  - map moves to (r, 23 − c) in the same order;
  - average by index j, not by key.
- Two ONNX calls per leaf. The model is exported at batch 1 (`dynamic_axes=None`), so there is no batch-2 call.

### 4.3 New: `server/root_halving.js` (~130 lines)

- `class RootSequentialHalvingMCTS extends MCTS` is a transcription of `root_sequential_halving.py`, with the
  same constants and a frozen-constants check.
- `search(state, opts)`:
  - below ply 20, delegates to `super.search`;
  - otherwise refuses a terminal root or a board other than 24×24, expands the root, and runs the rounds of §2;
  - returns `{visitCounts, rootValue, choice}`.
- Root-override simulations use a new `_simulate(root, overrideKey)` built from `MCTS`'s existing
  `_childFor`, `_selectChild`, `_expand` and `_backup`. `server/mcts.js` itself is **not edited**.
- Abort: if cancelled mid-rounds, it returns no choice. The caller treats that as an aborted search, never as a
  partial halving result.

### 4.4 Edited: `server/readout_policy.js` (~25 lines)

- `DIFFICULTY_TABLE.hard` becomes `{ nSims: 200, search: 'protocol', moveTemp: 1.0 }`. `moveTemp` applies only
  at plies 0–19; from ply 20 the halving choice is played.
- `resolvePolicy` returns `search`. `easy` and `medium` are unchanged.
- Client `temperature` and `deterministicMode` are **ignored** for `protocol` (decision D3), so a request
  cannot leave the tested protocol.

### 4.5 Edited: `server/index.js` (~60 lines)

- **Model wiring, option B (recommended, decision D1):** resolve a second pinned manifest, `HARD_MODEL_ID`, and
  load a second ORT session. `hard` uses `MirrorAveragingInference(student)` with `RootSequentialHalvingMCTS`;
  every other path uses the current session and `MCTS`.
- Both transports (`/api/move` and WS `computeBestMove`) select the search class from `policy.search`. When the
  search supplies a `choice`, it replaces `selectMoveForRequest`.
- **Cache scope** becomes `${modelPath}|${nSims}|${policy.search}`. The cache stores the halving `choice` with
  the visits, so a cached hard move replays the halving decision, not an argmax.
- **Non-24 boards** (unreachable from the current client) get an explicit error under `protocol`, never a
  silent plain search.
- The model-info endpoint reports both model ids.

### 4.6 Edited: `server/model_manifest.js` (~15 lines, option B only)

- Add `HARD_MODEL_ID` and its manifest path. Every validation is the existing code.
- `DEFAULT_MODEL_ID` is unchanged in option B. In option A it would be the one-line switch, and §4.5's second
  session would not exist.

### 4.7 Test bindings

- Add the two new modules and `server/index.js` to `EXECUTION_SURFACE_FILES`.
- Add a new frozen `STAGES` entry for the integrated surface. `eager`, `lazy` and `candidate-default` keep their
  digests and evidence (the August pattern).
- Re-point only the current-HEAD capture and attribution tests, recording the attribution change.

## 5. What this does not change

- **The student file** (`0e1cb621…`).
- **The current model** `c34b7ff3297c785a` and the earlier `1d64027db521a50f`, both byte-identical.
- **Existing search code:** `server/mcts.js`, `server/gameLogic.js` and `server/inference.js` stay unedited;
  `easy` and `medium` keep their search, readout and model.
- **The analysis and display endpoints** (`/api/evaluate`, `/api/analyze-position`, the eval bar): they keep
  the current model and plain search (decision D5).
- **The research code and the client.**

## 6. Validation steps (each needs its own approval; none has run)

**No model is loaded in V1, V4, V5 and V6.** V2, V3 and V7 load models, and V8 plays games.

- **V1. Mirror transform, exact (Python, no model):**
  - Fixture: run `_mirror_position_lr` over the Phase 2 parity corpus states (`tests/parity/corpus.json`).
  - JS's mirrored NCHW buffer and mirrored move list must equal Python's **exactly** (0.0), on every element
    and in the same order.
  - Mirroring twice must give the identity.
- **V2. Export identity and model-output parity:**
  - The export of §4.1, with the student's sha256 unchanged before and after.
  - The manifest is validated by `resolveModel` through `MODEL_MANIFEST`.
  - Phase 2's surfaces S1–S4 for the student on its corpus, with its preregistered tolerances and ordering
    rules: S2 ≤ 1e-4, S3 ≤ 1e-5, S4 ≤ 1.1e-4 for max abs logit and value; top-1 agreement outside near-ties;
    at most 6 exemptions per surface pair; top-5 sets ≥ 95%; median Kendall τ ≥ 0.99.
  - S1 confirms the input tensor and legal-move list stay exact.
- **V3. Averaged-output parity:** MLX `MirrorAveragingEvaluator(student)` against Node
  `MirrorAveragingInference` on the same corpus, with the S4 bounds and V2's ordering rules applied to the
  averaged priors and value.
- **V4. Halving transcription, exact (no model):**
  - Fixtures come from the Python `RootSequentialHalvingMCTS` driven by a **deterministic fake evaluator**
    designed to have no exact or 1e-8 ties.
  - Positions cover m = 16, m < 16, m = 2 and m = 1; a winning child inside the candidates; and a terminal
    child two plies down.
  - The JS port must match **exactly** on each round's survivors, the visit allotments, the final visit counts
    and the choice. The root value must match to 1e-12.
  - At N = 200, m = 16 the allotment must be 3 / 6 / 12 / 28.
  - Separate tests pin the known tie divergence (research: seeded random within 1e-8; product: lexicographic
    exact) as documented behaviour.
- **V5. Readout and routing (no model):**
  - With the fake evaluator and an injected `selectMove`: at ply < 20, `hard` runs a plain 200-simulation
    search and samples at T 1.0. At ply ≥ 20 it plays the halving choice and the readout is bypassed.
  - REST and WS give the same move for the same request.
  - Overrides are ignored under `protocol`.
  - The cache scope separates `protocol` from `plain`, and a cached hard move replays the choice.
  - An aborted halving returns no move.
  - A non-24 board under `protocol` is refused.
  - `easy` and `medium` behave byte-identically to today.
- **V6. Legal moves (no model):** for every search mode and over a seeded set of positions, the played move is
  in `legalMoves()`; halving candidates are a subset of legal moves; and m < 16 positions work.
- **V7. Unchanged paths and suites:**
  - The golden corpus comparison (92 cases) under the new stage must show the plain path's visit counts, root
    value, selected move and progress exactly equal. This loads the current model.
  - The server, parity, match and golden suites, plus lint and typecheck.
  - The full `tests/test_htb_*` research suites, unchanged.
- **V8. End-to-end (separately approved; plays moves):**
  - (a) A smoke test: `npm start` with the student staged through `MODEL_MANIFEST`, a handful of `hard` moves
    through both transports, the banner and the model ids.
  - (b) **Recommended before any deployment:** a preregistered product-stack match in the August Phase 3
    pattern: new `hard` (student, protocol) against current `hard` (c34b, 800-simulation argmax).
    - The confirmation result transfers to the product only by argument. Game-for-game reproduction is
      impossible because the RNG streams and tie-breaks differ, and the opponent differs.
    - This match is the only evidence about what the user-facing change does. Its size, rule and ceiling would
      be proposed separately after the smoke run's timing.

## 7. Rollback

- **Nothing is deleted:** `models/c34b7ff3297c785a/` and `models/1d64027db521a50f/` stay tracked and
  byte-identical. `server/test_model_manifest.js` already re-hashes the default pair (c34b7ff) but only reads
  1d64027's manifest; the rollback test below adds the same pair-hash check for both retained pins.
- **Option B:** set `DIFFICULTY_TABLE.hard` back to `{ nSims: 800, moveTemp: 0 }` and drop `HARD_MODEL_ID`. This
  is two tracked one-line reverts and restores today's `hard` exactly; easy and medium were never changed.
- **Option A:** revert `DEFAULT_MODEL_ID` to `'c34b7ff3297c785a'` (the August one-liner), plus the readout line.
- **Staging without a code change:** `MODEL_MANIFEST=<path>` is the supported override (`model_manifest.js:286`).
- A rollback test asserts that the reverted table and pins reproduce today's resolved policy and model id.

## 8. Product behaviour outside the tested protocol

1. **`easy` and `medium`**: 100 and 400 plain simulations, at temperatures 1.0 and 0.5. Untested with the
   student. `medium` is the default and was never measured for c34b either (August §5.2). Option A would
   change the default experience with no evidence; option B leaves it untouched.
2. **The new `hard` opening is stochastic** (temperature 1.0 for plies 0–19), as tested. Today's `hard` is
   deterministic. The UI's promise of the strongest deterministic play changes (decision D2).
3. **Client overrides** (`temperature`, `deterministicMode`) were never part of the protocol (decision D3).
4. **Opponents:** the PASS is against the original proven combination only. There is no evidence against
   humans, twixtbot, or c34b under today's 800-simulation `hard`.
5. **Analysis and display:** the raw-network win bar, `/api/analyze-position`, the game recorder's grading and
   the WS progress value. Under halving, the root value is an average over forced candidate visits, so it is
   not comparable to today's eval-bar value if exposed (option B keeps these on the current model and plain
   search).
6. **Games past 280 plies:** the protocol treated ply 280 as a terminal draw (1 game in 220 reached it). The
   product continues to 600, so search near and after 280 is untested.
7. **Ties and randomness:** the product is deterministic-lexicographic where research is seeded-random within
   1e-8. Mirror averaging makes exact prior ties between mirror-image moves likely in symmetric positions
   (a + b = b + a exactly). The match is equal in distribution, not game for game.
8. **Runtime:** ORT CPU float32 against MLX on the GPU with TF32 off. The difference is bounded by V2/V3's
   tolerances, not zero.
9. **Cancellation, caching, and non-24 boards:** none of these occurred in the protocol. §4.3 and §4.5 handle
   them by refusing rather than improvising.
10. **Latency:** about 400 network calls per move plus the root, against 800 for today's `hard`. This is
    unmeasured, and no benchmark is authorized.

## 9. Decisions for the user

- **D1:** apply the integration to `hard` only, with the student and the protocol, keeping easy and medium on
  c34b (**option B, recommended**). Or switch every level to the student (**option A**: smaller code, untested
  default).
- **D2:** reproduce the tested opening sampling (T 1.0, plies 0–19) for `hard` (**recommended**). Or use argmax
  (outside the protocol).
- **D3:** ignore client temperature and determinism overrides under `protocol` (**recommended**). Or honour them
  before ply 20.
- **D4:** require the V8(b) product-stack match before any deployment (**recommended**).
- **D5:** keep the analysis endpoints on the current model and plain search (**recommended**). Or move them to
  the student.

## 10. File changes and estimated effort

| file | change | size |
|---|---|---|
| `models/<new id>/{model.onnx, model.onnx.data, manifest.json}` | new, from the one export | ~7.6 MB, as the existing models |
| `server/mirror_inference.js` | new | ~70 lines |
| `server/root_halving.js` | new | ~130 lines |
| `server/readout_policy.js` | `hard` entry, `search` field, override rule | ~25 lines |
| `server/index.js` | second session, search routing, cache scope, refusals, model info | ~60 lines |
| `server/model_manifest.js` | `HARD_MODEL_ID` (option B) | ~15 lines |
| `server/test_mirror_inference.js`, `server/test_root_halving.js`, additions to `test_readout_policy.js` / `test_server.js` / `test_model_manifest.js` | V1, V4–V6 | ~600 lines |
| `tests/fixtures/htb_integration/*` and a Python fixture generator | V1, V4 fixtures (no model) | small |
| `tests/mcts_golden/cases.mjs`, `tests/product_match/p_decision.mjs` | new stage and surface list | ~40 lines |
| ledger and an integration note | records | — |

**Estimated effort, engineering only:** about **5–7 working days**.

| work | days |
|---|---|
| export, manifest and V2 | 1 |
| mirror wrapper and V1/V3 | 1 |
| halving port and V4 | 1.5–2 |
| routing, cache and refusals, with V5/V6 | 1 |
| option B second session | 0.5–1 |
| surface stage, V7, docs | 1 |

V8(b)'s match compute is extra; it is priced after V8(a) and needs its own approval.

**Suggested order:** V1/V4 fixtures and ports first, since they need no model; then the export and V2/V3;
then the wiring and V5–V7; then V8 on separate approval. Work on a branch from `htb-pilot-build`; the product
code there is identical to `main`'s.
