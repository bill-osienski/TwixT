# H-TB integration proposal — the validated student in the served product (2026-11-06, revision 1 2026-11-09)

**Status:** PROPOSAL, revision 1, for review. Nothing here is implemented, converted, loaded, benchmarked or played.

**History:**
- Revision 0 (`80acd87e`, 2026-11-06) was reviewed by the user on 2026-11-09. The review recommends proceeding with
  the hard-only integration, subject to five corrections, and decides D1–D5 (§11).
- This revision makes those corrections. §13 maps each review point to the change that answers it.

**Authorized (the user, 2026-11-06):** read-only source inspection, and writing this proposal and its ledger note.
The 2026-11-09 review asks for its points to be resolved before implementation; this revision and its ledger note
are the only changes.

**Not authorized:** implementation, ONNX export or conversion, loading any model, training, benchmarks, games,
deployment, pushing. The review groups the remaining work into three approvals (§7): G1 implementation with
fake-evaluator tests; G2 export and numerical checks; G3 product smoke test and match. Deployment remains a final,
separate decision.

## 1. Summary and recommendation

The product already serves the **reference** of the confirmation test. `DEFAULT_MODEL_ID = 'c34b7ff3297c785a'`
(`server/model_manifest.js:49`) is the ONNX export of `calib020_0001` (SHA-1 `209cf2d4…`;
`models/c34b7ff3297c785a/manifest.json`). The student has calib020_0001's architecture: 128 hidden, 6 blocks,
30 input channels, 576 moves, and a byte-identical file size (7,524,333). **Swapping the network is therefore
an export of the same kind as August's, with no format change.**

**The search is the real gap.** The product's `hard` level does not run the tested protocol:

| | tested protocol (entry 54) | product `hard` today | proposed `hard` |
|---|---|---|---|
| network | the whole student, frozen | calib020_0001 | the student |
| evaluator | **mirror averaging** | one evaluation per leaf | mirror averaging |
| simulations | **200** | 800 | 200 |
| plies 0–19 | PUCT; **sampled at T 1.0** | PUCT; argmax | PUCT; sampled at T 1.0 |
| ply ≥ 20 | **root sequential halving**; its choice is played | PUCT; argmax | root sequential halving |
| PUCT ties | **random among scores within 1e-8**, shuffled order | lexicographic, exact equality | the research rule, ported (§4.3) |
| game end | win, no legal move, or **ply 280 (draw)** | win or no legal move (§9 item 6) | the product rule, unchanged (a stated product variation) |

**Recommendation, minimal:**
- Export the student once with the unmodified exporter, into a new content-addressed `models/<id>/`.
- Add four new server modules: a mirror-averaging inference wrapper, a Python-compatible RNG, the research PUCT
  tie rule, and root halving. All are line-by-line transcriptions of the reviewed Python, used only by a new
  `protocol` search mode.
- Route **only `hard`** to that mode, with the student loaded through its **own** model-selection path (§4.7).
- Leave `easy`, `medium` and the analysis endpoints on the current model and search.
- Keep `c34b7ff3297c785a` byte-identical, and keep today's `hard` as a named, tested mode that one startup setting
  restores (§8).

## 2. What was validated, exactly

Source of truth: `scripts/GPU/alphazero/htb_strength.py` (the run), `mirror_halving_combination.py`,
`mirror_averaging.py`, `root_sequential_halving.py`, `mcts.py`, `eval_readout.py` and `eval_runner.py`. The
protocol is `external_learning_stage3.PROTOCOL` with 200 simulations per side.

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
  - Visit counts are returned over the legal moves in row-major order (`mcts.py:564–569`).
- **The PUCT tie rule** (`mcts.py:1015–1022` and `1117–1156`), used at every depth:
  - At each expansion, the (move, prior) pairs are **shuffled with the search RNG** (`self.rng.shuffle(pairs)`),
    and that shuffled order is the node's iteration order.
  - `_select_child` scores each move as q + c × prior × √(N+1) / (1 + n). It keeps a **running** best: a score
    more than 1e-8 above the current best starts a new set; a score within 1e-8 of the current best joins it.
  - It then plays `self.rng.choice` over that set. The set depends on the iteration order, and `choice`
    consumes the stream even when the set has one member.
- **Readout, plies 0–19:** `eval_readout.select`, MODE_OPENING_TEMPERATURE, temperature 1.0, on a **separate
  readout stream**. It samples ∝ count^(1/T) with the 1e-8 floor and max-subtraction, iterating in the visit
  counts' (row-major) order.
- **Streams:** per game, `random.Random(seed ^ 0xA5A5A5)` / `^ 0x5A5A5A` for the red / black search and
  `^ 0xC3C3C3` / `^ 0x3C3C3C` for their readouts (`mirror_halving_combination.py:92–95`). Each persists across that
  side's moves.
- **Search and move, ply ≥ 20** (`RootSequentialHalvingMCTS`, constants checked by `check_variant`): FROM_PLY 20,
  CANDIDATES 16, C_VISIT 50, C_SCALE 1.0, PRIOR_FLOOR 1e-12, 24×24 only, no noise.
  - Candidates: the 16 highest-prior legal moves, ties broken by (row, col). **Deterministic.**
  - Rounds: R = ⌈log₂ m⌉. Each round but the last gives every survivor N // (R × survivors) visits, round-robin
    in rank order, then keeps the top ⌈survivors / 2⌉. The last round splits the remainder evenly, with any
    extra visits going one each in rank order.
  - Score: score = log(max(prior, 1e-12)) + (50 + max visits) × (q + 1) / 2, with q the candidate's mean from
    the root player's side. Ties go to the higher prior, then (row, col). **Deterministic.**
  - The **top-ranked final survivor is played**, bypassing the readout.
  - At N = 200 and m = 16 the rounds give 3 / 6 / 12 / 28 visits per survivor (cumulative 3 / 9 / 21 / 49).
    The 14 eliminated candidates take 102 of the 200 visits (8 × 3 + 4 × 9 + 2 × 21).
  - Descent below the root is the ordinary PUCT, **with the tie rule above**, through the root-override seam
    (`_run_single_simulation(root, root_move_override=m)`).
- **Game:** the empty 24×24 board, red first, with no swap rule. The game loop stops at ply 280 and the state's
  `max_plies_limit` is 280, so the search treats ply 280 as terminal (a draw) (`eval_runner.py:222–226`). The
  input-feature move number is ply / 600 in both stacks.
- **Opponent and result:** both colours, against the original proven combination (calib020_0001 under the
  same evaluator and search). 217 wins, 2 losses and 1 draw over 220 games, within the declared compute
  allowances.

## 3. The product today (read-only inspection)

| concern | where | behaviour |
|---|---|---|
| model selection | `server/model_manifest.js` | `DEFAULT_MODEL_ID`, or the `MODEL_MANIFEST` override (`:285–289`), which is the **only** model path, so it reaches every level and endpoint; validates both files by size and SHA-256 and binds the ONNX contract; no fallback |
| artifacts | `models/<16-hex id>/{model.onnx, model.onnx.data, manifest.json}` | tracked in git as a deliberate `.gitignore` exception (`!models/*/*.onnx*`); id = sha256(graph_sha + ':' + data_sha)[:16] |
| inference | `server/inference.js` | onnxruntime-node, CPU float32, batch 1; HWC→NCHW conversion; softmax over legal moves; value clamped to [−1, 1] |
| encoding and game end | `server/gameLogic.js` | 30-channel `toTensorHWC`; MAX_PLIES 600; terminal on a win, ply ≥ 600, or no legal move (`:604–620`); Phase 2 S1 proved the encoding exactly equal to Python's |
| search | `server/mcts.js` | serial PUCT, c 1.5, √(N+1), unvisited q 0, iteration in legal-move order, **lexicographic exact-equality tie-break** (`:254–265`), lazy children, abort and progress hooks |
| readout | `server/readout_policy.js` | easy 100 sims / T 1.0; **medium 400 / T 0.5 (DEFAULT_DIFFICULTY)**; hard 800 / T 0 (argmax); client `temperature` and `deterministicMode` overrides; `selectMove` samples with `Math.random` (`mcts.js:333`) |
| transports | `server/index.js` | REST `/api/move` (search output cached under `${modelPath}\|${nSims}`, `:83–97`; not called by the shipping client); WS `computeBestMove` (uncached, cancellable, progress; the client's move path); `/api/evaluate` (raw network value); `/api/analyze-position` (plain MCTS, 10–1000 sims, used by the game recorder); a model-info endpoint (hard-codes 128×6) |
| client move path | `assets/js/ai/alphaZeroClient.js` | WS `move` with `difficulty`; sends `temperature` / `deterministicMode` only if a caller sets them (`:241–244`); its one caller passes only `includeVisits` (`gameController.js:406`); `onProgress` is never assigned (`:61`), so progress messages are not displayed |
| client displays | `assets/js/game/gameController.js` | **NN bar** (`win-bar`): `/api/evaluate`. **MCTS bar** (`win-bar-mcts`): after an AI move, the move response's `valueRed` (`:466–468`, described at `:452–453` as "the search's evaluation of the position the AI chose to play into"); after a human move, only while recording, the recorder's precomputed `/api/analyze-position` root value (`:347–357`) |
| recorder | `assets/js/ui/gameRecorder.js:295–325` | records the AI move's visits, top-5 visit shares and root value |
| test bindings | `tests/mcts_golden/cases.mjs` `STAGES`; `tests/product_match/p_decision.mjs` `EXECUTION_SURFACE_FILES` (10 files, including `server/{gameLogic,inference,mcts,model_manifest,readout_policy}.js`) | any edit to those files moves the surface digest; at the August switch, 17 tests failed from one pin until a new stage was added |

**Already the same as the protocol:** PUCT form and c_puct, FPU 0, no noise, terminal values, legal-move
order, input encoding (S1 exact), the temperature-sampling formula, serial simulation, and the halving's own
deterministic rank and candidate tie-breaks (once ported).

**Different today:** the evaluator (no mirror), the root rule (no halving), the simulation counts, the hard
readout, the PUCT tie rule, the RNG streams, the game-end rule (§9 item 6), and the runtime (ORT CPU against MLX
GPU, bounded by Phase 2's tolerances).

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
  - average by index j, not by key, with the research's operation order.
- Two ONNX calls per leaf. The model is exported at batch 1 (`dynamic_axes=None`), so there is no batch-2 call.

### 4.3 New: `server/py_random.js` (~90 lines) and `server/protocol_mcts.js` (~80 lines) — the research tie rule

**Why both:** the review's point. The research picks uniformly at random among moves within 1e-8 of the running
best, in an order shuffled at each expansion. The product picks the lexicographically smallest of exactly equal
scores. These are different distributions, and mirror averaging makes exact ties likely in symmetric positions
(§9 item 7). The new hard mode therefore uses the research rule; `easy` and `medium` keep the product rule.

- **`server/py_random.js`:** a transcription of the subset of CPython's `random.Random` the protocol uses:
  - seeding from a non-negative integer (`init_by_array` over its 32-bit words);
  - the MT19937 generator, `getrandbits(k)`, `_randbelow` by getrandbits rejection, `random()` (53 bits from two
    draws), `choice` and `shuffle` (Fisher–Yates from the end);
  - pinned to the `.venv`'s CPython (3.14.10) and checked stream-for-stream against it in V4.
  - **Why a port rather than any RNG:** it makes the transcription testable **seed for seed, including the ties
    themselves**, against the Python search (V4). In production the seed is fresh per request (below), so the
    choice of generator does not change the distribution.
- **`server/protocol_mcts.js`, `class ProtocolMCTS extends MCTS`:**
  - `_expand` calls the parent, then sets `node.order` to a copy of the expanded moves shuffled with the search
    stream, as `_expand_batch` does. `node.moves` keeps legal-move order, so the visit-count output order is
    unchanged.
  - `_selectChild` iterates `node.order`, computes the score with the research's operation order, keeps the
    running 1e-8 set exactly as `_select_child`, and returns `rng.choice` of it, consuming the stream as Python
    does (even for a one-member set).
  - `readout(counts, rng)`: a transcription of `eval_readout._sample_by_temperature` at T 1.0, used for plies
    0–19 with the **separate readout stream**. The product's `selectMove` (`Math.random`) is not used under
    `protocol`.
- **Streams in production:** each request creates two `PyRandom` streams, search and readout, from fresh seeds
  drawn with `crypto.randomInt`. The research kept one stream per side per game across moves, so the two are
  equal in distribution, not stream for stream. A test-only constructor argument accepts fixed seeds. It is not
  reachable from any request.
- `server/mcts.js` itself is **not edited**: `MCTS.search` already dispatches through `this._expand` and
  `this._selectChild`.

### 4.4 New: `server/root_halving.js` (~130 lines)

- `class RootSequentialHalvingMCTS extends ProtocolMCTS` is a transcription of `root_sequential_halving.py`, with
  the same constants and a frozen-constants check.
- `search(state, opts)`:
  - below ply 20, delegates to `ProtocolMCTS`'s search, so plies 0–19 use the research tie rule too;
  - otherwise refuses a terminal root or a board other than 24×24, expands the root, and runs the rounds of §2;
  - returns `{visitCounts, rootValue, choice, choiceValue}`, with `choiceValue` the played candidate's mean from
    the mover's side (§5).
- Root-override simulations use a new `_simulate(root, overrideKey)` built from `MCTS`'s existing `_childFor`,
  `_expand` and `_backup`. Below the root it uses `ProtocolMCTS._selectChild`.
- Abort: if cancelled mid-rounds, it returns no choice. The caller treats that as an aborted search, never as a
  partial halving result.

### 4.5 Edited: `server/readout_policy.js` (~40 lines)

- `DIFFICULTY_TABLE.hard` becomes `{ nSims: 200, search: 'protocol', moveTemp: 1.0 }`. `moveTemp` applies only
  at plies 0–19; from ply 20 the halving choice is played.
- **Today's entry is kept by name:** `LEGACY_HARD = { nSims: 800, search: 'plain', moveTemp: 0 }`, selected when
  the server starts with `HARD_MODE=legacy` (§8). `easy` and `medium` are unchanged.
- `resolvePolicy` returns `search`.
- **Conflicting overrides are refused, not ignored (D3, decided).** For a request that resolves to `protocol`:
  - `deterministicMode: true` or any `temperature` (anything not `undefined` or `null`) throws a typed
    `PolicyConflict`;
  - the message is "hard plays the validated protocol; temperature and deterministicMode overrides are not
    accepted";
  - `deterministicMode: false` or absent is accepted, since it asks for nothing different.
  - The shipping client sends neither field on its move path (§3), so this changes no current game.
- Unknown difficulties still fall back to `medium`, as today.

### 4.6 Edited: `server/index.js` (~90 lines)

- **Hard mode at startup.** `HARD_MODE` is read once: `protocol` (the default) or `legacy`; any other value fails
  startup.
  - Under `protocol`, the hard model is resolved and loaded as a second ORT session (§4.7). `hard` uses
    `MirrorAveragingInference(hardSession)` with `RootSequentialHalvingMCTS`.
  - Under `legacy`, the hard model is **not resolved or loaded**, and `hard` uses `LEGACY_HARD` on the default
    session with plain `MCTS`: today's code path.
  - The banner prints the hard mode and every loaded model id.
- **Routing:** both transports (`/api/move` and WS `computeBestMove`) select the search class and session from
  `policy.search`. When the search supplies a `choice`, it is played instead of `selectMoveForRequest`. At plies
  0–19 under `protocol`, the move comes from `ProtocolMCTS.readout`.
- **Caching: bypassed for `protocol`** (the review). The REST handler neither reads nor writes the cache for a
  `protocol` request and returns `cached: false`. The WS path was never cached. Plain requests keep today's cache
  and scope, `${modelPath}|${nSims}`, byte for byte.
- **Refusals,** as an HTTP 400 or a WS `error` message with the same text on both transports:
  - a `PolicyConflict` (§4.5);
  - a terminal root under `protocol` (today this is not a defined refusal: `MCTS.search` does not check the root, and only a root with no legal moves surfaces, as "Search aborted");
  - a board other than 24×24 under `protocol`, unreachable from the shipping client.
- **Returned value under `protocol`:** defined in §5 (D6).
- **Response fields:** both transports add `search` (`'protocol'` or `'plain'`), `valueKind` (§5) and `modelId`.
  Existing fields keep their names and types.
- **Progress:** unchanged in shape. `valueEstimate` stays the root's running mean; no shipping display reads it
  (§3).
- The model-info endpoint reports the hard mode and both model ids.

### 4.7 Edited: `server/model_manifest.js` (~40 lines) — the hard model's own selection path

The review's point: `MODEL_MANIFEST` is the server's only model path today, so staging the student through it
would move easy, medium and the analysis endpoints too. The hard model gets its own path:

- `HARD_MODEL_ID` (the new id) and `HARD_MODEL_MANIFEST`, an override that affects **only** the hard session.
  It is the staging route for the student before its id is pinned.
- `resolveHardModel(env, cwd)` reuses `loadManifest` and `validateArtifact` unchanged, with the same no-fallback
  rule. `resolveManifestPath` keeps its signature and reads only `MODEL_MANIFEST`.
- **Separation, tested both ways (V5):** `MODEL_MANIFEST` never changes the hard model, and `HARD_MODEL_MANIFEST`
  never changes the default model.
- `DEFAULT_MODEL_ID` is unchanged.
- An invalid hard artifact fails startup under `protocol`, as an invalid default does today. The recovery is
  `HARD_MODE=legacy` (§8), which never reads it.

### 4.8 Test bindings

- Add the new modules and `server/index.js` to `EXECUTION_SURFACE_FILES`.
- Add a new frozen `STAGES` entry for the integrated surface. `eager`, `lazy` and `candidate-default` keep their
  digests and evidence (the August pattern).
- Re-point only the current-HEAD capture and attribution tests, recording the attribution change.

## 5. Displays and records under the new `hard` (the review's point; D6)

Keeping the analysis endpoints on the current model (D5) does **not** keep every display unchanged: the MCTS bar
shows the AI move's returned value.

- **NN bar:** `/api/evaluate` on the default model. Unchanged.
- **MCTS bar after an AI `hard` move:** shows `valueRed` from the move response. Under `protocol` the server
  defines `rootValue` (side to move) as follows, with `valueKind` naming which, and derives `valueRed` from it as
  today:
  - **plies 0–19, `valueKind: 'root_mean'`:** the root mean over the 200 plain simulations. This is today's
    definition.
  - **ply ≥ 20, `valueKind: 'played_move_mean'` (recommended):** the played candidate's mean from the mover's
    side (`−child.qValue`), over its 49 visits. This matches the client's own description of the bar: "the
    search's evaluation of the position the AI chose to play into" (`gameController.js:452–453`).
  - **Why not the root mean from ply 20:** under halving, 102 of the 200 visits are forced onto the 14 candidates
    eliminated early. The root mean is therefore pulled toward moves the search rejected, and is not comparable
    with today's bar.
  - **Alternatives:** (a) the root mean throughout, simpler but biased as above; (b) no bar update after
    `protocol` moves, which needs a client edit.
- **MCTS bar after a human move (recording mode only):** the recorder's `/api/analyze-position` precompute, on
  the default model with plain search. **This is unchanged, so in recording mode the bar alternates sources:** the
  student's protocol value after AI moves, and calib020_0001's plain-search value after human moves. Outside
  recording mode, only the AI-move value appears. This is stated as a product variation, not fixed. Fixing it
  would move analysis to the student, against D5.
- **Recorder (`gameRecorder.js:295–325`):** for AI `hard` moves from ply 20 it records the halving's visits. Its
  top-5 visit shares are then the **allotment** (the two finalists hold 49/200 = 0.245 each), not a preference
  measure. Its root value is the §5 definition. The client is unchanged and does not store the new `search` and
  `valueKind` fields, so recordings from new-`hard` games are identified by date and difficulty only. Listed in
  §9.

## 6. What this does not change

- **The student file** (`0e1cb621…`).
- **The current model** `c34b7ff3297c785a` and the earlier `1d64027db521a50f`, both byte-identical.
- **Existing search code:** `server/mcts.js`, `server/gameLogic.js` and `server/inference.js` stay unedited.
  `easy` and `medium` keep their search, tie rule, readout, cache and model.
- **The analysis endpoints** (`/api/evaluate`, `/api/analyze-position`): current model and plain search (D5,
  decided). The displays that read them are covered in §5.
- **The game-end rule:** the product's, unchanged (§9 item 6).
- **The research code and the client.**

## 7. Validation, in three approvals (none has run)

### G1 — implementation with fake-evaluator tests (no export, no model loaded, no game)

- **Scope:** the code of §4.2–§4.8, plus a Python fixture generator that drives the research classes with a
  **deterministic fake evaluator**, with no model.
- **Tests run:** only those that open no ORT session. Suites that load a model wait for G2.
- **V1. Mirror transform, exact:**
  - Fixture: `_mirror_position_lr` over the Phase 2 parity corpus states (`tests/parity/corpus.json`).
  - JS's mirrored NCHW buffer and mirrored move list must equal Python's **exactly**, on every element and in
    the same order.
  - Mirroring twice must give the identity.
  - With a fake inner evaluator, the averaged priors and value must equal Python's `MirrorAveragingEvaluator`
    exactly.
- **V4. RNG, tie rule and halving, exact, seed for seed:**
  - **RNG:** for a fixed set of seeds, `PyRandom` reproduces CPython 3.14.10's `random()`, `getrandbits`,
    `choice` and `shuffle` streams exactly.
  - **Tie rule:** with the same seeds and fake evaluator, `ProtocolMCTS` reproduces `MCTS.search_with_root`'s
    visit counts and root value. This includes fake positions built to have **exact ties**, **ties within 1e-8**,
    and an **order-dependent running-best case**, where the 1e-8 set depends on the shuffled order.
  - **Halving:** `RootSequentialHalvingMCTS` must match **exactly** on each round's survivors, the visit
    allotments, the final visit counts, the choice and `choiceValue`, with the root value to 1e-12. Positions
    cover m = 16, m < 16, m = 2 and m = 1; a winning child inside the candidates; and a terminal child two plies
    down.
  - At N = 200, m = 16 the allotment must be 3 / 6 / 12 / 28.
  - **Readout:** `ProtocolMCTS.readout` reproduces `eval_readout.select` (MODE_OPENING_TEMPERATURE, ply < 20)
    seed for seed.
  - **The product rule is untouched:** plain `MCTS` still breaks the same tie lexicographically.
- **V5. Routing, refusals, cache, values and model paths:**
  - At ply < 20, `hard` runs a 200-simulation `ProtocolMCTS` search and samples with the readout stream. At
    ply ≥ 20 it plays the halving choice, and the readout is bypassed.
  - REST and WS give the same move for the same request and the same fixed seeds.
  - A `PolicyConflict` is refused with the same message on both transports. `deterministicMode: false` is
    accepted.
  - A `protocol` REST request neither reads nor writes the cache: size, hits and misses are unchanged.
  - `rootValue`, `valueRed` and `valueKind` follow §5 at plies 19 and 20.
  - An aborted halving returns no move. A terminal root and a non-24 board under `protocol` are refused.
  - `easy` and `medium`: the resolved policy, search class, session, cache scope and response are byte-identical
    to today.
  - **`HARD_MODE`:** `legacy` resolves exactly today's hard policy, plain `MCTS` and the default session, and never
    calls `resolveHardModel`. An invalid value fails startup.
  - **Model-path separation** (manifest resolution with fixture manifests, as the existing tests do; no
    session): `MODEL_MANIFEST` does not change the hard model, and `HARD_MODEL_MANIFEST` does not change the
    default model.
- **V6. Legal moves and the game-end boundary:**
  - For every search mode, over seeded positions generated by random legal play: the played move is in
    `legalMoves()`, and halving candidates are a subset.
  - Boundary positions at plies 19 / 20, **279 / 280 / 281**, a nearly full board with m < 16, m = 2 and m = 1,
    and a root with a won or no-legal-move position.
  - Past 280, `protocol` searches and plays normally under the product rule; it has no 280 cap.
  - A won or move-less root is refused explicitly under `protocol`.
  - The ply / 600 feature stays below 1 at every reachable ply.

### G2 — export and numerical checks (loads models; no games)

- **V2. Export identity and model-output parity:**
  - The export of §4.1, with the student's sha256 unchanged before and after.
  - The manifest is validated by `resolveHardModel` through `HARD_MODEL_MANIFEST`.
  - Phase 2's surfaces S1–S4 for the student on its corpus, with its preregistered tolerances and ordering
    rules: S2 ≤ 1e-4, S3 ≤ 1e-5, S4 ≤ 1.1e-4 for max abs logit and value; top-1 agreement outside near-ties;
    at most 6 exemptions per surface pair; top-5 sets ≥ 95%; median Kendall τ ≥ 0.99.
  - S1 confirms the input tensor and legal-move list stay exact.
- **V3. Averaged-output parity:** MLX `MirrorAveragingEvaluator(student)` against Node
  `MirrorAveragingInference` on the same corpus, with the S4 bounds and V2's ordering rules applied to the
  averaged priors and value.
- **V7. Unchanged paths and suites:**
  - The golden corpus comparison (92 cases) under the new stage must show the plain path's visit counts, root
    value, selected move and progress exactly equal. This loads the current model.
  - The server, parity, match and golden suites, including those that open an ORT session; lint; typecheck.
  - The full `tests/test_htb_*` research suites, unchanged.
- **R. Rollback, actually switched back (the review's point; §8):**
  - **Reference:** from a clean worktree at `80acd87e`, today's server answers WS `hard` requests for the 92
    golden positions. This is deterministic: argmax with the lexicographic rule. Its move, visit counts and root
    value are recorded.
  - **R1, the startup switch:** the integrated server, started with `HARD_MODE=legacy` and the hard artifact made
    **unavailable** (`HARD_MODEL_MANIFEST` pointed at an absent path), must:
    - start;
    - print one model;
    - reproduce the reference for every position exactly;
    - give `easy` and `medium` searches identical to `80acd87e`'s (visit counts and root value; their sampled
      moves are random).
  - **R2:** the same server under `HARD_MODE=protocol` serves the student for `hard` only, and easy and medium
    still match.
  - **R3, the code revert:** on a throwaway branch, revert the integration commits; the reference comparison and
    the suites pass; the branch is discarded.

### G3 — product smoke test and match (plays games)

- **V8a. Smoke:**
  - `npm start` with the pinned hard model; a handful of `hard` moves through both transports;
  - the banner, model ids, `search` and `valueKind`;
  - one game in the browser client, to see the bars per §5;
  - the per-move time at plies < 20 and ≥ 20.
- **V8b. Product-stack match (D4, decided: required before any deployment),** with its rule proposed here so
  that G3 is one approval:
  - **Players:** new `hard` (the student under `protocol`) against today's `hard` (`HARD_MODE=legacy`:
    calib020_0001, 800 simulations, argmax). Both run in-process on the product code. Per-game fixed seeds go
    through the test-only constructor argument (§4.3).
  - **Games:** 110 colour-reversed pairs from the empty board, under the product's game-end rule (no 280 cap).
    Today's `hard` is deterministic, so the pairs differ only through the new side's sampling and ties.
  - **Rule:** PASS if the mean pair score − √(ln 20 / 220) > 0.5, the confirmation's one-sided 95% Hoeffding
    rule.
  - **Prediction:** to be stated in the match card before launch.
  - **Ceiling:** set mechanically at 1.5× the smoke's projected time. If that projection exceeds 24 h, I stop and
    return it rather than resizing.
  - **What it answers:** the confirmation transfers to the product only by argument. Game-for-game reproduction is
    impossible because the streams differ and the opponent differs. This match is the only evidence about what
    the user-facing change does.
  - **Records:** card, outcome and ledger entry as for every match. A FAIL or VOID is returned for assessment,
    with no retry.
- **Deployment** remains a final, separate decision after G3.

## 8. Rollback

- **Nothing is deleted:** `models/c34b7ff3297c785a/` and `models/1d64027db521a50f/` stay tracked and
  byte-identical. `server/test_model_manifest.js` already re-hashes the default pair (c34b7ff) but only reads
  1d64027's manifest. G1 adds the same pair-hash check for both retained pins.
- **The switch is a startup setting, not an edit:** restart with `HARD_MODE=legacy`. Today's `hard` entry is kept
  by name (`LEGACY_HARD`), and its route is today's code: plain `MCTS`, the default session, argmax. The hard model
  is never resolved, so the switch works even if the student's files are missing or damaged.
- **Tested, not assumed:** R1 and R2 start the real server both ways against a recorded `80acd87e` reference, and
  R3 rehearses the code revert (§7, G2). The review's concern was that deleting a model constant might not
  restore working routing. R1 tests the switched-back server itself, with the hard artifact absent.
- **Code revert, if wanted later:** revert the integration commits (R3 shows this restores `80acd87e`'s
  behaviour).

## 9. Product behaviour outside the tested protocol

1. **`easy` and `medium`**: 100 and 400 plain simulations, at temperatures 1.0 and 0.5, on calib020_0001. They
   are unchanged; the student is untested at these settings, and `medium` (the default) was never measured for
   calib020_0001 either (August §5.2).
2. **The new `hard` opening is stochastic** (temperature 1.0 for plies 0–19), as tested (D2, decided). Today's
   `hard` is deterministic, so the UI's promise of the strongest deterministic play changes.
3. **Client overrides** are refused under `protocol` (D3, decided). No shipping caller sends them.
4. **Opponents:** the PASS is against the original proven combination only. V8b adds today's product `hard`.
   There is no evidence against humans or twixtbot.
5. **Displays and records** (§5): the MCTS bar's ply-20 value definition (D6); its mixed sources in recording
   mode; and recorded visit shares that are halving allotments.
6. **The game-end rule (a stated product variation):** the research ended games at ply 280 as a draw, and its
   search treated ply 280 as terminal (1 game in 220 reached it). The product keeps its own rule.
   - The product draws at ply 600 or when the side to move has no legal placement.
   - Each ply places one peg on an empty cell, with no removal and no swap, and the board has 576 cells. **Ply
     600 is therefore unreachable on 24×24**, so the product's effective rule is a win or a draw by exhaustion.
   - Search and play past ply 280 are untested in the protocol. V6 checks the boundary mechanics, and V8b plays
     under this rule.
7. **Ties and randomness:** the research tie rule is ported (§4.3), so ties are equal in distribution. The streams
   are fresh per request rather than per game and side, so play is not stream-for-stream reproducible against
   the research.
8. **Runtime:** ORT CPU float32 against MLX on the GPU with TF32 off. The difference is bounded by V2 and V3's
   tolerances, not zero.
9. **Cancellation, caching, and non-24 boards:** none of these occurred in the protocol. `protocol` bypasses the
   cache, and aborts and non-24 boards are refused (§4.4, §4.6).
10. **Latency:** about 400 network calls per move plus the root, against 800 for today's `hard`. This is
    unmeasured until V8a.

## 10. Decisions already made (the user, 2026-11-09)

- **D1:** apply the integration to **`hard` only** (option B). Easy and medium stay on calib020_0001.
- **D2:** **preserve the tested random opening** (T 1.0, plies 0–19) for `hard`.
- **D3:** **refuse** conflicting temperature or determinism overrides clearly, rather than silently ignoring them
  (§4.5).
- **D4:** **require a product-stack match** before deployment (V8b).
- **D5:** keep the **existing analysis endpoints on the current model** and plain search.
- **Also decided in the review:**
  - port the research tie rule for the new `hard` only (§4.3);
  - give the hard model its own selection path (§4.7);
  - keep the product's move limit as a stated variation, with boundary checks and the match under it (§9 item 6,
    V6, V8b);
  - bypass the cache for the new `hard` (§4.6);
  - define the display behaviour (§5);
  - test an actual switch back (R1–R3);
  - group the work into G1–G3.

## 11. Decisions remaining for the user

- **D6, the value returned for `hard` moves from ply 20** (shown in the MCTS bar and recorded): the played
  candidate's mean (**recommended**, §5); the root mean; or no bar update (a client edit).
- **D7, the rollback mechanism:** the `HARD_MODE=legacy` startup switch, with R1–R3 (**recommended**); or code
  revert only, with R3.
- **D8, V8b's rule as written in §7:** 110 pairs, the one-sided 95% Hoeffding rule, and a mechanical ceiling,
  with a stop if the projection exceeds 24 h (**recommended**, so that G3 is one approval). Or a match card
  reviewed separately after V8a.

## 12. File changes and estimated effort

| file | change | size |
|---|---|---|
| `models/<new id>/{model.onnx, model.onnx.data, manifest.json}` | new, from the one export (G2) | ~7.6 MB, as the existing models |
| `server/mirror_inference.js` | new | ~70 lines |
| `server/py_random.js` | new: CPython `random.Random` subset | ~90 lines |
| `server/protocol_mcts.js` | new: research tie rule, shuffled order, protocol readout | ~80 lines |
| `server/root_halving.js` | new | ~130 lines |
| `server/readout_policy.js` | `hard` entry, `LEGACY_HARD`, `search` field, `PolicyConflict` | ~40 lines |
| `server/index.js` | `HARD_MODE`, second session, routing, cache bypass, refusals, §5 values, response fields, model info | ~90 lines |
| `server/model_manifest.js` | `HARD_MODEL_ID`, `HARD_MODEL_MANIFEST`, `resolveHardModel` | ~40 lines |
| `server/test_py_random.js`, `server/test_mirror_inference.js`, `server/test_protocol_mcts.js`, `server/test_root_halving.js`; additions to `test_readout_policy.js`, `test_server.js` and `test_model_manifest.js` | V1, V4–V6, R1–R2 | ~900 lines |
| `tests/fixtures/htb_integration/*` and a Python fixture generator | V1, V4 fixtures and RNG streams (no model) | small |
| `tests/mcts_golden/cases.mjs`, `tests/product_match/p_decision.mjs` | new stage and surface list | ~40 lines |
| V8b match driver and card | G3 | ~200 lines |
| ledger and an integration note | records | — |

**Estimated effort, engineering only:** about **7–9 working days**. Revision 0 estimated 5–7; the RNG port,
the tie rule, the rollback switch and its tests, and the boundary checks add about two.

| work | group | days |
|---|---|---|
| RNG port and tie rule, with V4's RNG and tie fixtures | G1 | 1–1.5 |
| mirror wrapper and V1 | G1 | 0.5–1 |
| halving port and the rest of V4 | G1 | 1–1.5 |
| policy, routing, refusals, cache bypass, §5 values, hard model path, `HARD_MODE`; V5 and V6 | G1 | 1.5–2 |
| export, manifest, V2 and V3 | G2 | 1 |
| surface stage, V7, rollback R1–R3, docs | G2 | 1 |
| V8a smoke and the V8b driver and card | G3 | 0.5–1 |

V8b's match compute is extra. It is projected by V8a and bounded by D8's ceiling.

**Order:** G1, then G2, then G3, each on its own approval. Work on a branch from `htb-pilot-build`; the product
code there is identical to `main`'s.

## 13. Changes from revision 0, by review point

| review point | revision 0 | revision 1 |
|---|---|---|
| tie-breaking differs in distribution | pinned as a documented divergence | ported for `hard` only: `py_random.js` plus `ProtocolMCTS`, tested seed for seed **with** ties (§4.3, V4); easy and medium keep the product rule |
| `MODEL_MANIFEST` staging moves every level | staging through `MODEL_MANIFEST` (§4.1, V2, V8a) | `HARD_MODEL_MANIFEST` and `resolveHardModel`, separation tested both ways (§4.7, V5) |
| 280 against 600 plies | "untested" | the product rule is kept as a stated variation; ply 600 is unreachable on 24×24; boundary checks in V6; V8b under the product rule (§9 item 6) |
| caching | protocol-scoped cache that replays the choice | bypassed for `protocol` (§4.6) |
| the MCTS bar uses the AI's returned value | "analysis and display unchanged" | §5 defines the value (D6), the mixed sources in recording mode, and the recorder's shares |
| rollback | "two one-line reverts restore today's `hard` exactly" (assumed) | `HARD_MODE=legacy` keeps today's route by name and never reads the hard model; R1–R3 test the real switch and the revert against a recorded `80acd87e` reference (§8) |
| overrides | silently ignored | refused with a clear error (§4.5) |
| approvals | V1–V8 each separately | G1, G2, G3 (§7) |
