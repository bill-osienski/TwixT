# Durable training replay capture — design specification

**Date:** 2026-10-11

**Status:** DESIGN ONLY; awaiting review. This document authorizes no implementation or execution.

**Status update, 2026-10-22:** the minimal serial path is IMPLEMENTED (§2B). Everything else in this document remains a deferred design. This update authorizes no generation, training, promotion or push.

**Repository inspected:** `484e32c1`; clean tree before this document.

**Scope:** opt-in persistence of targets already produced by the native self-play path, on 24×24 boards under our unchanged rules.

## 1. Problem and intended result

The normal saved game contains moves and summary statistics but omits the full policy target and generating model identity. Consequently, a viewer replay cannot reliably become an offline policy-training example. A move, root value or top-move share does not determine a visit distribution.

When capture is explicitly enabled in a separately authorized generation task, each completed game should produce a separate, validated artifact containing every searched decision's complete legal-move order, search visits and provenance. The artifact must distinguish observations from labels and retain the history needed to reconstruct each position. It must be possible to inspect and validate it without loading a model or performing search.

This addresses **one** blocker recorded in R0. It creates neither an independent teacher nor a useful corpus by itself. Capacity sensitivity controls, a qualified learning mechanism, compute pricing and a meaningful falsifier remain unresolved. Capturing targets establishes their identity, not their quality or strength transfer.

## 2. Authority and boundaries

- `calib020_0001` remains incumbent. All 30 experimental gates remain closed.
- The independent exact-rule teacher investigation remains **NO_GO**. Its memo stays in chat.
- Ledger do-not-repeat #53 prohibits extending or rescuing the archived external-learning trial. This design does not do so.
- T1j remains evaluation-only. Neither T1j nor any external engine is connected to this recorder.
- Accepted evidence correction `484e32c1` and maintenance repairs accepted at `ca9cb1eb` authorize no generation or training.
- No installations, downloads, model loads, engine queries, games, optimizer steps, seed registration, gate changes, commits or pushes accompany this specification.
- Prior held-out material and closed tactical pilot outcomes are excluded from design fixtures, tuning and verification. No old replay is upgraded by guessing missing targets or provenance.

References: `../2026-08-27-r0-post-anchor-research-decision.md`, `../2026-08-27-r1-external-data-discovery.md`, `../../alphazero-value-search-experiment-ledger.md` (#53), and the three 2026-10-10 result/successor/criterion memos. The successor and criterion memos' §6 corrections govern interpretation; target softness is not a quality gate, and the bounded tactical scan is not a universal prerequisite.

## 2A. First checkpoint: a small viability prototype

**User steering, 2026-10-11:** establish viability quickly before adding comprehensive guards. This section governs the first implementation checkpoint; the broader design below is a deferred destination, not a requirement to build everything before getting a result. Updating this design authorizes no implementation or test execution.

The first question is narrow: can a small CPU-only recorder preserve and reopen the exact native decision targets and their attribution? This is a capture-mechanics check. It cannot establish target quality, learning benefit, live-game non-interference or production throughput.

Build only the pure accumulator, minimal context/envelope, serialization and reader. Use one fixed, hand-authored completed 24×24 history with scripted decision snapshots and one own-crossing state fixture. Include both mover colours, zero-visit actions, inherited visits, an explicit selection adjustment and learner-row membership. Use visibly synthetic model identities and small named-tensor fixtures, including BN buffers. No engine-generated history, MCTS search, model creation/loading or trainer/worker wiring belongs to this checkpoint.

Keep only checks whose omission could make this small test report a false success:

- exact legal ordering, integer before/after counts and unchanged cumulative policy targets after selection adjustment;
- reconstructed state, mover, outcome, model attribution and retained-row source/transform;
- context binding, successful reopen and rejection of a missing required row or mismatched identity;
- source dictionaries copied rather than aliased, and no RNG draws introduced by capture;
- a fresh output directory, refusal to overwrite and no completion marker after a failed validation;
- fixed fixture size and output-byte ceiling, so the test is bounded.

The scripted adjustment must explain the observed count change; a non-null adjustment is not a blanket exemption. Scripted root counters exercise representation only: equality to the live search budget remains unmeasured.

Defer concurrent publication, shared worker budgets, worker-death handling, the comprehensive corruption/fault matrix, mutation testing, full-repository regression, live snapshot export and serial/parallel integration. Do not impose production hostile-input streaming or filesystem-race defenses on this fixed-input local prototype. The prototype is explicitly not a production corpus reader or a training source. Before enabling any deferred path, add and review the checks needed for that actual path.

Proposed acceptance budget for a later authorized implementation: one focused test file, the fixed fixtures above with at most 32 decision rows, context at most 64 KiB, game envelope at most 1 MiB and all published files at most 2 MiB total. Use at most two invocations of at most 60 seconds each before review. Record serialized bytes and write/read time; do not extrapolate them to real-game throughput. No parameter sweep, extra fixture search or experiment follows a pass automatically.

**Stop/report:** failure to retain exact counts/provenance or reopen consistently, an exceeded test/byte budget, or any need for prohibited execution. A pass supports only proceeding to a separately reviewed minimal serial wiring check. A fail identifies the concrete mechanics blocker before broader engineering work. Real model binding, search-counter qualification, games and performance measurement need their own explicit authorization; synthetic success does not waive those requirements.

## 2B. Implemented: the minimal serial path (2026-10-22)

**Result:** serial capture mechanics verified with a fake evaluator. Live-network integration, real-game throughput, target quality and strength are unmeasured.

What exists:

| File | Role |
|---|---|
| `scripts/GPU/alphazero/training_replay_capture.py` | Pure schema module: context and envelope validation, `DecisionCapture` accumulator, `PhaseWriter` (create-only publication, receipts, completion marker last) and the strict `load_phase` reader. |
| `scripts/GPU/alphazero/training_capture_serial.py` | `run_serial_capture`: plays the scheduled games one after another exactly as `train()`'s serial loop does (seeded RNG, one start-player draw, `play_game` with `train()`'s defaults) and publishes the phase. |
| `scripts/GPU/alphazero/self_play.py` | Optional `training_capture` keyword (last); `None` changes nothing. Guarded observation hooks around search, learner-row creation and selection. |
| `tests/test_training_replay_capture*.py`, `tests/test_training_capture_serial_driver.py` | Focused tests (54). |

Contract as built (it narrows §5–§9 where they differ):

- **Model identity** is the `named-tensors/1` digest of the live network's `parameters()`, which include BatchNorm running statistics. It is bound in the context before the first game and recomputed after the last; any change withholds `finalized.json`. The evaluator must serve that exact network object. No model snapshot file is written.
- **Search identity:** the context records `dataclasses.asdict(mcts_config)`; the driver refuses before writing anything unless it equals the configuration being run. Every row's root-visit gain must equal the configured `n_simulations`.
- **Refused in V1, before the context is written:** mirror augmentation (`_MIRROR_PROB` must be 0; its default is 0.5, so a capture run needs `TWIXT_MIRROR_PROB=0`), td1 forced visits and the closeout selection tie-break. Also refused, inside `play_game`: the position cap, resign, adjudication, a learner-player filter, non-24 boards and a non-280 cap. Only same-model self-play is supported.
- **`play_game` keywords:** `train()`'s defaults, plus only the two goal-completion diagnostic switches, recorded in the context. That they leave play unchanged is assumed, not tested.
- **Publication:** a fresh directory; each game is validated before and after its create-only hard-link publication; `receipts.jsonl` then `finalized.json` only when every scheduled game has exactly one verified receipt. `load_phase` accepts exactly that file set with matching hashes, byte counts and row totals.

Verified with real `MCTS` search and the repo's CPU `FakeEvaluator` (uniform priors, value 0; a random-init network is only hashed): the same seeded games give identical moves, learner rows and final RNG state with capture off and on; the reopened per-row targets equal the learner `PositionRecord` visit counts and outcomes; unsupported setups refuse before any file is written; a tampered or incomplete phase is refused by the reader. Two 280-ply games serialize to 3.17 MB (0.14 s write, 0.13 s reopen and validate).

Not exercised: inherited child visits in driver games (covered only by the separate real-search probe), decisive outcomes in driver games (both were state-cap draws; wins are covered by the scripted test), stall flushes, workers and IPC, and integration into `train()`. Those remain deferred, together with the rest of §9–§11.

## 3. Current implementation findings

These are source observations, not executed behavior checks:

| Location | Observation and consequence |
|---|---|
| `self_play.py: PositionRecord` | Holds NHWC board tensor, explicit mover, legal moves, visit counts, outcome and ply. Its serializer can carry these fields, but the ordinary viewer-saving path does not use that serializer. |
| `self_play.py: play_game` | Searches with a reused root; adds learner rows, optionally mirrors them, assigns outcomes and applies a per-game position cap. The returned training rows are not a complete, unaugmented decision transcript. |
| `mcts.py: search_from_root` | Returns counts for all legal moves, including zeros, after completing pending backups. Counts include inherited child visits. Forced visits, when enabled, consume the configured simulation budget. |
| `self_play.py: closeout selection` | Current code constructs `PositionRecord` **before** the selection tie-break. The nearby comment says the override propagates to training targets, but the source order and the tie-break's copied dictionary do not support that claim. This spec records both stages and does not repair or reinterpret the training objective. |
| `trainer.py: _save_game_from_record / _save_game_from_ipc` | Both forward history and summaries to `GameSaver`; neither supplies full position targets. |
| `self_play_worker.py / ipc_messages.py` | Rows travel on `position_queue`; completion metadata travels separately on `stats_queue`. Completion lacks dispatch game id, actual game seed and full targets. Arrival order cannot safely join the streams into games. |
| `game_saver.py` | Saves a viewer format; `seed` is the save-order game index, `config_hash` is the literal `alphazero`, and model hashes are absent. These fields are not adequate generation provenance. |
| Seed creation | Serial callers draw a game seed and discard the integer after constructing the RNG. Workers derive a seed from worker seed and dispatch id. Different workers have different seeds; the recorded worker assignment is needed. Identical dispatch ids alone do not guarantee reproducibility across scheduling changes. |

## 4. Architecture decision

Add a **separate training-capture sidecar**, leaving existing viewer artifacts and the in-memory training recipe intact.

Proposed new module: `scripts/GPU/alphazero/training_replay_capture.py`. It owns the versioned schema, immutable CPU snapshots, validation, serialization, publication and strict reading. It must not import MLX, instantiate an evaluator, invoke an agent or reserve a seed.

Capture defaults to absent (`capture_context=None`). Disabled capture allocates no accumulator, takes no extra state or RNG snapshots, hashes no model, writes no file and adds no queue traffic. New optional IPC fields default to `None`; existing consumers remain valid.

Enabled capture uses a separate accumulator inside `play_game`, independent of `positions` and replay-buffer sampling. Both movers' searched decisions are recorded; each names its generating model and whether it is a learner-side decision. Recording opponent decisions does not add them to the training buffer or grant permission to train on them.

At game completion, the **process owning the game** builds and validates the complete envelope and publishes its sidecar. Serial generation and workers use the same builder and writer. Workers send only a compact receipt through `GameComplete`; full rows never have to be reassembled from the existing position queue. The parent verifies receipts and produces a phase manifest. `GameSaver` remains a viewer writer, not the authority for training capture.

The collector must surface capture errors through the existing fail-closed worker/parent path. A failed enabled capture is a failed capture phase, not a warning followed by an apparently complete corpus. Any changes to failure propagation require review against the accepted cleanup repair.

## 5. Supported generation contract

V1 accepts only native generation with:

- physical and active board sizes both **24**;
- automatic links; no link removal; crossings forbidden for both owners; swap disabled;
- an empty initial board, explicit starting player and a **280-ply draw cap**;
- resignation and model-based adjudication disabled;
- a frozen model instance for each role throughout the generation phase;
- a predeclared bounded phase context and capture for **every scheduled game** in that phase.

Reject incompatible enabled configurations before model setup or worker launch. Do not silently change a configuration to make it eligible. Starting-player policy is recorded explicitly and is a future card's decision; capture does not randomize or replace it. Nonempty opening starts need a separately reviewed schema extension.

Self-play may identify both colours with the same model. Existing native dual-model plumbing can carry different identities and learner colour; this support does not reopen frozen-parent opposition. A future authorized generation phase retains the single Metal-owning inference arbiter; capture adds no device owner.

Whole-game viewer sample limits must not determine corpus membership. The capture manifest lists the complete scheduled game-id set in advance. Any missing game prevents phase completion. No top-up, substitution or outcome-dependent retention is allowed by this infrastructure.

## 6. Schema: `twixt-training-capture/1`

One UTF-8 JSON envelope per game, plus a shared generation-context file and a final phase manifest. All hashes use SHA-256. Coordinates are zero-based native `[row, col]`; plies count pegs already played. No colour rotation or tensor canonicalization occurs in the recorder.

### Generation context

| Field | Required meaning |
|---|---|
| `schema`, `run_id`, `phase_id`, `iteration` | Version and supplied stable identities; no save-order identity. |
| `source_kind` | Exactly `native_self_play`; evaluation records and external-engine labels are rejected by V1. |
| `scheduled_game_ids`, `family_assignment` | Complete dispatch-id set; each assigned game names a supplied family. Standalone games may have distinct families. Related games, colour arms and later transforms retain the same family. IDs assert grouping, not statistical independence. |
| `authorization_identity` | Future card and approved execution/code identity; a data field is not an authorization check. Existing gate checks remain authoritative. |
| `code_identity` | Commit plus explicit source-closure list and hashes for recorder, game rules, tensor encoder, MCTS, self-play, worker/IPC, RNG setup and relevant configuration/launch code. A dirty relevant source closure requires an approved, hashed patch identity; no silent label of clean HEAD. |
| `rules` | Explicit fields for board, automatic links, removal, each crossing rule, swap and cap, plus rules-source hash. A generic ruleset name is insufficient. |
| `models` | Per-role immutable state identity and snapshot-file identity, architecture/encoder descriptor, inference dtype and loaded-state verification record. Paths are descriptive, hashes authoritative. |
| `configuration` | Resolved per-role MCTS configuration; effective noise settings, move-selection temperature schedule, tree reuse, visit-forcing and tie-break settings; effective mirror probability, position-cap settings and learner-side filtering; starting-player policy, workers, batching/runtime and RNG algorithm/version. Persist resolved values, including environment-derived settings, rather than CLI text alone. |
| `limits` | Scheduled game count, maximum bytes per game and maximum bytes for the phase; actual approved values required before capture is enabled. Context maximum is 1 MiB, and non-array metadata per game is at most 64 KiB. Count all context, game, receipt and finalization bytes against the phase limit. |
| `runtime` | Python and relevant runtime/library versions, platform and evaluator backend. This is provenance, not a determinism guarantee. |

**Model identity cannot be inferred from iteration number or the initial checkpoint.** A learner may have changed since that checkpoint. Before generation, the parent must bind the exact live parameters and BN buffers to an immutable generation snapshot, and verify that identity against the model instances serving the phase. Use a versioned state-digest algorithm over sorted tensor names, shapes, dtypes and exact bytes, covering every parameter and running statistic. File hash and state digest are separate fields. Creating/verifying a snapshot has its own cost and must be included in a future card; the recorder does not load or export a model itself. Workers receive the verified context and cannot supply arbitrary model identity. Model changes during generation invalidate the phase.

### Canonical encodings

- Structured digests use UTF-8 JSON with sorted object keys, compact separators, `ensure_ascii=False` and `allow_nan=False`. Lists retain their prescribed order. Hash the stored canonical context bytes; game-file receipts hash exact file bytes. No timestamps, paths or timing fields enter the semantic state digest.
- A state digest covers an object containing `board_size`, `active_size`, `to_move`, `ply`, `max_plies_limit`, `pegs` as sorted `[row,col,colour]` entries, and `bridges` as sorted endpoint pairs with each pair ordered lexicographically. Include all fields even when empty. This is separate from the current `TwixtState.to_dict`, which omits the cap.
- Model state digest version `named-tensors/1` starts with that UTF-8 version string followed by a zero byte. For every tensor in lexicographic name order, append the canonical JSON header `{name,shape,dtype}` and its contiguous C-order tensor bytes, each preceded by an unsigned 64-bit little-endian byte length. Numeric tensor bytes use explicitly little-endian storage; dtype includes width and kind. Include BN buffers. Reject unsupported/object/structured dtypes rather than coercing or quantizing. Architecture and tensor-encoder descriptors receive separate structured digests. The same procedure verifies the live generation state and the immutable snapshot state.
- Python `random.Random` state is stored as `{algorithm:"python-random",python_version,state_version,internal_state,gauss_next}` from the existing `getstate()` result; the integer tuple is an ordered JSON list, and the cached Gaussian is finite or null. A future reader reconstructs tuples explicitly. Every additional RNG stream requires its own named entry and reversible encoding; unknown algorithms refuse. Python version and source identity remain part of the reproducibility limits.

### Game envelope

| Field | Required meaning |
|---|---|
| `schema`, `context_sha256` | Schema and hash of the shared context bytes. |
| `run_id`, `phase_id`, `iteration`, `game_id`, `family_id`, `colour_arm` | Stable context-bound identities; `game_id` is dispatch identity, never viewer save order. Arm may be null for an unpaired game. |
| `worker_id`, `execution_mode` | Worker assignment or null for serial; mode is `serial` or `worker`. |
| `rng_provenance` | Actual game seed, parent/worker seed identity and seed derivation version, plus state of each RNG stream at game entry and at first search. Retain the already drawn opponent seed when applicable. Snapshot via observation, never by drawing a new number. Account for starting-player selection and augmentation/capping use of the same stream. Do not encode absent provenance as a plausible seed. |
| `start_player`, `models_by_colour`, `learner_player` | Actual start and colour-to-model bindings; learner colour null for ordinary self-play. |
| `move_history` | Complete ordered played moves, with no mirrored rows. |
| `rows` | Exactly one unaugmented searched-decision row per played move. |
| `outcome` | Winner or null, terminal reason (`win`, `board_full`, `state_cap`), final ply and final state digest. |
| `buffer_membership` | Existing trainer-retained rows expressed as source-row id plus `identity` or `mirror_lr` transform. Preserve repeated source plies as distinct transforms; record before/after cap counts. This documents selection without changing it. |
| `capture_summary` | Row count, total represented legal actions, measured file bytes and elapsed capture wall time. File byte count belongs in the receipt if including it here would make serialization self-referential. |

RNG state serialization must be reversible and include generator version; a hash alone cannot restore it. Preserve seed draws at their existing locations. Serial and worker modes are not asserted to generate the same games for the same nominal seed.

### Decision row

| Field | Required meaning |
|---|---|
| `row_id`, `ply`, `to_move`, `model_id`, `learner_side` | Stable `(game_id, ply)` identity and active generating model. |
| `state_sha256` | Canonical pre-move state digest over board sizes, mover, ply, cap, sorted pegs and sorted canonical bridge endpoints. Reconstruct from start and history; a board tensor alone is insufficient. |
| `legal_moves` | Exact native sorted full legal set from the pre-move state. Include zero-visit moves. |
| `child_visits_before` | Completed child counts at entry to this root search, before noise or forced simulations, aligned with `legal_moves`; absent children contribute zero. |
| `child_visits_after` | Exact completed search counts returned after all pending backups, aligned with the same legal order, **before** selection tie-break or temperature. |
| `root_visits_before`, `root_visits_after` | Root counters at the same boundaries; do not assume root count equals the sum of child counts. |
| `configured_simulations`, `completed_simulations`, `forced_simulations` | Budget and observed per-search counter deltas; forced simulations are a subset, not additional budget. No extra search is performed to measure them. |
| `root_value` | Finite searched value from mover perspective; diagnostic, not an outcome label. |
| `selection_counts`, `selection_adjustment`, `selected_move` | Counts actually passed to `select_move`, tie-break result or explicit absence, and played move. Record after the adjustment without substituting them for search visits. |
| `z` | Completed game outcome from this row's mover perspective: +1, −1 or 0. Provenance is `completed_game`, including cap draws; not a value estimate or proof of optimal play. |

Store integer counts, not a rounded floating distribution. Define the reconstructible policy target as

`pi[i] = child_visits_after[i] / sum(child_visits_after)`.

Its semantics are **cumulative completed root-child visits, including inheritance**. Also expose the exact derived vector `delta[i] = after[i] - before[i]`; it describes this search's new visits. V1 does not choose delta as a training objective. Zero total after-visits cannot supply a policy target and invalidates capture; do not fall back to uniform or the played move. One-hot and diffuse valid counts both pass structural validation.

No raw network priors, search-improvement claim, target entropy threshold or teacher-quality flag is included. These targets reflect the actual pinned search configuration, including any enabled noise, prior shaping or within-search forcing; they are not automatically unbiased or optimal labels.

## 7. Capture boundaries and existing learner rows

1. Establish and verify the phase context in the parent before generation; carry only CPU metadata into workers.
2. At each real decision, independently snapshot the pre-move state identity, complete legal set and active root's inherited counts before `search_from_root`.
3. Immediately after the existing search returns, take owned copies of counts/counters/value. Later dict or tree mutation must not alter them. Record actual active-agent identity, including opponent plies.
4. Keep the existing learner `PositionRecord` construction and mirror RNG behavior unchanged. The source currently uses the pre-tie-break counts. Associate transient training-row references with source row and transform, without persisting Python object IDs or changing the buffer's contents.
5. Observe the actual post-adjustment selection counts and selected move; finalize the decision row before advancing roots. Never sort or mutate the dictionary passed to selection.
6. Observe which existing position objects survive `apply_game_position_cap`; translate their transient associations into `buffer_membership`. Do not repeat sampling, introduce RNG draws or replace the cap algorithm.
7. At natural game termination or the 280-ply draw cap, attach outcome labels to all captured rows. Reconstruct and validate the complete envelope before publication.

Resignation, adjudication searches, abandoned searches, runtime interruption and exceptions never yield a usable V1 envelope. They are incompatible or incomplete capture, not silently skipped decisions. V1's empty-start contract makes `len(rows) == len(move_history) == final_ply` exact.

## 8. Validation and reading

One strict CPU reader validates context and game hashes, schema/version, complete identities and supported settings. It replays recorded moves through the pinned rules engine with the recorded starting player and cap. It must independently check, for every row:

- state digest, mover, ply and colour/model mapping agree with reconstructed history;
- legal order exactly equals the engine's full legal order; coordinates are distinct and in range;
- count lengths match legal order; counts are nonnegative integers (booleans are rejected);
- each after-count is at least its before-count; count and root deltas agree with the supported search's completed-budget contract; forced count is within that budget;
- root and child totals remain separately named; inherited totals are never compared directly to the configured fresh simulation budget;
- selection counts refer to the same legal set; recorded adjustment accounts for differences; selected move is legal and matches history;
- `z`, winner, reason and final state match replay; cap draw cannot overwrite an already established win;
- membership references existing learner-side rows, with allowed transforms and consistent cap counts; opponent-side rows cannot be marked trainer-retained;
- no duplicate game/row identity, missing row, truncated file, NaN or infinite scalar exists.

Validate budget-counter equalities against hand-constructed synthetic root snapshots before adopting them. If the existing search has an unrepresented counter case, stop implementation review and amend the contract explicitly; do not disable checking to accept it.

Reading reconstructs tensors through the pinned encoder and records its descriptor. It does not regenerate search labels, load a network, infer missing model identities or accept an old viewer file as this schema. Any adapter to `PositionRecord` or offline training is a separate work item; publishing these artifacts does not automatically put them into a learner.

## 9. Publication, completion and resource limits

Layout under a supplied, new capture directory:

```text
context.json
games/game_<dispatch-id>.json
receipts.jsonl
finalized.json
```

All authority-bearing files are create-only. Write a unique staging file on the same filesystem, validate its exact serialized bytes, flush/fsync, and publish with an atomic **no-replace** operation. On this host an implementation may use an exclusive staging file and hard-link publication; an ordinary replacing rename is insufficient. Unsupported publication semantics must refuse. Final-path existence is an error, even if bytes happen to match. No overwrite, resume, repair or automatic retry is in V1.

Receipts contain identity, relative path confined to the capture directory, file SHA-256, row count and byte length. Worker receipts travel in an optional `GameComplete` field. The parent verifies file contents and context binding before acknowledging them; process identity cannot override envelope identity. Different queue arrival orders cannot change filenames, membership or model attribution.

`finalized.json` is written only after all predeclared games have exactly one verified receipt, all workers and the inference server have stopped successfully, cleanup is verified and no capture failure occurred. It binds context and the complete sorted receipt list. Files without that phase marker are incomplete-phase evidence and cannot constitute an approved corpus, even if individual games finished. A crash between game publication and receipt leaves an orphan; do not adopt it automatically. A loader verifies the manifest's membership and rejects unlisted files offered as members.

Each future enabled phase must set hard game-count and serialized-byte limits before launch. Proposed V1 per-game ceiling: **8 MiB**; the phase byte ceiling is supplied explicitly by its card. A bounded serializer stops before exceeding these limits; memory must also be bounded by at most 280 rows with at most 528 legal actions per row, and by the supplied metadata-size limits. No board-tensor array is duplicated in every row. On limit or I/O failure, stop the phase and withhold finalization; do not drop large games or shorten distributions.

Measure serialization/validation duration and output bytes. Added copies, replay validation, hashing and disk writes have unmeasured cost. They add no searches or RNG draws, but wall-clock overhead can change worker scheduling and deadline behavior. Bit-identical parallel scheduling or zero overhead is not promised.

## 10. Synthetic acceptance suite for a later implementation

Tests are specified here, **not executed**. A separately approved implementation uses hand-authored 24×24 fixture snapshots and filesystem temporary directories. No historical held-out rows, closed tactical outputs, model files, teacher queries or agent-generated games are used. Synthetic source closures/model identities are visibly marked `fixture`, and production validation rejects that namespace.

| Test | Independent acceptance assertion |
|---|---|
| Full-target round trip | Hand-authored legal-order/count arrays retain every zero, integer and coordinate. A separately calculated rational normalization agrees; no played-move fallback. |
| Rule reconstruction | Hand-authored histories include both mover colours, a blocking own-link crossing and a terminal/cap edge. Replayed pegs, bridges, legal sets and tensor descriptor agree with independently expected states. |
| Inherited visits | Fixture with nonzero entry visits yields the exact expected delta and cumulative target. A misleading `sum(after) == fresh_budget` assertion is demonstrably rejected. |
| Selection separation | Synthetic adjustment changes played-move preference while the captured search counts and existing training-target snapshot remain unchanged. Mutation of source dictionaries after capture cannot change the artifact. |
| Filtering and augmentation | Both movers are observed; only actual learner rows appear in membership. Mirrored rows share source/family identity and name their transform; capping does not discard capture rows or consume a second RNG sample. |
| Distinct model provenance | Two visibly different fixture model identities are assigned opposite colours across envelopes. Deliberately swapped attribution fails; same-model self-play passes. No test oracle derives expected identity from the object being validated. |
| Serial/worker envelope equivalence | Feed the same scripted decision snapshots and context through both capture plumbing paths. Semantic payloads agree aside from declared execution metadata. This is a capture test, not a real self-play equivalence claim. |
| Queue reordering | Deliver two completion receipts in reverse order relative to synthetic position chunks, then duplicate or omit one receipt. Correct order-independent identity passes; duplicate/missing completion prevents finalization. |
| Capture disabled | With injected search/selection/I/O spies and scripted snapshots, no recorder snapshots, writes, RNG accesses or additional agent calls occur. Existing learner-row and completion-message behavior remains valid. Do not call a live game generator. |
| Corruption | Missing/extra legal move, reordered counts without reordered moves, negative/fractional/bool count, decreasing child count, zero mass, wrong state/mover/outcome, wrong context/model hash and truncated JSON each fail explicitly. |
| Unsupported configuration | Non-24 board, swap, removal/crossing change, nonempty start, different cap, resignation/adjudication, missing seed/model/byte-limit provenance and external-engine origin refuse before setup. |
| Publication and shutdown | Existing target, concurrent duplicate publication, mid-write interruption, fsync/link failure, byte limit, bad path, orphan and unverified cleanup each withhold finalization. A complete manifest remains verifiable after reopen. |

Pure tests must run without MLX or model access. Wiring tests may use injected collaborators and scripted events, not real evaluators or self-play. Stubs qualify capture mechanics only; they cannot prove hardware behavior, real-game non-interference, runtime throughput, target quality or strength.

## 11. Implementation boundary and subsequent approvals

The later implementation should add the pure capture module and focused tests, then thread optional context/receipt fields through `self_play.py`, `self_play_worker.py`, `ipc_messages.py` and the serial/parallel seams in `trainer.py`. Existing `PositionRecord`, learner filtering, optimizers, MCTS selection/search algorithms, `GameSaver` viewer schema and seed derivation remain behaviorally unchanged. Search-boundary observations may read existing counters; they may not alter backups, priors or counts.

Review must resolve and freeze the model-state digest/snapshot binding, RNG state encoding, source closure, byte-limit plumbing and publication primitive before implementation is accepted. These are engineering contracts, not parameter grids. Failure to represent an existing behavior is a stop requiring an explicit spec revision.

**Minimum next approval:** review this spec, then authorize capture implementation and the listed synthetic checks only. No new run gate should be opened in that implementation. Separately approved real generation would require a qualified research objective, fresh approved seed accounting, immutable model/context identities, measured overhead and fixed compute/storage/stop budgets. No such generation is recommended or authorized here. Training, strength evaluation and promotion would each retain their own subsequent requirements.

Success for this work is a reviewed and synthetically verified ability to preserve future targets faithfully. It is not a claim that any captured target is good, that a larger network will help, or that any closed experiment should resume.
