# H-TB pilot card — whole-network distillation of twixtbot's raw network (2026-11-03)

**Status:** FROZEN at the sha256 pinned in `scripts/GPU/alphazero/htb_pilot.py` (`CARD_SHA256`). Any edit means
re-pinning under review.

**Authorized by the user's decision (2026-11-03):** revising and freezing this card, and building the minimal CPU
pilot with focused CPU and fake checks. Narrow #53 and #87 exceptions cover **this card and its build only**. One
pinned `variables.index` metadata read was also authorized; it was done, and its results are in §2.

**Not authorized:** installing TensorFlow, acquiring the checkout, loading any weights, real labelling, fitting,
games, GPU work, benchmarks, deployment and pushing. Every runtime step in §10 needs its own approval.

**Reference to beat (later stages):** the original proven combination: `calib020_0001` (SHA-1 `209cf2d4…`) with
mirror averaging, 200 simulations, ordinary search through ply 19, and root sequential halving from ply 20.

## 1. Question and hypothesis

**H-TB:** can the unchanged 128×6 network, warm-started from `calib020_0001` and trained as a whole to imitate
twixtbot's raw network (its policy *and* value), imitate it much more closely than `calib020_0001` does? And, in later
stages only, does that student play more strongly than the reference inside the unchanged proven combination?

**This card runs only the fit check (§§4–8).** The screen and the strength test (§9) need their own build and
approvals.

**What the fit check can and cannot show (correction 3):**
- Passing F1–F3 establishes **imitation of the teacher on this position distribution**, and nothing more. It does not
  establish stronger play. It does not show that training scale, or any other source difference, caused the gap.
- Failing F1–F3 **cannot separate** architecture from optimization, data or budget.
- A STOP does not authorize anything, an architecture experiment included.
- Training the value head toward the teacher removes D2's value-*preservation* requirement. It does not establish
  that value learning will succeed; F3 tests that directly.

## 2. Corrections to the proposal and the source review (recorded here; their text stays unedited)

1. **Leakage.** C0 measured 9.17–9.60% of test-arm positions repeated in the train arm, all in the opening book.
   Splitting by whole game is not enough. §4 removes duplicate canonical states.
2. **Fitting specification.** The "about" counts are replaced by exact rules and counts (§§4–6).
3. **Conclusions narrowed** (§1).
4. **Compute.** The student's architecture is unchanged, but that does not by itself match search cost. Both
   measured guards stay (§9). This first trial runs on the **CPU**. The 5–12 h estimate is provisional until a bounded
   timing step supports it (§10).
5. **Source claims:**
   - The proposal's "±13 cells" describes only the *convolutional path*. Our 6 connectivity input planes carry
     non-local information, so the network's effective view is wider.
   - Our value head's global pooling makes that pooled summary translation-invariant, not the whole network: the
     convolutions are zero-padded, and the inputs include edge planes.
   - Matching variable blobs establish **unchanged checkpoint bytes**, not conversion parity. The teacher here is the
     converted graph we actually evaluated in G3 (#52), whatever its relation to the original TF1 graph.
   - The earlier imitation fit (#53) used **128 training rows plus 64 held-out rows**, not 192 training rows.
6. **From the authorized `variables.index` read** (evidence `docs/superpowers/evidence/htb-teacher-index/`;
   pin `1d4073c4…`; 240 variables whose byte extents sum to exactly 10,270,828, the data-shard size):
   - **Trunk:** **20 residual blocks** of two 5×5×48×48 convolutions each. The first layer is pegs 5×5×2×48, links
     4×4×8×48 and location 1×1×**2**×48, so there are **no recent-moves planes**: the teacher's input depends only on
     pegs, links and side to move.
   - **Value head:** **four** strided 5×5×48×48 convolutions, a 192×80 dense layer and an **80×3** output.
   - **Policy head:** 1×1×48×2, then 1×1×2×1.
   - **Other variables:** 47 BatchNorm layers; an int32 global step; no optimizer slots.
   - **What the read cannot establish:** the graph's operations (activations, strides, padding, wiring). The
     flattened 192 = 2×2×48 is *consistent with* four stride-2 SAME reductions of 24×24. The global step's *value* was
     not read.
   - **Corrections to my first-pass figures:** the builder defaults (12×40) are not this model's configuration. The
     size scan's "no match" came from my own scan bounds (reductions ≤ 3) and an int64 step assumption. On the
     convolutional path, the teacher's theoretical receptive field is ±82 cells.

## 3. Teacher, its inference path and its outputs

- **Identity:** stevens68/twixtbot-ui @ `83749f230a0bae1766b46a05bfde0ed87f0a9a0a`, `model/pb`. The sha256 pins are
  those already in `twixtbot_g3_schedule.ANCHOR`: weights `1958f847…`, saved_model `e0c2b882…`, index `1d4073c4…`. The
  runtime is verified by `external_learning.load_runtime`, which reuses G3's verifier, imports by module origin and
  hides every GPU from TensorFlow.
- **Inference:** `nneval.NNEvaluater("model/pb").eval_one(naf.NetInputs(game))`, one position per call, rotation off,
  no noise, no search. This is the raw-network path the UI's `nnmplayer` uses with rotation 0.
- **Building the teacher's position:** the teacher's `twixt.Game(allow_scl=False)` is built in lockstep with our
  `TwixtState` from the game's move list (`twixtbot_adapter`). At every labelled ply, `state_divergences` must be
  empty, and the teacher's legal set (`tb_legal_moves`) must equal ours. Otherwise the run STOPs.
- **Value:** the 3 logits (loss, draw, win) are converted by the engine's own `naf.three_to_one`, via
  `twixtbot_adapter.pwin_to_value`: v = p_win − p_loss in [−1, 1]. The result is **from the side to move's
  perspective** (`naf` always puts the side to move "on play", and the UI's search negates child scores). This is the
  same convention as our value head and the mirror-averaging evaluator. No rescaling is applied.
- **Policy:** the 528 move logits are mapped through `twixtbot_adapter.policy_index(colour_to_move, r, c)` onto our
  legal moves, in our `legal_moves()` order. The policy is the softmax over the legal moves only, which is exactly the
  UI search's prior before noise. Any non-finite value STOPs the run.

## 4. Position selection (deterministic; no RNG)

1. **Corpus:** `scripts/GPU/logs/games`. It must first verify byte-exactly against C0's committed
   `docs/superpowers/evidence/2026-08-24-capacity-c0/inventory_games.json`, through
   `capacity_c0_inventory.inventory()`: eligible digest `dc7713c6…`. Only C0-eligible games are used.
2. **Arms:** C0's `split_of(game_id)`. **train** and **val** only. No test-arm game is replayed, keyed, labelled or
   scored. The verification digest reads every file's bytes, test arm included, only to authenticate the corpus.
3. **Positions:** every position before each move of the game, plies 0 to n−1. In our rules all of these are
   non-terminal.
4. **Canonical key.** The state is (sorted red pegs, sorted black pegs, sorted canonical bridges, side to move). The
   key is the sha256 of the smaller of the state's serialization and its **left-right reflection's**, with column
   c → 23 − c and bridges re-canonicalized. This is the symmetry the agent uses (mirror averaging). Our colour
   canonicalization (rotate and swap) never maps one reachable state onto another: red-to-move states have equal peg
   counts, while black-to-move states have one more red peg. Both networks' inputs are functions of the state alone.
   The teacher has no recent-moves planes (§2.6), and our planes derive from pegs, links, side and ply, where ply is
   the peg count.
5. **Train pool:** one entry per distinct key over the train arm, keeping the first occurrence in (game_id, ply)
   order.
6. **Validation pool:** val-arm positions whose key occurs **nowhere in the train arm** (all positions, not only
   those selected), one entry per distinct key, first occurrence in (game_id, ply) order.
7. **Exact counts:** **N_TRAIN = 240,000** and **N_VAL = 24,000**. Each pool is ordered by
   sha256(`"htb-select/1|<arm>|<key>"`) and the first N are taken. A pool smaller than N **REFUSES**; there is no
   fallback.
8. **Selection record:** create-only, with the game_id, ply and key of every row, the pool sizes before sampling, and
   the counts of keys excluded for leakage. Its sha256 binds the labels and the fit.

## 5. Targets: one symmetric definition everywhere

Write R for the left-right reflection, with cells (r, c) ↦ (r, 23 − c). The teacher sees R(s) when the reflected
move list is replayed in lockstep through both engines. Our state from that replay must equal R applied to our state
(pegs and bridges); otherwise the run STOPs.

- **Teacher target:** T(s)[m] = ½(p_T(s)[m] + p_T(R s)[R m]), and v_T(s) = ½(v_T(s) + v_T(R s)). These are
  mirror-averaged, and they are the only targets used.
- **Any network Q** (the student or `calib020_0001`) is scored by the **mirror-averaged** prediction
  Q_MA(s)[m] = ½(p_Q(s)[m] + p_Q(R s)[R m]), with v_Q,MA defined the same way. This is computed exactly as
  `MirrorAveragingEvaluator` computes it (`_mirror_position_lr` boards, moves (r, 23 − c)), in evaluation mode.
- **Training** applies the loss to the student's **raw** outputs on **both orientations** of every row: s with T(s),
  and R s with T mapped by R. A symmetric target on both orientations is what drives Q_MA toward T.
- **Initial KL, validation and every reported metric** use Q_MA against T. "Raw" student, reference or teacher
  outputs appear only inside the definitions above.

## 6. Fit specification

- **Device:** MLX CPU (`mx.set_default_device(mx.cpu)` before any load). `MLX_ENABLE_TF32=0` is required in the
  environment anyway.
- **Initialization:** `calib020_0001`, loaded strictly through `probe_eval.load_network_for_scoring` after a SHA-1
  check (`209cf2d4…`). A second, untouched load is the reference arm. The student's state digest must equal the
  reference's before update 0.
- **Update surface:** every trainable parameter: encoder, policy head and value head. There is no value adapter.
- **BatchNorm:** **evaluation mode throughout**, in training and in scoring. The running statistics are never
  updated and must be bitwise unchanged after the fit; γ and β are trainable. This is D2's convention: it avoids
  batch-composition effects and train/eval mismatch.
- **Loss per update**, over B = 32 rows × 2 orientations = 64 tensors:
  - mean cross-entropy of the target over the legal moves (`compute_masked_log_probs`),
  - plus 1.0 × the mean of (v_student − v_T)²,
  - plus 1e-4 × the sum of squares over all trainable parameters.
- **Optimizer:** MLX `Adam`, learning rate 1e-4, with MLX's resolved defaults read back and checked: betas
  (0.9, 0.999), eps 1e-8, no bias correction. The gradient global norm is clipped at 1.0.
- **Updates:** **2 epochs × 240,000 / 32 = 15,000 updates**. Epoch e orders the train rows by
  sha256(`"htb-fit/1|<e>|<key>"`); batches are consecutive runs of 32. No RNG is used. A non-finite loss or gradient
  norm STOPs the run.
- **Start check** before update 0: on the first 2,048 rows of epoch 0's order, the mean KL(T ‖ reference_MA) must be
  ≥ 0.01. Otherwise the run STOPs, because there is nothing to learn.
- **Records:** a step log every 25 updates. The student is saved create-only as safetensors outside the repository
  (`~/twixt-cache/htb-pilot/`), with sha256 and state digest. The record names the card, selection, label and
  runtime digests.

## 7. Scoring (all 24,000 validation rows; frozen before labelling)

| | Quantity | Definition |
|---|---|---|
| KL_Q | policy divergence | mean over rows of Σ_m T[m] log(T[m] / Q_MA[m]), over legal moves with T[m] > 0 |
| TOP_Q | top-1 agreement | fraction of rows where argmax Q_MA = argmax T. Ties go to the first move in our `legal_moves()` order |
| MSE_Q | value error | mean of (v_Q,MA − v_T)² |

These are computed for Q = the student (S) and Q = `calib020_0001` (R).

- **F1:** KL_S ≤ 0.5 × KL_R
- **F2:** TOP_S > TOP_R (strict)
- **F3:** MSE_S ≤ 0.5 × MSE_R

The 0.5 factors were chosen before any data; they are not derived.

## 8. Verdict

- **ELIGIBLE** if F1, F2 and F3 all hold, every check in §§3–6 passed, and the BatchNorm statistics are unchanged.
  Otherwise **STOP**.
- An interrupted or timed-out run is **VOID**, with no automatic retry.
- ELIGIBLE authorizes nothing by itself. Interpretation is limited to §1.

## 9. Later stages (outline only; built and approved separately)

1. **Screen:** 32 colour-reversed pairs, the student in the proven combination against the reference. STOP if the mean
   pair score is below 0.50.
2. **Strength test:** **110 fresh pairs**, with a one-sided Hoeffding bound at α = 0.05. PASS if
   mean − √(ln 20 / 220) > 0.5, which means a mean above **0.61669**.
   - Power at a true 0.65 is about 76–84%. That is a normal-approximation planning estimate, for pair-score standard
     deviations of 0.50–0.35.
   - The same estimate gives about 33% power at a true 0.60.
3. **Common to both:**
   - **Measured** compute guards stay: evaluations ≤ 1.05× and search seconds ≤ 1.10×, candidate over reference.
   - TF32=0, fresh 2027-band blocks, and the idle and BOINC gates.
   - Screen games are never pooled with strength games.

## 10. Steps; each needs the user's approval

1. This card and the build. *(This approval.)*
2. Re-acquire the pinned twixtbot-ui checkout and a CPU TensorFlow runtime. A separate approval; not covered here.
3. A **bounded timing step**: a fixed small number of teacher evaluations and CPU fit updates, on rows from the
   **train** arm only, with its own ceiling. It sets the label and fit ceilings at 1.5× the projection.
4. `run label`, once: gate `HTB_LABEL_AUTHORIZED`, opened in a clean worktree and restored after.
5. `run fit`, once: gate `HTB_FIT_AUTHORIZED`, which computes F1–F3 and the verdict.
6. Review. The §9 stages are not authorized by any of the steps above.

## 11. Cost (provisional; not funding)

| Step | Estimate |
|---|---|
| Labelling | 528,000 `eval_one` calls (264,000 rows × 2 orientations) plus lockstep replays |
| Fit | 15,000 CPU updates over 64 tensors each |
| Validation scoring | 96,000 CPU forward passes (24,000 rows × 2 orientations × 2 networks) |
| §9 stages | about 0.6 h and 2.2 h at entry 51's pace, once their own timing is measured |

None of the labelling or fit times are measured yet. Step 3 replaces these estimates.

## 12. Prediction (for the record)

Directional only, with low-to-moderate confidence: F1–F3 pass. No probability is offered. No strength prediction is
made by this card.

## 13. Review clarifications and build repairs (2026-11-03, the user's review; recipe unchanged)

1. **The training loss is a surrogate for the scored objective.** They are mathematically different.
   - **Training** minimizes, per row, the average over the two orientations of the cross-entropy between the
     (symmetric) target and each **raw** orientation's prediction, plus the raw values' squared errors.
   - **Scoring** measures KL and MSE of the **mirror average** Q_MA.
   - By convexity of −log, the mean of the two raw cross-entropies bounds the cross-entropy of their average from
     above. So driving the surrogate down drives Q_MA's KL down too. The converse is not implied: the two optima need
     not coincide, and F1–F3 judge only the mirror-averaged quantities.
2. **Isolation of the test arm, stated exactly.** Corpus authentication reads, hashes and **JSON-parses every file,
   test arm included**, because C0's eligibility predicate needs each game's metadata and the digest needs its bytes.
   Test-arm move lists are parsed with the rest but are never returned, so **no test-arm position is replayed, keyed,
   selected, labelled or scored**. This replaces §4.2's "only to authenticate" wording.
3. **Fit bound to the approved labels.** Before any corpus replay, checkpoint load or update, `run fit` refuses
   unless all of the following hold:
   - the label evidence (`record.json`, `selection.json`) is committed and unchanged;
   - its status is COMPLETED, and its card hash and teacher pins equal this card's;
   - its recorded selection hash equals the committed selection file's;
   - the selection is this card's and holds exactly 240,000 train and 24,000 val rows, with no repeated key;
   - each arm's shards match the record's manifest hash, and their (game_id, ply, key) rows equal the selection's,
     in order.
4. **One run, refused before expensive work; always a terminal record.** Both entries refuse, writing nothing, if
   their evidence directory or any output destination already exists: the label shard directories for labelling;
   the fit evidence and student file for fitting. After the refusal checks, each entry creates its evidence
   directory with `started.json` and runs inside a boundary. Every exit writes a create-only `record.json` with
   status COMPLETED, STOP, FAILED (with the traceback) or INTERRUPTED (SIGTERM or SIGINT), and the matching exit code
   (0, 3, 4, 5). Refusals exit 2.

## 14. Amendment: red-first games only (2026-11-04, the user's review after T-label's STOP)

**Why.** T-label attempt 1 STOPPED at selection: "ply 0: recorded player black but red to move" (ledger #172).
Self-play picks the first mover at random and records `starting_player`; about half the corpus starts with black.
§4 replayed every game red-first, and the reviewed teacher adapter (`twixtbot_adapter`) maps red to twixtbot's WHITE,
which always moves first. This is a replay defect, not evidence about distillation.

**The rule (the user's choice: option 2).** This pilot uses **red-first games only**. They still contain positions with
both colours to move, and they match the planned red-first evaluation protocol. The teacher adapter is unchanged and
no colour-swap mapping is added. With red-first games only, §4.4's colour-canonicalization argument holds as written:
red-to-move states have equal peg counts and black-to-move states one more red peg.

1. **Authenticate first, filter after.** §4.1's verification runs over the complete original corpus, unchanged
   (eligible digest `dc7713c6…`, every file read and hashed). The restriction is applied only to the authenticated
   C0-eligible games.
2. **The starting player is checked, never defaulted.** For each eligible train or val game, the recorded
   `starting_player` must be `red` or `black`, and the moves' recorded `player` sequence must alternate from it,
   starting at ply 0. A missing or invalid `starting_player`, or a player sequence that disagrees with it, STOPs the
   run (step `corpus`). Black-first games are then excluded, and their number and positions are recorded.
3. **Unchanged:** the arms (§4.2), positions (§4.3), canonical key (§4.4), pools and leakage exclusion (§4.5–4.6),
   **N_TRAIN = 240,000 and N_VAL = 24,000** with the same ordering (§4.7), the short-pool **REFUSAL** with no fallback,
   the selection record (§4.8), targets (§5), the fit recipe (§6), scoring (§7) and the verdict (§8). A black-first
   game's positions are not in the training pool, so excluding them adds no leakage path.
4. **Applied consistently.** Every consumer of the corpus (selection, labelling, example construction and both
   timings) takes games from the one filtered pool and refuses (STOP) any game that is not a checked red-first game.
   Projections use the filtered pool's counts.
5. **No separate full selection scan now.** Whether the red-first pool holds 240,000 and 24,000 distinct positions is
   settled by `run label`'s own short-pool refusal.

## 15. Amendment: examples built per batch, not held (2026-11-04, the user's review after the timings)

**Why.** T-fit measured 246,540 B per built example (both orientations), so holding all 264,000 examples needs about
65.09 GB, and 66.37 GB with labels and weights. Both exceed this machine's 51.5 GB of physical memory (ledger #174).
Boards are 36.5 GB of the examples; the other 28.6 GB is non-board example data. So the change covers whole examples.

**The storage rule.**
1. **Kept for the whole fit:** compact labels (each arm's targets as a float32 array on the 576-cell index, values as
   float64, and each row's game_id, ply and key) and the move lists of the games the rows come from.
2. **Built only when used:** complete examples, both orientations, exactly as §5 and `build_examples` define them.
   They are built for the current training batch (32 rows) or scoring chunk (64 rows) and then released.
   Metadata (keys for the epoch order, values, counts) is read from the compact rows and never builds an example.
3. **Unchanged:** the selected rows and their order, the sha256 epoch ordering, the targets (the float32 label
   renormalized over the legal moves, as before), both orientations, the fit recipe (§6, Adam 1e-4, clip 1.0, value
   weight 1.0, L2 1e-4, 32 rows × 2 orientations, 15,000 updates), the start check, scoring (§7, with the same
   64-row chunks) and the verdict (§8). Built per batch, an example is identical to the one built up front, and so is
   every update and score.
4. **Memory.** A conservative budget of retained data, temporary examples, optimizer state, gradients and working
   buffers is recorded in the ledger. The fit still refuses until `memory_required_bytes` is set by a reviewed edit,
   and refuses at launch if less than that is available.

## 16. Amendment: labels compacted as they are produced; memory checks at both launches (2026-11-05, the user's review)

**Why.** The label path held every labelled row of an arm (its legal moves as Python lists, about 44.8 KB each)
until that arm's shards were written. Its conservative budget, 23.56 GB, left about 0.37 GB against the lowest
available memory observed (23.93 GB).

**The rule.**
1. **Compact as produced.** Each arm's labels are held in preallocated arrays, in selection order, from the moment
   each game is labelled: the target as float32 on the 576-cell index, and value, value_id, value_ref (float64) and
   ply (int32). The row metadata (game_id, ply, key) comes from the selection. No row's legal-move list is kept after
   its game is labelled. Every selected row must be filled exactly once; otherwise the run STOPs.
2. **Unchanged:** the target and value definitions (§5), the float32 rounding of the stored target, the selection and
   its order, the shard format and size (8,192 rows), each shard's row list, the manifest, the selection hash, and
   every binding check in §13.3. The shards, row lists and manifest are byte-identical to the previous path's.
3. **Memory checks at both launches.** `run label` refuses unless its requirement is set and at least that much is
   available, before the teacher or the corpus is loaded. `run fit` keeps its check, with the requirement the user
   approved: **12 GiB (12,884,901,888 B)**. Both budgets are estimates; peak memory has not been measured.
4. **Ceilings adopted by the user** for the full-run wrappers: label TERM 13,190 / KILL 13,200 / cleanup through
   13,210 s; fit TERM 18,590 / KILL 18,600 / cleanup through 18,610 s. The fit projection (about 3.42 h) and its
   2.441 ms rebuilt-row figure are planning estimates, not demonstrated bounds.
